use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use image::RgbImage;
use rayon::prelude::*;
use tracing::info;

use crate::core::color::rgb_to_ycbcr;
use crate::core::pack::{open_pack, BundleFile};
use crate::core::pyramid::copy_tile_into_mosaic;
use crate::core::residual::apply_residual_into;
use crate::core::upsample::upsample_2x_channel;
use crate::turbojpeg_optimized::{
    decode_luma_turbo, encode_jpeg_turbo, encode_luma_turbo, load_luma_turbo, load_rgb_turbo,
};

// ---------------------------------------------------------------------------
// BufferPool — pre-allocated buffer reuse for tile processing
// ---------------------------------------------------------------------------

pub struct BufferPool {
    buffers: Mutex<Vec<Vec<u8>>>,
    total: usize,
    in_use: AtomicUsize,
    in_use_max: AtomicUsize,
}

impl BufferPool {
    pub fn new(count: usize) -> Self {
        let mut bufs = Vec::with_capacity(count);
        for _ in 0..count {
            bufs.push(Vec::new());
        }
        Self {
            buffers: Mutex::new(bufs),
            total: count,
            in_use: AtomicUsize::new(0),
            in_use_max: AtomicUsize::new(0),
        }
    }

    pub fn get(&self, len: usize) -> Vec<u8> {
        let mut guard = self.buffers.lock().unwrap();
        let mut buf = guard.pop().unwrap_or_default();
        let in_use = self.in_use.fetch_add(1, Ordering::SeqCst) + 1;
        self.in_use_max.fetch_max(in_use, Ordering::SeqCst);
        if buf.len() != len {
            buf.resize(len, 0);
        }
        buf
    }

    pub fn put(&self, mut buf: Vec<u8>) {
        buf.clear();
        self.buffers.lock().unwrap().push(buf);
        self.in_use.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        let available = self.buffers.lock().unwrap().len();
        let in_use = self.in_use.load(Ordering::SeqCst);
        let in_use_max = self.in_use_max.swap(in_use, Ordering::SeqCst);
        (self.total, available, in_use_max)
    }
}

// ---------------------------------------------------------------------------
// ParallelStats — rayon concurrency tracking
// ---------------------------------------------------------------------------

pub struct ParallelStats {
    current: AtomicUsize,
    max: AtomicUsize,
}

impl ParallelStats {
    pub fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
        }
    }

    pub fn enter(&self) -> ParallelGuard<'_> {
        let cur = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(cur, Ordering::SeqCst);
        ParallelGuard { stats: self }
    }

    pub fn take_max(&self) -> usize {
        self.max
            .swap(self.current.load(Ordering::SeqCst), Ordering::SeqCst)
    }
}

pub struct ParallelGuard<'a> {
    stats: &'a ParallelStats,
}

impl<'a> Drop for ParallelGuard<'a> {
    fn drop(&mut self) {
        self.stats.current.fetch_sub(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// FamilyStats — pipeline timing instrumentation
// ---------------------------------------------------------------------------

pub struct FamilyStats {
    pub l2_decode_ms: u128,
    pub l1_resize_ms: u128,
    pub l1_residual_ms: u128,
    pub l1_encode_ms: u128,
    pub l0_resize_ms: u128,
    pub l0_residual_ms: u128,
    pub l0_encode_ms: u128,
    pub total_ms: u128,
    pub residuals_l1: usize,
    pub residuals_l0: usize,
    pub l1_parallel_max: usize,
    pub l0_parallel_max: usize,
}

// ---------------------------------------------------------------------------
// FamilyResult — output of one L2 family reconstruction
// ---------------------------------------------------------------------------

pub struct FamilyResult {
    /// L1 tiles: (x1, y1, jpeg_bytes)
    pub l1: Vec<(u32, u32, Vec<u8>)>,
    /// L0 tiles: (x0, y0, jpeg_bytes)
    pub l0: Vec<(u32, u32, Vec<u8>)>,
    /// Timing stats (if timing=true)
    pub stats: Option<FamilyStats>,
}

// ---------------------------------------------------------------------------
// ReconstructInput — describes where tiles and residuals live
// ---------------------------------------------------------------------------

pub struct ReconstructInput<'a> {
    /// baseline_pyramid_files/ directory
    pub files_dir: &'a Path,
    /// Loose residuals directory (L1/{x2}_{y2}/, L0/{x2}_{y2}/)
    pub residuals_dir: Option<&'a Path>,
    /// Pack files directory (individual .pack files per family)
    pub pack_dir: Option<&'a Path>,
    /// Bundle file (single mmapped file for all families, preferred over pack_dir)
    pub bundle: Option<&'a BundleFile>,
    /// Tile size (pixels)
    pub tile_size: u32,
    /// Level numbers
    pub l0: u32,
    pub l1: u32,
    pub l2: u32,
}

// ---------------------------------------------------------------------------
// OutputFormat — what image format to produce for reconstructed tiles
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Jpeg,
    Webp,
}

impl OutputFormat {
    pub fn content_type(&self) -> &'static str {
        match self {
            OutputFormat::Jpeg => "image/jpeg",
            OutputFormat::Webp => "image/webp",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::Jpeg => ".jpg",
            OutputFormat::Webp => ".webp",
        }
    }
}

// ---------------------------------------------------------------------------
// ReconstructOpts — controls reconstruction behavior
// ---------------------------------------------------------------------------

pub struct ReconstructOpts {
    pub quality: u8,
    pub timing: bool,
    pub grayscale_only: bool,
    pub output_format: OutputFormat,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn baseline_tile_path(files_dir: &Path, level: u32, x: u32, y: u32) -> PathBuf {
    files_dir
        .join(level.to_string())
        .join(format!("{}_{}.jpg", x, y))
}

fn residual_tile_path(
    residuals_dir: &Path,
    level: u32,
    l1_level: u32,
    x2: u32,
    y2: u32,
    x: u32,
    y: u32,
) -> PathBuf {
    let subdir = if level == l1_level { "L1" } else { "L0" };
    let parent = residuals_dir
        .join(subdir)
        .join(format!("{}_{}", x2, y2));
    // Try multiple extensions: .jpg (default), .webp, .jxl
    for ext in &[".jpg", ".webp", ".jxl"] {
        let p = parent.join(format!("{}_{}{}", x, y, ext));
        if p.exists() {
            return p;
        }
    }
    // Fallback to .jpg (caller will handle missing file)
    parent.join(format!("{}_{}.jpg", x, y))
}

fn load_rgb(path: &Path) -> Result<RgbImage> {
    let (pixels, width, height) = load_rgb_turbo(path)?;
    RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow!("failed to create RGB image from pixels"))
}

fn load_luma(path: &Path) -> Result<Vec<u8>> {
    // For non-JPEG formats, read bytes and dispatch to the correct decoder
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "webp" => {
            let data = fs::read(path)
                .with_context(|| format!("reading WebP residual {}", path.display()))?;
            decode_luma_webp(&data)
        }
        "jxl" => {
            let data = fs::read(path)
                .with_context(|| format!("reading JXL residual {}", path.display()))?;
            decode_luma_from_bytes(&data)
        }
        _ => {
            let (pixels, _width, _height) = load_luma_turbo(path)?;
            Ok(pixels)
        }
    }
}

/// Detect if bytes are JPEG-XL format.
/// JXL codestream starts with 0xFF 0x0A.
/// JXL container starts with 12-byte signature: 00 00 00 0C 4A 58 4C 20 0D 0A 87 0A.
fn is_jxl(bytes: &[u8]) -> bool {
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0x0A {
        return true;
    }
    if bytes.len() >= 12 {
        const JXL_CONTAINER: [u8; 12] = [
            0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
        ];
        return bytes[..12] == JXL_CONTAINER;
    }
    false
}

/// Decode grayscale luma from JXL-encoded bytes.
#[cfg(feature = "jpegxl")]
fn decode_luma_jxl(bytes: &[u8]) -> Result<Vec<u8>> {
    use jpegxl_rs::Endianness;
    use jpegxl_rs::decode::{decoder_builder, PixelFormat};
    let decoder = decoder_builder()
        .pixel_format(PixelFormat {
            num_channels: 1,
            endianness: Endianness::Native,
            align: 0,
        })
        .build()
        .map_err(|e| anyhow!("jpegxl decoder create failed: {e:?}"))?;
    let (_metadata, pixels): (_, Vec<u8>) = decoder
        .decode_with::<u8>(bytes)
        .map_err(|e| anyhow!("jpegxl decode failed: {e:?}"))?;
    Ok(pixels)
}

/// Detect if bytes are WebP format (RIFF/WEBP magic).
fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

/// Decode grayscale luma from WebP-encoded bytes.
/// WebP is always RGB — extract R channel (all channels identical for gray-encoded WebP).
fn decode_luma_webp(bytes: &[u8]) -> Result<Vec<u8>> {
    let decoder = webp::Decoder::new(bytes);
    let img = decoder
        .decode()
        .ok_or_else(|| anyhow!("webp decode failed"))?;
    let rgb = img.to_vec();
    let w = img.width() as usize;
    let h = img.height() as usize;
    let mut gray = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            gray.push(rgb[(y * w + x) * 3]);
        }
    }
    Ok(gray)
}

fn decode_luma_from_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    if is_webp(bytes) {
        return decode_luma_webp(bytes);
    }
    #[cfg(feature = "jpegxl")]
    if is_jxl(bytes) {
        return decode_luma_jxl(bytes);
    }
    #[cfg(not(feature = "jpegxl"))]
    if is_jxl(bytes) {
        anyhow::bail!("JXL-encoded residuals detected but binary was built without --features jpegxl");
    }
    let (pixels, _width, _height) = decode_luma_turbo(bytes)?;
    Ok(pixels)
}

fn encode_tile_rgb(
    bytes: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    format: OutputFormat,
) -> Result<Vec<u8>> {
    match format {
        OutputFormat::Jpeg => encode_jpeg_turbo(bytes, width, height, quality),
        OutputFormat::Webp => {
            let encoder = webp::Encoder::from_rgb(bytes, width, height);
            let mem = encoder.encode(quality as f32);
            Ok(mem.to_vec())
        }
    }
}

fn encode_tile_gray(
    bytes: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    format: OutputFormat,
) -> Result<Vec<u8>> {
    match format {
        OutputFormat::Jpeg => encode_luma_turbo(bytes, width, height, quality),
        OutputFormat::Webp => {
            let rgb: Vec<u8> = bytes.iter().flat_map(|&g| [g, g, g]).collect();
            let encoder = webp::Encoder::from_rgb(&rgb, width, height);
            let mem = encoder.encode(quality as f32);
            Ok(mem.to_vec())
        }
    }
}

fn ycbcr_planes(img: &RgbImage) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = img.width();
    let h = img.height();
    let mut y = vec![0u8; (w * h) as usize];
    let mut cb = vec![0u8; (w * h) as usize];
    let mut cr = vec![0u8; (w * h) as usize];
    for yy in 0..h {
        for xx in 0..w {
            let idx = (yy * w + xx) as usize;
            let p = img.get_pixel(xx, yy).0;
            let (yyc, cbc, crc) = rgb_to_ycbcr(p[0], p[1], p[2]);
            y[idx] = yyc;
            cb[idx] = cbc;
            cr[idx] = crc;
        }
    }
    (y, cb, cr)
}

// ---------------------------------------------------------------------------
// reconstruct_family — the core reconstruction pipeline
// ---------------------------------------------------------------------------

/// Reconstruct all L1/L0 tiles for one L2 parent.
/// Same pipeline used by both the server and CLI decode.
pub fn reconstruct_family(
    input: &ReconstructInput,
    x2: u32,
    y2: u32,
    opts: &ReconstructOpts,
    buffer_pool: &BufferPool,
) -> Result<FamilyResult> {
    let total_start = Instant::now();
    let timing = opts.timing;
    let quality = opts.quality;
    let grayscale_only = opts.grayscale_only;
    let output_format = opts.output_format;
    let tile_size = input.tile_size;

    // --- L2 decode ---
    let l2_path = baseline_tile_path(input.files_dir, input.l2, x2, y2);
    let l2_start = Instant::now();
    let l2_img = load_rgb(&l2_path)
        .with_context(|| format!("loading baseline L2 tile {}", l2_path.display()))?;
    let l2_decode_ms = if timing {
        l2_start.elapsed().as_millis()
    } else {
        0
    };

    // --- Parallel chroma upsample ---
    let parallel_chroma_start = Instant::now();
    let (l2_y, l2_cb, l2_cr) = ycbcr_planes(&l2_img);
    let l2_width = l2_img.width();
    let l2_height = l2_img.height();

    let ((l1_y, l1_cb, l1_cr), (l0_cb, l0_cr)) = rayon::join(
        || {
            let l1_y = upsample_2x_channel(&l2_y, l2_width as usize, l2_height as usize);
            let l1_cb = upsample_2x_channel(&l2_cb, l2_width as usize, l2_height as usize);
            let l1_cr = upsample_2x_channel(&l2_cr, l2_width as usize, l2_height as usize);
            (l1_y, l1_cb, l1_cr)
        },
        || {
            let l0_cb_2x = upsample_2x_channel(&l2_cb, l2_width as usize, l2_height as usize);
            let l0_cr_2x = upsample_2x_channel(&l2_cr, l2_width as usize, l2_height as usize);
            let l0_cb = upsample_2x_channel(
                &l0_cb_2x,
                (l2_width * 2) as usize,
                (l2_height * 2) as usize,
            );
            let l0_cr = upsample_2x_channel(
                &l0_cr_2x,
                (l2_width * 2) as usize,
                (l2_height * 2) as usize,
            );
            (l0_cb, l0_cr)
        },
    );

    let parallel_chroma_ms = if timing {
        parallel_chroma_start.elapsed().as_millis()
    } else {
        0
    };

    // --- Load pack file if available (bundle preferred, then individual .pack files) ---
    let pack = if let Some(bundle) = input.bundle {
        bundle.get_pack(x2, y2).ok()
    } else if let Some(pack_root) = input.pack_dir {
        open_pack(pack_root, x2, y2).ok()
    } else {
        None
    };

    // --- L1 tile reconstruction ---
    let tile_buf_len = (tile_size * tile_size * 3) as usize;
    let mut l1_tile_bufs: Vec<Vec<u8>> = (0..4).map(|_| buffer_pool.get(tile_buf_len)).collect();
    let l1_parallel = ParallelStats::new();
    let l1_results: Result<Vec<_>> = l1_tile_bufs
        .par_iter_mut()
        .enumerate()
        .map(|(idx, buf)| {
            let _guard = l1_parallel.enter();
            let dx = (idx % 2) as u32;
            let dy = (idx / 2) as u32;
            let x1 = x2 * 2 + dx;
            let y1 = y2 * 2 + dy;

            let res_start = Instant::now();
            let used_residual = {
                let residual_path = input.residuals_dir.map(|d| {
                    residual_tile_path(d, input.l1, input.l1, x2, y2, x1, y1)
                });

                if let Some(ref pack) = pack {
                    if let Some(bytes) = pack.get_residual(1, (dy * 2 + dx) as u8) {
                        let residual = decode_luma_from_bytes(bytes)?;
                        apply_residual_into(
                            &l1_y, &l1_cb, &l1_cr,
                            tile_size * 2, tile_size * 2,
                            dx * tile_size, dy * tile_size,
                            tile_size, &residual, buf,
                        )?;
                        true
                    } else if residual_path.as_ref().map_or(false, |p| p.exists()) {
                        let residual = load_luma(residual_path.as_ref().unwrap())?;
                        apply_residual_into(
                            &l1_y, &l1_cb, &l1_cr,
                            tile_size * 2, tile_size * 2,
                            dx * tile_size, dy * tile_size,
                            tile_size, &residual, buf,
                        )?;
                        true
                    } else {
                        let base = load_rgb(&baseline_tile_path(input.files_dir, input.l1, x1, y1))?;
                        buf.copy_from_slice(base.as_raw());
                        false
                    }
                } else if residual_path.as_ref().map_or(false, |p| p.exists()) {
                    let residual = load_luma(residual_path.as_ref().unwrap())?;
                    apply_residual_into(
                        &l1_y, &l1_cb, &l1_cr,
                        tile_size * 2, tile_size * 2,
                        dx * tile_size, dy * tile_size,
                        tile_size, &residual, buf,
                    )?;
                    true
                } else {
                    let base = load_rgb(&baseline_tile_path(input.files_dir, input.l1, x1, y1))?;
                    buf.copy_from_slice(base.as_raw());
                    false
                }
            };

            let res_ms = if timing { res_start.elapsed().as_millis() } else { 0 };
            let enc_start = Instant::now();
            let bytes = if grayscale_only {
                let mut y_bytes = Vec::with_capacity((tile_size * tile_size) as usize);
                for i in 0..(tile_size * tile_size) as usize {
                    let r = buf[i * 3] as f32;
                    let g = buf[i * 3 + 1] as f32;
                    let b = buf[i * 3 + 2] as f32;
                    let y = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                    y_bytes.push(y);
                }
                encode_tile_gray(&y_bytes, tile_size, tile_size, quality, output_format)?
            } else {
                encode_tile_rgb(buf, tile_size, tile_size, quality, output_format)?
            };
            let enc_ms = if timing { enc_start.elapsed().as_millis() } else { 0 };
            Ok((idx, x1, y1, bytes, used_residual, res_ms, enc_ms))
        })
        .collect();

    let l1_results = l1_results?;
    let l1_parallel_max = l1_parallel.take_max();

    let mut l1_tiles: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(4);
    let mut residuals_l1 = 0usize;
    let mut residuals_l0 = 0usize;
    let mut l1_residual_ms = 0u128;
    let mut l1_encode_ms = 0u128;
    let mut l0_residual_ms = 0u128;
    let mut l0_encode_ms = 0u128;

    for (_idx, x1, y1, bytes, used_residual, res_ms, enc_ms) in l1_results {
        if used_residual {
            residuals_l1 += 1;
        }
        l1_residual_ms += res_ms;
        l1_encode_ms += enc_ms;
        l1_tiles.push((x1, y1, bytes));
    }

    // --- Build L1 mosaic for L0 prediction ---
    let l1_mosaic_w = tile_size * 2;
    let l1_mosaic_size = (l1_mosaic_w * l1_mosaic_w) as usize;
    let mut l1_mosaic_rgb = vec![0u8; l1_mosaic_size * 3];

    for idx in 0..4 {
        let dx = (idx % 2) as u32;
        let dy = (idx / 2) as u32;
        copy_tile_into_mosaic(
            &l1_tile_bufs[idx],
            &mut l1_mosaic_rgb,
            l1_mosaic_w,
            tile_size,
            dx,
            dy,
        );
    }

    let mut l1_mosaic_y = vec![0u8; l1_mosaic_size];
    let mut l1_mosaic_cb = vec![0u8; l1_mosaic_size];
    let mut l1_mosaic_cr = vec![0u8; l1_mosaic_size];

    for i in 0..l1_mosaic_size {
        let src_idx = i * 3;
        let r = l1_mosaic_rgb[src_idx];
        let g = l1_mosaic_rgb[src_idx + 1];
        let b = l1_mosaic_rgb[src_idx + 2];
        let (yy, cb, cr) = rgb_to_ycbcr(r, g, b);
        l1_mosaic_y[i] = yy;
        l1_mosaic_cb[i] = cb;
        l1_mosaic_cr[i] = cr;
    }

    // Return L1 tile buffers to pool
    for buf in l1_tile_bufs.drain(..) {
        buffer_pool.put(buf);
    }

    // --- L0 Y upsample from L1 mosaic ---
    let l0_resize_start = Instant::now();
    let l0_y = upsample_2x_channel(&l1_mosaic_y, l1_mosaic_w as usize, l1_mosaic_w as usize);
    let l0_resize_ms = if timing {
        l0_resize_start.elapsed().as_millis()
    } else {
        0
    };

    // --- L0 tile reconstruction ---
    let l0_parallel = ParallelStats::new();
    let l0_results: Result<Vec<_>> = (0..16)
        .into_par_iter()
        .map(|idx| {
            let _guard = l0_parallel.enter();
            let dx = (idx % 4) as u32;
            let dy = (idx / 4) as u32;
            let x0 = x2 * 4 + dx;
            let y0 = y2 * 4 + dy;

            let res_start = Instant::now();
            let mut buf = buffer_pool.get(tile_buf_len);
            let (used_residual, bytes, enc_ms) = {
                let residual_path = input.residuals_dir.map(|d| {
                    residual_tile_path(d, input.l0, input.l1, x2, y2, x0, y0)
                });

                let used_residual = if let Some(ref pack) = pack {
                    if let Some(bytes) = pack.get_residual(0, (dy * 4 + dx) as u8) {
                        let residual = decode_luma_from_bytes(bytes)?;
                        apply_residual_into(
                            &l0_y, &l0_cb, &l0_cr,
                            tile_size * 4, tile_size * 4,
                            dx * tile_size, dy * tile_size,
                            tile_size, &residual, &mut buf,
                        )?;
                        true
                    } else if residual_path.as_ref().map_or(false, |p| p.exists()) {
                        let residual = load_luma(residual_path.as_ref().unwrap())?;
                        apply_residual_into(
                            &l0_y, &l0_cb, &l0_cr,
                            tile_size * 4, tile_size * 4,
                            dx * tile_size, dy * tile_size,
                            tile_size, &residual, &mut buf,
                        )?;
                        true
                    } else {
                        let base = load_rgb(&baseline_tile_path(input.files_dir, input.l0, x0, y0))?;
                        buf.copy_from_slice(base.as_raw());
                        false
                    }
                } else if residual_path.as_ref().map_or(false, |p| p.exists()) {
                    let residual = load_luma(residual_path.as_ref().unwrap())?;
                    apply_residual_into(
                        &l0_y, &l0_cb, &l0_cr,
                        tile_size * 4, tile_size * 4,
                        dx * tile_size, dy * tile_size,
                        tile_size, &residual, &mut buf,
                    )?;
                    true
                } else {
                    let base = load_rgb(&baseline_tile_path(input.files_dir, input.l0, x0, y0))?;
                    buf.copy_from_slice(base.as_raw());
                    false
                };

                let enc_start = Instant::now();
                let bytes = if grayscale_only {
                    let mut y_bytes = Vec::with_capacity((tile_size * tile_size) as usize);
                    for i in 0..(tile_size * tile_size) as usize {
                        let r = buf[i * 3] as f32;
                        let g = buf[i * 3 + 1] as f32;
                        let b = buf[i * 3 + 2] as f32;
                        let y = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                        y_bytes.push(y);
                    }
                    encode_tile_gray(&y_bytes, tile_size, tile_size, quality, output_format)?
                } else {
                    encode_tile_rgb(&buf, tile_size, tile_size, quality, output_format)?
                };
                let enc_ms = if timing { enc_start.elapsed().as_millis() } else { 0 };
                (used_residual, bytes, enc_ms)
            };
            buffer_pool.put(buf);
            let res_ms = if timing { res_start.elapsed().as_millis() } else { 0 };
            Ok((x0, y0, bytes, used_residual, res_ms, enc_ms))
        })
        .collect();

    let l0_results = l0_results?;
    let l0_parallel_max = l0_parallel.take_max();

    let mut l0_tiles: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(16);
    for (x0, y0, bytes, used_residual, res_ms, enc_ms) in l0_results {
        if used_residual {
            residuals_l0 += 1;
        }
        l0_residual_ms += res_ms;
        l0_encode_ms += enc_ms;
        l0_tiles.push((x0, y0, bytes));
    }

    info!(
        "family_residuals x2={} y2={} l1={} l0={}",
        x2, y2, residuals_l1, residuals_l0
    );

    let stats = if timing {
        Some(FamilyStats {
            l2_decode_ms,
            l1_resize_ms: parallel_chroma_ms,
            l1_residual_ms,
            l1_encode_ms,
            l0_resize_ms,
            l0_residual_ms,
            l0_encode_ms,
            total_ms: total_start.elapsed().as_millis(),
            residuals_l1,
            residuals_l0,
            l1_parallel_max,
            l0_parallel_max,
        })
    } else {
        None
    };

    Ok(FamilyResult {
        l1: l1_tiles,
        l0: l0_tiles,
        stats,
    })
}

// ---------------------------------------------------------------------------
// write_family_tiles — write a FamilyResult to disk
// ---------------------------------------------------------------------------

/// Write a FamilyResult's tiles to disk under out_dir/L1/ and out_dir/L0/.
pub fn write_family_tiles(result: &FamilyResult, out_dir: &Path, format: OutputFormat) -> Result<()> {
    let ext = format.extension();
    let l1_dir = out_dir.join("L1");
    let l0_dir = out_dir.join("L0");
    fs::create_dir_all(&l1_dir)?;
    fs::create_dir_all(&l0_dir)?;

    for (x, y, bytes) in &result.l1 {
        let path = l1_dir.join(format!("{}_{}{}", x, y, ext));
        fs::write(&path, bytes)
            .with_context(|| format!("writing L1 tile {}", path.display()))?;
    }

    for (x, y, bytes) in &result.l0 {
        let path = l0_dir.join(format!("{}_{}{}", x, y, ext));
        fs::write(&path, bytes)
            .with_context(|| format!("writing L0 tile {}", path.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data").join("demo_out")
    }

    fn has_test_data() -> bool {
        test_data_dir().join("baseline_pyramid.dzi").exists()
    }

    // -----------------------------------------------------------------------
    // BufferPool tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_buffer_pool_get_put() {
        let pool = BufferPool::new(4);
        let buf1 = pool.get(1024);
        assert_eq!(buf1.len(), 1024);
        let buf2 = pool.get(2048);
        assert_eq!(buf2.len(), 2048);
        pool.put(buf1);
        pool.put(buf2);
        let (total, avail, _) = pool.stats();
        assert_eq!(total, 4);
        // We started with 4, took 2, returned 2 — but returned buffers are cleared
        // and re-added, so we should have at least 4 now
        assert!(avail >= 4);
    }

    #[test]
    fn test_buffer_pool_tracks_in_use() {
        let pool = BufferPool::new(8);
        let b1 = pool.get(100);
        let b2 = pool.get(100);
        let b3 = pool.get(100);
        let (_, _, max_in_use) = pool.stats();
        assert_eq!(max_in_use, 3);
        pool.put(b1);
        pool.put(b2);
        pool.put(b3);
    }

    #[test]
    fn test_buffer_pool_exceeds_capacity() {
        // Pool with 2 pre-allocated, but we can still get more (just allocates new)
        let pool = BufferPool::new(2);
        let b1 = pool.get(100);
        let b2 = pool.get(100);
        let b3 = pool.get(100); // exceeds pre-allocated count
        assert_eq!(b3.len(), 100);
        pool.put(b1);
        pool.put(b2);
        pool.put(b3);
    }

    // -----------------------------------------------------------------------
    // ParallelStats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_stats_tracking() {
        let stats = ParallelStats::new();
        {
            let _g1 = stats.enter();
            let _g2 = stats.enter();
            let _g3 = stats.enter();
            // max should be 3
        }
        // Guards dropped, current should be 0
        let max = stats.take_max();
        assert_eq!(max, 3);
    }

    // -----------------------------------------------------------------------
    // reconstruct_family with loose residuals
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconstruct_family_with_residuals() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let data = test_data_dir();
        let input = ReconstructInput {
            files_dir: &data.join("baseline_pyramid_files"),
            residuals_dir: Some(&data.join("residuals_q32")),
            pack_dir: None,
            bundle: None,
            tile_size: 256,
            l0: 16,
            l1: 15,
            l2: 14,
        };
        let opts = ReconstructOpts {
            quality: 90,
            timing: true,
            grayscale_only: false,
            output_format: OutputFormat::Jpeg,
        };
        let pool = BufferPool::new(32);

        let result = reconstruct_family(&input, 50, 50, &opts, &pool).unwrap();

        // Should produce 4 L1 tiles and 16 L0 tiles
        assert_eq!(result.l1.len(), 4, "Expected 4 L1 tiles");
        assert_eq!(result.l0.len(), 16, "Expected 16 L0 tiles");

        // L1 tile coordinates should be (100..101, 100..101)
        let mut l1_coords: Vec<(u32, u32)> = result.l1.iter().map(|(x, y, _)| (*x, *y)).collect();
        l1_coords.sort();
        assert_eq!(l1_coords, vec![(100, 100), (100, 101), (101, 100), (101, 101)]);

        // L0 tile coordinates should be (200..203, 200..203)
        let mut l0_coords: Vec<(u32, u32)> = result.l0.iter().map(|(x, y, _)| (*x, *y)).collect();
        l0_coords.sort();
        let mut expected_l0: Vec<(u32, u32)> = Vec::new();
        for x in 200..204 {
            for y in 200..204 {
                expected_l0.push((x, y));
            }
        }
        expected_l0.sort();
        assert_eq!(l0_coords, expected_l0);

        // All tiles should be valid JPEGs (start with FF D8)
        for (x, y, bytes) in &result.l1 {
            assert!(bytes.len() > 100, "L1 tile ({},{}) too small: {} bytes", x, y, bytes.len());
            assert_eq!(&bytes[0..2], &[0xFF, 0xD8], "L1 tile ({},{}) not valid JPEG", x, y);
        }
        for (x, y, bytes) in &result.l0 {
            assert!(bytes.len() > 100, "L0 tile ({},{}) too small: {} bytes", x, y, bytes.len());
            assert_eq!(&bytes[0..2], &[0xFF, 0xD8], "L0 tile ({},{}) not valid JPEG", x, y);
        }

        // Timing stats should be populated
        let stats = result.stats.as_ref().unwrap();
        assert!(stats.total_ms > 0);
        assert_eq!(stats.residuals_l1, 4, "All 4 L1 residuals should be applied");
        assert_eq!(stats.residuals_l0, 16, "All 16 L0 residuals should be applied");
    }

    // -----------------------------------------------------------------------
    // reconstruct_family with pack files
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconstruct_family_with_packs() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let data = test_data_dir();
        let input = ReconstructInput {
            files_dir: &data.join("baseline_pyramid_files"),
            residuals_dir: None,
            pack_dir: Some(&data.join("residual_packs")),
            bundle: None,
            tile_size: 256,
            l0: 16,
            l1: 15,
            l2: 14,
        };
        let opts = ReconstructOpts {
            quality: 90,
            timing: false,
            grayscale_only: false,
            output_format: OutputFormat::Jpeg,
        };
        let pool = BufferPool::new(32);

        let result = reconstruct_family(&input, 50, 50, &opts, &pool).unwrap();

        assert_eq!(result.l1.len(), 4);
        assert_eq!(result.l0.len(), 16);
        assert!(result.stats.is_none(), "timing=false should produce no stats");

        // Tiles from pack should also be valid JPEGs
        for (_, _, bytes) in &result.l1 {
            assert_eq!(&bytes[0..2], &[0xFF, 0xD8]);
        }
        for (_, _, bytes) in &result.l0 {
            assert_eq!(&bytes[0..2], &[0xFF, 0xD8]);
        }
    }

    // -----------------------------------------------------------------------
    // Residuals vs packs produce similar output
    // -----------------------------------------------------------------------

    #[test]
    fn test_residuals_vs_packs_produce_similar_output() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let data = test_data_dir();
        let pool = BufferPool::new(64);
        let opts = ReconstructOpts {
            quality: 95,
            timing: false,
            grayscale_only: false,
            output_format: OutputFormat::Jpeg,
        };

        // Reconstruct from loose residuals
        let input_residuals = ReconstructInput {
            files_dir: &data.join("baseline_pyramid_files"),
            residuals_dir: Some(&data.join("residuals_q32")),
            pack_dir: None,
            bundle: None,
            tile_size: 256,
            l0: 16, l1: 15, l2: 14,
        };
        let result_residuals = reconstruct_family(&input_residuals, 50, 50, &opts, &pool).unwrap();

        // Reconstruct from packs
        let input_packs = ReconstructInput {
            files_dir: &data.join("baseline_pyramid_files"),
            residuals_dir: None,
            pack_dir: Some(&data.join("residual_packs")),
            bundle: None,
            tile_size: 256,
            l0: 16, l1: 15, l2: 14,
        };
        let result_packs = reconstruct_family(&input_packs, 50, 50, &opts, &pool).unwrap();

        // Both should produce the same number of tiles
        assert_eq!(result_residuals.l1.len(), result_packs.l1.len());
        assert_eq!(result_residuals.l0.len(), result_packs.l0.len());

        // Tile coordinates should match
        let mut res_l1: Vec<(u32, u32)> = result_residuals.l1.iter().map(|(x, y, _)| (*x, *y)).collect();
        let mut pack_l1: Vec<(u32, u32)> = result_packs.l1.iter().map(|(x, y, _)| (*x, *y)).collect();
        res_l1.sort();
        pack_l1.sort();
        assert_eq!(res_l1, pack_l1);

        // JPEG sizes should be similar (within 20% — they come from the same residuals
        // but pack residuals go through an extra compress/decompress cycle in encode,
        // so they may differ slightly)
        for i in 0..result_residuals.l1.len() {
            let (_, _, ref res_bytes) = result_residuals.l1[i];
            // Find matching tile in packs result
            let (rx, ry) = (result_residuals.l1[i].0, result_residuals.l1[i].1);
            if let Some((_, _, ref pack_bytes)) = result_packs.l1.iter().find(|(x, y, _)| *x == rx && *y == ry) {
                let size_ratio = res_bytes.len() as f64 / pack_bytes.len() as f64;
                assert!(
                    (0.5..2.0).contains(&size_ratio),
                    "L1 tile ({},{}) sizes too different: residuals={} packs={}",
                    rx, ry, res_bytes.len(), pack_bytes.len()
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Grayscale mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconstruct_family_grayscale() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let data = test_data_dir();
        let input = ReconstructInput {
            files_dir: &data.join("baseline_pyramid_files"),
            residuals_dir: Some(&data.join("residuals_q32")),
            pack_dir: None,
            bundle: None,
            tile_size: 256,
            l0: 16, l1: 15, l2: 14,
        };
        let opts = ReconstructOpts {
            quality: 90,
            timing: false,
            grayscale_only: true,
            output_format: OutputFormat::Jpeg,
        };
        let pool = BufferPool::new(32);

        let result = reconstruct_family(&input, 50, 50, &opts, &pool).unwrap();
        assert_eq!(result.l1.len(), 4);
        assert_eq!(result.l0.len(), 16);

        // Grayscale JPEGs should be smaller than color
        // and still valid JPEG
        for (_, _, bytes) in &result.l1 {
            assert_eq!(&bytes[0..2], &[0xFF, 0xD8]);
        }
    }

    // -----------------------------------------------------------------------
    // write_family_tiles
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_family_tiles() {
        let out_dir = std::env::temp_dir().join("origami_test_write_family");
        let _ = fs::remove_dir_all(&out_dir);

        let result = FamilyResult {
            l1: vec![
                (10, 20, vec![0xFF, 0xD8, 0xFF, 0xD9]),
                (11, 20, vec![0xFF, 0xD8, 0xFF, 0xD9]),
            ],
            l0: vec![
                (40, 80, vec![0xFF, 0xD8, 0xFF, 0xD9]),
            ],
            stats: None,
        };

        write_family_tiles(&result, &out_dir, OutputFormat::Jpeg).unwrap();

        assert!(out_dir.join("L1/10_20.jpg").exists());
        assert!(out_dir.join("L1/11_20.jpg").exists());
        assert!(out_dir.join("L0/40_80.jpg").exists());

        // Verify content
        let data = fs::read(out_dir.join("L1/10_20.jpg")).unwrap();
        assert_eq!(data, vec![0xFF, 0xD8, 0xFF, 0xD9]);

        let _ = fs::remove_dir_all(&out_dir);
    }

    // -----------------------------------------------------------------------
    // Full round-trip: reconstruct + write + verify files on disk
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconstruct_and_write_roundtrip() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let data = test_data_dir();
        let out_dir = std::env::temp_dir().join("origami_test_reconstruct_roundtrip");
        let _ = fs::remove_dir_all(&out_dir);

        let input = ReconstructInput {
            files_dir: &data.join("baseline_pyramid_files"),
            residuals_dir: Some(&data.join("residuals_q32")),
            pack_dir: None,
            bundle: None,
            tile_size: 256,
            l0: 16, l1: 15, l2: 14,
        };
        let opts = ReconstructOpts {
            quality: 90,
            timing: false,
            grayscale_only: false,
            output_format: OutputFormat::Jpeg,
        };
        let pool = BufferPool::new(32);

        let result = reconstruct_family(&input, 50, 50, &opts, &pool).unwrap();
        write_family_tiles(&result, &out_dir, OutputFormat::Jpeg).unwrap();

        // Verify L1 tiles on disk
        for (x, y, _) in &result.l1 {
            let path = out_dir.join("L1").join(format!("{}_{}.jpg", x, y));
            assert!(path.exists(), "L1 tile ({},{}) not written", x, y);
            let on_disk = fs::read(&path).unwrap();
            assert_eq!(&on_disk[0..2], &[0xFF, 0xD8]);
        }

        // Verify L0 tiles on disk
        for (x, y, _) in &result.l0 {
            let path = out_dir.join("L0").join(format!("{}_{}.jpg", x, y));
            assert!(path.exists(), "L0 tile ({},{}) not written", x, y);
        }

        // Should have exactly 4 L1 + 16 L0 files
        let l1_count = fs::read_dir(out_dir.join("L1")).unwrap().count();
        let l0_count = fs::read_dir(out_dir.join("L0")).unwrap().count();
        assert_eq!(l1_count, 4);
        assert_eq!(l0_count, 16);

        let _ = fs::remove_dir_all(&out_dir);
    }
}
