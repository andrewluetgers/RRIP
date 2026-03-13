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
use crate::core::ResampleFilter;
use crate::turbojpeg_optimized::{
    decode_luma_turbo, encode_jpeg_turbo, encode_luma_turbo, load_rgb_turbo,
};

use super::fir_resize;

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
    pub upsample_ms: u128,
    pub l0_residual_ms: u128,
    pub l0_encode_ms: u128,
    pub l1_downsample_ms: u128,
    pub l1_encode_ms: u128,
    pub total_ms: u128,
    pub l0_parallel_max: usize,
}

// ---------------------------------------------------------------------------
// FamilyResult — output of one L2 family reconstruction
// ---------------------------------------------------------------------------

pub struct FamilyResult {
    /// L2 tile: (x2, y2, jpeg_bytes) — only present when seed > tile_size
    pub l2: Option<(u32, u32, Vec<u8>)>,
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
    /// baseline_pyramid_files/ directory (for filesystem fallback)
    pub files_dir: &'a Path,
    /// Pack files directory (individual .pack files per family)
    pub pack_dir: Option<&'a Path>,
    /// Bundle file (single mmapped file for all families, preferred over pack_dir)
    pub bundle: Option<&'a BundleFile>,
    /// Tile size (pixels)
    pub tile_size: u32,
    /// Level numbers (used by server for tile key mapping)
    #[allow(dead_code)]
    pub l0: u32,
    #[allow(dead_code)]
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
    /// Optional sharpen strength to apply to L2 before upsampling (decode-time sharpening).
    pub sharpen: Option<f32>,
    /// Resample filter for prediction upsamples (default: Lanczos3).
    pub upsample_filter: ResampleFilter,
    /// Optional SR model for learned 4x super-resolution (replaces upsample filter).
    #[cfg(feature = "sr-model")]
    pub sr_model: Option<std::sync::Arc<crate::core::sr_model::SRModel>>,
    /// Optional refine model for same-resolution enhancement of L0 tiles.
    #[cfg(feature = "sr-model")]
    pub refine_model: Option<std::sync::Arc<crate::core::sr_model::SRModel>>,
    /// Unsharp mask strength applied to decoded residual before applying to prediction.
    /// Boosts residual detail at decode time without increasing encoded size.
    pub l0_sharpen: Option<f32>,
    /// Unsharp mask strength applied to final reconstructed tiles (decode-time).
    /// Applied after residual correction, before noise synthesis.
    pub tile_sharpen: Option<f32>,
    /// Enable wavelet-domain noise synthesis at decode time.
    /// If true and the pack contains synthesis params (level_kind=6),
    /// synthesized noise is added to the reconstructed Y plane.
    pub synth_noise: bool,
    /// Noise synthesis strength (0.0 = none, 1.0 = full measured sigma).
    /// Default 0.5 recommended since denoiser removes some real signal.
    pub synth_strength: f32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Decode an L2 baseline tile from raw bytes (JPEG or JXL).
/// Detects format by magic bytes.
pub fn decode_l2_rgb_from_bytes(data: &[u8]) -> Result<RgbImage> {
    // Check for JXL magic
    let is_jxl = (data.len() >= 2 && data[0] == 0xFF && data[1] == 0x0A)
        || (data.len() >= 12
            && data[..12]
                == [0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A]);
    if is_jxl {
        return decode_l2_rgb_jxl_bytes(data);
    }
    // Default: JPEG
    let (pixels, width, height) = crate::turbojpeg_optimized::decode_rgb_turbo(data)?;
    RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow!("failed to create RGB image from L2 pack bytes"))
}

#[cfg(feature = "jpegxl")]
fn decode_l2_rgb_jxl_bytes(data: &[u8]) -> Result<RgbImage> {
    let decoder = jpegxl_rs::decode::decoder_builder()
        .build()
        .map_err(|e| anyhow!("JXL decoder build failed: {e:?}"))?;
    let (_meta, result) = decoder.reconstruct(data)
        .map_err(|e| anyhow!("JXL reconstruct failed: {e:?}"))?;
    match result {
        jpegxl_rs::decode::Data::Jpeg(jpeg_bytes) => {
            let (pixels, width, height) = crate::turbojpeg_optimized::decode_rgb_turbo(&jpeg_bytes)?;
            RgbImage::from_raw(width, height, pixels)
                .ok_or_else(|| anyhow!("failed to create RGB image from JXL L2 pack bytes"))
        }
        jpegxl_rs::decode::Data::Pixels(pixels) => {
            let raw = match pixels {
                jpegxl_rs::decode::Pixels::Uint8(v) => v,
                _ => anyhow::bail!("JXL L2 decoded to non-u8 pixel type"),
            };
            let w = _meta.width;
            let h = _meta.height;
            RgbImage::from_raw(w, h, raw)
                .ok_or_else(|| anyhow!("failed to create RGB image from JXL L2 pixels"))
        }
    }
}

#[cfg(not(feature = "jpegxl"))]
fn decode_l2_rgb_jxl_bytes(_data: &[u8]) -> Result<RgbImage> {
    anyhow::bail!("L2 in pack is JXL but binary was built without --features jpegxl");
}

fn baseline_tile_path(files_dir: &Path, level: u32, x: u32, y: u32) -> PathBuf {
    // Check for JXL first (L2 may be stored as JXL for space savings)
    let jxl = files_dir.join(level.to_string()).join(format!("{}_{}.jxl", x, y));
    if jxl.exists() {
        return jxl;
    }
    files_dir
        .join(level.to_string())
        .join(format!("{}_{}.jpg", x, y))
}

fn load_rgb(path: &Path) -> Result<RgbImage> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "jxl" {
        load_rgb_jxl(path)
    } else {
        let (pixels, width, height) = load_rgb_turbo(path)?;
        RgbImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow!("failed to create RGB image from pixels"))
    }
}

#[cfg(feature = "jpegxl")]
fn load_rgb_jxl(path: &Path) -> Result<RgbImage> {
    let data = fs::read(path)
        .with_context(|| format!("reading JXL file {}", path.display()))?;
    let decoder = jpegxl_rs::decode::decoder_builder()
        .build()
        .map_err(|e| anyhow!("JXL decoder build failed: {e:?}"))?;
    let (_meta, result) = decoder.reconstruct(&data)
        .map_err(|e| anyhow!("JXL reconstruct failed: {e:?}"))?;
    match result {
        jpegxl_rs::decode::Data::Jpeg(jpeg_bytes) => {
            let (pixels, width, height) = crate::turbojpeg_optimized::decode_rgb_turbo(&jpeg_bytes)?;
            RgbImage::from_raw(width, height, pixels)
                .ok_or_else(|| anyhow!("failed to create RGB image from JPEG-reconstructed pixels"))
        }
        jpegxl_rs::decode::Data::Pixels(pixels) => {
            let w = _meta.width;
            let h = _meta.height;
            let raw = match pixels {
                jpegxl_rs::decode::Pixels::Uint8(v) => v,
                _ => anyhow::bail!("JXL L2 decoded to non-u8 pixel type"),
            };
            RgbImage::from_raw(w, h, raw)
                .ok_or_else(|| anyhow!("failed to create RGB image from JXL pixels"))
        }
    }
}

#[cfg(not(feature = "jpegxl"))]
fn load_rgb_jxl(path: &Path) -> Result<RgbImage> {
    anyhow::bail!("JXL L2 tile found at {} but binary was built without --features jpegxl", path.display());
}

/// Detect if bytes are WebP format (RIFF/WEBP magic).
fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

/// Detect if bytes are JPEG-XL format.
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

/// Decode grayscale luma from WebP-encoded bytes. Returns (pixels, width, height).
fn decode_luma_webp(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
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
    Ok((gray, w as u32, h as u32))
}

/// Decode grayscale luma from JXL-encoded bytes.
#[cfg(feature = "jpegxl")]
fn decode_luma_jxl(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
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
    let (metadata, pixels): (_, Vec<u8>) = decoder
        .decode_with::<u8>(bytes)
        .map_err(|e| anyhow!("jpegxl decode failed: {e:?}"))?;
    Ok((pixels, metadata.width, metadata.height))
}

/// Decode grayscale luma from bytes, returning (pixels, width, height).
fn decode_luma_from_bytes(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
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
    let (pixels, width, height) = decode_luma_turbo(bytes)?;
    Ok((pixels, width, height))
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

/// Apply centered residual to an entire Y plane (fused L0 residual).
/// residual_y = clamp(pred_y + (residual - 128), 0..255) for each pixel.
/// The residual may be smaller than the prediction if the family is at the edge;
/// only the overlapping region is corrected.
fn apply_fused_residual(pred_y: &mut [u8], residual: &[u8], pred_w: u32, pred_h: u32, res_w: u32, res_h: u32) {
    let apply_w = pred_w.min(res_w) as usize;
    let apply_h = pred_h.min(res_h) as usize;
    for y in 0..apply_h {
        let pred_row = y * pred_w as usize;
        let res_row = y * res_w as usize;
        for x in 0..apply_w {
            let pred_val = pred_y[pred_row + x] as i16;
            let res_val = residual[res_row + x] as i16 - 128;
            pred_y[pred_row + x] = (pred_val + res_val).clamp(0, 255) as u8;
        }
    }
}

/// Convert YCbCr planes to RGB for a tile region at (tx*tile_size, ty*tile_size).
fn ycbcr_tile_to_rgb(
    y_plane: &[u8], cb_plane: &[u8], cr_plane: &[u8],
    plane_w: u32, plane_h: u32,
    tx: u32, ty: u32, tile_size: u32,
    out: &mut [u8],
) {
    use crate::core::color::ycbcr_to_rgb;
    let x0 = tx * tile_size;
    let y0 = ty * tile_size;
    for dy in 0..tile_size {
        let py = y0 + dy;
        if py >= plane_h { continue; }
        for dx in 0..tile_size {
            let px = x0 + dx;
            if px >= plane_w { continue; }
            let pi = (py * plane_w + px) as usize;
            let ti = ((dy * tile_size + dx) * 3) as usize;
            let (r, g, b) = ycbcr_to_rgb(y_plane[pi], cb_plane[pi], cr_plane[pi]);
            out[ti] = r;
            out[ti + 1] = g;
            out[ti + 2] = b;
        }
    }
}

/// Apply refine model to prediction planes (Y/Cb/Cr u8) by converting tiles to RGB,
/// running the refiner, and converting back. Modifies planes in place.
#[cfg(feature = "sr-model")]
fn refine_prediction_planes(
    y: &mut [u8], cb: &mut [u8], cr: &mut [u8],
    w: u32, h: u32, tile_size: u32,
    refine: &crate::core::sr_model::SRModel,
) {
    use crate::core::color::{ycbcr_to_rgb, rgb_to_ycbcr};
    let tiles_x = (w + tile_size - 1) / tile_size;
    let tiles_y = (h + tile_size - 1) / tile_size;
    let ts = tile_size as usize;

    for ty_idx in 0..tiles_y {
        for tx_idx in 0..tiles_x {
            let x0 = (tx_idx * tile_size) as usize;
            let y0 = (ty_idx * tile_size) as usize;

            // Extract tile → RGB
            let mut tile_rgb = vec![0u8; ts * ts * 3];
            for dy in 0..ts {
                let py = y0 + dy;
                if py >= h as usize { continue; }
                for dx in 0..ts {
                    let px = x0 + dx;
                    if px >= w as usize { continue; }
                    let pi = py * w as usize + px;
                    let ti = (dy * ts + dx) * 3;
                    let (r, g, b) = ycbcr_to_rgb(y[pi], cb[pi], cr[pi]);
                    tile_rgb[ti] = r;
                    tile_rgb[ti + 1] = g;
                    tile_rgb[ti + 2] = b;
                }
            }

            // Refine
            if refine.refine_rgb_inplace(&mut tile_rgb, tile_size, tile_size).is_err() {
                continue; // skip on error, keep original prediction
            }

            // Write back → YCbCr
            for dy in 0..ts {
                let py = y0 + dy;
                if py >= h as usize { continue; }
                for dx in 0..ts {
                    let px = x0 + dx;
                    if px >= w as usize { continue; }
                    let pi = py * w as usize + px;
                    let ti = (dy * ts + dx) * 3;
                    let (yv, cbv, crv) = rgb_to_ycbcr(tile_rgb[ti], tile_rgb[ti + 1], tile_rgb[ti + 2]);
                    y[pi] = yv;
                    cb[pi] = cbv;
                    cr[pi] = crv;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// reconstruct_family — the core fused reconstruction pipeline (v2)
// ---------------------------------------------------------------------------

/// Reconstruct all L1/L0 tiles for one L2 parent.
///
/// V2 pipeline (fused L0 residual, no L1 residuals):
/// 1. Decode L2 (256x256) → optional sharpen
/// 2. Upsample Y 4x → L0 Y prediction (1024x1024)
/// 3. Upsample Cb/Cr 2x → L1 chroma, 4x → L0 chroma (parallel)
/// 4. Decode single fused L0 residual JPEG
/// 5. Apply residual: L0_Y_corrected = clamp(L0_Y_pred + (res - 128), 0..255)
/// 6. Slice L0 into tiles → encode (parallel)
/// 7. Downsample L0_Y_corrected 2x → L1 Y
/// 8. Slice L1 into tiles → encode
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

    // --- Load pack file (bundle preferred, then individual .pack files) ---
    let pack = if let Some(bundle) = input.bundle {
        bundle.get_pack(x2, y2).ok()
    } else if let Some(pack_root) = input.pack_dir {
        open_pack(pack_root, x2, y2).ok()
    } else {
        None
    };

    // Compute L0/L1 output dimensions from tile grid (pyramid properties, not seed-dependent)
    let l0_pred_w = tile_size * 4;
    let l0_pred_h = tile_size * 4;
    let l1_w = tile_size * 2;
    let l1_h = tile_size * 2;
    let up_filter = opts.upsample_filter;

    // Check if this is a v4 split-seed pack
    let is_v4 = pack.as_ref().map_or(false, |p| p.is_split_seed());

    // --- Decode and upsample seeds, apply residual ---
    // Produces: l0_y_corrected, l0_cb, l0_cr, l1_cb, l1_cr, seed_w/h (for L2 tile gen)

    let l2_start = Instant::now();
    let l0_y_corrected: Vec<u8>;
    let l0_cb: Vec<u8>;
    let l0_cr: Vec<u8>;
    let l1_cb: Vec<u8>;
    let l1_cr: Vec<u8>;
    let seed_w: u32; // for L2 tile generation (unified) or chroma size (v4)
    let seed_h: u32;
    let seed_cb_for_l2: Vec<u8>; // seed chroma for L2 tile generation
    let seed_cr_for_l2: Vec<u8>;

    if is_v4 {
        // --- V4 split-seed path ---
        let pack = pack.as_ref().unwrap();
        let luma_w = pack.metadata.seed_luma_w as u32;
        let luma_h = pack.metadata.seed_luma_h as u32;
        let chroma_w = pack.metadata.seed_chroma_w as u32;
        let chroma_h = pack.metadata.seed_chroma_h as u32;

        // Decode seed luma (grayscale JPEG)
        let luma_bytes = pack.get_seed_luma()
            .ok_or_else(|| anyhow!("v4 pack missing seed_luma entry"))?;
        let (luma_y, _lw, _lh) = decode_luma_from_bytes(luma_bytes)?;

        // Decode seed Cb and Cr
        let chroma_cb;
        let chroma_cr;
        if let Some(cr_bytes) = pack.get_seed_cr() {
            // New format: separate Cb/Cr grayscale entries (level_kind 4 = Cb, 5 = Cr)
            let cb_bytes = pack.get_seed_cb()
                .ok_or_else(|| anyhow!("v4 pack has seed_cr but missing seed_cb entry"))?;
            chroma_cb = decode_luma_from_bytes(cb_bytes)
                .with_context(|| "decoding seed Cb from v4 pack")?.0;
            chroma_cr = decode_luma_from_bytes(cr_bytes)
                .with_context(|| "decoding seed Cr from v4 pack")?.0;
        } else {
            // Legacy: single RGB chroma entry (level_kind=4 stores RGB, no level_kind=5)
            let chroma_rgb_bytes = pack.get_seed_chroma()
                .ok_or_else(|| anyhow!("v4 pack missing seed chroma entries"))?;
            let chroma_img = decode_l2_rgb_from_bytes(chroma_rgb_bytes)
                .with_context(|| "decoding chroma seed (legacy RGB) from v4 pack")?;
            let (_, cb, cr) = ycbcr_planes(&chroma_img);
            chroma_cb = cb;
            chroma_cr = cr;
        }

        let l2_decode_ms_val = if timing { l2_start.elapsed().as_millis() } else { 0 };
        let _ = l2_decode_ms_val; // used in stats below

        // Parallel upsample from independent source sizes
        let upsample_start = Instant::now();
        let (l0_y_pred, ((l1_cb_val, l1_cr_val), (l0_cb_val, l0_cr_val))) = rayon::join(
            || {
                // Y: upsample luma seed to L0 size
                fir_resize(&luma_y, luma_w, luma_h, l0_pred_w, l0_pred_h, up_filter)
            },
            || rayon::join(
                || {
                    // Cb/Cr: upsample chroma seed to L1 size
                    let cb = fir_resize(&chroma_cb, chroma_w, chroma_h, l1_w, l1_h, up_filter);
                    let cr = fir_resize(&chroma_cr, chroma_w, chroma_h, l1_w, l1_h, up_filter);
                    (cb, cr)
                },
                || {
                    // Cb/Cr: upsample chroma seed to L0 size
                    let cb = fir_resize(&chroma_cb, chroma_w, chroma_h, l0_pred_w, l0_pred_h, up_filter);
                    let cr = fir_resize(&chroma_cr, chroma_w, chroma_h, l0_pred_w, l0_pred_h, up_filter);
                    (cb, cr)
                },
            ),
        );
        let upsample_ms_val = if timing { upsample_start.elapsed().as_millis() } else { 0 };
        let _ = upsample_ms_val;

        // Apply refine model to prediction before residual (improves prediction → smaller residual match)
        #[cfg(feature = "sr-model")]
        let (mut l0_y_pred, mut l0_cb_val, mut l0_cr_val) = if let Some(ref refine) = opts.refine_model {
            let mut y = l0_y_pred;
            let mut cb = l0_cb_val;
            let mut cr = l0_cr_val;
            refine_prediction_planes(&mut y, &mut cb, &mut cr, l0_pred_w, l0_pred_h, tile_size, refine);
            (y, cb, cr)
        } else {
            (l0_y_pred, l0_cb_val, l0_cr_val)
        };

        // Apply fused L0 residual
        let l0_res_start = Instant::now();
        let mut l0_y = l0_y_pred;
        if let Some(fused_bytes) = pack.get_fused_l0() {
            let (residual, res_w, res_h) = decode_luma_from_bytes(fused_bytes)?;
            let residual = if let Some(strength) = opts.l0_sharpen {
                crate::core::sharpen::unsharp_mask_gray(&residual, res_w, res_h, strength)
            } else {
                residual
            };
            apply_fused_residual(&mut l0_y, &residual, l0_pred_w, l0_pred_h, res_w, res_h);
        }
        // Sharpen reconstructed Y (after residual, before noise)
        if let Some(strength) = opts.tile_sharpen {
            l0_y = crate::core::sharpen::unsharp_mask_gray(&l0_y, l0_pred_w, l0_pred_h, strength);
        }
        // Optionally synthesize noise from pack params
        if opts.synth_noise {
            if let Some(synth_params) = pack.get_synth_params() {
                crate::core::wavelet::synthesize_and_apply_noise(
                    &mut l0_y, l0_pred_w, l0_pred_h, &synth_params, opts.synth_strength,
                );
            }
        }
        let _l0_residual_ms = if timing { l0_res_start.elapsed().as_millis() } else { 0 };

        l0_y_corrected = l0_y;
        l0_cb = l0_cb_val;
        l0_cr = l0_cr_val;
        l1_cb = l1_cb_val;
        l1_cr = l1_cr_val;
        seed_w = chroma_w; // for L2 tile gen
        seed_h = chroma_h;
        seed_cb_for_l2 = chroma_cb;
        seed_cr_for_l2 = chroma_cr;
    } else {
        // --- Unified seed path (v2/v3) ---
        let l2_img = if let Some(ref pack) = pack {
            if let Some(l2_bytes) = pack.get_l2() {
                decode_l2_rgb_from_bytes(l2_bytes)
                    .with_context(|| "decoding L2 from pack/bundle")?
            } else {
                let l2_path = baseline_tile_path(input.files_dir, input.l2, x2, y2);
                load_rgb(&l2_path)
                    .with_context(|| format!("loading baseline L2 tile {}", l2_path.display()))?
            }
        } else {
            let l2_path = baseline_tile_path(input.files_dir, input.l2, x2, y2);
            load_rgb(&l2_path)
                .with_context(|| format!("loading baseline L2 tile {}", l2_path.display()))?
        };

        // Optionally sharpen L2 before upsample
        let l2_img = if let Some(strength) = opts.sharpen {
            use crate::core::sharpen::unsharp_mask_rgb;
            let w = l2_img.width();
            let h = l2_img.height();
            let sharpened = unsharp_mask_rgb(l2_img.as_raw(), w, h, strength);
            RgbImage::from_raw(w, h, sharpened).expect("sharpen produced wrong size")
        } else {
            l2_img
        };

        let s_w = l2_img.width();
        let s_h = l2_img.height();

        // --- SR model path: run learned 4x super-resolution on full RGB ---
        #[cfg(feature = "sr-model")]
        let use_sr = opts.sr_model.is_some();
        #[cfg(not(feature = "sr-model"))]
        let use_sr = false;

        if use_sr {
            #[cfg(feature = "sr-model")]
            {
                let sr = opts.sr_model.as_ref().unwrap();
                let upsample_start = Instant::now();

                // Run SR model: L2 RGB → L0 RGB (256×256 → 1024×1024)
                let (sr_rgb, sr_w, sr_h) = sr.infer_rgb(l2_img.as_raw(), s_w, s_h)
                    .with_context(|| "SR model inference failed")?;

                let _upsample_ms = if timing { upsample_start.elapsed().as_millis() } else { 0 };

                // Convert SR RGB output to YCbCr planes
                let sr_img = RgbImage::from_raw(sr_w, sr_h, sr_rgb)
                    .ok_or_else(|| anyhow!("SR model produced invalid image dimensions"))?;
                let (sr_y, sr_cb_full, sr_cr_full) = ycbcr_planes(&sr_img);

                // Apply refine model on top of SR prediction
                let (mut sr_y, mut sr_cb_full, mut sr_cr_full) = if let Some(ref refine) = opts.refine_model {
                    let mut y = sr_y;
                    let mut cb = sr_cb_full;
                    let mut cr = sr_cr_full;
                    refine_prediction_planes(&mut y, &mut cb, &mut cr, sr_w, sr_h, tile_size, refine);
                    (y, cb, cr)
                } else {
                    (sr_y, sr_cb_full, sr_cr_full)
                };

                // L1 chroma: downsample SR chroma from L0 size to L1 size
                let l1_cb_val = fir_resize(&sr_cb_full, sr_w, sr_h, l1_w, l1_h, up_filter);
                let l1_cr_val = fir_resize(&sr_cr_full, sr_w, sr_h, l1_w, l1_h, up_filter);

                // Apply fused L0 residual on top of SR Y prediction (if available)
                let l0_res_start = Instant::now();
                let mut l0_y = sr_y;
                if let Some(ref pack) = pack {
                    if let Some(fused_bytes) = pack.get_fused_l0() {
                        let (residual, res_w, res_h) = decode_luma_from_bytes(fused_bytes)?;
                        let residual = if let Some(strength) = opts.l0_sharpen {
                            crate::core::sharpen::unsharp_mask_gray(&residual, res_w, res_h, strength)
                        } else {
                            residual
                        };
                        apply_fused_residual(&mut l0_y, &residual, sr_w, sr_h, res_w, res_h);
                    }
                    // Sharpen reconstructed Y (after residual, before noise)
                    if let Some(strength) = opts.tile_sharpen {
                        l0_y = crate::core::sharpen::unsharp_mask_gray(&l0_y, sr_w, sr_h, strength);
                    }
                    if opts.synth_noise {
                        if let Some(synth_params) = pack.get_synth_params() {
                            crate::core::wavelet::synthesize_and_apply_noise(
                                &mut l0_y, sr_w, sr_h, &synth_params, opts.synth_strength,
                            );
                        }
                    }
                }
                let _l0_residual_ms = if timing { l0_res_start.elapsed().as_millis() } else { 0 };

                l0_y_corrected = l0_y;
                l0_cb = sr_cb_full;
                l0_cr = sr_cr_full;
                l1_cb = l1_cb_val;
                l1_cr = l1_cr_val;

                // For L2 tile generation, use seed chroma from original L2
                let (_, s_cb_orig, s_cr_orig) = ycbcr_planes(&l2_img);
                seed_cb_for_l2 = s_cb_orig;
                seed_cr_for_l2 = s_cr_orig;
                seed_w = s_w;
                seed_h = s_h;
            }
            #[cfg(not(feature = "sr-model"))]
            unreachable!();
        } else {
            // --- Traditional upsample path ---

            // Extract seed YCbCr planes
            let (s_y, s_cb, s_cr) = ycbcr_planes(&l2_img);

            // Parallel upsample: seed → L0 Y, seed → L1 Cb/Cr, seed → L0 Cb/Cr
            let upsample_start = Instant::now();

            let (l0_y_pred, ((l1_cb_val, l1_cr_val), (l0_cb_val, l0_cr_val))) = rayon::join(
                || {
                    fir_resize(&s_y, s_w, s_h, l0_pred_w, l0_pred_h, up_filter)
                },
                || rayon::join(
                    || {
                        let cb = fir_resize(&s_cb, s_w, s_h, l1_w, l1_h, up_filter);
                        let cr = fir_resize(&s_cr, s_w, s_h, l1_w, l1_h, up_filter);
                        (cb, cr)
                    },
                    || {
                        let cb = fir_resize(&s_cb, s_w, s_h, l0_pred_w, l0_pred_h, up_filter);
                        let cr = fir_resize(&s_cr, s_w, s_h, l0_pred_w, l0_pred_h, up_filter);
                        (cb, cr)
                    },
                ),
            );
            let _upsample_ms = if timing { upsample_start.elapsed().as_millis() } else { 0 };

            // Apply refine model to prediction before residual
            #[cfg(feature = "sr-model")]
            let (mut l0_y_pred, mut l0_cb_val, mut l0_cr_val) = if let Some(ref refine) = opts.refine_model {
                let mut y = l0_y_pred;
                let mut cb = l0_cb_val;
                let mut cr = l0_cr_val;
                refine_prediction_planes(&mut y, &mut cb, &mut cr, l0_pred_w, l0_pred_h, tile_size, refine);
                (y, cb, cr)
            } else {
                (l0_y_pred, l0_cb_val, l0_cr_val)
            };

            // Decode fused L0 residual
            let l0_res_start = Instant::now();
            let mut l0_y = l0_y_pred;
            if let Some(ref pack) = pack {
                if let Some(fused_bytes) = pack.get_fused_l0() {
                    let (residual, res_w, res_h) = decode_luma_from_bytes(fused_bytes)?;
                    let residual = if let Some(strength) = opts.l0_sharpen {
                        crate::core::sharpen::unsharp_mask_gray(&residual, res_w, res_h, strength)
                    } else {
                        residual
                    };
                    apply_fused_residual(&mut l0_y, &residual, l0_pred_w, l0_pred_h, res_w, res_h);
                }
                // Sharpen reconstructed Y (after residual, before noise)
                if let Some(strength) = opts.tile_sharpen {
                    l0_y = crate::core::sharpen::unsharp_mask_gray(&l0_y, l0_pred_w, l0_pred_h, strength);
                }
                if opts.synth_noise {
                    if let Some(synth_params) = pack.get_synth_params() {
                        crate::core::wavelet::synthesize_and_apply_noise(
                            &mut l0_y, l0_pred_w, l0_pred_h, &synth_params, opts.synth_strength,
                        );
                    }
                }
            }
            let _l0_residual_ms = if timing { l0_res_start.elapsed().as_millis() } else { 0 };

            l0_y_corrected = l0_y;
            l0_cb = l0_cb_val;
            l0_cr = l0_cr_val;
            l1_cb = l1_cb_val;
            l1_cr = l1_cr_val;
            seed_w = s_w;
            seed_h = s_h;
            seed_cb_for_l2 = s_cb;
            seed_cr_for_l2 = s_cr;
        }
    }

    let l2_decode_ms = if timing { l2_start.elapsed().as_millis() } else { 0 };
    let upsample_ms = 0u128; // included in l2_decode_ms above
    let l0_residual_ms = 0u128; // included in l2_decode_ms above

    // --- Slice L0 into tiles + encode (parallel) ---
    let l0_encode_start = Instant::now();
    let tiles_x = (l0_pred_w + tile_size - 1) / tile_size;
    let tiles_y = (l0_pred_h + tile_size - 1) / tile_size;
    let l0_tile_count = (tiles_x * tiles_y).min(16) as usize; // max 16 L0 tiles per family

    let l0_parallel = ParallelStats::new();
    let tile_buf_len = (tile_size * tile_size * 3) as usize;

    let l0_results: Result<Vec<_>> = (0..l0_tile_count)
        .into_par_iter()
        .map(|idx| {
            let _guard = l0_parallel.enter();
            let dx = (idx % 4) as u32;
            let dy = (idx / 4) as u32;
            let x0 = x2 * 4 + dx;
            let y0 = y2 * 4 + dy;

            let mut buf = buffer_pool.get(tile_buf_len);
            ycbcr_tile_to_rgb(
                &l0_y_corrected, &l0_cb, &l0_cr,
                l0_pred_w, l0_pred_h,
                dx, dy, tile_size,
                &mut buf,
            );

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
            buffer_pool.put(buf);
            Ok((x0, y0, bytes))
        })
        .collect();

    let l0_results = l0_results?;
    let l0_parallel_max = l0_parallel.take_max();
    let l0_encode_ms = if timing { l0_encode_start.elapsed().as_millis() } else { 0 };

    let l0_tiles: Vec<(u32, u32, Vec<u8>)> = l0_results;

    // --- Downsample corrected L0 Y 2x → L1 Y ---
    let l1_ds_start = Instant::now();
    let l1_y = fir_resize(&l0_y_corrected, l0_pred_w, l0_pred_h, l1_w, l1_h, up_filter);
    let l1_downsample_ms = if timing { l1_ds_start.elapsed().as_millis() } else { 0 };

    // --- Slice L1 into tiles + encode ---
    let l1_encode_start = Instant::now();
    let l1_tiles_x = (l1_w + tile_size - 1) / tile_size;
    let l1_tiles_y = (l1_h + tile_size - 1) / tile_size;
    let l1_tile_count = (l1_tiles_x * l1_tiles_y).min(4) as usize;

    let mut l1_tiles: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(l1_tile_count);
    for idx in 0..l1_tile_count {
        let dx = (idx % 2) as u32;
        let dy = (idx / 2) as u32;
        let x1 = x2 * 2 + dx;
        let y1 = y2 * 2 + dy;

        let mut buf = buffer_pool.get(tile_buf_len);
        ycbcr_tile_to_rgb(
            &l1_y, &l1_cb, &l1_cr,
            l1_w, l1_h,
            dx, dy, tile_size,
            &mut buf,
        );

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
        buffer_pool.put(buf);
        l1_tiles.push((x1, y1, bytes));
    }
    let l1_encode_ms = if timing { l1_encode_start.elapsed().as_millis() } else { 0 };

    // --- Generate L2 tile when seed > tile_size ---
    // The seed in the pack is larger than the L2 tile in the DZI hierarchy,
    // so we generate the L2 tile by downsampling the reconstructed L1 Y + seed chroma.
    let l2_tile = if seed_w > tile_size {
        let l2_out_w = tile_size;
        let l2_out_h = tile_size;
        // Downsample L1 Y to L2 size
        let l2_y = fir_resize(&l1_y, l1_w, l1_h, l2_out_w, l2_out_h, up_filter);
        // Downsample seed chroma to L2 size (already at seed_w which may be > tile_size)
        let l2_cb = fir_resize(&seed_cb_for_l2, seed_w, seed_h, l2_out_w, l2_out_h, up_filter);
        let l2_cr = fir_resize(&seed_cr_for_l2, seed_w, seed_h, l2_out_w, l2_out_h, up_filter);

        let mut buf = buffer_pool.get(tile_buf_len);
        ycbcr_tile_to_rgb(
            &l2_y, &l2_cb, &l2_cr,
            l2_out_w, l2_out_h,
            0, 0, tile_size,
            &mut buf,
        );
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
        buffer_pool.put(buf);
        Some((x2, y2, bytes))
    } else {
        None
    };

    info!(
        "family x2={} y2={} l0_tiles={} l1_tiles={} l2_generated={} seed={}x{}",
        x2, y2, l0_tiles.len(), l1_tiles.len(), l2_tile.is_some(), seed_w, seed_h
    );

    let stats = if timing {
        Some(FamilyStats {
            l2_decode_ms,
            upsample_ms,
            l0_residual_ms,
            l0_encode_ms,
            l1_downsample_ms,
            l1_encode_ms,
            total_ms: total_start.elapsed().as_millis(),
            l0_parallel_max,
        })
    } else {
        None
    };

    Ok(FamilyResult {
        l2: l2_tile,
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

    if let Some((x, y, ref bytes)) = result.l2 {
        let l2_dir = out_dir.join("L2");
        fs::create_dir_all(&l2_dir)?;
        let path = l2_dir.join(format!("{}_{}{}", x, y, ext));
        fs::write(&path, bytes)
            .with_context(|| format!("writing L2 tile {}", path.display()))?;
    }

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
        let pool = BufferPool::new(2);
        let b1 = pool.get(100);
        let b2 = pool.get(100);
        let b3 = pool.get(100);
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
        }
        let max = stats.take_max();
        assert_eq!(max, 3);
    }

    // -----------------------------------------------------------------------
    // apply_fused_residual unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_fused_residual() {
        // 4x4 prediction, 128 = neutral residual
        let mut pred = vec![100u8; 16];
        let residual = vec![128u8; 16]; // no change
        apply_fused_residual(&mut pred, &residual, 4, 4, 4, 4);
        assert_eq!(pred, vec![100u8; 16]);

        // +10 residual
        let mut pred = vec![100u8; 16];
        let residual = vec![138u8; 16]; // +10
        apply_fused_residual(&mut pred, &residual, 4, 4, 4, 4);
        assert_eq!(pred, vec![110u8; 16]);

        // Clamping at 255
        let mut pred = vec![250u8; 16];
        let residual = vec![148u8; 16]; // +20
        apply_fused_residual(&mut pred, &residual, 4, 4, 4, 4);
        assert_eq!(pred, vec![255u8; 16]); // clamped
    }

    #[test]
    fn test_apply_fused_residual_edge_case() {
        // Residual smaller than prediction (edge family)
        let mut pred = vec![100u8; 16]; // 4x4
        let residual = vec![138u8; 6]; // 3x2
        apply_fused_residual(&mut pred, &residual, 4, 4, 3, 2);
        // First 2 rows, first 3 cols should be 110, rest unchanged
        assert_eq!(pred[0], 110); // (0,0)
        assert_eq!(pred[1], 110); // (1,0)
        assert_eq!(pred[2], 110); // (2,0)
        assert_eq!(pred[3], 100); // (3,0) unchanged
        assert_eq!(pred[4], 110); // (0,1)
        assert_eq!(pred[5], 110); // (1,1)
        assert_eq!(pred[6], 110); // (2,1)
        assert_eq!(pred[7], 100); // (3,1) unchanged
        assert_eq!(pred[8], 100); // (0,2) unchanged
    }

    // -----------------------------------------------------------------------
    // write_family_tiles
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_family_tiles() {
        let out_dir = std::env::temp_dir().join("origami_test_write_family_v2");
        let _ = fs::remove_dir_all(&out_dir);

        let result = FamilyResult {
            l2: None,
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

        let data = fs::read(out_dir.join("L1/10_20.jpg")).unwrap();
        assert_eq!(data, vec![0xFF, 0xD8, 0xFF, 0xD9]);

        let _ = fs::remove_dir_all(&out_dir);
    }
}
