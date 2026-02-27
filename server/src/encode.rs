use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use tracing::info;

use crate::core::color::ycbcr_planes_from_rgb;
use crate::core::jpeg::{create_encoder, ChromaSubsampling};
use crate::core::pack::{write_pack, PackWriteEntry};
use crate::core::pyramid::{discover_pyramid, parse_tile_coords};
use crate::core::residual::{center_residual, compute_residual};
use crate::core::ResampleFilter;
use crate::turbojpeg_optimized::load_rgb_turbo;

use crate::core::{upsample_4x, fir_resize};

/// Downscale a grayscale buffer by a percentage (1-100).
/// Returns (downscaled_data, new_w, new_h). If scale == 100, returns original data.
fn downscale_gray(data: &[u8], w: u32, h: u32, scale_pct: u8, filter: ResampleFilter) -> (Vec<u8>, u32, u32) {
    if scale_pct >= 100 {
        return (data.to_vec(), w, h);
    }
    let new_w = ((w as u32 * scale_pct as u32 + 50) / 100).max(1);
    let new_h = ((h as u32 * scale_pct as u32 + 50) / 100).max(1);
    (fir_resize(data, w, h, new_w, new_h, filter), new_w, new_h)
}

/// Decode grayscale residual bytes, dispatching to the correct codec.
fn decode_residual_bytes(data: &[u8]) -> Result<Vec<u8>> {
    // WebP detection: bytes 0-3 = "RIFF", bytes 8-11 = "WEBP"
    let is_webp = data.len() >= 12
        && &data[0..4] == b"RIFF"
        && &data[8..12] == b"WEBP";
    if is_webp {
        let decoder = webp::Decoder::new(data);
        let img = decoder
            .decode()
            .ok_or_else(|| anyhow::anyhow!("webp decode failed"))?;
        let rgb = img.to_vec();
        let w = img.width() as usize;
        let h = img.height() as usize;
        // Extract R channel (all channels identical for gray-encoded WebP)
        let mut gray = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                gray.push(rgb[(y * w + x) * 3]);
            }
        }
        return Ok(gray);
    }

    #[cfg(feature = "jpegxl")]
    {
        // JXL codestream: 0xFF 0x0A; JXL container: 12-byte signature
        let is_jxl = (data.len() >= 2 && data[0] == 0xFF && data[1] == 0x0A)
            || (data.len() >= 12
                && data[..12]
                    == [0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A]);
        if is_jxl {
            use jpegxl_rs::Endianness;
            use jpegxl_rs::decode::{decoder_builder, PixelFormat};
            let decoder = decoder_builder()
                .pixel_format(PixelFormat {
                    num_channels: 1,
                    endianness: Endianness::Native,
                    align: 0,
                })
                .build()
                .map_err(|e| anyhow::anyhow!("jpegxl decoder create failed: {e:?}"))?;
            let (_meta, pixels): (_, Vec<u8>) = decoder
                .decode_with::<u8>(data)
                .map_err(|e| anyhow::anyhow!("jpegxl decode failed: {e:?}"))?;
            return Ok(pixels);
        }
    }
    let (pixels, _w, _h) = crate::turbojpeg_optimized::decode_luma_turbo(data)?;
    Ok(pixels)
}

#[derive(Args, Debug)]
pub struct EncodeArgs {
    /// Path to DZI pyramid directory (containing baseline_pyramid.dzi)
    #[arg(long, required_unless_present = "image")]
    pub pyramid: Option<PathBuf>,

    /// Single image path (creates one L2 family for eval convenience)
    #[arg(long, required_unless_present = "pyramid")]
    pub image: Option<PathBuf>,

    /// Output directory for residuals and pack files
    #[arg(long)]
    pub out: PathBuf,

    /// Tile size (must match pyramid)
    #[arg(long, default_value_t = 256)]
    pub tile: u32,

    /// JPEG quality for L0 residual encoding (1-100)
    #[arg(long, default_value_t = 50)]
    pub resq: u8,

    /// Override quality for L0 residuals (default: use --resq)
    #[arg(long)]
    pub l0q: Option<u8>,

    /// Quality for L2 baseline encoding. A number (1-100) encodes as lossy JPEG XL.
    /// "lossless" or "L" encodes JPEG first, then losslessly transcodes to JXL (~16% smaller).
    #[arg(long, default_value = "95")]
    pub baseq: String,

    /// Chroma subsampling: 444, 420, 420opt
    #[arg(long, default_value = "444")]
    pub subsamp: String,

    /// Encoder backend: jpegxl, turbojpeg, mozjpeg, jpegli, webp
    #[arg(long, default_value = "jpegxl")]
    pub encoder: String,

    /// Maximum number of L2 parent tiles to process (for testing)
    #[arg(long)]
    pub max_parents: Option<usize>,

    /// Also create pack files
    #[arg(long)]
    pub pack: bool,

    /// Write manifest.json with per-tile metrics (for evals)
    #[arg(long)]
    pub manifest: bool,

    /// Optimize L2 tile for better bilinear predictions (gradient descent)
    #[arg(long)]
    pub optl2: bool,

    /// Write debug PNG images (originals, predictions, reconstructions) for viewer
    #[arg(long)]
    pub debug_images: bool,

    /// Maximum per-pixel deviation for OptL2 gradient descent (default: 15)
    #[arg(long, default_value_t = 15)]
    pub max_delta: u8,

    /// Unsharp mask strength applied to L2 for better bilinear upsample predictions.
    /// By default, stores the unsharpened L2 (fewer bytes) and the decoder must apply
    /// the same sharpen at decode time. Use --save-sharpened to store the sharpened L2.
    #[arg(long)]
    pub sharpen: Option<f32>,

    /// When used with --sharpen, store the sharpened L2 in the output JPEG.
    /// Without this flag, the unsharpened L2 is stored (smaller) and the decoder
    /// must apply the sharpen kernel before upsampling.
    #[arg(long)]
    pub save_sharpened: bool,

    /// Downscale L0 residuals before encoding (percent, 1-100, default: 100 = no downscale).
    /// Lower values reduce residual byte cost at the expense of reconstruction quality.
    /// The decoder infers the scale from dimension mismatch and upscales automatically.
    #[arg(long, default_value_t = 100)]
    pub l0_scale: u8,

    /// Unsharp mask strength for L0 residuals (e.g. 0.5-2.0).
    /// Applied to the centered residual before JPEG encoding to preserve high-frequency detail.
    #[arg(long)]
    pub l0_sharpen: Option<f32>,

    /// Upsample filter for predictions: bilinear, bicubic, lanczos3 (default: lanczos3)
    #[arg(long, default_value = "lanczos3")]
    pub upsample_filter: String,

    /// Downsample filter for ground-truth and residual downscale: bilinear, bicubic, lanczos3 (default: lanczos3)
    #[arg(long, default_value = "lanczos3")]
    pub downsample_filter: String,

    // l2jxl removed — L2 is always stored as JXL, controlled by --baseq
}

/// Return the file extension for the given encoder name.
fn output_extension(encoder_name: &str) -> &'static str {
    match encoder_name {
        "webp" => ".webp",
        "jpegxl" => ".jxl",
        _ => ".jpg",
    }
}

/// Encode L2 tile as JXL. Two modes:
/// - "lossless": Lossless JPEG→JXL transcode (~16% smaller, bit-identical JPEG roundtrip)
/// - "<number>": Lossy JXL from pixels at that quality (bigger savings, no JPEG roundtrip)
#[cfg(feature = "jpegxl")]
fn encode_l2_as_jxl(jpeg_bytes: &[u8], _pixels: &[u8], _w: u32, _h: u32, mode: &str) -> Result<Vec<u8>> {
    use jpegxl_rs::encode::{encoder_builder, ColorEncoding, EncoderFrame};

    if mode == "lossless" || mode.eq_ignore_ascii_case("l") {
        let mut enc = encoder_builder()
            .use_container(true)
            .uses_original_profile(true)
            .build()
            .map_err(|e| anyhow::anyhow!("JXL encoder build failed: {e:?}"))?;
        let result = enc.encode_jpeg(jpeg_bytes)
            .map_err(|e| anyhow::anyhow!("JXL lossless JPEG transcode failed: {e:?}"))?;
        Ok(result.data)
    } else {
        let quality: f32 = mode.parse()
            .with_context(|| format!("--l2jxl value must be 'lossless' or a quality number, got '{mode}'"))?;
        let mut enc = encoder_builder()
            .jpeg_quality(quality)
            .color_encoding(ColorEncoding::Srgb)
            .build()
            .map_err(|e| anyhow::anyhow!("JXL encoder build failed: {e:?}"))?;
        let result: jpegxl_rs::encode::EncoderResult<u8> = enc.encode_frame(
            &EncoderFrame::new(_pixels).num_channels(3),
            _w, _h,
        ).map_err(|e| anyhow::anyhow!("JXL lossy encode failed: {e:?}"))?;
        Ok(result.data)
    }
}

#[cfg(not(feature = "jpegxl"))]
fn encode_l2_as_jxl(_jpeg_bytes: &[u8], _pixels: &[u8], _w: u32, _h: u32, _mode: &str) -> Result<Vec<u8>> {
    anyhow::bail!("--l2jxl requires building with --features jpegxl");
}

/// Write an RGB buffer as a PNG file.
fn save_rgb_png(path: &std::path::Path, data: &[u8], w: u32, h: u32) -> Result<()> {
    let img = image::RgbImage::from_raw(w, h, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("failed to create RgbImage"))?;
    img.save(path).with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

/// Write a grayscale buffer as a PNG file.
fn save_gray_png(path: &std::path::Path, data: &[u8], w: u32, h: u32) -> Result<()> {
    let img = image::GrayImage::from_raw(w, h, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("failed to create GrayImage"))?;
    img.save(path).with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

/// Normalize an f32 slice to [0, 255] u8 for visualization (maps min→0, max→255).
fn normalize_f32_to_u8(data: &[f32]) -> Vec<u8> {
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    if range < 1e-8 {
        return vec![128u8; data.len()];
    }
    data.iter().map(|&v| ((v - min_val) / range * 255.0).round() as u8).collect()
}

/// Reconstruct RGB tile from luma (Y) + chroma (Cb, Cr) planes using BT.601.
#[allow(dead_code)]
fn reconstruct_rgb_from_ycbcr(y: &[u8], cb: &[u8], cr: &[u8], w: u32, h: u32) -> Vec<u8> {
    let n = (w * h) as usize;
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        let yf = y[i] as f32;
        let cbf = cb[i] as f32 - 128.0;
        let crf = cr[i] as f32 - 128.0;
        let r = (yf + 1.402 * crf).round().clamp(0.0, 255.0) as u8;
        let g = (yf - 0.344136 * cbf - 0.714136 * crf).round().clamp(0.0, 255.0) as u8;
        let b = (yf + 1.772 * cbf).round().clamp(0.0, 255.0) as u8;
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    rgb
}

pub fn run(args: EncodeArgs) -> Result<()> {
    let subsamp: ChromaSubsampling = args.subsamp.parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;

    // Parse resample filters early so errors are caught before work begins
    let _up: ResampleFilter = args.upsample_filter.parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;
    let _down: ResampleFilter = args.downsample_filter.parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;

    if args.image.is_some() {
        return run_single_image(args, subsamp);
    }

    let pyramid_path = args.pyramid.clone()
        .ok_or_else(|| anyhow::anyhow!("either --pyramid or --image must be specified"))?;

    run_pyramid(args, pyramid_path, subsamp)
}

/// Encode residuals from a pre-built DZI pyramid.
/// V2 pipeline: no L1 residuals, single fused L0 residual per family.
fn run_pyramid(args: EncodeArgs, pyramid_path: PathBuf, subsamp: ChromaSubsampling) -> Result<()> {
    let start = Instant::now();
    let encoder = create_encoder(&args.encoder)?;
    let up_filter: ResampleFilter = args.upsample_filter.parse().unwrap();
    let down_filter: ResampleFilter = args.downsample_filter.parse().unwrap();
    info!("Using encoder: {} subsamp: {} upsample: {} downsample: {}", encoder.name(), subsamp, up_filter, down_filter);

    let pyramid = discover_pyramid(&pyramid_path, args.tile)?;
    info!(
        "Pyramid: max_level={} tile_size={} l0={} l1={} l2={}",
        pyramid.max_level, pyramid.tile_size, pyramid.l0, pyramid.l1, pyramid.l2
    );

    // Discover L2 parent tiles
    let l2_dir = pyramid.files_dir.join(pyramid.l2.to_string());
    let mut parents: Vec<(u32, u32)> = Vec::new();
    for entry in fs::read_dir(&l2_dir)
        .with_context(|| format!("reading L2 dir {}", l2_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(coords) = parse_tile_coords(&name) {
            parents.push(coords);
        }
    }
    parents.sort();
    info!("Found {} L2 parent tiles", parents.len());

    if let Some(max) = args.max_parents {
        parents.truncate(max);
        info!("Limited to {} parents (--max-parents)", max);
    }

    // Create output directories
    let residuals_dir = args.out.clone();
    let l0_out = residuals_dir.join("L0");
    fs::create_dir_all(&l0_out)?;

    let pack_dir = if args.pack {
        let d = args.out.join("packs");
        fs::create_dir_all(&d)?;
        Some(d)
    } else {
        None
    };

    let tile_size = pyramid.tile_size;
    let l0q = args.l0q.unwrap_or(args.resq);
    let baseq_str = args.baseq.clone();
    let baseq_is_lossless = matches!(baseq_str.to_lowercase().as_str(), "lossless" | "l");
    let baseq_num: u8 = if baseq_is_lossless { 95 } else {
        baseq_str.parse::<u8>().map_err(|_| anyhow::anyhow!(
            "--baseq must be a number (1-100) or 'lossless', got '{}'", baseq_str))?
    };
    let use_jxl_l2 = args.encoder == "jpegxl";
    let l2_jxl_mode: &str = if baseq_is_lossless { "lossless" } else { &baseq_str };
    let l2_format = if use_jxl_l2 { "jxl" } else { "jpg" };
    let ext = output_extension(encoder.name());
    let mut total_l0 = 0usize;
    let mut total_bytes = 0usize;
    info!("L2 will be stored as {} (baseq={})", l2_format.to_uppercase(), baseq_str);

    for (pi, &(x2, y2)) in parents.iter().enumerate() {
        let parent_start = Instant::now();

        // Load L2 baseline tile
        let l2_path = pyramid
            .files_dir
            .join(pyramid.l2.to_string())
            .join(format!("{}_{}.jpg", x2, y2));
        let (l2_rgb, l2_w, l2_h) = load_rgb_turbo(&l2_path)
            .with_context(|| format!("loading L2 tile {}", l2_path.display()))?;

        // Convert to YCbCr planes
        let (l2_y, _l2_cb, _l2_cr) = ycbcr_planes_from_rgb(&l2_rgb, l2_w, l2_h);

        // Upsample L2 Y → L0 prediction (4x direct)
        let l0_pred_y = upsample_4x(&l2_y, l2_w as usize, l2_h as usize, up_filter);
        let l0_w = l2_w * 4;
        let l0_h = l2_h * 4;

        let mut pack_entries: Vec<PackWriteEntry> = Vec::new();

        // Store L2 in pack — as JXL (lossless transcode) or raw JPEG depending on encoder
        if args.pack {
            let l2_jpeg_bytes = fs::read(&l2_path)?;
            let l2_stored = if use_jxl_l2 {
                encode_l2_as_jxl(&l2_jpeg_bytes, &l2_rgb, l2_w, l2_h, l2_jxl_mode)
                    .unwrap_or_else(|_| l2_jpeg_bytes.clone())
            } else {
                l2_jpeg_bytes
            };
            total_bytes += l2_stored.len();
            pack_entries.push(PackWriteEntry {
                level_kind: 2,
                idx_in_parent: 0,
                jpeg_data: l2_stored,
            });
        }

        // Assemble L0 ground truth mosaic from pyramid tiles
        let mut l0_gt_y = vec![0u8; (l0_w * l0_h) as usize];
        let mut tiles_found = 0u32;

        for dy in 0..4u32 {
            for dx in 0..4u32 {
                let x0 = x2 * 4 + dx;
                let y0 = y2 * 4 + dy;

                let l0_gt_path = pyramid
                    .files_dir
                    .join(pyramid.l0.to_string())
                    .join(format!("{}_{}.jpg", x0, y0));
                if !l0_gt_path.exists() {
                    continue;
                }
                tiles_found += 1;
                let (l0_gt_rgb, gt_w, gt_h) = load_rgb_turbo(&l0_gt_path)?;
                let (tile_y, _, _) = ycbcr_planes_from_rgb(&l0_gt_rgb, gt_w, gt_h);

                // Copy tile Y into mosaic
                for ty in 0..gt_h.min(tile_size) {
                    for tx in 0..gt_w.min(tile_size) {
                        let mx = dx * tile_size + tx;
                        let my = dy * tile_size + ty;
                        if mx < l0_w && my < l0_h {
                            l0_gt_y[(my * l0_w + mx) as usize] = tile_y[(ty * gt_w + tx) as usize];
                        }
                    }
                }
            }
        }

        if tiles_found == 0 {
            continue;
        }

        // Compute fused L0 residual (entire mosaic)
        let raw_residual = compute_residual(&l0_gt_y, &l0_pred_y);
        let centered = center_residual(&raw_residual);

        // Optionally sharpen residual before encoding
        let centered = if let Some(strength) = args.l0_sharpen {
            crate::core::sharpen::unsharp_mask_gray(&centered, l0_w, l0_h, strength)
        } else {
            centered
        };

        // Optionally downscale residual before encoding
        let (centered_enc, enc_w, enc_h) = downscale_gray(&centered, l0_w, l0_h, args.l0_scale, down_filter);
        let jpeg_data = encoder.encode_gray(&centered_enc, enc_w, enc_h, l0q)?;
        total_l0 += 1;
        total_bytes += jpeg_data.len();

        // Write fused residual file
        let l0_parent_dir = l0_out.join(format!("{}_{}", x2, y2));
        fs::create_dir_all(&l0_parent_dir)?;
        let out_path = l0_parent_dir.join(format!("fused{}", ext));
        fs::write(&out_path, &jpeg_data)?;

        if args.pack {
            pack_entries.push(PackWriteEntry {
                level_kind: 0,
                idx_in_parent: 0,
                jpeg_data,
            });
        }

        // Write pack file
        if let Some(ref pack_dir) = pack_dir {
            write_pack(pack_dir, x2, y2, &pack_entries)?;
        }

        let parent_ms = parent_start.elapsed().as_millis();
        info!(
            "[{}/{}] L2 parent ({},{}) — fused L0 residual ({} tiles) — {}ms",
            pi + 1, parents.len(), x2, y2, tiles_found, parent_ms
        );
    }

    let elapsed = start.elapsed();
    info!(
        "Encode complete: {} fused L0 residuals, {:.1} MB total, {:.1}s",
        total_l0,
        total_bytes as f64 / 1_048_576.0,
        elapsed.as_secs_f64()
    );

    // Write summary.json
    let summary = serde_json::json!({
        "encoder": args.encoder,
        "baseq": baseq_str,
        "l0q": l0q,
        "subsamp": subsamp.to_string(),
        "l0_scale": args.l0_scale,
        "l0_sharpen": args.l0_sharpen.map(|v| (v * 10.0).round() / 10.0),
        "upsample_filter": up_filter.to_string(),
        "downsample_filter": down_filter.to_string(),
        "tile_size": tile_size,
        "l0_residuals": total_l0,
        "total_bytes": total_bytes,
        "parents": parents.len(),
        "pack": args.pack,
        "elapsed_secs": elapsed.as_secs_f64(),
        "pipeline_version": 2,
        "l2_format": l2_format,
    });
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;

    Ok(())
}

/// Single-image encode mode: creates one L2 family from a single image (for evals).
/// V2 pipeline: L2 baseline + single fused L0 residual (no L1 residuals).
fn run_single_image(args: EncodeArgs, subsamp: ChromaSubsampling) -> Result<()> {
    // Enable Huffman optimization for all turbojpeg calls (smaller files, ~20% for small images)
    std::env::set_var("TJ_OPTIMIZE", "1");

    let start = Instant::now();
    let image_path = args.image.as_ref().unwrap();
    let encoder = create_encoder(&args.encoder)?;
    let up_filter: ResampleFilter = args.upsample_filter.parse().unwrap();
    let down_filter: ResampleFilter = args.downsample_filter.parse().unwrap();
    info!("Single-image mode: {} encoder={} subsamp={} upsample={} downsample={}", image_path.display(), encoder.name(), subsamp, up_filter, down_filter);

    // Per-stage timing accumulator
    let mut timings: Vec<(&str, f64)> = Vec::new();
    let mut t = Instant::now();

    // Load the source image
    let (src_rgb, src_w, src_h) = load_rgb_turbo(image_path)
        .with_context(|| format!("loading source image {}", image_path.display()))?;
    timings.push(("load_source", t.elapsed().as_secs_f64())); t = Instant::now();
    info!("Source image: {}x{}", src_w, src_h);

    let tile_size = args.tile;
    let l0q = args.l0q.unwrap_or(args.resq);
    let baseq_str = args.baseq.clone();
    let baseq_is_lossless = matches!(baseq_str.to_lowercase().as_str(), "lossless" | "l");
    let baseq_num: u8 = if baseq_is_lossless { 95 } else {
        baseq_str.parse::<u8>().map_err(|_| anyhow::anyhow!(
            "--baseq must be a number (1-100) or 'lossless', got '{}'", baseq_str))?
    };
    let ext = output_extension(encoder.name());

    // Create output directory
    fs::create_dir_all(&args.out)?;

    // The source image is L0. Compute grid dimensions.
    let tiles_x = (src_w + tile_size - 1) / tile_size;
    let tiles_y = (src_h + tile_size - 1) / tile_size;
    info!("L0 grid: {}x{} tiles ({}x{})", tiles_x, tiles_y, tile_size, tile_size);

    // Build L1 by downsampling L0 source 2:1 (needed for OptL2 target)
    let l1_w = (src_w + 1) / 2;
    let l1_h = (src_h + 1) / 2;
    let l1_rgb = {
        use image::{RgbImage, imageops};
        let src_img = RgbImage::from_raw(src_w, src_h, src_rgb.clone())
            .expect("failed to create RgbImage from source");
        let resized = imageops::resize(&src_img, l1_w, l1_h, down_filter.to_image_filter());
        resized.into_raw()
    };
    timings.push(("downsample_l0_to_l1", t.elapsed().as_secs_f64())); t = Instant::now();

    // Build L2 by downsampling directly from L0 source 4:1
    let l2_w = (l1_w + 1) / 2;
    let l2_h = (l1_h + 1) / 2;
    let l2_rgb = {
        use image::{RgbImage, imageops};
        let src_img = RgbImage::from_raw(src_w, src_h, src_rgb.clone())
            .expect("failed to create RgbImage from source");
        let resized = imageops::resize(&src_img, l2_w, l2_h, down_filter.to_image_filter());
        resized.into_raw()
    };
    timings.push(("downsample_l0_to_l2", t.elapsed().as_secs_f64())); t = Instant::now();

    // L2 pipeline order: OptL2 first (on smooth base), then sharpen (on optimized base).
    let mut l2_to_encode = l2_rgb.clone();

    // Step 1: Optionally optimize L2 for better predictions (gradient descent on smooth base)
    if args.optl2 {
        use crate::core::optimize_l2::optimize_l2_for_prediction;
        info!("Running OptL2 gradient descent optimization...");
        l2_to_encode = optimize_l2_for_prediction(&l2_to_encode, &l1_rgb, l2_w, l2_h, l1_w, l1_h, args.max_delta, 100, 0.3, up_filter);
    }
    timings.push(("optl2", t.elapsed().as_secs_f64())); t = Instant::now();

    // Step 2: Optionally sharpen L2
    let l2_sharpened = if let Some(strength) = args.sharpen {
        use crate::core::sharpen::unsharp_mask_rgb;
        info!("Applying unsharp mask (strength={:.2}) to L2...", strength);
        Some(unsharp_mask_rgb(&l2_to_encode, l2_w, l2_h, strength))
    } else {
        None
    };
    timings.push(("sharpen_l2", t.elapsed().as_secs_f64())); t = Instant::now();

    if args.save_sharpened {
        if let Some(ref sharpened) = l2_sharpened {
            l2_to_encode = sharpened.clone();
        }
    }

    // Encode L2 baseline using the selected encoder.
    // For JXL: encode as JPEG first (for roundtrip decode), then losslessly transcode to JXL for storage.
    // For JPEG-family encoders: encode once, store as JPEG.
    let use_jxl_l2 = args.encoder == "jpegxl";
    let l2_jpeg_encoder = create_encoder("turbojpeg")?;
    let l2_jpeg = l2_jpeg_encoder.encode_rgb_with_subsamp(&l2_to_encode, l2_w, l2_h, baseq_num, subsamp)?;
    timings.push(("encode_l2_jpeg", t.elapsed().as_secs_f64())); t = Instant::now();

    let (l2_stored, l2_format) = if use_jxl_l2 {
        let l2_jxl_mode = if baseq_is_lossless { "lossless" } else { &baseq_str };
        let l2_jxl_data = encode_l2_as_jxl(&l2_jpeg, &l2_to_encode, l2_w, l2_h, l2_jxl_mode)?;
        timings.push(("transcode_l2_jxl", t.elapsed().as_secs_f64())); t = Instant::now();
        fs::write(args.out.join("L2_0_0.jxl"), &l2_jxl_data)?;
        fs::write(args.out.join("L2_0_0.jpg"), &l2_jpeg)?; // viewer fallback
        info!("L2 baseline: {}x{} → {} bytes as JXL ({} mode, JPEG was {} bytes, Q{} {})",
            l2_w, l2_h, l2_jxl_data.len(), l2_jxl_mode, l2_jpeg.len(), baseq_str, subsamp);
        (l2_jxl_data, "jxl")
    } else {
        timings.push(("transcode_l2_jxl", 0.0)); t = Instant::now(); // skip
        fs::write(args.out.join(format!("L2_0_0{}", ext)), &l2_jpeg)?;
        info!("L2 baseline: {}x{} → {} bytes as JPEG (Q{} {})",
            l2_w, l2_h, l2_jpeg.len(), baseq_str, subsamp);
        (l2_jpeg.clone(), "jpg")
    };
    let l2_bytes = l2_stored.len();

    // Decode L2 back (lossy round-trip) to get actual L2 that decoder will see.
    let (l2_decoded_rgb, _, _) = crate::turbojpeg_optimized::decode_rgb_turbo(&l2_jpeg)?;
    timings.push(("decode_l2_roundtrip", t.elapsed().as_secs_f64())); t = Instant::now();

    // If --sharpen without --save-sharpened: sharpen the decoded L2 for predictions.
    let l2_decoded_for_pred = if l2_sharpened.is_some() && !args.save_sharpened {
        use crate::core::sharpen::unsharp_mask_rgb;
        let strength = args.sharpen.unwrap();
        info!("Sharpen-decode: applying unsharp mask (strength={:.2}) to decoded L2 for predictions...", strength);
        unsharp_mask_rgb(&l2_decoded_rgb, l2_w, l2_h, strength)
    } else {
        l2_decoded_rgb.clone()
    };
    timings.push(("sharpen_decode_l2", t.elapsed().as_secs_f64())); t = Instant::now();

    let l2_for_prediction = l2_decoded_for_pred;

    // Setup debug image directories
    let compress_dir = args.out.join("compress");
    let decompress_dir = args.out.join("decompress");

    if args.debug_images {
        fs::create_dir_all(&compress_dir)?;
        fs::create_dir_all(&decompress_dir)?;
        let (l2_y, l2_cb, l2_cr) = ycbcr_planes_from_rgb(&l2_decoded_rgb, l2_w, l2_h);
        save_rgb_png(&compress_dir.join("001_L2_original.png"), &l2_to_encode, l2_w, l2_h)?;
        save_gray_png(&compress_dir.join("002_L2_luma.png"), &l2_y, l2_w, l2_h)?;
        save_gray_png(&compress_dir.join("003_L2_chroma_cb.png"), &l2_cb, l2_w, l2_h)?;
        save_gray_png(&compress_dir.join("004_L2_chroma_cr.png"), &l2_cr, l2_w, l2_h)?;
        save_rgb_png(&decompress_dir.join("050_L2_decode.png"), &l2_decoded_rgb, l2_w, l2_h)?;
        save_rgb_png(&decompress_dir.join("050_L2_for_prediction.png"), &l2_for_prediction, l2_w, l2_h)?;
    }

    // Use float YCbCr for prediction (avoids u8 quantization loss)
    use crate::core::color::{ycbcr_planes_from_rgb_f32, rgb_from_ycbcr_f32};
    use crate::core::residual::compute_residual_f32;

    // Upsample L2 → L0 prediction directly (4x, skip L1 intermediate)
    let l0_pred_rgb = {
        use image::{RgbImage, imageops};
        let l2_img = RgbImage::from_raw(l2_w, l2_h, l2_for_prediction)
            .expect("failed to create RgbImage from L2 reconstructed");
        let resized = imageops::resize(&l2_img, src_w, src_h, up_filter.to_image_filter());
        resized.into_raw()
    };
    timings.push(("upsample_l2_to_l0_4x", t.elapsed().as_secs_f64())); t = Instant::now();

    let (l0_pred_y_f32, l0_pred_cb_f32, l0_pred_cr_f32) = ycbcr_planes_from_rgb_f32(&l0_pred_rgb, src_w, src_h);
    timings.push(("ycbcr_convert_prediction", t.elapsed().as_secs_f64())); t = Instant::now();

    if args.debug_images {
        save_rgb_png(&decompress_dir.join("060_L0_mosaic_prediction.png"), &l0_pred_rgb, src_w, src_h)?;
    }

    let mut total_bytes = l2_bytes;
    let mut manifest_tiles: Vec<serde_json::Value> = Vec::new();

    // Pack entries for single-image mode
    let mut pack_entries: Vec<PackWriteEntry> = Vec::new();
    if args.pack {
        pack_entries.push(PackWriteEntry {
            level_kind: 2,
            idx_in_parent: 0,
            jpeg_data: l2_stored.clone(),
        });
    }

    // Compute fused L0 residual (full source image size)
    // Extract ground truth Y from source
    let (src_gt_y, _, _) = ycbcr_planes_from_rgb(&src_rgb, src_w, src_h);
    timings.push(("ycbcr_convert_source", t.elapsed().as_secs_f64())); t = Instant::now();

    // Float-precision residual for the entire L0 mosaic
    let centered = compute_residual_f32(&src_gt_y, &l0_pred_y_f32);

    // Optionally sharpen residual before encoding
    let centered = if let Some(strength) = args.l0_sharpen {
        crate::core::sharpen::unsharp_mask_gray(&centered, src_w, src_h, strength)
    } else {
        centered
    };

    // Optionally downscale residual before encoding
    let (centered_enc, enc_w, enc_h) = downscale_gray(&centered, src_w, src_h, args.l0_scale, down_filter);
    timings.push(("compute_residual", t.elapsed().as_secs_f64())); t = Instant::now();

    let res_data = encoder.encode_gray(&centered_enc, enc_w, enc_h, l0q)?;
    timings.push(("encode_residual", t.elapsed().as_secs_f64())); t = Instant::now();
    total_bytes += res_data.len();

    // Write fused L0 residual
    let l0_out_dir = args.out.join("L0").join("0_0");
    fs::create_dir_all(&l0_out_dir)?;
    let out_path = l0_out_dir.join(format!("fused{}", ext));
    fs::write(&out_path, &res_data)?;

    if args.pack {
        pack_entries.push(PackWriteEntry {
            level_kind: 0,
            idx_in_parent: 0,
            jpeg_data: res_data.clone(),
        });
    }

    // Debug images for the fused residual
    if args.debug_images {
        // Raw residual (normalized for visualization)
        let residual_raw: Vec<f32> = src_gt_y.iter().zip(l0_pred_y_f32.iter())
            .map(|(&gt_val, &pred_val)| gt_val as f32 - pred_val).collect();
        let raw_normalized = normalize_f32_to_u8(&residual_raw);
        save_gray_png(&compress_dir.join("020_L0_fused_residual_raw.png"),
            &raw_normalized, src_w, src_h)?;
        save_gray_png(&compress_dir.join("021_L0_fused_residual_centered.png"),
            &centered, src_w, src_h)?;
    }

    // Compute per-tile metrics and debug images
    if args.manifest || args.debug_images {
        // Decode residual back for reconstruction quality measurement
        let decoded_res = decode_residual_bytes(&res_data)?;
        // If residual was downscaled, we need to upscale for metrics
        let decoded_res_full = if args.l0_scale < 100 {
            fir_resize(&decoded_res, enc_w, enc_h, src_w, src_h, up_filter)
        } else {
            decoded_res
        };

        // Reconstruct full L0 Y
        let mut recon_y = vec![0.0f32; (src_w * src_h) as usize];
        for i in 0..(src_w * src_h) as usize {
            let pred_val = l0_pred_y_f32[i];
            let res_val = decoded_res_full[i] as f32 - 128.0;
            recon_y[i] = (pred_val + res_val).clamp(0.0, 255.0);
        }

        if args.debug_images {
            // Full reconstructed RGB
            let recon_rgb = rgb_from_ycbcr_f32(&recon_y, &l0_pred_cb_f32, &l0_pred_cr_f32);
            save_rgb_png(&decompress_dir.join("070_L0_reconstructed.png"),
                &recon_rgb, src_w, src_h)?;

            // Per-tile reconstructed images
            for ty in 0..tiles_y {
                for tx in 0..tiles_x {
                    let tile_n = (tile_size * tile_size) as usize;
                    let mut tile_y = vec![0.0f32; tile_n];
                    let mut tile_cb = vec![0.0f32; tile_n];
                    let mut tile_cr = vec![0.0f32; tile_n];
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let ti = (y * tile_size + x) as usize;
                            let mx = tx * tile_size + x;
                            let my = ty * tile_size + y;
                            if mx < src_w && my < src_h {
                                let mi = (my * src_w + mx) as usize;
                                tile_y[ti] = recon_y[mi];
                                tile_cb[ti] = l0_pred_cb_f32[mi];
                                tile_cr[ti] = l0_pred_cr_f32[mi];
                            }
                        }
                    }
                    let tile_rgb = rgb_from_ycbcr_f32(&tile_y, &tile_cb, &tile_cr);
                    let step = 70 + (ty * tiles_x + tx) as u32;
                    save_rgb_png(&decompress_dir.join(format!("{:03}_L0_{}_{}_reconstructed.png", step, tx, ty)),
                        &tile_rgb, tile_size, tile_size)?;
                }
            }
        }

        if args.manifest {
            // Per-tile PSNR
            for ty in 0..tiles_y {
                for tx in 0..tiles_x {
                    let mut mse_sum = 0.0f64;
                    let mut count = 0usize;
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let mx = tx * tile_size + x;
                            let my = ty * tile_size + y;
                            if mx < src_w && my < src_h {
                                let mi = (my * src_w + mx) as usize;
                                let recon_val = recon_y[mi] as f64;
                                let gt_val = src_gt_y[mi] as f64;
                                let diff = recon_val - gt_val;
                                mse_sum += diff * diff;
                                count += 1;
                            }
                        }
                    }
                    let mse = if count > 0 { mse_sum / count as f64 } else { 0.0 };
                    let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 100.0 };

                    manifest_tiles.push(serde_json::json!({
                        "level": "L0",
                        "tx": tx, "ty": ty,
                        "y_psnr_db": (psnr * 100.0).round() / 100.0,
                        "y_mse": (mse * 100.0).round() / 100.0,
                    }));
                }
            }
        }

        // --- L1 debug images and metrics ---
        // L1 tiles are derived by downscaling corrected L0 Y + using L2 chroma 2x
        let l1_tiles_x = (l1_w + tile_size - 1) / tile_size;
        let l1_tiles_y = (l1_h + tile_size - 1) / tile_size;

        // Reconstruct L1 Y by downscaling corrected L0 Y 2x
        let l1_recon_y_u8: Vec<u8> = {
            let recon_y_u8: Vec<u8> = recon_y.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
            fir_resize(&recon_y_u8, src_w, src_h, l1_w, l1_h, up_filter)
        };

        // L1 chroma: upsample L2 chroma 2x (same as decode pipeline)
        let l1_pred_cb_f32_full = ycbcr_planes_from_rgb_f32(&{
            use image::{RgbImage, imageops};
            let l2_img = RgbImage::from_raw(l2_w, l2_h, l2_decoded_rgb.clone())
                .expect("l2 for l1 chroma");
            let resized = imageops::resize(&l2_img, l1_w, l1_h, up_filter.to_image_filter());
            resized.into_raw()
        }, l1_w, l1_h);
        let l1_pred_cb_f32_ds = l1_pred_cb_f32_full.1;
        let l1_pred_cr_f32_ds = l1_pred_cb_f32_full.2;

        if args.debug_images {
            // L0 per-tile source images (compress dir)
            for ty in 0..tiles_y {
                for tx in 0..tiles_x {
                    let mut tile = vec![0u8; (tile_size * tile_size * 3) as usize];
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let sx = tx * tile_size + x;
                            let sy = ty * tile_size + y;
                            let di = ((y * tile_size + x) * 3) as usize;
                            if sx < src_w && sy < src_h {
                                let si = ((sy * src_w + sx) * 3) as usize;
                                tile[di] = src_rgb[si];
                                tile[di + 1] = src_rgb[si + 1];
                                tile[di + 2] = src_rgb[si + 2];
                            }
                        }
                    }
                    let step = 20 + (ty * tiles_x + tx) as u32;
                    save_rgb_png(&compress_dir.join(format!("{:03}_L0_{}_{}_original.png", step, tx, ty)),
                        &tile, tile_size, tile_size)?;

                    // Luma of original
                    let (tile_gt_y, _, _) = ycbcr_planes_from_rgb(&tile, tile_size, tile_size);
                    save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_luma.png", step, tx, ty)),
                        &tile_gt_y, tile_size, tile_size)?;

                    // Prediction tile + chroma channels
                    let tile_n = (tile_size * tile_size) as usize;
                    let mut pred_tile = vec![0u8; tile_n * 3];
                    let mut pred_cb_tile = vec![0u8; tile_n];
                    let mut pred_cr_tile = vec![0u8; tile_n];
                    let mut pred_y_tile_f32 = vec![0.0f32; tile_n];
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let mx = tx * tile_size + x;
                            let my = ty * tile_size + y;
                            let ti = (y * tile_size + x) as usize;
                            if mx < src_w && my < src_h {
                                let si = ((my * src_w + mx) * 3) as usize;
                                let mi = (my * src_w + mx) as usize;
                                pred_tile[ti * 3] = l0_pred_rgb[si];
                                pred_tile[ti * 3 + 1] = l0_pred_rgb[si + 1];
                                pred_tile[ti * 3 + 2] = l0_pred_rgb[si + 2];
                                pred_cb_tile[ti] = l0_pred_cb_f32[mi].round().clamp(0.0, 255.0) as u8;
                                pred_cr_tile[ti] = l0_pred_cr_f32[mi].round().clamp(0.0, 255.0) as u8;
                                pred_y_tile_f32[ti] = l0_pred_y_f32[mi];
                            }
                        }
                    }
                    save_rgb_png(&compress_dir.join(format!("{:03}_L0_{}_{}_prediction.png", step, tx, ty)),
                        &pred_tile, tile_size, tile_size)?;
                    save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_chroma_cb.png", step, tx, ty)),
                        &pred_cb_tile, tile_size, tile_size)?;
                    save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_chroma_cr.png", step, tx, ty)),
                        &pred_cr_tile, tile_size, tile_size)?;

                    // Per-tile residual (extracted from the fused residual)
                    let residual_raw_tile: Vec<f32> = tile_gt_y.iter().zip(pred_y_tile_f32.iter())
                        .map(|(&gt_val, &pred_val)| gt_val as f32 - pred_val).collect();
                    let raw_norm = normalize_f32_to_u8(&residual_raw_tile);
                    save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_residual_raw.png", step, tx, ty)),
                        &raw_norm, tile_size, tile_size)?;

                    // Centered residual (what gets encoded)
                    let centered_tile: Vec<u8> = tile_gt_y.iter().zip(pred_y_tile_f32.iter())
                        .map(|(&gt_val, &pred_val)| (gt_val as f32 - pred_val + 128.0).round().clamp(0.0, 255.0) as u8)
                        .collect();
                    save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_residual_centered.png", step, tx, ty)),
                        &centered_tile, tile_size, tile_size)?;
                }
            }

            // L1 per-tile debug images
            for ty in 0..l1_tiles_y {
                for tx in 0..l1_tiles_x {
                    let tile_n = (tile_size * tile_size) as usize;
                    let step = 10 + (ty * l1_tiles_x + tx) as u32;

                    // L1 original (from downsampled source)
                    let mut l1_orig_tile = vec![0u8; tile_n * 3];
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let sx = tx * tile_size + x;
                            let sy = ty * tile_size + y;
                            let di = ((y * tile_size + x) * 3) as usize;
                            if sx < l1_w && sy < l1_h {
                                let si = ((sy * l1_w + sx) * 3) as usize;
                                l1_orig_tile[di] = l1_rgb[si];
                                l1_orig_tile[di + 1] = l1_rgb[si + 1];
                                l1_orig_tile[di + 2] = l1_rgb[si + 2];
                            }
                        }
                    }
                    save_rgb_png(&compress_dir.join(format!("{:03}_L1_{}_{}_original.png", step, tx, ty)),
                        &l1_orig_tile, tile_size, tile_size)?;

                    // L1 luma of original
                    let (l1_tile_gt_y, _, _) = ycbcr_planes_from_rgb(&l1_orig_tile, tile_size, tile_size);
                    save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_luma.png", step, tx, ty)),
                        &l1_tile_gt_y, tile_size, tile_size)?;

                    // L1 prediction (upsampled L2)
                    // Use the L2→L1 upsample for prediction
                    let l1_pred_rgb_full = {
                        use image::{RgbImage, imageops};
                        let l2_img = RgbImage::from_raw(l2_w, l2_h, l2_decoded_rgb.clone())
                            .expect("l2 for l1 pred");
                        let resized = imageops::resize(&l2_img, l1_w, l1_h, up_filter.to_image_filter());
                        resized.into_raw()
                    };
                    let mut l1_pred_tile = vec![0u8; tile_n * 3];
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let sx = tx * tile_size + x;
                            let sy = ty * tile_size + y;
                            let di = ((y * tile_size + x) * 3) as usize;
                            if sx < l1_w && sy < l1_h {
                                let si = ((sy * l1_w + sx) * 3) as usize;
                                l1_pred_tile[di] = l1_pred_rgb_full[si];
                                l1_pred_tile[di + 1] = l1_pred_rgb_full[si + 1];
                                l1_pred_tile[di + 2] = l1_pred_rgb_full[si + 2];
                            }
                        }
                    }
                    save_rgb_png(&compress_dir.join(format!("{:03}_L1_{}_{}_prediction.png", step, tx, ty)),
                        &l1_pred_tile, tile_size, tile_size)?;

                    // L1 chroma channels (from prediction)
                    let l1_pred_cb_tile: Vec<u8> = (0..tile_n).map(|i| {
                        let mx = (i as u32 % tile_size) + tx * tile_size;
                        let my = (i as u32 / tile_size) + ty * tile_size;
                        if mx < l1_w && my < l1_h {
                            l1_pred_cb_f32_ds[(my * l1_w + mx) as usize].round().clamp(0.0, 255.0) as u8
                        } else { 128 }
                    }).collect();
                    let l1_pred_cr_tile: Vec<u8> = (0..tile_n).map(|i| {
                        let mx = (i as u32 % tile_size) + tx * tile_size;
                        let my = (i as u32 / tile_size) + ty * tile_size;
                        if mx < l1_w && my < l1_h {
                            l1_pred_cr_f32_ds[(my * l1_w + mx) as usize].round().clamp(0.0, 255.0) as u8
                        } else { 128 }
                    }).collect();
                    save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_chroma_cb.png", step, tx, ty)),
                        &l1_pred_cb_tile, tile_size, tile_size)?;
                    save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_chroma_cr.png", step, tx, ty)),
                        &l1_pred_cr_tile, tile_size, tile_size)?;

                    // L1 reconstructed (from downscaled corrected L0 Y + L2 chroma 2x)
                    let mut l1_recon_tile_y = vec![0.0f32; tile_n];
                    let mut l1_recon_tile_cb = vec![0.0f32; tile_n];
                    let mut l1_recon_tile_cr = vec![0.0f32; tile_n];
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let ti = (y * tile_size + x) as usize;
                            let mx = tx * tile_size + x;
                            let my = ty * tile_size + y;
                            if mx < l1_w && my < l1_h {
                                let mi = (my * l1_w + mx) as usize;
                                l1_recon_tile_y[ti] = l1_recon_y_u8[mi] as f32;
                                l1_recon_tile_cb[ti] = l1_pred_cb_f32_ds[mi];
                                l1_recon_tile_cr[ti] = l1_pred_cr_f32_ds[mi];
                            }
                        }
                    }
                    let l1_recon_rgb = rgb_from_ycbcr_f32(&l1_recon_tile_y, &l1_recon_tile_cb, &l1_recon_tile_cr);
                    save_rgb_png(&decompress_dir.join(format!("{:03}_L1_{}_{}_reconstructed.png", 63, tx, ty)),
                        &l1_recon_rgb, tile_size, tile_size)?;
                }
            }
        }

        // L1 manifest metrics
        if args.manifest {
            let (l1_gt_y_all, _, _) = ycbcr_planes_from_rgb(&l1_rgb, l1_w, l1_h);
            for ty in 0..l1_tiles_y {
                for tx in 0..l1_tiles_x {
                    let mut mse_sum = 0.0f64;
                    let mut count = 0usize;
                    for y in 0..tile_size {
                        for x in 0..tile_size {
                            let mx = tx * tile_size + x;
                            let my = ty * tile_size + y;
                            if mx < l1_w && my < l1_h {
                                let mi = (my * l1_w + mx) as usize;
                                let recon_val = l1_recon_y_u8[mi] as f64;
                                let gt_val = l1_gt_y_all[mi] as f64;
                                let diff = recon_val - gt_val;
                                mse_sum += diff * diff;
                                count += 1;
                            }
                        }
                    }
                    let mse = if count > 0 { mse_sum / count as f64 } else { 0.0 };
                    let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 100.0 };

                    manifest_tiles.push(serde_json::json!({
                        "level": "L1",
                        "tx": tx, "ty": ty,
                        "y_psnr_db": (psnr * 100.0).round() / 100.0,
                        "y_mse": (mse * 100.0).round() / 100.0,
                    }));
                }
            }
        }
    }

    timings.push(("debug_images_and_metrics", t.elapsed().as_secs_f64())); t = Instant::now();

    // Write pack file for single-image mode (one family at 0,0)
    if args.pack && !pack_entries.is_empty() {
        let pack_dir = args.out.join("packs");
        fs::create_dir_all(&pack_dir)?;
        write_pack(&pack_dir, 0, 0, &pack_entries)?;
        info!("Pack file written: {} entries (L2=1, L0=1)", pack_entries.len());
    }
    timings.push(("write_pack", t.elapsed().as_secs_f64()));

    let elapsed = start.elapsed();
    info!(
        "Single-image encode complete: {} L0 tiles, fused residual {} bytes, {} total bytes, {:.1}s",
        tiles_x * tiles_y, res_data.len(), total_bytes, elapsed.as_secs_f64()
    );

    // Print per-stage timing breakdown
    info!("=== TIMING BREAKDOWN ===");
    let mut accounted = 0.0f64;
    for (stage, secs) in &timings {
        accounted += secs;
        info!("  {:30} {:>8.3}ms", stage, secs * 1000.0);
    }
    info!("  {:30} {:>8.3}ms", "TOTAL (accounted)", accounted * 1000.0);
    info!("  {:30} {:>8.3}ms", "TOTAL (wall clock)", elapsed.as_secs_f64() * 1000.0);

    // Write summary.json
    let summary = serde_json::json!({
        "mode": "single-image",
        "source": image_path.to_string_lossy(),
        "source_w": src_w,
        "source_h": src_h,
        "encoder": args.encoder,
        "subsamp": subsamp.to_string(),
        "baseq": baseq_str,
        "l0q": l0q,
        "optl2": args.optl2,
        "sharpen": args.sharpen.map(|v| (v * 10.0).round() / 10.0),
        "save_sharpened": args.save_sharpened,
        "l0_scale": args.l0_scale,
        "l0_sharpen": args.l0_sharpen.map(|v| (v * 10.0).round() / 10.0),
        "upsample_filter": up_filter.to_string(),
        "downsample_filter": down_filter.to_string(),
        "tile_size": tile_size,
        "l2_bytes": l2_bytes,
        "l2_w": l2_w,
        "l2_h": l2_h,
        "l0_tiles": tiles_x * tiles_y,
        "total_bytes": total_bytes,
        "elapsed_secs": elapsed.as_secs_f64(),
        "pipeline_version": 2,
        "timing_breakdown_ms": timings.iter().map(|(k, v)| (k.to_string(), (v * 1000.0 * 100.0).round() / 100.0)).collect::<std::collections::HashMap<_, _>>(),
    });
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;

    // Write manifest.json (detailed per-tile metrics for viewer)
    if args.manifest {
        let manifest = serde_json::json!({
            "mode": "single-image",
            "source": image_path.to_string_lossy(),
            "source_w": src_w,
            "source_h": src_h,
            "encoder": args.encoder,
            "subsamp": subsamp.to_string(),
            "baseq": baseq_str,
            "l0q": l0q,
            "optl2": args.optl2,
            "sharpen": args.sharpen.map(|v| (v * 10.0).round() / 10.0),
            "l0_scale": args.l0_scale,
            "l0_sharpen": args.l0_sharpen.map(|v| (v * 10.0).round() / 10.0),
            "l2_format": l2_format,
            "upsample_filter": up_filter.to_string(),
            "downsample_filter": down_filter.to_string(),
            "tile_size": tile_size,
            "l2_bytes": l2_bytes,
            "l2_w": l2_w,
            "l2_h": l2_h,
            "fused_l0_bytes": res_data.len(),
            "total_bytes": total_bytes,
            "tiles": manifest_tiles,
            "pipeline_version": 2,
        });
        fs::write(
            args.out.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;
    }

    Ok(())
}
