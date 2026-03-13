use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use tracing::info;

use crate::core::color::ycbcr_planes_from_rgb;
use crate::core::jpeg::{create_encoder, ChromaSubsampling};
use crate::core::pack::{write_pack, write_pack_v3, write_pack_v4, PackWriteEntry, PackMetadata};
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

    /// Seed image quality. A number (1-100) encodes as lossy JPEG/JXL.
    /// "lossless" or "L" encodes JPEG first, then losslessly transcodes to JXL (~16% smaller).
    #[arg(long, alias = "baseq", default_value = "95")]
    pub seedq: String,

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

    /// Optimize L2 luma only (preserves chroma exactly, avoids color shifts)
    #[arg(long)]
    pub optl2_luma: bool,

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
    /// Applied to the decoded residual at decode time (sharpens correction detail without increasing file size).
    #[arg(long)]
    pub l0_sharpen: Option<f32>,

    /// Unsharp mask strength applied to final reconstructed Y plane (after residual + l0-sharpen, before noise).
    #[arg(long)]
    pub tile_sharpen: Option<f32>,

    /// Upsample filter for predictions: bilinear, bicubic, lanczos3 (default: lanczos3)
    #[arg(long, default_value = "lanczos3")]
    pub upsample_filter: String,

    /// Downsample filter for ground-truth and residual downscale: bilinear, bicubic, lanczos3 (default: lanczos3)
    #[arg(long, default_value = "lanczos3")]
    pub downsample_filter: String,

    /// Path to SR ONNX model for learned 4x super-resolution prediction (replaces upsample filter)
    #[arg(long)]
    pub sr_model: Option<String>,

    /// Number of ONNX Runtime intra-op threads for SR model (default: 4)
    #[arg(long, default_value_t = 4)]
    pub sr_threads: usize,

    /// Path to refine ONNX model for same-resolution L0 tile enhancement
    #[arg(long)]
    pub refine_model: Option<String>,

    /// Number of ONNX Runtime intra-op threads for refine model (default: 2)
    #[arg(long, default_value_t = 2)]
    pub refine_threads: usize,

    /// Number of refine model session copies for concurrent inference (default: 4)
    #[arg(long, default_value_t = 4)]
    pub refine_pool: usize,

    /// Seed image size in pixels (default: tile_size = L2). Use "l2" for tile_size,
    /// "l1" for tile_size*2, or a number (e.g. 384) for a custom size.
    /// Larger seeds produce sharper predictions and smaller residuals at the cost of bigger seed images.
    #[arg(long)]
    pub seed_size: Option<String>,

    /// Split-seed mode: luma seed size in pixels (enables v4 pack format).
    /// When set, luma and chroma seeds are encoded separately for independent size/quality control.
    #[arg(long)]
    pub seed_luma_size: Option<String>,

    /// Split-seed mode: luma seed JPEG quality (default: --seedq).
    #[arg(long)]
    pub seed_luma_q: Option<u8>,

    /// Split-seed mode: chroma seed size in pixels (default: tile_size).
    #[arg(long)]
    pub seed_chroma_size: Option<String>,

    /// Split-seed mode: chroma seed JPEG quality (default: --seedq).
    #[arg(long)]
    pub seed_chroma_q: Option<u8>,

    /// Enable wavelet denoising of residuals before encoding.
    /// Removes noise-like components from the residual, reducing encoded size.
    /// Synthesis parameters (16 bytes) are stored in the pack for decode-time noise recovery.
    #[arg(long)]
    pub denoise: bool,

    /// Sigma multiplier for wavelet denoising threshold (default: 0.25).
    /// Lower values preserve more detail; higher values remove more noise.
    #[arg(long, default_value_t = 0.25)]
    pub denoise_sigma: f32,

    /// Wavelet basis for denoising: db2, db4, db6, sym4, coif2 (default: db4).
    #[arg(long, default_value = "db4")]
    pub denoise_wavelet: String,

    /// Wavelet decomposition level for denoising (default: 2).
    #[arg(long, default_value_t = 2)]
    pub denoise_level: usize,

    /// Denoise weight (0.0-1.0, default: 1.0). Controls how much noise is removed from
    /// the residual. At 0.0 the residual is unchanged but synth params are still stored,
    /// enabling noise synthesis without denoising.
    #[arg(long, default_value_t = 1.0)]
    pub denoise_weight: f32,

    /// Noise synthesis strength for debug images/metrics (0.0-1.0, default: 0.8).
    /// Controls how much of the removed noise is synthesized back for evaluation.
    #[arg(long, default_value_t = 0.8)]
    pub synth_strength: f32,
}

/// Decode RGB image bytes (auto-detects JPEG or JXL by magic bytes).
/// Returns (rgb_data, width, height).
fn decode_seed_rgb(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    // JXL detection
    let is_jxl = (data.len() >= 2 && data[0] == 0xFF && data[1] == 0x0A)
        || (data.len() >= 12
            && data[..12]
                == [0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A]);
    if is_jxl {
        return decode_seed_rgb_jxl(data);
    }
    // Default: JPEG
    crate::turbojpeg_optimized::decode_rgb_turbo(data)
}

#[cfg(feature = "jpegxl")]
fn decode_seed_rgb_jxl(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let decoder = jpegxl_rs::decode::decoder_builder()
        .build()
        .map_err(|e| anyhow::anyhow!("JXL decoder build failed: {e:?}"))?;
    let (_meta, result) = decoder.reconstruct(data)
        .map_err(|e| anyhow::anyhow!("JXL reconstruct failed: {e:?}"))?;
    match result {
        jpegxl_rs::decode::Data::Jpeg(jpeg_bytes) => {
            crate::turbojpeg_optimized::decode_rgb_turbo(&jpeg_bytes)
        }
        jpegxl_rs::decode::Data::Pixels(pixels) => {
            let raw = match pixels {
                jpegxl_rs::decode::Pixels::Uint8(v) => v,
                _ => anyhow::bail!("JXL decoded to non-u8 pixel type"),
            };
            Ok((raw, _meta.width, _meta.height))
        }
    }
}

#[cfg(not(feature = "jpegxl"))]
fn decode_seed_rgb_jxl(_data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    anyhow::bail!("JXL-encoded seed detected but binary was built without --features jpegxl");
}

/// Parse --seed-size value into pixel dimension.
/// "l2" → tile_size, "l1" → tile_size*2, or a raw number.
fn parse_seed_size(val: Option<&str>, tile_size: u32) -> Result<u32> {
    match val {
        None | Some("l2") => Ok(tile_size),
        Some("l1") => Ok(tile_size * 2),
        Some(s) => {
            let n: u32 = s.parse()
                .map_err(|_| anyhow::anyhow!("--seed-size must be 'l2', 'l1', or a number, got '{}'", s))?;
            if n < 16 {
                anyhow::bail!("--seed-size ({}) must be >= 16", n);
            }
            if n > tile_size * 4 {
                anyhow::bail!("--seed-size ({}) must be <= tile_size*4 ({})", n, tile_size * 4);
            }
            Ok(n)
        }
    }
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

/// Returns true if any split-seed CLI args are provided.
fn is_split_seed_mode(args: &EncodeArgs) -> bool {
    args.seed_luma_size.is_some()
        || args.seed_luma_q.is_some()
        || args.seed_chroma_size.is_some()
        || args.seed_chroma_q.is_some()
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
/// V2/V3/V4 pipeline: no L1 residuals, single fused L0 residual per family.
fn run_pyramid(args: EncodeArgs, pyramid_path: PathBuf, subsamp: ChromaSubsampling) -> Result<()> {
    if is_split_seed_mode(&args) {
        anyhow::bail!("Split-seed mode (--seed-luma-size/--seed-chroma-size) is not yet supported for pyramid encoding. Use single-image mode (--image) instead.");
    }
    let start = Instant::now();
    let encoder = create_encoder(&args.encoder)?;
    let up_filter: ResampleFilter = args.upsample_filter.parse().unwrap();
    let down_filter: ResampleFilter = args.downsample_filter.parse().unwrap();

    let pyramid = discover_pyramid(&pyramid_path, args.tile)?;
    let seed_size = parse_seed_size(args.seed_size.as_deref(), pyramid.tile_size)?;
    info!("Using encoder: {} subsamp: {} upsample: {} downsample: {} seed_size: {}",
        encoder.name(), subsamp, up_filter, down_filter, seed_size);
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
    let seedq_str = args.seedq.clone();
    let seedq_is_lossless = matches!(seedq_str.to_lowercase().as_str(), "lossless" | "l");
    let seedq_num: u8 = if seedq_is_lossless { 95 } else {
        seedq_str.parse::<u8>().map_err(|_| anyhow::anyhow!(
            "--seedq must be a number (1-100) or 'lossless', got '{}'", seedq_str))?
    };
    let use_jxl_l2 = args.encoder == "jpegxl";
    let l2_jxl_mode: &str = if seedq_is_lossless { "lossless" } else { &seedq_str };
    let l2_format = if use_jxl_l2 { "jxl" } else { "jpg" };
    let ext = output_extension(encoder.name());
    let mut total_l0 = 0usize;
    let mut total_bytes = 0usize;
    let use_custom_seed = seed_size != tile_size;
    info!("Seed will be stored as {} (seedq={}, seed_size={})", l2_format.to_uppercase(), seedq_str, seed_size);

    for (pi, &(x2, y2)) in parents.iter().enumerate() {
        let parent_start = Instant::now();

        let l0_w = tile_size * 4;
        let l0_h = tile_size * 4;

        // Assemble L0 ground truth mosaic from pyramid tiles (always needed)
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

        let mut pack_entries: Vec<PackWriteEntry> = Vec::new();

        // Compute seed and prediction
        let (seed_stored, seed_enc_w, seed_enc_h, l0_pred_y) = if use_custom_seed {
            // Custom seed: assemble L0 mosaic as RGB, downscale to seed_size, encode as JPEG
            // We need RGB mosaic for the seed
            let mut l0_gt_rgb = vec![0u8; (l0_w * l0_h * 3) as usize];
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let x0 = x2 * 4 + dx;
                    let y0 = y2 * 4 + dy;
                    let l0_gt_path = pyramid
                        .files_dir
                        .join(pyramid.l0.to_string())
                        .join(format!("{}_{}.jpg", x0, y0));
                    if !l0_gt_path.exists() { continue; }
                    let (tile_rgb, gt_w, gt_h) = load_rgb_turbo(&l0_gt_path)?;
                    for ty in 0..gt_h.min(tile_size) {
                        for tx in 0..gt_w.min(tile_size) {
                            let mx = dx * tile_size + tx;
                            let my = dy * tile_size + ty;
                            if mx < l0_w && my < l0_h {
                                let dst = ((my * l0_w + mx) * 3) as usize;
                                let src = ((ty * gt_w + tx) * 3) as usize;
                                l0_gt_rgb[dst..dst + 3].copy_from_slice(&tile_rgb[src..src + 3]);
                            }
                        }
                    }
                }
            }

            // Downscale L0 RGB mosaic to seed_size
            let seed_rgb = {
                use image::{RgbImage, imageops};
                let img = RgbImage::from_raw(l0_w, l0_h, l0_gt_rgb)
                    .expect("failed to create L0 mosaic RgbImage");
                let resized = imageops::resize(&img, seed_size, seed_size, down_filter.to_image_filter());
                resized.into_raw()
            };

            // Encode seed using selected encoder at baseq
            let seed_encoded = if use_jxl_l2 {
                // JXL path: encode JPEG first (for roundtrip decode), then transcode/encode JXL for storage
                let seed_jpeg = encoder.encode_rgb_with_subsamp(&seed_rgb, seed_size, seed_size, seedq_num, subsamp)?;
                let seed_jxl = encode_l2_as_jxl(&seed_jpeg, &seed_rgb, seed_size, seed_size, l2_jxl_mode)
                    .unwrap_or_else(|_| seed_jpeg.clone());
                seed_jxl
            } else {
                encoder.encode_rgb_with_subsamp(&seed_rgb, seed_size, seed_size, seedq_num, subsamp)?
            };

            // Decode seed back (lossy roundtrip) — handles JPEG, JXL, WebP via magic detection
            let (seed_decoded_rgb, _, _) = decode_seed_rgb(&seed_encoded)?;
            let (seed_y, _, _) = ycbcr_planes_from_rgb(&seed_decoded_rgb, seed_size, seed_size);

            // Upsample seed Y → L0 prediction
            let l0_pred_y = fir_resize(&seed_y, seed_size, seed_size, l0_w, l0_h, up_filter);

            (seed_encoded, seed_size, seed_size, l0_pred_y)
        } else {
            // Default: use L2 tile from pyramid as seed
            let l2_path = pyramid
                .files_dir
                .join(pyramid.l2.to_string())
                .join(format!("{}_{}.jpg", x2, y2));
            let (l2_rgb, l2_w, l2_h) = load_rgb_turbo(&l2_path)
                .with_context(|| format!("loading L2 tile {}", l2_path.display()))?;

            let (l2_y, _, _) = ycbcr_planes_from_rgb(&l2_rgb, l2_w, l2_h);
            let l0_pred_y = upsample_4x(&l2_y, l2_w as usize, l2_h as usize, up_filter);

            // Store L2 as-is (or as JXL)
            let l2_jpeg_bytes = fs::read(&l2_path)?;
            let l2_stored = if use_jxl_l2 {
                encode_l2_as_jxl(&l2_jpeg_bytes, &l2_rgb, l2_w, l2_h, l2_jxl_mode)
                    .unwrap_or_else(|_| l2_jpeg_bytes.clone())
            } else {
                l2_jpeg_bytes
            };

            (l2_stored, l2_w, l2_h, l0_pred_y)
        };

        // Store seed in pack
        if args.pack {
            total_bytes += seed_stored.len();
            pack_entries.push(PackWriteEntry {
                level_kind: 2,
                idx_in_parent: 0,
                jpeg_data: seed_stored,
            });
        }

        // Compute fused L0 residual (entire mosaic)
        let raw_residual = compute_residual(&l0_gt_y, &l0_pred_y);
        let centered = center_residual(&raw_residual);

        // Optionally denoise residual before encoding
        let (centered, denoise_result) = if args.denoise {
            let basis: crate::core::wavelet::WaveletBasis = args.denoise_wavelet.parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            let result = crate::core::wavelet::denoise_residual(
                &centered, l0_w, l0_h, args.denoise_sigma, basis, args.denoise_level, args.denoise_weight,
            );
            (result.denoised.clone(), Some(result))
        } else {
            (centered, None)
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

            // Store wavelet synthesis params (16 bytes) for decode-time noise recovery
            if let Some(ref dr) = denoise_result {
                pack_entries.push(PackWriteEntry {
                    level_kind: 6,
                    idx_in_parent: 0,
                    jpeg_data: dr.synth_params.to_bytes().to_vec(),
                });
            }
        }

        // Write pack file — v3 if custom seed, v2 otherwise
        if let Some(ref pack_dir) = pack_dir {
            if use_custom_seed {
                let meta = PackMetadata {
                    version: 3,
                    tile_size: tile_size as u16,
                    seed_w: seed_enc_w as u16,
                    seed_h: seed_enc_h as u16,
                    residual_w: enc_w as u16,
                    residual_h: enc_h as u16,
                    seed_luma_w: 0,
                    seed_luma_h: 0,
                    seed_chroma_w: 0,
                    seed_chroma_h: 0,
                };
                write_pack_v3(pack_dir, x2, y2, &pack_entries, &meta)?;
            } else {
                write_pack(pack_dir, x2, y2, &pack_entries)?;
            }
        }

        let parent_ms = parent_start.elapsed().as_millis();
        info!(
            "[{}/{}] L2 parent ({},{}) — fused L0 residual ({} tiles, seed={}x{}) — {}ms",
            pi + 1, parents.len(), x2, y2, tiles_found, seed_enc_w, seed_enc_h, parent_ms
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
    let pipeline_version = if use_custom_seed { 3 } else { 2 };
    let summary = serde_json::json!({
        "encoder": args.encoder,
        "seedq": seedq_str,
        "l0q": l0q,
        "subsamp": subsamp.to_string(),
        "l0_scale": args.l0_scale,
        "l0_sharpen": args.l0_sharpen.map(|v| (v * 10.0).round() / 10.0),
        "tile_sharpen": args.tile_sharpen.map(|v| (v * 10.0).round() / 10.0),
        "upsample_filter": up_filter.to_string(),
        "downsample_filter": down_filter.to_string(),
        "seed_size": seed_size,
        "tile_size": tile_size,
        "l0_residuals": total_l0,
        "total_bytes": total_bytes,
        "parents": parents.len(),
        "pack": args.pack,
        "elapsed_secs": elapsed.as_secs_f64(),
        "pipeline_version": pipeline_version,
        "l2_format": l2_format,
    });
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;

    Ok(())
}

/// Single-image encode mode: creates one L2 family from a single image (for evals).
/// V2/V3 pipeline: seed baseline + single fused L0 residual (no L1 residuals).
fn run_single_image(args: EncodeArgs, subsamp: ChromaSubsampling) -> Result<()> {
    // Enable Huffman optimization for all turbojpeg calls (smaller files, ~20% for small images)
    std::env::set_var("TJ_OPTIMIZE", "1");

    let start = Instant::now();
    let image_path = args.image.as_ref().unwrap();
    let encoder = create_encoder(&args.encoder)?;
    let up_filter: ResampleFilter = args.upsample_filter.parse().unwrap();
    let down_filter: ResampleFilter = args.downsample_filter.parse().unwrap();
    let tile_size_for_seed = args.tile;
    let seed_size = parse_seed_size(args.seed_size.as_deref(), tile_size_for_seed)?;
    let use_custom_seed = seed_size != tile_size_for_seed;
    info!("Single-image mode: {} encoder={} subsamp={} upsample={} downsample={} seed_size={}",
        image_path.display(), encoder.name(), subsamp, up_filter, down_filter, seed_size);

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
    let seedq_str = args.seedq.clone();
    let seedq_is_lossless = matches!(seedq_str.to_lowercase().as_str(), "lossless" | "l");
    let seedq_num: u8 = if seedq_is_lossless { 95 } else {
        seedq_str.parse::<u8>().map_err(|_| anyhow::anyhow!(
            "--seedq must be a number (1-100) or 'lossless', got '{}'", seedq_str))?
    };
    let ext = output_extension(encoder.name());

    // Create output directory
    fs::create_dir_all(&args.out)?;

    // The source image is L0. Compute grid dimensions.
    let tiles_x = (src_w + tile_size - 1) / tile_size;
    let tiles_y = (src_h + tile_size - 1) / tile_size;
    info!("L0 grid: {}x{} tiles ({}x{})", tiles_x, tiles_y, tile_size, tile_size);

    // Use float YCbCr for prediction (avoids u8 quantization loss)
    use crate::core::color::{ycbcr_planes_from_rgb_f32, rgb_from_ycbcr_f32};
    use crate::core::residual::compute_residual_f32;

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

    // Setup debug image directories (shared by both paths)
    let compress_dir = args.out.join("compress");
    let decompress_dir = args.out.join("decompress");

    let split_seed = is_split_seed_mode(&args);

    // --- Seed encoding and prediction pipeline ---
    // In split-seed mode (v4): separate luma and chroma seeds at independent sizes/qualities.
    // In unified mode (v2/v3): single RGB seed.
    //
    // Both paths produce: l0_pred_y_f32, l0_pred_cb_f32, l0_pred_cr_f32, l2_bytes,
    //                      l2_decoded_rgb (for L1 chroma / debug), pack_entries, total_bytes

    let l0_pred_y_f32: Vec<f32>;
    let l0_pred_cb_f32: Vec<f32>;
    let l0_pred_cr_f32: Vec<f32>;
    let l2_bytes: usize;
    let l2_decoded_rgb: Vec<u8>;
    let l2_w: u32;
    let l2_h: u32;
    let mut total_bytes: usize;
    let mut manifest_tiles: Vec<serde_json::Value> = Vec::new();
    let mut pack_entries: Vec<PackWriteEntry> = Vec::new();
    let l2_format: &str;
    // Additional split-seed metadata for manifest
    let mut seed_luma_size_out: u32 = 0;
    let mut seed_luma_q_out: u8 = 0;
    let mut seed_chroma_size_out: u32 = 0;
    let mut seed_chroma_q_out: u8 = 0;
    let mut seed_luma_bytes_out: usize = 0;
    let mut seed_chroma_bytes_out: usize = 0;

    if split_seed {
        // ─── Split-seed mode (v4) ───
        // Luma and chroma seeds encoded separately at independent sizes/qualities.
        let luma_size = parse_seed_size(args.seed_luma_size.as_deref(), tile_size)?;
        let luma_q = args.seed_luma_q.unwrap_or(seedq_num);
        let chroma_size = parse_seed_size(args.seed_chroma_size.as_deref(), tile_size)?;
        let chroma_q = args.seed_chroma_q.unwrap_or(seedq_num);

        info!("Split-seed mode: luma={}x{} Q{}, chroma={}x{} Q{}",
            luma_size, luma_size, luma_q, chroma_size, chroma_size, chroma_q);

        // Downsample source to luma seed size, extract Y
        let luma_seed_rgb = {
            use image::{RgbImage, imageops};
            let src_img = RgbImage::from_raw(src_w, src_h, src_rgb.clone())
                .expect("failed to create RgbImage from source");
            let resized = imageops::resize(&src_img, luma_size, luma_size, down_filter.to_image_filter());
            resized.into_raw()
        };
        let (luma_seed_y, _, _) = ycbcr_planes_from_rgb(&luma_seed_rgb, luma_size, luma_size);

        // Encode luma seed as grayscale JPEG
        let luma_encoded = encoder.encode_gray(&luma_seed_y, luma_size, luma_size, luma_q)?;
        timings.push(("encode_seed_luma", t.elapsed().as_secs_f64())); t = Instant::now();
        fs::write(args.out.join("seed_luma.jpg"), &luma_encoded)?;
        info!("Seed luma: {}x{} → {} bytes (Q{})", luma_size, luma_size, luma_encoded.len(), luma_q);

        // Downsample source to chroma seed size, extract Cb/Cr, encode as separate grayscale images
        let chroma_seed_rgb = {
            use image::{RgbImage, imageops};
            let src_img = RgbImage::from_raw(src_w, src_h, src_rgb.clone())
                .expect("failed to create RgbImage from source");
            let resized = imageops::resize(&src_img, chroma_size, chroma_size, down_filter.to_image_filter());
            resized.into_raw()
        };
        let (_, chroma_cb_raw, chroma_cr_raw) = ycbcr_planes_from_rgb(&chroma_seed_rgb, chroma_size, chroma_size);
        let cb_encoded = encoder.encode_gray(&chroma_cb_raw, chroma_size, chroma_size, chroma_q)?;
        let cr_encoded = encoder.encode_gray(&chroma_cr_raw, chroma_size, chroma_size, chroma_q)?;
        let chroma_encoded_total = cb_encoded.len() + cr_encoded.len();
        timings.push(("encode_seed_chroma", t.elapsed().as_secs_f64())); t = Instant::now();
        fs::write(args.out.join("seed_cb.jpg"), &cb_encoded)?;
        fs::write(args.out.join("seed_cr.jpg"), &cr_encoded)?;
        info!("Seed chroma: {}x{} → Cb {} + Cr {} = {} bytes (Q{})", chroma_size, chroma_size, cb_encoded.len(), cr_encoded.len(), chroma_encoded_total, chroma_q);

        // Decode both back (lossy roundtrip) — auto-detects JPEG or JXL
        let luma_decoded = decode_residual_bytes(&luma_encoded)?;
        let chroma_cb = decode_residual_bytes(&cb_encoded)?;
        let chroma_cr = decode_residual_bytes(&cr_encoded)?;
        timings.push(("decode_seeds_roundtrip", t.elapsed().as_secs_f64())); t = Instant::now();

        // Build L2-equivalent decoded RGB (needed for L1 chroma and debug images)
        // Must happen before upsample which consumes chroma_cb/cr
        // Also save copies for debug images
        let debug_cb = if args.debug_images { Some(chroma_cb.clone()) } else { None };
        let debug_cr = if args.debug_images { Some(chroma_cr.clone()) } else { None };
        {
            use crate::core::color::rgb_from_ycbcr_planes;
            let luma_at_chroma_size = if luma_size == chroma_size {
                luma_decoded.clone()
            } else {
                use image::{GrayImage, imageops};
                let gray_img = GrayImage::from_raw(luma_size, luma_size, luma_decoded.clone())
                    .expect("failed to create GrayImage");
                let resized = imageops::resize(&gray_img, chroma_size, chroma_size, up_filter.to_image_filter());
                resized.into_raw()
            };
            l2_decoded_rgb = rgb_from_ycbcr_planes(&luma_at_chroma_size, &chroma_cb, &chroma_cr, chroma_size, chroma_size);
        }
        l2_w = chroma_size;
        l2_h = chroma_size;
        l2_format = "jpg";

        // Upsample luma Y → L0 Y prediction
        let l0_y_pred = {
            use image::{GrayImage, imageops};
            let gray_img = GrayImage::from_raw(luma_size, luma_size, luma_decoded.clone())
                .expect("failed to create GrayImage");
            let resized = imageops::resize(&gray_img, src_w, src_h, up_filter.to_image_filter());
            resized.into_raw().iter().map(|&v| v as f32).collect::<Vec<f32>>()
        };

        // Upsample chroma Cb/Cr → L0 chroma prediction
        let l0_cb_pred = {
            use image::{GrayImage, imageops};
            let gray_img = GrayImage::from_raw(chroma_size, chroma_size, chroma_cb)
                .expect("failed to create GrayImage for Cb");
            let resized = imageops::resize(&gray_img, src_w, src_h, up_filter.to_image_filter());
            resized.into_raw().iter().map(|&v| v as f32).collect::<Vec<f32>>()
        };
        let l0_cr_pred = {
            use image::{GrayImage, imageops};
            let gray_img = GrayImage::from_raw(chroma_size, chroma_size, chroma_cr)
                .expect("failed to create GrayImage for Cr");
            let resized = imageops::resize(&gray_img, src_w, src_h, up_filter.to_image_filter());
            resized.into_raw().iter().map(|&v| v as f32).collect::<Vec<f32>>()
        };
        timings.push(("upsample_split_seeds_to_l0", t.elapsed().as_secs_f64())); t = Instant::now();

        l0_pred_y_f32 = l0_y_pred;
        l0_pred_cb_f32 = l0_cb_pred;
        l0_pred_cr_f32 = l0_cr_pred;
        l2_bytes = luma_encoded.len() + chroma_encoded_total;
        total_bytes = l2_bytes;

        // Store metadata for manifest
        seed_luma_size_out = luma_size;
        seed_luma_q_out = luma_q;
        seed_chroma_size_out = chroma_size;
        seed_chroma_q_out = chroma_q;
        seed_luma_bytes_out = luma_encoded.len();
        seed_chroma_bytes_out = chroma_encoded_total;

        // Pack entries
        if args.pack {
            pack_entries.push(PackWriteEntry {
                level_kind: 3,  // seed luma (Y)
                idx_in_parent: 0,
                jpeg_data: luma_encoded,
            });
            pack_entries.push(PackWriteEntry {
                level_kind: 4,  // seed Cb
                idx_in_parent: 0,
                jpeg_data: cb_encoded,
            });
            pack_entries.push(PackWriteEntry {
                level_kind: 5,  // seed Cr
                idx_in_parent: 0,
                jpeg_data: cr_encoded,
            });
        }

        if args.debug_images {
            fs::create_dir_all(&compress_dir)?;
            fs::create_dir_all(&decompress_dir)?;
            save_gray_png(&compress_dir.join("001_seed_luma.png"), &luma_decoded, luma_size, luma_size)?;
            save_gray_png(&compress_dir.join("002_seed_cb.png"), debug_cb.as_ref().unwrap(), chroma_size, chroma_size)?;
            save_gray_png(&compress_dir.join("003_seed_cr.png"), debug_cr.as_ref().unwrap(), chroma_size, chroma_size)?;
        }
    } else {
        // ─── Unified seed mode (v2/v3) ───
        // Build seed by downsampling from L0 source to seed_size
        // When seed_size == tile_size this is equivalent to L2 (4:1 downscale)
        let l2_w_local = seed_size;
        let l2_h_local = seed_size;
        let l2_rgb = {
            use image::{RgbImage, imageops};
            let src_img = RgbImage::from_raw(src_w, src_h, src_rgb.clone())
                .expect("failed to create RgbImage from source");
            let resized = imageops::resize(&src_img, l2_w_local, l2_h_local, down_filter.to_image_filter());
            resized.into_raw()
        };
        timings.push(("downsample_l0_to_seed", t.elapsed().as_secs_f64())); t = Instant::now();

        // L2 pipeline order: OptL2 first (on smooth base), then sharpen (on optimized base).
        let mut l2_to_encode = l2_rgb.clone();

        // Step 1: Optionally optimize L2 for better predictions (gradient descent on smooth base)
        if args.optl2_luma {
            use crate::core::optimize_l2::optimize_l2_luma_only;
            info!("Running OptL2 luma-only gradient descent optimization...");
            l2_to_encode = optimize_l2_luma_only(&l2_to_encode, &l1_rgb, l2_w_local, l2_h_local, l1_w, l1_h, args.max_delta, 100, 0.3, up_filter);
        } else if args.optl2 {
            use crate::core::optimize_l2::optimize_l2_for_prediction;
            info!("Running OptL2 gradient descent optimization...");
            l2_to_encode = optimize_l2_for_prediction(&l2_to_encode, &l1_rgb, l2_w_local, l2_h_local, l1_w, l1_h, args.max_delta, 100, 0.3, up_filter);
        }
        timings.push(("optl2", t.elapsed().as_secs_f64())); t = Instant::now();

        // Step 2: Optionally sharpen L2
        let l2_sharpened = if let Some(strength) = args.sharpen {
            use crate::core::sharpen::unsharp_mask_rgb;
            info!("Applying unsharp mask (strength={:.2}) to L2...", strength);
            Some(unsharp_mask_rgb(&l2_to_encode, l2_w_local, l2_h_local, strength))
        } else {
            None
        };
        timings.push(("sharpen_l2", t.elapsed().as_secs_f64())); t = Instant::now();

        if args.save_sharpened {
            if let Some(ref sharpened) = l2_sharpened {
                l2_to_encode = sharpened.clone();
            }
        }

        // Encode seed (L2) baseline using the selected encoder.
        let use_jxl_l2 = args.encoder == "jpegxl";
        let l2_encoded = encoder.encode_rgb_with_subsamp(&l2_to_encode, l2_w_local, l2_h_local, seedq_num, subsamp)?;
        timings.push(("encode_seed", t.elapsed().as_secs_f64())); t = Instant::now();

        let (l2_stored, l2_format_local) = if use_jxl_l2 {
            let l2_jxl_mode = if seedq_is_lossless { "lossless" } else { &seedq_str };
            let l2_jxl_data = encode_l2_as_jxl(&l2_encoded, &l2_to_encode, l2_w_local, l2_h_local, l2_jxl_mode)?;
            timings.push(("transcode_seed_jxl", t.elapsed().as_secs_f64())); t = Instant::now();
            fs::write(args.out.join("L2_0_0.jxl"), &l2_jxl_data)?;
            fs::write(args.out.join("L2_0_0.jpg"), &l2_encoded)?;
            info!("Seed: {}x{} → {} bytes as JXL ({} mode, JPEG was {} bytes, Q{} {})",
                l2_w_local, l2_h_local, l2_jxl_data.len(), l2_jxl_mode, l2_encoded.len(), seedq_str, subsamp);
            (l2_jxl_data, "jxl")
        } else {
            timings.push(("transcode_seed_jxl", 0.0)); t = Instant::now();
            fs::write(args.out.join(format!("L2_0_0{}", ext)), &l2_encoded)?;
            info!("Seed: {}x{} → {} bytes (Q{} {})",
                l2_w_local, l2_h_local, l2_encoded.len(), seedq_str, subsamp);
            (l2_encoded.clone(), "jpg")
        };
        let l2_bytes_local = l2_stored.len();

        // Decode seed back (lossy round-trip)
        let (l2_decoded_rgb_local, _, _) = decode_seed_rgb(&l2_stored)?;
        timings.push(("decode_seed_roundtrip", t.elapsed().as_secs_f64())); t = Instant::now();

        // If --sharpen without --save-sharpened: sharpen the decoded L2 for predictions.
        let l2_decoded_for_pred = if l2_sharpened.is_some() && !args.save_sharpened {
            use crate::core::sharpen::unsharp_mask_rgb;
            let strength = args.sharpen.unwrap();
            info!("Sharpen-decode: applying unsharp mask (strength={:.2}) to decoded L2 for predictions...", strength);
            unsharp_mask_rgb(&l2_decoded_rgb_local, l2_w_local, l2_h_local, strength)
        } else {
            l2_decoded_rgb_local.clone()
        };
        timings.push(("sharpen_decode_l2", t.elapsed().as_secs_f64())); t = Instant::now();

        let l2_for_prediction = l2_decoded_for_pred;

        if args.debug_images {
            fs::create_dir_all(&compress_dir)?;
            fs::create_dir_all(&decompress_dir)?;
            let (l2_y, l2_cb, l2_cr) = ycbcr_planes_from_rgb(&l2_decoded_rgb_local, l2_w_local, l2_h_local);
            save_rgb_png(&compress_dir.join("001_L2_original.png"), &l2_to_encode, l2_w_local, l2_h_local)?;
            save_gray_png(&compress_dir.join("002_L2_luma.png"), &l2_y, l2_w_local, l2_h_local)?;
            save_gray_png(&compress_dir.join("003_L2_chroma_cb.png"), &l2_cb, l2_w_local, l2_h_local)?;
            save_gray_png(&compress_dir.join("004_L2_chroma_cr.png"), &l2_cr, l2_w_local, l2_h_local)?;
            save_rgb_png(&decompress_dir.join("050_L2_decode.png"), &l2_decoded_rgb_local, l2_w_local, l2_h_local)?;
            save_rgb_png(&decompress_dir.join("050_L2_for_prediction.png"), &l2_for_prediction, l2_w_local, l2_h_local)?;
        }

        // Upsample L2 → L0 prediction directly (4x, skip L1 intermediate)
        // If SR model is available, use learned super-resolution instead of filter upsample.
        #[cfg(feature = "sr-model")]
        let l0_pred_rgb = if let Some(ref sr_path) = args.sr_model {
            let sr = crate::core::sr_model::SRModel::load(sr_path, args.sr_threads, 1)
                .with_context(|| format!("loading SR model from {}", sr_path))?;
            let (sr_rgb, _sr_w, _sr_h) = sr.infer_rgb(&l2_for_prediction, l2_w_local, l2_h_local)
                .with_context(|| "SR model inference failed")?;
            info!("SR model prediction: {}x{} → {}x{}", l2_w_local, l2_h_local, _sr_w, _sr_h);
            timings.push(("sr_model_inference", t.elapsed().as_secs_f64())); t = Instant::now();
            sr_rgb
        } else {
            let rgb = {
                use image::{RgbImage, imageops};
                let l2_img = RgbImage::from_raw(l2_w_local, l2_h_local, l2_for_prediction)
                    .expect("failed to create RgbImage from L2 reconstructed");
                let resized = imageops::resize(&l2_img, src_w, src_h, up_filter.to_image_filter());
                resized.into_raw()
            };
            timings.push(("upsample_l2_to_l0_4x", t.elapsed().as_secs_f64())); t = Instant::now();
            rgb
        };
        #[cfg(not(feature = "sr-model"))]
        let l0_pred_rgb = {
            if args.sr_model.is_some() {
                anyhow::bail!("--sr-model requires building with --features sr-model");
            }
            let rgb = {
                use image::{RgbImage, imageops};
                let l2_img = RgbImage::from_raw(l2_w_local, l2_h_local, l2_for_prediction)
                    .expect("failed to create RgbImage from L2 reconstructed");
                let resized = imageops::resize(&l2_img, src_w, src_h, up_filter.to_image_filter());
                resized.into_raw()
            };
            timings.push(("upsample_l2_to_l0_4x", t.elapsed().as_secs_f64())); t = Instant::now();
            rgb
        };

        // Apply refine model to improve prediction (per-tile, 256x256)
        #[cfg(feature = "sr-model")]
        let l0_pred_rgb = if let Some(ref refine_path) = args.refine_model {
            let refine = crate::core::sr_model::SRModel::load(refine_path, args.refine_threads, args.refine_pool)
                .with_context(|| format!("loading refine model from {}", refine_path))?;
            let mut rgb = l0_pred_rgb;
            let ts = tile_size;
            for ty in 0..tiles_y {
                for tx in 0..tiles_x {
                    // Extract tile
                    let mut tile = vec![0u8; (ts * ts * 3) as usize];
                    for y in 0..ts {
                        for x in 0..ts {
                            let mx = (tx * ts + x) as usize;
                            let my = (ty * ts + y) as usize;
                            if (mx as u32) < src_w && (my as u32) < src_h {
                                let si = (my * src_w as usize + mx) * 3;
                                let di = ((y * ts + x) * 3) as usize;
                                tile[di..di+3].copy_from_slice(&rgb[si..si+3]);
                            }
                        }
                    }
                    // Refine tile in place
                    refine.refine_rgb_inplace(&mut tile, ts, ts)
                        .with_context(|| format!("refine model failed for tile ({},{})", tx, ty))?;
                    // Write back
                    for y in 0..ts {
                        for x in 0..ts {
                            let mx = (tx * ts + x) as usize;
                            let my = (ty * ts + y) as usize;
                            if (mx as u32) < src_w && (my as u32) < src_h {
                                let si = (my * src_w as usize + mx) * 3;
                                let di = ((y * ts + x) * 3) as usize;
                                rgb[si..si+3].copy_from_slice(&tile[di..di+3]);
                            }
                        }
                    }
                }
            }
            info!("Refine model applied to {}x{} tiles ({}x{})", tiles_x, tiles_y, ts, ts);
            timings.push(("refine_model", t.elapsed().as_secs_f64())); t = Instant::now();
            rgb
        } else {
            l0_pred_rgb
        };
        #[cfg(not(feature = "sr-model"))]
        if args.refine_model.is_some() {
            anyhow::bail!("--refine-model requires building with --features sr-model");
        }

        let (y_f32, cb_f32, cr_f32) = ycbcr_planes_from_rgb_f32(&l0_pred_rgb, src_w, src_h);
        timings.push(("ycbcr_convert_prediction", t.elapsed().as_secs_f64())); t = Instant::now();

        if args.debug_images {
            save_rgb_png(&decompress_dir.join("060_L0_mosaic_prediction.png"), &l0_pred_rgb, src_w, src_h)?;
        }

        l0_pred_y_f32 = y_f32;
        l0_pred_cb_f32 = cb_f32;
        l0_pred_cr_f32 = cr_f32;
        l2_bytes = l2_bytes_local;
        total_bytes = l2_bytes;
        l2_decoded_rgb = l2_decoded_rgb_local;
        l2_w = l2_w_local;
        l2_h = l2_h_local;
        l2_format = l2_format_local;

        // Pack entries
        if args.pack {
            pack_entries.push(PackWriteEntry {
                level_kind: 2,
                idx_in_parent: 0,
                jpeg_data: l2_stored.clone(),
            });
        }
    }

    // Reconstruct L0 prediction RGB from f32 planes (needed for debug images)
    let l0_pred_rgb = if args.debug_images {
        rgb_from_ycbcr_f32(&l0_pred_y_f32, &l0_pred_cb_f32, &l0_pred_cr_f32)
    } else {
        Vec::new()
    };

    // Compute fused L0 residual (full source image size)
    // Extract ground truth Y from source
    let (src_gt_y, _, _) = ycbcr_planes_from_rgb(&src_rgb, src_w, src_h);
    timings.push(("ycbcr_convert_source", t.elapsed().as_secs_f64())); t = Instant::now();

    // Float-precision residual for the entire L0 mosaic
    let centered = compute_residual_f32(&src_gt_y, &l0_pred_y_f32);

    // Optionally denoise residual before encoding
    let (centered, denoise_result) = if args.denoise {
        let basis: crate::core::wavelet::WaveletBasis = args.denoise_wavelet.parse()
            .map_err(|e: String| anyhow::anyhow!(e))?;
        let result = crate::core::wavelet::denoise_residual(
            &centered, src_w, src_h, args.denoise_sigma, basis, args.denoise_level, args.denoise_weight,
        );
        info!(
            "Denoised residual: basis={}, sigma={}, level={}, weight={}, approx_sigma={:.3}",
            basis, args.denoise_sigma, args.denoise_level, args.denoise_weight, result.synth_params.approx_sigma,
        );
        (result.denoised.clone(), Some(result))
    } else {
        (centered, None)
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

        // Store wavelet synthesis params (16 bytes) for decode-time noise recovery
        if let Some(ref dr) = denoise_result {
            pack_entries.push(PackWriteEntry {
                level_kind: 6,
                idx_in_parent: 0,
                jpeg_data: dr.synth_params.to_bytes().to_vec(),
            });
        }
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

        // Debug images for wavelet denoising
        if let Some(ref dr) = denoise_result {
            save_gray_png(&compress_dir.join("022_L0_denoised_residual.png"),
                &dr.denoised, src_w, src_h)?;
            save_gray_png(&compress_dir.join("023_L0_removed_noise.png"),
                &dr.removed_noise, src_w, src_h)?;
        }
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

        // Optionally sharpen decoded residual (decode-time sharpening)
        let decoded_res_full = if let Some(strength) = args.l0_sharpen {
            crate::core::sharpen::unsharp_mask_gray(&decoded_res_full, src_w, src_h, strength)
        } else {
            decoded_res_full
        };

        // Save sharpened residual debug image
        if args.debug_images && args.l0_sharpen.is_some() {
            save_gray_png(&decompress_dir.join("025_L0_residual_sharpened.png"),
                &decoded_res_full, src_w, src_h)?;
        }

        // Reconstruct full L0 Y
        let mut recon_y = vec![0.0f32; (src_w * src_h) as usize];
        for i in 0..(src_w * src_h) as usize {
            let pred_val = l0_pred_y_f32[i];
            let res_val = decoded_res_full[i] as f32 - 128.0;
            recon_y[i] = (pred_val + res_val).clamp(0.0, 255.0);
        }

        // Apply tile sharpen to reconstructed Y (after residual, before noise)
        if let Some(strength) = args.tile_sharpen {
            let recon_u8: Vec<u8> = recon_y.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
            let sharpened = crate::core::sharpen::unsharp_mask_gray(&recon_u8, src_w, src_h, strength);
            for i in 0..recon_y.len() {
                recon_y[i] = sharpened[i] as f32;
            }
        }

        // Synthesize noise back into reconstructed Y for realistic metrics/debug images
        if let Some(ref dr) = denoise_result {
            let mut recon_y_u8: Vec<u8> = recon_y.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
            crate::core::wavelet::synthesize_and_apply_noise(
                &mut recon_y_u8, src_w, src_h, &dr.synth_params, args.synth_strength,
            );
            for i in 0..recon_y.len() {
                recon_y[i] = recon_y_u8[i] as f32;
            }
            info!("Applied noise synthesis to reconstructed Y for metrics/debug images");
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
        let pack_dir_path = args.out.join("packs");
        fs::create_dir_all(&pack_dir_path)?;
        if split_seed {
            let meta = PackMetadata {
                version: 4,
                tile_size: tile_size as u16,
                seed_w: 0,
                seed_h: 0,
                seed_luma_w: seed_luma_size_out as u16,
                seed_luma_h: seed_luma_size_out as u16,
                seed_chroma_w: seed_chroma_size_out as u16,
                seed_chroma_h: seed_chroma_size_out as u16,
                residual_w: enc_w as u16,
                residual_h: enc_h as u16,
            };
            write_pack_v4(&pack_dir_path, 0, 0, &pack_entries, &meta)?;
            info!("Pack file written: {} entries (seed_luma=1, seed_cb=1, seed_cr=1, L0=1, v4)", pack_entries.len());
        } else if use_custom_seed {
            let meta = PackMetadata {
                version: 3,
                tile_size: tile_size as u16,
                seed_w: seed_size as u16,
                seed_h: seed_size as u16,
                residual_w: enc_w as u16,
                residual_h: enc_h as u16,
                seed_luma_w: 0,
                seed_luma_h: 0,
                seed_chroma_w: 0,
                seed_chroma_h: 0,
            };
            write_pack_v3(&pack_dir_path, 0, 0, &pack_entries, &meta)?;
            info!("Pack file written: {} entries (seed=1, L0=1, v3)", pack_entries.len());
        } else {
            write_pack(&pack_dir_path, 0, 0, &pack_entries)?;
            info!("Pack file written: {} entries (seed=1, L0=1, v2)", pack_entries.len());
        }
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
    let pipeline_version = if split_seed { 4 } else if use_custom_seed { 3 } else { 2 };
    let mut summary = serde_json::json!({
        "mode": "single-image",
        "source": image_path.to_string_lossy(),
        "source_w": src_w,
        "source_h": src_h,
        "encoder": args.encoder,
        "subsamp": subsamp.to_string(),
        "seedq": seedq_str,
        "l0q": l0q,
        "optl2": args.optl2 || args.optl2_luma,
        "optl2_luma": args.optl2_luma,
        "sharpen": args.sharpen.map(|v| (v * 10.0).round() / 10.0),
        "save_sharpened": args.save_sharpened,
        "l0_scale": args.l0_scale,
        "l0_sharpen": args.l0_sharpen.map(|v| (v * 10.0).round() / 10.0),
        "tile_sharpen": args.tile_sharpen.map(|v| (v * 10.0).round() / 10.0),
        "upsample_filter": up_filter.to_string(),
        "downsample_filter": down_filter.to_string(),
        "seed_size": seed_size,
        "tile_size": tile_size,
        "l2_bytes": l2_bytes,
        "l2_w": l2_w,
        "l2_h": l2_h,
        "l0_tiles": tiles_x * tiles_y,
        "total_bytes": total_bytes,
        "elapsed_secs": elapsed.as_secs_f64(),
        "pipeline_version": pipeline_version,
        "timing_breakdown_ms": timings.iter().map(|(k, v)| (k.to_string(), (v * 1000.0 * 100.0).round() / 100.0)).collect::<std::collections::HashMap<_, _>>(),
    });
    if split_seed {
        summary["seed_luma_size"] = serde_json::json!(seed_luma_size_out);
        summary["seed_luma_q"] = serde_json::json!(seed_luma_q_out);
        summary["seed_chroma_size"] = serde_json::json!(seed_chroma_size_out);
        summary["seed_chroma_q"] = serde_json::json!(seed_chroma_q_out);
        summary["seed_luma_bytes"] = serde_json::json!(seed_luma_bytes_out);
        summary["seed_chroma_bytes"] = serde_json::json!(seed_chroma_bytes_out);
    }
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;

    // Write manifest.json (detailed per-tile metrics for viewer)
    if args.manifest {
        let mut manifest = serde_json::json!({
            "mode": "single-image",
            "source": image_path.to_string_lossy(),
            "source_w": src_w,
            "source_h": src_h,
            "encoder": args.encoder,
            "subsamp": subsamp.to_string(),
            "seedq": seedq_str,
            "l0q": l0q,
            "optl2": args.optl2 || args.optl2_luma,
        "optl2_luma": args.optl2_luma,
            "sharpen": args.sharpen.map(|v| (v * 10.0).round() / 10.0),
            "l0_scale": args.l0_scale,
            "l0_sharpen": args.l0_sharpen.map(|v| (v * 10.0).round() / 10.0),
        "tile_sharpen": args.tile_sharpen.map(|v| (v * 10.0).round() / 10.0),
            "l2_format": l2_format,
            "upsample_filter": up_filter.to_string(),
            "downsample_filter": down_filter.to_string(),
            "seed_size": seed_size,
            "tile_size": tile_size,
            "l2_bytes": l2_bytes,
            "l2_w": l2_w,
            "l2_h": l2_h,
            "fused_l0_bytes": res_data.len(),
            "total_bytes": total_bytes,
            "tiles": manifest_tiles,
            "pipeline_version": pipeline_version,
        });
        if split_seed {
            manifest["seed_luma_size"] = serde_json::json!(seed_luma_size_out);
            manifest["seed_luma_q"] = serde_json::json!(seed_luma_q_out);
            manifest["seed_chroma_size"] = serde_json::json!(seed_chroma_size_out);
            manifest["seed_chroma_q"] = serde_json::json!(seed_chroma_q_out);
            manifest["seed_luma_bytes"] = serde_json::json!(seed_luma_bytes_out);
            manifest["seed_chroma_bytes"] = serde_json::json!(seed_chroma_bytes_out);
        }
        fs::write(
            args.out.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;
    }

    Ok(())
}
