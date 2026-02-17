use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use tracing::info;

use crate::core::color::ycbcr_planes_from_rgb;
use crate::core::jpeg::{create_encoder, ChromaSubsampling};
use crate::core::pack::{write_pack, PackWriteEntry};
use crate::core::pyramid::{discover_pyramid, extract_tile_plane, parse_tile_coords};
use crate::core::residual::{center_residual, compute_residual};
use crate::core::upsample::upsample_2x_channel;
use crate::turbojpeg_optimized::load_rgb_turbo;

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

    /// JPEG quality for residual encoding (1-100)
    #[arg(long, default_value_t = 50)]
    pub resq: u8,

    /// Override quality for L1 residuals (default: use --resq)
    #[arg(long)]
    pub l1q: Option<u8>,

    /// Override quality for L0 residuals (default: use --resq)
    #[arg(long)]
    pub l0q: Option<u8>,

    /// JPEG quality for L2 baseline encoding (1-100, single-image mode only)
    #[arg(long, default_value_t = 95)]
    pub baseq: u8,

    /// Chroma subsampling: 444, 420, 420opt
    #[arg(long, default_value = "444")]
    pub subsamp: String,

    /// Encoder backend: turbojpeg, mozjpeg, jpegli, jpegxl, webp
    #[arg(long, default_value = "turbojpeg")]
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
}

/// Return the file extension for the given encoder name.
fn output_extension(encoder_name: &str) -> &'static str {
    match encoder_name {
        "webp" => ".webp",
        "jpegxl" => ".jxl",
        _ => ".jpg",
    }
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

    if args.image.is_some() {
        return run_single_image(args, subsamp);
    }

    let pyramid_path = args.pyramid.clone()
        .ok_or_else(|| anyhow::anyhow!("either --pyramid or --image must be specified"))?;

    run_pyramid(args, pyramid_path, subsamp)
}

/// Encode residuals from a pre-built DZI pyramid (original mode).
fn run_pyramid(args: EncodeArgs, pyramid_path: PathBuf, subsamp: ChromaSubsampling) -> Result<()> {
    let start = Instant::now();
    let encoder = create_encoder(&args.encoder)?;
    info!("Using encoder: {} subsamp: {}", encoder.name(), subsamp);

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
    let l1_out = residuals_dir.join("L1");
    let l0_out = residuals_dir.join("L0");
    fs::create_dir_all(&l1_out)?;
    fs::create_dir_all(&l0_out)?;

    let pack_dir = if args.pack {
        let d = args.out.join("packs");
        fs::create_dir_all(&d)?;
        Some(d)
    } else {
        None
    };

    let tile_size = pyramid.tile_size;
    let l1q = args.l1q.unwrap_or(args.resq);
    let l0q = args.l0q.unwrap_or(args.resq);
    let ext = output_extension(encoder.name());
    let mut total_l1 = 0usize;
    let mut total_l0 = 0usize;
    let mut total_bytes = 0usize;

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
        let (l2_y, l2_cb, l2_cr) = ycbcr_planes_from_rgb(&l2_rgb, l2_w, l2_h);

        // Upsample L2 → L1 prediction (2x bilinear for all channels)
        let l1_y = upsample_2x_channel(&l2_y, l2_w as usize, l2_h as usize);
        let l1_cb = upsample_2x_channel(&l2_cb, l2_w as usize, l2_h as usize);
        let l1_cr = upsample_2x_channel(&l2_cr, l2_w as usize, l2_h as usize);
        let l1_w = l2_w * 2;
        let l1_h = l2_h * 2;

        let mut pack_entries: Vec<PackWriteEntry> = Vec::new();

        // L1 parent output dir
        let l1_parent_dir = l1_out.join(format!("{}_{}", x2, y2));
        fs::create_dir_all(&l1_parent_dir)?;

        // Reconstructed L1 Y planes for building mosaic
        let mut l1_recon_y = vec![0u8; (l1_w * l1_h) as usize];

        // Process L1 children (2x2)
        for dy in 0..2u32 {
            for dx in 0..2u32 {
                let x1 = x2 * 2 + dx;
                let y1 = y2 * 2 + dy;

                // Load L1 ground truth
                let l1_gt_path = pyramid
                    .files_dir
                    .join(pyramid.l1.to_string())
                    .join(format!("{}_{}.jpg", x1, y1));
                if !l1_gt_path.exists() {
                    continue;
                }
                let (l1_gt_rgb, gt_w, gt_h) = load_rgb_turbo(&l1_gt_path)?;
                let (l1_gt_y, _, _) = ycbcr_planes_from_rgb(&l1_gt_rgb, gt_w, gt_h);


                // Extract L1 prediction Y for this tile region
                let mut l1_pred_y_tile = vec![0u8; (tile_size * tile_size) as usize];
                extract_tile_plane(
                    &l1_y, l1_w, l1_h,
                    dx * tile_size, dy * tile_size,
                    tile_size, tile_size,
                    &mut l1_pred_y_tile,
                );

                // Compute and center residual
                let raw_residual = compute_residual(&l1_gt_y, &l1_pred_y_tile);
                let centered = center_residual(&raw_residual);

                // Encode as grayscale JPEG
                let jpeg_data = encoder.encode_gray(&centered, tile_size, tile_size, l1q)?;
                total_l1 += 1;
                total_bytes += jpeg_data.len();

                // Write residual file
                let out_path = l1_parent_dir.join(format!("{}_{}{}", x1, y1, ext));
                fs::write(&out_path, &jpeg_data)?;

                // Decode the residual back to get the actual reconstructed Y
                // (accounts for JPEG lossy compression)
                let decoded_residual = decode_residual_bytes(&jpeg_data)?;

                // Reconstruct Y: Y_recon = clamp(Y_pred + (decoded_residual - 128), 0..255)
                for y in 0..tile_size {
                    for x in 0..tile_size {
                        let tile_idx = (y * tile_size + x) as usize;
                        let pred_val = l1_pred_y_tile[tile_idx] as i16;
                        let res_val = decoded_residual[tile_idx] as i16 - 128;
                        let recon = (pred_val + res_val).clamp(0, 255) as u8;

                        let mosaic_x = dx * tile_size + x;
                        let mosaic_y = dy * tile_size + y;
                        if mosaic_x < l1_w && mosaic_y < l1_h {
                            l1_recon_y[(mosaic_y * l1_w + mosaic_x) as usize] = recon;
                        }
                    }
                }

                if args.pack {
                    pack_entries.push(PackWriteEntry {
                        level_kind: 1,
                        idx_in_parent: (dy * 2 + dx) as u8,
                        jpeg_data,
                    });
                }
            }
        }

        // Build L1 mosaic → upsample 2x → L0 prediction
        // Use the reconstructed Y plus the bilinear-upsampled chroma
        let l0_y = upsample_2x_channel(&l1_recon_y, l1_w as usize, l1_h as usize);
        let _l0_cb = upsample_2x_channel(&l1_cb, l1_w as usize, l1_h as usize);
        let _l0_cr = upsample_2x_channel(&l1_cr, l1_w as usize, l1_h as usize);
        let l0_w = l1_w * 2;
        let l0_h = l1_h * 2;

        // L0 parent output dir
        let l0_parent_dir = l0_out.join(format!("{}_{}", x2, y2));
        fs::create_dir_all(&l0_parent_dir)?;

        // Process L0 children (4x4)
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
                let (l0_gt_rgb, gt_w, gt_h) = load_rgb_turbo(&l0_gt_path)?;
                let (l0_gt_y, _, _) = ycbcr_planes_from_rgb(&l0_gt_rgb, gt_w, gt_h);


                // Extract L0 prediction Y for this tile
                let mut l0_pred_y_tile = vec![0u8; (tile_size * tile_size) as usize];
                extract_tile_plane(
                    &l0_y, l0_w, l0_h,
                    dx * tile_size, dy * tile_size,
                    tile_size, tile_size,
                    &mut l0_pred_y_tile,
                );

                let raw_residual = compute_residual(&l0_gt_y, &l0_pred_y_tile);
                let centered = center_residual(&raw_residual);

                let jpeg_data = encoder.encode_gray(&centered, tile_size, tile_size, l0q)?;
                total_l0 += 1;
                total_bytes += jpeg_data.len();

                let out_path = l0_parent_dir.join(format!("{}_{}{}", x0, y0, ext));
                fs::write(&out_path, &jpeg_data)?;

                if args.pack {
                    pack_entries.push(PackWriteEntry {
                        level_kind: 0,
                        idx_in_parent: (dy * 4 + dx) as u8,
                        jpeg_data,
                    });
                }
            }
        }

        // Write pack file
        if let Some(ref pack_dir) = pack_dir {
            write_pack(pack_dir, x2, y2, &pack_entries)?;
        }

        let parent_ms = parent_start.elapsed().as_millis();
        info!(
            "[{}/{}] L2 parent ({},{}) — L1: {} residuals, L0: {} residuals — {}ms",
            pi + 1,
            parents.len(),
            x2,
            y2,
            pack_entries.iter().filter(|e| e.level_kind == 1).count(),
            pack_entries.iter().filter(|e| e.level_kind == 0).count(),
            parent_ms
        );
    }

    let elapsed = start.elapsed();
    info!(
        "Encode complete: {} L1 + {} L0 residuals, {:.1} MB total, {:.1}s",
        total_l1,
        total_l0,
        total_bytes as f64 / 1_048_576.0,
        elapsed.as_secs_f64()
    );

    // Write summary.json
    let summary = serde_json::json!({
        "encoder": args.encoder,
        "l1q": l1q,
        "l0q": l0q,
        "subsamp": subsamp.to_string(),
        "tile_size": tile_size,
        "l1_residuals": total_l1,
        "l0_residuals": total_l0,
        "total_bytes": total_bytes,
        "parents": parents.len(),
        "pack": args.pack,
        "elapsed_secs": elapsed.as_secs_f64(),
    });
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;

    Ok(())
}

/// Single-image encode mode: creates one L2 family from a single image (for evals).
fn run_single_image(args: EncodeArgs, subsamp: ChromaSubsampling) -> Result<()> {
    // Enable Huffman optimization for all turbojpeg calls (smaller files, ~20% for small images)
    std::env::set_var("TJ_OPTIMIZE", "1");

    let start = Instant::now();
    let image_path = args.image.as_ref().unwrap();
    let encoder = create_encoder(&args.encoder)?;
    info!("Single-image mode: {} encoder={} subsamp={}", image_path.display(), encoder.name(), subsamp);

    // Load the source image
    let (src_rgb, src_w, src_h) = load_rgb_turbo(image_path)
        .with_context(|| format!("loading source image {}", image_path.display()))?;
    info!("Source image: {}x{}", src_w, src_h);

    let tile_size = args.tile;
    let l1q = args.l1q.unwrap_or(args.resq);
    let l0q = args.l0q.unwrap_or(args.resq);
    let baseq = args.baseq;
    let ext = output_extension(encoder.name());

    // Create output directory
    fs::create_dir_all(&args.out)?;

    // The source image is L0. Compute grid dimensions.
    let tiles_x = (src_w + tile_size - 1) / tile_size;
    let tiles_y = (src_h + tile_size - 1) / tile_size;
    info!("L0 grid: {}x{} tiles ({}x{})", tiles_x, tiles_y, tile_size, tile_size);

    // Extract L0 tiles from source image
    let mut l0_tiles: Vec<Vec<u8>> = Vec::new();
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
            l0_tiles.push(tile);
        }
    }

    // Build L1 mosaic by downsampling L0 2:1 (Lanczos3)
    let l1_w = (src_w + 1) / 2;
    let l1_h = (src_h + 1) / 2;
    let l1_rgb = {
        use image::{RgbImage, imageops};
        let src_img = RgbImage::from_raw(src_w, src_h, src_rgb.clone())
            .expect("failed to create RgbImage from source");
        let resized = imageops::resize(&src_img, l1_w, l1_h, imageops::FilterType::Lanczos3);
        resized.into_raw()
    };

    // Extract L1 tiles
    let l1_tiles_x = (l1_w + tile_size - 1) / tile_size;
    let l1_tiles_y = (l1_h + tile_size - 1) / tile_size;
    let mut l1_tiles: Vec<Vec<u8>> = Vec::new();
    for ty in 0..l1_tiles_y {
        for tx in 0..l1_tiles_x {
            let mut tile = vec![0u8; (tile_size * tile_size * 3) as usize];
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let sx = tx * tile_size + x;
                    let sy = ty * tile_size + y;
                    let di = ((y * tile_size + x) * 3) as usize;
                    if sx < l1_w && sy < l1_h {
                        let si = ((sy * l1_w + sx) * 3) as usize;
                        tile[di] = l1_rgb[si];
                        tile[di + 1] = l1_rgb[si + 1];
                        tile[di + 2] = l1_rgb[si + 2];
                    }
                }
            }
            l1_tiles.push(tile);
        }
    }

    // Build L2 by downsampling directly from L0 source 4:1 (Lanczos3)
    let l2_w = (l1_w + 1) / 2;
    let l2_h = (l1_h + 1) / 2;
    let mut l2_rgb = {
        use image::{RgbImage, imageops};
        let src_img = RgbImage::from_raw(src_w, src_h, src_rgb.clone())
            .expect("failed to create RgbImage from source");
        let resized = imageops::resize(&src_img, l2_w, l2_h, imageops::FilterType::Lanczos3);
        resized.into_raw()
    };

    // Optionally sharpen L2 to counteract bilinear upsample blur.
    // --sharpen creates a sharpened version for predictions.
    // --save-sharpened controls whether the stored L2 JPEG contains sharpened pixels.
    let l2_sharpened = if let Some(strength) = args.sharpen {
        use crate::core::sharpen::unsharp_mask_rgb;
        info!("Applying unsharp mask (strength={:.2}) to L2...", strength);
        Some(unsharp_mask_rgb(&l2_rgb, l2_w, l2_h, strength))
    } else {
        None
    };

    // Determine what to encode: sharpened if --save-sharpened, otherwise original
    let l2_to_encode = if args.save_sharpened {
        l2_sharpened.as_ref().unwrap_or(&l2_rgb).clone()
    } else {
        l2_rgb.clone()
    };

    // Optionally optimize L2 for better predictions (operates on the version being stored)
    let mut l2_to_encode = l2_to_encode;
    if args.optl2 {
        use crate::core::optimize_l2::optimize_l2_for_prediction;
        info!("Running OptL2 gradient descent optimization...");
        l2_to_encode = optimize_l2_for_prediction(&l2_to_encode, &l1_rgb, l2_w, l2_h, l1_w, l1_h, args.max_delta, 100, 0.3);
    }

    // Encode L2 baseline
    let l2_jpeg = encoder.encode_rgb_with_subsamp(&l2_to_encode, l2_w, l2_h, baseq, subsamp)?;
    let l2_bytes = l2_jpeg.len();
    let l2_out_path = args.out.join(format!("L2_0_0{}", ext));
    fs::write(&l2_out_path, &l2_jpeg)?;
    info!("L2 baseline: {}x{} → {} bytes (Q{} {})", l2_w, l2_h, l2_bytes, baseq, subsamp);

    // Decode L2 back (lossy round-trip) to get actual L2 that decoder will see
    let (l2_decoded_rgb, _, _) = crate::turbojpeg_optimized::decode_rgb_turbo(&l2_jpeg)?;

    // If --sharpen without --save-sharpened: sharpen the decoded L2 for predictions.
    // This simulates what the decode server will do (sharpen after JPEG decode, before upsample).
    let l2_decoded_for_pred = if l2_sharpened.is_some() && !args.save_sharpened {
        use crate::core::sharpen::unsharp_mask_rgb;
        let strength = args.sharpen.unwrap();
        info!("Sharpen-decode: applying unsharp mask (strength={:.2}) to decoded L2 for predictions...", strength);
        unsharp_mask_rgb(&l2_decoded_rgb, l2_w, l2_h, strength)
    } else {
        l2_decoded_rgb.clone()
    };

    // L2 for prediction: the decoded (lossy) L2, possibly sharpened at decode time
    let l2_for_prediction = l2_decoded_for_pred;

    // Setup debug image directories
    let compress_dir = args.out.join("compress");
    let decompress_dir = args.out.join("decompress");

    if args.debug_images {
        fs::create_dir_all(&compress_dir)?;
        fs::create_dir_all(&decompress_dir)?;
        let (l2_y, l2_cb, l2_cr) = ycbcr_planes_from_rgb(&l2_decoded_rgb, l2_w, l2_h);
        // L2 original (pre-encode, may be sharpened if --save-sharpened)
        save_rgb_png(&compress_dir.join("001_L2_original.png"), &l2_to_encode, l2_w, l2_h)?;
        // L2 luma, Cb, Cr channels (from decoded round-trip)
        save_gray_png(&compress_dir.join("002_L2_luma.png"), &l2_y, l2_w, l2_h)?;
        save_gray_png(&compress_dir.join("003_L2_chroma_cb.png"), &l2_cb, l2_w, l2_h)?;
        save_gray_png(&compress_dir.join("004_L2_chroma_cr.png"), &l2_cr, l2_w, l2_h)?;
        // L2 decoded (post JPEG round-trip, before any decode-time sharpen)
        save_rgb_png(&decompress_dir.join("050_L2_decode.png"), &l2_decoded_rgb, l2_w, l2_h)?;
        // L2 used for prediction (may be sharpened if --sharpen without --save-sharpened)
        save_rgb_png(&decompress_dir.join("050_L2_for_prediction.png"), &l2_for_prediction, l2_w, l2_h)?;
    }

    // Upsample L2 → L1 prediction in RGB space using residual-reconstructed L2
    let l1_pred_rgb = {
        use image::{RgbImage, imageops};
        let l2_img = RgbImage::from_raw(l2_w, l2_h, l2_for_prediction)
            .expect("failed to create RgbImage from L2 reconstructed");
        let resized = imageops::resize(&l2_img, l1_w, l1_h, imageops::FilterType::Triangle);
        resized.into_raw()
    };
    // Use float YCbCr to avoid u8 quantization loss (matches Python pipeline)
    use crate::core::color::{ycbcr_planes_from_rgb_f32, rgb_from_ycbcr_f32};
    use crate::core::residual::compute_residual_f32;
    let (l1_pred_y_f32, l1_pred_cb_f32, l1_pred_cr_f32) = ycbcr_planes_from_rgb_f32(&l1_pred_rgb, l1_w, l1_h);

    // Write L1 prediction mosaic as debug image
    if args.debug_images {
        save_rgb_png(&decompress_dir.join("051_L1_mosaic_prediction.png"), &l1_pred_rgb, l1_w, l1_h)?;
    }

    // Process L1 residuals
    let l1_out = args.out.join("L1").join("0_0");
    fs::create_dir_all(&l1_out)?;
    // Float Y mosaic for L0 prediction (avoids u8 quantization in recon Y)
    let mut l1_recon_y_f32 = vec![0.0f32; (l1_w * l1_h) as usize];
    let mut l1_recon_rgb = vec![0u8; (l1_w * l1_h * 3) as usize];
    let mut total_bytes = l2_bytes;
    let mut manifest_tiles: Vec<serde_json::Value> = Vec::new();

    for (idx, l1_tile) in l1_tiles.iter().enumerate() {
        let tx = idx as u32 % l1_tiles_x;
        let ty = idx as u32 / l1_tiles_x;

        let (l1_gt_y, _, _) = ycbcr_planes_from_rgb(l1_tile, tile_size, tile_size);

        // Extract L1 float prediction Y for this tile
        let tile_n = (tile_size * tile_size) as usize;
        let mut pred_y_f32_tile = vec![0.0f32; tile_n];
        for y in 0..tile_size {
            for x in 0..tile_size {
                let ti = (y * tile_size + x) as usize;
                let mx = tx * tile_size + x;
                let my = ty * tile_size + y;
                if mx < l1_w && my < l1_h {
                    pred_y_f32_tile[ti] = l1_pred_y_f32[(my * l1_w + mx) as usize];
                }
            }
        }

        // Float-precision residual: avoids quantizing pred_Y to u8 before subtraction
        let centered = compute_residual_f32(&l1_gt_y, &pred_y_f32_tile);
        let res_data = encoder.encode_gray(&centered, tile_size, tile_size, l1q)?;
        total_bytes += res_data.len();

        let out_path = l1_out.join(format!("{}_{}{}", tx, ty, ext));
        fs::write(&out_path, &res_data)?;

        // Decode residual back and reconstruct
        let decoded_res = decode_residual_bytes(&res_data)?;

        // Extract float Cb/Cr prediction for this tile (for float-precision reconstruction)
        let mut pred_cb_f32_tile = vec![0.0f32; tile_n];
        let mut pred_cr_f32_tile = vec![0.0f32; tile_n];
        for y in 0..tile_size {
            for x in 0..tile_size {
                let ti = (y * tile_size + x) as usize;
                let mx = tx * tile_size + x;
                let my = ty * tile_size + y;
                if mx < l1_w && my < l1_h {
                    let mi = (my * l1_w + mx) as usize;
                    pred_cb_f32_tile[ti] = l1_pred_cb_f32[mi];
                    pred_cr_f32_tile[ti] = l1_pred_cr_f32[mi];
                }
            }
        }

        // Reconstruct Y (float) and build RGB mosaic
        let mut recon_y_f32_tile = vec![0.0f32; tile_n];
        for y in 0..tile_size {
            for x in 0..tile_size {
                let ti = (y * tile_size + x) as usize;
                let pred_val = pred_y_f32_tile[ti];
                let res_val = decoded_res[ti] as f32 - 128.0;
                let recon = (pred_val + res_val).clamp(0.0, 255.0);
                recon_y_f32_tile[ti] = recon;
                let mx = tx * tile_size + x;
                let my = ty * tile_size + y;
                if mx < l1_w && my < l1_h {
                    let mi = (my * l1_w + mx) as usize;
                    l1_recon_y_f32[mi] = recon;
                    // Reconstruct RGB with float precision (single quantization at end)
                    let cbf = pred_cb_f32_tile[ti] - 128.0;
                    let crf = pred_cr_f32_tile[ti] - 128.0;
                    let r = (recon + 1.402 * crf).round().clamp(0.0, 255.0) as u8;
                    let g = (recon - 0.344136 * cbf - 0.714136 * crf).round().clamp(0.0, 255.0) as u8;
                    let b = (recon + 1.772 * cbf).round().clamp(0.0, 255.0) as u8;
                    let ri = mi * 3;
                    l1_recon_rgb[ri] = r;
                    l1_recon_rgb[ri + 1] = g;
                    l1_recon_rgb[ri + 2] = b;
                }
            }
        }

        // Write per-tile debug images
        if args.debug_images {
            let step = 10 + idx as u32 * 10;
            // Original RGB tile
            save_rgb_png(&compress_dir.join(format!("{:03}_L1_{}_{}_original.png", step, tx, ty)),
                l1_tile, tile_size, tile_size)?;
            // Luma of original
            save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_luma.png", step + 1, tx, ty)),
                &l1_gt_y, tile_size, tile_size)?;
            // Chroma channels (from prediction, quantized for debug visualization)
            let pred_cb_u8: Vec<u8> = pred_cb_f32_tile.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
            let pred_cr_u8: Vec<u8> = pred_cr_f32_tile.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
            save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_chroma_cb.png", step + 2, tx, ty)),
                &pred_cb_u8, tile_size, tile_size)?;
            save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_chroma_cr.png", step + 3, tx, ty)),
                &pred_cr_u8, tile_size, tile_size)?;
            // Prediction (RGB) — extract from l1_pred_rgb mosaic
            let mut pred_rgb_tile = vec![0u8; (tile_size * tile_size * 3) as usize];
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let mx = tx * tile_size + x;
                    let my = ty * tile_size + y;
                    if mx < l1_w && my < l1_h {
                        let si = ((my * l1_w + mx) * 3) as usize;
                        let di = ((y * tile_size + x) * 3) as usize;
                        pred_rgb_tile[di] = l1_pred_rgb[si];
                        pred_rgb_tile[di + 1] = l1_pred_rgb[si + 1];
                        pred_rgb_tile[di + 2] = l1_pred_rgb[si + 2];
                    }
                }
            }
            save_rgb_png(&compress_dir.join(format!("{:03}_L1_{}_{}_prediction.png", step + 4, tx, ty)),
                &pred_rgb_tile, tile_size, tile_size)?;
            // Raw residual (normalized to [0,255] for visualization)
            let residual_raw: Vec<f32> = l1_gt_y.iter().zip(pred_y_f32_tile.iter())
                .map(|(&gt_val, &pred_val)| gt_val as f32 - pred_val).collect();
            let raw_normalized = normalize_f32_to_u8(&residual_raw);
            save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_residual_raw.png", step + 5, tx, ty)),
                &raw_normalized, tile_size, tile_size)?;
            // Centered residual
            save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_residual_centered.png", step + 7, tx, ty)),
                &centered, tile_size, tile_size)?;
            // Reconstructed RGB (float-precision YCbCr → RGB)
            let recon_rgb = rgb_from_ycbcr_f32(&recon_y_f32_tile, &pred_cb_f32_tile, &pred_cr_f32_tile);
            save_rgb_png(&decompress_dir.join(format!("{:03}_L1_{}_{}_reconstructed.png", 63, tx, ty)),
                &recon_rgb, tile_size, tile_size)?;
        }

        if args.manifest {
            // Compute PSNR for this L1 tile (Y-channel, reconstructed vs ground truth)
            let mut mse_sum = 0.0f64;
            for i in 0..l1_gt_y.len() {
                let ry = (i as u32) / tile_size;
                let rx = (i as u32) % tile_size;
                let my = ty * tile_size + ry;
                let mx = tx * tile_size + rx;
                if mx < l1_w && my < l1_h {
                    let recon_val = l1_recon_y_f32[(my * l1_w + mx) as usize] as f64;
                    let gt_val = l1_gt_y[i] as f64;
                    let diff = recon_val - gt_val;
                    mse_sum += diff * diff;
                }
            }
            let mse = mse_sum / l1_gt_y.len() as f64;
            let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 100.0 };

            manifest_tiles.push(serde_json::json!({
                "level": "L1",
                "tx": tx, "ty": ty,
                "residual_bytes": res_data.len(),
                "y_psnr_db": (psnr * 100.0).round() / 100.0,
                "y_mse": (mse * 100.0).round() / 100.0,
            }));
        }
    }

    // Upsample reconstructed L1 → L0 prediction in RGB space (matches L2→L1 approach)
    let l0_pred_rgb = {
        use image::{RgbImage, imageops};
        let l1_img = RgbImage::from_raw(l1_w, l1_h, l1_recon_rgb)
            .expect("failed to create RgbImage from L1 reconstructed");
        let resized = imageops::resize(&l1_img, src_w, src_h, imageops::FilterType::Triangle);
        resized.into_raw()
    };

    // Use float YCbCr for L0 prediction (matches Python pipeline, avoids u8 quantization)
    let (l0_pred_y_f32, l0_pred_cb_f32, l0_pred_cr_f32) = ycbcr_planes_from_rgb_f32(&l0_pred_rgb, src_w, src_h);

    // Write L0 prediction mosaic as debug image
    if args.debug_images {
        save_rgb_png(&decompress_dir.join("065_L0_mosaic_prediction.png"), &l0_pred_rgb, src_w, src_h)?;
    }

    // Process L0 residuals
    let l0_out_dir = args.out.join("L0").join("0_0");
    fs::create_dir_all(&l0_out_dir)?;

    for (idx, l0_tile) in l0_tiles.iter().enumerate() {
        let tx = idx as u32 % tiles_x;
        let ty = idx as u32 / tiles_x;

        let (l0_gt_y, _, _) = ycbcr_planes_from_rgb(l0_tile, tile_size, tile_size);

        // Extract float prediction Y for this L0 tile
        let tile_n = (tile_size * tile_size) as usize;
        let mut pred_y_f32_tile = vec![0.0f32; tile_n];
        for y in 0..tile_size {
            for x in 0..tile_size {
                let ti = (y * tile_size + x) as usize;
                let mx = tx * tile_size + x;
                let my = ty * tile_size + y;
                if mx < src_w && my < src_h {
                    pred_y_f32_tile[ti] = l0_pred_y_f32[(my * src_w + mx) as usize];
                }
            }
        }

        // Float-precision residual
        let centered = compute_residual_f32(&l0_gt_y, &pred_y_f32_tile);
        let res_data = encoder.encode_gray(&centered, tile_size, tile_size, l0q)?;
        total_bytes += res_data.len();

        let out_path = l0_out_dir.join(format!("{}_{}{}", tx, ty, ext));
        fs::write(&out_path, &res_data)?;

        // Extract float Cb/Cr prediction for this L0 tile
        let mut pred_cb_f32_tile = vec![0.0f32; tile_n];
        let mut pred_cr_f32_tile = vec![0.0f32; tile_n];
        for y in 0..tile_size {
            for x in 0..tile_size {
                let ti = (y * tile_size + x) as usize;
                let mx = tx * tile_size + x;
                let my = ty * tile_size + y;
                if mx < src_w && my < src_h {
                    let mi = (my * src_w + mx) as usize;
                    pred_cb_f32_tile[ti] = l0_pred_cb_f32[mi];
                    pred_cr_f32_tile[ti] = l0_pred_cr_f32[mi];
                }
            }
        }

        // Write per-tile debug images for L0
        if args.debug_images {
            let step = 20 + idx as u32;
            // Original RGB tile
            save_rgb_png(&compress_dir.join(format!("{:03}_L0_{}_{}_original.png", step, tx, ty)),
                l0_tile, tile_size, tile_size)?;
            // Luma of original
            save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_luma.png", step, tx, ty)),
                &l0_gt_y, tile_size, tile_size)?;
            // Chroma channels (from prediction, quantized for debug visualization)
            let pred_cb_u8: Vec<u8> = pred_cb_f32_tile.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
            let pred_cr_u8: Vec<u8> = pred_cr_f32_tile.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
            save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_chroma_cb.png", step, tx, ty)),
                &pred_cb_u8, tile_size, tile_size)?;
            save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_chroma_cr.png", step, tx, ty)),
                &pred_cr_u8, tile_size, tile_size)?;
            // Prediction (RGB) — extract from l0_pred_rgb mosaic
            let mut pred_rgb_tile = vec![0u8; (tile_size * tile_size * 3) as usize];
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let mx = tx * tile_size + x;
                    let my = ty * tile_size + y;
                    if mx < src_w && my < src_h {
                        let si = ((my * src_w + mx) * 3) as usize;
                        let di = ((y * tile_size + x) * 3) as usize;
                        pred_rgb_tile[di] = l0_pred_rgb[si];
                        pred_rgb_tile[di + 1] = l0_pred_rgb[si + 1];
                        pred_rgb_tile[di + 2] = l0_pred_rgb[si + 2];
                    }
                }
            }
            save_rgb_png(&compress_dir.join(format!("{:03}_L0_{}_{}_prediction.png", step, tx, ty)),
                &pred_rgb_tile, tile_size, tile_size)?;
            // Raw residual (normalized to [0,255] for visualization)
            let residual_raw: Vec<f32> = l0_gt_y.iter().zip(pred_y_f32_tile.iter())
                .map(|(&gt_val, &pred_val)| gt_val as f32 - pred_val).collect();
            let raw_normalized = normalize_f32_to_u8(&residual_raw);
            save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_residual_raw.png", step, tx, ty)),
                &raw_normalized, tile_size, tile_size)?;
            // Centered residual
            save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_residual_centered.png", step, tx, ty)),
                &centered, tile_size, tile_size)?;
        }

        if args.manifest {
            // Compute PSNR for this L0 tile
            let decoded_res = decode_residual_bytes(&res_data)?;

            // Write reconstructed RGB debug image for L0 (float-precision YCbCr → RGB)
            if args.debug_images {
                let mut recon_y_f32_tile = vec![0.0f32; tile_n];
                for i in 0..l0_gt_y.len() {
                    let pv = pred_y_f32_tile[i];
                    let rv = decoded_res[i] as f32 - 128.0;
                    recon_y_f32_tile[i] = (pv + rv).clamp(0.0, 255.0);
                }
                let recon_rgb = rgb_from_ycbcr_f32(&recon_y_f32_tile, &pred_cb_f32_tile, &pred_cr_f32_tile);
                let step = 70 + idx as u32;
                save_rgb_png(&decompress_dir.join(format!("{:03}_L0_{}_{}_reconstructed.png", step, tx, ty)),
                    &recon_rgb, tile_size, tile_size)?;
            }

            let mut mse_sum = 0.0f64;
            let n = l0_gt_y.len();
            for i in 0..n {
                let pred_val = pred_y_f32_tile[i] as f64;
                let res_val = decoded_res[i] as f64 - 128.0;
                let recon = (pred_val + res_val).max(0.0).min(255.0);
                let gt_val = l0_gt_y[i] as f64;
                let diff = recon - gt_val;
                mse_sum += diff * diff;
            }
            let mse = mse_sum / n as f64;
            let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 100.0 };

            manifest_tiles.push(serde_json::json!({
                "level": "L0",
                "tx": tx, "ty": ty,
                "residual_bytes": res_data.len(),
                "y_psnr_db": (psnr * 100.0).round() / 100.0,
                "y_mse": (mse * 100.0).round() / 100.0,
            }));
        }
    }

    let elapsed = start.elapsed();
    info!(
        "Single-image encode complete: {} L1 + {} L0 tiles, {} total bytes, {:.1}s",
        l1_tiles.len(), l0_tiles.len(), total_bytes, elapsed.as_secs_f64()
    );

    // Write summary.json
    let summary = serde_json::json!({
        "mode": "single-image",
        "source": image_path.to_string_lossy(),
        "source_w": src_w,
        "source_h": src_h,
        "encoder": args.encoder,
        "subsamp": subsamp.to_string(),
        "baseq": baseq,
        "l1q": l1q,
        "l0q": l0q,
        "optl2": args.optl2,
        "sharpen": args.sharpen,
        "save_sharpened": args.save_sharpened,
        "tile_size": tile_size,
        "l2_bytes": l2_bytes,
        "l2_w": l2_w,
        "l2_h": l2_h,
        "l1_tiles": l1_tiles.len(),
        "l0_tiles": l0_tiles.len(),
        "total_bytes": total_bytes,
        "elapsed_secs": elapsed.as_secs_f64(),
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
            "baseq": baseq,
            "l1q": l1q,
            "l0q": l0q,
            "optl2": args.optl2,
            "sharpen": args.sharpen,
            "tile_size": tile_size,
            "l2_bytes": l2_bytes,
            "l2_w": l2_w,
            "l2_h": l2_h,
            "total_bytes": total_bytes,
            "tiles": manifest_tiles,
        });
        fs::write(
            args.out.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;
    }

    Ok(())
}
