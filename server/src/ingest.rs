use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use rayon::prelude::*;
use tracing::info;

use openslide_rs::{Address, OpenSlide, Region, Size};

use crate::core::color::{ycbcr_planes_from_rgb, ycbcr_planes_from_rgb_f32};
use crate::core::jpeg::{create_encoder, ChromaSubsampling, JpegEncoder};
use crate::core::pack::{compress_pack_entries, write_bundle, write_pack, PackWriteEntry};
use crate::core::residual::compute_residual_f32;

#[derive(Args, Debug)]
pub struct IngestArgs {
    /// Path to WSI file (DICOM directory, .svs, .tiff, etc.)
    #[arg(long)]
    pub slide: PathBuf,

    /// Output slide directory
    #[arg(long)]
    pub out: PathBuf,

    /// DZI tile size
    #[arg(long, default_value_t = 256)]
    pub tile: u32,

    /// JPEG quality for L2 baseline tiles
    #[arg(long, default_value_t = 95)]
    pub baseq: u8,

    /// JPEG quality for L1 residuals
    #[arg(long, default_value_t = 60)]
    pub l1q: u8,

    /// JPEG quality for L0 residuals
    #[arg(long, default_value_t = 40)]
    pub l0q: u8,

    /// Chroma subsampling for L2: 444, 420, 420opt
    #[arg(long, default_value = "444")]
    pub subsamp: String,

    /// Optimize L2 for better bilinear predictions (gradient descent)
    #[arg(long)]
    pub optl2: bool,

    /// Encoder backend: turbojpeg, mozjpeg, jpegli
    #[arg(long, default_value = "turbojpeg")]
    pub encoder: String,

    /// Maximum number of families to process (0 = all, for testing)
    #[arg(long, default_value_t = 0)]
    pub max_parents: usize,

    /// Also create individual pack files (containing L1/L0 residuals)
    #[arg(long)]
    pub pack: bool,

    /// Write a single residuals.bundle file (mmapped serving, preferred over --pack)
    #[arg(long)]
    pub bundle: bool,

    /// L2 intermediate format: png (lossless, better lower levels) or jpg
    #[arg(long, default_value = "png")]
    pub l2_format: String,

    /// JPEG-only mode: write all tiles (L0/L1/L2) as plain JPEG, no residuals
    #[arg(long)]
    pub jpeg_only: bool,

    /// Maximum per-pixel deviation for OptL2 gradient descent (default: 15)
    #[arg(long, default_value_t = 15)]
    pub max_delta: u8,
}

/// Read a 1024x1024 region from the slide at the given pixel coordinates.
/// Returns RGB u8 pixels (1024*1024*3 bytes). Out-of-bounds pixels are white.
fn read_region_rgb(slide: &OpenSlide, x: u32, y: u32, size: u32, slide_w: u32, slide_h: u32) -> Result<Vec<u8>> {
    let region = Region {
        address: Address { x, y },
        level: 0,
        size: Size { w: size, h: size },
    };
    let bgra = slide.read_region(&region)
        .map_err(|e| anyhow::anyhow!("read_region({},{}) failed: {:?}", x, y, e))?;

    // Convert pre-multiplied BGRA to RGB, filling transparent pixels with white
    let n = (size * size) as usize;
    let mut rgb = vec![255u8; n * 3]; // default white
    for i in 0..n {
        let a = bgra[i * 4 + 3];
        if a == 0 {
            continue; // leave as white
        }
        if a == 255 {
            // No premultiplication needed
            rgb[i * 3] = bgra[i * 4 + 2];     // R (BGRA → RGB)
            rgb[i * 3 + 1] = bgra[i * 4 + 1]; // G
            rgb[i * 3 + 2] = bgra[i * 4];       // B
        } else {
            // Un-premultiply: pixel = premultiplied * 255 / alpha
            let af = a as f32;
            rgb[i * 3] = ((bgra[i * 4 + 2] as f32 * 255.0 / af).round().min(255.0)) as u8;
            rgb[i * 3 + 1] = ((bgra[i * 4 + 1] as f32 * 255.0 / af).round().min(255.0)) as u8;
            rgb[i * 3 + 2] = ((bgra[i * 4] as f32 * 255.0 / af).round().min(255.0)) as u8;
        }
    }

    // Clamp actual content to slide bounds (anything beyond slide dimensions → white)
    // OpenSlide already handles this via alpha=0, but let's be safe for partial tiles
    // at the right/bottom edge where slide dimensions aren't multiples of `size`
    if x + size > slide_w {
        let valid_w = if x < slide_w { slide_w - x } else { 0 };
        for py in 0..size {
            for px in valid_w..size {
                let idx = (py * size + px) as usize;
                rgb[idx * 3] = 255;
                rgb[idx * 3 + 1] = 255;
                rgb[idx * 3 + 2] = 255;
            }
        }
    }
    if y + size > slide_h {
        let valid_h = if y < slide_h { slide_h - y } else { 0 };
        for py in valid_h..size {
            for px in 0..size {
                let idx = (py * size + px) as usize;
                rgb[idx * 3] = 255;
                rgb[idx * 3 + 1] = 255;
                rgb[idx * 3 + 2] = 255;
            }
        }
    }

    Ok(rgb)
}

/// Extract a tile_size x tile_size sub-tile from a larger image.
fn extract_tile(src: &[u8], src_w: u32, src_h: u32, tx: u32, ty: u32, tile_size: u32) -> Vec<u8> {
    let mut tile = vec![0u8; (tile_size * tile_size * 3) as usize];
    for y in 0..tile_size {
        for x in 0..tile_size {
            let sx = tx * tile_size + x;
            let sy = ty * tile_size + y;
            let di = ((y * tile_size + x) * 3) as usize;
            if sx < src_w && sy < src_h {
                let si = ((sy * src_w + sx) * 3) as usize;
                tile[di] = src[si];
                tile[di + 1] = src[si + 1];
                tile[di + 2] = src[si + 2];
            } else {
                tile[di] = 255;
                tile[di + 1] = 255;
                tile[di + 2] = 255;
            }
        }
    }
    tile
}

/// Result from processing one family (one 1024x1024 region).
struct FamilyResult {
    col: u32,
    row: u32,
    l2_jpeg: Vec<u8>,
    l2_png: Option<Vec<u8>>,
    pack_entries: Vec<PackWriteEntry>,
    total_residual_bytes: usize,
    /// Individual L1 residual sizes (bytes)
    l1_residual_sizes: Vec<usize>,
    /// Individual L0 residual sizes (bytes)
    l0_residual_sizes: Vec<usize>,
}

/// Result from JPEG-only processing (all tiles, no residuals).
struct JpegOnlyFamilyResult {
    col: u32,
    row: u32,
    /// L2 tile (1 tile: 256x256)
    l2_jpeg: Vec<u8>,
    l2_png: Option<Vec<u8>>,
    /// L1 tiles: (dx, dy, jpeg_data)
    l1_tiles: Vec<(u32, u32, Vec<u8>)>,
    /// L0 tiles: (dx, dy, jpeg_data)
    l0_tiles: Vec<(u32, u32, Vec<u8>)>,
    total_bytes: usize,
}

/// Process one family in JPEG-only mode: read region, downsample, encode all tiles as JPEG.
fn process_family_jpeg_only(
    slide: &OpenSlide,
    col: u32,
    row: u32,
    slide_w: u32,
    slide_h: u32,
    tile_size: u32,
    encoder: &dyn JpegEncoder,
    baseq: u8,
    subsamp: ChromaSubsampling,
    save_png: bool,
) -> Result<JpegOnlyFamilyResult> {
    let region_size = tile_size * 4; // 1024 for tile_size=256

    // 1. Read the 1024x1024 L0 ground truth
    let l0_rgb = read_region_rgb(slide, col * region_size, row * region_size, region_size, slide_w, slide_h)?;
    let l0_w = region_size;
    let l0_h = region_size;

    // 2. Downsample L0 → L1 (512x512) and L0 → L2 (256x256)
    let l1_w = l0_w / 2;
    let l1_h = l0_h / 2;
    let l2_w = l0_w / 4;
    let l2_h = l0_h / 4;

    let l1_rgb = {
        use image::{RgbImage, imageops};
        let img = RgbImage::from_raw(l0_w, l0_h, l0_rgb.clone())
            .ok_or_else(|| anyhow::anyhow!("failed to create L0 RgbImage"))?;
        let resized = imageops::resize(&img, l1_w, l1_h, imageops::FilterType::Lanczos3);
        resized.into_raw()
    };

    let l2_rgb = {
        use image::{RgbImage, imageops};
        let img = RgbImage::from_raw(l0_w, l0_h, l0_rgb.clone())
            .ok_or_else(|| anyhow::anyhow!("failed to create L0 RgbImage for L2"))?;
        let resized = imageops::resize(&img, l2_w, l2_h, imageops::FilterType::Lanczos3);
        resized.into_raw()
    };

    let mut total_bytes = 0usize;

    // 3. Encode L2 tile
    let l2_jpeg = encoder.encode_rgb_with_subsamp(&l2_rgb, l2_w, l2_h, baseq, subsamp)?;
    total_bytes += l2_jpeg.len();

    // 4. Save L2 as PNG (lossless intermediate for lower-level synthesis)
    let l2_png = if save_png {
        let img = image::RgbImage::from_raw(l2_w, l2_h, l2_rgb)
            .ok_or_else(|| anyhow::anyhow!("failed to create L2 PNG image"))?;
        let mut buf = Vec::new();
        let png_encoder = image::codecs::png::PngEncoder::new(&mut buf);
        image::ImageEncoder::write_image(
            png_encoder, &img, l2_w, l2_h, image::ExtendedColorType::Rgb8,
        )?;
        Some(buf)
    } else {
        None
    };

    // 5. Encode L1 tiles (2x2 grid)
    let mut l1_tiles = Vec::with_capacity(4);
    for dy in 0..2u32 {
        for dx in 0..2u32 {
            let tile = extract_tile(&l1_rgb, l1_w, l1_h, dx, dy, tile_size);
            let jpeg = encoder.encode_rgb_with_subsamp(&tile, tile_size, tile_size, baseq, subsamp)?;
            total_bytes += jpeg.len();
            l1_tiles.push((dx, dy, jpeg));
        }
    }

    // 6. Encode L0 tiles (4x4 grid)
    let mut l0_tiles = Vec::with_capacity(16);
    for dy in 0..4u32 {
        for dx in 0..4u32 {
            let tile = extract_tile(&l0_rgb, l0_w, l0_h, dx, dy, tile_size);
            let jpeg = encoder.encode_rgb_with_subsamp(&tile, tile_size, tile_size, baseq, subsamp)?;
            total_bytes += jpeg.len();
            l0_tiles.push((dx, dy, jpeg));
        }
    }

    Ok(JpegOnlyFamilyResult {
        col,
        row,
        l2_jpeg,
        l2_png,
        l1_tiles,
        l0_tiles,
        total_bytes,
    })
}

/// Process one family: read region, downsample, compute residuals, return results.
fn process_family(
    slide: &OpenSlide,
    col: u32,
    row: u32,
    slide_w: u32,
    slide_h: u32,
    tile_size: u32,
    encoder: &dyn JpegEncoder,
    baseq: u8,
    l1q: u8,
    l0q: u8,
    subsamp: ChromaSubsampling,
    optl2: bool,
    max_delta: u8,
    save_png: bool,
    do_pack: bool,
) -> Result<FamilyResult> {
    let region_size = tile_size * 4; // 1024 for tile_size=256

    // 1. Read the 1024x1024 L0 ground truth
    let l0_rgb = read_region_rgb(slide, col * region_size, row * region_size, region_size, slide_w, slide_h)?;
    let l0_w = region_size;
    let l0_h = region_size;

    // 2. Downsample L0 → L1 (512x512) and L0 → L2 (256x256) using Lanczos3
    let l1_w = l0_w / 2;
    let l1_h = l0_h / 2;
    let l2_w = l0_w / 4;
    let l2_h = l0_h / 4;

    let l1_rgb = {
        use image::{RgbImage, imageops};
        let img = RgbImage::from_raw(l0_w, l0_h, l0_rgb.clone())
            .ok_or_else(|| anyhow::anyhow!("failed to create L0 RgbImage"))?;
        let resized = imageops::resize(&img, l1_w, l1_h, imageops::FilterType::Lanczos3);
        resized.into_raw()
    };

    let mut l2_rgb = {
        use image::{RgbImage, imageops};
        let img = RgbImage::from_raw(l0_w, l0_h, l0_rgb.clone())
            .ok_or_else(|| anyhow::anyhow!("failed to create L0 RgbImage for L2"))?;
        let resized = imageops::resize(&img, l2_w, l2_h, imageops::FilterType::Lanczos3);
        resized.into_raw()
    };

    // 3. Optionally optimize L2 for better bilinear predictions
    if optl2 {
        use crate::core::optimize_l2::optimize_l2_for_prediction;
        l2_rgb = optimize_l2_for_prediction(&l2_rgb, &l1_rgb, l2_w, l2_h, l1_w, l1_h, max_delta, 100, 0.3);
    }

    // 4. Save L2 as PNG (lossless intermediate for lower-level synthesis)
    let l2_png = if save_png {
        let img = image::RgbImage::from_raw(l2_w, l2_h, l2_rgb.clone())
            .ok_or_else(|| anyhow::anyhow!("failed to create L2 PNG image"))?;
        let mut buf = Vec::new();
        let png_encoder = image::codecs::png::PngEncoder::new(&mut buf);
        image::ImageEncoder::write_image(
            png_encoder,
            &img,
            l2_w,
            l2_h,
            image::ExtendedColorType::Rgb8,
        )?;
        Some(buf)
    } else {
        None
    };

    // 5. Encode L2 as JPEG
    let l2_jpeg = encoder.encode_rgb_with_subsamp(&l2_rgb, l2_w, l2_h, baseq, subsamp)?;

    // 6. Decode L2 JPEG back (lossy round-trip) for prediction
    let (l2_decoded_rgb, _, _) = crate::turbojpeg_optimized::decode_rgb_turbo(&l2_jpeg)?;

    // 7. Upsample L2 → L1 prediction (bilinear, in RGB space)
    let l1_pred_rgb = {
        use image::{RgbImage, imageops};
        let img = RgbImage::from_raw(l2_w, l2_h, l2_decoded_rgb)
            .ok_or_else(|| anyhow::anyhow!("failed to create L2 decoded image"))?;
        let resized = imageops::resize(&img, l1_w, l1_h, imageops::FilterType::Triangle);
        resized.into_raw()
    };

    // Float YCbCr for predictions (avoids u8 quantization loss)
    let (l1_pred_y_f32, l1_pred_cb_f32, l1_pred_cr_f32) = ycbcr_planes_from_rgb_f32(&l1_pred_rgb, l1_w, l1_h);

    let mut pack_entries: Vec<PackWriteEntry> = Vec::new();
    let mut total_residual_bytes = 0usize;
    let mut l1_residual_sizes: Vec<usize> = Vec::with_capacity(4);
    let mut l0_residual_sizes: Vec<usize> = Vec::with_capacity(16);

    // Float Y mosaic for L0 prediction
    let mut l1_recon_y_f32 = vec![0.0f32; (l1_w * l1_h) as usize];
    let mut l1_recon_rgb = vec![0u8; (l1_w * l1_h * 3) as usize];

    // 8. Process L1 children (2x2 grid of tile_size tiles)
    for dy in 0..2u32 {
        for dx in 0..2u32 {
            let l1_tile = extract_tile(&l1_rgb, l1_w, l1_h, dx, dy, tile_size);
            let (l1_gt_y, _, _) = ycbcr_planes_from_rgb(&l1_tile, tile_size, tile_size);

            // Extract float prediction Y for this tile
            let tile_n = (tile_size * tile_size) as usize;
            let mut pred_y_f32_tile = vec![0.0f32; tile_n];
            let mut pred_cb_f32_tile = vec![0.0f32; tile_n];
            let mut pred_cr_f32_tile = vec![0.0f32; tile_n];
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let ti = (y * tile_size + x) as usize;
                    let mx = dx * tile_size + x;
                    let my = dy * tile_size + y;
                    if mx < l1_w && my < l1_h {
                        let mi = (my * l1_w + mx) as usize;
                        pred_y_f32_tile[ti] = l1_pred_y_f32[mi];
                        pred_cb_f32_tile[ti] = l1_pred_cb_f32[mi];
                        pred_cr_f32_tile[ti] = l1_pred_cr_f32[mi];
                    }
                }
            }

            // Float-precision residual
            let centered = compute_residual_f32(&l1_gt_y, &pred_y_f32_tile);
            let res_data = encoder.encode_gray(&centered, tile_size, tile_size, l1q)?;
            l1_residual_sizes.push(res_data.len());
            total_residual_bytes += res_data.len();

            // Decode residual back and reconstruct Y
            let decoded_res = crate::turbojpeg_optimized::decode_luma_turbo(&res_data)?.0;

            for y in 0..tile_size {
                for x in 0..tile_size {
                    let ti = (y * tile_size + x) as usize;
                    let pred_val = pred_y_f32_tile[ti];
                    let res_val = decoded_res[ti] as f32 - 128.0;
                    let recon = (pred_val + res_val).clamp(0.0, 255.0);
                    let mx = dx * tile_size + x;
                    let my = dy * tile_size + y;
                    if mx < l1_w && my < l1_h {
                        let mi = (my * l1_w + mx) as usize;
                        l1_recon_y_f32[mi] = recon;
                        // Reconstruct RGB
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

            if do_pack {
                pack_entries.push(PackWriteEntry {
                    level_kind: 1,
                    idx_in_parent: (dy * 2 + dx) as u8,
                    jpeg_data: res_data,
                });
            }
        }
    }

    // 9. Upsample reconstructed L1 → L0 prediction (bilinear, RGB space)
    let l0_pred_rgb = {
        use image::{RgbImage, imageops};
        let img = RgbImage::from_raw(l1_w, l1_h, l1_recon_rgb)
            .ok_or_else(|| anyhow::anyhow!("failed to create L1 recon image"))?;
        let resized = imageops::resize(&img, l0_w, l0_h, imageops::FilterType::Triangle);
        resized.into_raw()
    };
    let (l0_pred_y_f32, _, _) = ycbcr_planes_from_rgb_f32(&l0_pred_rgb, l0_w, l0_h);

    // 10. Process L0 children (4x4 grid of tile_size tiles)
    for dy in 0..4u32 {
        for dx in 0..4u32 {
            let l0_tile = extract_tile(&l0_rgb, l0_w, l0_h, dx, dy, tile_size);
            let (l0_gt_y, _, _) = ycbcr_planes_from_rgb(&l0_tile, tile_size, tile_size);

            let tile_n = (tile_size * tile_size) as usize;
            let mut pred_y_f32_tile = vec![0.0f32; tile_n];
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let ti = (y * tile_size + x) as usize;
                    let mx = dx * tile_size + x;
                    let my = dy * tile_size + y;
                    if mx < l0_w && my < l0_h {
                        pred_y_f32_tile[ti] = l0_pred_y_f32[(my * l0_w + mx) as usize];
                    }
                }
            }

            let centered = compute_residual_f32(&l0_gt_y, &pred_y_f32_tile);
            let res_data = encoder.encode_gray(&centered, tile_size, tile_size, l0q)?;
            l0_residual_sizes.push(res_data.len());
            total_residual_bytes += res_data.len();

            if do_pack {
                pack_entries.push(PackWriteEntry {
                    level_kind: 0,
                    idx_in_parent: (dy * 4 + dx) as u8,
                    jpeg_data: res_data,
                });
            }
        }
    }

    Ok(FamilyResult {
        col,
        row,
        l2_jpeg,
        l2_png,
        pack_entries,
        total_residual_bytes,
        l1_residual_sizes,
        l0_residual_sizes,
    })
}

/// Compute distribution statistics for a sorted vector of sizes.
fn compute_size_stats(sizes: &mut Vec<usize>) -> serde_json::Value {
    if sizes.is_empty() {
        return serde_json::json!({});
    }
    sizes.sort();
    let n = sizes.len();
    let total: usize = sizes.iter().sum();
    let avg = total as f64 / n as f64;
    let min = sizes[0];
    let max = sizes[n - 1];
    let median = sizes[n / 2];
    let p95 = sizes[(n as f64 * 0.95) as usize];
    let p99 = sizes[((n as f64 * 0.99) as usize).min(n - 1)];
    let p5 = sizes[(n as f64 * 0.05) as usize];
    let p25 = sizes[(n as f64 * 0.25) as usize];
    let p75 = sizes[(n as f64 * 0.75) as usize];

    serde_json::json!({
        "count": n,
        "total_bytes": total,
        "total_mb": format!("{:.2}", total as f64 / 1_048_576.0),
        "avg": format!("{:.0}", avg),
        "min": min,
        "p5": p5,
        "p25": p25,
        "median": median,
        "p75": p75,
        "p95": p95,
        "p99": p99,
        "max": max,
    })
}

/// Compute DZI level number for a given dimension.
/// DZI level N has max(width, height) fitting in 2^N pixels.
fn dzi_max_level(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim == 0 { return 0; }
    (max_dim as f64).log2().ceil() as u32
}

/// Write a DZI manifest file.
fn write_dzi_manifest(path: &std::path::Path, width: u32, height: u32, tile_size: u32) -> Result<()> {
    let dzi = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpg"
  Overlap="0"
  TileSize="{}"
  >
  <Size
    Height="{}"
    Width="{}"
  />
</Image>
"#,
        tile_size, height, width
    );
    fs::write(path, &dzi)?;
    Ok(())
}

/// Synthesize lower DZI pyramid levels from L2 tiles.
/// Reads L2 source images (PNG or JPEG), combines 2x2 parents into children,
/// downsamples, and saves as JPEG. Continues down to level 0 (1x1).
fn synthesize_lower_levels(
    files_dir: &std::path::Path,
    l2_level: u32,
    l2_cols: u32,
    l2_rows: u32,
    tile_size: u32,
    l2_png_dir: Option<&std::path::Path>,
    encoder: &dyn JpegEncoder,
    baseq: u8,
    subsamp: ChromaSubsampling,
) -> Result<()> {
    use image::{RgbImage, imageops};

    if l2_level == 0 {
        return Ok(());
    }

    // For the first step down from L2, use PNG sources if available (lossless)
    // For subsequent levels, use JPEG from the previous level
    let mut prev_level = l2_level;
    let mut prev_cols = l2_cols;
    let mut prev_rows = l2_rows;

    for level in (0..l2_level).rev() {
        let cur_cols = (prev_cols + 1) / 2;
        let cur_rows = (prev_rows + 1) / 2;
        let level_dir = files_dir.join(level.to_string());
        fs::create_dir_all(&level_dir)?;

        for ty in 0..cur_rows {
            for tx in 0..cur_cols {
                // Build a 2x2 mosaic from parent tiles
                let mosaic_w = tile_size * 2;
                let mosaic_h = tile_size * 2;
                let mut mosaic = vec![255u8; (mosaic_w * mosaic_h * 3) as usize];

                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let px = tx * 2 + dx;
                        let py = ty * 2 + dy;
                        if px >= prev_cols || py >= prev_rows {
                            continue;
                        }

                        // Load parent tile
                        let parent_rgb = if prev_level == l2_level {
                            // Use PNG if available for first step
                            if let Some(png_dir) = l2_png_dir {
                                let png_path = png_dir.join(format!("{}_{}.png", px, py));
                                if png_path.exists() {
                                    let img = image::open(&png_path)
                                        .with_context(|| format!("loading L2 PNG {}", png_path.display()))?;
                                    img.to_rgb8().into_raw()
                                } else {
                                    // Fall back to JPEG
                                    let jpg_path = files_dir.join(prev_level.to_string()).join(format!("{}_{}.jpg", px, py));
                                    if !jpg_path.exists() { continue; }
                                    let data = fs::read(&jpg_path)?;
                                    crate::turbojpeg_optimized::decode_rgb_turbo(&data)?.0
                                }
                            } else {
                                let jpg_path = files_dir.join(prev_level.to_string()).join(format!("{}_{}.jpg", px, py));
                                if !jpg_path.exists() { continue; }
                                let data = fs::read(&jpg_path)?;
                                crate::turbojpeg_optimized::decode_rgb_turbo(&data)?.0
                            }
                        } else {
                            let jpg_path = files_dir.join(prev_level.to_string()).join(format!("{}_{}.jpg", px, py));
                            if !jpg_path.exists() { continue; }
                            let data = fs::read(&jpg_path)?;
                            crate::turbojpeg_optimized::decode_rgb_turbo(&data)?.0
                        };

                        // Copy into mosaic
                        for y in 0..tile_size {
                            for x in 0..tile_size {
                                let src_i = ((y * tile_size + x) * 3) as usize;
                                let dst_x = dx * tile_size + x;
                                let dst_y = dy * tile_size + y;
                                let dst_i = ((dst_y * mosaic_w + dst_x) * 3) as usize;
                                if src_i + 2 < parent_rgb.len() {
                                    mosaic[dst_i] = parent_rgb[src_i];
                                    mosaic[dst_i + 1] = parent_rgb[src_i + 1];
                                    mosaic[dst_i + 2] = parent_rgb[src_i + 2];
                                }
                            }
                        }
                    }
                }

                // Downsample mosaic to tile_size x tile_size
                let mosaic_img = RgbImage::from_raw(mosaic_w, mosaic_h, mosaic)
                    .ok_or_else(|| anyhow::anyhow!("failed to create mosaic image"))?;
                let resized = imageops::resize(&mosaic_img, tile_size, tile_size, imageops::FilterType::Lanczos3);
                let tile_rgb = resized.into_raw();

                // Encode as JPEG
                let jpeg_data = encoder.encode_rgb_with_subsamp(&tile_rgb, tile_size, tile_size, baseq, subsamp)?;
                let out_path = level_dir.join(format!("{}_{}.jpg", tx, ty));
                fs::write(&out_path, &jpeg_data)?;
            }
        }

        info!("  Level {}: {}x{} tiles", level, cur_cols, cur_rows);
        prev_level = level;
        prev_cols = cur_cols;
        prev_rows = cur_rows;
    }

    Ok(())
}

pub fn run(args: IngestArgs) -> Result<()> {
    // Enable Huffman optimization for turbojpeg
    std::env::set_var("TJ_OPTIMIZE", "1");

    let start = Instant::now();
    let subsamp: ChromaSubsampling = args.subsamp.parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;
    let encoder = create_encoder(&args.encoder)?;
    let tile_size = args.tile;
    let region_size = tile_size * 4; // 1024 for tile_size=256

    info!("Opening slide: {}", args.slide.display());
    let slide = OpenSlide::new(&args.slide)
        .map_err(|e| anyhow::anyhow!("failed to open slide: {:?}", e))?;

    let dims = slide.get_level_dimensions(0)
        .map_err(|e| anyhow::anyhow!("failed to get dimensions: {:?}", e))?;
    let slide_w = dims.w;
    let slide_h = dims.h;
    info!("Slide dimensions: {}x{}", slide_w, slide_h);

    let level_count = slide.get_level_count()
        .map_err(|e| anyhow::anyhow!("failed to get level count: {:?}", e))?;
    info!("Slide levels: {}", level_count);

    // Compute grid of families
    let grid_cols = (slide_w + region_size - 1) / region_size;
    let grid_rows = (slide_h + region_size - 1) / region_size;
    let total_families = (grid_cols * grid_rows) as usize;
    info!("Family grid: {}x{} = {} families ({}x{} regions)", grid_cols, grid_rows, total_families, region_size, region_size);

    // DZI levels: L0 = max_level, L1 = max_level-1, L2 = max_level-2
    let max_level = dzi_max_level(slide_w, slide_h);
    let l2_level = max_level.saturating_sub(2);
    info!("DZI max_level={}, L0={}, L1={}, L2={}", max_level, max_level, max_level - 1, l2_level);

    // L2 tile grid dimensions
    let l2_cols = grid_cols;
    let l2_rows = grid_rows;

    // Create output directories
    let files_dir = args.out.join("baseline_pyramid_files");
    let l2_tile_dir = files_dir.join(l2_level.to_string());
    fs::create_dir_all(&l2_tile_dir)?;

    // Build list of families to process
    let mut families: Vec<(u32, u32)> = Vec::with_capacity(total_families);
    for row in 0..grid_rows {
        for col in 0..grid_cols {
            families.push((col, row));
        }
    }
    if args.max_parents > 0 {
        families.truncate(args.max_parents);
        info!("Limited to {} families (--max-parents)", args.max_parents);
    }
    let families_to_process = families.len();
    let processed = AtomicUsize::new(0);

    if args.jpeg_only {
        // --- JPEG-only mode: write all tiles as plain JPEG, no residuals ---
        let l1_level = max_level.saturating_sub(1);
        let l0_level = max_level;
        let l1_tile_dir = files_dir.join(l1_level.to_string());
        let l0_tile_dir = files_dir.join(l0_level.to_string());
        fs::create_dir_all(&l1_tile_dir)?;
        fs::create_dir_all(&l0_tile_dir)?;

        let save_png = args.l2_format == "png";
        let l2_png_dir = if save_png {
            let d = args.out.join("l2_png_tmp");
            fs::create_dir_all(&d)?;
            Some(d)
        } else {
            None
        };

        info!("JPEG-only mode: writing all L0/L1/L2 tiles as JPEG Q{} with encoder={} subsamp={}",
            args.baseq, encoder.name(), subsamp);

        let total_bytes_atomic = AtomicUsize::new(0);

        // Shared state for collecting size distributions
        struct JpegOnlyWriteState {
            all_l2_sizes: Vec<usize>,
            all_l1_sizes: Vec<usize>,
            all_l0_sizes: Vec<usize>,
            first_error: Option<String>,
        }
        let write_state = Mutex::new(JpegOnlyWriteState {
            all_l2_sizes: Vec::with_capacity(families_to_process),
            all_l1_sizes: Vec::with_capacity(families_to_process * 4),
            all_l0_sizes: Vec::with_capacity(families_to_process * 16),
            first_error: None,
        });

        families.par_iter().for_each(|&(col, row)| {
            {
                let state = write_state.lock().unwrap();
                if state.first_error.is_some() { return; }
            }

            let result = process_family_jpeg_only(
                &slide, col, row, slide_w, slide_h, tile_size,
                encoder.as_ref(), args.baseq, subsamp, save_png,
            );
            let idx = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if idx % 50 == 0 || idx == families_to_process {
                info!("[{}/{}] families processed", idx, families_to_process);
            }

            let write_result = (|| -> Result<()> {
                let family = result?;
                let base_l1_x = family.col * 2;
                let base_l1_y = family.row * 2;
                let base_l0_x = family.col * 4;
                let base_l0_y = family.row * 4;

                // Write L2 tile
                let l2_path = l2_tile_dir.join(format!("{}_{}.jpg", family.col, family.row));
                fs::write(&l2_path, &family.l2_jpeg)?;

                // Write L2 PNG
                if let Some(ref png_data) = family.l2_png {
                    if let Some(ref png_dir) = l2_png_dir {
                        let png_path = png_dir.join(format!("{}_{}.png", family.col, family.row));
                        fs::write(&png_path, png_data)?;
                    }
                }

                // Write L1 tiles
                for (dx, dy, jpeg) in &family.l1_tiles {
                    let path = l1_tile_dir.join(format!("{}_{}.jpg", base_l1_x + dx, base_l1_y + dy));
                    fs::write(&path, jpeg)?;
                }

                // Write L0 tiles
                for (dx, dy, jpeg) in &family.l0_tiles {
                    let path = l0_tile_dir.join(format!("{}_{}.jpg", base_l0_x + dx, base_l0_y + dy));
                    fs::write(&path, jpeg)?;
                }

                total_bytes_atomic.fetch_add(family.total_bytes, Ordering::Relaxed);

                // Collect size stats under lock
                {
                    let mut state = write_state.lock().unwrap();
                    state.all_l2_sizes.push(family.l2_jpeg.len());
                    for (_, _, jpeg) in &family.l1_tiles {
                        state.all_l1_sizes.push(jpeg.len());
                    }
                    for (_, _, jpeg) in &family.l0_tiles {
                        state.all_l0_sizes.push(jpeg.len());
                    }
                }

                Ok(())
            })();

            if let Err(e) = write_result {
                let mut state = write_state.lock().unwrap();
                if state.first_error.is_none() {
                    state.first_error = Some(format!("Family ({},{}): {}", col, row, e));
                }
            }
        });

        // Check for errors
        let state = write_state.into_inner().unwrap();
        if let Some(err_msg) = state.first_error {
            return Err(anyhow::anyhow!("Error during JPEG-only ingest: {}", err_msg));
        }

        let mut all_l2_sizes = state.all_l2_sizes;
        let mut all_l1_sizes = state.all_l1_sizes;
        let mut all_l0_sizes = state.all_l0_sizes;

        // Synthesize lower pyramid levels
        info!("Synthesizing lower pyramid levels (below L2)...");
        synthesize_lower_levels(
            &files_dir, l2_level, l2_cols, l2_rows, tile_size,
            l2_png_dir.as_deref(), encoder.as_ref(), args.baseq, subsamp,
        )?;

        // Write DZI manifest
        let dzi_path = args.out.join("baseline_pyramid.dzi");
        write_dzi_manifest(&dzi_path, slide_w, slide_h, tile_size)?;
        info!("Wrote DZI manifest: {}", dzi_path.display());

        // Clean up temporary PNG directory
        if let Some(ref png_dir) = l2_png_dir {
            info!("Cleaning up temporary L2 PNGs...");
            let _ = fs::remove_dir_all(png_dir);
        }

        let elapsed = start.elapsed();
        let total_bytes = total_bytes_atomic.load(Ordering::Relaxed);
        info!(
            "JPEG-only ingest complete: {} families, total={:.1}MB, {:.1}s",
            families_to_process,
            total_bytes as f64 / 1_048_576.0,
            elapsed.as_secs_f64(),
        );

        // Compute and log tile size statistics
        let l2_stats = compute_size_stats(&mut all_l2_sizes);
        let l1_stats = compute_size_stats(&mut all_l1_sizes);
        let l0_stats = compute_size_stats(&mut all_l0_sizes);

        info!("Tile size distribution (bytes):");
        info!("  L2 (Q{}): count={} avg={} min={} p50={} p95={} p99={} max={}",
            args.baseq,
            l2_stats["count"], l2_stats["avg"], l2_stats["min"],
            l2_stats["median"], l2_stats["p95"], l2_stats["p99"], l2_stats["max"]);
        info!("  L1 (Q{}): count={} avg={} min={} p50={} p95={} p99={} max={}",
            args.baseq,
            l1_stats["count"], l1_stats["avg"], l1_stats["min"],
            l1_stats["median"], l1_stats["p95"], l1_stats["p99"], l1_stats["max"]);
        info!("  L0 (Q{}): count={} avg={} min={} p50={} p95={} p99={} max={}",
            args.baseq,
            l0_stats["count"], l0_stats["avg"], l0_stats["min"],
            l0_stats["median"], l0_stats["p95"], l0_stats["p99"], l0_stats["max"]);

        let summary = serde_json::json!({
            "mode": "ingest-jpeg-only",
            "slide": args.slide.to_string_lossy(),
            "slide_w": slide_w,
            "slide_h": slide_h,
            "encoder": args.encoder,
            "subsamp": subsamp.to_string(),
            "baseq": args.baseq,
            "jpeg_only": true,
            "tile_size": tile_size,
            "families": families_to_process,
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "total_bytes": total_bytes,
            "dzi_max_level": max_level,
            "elapsed_secs": elapsed.as_secs_f64(),
        });
        fs::write(
            args.out.join("summary.json"),
            serde_json::to_string_pretty(&summary)?,
        )?;

        // Write stats.json
        let stats = serde_json::json!({
            "l2_tiles": l2_stats,
            "l1_tiles": l1_stats,
            "l0_tiles": l0_stats,
        });
        fs::write(
            args.out.join("stats.json"),
            serde_json::to_string_pretty(&stats)?,
        )?;
        info!("Wrote stats.json");
    } else {
        // --- Normal mode: L2 tiles + residuals ---
        let do_pack = args.pack || args.bundle;
        let pack_dir = if args.pack {
            let d = args.out.join("residual_packs");
            fs::create_dir_all(&d)?;
            Some(d)
        } else {
            None
        };

        let save_png = args.l2_format == "png";
        let l2_png_dir = if save_png {
            let d = args.out.join("l2_png_tmp");
            fs::create_dir_all(&d)?;
            Some(d)
        } else {
            None
        };

        let total_residual = AtomicUsize::new(0);
        let total_l2 = AtomicUsize::new(0);

        info!("Processing {} families with encoder={} subsamp={} baseq={} l1q={} l0q={} optl2={}",
            families_to_process, encoder.name(), subsamp, args.baseq, args.l1q, args.l0q, args.optl2);

        // Shared state for collecting size distributions (written under lock as each family completes)
        struct WriteState {
            all_l2_sizes: Vec<usize>,
            all_l1_sizes: Vec<usize>,
            all_l0_sizes: Vec<usize>,
            all_pack_sizes: Vec<usize>,
            /// Accumulated (col, row, lz4_compressed_pack_bytes) for bundle writing
            bundle_entries: Vec<(u32, u32, Vec<u8>)>,
            first_error: Option<String>,
        }
        let write_state = Mutex::new(WriteState {
            all_l2_sizes: Vec::with_capacity(families_to_process),
            all_l1_sizes: Vec::with_capacity(families_to_process * 4),
            all_l0_sizes: Vec::with_capacity(families_to_process * 16),
            all_pack_sizes: Vec::with_capacity(families_to_process),
            bundle_entries: if args.bundle { Vec::with_capacity(families_to_process) } else { Vec::new() },
            first_error: None,
        });

        // Process families in parallel, streaming results to disk as they complete
        families.par_iter().for_each(|&(col, row)| {
            // Skip if a previous family had a write error
            {
                let state = write_state.lock().unwrap();
                if state.first_error.is_some() { return; }
            }

            let result = process_family(
                &slide,
                col, row,
                slide_w, slide_h,
                tile_size,
                encoder.as_ref(),
                args.baseq, args.l1q, args.l0q,
                subsamp, args.optl2, args.max_delta,
                save_png, do_pack,
            );
            let idx = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if idx % 50 == 0 || idx == families_to_process {
                info!("[{}/{}] families processed", idx, families_to_process);
            }

            // Write this family's results to disk immediately
            let write_result = (|| -> Result<()> {
                let family = result?;

                // Write L2 JPEG tile
                let l2_path = l2_tile_dir.join(format!("{}_{}.jpg", family.col, family.row));
                total_l2.fetch_add(family.l2_jpeg.len(), Ordering::Relaxed);
                fs::write(&l2_path, &family.l2_jpeg)?;

                // Write L2 PNG (lossless intermediate)
                if let Some(ref png_data) = family.l2_png {
                    if let Some(ref png_dir) = l2_png_dir {
                        let png_path = png_dir.join(format!("{}_{}.png", family.col, family.row));
                        fs::write(&png_path, png_data)?;
                    }
                }

                // Write individual pack files if requested
                if let Some(ref pd) = pack_dir {
                    if !family.pack_entries.is_empty() {
                        write_pack(pd, family.col, family.row, &family.pack_entries)?;
                    }
                }

                total_residual.fetch_add(family.total_residual_bytes, Ordering::Relaxed);

                // Collect size stats and bundle data under lock
                {
                    let mut state = write_state.lock().unwrap();
                    state.all_l2_sizes.push(family.l2_jpeg.len());
                    state.all_l1_sizes.extend_from_slice(&family.l1_residual_sizes);
                    state.all_l0_sizes.extend_from_slice(&family.l0_residual_sizes);
                    if do_pack && !family.pack_entries.is_empty() {
                        let pack_size: usize = family.pack_entries.iter().map(|e| e.jpeg_data.len()).sum();
                        let pack_total = pack_size + 8 + family.pack_entries.len() * 4;
                        state.all_pack_sizes.push(pack_total);
                    }
                    // Accumulate compressed pack bytes for bundle
                    if args.bundle && !family.pack_entries.is_empty() {
                        let lz4_data = compress_pack_entries(&family.pack_entries);
                        state.bundle_entries.push((family.col, family.row, lz4_data));
                    }
                }

                Ok(())
            })();

            if let Err(e) = write_result {
                let mut state = write_state.lock().unwrap();
                if state.first_error.is_none() {
                    state.first_error = Some(format!("Family ({},{}): {}", col, row, e));
                }
            }
        });

        // Check for errors
        let state = write_state.into_inner().unwrap();
        if let Some(err_msg) = state.first_error {
            return Err(anyhow::anyhow!("Error during ingest: {}", err_msg));
        }

        let (mut all_l2_sizes, mut all_l1_sizes, mut all_l0_sizes, mut all_pack_sizes) =
            (state.all_l2_sizes, state.all_l1_sizes, state.all_l0_sizes, state.all_pack_sizes);
        let bundle_entries = state.bundle_entries;

        // Write bundle file if requested
        if args.bundle && !bundle_entries.is_empty() {
            let bundle_dir = args.out.join("residual_packs");
            fs::create_dir_all(&bundle_dir)?;
            let bundle_path = bundle_dir.join("residuals.bundle");
            write_bundle(
                &bundle_path,
                grid_cols as u16,
                grid_rows as u16,
                &bundle_entries,
            )?;
            let bundle_size: usize = fs::metadata(&bundle_path)?.len() as usize;
            info!(
                "Wrote bundle: {} ({:.1} MB, {} families)",
                bundle_path.display(),
                bundle_size as f64 / 1_048_576.0,
                bundle_entries.len(),
            );
        }

        // L1 and L0 level directories (empty — tiles are reconstructed on demand)
        // The serve command reconstructs them from L2 + packs
        let l1_level = max_level.saturating_sub(1);
        fs::create_dir_all(files_dir.join(max_level.to_string()))?;
        fs::create_dir_all(files_dir.join(l1_level.to_string()))?;

        // Synthesize lower pyramid levels (below L2) from L2 tiles
        info!("Synthesizing lower pyramid levels (below L2)...");
        synthesize_lower_levels(
            &files_dir, l2_level, l2_cols, l2_rows, tile_size,
            l2_png_dir.as_deref(), encoder.as_ref(), args.baseq, subsamp,
        )?;

        // Write DZI manifest
        let dzi_path = args.out.join("baseline_pyramid.dzi");
        write_dzi_manifest(&dzi_path, slide_w, slide_h, tile_size)?;
        info!("Wrote DZI manifest: {}", dzi_path.display());

        // Clean up temporary PNG directory
        if let Some(ref png_dir) = l2_png_dir {
            info!("Cleaning up temporary L2 PNGs...");
            let _ = fs::remove_dir_all(png_dir);
        }

        let elapsed = start.elapsed();
        let residual_total = total_residual.load(Ordering::Relaxed);
        let l2_total = total_l2.load(Ordering::Relaxed);

        info!(
            "Ingest complete: {} families, L2={:.1}MB, residuals={:.1}MB, {:.1}s",
            families_to_process,
            l2_total as f64 / 1_048_576.0,
            residual_total as f64 / 1_048_576.0,
            elapsed.as_secs_f64(),
        );

        // Compute and log tile size statistics
        let l2_stats = compute_size_stats(&mut all_l2_sizes);
        let l1_stats = compute_size_stats(&mut all_l1_sizes);
        let l0_stats = compute_size_stats(&mut all_l0_sizes);
        let pack_stats = compute_size_stats(&mut all_pack_sizes);

        info!("Tile size distribution (bytes):");
        info!("  L2 (Q{}): count={} avg={} min={} p50={} p95={} p99={} max={}",
            args.baseq,
            l2_stats["count"], l2_stats["avg"], l2_stats["min"],
            l2_stats["median"], l2_stats["p95"], l2_stats["p99"], l2_stats["max"]);
        info!("  L1 residuals (Q{}): count={} avg={} min={} p50={} p95={} p99={} max={}",
            args.l1q,
            l1_stats["count"], l1_stats["avg"], l1_stats["min"],
            l1_stats["median"], l1_stats["p95"], l1_stats["p99"], l1_stats["max"]);
        info!("  L0 residuals (Q{}): count={} avg={} min={} p50={} p95={} p99={} max={}",
            args.l0q,
            l0_stats["count"], l0_stats["avg"], l0_stats["min"],
            l0_stats["median"], l0_stats["p95"], l0_stats["p99"], l0_stats["max"]);
        if !all_pack_sizes.is_empty() {
            info!("  Packs: count={} avg={} min={} p50={} p95={} p99={} max={}",
                pack_stats["count"], pack_stats["avg"], pack_stats["min"],
                pack_stats["median"], pack_stats["p95"], pack_stats["p99"], pack_stats["max"]);
        }

        // Write summary.json
        let summary = serde_json::json!({
            "mode": "ingest",
            "slide": args.slide.to_string_lossy(),
            "slide_w": slide_w,
            "slide_h": slide_h,
            "encoder": args.encoder,
            "subsamp": subsamp.to_string(),
            "baseq": args.baseq,
            "l1q": args.l1q,
            "l0q": args.l0q,
            "optl2": args.optl2,
            "tile_size": tile_size,
            "families": families_to_process,
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "l2_bytes": l2_total,
            "residual_bytes": residual_total,
            "pack": args.pack,
            "bundle": args.bundle,
            "dzi_max_level": max_level,
            "elapsed_secs": elapsed.as_secs_f64(),
        });
        fs::write(
            args.out.join("summary.json"),
            serde_json::to_string_pretty(&summary)?,
        )?;

        // Write stats.json with full distributions
        let stats = serde_json::json!({
            "l2_tiles": l2_stats,
            "l1_residuals": l1_stats,
            "l0_residuals": l0_stats,
            "packs": pack_stats,
        });
        fs::write(
            args.out.join("stats.json"),
            serde_json::to_string_pretty(&stats)?,
        )?;
        info!("Wrote stats.json");
    }

    Ok(())
}
