use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use tracing::info;

use crate::core::color::ycbcr_planes_from_rgb;
use crate::core::jpeg::create_encoder;
use crate::core::pack::{write_pack, PackWriteEntry};
use crate::core::pyramid::{discover_pyramid, extract_tile_plane, parse_tile_coords};
use crate::core::residual::{center_residual, compute_residual};
use crate::core::upsample::upsample_2x_channel;
use crate::turbojpeg_optimized::load_rgb_turbo;

/// Decode grayscale residual bytes, dispatching to the correct codec.
fn decode_residual_bytes(data: &[u8]) -> Result<Vec<u8>> {
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
    #[arg(long)]
    pub pyramid: PathBuf,

    /// Output directory for residuals and pack files
    #[arg(long)]
    pub out: PathBuf,

    /// Tile size (must match pyramid)
    #[arg(long, default_value_t = 256)]
    pub tile: u32,

    /// JPEG quality for residual encoding (1-100)
    #[arg(long, default_value_t = 50)]
    pub resq: u8,

    /// Encoder backend: turbojpeg, mozjpeg, jpegli, jpegxl
    #[arg(long, default_value = "turbojpeg")]
    pub encoder: String,

    /// Maximum number of L2 parent tiles to process (for testing)
    #[arg(long)]
    pub max_parents: Option<usize>,

    /// Also create pack files
    #[arg(long)]
    pub pack: bool,

}

pub fn run(args: EncodeArgs) -> Result<()> {
    let start = Instant::now();
    let encoder = create_encoder(&args.encoder)?;
    info!("Using encoder: {}", encoder.name());

    let pyramid = discover_pyramid(&args.pyramid, args.tile)?;
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
    let resq = args.resq;
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
                let jpeg_data = encoder.encode_gray(&centered, tile_size, tile_size, resq)?;
                total_l1 += 1;
                total_bytes += jpeg_data.len();

                // Write residual file
                let out_path = l1_parent_dir.join(format!("{}_{}.jpg", x1, y1));
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
        // L0 chroma: computed here for future eval use but not needed for residual encoding
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

                let jpeg_data = encoder.encode_gray(&centered, tile_size, tile_size, resq)?;
                total_l0 += 1;
                total_bytes += jpeg_data.len();

                let out_path = l0_parent_dir.join(format!("{}_{}.jpg", x0, y0));
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
            // Count how many L1/L0 entries were in this parent
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
        "resq": args.resq,
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
