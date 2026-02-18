use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Args;
use tracing::info;

use crate::core::pack::open_bundle;
use crate::core::pyramid::{discover_pyramid, parse_tile_coords};
use crate::core::reconstruct::{
    BufferPool, OutputFormat, ReconstructInput, ReconstructOpts, reconstruct_family, write_family_tiles,
};

#[derive(Args, Debug)]
pub struct DecodeArgs {
    /// Path to DZI pyramid directory (containing baseline_pyramid.dzi)
    #[arg(long)]
    pyramid: PathBuf,

    /// Output directory for reconstructed tiles
    #[arg(long)]
    out: PathBuf,

    /// Directory with loose residuals: L1/{x2}_{y2}/, L0/{x2}_{y2}/
    #[arg(long)]
    residuals: Option<PathBuf>,

    /// Directory with {x2}_{y2}.pack files
    #[arg(long)]
    packs: Option<PathBuf>,

    /// Single .pack file to decode
    #[arg(long)]
    pack_file: Option<PathBuf>,

    /// Bundle file (.bundle) containing all residual packs
    #[arg(long)]
    bundle: Option<PathBuf>,

    /// Output JPEG quality (1-100)
    #[arg(long, default_value_t = 95)]
    quality: u8,

    /// Tile size (must match pyramid)
    #[arg(long, default_value_t = 256)]
    tile: u32,

    /// Maximum number of L2 parents to process
    #[arg(long)]
    max_parents: Option<usize>,

    /// Output grayscale tiles (for debugging)
    #[arg(long, default_value_t = false)]
    grayscale: bool,

    /// Print per-family timing breakdown
    #[arg(long, default_value_t = false)]
    timing: bool,

    /// Output format for reconstructed tiles: jpeg or webp
    #[arg(long, default_value = "jpeg")]
    output_format: String,
}

pub fn run(args: DecodeArgs) -> Result<()> {
    let start = Instant::now();

    let output_format = match args.output_format.as_str() {
        "jpeg" | "jpg" => OutputFormat::Jpeg,
        "webp" => OutputFormat::Webp,
        other => return Err(anyhow!("unknown output format: '{}'. Available: jpeg, webp", other)),
    };

    // Validate: exactly one residual source must be provided
    let source_count = args.residuals.is_some() as u8
        + args.packs.is_some() as u8
        + args.pack_file.is_some() as u8
        + args.bundle.is_some() as u8;
    if source_count != 1 {
        return Err(anyhow!(
            "Exactly one of --residuals, --packs, --pack-file, or --bundle must be provided"
        ));
    }

    // Discover pyramid
    let pyramid = discover_pyramid(&args.pyramid, args.tile)?;
    info!(
        "Pyramid: max_level={} tile_size={} l0={} l1={} l2={}",
        pyramid.max_level, pyramid.tile_size, pyramid.l0, pyramid.l1, pyramid.l2
    );

    // Open bundle if specified
    let bundle_file = if let Some(ref bundle_path) = args.bundle {
        Some(open_bundle(bundle_path)?)
    } else {
        None
    };

    // Discover parents based on source type
    let (parents, residuals_dir, pack_dir) = if let Some(ref bundle) = bundle_file {
        // Bundle file: enumerate all non-empty families from index
        let mut parents = Vec::new();
        for row in 0..bundle.grid_rows() {
            for col in 0..bundle.grid_cols() {
                // Check if family has data by trying to get its pack
                if bundle.get_pack(col as u32, row as u32).is_ok() {
                    parents.push((col as u32, row as u32));
                }
            }
        }
        parents.sort();
        (parents, None, None)
    } else if let Some(ref pack_file) = args.pack_file {
        // Single pack file: parse coordinates from filename
        let fname = pack_file
            .file_name()
            .ok_or_else(|| anyhow!("invalid pack-file path"))?
            .to_string_lossy();
        let stem = fname.strip_suffix(".pack")
            .ok_or_else(|| anyhow!("pack-file must have .pack extension"))?;
        let coords = parse_tile_coords(&format!("{}.jpg", stem))
            .ok_or_else(|| anyhow!("cannot parse coordinates from pack-file name: {}", fname))?;

        // The pack dir is the parent directory of the pack file
        let pack_parent = pack_file
            .parent()
            .ok_or_else(|| anyhow!("pack-file has no parent directory"))?;

        (vec![coords], None, Some(pack_parent.to_path_buf()))
    } else if let Some(ref packs_dir) = args.packs {
        // Directory of pack files: glob *.pack
        let mut parents = Vec::new();
        for entry in fs::read_dir(packs_dir)
            .with_context(|| format!("reading packs dir {}", packs_dir.display()))?
        {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".pack") {
                let stem = name.strip_suffix(".pack").unwrap();
                if let Some(coords) = parse_tile_coords(&format!("{}.jpg", stem)) {
                    parents.push(coords);
                }
            }
        }
        parents.sort();
        (parents, None, Some(packs_dir.clone()))
    } else if let Some(ref residuals) = args.residuals {
        // Loose residuals dir: scan L1/ subdirs named {x2}_{y2}
        let l1_dir = residuals.join("L1");
        let mut parents = Vec::new();
        if l1_dir.exists() {
            for entry in fs::read_dir(&l1_dir)
                .with_context(|| format!("reading L1 residuals dir {}", l1_dir.display()))?
            {
                let entry = entry?;
                if !entry.file_type()?.is_dir() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                // Parse directory name like "3_2"
                let mut parts = name.split('_');
                if let (Some(x_str), Some(y_str)) = (parts.next(), parts.next()) {
                    if let (Ok(x), Ok(y)) = (x_str.parse::<u32>(), y_str.parse::<u32>()) {
                        parents.push((x, y));
                    }
                }
            }
        }
        parents.sort();
        (parents, Some(residuals.clone()), None)
    } else {
        unreachable!()
    };

    info!("Found {} L2 parents to decode", parents.len());

    let mut parents = parents;
    if let Some(max) = args.max_parents {
        parents.truncate(max);
        info!("Limited to {} parents (--max-parents)", max);
    }

    // Create output directory
    fs::create_dir_all(&args.out)?;

    // Create buffer pool
    let buffer_pool = BufferPool::new(128);

    let mut total_l1 = 0usize;
    let mut total_l0 = 0usize;

    for (pi, &(x2, y2)) in parents.iter().enumerate() {
        let family_start = Instant::now();

        let input = ReconstructInput {
            files_dir: &pyramid.files_dir,
            residuals_dir: residuals_dir.as_deref(),
            pack_dir: pack_dir.as_deref(),
            bundle: bundle_file.as_ref(),
            tile_size: pyramid.tile_size,
            l0: pyramid.l0,
            l1: pyramid.l1,
            l2: pyramid.l2,
        };
        let opts = ReconstructOpts {
            quality: args.quality,
            timing: args.timing,
            grayscale_only: args.grayscale,
            output_format,
            sharpen: None,
        };

        let result = reconstruct_family(&input, x2, y2, &opts, &buffer_pool)?;

        let l1_count = result.l1.len();
        let l0_count = result.l0.len();
        total_l1 += l1_count;
        total_l0 += l0_count;

        // Write tiles to disk
        write_family_tiles(&result, &args.out, output_format)?;

        let family_ms = family_start.elapsed().as_millis();

        if let Some(ref stats) = result.stats {
            info!(
                "family_breakdown x2={} y2={} l2_decode={}ms chroma={}ms l1_res={}ms l1_enc={}ms l0_resize={}ms l0_res={}ms l0_enc={}ms total={}ms l1_residuals={} l0_residuals={} l1_par={} l0_par={}",
                x2, y2,
                stats.l2_decode_ms, stats.l1_resize_ms,
                stats.l1_residual_ms, stats.l1_encode_ms,
                stats.l0_resize_ms, stats.l0_residual_ms, stats.l0_encode_ms,
                stats.total_ms, stats.residuals_l1, stats.residuals_l0,
                stats.l1_parallel_max, stats.l0_parallel_max
            );
        }

        info!(
            "[{}/{}] L2 parent ({},{}) — L1: {} tiles, L0: {} tiles — {}ms",
            pi + 1,
            parents.len(),
            x2, y2,
            l1_count, l0_count,
            family_ms
        );
    }

    let elapsed = start.elapsed();
    info!(
        "Decode complete: {} L1 + {} L0 tiles from {} parents, {:.1}s",
        total_l1, total_l0, parents.len(), elapsed.as_secs_f64()
    );

    // Write summary.json
    let summary = serde_json::json!({
        "quality": args.quality,
        "tile_size": pyramid.tile_size,
        "l1_tiles": total_l1,
        "l0_tiles": total_l0,
        "parents": parents.len(),
        "grayscale": args.grayscale,
        "elapsed_secs": elapsed.as_secs_f64(),
    });
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn test_data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data").join("demo_out")
    }

    fn has_test_data() -> bool {
        test_data_dir().join("baseline_pyramid.dzi").exists()
    }

    // -----------------------------------------------------------------------
    // Input validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_rejects_no_source() {
        let out_dir = std::env::temp_dir().join("origami_test_decode_no_src");
        let _ = fs::remove_dir_all(&out_dir);
        let args = DecodeArgs {
            pyramid: test_data_dir(),
            out: out_dir.clone(),
            residuals: None,
            packs: None,
            pack_file: None,
            quality: 90,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        let result = run(args);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("Exactly one"),
            "Should reject missing source"
        );
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn test_decode_rejects_multiple_sources() {
        let out_dir = std::env::temp_dir().join("origami_test_decode_multi_src");
        let _ = fs::remove_dir_all(&out_dir);
        let args = DecodeArgs {
            pyramid: test_data_dir(),
            out: out_dir.clone(),
            residuals: Some(PathBuf::from("/tmp/r")),
            packs: Some(PathBuf::from("/tmp/p")),
            pack_file: None,
            quality: 90,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        let result = run(args);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("Exactly one"),
            "Should reject multiple sources"
        );
        let _ = fs::remove_dir_all(&out_dir);
    }

    // -----------------------------------------------------------------------
    // End-to-end: decode from loose residuals
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_from_residuals() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let out_dir = std::env::temp_dir().join("origami_test_decode_residuals");
        let _ = fs::remove_dir_all(&out_dir);

        let args = DecodeArgs {
            pyramid: test_data_dir(),
            out: out_dir.clone(),
            residuals: Some(test_data_dir().join("residuals_q32")),
            packs: None,
            pack_file: None,
            quality: 90,
            tile: 256,
            max_parents: Some(1),
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        run(args).unwrap();

        // Verify output structure
        assert!(out_dir.join("L1").is_dir(), "L1 dir should exist");
        assert!(out_dir.join("L0").is_dir(), "L0 dir should exist");
        assert!(out_dir.join("summary.json").exists(), "summary.json should exist");

        // Verify tile counts
        let l1_count = fs::read_dir(out_dir.join("L1")).unwrap().count();
        let l0_count = fs::read_dir(out_dir.join("L0")).unwrap().count();
        assert_eq!(l1_count, 4, "Should have 4 L1 tiles");
        assert_eq!(l0_count, 16, "Should have 16 L0 tiles");

        // Verify specific tile files exist with correct names
        assert!(out_dir.join("L1/100_100.jpg").exists());
        assert!(out_dir.join("L1/101_101.jpg").exists());
        assert!(out_dir.join("L0/200_200.jpg").exists());
        assert!(out_dir.join("L0/203_203.jpg").exists());

        // Verify tiles are valid JPEGs with substantial content
        let tile_data = fs::read(out_dir.join("L1/100_100.jpg")).unwrap();
        assert!(tile_data.len() > 100, "Tile should have substantial content");
        assert_eq!(&tile_data[0..2], &[0xFF, 0xD8], "Should be valid JPEG");

        // Verify summary.json content
        let summary: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out_dir.join("summary.json")).unwrap()).unwrap();
        assert_eq!(summary["l1_tiles"], 4);
        assert_eq!(summary["l0_tiles"], 16);
        assert_eq!(summary["quality"], 90);
        assert_eq!(summary["parents"], 1);
        assert_eq!(summary["grayscale"], false);

        let _ = fs::remove_dir_all(&out_dir);
    }

    // -----------------------------------------------------------------------
    // End-to-end: decode from packs directory
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_from_packs() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let out_dir = std::env::temp_dir().join("origami_test_decode_packs");
        let _ = fs::remove_dir_all(&out_dir);

        let args = DecodeArgs {
            pyramid: test_data_dir(),
            out: out_dir.clone(),
            residuals: None,
            packs: Some(test_data_dir().join("residual_packs")),
            pack_file: None,
            quality: 95,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        run(args).unwrap();

        let l1_count = fs::read_dir(out_dir.join("L1")).unwrap().count();
        let l0_count = fs::read_dir(out_dir.join("L0")).unwrap().count();
        assert_eq!(l1_count, 4);
        assert_eq!(l0_count, 16);

        let _ = fs::remove_dir_all(&out_dir);
    }

    // -----------------------------------------------------------------------
    // End-to-end: decode from single pack file
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_from_single_pack_file() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let out_dir = std::env::temp_dir().join("origami_test_decode_single_pack");
        let _ = fs::remove_dir_all(&out_dir);

        let args = DecodeArgs {
            pyramid: test_data_dir(),
            out: out_dir.clone(),
            residuals: None,
            packs: None,
            pack_file: Some(test_data_dir().join("residual_packs/50_50.pack")),
            quality: 90,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        run(args).unwrap();

        let l1_count = fs::read_dir(out_dir.join("L1")).unwrap().count();
        let l0_count = fs::read_dir(out_dir.join("L0")).unwrap().count();
        assert_eq!(l1_count, 4);
        assert_eq!(l0_count, 16);

        let _ = fs::remove_dir_all(&out_dir);
    }

    // -----------------------------------------------------------------------
    // Decode with timing flag
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_with_timing() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let out_dir = std::env::temp_dir().join("origami_test_decode_timing");
        let _ = fs::remove_dir_all(&out_dir);

        let args = DecodeArgs {
            pyramid: test_data_dir(),
            out: out_dir.clone(),
            residuals: Some(test_data_dir().join("residuals_q32")),
            packs: None,
            pack_file: None,
            quality: 90,
            tile: 256,
            max_parents: Some(1),
            grayscale: false,
            timing: true,
            output_format: "jpeg".to_string(),
        };
        // Should not panic and should complete successfully
        run(args).unwrap();

        assert!(out_dir.join("summary.json").exists());

        let _ = fs::remove_dir_all(&out_dir);
    }

    // -----------------------------------------------------------------------
    // Decode with grayscale flag
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_with_grayscale() {
        if !has_test_data() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let out_dir = std::env::temp_dir().join("origami_test_decode_grayscale");
        let _ = fs::remove_dir_all(&out_dir);

        let args = DecodeArgs {
            pyramid: test_data_dir(),
            out: out_dir.clone(),
            residuals: Some(test_data_dir().join("residuals_q32")),
            packs: None,
            pack_file: None,
            quality: 90,
            tile: 256,
            max_parents: Some(1),
            grayscale: true,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        run(args).unwrap();

        // Grayscale JPEGs should be smaller than color equivalents
        let gs_tile = fs::read(out_dir.join("L1/100_100.jpg")).unwrap();
        assert_eq!(&gs_tile[0..2], &[0xFF, 0xD8], "Grayscale tile should be valid JPEG");

        let summary: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out_dir.join("summary.json")).unwrap()).unwrap();
        assert_eq!(summary["grayscale"], true);

        let _ = fs::remove_dir_all(&out_dir);
    }

    // -----------------------------------------------------------------------
    // Full encode → decode round-trip from a clean 1024x1024 source image
    // -----------------------------------------------------------------------

    /// Build a synthetic 3-level DZI pyramid from a 1024x1024 source image:
    ///   Level 2 (L0): 1024x1024 → 4x4 tiles
    ///   Level 1 (L1): 512x512  → 2x2 tiles
    ///   Level 0 (L2): 256x256  → 1x1 tile
    fn build_synthetic_pyramid(src_path: &Path, out_dir: &Path) {
        use crate::turbojpeg_optimized::{load_rgb_turbo, encode_jpeg_turbo};

        fs::create_dir_all(out_dir).unwrap();

        // Load 1024x1024 source
        let (rgb, w, h) = load_rgb_turbo(src_path).unwrap();
        assert_eq!(w, 1024);
        assert_eq!(h, 1024);

        // Write DZI manifest
        let dzi = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpg"
  Overlap="0"
  TileSize="256"
  >
  <Size
    Height="{}"
    Width="{}"
  />
</Image>"#,
            h, w
        );
        fs::write(out_dir.join("baseline_pyramid.dzi"), &dzi).unwrap();

        let files_dir = out_dir.join("baseline_pyramid_files");

        // Level 2 (L0 = highest res): 1024x1024, 4x4 tiles of 256x256
        let l0_dir = files_dir.join("2");
        fs::create_dir_all(&l0_dir).unwrap();
        for ty in 0..4u32 {
            for tx in 0..4u32 {
                let mut tile = vec![0u8; 256 * 256 * 3];
                for y in 0..256u32 {
                    for x in 0..256u32 {
                        let src_x = tx * 256 + x;
                        let src_y = ty * 256 + y;
                        let si = ((src_y * w + src_x) * 3) as usize;
                        let di = ((y * 256 + x) * 3) as usize;
                        tile[di] = rgb[si];
                        tile[di + 1] = rgb[si + 1];
                        tile[di + 2] = rgb[si + 2];
                    }
                }
                let jpeg = encode_jpeg_turbo(&tile, 256, 256, 95).unwrap();
                fs::write(l0_dir.join(format!("{}_{}.jpg", tx, ty)), &jpeg).unwrap();
            }
        }

        // Downsample 1024→512 with 2:1 box averaging
        let mut rgb_512 = vec![0u8; 512 * 512 * 3];
        for y in 0..512u32 {
            for x in 0..512u32 {
                for c in 0..3 {
                    let i00 = ((y * 2 * w + x * 2) * 3 + c) as usize;
                    let i10 = ((y * 2 * w + x * 2 + 1) * 3 + c) as usize;
                    let i01 = (((y * 2 + 1) * w + x * 2) * 3 + c) as usize;
                    let i11 = (((y * 2 + 1) * w + x * 2 + 1) * 3 + c) as usize;
                    let avg = ((rgb[i00] as u32 + rgb[i10] as u32 + rgb[i01] as u32 + rgb[i11] as u32) / 4) as u8;
                    rgb_512[((y * 512 + x) * 3 + c) as usize] = avg;
                }
            }
        }

        // Level 1 (L1): 512x512, 2x2 tiles of 256x256
        let l1_dir = files_dir.join("1");
        fs::create_dir_all(&l1_dir).unwrap();
        for ty in 0..2u32 {
            for tx in 0..2u32 {
                let mut tile = vec![0u8; 256 * 256 * 3];
                for y in 0..256u32 {
                    for x in 0..256u32 {
                        let src_x = tx * 256 + x;
                        let src_y = ty * 256 + y;
                        let si = ((src_y * 512 + src_x) * 3) as usize;
                        let di = ((y * 256 + x) * 3) as usize;
                        tile[di] = rgb_512[si];
                        tile[di + 1] = rgb_512[si + 1];
                        tile[di + 2] = rgb_512[si + 2];
                    }
                }
                let jpeg = encode_jpeg_turbo(&tile, 256, 256, 95).unwrap();
                fs::write(l1_dir.join(format!("{}_{}.jpg", tx, ty)), &jpeg).unwrap();
            }
        }

        // Downsample 512→256
        let mut rgb_256 = vec![0u8; 256 * 256 * 3];
        for y in 0..256u32 {
            for x in 0..256u32 {
                for c in 0..3 {
                    let i00 = ((y * 2 * 512 + x * 2) * 3 + c) as usize;
                    let i10 = ((y * 2 * 512 + x * 2 + 1) * 3 + c) as usize;
                    let i01 = (((y * 2 + 1) * 512 + x * 2) * 3 + c) as usize;
                    let i11 = (((y * 2 + 1) * 512 + x * 2 + 1) * 3 + c) as usize;
                    let avg = ((rgb_512[i00] as u32 + rgb_512[i10] as u32 + rgb_512[i01] as u32 + rgb_512[i11] as u32) / 4) as u8;
                    rgb_256[((y * 256 + x) * 3 + c) as usize] = avg;
                }
            }
        }

        // Level 0 (L2): 256x256, 1x1 tile
        let l2_dir = files_dir.join("0");
        fs::create_dir_all(&l2_dir).unwrap();
        let jpeg = encode_jpeg_turbo(&rgb_256, 256, 256, 95).unwrap();
        fs::write(l2_dir.join("0_0.jpg"), &jpeg).unwrap();
    }

    #[test]
    fn test_encode_decode_roundtrip_from_source_image() {
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data").join("L0-1024.jpg");
        if !src_path.exists() {
            eprintln!("Skipping: test-data/L0-1024.jpg not found");
            return;
        }

        let tmp = std::env::temp_dir().join("origami_test_encode_decode_roundtrip");
        let _ = fs::remove_dir_all(&tmp);

        let pyramid_dir = tmp.join("pyramid");
        let encode_out = tmp.join("encoded");
        let decode_out = tmp.join("decoded");

        // Step 1: Build synthetic pyramid
        build_synthetic_pyramid(&src_path, &pyramid_dir);

        // Verify pyramid structure
        assert!(pyramid_dir.join("baseline_pyramid.dzi").exists());
        assert!(pyramid_dir.join("baseline_pyramid_files/0/0_0.jpg").exists());
        assert!(pyramid_dir.join("baseline_pyramid_files/1/0_0.jpg").exists());
        assert!(pyramid_dir.join("baseline_pyramid_files/1/1_1.jpg").exists());
        assert!(pyramid_dir.join("baseline_pyramid_files/2/0_0.jpg").exists());
        assert!(pyramid_dir.join("baseline_pyramid_files/2/3_3.jpg").exists());

        // Step 2: Encode residuals + pack
        let encode_args = crate::encode::EncodeArgs {
            pyramid: Some(pyramid_dir.clone()),
            image: None,
            out: encode_out.clone(),
            tile: 256,
            resq: 50,
            l1q: None,
            l0q: None,
            baseq: 95,
            subsamp: "444".to_string(),
            encoder: "turbojpeg".to_string(),
            max_parents: None,
            pack: true,
            manifest: false,
            optl2: false,
            debug_images: false,

            max_delta: 15,
            sharpen: None,
            save_sharpened: false,
        };
        crate::encode::run(encode_args).unwrap();

        // Verify encode output
        assert!(encode_out.join("L1/0_0").is_dir(), "L1 residuals should exist");
        assert!(encode_out.join("L0/0_0").is_dir(), "L0 residuals should exist");
        assert!(encode_out.join("packs/0_0.pack").exists(), "Pack file should exist");
        assert!(encode_out.join("summary.json").exists());

        let encode_summary: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(encode_out.join("summary.json")).unwrap()).unwrap();
        assert_eq!(encode_summary["l1_residuals"], 4, "Should have 4 L1 residuals");
        assert_eq!(encode_summary["l0_residuals"], 16, "Should have 16 L0 residuals");
        assert_eq!(encode_summary["pack"], true);

        // Step 3a: Decode from loose residuals
        let decode_args = DecodeArgs {
            pyramid: pyramid_dir.clone(),
            out: decode_out.clone(),
            residuals: Some(encode_out.clone()),
            packs: None,
            pack_file: None,
            quality: 95,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: true,
            output_format: "jpeg".to_string(),
        };
        run(decode_args).unwrap();

        // Verify decode output from residuals
        let l1_count = fs::read_dir(decode_out.join("L1")).unwrap().count();
        let l0_count = fs::read_dir(decode_out.join("L0")).unwrap().count();
        assert_eq!(l1_count, 4, "Decode should produce 4 L1 tiles");
        assert_eq!(l0_count, 16, "Decode should produce 16 L0 tiles");

        // Verify tiles are valid JPEGs
        for ty in 0..2u32 {
            for tx in 0..2u32 {
                let path = decode_out.join(format!("L1/{}_{}.jpg", tx, ty));
                assert!(path.exists(), "L1 tile ({},{}) should exist", tx, ty);
                let data = fs::read(&path).unwrap();
                assert_eq!(&data[0..2], &[0xFF, 0xD8], "L1 tile should be valid JPEG");
                assert!(data.len() > 1000, "L1 tile should have substantial content");
            }
        }
        for ty in 0..4u32 {
            for tx in 0..4u32 {
                let path = decode_out.join(format!("L0/{}_{}.jpg", tx, ty));
                assert!(path.exists(), "L0 tile ({},{}) should exist", tx, ty);
                let data = fs::read(&path).unwrap();
                assert_eq!(&data[0..2], &[0xFF, 0xD8]);
            }
        }

        let decode_summary: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(decode_out.join("summary.json")).unwrap()).unwrap();
        assert_eq!(decode_summary["l1_tiles"], 4);
        assert_eq!(decode_summary["l0_tiles"], 16);

        // Step 3b: Decode from pack file
        let decode_pack_out = tmp.join("decoded_pack");
        let decode_pack_args = DecodeArgs {
            pyramid: pyramid_dir.clone(),
            out: decode_pack_out.clone(),
            residuals: None,
            packs: None,
            pack_file: Some(encode_out.join("packs/0_0.pack")),
            quality: 95,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        run(decode_pack_args).unwrap();

        let l1_pack = fs::read_dir(decode_pack_out.join("L1")).unwrap().count();
        let l0_pack = fs::read_dir(decode_pack_out.join("L0")).unwrap().count();
        assert_eq!(l1_pack, 4, "Pack decode should produce 4 L1 tiles");
        assert_eq!(l0_pack, 16, "Pack decode should produce 16 L0 tiles");

        // Step 4: Compare reconstructed tiles against original baseline tiles.
        // After encode→decode round-trip, the reconstructed tile should be
        // close to the original (PSNR > 30 dB, accounting for two JPEG cycles).
        use crate::turbojpeg_optimized::load_rgb_turbo;
        for ty in 0..4u32 {
            for tx in 0..4u32 {
                let original_path = pyramid_dir.join(format!("baseline_pyramid_files/2/{}_{}.jpg", tx, ty));
                let decoded_path = decode_out.join(format!("L0/{}_{}.jpg", tx, ty));
                let (orig_rgb, ow, oh) = load_rgb_turbo(&original_path).unwrap();
                let (dec_rgb, dw, dh) = load_rgb_turbo(&decoded_path).unwrap();
                assert_eq!((ow, oh), (dw, dh), "Tile ({},{}) dimensions should match", tx, ty);

                // Compute PSNR between original and reconstructed
                let n = orig_rgb.len();
                let mut mse_sum: f64 = 0.0;
                for i in 0..n {
                    let diff = orig_rgb[i] as f64 - dec_rgb[i] as f64;
                    mse_sum += diff * diff;
                }
                let mse = mse_sum / n as f64;
                let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 100.0 };
                assert!(
                    psnr > 30.0,
                    "L0 tile ({},{}) PSNR {:.1} dB is too low (expected >30 dB)",
                    tx, ty, psnr
                );
            }
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    // -----------------------------------------------------------------------
    // Visual HTML report: encode → decode pipeline validation
    // -----------------------------------------------------------------------

    struct TileInfo {
        path: String,       // relative path for HTML src
        file_size: u64,
        is_valid_jpeg: bool,
    }

    struct ReportData {
        timestamp: String,
        encode_ms: u128,
        decode_residuals_ms: u128,
        decode_packs_ms: u128,
        source: TileInfo,
        l2_baseline: TileInfo,
        l1_baselines: Vec<TileInfo>,
        l0_baselines: Vec<TileInfo>,
        l1_residuals: Vec<TileInfo>,
        l0_residuals: Vec<TileInfo>,
        pack_size: u64,
        decode_from_residuals_ok: bool,
        decode_from_packs_ok: bool,
        checks: Vec<(String, bool)>,
    }

    fn collect_tile(base_dir: &Path, rel_path: &str) -> TileInfo {
        let full = base_dir.join(rel_path);
        let file_size = fs::metadata(&full).map(|m| m.len()).unwrap_or(0);
        let is_valid_jpeg = fs::read(&full)
            .map(|d| d.len() >= 2 && d[0] == 0xFF && d[1] == 0xD8)
            .unwrap_or(false);
        TileInfo {
            path: rel_path.to_string(),
            file_size,
            is_valid_jpeg,
        }
    }

    fn fmt_size(bytes: u64) -> String {
        if bytes >= 1_048_576 {
            format!("{:.1} MB", bytes as f64 / 1_048_576.0)
        } else if bytes >= 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else {
            format!("{} B", bytes)
        }
    }

    fn render_report_html(data: &ReportData) -> String {
        let overall_pass = data.checks.iter().all(|(_, ok)| *ok);
        let badge = if overall_pass {
            r#"<span class="badge pass">PASS</span>"#
        } else {
            r#"<span class="badge fail">FAIL</span>"#
        };

        let total_originals: u64 = data.source.file_size
            + data.l2_baseline.file_size
            + data.l1_baselines.iter().map(|t| t.file_size).sum::<u64>()
            + data.l0_baselines.iter().map(|t| t.file_size).sum::<u64>();
        let l1_l0_originals: u64 = data.l1_baselines.iter().map(|t| t.file_size).sum::<u64>()
            + data.l0_baselines.iter().map(|t| t.file_size).sum::<u64>();
        let total_residuals: u64 = data.l1_residuals.iter().map(|t| t.file_size).sum::<u64>()
            + data.l0_residuals.iter().map(|t| t.file_size).sum::<u64>();
        let decode_res_badge = if data.decode_from_residuals_ok {
            r#"<span class="badge pass">PASS</span>"#
        } else {
            r#"<span class="badge fail">FAIL</span>"#
        };
        let decode_pack_badge = if data.decode_from_packs_ok {
            r#"<span class="badge pass">PASS</span>"#
        } else {
            r#"<span class="badge fail">FAIL</span>"#
        };
        let compression_ratio = if data.pack_size > 0 {
            l1_l0_originals as f64 / data.pack_size as f64
        } else {
            0.0
        };

        let mut html = String::with_capacity(16384);

        // Head
        html.push_str(&format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORIGAMI Pipeline Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background: #f5f5f5; color: #333; padding: 24px; line-height: 1.5; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; margin: 24px 0 12px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }}
  h3 {{ font-size: 1rem; margin: 16px 0 8px; }}
  .header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }}
  .timestamp {{ color: #888; font-size: 0.85rem; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold;
            font-size: 0.85rem; text-transform: uppercase; }}
  .badge.pass {{ background: #d4edda; color: #155724; }}
  .badge.fail {{ background: #f8d7da; color: #721c24; }}
  .card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
           padding: 16px; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid #eee; }}
  th {{ font-weight: 600; color: #555; }}
  .tile-grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .tile-item {{ text-align: center; }}
  .tile-item img {{ max-width: 256px; max-height: 256px; image-rendering: pixelated;
                    border: 1px solid #ddd; border-radius: 4px; display: block; }}
  .tile-item .label {{ font-size: 0.75rem; color: #666; margin-top: 4px; }}
  .check-ok {{ color: #155724; }}
  .check-fail {{ color: #721c24; }}
  .grid-4x4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
  .grid-2x2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }}
</style>
</head>
<body>
<div class="header">
  <h1>ORIGAMI Encode/Decode Pipeline Report</h1>
  {badge}
</div>
<div class="timestamp">Generated: {ts}</div>
"##, badge = badge, ts = data.timestamp));

        // Summary table
        html.push_str(&format!(r##"
<h2>Summary</h2>
<div class="card">
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Residual JPEG quality</td><td>90</td></tr>
  <tr><td>Decoded tile JPEG quality</td><td>90</td></tr>
  <tr><td>Encode time</td><td>{} ms</td></tr>
  <tr><td>Decode (residuals) time</td><td>{} ms</td></tr>
  <tr><td>Decode (packs) time</td><td>{} ms</td></tr>
  <tr><td>L1 tile count</td><td>{}</td></tr>
  <tr><td>L0 tile count</td><td>{}</td></tr>
  <tr><td>Total originals size</td><td>{}</td></tr>
  <tr><td>Total residuals size</td><td>{}</td></tr>
  <tr><td>Pack file size</td><td>{}</td></tr>
  <tr><td>Compression ratio (L1+L0 originals / pack)</td><td>{:.2}x ({} &rarr; {})</td></tr>
  <tr><td>Decoded from residuals</td><td>{}</td></tr>
  <tr><td>Decoded from packs</td><td>{}</td></tr>
</table>
</div>
"##,
            data.encode_ms,
            data.decode_residuals_ms,
            data.decode_packs_ms,
            data.l1_baselines.len(),
            data.l0_baselines.len(),
            fmt_size(total_originals),
            fmt_size(total_residuals),
            fmt_size(data.pack_size),
            compression_ratio,
            fmt_size(l1_l0_originals),
            fmt_size(data.pack_size),
            decode_res_badge,
            decode_pack_badge,
        ));

        // Helper closure for tile section
        fn render_tiles(html: &mut String, title: &str, tiles: &[TileInfo], grid_class: &str) {
            html.push_str(&format!("<h3>{}</h3>\n<div class=\"{}\">\n", title, grid_class));
            for t in tiles {
                let name = t.path.rsplit('/').next().unwrap_or(&t.path);
                html.push_str(&format!(
                    "<div class=\"tile-item\"><img src=\"{}\" alt=\"{}\"><div class=\"label\">{} ({})</div></div>\n",
                    t.path, name, name, fmt_size(t.file_size)
                ));
            }
            html.push_str("</div>\n");
        }

        // Originals section
        html.push_str("<h2>Originals (Baseline Pyramid)</h2>\n<div class=\"card\">\n");
        html.push_str(&format!(
            "<h3>Source Image (1024x1024)</h3>\n<div class=\"tile-grid\">\n<div class=\"tile-item\"><img src=\"{}\" alt=\"source\"><div class=\"label\">source.jpg ({})</div></div>\n</div>\n",
            data.source.path, fmt_size(data.source.file_size)
        ));
        html.push_str(&format!(
            "<h3>L2 Baseline (256x256)</h3>\n<div class=\"tile-grid\">\n<div class=\"tile-item\"><img src=\"{}\" alt=\"L2\"><div class=\"label\">0_0.jpg ({})</div></div>\n</div>\n",
            data.l2_baseline.path, fmt_size(data.l2_baseline.file_size)
        ));
        render_tiles(&mut html, "L1 Baseline Tiles", &data.l1_baselines, "tile-grid");
        render_tiles(&mut html, "L0 Baseline Tiles (4x4)", &data.l0_baselines, "grid-4x4");
        html.push_str("</div>\n");

        // Residuals section
        html.push_str("<h2>Residuals (Encoded)</h2>\n<div class=\"card\">\n");
        render_tiles(&mut html, "L1 Residuals", &data.l1_residuals, "tile-grid");
        render_tiles(&mut html, "L0 Residuals (4x4)", &data.l0_residuals, "grid-4x4");
        html.push_str(&format!("<p style=\"margin-top:8px;color:#666;\">Pack file: {}</p>\n", fmt_size(data.pack_size)));
        html.push_str("</div>\n");

        // Validation checklist
        html.push_str("<h2>Validation Checklist</h2>\n<div class=\"card\">\n<table>\n");
        html.push_str("<tr><th>Check</th><th>Result</th></tr>\n");
        for (label, ok) in &data.checks {
            let (cls, sym) = if *ok { ("check-ok", "PASS") } else { ("check-fail", "FAIL") };
            html.push_str(&format!(
                "<tr><td>{}</td><td class=\"{}\">{}</td></tr>\n",
                label, cls, sym
            ));
        }
        html.push_str("</table>\n</div>\n");

        html.push_str("</body>\n</html>\n");
        html
    }

    fn copy_file(src: &Path, dst: &Path) {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::copy(src, dst).unwrap();
    }

    #[test]
    fn test_generate_visual_report() {
        let src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join("L0-1024.jpg");
        if !src_path.exists() {
            eprintln!("Skipping: test-data/L0-1024.jpg not found");
            return;
        }

        let report_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join("report");
        let _ = fs::remove_dir_all(&report_dir);

        let tmp = std::env::temp_dir().join("origami_test_visual_report");
        let _ = fs::remove_dir_all(&tmp);

        let pyramid_dir = tmp.join("pyramid");
        let encode_out = tmp.join("encoded");
        let decode_res_out = report_dir.join("decoded_from_residuals");
        let decode_pack_out = report_dir.join("decoded_from_packs");

        // Step 1: Build synthetic pyramid
        build_synthetic_pyramid(&src_path, &pyramid_dir);

        // Step 2: Copy source and baseline tiles into report/originals/
        let originals = report_dir.join("originals");
        copy_file(&src_path, &originals.join("source.jpg"));
        copy_file(
            &pyramid_dir.join("baseline_pyramid_files/0/0_0.jpg"),
            &originals.join("L2/0_0.jpg"),
        );
        for ty in 0..2u32 {
            for tx in 0..2u32 {
                let name = format!("{}_{}.jpg", tx, ty);
                copy_file(
                    &pyramid_dir.join(format!("baseline_pyramid_files/1/{}", name)),
                    &originals.join(format!("L1/{}", name)),
                );
            }
        }
        for ty in 0..4u32 {
            for tx in 0..4u32 {
                let name = format!("{}_{}.jpg", tx, ty);
                copy_file(
                    &pyramid_dir.join(format!("baseline_pyramid_files/2/{}", name)),
                    &originals.join(format!("L0/{}", name)),
                );
            }
        }

        // Step 3: Encode (timed)
        let encode_start = Instant::now();
        let encode_args = crate::encode::EncodeArgs {
            pyramid: Some(pyramid_dir.clone()),
            image: None,
            out: encode_out.clone(),
            tile: 256,
            resq: 90,
            l1q: None,
            l0q: None,
            baseq: 95,
            subsamp: "444".to_string(),
            encoder: "turbojpeg".to_string(),
            max_parents: None,
            pack: true,
            manifest: false,
            optl2: false,
            debug_images: false,

            max_delta: 15,
            sharpen: None,
            save_sharpened: false,
        };
        crate::encode::run(encode_args).unwrap();
        let encode_ms = encode_start.elapsed().as_millis();

        // Step 4: Copy residuals and packs into report
        let residuals_out = report_dir.join("residuals");
        // L1 residuals
        for ty in 0..2u32 {
            for tx in 0..2u32 {
                let name = format!("{}_{}.jpg", tx, ty);
                copy_file(
                    &encode_out.join(format!("L1/0_0/{}", name)),
                    &residuals_out.join(format!("L1/0_0/{}", name)),
                );
            }
        }
        // L0 residuals
        for ty in 0..4u32 {
            for tx in 0..4u32 {
                let name = format!("{}_{}.jpg", tx, ty);
                copy_file(
                    &encode_out.join(format!("L0/0_0/{}", name)),
                    &residuals_out.join(format!("L0/0_0/{}", name)),
                );
            }
        }
        // Pack file
        let packs_out = report_dir.join("packs");
        fs::create_dir_all(&packs_out).unwrap();
        copy_file(
            &encode_out.join("packs/0_0.pack"),
            &packs_out.join("0_0.pack"),
        );

        // Step 5: Decode from residuals (timed)
        let decode_res_start = Instant::now();
        let decode_res_args = DecodeArgs {
            pyramid: pyramid_dir.clone(),
            out: decode_res_out.clone(),
            residuals: Some(encode_out.clone()),
            packs: None,
            pack_file: None,
            quality: 90,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        run(decode_res_args).unwrap();
        let decode_residuals_ms = decode_res_start.elapsed().as_millis();

        // Step 6: Decode from packs (timed)
        let decode_pack_start = Instant::now();
        let decode_pack_args = DecodeArgs {
            pyramid: pyramid_dir.clone(),
            out: decode_pack_out.clone(),
            residuals: None,
            packs: None,
            pack_file: Some(encode_out.join("packs/0_0.pack")),
            quality: 90,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
        };
        run(decode_pack_args).unwrap();
        let decode_packs_ms = decode_pack_start.elapsed().as_millis();

        // Step 7: Collect tile info
        let source = collect_tile(&report_dir, "originals/source.jpg");
        let l2_baseline = collect_tile(&report_dir, "originals/L2/0_0.jpg");

        let mut l1_baselines = Vec::new();
        let mut l0_baselines = Vec::new();
        let mut l1_residuals_info = Vec::new();
        let mut l0_residuals_info = Vec::new();

        for ty in 0..2u32 {
            for tx in 0..2u32 {
                let name = format!("{}_{}.jpg", tx, ty);
                l1_baselines.push(collect_tile(&report_dir, &format!("originals/L1/{}", name)));
                l1_residuals_info.push(collect_tile(&report_dir, &format!("residuals/L1/0_0/{}", name)));
            }
        }
        for ty in 0..4u32 {
            for tx in 0..4u32 {
                let name = format!("{}_{}.jpg", tx, ty);
                l0_baselines.push(collect_tile(&report_dir, &format!("originals/L0/{}", name)));
                l0_residuals_info.push(collect_tile(&report_dir, &format!("residuals/L0/0_0/{}", name)));
            }
        }

        let pack_size = fs::metadata(packs_out.join("0_0.pack"))
            .map(|m| m.len())
            .unwrap_or(0);

        // Validate decoded outputs (counts + valid JPEGs)
        let decode_from_residuals_ok = {
            let l1_ok = (0..2u32).all(|ty| (0..2u32).all(|tx| {
                let p = decode_res_out.join(format!("L1/{}_{}.jpg", tx, ty));
                fs::read(&p).map(|d| d.len() >= 2 && d[0] == 0xFF && d[1] == 0xD8).unwrap_or(false)
            }));
            let l0_ok = (0..4u32).all(|ty| (0..4u32).all(|tx| {
                let p = decode_res_out.join(format!("L0/{}_{}.jpg", tx, ty));
                fs::read(&p).map(|d| d.len() >= 2 && d[0] == 0xFF && d[1] == 0xD8).unwrap_or(false)
            }));
            l1_ok && l0_ok
        };
        let decode_from_packs_ok = {
            let l1_ok = (0..2u32).all(|ty| (0..2u32).all(|tx| {
                let p = decode_pack_out.join(format!("L1/{}_{}.jpg", tx, ty));
                fs::read(&p).map(|d| d.len() >= 2 && d[0] == 0xFF && d[1] == 0xD8).unwrap_or(false)
            }));
            let l0_ok = (0..4u32).all(|ty| (0..4u32).all(|tx| {
                let p = decode_pack_out.join(format!("L0/{}_{}.jpg", tx, ty));
                fs::read(&p).map(|d| d.len() >= 2 && d[0] == 0xFF && d[1] == 0xD8).unwrap_or(false)
            }));
            l1_ok && l0_ok
        };

        // Step 8: Validation checks
        let all_originals_exist = source.is_valid_jpeg
            && l2_baseline.is_valid_jpeg
            && l1_baselines.iter().all(|t| t.is_valid_jpeg)
            && l0_baselines.iter().all(|t| t.is_valid_jpeg);

        let all_residuals_valid = l1_residuals_info.iter().all(|t| t.is_valid_jpeg)
            && l0_residuals_info.iter().all(|t| t.is_valid_jpeg);

        let encode_counts_match = l1_residuals_info.len() == 4 && l0_residuals_info.len() == 16;

        let checks = vec![
            ("All original tiles exist and are valid JPEGs".to_string(), all_originals_exist),
            ("All residual tiles are valid JPEGs".to_string(), all_residuals_valid),
            ("Encode counts match (4 L1 + 16 L0)".to_string(), encode_counts_match),
            ("Decoded from residuals (4 L1 + 16 L0 valid JPEGs)".to_string(), decode_from_residuals_ok),
            ("Decoded from packs (4 L1 + 16 L0 valid JPEGs)".to_string(), decode_from_packs_ok),
        ];

        let all_pass = checks.iter().all(|(_, ok)| *ok);

        // Step 9: Build timestamp
        let timestamp = {
            let dur = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let secs = dur.as_secs();
            // Simple UTC timestamp without chrono dependency
            let days = secs / 86400;
            let time_secs = secs % 86400;
            let hours = time_secs / 3600;
            let minutes = (time_secs % 3600) / 60;
            let seconds = time_secs % 60;
            // Approximate date (good enough for a test report)
            // Days since 1970-01-01
            let mut y = 1970i64;
            let mut remaining = days as i64;
            loop {
                let days_in_year: i64 = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
                if remaining < days_in_year { break; }
                remaining -= days_in_year;
                y += 1;
            }
            let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
            let mdays = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut m = 0usize;
            while m < 12 && remaining >= mdays[m] {
                remaining -= mdays[m];
                m += 1;
            }
            format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m + 1, remaining + 1, hours, minutes, seconds)
        };

        let data = ReportData {
            timestamp,
            encode_ms,
            decode_residuals_ms,
            decode_packs_ms,
            source,
            l2_baseline,
            l1_baselines,
            l0_baselines,
            l1_residuals: l1_residuals_info,
            l0_residuals: l0_residuals_info,
            pack_size,
            decode_from_residuals_ok,
            decode_from_packs_ok,
            checks,
        };

        // Step 10: Render and write HTML
        let html = render_report_html(&data);
        fs::write(report_dir.join("index.html"), &html).unwrap();

        eprintln!("Visual report written to: {}", report_dir.display());
        eprintln!("  Encode: {}ms, Decode(residuals): {}ms, Decode(packs): {}ms",
            data.encode_ms, data.decode_residuals_ms, data.decode_packs_ms);

        // Assert all checks pass
        assert!(all_pass, "Some validation checks failed — see report");

        let _ = fs::remove_dir_all(&tmp);
    }
}
