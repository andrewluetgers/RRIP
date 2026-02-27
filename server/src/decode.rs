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
use crate::core::ResampleFilter;

#[derive(Args, Debug)]
pub struct DecodeArgs {
    /// Path to DZI pyramid directory (containing baseline_pyramid.dzi)
    #[arg(long)]
    pyramid: PathBuf,

    /// Output directory for reconstructed tiles
    #[arg(long)]
    out: PathBuf,

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

    /// Upsample filter for predictions: bilinear, bicubic, lanczos3 (default: lanczos3)
    #[arg(long, default_value = "lanczos3")]
    upsample_filter: String,
}

pub fn run(args: DecodeArgs) -> Result<()> {
    let start = Instant::now();

    let output_format = match args.output_format.as_str() {
        "jpeg" | "jpg" => OutputFormat::Jpeg,
        "webp" => OutputFormat::Webp,
        other => return Err(anyhow!("unknown output format: '{}'. Available: jpeg, webp", other)),
    };

    // Validate: exactly one residual source must be provided
    let source_count = args.packs.is_some() as u8
        + args.pack_file.is_some() as u8
        + args.bundle.is_some() as u8;
    if source_count != 1 {
        return Err(anyhow!(
            "Exactly one of --packs, --pack-file, or --bundle must be provided"
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
    let (parents, pack_dir) = if let Some(ref bundle) = bundle_file {
        // Bundle file: enumerate all non-empty families from index that also have baseline tiles
        let mut parents = Vec::new();
        let pyramid_files_dir = args.pyramid.join("baseline_pyramid_files").join(format!("{}", pyramid.l2));
        for row in 0..bundle.grid_rows() {
            for col in 0..bundle.grid_cols() {
                if bundle.get_pack(col as u32, row as u32).is_ok() {
                    let baseline_path = pyramid_files_dir.join(format!("{}_{}.jpg", col, row));
                    if baseline_path.exists() {
                        parents.push((col as u32, row as u32));
                    }
                }
            }
        }
        parents.sort();
        (parents, None)
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

        let pack_parent = pack_file
            .parent()
            .ok_or_else(|| anyhow!("pack-file has no parent directory"))?;

        (vec![coords], Some(pack_parent.to_path_buf()))
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
        (parents, Some(packs_dir.clone()))
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
            pack_dir: pack_dir.as_deref(),
            bundle: bundle_file.as_ref(),
            tile_size: pyramid.tile_size,
            l0: pyramid.l0,
            l1: pyramid.l1,
            l2: pyramid.l2,
        };
        let upsample_filter: ResampleFilter = args.upsample_filter.parse()
            .map_err(|e: String| anyhow!(e))?;
        let opts = ReconstructOpts {
            quality: args.quality,
            timing: args.timing,
            grayscale_only: args.grayscale,
            output_format,
            sharpen: None,
            upsample_filter,
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
                "family_breakdown x2={} y2={} l2_decode={}ms upsample={}ms l0_res={}ms l0_enc={}ms l1_ds={}ms l1_enc={}ms total={}ms l0_par={}",
                x2, y2,
                stats.l2_decode_ms, stats.upsample_ms,
                stats.l0_residual_ms, stats.l0_encode_ms,
                stats.l1_downsample_ms, stats.l1_encode_ms,
                stats.total_ms, stats.l0_parallel_max
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
        "pipeline_version": 2,
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
    use std::path::PathBuf;

    fn test_data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data").join("demo_out")
    }

    #[allow(dead_code)]
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
            packs: None,
            pack_file: None,
            bundle: None,
            quality: 90,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
            upsample_filter: "lanczos3".to_string(),
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
            packs: Some(PathBuf::from("/tmp/p")),
            pack_file: Some(PathBuf::from("/tmp/f.pack")),
            bundle: None,
            quality: 90,
            tile: 256,
            max_parents: None,
            grayscale: false,
            timing: false,
            output_format: "jpeg".to_string(),
            upsample_filter: "lanczos3".to_string(),
        };
        let result = run(args);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("Exactly one"),
            "Should reject multiple sources"
        );
        let _ = fs::remove_dir_all(&out_dir);
    }
}
