//! GPU encode pipeline orchestrator.
//!
//! Coordinates the full WSI encoding pipeline:
//! 1. Parse DICOM, identify tile byte offsets
//! 2. Batch-decode JPEG tiles on GPU via nvJPEG
//! 3. Composite tiles into family canvases on GPU
//! 4. Lanczos downsample to L1/L2 on GPU
//! 5. OptL2 gradient descent on GPU (optional)
//! 6. Encode L2 baseline tiles via nvJPEG
//! 7. Compute and encode L1/L0 residuals
//! 8. Write bundle file on CPU

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::info;

use crate::dicom::DicomSlide;
use crate::kernels::GpuContext;
use crate::nvjpeg::NvJpegHandle;

/// Configuration for the GPU encode pipeline.
pub struct EncodeConfig {
    pub tile_size: u32,
    pub baseq: u8,
    pub l1q: u8,
    pub l0q: u8,
    pub optl2: bool,
    pub max_delta: u8,
    pub subsamp: String,
    pub batch_size: usize,
    pub device: usize,
}

/// Summary of an encode run.
pub struct EncodeSummary {
    pub families_encoded: usize,
    pub l2_bytes: usize,
    pub residual_bytes: usize,
    pub elapsed_secs: f64,
}

/// Encode a full WSI from DICOM to bundle format.
pub fn encode_wsi(
    dicom_path: &Path,
    output_dir: &Path,
    config: EncodeConfig,
) -> Result<EncodeSummary> {
    let start = Instant::now();

    // 1. Parse DICOM and extract tile layout
    info!("Opening DICOM: {}", dicom_path.display());
    let slide = DicomSlide::open(dicom_path)
        .with_context(|| format!("Failed to open DICOM {}", dicom_path.display()))?;

    info!(
        "DICOM: {}x{} pixels, {} tiles ({}x{} grid), tile_size={}x{}",
        slide.width(),
        slide.height(),
        slide.tile_count(),
        slide.tiles_x(),
        slide.tiles_y(),
        slide.tile_w(),
        slide.tile_h(),
    );

    // Compute family grid (4x4 tiles per family)
    let region_size = config.tile_size * 4;
    let grid_cols = (slide.tiles_x() + 3) / 4;
    let grid_rows = (slide.tiles_y() + 3) / 4;
    let total_families = (grid_cols * grid_rows) as usize;
    info!(
        "Family grid: {}x{} = {} families",
        grid_cols, grid_rows, total_families
    );

    // 2. Initialize CUDA context and nvJPEG
    info!("Initializing CUDA device {}", config.device);
    let gpu = GpuContext::new(config.device)?;
    let nvjpeg = NvJpegHandle::new(&gpu)?;

    // Create output directories
    fs::create_dir_all(output_dir)?;
    let files_dir = output_dir.join("baseline_pyramid_files");
    fs::create_dir_all(&files_dir)?;

    let mut total_l2_bytes = 0usize;
    let mut total_residual_bytes = 0usize;
    let mut families_encoded = 0usize;

    // Accumulate bundle entries
    let mut bundle_entries: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(total_families);

    // 3. Process families in batches
    let families: Vec<(u32, u32)> = (0..grid_rows)
        .flat_map(|row| (0..grid_cols).map(move |col| (col, row)))
        .collect();

    for batch_start in (0..families.len()).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(families.len());
        let batch = &families[batch_start..batch_end];
        let batch_size = batch.len();

        info!(
            "Processing batch {}-{} of {} ({} families)",
            batch_start,
            batch_end,
            families.len(),
            batch_size,
        );

        // Per-batch GPU pipeline:
        //
        // a) Gather JPEG tile bytes for all families in batch
        //    Each family needs 16 tiles (4x4 grid)
        let mut tile_jpeg_bytes: Vec<Vec<u8>> = Vec::with_capacity(batch_size * 16);
        for &(col, row) in batch {
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let tx = col * 4 + dx;
                    let ty = row * 4 + dy;
                    let jpeg_data = slide.get_tile_bytes(tx, ty)
                        .unwrap_or_default();
                    tile_jpeg_bytes.push(jpeg_data);
                }
            }
        }

        // b) Batch decode on GPU
        let _decoded = nvjpeg.batch_decode(&tile_jpeg_bytes)?;

        // c) Composite, downsample, optl2, residual computation
        //    (GPU kernels — currently stubs, will be filled in with CUDA implementation)

        // d) Encode results
        //    For now, track progress
        for &(col, row) in batch {
            // TODO: Replace with actual GPU pipeline output
            // Placeholder: create empty bundle entry
            let lz4_data = Vec::new();
            if !lz4_data.is_empty() {
                bundle_entries.push((col, row, lz4_data));
            }
            families_encoded += 1;
        }
    }

    // 4. Write bundle
    if !bundle_entries.is_empty() {
        let bundle_dir = output_dir.join("residual_packs");
        fs::create_dir_all(&bundle_dir)?;
        let bundle_path = bundle_dir.join("residuals.bundle");
        origami::core::pack::write_bundle(
            &bundle_path,
            grid_cols as u16,
            grid_rows as u16,
            &bundle_entries,
        )?;
        info!("Wrote bundle: {}", bundle_path.display());
    }

    let elapsed = start.elapsed().as_secs_f64();

    // 5. Write summary
    let summary_json = serde_json::json!({
        "mode": "gpu-encode",
        "dicom": dicom_path.to_string_lossy(),
        "tile_size": config.tile_size,
        "baseq": config.baseq,
        "l1q": config.l1q,
        "l0q": config.l0q,
        "optl2": config.optl2,
        "families": families_encoded,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "l2_bytes": total_l2_bytes,
        "residual_bytes": total_residual_bytes,
        "elapsed_secs": elapsed,
    });
    fs::write(
        output_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary_json)?,
    )?;

    Ok(EncodeSummary {
        families_encoded,
        l2_bytes: total_l2_bytes,
        residual_bytes: total_residual_bytes,
        elapsed_secs: elapsed,
    })
}
