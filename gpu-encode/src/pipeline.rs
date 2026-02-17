//! GPU encode pipeline orchestrator.
//!
//! Two modes:
//! - `encode_single_image()` — single image encode (for evals), mirrors CPU `run_single_image()`
//! - `encode_wsi()` — DICOM WSI batch encode with GPU kernels
//!
//! Data stays on GPU as much as possible. Only host transfers for:
//! - Loading source images from disk
//! - Writing JPEG/WebP output files to disk
//! - nvJPEG encode output (compressed bytes returned on host)

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use cudarc::driver::CudaSlice;
use tracing::info;

use crate::kernels::GpuContext;
use crate::nvjpeg::NvJpegHandle;
use image::{RgbImage, imageops, ImageBuffer};

/// Configuration for the GPU encode pipeline.
pub struct EncodeConfig {
    pub tile_size: u32,
    pub baseq: u8,
    pub l1q: u8,
    pub l0q: u8,
    pub optl2: bool,
    pub max_delta: u8,
    pub subsamp: String,
    pub l2resq: u8,
    pub pack: bool,
    pub manifest: bool,
    pub debug_images: bool,
    pub max_parents: Option<usize>,
    pub batch_size: usize,
    pub device: usize,
    pub profile: bool,
    pub sharpen: Option<f32>,
    pub save_sharpened: bool,
    pub generate_pyramid: bool,
    pub pyramid_mode: String,  // "gpu" or "cpu"
    pub pyramid_sharpen: f32,  // Unsharp mask strength for pyramid levels
}

/// Summary of a single-image encode run.
pub struct SingleImageSummary {
    pub l1_tiles: usize,
    pub l0_tiles: usize,
    pub l2_bytes: usize,
    pub total_bytes: usize,
    pub elapsed_secs: f64,
}

/// Summary of a WSI encode run.
pub struct EncodeSummary {
    pub families_encoded: usize,
    pub l2_bytes: usize,
    pub residual_bytes: usize,
    pub elapsed_secs: f64,
}

/// Per-stage timing accumulators for profiling the WSI pipeline.
#[derive(Default)]
struct PipelineTimers {
    dicom_open: std::time::Duration,
    // Per-family accumulators (summed across all families):
    gather_tiles: std::time::Duration,
    nvjpeg_decode: std::time::Duration,
    composite: std::time::Duration,
    downsample: std::time::Duration,
    optl2: std::time::Duration,
    l2_encode: std::time::Duration,
    l2_decode: std::time::Duration,
    l1_predict: std::time::Duration,
    l1_residuals: std::time::Duration,
    l1_recon_rgb: std::time::Duration,
    l0_predict: std::time::Duration,
    l0_residuals: std::time::Duration,
    pack: std::time::Duration,
    bundle_write: std::time::Duration,
    families_total: usize,
    families_empty: usize,
    l1_tiles_processed: usize,
    l0_tiles_processed: usize,
    l0_tiles_empty: usize,
}

impl PipelineTimers {
    fn family_loop_total(&self) -> std::time::Duration {
        self.gather_tiles
            + self.nvjpeg_decode
            + self.composite
            + self.downsample
            + self.optl2
            + self.l2_encode
            + self.l2_decode
            + self.l1_predict
            + self.l1_residuals
            + self.l1_recon_rgb
            + self.l0_predict
            + self.l0_residuals
            + self.pack
    }

    fn print_report(&self) {
        let loop_total = self.family_loop_total();
        let loop_s = loop_total.as_secs_f64();
        let pct = |d: std::time::Duration| -> f64 {
            if loop_s > 0.0 { d.as_secs_f64() / loop_s * 100.0 } else { 0.0 }
        };
        let families_with_data = self.families_total - self.families_empty;

        info!("=== Pipeline Timing Report ===");
        info!("DICOM open:            {:.3}s", self.dicom_open.as_secs_f64());
        info!(
            "Families total:        {} ({} with data, {} empty/skipped)",
            self.families_total, families_with_data, self.families_empty,
        );
        info!("Time in family loop:   {:.3}s", loop_s);
        info!("  gather_tiles:        {:.3}s  ({:.1}%)", self.gather_tiles.as_secs_f64(), pct(self.gather_tiles));
        info!("  nvjpeg_decode:       {:.3}s  ({:.1}%)", self.nvjpeg_decode.as_secs_f64(), pct(self.nvjpeg_decode));
        info!("  composite:           {:.3}s  ({:.1}%)", self.composite.as_secs_f64(), pct(self.composite));
        info!("  downsample:          {:.3}s  ({:.1}%)", self.downsample.as_secs_f64(), pct(self.downsample));
        info!("  optl2:               {:.3}s  ({:.1}%)", self.optl2.as_secs_f64(), pct(self.optl2));
        info!("  l2_encode:           {:.3}s  ({:.1}%)", self.l2_encode.as_secs_f64(), pct(self.l2_encode));
        info!("  l2_decode:           {:.3}s  ({:.1}%)", self.l2_decode.as_secs_f64(), pct(self.l2_decode));
        info!("  l1_predict:          {:.3}s  ({:.1}%)", self.l1_predict.as_secs_f64(), pct(self.l1_predict));
        info!("  l1_residuals:        {:.3}s  ({:.1}%)", self.l1_residuals.as_secs_f64(), pct(self.l1_residuals));
        info!("  l1_recon_rgb:        {:.3}s  ({:.1}%)", self.l1_recon_rgb.as_secs_f64(), pct(self.l1_recon_rgb));
        info!("  l0_predict:          {:.3}s  ({:.1}%)", self.l0_predict.as_secs_f64(), pct(self.l0_predict));
        info!("  l0_residuals:        {:.3}s  ({:.1}%)", self.l0_residuals.as_secs_f64(), pct(self.l0_residuals));
        info!("  pack:                {:.3}s  ({:.1}%)", self.pack.as_secs_f64(), pct(self.pack));
        info!("Bundle write:          {:.3}s", self.bundle_write.as_secs_f64());
        info!("L1 tiles processed:    {}", self.l1_tiles_processed);
        info!("L0 tiles processed:    {} ({} empty/skipped)", self.l0_tiles_processed, self.l0_tiles_empty);
        if families_with_data > 0 {
            let loop_per_family = loop_s / families_with_data as f64;
            info!(
                "Throughput:            {:.1} families/s ({:.2} ms/family, with-data only)",
                families_with_data as f64 / loop_s,
                loop_per_family * 1000.0,
            );
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let loop_total = self.family_loop_total();
        let families_with_data = self.families_total - self.families_empty;
        serde_json::json!({
            "dicom_open_s": self.dicom_open.as_secs_f64(),
            "families_total": self.families_total,
            "families_with_data": families_with_data,
            "families_empty": self.families_empty,
            "family_loop_total_s": loop_total.as_secs_f64(),
            "stages": {
                "gather_tiles_s": self.gather_tiles.as_secs_f64(),
                "nvjpeg_decode_s": self.nvjpeg_decode.as_secs_f64(),
                "composite_s": self.composite.as_secs_f64(),
                "downsample_s": self.downsample.as_secs_f64(),
                "optl2_s": self.optl2.as_secs_f64(),
                "l2_encode_s": self.l2_encode.as_secs_f64(),
                "l2_decode_s": self.l2_decode.as_secs_f64(),
                "l1_predict_s": self.l1_predict.as_secs_f64(),
                "l1_residuals_s": self.l1_residuals.as_secs_f64(),
                "l1_recon_rgb_s": self.l1_recon_rgb.as_secs_f64(),
                "l0_predict_s": self.l0_predict.as_secs_f64(),
                "l0_residuals_s": self.l0_residuals.as_secs_f64(),
                "pack_s": self.pack.as_secs_f64(),
            },
            "bundle_write_s": self.bundle_write.as_secs_f64(),
            "l1_tiles_processed": self.l1_tiles_processed,
            "l0_tiles_processed": self.l0_tiles_processed,
            "l0_tiles_empty": self.l0_tiles_empty,
            "throughput_families_per_s": if loop_total.as_secs_f64() > 0.0 {
                families_with_data as f64 / loop_total.as_secs_f64()
            } else { 0.0 },
        })
    }
}

// ---------------------------------------------------------------------------
// GPU resource monitor (via NVML, for --profile mode)
// ---------------------------------------------------------------------------

/// A single GPU resource snapshot.
#[derive(Clone, Debug)]
struct GpuSnapshot {
    gpu_util_pct: u32,
    mem_util_pct: u32,
    vram_used_mb: f64,
    vram_total_mb: f64,
    power_w: f64,
    temp_c: u32,
}

/// Tracks GPU resource utilization per-stage via NVML.
struct GpuMonitor {
    device: nvml_wrapper::Device<'static>,
    // We need to keep Nvml alive for the device reference
    _nvml: &'static nvml_wrapper::Nvml,
    /// Per-stage snapshots: stage_name -> Vec<snapshot>
    stage_snapshots: Vec<(String, GpuSnapshot)>,
    peak_vram_mb: f64,
    peak_gpu_util: u32,
    peak_power_w: f64,
}

// We leak the Nvml handle to get a 'static lifetime — it's a singleton anyway
static NVML_INIT: std::sync::Once = std::sync::Once::new();
static mut NVML_INSTANCE: Option<nvml_wrapper::Nvml> = None;

fn get_nvml() -> Option<&'static nvml_wrapper::Nvml> {
    NVML_INIT.call_once(|| {
        match nvml_wrapper::Nvml::init() {
            Ok(nvml) => unsafe { NVML_INSTANCE = Some(nvml); },
            Err(e) => {
                tracing::warn!("Failed to initialize NVML: {}. GPU monitoring disabled.", e);
            }
        }
    });
    unsafe { NVML_INSTANCE.as_ref() }
}

impl GpuMonitor {
    fn new(device_index: u32) -> Option<Self> {
        let nvml = get_nvml()?;
        let device = nvml.device_by_index(device_index).ok()?;
        Some(GpuMonitor {
            device,
            _nvml: nvml,
            stage_snapshots: Vec::new(),
            peak_vram_mb: 0.0,
            peak_gpu_util: 0,
            peak_power_w: 0.0,
        })
    }

    fn sample(&mut self, stage: &str) {
        let snap = self.take_snapshot();
        if let Some(s) = &snap {
            if s.vram_used_mb > self.peak_vram_mb { self.peak_vram_mb = s.vram_used_mb; }
            if s.gpu_util_pct > self.peak_gpu_util { self.peak_gpu_util = s.gpu_util_pct; }
            if s.power_w > self.peak_power_w { self.peak_power_w = s.power_w; }
            self.stage_snapshots.push((stage.to_string(), s.clone()));
        }
    }

    fn take_snapshot(&self) -> Option<GpuSnapshot> {
        let util = self.device.utilization_rates().ok()?;
        let mem = self.device.memory_info().ok()?;
        let power = self.device.power_usage().unwrap_or(0);
        let temp = self.device.temperature(
            nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu
        ).unwrap_or(0);

        Some(GpuSnapshot {
            gpu_util_pct: util.gpu,
            mem_util_pct: util.memory,
            vram_used_mb: mem.used as f64 / 1_048_576.0,
            vram_total_mb: mem.total as f64 / 1_048_576.0,
            power_w: power as f64 / 1000.0,
            temp_c: temp,
        })
    }

    fn print_summary(&self) {
        if self.stage_snapshots.is_empty() { return; }
        let last = &self.stage_snapshots.last().unwrap().1;
        info!("=== GPU Resource Summary ===");
        info!("VRAM total:            {:.0} MB", last.vram_total_mb);
        info!("VRAM peak:             {:.0} MB ({:.1}%)", self.peak_vram_mb, self.peak_vram_mb / last.vram_total_mb * 100.0);
        info!("GPU util peak:         {}%", self.peak_gpu_util);
        info!("Power peak:            {:.1} W", self.peak_power_w);
        info!("Temperature (final):   {} C", last.temp_c);

        // Per-stage averages
        let stages: Vec<&str> = [
            "gather_tiles", "nvjpeg_decode", "composite", "downsample", "optl2",
            "l2_encode", "l2_decode", "l1_predict", "l1_residuals", "l1_recon_rgb",
            "l0_predict", "l0_residuals", "pack",
        ].into();
        info!("Per-stage GPU util (avg):");
        for stage in &stages {
            let samples: Vec<&GpuSnapshot> = self.stage_snapshots.iter()
                .filter(|(s, _)| s == *stage)
                .map(|(_, snap)| snap)
                .collect();
            if samples.is_empty() { continue; }
            let avg_gpu = samples.iter().map(|s| s.gpu_util_pct as f64).sum::<f64>() / samples.len() as f64;
            let avg_vram = samples.iter().map(|s| s.vram_used_mb).sum::<f64>() / samples.len() as f64;
            let avg_power = samples.iter().map(|s| s.power_w).sum::<f64>() / samples.len() as f64;
            info!(
                "  {:20} GPU {:5.1}%  VRAM {:6.0} MB  Power {:5.1} W  (n={})",
                stage, avg_gpu, avg_vram, avg_power, samples.len(),
            );
        }
    }

    fn to_json(&self) -> serde_json::Value {
        if self.stage_snapshots.is_empty() {
            return serde_json::json!(null);
        }
        let last = &self.stage_snapshots.last().unwrap().1;

        let stages: Vec<&str> = [
            "gather_tiles", "nvjpeg_decode", "composite", "downsample", "optl2",
            "l2_encode", "l2_decode", "l1_predict", "l1_residuals", "l1_recon_rgb",
            "l0_predict", "l0_residuals", "pack",
        ].into();

        let mut stage_stats = serde_json::Map::new();
        for stage in &stages {
            let samples: Vec<&GpuSnapshot> = self.stage_snapshots.iter()
                .filter(|(s, _)| s == *stage)
                .map(|(_, snap)| snap)
                .collect();
            if samples.is_empty() { continue; }
            let n = samples.len() as f64;
            stage_stats.insert(stage.to_string(), serde_json::json!({
                "samples": samples.len(),
                "avg_gpu_util_pct": (samples.iter().map(|s| s.gpu_util_pct as f64).sum::<f64>() / n * 10.0).round() / 10.0,
                "avg_mem_util_pct": (samples.iter().map(|s| s.mem_util_pct as f64).sum::<f64>() / n * 10.0).round() / 10.0,
                "avg_vram_used_mb": (samples.iter().map(|s| s.vram_used_mb).sum::<f64>() / n).round(),
                "avg_power_w": (samples.iter().map(|s| s.power_w).sum::<f64>() / n * 10.0).round() / 10.0,
            }));
        }

        serde_json::json!({
            "vram_total_mb": last.vram_total_mb.round(),
            "peak_vram_used_mb": self.peak_vram_mb.round(),
            "peak_gpu_util_pct": self.peak_gpu_util,
            "peak_power_w": (self.peak_power_w * 10.0).round() / 10.0,
            "final_temp_c": last.temp_c,
            "per_stage": stage_stats,
        })
    }
}

// ---------------------------------------------------------------------------
// Debug image helpers (only used when --debug-images is set)
// ---------------------------------------------------------------------------

fn save_rgb_png(path: &Path, data: &[u8], w: u32, h: u32) -> Result<()> {
    let img = image::RgbImage::from_raw(w, h, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("failed to create RgbImage"))?;
    img.save(path).with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

fn save_gray_png(path: &Path, data: &[u8], w: u32, h: u32) -> Result<()> {
    let img = image::GrayImage::from_raw(w, h, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("failed to create GrayImage"))?;
    img.save(path).with_context(|| format!("failed to save {}", path.display()))?;
    Ok(())
}

/// Download a GPU u8 buffer to host. Only for debug/manifest paths.
fn download_u8(gpu: &GpuContext, dev: &CudaSlice<u8>) -> Result<Vec<u8>> {
    gpu.stream.clone_dtoh(dev).map_err(|e| anyhow::anyhow!("dtoh u8 failed: {}", e))
}

/// Download a GPU f32 buffer to host. Only for debug/manifest paths.
fn download_f32(gpu: &GpuContext, dev: &CudaSlice<f32>) -> Result<Vec<f32>> {
    gpu.stream.clone_dtoh(dev).map_err(|e| anyhow::anyhow!("dtoh f32 failed: {}", e))
}

/// Upload host u8 data to GPU.
fn upload_u8(gpu: &GpuContext, data: &[u8]) -> Result<CudaSlice<u8>> {
    gpu.stream.clone_htod(data).map_err(|e| anyhow::anyhow!("htod u8 failed: {}", e))
}

/// Upload host f32 data to GPU.
fn upload_f32(gpu: &GpuContext, data: &[f32]) -> Result<CudaSlice<f32>> {
    gpu.stream.clone_htod(data).map_err(|e| anyhow::anyhow!("htod f32 failed: {}", e))
}

// ---------------------------------------------------------------------------
// Single-image encode pipeline (GPU-native)
// ---------------------------------------------------------------------------

/// Encode a single image using GPU kernels and nvJPEG.
pub fn encode_single_image(
    image_path: &Path,
    output_dir: &Path,
    config: EncodeConfig,
) -> Result<SingleImageSummary> {
    let start = Instant::now();

    // Initialize GPU + nvJPEG
    info!("Initializing CUDA device {}", config.device);
    let gpu = GpuContext::new(config.device)?;
    let jpeg = NvJpegHandle::new(&gpu)?;

    let tile_size = config.tile_size;
    let baseq = config.baseq;
    let l1q = config.l1q;
    let l0q = config.l0q;

    // 1. Load source image to host, then upload to GPU as u8
    let src_img = image::open(image_path)
        .with_context(|| format!("failed to open {}", image_path.display()))?
        .to_rgb8();
    let (src_w, src_h) = (src_img.width(), src_img.height());
    let src_rgb_host = src_img.into_raw();
    info!("Source image: {}x{}", src_w, src_h);

    let tiles_x = (src_w + tile_size - 1) / tile_size;
    let tiles_y = (src_h + tile_size - 1) / tile_size;
    info!("L0 grid: {}x{} tiles ({}x{})", tiles_x, tiles_y, tile_size, tile_size);

    // Upload source RGB to GPU as u8, then convert to f32 on GPU
    let src_u8_dev = upload_u8(&gpu, &src_rgb_host)?;
    let src_f32_dev = gpu.u8_to_f32(&src_u8_dev, (src_w * src_h * 3) as i32)?;

    // 2. GPU: Lanczos3 downsample to L1 and L2
    let l1_w = (src_w + 1) / 2;
    let l1_h = (src_h + 1) / 2;
    let l2_w = (l1_w + 1) / 2;
    let l2_h = (l1_h + 1) / 2;

    let l1_f32_dev = gpu.downsample_lanczos3(
        &src_f32_dev, 1, src_h as i32, src_w as i32, l1_h as i32, l1_w as i32, 3,
    )?;
    let l2_f32_dev = gpu.downsample_lanczos3(
        &src_f32_dev, 1, src_h as i32, src_w as i32, l2_h as i32, l2_w as i32, 3,
    )?;

    // Convert L1 and L2 to u8 on GPU (ground truth)
    let l1_u8_dev = gpu.f32_to_u8(&l1_f32_dev, (l1_w * l1_h * 3) as i32)?;
    let mut l2_u8_dev = gpu.f32_to_u8(&l2_f32_dev, (l2_w * l2_h * 3) as i32)?;

    // 3. OptL2 on GPU (optional) — operates on f32
    if config.optl2 {
        info!("Running OptL2 gradient descent on GPU...");
        // Need L2 and L1 as f32 for optimization
        let l2_opt_f32 = gpu.u8_to_f32(&l2_u8_dev, (l2_w * l2_h * 3) as i32)?;
        let l1_target_f32 = gpu.u8_to_f32(&l1_u8_dev, (l1_w * l1_h * 3) as i32)?;

        let mut l2_current = gpu.u8_to_f32(&l2_u8_dev, (l2_w * l2_h * 3) as i32)?;
        for _ in 0..100 {
            gpu.optl2_step(
                &mut l2_current, &l2_opt_f32, &l1_target_f32,
                1, l2_h as i32, l2_w as i32,
                0.3, config.max_delta as f32,
            )?;
        }
        // Convert optimized L2 back to u8
        l2_u8_dev = gpu.f32_to_u8(&l2_current, (l2_w * l2_h * 3) as i32)?;
        info!("OptL2 complete");
    }

    // Sharpen L2 before saving if --save-sharpened
    if config.save_sharpened {
        if let Some(strength) = config.sharpen {
            l2_u8_dev = gpu.sharpen_l2(&l2_u8_dev, l2_w, l2_h, strength)?;
        }
    }

    // Create output directory
    fs::create_dir_all(output_dir)?;

    // 4. Encode L2 baseline JPEG via nvJPEG (GPU encode → host bytes)
    let l2_jpeg = jpeg.encode_rgb(&gpu, &l2_u8_dev, l2_w, l2_h, baseq, &config.subsamp)?;
    let l2_bytes = l2_jpeg.len();
    fs::write(output_dir.join("L2_0_0.jpg"), &l2_jpeg)?;
    info!("L2 baseline: {}x{} -> {} bytes (Q{} {})", l2_w, l2_h, l2_bytes, baseq, config.subsamp);

    // 5. Decode L2 back on GPU via nvJPEG (host bytes → GPU u8)
    let (l2_decoded_dev, _, _) = jpeg.decode_to_device(&gpu, &l2_jpeg)?;

    // 6. L2 RGB residual (WebP lossless — only CPU operation, rare)
    let l2_res_bytes;
    let l2_for_pred_dev = if config.l2resq > 0 {
        // Download both L2 original and decoded to CPU for WebP encode
        gpu.sync()?;
        let l2_orig_host = download_u8(&gpu, &l2_u8_dev)?;
        let l2_decoded_host = download_u8(&gpu, &l2_decoded_dev)?;

        let n = (l2_w * l2_h * 3) as usize;
        let mut l2_residual = vec![0u8; n];
        for i in 0..n {
            l2_residual[i] = (l2_orig_host[i] as i16 - l2_decoded_host[i] as i16 + 128).clamp(0, 255) as u8;
        }

        let webp_encoder = webp::Encoder::from_rgb(&l2_residual, l2_w, l2_h);
        let webp_mem = webp_encoder.encode_lossless();
        fs::write(output_dir.join("L2_0_0_residual.webp"), &*webp_mem)?;
        l2_res_bytes = webp_mem.len();
        info!("L2 residual (WebP lossless): {} bytes", l2_res_bytes);

        // With lossless residual, prediction uses original L2
        l2_u8_dev.clone()
    } else {
        l2_res_bytes = 0;
        l2_decoded_dev.clone()
    };

    // Debug: L2 images
    if config.debug_images {
        let compress_dir = output_dir.join("compress");
        let decompress_dir = output_dir.join("decompress");
        fs::create_dir_all(&compress_dir)?;
        fs::create_dir_all(&decompress_dir)?;
        gpu.sync()?;
        let l2_orig_host = download_u8(&gpu, &l2_u8_dev)?;
        let l2_decoded_host = download_u8(&gpu, &l2_decoded_dev)?;
        save_rgb_png(&compress_dir.join("001_L2_original.png"), &l2_orig_host, l2_w, l2_h)?;
        save_rgb_png(&decompress_dir.join("050_L2_decode.png"), &l2_decoded_host, l2_w, l2_h)?;
        // L2 reconstructed (with residual applied if l2resq > 0, otherwise same as decode)
        let l2_for_pred_host = download_u8(&gpu, &l2_for_pred_dev)?;
        save_rgb_png(&decompress_dir.join("050_L2_reconstructed.png"), &l2_for_pred_host, l2_w, l2_h)?;
    }

    // Sharpen decoded L2 for prediction (simulates decode-time: decode → sharpen → upsample)
    // Always sharpen the JPEG-decoded L2, not the l2resq-reconstructed original
    let l2_for_pred_dev = if !config.save_sharpened {
        if let Some(strength) = config.sharpen {
            gpu.sharpen_l2(&l2_decoded_dev, l2_w, l2_h, strength)?
        } else {
            l2_for_pred_dev
        }
    } else {
        l2_for_pred_dev
    };

    // 7. GPU: upsample L2 → L1 prediction (bilinear 2x, stays on GPU as f32)
    let l2_pred_f32 = gpu.u8_to_f32(&l2_for_pred_dev, (l2_w * l2_h * 3) as i32)?;
    let l1_pred_f32 = gpu.upsample_bilinear_2x(&l2_pred_f32, 1, l2_h as i32, l2_w as i32, 3)?;
    let l1_pred_w = l2_w * 2;
    let l1_pred_h = l2_h * 2;

    // Debug: L1 mosaic prediction
    if config.debug_images {
        let decompress_dir = output_dir.join("decompress");
        gpu.sync()?;
        let l1_pred_host = gpu.stream.clone_dtoh(
            &gpu.f32_to_u8(&l1_pred_f32, (l1_pred_w * l1_pred_h * 3) as i32)?
        ).map_err(|e| anyhow::anyhow!("dtoh: {}", e))?;
        save_rgb_png(&decompress_dir.join("051_L1_mosaic_prediction.png"), &l1_pred_host, l1_pred_w, l1_pred_h)?;
    }

    // 8. GPU: rgb_to_ycbcr_f32 on L1 prediction (stays on GPU)
    let l1_pred_u8 = gpu.f32_to_u8(&l1_pred_f32, (l1_pred_w * l1_pred_h * 3) as i32)?;
    let l1_pred_pixels = (l1_pred_w * l1_pred_h) as i32;
    let (l1_pred_y_dev, l1_pred_cb_dev, l1_pred_cr_dev) =
        gpu.rgb_to_ycbcr_f32(&l1_pred_u8, l1_pred_pixels)?;

    // For tile-level processing, download prediction planes to host
    // (tiles are small enough that extraction on CPU is practical)
    gpu.sync()?;
    let l1_pred_y_host = download_f32(&gpu, &l1_pred_y_dev)?;
    let l1_pred_cb_host = download_f32(&gpu, &l1_pred_cb_dev)?;
    let l1_pred_cr_host = download_f32(&gpu, &l1_pred_cr_dev)?;
    let l1_rgb_host = download_u8(&gpu, &l1_u8_dev)?;

    // L0 ground truth Y tiles also needed on host for extraction
    // (source image already on host from initial load)

    // Process L1 residuals
    let l1_out = output_dir.join("L1").join("0_0");
    fs::create_dir_all(&l1_out)?;
    let l1_tiles_x = (l1_w + tile_size - 1) / tile_size;
    let l1_tiles_y = (l1_h + tile_size - 1) / tile_size;

    let mut l1_recon_y_f32_host = vec![0.0f32; (l1_pred_w * l1_pred_h) as usize];
    let mut l1_recon_rgb_host = vec![0u8; (l1_pred_w * l1_pred_h * 3) as usize];
    let mut total_bytes = l2_bytes + l2_res_bytes;
    let mut manifest_tiles: Vec<serde_json::Value> = Vec::new();

    for ty in 0..l1_tiles_y {
        for tx in 0..l1_tiles_x {
            let tile_n = (tile_size * tile_size) as usize;

            // Extract L1 ground truth Y for this tile (from host L1 RGB)
            let l1_gt_y = extract_y_tile_from_rgb(
                &l1_rgb_host, l1_w, l1_h, tx * tile_size, ty * tile_size, tile_size, tile_size,
            );

            // Extract L1 prediction Y for this tile (from host f32 mosaic)
            let pred_y_tile = extract_f32_tile(
                &l1_pred_y_host, l1_pred_w, l1_pred_h,
                tx * tile_size, ty * tile_size, tile_size, tile_size,
            );

            // GPU: compute residual (upload tile data, compute on GPU, get result)
            let gt_y_dev = upload_u8(&gpu, &l1_gt_y)?;
            let pred_y_dev = upload_f32(&gpu, &pred_y_tile)?;
            let res_dev = gpu.compute_residual(&gt_y_dev, &pred_y_dev, tile_n as i32)?;

            // GPU: encode residual as grayscale JPEG via nvJPEG
            let res_jpeg = jpeg.encode_gray(&gpu, &res_dev, tile_size, tile_size, l1q)?;
            total_bytes += res_jpeg.len();
            fs::write(l1_out.join(format!("{}_{}.jpg", tx, ty)), &res_jpeg)?;

            // GPU: decode residual back for reconstruction
            let (decoded_res_dev, _, _) = jpeg.decode_gray_to_device(&gpu, &res_jpeg)?;

            // GPU: reconstruct Y
            let recon_y_dev = gpu.reconstruct_y(&pred_y_dev, &decoded_res_dev, tile_n as i32)?;
            gpu.sync()?;
            let recon_y_tile = download_f32(&gpu, &recon_y_dev)?;

            // Extract Cb/Cr prediction for this tile
            let pred_cb_tile = extract_f32_tile(
                &l1_pred_cb_host, l1_pred_w, l1_pred_h,
                tx * tile_size, ty * tile_size, tile_size, tile_size,
            );
            let pred_cr_tile = extract_f32_tile(
                &l1_pred_cr_host, l1_pred_w, l1_pred_h,
                tx * tile_size, ty * tile_size, tile_size, tile_size,
            );

            // Store reconstructed Y + RGB into L1 mosaic (host)
            for y in 0..tile_size {
                for x in 0..tile_size {
                    let ti = (y * tile_size + x) as usize;
                    let mx = tx * tile_size + x;
                    let my = ty * tile_size + y;
                    if mx < l1_pred_w && my < l1_pred_h {
                        let mi = (my * l1_pred_w + mx) as usize;
                        let recon = recon_y_tile[ti];
                        l1_recon_y_f32_host[mi] = recon;
                        let cbf = pred_cb_tile[ti] - 128.0;
                        let crf = pred_cr_tile[ti] - 128.0;
                        let r = (recon + 1.402 * crf).round().clamp(0.0, 255.0) as u8;
                        let g = (recon - 0.344136 * cbf - 0.714136 * crf).round().clamp(0.0, 255.0) as u8;
                        let b = (recon + 1.772 * cbf).round().clamp(0.0, 255.0) as u8;
                        let ri = mi * 3;
                        l1_recon_rgb_host[ri] = r;
                        l1_recon_rgb_host[ri + 1] = g;
                        l1_recon_rgb_host[ri + 2] = b;
                    }
                }
            }

            if config.manifest {
                let mut mse_sum = 0.0f64;
                for i in 0..l1_gt_y.len() {
                    let ry = (i as u32) / tile_size;
                    let rx = (i as u32) % tile_size;
                    let my = ty * tile_size + ry;
                    let mx = tx * tile_size + rx;
                    if mx < l1_pred_w && my < l1_pred_h {
                        let recon_val = l1_recon_y_f32_host[(my * l1_pred_w + mx) as usize] as f64;
                        let gt_val = l1_gt_y[i] as f64;
                        let diff = recon_val - gt_val;
                        mse_sum += diff * diff;
                    }
                }
                let mse = mse_sum / l1_gt_y.len() as f64;
                let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 100.0 };
                manifest_tiles.push(serde_json::json!({
                    "level": "L1", "tx": tx, "ty": ty,
                    "residual_bytes": res_jpeg.len(),
                    "y_psnr_db": (psnr * 100.0).round() / 100.0,
                    "y_mse": (mse * 100.0).round() / 100.0,
                }));
            }

            if config.debug_images {
                let compress_dir = output_dir.join("compress");
                let decompress_dir = output_dir.join("decompress");
                let step = 10 + (ty * l1_tiles_x + tx) as u32 * 10;
                // L1 tile original
                let l1_tile_rgb = extract_rgb_tile(
                    &l1_rgb_host, l1_w, l1_h, tx * tile_size, ty * tile_size, tile_size, tile_size,
                );
                save_rgb_png(&compress_dir.join(format!("{:03}_L1_{}_{}_original.png", step, tx, ty)),
                    &l1_tile_rgb, tile_size, tile_size)?;
                save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_luma.png", step + 1, tx, ty)),
                    &l1_gt_y, tile_size, tile_size)?;
                // Chroma channels (from prediction, quantized for debug visualization)
                let pred_cb_u8: Vec<u8> = pred_cb_tile.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
                let pred_cr_u8: Vec<u8> = pred_cr_tile.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
                save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_chroma_cb.png", step + 2, tx, ty)),
                    &pred_cb_u8, tile_size, tile_size)?;
                save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_chroma_cr.png", step + 3, tx, ty)),
                    &pred_cr_u8, tile_size, tile_size)?;
                // Prediction RGB (constructed from predicted YCbCr)
                let pred_rgb_tile = rgb_from_ycbcr_f32(&pred_y_tile, &pred_cb_tile, &pred_cr_tile);
                save_rgb_png(&compress_dir.join(format!("{:03}_L1_{}_{}_prediction.png", step + 4, tx, ty)),
                    &pred_rgb_tile, tile_size, tile_size)?;
                // Raw residual (normalized to [0,255] for visualization)
                let residual_raw: Vec<f32> = l1_gt_y.iter().zip(pred_y_tile.iter())
                    .map(|(&gt_val, &pred_val)| gt_val as f32 - pred_val).collect();
                let raw_normalized = normalize_f32_to_u8(&residual_raw);
                save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_residual_raw.png", step + 5, tx, ty)),
                    &raw_normalized, tile_size, tile_size)?;
                gpu.sync()?;
                let centered_host = download_u8(&gpu, &res_dev)?;
                save_gray_png(&compress_dir.join(format!("{:03}_L1_{}_{}_residual_centered.png", step + 7, tx, ty)),
                    &centered_host, tile_size, tile_size)?;
                let recon_rgb = rgb_from_ycbcr_f32(&recon_y_tile, &pred_cb_tile, &pred_cr_tile);
                save_rgb_png(&decompress_dir.join(format!("{:03}_L1_{}_{}_reconstructed.png", 63, tx, ty)),
                    &recon_rgb, tile_size, tile_size)?;
            }
        }
    }

    // 9. GPU: upsample L1 recon → L0 prediction
    let l1_recon_u8_dev = upload_u8(&gpu, &l1_recon_rgb_host)?;
    let l1_recon_f32_dev = gpu.u8_to_f32(&l1_recon_u8_dev, (l1_pred_w * l1_pred_h * 3) as i32)?;
    let l0_pred_f32 = gpu.upsample_bilinear_2x(
        &l1_recon_f32_dev, 1, l1_pred_h as i32, l1_pred_w as i32, 3,
    )?;
    let l0_pred_w = l1_pred_w * 2;
    let l0_pred_h = l1_pred_h * 2;

    // Debug: L0 mosaic prediction
    if config.debug_images {
        let decompress_dir = output_dir.join("decompress");
        gpu.sync()?;
        let l0_pred_host = gpu.stream.clone_dtoh(
            &gpu.f32_to_u8(&l0_pred_f32, (l0_pred_w * l0_pred_h * 3) as i32)?
        ).map_err(|e| anyhow::anyhow!("dtoh: {}", e))?;
        save_rgb_png(&decompress_dir.join("065_L0_mosaic_prediction.png"), &l0_pred_host, l0_pred_w, l0_pred_h)?;
    }

    // GPU: rgb_to_ycbcr_f32 on L0 prediction
    let l0_pred_u8 = gpu.f32_to_u8(&l0_pred_f32, (l0_pred_w * l0_pred_h * 3) as i32)?;
    let l0_pred_pixels = (l0_pred_w * l0_pred_h) as i32;
    let (l0_pred_y_dev, _l0_pred_cb_dev, _l0_pred_cr_dev) =
        gpu.rgb_to_ycbcr_f32(&l0_pred_u8, l0_pred_pixels)?;

    // Download L0 prediction Y for tile extraction
    gpu.sync()?;
    let l0_pred_y_host = download_f32(&gpu, &l0_pred_y_dev)?;
    let l0_pred_cb_host = if config.debug_images {
        download_f32(&gpu, &_l0_pred_cb_dev)?
    } else {
        Vec::new()
    };
    let l0_pred_cr_host = if config.debug_images {
        download_f32(&gpu, &_l0_pred_cr_dev)?
    } else {
        Vec::new()
    };

    // 10. Process L0 residuals
    let l0_out_dir = output_dir.join("L0").join("0_0");
    fs::create_dir_all(&l0_out_dir)?;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let tile_n = (tile_size * tile_size) as usize;

            // Extract L0 ground truth Y from source image
            let l0_gt_y = extract_y_tile_from_rgb(
                &src_rgb_host, src_w, src_h, tx * tile_size, ty * tile_size, tile_size, tile_size,
            );

            // Extract L0 prediction Y
            let pred_y_tile = extract_f32_tile(
                &l0_pred_y_host, l0_pred_w, l0_pred_h,
                tx * tile_size, ty * tile_size, tile_size, tile_size,
            );

            // GPU: compute residual
            let gt_y_dev = upload_u8(&gpu, &l0_gt_y)?;
            let pred_y_dev = upload_f32(&gpu, &pred_y_tile)?;
            let res_dev = gpu.compute_residual(&gt_y_dev, &pred_y_dev, tile_n as i32)?;

            // GPU: encode residual via nvJPEG
            let res_jpeg = jpeg.encode_gray(&gpu, &res_dev, tile_size, tile_size, l0q)?;
            total_bytes += res_jpeg.len();
            fs::write(l0_out_dir.join(format!("{}_{}.jpg", tx, ty)), &res_jpeg)?;

            if config.manifest {
                let (decoded_res_dev, _, _) = jpeg.decode_gray_to_device(&gpu, &res_jpeg)?;
                gpu.sync()?;
                let decoded_res_host = download_u8(&gpu, &decoded_res_dev)?;

                let mut mse_sum = 0.0f64;
                for i in 0..l0_gt_y.len() {
                    let pred_val = pred_y_tile[i] as f64;
                    let res_val = decoded_res_host[i] as f64 - 128.0;
                    let recon = (pred_val + res_val).max(0.0).min(255.0);
                    let gt_val = l0_gt_y[i] as f64;
                    let diff = recon - gt_val;
                    mse_sum += diff * diff;
                }
                let mse = mse_sum / l0_gt_y.len() as f64;
                let psnr = if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { 100.0 };
                manifest_tiles.push(serde_json::json!({
                    "level": "L0", "tx": tx, "ty": ty,
                    "residual_bytes": res_jpeg.len(),
                    "y_psnr_db": (psnr * 100.0).round() / 100.0,
                    "y_mse": (mse * 100.0).round() / 100.0,
                }));
            }

            if config.debug_images {
                let compress_dir = output_dir.join("compress");
                let decompress_dir = output_dir.join("decompress");
                let idx = (ty * tiles_x + tx) as u32;
                let step = 20 + idx;
                let l0_tile_rgb = extract_rgb_tile(
                    &src_rgb_host, src_w, src_h, tx * tile_size, ty * tile_size, tile_size, tile_size,
                );
                save_rgb_png(&compress_dir.join(format!("{:03}_L0_{}_{}_original.png", step, tx, ty)),
                    &l0_tile_rgb, tile_size, tile_size)?;
                save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_luma.png", step, tx, ty)),
                    &l0_gt_y, tile_size, tile_size)?;
                // Chroma channels (from prediction, quantized for debug visualization)
                let pred_cb_tile = extract_f32_tile(
                    &l0_pred_cb_host, l0_pred_w, l0_pred_h,
                    tx * tile_size, ty * tile_size, tile_size, tile_size,
                );
                let pred_cr_tile = extract_f32_tile(
                    &l0_pred_cr_host, l0_pred_w, l0_pred_h,
                    tx * tile_size, ty * tile_size, tile_size, tile_size,
                );
                let pred_cb_u8: Vec<u8> = pred_cb_tile.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
                let pred_cr_u8: Vec<u8> = pred_cr_tile.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
                save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_chroma_cb.png", step, tx, ty)),
                    &pred_cb_u8, tile_size, tile_size)?;
                save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_chroma_cr.png", step, tx, ty)),
                    &pred_cr_u8, tile_size, tile_size)?;
                // Prediction RGB (constructed from predicted YCbCr)
                let pred_rgb_tile = rgb_from_ycbcr_f32(&pred_y_tile, &pred_cb_tile, &pred_cr_tile);
                save_rgb_png(&compress_dir.join(format!("{:03}_L0_{}_{}_prediction.png", step, tx, ty)),
                    &pred_rgb_tile, tile_size, tile_size)?;
                // Raw residual (normalized to [0,255] for visualization)
                let residual_raw: Vec<f32> = l0_gt_y.iter().zip(pred_y_tile.iter())
                    .map(|(&gt_val, &pred_val)| gt_val as f32 - pred_val).collect();
                let raw_normalized = normalize_f32_to_u8(&residual_raw);
                save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_residual_raw.png", step, tx, ty)),
                    &raw_normalized, tile_size, tile_size)?;
                gpu.sync()?;
                let centered_host = download_u8(&gpu, &res_dev)?;
                save_gray_png(&compress_dir.join(format!("{:03}_L0_{}_{}_residual_centered.png", step, tx, ty)),
                    &centered_host, tile_size, tile_size)?;

                // L0 reconstructed: decode residual, reconstruct Y, convert to RGB
                let (decoded_res_dev, _, _) = jpeg.decode_gray_to_device(&gpu, &res_jpeg)?;
                let recon_y_dev = gpu.reconstruct_y(&pred_y_dev, &decoded_res_dev, tile_n as i32)?;
                gpu.sync()?;
                let recon_y_tile = download_f32(&gpu, &recon_y_dev)?;
                let recon_rgb = rgb_from_ycbcr_f32(&recon_y_tile, &pred_cb_tile, &pred_cr_tile);
                let recon_step = 70 + idx;
                save_rgb_png(&decompress_dir.join(format!("{:03}_L0_{}_{}_reconstructed.png", recon_step, tx, ty)),
                    &recon_rgb, tile_size, tile_size)?;
            }
        }
    }

    let elapsed = start.elapsed();
    let l1_count = (l1_tiles_x * l1_tiles_y) as usize;
    let l0_count = (tiles_x * tiles_y) as usize;
    info!(
        "Single-image encode complete: {} L1 + {} L0 tiles, {} total bytes, {:.1}s",
        l1_count, l0_count, total_bytes, elapsed.as_secs_f64()
    );

    // Write summary.json
    let summary = serde_json::json!({
        "mode": "single-image-gpu",
        "source": image_path.to_string_lossy(),
        "source_w": src_w, "source_h": src_h,
        "encoder": "nvjpeg",
        "subsamp": config.subsamp,
        "baseq": baseq, "l1q": l1q, "l0q": l0q,
        "l2resq": config.l2resq, "optl2": config.optl2, "sharpen": config.sharpen,
        "tile_size": tile_size,
        "l2_bytes": l2_bytes, "l2_res_bytes": l2_res_bytes,
        "l2_w": l2_w, "l2_h": l2_h,
        "l1_tiles": l1_count, "l0_tiles": l0_count,
        "total_bytes": total_bytes,
        "elapsed_secs": elapsed.as_secs_f64(),
    });
    fs::write(output_dir.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;

    if config.manifest {
        let manifest = serde_json::json!({
            "mode": "single-image-gpu",
            "source": image_path.to_string_lossy(),
            "source_w": src_w, "source_h": src_h,
            "encoder": "nvjpeg",
            "subsamp": config.subsamp,
            "baseq": baseq, "l1q": l1q, "l0q": l0q,
            "l2resq": config.l2resq, "optl2": config.optl2, "sharpen": config.sharpen,
            "tile_size": tile_size,
            "l2_bytes": l2_bytes, "l2_res_bytes": l2_res_bytes,
            "l2_w": l2_w, "l2_h": l2_h,
            "total_bytes": total_bytes,
            "tiles": manifest_tiles,
        });
        fs::write(output_dir.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    }

    Ok(SingleImageSummary {
        l1_tiles: l1_count,
        l0_tiles: l0_count,
        l2_bytes,
        total_bytes,
        elapsed_secs: elapsed.as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// DICOM WSI encode pipeline (GPU-native)
// ---------------------------------------------------------------------------

/// Encode a full WSI from DICOM to bundle format using GPU kernels + nvJPEG.
pub fn encode_wsi(
    dicom_path: &Path,
    output_dir: &Path,
    config: EncodeConfig,
) -> Result<EncodeSummary> {
    use crate::dicom::DicomSlide;

    let start = Instant::now();
    let profile = config.profile;
    let mut timers = PipelineTimers::default();

    info!("Opening DICOM: {}", dicom_path.display());
    let t_dicom = Instant::now();
    let slide = DicomSlide::open(dicom_path)
        .with_context(|| format!("Failed to open DICOM {}", dicom_path.display()))?;
    timers.dicom_open = t_dicom.elapsed();

    info!(
        "DICOM: {}x{} pixels, {} tiles ({}x{} grid), tile_size={}x{}",
        slide.width(), slide.height(), slide.tile_count(),
        slide.tiles_x(), slide.tiles_y(), slide.tile_w(), slide.tile_h(),
    );

    let tile_w = slide.tile_w();
    let tile_h = slide.tile_h();
    let tiles_per_row = 4u32;
    let canvas_w = tile_w * tiles_per_row;
    let canvas_h = tile_h * tiles_per_row;

    let grid_cols = (slide.tiles_x() + 3) / 4;
    let grid_rows = (slide.tiles_y() + 3) / 4;
    let total_families = (grid_cols * grid_rows) as usize;
    info!("Family grid: {}x{} = {} families", grid_cols, grid_rows, total_families);
    if profile {
        info!("Profiling enabled — extra GPU sync points active");
    }

    info!("Initializing CUDA device {}", config.device);
    let gpu = GpuContext::new(config.device)?;
    let jpeg = NvJpegHandle::new(&gpu)?;

    let mut monitor = if profile {
        match GpuMonitor::new(config.device as u32) {
            Some(m) => {
                info!("NVML GPU monitoring active");
                Some(m)
            }
            None => {
                info!("NVML unavailable — GPU resource monitoring disabled");
                None
            }
        }
    } else {
        None
    };

    fs::create_dir_all(output_dir)?;
    let files_dir = output_dir.join("baseline_pyramid_files");
    fs::create_dir_all(&files_dir)?;

    let mut total_l2_bytes = 0usize;
    let mut total_residual_bytes = 0usize;
    let mut families_encoded = 0usize;
    let mut bundle_entries: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(total_families);

    let mut families: Vec<(u32, u32)> = (0..grid_rows)
        .flat_map(|row| (0..grid_cols).map(move |col| (col, row)))
        .collect();

    if let Some(max) = config.max_parents {
        families.truncate(max);
        info!("Limited to {} families (--max-parents)", max);
    }

    timers.families_total = families.len();

    for batch_start in (0..families.len()).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(families.len());
        let batch = &families[batch_start..batch_end];

        info!(
            "Processing batch {}-{} of {} ({} families)",
            batch_start, batch_end, families.len(), batch.len(),
        );

        for &(col, row) in batch {
            // --- gather_tiles ---
            let t0 = Instant::now();
            let mut tile_jpeg_bytes: Vec<Vec<u8>> = Vec::with_capacity(16);
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let tx = col * 4 + dx;
                    let ty = row * 4 + dy;
                    tile_jpeg_bytes.push(slide.get_tile_bytes(tx, ty).unwrap_or_default());
                }
            }
            timers.gather_tiles += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("gather_tiles"); }

            // Early skip: if ALL 16 tiles are empty, skip this family entirely
            let has_data = tile_jpeg_bytes.iter().any(|b| !b.is_empty());
            if !has_data {
                timers.families_empty += 1;
                families_encoded += 1;
                continue;
            }

            // --- nvjpeg_decode ---
            if profile { gpu.sync()?; }
            let t0 = Instant::now();
            let decoded_tiles = jpeg.batch_decode_to_device(&gpu, &tile_jpeg_bytes)?;
            if profile { gpu.sync()?; }
            timers.nvjpeg_decode += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("nvjpeg_decode"); }

            // --- composite ---
            let t0 = Instant::now();
            let cw = canvas_w as usize;
            let ch = canvas_h as usize;
            let tw = tile_w as usize;
            let th = tile_h as usize;
            let mut canvas_host = vec![0u8; cw * ch * 3];

            gpu.sync()?;
            let tile_hosts: Vec<Option<(Vec<u8>, usize, usize)>> = decoded_tiles.iter()
                .map(|(tile_dev, t_w, t_h)| {
                    if *t_w == 0 || *t_h == 0 { return Ok(None); }
                    let host = download_u8(&gpu, tile_dev)?;
                    Ok(Some((host, *t_w as usize, *t_h as usize)))
                })
                .collect::<Result<Vec<_>>>()?;

            for (tile_idx, tile_opt) in tile_hosts.iter().enumerate() {
                let Some((tile_host, ow, oh)) = tile_opt else { continue };
                let ow = *ow;
                let oh = *oh;
                let dy = tile_idx / 4;
                let dx = tile_idx % 4;
                let copy_w = ow.min(tw);
                for y in 0..oh.min(th) {
                    let cx = dx * tw;
                    let cy = dy * th + y;
                    if cy < ch && cx + copy_w <= cw {
                        let si = y * ow * 3;
                        let di = (cy * cw + cx) * 3;
                        canvas_host[di..di + copy_w * 3]
                            .copy_from_slice(&tile_host[si..si + copy_w * 3]);
                    }
                }
            }

            let canvas_u8_dev = upload_u8(&gpu, &canvas_host)?;
            let canvas_f32_dev = gpu.u8_to_f32(&canvas_u8_dev, (canvas_w * canvas_h * 3) as i32)?;
            timers.composite += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("composite"); }

            // --- downsample ---
            if profile { gpu.sync()?; }
            let t0 = Instant::now();
            let l1_w = canvas_w / 2;
            let l1_h = canvas_h / 2;
            let l2_w = canvas_w / 4;
            let l2_h = canvas_h / 4;

            let l1_f32 = gpu.downsample_lanczos3(
                &canvas_f32_dev, 1, canvas_h as i32, canvas_w as i32, l1_h as i32, l1_w as i32, 3,
            )?;
            let l2_f32 = gpu.downsample_lanczos3(
                &canvas_f32_dev, 1, canvas_h as i32, canvas_w as i32, l2_h as i32, l2_w as i32, 3,
            )?;

            let l1_u8 = gpu.f32_to_u8(&l1_f32, (l1_w * l1_h * 3) as i32)?;
            let mut l2_u8 = gpu.f32_to_u8(&l2_f32, (l2_w * l2_h * 3) as i32)?;
            if profile { gpu.sync()?; }
            timers.downsample += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("downsample"); }

            // --- optl2 ---
            if config.optl2 {
                if profile { gpu.sync()?; }
                let t0 = Instant::now();
                let l2_opt = gpu.u8_to_f32(&l2_u8, (l2_w * l2_h * 3) as i32)?;
                let l1_target = gpu.u8_to_f32(&l1_u8, (l1_w * l1_h * 3) as i32)?;
                let mut l2_cur = gpu.u8_to_f32(&l2_u8, (l2_w * l2_h * 3) as i32)?;
                for _ in 0..100 {
                    gpu.optl2_step(&mut l2_cur, &l2_opt, &l1_target,
                        1, l2_h as i32, l2_w as i32, 0.3, config.max_delta as f32)?;
                }
                l2_u8 = gpu.f32_to_u8(&l2_cur, (l2_w * l2_h * 3) as i32)?;
                if profile { gpu.sync()?; }
                timers.optl2 += t0.elapsed();
                if let Some(m) = &mut monitor { m.sample("optl2"); }
            }

            // --- sharpen before save (if --save-sharpened) ---
            if config.save_sharpened {
                if let Some(strength) = config.sharpen {
                    l2_u8 = gpu.sharpen_l2(&l2_u8, l2_w, l2_h, strength)?;
                }
            }

            // --- l2_encode ---
            if profile { gpu.sync()?; }
            let t0 = Instant::now();
            let l2_jpeg = jpeg.encode_rgb(&gpu, &l2_u8, l2_w, l2_h, config.baseq, &config.subsamp)?;
            total_l2_bytes += l2_jpeg.len();

            let l2_level_dir = files_dir.join("0");
            fs::create_dir_all(&l2_level_dir)?;
            fs::write(l2_level_dir.join(format!("{}_{}.jpg", col, row)), &l2_jpeg)?;
            timers.l2_encode += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("l2_encode"); }

            // --- l2_decode ---
            if profile { gpu.sync()?; }
            let t0 = Instant::now();
            let (l2_decoded, _, _) = jpeg.decode_to_device(&gpu, &l2_jpeg)?;
            if profile { gpu.sync()?; }
            timers.l2_decode += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("l2_decode"); }

            // --- sharpen decoded L2 for prediction (only when not already sharpened before save) ---
            let l2_for_pred = if !config.save_sharpened {
                if let Some(strength) = config.sharpen {
                    gpu.sharpen_l2(&l2_decoded, l2_w, l2_h, strength)?
                } else {
                    l2_decoded
                }
            } else {
                l2_decoded
            };

            // --- l1_predict ---
            if profile { gpu.sync()?; }
            let t0 = Instant::now();
            let l2_pred_f32 = gpu.u8_to_f32(&l2_for_pred, (l2_w * l2_h * 3) as i32)?;
            let l1_pred_f32 = gpu.upsample_bilinear_2x(&l2_pred_f32, 1, l2_h as i32, l2_w as i32, 3)?;
            let l1_pred_u8 = gpu.f32_to_u8(&l1_pred_f32, (l1_w * l1_h * 3) as i32)?;
            let l1_pred_pixels = (l1_w * l1_h) as i32;
            let (l1_pred_y, l1_pred_cb, l1_pred_cr) = gpu.rgb_to_ycbcr_f32(&l1_pred_u8, l1_pred_pixels)?;

            gpu.sync()?;
            let l1_pred_y_host = download_f32(&gpu, &l1_pred_y)?;
            let l1_rgb_host = download_u8(&gpu, &l1_u8)?;
            timers.l1_predict += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("l1_predict"); }

            // --- l1_residuals ---
            let t0 = Instant::now();
            let l1_tile_w = tile_w * 2;
            let l1_tile_h = tile_h * 2;
            let l1_pixels = (l1_w * l1_h) as usize;
            let mut l1_recon_y_host = vec![0.0f32; l1_pixels];
            let mut pack_entries: Vec<origami::core::pack::PackWriteEntry> = Vec::new();

            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let ox = dx * l1_tile_w;
                    let oy = dy * l1_tile_h;
                    let tw = l1_tile_w.min(l1_w.saturating_sub(ox));
                    let th = l1_tile_h.min(l1_h.saturating_sub(oy));
                    if tw == 0 || th == 0 { continue; }

                    let tile_n = (tw * th) as usize;
                    let gt_y = extract_y_tile_from_rgb(&l1_rgb_host, l1_w, l1_h, ox, oy, tw, th);
                    let pred_y_tile = extract_f32_tile(&l1_pred_y_host, l1_w, l1_h, ox, oy, tw, th);

                    let gt_y_dev = upload_u8(&gpu, &gt_y)?;
                    let pred_y_dev = upload_f32(&gpu, &pred_y_tile)?;
                    let res_dev = gpu.compute_residual(&gt_y_dev, &pred_y_dev, tile_n as i32)?;

                    let jpeg_data = jpeg.encode_gray(&gpu, &res_dev, tw, th, config.l1q)?;
                    total_residual_bytes += jpeg_data.len();

                    let (decoded_res, _, _) = jpeg.decode_gray_to_device(&gpu, &jpeg_data)?;
                    let recon_y_dev = gpu.reconstruct_y(&pred_y_dev, &decoded_res, tile_n as i32)?;
                    gpu.sync()?;
                    let recon_y_tile = download_f32(&gpu, &recon_y_dev)?;

                    for y in 0..th {
                        for x in 0..tw {
                            let mx = ox + x;
                            let my = oy + y;
                            if mx < l1_w && my < l1_h {
                                l1_recon_y_host[(my * l1_w + mx) as usize] =
                                    recon_y_tile[(y * tw + x) as usize];
                            }
                        }
                    }

                    if config.pack {
                        pack_entries.push(origami::core::pack::PackWriteEntry {
                            level_kind: 1,
                            idx_in_parent: (dy * 2 + dx) as u8,
                            jpeg_data,
                        });
                    }
                    timers.l1_tiles_processed += 1;
                }
            }
            if profile { gpu.sync()?; }
            timers.l1_residuals += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("l1_residuals"); }

            // --- l1_recon_rgb ---
            if profile { gpu.sync()?; }
            let t0 = Instant::now();
            let l1_recon_y_dev = upload_f32(&gpu, &l1_recon_y_host)?;
            let l1_recon_rgb_dev = gpu.ycbcr_to_rgb(
                &l1_recon_y_dev, &l1_pred_cb, &l1_pred_cr, l1_pixels as i32,
            )?;
            if profile { gpu.sync()?; }
            timers.l1_recon_rgb += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("l1_recon_rgb"); }

            // --- l0_predict ---
            if profile { gpu.sync()?; }
            let t0 = Instant::now();
            let l1_recon_f32 = gpu.u8_to_f32(&l1_recon_rgb_dev, (l1_w * l1_h * 3) as i32)?;
            let l0_pred_f32 = gpu.upsample_bilinear_2x(&l1_recon_f32, 1, l1_h as i32, l1_w as i32, 3)?;
            let l0_pred_u8 = gpu.f32_to_u8(&l0_pred_f32, (canvas_w * canvas_h * 3) as i32)?;
            let l0_pred_pixels = (canvas_w * canvas_h) as i32;
            let (l0_pred_y, _, _) = gpu.rgb_to_ycbcr_f32(&l0_pred_u8, l0_pred_pixels)?;
            gpu.sync()?;
            let l0_pred_y_host = download_f32(&gpu, &l0_pred_y)?;
            timers.l0_predict += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("l0_predict"); }

            // --- l0_residuals ---
            let t0 = Instant::now();
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let abs_tx = col * 4 + dx;
                    let abs_ty = row * 4 + dy;
                    if abs_tx >= slide.tiles_x() || abs_ty >= slide.tiles_y() { continue; }

                    let tile_idx = (dy * 4 + dx) as usize;
                    let (tile_dev, t_w, t_h) = &decoded_tiles[tile_idx];
                    if *t_w == 0 || *t_h == 0 {
                        timers.l0_tiles_empty += 1;
                        continue;
                    }

                    let tw = *t_w;
                    let th = *t_h;
                    let tile_n = (tw * th) as usize;

                    let (gt_y_f32, _, _) = gpu.rgb_to_ycbcr_f32(tile_dev, tile_n as i32)?;
                    let gt_y_u8 = gpu.f32_to_u8(&gt_y_f32, tile_n as i32)?;

                    let ox = dx * tile_w;
                    let oy = dy * tile_h;
                    let pred_y_tile = extract_f32_tile(&l0_pred_y_host, canvas_w, canvas_h, ox, oy, tw, th);
                    let pred_y_dev = upload_f32(&gpu, &pred_y_tile)?;

                    let res_dev = gpu.compute_residual(&gt_y_u8, &pred_y_dev, tile_n as i32)?;
                    let jpeg_data = jpeg.encode_gray(&gpu, &res_dev, tw, th, config.l0q)?;
                    total_residual_bytes += jpeg_data.len();

                    if config.pack {
                        pack_entries.push(origami::core::pack::PackWriteEntry {
                            level_kind: 0,
                            idx_in_parent: (dy * 4 + dx) as u8,
                            jpeg_data,
                        });
                    }
                    timers.l0_tiles_processed += 1;
                }
            }
            if profile { gpu.sync()?; }
            timers.l0_residuals += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("l0_residuals"); }

            // --- pack ---
            let t0 = Instant::now();
            if config.pack && !pack_entries.is_empty() {
                let lz4_data = origami::core::pack::compress_pack_entries(&pack_entries);
                bundle_entries.push((col, row, lz4_data));
            }
            timers.pack += t0.elapsed();
            if let Some(m) = &mut monitor { m.sample("pack"); }

            families_encoded += 1;
        }
    }

    // --- generate_pyramid (L3, L4, ... down to 1 tile) ---
    if config.generate_pyramid {
        let t0 = Instant::now();
        match config.pyramid_mode.as_str() {
            "gpu" => {
                info!("Generating DZI pyramid on GPU...");
                generate_dzi_pyramid_gpu(
                    &gpu,
                    &jpeg,
                    &files_dir,
                    config.tile_size,
                    grid_cols as usize,
                    grid_rows as usize,
                    config.baseq,
                    &config.subsamp,
                    config.pyramid_sharpen,
                )?;
                info!("Generated DZI pyramid (GPU): {:.2}s", t0.elapsed().as_secs_f64());
            }
            "cpu" => {
                info!("Generating DZI pyramid on CPU ({} cores)...", rayon::current_num_threads());
                generate_dzi_pyramid_cpu(
                    &files_dir,
                    config.tile_size,
                    grid_cols as usize,
                    grid_rows as usize,
                    config.baseq,
                    &config.subsamp,
                    config.pyramid_sharpen,
                )?;
                info!("Generated DZI pyramid (CPU): {:.2}s", t0.elapsed().as_secs_f64());
            }
            _ => anyhow::bail!("Invalid pyramid_mode: {}", config.pyramid_mode),
        }
    }

    // --- bundle_write ---
    let t0 = Instant::now();
    if config.pack && !bundle_entries.is_empty() {
        let bundle_dir = output_dir.join("residual_packs");
        fs::create_dir_all(&bundle_dir)?;
        let bundle_path = bundle_dir.join("residuals.bundle");
        origami::core::pack::write_bundle(
            &bundle_path, grid_cols as u16, grid_rows as u16, &bundle_entries,
        )?;
        info!("Wrote bundle: {}", bundle_path.display());
    }
    timers.bundle_write = t0.elapsed();

    let elapsed = start.elapsed().as_secs_f64();

    // Print and save timing report
    if profile {
        timers.print_report();
        if let Some(ref m) = monitor {
            m.print_summary();
        }

        let mut timing_json = timers.to_json();
        if let Some(ref m) = monitor {
            if let serde_json::Value::Object(ref mut map) = timing_json {
                map.insert("gpu_resources".to_string(), m.to_json());
            }
        }
        fs::write(
            output_dir.join("timing_report.json"),
            serde_json::to_string_pretty(&timing_json)?,
        )?;
        info!("Wrote timing_report.json");
    }

    let summary_json = serde_json::json!({
        "mode": "gpu-encode",
        "dicom": dicom_path.to_string_lossy(),
        "tile_size": config.tile_size,
        "baseq": config.baseq, "l1q": config.l1q, "l0q": config.l0q,
        "optl2": config.optl2, "sharpen": config.sharpen,
        "families": families_encoded,
        "families_empty": timers.families_empty,
        "grid_cols": grid_cols, "grid_rows": grid_rows,
        "l2_bytes": total_l2_bytes,
        "residual_bytes": total_residual_bytes,
        "elapsed_secs": elapsed,
    });
    fs::write(output_dir.join("summary.json"), serde_json::to_string_pretty(&summary_json)?)?;

    Ok(EncodeSummary {
        families_encoded,
        l2_bytes: total_l2_bytes,
        residual_bytes: total_residual_bytes,
        elapsed_secs: elapsed,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract Y (luma) channel from a tile region of an interleaved RGB buffer.
fn extract_y_tile_from_rgb(
    rgb: &[u8], mosaic_w: u32, mosaic_h: u32,
    ox: u32, oy: u32, tw: u32, th: u32,
) -> Vec<u8> {
    let mut y = Vec::with_capacity((tw * th) as usize);
    for py in 0..th {
        for px in 0..tw {
            let mx = ox + px;
            let my = oy + py;
            if mx < mosaic_w && my < mosaic_h {
                let si = ((my * mosaic_w + mx) * 3) as usize;
                let r = rgb[si] as f32;
                let g = rgb[si + 1] as f32;
                let b = rgb[si + 2] as f32;
                y.push((0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8);
            } else {
                y.push(0);
            }
        }
    }
    y
}

/// Extract a tile region of f32 values from a mosaic.
fn extract_f32_tile(
    mosaic: &[f32], mosaic_w: u32, mosaic_h: u32,
    ox: u32, oy: u32, tw: u32, th: u32,
) -> Vec<f32> {
    let mut tile = vec![0.0f32; (tw * th) as usize];
    for y in 0..th {
        for x in 0..tw {
            let mx = ox + x;
            let my = oy + y;
            if mx < mosaic_w && my < mosaic_h {
                tile[(y * tw + x) as usize] = mosaic[(my * mosaic_w + mx) as usize];
            }
        }
    }
    tile
}

/// Extract RGB tile from mosaic (for debug images).
fn extract_rgb_tile(
    rgb: &[u8], mosaic_w: u32, mosaic_h: u32,
    ox: u32, oy: u32, tw: u32, th: u32,
) -> Vec<u8> {
    let mut tile = vec![0u8; (tw * th * 3) as usize];
    for y in 0..th {
        for x in 0..tw {
            let mx = ox + x;
            let my = oy + y;
            if mx < mosaic_w && my < mosaic_h {
                let si = ((my * mosaic_w + mx) * 3) as usize;
                let di = ((y * tw + x) * 3) as usize;
                tile[di] = rgb[si];
                tile[di + 1] = rgb[si + 1];
                tile[di + 2] = rgb[si + 2];
            }
        }
    }
    tile
}

/// Normalize f32 values to [0, 255] u8 range (min→0, max→255) for debug visualization.
fn normalize_f32_to_u8(data: &[f32]) -> Vec<u8> {
    if data.is_empty() { return Vec::new(); }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range < 1e-6 {
        return vec![128u8; data.len()];
    }
    data.iter().map(|&v| ((v - min) / range * 255.0).round().clamp(0.0, 255.0) as u8).collect()
}

/// Convert f32 Y, Cb, Cr to interleaved RGB u8 (BT.601 inverse, for debug images).
fn rgb_from_ycbcr_f32(y: &[f32], cb: &[f32], cr: &[f32]) -> Vec<u8> {
    let n = y.len();
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        let yf = y[i];
        let cbf = cb[i] - 128.0;
        let crf = cr[i] - 128.0;
        let r = (yf + 1.402 * crf).round().clamp(0.0, 255.0) as u8;
        let g = (yf - 0.344136 * cbf - 0.714136 * crf).round().clamp(0.0, 255.0) as u8;
        let b = (yf + 1.772 * cbf).round().clamp(0.0, 255.0) as u8;
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    rgb
}


/// Generate full DZI pyramid from L2 baseline tiles.
/// Generate full DZI pyramid from L2 baseline tiles (GPU-accelerated).
/// 1. Decode L2 tiles and composite on CPU (image crate)
/// 2. Upload to GPU
/// 3. Repeatedly: Lanczos3 downsample 2×, tile, nvJPEG encode
/// GPU pyramid generation: box filter downsample + unsharp mask + nvJPEG encode.
/// Keeps image on GPU throughout pipeline for maximum performance.
fn generate_dzi_pyramid_gpu(
    gpu: &GpuContext,
    jpeg: &NvJpegHandle,
    files_dir: &Path,
    tile_size: u32,
    grid_cols: usize,
    grid_rows: usize,
    quality: u8,
    subsamp: &str,
    sharpen_strength: f32,
) -> Result<()> {
    let l2_dir = files_dir.join("0");
    let ts = tile_size as usize;
    let l2_width = grid_cols * ts;
    let l2_height = grid_rows * ts;

    info!("GPU pyramid generation: compositing {}×{} L2 tiles → {}x{} image",
          grid_cols, grid_rows, l2_width, l2_height);

    // --- Step 1: Decode all L2 tiles and composite on CPU (using image crate) ---
    let mut l2_image = vec![0u8; l2_width * l2_height * 3];
    for row in 0..grid_rows {
        for col in 0..grid_cols {
            let tile_path = l2_dir.join(format!("{}_{}.jpg", col, row));
            if !tile_path.exists() {
                continue; // skip empty tiles
            }
            let tile_bytes = fs::read(&tile_path)?;
            let tile_img = image::load_from_memory(&tile_bytes)?.to_rgb8();

            // Copy tile into full image
            for ty in 0..ts.min(tile_img.height() as usize) {
                for tx in 0..ts.min(tile_img.width() as usize) {
                    let src_idx = (ty * ts + tx) * 3;
                    let dst_x = col * ts + tx;
                    let dst_y = row * ts + ty;
                    if dst_x < l2_width && dst_y < l2_height {
                        let dst_idx = (dst_y * l2_width + dst_x) * 3;
                        l2_image[dst_idx..dst_idx + 3].copy_from_slice(&tile_img.as_raw()[src_idx..src_idx + 3]);
                    }
                }
            }
        }
    }

    // --- Step 2: Upload full L2 image to GPU ---
    let l2_dev = gpu.stream.clone_htod(&l2_image)
        .map_err(|e| anyhow::anyhow!("upload L2 failed: {}", e))?;

    // --- Step 3: Iteratively downsample and encode pyramid levels ---
    let mut current_dev = l2_dev;
    let mut current_w = l2_width as u32;
    let mut current_h = l2_height as u32;
    let mut level = 1; // Start at L3 (one below L2)

    loop {
        // Downsample by 2× using box filter on GPU (fast 2×2 averaging)
        let new_w = (current_w / 2).max(1);
        let new_h = (current_h / 2).max(1);

        let downsampled_dev = gpu.downsample_box_2x(&current_dev, current_w, current_h)?;

        // Apply unsharp mask to recover edge definition
        let sharpened_dev = gpu.sharpen_l2(&downsampled_dev, new_w, new_h, sharpen_strength)?;

        // Tile the sharpened image
        let cols = (((new_w as usize) + ts - 1) / ts);
        let rows = (((new_h as usize) + ts - 1) / ts);

        let level_dir = files_dir.join(format!("{}", level));
        fs::create_dir_all(&level_dir)?;

        info!("  Level {}: {}x{} → {}×{} tiles", level, new_w, new_h, cols, rows);

        // Download full sharpened image to CPU for tiling
        let sharpened_host: Vec<u8> = gpu.stream.clone_dtoh(&sharpened_dev)
            .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

        // Extract and encode each tile
        for row in 0..rows {
            for col in 0..cols {
                let x = (col as u32) * tile_size;
                let y = (row as u32) * tile_size;
                let w = if x < new_w { tile_size.min(new_w - x) } else { 0 };
                let h = if y < new_h { tile_size.min(new_h - y) } else { 0 };

                // Extract tile region (with black padding if needed)
                let mut tile_rgb = vec![0u8; (tile_size * tile_size * 3) as usize];
                for ty in 0..h {
                    for tx in 0..w {
                        let src_idx = (((y + ty) as usize * new_w as usize + (x + tx) as usize) * 3);
                        let dst_idx = ((ty * tile_size + tx) as usize * 3);
                        if src_idx + 2 < sharpened_host.len() {
                            tile_rgb[dst_idx..dst_idx + 3].copy_from_slice(&sharpened_host[src_idx..src_idx + 3]);
                        }
                    }
                }

                // Upload tile to GPU and encode with nvJPEG
                let tile_dev = gpu.stream.clone_htod(&tile_rgb)
                    .map_err(|e| anyhow::anyhow!("tile upload failed: {}", e))?;
                let jpeg_bytes = jpeg.encode_rgb(gpu, &tile_dev, tile_size, tile_size, quality, subsamp)?;

                fs::write(level_dir.join(format!("{}_{}.jpg", col, row)), &jpeg_bytes)?;
            }
        }

        // Move to next level
        current_dev = sharpened_dev;
        current_w = new_w;
        current_h = new_h;
        level += 1;

        // Stop when we reach 1 tile
        if cols == 1 && rows == 1 {
            break;
        }
    }

    info!("Generated {} pyramid levels (L3..L{}) using box+sharpen", level - 1, level);
    Ok(())
}

/// Generate full DZI pyramid from L2 baseline tiles (CPU version).
/// CPU pyramid generation: box filter downsample + unsharp mask + turbojpeg encode.
/// Fast alternative to Lanczos3: ~50× faster with comparable visual quality for preview levels.
fn generate_dzi_pyramid_cpu(
    files_dir: &Path,
    tile_size: u32,
    grid_cols: usize,
    grid_rows: usize,
    quality: u8,
    subsamp: &str,
    sharpen_strength: f32,
) -> Result<()> {
    use turbojpeg::{Compressor, Image, PixelFormat, Subsamp};
    use rayon::prelude::*;
    use std::sync::Mutex;

    let ts = tile_size as usize;
    let l2_dir = files_dir.join("0");

    // Composite all L2 tiles into full L2 image (CPU RAM)
    let l2_width = grid_cols * ts;
    let l2_height = grid_rows * ts;
    let mut l2_image = vec![0u8; l2_width * l2_height * 3];

    info!("CPU pyramid generation: compositing {}×{} L2 tiles → {}x{} image (using {} CPU cores)",
          grid_cols, grid_rows, l2_width, l2_height, rayon::current_num_threads());

    let l2_image_mutex = Mutex::new(&mut l2_image);

    // Parallel decode and composite L2 tiles
    (0..grid_rows).into_par_iter().for_each(|row| {
        for col in 0..grid_cols {
            let tile_path = l2_dir.join(format!("{}_{}.jpg", col, row));
            if !tile_path.exists() {
                continue;
            }
            let tile_bytes = match fs::read(&tile_path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let tile_img = match image::load_from_memory(&tile_bytes) {
                Ok(img) => img.to_rgb8(),
                Err(_) => continue,
            };

            // Copy tile into full image (mutex-protected)
            let mut l2_img = l2_image_mutex.lock().unwrap();
            for ty in 0..ts.min(tile_img.height() as usize) {
                for tx in 0..ts.min(tile_img.width() as usize) {
                    let src_idx = (ty * ts + tx) * 3;
                    let dst_x = col * ts + tx;
                    let dst_y = row * ts + ty;
                    if dst_x < l2_width && dst_y < l2_height {
                        let dst_idx = (dst_y * l2_width + dst_x) * 3;
                        l2_img[dst_idx..dst_idx + 3].copy_from_slice(&tile_img.as_raw()[src_idx..src_idx + 3]);
                    }
                }
            }
        }
    });

    let mut current_img = l2_image;
    let mut current_width = l2_width;
    let mut current_height = l2_height;
    let mut level = 1; // Start at L3

    let subsamp_tj = match subsamp {
        "444" => Subsamp::None,
        "420" => Subsamp::Sub2x2,
        _ => Subsamp::None,
    };

    loop {
        // Downsample by 2× using box filter (2×2 averaging)
        let new_width = (current_width / 2).max(1);
        let new_height = (current_height / 2).max(1);

        let downsampled = downsample_box_2x_cpu(&current_img, current_width, current_height, new_width, new_height);

        // Apply unsharp mask to recover edge definition
        let sharpened = unsharp_mask_cpu(&downsampled, new_width as u32, new_height as u32, sharpen_strength);

        // Tile the sharpened image
        let cols = ((new_width + ts - 1) / ts);
        let rows = ((new_height + ts - 1) / ts);

        let level_dir = files_dir.join(format!("{}", level));
        fs::create_dir_all(&level_dir)?;

        info!("  Level {}: {}x{} → {}×{} tiles", level, new_width, new_height, cols, rows);

        // Parallel tile extraction and encoding
        let tiles: Vec<(usize, usize)> = (0..rows).flat_map(|r| (0..cols).map(move |c| (r, c))).collect();

        tiles.par_iter().try_for_each(|(row, col)| -> Result<()> {
            let x = *col * ts;
            let y = *row * ts;
            let w = ts.min(new_width.saturating_sub(x));
            let h = ts.min(new_height.saturating_sub(y));

            let mut tile = vec![0u8; ts * ts * 3];
            // Copy region from sharpened image to tile
            for ty in 0..h {
                for tx in 0..w {
                    let src_idx = ((y + ty) * new_width + (x + tx)) * 3;
                    let dst_idx = (ty * ts + tx) * 3;
                    tile[dst_idx..dst_idx + 3].copy_from_slice(&sharpened[src_idx..src_idx + 3]);
                }
            }

            // Encode tile as JPEG using turbojpeg 0.5 API
            let mut compressor = Compressor::new()?;
            compressor.set_quality(quality as i32);
            compressor.set_subsamp(subsamp_tj);
            let img = Image {
                pixels: &tile,
                width: ts,
                pitch: ts * 3,
                height: ts,
                format: PixelFormat::RGB,
            };
            let jpeg_bytes = compressor.compress_to_vec(img)?;
            fs::write(level_dir.join(format!("{}_{}.jpg", col, row)), &jpeg_bytes)?;
            Ok(())
        })?;

        level += 1;
        current_img = sharpened;
        current_width = new_width;
        current_height = new_height;

        // Stop when we reach 1 tile
        if cols == 1 && rows == 1 {
            break;
        }
    }

    info!("Generated {} pyramid levels (L3..L{}) using box+sharpen", level - 1, level);
    Ok(())
}

/// Box filter 2×2 downsample (parallel via rayon).
/// Each output pixel = average of 2×2 block in source.
fn downsample_box_2x_cpu(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    use rayon::prelude::*;

    let dst_len = dst_w * dst_h * 3;
    let mut dst = vec![0u8; dst_len];

    dst.par_chunks_mut(dst_w * 3).enumerate().for_each(|(y, row)| {
        let sy = y * 2;
        let sy1 = (sy + 1).min(src_h - 1);

        for x in 0..dst_w {
            let sx = x * 2;
            let sx1 = (sx + 1).min(src_w - 1);

            for c in 0..3 {
                let v00 = src[(sy * src_w + sx) * 3 + c] as u32;
                let v10 = src[(sy * src_w + sx1) * 3 + c] as u32;
                let v01 = src[(sy1 * src_w + sx) * 3 + c] as u32;
                let v11 = src[(sy1 * src_w + sx1) * 3 + c] as u32;

                // Box filter: average of 4 pixels with rounding
                let avg = (v00 + v10 + v01 + v11 + 2) / 4;
                row[x * 3 + c] = avg as u8;
            }
        }
    });

    dst
}

/// CPU unsharp mask using separable 3×3 Gaussian blur (matches server/src/core/sharpen.rs scalar version).
fn unsharp_mask_cpu(src: &[u8], w: u32, h: u32, strength: f32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let stride = w * 3;
    let len = h * stride;
    debug_assert_eq!(src.len(), len);

    // Pass 1: horizontal blur [0.25, 0.5, 0.25] → store as u16 (×4 to stay integer)
    let mut hblur = vec![0u16; len];
    for y in 0..h {
        let row = y * stride;
        for x in 0..w {
            let x0 = if x > 0 { x - 1 } else { 0 };
            let x2 = if x + 1 < w { x + 1 } else { w - 1 };
            for c in 0..3 {
                let a = src[row + x0 * 3 + c] as u16;
                let b = src[row + x * 3 + c] as u16;
                let d = src[row + x2 * 3 + c] as u16;
                hblur[row + x * 3 + c] = a + 2 * b + d;
            }
        }
    }

    // Pass 2: vertical blur on hblur, fused with sharpen output
    let strength_i = (strength * 256.0).round() as i32; // fixed-point 8.8
    let mut out = vec![0u8; len];
    for y in 0..h {
        let y0 = if y > 0 { y - 1 } else { 0 };
        let y2 = if y + 1 < h { y + 1 } else { h - 1 };
        let row0 = y0 * stride;
        let row1 = y * stride;
        let row2 = y2 * stride;
        for i in 0..stride {
            let blur16 = hblur[row0 + i] as i32
                + 2 * hblur[row1 + i] as i32
                + hblur[row2 + i] as i32;
            let s16 = (src[row1 + i] as i32) << 4;
            let diff = s16 - blur16;
            let v = s16 * 256 + strength_i * diff;
            let v = (v + 2048) >> 12;
            out[row1 + i] = v.clamp(0, 255) as u8;
        }
    }

    out
}
