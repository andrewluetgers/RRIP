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
        "l2resq": config.l2resq, "optl2": config.optl2,
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
            "l2resq": config.l2resq, "optl2": config.optl2,
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

    info!("Opening DICOM: {}", dicom_path.display());
    let slide = DicomSlide::open(dicom_path)
        .with_context(|| format!("Failed to open DICOM {}", dicom_path.display()))?;

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

    info!("Initializing CUDA device {}", config.device);
    let gpu = GpuContext::new(config.device)?;
    let jpeg = NvJpegHandle::new(&gpu)?;

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

    for batch_start in (0..families.len()).step_by(config.batch_size) {
        let batch_end = (batch_start + config.batch_size).min(families.len());
        let batch = &families[batch_start..batch_end];

        info!(
            "Processing batch {}-{} of {} ({} families)",
            batch_start, batch_end, families.len(), batch.len(),
        );

        for &(col, row) in batch {
            // Gather JPEG tile bytes
            let mut tile_jpeg_bytes: Vec<Vec<u8>> = Vec::with_capacity(16);
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let tx = col * 4 + dx;
                    let ty = row * 4 + dy;
                    tile_jpeg_bytes.push(slide.get_tile_bytes(tx, ty).unwrap_or_default());
                }
            }

            // Decode tiles via nvJPEG → GPU device memory
            let decoded_tiles = jpeg.batch_decode_to_device(&gpu, &tile_jpeg_bytes)?;

            // Composite on CPU: single sync, batch download, row-segment memcpy
            let cw = canvas_w as usize;
            let ch = canvas_h as usize;
            let tw = tile_w as usize;
            let th = tile_h as usize;
            let mut canvas_host = vec![0u8; cw * ch * 3];

            // Single sync then batch-download all non-empty tiles
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

            // Upload canvas and convert to f32 on GPU
            let canvas_u8_dev = upload_u8(&gpu, &canvas_host)?;
            let canvas_f32_dev = gpu.u8_to_f32(&canvas_u8_dev, (canvas_w * canvas_h * 3) as i32)?;

            // GPU: downsample to L1 and L2
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

            // OptL2
            if config.optl2 {
                let l2_opt = gpu.u8_to_f32(&l2_u8, (l2_w * l2_h * 3) as i32)?;
                let l1_target = gpu.u8_to_f32(&l1_u8, (l1_w * l1_h * 3) as i32)?;
                let mut l2_cur = gpu.u8_to_f32(&l2_u8, (l2_w * l2_h * 3) as i32)?;
                for _ in 0..100 {
                    gpu.optl2_step(&mut l2_cur, &l2_opt, &l1_target,
                        1, l2_h as i32, l2_w as i32, 0.3, config.max_delta as f32)?;
                }
                l2_u8 = gpu.f32_to_u8(&l2_cur, (l2_w * l2_h * 3) as i32)?;
            }

            // Encode L2 baseline via nvJPEG
            let l2_jpeg = jpeg.encode_rgb(&gpu, &l2_u8, l2_w, l2_h, config.baseq, &config.subsamp)?;
            total_l2_bytes += l2_jpeg.len();

            let l2_level_dir = files_dir.join("0");
            fs::create_dir_all(&l2_level_dir)?;
            fs::write(l2_level_dir.join(format!("{}_{}.jpg", col, row)), &l2_jpeg)?;

            // Decode L2 back on GPU
            let (l2_decoded, _, _) = jpeg.decode_to_device(&gpu, &l2_jpeg)?;

            // Upsample L2 → L1 prediction
            let l2_pred_f32 = gpu.u8_to_f32(&l2_decoded, (l2_w * l2_h * 3) as i32)?;
            let l1_pred_f32 = gpu.upsample_bilinear_2x(&l2_pred_f32, 1, l2_h as i32, l2_w as i32, 3)?;
            let l1_pred_u8 = gpu.f32_to_u8(&l1_pred_f32, (l1_w * l1_h * 3) as i32)?;
            let l1_pred_pixels = (l1_w * l1_h) as i32;
            let (l1_pred_y, l1_pred_cb, l1_pred_cr) = gpu.rgb_to_ycbcr_f32(&l1_pred_u8, l1_pred_pixels)?;

            // Download only what's needed on host: Y prediction for tile extraction,
            // L1 ground truth RGB for Y extraction. Cb/Cr stay on GPU for ycbcr_to_rgb.
            gpu.sync()?;
            let l1_pred_y_host = download_f32(&gpu, &l1_pred_y)?;
            let l1_rgb_host = download_u8(&gpu, &l1_u8)?;

            // Process L1 residuals (2x2 tiles per family)
            // Accumulate reconstructed Y into a mosaic, then GPU YCbCr→RGB
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

                    // Decode and reconstruct Y on GPU
                    let (decoded_res, _, _) = jpeg.decode_gray_to_device(&gpu, &jpeg_data)?;
                    let recon_y_dev = gpu.reconstruct_y(&pred_y_dev, &decoded_res, tile_n as i32)?;
                    gpu.sync()?;
                    let recon_y_tile = download_f32(&gpu, &recon_y_dev)?;

                    // Write reconstructed Y into the L1 mosaic
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
                }
            }

            // GPU: YCbCr→RGB for L1 reconstruction (recon Y + predicted Cb/Cr)
            let l1_recon_y_dev = upload_f32(&gpu, &l1_recon_y_host)?;
            let l1_recon_rgb_dev = gpu.ycbcr_to_rgb(
                &l1_recon_y_dev, &l1_pred_cb, &l1_pred_cr, l1_pixels as i32,
            )?;

            // GPU: upsample L1 recon → L0 prediction (stays on GPU, no host roundtrip)
            let l1_recon_f32 = gpu.u8_to_f32(&l1_recon_rgb_dev, (l1_w * l1_h * 3) as i32)?;
            let l0_pred_f32 = gpu.upsample_bilinear_2x(&l1_recon_f32, 1, l1_h as i32, l1_w as i32, 3)?;
            let l0_pred_u8 = gpu.f32_to_u8(&l0_pred_f32, (canvas_w * canvas_h * 3) as i32)?;
            let l0_pred_pixels = (canvas_w * canvas_h) as i32;
            let (l0_pred_y, _, _) = gpu.rgb_to_ycbcr_f32(&l0_pred_u8, l0_pred_pixels)?;
            gpu.sync()?;
            let l0_pred_y_host = download_f32(&gpu, &l0_pred_y)?;

            // Process L0 residuals (4x4 tiles)
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let abs_tx = col * 4 + dx;
                    let abs_ty = row * 4 + dy;
                    if abs_tx >= slide.tiles_x() || abs_ty >= slide.tiles_y() { continue; }

                    let tile_idx = (dy * 4 + dx) as usize;
                    let (tile_dev, t_w, t_h) = &decoded_tiles[tile_idx];
                    if *t_w == 0 || *t_h == 0 { continue; }

                    let tw = *t_w;
                    let th = *t_h;
                    let tile_n = (tw * th) as usize;

                    // Get L0 ground truth Y from decoded tile on GPU
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
                }
            }

            if config.pack && !pack_entries.is_empty() {
                let lz4_data = origami::core::pack::compress_pack_entries(&pack_entries);
                bundle_entries.push((col, row, lz4_data));
            }

            families_encoded += 1;
        }
    }

    // Write bundle
    if config.pack && !bundle_entries.is_empty() {
        let bundle_dir = output_dir.join("residual_packs");
        fs::create_dir_all(&bundle_dir)?;
        let bundle_path = bundle_dir.join("residuals.bundle");
        origami::core::pack::write_bundle(
            &bundle_path, grid_cols as u16, grid_rows as u16, &bundle_entries,
        )?;
        info!("Wrote bundle: {}", bundle_path.display());
    }

    let elapsed = start.elapsed().as_secs_f64();

    let summary_json = serde_json::json!({
        "mode": "gpu-encode",
        "dicom": dicom_path.to_string_lossy(),
        "tile_size": config.tile_size,
        "baseq": config.baseq, "l1q": config.l1q, "l0q": config.l0q,
        "optl2": config.optl2,
        "families": families_encoded,
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
