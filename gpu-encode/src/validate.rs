//! GPU vs CPU quality validation.
//!
//! Loads a real test image, runs the same processing steps on both CPU and GPU,
//! and compares results numerically.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use image::{imageops, RgbImage};
use tracing::info;

use crate::kernels::GpuContext;

/// Run full GPU vs CPU validation on a test image.
pub fn validate_gpu(gpu: &GpuContext, image_path: &Path) -> Result<()> {
    info!("=== GPU vs CPU Quality Validation ===");
    info!("Image: {}", image_path.display());

    // Load source image
    let src_img = image::open(image_path)
        .with_context(|| format!("Failed to open {}", image_path.display()))?
        .to_rgb8();
    let (src_w, src_h) = (src_img.width(), src_img.height());
    let src_rgb = src_img.into_raw();
    info!("Source: {}x{} ({} bytes)", src_w, src_h, src_rgb.len());

    // --- Step 1: Lanczos3 downsample L0→L1 (CPU reference) ---
    info!("\n--- Step 1: Lanczos3 downsample ---");
    let l1_w = (src_w + 1) / 2;
    let l1_h = (src_h + 1) / 2;
    let cpu_l1 = {
        let img = RgbImage::from_raw(src_w, src_h, src_rgb.clone()).unwrap();
        imageops::resize(&img, l1_w, l1_h, imageops::FilterType::Lanczos3).into_raw()
    };

    // GPU Lanczos3 downsample
    let src_f32: Vec<f32> = src_rgb.iter().map(|&b| b as f32).collect();
    let src_dev = gpu.stream.clone_htod(&src_f32)
        .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
    let t = Instant::now();
    let l1_dev = gpu.downsample_lanczos3(
        &src_dev, 1, src_h as i32, src_w as i32, l1_h as i32, l1_w as i32, 3,
    )?;
    gpu.sync()?;
    let lanczos_ms = t.elapsed().as_secs_f64() * 1000.0;

    let gpu_l1_f32: Vec<f32> = gpu.stream.clone_dtoh(&l1_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;
    let gpu_l1: Vec<u8> = gpu_l1_f32.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();

    // Compare CPU vs GPU Lanczos
    let (mse, max_err, psnr) = compute_metrics(&cpu_l1, &gpu_l1);
    info!("  Lanczos3 {}x{}→{}x{}: GPU {:.2}ms", src_w, src_h, l1_w, l1_h, lanczos_ms);
    info!("  CPU vs GPU: MSE={:.2}, max_err={}, PSNR={:.1}dB", mse, max_err, psnr);

    // --- Step 2: L2 downsample ---
    let l2_w = (l1_w + 1) / 2;
    let l2_h = (l1_h + 1) / 2;
    let cpu_l2 = {
        let img = RgbImage::from_raw(src_w, src_h, src_rgb.clone()).unwrap();
        imageops::resize(&img, l2_w, l2_h, imageops::FilterType::Lanczos3).into_raw()
    };

    let l2_dev = gpu.downsample_lanczos3(
        &src_dev, 1, src_h as i32, src_w as i32, l2_h as i32, l2_w as i32, 3,
    )?;
    gpu.sync()?;
    let gpu_l2_f32: Vec<f32> = gpu.stream.clone_dtoh(&l2_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;
    let gpu_l2: Vec<u8> = gpu_l2_f32.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();

    let (mse2, max_err2, psnr2) = compute_metrics(&cpu_l2, &gpu_l2);
    info!("\n--- Step 2: L2 downsample ---");
    info!("  Lanczos3 {}x{}→{}x{}", src_w, src_h, l2_w, l2_h);
    info!("  CPU vs GPU: MSE={:.2}, max_err={}, PSNR={:.1}dB", mse2, max_err2, psnr2);

    // --- Step 3: Bilinear upsample L2→L1 prediction ---
    info!("\n--- Step 3: Bilinear upsample L2→L1 ---");
    let cpu_l1_pred = {
        let img = RgbImage::from_raw(l2_w, l2_h, cpu_l2.clone()).unwrap();
        imageops::resize(&img, l1_w, l1_h, imageops::FilterType::Triangle).into_raw()
    };

    // GPU: use the GPU-downsampled L2 as input (same data path)
    let t = Instant::now();
    let l1_pred_dev = gpu.upsample_bilinear_2x(&l2_dev, 1, l2_h as i32, l2_w as i32, 3)?;
    gpu.sync()?;
    let upsample_ms = t.elapsed().as_secs_f64() * 1000.0;

    let gpu_l1_pred_f32: Vec<f32> = gpu.stream.clone_dtoh(&l1_pred_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;
    // Note: GPU upsample outputs exactly 2*W x 2*H which may differ from L1 dims
    // if source dims are odd. For 1024x1024 → L2=256x256, upsample gives 512x512
    // but L1 is 512x512, so they match.
    let gpu_l1_pred: Vec<u8> = gpu_l1_pred_f32.iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();

    if gpu_l1_pred.len() == cpu_l1_pred.len() {
        let (mse3, max_err3, psnr3) = compute_metrics(&cpu_l1_pred, &gpu_l1_pred);
        info!("  Bilinear {}x{}→{}x{}: GPU {:.2}ms", l2_w, l2_h, l2_w*2, l2_h*2, upsample_ms);
        info!("  CPU vs GPU: MSE={:.2}, max_err={}, PSNR={:.1}dB", mse3, max_err3, psnr3);
    } else {
        info!("  Size mismatch: CPU {} vs GPU {} (L1={}x{})",
              cpu_l1_pred.len(), gpu_l1_pred.len(), l1_w, l1_h);
    }

    // --- Step 4: RGB→YCbCr conversion ---
    info!("\n--- Step 4: RGB→YCbCr conversion ---");
    // CPU: compute YCbCr from L1 ground truth
    let n_pixels = (l1_w * l1_h) as usize;
    let mut cpu_y = vec![0f32; n_pixels];
    let mut cpu_cb = vec![0f32; n_pixels];
    let mut cpu_cr = vec![0f32; n_pixels];
    for i in 0..n_pixels {
        let r = cpu_l1[i * 3] as f32;
        let g = cpu_l1[i * 3 + 1] as f32;
        let b = cpu_l1[i * 3 + 2] as f32;
        cpu_y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        cpu_cb[i] = -0.168736 * r - 0.331264 * g + 0.500 * b + 128.0;
        cpu_cr[i] = 0.500 * r - 0.418688 * g - 0.081312 * b + 128.0;
    }

    // GPU: RGB→YCbCr
    let l1_rgb_dev = gpu.stream.clone_htod(&cpu_l1.iter().map(|&b| b).collect::<Vec<u8>>())
        .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
    let t = Instant::now();
    let (y_dev, cb_dev, cr_dev) = gpu.rgb_to_ycbcr_f32(&l1_rgb_dev, n_pixels as i32)?;
    gpu.sync()?;
    let ycbcr_ms = t.elapsed().as_secs_f64() * 1000.0;

    let gpu_y: Vec<f32> = gpu.stream.clone_dtoh(&y_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;
    let gpu_cb: Vec<f32> = gpu.stream.clone_dtoh(&cb_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;
    let gpu_cr: Vec<f32> = gpu.stream.clone_dtoh(&cr_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

    let y_max_err = cpu_y.iter().zip(gpu_y.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cb_max_err = cpu_cb.iter().zip(gpu_cb.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cr_max_err = cpu_cr.iter().zip(gpu_cr.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    info!("  RGB→YCbCr ({}x{}, {} pixels): GPU {:.2}ms", l1_w, l1_h, n_pixels, ycbcr_ms);
    info!("  Y  max error: {:.6}", y_max_err);
    info!("  Cb max error: {:.6}", cb_max_err);
    info!("  Cr max error: {:.6}", cr_max_err);

    // --- Step 5: Residual compute + reconstruct ---
    info!("\n--- Step 5: Residual compute + reconstruct ---");
    // Use CPU L1 Y as ground truth, GPU L1 prediction Y as prediction
    // First compute prediction Y on CPU from the L1 prediction
    let pred_y_f32: Vec<f32> = (0..n_pixels).map(|i| {
        let r = cpu_l1_pred[i * 3] as f32;
        let g = cpu_l1_pred[i * 3 + 1] as f32;
        let b = cpu_l1_pred[i * 3 + 2] as f32;
        0.299 * r + 0.587 * g + 0.114 * b
    }).collect();

    // CPU residual
    let cpu_residual: Vec<u8> = (0..n_pixels).map(|i| {
        let diff = cpu_y[i] - pred_y_f32[i] + 128.0;
        diff.round().clamp(0.0, 255.0) as u8
    }).collect();

    // GPU residual
    let gt_y_u8: Vec<u8> = cpu_y.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
    let gt_y_dev = gpu.stream.clone_htod(&gt_y_u8)
        .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
    let pred_y_dev = gpu.stream.clone_htod(&pred_y_f32)
        .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;

    let t = Instant::now();
    let res_dev = gpu.compute_residual(&gt_y_dev, &pred_y_dev, n_pixels as i32)?;
    gpu.sync()?;
    let res_ms = t.elapsed().as_secs_f64() * 1000.0;

    let gpu_residual: Vec<u8> = gpu.stream.clone_dtoh(&res_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

    let (res_mse, res_max, res_psnr) = compute_metrics(&cpu_residual, &gpu_residual);
    info!("  Residual compute ({} pixels): GPU {:.2}ms", n_pixels, res_ms);
    info!("  CPU vs GPU residual: MSE={:.2}, max_err={}, PSNR={:.1}dB", res_mse, res_max, res_psnr);

    // Reconstruct and verify
    let recon_dev = gpu.reconstruct_y(&pred_y_dev, &res_dev, n_pixels as i32)?;
    gpu.sync()?;
    let gpu_recon_y: Vec<f32> = gpu.stream.clone_dtoh(&recon_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

    // Compare reconstructed Y against ground truth
    let recon_max_err = gt_y_u8.iter().zip(gpu_recon_y.iter())
        .map(|(&gt, &rc)| (gt as f32 - rc).abs())
        .fold(0.0f32, f32::max);
    let recon_mse: f32 = gt_y_u8.iter().zip(gpu_recon_y.iter())
        .map(|(&gt, &rc)| { let d = gt as f32 - rc; d * d })
        .sum::<f32>() / n_pixels as f32;
    info!("  Reconstructed Y vs ground truth: MSE={:.4}, max_err={:.1}", recon_mse, recon_max_err);

    // --- Step 6: OptL2 gradient descent ---
    info!("\n--- Step 6: OptL2 gradient descent ---");
    // Start with L2 values, optimize toward L1 target
    let l2_f32: Vec<f32> = cpu_l2.iter().map(|&b| b as f32).collect();
    let l1_f32: Vec<f32> = cpu_l1.iter().map(|&b| b as f32).collect();

    let mut l2_opt_dev = gpu.stream.clone_htod(&l2_f32)
        .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
    let l2_orig_dev = gpu.stream.clone_htod(&l2_f32)
        .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
    let l1_target_dev = gpu.stream.clone_htod(&l1_f32)
        .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;

    let t = Instant::now();
    for _ in 0..100 {
        gpu.optl2_step(
            &mut l2_opt_dev, &l2_orig_dev, &l1_target_dev,
            1, l2_h as i32, l2_w as i32, 0.3, 20.0,
        )?;
    }
    gpu.sync()?;
    let optl2_ms = t.elapsed().as_secs_f64() * 1000.0;

    let l2_optimized: Vec<f32> = gpu.stream.clone_dtoh(&l2_opt_dev)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

    // Check that values moved and are within delta bounds
    let mut moved_count = 0;
    let mut max_deviation = 0.0f32;
    let l2_pixels = (l2_w * l2_h * 3) as usize;
    for i in 0..l2_pixels {
        let delta = (l2_optimized[i] - l2_f32[i]).abs();
        if delta > 0.01 { moved_count += 1; }
        max_deviation = max_deviation.max(delta);
    }
    info!("  OptL2 100 iterations on {}x{}: GPU {:.2}ms", l2_w, l2_h, optl2_ms);
    info!("  Pixels moved: {}/{} ({:.1}%)", moved_count, l2_pixels,
          100.0 * moved_count as f64 / l2_pixels as f64);
    info!("  Max deviation from original: {:.2} (limit: 20)", max_deviation);
    assert!(max_deviation <= 20.5, "OptL2 exceeded max_delta: {}", max_deviation);

    // Measure improvement: upsample optimized L2 and compare to L1
    let l2_opt_dev_for_upsample = gpu.stream.clone_htod(&l2_optimized)
        .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
    let pred_before = gpu.upsample_bilinear_2x(&l2_orig_dev, 1, l2_h as i32, l2_w as i32, 3)?;
    let pred_after = gpu.upsample_bilinear_2x(&l2_opt_dev_for_upsample, 1, l2_h as i32, l2_w as i32, 3)?;
    gpu.sync()?;

    let pred_before_f32: Vec<f32> = gpu.stream.clone_dtoh(&pred_before)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;
    let pred_after_f32: Vec<f32> = gpu.stream.clone_dtoh(&pred_after)
        .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

    let mse_before = compute_f32_mse(&l1_f32, &pred_before_f32);
    let mse_after = compute_f32_mse(&l1_f32, &pred_after_f32);
    let psnr_before = if mse_before > 0.0 { 10.0 * (255.0f64 * 255.0 / mse_before).log10() } else { 999.0 };
    let psnr_after = if mse_after > 0.0 { 10.0 * (255.0f64 * 255.0 / mse_after).log10() } else { 999.0 };
    info!("  L1 prediction PSNR: before={:.1}dB, after={:.1}dB (improvement: {:.1}dB)",
          psnr_before, psnr_after, psnr_after - psnr_before);

    info!("\n=== Validation complete ===");
    Ok(())
}

fn compute_metrics(a: &[u8], b: &[u8]) -> (f64, u32, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    let mut sum_sq = 0.0f64;
    let mut max_err = 0u32;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let d = (av as i32 - bv as i32).unsigned_abs();
        sum_sq += (d as f64) * (d as f64);
        max_err = max_err.max(d);
    }
    let mse = sum_sq / a.len() as f64;
    let psnr = if mse > 0.0 { 10.0 * (255.0 * 255.0 / mse).log10() } else { 999.0 };
    (mse, max_err, psnr)
}

fn compute_f32_mse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let sum: f64 = a[..n].iter().zip(b[..n].iter())
        .map(|(&av, &bv)| { let d = (av - bv) as f64; d * d })
        .sum();
    sum / n as f64
}
