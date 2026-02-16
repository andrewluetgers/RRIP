use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

mod dicom;
mod kernels;
mod nvjpeg;
mod pipeline;
mod validate;

#[derive(Parser)]
#[command(name = "origami-gpu-encode", about = "GPU-accelerated WSI encoding for ORIGAMI")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// CUDA device index
    #[arg(long, default_value_t = 0, global = true)]
    device: usize,
}

#[derive(Subcommand)]
enum Command {
    /// Encode a DICOM WSI
    Encode {
        /// Path to DICOM WSI file
        #[arg(long)]
        slide: PathBuf,

        /// Output directory
        #[arg(long)]
        out: PathBuf,

        /// Tile size (must match DICOM tile size, typically 224)
        #[arg(long, default_value_t = 224)]
        tile: u32,

        /// JPEG quality for L2 baseline tiles
        #[arg(long, default_value_t = 95)]
        baseq: u8,

        /// JPEG quality for L1 residuals
        #[arg(long, default_value_t = 60)]
        l1q: u8,

        /// JPEG quality for L0 residuals
        #[arg(long, default_value_t = 40)]
        l0q: u8,

        /// Enable OptL2 gradient descent optimization
        #[arg(long)]
        optl2: bool,

        /// Maximum per-pixel deviation for OptL2 (default: 15)
        #[arg(long, default_value_t = 15)]
        max_delta: u8,

        /// Chroma subsampling for L2: 444, 420
        #[arg(long, default_value = "444")]
        subsamp: String,

        /// Batch size (number of families per GPU batch)
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
    },

    /// Run GPU kernel smoke tests
    TestGpu,

    /// Validate GPU kernels against CPU reference using a real image
    Validate {
        /// Path to test image (e.g., evals/test-images/L0-1024.jpg)
        #[arg(long)]
        image: PathBuf,
    },
}

fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Encode {
            slide, out, tile, baseq, l1q, l0q, optl2, max_delta, subsamp, batch_size,
        } => {
            info!("ORIGAMI GPU Encoder");
            info!("  slide: {}", slide.display());
            info!("  out: {}", out.display());

            let config = pipeline::EncodeConfig {
                tile_size: tile,
                baseq, l1q, l0q, optl2, max_delta, subsamp, batch_size,
                device: cli.device,
            };

            let summary = pipeline::encode_wsi(&slide, &out, config)?;

            info!("Encode complete:");
            info!("  families: {}", summary.families_encoded);
            info!("  L2 bytes: {:.1} MB", summary.l2_bytes as f64 / 1_048_576.0);
            info!("  residual bytes: {:.1} MB", summary.residual_bytes as f64 / 1_048_576.0);
            info!("  elapsed: {:.2}s", summary.elapsed_secs);
            info!(
                "  throughput: {:.1} families/s ({:.2} ms/family)",
                summary.families_encoded as f64 / summary.elapsed_secs,
                summary.elapsed_secs * 1000.0 / summary.families_encoded as f64
            );
        }

        Command::TestGpu => {
            run_gpu_tests(cli.device)?;
        }

        Command::Validate { image } => {
            let gpu = kernels::GpuContext::new(cli.device)?;
            validate::validate_gpu(&gpu, &image)?;
        }
    }

    Ok(())
}

fn run_gpu_tests(device: usize) -> Result<()> {
    info!("=== GPU Kernel Smoke Tests ===");
    let t0 = Instant::now();

    let gpu = kernels::GpuContext::new(device)?;
    info!("CUDA initialized in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    // Test 1: RGB → YCbCr → RGB roundtrip
    {
        info!("--- Test 1: RGB → YCbCr → RGB roundtrip ---");
        // Create a 4x4 RGB image (48 bytes)
        let rgb_host: Vec<u8> = (0..48).collect();
        let rgb_dev = gpu.stream.clone_htod(&rgb_host)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;

        let (y, cb, cr) = gpu.rgb_to_ycbcr_f32(&rgb_dev, 16)?;
        let rgb_out = gpu.ycbcr_to_rgb(&y, &cb, &cr, 16)?;
        gpu.sync()?;

        let rgb_result: Vec<u8> = gpu.stream.clone_dtoh(&rgb_out)
            .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

        // Check roundtrip error (should be ≤2 LSB per channel)
        let max_err = rgb_host.iter().zip(rgb_result.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .max().unwrap_or(0);
        info!("  RGB→YCbCr→RGB max roundtrip error: {} (expect ≤2)", max_err);
        assert!(max_err <= 2, "Roundtrip error too high: {}", max_err);
        info!("  PASS");
    }

    // Test 2: Bilinear upsample 2x
    {
        info!("--- Test 2: Bilinear upsample 2x ---");
        // 2x2 image with 3 channels, all ones → should upsample to 4x4 all ones
        let src: Vec<f32> = vec![1.0; 2 * 2 * 3];
        let src_dev = gpu.stream.clone_htod(&src)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;

        let dst_dev = gpu.upsample_bilinear_2x(&src_dev, 1, 2, 2, 3)?;
        gpu.sync()?;

        let dst: Vec<f32> = gpu.stream.clone_dtoh(&dst_dev)
            .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

        assert_eq!(dst.len(), 4 * 4 * 3, "Wrong output size");
        let max_err = dst.iter().map(|&v| (v - 1.0).abs()).fold(0.0f32, f32::max);
        info!("  Constant upsample max error: {:.6} (expect ~0)", max_err);
        assert!(max_err < 0.01, "Upsample of constant should stay constant");
        info!("  PASS");
    }

    // Test 3: Compute residual + reconstruct roundtrip
    {
        info!("--- Test 3: Residual compute + reconstruct ---");
        let gt_y: Vec<u8> = vec![100, 150, 200, 50];
        let pred_y: Vec<f32> = vec![95.0, 155.0, 198.0, 52.0];

        let gt_dev = gpu.stream.clone_htod(&gt_y)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
        let pred_dev = gpu.stream.clone_htod(&pred_y)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;

        let res_dev = gpu.compute_residual(&gt_dev, &pred_dev, 4)?;
        let recon_dev = gpu.reconstruct_y(&pred_dev, &res_dev, 4)?;
        gpu.sync()?;

        let residual: Vec<u8> = gpu.stream.clone_dtoh(&res_dev)
            .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;
        let recon: Vec<f32> = gpu.stream.clone_dtoh(&recon_dev)
            .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

        info!("  gt_y:     {:?}", gt_y);
        info!("  pred_y:   {:?}", pred_y);
        info!("  residual: {:?} (128 = no diff)", residual);
        info!("  recon:    {:?}", recon);

        // Reconstructed should match ground truth (within rounding)
        for (i, (&gt, &rc)) in gt_y.iter().zip(recon.iter()).enumerate() {
            let err = (gt as f32 - rc).abs();
            assert!(err <= 1.0, "Pixel {}: gt={} recon={} err={}", i, gt, rc, err);
        }
        info!("  PASS");
    }

    // Test 4: OptL2 gradient descent (should reduce error)
    {
        info!("--- Test 4: OptL2 gradient descent ---");
        // Simple 2x2 L2, 4x4 L1 target
        // L2 all 128.0, L1 target all 200.0
        // After optimization, L2 should move toward values that predict 200
        let l2_init: Vec<f32> = vec![128.0; 2 * 2 * 3];
        let l1_target: Vec<f32> = vec![200.0; 4 * 4 * 3];

        let mut l2_dev = gpu.stream.clone_htod(&l2_init)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
        let l2_orig = gpu.stream.clone_htod(&l2_init)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
        let l1_dev = gpu.stream.clone_htod(&l1_target)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;

        // Run 50 iterations
        for _ in 0..50 {
            gpu.optl2_step(&mut l2_dev, &l2_orig, &l1_dev, 1, 2, 2, 0.01, 255.0)?;
        }
        gpu.sync()?;

        let l2_result: Vec<f32> = gpu.stream.clone_dtoh(&l2_dev)
            .map_err(|e| anyhow::anyhow!("dtoh failed: {}", e))?;

        let avg = l2_result.iter().sum::<f32>() / l2_result.len() as f32;
        info!("  L2 initial: 128.0, L1 target: 200.0");
        info!("  After 50 iterations: avg L2 = {:.1} (should be > 128, moving toward 200)", avg);
        assert!(avg > 128.0, "OptL2 should have increased values toward target");
        info!("  PASS");
    }

    // Test 5: Large-scale performance test
    {
        info!("--- Test 5: Performance benchmark ---");
        // Simulate 64 families: 64 × 224 × 224 × 3 RGB
        let n = 64;
        let h = 224;
        let w = 224;
        let c = 3;
        let total = n * h * w * c;
        let src: Vec<f32> = vec![128.0; total];
        let src_dev = gpu.stream.clone_htod(&src)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
        gpu.sync()?;

        // Benchmark upsample 2x
        let t = Instant::now();
        let _dst = gpu.upsample_bilinear_2x(&src_dev, n as i32, h as i32, w as i32, c as i32)?;
        gpu.sync()?;
        let upsample_ms = t.elapsed().as_secs_f64() * 1000.0;
        info!("  Upsample 2x ({} families, {}x{}x{} → {}x{}x{}): {:.2}ms",
              n, h, w, c, h*2, w*2, c, upsample_ms);

        // Benchmark RGB→YCbCr
        let pixels = n * h * w;
        let rgb_data: Vec<u8> = vec![128u8; pixels * 3];
        let rgb_dev = gpu.stream.clone_htod(&rgb_data)
            .map_err(|e| anyhow::anyhow!("htod failed: {}", e))?;
        gpu.sync()?;

        let t = Instant::now();
        let (_y, _cb, _cr) = gpu.rgb_to_ycbcr_f32(&rgb_dev, pixels as i32)?;
        gpu.sync()?;
        let ycbcr_ms = t.elapsed().as_secs_f64() * 1000.0;
        info!("  RGB→YCbCr ({} pixels): {:.2}ms", pixels, ycbcr_ms);

        info!("  PASS");
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    info!("=== All GPU tests passed in {:.1}ms ===", total_ms);
    Ok(())
}
