//! ONNX Runtime SR model wrapper for learned 4x super-resolution.
//!
//! Replaces the traditional lanczos3/bicubic upsample with a tiny CNN
//! (WSISRX4: 19K params, ~24 KB ONNX) that produces better predictions.
//! The SR output can optionally be corrected by applying a fused L0 residual.
//!
//! Uses a pool of ORT sessions for concurrent inference without lock contention.
//! Each session is ~96KB (19K params × 4 bytes + overhead) — trivial memory cost.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use anyhow::{anyhow, Result};
use ort::session::Session;
use tracing::info;

pub struct SRModel {
    sessions: Vec<Mutex<Session>>,
    next: AtomicUsize,
}

// Safety: Each Session is behind its own Mutex, and AtomicUsize is Sync
unsafe impl Sync for SRModel {}

impl SRModel {
    /// Load an ONNX SR model from disk with a pool of `pool_size` sessions.
    /// Each session gets `intra_threads` ORT threads for SIMD parallelism within a single inference.
    /// `pool_size` controls how many concurrent inferences can run simultaneously.
    pub fn load(onnx_path: &str, intra_threads: usize, pool_size: usize) -> Result<Self> {
        let pool_size = pool_size.max(1);
        let mut sessions = Vec::with_capacity(pool_size);

        for i in 0..pool_size {
            let session = Session::builder()
                .map_err(|e| anyhow!("Failed to create session builder: {}", e))?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow!("Failed to set optimization level: {}", e))?
                .with_intra_threads(intra_threads)
                .map_err(|e| anyhow!("Failed to set intra threads: {}", e))?
                .commit_from_file(onnx_path)
                .map_err(|e| anyhow!("Failed to load ONNX model from {}: {}", onnx_path, e))?;

            if i == 0 {
                let inputs: Vec<_> = session.inputs().iter().map(|i| i.name().to_string()).collect();
                let outputs: Vec<_> = session.outputs().iter().map(|o| o.name().to_string()).collect();
                info!("SR model loaded from {} (pool={}, intra_threads={})", onnx_path, pool_size, intra_threads);
                info!("  inputs: {:?}", inputs);
                info!("  outputs: {:?}", outputs);
            }

            sessions.push(Mutex::new(session));
        }

        Ok(Self {
            sessions,
            next: AtomicUsize::new(0),
        })
    }

    /// Run 4x SR inference on an RGB image.
    ///
    /// Input: RGB u8 pixels (HWC layout), width, height
    /// Output: (rgb_u8_1024x1024, out_w, out_h)
    ///
    /// The model expects BCHW float32 [0,1] and outputs BCHW float32 [0,1].
    /// Uses round-robin session selection for concurrent access.
    pub fn infer_rgb(&self, rgb: &[u8], width: u32, height: u32) -> Result<(Vec<u8>, u32, u32)> {
        let h = height as usize;
        let w = width as usize;
        let npixels = h * w;

        // Convert HWC u8 → BCHW f32 [0,1]
        let mut input_data = vec![0.0f32; 3 * npixels];
        for i in 0..npixels {
            input_data[i] = rgb[i * 3] as f32 / 255.0;             // R plane
            input_data[npixels + i] = rgb[i * 3 + 1] as f32 / 255.0; // G plane
            input_data[2 * npixels + i] = rgb[i * 3 + 2] as f32 / 255.0; // B plane
        }

        // Create input tensor [1, 3, H, W]
        let input_tensor = ort::value::Tensor::from_array(([1usize, 3, h, w], input_data.into_boxed_slice()))
            .map_err(|e| anyhow!("Failed to create input tensor: {}", e))?;

        // Round-robin session selection
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let mut session = self.sessions[idx].lock()
            .map_err(|e| anyhow!("SR model session {} lock poisoned: {}", idx, e))?;
        let outputs = session.run(ort::inputs![input_tensor])
            .map_err(|e| anyhow!("SR inference failed: {}", e))?;

        let output_value = &outputs[0];
        let (shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract output tensor: {}", e))?;

        let out_h = shape[2] as u32;
        let out_w = shape[3] as u32;
        let out_npixels = (out_h * out_w) as usize;

        // Convert BCHW f32 [0,1] → HWC u8
        let mut rgb_out = vec![0u8; out_npixels * 3];
        for i in 0..out_npixels {
            rgb_out[i * 3] = (data[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgb_out[i * 3 + 1] = (data[out_npixels + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgb_out[i * 3 + 2] = (data[2 * out_npixels + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }

        Ok((rgb_out, out_w, out_h))
    }

    /// Run same-resolution refinement inference on an RGB tile, writing output back to `rgb`.
    ///
    /// Input/Output: RGB u8 pixels (HWC layout), width, height — modified in place.
    /// The model expects BCHW float32 [0,1] and outputs BCHW float32 [0,1].
    /// Output dimensions must match input (same-resolution enhancement model).
    pub fn refine_rgb_inplace(&self, rgb: &mut [u8], width: u32, height: u32) -> Result<()> {
        let h = height as usize;
        let w = width as usize;
        let npixels = h * w;

        // Convert HWC u8 → BCHW f32 [0,1]
        let mut input_data = vec![0.0f32; 3 * npixels];
        for i in 0..npixels {
            input_data[i] = rgb[i * 3] as f32 / 255.0;
            input_data[npixels + i] = rgb[i * 3 + 1] as f32 / 255.0;
            input_data[2 * npixels + i] = rgb[i * 3 + 2] as f32 / 255.0;
        }

        let input_tensor = ort::value::Tensor::from_array(([1usize, 3, h, w], input_data.into_boxed_slice()))
            .map_err(|e| anyhow!("Failed to create input tensor: {}", e))?;

        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let mut session = self.sessions[idx].lock()
            .map_err(|e| anyhow!("Refine model session {} lock poisoned: {}", idx, e))?;
        let outputs = session.run(ort::inputs![input_tensor])
            .map_err(|e| anyhow!("Refine inference failed: {}", e))?;

        let output_value = &outputs[0];
        let (_shape, data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract output tensor: {}", e))?;

        // Convert BCHW f32 [0,1] → HWC u8 in place
        for i in 0..npixels {
            rgb[i * 3] = (data[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgb[i * 3 + 1] = (data[npixels + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            rgb[i * 3 + 2] = (data[2 * npixels + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }

        Ok(())
    }

    /// Return the number of sessions in the pool.
    pub fn pool_size(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn model_path() -> String {
        // Look for model relative to the server crate root (Cargo.toml dir)
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("../models/model_sr.onnx");
        path.to_string_lossy().to_string()
    }

    fn has_model() -> bool {
        Path::new(&model_path()).exists()
    }

    #[test]
    fn test_load_missing_model() {
        let result = SRModel::load("/nonexistent/model.onnx", 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_pool_sizes() {
        if !has_model() {
            eprintln!("skipping: model not found at {}", model_path());
            return;
        }
        let m1 = SRModel::load(&model_path(), 1, 1).unwrap();
        assert_eq!(m1.pool_size(), 1);

        let m4 = SRModel::load(&model_path(), 1, 4).unwrap();
        assert_eq!(m4.pool_size(), 4);

        // pool_size=0 should be clamped to 1
        let m0 = SRModel::load(&model_path(), 1, 0).unwrap();
        assert_eq!(m0.pool_size(), 1);
    }

    #[test]
    fn test_infer_rgb_output_shape() {
        if !has_model() {
            eprintln!("skipping: model not found at {}", model_path());
            return;
        }
        let model = SRModel::load(&model_path(), 1, 1).unwrap();

        // 256x256 input → expect 1024x1024 output (4x)
        let input = vec![128u8; 256 * 256 * 3];
        let (output, w, h) = model.infer_rgb(&input, 256, 256).unwrap();
        assert_eq!(w, 1024);
        assert_eq!(h, 1024);
        assert_eq!(output.len(), 1024 * 1024 * 3);
    }

    #[test]
    fn test_infer_rgb_output_range() {
        if !has_model() {
            eprintln!("skipping: model not found at {}", model_path());
            return;
        }
        let model = SRModel::load(&model_path(), 1, 1).unwrap();

        let input = vec![100u8; 256 * 256 * 3];
        let (output, _, _) = model.infer_rgb(&input, 256, 256).unwrap();

        // Output should be valid u8 RGB (0-255) — verify non-trivial output
        let mean: f64 = output.iter().map(|&v| v as f64).sum::<f64>() / output.len() as f64;
        assert!(mean > 10.0 && mean < 245.0,
            "mean pixel value {} should be in reasonable range for uniform input", mean);
    }

    #[test]
    fn test_infer_deterministic() {
        if !has_model() {
            eprintln!("skipping: model not found at {}", model_path());
            return;
        }
        let model = SRModel::load(&model_path(), 1, 1).unwrap();

        let input = vec![128u8; 256 * 256 * 3];
        let (out1, _, _) = model.infer_rgb(&input, 256, 256).unwrap();
        let (out2, _, _) = model.infer_rgb(&input, 256, 256).unwrap();
        assert_eq!(out1, out2, "inference should be deterministic");
    }

    #[test]
    fn test_pool_round_robin() {
        if !has_model() {
            eprintln!("skipping: model not found at {}", model_path());
            return;
        }
        let model = SRModel::load(&model_path(), 1, 4).unwrap();

        // Run 8 inferences — should cycle through 4 sessions twice
        let input = vec![128u8; 256 * 256 * 3];
        for _ in 0..8 {
            let (_, w, h) = model.infer_rgb(&input, 256, 256).unwrap();
            assert_eq!((w, h), (1024, 1024));
        }
        // Counter should be at 8
        assert_eq!(model.next.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn test_concurrent_inference() {
        if !has_model() {
            eprintln!("skipping: model not found at {}", model_path());
            return;
        }
        let model = std::sync::Arc::new(SRModel::load(&model_path(), 1, 4).unwrap());

        // Run 8 concurrent inferences via rayon
        let results: Vec<_> = (0..8).into_iter().map(|_| {
            let m = model.clone();
            std::thread::spawn(move || {
                let input = vec![128u8; 256 * 256 * 3];
                m.infer_rgb(&input, 256, 256).unwrap()
            })
        }).collect();

        for handle in results {
            let (_, w, h) = handle.join().unwrap();
            assert_eq!((w, h), (1024, 1024));
        }
    }
}
