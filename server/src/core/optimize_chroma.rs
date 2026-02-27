/// Gradient-descent optimizer for chroma planes.
///
/// Given a half-resolution chroma plane (starting point from naive downsample)
/// and the full-resolution target chroma, optimizes the half-res plane so that
/// upsampling of the optimized plane best approximates the target.
///
/// This is the adjoint-based approach: the gradient of ||target - upsample(source)||²
/// with respect to source is downsample(residual), where residual = target - upsample(source).

use super::ResampleFilter;

use super::upsample_2x;

/// Optimize a half-resolution chroma plane for bicubic upsample fidelity.
///
/// * `source_half` — W/2 × H/2 starting point (naive box downsample)
/// * `target_full` — W × H original full-resolution chroma channel
/// * `w_half`, `h_half` — dimensions of the half-resolution plane
/// * `max_delta` — maximum deviation from original value per pixel
/// * `n_iterations` — number of gradient descent iterations
/// * `lr` — learning rate
///
/// Returns the optimized half-resolution plane as u8 values.
pub fn optimize_chroma_for_upsample(
    source_half: &[u8],
    target_full: &[u8],
    w_half: usize,
    h_half: usize,
    max_delta: f32,
    n_iterations: u32,
    lr: f32,
    filter: ResampleFilter,
) -> Vec<u8> {
    let w_full = w_half * 2;
    let h_full = h_half * 2;
    let half_size = w_half * h_half;

    // Work in f32 for gradient descent
    let mut src_f: Vec<f32> = source_half.iter().map(|&v| v as f32).collect();
    let orig_f: Vec<f32> = source_half.iter().map(|&v| v as f32).collect();
    let target_f: Vec<f32> = target_full.iter().map(|&v| v as f32).collect();

    for _ in 0..n_iterations {
        // Forward: upsample current source
        let src_u8: Vec<u8> = src_f.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
        let upsampled = upsample_2x(&src_u8, w_half, h_half, filter);

        // Compute residual: target - upsample(source)
        let mut residual = vec![0.0f32; w_full * h_full];
        for i in 0..residual.len().min(target_f.len()).min(upsampled.len()) {
            residual[i] = target_f[i] - upsampled[i] as f32;
        }

        // Gradient: downsample the residual (matches forward operator)
        let gradient = downsample_2x_f32(&residual, w_full, h_full, w_half, h_half, filter);

        // Update: source += lr * gradient
        for i in 0..half_size {
            src_f[i] += lr * gradient[i];
            // Clamp to [orig - max_delta, orig + max_delta] and [0, 255]
            src_f[i] = src_f[i]
                .max(orig_f[i] - max_delta)
                .min(orig_f[i] + max_delta)
                .max(0.0)
                .min(255.0);
        }
    }

    // Convert back to u8
    src_f.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
}

/// Downsample of a f32 channel — gradient approximation for the upsample operator.
///
/// Normalizes to [0, 255], resizes with the specified filter, then maps back to original range.
fn downsample_2x_f32(src: &[f32], src_w: usize, src_h: usize,
                      dst_w: usize, dst_h: usize, filter: ResampleFilter) -> Vec<f32> {
    use image::{GrayImage, imageops};
    let min_val = src.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(1e-8);

    let src_u8: Vec<u8> = src.iter()
        .map(|&v| ((v - min_val) / range * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    let img = GrayImage::from_raw(src_w as u32, src_h as u32, src_u8)
        .expect("failed to create GrayImage for downsample");
    let resized = imageops::resize(&img, dst_w as u32, dst_h as u32, filter.to_image_filter());
    let dst_u8 = resized.into_raw();

    dst_u8.iter()
        .map(|&v| v as f32 / 255.0 * range + min_val)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_chroma_improves_psnr() {
        // Create a test with a gradient that naive downsample blurs
        let w_full = 8;
        let h_full = 8;
        let w_half = 4;
        let h_half = 4;

        // Target: gradient left-to-right (0..255 across the width)
        let target: Vec<u8> = (0..w_full * h_full)
            .map(|i| {
                let x = i % w_full;
                (x * 255 / (w_full - 1)) as u8
            })
            .collect();

        // Naive downsample: box average of 2x2 blocks
        let naive: Vec<u8> = (0..w_half * h_half)
            .map(|i| {
                let dx = i % w_half;
                let dy = i / w_half;
                let sx = dx * 2;
                let sy = dy * 2;
                let p00 = target[sy * w_full + sx] as u32;
                let p10 = target[sy * w_full + sx + 1] as u32;
                let p01 = target[(sy + 1) * w_full + sx] as u32;
                let p11 = target[(sy + 1) * w_full + sx + 1] as u32;
                ((p00 + p10 + p01 + p11 + 2) / 4) as u8
            })
            .collect();

        let optimized = optimize_chroma_for_upsample(
            &naive, &target, w_half, h_half, 50.0, 200, 0.5, ResampleFilter::Bicubic,
        );

        // Compute MSE for naive vs target and optimized vs target
        let naive_upsampled = upsample_2x(&naive, w_half, h_half, ResampleFilter::Bicubic);
        let opt_upsampled = upsample_2x(&optimized, w_half, h_half, ResampleFilter::Bicubic);

        let naive_mse: f64 = naive_upsampled.iter().zip(target.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>() / target.len() as f64;
        let opt_mse: f64 = opt_upsampled.iter().zip(target.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>() / target.len() as f64;

        assert!(opt_mse <= naive_mse, "Optimized MSE ({:.2}) should be <= naive MSE ({:.2})", opt_mse, naive_mse);
    }
}
