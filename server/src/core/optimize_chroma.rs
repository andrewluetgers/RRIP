/// Gradient-descent optimizer for chroma planes.
///
/// Given a half-resolution chroma plane (starting point from naive downsample)
/// and the full-resolution target chroma, optimizes the half-res plane so that
/// bilinear upsampling of the optimized plane best approximates the target.
///
/// This is the adjoint-based approach: the gradient of ||target - upsample(source)||²
/// with respect to source is downsample(residual), where residual = target - upsample(source).
use crate::core::upsample::upsample_2x_channel;

/// Optimize a half-resolution chroma plane for bilinear upsample fidelity.
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
        let upsampled = upsample_2x_channel(&src_u8, w_half, h_half);

        // Compute residual: target - upsample(source)
        let mut residual = vec![0.0f32; w_full * h_full];
        for i in 0..residual.len().min(target_f.len()).min(upsampled.len()) {
            residual[i] = target_f[i] - upsampled[i] as f32;
        }

        // Gradient: downsample the residual (adjoint of bilinear upsample = area average)
        let gradient = downsample_2x_f32(&residual, w_full, h_full);

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

/// Downsample a f32 channel by 2x using area average (adjoint of bilinear upsample).
fn downsample_2x_f32(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let half_w = (w + 1) / 2;
    let half_h = (h + 1) / 2;
    let mut dst = vec![0.0f32; half_w * half_h];

    for dy in 0..half_h {
        for dx in 0..half_w {
            let sx = dx * 2;
            let sy = dy * 2;
            let p00 = src[sy * w + sx];
            let p10 = if sx + 1 < w { src[sy * w + sx + 1] } else { p00 };
            let p01 = if sy + 1 < h { src[(sy + 1) * w + sx] } else { p00 };
            let p11 = if sx + 1 < w && sy + 1 < h {
                src[(sy + 1) * w + sx + 1]
            } else if sy + 1 < h {
                p01
            } else {
                p10
            };
            dst[dy * half_w + dx] = (p00 + p10 + p01 + p11) / 4.0;
        }
    }
    dst
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
            &naive, &target, w_half, h_half, 50.0, 200, 0.5,
        );

        // Compute MSE for naive vs target and optimized vs target
        let naive_upsampled = upsample_2x_channel(&naive, w_half, h_half);
        let opt_upsampled = upsample_2x_channel(&optimized, w_half, h_half);

        let naive_mse: f64 = naive_upsampled.iter().zip(target.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>() / target.len() as f64;
        let opt_mse: f64 = opt_upsampled.iter().zip(target.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>() / target.len() as f64;

        assert!(opt_mse <= naive_mse, "Optimized MSE ({:.2}) should be <= naive MSE ({:.2})", opt_mse, naive_mse);
    }
}
