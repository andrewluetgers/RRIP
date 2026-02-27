/// Gradient-descent optimizer for L2 RGB tile.
///
/// Optimizes the L2 tile (low-res) so that 2x upsampling of the
/// optimized L2 best approximates the L1 mosaic. This produces better
/// predictions and therefore smaller residuals.
///
/// The decoder is unchanged — it just sees an L2 tile and residuals.

use super::ResampleFilter;

use super::upsample_2x;

/// Downsample of a f32 channel — gradient approximation for the upsample operator.
///
/// For the gradient of ||target - upsample(source)||², we need the adjoint of the
/// upsample operator applied to the residual. We approximate this by doing a
/// resize from the high-res residual back to the low-res source size.
/// This uses the same kernel family as the forward pass, giving well-directed gradients.
fn downsample_2x(src: &[f32], src_w: usize, src_h: usize,
                  dst_w: usize, dst_h: usize, filter: ResampleFilter) -> Vec<f32> {
    use image::{GrayImage, imageops};
    // Convert f32 residual to u8 for image crate (shift by 128 to handle negatives)
    // We need to preserve sign information, so we use a float-aware approach:
    // scale to [0, 255] range, downsample, then scale back.
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

    // Convert back to f32 in original value range
    dst_u8.iter()
        .map(|&v| v as f32 / 255.0 * range + min_val)
        .collect()
}

/// Optimize L2 RGB for better predictions of L1.
///
/// * `l2_rgb` — L2 tile as interleaved RGB, l2_w × l2_h × 3
/// * `l1_rgb` — L1 mosaic as interleaved RGB, l1_w × l1_h × 3
/// * `l2_w`, `l2_h` — L2 dimensions
/// * `l1_w`, `l1_h` — L1 dimensions (should be ~2x L2)
/// * `max_delta` — maximum per-pixel deviation from original L2
/// * `n_iterations` — gradient descent iterations
/// * `lr` — learning rate
/// * `filter` — resample filter (must match prediction pipeline)
///
/// Returns optimized L2 RGB.
pub fn optimize_l2_for_prediction(
    l2_rgb: &[u8],
    l1_rgb: &[u8],
    l2_w: u32, l2_h: u32,
    l1_w: u32, l1_h: u32,
    max_delta: u8,
    n_iterations: u32,
    lr: f32,
    filter: ResampleFilter,
) -> Vec<u8> {
    let l2_size = (l2_w * l2_h) as usize;
    let l1_size = (l1_w * l1_h) as usize;

    // Separate into R, G, B channels for per-channel optimization
    let mut l2_channels: [Vec<f32>; 3] = [
        vec![0.0; l2_size],
        vec![0.0; l2_size],
        vec![0.0; l2_size],
    ];
    let orig_channels: [Vec<f32>; 3] = {
        let mut ch = [vec![0.0f32; l2_size], vec![0.0; l2_size], vec![0.0; l2_size]];
        for i in 0..l2_size {
            ch[0][i] = l2_rgb[i * 3] as f32;
            ch[1][i] = l2_rgb[i * 3 + 1] as f32;
            ch[2][i] = l2_rgb[i * 3 + 2] as f32;
        }
        ch
    };
    for c in 0..3 {
        l2_channels[c] = orig_channels[c].clone();
    }

    let mut l1_channels: [Vec<f32>; 3] = [
        vec![0.0; l1_size],
        vec![0.0; l1_size],
        vec![0.0; l1_size],
    ];
    for i in 0..l1_size {
        l1_channels[0][i] = l1_rgb[i * 3] as f32;
        l1_channels[1][i] = l1_rgb[i * 3 + 1] as f32;
        l1_channels[2][i] = l1_rgb[i * 3 + 2] as f32;
    }

    let max_d = max_delta as f32;

    // Track best result across iterations
    let mut best_energy = f32::INFINITY;
    let mut best_channels: [Vec<f32>; 3] = [
        l2_channels[0].clone(),
        l2_channels[1].clone(),
        l2_channels[2].clone(),
    ];

    for _ in 0..n_iterations {
        let mut total_energy: f32 = 0.0;

        for c in 0..3 {
            // Forward: upsample current L2 channel
            let src_u8: Vec<u8> = l2_channels[c].iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect();
            let upsampled = upsample_2x(&src_u8, l2_w as usize, l2_h as usize, filter);

            // Compute residual: L1_target - upsample(L2)
            let up_len = upsampled.len().min(l1_channels[c].len());
            let mut residual = vec![0.0f32; l1_w as usize * l1_h as usize];
            for i in 0..up_len.min(residual.len()) {
                residual[i] = l1_channels[c][i] - upsampled[i] as f32;
            }

            // Accumulate energy for best-tracking
            for i in 0..up_len.min(residual.len()) {
                total_energy += residual[i] * residual[i];
            }

            // Gradient: downsample residual (matches forward operator)
            let gradient = downsample_2x(&residual, l1_w as usize, l1_h as usize,
                                          l2_w as usize, l2_h as usize, filter);

            // Update
            for i in 0..l2_size {
                l2_channels[c][i] += lr * gradient[i];
                l2_channels[c][i] = l2_channels[c][i]
                    .max(orig_channels[c][i] - max_d)
                    .min(orig_channels[c][i] + max_d)
                    .max(0.0)
                    .min(255.0);
            }
        }

        // Track best
        if total_energy < best_energy {
            best_energy = total_energy;
            for c in 0..3 {
                best_channels[c] = l2_channels[c].clone();
            }
        }
    }

    // Interleave best result back to RGB
    let mut result = vec![0u8; l2_size * 3];
    for i in 0..l2_size {
        result[i * 3] = best_channels[0][i].round().clamp(0.0, 255.0) as u8;
        result[i * 3 + 1] = best_channels[1][i].round().clamp(0.0, 255.0) as u8;
        result[i * 3 + 2] = best_channels[2][i].round().clamp(0.0, 255.0) as u8;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_l2_changes_tile() {
        // Simple test: L2 = 4x4 flat gray, L1 = 8x8 with some detail
        let l2_w = 4u32;
        let l2_h = 4u32;
        let l1_w = 8u32;
        let l1_h = 8u32;

        let l2_rgb = vec![128u8; (l2_w * l2_h * 3) as usize];
        let mut l1_rgb = vec![128u8; (l1_w * l1_h * 3) as usize];
        // Add some detail to L1 that L2 can't predict well
        for i in 0..32 {
            l1_rgb[i * 3] = 200;
        }

        let result = optimize_l2_for_prediction(
            &l2_rgb, &l1_rgb, l2_w, l2_h, l1_w, l1_h, 20, 50, 0.5, ResampleFilter::Bicubic,
        );

        // Should differ from original
        assert_ne!(result, l2_rgb, "OptL2 should modify the L2 tile");
    }

    #[test]
    fn test_downsample_produces_reasonable_output() {
        // Verify that downsample produces values in expected range
        let w = 8usize;
        let h = 8usize;
        let src: Vec<f32> = (0..w * h).map(|i| (i as f32 * 4.0) - 128.0).collect();

        let dst = downsample_2x(&src, w, h, w / 2, h / 2, ResampleFilter::Bicubic);
        assert_eq!(dst.len(), (w / 2) * (h / 2));

        // All values should be within the range of the input
        let src_min = src.iter().cloned().fold(f32::INFINITY, f32::min);
        let src_max = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for &v in &dst {
            assert!(v >= src_min - 1.0 && v <= src_max + 1.0,
                "Downsample value {v} out of range [{src_min}, {src_max}]");
        }
    }
}
