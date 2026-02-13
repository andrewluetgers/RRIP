/// Gradient-descent optimizer for L2 RGB tile.
///
/// Optimizes the L2 tile (low-res) so that bilinear 2x upsampling of the
/// optimized L2 best approximates the L1 mosaic. This produces better
/// predictions and therefore smaller residuals.
///
/// The decoder is unchanged — it just sees an L2 tile and residuals.
use crate::core::upsample::upsample_2x_channel;

/// Optimize L2 RGB for better bilinear predictions of L1.
///
/// * `l2_rgb` — L2 tile as interleaved RGB, l2_w × l2_h × 3
/// * `l1_rgb` — L1 mosaic as interleaved RGB, l1_w × l1_h × 3
/// * `l2_w`, `l2_h` — L2 dimensions
/// * `l1_w`, `l1_h` — L1 dimensions (should be ~2x L2)
/// * `max_delta` — maximum per-pixel deviation from original L2
/// * `n_iterations` — gradient descent iterations
/// * `lr` — learning rate
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
            let upsampled = upsample_2x_channel(&src_u8, l2_w as usize, l2_h as usize);

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

            // Gradient: downsample residual using bilinear (adjoint of bilinear upsample)
            // This must match the forward operator for correct gradient descent
            let gradient = downsample_2x_bilinear(&residual, l1_w as usize, l1_h as usize,
                                                   l2_w as usize, l2_h as usize);

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

/// Bilinear downsample of a f32 channel — adjoint of center-aligned bilinear upsample.
///
/// This is the transpose of `upsample_2x_channel`: for each destination pixel in the
/// upsampled domain, the upsample scatters the source pixel's value to 4 neighbors
/// with bilinear weights. The adjoint gathers those contributions back: each source
/// pixel accumulates the weighted sum of all destination pixels it contributed to.
///
/// Uses the same center-aligned coordinate mapping as `upsample_2x_channel`:
///   src_coord = dst_coord * 0.5 - 0.25 (clamped to [0, src_max])
fn downsample_2x_bilinear(src: &[f32], src_w: usize, src_h: usize,
                           dst_w: usize, dst_h: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_w * dst_h];
    let src_w_max = (dst_w as i32 - 1) * 256; // max in source (L2) fixed-point coords
    let src_h_max = (dst_h as i32 - 1) * 256;

    // For each pixel in the high-res (L1) domain, compute its bilinear weights
    // into the low-res (L2) domain and scatter-add the residual value
    for dy in 0..src_h {
        // Same coordinate mapping as upsample_2x_channel
        let l2_y_fp = (dy as i32 * 128 - 64).max(0).min(src_h_max);
        let l2_y = (l2_y_fp >> 8) as usize;
        let y_frac = (l2_y_fp & 0xFF) as f32 / 256.0;
        let l2_y_next = (l2_y + 1).min(dst_h - 1);

        for dx in 0..src_w {
            let l2_x_fp = (dx as i32 * 128 - 64).max(0).min(src_w_max);
            let l2_x = (l2_x_fp >> 8) as usize;
            let x_frac = (l2_x_fp & 0xFF) as f32 / 256.0;
            let l2_x_next = (l2_x + 1).min(dst_w - 1);

            let val = src[dy * src_w + dx];

            // Scatter-add with bilinear weights (transpose of the gather in upsample)
            let w00 = (1.0 - x_frac) * (1.0 - y_frac);
            let w10 = x_frac * (1.0 - y_frac);
            let w01 = (1.0 - x_frac) * y_frac;
            let w11 = x_frac * y_frac;

            dst[l2_y * dst_w + l2_x] += val * w00;
            dst[l2_y * dst_w + l2_x_next] += val * w10;
            dst[l2_y_next * dst_w + l2_x] += val * w01;
            dst[l2_y_next * dst_w + l2_x_next] += val * w11;
        }
    }

    dst
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
            &l2_rgb, &l1_rgb, l2_w, l2_h, l1_w, l1_h, 20, 50, 0.5,
        );

        // Should differ from original
        assert_ne!(result, l2_rgb, "OptL2 should modify the L2 tile");
    }

    #[test]
    fn test_downsample_adjoint_symmetry() {
        // The adjoint property: <Ux, y> == <x, U^T y>
        // For upsample U and downsample U^T, verify this dot-product identity
        let w = 4usize;
        let h = 4usize;

        // x in L2 domain
        let x_u8: Vec<u8> = (0..w*h).map(|i| ((i * 17 + 3) % 256) as u8).collect();
        // y in L1 domain
        let y: Vec<f32> = (0..w*2*h*2).map(|i| ((i * 13 + 7) % 256) as f32).collect();

        // Ux = upsample(x)
        let ux = upsample_2x_channel(&x_u8, w, h);

        // U^T y = downsample_bilinear(y)
        let uty = downsample_2x_bilinear(&y, w * 2, h * 2, w, h);

        // <Ux, y>
        let dot1: f32 = ux.iter().zip(y.iter()).map(|(&a, &b)| a as f32 * b).sum();
        // <x, U^T y>
        let dot2: f32 = x_u8.iter().zip(uty.iter()).map(|(&a, &b)| a as f32 * b).sum();

        // Should be approximately equal (within floating point tolerance)
        let rel_err = ((dot1 - dot2) / dot1.max(1.0)).abs();
        assert!(rel_err < 0.01, "Adjoint property violated: <Ux,y>={dot1}, <x,U^Ty>={dot2}, rel_err={rel_err}");
    }
}
