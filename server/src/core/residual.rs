use crate::core::color::ycbcr_to_rgb;
use crate::turbojpeg_optimized::apply_residual_fast;
use anyhow::{anyhow, Result};

/// Apply a centered residual to a Y prediction plane and produce RGB output.
///
/// Extracts a tile_size x tile_size region from the prediction planes at (x0, y0),
/// applies the residual (which is centered around 128), and writes interleaved RGB
/// to `out`.
pub fn apply_residual_into(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    tile_size: u32,
    residual: &[u8],
    out: &mut [u8],
) -> Result<()> {
    let expected_len = (tile_size * tile_size * 3) as usize;
    if out.len() != expected_len {
        return Err(anyhow!("output buffer size mismatch"));
    }

    let mut y_reconstructed = vec![0u8; (tile_size * tile_size) as usize];

    // First pass: Apply residual using SIMD optimization
    for y in 0..tile_size {
        let py = y0 + y;
        if py >= height {
            continue;
        }

        let row_offset = (y * tile_size) as usize;
        let plane_row_offset = (py * width) as usize;

        let mut y_pred_row = vec![0u8; tile_size as usize];
        for x in 0..tile_size {
            let px = x0 + x;
            if px < width {
                y_pred_row[x as usize] = y_plane[plane_row_offset + px as usize];
            }
        }

        let residual_row_start = row_offset;
        let residual_row_end = residual_row_start + tile_size as usize;
        let residual_row = &residual[residual_row_start..residual_row_end];

        apply_residual_fast(
            &y_pred_row,
            residual_row,
            &mut y_reconstructed[residual_row_start..residual_row_end],
            tile_size as usize,
        );
    }

    // Second pass: Convert YCbCr to RGB
    for y in 0..tile_size {
        let py = y0 + y;
        if py >= height {
            continue;
        }

        let row_offset = (y * tile_size) as usize;
        let plane_row_offset = (py * width) as usize;

        for x in 0..tile_size {
            let px = x0 + x;
            if px < width {
                let idx = plane_row_offset + px as usize;
                let residual_idx = row_offset + x as usize;

                let y_recon = y_reconstructed[residual_idx];
                let cb_pred = cb_plane[idx];
                let cr_pred = cr_plane[idx];

                let (r_out, g_out, b_out) = ycbcr_to_rgb(y_recon, cb_pred, cr_pred);

                let out_idx = residual_idx * 3;
                out[out_idx] = r_out;
                out[out_idx + 1] = g_out;
                out[out_idx + 2] = b_out;
            } else {
                let residual_idx = row_offset + x as usize;
                let out_idx = residual_idx * 3;
                out[out_idx] = 0;
                out[out_idx + 1] = 0;
                out[out_idx + 2] = 0;
            }
        }
    }
    Ok(())
}

/// Compute the raw residual between ground truth Y and predicted Y.
/// Returns: residual[i] = y_gt[i] as i16 - y_pred[i] as i16
pub fn compute_residual(y_gt: &[u8], y_pred: &[u8]) -> Vec<i16> {
    y_gt.iter()
        .zip(y_pred.iter())
        .map(|(&gt_val, &pred_val)| gt_val as i16 - pred_val as i16)
        .collect()
}

/// Center a raw residual for JPEG encoding: enc = clamp(round(residual + 128), 0, 255)
pub fn center_residual(residual: &[i16]) -> Vec<u8> {
    residual
        .iter()
        .map(|&r| (r + 128).clamp(0, 255) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_and_center() {
        let gt = vec![150u8, 100, 200, 50];
        let pred = vec![140u8, 110, 180, 60];
        let raw = compute_residual(&gt, &pred);
        assert_eq!(raw, vec![10, -10, 20, -10]);

        let centered = center_residual(&raw);
        assert_eq!(centered, vec![138, 118, 148, 118]);
    }

    #[test]
    fn test_center_clamp() {
        // Test clamping at boundaries
        let raw = vec![200i16, -200];
        let centered = center_residual(&raw);
        assert_eq!(centered, vec![255, 0]);
    }
}
