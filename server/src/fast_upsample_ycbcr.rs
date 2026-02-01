// Fast upsampling in YCbCr color space
// Avoids RGB conversion overhead and works directly with JPEG's native format

use rayon::prelude::*;

/// Fast 2x upsampling for a single channel (Y, Cb, or Cr)
/// Uses bilinear interpolation optimized for 2x scaling
pub fn upsample_2x_channel(src: &[u8], src_width: usize, src_height: usize) -> Vec<u8> {
    let dst_width = src_width * 2;
    let dst_height = src_height * 2;
    let mut dst = vec![0u8; dst_width * dst_height];

    // Process in parallel by destination rows
    dst.par_chunks_mut(dst_width)
        .enumerate()
        .for_each(|(dst_y, dst_row)| {
            let src_y = dst_y >> 1;
            let y_frac = (dst_y & 1) as u16;
            let src_y_next = (src_y + 1).min(src_height - 1);

            for dst_x in 0..dst_width {
                let src_x = dst_x >> 1;
                let x_frac = (dst_x & 1) as u16;
                let src_x_next = (src_x + 1).min(src_width - 1);

                // Get 4 source pixels
                let p00 = src[src_y * src_width + src_x] as u16;
                let p10 = src[src_y * src_width + src_x_next] as u16;
                let p01 = src[src_y_next * src_width + src_x] as u16;
                let p11 = src[src_y_next * src_width + src_x_next] as u16;

                // Fast bilinear with bit shifts
                // When frac is 0: takes p00
                // When frac is 1: averages pixels
                let top = p00 + ((p10.saturating_sub(p00)) * x_frac);
                let bottom = p01 + ((p11.saturating_sub(p01)) * x_frac);
                dst_row[dst_x] = (top + ((bottom.saturating_sub(top)) * y_frac)) as u8;
            }
        });

    dst
}

/// Ultra-fast nearest neighbor 2x upsampling for chroma channels
/// Since human vision is less sensitive to chroma resolution
pub fn upsample_2x_nearest(src: &[u8], src_width: usize, src_height: usize) -> Vec<u8> {
    let dst_width = src_width * 2;
    let dst_height = src_height * 2;
    let dst = vec![0u8; dst_width * dst_height];

    // Each source pixel becomes 2x2 block
    (0..src_height).into_par_iter().for_each(|src_y| {
        let dst_y0 = src_y * 2;
        let dst_y1 = dst_y0 + 1;

        for src_x in 0..src_width {
            let pixel = src[src_y * src_width + src_x];
            let dst_x0 = src_x * 2;
            let dst_x1 = dst_x0 + 1;

            // Write to 4 destination locations
            unsafe {
                let base_ptr = dst.as_ptr() as *mut u8;
                *base_ptr.add(dst_y0 * dst_width + dst_x0) = pixel;
                *base_ptr.add(dst_y0 * dst_width + dst_x1) = pixel;
                *base_ptr.add(dst_y1 * dst_width + dst_x0) = pixel;
                *base_ptr.add(dst_y1 * dst_width + dst_x1) = pixel;
            }
        }
    });

    dst
}

/// Fast 4x upsampling for luma channel
pub fn upsample_4x_channel(src: &[u8], src_width: usize, src_height: usize) -> Vec<u8> {
    let dst_width = src_width * 4;
    let dst_height = src_height * 4;
    let mut dst = vec![0u8; dst_width * dst_height];

    dst.par_chunks_mut(dst_width)
        .enumerate()
        .for_each(|(dst_y, dst_row)| {
            let src_y = dst_y >> 2;
            let y_frac = dst_y & 3;
            let src_y_next = (src_y + 1).min(src_height - 1);

            for dst_x in 0..dst_width {
                let src_x = dst_x >> 2;
                let x_frac = dst_x & 3;
                let src_x_next = (src_x + 1).min(src_width - 1);

                let p00 = src[src_y * src_width + src_x] as u32;
                let p10 = src[src_y * src_width + src_x_next] as u32;
                let p01 = src[src_y_next * src_width + src_x] as u32;
                let p11 = src[src_y_next * src_width + src_x_next] as u32;

                // Weight by distance (0-3 becomes 4-1)
                let w00 = ((4 - x_frac) * (4 - y_frac)) as u32;
                let w10 = (x_frac * (4 - y_frac)) as u32;
                let w01 = ((4 - x_frac) * y_frac) as u32;
                let w11 = (x_frac * y_frac) as u32;

                dst_row[dst_x] = ((p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11) >> 4) as u8;
            }
        });

    dst
}

/// Fast 4x nearest neighbor for chroma
pub fn upsample_4x_nearest(src: &[u8], src_width: usize, src_height: usize) -> Vec<u8> {
    let dst_width = src_width * 4;
    let dst_height = src_height * 4;
    let dst = vec![0u8; dst_width * dst_height];

    // Each source pixel becomes 4x4 block
    (0..src_height).into_par_iter().for_each(|src_y| {
        let dst_y_start = src_y * 4;

        for src_x in 0..src_width {
            let pixel = src[src_y * src_width + src_x];
            let dst_x_start = src_x * 4;

            // Fill 4x4 block
            for dy in 0..4 {
                let dst_y = dst_y_start + dy;
                let row_start = dst_y * dst_width + dst_x_start;
                unsafe {
                    let ptr = dst.as_ptr() as *mut u8;
                    std::ptr::write_bytes(ptr.add(row_start), pixel, 4);
                }
            }
        }
    });

    dst
}

/// Apply residual to luma channel with clamping
/// residual is assumed to be offset by 128 (JPEG standard)
#[inline]
pub fn apply_residual(prediction: &[u8], residual: &[u8]) -> Vec<u8> {
    prediction.par_iter()
        .zip(residual.par_iter())
        .map(|(&pred, &res)| {
            let result = pred as i16 + (res as i16 - 128);
            result.clamp(0, 255) as u8
        })
        .collect()
}

/// Structure to hold YCbCr planes
pub struct YCbCrPlanes {
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl YCbCrPlanes {
    /// Upsample YCbCr planes by 2x
    /// Uses high quality for Y, fast nearest for Cb/Cr
    pub fn upsample_2x(&self) -> YCbCrPlanes {
        let new_width = self.width * 2;
        let new_height = self.height * 2;

        // Use parallel processing for all three channels
        let (y, (cb, cr)) = rayon::join(
            || upsample_2x_channel(&self.y, self.width, self.height),
            || rayon::join(
                || upsample_2x_nearest(&self.cb, self.width, self.height),
                || upsample_2x_nearest(&self.cr, self.width, self.height),
            )
        );

        YCbCrPlanes {
            y,
            cb,
            cr,
            width: new_width,
            height: new_height,
        }
    }

    /// Upsample YCbCr planes by 4x
    pub fn upsample_4x(&self) -> YCbCrPlanes {
        let new_width = self.width * 4;
        let new_height = self.height * 4;

        let (y, (cb, cr)) = rayon::join(
            || upsample_4x_channel(&self.y, self.width, self.height),
            || rayon::join(
                || upsample_4x_nearest(&self.cb, self.width, self.height),
                || upsample_4x_nearest(&self.cr, self.width, self.height),
            )
        );

        YCbCrPlanes {
            y,
            cb,
            cr,
            width: new_width,
            height: new_height,
        }
    }

    /// Apply residual to Y channel only
    pub fn apply_y_residual(&mut self, residual: &[u8]) {
        self.y = apply_residual(&self.y, residual);
    }

    /// Convert back to interleaved RGB
    /// Note: This is only needed for final output
    pub fn to_rgb(&self) -> Vec<u8> {
        let size = self.width * self.height;
        let mut rgb = vec![0u8; size * 3];

        rgb.par_chunks_mut(self.width * 3)
            .enumerate()
            .for_each(|(row, rgb_row)| {
                let offset = row * self.width;
                for x in 0..self.width {
                    let y = self.y[offset + x] as i32;
                    let cb = self.cb[offset + x] as i32 - 128;
                    let cr = self.cr[offset + x] as i32 - 128;

                    // ITU-R BT.601 conversion
                    let r = y + ((cr * 1436) >> 10);
                    let g = y - ((cb * 352 + cr * 731) >> 10);
                    let b = y + ((cb * 1815) >> 10);

                    let dst_idx = x * 3;
                    rgb_row[dst_idx] = r.clamp(0, 255) as u8;
                    rgb_row[dst_idx + 1] = g.clamp(0, 255) as u8;
                    rgb_row[dst_idx + 2] = b.clamp(0, 255) as u8;
                }
            });

        rgb
    }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
/// NEON-optimized 2x upsampling for ARM processors (like M5)
pub unsafe fn upsample_2x_channel_neon(src: &[u8], src_width: usize, src_height: usize) -> Vec<u8> {
    let dst_width = src_width * 2;
    let dst_height = src_height * 2;
    let mut dst = vec![0u8; dst_width * dst_height];

    for src_y in 0..src_height {
        let dst_y0 = src_y * 2;
        let dst_y1 = dst_y0 + 1;
        let src_y_next = (src_y + 1).min(src_height - 1);

        let mut src_x = 0;

        // Process 8 pixels at a time with NEON
        while src_x + 8 <= src_width {
            // Load 8 source pixels
            let row0 = vld1_u8(src.as_ptr().add(src_y * src_width + src_x));
            let row1 = vld1_u8(src.as_ptr().add(src_y_next * src_width + src_x));

            // Duplicate each pixel horizontally
            let row0_dup = vzip1_u8(row0, row0);
            let row1_dup = vzip1_u8(row1, row1);

            // Average vertically for intermediate row
            let avg = vrhadd_u8(row0_dup, row1_dup);

            // Store results
            vst1q_u8(dst.as_mut_ptr().add(dst_y0 * dst_width + src_x * 2),
                     vcombine_u8(row0_dup, vdup_n_u8(0)));
            vst1q_u8(dst.as_mut_ptr().add(dst_y1 * dst_width + src_x * 2),
                     vcombine_u8(avg, vdup_n_u8(0)));

            src_x += 8;
        }

        // Handle remaining pixels
        while src_x < src_width {
            let p0 = src[src_y * src_width + src_x];
            let p1 = src[src_y_next * src_width + src_x];
            let avg = ((p0 as u16 + p1 as u16) >> 1) as u8;

            let dst_x = src_x * 2;
            dst[dst_y0 * dst_width + dst_x] = p0;
            dst[dst_y0 * dst_width + dst_x + 1] = p0;
            dst[dst_y1 * dst_width + dst_x] = avg;
            dst[dst_y1 * dst_width + dst_x + 1] = avg;

            src_x += 1;
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsample_channel() {
        let src = vec![100, 150, 200, 250];
        let result = upsample_2x_channel(&src, 2, 2);
        assert_eq!(result.len(), 16);

        // Check corners match source
        assert_eq!(result[0], 100);
        assert!(result[3] <= 150); // Interpolated
    }

    #[test]
    fn test_apply_residual() {
        let pred = vec![100, 150, 200];
        let residual = vec![138, 118, 148]; // +10, -10, +20 after removing 128 offset
        let result = apply_residual(&pred, &residual);

        assert_eq!(result[0], 110);
        assert_eq!(result[1], 140);
        assert_eq!(result[2], 220);
    }
}