// Fast 2x and 4x upsampling for RGB images
// Optimized for speed over quality for real-time tile generation

use rayon::prelude::*;

/// Fast 2x upsampling using simple bilinear interpolation
/// Optimized for the specific case of doubling dimensions
pub fn upsample_2x_rgb(src: &[u8], src_width: usize, src_height: usize) -> Vec<u8> {
    let dst_width = src_width * 2;
    let dst_height = src_height * 2;
    let mut dst = vec![0u8; dst_width * dst_height * 3];

    // Process in parallel by rows for better cache locality
    dst.par_chunks_mut(dst_width * 3)
        .enumerate()
        .for_each(|(dst_y, dst_row)| {
            let src_y = dst_y >> 1;  // dst_y / 2
            let y_frac = (dst_y & 1) as u8;  // dst_y % 2

            let src_y_next = (src_y + 1).min(src_height - 1);

            for dst_x in 0..dst_width {
                let src_x = dst_x >> 1;  // dst_x / 2
                let x_frac = (dst_x & 1) as u8;  // dst_x % 2

                let src_x_next = (src_x + 1).min(src_width - 1);

                // Get the 4 source pixels
                let src_idx00 = (src_y * src_width + src_x) * 3;
                let src_idx10 = (src_y * src_width + src_x_next) * 3;
                let src_idx01 = (src_y_next * src_width + src_x) * 3;
                let src_idx11 = (src_y_next * src_width + src_x_next) * 3;

                let dst_idx = dst_x * 3;

                // Fast bilinear interpolation using integer math
                // When x_frac or y_frac is 0, this reduces to simple copying
                // When both are 1, this averages all 4 pixels
                for c in 0..3 {
                    let p00 = src[src_idx00 + c] as u16;
                    let p10 = src[src_idx10 + c] as u16;
                    let p01 = src[src_idx01 + c] as u16;
                    let p11 = src[src_idx11 + c] as u16;

                    // Optimized bilinear interpolation
                    // When frac is 0: result = p00
                    // When frac is 1: result = average of pixels
                    let top = p00 + ((p10 - p00) * x_frac as u16);
                    let bottom = p01 + ((p11 - p01) * x_frac as u16);
                    let result = top + ((bottom - top) * y_frac as u16);

                    dst_row[dst_idx + c] = result as u8;
                }
            }
        });

    dst
}

/// Fast 4x upsampling - essentially two 2x operations but optimized
/// Uses nearest neighbor for the intermediate pixels to save computation
pub fn upsample_4x_rgb(src: &[u8], src_width: usize, src_height: usize) -> Vec<u8> {
    let dst_width = src_width * 4;
    let dst_height = src_height * 4;
    let mut dst = vec![0u8; dst_width * dst_height * 3];

    // Process in parallel by rows
    dst.par_chunks_mut(dst_width * 3)
        .enumerate()
        .for_each(|(dst_y, dst_row)| {
            let src_y = dst_y >> 2;  // dst_y / 4
            let y_frac = dst_y & 3;  // dst_y % 4

            let src_y_next = (src_y + 1).min(src_height - 1);

            for dst_x in 0..dst_width {
                let src_x = dst_x >> 2;  // dst_x / 4
                let x_frac = dst_x & 3;  // dst_x % 4

                let src_x_next = (src_x + 1).min(src_width - 1);

                // Get the 4 source pixels
                let src_idx00 = (src_y * src_width + src_x) * 3;
                let src_idx10 = (src_y * src_width + src_x_next) * 3;
                let src_idx01 = (src_y_next * src_width + src_x) * 3;
                let src_idx11 = (src_y_next * src_width + src_x_next) * 3;

                let dst_idx = dst_x * 3;

                // Simple bilinear for 4x - prioritize speed
                // Use weights based on distance
                let w00 = (4 - x_frac) * (4 - y_frac);
                let w10 = x_frac * (4 - y_frac);
                let w01 = (4 - x_frac) * y_frac;
                let w11 = x_frac * y_frac;

                for c in 0..3 {
                    let p00 = src[src_idx00 + c] as u32;
                    let p10 = src[src_idx10 + c] as u32;
                    let p01 = src[src_idx01 + c] as u32;
                    let p11 = src[src_idx11 + c] as u32;

                    let result = (p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11) >> 4;
                    dst_row[dst_idx + c] = result.min(255) as u8;
                }
            }
        });

    dst
}

/// Optimized nearest-neighbor 2x upsampling (fastest, lower quality)
/// Each source pixel becomes a 2x2 block in the destination
pub fn upsample_2x_rgb_nearest(src: &[u8], src_width: usize, src_height: usize) -> Vec<u8> {
    let dst_width = src_width * 2;
    let dst_height = src_height * 2;
    let mut dst = vec![0u8; dst_width * dst_height * 3];

    // Process in parallel by source rows (each produces 2 dst rows)
    (0..src_height).into_par_iter().for_each(|src_y| {
        let dst_y0 = src_y * 2;
        let dst_y1 = dst_y0 + 1;

        for src_x in 0..src_width {
            let src_idx = (src_y * src_width + src_x) * 3;
            let dst_x0 = src_x * 2;
            let dst_x1 = dst_x0 + 1;

            // Copy the pixel to 4 destination locations
            let dst_idx00 = (dst_y0 * dst_width + dst_x0) * 3;
            let dst_idx10 = (dst_y0 * dst_width + dst_x1) * 3;
            let dst_idx01 = (dst_y1 * dst_width + dst_x0) * 3;
            let dst_idx11 = (dst_y1 * dst_width + dst_x1) * 3;

            // Unroll the channel loop for better performance
            unsafe {
                // Use unsafe for direct memory copy (safe because indices are valid)
                std::ptr::copy_nonoverlapping(src.as_ptr().add(src_idx),
                                             dst.as_mut_ptr().add(dst_idx00), 3);
                std::ptr::copy_nonoverlapping(src.as_ptr().add(src_idx),
                                             dst.as_mut_ptr().add(dst_idx10), 3);
                std::ptr::copy_nonoverlapping(src.as_ptr().add(src_idx),
                                             dst.as_mut_ptr().add(dst_idx01), 3);
                std::ptr::copy_nonoverlapping(src.as_ptr().add(src_idx),
                                             dst.as_mut_ptr().add(dst_idx11), 3);
            }
        }
    });

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsample_2x() {
        // Test with a simple 2x2 RGB image
        let src = vec![
            255, 0, 0,    0, 255, 0,  // Red, Green
            0, 0, 255,    255, 255, 0, // Blue, Yellow
        ];

        let result = upsample_2x_rgb(&src, 2, 2);
        assert_eq!(result.len(), 4 * 4 * 3); // 4x4 output

        // Check that first pixel is red-ish
        assert!(result[0] > 200); // R
        assert!(result[1] < 100); // G
        assert!(result[2] < 100); // B
    }

    #[test]
    fn test_upsample_4x() {
        let src = vec![
            255, 0, 0,    0, 255, 0,  // Red, Green
            0, 0, 255,    255, 255, 0, // Blue, Yellow
        ];

        let result = upsample_4x_rgb(&src, 2, 2);
        assert_eq!(result.len(), 8 * 8 * 3); // 8x8 output
    }
}