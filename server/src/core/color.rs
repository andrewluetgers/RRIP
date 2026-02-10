/// Convert RGB to YCbCr using fixed-point BT.601 (10-bit precision)
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    // ITU-R BT.601 coefficients scaled by 1024
    let y = ((306 * r + 601 * g + 117 * b) >> 10).clamp(0, 255) as u8;
    let cb = (((-173 * r - 339 * g + 512 * b) >> 10) + 128).clamp(0, 255) as u8;
    let cr = (((512 * r - 429 * g - 83 * b) >> 10) + 128).clamp(0, 255) as u8;

    (y, cb, cr)
}

/// Convert YCbCr to RGB using fixed-point BT.601 (10-bit precision)
#[inline]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;

    // ITU-R BT.601 coefficients scaled by 1024
    let r = (y + ((cr * 1436) >> 10)).clamp(0, 255) as u8;
    let g = (y - ((cb * 352 + cr * 731) >> 10)).clamp(0, 255) as u8;
    let b = (y + ((cb * 1815) >> 10)).clamp(0, 255) as u8;

    (r, g, b)
}

/// Extract Y, Cb, Cr planes from interleaved RGB pixels
pub fn ycbcr_planes_from_rgb(rgb: &[u8], width: u32, height: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let size = (width * height) as usize;
    let mut y = vec![0u8; size];
    let mut cb = vec![0u8; size];
    let mut cr = vec![0u8; size];
    for i in 0..size {
        let src = i * 3;
        let (yy, cbc, crc) = rgb_to_ycbcr(rgb[src], rgb[src + 1], rgb[src + 2]);
        y[i] = yy;
        cb[i] = cbc;
        cr[i] = crc;
    }
    (y, cb, cr)
}

/// Convert Y, Cb, Cr planes to interleaved RGB pixels
#[allow(dead_code)]
pub fn rgb_from_ycbcr_planes(y: &[u8], cb: &[u8], cr: &[u8], width: u32, height: u32) -> Vec<u8> {
    let size = (width * height) as usize;
    let mut rgb = vec![0u8; size * 3];
    for i in 0..size {
        let (r, g, b) = ycbcr_to_rgb(y[i], cb[i], cr[i]);
        let dst = i * 3;
        rgb[dst] = r;
        rgb[dst + 1] = g;
        rgb[dst + 2] = b;
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        // Test that RGB -> YCbCr -> RGB is close to identity
        // Fixed-point BT.601 can have up to ~3 LSB error due to rounding
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
                    let (r2, g2, b2) = ycbcr_to_rgb(y, cb, cr);
                    assert!((r as i32 - r2 as i32).abs() <= 3, "R mismatch: {} vs {}", r, r2);
                    assert!((g as i32 - g2 as i32).abs() <= 3, "G mismatch: {} vs {}", g, g2);
                    assert!((b as i32 - b2 as i32).abs() <= 3, "B mismatch: {} vs {}", b, b2);
                }
            }
        }
    }
}
