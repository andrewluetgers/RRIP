/// Convert RGB to YCbCr using float BT.601
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    let y = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
    let cb = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0).round().clamp(0.0, 255.0) as u8;
    let cr = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0).round().clamp(0.0, 255.0) as u8;

    (y, cb, cr)
}

/// Convert YCbCr to RGB using float BT.601
#[inline]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as f32;
    let cb = cb as f32 - 128.0;
    let cr = cr as f32 - 128.0;

    let r = (y + 1.402 * cr).round().clamp(0.0, 255.0) as u8;
    let g = (y - 0.344136 * cb - 0.714136 * cr).round().clamp(0.0, 255.0) as u8;
    let b = (y + 1.772 * cb).round().clamp(0.0, 255.0) as u8;

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

/// Extract Y, Cb, Cr planes from interleaved RGB pixels as f32 (no u8 quantization)
pub fn ycbcr_planes_from_rgb_f32(rgb: &[u8], width: u32, height: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let size = (width * height) as usize;
    let mut y = vec![0.0f32; size];
    let mut cb = vec![0.0f32; size];
    let mut cr = vec![0.0f32; size];
    for i in 0..size {
        let src = i * 3;
        let r = rgb[src] as f32;
        let g = rgb[src + 1] as f32;
        let b = rgb[src + 2] as f32;
        y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        cr[i] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
    }
    (y, cb, cr)
}

/// Convert f32 Y, Cb, Cr to interleaved RGB u8 (single quantization at end)
pub fn rgb_from_ycbcr_f32(y: &[f32], cb: &[f32], cr: &[f32]) -> Vec<u8> {
    let n = y.len();
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        let yf = y[i];
        let cbf = cb[i] - 128.0;
        let crf = cr[i] - 128.0;
        let r = (yf + 1.402 * crf).round().clamp(0.0, 255.0) as u8;
        let g = (yf - 0.344136 * cbf - 0.714136 * crf).round().clamp(0.0, 255.0) as u8;
        let b = (yf + 1.772 * cbf).round().clamp(0.0, 255.0) as u8;
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    rgb
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
        // Float BT.601 roundtrip has ≤2 LSB error due to u8 quantization of Y/Cb/Cr
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
                    let (r2, g2, b2) = ycbcr_to_rgb(y, cb, cr);
                    assert!((r as i32 - r2 as i32).abs() <= 2, "R mismatch: {} vs {} (from {},{},{})", r, r2, r, g, b);
                    assert!((g as i32 - g2 as i32).abs() <= 2, "G mismatch: {} vs {} (from {},{},{})", g, g2, r, g, b);
                    assert!((b as i32 - b2 as i32).abs() <= 2, "B mismatch: {} vs {} (from {},{},{})", b, b2, r, g, b);
                }
            }
        }
    }
}
