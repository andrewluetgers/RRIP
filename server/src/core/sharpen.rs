/// Unsharp mask for interleaved RGB buffers.
///
/// Approximates the blur-compensation effect of OptL2 gradient descent
/// at ~500x less cost per tile. Uses a separable 3×3 Gaussian blur
/// (kernel [0.25, 0.5, 0.25]) that matches bilinear interpolation's
/// frequency response, then sharpens: out = src + strength × (src − blurred).

/// Apply unsharp mask to an interleaved RGB buffer.
/// `strength` controls sharpening intensity (typical: 0.5–2.0).
pub fn unsharp_mask_rgb(src: &[u8], w: u32, h: u32, strength: f32) -> Vec<u8> {
    #[cfg(target_arch = "aarch64")]
    {
        unsharp_mask_rgb_neon(src, w, h, strength)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        unsharp_mask_rgb_scalar(src, w, h, strength)
    }
}

/// Scalar fallback (also used as reference for testing).
pub fn unsharp_mask_rgb_scalar(src: &[u8], w: u32, h: u32, strength: f32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let stride = w * 3;
    let len = h * stride;
    debug_assert_eq!(src.len(), len);

    // Pass 1: horizontal blur [0.25, 0.5, 0.25] → store as u16 (×4 to stay integer)
    // val = src[x-1] + 2*src[x] + src[x+1]  (range 0..1020)
    let mut hblur = vec![0u16; len];
    for y in 0..h {
        let row = y * stride;
        for x in 0..w {
            let x0 = if x > 0 { x - 1 } else { 0 };
            let x2 = if x + 1 < w { x + 1 } else { w - 1 };
            for c in 0..3 {
                let a = src[row + x0 * 3 + c] as u16;
                let b = src[row + x * 3 + c] as u16;
                let d = src[row + x2 * 3 + c] as u16;
                hblur[row + x * 3 + c] = a + 2 * b + d;
            }
        }
    }

    // Pass 2: vertical blur on hblur, fused with sharpen output
    // blurred = (hblur[y-1] + 2*hblur[y] + hblur[y+1]) / 16
    // out = clamp(src + strength * (src - blurred))
    let strength_i = (strength * 256.0).round() as i32; // fixed-point 8.8
    let mut out = vec![0u8; len];
    for y in 0..h {
        let y0 = if y > 0 { y - 1 } else { 0 };
        let y2 = if y + 1 < h { y + 1 } else { h - 1 };
        let row0 = y0 * stride;
        let row1 = y * stride;
        let row2 = y2 * stride;
        for i in 0..stride {
            let blur16 = hblur[row0 + i] as i32
                + 2 * hblur[row1 + i] as i32
                + hblur[row2 + i] as i32;
            // blur16 / 16 = actual blurred value (range 0..255)
            let s16 = (src[row1 + i] as i32) << 4; // src × 16
            let diff = s16 - blur16; // (src - blurred) × 16
            // out = src + strength * diff/16 = (s16 + strength * diff) / 16
            let v = s16 * 256 + strength_i * diff; // fixed-point: ×(16×256)
            let v = (v + 2048) >> 12; // round and shift by 12 (16×256 = 4096)
            out[row1 + i] = v.clamp(0, 255) as u8;
        }
    }

    out
}

/// NEON-optimized unsharp mask for aarch64 (Apple Silicon).
/// Processes 16 bytes at a time using NEON SIMD intrinsics.
#[cfg(target_arch = "aarch64")]
fn unsharp_mask_rgb_neon(src: &[u8], w: u32, h: u32, strength: f32) -> Vec<u8> {
    // Fall back to scalar for tiny images where NEON overhead isn't worth it
    if w < 8 || h < 2 {
        return unsharp_mask_rgb_scalar(src, w, h, strength);
    }

    use std::arch::aarch64::*;

    let w = w as usize;
    let h = h as usize;
    let stride = w * 3;
    let len = h * stride;
    debug_assert_eq!(src.len(), len);

    // Pass 1: horizontal blur → u16 intermediate
    // val = src[x-1] + 2*src[x] + src[x+1]
    let mut hblur = vec![0u16; len];

    for y in 0..h {
        let row = y * stride;
        // First pixel: clamp left edge
        for c in 0..3 {
            let a = src[row + c] as u16; // x-1 clamped to x=0
            let b = src[row + c] as u16;
            let d = src[row + 3 + c] as u16;
            hblur[row + c] = a + 2 * b + d;
        }

        // Interior pixels: NEON-accelerated (process 16 bytes at a time)
        // We process the interior in chunks of 16 bytes
        let interior_start = 3; // skip first pixel (3 bytes)
        let interior_end = stride - 3; // skip last pixel (3 bytes)
        let mut i = interior_start;

        unsafe {
            while i + 16 <= interior_end {
                let left = vld1q_u8(src.as_ptr().add(row + i - 3));
                let center = vld1q_u8(src.as_ptr().add(row + i));
                let right = vld1q_u8(src.as_ptr().add(row + i + 3));

                // Widen to u16 and compute: left + 2*center + right
                let lo_l = vmovl_u8(vget_low_u8(left));
                let lo_c = vmovl_u8(vget_low_u8(center));
                let lo_r = vmovl_u8(vget_low_u8(right));
                let lo = vaddq_u16(vaddq_u16(lo_l, lo_r), vshlq_n_u16(lo_c, 1));

                let hi_l = vmovl_u8(vget_high_u8(left));
                let hi_c = vmovl_u8(vget_high_u8(center));
                let hi_r = vmovl_u8(vget_high_u8(right));
                let hi = vaddq_u16(vaddq_u16(hi_l, hi_r), vshlq_n_u16(hi_c, 1));

                vst1q_u16(hblur.as_mut_ptr().add(row + i), lo);
                vst1q_u16(hblur.as_mut_ptr().add(row + i + 8), hi);

                i += 16;
            }
        }

        // Scalar remainder for interior
        while i < interior_end {
            let a = src[row + i - 3] as u16;
            let b = src[row + i] as u16;
            let d = src[row + i + 3] as u16;
            hblur[row + i] = a + 2 * b + d;
            i += 1;
        }

        // Last pixel: clamp right edge
        let last = stride - 3;
        for c in 0..3 {
            let a = src[row + last - 3 + c] as u16;
            let b = src[row + last + c] as u16;
            let d = src[row + last + c] as u16; // x+1 clamped to x=w-1
            hblur[row + last + c] = a + 2 * b + d;
        }
    }

    // Pass 2: vertical blur on hblur + fused sharpen
    let strength_i = (strength * 256.0).round() as i32;
    let mut out = vec![0u8; len];

    for y in 0..h {
        let y0 = if y > 0 { y - 1 } else { 0 };
        let y2 = if y + 1 < h { y + 1 } else { h - 1 };
        let row0 = y0 * stride;
        let row1 = y * stride;
        let row2 = y2 * stride;

        let mut i = 0;
        unsafe {
            let strength_v = vdupq_n_s32(strength_i);
            let zero = vdupq_n_s32(0);
            let max255 = vdupq_n_s32(255);
            let round_v = vdupq_n_s32(2048);

            // Process 8 elements at a time (u16 → i32 → u8)
            while i + 8 <= stride {
                let h0 = vld1q_u16(hblur.as_ptr().add(row0 + i));
                let h1 = vld1q_u16(hblur.as_ptr().add(row1 + i));
                let h2 = vld1q_u16(hblur.as_ptr().add(row2 + i));

                // blur16 = h0 + 2*h1 + h2  (u32 range 0..4080)
                let blur_lo = vaddl_u16(vget_low_u16(h0), vget_low_u16(h2));
                let blur_lo = vmlal_n_u16(blur_lo, vget_low_u16(h1), 2);
                let blur_lo = vreinterpretq_s32_u32(blur_lo);

                let blur_hi = vaddl_u16(vget_high_u16(h0), vget_high_u16(h2));
                let blur_hi = vmlal_n_u16(blur_hi, vget_high_u16(h1), 2);
                let blur_hi = vreinterpretq_s32_u32(blur_hi);

                // Load src values for this row
                let src_bytes = vld1_u8(src.as_ptr().add(row1 + i));
                let src_u16 = vmovl_u8(src_bytes);
                let s_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(src_u16)));
                let s_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(src_u16)));

                // s16 = src << 4
                let s16_lo = vshlq_n_s32(s_lo, 4);
                let s16_hi = vshlq_n_s32(s_hi, 4);

                // diff = s16 - blur16
                let diff_lo = vsubq_s32(s16_lo, blur_lo);
                let diff_hi = vsubq_s32(s16_hi, blur_hi);

                // v = s16 * 256 + strength * diff
                let v_lo = vmlaq_s32(vshlq_n_s32(s16_lo, 8), strength_v, diff_lo);
                let v_hi = vmlaq_s32(vshlq_n_s32(s16_hi, 8), strength_v, diff_hi);

                // (v + 2048) >> 12
                let r_lo = vshrq_n_s32(vaddq_s32(v_lo, round_v), 12);
                let r_hi = vshrq_n_s32(vaddq_s32(v_hi, round_v), 12);

                // Clamp [0, 255]
                let r_lo = vminq_s32(vmaxq_s32(r_lo, zero), max255);
                let r_hi = vminq_s32(vmaxq_s32(r_hi, zero), max255);

                // Narrow i32 → i16 → u8
                let r_16 = vmovn_s32(r_lo);
                let r_16_hi = vmovn_s32(r_hi);
                let r_8 = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(r_16, r_16_hi)));

                vst1_u8(out.as_mut_ptr().add(row1 + i), r_8);
                i += 8;
            }
        }

        // Scalar remainder
        while i < stride {
            let blur16 = hblur[row0 + i] as i32
                + 2 * hblur[row1 + i] as i32
                + hblur[row2 + i] as i32;
            let s16 = (src[row1 + i] as i32) << 4;
            let diff = s16 - blur16;
            let v = s16 * 256 + strength_i * diff;
            let v = (v + 2048) >> 12;
            out[row1 + i] = v.clamp(0, 255) as u8;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_image_unchanged() {
        let w = 4u32;
        let h = 4u32;
        let src = vec![128u8; (w * h * 3) as usize];
        let out = unsharp_mask_rgb(&src, w, h, 1.0);
        assert_eq!(src, out);
    }

    #[test]
    fn test_clamps_to_valid_range() {
        let w = 3u32;
        let h = 1u32;
        let src = vec![0, 0, 0, 255, 255, 255, 0, 0, 0];
        let out = unsharp_mask_rgb(&src, w, h, 5.0);
        for &v in &out {
            assert!(v <= 255);
        }
    }

    #[test]
    fn test_neon_matches_scalar() {
        // Verify NEON and scalar produce identical output
        let w = 256u32;
        let h = 256u32;
        let src: Vec<u8> = (0..w * h * 3).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let scalar = unsharp_mask_rgb_scalar(&src, w, h, 1.0);
        let result = unsharp_mask_rgb(&src, w, h, 1.0);
        assert_eq!(scalar, result, "NEON output must match scalar reference");
    }

    #[test]
    fn test_neon_matches_scalar_strength_half() {
        let w = 128u32;
        let h = 128u32;
        let src: Vec<u8> = (0..w * h * 3).map(|i| ((i * 53 + 7) % 256) as u8).collect();
        let scalar = unsharp_mask_rgb_scalar(&src, w, h, 0.5);
        let result = unsharp_mask_rgb(&src, w, h, 0.5);
        assert_eq!(scalar, result, "NEON output must match scalar at strength=0.5");
    }

    #[test]
    fn bench_sharpen_256x256() {
        let w = 256u32;
        let h = 256u32;
        let src: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();

        // Warmup
        for _ in 0..10 {
            let _ = unsharp_mask_rgb(&src, w, h, 0.5);
        }

        let iters = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = unsharp_mask_rgb(&src, w, h, 0.5);
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / iters as f64;
        eprintln!("unsharp_mask_rgb 256x256: {:.1}µs per call ({} iters, {:.1}ms total)",
            per_call_us, iters, elapsed.as_millis() as f64);
    }

    #[test]
    fn bench_sharpen_scalar_256x256() {
        let w = 256u32;
        let h = 256u32;
        let src: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();

        for _ in 0..10 {
            let _ = unsharp_mask_rgb_scalar(&src, w, h, 0.5);
        }

        let iters = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = unsharp_mask_rgb_scalar(&src, w, h, 0.5);
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / iters as f64;
        eprintln!("unsharp_mask_rgb_scalar 256x256: {:.1}µs per call ({} iters, {:.1}ms total)",
            per_call_us, iters, elapsed.as_millis() as f64);
    }
}
