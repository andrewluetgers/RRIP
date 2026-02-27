// Optimized TurboJPEG operations for fast JPEG encoding/decoding
// Replaces slow image crate operations with direct TurboJPEG calls

use anyhow::{Result, Context};
use turbojpeg::{Compressor, Decompressor, Image, PixelFormat, Subsamp};
use std::fs;
use std::path::Path;

// Thread-local TurboJPEG instances for best performance
thread_local! {
    static DECOMPRESSOR: std::cell::RefCell<Decompressor> = std::cell::RefCell::new(
        Decompressor::new().expect("Failed to create TurboJPEG decompressor")
    );

    static COMPRESSOR: std::cell::RefCell<Compressor> = std::cell::RefCell::new(
        Compressor::new().expect("Failed to create TurboJPEG compressor")
    );
}

/// Fast grayscale JPEG loading using TurboJPEG
/// Returns raw grayscale bytes (much faster than image crate)
pub fn load_luma_turbo(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let jpeg_data = fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;

    decode_luma_turbo(&jpeg_data)
}

/// Decode RGB JPEG from bytes using TurboJPEG
pub fn decode_rgb_turbo(jpeg_data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    DECOMPRESSOR.with(|dec| {
        let mut decompressor = dec.borrow_mut();
        let header = decompressor.read_header(jpeg_data)?;
        let width = header.width;
        let height = header.height;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        let image = Image {
            pixels: &mut pixels[..],
            width,
            height,
            pitch: (width * 3) as usize,
            format: PixelFormat::RGB,
        };
        decompressor.decompress(jpeg_data, image)?;
        Ok((pixels, width as u32, height as u32))
    })
}

/// Decode grayscale JPEG from bytes using TurboJPEG
pub fn decode_luma_turbo(jpeg_data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    DECOMPRESSOR.with(|dec| {
        let mut decompressor = dec.borrow_mut();

        // Get header info
        let header = decompressor.read_header(jpeg_data)?;
        let width = header.width;
        let height = header.height;

        // Decode directly to grayscale
        let mut pixels = vec![0u8; (width * height) as usize];
        let image = Image {
            pixels: &mut pixels[..],
            width,
            height,
            pitch: width as usize,
            format: PixelFormat::GRAY,
        };

        decompressor.decompress(jpeg_data, image)?;

        Ok((pixels, width as u32, height as u32))
    })
}

/// Fast RGB JPEG loading using TurboJPEG
pub fn load_rgb_turbo(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let jpeg_data = fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;

    DECOMPRESSOR.with(|dec| {
        let mut decompressor = dec.borrow_mut();

        let header = decompressor.read_header(&jpeg_data)?;
        let width = header.width;
        let height = header.height;

        let mut pixels = vec![0u8; (width * height * 3) as usize];
        let image = Image {
            pixels: &mut pixels[..],
            width,
            height,
            pitch: (width * 3) as usize,
            format: PixelFormat::RGB,
        };

        decompressor.decompress(&jpeg_data, image)?;

        Ok((pixels, width as u32, height as u32))
    })
}

/// Fast JPEG encoding using TurboJPEG (replaces JpegEncoder)
pub fn encode_jpeg_turbo(pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    COMPRESSOR.with(|comp| {
        let mut compressor = comp.borrow_mut();

        // Set quality (1-100 scale)
        compressor.set_quality(quality as i32);

        // Use 4:4:4 subsampling (no subsampling) for better quality
        // This preserves full chroma resolution at the cost of larger file size
        compressor.set_subsamp(Subsamp::None);

        let image = Image {
            pixels,
            width: width as usize,
            height: height as usize,
            pitch: (width * 3) as usize,
            format: PixelFormat::RGB,
        };

        compressor.compress_to_vec(image)
            .context("TurboJPEG compression failed")
    })
}

/// Fast JPEG encoding with 4:2:0 subsampling (naive).
/// Used by the turbojpeg API path; 420 encoding via libjpeg FFI is preferred
/// for file storage (has Huffman optimization).
#[allow(dead_code)]
pub fn encode_jpeg_turbo_420(pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    COMPRESSOR.with(|comp| {
        let mut compressor = comp.borrow_mut();
        compressor.set_quality(quality as i32);
        compressor.set_subsamp(Subsamp::Sub2x2);

        let image = Image {
            pixels,
            width: width as usize,
            height: height as usize,
            pitch: (width * 3) as usize,
            format: PixelFormat::RGB,
        };

        compressor.compress_to_vec(image)
            .context("TurboJPEG 4:2:0 compression failed")
    })
}

/// JPEG encoding with 4:2:0 and gradient-descent-optimized chroma planes.
/// Uses tjCompressFromYUVPlanes for precise control over chroma content.
pub fn encode_jpeg_turbo_420opt(pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    use crate::core::color::ycbcr_planes_from_rgb;
    use crate::core::optimize_chroma::optimize_chroma_for_upsample;
    use crate::core::ResampleFilter;

    let w = width as usize;
    let h = height as usize;
    let half_w = (w + 1) / 2;
    let half_h = (h + 1) / 2;

    // 1. RGB → Y/Cb/Cr planes
    let (y_plane, cb_full, cr_full) = ycbcr_planes_from_rgb(pixels, width, height);

    // 2. Downsample Cb/Cr from WxH → W/2×H/2 (box average)
    let mut cb_half = downsample_2x_channel(&cb_full, w, h);
    let mut cr_half = downsample_2x_channel(&cr_full, w, h);

    // 3. Optimize each half-res chroma plane via gradient descent
    // Uses Bicubic (CatmullRom) to match the server's default upsample filter.
    cb_half = optimize_chroma_for_upsample(&cb_half, &cb_full, half_w, half_h, 8.0, 50, 0.25, ResampleFilter::Bicubic);
    cr_half = optimize_chroma_for_upsample(&cr_half, &cr_full, half_w, half_h, 8.0, 50, 0.25, ResampleFilter::Bicubic);

    // 4. Encode via tjCompressFromYUVPlanes
    encode_yuv_planes_420(&y_plane, &cb_half, &cr_half, width, height, quality)
}

/// Downsample a single channel by 2x using box average (area-average of 2x2 blocks).
pub fn downsample_2x_channel(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let half_w = (w + 1) / 2;
    let half_h = (h + 1) / 2;
    let mut dst = vec![0u8; half_w * half_h];

    for dy in 0..half_h {
        for dx in 0..half_w {
            let sx = dx * 2;
            let sy = dy * 2;
            let p00 = src[sy * w + sx] as u32;
            let p10 = if sx + 1 < w { src[sy * w + sx + 1] as u32 } else { p00 };
            let p01 = if sy + 1 < h { src[(sy + 1) * w + sx] as u32 } else { p00 };
            let p11 = if sx + 1 < w && sy + 1 < h {
                src[(sy + 1) * w + sx + 1] as u32
            } else if sy + 1 < h {
                p01
            } else {
                p10
            };
            dst[dy * half_w + dx] = ((p00 + p10 + p01 + p11 + 2) / 4) as u8;
        }
    }
    dst
}

/// Encode pre-separated YUV planes in 4:2:0 layout via turbojpeg raw FFI.
/// Y plane: width x height, Cb/Cr planes: (width+1)/2 x (height+1)/2
fn encode_yuv_planes_420(
    y: &[u8], cb: &[u8], cr: &[u8],
    width: u32, height: u32, quality: u8,
) -> Result<Vec<u8>> {
    let w = width as i32;
    let h = height as i32;
    let half_w = ((width + 1) / 2) as usize;
    let half_h = ((height + 1) / 2) as usize;

    // Build contiguous YUV planar buffer for tjCompressFromYUV
    // Layout: Y (w*h) + Cb (half_w*half_h) + Cr (half_w*half_h)
    // pad=1 means no row padding
    let y_size = width as usize * height as usize;
    let c_size = half_w * half_h;
    let yuv_size = y_size + c_size * 2;
    let mut yuv_buf = vec![0u8; yuv_size];

    yuv_buf[..y_size].copy_from_slice(&y[..y_size]);
    yuv_buf[y_size..y_size + c_size].copy_from_slice(&cb[..c_size]);
    yuv_buf[y_size + c_size..y_size + c_size * 2].copy_from_slice(&cr[..c_size]);

    // Call tjCompressFromYUV directly
    unsafe {
        let handle = turbojpeg::raw::tjInitCompress();
        if handle.is_null() {
            anyhow::bail!("tjInitCompress failed");
        }

        // Allocate output buffer
        let _buf_size = turbojpeg::raw::tjBufSize(w, h, turbojpeg::raw::TJSAMP_TJSAMP_420 as i32);
        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: libc::c_ulong = 0;

        let ret = turbojpeg::raw::tjCompressFromYUV(
            handle,
            yuv_buf.as_ptr(),
            w,
            1, // pad = 1 (no row padding)
            h,
            turbojpeg::raw::TJSAMP_TJSAMP_420 as i32,
            &mut jpeg_buf,
            &mut jpeg_size,
            quality as i32,
            0, // flags
        );

        if ret != 0 {
            let err_msg = std::ffi::CStr::from_ptr(turbojpeg::raw::tjGetErrorStr2(handle))
                .to_string_lossy()
                .to_string();
            turbojpeg::raw::tjDestroy(handle);
            anyhow::bail!("tjCompressFromYUV failed: {}", err_msg);
        }

        let result = std::slice::from_raw_parts(jpeg_buf, jpeg_size as usize).to_vec();
        turbojpeg::raw::tjFree(jpeg_buf);
        turbojpeg::raw::tjDestroy(handle);

        Ok(result)
    }
}

/// Fast grayscale JPEG encoding
#[allow(dead_code)]
pub fn encode_luma_turbo(pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    COMPRESSOR.with(|comp| {
        let mut compressor = comp.borrow_mut();

        compressor.set_quality(quality as i32);
        // Grayscale doesn't need subsampling
        compressor.set_subsamp(Subsamp::Gray);

        let image = Image {
            pixels,
            width: width as usize,
            height: height as usize,
            pitch: width as usize,
            format: PixelFormat::GRAY,
        };

        compressor.compress_to_vec(image)
            .context("TurboJPEG grayscale compression failed")
    })
}

/// Optimized residual application with automatic SIMD support
#[inline(always)]
pub fn apply_residual_fast(
    y_pred: &[u8],
    residual: &[u8],
    output: &mut [u8],
    len: usize,
) {
    // Use platform-specific SIMD when available
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { return apply_residual_neon(y_pred, residual, output, len); }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { return apply_residual_avx2(y_pred, residual, output, len); }
    }

    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        unsafe { return apply_residual_sse2(y_pred, residual, output, len); }
    }

    // Fallback to optimized scalar code
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        // Use chunks for better auto-vectorization
        let chunks = len / 8;
        let _remainder = len % 8;

        // Process 8 pixels at a time for SIMD
        for i in 0..chunks {
            let offset = i * 8;
            for j in 0..8 {
                let idx = offset + j;
                let pred = y_pred[idx] as i16;
                let res = residual[idx] as i16 - 128;
                output[idx] = (pred + res).clamp(0, 255) as u8;
            }
        }

        // Handle remainder
        for i in (chunks * 8)..len {
            let pred = y_pred[i] as i16;
            let res = residual[i] as i16 - 128;
            output[i] = (pred + res).clamp(0, 255) as u8;
        }
    }
}

// BufferPool removed - using the one from main.rs instead

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-optimized residual application for ARM processors
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub unsafe fn apply_residual_neon(
    y_pred: &[u8],
    residual: &[u8],
    output: &mut [u8],
    len: usize,
) {
    let chunks = len / 16;
    let _remainder = len % 16;

    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 pixels
        let pred = vld1q_u8(y_pred.as_ptr().add(offset));
        let res = vld1q_u8(residual.as_ptr().add(offset));

        // Convert to signed 16-bit for proper arithmetic
        // Split into low and high halves for 16-bit processing
        let pred_low = vmovl_u8(vget_low_u8(pred));
        let pred_high = vmovl_u8(vget_high_u8(pred));
        let res_low = vmovl_u8(vget_low_u8(res));
        let res_high = vmovl_u8(vget_high_u8(res));

        // Subtract 128 from residual (now in 16-bit, can handle negative)
        let offset_128_16 = vdupq_n_s16(128);
        let res_adjusted_low = vsubq_s16(vreinterpretq_s16_u16(res_low), offset_128_16);
        let res_adjusted_high = vsubq_s16(vreinterpretq_s16_u16(res_high), offset_128_16);

        // Add prediction and residual
        let result_low = vaddq_s16(vreinterpretq_s16_u16(pred_low), res_adjusted_low);
        let result_high = vaddq_s16(vreinterpretq_s16_u16(pred_high), res_adjusted_high);

        // Saturate to unsigned 8-bit range
        let result_u8 = vcombine_u8(
            vqmovun_s16(result_low),
            vqmovun_s16(result_high)
        );

        // Store result
        vst1q_u8(output.as_mut_ptr().add(offset), result_u8);
    }

    // Handle remainder with scalar code
    for i in (chunks * 16)..len {
        let pred = y_pred[i] as i16;
        let res = residual[i] as i16 - 128;
        output[i] = (pred + res).clamp(0, 255) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-optimized residual application for x86_64 processors
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub unsafe fn apply_residual_avx2(
    y_pred: &[u8],
    residual: &[u8],
    output: &mut [u8],
    len: usize,
) {
    let chunks = len / 32;
    let remainder = len % 32;

    let offset_128 = _mm256_set1_epi16(128);
    let zero = _mm256_setzero_si256();

    for i in 0..chunks {
        let offset = i * 32;

        // Load 32 pixels
        let pred = _mm256_loadu_si256(y_pred.as_ptr().add(offset) as *const __m256i);
        let res = _mm256_loadu_si256(residual.as_ptr().add(offset) as *const __m256i);

        // Convert to 16-bit for proper signed arithmetic
        // Process low 16 bytes
        let pred_lo = _mm256_unpacklo_epi8(pred, zero);
        let res_lo = _mm256_unpacklo_epi8(res, zero);

        // Process high 16 bytes
        let pred_hi = _mm256_unpackhi_epi8(pred, zero);
        let res_hi = _mm256_unpackhi_epi8(res, zero);

        // Subtract 128 from residual (now in 16-bit, can handle negative)
        let res_adjusted_lo = _mm256_sub_epi16(res_lo, offset_128);
        let res_adjusted_hi = _mm256_sub_epi16(res_hi, offset_128);

        // Add prediction and residual
        let result_lo = _mm256_add_epi16(pred_lo, res_adjusted_lo);
        let result_hi = _mm256_add_epi16(pred_hi, res_adjusted_hi);

        // Pack back to 8-bit with unsigned saturation
        let result = _mm256_packus_epi16(result_lo, result_hi);

        // Store result
        _mm256_storeu_si256(output.as_mut_ptr().add(offset) as *mut __m256i, result);
    }

    // Handle remainder with SSE2 (16 bytes at a time)
    let sse_offset = chunks * 32;
    let sse_chunks = remainder / 16;

    if sse_chunks > 0 {
        let offset_128_sse = _mm_set1_epi16(128);
        let zero_sse = _mm_setzero_si128();
        let offset = sse_offset;

        let pred = _mm_loadu_si128(y_pred.as_ptr().add(offset) as *const __m128i);
        let res = _mm_loadu_si128(residual.as_ptr().add(offset) as *const __m128i);

        // Convert to 16-bit for proper signed arithmetic
        let pred_lo = _mm_unpacklo_epi8(pred, zero_sse);
        let res_lo = _mm_unpacklo_epi8(res, zero_sse);
        let pred_hi = _mm_unpackhi_epi8(pred, zero_sse);
        let res_hi = _mm_unpackhi_epi8(res, zero_sse);

        // Subtract 128 and add
        let res_adjusted_lo = _mm_sub_epi16(res_lo, offset_128_sse);
        let res_adjusted_hi = _mm_sub_epi16(res_hi, offset_128_sse);
        let result_lo = _mm_add_epi16(pred_lo, res_adjusted_lo);
        let result_hi = _mm_add_epi16(pred_hi, res_adjusted_hi);

        // Pack back to 8-bit with unsigned saturation
        let result = _mm_packus_epi16(result_lo, result_hi);
        _mm_storeu_si128(output.as_mut_ptr().add(offset) as *mut __m128i, result);
    }

    // Handle final remainder with scalar code
    for i in (chunks * 32 + sse_chunks * 16)..len {
        let pred = y_pred[i] as i16;
        let res = residual[i] as i16 - 128;
        output[i] = (pred + res).clamp(0, 255) as u8;
    }
}

/// SSE2-optimized residual application for older x86_64 processors
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub unsafe fn apply_residual_sse2(
    y_pred: &[u8],
    residual: &[u8],
    output: &mut [u8],
    len: usize,
) {
    let chunks = len / 16;
    let remainder = len % 16;

    let offset_128 = _mm_set1_epi16(128);
    let zero = _mm_setzero_si128();

    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 pixels
        let pred = _mm_loadu_si128(y_pred.as_ptr().add(offset) as *const __m128i);
        let res = _mm_loadu_si128(residual.as_ptr().add(offset) as *const __m128i);

        // Convert to 16-bit for proper signed arithmetic
        let pred_lo = _mm_unpacklo_epi8(pred, zero);
        let res_lo = _mm_unpacklo_epi8(res, zero);
        let pred_hi = _mm_unpackhi_epi8(pred, zero);
        let res_hi = _mm_unpackhi_epi8(res, zero);

        // Subtract 128 from residual (now in 16-bit, can handle negative)
        let res_adjusted_lo = _mm_sub_epi16(res_lo, offset_128);
        let res_adjusted_hi = _mm_sub_epi16(res_hi, offset_128);

        // Add prediction and residual
        let result_lo = _mm_add_epi16(pred_lo, res_adjusted_lo);
        let result_hi = _mm_add_epi16(pred_hi, res_adjusted_hi);

        // Pack back to 8-bit with unsigned saturation
        let result = _mm_packus_epi16(result_lo, result_hi);

        // Store result
        _mm_storeu_si128(output.as_mut_ptr().add(offset) as *mut __m128i, result);
    }

    // Handle remainder with scalar code
    for i in (chunks * 16)..len {
        let pred = y_pred[i] as i16;
        let res = residual[i] as i16 - 128;
        output[i] = (pred + res).clamp(0, 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_residual_fast() {
        let y_pred = vec![100u8; 256];
        let residual = vec![138u8; 256]; // +10 after subtracting 128
        let mut output = vec![0u8; 256];

        apply_residual_fast(&y_pred, &residual, &mut output, 256);

        assert_eq!(output[0], 110);
        assert_eq!(output[255], 110);
    }

}