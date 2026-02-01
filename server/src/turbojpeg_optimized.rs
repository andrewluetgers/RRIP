// Optimized TurboJPEG operations for fast JPEG encoding/decoding
// Replaces slow image crate operations with direct TurboJPEG calls

use anyhow::{Result, Context};
use turbojpeg::{Compressor, Decompressor, Image, PixelFormat, Subsamp};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use parking_lot::Mutex;

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

        // Use 4:2:0 subsampling for better compression
        compressor.set_subsamp(Subsamp::Sub2x2);

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

/// Fast grayscale JPEG encoding
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

/// Optimized residual application with SIMD support
#[inline(always)]
pub fn apply_residual_fast(
    y_pred: &[u8],
    residual: &[u8],
    output: &mut [u8],
    len: usize,
) {
    // Use chunks for better auto-vectorization
    let chunks = len / 8;
    let remainder = len % 8;

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

/// Pool of reusable buffers to reduce allocations
pub struct BufferPool {
    pools: Vec<Arc<Mutex<Vec<Vec<u8>>>>>,
    sizes: Vec<usize>,
}

impl BufferPool {
    pub fn new() -> Self {
        // Common buffer sizes for tiles
        let sizes = vec![
            256 * 256,       // Grayscale tile
            256 * 256 * 3,   // RGB tile
            512 * 512,       // L1 grayscale mosaic
            512 * 512 * 3,   // L1 RGB mosaic
            1024 * 1024,     // L0 grayscale mosaic
            1024 * 1024 * 3, // L0 RGB mosaic
        ];

        let pools = sizes.iter()
            .map(|_| Arc::new(Mutex::new(Vec::with_capacity(32))))
            .collect();

        BufferPool { pools, sizes }
    }

    pub fn get(&self, size: usize) -> Vec<u8> {
        // Find the appropriate pool
        for (i, &pool_size) in self.sizes.iter().enumerate() {
            if size <= pool_size {
                if let Some(mut buf) = self.pools[i].lock().pop() {
                    buf.resize(size, 0);
                    return buf;
                }
                return vec![0u8; size];
            }
        }

        // Fallback for unusual sizes
        vec![0u8; size]
    }

    pub fn put(&self, mut buf: Vec<u8>) {
        let capacity = buf.capacity();

        // Find matching pool
        for (i, &pool_size) in self.sizes.iter().enumerate() {
            if capacity <= pool_size * 2 {  // Allow some overhead
                buf.clear();
                let mut pool = self.pools[i].lock();
                if pool.len() < 32 {  // Limit pool size
                    pool.push(buf);
                }
                return;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-optimized residual application for ARM processors
#[cfg(target_arch = "aarch64")]
pub unsafe fn apply_residual_neon(
    y_pred: &[u8],
    residual: &[u8],
    output: &mut [u8],
    len: usize,
) {
    let chunks = len / 16;
    let remainder = len % 16;

    let offset_128 = vdupq_n_u8(128);

    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 pixels
        let pred = vld1q_u8(y_pred.as_ptr().add(offset));
        let res = vld1q_u8(residual.as_ptr().add(offset));

        // Subtract 128 from residual
        let res_adjusted = vqsubq_u8(res, offset_128);

        // Add prediction and residual with saturation
        let result = vqaddq_u8(pred, res_adjusted);

        // Store result
        vst1q_u8(output.as_mut_ptr().add(offset), result);
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

    let offset_128 = _mm256_set1_epi8(128i8);

    for i in 0..chunks {
        let offset = i * 32;

        // Load 32 pixels
        let pred = _mm256_loadu_si256(y_pred.as_ptr().add(offset) as *const __m256i);
        let res = _mm256_loadu_si256(residual.as_ptr().add(offset) as *const __m256i);

        // Subtract 128 from residual using saturated subtraction
        let res_adjusted = _mm256_subs_epu8(res, offset_128 as __m256i);

        // Add prediction and residual with saturation
        let result = _mm256_adds_epu8(pred, res_adjusted);

        // Store result
        _mm256_storeu_si256(output.as_mut_ptr().add(offset) as *mut __m256i, result);
    }

    // Handle remainder with SSE2 (16 bytes at a time)
    let sse_offset = chunks * 32;
    let sse_chunks = remainder / 16;

    if sse_chunks > 0 {
        let offset_128_sse = _mm_set1_epi8(128i8);
        let offset = sse_offset;

        let pred = _mm_loadu_si128(y_pred.as_ptr().add(offset) as *const __m128i);
        let res = _mm_loadu_si128(residual.as_ptr().add(offset) as *const __m128i);
        let res_adjusted = _mm_subs_epu8(res, offset_128_sse as __m128i);
        let result = _mm_adds_epu8(pred, res_adjusted);
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

    let offset_128 = _mm_set1_epi8(128i8);

    for i in 0..chunks {
        let offset = i * 16;

        // Load 16 pixels
        let pred = _mm_loadu_si128(y_pred.as_ptr().add(offset) as *const __m128i);
        let res = _mm_loadu_si128(residual.as_ptr().add(offset) as *const __m128i);

        // Subtract 128 from residual using saturated subtraction
        let res_adjusted = _mm_subs_epu8(res, offset_128 as __m128i);

        // Add prediction and residual with saturation
        let result = _mm_adds_epu8(pred, res_adjusted);

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

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new();

        let buf1 = pool.get(256 * 256);
        assert_eq!(buf1.len(), 256 * 256);

        pool.put(buf1);

        let buf2 = pool.get(256 * 256);
        assert_eq!(buf2.len(), 256 * 256);
    }
}