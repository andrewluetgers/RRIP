// OPTIMIZATION 1: Fixed-size buffer pools to eliminate allocation overhead
use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

pub struct FixedBufferPool {
    buffers: ArrayQueue<Vec<u8>>,
    size: usize,
}

impl FixedBufferPool {
    pub fn new(capacity: usize, buffer_size: usize) -> Arc<Self> {
        let buffers = ArrayQueue::new(capacity);

        // Pre-allocate all buffers
        for _ in 0..capacity {
            let _ = buffers.push(vec![0u8; buffer_size]);
        }

        Arc::new(Self {
            buffers,
            size: buffer_size,
        })
    }

    #[inline(always)]
    pub fn get(&self) -> Vec<u8> {
        self.buffers.pop().unwrap_or_else(|| vec![0u8; self.size])
    }

    #[inline(always)]
    pub fn put(&self, mut buf: Vec<u8>) {
        if buf.capacity() == self.size {
            buf.clear();
            buf.resize(self.size, 0);
            let _ = self.buffers.push(buf);
        }
    }
}

// OPTIMIZATION 2: Tiered buffer pools for different sizes
pub struct TieredBufferPools {
    pub tile_256: Arc<FixedBufferPool>,   // 256x256x3 = 196KB per buffer
    pub tile_512: Arc<FixedBufferPool>,   // 512x512x3 = 768KB per buffer
    pub tile_1024: Arc<FixedBufferPool>,  // 1024x1024x3 = 3MB per buffer
    pub residual: Arc<FixedBufferPool>,   // 256x256 = 64KB per buffer
    pub planes: Arc<FixedBufferPool>,     // For YCbCr planes
}

impl TieredBufferPools {
    pub fn new(tile_size: u32) -> Self {
        let tile_rgb_size = (tile_size * tile_size * 3) as usize;
        let l1_size = (tile_size * 2 * tile_size * 2 * 3) as usize;
        let l0_size = (tile_size * 4 * tile_size * 4 * 3) as usize;
        let residual_size = (tile_size * tile_size) as usize;

        Self {
            tile_256: FixedBufferPool::new(128, tile_rgb_size),
            tile_512: FixedBufferPool::new(64, l1_size),
            tile_1024: FixedBufferPool::new(32, l0_size),
            residual: FixedBufferPool::new(128, residual_size),
            planes: FixedBufferPool::new(96, l0_size / 3), // Max plane size
        }
    }

    pub fn get_for_size(&self, size: usize) -> Vec<u8> {
        match size {
            0..=200_000 => self.tile_256.get(),
            200_001..=800_000 => self.tile_512.get(),
            _ => self.tile_1024.get(),
        }
    }

    pub fn put_for_size(&self, buf: Vec<u8>) {
        match buf.capacity() {
            0..=200_000 => self.tile_256.put(buf),
            200_001..=800_000 => self.tile_512.put(buf),
            _ => self.tile_1024.put(buf),
        }
    }
}

// OPTIMIZATION 3: TurboJPEG pool for zero-copy operations
use turbojpeg::{Compressor, Decompressor, PixelFormat};

pub struct TurboJpegPool {
    compressors: ArrayQueue<Compressor>,
    decompressors: ArrayQueue<Decompressor>,
}

impl TurboJpegPool {
    pub fn new(size: usize) -> Arc<Self> {
        let compressors = ArrayQueue::new(size);
        let decompressors = ArrayQueue::new(size);

        // Pre-create instances
        for _ in 0..size/2 {
            let _ = compressors.push(Compressor::new().unwrap());
            let _ = decompressors.push(Decompressor::new().unwrap());
        }

        Arc::new(Self {
            compressors,
            decompressors,
        })
    }

    #[inline(always)]
    pub fn with_compressor<F, R>(&self, quality: i32, f: F) -> anyhow::Result<R>
    where
        F: FnOnce(&mut Compressor) -> anyhow::Result<R>,
    {
        let mut comp = self.compressors.pop()
            .unwrap_or_else(|| Compressor::new().unwrap());
        comp.set_quality(quality);
        let result = f(&mut comp);
        let _ = self.compressors.push(comp);
        result
    }

    #[inline(always)]
    pub fn with_decompressor<F, R>(&self, f: F) -> anyhow::Result<R>
    where
        F: FnOnce(&mut Decompressor) -> anyhow::Result<R>,
    {
        let mut decomp = self.decompressors.pop()
            .unwrap_or_else(|| Decompressor::new().unwrap());
        let result = f(&mut decomp);
        let _ = self.decompressors.push(decomp);
        result
    }

    // Helper method for JPEG compression
    pub fn compress_rgb(&self, rgb_data: &[u8], width: usize, height: usize, quality: i32) -> anyhow::Result<Vec<u8>> {
        self.with_compressor(quality, |comp| {
            let image = turbojpeg::Image {
                pixels: rgb_data,
                width,
                pitch: width * 3,
                height,
                format: PixelFormat::RGB,
            };
            Ok(comp.compress_to_vec(image)?)
        })
    }

    // Helper method for JPEG decompression
    pub fn decompress_rgb(&self, jpeg_bytes: &[u8], output: &mut [u8], width: usize, height: usize) -> anyhow::Result<()> {
        self.with_decompressor(|decomp| {
            let output_image = turbojpeg::Image {
                pixels: output,
                width,
                pitch: width * 3,
                height,
                format: PixelFormat::RGB,
            };
            decomp.decompress(jpeg_bytes, output_image)?;
            Ok(())
        })
    }
}