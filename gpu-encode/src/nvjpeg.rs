//! Native nvJPEG wrapper for GPU-accelerated JPEG encode/decode.
//!
//! All JPEG operations happen on GPU. Compressed bytes live on host,
//! decoded/encoded pixel data lives on GPU device memory.

use std::ffi::c_void;
use std::ptr;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use tracing::info;

use crate::kernels::GpuContext;

// ---------------------------------------------------------------------------
// nvJPEG C API types (manual FFI bindings)
// ---------------------------------------------------------------------------

// Opaque handles
type NvjpegHandle = *mut c_void;
type NvjpegJpegState = *mut c_void;
type NvjpegEncoderState = *mut c_void;
type NvjpegEncoderParams = *mut c_void;

// CUDA types matching cudarc's sys definitions
type CudaStream_t = *mut c_void;

type NvjpegStatus = i32;

const NVJPEG_STATUS_SUCCESS: NvjpegStatus = 0;

// nvjpegOutputFormat_t
const NVJPEG_OUTPUT_RGBI: i32 = 5; // interleaved RGB
#[allow(dead_code)]
const NVJPEG_OUTPUT_Y: i32 = 2;    // luma only

// nvjpegInputFormat_t
const NVJPEG_INPUT_RGBI: i32 = 5;  // interleaved RGB

// nvjpegChromaSubsampling_t
const NVJPEG_CSS_444: i32 = 0;
const NVJPEG_CSS_420: i32 = 2;
const NVJPEG_CSS_GRAY: i32 = 6;

/// nvjpegImage_t — channel pointers + pitch for up to 4 planes.
/// For interleaved RGB: channel[0] = rgb_ptr, pitch[0] = width * 3.
#[repr(C)]
#[derive(Clone)]
struct NvjpegImage {
    channel: [*mut u8; 4],
    pitch: [usize; 4],  // size_t in C
}

impl Default for NvjpegImage {
    fn default() -> Self {
        NvjpegImage {
            channel: [ptr::null_mut(); 4],
            pitch: [0; 4],
        }
    }
}

#[link(name = "nvjpeg")]
extern "C" {
    fn nvjpegCreateSimple(handle: *mut NvjpegHandle) -> NvjpegStatus;
    fn nvjpegDestroy(handle: NvjpegHandle) -> NvjpegStatus;

    fn nvjpegJpegStateCreate(
        handle: NvjpegHandle,
        state: *mut NvjpegJpegState,
    ) -> NvjpegStatus;
    fn nvjpegJpegStateDestroy(state: NvjpegJpegState) -> NvjpegStatus;

    fn nvjpegGetImageInfo(
        handle: NvjpegHandle,
        data: *const u8,
        length: usize,
        nComponents: *mut i32,
        subsampling: *mut i32,
        widths: *mut i32,
        heights: *mut i32,
    ) -> NvjpegStatus;

    // Batched decode API
    fn nvjpegDecodeBatchedInitialize(
        handle: NvjpegHandle,
        state: NvjpegJpegState,
        batch_size: i32,
        max_cpu_threads: i32,
        output_format: i32,
    ) -> NvjpegStatus;

    fn nvjpegDecodeBatched(
        handle: NvjpegHandle,
        state: NvjpegJpegState,
        data: *const *const u8,
        lengths: *const usize,
        destinations: *mut NvjpegImage,
        stream: CudaStream_t,
    ) -> NvjpegStatus;

    // Single-image decode (simpler for small batches)
    fn nvjpegDecode(
        handle: NvjpegHandle,
        state: NvjpegJpegState,
        data: *const u8,
        length: usize,
        output_format: i32,
        destination: *mut NvjpegImage,
        stream: CudaStream_t,
    ) -> NvjpegStatus;

    // Encoder API
    fn nvjpegEncoderStateCreate(
        handle: NvjpegHandle,
        state: *mut NvjpegEncoderState,
        stream: CudaStream_t,
    ) -> NvjpegStatus;
    fn nvjpegEncoderStateDestroy(state: NvjpegEncoderState) -> NvjpegStatus;

    fn nvjpegEncoderParamsCreate(
        handle: NvjpegHandle,
        params: *mut NvjpegEncoderParams,
        stream: CudaStream_t,
    ) -> NvjpegStatus;
    fn nvjpegEncoderParamsDestroy(params: NvjpegEncoderParams) -> NvjpegStatus;

    fn nvjpegEncoderParamsSetQuality(
        params: NvjpegEncoderParams,
        quality: i32,
        stream: CudaStream_t,
    ) -> NvjpegStatus;

    fn nvjpegEncoderParamsSetSamplingFactors(
        params: NvjpegEncoderParams,
        chroma_subsampling: i32,
        stream: CudaStream_t,
    ) -> NvjpegStatus;

    fn nvjpegEncoderParamsSetOptimizedHuffman(
        params: NvjpegEncoderParams,
        optimized: i32,
        stream: CudaStream_t,
    ) -> NvjpegStatus;

    fn nvjpegEncodeImage(
        handle: NvjpegHandle,
        state: NvjpegEncoderState,
        params: NvjpegEncoderParams,
        source: *const NvjpegImage,
        input_format: i32,
        width: i32,
        height: i32,
        stream: CudaStream_t,
    ) -> NvjpegStatus;

    fn nvjpegEncodeYUV(
        handle: NvjpegHandle,
        state: NvjpegEncoderState,
        params: NvjpegEncoderParams,
        source: *const NvjpegImage,
        chroma_subsampling: i32,
        width: i32,
        height: i32,
        stream: CudaStream_t,
    ) -> NvjpegStatus;

    fn nvjpegEncodeRetrieveBitstream(
        handle: NvjpegHandle,
        state: NvjpegEncoderState,
        data: *mut u8,
        length: *mut usize,
        stream: CudaStream_t,
    ) -> NvjpegStatus;
}

fn check_nvjpeg(status: NvjpegStatus, msg: &str) -> Result<()> {
    if status != NVJPEG_STATUS_SUCCESS {
        Err(anyhow!("nvJPEG error {}: {}", status, msg))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Safe wrapper
// ---------------------------------------------------------------------------

/// Native nvJPEG handle for GPU JPEG encode/decode.
pub struct NvJpegHandle {
    handle: NvjpegHandle,
    decode_state: NvjpegJpegState,
    encoder_state: NvjpegEncoderState,
    encoder_params: NvjpegEncoderParams,
    stream_ptr: CudaStream_t,
}

// nvJPEG handles are thread-safe when using separate states per thread
unsafe impl Send for NvJpegHandle {}

impl NvJpegHandle {
    /// Create a new nvJPEG handle on the given GPU context.
    pub fn new(gpu: &GpuContext) -> Result<Self> {
        let stream_ptr = gpu.stream.cu_stream() as CudaStream_t;

        let mut handle: NvjpegHandle = ptr::null_mut();
        unsafe {
            check_nvjpeg(nvjpegCreateSimple(&mut handle), "nvjpegCreateSimple")?;
        }

        let mut decode_state: NvjpegJpegState = ptr::null_mut();
        unsafe {
            check_nvjpeg(
                nvjpegJpegStateCreate(handle, &mut decode_state),
                "nvjpegJpegStateCreate",
            )?;
        }

        let mut encoder_state: NvjpegEncoderState = ptr::null_mut();
        unsafe {
            check_nvjpeg(
                nvjpegEncoderStateCreate(handle, &mut encoder_state, stream_ptr),
                "nvjpegEncoderStateCreate",
            )?;
        }

        let mut encoder_params: NvjpegEncoderParams = ptr::null_mut();
        unsafe {
            check_nvjpeg(
                nvjpegEncoderParamsCreate(handle, &mut encoder_params, stream_ptr),
                "nvjpegEncoderParamsCreate",
            )?;
        }

        info!("nvJPEG initialized (native GPU JPEG)");

        Ok(NvJpegHandle {
            handle,
            decode_state,
            encoder_state,
            encoder_params,
            stream_ptr,
        })
    }

    /// Decode a single JPEG from host bytes into GPU device memory (interleaved RGB u8).
    ///
    /// Returns a CudaSlice<u8> of size width * height * 3 on device, plus dimensions.
    pub fn decode_to_device(
        &self,
        gpu: &GpuContext,
        jpeg_data: &[u8],
    ) -> Result<(CudaSlice<u8>, u32, u32)> {
        if jpeg_data.is_empty() {
            return Err(anyhow!("empty JPEG data"));
        }

        // Query image dimensions
        let mut n_components: i32 = 0;
        let mut subsampling: i32 = 0;
        let mut widths = [0i32; 4];
        let mut heights = [0i32; 4];
        unsafe {
            check_nvjpeg(
                nvjpegGetImageInfo(
                    self.handle,
                    jpeg_data.as_ptr(),
                    jpeg_data.len(),
                    &mut n_components,
                    &mut subsampling,
                    widths.as_mut_ptr(),
                    heights.as_mut_ptr(),
                ),
                "nvjpegGetImageInfo",
            )?;
        }

        let w = widths[0] as u32;
        let h = heights[0] as u32;
        let size = (w * h * 3) as usize;

        // Allocate device buffer for decoded RGB
        let mut dst = unsafe { gpu.stream.alloc::<u8>(size) }
            .map_err(|e| anyhow!("alloc decoded RGB failed: {}", e))?;

        // Set up nvjpegImage_t pointing to device memory
        let (dev_ptr, _write_guard) = dst.device_ptr_mut(&gpu.stream);
        let mut nv_image = NvjpegImage::default();
        nv_image.channel[0] = dev_ptr as *mut u8;
        nv_image.pitch[0] = (w * 3) as usize;

        unsafe {
            check_nvjpeg(
                nvjpegDecode(
                    self.handle,
                    self.decode_state,
                    jpeg_data.as_ptr(),
                    jpeg_data.len(),
                    NVJPEG_OUTPUT_RGBI,
                    &mut nv_image,
                    self.stream_ptr,
                ),
                "nvjpegDecode",
            )?;
        }
        drop(_write_guard);

        Ok((dst, w, h))
    }

    /// Batch decode multiple JPEGs from host bytes into GPU device memory.
    ///
    /// Returns Vec of (CudaSlice<u8>, width, height) for each decoded image.
    pub fn batch_decode_to_device(
        &self,
        gpu: &GpuContext,
        jpeg_data: &[Vec<u8>],
    ) -> Result<Vec<(CudaSlice<u8>, u32, u32)>> {
        let mut results = Vec::with_capacity(jpeg_data.len());
        for (i, data) in jpeg_data.iter().enumerate() {
            if data.is_empty() {
                // Allocate a zero-filled placeholder
                let placeholder = gpu.stream.alloc_zeros::<u8>(1)
                    .map_err(|e| anyhow!("alloc placeholder failed: {}", e))?;
                results.push((placeholder, 0, 0));
                continue;
            }
            let (slice, w, h) = self.decode_to_device(gpu, data)
                .with_context(|| format!("decoding JPEG tile {}", i))?;
            results.push((slice, w, h));
        }
        Ok(results)
    }

    /// Decode a grayscale JPEG to GPU device memory (single channel u8).
    ///
    /// nvJPEG decodes to RGB interleaved even for grayscale, so we decode
    /// to RGBI and then just take every 3rd byte (R channel = G = B for gray).
    /// For efficiency we decode to full RGB and extract in a kernel, but for now
    /// we decode and use the existing data (all 3 channels are identical for gray input).
    pub fn decode_gray_to_device(
        &self,
        gpu: &GpuContext,
        jpeg_data: &[u8],
    ) -> Result<(CudaSlice<u8>, u32, u32)> {
        // Decode as RGB, then extract Y via the existing kernel or just use R channel
        let (rgb, w, h) = self.decode_to_device(gpu, jpeg_data)?;
        // For grayscale JPEG, R=G=B, so the RGB buffer has the gray value at every 3rd byte.
        // Extract single channel by taking every 3rd byte via a simple kernel.
        // We'll use rgb_to_ycbcr to get Y, but for pure gray input R=G=B so Y=R.
        // Simpler: just copy stride-3 on CPU after download... but that defeats the purpose.
        // Instead, use the rgb_to_ycbcr kernel and take Y output.
        let pixels = (w * h) as i32;
        let (y, _cb, _cr) = gpu.rgb_to_ycbcr_f32(&rgb, pixels)?;
        // Convert Y f32 back to u8
        let y_u8 = gpu.f32_to_u8(&y, pixels)?;
        Ok((y_u8, w, h))
    }

    /// Encode RGB u8 device buffer to JPEG, returning compressed bytes on host.
    pub fn encode_rgb(
        &self,
        gpu: &GpuContext,
        rgb_dev: &CudaSlice<u8>,
        width: u32,
        height: u32,
        quality: u8,
        subsamp: &str,
    ) -> Result<Vec<u8>> {
        let css = match subsamp {
            "420" => NVJPEG_CSS_420,
            _ => NVJPEG_CSS_444,
        };

        unsafe {
            check_nvjpeg(
                nvjpegEncoderParamsSetQuality(self.encoder_params, quality as i32, self.stream_ptr),
                "set quality",
            )?;
            check_nvjpeg(
                nvjpegEncoderParamsSetSamplingFactors(self.encoder_params, css, self.stream_ptr),
                "set subsampling",
            )?;
            check_nvjpeg(
                nvjpegEncoderParamsSetOptimizedHuffman(self.encoder_params, 1, self.stream_ptr),
                "set optimized huffman",
            )?;
        }

        let (dev_ptr, _read_guard) = rgb_dev.device_ptr(&gpu.stream);
        let mut nv_image = NvjpegImage::default();
        nv_image.channel[0] = dev_ptr as *mut u8;
        nv_image.pitch[0] = (width * 3) as usize;

        unsafe {
            check_nvjpeg(
                nvjpegEncodeImage(
                    self.handle,
                    self.encoder_state,
                    self.encoder_params,
                    &nv_image,
                    NVJPEG_INPUT_RGBI,
                    width as i32,
                    height as i32,
                    self.stream_ptr,
                ),
                "nvjpegEncodeImage RGB",
            )?;
        }
        drop(_read_guard);

        // Retrieve bitstream size
        let mut length: usize = 0;
        unsafe {
            check_nvjpeg(
                nvjpegEncodeRetrieveBitstream(
                    self.handle,
                    self.encoder_state,
                    ptr::null_mut(),
                    &mut length,
                    self.stream_ptr,
                ),
                "nvjpegEncodeRetrieveBitstream (size)",
            )?;
        }

        // Retrieve bitstream data
        let mut jpeg_bytes = vec![0u8; length];
        unsafe {
            check_nvjpeg(
                nvjpegEncodeRetrieveBitstream(
                    self.handle,
                    self.encoder_state,
                    jpeg_bytes.as_mut_ptr(),
                    &mut length,
                    self.stream_ptr,
                ),
                "nvjpegEncodeRetrieveBitstream (data)",
            )?;
        }
        jpeg_bytes.truncate(length);

        Ok(jpeg_bytes)
    }

    /// Encode grayscale u8 device buffer to JPEG, returning compressed bytes on host.
    ///
    /// Expands gray to RGB on GPU (via gray_to_rgbi kernel), then encodes
    /// with nvjpegEncodeImage + NVJPEG_CSS_GRAY to produce a grayscale JPEG.
    pub fn encode_gray(
        &self,
        gpu: &GpuContext,
        gray_dev: &CudaSlice<u8>,
        width: u32,
        height: u32,
        quality: u8,
    ) -> Result<Vec<u8>> {
        // Expand gray → interleaved RGB on GPU (R=G=B=gray)
        let pixels = (width * height) as i32;
        let rgb_dev = gpu.gray_to_rgbi(gray_dev, pixels)?;

        // Encode as JPEG with GRAY subsampling (nvJPEG extracts Y from RGB)
        unsafe {
            check_nvjpeg(
                nvjpegEncoderParamsSetQuality(self.encoder_params, quality as i32, self.stream_ptr),
                "set quality",
            )?;
            check_nvjpeg(
                nvjpegEncoderParamsSetSamplingFactors(self.encoder_params, NVJPEG_CSS_GRAY, self.stream_ptr),
                "set gray subsampling",
            )?;
            check_nvjpeg(
                nvjpegEncoderParamsSetOptimizedHuffman(self.encoder_params, 1, self.stream_ptr),
                "set optimized huffman",
            )?;
        }

        let (dev_ptr, _read_guard) = rgb_dev.device_ptr(&gpu.stream);
        let mut nv_image = NvjpegImage::default();
        nv_image.channel[0] = dev_ptr as *mut u8;
        nv_image.pitch[0] = (width * 3) as usize;

        unsafe {
            check_nvjpeg(
                nvjpegEncodeImage(
                    self.handle,
                    self.encoder_state,
                    self.encoder_params,
                    &nv_image,
                    NVJPEG_INPUT_RGBI,
                    width as i32,
                    height as i32,
                    self.stream_ptr,
                ),
                "nvjpegEncodeImage gray",
            )?;
        }
        drop(_read_guard);

        let mut length: usize = 0;
        unsafe {
            check_nvjpeg(
                nvjpegEncodeRetrieveBitstream(
                    self.handle,
                    self.encoder_state,
                    ptr::null_mut(),
                    &mut length,
                    self.stream_ptr,
                ),
                "nvjpegEncodeRetrieveBitstream gray (size)",
            )?;
        }

        let mut jpeg_bytes = vec![0u8; length];
        unsafe {
            check_nvjpeg(
                nvjpegEncodeRetrieveBitstream(
                    self.handle,
                    self.encoder_state,
                    jpeg_bytes.as_mut_ptr(),
                    &mut length,
                    self.stream_ptr,
                ),
                "nvjpegEncodeRetrieveBitstream gray (data)",
            )?;
        }
        jpeg_bytes.truncate(length);

        Ok(jpeg_bytes)
    }

    /// Decode RGB JPEG to device, then get host pixels (for the rare CPU operations like WebP).
    pub fn decode_rgb_to_host(
        &self,
        gpu: &GpuContext,
        jpeg_data: &[u8],
    ) -> Result<(Vec<u8>, u32, u32)> {
        let (rgb_dev, w, h) = self.decode_to_device(gpu, jpeg_data)?;
        gpu.sync()?;
        let rgb_host: Vec<u8> = gpu.stream.clone_dtoh(&rgb_dev)
            .map_err(|e| anyhow!("dtoh failed: {}", e))?;
        Ok((rgb_host, w, h))
    }
}

impl Drop for NvJpegHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.encoder_params.is_null() {
                nvjpegEncoderParamsDestroy(self.encoder_params);
            }
            if !self.encoder_state.is_null() {
                nvjpegEncoderStateDestroy(self.encoder_state);
            }
            if !self.decode_state.is_null() {
                nvjpegJpegStateDestroy(self.decode_state);
            }
            if !self.handle.is_null() {
                nvjpegDestroy(self.handle);
            }
        }
    }
}
