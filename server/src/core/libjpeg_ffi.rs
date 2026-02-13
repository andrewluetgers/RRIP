//! Minimal FFI bindings to the standard libjpeg62 compress API.
//!
//! This module wraps `libjpeg_compress.c` (compiled by build.rs via the `cc`
//! crate). The C wrapper is always compiled against libjpeg-turbo's libjpeg
//! API (provided by turbojpeg-sys). When mozjpeg or jpegli features are
//! enabled, the wrapper is instead linked against those libraries.

use anyhow::{bail, Result};
use libc::{c_int, c_uchar, c_ulong};

extern "C" {
    /// Compress raw pixels to JPEG using the libjpeg62 API.
    /// Returns 0 on success, non-zero on error.
    /// On success, `*out_buf` and `*out_size` are set to the JPEG data
    /// (caller must free `*out_buf` with `libc::free`).
    fn libjpeg_compress(
        pixels: *const c_uchar,
        width: c_int,
        height: c_int,
        components: c_int,
        quality: c_int,
        subsamp: c_int,
        optimize: c_int,
        out_buf: *mut *mut c_uchar,
        out_size: *mut c_ulong,
    ) -> c_int;
}

/// Encode raw pixels to JPEG using the linked libjpeg62-compatible library.
///
/// - `pixels`: row-major pixel data (RGB or grayscale)
/// - `width`, `height`: image dimensions
/// - `quality`: JPEG quality 1-100
/// - `grayscale`: if true, input is 1-byte-per-pixel grayscale
/// - `subsamp_420`: if true, use 4:2:0 chroma subsampling; if false, 4:4:4
/// - `optimize`: if true, optimize Huffman tables for smaller files
pub fn encode_libjpeg(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    grayscale: bool,
    subsamp_420: bool,
    optimize: bool,
) -> Result<Vec<u8>> {
    let components: u32 = if grayscale { 1 } else { 3 };
    let expected = (width * height * components) as usize;
    if pixels.len() < expected {
        bail!(
            "pixel buffer too small: got {} bytes, expected {} ({}x{}x{})",
            pixels.len(),
            expected,
            width,
            height,
            components
        );
    }

    unsafe {
        let mut out_buf: *mut c_uchar = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;

        let rc = libjpeg_compress(
            pixels.as_ptr(),
            width as c_int,
            height as c_int,
            components as c_int,
            quality as c_int,
            if subsamp_420 { 1 } else { 0 },
            if optimize { 1 } else { 0 },
            &mut out_buf,
            &mut out_size,
        );

        if rc != 0 || out_buf.is_null() {
            bail!("libjpeg compress failed (rc={})", rc);
        }

        // Copy to owned Vec, then free the C buffer
        let result = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        libc::free(out_buf as *mut libc::c_void);

        Ok(result)
    }
}
