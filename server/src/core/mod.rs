pub mod color;
pub mod residual;
pub mod pack;
pub mod upsample;
pub mod jpeg;
pub mod pyramid;
pub mod reconstruct;
pub mod optimize_chroma;
pub mod optimize_l2;
pub mod sharpen;

pub mod libjpeg_ffi;

use std::fmt;
use std::str::FromStr;

/// Resampling filter for up/downscale operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleFilter {
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl ResampleFilter {
    /// Convert to the corresponding `image` crate filter type.
    pub fn to_image_filter(self) -> image::imageops::FilterType {
        match self {
            ResampleFilter::Bilinear => image::imageops::FilterType::Triangle,
            ResampleFilter::Bicubic => image::imageops::FilterType::CatmullRom,
            ResampleFilter::Lanczos3 => image::imageops::FilterType::Lanczos3,
        }
    }
}

impl fmt::Display for ResampleFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResampleFilter::Bilinear => write!(f, "bilinear"),
            ResampleFilter::Bicubic => write!(f, "bicubic"),
            ResampleFilter::Lanczos3 => write!(f, "lanczos3"),
        }
    }
}

impl FromStr for ResampleFilter {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bilinear" | "triangle" => Ok(ResampleFilter::Bilinear),
            "bicubic" | "catmullrom" | "catmull-rom" => Ok(ResampleFilter::Bicubic),
            "lanczos3" | "lanczos" => Ok(ResampleFilter::Lanczos3),
            _ => Err(format!("unknown resample filter '{}'. Available: bilinear, bicubic, lanczos3", s)),
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated resize helpers (via fast_image_resize)
// ---------------------------------------------------------------------------

/// Resize a single-channel (grayscale) plane using SIMD-accelerated fast_image_resize.
pub fn fir_resize(plane: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32, filter: ResampleFilter) -> Vec<u8> {
    use fast_image_resize as fir;
    let src = fir::images::Image::from_vec_u8(src_w, src_h, plane.to_vec(), fir::pixels::PixelType::U8)
        .expect("failed to create fir source image");
    let mut dst = fir::images::Image::new(dst_w, dst_h, fir::pixels::PixelType::U8);
    let alg = match filter {
        ResampleFilter::Bilinear => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        ResampleFilter::Bicubic => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
        ResampleFilter::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
    };
    let mut resizer = fir::Resizer::new();
    resizer.resize(&src, &mut dst, &fir::ResizeOptions::new().resize_alg(alg)).unwrap();
    dst.into_vec()
}

/// 2x upsample of a single-channel plane using SIMD-accelerated resize.
pub fn upsample_2x(plane: &[u8], w: usize, h: usize, filter: ResampleFilter) -> Vec<u8> {
    fir_resize(plane, w as u32, h as u32, (w * 2) as u32, (h * 2) as u32, filter)
}

/// 4x upsample of a single-channel plane using SIMD-accelerated resize (direct, no intermediate).
pub fn upsample_4x(plane: &[u8], w: usize, h: usize, filter: ResampleFilter) -> Vec<u8> {
    fir_resize(plane, w as u32, h as u32, (w * 4) as u32, (h * 4) as u32, filter)
}
