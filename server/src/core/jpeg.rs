use anyhow::Result;
use std::fmt;
use std::str::FromStr;

/// Chroma subsampling mode for JPEG encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:4:4 — no chroma subsampling, best quality, largest files
    Css444,
    /// 4:2:0 — standard chroma subsampling (naive downsample), smallest files
    Css420,
    /// 4:2:0 with gradient-descent-optimized chroma planes, near-4:4:4 quality at 4:2:0 size
    Css420Opt,
}

impl Default for ChromaSubsampling {
    fn default() -> Self {
        ChromaSubsampling::Css444
    }
}

impl fmt::Display for ChromaSubsampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChromaSubsampling::Css444 => write!(f, "444"),
            ChromaSubsampling::Css420 => write!(f, "420"),
            ChromaSubsampling::Css420Opt => write!(f, "420opt"),
        }
    }
}

impl FromStr for ChromaSubsampling {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "444" => Ok(ChromaSubsampling::Css444),
            "420" => Ok(ChromaSubsampling::Css420),
            "420opt" => Ok(ChromaSubsampling::Css420Opt),
            other => Err(format!("unknown subsampling: '{}'. Use 444, 420, or 420opt", other)),
        }
    }
}

/// Trait for JPEG encoding backends
pub trait JpegEncoder: Send + Sync {
    fn encode_rgb(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>>;
    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>>;
    /// Encode RGB with a specific chroma subsampling mode.
    /// Default impl ignores subsamp and calls encode_rgb (for non-turbojpeg backends).
    fn encode_rgb_with_subsamp(
        &self, pixels: &[u8], width: u32, height: u32, quality: u8, subsamp: ChromaSubsampling,
    ) -> Result<Vec<u8>> {
        let _ = subsamp;
        self.encode_rgb(pixels, width, height, quality)
    }
    fn name(&self) -> &str;
}

/// TurboJPEG encoder backend (default, uses libjpeg API with Huffman optimization)
pub struct TurboJpegEncoder;

impl JpegEncoder for TurboJpegEncoder {
    fn encode_rgb(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        // Use libjpeg API with Huffman optimization (4:4:4)
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, false, false, true)
    }

    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        // Grayscale: subsamp is irrelevant
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, true, false, true)
    }

    fn encode_rgb_with_subsamp(
        &self, pixels: &[u8], width: u32, height: u32, quality: u8, subsamp: ChromaSubsampling,
    ) -> Result<Vec<u8>> {
        match subsamp {
            ChromaSubsampling::Css444 => {
                crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, false, false, true)
            }
            ChromaSubsampling::Css420 => {
                crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, false, true, true)
            }
            ChromaSubsampling::Css420Opt => {
                // 420opt uses turbojpeg's tjCompressFromYUVPlanes for precise chroma control
                crate::turbojpeg_optimized::encode_jpeg_turbo_420opt(pixels, width, height, quality)
            }
        }
    }

    fn name(&self) -> &str {
        "turbojpeg"
    }
}

/// mozjpeg encoder backend (better compression via trellis quantization).
/// Links against a pre-built mozjpeg static library via libjpeg62 FFI.
#[cfg(feature = "mozjpeg")]
pub struct MozJpegEncoder;

#[cfg(feature = "mozjpeg")]
impl JpegEncoder for MozJpegEncoder {
    fn encode_rgb(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        // mozjpeg does its own Huffman optimization internally
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, false, false, true)
    }

    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, true, false, true)
    }

    fn name(&self) -> &str {
        "mozjpeg"
    }
}

/// jpegli encoder backend (~35% better compression than libjpeg-turbo).
/// Links against a pre-built jpegli static library via libjpeg62 FFI.
#[cfg(feature = "jpegli")]
pub struct JpegliEncoder;

#[cfg(feature = "jpegli")]
impl JpegEncoder for JpegliEncoder {
    fn encode_rgb(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, false, false, true)
    }

    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, true, false, true)
    }

    fn name(&self) -> &str {
        "jpegli"
    }
}

/// JPEG-XL encoder backend (best compression, different codec from JPEG).
/// Can coexist with any JPEG backend since it has no symbol conflicts.
#[cfg(feature = "jpegxl")]
pub struct JpegXlEncoder;

#[cfg(feature = "jpegxl")]
impl JpegEncoder for JpegXlEncoder {
    fn encode_rgb(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        use jpegxl_rs::encode::{encoder_builder, ColorEncoding, EncoderFrame};
        let mut enc = encoder_builder()
            .jpeg_quality(quality as f32)
            .color_encoding(ColorEncoding::Srgb)
            .build()
            .map_err(|e| anyhow::anyhow!("jpegxl encoder create failed: {e:?}"))?;
        let result: jpegxl_rs::encode::EncoderResult<u8> = enc
            .encode_frame(&EncoderFrame::new(pixels).num_channels(3), width, height)
            .map_err(|e| anyhow::anyhow!("jpegxl encode_rgb failed: {e:?}"))?;
        Ok(result.data)
    }

    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        use jpegxl_rs::encode::{encoder_builder, ColorEncoding, EncoderFrame};
        let mut enc = encoder_builder()
            .jpeg_quality(quality as f32)
            .color_encoding(ColorEncoding::SrgbLuma)
            .build()
            .map_err(|e| anyhow::anyhow!("jpegxl encoder create failed: {e:?}"))?;
        let result: jpegxl_rs::encode::EncoderResult<u8> = enc
            .encode_frame(&EncoderFrame::new(pixels).num_channels(1), width, height)
            .map_err(|e| anyhow::anyhow!("jpegxl encode_gray failed: {e:?}"))?;
        Ok(result.data)
    }

    fn name(&self) -> &str {
        "jpegxl"
    }
}

/// WebP encoder backend (better than JPEG without JXL's aggressive perceptual smoothing).
/// Always available — no feature flag or external tools needed.
pub struct WebpEncoder;

impl JpegEncoder for WebpEncoder {
    fn encode_rgb(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        let encoder = webp::Encoder::from_rgb(pixels, width, height);
        let mem = encoder.encode(quality as f32);
        Ok(mem.to_vec())
    }

    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        // libwebp only accepts RGB/RGBA — expand gray to RGB (repeat each byte 3x)
        let rgb: Vec<u8> = pixels.iter().flat_map(|&g| [g, g, g]).collect();
        let encoder = webp::Encoder::from_rgb(&rgb, width, height);
        let mem = encoder.encode(quality as f32);
        Ok(mem.to_vec())
    }

    fn name(&self) -> &str {
        "webp"
    }
}

/// Create an encoder by name
pub fn create_encoder(name: &str) -> Result<Box<dyn JpegEncoder>> {
    match name {
        "turbojpeg" => Ok(Box::new(TurboJpegEncoder)),
        #[cfg(feature = "mozjpeg")]
        "mozjpeg" => Ok(Box::new(MozJpegEncoder)),
        #[cfg(feature = "jpegli")]
        "jpegli" => Ok(Box::new(JpegliEncoder)),
        #[cfg(feature = "jpegxl")]
        "jpegxl" => Ok(Box::new(JpegXlEncoder)),
        "webp" => Ok(Box::new(WebpEncoder)),
        other => {
            let mut available = String::from("turbojpeg, webp");
            if cfg!(feature = "mozjpeg") {
                available.push_str(", mozjpeg");
            }
            if cfg!(feature = "jpegli") {
                available.push_str(", jpegli");
            }
            if cfg!(feature = "jpegxl") {
                available.push_str(", jpegxl");
            }
            anyhow::bail!(
                "unknown encoder: '{}'. Available: {} \
                 (mozjpeg/jpegli/jpegxl require --features flag at build time)",
                other,
                available,
            );
        }
    }
}
