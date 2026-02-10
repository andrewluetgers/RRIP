use anyhow::Result;

/// Trait for JPEG encoding backends
pub trait JpegEncoder: Send + Sync {
    fn encode_rgb(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>>;
    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>>;
    fn name(&self) -> &str;
}

/// TurboJPEG encoder backend (default, fastest)
pub struct TurboJpegEncoder;

impl JpegEncoder for TurboJpegEncoder {
    fn encode_rgb(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        crate::turbojpeg_optimized::encode_jpeg_turbo(pixels, width, height, quality)
    }

    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        crate::turbojpeg_optimized::encode_luma_turbo(pixels, width, height, quality)
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
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, false)
    }

    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, true)
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
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, false)
    }

    fn encode_gray(&self, pixels: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
        crate::core::libjpeg_ffi::encode_libjpeg(pixels, width, height, quality, true)
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
        other => {
            let mut available = String::from("turbojpeg");
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
