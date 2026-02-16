//! DICOM WSI reader.
//!
//! Uses the `dicom` crate to parse DICOM metadata and extract JPEG tile
//! bytes from the encapsulated pixel data fragments. Each fragment is one
//! tile's JPEG data, ready for nvJPEG decode.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use tracing::info;

/// A parsed DICOM WSI with pre-extracted tile JPEG bytes.
pub struct DicomSlide {
    /// Full image width in pixels (TotalPixelMatrixColumns or inferred)
    width: u32,
    /// Full image height in pixels (TotalPixelMatrixRows or inferred)
    height: u32,
    /// Per-frame tile width in pixels (COLUMNS tag)
    tile_w: u32,
    /// Per-frame tile height in pixels (ROWS tag)
    tile_h: u32,
    /// Number of tile columns
    tiles_x: u32,
    /// Number of tile rows
    tiles_y: u32,
    /// Actual JPEG bytes per frame, indexed as tile_fragments[row * tiles_x + col]
    tile_fragments: Vec<Vec<u8>>,
}

impl DicomSlide {
    /// Open a DICOM WSI file and parse tile layout.
    pub fn open(path: &Path) -> Result<Self> {
        info!("Opening DICOM: {}", path.display());

        // Parse DICOM metadata
        let obj = dicom::object::open_file(path)
            .map_err(|e| anyhow!("DICOM parse error: {}", e))?;

        // Per-frame tile dimensions (COLUMNS/ROWS = per-frame in DICOM WSI)
        let tile_w = get_u32_tag(&obj, dicom::dictionary_std::tags::COLUMNS)
            .context("Missing Columns tag")?;
        let tile_h = get_u32_tag(&obj, dicom::dictionary_std::tags::ROWS)
            .context("Missing Rows tag")?;

        // Full image dimensions from TotalPixelMatrix tags (DICOM WSI standard)
        let total_w = get_u32_tag(&obj, dicom::dictionary_std::tags::TOTAL_PIXEL_MATRIX_COLUMNS);
        let total_h = get_u32_tag(&obj, dicom::dictionary_std::tags::TOTAL_PIXEL_MATRIX_ROWS);

        // Extract pixel data fragments (actual JPEG bytes per tile)
        let pixel_data = obj.element(dicom::dictionary_std::tags::PIXEL_DATA)
            .map_err(|_| anyhow!("No PixelData element found"))?;
        let fragments = pixel_data.fragments()
            .ok_or_else(|| anyhow!("PixelData is not encapsulated (no fragments)"))?;

        let tile_fragments: Vec<Vec<u8>> = fragments.iter().map(|f| f.clone()).collect();

        if tile_fragments.is_empty() {
            return Err(anyhow!("No pixel data fragments found"));
        }

        info!(
            "Extracted {} tile fragments ({:.1} MB total JPEG data)",
            tile_fragments.len(),
            tile_fragments.iter().map(|f| f.len()).sum::<usize>() as f64 / 1_048_576.0,
        );

        // Determine full image dimensions
        let (width, height) = match (total_w, total_h) {
            (Ok(w), Ok(h)) => {
                info!("Full image dims from TotalPixelMatrix tags: {}x{}", w, h);
                (w, h)
            }
            _ => {
                // Infer from frame count and tile dimensions
                let n = tile_fragments.len() as u32;
                let tiles_x_est = ((n as f64).sqrt().ceil() as u32).max(1);
                let tiles_y_est = (n + tiles_x_est - 1) / tiles_x_est;
                let w = tiles_x_est * tile_w;
                let h = tiles_y_est * tile_h;
                info!(
                    "No TotalPixelMatrix tags; inferring {}x{} from {} frames ({}x{} grid)",
                    w, h, n, tiles_x_est, tiles_y_est
                );
                (w, h)
            }
        };

        let tiles_x = (width + tile_w - 1) / tile_w;
        let tiles_y = (height + tile_h - 1) / tile_h;

        info!(
            "DICOM WSI: {}x{} pixels, tile={}x{}, grid={}x{} ({} tiles), {} fragments",
            width, height, tile_w, tile_h, tiles_x, tiles_y,
            tiles_x * tiles_y, tile_fragments.len(),
        );

        Ok(DicomSlide {
            width,
            height,
            tile_w,
            tile_h,
            tiles_x,
            tiles_y,
            tile_fragments,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn tile_w(&self) -> u32 {
        self.tile_w
    }

    pub fn tile_h(&self) -> u32 {
        self.tile_h
    }

    pub fn tiles_x(&self) -> u32 {
        self.tiles_x
    }

    pub fn tiles_y(&self) -> u32 {
        self.tiles_y
    }

    pub fn tile_count(&self) -> usize {
        self.tile_fragments.len()
    }

    /// Get the raw JPEG bytes for a tile at grid position (tx, ty).
    /// Returns None if the tile is out of range or has no data.
    pub fn get_tile_bytes(&self, tx: u32, ty: u32) -> Option<Vec<u8>> {
        if tx >= self.tiles_x || ty >= self.tiles_y {
            return None;
        }
        let idx = (ty * self.tiles_x + tx) as usize;
        if idx >= self.tile_fragments.len() {
            return None;
        }
        let fragment = &self.tile_fragments[idx];
        if fragment.is_empty() {
            return None;
        }
        Some(fragment.clone())
    }
}

/// Extract a u32 value from a DICOM tag.
fn get_u32_tag(
    obj: &dicom::object::DefaultDicomObject,
    tag: dicom::core::Tag,
) -> Result<u32> {
    let elem = obj.element(tag)
        .map_err(|_| anyhow!("Tag {:?} not found", tag))?;
    let val = elem.to_int::<u32>()
        .map_err(|e| anyhow!("Tag {:?} not u32: {}", tag, e))?;
    Ok(val)
}
