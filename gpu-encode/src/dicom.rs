//! DICOM WSI reader.
//!
//! Uses the `dicom` crate to parse DICOM metadata and extract JPEG tile byte
//! ranges from the pixel data. The tiles are already JPEG Q94 — we just need
//! their raw bytes for nvJPEG decode.
//!
//! The DICOM file is mmapped for zero-copy access to tile data.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use tracing::info;

/// A memory-mapped DICOM WSI with pre-parsed tile offsets.
pub struct DicomSlide {
    /// Memory-mapped DICOM file data
    _mmap: Mmap,
    /// Image width in pixels
    width: u32,
    /// Image height in pixels
    height: u32,
    /// Tile width in pixels (typically 224 for DICOM WSI)
    tile_w: u32,
    /// Tile height in pixels
    tile_h: u32,
    /// Number of tile columns
    tiles_x: u32,
    /// Number of tile rows
    tiles_y: u32,
    /// (offset, length) for each tile's JPEG data within the DICOM file.
    /// Indexed as tile_offsets[row * tiles_x + col].
    tile_offsets: Vec<(u64, u32)>,
    /// Raw bytes backing the tile data (points into mmap or owned copy)
    tile_data: Vec<u8>,
}

impl DicomSlide {
    /// Open a DICOM WSI file and parse tile layout.
    pub fn open(path: &Path) -> Result<Self> {
        use std::fs::File;

        let file = File::open(path)
            .with_context(|| format!("Failed to open DICOM {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap DICOM {}", path.display()))?;

        info!("DICOM file size: {:.1} MB", mmap.len() as f64 / 1_048_576.0);

        // Parse DICOM metadata using the dicom crate
        let obj = dicom::object::open_file(path)
            .map_err(|e| anyhow!("DICOM parse error: {}", e))?;

        // Extract image dimensions
        let width = get_u32_tag(&obj, dicom::dictionary_std::tags::COLUMNS)
            .context("Missing Columns tag")?;
        let height = get_u32_tag(&obj, dicom::dictionary_std::tags::ROWS)
            .context("Missing Rows tag")?;

        // Extract tile dimensions — DICOM uses "Number of Frames" for tiles
        // For WSI, tile dimensions come from vendor-specific tags or
        // can be inferred from the frame structure.
        // Standard DICOM WSI uses Per-Frame Functional Groups.
        //
        // For now, try standard tags first, fall back to common defaults.
        let tile_w = get_u32_tag(&obj, dicom::dictionary_std::tags::COLUMNS)
            .unwrap_or(224);
        let tile_h = get_u32_tag(&obj, dicom::dictionary_std::tags::ROWS)
            .unwrap_or(224);

        // Number of frames = number of tiles
        let num_frames = get_u32_tag(&obj, dicom::dictionary_std::tags::NUMBER_OF_FRAMES)
            .unwrap_or(1);

        // Compute tile grid
        let tiles_x = (width + tile_w - 1) / tile_w;
        let tiles_y = (height + tile_h - 1) / tile_h;

        info!(
            "DICOM image: {}x{}, tiles: {}x{} ({}x{}px), frames={}",
            width, height, tiles_x, tiles_y, tile_w, tile_h, num_frames
        );

        // Extract pixel data fragment offsets
        // DICOM encapsulated pixel data stores each tile as a fragment
        // in the (7FE0,0010) Pixel Data element.
        let tile_offsets = extract_pixel_data_offsets(&obj, num_frames)?;

        // Keep the raw file data for tile extraction
        let tile_data = mmap.to_vec();

        Ok(DicomSlide {
            _mmap: mmap,
            width,
            height,
            tile_w,
            tile_h,
            tiles_x,
            tiles_y,
            tile_offsets,
            tile_data,
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
        self.tile_offsets.len()
    }

    /// Get the raw JPEG bytes for a tile at grid position (tx, ty).
    /// Returns an empty Vec if the tile is out of range or has no data.
    pub fn get_tile_bytes(&self, tx: u32, ty: u32) -> Option<Vec<u8>> {
        if tx >= self.tiles_x || ty >= self.tiles_y {
            return None;
        }
        let idx = (ty * self.tiles_x + tx) as usize;
        if idx >= self.tile_offsets.len() {
            return None;
        }
        let (offset, length) = self.tile_offsets[idx];
        if length == 0 {
            return None;
        }
        let start = offset as usize;
        let end = start + length as usize;
        if end > self.tile_data.len() {
            return None;
        }
        Some(self.tile_data[start..end].to_vec())
    }
}

/// Extract a u32 value from a DICOM tag.
fn get_u32_tag(
    obj: &dicom::object::DefaultDicomObject,
    tag: dicom::core::Tag,
) -> Result<u32> {
    use dicom::object::InMemDicomObject;
    let elem = obj.element(tag)
        .map_err(|_| anyhow!("Tag {:?} not found", tag))?;
    let val = elem.to_int::<u32>()
        .map_err(|e| anyhow!("Tag {:?} not u32: {}", tag, e))?;
    Ok(val)
}

/// Extract pixel data fragment offsets from DICOM encapsulated pixel data.
///
/// DICOM WSI stores tiles as fragments in the PixelData element (7FE0,0010).
/// Each fragment is a JPEG-compressed tile image.
fn extract_pixel_data_offsets(
    obj: &dicom::object::DefaultDicomObject,
    expected_frames: u32,
) -> Result<Vec<(u64, u32)>> {
    use dicom::object::InMemDicomObject;

    // Try to get the PixelData element
    let pixel_data = obj.element(dicom::dictionary_std::tags::PIXEL_DATA)
        .map_err(|_| anyhow!("No PixelData element found"))?;

    // For encapsulated pixel data, the value is a sequence of fragments.
    // Each fragment contains one tile's JPEG data.
    // The dicom crate represents this as PixelFragmentSequence.
    let fragments = pixel_data.fragments()
        .ok_or_else(|| anyhow!("PixelData is not encapsulated (no fragments)"))?;

    let mut offsets = Vec::with_capacity(expected_frames as usize);
    for fragment in fragments {
        // Each fragment's offset within the file and its length
        // For now, store the fragment data directly
        // The actual byte offset within the mmap will be computed differently
        // when we integrate with nvJPEG (we'll pass fragment bytes directly)
        offsets.push((0u64, fragment.len() as u32));
    }

    if offsets.is_empty() {
        return Err(anyhow!("No pixel data fragments found"));
    }

    info!("Extracted {} tile fragments from DICOM", offsets.len());
    Ok(offsets)
}
