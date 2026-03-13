use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use std::fs::{self, File};
use std::path::Path;

pub struct PackIndexEntry {
    pub level_kind: u8,
    pub idx_in_parent: u8,
    pub offset: u32,
    pub length: u32,
}

/// Metadata embedded in a pack header (v3+). For v1/v2 packs, defaults are inferred.
#[derive(Debug, Clone, Copy)]
pub struct PackMetadata {
    pub version: u16,
    pub tile_size: u16,
    pub seed_w: u16,
    pub seed_h: u16,
    pub residual_w: u16,
    pub residual_h: u16,
    /// V4 split-seed: luma seed dimensions (0 = not split, use seed_w/h)
    pub seed_luma_w: u16,
    pub seed_luma_h: u16,
    /// V4 split-seed: chroma seed dimensions (0 = not split, use seed_w/h)
    pub seed_chroma_w: u16,
    pub seed_chroma_h: u16,
}

impl PackMetadata {
    /// Infer defaults for v1/v2 packs (seed = tile_size, residual = tile_size * 4).
    pub fn default_for_v2(tile_size: u16) -> Self {
        Self {
            version: 2,
            tile_size,
            seed_w: tile_size,
            seed_h: tile_size,
            residual_w: tile_size * 4,
            residual_h: tile_size * 4,
            seed_luma_w: 0,
            seed_luma_h: 0,
            seed_chroma_w: 0,
            seed_chroma_h: 0,
        }
    }

    /// Whether this pack uses split luma/chroma seeds (v4).
    pub fn is_split_seed(&self) -> bool {
        self.version >= 4 && self.seed_luma_w > 0
    }
}

pub struct PackFile {
    pub data: Vec<u8>,
    pub data_offset: usize,
    pub index: Vec<PackIndexEntry>,
    pub metadata: PackMetadata,
}

impl PackFile {
    /// Get the L2 baseline JPEG bytes from the pack.
    pub fn get_l2(&self) -> Option<&[u8]> {
        self.get_residual(2, 0)
    }

    /// Get the fused L0 residual (single grayscale JPEG mosaic) from the pack.
    pub fn get_fused_l0(&self) -> Option<&[u8]> {
        self.get_residual(0, 0)
    }

    /// Get the seed luma (grayscale JPEG) from a v4 split-seed pack.
    pub fn get_seed_luma(&self) -> Option<&[u8]> {
        self.get_residual(3, 0)
    }

    /// Get the seed chroma (RGB JPEG — decoder extracts Cb/Cr, discards Y) from a v4 split-seed pack.
    /// Legacy: used when level_kind=4 stores RGB chroma.
    pub fn get_seed_chroma(&self) -> Option<&[u8]> {
        self.get_residual(4, 0)
    }

    /// Get the seed Cb (grayscale) from a v4 split-seed pack (level_kind=4).
    pub fn get_seed_cb(&self) -> Option<&[u8]> {
        self.get_residual(4, 0)
    }

    /// Get the seed Cr (grayscale) from a v4 split-seed pack (level_kind=5).
    pub fn get_seed_cr(&self) -> Option<&[u8]> {
        self.get_residual(5, 0)
    }

    /// Get wavelet synthesis parameters (level_kind=6, 16 bytes) if present.
    pub fn get_synth_params(&self) -> Option<crate::core::wavelet::SynthesisParams> {
        let data = self.get_residual(6, 0)?;
        if data.len() < 16 {
            return None;
        }
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&data[..16]);
        Some(crate::core::wavelet::SynthesisParams::from_bytes(&buf))
    }

    /// Whether this pack uses split luma/chroma seeds (v4).
    pub fn is_split_seed(&self) -> bool {
        self.metadata.is_split_seed()
    }

    pub fn get_residual(&self, level_kind: u8, idx_in_parent: u8) -> Option<&[u8]> {
        let entry = self
            .index
            .iter()
            .find(|e| e.level_kind == level_kind && e.idx_in_parent == idx_in_parent)?;
        let start = self.data_offset + entry.offset as usize;
        let end = start + entry.length as usize;
        Some(&self.data[start..end])
    }
}

/// Open and decompress a pack file. Accepts both "ORIG" and "RRIP" magic for backwards compat.
pub fn open_pack(pack_dir: &Path, x2: u32, y2: u32) -> Result<PackFile> {
    let path = pack_dir.join(format!("{}_{}.pack", x2, y2));

    let compressed_data = fs::read(&path)
        .with_context(|| format!("Failed to read pack file {}", path.display()))?;

    let data = lz4_flex::decompress_size_prepended(&compressed_data)
        .with_context(|| format!("Failed to decompress pack file {}", path.display()))?;

    parse_pack_data(data)
}

/// Entry for writing a pack file
pub struct PackWriteEntry {
    pub level_kind: u8,
    pub idx_in_parent: u8,
    pub jpeg_data: Vec<u8>,
}

/// Write a v2 pack file with LZ4 compression. Uses "ORIG" magic, version 2.
///
/// V2 pack format (before LZ4 compression):
/// - Header (24 bytes):
///   [0..4]   magic "ORIG"
///   [4..6]   version (u16 LE) = 2
///   [6..8]   reserved (zeroes)
///   [8..12]  entry count (u32 LE) — typically 2 (L2 + fused L0)
///   [12..16] index_offset (u32 LE)
///   [16..20] data_offset (u32 LE)
///   [20..24] reserved (zeroes)
/// - Index (16 bytes per entry):
///   [0]      level_kind (2=L2 baseline, 0=fused L0 residual)
///   [1]      idx_in_parent (always 0)
///   [2..4]   reserved
///   [4..8]   offset relative to data_offset (u32 LE)
///   [8..12]  length (u32 LE)
///   [12..16] reserved
/// - Data: concatenated JPEG blobs
pub fn write_pack(pack_dir: &Path, x2: u32, y2: u32, entries: &[PackWriteEntry]) -> Result<()> {
    let compressed = build_pack_v2(entries);

    // Write to file
    fs::create_dir_all(pack_dir)?;
    let path = pack_dir.join(format!("{}_{}.pack", x2, y2));
    fs::write(&path, &compressed)
        .with_context(|| format!("Failed to write pack file {}", path.display()))?;

    Ok(())
}

/// Write a v3 pack file with seed/residual dimensions in the header.
///
/// V3 pack format (before LZ4 compression):
/// - Header (32 bytes):
///   [0..4]   magic "ORIG"
///   [4..6]   version (u16 LE) = 3
///   [6..8]   tile_size (u16 LE)
///   [8..12]  entry count (u32 LE)
///   [12..16] index_offset (u32 LE)
///   [16..20] data_offset (u32 LE)
///   [20..22] seed_w (u16 LE)
///   [22..24] seed_h (u16 LE)
///   [24..26] residual_w (u16 LE)
///   [26..28] residual_h (u16 LE)
///   [28..32] reserved (zeroes)
/// - Index and Data: same as v2
pub fn write_pack_v3(
    pack_dir: &Path,
    x2: u32,
    y2: u32,
    entries: &[PackWriteEntry],
    metadata: &PackMetadata,
) -> Result<()> {
    let compressed = build_pack_v3(entries, metadata);

    fs::create_dir_all(pack_dir)?;
    let path = pack_dir.join(format!("{}_{}.pack", x2, y2));
    fs::write(&path, &compressed)
        .with_context(|| format!("Failed to write pack file {}", path.display()))?;

    Ok(())
}

/// Write a v4 pack file with split luma/chroma seed dimensions in the header.
///
/// V4 pack format (before LZ4 compression):
/// - Header (48 bytes):
///   [0..4]   magic "ORIG"
///   [4..6]   version (u16 LE) = 4
///   [6..8]   tile_size (u16 LE)
///   [8..12]  entry count (u32 LE) — typically 3 (seed_luma + seed_chroma + fused L0)
///   [12..16] index_offset (u32 LE)
///   [16..20] data_offset (u32 LE)
///   [20..22] seed_luma_w (u16 LE)
///   [22..24] seed_luma_h (u16 LE)
///   [24..26] seed_chroma_w (u16 LE)
///   [26..28] seed_chroma_h (u16 LE)
///   [28..30] residual_w (u16 LE)
///   [30..32] residual_h (u16 LE)
///   [32..48] reserved (zeroes)
/// - Index (16 bytes per entry):
///   [0]      level_kind (3=seed_luma, 4=seed_chroma, 0=fused L0 residual)
///   [1]      idx_in_parent (always 0)
///   [2..4]   reserved
///   [4..8]   offset relative to data_offset (u32 LE)
///   [8..12]  length (u32 LE)
///   [12..16] reserved
/// - Data: concatenated JPEG blobs
pub fn write_pack_v4(
    pack_dir: &Path,
    x2: u32,
    y2: u32,
    entries: &[PackWriteEntry],
    metadata: &PackMetadata,
) -> Result<()> {
    let compressed = build_pack_v4(entries, metadata);

    fs::create_dir_all(pack_dir)?;
    let path = pack_dir.join(format!("{}_{}.pack", x2, y2));
    fs::write(&path, &compressed)
        .with_context(|| format!("Failed to write pack file {}", path.display()))?;

    Ok(())
}

/// Build v4 pack bytes (LZ4 compressed) with split luma/chroma seed dims in header.
fn build_pack_v4(entries: &[PackWriteEntry], meta: &PackMetadata) -> Vec<u8> {
    let header_size: usize = 48;
    let index_size = entries.len() * 16;
    let index_offset = header_size;
    let data_offset = header_size + index_size;

    let total_data_size: usize = entries.iter().map(|e| e.jpeg_data.len()).sum();
    let total_size = data_offset + total_data_size;

    let mut buf = vec![0u8; total_size];

    buf[0..4].copy_from_slice(b"ORIG");
    buf[4..6].copy_from_slice(&4u16.to_le_bytes());
    buf[6..8].copy_from_slice(&meta.tile_size.to_le_bytes());
    buf[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(index_offset as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(data_offset as u32).to_le_bytes());
    buf[20..22].copy_from_slice(&meta.seed_luma_w.to_le_bytes());
    buf[22..24].copy_from_slice(&meta.seed_luma_h.to_le_bytes());
    buf[24..26].copy_from_slice(&meta.seed_chroma_w.to_le_bytes());
    buf[26..28].copy_from_slice(&meta.seed_chroma_h.to_le_bytes());
    buf[28..30].copy_from_slice(&meta.residual_w.to_le_bytes());
    buf[30..32].copy_from_slice(&meta.residual_h.to_le_bytes());
    // [32..48] reserved

    write_pack_index_and_data(&mut buf, entries, index_offset, data_offset);
    lz4_flex::compress_prepend_size(&buf)
}

/// LZ4-compress pack entries into v4 wire format with split seed metadata.
pub fn compress_pack_entries_v4(entries: &[PackWriteEntry], metadata: &PackMetadata) -> Vec<u8> {
    build_pack_v4(entries, metadata)
}

/// Build v2 pack bytes (LZ4 compressed).
fn build_pack_v2(entries: &[PackWriteEntry]) -> Vec<u8> {
    let header_size: usize = 24;
    let index_size = entries.len() * 16;
    let index_offset = header_size;
    let data_offset = header_size + index_size;

    let total_data_size: usize = entries.iter().map(|e| e.jpeg_data.len()).sum();
    let total_size = data_offset + total_data_size;

    let mut buf = vec![0u8; total_size];

    buf[0..4].copy_from_slice(b"ORIG");
    buf[4..6].copy_from_slice(&2u16.to_le_bytes());
    buf[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(index_offset as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(data_offset as u32).to_le_bytes());

    write_pack_index_and_data(&mut buf, entries, index_offset, data_offset);
    lz4_flex::compress_prepend_size(&buf)
}

/// Build v3 pack bytes (LZ4 compressed) with seed/residual dims in header.
fn build_pack_v3(entries: &[PackWriteEntry], meta: &PackMetadata) -> Vec<u8> {
    let header_size: usize = 32;
    let index_size = entries.len() * 16;
    let index_offset = header_size;
    let data_offset = header_size + index_size;

    let total_data_size: usize = entries.iter().map(|e| e.jpeg_data.len()).sum();
    let total_size = data_offset + total_data_size;

    let mut buf = vec![0u8; total_size];

    buf[0..4].copy_from_slice(b"ORIG");
    buf[4..6].copy_from_slice(&3u16.to_le_bytes());
    buf[6..8].copy_from_slice(&meta.tile_size.to_le_bytes());
    buf[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(index_offset as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(data_offset as u32).to_le_bytes());
    buf[20..22].copy_from_slice(&meta.seed_w.to_le_bytes());
    buf[22..24].copy_from_slice(&meta.seed_h.to_le_bytes());
    buf[24..26].copy_from_slice(&meta.residual_w.to_le_bytes());
    buf[26..28].copy_from_slice(&meta.residual_h.to_le_bytes());
    // [28..32] reserved

    write_pack_index_and_data(&mut buf, entries, index_offset, data_offset);
    lz4_flex::compress_prepend_size(&buf)
}

/// Write index entries and data blobs into a pre-allocated buffer.
fn write_pack_index_and_data(buf: &mut [u8], entries: &[PackWriteEntry], index_offset: usize, data_offset: usize) {
    let mut data_cursor: u32 = 0;
    for (i, entry) in entries.iter().enumerate() {
        let idx_start = index_offset + i * 16;
        buf[idx_start] = entry.level_kind;
        buf[idx_start + 1] = entry.idx_in_parent;
        buf[idx_start + 4..idx_start + 8].copy_from_slice(&data_cursor.to_le_bytes());
        buf[idx_start + 8..idx_start + 12]
            .copy_from_slice(&(entry.jpeg_data.len() as u32).to_le_bytes());

        let dst_start = data_offset + data_cursor as usize;
        buf[dst_start..dst_start + entry.jpeg_data.len()].copy_from_slice(&entry.jpeg_data);
        data_cursor += entry.jpeg_data.len() as u32;
    }
}

// ---------------------------------------------------------------------------
// Bundle format — single mmapped file containing all families for a WSI
// ---------------------------------------------------------------------------
//
// Format:
//   Header (32 bytes):
//     [0..4]    magic "ORIB"
//     [4..6]    version u16 = 1
//     [6..8]    grid_cols u16
//     [8..10]   grid_rows u16
//     [10..14]  family_count u32
//     [14..22]  index_offset u64
//     [22..32]  reserved
//
//   Pack data (variable):
//     [family 0 LZ4 bytes] [family 1 LZ4 bytes] ...
//
//   Index (12 bytes per family, at index_offset):
//     offset: u64   — byte offset into file for this family's LZ4 data
//     length: u32   — byte length of LZ4 data
//     (ordered row-major: family_idx = row * grid_cols + col)

const BUNDLE_MAGIC: &[u8; 4] = b"ORIB";
const BUNDLE_VERSION: u16 = 1;
const BUNDLE_HEADER_SIZE: usize = 32;
const BUNDLE_INDEX_ENTRY_SIZE: usize = 12;

/// A memory-mapped bundle file for serving. Provides lock-free concurrent reads.
pub struct BundleFile {
    mmap: Mmap,
    grid_cols: u16,
    grid_rows: u16,
    #[allow(dead_code)]
    family_count: u32,
    index_offset: u64,
}

/// Index entry for one family within the bundle.
struct BundleIndexEntry {
    offset: u64,
    length: u32,
}

impl BundleFile {
    /// Get the number of grid columns.
    pub fn grid_cols(&self) -> u16 {
        self.grid_cols
    }

    /// Get the number of grid rows.
    pub fn grid_rows(&self) -> u16 {
        self.grid_rows
    }

    /// Retrieve and decompress the pack for family at (col, row).
    /// Returns the same PackFile as open_pack() would.
    pub fn get_pack(&self, col: u32, row: u32) -> Result<PackFile> {
        if col >= self.grid_cols as u32 || row >= self.grid_rows as u32 {
            return Err(anyhow!(
                "bundle family ({},{}) out of range ({}x{})",
                col, row, self.grid_cols, self.grid_rows
            ));
        }
        let idx = (row as u64) * (self.grid_cols as u64) + (col as u64);
        let entry = self.read_index_entry(idx)?;
        if entry.length == 0 {
            return Err(anyhow!("bundle family ({},{}) has no data", col, row));
        }
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        if end > self.mmap.len() {
            return Err(anyhow!("bundle family ({},{}) data extends past EOF", col, row));
        }
        let compressed = &self.mmap[start..end];
        let data = lz4_flex::decompress_size_prepended(compressed)
            .with_context(|| format!("LZ4 decompress failed for family ({},{})", col, row))?;
        parse_pack_data(data)
    }

    fn read_index_entry(&self, idx: u64) -> Result<BundleIndexEntry> {
        let entry_offset = self.index_offset + idx * BUNDLE_INDEX_ENTRY_SIZE as u64;
        let o = entry_offset as usize;
        if o + BUNDLE_INDEX_ENTRY_SIZE > self.mmap.len() {
            return Err(anyhow!("bundle index entry {} out of range", idx));
        }
        let offset = u64::from_le_bytes([
            self.mmap[o], self.mmap[o+1], self.mmap[o+2], self.mmap[o+3],
            self.mmap[o+4], self.mmap[o+5], self.mmap[o+6], self.mmap[o+7],
        ]);
        let length = u32::from_le_bytes([
            self.mmap[o+8], self.mmap[o+9], self.mmap[o+10], self.mmap[o+11],
        ]);
        Ok(BundleIndexEntry { offset, length })
    }
}

/// Open and mmap a bundle file.
pub fn open_bundle(path: &Path) -> Result<BundleFile> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open bundle {}", path.display()))?;
    // SAFETY: We only read from the mmap, and the file is not modified while we hold it.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("Failed to mmap bundle {}", path.display()))?;

    if mmap.len() < BUNDLE_HEADER_SIZE {
        return Err(anyhow!("bundle too small: {} bytes", mmap.len()));
    }
    if &mmap[0..4] != BUNDLE_MAGIC {
        return Err(anyhow!("bundle magic mismatch"));
    }
    let version = u16::from_le_bytes([mmap[4], mmap[5]]);
    if version != BUNDLE_VERSION {
        return Err(anyhow!("bundle version {} not supported", version));
    }
    let grid_cols = u16::from_le_bytes([mmap[6], mmap[7]]);
    let grid_rows = u16::from_le_bytes([mmap[8], mmap[9]]);
    let family_count = u32::from_le_bytes([mmap[10], mmap[11], mmap[12], mmap[13]]);
    let index_offset = u64::from_le_bytes([
        mmap[14], mmap[15], mmap[16], mmap[17],
        mmap[18], mmap[19], mmap[20], mmap[21],
    ]);

    // Validate index fits in file
    let expected_index_end = index_offset as usize
        + family_count as usize * BUNDLE_INDEX_ENTRY_SIZE;
    if expected_index_end > mmap.len() {
        return Err(anyhow!(
            "bundle index extends past EOF (index_offset={}, count={}, file_len={})",
            index_offset, family_count, mmap.len()
        ));
    }

    Ok(BundleFile {
        mmap,
        grid_cols,
        grid_rows,
        family_count,
        index_offset,
    })
}

/// Write a bundle file from a collection of families.
///
/// `families` is a slice of `(col, row, lz4_compressed_pack_bytes)`.
/// Families not present in the input will have zero-length entries in the index.
pub fn write_bundle(
    path: &Path,
    grid_cols: u16,
    grid_rows: u16,
    families: &[(u32, u32, Vec<u8>)],
) -> Result<()> {
    let family_count = (grid_cols as u32) * (grid_rows as u32);

    // Build offset/length arrays (row-major order)
    let mut offsets = vec![0u64; family_count as usize];
    let mut lengths = vec![0u32; family_count as usize];

    // Data starts right after header
    let mut data_cursor = BUNDLE_HEADER_SIZE as u64;
    for (col, row, lz4_data) in families {
        let idx = (*row as usize) * (grid_cols as usize) + (*col as usize);
        offsets[idx] = data_cursor;
        lengths[idx] = lz4_data.len() as u32;
        data_cursor += lz4_data.len() as u64;
    }

    let index_offset = data_cursor;
    let total_size = index_offset as usize
        + family_count as usize * BUNDLE_INDEX_ENTRY_SIZE;

    let mut buf = vec![0u8; total_size];

    // Write header
    buf[0..4].copy_from_slice(BUNDLE_MAGIC);
    buf[4..6].copy_from_slice(&BUNDLE_VERSION.to_le_bytes());
    buf[6..8].copy_from_slice(&grid_cols.to_le_bytes());
    buf[8..10].copy_from_slice(&grid_rows.to_le_bytes());
    buf[10..14].copy_from_slice(&family_count.to_le_bytes());
    buf[14..22].copy_from_slice(&index_offset.to_le_bytes());
    // [22..32] reserved (already zeroed)

    // Write pack data
    for (col, row, lz4_data) in families {
        let idx = (*row as usize) * (grid_cols as usize) + (*col as usize);
        let start = offsets[idx] as usize;
        buf[start..start + lz4_data.len()].copy_from_slice(lz4_data);
        let _ = idx; // already used above
    }

    // Write index
    let idx_base = index_offset as usize;
    for i in 0..family_count as usize {
        let entry_offset = idx_base + i * BUNDLE_INDEX_ENTRY_SIZE;
        buf[entry_offset..entry_offset + 8].copy_from_slice(&offsets[i].to_le_bytes());
        buf[entry_offset + 8..entry_offset + 12].copy_from_slice(&lengths[i].to_le_bytes());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, &buf)
        .with_context(|| format!("Failed to write bundle {}", path.display()))?;
    Ok(())
}

/// LZ4-compress pack entries into v2 wire format used inside bundles.
/// This is the same LZ4 compression used by write_pack, producing identical bytes.
pub fn compress_pack_entries(entries: &[PackWriteEntry]) -> Vec<u8> {
    build_pack_v2(entries)
}

/// LZ4-compress pack entries into v3 wire format with seed/residual metadata.
pub fn compress_pack_entries_v3(entries: &[PackWriteEntry], metadata: &PackMetadata) -> Vec<u8> {
    build_pack_v3(entries, metadata)
}

/// Parse decompressed pack data into a PackFile.
/// Shared between open_pack and BundleFile::get_pack.
/// Accepts v1, v2, and v3 pack formats.
fn parse_pack_data(data: Vec<u8>) -> Result<PackFile> {
    if data.len() < 24 {
        return Err(anyhow!("pack too small"));
    }
    if &data[0..4] != b"ORIG" && &data[0..4] != b"RRIP" {
        return Err(anyhow!("pack magic mismatch"));
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version < 1 || version > 4 {
        return Err(anyhow!("pack version {} not supported (expected 1-4)", version));
    }

    // Parse metadata fields based on version
    let metadata = if version == 4 {
        if data.len() < 48 {
            return Err(anyhow!("v4 pack too small for 48-byte header"));
        }
        PackMetadata {
            version,
            tile_size: u16::from_le_bytes([data[6], data[7]]),
            seed_w: 0, // v4 uses split luma/chroma, not unified seed
            seed_h: 0,
            seed_luma_w: u16::from_le_bytes([data[20], data[21]]),
            seed_luma_h: u16::from_le_bytes([data[22], data[23]]),
            seed_chroma_w: u16::from_le_bytes([data[24], data[25]]),
            seed_chroma_h: u16::from_le_bytes([data[26], data[27]]),
            residual_w: u16::from_le_bytes([data[28], data[29]]),
            residual_h: u16::from_le_bytes([data[30], data[31]]),
        }
    } else if version == 3 {
        if data.len() < 32 {
            return Err(anyhow!("v3 pack too small for 32-byte header"));
        }
        PackMetadata {
            version,
            tile_size: u16::from_le_bytes([data[6], data[7]]),
            seed_w: u16::from_le_bytes([data[20], data[21]]),
            seed_h: u16::from_le_bytes([data[22], data[23]]),
            residual_w: u16::from_le_bytes([data[24], data[25]]),
            residual_h: u16::from_le_bytes([data[26], data[27]]),
            seed_luma_w: 0,
            seed_luma_h: 0,
            seed_chroma_w: 0,
            seed_chroma_h: 0,
        }
    } else {
        // v1/v2: tile_size stored in reserved [6..8] was zero; default to 256
        PackMetadata::default_for_v2(256)
    };

    let count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let index_offset = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let data_offset = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
    let mut index = Vec::with_capacity(count);
    let mut cursor = index_offset;
    for _ in 0..count {
        let level_kind = data[cursor];
        let idx_in_parent = data[cursor + 1];
        let offset = u32::from_le_bytes([
            data[cursor + 4],
            data[cursor + 5],
            data[cursor + 6],
            data[cursor + 7],
        ]);
        let length = u32::from_le_bytes([
            data[cursor + 8],
            data[cursor + 9],
            data[cursor + 10],
            data[cursor + 11],
        ]);
        index.push(PackIndexEntry {
            level_kind,
            idx_in_parent,
            offset,
            length,
        });
        cursor += 16;
    }
    Ok(PackFile {
        data,
        data_offset,
        index,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_v2_roundtrip() {
        let dir = std::env::temp_dir().join("origami_test_packs_v2");
        let entries = vec![
            PackWriteEntry {
                level_kind: 2,
                idx_in_parent: 0,
                jpeg_data: vec![0xFF, 0xD8, 0xFF, 0xD9], // L2 baseline
            },
            PackWriteEntry {
                level_kind: 0,
                idx_in_parent: 0,
                jpeg_data: vec![1, 2, 3, 4, 5], // fused L0 residual
            },
        ];

        write_pack(&dir, 5, 7, &entries).unwrap();
        let pack = open_pack(&dir, 5, 7).unwrap();

        let l2 = pack.get_l2().unwrap();
        assert_eq!(l2, &[0xFF, 0xD8, 0xFF, 0xD9]);

        let l0 = pack.get_fused_l0().unwrap();
        assert_eq!(l0, &[1, 2, 3, 4, 5]);

        // No L1 entries
        assert!(pack.get_residual(1, 0).is_none());

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bundle_roundtrip() {
        let dir = std::env::temp_dir().join("origami_test_bundle_v2");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // Create two families worth of v2 pack entries
        let entries_0_0 = vec![
            PackWriteEntry { level_kind: 2, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8, 1, 2] },
            PackWriteEntry { level_kind: 0, idx_in_parent: 0, jpeg_data: vec![10, 20, 30] },
        ];
        let entries_1_0 = vec![
            PackWriteEntry { level_kind: 2, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8, 3, 4, 5] },
            PackWriteEntry { level_kind: 0, idx_in_parent: 0, jpeg_data: vec![40, 50] },
        ];

        let lz4_0 = compress_pack_entries(&entries_0_0);
        let lz4_1 = compress_pack_entries(&entries_1_0);

        let families = vec![
            (0u32, 0u32, lz4_0),
            (1u32, 0u32, lz4_1),
        ];

        let bundle_path = dir.join("residuals.bundle");
        write_bundle(&bundle_path, 2, 1, &families).unwrap();

        let bundle = open_bundle(&bundle_path).unwrap();
        assert_eq!(bundle.grid_cols(), 2);
        assert_eq!(bundle.grid_rows(), 1);

        // Read family (0,0)
        let pack_0 = bundle.get_pack(0, 0).unwrap();
        let l2 = pack_0.get_l2().unwrap();
        assert_eq!(l2, &[0xFF, 0xD8, 1, 2]);
        let l0 = pack_0.get_fused_l0().unwrap();
        assert_eq!(l0, &[10, 20, 30]);

        // Read family (1,0)
        let pack_1 = bundle.get_pack(1, 0).unwrap();
        let l2 = pack_1.get_l2().unwrap();
        assert_eq!(l2, &[0xFF, 0xD8, 3, 4, 5]);
        let l0 = pack_1.get_fused_l0().unwrap();
        assert_eq!(l0, &[40, 50]);

        // Out of range should error
        assert!(bundle.get_pack(2, 0).is_err());
        assert!(bundle.get_pack(0, 1).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pack_v3_roundtrip() {
        let dir = std::env::temp_dir().join("origami_test_packs_v3");
        let _ = fs::remove_dir_all(&dir);
        let entries = vec![
            PackWriteEntry {
                level_kind: 2,
                idx_in_parent: 0,
                jpeg_data: vec![0xFF, 0xD8, 0xFF, 0xD9], // seed image
            },
            PackWriteEntry {
                level_kind: 0,
                idx_in_parent: 0,
                jpeg_data: vec![1, 2, 3, 4, 5], // fused L0 residual
            },
        ];

        let meta = PackMetadata {
            version: 3,
            tile_size: 256,
            seed_w: 384,
            seed_h: 384,
            residual_w: 1024,
            residual_h: 1024,
            seed_luma_w: 0,
            seed_luma_h: 0,
            seed_chroma_w: 0,
            seed_chroma_h: 0,
        };

        write_pack_v3(&dir, 3, 5, &entries, &meta).unwrap();
        let pack = open_pack(&dir, 3, 5).unwrap();

        // Verify metadata
        assert_eq!(pack.metadata.version, 3);
        assert_eq!(pack.metadata.tile_size, 256);
        assert_eq!(pack.metadata.seed_w, 384);
        assert_eq!(pack.metadata.seed_h, 384);
        assert_eq!(pack.metadata.residual_w, 1024);
        assert_eq!(pack.metadata.residual_h, 1024);

        // Verify data
        let l2 = pack.get_l2().unwrap();
        assert_eq!(l2, &[0xFF, 0xD8, 0xFF, 0xD9]);
        let l0 = pack.get_fused_l0().unwrap();
        assert_eq!(l0, &[1, 2, 3, 4, 5]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compress_pack_entries_v3_roundtrip() {
        let entries = vec![
            PackWriteEntry { level_kind: 2, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8, 10, 20] },
            PackWriteEntry { level_kind: 0, idx_in_parent: 0, jpeg_data: vec![30, 40, 50] },
        ];
        let meta = PackMetadata {
            version: 3,
            tile_size: 256,
            seed_w: 512,
            seed_h: 512,
            residual_w: 900,
            residual_h: 900,
            seed_luma_w: 0,
            seed_luma_h: 0,
            seed_chroma_w: 0,
            seed_chroma_h: 0,
        };

        let lz4 = compress_pack_entries_v3(&entries, &meta);
        let data = lz4_flex::decompress_size_prepended(&lz4).unwrap();
        let pack = parse_pack_data(data).unwrap();

        assert_eq!(pack.metadata.version, 3);
        assert_eq!(pack.metadata.seed_w, 512);
        assert_eq!(pack.metadata.seed_h, 512);
        assert_eq!(pack.metadata.residual_w, 900);
        assert_eq!(pack.metadata.residual_h, 900);
        assert_eq!(pack.get_l2().unwrap(), &[0xFF, 0xD8, 10, 20]);
        assert_eq!(pack.get_fused_l0().unwrap(), &[30, 40, 50]);
    }

    #[test]
    fn test_pack_v4_roundtrip() {
        let dir = std::env::temp_dir().join("origami_test_packs_v4");
        let _ = fs::remove_dir_all(&dir);
        let entries = vec![
            PackWriteEntry {
                level_kind: 3,
                idx_in_parent: 0,
                jpeg_data: vec![0xFF, 0xD8, 0x01, 0x02], // seed luma
            },
            PackWriteEntry {
                level_kind: 4,
                idx_in_parent: 0,
                jpeg_data: vec![0xFF, 0xD8, 0x03, 0x04, 0x05], // seed chroma
            },
            PackWriteEntry {
                level_kind: 0,
                idx_in_parent: 0,
                jpeg_data: vec![1, 2, 3, 4, 5], // fused L0 residual
            },
        ];

        let meta = PackMetadata {
            version: 4,
            tile_size: 256,
            seed_w: 0,
            seed_h: 0,
            seed_luma_w: 512,
            seed_luma_h: 512,
            seed_chroma_w: 128,
            seed_chroma_h: 128,
            residual_w: 1024,
            residual_h: 1024,
        };

        write_pack_v4(&dir, 2, 3, &entries, &meta).unwrap();
        let pack = open_pack(&dir, 2, 3).unwrap();

        // Verify metadata
        assert_eq!(pack.metadata.version, 4);
        assert_eq!(pack.metadata.tile_size, 256);
        assert_eq!(pack.metadata.seed_luma_w, 512);
        assert_eq!(pack.metadata.seed_luma_h, 512);
        assert_eq!(pack.metadata.seed_chroma_w, 128);
        assert_eq!(pack.metadata.seed_chroma_h, 128);
        assert_eq!(pack.metadata.residual_w, 1024);
        assert_eq!(pack.metadata.residual_h, 1024);
        assert!(pack.is_split_seed());

        // Verify data
        let luma = pack.get_seed_luma().unwrap();
        assert_eq!(luma, &[0xFF, 0xD8, 0x01, 0x02]);
        let chroma = pack.get_seed_chroma().unwrap();
        assert_eq!(chroma, &[0xFF, 0xD8, 0x03, 0x04, 0x05]);
        let l0 = pack.get_fused_l0().unwrap();
        assert_eq!(l0, &[1, 2, 3, 4, 5]);

        // Old accessors should not find split seed entries
        assert!(pack.get_l2().is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compress_pack_entries_v4_roundtrip() {
        let entries = vec![
            PackWriteEntry { level_kind: 3, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8, 10, 20] },
            PackWriteEntry { level_kind: 4, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8, 30, 40] },
            PackWriteEntry { level_kind: 0, idx_in_parent: 0, jpeg_data: vec![50, 60, 70] },
        ];
        let meta = PackMetadata {
            version: 4,
            tile_size: 256,
            seed_w: 0,
            seed_h: 0,
            seed_luma_w: 384,
            seed_luma_h: 384,
            seed_chroma_w: 64,
            seed_chroma_h: 64,
            residual_w: 1024,
            residual_h: 1024,
        };

        let lz4 = compress_pack_entries_v4(&entries, &meta);
        let data = lz4_flex::decompress_size_prepended(&lz4).unwrap();
        let pack = parse_pack_data(data).unwrap();

        assert_eq!(pack.metadata.version, 4);
        assert_eq!(pack.metadata.seed_luma_w, 384);
        assert_eq!(pack.metadata.seed_chroma_w, 64);
        assert!(pack.is_split_seed());
        assert_eq!(pack.get_seed_luma().unwrap(), &[0xFF, 0xD8, 10, 20]);
        assert_eq!(pack.get_seed_chroma().unwrap(), &[0xFF, 0xD8, 30, 40]);
        assert_eq!(pack.get_fused_l0().unwrap(), &[50, 60, 70]);
    }

    #[test]
    fn test_v2_pack_not_split_seed() {
        // Verify v2 packs report is_split_seed() = false
        let dir = std::env::temp_dir().join("origami_test_v2_not_split");
        let _ = fs::remove_dir_all(&dir);
        let entries = vec![
            PackWriteEntry { level_kind: 2, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8] },
            PackWriteEntry { level_kind: 0, idx_in_parent: 0, jpeg_data: vec![1, 2] },
        ];
        write_pack(&dir, 0, 0, &entries).unwrap();
        let pack = open_pack(&dir, 0, 0).unwrap();
        assert!(!pack.is_split_seed());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compress_pack_entries_matches_write_pack() {
        // Verify compress_pack_entries produces the same LZ4 bytes as write_pack
        let dir = std::env::temp_dir().join("origami_test_compress_match_v2");
        let _ = fs::remove_dir_all(&dir);

        let entries = vec![
            PackWriteEntry { level_kind: 2, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8, 0xFF, 0xD9] },
            PackWriteEntry { level_kind: 0, idx_in_parent: 0, jpeg_data: vec![1, 2, 3] },
        ];

        // write_pack writes to disk
        write_pack(&dir, 0, 0, &entries).unwrap();
        let on_disk = fs::read(dir.join("0_0.pack")).unwrap();

        // compress_pack_entries produces in-memory
        let in_memory = compress_pack_entries(&entries);

        assert_eq!(on_disk, in_memory);

        let _ = fs::remove_dir_all(&dir);
    }
}
