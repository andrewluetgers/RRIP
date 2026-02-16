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

pub struct PackFile {
    pub data: Vec<u8>,
    pub data_offset: usize,
    pub index: Vec<PackIndexEntry>,
}

impl PackFile {
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

/// Write a pack file with LZ4 compression. Uses "ORIG" magic, version 1.
///
/// Binary format (before LZ4 compression):
/// - Header (24 bytes):
///   [0..4]   magic "ORIG"
///   [4..6]   version (u16 LE) = 1
///   [6..8]   reserved (zeroes)
///   [8..12]  entry count (u32 LE)
///   [12..16] index_offset (u32 LE)
///   [16..20] data_offset (u32 LE)
///   [20..24] reserved (zeroes)
/// - Index (16 bytes per entry):
///   [0]      level_kind (1=L1, 0=L0)
///   [1]      idx_in_parent
///   [2..4]   reserved
///   [4..8]   offset relative to data_offset (u32 LE)
///   [8..12]  length (u32 LE)
///   [12..16] reserved
/// - Data: concatenated JPEG blobs
pub fn write_pack(pack_dir: &Path, x2: u32, y2: u32, entries: &[PackWriteEntry]) -> Result<()> {
    let header_size: usize = 24;
    let index_size = entries.len() * 16;
    let index_offset = header_size;
    let data_offset = header_size + index_size;

    // Calculate total data size
    let total_data_size: usize = entries.iter().map(|e| e.jpeg_data.len()).sum();
    let total_size = data_offset + total_data_size;

    let mut buf = vec![0u8; total_size];

    // Write header
    buf[0..4].copy_from_slice(b"ORIG");
    buf[4..6].copy_from_slice(&1u16.to_le_bytes()); // version
    // [6..8] reserved
    buf[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(index_offset as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(data_offset as u32).to_le_bytes());
    // [20..24] reserved

    // Write index and data
    let mut data_cursor: u32 = 0;
    for (i, entry) in entries.iter().enumerate() {
        let idx_start = index_offset + i * 16;
        buf[idx_start] = entry.level_kind;
        buf[idx_start + 1] = entry.idx_in_parent;
        // [2..4] reserved
        buf[idx_start + 4..idx_start + 8].copy_from_slice(&data_cursor.to_le_bytes());
        buf[idx_start + 8..idx_start + 12]
            .copy_from_slice(&(entry.jpeg_data.len() as u32).to_le_bytes());
        // [12..16] reserved

        let dst_start = data_offset + data_cursor as usize;
        buf[dst_start..dst_start + entry.jpeg_data.len()].copy_from_slice(&entry.jpeg_data);
        data_cursor += entry.jpeg_data.len() as u32;
    }

    // LZ4 compress
    let compressed = lz4_flex::compress_prepend_size(&buf);

    // Write to file
    fs::create_dir_all(pack_dir)?;
    let path = pack_dir.join(format!("{}_{}.pack", x2, y2));
    fs::write(&path, &compressed)
        .with_context(|| format!("Failed to write pack file {}", path.display()))?;

    Ok(())
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

/// LZ4-compress pack entries into the wire format used inside bundles.
/// This is the same LZ4 compression used by write_pack, producing identical bytes.
pub fn compress_pack_entries(entries: &[PackWriteEntry]) -> Vec<u8> {
    let header_size: usize = 24;
    let index_size = entries.len() * 16;
    let index_offset = header_size;
    let data_offset = header_size + index_size;
    let total_data_size: usize = entries.iter().map(|e| e.jpeg_data.len()).sum();
    let total_size = data_offset + total_data_size;

    let mut buf = vec![0u8; total_size];
    buf[0..4].copy_from_slice(b"ORIG");
    buf[4..6].copy_from_slice(&1u16.to_le_bytes());
    buf[8..12].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(index_offset as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(data_offset as u32).to_le_bytes());

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

    lz4_flex::compress_prepend_size(&buf)
}

/// Parse decompressed pack data into a PackFile.
/// Shared between open_pack and BundleFile::get_pack.
fn parse_pack_data(data: Vec<u8>) -> Result<PackFile> {
    if data.len() < 24 {
        return Err(anyhow!("pack too small"));
    }
    if &data[0..4] != b"ORIG" && &data[0..4] != b"RRIP" {
        return Err(anyhow!("pack magic mismatch"));
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != 1 {
        return Err(anyhow!("pack version mismatch"));
    }
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_roundtrip() {
        let dir = std::env::temp_dir().join("origami_test_packs");
        let entries = vec![
            PackWriteEntry {
                level_kind: 1,
                idx_in_parent: 0,
                jpeg_data: vec![0xFF, 0xD8, 0xFF, 0xD9], // minimal JPEG
            },
            PackWriteEntry {
                level_kind: 0,
                idx_in_parent: 3,
                jpeg_data: vec![1, 2, 3, 4, 5],
            },
        ];

        write_pack(&dir, 5, 7, &entries).unwrap();
        let pack = open_pack(&dir, 5, 7).unwrap();

        let r1 = pack.get_residual(1, 0).unwrap();
        assert_eq!(r1, &[0xFF, 0xD8, 0xFF, 0xD9]);

        let r2 = pack.get_residual(0, 3).unwrap();
        assert_eq!(r2, &[1, 2, 3, 4, 5]);

        assert!(pack.get_residual(0, 0).is_none());

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bundle_roundtrip() {
        let dir = std::env::temp_dir().join("origami_test_bundle");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // Create two families worth of pack entries
        let entries_0_0 = vec![
            PackWriteEntry { level_kind: 1, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8, 1, 2] },
            PackWriteEntry { level_kind: 0, idx_in_parent: 3, jpeg_data: vec![10, 20, 30] },
        ];
        let entries_1_0 = vec![
            PackWriteEntry { level_kind: 1, idx_in_parent: 1, jpeg_data: vec![0xFF, 0xD8, 3, 4, 5] },
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
        let r = pack_0.get_residual(1, 0).unwrap();
        assert_eq!(r, &[0xFF, 0xD8, 1, 2]);
        let r = pack_0.get_residual(0, 3).unwrap();
        assert_eq!(r, &[10, 20, 30]);

        // Read family (1,0)
        let pack_1 = bundle.get_pack(1, 0).unwrap();
        let r = pack_1.get_residual(1, 1).unwrap();
        assert_eq!(r, &[0xFF, 0xD8, 3, 4, 5]);

        // Out of range should error
        assert!(bundle.get_pack(2, 0).is_err());
        assert!(bundle.get_pack(0, 1).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compress_pack_entries_matches_write_pack() {
        // Verify compress_pack_entries produces the same LZ4 bytes as write_pack
        let dir = std::env::temp_dir().join("origami_test_compress_match");
        let _ = fs::remove_dir_all(&dir);

        let entries = vec![
            PackWriteEntry { level_kind: 1, idx_in_parent: 0, jpeg_data: vec![0xFF, 0xD8, 0xFF, 0xD9] },
            PackWriteEntry { level_kind: 0, idx_in_parent: 5, jpeg_data: vec![1, 2, 3] },
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
