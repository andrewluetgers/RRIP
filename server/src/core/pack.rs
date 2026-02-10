use anyhow::{anyhow, Context, Result};
use std::fs;
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

    if data.len() < 24 {
        return Err(anyhow!("pack too small"));
    }
    // Accept both "ORIG" (new) and "RRIP" (legacy) magic
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
}
