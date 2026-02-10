use std::fs;
use std::path::Path;

use anyhow::{anyhow, Result};

/// Pyramid metadata discovered from a DZI directory.
pub struct PyramidInfo {
    pub files_dir: std::path::PathBuf,
    pub tile_size: u32,
    pub max_level: u32,
    pub l0: u32,
    pub l1: u32,
    pub l2: u32,
}

/// Discover a pyramid from a DZI directory, returning its metadata.
pub fn discover_pyramid(pyramid_dir: &Path, tile_size: u32) -> Result<PyramidInfo> {
    let dzi_path = pyramid_dir.join("baseline_pyramid.dzi");
    let files_dir = pyramid_dir.join("baseline_pyramid_files");

    if !files_dir.exists() {
        return Err(anyhow!(
            "Pyramid files directory not found: {}",
            files_dir.display()
        ));
    }

    let actual_tile_size = if dzi_path.exists() {
        parse_tile_size(&dzi_path).unwrap_or(tile_size)
    } else {
        tile_size
    };

    let max_level = max_level_from_files(&files_dir)?;
    let l0 = max_level;
    let l1 = max_level.saturating_sub(1);
    let l2 = max_level.saturating_sub(2);

    Ok(PyramidInfo {
        files_dir,
        tile_size: actual_tile_size,
        max_level,
        l0,
        l1,
        l2,
    })
}

/// Parse TileSize from a .dzi XML file.
pub fn parse_tile_size(dzi_path: &Path) -> Option<u32> {
    let contents = fs::read_to_string(dzi_path).ok()?;
    let key = "TileSize=\"";
    let start = contents.find(key)? + key.len();
    let end = contents[start..].find('"')? + start;
    contents[start..end].parse().ok()
}

/// Find the maximum level number from subdirectories in a pyramid files dir.
pub fn max_level_from_files(files_dir: &Path) -> Result<u32> {
    let mut max_level: Option<u32> = None;
    for entry in fs::read_dir(files_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if let Ok(level) = name.parse::<u32>() {
            max_level = Some(max_level.map_or(level, |m| m.max(level)));
        }
    }
    max_level.ok_or_else(|| anyhow!("no level directories in {}", files_dir.display()))
}

/// Parse tile coordinates from a filename like "3_2.jpg" → Some((3, 2)).
pub fn parse_tile_coords(name: &str) -> Option<(u32, u32)> {
    let trimmed = name.strip_suffix(".jpg")?;
    let mut parts = trimmed.split('_');
    let x = parts.next()?.parse().ok()?;
    let y = parts.next()?.parse().ok()?;
    Some((x, y))
}

/// Extract a tile_w x tile_h region from a single-channel plane at (x0, y0).
pub fn extract_tile_plane(
    plane: &[u8],
    plane_w: u32,
    plane_h: u32,
    x0: u32,
    y0: u32,
    tile_w: u32,
    tile_h: u32,
    out: &mut [u8],
) {
    for y in 0..tile_h {
        let py = y0 + y;
        if py >= plane_h {
            continue;
        }
        for x in 0..tile_w {
            let px = x0 + x;
            if px >= plane_w {
                continue;
            }
            out[(y * tile_w + x) as usize] = plane[(py * plane_w + px) as usize];
        }
    }
}

/// Copy an RGB tile into an RGB mosaic at grid position (dx, dy).
pub fn copy_tile_into_mosaic(
    tile: &[u8],
    mosaic: &mut [u8],
    mosaic_width: u32,
    tile_size: u32,
    dx: u32,
    dy: u32,
) {
    let tile_stride = (tile_size * 3) as usize;
    let mosaic_stride = (mosaic_width * 3) as usize;
    let base_x = (dx * tile_size * 3) as usize;
    let base_y = (dy * tile_size) as usize;
    for y in 0..tile_size as usize {
        let tile_off = y * tile_stride;
        let mosaic_off = (base_y + y) * mosaic_stride + base_x;
        let dst = &mut mosaic[mosaic_off..mosaic_off + tile_stride];
        let src = &tile[tile_off..tile_off + tile_stride];
        dst.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data").join("demo_out")
    }

    #[test]
    fn test_parse_tile_coords_valid() {
        assert_eq!(parse_tile_coords("3_2.jpg"), Some((3, 2)));
        assert_eq!(parse_tile_coords("100_200.jpg"), Some((100, 200)));
        assert_eq!(parse_tile_coords("0_0.jpg"), Some((0, 0)));
    }

    #[test]
    fn test_parse_tile_coords_invalid() {
        assert_eq!(parse_tile_coords("foo.jpg"), None);
        assert_eq!(parse_tile_coords("3_2.png"), None);
        assert_eq!(parse_tile_coords("3_2"), None);
        assert_eq!(parse_tile_coords(""), None);
        assert_eq!(parse_tile_coords("abc"), None);
    }

    #[test]
    fn test_parse_tile_size_from_dzi() {
        let dzi_path = test_data_dir().join("baseline_pyramid.dzi");
        if !dzi_path.exists() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let size = parse_tile_size(&dzi_path);
        assert_eq!(size, Some(256));
    }

    #[test]
    fn test_max_level_from_files() {
        let files_dir = test_data_dir().join("baseline_pyramid_files");
        if !files_dir.exists() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let max = max_level_from_files(&files_dir).unwrap();
        assert_eq!(max, 16);
    }

    #[test]
    fn test_discover_pyramid() {
        let data_dir = test_data_dir();
        if !data_dir.exists() {
            eprintln!("Skipping: test data not found");
            return;
        }
        let info = discover_pyramid(&data_dir, 256).unwrap();
        assert_eq!(info.tile_size, 256);
        assert_eq!(info.max_level, 16);
        assert_eq!(info.l0, 16);
        assert_eq!(info.l1, 15);
        assert_eq!(info.l2, 14);
        assert!(info.files_dir.exists());
    }

    #[test]
    fn test_discover_pyramid_missing_dir() {
        let result = discover_pyramid(std::path::Path::new("/nonexistent/path"), 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_tile_plane() {
        // 4x4 plane with known values
        let plane: Vec<u8> = (0..16).collect();
        let mut out = vec![0u8; 4];
        // Extract 2x2 tile at (1, 1)
        extract_tile_plane(&plane, 4, 4, 1, 1, 2, 2, &mut out);
        assert_eq!(out, vec![5, 6, 9, 10]);
    }

    #[test]
    fn test_extract_tile_plane_boundary() {
        // Extract at boundary — should not panic, out-of-bound pixels stay 0
        let plane: Vec<u8> = (0..16).collect();
        let mut out = vec![0u8; 4];
        extract_tile_plane(&plane, 4, 4, 3, 3, 2, 2, &mut out);
        // Only (3,3) is in bounds
        assert_eq!(out[0], 15);
        assert_eq!(out[1], 0); // out of x range
        assert_eq!(out[2], 0); // out of y range
        assert_eq!(out[3], 0); // out of both
    }

    #[test]
    fn test_copy_tile_into_mosaic() {
        // 2x2 tile of red pixels (R=255, G=0, B=0)
        let tile = vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
        // 4x4 mosaic (2x2 grid of 2x2 tiles) initialized to black
        let mut mosaic = vec![0u8; 4 * 4 * 3];

        // Place tile at grid position (1, 0) — top-right
        copy_tile_into_mosaic(&tile, &mut mosaic, 4, 2, 1, 0);

        // Check top-left 2x2 is still black
        assert_eq!(mosaic[0], 0);
        // Check top-right 2x2 has red
        let off = 2 * 3; // x=2, y=0
        assert_eq!(mosaic[off], 255);
        assert_eq!(mosaic[off + 1], 0);
        assert_eq!(mosaic[off + 2], 0);
    }

    #[test]
    fn test_copy_tile_into_mosaic_roundtrip() {
        // Create 4 distinct 2x2 tiles, place them into a 4x4 mosaic, verify placement
        let tile_size = 2u32;
        let mosaic_w = tile_size * 2;
        let mut mosaic = vec![0u8; (mosaic_w * mosaic_w * 3) as usize];

        for idx in 0..4u32 {
            let dx = idx % 2;
            let dy = idx / 2;
            let val = ((idx + 1) * 50) as u8; // 50, 100, 150, 200
            let tile = vec![val; (tile_size * tile_size * 3) as usize];
            copy_tile_into_mosaic(&tile, &mut mosaic, mosaic_w, tile_size, dx, dy);
        }

        // Check each quadrant has the right value
        // Top-left (dx=0,dy=0): val=50
        assert_eq!(mosaic[0], 50);
        // Top-right (dx=1,dy=0): val=100
        assert_eq!(mosaic[2 * 3], 100);
        // Bottom-left (dx=0,dy=1): val=150
        assert_eq!(mosaic[2 * (mosaic_w * 3) as usize], 150);
        // Bottom-right (dx=1,dy=1): val=200
        assert_eq!(mosaic[2 * (mosaic_w * 3) as usize + 2 * 3], 200);
    }
}
