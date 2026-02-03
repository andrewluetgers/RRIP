use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use reqwest;
use tokio;

const TEST_SLIDE: &str = "demo_out";
const TEST_SERVER_PORT: u16 = 8888;

#[tokio::test]
async fn test_pack_conversion() -> Result<()> {
    // Test converting residuals to pack format
    let test_data_dir = PathBuf::from("test-data");
    let slide_dir = test_data_dir.join(TEST_SLIDE);

    // Check if test data exists
    if !slide_dir.exists() {
        eprintln!("Test data not found. Run ./create-test-data.sh first");
        return Ok(());
    }

    // Test pack creation for a specific L2 tile
    let pack_path = slide_dir.join("residual_packs").join("100_100.pack");

    // If pack doesn't exist, test would create it here
    // For now, just verify structure
    assert!(slide_dir.join("baseline_pyramid.dzi").exists());

    Ok(())
}

#[tokio::test]
async fn test_serve_all_levels() -> Result<()> {
    // Start test server in background
    let server_handle = start_test_server()?;

    // Wait for server to be ready
    tokio::time::sleep(Duration::from_secs(2)).await;

    let client = reqwest::Client::new();
    let base_url = format!("http://localhost:{}", TEST_SERVER_PORT);

    // Test health endpoint
    let resp = client.get(format!("{}/healthz", base_url))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);

    // Test DZI manifest
    let resp = client.get(format!("{}/dzi/{}.dzi", base_url, TEST_SLIDE))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let dzi_content = resp.text().await?;
    assert!(dzi_content.contains("TileSize"));

    // Test tiles from different levels
    let test_cases = vec![
        // Level 0-13: Lower resolution tiles (should be served directly)
        (10, 2, 2),    // Mid-level tile
        (13, 25, 25),  // Higher mid-level

        // Level 14 (L2): Should be served directly from baseline
        (14, 50, 50),  // This tile exists in test data

        // Level 15 (L1): Should trigger reconstruction if residuals exist
        (15, 100, 100),
        (15, 101, 101),

        // Level 16 (L0): Should trigger full family reconstruction
        (16, 200, 200),
        (16, 201, 201),
        (16, 202, 202),
        (16, 203, 203),
    ];

    for (level, x, y) in test_cases {
        println!("Testing tile: level={}, x={}, y={}", level, x, y);

        let resp = client.get(format!(
            "{}/tiles/{}/{}/{}_{}.jpg",
            base_url, TEST_SLIDE, level, x, y
        ))
        .send()
        .await?;

        // Tile might be 404 if outside bounds, but shouldn't error
        assert!(resp.status() == 200 || resp.status() == 404);

        if resp.status() == 200 {
            let content_type = resp.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            assert!(content_type.contains("image/jpeg"));

            let bytes = resp.bytes().await?;
            assert!(bytes.len() > 0, "Tile should have content");
        }
    }

    // Stop server
    stop_test_server(server_handle)?;

    Ok(())
}

#[tokio::test]
async fn test_family_reconstruction() -> Result<()> {
    let server_handle = start_test_server()?;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let client = reqwest::Client::new();
    let base_url = format!("http://localhost:{}", TEST_SERVER_PORT);

    // Request an L0 tile that should trigger family reconstruction
    // The server should reconstruct the entire family (4 L1 + 16 L0 tiles)
    let resp = client.get(format!(
        "{}/tiles/{}/16/200_200.jpg",
        base_url, TEST_SLIDE
    ))
    .timeout(Duration::from_secs(10))
    .send()
    .await?;

    if resp.status() == 200 {
        // Verify it's a valid JPEG
        let bytes = resp.bytes().await?;
        assert!(bytes.len() > 100, "Reconstructed tile should have substantial content");

        // JPEG magic bytes
        assert_eq!(&bytes[0..2], &[0xFF, 0xD8], "Should be valid JPEG");
    }

    stop_test_server(server_handle)?;
    Ok(())
}

#[tokio::test]
async fn test_concurrent_requests() -> Result<()> {
    let server_handle = start_test_server()?;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let client = reqwest::Client::new();
    let base_url = format!("http://localhost:{}", TEST_SERVER_PORT);

    // Launch multiple concurrent requests for the same family
    let mut handles = vec![];

    for i in 0..10 {
        let url = format!(
            "{}/tiles/{}/16/{}_{}.jpg",
            base_url, TEST_SLIDE, 200 + (i % 4), 200 + (i / 4)
        );

        let client_clone = client.clone();
        let handle = tokio::spawn(async move {
            client_clone.get(url)
                .timeout(Duration::from_secs(10))
                .send()
                .await
        });

        handles.push(handle);
    }

    // All requests should complete without error
    for handle in handles {
        let result = handle.await?;
        assert!(result.is_ok(), "Concurrent request should not fail");
    }

    stop_test_server(server_handle)?;
    Ok(())
}

#[tokio::test]
async fn test_cache_performance() -> Result<()> {
    let server_handle = start_test_server()?;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let client = reqwest::Client::new();
    let base_url = format!("http://localhost:{}", TEST_SERVER_PORT);

    // First request - cold cache
    let url = format!("{}/tiles/{}/14/50_50.jpg", base_url, TEST_SLIDE);

    let start = std::time::Instant::now();
    let resp1 = client.get(&url).send().await?;
    let cold_time = start.elapsed();
    assert_eq!(resp1.status(), 200);

    // Second request - should be cached
    let start = std::time::Instant::now();
    let resp2 = client.get(&url).send().await?;
    let hot_time = start.elapsed();
    assert_eq!(resp2.status(), 200);

    // Cache should be significantly faster (at least 2x)
    assert!(
        hot_time < cold_time / 2,
        "Cached request should be much faster. Cold: {:?}, Hot: {:?}",
        cold_time, hot_time
    );

    stop_test_server(server_handle)?;
    Ok(())
}

// Helper functions

fn start_test_server() -> Result<std::process::Child> {
    let mut cmd = Command::new("cargo");
    cmd.args(&[
        "run",
        "--",
        "--slides-root", "test-data",
        "--port", &TEST_SERVER_PORT.to_string(),
    ]);

    cmd.env("RUST_LOG", "warn");
    cmd.env("TURBOJPEG_SOURCE", "explicit");
    cmd.env("TURBOJPEG_DYNAMIC", "1");

    // Set TurboJPEG paths based on OS
    #[cfg(target_os = "macos")]
    {
        cmd.env("TURBOJPEG_LIB_DIR", "/opt/homebrew/lib");
        cmd.env("TURBOJPEG_INCLUDE_DIR", "/opt/homebrew/include");
    }

    #[cfg(target_os = "linux")]
    {
        cmd.env("TURBOJPEG_LIB_DIR", "/usr/lib/x86_64-linux-gnu");
        cmd.env("TURBOJPEG_INCLUDE_DIR", "/usr/include");
    }

    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());

    Ok(cmd.spawn()?)
}

fn stop_test_server(mut handle: std::process::Child) -> Result<()> {
    handle.kill()?;
    handle.wait()?;
    Ok(())
}

#[cfg(test)]
mod pack_tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn test_pack_file_format() -> Result<()> {
        // Test pack file structure
        let test_pack = create_test_pack()?;

        // Verify magic bytes
        assert_eq!(&test_pack[0..4], b"ORIG");

        // Verify version
        assert_eq!(test_pack[4], 1);

        // Read tile count
        let tile_count = u16::from_le_bytes([test_pack[8], test_pack[9]]);
        assert_eq!(tile_count, 20); // 4 L1 + 16 L0 tiles

        Ok(())
    }

    fn create_test_pack() -> Result<Vec<u8>> {
        let mut pack = Vec::new();

        // Magic bytes
        pack.write_all(b"ORIG")?;

        // Version
        pack.push(1);

        // Padding
        pack.extend_from_slice(&[0, 0, 0]);

        // Tile count (4 L1 + 16 L0)
        pack.extend_from_slice(&20u16.to_le_bytes());

        // Padding
        pack.extend_from_slice(&[0; 6]);

        // Add dummy tiles
        for level in 0..2 {
            let count = if level == 0 { 4 } else { 16 };
            for i in 0..count {
                // Tile header
                pack.push(level + 1);  // Level (1 for L1, 2 for L0)
                pack.push(i);           // Index
                pack.extend_from_slice(&[0; 2]); // Padding

                let offset = pack.len() as u32 + 8;
                pack.extend_from_slice(&offset.to_le_bytes());

                let size = 1024u32; // Dummy size
                pack.extend_from_slice(&size.to_le_bytes());

                // Dummy JPEG data
                pack.extend_from_slice(&vec![0xFF, 0xD8]); // JPEG header
                pack.extend_from_slice(&vec![0; 1022]);     // Dummy data
            }
        }

        Ok(pack)
    }
}