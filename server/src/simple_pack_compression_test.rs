// Simple test to verify pack file compression approach
// The pack file is accessed as a WHOLE UNIT when any tile is needed

use anyhow::Result;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    // Load a test pack file
    let pack_path = Path::new("test-data/demo_out/residual_packs/50_50.pack");

    if !pack_path.exists() {
        eprintln!("Test pack file not found at {:?}", pack_path);
        return Ok(());
    }

    // Read the entire pack file
    let original_data = fs::read(pack_path)?;
    println!("Original pack size: {} bytes", original_data.len());

    // Compress the ENTIRE pack file as one unit
    let compress_start = Instant::now();
    let compressed = compress_prepend_size(&original_data);
    let compress_time = compress_start.elapsed();
    println!("Compressed size: {} bytes", compressed.len());
    println!("Compression ratio: {:.2}x", original_data.len() as f64 / compressed.len() as f64);
    println!("Compression time: {:.3}ms", compress_time.as_secs_f64() * 1000.0);
    println!("Space savings: {:.1}%", 100.0 * (1.0 - compressed.len() as f64 / original_data.len() as f64));

    // Decompress the entire pack
    let decompress_start = Instant::now();
    let decompressed = decompress_size_prepended(&compressed)?;
    let decompress_time = decompress_start.elapsed();
    println!("Decompression time: {:.3}ms", decompress_time.as_secs_f64() * 1000.0);

    // Verify correctness
    assert_eq!(original_data, decompressed);
    println!("✓ Decompression verified correct");

    // Run multiple decompression iterations for accurate timing
    let iterations = 1000;
    let bench_start = Instant::now();
    for _ in 0..iterations {
        let _ = decompress_size_prepended(&compressed)?;
    }
    let total_time = bench_start.elapsed();
    let avg_decompress_time = total_time.as_secs_f64() * 1000.0 / iterations as f64;
    println!("\nAverage decompression time ({} iterations): {:.3}ms", iterations, avg_decompress_time);
    println!("Decompression throughput: {:.1} MB/s",
        original_data.len() as f64 / (avg_decompress_time / 1000.0) / 1_000_000.0);

    Ok(())
}