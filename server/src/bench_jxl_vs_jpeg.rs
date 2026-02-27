//! Benchmark: JPEGXL vs JPEG for grayscale residual decode
//!
//! Uses existing JPEG residual files from eval runs.
//! Re-encodes them as JXL at multiple quality levels, then benchmarks decode speed.
//!
//! Usage:
//!   cargo build --release --features jpegxl --bin bench_jxl_vs_jpeg
//!   ./target/release/bench_jxl_vs_jpeg <run_dir> [quality_levels...]
//!
//! Example:
//!   ./target/release/bench_jxl_vs_jpeg ../evals/runs/b95_l1q80_l0q60_optl2_d20
//!   ./target/release/bench_jxl_vs_jpeg ../evals/runs/b95_l1q80_l0q60_optl2_d20 40 60 80

use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <run_dir> [quality_levels...]", args[0]);
        eprintln!("Example: {} ../evals/runs/b95_l1q80_l0q60_optl2_d20 40 60 80", args[0]);
        std::process::exit(1);
    }

    let run_dir = PathBuf::from(&args[1]);
    let quality_levels: Vec<u8> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().expect("invalid quality")).collect()
    } else {
        vec![30, 40, 50, 60, 70, 80, 90]
    };

    // Collect all residual JPEGs from L0/ and L1/ dirs
    let mut residual_files: Vec<PathBuf> = Vec::new();
    for level in &["L0", "L1"] {
        let level_dir = run_dir.join(level);
        if !level_dir.exists() { continue; }
        collect_jpgs(&level_dir, &mut residual_files);
    }

    if residual_files.is_empty() {
        eprintln!("No .jpg residual files found in {}", run_dir.display());
        std::process::exit(1);
    }

    residual_files.sort();
    println!("Found {} residual JPEG files", residual_files.len());
    println!();

    // Load and decode all residuals to raw grayscale pixels
    let mut tiles: Vec<TileData> = Vec::new();
    for path in &residual_files {
        let jpeg_bytes = std::fs::read(path).expect("read file");
        let (pixels, w, h) = decode_jpeg_gray(&jpeg_bytes);
        let name = path.strip_prefix(&run_dir).unwrap_or(path).display().to_string();
        tiles.push(TileData { name, jpeg_bytes, pixels, width: w, height: h });
    }

    // Print JPEG baseline sizes
    let total_jpeg_bytes: usize = tiles.iter().map(|t| t.jpeg_bytes.len()).sum();
    println!("=== JPEG Baseline (existing files) ===");
    println!("  Total: {} bytes ({:.1} KB)", total_jpeg_bytes, total_jpeg_bytes as f64 / 1024.0);
    println!("  Avg per tile: {} bytes ({:.1} KB)", total_jpeg_bytes / tiles.len(), total_jpeg_bytes as f64 / tiles.len() as f64 / 1024.0);
    println!();

    // Benchmark JPEG decode speed
    let jpeg_decode_us = bench_jpeg_decode(&tiles);
    println!("=== JPEG Decode Benchmark ({} tiles) ===", tiles.len());
    println!("  Total: {:.0} us ({:.2} ms)", jpeg_decode_us, jpeg_decode_us / 1000.0);
    println!("  Per tile: {:.0} us", jpeg_decode_us / tiles.len() as f64);
    println!();

    // For each quality level, re-encode as JXL and benchmark
    println!("=== JPEGXL Comparison ===");
    println!("{:<8} {:>12} {:>12} {:>10} {:>14} {:>14} {:>10}",
        "Quality", "JXL Total", "JPEG Total", "Savings", "JXL Decode", "JPEG Decode", "Slowdown");
    println!("{}", "-".repeat(82));

    for &q in &quality_levels {
        // Encode all tiles as JXL at this quality
        let mut jxl_blobs: Vec<Vec<u8>> = Vec::new();
        for tile in &tiles {
            let jxl = encode_jxl_gray(&tile.pixels, tile.width, tile.height, q);
            jxl_blobs.push(jxl);
        }

        let total_jxl_bytes: usize = jxl_blobs.iter().map(|b| b.len()).sum();
        let savings = 1.0 - (total_jxl_bytes as f64 / total_jpeg_bytes as f64);

        // Benchmark JXL decode
        let jxl_decode_us = bench_jxl_decode(&jxl_blobs);

        let slowdown = jxl_decode_us / jpeg_decode_us;

        println!("{:<8} {:>9} KB {:>9} KB {:>9.1}% {:>11.0} us {:>11.0} us {:>9.1}x",
            q,
            format!("{:.1}", total_jxl_bytes as f64 / 1024.0),
            format!("{:.1}", total_jpeg_bytes as f64 / 1024.0),
            savings * 100.0,
            jxl_decode_us,
            jpeg_decode_us,
            slowdown,
        );
    }

    println!();

    // Fair head-to-head: re-encode same pixels as BOTH JPEG and JXL at each quality
    println!();
    println!("=== Fair Head-to-Head (same pixels, same quality) ===");
    println!("{:<8} {:>12} {:>12} {:>10} {:>14} {:>14} {:>10}",
        "Quality", "JXL Total", "JPEG Total", "JXL Saves", "JXL Decode", "JPEG Decode", "Slowdown");
    println!("{}", "-".repeat(82));

    for &q in &quality_levels {
        let mut jxl_blobs: Vec<Vec<u8>> = Vec::new();
        let mut jpeg_blobs: Vec<Vec<u8>> = Vec::new();
        for tile in &tiles {
            jxl_blobs.push(encode_jxl_gray(&tile.pixels, tile.width, tile.height, q));
            jpeg_blobs.push(encode_jpeg_gray(&tile.pixels, tile.width, tile.height, q));
        }

        let total_jxl: usize = jxl_blobs.iter().map(|b| b.len()).sum();
        let total_jpeg: usize = jpeg_blobs.iter().map(|b| b.len()).sum();
        let savings = 1.0 - (total_jxl as f64 / total_jpeg as f64);

        let jxl_us = bench_jxl_decode(&jxl_blobs);
        let jpeg_us = bench_jpeg_decode_blobs(&jpeg_blobs);
        let slowdown = jxl_us / jpeg_us;

        println!("{:<8} {:>9} KB {:>9} KB {:>9.1}% {:>11.0} us {:>11.0} us {:>9.1}x",
            q,
            format!("{:.1}", total_jxl as f64 / 1024.0),
            format!("{:.1}", total_jpeg as f64 / 1024.0),
            savings * 100.0,
            jxl_us, jpeg_us, slowdown,
        );
    }
    println!();

    // Detailed per-tile comparison at a representative quality
    let repr_q = if quality_levels.contains(&60) { 60 } else { quality_levels[quality_levels.len() / 2] };
    println!("=== Per-Tile Detail (quality={}) ===", repr_q);
    println!("{:<30} {:>10} {:>10} {:>10}", "Tile", "JPEG", "JXL", "Savings");
    println!("{}", "-".repeat(62));

    for tile in &tiles {
        let jxl = encode_jxl_gray(&tile.pixels, tile.width, tile.height, repr_q);
        let savings = 1.0 - (jxl.len() as f64 / tile.jpeg_bytes.len() as f64);
        println!("{:<30} {:>7} B {:>7} B {:>9.1}%",
            tile.name,
            tile.jpeg_bytes.len(),
            jxl.len(),
            savings * 100.0,
        );
    }
}

struct TileData {
    name: String,
    jpeg_bytes: Vec<u8>,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

fn collect_jpgs(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_jpgs(&path, out);
            } else if path.extension().map_or(false, |e| e == "jpg") {
                out.push(path);
            }
        }
    }
}

fn encode_jpeg_gray(pixels: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    let mut compressor = turbojpeg::Compressor::new().expect("create compressor");
    compressor.set_quality(quality as i32);
    compressor.set_subsamp(turbojpeg::Subsamp::Gray);
    let image = turbojpeg::Image {
        pixels,
        width: width as usize,
        pitch: width as usize,
        height: height as usize,
        format: turbojpeg::PixelFormat::GRAY,
    };
    compressor.compress_to_vec(image).expect("compress")
}

fn bench_jpeg_decode_blobs(blobs: &[Vec<u8>]) -> f64 {
    let warmup = 20;
    let iters = 100;
    let mut decompressor = turbojpeg::Decompressor::new().expect("create decompressor");

    for _ in 0..warmup {
        for blob in blobs {
            let header = decompressor.read_header(blob).expect("header");
            let mut buf = vec![0u8; header.width * header.height];
            let image = turbojpeg::Image {
                pixels: buf.as_mut_slice(),
                width: header.width,
                pitch: header.width,
                height: header.height,
                format: turbojpeg::PixelFormat::GRAY,
            };
            decompressor.decompress(blob, image).expect("decompress");
            std::hint::black_box(&buf);
        }
    }

    let start = Instant::now();
    for _ in 0..iters {
        for blob in blobs {
            let header = decompressor.read_header(blob).expect("header");
            let mut buf = vec![0u8; header.width * header.height];
            let image = turbojpeg::Image {
                pixels: buf.as_mut_slice(),
                width: header.width,
                pitch: header.width,
                height: header.height,
                format: turbojpeg::PixelFormat::GRAY,
            };
            decompressor.decompress(blob, image).expect("decompress");
            std::hint::black_box(&buf);
        }
    }
    let elapsed = start.elapsed();
    elapsed.as_micros() as f64 / iters as f64
}

fn decode_jpeg_gray(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let mut decompressor = turbojpeg::Decompressor::new().expect("create decompressor");
    let header = decompressor.read_header(bytes).expect("read header");
    let mut buf = vec![0u8; header.width * header.height];
    let image = turbojpeg::Image {
        pixels: buf.as_mut_slice(),
        width: header.width,
        pitch: header.width,
        height: header.height,
        format: turbojpeg::PixelFormat::GRAY,
    };
    decompressor.decompress(bytes, image).expect("decompress");
    (buf, header.width as u32, header.height as u32)
}

fn bench_jpeg_decode(tiles: &[TileData]) -> f64 {
    let warmup = 20;
    let iters = 100;
    let mut decompressor = turbojpeg::Decompressor::new().expect("create decompressor");

    // Warmup
    for _ in 0..warmup {
        for tile in tiles {
            let header = decompressor.read_header(&tile.jpeg_bytes).expect("header");
            let mut buf = vec![0u8; header.width * header.height];
            let image = turbojpeg::Image {
                pixels: buf.as_mut_slice(),
                width: header.width,
                pitch: header.width,
                height: header.height,
                format: turbojpeg::PixelFormat::GRAY,
            };
            decompressor.decompress(&tile.jpeg_bytes, image).expect("decompress");
            std::hint::black_box(&buf);
        }
    }

    // Timed
    let start = Instant::now();
    for _ in 0..iters {
        for tile in tiles {
            let header = decompressor.read_header(&tile.jpeg_bytes).expect("header");
            let mut buf = vec![0u8; header.width * header.height];
            let image = turbojpeg::Image {
                pixels: buf.as_mut_slice(),
                width: header.width,
                pitch: header.width,
                height: header.height,
                format: turbojpeg::PixelFormat::GRAY,
            };
            decompressor.decompress(&tile.jpeg_bytes, image).expect("decompress");
            std::hint::black_box(&buf);
        }
    }
    let elapsed = start.elapsed();
    elapsed.as_micros() as f64 / iters as f64
}

#[cfg(feature = "jpegxl")]
fn encode_jxl_gray(pixels: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use jpegxl_rs::encode::{encoder_builder, ColorEncoding, EncoderFrame};
    let mut enc = encoder_builder()
        .jpeg_quality(quality as f32)
        .color_encoding(ColorEncoding::SrgbLuma)
        .build()
        .expect("build encoder");
    let result: jpegxl_rs::encode::EncoderResult<u8> = enc.encode_frame(
        &EncoderFrame::new(pixels).num_channels(1),
        width, height,
    ).expect("encode");
    result.data
}

#[cfg(not(feature = "jpegxl"))]
fn encode_jxl_gray(_pixels: &[u8], _width: u32, _height: u32, _quality: u8) -> Vec<u8> {
    panic!("Build with --features jpegxl to run this benchmark");
}

#[cfg(feature = "jpegxl")]
fn bench_jxl_decode(blobs: &[Vec<u8>]) -> f64 {
    use jpegxl_rs::decode::{decoder_builder, PixelFormat};
    use jpegxl_rs::Endianness;

    let warmup = 20;
    let iters = 100;

    // Warmup
    for _ in 0..warmup {
        for blob in blobs {
            let decoder = decoder_builder()
                .pixel_format(PixelFormat {
                    num_channels: 1,
                    endianness: Endianness::Native,
                    align: 0,
                })
                .build()
                .expect("build decoder");
            let (_meta, pixels): (_, Vec<u8>) = decoder.decode_with::<u8>(blob).expect("decode");
            std::hint::black_box(&pixels);
        }
    }

    // Timed
    let start = Instant::now();
    for _ in 0..iters {
        for blob in blobs {
            let decoder = decoder_builder()
                .pixel_format(PixelFormat {
                    num_channels: 1,
                    endianness: Endianness::Native,
                    align: 0,
                })
                .build()
                .expect("build decoder");
            let (_meta, pixels): (_, Vec<u8>) = decoder.decode_with::<u8>(blob).expect("decode");
            std::hint::black_box(&pixels);
        }
    }
    let elapsed = start.elapsed();
    elapsed.as_micros() as f64 / iters as f64
}

#[cfg(not(feature = "jpegxl"))]
fn bench_jxl_decode(_blobs: &[Vec<u8>]) -> f64 {
    panic!("Build with --features jpegxl to run this benchmark");
}
