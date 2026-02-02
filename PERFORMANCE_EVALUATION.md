# RRIP Performance and Compression Evaluation

## Executive Summary

RRIP (Residual Reconstruction from Interpolated Priors) achieves **6.7x compression** (85.1% storage reduction) on whole-slide images while maintaining high-performance tile serving at **278 pack files/second** throughput, generating 5,560 tiles/second total.

## Compression Results

### Original Baseline Storage
- **L0 tiles (level 16)**: 484,079,351 bytes (484.1 MB)
- **L1 tiles (level 15)**: 155,049,778 bytes (155.0 MB)
- **L0+L1 total**: 639,129,129 bytes (639.1 MB)
- **L2+ tiles**: 62,128,167 bytes (62.1 MB)
- **Total baseline**: 701,257,296 bytes (701.3 MB)

### RRIP Compressed Storage
- **L2+ baseline tiles**: 62,128,167 bytes (62.1 MB) - retained as-is
- **LZ4 pack files**: 42,581,763 bytes (42.6 MB) - replacing all L0+L1 tiles
- **Total RRIP storage**: 104,709,930 bytes (104.7 MB)

### Compression Achievement
- **Overall compression ratio**: 701.3 MB → 104.7 MB = **6.7x reduction**
- **L0+L1 specific compression**: 639.1 MB → 42.6 MB = **15.0x reduction**
- **Storage savings**: **85.1% reduction**

## Performance Results

### Peak Performance (128 concurrent connections)
- **Pack file throughput**: 278 pack files/second
- **Tile generation**: 5,560 tiles/second (278 packs × 20 tiles per pack)
- **Success rate**: 100% across all concurrency levels
- **LZ4 decompression rate**: ~3.0 MB/sec

### Latency Characteristics
| Concurrency | Median Latency | P95 Latency | Pack Files/sec |
|------------|----------------|-------------|----------------|
| 1          | 6.1 ms         | 8.9 ms      | 20             |
| 2          | 8.8 ms         | 108.2 ms    | 40             |
| 4          | 15.0 ms        | 122.7 ms    | 80             |
| 8          | 112.5 ms       | 135.0 ms    | 89             |
| 16         | 128.3 ms       | 164.0 ms    | 154            |
| 32         | 146.8 ms       | 191.6 ms    | 244            |
| 64         | 131.9 ms       | 733.1 ms    | 271            |
| 128        | 367.3 ms       | 528.0 ms    | 278            |

### Scaling Characteristics
- Linear scaling up to ~32 concurrent connections
- Saturates at ~278 pack files/sec with 128 concurrent connections
- Each pack file contains 20 tiles (4 L1 + 16 L0) from same L2 parent

## Replication Instructions

### Prerequisites
```bash
# Install Python dependencies
pip install pyvips Pillow numpy scikit-image openslide-python lz4

# Ensure system libraries are installed
# macOS: brew install libvips openslide libjpeg-turbo
# Linux: apt-get install libvips openslide-tools libturbojpeg

# Build the RRIP tile server
cd server
TURBOJPEG_SOURCE=explicit \
TURBOJPEG_DYNAMIC=1 \
TURBOJPEG_LIB_DIR=/opt/homebrew/lib \
TURBOJPEG_INCLUDE_DIR=/opt/homebrew/include \
cargo build --release
```

### Step 1: Generate Baseline Pyramid
```bash
python cli/wsi_residual_tool.py build \
  --slide /path/to/slide.svs \
  --out data/demo_out \
  --tile 256 \
  --q 90
```

### Step 2: Generate Residuals (ALL tiles, not sampled)
```bash
# IMPORTANT: Do not use --max-parents to ensure ALL tiles are processed
python cli/wsi_residual_tool.py encode \
  --pyramid data/demo_out/baseline_pyramid \
  --out data/demo_out \
  --tile 256 \
  --resq 32
```

### Step 3: Create LZ4-Compressed Pack Files
```bash
python cli/wsi_residual_tool.py pack \
  --residuals data/demo_out/residuals_q32 \
  --out data/demo_out/residual_packs_lz4
```

### Step 4: Start RRIP Tile Server
```bash
# Start server with LZ4 pack directory
RUST_LOG=info /tmp/rrip-build/release/rrip-tile-server \
  --slides-root data \
  --port 3007 \
  --pack-dir residual_packs_lz4
```

### Step 5: Run Performance Tests
```bash
# Comprehensive performance test
python comprehensive_perf_test.py

# The test will:
# - Test concurrency levels: 1, 2, 4, 8, 16, 32, 64, 128
# - Run each level for 30 seconds
# - Measure tiles/sec, latencies, memory, CPU
# - Save results to performance_results.json
```

## Key Technical Details

### Pack File Structure
- Each pack file corresponds to one L2 parent tile
- Contains 20 tiles total:
  - 4 L1 tiles (2×2 grid)
  - 16 L0 tiles (4×4 grid)
- Pack files use LZ4 compression with size-prepended format
- Average compressed pack size: ~11 KB
- Average uncompressed pack size: ~60 KB

### Residual Encoding
- Luma-only residuals (grayscale JPEG)
- Quality setting: 32 for residuals
- Chroma channels reused from upsampled predictions
- Residuals stored as: `encoded = clip(Y_actual - Y_predicted + 128, 0, 255)`

### Reconstruction Process
1. Load L2 baseline tile (256×256)
2. Upsample 2× → L1 prediction (512×512)
3. Apply L1 residuals to get L1 tiles
4. Upsample L1 mosaic 2× → L0 prediction (1024×1024)
5. Apply L0 residuals to get L0 tiles
6. All 20 tiles cached after generation

## Validation Commands

### Check Compression Ratios
```bash
# Original L0+L1 size
find data/demo_out/baseline_pyramid_files/16 -name "*.jpg" -exec stat -f%z {} \; | awk '{sum += $1} END {print sum}'
find data/demo_out/baseline_pyramid_files/15 -name "*.jpg" -exec stat -f%z {} \; | awk '{sum += $1} END {print sum}'

# Compressed pack file size
find data/demo_out/residual_packs_lz4 -name "*.pack" -exec stat -f%z {} \; | awk '{sum += $1} END {print sum}'
```

### Test Individual Tile Serving
```bash
# Test L0 tile request (triggers family generation)
curl -o test.jpg "http://localhost:3007/tiles/demo_out/16/100_100.jpg"
ls -lh test.jpg  # Should be ~14KB JPEG
```

### Monitor Server Performance
```bash
# Watch server logs during testing
tail -f server.log | grep -E "(tiles_per_second|family_generated|cache_hit)"
```

## Notes

- The --max-parents parameter in wsi_residual_tool.py defaults to None (process all tiles)
- Earlier tests with --max-parents=200 only processed 6% of tiles, leading to incorrect metrics
- Memory usage reported by the test script may be inaccurate (shows 5.7MB consistently)
- Server implements singleflight control to prevent duplicate family generation
- Dual-tier cache: in-memory LRU + RocksDB persistent storage