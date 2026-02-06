# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ORIGAMI (Residual-Pyramid Image Processor) is a high-performance Rust tile server for serving Deep Zoom Image (DZI) pyramids with dynamic reconstruction of high-resolution tiles using residual compression techniques. The project is designed to efficiently serve whole-slide images (WSI) with reduced storage requirements.

This project also includes Python-based CLI tools managed with UV (Python package and project manager) for preprocessing whole-slide images and generating residual pyramids.

## Architecture

### Core Components

The system implements a tile server with the following key features:

1. **Tile Reconstruction Pipeline**: Dynamically reconstructs L0/L1 tiles from L2 tiles plus luma-only residuals
2. **Dual-tier Cache**:
   - Hot cache: In-memory LRU for encoded JPEG bytes
   - Warm cache: RocksDB persistent storage
3. **Family Generation**: When any L0/L1 tile is requested, generates entire L2 family (20 tiles total: 4 L1 + 16 L0)
4. **Singleflight Control**: Prevents duplicate reconstruction of the same tile family under concurrent requests

### Directory Structure

```
slides/{slide_id}/
  baseline_pyramid.dzi          # DZI manifest
  baseline_pyramid_files/        # Standard pyramid tiles
    {level}/
      {x}_{y}.jpg
  residuals_j{quality}/          # Residual tiles for reconstruction
    L1/{x2}_{y2}/{x1}_{y1}.jpg
    L0/{x2}_{y2}/{x0}_{y0}.jpg
```

### API Endpoints

- `GET /dzi/{slide_id}.dzi` - DZI manifest
- `GET /tiles/{slide_id}/{level}/{x}_{y}.jpg` - Tile data
- `GET /viewer/{slide_id}` - OpenSeadragon viewer
- `GET /healthz` - Health check
- `GET /metrics` - Prometheus metrics

## Development Commands

### Building the Server

```bash
# Build the Rust server
cd server
cargo build --release

# Run in development mode
cargo run

# Run tests
cargo test

# Run with specific configuration
RUST_LOG=debug cargo run -- --config config.toml
```

### Python CLI Tool (UV)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Run the WSI residual tool
uv run wsi-residual-tool --input /path/to/slide.svs --output /path/to/output --quality 32

# Or activate the virtual environment and run directly
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
wsi-residual-tool --input /path/to/slide.svs --output /path/to/output --quality 32

# Install in development mode
uv pip install -e .
```

### Performance Testing

```bash
# Run benchmarks
cargo bench

# Profile with flamegraph (requires cargo-flamegraph)
cargo flamegraph --bin origami-server

# Load testing with vegeta (example)
echo "GET http://localhost:8080/tiles/slide1/10/5_3.jpg" | vegeta attack -duration=30s | vegeta report
```

## Key Implementation Details

### Tile Coordinate Mapping

- Deep Zoom levels: 0 (smallest) to N (highest resolution)
- L0 = N (highest-res), L1 = N-1, L2 = N-2
- Parent calculation:
  - L1 tile (x,y) → L2 parent (x>>1, y>>1)
  - L0 tile (x,y) → L2 parent (x>>2, y>>2)

### Reconstruction Algorithm

1. Decode L2 baseline JPEG → RGB/YCbCr
2. Upsample L2 (2×) → L1 prediction mosaic (256×256 → 512×512)
3. For each L1 tile:
   - Decode residual grayscale JPEG
   - Apply: Y_recon = clamp(Y_pred + (R - 128), 0..255)
   - Reuse predicted chroma (Cb/Cr)
4. Build L1 mosaic → Upsample (2×) → L0 prediction
5. Apply L0 residuals similarly

### Performance Targets

- Hot cache hit: P95 < 10ms
- Warm cache hit: P95 < 25ms
- Cold miss (generate family): P95 < 200-500ms

### RocksDB Configuration

- Key format: `tile:{slide_id}:{level}:{x}:{y}`
- WriteBatch for family writes (20 tiles)
- Optional: `disableWAL=true` for performance
- Consider TTL or manual eviction strategy

## Dependencies

### Core Rust Crates

- **axum** or **actix-web**: HTTP server framework
- **rocksdb**: Persistent cache storage
- **turbojpeg**: JPEG encoding/decoding (via libjpeg-turbo bindings)
- **lru**: In-memory cache
- **tokio**: Async runtime
- **serde**: Serialization
- **tracing**: Structured logging
- **prometheus**: Metrics

### Python Dependencies (UV managed)

- **pyvips**: Python bindings for libvips image processing
- **Pillow**: Python Imaging Library
- **numpy**: Numerical computing
- **scikit-image**: Image processing algorithms
- **openslide-python**: Python interface to OpenSlide for WSI reading
- **requests**: HTTP library
- **tqdm**: Progress bars

### External Requirements

- libjpeg-turbo: System library for fast JPEG operations
- OpenSeadragon: Client-side viewer (vendored or CDN)
- UV: Modern Python package and project manager
- libvips: Image processing library (for pyvips)
- OpenSlide: C library for reading whole slide images

## Testing Strategy

1. **Unit Tests**: Coordinate mapping, residual math, cache logic
2. **Integration Tests**: Full reconstruction pipeline, API endpoints
3. **Quality Tests**: PSNR/ΔE00 comparison with reference implementation
4. **Load Tests**: Concurrent request handling, cache stampede prevention
5. **Persistence Tests**: RocksDB restart behavior

## Debugging Tips

- Enable debug logging: `RUST_LOG=debug`
- Check metrics endpoint for cache hit rates
- Monitor RocksDB stats via metrics
- Use request IDs in logs for tracing
- Profile hotspots with perf/flamegraph for optimization