# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ORIGAMI (Residual-Pyramid Image Processor) is a high-performance Rust tile server and residual encoder for serving Deep Zoom Image (DZI) pyramids with dynamic reconstruction of high-resolution tiles using residual compression techniques. The project is designed to efficiently serve whole-slide images (WSI) with reduced storage requirements.

The `origami` binary provides two subcommands:
- **`origami serve`** — Tile server with dynamic reconstruction and caching
- **`origami encode`** — Residual encoder that generates luma residuals from a DZI pyramid

This project also includes Python-based CLI tools managed with UV (Python package and project manager) for preprocessing whole-slide images and generating residual pyramids.

## Architecture

### Binary Structure

The `origami` binary is a thin clap dispatcher (`server/src/main.rs`) with subcommands:
- `server/src/serve.rs` — All tile server logic
- `server/src/encode.rs` — Residual encoding pipeline

### Shared Core Modules (`server/src/core/`)

| Module | Purpose |
|--------|---------|
| `color.rs` | Fixed-point BT.601 YCbCr conversion (10-bit precision) |
| `residual.rs` | Residual computation and centering |
| `pack.rs` | Pack file format (read/write, "ORIG"/"RRIP" magic) |
| `upsample.rs` | 2x bilinear upsampling |
| `jpeg.rs` | `JpegEncoder` trait + backends (TurboJpeg, MozJpeg, Jpegli) |
| `libjpeg_ffi.rs` | Safe FFI wrapper for libjpeg62 compress API (conditional) |

### JPEG Encoder Backends

The `JpegEncoder` trait in `core/jpeg.rs` provides a unified interface. Three backends are available:

| Encoder | Library | Build | Feature Flag |
|---------|---------|-------|-------------|
| **turbojpeg** | libjpeg-turbo (via `turbojpeg` crate) | Always available | (default) |
| **mozjpeg** | Pre-built `libmozjpeg.a` | `scripts/build_mozjpeg.sh` | `--features mozjpeg` |
| **jpegli** | Pre-built `libjpegli-static.a` | `scripts/build_jpegli.sh` | `--features jpegli` |

- mozjpeg and jpegli are **mutually exclusive** (both define libjpeg62 symbols)
- mozjpeg and jpegli share `libjpeg_ffi.rs` + `csrc/libjpeg_compress.c` (libjpeg62 C API)
- `build.rs` compiles the C wrapper and links via `cargo:rustc-link-arg`

### Tile Server Components

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

### API Endpoints (serve subcommand)

- `GET /dzi/{slide_id}.dzi` - DZI manifest
- `GET /tiles/{slide_id}/{level}/{x}_{y}.jpg` - Tile data
- `GET /viewer/{slide_id}` - OpenSeadragon viewer
- `GET /healthz` - Health check
- `GET /metrics` - Prometheus metrics

## Development Commands

### Building the Server (turbojpeg only, default)

```bash
cd server
cargo build --release
```

### Building with mozjpeg Encoder

```bash
# 1. Build mozjpeg static library (one-time)
./scripts/build_mozjpeg.sh

# 2. Build origami with mozjpeg support
cd server
cargo build --release --features mozjpeg
```

### Building with jpegli Encoder

```bash
# 1. Install prerequisites (macOS)
brew install llvm coreutils cmake giflib jpeg-turbo libpng ninja zlib

# 2. Build jpegli static library (one-time)
./scripts/build_jpegli.sh

# 3. Build origami with jpegli support
cd server
cargo build --release --features jpegli
```

### Running

```bash
# Start the tile server
origami serve --slides-root /path/to/slides --port 3007

# Encode residuals from a DZI pyramid
origami encode --pyramid /path/to/slide --out /path/to/output --resq 50

# Encode with a specific backend
origami encode --pyramid /path/to/slide --out /path/to/output --resq 50 --encoder mozjpeg

# Also generate pack files
origami encode --pyramid /path/to/slide --out /path/to/output --resq 50 --pack
```

### Using System libjpeg-turbo

To use Homebrew's libjpeg-turbo instead of building from source:

```bash
brew install libjpeg-turbo
TURBOJPEG_SOURCE=pkg-config cargo build --release
```

### Running Tests

```bash
cd server
cargo test --bins                     # Unit tests (no features)
cargo test --bins --features mozjpeg  # Unit tests with mozjpeg
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
source .venv/bin/activate
wsi-residual-tool --input /path/to/slide.svs --output /path/to/output --quality 32
```

### Evaluation & Comparison Viewer

All evaluation infrastructure lives under `evals/`. See `evals/README.md` for full details.

```bash
# Generate an ORIGAMI run
python evals/scripts/wsi_residual_debug_with_manifest.py \
    --image evals/test-images/L0-1024.jpg --resq 50 --pac

# Generate a JPEG baseline
uv run python evals/scripts/jpeg_baseline.py \
    --image evals/test-images/L0-1024.jpg --quality 70

# Batch runs
bash evals/scripts/run_all_captures.sh
bash evals/scripts/run_jpegli_captures.sh

# Start the comparison viewer
cd evals/viewer && pnpm install && pnpm start  # http://localhost:8084
```

### Performance Testing

```bash
cargo bench
cargo flamegraph --bin origami
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
2. Upsample L2 (2x) → L1 prediction mosaic (256x256 → 512x512)
3. For each L1 tile:
   - Decode residual grayscale JPEG
   - Apply: Y_recon = clamp(Y_pred + (R - 128), 0..255)
   - Reuse predicted chroma (Cb/Cr)
4. Build L1 mosaic → Upsample (2x) → L0 prediction
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

- **axum**: HTTP server framework
- **turbojpeg**: JPEG encoding/decoding (via libjpeg-turbo)
- **moka**: In-memory cache (hot cache)
- **tokio**: Async runtime
- **rayon**: Parallel computation
- **clap**: CLI argument parsing (subcommands)
- **serde** / **serde_json**: Serialization
- **tracing**: Structured logging
- **libc**: FFI support for libjpeg62 bindings

### Optional Vendor Libraries (pre-built, not Rust crates)

- **mozjpeg**: Better JPEG compression via trellis quantization (`vendor/mozjpeg/`)
- **jpegli**: ~35% better compression than libjpeg-turbo (`vendor/jpegli/`)

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

1. **Unit Tests**: Coordinate mapping, residual math, cache logic, pack format
2. **Integration Tests**: Full reconstruction pipeline, API endpoints
3. **Quality Tests**: PSNR/dE00 comparison with reference implementation
4. **Load Tests**: Concurrent request handling, cache stampede prevention
5. **Encoder A/B Tests**: Compare JPEG sizes and quality across turbojpeg/mozjpeg/jpegli

## Debugging Tips

- Enable debug logging: `RUST_LOG=debug`
- Check metrics endpoint for cache hit rates
- Monitor RocksDB stats via metrics
- Use request IDs in logs for tracing
- Profile hotspots with perf/flamegraph for optimization