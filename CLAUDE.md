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

### Building with mozjpeg Encoder`

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

### Quality Parameters

When the user specifies quality values like "80, 60, 40 with 444 and delta 20", those map to the ORIGAMI quality parameters in order:

| Position | Parameter | Controls | Default |
|----------|-----------|----------|---------|
| 1st | `--baseq` | L2 baseline JPEG quality | 95 |
| 2nd | `--l1q` | L1 residual JPEG quality | 60 |
| 3rd | `--l0q` | L0 residual JPEG quality | 40 |

Additional encoding parameters:
- `--subsamp` — Chroma subsampling for L2 JPEG: `444`, `420`, `420opt` (default: 444)
- `--optl2` — Enable gradient descent optimization on L2 tiles
- `--max-delta` — Max per-pixel deviation for optL2 (default: 15)

So "80, 60, 40 with 444 and delta 20" means: `--baseq 80 --l1q 60 --l0q 40 --subsamp 444 --optl2 --max-delta 20`

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

#### Generating Runs

```bash
# Generate an ORIGAMI run (Python — includes full metrics: PSNR, SSIM, VIF, Delta E, LPIPS)
python evals/scripts/wsi_residual_debug_with_manifest.py \
    --image evals/test-images/L0-1024.jpg --resq 50 --pac

# Generate an ORIGAMI run (Rust encoder — Y-PSNR/Y-MSE only, needs compute_metrics.py for full metrics)
origami encode --image evals/test-images/L0-1024.jpg --out evals/runs/rs_444_optl2_d20_l1q60_l0q40 \
    --baseq 95 --l1q 60 --l0q 40 --subsamp 444 --optl2 --max-delta 20 --manifest --debug-images

# Generate a JPEG baseline
uv run python evals/scripts/jpeg_baseline.py \
    --image evals/test-images/L0-1024.jpg --quality 70

# Batch runs
bash evals/scripts/run_all_captures.sh
bash evals/scripts/run_jpegli_captures.sh
```

#### Computing Visual Metrics

The Rust encoder (`origami encode`) and GPU encoder only produce Y-PSNR and Y-MSE in their manifest.json. Full visual metrics (SSIM, VIF, Delta E, LPIPS) require `--debug-images` and a post-processing step:

```bash
# Compute full metrics for specific runs (requires compress/ and decompress/ dirs from --debug-images)
uv run python evals/scripts/compute_metrics.py evals/runs/rs_444_optl2_d20_l1q60_l0q40

# Compute for all rs_* runs
uv run python evals/scripts/compute_metrics.py

# Compute for GPU runs
uv run python evals/scripts/compute_metrics.py evals/runs/gpu_444_b90_optl2_d20_l1q60_l0q40
```

This updates manifest.json in-place, adding `decompression_phase` (per-tile PSNR, SSIM, VIF, Delta E, LPIPS) and `size_comparison` fields. Dependencies: scikit-image, sewar (VIF), lpips + torch (LPIPS).

#### Starting the Viewer

```bash
cd evals/viewer && pnpm install && pnpm start  # http://localhost:8084
```

The viewer auto-discovers runs from `evals/runs/` by directory name patterns.

#### Directory Naming Conventions

The viewer server (`viewer-server.js`) auto-discovers runs based on directory names. Each pattern maps to a display name:

| Pattern | Example | Display Name |
|---------|---------|-------------|
| `jpeg_baseline_q{N}` | `jpeg_baseline_q70` | JPEG turbo 70 |
| `jp2_baseline_q{N}` | `jp2_baseline_q50` | JP2 50 |
| `jpegxl_jpeg_baseline_q{N}` | `jpegxl_jpeg_baseline_q60` | JPEG jpegxl 60 |
| `rs_{subsamp}[_optl2]_j{N}` | `rs_444_optl2_j50` | RS ORIGAMI turbo 50 444 optL2 |
| `rs_{subsamp}[_optl2]_l1q{N}_l0q{N}` | `rs_444_optl2_l1q60_l0q40` | RS ORIGAMI turbo L1=60 L0=40 444 optL2 |
| `rs_{subsamp}_optl2_d{N}_l1q{N}_l0q{N}` | `rs_444_optl2_d20_l1q60_l0q40` | RS ORIGAMI turbo L1=60 L0=40 444 optL2 ±20 |
| `rs_{subsamp}_b{N}_optl2_d{N}_l1q{N}_l0q{N}` | `rs_444_b90_optl2_d20_l1q60_l0q40` | RS ORIGAMI turbo B90 L1=60 L0=40 444 optL2 ±20 |
| `gpu_{subsamp}_b{N}[_optl2[_d{N}]]_l1q{N}_l0q{N}` | `gpu_444_b90_optl2_d20_l1q60_l0q40` | GPU nvjpeg B90 L1=60 L0=40 444 optL2 ±20 |
| `optl2_debug_j{N}_pac` | `optl2_debug_j50_pac` | OPTL2 turbo 50 |
| `optl2_debug_l1q{N}_l0q{N}_pac` | `optl2_debug_l1q60_l0q40_pac` | OPTL2 turbo L1=60 L0=40 |

Subsamp values: `444`, `420`, `420opt`. Directories must contain `compress/`, `decompress/`, `images/`, or `manifest.json` to be recognized.

#### Adding a New Run Type to the Viewer

To add a new naming pattern to the viewer:

1. Edit `evals/viewer/viewer-server.js` in the `discoverCaptures()` function
2. Add a new regex match block **before** the fallback block (line ~398)
3. Pattern should parse directory name components and create a captures entry with: `type`, `encoder`, `q`, `j`, `name`, and optional `baseq`, `l1q`, `l0q`, `subsamp`, `optl2`, `delta`
4. Restart the viewer server

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

## RunPod GPU Development

### Setup

- **API key**: stored in env var `RUNPOD_API_KEY`
- **SSH key**: `~/.ssh/id_runpod`
- **Pod image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

### Querying Pods

**Important**: RunPod API uses `Authorization: Bearer` header, NOT `api-key` header.

```bash
# List all pods via RunPod GraphQL API
curl -s -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  --data '{"query":"{ myself { pods { id name desiredStatus machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } gpus { id } } } } }"}' \
  https://api.runpod.io/graphql
```

### SSH Access

```bash
# Connect to a pod (get IP and port from API query above)
ssh -i ~/.ssh/id_runpod root@<IP> -p <PORT>
```

### Syncing Code

**IMPORTANT: Do NOT use rsync.** It copies too many files and is slow/fragile. Instead:

1. **Commit and push** changes locally, then **`git pull`** on the pod
2. For individual files not in git, use **`scp`** for targeted transfers

**CRITICAL: Disk Quota Management**

Git operations on RunPod can fail with "Disk quota exceeded" if the repository contains large generated files. Before pulling code:

```bash
# Clean up old run data to free space (run data is regenerable)
ssh -i ~/.ssh/id_runpod root@<IP> -p <PORT> "cd /workspace/RRIP/evals/runs && rm -rf *"

# If git pull still fails, reset the repo to clean state
ssh -i ~/.ssh/id_runpod root@<IP> -p <PORT> "cd /workspace/RRIP && git reset --hard HEAD && git clean -fd && git pull"
```

Code sync workflow:

```bash
# Preferred: push locally, pull on pod
git push
ssh -i ~/.ssh/id_runpod root@<IP> -p <PORT> "cd /workspace/RRIP && git pull"

# For one-off files (e.g. test images, data files):
scp -i ~/.ssh/id_runpod -P <PORT> local/file.jpg root@<IP>:/workspace/RRIP/path/

# Downloading results from pod:
scp -i ~/.ssh/id_runpod -P <PORT> root@<IP>:/workspace/RRIP/evals/runs/some_run/timing_report.json ./
```

### Building on Pod

```bash
# Rust should already be installed; if not:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build CPU encoder (turbojpeg only)
cd /workspace/RRIP/server
cargo build --release

# Build GPU encoder
cd /workspace/RRIP/gpu-encode
cargo build --release
```

### Current Pod: origami-b200

- **Pod ID**: `tyjyett3vxrvbe`
- **GPU**: NVIDIA B200 (183 GB VRAM)
- **CUDA driver**: 580.126.09
- **Workspace**: `/workspace/RRIP`

## Safety Rules

- **Never use `rm -rf` without discussing it with the user first.** Always explain what will be deleted and get explicit confirmation before running any recursive delete.
- **Never use `rsync` to sync code to/from RunPod.** Use `git push` + `git pull` for code, and `scp` for targeted file transfers. rsync copies too many files and is slow/fragile.

## Debugging Tips

- Enable debug logging: `RUST_LOG=debug`
- Check metrics endpoint for cache hit rates
- Monitor RocksDB stats via metrics
- Use request IDs in logs for tracing
- Profile hotspots with perf/flamegraph for optimization