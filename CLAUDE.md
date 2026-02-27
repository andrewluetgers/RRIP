# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ORIGAMI (Residual-Pyramid Image Processor) is a high-performance Rust tile server and residual encoder for serving Deep Zoom Image (DZI) pyramids with dynamic reconstruction of high-resolution tiles using residual compression techniques. The project is designed to efficiently serve whole-slide images (WSI) with reduced storage requirements.

The `origami` binary provides three subcommands:
- **`origami serve`** — Tile server with dynamic reconstruction and caching
- **`origami encode`** — Residual encoder that generates luma residuals from a DZI pyramid
- **`origami decode`** — Offline decoder that reconstructs tiles from a pyramid + residuals

This project also includes Python-based CLI tools managed with UV (Python package and project manager) for preprocessing whole-slide images and generating residual pyramids.

## Architecture

### Binary Structure

The `origami` binary is a thin clap dispatcher (`server/src/main.rs`) with subcommands:
- `server/src/serve.rs` — All tile server logic
- `server/src/encode.rs` — Residual encoding pipeline
- `server/src/decode.rs` — Offline tile reconstruction from pyramid + residuals

### Shared Core Modules (`server/src/core/`)

| Module | Purpose |
|--------|---------|
| `mod.rs` | `ResampleFilter` enum (Bilinear, Bicubic, Lanczos3) with `FromStr`/`Display`/`to_image_filter()` |
| `color.rs` | Float BT.601 YCbCr conversion (u8 and f32 variants) |
| `residual.rs` | Residual computation and centering (u8 and f32 variants) |
| `pack.rs` | Pack file format (read/write, "ORIG"/"RRIP" magic) |
| `upsample.rs` | 2x bilinear upsampling (legacy, used by server fast path) |
| `jpeg.rs` | `JpegEncoder` trait + backends (TurboJpeg, MozJpeg, Jpegli) |
| `libjpeg_ffi.rs` | Safe FFI wrapper for libjpeg62 compress API (conditional) |
| `reconstruct.rs` | Shared reconstruction pipeline (used by both serve and decode) |
| `optimize_l2.rs` | Gradient-descent L2 optimizer for better upsample predictions |
| `optimize_chroma.rs` | Gradient-descent chroma optimizer for 420opt encoding |
| `sharpen.rs` | Unsharp mask filter (RGB and grayscale, scalar + NEON) |
| `pyramid.rs` | DZI pyramid discovery and tile coordinate helpers |

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

1. **Tile Reconstruction Pipeline**: Dynamically reconstructs L0/L1 tiles from L2 baseline + single fused L0 luma residual
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
  residual_packs/                # V2 pack files (L2 baseline + fused L0 residual)
    {x2}_{y2}.pack
    residuals.bundle             # Optional: all families in single file
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

### CLI Reference: `origami encode`

Encodes residuals from a DZI pyramid or a single image. Two modes:
- **Pyramid mode**: `--pyramid` — encodes residuals for an existing DZI tile pyramid
- **Single-image mode**: `--image` — creates one L2 family from a single image (for evals)

**Quality parameters:**

V2 pipeline uses a single fused L0 residual per family (no L1 residuals).

| Parameter | Controls | Default |
|-----------|----------|---------|
| `--baseq` | L2 baseline JPEG quality | 95 |
| `--l0q` | L0 residual JPEG quality (overrides `--resq`) | (falls back to `--resq`) |
| `--resq` | Default residual quality for L0 | 50 |

Example: `--baseq 80 --resq 40 --subsamp 444 --optl2 --max-delta 20`

**Encoding parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--subsamp` | Chroma subsampling for L2 JPEG: `444`, `420`, `420opt` | 444 |
| `--encoder` | JPEG backend: `turbojpeg`, `mozjpeg`, `jpegli`, `jpegxl`, `webp` | turbojpeg |
| `--optl2` | Enable gradient descent optimization on L2 tiles | off |
| `--max-delta` | Max per-pixel deviation for optL2 | 15 |
| `--sharpen` | Unsharp mask strength on L2 (e.g. 0.5) | off |
| `--save-sharpened` | Store sharpened L2 in JPEG (else decoder must sharpen) | off |

**Resample filter parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--upsample-filter` | Filter for prediction upsamples: `bilinear`, `bicubic`, `lanczos3` | lanczos3 |
| `--downsample-filter` | Filter for ground-truth/residual downscales: `bilinear`, `bicubic`, `lanczos3` | lanczos3 |

**Residual scaling parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--l0-scale` | Downscale fused L0 residual before encoding (percent, 1-100) | 100 |
| `--l0-sharpen` | Unsharp mask strength for L0 residuals (e.g. 0.5-2.0) | off |

**Output parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--out` | Output directory | (required) |
| `--tile` | Tile size (must match pyramid) | 256 |
| `--pack` | Also create `.pack` files | off |
| `--manifest` | Write `manifest.json` with per-tile metrics | off |
| `--debug-images` | Write debug PNGs (originals, predictions, reconstructions) | off |
| `--max-parents` | Limit number of L2 parents to process (for testing) | all |

**Encoding pipeline order (CPU):** Downsample → OptL2 → Sharpen → JPEG encode (matches GPU pipeline).

### CLI Reference: `origami decode`

Offline decoder that reconstructs L1/L0 tiles from a pyramid plus v2 pack files. Exactly one residual source must be provided.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pyramid` | Path to DZI pyramid directory | (required) |
| `--out` | Output directory for reconstructed tiles | (required) |
| `--packs` | Directory with `.pack` files | — |
| `--pack-file` | Single `.pack` file to decode | — |
| `--bundle` | Bundle file (`.bundle`) containing all residual packs | — |
| `--quality` | Output JPEG quality | 95 |
| `--tile` | Tile size | 256 |
| `--upsample-filter` | Filter for prediction upsamples: `bilinear`, `bicubic`, `lanczos3` | lanczos3 |
| `--output-format` | Output format: `jpeg`, `webp` | jpeg |
| `--max-parents` | Limit number of L2 parents | all |
| `--grayscale` | Output grayscale tiles (for debugging) | off |
| `--timing` | Print per-family timing breakdown | off |

### CLI Reference: `origami serve`

Tile server with dynamic reconstruction and caching.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--slides-root` | Root directory containing slide data | data |
| `--port` | HTTP server port | 8080 |
| `--residuals-dir` | Subdirectory name for residuals | residuals_q32 |
| `--pack-dir` | Subdirectory name for pack files | residual_packs |
| `--tile-quality` | Output JPEG quality for reconstructed tiles | 95 |
| `--upsample-filter` | Filter for prediction upsamples: `bilinear`, `bicubic`, `lanczos3` | lanczos3 |
| `--sharpen` | Unsharp mask strength on L2 before upsampling (decode-time) | off |
| `--output-format` | Output format: `jpeg`, `webp` | jpeg |
| `--cache-entries` | In-memory hot cache size (tiles) | 2048 |
| `--cache-dir` | Optional disk cache directory | off |
| `--buffer-pool-size` | Pre-allocated buffer pool size | 128 |
| `--max-inflight-families` | Max concurrent family reconstructions | 32 |
| `--rayon-threads` | Rayon worker threads (CPU parallelism) | auto |
| `--tokio-workers` | Tokio async worker threads | 8 |
| `--tokio-blocking-threads` | Tokio blocking threads | 32 |
| `--prewarm-on-l2` | Prewarm family cache on L2 tile requests | off |
| `--grayscale-only` | Output grayscale tiles only | off |
| `--timing-breakdown` | Log per-request timing breakdown | off |
| `--metrics-interval-secs` | Metrics reporting interval | 30 |

### Running Examples

```bash
# Start the tile server
origami serve --slides-root /path/to/slides --port 3007

# Encode residuals from a DZI pyramid (v2: fused L0 residual, no L1 residuals)
origami encode --pyramid /path/to/slide --out /path/to/output --resq 50 --pack

# Encode with a specific backend
origami encode --pyramid /path/to/slide --out /path/to/output --resq 50 --encoder mozjpeg --pack

# Single-image encode with full options
origami encode --image evals/test-images/L0-1024.jpg --out /tmp/run \
    --baseq 95 --l0q 40 --subsamp 444 --optl2 --max-delta 20 \
    --upsample-filter lanczos3 --manifest --debug-images --pack

# Decode from pack file
origami decode --pyramid /path/to/slide --out /path/to/output \
    --pack-file /path/to/0_0.pack --upsample-filter lanczos3

# Decode from pack directory
origami decode --pyramid /path/to/slide --out /path/to/output \
    --packs /path/to/packs --quality 95
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
origami encode --image evals/test-images/L0-1024.jpg --out evals/runs/rs_444_optl2_d20_l0q40 \
    --baseq 95 --l0q 40 --subsamp 444 --optl2 --max-delta 20 --manifest --debug-images --pack

# Generate a JPEG baseline
uv run python evals/scripts/jpeg_baseline.py \
    --image evals/test-images/L0-1024.jpg --quality 70

# Batch runs
bash evals/scripts/run_all_captures.sh
bash evals/scripts/run_jpegli_captures.sh
bash evals/scripts/run_bc_split_sweep.sh
bash evals/scripts/run_upsample_filter_sweep.sh
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
| `uf_{filter}_{subsamp}[_optl2]_l1q{N}_l0q{N}` | `uf_l3_444_optl2_l1q60_l0q40` | UF lanczos3 L1=60 L0=40 444 optL2 |

Filter codes: `bl` (bilinear), `bc` (bicubic), `l3` (lanczos3).

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

### Reconstruction Algorithm (V2 — Fused L0 Residual)

1. Decode L2 baseline JPEG → RGB/YCbCr
2. Optionally sharpen L2 (if decode-time `--sharpen` is set)
3. Upsample L2 Y (4x) → L0 Y prediction (256x256 → 1024x1024)
4. Upsample L2 Cb/Cr (2x) → L1 chroma, (4x) → L0 chroma (parallel)
5. Decode single fused L0 residual JPEG (1024x1024 grayscale)
6. Apply: L0_Y_corrected = clamp(L0_Y_pred + (residual - 128), 0..255)
7. Slice L0 into 16 tiles → combine with L0 Cb/Cr → encode as JPEG (parallel)
8. Downsample L0_Y_corrected (2x) → L1 Y (512x512)
9. Slice L1 into 4 tiles → combine with L1 Cb/Cr → encode as JPEG

### Resample Filters

The `--upsample-filter` parameter controls the resampling kernel used for prediction upsamples at both encode and decode time. **The encode and decode filters must match** for correct reconstruction.

| Filter | Kernel | Speed | Quality (PSNR) | File Size |
|--------|--------|-------|----------------|-----------|
| `bilinear` | Triangle (2-tap) | Fastest | Lowest | Largest (blurry predictions → bigger residuals) |
| `bicubic` | Catmull-Rom (4-tap) | Medium | Middle | Middle |
| `lanczos3` | Lanczos (6-tap) | Slowest | Best | Smallest (sharpest predictions → smaller residuals) |

**Decode latency benchmark** (256px tiles, 1 L2 family = 4 L1 + 16 L0 tiles, Apple Silicon):

| Filter | Chroma upsample | L0 Y resize | Total family |
|--------|:-:|:-:|:-:|
| bilinear | 24ms | 10ms | **38ms** |
| bicubic | 29ms | 11ms | **43ms** |
| lanczos3 | 32ms | 13ms | **48ms** |

Lanczos3 adds ~10ms (~25%) per family vs bilinear, but still well within the 200-500ms cold-miss target. The quality/size benefits (better PSNR, 5-8% smaller files) justify the decode cost.

**Default: `lanczos3`** — best quality/size tradeoff.

### Performance Targets

- Hot cache hit: P95 < 10ms
- Warm cache hit: P95 < 25ms
- Cold miss (generate family): P95 < 200-500ms (lanczos3: ~48ms, bilinear: ~38ms)

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

**CRITICAL WORKFLOW**: Always query the API first to get current pod status and connection info. Pod IPs and ports can change, so never assume old connection details are still valid.

```bash
# Step 1: Query API for current pod info (IP, port, status)
curl -s -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  --data '{"query":"{ myself { pods { id name desiredStatus machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } gpus { id } } } } }"}' \
  https://api.runpod.io/graphql | jq '.'

# Step 2: Extract SSH connection info for origami-b200 pod
# Look for: runtime.ports[] where privatePort == 22
# Use: ip (public IP) and publicPort for SSH connection
```

### SSH Access

**Always get current IP/port from API first**, then connect:

```bash
# Connect to a pod using IP and port from API query
ssh -i ~/.ssh/id_runpod root@<IP> -p <PORT>

# Example workflow:
# 1. Query API → get IP=38.80.152.146, port=31867
# 2. ssh -i ~/.ssh/id_runpod root@38.80.152.146 -p 31867
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