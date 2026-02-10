# ORIGAMI

ORIGAMI is a Rust tile server and residual encoder for Deep Zoom Image (DZI) pyramids. The `origami` binary provides two subcommands:

- **`origami serve`** — HTTP tile server with dynamic L0/L1 reconstruction and caching
- **`origami encode`** — Residual encoder that generates luma residuals from a DZI pyramid

## Install Rust

If `cargo` is not available, install Rust with rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then restart your shell so `cargo` is on `PATH`.

## Building

### Default Build (turbojpeg only)

The default build links against libjpeg-turbo for JPEG encoding/decoding. No extra setup is needed:

```bash
cd server
cargo build --release
```

### Building with mozjpeg

mozjpeg provides better JPEG compression via trellis quantization. It must be built as a static library first:

```bash
# 1. Build mozjpeg static library (one-time setup)
#    Prerequisites: cmake, nasm
./scripts/build_mozjpeg.sh
# Installs to: vendor/mozjpeg/

# 2. Build origami with mozjpeg support
cd server
cargo build --release --features mozjpeg
```

To use a custom mozjpeg install location, set `MOZJPEG_LIB_DIR` and `MOZJPEG_INCLUDE_DIR`:

```bash
MOZJPEG_LIB_DIR=/opt/mozjpeg/lib MOZJPEG_INCLUDE_DIR=/opt/mozjpeg/include \
  cargo build --release --features mozjpeg
```

### Building with jpegli

jpegli (from Google) achieves ~35% better compression than libjpeg-turbo at equivalent perceptual quality. It requires LLVM on macOS:

```bash
# 1. Install prerequisites (macOS)
brew install llvm coreutils cmake giflib jpeg-turbo libpng ninja zlib

# 2. Build jpegli static library (one-time setup)
./scripts/build_jpegli.sh
# Installs to: vendor/jpegli/

# 3. Build origami with jpegli support
cd server
cargo build --release --features jpegli
```

To use a custom jpegli install location, set `JPEGLI_LIB_DIR` and `JPEGLI_INCLUDE_DIR`:

```bash
JPEGLI_LIB_DIR=/opt/jpegli/lib JPEGLI_INCLUDE_DIR=/opt/jpegli/include/jpegli \
  cargo build --release --features jpegli
```

> **Note:** mozjpeg and jpegli are **mutually exclusive** — both define libjpeg62 symbols and cannot be linked together. Use `--features mozjpeg` OR `--features jpegli`, not both.

### Using System libjpeg-turbo (Homebrew)

To use Homebrew's libjpeg-turbo instead of building from source:

```bash
brew install libjpeg-turbo
TURBOJPEG_SOURCE=pkg-config cargo build --release
```

Or with explicit paths:

```bash
TURBOJPEG_SOURCE=explicit \
TURBOJPEG_DYNAMIC=1 \
TURBOJPEG_LIB_DIR=$(brew --prefix jpeg-turbo)/lib \
TURBOJPEG_INCLUDE_DIR=$(brew --prefix jpeg-turbo)/include \
cargo build --release
```

### Running Tests

```bash
cd server
cargo test --bins                     # Default (turbojpeg only)
cargo test --bins --features mozjpeg  # With mozjpeg
```

## Encoder Comparison

| Encoder | Speed | Compression | Use Case |
|---------|-------|-------------|----------|
| **turbojpeg** | Fastest | Baseline | Server decoding, fast encoding |
| **mozjpeg** | Slower | ~10-15% smaller | Production residual encoding |
| **jpegli** | Medium | ~35% smaller at same quality | Best compression for residuals |

## Usage

### Serve Subcommand

Start the tile server:

```bash
origami serve \
  --slides-root /path/to/slides \
  --port 3007 \
  --cache-dir /tmp/origami-cache \
  --pack-dir residual_packs
```

Then open: `http://localhost:3007/viewer/{slide_id}`

The server uses turbojpeg for decoding only — no encoder features are needed for serving.

#### Full Production Config

```bash
origami serve \
  --slides-root /data/slides \
  --port 3007 \
  --cache-dir /var/cache/origami \
  --pack-dir residual_packs \
  --hot-cache-mb 1024 \
  --warm-cache-gb 10 \
  --rayon-threads 16 \
  --max-inflight-families 32 \
  --timing-breakdown
```

### Encode Subcommand

Generate residuals from a DZI pyramid:

```bash
# Encode with turbojpeg (default, no extra build flags needed)
origami encode \
  --pyramid /path/to/slide \
  --out /path/to/output \
  --resq 50

# Encode with mozjpeg (requires --features mozjpeg at build time)
origami encode \
  --pyramid /path/to/slide \
  --out /path/to/output \
  --resq 50 \
  --encoder mozjpeg

# Encode with jpegli (requires --features jpegli at build time)
origami encode \
  --pyramid /path/to/slide \
  --out /path/to/output \
  --resq 30 \
  --encoder jpegli

# Also generate pack files
origami encode \
  --pyramid /path/to/slide \
  --out /path/to/output \
  --resq 50 \
  --encoder mozjpeg \
  --pack
```

#### Encode Options

```
--pyramid <path>        Path to DZI pyramid directory (containing baseline_pyramid.dzi)
--out <path>            Output directory for residuals and pack files
--tile <size>           Tile size, must match pyramid (default: 256)
--resq <quality>        JPEG quality for residual encoding, 1-100 (default: 50)
--encoder <name>        JPEG encoder backend: turbojpeg, mozjpeg, jpegli (default: turbojpeg)
--max-parents <n>       Maximum L2 parent tiles to process (for testing)
--pack                  Also create pack files
```

## Serve Parameters

### --slides-root (Required)
Root directory containing all slide folders. The server recursively scans this directory at startup for valid slide structures.

### --cache-dir
Directory where RocksDB stores the persistent warm cache (default: `/tmp/origami-cache`).

### --pack-dir
Subdirectory name within each slide folder containing pre-generated pack files.

**Pack file format:**
- One pack per L2 tile coordinate: `{x2}_{y2}.pack`
- Contains: 1 L2 + 4 L1 residuals + 16 L0 residuals
- Reduces 21 file reads to 1 seek operation

## Data Layout

The server expects this directory structure under `--slides-root`:

```
<slides-root>/
├── slide1/
│   ├── baseline_pyramid.dzi          # DZI manifest
│   ├── baseline_pyramid_files/        # Standard pyramid tiles
│   │   ├── 0/                        # Lowest resolution
│   │   │   └── 0_0.jpg
│   │   ├── ...
│   │   └── 14/                       # Highest resolution (L0)
│   │       ├── 0_0.jpg
│   │       └── ...
│   ├── residuals_q32/                # Individual residual tiles
│   │   ├── L1/{x2}_{y2}/{x1}_{y1}.jpg
│   │   └── L0/{x2}_{y2}/{x0}_{y0}.jpg
│   └── residual_packs/               # Optional pack files
│       ├── 0_0.pack                  # Bundle for L2 tile (0,0)
│       ├── 0_1.pack
│       └── ...
└── slide2/
    └── (same structure)
```

## Performance Tuning Guide

### Thread Pool Configuration

The server uses a layered concurrency design with separate thread pools for different workloads:

- **Rayon threads**: CPU-intensive work (JPEG decode/encode, tile reconstruction)
- **Tokio workers**: Async I/O coordination (HTTP handling, task scheduling)
- **Tokio blocking threads**: Blocking I/O operations (file reads, RocksDB)

### Recommended Settings by Hardware

#### Apple Silicon (M1/M2/M3)

| Processor | P-cores | E-cores | --rayon-threads | --tokio-workers | --tokio-blocking-threads | --max-inflight-families |
|-----------|---------|---------|-----------------|-----------------|---------------------------|-------------------------|
| M1/M2/M3  | 4       | 4       | 6               | 4               | 18                        | 12                      |
| M1/M2 Pro | 6-8     | 2-4     | 8               | 4               | 24                        | 16                      |
| M1/M2 Max | 8-10    | 2-4     | 10              | 6               | 30                        | 20                      |
| M4/M5     | 4       | 6       | 8               | 4               | 24                        | 16                      |

#### x86_64 Processors

| Cores | --rayon-threads | --tokio-workers | --tokio-blocking-threads | --max-inflight-families |
|-------|-----------------|-----------------|---------------------------|-------------------------|
| 8     | 8               | 4               | 24                        | 16                      |
| 16    | 14              | 6               | 42                        | 28                      |
| 32    | 28              | 8               | 84                        | 56                      |
| 64    | 56              | 12              | 168                       | 112                     |

### Memory Tuning

```bash
--buffer-pool-size <size>  # Pre-allocated buffers (default: 128)
--hot-cache-mb <mb>        # In-memory LRU cache size
--warm-cache-gb <gb>       # RocksDB cache size
```

### All Serve Flags

```
--timing-breakdown                    # Enable detailed timing logs
--write-generated-dir /path/to/output # Save generated tiles to disk
--write-queue-size 2048               # Queue size for disk writes
--metrics-interval-secs 30            # Prometheus metrics interval
--buffer-pool-size 128                # Pre-allocated buffer count
--residual-pack-dir /path/to/packs    # Directory for packfiles
--rayon-threads 24                    # CPU worker threads
--tokio-workers 8                     # Async worker threads
--tokio-blocking-threads 64           # Blocking I/O threads
--max-inflight-families 64            # Concurrent tile families
--prewarm-on-l2                       # Pre-generate L1/L0 when L2 accessed
```

## Notes

- L2 and above are served from `baseline_pyramid_files`.
- L1/L0 tiles are reconstructed from residuals when present; missing residuals fall back to baseline tiles.
- The `serve` subcommand uses turbojpeg for decoding only — no encoder features needed.
- The `encode` subcommand uses whichever encoder backend was compiled in via feature flags.
