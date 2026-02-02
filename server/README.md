# RRIP Tile Server (MVP)

This is a Rust HTTP server that serves DeepZoom tiles and reconstructs L1/L0 tiles on demand from L2 + residuals.

## Core Parameters

### --slides-root (Required)
Root directory containing all slide folders. The server recursively scans this directory at startup for valid slide structures.

```bash
--slides-root /path/to/slides  # e.g., --slides-root ../data
```

### --cache-dir
Directory where RocksDB stores the persistent warm cache. This is a key-value database that caches reconstructed tiles across server restarts.

```bash
--cache-dir /var/cache/rrip  # Default: /tmp/rrip-cache
```

**How it works:**
- RocksDB creates SST (Sorted String Table) files in this directory
- Keys: `tile:{slide_id}:{level}:{x}:{y}`
- Values: Compressed JPEG bytes of reconstructed tiles
- Survives server restarts - tiles don't need re-reconstruction
- Can be shared between multiple server instances (read-only)
- Typical size: 10-50GB depending on usage patterns

**Cache directory structure:**
```
/var/cache/rrip/
├── 000003.log          # Write-ahead log
├── 000004.sst          # Sorted String Table files
├── 000005.sst          # (immutable tile data)
├── CURRENT             # Points to current manifest
├── IDENTITY            # DB identity file
├── LOCK                # Process lock
├── LOG                 # RocksDB logs
├── MANIFEST-000002     # DB metadata
└── OPTIONS-000006      # Configuration
```

### --pack-dir
Subdirectory name within each slide folder containing pre-generated pack files. Pack files bundle all residuals for a tile family into a single file for faster I/O.

```bash
--pack-dir residual_packs  # Default: looks for this name
```

**Pack file format:**
- One pack per L2 tile coordinate: `{x2}_{y2}.pack`
- Contains: 1 L2 + 4 L1 residuals + 16 L0 residuals
- Reduces 21 file reads to 1 seek operation
- ~100-400KB per pack file

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

## Quick Start

```bash
# Build with optimizations
./build.sh --release

# Run with essential parameters
./run-server.sh \
  --slides-root ../data \
  --port 3007 \
  --cache-dir /tmp/rrip-cache \
  --pack-dir residual_packs

# Full production config
cargo run --release --manifest-path server/Cargo.toml -- \
  --slides-root /data/slides \
  --port 3007 \
  --cache-dir /var/cache/rrip \
  --pack-dir residual_packs \
  --hot-cache-mb 1024 \
  --warm-cache-gb 10 \
  --rayon-threads 16 \
  --max-inflight-families 32 \
  --timing-breakdown
```

## Install Rust

If `cargo` is not available, install Rust with rustup:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then restart your shell so `cargo` is on `PATH`.

Then open:

```
http://localhost:3007/viewer/demo_out
```

## Performance Tuning Guide

### Thread Pool Configuration

The server uses a layered concurrency design with separate thread pools for different workloads:

- **Rayon threads**: CPU-intensive work (JPEG decode/encode, tile reconstruction)
- **Tokio workers**: Async I/O coordination (HTTP handling, task scheduling)
- **Tokio blocking threads**: Blocking I/O operations (file reads, RocksDB)

### Recommended Settings by Hardware

#### Apple Silicon (M1/M2/M3)

For Apple M-series processors with unified memory architecture:

| Processor | P-cores | E-cores | --rayon-threads | --tokio-workers | --tokio-blocking-threads | --max-inflight-families |
|-----------|---------|---------|-----------------|-----------------|---------------------------|-------------------------|
| M1/M2/M3  | 4       | 4       | 6               | 4               | 18                        | 12                      |
| M1/M2 Pro | 6-8     | 2-4     | 8               | 4               | 24                        | 16                      |
| M1/M2 Max | 8-10    | 2-4     | 10              | 6               | 30                        | 20                      |
| M4/M5     | 4       | 6       | 8               | 4               | 24                        | 16                      |

**Example for M5 (4P + 6E cores):**
```bash
cargo run --manifest-path server/Cargo.toml -- \
  --slides-root data \
  --port 3007 \
  --rayon-threads 8 \
  --tokio-workers 4 \
  --tokio-blocking-threads 24 \
  --max-inflight-families 16 \
  --buffer-pool-size 256
```

**Rationale:**
- **rayon-threads**: Slight undercommit (8 vs 10 logical cores) since E-cores are ~65% the throughput of P-cores
- **tokio-workers**: Async coordination doesn't need much CPU
- **tokio-blocking-threads**: 3× rayon threads (mostly parked waiting)
- **max-inflight-families**: 2× rayon threads (real concurrency cap)
- **buffer-pool-size**: 16 × max-inflight-families (adjust based on memory)

⚠️ **Apple Silicon Note**: Unified memory means CPU and GPU share bandwidth. Display compositing or GPU work may affect performance.

#### x86_64 Processors

For Intel/AMD processors:

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

**Memory calculation:**
- Each L0 tile: ~100-200KB encoded
- Buffer pool: `buffer_pool_size × 256KB` (for 256×256 RGB buffers)
- Hot cache: Store frequently accessed tiles in memory
- Warm cache: Persistent storage for generated tiles

### All Available Flags

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

## libjpeg-turbo (Homebrew)

If you prefer using Homebrew’s libjpeg-turbo instead of building from source:

```
brew install libjpeg-turbo
```

Then build with:

```
TURBOJPEG_SOURCE=explicit \
TURBOJPEG_DYNAMIC=1 \
TURBOJPEG_LIB_DIR=$(brew --prefix jpeg-turbo)/lib \
TURBOJPEG_INCLUDE_DIR=$(brew --prefix jpeg-turbo)/include \
cargo build --manifest-path server/Cargo.toml
```

## Notes

- L2 and above are served from `baseline_pyramid_files`.
- L1/L0 tiles are reconstructed from residuals when present; missing residuals fall back to baseline tiles.
