# RRIP Tile Server (MVP)

This is a Rust HTTP server that serves DeepZoom tiles and reconstructs L1/L0 tiles on demand from L2 + residuals.

## Data layout

The server scans `--slides-root` for slide folders. Each slide folder should contain:

```
baseline_pyramid.dzi
baseline_pyramid_files/{level}/{x}_{y}.jpg
residuals_q32/L1/{x2}_{y2}/{x1}_{y1}.jpg
residuals_q32/L0/{x2}_{y2}/{x0}_{y0}.jpg
```

Optional packfiles:

```
residual_packs/{x2}_{y2}.pack
```

## Run

From the repo root:

```
cargo run --manifest-path server/Cargo.toml -- --slides-root data --port 3007
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
