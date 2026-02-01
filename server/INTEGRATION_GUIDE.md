# Integration Guide: Applying All Optimizations

## Quick Integration Steps

To integrate all optimizations into your `main.rs`:

### 1. Add to Cargo.toml (✅ Done)
```toml
crossbeam = "0.8"
parking_lot = "0.12"
```

### 2. Add Module Declarations

At the top of `main.rs`:
```rust
mod optimized;
mod metrics_optimized;
mod parallel_generation;

use optimized::{TieredBufferPools, TurboJpegPool};
use metrics_optimized::LockFreeMetrics;
use parallel_generation::generate_family_optimized;
```

### 3. Replace AppState

```rust
#[derive(Clone)]
struct AppState {
    slides: Arc<HashMap<String, Slide>>,
    cache: Arc<Cache<TileKey, Bytes>>,  // Already optimized with Moka ✅
    tile_quality: u8,
    timing_breakdown: bool,
    writer: Option<mpsc::Sender<WriteJob>>,
    write_generated_dir: Option<PathBuf>,
    metrics: Arc<LockFreeMetrics>,  // CHANGED: Lock-free
    buffer_pools: Arc<TieredBufferPools>,  // CHANGED: Tiered pools
    turbo_pool: Arc<TurboJpegPool>,  // NEW: TurboJPEG pool
    pack_dir: Option<PathBuf>,
    inflight: Arc<InflightFamilies>,
    inflight_limit: Arc<Semaphore>,
    prewarm_on_l2: bool,
}
```

### 4. Initialize in async_main

```rust
// Initialize optimized components
let buffer_pools = Arc::new(TieredBufferPools::new(256)); // Assuming 256x256 tiles
let turbo_pool = TurboJpegPool::new(args.rayon_threads.unwrap_or(8) * 2);
let metrics = LockFreeMetrics::new();

let state = AppState {
    slides: slides.clone(),
    cache: Arc::new(cache),
    tile_quality: args.tile_quality,
    timing_breakdown: args.timing_breakdown,
    writer,
    write_generated_dir,
    metrics,  // Lock-free metrics
    buffer_pools,  // Tiered pools
    turbo_pool,  // TurboJPEG pool
    pack_dir: args.residual_pack_dir.clone(),
    inflight,
    inflight_limit,
    prewarm_on_l2: args.prewarm_on_l2,
};
```

### 5. Update Metrics Recording

Replace all `state.metrics.lock().unwrap().record_tile()` with:
```rust
state.metrics.record_tile("cache_hit", start.elapsed().as_millis());
// No lock needed!
```

### 6. Replace generate_family

In the spawn_blocking section, replace:
```rust
let result = generate_family(
    &slide,
    x2,
    y2,
    quality,
    timing,
    &writer,
    &write_root,
    &buffer_pool,  // Old buffer pool
    pack_dir.as_deref(),
);
```

With:
```rust
let result = generate_family_optimized(
    &slide,
    x2,
    y2,
    quality,
    timing,
    &writer,
    &write_root,
    &buffer_pools,  // Tiered pools
    &turbo_pool,    // TurboJPEG pool
    pack_dir.as_deref(),
);
```

### 7. Update Metrics Task

```rust
// In the metrics reporting task
let snapshot = metrics.snapshot_and_reset();  // Atomic snapshot
info!(
    "metrics tiles_total={} cache_hit={} avg_ms={:.1} max_ms={:.1}",
    snapshot.tile_total,
    snapshot.tile_cache_hit,
    snapshot.tile_avg_ms(),
    snapshot.tile_max_ms()
);
```

## Performance Testing Script

```bash
#!/bin/bash
# test_optimized.sh

echo "Building optimized server..."
cd server
export TURBOJPEG_SOURCE=explicit
export TURBOJPEG_DYNAMIC=1
export TURBOJPEG_LIB_DIR=/opt/homebrew/lib
export TURBOJPEG_INCLUDE_DIR=/opt/homebrew/include

cargo build --release

echo "Running with optimizations..."
cargo run --release -- \
  --slides-root ../data \
  --port 3007 \
  --rayon-threads 8 \
  --tokio-workers 4 \
  --tokio-blocking-threads 24 \
  --max-inflight-families 32 \
  --cache-entries 8192 \
  --timing-breakdown
```

## Expected Performance Improvements

| Optimization | Improvement | Cumulative |
|--------------|-------------|------------|
| Moka Cache (already done) | 10-50x cache ops | Baseline |
| Fixed Buffer Pools | 15-20% less allocation | +15-20% |
| TurboJPEG Zero-copy | 20-30% faster decode/encode | +35-50% |
| Lock-free Metrics | 5-10% less contention | +40-60% |
| **Parallel Chroma** | **30-40% faster generation** | **+70-100%** |
| Integer YCbCr Math | 5-10% faster conversions | +75-110% |

**Total Expected: 2-3x throughput improvement**

## Verification Checklist

- [ ] No more `buffer.resize()` calls in hot paths
- [ ] No mutex locks for metrics
- [ ] L2→L0 chroma starts immediately (not waiting for L1)
- [ ] TurboJPEG used directly (no image crate in hot path)
- [ ] Fixed-size pools for each buffer size
- [ ] Integer math for YCbCr conversions

## Monitoring

Watch for these improvements in logs:
1. `family_breakdown` timing should show overlapped L1/L0 processing
2. Lower `total_ms` for family generation
3. Higher requests/second throughput
4. Lower CPU usage per request
5. Stable memory usage (no allocation spikes)

## Troubleshooting

If compilation fails:
```bash
# Ensure turbojpeg is available
brew install jpeg-turbo

# Clean build
cargo clean
cargo build --release
```

If performance doesn't improve:
1. Check `--timing-breakdown` output
2. Verify parallel execution with `htop` (should see all cores active)
3. Monitor with `cargo flamegraph` to identify bottlenecks

## Next Level Optimizations (Future)

1. **SIMD with `packed_simd2`** for YCbCr conversion
2. **io_uring** on Linux for async file I/O
3. **Memory-mapped residual packs** with `mmap`
4. **GPU acceleration** for upsampling (if applicable)
5. **Custom allocator** like `mimalloc`