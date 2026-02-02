# 🚀 HYPER-OPTIMIZATION PLAN FOR ORIGAMI SERVER

## Critical Bottlenecks Found

### 1. **Sequential Chroma Upsampling** 🔴 MAJOR
- L2→L1 resize happens, THEN L1 tiles process, THEN L1→L0 resize
- **Waste**: L0 chroma could start immediately from L2 in parallel!

### 2. **Memory Allocation Thrashing** 🔴 MAJOR
- `buffer_pool.get()` resizes buffers on every call
- Allocating new vectors for YCbCr planes repeatedly
- No reuse of intermediate buffers

### 3. **Redundant Image Decoding** 🟡 MEDIUM
- Using `image::open()` → slow PNG/JPEG detection
- RGB→YCbCr conversion happens multiple times
- No direct turbojpeg usage for zero-copy

### 4. **Lock Contention in Metrics** 🟡 MEDIUM
- Mutex on metrics for EVERY tile request
- BufferPool uses Mutex instead of lock-free queue

### 5. **Missed Parallelism Opportunities** 🔴 MAJOR
- L1 and L0 tile processing could overlap
- No pipelining between decode→process→encode stages
- Sequential mosaic building

## Optimization Strategy

### 🎯 **1. Parallel Chroma Pipeline**

```rust
// CURRENT (Sequential - BAD)
L2 decode → L2→L1 resize → Process L1 tiles → Build mosaic → L1→L0 resize → Process L0 tiles

// OPTIMIZED (Parallel - GOOD)
L2 decode ─┬→ L2→L1 resize → Process L1 tiles ─┐
           └→ L2→L0 resize ─────────────────────┴→ Process L0 tiles
```

**Implementation**:
```rust
// Start BOTH upsamples immediately
rayon::join(
    || {
        let l1_chroma = upsample_chroma(l2, 2x);
        process_l1_tiles(l1_chroma)
    },
    || {
        let l0_chroma_coarse = upsample_chroma(l2, 4x);
        // Wait for L1 to refine, but chroma is ready
    }
);
```

### 🎯 **2. Zero-Copy TurboJPEG Operations**

```rust
// CURRENT (Multiple copies)
fs::read() → image::open() → to_rgb8() → encode_jpeg()

// OPTIMIZED (Zero-copy)
mmap → turbojpeg::decompress_direct() → process_in_place() → turbojpeg::compress_direct()
```

**Benefits**:
- Direct YCbCr output from JPEG decoder
- In-place residual application
- 50% less memory copying

### 🎯 **3. Lock-Free Data Structures**

```rust
// Replace Mutex<BufferPool> with crossbeam::queue::ArrayQueue
struct LockFreeBufferPool {
    buffers: ArrayQueue<Vec<u8>>,
    // No mutex needed!
}

// Replace Mutex<Metrics> with atomic counters
struct LockFreeMetrics {
    tile_total: AtomicU64,
    tile_cache_hit: AtomicU64,
    // All atomics, no locks
}
```

### 🎯 **4. Memory Pool Hierarchy**

```rust
struct TieredBufferPools {
    // Size-specific pools (no resizing!)
    tile_256: BufferPool,    // 256x256x3 = 196KB
    tile_512: BufferPool,    // 512x512x3 = 768KB
    tile_1024: BufferPool,   // 1024x1024x3 = 3MB

    // Reusable plane buffers
    y_planes: BufferPool,
    cb_planes: BufferPool,
    cr_planes: BufferPool,
}
```

### 🎯 **5. Pipeline Parallelism with Crossbeam**

```rust
// 3-stage pipeline with bounded channels
let (decode_tx, decode_rx) = channel::bounded(32);
let (process_tx, process_rx) = channel::bounded(32);
let (encode_tx, encode_rx) = channel::bounded(32);

// Stage 1: Decode thread pool
rayon::spawn(|| {
    while let Ok(job) = decode_rx.recv() {
        let decoded = turbojpeg_decode(job);
        process_tx.send(decoded);
    }
});

// Stage 2: Processing thread pool
rayon::spawn(|| {
    while let Ok(job) = process_rx.recv() {
        let processed = apply_residuals(job);
        encode_tx.send(processed);
    }
});

// Stage 3: Encode thread pool
rayon::spawn(|| {
    while let Ok(job) = encode_rx.recv() {
        let encoded = turbojpeg_encode(job);
        cache.insert(encoded);
    }
});
```

### 🎯 **6. SIMD Optimizations**

```rust
// Use packed_simd for YCbCr conversion
use packed_simd::*;

fn rgb_to_ycbcr_simd(rgb: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // Process 8 pixels at once with AVX2
    let coeffs_y = f32x8::new(0.299, 0.587, 0.114, ...);

    for chunk in rgb.chunks_exact(24) { // 8 pixels
        let r = u8x8::from_slice(&chunk[0..8]);
        let g = u8x8::from_slice(&chunk[8..16]);
        let b = u8x8::from_slice(&chunk[16..24]);

        // SIMD multiply-accumulate
        let y = coeffs_y.mul_add(r, g, b);
        // ...
    }
}
```

### 🎯 **7. Smarter Residual Loading**

```rust
// Pre-load residuals while processing previous tile
struct ResidualPrefetcher {
    cache: DashMap<TileKey, Bytes>,
    prefetch_queue: ArrayQueue<TileKey>,
}

// In parallel with tile processing:
rayon::spawn(|| {
    for next_tile in prefetch_queue {
        let residual = fs::read(residual_path(next_tile));
        cache.insert(next_tile, residual);
    }
});
```

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
✅ Replace Mutex<LruCache> with Moka (DONE)
⬜ Add TurboJPEG pool for zero-copy operations
⬜ Fix BufferPool to use fixed sizes (no resize)
⬜ Replace Mutex<Metrics> with atomics

### Phase 2: Parallelism (2-4 hours)
⬜ Implement parallel L1/L0 chroma upsampling
⬜ Add crossbeam channels for pipeline stages
⬜ Overlap L0 preparation with L1 processing
⬜ Parallel mosaic assembly

### Phase 3: Advanced (4-8 hours)
⬜ SIMD YCbCr conversions
⬜ Memory-mapped residual packs
⬜ Prefetching and speculative processing
⬜ Custom allocator for hot paths

## Expected Performance Gains

| Optimization | Expected Gain | Impact |
|-------------|--------------|---------|
| Lock-free cache | 10-50x | ✅ DONE |
| Parallel chroma | 30-40% faster | Critical |
| Zero-copy JPEG | 20-30% faster | High |
| Fixed-size pools | 15-20% faster | Medium |
| Pipeline stages | 25-35% faster | High |
| SIMD operations | 10-15% faster | Medium |

**Combined**: **2-3x overall throughput improvement**

## Measurement Strategy

```bash
# Before optimization
wrk -t12 -c400 -d30s --latency http://localhost:8080/tiles/slide/10/{random}

# Key metrics to track:
- Requests/sec
- P50/P95/P99 latency
- CPU utilization per core
- Memory allocation rate (heaptrack)
- Lock contention (perf record -g)
```

## Rust-Specific Optimizations

### 1. **Tokio Runtime Tuning**
```rust
runtime::Builder::new_multi_thread()
    .worker_threads(4)          // Fewer workers, less contention
    .max_blocking_threads(128)   // More blocking for I/O
    .thread_stack_size(4 * 1024 * 1024) // Larger stacks
    .enable_all()
    .build()
```

### 2. **Rayon Configuration**
```rust
rayon::ThreadPoolBuilder::new()
    .num_threads(physical_cores) // Not logical cores
    .stack_size(4 * 1024 * 1024)
    .thread_name(|i| format!("rayon-{}", i))
    .build_global()
```

### 3. **CPU Affinity** (Linux only)
```rust
// Pin Rayon threads to P-cores
use core_affinity;
rayon::ThreadPoolBuilder::new()
    .start_handler(|idx| {
        // Pin to P-cores (0-3 on M5)
        core_affinity::set_for_current(core_affinity::CoreId { id: idx % 4 });
    })
```

### 4. **Huge Pages** (Linux)
```bash
# Enable transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# Or use explicit huge pages
cargo build --release
hugectl --heap ./target/release/origami-server
```

## The Nuclear Option: Custom Allocator

```rust
// Use mimalloc for better multithreaded performance
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

## Summary

The server has **massive untapped parallelism**. The current implementation is essentially sequential with some parallel tile processing. By implementing:

1. **Concurrent chroma upsampling** (L2→L1 and L2→L0 in parallel)
2. **Zero-copy JPEG operations** (TurboJPEG direct)
3. **Lock-free structures** (already started with Moka)
4. **Pipeline parallelism** (decode|process|encode stages)
5. **Fixed-size buffer pools** (no allocation in hot path)

We can achieve **2-3x throughput improvement** with **50% lower P99 latency**.