# DashMap Optimization for ORIGAMI Server

## Problem Analysis

Your current cache implementation has a **critical bottleneck**:
- `Arc<Mutex<LruCache<TileKey, Bytes>>>` creates a single global lock
- ALL operations (reads/writes) must acquire this lock
- Rayon parallel workers are **serialized** waiting for cache access

## Why Performance Isn't Improving

Even with more Rayon threads, they're all waiting on the same Mutex! This explains why your optimizations aren't helping.

## DashMap Solution

Replace the Mutex<LruCache> with DashMap for lock-free concurrent access:

### 1. Add DashMap to Cargo.toml

```toml
[dependencies]
dashmap = "6.1"
lru = "0.12"  # Keep for LRU eviction logic
```

### 2. Replace Cache Structure

```rust
// Old (line 70)
cache: Arc<Mutex<LruCache<TileKey, Bytes>>>,

// New - Two-tier cache
hot_cache: Arc<DashMap<TileKey, CacheEntry>>,
lru_tracker: Arc<Mutex<LruTracker>>,  // Only for eviction tracking

struct CacheEntry {
    bytes: Bytes,
    last_access: Instant,
    access_count: AtomicU32,
}

struct LruTracker {
    order: VecDeque<TileKey>,
    max_entries: usize,
}
```

### 3. Lock-Free Cache Reads

```rust
// Old (line 624) - BLOCKS ALL THREADS
if let Some(bytes) = state.cache.lock().unwrap().get(&key).cloned()

// New - CONCURRENT READS
if let Some(entry) = state.hot_cache.get(&key) {
    entry.access_count.fetch_add(1, Ordering::Relaxed);
    let bytes = entry.bytes.clone();
    // No lock held!
}
```

### 4. Parallel Cache Writes

```rust
// Old (line 677-679) - HOLDS LOCK FOR ENTIRE LOOP
let mut cache_guard = cache.lock().unwrap();
for (k, v) in family.iter() {
    cache_guard.put(k.clone(), v.clone());
}

// New - PARALLEL INSERTS
family.par_iter().for_each(|(k, v)| {
    state.hot_cache.insert(k.clone(), CacheEntry {
        bytes: v.clone(),
        last_access: Instant::now(),
        access_count: AtomicU32::new(0),
    });
});

// Eviction handled separately (async background task)
```

### 5. Smart Eviction Strategy

```rust
// Background task (runs every N seconds)
async fn evict_task(cache: Arc<DashMap<TileKey, CacheEntry>>, max_size: usize) {
    if cache.len() > max_size {
        // Collect entries with access stats
        let mut entries: Vec<_> = cache.iter()
            .map(|entry| {
                let key = entry.key().clone();
                let score = calculate_eviction_score(&entry);
                (key, score)
            })
            .collect();

        // Sort by eviction score
        entries.sort_by_key(|(_, score)| *score);

        // Remove bottom 10%
        let to_remove = entries.len() / 10;
        for (key, _) in entries.iter().take(to_remove) {
            cache.remove(key);
        }
    }
}

fn calculate_eviction_score(entry: &CacheEntry) -> i64 {
    let age = entry.last_access.elapsed().as_secs() as i64;
    let frequency = entry.access_count.load(Ordering::Relaxed) as i64;
    age - (frequency * 10)  // LFU with age factor
}
```

## Alternative: Moka Cache

Even simpler - use Moka, which is already concurrent and has built-in LRU:

```toml
[dependencies]
moka = { version = "0.12", features = ["sync"] }
```

```rust
use moka::sync::Cache;

// In AppState
cache: Arc<Cache<TileKey, Bytes>>,

// Initialize
let cache = Cache::builder()
    .max_capacity(args.cache_entries as u64)
    .time_to_live(Duration::from_secs(300))
    .build();

// Usage - ALL OPERATIONS ARE LOCK-FREE
if let Some(bytes) = state.cache.get(&key) {
    // Concurrent read
}

state.cache.insert(key, bytes);  // Concurrent write
```

## Expected Performance Gains

With DashMap or Moka:
- **10-50x faster cache reads** under contention
- **Linear scaling** with Rayon threads (finally!)
- **No thread starvation**
- **Lower P99 latency**

## Quick Test

Try this minimal change first with Moka:

1. Add `moka = "0.12"` to Cargo.toml
2. Replace cache initialization (line ~371)
3. Update cache access patterns (3 locations)
4. Run with your original thread settings

This should immediately show if lock contention is your bottleneck (it almost certainly is).

## Flame Graph Focus

When you get your flame graph, look for:
- Time spent in `parking_lot::mutex::lock`
- Thread park/unpark overhead
- Rayon workers blocked on cache access

These will confirm the diagnosis and show the improvement after switching to DashMap/Moka.