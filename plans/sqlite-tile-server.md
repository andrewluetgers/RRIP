# Plan: Migrate Tile Server to Per-Slide SQLite Storage

## Context

The ORIGAMI tile server currently stores tiles as individual files in directory trees and uses RocksDB as a warm cache for decoded tiles. This creates billions of objects at scale (44B for 20M ORIGAMI slides), massive GCS operation costs ($220K for upload alone), and requires managing two separate cache systems (RocksDB + filesystem).

The goal is to consolidate into **per-slide SQLite files** — one file per slide per tier (L3+ thumbnails, L0 full-res). SQLite is 35% faster than filesystem for small blob reads, provides built-in indexing, and reduces object count to 2 files per slide. The moka in-memory LRU stays as Tier 0.

## Architecture

```
Request → Tier 0: moka LRU (decoded L0/L1/L2 JPEG, ~1ms)
        → Tier 1: Per-slide SQLite on SSD
            L3+: query → raw JPEG → serve directly
            L0:  query → JXL bytes → decode → serve + insert Tier 0
        → Tier 2: GCS Archive (future) → fetch SQLite file → write to SSD
```

Two SQLite files per slide variant:
- `{slide_id}_l3.db` (~1.4 MB) — all L3+ JPEG thumbnail tiles
- `{slide_id}_l0.db` (~65 MB) — all L0 1024px JXL tiles + tissue map + metadata

LRU managed by a small `cache_lru.db` (SQLite) tracking slide access times. Eviction deletes whole SQLite files when SSD budget exceeded.

## Implementation

### Phase 1: `scripts/pack_sqlite.py` (Python)

New script that reads existing tile files and packs into SQLite:

**Schema (same for both DBs):**
```sql
CREATE TABLE tiles (
    level INTEGER, x INTEGER, y INTEGER,
    data BLOB, format TEXT,
    PRIMARY KEY (level, x, y)
) WITHOUT ROWID;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY, value BLOB
) WITHOUT ROWID;
```

**L3+ DB:** Insert all JPEG tiles from levels below L2.
**L0 DB:** Insert all 1024px JXL tiles. Metadata rows for: `dzi_manifest` (XML), `summary` (JSON), `tissue_map` (TMAP binary), `blank_color_grid` (JSON).

**Input:** `tile_server/jxl_q{80,40}/{slide_id}/` + tissue assets from `dzi/`
**Output:** `sqlite_slides/{variant}_{slide_id}_l3.db`, `..._l0.db`

### Phase 2: `server/src/core/slide_db.rs` (Rust)

New module — per-slide SQLite access:

```rust
pub struct SlideDb {
    l3_conn: Option<Connection>,
    l0_conn: Option<Connection>,
    l2_level: u32,  // boundary between L3+ and L0
}
```

Methods: `open()`, `get_tile(level, x, y) -> Option<(Vec<u8>, TileFormat)>`, `get_metadata(key) -> Option<Vec<u8>>`.

PRAGMAs: `journal_mode=WAL`, `mmap_size=268435456`, `synchronous=NORMAL`, `temp_store=MEMORY`.

**Dependency:** Add `rusqlite = { version = "0.32", features = ["bundled"] }`, remove `rocksdb`.

### Phase 3: `server/src/core/slide_cache.rs` (Rust)

Connection pool + disk LRU:

```rust
pub struct SlideCache {
    open_slides: Mutex<LruCache<String, Arc<SlideDb>>>,
    cache_dir: PathBuf,
    lru_db: Connection,  // cache_lru.db
    max_cache_bytes: u64,
}
```

- Keeps ~50 most recent SlideDb connections open (in-memory LRU via `lru` crate)
- `cache_lru.db` tracks all files on SSD with last_access timestamps
- Eviction: delete oldest SQLite files when over budget

**Dependency:** Add `lru = "0.12"`.

### Phase 4: Wire into `serve.rs`

Add `--sqlite-dir` CLI flag. When set:

- **Slide discovery:** Scan for `*_l3.db` files, read DZI manifest + summary from metadata table
- **L3+ requests:** `slide_cache.get_slide(id).get_tile(level, x, y)` → JPEG bytes → serve
- **L0/L1/L2 requests:** Same as current `source_1024` path but read JXL from SQLite instead of filesystem
- **TissueMap:** Load from L0 DB metadata table
- **DZI endpoint:** Read manifest from metadata table
- **RocksDB removed:** SQLite replaces it entirely; moka stays for Tier 0

Keep existing filesystem path working when `--sqlite-dir` is not set (backward compat).

### Phase 5: Update setup/packaging scripts

New `scripts/pack_all_sqlite.sh` that iterates slides and calls `pack_sqlite.py`.

Output structure:
```
sqlite_slides/
  3DHISTECH-1_jxl80_l3.db
  3DHISTECH-1_jxl80_l0.db
  Mayosh-1_jxl40_l3.db
  Mayosh-1_jxl40_l0.db
  ...
  cache_lru.db
```

## Files to Create/Modify

| File | Action | Notes |
|------|--------|-------|
| `scripts/pack_sqlite.py` | **Create** | Python packer |
| `scripts/pack_all_sqlite.sh` | **Create** | Batch runner |
| `server/src/core/slide_db.rs` | **Create** | Per-slide SQLite access |
| `server/src/core/slide_cache.rs` | **Create** | Connection pool + LRU |
| `server/src/core/mod.rs` | Modify | Register new modules |
| `server/Cargo.toml` | Modify | Swap rocksdb → rusqlite + lru |
| `server/src/serve.rs` | Modify | Add --sqlite-dir path, new discover_slides_sqlite, modify serve_tile |

## Verification

1. **Pack round-trip:** `pack_sqlite.py` → read back every tile → byte-compare with source files
2. **Server integration:** Start with `--sqlite-dir`, request tiles at all levels, compare with filesystem-served output
3. **Benchmark:** `pnpm bench:jxl80` against SQLite vs filesystem, compare latency distributions
4. **LRU test:** Pack more slides than SSD budget, verify eviction works
5. **Regression:** Existing filesystem mode still works when `--sqlite-dir` omitted
