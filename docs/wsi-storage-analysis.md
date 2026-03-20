# WSI Storage Analysis

## Overview

A whole-slide image (WSI) is a gigapixel scan of a tissue specimen on a glass slide. A single slide can be 100,000+ pixels on a side. To view these at any zoom level without loading the entire image, we use a **Deep Zoom Image (DZI) pyramid** — a hierarchical set of pre-rendered tiles at progressively lower resolutions.

The ORIGAMI pipeline compresses these pyramids for efficient storage and serving, using tissue detection to skip blank regions and JPEG XL encoding for the high-resolution tiles.

## Example Slide: Mayosh-3
Here are sample views from an Open Seadragon viewer of a slide from the Mayo Safe Harbor 40K cohort in JXL 80 and 40 variants vs original JPEG:

1x zoom of full resolution L0 tiles
![Mayosh-3-1x.png](Mayosh-3-1x.png)

2x zoom
![Mayosh-3-2x.png](Mayosh-3-2x.png)

5x zoom subtle loss of detail and edge artifacts visible upon close inspection of the Q40 variant
![Mayosh-3-5x.png](Mayosh-3-5x.png)

### DZI Pyramid Structure

A DZI pyramid organizes tiles into numbered levels. Level numbers count up from 0 (single pixel) to N (full resolution). We refer to these relative to the full-resolution level:

```
slide_files/
│
├── 17/                  ← L0: Full original resolution
│   ├── 0_0.jxl             1024px tiles, JXL compressed
│   ├── 0_1.jxl             Only tissue tiles stored (blank regions omitted)
│   └── ...                  ~70-80% of tiles skipped on sparse slides
│
├── 16/                  ← L1: Half resolution (OMITTED — generated on demand)
│                            Tile server reconstructs from L0 at request time
│
├── 15/                  ← L2: Quarter resolution (OMITTED — generated on demand)
│                            Tile server reconstructs from L0 at request time
│
├── 14/                  ← L3: Eighth resolution
│   ├── 0_0.jpeg             256px tiles, pre-rendered JPEG Q90
│   └── ...                  Small — included in full for all variants
│
├── 13/ ... 8/           ← L4+: Lower resolutions, progressively fewer tiles
│                            Pre-rendered JPEG Q90, shared via symlinks
│                            Total L3+ overhead: ~9 MB per 100K-tile slide
│
└── (levels 0-7 omitted — single tile or smaller, not generated)
```

**Original slides** use the same structure but with 256px JPEG tiles at L0 (raw bytes extracted from the DICOM, zero re-encoding) and all levels present — nothing omitted or generated on demand.

### Two Compression Levers

Storage savings come from two independent mechanisms:

1. **Blank tile removal** — Tissue detection identifies which tiles contain actual specimen vs. empty glass background. Blank tiles are omitted from L0 through L2. However, blank tiles compress very well as JPEGs (averaging 3.4 KB vs 12.0 KB for tissue tiles — 3.6x smaller), so the byte savings are less dramatic than the tile count suggests. Across our test set, blank tiles are **85% of tile count but only 61% of total bytes**. The savings still scale with how blank the slide is, but the effective compression from blank removal alone is ~1.6x, not the 4x that tile count ratios imply.

2. **JXL compression** — JPEG XL re-encodes the tissue tiles at either Q80 (high quality, 3.1x smaller per tile) or Q40 with noise synthesis (8.0x smaller per tile). This ratio is constant regardless of tissue coverage and is the dominant source of savings.

These multiply: a sparse slide (23% tissue) at Q40 achieves 22x total compression. A dense slide (55%) still achieves 12x. But the majority of the savings come from JXL compression, not blank removal.

### Blank vs Tissue Tile Sizes

| Metric | Blank tiles | Tissue tiles |
|---|---|---|
| Count (% of total) | 85% | 15% |
| Bytes (% of total) | 61% | 39% |
| Avg size | 3.4 KB | 12.0 KB |
| Size ratio | 1x | 3.6x |

Blank tiles are small because they compress to near-minimum JPEG size (uniform color → tiny DCT coefficients). This means the per-byte benefit of skipping them is roughly half the per-tile benefit. The primary value of tissue detection is enabling the JXL pipeline to avoid wasting encode time and decode resources on blank regions, not just raw storage savings.

### Why 1024px JXL Tiles Instead of 256px?

The compressed variants store L0 as 1024px tiles rather than the DZI-native 256px for two reasons:

1. **Compression efficiency** — JXL's VarDCT transform and adaptive quantization work dramatically better with more spatial context. A 1024px tile gives the encoder 16x more area to find patterns, use larger DCT block sizes, and amortize per-tile header overhead. At 256px, each tile is an independent file with its own metadata, and the encoder has very limited context for its adaptive decisions. In practice, a 1024px JXL tile is significantly smaller than 16 equivalent 256px JXL tiles.

2. **Fewer files** — 16x fewer files means less filesystem overhead, faster directory listing, and simpler pack/transfer operations. A slide with 100K 256px tiles becomes ~6K 1024px tiles.

The tile server transparently slices 1024px tiles back into 256px for serving, and downsamples them to generate L1/L2 on demand.

### Empirical Evidence: 1024px vs 16x256px JXL

Measured across 10 randomly sampled full L2 families (16 tissue tiles each) from 3DHISTECH-1:

| Approach | Q80 Total | Q40 Total |
|---|---|---|
| **16 individual 256px JXL** | 768 KB | 292 KB |
| **1 stitched 1024px JXL** | 687 KB | 237 KB |
| **1024px savings** | **10.5%** | **18.6%** |

Per-family savings range:

| Quality | Min | Max | Avg |
|---|---|---|---|
| **Q80** | 9.0% | 18.2% | 11.3% |
| **Q40** | 15.9% | 30.8% | 20.3% |

Compression ratio vs original JPEG:

| Approach | Q80 | Q40 |
|---|---|---|
| 16x256px JXL | 2.2x | 5.9x |
| 1x1024px JXL | 2.5x | 7.2x |

The benefit is more pronounced at lower quality (Q40) because the encoder has more freedom to redistribute bits across the larger tile area. At Q80, the savings are modest but consistent (~11%). At Q40 with noise synthesis, the 1024px tile is **23% smaller** on average — a meaningful win that compounds across hundreds of thousands of tiles.

There is no server-side performance cost to this approach. The tile server must decode the full 1024px image regardless — L1 and L2 generation requires the complete pixel data for downsampling. Slicing 256px tiles from a decoded 1024px buffer is just pointer arithmetic (effectively free), while the 16x256px alternative would require 16 separate file reads and re-compositing for L1/L2 generation — strictly more work.

## Tile Server Architecture

The ORIGAMI tile server (`origami serve`) is a Rust HTTP server built on axum/tokio that serves DZI tiles with on-demand reconstruction and multi-tier caching.

### L2 Families

The fundamental unit of tile generation is the **L2 family**. One L2 tile (256px) corresponds to a region that contains:

- **1 L2 tile** — the parent (256px, quarter resolution)
- **4 L1 tiles** — half resolution (each 256px, covering one quadrant)
- **16 L0 tiles** — full resolution (each 256px, covering one sixteenth)

That's **21 tiles total** per L2 family. The L2 tile's coordinates determine which L1 and L0 tiles belong to it:

```
L2 tile (x, y) covers:
  L1 tiles: (2x, 2y), (2x+1, 2y), (2x, 2y+1), (2x+1, 2y+1)
  L0 tiles: (4x..4x+3, 4y..4y+3)  — a 4x4 grid
```

### Request Flow

When any L0, L1, or L2 tile is requested:

1. **Hot cache check** — In-memory LRU (moka). If hit, return immediately (~1ms).
2. **Warm cache check** — RocksDB on-disk persistent cache. If hit, promote to hot cache and return (~10-25ms).
3. **Cold miss** — Generate the **entire L2 family** (all 21 tiles):
   - For JXL baseline slides: decode the 1024px JXL source, slice into 256px L0 tiles, downsample for L1/L2
   - For ORIGAMI residual slides: decode L2 baseline, upsample, apply residual, slice
   - Store all 21 tiles in both hot and warm caches via RocksDB WriteBatch
   - Return the requested tile (~50-200ms)

4. **Singleflight** — If the same family is already being generated by another concurrent request, subsequent requests wait for the in-progress generation rather than duplicating work.

### Why Generate Entire Families?

When a viewer zooms into a region, it will immediately request neighboring tiles. By generating all 21 tiles in the family on first access, subsequent requests for siblings hit the cache. The amortized cost per tile is much lower than generating tiles individually, and the L2→L1→L0 reconstruction pipeline naturally produces all resolution levels in a single pass.

### Cache Tiers

| Tier | Storage | Latency | Capacity | Eviction |
|---|---|---|---|---|
| **Hot** | In-memory LRU (moka) | ~1ms | Configurable (default 2048 tiles) | LRU |
| **Warm** | RocksDB on SSD | ~10-25ms | Disk-limited | Optional TTL |

RocksDB key format: `tile:{slide_id}:{level}:{x}:{y}`

Family writes use WriteBatch for atomic insertion of all 21 tiles.

### Tissue-Aware Blank Tiles

When a requested tile falls outside the tissue region (as determined by the `.tissue.map` bitmap), the server returns a solid-color JPEG filled with the interpolated background color for that region of the slide. This avoids baking pure white into L1/L2 tiles at tissue margins, where the downsampled result would otherwise blend tissue with white rather than the actual slide background.

### Performance Targets

| Scenario | P95 Latency |
|---|---|
| Hot cache hit | < 10ms |
| Warm cache hit | < 25ms |
| Cold miss (family generation) | < 200ms |

## Source Data

| Metric | Mayosh (9 slides) | 3DHISTECH-1 | All 10 |
|---|---|---|---|
| Total L0 tiles | 1,035,503 | 52,864 | 1,088,367 |
| Tissue tiles | 232,754 | 29,184 | 261,938 |
| Blank tiles | 802,749 | 23,680 | 826,429 |
| **Tissue coverage** | **22.6% avg** | **55.2%** | **24.1%** |
| Original JPEG size | 5,960 MB | 548 MB | 6,508 MB |

The Mayosh slides are unusually sparse (~23% tissue). 3DHISTECH-1 represents a more typical dense specimen at 55% coverage.

## Pipeline Stages

### Tile Formats

| Variant | L0 tiles | L1/L2 | L3+ thumbnails | L0 tile size |
|---|---|---|---|---|
| **Original** | 256px JPEG (raw from DICOM) | Pre-rendered | 256px JPEG Q90 | varies |
| **JPEG Q80** | 256px JPEG Q80 (recompressed) | Pre-rendered | 256px JPEG Q80 | varies |
| **JPEG Q40** | 256px JPEG Q40 (recompressed) | Pre-rendered | 256px JPEG Q40 | varies |
| **ORIGAMI JXL Q80** | 1024px JXL Q80 (retiled) | On-demand | 256px JPEG Q90 (symlinked) | ~1.8 KB |
| **ORIGAMI JXL Q40** | 1024px JXL Q40 (retiled) | On-demand | 256px JPEG Q90 (symlinked) | ~0.7 KB |

**Static JPEG variants** are conventional DZI pyramids with every tile recompressed at the target quality. All levels are pre-rendered and served as static files — no tile server reconstruction needed.

**ORIGAMI JXL variants** retile L0 to 1024px, encode as JPEG XL, and omit L1/L2 entirely. The tile server reconstructs L1/L2 on demand from the 1024px L0 source. L3+ thumbnails are JPEG Q90, shared across all variants via symlinks.

### Per-Stage Sizes (All 10 Slides)

| Variant | L0 | L3+ thumbs | Total | vs Original |
|---|---|---|---|---|
| **Original JPEG** | 6,508 MB | (included) | 6,508 MB | 1.0x |
| **Trimmed JPEG** | — | — | 2,963 MB | 2.2x (54% saved) |
| **JPEG Q80** (static recompression) | — | — | 1,657 MB | 3.9x (75% saved) |
| **JPEG Q40** (static recompression) | — | — | 948 MB | 6.9x (85% saved) |
| **ORIGAMI JXL Q80** | 551 MB | 92 MB | 643 MB | 10.1x (90% saved) |
| **ORIGAMI JXL Q40** (with noise) | 208 MB | 92 MB | 300 MB | 21.7x (95% saved) |
| **ORIGAMI JXL Q40** (no noise) | 195 MB | 92 MB | 287 MB | 22.7x (96% saved) |

### Blank Tile Removal

| Metric | Min | Max | Avg |
|---|---|---|---|
| Tiles skipped | 44.8% | 81.3% | 74.4% |
| Size reduction ratio | 1.3x | 3.7x | 2.3x |

### JXL Compression (End-to-End vs Original)

| Variant | Min ratio | Max ratio | Avg ratio |
|---|---|---|---|
| **JXL Q80** | 4.7x | 23.6x | 14.0x |
| **JXL Q40 + noise** | 13.9x | 71.6x | 38.0x |

### JXL Compression Only (vs Trimmed, Excluding Blank Removal)

| Variant | Min ratio | Max ratio | Avg ratio |
|---|---|---|---|
| **Q80** | 3.7x | 7.8x | 5.9x |
| **Q40 + noise** | 10.9x | 23.6x | 16.2x |

### Pure Per-Tile Compression (Coverage-Independent)

| Conversion | Ratio |
|---|---|
| Original JPEG → Q80 JXL | **3.1x** |
| Original JPEG → Q40 JXL | **8.0x** |
| Q80 JXL → Q40 JXL | **2.6x** |

Average tile sizes: Original JPEG 5.5 KB, Q80 JXL 1.8 KB, Q40 JXL 0.7 KB (all at 1024px).

## Extrapolation by Tissue Coverage

For a representative 100K L0-tile slide at different tissue coverage levels, assuming the same per-tile compression ratios and ~9 MB of L3+ JPEG Q90 thumbnail overhead:

| Tissue Coverage | Tissue Tiles | Original | Trim Ratio | Q80 Total | Q80 Ratio | Q40 Total | Q40 Ratio |
|---|---|---|---|---|---|---|---|
| **23% (Mayosh avg)** | 22,569 | 558 MB | 4.4x | 50 MB | 11.1x | 25 MB | 22.4x |
| **33%** | 33,000 | 558 MB | 3.0x | 69 MB | 8.1x | 32 MB | 17.4x |
| **45% (2x Mayosh)** | 45,139 | 558 MB | 2.2x | 91 MB | 6.1x | 41 MB | 13.8x |
| **55% (3DHISTECH)** | 55,205 | 558 MB | 1.8x | 109 MB | 5.1x | 48 MB | 11.7x |

### Key Observations

1. **Blank removal** scales linearly with blank fraction: 4.4x at 23% coverage, 1.8x at 55%, 1.0x at 100%. Major win for sparse slides, negligible for dense ones.

2. **JXL compression** is the consistent win regardless of coverage: 3.1x at Q80, 8.0x at Q40 per tissue tile. Even a 100% tissue slide gets 2.9x (Q80) or 7.1x (Q40) total.

3. **L3+ thumbnail overhead** is fixed at ~92 MB for these 10 slides (~9 MB per 100K tiles). At Q40, thumbnails are 31% of total served data — the JXL L0 tiles are so small that the JPEG thumbnails become the dominant storage cost.

4. **Noise synthesis** (Q40) provides an additional 2.6x over Q80 by allowing more aggressive compression while maintaining perceptual quality through auto-calibrated photon noise ISO.

## Storage at Scale

Extrapolation assuming 33% tissue coverage, 100K L0 tiles per slide, and the measured per-tile compression ratios.

**Per-slide sizes (33% tissue coverage, 100K L0 tiles):**

| Variant | Per-slide | vs Original |
|---|---|---|
| Original JPEG | 537 MB | 1.0x |
| JPEG Q80 (static) | 166 MB | 3.2x |
| JPEG Q40 (static) | 95 MB | 5.7x |
| ORIGAMI JXL Q80 | 67 MB | 8.0x |
| ORIGAMI JXL Q40 | 32 MB | 16.9x |

ORIGAMI JXL Q80 is **2.5x smaller** than static JPEG Q80 at the same visual quality tier. At Q40, the gap widens to **3.0x** — the combination of blank tile removal, 1024px retiling, and JXL's superior compression compounds significantly.

### Projected Storage

| Slides | Original | JPEG Q80 | JPEG Q40 | ORIGAMI JXL Q80 | ORIGAMI JXL Q40 |
|---|---|---|---|---|---|
| **40,000** | 21 TB | 6.6 TB | 3.8 TB | 2.7 TB | 1.3 TB |
| **1,000,000** | 537 TB | 166 TB | 95 TB | 67 TB | 32 TB |
| **20,000,000** | 10.7 PB | 3.3 PB | 1.9 PB | 1.3 PB | 635 TB |

### Estimated Storage Costs ($0.02/GB/month, e.g. S3 Standard)

| Slides | Original | JPEG Q80 | JPEG Q40 | ORIGAMI JXL Q80 | ORIGAMI JXL Q40 |
|---|---|---|---|---|---|
| **40,000** | $420/mo | $129/mo | $74/mo | $53/mo | $25/mo |
| **1,000,000** | $10K/mo | $3.2K/mo | $1.9K/mo | $1.3K/mo | $620/mo |
| **20,000,000** | $210K/mo | $65K/mo | $37K/mo | $26K/mo | $12K/mo |

At 20M slides, ORIGAMI JXL Q40 saves **$197K/month ($2.4M/year)** vs original JPEG. Even compared to aggressive static JPEG Q40 recompression, ORIGAMI JXL Q40 saves an additional **$25K/month ($300K/year)** at 20M slides.

## JXL Noise Synthesis

The Q40 variant uses JPEG XL's `--photon_noise_iso` parameter to synthesize texture that masks compression artifacts. The ISO value is auto-determined per slide:

1. Sample ~20 central tissue tiles at 256px
2. Measure noise sigma via wavelet MAD (RMS across all subbands)
3. Scale to 1024px equivalent (×2.32 variance scale factor)
4. Interpolate pre-built calibration table → ISO value
5. Apply at 0.5x scale (half strength) for conservative synthesis
6. Clamped to [12800, 25600]

Measured ISO values: 12,142–12,800 across all slides (after 0.5x scaling).
