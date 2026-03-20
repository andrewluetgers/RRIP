# WSI Storage Analysis for ORIGAMI Tile Server

## Overview

A whole-slide image (WSI) is a gigapixel scan of a tissue specimen on a glass slide. A single slide can be 100,000+ pixels on a side. To view these at any zoom level without loading the entire image, we use a **Deep Zoom Image (DZI) pyramid** — a hierarchical set of pre-rendered tiles at progressively lower resolutions.

The ORIGAMI pipeline compresses these pyramids into a **modified DZI structure** that achieves **10–22x total compression vs. the original DICOM-extracted tiles** (10.1x at Q80, 21.7x at Q40). Both the original and compressed variants use tissue detection to omit blank tiles, so the apples-to-apples comparison against the **trimmed JPEG** baseline is 4.6x (Q80) to 9.9x (Q40) — achieved through three independent, multiplicative sources:

| Source | Mechanism | Compression |
|---|---|---|
| **L1/L2 omission** | Tile server generates L1/L2 on demand from L0 | 1.31x |
| **JXL encoding** | JPEG XL re-encodes tissue tiles | 3.1x (Q80) / 8.0x (Q40) |
| **1024px tile packing** | 4x4 tile groups encoded as single JXL | 1.13x (Q80) / 1.23x (Q40) |

Rather than pre-rendering every resolution level, L1 and L2 are omitted — the tile server reconstructs them from L0 at request time. Levels L3+ are pre-computed and stored as JPEG, since they constitute a negligible fraction of total storage (~9 MB per 100K-tile slide). This tradeoff is optimal at the L2 family boundary: serving any L0/L1/L2 tile requires reading at most one 1024px source tile, but extending on-demand generation to L3 would require 4 source tiles, L4 would require 16 — IO cost grows 4x per level.

The tile server's caching strategy complements this structure. When any L0, L1, or L2 tile is requested, the server generates the **entire L2 family** (1 L2 + 4 L1 + 16 L0 = 21 tiles) and inserts them into a two-tier cache. This amortizes typical viewer behavior: zooming in from an L2 finds all deeper tiles already cached, and panning at L0 finds neighbors pre-generated from the same family. L3+ tiles are served directly from pre-rendered files, so overview zoom loads are instantaneous.


## Example Slide: Mayosh-3
Here are sample views from an Open Seadragon viewer of a slide from the Mayo Safe Harbor 40K cohort in JXL 80 and 40 variants vs original JPEG:

1x zoom of full resolution L0 tiles
![Mayosh-3-1x.png](Mayosh-3-1x.png)

2x zoom
![Mayosh-3-2x.png](Mayosh-3-2x.png)

5x zoom subtle loss of detail and edge artifacts visible upon close inspection of the Q40 variant
![Mayosh-3-5x.png](Mayosh-3-5x.png)

### Tile Server Modified DZI Pyramid Structure

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

### Three Sources of Compression

All comparisons use the **trimmed JPEG** as baseline — both the original and compressed variants omit blank tiles via tissue detection, so blank removal is not counted as a compression source. The three mechanisms below multiply together to produce the aggregate savings. L3+ thumbnails are a fixed ~9 MB overhead regardless of variant.

#### Blank Tile Removal (Shared Baseline)

Tissue detection identifies which tiles contain actual specimen vs. empty glass background. Blank tiles are omitted from storage in all variants, including the JPEG baseline. Across our test set, blank tiles are **85% of tile count but only 61% of total bytes** (blank tiles compress very well as JPEGs — averaging 3.4 KB vs 12.0 KB for tissue tiles).

| Metric | Blank tiles | Tissue tiles |
|---|---|---|
| Count (% of total) | 85% | 15% |
| Bytes (% of total) | 61% | 39% |
| Avg size | 3.4 KB | 12.0 KB |
| Size ratio | 1x | 3.6x |

The primary value of tissue detection beyond storage is enabling the JXL pipeline to avoid wasting encode time and decode resources on blank regions.

#### 1. L1/L2 Omission (1.31x)

A standard DZI pyramid stores tiles at every resolution level. In the quadtree structure, each level has 1/4 the tiles of the level below, so a complete L0+L1+L2 pyramid contains N + N/4 + N/16 = 21N/16 tiles worth of data. By omitting L1 and L2 (generating them on demand from L0), we store only 16/21 of the high-resolution data — a **1.31x reduction** independent of tissue coverage or encoding format.

This ratio is a geometric constant of the quadtree: for any three consecutive levels, the lowest-resolution level is always 5/21 of the total, making L2 family–based reconstruction an inherently efficient tradeoff.

#### 2. JXL Encoding (3.1x at Q80, 8.0x at Q40)

JPEG XL re-encodes tissue tiles at either Q80 (high quality, 3.1x smaller per tile) or Q40 with noise synthesis (8.0x smaller per tile). This ratio is constant regardless of tissue coverage and is the **dominant source of savings**.

| Conversion | Ratio |
|---|---|
| Trimmed JPEG → Q80 JXL | **3.1x** |
| Trimmed JPEG → Q40 JXL | **8.0x** |
| Q80 JXL → Q40 JXL | **2.6x** |

Average tile sizes: Trimmed JPEG 5.5 KB, Q80 JXL 1.8 KB, Q40 JXL 0.7 KB (all at 1024px).

#### 3. 1024px Tile Packing (1.13x at Q80, 1.23x at Q40)

The compressed variants store L0 as 1024px tiles (4x4 groups of the DZI-native 256px tiles) rather than individual 256px files. This improves compression because JXL's VarDCT transform and adaptive quantization work dramatically better with more spatial context — 16x more area to find patterns, use larger DCT block sizes, and amortize per-tile header overhead. It also reduces file count by 16x (100K tiles → ~6K files).

The tile server transparently slices 1024px tiles back into 256px for serving, and downsamples them to generate L1/L2 on demand. There is no server-side performance cost — the tile server must decode the full 1024px image regardless for L1/L2 generation, and slicing 256px tiles from a decoded buffer is just pointer arithmetic.

**Empirical evidence** (measured across 10 randomly sampled L2 families from 3DHISTECH-1):

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

The benefit is more pronounced at Q40 because the encoder has more freedom to redistribute bits across the larger tile area.

#### Combined Effect

All three sources multiply against the trimmed JPEG baseline: **4.6x at Q80, 9.9x at Q40** across our 10-slide test set (2,963 MB trimmed → 643 MB Q80 / 300 MB Q40). JXL encoding is the dominant factor, but each source contributes meaningfully — at scale, even 1.31x from L1/L2 omission saves hundreds of terabytes across millions of slides.

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

| Stage | L0 tiles | L3+ thumbnails | L0 tile size | L3+ quality |
|---|---|---|---|---|
| **Full DZI** | 256px JPEG (raw from DICOM) | 256px JPEG Q90 (Lanczos3 downsample) | varies | 90 |
| **Trimmed DZI** | 256px JPEG (tissue only) | 256px JPEG Q90 (unchanged) | varies | 90 |
| **JXL Q80** | 1024px JXL Q80 | 256px JPEG Q90 (symlinked) | ~1.8 KB/tile | 90 |
| **JXL Q40** | 1024px JXL Q40 + noise synthesis | 256px JPEG Q90 (symlinked) | ~0.7 KB/tile | 90 |

L3+ thumbnail levels (L3 through the single-tile thumbnail) are always JPEG Q90, generated during initial DZI pyramid creation by Lanczos3 downsampling. They are shared across all variants via symlinks — never re-encoded.

### Per-Stage Sizes (All 10 Slides)

| Stage | L0 | L3+ thumbs | Total | vs Original |
|---|---|---|---|---|
| **Original JPEG** | 6,508 MB | (included) | 6,508 MB | 1.0x |
| **Trimmed JPEG** | — | — | 2,963 MB | 2.2x (54% saved) |
| **JXL Q80** | 551 MB | 92 MB | 643 MB | 10.1x (90% saved) |
| **JXL Q40 + noise** | 208 MB | 92 MB | 300 MB | 21.7x (95% saved) |

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

**Per-slide sizes:**
- Original JPEG: 537 MB
- JXL Q80: 67 MB (8.0x)
- JXL Q40: 32 MB (16.9x)

### Projected Storage

| Slides | Original | JXL Q80 | Q80 Saved | JXL Q40 | Q40 Saved |
|---|---|---|---|---|---|
| **40,000** | 21 TB | 3 TB | 19 TB | 1 TB | 20 TB |
| **1,000,000** | 537 TB | 67 TB | 470 TB | 32 TB | 505 TB |
| **20,000,000** | 10.7 PB | 1.3 PB | 9.4 PB | 635 TB | 10.1 PB |

### Estimated Storage Costs ($0.02/GB/month, e.g. S3 Standard)

| Slides | Original | JXL Q80 | Q80 Savings | JXL Q40 | Q40 Savings |
|---|---|---|---|---|---|
| **40,000** | $420/mo | $53/mo | $367/mo | $25/mo | $395/mo |
| **1,000,000** | $10K/mo | $1K/mo | $9K/mo | $620/mo | $10K/mo |
| **20,000,000** | $210K/mo | $26K/mo | $184K/mo | $12K/mo | $197K/mo |

At 20M slides, Q40 saves $197K/month ($2.4M/year) vs original JPEG storage.

## JXL Noise Synthesis

The Q40 variant uses JPEG XL's `--photon_noise_iso` parameter to synthesize texture that masks compression artifacts. The ISO value is auto-determined per slide:

1. Sample ~20 central tissue tiles at 256px
2. Measure noise sigma via wavelet MAD (RMS across all subbands)
3. Scale to 1024px equivalent (×2.32 variance scale factor)
4. Interpolate pre-built calibration table → ISO value
5. Apply at 0.5x scale (half strength) for conservative synthesis
6. Clamped to [12800, 25600]

Measured ISO values: 12,142–12,800 across all slides (after 0.5x scaling).
