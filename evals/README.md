# Evaluation Infrastructure

All evaluation tools, run data, and the comparison viewer for ORIGAMI.

## Encode Pipeline

The ORIGAMI encoder compresses an image into a 3-level pyramid (L2 → L1 → L0) where each level is predicted from the one below it, and only the prediction error (residual) is stored.

### Pipeline Flow

```
                              ENCODE
┌──────────────┐
│ Source Image  │  1024x1024
│  (L0 ground  │──────────────────────────────────────────────────────────────┐
│   truth)     │                                                              │
└──────┬───────┘                                                              │
       │ downsample 2x                                                        │
       ▼                                                                      │
┌──────────────┐                                                              │
│  L1 ground   │  512x512                                                     │
│   truth      │─────────────────────────────────────┐                        │
└──────┬───────┘                                     │                        │
       │ downsample 2x                               │                        │
       ▼                                             │                        │
┌──────────────┐    ┌─────────────┐                  │                        │
│  L2 (256x256)│───▶│   OptL2?    │  gradient        │                        │
│   original   │    │  optimize   │  descent         │                        │
└──────────────┘    └──────┬──────┘                   │                        │
                           ▼                          │                        │
                    ┌──────────────┐                  │                        │
                    │ JPEG encode  │  baseq=95        │                        │
                    │ (444 or 420) │  ◄── subsamp     │                        │
                    └──────┬───────┘  decision        │                        │
                           │                          │                        │
                    ┌──────▼───────┐                  │                        │
                    │ JPEG decode  │                   │                        │
                    │ (what the    │                   │                        │
                    │ decoder sees)│                   │                        │
                    └──────┬───────┘                   │                        │
                           │                          │                        │
                           │ RGB bilinear             │                        │
                           │ upsample 2x              │                        │
                           ▼                          ▼                        │
                    ┌──────────────┐           ┌──────────────┐                │
                    │L1 prediction │           │ L1 ground    │                │
                    │  (512x512)   │           │  truth Y     │                │
                    └──────┬───────┘           └──────┬───────┘                │
                           │ RGB→YCbCr                │                        │
                           │ (float32)                │                        │
                           ▼                          ▼                        │
                    ┌─────────────────────────────────────┐                    │
                    │  L1 residual = gt_Y - pred_Y + 128  │  per tile          │
                    │  (float32 precision)                 │  (256x256)         │
                    └──────────────────┬──────────────────┘                    │
                                       │                                       │
                                       ▼                                       │
                                ┌──────────────┐                               │
                                │ JPEG encode  │  l1q (grayscale)              │
                                │  residual    │                               │
                                └──────┬───────┘                               │
                                       │                                       │
                                       │ decode + reconstruct                  │
                                       │ L1_recon_Y = pred_Y + (R_decoded-128) │
                                       ▼                                       │
                                ┌──────────────┐                               │
                                │L1 recon RGB  │  512x512                      │
                                │  bilinear    │                               │
                                │  upsample 2x │                               │
                                └──────┬───────┘                               │
                                       │                                       │
                                       ▼                          ▼            │
                                ┌──────────────┐           ┌──────────────┐    │
                                │L0 prediction │           │ L0 ground    │    │
                                │ (1024x1024)  │           │  truth Y     │◄───┘
                                └──────┬───────┘           └──────┬───────┘
                                       │ RGB→YCbCr                │
                                       │ (float32)                │
                                       ▼                          ▼
                                ┌─────────────────────────────────────┐
                                │  L0 residual = gt_Y - pred_Y + 128  │  per tile
                                │  (float32 precision)                 │  (256x256)
                                └──────────────────┬──────────────────┘
                                                   │
                                                   ▼
                                            ┌──────────────┐
                                            │ JPEG encode  │  l0q (grayscale)
                                            │  residual    │
                                            └──────────────┘
```

### What Gets Stored

| Component | Description | Typical Size (j40) |
|-----------|-------------|-------------------|
| L2 baseline | JPEG-encoded 256x256 tile at baseq=95 | ~63 KB (444) / ~47 KB (420) |
| L1 residuals | 4 grayscale JPEG tiles (256x256 each) | ~5 KB total |
| L0 residuals | 16 grayscale JPEG tiles (256x256 each) | ~47 KB total |
| **Total** | | **~115 KB (444+OptL2)** |

### Key Design Decisions

#### 1. Chroma Subsampling on L2: 4:4:4 vs 4:2:0

**Recommendation: 4:4:4** (current default)

The L2 tile is the foundation of the entire pyramid. Its chroma channels are bilinear-upsampled 2x for L1 predictions and 4x for L0 predictions. Any chroma quantization error in L2 is amplified through these upsampling stages.

| Config | Total Bytes (j40) | Avg L0 Delta E |
|--------|-------------------|----------------|
| **444 + OptL2** | **114,147** | **2.27** |
| 420opt + OptL2 | 102,440 | 3.09 |
| 420 + OptL2 | 98,926 | 3.36 |
| JPEG baseline | 165,972 | 2.95 |

- 4:4:4 costs ~15 KB more than 4:2:0 but achieves 1 full Delta E unit better color accuracy
- 4:4:4 actually beats the JPEG baseline in both size AND perceptual quality
- 4:2:0 saves bytes on the L2 base but the chroma loss propagates through L1 and L0

#### 2. OptL2 Gradient Descent

**Recommendation: Always enable** (`--optl2`)

OptL2 optimizes the L2 pixel values before JPEG encoding so that when the JPEG-decoded L2 is bilinear-upsampled to predict L1, the prediction error is minimized. This costs ~2.5s of encode time.

| Config | L1 Residuals | L0 Residuals | Total | Delta E |
|--------|-------------|-------------|-------|---------|
| 444 (no opt) | 11,372 | 51,483 | 115,730 | 2.38 |
| **444 + OptL2** | **4,858** | **46,653** | **114,147** | **2.27** |

OptL2 reduces L1 residual size by 57% and improves Delta E by 0.11 — a free win.

#### 3. Chroma Optimization (420opt)

**Conclusion: Not worth it when using 4:4:4.**

We experimented with gradient-descent-optimized chroma downsampling for 4:2:0. The `420opt` variant applies the same optimization technique to the half-resolution Cb/Cr planes, finding values that minimize error after the JPEG decoder's chroma upsample.

| Config | Total Bytes (j40) | Delta E | Delta vs 420 |
|--------|-------------------|---------|-------------|
| 420 + OptL2 | 98,926 | 3.36 | baseline |
| **420opt + OptL2** | **102,440** | **3.09** | **-0.27 dE, +3.5 KB** |

The 420opt optimization consistently improves Delta E by ~0.26 across all quality levels for ~3-4 KB extra. However, since 4:4:4 is the recommended subsampling mode (and eliminates chroma loss entirely), this optimization is unnecessary for the default pipeline. It remains available for size-constrained scenarios where 4:2:0 is required.

#### 4. L2 Baseline Quality (baseq)

**Default: Q95** (see [l2_baseq_tradeoff.md](scripts/l2_baseq_tradeoff.md))

We swept Q95–Q99 and found Q97 to be the sweet spot (51% of Q99's quality gain at 32% of byte cost). However, since the improvement is modest (Delta E 2.42→2.26 at Q97, costing +6 KB), we keep the default at Q95 which is already excellent.

#### 5. Split Quality (L1 vs L0 residuals)

**Recommendation: L1 quality = L0 quality + 20** (see [split-quality-research.md](split-quality-research.md))

L1 tiles have 4x downstream impact (each L1 predicts 4 L0 tiles), so investing more bits in L1 residuals pays off. A split like `--l1q 60 --l0q 40` achieves +1.5 dB better minimum PSNR than uniform `--resq 40` at the same byte count.

#### 6. Residuals Are Grayscale

Only the Y (luma) channel residual is stored. The Cb/Cr (chroma) channels are predicted from the parent level's upsampled chroma and reused directly. This means chroma subsampling on L2 affects chroma quality at all levels, but residual quality settings only affect luma.

#### 7. Float32 Precision Pipeline

The encoder uses float32 throughout the prediction pipeline:
- YCbCr conversion preserves Cb/Cr as f32 (not quantized to u8)
- Bilinear upsample operates in RGB space (not YCbCr)
- Residual computation: `round(gt_u8 - pred_f32 + 128)` avoids u8 quantization before subtraction
- This reduces Delta E by ~25% compared to the u8-quantized pipeline (3.24→2.42)

### Recommended Configuration

**`origami encode --image <path> --subsamp 444 --optl2 --l1q 60 --l0q 40`**

4:4:4 chroma + OptL2 + split quality (L1=60, L0=40). This balances file size, perceptual quality, and luma fidelity:

| Metric | Value |
|--------|-------|
| Total size | ~116 KB (for 1024x1024) |
| L0 avg Delta E | 2.17 |
| L0 avg PSNR | 36.15 dB |
| L0 avg SSIM | 0.935 |
| vs JPEG Q40 | 30% smaller, better Delta E |

Run name in viewer: `rs_444_optl2_l1q60_l0q40`

## Directory Structure

```
evals/
  scripts/                    # Run generation scripts
    wsi_residual_debug_with_manifest.py  # ORIGAMI run generator
    jpeg_baseline.py           # JPEG baseline generator
    jpeg_encoder.py            # Shared encoder module
    run_all_captures.sh        # Batch runner (all JPEG qualities)
    run_jpegli_captures.sh     # Jpegli batch runner
  analysis/                   # Metrics & analysis tools
    bd_metrics.py              # BD-Rate calculations
    wsi_residual_analyze.py    # Analysis tool
    generate_metric_heatmaps.py # Heatmap generator
  viewer/                     # Node.js comparison viewer
    viewer-server.js           # Express server (API + static)
    public/index.html          # Standalone HTML frontend
  runs/                       # All run output data (gitignored)
  test-images/                # Source images for generation
    L0-1024.jpg
    L0-2048.jpg
  archive/                    # Legacy scripts & data (gitignored)
```

## Quick Start

### Generating Runs

All scripts should be run from the project root directory.

```bash
# Single ORIGAMI run
python evals/scripts/wsi_residual_debug_with_manifest.py \
    --image evals/test-images/L0-1024.jpg \
    --resq 50 --pac

# Single JPEG baseline
uv run python evals/scripts/jpeg_baseline.py \
    --image evals/test-images/L0-1024.jpg --quality 70

# All JPEG qualities (batch)
bash evals/scripts/run_all_captures.sh

# Jpegli baselines + ORIGAMI (batch)
bash evals/scripts/run_jpegli_captures.sh
```

Parameters for ORIGAMI runs:
- `--resq` - JPEG quality for residual images (10-100)
- `--l1q` - Override quality for L1 residuals (default: use `--resq`)
- `--l0q` - Override quality for L0 residuals (default: use `--resq`)
- `--baseq` - JPEG quality for baseline L2 tile (default 95)
- `--pac` - Create a PAC file for tile serving
- `--tile` - Tile size in pixels (default 256)
- `--encoder` - `libjpeg-turbo` (default), `jpegli`, `mozjpeg`, `jpegxl`, or `webp`
- `--out` - Output directory (defaults to `evals/runs/...`)

Split-quality example (higher L1, lower L0):
```bash
python evals/scripts/wsi_residual_debug_with_manifest.py \
    --image evals/test-images/L0-1024.jpg \
    --l1q 70 --l0q 40 --pac
```

See [split-quality-research.md](split-quality-research.md) for the full analysis showing that balanced splits gain +1.5 dB for free.

### Viewing Results

```bash
cd evals/viewer
pnpm install   # first time only
pnpm start     # http://localhost:8084
```

The viewer automatically scans `evals/runs/` for all run directories.

## Run Matrix

### Rust Encoder Runs (current)

| Category | Subsamp | OptL2 | Dir Pattern |
|----------|---------|-------|-------------|
| ORIGAMI (uniform) | 4:4:4 | No | `rs_444_j{Q}` |
| ORIGAMI (uniform) | 4:4:4 | Yes | `rs_444_optl2_j{Q}` |
| ORIGAMI (uniform) | 4:2:0 | Yes | `rs_420_optl2_j{Q}` |
| ORIGAMI (uniform) | 4:2:0 opt | Yes | `rs_420opt_optl2_j{Q}` |
| ORIGAMI (split) | 4:4:4 | No | `rs_444_l1q{N}_l0q{N}` |
| ORIGAMI (split) | 4:4:4 | Yes | `rs_444_optl2_l1q{N}_l0q{N}` |
| ORIGAMI (split) | 4:2:0 | Yes | `rs_420_optl2_l1q{N}_l0q{N}` |
| ORIGAMI (split) | 4:2:0 opt | Yes | `rs_420opt_optl2_l1q{N}_l0q{N}` |

### Baselines & Legacy

| Category | Encoder | Dir Pattern |
|----------|---------|-------------|
| JPEG Baseline | Pillow (libjpeg) | `jpeg_baseline_q{Q}` |
| JPEG Baseline | jpegli | `jpegli_jpeg_baseline_q{Q}` |
| JPEG Baseline | webp | `webp_jpeg_baseline_q{Q}` |
| ORIGAMI (Python, legacy) | libjpeg-turbo | `optl2_debug_j{J}_pac` |
| ORIGAMI (Python, legacy) | libjpeg-turbo | `debug_j{J}_pac` |

## Metrics

- **PSNR** - Peak Signal-to-Noise Ratio (dB)
- **SSIM** - Structural Similarity Index (0-1)
- **MSE** - Mean Squared Error
- **VIF** - Visual Information Fidelity (0-1)
- **Delta E** - CIE76 color difference

## Analysis Tools

```bash
# BD-Rate calculations
python evals/analysis/bd_metrics.py --runs-dir evals/runs

# Generate metric heatmaps
python evals/analysis/generate_metric_heatmaps.py --runs-dir evals/runs

# Analyze a specific run
python evals/analysis/wsi_residual_analyze.py --run evals/runs/debug_j50_pac
```
