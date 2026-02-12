# Evaluation Infrastructure

All evaluation tools, run data, and the comparison viewer for ORIGAMI.

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

| Category | Encoder | Dir Pattern |
|----------|---------|-------------|
| JPEG Baseline | libjpeg-turbo | `jpeg_baseline_q{Q}` |
| JPEG Baseline | jpegli | `jpegli_jpeg_baseline_q{Q}` |
| JPEG Baseline | webp | `webp_jpeg_baseline_q{Q}` |
| ORIGAMI (uniform) | libjpeg-turbo | `debug_j{J}_pac` |
| ORIGAMI (uniform) | jpegli | `jpegli_debug_j{J}_pac` |
| ORIGAMI (uniform) | webp | `webp_debug_j{J}_pac` |
| ORIGAMI (split) | libjpeg-turbo | `debug_l1q{N}_l0q{N}_pac` |
| ORIGAMI (legacy) | libjpeg-turbo | `debug_q{Q}_j{J}_pac` |

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
