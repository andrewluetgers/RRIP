# ORIGAMI GPU Encoder

GPU-accelerated residual pyramid encoder using CUDA and nvJPEG on NVIDIA GPUs.

## Overview

The GPU encoder (`origami-gpu-encode`) provides two modes:
- **Single-image encode** — for eval comparisons (mirrors the CPU `origami encode` pipeline)
- **DICOM WSI encode** — batch encode entire whole-slide images from DICOM files

All compute-intensive operations run on GPU: downsample, upsample, YCbCr conversion, OptL2 gradient descent, unsharp mask sharpening, residual computation, and JPEG encode/decode (via nvJPEG).

## Requirements

- NVIDIA GPU with CUDA 12.4+ (tested on B200)
- CUDA Toolkit 12.6+ (for nvcc/PTX compilation)
- Rust 1.75+

## Building

```bash
cd gpu-encode
cargo build --release
```

The build system compiles CUDA kernels (.cu files) to PTX via nvcc during `cargo build`. If nvcc is not in `$PATH`, set `NVCC=/usr/local/cuda/bin/nvcc` or add CUDA to PATH before building.

For non-default GPU architectures, set `CUDA_ARCH`:
```bash
CUDA_ARCH=sm_90 cargo build --release   # H100
CUDA_ARCH=sm_80 cargo build --release   # A100
CUDA_ARCH=sm_100 cargo build --release  # B200 (default)
```

## Usage

### Single-Image Encode

```bash
origami-gpu-encode encode \
    --image path/to/image.jpg \
    --out output_dir \
    --baseq 95 --l1q 60 --l0q 40 \
    --subsamp 444 --optl2 --max-delta 20 \
    --manifest --debug-images
```

### DICOM WSI Encode

```bash
origami-gpu-encode encode \
    --slide path/to/file.dcm \
    --out output_dir \
    --baseq 60 --l1q 60 --l0q 40 \
    --subsamp 444 --optl2 --max-delta 15 \
    --pack --profile
```

### L2 Sharpening (OptL2 Approximation)

Unsharp mask sharpening on L2 tiles approximates OptL2 gradient descent at ~500x less cost. Two modes are available:

**Decode-time sharpen (default)** — saves the un-sharpened L2 to disk; sharpening is applied to the decoded L2 before upsampling for prediction. The decoder applies the same sharpen at serve time:
```bash
origami-gpu-encode encode \
    --slide path/to/file.dcm --out output_dir \
    --baseq 60 --l1q 60 --l0q 40 --subsamp 444 \
    --sharpen 0.5 --pack
```

**Save-sharpened** — bakes sharpening into the saved L2 file. The decoder does NOT need to sharpen:
```bash
origami-gpu-encode encode \
    --slide path/to/file.dcm --out output_dir \
    --baseq 60 --l1q 60 --l0q 40 --subsamp 444 \
    --sharpen 0.5 --save-sharpened --pack
```

Typical strength values: 0.5–2.0. The sharpen kernel is a separable 3×3 Gaussian blur followed by `out = src + strength × (src − blurred)`.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | — | Single image path (mutually exclusive with `--slide`) |
| `--slide` | — | DICOM WSI file path |
| `--out` | — | Output directory |
| `--baseq` | 95 | JPEG quality for L2 baseline |
| `--l1q` | (resq) | JPEG quality for L1 residuals |
| `--l0q` | (resq) | JPEG quality for L0 residuals |
| `--resq` | 50 | Default residual quality (if l1q/l0q not set) |
| `--subsamp` | 444 | Chroma subsampling (444 or 420) |
| `--optl2` | off | Enable OptL2 gradient descent (100 iterations) |
| `--max-delta` | 15 | Max pixel deviation for OptL2 |
| `--sharpen` | — | Unsharp mask strength for L2 (e.g. 0.5–2.0) |
| `--save-sharpened` | off | Save the sharpened L2 (default: save un-sharpened, sharpen at decode time) |
| `--pack` | off | Generate residual pack/bundle files |
| `--manifest` | off | Write manifest.json with per-tile metrics |
| `--debug-images` | off | Save debug PNGs (originals, predictions, residuals) |
| `--l2resq` | 95 | L2 RGB residual quality (0 = skip) |
| `--max-parents` | all | Limit number of families to encode (for testing) |
| `--batch-size` | 64 | Families per GPU batch (WSI mode) |
| `--tile` | 256 | Tile size in pixels |
| `--profile` | off | Enable per-stage timing (adds GPU sync points, writes timing_report.json) |
| `--device` | 0 | CUDA device index |

## Pipeline Timing & Profiling

Use `--profile` to get a detailed per-stage timing breakdown. This adds GPU sync points between stages (serializes the pipeline) and prints a report at the end:

```
=== Pipeline Timing Report ===
DICOM open:            0.93s
Families total:        94844 (6448 with data, 88396 empty/skipped)
Time in family loop:   101.31s
  gather_tiles:        0.87s  (0.9%)
  nvjpeg_decode:       9.16s  (9.0%)
  composite:           3.59s  (3.5%)
  downsample:          1.07s  (1.1%)
  optl2:               0.00s  (0.0%)
  l2_encode:           48.39s  (47.8%)
  l2_decode:           0.66s  (0.6%)
  l1_predict:          0.88s  (0.9%)
  l1_residuals:        11.06s  (10.9%)
  l1_recon_rgb:        1.06s  (1.0%)
  l0_predict:          1.48s  (1.5%)
  l0_residuals:        23.00s  (22.7%)
  pack:                0.12s  (0.1%)
```

Also writes `timing_report.json` and reports GPU utilization, VRAM usage, and power draw per stage.

### Known Bottlenecks

The dominant cost is **nvJPEG encode** — 21 sequential JPEG encodes per family (1 L2 RGB + 4 L1 grayscale + 16 L0 grayscale). This accounts for ~75% of family loop time (l2_encode + l1_residuals + l0_residuals). GPU utilization is typically 15–25% for a single pipeline; running multiple concurrent encodes (5–15) can saturate the GPU to ~98%.

## RunPod GPU Development

### Quick Start

```bash
# 1. Set up environment
export RUNPOD_API_KEY="your_key_here"
# Ensure ~/.ssh/id_runpod exists

# 2. Set up the pod (sync code, install Rust, build)
./gpu-encode/scripts/pod-setup.sh

# 3. Run single-image eval
./gpu-encode/scripts/encode-single-image.sh

# 4. Run WSI encode
./gpu-encode/scripts/encode-wsi.sh --dataset 3DHISTECH-1
./gpu-encode/scripts/encode-wsi.sh --dataset all
```

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/pod-setup.sh` | Query RunPod API, rsync code, install Rust, build encoder |
| `scripts/encode-single-image.sh` | Encode a single test image on the pod, download results |
| `scripts/encode-wsi.sh` | Download DICOM test data, encode WSIs, report timing |

### Manual SSH

```bash
# Get pod SSH details
curl -s -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{"query":"{ myself { pods { id name desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"}' \
  https://api.runpod.io/graphql

# SSH in
ssh -i ~/.ssh/id_runpod root@<IP> -p <PORT>
```

### Code Sync

**Important**: Use `git push` + `git pull` to sync code to/from the pod. Do NOT use rsync — it copies massive build artifacts. Use `scp` for targeted file transfers (results, data).

```bash
# Push code changes to pod
git push
ssh -i ~/.ssh/id_runpod root@<IP> -p <PORT> \
  "source ~/.cargo/env && cd /workspace/RRIP && git pull && cd gpu-encode && cargo build --release"

# Download results
scp -i ~/.ssh/id_runpod -P <PORT> -r root@<IP>:/workspace/RRIP/evals/runs/my_run ./evals/runs/
```

### Test Datasets

All datasets from the [OpenSlide test data collection](https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/):

| Dataset | Dimensions | Tile Size | Families | File Size |
|---------|-----------|-----------|----------|-----------|
| 3DHISTECH-1 | 57344×60416 | 1024×1024 | 210 | 270 MB |
| 3DHISTECH-2 | 267776×370688 | 512×512 | 23,711 | 1.6 GB |
| 3DHISTECH-2-256 | 267776×370688 | 256×256 | 94,844 (6,448 with data) | 1.2 GB |

3DHISTECH-2-256 is created by retiling 3DHISTECH-2 to 256×256 tiles using `evals/scripts/retile.py`. Blank tiles are stored as empty DICOM fragments so the encoder can skip them (93.2% of families are empty).

Download URLs:
- https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-1.zip
- https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-2.zip

## Architecture

### CUDA Kernels (`src/kernels/`)

| Kernel | File | Description |
|--------|------|-------------|
| `upsample_bilinear_2x` | upsample.cu | 2× bilinear upsample (f32, batched) |
| `downsample_lanczos3_{h,v}` | upsample.cu | Lanczos3 downsample (separable h/v) |
| `rgb_to_ycbcr_f32` | residual.cu | RGB u8 → YCbCr f32 (BT.601) |
| `ycbcr_to_rgb` | residual.cu | YCbCr f32 → RGB u8 (BT.601 inverse) |
| `compute_residual` | residual.cu | gt_y(u8) − pred_y(f32) + 128, clamped |
| `reconstruct_y` | residual.cu | pred_y(f32) + residual(u8) − 128, clamped |
| `optl2_step` | optl2.cu | One gradient descent step for OptL2 |
| `composite_kernel` | composite.cu | Composite 4×4 tiles into canvas |
| `unsharp_hblur_kernel` | sharpen.cu | Horizontal Gaussian blur [0.25, 0.5, 0.25] |
| `unsharp_vblur_sharpen_kernel` | sharpen.cu | Vertical blur + sharpen: out = src + strength × (src − blurred) |

### Pipeline Flow (WSI Mode)

```
For each family (4×4 tile group):
  0. Skip if all 16 tiles are empty (93% of families for sparse WSIs)
  1. Extract 16 tile JPEG bytes from DICOM fragments
  2. nvJPEG decode → 16 GPU buffers (RGB u8)
  3. Download + composite into 4T×4T canvas (CPU memcpy)
  4. Upload canvas → GPU f32
  5. Lanczos3 downsample → L1 (2T×2T) and L2 (T×T)
  6. [if --optl2] Gradient descent optimization (100 iterations, GPU)
  7. [if --sharpen --save-sharpened] GPU unsharp mask on L2
  8. nvJPEG encode L2 baseline → save to disk
  9. nvJPEG decode L2
 10. [if --sharpen, no --save-sharpened] GPU unsharp mask on decoded L2
 11. Upsample L2 2× → L1 prediction (GPU)
 12. For each 2×2 L1 tile:
     - Extract Y, compute residual (GPU)
     - nvJPEG encode grayscale residual
     - Decode + reconstruct Y (GPU)
 13. GPU: YCbCr→RGB on reconstructed L1 mosaic
 14. Upsample L1 → L0 prediction (GPU)
 15. For each 4×4 L0 tile:
     - Compute Y residual (GPU, using original decoded tile)
     - nvJPEG encode grayscale residual
 16. Pack L1+L0 residual entries into LZ4-compressed bundle
```

L2 baseline JPEGs are saved as separate files (not in the pack). Only L1 and L0 residuals go in the bundle.

### DICOM Reader

The DICOM reader (`src/dicom.rs`) parses DICOM WSI files:
- Extracts per-frame JPEG bytes from encapsulated pixel data fragments
- Reads `TotalPixelMatrixColumns`/`TotalPixelMatrixRows` for full image dimensions
- Uses `Columns`/`Rows` for per-tile dimensions
- Empty fragments (zero-length) are treated as blank tiles and skipped
