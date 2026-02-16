# ORIGAMI GPU Encoder

GPU-accelerated residual pyramid encoder using CUDA and nvJPEG on NVIDIA GPUs.

## Overview

The GPU encoder (`origami-gpu-encode`) provides two modes:
- **Single-image encode** — for eval comparisons (mirrors the CPU `origami encode` pipeline)
- **DICOM WSI encode** — batch encode entire whole-slide images from DICOM files

All compute-intensive operations run on GPU: downsample, upsample, YCbCr conversion, OptL2 gradient descent, residual computation, and JPEG encode/decode (via nvJPEG).

## Requirements

- NVIDIA GPU with CUDA 12.4+ (tested on B200)
- CUDA Toolkit 12.6+ (for nvcc/PTX compilation)
- Rust 1.75+

## Building

```bash
cd gpu-encode
cargo build --release
```

The build system compiles CUDA kernels (.cu files) to PTX via nvcc during `cargo build`.

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
    --baseq 80 --l1q 60 --l0q 40 \
    --subsamp 444 --optl2 --max-delta 20 \
    --pack
```

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
| `--optl2` | off | Enable OptL2 gradient descent |
| `--max-delta` | 15 | Max pixel deviation for OptL2 |
| `--pack` | off | Generate residual pack/bundle files |
| `--manifest` | off | Write manifest.json with per-tile metrics |
| `--debug-images` | off | Save debug PNGs (originals, predictions, residuals) |
| `--max-parents` | all | Limit number of families to encode (for testing) |
| `--batch-size` | 64 | Families per GPU batch (WSI mode) |
| `--tile` | 256 | Tile size in pixels |
| `--device` | 0 | CUDA device index |

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

### Test Datasets

Both datasets are from the [OpenSlide test data collection](https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/):

| Dataset | Dimensions | Tile Size | Frames | File Size |
|---------|-----------|-----------|--------|-----------|
| 3DHISTECH-1 | 57344x60416 | 1024x1024 | 3,304 | 270 MB |
| 3DHISTECH-2 | 267776x370688 | 512x512 | 41,382 (split: 25264+16118) | 1.6 GB |

Download URLs:
- https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-1.zip
- https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-2.zip

## Architecture

### CUDA Kernels (`src/kernels/`)

| Kernel | File | Description |
|--------|------|-------------|
| `upsample_bilinear_2x` | upsample.cu | 2x bilinear upsample (f32) |
| `downsample_lanczos3_{h,v}` | upsample.cu | Lanczos3 downsample (separable h/v) |
| `rgb_to_ycbcr_f32` | residual.cu | RGB u8 → YCbCr f32 (BT.601) |
| `ycbcr_to_rgb` | residual.cu | YCbCr f32 → RGB u8 (BT.601 inverse) |
| `compute_residual` | residual.cu | gt_y(u8) - pred_y(f32) + 128, clamped |
| `reconstruct_y` | residual.cu | pred_y(f32) + residual(u8) - 128, clamped |
| `optl2_step` | optl2.cu | One gradient descent step for OptL2 |
| `composite_kernel` | composite.cu | Composite 4x4 tiles into canvas |

### Pipeline Flow (WSI Mode)

```
For each family (4x4 tile group):
  1. Extract 16 tile JPEG bytes from DICOM fragments
  2. nvJPEG decode → 16 GPU buffers (RGB u8)
  3. Download + composite into 4Kx4K canvas (CPU memcpy)
  4. Upload canvas → GPU f32
  5. Lanczos3 downsample → L1 (2Kx2K) and L2 (1Kx1K)
  6. OptL2 gradient descent (100 iterations, GPU)
  7. nvJPEG encode L2 baseline
  8. nvJPEG decode L2 → upsample 2x → L1 prediction
  9. For each 2x2 L1 tile:
     - Extract Y, compute residual (GPU)
     - nvJPEG encode grayscale residual
     - Decode + reconstruct Y (GPU)
  10. GPU: YCbCr→RGB on reconstructed L1 mosaic
  11. Upsample L1 → L0 prediction (GPU)
  12. For each 4x4 L0 tile:
      - Compute Y residual (GPU, using original decoded tile)
      - nvJPEG encode grayscale residual
  13. Pack entries into LZ4-compressed bundle
```

### DICOM Reader

The DICOM reader (`src/dicom.rs`) parses DICOM WSI files:
- Extracts per-frame JPEG bytes from encapsulated pixel data fragments
- Reads `TotalPixelMatrixColumns`/`TotalPixelMatrixRows` for full image dimensions
- Uses `Columns`/`Rows` for per-tile dimensions
- No memory mapping needed — fragment bytes are owned by the `dicom` crate parser
