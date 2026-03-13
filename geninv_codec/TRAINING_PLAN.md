# Training Plan: Domain-Specific 4x Super-Resolution for WSI Tiles

## Task

Train a tiny CNN to do 4x super-resolution:
- **Input:** 256x256 L2 base tile (JXL-compressed, the same tile ORIGAMI already stores)
- **Output:** 1024x1024 L0 tile (matching original WSI quality)
- **Domain:** H&E and IHC stained whole-slide images

At decode time this replaces both the lanczos3 upsample AND the residual — just the L2 base + model forward pass = high-quality L0.

## Architecture: SESR-M5 (Reparameterizable)

Based on ARM Research's "Collapsible Linear Blocks" (MLSys 2022).

### Training Mode (multi-branch, ~73K params)
```
Input (256x256x3)
  → CollapsibleBlock x5 (each: 3x3 conv + residual + batch norm, ch=16)
  → 3x3 conv → 48 channels (3 * 4^2 for pixel shuffle)
  → PixelShuffle(4) → 1024x1024x3
  → residual connection: input bilinear-upsampled 4x + model output
```

Each CollapsibleBlock at training time:
```
x → [3x3 conv → BN → ReLU] + [1x1 conv → BN] + identity → ReLU
```

### Inference Mode (collapsed, plain convs)
All branches mathematically merged into single 3x3 convolutions:
```
Input (256x256x3)
  → 3x3 conv+ReLU x5 (ch=16)
  → 3x3 conv (ch=48)
  → PixelShuffle(4) → 1024x1024x3
  + bilinear upsample of input
```

Only operations at inference: 3x3 conv, ReLU, pixel shuffle, bilinear upsample, add. All trivially SIMD-able.

### Why This Architecture
- Reparameterization means training quality without inference cost
- 73K params = ~300KB model file (float32), ~75KB quantized INT8
- Only 3x3 convolutions at inference — perfect for SIMD (AVX2/NEON)
- Global residual learning (predict the delta from bilinear upsample) — easier to learn
- Pixel shuffle for upscaling — more efficient than transposed convolution

## Data Preparation

### From WSI Files
```
For each WSI (.svs, .ndpi, .tiff):
  1. Extract tiles at L0 resolution (e.g., 1024x1024 at 40x)
  2. Downsample each L0 tile 4x → 256x256 (using lanczos3, matching ORIGAMI)
  3. Encode the 256x256 tile as JXL at base quality (e.g., q95)
  4. Decode JXL back → this is the model input
  5. Original L0 tile is the target

  Save pairs as:
    tiles/train/0001_input.png   (256x256)
    tiles/train/0001_target.png  (1024x1024)
```

### From Pre-Existing ORIGAMI Runs
If ORIGAMI encode with `--debug-images` has been run:
```
  Input:  compress/L2_x_y.jxl (decoded to 256x256 PNG)
  Target: compress/L0_orig_x_y.png (1024x1024)
```

### Data Scale
- **Minimum viable:** 1,000 tile pairs (smoke test / proof of concept)
- **Good:** 10,000-50,000 tile pairs from 10-50 WSIs
- **Production:** 100,000+ tile pairs from 100+ WSIs, mixed H&E and IHC
- Each 40x WSI at 100K x 100K pixels yields ~10,000 non-overlapping 1024x1024 tiles

### Augmentation
- Random horizontal/vertical flip
- Random 90/180/270 rotation
- Random crop (train on 256→1024 patches from larger tiles if available)
- Color jitter (small, to handle stain variation)
- No: geometric distortion, heavy color changes (would break the compression task)

## Training Configuration

### Loss Function
```python
loss = L1(output, target) + 0.01 * perceptual_loss(output, target)
```

- **L1 (primary):** Optimizes PSNR, no hallucination risk
- **Perceptual (VGG features, small weight):** Improves texture quality without hallucinating structures
- **NO GAN loss:** Cannot risk hallucinating cellular structures in pathology

### Hyperparameters
```
Optimizer:      Adam, lr=2e-4, betas=(0.9, 0.999)
LR schedule:    CosineAnnealing to 1e-6 over total steps
Batch size:     16-32 (per GPU)
Patch size:     Train on 64→256 random crops (faster), validate on full 256→1024
Epochs:         200-500 (depending on dataset size)
```

### Hardware
- **RunPod:** Single A40/A100 ($0.50-1.00/hr), 1-4 hours for 10K tiles
- **Mayo H200 cluster:** Multi-GPU, scale to larger datasets
- **Memory:** ~8GB VRAM for batch=16, 64→256 crops

### Validation
- Hold out 10% of tiles
- Metrics: PSNR, SSIM, Delta E, LPIPS, Butteraugli, SSIMULACRA2
- Compare against: lanczos3 upsample (baseline), ORIGAMI b95/l0q75 (current best at ~100KB)

## Training Script Structure

```
wsi_sr/
  train.py              # Main training loop
  model.py              # SESR-M5 with reparameterization
  dataset.py            # Tile pair dataset + augmentation
  export.py             # Collapse + ONNX export + INT8 quantization
  evaluate.py           # Metrics on held-out tiles
  prepare_tiles.py      # Extract tile pairs from WSIs or ORIGAMI runs
  requirements.txt      # torch, torchvision, lpips, scikit-image, pillow
```

## Export Pipeline

```
1. Train model (multi-branch mode)
2. Reparameterize: collapse all branches → plain 3x3 convs
3. Validate: confirm collapsed model matches training model output (should be bitwise identical)
4. Export to ONNX (float32)
5. Quantize to INT8 (optional, for fastest CPU inference)
6. Benchmark inference time on CPU
```

## Integration Into ORIGAMI

### Decode Path (Rust)
```
Current:
  read L2 JPEG/JXL (256x256)
  → lanczos3 upsample 4x (1024x1024)
  → decode residual JPEG/JXL (1024x1024 grayscale)
  → apply residual to Y channel
  → slice into 16 L0 tiles (256x256 each)
  → encode as output JPEG

Proposed:
  read L2 JPEG/JXL (256x256)
  → run SR model (256x256 → 1024x1024 RGB)
  → (optional: apply small residual if hybrid mode)
  → slice into 16 L0 tiles (256x256 each)
  → encode as output JPEG
```

### Rust Inference Options
1. **ort crate** — ONNX Runtime, fastest, links C++ lib
2. **rten crate** — Pure Rust ONNX, no C++ deps, good for simple models
3. **Hand-implemented** — For a 7-layer 3x3 conv network, direct SIMD implementation is feasible and would be fastest. Same approach as `core/sharpen.rs`.

## Success Criteria

At matched file size (~33KB per family = just the L2 base):
- Delta E ≤ 1.5 (current ORIGAMI b95/l0q75 at 104KB: 1.66)
- LPIPS ≤ 0.05 (current: 0.055)
- Butteraugli ≤ 2.5 (current: 2.42)
- PSNR ≥ 37 dB (current: 38.4)

If these are met, the model eliminates 70KB of residual data per family while maintaining quality — hitting the 50-100KB target with room to spare.

If not fully met, hybrid mode (model + small residual) should close the gap at 48-68KB total.

## Timeline

1. **Day 1:** Write training script, test on 2 available test images (smoke test)
2. **Day 2-3:** Prepare 1,000-10,000 tile pairs from available WSIs
3. **Day 3-4:** Train on RunPod (A40, ~2-4 hours)
4. **Day 4-5:** Evaluate, compare metrics, iterate on architecture if needed
5. **Week 2:** If results are promising, scale training data and integrate into Rust
