# Learned Residual Codec — Training Plan

## STATUS: ABORTED — Compressibility analysis shows insufficient headroom

Empirical analysis (March 2026) found only **0.29 bpp of spatial redundancy** in
our residuals — the ceiling for any learned codec's improvement over JPEG. The
30-50% compression target is unreachable. SRA training confirmed this: 15-150K
param models achieve 13 dB worse PSNR than JPEG at matched bitrate.

See `LEARNED_RESIDUAL_CODEC_RESEARCH.md` sections 12-14 for full analysis.

**Path forward:** Improve the predictor (SR model / refiner models) to shrink
residuals at the source. JPEG is a reasonable codec for what remains.

---

## Original Goal (not achievable)

Replace JPEG-encoded luma residuals in `.pack` files with a learned codec that achieves **30-50% smaller residuals at matched quality**, with **<10ms CPU decode** and **<100ms GPU encode** per 1024x1024 residual.

## Constraint Recap

| Requirement | Target |
|-------------|--------|
| Decode device | CPU only (no GPU at serve time) |
| Decode latency | <10ms per 1024x1024 grayscale residual |
| Encode device | GPU (offline, but must be fast — WSI have thousands of families) |
| Encode latency | <100ms per residual (Cool-Chic's 30-120s is too expensive) |
| Model size | <1MB ONNX |
| Compression target | 30-50% smaller than JPEG at same PSNR |

## Why JPEG Is Bad for Residuals

JPEG's DCT + Huffman was designed for natural images with smooth low-frequency content. Our residuals are:
- **Sparse**: Most pixels near 128 (zero-centered), occasional large deviations
- **Flat spectrum**: Energy spread across frequencies (prediction removed structure)
- **Laplacian distribution**: Sharp peak at zero, heavy tails
- **No spatial priors**: No edges/textures for DCT to exploit

Result: JPEG wastes bits on the transform overhead and can't efficiently code the sparsity.

## Architecture: Sparse Residual Autoencoder (SRA)

A tiny convolutional autoencoder with a learned factorized entropy model. The encoder runs on GPU at encode time; the decoder runs on CPU at serve time via ONNX Runtime.

```
ENCODER (GPU, offline, <100ms):            DECODER (CPU, fast, <10ms):
  residual (1024x1024x1)                     bitstream
  → 3x3 conv, C ch, stride 2 (512x512)      → entropy decode (ANS)
  → 3x3 conv, C ch, stride 2 (256x256)      → dequantize
  → 3x3 conv, C ch, stride 2 (128x128)      → latent (64x64 × C_latent)
  → 3x3 conv, C_latent, stride 2 (64x64)    → 3x3 deconv, C ch (128x128)
  → quantize                                 → 3x3 deconv, C ch (256x256)
  → entropy encode (ANS)                     → 3x3 deconv, C ch (512x512)
  → bitstream                                → 3x3 deconv, 1 ch (1024x1024)
```

**Key design choices:**
- **Factorized entropy model** (Ballé 2017): Per-channel learned CDF, no autoregressive context (too slow). ~200 bytes side info.
- **Quantization-aware training**: Uniform noise during training, round at inference.
- **No hyperprior**: Keeps decoder tiny. Hyperprior adds a second decoder network.
- **GeLU activations**: Better gradient flow than ReLU for sparse signals.
- **Residual connections** in deeper variants: Skip connections from encoder to decoder at matching spatial scales (like U-Net). Cheap and helps preserve fine detail.

## Training Data

### Source
Existing TCGA tiles from GCS: `gs://wsi-1-480715-tcga-tiles/`
- **Stage 1**: 5K tiles, ~3GB (fast iteration)
- **Stage 2**: 1.4M tiles, ~246GB (final training, subsample to 50K)

### Residual Generation Pipeline

```bash
python learned_codec/generate_residual_dataset.py \
    --tiles-dir /workspace/tcga_tiles/stage1/ \
    --output /workspace/residual_dataset/ \
    --upsample-filter lanczos3 \
    --baseq 95
```

For each 1024x1024 tile:
1. Downsample 4x → 256x256 (L2 simulation)
2. JPEG encode L2 at Q95 → decode (simulate real L2 artifacts)
3. Upsample L2 Y channel 4x with lanczos3 → prediction
4. `residual = original_Y - prediction_Y + 128` (centered at 128)
5. Save as 1024x1024 grayscale PNG + prediction PNG

This exactly replicates the residuals produced by `origami encode`.

## Model Variants (4 Parallel on 4090)

24GB VRAM on 4090. Each model uses ~2-4GB at batch_size=16. Run all 4 simultaneously.

| Variant | Conv Channels | Latent Channels | Decoder Params | Skip Connections | Notes |
|---------|--------------|----------------|---------------|-----------------|-------|
| **SRA-Tiny** | 16 | 16 | ~15K | No | Fastest decode, baseline |
| **SRA-Small** | 32 | 32 | ~50K | No | Sweet spot candidate |
| **SRA-Medium** | 64 | 32 | ~150K | No | More capacity |
| **SRA-UNet** | 32 | 32 | ~80K | Yes (encoder→decoder) | U-Net style, preserves detail |

## Training Configuration

```python
# Per variant (adjust channels/layers)
config = {
    "batch_size": 16,           # 1024x1024 grayscale crops
    "crop_size": 256,           # Train on 256x256 random crops (faster, regularizing)
    "test_size": 1024,          # Evaluate on full 1024x1024
    "epochs": 100,              # ~5K images × 100 epochs ÷ 16 batch = 31K steps
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "lr_schedule": "cosine",    # Anneal to 1e-6
    "weight_decay": 0,

    # Rate-distortion tradeoff
    "lambda": 1e-3,             # Phase 1: single lambda per variant
    # Phase 2 sweep: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

    # Loss = MSE(residual, reconstructed) + lambda * rate(latent)
    "distortion_loss": "MSE",
    "rate_loss": "factorized_entropy",

    # Augmentation
    "random_crop": True,
    "random_flip": True,
    "random_rotate90": True,
    # No color augmentation (single-channel grayscale)
}
```

### Lambda Explained

Lambda controls the rate-distortion tradeoff:
- **Low lambda (1e-4)**: Prioritize quality → larger bitstream, higher PSNR
- **High lambda (5e-3)**: Prioritize size → smaller bitstream, lower PSNR
- Each lambda produces a different operating point on the RD curve
- We need the operating points that match JPEG Q20-Q80's quality range

## Parallel Training Strategy

### Phase 1: Architecture Search (2 hrs, 4 parallel)

```bash
# Launch all 4 on single 4090 (each uses ~3GB VRAM)
# Use CUDA_VISIBLE_DEVICES or torch.cuda.set_device with memory fractions

python learned_codec/train.py --variant sra-tiny   --lambda 1e-3 --epochs 100 --data /workspace/residual_dataset/stage1 --run-id sra_tiny_1e3   &
python learned_codec/train.py --variant sra-small  --lambda 1e-3 --epochs 100 --data /workspace/residual_dataset/stage1 --run-id sra_small_1e3  &
python learned_codec/train.py --variant sra-medium --lambda 1e-3 --epochs 100 --data /workspace/residual_dataset/stage1 --run-id sra_medium_1e3 &
python learned_codec/train.py --variant sra-unet   --lambda 1e-3 --epochs 100 --data /workspace/residual_dataset/stage1 --run-id sra_unet_1e3   &
wait
```

**Evaluate Phase 1:**
- Rate-distortion on held-out residuals (bits-per-pixel vs PSNR)
- CPU decode latency (ONNX Runtime, 1024x1024)
- Pick best 2 variants

### Phase 2: Lambda Sweep (2 hrs, 5 parallel per variant)

For the 2 winning variants, sweep 5 lambda values:

```bash
for lam in 1e-4 5e-4 1e-3 2e-3 5e-3; do
    python learned_codec/train.py --variant $BEST1 --lambda $lam --epochs 100 --data /workspace/residual_dataset/stage1 --run-id ${BEST1}_${lam} &
done
wait

# Repeat for BEST2 (or interleave if VRAM allows)
```

**Evaluate Phase 2:**
- Full RD curves for both variants across all lambdas
- Overlay JPEG RD curve (encode residuals at Q10-Q95 for comparison)
- Pick best variant + identify 2-3 optimal lambda operating points

### Phase 3: Final Training (4 hrs)

Train winning variant on 50K residuals (subsample from Stage 2):

```bash
# Download 50K tiles from Stage 2
gsutil -m cp -r gs://wsi-1-480715-tcga-tiles/stage2/ /workspace/tcga_tiles/stage2/
# Subsample + generate residuals
python learned_codec/generate_residual_dataset.py \
    --tiles-dir /workspace/tcga_tiles/stage2/ \
    --output /workspace/residual_dataset/stage2_50k/ \
    --max-tiles 50000 --upsample-filter lanczos3 --baseq 95

# Train at best 2-3 lambda operating points
python learned_codec/train.py --variant $BEST --lambda $LAM1 --epochs 200 --data /workspace/residual_dataset/stage2_50k/ --run-id final_lam1 &
python learned_codec/train.py --variant $BEST --lambda $LAM2 --epochs 200 --data /workspace/residual_dataset/stage2_50k/ --run-id final_lam2 &
python learned_codec/train.py --variant $BEST --lambda $LAM3 --epochs 200 --data /workspace/residual_dataset/stage2_50k/ --run-id final_lam3 &
wait
```

### Phase 4: Export + Integration Test (30 min)

```bash
# Export to ONNX (reuse wsi_sr/export.py patterns)
python learned_codec/export.py --checkpoint checkpoints/final_lam2/best.pt --output models/residual_codec.onnx

# Optional INT8 quantization
python learned_codec/export.py --checkpoint checkpoints/final_lam2/best.pt --output models/residual_codec_int8.onnx --quantize

# Benchmark decode speed
python learned_codec/benchmark_decode.py --model models/residual_codec.onnx --input /workspace/residual_dataset/stage1/
```

## Timeline Summary

| Phase | Duration | GPU Util | What |
|-------|----------|----------|------|
| Data prep | 30 min | Low | Generate 5K residuals from Stage 1 tiles |
| Phase 1 | 2 hrs | 4 parallel | 4 architecture variants × lambda=1e-3 |
| Eval 1 | 15 min | — | Pick best 2 |
| Phase 2 | 2 hrs | 5 parallel | Best 2 × 5 lambdas |
| Eval 2 | 30 min | — | RD curves, pick winner |
| Phase 3 | 4 hrs | 3 parallel | Winner × 3 lambdas on 50K residuals |
| Phase 4 | 30 min | — | Export ONNX, benchmark, integration test |
| **Total** | **~10 hrs** | **~22 GPU-hrs** | Single 4090 session |

## Evaluation Criteria

For each trained model, measure:

1. **Rate-Distortion curve**: Bits-per-pixel vs PSNR on held-out residuals
   - Overlay JPEG baseline (Q10-Q95) on same plot
   - Target: 30-50% fewer bits at same PSNR
2. **Decode speed**: ONNX Runtime CPU time for 1024x1024 grayscale
   - Target: <10ms (Apple Silicon and x86-64)
3. **Encode speed**: GPU inference time for 1024x1024
   - Target: <100ms
4. **End-to-end tile quality**: Decode residual → apply to prediction → measure vs ground truth
   - PSNR, SSIM, LPIPS, Delta E (same metrics as existing evals)
   - Target: within 0.1 dB of JPEG residual path at same residual size
5. **Pack size**: Total .pack file size (L2 JPEG + learned-coded residual)
   - Compare vs current pack sizes at Q20, Q40, Q60, Q80

**Success gate (Phase 1 → Phase 2):**
- At least one variant beats JPEG by ≥15% at matched PSNR
- Decode <15ms on CPU

**Success gate (Phase 2 → Phase 3):**
- Best variant beats JPEG by ≥25% across the quality range
- Decode <10ms on CPU

**Kill criteria:**
- If no variant beats JPEG by >10% → abort, JPEG is fine for our residuals
- If decode >20ms → reduce model size or abort

## Implementation Files

```
learned_codec/
├── README.md                        # Setup instructions
├── generate_residual_dataset.py     # TCGA tiles → residual PNGs
├── model.py                         # SRA encoder/decoder/entropy model variants
├── train.py                         # Training loop (reuse wsi_sr/train.py patterns)
├── evaluate.py                      # RD curves, decode speed benchmarks
├── export.py                        # PyTorch → ONNX (reuse wsi_sr/export.py)
├── benchmark_decode.py              # CPU decode speed profiling
├── plot_rd_curves.py                # Generate rate-distortion comparison plots
└── entropy/
    ├── factorized.py                # Factorized entropy model (learned CDFs)
    └── ans.py                       # ANS coder (or use constriction/torchac)
```

## Entropy Coding

The latent tensor from the encoder must be serialized to a bitstream. Two components:

### 1. Entropy Model (during training)
Factorized prior (Ballé 2017): each latent channel has a learned piecewise-linear CDF. During training, rate is estimated as `-log2(P(y_hat))` summed over all latent elements. No autoregressive context — keeps decode fast.

### 2. Bitstream Codec (encode/decode)
Use **`constriction`** library — has both Python (training) and Rust (inference) APIs with guaranteed bitstream compatibility.

| Library | Python API | Rust API | ANS | Range | Notes |
|---------|-----------|---------|-----|-------|-------|
| **constriction** | Yes | Yes | Yes | Yes | Best fit — same bitstream format for train + deploy |
| torchac | Yes | No | No | Yes | Python only, need separate Rust impl |
| CompressAI | Yes | No | Yes | Yes | Heavy dependency, Python only |

```toml
# Rust Cargo.toml addition
constriction = "0.4"
```

```python
# Python training
import constriction
# Use constriction.stream.queue.RangeEncoder for training rate estimation
```

## Rust Integration (Post-Training)

### Pack Format V5

Add new entry type for learned-coded residuals:

```
Pack V5 header:
  magic: "ORIG"
  version: 5
  entry_count: 2
  entries:
    [0] L2 baseline JPEG (unchanged)
    [1] Learned-coded L0 residual:
         - 4 bytes: latent height (u16) + latent width (u16)
         - 4 bytes: latent channels (u16) + lambda_idx (u16)
         - N bytes: ANS-coded latent bitstream
```

### Decode Path

```rust
// In reconstruct.rs:
// 1. Read ANS bitstream from pack entry
// 2. Entropy decode → quantized latent tensor (64x64 × C_latent)
// 3. Dequantize (just cast to f32)
// 4. Run ONNX decoder → 1024x1024 grayscale residual
// 5. Apply residual to prediction (same as current JPEG path)
```

Uses existing `sr_model.rs` ONNX session pool pattern — just a different model.

### Backward Compatibility

- Pack v2/v3/v4 files still work (JPEG residuals)
- Pack v5 auto-detected by version field
- Server can decode both — just checks entry type

## Dependencies

### Python (training)
```
torch >= 2.0
torchvision
onnx
onnxruntime
constriction >= 0.4     # Entropy coding
scikit-image             # Already installed
lpips                    # Already installed
Pillow                   # Already installed
```

### Rust (decode only)
```toml
ort = "2.0.0-rc.12"     # Already in Cargo.toml
constriction = "0.4"     # ANS/range coding (new)
```

### Reusable Existing Code
- `wsi_sr/train.py` — training loop, cosine annealing, cloud status uploads
- `wsi_sr/dataset.py` — tile loading, augmentation patterns
- `wsi_sr/export.py` — PyTorch → ONNX + INT8 quantization
- `wsi_sr/evaluate.py` — PSNR, SSIM, VIF, Delta E, LPIPS, Butteraugli
- `server/src/core/sr_model.rs` — ONNX Runtime session pool
- `server/src/core/pack.rs` — pack format reader/writer
- `evals/scripts/compute_metrics.py` — end-to-end quality evaluation

## Reference: Key Papers

- Ballé et al. 2017 — "End-to-end Optimized Image Compression" (factorized entropy model)
- Ballé et al. 2018 — "Variational Image Compression with a Scale Hyperprior" (hyperprior, we skip this)
- Minnen et al. 2018 — "Joint Autoregressive and Hierarchical Priors" (context model, we skip this)
- The `constriction` library handles the gap between learned CDFs and actual entropy coding efficiently.
