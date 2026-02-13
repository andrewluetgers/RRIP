# ORIGAMI Encode-Time Optimizations

This document describes the optimizations available within ORIGAMI's residual pyramid compression algorithm. All optimizations operate at encode time only — the decoder and file format are unchanged.

## Background: The Residual Pyramid

ORIGAMI compresses a tile family (1 L2 + 4 L1 + 16 L0 tiles, each 256x256) by storing:

- **L2 tile**: Stored directly as a baseline JPEG (q=95)
- **L1 residuals**: L2 is bilinearly upsampled 2x to predict L1; the luma prediction error is JPEG-compressed
- **L0 residuals**: Reconstructed L1 is upsampled 2x to predict L0; the luma prediction error is JPEG-compressed

Chroma (Cb/Cr) is predicted from the parent and reused — only luma residuals are stored.

The two degrees of freedom at encode time are:

1. **What quality to use** for each level's residual JPEG encoding
2. **What L2 pixels to store** (they don't have to be the "natural" downsampled image)

---

## Optimization 1: Split-Quality Encoding

### The Idea

In a flat-quality configuration (e.g., all residuals at q=50), L1 and L0 residuals use the same JPEG quality. But L1 residuals serve double duty: they reconstruct L1 tiles *and* those tiles become the prediction source for L0. Higher-fidelity L1 reconstruction means better L0 predictions, which means smaller L0 residuals.

Split-quality encoding gives L1 residuals a higher quality than L0 (e.g., L1=70 L0=50), trading slightly larger L1 residuals for smaller L0 residuals and better overall quality.

### Mechanism

The trade-off is direct and measurable:

| Config | L1 Res Cost | L0 Res Savings | Net Size | PSNR Gain |
|--------|-------------|----------------|----------|-----------|
| q=30 flat vs L1=50 L0=30 | +6.7 KB | -5.2 KB | +1.5 KB (+1.8%) | **+0.28 dB** |
| q=40 flat vs L1=60 L0=40 | +7.2 KB | -5.2 KB | +2.0 KB (+2.0%) | **+0.24 dB** |
| q=50 flat vs L1=70 L0=50 | +9.6 KB | -6.5 KB | +3.1 KB (+2.7%) | **+0.28 dB** |
| q=60 flat vs L1=80 L0=60 | +15.4 KB | -9.4 KB | +6.0 KB (+4.6%) | **+0.39 dB** |

The extra bytes spent on L1 are partially offset by L0 savings, with a net cost of 1.8-4.6% in file size. The PSNR gain is +0.24 to +0.39 dB, concentrated in L1 tile quality (which improves by +1.0 to +1.8 dB) while L0 quality remains nearly unchanged.

### How Split Quality Allocates Bits

Examining the byte breakdown reveals why this works:

| Config | L1 Residuals | L0 Residuals | Total | Ratio |
|--------|-------------|-------------|-------|-------|
| flat q=30 | 7.5 KB | 38.7 KB | 87,836 B | 6.84x |
| L1=50 L0=30 | 14.3 KB | 33.5 KB | 89,404 B | 6.72x |
| L1=60 L0=30 | 18.0 KB | 31.1 KB | 90,737 B | 6.62x |
| | | | | |
| flat q=50 | 14.3 KB | 60.9 KB | 117,408 B | 5.12x |
| L1=70 L0=50 | 23.9 KB | 54.3 KB | 120,587 B | 4.98x |
| L1=80 L0=50 | 33.4 KB | 49.1 KB | 124,999 B | 4.81x |

L0 dominates the byte budget (16 tiles vs 4 L1 tiles), so even a small percentage reduction in L0 residual size partially offsets the L1 increase. The quality gain comes from L1 tiles being reconstructed more faithfully, producing predictions that capture more of L0's structure.

### Optimal Split

A +20 quality gap (L1=q+20, L0=q) provides a good balance across all operating points. Larger gaps (e.g., +30) continue to improve L1 quality but with diminishing returns on L0 and increasing size costs.

---

## Optimization 2: L2 Prediction Optimization (OptL2)

### The Insight

The standard pipeline produces L2 by Lanczos-downsampling the source image. This creates a "natural" L2 that looks correct at its resolution, but it's not optimized for what L2 is actually *used for*: predicting L1 via bilinear upsampling.

**L2 doesn't have to be the "true" downsampled image.** L2 should be whatever 256x256 image, when bilinearly upsampled, best predicts L1. The decoder doesn't know or care — it just reads L2, upsamples, and adds residuals.

### Algorithm

The standard ORIGAMI pipeline downsamples the source image via Lanczos to produce L2, then bilinearly upsamples L2 to predict L1. The prediction error (residual) is JPEG-compressed and stored. The key observation is that the decoder never sees the "true" downsampled image — it only sees whatever L2 tile we give it. So L2 should be whatever 256x256 image produces the best L1 predictions when bilinearly upsampled, not necessarily the natural downsampling.

The core insight is fully general: anywhere an image is downsampled for storage and later bilinearly upsampled for display or prediction, the downsampled pixels can be optimized to minimize reconstruction error. Bilinear upsampling is a linear operator, so the gradient of reconstruction error with respect to the source pixels is exact and cheap to compute — it's simply the error field downsampled back to source resolution (the adjoint/transpose of the upsample).

The optimization works on all three RGB channels simultaneously. Although ORIGAMI only stores luma residuals, the predicted chroma (Cb/Cr) from the upsampled parent is used directly in reconstruction without correction. Optimizing all channels improves both the luma prediction (reducing residual energy) and the chroma prediction (improving color accuracy for free, since chroma costs no additional residual bytes).

```
OPTIMIZE_FOR_UPSAMPLE(source, target, max_delta=15, lr=0.3, N=100):
    # source: downsampled image, shape (H, W, 3) — e.g. 256x256 L2 tile
    # target: high-res ground truth, shape (2H, 2W, 3) — e.g. 512x512 L1 mosaic

    current = copy(source)      # mutable float64 working copy
    original = copy(source)     # frozen reference for delta constraint

    for iteration in 1..N:

        # FORWARD: bilinear upsample current source to target resolution
        predicted = bilinear_upsample(current)    # (H,W,3) → (2H,2W,3)

        # LOSS: sum of squared error across all pixels and channels
        residual = target - predicted             # (2H,2W,3) error field
        energy = sum(residual^2)                  # scalar loss

        # GRADIENT: adjoint of bilinear upsample = bilinear downsample
        # Each source pixel receives the weighted average error from
        # the target pixels in its bilinear interpolation footprint
        gradient = bilinear_downsample(residual)  # (2H,2W,3) → (H,W,3)

        # UPDATE: step toward lower prediction error
        current += lr * gradient

        # CONSTRAIN: keep close to original (perceptual similarity)
        current = clip(current, original - max_delta, original + max_delta)
        current = clip(current, 0, 255)

    return current
```

**Why this works:** Each source pixel contributes (via bilinear interpolation) to a 2x2 neighborhood of target pixels. The gradient tells each pixel which direction to shift to reduce the total prediction error of its neighborhood. The `max_delta` constraint ensures the result remains perceptually similar to the original — at ±15, the mean pixel change is ~5 levels and PSNR between original and optimized is ~31 dB.

**Why the gradient is a downsample:** Bilinear upsampling is a matrix multiply `predicted = U * source` where U is a sparse interpolation matrix. The gradient of `||target - U*source||^2` with respect to source is `U^T * (target - U*source)`. The transpose of bilinear upsampling is bilinear downsampling — each source pixel receives a weighted average of the errors from the target pixels it influences.

**Why all three channels:** Optimizing luma alone leaves chroma prediction untouched, and at high residual quality (q=80+) where luma residuals already capture everything, luma-only optimization provides zero benefit. But chroma prediction error is independent of residual quality (it's never corrected) — so the chroma component of the optimization provides a consistent +0.2-0.3 dB PSNR improvement and -0.1 Delta E color accuracy improvement at ALL quality levels, including the high end where luma-only gains vanish.

**Performance:** ~6.7ms per iteration for 256→512 optimization on Apple M-series (ARM NEON). NumPy array operations use SIMD via the Accelerate/OpenBLAS backend; PIL's C-level bilinear resize is the fastest available upsample/downsample. 100 iterations is sufficient for convergence (the ±max_delta constraint is the binding limit, not iteration count). Total cost: ~0.7s per tile family.

At encode time, the optimized L2 replaces the natural L2 in the pyramid. Residuals are then computed against the optimized L2's predictions, which are smaller (lower energy), compress to fewer bytes, and suffer less quantization error.

At decode time: **completely unchanged**. The decoder reads L2, upsamples, adds residuals — it doesn't know or care that L2 was optimized. Zero added cost.

### General Applicability

This optimization applies anywhere an image is downsampled and later upsampled:

- **Residual pyramid prediction** (ORIGAMI's L2→L1 prediction, the primary use case)
- **Thumbnail/preview generation** — optimize thumbnails so they upscale with less blur (+2.3 dB in testing)
- **Chroma subsampling** — the same math applies to 4:2:0 chroma: optimize the subsampled Cb/Cr planes so bilinear upsample best reconstructs the original. Testing showed 44% chroma energy reduction. Standard JPEG encoders use box-filter downsampling internally and don't expose this control, but a custom encoder could apply it directly.

The optimization is NOT equivalent to "just storing more data." At matched file sizes, raising JPEG quality from 95 to 96 reduces L1 prediction energy by 1.1%, while the optimization achieves 37% — a 34x difference (see "L2 File Size" section below).

### Results

| Residual Quality | Size Savings | PSNR Gain |
|-----------------|-------------|-----------|
| q=30 | **-5.2%** | **+0.40 dB** |
| q=40 | **-5.6%** | **+0.23 dB** |
| q=50 | **-5.4%** | **+0.15 dB** |
| q=60 | **-5.0%** | **+0.09 dB** |
| q=70 | **-4.4%** | **+0.04 dB** |
| q=80 | **-3.7%** | **+0.00 dB** |

The optimization reduces L1 prediction residual energy by ~38%. This simultaneously **reduces file size** (smaller residuals compress to fewer bytes) and **improves quality** (less quantization error in the compressed residuals). Gains are largest at high compression where JPEG quantization is most aggressive.

### L2 File Size and the "Just More Data?" Question

The optimized L2 tile is larger than the natural one — 45.7 KB vs 39.5 KB at q=95, a fixed +6.3 KB (+15.8%) overhead. This raises a natural question: are the downstream gains simply the result of storing more information in L2, and could we achieve the same thing by raising L2 JPEG quality?

The answer is no. The optimization is fundamentally different from higher-fidelity storage.

| Configuration | L2 Size | L1 Residual Energy Reduction |
|---|---|---|
| Natural L2 at q=95 (baseline) | 42.8 KB | — |
| Natural L2 at q=96 | 47.7 KB | **1.1%** |
| Natural L2 at q=97 | 53.1 KB | **1.7%** |
| Natural L2 at q=98 | 59.6 KB | **2.1%** |
| Natural L2 at q=99 | 70.4 KB | **2.4%** |
| **OptL2 at q=95** | **49.0 KB** | **37.1%** |

At roughly matched file size (~48 KB), raising quality from 95 to 96 reduces L1 residual energy by 1.1%. OptL2 at the same size reduces it by 37.1% — a **34x difference**. Even at q=99 (70 KB, 43% larger than OptL2), the natural image only achieves 2.4% energy reduction.

**Why raising quality doesn't help:** Higher JPEG quality preserves the natural Lanczos-downsampled image more faithfully, but the natural downsampling itself is not optimized for bilinear upsampling prediction. The Lanczos kernel preserves perceptual fidelity at L2's resolution — it makes L2 *look correct* — but what matters for residual compression is how L2 looks *after bilinear upsampling to L1 resolution*. These are different objectives, and the optimization directly targets the one that matters.

**Where the extra bytes come from:** The optimized L2 has slightly higher spatial frequency content than the natural downsampling (because it encodes prediction-relevant detail that Lanczos would smooth away), which makes it compress ~16% larger at the same JPEG quality. This is a side effect, not the mechanism — the extra 6.3 KB in L2 is vastly outweighed by the 10-13 KB saved in residuals.

| Quality | L2 Increase | Residual Savings | Net |
|---------|------------|-----------------|-----|
| q=30 | +6.3 KB | -10.9 KB | **-4.6 KB (-5.2%)** |
| q=50 | +6.3 KB | -12.6 KB | **-6.3 KB (-5.4%)** |
| q=70 | +6.3 KB | -13.3 KB | **-6.9 KB (-4.4%)** |

At every operating point, the residual savings exceed the L2 cost by 2-3x.

### Why All Three Channels Matter

The luma-only version of this optimization hits zero benefit at q=80+ because luma residuals already capture everything. But the three-channel version maintains consistent gains across the entire quality range because chroma prediction is never corrected by residuals:

| Quality | Luma-only PSNR | All-channel PSNR | All-channel Delta E |
|---------|---------------|-----------------|-------------------|
| q=10 | +1.13 dB | **+1.31 dB** | -0.34 |
| q=30 | +0.73 dB | **+0.92 dB** | -0.24 |
| q=50 | +0.39 dB | **+0.60 dB** | -0.18 |
| q=70 | +0.11 dB | **+0.34 dB** | -0.13 |
| q=80 | +0.00 dB | **+0.27 dB** | -0.11 |
| q=90 | -0.03 dB | **+0.30 dB** | -0.11 |

The chroma improvement provides a steady floor: ~0.2-0.3 dB PSNR and ~0.1-0.3 Delta E at ALL quality levels, even where luma-only gains vanish. L2 file size is essentially the same for both approaches (~49-50 KB vs 43 KB natural), and residual byte counts are unchanged since only luma residuals are stored.

### L2 Chroma Subsampling: 4:2:0 vs 4:4:4

The L2 tile is currently encoded as JPEG with 4:2:0 chroma subsampling, which downsamples L2's chroma from 256x256 to 128x128 inside the JPEG encoder and upsamples it back to 256x256 on decode. This means the chroma we carefully optimize passes through a destructive 256→128→256 round-trip *before* the 256→512 bilinear upsample that produces L1 predictions.

The chroma path through the pipeline has two subsampling stages:

```
Optimized L2 chroma (256x256)
        ↓
    JPEG 4:2:0 encode → 128x128     ← destroys half the chroma resolution
        ↓
    JPEG decode → 256x256            ← upsampled, but detail is lost
        ↓
    Bilinear upsample → 512x512     ← L1 chroma prediction
```

Testing 4:4:4 encoding (no chroma subsampling) reveals this is the single largest quality bottleneck in the pipeline:

| Config | L2 Size | L1 Residuals | Total | L1 PSNR | vs Baseline |
|--------|---------|-------------|-------|---------|-------------|
| | | | | | |
| **q=30 residuals** | | | | | |
| Natural, 4:2:0 (current) | 42.8 KB | 11.4 KB | 54.3 KB | 31.47 dB | — |
| Optimized, 4:2:0 | 50.0 KB | 7.0 KB | 57.0 KB | 32.37 dB | +0.90 dB |
| Natural, 4:4:4 | 56.0 KB | 11.4 KB | 67.4 KB | 32.61 dB | +1.14 dB |
| **Optimized, 4:4:4** | **66.3 KB** | **7.0 KB** | **73.3 KB** | **33.73 dB** | **+2.26 dB** |
| | | | | | |
| **q=50 residuals** | | | | | |
| Natural, 4:2:0 (current) | 42.8 KB | 18.9 KB | 61.8 KB | 32.20 dB | — |
| Optimized, 4:2:0 | 50.0 KB | 11.4 KB | 61.4 KB | 32.78 dB | +0.58 dB |
| Natural, 4:4:4 | 56.0 KB | 19.0 KB | 74.9 KB | 33.61 dB | +1.41 dB |
| **Optimized, 4:4:4** | **66.3 KB** | **11.4 KB** | **77.7 KB** | **34.32 dB** | **+2.12 dB** |
| | | | | | |
| **q=70 residuals** | | | | | |
| Natural, 4:2:0 (current) | 42.8 KB | 29.1 KB | 72.0 KB | 32.97 dB | — |
| Optimized, 4:2:0 | 50.0 KB | 20.1 KB | 70.1 KB | 33.29 dB | +0.32 dB |
| Natural, 4:4:4 | 56.0 KB | 29.1 KB | 85.1 KB | 34.73 dB | +1.76 dB |
| **Optimized, 4:4:4** | **66.3 KB** | **20.1 KB** | **86.4 KB** | **35.08 dB** | **+2.11 dB** |

Key observations:

1. **Switching to 4:4:4 alone (no optimization) gives +1.1 to +1.8 dB** — larger than the entire OptL2 gain under 4:2:0. The 4:2:0 chroma loss cascades through both L1 and L0 reconstruction levels.

2. **Optimization + 4:4:4 combined gives +2.1 to +2.3 dB** over the current baseline. The gains are additive: 4:4:4 preserves chroma fidelity, and the optimization reshapes both luma and chroma for better prediction.

3. **The cost is ~13-16 KB more in L2** (4:4:4 stores full-resolution chroma). At low residual quality this is significant relative to total size; at higher quality the L2 overhead becomes a smaller fraction.

4. **Residual byte counts are unaffected** by L2 subsampling mode — only luma residuals are stored, and luma is not subsampled in either mode.

5. **The optimization is not fighting the encoder under 4:4:4.** With 4:2:0, the encoder's internal chroma subsampling partially destroys the optimized chroma before it can be used for prediction. With 4:4:4, the encoder preserves the optimized pixels exactly (modulo DCT quantization at q=95, which is very light).

**Recommendation:** Encoding L2 at 4:4:4 is the single most impactful change available. It requires no algorithmic changes to the decoder — the JPEG decoder handles both subsampling modes transparently. The combination of 4:4:4 + all-channel optimization represents a potential +2 dB improvement over the current pipeline.

### Convergence

| Iterations | Energy Reduction |
|-----------|-----------------|
| 10 | 32.8% |
| 50 | 37.7% |
| 100 | 37.8% |
| 500 | 37.8% |
| 3000 | 37.8% |

The binding constraint is the ±15 pixel delta, not iteration count. The optimization converges by ~50 iterations. 100 iterations is more than sufficient. Encode cost is ~0.5 seconds per L2 family in Python.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_delta` | 15 | Maximum luma pixel deviation from original (±) |
| `n_iterations` | 500 | Gradient descent iterations (100 is sufficient) |
| `lr` | 0.3 | Learning rate |

At ±15, the optimized L2 is perceptually similar to the original (PSNR ~31.5 dB between them). The mean pixel change is ~5 levels out of 255.

---

## Combined: OptL2 + Split Quality

The two optimizations are independent and additive. They attack different aspects of the compression:

- **OptL2** reduces the residual energy that needs to be encoded (better prediction → smaller residuals)
- **Split quality** allocates bits more efficiently across levels (more bits where they have more leverage)

### Four-Way Comparison

| Config | Total Size | Avg PSNR | vs Flat Baseline |
|--------|-----------|----------|-----------------|
| | | | |
| **q=30 operating point** | | | |
| ORIGAMI flat q=30 | 87,836 B | 34.38 dB | — |
| ORIGAMI split L1=50 L0=30 | 89,404 B | 34.66 dB | +0.28 dB / +1.8% |
| OptL2 flat q=30 | 83,277 B | 34.78 dB | +0.40 dB / -5.2% |
| **OptL2 split L1=50 L0=30** | **84,419 B** | **34.92 dB** | **+0.54 dB / -3.9%** |
| | | | |
| **q=40 operating point** | | | |
| ORIGAMI flat q=40 | 102,569 B | 35.12 dB | — |
| ORIGAMI split L1=60 L0=40 | 104,589 B | 35.35 dB | +0.24 dB / +2.0% |
| OptL2 flat q=40 | 96,845 B | 35.35 dB | +0.23 dB / -5.6% |
| **OptL2 split L1=60 L0=40** | **98,307 B** | **35.50 dB** | **+0.38 dB / -4.2%** |
| | | | |
| **q=50 operating point** | | | |
| ORIGAMI flat q=50 | 117,408 B | 35.72 dB | — |
| ORIGAMI split L1=70 L0=50 | 120,587 B | 35.99 dB | +0.28 dB / +2.7% |
| OptL2 flat q=50 | 111,115 B | 35.87 dB | +0.15 dB / -5.4% |
| **OptL2 split L1=70 L0=50** | **113,565 B** | **36.06 dB** | **+0.34 dB / -3.3%** |
| | | | |
| **q=60 operating point** | | | |
| ORIGAMI flat q=60 | 133,762 B | 36.28 dB | — |
| ORIGAMI split L1=80 L0=60 | 139,915 B | 36.67 dB | +0.39 dB / +4.6% |
| OptL2 flat q=60 | 127,107 B | 36.38 dB | +0.09 dB / -5.0% |
| **OptL2 split L1=80 L0=60** | **132,374 B** | **36.68 dB** | **+0.40 dB / -1.0%** |

At every operating point, the combined configuration (OptL2 + split) delivers the best quality, and at lower total size than the flat baseline. At q=30, the combined approach gives **+0.54 dB and saves 3.9%** compared to vanilla flat-quality ORIGAMI.

The gains stack cleanly: split's PSNR contribution (~0.14-0.30 dB on top of OptL2) is roughly the same as without OptL2, and OptL2's size savings (~3-5%) is roughly the same with or without split.

---

## Why L1 Prediction Optimization Doesn't Work

We thoroughly investigated applying the same gradient descent optimization at L1 — optimizing L1 tile pixels to better predict L0 via bilinear upsampling. Despite achieving a similar ~41% energy reduction in the L1→L0 prediction error, every end-to-end configuration showed a **net regression** in both quality and size.

### The Structural Asymmetry

**L2 optimization works because L2 is stored directly.** The optimized L2 tile replaces the original JPEG in the pyramid at q=95. The file size barely changes. The decoder reads exactly the pixels we chose.

**L1 tiles are not stored directly — they're stored as residuals.** L1 residual = L1_target - bilinear(L2). If we change L1_target (optimize it for L0 prediction), the residual changes too. Even when we re-optimize L2 to predict the modified L1, the L1 residuals grow substantially:

| Tile | Standard Residual Std | After L1 Optimization | Ratio |
|------|----------------------|----------------------|-------|
| (0,0) | 5.41 | 9.19 | 1.70x |
| (1,0) | 4.33 | 7.72 | 1.78x |
| (0,1) | 5.60 | 9.53 | 1.70x |
| (1,1) | 5.22 | 9.00 | 1.72x |

L1 residuals nearly double in magnitude. The problem: L1 optimization moves each tile's pixels **independently** to suit its own L0 children. But L2 is a single 256x256 image that must predict all 4 L1 tiles via bilinear upsampling. It can't independently track what each tile wants — the inter-tile divergence creates high-frequency residuals that L2's 4:1 pixel ratio can't capture.

### End-to-End Results

With the correct ordering (optimize L1 for L0, then re-derive L2 from optimized L1 mosaic, then optimize L2 for optimized L1):

| Quality | L1 PSNR Change | L0 PSNR Change | Size Change |
|---------|---------------|---------------|-------------|
| q=30 | -0.89 dB | -0.01 dB | +1.6% |
| q=40 | -1.27 dB | -0.03 dB | +2.0% |
| q=50 | -1.57 dB | -0.05 dB | +2.4% |
| q=60 | -1.98 dB | -0.06 dB | +2.9% |

L0 quality barely changes (the prediction improvement is real but gets consumed by the larger L1 residuals), L1 quality drops significantly, and files get larger.

### The Correct L1 Strategy: Split Quality

Rather than manipulating L1 pixels (which must be stored through a lossy codec), split-quality encoding directly gives L1 more bits. This achieves the same goal — better L1 reconstruction for better L0 prediction — without fighting the residual codec.

---

## Other Approaches Investigated

Several other encode-time optimization strategies were systematically evaluated and found to provide no benefit:

### Residual Pre-Compensation

**Idea:** Scale residuals before JPEG encoding (e.g., 1.2x) to counteract JPEG's systematic attenuation of high-frequency components.

**Finding:** At matched byte counts, pre-compensation provides zero benefit over simply increasing JPEG quality. The PSNR difference ranged from -0.13 to +0.16 dB — statistically noise. JPEG's quality parameter already controls the rate-distortion trade-off optimally.

### Steganographic Embedding (QIM in DCT Domain)

**Idea:** Embed residual correction data inside the L2 tile's DCT coefficients using Quantization Index Modulation, surviving JPEG round-trips.

**Finding:** Achieved perfect bit error rate (BER=0) at q=75 with 1-3 DCT coefficients per 8x8 block. However, maximum reliable capacity is ~8 KB (grayscale) or ~24 KB (RGB), while residuals require 61-91 KB. The capacity gap is 3-10x too large.

### Parametric Correction Functions

**Idea:** Instead of storing full residuals, embed compact parameters for correction functions (bias, affine, quadratic, DCT top-N coefficients, per-block models) that could fit within QIM capacity.

**Finding:** Even the most expressive model that fits in embeddable capacity (DCT top-128, 640 bytes) captures only 1.7% of residual energy. Residual fields have too much spatial entropy for compact parametric models.

### Alternative Residual Codecs

**Idea:** Replace JPEG for residual encoding with deadzone+zlib, wavelets (PyWavelets), vector quantization, sparse thresholding, or edge-guided methods.

**Finding:** Per-tile analysis showed deadzone+zlib competitive at high bitrates, but end-to-end testing showed JPEG winning by 0.8-1.2 dB at every practical operating point. JPEG's DCT decorrelation and perceptual quantization are well-suited to residual statistics.

---

## Summary

Two encode-time optimizations provide measurable gains within ORIGAMI's algorithm:

| Optimization | How It Works | Quality Gain | Size Impact | Decode Cost |
|-------------|-------------|-------------|-------------|-------------|
| **Split Quality** | Higher JPEG quality for L1 residuals than L0 | +0.24 to +0.39 dB | +1.8% to +4.6% | None |
| **OptL2** | Gradient descent on L2 luma to minimize L1 prediction error | +0.09 to +0.40 dB | -3.7% to -5.6% | None |
| **Combined** | Both together | +0.34 to +0.54 dB | -1.0% to -3.9% | None |

Both are decoder-transparent, compatible with all encoder backends, and stack cleanly. The combined configuration is recommended as the default encoding mode.

---

## Prior Work and Related Research

The OptL2 optimization — using gradient descent with the adjoint of bilinear upsampling to optimize low-resolution pixels for better high-resolution prediction — sits at the intersection of several research threads in image processing, compression, and signal processing.

### Processing-Aware Filtering (Wronski, 2021)

The closest published work to ORIGAMI's OptL2. Wronski frames the same core problem: given a fixed bilinear upsampler you cannot change, optimize the downsampling step to minimize reconstruction error. He explores two approaches:

- **Per-image optimization (oracle):** Directly solve for low-resolution pixels that minimize `||target - upsample(source)||^2`. This is the same objective as OptL2. Wronski notes this is "just a linear least squares problem" since bilinear upsampling is a linear operator.
- **Generic filter design:** Optimize a single convolution kernel (content-independent). The result is a simple unsharp mask `[-0.175, 1.35, -0.175]`, which pre-sharpens to compensate for bilinear's blurring tent response.

Wronski uses JAX autodiff for the optimization, which implicitly computes the adjoint (transpose of the upsample operator) — the same gradient ORIGAMI derives explicitly as bilinear downsampling.

OptL2 differs in applying the per-pixel oracle optimization within a residual compression pipeline, where the downstream residuals pass through a lossy JPEG codec. The interaction between pixel optimization, chroma subsampling modes, and split-quality encoding is specific to ORIGAMI.

> Wronski, B. (2021). "Processing aware image filtering: compensating for the upsampling." Blog post. https://bartwronski.com/2021/07/20/processing-aware-image-filtering-compensating-for-the-upsampling/

### A Fresh Look at Generalized Sampling (Nehab & Hoppe, 2014)

A theoretical framework for decomposing discretization and reconstruction into a compactly supported continuous-domain function and a digital filter. They demonstrate that when the reconstruction filter is fixed (e.g., bilinear), the prefilter can be optimized to minimize reconstruction error. This provides the signal-processing-theoretic foundation for OptL2's per-pixel approach: the optimal prefilter for a known reconstruction kernel can be derived from the inverse (or pseudoinverse) of that kernel's frequency response.

> Nehab, D. and Hoppe, H. (2014). "A Fresh Look at Generalized Sampling." Foundations and Trends in Computer Graphics and Vision, Vol. 8, No. 1, pp. 1–84. https://hhoppe.com/proj/filtering/

### Gradient Descent-Based Chroma Subsampling (Chung & Lee, 2019; Chung, Lee & Chien, 2020)

Directly applies gradient descent to optimize chroma pixel values during 4:2:0 subsampling in HEVC — the same mathematical idea as OptL2 applied to chroma channels rather than luma prediction. Key contributions:

- They prove the bilinear-interpolation-based 2x2 block distortion function is **convex**, guaranteeing a global optimum.
- They derive a closed-form initial solution then refine via iterative gradient descent.
- Achieves substantial quality and rate-distortion gains over standard box-filter chroma downsampling.

This directly validates ORIGAMI's observation (in the "General Applicability" section) that the same optimization math applies to 4:2:0 chroma subsampling: optimize the subsampled Cb/Cr planes so bilinear upsample best reconstructs the original chroma.

> Chung, K.-L. and Lee, Y.-L. (2019). "Effective Gradient Descent-Based Chroma Subsampling Method for Bayer CFA Images in HEVC." IEEE Transactions on Circuits and Systems for Video Technology, Vol. 29, No. 10. https://ieeexplore.ieee.org/document/8519784/

> Chung, K.-L., Lee, Y.-L. and Chien, W.-C. (2020). "Improved Gradient Descent-Based Chroma Subsampling Method for Color Images in VVC." arXiv:2009.10934. https://arxiv.org/abs/2009.10934

### Perceptually Based Downscaling of Images (Öztireli & Gross, SIGGRAPH 2015)

Formulates image downscaling as an optimization problem where the difference between input and output images is measured using a perceptual quality metric. The key distinction from OptL2: this optimizes the downscaled image to *look good at low resolution* (perceptual fidelity), whereas OptL2 optimizes the downscaled image to *reconstruct well when upsampled* (prediction fidelity). These are fundamentally different objectives. The solution is derived in closed-form, leading to a simple parallelizable implementation.

> Öztireli, A. C. and Gross, M. (2015). "Perceptually Based Downscaling of Images." ACM Transactions on Graphics (Proc. SIGGRAPH), Vol. 34, No. 4, pp. 77:1–77:10. https://dl.acm.org/doi/10.1145/2766891

### Learned Image Downscaling for Upscaling using Content Adaptive Resampler (Sun & Chen, IEEE TIP 2020)

The neural network approach to the same problem: train a Content Adaptive Resampler (CAR) network that generates per-pixel downsampling kernels, optimized end-to-end by backpropagating reconstruction error through a super-resolution upsampling network. Achieves state-of-the-art results by making the downsampling content-aware.

Key differences from OptL2: CAR learns a general model across images (amortized), uses a neural upsampler (not bilinear), and requires training data. OptL2 optimizes per-image with an exact gradient, exploits the linearity of bilinear upsampling directly, requires no training, and runs in ~0.5s per tile family.

> Sun, W. and Chen, Z. (2020). "Learned Image Downscaling for Upscaling using Content Adaptive Resampler." IEEE Transactions on Image Processing, Vol. 29, pp. 4027–4040. https://arxiv.org/abs/1907.12904

### Dual-Layer Image Compression via Adaptive Downsampling (2023)

The ADDL system jointly optimizes content-adaptive downsampling kernels (constrained to Gabor filter forms) with a deep neural upsampler for image compression. The downsampled image is compressed as a base layer, then the decoder uses a neural network aided by the downsampling kernel parameters to upconvert. Unlike OptL2, both downsampler and upsampler are learned, and a neural network is required at decode time — breaking the decoder-transparency that ORIGAMI maintains.

> "Dual-layer Image Compression via Adaptive Downsampling and Spatially Varying Upconversion." (2023). arXiv:2302.06096. https://arxiv.org/abs/2302.06096

### Encoder-Side Downsampling for Low Bitrate Compression (Brandi et al., 2006; Lin et al., 2009)

A long line of work on downsampling images or video frames before encoding and upsampling after decoding to improve rate-distortion performance at low bitrates. The core insight is shared with OptL2: the downsampled representation should be optimized for reconstruction quality, not visual fidelity at low resolution. These methods typically use adaptive downsampling ratios and directions rather than per-pixel optimization.

> Brandi, F. et al. (2006). "Adaptive Downsampling to Improve Image Compression at Low Bit Rates." IEEE Transactions on Image Processing, Vol. 15, No. 9. https://ieeexplore.ieee.org/document/1673434/

> Lin, Y.-H. and Wu, J.-L. (2009). "Low Bit-Rate Image Compression via Adaptive Down-Sampling and Constrained Least Squares Upconversion." IEEE Transactions on Image Processing. https://pubmed.ncbi.nlm.nih.gov/19211331/

### Differentiable JPEG and Soft Quantization

For optimizing pixel values that will pass through a quantization step (as OptL2's results do when residuals are JPEG-encoded), the key techniques in the literature are:

- **Straight-Through Estimator (STE):** Treat the non-differentiable quantizer as identity in the backward pass, allowing gradient flow through rounding operations.
- **Task-Aware Quantization:** Learn JPEG quantization tables via gradient descent for specific downstream objectives.

OptL2 sidesteps this problem entirely: it optimizes L2 pixels to minimize *pre-quantization* residual energy, which is a smooth differentiable objective. The JPEG quantization of residuals is downstream and not differentiated through. This is valid because smaller residuals compress better at any fixed quality — the relationship between residual energy and compressed size is monotonic.

> "Task-Aware Quantization Network for JPEG Image Compression." (2020). ECCV 2020. https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650307.pdf

### Where ORIGAMI's OptL2 Fits

The per-pixel optimization of a downsampled image for reconstruction quality through a fixed linear upsampler is well-established in signal processing (Nehab & Hoppe, 2014) and has been independently described for practical use (Wronski, 2021) and for chroma subsampling specifically (Chung & Lee, 2019). Neural approaches generalize this to learned upsamplers (Sun & Chen, 2020).

ORIGAMI's contribution is the application of this optimization within a **residual pyramid compression pipeline**, where:

1. The optimized L2 feeds into a multi-level prediction chain (L2→L1→L0) with cascading benefits
2. The interaction with chroma subsampling modes (4:2:0 vs 4:4:4) is analyzed, revealing that 4:2:0 destroys optimized chroma and that switching to 4:4:4 is the single largest quality improvement available (+1.1 to +1.8 dB)
3. The optimization stacks cleanly with split-quality encoding, attacking different aspects of the compression (prediction quality vs bit allocation)
4. All three RGB channels are optimized jointly, providing consistent chroma improvement even at high residual quality where luma-only gains vanish
5. The decoder remains completely unchanged — no neural networks, no side information, no added complexity
