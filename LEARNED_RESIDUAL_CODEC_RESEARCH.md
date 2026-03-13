# Learned Residual Codec — Research Notes

## STATUS: CONCLUDED — Learned codec not viable for ORIGAMI residuals

Compressibility analysis (March 2026) showed that ORIGAMI's luma residuals are
near-noise-like with only **0.29 bpp of spatial redundancy** — the theoretical
ceiling for any context-based codec improvement over memoryless coding. JPEG
already captures most of this. A learned codec cannot achieve the 30-50%
compression target. See [Section 12: Compressibility Analysis Results](#12-compressibility-analysis-results)
and [Section 13: Conclusion](#13-conclusion).

**Recommended path forward:** Improve the predictor (SR model / refiner models)
to shrink residuals at the source. R6 refiner already reduced residuals from
128→124 KB; deeper models should push further.

---

## Table of Contents
1. [Paradigm: Conditional Coding vs Residual Coding](#1-paradigm-conditional-coding-vs-residual-coding)
2. [Tiny Decoder Architectures](#2-tiny-decoder-architectures)
3. [Lightweight Attention Mechanisms](#3-lightweight-attention-mechanisms)
4. [Entropy Models](#4-entropy-models)
5. [Sparsity-Aware Techniques](#5-sparsity-aware-techniques)
6. [Loss Functions](#6-loss-functions)
7. [Quantization-Aware Training](#7-quantization-aware-training)
8. [Edge-Aware and Structure-Aware Techniques](#8-edge-aware-and-structure-aware-techniques)
9. [Skip Connections in Compression Autoencoders](#9-skip-connections-in-compression-autoencoders)
10. [Video Codec Residual Coding (Closest Analogy)](#10-video-codec-residual-coding-closest-analogy)
11. [Recommended Architecture for ORIGAMI](#11-recommended-architecture-for-origami)

---

## 1. Paradigm: Conditional Coding vs Residual Coding

The most impactful finding: modern learned codecs are moving away from pixel-domain residual coding toward **conditional coding**, where the prediction is a side input to the encoder and decoder rather than being subtracted in pixel space.

### Why Pixel-Domain Residuals Are Suboptimal

Traditional approach:
```
residual = ground_truth - prediction
compressed = Encoder(residual)
reconstructed = Decoder(compressed) + prediction
```

Problems:
- Bad predictions create high-entropy residuals that are hard to compress
- The fixed subtraction is rigid — the network can't learn a better decomposition
- Residuals from bad predictions carry redundant information the network can't exploit

### Conditional Coding (Better)

```
latent = Encoder(ground_truth, prediction)    # prediction as side input
output = Decoder(latent, prediction)          # prediction as side input
```

The network learns what information is already in the prediction and what needs to be transmitted. The prediction acts as free side information at the decoder — no bits spent on it.

### Key Papers

**MaskCRT** — Masked Conditional Residual Transformer. Learns a soft pixel-adaptive mask that blends conditional coding with residual coding. Shows that pure residual coding is suboptimal when prediction quality varies spatially.
- Paper: https://arxiv.org/abs/2312.15829
- Published: IEEE TCSVT 2024

**DCVC** — Deep Contextual Video Compression (Microsoft). Uses conditional coding in feature domain. The prediction is a "context" that conditions the autoencoder. Outperforms residual-based video codecs.
- Paper: https://arxiv.org/abs/2111.11639
- Code: https://github.com/microsoft/DCVC

**Masked Feature Residual Coding** — Performs masking and subtraction in feature domain rather than pixel domain. Features are more compressible than pixel-domain residuals.
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12299318/
- Published: Sensors 2025

### Relevance to ORIGAMI

Our decoder already has the L2 upsampled prediction available (it computes it from the L2 baseline in the pack file). Feeding this prediction into the learned decoder as a conditioning signal costs zero extra bits and lets the network adapt its reconstruction based on prediction quality at each spatial location.

---

## 2. Tiny Decoder Architectures

For CPU decode <10ms on 1024x1024 grayscale, we need decoders under ~200K params and ~50K MACs/pixel.

### Shallow-NTC (Yang & Mandt, ICCV 2023) — Most Relevant

The most directly applicable paper. Designs an asymmetric codec with a heavy encoder (iterative optimization at encode time) and a shallow 2-layer decoder.

- **Decoder**: Just 2 transposed convolutions (kernel 13 stride 8 + kernel 5 stride 2)
- **Latent**: 12 channels at 1/16 spatial resolution
- **Decode complexity**: <50K FLOPs/pixel — 80% reduction vs standard hyperprior
- **R-D performance**: Competitive with full mean-scale hyperprior (Minnen 2018)
- **Key insight**: Encoder complexity doesn't matter for deployment. Spend all your parameter budget on the encoder, keep decoder minimal.

- Paper: https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Computationally-Efficient_Neural_Image_Compression_with_Shallow_Decoders_ICCV_2023_paper.pdf
- Code: https://github.com/mandt-lab/shallow-ntc

### Cool-Chic Decoder (~800 params, per-image overfitted)

Not amortized (per-image overfitted), but the decoder architecture is instructive as a lower bound on what's possible.

- **Decoder**: MLP (7→40→ReLU) + two 3x3 residual convolutions
- **Decode complexity**: ~900 MACs/pixel
- **R-D performance**: Matches VVC at very low decoder complexity
- **Limitation**: 30-120s encode per image (gradient descent optimization) — too expensive for ORIGAMI

- Paper: https://arxiv.org/abs/2401.02156
- Code: https://github.com/Orange-OpenSource/Cool-Chic

### ShiftLIC (2025) — Lightweight via Shift Operations

Replaces expensive convolutions with shift operations for ultra-low-complexity decode.

- Paper: https://arxiv.org/abs/2503.23052

### AsymLLIC (Dec 2024) — Asymmetric Lightweight Codec

Introduces "gradual substitution training" — train with a complex decoder, then progressively replace modules with simpler ones. The simpler decoder retains most of the quality because the encoder adapts.

- Paper: https://arxiv.org/abs/2412.17270

### Ultra-Efficient Decoding (2025)

Survey and techniques for reducing end-to-end neural codec decode cost.

- Paper: https://arxiv.org/abs/2510.01407

### EVC — Towards Real-Time with Mask Decay (2023)

Uses mask decay to progressively prune the decoder during training, achieving real-time decode.

- Paper: https://arxiv.org/abs/2302.05071

---

## 3. Lightweight Attention Mechanisms

For a <200K param budget, full self-attention is too expensive. These mechanisms add negligible params but help the codec focus on non-zero residual regions.

### ECA — Efficient Channel Attention (Best Fit)

Replaces SE's two fully-connected layers with a single 1D convolution (kernel size k=5). Adds literally 5 parameters per attention block. Channel attention without dimensionality reduction.

- Paper: https://arxiv.org/abs/1910.03151
- Code: https://github.com/BangguWu/ECANet

| Mechanism | Extra Params (64ch) | Notes |
|-----------|-------------------|-------|
| **ECA** | 5 per block | 1D conv, no FC, no reduction ratio |
| SE | 512 per block | Two FC layers with reduction r=16 |
| CBAM | ~520 per block | SE + 7x7 spatial conv |
| Coordinate Attention | ~512 per block | Encodes positional info |

### SE — Squeeze-and-Excitation

The original channel attention. Two FC layers with reduction ratio r.

- Paper: https://arxiv.org/abs/1709.01507
- Code: https://github.com/hujie-frank/SENet

### CBAM — Convolutional Block Attention Module

Sequential channel + spatial attention. The spatial component (7x7 conv on max/avg pooled features) adds cheap spatial gating.

- Paper: https://arxiv.org/abs/1807.06521
- Code: https://github.com/Jongchan/attention-module

### Coordinate Attention

Encodes horizontal and vertical positional information into channel attention. Good for tasks where spatial position matters (less clear for residuals).

- Paper: https://arxiv.org/abs/2103.02907

### Recommendation for ORIGAMI

Use **ECA** (5 params per block) in every conv block of the decoder. For spatial attention on non-zero regions, add a single 1x1 conv → sigmoid spatial gate (~C params) after the final ECA block. Total attention overhead: <100 params.

---

## 4. Entropy Models

The entropy model determines how efficiently the quantized latent tensor is serialized to bits. This is where most of the compression improvement comes from.

### Spectrum: Speed vs Compression

| Model | Serial Decode Steps | BD-Rate vs Factorized | Decode Speed |
|-------|--------------------|-----------------------|-------------|
| Factorized prior | 1 (fully parallel) | Baseline | Fastest |
| Channel-wise AR | N_groups (~5-10) | -5 to -10% | Fast |
| **Checkerboard context** | **2** | **-10 to -15%** | **Very fast** |
| ELIC unevenly-grouped | 5-10 groups | -15 to -20% | Fast |
| Full spatial AR | H×W | -20 to -25% | Very slow |

### Factorized Prior (Ballé 2017) — Simplest

Each latent channel has a learned piecewise-linear CDF. All elements decoded in parallel. No context.

- Paper: https://arxiv.org/abs/1611.01704 ("End-to-end Optimized Image Compression")
- Code (CompressAI): https://github.com/InterDigitalInc/CompressAI

### Checkerboard Context (He et al., CVPR 2021) — Best Speed/Quality Tradeoff

Divides latent into "anchor" and "non-anchor" positions in a checkerboard pattern. Anchors decoded first (parallel), then non-anchors conditioned on anchors (parallel). Only 2 serial passes, gets ~70% of full autoregressive benefit.

- Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/He_Checkerboard_Context_Model_for_Efficient_Learned_Image_Compression_CVPR_2021_paper.pdf
- Also described in ELIC paper below

### ELIC — Unevenly Grouped Space-Channel Context (He et al., CVPR 2022)

Groups latent channels into uneven groups. Decodes group-by-group, each conditioned on previous groups. 5-10 serial steps. Best practical entropy model.

- Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.pdf
- Code: https://github.com/VincentChandworker/ELIC

### Laplacian-Guided Entropy Model (CVPR 2024)

Uses Laplacian-shaped positional encoding with learnable parameters, adaptively adjusted per channel cluster. Specifically designed for distributions matching our residual statistics (Laplacian, not Gaussian).

- Paper: https://arxiv.org/abs/2403.16258

### FlashGMM — Fast Gaussian Mixture Entropy Coding (2025)

Accelerates Gaussian Mixture Model entropy coding by 90x using dynamic binary search instead of CDF table lookup. Eliminates the main computational bottleneck of GMM-based entropy models.

- Paper: https://arxiv.org/abs/2509.18815

### Laplacian vs Gaussian Mixture Models

Standard learned codecs use Gaussian mixture models (GMM) for the latent distribution. But our residuals are Laplacian-distributed (sharp peak at zero, heavy tails). Using a **Laplacian mixture model** (3 components: 3 means + 3 scales + 3 weights = 9 params per channel) better matches the actual distribution and saves bits.

### Recommendation for ORIGAMI

**Checkerboard context** with **Laplacian mixture** (3 components). Only 2 serial decode passes (anchors, then non-anchors). Combined with a lightweight hyperprior (~40K params) for spatial adaptation of mixture parameters.

---

## 5. Sparsity-Aware Techniques

Our residuals are centered at 128 with most values near zero. 40-60% of 8x8 blocks in typical WSI residuals are near-zero. Exploiting this sparsity saves both bits and compute.

### Learned Block-Skip Mask

Transmit a 1-bit-per-block mask indicating which 8x8 blocks have non-trivial residuals. Skip encoding/decoding for near-zero blocks. For typical WSI residuals, this saves 40-60% of compute and bits.

Implementation: encoder predicts mask from input, threshold at inference. Mask entropy-coded (very cheap — high spatial correlation in mask).

### Soft Gating

A learned 1x1 conv → sigmoid produces a spatial attention mask. Multiply features by this mask. The network learns to zero out flat regions. Cost: C extra parameters per gate.

### Channel Gating (NeurIPS 2024)

Achieves up to 8x measured speedup with 96-98% sparsity in standard vision networks. Apply binary gating to channels — skip entire channels that are near-zero for a given input.

### Generative Sparse Representation (ACCV 2024)

Embeds inputs into discrete latent space spanned by learned visual codebooks. Transmits integer codeword indices. For sparse residuals, most codewords map to "zero region" requiring very few bits.

- Paper: https://openaccess.thecvf.com/content/ACCV2024W/RichMediaGAI/papers/Zhou_Image_and_Video_Compression_using_Generative_Sparse_Representation_with_Fidelity_ACCVW_2024_paper.pdf

### Recommendation for ORIGAMI

Start simple: **learned block-skip mask** (1 bit per 8x8 block) as side information. Encoder predicts which blocks to skip; decoder fills skipped blocks with 128 (zero residual). More sophisticated sparsity can be added later if needed.

---

## 6. Loss Functions

### Charbonnier Loss (Primary Distortion Loss)

```python
L_charb = sqrt((output - target)^2 + epsilon^2)    # epsilon = 1e-6
```

A differentiable approximation to L1. Less sensitive to outliers than MSE. Better handles the peaked Laplacian distribution of residuals — MSE over-penalizes large deviations (which are rare) and under-penalizes small ones (which are common).

- Originally from: Charbonnier et al. 1994, "Two deterministic half-quadratic regularization algorithms for computed imaging"
- Widely used in super-resolution: https://arxiv.org/abs/2004.02967 (SwinIR)

### Frequency-Domain Loss (DCT or FFT)

```python
L_freq = MSE(DCT(output), DCT(target))    # or FFT
```

Residuals have energy spread across frequencies (flat spectrum). A frequency-domain loss ensures high-frequency components (the most perceptually important in residuals) are preserved. Optional frequency weighting to emphasize high frequencies.

- FDNet (2024): https://arxiv.org/abs/2401.08895 — Frequency Decomposition Network, decomposes into high/low frequency with separate losses.
- FFT loss used in our existing `wsi_sr/train.py` (already implemented)

### Sobel Edge-Weighted MSE (Structure Preservation)

```python
grad_weight = 1 + alpha * |Sobel(target)|    # alpha = 2-5
L_edge = MSE(output * grad_weight, target * grad_weight)
```

Weights reconstruction error by local gradient magnitude. Directs capacity toward edges and tissue boundaries where prediction errors are largest and most perceptually visible. Zero extra parameters.

- Related: EAGLE loss (2024) — Edge-Aware Gradient Localization Enhanced loss. Applies spectral analysis of localized features within gradient changes.

### Magnitude-Weighted MSE (Residual-Specific)

```python
L_mag = MSE(output, target) * (1 + beta * |target - 128|)    # beta = 0.5-2.0
```

Weights errors proportionally to residual magnitude. Near-zero residuals (near 128) matter less — errors there have minimal visual impact on the reconstructed tile. Large residuals (far from 128) carry the critical correction signal and must be preserved precisely.

### Uncertainty-Weighted R-D Loss (UGDiff, 2024)

Learns per-pixel uncertainty and uses it to weight the rate-distortion tradeoff adaptively. High-uncertainty regions get more bits.

- Paper: https://arxiv.org/abs/2407.12538

### Recommended Loss Combination

```python
L_total = lambda_rate * R
        + 1.0 * L_charb
        + 0.1 * L_freq
        + 0.05 * L_edge
```

Where:
- `R` = estimated bitrate from entropy model
- `L_charb` = Charbonnier (primary reconstruction loss, replaces MSE)
- `L_freq` = DCT/FFT domain MSE (preserves high frequencies)
- `L_edge` = Sobel gradient-weighted MSE (preserves structure)
- `lambda_rate` = rate-distortion tradeoff (sweep across training runs)

---

## 7. Quantization-Aware Training

The gap between training (continuous noise) and inference (discrete rounding) is a key challenge in learned compression.

### Standard: Uniform Noise (Ballé 2017)

During training, replace `round(y)` with `y + U(-0.5, 0.5)`. This makes the rate estimation differentiable. Problem: train-test mismatch because uniform noise ≠ rounding.

- Paper: https://arxiv.org/abs/1611.01704

### Soft-Then-Hard with Annealing (Best Practice)

Train in two phases:
1. **Phase 1 (soft)**: Use scaled additive uniform noise with learnable scale per channel. Noise scale starts at 0.5, can adapt.
2. **Phase 2 (hard)**: Switch to rounding with straight-through estimator (STE). Gradient flows through the round operation unchanged.

The scaled noise variant adds ~N_channels learnable parameters (one sigma per channel). Each channel learns its own quantization granularity.

- Paper: Guo et al., "Soft then Hard: Rethinking the Quantization in Neural Image Compression", ICML 2021
- https://arxiv.org/abs/2104.05168

### Quantization Rectifier (2024)

Adds a lightweight network that predicts and corrects quantization error at decode time. Extra cost at decode (~5K params) but improves R-D by 2-5%.

- Paper: https://arxiv.org/abs/2403.17236

### STE Variants

- **Vanilla STE**: `forward: round(y)`, `backward: identity` (gradient passes through round)
- **Annealed STE**: Gradually sharpen a soft quantization function (e.g., `tanh(beta * (y - round(y)))`) toward hard rounding. Start with small beta (soft), increase during training.
- **Stochastic rounding**: During training, round up/down with probability proportional to distance. Unbiased estimator.

### Recommendation for ORIGAMI

Use **soft-then-hard** with annealing:
- Epochs 1-70: Uniform noise with learnable per-channel scale (N_channels extra params)
- Epochs 70-100: Switch to STE with hard rounding for fine-tuning
- This is current best practice and closes most of the train-test gap

---

## 8. Edge-Aware and Structure-Aware Techniques

### Sobel/Scharr Gradient Maps as Auxiliary Loss

Compute edge maps from the target residual and use them to weight the reconstruction loss (see Loss Functions section). Zero parameters, trivially computed.

### Edge Maps as Side Information

Some codecs transmit edge detection results as compact side information to guide the decoder. For residuals, this could help the decoder know where to allocate capacity. However, edge maps of residuals are themselves sparse, so the benefit may be marginal — worth testing.

- Paper: "Edge-based Denoising Image Compression" (2024): https://arxiv.org/abs/2409.10978 — Uses edge maps + depth maps from latent space as sub-information for diffusion-based enhancement.

### EAGLE Loss (2024)

Edge-Aware Gradient Localization Enhanced loss. Applies spectral analysis of localized features within gradient changes. Cheap to compute, can be added as auxiliary training loss.

### Structure Similarity Loss (MS-SSIM)

Multi-scale SSIM as a loss term. Captures structural distortion better than MSE. Well-suited for residuals where structural integrity matters more than absolute pixel values.

- Used in many learned codecs as an alternative/complement to MSE
- Available in `pytorch_msssim`: https://github.com/VainF/pytorch-msssim

---

## 9. Skip Connections in Compression Autoencoders

### The Core Tension

Skip connections from encoder to decoder bypass the information bottleneck. Great for reconstruction quality, terrible for compression because information flows "for free" without being entropy-coded.

**In standard image compression autoencoders, encoder-to-decoder skip connections are NOT used.** All information must pass through the bottleneck and be entropy-coded.

### The Hyperprior IS a Skip Connection

The hyperprior (Ballé 2018) sends side information (mean/variance of latent distribution) from encoder to decoder through its own bottleneck. This is a principled skip connection — the skip path goes through its own quantization and entropy coding.

- Paper: https://arxiv.org/abs/1802.01436 ("Variational Image Compression with a Scale Hyperprior")

### QARV — Hierarchical VAE with Multi-Scale Bottlenecks (2023)

Uses U-Net-like multi-scale connections, but each skip path goes through its own quantization and entropy coding. This is the correct way to add multi-scale information in a compression autoencoder.

### Why Conditional Coding Changes This

For ORIGAMI, the prediction (L2 upsampled) is already available at the decoder — it's free side information that doesn't go through the bottleneck. Feeding the prediction into the decoder at multiple scales as conditioning is NOT a skip connection violation because the prediction isn't encoder output — it's independently available.

```
Decoder input: latent (from bottleneck) + prediction (free, from L2 baseline)
                    ↓ costs bits              ↓ costs zero bits
```

This is exactly why conditional coding outperforms residual coding — the decoder gets useful side information without spending bits on it.

### Recommendation for ORIGAMI

- **No encoder-to-decoder skip connections** (would bypass bottleneck)
- **Prediction as multi-scale conditioning**: Downsample the L2 prediction to match each decoder layer's spatial resolution and concatenate. This is free side information.

---

## 10. Video Codec Residual Coding (Closest Analogy)

ORIGAMI's residual compression is structurally identical to video codec prediction residual coding. The prediction (upsampled L2) is analogous to a motion-compensated prediction frame.

### LCEVC (MPEG-5 Part 2) — Production Standard

The closest existing standard to ORIGAMI's architecture. Encodes residual enhancement layers at two scales on top of a base codec. The neural extension uses a 5-layer factorized CNN with <2KB params and ~500 MACs/pixel.

- Overview: https://www.lcevc.org/how-lcevc-works/
- Spec: ISO/IEC 23094-2

### DCVC-FM (Microsoft, 2024)

Feature-domain conditional coding for video. The prediction frame is a context that conditions the autoencoder. Does not subtract prediction in pixel domain.

- Paper: https://arxiv.org/abs/2302.02721
- Code: https://github.com/microsoft/DCVC

### Learned Residual Prediction

Multiple papers show that learning to predict the residual (from the base reconstruction + spatial context) before entropy coding yields 10-15% bitrate savings. The codec only needs to transmit the residual-of-the-residual.

- Paper: "Learned Video Compression with Residual Prediction and Loop Filter"
- https://arxiv.org/abs/2108.08551

### Paper Collections

Comprehensive collection of learned image/video compression papers:
- https://github.com/cshw2021/Learned-Image-Video-Compression

Introduction to learned image compression (excellent tutorial):
- https://yodaembedding.github.io/post/learned-image-compression/

---

## 11. Recommended Architecture for ORIGAMI

Based on all of the above research, here is the recommended architecture. See `LEARNED_RESIDUAL_CODEC_PLAN.md` for the training plan.

### Paradigm

**Conditional coding** — prediction as free side information to encoder and decoder, not pixel-domain subtraction.

### Encoder (GPU, offline, can be heavy)

```
Input: ground_truth_Y (1024x1024x1) + prediction_Y (1024x1024x1) → concat → 2ch

Conv 2ch → 32ch, k=5, s=2   (512x512)
Conv 32 → 64, k=5, s=2      (256x256)
Conv 64 → 64, k=3, s=2      (128x128)
ECA attention
Conv 64 → C_latent, k=3, s=2 (64x64)
Quantize → entropy encode (ANS)

Encoder params: ~100K (doesn't matter, runs offline)
```

### Decoder (CPU, must be fast)

```
Entropy decode (ANS) → latent (64x64 × C_latent)

TransConv C_latent → 64ch, k=5, s=2    (128x128)
  + concat downsampled prediction (128x128)
ECA attention
TransConv 64+1 → 32ch, k=5, s=2        (256x256)
  + concat downsampled prediction (256x256)
TransConv 32+1 → 16ch, k=5, s=2        (512x512)
  + concat downsampled prediction (512x512)
TransConv 16+1 → 1ch, k=5, s=2         (1024x1024)

Decoder params: ~60-90K
```

At each scale, the L2 prediction is downsampled and concatenated as a free conditioning channel.

### Entropy Model

**Checkerboard context** with **Laplacian mixture** (3 components per channel):
- Lightweight hyperprior: 3-layer conv encoder + decoder (~40K params)
- 2 serial decode passes (anchors, then non-anchors)
- FlashGMM-style fast CDF lookup for Laplacian mixture

Entropy model params: ~45K

### Loss

```python
L = lambda_rate * Rate
  + 1.0 * Charbonnier(output, target)
  + 0.1 * FFT_loss(output, target)
  + 0.05 * Sobel_weighted_MSE(output, target)
```

### Quantization

Soft-then-hard with learnable per-channel noise scale:
- Epochs 1-70: Uniform noise, learnable sigma per channel
- Epochs 70-100: STE with hard rounding

### Expected Performance

| Metric | Target | Basis |
|--------|--------|-------|
| Decoder params | <100K | Shallow-NTC achieves competitive R-D with <50K decoder FLOPs/pixel |
| Decode latency | <10ms / 1024x1024 | 4 transposed convs on CPU via ONNX Runtime |
| Encode latency | <100ms / 1024x1024 | 4 convs + entropy coding on GPU |
| Bitrate savings | 25-40% vs JPEG | Conditional coding + Laplacian entropy + checkerboard context |
| R-D quality | Within 0.1 dB of JPEG path | At matched residual size |

### Entropy Coding Library

**`constriction`** — has both Python (training) and Rust (inference) APIs with guaranteed bitstream compatibility. Use its ANS backend.

- Docs: https://bamler-lab.github.io/constriction/
- Code: https://github.com/bamler-lab/constriction
- Rust crate: https://crates.io/crates/constriction
- Python: `pip install constriction`

---

## Key References (Sorted by Relevance)

### Must-Read (Directly Applicable)

1. **Shallow-NTC** — Computationally-Efficient Neural Image Compression with Shallow Decoders (ICCV 2023)
   - Paper: https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Computationally-Efficient_Neural_Image_Compression_with_Shallow_Decoders_ICCV_2023_paper.pdf
   - Code: https://github.com/mandt-lab/shallow-ntc

2. **Checkerboard Context Model** — Efficient Learned Image Compression (CVPR 2021)
   - Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/He_Checkerboard_Context_Model_for_Efficient_Learned_Image_Compression_CVPR_2021_paper.pdf

3. **ELIC** — Unevenly Grouped Space-Channel Contextual Adaptive Coding (CVPR 2022)
   - Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/He_ELIC_Efficient_Learned_Image_Compression_With_Unevenly_Grouped_Space-Channel_Contextual_CVPR_2022_paper.pdf
   - Code: https://github.com/VincentChandworker/ELIC

4. **End-to-end Optimized Image Compression** — Factorized prior (Ballé et al., ICLR 2017)
   - Paper: https://arxiv.org/abs/1611.01704

5. **Variational Image Compression with a Scale Hyperprior** (Ballé et al., ICLR 2018)
   - Paper: https://arxiv.org/abs/1802.01436

6. **Soft then Hard Quantization** (Guo et al., ICML 2021)
   - Paper: https://arxiv.org/abs/2104.05168

### Important (Techniques to Borrow)

7. **MaskCRT** — Conditional vs residual coding (IEEE TCSVT 2024)
   - Paper: https://arxiv.org/abs/2312.15829

8. **DCVC** — Deep Contextual Video Compression (Microsoft)
   - Paper: https://arxiv.org/abs/2111.11639
   - Code: https://github.com/microsoft/DCVC

9. **ECA-Net** — Efficient Channel Attention (CVPR 2020)
   - Paper: https://arxiv.org/abs/1910.03151
   - Code: https://github.com/BangguWu/ECANet

10. **Laplacian-Guided Entropy Model** (CVPR 2024)
    - Paper: https://arxiv.org/abs/2403.16258

11. **FlashGMM** — Fast Gaussian Mixture Entropy Coding (2025)
    - Paper: https://arxiv.org/abs/2509.18815

12. **Quantization Rectifier** (2024)
    - Paper: https://arxiv.org/abs/2403.17236

### Background (Good to Know)

13. **Cool-Chic** — Per-image overfitted codec (Orange, 2024)
    - Paper: https://arxiv.org/abs/2401.02156
    - Code: https://github.com/Orange-OpenSource/Cool-Chic

14. **CompressAI** — Learned compression library (InterDigital)
    - Code: https://github.com/InterDigitalInc/CompressAI
    - Note: Decoder too heavy for CPU. Useful as reference implementation only.

15. **constriction** — Entropy coding library (Python + Rust)
    - Code: https://github.com/bamler-lab/constriction
    - Docs: https://bamler-lab.github.io/constriction/

16. **L3IC** — Lightweight Learned Lossless Image Compression (ICLR 2020)
    - Code: https://github.com/pkorus/l3ic

17. **Learned Image/Video Compression Paper Collection**
    - https://github.com/cshw2021/Learned-Image-Video-Compression

18. **Introduction to Learned Image Compression** (Tutorial)
    - https://yodaembedding.github.io/post/learned-image-compression/

---

## 12. Compressibility Analysis Results (March 2026)

Ran `evals/scripts/analyze_residual_compressibility.py` on 72 actual residual images (42 lanczos3, 28 SR model, 2 other) from `evals/runs/`.

Charts: `evals/analysis/residual_compressibility/`

### Key Numbers

| Metric | Lanczos3 | SR Model | Implication |
|--------|----------|----------|-------------|
| 0th-order entropy | 4.72 bpp | 4.67 bpp | Per-pixel information content |
| 1st-order entropy | 4.43 bpp | 4.43 bpp | With 1-pixel context |
| **Spatial redundancy** | **0.29 bpp** | **0.24 bpp** | **Ceiling for context-based coding gain** |
| PNG lossless | 4.18 bpp | 4.23 bpp | Best lossless achievable |
| Autocorrelation lag-1 | 0.63 | 0.58 | Moderate immediate-neighbor correlation |
| Autocorrelation lag-4 | -0.08 | -0.04 | Oscillatory (lanczos kernel ringing) |
| Autocorrelation lag-16 | 0.04 | 0.06 | Effectively zero — no long-range structure |
| Spectral slope | +0.39 | +0.39 | Flat/rising — opposite of natural images |
| Block sparsity (<3) | 10.3% | 10.9% | Not sparse — block-skip won't help |
| Std deviation | 6.6 | 6.4 | Narrow, consistent range |
| Laplacian scale (b) | 4.8 | 4.6 | Excellent Laplacian fit |

### Interpretation

1. **Spatial redundancy is tiny (0.24-0.29 bpp).** This is the theoretical maximum gain any context-based codec (checkerboard, autoregressive, learned) could achieve over memoryless coding. JPEG's Huffman already captures a portion of this. Real-world improvement ceiling: ~0.15-0.20 bpp.

2. **Spectrum is flat/rising, not falling.** Natural images follow 1/f² power law (strong low-frequency dominance). Our residuals have the opposite — energy peaks in the mid-to-high frequencies. This means:
   - DCT doesn't compact energy well (no low-frequency dominance to exploit)
   - But there's also no LF/HF split to leverage (wavelet decomposition won't help)
   - The entire signal is essentially high-frequency texture

3. **Autocorrelation is short-range and oscillatory.** Strong at lag-1 (0.63), negative at lag-3/4 (-0.08), small positive at lag-7/8 (0.12), then flat. This oscillatory pattern is the **lanczos3 kernel's frequency signature** — the 6-tap filter creates predictable ringing. This is NOT exploitable tissue structure.

4. **Not sparse.** Only 10% of 8x8 blocks have mean |residual| < 3. Block-skip masks won't help. The residual energy is spread uniformly across the tile.

5. **Distribution is consistently Laplacian (b=4.6-4.8).** Very tight across images — confirms domain specificity. Gaussian is a poor fit. A Laplacian entropy model is correct but doesn't help much because the spatial redundancy is so small.

6. **SR model residuals are slightly less compressible** (0.24 bpp redundancy vs 0.29 for lanczos3). The SR model removes more spatial structure, leaving even more noise-like residuals.

### Why No Amortized Learned Codec Can Beat JPEG Here

The minimum amortized codec to beat JPEG on natural images needs ~3-4M params (Ballé 2017). Natural images have ~2-3 bpp of spatial redundancy. Our residuals have 0.29 bpp. That's an order of magnitude less exploitable structure. Even a perfect learned entropy model gains at most 0.29 bpp — roughly 6% bitrate reduction at our operating point (~4.7 bpp).

JPEG at Q40-Q60 operates at 0.4-0.6 bpp on these residuals via lossy quantization of DCT coefficients. The quantization does most of the work, not the entropy coding. A learned codec that does better quantization (rate-distortion optimized) might gain a few percent, but the 30-50% target is unreachable.

### SRA Training Results Confirm This

| Codec | BPP | PSNR | KB/residual |
|-------|-----|------|-------------|
| JPEG Q40 | 0.41 | 35.56 | 52 KB |
| JPEG Q60 | 0.62 | 38.10 | 79 KB |
| JPEG Q80 | 1.13 | 41.36 | 145 KB |
| SRA-Tiny (ep40) | 1.07 | 28.38 | 137 KB |
| SRA-Medium (ep35) | 2.37 | 28.95 | 302 KB |

SRA models use similar or more bits than JPEG but achieve 13 dB worse PSNR. The bottleneck is not entropy coding — it's that a tiny network (15-150K params) can't learn an effective transform for noise-like signals. Literature confirms: minimum JPEG-beating amortized codec is ~3-4M params on natural images; our signal has even less structure.

---

## 13. Texture Synthesis Approach (Explored, Not Viable)

We explored decomposing residuals into low-frequency structure (code traditionally) + high-frequency texture (synthesize from a tiny latent). The compressibility analysis killed this:

**Why it doesn't work for our residuals:**
- Power spectrum is flat/rising — there's barely any LL subband content to separate
- The residual IS essentially all high-frequency texture
- A wavelet LL/HF split produces an LL that's near-zero (nothing to code cheaply)
- The "synthesize HF from a compact description" approach requires the HF to have learnable statistical patterns; our HF is dominated by lanczos kernel ringing, not tissue texture

**Texture synthesis references explored (for future reference):**
- HiFiC (NeurIPS 2020): GAN-based perceptual compression. ~50-100M decoder params. GPU-only. https://arxiv.org/abs/2006.09965
- Conceptual Compression (IEEE TIP 2022): Structure/texture split + HF-GAN synthesis. https://arxiv.org/abs/2011.04976
- GLC (CVPR 2024): VQ-VAE latent compression, <0.04 bpp natural images. https://openaccess.thecvf.com/content/CVPR2024/papers/Jia_Generative_Latent_Coding_for_Ultra-Low_Bitrate_Image_Compression_CVPR_2024_paper.pdf
- CDC (NeurIPS 2023): Conditional diffusion codec, content/texture split. https://arxiv.org/abs/2209.06950
- DyNCA (CVPR 2023): Neural cellular automata for texture, <1K params. https://dynca.github.io/
- SGAN: Fully convolutional texture GAN, ~50-200K params. https://arxiv.org/abs/1611.08207
- GAN Compression (MIT, CVPR 2020): 9-21x MAC reduction for conditional GANs. https://github.com/mit-han-lab/gan-compression
- ProGIC (2025): RVQ + lightweight backbone for CPU decode. https://arxiv.org/abs/2603.02897
- SQLC (2024): Stain-aware compression for H&E. https://arxiv.org/abs/2406.12623
- LCEVC (MPEG-5): Standardized residual enhancement layers. https://www.lcevc.org/how-lcevc-works/

---

## 14. Conclusion

### What We Learned

1. **ORIGAMI's luma residuals are near-noise-like.** The prediction step (lanczos3 or SR model) removes almost all spatial structure. What remains has only 0.29 bpp of spatial redundancy — an order of magnitude less than natural images.

2. **No amortized learned codec under ~3-4M params has ever beaten JPEG** on natural images (which have 10x more spatial redundancy). For our noise-like residuals, the situation is worse.

3. **Cool-Chic (per-image overfitting)** is the only sub-1K param approach that beats JPEG, but its 30-120s encode time per image is prohibitive for WSI with thousands of families.

4. **The autocorrelation structure is from the lanczos3 kernel**, not from tissue. It's a fixed ringing pattern that could theoretically be exploited by a filter-aware codec, but the gain is tiny.

5. **Texture synthesis won't help** because there's no meaningful LF/HF split in the residual — it's all HF.

### What To Do Instead

**Improve the predictor, not the codec.** Every dB of better prediction translates directly to smaller residuals:

- **SR model improvements** (deeper/wider models, Stage 2 training on 1.4M tiles)
- **Refiner models** (R6/R9/R10) that iteratively improve the prediction
- **Better upsampling filters** or learned upsampling kernels

The prediction quality is the bottleneck. JPEG is a reasonable codec for the residuals that remain.

### When Learned Codecs Might Become Viable

- If future SR models reduce residual std from ~6.5 to ~3.0, the residuals become sparser and more structured — reopening the learned codec path
- If decode hardware changes (GPU at serve time), larger decoder models become feasible
- If Cool-Chic encode speed improves 10x (ongoing research at Orange), per-image overfitting becomes practical
