# Research Report: Neural Residual Compression for WSI Tiles

## Goal

**JPEG q80 perceptual quality at 50-100KB per L0 family (16 tiles of 256x256).**

For H&E and IHC stained whole-slide images. Fast CPU decode, GPU encode is acceptable.

---

## Current ORIGAMI Pipeline Performance

All runs use JXL for both base (L2) and residual (fused L0). Test image: `L0-1024.jpg` (1024x1024, 16 L0 tiles).

### Full Comparison Table (sorted by total size)

| Run | Total KB | PSNR | SSIM | Delta E | LPIPS | Butteraugli | SSIMULACRA2 |
|-----|----------|------|------|---------|-------|-------------|-------------|
| **Target: JPEG q80** | **240.1** | **38.8** | **0.9596** | **0.99** | **0.0147** | **1.70** | **85.9** |
| v2_b90_l0q45 | 50.3 | 35.0 | 0.9069 | 2.12 | 0.1897 | 4.15 | 61.5 |
| v2_b95_l0q45 | 56.6 | 35.1 | 0.9077 | 1.91 | 0.1856 | 3.71 | 65.3 |
| v2_b90_l0q55 | 58.8 | 35.8 | 0.9201 | 2.06 | 0.1569 | 3.69 | 64.8 |
| v2_b95_l0q55 | 65.6 | 35.9 | 0.9209 | 1.84 | 0.1514 | 3.12 | 68.4 |
| v2_b90_l0q65 | 71.8 | 36.9 | 0.9358 | 1.98 | 0.1185 | 3.40 | 68.1 |
| v2_b95_l0q65 | 78.5 | 36.9 | 0.9362 | 1.76 | 0.1152 | 2.68 | 71.8 |
| v2_b90_l0q75 | 96.7 | 38.4 | 0.9546 | 1.89 | 0.0618 | 3.25 | 71.7 |
| v2_b95_l0q75 | 103.7 | 38.4 | 0.9551 | 1.66 | 0.0548 | 2.42 | 75.6 |
| v2_b95_l0q80 | 122.9 | 39.5 | 0.9646 | 1.61 | 0.0369 | 2.29 | 77.6 |
| JXL q85 | 166.6 | 38.7 | 0.9614 | 1.11 | 0.0224 | 1.60 | 85.5 |
| JPEG q80 | 240.1 | 38.8 | 0.9596 | 0.99 | 0.0147 | 1.70 | 85.9 |
| JXL q95 | 345.3 | 44.2 | 0.9890 | 0.66 | 0.0033 | 0.65 | 93.3 |

### Key Observations

1. **ORIGAMI b95/l0q75 at 104KB** matches JPEG q80 on PSNR (38.4 vs 38.8) and SSIM (0.955 vs 0.960) at **2.3x compression**. But perceptual metrics are worse:
   - Delta E: 1.66 vs 0.99 (67% worse)
   - Butteraugli: 2.42 vs 1.70 (42% worse)
   - SSIMULACRA2: 75.6 vs 85.9 (12% worse)

2. **The perceptual gap is NOT from residual JPEG artifacts** — these runs already use JXL for both base and residual.

3. **The gap is likely chroma-driven.** ORIGAMI corrects luma via residual but uses 4x-upsampled L2 chroma for L0 tiles. The L2 base is 256x256 for a 1024x1024 image — that's a 4x chroma upsample with no correction path.

4. **JXL q85 at 167KB** gets much closer to the target perceptual quality (Delta E 1.11, Butteraugli 1.60) but is 67% over the size target.

---

## geninv_codec Prototype Results

The prototype tests replacing the JXL-coded residual with a tiny neural network (encoder E + generator G) that predicts the residual from a compact latent vector.

| Method | Extra bytes | Total KB | Y-PSNR | SSIM | Delta E | LPIPS |
|--------|------------|----------|--------|------|---------|-------|
| Base only (JPEG q50) | 0 | 130 | 36.87 | 0.938 | 1.48 | 0.077 |
| **Latent-only (zdim=8, patch=64)** | **539 B** | **131** | **37.69** | **0.946** | **1.42** | **0.138** |
| True residual JPEG90 | 172 KB | 302 | 42.63 | 0.983 | 1.18 | 0.011 |
| Latent + correction JPEG90 | 160 KB | 290 | 42.79 | 0.983 | 1.17 | 0.011 |

### Assessment

- Latent-only adds +0.8 dB PSNR for 539 bytes — impressive byte efficiency
- But **LPIPS gets worse** (0.077 -> 0.138) — the generator introduces perceptual artifacts
- With correction, the correction layer dominates (160KB correction vs 539B latents)
- The neural path contributes <1% of the enhancement layer's bytes
- **Verdict: Not viable for the stated goal.** The tiny network can't capture enough residual structure to matter, and when it does work, it worsens perceptual quality.

---

## Prior Art Survey

### Most Relevant to Our Problem

#### 1. COOL-CHIC (Orange, ICCV 2023)
- **What:** Per-image overfitted tiny neural decoder + hierarchical latent grid
- **Decode:** ~1728 MACs/pixel, 100ms for 720p on CPU — fast
- **Encode:** Expensive (10k-100k gradient descent iterations per image)
- **Quality:** Matches or beats VVC/H.266 on rate-distortion
- **How it works:** No traditional base codec. The neural net IS the entire codec. Hierarchical multi-resolution latent grid, autoregressive entropy model. Lambda parameter controls rate-distortion tradeoff.
- **Open source:** [github.com/Orange-OpenSource/Cool-Chic](https://github.com/Orange-OpenSource/Cool-Chic)
- **Relevance:** Could either replace ORIGAMI entirely, or replace the JXL residual layer. Best candidate for evaluation.

#### 2. C3 (Google DeepMind, CVPR 2024)
- **What:** Per-image overfitted synthesis network, similar to COOL-CHIC
- **Decode:** <3k MACs/pixel, matches VVC
- **Open source:** [github.com/google-deepmind/c3_neural_compression](https://github.com/google-deepmind/c3_neural_compression)
- **Relevance:** Alternative to COOL-CHIC with potentially better performance.

#### 3. Hybrid Learned Residual (Lee & Hang, CVPR-W 2020)
- **What:** VVC base + pre-trained neural residual encoder
- **Key difference from our approach:** Uses a pre-trained autoencoder for residual (amortized), not per-image training. Outperforms single-layer VVC at ~0.15 bpp.
- **No open source.**

#### 4. LCEVC (MPEG-5 Part 2, ISO 2021)
- **What:** Standardized "base codec + residual enhancement layers" architecture
- **Key difference:** Enhancement is traditional signal processing, not neural. Two sub-layers.
- **Relevance:** Validates the layered architecture concept at production/standards level.

### Less Directly Relevant

| Work | Year | Venue | Key Idea | Open Source |
|------|------|-------|----------|-------------|
| COIN/COIN++ | 2021 | ICLR-W | Per-image MLP weights = compressed image | [GitHub](https://github.com/EmilienDupont/coin) |
| HiFiC | 2020 | NeurIPS | GAN decoder from compact latent | Demo only |
| DLPR | 2021 | CVPR | Lossy base + lossless residual for near-lossless | [GitHub](https://github.com/BYchao100/Deep-Lossy-Plus-Residual-Coding) |
| RECOMBINER | 2024 | ICLR | Bayesian INR, patch-based | [GitHub](https://github.com/cambridge-mlg/RECOMBINER) |
| JPEG AI | 2025 | ISO | First neural image compression standard | Reference software |
| SGA+ | 2024 | NeurIPS | Per-image latent refinement for pre-trained codec | No |

---

## Potential Paths Forward

### Path A: COOL-CHIC as Residual Encoder
Replace JXL residual with COOL-CHIC-encoded residual. Keep ORIGAMI's L2 base layer for fallback/progressive decode.
- **Pro:** Potentially much better compression of the residual signal
- **Con:** Adds neural decode dependency; encode is slow (minutes per tile on CPU, faster on GPU)
- **Need to test:** Does COOL-CHIC encode a grayscale residual map more efficiently than JXL?

### Path B: COOL-CHIC as Full Replacement
Encode each 256x256 tile independently with COOL-CHIC. No ORIGAMI pipeline.
- **Pro:** Handles both luma and chroma — eliminates the chroma gap. State-of-the-art R-D.
- **Con:** Loses the hierarchical pyramid structure (L2/L1/L0 progressive loading). 16 independent encodes per family. Encode is very slow.
- **Need to test:** What size does COOL-CHIC produce at JPEG q80 equivalent quality for a 256x256 H&E tile?

### Path C: Fix ORIGAMI's Chroma Gap
Add a chroma residual to the ORIGAMI pipeline alongside the luma residual.
- **Pro:** Directly addresses the main quality gap. Keeps fast decode. Incremental change.
- **Con:** Adds bytes. May not close the gap enough at 50-100KB.
- **Need to test:** How many bytes does a JXL-coded chroma residual cost? What's the Delta E improvement?

### Path D: Higher-Quality Base (L2)
Increase L2 quality (currently q90/q95) so that 4x-upsampled chroma is better.
- **Pro:** Improves chroma predictions without adding a new residual channel
- **Con:** Bigger L2 eats into the byte budget. Already tested b95 vs b90 — marginal improvement.

---

## Next Steps

1. **Benchmark COOL-CHIC on a single 256x256 H&E tile** at several lambda values to find the size/quality sweet spot. Compare directly against JXL and ORIGAMI. (Encode is slow — expect minutes per tile on CPU.)

2. **Quantify the chroma contribution to Delta E / Butteraugli** in current ORIGAMI runs to confirm the chroma hypothesis.

3. **If COOL-CHIC wins on R-D for single tiles:** evaluate decode latency for 16 tiles on CPU to see if it fits the cold-miss budget (200-500ms).
