# Neural Codec Comparison for WSI Tile Compression

## Requirements

- **Quality target:** JPEG q80 equivalent (Delta E ~1.0, LPIPS ~0.015, Butteraugli ~1.7)
- **Size target:** 50-100KB per L0 family (16 tiles of 256x256 = 1024x1024)
- **Decode:** Fast CPU, no GPU needed
- **Encode:** GPU acceptable, encode once / serve forever
- **Domain:** H&E and IHC stained whole-slide images

---

## Approach Categories

### 1. Per-Image Overfitted Codecs

Train a tiny neural network from scratch for each image. Best compression ratio, slow encode, fast decode.

#### COOL-CHIC (Orange, ICCV 2023)

- **How it works:** Overfits a small coordinate-based neural decoder + hierarchical multi-resolution latent grid per image. Latents and tiny decoder weights are entropy-coded and transmitted. No traditional base codec — the neural net IS the entire codec.
- **Decode:** ~1728 MACs/pixel. 100ms for 1280x720 on CPU. For 256x256 tile: ~112M MACs, trivially fast (<5ms).
- **Encode:** 10k-100k gradient descent iterations per image. Expect 30-120 seconds per 256x256 tile on GPU (fast preset). 16 tiles per family = 8-32 minutes per L0 family.
- **Quality:** Matches VVC/H.266 with 30% less rate. State-of-the-art for low-complexity decoders.
- **Rate control:** Lambda parameter (1e-4 to 1e-2) controls rate-distortion tradeoff.
- **Presets:** fast_10k (10k iters), medium_30k, slow_100k, perceptive (Wasserstein distance tuning for subjective quality).
- **Open source:** [github.com/Orange-OpenSource/Cool-Chic](https://github.com/Orange-OpenSource/Cool-Chic) (Python/PyTorch, C decoder)
- **Venue:** ICCV 2023, continued development through 2025.

#### C3 (Google DeepMind, CVPR 2024)

- **How it works:** Per-image overfitted synthesis network + entropy model. Similar philosophy to COOL-CHIC with different architecture.
- **Decode:** <3k MACs/pixel (slightly heavier than COOL-CHIC).
- **Encode:** Per-image overfitting, similar cost to COOL-CHIC.
- **Quality:** Matches VTM (VVC reference) on CLIC2020.
- **Open source:** [github.com/google-deepmind/c3_neural_compression](https://github.com/google-deepmind/c3_neural_compression) (JAX)
- **Venue:** CVPR 2024.
- **Caveat:** JAX-based, harder to integrate than PyTorch. Tested on P100/V100.

### 2. Pre-Trained Amortized Codecs

Large pre-trained encoder+decoder. Fast encode (single forward pass), but decode requires a large neural network.

#### CompressAI (InterDigital)

- **Models:** bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor, cheng2020_attn. Quality levels 1-8.
- **Encode:** Single forward pass, <1 second per image.
- **Decode:** Single forward pass through large model (~10M+ parameters). Fast on GPU, slow on CPU.
- **Quality:** Beats JPEG and BPG, below VVC at most bitrates.
- **Open source:** [github.com/InterDigitalInc/CompressAI](https://github.com/InterDigitalInc/CompressAI) (PyTorch, pre-trained weights)
- **NOT SUITABLE:** Decode requires large neural net on GPU. Doesn't meet "fast CPU decode" requirement.

#### HiFiC (Google, NeurIPS 2020)

- **How it works:** Learned autoencoder + conditional GAN decoder for perceptually realistic reconstructions at very low bitrates.
- **NOT SUITABLE:** Large decoder, GPU required. End-to-end learned, not per-image.

#### JPEG AI (ISO/IEC 6048-1:2025)

- **How it works:** First international standard for end-to-end neural image coding. Autoencoder + hyperprior entropy model.
- **NOT SUITABLE:** Large decoder, standardized but not lightweight.

### 3. Hybrid Amortized + Refinement

#### SGA+ / Robustly Overfitting Latents (NeurIPS 2024)

- **How it works:** Pre-trained neural codec + per-image latent refinement via gradient descent. Closes the "amortization gap."
- **NOT SUITABLE:** Still needs the large pre-trained decoder at decode time.

#### Instance-Adaptive Compression (ICLR 2021)

- **How it works:** Fine-tune adapter modules of a pre-trained codec per image. Transmit small weight deltas.
- **NOT SUITABLE:** Large base decoder required.

### 4. Implicit Neural Representations (INR)

#### COIN / COIN++ (ICLR 2022)

- **How it works:** MLP maps (x,y) coordinates to RGB. MLP weights are quantized and entropy-coded as the compressed representation.
- **Decode:** Evaluate MLP at every pixel. Moderate complexity.
- **Quality:** Outperforms JPEG at low bitrates, below VVC.
- **Open source:** [github.com/EmilienDupont/coin](https://github.com/EmilienDupont/coin)
- **Less competitive** than COOL-CHIC / C3 on rate-distortion.

#### RECOMBINER (Cambridge, ICLR 2024)

- **How it works:** Bayesian INR with hierarchical priors, patch-based for high resolution.
- **Open source:** [github.com/cambridge-mlg/RECOMBINER](https://github.com/cambridge-mlg/RECOMBINER)
- **Less competitive** than COOL-CHIC / C3.

### 5. Traditional Hybrid Approaches

#### LCEVC (MPEG-5 Part 2, ISO 2021)

- **How it works:** Standardized enhancement layer for any base codec. Base at lower resolution + up to two traditional signal processing enhancement sub-layers.
- **Relevance:** Validates the layered base+residual architecture at standards/production level. ORIGAMI is conceptually similar but with neural potential.

#### Hybrid Learned Residual (Lee & Hang, CVPR-W 2020)

- **How it works:** VVC base + pre-trained neural residual autoencoder + hyperprior entropy coding + refinement CNN.
- **No open source.**
- **NOT SUITABLE:** Pre-trained residual encoder means GPU decode.

---

## Decision Matrix

| Approach | Decode Speed (CPU) | Compression | Encode Speed | Open Source | Fits Requirements? |
|----------|-------------------|-------------|--------------|-------------|-------------------|
| **COOL-CHIC** | ~5ms / 256x256 tile | Matches VVC | 30-120s/tile GPU | Yes (PyTorch + C) | **YES** |
| **C3** | ~10ms / tile | Matches VVC | Similar to COOL-CHIC | Yes (JAX) | Yes, but JAX |
| CompressAI | Slow (large model) | Below VVC | <1s | Yes | No — CPU decode too slow |
| HiFiC | Slow (large model) | Good perceptual | <1s | Demo only | No |
| COIN | Moderate | Below VVC | Minutes | Yes | Marginal |
| JPEG AI | Slow (large model) | Above VVC | Fast | Reference impl | No |
| SGA+ | Slow (large model) | Near VVC | Minutes | No | No |
| LCEVC | Fast (traditional) | Below VVC | Fast | Proprietary | N/A (traditional) |

---

## Recommendation

**COOL-CHIC is the clear first choice to benchmark.**

Reasons:
1. **Only codec with both tiny decoder AND state-of-the-art compression.** Everything else with competitive compression needs a large neural network at decode time.
2. **~5ms decode per 256x256 tile on CPU.** 16 tiles = ~80ms per family — well within the 200-500ms cold-miss budget.
3. **PyTorch encode + C decoder.** Easy to integrate: encode on GPU during WSI ingest, decode in Rust via C FFI or subprocess.
4. **Active development** with perceptive tuning mode (Wasserstein distance) which may be relevant for pathology.
5. **Encode cost is acceptable.** WSI are encoded once during ingest. Even at 30 minutes per L0 family, a full WSI with ~1000 families would take ~20 hours on a single GPU — parallelizable across multiple GPUs.

**C3 is the backup** if COOL-CHIC doesn't hit the quality/size targets, but JAX makes integration harder.

**CompressAI and all other amortized codecs are ruled out** — they all require large neural networks at decode time, making CPU decode too slow.

---

## Next Steps

1. **Benchmark COOL-CHIC on a 256x256 H&E tile** at lambda values spanning the 3-7 KB/tile range (50-100KB / 16 tiles). Compare PSNR, SSIM, Delta E, LPIPS, Butteraugli against JXL and ORIGAMI at the same file sizes.

2. **If COOL-CHIC wins:** evaluate the C decoder for Rust integration (FFI or subprocess), and test encode throughput on GPU for a full WSI.

3. **If COOL-CHIC doesn't win:** the remaining option is to fix ORIGAMI's chroma gap (add chroma residual or increase L2 quality) rather than switching codecs.

---

## References

- [COOL-CHIC — GitHub](https://github.com/Orange-OpenSource/Cool-Chic)
- [COOL-CHIC — ICCV 2023](https://arxiv.org/abs/2212.05458)
- [C3 — GitHub](https://github.com/google-deepmind/c3_neural_compression)
- [C3 — CVPR 2024](https://arxiv.org/abs/2312.02753)
- [CompressAI — GitHub](https://github.com/InterDigitalInc/CompressAI)
- [COIN — GitHub](https://github.com/EmilienDupont/coin)
- [RECOMBINER — GitHub](https://github.com/cambridge-mlg/RECOMBINER)
- [DLPR — GitHub](https://github.com/BYchao100/Deep-Lossy-Plus-Residual-Coding)
- [HiFiC — Project Page](https://hific.github.io/)
- [LCEVC — How It Works](https://www.lcevc.org/how-lcevc-works/)
- [JPEG AI — ISO Standard](https://www.iso.org/standard/88911.html)
