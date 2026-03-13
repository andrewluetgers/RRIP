# Domain-Trained Enhancement Model for ORIGAMI

## The Idea

Instead of storing per-tile residuals, train a small CNN once on thousands of WSI tiles to learn the mapping:

**blurry 4x-upsampled L2 prediction → high-quality L0 tile**

At decode time: just run the base tile through the model. No residual data needed per tile. The model weights ship once (a few hundred KB), then every tile decode is a single forward pass.

This is well-established in video coding as **Neural Network Post-Filters (NNPF)** — Nokia pioneered this for VVC/H.266, achieving ~8% bitrate reduction with a small model that runs after traditional decoding. For a constrained domain like H&E/IHC pathology, the gains could be much larger.

---

## Why This Could Work Especially Well for WSI

1. **Constrained domain.** H&E and IHC tiles have limited color palette, repetitive structures (nuclei, stroma, glands, background). A domain-specific model can learn these patterns far better than a general-purpose one.

2. **Predictable degradation.** The artifacts are always the same: lanczos3 upsample blur + JPEG/JXL quantization. The model only needs to learn one type of correction.

3. **Structured textures.** Histopathology has highly regular textures that are learnable — research confirms domain-specific SR models significantly outperform general models on pathology (JMI 2023).

4. **Chroma and luma together.** Unlike the current ORIGAMI residual (luma-only), a CNN can enhance both luma and chroma in one pass, directly addressing the chroma gap that drives Delta E.

---

## Candidate Models (Ranked by Speed)

All times estimated for 256x256 output on a server CPU (Xeon/EPYC with AVX2).

### Tier 1: Sub-millisecond (<1ms per tile)

| Model | Params | How It Works | Est. Speed | Quality | Open Source |
|-------|--------|-------------|------------|---------|-------------|
| **SR-LUT / HKLUT** | 0 at inference | Pre-computed lookup tables. No neural net at decode. | <0.5ms | Modest (+0.5-1.0 dB over bicubic) | [SR-LUT](https://openaccess.thecvf.com/content/CVPR2021/papers/Jo_Practical_Single-Image_Super-Resolution_Using_Look-Up_Table_CVPR_2021_paper.pdf), [HKLUT](https://arxiv.org/html/2312.06101v2) |
| **SESR-M5** | 73K | Reparameterizable: multi-branch at train time, plain 3x3 convs at inference. INT8 quantized. | <1ms | Good (35.2 dB DIV2K) | [GitHub ARM](https://github.com/ARM-software/sesr), [Qualcomm AI Hub](https://aihub.qualcomm.com/models/sesr_m5) |
| **FSRCNN-s** | ~8K | Tiny: 3 conv layers + deconv. Classic, proven. | <1ms | Baseline (+1-2 dB over bicubic) | Multiple PyTorch impls, OpenCV DNN |

### Tier 2: 1-5ms per tile

| Model | Params | How It Works | Est. Speed | Quality | Open Source |
|-------|--------|-------------|------------|---------|-------------|
| **ECBSR-M10C16** | ~50K | Edge-oriented conv blocks, reparameterizable. | 1-2ms | Good | [GitHub](https://github.com/xindongzhang/ECBSR) |
| **SPAN** | ~200K | Parameter-free attention. Won NTIRE 2024 ESR. | 2-5ms | Best quality/speed tradeoff (2024) | [GitHub](https://github.com/hongyuanyu/SPAN) |
| **SAFMN** | ~300K | Spatially-adaptive feature modulation. | 3-8ms | Very good | [GitHub](https://github.com/sunny2109/SAFMN) |

### Tier 3: 5-15ms per tile (still within budget)

| Model | Params | Notes |
|-------|--------|-------|
| **RFDN** | ~550K | Residual feature distillation. AIM 2020 winner. |
| **IMDN** | ~700K | Information multi-distillation. |

### Reference: NTIRE 2025 Efficient SR Challenge

43 entries, top-3 runtimes all under 10ms for x4 SR. All solutions available at [GitHub](https://github.com/Amazingren/NTIRE2025_ESR).

---

## Key Insight: Reparameterizable Architectures

Models like **SESR** and **ECBSR** use a trick: train with complex multi-branch topology (residual connections, batch norm, etc.) for better learning, then **mathematically collapse** everything into plain 3x3 convolutions at inference time. This means:

- Training: complex, powerful, slow — fine, we train once
- Inference: simple stack of 3x3 convs — trivially fast, easy to implement in Rust

A model with 73K params and only 3x3 convs can run in **<1ms on a server CPU** for a 256x256 tile. That's faster than the current JPEG decode step.

---

## How It Would Fit Into ORIGAMI

### Option A: Full Replacement of Residual

```
Current:  L2 base → 4x upsample → + residual JPEG → L0 tile
Proposed: L2 base → 4x upsample → enhancement CNN → L0 tile
```

- **No residual data per tile.** Total size = just the L2 base (~33KB for b95).
- Model weights shipped once (100KB-1MB depending on model size).
- Risk: if the model can't fully reconstruct fine detail, quality suffers with no fallback.

### Option B: Hybrid — Model Reduces Residual

```
Current:  L2 base → 4x upsample → + residual JPEG → L0 tile
Proposed: L2 base → 4x upsample → enhancement CNN → + small residual → L0 tile
```

- Model does most of the work, residual only encodes what the model misses.
- Could shrink residual by 50-80% while maintaining quality.
- Graceful fallback: if model prediction is good, residual is near-zero and compresses tiny.
- This is exactly how **Nokia's NNPF works for VVC** — ~8% bitrate reduction for general video, likely much more for constrained domains.

### Option C: Model Replaces Both Upsample + Residual

```
Current:  L2 base (256x256) → lanczos3 4x upsample → + residual → L0 (1024x1024)
Proposed: L2 base (256x256) → learned 4x super-resolution CNN → L0 (1024x1024)
```

- The model IS the upsampler. Learns to upsample better than lanczos3.
- Sub-pixel convolution (pixel shuffle) does the upscaling inside the network.
- Input is small (256x256), compute is proportionally less.
- This is the most natural fit for SR models which are designed for exactly this task.

---

## Training Strategy

### Data

- Input: L2 base tiles at chosen quality (e.g., JXL q95), upsampled 4x via lanczos3
- Target: Original L0 tiles from WSI
- Source: Thousands of WSI slides, millions of tile pairs
- Augmentation: random crops, flips, rotations (standard for SR)

### Loss Function Options

| Loss | What It Optimizes | Effect |
|------|------------------|--------|
| L1 / L2 (pixel) | PSNR | Sharp but can hallucinate |
| Perceptual (VGG) | Feature similarity | Better textures |
| Wasserstein / GAN | Perceptual realism | Crisp but may add false detail |
| **L1 + small perceptual** | **Balance** | **Recommended for pathology — don't hallucinate** |

For pathology: **avoid GAN losses**. They can hallucinate cellular structures that don't exist. Stick to L1 + mild perceptual loss.

### Stain-Specific Models

Could train separate models for H&E vs IHC vs special stains, or a single model with stain-type conditioning. H&E alone covers the vast majority of WSI.

---

## Rust Integration

For running the model at decode time in the ORIGAMI server:

| Runtime | Type | Speed | Dependencies |
|---------|------|-------|-------------|
| **ort** (pyke) | ONNX Runtime wrapper | Fastest | Links C++ ORT library |
| **rten** | Pure Rust ONNX | Good | No C++ deps |
| **burn** | Native Rust DL | Good | Can import ONNX |
| **tract** | Pure Rust ONNX | OK | Minimal deps |

For a model with only 3x3 convs + pixel shuffle + ReLU, even a hand-written Rust implementation would be straightforward — no need for a full ML framework. The operations are:
- 3x3 convolution (SIMD-friendly)
- ReLU (trivial)
- Pixel shuffle (memory reorder)

Could use the same NEON/AVX2 SIMD infrastructure already in `core/sharpen.rs`.

---

## Estimated Impact

### Size

| Scenario | L2 Base | Residual | Model | Total per Family |
|----------|---------|----------|-------|-----------------|
| Current ORIGAMI b95/l0q75 | 33KB | 70KB | — | 103KB |
| Option A (model only) | 33KB | 0 | shared | **33KB** |
| Option B (model + small residual) | 33KB | 15-35KB | shared | **48-68KB** |
| Option C (learned SR) | 33KB | 0 | shared | **33KB** |

### Quality (Speculative)

- Domain-specific SR on pathology should significantly outperform lanczos3 upsample, especially for chroma
- Nokia's general-purpose NNPF achieves ~8% bitrate savings; domain-specific should be 2-5x better
- The chroma gap (current main source of Delta E / Butteraugli degradation) would be directly addressed since the model outputs RGB

---

## Relevant Prior Work

### Neural Post-Filters for Video Coding (Nokia, 2024)
- Small CNN runs after VVC decode, ~8% bitrate reduction
- Standardized via VSEI in H.266
- [Nokia blog](https://www.nokia.com/blog/unlocking-next-generation-video-quality-with-neural-network-post-filtering/)
- [ACM paper](https://dl.acm.org/doi/10.1145/3638036.3640809)

### Pathology Image Compression with Pre-trained Autoencoders (MICCAI 2025)
- Repurposes Stable Diffusion / DC-AE autoencoders for WSI compression
- Fine-tunes with pathology-specific perceptual loss
- Outperforms JPEG-XL on downstream tasks
- [arXiv](https://arxiv.org/abs/2503.11591)

### CLERIC: Learned Compression for Digital Pathology (2025)
- Domain-specific learned codec with deformable residual blocks
- Outperforms SOTA learned compression on pathology
- Decode too slow for real-time (~2s per 512x512)
- [arXiv](https://arxiv.org/abs/2503.23862)

### Single-Patch Super-Resolution for WSI (JMI 2023)
- Compared DBPN, RCAN, and other SR networks on histopathology from TCGA
- Domain-specific training significantly improves quality vs general models
- Different networks perform best for different cancer subtypes
- [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9888549/)

### AIS 2024: Real-Time 4K SR of Compressed Images
- SR specifically designed to enhance compressed image artifacts
- Tested on AVIF-compressed inputs — directly analogous to enhancing JXL base tiles
- [arXiv](https://arxiv.org/abs/2404.16484)

---

## Recommended Next Steps

1. **Quick prototype with SESR-M5 or FSRCNN-s.** Train on 10k+ WSI tile pairs (L2 upsampled → L0 original). Measure PSNR/SSIM/Delta E/LPIPS on held-out tiles. This can be done in a few hours on a single GPU.

2. **Compare against current ORIGAMI** at matched file sizes. The model-only path (Option A) at 33KB should beat or match ORIGAMI at 103KB if the domain-specific model is good enough.

3. **If quality gap remains**, try Option B (model + tiny residual). The residual should be much smaller since the model handles the bulk of the correction.

4. **For Rust integration**, start with ONNX export + ort/rten. If latency matters, hand-implement the collapsed 3x3 conv stack with SIMD.

---

## References

- [SESR — ARM Research](https://github.com/ARM-software/sesr) | [MLSys 2022 Paper](https://proceedings.mlsys.org/paper_files/paper/2022/file/3134f61af2136e249b0d8f190cbdc508-Paper.pdf)
- [ECBSR](https://github.com/xindongzhang/ECBSR)
- [SPAN — NTIRE 2024 Winner](https://github.com/hongyuanyu/SPAN)
- [SAFMN](https://github.com/sunny2109/SAFMN)
- [NTIRE 2025 ESR Solutions](https://github.com/Amazingren/NTIRE2025_ESR)
- [Catid's Tiny SR](https://github.com/catid/upsampling) | [Blog](https://catid.io/posts/tiny_sr/)
- [SR-LUT (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Jo_Practical_Single-Image_Super-Resolution_Using_Look-Up_Table_CVPR_2021_paper.pdf)
- [HKLUT](https://arxiv.org/html/2312.06101v2)
- [FSRCNN](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)
- [ESPCN](https://arxiv.org/pdf/1609.05158)
- [Nokia NNPF](https://www.nokia.com/blog/unlocking-next-generation-video-quality-with-neural-network-post-filtering/)
