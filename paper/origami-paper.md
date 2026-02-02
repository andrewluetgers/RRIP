Below is a **full draft paper** you can use as a starting point. It’s written **as if results exist**, but with a **very explicit caveat** up top that the numbers are placeholders and the claims are speculative until you run the full evaluation.

I also bake in the key framing you gave: **L0+L1 dominate storage**, so we keep **L2+ high quality** and reconstruct downward/upward as needed.

---

# ORIGAMI: Residual Reconstruction from Interpolated Priors for Efficient Whole-Slide Image Tile Serving

> **Important caveat (pre-print / speculative draft):**
> This manuscript is an *early, pre-implementation / pre-benchmark draft* written “as if” results are available. All quantitative values, comparisons, and conclusions should be treated as **illustrative placeholders** until validated on representative WSI datasets with a reproducible evaluation pipeline. The purpose of this document is to define the method clearly, identify what must be measured, and provide a complete paper structure suitable for a future real submission.

## Abstract

Whole-slide images (WSIs) routinely reach gigapixel scale, making storage and interactive viewing expensive. In common tiled pyramid formats, the highest-resolution levels dominate total bytes; in practice, the finest two pyramid levels (L0 and L1) account for the vast majority of stored data, while coarser levels contribute little. We present **ORIGAMI (Residual Reconstruction from Interpolated Priors)**, a serving-oriented compression approach that preserves a conventional high-quality pyramid for levels L2 and above, but encodes levels L1 and L0 as **residuals** relative to **interpolated priors** derived from L2. ORIGAMI reconstructs requested tiles on-demand using CPU-friendly operations (upsampling + residual add) and a cache policy aligned with viewer access patterns (generate and cache all descendants under the covering L2 tile). ORIGAMI further exploits perceptual redundancy by carrying chroma from the L2 prior and encoding residuals primarily in luma, yielding a simple, deployable trade-off between storage cost and visual fidelity. In preliminary experiments (illustrative), ORIGAMI achieves meaningful storage savings over a JPEG pyramid baseline at comparable perceptual quality, while requiring only commodity JPEG tooling and avoiding repeated lossy re-encoding of the original imagery.

## 1. Introduction

Whole-slide imaging has become foundational in digital pathology, powering clinical workflows, cohort discovery, AI model training, and retrospective analysis. However, WSIs are expensive to store and serve: a single slide can contain tens of thousands of 256×256 tiles at the highest resolution, and institutional archives can grow to petabyte scale. Interactive viewers further exacerbate cost by encouraging pyramid duplication across storage tiers (object stores, CDN caches, local SSD caches) to meet latency constraints.

A key structural observation motivates this work: for a typical 2× downsample pyramid, the number of tiles grows by ~4× per level toward higher resolution. Consequently, the finest levels dominate total bytes—often **L0 and L1 together comprise on the order of ~80–95%** of pyramid storage (dataset dependent). This suggests a compression strategy focused on the finest levels can achieve substantial savings while keeping the rest of the pyramid conventional and highly compatible.

We propose **ORIGAMI**, which treats **L2** as a “covering prior” for its descendants. Instead of storing L0 and L1 as independent JPEG tiles, ORIGAMI stores:

* a conventional pyramid for **L2 and above** (any standard codec),
* plus **residual tiles** for **L1 and L0**, computed against interpolated predictions derived from L2 (and L1 for L0 reconstruction).

At serving time, the tile server loads the L2 prior and residuals and reconstructs required tiles, caching generated outputs to amortize decode cost.

### Contributions (intended)

1. **A WSI-serving-oriented pyramid factorization**: keep standard L2+ tiles; encode only L1/L0 as residuals against interpolated priors.
2. **A component-asymmetric reconstruction policy**: carry chroma from the L2 prior (multi-scale chroma subsampling) and encode mainly luma residuals.
3. **A cache-aligned serving strategy**: generate all 4 L1 and 16 L0 descendants whenever any tile under a covering L2 is requested.
4. **An evaluation protocol blueprint** for quantifying storage/latency-quality trade-offs and comparing ORIGAMI to conventional pyramids and scalable codecs.

## 2. Related Work

### 2.1 Multi-resolution residual representations

ORIGAMI is conceptually related to classic multi-resolution methods such as Laplacian pyramids and residual pyramids, where a coarse image is refined by adding band-limited residual detail across scales. JPEG 2000 similarly supports multi-resolution decode via wavelet subbands and progressive refinement, and scalable video codecs (e.g., SHVC) perform base-layer upsampling followed by enhancement residual decoding.

ORIGAMI differs primarily in *where the method is implemented* (tile server / pyramid format rather than a monolithic codec), and in its *deployment constraints* (CPU-friendly reconstruction, existing JPEG pipelines, viewer-driven cache locality).

### 2.2 WSI compression practice

WSI ecosystems commonly use JPEG pyramids due to tooling simplicity and fast decode, while JPEG 2000 is used when multi-resolution streaming and region access are prioritized. HEVC-based approaches and scalable coding have been explored for pathology imagery and WSI streaming. WISE (CVPR 2025) proposes a lossless, WSI-specific pipeline combining hierarchical projection, bitplane/bitmap coding, and dictionary methods to achieve very high lossless compression ratios on benchmark datasets.

ORIGAMI’s target is distinct from WISE: ORIGAMI is primarily a **serving format** that may be lossy, prioritizing compatibility and decode speed, though its residuals could be further compressed by WISE-like bitplane/dictionary methods in future work.

## 3. Problem Setting and Pyramid Byte Dominance

Let a Deep Zoom style pyramid have levels `0..N`, where `N` is full resolution (L0) and each coarser level halves width and height. The tile count at level `k` is approximately proportional to `4^k` (ignoring boundary effects), so total bytes are dominated by the finest levels. In typical deployments we observe (illustrative) that:

* L0 and L1 occupy **the majority** of pyramid bytes (e.g., ~83–93%),
* L2+ occupy the remaining minority, even at high quality.

This motivates a hybrid: store L2+ conventionally (high quality, standard tooling), and focus compression innovations on L1/L0.

## 4. Method: ORIGAMI

### 4.1 Overview

ORIGAMI stores:

1. **Baseline tiles** for levels L2 and above: `T_L2+` (standard JPEG pyramid or equivalent)
2. **Residuals** for L1 and L0: `R_L1`, `R_L0`

Reconstruction uses interpolated priors:

* Predict L1 from L2:
  `P_L1 = Upsample(T_L2)` (split into 4 tiles)

* Reconstruct L1:
  `\hat{T}_L1 = f(P_L1, R_L1)`

* Predict L0 from reconstructed L1:
  `P_L0 = Upsample(\hat{T}_L1)` (split into 16 tiles)

* Reconstruct L0:
  `\hat{T}_L0 = f(P_L0, R_L0)`

In the simplest case, `f` is pixelwise addition:
`f(P, R) = clamp(P + R)`, where residuals are stored as an unsigned image with a bias (e.g., +128).

### 4.2 Interpolated priors

ORIGAMI uses interpolation (e.g., bilinear) to create priors. The intuition is that interpolation captures low-frequency structure and much of perceived content, leaving residuals with lower energy and entropy than the original tile.

ORIGAMI explicitly treats the “prior generation” filter as a design parameter (bilinear vs bicubic vs Lanczos vs edge-aware filters). In the current design we favor simple filters due to CPU cost.

### 4.3 Component-asymmetric coding (luma refinement, chroma carry)

ORIGAMI operates in Y′CbCr (or equivalently treats luma separately). We apply residuals primarily to luma:

* `\hat{Y} = clamp(Y_pred + rY)`
* chroma planes are inherited from the prior (e.g., upsampled Cb/Cr from L2)

This is analogous in spirit to chroma subsampling (4:2:0), but applied *across pyramid levels*: carrying chroma from L2 to L0 corresponds to chroma stored at 1/4 linear resolution (1/16 samples) relative to L0 luma.

### 4.4 Residual coding format

ORIGAMI deliberately uses simple, ubiquitous codecs for residual storage. In the baseline configuration:

* residual luma tiles are encoded as **grayscale JPEG** at quality `Q_resid`
* residual scaling/normalization can be applied to reduce entropy before encoding:

    * `rY_scaled = round(rY / s)` with a scale factor `s`
    * store scale metadata per tile or per family

The key property is that the server can decode residuals quickly on CPU and reconstruct tiles with minimal overhead.

### 4.5 Serving-time generation and caching policy

ORIGAMI assumes typical viewer locality: when a client requests a tile, neighboring tiles under the same covering L2 are likely to be requested soon (pan/zoom).

Therefore, when any descendant tile under a covering L2 `(x2, y2)` is requested, the server:

1. loads `T_L2(x2,y2)`
2. reconstructs all 4 L1 tiles and all 16 L0 tiles under that L2
3. caches the encoded JPEG bytes for these tiles:

    * hot in-memory LRU cache
    * warm persistent cache (RocksDB)

This can reduce compute per interactive session and amortize decoding.

## 5. Evaluation Protocol (what must be measured)

To publish ORIGAMI credibly, you need to quantify three axes:

### 5.1 Storage / compression

Report:

* bytes per tile (mean/median distribution) for each level
* total pyramid bytes for:

    * baseline JPEG pyramid (current practice)
    * ORIGAMI hybrid: L2+ baseline + residuals for L1/L0 (+ metadata)
* compression ratio vs baseline:

    * `CR = bytes_baseline / bytes_ORIGAMI`

Important: define baseline precisely (e.g., “existing JPEG tiles at quality 90 stored for all levels” or “scanner-native pyramid” etc.).

### 5.2 Fidelity / image quality

Since pathology is sensitive to small hue shifts and edge detail, use **multiple metrics**:

**Pixel-space / signal metrics**

* PSNR on luma (Y) for tile reconstructions
* SSIM / MS-SSIM (possibly on luma)

**Perceptual / color**

* ΔE (e.g., CIEDE2000) on reconstructed RGB tiles, with a focus on chroma policy impact

**Pathology-relevant downstream**

* Task-based metrics on AI models or classical CV tasks (if feasible):

    * nuclei detection F1
    * tissue segmentation IoU
    * stain vector consistency / stain normalization stability
      This is the strongest evidence that chroma carry doesn’t break pathology utility.

**Clinical-style (if possible)**

* small reader study or “diagnostic acceptability” proxy comparisons (even informal) for H&E.

### 5.3 Serving latency and system performance

Measure:

* tile server latency for:

    * cache hit (RAM)
    * cache hit (RocksDB)
    * cold miss generation (L2 decode + residual decode + recon + encode)
* CPU utilization per QPS at different concurrency
* cache efficiency under realistic pan/zoom traces:

    * “tiles served per family generated”
    * “family generation duplication rate” (should be near zero with singleflight)

## 6. Experimental Results (illustrative placeholders)

> **Placeholder results:** the following numeric values are *examples* reflecting what we would expect based on early prototypes and general codec behavior, not validated benchmarks.

### 6.1 Compression vs baseline JPEG pyramid

ORIGAMI’s savings come primarily from replacing L0/L1 stored JPEG tiles with:

* a single retained L2 covering tile (already in pyramid)
* and residual grayscale JPEGs that can be encoded aggressively (low Q) without obvious artifacts due to their lower energy.

Illustratively, ORIGAMI achieves:

* **~X× reduction** for L0+L1 storage at a residual JPEG quality in the ~30–35 range,
* leading to **~Y× total pyramid reduction** depending on how dominant L0/L1 are.

Because JPEG is already strong, ORIGAMI’s gains are meaningful but not “orders of magnitude” unless combined with ROI/background strategies or WISE-style lossless bitplane/dictionary compression.

### 6.2 Visual fidelity and chroma carry impact

With chroma carried from L2, observed effects are:

* minimal perceptual difference in most tissue regions,
* occasional chroma edge “softening” where high-frequency color transitions exist,
* luma edges largely preserved due to luma residuals.

Quantitatively (illustrative):

* luma PSNR remains high (e.g., > 35 dB median on tiles),
* ΔE remains within an acceptable range for typical H&E tiles, with outliers near sharp color boundaries.

### 6.3 Serving performance

In a tile server implementation with a “generate family on first touch” policy:

* cold miss cost is amortized over 20 tiles
* after the first tile in a region, most subsequent requests are cache hits

This aligns with interactive usage: a user zooms into a region and pans within it, repeatedly accessing siblings.

## 7. Discussion: why these decisions help, and trade-offs vs existing approaches

### 7.1 Why focus on L0/L1

The pyramid byte dominance means any method that reduces L0/L1 has leverage. Keeping L2+ standard preserves:

* compatibility with existing tooling
* fast overview browsing
* minimal changes to data pipelines

### 7.2 ORIGAMI vs JPEG 2000 / scalable video codecs

JPEG 2000 and scalable HEVC (SHVC) natively provide “base layer + enhancement” structure, but:

* are harder to integrate into typical WSI tile-serving stacks
* require specialized decoders and sometimes less friendly random-access patterns depending on configuration
* may be less straightforward to cache and precompute at the tile granularity used by web viewers

ORIGAMI’s strength is **deployment simplicity** and “tile server friendliness,” not ultimate rate-distortion optimality.

### 7.3 ORIGAMI vs WISE

WISE targets **lossless compression** with WSI-specific pre-processing and dictionary coding. ORIGAMI targets **lossy serving** with CPU cheap decode. However, ORIGAMI residuals may be well-suited to WISE-like bitplane/dictionary coding:

* residuals often have many leading zero bits
* structured local repetition across tissue regions

A hybrid ORIGAMI+WISE pipeline could potentially deliver larger storage reduction while preserving serving-time simplicity if the residual decode remains CPU efficient.

### 7.4 Failure modes

* Chromatic edge fidelity: carried chroma may blur sharp hue transitions.
* Stain variability: different stains may be less tolerant than H&E.
* Pathology tasks sensitive to color: immunohistochemistry or subtle hue cues may require chroma residual refinement.

Mitigations:

* optional chroma residual at L1 only (small overhead)
* ROI-aware chroma refinement (only where needed)
* adaptive residual allocation based on tile content

## 8. Conclusion

ORIGAMI proposes a practical middle ground between “store everything as JPEG tiles” and adopting heavyweight scalable codecs. By treating L2 as a covering prior, reconstructing L1 and L0 via interpolated priors and residuals, and carrying chroma across pyramid levels, ORIGAMI can reduce storage and duplication costs while preserving interactive serving performance. The method is compatible with commodity tooling and integrates naturally with tile server caching strategies. Future work includes rigorous dataset-scale evaluation, task-based validation for pathology, and exploring WISE-style residual entropy coding and ROI-aware refinement.

---

# What you still need to do to make this publishable (a concrete checklist)

## A) Benchmarks / datasets

* Multiple organs/species/scanners, and multiple stains (not just H&E)
* Include “hard tiles”: high cellularity, sharp boundaries, saturated color, pen marks, artifacts

## B) Baselines (must-have)

* JPEG pyramid baseline (your current)
* JPEG2000 baseline (tile/precinct settings disclosed)
* HEVC-based baseline (if feasible; even patch-level HEVC intra coding)
* ROI/background-aware strategy baseline (optional but persuasive)
* If feasible: cite/compare to WISE conceptually and/or run WISE if code is available

## C) Metrics (must-have)

* PSNR(Y), SSIM/MS-SSIM
* ΔE2000 (and show distribution tails, not just mean)
* Task-based: at least one downstream pathology model metric if you can (even segmentation)

## D) Serving metrics

* time-to-first-tile
* time-to-zoom (request L2 then L1 then L0)
* cache hit rates under recorded pan/zoom traces
* CPU cost per tile and per family generation

## E) Ablations (this is where reviewers nod)

* interpolation filter choice
* chroma policy (carry from L2 vs carry from L1 vs normal 4:2:0 vs chroma residual at L1 only)
* residual codec settings (JPEG Q, residual scaling factor)
* “generate family” policy vs on-demand per tile

---

If you want, I can convert this draft into **LaTeX structure** (IEEE/ACM style) and produce:

* a proper “Related Work” subsection with citations slotted in the right places,
* a figure plan (method diagram + cache policy diagram),
* and the exact experiments/ablations table you can execute to fill in the placeholders.
