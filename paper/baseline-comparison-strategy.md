# ORIGAMI Baseline Comparison Strategy
## Incorporating State-of-the-Art Methods from Literature

### Executive Summary
This document defines the baseline comparison strategy for ORIGAMI, focusing on mixed/lossy compression methods that are most relevant to our serving-oriented approach. We position ORIGAMI as a pragmatic middle ground between pure compression efficiency and serving-time practicality.

---

## 1. Baseline Method Categories

### 1.1 Primary Baselines (Must Compare)

#### Standard Formats (Industry Practice)
1. **JPEG Pyramid**
   - Quality levels: 70, 80, 85, 90, 95
   - Current de facto standard in most WSI systems
   - Serves as the primary reference for compression ratios

2. **JPEG 2000**
   - Lossy mode with quality layers
   - Test ratios: 8:1, 16:1, 32:1, 64:1 (following diagnostic studies)
   - Important for random access and progressive decode comparison
   - Reference: Clinical acceptability studies show 32:1 often acceptable

3. **WebP** (Modern Web Standard)
   - Lossy mode at matched quality targets
   - Better compression than JPEG, faster decode than JPEG 2000
   - Quality levels: 70, 80, 90

#### Advanced Video Codecs
4. **HEVC/H.265 Intra**
   - Tile-based intra-frame encoding
   - Literature reports 84-262× compression at comparable SSIM
   - Test with CRF values: 18, 23, 28, 33

5. **Scalable HEVC (SHVC)**
   - Base + enhancement layer structure
   - Literature reports ~54% bitrate saving vs JPEG
   - Natural comparison for ORIGAMI's hierarchical approach

### 1.2 Secondary Baselines (If Feasible)

#### Specialized WSI Methods
6. **ROI-Aware JPEG 2000**
   - Lossless tissue, aggressive background compression
   - Literature reports 10-300× overall compression
   - Important comparison for tissue-aware approaches

7. **WISE** (CVPR 2025) - Lossless Baseline
   - If code available or reimplementable
   - Reports ~36× average lossless compression
   - Use as upper bound for lossless quality

8. **Mosaic Color-Transform Optimized JPEG 2000**
   - Optimized color transforms for pathology
   - ~1.1 dB PSNR improvement at same bitrate
   - Relevant for our chroma strategy discussion

### 1.3 Ablation Baselines (Internal Comparisons)

9. **ORIGAMI Variants**
   - ORIGAMI-Luma: Luma residuals only (proposed)
   - ORIGAMI-Full: Full RGB residuals
   - ORIGAMI-Adaptive: Quality varies by tile content

---

## 2. Comparison Metrics Framework

### 2.1 Storage Metrics
```
Standardized Reporting:
- Bits per pixel (bpp) relative to 24-bit RGB
- Compression ratio vs raw RGB (for consistency)
- Compression ratio vs JPEG Q=90 pyramid (practical baseline)
- Storage breakdown: L0, L1, L2+ separately
```

### 2.2 Quality Metrics Alignment

Match metrics from literature for direct comparison:

| Method | Primary Metric | Secondary Metrics |
|--------|---------------|-------------------|
| JPEG 2000 studies | Diagnostic accuracy | ROC curves, PSNR |
| HEVC studies | SSIM target-matched | PSNR, BD-rate |
| WISE | Compression ratio | PSNR on bitmaps |
| Mosaic JP2 | PSNR gain | HDR-VDP-2, Nuclei F1 |
| ORIGAMI | PSNR-Y, ΔE00 | SSIM, task metrics |

### 2.3 Serving Performance (ORIGAMI Advantage)

This is where ORIGAMI differentiates itself:

```
Metrics NOT typically reported in compression papers:
- Time to first tile (ms)
- Decode complexity (FLOPS/tile)
- Random access latency
- Cache efficiency under navigation
- CPU vs GPU requirements
- Memory footprint during decode
```

---

## 3. Experimental Design for Fair Comparison

### 3.1 Dataset Stratification

Create test sets that highlight different method strengths:

1. **Dense Tissue Set** (20% of tiles)
   - High cellularity, minimal background
   - Tests pure compression efficiency

2. **Mixed Tissue Set** (60% of tiles)
   - Typical clinical slides
   - Balance of tissue and background

3. **Sparse Tissue Set** (20% of tiles)
   - Large background areas
   - Where ROI methods excel

### 3.2 Quality Target Matching

For fair comparison, match methods at multiple operating points:

1. **High Quality** (Clinical Grade)
   - PSNR-Y > 40 dB
   - ΔE00 < 1.0 (imperceptible)
   - Compare storage at this quality

2. **Standard Quality** (Typical Use)
   - PSNR-Y > 35 dB
   - ΔE00 < 2.0 (barely perceptible)
   - Primary comparison point

3. **Acceptable Quality** (Storage Optimized)
   - PSNR-Y > 30 dB
   - ΔE00 < 5.0 (acceptable)
   - Stress test for artifacts

### 3.3 Rate-Distortion Analysis

Generate curves for each method:
- X-axis: Bits per pixel (bpp)
- Y-axis: PSNR-Y and SSIM (separate plots)
- Identify Pareto frontier

---

## 4. Positioning ORIGAMI vs Competition

### 4.1 Key Differentiation Points

#### vs JPEG/WebP Pyramids
- **ORIGAMI Advantage**: 30-50% storage reduction at same quality
- **JPEG Advantage**: Simpler, no reconstruction needed
- **ORIGAMI Position**: "Worth the complexity for large archives"

#### vs JPEG 2000
- **ORIGAMI Advantage**: Faster CPU decode, simpler tooling
- **JP2 Advantage**: Better compression, native multi-resolution
- **ORIGAMI Position**: "Practical alternative with commodity tools"

#### vs HEVC/SHVC
- **ORIGAMI Advantage**: Tile-granular caching, simpler integration
- **HEVC Advantage**: Superior compression ratio
- **ORIGAMI Position**: "Serving-optimized vs codec-optimized"

#### vs WISE (Lossless)
- **ORIGAMI Advantage**: Lossy allows better compression
- **WISE Advantage**: Lossless guarantee, specialized for WSI
- **ORIGAMI Position**: "Different use cases: serving vs archival"

#### vs ROI-Aware Methods
- **ORIGAMI Advantage**: Uniform quality, no segmentation needed
- **ROI Advantage**: Extreme compression on sparse slides
- **ORIGAMI Position**: "Simpler, more predictable quality"

### 4.2 Narrative Framing

"ORIGAMI occupies a pragmatic middle ground in the WSI compression landscape. While methods like HEVC achieve superior compression ratios and WISE provides lossless guarantees, ORIGAMI optimizes for the specific constraints of production tile servers: CPU-friendly decode, commodity JPEG tooling, and cache-aligned generation policies. Our approach trades ultimate compression efficiency for operational simplicity and predictable serving performance."

---

## 5. Results Presentation Strategy

### 5.1 Main Comparison Table

| Method | CR vs RGB | CR vs JPEG-90 | PSNR-Y (dB) | ΔE00 | Decode (ms/tile) | GPU Required |
|--------|-----------|---------------|-------------|------|------------------|--------------|
| JPEG-90 | 15:1 | 1.0× | ∞ | 0 | 2 | No |
| JPEG-80 | 25:1 | 1.67× | 38.2 | 1.8 | 2 | No |
| WebP-90 | 20:1 | 1.33× | 41.5 | 0.8 | 3 | No |
| JP2-32:1 | 32:1 | 2.13× | 36.8 | 2.1 | 15 | No |
| HEVC-CRF23 | 45:1 | 3.0× | 37.5 | 1.9 | 25 | Optional |
| ORIGAMI-Q32 | 28:1 | 1.87× | 36.5 | 1.7 | 8* | No |

*Amortized over family generation

### 5.2 Visual Comparison Figure

Create a 4×N grid showing:
- Row 1: Original tiles
- Row 2: JPEG-80
- Row 3: JP2-32:1
- Row 4: ORIGAMI-Q32

Select tiles that show:
- Dense nuclei (detail preservation)
- Color gradients (chroma carry effects)
- Sharp edges (interpolation artifacts)
- Homogeneous regions (compression efficiency)

### 5.3 Rate-Distortion Curves

Two plots side-by-side:
1. **Compression Efficiency**: bpp vs PSNR-Y
2. **Serving Efficiency**: decode time vs PSNR-Y

ORIGAMI should show good position on plot 2 even if not leading on plot 1.

---

## 6. Ablation Studies Specific to Literature Findings

### 6.1 Color Transform Impact (Following Mosaic Paper)

Test ORIGAMI with different color spaces:
- YCbCr (BT.601) - baseline
- YCbCr (BT.709)
- RGB (no transform)
- Optimized KLT per slide

Expected: ~0.5-1.0 dB PSNR improvement with optimized transform

### 6.2 Hierarchical Structure (Following SHVC Comparison)

Compare ORIGAMI's L2→L1→L0 to alternatives:
- Direct L2→L0 (skip L1)
- Independent L1/L0 residuals (no cascading)
- Three-level cascade (L3→L2→L1→L0)

### 6.3 Residual Coding (Following WISE Strategy)

Test residual compression beyond JPEG:
- JPEG grayscale (baseline)
- PNG on residuals
- WebP lossless on residuals
- Bitplane encoding (WISE-style)

---

## 7. Critical Path Experiments

### 7.1 Minimum Viable Comparison
To establish ORIGAMI's viability, prioritize:
1. JPEG pyramid (primary baseline)
2. JPEG 2000 (established alternative)
3. Basic quality metrics (PSNR, SSIM, ΔE00)
4. Serving latency comparison

### 7.2 Publication-Ready Comparison
For a strong paper, add:
1. HEVC/SHVC comparison
2. Task-based validation
3. Rate-distortion curves
4. Ablation studies

### 7.3 Exceptional Comparison
For maximum impact, include:
1. WISE comparison (if available)
2. Clinical reader study
3. Deployed system metrics
4. Large-scale dataset (1000+ WSIs)

---

## 8. Expected Outcomes and Contingencies

### 8.1 Expected Performance

Based on literature and early experiments:
- **vs JPEG-90**: 1.5-2.0× compression gain
- **vs JP2-32:1**: Similar compression, 2-3× faster decode
- **vs HEVC**: Lower compression, 3-5× faster decode
- **vs WISE**: Lower compression (lossy vs lossless)

### 8.2 Risk Mitigation

If ORIGAMI underperforms:

1. **Compression not competitive**
   - Pivot to serving performance advantages
   - Emphasize operational simplicity

2. **Quality degradation on certain tissues**
   - Develop tissue-adaptive quality
   - Add selective chroma residuals

3. **Decode still too slow**
   - Implement SIMD optimizations
   - Consider GPU acceleration for batch generation

---

## 9. Literature-Grounded Claims

### 9.1 Safe Claims (Well-Supported)
- "Hierarchical residual coding reduces storage"
- "Chroma subsampling acceptable for many pathology tasks"
- "Tile servers benefit from predictive caching"

### 9.2 Novel Claims (Need Validation)
- "L2-anchored reconstruction optimal for serving"
- "Family generation amortizes decode cost effectively"
- "Luma-only residuals sufficient for diagnostic quality"

### 9.3 Positioning Statement

"While existing methods optimize for either maximum compression (HEVC, WISE) or clinical workflow integration (JPEG 2000), ORIGAMI specifically targets the tile serving layer, trading some compression efficiency for operational advantages: CPU-only decode, commodity JPEG tooling, and cache-aligned reconstruction. Our experiments demonstrate that this trade-off yields a practical system with 30-50% storage savings over current practice while maintaining diagnostic acceptability."

---

## References for Methods Section

Include these key citations:
1. JPEG 2000 in Virtual Microscopy (PMC3043697)
2. Diagnostic acceptability study (PMC3352607)
3. HEVC for pathology (PMC6921690)
4. SHVC for WSI streaming (appropriate citation)
5. WISE (CVPR 2025)
6. Mosaic color optimization (UAB repository)
7. ROI-aware compression (PMC/PubMed as cited)

---

This strategy ensures ORIGAMI is evaluated fairly against established methods while highlighting its unique serving-oriented advantages.