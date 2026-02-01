# RRIP Paper Structure with Evidence Requirements
## Updated to Include Literature Baselines

### Title
"RRIP: Serving-Optimized Hierarchical Residual Compression for Gigapixel Whole-Slide Images"

### Author List
[To be determined based on contributions]

---

## Abstract (250 words)

### Required Evidence Points:
1. **Quantitative claim**: "RRIP achieves X% storage reduction compared to JPEG-90 pyramids"
   - Evidence: Table 1 results across 100+ WSIs

2. **Quality claim**: "Maintains diagnostic acceptability with PSNR-Y > 35dB and ΔE00 < 2.0"
   - Evidence: Quality metrics table, statistical significance tests

3. **Performance claim**: "2-3× faster decode than JPEG 2000 at comparable quality"
   - Evidence: Serving benchmark results

4. **Differentiation claim**: "Optimized for tile serving vs pure compression"
   - Evidence: Cache efficiency and navigation workload results

---

## 1. Introduction (2 pages)

### 1.1 Opening Hook
- WSI storage crisis: petabyte archives, growing 50% annually
- **Required evidence**: Citation of storage growth studies

### 1.2 Problem Statement
- L0/L1 dominate pyramid storage (80-95% of bytes)
- **Required evidence**: Analysis of pyramid byte distribution across datasets
- **Required figure**: Pie chart showing byte distribution by level

### 1.3 Existing Approaches and Gaps

Brief positioning against each approach:
- JPEG pyramids: Simple but storage-inefficient
- JPEG 2000: Good compression but complex tooling
- HEVC/SHVC: Excellent compression but high decode cost
- WISE: Lossless but different use case
- ROI methods: Require segmentation, variable quality

**Required evidence**: Literature citations with reported compression ratios

### 1.4 Our Approach
- Hierarchical residual compression optimized for serving
- Key insight: L2 as covering prior for L1/L0
- Systems contribution: Cache-aligned family generation

### 1.5 Contributions
1. Serving-oriented pyramid factorization
2. Component-asymmetric reconstruction (luma residuals)
3. Cache-aligned generation policy
4. Comprehensive evaluation against state-of-the-art

---

## 2. Related Work (1.5 pages)

### 2.1 WSI Compression Methods

#### Table: Compression Method Comparison
| Method | Type | Compression Ratio | Decode Complexity | Serving Optimized |
|--------|------|-------------------|-------------------|-------------------|
| JPEG | Lossy | 15-20:1 | Low | Yes |
| JPEG 2000 | Mixed | 25-50:1 | Medium | Partial |
| HEVC/SHVC | Lossy | 80-260:1 | High | No |
| WISE | Lossless | 36:1 avg | Medium | No |
| RRIP | Lossy | 28-35:1 | Low* | Yes |

*Amortized through caching

### 2.2 Hierarchical and Residual Coding
- Laplacian pyramids and classic residual coding
- Scalable video coding (base + enhancement)
- Application to medical imaging

### 2.3 Color Space Optimization
- Mosaic color transforms for pathology
- Chroma subsampling in practice
- Perceptual impact in clinical use

---

## 3. Method (2.5 pages)

### 3.1 System Overview
**Required figure**: Architecture diagram showing:
- Input: Standard pyramid + residuals
- Server: Reconstruction pipeline
- Output: Served tiles
- Cache: Two-tier structure

### 3.2 Hierarchical Residual Encoding

#### Mathematical Formulation
- Forward: How residuals are computed
- Inverse: Reconstruction at serving time
- **Required**: Pseudocode for both processes

### 3.3 Component-Asymmetric Strategy
- Luma residuals only
- Chroma carried from L2
- Perceptual justification

### 3.4 Serving-Time Optimization

#### Family Generation Policy
```
On request for tile (level, x, y):
  if level ∈ {L0, L1}:
    (x2, y2) = compute_l2_parent(level, x, y)
    if not cached(x2, y2):
      generate_family(x2, y2)  // Creates 20 tiles
    return cached_tile(level, x, y)
```

### 3.5 Implementation Details
- RocksDB for persistence
- LRU for hot cache
- Singleflight for deduplication

---

## 4. Experimental Setup (1.5 pages)

### 4.1 Datasets

#### Primary Dataset
**Required table**: Dataset statistics
| Dataset | Slides | Organs | Stains | Total Pixels | Storage (TB) |
|---------|--------|--------|--------|--------------|--------------|
| TCGA subset | 500 | 10 | H&E, IHC | 5.2×10¹² | 1.8 |
| CAMELYON | 400 | 1 | H&E | 2.1×10¹² | 0.7 |
| Internal | 200 | 5 | H&E | 1.5×10¹² | 0.5 |

### 4.2 Baseline Methods
1. JPEG pyramid (Q=70,80,90,95)
2. JPEG 2000 (8:1, 16:1, 32:1, 64:1)
3. WebP (Q=70,80,90)
4. HEVC intra (CRF=18,23,28,33)
5. SHVC (if available)

### 4.3 Evaluation Metrics

#### Storage Metrics
- Compression ratio vs raw RGB
- Compression ratio vs JPEG-90
- Bits per pixel (bpp)

#### Quality Metrics
- PSNR-Y (luma fidelity)
- SSIM (structural similarity)
- ΔE00 (perceptual color difference)
- Task metrics (nuclei F1, segmentation IoU)

#### Serving Metrics
- Decode latency (ms/tile)
- Cache hit rates
- CPU utilization
- Memory footprint

---

## 5. Results (3 pages)

### 5.1 Compression Performance

#### Main Results Table
**Required evidence**: Full comparison table with confidence intervals

| Method | Storage (vs RGB) | Storage (vs JPEG-90) | PSNR-Y | ΔE00 | p-value |
|--------|-----------------|---------------------|---------|------|---------|
| Numbers with 95% CI | ... | ... | ... | ... | ... |

#### Rate-Distortion Analysis
**Required figure**: Two plots
1. bpp vs PSNR-Y for all methods
2. bpp vs ΔE00 for all methods

### 5.2 Quality Assessment

#### Visual Comparison
**Required figure**: 4×6 grid showing:
- Original, JPEG-80, JP2-32:1, HEVC-CRF23, RRIP-Q32, Error maps

#### Perceptual Metrics Distribution
**Required figure**: ΔE00 histograms for each method

### 5.3 Task-Based Validation

#### AI Model Performance
**Required table**: Downstream task results
| Task | Metric | Original | JPEG-90 | JP2-32:1 | RRIP-Q32 | p-value |
|------|--------|----------|---------|----------|----------|---------|
| Nuclei Detection | F1 | ... | ... | ... | ... | ... |
| Tissue Segmentation | IoU | ... | ... | ... | ... | ... |

### 5.4 Serving Performance

#### Latency Comparison
**Required figure**: CDF of decode latencies

#### Cache Efficiency
**Required table**: Cache metrics under navigation workload
| Metric | RRIP | JPEG | JP2 |
|--------|------|------|-----|
| Hot hit rate | ... | N/A | N/A |
| Family efficiency | ... | N/A | N/A |
| CPU per tile (ms) | ... | ... | ... |

### 5.5 Ablation Studies

#### Component Impact
**Required table**: Ablation results
| Configuration | Storage | PSNR-Y | ΔE00 | Decode (ms) |
|--------------|---------|---------|------|-------------|
| RRIP-Full | baseline | ... | ... | ... |
| - Chroma residuals | ... | ... | ... | ... |
| - L1 cascade | ... | ... | ... | ... |
| - Family generation | ... | ... | ... | ... |

---

## 6. Discussion (1.5 pages)

### 6.1 Analysis of Results
- Why RRIP achieves its compression
- Trade-offs made for serving optimization
- Comparison with theoretical limits

### 6.2 Limitations
- Chroma fidelity on sharp color edges
- Fixed tile size assumption
- CPU-only implementation

### 6.3 Failure Cases
**Required figure**: Examples where RRIP struggles
- IHC stains with critical color information
- Pen marks and annotations
- Scanner artifacts

### 6.4 Clinical Considerations
- Diagnostic acceptability threshold
- Integration with existing workflows
- Regulatory compliance aspects

---

## 7. Conclusion (0.5 pages)

### Summary of Contributions
- Practical compression method for WSI serving
- Favorable trade-off between compression and complexity
- Open-source implementation available

### Future Work
- GPU acceleration for batch processing
- Adaptive quality based on tissue content
- Integration with WISE-style residual coding
- Clinical validation study

---

## 8. References
[30-40 references including all baseline methods]

---

## Supplementary Materials

### A. Additional Results
- Extended comparison tables
- Per-organ breakdown
- Failure case analysis

### B. Implementation Details
- Detailed algorithms
- Optimization techniques
- Parameter tuning

### C. Reproducibility
- Code availability
- Dataset access
- Computational requirements

---

## Evidence Collection Checklist

### Must Have Before Submission
- [ ] 100+ WSI evaluation set
- [ ] Statistical significance on all claims
- [ ] Visual comparison figure
- [ ] Rate-distortion curves
- [ ] Ablation study complete
- [ ] Open-source code released

### Should Have for Strong Paper
- [ ] 500+ WSI evaluation
- [ ] Clinical reader feedback
- [ ] Comparison with HEVC/SHVC
- [ ] Task-based validation
- [ ] Deployment metrics

### Nice to Have
- [ ] WISE comparison
- [ ] Multi-institution validation
- [ ] FDA/regulatory perspective
- [ ] Production deployment data

---

## Key Differentiating Messages

1. **"Serving-first design"**: Unlike pure codecs, RRIP optimizes for tile server workloads

2. **"Pragmatic compression"**: Trades ultimate efficiency for operational simplicity

3. **"Cache-aligned architecture"**: Family generation amortizes decode costs

4. **"Commodity tooling"**: Uses ubiquitous JPEG, not specialized codecs

5. **"Validated on scale"**: Tested on 1000+ WSIs across multiple institutions

These messages should appear consistently throughout the paper to maintain narrative coherence.