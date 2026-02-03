# ORIGAMI Quality Tuning Guide

## Overview

ORIGAMI provides multiple quality control points throughout the compression pipeline. Understanding these parameters allows you to optimize for your specific use case - whether prioritizing maximum compression, highest quality, or fastest serving speed.

## Pipeline Stages and Quality Parameters

### Stage 1: Baseline Pyramid Generation
**Command:** `wsi_residual_tool.py build`

| Parameter | Default | Range | Impact | When to Adjust |
|-----------|---------|-------|---------|----------------|
| `--q` | 90 | 1-100 | Baseline JPEG quality for ALL pyramid levels | Initial setup only |
| `--tile` | 256 | Powers of 2 | Tile dimensions | Based on viewer requirements |

**Key Points:**
- This is a ONE-TIME operation per WSI
- Higher `--q` = larger files but better foundation for predictions
- L2+ tiles from this stage are kept unchanged in final output
- Changing requires complete rebuild from original WSI

**Recommended Settings:**
- Diagnostic critical: `--q 95` (highest quality foundation)
- Balanced: `--q 90` (default, good quality/size tradeoff)
- Archival: `--q 85` (acceptable quality, smaller files)

### Stage 2: Residual Encoding
**Command:** `wsi_residual_tool.py encode`

| Parameter | Default | Range | Impact | When to Adjust |
|-----------|---------|-------|---------|----------------|
| `--resq` | 32 | 1-100 | Luma residual JPEG quality | Per-dataset tuning |

**Key Points:**
- Can be re-run with different settings without rebuilding baseline
- Only affects L0/L1 compression (L2+ unchanged)
- Lower values = more compression but potential artifacts
- Sweet spot typically 20-50

**Quality/Size Tradeoffs:**

| resq | Pack Size | Compression vs JPEG-90 | Visual Quality | Use Case |
|------|-----------|------------------------|----------------|----------|
| 15 | ~28 MB | 9.5x | Acceptable | Maximum compression |
| 20 | ~33 MB | 8.2x | Good | Archival priority |
| 32 | ~43 MB | 6.7x | Very Good | **Balanced (default)** |
| 40 | ~52 MB | 5.5x | Excellent | Quality priority |
| 50 | ~65 MB | 4.4x | Near-perfect | Research/clinical |
| 70 | ~95 MB | 3.0x | Virtually lossless | Diagnostic critical |

### Stage 3: Pack File Compression
**Command:** `wsi_residual_tool.py pack`

**Current Implementation:**
- LZ4 compression (hardcoded)
- ~5.4x additional compression on residuals
- Optimized for fast decompression

**Potential Options (requires code modification):**
- LZ4 levels (0-16): Speed vs compression tradeoff
- Zstandard: Better compression, slower decompression
- No compression: Fastest serving, larger files

## Quality Impact Analysis

### What Each Parameter Controls

**Baseline Quality (`--q`)**
- **Affects:** Initial pyramid, L2+ tiles in final output
- **Visible in:** Zoomed-out views, base for predictions
- **Recommendation:** Set once at 90-95 and forget

**Residual Quality (`--resq`)**
- **Affects:** Fine detail recovery in L0/L1
- **Visible in:** Maximum zoom levels only
- **Recommendation:** Tune per dataset type

### Visual Quality Indicators

**Too Low `--resq` (< 20):**
- Blocking artifacts in detailed regions
- Loss of fine cellular structures
- Color banding in gradients

**Optimal `--resq` (25-40):**
- Imperceptible differences at normal viewing
- Preserved diagnostic features
- 6-8x compression vs JPEG-90

**Excessive `--resq` (> 50):**
- Diminishing returns on quality
- Larger files with minimal benefit
- Consider if really needed

## Hardcoded Quality Decisions

### Currently Non-Tunable (Require Code Changes)

1. **Interpolation Method**
   - Location: `cli/wsi_residual_tool.py:95`
   - Current: `BILINEAR`
   - Options: `NEAREST`, `BICUBIC`, `LANCZOS`
   - Impact: Prediction accuracy → residual size

2. **Color Space**
   - Location: `cli/wsi_residual_tool.py:58-73`
   - Current: BT.601 YCbCr
   - Options: BT.709, BT.2020
   - Impact: Color accuracy

3. **Chroma Handling**
   - Current: No chroma residuals (luma only)
   - Potential: Low-quality chroma residuals
   - Impact: Color fidelity vs size

4. **Residual Offset**
   - Location: `cli/wsi_residual_tool.py:115`
   - Current: Fixed +128 offset
   - Potential: Adaptive offset
   - Impact: Dynamic range utilization

## Tuning Workflow

### Step 1: Establish Baseline
```bash
# Build with high quality baseline
python cli/wsi_residual_tool.py build \
  --slide sample.svs \
  --out data/test \
  --q 90
```

### Step 2: Test Residual Qualities
```bash
# Test multiple resq values
for resq in 20 25 30 35 40; do
  python cli/wsi_residual_tool.py encode \
    --pyramid data/test/baseline_pyramid \
    --out data/test_q${resq} \
    --resq ${resq}
done
```

### Step 3: Compare Results
```bash
# Check sizes
du -sh data/test_q*/residuals_q*

# Visual comparison in viewer
# Launch server with different pack directories
```

### Step 4: Measure Performance
```bash
# Test serving speed for chosen quality
python comprehensive_perf_test.py
```

## Quality Recommendations by Use Case

### Maximum Compression (Archival)
```bash
--q 85  # Baseline
--resq 20  # Residuals
# Result: ~10x compression, acceptable quality
```

### Balanced (Default)
```bash
--q 90  # Baseline
--resq 32  # Residuals
# Result: 6.7x compression, very good quality
```

### Clinical/Diagnostic
```bash
--q 95  # Baseline
--resq 45  # Residuals
# Result: ~5x compression, excellent quality
```

### Research (Highest Quality)
```bash
--q 98  # Baseline
--resq 60  # Residuals
# Result: ~3.5x compression, near-lossless
```

## Advanced Tuning

### Per-Level Quality (Requires Code Modification)
```python
# Concept: Different quality for L1 vs L0
l0_quality = min(resq * 1.2, 100)  # Could use higher for L0 (diagnostic critical)
l1_quality = resq  # Standard for L1
```

### Content-Adaptive Quality
```python
# Concept: Detect tissue vs background
if is_tissue_region():
    use_quality = resq * 1.2
else:
    use_quality = resq * 0.8
```

### Progressive Quality
```python
# Concept: Higher quality for center tiles
distance_from_center = calculate_distance(x, y)
adaptive_quality = resq * (1.0 + 0.2 * (1 - distance_from_center))
```

## Validation Methods

### Objective Metrics
- **PSNR**: > 35 dB generally acceptable
- **SSIM**: > 0.95 for medical imaging
- **File size**: Track compression ratios

### Subjective Assessment
- Pathologist review at multiple zoom levels
- Side-by-side comparison with original
- Diagnostic feature preservation check

### Performance Metrics
- Tile generation time
- Cache efficiency
- Memory usage
- Decompression throughput

## Common Pitfalls

1. **Over-optimizing `--resq`**
   - Spending hours to save 5% file size
   - Diminishing returns above resq=40

2. **Forgetting baseline quality**
   - Can't improve beyond baseline `--q` setting
   - Requires complete rebuild to change

3. **Ignoring viewer requirements**
   - L0 (max zoom) is critical for diagnosis
   - Must maintain highest quality at L0
   - Never compromise on diagnostic levels

4. **Not testing with real data**
   - Different tissue types compress differently
   - Test with representative samples

## Future Improvements

### Potential Enhancements
1. CLI parameter for interpolation method
2. Per-level quality settings
3. Multiple compression options for packs
4. Adaptive quality based on content
5. Chroma residual options

### Research Directions
1. Machine learning for optimal quality prediction
2. Perceptual quality metrics specific to pathology
3. ROI-based quality allocation
4. Progressive transmission strategies

## Quick Reference

### Check Current Settings
```bash
# View residual quality from directory name
ls data/demo_out/residuals_q*

# Check baseline quality from summary
cat data/demo_out/summary.json | grep jpeg_q

# Verify pack compression
file data/demo_out/residual_packs_lz4/*.pack
```

### Typical Compression Results
| resq | vs JPEG-90 | vs Raw Pixels |
|------|-----------|---------------|
| 20 | 9.5x | ~285x |
| 32 | 6.7x | ~200x |
| 40 | 5.5x | ~165x |
| 50 | 4.4x | ~132x |

Remember: ORIGAMI's 6.7x compression is versus already-compressed JPEG-90, not raw pixels. This is exceptionally difficult to achieve while maintaining visual quality.