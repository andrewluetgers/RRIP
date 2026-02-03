# ORIGAMI Evaluation Setup with Lossless Source Images

## The Problem
Your current baseline pyramid uses JPEG tiles, making it invalid as "ground truth" for compression comparisons. JPEG artifacts compound with further compression, making it impossible to isolate ORIGAMI's actual performance.

## The Solution: Two-Track Evaluation

### Track 1: Algorithm Quality (Lossless Source)
Test ORIGAMI's compression algorithm in ideal conditions with uncompressed source material.

### Track 2: Real-World Performance (JPEG Source)
Test ORIGAMI's performance in production conditions where source is already JPEG-compressed.

## Recommended Lossless WSI Sources

### 1. CAMELYON16 Dataset (Best Option)
- **Format**: Standard TIFF pyramids
- **Content**: 399 breast cancer lymph node WSIs
- **Size**: ~700GB for training set
- **Resolution**: Up to 100,000×200,000 pixels
- **License**: Creative Commons CC0 (public domain)
- **Download**: https://camelyon16.grand-challenge.org/Data/

**Why it's ideal:**
- Well-documented, widely used benchmark
- TIFF format preserves pixel data
- Large enough for statistical validity
- Clinical relevance

### 2. OpenSlide Test Data
- **CMU-1 series**: Sample Aperio slides from Carnegie Mellon
- **Format**: SVS (TIFF-based)
- **Download**: http://openslide.cs.cmu.edu/download/openslide-testdata/
- **Includes**: Various compression types (some uncompressed layers)

**Extract uncompressed layers:**
```bash
# Use vips to extract highest resolution layer as uncompressed TIFF
vips dzsave CMU-1.svs output_folder --suffix .tif --tile-size 256 --overlap 0 --depth one --compression none
```

### 3. TCGA (The Cancer Genome Atlas)
- **Format**: SVS files (some with uncompressed layers)
- **Access**: https://portal.gdc.cancer.gov/
- **Note**: Requires data access approval for some datasets

### 4. Create Your Own Test Set
Convert existing JPEG tiles to a lossless baseline:

```python
import numpy as np
from PIL import Image
import os

def create_lossless_reference(jpeg_pyramid_path, output_path):
    """
    Load JPEG pyramid once, save as uncompressed TIFF.
    This becomes your 'ground truth' for comparisons.
    """
    # Read highest resolution JPEG tiles
    level_0_tiles = load_jpeg_tiles(jpeg_pyramid_path, level=0)

    # Stitch into full resolution image
    full_image = stitch_tiles(level_0_tiles)

    # Save as uncompressed BigTIFF
    Image.fromarray(full_image).save(
        output_path,
        format='TIFF',
        compression='none',
        save_all=True
    )

    return full_image
```

## Evaluation Protocol

### Step 1: Prepare Lossless Ground Truth

```python
# For CAMELYON16 or other TIFF sources
import openslide

def extract_test_regions(slide_path, num_regions=100):
    """Extract random 4096×4096 regions as ground truth"""
    slide = openslide.OpenSlide(slide_path)
    regions = []

    for i in range(num_regions):
        # Random position at highest resolution
        x = np.random.randint(0, slide.dimensions[0] - 4096)
        y = np.random.randint(0, slide.dimensions[1] - 4096)

        # Extract uncompressed pixels
        region = slide.read_region((x, y), 0, (4096, 4096))
        region = np.array(region.convert('RGB'))

        regions.append({
            'pixels': region,
            'position': (x, y),
            'slide': slide_path
        })

    return regions
```

### Step 2: Generate Test Pyramids

```python
def generate_test_pyramid(ground_truth_region, method):
    """
    Generate pyramid using different methods:
    - 'jpeg': Standard JPEG pyramid
    - 'jpeg2000': JPEG 2000 pyramid
    - 'origami': Your residual approach
    """
    if method == 'origami':
        # Your ORIGAMI implementation
        pyramid = generate_origami_pyramid(ground_truth_region)

    elif method == 'jpeg':
        # Standard JPEG pyramid at various qualities
        pyramid = generate_jpeg_pyramid(ground_truth_region, quality=90)

    elif method == 'jpeg2000':
        # JPEG 2000 with standard settings
        pyramid = generate_j2k_pyramid(ground_truth_region, rate=0.05)

    return pyramid
```

### Step 3: Measure Quality vs Compression

```python
def evaluate_compression(ground_truth, compressed_pyramid):
    """Compare reconstructed tiles to ground truth"""

    results = {
        'compression_ratio': len(ground_truth) / len(compressed_pyramid),
        'metrics': {}
    }

    # Reconstruct from compressed pyramid
    reconstructed = decompress_pyramid(compressed_pyramid)

    # Calculate metrics
    results['metrics']['psnr'] = calculate_psnr(ground_truth, reconstructed)
    results['metrics']['ms_ssim'] = calculate_ms_ssim(ground_truth, reconstructed)
    results['metrics']['delta_e'] = calculate_delta_e_2000(ground_truth, reconstructed)

    # Measure at tile boundaries (ORIGAMI weakness)
    results['metrics']['edge_delta_e'] = measure_edge_artifacts(ground_truth, reconstructed)

    return results
```

## Comparison Matrix

Create comparisons at matched quality levels:

| Method | Settings | File Size | PSNR | MS-SSIM | ΔE2000 | Notes |
|--------|----------|-----------|------|---------|--------|-------|
| **Uncompressed** | TIFF | 48MB | ∞ | 1.00 | 0 | Ground truth |
| **JPEG Q95** | Standard | 12MB | 45dB | 0.99 | 0.8 | Baseline |
| **JPEG Q90** | Standard | 8MB | 42dB | 0.98 | 1.2 | Common setting |
| **JPEG Q80** | Standard | 5MB | 38dB | 0.96 | 2.1 | Your current L2 |
| **JPEG 2000** | Rate 0.1 | 4.8MB | 40dB | 0.97 | 1.5 | Lossless option available |
| **HTJ2K** | Rate 0.1 | 4.8MB | 39dB | 0.96 | 1.6 | Fast decode |
| **ORIGAMI** | Q32 residuals | 2.4MB | 37dB | 0.95 | 2.3 | Your method |

## Key Metrics to Report

### For the Paper:

1. **BD-Rate** (Bjøntegaard Delta Rate)
   - Compare to JPEG baseline
   - "ORIGAMI achieves -45% BD-rate compared to JPEG"

2. **Quality at Fixed Compression**
   - At 20:1 compression: Compare PSNR/SSIM
   - "At 20:1, ORIGAMI maintains 37dB PSNR vs 35dB for JPEG"

3. **Compression at Fixed Quality**
   - At PSNR=38dB: Compare file sizes
   - "For 38dB quality, ORIGAMI uses 2.4MB vs JPEG's 5MB"

4. **Perceptual Quality**
   - ΔE2000 distribution (mean, 95th percentile)
   - "Mean ΔE of 2.3, with 95% of pixels < 4.0"

## Implementation Tools

### Required Libraries:
```bash
pip install openslide-python pillow numpy scikit-image opencv-python
pip install jpeg2000  # For J2K comparison
pip install piq  # For advanced metrics
```

### JPEG 2000 Encoding:
```python
import glymur

def encode_jpeg2000(image, quality_layers=[60, 40, 20]):
    """Encode as JPEG 2000 with multiple quality layers"""
    jp2 = glymur.Jp2k('temp.jp2')
    jp2[:] = image
    jp2.layer = quality_layers
    return jp2.read()
```

### Running the Evaluation:
```python
# Main evaluation script
def run_full_evaluation():
    # 1. Load lossless test images
    test_images = load_camelyon16_samples(n=100)

    # 2. Run each compression method
    methods = ['jpeg', 'jpeg2000', 'htj2k', 'origami']
    results = {}

    for method in methods:
        results[method] = []
        for img in test_images:
            compressed = compress_method(img, method)
            metrics = evaluate_compression(img, compressed)
            results[method].append(metrics)

    # 3. Generate R-D curves
    plot_rd_curves(results)

    # 4. Calculate BD-rates
    bd_rates = calculate_bd_rates(results, baseline='jpeg')

    # 5. Generate paper tables
    generate_latex_tables(results, bd_rates)
```

## Expected Outcomes

With clean, lossless source material:

**ORIGAMI Strengths:**
- 50-82% better compression than JPEG at similar quality
- Fast decode (0.35ms amortized per tile)
- Good PSNR/SSIM on uniform tissue regions

**ORIGAMI Weaknesses:**
- Higher ΔE at sharp edges (chroma interpolation)
- Slightly lower peak quality than lossless J2K
- Block artifacts at L2 family boundaries

## Two-Track Results Section for Paper

### Section 5.1: Algorithm Performance (Lossless Source)
"When evaluated on uncompressed TIFF sources from CAMELYON16, ORIGAMI achieves..."

### Section 5.2: Production Performance (JPEG Source)
"In production scenarios with JPEG-compressed pyramids, ORIGAMI demonstrates..."

This approach gives you defensible metrics while acknowledging real-world constraints.

## Next Steps

1. Download CAMELYON16 sample (even just 10 slides for testing)
2. Extract test regions as uncompressed TIFF
3. Run comparison suite
4. Generate R-D curves
5. Calculate BD-rates
6. Draft results section with both tracks

This gives you bulletproof evaluation methodology that matches what JPEG/J2K papers use.