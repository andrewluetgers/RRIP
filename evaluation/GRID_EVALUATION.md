# ORIGAMI Parameter Grid Evaluation

This document describes how to run the comprehensive parameter grid evaluation for ORIGAMI compression.

## Overview

The parameter grid evaluation tests all combinations of:
- **Quantization levels**: 16, 32, 64
- **JPEG qualities**: 30, 60, 90

This results in 9 total configurations to evaluate.

## Quick Start

### 1. Prerequisites

Ensure you have:
- Python environment with dependencies installed (via UV):
  ```bash
  uv sync
  ```
- A baseline pyramid already generated (e.g., `data/demo_out/baseline_pyramid`)

### 2. Run the Evaluation

Using the default demo_out pyramid:
```bash
python evaluation/run_parameter_grid.py
```

Or specify a custom pyramid:
```bash
python evaluation/run_parameter_grid.py \
  --input-pyramid data/my_slide/baseline_pyramid \
  --output-dir evaluation/my_results
```

### 3. Results

The script will generate:
- `aggregated_results.json` - Summary of all configurations
- `parameter_grid_analysis.png` - Visualization plots
- `grid_results/` - Individual results for each configuration
  - `quant16_jpeg30_result.json`
  - `quant16_jpeg60_result.json`
  - ... (9 total)

## What It Does

For each of the 9 parameter combinations:

1. **Generate Residuals**
   - Calls `wsi_residual_tool.py encode` with the specified JPEG quality
   - Creates residual pyramids in `grid_results/<config_name>/residuals_q<quality>/`

2. **Measure Compression**
   - Calculates file sizes for:
     - L2+ baseline tiles (retained levels)
     - L1 residuals
     - L0 residuals
   - Computes compression ratio vs JPEG Q90 baseline

3. **Measure Quality**
   - Samples 20 tiles from L0 (highest resolution)
   - Simulates reconstruction at residual quality
   - Calculates PSNR and SSIM vs original Q90 baseline

4. **Save Results**
   - Individual JSON file per configuration
   - Aggregated summary with optimal configurations

## Output Structure

```
evaluation/grid_evaluation/
├── aggregated_results.json          # Summary of all tests
├── parameter_grid_analysis.png      # Visualization plots
└── grid_results/
    ├── quant16_jpeg30/
    │   ├── residuals_q30/           # Generated residuals
    │   └── summary.json             # Compression stats from tool
    ├── quant16_jpeg30_result.json   # Full metrics
    ├── quant16_jpeg60/
    ├── quant16_jpeg60_result.json
    ... (9 configurations total)
```

## Understanding Results

### JSON Result Format

Each configuration result contains:

```json
{
  "config_name": "quant32_jpeg60",
  "quantization_levels": 32,
  "jpeg_quality": 60,
  "total_size_mb": 125.4,
  "compression_ratio_vs_q90": 2.15,
  "psnr_vs_q90": 42.3,
  "ssim_vs_q90": 0.985,
  "l0_residual_size_mb": 45.2,
  "l1_residual_size_mb": 12.8,
  "l2_baseline_size_mb": 67.4,
  "processing_time_seconds": 45.6,
  "num_tiles_tested": 20
}
```

### Aggregated Results

The `aggregated_results.json` includes:

- **Best Compression**: Configuration with highest compression ratio
- **Best PSNR**: Configuration with highest quality (PSNR)
- **Best SSIM**: Configuration with highest structural similarity
- **Best Balance**: Configuration with optimal quality/compression tradeoff

### Visualization Plots

The generated PNG includes 4 plots:

1. **Rate-Distortion Curve**: Compression ratio vs PSNR
2. **Storage Breakdown**: Stacked bar chart showing L0/L1/L2 sizes
3. **SSIM Comparison**: SSIM vs JPEG quality by quantization level
4. **Pareto Frontier**: Parameter space showing quality vs compression

## Interpreting Results

### Compression Ratio
- Higher is better (more compression)
- Example: 2.5x means the ORIGAMI version is 2.5× smaller than JPEG Q90

### PSNR (Peak Signal-to-Noise Ratio)
- Measured in dB, higher is better
- 40+ dB: Excellent quality, visually lossless
- 35-40 dB: Very good quality
- 30-35 dB: Good quality, minor artifacts
- <30 dB: Noticeable quality loss

### SSIM (Structural Similarity Index)
- Range: 0 to 1, higher is better
- 0.98+: Excellent perceptual quality
- 0.95-0.98: Very good quality
- 0.90-0.95: Good quality
- <0.90: Noticeable degradation

## Example Usage Scenarios

### Find Optimal for Storage
Look at the "Best Compression" configuration:
```bash
python evaluation/run_parameter_grid.py
# Check aggregated_results.json -> best_configurations.best_compression
```

### Find Optimal for Quality
Look at the "Best PSNR" or "Best SSIM" configuration:
```bash
# Same command, check best_psnr or best_ssim
```

### Find Balanced Configuration
Look at the "Best Balance" configuration (50/50 weighted):
```bash
# Same command, check best_balance
```

### Test on Your Own Slide
```bash
# First, generate baseline pyramid at Q90
python cli/wsi_residual_tool.py build \
  --slide /path/to/your_slide.svs \
  --out data/your_slide \
  --q 90

# Then run grid evaluation
python evaluation/run_parameter_grid.py \
  --input-pyramid data/your_slide/baseline_pyramid \
  --output-dir evaluation/your_slide_results
```

## Notes

### Quality Measurement Approximation
The current implementation uses a simplified quality measurement:
- Recompresses tiles at the residual JPEG quality
- Approximates the quality loss from residual encoding
- For exact measurements, implement full reconstruction pipeline

### Processing Time
- Each configuration takes 30-60 seconds depending on pyramid size
- Total runtime for 9 configurations: ~5-10 minutes
- Can be parallelized in future versions

### Disk Space
Each configuration generates residuals, requiring:
- Approximately same size as original pyramid per config
- ~1-2 GB per configuration for large whole-slide images
- Results cleaned automatically if re-run

## Customization

### Change Parameter Grid
Edit the script constants:
```python
QUANT_LEVELS = [16, 32, 64]  # Modify as needed
JPEG_QUALITIES = [30, 60, 90]  # Modify as needed
```

### Change Sample Size
Modify the quality measurement sample size:
```python
# In evaluate_configuration method
psnr, ssim = self.measure_quality_metrics(residuals_dir, num_samples=50)
```

### Add More Metrics
Extend the `ParameterResult` dataclass and measurement functions.

## Troubleshooting

### "Pyramid files not found"
Ensure the input pyramid exists with a `_files` directory:
```bash
ls data/demo_out/
# Should show: baseline_pyramid.dzi, baseline_pyramid_files/
```

### "Module not found" errors
Install dependencies:
```bash
uv sync
```

### Low PSNR/SSIM values
This is expected for low JPEG qualities (30). Check the rate-distortion
curve to find acceptable quality/compression tradeoffs.

### Script crashes during processing
Check available disk space and memory. Large pyramids may require
significant resources.

## Future Enhancements

Potential improvements:
- [ ] Parallel processing of configurations
- [ ] Full reconstruction pipeline for exact quality metrics
- [ ] Additional quality metrics (MS-SSIM, VMAF, ΔE00)
- [ ] Interactive visualization dashboard
- [ ] Comparison with other compression methods (WebP, AVIF, etc.)
