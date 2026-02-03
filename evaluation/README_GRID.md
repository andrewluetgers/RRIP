# Parameter Grid Evaluation - Quick Reference

## Files Created

1. **`run_parameter_grid.py`** - Main evaluation script
   - Tests all 9 parameter combinations (3 quant levels × 3 JPEG qualities)
   - Generates residuals, measures compression and quality
   - Outputs structured JSON results and visualization plots

2. **`test_single_config.py`** - Single configuration test
   - Quick test to verify the pipeline works
   - Tests one config (quant=32, jpeg_q=60)
   - Useful before running the full grid

3. **`GRID_EVALUATION.md`** - Comprehensive documentation
   - Detailed usage instructions
   - Output format documentation
   - Interpretation guide for metrics
   - Troubleshooting tips

## Quick Start

### Option 1: Test Single Configuration First
```bash
# Verify everything works with a single config test
python evaluation/test_single_config.py
```

### Option 2: Run Full Grid Evaluation
```bash
# Run all 9 configurations (~5-10 minutes)
python evaluation/run_parameter_grid.py
```

### Option 3: Custom Pyramid
```bash
# Use your own pyramid
python evaluation/run_parameter_grid.py \
  --input-pyramid data/my_slide/baseline_pyramid \
  --output-dir evaluation/my_results
```

## Parameter Grid

The script tests these 9 combinations:

| Quantization | JPEG Quality | Config Name       |
|--------------|--------------|-------------------|
| 16           | 30           | quant16_jpeg30    |
| 16           | 60           | quant16_jpeg60    |
| 16           | 90           | quant16_jpeg90    |
| 32           | 30           | quant32_jpeg30    |
| 32           | 60           | quant32_jpeg60    |
| 32           | 90           | quant32_jpeg90    |
| 64           | 30           | quant64_jpeg30    |
| 64           | 60           | quant64_jpeg60    |
| 64           | 90           | quant64_jpeg90    |

**Note**: Quantization parameter is not used in current implementation but prepared for future use.

## Expected Output

```
evaluation/grid_evaluation/
├── aggregated_results.json          # Summary with best configurations
├── parameter_grid_analysis.png      # 4-panel visualization
└── grid_results/
    ├── quant16_jpeg30/              # Residuals for each config
    ├── quant16_jpeg30_result.json   # Metrics for each config
    ... (9 configs total)
```

## Understanding Results

### Compression Ratio
- **Higher is better** (more compression)
- Example: 2.5x = ORIGAMI is 2.5× smaller than JPEG Q90

### PSNR (Peak Signal-to-Noise Ratio)
- Measured in dB, **higher is better**
- 40+ dB: Excellent (visually lossless)
- 35-40 dB: Very good
- 30-35 dB: Good
- <30 dB: Noticeable degradation

### SSIM (Structural Similarity)
- Range: 0 to 1, **higher is better**
- 0.98+: Excellent perceptual quality
- 0.95-0.98: Very good
- 0.90-0.95: Good
- <0.90: Noticeable degradation

## What to Look For

The `aggregated_results.json` identifies:

1. **Best Compression** - Highest compression ratio
   - Best if storage is primary concern

2. **Best PSNR** - Highest quality metric
   - Best if quality is primary concern

3. **Best SSIM** - Highest perceptual quality
   - Best if visual appearance matters most

4. **Best Balance** - Optimal quality/compression tradeoff
   - Best for general use (50/50 weighting)

## Typical Results

Based on the ORIGAMI algorithm, you should expect:

- **High JPEG Quality (90)**:
  - Compression: 1.5-2.5x
  - PSNR: 38-42 dB
  - SSIM: 0.96-0.99
  - Best for: High quality requirements

- **Medium JPEG Quality (60)**:
  - Compression: 2.5-4.0x
  - PSNR: 35-38 dB
  - SSIM: 0.93-0.96
  - Best for: Balanced use cases

- **Low JPEG Quality (30)**:
  - Compression: 4.0-6.0x
  - PSNR: 30-35 dB
  - SSIM: 0.88-0.93
  - Best for: Maximum compression

## Next Steps

1. **Run the evaluation**:
   ```bash
   python evaluation/run_parameter_grid.py
   ```

2. **Review results**:
   - Check `aggregated_results.json` for best configurations
   - View `parameter_grid_analysis.png` for visual comparison
   - Review individual config results in `grid_results/`

3. **Choose optimal config** based on your requirements:
   - Storage constrained? → Use best_compression
   - Quality critical? → Use best_psnr or best_ssim
   - General purpose? → Use best_balance

4. **Generate final residuals** with chosen parameters:
   ```bash
   python cli/wsi_residual_tool.py encode \
     --pyramid data/demo_out/baseline_pyramid \
     --out data/demo_out \
     --resq <chosen_jpeg_quality>
   ```

## Troubleshooting

### Script fails to import modules
```bash
# Ensure dependencies are installed
uv sync
```

### "Pyramid files not found"
```bash
# Check pyramid structure
ls data/demo_out/
# Should show: baseline_pyramid.dzi, baseline_pyramid_files/
```

### Low quality metrics
- Expected for low JPEG qualities (30)
- Check rate-distortion curve for acceptable tradeoffs
- Consider using higher JPEG quality (60 or 90)

## Advanced Usage

### Modify Parameter Grid

Edit `run_parameter_grid.py`:
```python
# Line ~94
QUANT_LEVELS = [16, 32, 64, 128]  # Add more levels
JPEG_QUALITIES = [30, 50, 70, 90]  # Add more qualities
```

### Change Sample Size

Edit `evaluate_configuration` method:
```python
# Line ~358
psnr, ssim = self.measure_quality_metrics(residuals_dir, num_samples=50)
```

### Run Parallel Configurations

Currently sequential, but can be parallelized:
```python
# Future: Use multiprocessing.Pool to run configs in parallel
```

## References

- Main documentation: `GRID_EVALUATION.md`
- ORIGAMI architecture: `../CLAUDE.md`
- Tool documentation: `../cli/wsi_residual_tool.py`
