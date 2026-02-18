# ORIGAMI WSI Evaluation - Downloaded Artifacts

**Date**: 2026-02-18  
**Dataset**: 3DHISTECH DICOM WSI (99 gigapixels)  
**Compression**: 4.58x (1159.5 MB → 253 MB)  
**Quality**: L0 PSNR 25.1 dB, SSIM 0.61

## Contents

### Documentation
- **`wsi_evaluation_report.md`** - Full evaluation report (13 KB)

### Metrics Data
- **`family_analysis.json`** - Family size distribution statistics
  - Grid: 262×362 = 94,844 families
  - Non-empty: 6,448 families
  - Size range: 0.4 KB to 135 KB (P50: 41 KB, P95: 96.5 KB)

- **`quality_metrics.json`** - PSNR/SSIM results
  - L0: 96 tiles, 25.13 dB PSNR, 0.61 SSIM
  - L1: 24 tiles, 23.37 dB PSNR, 0.55 SSIM

- **`summary.json`** - Encoding performance
  - Encoding time: 118.3 seconds
  - Throughput: 801.4 families/sec
  - GPU: NVIDIA B200

### Sample Tiles
- **`sample_tiles/decoded/`** - Reconstructed tiles from residual bundle
- **`sample_tiles/source/`** - Original DICOM tiles

**Tile coordinates**: 104_24, 104_25, 104_26 (family 26,6)

**Note**: Decoded tiles are larger (~36 KB) than source (~2.4 KB) because:
- Source: Highly compressed JPEG from DICOM
- Decoded: Re-encoded after residual reconstruction (quality 60)
- This is expected - residual encoding stores deltas, not raw tiles

## Key Findings

1. **4.6x compression** - Bundle stores residuals, not full tiles
2. **25 dB PSNR** - Good quality for digital pathology viewing
3. **801 families/sec** - GPU-accelerated encoding performance
4. **2.4x size variability** - Adaptive compression based on tissue density

## Implementation

Successfully added `--bundle` decode support to ORIGAMI CLI:

```bash
origami decode \
  --pyramid /path/to/baseline_pyramid \
  --bundle /path/to/residuals.bundle \
  --out /path/to/decoded_tiles \
  --tile 256
```

**Decoder performance**: ~130ms/family (CPU), potential for 5-10x speedup with GPU/parallel

## Files on RunPod

Full evaluation data remains on RunPod at:
```
/workspace/RRIP/evals/runs/wsi_for_family_eval/
├── residuals.bundle (253 MB)
├── baseline_pyramid_files/ (6,448 L2 tiles)
├── decoded_tiles_selected/ (120 reconstructed tiles)
└── dicom_source_tiles/ (120 original tiles)
```

## Next Steps

1. Integrate bundle decoder into production tile server
2. Implement GPU-accelerated reconstruction (<5ms/family)
3. Add warm/hot caching for frequently accessed families
4. Benchmark against JPEG 2000, JPEG XL for WSI compression
