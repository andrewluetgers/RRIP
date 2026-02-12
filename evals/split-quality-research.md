# Split-Quality Residuals: Balancing the Error Budget Across Pyramid Levels

## Summary

ORIGAMI's residual pyramid has an inherent asymmetry: L1 residuals (4 tiles) feed into L0 predictions (16 tiles). L1 encoding error propagates downstream, inflating L0 residuals. By investing more bits in L1 and fewer in L0, we can balance the error budget across levels, achieving **+1.5 dB better minimum PSNR at the same total byte count** compared to uniform quality settings.

## The Problem: L1 is Always the Bottleneck

In a uniform-quality ORIGAMI run, L1 reconstructed quality is always lower than L0:

| Flat Quality | L1 PSNR | L0 PSNR | Gap |
|-------------|---------|---------|-----|
| q=30 | 33.19 | 34.68 | -1.49 |
| q=40 | 33.80 | 35.44 | -1.64 |
| q=50 | 34.34 | 36.06 | -1.72 |
| q=60 | 34.88 | 36.63 | -1.75 |
| q=80 | 36.72 | 38.30 | -1.58 |

The gap is ~1.6 dB at every operating point. L1 is always the weakest link, and the overall system quality is bottlenecked by min(L1, L0).

## Why L1 Quality Matters More Per Byte

The reconstruction pipeline is:

```
L2 (baseline) -> upsample -> L1 prediction
L1 prediction + L1 residual -> L1 reconstructed
L1 reconstructed -> upsample -> L0 prediction
L0 prediction + L0 residual -> L0 reconstructed
```

L1 encoding error has two costs:
1. **Direct**: L1 reconstructed quality degrades
2. **Indirect**: L1 error propagates into L0 predictions, making L0 residuals larger

There are only 4 L1 tiles but 16 L0 tiles. Each byte spent improving L1 quality reduces residual magnitude across all 16 L0 tiles. This is the leverage: **L1 bytes have 4x the downstream impact of L0 bytes**.

## The Experiment

We swept L1 quality from L0+10 to L0+50 across L0 quality settings of 30, 35, 40, 45, and 50, producing 21 split-quality runs plus 12 uniform baselines.

### CLI Usage

```bash
# Split quality: L1 at q=60, L0 at q=40
python evals/scripts/wsi_residual_debug_with_manifest.py \
    --image evals/test-images/L0-1024.jpg \
    --l1q 60 --l0q 40 --pac

# Uniform quality (existing behavior, unchanged)
python evals/scripts/wsi_residual_debug_with_manifest.py \
    --image evals/test-images/L0-1024.jpg \
    --resq 40 --pac
```

When `--l1q` and/or `--l0q` are specified, they override `--resq` for the respective level. Output directories use the pattern `debug_l1q{N}_l0q{N}_pac`.

## Key Finding: Balanced Splits Gain +1.5 dB for Free

A "balanced" split is one where L1 PSNR ~ L0 PSNR (gap < 0.3 dB), eliminating the L1 bottleneck. The optimal L1 quality offset is approximately +30 over L0 quality.

### Balanced Splits vs Uniform at Same Byte Count

| Split Config | Total Bytes | Ratio | L1 PSNR | L0 PSNR | Flat min PSNR @ same size | Gain |
|-------------|-------------|-------|---------|---------|--------------------------|------|
| L1=60 L0=30 | 90,737 | 6.62x | 34.88 | 34.78 | 33.31 | **+1.47 dB** |
| L1=65 L0=35 | 99,182 | 6.06x | 35.23 | 35.20 | 33.66 | **+1.54 dB** |
| L1=70 L0=40 | 106,723 | 5.63x | 35.64 | 35.51 | 33.95 | **+1.56 dB** |
| L1=75 L0=45 | 115,750 | 5.19x | 36.08 | 35.83 | 34.28 | **+1.55 dB** |
| L1=80 L0=60 | 139,915 | 4.30x | 36.72 | 36.66 | 35.08 | **+1.58 dB** |

The gain is remarkably consistent: **+1.5 dB across all compression levels**. This is not a small effect — 1.5 dB is the difference between two full JPEG quality steps.

### How the Bytes Flow

The reason this is "free" is that better L1 quality reduces L0 residual sizes:

| Config | L1 Residuals | L0 Residuals | Total | vs Flat |
|--------|-------------|-------------|-------|---------|
| Flat q=40 | 11,113 | 50,965 | 102,569 | (baseline) |
| L1=60 L0=40 | 18,442 (+7,329) | 45,656 (-5,309) | 104,589 | +2.0% |
| L1=70 L0=40 | 24,442 (+13,329) | 41,790 (-9,175) | 106,723 | +4.0% |
| L1=80 L0=40 | 34,221 (+23,108) | 36,705 (-14,260) | 111,417 | +8.6% |

Spending 7K more on L1 residuals saves 5K on L0 residuals — the L1 investment partially pays for itself. The net cost is just 2K (2%) while L1 PSNR jumps from 33.80 to 34.88 (+1.08 dB).

## Practical Operating Points

### For maximum compression (matching Flat q=40 quality)

**Use L1=60 L0=30**: Achieves min PSNR = 34.78 (1 dB *better* than flat q=40's 33.80) at 90,737 bytes (11.5% *smaller* than flat q=40's 102,569 bytes).

### For high quality (matching Flat q=60 quality)

**Use L1=70 L0=40**: Achieves min PSNR = 35.51 (0.6 dB better than flat q=60's 34.88) at 106,723 bytes (20% smaller than flat q=60's 133,762 bytes).

### The +20 rule of thumb

For a moderate split with minimal risk: set L1 quality = L0 quality + 20. This consistently yields ~+1.0 dB on L1 for ~-0.05 dB on L0 — effectively free.

## Diminishing Returns

The L1/L0 gap shrinks as L1 quality increases beyond the balance point:

| Config | L1 PSNR | L0 PSNR | Gap | L1 "wasted" |
|--------|---------|---------|-----|-------------|
| Flat 40 | 33.80 | 35.44 | -1.64 | — |
| L1=60 L0=40 | 34.88 | 35.47 | -0.59 | 0 |
| L1=70 L0=40 | 35.64 | 35.51 | +0.13 | 0 |
| L1=80 L0=40 | 36.72 | 35.58 | +1.14 | ~1.14 dB |

Once L1 PSNR exceeds L0 PSNR, the extra L1 bytes no longer improve the system bottleneck. The 80/40 split has L1 quality 1.14 dB above L0 — those bytes would produce more quality improvement if moved to L0.

## Full Data: Splits vs Interpolated Uniform Baseline

Every split compared to a uniform run interpolated to the same total byte count. The `dL1` and `dL0` columns show the quality gain/loss vs uniform at the same size.

| Config | Total | Ratio | L1 dB | L0 dB | Unif L1 | Unif L0 | dL1 | dL0 |
|--------|-------|-------|-------|-------|---------|---------|-----|-----|
| Flat 30 | 87,836 | 6.84x | 33.19 | 34.68 | | | | |
| 40/30 | 88,415 | 6.80x | 33.80 | 34.71 | 33.21 | 34.71 | +0.59 | -0.00 |
| 50/30 | 89,404 | 6.72x | 34.34 | 34.74 | 33.26 | 34.76 | +1.08 | -0.02 |
| 60/30 | 90,737 | 6.62x | 34.88 | 34.78 | 33.31 | 34.84 | +1.57 | -0.06 |
| 70/30 | 93,430 | 6.43x | 35.64 | 34.87 | 33.42 | 34.98 | +2.22 | -0.11 |
| Flat 35 | 96,006 | 6.26x | 33.53 | 35.12 | | | | |
| 45/35 | 96,696 | 6.22x | 34.09 | 35.14 | 33.56 | 35.15 | +0.53 | -0.01 |
| 55/35 | 97,552 | 6.16x | 34.59 | 35.16 | 33.59 | 35.20 | +1.00 | -0.04 |
| 80/30 | 98,695 | 6.09x | 36.72 | 35.01 | 33.64 | 35.25 | +3.08 | -0.24 |
| 65/35 | 99,182 | 6.06x | 35.23 | 35.20 | 33.66 | 35.27 | +1.57 | -0.07 |
| 75/35 | 102,513 | 5.86x | 36.08 | 35.26 | 33.80 | 35.44 | +2.28 | -0.18 |
| Flat 40 | 102,569 | 5.86x | 33.80 | 35.44 | | | | |
| 50/40 | 103,520 | 5.81x | 34.34 | 35.46 | 33.83 | 35.48 | +0.51 | -0.02 |
| 60/40 | 104,589 | 5.75x | 34.88 | 35.47 | 33.87 | 35.52 | +1.01 | -0.05 |
| 70/40 | 106,723 | 5.63x | 35.64 | 35.51 | 33.95 | 35.61 | +1.69 | -0.10 |
| Flat 45 | 110,623 | 5.43x | 34.09 | 35.77 | | | | |
| 55/45 | 111,338 | 5.40x | 34.59 | 35.78 | 34.12 | 35.80 | +0.47 | -0.02 |
| 80/40 | 111,417 | 5.39x | 36.72 | 35.58 | 34.12 | 35.80 | +2.60 | -0.22 |
| 65/45 | 112,809 | 5.33x | 35.23 | 35.79 | 34.17 | 35.86 | +1.06 | -0.07 |
| 75/45 | 115,750 | 5.19x | 36.08 | 35.83 | 34.28 | 35.99 | +1.80 | -0.16 |
| Flat 50 | 117,408 | 5.12x | 34.34 | 36.06 | | | | |
| 80/45 | 118,561 | 5.07x | 36.72 | 35.86 | 34.38 | 36.10 | +2.34 | -0.24 |
| 60/50 | 118,587 | 5.07x | 34.88 | 36.06 | 34.38 | 36.10 | +0.50 | -0.04 |
| 70/50 | 120,587 | 4.98x | 35.64 | 36.08 | 34.45 | 36.18 | +1.19 | -0.10 |
| Flat 55 | 124,529 | 4.83x | 34.59 | 36.33 | | | | |
| 80/50 | 124,999 | 4.81x | 36.72 | 36.12 | 34.60 | 36.35 | +2.12 | -0.23 |
| Flat 60 | 133,762 | 4.49x | 34.88 | 36.63 | | | | |
| 80/60 | 139,915 | 4.30x | 36.72 | 36.66 | 35.08 | 36.83 | +1.64 | -0.17 |

## Test Conditions

- Input image: `evals/test-images/L0-1024.jpg` (1024x1024)
- Tile size: 256x256
- L2 baseline quality: 95
- Encoder: libjpeg-turbo (turbojpeg)
- Pyramid: 1 L2 tile, 4 L1 tiles, 16 L0 tiles
- Baseline total (all tiles at q=95): 601,042 bytes
