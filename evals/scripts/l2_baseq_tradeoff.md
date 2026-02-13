# L2 Baseline Quality (baseq) Tradeoff Analysis

## Context

When ORIGAMI bakes in expensive encode-time pixel optimizations (OptL2, chroma
pre-compensation), the L2 tile is the bottleneck that all downstream predictions
flow through. JPEG compression of L2 at `--baseq` discards some of those
optimizations. Higher baseq preserves more of the optimization investment but
increases the L2 file size.

This analysis quantifies the tradeoff: how much quality improvement do we get
per additional byte of L2 storage?

## Test Setup

- Source: 1024x1024 test image
- Encoder: turbojpeg, 4:4:4, no OptL2
- Residual quality: Q40 for L1 and L0
- L2 tile: 256x256
- Pipeline: float32 YCbCr, RGB-space bilinear upsample, float-precision residuals

## Results

```
baseq   L2 bytes    L1 res    L0 res     Total   DeltaE     MSE   PSNR   SSIM   LPIPS
-----  ---------  --------  --------  --------  -------  ------  -----  ------  ------
   95     52,875    11,372    51,483   115,730   2.4168   22.57  35.77  0.9311  0.0916
   96     59,488    11,226    51,292   122,006   2.3088   21.84  35.78  0.9313  0.0903
   97     66,853    11,167    51,150   129,170   2.2015   21.19  35.79  0.9314  0.0903
   98     76,870    11,100    51,014   138,984   2.0899   20.61  35.80  0.9315  0.0907
   99     95,931    11,088    50,963   157,982   1.9937   20.18  35.80  0.9317  0.0901
```

## Marginal Cost-Benefit

```
Step       + Bytes   DeltaE drop   Bytes/0.01 dE   Cumul bytes   Cumul dE drop
--------  --------  -----------  ---------------  -----------  ---------------
Q95->Q96     6,276       0.1080            581        6,276          0.108
Q96->Q97     7,164       0.1073            668       13,440          0.215
Q97->Q98     9,814       0.1116            879       23,254          0.327
Q98->Q99    18,998       0.0962           1975       42,252          0.423
```

## Key Observations

1. **Diminishing returns above Q97.** Each quality step costs progressively more
   bytes for roughly the same Delta E improvement (~0.1 per step), until Q99
   where the cost nearly doubles for *less* improvement.

2. **Downstream residuals barely shrink.** Better L2 predictions reduce L1/L0
   residual sizes by only 1-2% across the entire Q95-Q99 range. The L1+L0
   residuals drop from 62,855 to 62,051 bytes — a savings of just 804 bytes
   while L2 grows by 43,056 bytes.

3. **Quality improvement is real but comes from prediction accuracy**, not
   residual compression. The better L2 means better L1/L0 chroma predictions
   (since chroma has no residual), which directly lowers Delta E.

4. **The sweet spot is Q97.** It provides 51% of Q99's quality improvement
   (0.215 vs 0.423 Delta E reduction) at only 32% of the byte cost (13.4KB vs
   42.3KB). Cost per 0.01 Delta E is 668 bytes vs 1975 at Q99.

5. **Lossless L2 residual is not worth it.** A separate lossless WebP residual
   achieves near-perfect L2 reconstruction (Delta E 1.93) but costs 86KB — more
   than doubling L2 storage for only 0.06 Delta E improvement over Q99.

## Recommendation

- **Default: `--baseq 97`** for production use with OptL2. Best ROI.
- **`--baseq 95`** for minimum file size when L2 quality is less critical.
- **`--baseq 99`** when maximum quality is needed regardless of size.
- **L2 lossless residual (`--l2resq`)**: available but rarely justified. Only
  useful when exact L2 pixel values matter for downstream processing.

## Interaction with OptL2

When OptL2 gradient descent is enabled, higher baseq becomes *more* important:
the optimization specifically tailors L2 pixels for better bilinear-upsampled
predictions. JPEG quantization at L2 partially undoes this work. At Q95 the
mean L2 compression error is ~2.1 per channel; at Q97 it drops to ~1.3; at Q99
it's ~0.75. Each 0.5 reduction in L2 error compounds through all 20 downstream
tile predictions.

## Reproducing

```bash
# Run the sweep (each takes ~0.2s)
for q in 95 96 97 98 99; do
  origami encode \
    --image evals/test-images/L0-1024.jpg \
    --out evals/runs/rs_debug_j40_bq${q}_pac \
    --resq 40 --baseq $q --subsamp 444 --l2resq 0 \
    --manifest --debug-images
done

# Compute visual metrics
uv run python evals/scripts/compute_metrics.py evals/runs/rs_debug_j40_bq9*_pac
```
