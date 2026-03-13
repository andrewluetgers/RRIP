# Wide Dual Training Plan

## Motivation

End-to-end Rust benchmarks (decode quality + server throughput) identified what makes the Dual architecture win and where it falls short. This plan exploits those insights with targeted experiments.

### What the Dual Gets Right (confirmed by Rust eval pipeline)

1. **YCbCr decomposition** — the color transform is free; the model doesn't waste capacity learning it implicitly inside RGB filters
2. **Asymmetric capacity** — Y gets 5 blocks × 16ch, CbCr gets 2 blocks × 8ch. Human vision has ~4x more spatial resolution for luma than chroma, and the residual only corrects Y
3. **Fewer params = less overfitting** — 17,736 params (smallest model) yet best PSNR and smallest residuals

### Where the Dual Falls Short

| Metric | Dual vs Lanczos3 @Q70 | Analysis |
|--------|:-----:|---------|
| PSNR | -0.12 dB | Consistent across Q levels, not a training issue |
| SSIM | -0.0010 | Negligible |
| VIF | **+0.0009** | SR model wins — predicts correct structure |
| Delta E | **-0.00** | Ties or beats lanczos3 at every Q level |
| LPIPS | **-0.0047** | SR model looks more natural perceptually |
| Butteraugli | **+0.35** | Worst gap — SR shifts edges by subpixel amounts |
| DSSIM | **-0.000005** | Marginal win |
| SSIMULACRA2 | -0.11 | Marginal |

**The Butteraugli gap is the primary weakness.** Butteraugli specifically penalizes spatial displacement of edges and fine structure. Lanczos3 is a deterministic filter that never shifts edges; the learned convolutions introduce subpixel edge displacement that Butteraugli detects.

### Server Performance Constraints

From Rust benchmarks (Apple Silicon, pool=8, intra_threads=1):

| Model | Params | c=1 p50 | c=8 p50 | Status |
|-------|:------:|:-------:|:-------:|--------|
| Lanczos3 | — | 28ms | 51ms | Baseline |
| WSISRX4 | 19K | 85ms | 167ms | OK, within budget |
| **Dual** | **18K** | **90ms** | **153ms** | **OK, within budget** |
| ESPCNR | 37K | 123ms | 267ms | Marginal |
| WideDeep | 112K | ~300ms est | ~600ms est | **Too slow** |
| Large | 295K | ~700ms est | ~1400ms est | **Way too slow** |

**Budget: p50 < 150ms at c=1.** Models above ~40K params blow this on standard CPU. All experiments below stay within budget.

---

## Experiments

All experiments use Stage 1 data (5,180 tiles) for quick iteration. Winners advance to Stage 2 (1.4M tiles).

### Exp 1: Deeper Y Branch (targets Butteraugli)

Increase Y receptive field from 15×15 (5 blocks) to 19×19 (7 blocks). Larger context window means the model sees more structure around edges, reducing subpixel displacement.

```
Dual-Y7C1:  Y = 16ch / 7 blocks,  CbCr = 8ch / 1 block
            ~16K params, 62 KB collapsed
            Receptive field: 19×19 (Y), 7×7 (CbCr)
```

CbCr reduced to 1 block because our benchmarks show chroma prediction is already near-perfect (Delta E 1.42 at Q90 — essentially matching lanczos3). One block with 8 channels is plenty.

```bash
python train.py --tiles $TILES --arch wsisrx4dual \
    --y-blocks 7 --c-blocks 1 \
    --epochs 50 --batch 16 --lr 2e-4 \
    --run-id dual_y7c1_s1
```

**Success criteria:** Butteraugli improves (gap < +0.2 vs lanczos3) without degrading PSNR or Delta E.

### Exp 2: Wider Y Branch

More feature channels in Y branch to capture more structural patterns per layer. Same depth as current Dual, so inference speed is similar.

```
Dual-Wide:  Y = 24ch / 5 blocks,  CbCr = 8ch / 2 blocks
            ~28K params, 109 KB collapsed
            Receptive field: 15×15 (Y), 9×9 (CbCr)
```

```bash
python train.py --tiles $TILES --arch wsisrx4dual \
    --y-channels 24 --c-channels 8 \
    --epochs 50 --batch 16 --lr 2e-4 \
    --run-id dual_wide_s1
```

**Success criteria:** PSNR improves by >0.1 dB over current Dual (30.52 → 30.6+) without exceeding 150ms p50 at c=1 in Rust.

### Exp 3: Deep + Wide Y Branch

Combine Exp 1 and 2 — the maximum reasonable model within the latency budget.

```
Dual-DW:    Y = 24ch / 7 blocks,  CbCr = 8ch / 1 block
            ~32K params, 125 KB collapsed
            Receptive field: 19×19 (Y), 7×7 (CbCr)
```

```bash
python train.py --tiles $TILES --arch wsisrx4dual \
    --y-channels 24 --y-blocks 7 --c-channels 8 --c-blocks 1 \
    --epochs 50 --batch 16 --lr 2e-4 \
    --run-id dual_dw_s1
```

**Success criteria:** Best overall quality. Must stay under 150ms p50 in Rust benchmark.

### Exp 4: Separate Y/CbCr Loss Weighting

Currently the model uses a single RGB L1 loss. Since the architecture already separates Y and CbCr, we can apply different loss functions to each branch.

```python
loss_y = L1(y_pred, y_gt)                    # structural fidelity
loss_cbcr = L1(cbcr_pred, cbcr_gt)           # color accuracy
loss = loss_y + 0.5 * loss_cbcr              # weight Y 2x more than chroma
```

This directly optimizes what matters: Y quality drives PSNR/SSIM/residual size, CbCr quality drives Delta E. A single RGB loss dilutes Y gradients with chroma error that doesn't need as much correction.

```bash
python train.py --tiles $TILES --arch wsisrx4dual \
    --ycbcr-loss --cbcr-weight 0.5 \
    --epochs 50 --batch 16 --lr 2e-4 \
    --run-id dual_ycloss_s1
```

**Success criteria:** PSNR improves (better Y prediction) without Delta E regressing (chroma still trained).

### Exp 5: FFT + Edge Loss (targets Butteraugli)

Add frequency-domain loss and edge-aware penalty. FFT loss penalizes high-frequency discrepancies (edge detail). Edge loss penalizes spatial gradients of the prediction error (subpixel edge shifts).

```python
# FFT loss: penalize frequency-domain differences
fft_pred = torch.fft.rfft2(y_pred)
fft_gt = torch.fft.rfft2(y_gt)
fft_loss = F.l1_loss(fft_pred.abs(), fft_gt.abs())

# Edge loss: penalize spatial derivatives of the error
error = y_pred - y_gt
grad_x = torch.diff(error, dim=-1)
grad_y = torch.diff(error, dim=-2)
edge_loss = grad_x.abs().mean() + grad_y.abs().mean()

loss = L1_loss + 0.1 * fft_loss + 0.05 * edge_loss
```

```bash
python train.py --tiles $TILES --arch wsisrx4dual \
    --fft-weight 0.1 --edge-weight 0.05 \
    --epochs 50 --batch 16 --lr 2e-4 \
    --run-id dual_fft_edge_s1
```

**Success criteria:** Butteraugli gap drops below +0.2 (currently +0.35). PSNR should not degrade by more than 0.1 dB.

### Exp 6: Combined Best (after Exp 1-5 results)

Take the winning architecture variant + the winning loss combination and train together.

Expected best combo (hypothesis):
```
Architecture: Dual-Y7C1 (deeper Y, minimal chroma)
Loss: L1 + 0.1*FFT + 0.05*edge + separate Y/CbCr weighting
```

```bash
python train.py --tiles $TILES --arch wsisrx4dual \
    --y-blocks 7 --c-blocks 1 \
    --ycbcr-loss --cbcr-weight 0.5 \
    --fft-weight 0.1 --edge-weight 0.05 \
    --epochs 100 --batch 16 --lr 2e-4 \
    --run-id dual_combined_s1
```

---

## Implementation Requirements

### Model Architecture Changes (model.py)

The `WSISRX4Dual` constructor already accepts `y_channels`, `y_blocks`, `c_channels`, `c_blocks`. No architecture changes needed — just pass different values from `train.py`.

However, the checkpoint config currently stores only `channels` and `blocks` (not per-branch). Need to update:

```python
# Current config format
config = {"mode": "sr", "channels": 16, "blocks": 5}

# Required config format for dual variants
config = {
    "mode": "sr",
    "arch": "wsisrx4dual",   # NEW: architecture name
    "y_channels": 24,         # NEW: per-branch
    "y_blocks": 7,
    "c_channels": 8,
    "c_blocks": 1,
    "channels": 24,           # KEEP: backward compat (max of y/c)
    "blocks": 7,              # KEEP: backward compat (max of y/c)
}
```

The export script (`export.py`) already auto-detects Dual from state dict keys, so it will handle new variants without changes.

### Training Script Changes (train.py)

New CLI flags needed:

```
--arch               Architecture: wsisrx4, wsisrx4dual, espcnr (default: wsisrx4dual)
--y-channels N       Dual Y branch channels (default: 16)
--y-blocks N         Dual Y branch blocks (default: 5)
--c-channels N       Dual CbCr branch channels (default: 8)
--c-blocks N         Dual CbCr branch blocks (default: 2)
--ycbcr-loss         Use separate Y and CbCr losses instead of RGB L1
--cbcr-weight F      Weight for CbCr loss when using --ycbcr-loss (default: 0.5)
--fft-weight F       Weight for FFT frequency-domain loss (default: 0, disabled)
--edge-weight F      Weight for edge-aware gradient penalty (default: 0, disabled)
```

### Eval Pipeline

No changes needed — the Rust eval pipeline (`origami encode --sr-model`) and `compute_metrics.py` already handle any ONNX model with the standard input/output shape.

Workflow for each experiment:
```bash
# 1. Train
python train.py --tiles $TILES --arch wsisrx4dual ... --run-id <name>

# 2. Export ONNX
python export.py --checkpoint checkpoints/<name>/best.pt \
    --output models/<name>.onnx --quantize

# 3. Eval in Rust (multiple Q levels)
for q in 90 80 70 60 50; do
    origami encode --image evals/test-images/L0-1024.jpg \
        --out evals/runs/sr_<name>_l0q$q \
        --baseq 95 --l0q $q --subsamp 444 \
        --pack --manifest --debug-images \
        --sr-model models/<name>.onnx
done

# 4. Compute full visual metrics
uv run python evals/scripts/compute_metrics.py evals/runs/sr_<name>_l0q*

# 5. Server throughput benchmark (if architecture changed)
bash evals/scripts/bench_sr_model.sh --sr-model models/<name>.onnx
```

---

## Experiment Priority and Schedule

All Phase 1 experiments run on existing AWS A10G with Stage 1 data (5K tiles, ~20 min per 50-epoch run).

| Priority | Experiment | Time | What It Tests |
|:--------:|-----------|:----:|---------------|
| 1 | Exp 4: Separate Y/CbCr loss | 20 min | Free quality from better loss — no architecture change |
| 2 | Exp 5: FFT + edge loss | 20 min | Directly targets Butteraugli weakness |
| 3 | Exp 1: Dual-Y7C1 (deeper Y) | 20 min | Larger receptive field for edges |
| 4 | Exp 2: Dual-Wide (wider Y) | 20 min | More feature capacity |
| 5 | Exp 3: Dual-DW (deep + wide) | 25 min | Maximum model within latency budget |
| 6 | Exp 6: Combined best | 40 min | Winner architecture + winner loss |

**Total Phase 1 time: ~2.5 hours on A10G (~$2.50)**

Loss experiments (Exp 4, 5) run first because they're free — same model, same inference cost, potentially large quality gains. Architecture experiments (Exp 1-3) run second because they change inference cost and need Rust benchmarking to verify latency.

---

## Decision Tree

```
Exp 4 (Y/CbCr loss) → PSNR improved?
  YES → use Y/CbCr loss for all subsequent experiments
  NO  → stick with RGB L1

Exp 5 (FFT + edge) → Butteraugli improved?
  YES → use FFT+edge loss for Stage 2
  NO  → Butteraugli gap is structural, accept it

Exp 1 (Y7C1) → PSNR > 30.6 and Butteraugli < +0.2?
  YES → deeper Y is the answer
  NO  → depth doesn't help, try width

Exp 2 (Wide) → PSNR > 30.6?
  YES → wider Y helps
  NO  → current 16ch is capacity-optimal

Exp 3 (DW) → better than both Exp 1 and 2?
  YES → use DW for Stage 2 (if latency OK)
  NO  → use whichever of Exp 1/2 was better

Exp 6 (Combined) → advance to Stage 2
  Run on 1.4M tiles, 200 epochs, hard mining
  Target: PSNR > 31.5 dB, residual < 110 KB
```

---

## Relationship to Existing Plans

This plan **replaces** Phase 1 experiments in `NEXT_STEPS.md`:

| Original Experiment | Replacement | Rationale |
|---------------------|-------------|-----------|
| Dual 200 epochs | Keep (run after Exp 6 winner decided) | Still useful for saturation test |
| Dual + FFT loss | Exp 5 (FFT + edge loss) | Expanded: adds edge penalty targeting Butteraugli |
| WideDeep lr=5e-5 | **Dropped** | WideDeep (112K) too slow for latency budget. Wide Dual (28K) achieves the same test cheaply |

Phase 2-5 of `NEXT_STEPS.md` remain unchanged — the winner from this plan feeds into Stage 2 training on 1.4M tiles.

---

## Appendix: Param Count Estimates

```
Dual (current):     Y=16ch/5blk + CbCr=8ch/2blk = 17,736 params (69 KB)
Dual-Y7C1:          Y=16ch/7blk + CbCr=8ch/1blk = 16,552 params (65 KB)
Dual-Wide:          Y=24ch/5blk + CbCr=8ch/2blk = 28,200 params (110 KB)
Dual-DW:            Y=24ch/7blk + CbCr=8ch/1blk = 32,440 params (127 KB)

Reference:
WSISRX4:            16ch/5blk (RGB)              = 19,008 params (74 KB)
ESPCNR:             64→32→48 (3 conv)            = 37,200 params (145 KB)
WideDeep:           Y=32ch/10blk + CbCr=16ch/4blk = 111,648 params (436 KB)  ← TOO SLOW
```

All proposed variants are under 33K params and should stay within the 150ms p50 latency budget.
