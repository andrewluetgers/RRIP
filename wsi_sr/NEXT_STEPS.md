# WSI SR — Next Steps

## Status (2026-03-09)

### Stage 1: COMPLETE

**Extraction**: 798/800 slides, 1.4M tiles, 246 GiB in GCS. GCP VM deleted.
**Architecture comparison**: All 5 models trained on 5,180 tiles, 50 epochs each.

| Model | Params | PSNR | SSIM | Delta E | Residual KB | vs Lanczos |
|-------|--------|------|------|---------|-------------|------------|
| **WSISRX4Dual** | 18K | **30.52** | **0.860** | 2.54 | **125.3** | **+1.93 dB** |
| ESPCNR | 37K | 30.49 | 0.861 | **2.41** | 126.7 | +1.90 dB |
| WSISRX4 | 19K | 30.38 | 0.858 | 2.55 | 127.0 | +1.79 dB |
| WSISRX4Large | 295K | 29.67 | — | — | 127.0 | +1.08 dB |
| WSISRX4WideDeep | 112K | 29.57 | 0.855 | 2.63 | 127.2 | +0.98 dB |
| *Lanczos3 baseline* | — | 28.59 | 0.830 | 2.80 | 141.4 | — |

**Key findings:**
1. Dual-branch (18K) wins: best PSNR, smallest residuals, fewest outliers, smallest model
2. Larger models (WideDeep 112K, Large 295K) are WORSE on 5K tiles / 50 epochs — underfitting
3. All SR models beat lanczos3 by 1-2 dB and reduce residuals 10-11%
4. ONNX integration working in Rust server (sr_model.rs)

### Checkpoints Downloaded
All model checkpoints in `models/`:
- `dual_best.pt`, `wsisrx4_v2_best.pt`, `espcnr_best.pt`, `stage1_best.pt`
- `widedeep_best.pt`, `large_best.pt`
- ONNX exports: `model_sr.onnx`, `model_sr_int8.onnx`, `dual.onnx`, `dual_int8.onnx`, etc.

### Active Infrastructure

| Provider | Instance | Type | Purpose | Cost/hr |
|----------|----------|------|---------|---------|
| AWS | i-0c16c38153c56df36 | g5.xlarge (A10G) | Available (running duplicate Large) | $1.01 |
| RunPod | vmcy46xale4sbo | H100 NVL | Available (Large training done) | ~$3.00 |

**Total spend so far**: ~$25 (training) + ~$5 (extraction) = ~$30

---

## Stage 2 Training Plan

### Goal
Train on 1.4M tiles (280x more data) to see if:
1. The Dual model improves further (currently 30.52 PSNR, 125 KB residuals)
2. Larger models (WideDeep, Large) finally converge with enough data
3. We can push residual size below 110 KB (>22% reduction vs lanczos3)

### Phase 0: Prep (30 min, free)

- [ ] Kill duplicate Large run on AWS
- [ ] Verify Stage 2 data: `gsutil du -s gs://wsi-1-480715-tcga-tiles/stage2/`
- [ ] Fix train.py outdir bug (DONE — now uses `checkpoints/{run_id}/`)

### Phase 1: Quick Validation on A10G (4 hrs, ~$4)

Run on existing AWS A10G with Stage 1 data (5K tiles). Answer hyperparameter questions before committing to expensive Stage 2 runs.

**Experiment 1.1: Dual 200 epochs**
Does the winning model keep improving past 50 epochs, or has it saturated on 5K tiles?
```bash
python train.py --tiles $TILES --arch wsisrx4dual --epochs 200 \
    --batch 16 --lr 2e-4 --run-id dual_s1_200ep \
    --gcs-status s3://wsi-sr-training-results/stage1
```
Decision: If PSNR plateaus by epoch 80-100 → model saturated, more data needed. If still climbing → more epochs matter.

**Experiment 1.2: Dual + FFT loss**
Does frequency-domain loss shrink residuals?
```bash
python train.py --tiles $TILES --arch wsisrx4dual --epochs 50 \
    --batch 16 --lr 2e-4 --fft-weight 0.1 --run-id dual_fft_s1 \
    --gcs-status s3://wsi-sr-training-results/stage1
```
Decision: If residual KB drops → use FFT loss for Stage 2. If PSNR drops → skip FFT.

**Experiment 1.3: WideDeep lower LR**
Was the 112K model's poor Stage 1 result a learning rate issue?
```bash
python train.py --tiles $TILES --arch widedeep --epochs 100 \
    --batch 8 --lr 5e-5 --run-id widedeep_lr5e5_s1 \
    --gcs-status s3://wsi-sr-training-results/stage1
```
Decision: If WideDeep beats Dual → LR was the issue. If still worse → genuinely needs more data.

### Phase 2: Data Transfer to H100 (1-2 hrs, ~$18)

Transfer 246 GiB from GCS to RunPod H100 local storage.
```bash
# On RunPod H100
pip install google-cloud-storage
gsutil -m cp -r gs://wsi-1-480715-tcga-tiles/stage2/ /workspace/tiles/

# Generate manifest for fast loading
find /workspace/tiles -name "*.jpg" | sort > /workspace/tile_manifest.txt
```

Cost: ~$15 GCS egress + $3 compute while transferring.

### Phase 3: Primary Stage 2 Runs on H100 (14-21 hrs, ~$42-63)

Training parameters for 1.4M tiles:
- Batch=64 on H100 NVL
- ~19,700 batches/epoch → ~6.6 min/epoch
- Validation every 5 epochs

**Run 3.1: Dual on Stage 2 (PRIORITY 1)**
Most important run — does 280x more data push the small model further?
```bash
python train.py --tiles /workspace/tiles --arch wsisrx4dual \
    --epochs 50 --batch 64 --lr 2e-4 --workers 8 \
    --run-id dual_s2_50ep \
    --gcs-status s3://wsi-sr-training-results/stage2
```
Time: ~7 hrs. Decision point at epoch 20: if PSNR > 31 dB and climbing → data is helping.

**Run 3.2: WideDeep on Stage 2 (PRIORITY 2)**
Does the 112K model converge with enough data?
```bash
python train.py --tiles /workspace/tiles --arch widedeep \
    --epochs 50 --batch 64 --lr 1e-4 --workers 8 \
    --run-id widedeep_s2_50ep \
    --gcs-status s3://wsi-sr-training-results/stage2
```
Time: ~7 hrs. If WideDeep beats Dual → bigger model justified. If not → Dual is the right size.

**Run 3.3: Dual + FFT on Stage 2 (PRIORITY 3, only if Phase 1 shows FFT helps)**
```bash
python train.py --tiles /workspace/tiles --arch wsisrx4dual \
    --epochs 50 --batch 64 --lr 2e-4 --workers 8 \
    --fft-weight 0.1 --run-id dual_fft_s2_50ep \
    --gcs-status s3://wsi-sr-training-results/stage2
```

### Phase 4: Extended Training for Winner (8-16 hrs, ~$24-48)

Take the best model from Phase 3 and train to 200 epochs with hard mining:
```bash
python train.py --tiles /workspace/tiles --arch <winner> \
    --epochs 200 --batch 64 --lr 2e-4 --workers 8 \
    --hard-mining 2.0 \
    --resume checkpoints/<winner_run>/best.pt \
    --run-id <winner>_s2_200ep \
    --gcs-status s3://wsi-sr-training-results/stage2
```

### Phase 5: Export & Integration (1 hr, free)

```bash
# Export ONNX
python export.py --checkpoint checkpoints/<best>/best.pt \
    --output models/model_sr_v2.onnx --quantize --benchmark

# Test in Rust pipeline
origami encode --image evals/test-images/L0-1024.jpg --out /tmp/sr_v2 \
    --baseq 95 --l0q 50 --subsamp 444 --manifest --debug-images --pack \
    --sr-model models/model_sr_v2.onnx
```

### Success Targets

| Metric | Stage 1 Dual | Stage 2 Target | Stretch |
|--------|-------------|----------------|---------|
| Val PSNR | 30.52 dB | 31.5+ dB | 32.5+ dB |
| Residual KB | 125.3 | <110 | <100 |
| vs Lanczos3 | +1.93 dB | +2.5 dB | +3.5 dB |
| ONNX size | 69 KB | <500 KB | <100 KB |
| Decode latency | ~48ms/family | <50ms | <30ms |

### Cost Summary

| Phase | GPU | Hours | Cost |
|-------|-----|-------|------|
| Phase 0: Prep | — | 0.5 | $0 |
| Phase 1: Quick experiments | A10G | 4 | $4 |
| Phase 2: Data transfer | H100 (idle) | 1 | $18 |
| Phase 3: Primary runs (2-3) | H100 | 14-21 | $42-63 |
| Phase 4: Extended training | H100 | 8-16 | $24-48 |
| Phase 5: Export/eval | local | 1 | $0 |
| **Total** | | **28-44 hrs** | **$88-133** |

### What NOT to Do

- **Don't use attention layers** — blows the 50ms decode latency budget
- **Don't use GANs** — for a codec, pixel accuracy matters; hallucinations = incorrect reconstruction
- **Don't train Large (295K) on Stage 2** — if WideDeep (112K) doesn't benefit, Large won't either. Only try Large if WideDeep clearly wins.

---

## Cleanup Checklist (after all training done)

- [ ] Download all model checkpoints + eval results locally
- [ ] Stop/terminate AWS instance
- [ ] Stop/terminate RunPod pod
- [ ] Verify GCS data is preserved (keep tiles for future runs)
- [ ] Delete AWS key pair + security group
- [ ] Update model_cards.json with final results
- [ ] Cost audit: verify actual spend vs estimates
