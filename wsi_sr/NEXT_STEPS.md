# WSI SR — Next Steps

## Status (after Stage 1)

**Stage 1 complete**: 5 cancer types, 35 slides, 5,180 tiles, 50 epochs on AWS A10G.

| Metric | Bilinear | Bicubic | WSISRX4 (ours) |
|--------|----------|---------|----------------|
| Val PSNR | 27.02 dB | 28.34 dB | **30.38 dB** |
| Val SSIM | 0.7759 | 0.8224 | **0.8583** |
| Val Delta E | 3.30 | 2.87 | **2.55** |
| Residual KB | 150.5 | 142.5 | **127.0** |

**Missing**: Lanczos3 baseline (what ORIGAMI actually uses). The model must beat lanczos3, not just bicubic.

## Immediate Tasks

### 1. Add Lanczos3 Baseline
- **Why**: ORIGAMI uses lanczos3 for upsampling. Beating bicubic is meaningless if we don't beat lanczos3.
- **Where**: `train.py` baseline computation — add `F.interpolate(mode="bicubic")` approximation or use PIL Lanczos resize
- **Note**: PyTorch doesn't have a native lanczos3 interpolate mode. Use PIL/torchvision for the baseline, same as the dataset's `_simulate_base()`.
- **Expected**: Lanczos3 is ~0.5-1 dB better than bicubic. Our model at 30.38 dB should still win.

### 2. Shut Down Idle VMs
- **GCP `tcga-extract-s1`** in us-central1-c: extraction done, burning $0.14/hr
  ```bash
  gcloud compute instances delete tcga-extract-s1 --zone=us-central1-c --project=wsi-1-480715 --quiet
  ```
- **RunPod `origami-b200-sr`**: been provisioning for 1+ hour, not coming up
  ```bash
  curl -s -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    --data '{"query": "mutation { podTerminate(input: { podId: \"fsynrbrauorhvs\" }) }"}' \
    https://api.runpod.io/graphql
  ```
- **AWS `sr-train-gpu`** (i-0c16c38153c56df36): keep alive if running more experiments, otherwise terminate
  ```bash
  aws ec2 terminate-instances --instance-ids i-0c16c38153c56df36 --region us-east-1
  ```
- **Cleanup AWS resources** when done:
  ```bash
  aws ec2 delete-key-pair --key-name sr-train --region us-east-1
  aws ec2 delete-security-group --group-id sg-08a64f7d8e0f39bd1 --region us-east-1
  aws iam remove-role-from-instance-profile --instance-profile-name sr-train-s3 --role-name sr-train-s3
  aws iam delete-instance-profile --instance-profile-name sr-train-s3
  aws iam detach-role-policy --role-name sr-train-s3 --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  aws iam delete-role --role-name sr-train-s3
  ```

### 3. Run ESPCNR + Dual-Branch Comparisons
Run on the AWS A10G (if still up) or locally:
```bash
# ESPCNR baseline (37K params)
python train.py --tiles /home/ubuntu/tiles --arch espcnr --epochs 50 \
    --batch 8 --fft-weight 0.1 --run-id stage1_espcnr --gcs-status gs://...

# Dual-branch YCbCr (17.7K params)
python train.py --tiles /home/ubuntu/tiles --arch wsisrx4dual --epochs 50 \
    --batch 8 --fft-weight 0.1 --run-id stage1_dual --gcs-status gs://...
```

### 4. Fix Monitor Dashboard
- [ ] **Plan hierarchy**: Show Stage 0/1/2 with progress, not just pipeline steps
- [ ] **35 vs 800 context**: Make clear that current extraction is Stage 1 (35 slides), not the full plan (800)
- [ ] **Eval review page**: Load specific eval results, per-tile metrics
- [ ] **VM cost tracking**: Show actual spend per VM with running total
- [ ] **Extraction completed state**: Don't show stale "57%" when extraction is done
- [ ] **Training charts**: Pull training_log.jsonl from S3 and render Chart.js graphs
- [ ] **Layout**: Dataset overview + training plan on top; bars + cancer types full-height below (no scroll)
- [ ] **FFPE vs frozen**: Show that training data is 100% FFPE diagnostic slides

### 5. Fix GCS Write Auth for AWS
The current workaround (relay through local) is fragile. Options:
- **Best**: Give the AWS instance a GCS service account key file
- **Current**: Write to S3, monitor polls both S3 + GCS
- **Future**: Use a shared storage layer (e.g., just use S3 for everything)

### 6. Fix Baseline Computation Speed
The baseline SSIM/Delta E on full-res tiles was taking 30+ minutes. Fixed by sampling 20 tiles.
For validation during training, the same issue exists — each val epoch computes metrics on all 518 tiles.
Consider: only compute full metrics every 25 epochs, just PSNR + residual stats every 5 epochs.

### 7. Train on Full TCGA (Stage 2)
Gated on Stage 1 results passing:
- [ ] WSISRX4 beats lanczos3 baseline on PSNR and residual size
- [ ] No cancer type has catastrophic failure
- [ ] 20x and 40x magnification both converge
- [ ] Architecture decision made (WSISRX4 vs ESPCNR vs Dual)

## Decision Matrix After Stage 1

| If... | Then... |
|-------|---------|
| WSISRX4 beats lanczos3 by >1 dB | Proceed to Stage 2 with WSISRX4 |
| ESPCNR beats WSISRX4 significantly | Consider larger WSISRX4 (channels=32) or use ESPCNR |
| Dual-branch has better Delta E | Use dual-branch for chroma improvement |
| Model doesn't beat lanczos3 | Increase model capacity, add more blocks or channels |
| FFT loss helps SSIM but not PSNR | Keep FFT loss, it's helping structural quality |

## Timeline

| Priority | Task | Time | Cost |
|----------|------|------|------|
| NOW | Shut down GCP VM + RunPod B200 | 2 min | saves $0.14/hr + $4.99/hr |
| NOW | Add lanczos3 baseline, rerun eval | 30 min | $0.50 (AWS) |
| SOON | Run ESPCNR + Dual comparisons | 1 hr | $1.00 (AWS) |
| SOON | Fix monitor dashboard | 2 hrs | $0 (local) |
| NEXT | Stage 2 full training (if gates pass) | 8 hrs | $8-10 (AWS g5) |
