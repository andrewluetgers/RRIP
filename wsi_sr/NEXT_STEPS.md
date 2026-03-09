# WSI SR — Next Steps

## Status (Stage 1 training + architecture comparison IN PROGRESS)

**Stage 1 extraction**: COMPLETE — 35 slides, 24,092 tiles from 5 cancer types (BRCA, GBM, KIRC, LUAD, SKCM)
**Stage 2 extraction**: IN PROGRESS — 104/800 slides, 191K tiles on GCP c3-highcpu-22 (11 workers)

### Architecture comparison (3 runs on AWS A10G, 5,180 tiles, 50 epochs each)

| Metric | Bilinear | Bicubic | Lanczos3 (ORIGAMI) | WSISRX4 (ours) | ESPCNR | Dual |
|--------|----------|---------|-------------------|----------------|--------|------|
| Val PSNR | 27.02 | 28.34 | **28.59** | **30.38** | running | pending |
| Val SSIM | 0.776 | 0.822 | **0.830** | **0.858** | running | pending |
| Val Delta E | 3.30 | 2.87 | **2.80** | **2.55** | running | pending |
| Residual KB | 150.5 | 142.5 | **141.4** | **127.0** | running | pending |
| Params | — | — | — | 19,008 | 37,200 | 17,736 |

**KEY RESULT: WSISRX4 beats lanczos3 by +1.8 dB PSNR, 10% smaller residuals, better Delta E.**

### Active Infrastructure

| Provider | Instance | Type | Purpose | Cost/hr |
|----------|----------|------|---------|---------|
| GCP | tcga-extract-s2 | c3-highcpu-22 spot | Stage 2 extraction (104/800) | ~$0.35 |
| AWS | i-0c16c38153c56df36 | g5.xlarge (A10G) | Architecture comparison runs | $1.01 |
| RunPod | — | — | No active pods (capacity issues) | $0 |

**Total spend so far**: ~$7.16

## Immediate Tasks

### 1. ~~Add Lanczos3 Baseline~~ DONE
Lanczos3 baseline added and confirmed: PSNR 28.59, SSIM 0.830, Delta E 2.80.
WSISRX4 beats it: +1.8 dB PSNR, +0.028 SSIM, -0.25 Delta E, -10% residual size.

### 2. ~~Shut Down Idle VMs~~ PARTIALLY DONE
- **GCP `tcga-extract-s1`**: DELETED (was preempted then deleted)
- **RunPod `origami-b200-sr`**: TERMINATED (never provisioned)
- **GCP `tcga-extract-s2`**: RUNNING — doing Stage 2 extraction (104/800 slides)
- **AWS `i-0c16c38153c56df36`**: RUNNING — doing architecture comparison runs
- When all work is done, clean up:
  ```bash
  # GCP
  gcloud compute instances delete tcga-extract-s2 --zone=us-central1-c --project=wsi-1-480715 --quiet
  # AWS
  aws ec2 terminate-instances --instance-ids i-0c16c38153c56df36 --region us-east-1
  aws ec2 delete-key-pair --key-name sr-train --region us-east-1
  aws ec2 delete-security-group --group-id sg-08a64f7d8e0f39bd1 --region us-east-1
  aws iam remove-role-from-instance-profile --instance-profile-name sr-train-s3 --role-name sr-train-s3
  aws iam delete-instance-profile --instance-profile-name sr-train-s3
  aws iam detach-role-policy --role-name sr-train-s3 --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
  aws iam delete-role --role-name sr-train-s3
  ```

### 3. Run ESPCNR + Dual-Branch Comparisons — IN PROGRESS
Running on AWS A10G (i-0c16c38153c56df36, 98.80.233.222):
- Run 1 (WSISRX4 v2 with lanczos3 baseline): **DONE** — PSNR 30.38
- Run 2 (ESPCNR 37K params): **IN PROGRESS** — epoch ~14/50
- Run 3 (Dual-branch 17.7K params): **PENDING**
- All 3 runs sequenced in `/home/ubuntu/comparisons.log`
- Results auto-upload to `s3://wsi-sr-training-results/stage1/`
- Status cron writes `active_run.json` to S3 every 30s
- SSH: `ssh -o StrictHostKeyChecking=no -i ~/.ssh/sr-train.pem ubuntu@98.80.233.222`
- Log: `tail -f /home/ubuntu/comparisons.log`

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
