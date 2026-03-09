# TCGA Multi-WSI SR Training Plan

## 1. Dataset Overview (from GDC API query, March 2026)

**30,326 SVS slides** across 32 cancer types, 25 tissue sites, all open access, all H&E stained.

| Dimension | Count | Notes |
|-----------|-------|-------|
| Total slides | 30,326 | All SVS, all open access |
| Diagnostic (FFPE) | 11,901 | Higher quality, preferred for training |
| Tissue (frozen) | 18,425 | More artifacts, lower quality |
| Cancer types | 32 | BRCA, KIRC, GBM, LUSC, LUAD, ... |
| Tissue sites | 25 | Brain, Kidney, Lung, Breast, ... |
| Sample types | 6 | Primary Tumor (26,908), Normal (2,801), Metastatic (566), ... |
| Total storage | 17.25 TB | Median 269 MB, Mean 569 MB, Max 5.1 GB |

### Slides per cancer type

```
TCGA-BRCA:  3,112    TCGA-KIRC:  2,173    TCGA-GBM:   2,053
TCGA-LUSC:  1,612    TCGA-LUAD:  1,608    TCGA-LGG:   1,572
TCGA-OV:    1,481    TCGA-COAD:  1,442    TCGA-UCEC:  1,371
TCGA-HNSC:  1,263    TCGA-STAD:  1,197    TCGA-PRAD:  1,172
TCGA-THCA:  1,158    TCGA-SKCM:    950    TCGA-BLCA:    926
TCGA-SARC:    890    TCGA-LIHC:    870    TCGA-KIRP:    773
TCGA-TGCT:    663    TCGA-CESC:    604    TCGA-READ:    530
TCGA-PAAD:    466    TCGA-ESCA:    396    TCGA-PCPG:    385
TCGA-KICH:    326    TCGA-ACC:     323    TCGA-THYM:    318
TCGA-MESO:    175    TCGA-UCS:     154    TCGA-UVM:     150
TCGA-CHOL:    110    TCGA-DLBC:    103
```

### Key facts

- **All H&E stained** — TCGA does not include IHC slides
- **Diagnostic slides (FFPE)** are higher quality than tissue slides (frozen sections)
- For SR training, the model needs to generalize across tissue morphology, not cancer type per se
- The 25 tissue sites provide genuine morphological diversity (brain vs kidney vs lung vs skin, etc.)

## 2. Data Selection Strategy

### 2.1 Principles

1. **Coverage**: Include all 32 cancer types and all 25 tissue sites
2. **Balance**: Don't let BRCA (3,112 slides) dominate over CHOL (110 slides)
3. **Hold-outs**: Reserve entire slides for evaluation (never train on any tile from held-out slides)
4. **Tile subsampling**: Use every-Nth tile pattern, not all tiles — reduces processing time and creates adjacent-tile hold-outs
5. **Diagnostic only**: Use FFPE diagnostic slides (higher quality, fewer artifacts)

### 2.2 Slide Selection

**Per cancer type:**
- **Train**: Up to 20 diagnostic slides (randomly selected)
- **Eval**: 5 diagnostic slides (randomly selected, completely held out)
- **For small cancer types** (< 30 diagnostic slides): Use 60% train, 20% eval, 20% reserve

This gives us:
- ~640 training slides (32 types × 20, capped by availability)
- ~160 eval slides (32 types × 5, capped by availability)
- Total: ~800 slides to download

**Why 20 per type (not more)?**
- 20 slides × ~800 tiles/slide × 1-in-4 sampling = ~4,000 tiles per cancer type
- 32 types × 4,000 = **~128,000 training tiles** — more than enough
- Diminishing returns beyond 20 slides per type for morphological coverage
- Keeps total download to ~450 GB (manageable)

### 2.3 Tile Subsampling

Within each slide, extract 1024×1024 tiles using a **configurable stride pattern**:

```
--tile-stride N    Extract every Nth tile (default: 4)
                   N=1: all tiles
                   N=2: checkerboard (every other tile)
                   N=4: every 4th tile (1/4 of tiles)
```

**Tile coordinate assignment:**

```python
# For a slide with tiles at grid positions (tx, ty):
tile_id = tx + ty * tiles_per_row

if tile_id % stride == 0:
    → TRAIN tile (extracted and used for training)
elif tile_id % stride == 1:
    → ADJACENT EVAL tile (extracted separately, used to test generalization to nearby tissue)
else:
    → SKIP (not extracted, saves storage and processing time)
```

This gives us:
- **Training tiles**: 1/N of all tissue tiles
- **Adjacent eval tiles**: 1/N of all tissue tiles (interleaved with training tiles)
- **Held-out slide eval tiles**: All tiles from held-out slides (separate)

### 2.4 Evaluation Plan

Three evaluation levels:

| Level | What | Tests | Expected difficulty |
|-------|------|-------|-------------------|
| **Eval-1: Adjacent tiles** | Tiles from training slides, but not trained on (stride offset) | Can the model handle tissue it "almost" saw? | Easiest |
| **Eval-2: Held-out slides** | All tiles from slides in the same cancer type, never trained on | Can the model generalize across slides of the same tissue? | Medium |
| **Eval-3: Unseen cancer types** | Reserve 2-3 rare cancer types entirely (e.g., UVM, CHOL, DLBC) | Can the model generalize to completely unseen tissue morphology? | Hardest |

**Metrics** (matching ORIGAMI eval pipeline):
- PSNR, MSE, SSIM, MS-SSIM, Delta E (CIE2000), VIF, LPIPS, SSIMULACRA2
- Residual size at JPEG q60, q80, q90 (measures how compressible the prediction error is)
- All metrics computed per-tile and aggregated per cancer type

## 3. Data Manifest

Before downloading anything, generate a complete manifest:

**`tcga_training_manifest.json`**:
```json
{
  "created": "2026-03-07T...",
  "config": {
    "slides_per_type_train": 20,
    "slides_per_type_eval": 5,
    "tile_stride": 4,
    "tile_size": 1024,
    "holdout_cancer_types": ["TCGA-UVM", "TCGA-CHOL", "TCGA-DLBC"],
    "slide_filter": "diagnostic_only",
    "random_seed": 42
  },
  "summary": {
    "total_slides": 800,
    "train_slides": 640,
    "eval_slides_held_out": 160,
    "holdout_type_slides": 50,
    "estimated_tiles_per_slide": 800,
    "tile_stride": 4,
    "estimated_train_tiles": 128000,
    "estimated_eval_adjacent_tiles": 128000,
    "estimated_eval_holdout_tiles": 32000,
    "estimated_download_gb": 450,
    "estimated_tile_storage_gb_jpeg": 38
  },
  "cancer_types": [
    {
      "project_id": "TCGA-BRCA",
      "primary_site": "Breast",
      "total_diagnostic_slides": 1133,
      "train_slides": ["file_id_1", "file_id_2", ...],
      "eval_slides": ["file_id_21", "file_id_22", ...],
      "role": "train+eval"
    },
    ...
    {
      "project_id": "TCGA-UVM",
      "primary_site": "Eye",
      "total_diagnostic_slides": 80,
      "train_slides": [],
      "eval_slides": ["file_id_x", ...],
      "role": "holdout"
    }
  ],
  "slides": [
    {
      "file_id": "abc123",
      "file_name": "TCGA-A1-A0SB-01Z-00-DX1.svs",
      "file_size": 524288000,
      "project_id": "TCGA-BRCA",
      "primary_site": "Breast",
      "sample_type": "Primary Tumor",
      "role": "train",
      "tiles": {
        "train": [0, 4, 8, 12, ...],
        "eval_adjacent": [1, 5, 9, 13, ...]
      }
    },
    ...
  ]
}
```

**`tcga_training_manifest.csv`** (flat version for quick inspection):
```
file_id,file_name,project_id,primary_site,sample_type,role,file_size_mb
abc123,TCGA-A1-A0SB-01Z-00-DX1.svs,TCGA-BRCA,Breast,Primary Tumor,train,500
def456,TCGA-A1-A0SE-01Z-00-DX1.svs,TCGA-BRCA,Breast,Primary Tumor,eval,480
...
```

## 4. Infrastructure Plan

### 4.1 Phase 1: Manifest Generation (local, ~5 min)

Run locally — no cloud resources needed.

```bash
python wsi_sr/generate_manifest.py \
    --metadata wsi_sr/tcga_slides_metadata.json \
    --slides-per-type-train 20 \
    --slides-per-type-eval 5 \
    --tile-stride 4 \
    --holdout-types TCGA-UVM,TCGA-CHOL,TCGA-DLBC \
    --diagnostic-only \
    --seed 42 \
    --output wsi_sr/tcga_training_manifest.json
```

**Output**: Manifest files (JSON + CSV) with every slide and tile assignment.
**Cost**: $0

### 4.2 Phase 2: Download + Tile Extraction on GCP (~$5)

**Machine**: c3-highcpu-88 spot VM in us-central1 + 2× 375 GB local NVMe SSD (750 GB)

**Pipeline** (streaming, never stores all SVS files at once):

```
For each slide in manifest:
  1. gsutil cp gs://gdc-tcga-phs000178-open/{file_id}/{file_name} /mnt/ssd/
  2. Extract tiles at stride=4 using OpenSlide (parallel across 44 cores, 2 per slide)
  3. Save JPEG q95 tiles to /mnt/ssd/tiles/{project_id}/{slide_id}/
  4. Delete the SVS file
  5. When batch complete: gsutil -m cp tiles/ gs://our-bucket/tcga-tiles/
```

| Metric | Value |
|--------|-------|
| Slides to download | 800 |
| SVS download volume | 960 GB |
| GCS → VM throughput (same region) | ~500 MB/s |
| Download time (streaming, overlapped) | ~32 min |
| Tiles to extract | ~737,000 (206K train + 206K eval adj + 325K eval holdout) |
| Extraction rate (44 parallel slides × 15 tiles/sec) | ~660 tiles/sec |
| Extraction time | ~19 min |
| Output tile volume (JPEG q95) | ~110 GB (737K × 150 KB avg) |
| Upload to GCS | ~4 min |
| **VM runtime** | **~45 min** |
| **VM cost** (c3-highcpu-88 spot ~$1.50/hr) | **~$1.15** |
| **GCS storage** (110 GB × $0.02/GB/mo) | **$2.20/mo** |
| **Total Phase 2 cost** | **~$3.50** |

### 4.3 Phase 3: Transfer to RunPod (~$14)

```bash
# On RunPod pod, pull from GCS
gcloud storage cp -r gs://our-bucket/tcga-tiles/ /workspace/tiles/
```

| Metric | Value |
|--------|-------|
| Data volume | 110 GB |
| GCS egress ($0.12/GB) | **$13.20** |
| Transfer time (~500 MB/s) | ~4 min |
| RunPod network volume (110 GB @ $0.07/GB/mo) | $7.70/mo |

### 4.4 Phase 4: Training on RunPod B200 (~$25)

**Machine**: NVIDIA B200 (183 GB VRAM) — existing pod `origami-b200` (`tyjyett3vxrvbe`)

| Parameter | Value |
|-----------|-------|
| Training tiles | 206,242 |
| Batch size | 64 (B200 has 183 GB VRAM — can handle large batches) |
| Batches per epoch | 206,242 / 64 = 3,222 |
| Time per batch (B200, estimated) | ~20ms |
| **Time per epoch** | **~64 seconds** |
| Epochs | 200 |
| **Total training time** | **~3.6 hours** |
| Validation (every 5 epochs, 40 val passes) | ~1.5 hours |
| **Total GPU time** | **~5 hours** |
| B200 cost/hr (estimated) | ~$4-5/hr |
| **Training cost** | **~$22-25** |

Full evaluation pass on held-out data after training:

| Eval task | Tiles | Time (est.) |
|-----------|-------|-------------|
| Eval-1: Adjacent tiles | 206,242 | ~45 min |
| Eval-2: Held-out slides (5 per type × 29 types) | ~118,000 | ~25 min |
| Eval-3: Holdout cancer types (UVM + CHOL + DLBC) | ~108,784 | ~25 min |
| **Total eval** | **~433,000 tiles** | **~1.5 hours** |

**Total Phase 4 (train + eval)**: ~6.5 hours × ~$4.50/hr = **~$30**

### 4.5 Total Cost Summary

| Phase | Time | Cost |
|-------|------|------|
| 1. Manifest generation (local) | 5 min | $0 |
| 2. GCP download + extract | 45 min | $3.50 |
| 3. Transfer to RunPod | 4 min | $13.20 |
| 4. Training + eval on B200 | 6.5 hrs | ~$30 |
| GCS storage (1 month) | — | $2.20 |
| RunPod network volume (1 month) | — | $7.70 |
| **Total** | **~7.5 hours** | **~$57** |

## 5. Scripts Needed

| Script | Purpose | Status |
|--------|---------|--------|
| `tcga_metadata.py` | Query GDC API for all slide metadata | Done |
| `generate_manifest.py` | Select slides, assign train/eval roles, define tile IDs | Done |
| `extract_tiles_tcga.py` | Download SVS from GCS + extract tiles per manifest | TODO |
| `gcp_setup.sh` | Provision c3-highcpu-88 spot VM, install deps, run extraction | TODO |
| `prepare_tiles.py` | Already exists — needs OpenSlide stride support added | Update |
| `evaluate.py` | Full visual metrics (PSNR, MSE, SSIM, Delta E, VIF, LPIPS, SSIMULACRA2) + residual sizes | Done |
| `train.py` | Val loop logs PSNR, MSE, SSIM, Delta E + residual stats; needs multi-dir tile support | Partial |
| `plot_training.py` | 12-panel plots: Loss, PSNR, SSIM, Delta E, SSIMULACRA2, MSE, residuals, LR, scatter | Done |
| `db.py` | DuckDB store: stages, slides, manifests, runs, epochs, eval_tiles, tile_losses | Done |
| `load_plan.py` | Seed DuckDB with metadata (30K slides), manifest (800 slides), pipeline stages | Done |
| `monitor/server.js` | Node.js Express server, reads DuckDB, serves REST API + dashboard | Done |
| `monitor/public/index.html` | Live dashboard: pipeline stages, dataset, manifest, training charts | Done |
| `pod_train_tcga.sh` | RunPod training script for TCGA dataset | TODO |

## 6. Zoom / Scale Handling

TCGA slides vary between 20x and 40x magnification. This is critical because:
- A 1024×1024 tile at **40x** covers **~256×256 µm** (cell-level detail, fine structures)
- A 1024×1024 tile at **20x** covers **~512×512 µm** (tissue-level context, larger structures)
- The same model must handle both — the frequency content is fundamentally different

### Approach

1. **Record magnification** during tile extraction — read from SVS metadata via OpenSlide:
   ```python
   mpp = float(slide.properties.get('openslide.mpp-x', 0))  # microns per pixel
   objective = slide.properties.get('openslide.objective-power', '?')
   ```

2. **Train on both 20x and 40x** — the model should learn to be scale-invariant.
   The diversity is a feature, not a bug. Our production pipeline serves slides at
   whatever magnification they were scanned at.

3. **Stratify evaluation by magnification** — report metrics separately for 20x and 40x
   tiles to see if the model struggles with one more than the other.

4. **Tile metadata** — each extracted tile gets a sidecar JSON or is logged in a CSV:
   ```
   tile_id, slide_id, project_id, x, y, magnification, mpp, scanner
   ```

5. **Early training**: Expect higher deviations initially because the model sees both
   scales. The max_dev and p99_dev metrics will be noisy early on — this is normal.
   Track separately by magnification to understand convergence per scale.

### Future option: Scale-aware model

If evaluation shows the model struggles with one scale, we could:
- Add magnification as a conditioning input (embed 20x/40x as a learned vector)
- Train separate models for 20x and 40x
- Extract all tiles at 20x by downsampling 40x slides 2x first

## 7. Training Monitoring and Plots

### Training log

`train.py` writes a JSON-lines log file (`training_log.jsonl`) with:
- **Every epoch**: loss, train_psnr, learning rate, elapsed time
- **Every 5 epochs** (validation): full visual quality metrics + residual stats:
  - `val_psnr` — peak signal-to-noise ratio (dB)
  - `val_mse` — mean squared error (0-65025 scale)
  - `val_ssim` — structural similarity index (0-1)
  - `val_delta_e` — CIE2000 color difference (perceptual units)
  - `val_ssimulacra2` — SSIMULACRA2 perceptual score (>90 excellent, >70 good)
  - `residual.size_kb` — JPEG-compressed residual size at q80
  - `residual.max_dev` — worst single-pixel error
  - `residual.p99_dev`, `p95_dev`, `mean_dev` — deviation percentiles
  - `residual.pct_over_10/20/30` — percentage of outlier pixels
- **Header**: config, baseline stats for bilinear/bicubic (including PSNR, MSE, SSIM, Delta E, residual stats)

### Plot generation

```bash
# Generate plots (can run while training is in progress)
python plot_training.py --log checkpoints/training_log.jsonl

# Live mode — updates every 30s during training
python plot_training.py --log checkpoints/training_log.jsonl --live
```

### 12-panel plot layout

| Panel | What it shows | What to look for |
|-------|--------------|------------------|
| **Loss** (log scale) | MAE training loss over epochs | Smooth decrease, no divergence |
| **PSNR** | Train + val PSNR with best marker | Val should track train, not diverge (overfit) |
| **SSIM** | Val SSIM with bilinear/bicubic baselines | Should climb above baselines; higher = better structural fidelity |
| **Delta E** (CIE2000) | Val Delta E with baselines | Should drop below baselines; lower = less perceptible color error |
| **SSIMULACRA2** | Val SSIMULACRA2 with baselines | Should climb; >90 excellent, >70 good, >50 ok |
| **MSE** | Val MSE with baselines | Should drop below baselines; lower = better |
| **Residual Size** | SR model vs bilinear/bicubic baselines | SR line should drop below bicubic dashed line |
| **Deviation Percentiles** | max, p99, p95, mean deviation | max should decrease; p99 tells you about outliers |
| **Outlier %** | % of pixels >10, >20, >30 error | >20% should approach 0; >30% should hit 0 early |
| **Learning Rate** | Cosine annealing schedule | Verify schedule is correct |
| **PSNR vs Residual Size** | Quality-efficiency tradeoff by epoch (with baseline points) | Points should move up-left over time |
| **Storage Savings** | % reduction vs bicubic baseline | Target: >30% reduction |

### Hard example mining

`train.py --hard-mining 2.0` enables difficulty-aware oversampling:

1. **Per-tile loss tracking**: EMA of MAE (mean absolute error) loss per tile during training
2. **Additive oversampling**: After each validation, rebuild the sampler — every tile keeps its baseline frequency, hard tiles get *extra* samples on top. No diversity is lost; epochs get slightly longer.
3. **Meaningful permutations**: Every oversampled access gets a different random flip/rotation (16 unique orientations: hflip × vflip × 0°/90°/180°/270°) — never an identical repeat
4. **Principle**: Any oversampling (hard examples, rare cancer types, etc.) must always apply permutation (rotation, flip) so the model sees genuinely different views, not memorized duplicates. Oversampling must *add* to the training set, never crowd out easy examples.

**Full difficulty tracking**: `evaluate.py` computes a composite difficulty score for *every* tile (not just top-N) combining max_dev, p99_dev, and residual size. Results are saved as a complete ranked JSON + CSV for analysis, sliceable by cancer type, magnification, slide, etc.

**Future: Overlapping re-cuts** — For production TCGA training, hard tiles should be re-cut from the WSI at overlapping positions (offset by half a tile) to expose the model to the challenging morphology in different spatial contexts. This requires adjacent tile data and is planned for `extract_tiles_tcga.py`.

### Plateau detection and exploration mode

Training never stops early. Instead, when the model plateaus, **exploration runs alongside training** during validation epochs:

```bash
python train.py --tiles /path/to/tiles --patience 30 \
    --explore-tiles /path/to/new_tiles --db wsi_sr.duckdb
```

| Phase | What happens |
|-------|-------------|
| **Training** | Normal training + validation every 5 epochs |
| **Plateau detected** | Val PSNR hasn't improved for `--patience` val epochs |
| **Training + exploration** | Training continues normally. During each val epoch, a batch of new tiles from `--explore-tiles` is also evaluated and results stored in DuckDB |
| **Hard tile feedback** | Explored tiles harder than the median training tile are added to the training pool immediately, with high initial weight for hard mining |

Training never stops because:
- The LR schedule (cosine annealing) can break through plateaus later
- Random augmentation may expose the model to new perspectives on hard tiles
- Hard tiles from exploration feed directly into the current run's training pool
- The training pool grows organically as harder morphology is discovered

The feedback loop: **explore → identify hard tiles → add to training pool → hard mining upweights them → model trains on them with rotations/flips → next val epoch explores more tiles → repeat**

All exploration results also go into DuckDB for cross-run analysis:
- Per-cancer-type difficulty → adjust future oversampling ratios
- Persistent hard tiles → candidates for overlapping re-cuts from WSI

### What to watch for

- **Early epochs (1-20)**: Large deviations are normal. The model is learning basic structure.
  max_dev might be 80-100, pct_over_20 might be 10-20%. Don't panic.
- **Mid training (20-100)**: PSNR should climb steadily. Residual size should cross below
  the bicubic baseline. pct_over_20 should drop to <5%.
- **Late training (100-200)**: Diminishing returns. If val PSNR plateaus but max_dev is
  still high, the model has systematic failures on certain tissue types.
- **Overfitting**: Train PSNR keeps climbing but val PSNR drops. Stop training, reduce model
  complexity or increase augmentation.
- **Scale issues**: If residual stats are bimodal, the model might be handling one magnification
  well and struggling with the other. Check per-magnification eval.

## 8. Data Store (DuckDB)

All training lifecycle data goes into a single DuckDB file (`wsi_sr.duckdb`):

| Table | What it stores | Loaded |
|-------|---------------|--------|
| `stages` | Pipeline stages (manifest→extract→transfer→train→eval→export) with status, progress, cost | 6 rows |
| `slides` | All TCGA slide metadata: file_id, project, site, strategy, file size | 30,326 rows |
| `manifests` | Training plan configs and summaries | 1 row |
| `manifest_slides` | Slide assignments: file_id → role (train/eval/holdout) + tile estimates | 800 rows |
| `manifest_cancer_types` | Per-cancer-type summary: slide counts, tile estimates, role | 32 rows |
| `runs` | Training run configs, status, best metrics, exploration stats | — |
| `baselines` | Bilinear/bicubic baseline metrics per run | — |
| `epochs` | Per-epoch metrics (loss, PSNR, SSIM, Delta E, SSIMULACRA2, residuals, exploration) | — |
| `tile_losses` | Per-tile MAE loss per epoch (for hard example mining) | — |
| `eval_tiles` | Per-tile eval metrics for every method, with cancer type, magnification, fed_back flag | — |

**Key queries:**
```sql
-- Difficulty by cancer type
SELECT cancer_type, AVG(difficulty), COUNT(*) FROM eval_tiles
WHERE run_id='tcga_v1' AND method='sr' GROUP BY cancer_type ORDER BY 2 DESC;

-- Compare two runs
SELECT run_id, AVG(psnr), AVG(ssim), AVG(difficulty) FROM eval_tiles
WHERE run_id IN ('v1', 'v2') AND method='sr' GROUP BY run_id;

-- Tiles that never improved (stayed hard across epochs)
SELECT tile_idx, MIN(loss) as best, MAX(loss) as worst FROM tile_losses
WHERE run_id='tcga_v1' GROUP BY tile_idx HAVING MIN(loss) > 0.05 ORDER BY best DESC;

-- Training curve
SELECT epoch, val_psnr, val_ssim, residual_size_kb FROM epochs
WHERE run_id='tcga_v1' AND type='val' ORDER BY epoch;
```

**Usage:** `train.py --db path/to/wsi_sr.duckdb --run-id tcga_v1`
Falls back to JSONL-only if duckdb is not installed.

See `wsi_sr/db.py` for the full API (`WSISRDB` class).

## 9. Monitoring Dashboard

Live web dashboard at `http://localhost:8090` (Node.js + DuckDB):

```bash
cd wsi_sr/monitor && npm install && npm start
```

| Section | What it shows |
|---------|-------------|
| **Pipeline** | 6 stage cards with status (pending/running/completed), progress bars, estimated vs actual cost |
| **TCGA Dataset** | Total slide count, diagnostic vs tissue, slides-per-cancer-type bar chart |
| **Training Plan** | Manifest summary: slides by role, tile estimates, holdout types, cancer type table |
| **Training** | Run summary, baseline metrics table, 6 live-updating Chart.js charts (loss, PSNR, SSIM, Delta E, residual size, deviations) |
| **Evaluation** | Difficulty by cancer type, hardest tiles table |

Auto-refreshes every 10 seconds. Python writes to DuckDB, Node reads it (read-only mode).

The `/api/query?sql=SELECT...` endpoint allows ad-hoc SQL for exploration.

## 10. Related Work and Baselines (March 2026 research)

### Prior WSI super-resolution

| Paper/Project | Year | Approach | Scale | Params | PSNR (4x) | Notes |
|---------------|------|----------|-------|--------|-----------|-------|
| WSISR (uw-loci) | 2021 | U-Net + GAN, full RGB | 5x | ~5.6M | n/a (5x) | Outdated, not a good baseline |
| TCGA comparative | 2023 | DBPN/RCAN/ESRGAN | 4x | 1-10M | 24-26 dB | Pathologists showed no diagnostic difference |
| Frozen Section SR | 2024 | Residual learning + FFT loss | 4x | ~1M | 32.12 dB | **Most relevant** — validates residual prediction for histopath |
| Histo-Diffusion | 2024 | SwinIR + ControlNet SD2.1 | 4x | ~1B | 24-26 dB | Perceptual quality focus, impractical for serving |
| MISRNN | 2025 | Multi-input lightweight | 4x | ~500K | 33.98 dB | Best distortion metrics |

### Efficient SR architectures (relevant to our 19K-param target)

| Architecture | Year | Params | PSNR (Set5 4x) | Latency | Key idea |
|-------------|------|--------|-----------------|---------|----------|
| ESPCN | 2016 | 24K | 30.66 dB | 2.7ms | Sub-pixel convolution baseline |
| ECBSR (M4C16) | 2021 | ~20K | 30.89 dB | <5ms | Reparameterizable blocks (our approach) |
| PlainUSR-U | 2024 | ~20K | 30.77 dB | 3.2ms | RepMBConv, fastest ConvNet for SR |
| PlainUSR-B | 2024 | 333K | 32.02 dB | 14.1ms | Sweet spot for quality/speed |

### Key insights

1. **Our approach is validated**: Residual prediction for histopathology (Yoshai 2024) + reparameterizable blocks for efficient SR (ECBSR, PlainUSR) are both proven techniques. Our combination is novel.
2. **Don't use GANs**: For a codec, pixel accuracy matters — GAN hallucinations are incorrect reconstruction. Pure L1/L2 on residuals is safest.
3. **Consider FFT loss**: Frozen section SR paper found frequency-domain weighting on high-frequency components improved results. Worth testing alongside L1.
4. **Our task is easier than general SR**: Known degradation (JPEG + interpolation), luma-only prediction, chroma handled by interpolation. This justifies the 19K param budget.
5. **Target PSNR**: 32+ dB at 4x is competitive with residual methods. Below 30 dB would indicate problems.
6. **Delta E is diagnostically relevant**: Literature confirms PSNR/SSIM can be misleading for histopathology. Our comprehensive metric suite (especially Delta E CIE2000) is well-chosen.

### Evaluated and rejected

- **autoresearch (Karpathy 2026)**: LLM-driven autonomous architecture search. Interesting pattern but text-only, single metric, not applicable as a tool. Our DuckDB + monitor infrastructure is more comprehensive.
- **demo_wsi_superres (uw-loci 2021)**: Too large (5.6M params), wrong scale (5x), GAN-based, outdated deps. Not useful as baseline.

### Useful baselines for comparison

1. Bicubic/Lanczos3 interpolation (already implemented)
2. JPEG quality sweep (already in evals)
3. ESPCN at ~24K params (well-understood lightweight SR baseline) — **TODO: implement**
4. If we want an upper bound: EDSR or SwinIR at 1M+ params

## 11. Open Questions

1. **FFT loss**: Should we add frequency-domain weighting to the L1 loss? The frozen section SR paper found it helps with high-frequency detail. Low risk to try.

2. **ESPCN baseline**: Implement ESPCN (~24K params) as a baseline comparison to validate our reparameterizable block advantage.

3. **Frozen sections**: Should we include tissue slides (frozen) or stick to diagnostic (FFPE) only?
   - Frozen sections have more artifacts but represent a real use case
   - Option: Train on FFPE only, evaluate on both FFPE and frozen to test robustness

2. **Scanner variation**: TCGA slides come from multiple scanners (Aperio, Hamamatsu, 3DHISTECH, etc.)
   - Scanner info is embedded in SVS headers, not in GDC metadata
   - Different scanners produce different JPEG compression, color profiles, and artifacts
   - This is actually good for training — forces the model to be scanner-agnostic

3. **IHC data**: TCGA is H&E only. If IHC support is needed, we'd need additional datasets:
   - Multi-Stain Breast Cancer dataset
   - HPA (Human Protein Atlas)
   - Custom data from our own pipeline

## 12. Staged Execution

The pipeline runs in 3 stages. Each stage validates everything works before scaling up. Gate criteria must pass before proceeding to the next stage.

### Stage 0: Local Smoke Test — COMPLETE ✓

All architectures train, converge, and produce valid metrics. Monitor dashboard works.

### Stage 1: Small TCGA Slice — COMPLETE ✓ (2026-03-09)

**Data**: 798/800 slides extracted, 1.4M tiles, 246 GiB in GCS `gs://wsi-1-480715-tcga-tiles/stage2/`
**Training**: 5 architectures compared on 5,180 tiles, 50 epochs each (AWS A10G + RunPod H100)

**Gate criteria results**:
- [x] Tile extraction pipeline works end-to-end (798/800 slides, GCP c3-highcpu-22)
- [x] Model trains and converges on real TCGA data
- [x] WSISRX4 beats lanczos3 by +1.8 dB PSNR, 10% smaller residuals
- [x] Dual-branch wins: best PSNR (30.52), smallest residuals (125.3 KB), smallest model (18K)
- [x] ONNX integration working in Rust server (sr_model.rs, pool=8)
- [ ] 20x vs 40x stratification — not yet tested (tiles not labeled by magnification)
- [ ] Hard mining — not yet tested in training

**Architecture comparison results**:

| Model | Params | PSNR | SSIM | Delta E | Residual KB |
|-------|--------|------|------|---------|-------------|
| **WSISRX4Dual** | 18K | **30.52** | 0.860 | 2.54 | **125.3** |
| ESPCNR | 37K | 30.49 | **0.861** | **2.41** | 126.7 |
| WSISRX4 | 19K | 30.38 | 0.858 | 2.55 | 127.0 |
| WSISRX4Large | 295K | 29.67 | — | — | 127.0 |
| WSISRX4WideDeep | 112K | 29.57 | 0.855 | 2.63 | 127.2 |
| *Lanczos3* | — | 28.59 | 0.830 | 2.80 | 141.4 |

**Key findings**:
1. Dual-branch (18K) wins on PSNR and residual size
2. Larger models (112K, 295K) WORSE on 5K tiles / 50 epochs — underfitting
3. All SR models beat lanczos3 by 1-2 dB
4. Model capacity is not the bottleneck on small data — data volume is

**Actual cost**: ~$30 (training ~$25, extraction ~$5)

### Stage 2: Full TCGA Training — IN PROGRESS

**Goal**: Train on 1.4M tiles (280x more data) to push quality further and test whether larger models benefit from more data.

**Data available**: 1.4M tiles, 246 GiB in GCS (already extracted)

#### Phase 1: Quick Validation on A10G (4 hrs, ~$4)

Run on existing AWS A10G with Stage 1 data (5K tiles). Answer hyperparameter questions cheaply before committing to expensive Stage 2 runs.

| Experiment | Purpose | Decision Point |
|------------|---------|----------------|
| Dual 200 epochs | Does the winning model saturate on 5K tiles? | Plateau by ep 80-100 → needs more data |
| Dual + FFT loss | Does frequency-domain loss shrink residuals? | Residual KB drops → use FFT for Stage 2 |
| WideDeep lr=5e-5 | Was poor Stage 1 result a learning rate issue? | Beats Dual → LR was the problem |

#### Phase 2: Data Transfer to H100 (1-2 hrs, ~$18)

Transfer 246 GiB from GCS to RunPod H100 local NVMe storage. Generate manifest for fast loading.

#### Phase 3: Primary Stage 2 Runs on H100 (14-21 hrs, ~$42-63)

1.4M tiles, batch=64, ~6.6 min/epoch, validation every 5 epochs.

| Priority | Run | Key Question | Time |
|----------|-----|-------------|------|
| 1 | Dual on Stage 2 (50 ep) | Does 280x more data push past 31 dB? | ~7 hrs |
| 2 | WideDeep on Stage 2 (50 ep) | Does the 112K model converge with enough data? | ~7 hrs |
| 3 | Dual + FFT (50 ep) | Does FFT loss help residual compression? | ~7 hrs |

Decision point at epoch 20 (~2 hrs): if Dual PSNR > 31 dB and climbing → data is helping.

#### Phase 4: Extended Training for Winner (8-16 hrs, ~$24-48)

Best model from Phase 3, 200 epochs with hard mining enabled (`--hard-mining 2.0`).

#### Phase 5: Export & Integration (1 hr, free)

ONNX export + INT8 quantization → test in Rust pipeline.

#### Stage 2 Success Targets

| Metric | Stage 1 Dual | Target | Stretch |
|--------|-------------|--------|---------|
| Val PSNR | 30.52 dB | 31.5+ dB | 32.5+ dB |
| Residual KB | 125.3 | <110 | <100 |
| vs Lanczos3 | +1.93 dB | +2.5 dB | +3.5 dB |
| ONNX size | 69 KB | <500 KB | <100 KB |
| Decode latency | ~48ms/family | <50ms | <30ms |

#### Stage 2 Cost Estimate

| Phase | GPU | Hours | Cost |
|-------|-----|-------|------|
| Phase 1: Quick experiments | A10G | 4 | $4 |
| Phase 2: Data transfer | H100 | 1 | $18 |
| Phase 3: Primary runs | H100 | 14-21 | $42-63 |
| Phase 4: Extended training | H100 | 8-16 | $24-48 |
| Phase 5: Export/eval | local | 1 | $0 |
| **Total** | | **28-44 hrs** | **$88-133** |

### Decision Points

| If... | Then... |
|-------|---------|
| Dual PSNR > 31.5 on Stage 2 data | Dual is the right architecture; train to 200 epochs |
| WideDeep beats Dual on Stage 2 | Larger model justified; evaluate inference cost tradeoff |
| WideDeep still worse on Stage 2 | Dual is capacity-optimal; don't try Large |
| FFT loss reduces residual KB | Add FFT to final training run |
| Model plateaus before epoch 50 | Architecture is the bottleneck; try wider Dual (channels=24) |
| Stage 2 PSNR stays at ~30.5 | Data diversity doesn't help; focus on loss functions or augmentation |

## 13. Timeline

| Day | Activity | Status |
|-----|----------|--------|
| Day 1 | Stage 0: local smoke test | ✓ DONE |
| Day 1-2 | Stage 1: extraction + architecture comparison | ✓ DONE |
| Day 3 | Stage 2 Phase 1: quick validation experiments | NEXT |
| Day 3 | Stage 2 Phase 2: data transfer to H100 | NEXT |
| Day 3-4 | Stage 2 Phase 3: primary training runs | PLANNED |
| Day 4-5 | Stage 2 Phase 4: extended training for winner | PLANNED |
| Day 5 | Stage 2 Phase 5: export, eval, integration | PLANNED |

## 13. Cleanup (REQUIRED — plan is not complete until done)

**Do NOT auto-clean. All cleanups require manual confirmation.**

### Compute

| Resource | Provider | How to verify | How to clean |
|----------|----------|--------------|-------------|
| `tcga-extract-s1` | GCP us-central1-c | `gcloud compute instances list --project=wsi-1-480715` | `gcloud compute instances delete tcga-extract-s1 --zone=us-central1-c --project=wsi-1-480715` |
| `sr-train-gpu` | AWS us-east-1 | `aws ec2 describe-instances --filters Name=tag:Name,Values=sr-train-gpu --region us-east-1` | `aws ec2 terminate-instances --instance-ids <id> --region us-east-1` |
| `origami-b200-sr` | RunPod | Query RunPod API | `podTerminate` mutation |
| Any other pods | RunPod | Query RunPod API | Stop or terminate via API |

### Storage

| Resource | Provider | Size | How to clean |
|----------|----------|------|-------------|
| `gs://wsi-1-480715-tcga-tiles/` | GCS | ~110 GB (full) / ~3 GB (stage1) | `gsutil -m rm -r gs://wsi-1-480715-tcga-tiles/stage1/` |
| EBS volume on sr-train-gpu | AWS | 200 GB | Deleted with instance termination |
| RunPod network volume | RunPod | 150 GB | Deleted with pod termination |

### Keys / Credentials

| Resource | Where | How to clean |
|----------|-------|-------------|
| `sr-train` key pair | AWS us-east-1 + `~/.ssh/sr-train.pem` | `aws ec2 delete-key-pair --key-name sr-train --region us-east-1` |
| `sr-train-sg` security group | AWS us-east-1 | `aws ec2 delete-security-group --group-id <id> --region us-east-1` |

### Cleanup checklist

- [ ] All GCP VMs stopped/deleted
- [ ] All AWS instances terminated
- [ ] All RunPod pods stopped/terminated
- [ ] GCS staging data deleted (keep final model + eval results)
- [ ] AWS EBS volumes deleted
- [ ] AWS key pair + security group deleted
- [ ] Final model checkpoint + eval results downloaded locally
- [ ] Cost audit: verify actual spend matches estimates
