# Residual Denoising & Structure/Noise Separation — Experiment Plan

## Goal

Separate ORIGAMI's luma residuals into **structure** (compressible signal carrying real correction information), **noise** (incompressible random content), and **artifacts** (periodic JPEG grid bleed-through). This could:

1. **Reduce residual file size** by only coding the structure component
2. **Improve SR model quality** by retraining on denoised targets (the predictor stops wasting capacity predicting noise)
3. **Maintain perceptual quality** by synthesizing statistically equivalent noise at decode time (optional)

## Background

### What's in the residual?

ORIGAMI residuals = `ground_truth_Y - predicted_Y + 128` (centered at 128). Three components:

| Component | Source | Compressible? |
|-----------|--------|---------------|
| **Prediction error structure** | Edges, textures the upsampler missed | Yes — spatially correlated |
| **Prediction error noise** | Quantization noise, sensor noise, stochastic texture | No — near-random |
| **JPEG block artifacts** | L2 JPEG 8x8 grid bleeding through 4x upsample | Partially — periodic at 32px intervals |

### Key insight: at encode time we have all three source images

```
original_Y    (1024x1024)  — the ground truth luma
prediction_Y  (1024x1024)  — lanczos3/SR upsample of L2
residual      (1024x1024)  — original_Y - prediction_Y + 128
```

Structure in the residual **must correlate with edges in the original that the prediction got wrong**. This gives us a per-pixel signal for what's real correction vs noise.

### Empirical findings (March 2026)

From `evals/scripts/analyze_residual_compressibility.py`:

- **Distribution**: Laplacian, b=4.6-4.8, std≈6.5, centered at 128
- **Spatial redundancy**: Only 0.29 bpp (0th-order entropy 5.41 bpp vs 1st-order 5.12 bpp)
- **Autocorrelation**: Lag-1 = 0.35-0.46, drops to <0.1 by lag-5 (short-range only)
- **JPEG artifacts confirmed**:
  - Boundary energy ratio 1.05-1.10 at periods 8/32/64
  - FFT peak at 1/8 frequency: 1.75x (strong 8px grid from L2 JPEG)
  - Column variance autocorrelation: 0.61 at lag-8, 0.52 at lag-32
- **Power spectrum**: Flat/rising — no low-frequency dominance, it's all high-frequency
- **Conclusion**: JPEG already captures most exploitable structure. A learned codec cannot beat JPEG here.

---

## Phase 1: Noise-Floor Evaluation Framework

**Purpose**: Build an automated eval that quantifies how much of a residual (or any image) is incompressible noise vs compressible structure. This is the foundation — every later experiment needs this to measure progress.

### 1.1 Metrics to implement

All metrics operate on a single-channel image (uint8, centered at 128 for residuals). The eval takes an image and outputs a "noise report."

| Metric | What it measures | Noise-like value |
|--------|-----------------|-----------------|
| **Incompressibility ratio** | `LZ4_compressed_size / raw_size` | >0.95 = pure noise |
| **Spatial autocorrelation** | Mean abs correlation at lags 1-5 | <0.05 = no spatial structure |
| **Power spectrum flatness** | Geometric mean / arithmetic mean of PSD | >0.8 = white noise (flat spectrum) |
| **Block variance uniformity** | CV of variance across 8x8 blocks | <0.3 = uniform noise (no structured regions) |
| **Distribution normality** | KS test vs Laplacian fit | p>0.05 = consistent with Laplacian noise |
| **Run-length entropy** | Entropy of run lengths of above/below median | Near theoretical max = random |
| **JPEG artifact energy** | FFT peak ratios at 1/8 and 1/32 freq | >1.5 = JPEG artifacts present |

### 1.2 Composite score

Combine into a single 0-1 "noise fraction" score:
- 1.0 = pure incompressible noise
- 0.0 = fully structured/compressible signal
- The individual metrics serve as diagnostics; the composite enables A/B comparison

### 1.3 Implementation

```
Script: evals/scripts/eval_noise_floor.py
Input:  One or more residual images (grayscale JPEG or PNG)
Output: JSON report + optional plots
```

Run on the existing residual images in `evals/runs/` (any run with `--debug-images` has `compress/` dir with L0 residuals).

### 1.4 Validation

- **Positive control**: Generate synthetic white Laplacian noise (same std as real residuals) → should score ~1.0
- **Negative control**: Use an actual photograph → should score ~0.0-0.3
- **Calibration**: Run on 10+ real residual images to establish baseline distribution

---

## Phase 2: JXL Quality Sweep as Morphological Decomposition

**Purpose**: Use JXL compression at decreasing quality levels as a probe to peel the residual into layers. Each quality step removes a layer — classify each layer as structure, noise, or artifact using correlation with the source images.

### 2.1 Core idea

As JXL quality decreases on a residual:
- **Noise** disappears first (random, incompressible — JXL drops it immediately)
- **Structure** persists longest (edges, spatially correlated)
- **JPEG artifacts** behave distinctly — periodic, so they persist at certain quality levels then vanish sharply

The **differential between adjacent quality levels** tells you what each step removed:

```python
quality_levels = [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]

R = {}  # roundtrip results
for q in quality_levels:
    R[q] = jxl_roundtrip(residual, quality=q)

# What each quality step removed
for i in range(len(quality_levels) - 1):
    q_hi = quality_levels[i]
    q_lo = quality_levels[i + 1]
    delta = R[q_hi] - R[q_lo]  # what this step removed

    # Classify delta:
    noise_floor_score(delta)                        # incompressible? → noise
    correlate(abs(delta), missed_edges)              # correlated with missed edges? → structure
    fft_peak_ratio(delta, freq=1/32)                # periodic at 32px? → JPEG artifact
```

### 2.2 Three-signal correlation (the key analysis)

At encode time, we have structural knowledge to classify every pixel of the residual:

```python
from skimage.filters import sobel

# What the prediction got wrong — edges present in original but not prediction
original_edges = sobel(original_Y)
prediction_edges = sobel(prediction_Y)
missed_edges = np.clip(original_edges - prediction_edges, 0, None)

# For each quality level's residual, measure correlation with missed edges
residual_centered = R[q] - 128
pixel_correlation = correlate(abs(residual_centered), missed_edges)
# High correlation = structure (real correction signal)
# Low correlation = noise or artifacts
```

This creates a **per-pixel confidence map**: "this residual value is real correction" vs "this is noise." More powerful than blind denoising because we're using the actual source images.

### 2.3 JXL encoder statistics as analysis tool

libjxl's `JxlEncoderStats` API provides an aggregate bit budget breakdown after encoding. At each quality level, collect:

**Bit budget breakdown** — where the bits went:

| Stat key | What it tells us |
|----------|-----------------|
| `AC_BITS` | Bulk of content bits — if this dominates at low quality, structure is present |
| `DC_BITS` | Low-frequency content — should be minimal for residuals |
| `QUANT_BITS` | Quantization overhead — grows if encoder needs spatially varying quantization |
| `NOISE_BITS` | Noise model bits — if JXL allocates bits here, it detected noise |
| `CONTROL_FIELDS_BITS` | Includes quant map + AC strategy map — overhead for spatial adaptation |
| `MODULAR_TREE_BITS` | MA tree complexity (modular mode) |

**Block size distribution** — what DCT sizes the encoder chose:

| Stat key | Interpretation |
|----------|---------------|
| `NUM_DCT8_BLOCKS` | 8x8 — default, noise-like regions |
| `NUM_DCT16_BLOCKS` | 16x16 — encoder found medium-scale structure |
| `NUM_DCT32_BLOCKS` | 32x32 — encoder found large-scale structure |
| `NUM_DCT64_BLOCKS` | 64x64 — very coherent regions |
| `NUM_SMALL_BLOCKS` | 4x4/4x8 — fine detail or high-frequency edges |

**Key signal**: If block sizes skew heavily 8x8 (the default), JXL isn't finding exploitable structure — confirming noise-like content. If larger blocks appear, those regions have compressible structure.

### 2.4 JXL debug image callback (optional, requires debug build)

`JxlEncoderSetDebugImageCallback()` provides:
- **AC strategy color map**: per-region visualization of which DCT block size was chosen. Large blocks = structure, small blocks = noise. This IS the structure/noise map.
- **Butteraugli heatmaps**: per-iteration perceptual distortion maps.
- Requires building libjxl from source with `JXL_DEBUG_ADAPTIVE_QUANTIZATION` defined.

If building a debug libjxl is tractable, the AC strategy map is the most direct answer to "where is structure vs noise." **JXL has already solved this problem internally — we just need to extract its answer.**

### 2.5 Using the stats API from Python

The stats API is a C API. Access options:
1. **Subprocess `cjxl`**: No stats output available via CLI (no verbose flag for this)
2. **Python ctypes/cffi wrapper**: Call `JxlEncoderCollectStats()` + `JxlEncoderStatsGet()` directly
3. **Build a small C tool**: `jxl_analyze_residual` that encodes and prints stats JSON
4. **`imagecodecs` Python package**: May expose stats (check API)

Recommendation: build a small C CLI tool `tools/jxl_stats` that takes an image + quality, encodes with libjxl, and prints stats as JSON. ~100 lines of C. This is reusable for all experiments.

### 2.6 JXL encoder settings for analysis

For **production compression** of residuals, JXL's defaults were already tested and found optimal. But for **analysis** — using JXL as a morphological probe — we want to control its behavior so each layer is cleanly interpretable.

Run the sweep in two configurations:

**Config A: Defaults** (baseline, matches production behavior)
```
cjxl -q <quality> residual.png output.jxl
```

**Config B: Analysis mode** (disable heuristics that assume natural image statistics)
```
cjxl -q <quality> --epf=0 --gaborish=0 --patches=0 --dots=0 residual.png output.jxl
```
Plus `DISABLE_PERCEPTUAL_HEURISTICS=1` via the C API in `tools/jxl_stats.c`.

Rationale for each:
- `--epf=0` — edge-preserving filter smooths boundaries between structure/noise regions, muddying the layer deltas
- `--gaborish=0` — deblocking filter designed for natural images, not centered-at-128 residuals
- `--patches=0`, `--dots=0` — pattern-matching heuristics that would fire on JPEG artifacts or random texture, polluting analysis
- `DISABLE_PERCEPTUAL_HEURISTICS` — Butteraugli masking assumes natural image statistics; with it on, JXL might preserve noise in "important" regions and drop structure in "unimportant" ones

**Config C: Modular mode** (optional, different decomposition)
```
cjxl -q <quality> --modular residual.png output.jxl
```
Modular's MA tree prediction may capture spatial correlations differently than VarDCT, giving a second perspective on where structure lives.

Comparing A vs B reveals how much JXL's perceptual model affects residual compression. Comparing A vs C reveals whether the residual is better modeled by VarDCT or prediction-tree approaches.

### 2.7 Experiment matrix

For each quality level q in [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]:

| Metric | Purpose |
|--------|---------|
| JXL compressed size (bytes) | Rate-distortion curve |
| JXL stats: bit budget breakdown | Where bits go at each quality |
| JXL stats: block size distribution | Structure vs noise regions |
| Roundtrip PSNR vs original residual | What's preserved |
| Delta image (R[q+10] - R[q]) | What this step removed |
| noise_floor_score(delta) | Was the removed content noise-like? |
| correlation(delta, missed_edges) | Was it real correction signal? |
| fft_peak_ratio(delta, 1/32) | Was it JPEG artifact? |
| Reconstruction PSNR (pred + R[q] vs original) | End-to-end quality |

### 2.8 Expected outcome

A curve showing:
- At high quality (90-100): tiny deltas, mostly noise → noise-floor score near 1.0
- At medium quality (50-70): deltas start containing structure → edge correlation rises
- At low quality (20-40): significant structure loss → reconstruction PSNR drops
- JPEG artifacts: should show up as a spike in FFT peak ratio at specific quality levels

The **optimal operating point** is where noise-floor score of the delta is still >0.85 but reconstruction PSNR hasn't dropped more than ~1 dB. This quality level is the natural structure/noise boundary.

---

## Phase 3: Structure/Noise Separation Methods

**Purpose**: Once Phase 2 identifies the structure/noise boundary, implement targeted separation methods and compare against the JXL sweep baseline.

### 3.1 Method A: Guided filter (prediction as guide)

```python
import cv2

prediction = lanczos3_upsample(L2_Y, 4)   # or SR model prediction
residual = ground_truth_Y - prediction + 128

# Guided filter: preserves edges aligned with prediction, smooths everything else
structure = cv2.guidedFilter(
    guide=prediction,       # uint8, the prediction image
    src=residual,           # uint8, the residual
    radius=r,               # window radius (try 2, 4, 8, 16)
    eps=eps                 # regularization (try 1e-2, 1e-1, 1, 10, 100)
)
noise = residual - structure + 128  # re-center
```

**Why guided filter**: Edges in the residual that correlate with edges in the prediction are real correction signal. Texture uncorrelated with the prediction is likely noise. The prediction is a free guide signal — we already have it at both encode and decode time.

Sweep `(radius, eps)` and measure noise-floor score of the noise component.

### 3.2 Method B: Edge-correlation mask

Use the three-signal correlation from Phase 2 to build a per-pixel mask:

```python
missed_edges = np.clip(sobel(original_Y) - sobel(prediction_Y), 0, None)
mask = (missed_edges > threshold).astype(float)
mask = cv2.GaussianBlur(mask, (5, 5), 1.0)  # soft edges

structure = residual_centered * mask + 128
noise = residual_centered * (1 - mask) + 128
```

This is the most direct use of our encode-time knowledge. Only works at encode time (need original), but that's fine — we only need to separate at encode time.

### 3.3 Method C: Wavelet thresholding

```python
import pywt

coeffs = pywt.wavedec2(residual_centered, 'db4', level=2)
threshold = sigma * sqrt(2 * log(N))  # Universal threshold (VisuShrink)
coeffs_denoised = [coeffs[0]]
for detail in coeffs[1:]:
    coeffs_denoised.append(tuple(
        pywt.threshold(d, threshold, mode='soft') for d in detail
    ))
structure = pywt.waverec2(coeffs_denoised, 'db4') + 128
noise = residual - structure + 128
```

### 3.4 Evaluation matrix

For each method × parameter combination:

| Metric | Target |
|--------|--------|
| Noise component noise-floor score | >0.85 |
| Structure JPEG compressed size | <70% of original residual JPEG size |
| Reconstruction PSNR (structure only) | Within 1 dB of full residual |
| Reconstruction SSIM (structure only) | Within 0.005 of full residual |
| Reconstruction LPIPS (structure only) | Within 0.01 of full residual |
| Edge preservation correlation | >0.9 (structure edges ∝ original missed edges) |

### 3.5 Edge-correlation validation

```python
from skimage.filters import sobel

pred_edges = sobel(prediction)
orig_edges = sobel(original_Y)
missed_edges = np.clip(orig_edges - pred_edges, 0, None)
struct_edges = sobel(structure - 128)

# Structure should correlate with missed edges, not all edges
correlation = np.corrcoef(missed_edges.ravel(), struct_edges.ravel())[0, 1]
```

If correlation is low (<0.3), the separation method is keeping random texture as "structure."

---

## Phase 4: JPEG Artifact Decontamination

**Purpose**: Remove the periodic JPEG block grid artifacts from residuals, either before or as part of separation.

### 4.1 Why this matters

The 8x8 block grid from L2 JPEG compression bleeds through the 4x upsample as a period-32 pattern in the L0 residual. This is:
- Not real tissue information
- Not prediction error
- Periodic structure that inflates the "structure" component
- Responsible for some of the 0.29 bpp spatial redundancy

### 4.2 Approach: Notch filter in frequency domain

```python
F = np.fft.fft2(residual_centered)

# Notch narrow bands at JPEG artifact frequencies
for freq in [1/32, 1/16, 3/32, 1/8]:  # artifact harmonics in L0 space
    notch_mask = create_notch(F.shape, freq, bandwidth=2)
    F *= notch_mask

residual_clean = np.real(np.fft.ifft2(F)) + 128
```

### 4.3 Alternative: Raise L2 quality

Encode L2 at higher quality (Q97-Q99) so there's less artifact to bleed through. We already know Q97 is the sweet spot for quality/size. Measure residual artifact energy at Q95 vs Q97 vs Q99.

### 4.4 Measure artifact removal

Run noise-floor eval before and after. Decontamination should:
- Reduce autocorrelation at lag-8 and lag-32
- Flatten FFT peaks at 1/8 and 1/32
- Reduce the "structure" component size after separation

---

## Phase 5: SR Model Retraining on Denoised Targets

**Purpose**: If Phase 3 successfully separates structure from noise, retrain the SR model to predict denoised ground truth. The predictor stops wasting capacity predicting noise.

### 5.1 Hypothesis

Current SR model minimizes `loss(SR(L2), ground_truth)`. The ground truth contains noise that the SR model tries (and partially fails) to reproduce. This wastes model capacity.

If we train on `loss(SR(L2), denoised_ground_truth)`:
- The model focuses on predicting structure
- Predictions may be smoother but more accurate where it matters
- Residuals (vs original ground truth) may be MORE compressible because the prediction no longer introduces false noise-like patterns

### 5.2 Experiment design

```
1. Take existing TCGA training tiles (5K stage 1, 50K stage 2)
2. For each tile:
   a. L2 = downsample 4x + JPEG compress
   b. prediction = lanczos3_upsample(L2, 4)
   c. structure = best_method(ground_truth_Y, prediction, residual)
   d. denoised_target = prediction + (structure - 128)
3. Train SR model variants:
   - SR-orig: trained on original ground truth (baseline)
   - SR-denoised: trained on denoised targets
   - SR-mixed: MSE on denoised + small perceptual loss on original (hedge)
4. For each variant, measure:
   - Prediction PSNR/SSIM vs original ground truth
   - Residual compressibility (JPEG size at Q50)
   - End-to-end quality (prediction + residual vs ground truth)
```

### 5.3 Expected outcome

SR-denoised may have *lower* PSNR on prediction alone (not trying to predict noise), but residuals should be more compressible. End-to-end quality (after adding residuals back) should be similar or better at smaller total file size.

### 5.4 Risk

If the "noise" contains perceptually important micro-texture, SR-denoised predictions look over-smoothed and residuals need to carry that texture back. Net effect: no improvement. This is why Phase 1 (noise-floor eval) and Phase 3 (separation quality metrics) must be solid first.

---

## Phase 6 (Optional): Noise Synthesis at Decode Time

**Purpose**: If discarding noise causes perceptual degradation, synthesize statistically equivalent noise at decode time.

### 6.1 Approach

```
Encode-side: store noise statistics per family (std, distribution params) — ~10 bytes
Decode-side: generate Laplacian noise with matching statistics, add to reconstruction
```

### 6.2 When to skip

If discarding noise has <0.5 dB PSNR impact and LPIPS doesn't meaningfully change, synthesis is unnecessary overhead. Just drop the noise.

---

## Visual Inspection & Asset Retention

Every experiment must produce visual assets for qualitative review alongside the statistical analysis. Numbers can mislead — especially for pathology where "does this still look like tissue?" matters more than PSNR.

### Output structure

Each experiment run writes to a structured directory under `evals/analysis/noise_floor/`:

```
evals/analysis/noise_floor/
  {experiment_id}/                      # e.g. "jxl_sweep_L0-1024_configA"
    metadata.json                       # experiment params, timestamp, source image
    metrics.json                        # all computed metrics (noise-floor scores, PSNR, etc.)

    sources/                            # input images (retained for side-by-side)
      original_Y.png                    # ground truth luma
      prediction_Y.png                  # lanczos3/SR prediction
      residual.png                      # original - prediction + 128
      missed_edges.png                  # sobel(original) - sobel(prediction)

    jxl_sweep/                          # Phase 2: per-quality-level assets
      q100/
        roundtrip.png                   # JXL roundtrip result
        delta.png                       # what this quality step removed (vs q_above)
        delta_heatmap.png               # abs(delta) as heatmap, hot = large removal
        reconstruction.png              # prediction + (roundtrip - 128) = end-to-end result
        edge_correlation_map.png        # per-pixel correlation of delta with missed_edges
        stats.json                      # JXL encoder stats (bit budget, block sizes)
      q95/
        ...
      q90/
        ...
      (etc)
      summary_strip.png                 # horizontal strip: residual | q90 | q70 | q50 | q30
      rd_curve.png                      # rate-distortion curve across quality levels
      layer_classification.png          # per-quality: noise score, edge correlation, artifact FFT

    separation/                         # Phase 3: per-method assets
      guided_r4_eps10/
        structure.png                   # structure component
        noise.png                       # noise component
        structure_heatmap.png           # abs(structure - 128) as heatmap
        noise_heatmap.png               # abs(noise - 128) as heatmap
        reconstruction.png             # prediction + (structure - 128)
        edge_overlay.png                # structure edges overlaid on original
        metrics.json                    # noise-floor score, compression ratio, PSNR, etc.
      guided_r8_eps100/
        ...
      edge_mask_t0.1/
        ...
      wavelet_db4_l2/
        ...
      comparison_strip.png              # horizontal: residual | struct_A | noise_A | struct_B | ...

    artifacts/                          # Phase 4: decontamination assets
      before_fft.png                    # power spectrum before
      after_fft.png                     # power spectrum after
      removed_artifacts.png             # what the notch filter removed
      residual_clean.png                # decontaminated residual
```

### Image format conventions

- **All intermediate images saved as PNG** (lossless) — JPEG would add its own artifacts and confuse the analysis
- **Heatmaps**: use a perceptually uniform colormap (viridis or inferno), include colorbar with value range
- **Delta images**: save both the raw centered delta (for computation) and an amplified visualization (e.g. 4x contrast) for human review
- **Summary strips**: fixed-height horizontal concatenation with labels, suitable for quick comparison
- **Edge overlays**: structure edges in red/cyan overlaid on grayscale original at 50% opacity

### Viewer page: `evals/viewer/public/denoise.html`

Add a new page to the existing evals viewer for browsing experiment results. The viewer server already serves static files from `evals/viewer/public/` and has API endpoints for run discovery.

**Layout:**

```
┌─────────────────────────────────────────────────────┐
│ ORIGAMI  │  Runs  │  Comparison  │  Denoise ◄active │
├──────────┴────────┴─────────────┴───────────────────┤
│                                                     │
│  Experiment: [dropdown of experiment_ids]            │
│                                                     │
│  ┌─── Source Images ────────────────────────────┐   │
│  │ Original │ Prediction │ Residual │ Missed    │   │
│  │          │            │          │ Edges     │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌─── JXL Quality Sweep ───────────────────────┐   │
│  │ Quality: [slider: 10 ──── 50 ──── 100]      │   │
│  │                                              │   │
│  │ Roundtrip  │  Delta    │  Reconstruction     │   │
│  │            │ (removed) │  (pred + structure)  │   │
│  │            │           │                      │   │
│  │ Edge Correlation Map   │  Delta Heatmap       │   │
│  │                        │                      │   │
│  │ ── Metrics at this quality ──                │   │
│  │ Noise score: 0.92  Edge corr: 0.12           │   │
│  │ Artifact FFT: 1.02  PSNR: 38.2 dB           │   │
│  │ JXL size: 42 KB  Block dist: 85% 8x8        │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌─── Separation Methods ──────────────────────┐   │
│  │ Method: [tabs: Guided | Edge Mask | Wavelet] │   │
│  │ Params: [dropdown of param combos]           │   │
│  │                                              │   │
│  │ Structure  │  Noise     │  Reconstruction    │   │
│  │            │            │                     │   │
│  │ Edge Overlay (structure on original)          │   │
│  │                                              │   │
│  │ ── Metrics ──                                │   │
│  │ Noise score: 0.88  Struct size: 58 KB (-34%) │   │
│  │ Recon PSNR: 37.8 dB  Edge corr: 0.91        │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌─── Charts ──────────────────────────────────┐   │
│  │ [RD Curve] [Layer Classification] [Summary]  │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌─── Notes ───────────────────────────────────┐   │
│  │ [editable text area for qualitative notes]   │   │
│  │ [Save] — persists to metadata.json           │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Key interactions:**
- **Quality slider**: scrub through JXL sweep levels, images update in real-time
- **Zoom sync**: all images in a row zoom/pan together (click to toggle full-size)
- **A/B flip**: click any image to toggle between it and the original (instant visual diff)
- **Notes field**: free-text area saved to `metadata.json` for documenting observations during review

### Viewer server additions

Add to `evals/viewer/viewer-server.js`:

```javascript
// API: list denoise experiments
app.get('/api/denoise/experiments', (req, res) => {
  // Scan evals/analysis/noise_floor/ for experiment directories
  // Return list of { id, timestamp, source_image, params }
});

// API: get experiment data
app.get('/api/denoise/experiment/:id', (req, res) => {
  // Return metadata.json + metrics.json
});

// API: save notes
app.post('/api/denoise/experiment/:id/notes', (req, res) => {
  // Update metadata.json with notes field
});

// Static: serve experiment image assets
app.use('/denoise-assets', express.static(
  path.join(__dirname, '../analysis/noise_floor')
));
```

### Documentation during analysis

Each experiment's `metadata.json` includes:

```json
{
  "experiment_id": "jxl_sweep_L0-1024_configA",
  "timestamp": "2026-03-11T14:30:00Z",
  "source_image": "evals/test-images/L0-1024.jpg",
  "params": {
    "jxl_config": "A",
    "quality_levels": [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    "predictor": "lanczos3"
  },
  "notes": "Structure visible at tissue boundaries even at Q30. Noise removal at Q90 looks clean — no visible structure loss in stroma regions. JPEG artifacts most visible in delta at Q70-80 range.",
  "conclusions": []
}
```

Scripts append to `conclusions` as analysis proceeds, creating a traceable record of what was tried, what was observed, and what was decided.

---

## Execution Order

1. **Phase 1** — build noise-floor eval. Everything depends on this.
2. **Phase 2** — JXL quality sweep with stats collection + three-signal correlation. This is the primary experiment — it simultaneously answers "is separation possible?" and "where is the structure/noise boundary?" using JXL's own internal analysis.
3. **Phase 3** — guided filter and edge-correlation methods, compared against Phase 2's JXL boundary.
4. **Phase 4** — JPEG artifact decontamination, can run in parallel with Phase 3.
5. **Phase 5** — SR model retraining, only if Phase 3 finds ≥30% size reduction.
6. **Phase 6** — noise synthesis, only if noise removal causes perceptual degradation.

## Build Dependency: JXL Stats Tool

Phase 2 requires access to `JxlEncoderStats`. Options in order of preference:

1. **Small C CLI tool** (`tools/jxl_stats.c`, ~100 lines): takes image path + quality, encodes with libjxl, prints stats as JSON. Build against system libjxl (`brew install jpeg-xl`). Most reliable, reusable.
2. **Python ctypes wrapper**: call libjxl directly from Python. More complex, fragile across platforms.
3. **Stats-free fallback**: if building the C tool is impractical, skip the per-quality bit budget breakdown and rely only on the roundtrip + three-signal correlation analysis. Still valuable, just less insight into JXL's internal decisions.

Optional: **Debug build of libjxl** for the AC strategy map (visual block size map). Only worth it if the stats API doesn't give enough information. Build libjxl from source with `-DJXL_DEBUG_ADAPTIVE_QUANTIZATION=ON`.

## Files to Create

| File | Purpose |
|------|---------|
| `evals/scripts/eval_noise_floor.py` | Phase 1: noise-floor evaluation framework |
| `evals/scripts/jxl_quality_sweep.py` | Phase 2: JXL sweep + three-signal correlation + asset generation |
| `tools/jxl_stats.c` | Phase 2: C tool to extract JXL encoder stats as JSON |
| `evals/scripts/separate_residual.py` | Phase 3: guided filter + edge-correlation separation |
| `evals/scripts/decontaminate_artifacts.py` | Phase 4: JPEG artifact removal |
| `evals/viewer/public/denoise.html` | Viewer page for visual inspection of experiments |
| `evals/analysis/noise_floor/` | Output directory for all experiment results + assets |

## Dependencies

- numpy, scipy (stats, signal, fft)
- scikit-image (SSIM, Sobel edge detection, filters)
- pywt (wavelets, for Method C)
- opencv-python (guided filter, bilateral filter)
- lz4 (incompressibility test)
- pillow, matplotlib (I/O, plots)
- lpips, torch (perceptual quality, already installed)
- libjxl (system install via `brew install jpeg-xl`, or build from source for debug images)
- `cjxl`/`djxl` CLI tools (for roundtrip compression in the sweep)

## Input Data

Residual images from any `origami encode --debug-images` run. Located at:
```
evals/runs/{run_name}/compress/{x2}_{y2}_l0_residual.jpg
```

The `--debug-images` flag also writes `original_Y` and `prediction_Y` images needed for three-signal correlation. Verify these exist; if not, the sweep script needs to regenerate them from the L2 + ground truth.

For quick iteration, use the single-image encode output:
```
origami encode --image evals/test-images/L0-1024.jpg --out /tmp/denoise_test \
    --baseq 95 --l0q 50 --subsamp 444 --manifest --debug-images --pack
```
