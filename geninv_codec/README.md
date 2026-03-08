# geninv_codec — base codec + luma residual vs latent-predicted residual (with correction)

This repo is a **prototype harness** for a hybrid compression idea:

- Use a standard codec to create a **base image** `B` (currently JPEG q=50 as a stand-in for JPEG XL).
- Work in **luminance only** (Y / Rec.601). Ignore chroma.
- Compute the **true luma residual** at encode time:
  - `R_true = Y(orig) - Y(base)`
- Compare two correction paths:

## A) True residual path (baseline)
1. Quantize residual to signed int8-ish values in units of 1/255:
   - `Rq = clamp(round(R_true * 255), -127..127)`
2. Convert to a “centered residual image” for easy compression:
   - `Ru8 = Rq + 128` in `[0..255]`
3. Store `Ru8` as **JPEG90 / JPEG95** (apples-to-apples request).
4. Decode residual JPEG back, de-center, dequantize, and apply:
   - `Y_hat = clip(Y_base + R_decoded, 0..1)`

This measures “how big is it to store the residual map” and how well it corrects after JPEG90/95 compression of the residual.

## B) Latent-predicted residual path (generator inversion harness)
Here we replace the residual image with a compact latent payload.

### Model
We train a tiny toy model:

- Encoder `E(B, R) -> z` (amortized inversion)
- Generator `G(B, z) -> R_hat` (predicted residual)

### Encode-time
At encode time we know `R_true`, so we compute a latent per patch:
- split image into patches (default **64×64**)
- for each patch:
  - `z = E(B_patch, R_true_patch)`
  - `R_hat_patch = G(B_patch, z)`

### “Bitstream” for the prototype
We **pack the latents** into a portable payload:
- quantize `z` to int8 with a **single global scale** (`maxabs/127`)
- pack as:
  - `[4-byte header length][header JSON][int8 latent bytes...]`
- we record both:
  - raw payload bytes
  - zlib-compressed payload bytes (rough proxy for entropy coding)

### Decode-time
Given `B` and the packed latents, the decoder can regenerate `R_hat` and apply:
- `Y_hat_latent = clip(Y_base + R_hat, 0..1)`

### Optional correction layer (“residual-of-residual”)
To measure how wrong the predicted residual is, we compute:
- `Corr = R_true - R_hat`
Then quantize/store Corr exactly like the true residual map, using **JPEG90/95**:
- `Y_hat_latent_corr = clip(Y_base + R_hat + Corr_decoded, 0..1)`

This gives you:
- an error map (Corr)
- a compressed correction size
- and an apples-to-apples comparison versus storing the full residual

---

# Files

## 1) Train a portable model checkpoint
`train_model.py` trains `E` + `G` and saves a single file checkpoint you can reuse.

Example (train on one image):
```bash
python train_model.py --data L0-1024.jpg --base_quality 50 --patch 64 --zdim 8 --ch 64 --steps 5000 --out model.pt
```

Example (train on a directory of tiles):
```bash
python train_model.py --data ./tiles/ --base_quality 50 --patch 64 --zdim 16 --ch 64 --steps 30000 --out model_tiles.pt
```

The checkpoint (`.pt`) contains:
- `config` (patch/zdim/ch)
- `E_state` and `G_state` `state_dict`s

## 2) Run the full report on a single image
`run_full_report.py` produces the full “A vs B (+ correction)” report folder, with images, CSV, JSON, montage, and a zip.

```bash
python run_full_report.py --image L0-1024.jpg --model model.pt --outdir report_out --base_quality 50 \
  --residual_jpeg 90 95 --corr_jpeg 90 95
```

### Outputs (report_out/)
- `base_codec_q50.jpg` + `orig_rgb.png` + `base_rgb.png`
- `Y_orig.png`, `Y_base.png`
- `true_residual_centered_JPEG90.jpg`, `true_residual_centered_JPEG95.jpg`
- `pred_residual_visual.png` (visualization only)
- `latent_payload_raw.bin`, `latent_payload_zlib.bin`
- `correction_centered_JPEG90.jpg`, `correction_centered_JPEG95.jpg`
- reconstructed images:
  - `Y_true_residual_applied_JPEG95.png`
  - `Y_latent_only.png`
  - `Y_latent_plus_corr_JPEG95.png`
- error visualizations (centered at 128 with gain):
  - `err_base_vis.png`, `err_latent_only_vis.png`, etc.
- `results.csv` (bytes + PSNR/MAE)
- `metrics.json` (summary)
- `montage.png`
- `report.zip`

---

# Next step: swap JPEG base for JPEG XL
Right now `jpeg_encode_decode_rgb()` is the base codec. Swap it for JPEG XL by replacing that function with a call to your `cjxl/djxl` pipeline and keeping everything else identical.

---

# Notes & caveats
- This is **not intended to be lossless**. It’s a harness to test whether a **tiny latent** can approximate a luma residual well enough that you need little/no correction.
- The current model is tiny and will underperform if trained on one image. Train on many tiles/WSI patches for meaningful results.
- The `E(B,R)` step uses the true residual at encode time. That’s correct for a codec experiment: the encoder has the target; the decoder does not.
