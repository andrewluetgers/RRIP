#!/usr/bin/env python3
"""
parametric_correction_experiment.py

What if we embed a small number of parameters into L2 that describe a correction
function? Instead of storing raw residuals, we store a compact model that
reduces prediction error.

We test several models with different parameter counts:

1. Per-tile bias (4 params = 4 bytes): global brightness correction per L1 tile
2. Per-tile affine (12 params = 12 bytes): gain + bias per tile
3. Per-tile polynomial (36 params): quadratic spatial correction per tile
4. Low-rank DCT correction (N coefficients): store the N largest DCT coefficients
   of the residual field — this is literally what JPEG does, but we pick the
   coefficients optimally rather than using JPEG's fixed scanning order
5. Block-level gain map (1024 params = 1 KB): per-8x8-block gain correction
6. Wavelet/frequency band corrections

For each, we measure:
- How much residual energy is reduced
- How this translates to PSNR improvement
- How many bytes are needed
- Whether it fits in QIM embedding capacity

Usage:
  python parametric_correction_experiment.py --image evals/test-images/L0-1024.jpg
"""
import argparse
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
import io
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_bytes


def rgb_to_ycbcr_bt601(rgb_u8):
    rgb = rgb_u8.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128.0
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 128.0
    return Y, Cb, Cr


def ycbcr_to_rgb_bt601(Y, Cb, Cr):
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0
    R = Y + 1.402*Cr
    G = Y - 0.344136*Cb - 0.714136*Cr
    B = Y + 1.772*Cb
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)


def psnr(a, b, data_range=255):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range**2 / mse)


def encode_decode_gray(pixels_u8, quality, encoder=JpegEncoder.LIBJPEG_TURBO):
    img = Image.fromarray(pixels_u8, mode="L")
    data = encode_jpeg_to_bytes(img, quality, encoder)
    decoded = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return decoded, len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--quality", type=int, default=40)
    args = parser.parse_args()

    # Load and tile
    img = Image.open(args.image).convert("RGB")
    img_array = np.array(img)
    tile_size = 256
    UPSAMPLE = Image.Resampling.BILINEAR

    l2 = np.array(Image.fromarray(img_array[:1024, :1024]).resize((256, 256), Image.LANCZOS))
    l1_source = np.array(Image.fromarray(img_array[:1024, :1024]).resize((512, 512), Image.LANCZOS))

    l1_tiles = {}
    for dy in range(2):
        for dx in range(2):
            l1_tiles[(dx, dy)] = l1_source[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    l0_tiles = {}
    for dy in range(4):
        for dx in range(4):
            l0_tiles[(dx, dy)] = img_array[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    # Get L1 predictions and residuals
    l1_pred_mosaic = np.array(Image.fromarray(l2).resize((512, 512), resample=UPSAMPLE))

    print(f"{'='*70}")
    print(f"PARAMETRIC CORRECTION EXPERIMENT")
    print(f"{'='*70}")

    # Collect all L1 residuals
    l1_residuals = {}
    l1_preds = {}
    l1_Y_gts = {}
    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)
        l1_residuals[(dx, dy)] = Y_gt - Y_pred  # float, centered at 0
        l1_preds[(dx, dy)] = (Y_pred, Cb_pred, Cr_pred)
        l1_Y_gts[(dx, dy)] = Y_gt

    # Print baseline residual stats
    print(f"\n--- Baseline L1 residual statistics ---")
    for (dx, dy), res in l1_residuals.items():
        print(f"  L1({dx},{dy}): energy={np.sum(res**2):.0f}, std={res.std():.2f}, "
              f"abs_mean={np.abs(res).mean():.2f}, range=[{res.min():.0f}, {res.max():.0f}]")

    total_baseline_energy = sum(np.sum(r**2) for r in l1_residuals.values())
    print(f"  Total energy: {total_baseline_energy:.0f}")

    # ===== TEST CORRECTION MODELS =====

    models = {}

    # Model 0: No correction (baseline)
    models["baseline"] = {
        "params": 0,
        "bytes": 0,
        "residual_energy": total_baseline_energy,
        "corrections": {k: np.zeros_like(v) for k, v in l1_residuals.items()},
    }

    # Model 1: Per-tile DC bias
    # For each L1 tile, compute optimal bias (= mean of residual)
    corrections = {}
    n_params = 0
    for (dx, dy), res in l1_residuals.items():
        bias = np.mean(res)
        corrections[(dx, dy)] = np.full_like(res, bias)
        n_params += 1

    corrected_energy = sum(np.sum((l1_residuals[k] - corrections[k])**2) for k in l1_residuals)
    models["per-tile bias"] = {
        "params": n_params,
        "bytes": n_params * 1,  # 1 byte per bias (int8, range ±127)
        "residual_energy": corrected_energy,
        "corrections": corrections,
    }

    # Model 2: Per-tile affine (gain * prediction + bias)
    # Y_corrected = alpha * Y_pred + beta → residual_corrected = Y_gt - Y_corrected
    # Optimal alpha, beta via least squares: Y_gt ≈ alpha * Y_pred + beta
    corrections = {}
    n_params = 0
    for (dx, dy), res in l1_residuals.items():
        Y_pred = l1_preds[(dx, dy)][0]
        Y_gt = l1_Y_gts[(dx, dy)]
        # Solve: Y_gt = alpha * Y_pred + beta
        A = np.column_stack([Y_pred.ravel(), np.ones(Y_pred.size)])
        b = Y_gt.ravel()
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        alpha, beta = params
        correction = (alpha - 1.0) * Y_pred + beta
        corrections[(dx, dy)] = correction
        n_params += 2

    corrected_energy = sum(np.sum((l1_residuals[k] - corrections[k])**2) for k in l1_residuals)
    models["per-tile affine"] = {
        "params": n_params,
        "bytes": n_params * 2,  # 2 bytes per param (float16)
        "residual_energy": corrected_energy,
        "corrections": corrections,
    }

    # Model 3: Per-tile quadratic spatial
    # Y_correction(x,y) = a*x^2 + b*y^2 + c*xy + d*x + e*y + f
    corrections = {}
    n_params = 0
    for (dx, dy), res in l1_residuals.items():
        h, w = res.shape
        yy, xx = np.mgrid[0:h, 0:w]
        xx_n = xx.ravel() / w  # normalize to [0,1]
        yy_n = yy.ravel() / h
        A = np.column_stack([xx_n**2, yy_n**2, xx_n*yy_n, xx_n, yy_n, np.ones_like(xx_n)])
        b = res.ravel()
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        correction = A @ params
        corrections[(dx, dy)] = correction.reshape(h, w)
        n_params += 6

    corrected_energy = sum(np.sum((l1_residuals[k] - corrections[k])**2) for k in l1_residuals)
    models["per-tile quadratic"] = {
        "params": n_params,
        "bytes": n_params * 2,
        "residual_energy": corrected_energy,
        "corrections": corrections,
    }

    # Model 4: Low-rank DCT of full L1 residual field
    # Concatenate all 4 residuals into a 512x512 field, take DCT, keep top N coefficients
    residual_field = np.zeros((512, 512), dtype=np.float64)
    for (dx, dy), res in l1_residuals.items():
        residual_field[dy*256:(dy+1)*256, dx*256:(dx+1)*256] = res

    # Full DCT of residual field
    dct_full = dctn(residual_field, type=2, norm='ortho')

    for n_keep in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        # Keep top N coefficients by magnitude
        flat = dct_full.ravel()
        indices = np.argsort(np.abs(flat))[::-1]
        top_n = indices[:n_keep]

        # Reconstruct from top N
        sparse = np.zeros_like(flat)
        sparse[top_n] = flat[top_n]
        reconstruction = idctn(sparse.reshape(512, 512), type=2, norm='ortho')

        # Compute corrected residual energy
        corrected_energy = 0
        corrections = {}
        for (dx, dy), res in l1_residuals.items():
            corr = reconstruction[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
            corrections[(dx, dy)] = corr
            corrected_energy += np.sum((res - corr)**2)

        # Storage: each coefficient needs position (20 bits for 512x512 → 3 bytes) + value (2 bytes float16) = 5 bytes
        storage = n_keep * 5

        models[f"DCT top-{n_keep}"] = {
            "params": n_keep * 2,  # position + value
            "bytes": storage,
            "residual_energy": corrected_energy,
            "corrections": corrections,
        }

    # Model 5: Per-8x8-block bias (same grid as JPEG)
    corrections = {}
    n_params = 0
    for (dx, dy), res in l1_residuals.items():
        h, w = res.shape
        corr = np.zeros_like(res)
        for by in range(h // 8):
            for bx in range(w // 8):
                block = res[by*8:(by+1)*8, bx*8:(bx+1)*8]
                bias = np.mean(block)
                corr[by*8:(by+1)*8, bx*8:(bx+1)*8] = bias
                n_params += 1
        corrections[(dx, dy)] = corr

    corrected_energy = sum(np.sum((l1_residuals[k] - corrections[k])**2) for k in l1_residuals)
    models["per-block bias"] = {
        "params": n_params,
        "bytes": n_params * 1,  # 1 byte per bias
        "residual_energy": corrected_energy,
        "corrections": corrections,
    }

    # Model 6: Per-8x8-block gain+bias
    corrections = {}
    n_params = 0
    for (dx, dy), res in l1_residuals.items():
        Y_pred = l1_preds[(dx, dy)][0]
        h, w = res.shape
        corr = np.zeros_like(res)
        for by in range(h // 8):
            for bx in range(w // 8):
                block_res = res[by*8:(by+1)*8, bx*8:(bx+1)*8]
                block_pred = Y_pred[by*8:(by+1)*8, bx*8:(bx+1)*8]
                # Solve: residual ≈ alpha * pred + beta
                A = np.column_stack([block_pred.ravel(), np.ones(64)])
                b = block_res.ravel()
                params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                corr[by*8:(by+1)*8, bx*8:(bx+1)*8] = (A @ params).reshape(8, 8)
                n_params += 2
        corrections[(dx, dy)] = corr

    corrected_energy = sum(np.sum((l1_residuals[k] - corrections[k])**2) for k in l1_residuals)
    models["per-block affine"] = {
        "params": n_params,
        "bytes": n_params * 2,
        "residual_energy": corrected_energy,
        "corrections": corrections,
    }

    # ===== RESULTS =====
    print(f"\n{'='*70}")
    print(f"RESULTS — Residual energy reduction from parametric correction")
    print(f"{'='*70}")
    print(f"\n{'Model':<25s} {'Params':>8s} {'Bytes':>8s} {'Energy':>12s} {'Reduction':>10s} {'Equiv PSNR':>10s}")
    print("-" * 80)

    baseline_energy = models["baseline"]["residual_energy"]
    for name, m in sorted(models.items(), key=lambda x: x[1]["bytes"]):
        reduction = 1.0 - m["residual_energy"] / baseline_energy
        # Energy reduction → PSNR improvement: if energy drops by factor F,
        # PSNR improves by 10*log10(F)
        if m["residual_energy"] > 0 and m["residual_energy"] < baseline_energy:
            psnr_gain = 10 * np.log10(baseline_energy / m["residual_energy"])
        else:
            psnr_gain = 0.0

        fits_qim = "<<" if m["bytes"] <= 128 else ("<" if m["bytes"] <= 512 else "")
        print(f"{name:<25s} {m['params']:>8d} {m['bytes']:>7d}B {m['residual_energy']:>12.0f} "
              f"{reduction:>9.1%} {psnr_gain:>8.2f}dB {fits_qim}")

    # ===== END-TO-END TEST =====
    # Pick the best model that fits in QIM capacity and run full pipeline
    print(f"\n{'='*70}")
    print(f"END-TO-END PIPELINE TEST")
    print(f"{'='*70}")

    quality = args.quality

    # Test: baseline vs best embeddable correction
    for model_name in ["baseline", "per-tile bias", "per-tile affine", "per-tile quadratic",
                        "DCT top-16", "DCT top-32", "DCT top-64", "DCT top-128"]:
        if model_name not in models:
            continue
        m = models[model_name]
        corrections = m["corrections"]

        l1_psnrs = []
        total_l1_bytes = 0
        total_l0_bytes = 0
        l1_reconstructed = {}

        for (dx, dy), l1_gt in l1_tiles.items():
            Y_pred, Cb_pred, Cr_pred = l1_preds[(dx, dy)]
            Y_gt = l1_Y_gts[(dx, dy)]
            corr = corrections[(dx, dy)]

            # Corrected residual: what we actually need to JPEG-encode
            corrected_residual = l1_residuals[(dx, dy)] - corr
            centered = np.clip(np.round(corrected_residual + 128.0), 0, 255).astype(np.uint8)
            decoded, size = encode_decode_gray(centered, quality)
            total_l1_bytes += size

            # Reconstruct: prediction + correction + decoded_residual
            decoded_residual = decoded.astype(np.float32) - 128.0
            Y_recon = np.clip(Y_pred + corr + decoded_residual, 0, 255)
            rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
            l1_reconstructed[(dx, dy)] = rgb_recon
            l1_psnrs.append(psnr(l1_gt, rgb_recon))

        # L0
        l1_mosaic = np.zeros((512, 512, 3), dtype=np.uint8)
        for dy in range(2):
            for dx in range(2):
                l1_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256] = l1_reconstructed[(dx, dy)]

        l0_pred_mosaic = np.array(Image.fromarray(l1_mosaic).resize((1024, 1024), resample=UPSAMPLE))

        l0_psnrs = []
        for (dx, dy), l0_gt in l0_tiles.items():
            pred = l0_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(l0_gt)

            residual = Y_gt - Y_pred
            centered = np.clip(np.round(residual + 128.0), 0, 255).astype(np.uint8)
            decoded, size = encode_decode_gray(centered, quality)
            total_l0_bytes += size

            decoded_residual = decoded.astype(np.float32) - 128.0
            Y_recon = np.clip(Y_pred + decoded_residual, 0, 255)
            rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
            l0_psnrs.append(psnr(l0_gt, rgb_recon))

        total = total_l1_bytes + total_l0_bytes
        l1_avg = np.mean(l1_psnrs)
        l0_avg = np.mean(l0_psnrs)
        min_p = min(min(l1_psnrs), min(l0_psnrs))

        baseline_result = None
        if model_name == "baseline":
            baseline_result = {"l1": l1_avg, "l0": l0_avg, "min": min_p, "bytes": total}

        delta_str = ""
        if model_name != "baseline" and baseline_result is None:
            # Look up baseline
            pass

        print(f"  {model_name:<25s} ({m['bytes']:>5d}B): "
              f"L1={l1_avg:.2f}dB  L0={l0_avg:.2f}dB  min={min_p:.2f}dB  "
              f"size={total/1024:.1f}KB  L1={total_l1_bytes/1024:.1f}KB")

    # Also measure what the corrected residual looks like
    print(f"\n--- Corrected residual statistics ---")
    for model_name in ["baseline", "per-tile quadratic", "DCT top-64", "DCT top-128"]:
        if model_name not in models:
            continue
        corrections = models[model_name]["corrections"]
        print(f"  {model_name}:")
        for (dx, dy) in sorted(l1_residuals.keys()):
            res = l1_residuals[(dx, dy)]
            corr = corrections[(dx, dy)]
            corrected = res - corr
            print(f"    L1({dx},{dy}): orig_std={res.std():.2f} → corrected_std={corrected.std():.2f} "
                  f"(Δ={corrected.std()-res.std():+.2f})")


if __name__ == "__main__":
    main()
