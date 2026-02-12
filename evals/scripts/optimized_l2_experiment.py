#!/usr/bin/env python3
"""
optimized_l2_experiment.py

THE KEY INSIGHT: L2 doesn't have to be the "true" downsampled image.
L2 should be the image that, when bilinearly upsampled, best predicts L1.

At encode time (unlimited budget): optimize L2 pixels via gradient descent.
At decode time (sub-ms budget): unchanged — just bilinear upsample + add residual.

The decoder doesn't know or care that L2 was optimized. Zero added decode complexity.
"""
import argparse
import numpy as np
from PIL import Image
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


def bilinear_upsample_2x(img_u8):
    h, w = img_u8.shape[:2]
    return np.array(Image.fromarray(img_u8).resize((w*2, h*2), Image.Resampling.BILINEAR))


def optimize_l2_luma(l2_rgb, l1_tiles, max_delta=10, n_iterations=500, lr=0.3):
    """
    Gradient descent on L2 luma to minimize L1 prediction residual energy.

    The gradient is computed by:
    1. Upsample current L2 → L1 prediction
    2. Compute residual = L1_gt - L1_pred
    3. Downsample residual back to L2 resolution (transpose of upsample)
    4. Update L2 luma in the direction of the gradient

    This is fast because it only uses upsample/downsample operations.
    """
    # Build L1 target luma
    tile_size = 256
    l1_mosaic_Y = np.zeros((512, 512), dtype=np.float64)
    for (dx, dy), l1_gt in l1_tiles.items():
        Y, _, _ = rgb_to_ycbcr_bt601(l1_gt)
        l1_mosaic_Y[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size] = Y

    l2_Y_orig, l2_Cb, l2_Cr = rgb_to_ycbcr_bt601(l2_rgb)
    l2_Y = l2_Y_orig.astype(np.float64).copy()
    l2_Y_start = l2_Y.copy()

    best_energy = float('inf')
    best_l2_Y = l2_Y.copy()

    for it in range(n_iterations):
        # Forward: upsample
        l2_rgb_cur = ycbcr_to_rgb_bt601(l2_Y.astype(np.float32), l2_Cb, l2_Cr)
        pred = bilinear_upsample_2x(l2_rgb_cur)
        pred_Y, _, _ = rgb_to_ycbcr_bt601(pred)

        # Residual
        residual = l1_mosaic_Y - pred_Y
        energy = np.sum(residual**2)

        if energy < best_energy:
            best_energy = energy
            best_l2_Y = l2_Y.copy()

        # Backward: downsample residual (approximate gradient)
        grad = np.array(Image.fromarray(residual.astype(np.float32)).resize(
            (256, 256), Image.Resampling.BILINEAR))

        # Step
        l2_Y += lr * grad

        # Constrain: stay within max_delta of original and within [0, 255]
        l2_Y = np.clip(l2_Y, l2_Y_start - max_delta, l2_Y_start + max_delta)
        l2_Y = np.clip(l2_Y, 0, 255)

    # Return optimized L2
    l2_opt = ycbcr_to_rgb_bt601(best_l2_Y.astype(np.float32), l2_Cb, l2_Cr)
    return l2_opt, best_l2_Y, best_energy


def full_pipeline(l2_rgb, l1_tiles, l0_tiles, quality, tile_size=256):
    """Full L2→L1→L0 reconstruction pipeline."""
    UPSAMPLE = Image.Resampling.BILINEAR
    l1_pred_mosaic = np.array(Image.fromarray(l2_rgb).resize(
        (tile_size*2, tile_size*2), resample=UPSAMPLE))

    l1_psnrs = []
    l1_pred_psnrs = []
    total_l1_bytes = 0
    total_l0_bytes = 0
    l1_reconstructed = {}
    l1_residual_stds = []

    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)

        # Prediction quality
        l1_pred_psnrs.append(psnr(Y_gt, Y_pred, data_range=255))

        residual = Y_gt - Y_pred
        l1_residual_stds.append(np.std(residual))

        centered = np.clip(np.round(residual + 128.0), 0, 255).astype(np.uint8)
        decoded, size = encode_decode_gray(centered, quality)
        total_l1_bytes += size

        Y_recon = np.clip(Y_pred + (decoded.astype(np.float32) - 128.0), 0, 255)
        rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
        l1_reconstructed[(dx, dy)] = rgb_recon
        l1_psnrs.append(psnr(l1_gt, rgb_recon))

    l1_mosaic = np.zeros((tile_size*2, tile_size*2, 3), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            l1_mosaic[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size] = l1_reconstructed[(dx, dy)]

    l0_pred_mosaic = np.array(Image.fromarray(l1_mosaic).resize(
        (tile_size*4, tile_size*4), resample=UPSAMPLE))

    l0_psnrs = []
    for (dx, dy), l0_gt in l0_tiles.items():
        pred = l0_pred_mosaic[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l0_gt)

        residual = Y_gt - Y_pred
        centered = np.clip(np.round(residual + 128.0), 0, 255).astype(np.uint8)
        decoded, size = encode_decode_gray(centered, quality)
        total_l0_bytes += size

        Y_recon = np.clip(Y_pred + (decoded.astype(np.float32) - 128.0), 0, 255)
        rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
        l0_psnrs.append(psnr(l0_gt, rgb_recon))

    return {
        "l1_pred_psnr": np.mean(l1_pred_psnrs),
        "l1_psnr": np.mean(l1_psnrs),
        "l0_psnr": np.mean(l0_psnrs),
        "min_psnr": min(min(l1_psnrs), min(l0_psnrs)),
        "total_bytes": total_l1_bytes + total_l0_bytes,
        "l1_bytes": total_l1_bytes,
        "l0_bytes": total_l0_bytes,
        "l1_residual_std": np.mean(l1_residual_stds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--quality", type=int, default=40)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    img_array = np.array(img)

    l2_natural = np.array(Image.fromarray(img_array[:1024, :1024]).resize((256, 256), Image.LANCZOS))
    l1_source = np.array(Image.fromarray(img_array[:1024, :1024]).resize((512, 512), Image.LANCZOS))

    l1_tiles = {}
    for dy in range(2):
        for dx in range(2):
            l1_tiles[(dx, dy)] = l1_source[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    l0_tiles = {}
    for dy in range(4):
        for dx in range(4):
            l0_tiles[(dx, dy)] = img_array[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    print(f"{'='*80}")
    print(f"OPTIMIZED L2 FOR PREDICTION — Encode-Heavy, Decode-Light")
    print(f"{'='*80}")

    # Baseline
    print(f"\n--- Baseline (natural Lanczos L2) ---")
    baseline = full_pipeline(l2_natural, l1_tiles, l0_tiles, args.quality)
    print(f"  L1 prediction PSNR:   {baseline['l1_pred_psnr']:.2f} dB")
    print(f"  L1 residual std:      {baseline['l1_residual_std']:.2f}")
    print(f"  L1 final:             {baseline['l1_psnr']:.2f} dB")
    print(f"  L0 final:             {baseline['l0_psnr']:.2f} dB")
    print(f"  Min PSNR:             {baseline['min_psnr']:.2f} dB")
    print(f"  Total size:           {baseline['total_bytes']/1024:.1f} KB")

    # Test different max_delta values
    print(f"\n--- Optimization with different L2 perturbation budgets ---")
    for max_delta in [3, 5, 10, 15, 20, 30, 50]:
        print(f"\n  max_delta = ±{max_delta}:")
        l2_opt, l2_Y_opt, energy = optimize_l2_luma(l2_natural, l1_tiles, max_delta=max_delta, n_iterations=500, lr=0.3)

        # How much did L2 change?
        l2_psnr = psnr(l2_natural, l2_opt)
        l2_Y_natural, _, _ = rgb_to_ycbcr_bt601(l2_natural)
        l2_Y_opt_f = l2_Y_opt.astype(np.float64)
        l2_Y_nat_f = l2_Y_natural.astype(np.float64)
        actual_max = np.max(np.abs(l2_Y_opt_f - l2_Y_nat_f))
        actual_mean = np.mean(np.abs(l2_Y_opt_f - l2_Y_nat_f))

        result = full_pipeline(l2_opt, l1_tiles, l0_tiles, args.quality)

        d_pred = result['l1_pred_psnr'] - baseline['l1_pred_psnr']
        d_l1 = result['l1_psnr'] - baseline['l1_psnr']
        d_l0 = result['l0_psnr'] - baseline['l0_psnr']
        d_min = result['min_psnr'] - baseline['min_psnr']
        d_size = (result['total_bytes'] - baseline['total_bytes']) / baseline['total_bytes'] * 100

        print(f"    L2 change: PSNR={l2_psnr:.1f}dB, actual_max={actual_max:.1f}, mean={actual_mean:.1f}")
        print(f"    L1 pred:   {result['l1_pred_psnr']:.2f}dB ({d_pred:+.2f})  residual_std={result['l1_residual_std']:.2f}")
        print(f"    L1 final:  {result['l1_psnr']:.2f}dB ({d_l1:+.2f})")
        print(f"    L0 final:  {result['l0_psnr']:.2f}dB ({d_l0:+.2f})")
        print(f"    Min PSNR:  {result['min_psnr']:.2f}dB ({d_min:+.2f})")
        print(f"    Size:      {result['total_bytes']/1024:.1f}KB ({d_size:+.1f}%)")

    # Cross-quality consistency
    print(f"\n{'='*80}")
    print(f"CROSS-QUALITY CHECK: Does optimized L2 help at all JPEG quality levels?")
    print(f"Using max_delta=15")
    print(f"{'='*80}")

    l2_opt_15, _, _ = optimize_l2_luma(l2_natural, l1_tiles, max_delta=15, n_iterations=500, lr=0.3)

    print(f"\n{'Q':>4s}  {'Base min':>10s}  {'Opt min':>10s}  {'Δmin':>8s}  {'Base L1pred':>11s}  {'Opt L1pred':>11s}  {'Δpred':>8s}  {'Base KB':>8s}  {'Opt KB':>8s}  {'ΔKB':>8s}")
    print("-" * 110)

    for q in [20, 30, 40, 50, 60, 70, 80, 90]:
        b = full_pipeline(l2_natural, l1_tiles, l0_tiles, q)
        o = full_pipeline(l2_opt_15, l1_tiles, l0_tiles, q)
        print(f"  {q:2d}   {b['min_psnr']:8.2f}dB  {o['min_psnr']:8.2f}dB  {o['min_psnr']-b['min_psnr']:+6.2f}dB  "
              f"{b['l1_pred_psnr']:9.2f}dB  {o['l1_pred_psnr']:9.2f}dB  {o['l1_pred_psnr']-b['l1_pred_psnr']:+6.2f}dB  "
              f"{b['total_bytes']/1024:6.1f}KB  {o['total_bytes']/1024:6.1f}KB  "
              f"{(o['total_bytes']-b['total_bytes'])/1024:+5.1f}KB")

    # Combined: optimized L2 + split quality
    print(f"\n{'='*80}")
    print(f"COMBINED: Optimized L2 + Split Quality (+20)")
    print(f"{'='*80}")

    print(f"\n{'Config':<35s}  {'L1 PSNR':>8s}  {'L0 PSNR':>8s}  {'Min':>8s}  {'Size':>8s}")
    print("-" * 75)

    for q in [30, 40, 50, 60]:
        # Baseline flat
        b = full_pipeline(l2_natural, l1_tiles, l0_tiles, q)
        print(f"  Natural L2, flat q={q:<14d}  {b['l1_psnr']:6.2f}dB  {b['l0_psnr']:6.2f}dB  {b['min_psnr']:6.2f}dB  {b['total_bytes']/1024:6.1f}KB")

        # Baseline +20 split — need to modify pipeline for split quality
        # Just report the numbers from our earlier runs
        # For now, test optimized L2 flat
        o = full_pipeline(l2_opt_15, l1_tiles, l0_tiles, q)
        print(f"  Opt L2(±15), flat q={q:<11d}  {o['l1_psnr']:6.2f}dB  {o['l0_psnr']:6.2f}dB  {o['min_psnr']:6.2f}dB  {o['total_bytes']/1024:6.1f}KB  ({o['min_psnr']-b['min_psnr']:+.2f}dB)")


if __name__ == "__main__":
    main()
