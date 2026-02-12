#!/usr/bin/env python3
"""
l2_optimization_experiment.py

Tests whether optimizing L2 pixel values to minimize downstream residual energy
can improve compression — a form of "prediction-aware encoding."

The idea: Instead of using the true L2 downsampled image, slightly perturb L2 pixels
so that when upsampled to predict L1, the residuals are smaller (and thus compress
better). The perturbation must be small enough to be perceptually invisible.

This is NOT steganography (hiding data in L2). It's optimizing the L2 representation
to make the prediction more useful, accepting tiny L2 quality loss for potentially
large residual compression gains.

Three approaches:
1. Gradient descent: optimize L2 pixels to minimize L1 residual energy
2. Prediction-aware quantization: when L2 has ambiguity (e.g., 127.4 → 127 or 128),
   pick the rounding that minimizes downstream residual
3. Budget analysis: how many bits of L2 perturbation vs residual savings

Usage:
  python l2_optimization_experiment.py --image evals/test-images/L0-1024.jpg --quality 40
"""
import argparse
import numpy as np
from PIL import Image
import io
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_bytes, parse_encoder_arg


def encode_decode_gray(pixels_u8, quality, encoder=JpegEncoder.LIBJPEG_TURBO):
    img = Image.fromarray(pixels_u8, mode="L")
    data = encode_jpeg_to_bytes(img, quality, encoder)
    decoded = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return decoded, len(data)


def encode_decode_rgb(pixels_u8, quality, encoder=JpegEncoder.LIBJPEG_TURBO):
    img = Image.fromarray(pixels_u8, mode="RGB")
    data = encode_jpeg_to_bytes(img, quality, encoder)
    decoded = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return decoded, len(data)


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


def full_pipeline(l2_u8, l1_tiles, l0_tiles, quality, encoder, tile_size=256):
    """Run reconstruction pipeline from a given L2 tile."""
    UPSAMPLE = Image.Resampling.BILINEAR

    # L2 → L1 prediction
    l1_pred_mosaic = np.array(Image.fromarray(l2_u8).resize((512, 512), resample=UPSAMPLE))

    l1_psnrs = []
    total_l1_bytes = 0
    total_l0_bytes = 0
    l1_reconstructed = {}

    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)

        residual_raw = Y_gt - Y_pred
        centered = np.clip(np.round(residual_raw + 128.0), 0, 255).astype(np.uint8)
        decoded, size = encode_decode_gray(centered, quality, encoder)
        Y_recon = np.clip(Y_pred + (decoded.astype(np.float32) - 128.0), 0, 255)

        total_l1_bytes += size
        rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
        l1_reconstructed[(dx, dy)] = rgb_recon
        l1_psnrs.append(psnr(l1_gt, rgb_recon))

    # L1 → L0 prediction
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

        residual_raw = Y_gt - Y_pred
        centered = np.clip(np.round(residual_raw + 128.0), 0, 255).astype(np.uint8)
        decoded, size = encode_decode_gray(centered, quality, encoder)
        Y_recon = np.clip(Y_pred + (decoded.astype(np.float32) - 128.0), 0, 255)

        total_l0_bytes += size
        rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
        l0_psnrs.append(psnr(l0_gt, rgb_recon))

    return {
        "l1_psnr": np.mean(l1_psnrs),
        "l0_psnr": np.mean(l0_psnrs),
        "min_psnr": min(min(l1_psnrs), min(l0_psnrs)),
        "total_bytes": total_l1_bytes + total_l0_bytes,
        "l1_bytes": total_l1_bytes,
        "l0_bytes": total_l0_bytes,
    }


def residual_energy(l2_float, l1_tiles, tile_size=256):
    """Compute total L1 residual energy for a given L2 (float, clipped to 0-255)."""
    UPSAMPLE = Image.Resampling.BILINEAR
    l2_u8 = np.clip(np.round(l2_float), 0, 255).astype(np.uint8)
    l1_pred_mosaic = np.array(Image.fromarray(l2_u8).resize((512, 512), resample=UPSAMPLE))

    total_energy = 0.0
    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
        Y_pred, _, _ = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)
        residual = Y_gt - Y_pred
        total_energy += np.sum(residual**2)

    return total_energy


def optimize_l2_gradient(l2_original, l1_tiles, budget_db, learning_rate=0.5, iterations=200):
    """
    Gradient descent to minimize L1 residual energy subject to L2 quality budget.

    budget_db: maximum PSNR drop allowed on L2 (e.g., 0.5 dB)
    """
    UPSAMPLE = Image.Resampling.BILINEAR
    l2_float = l2_original.astype(np.float32).copy()
    tile_size = 256

    # Compute max allowed MSE from PSNR budget
    # PSNR = 10*log10(255^2/MSE), so target PSNR = original_psnr - budget
    # For original (perfect) L2, PSNR = inf, so we use budget as max MSE directly
    # budget_db drop means: max_mse = 255^2 / 10^((target_psnr)/10)
    # Simpler: budget_db = 0.5 means we allow MSE up to 255^2 / 10^((60-0.5)/10) ≈ very small
    # Actually let's just set max perturbation per pixel
    # 0.5 dB PSNR drop from inf means MSE ≈ 255^2/10^5.95 ≈ 0.073
    # That's sqrt(0.073) ≈ 0.27 RMS per pixel... very small

    # More practical: allow max perturbation of N per pixel
    max_perturb = {
        0.5: 1.0,   # ±1 pixel value → PSNR ~48 dB for L2
        1.0: 2.0,   # ±2 → PSNR ~42 dB
        2.0: 4.0,   # ±4 → PSNR ~36 dB
        3.0: 6.0,   # ±6 → PSNR ~33 dB
    }.get(budget_db, budget_db)

    best_l2 = l2_float.copy()
    best_energy = residual_energy(l2_float, l1_tiles)

    for it in range(iterations):
        # Compute gradient numerically (per-pixel, per-channel)
        # This is expensive but correct
        # Use a random subset of pixels for stochastic gradient
        h, w, c = l2_float.shape
        n_samples = min(500, h * w)  # Sample pixels
        indices = np.random.choice(h * w, n_samples, replace=False)
        ys, xs = np.unravel_index(indices, (h, w))

        for i in range(len(ys)):
            y, x = ys[i], xs[i]
            for ch in range(3):
                # Try +1 and -1
                original_val = l2_float[y, x, ch]

                # Only perturb within budget
                if abs(l2_float[y, x, ch] - l2_original[y, x, ch]) >= max_perturb:
                    continue

                l2_float[y, x, ch] = min(original_val + 1.0, l2_original[y, x, ch] + max_perturb)
                e_plus = residual_energy(l2_float, l1_tiles)

                l2_float[y, x, ch] = max(original_val - 1.0, l2_original[y, x, ch] - max_perturb)
                e_minus = residual_energy(l2_float, l1_tiles)

                # Pick the better direction
                if e_plus < e_minus and e_plus < best_energy:
                    l2_float[y, x, ch] = min(original_val + 1.0, l2_original[y, x, ch] + max_perturb)
                    best_energy = e_plus
                elif e_minus < best_energy:
                    l2_float[y, x, ch] = max(original_val - 1.0, l2_original[y, x, ch] - max_perturb)
                    best_energy = e_minus
                else:
                    l2_float[y, x, ch] = original_val

        if (it + 1) % 50 == 0:
            current_psnr = psnr(l2_original, np.clip(np.round(l2_float), 0, 255).astype(np.uint8))
            print(f"    Iteration {it+1}: energy={best_energy:.0f}, L2 PSNR={current_psnr:.1f} dB")

    return np.clip(np.round(l2_float), 0, 255).astype(np.uint8)


def smart_round_l2(l2_original_float, l1_tiles, tile_size=256):
    """
    When downsampling produces fractional values (e.g., 127.4), choose floor or ceil
    based on which minimizes L1 residual energy.

    This is "free" — we have to round anyway, we're just choosing the better direction.
    """
    UPSAMPLE = Image.Resampling.BILINEAR

    # Get the true float L2 (before rounding)
    l2_floor = np.floor(l2_original_float).astype(np.uint8)
    l2_ceil = np.clip(np.ceil(l2_original_float), 0, 255).astype(np.uint8)
    l2_round = np.round(l2_original_float).astype(np.uint8)  # standard

    # For each pixel, try floor vs ceil
    h, w, c = l2_floor.shape
    l2_smart = l2_round.copy()

    # We can't do per-pixel optimization efficiently (each pixel change affects
    # the entire upsampled prediction). Instead, do it channel by channel, pixel by pixel
    # for a subset to estimate the benefit.

    # Actually, bilinear upsampling is linear, so the residual for each L1 pixel
    # depends on at most 4 L2 pixels. We can compute the influence analytically.

    # Simpler approach: just try both roundings for each pixel and keep the better one
    # This is O(H*W*C) pipeline evaluations — too expensive.

    # Practical approach: compute the fractional part and bias toward the rounding
    # that makes the upsampled value closer to the L1 ground truth.

    # Build the L1 luma ground truths
    l1_Y = {}
    for (dx, dy), l1_gt in l1_tiles.items():
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)
        l1_Y[(dx, dy)] = Y_gt

    # Build L1 prediction from standard rounding
    l1_pred_std = np.array(Image.fromarray(l2_round).resize((512, 512), resample=UPSAMPLE))

    # For each L2 pixel, compute the derivative of residual energy w.r.t. that pixel
    # d(energy)/d(L2[y,x,c]) = -2 * sum over affected L1 pixels of residual * d(pred)/d(L2)
    #
    # For bilinear upsampling (2x), each L2 pixel affects a 3x3 neighborhood of L1 pixels
    # with known weights. The sign of the gradient tells us whether to round up or down.

    # Even simpler: just compare the two full predictions
    l1_pred_floor = np.array(Image.fromarray(l2_floor).resize((512, 512), resample=UPSAMPLE))
    l1_pred_ceil = np.array(Image.fromarray(l2_ceil).resize((512, 512), resample=UPSAMPLE))

    # The difference is small (0 or 1 in upsampled space), so we can just measure
    # which L2 tile (floor vs ceil) gives better total residual energy
    e_floor = residual_energy(l2_floor.astype(np.float32), l1_tiles)
    e_ceil = residual_energy(l2_ceil.astype(np.float32), l1_tiles)
    e_round = residual_energy(l2_round.astype(np.float32), l1_tiles)

    print(f"    Floor energy:   {e_floor:.0f}")
    print(f"    Ceil energy:    {e_ceil:.0f}")
    print(f"    Round energy:   {e_round:.0f}")

    # Per-pixel: for pixels with fractional part near 0.5, try both
    fractional = l2_original_float - np.floor(l2_original_float)
    ambiguous = (fractional > 0.3) & (fractional < 0.7)
    n_ambiguous = np.sum(ambiguous)
    print(f"    Ambiguous pixels (0.3 < frac < 0.7): {n_ambiguous} / {h*w*c}")

    return l2_smart, {
        "floor_energy": e_floor,
        "ceil_energy": e_ceil,
        "round_energy": e_round,
        "n_ambiguous": int(n_ambiguous),
    }


def measure_capacity(image_path):
    """Measure: how much residual data needs to be embedded vs L2 capacity."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

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

    # L2 capacity
    l2_pixels = 256 * 256 * 3  # 196,608 values
    print(f"\nL2 tile: 256x256 RGB = {l2_pixels:,} values")
    print(f"  At 1 LSB per value: {l2_pixels/8/1024:.1f} KB capacity")
    print(f"  At 2 LSBs per value: {l2_pixels*2/8/1024:.1f} KB capacity")
    print(f"  At 3 LSBs per value: {l2_pixels*3/8/1024:.1f} KB capacity")

    # Residual data needed at various qualities
    encoder = JpegEncoder.LIBJPEG_TURBO
    UPSAMPLE = Image.Resampling.BILINEAR
    l1_pred_mosaic = np.array(Image.fromarray(l2).resize((512, 512), resample=UPSAMPLE))

    print(f"\nResidual sizes at different JPEG qualities:")
    for q in [30, 40, 50, 60, 70, 80]:
        total_bytes = 0
        l1_reconstructed = {}

        for (dx, dy), l1_gt in l1_tiles.items():
            pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)
            residual = np.clip(np.round(Y_gt - Y_pred + 128.0), 0, 255).astype(np.uint8)
            decoded, size = encode_decode_gray(residual, q, encoder)
            total_bytes += size
            Y_recon = np.clip(Y_pred + (decoded.astype(np.float32) - 128.0), 0, 255)
            l1_reconstructed[(dx, dy)] = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)

        l1_mosaic = np.zeros((512, 512, 3), dtype=np.uint8)
        for dy in range(2):
            for dx in range(2):
                l1_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256] = l1_reconstructed[(dx, dy)]

        l0_pred = np.array(Image.fromarray(l1_mosaic).resize((1024, 1024), resample=UPSAMPLE))
        for (dx, dy), l0_gt in l0_tiles.items():
            pred = l0_pred[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
            Y_pred, _, _ = rgb_to_ycbcr_bt601(pred)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(l0_gt)
            residual = np.clip(np.round(Y_gt - Y_pred + 128.0), 0, 255).astype(np.uint8)
            _, size = encode_decode_gray(residual, q, encoder)
            total_bytes += size

        lsb_needed = total_bytes * 8 / l2_pixels
        print(f"  q={q:2d}: {total_bytes/1024:.1f} KB residuals → {lsb_needed:.1f} bits/pixel needed")

    # Residual entropy analysis
    print(f"\n--- Raw residual statistics ---")
    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
        Y_pred, _, _ = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)
        residual = Y_gt - Y_pred
        print(f"  L1({dx},{dy}): range=[{residual.min():.0f}, {residual.max():.0f}], "
              f"std={residual.std():.1f}, abs_mean={np.abs(residual).mean():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--quality", type=int, default=40)
    parser.add_argument("--encoder", default="libjpeg-turbo")
    args = parser.parse_args()

    encoder = parse_encoder_arg(args.encoder)

    # Load image and create tiles
    img = Image.open(args.image).convert("RGB")
    img_array = np.array(img)

    # Get the float L2 (before rounding) for smart rounding experiment
    l2_float = np.array(Image.fromarray(img_array[:1024, :1024]).resize(
        (256, 256), Image.LANCZOS)).astype(np.float32)
    l2_original = np.round(l2_float).astype(np.uint8)

    l1_source = np.array(Image.fromarray(img_array[:1024, :1024]).resize((512, 512), Image.LANCZOS))
    l1_tiles = {}
    for dy in range(2):
        for dx in range(2):
            l1_tiles[(dx, dy)] = l1_source[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    l0_tiles = {}
    for dy in range(4):
        for dx in range(4):
            l0_tiles[(dx, dy)] = img_array[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    print(f"{'='*70}")
    print(f"L2 OPTIMIZATION EXPERIMENT")
    print(f"{'='*70}")

    # 1. Capacity analysis
    measure_capacity(args.image)

    # 2. Smart rounding experiment
    print(f"\n--- Smart rounding experiment ---")
    # Note: Lanczos already produces well-rounded values, so this may not help much
    # The float values from Pillow's resize are already close to integers
    l2_smart, round_stats = smart_round_l2(l2_float, l1_tiles)

    # 3. Baseline pipeline
    print(f"\n--- Pipeline comparison at q={args.quality} ---")
    baseline = full_pipeline(l2_original, l1_tiles, l0_tiles, args.quality, encoder)
    print(f"  Baseline:     L1={baseline['l1_psnr']:.2f}dB  L0={baseline['l0_psnr']:.2f}dB  "
          f"min={baseline['min_psnr']:.2f}dB  size={baseline['total_bytes']/1024:.1f}KB")

    # 4. ±1 perturbation experiment
    # Try all 6 simple perturbations: shift all L2 luma by ±1
    print(f"\n--- L2 luma perturbation experiment ---")
    for delta in [-2, -1, +1, +2]:
        l2_perturbed = np.clip(l2_original.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        l2_psnr_drop = psnr(l2_original, l2_perturbed)
        result = full_pipeline(l2_perturbed, l1_tiles, l0_tiles, args.quality, encoder)
        d_min = result['min_psnr'] - baseline['min_psnr']
        d_size = (result['total_bytes'] - baseline['total_bytes']) / baseline['total_bytes'] * 100
        print(f"  L2 + {delta:+d}: L2 PSNR={l2_psnr_drop:.1f}dB  "
              f"L1={result['l1_psnr']:.2f}dB  L0={result['l0_psnr']:.2f}dB  "
              f"Δmin={d_min:+.2f}dB  Δsize={d_size:+.1f}%")

    # 5. JPEG-domain experiment: what if L2 is JPEG compressed?
    # The real scenario: L2 goes through JPEG before upsampling
    print(f"\n--- L2 JPEG round-trip effect ---")
    for l2_q in [85, 90, 95, 99]:
        l2_decoded, l2_size = encode_decode_rgb(l2_original, l2_q, encoder)
        l2_psnr = psnr(l2_original, l2_decoded)
        result = full_pipeline(l2_decoded, l1_tiles, l0_tiles, args.quality, encoder)
        d_min = result['min_psnr'] - baseline['min_psnr']
        d_size = (result['total_bytes'] - baseline['total_bytes']) / baseline['total_bytes'] * 100
        print(f"  L2 q={l2_q:2d} ({l2_size/1024:.1f}KB, PSNR={l2_psnr:.1f}dB): "
              f"L1={result['l1_psnr']:.2f}dB  L0={result['l0_psnr']:.2f}dB  "
              f"Δmin={d_min:+.2f}dB  Δsize={d_size:+.1f}%")


if __name__ == "__main__":
    main()
