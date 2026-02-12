#!/usr/bin/env python3
"""
encode_heavy_decode_light.py

Asymmetric codec: unlimited encode budget, sub-ms decode on CPU with SIMD.

Core idea: at encode time, we find the OPTIMAL pixel values for L2 such that
bilinear upsampling produces the best possible L1 prediction. This is NOT the
true downsampled L2 — it's an L2 that's been optimized to minimize prediction
error when upsampled.

Think of it as: instead of L2 representing "what the image looks like at low res,"
L2 represents "the values that, when bilinearly upsampled, best predict L1."

Decode is unchanged: bilinear upsample + add residual. Zero added complexity.

Additional strategies:
1. OPTIMIZED L2: Find L2 pixels that minimize L1 residual energy
2. CUSTOM UPSAMPLING KERNELS: Instead of bilinear (fixed weights), precompute
   per-pixel optimal weights from a small kernel set. Store kernel indices as
   side data (tiny). Decode: table lookup + multiply-add (SIMD-native).
3. PREDICTION REFINEMENT MAP: A low-res correction field (e.g., 32x32) that
   the decoder adds to the prediction before applying residuals. Very cheap
   to upsample and apply.

Usage:
  python encode_heavy_decode_light.py --image evals/test-images/L0-1024.jpg --quality 40
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
    """Bilinear 2x upsample, matching the pipeline."""
    h, w = img_u8.shape[:2]
    return np.array(Image.fromarray(img_u8).resize(
        (w*2, h*2), Image.Resampling.BILINEAR))


def build_bilinear_matrix(src_h, src_w, dst_h, dst_w):
    """
    Build the explicit linear mapping matrix A such that:
        dst_pixels = A @ src_pixels
    for bilinear upsampling. This lets us solve for optimal src_pixels
    given desired dst_pixels using least squares.

    Only for single channel (luma).
    """
    n_dst = dst_h * dst_w
    n_src = src_h * src_w
    A = np.zeros((n_dst, n_src), dtype=np.float64)

    for dy in range(dst_h):
        for dx in range(dst_w):
            # Map destination pixel to source coordinates
            # PIL bilinear maps dst center to src center
            sx = (dx + 0.5) * src_w / dst_w - 0.5
            sy = (dy + 0.5) * src_h / dst_h - 0.5

            x0 = int(np.floor(sx))
            y0 = int(np.floor(sy))
            x1 = min(x0 + 1, src_w - 1)
            y1 = min(y0 + 1, src_h - 1)
            x0 = max(x0, 0)
            y0 = max(y0, 0)

            fx = sx - np.floor(sx)
            fy = sy - np.floor(sy)
            if sx < 0: fx = 0
            if sy < 0: fy = 0

            dst_idx = dy * dst_w + dx
            A[dst_idx, y0 * src_w + x0] += (1 - fx) * (1 - fy)
            A[dst_idx, y0 * src_w + x1] += fx * (1 - fy)
            A[dst_idx, y1 * src_w + x0] += (1 - fx) * fy
            A[dst_idx, y1 * src_w + x1] += fx * fy

    return A


def optimize_l2_for_prediction(l2_rgb, l1_tiles, tile_size=256, n_iterations=50):
    """
    Optimize L2 pixel values to minimize L1 residual energy.

    For luma: solve the least-squares problem
        min ||A @ l2_Y - l1_Y_target||^2
    where A is the bilinear upsampling matrix.

    This has a closed-form solution: l2_Y_opt = (A^T A)^{-1} A^T l1_Y_target

    The catch: L2 must be uint8 (0-255) for JPEG storage.
    """
    # Build L1 target luma mosaic
    l1_mosaic_Y = np.zeros((tile_size*2, tile_size*2), dtype=np.float64)
    l1_mosaic_Cb = np.zeros_like(l1_mosaic_Y)
    l1_mosaic_Cr = np.zeros_like(l1_mosaic_Y)

    for (dx, dy), l1_gt in l1_tiles.items():
        Y, Cb, Cr = rgb_to_ycbcr_bt601(l1_gt)
        l1_mosaic_Y[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size] = Y
        l1_mosaic_Cb[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size] = Cb
        l1_mosaic_Cr[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size] = Cr

    # Build bilinear matrix (256x256 → 512x512)
    print("    Building bilinear matrix...")
    A = build_bilinear_matrix(tile_size, tile_size, tile_size*2, tile_size*2)

    # Solve for optimal L2 luma
    print("    Solving least squares (this is the heavy encode step)...")
    target = l1_mosaic_Y.ravel()

    # Normal equations: (A^T A) x = A^T b
    ATA = A.T @ A
    ATb = A.T @ target

    # Add small regularization to keep L2 close to natural appearance
    # (prevents extreme values that would look bad at L2 zoom level)
    l2_Y_natural, _, _ = rgb_to_ycbcr_bt601(l2_rgb)
    lambda_reg = 0.01  # regularization strength
    l2_Y_opt = np.linalg.solve(
        ATA + lambda_reg * np.eye(ATA.shape[0]),
        ATb + lambda_reg * l2_Y_natural.ravel()
    )

    l2_Y_opt = l2_Y_opt.reshape(tile_size, tile_size)

    # Clamp to valid uint8 range
    l2_Y_opt = np.clip(l2_Y_opt, 0, 255)

    # Reconstruct RGB from optimized luma + original chroma
    l2_Cb = rgb_to_ycbcr_bt601(l2_rgb)[1]
    l2_Cr = rgb_to_ycbcr_bt601(l2_rgb)[2]
    l2_opt_rgb = ycbcr_to_rgb_bt601(l2_Y_opt, l2_Cb, l2_Cr)

    return l2_opt_rgb, l2_Y_opt


def optimize_l2_iterative(l2_rgb, l1_tiles, tile_size=256, n_iterations=200, lr=0.5):
    """
    Iterative pixel-wise optimization of L2 luma values.
    Faster than full matrix solve for large images.
    Uses gradient descent on the residual energy.
    """
    l1_mosaic_Y = np.zeros((tile_size*2, tile_size*2), dtype=np.float64)
    for (dx, dy), l1_gt in l1_tiles.items():
        Y, _, _ = rgb_to_ycbcr_bt601(l1_gt)
        l1_mosaic_Y[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size] = Y

    l2_Y_current, l2_Cb, l2_Cr = rgb_to_ycbcr_bt601(l2_rgb)
    l2_Y = l2_Y_current.astype(np.float64).copy()
    l2_Y_original = l2_Y.copy()

    best_energy = float('inf')
    best_l2_Y = l2_Y.copy()

    for it in range(n_iterations):
        # Upsample current L2
        l2_u8 = np.clip(np.round(l2_Y), 0, 255).astype(np.uint8)
        l2_rgb_current = ycbcr_to_rgb_bt601(l2_Y.astype(np.float32), l2_Cb, l2_Cr)
        pred = bilinear_upsample_2x(l2_rgb_current)
        pred_Y, _, _ = rgb_to_ycbcr_bt601(pred)

        # Residual
        residual = l1_mosaic_Y - pred_Y
        energy = np.sum(residual**2)

        if energy < best_energy:
            best_energy = energy
            best_l2_Y = l2_Y.copy()

        if (it + 1) % 50 == 0:
            print(f"      Iteration {it+1}: energy={energy:.0f}")

        # Approximate gradient: for bilinear 2x, each L2 pixel affects a ~3x3
        # neighborhood in L1 space. The gradient is the sum of residuals in that area
        # weighted by the bilinear kernel.
        #
        # Simplified: downsample the residual back to L2 resolution
        residual_down = np.array(Image.fromarray(residual.astype(np.float32)).resize(
            (tile_size, tile_size), Image.Resampling.BILINEAR))

        # Gradient step
        l2_Y += lr * residual_down

        # Regularize: don't stray too far from original
        max_delta = 10.0  # max ±10 from original
        l2_Y = np.clip(l2_Y, l2_Y_original - max_delta, l2_Y_original + max_delta)
        l2_Y = np.clip(l2_Y, 0, 255)

    # Use best
    l2_opt_rgb = ycbcr_to_rgb_bt601(
        best_l2_Y.astype(np.float32), l2_Cb, l2_Cr)

    return l2_opt_rgb, best_l2_Y


def full_pipeline(l2_rgb, l1_tiles, l0_tiles, quality, tile_size=256):
    """Standard pipeline: upsample L2 → predict L1 → encode residuals → reconstruct."""
    UPSAMPLE = Image.Resampling.BILINEAR
    l1_pred_mosaic = np.array(Image.fromarray(l2_rgb).resize(
        (tile_size*2, tile_size*2), resample=UPSAMPLE))

    l1_psnrs = []
    l1_pred_psnrs = []
    total_l1_bytes = 0
    total_l0_bytes = 0
    l1_reconstructed = {}

    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)

        # Prediction quality (before residuals)
        pred_rgb = ycbcr_to_rgb_bt601(Y_pred, Cb_pred, Cr_pred)
        l1_pred_psnrs.append(psnr(l1_gt, pred_rgb))

        # Encode residual
        residual = Y_gt - Y_pred
        centered = np.clip(np.round(residual + 128.0), 0, 255).astype(np.uint8)
        decoded, size = encode_decode_gray(centered, quality)
        total_l1_bytes += size

        Y_recon = np.clip(Y_pred + (decoded.astype(np.float32) - 128.0), 0, 255)
        rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
        l1_reconstructed[(dx, dy)] = rgb_recon
        l1_psnrs.append(psnr(l1_gt, rgb_recon))

    # L0
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
        "residual_std_l1": np.mean([np.std(rgb_to_ycbcr_bt601(l1_tiles[k])[0] -
                                           rgb_to_ycbcr_bt601(l1_pred_mosaic[k[1]*256:(k[1]+1)*256, k[0]*256:(k[0]+1)*256])[0])
                                    for k in l1_tiles]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--quality", type=int, default=40)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    img_array = np.array(img)
    tile_size = 256

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
    print(f"ENCODE-HEAVY / DECODE-LIGHT OPTIMIZATION")
    print(f"Residual quality: q={args.quality}")
    print(f"{'='*80}")

    # Baseline
    print(f"\n--- Baseline (natural L2 + bilinear upsample) ---")
    baseline = full_pipeline(l2_natural, l1_tiles, l0_tiles, args.quality)
    print(f"  L1 prediction PSNR: {baseline['l1_pred_psnr']:.2f} dB (before residuals)")
    print(f"  L1 final PSNR:      {baseline['l1_psnr']:.2f} dB")
    print(f"  L0 final PSNR:      {baseline['l0_psnr']:.2f} dB")
    print(f"  Min PSNR:           {baseline['min_psnr']:.2f} dB")
    print(f"  Total size:         {baseline['total_bytes']/1024:.1f} KB")
    print(f"  L2 vs original:     {psnr(img_array[:1024,:1024], bilinear_upsample_2x(bilinear_upsample_2x(l2_natural))):.2f} dB")

    # Strategy 1: Optimized L2 (iterative)
    print(f"\n--- Strategy 1: Optimized L2 (iterative, max_delta=10) ---")
    l2_opt_10, l2_Y_opt_10 = optimize_l2_iterative(l2_natural, l1_tiles, n_iterations=200, lr=0.5)
    l2_psnr_10 = psnr(l2_natural, l2_opt_10)
    result_10 = full_pipeline(l2_opt_10, l1_tiles, l0_tiles, args.quality)
    print(f"  L2 modification:    PSNR vs natural L2 = {l2_psnr_10:.1f} dB")
    print(f"  L1 prediction PSNR: {result_10['l1_pred_psnr']:.2f} dB ({result_10['l1_pred_psnr']-baseline['l1_pred_psnr']:+.2f})")
    print(f"  L1 final PSNR:      {result_10['l1_psnr']:.2f} dB ({result_10['l1_psnr']-baseline['l1_psnr']:+.2f})")
    print(f"  L0 final PSNR:      {result_10['l0_psnr']:.2f} dB ({result_10['l0_psnr']-baseline['l0_psnr']:+.2f})")
    print(f"  Min PSNR:           {result_10['min_psnr']:.2f} dB ({result_10['min_psnr']-baseline['min_psnr']:+.2f})")
    print(f"  Total size:         {result_10['total_bytes']/1024:.1f} KB ({(result_10['total_bytes']-baseline['total_bytes'])/baseline['total_bytes']*100:+.1f}%)")

    # Strategy 1b: Larger delta
    print(f"\n--- Strategy 1b: Optimized L2 (iterative, max_delta=25) ---")
    l2_opt_25, l2_Y_opt_25 = optimize_l2_iterative(l2_natural, l1_tiles, n_iterations=300, lr=0.8)
    # Override the max_delta in the function — let me just do it inline
    l1_mosaic_Y = np.zeros((512, 512), dtype=np.float64)
    for (dx, dy), l1_gt in l1_tiles.items():
        Y, _, _ = rgb_to_ycbcr_bt601(l1_gt)
        l1_mosaic_Y[dy*256:(dy+1)*256, dx*256:(dx+1)*256] = Y

    l2_Y, l2_Cb, l2_Cr = rgb_to_ycbcr_bt601(l2_natural)
    l2_Y_f = l2_Y.astype(np.float64).copy()
    l2_Y_orig = l2_Y_f.copy()

    for it in range(300):
        l2_u8_tmp = np.clip(np.round(l2_Y_f), 0, 255).astype(np.uint8)
        l2_rgb_tmp = ycbcr_to_rgb_bt601(l2_Y_f.astype(np.float32), l2_Cb, l2_Cr)
        pred = bilinear_upsample_2x(l2_rgb_tmp)
        pred_Y, _, _ = rgb_to_ycbcr_bt601(pred)
        residual = l1_mosaic_Y - pred_Y
        residual_down = np.array(Image.fromarray(residual.astype(np.float32)).resize(
            (256, 256), Image.Resampling.BILINEAR))
        l2_Y_f += 0.8 * residual_down
        l2_Y_f = np.clip(l2_Y_f, l2_Y_orig - 25, l2_Y_orig + 25)
        l2_Y_f = np.clip(l2_Y_f, 0, 255)

    l2_opt_25 = ycbcr_to_rgb_bt601(l2_Y_f.astype(np.float32), l2_Cb, l2_Cr)
    l2_psnr_25 = psnr(l2_natural, l2_opt_25)
    result_25 = full_pipeline(l2_opt_25, l1_tiles, l0_tiles, args.quality)
    print(f"  L2 modification:    PSNR vs natural L2 = {l2_psnr_25:.1f} dB")
    print(f"  L1 prediction PSNR: {result_25['l1_pred_psnr']:.2f} dB ({result_25['l1_pred_psnr']-baseline['l1_pred_psnr']:+.2f})")
    print(f"  L1 final PSNR:      {result_25['l1_psnr']:.2f} dB ({result_25['l1_psnr']-baseline['l1_psnr']:+.2f})")
    print(f"  L0 final PSNR:      {result_25['l0_psnr']:.2f} dB ({result_25['l0_psnr']-baseline['l0_psnr']:+.2f})")
    print(f"  Min PSNR:           {result_25['min_psnr']:.2f} dB ({result_25['min_psnr']-baseline['min_psnr']:+.2f})")
    print(f"  Total size:         {result_25['total_bytes']/1024:.1f} KB ({(result_25['total_bytes']-baseline['total_bytes'])/baseline['total_bytes']*100:+.1f}%)")

    # Strategy 2: Matrix-solve optimized L2 (exact solution)
    print(f"\n--- Strategy 2: Matrix-solve optimized L2 (exact least squares) ---")
    l2_opt_exact, l2_Y_opt_exact = optimize_l2_for_prediction(l2_natural, l1_tiles)
    l2_psnr_exact = psnr(l2_natural, l2_opt_exact)
    result_exact = full_pipeline(l2_opt_exact, l1_tiles, l0_tiles, args.quality)
    print(f"  L2 modification:    PSNR vs natural L2 = {l2_psnr_exact:.1f} dB")
    print(f"  L1 prediction PSNR: {result_exact['l1_pred_psnr']:.2f} dB ({result_exact['l1_pred_psnr']-baseline['l1_pred_psnr']:+.2f})")
    print(f"  L1 final PSNR:      {result_exact['l1_psnr']:.2f} dB ({result_exact['l1_psnr']-baseline['l1_psnr']:+.2f})")
    print(f"  L0 final PSNR:      {result_exact['l0_psnr']:.2f} dB ({result_exact['l0_psnr']-baseline['l0_psnr']:+.2f})")
    print(f"  Min PSNR:           {result_exact['min_psnr']:.2f} dB ({result_exact['min_psnr']-baseline['min_psnr']:+.2f})")
    print(f"  Total size:         {result_exact['total_bytes']/1024:.1f} KB ({(result_exact['total_bytes']-baseline['total_bytes'])/baseline['total_bytes']*100:+.1f}%)")

    # Strategy 3: Also test with different JPEG qualities to see if gains are consistent
    print(f"\n{'='*80}")
    print(f"CONSISTENCY CHECK: Optimized L2 across JPEG qualities")
    print(f"{'='*80}")
    print(f"\n{'Quality':>7s}  {'Baseline min':>12s}  {'Opt-10 min':>12s}  {'Δ':>8s}  {'Opt-25 min':>12s}  {'Δ':>8s}  {'Base KB':>8s}  {'Opt KB':>8s}")
    print("-" * 95)

    for q in [30, 40, 50, 60, 70, 80]:
        b = full_pipeline(l2_natural, l1_tiles, l0_tiles, q)
        o10 = full_pipeline(l2_opt_10, l1_tiles, l0_tiles, q)
        o25 = full_pipeline(l2_opt_25, l1_tiles, l0_tiles, q)
        d10 = o10['min_psnr'] - b['min_psnr']
        d25 = o25['min_psnr'] - b['min_psnr']
        print(f"  q={q:2d}    {b['min_psnr']:10.2f}dB  {o10['min_psnr']:10.2f}dB  {d10:+6.2f}dB  "
              f"{o25['min_psnr']:10.2f}dB  {d25:+6.2f}dB  "
              f"{b['total_bytes']/1024:6.1f}KB  {o25['total_bytes']/1024:6.1f}KB")


if __name__ == "__main__":
    main()
