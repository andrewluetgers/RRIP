#!/usr/bin/env python3
"""
jpeg_in_loop_experiment.py — Test JPEG-in-the-loop optimization for L2 tiles.

Current OptL2 optimizes pixel values in the "ideal" domain, then the JPEG encoder
destroys some of those adjustments (especially chroma via 4:2:0 subsampling).

JPEG-in-the-loop ("closed-loop") optimization:
  1. Start with source pixels
  2. Each iteration: JPEG encode → JPEG decode → bilinear upsample → measure error
  3. Gradient: computed as surrogate (adjoint of upsample, ignoring JPEG)
  4. The loss INCLUDES JPEG quantization artifacts, so the optimizer learns pixels
     that survive the round-trip well

This is a straight-through estimator: forward pass includes JPEG, backward pass
approximates the gradient by ignoring the non-differentiable quantization step.

Compares three approaches at each L2 quality:
  A. No optimization (baseline)
  B. Open-loop optimization (current OptL2 — optimize then JPEG encode once)
  C. JPEG-in-the-loop (optimize with JPEG encode/decode every iteration)
"""

import argparse
import io
import time
import numpy as np
from PIL import Image
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from wsi_residual_debug_with_manifest import tile_image, psnr


def jpeg_roundtrip(rgb_uint8: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode then decode an RGB image. Returns decoded RGB uint8."""
    img = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def jpeg_roundtrip_size(rgb_uint8: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    """JPEG encode then decode, also return file size."""
    img = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    size = buf.tell()
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB")), size


def optimize_open_loop(
    source: np.ndarray,
    target: np.ndarray,
    max_delta: int = 15,
    n_iterations: int = 100,
    lr: float = 0.3,
) -> np.ndarray:
    """Current approach: optimize pixels directly, no JPEG in the loop."""
    target_h, target_w = target.shape[:2]
    source_h, source_w = source.shape[:2]

    target_f = target.astype(np.float64)
    source_f = source.astype(np.float64)
    source_orig = source_f.copy()
    best_energy = float("inf")
    best_source = source_f.copy()

    for _ in range(n_iterations):
        cur = np.clip(source_f, 0, 255).astype(np.uint8)
        pred = np.array(
            Image.fromarray(cur).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
        ).astype(np.float64)

        residual = target_f - pred
        energy = np.sum(residual * residual)

        if energy < best_energy:
            best_energy = energy
            best_source = source_f.copy()

        grad = np.empty_like(source_f)
        for c in range(source_f.shape[2]):
            grad[:, :, c] = np.array(
                Image.fromarray(residual[:, :, c].astype(np.float32)).resize(
                    (source_w, source_h), Image.Resampling.BILINEAR
                )
            )

        source_f += lr * grad
        np.clip(source_f, source_orig - max_delta, source_orig + max_delta, out=source_f)
        np.clip(source_f, 0, 255, out=source_f)

    return np.clip(best_source, 0, 255).astype(np.uint8)


def optimize_jpeg_in_loop(
    source: np.ndarray,
    target: np.ndarray,
    jpeg_quality: int,
    max_delta: int = 15,
    n_iterations: int = 100,
    lr: float = 0.3,
) -> np.ndarray:
    """JPEG-in-the-loop: encode/decode L2 each iteration before measuring error.

    The forward pass is:  source → quantize → JPEG encode → JPEG decode → upsample → error
    The gradient is computed as a straight-through estimator: we approximate the
    gradient of the JPEG step as identity (pass gradients through unchanged).
    """
    target_h, target_w = target.shape[:2]
    source_h, source_w = source.shape[:2]

    target_f = target.astype(np.float64)
    source_f = source.astype(np.float64)
    source_orig = source_f.copy()
    best_energy = float("inf")
    best_source = source_f.copy()

    for _ in range(n_iterations):
        # Quantize to uint8, then JPEG round-trip
        cur = np.clip(source_f, 0, 255).astype(np.uint8)
        jpeg_decoded = jpeg_roundtrip(cur, jpeg_quality)

        # Upsample the JPEG-decoded version (what the decoder actually sees)
        pred = np.array(
            Image.fromarray(jpeg_decoded).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
        ).astype(np.float64)

        # Loss is measured THROUGH the JPEG codec
        residual = target_f - pred
        energy = np.sum(residual * residual)

        if energy < best_energy:
            best_energy = energy
            best_source = source_f.copy()

        # Straight-through gradient: ignore JPEG in backward pass
        grad = np.empty_like(source_f)
        for c in range(source_f.shape[2]):
            grad[:, :, c] = np.array(
                Image.fromarray(residual[:, :, c].astype(np.float32)).resize(
                    (source_w, source_h), Image.Resampling.BILINEAR
                )
            )

        source_f += lr * grad
        np.clip(source_f, source_orig - max_delta, source_orig + max_delta, out=source_f)
        np.clip(source_f, 0, 255, out=source_f)

    return np.clip(best_source, 0, 255).astype(np.uint8)


def measure_prediction_quality(l2_rgb, l1_mosaic, jpeg_quality=None):
    """Measure prediction quality of L2 → L1 via bilinear upsample.

    If jpeg_quality is given, JPEG-encode/decode L2 first (simulating real pipeline).
    Returns (psnr_y, psnr_cb, psnr_cr, psnr_rgb, energy, l2_size_bytes).
    """
    tile_h, tile_w = l1_mosaic.shape[:2]

    if jpeg_quality is not None:
        l2_decoded, l2_size = jpeg_roundtrip_size(l2_rgb, jpeg_quality)
    else:
        l2_decoded = l2_rgb
        l2_size = 0

    pred = np.array(
        Image.fromarray(l2_decoded).resize(
            (tile_w, tile_h), Image.Resampling.BILINEAR
        )
    )

    # Overall RGB PSNR
    psnr_rgb = psnr(l1_mosaic, pred, data_range=255)

    # Per-channel in YCbCr
    from wsi_residual_debug_with_manifest import rgb_to_ycbcr_bt601
    tY, tCb, tCr = rgb_to_ycbcr_bt601(l1_mosaic)
    pY, pCb, pCr = rgb_to_ycbcr_bt601(pred)

    psnr_y = psnr(tY, pY, data_range=255)
    psnr_cb = psnr(tCb, pCb, data_range=255)
    psnr_cr = psnr(tCr, pCr, data_range=255)

    energy = np.sum((l1_mosaic.astype(np.float64) - pred.astype(np.float64)) ** 2)

    return psnr_y, psnr_cb, psnr_cr, psnr_rgb, energy, l2_size


def main():
    parser = argparse.ArgumentParser(description="JPEG-in-the-loop optimization experiment")
    parser.add_argument("--image", required=True, help="1024x1024 test image")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--max-delta", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--qualities", type=str, default="80,85,90,95,99",
                        help="Comma-separated L2 JPEG qualities to test")
    args = parser.parse_args()

    qualities = [int(q) for q in args.qualities.split(",")]

    print(f"Image: {args.image}")
    print(f"Max delta: ±{args.max_delta}, Iterations: {args.iterations}, LR: {args.lr}")
    print()

    # Tile the image
    l2_tile, l1_tiles, l0_tiles = tile_image(args.image, args.tile)

    # Build L1 mosaic
    tile_size = args.tile
    l1_mosaic = np.zeros((tile_size * 2, tile_size * 2, 3), dtype=np.uint8)
    for (dx, dy), l1_gt in l1_tiles.items():
        l1_mosaic[dy * tile_size:(dy + 1) * tile_size,
                  dx * tile_size:(dx + 1) * tile_size] = l1_gt

    print(f"{'Q':>3}  {'Method':<20}  {'RGB PSNR':>9}  {'Y PSNR':>8}  {'Cb PSNR':>8}  "
          f"{'Cr PSNR':>8}  {'Energy':>12}  {'L2 KB':>7}  {'Time':>6}")
    print("-" * 105)

    for q in qualities:
        # A. No optimization — just JPEG round-trip the original L2
        py, pcb, pcr, prgb, energy_a, size_a = measure_prediction_quality(
            l2_tile, l1_mosaic, jpeg_quality=q
        )
        print(f"{q:3d}  {'No optimization':<20}  {prgb:9.2f}  {py:8.2f}  {pcb:8.2f}  "
              f"{pcr:8.2f}  {energy_a:12.0f}  {size_a/1024:7.1f}  {'—':>6}")

        # B. Open-loop optimization (current approach)
        t0 = time.time()
        l2_open = optimize_open_loop(
            l2_tile, l1_mosaic,
            max_delta=args.max_delta, n_iterations=args.iterations, lr=args.lr
        )
        t_open = time.time() - t0

        # Measure through JPEG (the real pipeline)
        py, pcb, pcr, prgb, energy_b, size_b = measure_prediction_quality(
            l2_open, l1_mosaic, jpeg_quality=q
        )
        delta_b = prgb - measure_prediction_quality(l2_tile, l1_mosaic, jpeg_quality=q)[3]
        print(f"{q:3d}  {'Open-loop OptL2':<20}  {prgb:9.2f}  {py:8.2f}  {pcb:8.2f}  "
              f"{pcr:8.2f}  {energy_b:12.0f}  {size_b/1024:7.1f}  {t_open:5.1f}s")

        # C. JPEG-in-the-loop optimization
        t0 = time.time()
        l2_jloop = optimize_jpeg_in_loop(
            l2_tile, l1_mosaic, jpeg_quality=q,
            max_delta=args.max_delta, n_iterations=args.iterations, lr=args.lr
        )
        t_jloop = time.time() - t0

        # Measure through JPEG
        py, pcb, pcr, prgb, energy_c, size_c = measure_prediction_quality(
            l2_jloop, l1_mosaic, jpeg_quality=q
        )
        delta_c = prgb - measure_prediction_quality(l2_tile, l1_mosaic, jpeg_quality=q)[3]
        print(f"{q:3d}  {'JPEG-in-loop OptL2':<20}  {prgb:9.2f}  {py:8.2f}  {pcb:8.2f}  "
              f"{pcr:8.2f}  {energy_c:12.0f}  {size_c/1024:7.1f}  {t_jloop:5.1f}s")

        # Summary for this quality
        improve_open = 1.0 - energy_b / energy_a if energy_a > 0 else 0
        improve_jloop = 1.0 - energy_c / energy_a if energy_a > 0 else 0
        print(f"     → Open-loop: {improve_open:+.1%} energy, {delta_b:+.2f} dB  |  "
              f"JPEG-in-loop: {improve_jloop:+.1%} energy, {delta_c:+.2f} dB")
        print()


if __name__ == "__main__":
    main()
