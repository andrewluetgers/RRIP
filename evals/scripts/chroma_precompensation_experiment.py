#!/usr/bin/env python3
"""
chroma_precompensation_experiment.py — Test chroma pre-compensation for 4:2:0.

The problem: RGB open-loop optimization produces great L2 pixel values, but
JPEG encoding at 4:2:0 subsamples chroma (256→128→256), destroying the careful
chroma adjustments.

The idea: Run a second optimization pass that pre-distorts the L2 pixels so
that AFTER the JPEG 4:2:0 round-trip, the result is as close as possible to
the ideal (pre-JPEG) optimized values.

Pipeline:
  1. RGB open-loop optimize L2 → L2_ideal (what we want the decoder to see)
  2. Pre-compensate: find L2_precomp such that JPEG(L2_precomp) ≈ L2_ideal
  3. JPEG encode L2_precomp → decoder gets something close to L2_ideal

This is a two-stage optimization:
  Stage 1: optimize for L1 prediction (existing)
  Stage 2: optimize for surviving JPEG quantization (new)

Also tests: does iterative JPEG pre-compensation converge? How many iterations?
"""

import argparse
import io
import time
import json
import numpy as np
from PIL import Image
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from wsi_residual_debug_with_manifest import (
    tile_image, rgb_to_ycbcr_bt601, ycbcr_to_rgb_bt601, psnr,
)
from optimize_downsample import optimize_for_upsample


def jpeg_roundtrip(rgb_uint8: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode then decode."""
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def jpeg_roundtrip_size(rgb_uint8: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    """JPEG encode then decode, also return size."""
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=quality, optimize=True)
    size = buf.tell()
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB")), size


def measure_prediction(l2_rgb, l1_mosaic, tile_size, baseq=None):
    """Measure L1 prediction quality. If baseq given, JPEG round-trip first."""
    if baseq is not None:
        l2_dec, l2_size = jpeg_roundtrip_size(l2_rgb, baseq)
    else:
        l2_dec = l2_rgb
        l2_size = 0

    pred = np.array(Image.fromarray(l2_dec).resize(
        (tile_size * 2, tile_size * 2), Image.Resampling.BILINEAR))

    tY, tCb, tCr = rgb_to_ycbcr_bt601(l1_mosaic)
    pY, pCb, pCr = rgb_to_ycbcr_bt601(pred)

    return {
        "psnr_y": float(psnr(tY, pY, data_range=255)),
        "psnr_cb": float(psnr(tCb, pCb, data_range=255)),
        "psnr_cr": float(psnr(tCr, pCr, data_range=255)),
        "psnr_rgb": float(psnr(l1_mosaic, pred, data_range=255)),
        "l2_size": l2_size,
        "energy": float(np.sum((l1_mosaic.astype(np.float64) - pred.astype(np.float64)) ** 2)),
    }


def precompensate_jpeg(source: np.ndarray, target: np.ndarray, quality: int,
                       n_iterations: int = 50, lr: float = 0.5) -> np.ndarray:
    """Find pixels that, after JPEG encoding, produce output close to target.

    Uses gradient descent: the loss is ||JPEG(source) - target||^2.
    Gradient is approximated as straight-through (identity through JPEG).

    Unlike JPEG-in-loop for L1 prediction (which failed because it conflicted
    with residual coding), here the target IS what we want the JPEG output to be.
    There's no downstream residual stage to conflict with.

    Args:
        source: Starting point (typically the open-loop optimized L2), uint8.
        target: What we want the JPEG output to look like (same as source ideally), uint8.
        quality: JPEG quality for the round-trip.
        n_iterations: Optimization iterations.
        lr: Learning rate.

    Returns:
        Pre-compensated pixels, uint8.
    """
    target_f = target.astype(np.float64)
    source_f = source.astype(np.float64)
    best_mse = float("inf")
    best_source = source_f.copy()

    for i in range(n_iterations):
        cur = np.clip(source_f, 0, 255).astype(np.uint8)
        decoded = jpeg_roundtrip(cur, quality).astype(np.float64)

        residual = target_f - decoded
        mse = np.mean(residual * residual)

        if mse < best_mse:
            best_mse = mse
            best_source = source_f.copy()

        # Straight-through gradient: d(loss)/d(source) ≈ -2*(target - decoded)
        # This works here because we're directly optimizing source→JPEG(source)→target,
        # not going through an additional upsample/residual chain.
        source_f += lr * residual
        np.clip(source_f, 0, 255, out=source_f)

    return np.clip(best_source, 0, 255).astype(np.uint8)


def precompensate_iterative(source: np.ndarray, quality: int,
                            n_iterations: int = 10) -> np.ndarray:
    """Simple iterative pre-compensation: source += (source - JPEG(source)) each step.

    This is the classic "predict the error, add it back" approach.
    Converges quickly for small quantization errors (high quality).
    """
    current = source.astype(np.float64)
    target = source.astype(np.float64)

    for i in range(n_iterations):
        cur_u8 = np.clip(current, 0, 255).astype(np.uint8)
        decoded = jpeg_roundtrip(cur_u8, quality).astype(np.float64)
        error = target - decoded
        current = current + error
        np.clip(current, 0, 255, out=current)

    return np.clip(current, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Chroma pre-compensation experiment for 4:2:0")
    parser.add_argument("--image", required=True, help="1024x1024 test image")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--baseq", type=int, default=95)
    parser.add_argument("--max-delta", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    print(f"Image: {args.image}")
    print(f"L2 JPEG quality: {args.baseq}")
    print()

    l2_tile, l1_tiles, l0_tiles = tile_image(args.image, args.tile)
    tile_size = args.tile

    l1_mosaic = np.zeros((tile_size * 2, tile_size * 2, 3), dtype=np.uint8)
    for (dx, dy), l1_gt in l1_tiles.items():
        l1_mosaic[dy * tile_size:(dy + 1) * tile_size,
                  dx * tile_size:(dx + 1) * tile_size] = l1_gt

    # --- Step 1: Measure baseline and open-loop ---
    print("=== Baseline Measurements ===")

    m_none = measure_prediction(l2_tile, l1_mosaic, tile_size, baseq=args.baseq)
    print(f"No optimization (through JPEG q={args.baseq}):")
    print(f"  Y={m_none['psnr_y']:.2f}  Cb={m_none['psnr_cb']:.2f}  "
          f"Cr={m_none['psnr_cr']:.2f}  RGB={m_none['psnr_rgb']:.2f}  "
          f"L2={m_none['l2_size']/1024:.1f}KB")

    print(f"\nOptimizing L2 (RGB open-loop, {args.iterations} iters)...")
    t0 = time.time()
    l2_opt = optimize_for_upsample(
        l2_tile, l1_mosaic, max_delta=args.max_delta,
        n_iterations=args.iterations, lr=0.3
    )
    t_opt = time.time() - t0
    print(f"  Optimization time: {t_opt:.1f}s")

    m_opt_ideal = measure_prediction(l2_opt, l1_mosaic, tile_size, baseq=None)
    print(f"RGB open-loop (no JPEG, ideal):")
    print(f"  Y={m_opt_ideal['psnr_y']:.2f}  Cb={m_opt_ideal['psnr_cb']:.2f}  "
          f"Cr={m_opt_ideal['psnr_cr']:.2f}  RGB={m_opt_ideal['psnr_rgb']:.2f}")

    m_opt_jpeg = measure_prediction(l2_opt, l1_mosaic, tile_size, baseq=args.baseq)
    print(f"RGB open-loop (through JPEG q={args.baseq}):")
    print(f"  Y={m_opt_jpeg['psnr_y']:.2f}  Cb={m_opt_jpeg['psnr_cb']:.2f}  "
          f"Cr={m_opt_jpeg['psnr_cr']:.2f}  RGB={m_opt_jpeg['psnr_rgb']:.2f}  "
          f"L2={m_opt_jpeg['l2_size']/1024:.1f}KB")

    # How much does JPEG destroy?
    jpeg_loss_y = m_opt_ideal['psnr_y'] - m_opt_jpeg['psnr_y']
    jpeg_loss_cb = m_opt_ideal['psnr_cb'] - m_opt_jpeg['psnr_cb']
    jpeg_loss_cr = m_opt_ideal['psnr_cr'] - m_opt_jpeg['psnr_cr']
    jpeg_loss_rgb = m_opt_ideal['psnr_rgb'] - m_opt_jpeg['psnr_rgb']
    print(f"\nJPEG q={args.baseq} damage to optimized L2:")
    print(f"  dY={jpeg_loss_y:+.2f}  dCb={jpeg_loss_cb:+.2f}  "
          f"dCr={jpeg_loss_cr:+.2f}  dRGB={jpeg_loss_rgb:+.2f}")

    # --- Step 2: Pre-compensation approaches ---
    print("\n=== Pre-compensation Approaches ===")

    # 2a. Iterative pre-compensation (simple error feedback)
    for n_iters in [1, 3, 5, 10, 20]:
        t0 = time.time()
        l2_precomp = precompensate_iterative(l2_opt, args.baseq, n_iterations=n_iters)
        t_precomp = time.time() - t0

        m = measure_prediction(l2_precomp, l1_mosaic, tile_size, baseq=args.baseq)
        d_y = m['psnr_y'] - m_opt_jpeg['psnr_y']
        d_cb = m['psnr_cb'] - m_opt_jpeg['psnr_cb']
        d_cr = m['psnr_cr'] - m_opt_jpeg['psnr_cr']
        d_rgb = m['psnr_rgb'] - m_opt_jpeg['psnr_rgb']
        print(f"Iterative precomp ({n_iters:2d} iters, {t_precomp:.2f}s):"
              f"  Y={m['psnr_y']:.2f}({d_y:+.2f})  Cb={m['psnr_cb']:.2f}({d_cb:+.2f})"
              f"  Cr={m['psnr_cr']:.2f}({d_cr:+.2f})  RGB={m['psnr_rgb']:.2f}({d_rgb:+.2f})"
              f"  L2={m['l2_size']/1024:.1f}KB")

    # 2b. Gradient descent pre-compensation
    for n_iters in [10, 30, 50]:
        t0 = time.time()
        l2_precomp_gd = precompensate_jpeg(l2_opt, l2_opt, args.baseq,
                                           n_iterations=n_iters, lr=0.5)
        t_precomp = time.time() - t0

        m = measure_prediction(l2_precomp_gd, l1_mosaic, tile_size, baseq=args.baseq)
        d_y = m['psnr_y'] - m_opt_jpeg['psnr_y']
        d_cb = m['psnr_cb'] - m_opt_jpeg['psnr_cb']
        d_cr = m['psnr_cr'] - m_opt_jpeg['psnr_cr']
        d_rgb = m['psnr_rgb'] - m_opt_jpeg['psnr_rgb']
        print(f"GD precomp ({n_iters:2d} iters, {t_precomp:.2f}s):"
              f"  Y={m['psnr_y']:.2f}({d_y:+.2f})  Cb={m['psnr_cb']:.2f}({d_cb:+.2f})"
              f"  Cr={m['psnr_cr']:.2f}({d_cr:+.2f})  RGB={m['psnr_rgb']:.2f}({d_rgb:+.2f})"
              f"  L2={m['l2_size']/1024:.1f}KB")

    # --- Step 3: Combined pipeline (optimize → precompensate) vs baseline ---
    print("\n=== Combined Pipeline: Optimize + Precompensate ===")

    best_precomp = precompensate_iterative(l2_opt, args.baseq, n_iterations=10)
    m_combined = measure_prediction(best_precomp, l1_mosaic, tile_size, baseq=args.baseq)

    print(f"\n{'Approach':<35}  {'Y':>7}  {'Cb':>7}  {'Cr':>7}  {'RGB':>7}  {'L2 KB':>7}  {'Energy':>12}")
    print("-" * 95)

    for label, m in [
        ("No optimization", m_none),
        ("RGB open-loop (through JPEG)", m_opt_jpeg),
        ("RGB open-loop (ideal, no JPEG)", m_opt_ideal),
        ("RGB open-loop + precompensate", m_combined),
    ]:
        size_str = f"{m['l2_size']/1024:.1f}" if m['l2_size'] > 0 else "—"
        print(f"{label:<35}  {m['psnr_y']:7.2f}  {m['psnr_cb']:7.2f}  "
              f"{m['psnr_cr']:7.2f}  {m['psnr_rgb']:7.2f}  {size_str:>7}  {m['energy']:12.0f}")

    # How much of the JPEG damage did we recover?
    recovered_y = (m_combined['psnr_y'] - m_opt_jpeg['psnr_y']) / jpeg_loss_y * 100 if jpeg_loss_y > 0 else 0
    recovered_cb = (m_combined['psnr_cb'] - m_opt_jpeg['psnr_cb']) / jpeg_loss_cb * 100 if jpeg_loss_cb > 0 else 0
    recovered_cr = (m_combined['psnr_cr'] - m_opt_jpeg['psnr_cr']) / jpeg_loss_cr * 100 if jpeg_loss_cr > 0 else 0
    recovered_rgb = (m_combined['psnr_rgb'] - m_opt_jpeg['psnr_rgb']) / jpeg_loss_rgb * 100 if jpeg_loss_rgb > 0 else 0
    print(f"\nJPEG damage recovered: Y={recovered_y:.0f}%  Cb={recovered_cb:.0f}%  "
          f"Cr={recovered_cr:.0f}%  RGB={recovered_rgb:.0f}%")


if __name__ == "__main__":
    main()
