#!/usr/bin/env python3
"""
chroma_downsample_optimization_experiment.py

Test optimizing the 4:2:0 chroma downsample using the same open-loop gradient
descent we use for L2→L1 prediction.

The 4:2:0 chroma pipeline is:
  Cb/Cr at 256x256 → downsample to 128x128 → store → upsample to 256x256

This is exactly the same pattern as our L2 optimization:
  L2 at 256x256 → store → upsample to 512x512

So we apply optimize_for_upsample_grayscale to each chroma plane:
  optimize_for_upsample_grayscale(chroma_128, chroma_256)

The question: can we control the chroma planes that go into JPEG independently?
Pillow/libjpeg doesn't expose subsampled chroma directly. But we CAN:
  1. Convert RGB → YCbCr
  2. Downsample Cb/Cr to 128x128 (our optimized version)
  3. Upsample back to 256x256 (simulating what the decoder does)
  4. Convert back to RGB with the upsampled chroma
  5. JPEG encode at 4:4:4 — the chroma is already at "post-4:2:0" quality
     but now optimized, and 4:4:4 preserves it exactly

OR even simpler: since we can't control libjpeg's internal subsampling,
we pre-bake the optimized chroma into the RGB pixels before encoding.
The JPEG encoder's own 4:2:0 will then subsample our already-optimized chroma,
which should be close to a no-op since our optimized values are already
the best 128x128 representation.

Let's test both: the ideal case (what if we could control chroma directly)
and the practical case (pre-baked into RGB before JPEG encoding).
"""

import argparse
import io
import time
import numpy as np
from PIL import Image
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from wsi_residual_debug_with_manifest import (
    tile_image, rgb_to_ycbcr_bt601, ycbcr_to_rgb_bt601, psnr,
)
from optimize_downsample import optimize_for_upsample, optimize_for_upsample_grayscale


def jpeg_roundtrip(rgb_uint8: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def jpeg_roundtrip_size(rgb_uint8: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=quality, optimize=True)
    size = buf.tell()
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB")), size


def measure_l1_prediction(l2_rgb, l1_mosaic, tile_size, baseq=None):
    """Measure L1 prediction quality, optionally through JPEG."""
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
        "y": float(psnr(tY, pY, data_range=255)),
        "cb": float(psnr(tCb, pCb, data_range=255)),
        "cr": float(psnr(tCr, pCr, data_range=255)),
        "rgb": float(psnr(l1_mosaic, pred, data_range=255)),
        "l2_size": l2_size,
    }


def optimize_chroma_downsample(rgb_256: np.ndarray, n_iterations=100, lr=0.3,
                                max_delta=30):
    """Optimize the 4:2:0 chroma downsample for an RGB image.

    Takes 256x256 RGB, optimizes Cb/Cr at 128x128 so that when upsampled
    back to 256x256, they best match the original 256x256 Cb/Cr.

    Returns RGB with the optimized chroma baked in (Y unchanged).
    """
    Y, Cb, Cr = rgb_to_ycbcr_bt601(rgb_256)

    # Downsample chroma to 128x128 (starting point)
    h, w = Cb.shape
    Cb_128 = np.array(Image.fromarray(Cb.astype(np.float32)).resize(
        (w // 2, h // 2), Image.Resampling.LANCZOS))
    Cr_128 = np.array(Image.fromarray(Cr.astype(np.float32)).resize(
        (w // 2, h // 2), Image.Resampling.LANCZOS))

    # Optimize each chroma plane
    Cb_128_opt = optimize_for_upsample_grayscale(
        Cb_128, Cb, max_delta=max_delta, n_iterations=n_iterations, lr=lr)
    Cr_128_opt = optimize_for_upsample_grayscale(
        Cr_128, Cr, max_delta=max_delta, n_iterations=n_iterations, lr=lr)

    # Upsample optimized chroma back to 256x256
    Cb_opt_256 = np.array(Image.fromarray(Cb_128_opt.astype(np.float32)).resize(
        (w, h), Image.Resampling.BILINEAR))
    Cr_opt_256 = np.array(Image.fromarray(Cr_128_opt.astype(np.float32)).resize(
        (w, h), Image.Resampling.BILINEAR))

    # Reconstruct RGB with original Y + optimized chroma
    return ycbcr_to_rgb_bt601(Y, Cb_opt_256, Cr_opt_256)


def main():
    parser = argparse.ArgumentParser(
        description="Chroma downsample optimization experiment")
    parser.add_argument("--image", required=True, help="1024x1024 test image")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--baseq", type=int, default=95)
    parser.add_argument("--max-delta", type=int, default=15)
    parser.add_argument("--opt-iterations", type=int, default=200,
                        help="Iterations for L2→L1 optimization")
    parser.add_argument("--chroma-iterations", type=int, default=100,
                        help="Iterations for chroma 256→128 optimization")
    parser.add_argument("--chroma-max-delta", type=int, default=30,
                        help="Max delta for chroma optimization (chroma has more room)")
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

    # --- Measure chroma downsample damage in isolation ---
    print("=== Chroma 4:2:0 Downsample Quality (L2 tile only) ===")
    Y_orig, Cb_orig, Cr_orig = rgb_to_ycbcr_bt601(l2_tile)
    h, w = Cb_orig.shape

    # Naive downsample→upsample
    Cb_128_naive = np.array(Image.fromarray(Cb_orig.astype(np.float32)).resize(
        (w // 2, h // 2), Image.Resampling.LANCZOS))
    Cr_128_naive = np.array(Image.fromarray(Cr_orig.astype(np.float32)).resize(
        (w // 2, h // 2), Image.Resampling.LANCZOS))
    Cb_naive_up = np.array(Image.fromarray(Cb_128_naive).resize(
        (w, h), Image.Resampling.BILINEAR))
    Cr_naive_up = np.array(Image.fromarray(Cr_128_naive).resize(
        (w, h), Image.Resampling.BILINEAR))

    print(f"  Naive 4:2:0 Cb PSNR: {psnr(Cb_orig, Cb_naive_up, data_range=255):.2f} dB")
    print(f"  Naive 4:2:0 Cr PSNR: {psnr(Cr_orig, Cr_naive_up, data_range=255):.2f} dB")

    # Optimized downsample→upsample
    print(f"\n  Optimizing chroma downsample ({args.chroma_iterations} iters, "
          f"max_delta=±{args.chroma_max_delta})...")
    t0 = time.time()
    Cb_128_opt = optimize_for_upsample_grayscale(
        Cb_128_naive, Cb_orig, max_delta=args.chroma_max_delta,
        n_iterations=args.chroma_iterations, lr=0.3)
    Cr_128_opt = optimize_for_upsample_grayscale(
        Cr_128_naive, Cr_orig, max_delta=args.chroma_max_delta,
        n_iterations=args.chroma_iterations, lr=0.3)
    t_chroma = time.time() - t0

    Cb_opt_up = np.array(Image.fromarray(Cb_128_opt.astype(np.float32)).resize(
        (w, h), Image.Resampling.BILINEAR))
    Cr_opt_up = np.array(Image.fromarray(Cr_128_opt.astype(np.float32)).resize(
        (w, h), Image.Resampling.BILINEAR))

    print(f"  Optimized 4:2:0 Cb PSNR: {psnr(Cb_orig, Cb_opt_up, data_range=255):.2f} dB  "
          f"(+{psnr(Cb_orig, Cb_opt_up, data_range=255) - psnr(Cb_orig, Cb_naive_up, data_range=255):.2f})")
    print(f"  Optimized 4:2:0 Cr PSNR: {psnr(Cr_orig, Cr_opt_up, data_range=255):.2f} dB  "
          f"(+{psnr(Cr_orig, Cr_opt_up, data_range=255) - psnr(Cr_orig, Cr_naive_up, data_range=255):.2f})")
    print(f"  Chroma optimization time: {t_chroma:.2f}s")

    # --- Full pipeline comparison ---
    print("\n=== Full Pipeline: L2→L1 Prediction Quality ===")
    strategies = {}

    # A. No optimization
    m = measure_l1_prediction(l2_tile, l1_mosaic, tile_size, baseq=args.baseq)
    strategies["none"] = m
    print(f"[A] No optimization:                "
          f"Y={m['y']:.2f}  Cb={m['cb']:.2f}  Cr={m['cr']:.2f}  RGB={m['rgb']:.2f}  "
          f"L2={m['l2_size']/1024:.1f}KB")

    # B. RGB open-loop L2 optimization only
    print(f"\n  Optimizing L2 (RGB open-loop, {args.opt_iterations} iters)...")
    t0 = time.time()
    l2_opt = optimize_for_upsample(
        l2_tile, l1_mosaic, max_delta=args.max_delta,
        n_iterations=args.opt_iterations, lr=0.3)
    t_l2_opt = time.time() - t0

    m = measure_l1_prediction(l2_opt, l1_mosaic, tile_size, baseq=args.baseq)
    strategies["rgb_open"] = m
    print(f"[B] RGB open-loop:                  "
          f"Y={m['y']:.2f}  Cb={m['cb']:.2f}  Cr={m['cr']:.2f}  RGB={m['rgb']:.2f}  "
          f"L2={m['l2_size']/1024:.1f}KB  ({t_l2_opt:.1f}s)")

    # C. RGB open-loop + chroma downsample optimization
    # Optimize the chroma planes of the already-L2-optimized tile
    print(f"\n  Optimizing chroma downsample of L2-optimized tile...")
    t0 = time.time()
    l2_opt_chroma = optimize_chroma_downsample(
        l2_opt, n_iterations=args.chroma_iterations, lr=0.3,
        max_delta=args.chroma_max_delta)
    t_chroma_opt = time.time() - t0

    m = measure_l1_prediction(l2_opt_chroma, l1_mosaic, tile_size, baseq=args.baseq)
    strategies["rgb_open_chroma"] = m
    total_time = t_l2_opt + t_chroma_opt
    print(f"[C] RGB open-loop + chroma opt:     "
          f"Y={m['y']:.2f}  Cb={m['cb']:.2f}  Cr={m['cr']:.2f}  RGB={m['rgb']:.2f}  "
          f"L2={m['l2_size']/1024:.1f}KB  ({total_time:.1f}s)")

    # D. Chroma optimization ONLY (no L2 optimization)
    print(f"\n  Optimizing chroma downsample of original tile...")
    t0 = time.time()
    l2_chroma_only = optimize_chroma_downsample(
        l2_tile, n_iterations=args.chroma_iterations, lr=0.3,
        max_delta=args.chroma_max_delta)
    t_chroma_only = time.time() - t0

    m = measure_l1_prediction(l2_chroma_only, l1_mosaic, tile_size, baseq=args.baseq)
    strategies["chroma_only"] = m
    print(f"[D] Chroma opt only (no L2 opt):    "
          f"Y={m['y']:.2f}  Cb={m['cb']:.2f}  Cr={m['cr']:.2f}  RGB={m['rgb']:.2f}  "
          f"L2={m['l2_size']/1024:.1f}KB  ({t_chroma_only:.1f}s)")

    # --- Deltas ---
    print(f"\n=== Deltas vs No Optimization (through JPEG q={args.baseq}) ===")
    base = strategies["none"]
    for label, key in [
        ("RGB open-loop", "rgb_open"),
        ("RGB open-loop + chroma opt", "rgb_open_chroma"),
        ("Chroma opt only", "chroma_only"),
    ]:
        s = strategies[key]
        print(f"  {label:<30}  dY={s['y']-base['y']:+.2f}  dCb={s['cb']-base['cb']:+.2f}  "
              f"dCr={s['cr']-base['cr']:+.2f}  dRGB={s['rgb']-base['rgb']:+.2f}  "
              f"dL2={+(s['l2_size']-base['l2_size'])/1024:+.1f}KB")

    # --- What does the chroma optimization do to the L2 tile? ---
    print(f"\n=== L2 Tile Analysis ===")
    # Compare L2 pixels: original vs optimized vs chroma-optimized
    for label, l2 in [("Original", l2_tile), ("RGB open-loop", l2_opt),
                       ("RGB+chroma opt", l2_opt_chroma)]:
        Y, Cb, Cr = rgb_to_ycbcr_bt601(l2)
        print(f"  {label:<20}  Y range=[{Y.min():.0f},{Y.max():.0f}]  "
              f"Cb range=[{Cb.min():.0f},{Cb.max():.0f}]  "
              f"Cr range=[{Cr.min():.0f},{Cr.max():.0f}]")

    # Show pixel differences
    diff = np.abs(l2_opt_chroma.astype(np.float64) - l2_opt.astype(np.float64))
    print(f"\n  Chroma opt changes to L2 pixels (vs RGB open-loop):")
    print(f"    Mean abs diff: R={diff[:,:,0].mean():.2f}  G={diff[:,:,1].mean():.2f}  "
          f"B={diff[:,:,2].mean():.2f}")
    print(f"    Max abs diff:  R={diff[:,:,0].max():.0f}  G={diff[:,:,1].max():.0f}  "
          f"B={diff[:,:,2].max():.0f}")


if __name__ == "__main__":
    main()
