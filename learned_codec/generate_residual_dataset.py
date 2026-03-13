#!/usr/bin/env python3
"""
Generate residual dataset from 1024x1024 WSI tiles.

For each tile:
  1. Downsample 4x → 256x256 (simulate L2)
  2. Optionally JPEG compress at baseq (simulate real L2 artifacts)
  3. Upsample Y channel 4x with lanczos3 → prediction
  4. residual = original_Y - prediction_Y + 128 (centered at 128)
  5. Save as grayscale PNG

This exactly replicates the residuals produced by `origami encode`.

Usage:
  python generate_residual_dataset.py --tiles-dir /path/to/tiles --output /path/to/residuals
  python generate_residual_dataset.py --tiles-dir /path/to/tiles --output /path/to/residuals --baseq 95
"""

import os
import io
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def rgb_to_y(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma from RGB uint8 array [H, W, 3] → [H, W] float32."""
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def process_tile(tile_path: str, output_dir: str, baseq: int = None) -> dict:
    """Process a single tile into a centered luma residual.

    Returns dict with stats or None on failure.
    """
    try:
        img = Image.open(tile_path).convert("RGB")
        w, h = img.size
        if w < 128 or h < 128:
            return None

        original = np.array(img, dtype=np.float32)
        original_y = rgb_to_y(original)

        # Downsample 4x (simulate L2)
        small = img.resize((w // 4, h // 4), Image.LANCZOS)

        # Optional JPEG compression of L2
        if baseq is not None:
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=baseq, subsampling=0)
            buf.seek(0)
            small = Image.open(buf).convert("RGB")

        # Extract Y from L2 and upsample with lanczos3
        small_rgb = np.array(small, dtype=np.float32)
        small_y = rgb_to_y(small_rgb)
        small_y_pil = Image.fromarray(small_y.astype(np.uint8), mode='L')
        pred_y_pil = small_y_pil.resize((w, h), Image.LANCZOS)
        pred_y = np.array(pred_y_pil, dtype=np.float32)

        # Centered residual
        residual = original_y - pred_y + 128.0
        residual = np.clip(residual, 0, 255).astype(np.uint8)

        # Save
        stem = Path(tile_path).stem
        out_path = os.path.join(output_dir, f"{stem}.png")
        Image.fromarray(residual, mode='L').save(out_path)

        # Stats
        raw_diff = original_y - pred_y
        return {
            "file": stem,
            "mean_abs_dev": float(np.mean(np.abs(raw_diff))),
            "max_abs_dev": float(np.max(np.abs(raw_diff))),
            "std_dev": float(np.std(raw_diff)),
            "pct_near_zero": float(np.mean(np.abs(raw_diff) < 5) * 100),
        }
    except Exception as e:
        print(f"Error processing {tile_path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Generate residual dataset from WSI tiles")
    ap.add_argument("--tiles-dir", required=True, help="Directory of 1024x1024 tiles")
    ap.add_argument("--output", required=True, help="Output directory for residual PNGs")
    ap.add_argument("--baseq", type=int, default=None,
                    help="JPEG quality for L2 simulation (None=no compression)")
    ap.add_argument("--upsample-filter", default="lanczos3",
                    help="Upsample filter (only lanczos3 supported currently)")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Max tiles to process (for testing)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel workers")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find tiles
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    tile_paths = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(args.tiles_dir)
        for f in files
        if Path(f).suffix.lower() in exts
    ])

    if args.max_tiles:
        tile_paths = tile_paths[:args.max_tiles]

    print(f"Processing {len(tile_paths)} tiles → {args.output}")
    if args.baseq:
        print(f"  L2 JPEG simulation: Q{args.baseq}")
    else:
        print(f"  L2 JPEG simulation: none (lossless downsample)")

    # Process in parallel
    stats = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_tile, p, args.output, args.baseq): p
            for p in tile_paths
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                stats.append(result)
            if (i + 1) % 500 == 0 or i + 1 == len(tile_paths):
                print(f"  {i + 1}/{len(tile_paths)} tiles processed")

    # Summary
    if stats:
        mean_devs = [s["mean_abs_dev"] for s in stats]
        max_devs = [s["max_abs_dev"] for s in stats]
        near_zero = [s["pct_near_zero"] for s in stats]
        print(f"\nDataset summary ({len(stats)} residuals):")
        print(f"  Mean abs deviation:  {np.mean(mean_devs):.2f} (range: {np.min(mean_devs):.2f} - {np.max(mean_devs):.2f})")
        print(f"  Max abs deviation:   {np.mean(max_devs):.1f} (range: {np.min(max_devs):.1f} - {np.max(max_devs):.1f})")
        print(f"  Near-zero (<5):      {np.mean(near_zero):.1f}% of pixels")

        # Save stats
        import json
        stats_path = os.path.join(args.output, "dataset_stats.json")
        with open(stats_path, "w") as f:
            json.dump({"count": len(stats), "summary": {
                "mean_abs_dev": float(np.mean(mean_devs)),
                "max_abs_dev": float(np.mean(max_devs)),
                "pct_near_zero": float(np.mean(near_zero)),
            }, "per_tile": stats}, f, indent=2)
        print(f"  Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
