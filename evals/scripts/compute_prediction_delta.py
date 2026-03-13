#!/usr/bin/env python3
"""
Compute prediction delta images: SR_prediction - Lanczos3_prediction.

For each SR model run, loads the L0 mosaic prediction (060_L0_mosaic_prediction.png)
and subtracts the corresponding Lanczos3 prediction to visualize what the model
learned beyond simple interpolation.

Outputs per SR run:
  decompress/090_prediction_delta.png          — normalized difference (gray, 128=zero)
  decompress/091_prediction_delta_color.png    — signed color difference (amplified 4x)
  decompress/092_prediction_delta_abs.png      — absolute difference (amplified 4x)

Usage:
  python evals/scripts/compute_prediction_delta.py [run_dirs...]
  python evals/scripts/compute_prediction_delta.py   # all sr_* runs
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image


def compute_delta(sr_run_dir, lanczos_run_dir):
    """Compute prediction delta between SR and Lanczos3 runs at the same L0Q."""
    sr_pred_path = os.path.join(sr_run_dir, "decompress", "060_L0_mosaic_prediction.png")
    l3_pred_path = os.path.join(lanczos_run_dir, "decompress", "060_L0_mosaic_prediction.png")

    if not os.path.exists(sr_pred_path):
        print(f"  SKIP: no prediction image at {sr_pred_path}")
        return False
    if not os.path.exists(l3_pred_path):
        print(f"  SKIP: no lanczos3 prediction at {l3_pred_path}")
        return False

    sr_pred = np.array(Image.open(sr_pred_path)).astype(np.float32)
    l3_pred = np.array(Image.open(l3_pred_path)).astype(np.float32)

    if sr_pred.shape != l3_pred.shape:
        print(f"  SKIP: shape mismatch {sr_pred.shape} vs {l3_pred.shape}")
        return False

    # Signed difference: SR - Lanczos3
    delta = sr_pred - l3_pred

    out_dir = os.path.join(sr_run_dir, "decompress")

    # 1. Normalized gray: center at 128, scale to use full range
    max_abs = max(np.abs(delta).max(), 1e-6)
    gray = ((delta.mean(axis=2) / max_abs) * 127 + 128).clip(0, 255).astype(np.uint8)
    Image.fromarray(gray).save(os.path.join(out_dir, "090_prediction_delta.png"))

    # 2. Signed color difference (amplified 4x, centered at 128)
    color = (delta * 4 + 128).clip(0, 255).astype(np.uint8)
    Image.fromarray(color).save(os.path.join(out_dir, "091_prediction_delta_color.png"))

    # 3. Absolute difference (amplified 4x)
    abs_delta = (np.abs(delta) * 4).clip(0, 255).astype(np.uint8)
    Image.fromarray(abs_delta).save(os.path.join(out_dir, "092_prediction_delta_abs.png"))

    # Stats
    mean_abs = np.abs(delta).mean()
    max_abs_val = np.abs(delta).max()
    std = delta.std()
    print(f"  mean_abs={mean_abs:.2f}  max_abs={max_abs_val:.1f}  std={std:.2f}")

    # Save delta stats to manifest
    manifest_path = os.path.join(sr_run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        manifest["prediction_delta"] = {
            "vs": "lanczos3",
            "mean_abs_per_pixel": round(float(mean_abs), 3),
            "max_abs_per_pixel": round(float(max_abs_val), 1),
            "std_per_pixel": round(float(std), 3),
        }
        json.dump(manifest, open(manifest_path, "w"), indent=2)

    return True


def main():
    runs_dir = "evals/runs"

    if len(sys.argv) > 1:
        sr_dirs = sys.argv[1:]
    else:
        # Find all SR model runs (not lanczos3)
        sr_dirs = sorted([
            os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
            if d.startswith("sr_") and "_f32_" in d or "_i8_" in d
        ])

    if not sr_dirs:
        print("No SR runs found")
        return

    print(f"Computing prediction deltas for {len(sr_dirs)} runs...\n")

    for sr_dir in sr_dirs:
        dir_name = os.path.basename(sr_dir)

        # Extract L0Q from directory name to find matching lanczos3 run
        # Pattern: sr_{model}_{prec}_444_b{seedq}_l0q{N}
        import re
        m = re.search(r"_l0q(\d+)$", dir_name)
        if not m:
            print(f"  SKIP {dir_name}: can't parse L0Q")
            continue
        l0q = m.group(1)

        # Also extract subsamp and seedq
        m2 = re.search(r"_(\d{3})_b(\d+)_l0q", dir_name)
        if not m2:
            print(f"  SKIP {dir_name}: can't parse subsamp/seedq")
            continue
        subsamp, seedq = m2.group(1), m2.group(2)

        lanczos_dir = os.path.join(runs_dir, f"sr_lanczos3_{subsamp}_b{seedq}_l0q{l0q}")
        if not os.path.exists(lanczos_dir):
            print(f"  SKIP {dir_name}: no matching lanczos3 run at {lanczos_dir}")
            continue

        print(f"{dir_name}:")
        compute_delta(sr_dir, lanczos_dir)


if __name__ == "__main__":
    main()
