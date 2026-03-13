#!/usr/bin/env python3
"""Patch baseline manifests with additional visual metrics.

Adds ms_ssim, mse, dssim, ssimulacra, ssimulacra2, butteraugli, blockiness
to existing jpeg_baseline and jpegxl_jpeg_baseline manifests.

Usage:
    uv run python evals/scripts/patch_baseline_metrics.py [run_dirs...]
    uv run python evals/scripts/patch_baseline_metrics.py   # patches all baselines
"""

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

# Add scripts dir for compute_metrics imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compute_metrics import (
    _compute_ms_ssim, calculate_blockiness,
    calculate_ssimulacra, calculate_ssimulacra2, calculate_dssim,
    calculate_butteraugli,
)


def patch_run(run_dir):
    run_dir = pathlib.Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    tiles_dir = run_dir / "tiles"

    if not manifest_path.exists():
        print(f"  Skip {run_dir.name}: no manifest.json")
        return
    if not tiles_dir.exists():
        print(f"  Skip {run_dir.name}: no tiles/")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("type") not in ("jpeg_baseline",):
        # Also handle JXL baselines which have same structure
        pass

    tiles = manifest.get("tiles", {})
    if not tiles:
        print(f"  Skip {run_dir.name}: no tiles in manifest")
        return

    config = manifest.get("configuration", {})
    tile_size = config.get("tile_size", 256)
    input_image = config.get("input_image", "")
    encoder = config.get("encoder", "libjpeg-turbo")
    is_jxl = "jpegxl" in encoder or "jxl" in encoder

    # Load source image
    img_path = pathlib.Path(input_image)
    if not img_path.is_absolute():
        img_path = pathlib.Path(__file__).resolve().parent.parent.parent / img_path
    if not img_path.exists():
        # Try relative to evals/
        img_path = pathlib.Path(__file__).resolve().parent.parent / "test-images" / "L0-1024.jpg"
    if not img_path.exists():
        print(f"  Skip {run_dir.name}: source image not found")
        return

    img_array = np.array(Image.open(img_path).convert("RGB"))
    updated = 0

    print(f"  Patching {run_dir.name}...")

    for tile_key, tile_data in tiles.items():
        # Parse tile key: L0_dx_dy, L1_dx_dy, L2_0_0
        parts = tile_key.split("_")
        level = parts[0]
        dx, dy = int(parts[1]), int(parts[2])

        # Get original tile from source image
        if level == "L0":
            x_start = dx * tile_size
            y_start = dy * tile_size
            original = img_array[y_start:y_start+tile_size, x_start:x_start+tile_size]
        elif level == "L1":
            # L1 is 2x downsampled from 1024 source
            source_1024 = img_array[:1024, :1024]
            l1_full = np.array(Image.fromarray(source_1024).resize((512, 512), Image.LANCZOS))
            x_start = dx * tile_size
            y_start = dy * tile_size
            original = l1_full[y_start:y_start+tile_size, x_start:x_start+tile_size]
        elif level == "L2":
            source_1024 = img_array[:1024, :1024]
            original = np.array(Image.fromarray(source_1024).resize((256, 256), Image.LANCZOS))
        else:
            continue

        # Load compressed tile
        tile_file = tile_data.get("file", "")
        tile_path = tiles_dir / tile_file
        if not tile_path.exists():
            # Try .png version (JXL baselines store decoded PNGs)
            tile_path = tiles_dir / (tile_key + ".png")
        if not tile_path.exists():
            continue

        compressed = np.array(Image.open(tile_path).convert("RGB"))

        if original.shape != compressed.shape:
            continue

        # Compute missing metrics
        needs_update = False

        if "ms_ssim" not in tile_data:
            y_orig = (0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]).astype(np.float32)
            y_comp = (0.299 * compressed[:,:,0] + 0.587 * compressed[:,:,1] + 0.114 * compressed[:,:,2]).astype(np.float32)
            ms_val = _compute_ms_ssim(y_orig, y_comp)
            if ms_val is not None:
                tile_data["ms_ssim"] = ms_val
                needs_update = True

        if "blockiness" not in tile_data:
            tile_data["blockiness"] = calculate_blockiness(compressed)
            tile_data["blockiness_delta"] = tile_data["blockiness"] - calculate_blockiness(original)
            needs_update = True

        # CLI-based metrics need PNG files
        if any(k not in tile_data for k in ["dssim", "ssimulacra", "ssimulacra2", "butteraugli"]):
            # Save temp PNGs for CLI tools
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_ref, \
                 tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_test:
                Image.fromarray(original).save(tmp_ref.name)
                Image.fromarray(compressed).save(tmp_test.name)

                if "dssim" not in tile_data:
                    val = calculate_dssim(tmp_ref.name, tmp_test.name)
                    if val is not None:
                        tile_data["dssim"] = val
                        needs_update = True

                if "ssimulacra" not in tile_data:
                    val = calculate_ssimulacra(tmp_ref.name, tmp_test.name)
                    if val is not None:
                        tile_data["ssimulacra"] = val
                        needs_update = True

                if "ssimulacra2" not in tile_data:
                    val = calculate_ssimulacra2(tmp_ref.name, tmp_test.name)
                    if val is not None:
                        tile_data["ssimulacra2"] = val
                        needs_update = True

                if "butteraugli" not in tile_data:
                    val = calculate_butteraugli(tmp_ref.name, tmp_test.name)
                    if val is not None:
                        tile_data["butteraugli"] = val
                        needs_update = True

                pathlib.Path(tmp_ref.name).unlink(missing_ok=True)
                pathlib.Path(tmp_test.name).unlink(missing_ok=True)

        if needs_update:
            updated += 1

    if updated > 0:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"    Updated {updated}/{len(tiles)} tiles")
    else:
        print(f"    Already up to date")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()

    if args.runs:
        run_dirs = [pathlib.Path(r) for r in args.runs]
    else:
        runs_dir = pathlib.Path("evals/runs")
        run_dirs = sorted(
            d for d in runs_dir.iterdir()
            if d.is_dir() and (d.name.startswith("jpeg_baseline_") or d.name.startswith("jpegxl_jpeg_baseline_"))
        )

    print(f"Patching {len(run_dirs)} baseline run(s)...")
    for run_dir in run_dirs:
        patch_run(run_dir)
    print("Done.")


if __name__ == "__main__":
    main()
