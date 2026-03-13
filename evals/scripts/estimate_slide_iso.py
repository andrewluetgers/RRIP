#!/usr/bin/env python3
"""Estimate optimal JXL photon_noise_iso for a whole slide.

Samples tissue tiles from a DZI pyramid at 256px, measures noise statistics via
wavelet MAD (RMS across all subbands), scales to 1024px equivalent, and
interpolates a pre-built calibration table to recommend an ISO setting.

The calibration table maps 1024px σ² → ISO at a given JXL quality level.
It was built by encoding a reference 1024px tissue image at various ISOs and
measuring the decoded σ² with the same estimator.

Usage:
    uv run python evals/scripts/estimate_slide_iso.py \
        --pyramid data/dzi/jpeg90 \
        --quality 40 \
        [--samples 10]
"""

import argparse
import json
import pathlib
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pywt


# Pre-built calibration tables: quality → list of (iso, σ²_1024)
# Built from evals/test-images/L0-1024.jpg encoded at each quality+ISO, decoded,
# measured with sigma_rms_all. The 256px→1024px variance scale factor is ~2.3x.
CALIBRATION = {
    40: [
        (0,     21.3481),
        (6400,  29.5125),
        (9600,  33.9719),
        (12800, 37.7548),
        (14400, 39.9561),
        (16000, 40.9626),
        (19200, 44.8394),
        (22400, 49.1482),
        (25600, 52.3932),
    ],
}

# Variance scale factor: 256px measurement → 1024px equivalent
# Empirically measured: avg σ²_256 = 20.16, σ²_1024 = 46.85 → ratio = 2.32
VARIANCE_SCALE_256_TO_1024 = 2.32

# ISO clamp range
ISO_MIN = 12800
ISO_MAX = 25600


def parse_dzi(dzi_path):
    """Parse DZI manifest for tile size and image dimensions."""
    tree = ET.parse(dzi_path)
    root = tree.getroot()
    ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
    tile_size = int(root.get('TileSize'))
    size_el = root.find(f'{ns}Size')
    width = int(size_el.get('Width'))
    height = int(size_el.get('Height'))
    return tile_size, width, height


def sigma_rms_all(img_gray):
    """Estimate noise sigma via RMS of MAD across all wavelet detail subbands.

    Uses all 6 subbands (H/V/D at 2 decomposition levels) to capture the full
    spectrum of high-frequency energy.
    """
    coeffs = pywt.wavedec2(img_gray.astype(np.float64), 'db4', level=2)
    mads = []
    for cH, cV, cD in coeffs[1:]:
        mads.append(np.median(np.abs(cH)) / 0.6745)
        mads.append(np.median(np.abs(cV)) / 0.6745)
        mads.append(np.median(np.abs(cD)) / 0.6745)
    return np.sqrt(np.mean([m ** 2 for m in mads]))


def is_tissue_tile(img_gray, min_std=10, max_mean=220):
    """Check if a tile contains tissue (not blank background)."""
    return np.std(img_gray) > min_std and np.mean(img_gray) < max_mean


def interpolate_iso(target_var, table):
    """Interpolate calibration table to find ISO for a target 1024px variance.

    table is a list of (iso, var_1024) tuples sorted by iso.
    Returns ISO clamped to [ISO_MIN, ISO_MAX].
    """
    if not table:
        return ISO_MIN

    # Find bracketing entries
    for i in range(len(table) - 1):
        iso_lo, var_lo = table[i]
        iso_hi, var_hi = table[i + 1]
        if var_lo <= target_var <= var_hi:
            t = (target_var - var_lo) / (var_hi - var_lo) if var_hi != var_lo else 0
            iso = iso_lo + t * (iso_hi - iso_lo)
            return max(ISO_MIN, min(ISO_MAX, iso))

    # Out of range — clamp
    if target_var <= table[0][1]:
        return ISO_MIN
    return ISO_MAX


def main():
    parser = argparse.ArgumentParser(description='Estimate optimal JXL ISO for a slide')
    parser.add_argument('--pyramid', required=True, help='Path to DZI pyramid directory')
    parser.add_argument('--quality', type=int, default=40, help='JXL quality level')
    parser.add_argument('--samples', type=int, default=10, help='Number of tissue tiles to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    args = parser.parse_args()

    pyramid_dir = pathlib.Path(args.pyramid)
    dzi_path = pyramid_dir / 'baseline_pyramid.dzi'
    tile_size, img_w, img_h = parse_dzi(dzi_path)
    files_dir = pyramid_dir / 'baseline_pyramid_files'

    # Find L0 level
    levels = sorted([int(d.name) for d in files_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    l0_level = levels[-1]
    l0_dir = files_dir / str(l0_level)

    tile_files = sorted(l0_dir.glob('*.jpg')) + sorted(l0_dir.glob('*.jxl'))

    if not args.json:
        print(f"Pyramid: {pyramid_dir}")
        print(f"  Tile size: {tile_size}px, Image: {img_w}x{img_h}")
        print(f"  L0 level: {l0_level}, Tiles: {len(tile_files)}")
        print(f"  Quality: JXL Q{args.quality}")

    # Sample central tissue tiles
    random.seed(args.seed)
    tiles_x = (img_w + tile_size - 1) // tile_size
    tiles_y = (img_h + tile_size - 1) // tile_size
    cx, cy = tiles_x / 2, tiles_y / 2
    qx, qy = tiles_x / 4, tiles_y / 4

    def parse_coords(path):
        parts = path.stem.split('_')
        try:
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return None, None

    central = []
    peripheral = []
    for tf in tile_files:
        tx, ty = parse_coords(tf)
        if tx is None:
            peripheral.append(tf)
        elif abs(tx - cx) <= qx and abs(ty - cy) <= qy:
            central.append(tf)
        else:
            peripheral.append(tf)

    random.shuffle(central)
    random.shuffle(peripheral)
    candidates = central + peripheral

    tissue_sigmas = []
    skipped = 0
    for tf in candidates:
        if len(tissue_sigmas) >= args.samples:
            break
        img = np.array(Image.open(tf).convert('L')).astype(np.float64)
        if is_tissue_tile(img):
            tissue_sigmas.append(sigma_rms_all(img))
        else:
            skipped += 1

    if not tissue_sigmas:
        if args.json:
            print(json.dumps({"error": "no tissue tiles found", "recommended_iso": ISO_MIN}))
        else:
            print("ERROR: No tissue tiles found")
        return

    avg_var_256 = np.mean([s ** 2 for s in tissue_sigmas])
    avg_sigma_256 = np.mean(tissue_sigmas)

    # Scale to 1024px equivalent
    target_var_1024 = avg_var_256 * VARIANCE_SCALE_256_TO_1024

    # Get calibration table for this quality
    table = CALIBRATION.get(args.quality)
    if table is None:
        # Fall back to Q40 table with a rough quality adjustment
        # Higher quality removes less noise, so target variance is lower
        q40_table = CALIBRATION[40]
        quality_factor = max(0.1, (70 - args.quality) / (70 - 40))  # 1.0 at Q40, 0 at Q70
        target_var_1024 *= quality_factor
        table = q40_table
        if not args.json:
            print(f"  (No calibration for Q{args.quality}, using Q40 table with {quality_factor:.2f}x factor)")

    recommended_iso = interpolate_iso(target_var_1024, table)

    if not args.json:
        print(f"\n  Sampled {len(tissue_sigmas)} tissue tiles (skipped {skipped} background)")
        print(f"  256px avg σ = {avg_sigma_256:.4f}  σ² = {avg_var_256:.4f}")
        print(f"  1024px target σ² = {target_var_1024:.4f}  (×{VARIANCE_SCALE_256_TO_1024})")
        print(f"\n  === Recommended ISO: {recommended_iso:.0f} ===")
        print(f"  (clamped to [{ISO_MIN}, {ISO_MAX}])")

    result = {
        "pyramid": str(pyramid_dir),
        "quality": args.quality,
        "tiles_sampled": len(tissue_sigmas),
        "avg_sigma_256": round(avg_sigma_256, 4),
        "avg_variance_256": round(avg_var_256, 4),
        "target_variance_1024": round(target_var_1024, 4),
        "recommended_iso": round(recommended_iso),
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"\n{json.dumps(result, indent=2)}")


if __name__ == '__main__':
    main()
