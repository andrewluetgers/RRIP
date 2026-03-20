#!/usr/bin/env python3
"""
Prepare DZI pyramids for the tile server with JXL-encoded L0 at 1024px tiles.

Creates two variants from a trimmed DZI source:
  - jxl_q80: JXL quality 80, no noise synthesis
  - jxl_q40: JXL quality 40, with auto-determined photon_noise_iso

L0 tiles are retiled from 256px→1024px and encoded as JXL.
L1 and L2 are empty directories (reconstructed by the tile server).
L3+ levels are symlinked from the source DZI.

Usage:
    uv run python scripts/prep_tile_server.py \
        --src ~/dev/data/WSI/dzi_trimmed \
        --out ~/dev/data/WSI/tile_server

    # Single slide
    uv run python scripts/prep_tile_server.py \
        --src ~/dev/data/WSI/dzi_trimmed \
        --out ~/dev/data/WSI/tile_server \
        --slide 3DHISTECH-1
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

# Import noise estimation from evals
_SCRIPT_DIR = Path(__file__).resolve().parent
_EVALS_SCRIPTS = _SCRIPT_DIR.parent / "evals" / "scripts"
sys.path.insert(0, str(_EVALS_SCRIPTS))
from estimate_slide_iso import (
    sigma_rms_all, is_tissue_tile, interpolate_iso,
    CALIBRATION, VARIANCE_SCALE_256_TO_1024, ISO_MIN, ISO_MAX,
)


def parse_dzi(dzi_path: str) -> tuple[int, int, int]:
    """Parse DZI manifest for tile size, width, height."""
    tree = ET.parse(dzi_path)
    root = tree.getroot()
    ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
    tile_size = int(root.get('TileSize'))
    size_el = root.find(f'{ns}Size')
    width = int(size_el.get('Width'))
    height = int(size_el.get('Height'))
    return tile_size, width, height


def estimate_iso_from_tiles(l0_dir: str, tiles_x: int, tiles_y: int,
                            quality: int = 40, n_samples: int = 20) -> int:
    """Estimate optimal JXL photon_noise_iso by sampling L0 tissue tiles.

    Samples 256px tiles, measures noise via wavelet MAD, scales to 1024px
    equivalent, and interpolates the calibration table.
    """
    import random
    random.seed(42)

    # Collect tissue tile paths from the center region
    cx, cy = tiles_x / 2, tiles_y / 2
    qx, qy = tiles_x / 4, tiles_y / 4

    central = []
    peripheral = []
    for f in os.listdir(l0_dir):
        if not f.endswith('.jpeg'):
            continue
        tx, ty = f.replace('.jpeg', '').split('_')
        tx, ty = int(tx), int(ty)
        if abs(tx - cx) <= qx and abs(ty - cy) <= qy:
            central.append(os.path.join(l0_dir, f))
        else:
            peripheral.append(os.path.join(l0_dir, f))

    random.shuffle(central)
    random.shuffle(peripheral)
    candidates = central + peripheral

    tissue_sigmas = []
    for path in candidates:
        if len(tissue_sigmas) >= n_samples:
            break
        img = np.array(Image.open(path).convert('L')).astype(np.float64)
        if is_tissue_tile(img):
            tissue_sigmas.append(sigma_rms_all(img))

    if not tissue_sigmas:
        return ISO_MIN

    avg_var_256 = np.mean([s ** 2 for s in tissue_sigmas])
    target_var_1024 = avg_var_256 * VARIANCE_SCALE_256_TO_1024

    table = CALIBRATION.get(quality)
    if table is None:
        q40_table = CALIBRATION[40]
        quality_factor = max(0.1, (70 - quality) / (70 - 40))
        target_var_1024 *= quality_factor
        table = q40_table

    return round(interpolate_iso(target_var_1024, table))


def retile_and_encode_jxl(src_l0_dir: str, dst_l0_dir: str,
                          src_tile_size: int, dst_tile_size: int,
                          tiles_x: int, tiles_y: int,
                          quality: int, photon_noise_iso: int = 0,
                          workers: int = 4,
                          blank_color_grid: dict = None):
    """Retile 256px→1024px and encode as JXL.

    Composites 4x4 source tiles into one 1024px tile, encodes to JXL.
    Missing child tiles are filled with the blank color from the color grid
    instead of white, so that L1/L2 downsamples at tissue margins match
    the slide background.
    """
    ratio = dst_tile_size // src_tile_size  # 4 for 256→1024
    dst_tiles_x = math.ceil(tiles_x / ratio)
    dst_tiles_y = math.ceil(tiles_y / ratio)

    # Pre-parse blank color grid for fast lookup
    bcg = blank_color_grid
    bcg_cols = bcg['cols'] if bcg else 10
    bcg_rows = bcg['rows'] if bcg else 10
    bcg_rgb = bcg['rgb'] if bcg else None

    def get_blank_color(tx, ty):
        """Get interpolated blank color for an L0 tile position."""
        if not bcg_rgb:
            return (255, 255, 255)
        gx = tx / max(tiles_x, 1) * (bcg_cols - 1)
        gy = ty / max(tiles_y, 1) * (bcg_rows - 1)
        gx0 = min(int(gx), bcg_cols - 2)
        gy0 = min(int(gy), bcg_rows - 2)
        fx = gx - gx0
        fy = gy - gy0
        c00 = bcg_rgb[gy0 * bcg_cols + gx0]
        c10 = bcg_rgb[gy0 * bcg_cols + gx0 + 1]
        c01 = bcg_rgb[(gy0 + 1) * bcg_cols + gx0]
        c11 = bcg_rgb[(gy0 + 1) * bcg_cols + gx0 + 1]
        r = int(c00[0]*(1-fx)*(1-fy) + c10[0]*fx*(1-fy) + c01[0]*(1-fx)*fy + c11[0]*fx*fy + 0.5)
        g = int(c00[1]*(1-fx)*(1-fy) + c10[1]*fx*(1-fy) + c01[1]*(1-fx)*fy + c11[1]*fx*fy + 0.5)
        b = int(c00[2]*(1-fx)*(1-fy) + c10[2]*fx*(1-fy) + c01[2]*(1-fx)*fy + c11[2]*fx*fy + 0.5)
        return (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))

    os.makedirs(dst_l0_dir, exist_ok=True)

    def encode_tile(args):
        dtx, dty = args
        # Composite ratio x ratio source tiles into one dst tile.
        # Use the blank color for the center of this 1024px tile region.
        center_tx = dtx * ratio + ratio // 2
        center_ty = dty * ratio + ratio // 2
        bg = get_blank_color(center_tx, center_ty)
        mosaic = Image.new('RGB', (dst_tile_size, dst_tile_size), bg)
        has_any = False
        for dy in range(ratio):
            for dx in range(ratio):
                stx = dtx * ratio + dx
                sty = dty * ratio + dy
                src_path = os.path.join(src_l0_dir, f'{stx}_{sty}.jpeg')
                if os.path.exists(src_path):
                    tile = Image.open(src_path)
                    mosaic.paste(tile, (dx * src_tile_size, dy * src_tile_size))
                    has_any = True

        if not has_any:
            return None

        # Encode to JXL via cjxl
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name

        dst_path = os.path.join(dst_l0_dir, f'{dtx}_{dty}.jxl')
        try:
            mosaic.save(tmp_path, format='PNG')
            cmd = ['cjxl', tmp_path, dst_path, '-q', str(quality),
                   '--effort', '7']
            if photon_noise_iso > 0:
                cmd.extend([f'--photon_noise_iso={photon_noise_iso}'])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'    cjxl error on {dtx}_{dty}: {result.stderr.strip()}')
                return None
            return dst_path
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    jobs = [(dtx, dty) for dty in range(dst_tiles_y)
            for dtx in range(dst_tiles_x)]

    encoded = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(encode_tile, jobs):
            if result:
                encoded += 1
                total_bytes += os.path.getsize(result)

    return encoded, total_bytes


def prep_slide(src_dir: str, slide_name: str, out_dir: str, workers: int = 4,
               noise_scale: float = 0.5):
    """Prepare one slide for tile server in both Q80 and Q40 variants."""
    src_dzi = os.path.join(src_dir, f'{slide_name}.dzi')
    src_files = os.path.join(src_dir, f'{slide_name}_files')

    if not os.path.exists(src_dzi):
        print(f'  SKIP: no .dzi found')
        return

    tile_size, img_w, img_h = parse_dzi(src_dzi)
    levels = sorted(int(d) for d in os.listdir(src_files) if d.isdigit())
    max_level = levels[-1]
    tiles_x = math.ceil(img_w / tile_size)
    tiles_y = math.ceil(img_h / tile_size)

    src_l0_dir = os.path.join(src_files, str(max_level))
    dst_tile_size = 1024

    # Load blank color grid from tissue.json (for filling missing tile regions)
    import json as _json
    blank_color_grid = None
    # Check DZI source dir (dzi/) for the tissue.json
    dzi_dir = os.path.join(os.path.dirname(src_dir), 'dzi')
    tissue_json = os.path.join(dzi_dir, f'{slide_name}.tissue.json')
    if not os.path.exists(tissue_json):
        tissue_json = os.path.join(src_dir, f'{slide_name}.tissue.json')
    if os.path.exists(tissue_json):
        with open(tissue_json) as f:
            meta = _json.load(f)
        blank_color_grid = meta.get('blank_color_grid')
        if blank_color_grid:
            print(f'  Blank color grid: {blank_color_grid["cols"]}x{blank_color_grid["rows"]}')

    # Estimate ISO for Q40 noise synthesis
    print(f'  Estimating noise ISO...')
    t0 = time.time()
    iso_raw = estimate_iso_from_tiles(src_l0_dir, tiles_x, tiles_y, quality=40)
    iso = round(iso_raw * noise_scale)
    print(f'  ISO={iso} (raw={iso_raw}, scale={noise_scale}, {time.time() - t0:.1f}s)')

    # Write updated DZI manifest with 1024px tile size
    dzi_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jxl"
       Overlap="0"
       TileSize="{dst_tile_size}">
  <Size Width="{img_w}" Height="{img_h}"/>
</Image>
"""

    all_variants = [('jxl_q80', 80, 0), ('jxl_q40', 40, iso), ('jxl_q40_nonoise', 40, 0)]
    # Skip variants whose L0 dir already has JXL tiles
    for variant, quality, noise_iso in all_variants:
        var_l0 = os.path.join(out_dir, variant, slide_name, f'{slide_name}_files', str(max_level))
        if os.path.isdir(var_l0) and any(f.endswith('.jxl') for f in os.listdir(var_l0)):
            print(f'\n  --- {variant} (Q{quality}) --- SKIP (already exists)')
            continue
        print(f'\n  --- {variant} (Q{quality}'
              f'{f", ISO={noise_iso}" if noise_iso else ""}) ---')

        var_dir = os.path.join(out_dir, variant, slide_name)
        var_files = os.path.join(var_dir, f'{slide_name}_files')
        os.makedirs(var_dir, exist_ok=True)

        # Write DZI manifest
        with open(os.path.join(var_dir, f'{slide_name}.dzi'), 'w') as f:
            f.write(dzi_xml)

        # Symlink bitmaps
        for suffix in ['.tissue.bitmap', '.tissue_margin.bitmap']:
            src_bm = os.path.join(src_dir, f'{slide_name}{suffix}')
            if os.path.exists(src_bm):
                dst_bm = os.path.join(var_dir, f'{slide_name}{suffix}')
                if not os.path.exists(dst_bm):
                    os.symlink(os.path.abspath(src_bm), dst_bm)

        # L3+ levels: symlink from source (only numeric level dirs)
        os.makedirs(var_files, exist_ok=True)
        for level in levels:
            level_diff = max_level - level
            dst_level = os.path.join(var_files, str(level))
            if level_diff >= 3:
                src_level = os.path.abspath(os.path.join(src_files, str(level)))
                if os.path.isdir(src_level) and not os.path.exists(dst_level):
                    os.symlink(src_level, dst_level)

        # L1 and L2: empty directories (tile server reconstructs these)
        for offset in [1, 2]:
            level = max_level - offset
            dst_level = os.path.join(var_files, str(level))
            os.makedirs(dst_level, exist_ok=True)

        # L0: retile 256→1024 and encode to JXL
        dst_l0 = os.path.join(var_files, str(max_level))
        t1 = time.time()
        encoded, total_bytes = retile_and_encode_jxl(
            src_l0_dir, dst_l0, tile_size, dst_tile_size,
            tiles_x, tiles_y, quality=quality,
            photon_noise_iso=noise_iso, workers=workers,
            blank_color_grid=blank_color_grid)
        dt = time.time() - t1
        print(f'  L0: {encoded} JXL tiles, {total_bytes / 1e6:.1f} MB, '
              f'{dt:.1f}s ({encoded / max(dt, 0.1):.0f} tiles/s)')


def main():
    ap = argparse.ArgumentParser(
        description='Prepare DZI pyramids for tile server with JXL L0')
    ap.add_argument('--src', required=True,
                    help='Source trimmed DZI directory')
    ap.add_argument('--out', required=True,
                    help='Output directory for tile server variants')
    ap.add_argument('--slide', default=None,
                    help='Process only this slide')
    ap.add_argument('--workers', type=int, default=4,
                    help='Parallel cjxl workers (default: 4)')
    ap.add_argument('--noise-scale', type=float, default=0.0,
                    help='Noise synthesis scale: 0=off, 0.5=half, 1=full auto (default: 0.0)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.slide:
        slides = [args.slide]
    else:
        slides = sorted(
            f.replace('.dzi', '')
            for f in os.listdir(args.src)
            if f.endswith('.dzi')
        )

    print(f'Preparing {len(slides)} slide(s) for tile server')
    print(f'Source: {args.src}')
    print(f'Output: {args.out}')
    print(f'Variants: jxl_q80 (no noise), jxl_q40 (noise scale={args.noise_scale})\n')

    for slide in slides:
        print(f'{slide[:40]}...')
        t0 = time.time()
        prep_slide(args.src, slide, args.out, workers=args.workers,
                   noise_scale=args.noise_scale)
        dt = time.time() - t0
        print(f'  Done in {dt:.1f}s\n')

    # Summary
    for variant in ['jxl_q80', 'jxl_q40']:
        var_dir = os.path.join(args.out, variant)
        if os.path.isdir(var_dir):
            total = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(var_dir, followlinks=False)
                for f in fns if f.endswith('.jxl')
            )
            print(f'{variant}: {total / 1e6:.0f} MB JXL data')


if __name__ == '__main__':
    main()
