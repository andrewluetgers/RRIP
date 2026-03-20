#!/usr/bin/env python3
"""
Recompress a DZI pyramid's JPEG tiles at a specified quality.

Creates a full static DZI copy where every tile is decoded and re-encoded
at the target JPEG quality. No retiling, no JXL — just quality reduction.
Useful as a baseline comparison against the ORIGAMI JXL pipeline.

Usage:
    uv run python scripts/recompress_jpeg.py \
        --src ~/dev/data/WSI/dzi_trimmed \
        --out ~/dev/data/WSI/tile_server \
        --quality 80 \
        --slide 3DHISTECH-1

    # Multiple qualities
    uv run python scripts/recompress_jpeg.py \
        --src ~/dev/data/WSI/dzi_trimmed \
        --out ~/dev/data/WSI/tile_server \
        --quality 80 40
"""

import argparse
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def recompress_slide(src_dir: str, slide_name: str, out_dir: str,
                     quality: int, workers: int = 8):
    """Recompress all JPEG tiles in a slide's DZI at the given quality."""
    src_dzi = os.path.join(src_dir, f'{slide_name}.dzi')
    src_files = os.path.join(src_dir, f'{slide_name}_files')

    if not os.path.exists(src_dzi):
        print(f'  SKIP: no .dzi found')
        return

    variant = f'jpeg_q{quality}'
    dst_dir = os.path.join(out_dir, variant, slide_name)
    dst_files = os.path.join(dst_dir, f'{slide_name}_files')

    # Check if already done
    if os.path.isdir(dst_files):
        levels = [d for d in os.listdir(dst_files) if d.isdigit()]
        if levels:
            sample_level = os.path.join(dst_files, levels[0])
            if any(f.endswith('.jpeg') for f in os.listdir(sample_level)):
                print(f'  SKIP {variant} (already exists)')
                return

    os.makedirs(dst_dir, exist_ok=True)

    # Copy DZI manifest (update format to jpeg if needed)
    shutil.copy2(src_dzi, os.path.join(dst_dir, f'{slide_name}.dzi'))

    # Symlink tissue assets from dzi dir
    dzi_dir = os.path.join(os.path.dirname(src_dir), 'dzi')
    for ext in ['tissue.map', 'tissue.json', 'tissue.svg']:
        src_asset = os.path.join(dzi_dir, f'{slide_name}.{ext}')
        dst_asset = os.path.join(dst_dir, f'{slide_name}.{ext}')
        if os.path.exists(src_asset) and not os.path.exists(dst_asset):
            os.symlink(os.path.abspath(src_asset), dst_asset)

    levels = sorted(int(d) for d in os.listdir(src_files) if d.isdigit())

    total_src_bytes = 0
    total_dst_bytes = 0
    total_tiles = 0

    for level in levels:
        src_level = os.path.join(src_files, str(level))
        dst_level = os.path.join(dst_files, str(level))
        os.makedirs(dst_level, exist_ok=True)

        tiles = [f for f in os.listdir(src_level)
                 if f.endswith('.jpeg') or f.endswith('.jpg')]

        def recompress_tile(fname):
            src_path = os.path.join(src_level, fname)
            dst_path = os.path.join(dst_level, fname)
            src_size = os.path.getsize(src_path)

            img = Image.open(src_path)
            img.save(dst_path, 'JPEG', quality=quality)
            dst_size = os.path.getsize(dst_path)

            return src_size, dst_size

        level_src = 0
        level_dst = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for src_size, dst_size in pool.map(recompress_tile, tiles):
                level_src += src_size
                level_dst += dst_size

        total_src_bytes += level_src
        total_dst_bytes += level_dst
        total_tiles += len(tiles)

        ratio = level_src / level_dst if level_dst > 0 else 0
        print(f'  Level {level}: {len(tiles)} tiles, '
              f'{level_src/1e6:.1f}→{level_dst/1e6:.1f} MB ({ratio:.1f}x)')

    ratio = total_src_bytes / total_dst_bytes if total_dst_bytes > 0 else 0
    print(f'  Total: {total_tiles} tiles, '
          f'{total_src_bytes/1e6:.0f}→{total_dst_bytes/1e6:.0f} MB ({ratio:.1f}x)')

    # Write summary.json
    import json
    summary = {
        'label': f'{slide_name} JPEG Q{quality}' if len(slide_name) < 20
                 else f'JPEG Q{quality}',
        'mode': 'ingest-jpeg-only',
        'quality': quality,
        'total_bytes': total_dst_bytes,
        'pipeline_version': 2,
    }
    with open(os.path.join(dst_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description='Recompress DZI JPEG tiles at specified quality')
    ap.add_argument('--src', required=True,
                    help='Source DZI directory (e.g. dzi_trimmed)')
    ap.add_argument('--out', required=True,
                    help='Output directory')
    ap.add_argument('--quality', type=int, nargs='+', default=[80, 40],
                    help='JPEG quality levels (default: 80 40)')
    ap.add_argument('--slide', default=None,
                    help='Process only this slide')
    ap.add_argument('--workers', type=int, default=8,
                    help='Thread pool workers (default: 8)')
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

    for quality in args.quality:
        print(f'\n=== JPEG Q{quality} ===\n')
        for slide in slides:
            print(f'{slide[:40]}...')
            t0 = time.time()
            recompress_slide(args.src, slide, args.out, quality,
                             workers=args.workers)
            print(f'  Done in {time.time() - t0:.1f}s\n')


if __name__ == '__main__':
    main()
