#!/usr/bin/env python3
"""
Create a trimmed DZI pyramid containing only tissue tiles for L0-L2.

For L0, L1, L2: only copies tiles that fall within the tissue bounding
polygons (loaded from .tissue.bitmap). All other levels are copied in full
(they're tiny and needed for the viewer).

This dramatically reduces storage — typically 30-80% of L0 tiles are blank.

Usage:
    uv run python scripts/trim_dzi.py \
        --dzi-dir ~/dev/data/WSI/dzi \
        --out ~/dev/data/WSI/dzi_trimmed

    # Single slide
    uv run python scripts/trim_dzi.py \
        --dzi-dir ~/dev/data/WSI/dzi \
        --out ~/dev/data/WSI/dzi_trimmed \
        --slide 3DHISTECH-1
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from tissue_filter import TissueFilter


def trim_slide(dzi_dir: str, slide_name: str, out_dir: str):
    """Create a trimmed copy of one slide's DZI pyramid."""
    src_files = os.path.join(dzi_dir, f'{slide_name}_files')
    src_dzi = os.path.join(dzi_dir, f'{slide_name}.dzi')
    # Use the margin bitmap for tile inclusion (safe boundary)
    margin_bitmap = os.path.join(dzi_dir, f'{slide_name}.tissue_margin.bitmap')
    tissue_bitmap = os.path.join(dzi_dir, f'{slide_name}.tissue.bitmap')

    if os.path.exists(margin_bitmap):
        filt = TissueFilter.load_bitmap(margin_bitmap)
    elif os.path.exists(tissue_bitmap):
        filt = TissueFilter.load_bitmap(tissue_bitmap)
    else:
        print(f'  SKIP: no .bitmap found')
        return

    max_level = filt._max_level

    # Copy the .dzi manifest and both bitmaps
    dst_dzi = os.path.join(out_dir, f'{slide_name}.dzi')
    shutil.copy2(src_dzi, dst_dzi)
    for bm in [tissue_bitmap, margin_bitmap]:
        if os.path.exists(bm):
            shutil.copy2(bm, os.path.join(out_dir, os.path.basename(bm)))

    dst_files = os.path.join(out_dir, f'{slide_name}_files')
    levels = sorted(int(d) for d in os.listdir(src_files) if d.isdigit())

    total_copied = 0
    total_skipped = 0
    total_bytes_copied = 0
    total_bytes_skipped = 0

    for level in levels:
        src_level = os.path.join(src_files, str(level))
        dst_level = os.path.join(dst_files, str(level))

        # L3 and above (level <= max_level - 3): copy everything
        # L0, L1, L2 (level >= max_level - 2): filter by tissue
        filter_this_level = level >= max_level - 2

        if not filter_this_level:
            # Copy entire level directory (don't follow symlinks to avoid loops)
            shutil.copytree(src_level, dst_level, symlinks=True)
            count = len(os.listdir(dst_level))
            total_copied += count
            continue

        os.makedirs(dst_level, exist_ok=True)
        level_copied = 0
        level_skipped = 0
        level_bytes_copied = 0
        level_bytes_skipped = 0

        for f in os.listdir(src_level):
            if not f.endswith('.jpeg'):
                continue
            tx, ty = f.replace('.jpeg', '').split('_')
            tx, ty = int(tx), int(ty)

            src_path = os.path.join(src_level, f)
            fsize = os.path.getsize(src_path)

            if filt.includes(level, tx, ty):
                dst_path = os.path.join(dst_level, f)
                # Hard link instead of copy for speed and disk savings
                # Falls back to copy if cross-device
                try:
                    os.link(src_path, dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)
                level_copied += 1
                level_bytes_copied += fsize
            else:
                level_skipped += 1
                level_bytes_skipped += fsize

        level_name = f'L{max_level - level}' if level >= max_level - 2 else f'L{max_level - level}+'
        print(f'  Level {level} ({level_name}): {level_copied} copied, '
              f'{level_skipped} skipped '
              f'({level_bytes_skipped / 1e6:.1f} MB saved)')

        total_copied += level_copied
        total_skipped += level_skipped
        total_bytes_copied += level_bytes_copied
        total_bytes_skipped += level_bytes_skipped

    print(f'  Total: {total_copied} tiles copied, {total_skipped} skipped, '
          f'{total_bytes_skipped / 1e6:.1f} MB saved')


def main():
    ap = argparse.ArgumentParser(
        description='Create trimmed DZI pyramids with only tissue tiles')
    ap.add_argument('--dzi-dir', required=True,
                    help='Source DZI directory')
    ap.add_argument('--out', required=True,
                    help='Output directory for trimmed pyramids')
    ap.add_argument('--slide', default=None,
                    help='Process only this slide')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.slide:
        slides = [args.slide]
    else:
        slides = sorted(
            f.replace('.dzi', '')
            for f in os.listdir(args.dzi_dir)
            if f.endswith('.dzi')
        )

    print(f'Trimming {len(slides)} slide(s)')
    print(f'Source: {args.dzi_dir}')
    print(f'Output: {args.out}\n')

    for slide in slides:
        print(f'{slide[:40]}...')
        t0 = time.time()
        trim_slide(args.dzi_dir, slide, args.out)
        dt = time.time() - t0
        print(f'  Done in {dt:.1f}s\n')

    # Summary
    src_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(args.dzi_dir)
        for f in fns if f.endswith('.jpeg')
    )
    dst_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(args.out)
        for f in fns if f.endswith('.jpeg')
    )
    print(f'Source total: {src_size / 1e9:.2f} GB')
    print(f'Trimmed total: {dst_size / 1e9:.2f} GB')
    print(f'Saved: {(src_size - dst_size) / 1e9:.2f} GB '
          f'({100 * (src_size - dst_size) / src_size:.1f}%)')


if __name__ == '__main__':
    main()
