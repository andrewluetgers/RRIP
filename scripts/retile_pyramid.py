#!/usr/bin/env python3
"""
Retile a 1024px DZI pyramid into a 256px DZI pyramid.

Reads tiles from a source pyramid with 1024px tiles, splits them into 256px
tiles, and builds a complete DZI pyramid with all levels down to level 0.

Usage:
    python retile_pyramid.py \
        --src data/dzi/dicom_extract_1024/baseline_pyramid_files \
        --out data/dzi/jpeg90/baseline_pyramid_files \
        --quality 90

The script also writes a baseline_pyramid.dzi manifest in the parent of --out.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow large images


def parse_args():
    p = argparse.ArgumentParser(description="Retile 1024px DZI pyramid to 256px")
    p.add_argument("--src", required=True, help="Source baseline_pyramid_files/ with 1024px tiles")
    p.add_argument("--out", required=True, help="Output baseline_pyramid_files/ for 256px tiles")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality (default: 90)")
    p.add_argument("--src-tile", type=int, default=1024, help="Source tile size (default: 1024)")
    p.add_argument("--dst-tile", type=int, default=256, help="Destination tile size (default: 256)")
    p.add_argument("--width", type=int, default=None, help="Image width (auto-detected if omitted)")
    p.add_argument("--height", type=int, default=None, help="Image height (auto-detected if omitted)")
    return p.parse_args()


def discover_max_level(src_dir):
    """Find the maximum level number in the source pyramid."""
    max_lvl = -1
    for name in os.listdir(src_dir):
        try:
            lvl = int(name)
            if lvl > max_lvl:
                max_lvl = lvl
        except ValueError:
            pass
    return max_lvl


def discover_grid(src_dir, level):
    """Return (max_col, max_row) for a given level directory."""
    level_dir = os.path.join(src_dir, str(level))
    max_x = max_y = -1
    for name in os.listdir(level_dir):
        if name.endswith(".jpg"):
            parts = name.replace(".jpg", "").split("_")
            if len(parts) == 2:
                max_x = max(max_x, int(parts[0]))
                max_y = max(max_y, int(parts[1]))
    return max_x + 1, max_y + 1


def split_level(src_dir, out_dir, src_level, src_tile, dst_tile, quality):
    """Split source tiles at src_level into smaller tiles.

    Each 1024px tile becomes a 4x4 grid of 256px tiles.
    Returns (new_cols, new_rows) at the new level.
    """
    ratio = src_tile // dst_tile  # 4 for 1024->256
    src_cols, src_rows = discover_grid(src_dir, src_level)
    new_cols = src_cols * ratio
    new_rows = src_rows * ratio

    # The new level number is the same as source since we're re-tiling
    level_dir = os.path.join(out_dir, str(src_level))
    os.makedirs(level_dir, exist_ok=True)

    total = src_cols * src_rows
    done = 0
    for sy in range(src_rows):
        for sx in range(src_cols):
            src_path = os.path.join(src_dir, str(src_level), f"{sx}_{sy}.jpg")
            if not os.path.exists(src_path):
                done += 1
                continue

            img = Image.open(src_path)
            w, h = img.size

            for dy in range(ratio):
                for dx in range(ratio):
                    x0 = dx * dst_tile
                    y0 = dy * dst_tile
                    x1 = min(x0 + dst_tile, w)
                    y1 = min(y0 + dst_tile, h)

                    if x0 >= w or y0 >= h:
                        continue

                    tile = img.crop((x0, y0, x1, y1))
                    new_x = sx * ratio + dx
                    new_y = sy * ratio + dy
                    out_path = os.path.join(level_dir, f"{new_x}_{new_y}.jpg")
                    tile.save(out_path, "JPEG", quality=quality)

            done += 1
            if done % 100 == 0 or done == total:
                print(f"  Split L{src_level}: {done}/{total} source tiles", flush=True)

    return new_cols, new_rows


def build_lower_levels(out_dir, top_level, top_cols, top_rows, dst_tile, quality):
    """Build pyramid levels from top_level-1 down to 0 by downsampling."""
    prev_level = top_level
    prev_cols = top_cols
    prev_rows = top_rows

    for level in range(top_level - 1, -1, -1):
        cur_cols = (prev_cols + 1) // 2
        cur_rows = (prev_rows + 1) // 2
        level_dir = os.path.join(out_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)

        for ty in range(cur_rows):
            for tx in range(cur_cols):
                # Build 2x2 mosaic from parent tiles
                mosaic = Image.new("RGB", (dst_tile * 2, dst_tile * 2), (255, 255, 255))
                for dy in range(2):
                    for dx in range(2):
                        px = tx * 2 + dx
                        py = ty * 2 + dy
                        if px >= prev_cols or py >= prev_rows:
                            continue
                        parent_path = os.path.join(out_dir, str(prev_level), f"{px}_{py}.jpg")
                        if not os.path.exists(parent_path):
                            continue
                        parent = Image.open(parent_path)
                        mosaic.paste(parent, (dx * dst_tile, dy * dst_tile))

                # Downsample to tile_size
                tile = mosaic.resize((dst_tile, dst_tile), Image.LANCZOS)
                out_path = os.path.join(level_dir, f"{tx}_{ty}.jpg")
                tile.save(out_path, "JPEG", quality=quality)

        print(f"  Level {level}: {cur_cols}x{cur_rows} = {cur_cols * cur_rows} tiles", flush=True)
        prev_level = level
        prev_cols = cur_cols
        prev_rows = cur_rows


def write_dzi(out_dir, width, height, tile_size):
    """Write a DZI manifest file in the parent directory of out_dir."""
    parent = os.path.dirname(out_dir)
    dzi_path = os.path.join(parent, "baseline_pyramid.dzi")
    dzi = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
  Format="jpg"
  Overlap="0"
  TileSize="{tile_size}"
  >
  <Size
    Height="{height}"
    Width="{width}"
  />
</Image>
"""
    with open(dzi_path, "w") as f:
        f.write(dzi)
    print(f"Wrote DZI manifest: {dzi_path}")


def main():
    args = parse_args()
    src_dir = args.src
    out_dir = args.out
    quality = args.quality
    src_tile = args.src_tile
    dst_tile = args.dst_tile

    max_level = discover_max_level(src_dir)
    print(f"Source pyramid: max_level={max_level}, src_tile={src_tile}, dst_tile={dst_tile}")

    src_cols, src_rows = discover_grid(src_dir, max_level)
    print(f"Source L{max_level}: {src_cols}x{src_rows} tiles")

    # Auto-detect dimensions from source grid
    width = args.width or src_cols * src_tile
    height = args.height or src_rows * src_tile
    print(f"Image dimensions: {width}x{height}")

    # Step 1: Split top level
    print(f"\nSplitting L{max_level} ({src_tile}px -> {dst_tile}px)...")
    new_cols, new_rows = split_level(src_dir, out_dir, max_level, src_tile, dst_tile, quality)
    print(f"New L{max_level}: {new_cols}x{new_rows} = {new_cols * new_rows} tiles")

    # Step 2: For levels below max, we also need to re-split them
    # But it's simpler and higher quality to build them by downsampling from the new top level
    print(f"\nBuilding lower levels from L{max_level} down to L0...")
    build_lower_levels(out_dir, max_level, new_cols, new_rows, dst_tile, quality)

    # Step 3: Write DZI manifest
    write_dzi(out_dir, width, height, dst_tile)

    # Count total tiles
    total = 0
    for level in range(max_level + 1):
        level_dir = os.path.join(out_dir, str(level))
        if os.path.isdir(level_dir):
            n = len([f for f in os.listdir(level_dir) if f.endswith(".jpg")])
            total += n
    print(f"\nDone! {total} tiles across {max_level + 1} levels")


if __name__ == "__main__":
    main()