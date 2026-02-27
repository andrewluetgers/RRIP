#!/usr/bin/env python3
"""
Retile a 1024px DZI pyramid into a 256px DZI pyramid using pyvips + Lanczos3.

Reads tiles from a source pyramid with 1024px tiles, splits the top level
into 256px tiles (crop only, no resampling needed), then builds all lower
levels by downsampling 2x with Lanczos3 at each step.

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python scripts/retile_pyvips.py \
        --src data/dzi/dicom_extract_1024/baseline_pyramid_files \
        --out data/dzi/jpeg90/baseline_pyramid_files \
        --quality 90
"""

import argparse
import os
import sys
import time

import pyvips


def parse_args():
    p = argparse.ArgumentParser(description="Retile 1024px DZI pyramid to 256px using pyvips Lanczos3")
    p.add_argument("--src", required=True, help="Source baseline_pyramid_files/ with 1024px tiles")
    p.add_argument("--out", required=True, help="Output baseline_pyramid_files/ for 256px tiles")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality (default: 90)")
    p.add_argument("--src-tile", type=int, default=1024, help="Source tile size (default: 1024)")
    p.add_argument("--dst-tile", type=int, default=256, help="Destination tile size (default: 256)")
    p.add_argument("--width", type=int, default=None, help="Image width (auto-detected if omitted)")
    p.add_argument("--height", type=int, default=None, help="Image height (auto-detected if omitted)")
    return p.parse_args()


def discover_max_level(src_dir):
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
    level_dir = os.path.join(src_dir, str(level))
    max_x = max_y = -1
    for name in os.listdir(level_dir):
        if name.endswith(".jpg"):
            parts = name.replace(".jpg", "").split("_")
            if len(parts) == 2:
                max_x = max(max_x, int(parts[0]))
                max_y = max(max_y, int(parts[1]))
    return max_x + 1, max_y + 1


def split_top_level(src_dir, out_dir, src_level, src_tile, dst_tile, quality):
    """Split source 1024px tiles into 256px tiles (crop, no resampling)."""
    ratio = src_tile // dst_tile
    src_cols, src_rows = discover_grid(src_dir, src_level)
    new_cols = src_cols * ratio
    new_rows = src_rows * ratio

    level_dir = os.path.join(out_dir, str(src_level))
    os.makedirs(level_dir, exist_ok=True)

    total = src_cols * src_rows
    done = 0
    t0 = time.time()

    for sy in range(src_rows):
        for sx in range(src_cols):
            src_path = os.path.join(src_dir, str(src_level), f"{sx}_{sy}.jpg")
            if not os.path.exists(src_path):
                done += 1
                continue

            img = pyvips.Image.new_from_file(src_path, access="sequential")
            w, h = img.width, img.height

            for dy in range(ratio):
                for dx in range(ratio):
                    x0 = dx * dst_tile
                    y0 = dy * dst_tile
                    if x0 >= w or y0 >= h:
                        continue

                    crop_w = min(dst_tile, w - x0)
                    crop_h = min(dst_tile, h - y0)
                    tile = img.crop(x0, y0, crop_w, crop_h)

                    new_x = sx * ratio + dx
                    new_y = sy * ratio + dy
                    out_path = os.path.join(level_dir, f"{new_x}_{new_y}.jpg")
                    tile.jpegsave(out_path, Q=quality, optimize_coding=True, strip=True)

            done += 1
            if done % 200 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  Split L{src_level}: {done}/{total} ({rate:.0f} tiles/s)", flush=True)

    return new_cols, new_rows


def build_lower_levels(out_dir, top_level, top_cols, top_rows, dst_tile, quality):
    """Build pyramid levels from top_level-1 down to 0 by 2x Lanczos3 downsampling."""
    prev_level = top_level
    prev_cols = top_cols
    prev_rows = top_rows

    for level in range(top_level - 1, -1, -1):
        cur_cols = (prev_cols + 1) // 2
        cur_rows = (prev_rows + 1) // 2
        level_dir = os.path.join(out_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)

        t0 = time.time()
        total = cur_cols * cur_rows
        done = 0

        for ty in range(cur_rows):
            for tx in range(cur_cols):
                # Build 2x2 mosaic from parent tiles
                parts = []
                for dy in range(2):
                    row_parts = []
                    for dx in range(2):
                        px = tx * 2 + dx
                        py = ty * 2 + dy
                        parent_path = os.path.join(out_dir, str(prev_level), f"{px}_{py}.jpg")
                        if px < prev_cols and py < prev_rows and os.path.exists(parent_path):
                            row_parts.append(pyvips.Image.new_from_file(parent_path, access="sequential"))
                        else:
                            # White fill for edge tiles
                            row_parts.append(pyvips.Image.black(dst_tile, dst_tile).invert())
                    parts.append(row_parts[0].join(row_parts[1], "horizontal"))

                mosaic = parts[0].join(parts[1], "vertical")

                # Downscale 2x with Lanczos3
                tile = mosaic.resize(0.5, kernel="lanczos3")
                out_path = os.path.join(level_dir, f"{tx}_{ty}.jpg")
                tile.jpegsave(out_path, Q=quality, optimize_coding=True, strip=True)

                done += 1

        elapsed = time.time() - t0
        print(f"  Level {level}: {cur_cols}x{cur_rows} = {total} tiles ({elapsed:.1f}s)", flush=True)

        prev_level = level
        prev_cols = cur_cols
        prev_rows = cur_rows


def write_dzi(out_dir, width, height, tile_size):
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


def write_summary(out_dir, quality):
    """Write summary.json with total byte count."""
    parent = os.path.dirname(out_dir)
    total_bytes = 0
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".jpg"):
                total_bytes += os.path.getsize(os.path.join(root, f))

    import json
    summary = {
        "mode": "ingest-jpeg-only",
        "baseq": quality,
        "total_bytes": total_bytes,
        "note": f"256px JPEG Q{quality} retiled from 1024px DICOM extract (pyvips Lanczos3)"
    }
    summary_path = os.path.join(parent, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Wrote summary: {summary_path} ({total_bytes / 1e6:.1f} MB)")


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

    width = args.width or src_cols * src_tile
    height = args.height or src_rows * src_tile
    print(f"Image dimensions: {width}x{height}")

    # Step 1: Split top level (crop 1024→256, no resampling)
    t_start = time.time()
    print(f"\nSplitting L{max_level} ({src_tile}px -> {dst_tile}px)...")
    new_cols, new_rows = split_top_level(src_dir, out_dir, max_level, src_tile, dst_tile, quality)
    print(f"New L{max_level}: {new_cols}x{new_rows} = {new_cols * new_rows} tiles")

    # Step 2: Build lower levels by 2x Lanczos3 downsampling
    print(f"\nBuilding lower levels from L{max_level-1} down to L0 (Lanczos3)...")
    build_lower_levels(out_dir, max_level, new_cols, new_rows, dst_tile, quality)

    # Step 3: Write DZI manifest
    write_dzi(out_dir, width, height, dst_tile)

    # Step 4: Write summary.json
    write_summary(out_dir, quality)

    # Count total tiles
    total = 0
    for level in range(max_level + 1):
        level_dir = os.path.join(out_dir, str(level))
        if os.path.isdir(level_dir):
            n = len([f for f in os.listdir(level_dir) if f.endswith(".jpg")])
            total += n

    elapsed = time.time() - t_start
    print(f"\nDone! {total} tiles across {max_level + 1} levels in {elapsed:.1f}s")


if __name__ == "__main__":
    main()