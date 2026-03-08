#!/usr/bin/env python3
"""
Extract 1024x1024 tiles from a DICOM WSI for SR training.

Blank tile detection: a tile is considered blank (pure background) if ALL pixels
are within --blank-tolerance of white (255). This means the pixel value range
(max - min) is very small and centered near 255. Tiles with any tissue — even
faint margin — will have a wider intensity distribution and are kept.

Usage:
  python prepare_tiles.py --dcm /path/to/000005.dcm --outdir tiles/train
  python prepare_tiles.py --dcm /path/to/000005.dcm --outdir tiles/train --blank-tolerance 15
"""

import os
import argparse

import numpy as np
from PIL import Image


def is_blank_tile(frame: np.ndarray, tolerance: int = 10) -> bool:
    """Check if a tile is uniformly near-white (pure background).

    A tile is blank if:
      - All pixels are within `tolerance` of 255 (every channel)
      - The standard deviation is very low (uniform)

    This keeps tiles with tissue at the margin (which have a bimodal
    histogram: white background + colored tissue).
    """
    # Check: are ALL pixels close to white?
    min_val = frame.min()
    if min_val < (255 - tolerance):
        # At least one pixel is far from white → has tissue
        return False

    # Double-check: very low variance confirms uniform background
    if frame.std() > tolerance / 2:
        return False

    return True


def extract_tiles_dicom(dcm_path: str, outdir: str, max_tiles: int = 0,
                        blank_tolerance: int = 10):
    """Extract tiles from a DICOM WSI file.

    Args:
        dcm_path: Path to DICOM file with tiled frames
        outdir: Output directory for PNG tiles
        max_tiles: Max tiles to extract (0=all)
        blank_tolerance: Pixel tolerance from 255 to consider blank
    """
    import pydicom

    os.makedirs(outdir, exist_ok=True)

    ds = pydicom.dcmread(dcm_path)
    rows = ds.Rows
    cols = ds.Columns
    n_frames = int(getattr(ds, 'NumberOfFrames', 1))
    total_w = int(getattr(ds, 'TotalPixelMatrixColumns', cols))
    total_h = int(getattr(ds, 'TotalPixelMatrixRows', rows))

    print(f"DICOM: {n_frames} frames of {cols}x{rows}, total {total_w}x{total_h}")
    print(f"Output: {outdir}")
    print(f"Blank tolerance: {blank_tolerance} (skip tiles where all pixels > {255 - blank_tolerance})")

    saved = 0
    skipped_blank = 0
    skipped_size = 0
    pixel_data = None

    for frame_idx in range(n_frames):
        if max_tiles > 0 and saved >= max_tiles:
            break

        # Extract frame
        try:
            if pixel_data is None:
                ds.decompress()
                pixel_data = ds.pixel_array

            if pixel_data.ndim == 4:
                frame = pixel_data[frame_idx]
            elif pixel_data.ndim == 3:
                frame = pixel_data
            else:
                continue
        except Exception:
            try:
                from pydicom.encaps import decode_data_sequence
                from io import BytesIO
                frames_data = decode_data_sequence(ds[0x7FE00010].value)
                frame_bytes = frames_data[frame_idx]
                img = Image.open(BytesIO(frame_bytes))
                frame = np.array(img)
            except Exception as e:
                print(f"  Frame {frame_idx}: extraction failed ({e}), skipping")
                continue

        # Ensure correct size (edge tiles may be smaller)
        h, w = frame.shape[:2]
        if h != 1024 or w != 1024:
            skipped_size += 1
            continue

        # Skip blank tiles
        if is_blank_tile(frame, tolerance=blank_tolerance):
            skipped_blank += 1
            continue

        # Save as PNG
        out_path = os.path.join(outdir, f"tile_{frame_idx:05d}.png")
        Image.fromarray(frame).save(out_path)
        saved += 1

        if saved % 100 == 0:
            print(f"  Saved {saved} tiles (skipped {skipped_blank} blank, {skipped_size} undersized)")

    print(f"Done: {saved} tiles saved, {skipped_blank} blank skipped, {skipped_size} undersized skipped")


def extract_tiles_openslide(svs_path: str, outdir: str, max_tiles: int = 0,
                             blank_tolerance: int = 10, tile_size: int = 1024):
    """Extract tiles from an SVS/NDPI/TIFF WSI using OpenSlide."""
    import openslide

    os.makedirs(outdir, exist_ok=True)

    slide = openslide.open_slide(svs_path)
    w, h = slide.dimensions
    print(f"Slide: {w}x{h}, levels: {slide.level_count}")

    saved = 0
    skipped_blank = 0

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if max_tiles > 0 and saved >= max_tiles:
                break

            tw = min(tile_size, w - x)
            th = min(tile_size, h - y)
            if tw < tile_size or th < tile_size:
                continue

            tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
            arr = np.array(tile)

            if is_blank_tile(arr, tolerance=blank_tolerance):
                skipped_blank += 1
                continue

            out_path = os.path.join(outdir, f"tile_{x}_{y}.png")
            tile.save(out_path)
            saved += 1

            if saved % 100 == 0:
                print(f"  Saved {saved} tiles (skipped {skipped_blank} blank)")
        else:
            continue
        break

    print(f"Done: {saved} tiles saved, {skipped_blank} blank skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcm", help="DICOM WSI file")
    ap.add_argument("--svs", help="SVS/NDPI/TIFF WSI file (via OpenSlide)")
    ap.add_argument("--outdir", default="tiles/train", help="Output tile directory")
    ap.add_argument("--max-tiles", type=int, default=0, help="Max tiles (0=all)")
    ap.add_argument("--blank-tolerance", type=int, default=10,
                    help="Blank detection: skip tiles where ALL pixels are within N of 255")
    args = ap.parse_args()

    if args.dcm:
        extract_tiles_dicom(args.dcm, args.outdir, args.max_tiles, args.blank_tolerance)
    elif args.svs:
        extract_tiles_openslide(args.svs, args.outdir, args.max_tiles, args.blank_tolerance)
    else:
        print("Provide --dcm or --svs")


if __name__ == "__main__":
    main()
