#!/usr/bin/env python3
"""retile.py — Re-tile a DICOM WSI to a different tile size.

Reads a DICOM WSI file, decodes all highest-resolution tiles,
re-tiles them to the target size, re-encodes as JPEG, and writes
a new DICOM file with the new tile layout.

Usage:
    python retile.py --input /path/to/slide.dcm --output /path/to/output.dcm --tile-size 256 --quality 98
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def decode_jpeg_fragment(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB numpy array."""
    img = Image.open(io.BytesIO(jpeg_bytes))
    return np.array(img.convert("RGB"))


def encode_jpeg_tile(tile: np.ndarray, quality: int) -> bytes:
    """Encode RGB numpy array to JPEG bytes."""
    img = Image.fromarray(tile)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=0)  # 4:4:4
    return buf.getvalue()


def retile(input_path: str, output_path: str, new_tile_size: int, quality: int):
    t0 = time.time()

    print(f"Reading DICOM: {input_path}")
    ds = pydicom.dcmread(input_path)

    orig_tile_w = int(ds.Columns)
    orig_tile_h = int(ds.Rows)
    total_w = int(ds.TotalPixelMatrixColumns)
    total_h = int(ds.TotalPixelMatrixRows)

    orig_tiles_x = (total_w + orig_tile_w - 1) // orig_tile_w
    orig_tiles_y = (total_h + orig_tile_h - 1) // orig_tile_h

    pixel_data = ds["PixelData"]
    if not pixel_data.is_undefined_length:
        print("ERROR: PixelData is not encapsulated")
        sys.exit(1)

    fragments = pydicom.encaps.decode_data_sequence(pixel_data.value)
    n_frags = len(fragments)
    frag_mb = sum(len(f) for f in fragments) / 1_048_576

    print(f"Original: {total_w}x{total_h}, tile={orig_tile_w}x{orig_tile_h}, "
          f"grid={orig_tiles_x}x{orig_tiles_y}, {n_frags} fragments ({frag_mb:.1f} MB)")
    print(f"New tile size: {new_tile_size}x{new_tile_size}, quality={quality}")

    new_tiles_x = (total_w + new_tile_size - 1) // new_tile_size
    new_tiles_y = (total_h + new_tile_size - 1) // new_tile_size

    print(f"New grid: {new_tiles_x}x{new_tiles_y} = {new_tiles_x * new_tiles_y} tiles")

    # Process in horizontal strips of height = new_tile_size.
    # For each strip, decode original tiles that overlap, assemble, re-cut.
    new_fragments = []
    tiles_done = 0

    for new_ty in range(new_tiles_y):
        y_start = new_ty * new_tile_size
        y_end = min(y_start + new_tile_size, total_h)
        strip_h = y_end - y_start

        orig_ty_start = y_start // orig_tile_h
        orig_ty_end = min((y_end - 1) // orig_tile_h + 1, orig_tiles_y)

        strip = np.zeros((strip_h, total_w, 3), dtype=np.uint8)

        for orig_ty in range(orig_ty_start, orig_ty_end):
            for orig_tx in range(orig_tiles_x):
                frag_idx = orig_ty * orig_tiles_x + orig_tx
                if frag_idx >= n_frags:
                    continue
                frag = fragments[frag_idx]
                if len(frag) == 0:
                    continue

                try:
                    tile_img = decode_jpeg_fragment(frag)
                except Exception:
                    continue

                th, tw = tile_img.shape[:2]
                ox = orig_tx * orig_tile_w
                oy = orig_ty * orig_tile_h

                src_y0 = max(0, y_start - oy)
                src_y1 = min(th, y_end - oy)
                dst_y0 = max(0, oy - y_start)
                dst_y1 = dst_y0 + (src_y1 - src_y0)

                src_x0 = 0
                src_x1 = min(tw, total_w - ox)
                dst_x0 = ox
                dst_x1 = dst_x0 + (src_x1 - src_x0)

                if src_y1 > src_y0 and src_x1 > src_x0:
                    strip[dst_y0:dst_y1, dst_x0:dst_x1] = \
                        tile_img[src_y0:src_y1, src_x0:src_x1]

        for new_tx in range(new_tiles_x):
            x_start = new_tx * new_tile_size
            x_end = min(x_start + new_tile_size, total_w)
            tw = x_end - x_start
            th = strip_h

            tile = strip[:th, x_start:x_end]

            if tw < new_tile_size or th < new_tile_size:
                padded = np.zeros((new_tile_size, new_tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            jpeg_bytes = encode_jpeg_tile(tile, quality)
            new_fragments.append(jpeg_bytes)
            tiles_done += 1

        if new_ty % 100 == 0 or new_ty == new_tiles_y - 1:
            elapsed = time.time() - t0
            pct = (new_ty + 1) / new_tiles_y * 100
            print(f"  row {new_ty+1}/{new_tiles_y} ({pct:.0f}%) — "
                  f"{tiles_done} tiles, {elapsed:.1f}s")

    print(f"Re-tiled {tiles_done} tiles in {time.time() - t0:.1f}s")

    # Write new DICOM
    print("Writing output DICOM...")
    t_write = time.time()

    ds.Rows = new_tile_size
    ds.Columns = new_tile_size
    ds.NumberOfFrames = str(len(new_fragments))

    encapsulated = pydicom.encaps.encapsulate(new_fragments)
    ds.PixelData = encapsulated
    ds["PixelData"].is_undefined_length = True
    ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.50"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))

    out_size_mb = output_path.stat().st_size / 1_048_576
    new_frag_mb = sum(len(f) for f in new_fragments) / 1_048_576

    print(f"Wrote {output_path} ({out_size_mb:.1f} MB) in {time.time() - t_write:.1f}s")
    print(f"Total: {time.time() - t0:.1f}s")
    print(f"  {total_w}x{total_h} @ {orig_tile_w}x{orig_tile_h} -> "
          f"{new_tile_size}x{new_tile_size} ({len(new_fragments)} tiles, Q{quality})")
    print(f"  JPEG data: {frag_mb:.1f} MB -> {new_frag_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Re-tile a DICOM WSI to a different tile size")
    parser.add_argument("--input", "-i", required=True, help="Input DICOM WSI file")
    parser.add_argument("--output", "-o", required=True, help="Output DICOM WSI file")
    parser.add_argument("--tile-size", "-t", type=int, default=256, help="New tile size (default: 256)")
    parser.add_argument("--quality", "-q", type=int, default=98, help="JPEG quality (default: 98)")
    args = parser.parse_args()

    retile(args.input, args.output, args.tile_size, args.quality)


if __name__ == "__main__":
    main()
