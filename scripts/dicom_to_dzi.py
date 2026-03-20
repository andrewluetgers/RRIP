#!/usr/bin/env python3
"""
Process DICOM WSI files → DZI image pyramids.

1. Scans the raw DICOM binary for encapsulated JPEG frames (no pydicom needed
   at runtime once the offset table exists).
2. Writes a .bot (Binary Offset Table) index beside each DICOM.
3. Extracts L0 JPEG tiles and builds a full DZI pyramid by downsampling.

Watch mode (--watch): polls the input directory and processes each DICOM as
soon as it stops growing (download complete).

Usage:
    # Process all DICOMs in a folder
    uv run python scripts/dicom_to_dzi.py --indir ~/dev/data/WSI/20260212_mayosh_example_dicoms --outdir ~/dev/data/WSI/dzi_output

    # Watch mode — process as files finish downloading
    uv run python scripts/dicom_to_dzi.py --indir ~/dev/data/WSI/20260212_mayosh_example_dicoms --outdir ~/dev/data/WSI/dzi_output --watch

    # Just generate offset tables (no DZI)
    uv run python scripts/dicom_to_dzi.py --indir ~/dev/data/WSI/20260212_mayosh_example_dicoms --bot-only
"""

import argparse
import math
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image

# --------------------------------------------------------------------------- #
#  Binary Offset Table (.bot) format
# --------------------------------------------------------------------------- #
#
#  Header (40 bytes):
#    magic     : 4s   "DBOT"
#    version   : u32  1
#    n_frames  : u32
#    tile_w    : u32
#    tile_h    : u32
#    total_w   : u32
#    total_h   : u32
#    tiles_x   : u32
#    tiles_y   : u32
#    reserved  : u32  0
#
#  Per-frame entry (12 bytes each):
#    offset    : u64  absolute file offset of JPEG data
#    length    : u32  JPEG data length in bytes
#
# --------------------------------------------------------------------------- #

BOT_MAGIC = b"DBOT"
BOT_VERSION = 1
BOT_HEADER_FMT = "<4sIIIIIIIII"  # 40 bytes
BOT_HEADER_SIZE = struct.calcsize(BOT_HEADER_FMT)
BOT_ENTRY_FMT = "<QI"  # 12 bytes
BOT_ENTRY_SIZE = struct.calcsize(BOT_ENTRY_FMT)

# DICOM constants
PIXEL_DATA_TAG = bytes([0xE0, 0x7F, 0x10, 0x00])  # (7FE0,0010) LE
ITEM_TAG = bytes([0xFE, 0xFF, 0x00, 0xE0])  # (FFFE,E000)
SEQ_DELIM = bytes([0xFE, 0xFF, 0xDD, 0xE0])  # (FFFE,E00D)


def parse_dicom_metadata(path: str) -> dict:
    """Parse DICOM metadata using pydicom (lightweight header-only read)."""
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    meta = {
        "tile_w": int(ds.Columns),
        "tile_h": int(ds.Rows),
        "n_frames": int(getattr(ds, "NumberOfFrames", 1)),
        "total_w": int(getattr(ds, "TotalPixelMatrixColumns", ds.Columns)),
        "total_h": int(getattr(ds, "TotalPixelMatrixRows", ds.Rows)),
    }
    meta["tiles_x"] = math.ceil(meta["total_w"] / meta["tile_w"])
    meta["tiles_y"] = math.ceil(meta["total_h"] / meta["tile_h"])
    return meta


def scan_frame_offsets(path: str) -> list[tuple[int, int]]:
    """Scan the raw DICOM file for encapsulated JPEG frame offsets.

    Returns list of (file_offset, length) for each frame.
    """
    with open(path, "rb") as f:
        data = f.read()

    # Find the PixelData tag
    pd_pos = data.find(PIXEL_DATA_TAG)
    if pd_pos < 0:
        raise ValueError(f"No PixelData tag found in {path}")

    pos = pd_pos + 4  # skip tag

    # Check VR
    vr = data[pos:pos + 2]
    if vr in (b"OB", b"OW"):
        pos += 2 + 2  # skip VR + reserved
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
    else:
        # Implicit VR
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4

    # First item = Basic Offset Table (skip it)
    tag = data[pos:pos + 4]
    if tag != ITEM_TAG:
        raise ValueError(f"Expected item tag at offset {pos}, got {tag.hex()}")
    bot_len = struct.unpack_from("<I", data, pos + 4)[0]
    pos += 8 + bot_len  # skip BOT

    # Scan data frames
    offsets = []
    while pos < len(data):
        tag = data[pos:pos + 4]
        if tag == SEQ_DELIM:
            break
        if tag != ITEM_TAG:
            raise ValueError(f"Unexpected tag {tag.hex()} at offset {pos}")
        flen = struct.unpack_from("<I", data, pos + 4)[0]
        offsets.append((pos + 8, flen))  # file offset of JPEG data, length
        pos += 8 + flen

    return offsets


def write_bot(bot_path: str, meta: dict, offsets: list[tuple[int, int]]):
    """Write a .bot (Binary Offset Table) file."""
    with open(bot_path, "wb") as f:
        f.write(struct.pack(
            BOT_HEADER_FMT,
            BOT_MAGIC,
            BOT_VERSION,
            len(offsets),
            meta["tile_w"],
            meta["tile_h"],
            meta["total_w"],
            meta["total_h"],
            meta["tiles_x"],
            meta["tiles_y"],
            0,  # reserved
        ))
        for offset, length in offsets:
            f.write(struct.pack(BOT_ENTRY_FMT, offset, length))
    return bot_path


def read_bot(bot_path: str) -> tuple[dict, list[tuple[int, int]]]:
    """Read a .bot file. Returns (meta_dict, [(offset, length), ...])."""
    with open(bot_path, "rb") as f:
        header = f.read(BOT_HEADER_SIZE)
        magic, version, n_frames, tw, th, totw, toth, tx, ty, _ = \
            struct.unpack(BOT_HEADER_FMT, header)
        assert magic == BOT_MAGIC, f"Bad magic: {magic}"
        assert version == BOT_VERSION, f"Bad version: {version}"
        meta = {
            "n_frames": n_frames, "tile_w": tw, "tile_h": th,
            "total_w": totw, "total_h": toth, "tiles_x": tx, "tiles_y": ty,
        }
        offsets = []
        for _ in range(n_frames):
            entry = f.read(BOT_ENTRY_SIZE)
            off, length = struct.unpack(BOT_ENTRY_FMT, entry)
            offsets.append((off, length))
    return meta, offsets


def extract_jpeg_frame(dcm_path: str, offset: int, length: int) -> bytes:
    """Read a single JPEG frame from the DICOM at a specific file offset."""
    with open(dcm_path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def generate_bot(dcm_path: str) -> tuple[str, dict]:
    """Generate the .bot offset table for a DICOM file.

    Returns (bot_path, meta).
    """
    bot_path = dcm_path + ".bot"
    if os.path.exists(bot_path):
        meta, _ = read_bot(bot_path)
        print(f"  BOT exists: {bot_path} ({meta['n_frames']} frames)")
        return bot_path, meta

    t0 = time.time()
    print(f"  Parsing metadata...")
    meta = parse_dicom_metadata(dcm_path)
    print(f"  {meta['total_w']}x{meta['total_h']} px, "
          f"{meta['tiles_x']}x{meta['tiles_y']} tiles of {meta['tile_w']}x{meta['tile_h']}, "
          f"{meta['n_frames']} frames")

    print(f"  Scanning frame offsets...")
    offsets = scan_frame_offsets(dcm_path)
    assert len(offsets) == meta["n_frames"], \
        f"Frame count mismatch: scanned {len(offsets)} vs metadata {meta['n_frames']}"

    write_bot(bot_path, meta, offsets)
    dt = time.time() - t0
    size_mb = os.path.getsize(dcm_path) / 1e6
    print(f"  BOT written: {bot_path} ({len(offsets)} frames, "
          f"{dt:.1f}s, {size_mb/dt:.0f} MB/s)")
    return bot_path, meta


def build_dzi_pyramid(dcm_path: str, bot_path: str, out_dir: str,
                      tile_size: int = 256, jpeg_quality: int = 90,
                      workers: int = 8):
    """Build a DZI pyramid from a DICOM WSI using the offset table.

    Extracts L0 JPEG tiles directly from the DICOM (zero-decode for the
    highest-resolution level), then generates lower levels by downsampling.
    """
    meta, offsets = read_bot(bot_path)
    tw, th = meta["tile_w"], meta["tile_h"]
    total_w, total_h = meta["total_w"], meta["total_h"]
    tiles_x, tiles_y = meta["tiles_x"], meta["tiles_y"]
    n_frames = meta["n_frames"]

    slide_name = Path(dcm_path).stem
    pyramid_dir = os.path.join(out_dir, f"{slide_name}_files")

    # Compute DZI level numbers
    max_dim = max(total_w, total_h)
    max_level = math.ceil(math.log2(max_dim))  # highest-res level
    print(f"  DZI: {max_level + 1} levels, max_level={max_level}")

    # --- Level max_level: extract L0 tiles directly from DICOM --- #
    l0_dir = os.path.join(pyramid_dir, str(max_level))
    os.makedirs(l0_dir, exist_ok=True)

    t0 = time.time()
    print(f"  Extracting {n_frames} L0 tiles (level {max_level})...")

    def extract_tile(args):
        idx, tx, ty = args
        if idx >= n_frames:
            return None
        offset, length = offsets[idx]
        jpeg_data = extract_jpeg_frame(dcm_path, offset, length)
        tile_path = os.path.join(l0_dir, f"{tx}_{ty}.jpeg")
        with open(tile_path, "wb") as f:
            f.write(jpeg_data)
        return tile_path

    # Build tile extraction jobs
    jobs = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            idx = ty * tiles_x + tx
            jobs.append((idx, tx, ty))

    extracted = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(extract_tile, jobs):
            if result:
                extracted += 1

    dt = time.time() - t0
    print(f"  L0: {extracted} tiles extracted in {dt:.1f}s "
          f"({extracted/dt:.0f} tiles/s)")

    # --- Generate lower levels by downsampling --- #
    for level in range(max_level - 1, -1, -1):
        parent_level = level + 1
        parent_dir = os.path.join(pyramid_dir, str(parent_level))
        level_dir = os.path.join(pyramid_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)

        # Dimensions at this level
        level_w = math.ceil(total_w / (2 ** (max_level - level)))
        level_h = math.ceil(total_h / (2 ** (max_level - level)))
        level_tiles_x = math.ceil(level_w / tile_size)
        level_tiles_y = math.ceil(level_h / tile_size)

        if level_w < 1 or level_h < 1:
            break

        t1 = time.time()

        def downsample_tile(args):
            tx, ty = args
            # This tile corresponds to a 2x2 group from the parent level
            children = []
            for dy in range(2):
                for dx in range(2):
                    px, py = tx * 2 + dx, ty * 2 + dy
                    child_path = os.path.join(parent_dir, f"{px}_{py}.jpeg")
                    if os.path.exists(child_path):
                        children.append((dx, dy, child_path))

            if not children:
                return None

            # Compose 2x2 mosaic from parent tiles
            mosaic = Image.new("RGB", (tile_size * 2, tile_size * 2), (255, 255, 255))
            for dx, dy, child_path in children:
                try:
                    child = Image.open(child_path)
                    mosaic.paste(child, (dx * tile_size, dy * tile_size))
                except Exception:
                    pass

            # Downsample 2x to tile_size
            tile = mosaic.resize((tile_size, tile_size), Image.LANCZOS)
            tile_path = os.path.join(level_dir, f"{tx}_{ty}.jpeg")
            tile.save(tile_path, "JPEG", quality=jpeg_quality)
            return tile_path

        jobs = [(tx, ty) for ty in range(level_tiles_y)
                for tx in range(level_tiles_x)]

        generated = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(downsample_tile, jobs):
                if result:
                    generated += 1

        dt1 = time.time() - t1
        print(f"  Level {level}: {level_w}x{level_h} px, "
              f"{generated} tiles ({dt1:.1f}s)")

        # Stop once we've produced a single-tile level (thumbnail)
        if level_tiles_x <= 1 and level_tiles_y <= 1:
            print(f"  Stopping at level {level} (single tile)")
            break

    # --- Write DZI manifest --- #
    dzi_path = os.path.join(out_dir, f"{slide_name}.dzi")
    dzi_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jpeg"
       Overlap="0"
       TileSize="{tile_size}">
  <Size Width="{total_w}" Height="{total_h}"/>
</Image>
"""
    with open(dzi_path, "w") as f:
        f.write(dzi_xml)

    total_dt = time.time() - t0
    print(f"  DZI written: {dzi_path}")
    print(f"  Total pyramid build: {total_dt:.1f}s")


def is_file_stable(path: str, wait_secs: float = 3.0) -> bool:
    """Check if a file has stopped growing (download complete)."""
    try:
        size1 = os.path.getsize(path)
        time.sleep(wait_secs)
        size2 = os.path.getsize(path)
        return size1 == size2 and size1 > 0
    except OSError:
        return False


def process_dicom(dcm_path: str, out_dir: str | None, bot_only: bool,
                  tile_size: int, jpeg_quality: int, workers: int) -> bool:
    """Process a single DICOM: generate BOT, optionally build DZI.

    Returns True if work was done, False if skipped (already complete).
    """
    name = os.path.basename(dcm_path)
    slide_name = Path(dcm_path).stem
    size_mb = os.path.getsize(dcm_path) / 1e6

    # Skip if DZI already exists (complete)
    if not bot_only and out_dir:
        dzi_path = os.path.join(out_dir, f"{slide_name}.dzi")
        if os.path.exists(dzi_path):
            print(f"  Skipping {name[:40]}... (DZI already exists)")
            return False
    print(f"\n{'='*60}")
    print(f"Processing: {name} ({size_mb:.0f} MB)")
    print(f"{'='*60}")

    bot_path, meta = generate_bot(dcm_path)

    if not bot_only and out_dir:
        build_dzi_pyramid(dcm_path, bot_path, out_dir,
                          tile_size=tile_size, jpeg_quality=jpeg_quality,
                          workers=workers)


def main():
    ap = argparse.ArgumentParser(
        description="Process DICOM WSI files → DZI image pyramids")
    ap.add_argument("--indir", required=True,
                    help="Directory containing .dcm files")
    ap.add_argument("--outdir", default=None,
                    help="Output directory for DZI pyramids")
    ap.add_argument("--bot-only", action="store_true",
                    help="Only generate .bot offset tables, no DZI")
    ap.add_argument("--watch", action="store_true",
                    help="Watch mode: poll for new/completed downloads")
    ap.add_argument("--poll-interval", type=float, default=5.0,
                    help="Watch mode poll interval in seconds (default: 5)")
    ap.add_argument("--tile-size", type=int, default=256,
                    help="DZI tile size (default: 256)")
    ap.add_argument("--quality", type=int, default=90,
                    help="JPEG quality for downsampled levels (default: 90)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Thread pool workers (default: 8)")
    args = ap.parse_args()

    if not args.bot_only and not args.outdir:
        print("Error: --outdir required unless --bot-only is set")
        sys.exit(1)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    processed = set()

    def scan_and_process():
        dcm_files = sorted(
            f for f in os.listdir(args.indir)
            if f.endswith(".dcm") and f not in processed
        )
        for fname in dcm_files:
            dcm_path = os.path.join(args.indir, fname)
            if not is_file_stable(dcm_path, wait_secs=2.0):
                print(f"  Skipping {fname} (still downloading...)")
                continue
            try:
                process_dicom(dcm_path, args.outdir, args.bot_only,
                              args.tile_size, args.quality, args.workers)
                processed.add(fname)
            except Exception as e:
                print(f"  ERROR processing {fname}: {e}")
                import traceback
                traceback.print_exc()

    if args.watch:
        print(f"Watching {args.indir} for .dcm files (poll every {args.poll_interval}s)...")
        print("Press Ctrl+C to stop.\n")
        while True:
            scan_and_process()
            all_dcm = [f for f in os.listdir(args.indir) if f.endswith(".dcm")]
            remaining = len(all_dcm) - len(processed)
            if remaining > 0:
                print(f"\n  Waiting... ({len(processed)} done, {remaining} remaining)")
            time.sleep(args.poll_interval)
    else:
        scan_and_process()

    print(f"\nDone. Processed {len(processed)} DICOM files.")


if __name__ == "__main__":
    main()
