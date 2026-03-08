#!/usr/bin/env python3
"""
Download TCGA SVS slides from GDC API and extract tiles.

Runs on a GCP VM (not locally). Streams slides one at a time:
  1. Download SVS from GDC API
  2. Extract tiles at configured stride using OpenSlide
  3. Save JPEG tiles to local disk
  4. Delete SVS file
  5. Upload tiles to GCS bucket

Usage:
  python extract_tiles_tcga.py --manifest tcga_stage1_manifest.json \
      --output /mnt/ssd/tiles --bucket gs://wsi-1-480715-tcga-tiles/stage1

  # Limit to N slides (for testing)
  python extract_tiles_tcga.py --manifest tcga_stage1_manifest.json \
      --output /mnt/ssd/tiles --bucket gs://wsi-1-480715-tcga-tiles/stage1 --limit 2
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import openslide
except ImportError:
    print("ERROR: pip install openslide-python")
    print("Also need: apt-get install openslide-tools (Ubuntu) or brew install openslide (macOS)")
    sys.exit(1)

from PIL import Image


def download_slide(file_id: str, output_path: str) -> bool:
    """Download a single SVS file from GDC API."""
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    try:
        result = subprocess.run(
            ["curl", "-sL", "-o", output_path,
             "--retry", "3", "--retry-delay", "5",
             "-H", "Accept: application/octet-stream",
             url],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    curl failed: {result.stderr[:200]}")
            return False
        # Verify file is not empty / not an error response
        size = os.path.getsize(output_path)
        if size < 1_000_000:  # SVS files are always > 1MB
            with open(output_path, "r", errors="ignore") as f:
                head = f.read(200)
            if "error" in head.lower() or "html" in head.lower():
                print(f"    Got error response instead of SVS ({size} bytes)")
                return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    Download timed out")
        return False
    except Exception as e:
        print(f"    Download error: {e}")
        return False


def extract_tiles(svs_path: str, output_dir: str, tile_size: int = 1024,
                  stride: int = 4, quality: int = 95, role_filter: str = None) -> dict:
    """Extract tiles from an SVS file using OpenSlide.

    Returns dict with tile counts and metadata.
    """
    slide = openslide.OpenSlide(svs_path)

    # Get metadata
    mpp = float(slide.properties.get("openslide.mpp-x", 0))
    objective = slide.properties.get("openslide.objective-power", "unknown")
    vendor = slide.properties.get("openslide.vendor", "unknown")
    w, h = slide.dimensions  # Level 0 dimensions

    os.makedirs(output_dir, exist_ok=True)

    tiles_x = w // tile_size
    tiles_y = h // tile_size

    extracted = {"train": 0, "eval_adjacent": 0, "total": 0}

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_id = tx + ty * tiles_x

            # Determine role based on stride
            if tile_id % stride == 0:
                role = "train"
            elif tile_id % stride == 1:
                role = "eval_adjacent"
            else:
                continue  # Skip

            if role_filter and role != role_filter:
                continue

            # Read tile from level 0
            x_pos = tx * tile_size
            y_pos = ty * tile_size

            try:
                tile = slide.read_region((x_pos, y_pos), 0, (tile_size, tile_size))
                tile = tile.convert("RGB")
            except Exception as e:
                continue

            # Skip mostly-white/background tiles
            import numpy as np
            arr = np.array(tile)
            mean_val = arr.mean()
            if mean_val > 230:  # mostly white background
                continue
            # Skip mostly-black tiles too
            if mean_val < 20:
                continue

            # Save as JPEG
            tile_name = f"{tx}_{ty}.jpg"
            role_dir = os.path.join(output_dir, role)
            os.makedirs(role_dir, exist_ok=True)
            tile_path = os.path.join(role_dir, tile_name)
            tile.save(tile_path, "JPEG", quality=quality)

            extracted[role] = extracted.get(role, 0) + 1
            extracted["total"] += 1

    slide.close()

    metadata = {
        "mpp": mpp,
        "objective_power": objective,
        "vendor": vendor,
        "dimensions": [w, h],
        "tiles_grid": [tiles_x, tiles_y],
        "extracted": extracted,
    }

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def upload_to_gcs(local_dir: str, gcs_path: str):
    """Upload a directory to GCS using gsutil."""
    result = subprocess.run(
        ["gsutil", "-m", "-q", "cp", "-r", local_dir, gcs_path],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    Upload failed: {result.stderr[:200]}")
        return False
    return True


def write_progress(bucket: str, progress: dict):
    """Write progress JSON to GCS for live monitoring."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(progress, f)
            tmp = f.name
        subprocess.run(
            ["gsutil", "-q", "cp", tmp, f"{bucket}/status/extract_progress.json"],
            capture_output=True, timeout=15)
        os.unlink(tmp)
    except Exception:
        pass  # Non-fatal — monitoring is best-effort


def main():
    ap = argparse.ArgumentParser(description="Extract tiles from TCGA slides")
    ap.add_argument("--manifest", required=True, help="Training manifest JSON")
    ap.add_argument("--output", required=True, help="Local output directory for tiles")
    ap.add_argument("--bucket", required=True, help="GCS bucket path (e.g. gs://bucket/prefix)")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of slides (0=all)")
    ap.add_argument("--skip-upload", action="store_true", help="Skip GCS upload")
    ap.add_argument("--keep-svs", action="store_true", help="Don't delete SVS after extraction")
    ap.add_argument("--roles", default="train,eval_adjacent",
                    help="Tile roles to extract (comma-separated)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    config = manifest.get("config", {})
    stride = config.get("tile_stride", 4)

    # Build slide list from manifest
    slides = []
    for ct in manifest.get("cancer_types", []):
        project_id = ct["project_id"]
        for fid in ct.get("train_file_ids", []):
            slides.append({"file_id": fid, "project_id": project_id, "role": "train"})
        for fid in ct.get("eval_file_ids", []):
            slides.append({"file_id": fid, "project_id": project_id, "role": "eval"})

    if args.limit > 0:
        slides = slides[:args.limit]

    print(f"Extraction plan:")
    print(f"  Slides: {len(slides)}")
    print(f"  Tile size: {args.tile_size}")
    print(f"  Stride: {stride}")
    print(f"  Quality: {args.quality}")
    print(f"  Output: {args.output}")
    print(f"  Bucket: {args.bucket}")
    print()

    os.makedirs(args.output, exist_ok=True)
    svs_dir = os.path.join(args.output, "_svs_tmp")
    os.makedirs(svs_dir, exist_ok=True)

    total_tiles = 0
    total_bytes = 0
    failed = []
    t_start = time.time()

    # Process results CSV for tracking
    results_path = os.path.join(args.output, "extraction_results.csv")
    with open(results_path, "w") as rf:
        writer = csv.writer(rf)
        writer.writerow(["file_id", "project_id", "role", "status",
                         "train_tiles", "eval_tiles", "mpp", "objective",
                         "elapsed_s"])

    for i, slide in enumerate(slides):
        fid = slide["file_id"]
        project = slide["project_id"]
        slide_role = slide["role"]

        print(f"[{i+1}/{len(slides)}] {project} {fid[:12]}... ({slide_role})")

        # 1. Download
        svs_path = os.path.join(svs_dir, f"{fid}.svs")
        t0 = time.time()

        print(f"  Downloading...")
        # Write "downloading" status so monitor shows activity
        if not args.skip_upload:
            write_progress(args.bucket, {
                "stage": "extract",
                "slides_done": i,
                "total_slides": len(slides),
                "total_tiles": total_tiles,
                "errors": len(failed),
                "current_slide": f"{project}/{fid[:12]}",
                "current_step": "downloading",
                "elapsed_min": round((time.time() - t_start) / 60, 1),
                "eta_min": round((len(slides) - i) * (time.time() - t_start) / max(i, 1) / 60, 0) if i > 0 else None,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        if not download_slide(fid, svs_path):
            failed.append(fid)
            print(f"  FAILED to download")
            continue

        svs_size = os.path.getsize(svs_path) / 1e6
        print(f"  Downloaded {svs_size:.0f} MB in {time.time()-t0:.1f}s")

        # Write "extracting" status
        if not args.skip_upload:
            write_progress(args.bucket, {
                "stage": "extract",
                "slides_done": i,
                "total_slides": len(slides),
                "total_tiles": total_tiles,
                "errors": len(failed),
                "current_slide": f"{project}/{fid[:12]}",
                "current_step": f"extracting ({svs_size:.0f}MB)",
                "elapsed_min": round((time.time() - t_start) / 60, 1),
                "eta_min": round((len(slides) - i) * (time.time() - t_start) / max(i, 1) / 60, 0) if i > 0 else None,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        # 2. Extract tiles
        tile_dir = os.path.join(args.output, project, fid[:12])
        t1 = time.time()

        try:
            meta = extract_tiles(svs_path, tile_dir,
                                 tile_size=args.tile_size,
                                 stride=stride,
                                 quality=args.quality)
        except Exception as e:
            print(f"  FAILED to extract: {e}")
            failed.append(fid)
            if not args.keep_svs:
                os.remove(svs_path)
            continue

        ext = meta["extracted"]
        n_train = ext.get("train", 0)
        n_eval = ext.get("eval_adjacent", 0)
        total_tiles += ext["total"]

        print(f"  Extracted {ext['total']} tiles "
              f"(train={n_train}, eval={n_eval}) "
              f"in {time.time()-t1:.1f}s  "
              f"[{meta.get('objective_power', '?')}x, mpp={meta.get('mpp', 0):.3f}]")

        # 3. Delete SVS
        if not args.keep_svs:
            os.remove(svs_path)

        # 4. Upload to GCS
        if not args.skip_upload:
            gcs_dest = f"{args.bucket}/{project}/{fid[:12]}"
            if upload_to_gcs(tile_dir, gcs_dest):
                print(f"  Uploaded to {gcs_dest}")
            else:
                print(f"  Upload failed!")

        # Log result
        with open(results_path, "a") as rf:
            writer = csv.writer(rf)
            writer.writerow([fid, project, slide_role, "ok",
                             n_train, n_eval,
                             meta.get("mpp", ""), meta.get("objective_power", ""),
                             f"{time.time()-t0:.1f}"])

        elapsed_total = time.time() - t_start
        rate = (i + 1) / elapsed_total * 60
        remaining = (len(slides) - i - 1) / max(rate, 0.01)
        print(f"  Total: {total_tiles} tiles, {elapsed_total:.0f}s elapsed, "
              f"~{remaining:.0f} min remaining")
        print()

        # Write live progress to GCS for monitoring
        if not args.skip_upload:
            write_progress(args.bucket, {
                "stage": "extract",
                "slides_done": i + 1,
                "total_slides": len(slides),
                "total_tiles": total_tiles,
                "errors": len(failed),
                "current_slide": f"{project}/{fid[:12]}",
                "elapsed_min": round(elapsed_total / 60, 1),
                "eta_min": round(remaining, 0),
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    # Summary
    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"EXTRACTION COMPLETE")
    print(f"  Slides processed: {len(slides) - len(failed)}/{len(slides)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if failed:
        print(f"  Failed IDs: {failed}")
    print(f"  Results CSV: {results_path}")


if __name__ == "__main__":
    main()
