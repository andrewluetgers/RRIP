#!/usr/bin/env python3
"""
Watch GCS for newly extracted slides and sync their tiles locally.

Runs on RunPod alongside training. Polls GCS for new slide directories,
downloads their tiles, and signals the training process via a manifest file.

Usage:
  # Run in background while training
  python tile_watcher.py \
      --bucket gs://wsi-1-480715-tcga-tiles/stage1 \
      --local /workspace/tiles \
      --poll-interval 30 &

  # Training reads from /workspace/tiles, which grows as slides arrive
  python train.py --tiles /workspace/tiles --watch-manifest /workspace/tiles/available.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def gsutil_ls(gcs_path: str) -> list:
    """List directories in GCS path."""
    try:
        out = subprocess.run(
            ["gsutil", "ls", gcs_path],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return []
        return [l.strip().rstrip("/") for l in out.stdout.strip().split("\n") if l.strip()]
    except Exception:
        return []


def gsutil_sync(gcs_path: str, local_path: str) -> bool:
    """Download a GCS directory to local."""
    os.makedirs(local_path, exist_ok=True)
    try:
        result = subprocess.run(
            ["gsutil", "-m", "-q", "cp", "-r", gcs_path + "/*", local_path + "/"],
            capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception:
        return False


def count_tiles(local_dir: str) -> int:
    """Count .jpg files recursively."""
    count = 0
    for root, _, files in os.walk(local_dir):
        count += sum(1 for f in files if f.endswith(".jpg"))
    return count


def main():
    ap = argparse.ArgumentParser(description="Watch GCS for new tiles")
    ap.add_argument("--bucket", required=True, help="GCS path (e.g. gs://bucket/stage1)")
    ap.add_argument("--local", required=True, help="Local tile directory")
    ap.add_argument("--poll-interval", type=int, default=30, help="Seconds between polls")
    ap.add_argument("--role", default="train", help="Tile role subdirectory to sync (train, eval_adjacent)")
    args = ap.parse_args()

    os.makedirs(args.local, exist_ok=True)
    manifest_path = os.path.join(args.local, "available.json")
    synced_slides = set()

    # Load previously synced slides
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            data = json.load(f)
            synced_slides = set(data.get("synced_slides", []))
        print(f"Resuming: {len(synced_slides)} slides already synced")

    print(f"Watching: {args.bucket}")
    print(f"Local:    {args.local}")
    print(f"Role:     {args.role}")
    print(f"Poll:     every {args.poll_interval}s")
    print()

    while True:
        # List cancer type directories
        type_dirs = gsutil_ls(args.bucket)
        # Filter out status/ and other non-slide dirs
        type_dirs = [d for d in type_dirs if not d.endswith("/status") and
                     not d.endswith("/manifests") and not d.endswith("/checkpoints")]

        new_count = 0
        for type_dir in type_dirs:
            type_name = type_dir.split("/")[-1]
            if not type_name.startswith("TCGA-"):
                continue

            # List slide directories within this cancer type
            slide_dirs = gsutil_ls(type_dir)
            for slide_dir in slide_dirs:
                slide_id = slide_dir.split("/")[-1]
                slide_key = f"{type_name}/{slide_id}"

                if slide_key in synced_slides:
                    continue

                # New slide — sync its tiles
                gcs_tile_path = f"{slide_dir}/{args.role}"
                local_tile_path = os.path.join(args.local, slide_key, args.role)

                # Check if the role subdir exists in GCS
                role_contents = gsutil_ls(gcs_tile_path)
                if not role_contents:
                    continue

                print(f"  Syncing: {slide_key}/{args.role}...")
                if gsutil_sync(gcs_tile_path, local_tile_path):
                    n = count_tiles(local_tile_path)
                    synced_slides.add(slide_key)
                    new_count += 1
                    print(f"    Got {n} tiles")
                else:
                    print(f"    Sync failed, will retry")

        if new_count > 0:
            # Update manifest
            total_tiles = count_tiles(args.local)
            manifest = {
                "synced_slides": sorted(synced_slides),
                "total_slides": len(synced_slides),
                "total_tiles": total_tiles,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tile_dir": args.local,
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"  Updated manifest: {len(synced_slides)} slides, {total_tiles} tiles")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
