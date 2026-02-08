#!/usr/bin/env python3
"""
Reorganize debug output folders with timestamped naming convention.
Format: debug_{image_name}_{params}_{timestamp}/
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import argparse


def get_folder_metadata(folder_path):
    """Extract metadata from existing debug folder."""
    manifest_path = folder_path / "manifest.json"

    # Default values
    metadata = {
        "image_name": "unknown",
        "quantization": None,
        "jpeg_quality": None,
        "is_pac": "_pac" in folder_path.name
    }

    # Parse from folder name
    folder_name = folder_path.name
    if "_q" in folder_name:
        parts = folder_name.split("_q")[-1]
        if "_j" in parts:
            q_part = parts.split("_j")[0]
            j_part = parts.split("_j")[1].replace("_pac", "")
            metadata["quantization"] = q_part
            metadata["jpeg_quality"] = j_part

    # Try to read manifest for more info
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
                # Check for image path in different possible locations
                if "image_path" in manifest:
                    # Extract image name from path
                    image_path = Path(manifest["image_path"])
                    metadata["image_name"] = image_path.stem
                elif "configuration" in manifest and "image_path" in manifest["configuration"]:
                    image_path = Path(manifest["configuration"]["image_path"])
                    metadata["image_name"] = image_path.stem
        except Exception as e:
            print(f"Warning: Could not read manifest from {manifest_path}: {e}")

    # If still unknown, try to extract from special folder names
    if metadata["image_name"] == "unknown":
        if "production" in folder_name:
            metadata["image_name"] = "production"

    # Get folder modification time as timestamp
    folder_stat = os.stat(folder_path)
    metadata["timestamp"] = datetime.fromtimestamp(folder_stat.st_mtime)

    return metadata


def generate_new_name(metadata):
    """Generate new folder name with timestamp."""
    parts = ["debug"]

    # Add image name (sanitized)
    image_name = metadata["image_name"].replace(" ", "_").replace("/", "_")
    parts.append(image_name)

    # Add parameters
    if metadata["quantization"]:
        parts.append(f"q{metadata['quantization']}")
    if metadata["jpeg_quality"]:
        parts.append(f"j{metadata['jpeg_quality']}")
    if metadata["is_pac"]:
        parts.append("pac")

    # Add timestamp
    timestamp_str = metadata["timestamp"].strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp_str)

    return "_".join(parts)


def reorganize_debug_folders(paper_dir, dry_run=False):
    """Reorganize all debug folders in the paper directory."""
    paper_path = Path(paper_dir)

    # Find all debug folders
    debug_folders = [f for f in paper_path.iterdir()
                    if f.is_dir() and f.name.startswith("debug_")]

    print(f"Found {len(debug_folders)} debug folders to reorganize")

    renames = []

    for folder in debug_folders:
        # Skip if already has timestamp format (contains 8 digits followed by underscore)
        import re
        if re.search(r'\d{8}_\d{6}', folder.name):
            print(f"Skipping {folder.name} (already has timestamp)")
            continue

        metadata = get_folder_metadata(folder)
        new_name = generate_new_name(metadata)
        new_path = paper_path / new_name

        # Handle duplicates by appending counter
        if new_path.exists():
            counter = 1
            while True:
                alt_name = f"{new_name}_{counter}"
                alt_path = paper_path / alt_name
                if not alt_path.exists():
                    new_path = alt_path
                    new_name = alt_name
                    break
                counter += 1

        renames.append((folder, new_path))
        print(f"  {folder.name} → {new_name}")

    if not dry_run and renames:
        confirm = input(f"\nRename {len(renames)} folders? (y/n): ")
        if confirm.lower() == 'y':
            for old_path, new_path in renames:
                shutil.move(str(old_path), str(new_path))
                print(f"Renamed: {old_path.name} → {new_path.name}")
            print(f"\nSuccessfully reorganized {len(renames)} folders")
        else:
            print("Operation cancelled")
    elif dry_run:
        print(f"\n[DRY RUN] Would rename {len(renames)} folders")
    else:
        print("\nNo folders to reorganize")


def main():
    parser = argparse.ArgumentParser(description="Reorganize debug output folders")
    parser.add_argument(
        "--paper-dir",
        default="paper",
        help="Path to paper directory containing debug folders"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually doing it"
    )

    args = parser.parse_args()

    reorganize_debug_folders(args.paper_dir, args.dry_run)


if __name__ == "__main__":
    main()