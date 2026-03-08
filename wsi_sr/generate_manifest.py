#!/usr/bin/env python3
"""
Generate a training/evaluation manifest from TCGA slide metadata.

Selects slides per cancer type, assigns train/eval roles, defines tile
sampling strategy. Outputs JSON + CSV manifests that drive all downstream
processing — nothing is downloaded until this manifest is reviewed.

Three evaluation levels:
  Eval-1 (adjacent):     Tiles from training slides at stride offset (tests local generalization)
  Eval-2 (held-out):     All tiles from held-out slides in same cancer type (tests cross-slide)
  Eval-3 (unseen types): All tiles from entirely held-out cancer types (tests cross-morphology)

Usage:
  python generate_manifest.py --metadata tcga_slides_metadata.json
  python generate_manifest.py --metadata tcga_slides_metadata.json \
      --slides-per-type-train 20 --slides-per-type-eval 5 \
      --tile-stride 4 --holdout-types TCGA-UVM,TCGA-CHOL,TCGA-DLBC \
      --diagnostic-only --seed 42
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone


def load_metadata(path: str) -> list:
    """Load tcga_slides_metadata.json and return list of file records."""
    with open(path) as f:
        data = json.load(f)
    return data["files"]


def extract_slide_info(hit: dict) -> dict:
    """Extract a flat record from a GDC API hit."""
    info = {
        "file_id": hit.get("file_id"),
        "file_name": hit.get("file_name"),
        "file_size": hit.get("file_size", 0),
        "experimental_strategy": hit.get("experimental_strategy"),
        "data_format": hit.get("data_format"),
        "access": hit.get("access"),
    }

    cases = hit.get("cases", [])
    if cases:
        case = cases[0]
        proj = case.get("project", {})
        info["project_id"] = proj.get("project_id")
        info["primary_site"] = proj.get("primary_site")
        info["case_id"] = case.get("case_id")
        info["submitter_id"] = case.get("submitter_id")

        samples = case.get("samples", [])
        if samples:
            sample = samples[0]
            info["sample_type"] = sample.get("sample_type")
            info["tissue_type"] = sample.get("tissue_type")
        else:
            info["sample_type"] = None
            info["tissue_type"] = None
    else:
        info["project_id"] = None
        info["primary_site"] = None
        info["case_id"] = None
        info["submitter_id"] = None
        info["sample_type"] = None
        info["tissue_type"] = None

    return info


def select_slides(
    slides: list,
    slides_per_type_train: int,
    slides_per_type_eval: int,
    holdout_types: list,
    diagnostic_only: bool,
    seed: int,
) -> dict:
    """Select slides per cancer type with train/eval/holdout assignments.

    Returns dict keyed by project_id with slide lists and roles.
    """
    rng = random.Random(seed)

    # Group by project_id
    by_project = defaultdict(list)
    for s in slides:
        pid = s.get("project_id")
        if pid is None:
            continue
        if diagnostic_only and s.get("experimental_strategy") != "Diagnostic Slide":
            continue
        by_project[pid].append(s)

    result = {}
    total_train = 0
    total_eval = 0
    total_holdout = 0

    for pid in sorted(by_project.keys()):
        available = by_project[pid]
        rng.shuffle(available)

        is_holdout = pid in holdout_types

        if is_holdout:
            # All slides in holdout types go to L3 eval
            n_eval = min(len(available), slides_per_type_eval + slides_per_type_train)
            eval_slides = available[:n_eval]
            train_slides = []
            role = "holdout"
            total_holdout += n_eval
        else:
            n_available = len(available)
            if n_available < 5:
                # Very small type: use 60/40 split
                n_train = max(1, int(n_available * 0.6))
                n_eval = n_available - n_train
            else:
                n_train = min(slides_per_type_train, n_available - slides_per_type_eval)
                n_eval = min(slides_per_type_eval, n_available - n_train)
                # Ensure we have at least 1 eval
                if n_eval < 1 and n_available > 1:
                    n_eval = 1
                    n_train = n_available - 1

            train_slides = available[:n_train]
            eval_slides = available[n_train:n_train + n_eval]
            role = "train+eval"
            total_train += n_train
            total_eval += n_eval

        result[pid] = {
            "project_id": pid,
            "primary_site": available[0].get("primary_site") if available else None,
            "total_available": len(by_project[pid]),
            "diagnostic_available": len(available),
            "train_slides": train_slides,
            "eval_slides": eval_slides,
            "role": role,
        }

    return result, total_train, total_eval, total_holdout


def estimate_tiles(file_size_bytes: int, tile_stride: int) -> dict:
    """Estimate tile counts for a slide based on file size.

    Conservative heuristic: larger SVS files have more tissue.
    At 1024x1024, typical diagnostic slides have 200-2000 tiles.
    """
    size_mb = file_size_bytes / 1e6

    # Rough model: tiles ≈ file_size_mb * 1.5 (based on TCGA averages)
    # Median SVS = 269 MB → ~400 tiles, Mean SVS = 569 MB → ~850 tiles
    estimated_total = max(50, int(size_mb * 1.5))

    # Cap at 3000 (very large slides)
    estimated_total = min(estimated_total, 3000)

    # Apply blank filtering (~15% are blank based on our DICOM experience)
    estimated_tissue = int(estimated_total * 0.85)

    # Apply stride
    train_tiles = estimated_tissue // tile_stride
    eval_adjacent_tiles = estimated_tissue // tile_stride  # stride offset 1

    return {
        "estimated_total": estimated_total,
        "estimated_tissue": estimated_tissue,
        "train_tiles": train_tiles,
        "eval_adjacent_tiles": eval_adjacent_tiles,
    }


def build_manifest(
    selection: dict,
    tile_stride: int,
    holdout_types: list,
    config: dict,
) -> dict:
    """Build the full manifest with tile estimates."""

    cancer_types = []
    all_slides = []
    totals = {
        "train_slides": 0,
        "eval_slides": 0,
        "holdout_slides": 0,
        "train_tiles": 0,
        "eval_adjacent_tiles": 0,
        "eval_holdout_tiles": 0,
        "download_bytes": 0,
    }

    for pid in sorted(selection.keys()):
        entry = selection[pid]
        ct_info = {
            "project_id": pid,
            "primary_site": entry["primary_site"],
            "total_available": entry["total_available"],
            "diagnostic_available": entry["diagnostic_available"],
            "train_slide_count": len(entry["train_slides"]),
            "eval_slide_count": len(entry["eval_slides"]),
            "role": entry["role"],
            "train_file_ids": [s["file_id"] for s in entry["train_slides"]],
            "eval_file_ids": [s["file_id"] for s in entry["eval_slides"]],
        }

        # Estimate tiles for this cancer type
        ct_train_tiles = 0
        ct_eval_adj_tiles = 0
        ct_eval_holdout_tiles = 0

        for s in entry["train_slides"]:
            te = estimate_tiles(s["file_size"], tile_stride)
            slide_record = {
                "file_id": s["file_id"],
                "file_name": s["file_name"],
                "file_size": s["file_size"],
                "project_id": pid,
                "primary_site": entry["primary_site"],
                "sample_type": s.get("sample_type"),
                "role": "train",
                "tile_stride": tile_stride,
                "estimated_train_tiles": te["train_tiles"],
                "estimated_eval_adjacent_tiles": te["eval_adjacent_tiles"],
            }
            all_slides.append(slide_record)
            ct_train_tiles += te["train_tiles"]
            ct_eval_adj_tiles += te["eval_adjacent_tiles"]
            totals["download_bytes"] += s["file_size"]

        for s in entry["eval_slides"]:
            te = estimate_tiles(s["file_size"], tile_stride)
            if entry["role"] == "holdout":
                slide_role = "holdout"
                # Holdout slides: extract all tissue tiles (stride=1) for thorough eval
                est_tiles = te["estimated_tissue"]
                ct_eval_holdout_tiles += est_tiles
            else:
                slide_role = "eval"
                # Held-out slides in train types: also extract all tissue tiles
                est_tiles = te["estimated_tissue"]
                ct_eval_holdout_tiles += est_tiles

            slide_record = {
                "file_id": s["file_id"],
                "file_name": s["file_name"],
                "file_size": s["file_size"],
                "project_id": pid,
                "primary_site": entry["primary_site"],
                "sample_type": s.get("sample_type"),
                "role": slide_role,
                "tile_stride": 1,  # eval slides get all tiles
                "estimated_eval_tiles": est_tiles,
            }
            all_slides.append(slide_record)
            totals["download_bytes"] += s["file_size"]

        ct_info["estimated_train_tiles"] = ct_train_tiles
        ct_info["estimated_eval_adjacent_tiles"] = ct_eval_adj_tiles
        ct_info["estimated_eval_holdout_tiles"] = ct_eval_holdout_tiles
        cancer_types.append(ct_info)

        totals["train_slides"] += len(entry["train_slides"])
        if entry["role"] == "holdout":
            totals["holdout_slides"] += len(entry["eval_slides"])
        else:
            totals["eval_slides"] += len(entry["eval_slides"])
        totals["train_tiles"] += ct_train_tiles
        totals["eval_adjacent_tiles"] += ct_eval_adj_tiles
        totals["eval_holdout_tiles"] += ct_eval_holdout_tiles

    # Estimate storage
    avg_tile_bytes_jpeg = 150_000  # 150 KB per 1024x1024 JPEG q95 (conservative)
    total_tiles = (totals["train_tiles"] + totals["eval_adjacent_tiles"]
                   + totals["eval_holdout_tiles"])
    estimated_tile_storage = total_tiles * avg_tile_bytes_jpeg

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": {
            "total_slides": totals["train_slides"] + totals["eval_slides"] + totals["holdout_slides"],
            "train_slides": totals["train_slides"],
            "eval_slides": totals["eval_slides"],
            "holdout_slides": totals["holdout_slides"],
            "cancer_types_covered": len([ct for ct in cancer_types if ct["role"] == "train+eval"]),
            "cancer_types_holdout": len([ct for ct in cancer_types if ct["role"] == "holdout"]),
            "tissue_sites_covered": len(set(ct["primary_site"] for ct in cancer_types if ct["primary_site"])),
            "estimated_train_tiles": totals["train_tiles"],
            "estimated_eval_adjacent_tiles": totals["eval_adjacent_tiles"],
            "estimated_eval_holdout_tiles": totals["eval_holdout_tiles"],
            "estimated_total_tiles": total_tiles,
            "estimated_download_gb": round(totals["download_bytes"] / 1e9, 1),
            "estimated_tile_storage_gb": round(estimated_tile_storage / 1e9, 1),
        },
        "cancer_types": cancer_types,
        "slides": all_slides,
    }

    return manifest


def write_csv(manifest: dict, csv_path: str):
    """Write a flat CSV for quick inspection."""
    fieldnames = [
        "file_id", "file_name", "project_id", "primary_site",
        "sample_type", "role", "tile_stride", "file_size_mb",
        "estimated_train_tiles", "estimated_eval_adjacent_tiles",
        "estimated_eval_tiles",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for slide in manifest["slides"]:
            row = dict(slide)
            row["file_size_mb"] = round(slide["file_size"] / 1e6, 1)
            writer.writerow(row)


def print_summary(manifest: dict):
    """Print a human-readable summary."""
    s = manifest["summary"]
    c = manifest["config"]

    print("\n" + "=" * 70)
    print("TCGA TRAINING MANIFEST SUMMARY")
    print("=" * 70)

    print(f"\nConfig:")
    print(f"  Slides per type (train): {c['slides_per_type_train']}")
    print(f"  Slides per type (eval):  {c['slides_per_type_eval']}")
    print(f"  Tile stride:             {c['tile_stride']}")
    print(f"  Holdout cancer types:    {', '.join(c['holdout_types'])}")
    print(f"  Diagnostic only:         {c['diagnostic_only']}")
    print(f"  Random seed:             {c['seed']}")

    print(f"\nSlides:")
    print(f"  Total:    {s['total_slides']}")
    print(f"  Train:    {s['train_slides']}")
    print(f"  Eval:     {s['eval_slides']} (held-out slides in training cancer types)")
    print(f"  Holdout:  {s['holdout_slides']} (entirely held-out cancer types)")

    print(f"\nCoverage:")
    print(f"  Cancer types (train+eval): {s['cancer_types_covered']}")
    print(f"  Cancer types (holdout):    {s['cancer_types_holdout']}")
    print(f"  Tissue sites:              {s['tissue_sites_covered']}")

    print(f"\nEstimated tiles:")
    print(f"  Train tiles:          {s['estimated_train_tiles']:>8,}")
    print(f"  Eval adjacent tiles:  {s['estimated_eval_adjacent_tiles']:>8,}")
    print(f"  Eval holdout tiles:   {s['estimated_eval_holdout_tiles']:>8,}")
    print(f"  Total tiles:          {s['estimated_total_tiles']:>8,}")

    print(f"\nEstimated storage:")
    print(f"  SVS download:  {s['estimated_download_gb']:.1f} GB")
    print(f"  Tile storage:  {s['estimated_tile_storage_gb']:.1f} GB (JPEG q95)")

    print(f"\nPer cancer type:")
    print(f"  {'Project':<15} {'Site':<20} {'Role':<12} {'Train':<6} {'Eval':<6} "
          f"{'TrainTiles':<11} {'EvalAdj':<9} {'EvalHold':<9}")
    print(f"  {'-'*15} {'-'*20} {'-'*12} {'-'*6} {'-'*6} {'-'*11} {'-'*9} {'-'*9}")

    for ct in manifest["cancer_types"]:
        print(f"  {ct['project_id']:<15} {(ct['primary_site'] or '?'):<20} {ct['role']:<12} "
              f"{ct['train_slide_count']:<6} {ct['eval_slide_count']:<6} "
              f"{ct.get('estimated_train_tiles', 0):<11,} "
              f"{ct.get('estimated_eval_adjacent_tiles', 0):<9,} "
              f"{ct.get('estimated_eval_holdout_tiles', 0):<9,}")

    print("\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser(
        description="Generate TCGA training/evaluation manifest")
    ap.add_argument("--metadata", required=True,
                    help="Path to tcga_slides_metadata.json")
    ap.add_argument("--slides-per-type-train", type=int, default=20,
                    help="Max training slides per cancer type (default: 20)")
    ap.add_argument("--slides-per-type-eval", type=int, default=5,
                    help="Max eval slides per cancer type (default: 5)")
    ap.add_argument("--tile-stride", type=int, default=4,
                    help="Extract every Nth tile for training (default: 4)")
    ap.add_argument("--holdout-types", default="TCGA-UVM,TCGA-CHOL,TCGA-DLBC",
                    help="Cancer types to hold out entirely (comma-separated)")
    ap.add_argument("--types", default=None,
                    help="Only include these cancer types (comma-separated, e.g. TCGA-BRCA,TCGA-GBM). "
                         "Default: all types. Useful for Stage 1 small runs.")
    ap.add_argument("--diagnostic-only", action="store_true", default=True,
                    help="Use only diagnostic (FFPE) slides (default: True)")
    ap.add_argument("--include-frozen", action="store_true",
                    help="Include tissue (frozen) slides too")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for slide selection (default: 42)")
    ap.add_argument("--output", "-o", default=None,
                    help="Output JSON path (default: tcga_training_manifest.json)")
    args = ap.parse_args()

    if args.include_frozen:
        args.diagnostic_only = False

    holdout_types = [t.strip() for t in args.holdout_types.split(",") if t.strip()]
    include_types = None
    if args.types:
        include_types = set(t.strip() for t in args.types.split(",") if t.strip())
        # Remove holdout types from include list — they're separate
        holdout_types = [t for t in holdout_types if t in include_types]

    # Default output path
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_json = os.path.join(script_dir, "tcga_training_manifest.json")
    else:
        output_json = args.output

    output_csv = output_json.replace(".json", ".csv")

    # Load metadata
    print(f"Loading metadata from: {args.metadata}")
    raw_hits = load_metadata(args.metadata)
    print(f"  {len(raw_hits)} total slide records")

    # Extract flat slide info
    slides = [extract_slide_info(h) for h in raw_hits]

    # Filter to valid records
    slides = [s for s in slides if s["file_id"] and s["project_id"]]
    print(f"  {len(slides)} valid slide records")

    if args.diagnostic_only:
        diag_count = sum(1 for s in slides if s["experimental_strategy"] == "Diagnostic Slide")
        print(f"  {diag_count} diagnostic slides (filtering to these)")

    # Filter to specific cancer types if requested (Stage 1 small runs)
    if include_types:
        before = len(slides)
        slides = [s for s in slides if s["project_id"] in include_types]
        print(f"  Filtered to {len(include_types)} types: {len(slides)} slides (from {before})")

    # Select slides
    print(f"\nSelecting slides (train={args.slides_per_type_train}, "
          f"eval={args.slides_per_type_eval}, stride={args.tile_stride})...")
    print(f"Holdout types: {holdout_types}")

    selection, n_train, n_eval, n_holdout = select_slides(
        slides,
        slides_per_type_train=args.slides_per_type_train,
        slides_per_type_eval=args.slides_per_type_eval,
        holdout_types=holdout_types,
        diagnostic_only=args.diagnostic_only,
        seed=args.seed,
    )

    config = {
        "slides_per_type_train": args.slides_per_type_train,
        "slides_per_type_eval": args.slides_per_type_eval,
        "tile_stride": args.tile_stride,
        "holdout_types": holdout_types,
        "diagnostic_only": args.diagnostic_only,
        "seed": args.seed,
        "tile_size": 1024,
        "tile_format": "jpeg",
        "tile_quality": 95,
    }

    # Build manifest
    manifest = build_manifest(selection, args.tile_stride, holdout_types, config)

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest JSON: {output_json}")
    print(f"  Size: {os.path.getsize(output_json) / 1e6:.1f} MB")

    # Save CSV
    write_csv(manifest, output_csv)
    print(f"Saved manifest CSV:  {output_csv}")

    # Print summary
    print_summary(manifest)


if __name__ == "__main__":
    main()
