#!/usr/bin/env python3
"""
Seed the DuckDB database with all plan data.

Loads:
  1. Pipeline stages (with time/cost estimates)
  2. All 30K TCGA slide metadata
  3. Training manifest (800 slides, train/eval/holdout assignments)

Usage:
  python load_plan.py
  python load_plan.py --db custom_path.duckdb
"""

import argparse
import os
import sys

# Ensure wsi_sr/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import WSISRDB


def main():
    ap = argparse.ArgumentParser(description="Load plan data into DuckDB")
    ap.add_argument("--db", default="wsi_sr.duckdb", help="DuckDB path")
    ap.add_argument("--metadata", default=None,
                    help="TCGA slides metadata JSON (default: tcga_slides_metadata.json)")
    ap.add_argument("--manifest", default=None,
                    help="Training manifest JSON (default: tcga_training_manifest.json)")
    ap.add_argument("--manifest-id", default="tcga_v1",
                    help="Manifest ID (default: tcga_v1)")
    ap.add_argument("--stage", type=int, default=2, choices=[0, 1, 2],
                    help="Execution stage: 0=local smoke, 1=small TCGA, 2=full TCGA")
    args = ap.parse_args()

    # Default paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = args.metadata or os.path.join(script_dir, "tcga_slides_metadata.json")
    manifest_path = args.manifest or os.path.join(script_dir, "tcga_training_manifest.json")

    db = WSISRDB(args.db)

    # 1. Pipeline stages
    print(f"=== Pipeline Stages (stage {args.stage}) ===")
    db.init_stages(execution_stage=args.stage)
    stages = db.get_stages()
    for s in stages:
        print(f"  {s['stage_id']:>10}: {s['name']} — est ${s.get('estimated_cost_usd', 0):.2f}")

    # 2. Slide metadata
    if os.path.exists(metadata_path):
        print(f"\n=== Slide Metadata ===")
        db.load_slide_metadata(metadata_path)
        summary = db.dataset_summary()
        print(f"  Total slides: {summary['total_slides']}")
        print(f"  By strategy: {summary['by_strategy']}")
        print(f"  Top 5 projects:")
        for p in summary["by_project"][:5]:
            print(f"    {p['project_id']:>12}: {p['n']} slides, {p['total_gb']:.1f} GB")
    else:
        print(f"\n  Skipping metadata (not found: {metadata_path})")

    # 3. Manifest
    if os.path.exists(manifest_path):
        print(f"\n=== Training Manifest ===")
        mid = db.load_manifest(manifest_path, manifest_id=args.manifest_id)

        # Mark manifest stage as completed
        db.update_stage("manifest", status="completed")

        ms = db.manifest_summary(mid)
        print(f"  Slides by role:")
        for r in ms.get("slides_by_role", []):
            print(f"    {r['role']:>10}: {r['n']} slides, {r.get('total_gb', 0):.1f} GB, "
                  f"~{r.get('est_train_tiles') or 0} train tiles")
        print(f"  Cancer types: {len(ms.get('cancer_types', []))}")
    else:
        print(f"\n  Skipping manifest (not found: {manifest_path})")

    # Summary
    print(f"\n=== Database ===")
    print(f"  Path: {os.path.abspath(args.db)}")
    for table in ["slides", "manifests", "manifest_slides", "manifest_cancer_types", "stages"]:
        n = db.query(f"SELECT COUNT(*) as n FROM {table}")[0]["n"]
        print(f"  {table}: {n} rows")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
