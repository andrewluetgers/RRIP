#!/usr/bin/env python3
"""
Generate residuals for a range of JPEG quality levels.
Uses the wsi_residual_tool_grid.py CLI.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import shutil

# Configuration
PYRAMID_PATH = Path("data/demo_out/baseline_pyramid")
OUTPUT_BASE = Path("evaluation/grid_results")
TILE_SIZE = 256
MAX_PARENTS = 100  # Limit for testing

# Parameter grid
JPEG_QUALITIES = [30, 50, 70, 90]

def run_command(cmd):
    """Run command and return output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def generate_residuals(jpeg_quality):
    """Generate residuals for a specific JPEG quality."""
    config_name = f"j{jpeg_quality}"
    print(f"\n{'='*60}")
    print(f"Generating configuration: {config_name}")
    print(f"  JPEG quality: {jpeg_quality}")
    print(f"{'='*60}")

    # Create output directory for this config
    config_dir = OUTPUT_BASE / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    # Run the encode command
    cmd = [
        "python", "cli/wsi_residual_tool_grid.py", "encode",
        "--pyramid", str(PYRAMID_PATH),
        "--out", str(config_dir),
        "--tile", str(TILE_SIZE),
        "--resq", str(jpeg_quality),
        "--max-parents", str(MAX_PARENTS)
    ]

    start_time = time.time()
    success = run_command(cmd)
    elapsed = time.time() - start_time

    if success:
        # Read the summary file
        summary_path = config_dir / f"summary_j{jpeg_quality}.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

            # Add timing info
            summary["generation_time_seconds"] = elapsed
            summary["config_name"] = config_name

            # Save enhanced summary
            enhanced_summary_path = config_dir / "enhanced_summary.json"
            with open(enhanced_summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"  Compression ratio: {summary.get('compression_ratio', 'N/A'):.2f}x")
            print(f"  Storage savings: {summary.get('savings_pct', 'N/A'):.1f}%")
            print(f"  Generation time: {elapsed:.1f}s")

            return summary
        else:
            print(f"Warning: Summary file not found at {summary_path}")

    return None

def main():
    """Generate all configurations."""
    print("ORIGAMI Parameter Grid Generation")
    print(f"Pyramid: {PYRAMID_PATH}")
    print(f"Output: {OUTPUT_BASE}")

    # Clean and create output directory
    if OUTPUT_BASE.exists():
        print(f"Cleaning existing results at {OUTPUT_BASE}")
        shutil.rmtree(OUTPUT_BASE)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Track all results
    all_results = {}
    successful = 0
    failed = 0

    # Generate all configurations
    for jpeg_q in JPEG_QUALITIES:
        result = generate_residuals(jpeg_q)

        if result:
            config_name = f"j{jpeg_q}"
            all_results[config_name] = result
            successful += 1
        else:
            failed += 1

    total = len(JPEG_QUALITIES)

    # Save aggregated results
    aggregated_path = OUTPUT_BASE / "all_configurations.json"
    with open(aggregated_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"  Successful: {successful}/{total}")
    print(f"  Failed: {failed}/{total}")
    print(f"  Results saved to: {aggregated_path}")

    # Print comparison table
    if all_results:
        print(f"\n{'='*60}")
        print("COMPRESSION COMPARISON")
        print(f"{'Config':<15} {'JPEG':<8} {'Ratio':<10} {'Savings':<10}")
        print("-" * 50)

        for config_name in sorted(all_results.keys()):
            data = all_results[config_name]
            jpeg = data.get('residual_jpeg_quality', 'N/A')
            ratio = data.get('compression_ratio', 0)
            savings = data.get('savings_pct', 0)

            print(f"{config_name:<15} {jpeg:<8} {ratio:<10.2f} {savings:<10.1f}%")

    # Find optimal configurations
    if all_results:
        print(f"\n{'='*60}")
        print("OPTIMAL CONFIGURATIONS")

        # Best compression
        best_compression = max(all_results.items(),
                              key=lambda x: x[1].get('compression_ratio', 0))
        print(f"Best Compression: {best_compression[0]} "
              f"(Ratio: {best_compression[1].get('compression_ratio', 0):.2f}x)")

if __name__ == "__main__":
    main()
