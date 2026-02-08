#!/usr/bin/env python3
"""
Convert old capture manifests to new format for heat map generation.
"""

import json
import pathlib
import argparse
import glob


def convert_old_manifest(old_manifest_path):
    """Convert old manifest.json to analysis_manifest.json format."""

    with open(old_manifest_path) as f:
        old_data = json.load(f)

    # Extract configuration
    config = old_data.get("configuration", {})
    quantization = config.get("quantization_levels", 32)
    jpeg_quality = config.get("residual_jpeg_quality", 30)

    # Initialize new format
    new_data = {
        "configuration": config,
        "statistics": {},
        "compression_phase": old_data.get("compression_phase", {}),
        "decompression_phase": old_data.get("decompression_phase", {})
    }

    # Use size_comparison section if available
    if "size_comparison" in old_data:
        sc = old_data["size_comparison"]
        compression_ratio = sc.get("overall_compression_ratio", 1.0)
        savings_pct = sc.get("overall_space_savings_pct", 0.0)
        baseline_bytes = sc.get("baseline_total", 0)
        origami_bytes = sc.get("origami_total", 0)

        new_data["statistics"] = {
            "compression_ratio": compression_ratio,
            "savings_pct": savings_pct,
            "baseline_bytes_all_levels": baseline_bytes,
            "retained_bytes_L2plus": sc.get("origami_L2_baseline", 0),
            "residual_bytes_L1": sc.get("origami_L1_residuals", 0),
            "residual_bytes_L0": sc.get("origami_L0_residuals", 0),
            "proposed_bytes": origami_bytes
        }
    else:
        # Fallback - try to calculate from file sizes
        baseline_bytes = 0
        residual_bytes = 0

        # Calculate stats
        total_bytes = baseline_bytes + residual_bytes
        if total_bytes > 0 and baseline_bytes > 0:
            compression_ratio = baseline_bytes / total_bytes
            savings_pct = (1 - total_bytes / baseline_bytes) * 100
        else:
            compression_ratio = 1.0
            savings_pct = 0.0

        new_data["statistics"] = {
            "compression_ratio": compression_ratio,
            "savings_pct": savings_pct,
            "baseline_bytes": baseline_bytes,
            "residual_bytes": residual_bytes,
            "total_bytes": total_bytes
        }

    # Try to extract PSNR/SSIM from existing data
    if "compression_phase" in old_data:
        l0_psnrs = []
        l0_ssims = []

        for key, info in old_data["compression_phase"].items():
            if key.startswith("L0_") and "prediction_metrics" in info:
                m = info["prediction_metrics"]
                if "psnr" in m:
                    l0_psnrs.append(m["psnr"])
                if "ssim" in m:
                    l0_ssims.append(m["ssim"])

        # Add averaged metrics to compression phase for analyzer
        if l0_psnrs:
            import numpy as np
            for key in list(old_data["compression_phase"].keys()):
                if key.startswith("L0_"):
                    if "prediction_metrics" not in old_data["compression_phase"][key]:
                        old_data["compression_phase"][key]["prediction_metrics"] = {}
                    # Keep existing metrics

    # Save analysis manifest
    output_path = old_manifest_path.parent / "analysis_manifest.json"
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert old capture manifests")
    parser.add_argument("--capture-pattern", default="paper/debug_*",
                       help="Pattern to match capture directories")

    args = parser.parse_args()

    # Find all capture directories
    capture_dirs = glob.glob(args.capture_pattern)
    converted = 0

    for capture_dir in capture_dirs:
        capture_path = pathlib.Path(capture_dir)
        old_manifest = capture_path / "manifest.json"
        new_manifest = capture_path / "analysis_manifest.json"

        if old_manifest.exists() and not new_manifest.exists():
            print(f"Converting {capture_dir}...")
            try:
                convert_old_manifest(old_manifest)
                converted += 1
            except Exception as e:
                print(f"  Error: {e}")

    print(f"Converted {converted} captures")


if __name__ == "__main__":
    main()