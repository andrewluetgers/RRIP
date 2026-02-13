#!/usr/bin/env python3
"""Compare quality metrics across Rust and Python eval runs.

Reads PSNR and MSE directly from manifests (no reconstruction needed).
Shows L2 tile bytes vs L1/L0 residual bytes separately.

Usage:
    uv run python evals/scripts/compare_quality.py
"""

import json
import pathlib

import numpy as np


RUNS_DIR = pathlib.Path("evals/runs")


def extract_rs_metrics(run_dir):
    """Extract per-tile metrics from a Rust manifest."""
    manifest_path = pathlib.Path(run_dir) / "manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)

    l2_bytes = m.get("l2_bytes", 0)
    l1_bytes = sum(t["residual_bytes"] for t in m["tiles"] if t["level"] == "L1")
    l0_bytes = sum(t["residual_bytes"] for t in m["tiles"] if t["level"] == "L0")

    tiles = {"L1": [], "L0": []}
    for t in m["tiles"]:
        tiles[t["level"]].append({
            "psnr": t["y_psnr_db"],
            "mse": t["y_mse"],
        })

    return {"l2": l2_bytes, "l1": l1_bytes, "l0": l0_bytes}, tiles


def extract_py_metrics(run_dir):
    """Extract per-tile metrics from a Python manifest."""
    manifest_path = pathlib.Path(run_dir) / "manifest.json"
    with open(manifest_path) as f:
        m = json.load(f)

    sc = m["size_comparison"]
    l2_bytes = sc["origami_L2_baseline"]
    l1_bytes = sc["origami_L1_residuals"]
    l0_bytes = sc["origami_L0_residuals"]

    tiles = {"L1": [], "L0": []}
    for level in ["L1", "L0"]:
        phase = m.get("decompression_phase", {}).get(level, {})
        for key, tile_data in phase.items():
            if not key.startswith("tile_"):
                continue
            if "final_psnr" in tile_data:
                tiles[level].append({
                    "psnr": tile_data["final_psnr"],
                    "mse": tile_data["final_mse"],
                })

    return {"l2": l2_bytes, "l1": l1_bytes, "l0": l0_bytes}, tiles


def avg(values):
    return np.mean(values) if values else 0.0


def print_table(runs, label):
    print(f"\n{'='*105}")
    print(f"  {label}")
    print(f"{'='*105}")
    print(f"{'Run':<22} {'L2':>7} {'L1 res':>7} {'L0 res':>7} {'Total':>7}"
          f"  {'L1 PSNR':>7} {'L0 PSNR':>7} {'Avg PSNR':>8}"
          f"  {'L1 MSE':>6} {'L0 MSE':>6} {'Avg MSE':>7}")
    print("-" * 105)

    for name, run_name, run_type in runs:
        run_dir = RUNS_DIR / run_name
        if not run_dir.exists():
            print(f"{name:<22} -- run not found --")
            continue

        if run_type == "py":
            sizes, tiles = extract_py_metrics(run_dir)
        else:
            sizes, tiles = extract_rs_metrics(run_dir)

        total = sizes["l2"] + sizes["l1"] + sizes["l0"]

        l1_psnr = avg([t["psnr"] for t in tiles["L1"]])
        l0_psnr = avg([t["psnr"] for t in tiles["L0"]])
        all_psnr = avg([t["psnr"] for t in tiles["L1"] + tiles["L0"]])

        l1_mse = avg([t["mse"] for t in tiles["L1"]])
        l0_mse = avg([t["mse"] for t in tiles["L0"]])
        all_mse = avg([t["mse"] for t in tiles["L1"] + tiles["L0"]])

        print(f"{name:<22} {sizes['l2']:>7,} {sizes['l1']:>7,} {sizes['l0']:>7,} {total:>7,}"
              f"  {l1_psnr:>7.2f} {l0_psnr:>7.2f} {all_psnr:>8.2f}"
              f"  {l1_mse:>6.2f} {l0_mse:>6.2f} {all_mse:>7.2f}")


def main():
    uniform_runs = [
        ("Py 444",            "py_debug_j40_pac",                        "py"),
        ("Rs 444",            "rs_debug_j40_pac",                        "rs"),
        ("Rs 420",            "rs_debug_j40_s420_pac",                   "rs"),
        ("Rs 420opt",         "rs_debug_j40_s420opt_pac",                "rs"),
        ("Rs 444+optL2",      "rs_debug_j40_optl2_pac",                  "rs"),
        ("Rs 420+optL2",      "rs_debug_j40_s420_optl2_pac",             "rs"),
        ("Rs 420opt+optL2",   "rs_debug_j40_s420opt_optl2_pac",          "rs"),
    ]

    split_runs = [
        ("Py 60/40",          "py_debug_l1q60_l0q40_pac",                "py"),
        ("Rs 444",            "rs_debug_l1q60_l0q40_pac",                "rs"),
        ("Rs 420",            "rs_debug_l1q60_l0q40_s420_pac",           "rs"),
        ("Rs 420opt",         "rs_debug_l1q60_l0q40_s420opt_pac",        "rs"),
        ("Rs 444+optL2",      "rs_debug_l1q60_l0q40_optl2_pac",          "rs"),
        ("Rs 420+optL2",      "rs_debug_l1q60_l0q40_s420_optl2_pac",     "rs"),
        ("Rs 420opt+optL2",   "rs_debug_l1q60_l0q40_s420opt_optl2_pac",  "rs"),
    ]

    print_table(uniform_runs, "Uniform Quality (resq=40)")
    print_table(split_runs, "Split Quality (l1q=60, l0q=40)")
    print()


if __name__ == "__main__":
    main()
