#!/usr/bin/env python3
"""Generate clean metric-vs-size charts for the ORIGAMI homepage.

Four series:
  - JPEG: baseline Q30..Q90, labeled "Q30".."Q90"
  - JPEG XL: baseline Q30..Q90, labeled "Q30".."Q90"
  - JPEG 2000: baseline Q30..Q90, labeled "Q30".."Q90"
  - ORIGAMI: B90 444 optL2 +-20, split 30/10..90/90, labeled "30/10".."90/90"

Charts: PSNR, SSIM, VIF, Delta E, LPIPS — all vs family size (KB).

Usage:
    uv run python evals/scripts/generate_homepage_charts.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "charts" / "homepage"

# JPEG baseline: Q30..Q90 step 10
JPEG_QUALITIES = list(range(30, 100, 10))

# ORIGAMI B90 +20 split: l1q/l0q pairs
ORIGAMI_SPLITS = [
    (30, 10), (40, 20), (50, 30), (60, 40), (70, 50), (80, 60), (90, 70),
    (90, 80), (90, 90),
]

METRICS = {
    "psnr":    {"label": "PSNR (dB)",          "higher_better": True},
    "ssim":    {"label": "SSIM",               "higher_better": True},
    "vif":     {"label": "VIF",                "higher_better": True},
    "delta_e": {"label": "Delta E (CIEDE2000)","higher_better": False},
    "lpips":   {"label": "LPIPS",              "higher_better": False},
}

JPEG_STYLE = {"color": "#e74c3c", "marker": "s", "linestyle": "--", "markersize": 8}
JXL_STYLE = {"color": "#9b59b6", "marker": "^", "linestyle": "--", "markersize": 8}
JP2_STYLE = {"color": "#e67e22", "marker": "D", "linestyle": "--", "markersize": 7}
ORIGAMI_STYLE = {"color": "#2ecc71", "marker": "o", "linestyle": "-", "markersize": 8}


def load_baseline_metrics(run_dir):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    m = json.loads(manifest_path.read_text())
    if "tiles" not in m:
        return None
    metrics = {}
    for mk in METRICS:
        values = [t[mk] for t in m["tiles"].values() if mk in t]
        if values:
            metrics[mk] = np.mean(values)
    metrics["total_size_bytes"] = sum(t["size_bytes"] for t in m["tiles"].values())
    return metrics


def load_origami_metrics(run_dir):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    m = json.loads(manifest_path.read_text())
    all_metrics = {k: [] for k in METRICS}
    for level_key in ["L1", "L0"]:
        level = m.get("decompression_phase", {}).get(level_key, {})
        for tile_key, tile_data in level.items():
            if not tile_key.startswith("tile_"):
                continue
            for mk in METRICS:
                val = tile_data.get(f"final_{mk}")
                if val is not None:
                    all_metrics[mk].append(val)
    metrics = {}
    for mk, values in all_metrics.items():
        if values:
            metrics[mk] = np.mean(values)
    sc = m.get("size_comparison", {})
    metrics["total_size_bytes"] = sc.get("origami_total", 0)
    return metrics


def collect():
    jpeg = {"labels": [], "sizes": [], "metrics": {k: [] for k in METRICS}}
    for q in JPEG_QUALITIES:
        run_dir = RUNS_DIR / f"jpeg_baseline_q{q}"
        m = load_baseline_metrics(run_dir)
        if m:
            jpeg["labels"].append(f"Q{q}")
            jpeg["sizes"].append(m["total_size_bytes"])
            for mk in METRICS:
                jpeg["metrics"][mk].append(m.get(mk))

    jxl = {"labels": [], "sizes": [], "metrics": {k: [] for k in METRICS}}
    for q in JPEG_QUALITIES:
        run_dir = RUNS_DIR / f"jpegxl_jpeg_baseline_q{q}"
        m = load_baseline_metrics(run_dir)
        if m:
            jxl["labels"].append(f"Q{q}")
            jxl["sizes"].append(m["total_size_bytes"])
            for mk in METRICS:
                jxl["metrics"][mk].append(m.get(mk))

    jp2 = {"labels": [], "sizes": [], "metrics": {k: [] for k in METRICS}}
    for q in JPEG_QUALITIES:
        run_dir = RUNS_DIR / f"jp2_baseline_q{q}"
        m = load_baseline_metrics(run_dir)
        if m:
            jp2["labels"].append(f"Q{q}")
            jp2["sizes"].append(m["total_size_bytes"])
            for mk in METRICS:
                jp2["metrics"][mk].append(m.get(mk))

    origami = {"labels": [], "sizes": [], "metrics": {k: [] for k in METRICS}}
    for l1q, l0q in ORIGAMI_SPLITS:
        run_dir = RUNS_DIR / f"rs_444_b90_optl2_d20_l1q{l1q}_l0q{l0q}"
        m = load_origami_metrics(run_dir)
        if m:
            origami["labels"].append(f"{l1q}/{l0q}")
            origami["sizes"].append(m["total_size_bytes"])
            for mk in METRICS:
                origami["metrics"][mk].append(m.get(mk))

    return jpeg, jxl, jp2, origami


def make_chart(jpeg, jxl, jp2, origami, metric_key, metric_info, output_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # JPEG
    sizes_kb = [s / 1024 for s in jpeg["sizes"]]
    vals = jpeg["metrics"][metric_key]
    if sizes_kb and vals:
        ax.plot(sizes_kb, vals, label="JPEG", linewidth=2.5, **JPEG_STYLE)
        for lbl, x, y in zip(jpeg["labels"], sizes_kb, vals):
            if y is None:
                continue
            ax.annotate(lbl, (x, y), fontsize=8, fontweight="600",
                        color=JPEG_STYLE["color"], alpha=0.85,
                        textcoords="offset points", xytext=(6, -12))

    # JPEG XL
    sizes_kb = [s / 1024 for s in jxl["sizes"]]
    vals = jxl["metrics"][metric_key]
    if sizes_kb and vals:
        ax.plot(sizes_kb, vals, label="JPEG XL", linewidth=2.5, **JXL_STYLE)
        for lbl, x, y in zip(jxl["labels"], sizes_kb, vals):
            if y is None:
                continue
            ax.annotate(lbl, (x, y), fontsize=8, fontweight="600",
                        color=JXL_STYLE["color"], alpha=0.85,
                        textcoords="offset points", xytext=(6, -12))

    # JPEG 2000
    sizes_kb = [s / 1024 for s in jp2["sizes"]]
    vals = jp2["metrics"][metric_key]
    if sizes_kb and vals:
        ax.plot(sizes_kb, vals, label="JPEG 2000", linewidth=2.5, **JP2_STYLE)
        for lbl, x, y in zip(jp2["labels"], sizes_kb, vals):
            if y is None:
                continue
            ax.annotate(lbl, (x, y), fontsize=8, fontweight="600",
                        color=JP2_STYLE["color"], alpha=0.85,
                        textcoords="offset points", xytext=(6, -12))

    # ORIGAMI
    sizes_kb = [s / 1024 for s in origami["sizes"]]
    vals = origami["metrics"][metric_key]
    if sizes_kb and vals:
        ax.plot(sizes_kb, vals, label="ORIGAMI", linewidth=2.5, **ORIGAMI_STYLE)
        for lbl, x, y in zip(origami["labels"], sizes_kb, vals):
            if y is None:
                continue
            ax.annotate(lbl, (x, y), fontsize=8, fontweight="600",
                        color=ORIGAMI_STYLE["color"], alpha=0.85,
                        textcoords="offset points", xytext=(6, 5))

    arrow = "\u2191" if metric_info["higher_better"] else "\u2193"
    ax.set_xlabel("Family Size (KB)", fontsize=12)
    ax.set_ylabel(f'{metric_info["label"]} {arrow}', fontsize=12)
    ax.set_title(f'{metric_info["label"]} vs Family Size', fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {output_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting data...")
    jpeg, jxl, jp2, origami = collect()
    print(f"  JPEG: {len(jpeg['labels'])} points ({', '.join(jpeg['labels'])})")
    print(f"  JPEG XL: {len(jxl['labels'])} points ({', '.join(jxl['labels'])})")
    print(f"  JPEG 2000: {len(jp2['labels'])} points ({', '.join(jp2['labels'])})")
    print(f"  ORIGAMI: {len(origami['labels'])} points ({', '.join(origami['labels'])})")

    print("Generating charts...")
    for mk, info in METRICS.items():
        make_chart(jpeg, jxl, jp2, origami, mk, info, OUTPUT_DIR / f"{mk}_vs_size.png")

    print(f"Done. Charts in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
