#!/usr/bin/env python3
"""Generate metric charts comparing ORIGAMI and JPEG baseline encoders.

Reads manifest.json from each run in evals/runs/ and produces charts showing
how metrics vary across quality levels (30–90) for all encoder backends.

Usage:
    uv run python evals/scripts/generate_charts.py

Output directory: evals/charts/

Charts generated:
  Metric vs Quality (5):
    - psnr_vs_quality.png, ssim_vs_quality.png, vif_vs_quality.png,
      delta_e_vs_quality.png, mse_vs_quality.png

  Size vs Quality (2):
    - size_vs_quality.png    — raw tile bytes summed
    - pack_size_vs_quality.png — LZ4-packed family size

  Rate-Distortion — metric vs raw size (2):
    - psnr_vs_size.png, ssim_vs_size.png

  Rate-Distortion — metric vs pack size (5):
    - psnr_vs_pack_size.png, ssim_vs_pack_size.png, vif_vs_pack_size.png,
      delta_e_vs_pack_size.png, mse_vs_pack_size.png

Series style:
  - Dashed lines + square markers = JPEG baselines
  - Solid lines + circle markers = ORIGAMI residual encoding
  - Blue = libjpeg-turbo, Orange = mozjpeg, Green = JPEG XL
  - Arrows (↑/↓) on axis labels indicate which direction is better

Prerequisites:
  - Run data must exist in evals/runs/ (generate with jpeg_baseline.py
    and wsi_residual_debug_with_manifest.py)
  - Requires matplotlib and numpy (available via uv)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "charts"

QUALITIES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Run directory patterns
BASELINE_DIRS = {
    "JPEG turbo": "jpeg_baseline_q{q}",
    "JPEG mozjpeg": "mozjpeg_jpeg_baseline_q{q}",
    "JPEG jpegxl": "jpegxl_jpeg_baseline_q{q}",
}

ORIGAMI_DIRS = {
    "ORIGAMI turbo": "debug_j{q}_pac",
    "ORIGAMI mozjpeg": "mozjpeg_debug_j{q}_pac",
    "ORIGAMI jpegxl": "jpegxl_debug_j{q}_pac",
}

# Colors and markers for each series
STYLES = {
    "JPEG turbo":     {"color": "#1f77b4", "marker": "s", "linestyle": "--"},
    "JPEG mozjpeg":   {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "JPEG jpegxl":    {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
    "ORIGAMI turbo":  {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "ORIGAMI mozjpeg":{"color": "#ff7f0e", "marker": "o", "linestyle": "-"},
    "ORIGAMI jpegxl": {"color": "#2ca02c", "marker": "o", "linestyle": "-"},
}

METRICS = {
    "psnr":    {"label": "PSNR (dB)",           "higher_better": True,  "arrow": "\u2191"},
    "ssim":    {"label": "SSIM",                 "higher_better": True,  "arrow": "\u2191"},
    "vif":     {"label": "VIF",                  "higher_better": True,  "arrow": "\u2191"},
    "delta_e": {"label": "Delta E (CIEDE2000)",  "higher_better": False, "arrow": "\u2193"},
    "mse":     {"label": "MSE",                  "higher_better": False, "arrow": "\u2193"},
    "lpips":   {"label": "LPIPS",                "higher_better": False, "arrow": "\u2193"},
}



def _filter_data(data, min_quality):
    """Return a copy of data filtered to only qualities >= min_quality."""
    filtered = {}
    for series_name, series in data.items():
        indices = [i for i, q in enumerate(series["qualities"]) if q >= min_quality]
        filtered[series_name] = {
            "qualities": [series["qualities"][i] for i in indices],
            "metrics": {k: [v[i] for i in indices] for k, v in series["metrics"].items()},
            "sizes": [series["sizes"][i] for i in indices],
            "pack_sizes": [series["pack_sizes"][i] for i in indices],
        }
    return filtered


def _annotate_quality_labels(ax, data, x_source, metric_key):
    """Annotate quality labels on JXL series dots for KB-based x-axis charts."""
    for series_name, series in data.items():
        if "jpegxl" not in series_name.lower():
            continue
        qs = series["qualities"]
        xs = [s / 1024 for s in series[x_source]]
        vals = series["metrics"][metric_key]
        if not qs or not xs or not vals:
            continue
        style = STYLES[series_name]
        for q, x, y in zip(qs, xs, vals):
            if y is None:
                continue
            ax.annotate(f"{q}", (x, y),
                        fontsize=7, fontweight="bold",
                        color=style["color"], alpha=0.8,
                        textcoords="offset points", xytext=(5, 3))



def load_baseline_metrics(run_dir):
    """Extract per-tile metrics from a baseline manifest, return averages."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    m = json.loads(manifest_path.read_text())
    if "tiles" not in m:
        return None

    metrics = {}
    for metric_key in METRICS:
        values = [t[metric_key] for t in m["tiles"].values() if metric_key in t]
        if values:
            metrics[metric_key] = np.mean(values)

    # Total size
    metrics["total_size_bytes"] = sum(t["size_bytes"] for t in m["tiles"].values())

    # Pack size (LZ4 compressed family)
    pack = m.get("pack", {})
    metrics["pack_size_bytes"] = pack.get("size", metrics["total_size_bytes"])
    return metrics


def load_origami_metrics(run_dir):
    """Extract per-tile metrics from an ORIGAMI manifest, return averages."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    m = json.loads(manifest_path.read_text())

    # Collect final_* metrics from decompression_phase L0 and L1 tiles
    all_metrics = {k: [] for k in METRICS}

    for level_key in ["L1", "L0"]:
        level = m.get("decompression_phase", {}).get(level_key, {})
        for tile_key, tile_data in level.items():
            if not tile_key.startswith("tile_"):
                continue
            for metric_key in METRICS:
                val = tile_data.get(f"final_{metric_key}")
                if val is not None:
                    all_metrics[metric_key].append(val)

    metrics = {}
    for metric_key, values in all_metrics.items():
        if values:
            metrics[metric_key] = np.mean(values)

    # Size from size_comparison
    sc = m.get("size_comparison", {})
    metrics["total_size_bytes"] = sc.get("origami_total", 0)
    metrics["compression_ratio"] = sc.get("overall_compression_ratio", 0)
    metrics["space_savings_pct"] = sc.get("overall_space_savings_pct", 0)

    # Pack size (pac file)
    pac = m.get("pac_file", {})
    metrics["pack_size_bytes"] = pac.get("size", metrics["total_size_bytes"])

    return metrics


def collect_all_data():
    """Collect metric data for all series across all quality levels."""
    data = {}

    # Baselines
    for series_name, dir_pattern in BASELINE_DIRS.items():
        series = {"qualities": [], "metrics": {k: [] for k in METRICS}, "sizes": [], "pack_sizes": []}
        for q in QUALITIES:
            run_dir = RUNS_DIR / dir_pattern.format(q=q)
            m = load_baseline_metrics(run_dir)
            if m:
                series["qualities"].append(q)
                for metric_key in METRICS:
                    series["metrics"][metric_key].append(m.get(metric_key))
                series["sizes"].append(m.get("total_size_bytes", 0))
                series["pack_sizes"].append(m.get("pack_size_bytes", 0))
        data[series_name] = series

    # ORIGAMI
    for series_name, dir_pattern in ORIGAMI_DIRS.items():
        series = {"qualities": [], "metrics": {k: [] for k in METRICS}, "sizes": [], "pack_sizes": []}
        for q in QUALITIES:
            run_dir = RUNS_DIR / dir_pattern.format(q=q)
            m = load_origami_metrics(run_dir)
            if m:
                series["qualities"].append(q)
                for metric_key in METRICS:
                    series["metrics"][metric_key].append(m.get(metric_key))
                series["sizes"].append(m.get("total_size_bytes", 0))
                series["pack_sizes"].append(m.get("pack_size_bytes", 0))
        data[series_name] = series

    return data


def make_chart(data, metric_key, metric_info, output_path):
    """Create a single metric chart with all series."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for series_name, series in data.items():
        qs = series["qualities"]
        vals = series["metrics"][metric_key]
        if not qs or not vals or all(v is None for v in vals):
            continue

        style = STYLES[series_name]
        ax.plot(qs, vals,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
                label=series_name)

    arrow = metric_info["arrow"]
    ax.set_xlabel("Quality", fontsize=13)
    ax.set_ylabel(f'{metric_info["label"]} {arrow}', fontsize=13)
    ax.set_title(f'{metric_info["label"]} {arrow} vs Quality', fontsize=15)
    ax.set_xticks(QUALITIES)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)


    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_size_chart(data, output_path):
    """Create a total-size-vs-quality chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for series_name, series in data.items():
        qs = series["qualities"]
        sizes_kb = [s / 1024 for s in series["sizes"]]
        if not qs or not sizes_kb:
            continue

        style = STYLES[series_name]
        ax.plot(qs, sizes_kb,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
                label=series_name)

    ax.set_xlabel("Quality", fontsize=13)
    ax.set_ylabel("Total Size \u2193 (KB)", fontsize=13)
    ax.set_title("Total Size \u2193 vs Quality", fontsize=15)
    ax.set_xticks(QUALITIES)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_rd_chart(data, metric_key, metric_info, output_path, min_quality=None):
    """Create a rate-distortion chart: metric vs size (KB)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for series_name, series in data.items():
        indices = range(len(series["qualities"]))
        if min_quality is not None:
            indices = [i for i in indices if series["qualities"][i] >= min_quality]
        sizes_kb = [series["sizes"][i] / 1024 for i in indices]
        vals = [series["metrics"][metric_key][i] for i in indices]
        if not sizes_kb or all(v is None for v in vals):
            continue

        style = STYLES[series_name]
        ax.plot(sizes_kb, vals,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
                label=series_name)

    arrow = metric_info["arrow"]
    zoom_label = f" (Q{min_quality}\u2013Q90 zoom)" if min_quality else ""
    ax.set_xlabel("Total Size \u2193 (KB)", fontsize=13)
    ax.set_ylabel(f'{metric_info["label"]} {arrow}', fontsize=13)
    ax.set_title(f'{metric_info["label"]} {arrow} vs Size \u2193 (Rate-Distortion){zoom_label}', fontsize=15)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # Filter data for annotations when zoomed
    if min_quality:
        filtered = _filter_data(data, min_quality)
        _annotate_quality_labels(ax, filtered, "sizes", metric_key)
    else:
        _annotate_quality_labels(ax, data, "sizes", metric_key)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_pack_size_chart(data, output_path):
    """Create a pack-size-vs-quality chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for series_name, series in data.items():
        qs = series["qualities"]
        sizes_kb = [s / 1024 for s in series["pack_sizes"]]
        if not qs or not sizes_kb:
            continue

        style = STYLES[series_name]
        ax.plot(qs, sizes_kb,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
                label=series_name)

    ax.set_xlabel("Quality", fontsize=13)
    ax.set_ylabel("Pack Size \u2193 (KB)", fontsize=13)
    ax.set_title("Pack Size \u2193 vs Quality", fontsize=15)
    ax.set_xticks(QUALITIES)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_metric_vs_pack_chart(data, metric_key, metric_info, output_path, min_quality=None):
    """Create a metric-vs-pack-size chart (rate-distortion using pack KB)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for series_name, series in data.items():
        indices = range(len(series["qualities"]))
        if min_quality is not None:
            indices = [i for i in indices if series["qualities"][i] >= min_quality]
        sizes_kb = [series["pack_sizes"][i] / 1024 for i in indices]
        vals = [series["metrics"][metric_key][i] for i in indices]
        if not sizes_kb or all(v is None for v in vals):
            continue

        style = STYLES[series_name]
        ax.plot(sizes_kb, vals,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
                label=series_name)

    arrow = metric_info["arrow"]
    zoom_label = f" (Q{min_quality}\u2013Q90 zoom)" if min_quality else ""
    ax.set_xlabel("Pack Size \u2193 (KB)", fontsize=13)
    ax.set_ylabel(f'{metric_info["label"]} {arrow}', fontsize=13)
    ax.set_title(f'{metric_info["label"]} {arrow} vs Pack Size \u2193{zoom_label}', fontsize=15)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    if min_quality:
        filtered = _filter_data(data, min_quality)
        _annotate_quality_labels(ax, filtered, "pack_sizes", metric_key)
    else:
        _annotate_quality_labels(ax, data, "pack_sizes", metric_key)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def make_zoomed_chart(data, metric_key, metric_info, output_path, min_quality=50):
    """Create a zoomed metric-vs-quality chart starting at the given quality."""
    fig, ax = plt.subplots(figsize=(10, 6))

    zoomed_qs = [q for q in QUALITIES if q >= min_quality]

    for series_name, series in data.items():
        # Filter to only qualities >= min_quality
        indices = [i for i, q in enumerate(series["qualities"]) if q >= min_quality]
        qs = [series["qualities"][i] for i in indices]
        vals = [series["metrics"][metric_key][i] for i in indices]
        if not qs or all(v is None for v in vals):
            continue

        style = STYLES[series_name]
        ax.plot(qs, vals,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=7,
                label=series_name)

    arrow = metric_info["arrow"]
    ax.set_xlabel("Quality", fontsize=13)
    ax.set_ylabel(f'{metric_info["label"]} {arrow}', fontsize=13)
    ax.set_title(f'{metric_info["label"]} {arrow} vs Quality (Q{min_quality}\u2013Q90 zoom)', fontsize=15)
    ax.set_xticks(zoomed_qs)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)


    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting data from runs...")
    data = collect_all_data()

    # Print summary
    for name, series in data.items():
        print(f"  {name}: {len(series['qualities'])} quality levels")

    # Metric vs Quality charts
    print("\nGenerating metric vs quality charts...")
    for metric_key, metric_info in METRICS.items():
        output_path = OUTPUT_DIR / f"{metric_key}_vs_quality.png"
        make_chart(data, metric_key, metric_info, output_path)

    # Size vs Quality chart
    print("\nGenerating size chart...")
    make_size_chart(data, OUTPUT_DIR / "size_vs_quality.png")

    # Rate-distortion charts (metric vs size)
    # SSIM and LPIPS are zoomed to Q50+ for better detail
    print("\nGenerating rate-distortion charts...")
    for metric_key in ["psnr", "ssim"]:
        metric_info = METRICS[metric_key]
        zoom = 50 if metric_key in ("ssim", "lpips") else None
        output_path = OUTPUT_DIR / f"{metric_key}_vs_size.png"
        make_rd_chart(data, metric_key, metric_info, output_path, min_quality=zoom)

    # Pack size charts
    print("\nGenerating pack size charts...")
    make_pack_size_chart(data, OUTPUT_DIR / "pack_size_vs_quality.png")

    # Metric vs Pack Size charts
    # SSIM and LPIPS are zoomed to Q50+ for better detail
    print("\nGenerating metric vs pack size charts...")
    for metric_key, metric_info in METRICS.items():
        zoom = 50 if metric_key in ("ssim", "lpips") else None
        output_path = OUTPUT_DIR / f"{metric_key}_vs_pack_size.png"
        make_metric_vs_pack_chart(data, metric_key, metric_info, output_path, min_quality=zoom)

    # Zoomed charts (Q50–Q90)
    print("\nGenerating zoomed charts...")
    for metric_key in ["ssim", "lpips"]:
        metric_info = METRICS[metric_key]
        output_path = OUTPUT_DIR / f"{metric_key}_vs_quality_zoomed.png"
        make_zoomed_chart(data, metric_key, metric_info, output_path, min_quality=50)

    print(f"\nAll charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
