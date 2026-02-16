#!/usr/bin/env python3
"""Generate metric charts comparing ORIGAMI and JPEG baseline encoders.

Reads manifest.json from each run in evals/runs/ and produces charts showing
how metrics vary across quality levels for Rust encoder variants and baselines.

Usage:
    uv run python evals/scripts/generate_charts.py

Output directory: evals/charts/

Chart sets:
  rust/       — All Rust encoder variants (444, 444+OptL2, 420+OptL2,
                420opt+OptL2) with uniform and +20 split quality, plus
                JPEG baseline
  comparison/ — Rust 444+OptL2 vs Python OptL2 (legacy 420 pipeline)

Series style:
  - Dashed lines + square markers = JPEG baselines
  - Solid lines + circle markers = ORIGAMI uniform quality
  - Dotted lines + diamond markers = ORIGAMI +20 split quality
  - Arrows on axis labels indicate which direction is better

Prerequisites:
  - Run data must exist in evals/runs/ (generate with generate-evals.sh)
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

QUALITIES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

# --- "rust" chart set: Rust encoder variants + JPEG baseline ---
RUST_BASELINE_DIRS = {
    "JPEG baseline": "jpeg_baseline_q{q}",
}

RUST_ORIGAMI_DIRS = {
    "RS 444": "rs_444_j{q}",
    "RS 444+OptL2": "rs_444_optl2_j{q}",
    "RS 420+OptL2": "rs_420_optl2_j{q}",
    "RS 420opt+OptL2": "rs_420opt_optl2_j{q}",
}

RUST_SPLIT_SERIES = {
    "RS 444 +20 split": {"offset": 20, "dir": "rs_444_l1q{l1q}_l0q{l0q}"},
    "RS 444+OptL2 +20 split": {"offset": 20, "dir": "rs_444_optl2_l1q{l1q}_l0q{l0q}"},
    "RS 420+OptL2 +20 split": {"offset": 20, "dir": "rs_420_optl2_l1q{l1q}_l0q{l0q}"},
    "RS 420opt+OptL2 +20 split": {"offset": 20, "dir": "rs_420opt_optl2_l1q{l1q}_l0q{l0q}"},
}

# --- "comparison" chart set: Rust best vs Python OptL2 legacy ---
COMP_BASELINE_DIRS = {
    "JPEG baseline": "jpeg_baseline_q{q}",
}

COMP_ORIGAMI_DIRS = {
    "RS 444+OptL2": "rs_444_optl2_j{q}",
    "Py OptL2 (420)": "optl2_debug_j{q}_pac",
}

COMP_SPLIT_SERIES = {
    "RS 444+OptL2 +20 split": {"offset": 20, "dir": "rs_444_optl2_l1q{l1q}_l0q{l0q}"},
    "Py OptL2 +20 split": {"offset": 20, "dir": "optl2_debug_l1q{l1q}_l0q{l0q}_pac"},
}

# Colors and markers for each series
STYLES = {
    # Baselines (dashed, squares)
    "JPEG baseline":  {"color": "#1f77b4", "marker": "s", "linestyle": "--"},
    # Rust uniform (solid, circles)
    "RS 444":                {"color": "#aec7e8", "marker": "o", "linestyle": "-"},
    "RS 444+OptL2":          {"color": "#2ca02c", "marker": "o", "linestyle": "-"},
    "RS 420+OptL2":          {"color": "#ff7f0e", "marker": "o", "linestyle": "-"},
    "RS 420opt+OptL2":       {"color": "#d62728", "marker": "o", "linestyle": "-"},
    # Rust splits (diamonds)
    "RS 444 +20 split":           {"color": "#aec7e8", "marker": "D", "linestyle": ":"},
    "RS 444+OptL2 +20 split":    {"color": "#2ca02c", "marker": "D", "linestyle": ":"},
    "RS 420+OptL2 +20 split":    {"color": "#ff7f0e", "marker": "D", "linestyle": ":"},
    "RS 420opt+OptL2 +20 split": {"color": "#d62728", "marker": "D", "linestyle": ":"},
    # Python legacy (dashed, triangles)
    "Py OptL2 (420)":       {"color": "#9467bd", "marker": "^", "linestyle": "--"},
    "Py OptL2 +20 split":   {"color": "#9467bd", "marker": "D", "linestyle": ":"},
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


LABEL_SERIES = {"RS 444+OptL2 +20 split", "JPEG baseline"}

def _annotate_quality_labels(ax, data, x_source, metric_key):
    """Annotate quality labels on key series for KB-based x-axis charts."""
    for series_name, series in data.items():
        if series_name not in LABEL_SERIES:
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
            # Label every other quality to reduce clutter
            if q % 20 != 0 and q != 10:
                continue
            ax.annotate(f"q{q}", (x, y),
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


def collect_data(baseline_dirs, origami_dirs, split_series):
    """Collect metric data for the given series across all quality levels.

    Automatically discovers which quality levels exist for each series
    by probing the runs directory, so it handles both step-5 (Python)
    and step-10 (Rust) quality grids.
    """
    data = {}

    # Baselines
    for series_name, dir_pattern in baseline_dirs.items():
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

    # ORIGAMI (uniform quality)
    for series_name, dir_pattern in origami_dirs.items():
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

    # Split-quality ORIGAMI series
    for series_name, spec in split_series.items():
        offset = spec["offset"]
        dir_pattern = spec["dir"]
        series = {"qualities": [], "metrics": {k: [] for k in METRICS}, "sizes": [], "pack_sizes": []}
        for l0q in QUALITIES:
            l1q = l0q + offset
            if l1q > 90:
                continue
            run_dir = RUNS_DIR / dir_pattern.format(l1q=l1q, l0q=l0q)
            m = load_origami_metrics(run_dir)
            if m:
                series["qualities"].append(l0q)
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
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
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
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
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
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
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


def generate_chart_set(data, out_dir):
    """Generate all chart types for a given dataset into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metric vs Quality charts
    print("\n  Metric vs quality charts...")
    for metric_key, metric_info in METRICS.items():
        make_chart(data, metric_key, metric_info, out_dir / f"{metric_key}_vs_quality.png")

    # Size vs Quality chart
    print("  Size chart...")
    make_size_chart(data, out_dir / "size_vs_quality.png")

    # Rate-distortion charts (metric vs size)
    print("  Rate-distortion charts...")
    for metric_key in ["psnr", "ssim"]:
        metric_info = METRICS[metric_key]
        zoom = 50 if metric_key in ("ssim", "lpips") else None
        make_rd_chart(data, metric_key, metric_info, out_dir / f"{metric_key}_vs_size.png", min_quality=zoom)

    # Pack size charts
    print("  Pack size charts...")
    make_pack_size_chart(data, out_dir / "pack_size_vs_quality.png")

    # Metric vs Pack Size charts
    print("  Metric vs pack size charts...")
    for metric_key, metric_info in METRICS.items():
        zoom = 50 if metric_key in ("ssim", "lpips") else None
        make_metric_vs_pack_chart(data, metric_key, metric_info, out_dir / f"{metric_key}_vs_pack_size.png", min_quality=zoom)

    # Zoomed charts (Q50-Q90)
    print("  Zoomed charts...")
    for metric_key in ["ssim", "lpips"]:
        metric_info = METRICS[metric_key]
        make_zoomed_chart(data, metric_key, metric_info, out_dir / f"{metric_key}_vs_quality_zoomed.png", min_quality=50)


def main():
    # --- Rust encoder chart set (all Rust variants + JPEG baseline) ---
    print("=== Rust encoder charts ===")
    print("Collecting data...")
    rust_data = collect_data(RUST_BASELINE_DIRS, RUST_ORIGAMI_DIRS, RUST_SPLIT_SERIES)
    for name, series in rust_data.items():
        print(f"  {name}: {len(series['qualities'])} quality levels")
    rust_dir = OUTPUT_DIR / "rust"
    generate_chart_set(rust_data, rust_dir)
    print(f"Rust charts saved to {rust_dir}")

    # --- Comparison chart set (Rust best vs Python OptL2 legacy) ---
    print("\n=== Rust vs Python comparison charts ===")
    print("Collecting data...")
    comp_data = collect_data(COMP_BASELINE_DIRS, COMP_ORIGAMI_DIRS, COMP_SPLIT_SERIES)
    for name, series in comp_data.items():
        print(f"  {name}: {len(series['qualities'])} quality levels")
    comp_dir = OUTPUT_DIR / "comparison"
    generate_chart_set(comp_data, comp_dir)
    print(f"Comparison charts saved to {comp_dir}")

    print(f"\nAll charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
