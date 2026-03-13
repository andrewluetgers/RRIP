#!/usr/bin/env python3
"""Generate charts comparing JPEG XL served tiles vs ORIGAMI v2 at fixed seed qualities.

Compares STORAGE SIZE (single 1024x1024 source image encoded once) vs QUALITY
(metrics measured on rendered tiles served to clients).

Series:
  - JPEGXL baseline:  jpegxl_jpeg_baseline_q{Q}  (Q=30..90)
  - JPEG baseline:    jpeg_baseline_q{Q}          (Q=30..90)
  - ORIGAMI v2 b90:   v2_b90_l0q{Q}_ss256_nooptl2 (Q=30..90)
  - ORIGAMI v2 b95:   v2_b95_l0q{Q}_ss256_nooptl2 (Q=30..90)

Size metric:
  - Baselines: size of single 1024x1024 source image encoded at that quality
  - ORIGAMI: seed (L2) + fused L0 residual

Quality metrics:
  - All metrics measured against served tiles (re-encoded through JPEG Q95 for ORIGAMI)

Usage:
    uv run python evals/scripts/generate_jxl_vs_origami_charts.py

Output: evals/charts/jxl_vs_origami/
"""

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "charts" / "jxl_vs_origami"
SOURCE_IMAGE = Path(__file__).resolve().parent.parent / "test-images" / "L0-1024.jpg"

QUALITIES = list(range(30, 100, 5)) + [97, 99]  # 30, 35, ..., 95, 97, 99

METRICS = {
    "psnr":         {"label": "PSNR (dB)",           "higher_better": True,  "arrow": "\u2191"},
    "ssim":         {"label": "SSIM",                 "higher_better": True,  "arrow": "\u2191"},
    "ms_ssim":      {"label": "MS-SSIM",              "higher_better": True,  "arrow": "\u2191"},
    "mse":          {"label": "MSE",                  "higher_better": False, "arrow": "\u2193"},
    "vif":          {"label": "VIF",                  "higher_better": True,  "arrow": "\u2191"},
    "delta_e":      {"label": "Delta E (CIEDE2000)",  "higher_better": False, "arrow": "\u2193"},
    "lpips":        {"label": "LPIPS",                "higher_better": False, "arrow": "\u2193"},
    "dssim":        {"label": "DSSIM",                "higher_better": False, "arrow": "\u2193"},
    "ssimulacra":   {"label": "SSIMULACRA",           "higher_better": False, "arrow": "\u2193"},
    "ssimulacra2":  {"label": "SSIMULACRA2",          "higher_better": True,  "arrow": "\u2191"},
    "butteraugli":  {"label": "Butteraugli",          "higher_better": False, "arrow": "\u2193"},
    "blockiness":   {"label": "Blockiness",           "higher_better": False, "arrow": "\u2193"},
}

# Which metrics to include in rate-distortion charts (metric vs size)
RD_METRICS = ["psnr", "ssim", "ms_ssim", "mse", "delta_e", "lpips",
              "dssim", "ssimulacra2", "butteraugli"]

# Perceptual quality tier bands for each metric.
# Each band: (lo, hi, label, color)  — bands drawn bottom-to-top.
# For "higher_better" metrics, best tier is at top; for "lower_better", best is at bottom.
# None means extend to axis limit.
QUALITY_BANDS = {
    "psnr": [
        (None, 30,   "Poor",      "#fee2e2"),
        (30,   40,   "Good",      "#fef9c3"),
        (40,   None, "Lossless",  "#dcfce7"),
    ],
    "ssim": [
        (None, 0.80, "Poor",      "#fee2e2"),
        (0.80, 0.90, "Fair",      "#ffedd5"),
        (0.90, 0.95, "Good",      "#fef9c3"),
        (0.95, 0.99, "Excellent", "#dcfce7"),
        (0.99, None, "Lossless",  "#bbf7d0"),
    ],
    "ms_ssim": [
        (None,  0.990, "Good",      "#fef9c3"),
        (0.990, 0.995, "Excellent", "#dcfce7"),
        (0.995, 0.997, "Very Good", "#dcfce7"),
        (0.997, None,  "Lossless",  "#bbf7d0"),
    ],
    "mse": [
        (None, 6.5,  "Lossless",  "#bbf7d0"),
        (6.5,  65,   "Good",      "#fef9c3"),
        (65,   None, "Poor",      "#fee2e2"),
    ],
    "delta_e": [
        (None, 1.0, "Lossless",   "#bbf7d0"),
        (1.0,  2.0, "Excellent",  "#dcfce7"),
        (2.0,  3.5, "Good",       "#fef9c3"),
        (3.5,  5.0, "Fair",       "#ffedd5"),
        (5.0,  None,"Poor",       "#fee2e2"),
    ],
    "lpips": [
        (None, 0.02, "Lossless",  "#bbf7d0"),
        (0.02, 0.05, "Excellent", "#dcfce7"),
        (0.05, 0.10, "Good",      "#fef9c3"),
        (0.10, 0.15, "Fair",      "#ffedd5"),
        (0.15, None, "Poor",      "#fee2e2"),
    ],
    "ssimulacra2": [
        (None, 30,  "Poor",       "#fee2e2"),
        (30,   50,  "Low",        "#ffedd5"),
        (50,   70,  "Fair",       "#fef9c3"),
        (70,   80,  "Good",       "#dcfce7"),
        (80,   90,  "Excellent",  "#bbf7d0"),
        (90,   None,"Lossless",   "#86efac"),
    ],
    "butteraugli": [
        (None, 1.0, "Lossless",   "#bbf7d0"),
        (1.0,  1.1, "JND",        "#dcfce7"),
        (1.1,  None,"Visible",    "#fef9c3"),
    ],
    "dssim": [
        (None,  0.003, "Lossless", "#bbf7d0"),
        (0.003, 0.01,  "Good",     "#fef9c3"),
        (0.01,  None,  "Visible",  "#ffedd5"),
    ],
}


def draw_quality_bands(ax, metric_key):
    """Draw horizontal quality tier bands as background shading with a stacked legend."""
    from matplotlib.patches import Patch

    bands = QUALITY_BANDS.get(metric_key)
    if not bands:
        return

    ymin, ymax = ax.get_ylim()
    visible_bands = []

    for lo, hi, label, color in bands:
        band_lo = lo if lo is not None else ymin
        band_hi = hi if hi is not None else ymax

        # Clip to visible range
        band_lo = max(band_lo, ymin)
        band_hi = min(band_hi, ymax)

        if band_hi <= band_lo:
            continue

        ax.axhspan(band_lo, band_hi, color=color, alpha=0.35, zorder=0)
        visible_bands.append((label, color))

    # Build band legend handles (de-duplicated, preserving order)
    if visible_bands:
        seen = set()
        unique = []
        for label, color in visible_bands:
            if label not in seen:
                seen.add(label)
                unique.append((label, color))

        band_handles = [Patch(facecolor=c, alpha=0.35, edgecolor='#ccc', label=l)
                        for l, c in unique]

        from matplotlib.legend_handler import HandlerPatch
        import matplotlib.lines as mlines

        # Per-metric legend corner overrides
        LEGEND_CORNERS = {
            "delta_e":     "lower left",
            "butteraugli": "upper right",
            "dssim":       "upper right",
            "lpips":       "upper right",
            "mse":         "upper right",
        }
        loc = LEGEND_CORNERS.get(metric_key, "lower right")

        # Build a single combined legend: band section header + band entries +
        # blank spacer + data section header + data entries
        # Use invisible patches as section headers
        section_band = Patch(facecolor='none', edgecolor='none', label='$\\bf{Perceptual\\ Quality}$')
        spacer = Patch(facecolor='none', edgecolor='none', label=' ')

        main_legend = ax.get_legend()
        main_handles = main_legend.legend_handles if main_legend else []
        main_labels = [t.get_text() for t in main_legend.get_texts()] if main_legend else []

        combined_handles = main_handles + [spacer] + list(reversed(band_handles))
        combined_labels = main_labels + [' '] + [h.get_label() for h in reversed(band_handles)]

        # Remove old legend and create combined one
        if main_legend:
            main_legend.remove()

        ax.legend(combined_handles, combined_labels, loc=loc,
                  fontsize=8, framealpha=0.85, edgecolor="#ccc",
                  handlelength=1.5, handleheight=0.8)

SERIES_DEFS = [
    {
        "name": "JPEG baseline",
        "dir_pattern": "jpeg_baseline_q{q}",
        "loader": "baseline",
        "encoder_type": "jpeg",
        "style": {"color": "#888888", "marker": "s", "linestyle": "--", "linewidth": 1.5},
    },
    {
        "name": "JPEG XL baseline",
        "dir_pattern": "jpegxl_jpeg_baseline_q{q}",
        "loader": "baseline",
        "encoder_type": "jpegxl",
        "style": {"color": "#e377c2", "marker": "^", "linestyle": "--", "linewidth": 2},
    },
    {
        "name": "ORIGAMI v2 SQ90",
        "dir_pattern": "v2_b90_l0q{q}_ss256_nooptl2",
        "loader": "origami",
        "style": {"color": "#2ca02c", "marker": "o", "linestyle": "-", "linewidth": 2},
    },
    {
        "name": "ORIGAMI v2 SQ95",
        "dir_pattern": "v2_b95_l0q{q}_ss256_nooptl2",
        "loader": "origami",
        "style": {"color": "#1f77b4", "marker": "D", "linestyle": "-", "linewidth": 2},
    },
]


def _encode_source_jpeg(quality):
    """Encode the 1024x1024 source image as JPEG at the given quality, return size in bytes."""
    img = Image.open(SOURCE_IMAGE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.tell()


def _encode_source_jxl(quality):
    """Encode the 1024x1024 source image as JPEG XL at the given quality, return size in bytes."""
    try:
        # Add evals/scripts to path for jpeg_encoder import
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from jpeg_encoder import encode_jpeg_to_bytes, JpegEncoder
        img = Image.open(SOURCE_IMAGE)
        data = encode_jpeg_to_bytes(img, quality, JpegEncoder.JPEGXL)
        return len(data)
    except ImportError:
        # Fallback: check if .jxl files exist in the run dir
        return None


# Cache source image sizes to avoid re-encoding
_source_size_cache = {}


def get_source_size(quality, encoder_type):
    """Get the size of the 1024x1024 source image encoded at the given quality."""
    key = (quality, encoder_type)
    if key not in _source_size_cache:
        if encoder_type == "jpegxl":
            _source_size_cache[key] = _encode_source_jxl(quality)
        else:
            _source_size_cache[key] = _encode_source_jpeg(quality)
    return _source_size_cache[key]


def load_baseline_metrics(run_dir, encoder_type="jpeg"):
    """Extract per-tile metrics from a baseline manifest, return averages.

    Size is the single 1024x1024 source image encoded at that quality (storage cost),
    NOT the sum of all 21 pyramid tiles.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    m = json.loads(manifest_path.read_text())

    # Baseline manifests have "tiles" dict with per-tile metrics
    tiles = m.get("tiles", {})
    if isinstance(tiles, list):
        tiles = {f"t{i}": t for i, t in enumerate(tiles)}

    if not tiles:
        return None

    # Average quality metrics across L0 + L1 tiles only (not L2)
    # to match ORIGAMI which reports L0 + L1 metrics
    metrics = {}
    for metric_key in METRICS:
        values = [t[metric_key] for k, t in tiles.items()
                  if metric_key in t and not k.startswith("L2_")]
        if values:
            metrics[metric_key] = np.mean(values)

    # Storage size = single 1024x1024 source image at this quality
    quality = m.get("configuration", {}).get("jpeg_quality", 0)
    source_size = get_source_size(quality, encoder_type)
    if source_size is not None:
        metrics["total_size_bytes"] = source_size
    else:
        # Fallback: use sum of all tile sizes (old behavior)
        total = m.get("statistics", {}).get("total_bytes", 0)
        if total == 0:
            total = sum(t.get("size_bytes", 0) for t in tiles.values())
        metrics["total_size_bytes"] = total

    return metrics


def load_origami_metrics(run_dir):
    """Extract per-tile metrics from an ORIGAMI manifest, return averages."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    m = json.loads(manifest_path.read_text())

    # Collect final_* metrics from decompression_phase L0 and L1 tiles
    all_metrics = {k: [] for k in METRICS}

    dp = m.get("decompression_phase", {})
    for level_key in ["L1", "L0"]:
        level = dp.get(level_key, {})
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

    # Size from manifest top-level or size_comparison
    sc = m.get("size_comparison", {})
    metrics["total_size_bytes"] = m.get("total_bytes", sc.get("origami_total", 0))

    return metrics


def collect_all_data():
    """Collect metric data for all series."""
    data = {}

    for sdef in SERIES_DEFS:
        series = {"qualities": [], "metrics": {k: [] for k in METRICS}, "sizes": []}

        for q in QUALITIES:
            run_dir = RUNS_DIR / sdef["dir_pattern"].format(q=q)
            if sdef["loader"] == "baseline":
                m = load_baseline_metrics(run_dir, encoder_type=sdef.get("encoder_type", "jpeg"))
            else:
                m = load_origami_metrics(run_dir)
            if m:
                series["qualities"].append(q)
                for metric_key in METRICS:
                    series["metrics"][metric_key].append(m.get(metric_key))
                series["sizes"].append(m.get("total_size_bytes", 0))

        data[sdef["name"]] = {"series": series, "style": sdef["style"]}

    return data


def make_metric_vs_quality(data, metric_key, metric_info, output_path):
    """Metric vs L0 quality chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, d in data.items():
        s = d["series"]
        qs = s["qualities"]
        vals = s["metrics"][metric_key]
        if not qs or all(v is None for v in vals):
            continue
        st = d["style"]
        ax.plot(qs, vals, color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], linewidth=st["linewidth"],
                markersize=7, label=name)

    arrow = metric_info["arrow"]
    ax.set_xlabel("Quality", fontsize=13)
    ax.set_ylabel(f'{metric_info["label"]} {arrow}', fontsize=13)
    ax.set_title(f'{metric_info["label"]} {arrow} vs Quality — JXL vs ORIGAMI v2', fontsize=14)
    ax.set_xticks([q for q in QUALITIES if q % 10 == 0])
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.tick_params(labelsize=11)
    draw_quality_bands(ax, metric_key)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {output_path.name}")


def make_size_vs_quality(data, output_path):
    """Total tile size vs quality chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, d in data.items():
        s = d["series"]
        qs = s["qualities"]
        sizes_kb = [sz / 1024 for sz in s["sizes"]]
        if not qs:
            continue
        st = d["style"]
        ax.plot(qs, sizes_kb, color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], linewidth=st["linewidth"],
                markersize=7, label=name)

    ax.set_xlabel("Quality", fontsize=13)
    ax.set_ylabel("Storage Size \u2193 (KB)", fontsize=13)
    ax.set_title("Storage Size \u2193 vs Quality — JXL vs ORIGAMI v2", fontsize=14)
    ax.set_xticks([q for q in QUALITIES if q % 10 == 0])
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {output_path.name}")


def make_rd_chart(data, metric_key, metric_info, output_path, min_quality=None):
    """Rate-distortion: metric vs total size (KB)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, d in data.items():
        s = d["series"]
        indices = range(len(s["qualities"]))
        if min_quality is not None:
            indices = [i for i in indices if s["qualities"][i] >= min_quality]
        sizes_kb = [s["sizes"][i] / 1024 for i in indices]
        vals = [s["metrics"][metric_key][i] for i in indices]
        qs = [s["qualities"][i] for i in indices]
        if not sizes_kb or all(v is None for v in vals):
            continue
        st = d["style"]
        ax.plot(sizes_kb, vals, color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], linewidth=st["linewidth"],
                markersize=7, label=name)

        # Label select quality points
        for q, x, y in zip(qs, sizes_kb, vals):
            if y is None:
                continue
            if q % 20 == 0 or q == 30:
                ax.annotate(f"q{q}", (x, y), fontsize=7, fontweight="bold",
                            color=st["color"], alpha=0.8,
                            textcoords="offset points", xytext=(5, 3))

    arrow = metric_info["arrow"]
    zoom_label = f" (Q{min_quality}+ zoom)" if min_quality else ""
    ax.set_xlabel("Storage Size \u2193 (KB)", fontsize=13)
    ax.set_ylabel(f'{metric_info["label"]} {arrow}', fontsize=13)
    ax.set_title(f'{metric_info["label"]} {arrow} vs Storage Size — JXL vs ORIGAMI v2{zoom_label}', fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.tick_params(labelsize=11)
    draw_quality_bands(ax, metric_key)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {output_path.name}")


def main():
    print("Collecting data...")
    data = collect_all_data()

    for name, d in data.items():
        n = len(d["series"]["qualities"])
        print(f"  {name}: {n} quality levels")
        if n == 0:
            print(f"    WARNING: no data found for {name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Metric vs Quality
    print("\nMetric vs Quality charts:")
    for mk, mi in METRICS.items():
        make_metric_vs_quality(data, mk, mi, OUTPUT_DIR / f"{mk}_vs_quality.png")

    # Size vs Quality
    print("\nSize vs Quality:")
    make_size_vs_quality(data, OUTPUT_DIR / "size_vs_quality.png")

    # Rate-distortion (full range)
    print("\nRate-distortion charts (full range):")
    for mk in RD_METRICS:
        mi = METRICS[mk]
        make_rd_chart(data, mk, mi, OUTPUT_DIR / f"{mk}_vs_size.png")

    # Rate-distortion (zoomed Q50+)
    print("\nRate-distortion charts (Q50+ zoom):")
    for mk in RD_METRICS:
        mi = METRICS[mk]
        make_rd_chart(data, mk, mi, OUTPUT_DIR / f"{mk}_vs_size_zoomed.png", min_quality=50)

    print(f"\nAll charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()