"""
JXL Quality Sweep — Morphological Decomposition (Phase 2)
==========================================================

Uses JXL compression at decreasing quality levels to peel a residual into layers.
Each quality step removes a layer — classified as structure, noise, or artifact
using correlation with source images.

Usage:
    # Run sweep on a single-image encode output (needs --debug-images)
    uv run python evals/scripts/jxl_quality_sweep.py \
        --run-dir evals/runs/rs_444_b95_l0q50

    # Run with analysis mode (JXL --epf=0 --gaborish=0)
    uv run python evals/scripts/jxl_quality_sweep.py \
        --run-dir evals/runs/rs_444_b95_l0q50 --config analysis

    # Custom quality levels
    uv run python evals/scripts/jxl_quality_sweep.py \
        --run-dir evals/runs/rs_444_b95_l0q50 --qualities 100,90,70,50,30

    # Specify a particular family (L2 parent coordinates)
    uv run python evals/scripts/jxl_quality_sweep.py \
        --run-dir evals/runs/rs_444_b95_l0q50 --family 0_2_1

Prerequisites:
    - cjxl and djxl installed (brew install jpeg-xl)
    - Run generated with --debug-images (needs compress/ dir with residuals)
    - eval_noise_floor.py (Phase 1) for noise scoring

Output:
    evals/analysis/noise_floor/{experiment_id}/
    ├── metadata.json
    ├── metrics.json
    ├── sources/
    │   ├── original_Y.png
    │   ├── prediction_Y.png
    │   ├── residual.png
    │   └── missed_edges.png
    ├── jxl_sweep/
    │   ├── q100/
    │   │   ├── roundtrip.png
    │   │   ├── delta.png
    │   │   ├── delta_heatmap.png
    │   │   ├── reconstruction.png
    │   │   ├── edge_correlation_map.png
    │   │   └── stats.json
    │   ├── q95/ ...
    │   ├── summary_strip.png
    │   ├── rd_curve.png
    │   └── layer_classification.png
    └── notes.txt
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
from scipy.fft import fft2
from skimage.filters import sobel

# Import noise-floor evaluation from Phase 1
sys.path.insert(0, os.path.dirname(__file__))
from eval_noise_floor import evaluate_image, NumpyEncoder

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# JXL roundtrip
# ---------------------------------------------------------------------------

def jxl_roundtrip(input_png: str, quality: int, config: str = "default") -> tuple:
    """Encode PNG → JXL → decode back to PNG. Returns (output_png_path, jxl_size_bytes)."""
    with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as jxl_f:
        jxl_path = jxl_f.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out_f:
        out_path = out_f.name

    try:
        # Encode
        cmd = ["cjxl", input_png, jxl_path, "-q", str(quality)]
        if config == "analysis":
            cmd.extend(["--epf=0", "--gaborish=0", "--patches=0", "--dots=0"])
        elif config == "modular":
            cmd.append("--modular")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"cjxl failed: {result.stderr}")

        jxl_size = os.path.getsize(jxl_path)

        # Decode
        result = subprocess.run(
            ["djxl", jxl_path, out_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"djxl failed: {result.stderr}")

        return out_path, jxl_size

    finally:
        if os.path.exists(jxl_path):
            os.unlink(jxl_path)


# ---------------------------------------------------------------------------
# Three-signal analysis
# ---------------------------------------------------------------------------

def compute_missed_edges(original_Y: np.ndarray, prediction_Y: np.ndarray) -> np.ndarray:
    """Edges present in original but not in prediction."""
    orig_edges = sobel(original_Y.astype(np.float64) / 255.0)
    pred_edges = sobel(prediction_Y.astype(np.float64) / 255.0)
    return np.clip(orig_edges - pred_edges, 0, None)


def edge_correlation(delta: np.ndarray, missed_edges: np.ndarray) -> float:
    """Pearson correlation between |delta| and missed edges."""
    d = np.abs(delta.astype(np.float64))
    me = missed_edges.ravel()
    d_flat = d.ravel()

    if np.std(d_flat) < 1e-10 or np.std(me) < 1e-10:
        return 0.0

    return float(np.corrcoef(d_flat, me)[0, 1])


def fft_peak_ratio(data: np.ndarray, period: int) -> float:
    """Ratio of FFT energy at given period vs neighboring frequencies."""
    centered = data.astype(np.float64) - data.mean()
    F = np.abs(fft2(centered))
    h, w = F.shape

    freq_idx = w // period
    if freq_idx < 2 or freq_idx >= w // 2 - 3:
        return 1.0

    # Peak energy (narrow band)
    peak = np.mean(F[:, freq_idx-1:freq_idx+2]) + np.mean(F[freq_idx-1:freq_idx+2, :])

    # Baseline (nearby, excluding harmonics)
    bl_lo = np.mean(F[:, max(1,freq_idx-5):freq_idx-2]) + np.mean(F[max(1,freq_idx-5):freq_idx-2, :])
    bl_hi = np.mean(F[:, freq_idx+3:freq_idx+6]) + np.mean(F[freq_idx+3:freq_idx+6, :])
    baseline = (bl_lo + bl_hi) / 2

    if baseline < 1e-10:
        return 1.0
    return float(peak / baseline)


# ---------------------------------------------------------------------------
# Discovery: find source images from a run directory
# ---------------------------------------------------------------------------

def rgb_to_y(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma from RGB uint8 image."""
    r, g, b = rgb[:,:,0].astype(np.float32), rgb[:,:,1].astype(np.float32), rgb[:,:,2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)


def assemble_l0_mosaic(compress_dir: str, pattern_type: str, tile_size: int = 256) -> str:
    """Assemble 4x4 grid of L0 tile images into a 1024x1024 mosaic.
    pattern_type: 'original' or 'prediction'
    Returns path to saved mosaic PNG, or None if tiles not found."""
    # L0 tiles are named like: 020_L0_0_0_{type}.png, 021_L0_1_0_{type}.png, etc.
    # Grid is 4x4 (x=0..3, y=0..3)
    tiles = sorted(glob.glob(os.path.join(compress_dir, f"*_L0_*_{pattern_type}.png")))
    if len(tiles) < 16:
        return None

    # Parse tile coordinates from filenames
    grid = {}
    for t in tiles:
        name = os.path.basename(t)
        # Pattern: NNN_L0_X_Y_type.png
        parts = name.split("_")
        # Find L0 marker, then x, y follow
        try:
            l0_idx = parts.index("L0")
            x = int(parts[l0_idx + 1])
            y = int(parts[l0_idx + 2])
            grid[(x, y)] = t
        except (ValueError, IndexError):
            continue

    if len(grid) < 16:
        return None

    # Find grid dimensions
    max_x = max(x for x, y in grid.keys())
    max_y = max(y for x, y in grid.keys())
    mosaic_w = (max_x + 1) * tile_size
    mosaic_h = (max_y + 1) * tile_size
    mosaic = np.zeros((mosaic_h, mosaic_w), dtype=np.uint8)

    for (x, y), path in grid.items():
        img = np.array(Image.open(path))
        if img.ndim == 3:
            img = rgb_to_y(img)
        th, tw = img.shape[:2]
        mosaic[y*tile_size:y*tile_size+th, x*tile_size:x*tile_size+tw] = img

    # Save mosaic
    out_path = os.path.join(compress_dir, f"_assembled_L0_{pattern_type}_Y.png")
    Image.fromarray(mosaic).save(out_path)
    return out_path


def find_source_images(run_dir: str, family: str = None) -> dict:
    """Find original_Y, prediction_Y, and residual from a --debug-images run."""
    compress_dir = os.path.join(run_dir, "compress")
    if not os.path.isdir(compress_dir):
        raise FileNotFoundError(f"No compress/ directory in {run_dir}. Run with --debug-images.")

    # Find fused residual PNGs
    residuals = sorted(glob.glob(os.path.join(compress_dir, "*fused_residual_centered.png")))
    if not residuals:
        raise FileNotFoundError(f"No fused_residual_centered.png in {compress_dir}")

    # If family specified, filter
    if family:
        residuals = [r for r in residuals if family in os.path.basename(r)]

    if not residuals:
        raise FileNotFoundError(f"No residual matching family '{family}' in {compress_dir}")

    # Take first family
    residual_path = residuals[0]
    basename = os.path.basename(residual_path)
    prefix = basename.replace("_L0_fused_residual_centered.png", "")

    # Look for pre-assembled Y channel images first
    original_path = os.path.join(compress_dir, f"{prefix}_L0_original_Y.png")
    prediction_path = os.path.join(compress_dir, f"{prefix}_L0_prediction_Y.png")

    # If not found, try assembled mosaics
    if not os.path.exists(original_path):
        original_path = os.path.join(compress_dir, "_assembled_L0_original_Y.png")
    if not os.path.exists(prediction_path):
        prediction_path = os.path.join(compress_dir, "_assembled_L0_prediction_Y.png")

    # If still not found, assemble from per-tile RGB images
    if not os.path.exists(original_path):
        print("  Assembling L0 original mosaic from tiles...")
        original_path = assemble_l0_mosaic(compress_dir, "original")
    if not os.path.exists(prediction_path):
        print("  Assembling L0 prediction mosaic from tiles...")
        prediction_path = assemble_l0_mosaic(compress_dir, "prediction")

    return {
        "residual": residual_path,
        "original_Y": original_path,
        "prediction_Y": prediction_path,
        "family_prefix": prefix,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    run_dir: str,
    quality_levels: list,
    config: str,
    family: str = None,
    output_base: str = "evals/analysis/noise_floor",
) -> dict:
    """Run the full JXL quality sweep experiment."""

    # Discover source images
    sources = find_source_images(run_dir, family)
    run_name = os.path.basename(run_dir)
    experiment_id = f"jxl_sweep_{run_name}_config{config[0].upper()}"

    print(f"Experiment: {experiment_id}")
    print(f"  Residual:   {sources['residual']}")
    print(f"  Original Y: {sources['original_Y']}")
    print(f"  Prediction: {sources['prediction_Y']}")
    print(f"  Config:     {config}")
    print(f"  Qualities:  {quality_levels}")

    # Create output directory
    out_dir = os.path.join(output_base, experiment_id)
    os.makedirs(out_dir, exist_ok=True)

    # Load images
    residual = np.array(Image.open(sources["residual"]).convert("L"))

    has_three_signal = sources["original_Y"] and sources["prediction_Y"]
    if has_three_signal and os.path.exists(sources["original_Y"]) and os.path.exists(sources["prediction_Y"]):
        original_Y = np.array(Image.open(sources["original_Y"]).convert("L"))
        prediction_Y = np.array(Image.open(sources["prediction_Y"]).convert("L"))
        missed_edges = compute_missed_edges(original_Y, prediction_Y)
        print(f"  Three-signal analysis: ENABLED")
    else:
        original_Y = prediction_Y = missed_edges = None
        has_three_signal = False
        print(f"  Three-signal analysis: DISABLED (missing original/prediction)")

    # Copy/save source images
    src_dir = os.path.join(out_dir, "sources")
    os.makedirs(src_dir, exist_ok=True)
    shutil.copy2(sources["residual"], os.path.join(src_dir, "residual.png"))
    if has_three_signal:
        shutil.copy2(sources["original_Y"], os.path.join(src_dir, "original_Y.png"))
        shutil.copy2(sources["prediction_Y"], os.path.join(src_dir, "prediction_Y.png"))
        # Save missed edges visualization
        me_norm = (missed_edges / (missed_edges.max() + 1e-10) * 255).astype(np.uint8)
        Image.fromarray(me_norm).save(os.path.join(src_dir, "missed_edges.png"))

    # Run JXL roundtrip at each quality level
    sweep_dir = os.path.join(out_dir, "jxl_sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    roundtrips = {}  # quality → numpy array
    sweep_metrics = {}  # quality → metrics dict
    prev_quality = None

    for q in sorted(quality_levels, reverse=True):
        print(f"  Q{q:3d}: ", end="", flush=True)
        q_dir = os.path.join(sweep_dir, f"q{q}")
        os.makedirs(q_dir, exist_ok=True)

        # Roundtrip
        rt_path, jxl_size = jxl_roundtrip(sources["residual"], q, config)
        rt_img = np.array(Image.open(rt_path).convert("L"))
        roundtrips[q] = rt_img

        # Save roundtrip
        Image.fromarray(rt_img).save(os.path.join(q_dir, "roundtrip.png"))

        # Compute delta (what this step removed vs previous quality)
        delta = None
        if prev_quality is not None:
            delta = roundtrips[prev_quality].astype(np.float32) - rt_img.astype(np.float32)
            # Save delta (centered at 128 for visualization)
            delta_vis = np.clip(delta + 128, 0, 255).astype(np.uint8)
            Image.fromarray(delta_vis).save(os.path.join(q_dir, "delta.png"))

            # Heatmap of |delta|
            delta_abs = np.abs(delta)
            if delta_abs.max() > 0:
                delta_heat = (delta_abs / delta_abs.max() * 255).astype(np.uint8)
            else:
                delta_heat = np.zeros_like(rt_img)
            Image.fromarray(delta_heat).save(os.path.join(q_dir, "delta_heatmap.png"))

        # Cumulative delta: total removal from original residual at this quality
        cum_delta = residual.astype(np.float32) - rt_img.astype(np.float32)
        cum_delta_vis = np.clip(cum_delta + 128, 0, 255).astype(np.uint8)
        Image.fromarray(cum_delta_vis).save(os.path.join(q_dir, "cumulative_delta.png"))

        # Cumulative heatmap (normalized per-image for visibility)
        cum_abs = np.abs(cum_delta)
        cum_max = cum_abs.max()
        if cum_max > 0:
            cum_heat = (cum_abs / cum_max * 255).astype(np.uint8)
        else:
            cum_heat = np.zeros_like(rt_img)
        Image.fromarray(cum_heat).save(os.path.join(q_dir, "cumulative_delta_heatmap.png"))

        # Cumulative edge correlation
        cum_edge_corr = None
        if has_three_signal:
            cum_edge_corr = edge_correlation(cum_delta, missed_edges)

            # Cumulative edge correlation map
            cum_abs_f = np.abs(cum_delta)
            me_norm_f = missed_edges / (missed_edges.max() + 1e-10)
            cum_corr_map = cum_abs_f * me_norm_f
            cum_corr_vis = (cum_corr_map / (cum_corr_map.max() + 1e-10) * 255).astype(np.uint8)
            Image.fromarray(cum_corr_vis).save(os.path.join(q_dir, "cumulative_edge_correlation_map.png"))

        # Reconstruction (prediction + roundtrip residual)
        if has_three_signal:
            residual_centered = rt_img.astype(np.float32) - 128.0
            reconstruction = np.clip(prediction_Y.astype(np.float32) + residual_centered, 0, 255).astype(np.uint8)
            Image.fromarray(reconstruction).save(os.path.join(q_dir, "reconstruction.png"))

            # Reconstruction PSNR vs original
            mse = np.mean((original_Y.astype(np.float64) - reconstruction.astype(np.float64)) ** 2)
            recon_psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99.0
        else:
            recon_psnr = None

        # Residual PSNR (roundtrip vs original residual)
        res_mse = np.mean((residual.astype(np.float64) - rt_img.astype(np.float64)) ** 2)
        res_psnr = 10 * np.log10(255**2 / res_mse) if res_mse > 0 else 99.0

        # Noise-floor score of step delta
        delta_noise_score = None
        delta_edge_corr = None
        delta_artifact_8 = None
        delta_artifact_32 = None

        if delta is not None:
            # Save delta as temp PNG for noise-floor eval
            delta_centered_u8 = np.clip(delta + 128, 0, 255).astype(np.uint8)
            delta_tmp = os.path.join(q_dir, "delta.png")
            delta_metrics = evaluate_image(delta_tmp)
            delta_noise_score = delta_metrics["noise_floor_score"]

            # Step edge correlation
            if has_three_signal:
                delta_edge_corr = edge_correlation(delta, missed_edges)

                # Step edge correlation map
                d_abs = np.abs(delta)
                me_norm_f = missed_edges / (missed_edges.max() + 1e-10)
                corr_map = d_abs * me_norm_f
                corr_vis = (corr_map / (corr_map.max() + 1e-10) * 255).astype(np.uint8)
                Image.fromarray(corr_vis).save(os.path.join(q_dir, "edge_correlation_map.png"))

            # Artifact detection
            delta_artifact_8 = fft_peak_ratio(delta, 8)
            delta_artifact_32 = fft_peak_ratio(delta, 32)

        # Noise-floor score of cumulative delta
        cum_delta_tmp = os.path.join(q_dir, "cumulative_delta.png")
        cum_delta_metrics = evaluate_image(cum_delta_tmp)
        cum_noise_score = cum_delta_metrics["noise_floor_score"]

        # Collect metrics
        metrics_q = {
            "quality": q,
            "jxl_size_bytes": jxl_size,
            "jxl_bpp": jxl_size * 8 / (residual.shape[0] * residual.shape[1]),
            "residual_psnr_db": float(res_psnr),
            "reconstruction_psnr_db": float(recon_psnr) if recon_psnr is not None else None,
            "delta_noise_floor_score": delta_noise_score,
            "delta_edge_correlation": delta_edge_corr,
            "delta_artifact_fft_8": delta_artifact_8,
            "delta_artifact_fft_32": delta_artifact_32,
            "delta_mean_abs": float(np.mean(np.abs(delta))) if delta is not None else None,
            "cumulative_noise_floor_score": float(cum_noise_score),
            "cumulative_edge_correlation": float(cum_edge_corr) if cum_edge_corr is not None else None,
            "cumulative_mean_abs": float(np.mean(cum_abs)),
        }
        sweep_metrics[q] = metrics_q

        # Save per-quality stats
        with open(os.path.join(q_dir, "stats.json"), "w") as f:
            json.dump(metrics_q, f, indent=2, cls=NumpyEncoder)

        # Clean up temp roundtrip file
        if os.path.exists(rt_path):
            os.unlink(rt_path)

        # Print summary
        parts = [f"size={jxl_size:6d}B", f"bpp={metrics_q['jxl_bpp']:.3f}"]
        parts.append(f"res_psnr={res_psnr:.1f}dB")
        if recon_psnr is not None:
            parts.append(f"recon_psnr={recon_psnr:.1f}dB")
        if delta_noise_score is not None:
            parts.append(f"step_noise={delta_noise_score:.3f}")
        parts.append(f"cum_noise={cum_noise_score:.3f}")
        if cum_edge_corr is not None:
            parts.append(f"cum_edge={cum_edge_corr:.3f}")
        print("  ".join(parts))

        prev_quality = q

    # Generate charts
    if HAS_MPL:
        generate_charts(sweep_metrics, quality_levels, sweep_dir, has_three_signal)

    # Generate summary strip
    generate_summary_strip(residual, roundtrips, quality_levels, sweep_dir)

    # Save experiment metadata
    metadata = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run": run_dir,
        "source_residual": sources["residual"],
        "has_three_signal": has_three_signal,
        "params": {
            "jxl_config": config,
            "quality_levels": quality_levels,
            "predictor": "lanczos3",
        },
        "notes": "",
        "conclusions": [],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)

    # Save all metrics
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(sweep_metrics, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {out_dir}")
    return sweep_metrics


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def generate_charts(sweep_metrics: dict, quality_levels: list, sweep_dir: str, has_three_signal: bool):
    """Generate rate-distortion and layer classification charts."""
    qs = sorted(sweep_metrics.keys(), reverse=True)
    sizes = [sweep_metrics[q]["jxl_size_bytes"] / 1024 for q in qs]
    bpps = [sweep_metrics[q]["jxl_bpp"] for q in qs]
    res_psnrs = [sweep_metrics[q]["residual_psnr_db"] for q in qs]

    # Rate-distortion curve
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(qs, sizes, "b-o", label="JXL size (KB)", linewidth=2)
    ax1.set_xlabel("JXL Quality")
    ax1.set_ylabel("Size (KB)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(qs, res_psnrs, "r-s", label="Residual PSNR (dB)", linewidth=2)
    ax2.set_ylabel("PSNR (dB)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    if has_three_signal:
        recon_psnrs = [sweep_metrics[q].get("reconstruction_psnr_db") for q in qs]
        if all(v is not None for v in recon_psnrs):
            ax2.plot(qs, recon_psnrs, "g-^", label="Recon PSNR (dB)", linewidth=2)

    ax1.set_title("JXL Quality Sweep — Rate-Distortion")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, "rd_curve.png"), dpi=150)
    plt.close()

    # Layer classification chart (step deltas)
    delta_qs = [q for q in qs if sweep_metrics[q]["delta_noise_floor_score"] is not None]
    if not delta_qs:
        return

    fig, axes = plt.subplots(2 if has_three_signal else 1, 1,
                              figsize=(10, 8 if has_three_signal else 5),
                              sharex=True)
    if not has_three_signal:
        axes = [axes]

    # Panel 1: Step noise score + artifact FFT
    ax = axes[0]
    noise_scores = [sweep_metrics[q]["delta_noise_floor_score"] for q in delta_qs]
    ax.plot(delta_qs, noise_scores, "b-o", label="Step delta noise score", linewidth=2)
    ax.axhline(y=0.85, color="b", linestyle="--", alpha=0.5, label="Noise threshold (0.85)")

    artifact_8 = [sweep_metrics[q]["delta_artifact_fft_8"] for q in delta_qs]
    artifact_32 = [sweep_metrics[q]["delta_artifact_fft_32"] for q in delta_qs]
    if any(v is not None for v in artifact_8):
        ax.plot(delta_qs, [v or 1.0 for v in artifact_8], "m-v", label="Artifact FFT 1/8", alpha=0.7)
    if any(v is not None for v in artifact_32):
        ax.plot(delta_qs, [v or 1.0 for v in artifact_32], "c-v", label="Artifact FFT 1/32", alpha=0.7)
    ax.axhline(y=1.5, color="m", linestyle="--", alpha=0.3, label="Artifact threshold (1.5)")

    ax.set_ylabel("Score / Ratio")
    ax.set_title("Layer Classification — Step Deltas")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Step edge correlation
    if has_three_signal:
        ax = axes[1]
        edge_corrs = [sweep_metrics[q]["delta_edge_correlation"] for q in delta_qs]
        if any(v is not None for v in edge_corrs):
            ax.plot(delta_qs, [v or 0.0 for v in edge_corrs], "g-s", label="Step edge correlation", linewidth=2)
        ax.axhline(y=0.3, color="g", linestyle="--", alpha=0.5, label="Structure threshold (0.3)")
        ax.set_ylabel("Correlation")
        ax.set_xlabel("JXL Quality")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("JXL Quality")

    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, "layer_classification.png"), dpi=150)
    plt.close()

    # Cumulative chart — noise score + edge correlation vs quality
    fig, axes = plt.subplots(2 if has_three_signal else 1, 1,
                              figsize=(10, 8 if has_three_signal else 5),
                              sharex=True)
    if not has_three_signal:
        axes = [axes]

    ax = axes[0]
    cum_noise = [sweep_metrics[q]["cumulative_noise_floor_score"] for q in qs]
    cum_abs = [sweep_metrics[q]["cumulative_mean_abs"] for q in qs]
    ax.plot(qs, cum_noise, "b-o", label="Cumulative noise score", linewidth=2)
    ax.set_ylabel("Noise Score", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax.set_title("Cumulative Removal — Total Discarded from Original Residual")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(qs, cum_abs, "r-s", label="Cumulative mean |delta|", linewidth=2, alpha=0.7)
    ax2.set_ylabel("Mean |Delta|", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    if has_three_signal:
        ax = axes[1]
        cum_edge = [sweep_metrics[q].get("cumulative_edge_correlation") for q in qs]
        if any(v is not None for v in cum_edge):
            ax.plot(qs, [v or 0.0 for v in cum_edge], "g-s", label="Cumulative edge correlation", linewidth=2)
        ax.axhline(y=0.3, color="g", linestyle="--", alpha=0.5, label="Structure threshold (0.3)")
        ax.set_ylabel("Correlation")
        ax.set_xlabel("JXL Quality")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("JXL Quality")

    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, "cumulative_classification.png"), dpi=150)
    plt.close()


def generate_summary_strip(
    residual: np.ndarray,
    roundtrips: dict,
    quality_levels: list,
    sweep_dir: str,
):
    """Generate horizontal summary strip: residual | q90 | q70 | q50 | q30."""
    # Pick representative quality levels
    show_qs = [q for q in [90, 70, 50, 30] if q in roundtrips]
    if not show_qs:
        show_qs = sorted(roundtrips.keys(), reverse=True)[:4]

    # Crop to 256x256 center for readability
    h, w = residual.shape
    ch, cw = h // 2, w // 2
    crop = 128
    sl = (slice(ch - crop, ch + crop), slice(cw - crop, cw + crop))

    images = [residual[sl]]
    labels = ["Original"]
    for q in show_qs:
        images.append(roundtrips[q][sl])
        labels.append(f"Q{q}")

    n = len(images)
    strip = np.zeros((256 + 20, 256 * n + 4 * (n - 1), 3), dtype=np.uint8)
    strip[:] = 255  # white background

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * (256 + 4)
        # Convert grayscale to RGB
        strip[20:20+256, x:x+256, 0] = img
        strip[20:20+256, x:x+256, 1] = img
        strip[20:20+256, x:x+256, 2] = img

    Image.fromarray(strip).save(os.path.join(sweep_dir, "summary_strip.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JXL quality sweep for residual decomposition")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to an origami encode run (with --debug-images)")
    parser.add_argument("--config", choices=["default", "analysis", "modular"], default="default",
                        help="JXL config: default, analysis (no perceptual), modular")
    parser.add_argument("--qualities", type=str, default="100,95,90,80,70,60,50,40,30,20,10",
                        help="Comma-separated quality levels")
    parser.add_argument("--family", type=str, default=None,
                        help="Specific family prefix to analyze")
    parser.add_argument("--output-dir", type=str, default="evals/analysis/noise_floor",
                        help="Base output directory")
    args = parser.parse_args()

    quality_levels = [int(q) for q in args.qualities.split(",")]

    # Verify cjxl/djxl are available
    for tool in ["cjxl", "djxl"]:
        if shutil.which(tool) is None:
            print(f"Error: {tool} not found. Install with: brew install jpeg-xl")
            sys.exit(1)

    run_sweep(
        run_dir=args.run_dir,
        quality_levels=quality_levels,
        config=args.config,
        family=args.family,
        output_base=args.output_dir,
    )


if __name__ == "__main__":
    main()
