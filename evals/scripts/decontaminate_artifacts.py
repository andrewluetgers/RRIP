"""
JPEG Artifact Decontamination (Phase 4)
========================================

Removes periodic JPEG block grid artifacts from residuals using frequency-domain
notch filtering. The 8x8 block grid from L2 JPEG bleeds through 4x upsample as
a period-32 pattern in L0 residuals.

Usage:
    # Decontaminate a residual image
    uv run python evals/scripts/decontaminate_artifacts.py \
        --run-dir evals/runs/rs_444_b95_l0q50

    # Custom notch bandwidth
    uv run python evals/scripts/decontaminate_artifacts.py \
        --run-dir evals/runs/rs_444_b95_l0q50 --bandwidth 3

Output:
    evals/analysis/noise_floor/{experiment_id}/artifacts/
    ├── before_fft.png
    ├── after_fft.png
    ├── removed_artifacts.png
    ├── residual_clean.png
    ├── autocorrelation_comparison.png
    └── metrics.json
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift

sys.path.insert(0, os.path.dirname(__file__))
from eval_noise_floor import evaluate_image, NumpyEncoder
from jxl_quality_sweep import find_source_images

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def create_notch_mask(shape: tuple, period: int, bandwidth: int = 2) -> np.ndarray:
    """Create a notch filter mask that suppresses a specific frequency and its harmonics.
    Works in unshifted FFT space."""
    h, w = shape
    mask = np.ones((h, w), dtype=np.float64)

    # Frequency indices for this period (and harmonics)
    for harmonic in range(1, min(8, w // (2 * period) + 1)):
        freq_w = harmonic * (w // period)
        freq_h = harmonic * (h // period)

        if freq_w < w:
            # Suppress vertical artifact lines (column frequencies)
            for bw in range(-bandwidth, bandwidth + 1):
                col = (freq_w + bw) % w
                mask[:, col] *= 0.0
                # Mirror
                col_mirror = (w - freq_w + bw) % w
                mask[:, col_mirror] *= 0.0

        if freq_h < h:
            # Suppress horizontal artifact lines (row frequencies)
            for bw in range(-bandwidth, bandwidth + 1):
                row = (freq_h + bw) % h
                mask[row, :] *= 0.0
                row_mirror = (h - freq_h + bw) % h
                mask[row_mirror, :] *= 0.0

    return mask


def decontaminate(
    residual: np.ndarray,
    periods: list = None,
    bandwidth: int = 2,
) -> tuple:
    """Remove JPEG artifacts via frequency-domain notch filtering.
    Returns (cleaned_residual, removed_artifacts) both as uint8."""
    if periods is None:
        periods = [8, 16, 32, 64]  # JPEG artifact harmonics in L0 space

    centered = residual.astype(np.float64) - residual.mean()
    F = fft2(centered)

    # Apply notch filters for each artifact period
    combined_mask = np.ones_like(F, dtype=np.float64)
    for period in periods:
        notch = create_notch_mask(F.shape, period, bandwidth)
        combined_mask *= notch

    F_clean = F * combined_mask
    clean_centered = np.real(ifft2(F_clean))
    clean = np.clip(clean_centered + residual.mean(), 0, 255).astype(np.uint8)

    # What was removed
    removed = residual.astype(np.float32) - clean.astype(np.float32)
    removed_vis = np.clip(removed + 128, 0, 255).astype(np.uint8)

    return clean, removed_vis, combined_mask


def compute_autocorrelation_profile(data: np.ndarray, max_lag: int = 64) -> list:
    """Compute autocorrelation at lags 1..max_lag (horizontal only for speed)."""
    centered = data.astype(np.float64) - data.mean()
    var = np.var(centered)
    if var < 1e-10:
        return [0.0] * max_lag
    return [float(np.mean(centered[:, lag:] * centered[:, :-lag]) / var)
            for lag in range(1, max_lag + 1)]


def run_decontamination(
    run_dir: str,
    family: str = None,
    bandwidth: int = 2,
    output_base: str = "evals/analysis/noise_floor",
):
    sources = find_source_images(run_dir, family)
    run_name = os.path.basename(run_dir)
    experiment_id = f"decontam_{run_name}"

    residual = np.array(Image.open(sources["residual"]).convert("L"))

    print(f"Decontaminating: {sources['residual']}")
    print(f"  Shape: {residual.shape}, bandwidth: {bandwidth}")

    # Run decontamination
    clean, removed_vis, mask = decontaminate(residual, bandwidth=bandwidth)

    # Output directory
    out_dir = os.path.join(output_base, experiment_id, "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    # Save images
    Image.fromarray(clean).save(os.path.join(out_dir, "residual_clean.png"))
    Image.fromarray(removed_vis).save(os.path.join(out_dir, "removed_artifacts.png"))

    # FFT visualizations
    F_before = np.abs(fftshift(fft2(residual.astype(np.float64) - residual.mean())))
    F_after = np.abs(fftshift(fft2(clean.astype(np.float64) - clean.mean())))

    # Log scale for visibility
    F_before_log = np.log1p(F_before)
    F_after_log = np.log1p(F_after)
    vmax = max(F_before_log.max(), 1)

    Image.fromarray((F_before_log / vmax * 255).astype(np.uint8)).save(
        os.path.join(out_dir, "before_fft.png"))
    Image.fromarray((F_after_log / vmax * 255).astype(np.uint8)).save(
        os.path.join(out_dir, "after_fft.png"))

    # Autocorrelation comparison
    ac_before = compute_autocorrelation_profile(residual)
    ac_after = compute_autocorrelation_profile(clean)

    # Noise-floor scores
    residual_path_tmp = os.path.join(out_dir, "_residual_before.png")
    Image.fromarray(residual).save(residual_path_tmp)
    before_metrics = evaluate_image(residual_path_tmp)
    os.unlink(residual_path_tmp)

    after_metrics = evaluate_image(os.path.join(out_dir, "residual_clean.png"))

    # Metrics
    metrics = {
        "before_noise_score": before_metrics["noise_floor_score"],
        "after_noise_score": after_metrics["noise_floor_score"],
        "before_autocorr_lag8": ac_before[7] if len(ac_before) > 7 else None,
        "after_autocorr_lag8": ac_after[7] if len(ac_after) > 7 else None,
        "before_autocorr_lag32": ac_before[31] if len(ac_before) > 31 else None,
        "after_autocorr_lag32": ac_after[31] if len(ac_after) > 31 else None,
        "removed_energy_mean": float(np.mean(np.abs(removed_vis.astype(np.float32) - 128))),
        "removed_energy_max": float(np.max(np.abs(removed_vis.astype(np.float32) - 128))),
        "bandwidth": bandwidth,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    # Print summary
    print(f"\n  Noise score: {metrics['before_noise_score']:.3f} → {metrics['after_noise_score']:.3f}")
    print(f"  Autocorr lag-8:  {metrics['before_autocorr_lag8']:.4f} → {metrics['after_autocorr_lag8']:.4f}")
    print(f"  Autocorr lag-32: {metrics['before_autocorr_lag32']:.4f} → {metrics['after_autocorr_lag32']:.4f}")
    print(f"  Removed energy: mean={metrics['removed_energy_mean']:.2f}, max={metrics['removed_energy_max']:.1f}")

    # Generate autocorrelation comparison chart
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 5))
        lags = list(range(1, len(ac_before) + 1))
        ax.plot(lags, ac_before, "r-", label="Before decontamination", linewidth=1.5)
        ax.plot(lags, ac_after, "b-", label="After decontamination", linewidth=1.5)
        for lag in [8, 16, 32, 64]:
            ax.axvline(x=lag, color="gray", linestyle="--", alpha=0.3)
            ax.text(lag, ax.get_ylim()[1] * 0.95, f"{lag}", ha="center", fontsize=8, color="gray")
        ax.set_xlabel("Lag (pixels)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Autocorrelation Before/After JPEG Artifact Decontamination")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "autocorrelation_comparison.png"), dpi=150)
        plt.close()

    # Write metadata.json at experiment root for viewer discovery
    exp_root = os.path.dirname(out_dir)
    meta = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run": run_dir,
        "params": {"bandwidth": bandwidth},
        "notes": "",
    }
    with open(os.path.join(exp_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="JPEG artifact decontamination")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--bandwidth", type=int, default=2, help="Notch filter bandwidth")
    parser.add_argument("--output-dir", type=str, default="evals/analysis/noise_floor")
    args = parser.parse_args()

    run_decontamination(
        args.run_dir, args.family, args.bandwidth, args.output_dir
    )


if __name__ == "__main__":
    main()
