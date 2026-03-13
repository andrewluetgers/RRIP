"""
Noise-Floor Evaluation Framework (Phase 1)
===========================================

Quantifies how much of an image (especially a residual) is incompressible noise
vs compressible structure. This is the foundation for all subsequent residual
denoising experiments.

Usage:
    # Evaluate a single residual image
    uv run python evals/scripts/eval_noise_floor.py --image path/to/residual.png

    # Evaluate all fused residuals from debug runs
    uv run python evals/scripts/eval_noise_floor.py --glob "evals/runs/*/compress/*fused_residual_centered.png"

    # Run with synthetic controls for validation
    uv run python evals/scripts/eval_noise_floor.py --validate

    # Output JSON report instead of text
    uv run python evals/scripts/eval_noise_floor.py --image path.png --json

Metrics:
    1. Incompressibility ratio   — LZ4 compressed / raw size (>0.95 = pure noise)
    2. Spatial autocorrelation   — Mean abs correlation at lags 1-5 (<0.05 = no structure)
    3. Power spectrum flatness   — Geometric/arithmetic mean of PSD (>0.8 = white noise)
    4. Block variance uniformity — CV of variance across 8x8 blocks (<0.3 = uniform noise)
    5. Distribution normality    — KS test vs Laplacian fit (p>0.05 = Laplacian noise)
    6. Run-length entropy        — Entropy of run lengths vs theoretical max (~1.0 = random)
    7. JPEG artifact energy      — FFT peak ratios at 1/8 and 1/32 freq (>1.5 = artifacts)

Composite score: 0.0 (fully structured) to 1.0 (pure incompressible noise).

Output:
    evals/analysis/noise_floor/
    ├── noise_floor_report.json   # Per-image metrics + composite scores
    ├── noise_floor_summary.txt   # Human-readable summary
    └── validation/               # Synthetic control results (if --validate)
"""

import argparse
import glob
import json
import os
import sys
import time

import lz4.frame
import numpy as np
from PIL import Image
from scipy import stats as scipy_stats
from scipy import signal
from scipy.fft import fft2, fftshift

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def incompressibility_ratio(data: np.ndarray) -> float:
    """LZ4 compressed size / raw size. Near 1.0 = incompressible (noise-like)."""
    raw = data.tobytes()
    compressed = lz4.frame.compress(raw, compression_level=0)
    return len(compressed) / len(raw)


def spatial_autocorrelation(data: np.ndarray, max_lag: int = 5) -> dict:
    """Normalized autocorrelation at lags 1..max_lag.
    Returns mean absolute correlation and per-lag values."""
    centered = data.astype(np.float32) - data.mean()
    var = np.var(centered)
    if var < 1e-10:
        return {"mean_abs": 0.0, "per_lag": [0.0] * max_lag}

    per_lag = []
    for lag in range(1, max_lag + 1):
        # Horizontal autocorrelation
        h = np.mean(centered[:, lag:] * centered[:, :-lag]) / var
        # Vertical autocorrelation
        v = np.mean(centered[lag:, :] * centered[:-lag, :]) / var
        per_lag.append(float((abs(h) + abs(v)) / 2))

    return {
        "mean_abs": float(np.mean(per_lag)),
        "per_lag": per_lag,
    }


def power_spectrum_flatness(data: np.ndarray) -> float:
    """Spectral flatness: geometric mean / arithmetic mean of power spectrum.
    1.0 = perfectly flat (white noise), 0.0 = tonal (structured)."""
    centered = data.astype(np.float64) - data.mean()
    F = np.abs(fft2(centered)) ** 2
    # Exclude DC component
    F_flat = F.ravel()
    F_flat = F_flat[1:]  # drop DC
    F_flat = F_flat[F_flat > 0]  # drop zeros for log

    if len(F_flat) == 0:
        return 0.0

    log_mean = np.mean(np.log(F_flat))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(F_flat)

    if arith_mean < 1e-10:
        return 0.0

    return float(geo_mean / arith_mean)


def block_variance_uniformity(data: np.ndarray, block_size: int = 8) -> float:
    """Coefficient of variation of variance across non-overlapping blocks.
    Low CV (<0.3) = uniform noise. High CV = spatially varying (structured)."""
    h, w = data.shape
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        return 0.0

    cropped = data[:bh * block_size, :bw * block_size].astype(np.float32)
    blocks = cropped.reshape(bh, block_size, bw, block_size)
    block_vars = np.var(blocks, axis=(1, 3))

    mean_var = np.mean(block_vars)
    if mean_var < 1e-10:
        return 0.0

    return float(np.std(block_vars) / mean_var)


def distribution_normality(data: np.ndarray) -> dict:
    """KS test of (data - 128) against best-fit Laplacian.
    High p-value = consistent with Laplacian noise model."""
    centered = data.astype(np.float64).ravel() - 128.0
    loc = np.median(centered)
    b = np.mean(np.abs(centered - loc))  # Laplacian scale parameter
    if b < 1e-10:
        b = 1e-10

    ks_stat, p_value = scipy_stats.kstest(
        centered, "laplace", args=(loc, b), N=min(len(centered), 100000)
    )
    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "laplacian_loc": float(loc),
        "laplacian_scale": float(b),
    }


def run_length_entropy(data: np.ndarray) -> float:
    """Entropy of run lengths of above/below median, normalized to [0, 1].
    Near 1.0 = random (geometric distribution of run lengths).
    Low = structured (long runs of similar values)."""
    flat = data.ravel()
    median_val = np.median(flat)
    above = flat >= median_val

    # Compute run lengths
    changes = np.diff(above.astype(np.int8))
    change_indices = np.where(changes != 0)[0]

    if len(change_indices) == 0:
        return 0.0

    run_lengths = np.diff(np.concatenate([[-1], change_indices, [len(flat) - 1]]))

    # Compute entropy of run-length distribution
    max_rl = min(int(run_lengths.max()), 256)
    counts = np.bincount(np.clip(run_lengths, 0, max_rl), minlength=max_rl + 1)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    # Theoretical max entropy for geometric distribution with p=0.5
    # For a fair coin, expected run length = 2, entropy ≈ 2.0 bits
    # Normalize by this theoretical value
    p = 0.5
    theoretical_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) / p
    # More practically: entropy of geometric(0.5) = 2.0 bits
    theoretical_entropy = 2.0

    return float(min(entropy / theoretical_entropy, 1.0))


def jpeg_artifact_energy(data: np.ndarray) -> dict:
    """FFT peak ratios at JPEG artifact frequencies (1/8, 1/32).
    Ratio > 1.5 indicates JPEG artifacts present."""
    centered = data.astype(np.float64) - data.mean()
    F = np.abs(fft2(centered))
    h, w = F.shape

    results = {}
    for period, name in [(8, "period_8"), (32, "period_32")]:
        # Check both horizontal and vertical artifact frequencies
        freq_idx_h = w // period
        freq_idx_v = h // period

        if freq_idx_h == 0 or freq_idx_v == 0:
            results[name] = {"ratio": 1.0, "detected": False}
            continue

        # Peak energy at artifact frequency (narrow band ±1)
        peak_energy = 0.0
        baseline_energy = 0.0
        n_peak = 0
        n_baseline = 0

        for fi in range(max(1, freq_idx_h - 1), min(w // 2, freq_idx_h + 2)):
            peak_energy += np.mean(F[:, fi]) + np.mean(F[fi, :])
            n_peak += 2

        # Baseline: nearby frequencies (exclude artifact harmonics)
        for fi in range(max(1, freq_idx_h - 5), max(1, freq_idx_h - 2)):
            baseline_energy += np.mean(F[:, fi]) + np.mean(F[fi, :])
            n_baseline += 2
        for fi in range(min(w // 2, freq_idx_h + 3), min(w // 2, freq_idx_h + 6)):
            baseline_energy += np.mean(F[:, fi]) + np.mean(F[fi, :])
            n_baseline += 2

        if n_peak > 0:
            peak_energy /= n_peak
        if n_baseline > 0:
            baseline_energy /= n_baseline

        ratio = peak_energy / baseline_energy if baseline_energy > 1e-10 else 1.0
        results[name] = {
            "ratio": float(ratio),
            "detected": ratio > 1.5,
        }

    return results


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_noise_floor_score(metrics: dict) -> float:
    """Combine individual metrics into a 0-1 noise fraction score.
    1.0 = pure incompressible noise, 0.0 = fully structured signal.

    Weights reflect how diagnostic each metric is:
    - Incompressibility ratio is the most direct measure
    - Autocorrelation and spectral flatness measure spatial structure
    - Block variance and run-length capture non-uniformity
    - Distribution fit is a soft signal (Laplacian is expected for both noise and residuals)
    """
    scores = {}

    # 1. Incompressibility: 0.95+ → noise-like
    ir = metrics["incompressibility_ratio"]
    scores["incompressibility"] = min(max((ir - 0.5) / 0.5, 0.0), 1.0)

    # 2. Autocorrelation: low → noise-like
    ac = metrics["spatial_autocorrelation"]["mean_abs"]
    scores["autocorrelation"] = min(max(1.0 - ac / 0.3, 0.0), 1.0)

    # 3. Spectral flatness: high → noise-like
    sf = metrics["power_spectrum_flatness"]
    scores["spectral_flatness"] = min(max(sf / 0.8, 0.0), 1.0)

    # 4. Block variance uniformity: low CV → noise-like
    bv = metrics["block_variance_uniformity"]
    scores["block_uniformity"] = min(max(1.0 - bv / 1.5, 0.0), 1.0)

    # 5. Distribution normality: high p-value → noise-like
    p = metrics["distribution_normality"]["p_value"]
    scores["distribution_fit"] = min(max(p / 0.05, 0.0), 1.0)

    # 6. Run-length entropy: high → noise-like
    rle = metrics["run_length_entropy"]
    scores["run_length"] = rle

    # Weighted combination
    weights = {
        "incompressibility": 0.25,
        "autocorrelation": 0.20,
        "spectral_flatness": 0.20,
        "block_uniformity": 0.15,
        "run_length": 0.10,
        "distribution_fit": 0.10,
    }

    composite = sum(scores[k] * weights[k] for k in weights)

    return float(composite), scores


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_image(img_path: str) -> dict:
    """Run all noise-floor metrics on a single grayscale image."""
    img = Image.open(img_path).convert("L")
    data = np.array(img, dtype=np.uint8)

    t0 = time.time()

    metrics = {
        "image": os.path.basename(img_path),
        "path": img_path,
        "shape": list(data.shape),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "incompressibility_ratio": incompressibility_ratio(data),
        "spatial_autocorrelation": spatial_autocorrelation(data),
        "power_spectrum_flatness": power_spectrum_flatness(data),
        "block_variance_uniformity": block_variance_uniformity(data),
        "distribution_normality": distribution_normality(data),
        "run_length_entropy": run_length_entropy(data),
        "jpeg_artifact_energy": jpeg_artifact_energy(data),
    }

    composite, sub_scores = compute_noise_floor_score(metrics)
    metrics["noise_floor_score"] = composite
    metrics["sub_scores"] = sub_scores
    metrics["eval_time_ms"] = float((time.time() - t0) * 1000)

    return metrics


def generate_synthetic_noise(shape: tuple, std: float = 6.5, seed: int = 42) -> np.ndarray:
    """Generate synthetic Laplacian noise centered at 128 (matching real residuals)."""
    rng = np.random.default_rng(seed)
    b = std / np.sqrt(2)  # Laplacian scale from std
    noise = rng.laplace(loc=0, scale=b, size=shape)
    return np.clip(noise + 128, 0, 255).astype(np.uint8)


def run_validation(output_dir: str) -> list:
    """Run positive and negative controls to validate the scoring system."""
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    results = []

    # Positive control: synthetic Laplacian noise (should score ~1.0)
    print("  Validating: synthetic Laplacian noise (expected ~1.0)...")
    noise = generate_synthetic_noise((1024, 1024), std=6.5)
    noise_path = os.path.join(val_dir, "synthetic_laplacian_noise.png")
    Image.fromarray(noise).save(noise_path)
    m = evaluate_image(noise_path)
    m["control"] = "positive_laplacian_noise"
    results.append(m)
    print(f"    Score: {m['noise_floor_score']:.3f}")

    # Positive control: uniform random (should score ~1.0)
    print("  Validating: uniform random noise (expected ~1.0)...")
    rng = np.random.default_rng(42)
    uniform = rng.integers(0, 256, size=(1024, 1024), dtype=np.uint8)
    uniform_path = os.path.join(val_dir, "synthetic_uniform_noise.png")
    Image.fromarray(uniform).save(uniform_path)
    m = evaluate_image(uniform_path)
    m["control"] = "positive_uniform_noise"
    results.append(m)
    print(f"    Score: {m['noise_floor_score']:.3f}")

    # Negative control: gradient (should score ~0.0-0.3)
    print("  Validating: smooth gradient (expected ~0.0-0.3)...")
    gradient = np.tile(np.linspace(0, 255, 1024).astype(np.uint8), (1024, 1))
    grad_path = os.path.join(val_dir, "synthetic_gradient.png")
    Image.fromarray(gradient).save(grad_path)
    m = evaluate_image(grad_path)
    m["control"] = "negative_gradient"
    results.append(m)
    print(f"    Score: {m['noise_floor_score']:.3f}")

    # Negative control: checkerboard pattern (structured, should score low)
    print("  Validating: checkerboard (expected ~0.2-0.5)...")
    checker = np.zeros((1024, 1024), dtype=np.uint8)
    for i in range(0, 1024, 32):
        for j in range(0, 1024, 32):
            if (i // 32 + j // 32) % 2 == 0:
                checker[i:i+32, j:j+32] = 200
            else:
                checker[i:i+32, j:j+32] = 56
    checker_path = os.path.join(val_dir, "synthetic_checkerboard.png")
    Image.fromarray(checker).save(checker_path)
    m = evaluate_image(checker_path)
    m["control"] = "negative_checkerboard"
    results.append(m)
    print(f"    Score: {m['noise_floor_score']:.3f}")

    # Negative control: edges (high structure, should score low)
    print("  Validating: edge pattern (expected ~0.1-0.3)...")
    edges = np.full((1024, 1024), 128, dtype=np.uint8)
    # Add horizontal and vertical edges
    for i in range(0, 1024, 64):
        edges[i:i+2, :] = 200
        edges[:, i:i+2] = 200
    edge_path = os.path.join(val_dir, "synthetic_edges.png")
    Image.fromarray(edges).save(edge_path)
    m = evaluate_image(edge_path)
    m["control"] = "negative_edges"
    results.append(m)
    print(f"    Score: {m['noise_floor_score']:.3f}")

    # Save validation report
    with open(os.path.join(val_dir, "validation_report.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def format_report(all_metrics: list) -> str:
    """Format metrics as human-readable text report."""
    lines = []
    lines.append("=" * 72)
    lines.append("NOISE-FLOOR EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    for m in all_metrics:
        lines.append(f"Image: {m['image']}")
        lines.append(f"  Shape: {m['shape']}, Mean: {m['mean']:.1f}, Std: {m['std']:.2f}")
        lines.append(f"  Noise-floor score: {m['noise_floor_score']:.3f}")
        lines.append(f"  Sub-scores:")
        for k, v in m.get("sub_scores", {}).items():
            lines.append(f"    {k:25s} {v:.3f}")
        lines.append(f"  Metrics:")
        lines.append(f"    Incompressibility ratio:  {m['incompressibility_ratio']:.4f}")
        ac = m["spatial_autocorrelation"]
        lines.append(f"    Autocorrelation mean:     {ac['mean_abs']:.4f}")
        lines.append(f"      Per-lag: {', '.join(f'{v:.3f}' for v in ac['per_lag'])}")
        lines.append(f"    Spectral flatness:        {m['power_spectrum_flatness']:.4f}")
        lines.append(f"    Block var uniformity (CV): {m['block_variance_uniformity']:.4f}")
        dn = m["distribution_normality"]
        lines.append(f"    Distribution KS stat:     {dn['ks_statistic']:.4f} (p={dn['p_value']:.4f})")
        lines.append(f"      Laplacian fit: loc={dn['laplacian_loc']:.2f}, scale={dn['laplacian_scale']:.2f}")
        lines.append(f"    Run-length entropy:       {m['run_length_entropy']:.4f}")
        ae = m["jpeg_artifact_energy"]
        for period_name, info in ae.items():
            detected = "YES" if info["detected"] else "no"
            lines.append(f"    JPEG artifact {period_name}: ratio={info['ratio']:.3f} ({detected})")
        lines.append(f"  Eval time: {m.get('eval_time_ms', 0):.0f} ms")
        lines.append("")

    # Summary table
    if len(all_metrics) > 1:
        lines.append("-" * 72)
        lines.append("SUMMARY")
        lines.append("-" * 72)
        scores = [m["noise_floor_score"] for m in all_metrics]
        lines.append(f"  Images evaluated: {len(all_metrics)}")
        lines.append(f"  Noise-floor score: mean={np.mean(scores):.3f}, "
                     f"std={np.std(scores):.3f}, "
                     f"min={np.min(scores):.3f}, max={np.max(scores):.3f}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Noise-floor evaluation for residual images")
    parser.add_argument("--image", type=str, help="Single image to evaluate")
    parser.add_argument("--glob", type=str, help="Glob pattern for multiple images")
    parser.add_argument("--validate", action="store_true", help="Run synthetic controls")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--output-dir", type=str, default="evals/analysis/noise_floor",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []

    # Collect images to evaluate
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.glob:
        image_paths.extend(sorted(glob.glob(args.glob)))

    # Default: find all fused residuals in evals/runs/
    if not image_paths and not args.validate:
        default_pattern = "evals/runs/*/compress/*fused_residual_centered.png"
        image_paths = sorted(glob.glob(default_pattern))
        if not image_paths:
            print(f"No images found with default pattern: {default_pattern}")
            print("Use --image or --glob to specify input images, or --validate for controls.")
            sys.exit(1)

    # Evaluate images
    if image_paths:
        print(f"Evaluating {len(image_paths)} image(s)...")
        for i, path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {os.path.basename(path)}...", end=" ", flush=True)
            try:
                m = evaluate_image(path)
                all_metrics.append(m)
                print(f"score={m['noise_floor_score']:.3f} ({m['eval_time_ms']:.0f}ms)")
            except Exception as e:
                print(f"ERROR: {e}")

    # Run validation controls
    if args.validate:
        print("\nRunning validation controls...")
        val_results = run_validation(args.output_dir)
        all_metrics.extend(val_results)

    if not all_metrics:
        print("No images evaluated.")
        sys.exit(1)

    # Output
    if args.json:
        print(json.dumps(all_metrics, indent=2, cls=NumpyEncoder))
    else:
        report = format_report(all_metrics)
        print(report)

    # Save results
    json_path = os.path.join(args.output_dir, "noise_floor_report.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2, cls=NumpyEncoder)
    print(f"Saved JSON report: {json_path}")

    txt_path = os.path.join(args.output_dir, "noise_floor_summary.txt")
    with open(txt_path, "w") as f:
        f.write(format_report(all_metrics))
    print(f"Saved text report: {txt_path}")


if __name__ == "__main__":
    main()