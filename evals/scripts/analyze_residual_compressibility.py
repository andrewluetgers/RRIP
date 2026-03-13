"""
Residual Compressibility Analysis
==================================

Answers the fundamental question: Is there enough spatial structure in our
residuals for a learned codec to exploit, or are they noise-like (in which
case JPEG is near-optimal and we should focus on better prediction instead)?

Usage:
    uv run python evals/scripts/analyze_residual_compressibility.py

Inputs:
    - Fused residual PNGs from evals/runs/*/compress/*fused_residual_centered.png
    - Also analyzes per-tile residuals (*_residual_centered.png)
    - Compares lanczos3 vs SR model residuals if both exist

Outputs:
    evals/analysis/residual_compressibility/
    ├── report.txt                          # Summary statistics
    ├── spatial_autocorrelation.png         # Autocorrelation vs lag distance
    ├── pixel_distribution.png             # Histogram + fitted Laplacian/Gaussian
    ├── power_spectrum.png                 # Radial power spectrum (flat = noise)
    ├── block_sparsity.png                # % of 8x8 blocks below threshold vs threshold
    ├── spatial_variance_map.png          # Where is the residual energy concentrated?
    ├── entropy_vs_jpeg.png               # Theoretical entropy vs JPEG actual bits
    └── cross_image_similarity.png        # How similar are residuals across images?

Analysis:
    1. Spatial Autocorrelation
       - Compute normalized autocorrelation at lags 1-64 pixels
       - If autocorrelation drops to ~0 within 1-2 pixels → noise-like → JPEG is fine
       - If significant autocorrelation at 4-16px → spatial structure → learned model can exploit
       - Compare: natural image autocorrelation vs our residuals vs white noise

    2. Pixel Value Distribution
       - Histogram of (pixel - 128) values
       - Fit Laplacian and Gaussian distributions, compare KL divergence
       - Compute empirical entropy (bits per pixel)
       - Compare empirical entropy to: JPEG actual bits, PNG lossless bits

    3. Power Spectrum Analysis
       - 2D FFT → radial average → power vs spatial frequency
       - Flat spectrum = white noise (no exploitable structure)
       - Rolloff at high frequencies = smooth structure (exploitable)
       - Compare to natural image power spectrum (1/f falloff)

    4. Block Sparsity Analysis
       - For 8x8 blocks: compute mean |pixel - 128| per block
       - Plot CDF: what fraction of blocks are below threshold T?
       - If 50%+ blocks are near-zero → block-skip mask saves significant bits
       - Compute: skip mask overhead (bits) vs savings (skipped blocks)

    5. Spatial Variance Map
       - Divide into 16x16 or 32x32 regions, compute local variance
       - Visualize: where does the residual energy concentrate?
       - If concentrated at edges/boundaries → structured → learned model can focus there
       - If uniform → noise-like → no spatial prior to exploit

    6. Theoretical Entropy Bound
       - Compute 0th-order entropy (per-pixel, no context)
       - Compute 1st-order entropy (conditioned on left/above neighbor)
       - Compare to JPEG actual BPP
       - Gap between 1st-order entropy and JPEG = room for improvement
       - Gap between 0th-order and 1st-order = spatial redundancy available

    7. Cross-Image Consistency
       - Compute autocorrelation profile for multiple residuals
       - How consistent is the structure across different tile families?
       - If profiles are similar → amortized model can learn a consistent prior
       - If profiles vary wildly → per-image approach (Cool-Chic) needed

    8. Lanczos3 vs SR Model Comparison
       - Run all above analyses on both lanczos3 and SR model residuals
       - SR model should produce smaller, sparser residuals
       - Does SR also change the spatial structure (more/less autocorrelation)?
"""

import os
import sys
import glob
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

# Optional imports — gracefully degrade if not available
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping plots")

try:
    from scipy import stats, signal, fft as scipy_fft
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, using numpy fallbacks")


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_residual(path):
    """Load a residual PNG as float array centered at 0 (subtract 128)."""
    img = np.array(Image.open(path).convert('L'), dtype=np.float32)
    return img - 128.0


def spatial_autocorrelation(img, max_lag=64):
    """
    Compute normalized spatial autocorrelation at horizontal lags 1..max_lag.
    Returns array of autocorrelation values (1.0 at lag 0, decaying).
    """
    h, w = img.shape
    mean = img.mean()
    var = img.var()
    if var < 1e-10:
        return np.zeros(max_lag)

    centered = img - mean
    autocorr = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        autocorr[lag - 1] = np.mean(centered[:, :w-lag] * centered[:, lag:]) / var
    return autocorr


def radial_power_spectrum(img):
    """
    Compute radially averaged power spectrum.
    Returns (frequencies, power) arrays.
    """
    h, w = img.shape
    f2d = np.fft.fft2(img - img.mean())
    psd2d = np.abs(np.fft.fftshift(f2d)) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r = np.sqrt(x**2 + y**2).astype(int)

    max_r = min(cy, cx)
    radial_mean = np.zeros(max_r)
    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            radial_mean[ri] = psd2d[mask].mean()

    freqs = np.arange(max_r) / max_r
    return freqs, radial_mean


def block_sparsity(img, block_size=8):
    """
    Compute mean absolute value per block_size x block_size block.
    Returns array of block energies.
    """
    h, w = img.shape
    bh, bw = h // block_size, w // block_size
    blocks = img[:bh*block_size, :bw*block_size].reshape(bh, block_size, bw, block_size)
    return np.mean(np.abs(blocks), axis=(1, 3)).flatten()


def empirical_entropy(img_uint8):
    """Compute 0th-order empirical entropy in bits per pixel."""
    hist, _ = np.histogram(img_uint8, bins=256, range=(0, 256))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def conditional_entropy_1st_order(img_uint8):
    """
    Compute 1st-order conditional entropy: H(pixel | left neighbor).
    The gap between 0th-order and 1st-order entropy = spatial redundancy.
    """
    h, w = img_uint8.shape
    # Joint distribution of (pixel, left_neighbor)
    joint = np.zeros((256, 256), dtype=np.int64)
    for row in range(h):
        for col in range(1, w):
            joint[img_uint8[row, col], img_uint8[row, col-1]] += 1

    # P(x, context)
    total = joint.sum()
    if total == 0:
        return 0.0

    p_joint = joint / total
    # P(context) = marginal over x
    p_context = p_joint.sum(axis=0)

    # H(X|C) = -sum P(x,c) log2(P(x|c))
    entropy = 0.0
    for x in range(256):
        for c in range(256):
            if p_joint[x, c] > 0 and p_context[c] > 0:
                entropy -= p_joint[x, c] * np.log2(p_joint[x, c] / p_context[c])
    return entropy


def jpeg_artifact_analysis(img):
    """
    Detect JPEG 8x8 block grid artifacts in the residual.

    JPEG compression of the L2 baseline creates block boundary discontinuities.
    When L2 is upsampled 4x, these 8px boundaries in L2 become 32px boundaries
    in L0 space. The residual captures these as periodic structure.

    Returns dict with:
      - boundary_energy_ratio: energy at block boundaries vs interior (>1.0 = artifacts)
      - autocorr_at_multiples: autocorrelation specifically at lag 8, 16, 32, 64
      - fft_peak_8: power at frequency 1/8 relative to neighbors
      - fft_peak_32: power at frequency 1/32 relative to neighbors
      - boundary_variance_map: variance along rows/cols at boundary positions
    """
    h, w = img.shape
    results = {}

    # 1. Boundary vs interior energy
    # Check for elevated energy at 8x8 block boundaries (L0 tile grid = every 256px)
    # and at 32px intervals (L2 8x8 blocks upsampled 4x)
    for period, label in [(8, '8'), (32, '32'), (64, '64'), (256, '256')]:
        if period >= w or period >= h:
            results[f'boundary_energy_{label}'] = float('nan')
            continue
        # Horizontal boundary pixels: columns at multiples of period
        boundary_cols = np.arange(0, w, period)
        interior_cols = np.array([c for c in range(w) if c % period != 0 and (c-1) % period != 0 and (c+1) % period != 0])
        if len(boundary_cols) < 2 or len(interior_cols) < 10:
            results[f'boundary_energy_{label}'] = float('nan')
            continue

        # Energy at boundary columns (±1 pixel) vs interior
        boundary_mask = np.zeros(w, dtype=bool)
        for bc in boundary_cols:
            for offset in [-1, 0, 1]:
                idx = bc + offset
                if 0 <= idx < w:
                    boundary_mask[idx] = True

        boundary_energy = np.mean(img[:, boundary_mask] ** 2)
        interior_energy = np.mean(img[:, ~boundary_mask] ** 2)
        results[f'boundary_energy_{label}'] = float(boundary_energy / max(interior_energy, 1e-10))

    # 2. Autocorrelation at exact multiples of 8 and 32
    # If JPEG artifacts are present, autocorrelation spikes at these lags
    centered = img - img.mean()
    var = img.var()
    if var > 1e-10:
        for lag in [8, 16, 24, 32, 40, 48, 56, 64]:
            if lag < w:
                ac = np.mean(centered[:, :w-lag] * centered[:, lag:]) / var
                results[f'autocorr_lag{lag}'] = float(ac)

    # 3. FFT analysis — look for spectral peaks at JPEG artifact frequencies
    # 1D FFT along rows, then average
    row_fft = np.fft.rfft(img - img.mean(), axis=1)
    row_power = np.mean(np.abs(row_fft) ** 2, axis=0)
    freqs = np.fft.rfftfreq(w)

    # Peak at frequency 1/8 (JPEG block grid)
    target_freq_8 = 1.0 / 8.0
    idx_8 = np.argmin(np.abs(freqs - target_freq_8))
    # Compare peak to average of neighbors (±5 bins)
    neighborhood = slice(max(0, idx_8-5), min(len(row_power), idx_8+6))
    neighbor_power = np.mean(np.concatenate([row_power[max(0,idx_8-5):idx_8-1], row_power[idx_8+2:idx_8+6]]))
    results['fft_peak_ratio_8'] = float(row_power[idx_8] / max(neighbor_power, 1e-10))

    # Peak at frequency 1/32 (L2 JPEG blocks upsampled 4x)
    target_freq_32 = 1.0 / 32.0
    idx_32 = np.argmin(np.abs(freqs - target_freq_32))
    neighbor_power_32 = np.mean(np.concatenate([row_power[max(0,idx_32-5):idx_32-1], row_power[idx_32+2:idx_32+6]]))
    results['fft_peak_ratio_32'] = float(row_power[idx_32] / max(neighbor_power_32, 1e-10))

    # 4. Row/column variance profile — JPEG artifacts create periodic variance bumps
    # Compute variance of each column across all rows
    col_variance = np.var(img, axis=0)  # variance per column
    row_variance = np.var(img, axis=1)  # variance per row

    # Autocorrelation of the variance profile itself
    # If JPEG artifacts exist, variance profile has period-8 or period-32 pattern
    cv_centered = col_variance - col_variance.mean()
    cv_var = cv_centered.var()
    if cv_var > 1e-10 and len(cv_centered) > 32:
        cv_ac8 = np.mean(cv_centered[:-8] * cv_centered[8:]) / cv_var
        cv_ac32 = np.mean(cv_centered[:-32] * cv_centered[32:]) / cv_var
        results['col_var_autocorr_8'] = float(cv_ac8)
        results['col_var_autocorr_32'] = float(cv_ac32)
    else:
        results['col_var_autocorr_8'] = 0.0
        results['col_var_autocorr_32'] = 0.0

    # Store the raw profiles for plotting
    results['row_power_spectrum'] = row_power
    results['row_power_freqs'] = freqs
    results['col_variance_profile'] = col_variance

    return results


def fit_laplacian(data):
    """Fit a Laplacian distribution to centered residual data. Returns (loc, scale)."""
    loc = np.median(data)
    scale = np.mean(np.abs(data - loc))
    return loc, scale


def fit_gaussian(data):
    """Fit a Gaussian distribution. Returns (mean, std)."""
    return data.mean(), data.std()


# ── Main Analysis ────────────────────────────────────────────────────────────

def find_residual_images(runs_dir):
    """Find all fused residual centered PNGs, grouped by predictor type."""
    groups = defaultdict(list)

    for path in sorted(glob.glob(os.path.join(runs_dir, '*/compress/*fused_residual_centered.png'))):
        run_name = Path(path).parent.parent.name
        if run_name.startswith('sr_'):
            if 'lanczos3' in run_name:
                groups['lanczos3'].append(path)
            elif 'dual' in run_name or 'wsisrx4' in run_name or 'espcnr' in run_name:
                groups['sr_model'].append(path)
            else:
                groups['other'].append(path)
        elif run_name.startswith('v2_'):
            groups['lanczos3'].append(path)
        elif run_name.startswith('rs_'):
            groups['lanczos3'].append(path)
        else:
            groups['other'].append(path)

    return groups


def analyze_single_residual(path, label=""):
    """Run all analyses on a single residual image. Returns dict of metrics."""
    img_float = load_residual(path)  # centered at 0
    img_uint8 = np.array(Image.open(path).convert('L'))

    h, w = img_float.shape
    results = {
        'path': path,
        'label': label,
        'shape': (h, w),
        'pixels': h * w,
    }

    # 1. Basic statistics
    results['mean'] = float(img_float.mean())
    results['std'] = float(img_float.std())
    results['min'] = float(img_float.min())
    results['max'] = float(img_float.max())
    results['median'] = float(np.median(img_float))
    results['pct_near_zero'] = float(np.mean(np.abs(img_float) < 3.0))  # within ±3 of center

    # 2. Spatial autocorrelation
    results['autocorr'] = spatial_autocorrelation(img_float, max_lag=64)
    results['autocorr_lag1'] = float(results['autocorr'][0])
    results['autocorr_lag4'] = float(results['autocorr'][3])
    results['autocorr_lag8'] = float(results['autocorr'][7])
    results['autocorr_lag16'] = float(results['autocorr'][15])

    # 3. Power spectrum
    results['ps_freqs'], results['ps_power'] = radial_power_spectrum(img_float)

    # Compute spectral slope (log-log regression of power vs frequency)
    valid = results['ps_power'] > 0
    if valid.sum() > 10:
        log_f = np.log10(results['ps_freqs'][1:][valid[1:]] + 1e-10)
        log_p = np.log10(results['ps_power'][1:][valid[1:]] + 1e-10)
        if len(log_f) > 2:
            slope, intercept = np.polyfit(log_f[:len(log_f)//2], log_p[:len(log_p)//2], 1)
            results['spectral_slope'] = float(slope)
        else:
            results['spectral_slope'] = 0.0
    else:
        results['spectral_slope'] = 0.0

    # 4. Block sparsity
    block_energies = block_sparsity(img_float, block_size=8)
    results['block_energies'] = block_energies
    results['pct_blocks_below_1'] = float(np.mean(block_energies < 1.0))
    results['pct_blocks_below_3'] = float(np.mean(block_energies < 3.0))
    results['pct_blocks_below_5'] = float(np.mean(block_energies < 5.0))
    results['pct_blocks_below_10'] = float(np.mean(block_energies < 10.0))

    # 5. Distribution fitting
    flat = img_float.flatten()
    lap_loc, lap_scale = fit_laplacian(flat)
    gauss_mean, gauss_std = fit_gaussian(flat)
    results['laplacian_loc'] = float(lap_loc)
    results['laplacian_scale'] = float(lap_scale)
    results['gaussian_mean'] = float(gauss_mean)
    results['gaussian_std'] = float(gauss_std)

    # 6. Entropy
    results['entropy_0th'] = empirical_entropy(img_uint8)
    # 1st-order is slow on 1024x1024, do it on a center crop
    crop_size = min(256, h, w)
    cy, cx = h // 2, w // 2
    crop = img_uint8[cy-crop_size//2:cy+crop_size//2, cx-crop_size//2:cx+crop_size//2]
    results['entropy_1st'] = conditional_entropy_1st_order(crop)
    results['spatial_redundancy'] = results['entropy_0th'] - results['entropy_1st']

    # 7. PNG lossless size (empirical compressibility)
    import io
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format='PNG', optimize=True)
    png_bytes = buf.tell()
    results['png_bpp'] = float(png_bytes * 8) / (h * w)

    # 8. JPEG artifact analysis
    jpeg_results = jpeg_artifact_analysis(img_float)
    results.update(jpeg_results)

    return results


def print_report(all_results, output_dir):
    """Print and save summary report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("RESIDUAL COMPRESSIBILITY ANALYSIS")
    lines.append("=" * 80)

    for group_name, results_list in all_results.items():
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Group: {group_name} ({len(results_list)} images)")
        lines.append(f"{'─' * 80}")

        if not results_list:
            continue

        # Aggregate stats
        metrics = ['std', 'pct_near_zero', 'autocorr_lag1', 'autocorr_lag4',
                    'autocorr_lag8', 'autocorr_lag16', 'spectral_slope',
                    'pct_blocks_below_1', 'pct_blocks_below_3', 'pct_blocks_below_5',
                    'entropy_0th', 'entropy_1st', 'spatial_redundancy', 'png_bpp',
                    'boundary_energy_8', 'boundary_energy_32', 'boundary_energy_64',
                    'fft_peak_ratio_8', 'fft_peak_ratio_32',
                    'col_var_autocorr_8', 'col_var_autocorr_32']

        for metric in metrics:
            vals = [r[metric] for r in results_list if metric in r]
            if vals:
                lines.append(f"  {metric:30s}: mean={np.mean(vals):8.4f}  std={np.std(vals):8.4f}  "
                             f"min={np.min(vals):8.4f}  max={np.max(vals):8.4f}")

        # Interpretation
        lines.append("")
        avg_ac1 = np.mean([r['autocorr_lag1'] for r in results_list])
        avg_ac4 = np.mean([r['autocorr_lag4'] for r in results_list])
        avg_ac16 = np.mean([r['autocorr_lag16'] for r in results_list])
        avg_redundancy = np.mean([r['spatial_redundancy'] for r in results_list])
        avg_entropy_0 = np.mean([r['entropy_0th'] for r in results_list])
        avg_entropy_1 = np.mean([r['entropy_1st'] for r in results_list])
        avg_slope = np.mean([r['spectral_slope'] for r in results_list])
        avg_sparse = np.mean([r['pct_blocks_below_3'] for r in results_list])

        lines.append("  INTERPRETATION:")
        lines.append(f"  • Autocorrelation lag-1:  {avg_ac1:.4f}  ", )
        if avg_ac1 > 0.7:
            lines.append("    → STRONG spatial correlation — learned model CAN exploit this")
        elif avg_ac1 > 0.3:
            lines.append("    → MODERATE spatial correlation — learned model may help")
        else:
            lines.append("    → WEAK spatial correlation — residuals are noise-like")

        lines.append(f"  • Autocorrelation lag-16: {avg_ac16:.4f}")
        if avg_ac16 > 0.2:
            lines.append("    → Long-range structure exists — larger receptive fields help")
        else:
            lines.append("    → No long-range structure — only local patterns")

        lines.append(f"  • Spectral slope: {avg_slope:.2f}")
        if avg_slope < -1.5:
            lines.append("    → Strong low-frequency dominance (like natural images) — very compressible")
        elif avg_slope < -0.5:
            lines.append("    → Some frequency structure — moderately compressible")
        else:
            lines.append("    → Flat/white spectrum — noise-like, hard to compress beyond entropy")

        lines.append(f"  • Spatial redundancy (H0 - H1): {avg_redundancy:.4f} bits/pixel")
        lines.append(f"    0th-order entropy: {avg_entropy_0:.4f} bpp")
        lines.append(f"    1st-order entropy: {avg_entropy_1:.4f} bpp")
        if avg_redundancy > 1.0:
            lines.append("    → SIGNIFICANT spatial redundancy — context-based coding wins big")
        elif avg_redundancy > 0.3:
            lines.append("    → MODERATE spatial redundancy — context helps somewhat")
        else:
            lines.append("    → MINIMAL spatial redundancy — context coding gives little benefit")

        lines.append(f"  • Block sparsity (% blocks with mean |residual| < 3): {avg_sparse:.1%}")
        if avg_sparse > 0.5:
            lines.append("    → Highly sparse — block-skip mask saves significant bits")
        elif avg_sparse > 0.2:
            lines.append("    → Moderately sparse — some block-skip savings")
        else:
            lines.append("    → Dense residuals — block-skip not helpful")

        # JPEG artifact analysis
        lines.append("")
        lines.append("  JPEG ARTIFACT ANALYSIS:")
        be8 = np.mean([r.get('boundary_energy_8', float('nan')) for r in results_list
                       if not np.isnan(r.get('boundary_energy_8', float('nan')))] or [0])
        be32 = np.mean([r.get('boundary_energy_32', float('nan')) for r in results_list
                        if not np.isnan(r.get('boundary_energy_32', float('nan')))] or [0])
        be64 = np.mean([r.get('boundary_energy_64', float('nan')) for r in results_list
                        if not np.isnan(r.get('boundary_energy_64', float('nan')))] or [0])
        fft8 = np.mean([r.get('fft_peak_ratio_8', 1.0) for r in results_list])
        fft32 = np.mean([r.get('fft_peak_ratio_32', 1.0) for r in results_list])
        cva8 = np.mean([r.get('col_var_autocorr_8', 0.0) for r in results_list])
        cva32 = np.mean([r.get('col_var_autocorr_32', 0.0) for r in results_list])

        lines.append(f"  • Boundary energy ratio (period 8):  {be8:.4f}  (1.0 = no artifacts, >1.05 = artifacts)")
        lines.append(f"  • Boundary energy ratio (period 32): {be32:.4f}  (L2 JPEG blocks upsampled 4x)")
        lines.append(f"  • Boundary energy ratio (period 64): {be64:.4f}")
        lines.append(f"  • FFT peak at 1/8 frequency:  {fft8:.4f}x neighbors  (>1.5 = strong grid)")
        lines.append(f"  • FFT peak at 1/32 frequency: {fft32:.4f}x neighbors")
        lines.append(f"  • Column variance autocorrelation at lag 8:  {cva8:.4f}")
        lines.append(f"  • Column variance autocorrelation at lag 32: {cva32:.4f}")

        has_artifacts = be32 > 1.05 or fft8 > 1.5 or fft32 > 1.5 or cva32 > 0.1
        if has_artifacts:
            lines.append("    → JPEG BLOCK ARTIFACTS DETECTED in residuals")
            lines.append("      L2 baseline JPEG compression creates periodic structure that")
            lines.append("      contaminates the residual. This is NOT tissue structure.")
            if be32 > 1.1:
                lines.append(f"      Period-32 artifacts are strongest (L2 8x8 blocks × 4x upsample)")
        else:
            lines.append("    → No significant JPEG block artifacts detected")

        # Bottom line
        lines.append("")
        lines.append("  BOTTOM LINE:")
        if avg_ac1 > 0.5 and avg_redundancy > 0.5:
            lines.append("  ✓ Residuals have exploitable structure. A learned codec SHOULD beat JPEG.")
            lines.append(f"    Estimated headroom: ~{avg_redundancy:.1f} bpp over memoryless coding.")
        elif avg_ac1 > 0.3 or avg_redundancy > 0.3:
            lines.append("  ~ Residuals have some structure. A learned codec MIGHT beat JPEG by a small margin.")
            lines.append("    Better prediction (SR model) is likely more impactful than better compression.")
        else:
            lines.append("  ✗ Residuals are noise-like. JPEG is near-optimal.")
            lines.append("    Focus on improving the predictor (SR model) to shrink residuals, not compressing harder.")

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(report)


def plot_results(all_results, output_dir):
    """Generate comparison plots."""
    if not HAS_MPL:
        print("Skipping plots (matplotlib not available)")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Autocorrelation comparison ──
    fig, ax = plt.subplots(figsize=(10, 6))
    lags = np.arange(1, 65)

    # White noise reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='White noise')

    colors = {'lanczos3': '#2196F3', 'sr_model': '#4CAF50', 'other': '#FF9800'}
    for group_name, results_list in all_results.items():
        if not results_list:
            continue
        autocorrs = np.array([r['autocorr'] for r in results_list])
        mean_ac = autocorrs.mean(axis=0)
        std_ac = autocorrs.std(axis=0)
        color = colors.get(group_name, '#999999')
        ax.plot(lags, mean_ac, color=color, linewidth=2, label=f'{group_name} (n={len(results_list)})')
        ax.fill_between(lags, mean_ac - std_ac, mean_ac + std_ac, color=color, alpha=0.15)

    ax.set_xlabel('Lag (pixels)')
    ax.set_ylabel('Normalized Autocorrelation')
    ax.set_title('Spatial Autocorrelation of Residuals')
    ax.legend()
    ax.set_xlim(1, 64)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'spatial_autocorrelation.png'), dpi=150)
    plt.close(fig)

    # ── 2. Pixel distribution ──
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5), squeeze=False)
    for idx, (group_name, results_list) in enumerate(all_results.items()):
        if not results_list:
            continue
        ax = axes[0][idx]
        # Use first image for histogram
        r = results_list[0]
        img = load_residual(r['path'])
        flat = img.flatten()

        ax.hist(flat, bins=256, range=(-128, 128), density=True, alpha=0.6, color=colors.get(group_name, '#999'))

        # Fitted Laplacian
        x = np.linspace(-128, 128, 500)
        lap_pdf = np.exp(-np.abs(x - r['laplacian_loc']) / r['laplacian_scale']) / (2 * r['laplacian_scale'])
        ax.plot(x, lap_pdf, 'r-', linewidth=2, label=f"Laplacian (b={r['laplacian_scale']:.1f})")

        # Fitted Gaussian
        gauss_pdf = np.exp(-0.5 * ((x - r['gaussian_mean']) / r['gaussian_std'])**2) / (r['gaussian_std'] * np.sqrt(2 * np.pi))
        ax.plot(x, gauss_pdf, 'g--', linewidth=2, label=f"Gaussian (σ={r['gaussian_std']:.1f})")

        ax.set_title(f'{group_name}')
        ax.set_xlabel('Residual value (centered)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_xlim(-60, 60)

    fig.suptitle('Residual Pixel Distribution', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pixel_distribution.png'), dpi=150)
    plt.close(fig)

    # ── 3. Power spectrum ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for group_name, results_list in all_results.items():
        if not results_list:
            continue
        # Average power spectrum across group
        powers = []
        freqs = None
        for r in results_list:
            if freqs is None:
                freqs = r['ps_freqs']
            powers.append(r['ps_power'])
        if powers:
            mean_power = np.mean(powers, axis=0)
            valid = (freqs > 0) & (mean_power > 0)
            color = colors.get(group_name, '#999')
            ax.loglog(freqs[valid], mean_power[valid], color=color, linewidth=2, label=group_name)

    # Reference: 1/f^2 (natural images) and flat (white noise)
    if freqs is not None:
        f_ref = freqs[freqs > 0]
        ax.loglog(f_ref, f_ref**(-2) * mean_power[1] * freqs[1]**2, 'k--', alpha=0.3, label='1/f² (natural image)')
        ax.axhline(y=np.median(mean_power[len(mean_power)//2:]), color='gray', linestyle=':', alpha=0.3, label='Flat (white noise)')

    ax.set_xlabel('Spatial Frequency')
    ax.set_ylabel('Power')
    ax.set_title('Radial Power Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'power_spectrum.png'), dpi=150)
    plt.close(fig)

    # ── 4. Block sparsity CDF ──
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = np.linspace(0, 30, 300)
    for group_name, results_list in all_results.items():
        if not results_list:
            continue
        all_blocks = np.concatenate([r['block_energies'] for r in results_list])
        cdf = [np.mean(all_blocks < t) for t in thresholds]
        color = colors.get(group_name, '#999')
        ax.plot(thresholds, cdf, color=color, linewidth=2, label=group_name)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean |residual| per 8x8 block')
    ax.set_ylabel('Fraction of blocks below threshold')
    ax.set_title('Block Sparsity CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'block_sparsity.png'), dpi=150)
    plt.close(fig)

    # ── 5. Spatial variance heatmap (first image per group) ──
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5), squeeze=False)
    for idx, (group_name, results_list) in enumerate(all_results.items()):
        if not results_list:
            continue
        ax = axes[0][idx]
        img = load_residual(results_list[0]['path'])
        h, w = img.shape
        bsize = 32
        bh, bw = h // bsize, w // bsize
        var_map = np.zeros((bh, bw))
        for by in range(bh):
            for bx in range(bw):
                block = img[by*bsize:(by+1)*bsize, bx*bsize:(bx+1)*bsize]
                var_map[by, bx] = np.var(block)
        im = ax.imshow(var_map, cmap='hot', interpolation='nearest')
        ax.set_title(f'{group_name} — local variance')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('Spatial Variance Map (32x32 blocks)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'spatial_variance_map.png'), dpi=150)
    plt.close(fig)

    # ── 6. Entropy comparison bar chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    group_names = []
    h0_vals = []
    h1_vals = []
    png_vals = []
    for group_name, results_list in all_results.items():
        if not results_list:
            continue
        group_names.append(group_name)
        h0_vals.append(np.mean([r['entropy_0th'] for r in results_list]))
        h1_vals.append(np.mean([r['entropy_1st'] for r in results_list]))
        png_vals.append(np.mean([r['png_bpp'] for r in results_list]))

    x = np.arange(len(group_names))
    width = 0.25
    ax.bar(x - width, h0_vals, width, label='0th-order entropy', color='#2196F3')
    ax.bar(x, h1_vals, width, label='1st-order entropy', color='#4CAF50')
    ax.bar(x + width, png_vals, width, label='PNG lossless (actual)', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel('Bits per pixel')
    ax.set_title('Entropy Analysis: Theoretical vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'entropy_vs_jpeg.png'), dpi=150)
    plt.close(fig)

    # ── 7. Cross-image autocorrelation consistency ──
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5), squeeze=False)
    for idx, (group_name, results_list) in enumerate(all_results.items()):
        if not results_list:
            continue
        ax = axes[0][idx]
        for r in results_list[:10]:  # Plot up to 10
            ax.plot(lags, r['autocorr'], alpha=0.4, color=colors.get(group_name, '#999'))
        # Mean
        mean_ac = np.mean([r['autocorr'] for r in results_list], axis=0)
        ax.plot(lags, mean_ac, color='black', linewidth=2, label='Mean')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{group_name}')
        ax.set_xlabel('Lag (pixels)')
        ax.set_ylabel('Autocorrelation')
        ax.legend()
        ax.set_xlim(1, 64)

    fig.suptitle('Cross-Image Autocorrelation Consistency', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cross_image_similarity.png'), dpi=150)
    plt.close(fig)

    # ── 8. JPEG Artifact Analysis ──

    # 8a. Row-wise power spectrum with artifact frequency markers
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5), squeeze=False)
    for idx, (group_name, results_list) in enumerate(all_results.items()):
        if not results_list:
            continue
        ax = axes[0][idx]
        # Average row power spectrum
        spectra = [r['row_power_spectrum'] for r in results_list if 'row_power_spectrum' in r]
        if not spectra:
            continue
        mean_spectrum = np.mean(spectra, axis=0)
        freqs = results_list[0].get('row_power_freqs', np.arange(len(mean_spectrum)) / len(mean_spectrum))
        color = colors.get(group_name, '#999')
        ax.semilogy(freqs, mean_spectrum, color=color, linewidth=1, alpha=0.8)

        # Mark JPEG artifact frequencies
        for freq_val, label, c in [(1/8, '1/8 (8px grid)', 'red'),
                                     (1/32, '1/32 (L2 blocks×4)', 'orange'),
                                     (1/64, '1/64', 'purple')]:
            ax.axvline(x=freq_val, color=c, linestyle='--', alpha=0.6, label=label)

        ax.set_title(f'{group_name}')
        ax.set_xlabel('Frequency (cycles/pixel)')
        ax.set_ylabel('Power')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 0.5)

    fig.suptitle('Row-wise Power Spectrum — JPEG Artifact Frequencies Marked', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'jpeg_artifact_spectrum.png'), dpi=150)
    plt.close(fig)

    # 8b. Column variance profile — periodic bumps = JPEG block boundaries
    fig, axes = plt.subplots(1, len(all_results), figsize=(6*len(all_results), 5), squeeze=False)
    for idx, (group_name, results_list) in enumerate(all_results.items()):
        if not results_list:
            continue
        ax = axes[0][idx]
        # Average column variance profile (first 256 columns to see detail)
        profiles = [r['col_variance_profile'][:256] for r in results_list if 'col_variance_profile' in r]
        if not profiles:
            continue
        mean_profile = np.mean(profiles, axis=0)
        color = colors.get(group_name, '#999')
        ax.plot(mean_profile, color=color, linewidth=1, alpha=0.8)

        # Mark expected JPEG boundary positions
        for pos in range(0, 256, 32):
            ax.axvline(x=pos, color='red', linestyle=':', alpha=0.3)
        for pos in range(0, 256, 8):
            ax.axvline(x=pos, color='orange', linestyle=':', alpha=0.15)

        ax.set_title(f'{group_name} (red=32px, orange=8px grid)')
        ax.set_xlabel('Column index')
        ax.set_ylabel('Variance')

    fig.suptitle('Column Variance Profile — Periodic Bumps = JPEG Block Boundaries', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'jpeg_artifact_column_variance.png'), dpi=150)
    plt.close(fig)

    # 8c. Boundary energy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    periods = ['8', '32', '64']
    x = np.arange(len(periods))
    width = 0.25
    for i, (group_name, results_list) in enumerate(all_results.items()):
        if not results_list:
            continue
        vals = []
        for p in periods:
            v = [r.get(f'boundary_energy_{p}', float('nan')) for r in results_list]
            v = [x for x in v if not np.isnan(x)]
            vals.append(np.mean(v) if v else 0)
        color = colors.get(group_name, '#999')
        ax.bar(x + i * width, vals, width, label=group_name, color=color)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No artifacts (1.0)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Period {p}px' for p in periods])
    ax.set_ylabel('Boundary / Interior Energy Ratio')
    ax.set_title('JPEG Block Boundary Energy — Ratio > 1.0 Indicates Artifacts')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'jpeg_artifact_boundary_energy.png'), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    runs_dir = os.path.join(os.path.dirname(__file__), '..', 'runs')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'residual_compressibility')

    print("Scanning for residual images...")
    groups = find_residual_images(runs_dir)

    for name, paths in groups.items():
        print(f"  {name}: {len(paths)} images")

    if not any(groups.values()):
        print("ERROR: No residual images found in evals/runs/*/compress/")
        print("Run: origami encode --image ... --debug-images --pack")
        sys.exit(1)

    # Analyze each group
    all_results = {}
    for group_name, paths in groups.items():
        if not paths:
            continue
        print(f"\nAnalyzing {group_name} ({len(paths)} images)...")
        results = []
        for i, path in enumerate(paths):
            print(f"  [{i+1}/{len(paths)}] {os.path.basename(path)}...")
            r = analyze_single_residual(path, label=group_name)
            results.append(r)
        all_results[group_name] = results

    # Report
    print_report(all_results, output_dir)
    plot_results(all_results, output_dir)

    print(f"\nDone. Report: {output_dir}/report.txt")


if __name__ == '__main__':
    main()
