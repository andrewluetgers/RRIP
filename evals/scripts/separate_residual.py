"""
Residual Structure/Noise Separation (Phase 3)
===============================================

Implements three separation methods and evaluates each against the JXL sweep
baseline from Phase 2.

Methods:
    A. Guided filter  — uses prediction as guide image
    B. Edge-correlation mask — uses missed edges from original vs prediction
    C. Wavelet thresholding — blind separation using wavelet domain

Usage:
    # Run all methods on a run's residual
    uv run python evals/scripts/separate_residual.py \
        --run-dir evals/runs/rs_444_b95_l0q50

    # Run specific method only
    uv run python evals/scripts/separate_residual.py \
        --run-dir evals/runs/rs_444_b95_l0q50 --method guided

    # Custom guided filter parameters
    uv run python evals/scripts/separate_residual.py \
        --run-dir evals/runs/rs_444_b95_l0q50 --method guided --radius 4 --eps 10

Output:
    evals/analysis/noise_floor/{experiment_id}/separation/
    ├── {method_params}/
    │   ├── structure.png
    │   ├── noise.png
    │   ├── structure_heatmap.png
    │   ├── noise_heatmap.png
    │   ├── reconstruction.png
    │   ├── edge_overlay.png
    │   └── metrics.json
    └── comparison_strip.png
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image
from skimage.filters import sobel

sys.path.insert(0, os.path.dirname(__file__))
from eval_noise_floor import evaluate_image, NumpyEncoder
from jxl_quality_sweep import find_source_images, rgb_to_y, edge_correlation, fft_peak_ratio


# ---------------------------------------------------------------------------
# Separation methods
# ---------------------------------------------------------------------------

def separate_guided(
    residual: np.ndarray,
    guide: np.ndarray,
    radius: int = 4,
    eps: float = 10.0,
) -> tuple:
    """Method A: Guided filter using prediction as guide.
    Returns (structure, noise) both centered at 128."""
    structure = cv2.ximgproc.guidedFilter(
        guide=guide.astype(np.uint8),
        src=residual.astype(np.uint8),
        radius=radius,
        eps=eps,
    )
    noise_component = residual.astype(np.float32) - structure.astype(np.float32) + 128
    noise_component = np.clip(noise_component, 0, 255).astype(np.uint8)
    return structure, noise_component


def separate_edge_mask(
    residual: np.ndarray,
    original_Y: np.ndarray,
    prediction_Y: np.ndarray,
    threshold: float = 0.1,
    blur_sigma: float = 1.0,
) -> tuple:
    """Method B: Edge-correlation mask using missed edges.
    Returns (structure, noise) both centered at 128."""
    missed_edges = np.clip(
        sobel(original_Y.astype(np.float64) / 255.0) -
        sobel(prediction_Y.astype(np.float64) / 255.0),
        0, None
    )

    # Create soft mask
    mask = (missed_edges > threshold).astype(np.float32)
    ksize = int(blur_sigma * 6) | 1  # odd kernel
    mask = cv2.GaussianBlur(mask, (ksize, ksize), blur_sigma)

    residual_centered = residual.astype(np.float32) - 128.0
    structure_centered = residual_centered * mask
    noise_centered = residual_centered * (1.0 - mask)

    structure = np.clip(structure_centered + 128, 0, 255).astype(np.uint8)
    noise = np.clip(noise_centered + 128, 0, 255).astype(np.uint8)
    return structure, noise


def separate_wavelet(
    residual: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
    sigma_multiplier: float = 1.0,
) -> tuple:
    """Method C: Wavelet soft thresholding (VisuShrink).
    Returns (structure, noise) both centered at 128."""
    import pywt

    centered = residual.astype(np.float64) - 128.0
    coeffs = pywt.wavedec2(centered, wavelet, level=level)

    # Estimate noise sigma from finest detail coefficients (MAD estimator)
    detail_fine = coeffs[-1]  # tuple of (cH, cV, cD) at finest level
    sigma = np.median(np.abs(detail_fine[0])) / 0.6745

    # VisuShrink threshold
    N = residual.size
    threshold = sigma * np.sqrt(2 * np.log(N)) * sigma_multiplier

    # Soft threshold detail coefficients
    coeffs_denoised = [coeffs[0]]  # keep approximation
    for detail in coeffs[1:]:
        coeffs_denoised.append(tuple(
            pywt.threshold(d, threshold, mode="soft") for d in detail
        ))

    structure_centered = pywt.waverec2(coeffs_denoised, wavelet)
    # Trim to original size (wavelet may pad)
    structure_centered = structure_centered[:residual.shape[0], :residual.shape[1]]

    structure = np.clip(structure_centered + 128, 0, 255).astype(np.uint8)
    noise_centered = centered - structure_centered
    noise = np.clip(noise_centered + 128, 0, 255).astype(np.uint8)
    return structure, noise


def separate_wavelet_adaptive(
    residual: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    fine_sigma: float = 0.3,
    coarse_sigma: float = 1.5,
) -> tuple:
    """Scale-adaptive wavelet: gentle threshold on fine details (preserve edges),
    aggressive on coarse levels (remove low-freq noise).
    Sigma linearly interpolated between fine_sigma (finest) and coarse_sigma (coarsest).
    Returns (structure, noise) both centered at 128."""
    import pywt

    centered = residual.astype(np.float64) - 128.0
    coeffs = pywt.wavedec2(centered, wavelet, level=level)

    # Estimate noise sigma from finest detail coefficients (MAD estimator)
    detail_fine = coeffs[-1]
    sigma = np.median(np.abs(detail_fine[0])) / 0.6745
    N = residual.size
    base_threshold = sigma * np.sqrt(2 * np.log(N))

    # Apply per-level thresholds: interpolate from fine_sigma to coarse_sigma
    n_detail = len(coeffs) - 1  # number of detail levels
    coeffs_denoised = [coeffs[0]]  # keep approximation
    for i, detail in enumerate(coeffs[1:]):
        # i=0 is coarsest detail, i=n_detail-1 is finest
        # Map so finest gets fine_sigma, coarsest gets coarse_sigma
        t = i / max(n_detail - 1, 1)  # 0 (coarsest) to 1 (finest)
        level_sigma = coarse_sigma * (1 - t) + fine_sigma * t
        threshold = base_threshold * level_sigma
        coeffs_denoised.append(tuple(
            pywt.threshold(d, threshold, mode="soft") for d in detail
        ))

    structure_centered = pywt.waverec2(coeffs_denoised, wavelet)
    structure_centered = structure_centered[:residual.shape[0], :residual.shape[1]]

    structure = np.clip(structure_centered + 128, 0, 255).astype(np.uint8)
    noise_centered = centered - structure_centered
    noise = np.clip(noise_centered + 128, 0, 255).astype(np.uint8)
    return structure, noise


def separate_wavelet_fusion(
    residual: np.ndarray,
    wavelets: list = None,
    level: int = 2,
    sigma_multiplier: float = 0.5,
) -> tuple:
    """Multi-wavelet fusion: run multiple wavelet bases independently and average.
    Different bases capture different features, so averaging preserves more structure
    while still removing noise that all bases agree is noise.
    Returns (structure, noise) both centered at 128."""
    if wavelets is None:
        wavelets = ["db4", "sym4", "coif2"]

    structures = []
    for wv in wavelets:
        s, _ = separate_wavelet(residual, wavelet=wv, level=level,
                                sigma_multiplier=sigma_multiplier)
        structures.append(s.astype(np.float64))

    # Average the structure estimates
    avg_structure = np.mean(structures, axis=0)
    structure = np.clip(avg_structure, 0, 255).astype(np.uint8)

    centered = residual.astype(np.float64) - 128.0
    structure_centered = avg_structure - 128.0
    noise_centered = centered - structure_centered
    noise = np.clip(noise_centered + 128, 0, 255).astype(np.uint8)
    return structure, noise


def separate_wavelet_sigma_fusion(
    residual: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
    sigmas: list = None,
    wavelets_for_fusion: list = None,
) -> tuple:
    """Multi-sigma fusion: run wavelet denoising at multiple sigma thresholds
    and average the structure estimates. Each sigma catches different detail levels.
    Optionally also fuses across wavelet bases at each sigma.
    Returns (structure, noise) both centered at 128."""
    if sigmas is None:
        sigmas = [0.25, 0.35, 0.5]

    structures = []
    for s in sigmas:
        if wavelets_for_fusion:
            # Fusion across bases at each sigma
            struct, _ = separate_wavelet_fusion(
                residual, wavelets=wavelets_for_fusion, level=level,
                sigma_multiplier=s,
            )
        else:
            struct, _ = separate_wavelet(residual, wavelet=wavelet, level=level,
                                         sigma_multiplier=s)
        structures.append(struct.astype(np.float64))

    avg_structure = np.mean(structures, axis=0)
    structure = np.clip(avg_structure, 0, 255).astype(np.uint8)

    centered = residual.astype(np.float64) - 128.0
    structure_centered = avg_structure - 128.0
    noise_centered = centered - structure_centered
    noise = np.clip(noise_centered + 128, 0, 255).astype(np.uint8)
    return structure, noise


def separate_combined(
    residual: np.ndarray,
    original_Y: np.ndarray,
    prediction_Y: np.ndarray,
    wavelet_sigma: float = 0.5,
    edge_threshold: float = 0.02,
    edge_weight: float = 0.7,
    wavelet_weight: float = 0.3,
) -> tuple:
    """Method D: Weighted combination of wavelet denoising and edge mask.
    Wavelet catches frequency-domain noise, edge mask preserves tissue structure.
    Returns (structure, noise) both centered at 128."""
    import pywt

    residual_centered = residual.astype(np.float32) - 128.0

    # Wavelet component
    coeffs = pywt.wavedec2(residual_centered.astype(np.float64), "db4", level=2)
    detail_fine = coeffs[-1]
    sigma = np.median(np.abs(detail_fine[0])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(residual.size)) * wavelet_sigma
    coeffs_dn = [coeffs[0]]
    for detail in coeffs[1:]:
        coeffs_dn.append(tuple(pywt.threshold(d, threshold, mode="soft") for d in detail))
    wavelet_structure = pywt.waverec2(coeffs_dn, "db4")[:residual.shape[0], :residual.shape[1]].astype(np.float32)

    # Edge mask component
    missed_edges = np.clip(
        sobel(original_Y.astype(np.float64) / 255.0) -
        sobel(prediction_Y.astype(np.float64) / 255.0),
        0, None
    )
    mask = (missed_edges > edge_threshold).astype(np.float32)
    ksize = 7
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 1.0)
    edge_structure = residual_centered * mask

    # Weighted combination
    combined = edge_weight * edge_structure + wavelet_weight * wavelet_structure

    structure = np.clip(combined + 128, 0, 255).astype(np.uint8)
    noise = np.clip(residual_centered - combined + 128, 0, 255).astype(np.uint8)
    return structure, noise


def sharpen_prediction_residual(
    original_Y: np.ndarray,
    prediction_Y: np.ndarray,
    strength: float = 0.5,
    radius: int = 3,
) -> tuple:
    """Apply unsharp mask to prediction and compute the resulting residual.
    Returns (sharpened_prediction, new_residual_centered_at_128, residual_energy)."""
    pred_f = prediction_Y.astype(np.float32)
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(pred_f, (ksize, ksize), radius / 2)
    sharpened = np.clip(pred_f + strength * (pred_f - blurred), 0, 255).astype(np.uint8)

    new_residual = original_Y.astype(np.float32) - sharpened.astype(np.float32) + 128
    new_residual = np.clip(new_residual, 0, 255).astype(np.uint8)
    energy = float(np.mean(np.abs(new_residual.astype(np.float32) - 128)))

    return sharpened, new_residual, energy


# ---------------------------------------------------------------------------
# Wavelet-domain noise synthesis
# ---------------------------------------------------------------------------

def synthesize_wavelet_noise(
    structure: np.ndarray,
    residual: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
    sigma_multiplier: float = 0.25,
    seed: int = 42,
) -> tuple:
    """Synthesize noise in wavelet domain to replace removed noise.

    Decomposes the actual noise (residual - structure) into wavelet domain,
    measures per-subband sigma (including approximation coefficients), then
    samples from Laplacian distributions matched to each subband. Inverse DWT
    automatically reproduces the correct spatial autocorrelation.

    Returns (synthesized_noise_centered_at_128, synthesis_params_dict).
    synthesis_params_dict contains what the decoder would need (total ~16 bytes).
    """
    import pywt

    # Recover what was removed
    actual_noise = residual.astype(np.float64) - structure.astype(np.float64)

    # Decompose the actual noise into wavelet domain
    noise_coeffs = pywt.wavedec2(actual_noise, wavelet, level=level)

    # Measure per-subband sigma (including approximation)
    approx_sigma = float(np.std(noise_coeffs[0]))
    subband_sigmas = []
    for detail_tuple in noise_coeffs[1:]:
        level_sigmas = []
        for d in detail_tuple:
            level_sigmas.append(float(np.std(d)))
        subband_sigmas.append(level_sigmas)

    # Synthesis: sample from Laplacian matched to each subband
    rng = np.random.default_rng(seed)

    # Approximation coefficients (low-freq noise component)
    if approx_sigma > 1e-10:
        b = approx_sigma / np.sqrt(2)
        synth_approx = rng.laplace(0, b, size=noise_coeffs[0].shape)
    else:
        synth_approx = np.zeros_like(noise_coeffs[0])
    synth_coeffs = [synth_approx]

    # Detail coefficients
    for lvl_idx, detail_tuple in enumerate(noise_coeffs[1:]):
        synth_detail = []
        for sub_idx, d in enumerate(detail_tuple):
            sub_sigma = subband_sigmas[lvl_idx][sub_idx]
            if sub_sigma < 1e-10:
                synth_detail.append(np.zeros_like(d))
                continue
            b = sub_sigma / np.sqrt(2)
            samples = rng.laplace(0, b, size=d.shape)
            synth_detail.append(samples)
        synth_coeffs.append(tuple(synth_detail))

    # Inverse DWT
    synth_spatial = pywt.waverec2(synth_coeffs, wavelet)
    synth_spatial = synth_spatial[:residual.shape[0], :residual.shape[1]]

    synth_noise = np.clip(synth_spatial + 128, 0, 255).astype(np.uint8)

    # Parameters that would be transmitted (what the decoder needs)
    # 1 approx sigma + 6 subband sigmas = 7 x float16 = 14 bytes + seed 2 bytes = 16 bytes
    params = {
        "wavelet": wavelet,
        "level": level,
        "seed": seed,
        "approx_sigma": approx_sigma,
        "subband_sigmas": [[float(s) for s in lvl] for lvl in subband_sigmas],
        "param_bytes": 16,
    }

    return synth_noise, params


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_separation(
    structure: np.ndarray,
    noise: np.ndarray,
    residual: np.ndarray,
    original_Y: np.ndarray,
    prediction_Y: np.ndarray,
    out_dir: str,
    original_rgb: np.ndarray = None,
    prediction_rgb: np.ndarray = None,
) -> dict:
    """Evaluate a separation result. Save assets and return metrics.

    If original_rgb and prediction_rgb are provided, generates color
    reconstructions and computes SSIM/LPIPS perceptual metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save images
    Image.fromarray(structure).save(os.path.join(out_dir, "structure.png"))
    Image.fromarray(noise).save(os.path.join(out_dir, "noise.png"))

    # Heatmaps
    struct_abs = np.abs(structure.astype(np.float32) - 128)
    noise_abs = np.abs(noise.astype(np.float32) - 128)

    if struct_abs.max() > 0:
        sh = (struct_abs / struct_abs.max() * 255).astype(np.uint8)
    else:
        sh = np.zeros_like(structure)
    Image.fromarray(sh).save(os.path.join(out_dir, "structure_heatmap.png"))

    if noise_abs.max() > 0:
        nh = (noise_abs / noise_abs.max() * 255).astype(np.uint8)
    else:
        nh = np.zeros_like(noise)
    Image.fromarray(nh).save(os.path.join(out_dir, "noise_heatmap.png"))

    # Reconstruction: prediction + (structure - 128)
    has_originals = original_Y is not None and prediction_Y is not None
    recon_psnr = None
    recon_ssim_val = None
    recon_lpips_val = None

    if has_originals:
        recon_Y = np.clip(
            prediction_Y.astype(np.float32) + (structure.astype(np.float32) - 128),
            0, 255
        ).astype(np.uint8)
        Image.fromarray(recon_Y).save(os.path.join(out_dir, "reconstruction.png"))

        # PSNR
        mse = np.mean((original_Y.astype(np.float64) - recon_Y.astype(np.float64)) ** 2)
        recon_psnr = float(10 * np.log10(255**2 / mse)) if mse > 0 else 99.0

        # Full residual reconstruction for comparison
        full_recon = np.clip(
            prediction_Y.astype(np.float32) + (residual.astype(np.float32) - 128),
            0, 255
        ).astype(np.uint8)
        full_mse = np.mean((original_Y.astype(np.float64) - full_recon.astype(np.float64)) ** 2)
        full_psnr = float(10 * np.log10(255**2 / full_mse)) if full_mse > 0 else 99.0

        # SSIM (structural similarity — rewards structure preservation, tolerant of noise removal)
        from skimage.metrics import structural_similarity as ssim
        recon_ssim_val = float(ssim(original_Y, recon_Y, data_range=255))
        full_ssim_val = float(ssim(original_Y, full_recon, data_range=255))

        # Color reconstruction: replace Y channel in prediction RGB with corrected Y
        if prediction_rgb is not None:
            # YCbCr approach: take Cb/Cr from prediction, Y from reconstruction
            pred_ycbcr = cv2.cvtColor(prediction_rgb, cv2.COLOR_RGB2YCrCb)
            recon_ycbcr = pred_ycbcr.copy()
            recon_ycbcr[:, :, 0] = recon_Y
            recon_color = cv2.cvtColor(recon_ycbcr, cv2.COLOR_YCrCb2RGB)
            Image.fromarray(recon_color).save(os.path.join(out_dir, "reconstruction_color.png"))

            # Also save prediction color and original color for side-by-side
            if original_rgb is not None:
                Image.fromarray(original_rgb).save(os.path.join(out_dir, "original_color.png"))
            Image.fromarray(prediction_rgb).save(os.path.join(out_dir, "prediction_color.png"))

            # LPIPS (perceptual similarity — learned metric, rewards structure over noise)
            try:
                import torch
                import lpips
                loss_fn = lpips.LPIPS(net="alex", verbose=False)
                def to_tensor(img):
                    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    return t * 2 - 1  # normalize to [-1, 1]
                if original_rgb is not None:
                    with torch.no_grad():
                        recon_lpips_val = float(loss_fn(to_tensor(original_rgb), to_tensor(recon_color)).item())
                    # Also compute for full reconstruction and prediction-only
                    full_recon_ycbcr = pred_ycbcr.copy()
                    full_recon_ycbcr[:, :, 0] = full_recon
                    full_recon_color = cv2.cvtColor(full_recon_ycbcr, cv2.COLOR_YCrCb2RGB)
                    with torch.no_grad():
                        full_lpips_val = float(loss_fn(to_tensor(original_rgb), to_tensor(full_recon_color)).item())
                        pred_lpips_val = float(loss_fn(to_tensor(original_rgb), to_tensor(prediction_rgb)).item())
                else:
                    full_lpips_val = None
                    pred_lpips_val = None
            except Exception as e:
                print(f" (LPIPS unavailable: {e})", end="")
                recon_lpips_val = None
                full_lpips_val = None
                pred_lpips_val = None
        else:
            full_lpips_val = None
            pred_lpips_val = None

        # Wavelet-domain noise synthesis
        synth_noise, synth_params = synthesize_wavelet_noise(
            structure, residual, wavelet="db4", level=2, sigma_multiplier=0.25,
        )
        Image.fromarray(synth_noise).save(os.path.join(out_dir, "noise_synthesized.png"))

        # Synthesized noise heatmap
        synth_abs = np.abs(synth_noise.astype(np.float32) - 128)
        if synth_abs.max() > 0:
            synth_hm = (synth_abs / max(noise_abs.max(), synth_abs.max(), 1) * 255).astype(np.uint8)
        else:
            synth_hm = np.zeros_like(synth_noise)
        Image.fromarray(synth_hm).save(os.path.join(out_dir, "noise_synth_heatmap.png"))

        # Reconstruction with synthesized noise: prediction + structure + synth_noise
        recon_with_noise_Y = np.clip(
            prediction_Y.astype(np.float32)
            + (structure.astype(np.float32) - 128)
            + (synth_noise.astype(np.float32) - 128),
            0, 255
        ).astype(np.uint8)
        Image.fromarray(recon_with_noise_Y).save(os.path.join(out_dir, "reconstruction_with_noise.png"))

        # Color reconstruction with synthesized noise
        if prediction_rgb is not None:
            recon_noisy_ycbcr = pred_ycbcr.copy()
            recon_noisy_ycbcr[:, :, 0] = recon_with_noise_Y
            recon_noisy_color = cv2.cvtColor(recon_noisy_ycbcr, cv2.COLOR_YCrCb2RGB)
            Image.fromarray(recon_noisy_color).save(os.path.join(out_dir, "reconstruction_noisy_color.png"))

        # Metrics for synthesized noise vs actual noise
        actual_noise_c = noise.astype(np.float32) - 128
        synth_noise_c = synth_noise.astype(np.float32) - 128
        synth_mse = float(np.mean((actual_noise_c - synth_noise_c) ** 2))
        # Correlation between actual and synth noise spatial patterns
        if np.std(actual_noise_c) > 1e-10 and np.std(synth_noise_c) > 1e-10:
            synth_corr = float(np.corrcoef(actual_noise_c.ravel(), synth_noise_c.ravel())[0, 1])
        else:
            synth_corr = 0.0
        # Synthesized noise stats match
        synth_std_ratio = float(np.std(synth_noise_c) / max(np.std(actual_noise_c), 1e-10))
        # SSIM of reconstruction-with-noise vs original
        recon_noisy_ssim = float(ssim(original_Y, recon_with_noise_Y, data_range=255))
        # LPIPS of reconstruction-with-noise vs original (color)
        recon_noisy_lpips = None
        if prediction_rgb is not None and original_rgb is not None:
            try:
                with torch.no_grad():
                    recon_noisy_lpips = float(loss_fn(to_tensor(original_rgb), to_tensor(recon_noisy_color)).item())
            except Exception:
                pass

        # Save synthesis params
        with open(os.path.join(out_dir, "noise_synthesis_params.json"), "w") as f:
            json.dump(synth_params, f, indent=2, cls=NumpyEncoder)

        # Edge overlay: structure edges on original — THICK, visible
        struct_edges = sobel(structure.astype(np.float64) / 255.0)
        # Dilate edge mask for visibility (3 iterations = ~6px thick)
        edge_binary = (struct_edges > 0.015).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        edge_thick = cv2.dilate(edge_binary, kernel, iterations=3)
        edge_mask = edge_thick.astype(np.float32)

        if original_rgb is not None:
            overlay = original_rgb.astype(np.float32).copy()
        else:
            overlay = np.stack([original_Y, original_Y, original_Y], axis=-1).astype(np.float32)
        # Cyan overlay for better visibility on H&E tissue
        overlay[:, :, 0] = np.where(edge_mask > 0, np.clip(overlay[:, :, 0] * 0.4, 0, 255), overlay[:, :, 0])
        overlay[:, :, 1] = np.where(edge_mask > 0, np.clip(overlay[:, :, 1] * 0.4 + 180, 0, 255), overlay[:, :, 1])
        overlay[:, :, 2] = np.where(edge_mask > 0, np.clip(overlay[:, :, 2] * 0.4 + 200, 0, 255), overlay[:, :, 2])
        Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(out_dir, "edge_overlay.png"))
    else:
        full_psnr = None
        full_ssim_val = None
        full_lpips_val = None
        pred_lpips_val = None
        synth_mse = None
        synth_corr = None
        synth_std_ratio = None
        recon_noisy_ssim = None
        recon_noisy_lpips = None
        synth_params = None

    # Noise-floor score of noise component
    noise_path = os.path.join(out_dir, "noise.png")
    noise_metrics = evaluate_image(noise_path)
    noise_score = noise_metrics["noise_floor_score"]

    # Noise uniformity: block variance coefficient of variation (lower = more uniform)
    noise_centered = noise.astype(np.float32) - 128.0
    block_sz = 32
    block_vars = []
    for by in range(0, noise_centered.shape[0] - block_sz + 1, block_sz):
        for bx in range(0, noise_centered.shape[1] - block_sz + 1, block_sz):
            block_vars.append(np.var(noise_centered[by:by+block_sz, bx:bx+block_sz]))
    block_vars = np.array(block_vars)
    noise_uniformity = float(np.std(block_vars) / np.mean(block_vars)) if np.mean(block_vars) > 0 else 1.0

    # Structure size: JXL Q80 single-channel
    struct_path = os.path.join(out_dir, "structure.png")
    with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as f:
        struct_jxl_path = f.name
    subprocess.run(
        ["cjxl", struct_path, struct_jxl_path, "-q", "80", "--quiet"],
        capture_output=True, timeout=30
    )
    struct_jxl_size = os.path.getsize(struct_jxl_path) if os.path.exists(struct_jxl_path) else 0

    # Original residual JXL Q80 size for comparison
    with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as f:
        orig_jxl_path = f.name
    residual_tmp = os.path.join(out_dir, "_residual_tmp.png")
    Image.fromarray(residual).save(residual_tmp)
    subprocess.run(
        ["cjxl", residual_tmp, orig_jxl_path, "-q", "80", "--quiet"],
        capture_output=True, timeout=30
    )
    orig_jxl_size = os.path.getsize(orig_jxl_path) if os.path.exists(orig_jxl_path) else 1
    os.unlink(residual_tmp)

    compression_ratio = struct_jxl_size / orig_jxl_size if orig_jxl_size > 0 else 1.0

    # Edge correlation
    edge_corr_val = None
    if has_originals:
        missed_edges = np.clip(
            sobel(original_Y.astype(np.float64) / 255.0) -
            sobel(prediction_Y.astype(np.float64) / 255.0),
            0, None
        )
        struct_edges_val = sobel((structure.astype(np.float64) - 128) / 255.0)
        me_flat = missed_edges.ravel()
        se_flat = struct_edges_val.ravel()
        if np.std(me_flat) > 1e-10 and np.std(se_flat) > 1e-10:
            edge_corr_val = float(np.corrcoef(me_flat, se_flat)[0, 1])

    metrics = {
        "noise_floor_score": noise_score,
        "structure_jxl_bytes": struct_jxl_size,
        "original_jxl_bytes": orig_jxl_size,
        "compression_ratio": float(compression_ratio),
        "reconstruction_psnr_db": recon_psnr,
        "full_residual_psnr_db": full_psnr,
        "psnr_delta_db": float(full_psnr - recon_psnr) if (recon_psnr and full_psnr) else None,
        "reconstruction_ssim": recon_ssim_val,
        "full_residual_ssim": full_ssim_val if has_originals else None,
        "reconstruction_lpips": recon_lpips_val,
        "full_residual_lpips": full_lpips_val if has_originals else None,
        "prediction_lpips": pred_lpips_val if has_originals else None,
        "edge_correlation": edge_corr_val,
        "noise_uniformity_cov": noise_uniformity,
        "structure_energy": float(np.mean(struct_abs)),
        "noise_energy": float(np.mean(noise_abs)),
        "synth_noise_mse": synth_mse,
        "synth_noise_corr": synth_corr,
        "synth_noise_std_ratio": synth_std_ratio,
        "synth_recon_ssim": recon_noisy_ssim,
        "synth_recon_lpips": recon_noisy_lpips,
        "synth_param_bytes": synth_params["param_bytes"] if synth_params else None,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    # Clean up temp files
    for p in [struct_jxl_path, orig_jxl_path]:
        if os.path.exists(p):
            os.unlink(p)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Residual structure/noise separation")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to an origami encode run (with --debug-images)")
    parser.add_argument("--method", choices=["guided", "edge_mask", "wavelet", "wavelet_adaptive", "wavelet_fusion", "wavelet_sigma_fusion", "combined", "sharpen", "all"], default="all",
                        help="Separation method to use")
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="evals/analysis/noise_floor")

    # Guided filter params
    parser.add_argument("--radius", type=int, default=None, help="Guided filter radius")
    parser.add_argument("--eps", type=float, default=None, help="Guided filter eps")

    # Edge mask params
    parser.add_argument("--threshold", type=float, default=None, help="Edge mask threshold")

    # Wavelet params
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--level", type=int, default=2)

    args = parser.parse_args()

    # Discover sources
    sources = find_source_images(args.run_dir, args.family)
    run_name = os.path.basename(args.run_dir)
    experiment_id = f"separation_{run_name}"

    residual = np.array(Image.open(sources["residual"]).convert("L"))

    has_originals = (
        sources["original_Y"] and sources["prediction_Y"] and
        os.path.exists(sources["original_Y"]) and os.path.exists(sources["prediction_Y"])
    )
    if has_originals:
        original_Y = np.array(Image.open(sources["original_Y"]).convert("L"))
        prediction_Y = np.array(Image.open(sources["prediction_Y"]).convert("L"))
    else:
        original_Y = prediction_Y = None

    # Load color images for perceptual evaluation
    original_rgb = prediction_rgb = None
    compress_dir = os.path.join(args.run_dir, "compress")
    decompress_dir = os.path.join(args.run_dir, "decompress")

    # Prediction RGB: from decompress mosaic
    pred_rgb_path = os.path.join(decompress_dir, "060_L0_mosaic_prediction.png")
    if os.path.exists(pred_rgb_path):
        prediction_rgb = np.array(Image.open(pred_rgb_path).convert("RGB"))
        print(f"  Prediction RGB: {pred_rgb_path} ({prediction_rgb.shape})")

    # Original RGB: assemble from L0 tiles
    orig_rgb_assembled = os.path.join(compress_dir, "_assembled_L0_original_RGB.png")
    if os.path.exists(orig_rgb_assembled):
        original_rgb = np.array(Image.open(orig_rgb_assembled).convert("RGB"))
    else:
        # Assemble from 16 individual 256x256 RGB tiles
        orig_tiles = sorted(glob.glob(os.path.join(compress_dir, "*_L0_*_original.png")))
        if len(orig_tiles) >= 16:
            print("  Assembling L0 original RGB mosaic from tiles...")
            grid = {}
            for t in orig_tiles:
                parts = os.path.basename(t).split("_")
                try:
                    l0_idx = parts.index("L0")
                    x, y = int(parts[l0_idx + 1]), int(parts[l0_idx + 2])
                    grid[(x, y)] = t
                except (ValueError, IndexError):
                    continue
            if len(grid) >= 16:
                max_x = max(x for x, y in grid.keys())
                max_y = max(y for x, y in grid.keys())
                mosaic = np.zeros(((max_y + 1) * 256, (max_x + 1) * 256, 3), dtype=np.uint8)
                for (x, y), path in grid.items():
                    tile = np.array(Image.open(path).convert("RGB"))
                    th, tw = tile.shape[:2]
                    mosaic[y*256:y*256+th, x*256:x*256+tw] = tile
                original_rgb = mosaic
                Image.fromarray(mosaic).save(orig_rgb_assembled)
                print(f"  Original RGB assembled: {mosaic.shape}")

    if original_rgb is not None:
        print(f"  Color comparison: ENABLED (SSIM + LPIPS)")
    else:
        print(f"  Color comparison: DISABLED (no original RGB)")

    out_base = os.path.join(args.output_dir, experiment_id, "separation")
    os.makedirs(out_base, exist_ok=True)

    results = {}

    # Method A: Guided filter
    if args.method in ("guided", "all") and has_originals:
        radii = [args.radius] if args.radius else [2, 4, 8, 16]
        epsilons = [args.eps] if args.eps else [1, 10, 100]

        for r in radii:
            for eps in epsilons:
                name = f"guided_r{r}_eps{int(eps)}"
                print(f"  {name}...", end=" ", flush=True)
                structure, noise = separate_guided(residual, prediction_Y, r, eps)
                m = evaluate_separation(
                    structure, noise, residual, original_Y, prediction_Y,
                    os.path.join(out_base, name),
                    original_rgb=original_rgb, prediction_rgb=prediction_rgb,
                )
                results[name] = m
                ssim_str = f" ssim={m['reconstruction_ssim']:.4f}" if m.get('reconstruction_ssim') else ""
                lpips_str = f" lpips={m['reconstruction_lpips']:.4f}" if m.get('reconstruction_lpips') else ""
                print(f"noise={m['noise_floor_score']:.3f} "
                      f"ratio={m['compression_ratio']:.2f}{ssim_str}{lpips_str}")

    # Method B: Edge-correlation mask
    if args.method in ("edge_mask", "all") and has_originals:
        thresholds = [args.threshold] if args.threshold else [0.02, 0.05, 0.1, 0.2]

        for t in thresholds:
            name = f"edge_mask_t{t}"
            print(f"  {name}...", end=" ", flush=True)
            structure, noise = separate_edge_mask(
                residual, original_Y, prediction_Y, threshold=t
            )
            m = evaluate_separation(
                structure, noise, residual, original_Y, prediction_Y,
                os.path.join(out_base, name),
                original_rgb=original_rgb, prediction_rgb=prediction_rgb,
            )
            results[name] = m
            ssim_str = f" ssim={m['reconstruction_ssim']:.4f}" if m.get('reconstruction_ssim') else ""
            lpips_str = f" lpips={m['reconstruction_lpips']:.4f}" if m.get('reconstruction_lpips') else ""
            print(f"noise={m['noise_floor_score']:.3f} "
                  f"ratio={m['compression_ratio']:.2f}{ssim_str}{lpips_str}")

    # Method C: Wavelet thresholding
    if args.method in ("wavelet", "all"):
        for sigma_mult in [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]:
            name = f"wavelet_{args.wavelet}_l{args.level}_s{sigma_mult}"
            print(f"  {name}...", end=" ", flush=True)
            structure, noise = separate_wavelet(
                residual, args.wavelet, args.level, sigma_mult
            )
            m = evaluate_separation(
                structure, noise, residual, original_Y, prediction_Y,
                os.path.join(out_base, name),
                original_rgb=original_rgb, prediction_rgb=prediction_rgb,
            )
            results[name] = m
            ssim_str = f" ssim={m['reconstruction_ssim']:.4f}" if m.get('reconstruction_ssim') else ""
            lpips_str = f" lpips={m['reconstruction_lpips']:.4f}" if m.get('reconstruction_lpips') else ""
            print(f"noise={m['noise_floor_score']:.3f} "
                  f"ratio={m['compression_ratio']:.2f}{ssim_str}{lpips_str}")

    # Method C2: Scale-adaptive wavelet (gentle fine, aggressive coarse)
    if args.method in ("wavelet_adaptive", "all"):
        configs = [
            ("db4", 3, 0.3, 1.5),   # gentle fine, aggressive coarse
            ("db4", 3, 0.5, 2.0),   # moderate fine, very aggressive coarse
            ("db4", 4, 0.3, 1.5),   # deeper decomposition
        ]
        for wv, lvl, fs, cs in configs:
            name = f"wavelet_adaptive_{wv}_l{lvl}_f{fs}_c{cs}"
            print(f"  {name}...", end=" ", flush=True)
            structure, noise = separate_wavelet_adaptive(
                residual, wavelet=wv, level=lvl, fine_sigma=fs, coarse_sigma=cs
            )
            m = evaluate_separation(
                structure, noise, residual, original_Y, prediction_Y,
                os.path.join(out_base, name),
                original_rgb=original_rgb, prediction_rgb=prediction_rgb,
            )
            results[name] = m
            ssim_str = f" ssim={m['reconstruction_ssim']:.4f}" if m.get('reconstruction_ssim') else ""
            lpips_str = f" lpips={m['reconstruction_lpips']:.4f}" if m.get('reconstruction_lpips') else ""
            print(f"noise={m['noise_floor_score']:.3f} "
                  f"ratio={m['compression_ratio']:.2f}{ssim_str}{lpips_str}")

    # Method C3: Multi-wavelet fusion
    if args.method in ("wavelet_fusion", "all"):
        fusion_configs = [
            (["db4", "sym4", "coif2"], 2, 0.25),
            (["db4", "sym4", "coif2"], 2, 0.35),
            (["db4", "sym4", "coif2"], 2, 0.5),
            (["db4", "sym4", "coif2"], 2, 1.0),
            (["db6", "sym6", "coif3"], 2, 0.35),
            (["db6", "sym6", "coif3"], 2, 0.5),
            (["db8", "sym8", "coif3"], 2, 0.35),
            (["db8", "sym8", "coif3"], 2, 0.5),
        ]
        for wavelets, lvl, sigma_mult in fusion_configs:
            wv_tag = "+".join(wavelets)
            name = f"wavelet_fusion_{wv_tag}_l{lvl}_s{sigma_mult}"
            print(f"  {name}...", end=" ", flush=True)
            structure, noise = separate_wavelet_fusion(
                residual, wavelets=wavelets,
                level=lvl, sigma_multiplier=sigma_mult,
            )
            m = evaluate_separation(
                structure, noise, residual, original_Y, prediction_Y,
                os.path.join(out_base, name),
                original_rgb=original_rgb, prediction_rgb=prediction_rgb,
            )
            results[name] = m
            ssim_str = f" ssim={m['reconstruction_ssim']:.4f}" if m.get('reconstruction_ssim') else ""
            lpips_str = f" lpips={m['reconstruction_lpips']:.4f}" if m.get('reconstruction_lpips') else ""
            print(f"noise={m['noise_floor_score']:.3f} "
                  f"ratio={m['compression_ratio']:.2f}{ssim_str}{lpips_str}")

    # Method C4: Multi-sigma fusion (average across thresholds)
    if args.method in ("wavelet_sigma_fusion", "all"):
        sigma_configs = [
            # (sigmas, wavelet_bases_or_None, label_suffix)
            ([0.25, 0.35, 0.5], None, "db4"),                         # single base, 3 sigmas
            ([0.25, 0.35, 0.5], ["db4", "sym4", "coif2"], "fused"),   # fused bases + 3 sigmas
            ([0.25, 0.5], None, "db4"),                                # just 2 sigmas
            ([0.25, 0.35, 0.5], ["db8", "sym8", "coif3"], "fused8"),  # longer bases + 3 sigmas
        ]
        for sigmas, wv_bases, tag in sigma_configs:
            s_str = "+".join(str(s) for s in sigmas)
            name = f"wavelet_sigma_{tag}_s{s_str}"
            print(f"  {name}...", end=" ", flush=True)
            structure, noise = separate_wavelet_sigma_fusion(
                residual, wavelet="db4", level=2, sigmas=sigmas,
                wavelets_for_fusion=wv_bases,
            )
            m = evaluate_separation(
                structure, noise, residual, original_Y, prediction_Y,
                os.path.join(out_base, name),
                original_rgb=original_rgb, prediction_rgb=prediction_rgb,
            )
            results[name] = m
            ssim_str = f" ssim={m['reconstruction_ssim']:.4f}" if m.get('reconstruction_ssim') else ""
            lpips_str = f" lpips={m['reconstruction_lpips']:.4f}" if m.get('reconstruction_lpips') else ""
            cov_str = f" cov={m['noise_uniformity_cov']:.4f}" if m.get('noise_uniformity_cov') else ""
            print(f"noise={m['noise_floor_score']:.3f} "
                  f"ratio={m['compression_ratio']:.2f}{ssim_str}{lpips_str}{cov_str}")

    # Method D: Combined (wavelet + edge mask)
    if args.method in ("combined", "all") and has_originals:
        combos = [
            (0.5, 0.02, 0.7, 0.3),  # gentle wavelet, low edge threshold, edge-heavy
            (0.5, 0.02, 0.5, 0.5),  # equal weight
            (0.5, 0.02, 0.3, 0.7),  # wavelet-heavy
            (1.0, 0.05, 0.7, 0.3),  # stronger wavelet, higher edge threshold
            (1.0, 0.05, 0.5, 0.5),
        ]
        for ws, et, ew, ww in combos:
            name = f"combined_ws{ws}_et{et}_ew{int(ew*10)}_ww{int(ww*10)}"
            print(f"  {name}...", end=" ", flush=True)
            structure, noise = separate_combined(
                residual, original_Y, prediction_Y,
                wavelet_sigma=ws, edge_threshold=et,
                edge_weight=ew, wavelet_weight=ww,
            )
            m = evaluate_separation(
                structure, noise, residual, original_Y, prediction_Y,
                os.path.join(out_base, name),
                original_rgb=original_rgb, prediction_rgb=prediction_rgb,
            )
            results[name] = m
            ssim_str = f" ssim={m['reconstruction_ssim']:.4f}" if m.get('reconstruction_ssim') else ""
            lpips_str = f" lpips={m['reconstruction_lpips']:.4f}" if m.get('reconstruction_lpips') else ""
            print(f"noise={m['noise_floor_score']:.3f} "
                  f"ratio={m['compression_ratio']:.2f}{ssim_str}{lpips_str}")

    # Sharpening experiment: does sharpening the prediction reduce residual energy?
    if args.method in ("sharpen", "all") and has_originals:
        print("\n  Sharpening experiment (residual energy reduction):")
        orig_energy = float(np.mean(np.abs(residual.astype(np.float32) - 128)))
        print(f"    Original residual energy: {orig_energy:.3f}")

        for strength in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            for radius in [2, 3, 5]:
                sharpened, new_res, energy = sharpen_prediction_residual(
                    original_Y, prediction_Y, strength=strength, radius=radius
                )
                reduction = (1 - energy / orig_energy) * 100
                # PSNR of sharpened prediction vs original
                mse = np.mean((original_Y.astype(np.float64) - sharpened.astype(np.float64)) ** 2)
                pred_psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99.0

                name = f"sharpen_s{strength}_r{radius}"
                print(f"    {name}: energy={energy:.3f} ({reduction:+.1f}%) pred_psnr={pred_psnr:.1f}dB")

                # Save the best ones as separation results
                if strength in [0.5, 1.0, 2.0] and radius == 3:
                    out = os.path.join(out_base, name)
                    os.makedirs(out, exist_ok=True)
                    Image.fromarray(new_res).save(os.path.join(out, "structure.png"))
                    Image.fromarray(sharpened).save(os.path.join(out, "reconstruction.png"))
                    # The "noise" is what sharpening added (difference between old and new residual)
                    noise_from_sharp = residual.astype(np.float32) - new_res.astype(np.float32) + 128
                    noise_from_sharp = np.clip(noise_from_sharp, 0, 255).astype(np.uint8)
                    m = evaluate_separation(
                        new_res, noise_from_sharp, residual, original_Y, prediction_Y,
                        out, original_rgb=original_rgb, prediction_rgb=prediction_rgb,
                    )
                    m["sharpening_strength"] = strength
                    m["sharpening_radius"] = radius
                    m["residual_energy"] = energy
                    m["energy_reduction_pct"] = reduction
                    m["prediction_psnr_db"] = pred_psnr
                    results[name] = m

    if not results:
        print("No methods run. Check --method and available source images.")
        sys.exit(1)

    # Summary
    print(f"\n{'Method':<35s} {'Noise Score':>11s} {'Comp Ratio':>10s} {'PSNR Loss':>10s} {'Edge Corr':>10s}")
    print("-" * 80)
    for name, m in sorted(results.items()):
        psnr_loss = f"{m['psnr_delta_db']:.2f}" if m.get('psnr_delta_db') is not None else "N/A"
        edge_corr = f"{m['edge_correlation']:.3f}" if m.get('edge_correlation') is not None else "N/A"
        print(f"{name:<35s} {m['noise_floor_score']:>11.3f} {m['compression_ratio']:>10.2f} {psnr_loss:>10s} {edge_corr:>10s}")

    # Save combined results
    combined_path = os.path.join(out_base, "all_methods_metrics.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Write metadata.json at experiment root for viewer discovery
    exp_root = os.path.dirname(out_base)
    meta = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run": args.run_dir,
        "has_three_signal": has_originals,
        "params": {"methods": list(results.keys())},
        "notes": "",
    }
    with open(os.path.join(exp_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)
    with open(os.path.join(exp_root, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {out_base}")


if __name__ == "__main__":
    main()
