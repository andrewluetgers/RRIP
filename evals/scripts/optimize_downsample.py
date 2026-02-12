#!/usr/bin/env python3
"""
optimize_downsample.py — Generic gradient descent optimization for downsampled images.

Given a downsampled image and a high-resolution target, optimizes the downsampled
pixels so that bilinear upsampling produces the best possible approximation of the
target. Works on RGB images directly (all channels optimized simultaneously).

This is the generalized form of ORIGAMI's OptL2 optimization. It applies anywhere
an image is downsampled for storage and later upsampled for display or prediction:
  - ORIGAMI L2 tile optimization (L2 predicts L1 via bilinear upsample)
  - Thumbnail/preview optimization (thumbnails that upscale with less blur)
  - Chroma subsampling (4:2:0 chroma optimized for bilinear reconstruction)

The math: bilinear upsampling is a linear operator U, so the loss
  L = ||target - U * source||^2
has gradient dL/d(source) = -2 * U^T * (target - U * source).
U^T (the adjoint of bilinear upsample) is bilinear downsample.

Performance: ~6.7ms per iteration for 256x256 → 512x512 on Apple M-series.
100 iterations is sufficient for convergence (binding constraint is max_delta, not
iteration count). Total: ~0.7s per tile family.

NumPy array operations use SIMD (NEON on ARM, SSE/AVX on x86) via the Accelerate
or OpenBLAS backend. PIL's C-level resize is faster than pure NumPy bilinear.
"""

import numpy as np
from PIL import Image


def optimize_for_upsample(
    source: np.ndarray,
    target: np.ndarray,
    max_delta: int = 15,
    n_iterations: int = 100,
    lr: float = 0.3,
) -> np.ndarray:
    """Optimize a downsampled image so its bilinear upsample best matches a target.

    Works directly in RGB space on all channels simultaneously. This avoids
    YCbCr conversion overhead (~34% of iteration time) and correctly optimizes
    the cross-channel interactions inherent in bilinear interpolation of RGB.

    Args:
        source: Downsampled image, shape (H, W, 3) uint8.
                Typically a Lanczos-downsampled version of the original.
        target: High-resolution target, shape (H*scale, W*scale, 3) uint8.
                The image we want the upsampled source to approximate.
        max_delta: Maximum per-pixel, per-channel deviation from original (±).
                   Controls the perceptual similarity to the original downsampled
                   image. At ±15, PSNR between original and optimized is ~31 dB
                   and the mean pixel change is ~5 levels.
        n_iterations: Gradient descent iterations. 100 is sufficient; the ±max_delta
                      constraint is the binding limit, not iteration count.
        lr: Learning rate for gradient descent. 0.3 works well across all tested
            configurations.

    Returns:
        Optimized source image, same shape as input, uint8.
    """
    target_h, target_w = target.shape[:2]
    source_h, source_w = source.shape[:2]

    target_f = target.astype(np.float64)
    source_f = source.astype(np.float64)
    source_orig = source_f.copy()
    best_energy = float("inf")
    best_source = source_f.copy()

    for _ in range(n_iterations):
        # Forward: upsample current source to target resolution
        cur = np.clip(source_f, 0, 255).astype(np.uint8)
        pred = np.array(
            Image.fromarray(cur).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
        ).astype(np.float64)

        # Loss: sum of squared error across all channels
        residual = target_f - pred
        energy = np.sum(residual * residual)

        if energy < best_energy:
            best_energy = energy
            best_source = source_f.copy()

        # Gradient: downsample residual (adjoint of bilinear upsample)
        # PIL requires per-channel resize for float data
        grad = np.empty_like(source_f)
        for c in range(source_f.shape[2]):
            grad[:, :, c] = np.array(
                Image.fromarray(residual[:, :, c].astype(np.float32)).resize(
                    (source_w, source_h), Image.Resampling.BILINEAR
                )
            )

        # Update and constrain
        source_f += lr * grad
        np.clip(source_f, source_orig - max_delta, source_orig + max_delta, out=source_f)
        np.clip(source_f, 0, 255, out=source_f)

    return np.clip(best_source, 0, 255).astype(np.uint8)


def optimize_for_upsample_grayscale(
    source: np.ndarray,
    target: np.ndarray,
    max_delta: int = 15,
    n_iterations: int = 100,
    lr: float = 0.3,
) -> np.ndarray:
    """Single-channel variant for grayscale images or individual YCbCr planes.

    Args:
        source: Downsampled single-channel image, shape (H, W) float or uint8.
        target: High-resolution target, shape (H*scale, W*scale) float or uint8.
        max_delta: Maximum pixel deviation from original (±).
        n_iterations: Gradient descent iterations.
        lr: Learning rate.

    Returns:
        Optimized source, same shape, float64 (preserves precision for chaining).
    """
    target_h, target_w = target.shape[:2]
    source_h, source_w = source.shape[:2]

    target_f = target.astype(np.float64)
    source_f = source.astype(np.float64)
    source_orig = source_f.copy()
    best_energy = float("inf")
    best_source = source_f.copy()

    for _ in range(n_iterations):
        # Forward: upsample
        cur = np.clip(source_f, 0, 255).astype(np.float32)
        pred = np.array(
            Image.fromarray(cur).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
        ).astype(np.float64)

        # Loss
        residual = target_f - pred
        energy = np.sum(residual * residual)

        if energy < best_energy:
            best_energy = energy
            best_source = source_f.copy()

        # Gradient: downsample residual
        grad = np.array(
            Image.fromarray(residual.astype(np.float32)).resize(
                (source_w, source_h), Image.Resampling.BILINEAR
            )
        ).astype(np.float64)

        # Update and constrain
        source_f += lr * grad
        np.clip(source_f, source_orig - max_delta, source_orig + max_delta, out=source_f)
        np.clip(source_f, 0, 255, out=source_f)

    return best_source
