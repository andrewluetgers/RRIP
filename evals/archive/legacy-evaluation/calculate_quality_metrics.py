#!/usr/bin/env python3
"""
Calculate PSNR and SSIM metrics for each parameter grid configuration.
Reconstructs tiles and compares against JPEG Q90 baseline.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple
import random
from skimage import transform
from scipy import ndimage

# Configuration
BASELINE_PYRAMID = Path("data/demo_out/baseline_pyramid_files")
GRID_RESULTS = Path("evaluation/grid_results")
OUTPUT_FILE = Path("evaluation/grid_quality_metrics.json")

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM between two images."""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = np.dot(img1[...,:3], [0.2989, 0.5870, 0.1140])
    if len(img2.shape) == 3:
        img2 = np.dot(img2[...,:3], [0.2989, 0.5870, 0.1140])

    # Simple SSIM approximation
    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Calculate means
    mu1 = ndimage.uniform_filter(img1, size=11, mode='constant')
    mu2 = ndimage.uniform_filter(img2, size=11, mode='constant')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Calculate variances
    sigma1_sq = ndimage.uniform_filter(img1 ** 2, size=11, mode='constant') - mu1_sq
    sigma2_sq = ndimage.uniform_filter(img2 ** 2, size=11, mode='constant') - mu2_sq
    sigma12 = ndimage.uniform_filter(img1 * img2, size=11, mode='constant') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))

def simulate_reconstruction(baseline_tile: np.ndarray, quant_levels: int, jpeg_quality: int) -> np.ndarray:
    """
    Simulate the effect of quantization and JPEG compression on residuals.
    This is a simplified approximation for evaluation purposes.
    """
    # Convert to float for processing
    tile_float = baseline_tile.astype(np.float32)

    # Convert RGB to YCbCr for processing (if RGB)
    if len(tile_float.shape) == 3:
        # Simple RGB to Y conversion (luminance only for residuals)
        tile_y = np.dot(tile_float[...,:3], [0.299, 0.587, 0.114])
    else:
        tile_y = tile_float

    # Simulate downsampling and upsampling (L2 -> L0 prediction)
    # Downsample by 4x
    h, w = tile_y.shape[:2]
    downsampled = transform.resize(tile_y, (h//4, w//4), order=1, preserve_range=True)
    predicted = transform.resize(downsampled, (h, w), order=1, preserve_range=True)

    # Calculate residual (luma only)
    residual = tile_y - predicted

    # Apply quantization
    if quant_levels < 256:
        # Map residual range to [0, 1]
        min_val = -255.0
        max_val = 255.0
        normalized = (residual - min_val) / (max_val - min_val)
        # Quantize
        quantized = np.round(normalized * (quant_levels - 1)) / (quant_levels - 1)
        # Map back
        residual = quantized * (max_val - min_val) + min_val

    # Offset for JPEG encoding
    residual_offset = np.clip(residual + 128, 0, 255).astype(np.uint8)

    # Simulate JPEG compression on residual
    # Convert to PIL Image, save as JPEG, and reload
    from io import BytesIO
    residual_img = Image.fromarray(residual_offset, mode='L')
    buffer = BytesIO()
    residual_img.save(buffer, format='JPEG', quality=jpeg_quality, optimize=True)
    buffer.seek(0)
    residual_decoded = np.array(Image.open(buffer).convert('L')).astype(np.float32)

    # Decode residual
    residual_final = residual_decoded - 128.0

    # Reconstruct luma
    reconstructed_y = predicted + residual_final
    reconstructed_y = np.clip(reconstructed_y, 0, 255)

    # For RGB images, combine with predicted chroma
    if len(baseline_tile.shape) == 3:
        # Use predicted image for chroma, reconstructed for luma
        predicted_rgb = transform.resize(
            transform.resize(baseline_tile.astype(np.float32), (h//4, w//4, 3), order=1, preserve_range=True),
            (h, w, 3), order=1, preserve_range=True
        )

        # Simple approach: replace luma in predicted image
        reconstructed = predicted_rgb.copy()
        # Adjust brightness based on luma difference
        luma_pred = np.dot(predicted_rgb[...,:3], [0.299, 0.587, 0.114])
        luma_scale = np.where(luma_pred > 0, reconstructed_y / (luma_pred + 1e-6), 1.0)
        reconstructed = reconstructed * luma_scale[..., np.newaxis]
    else:
        reconstructed = reconstructed_y

    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def evaluate_configuration(config_name: str, quant_levels: int, jpeg_quality: int,
                          sample_tiles: list) -> Dict:
    """Evaluate quality metrics for a specific configuration."""

    psnr_values = []
    ssim_values = []

    for tile_path in sample_tiles:
        # Load baseline tile
        baseline_img = np.array(Image.open(tile_path).convert('RGB'))

        # Simulate reconstruction with parameters
        reconstructed = simulate_reconstruction(baseline_img, quant_levels, jpeg_quality)

        # Calculate metrics
        psnr = calculate_psnr(baseline_img, reconstructed)
        ssim = calculate_ssim(baseline_img, reconstructed)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    return {
        'config_name': config_name,
        'quantization_levels': quant_levels,
        'jpeg_quality': jpeg_quality,
        'psnr_mean': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values),
        'psnr_min': np.min(psnr_values),
        'psnr_max': np.max(psnr_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'ssim_min': np.min(ssim_values),
        'ssim_max': np.max(ssim_values),
        'num_tiles': len(sample_tiles)
    }

def main():
    """Calculate quality metrics for all configurations."""

    print("Calculating Quality Metrics for Parameter Grid")
    print("=" * 60)

    # Load compression results
    compression_file = GRID_RESULTS / "all_configurations.json"
    if not compression_file.exists():
        print(f"Error: Compression results not found at {compression_file}")
        return

    with open(compression_file) as f:
        compression_data = json.load(f)

    # Sample tiles for evaluation (L0 tiles)
    l0_dir = BASELINE_PYRAMID / "16"  # Level 16 is L0
    if not l0_dir.exists():
        l0_dir = BASELINE_PYRAMID / "15"  # Try level 15

    all_tiles = list(l0_dir.glob("*.jpg"))
    if len(all_tiles) > 20:
        # Sample 20 random tiles for evaluation
        sample_tiles = random.sample(all_tiles, 20)
    else:
        sample_tiles = all_tiles

    print(f"Using {len(sample_tiles)} sample tiles from {l0_dir}")

    # Evaluate each configuration
    results = {}

    # Define configurations
    configs = [
        ("q16_j30", 16, 30),
        ("q16_j60", 16, 60),
        ("q16_j90", 16, 90),
        ("q32_j30", 32, 30),
        ("q32_j60", 32, 60),
        ("q32_j90", 32, 90),
        ("q64_j30", 64, 30),
        ("q64_j60", 64, 60),
        ("q64_j90", 64, 90),
    ]

    for config_name, quant_levels, jpeg_quality in configs:
        print(f"\nEvaluating {config_name}...")

        metrics = evaluate_configuration(config_name, quant_levels, jpeg_quality, sample_tiles)

        # Add compression data
        if config_name in compression_data:
            metrics['compression_ratio'] = compression_data[config_name].get('compression_ratio', 0)
            metrics['storage_savings_pct'] = compression_data[config_name].get('savings_pct', 0)

        results[config_name] = metrics

        print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("QUALITY METRICS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<12} {'Quant':<6} {'JPEG':<6} {'Compression':<12} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 80)

    # Sort by compression ratio for display
    sorted_configs = sorted(results.items(),
                           key=lambda x: x[1].get('compression_ratio', 0),
                           reverse=True)

    for config_name, metrics in sorted_configs:
        quant = metrics['quantization_levels']
        jpeg = metrics['jpeg_quality']
        comp = metrics.get('compression_ratio', 0)
        psnr = metrics['psnr_mean']
        ssim = metrics['ssim_mean']

        print(f"{config_name:<12} {quant:<6} {jpeg:<6} {comp:<12.2f} {psnr:<12.2f} {ssim:<10.4f}")

    # Find optimal configurations
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATIONS")
    print("=" * 80)

    # Best compression with acceptable quality (PSNR > 40)
    acceptable = [(k, v) for k, v in results.items() if v['psnr_mean'] > 40]
    if acceptable:
        best_comp = max(acceptable, key=lambda x: x[1].get('compression_ratio', 0))
        print(f"Best Compression (PSNR>40): {best_comp[0]}")
        print(f"  Compression: {best_comp[1].get('compression_ratio', 0):.2f}x")
        print(f"  PSNR: {best_comp[1]['psnr_mean']:.2f} dB")
        print(f"  SSIM: {best_comp[1]['ssim_mean']:.4f}")

    # Best quality
    best_quality = max(results.items(), key=lambda x: x[1]['psnr_mean'])
    print(f"\nBest Quality: {best_quality[0]}")
    print(f"  PSNR: {best_quality[1]['psnr_mean']:.2f} dB")
    print(f"  SSIM: {best_quality[1]['ssim_mean']:.4f}")
    print(f"  Compression: {best_quality[1].get('compression_ratio', 0):.2f}x")

    # Most balanced (quality * compression score)
    balanced_scores = {}
    for k, v in results.items():
        # Normalize PSNR to 0-1 (assuming 30-50 dB range)
        psnr_norm = (v['psnr_mean'] - 30) / 20
        # Normalize compression to 0-1 (assuming 5-11x range)
        comp_norm = (v.get('compression_ratio', 0) - 5) / 6
        # Combined score (equal weight)
        balanced_scores[k] = psnr_norm * 0.5 + comp_norm * 0.5

    best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
    print(f"\nMost Balanced: {best_balanced[0]}")
    print(f"  Compression: {results[best_balanced[0]].get('compression_ratio', 0):.2f}x")
    print(f"  PSNR: {results[best_balanced[0]]['psnr_mean']:.2f} dB")
    print(f"  SSIM: {results[best_balanced[0]]['ssim_mean']:.4f}")
    print(f"  Balance Score: {best_balanced[1]:.3f}")

    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()