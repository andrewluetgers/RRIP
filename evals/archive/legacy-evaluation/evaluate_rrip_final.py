#!/usr/bin/env python3
"""
ORIGAMI Final Evaluation Script
Compares ORIGAMI compression against JPEG using actual pyramid levels
L16 = highest res (L0 in ORIGAMI), L15 = L1 in ORIGAMI, L14 = L2 in ORIGAMI
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import requests
import io
from dataclasses import dataclass, asdict

@dataclass
class TestResult:
    """Single compression test result"""
    method: str
    quality_param: float
    level: int
    tile_coords: Tuple[int, int]
    file_size_bytes: int
    decode_time_ms: float
    psnr_db: float
    ssim: float
    ms_ssim: float
    bits_per_pixel: float

class QualityMetrics:
    """Calculate image quality metrics"""

    @staticmethod
    def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate PSNR in dB"""
        # Ensure same shape
        if original.shape != compressed.shape:
            return 0.0

        mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM"""
        if img1.shape != img2.shape:
            return 0.0

        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Constants for SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Compute SSIM
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    @staticmethod
    def calculate_ms_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Multi-Scale SSIM"""
        if img1.shape != img2.shape:
            return 0.0

        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        ms_ssim_values = []

        for i in range(len(weights)):
            ssim_val = QualityMetrics.calculate_ssim(img1, img2)
            ms_ssim_values.append(ssim_val)

            if i < len(weights) - 1:
                # Downsample for next scale
                img1 = cv2.pyrDown(img1)
                img2 = cv2.pyrDown(img2)

                # Check minimum size
                if img1.shape[0] < 11 or img1.shape[1] < 11:
                    break

        # Calculate weighted product
        ms_ssim = 1.0
        for i, val in enumerate(ms_ssim_values[:len(weights)]):
            ms_ssim *= val ** weights[i]

        return float(ms_ssim)

def get_test_tiles(data_dir: Path, level: int, count: int = 50) -> List[Tuple[int, int]]:
    """Get random test tiles from a specific level"""
    level_dir = data_dir / "baseline_pyramid_files" / str(level)

    if not level_dir.exists():
        return []

    # Get all available tiles
    tiles = list(level_dir.glob("*.jpg"))

    # Extract coordinates
    coords = []
    for tile in tiles:
        try:
            x, y = tile.stem.split('_')
            coords.append((int(x), int(y)))
        except:
            continue

    # Random sample
    np.random.seed(42)  # For reproducibility
    if len(coords) > count:
        indices = np.random.choice(len(coords), count, replace=False)
        coords = [coords[i] for i in indices]

    return coords

def test_origami_compression(slide_id: str, level: int, x: int, y: int,
                         data_dir: Path, server_url: str) -> TestResult:
    """Test ORIGAMI compression for a specific tile"""

    # Get original tile
    orig_path = data_dir / "baseline_pyramid_files" / str(level) / f"{x}_{y}.jpg"
    if not orig_path.exists():
        return None

    original = np.array(Image.open(orig_path))

    # Fetch ORIGAMI reconstruction from server
    start = time.time()
    response = requests.get(f"{server_url}/tiles/{slide_id}/{level}/{x}_{y}.jpg")
    decode_time = (time.time() - start) * 1000  # ms

    if response.status_code != 200:
        return None

    # Decode response
    reconstructed = np.array(Image.open(io.BytesIO(response.content)))

    # Estimate ORIGAMI size
    # For L16/L15 (L0/L1 in ORIGAMI): size includes L14 baseline + residuals
    if level >= 15:
        # L14 coordinates (L2 in ORIGAMI)
        shift = 16 - 14  # 2 levels up
        x14 = x >> shift
        y14 = y >> shift

        # Check for pack file
        pack_path = data_dir / "residual_packs" / f"{x14}_{y14}.pack"
        if pack_path.exists():
            # Pack contains 4 L15 + 16 L16 tiles = 20 tiles
            pack_size = pack_path.stat().st_size
            effective_size = pack_size // 20
        else:
            # Estimate from individual files
            l14_path = data_dir / "baseline_pyramid_files" / "14" / f"{x14}_{y14}.jpg"
            l14_size = l14_path.stat().st_size if l14_path.exists() else 25000

            # Add residual size
            if level == 16:  # L0 in ORIGAMI
                res_path = data_dir / f"residuals_q32/L0/{x14}_{y14}/{x}_{y}.jpg"
                tiles_per_l14 = 16
            else:  # L15 = L1 in ORIGAMI
                res_path = data_dir / f"residuals_q32/L1/{x14}_{y14}/{x}_{y}.jpg"
                tiles_per_l14 = 4

            res_size = res_path.stat().st_size if res_path.exists() else 5000
            effective_size = l14_size // tiles_per_l14 + res_size
    else:
        # L14 and below served directly
        effective_size = orig_path.stat().st_size

    # Calculate metrics
    metrics = QualityMetrics()

    return TestResult(
        method="ORIGAMI",
        quality_param=32,  # Residual quality
        level=level,
        tile_coords=(x, y),
        file_size_bytes=effective_size,
        decode_time_ms=decode_time,
        psnr_db=metrics.calculate_psnr(original, reconstructed),
        ssim=metrics.calculate_ssim(original, reconstructed),
        ms_ssim=metrics.calculate_ms_ssim(original, reconstructed),
        bits_per_pixel=effective_size * 8 / (256 * 256)
    )

def test_jpeg_recompression(image_path: Path, quality: int) -> TestResult:
    """Test JPEG recompression"""
    original = np.array(Image.open(image_path))

    # Recompress
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_bytes = cv2.imencode('.jpg', cv2.cvtColor(original, cv2.COLOR_RGB2BGR), encode_param)

    # Decode
    decoded = cv2.imdecode(compressed_bytes, cv2.IMREAD_COLOR)
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    # Calculate metrics
    metrics = QualityMetrics()

    # Extract coords from filename
    try:
        x, y = image_path.stem.split('_')
        coords = (int(x), int(y))
    except:
        coords = (0, 0)

    return TestResult(
        method="JPEG",
        quality_param=quality,
        level=int(image_path.parent.name),
        tile_coords=coords,
        file_size_bytes=len(compressed_bytes),
        decode_time_ms=0,  # Not measured for JPEG
        psnr_db=metrics.calculate_psnr(original, decoded),
        ssim=metrics.calculate_ssim(original, decoded),
        ms_ssim=metrics.calculate_ms_ssim(original, decoded),
        bits_per_pixel=len(compressed_bytes) * 8 / (256 * 256)
    )

def plot_results(results: List[TestResult], output_dir: Path):
    """Generate publication-quality plots"""

    # Group by method and level
    origami_results = [r for r in results if r.method == "ORIGAMI"]
    jpeg_results = [r for r in results if r.method == "JPEG"]

    # Separate by level
    levels = [16, 15, 14]  # L0, L1, L2 in ORIGAMI terms

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ORIGAMI Compression Evaluation: Rate-Distortion Performance by Level',
                fontsize=16, fontweight='bold')

    for idx, level in enumerate(levels):
        ax_psnr = axes[0, idx]
        ax_ssim = axes[1, idx]

        # Filter results for this level
        origami_level = [r for r in origami_results if r.level == level]
        jpeg_level = [r for r in jpeg_results if r.level == level]

        # ORIGAMI point
        if origami_level:
            origami_bpp = np.mean([r.bits_per_pixel for r in origami_level])
            origami_psnr = np.mean([r.psnr_db for r in origami_level])
            origami_ssim = np.mean([r.ms_ssim for r in origami_level])

            ax_psnr.plot(origami_bpp, origami_psnr, 'ro', markersize=12, label='ORIGAMI')
            ax_ssim.plot(origami_bpp, origami_ssim, 'ro', markersize=12, label='ORIGAMI')

        # JPEG curve
        if jpeg_level:
            # Group by quality
            jpeg_by_q = {}
            for r in jpeg_level:
                q = int(r.quality_param)
                if q not in jpeg_by_q:
                    jpeg_by_q[q] = []
                jpeg_by_q[q].append(r)

            # Calculate averages
            jpeg_points = []
            for q in sorted(jpeg_by_q.keys(), reverse=True):
                results_q = jpeg_by_q[q]
                jpeg_points.append({
                    'quality': q,
                    'bpp': np.mean([r.bits_per_pixel for r in results_q]),
                    'psnr': np.mean([r.psnr_db for r in results_q]),
                    'ssim': np.mean([r.ms_ssim for r in results_q])
                })

            # Plot JPEG curve
            if jpeg_points:
                jpeg_bpp = [p['bpp'] for p in jpeg_points]
                jpeg_psnr = [p['psnr'] for p in jpeg_points]
                jpeg_ssim = [p['ssim'] for p in jpeg_points]

                ax_psnr.plot(jpeg_bpp, jpeg_psnr, 'b-', marker='s', label='JPEG')
                ax_ssim.plot(jpeg_bpp, jpeg_ssim, 'b-', marker='s', label='JPEG')

        # Format axes
        origami_level_name = {16: 'L0', 15: 'L1', 14: 'L2'}[level]
        ax_psnr.set_title(f'Level {level} ({origami_level_name} in ORIGAMI)', fontweight='bold')
        ax_psnr.set_xlabel('Bits Per Pixel')
        ax_psnr.set_ylabel('PSNR (dB)')
        ax_psnr.grid(True, alpha=0.3)
        ax_psnr.legend()

        ax_ssim.set_xlabel('Bits Per Pixel')
        ax_ssim.set_ylabel('MS-SSIM')
        ax_ssim.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rd_curves_by_level.png', dpi=300, bbox_inches='tight')
    print(f"Saved R-D curves to {output_dir / 'rd_curves_by_level.png'}")

    # Generate combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ORIGAMI vs JPEG: Combined Rate-Distortion Performance',
                fontsize=14, fontweight='bold')

    # Combined ORIGAMI (all levels)
    if origami_results:
        origami_bpp = np.mean([r.bits_per_pixel for r in origami_results])
        origami_psnr = np.mean([r.psnr_db for r in origami_results])
        origami_ssim = np.mean([r.ms_ssim for r in origami_results])

        ax1.plot(origami_bpp, origami_psnr, 'ro', markersize=12, label='ORIGAMI', zorder=10)
        ax2.plot(origami_bpp, origami_ssim, 'ro', markersize=12, label='ORIGAMI', zorder=10)

        # Add annotation
        ax1.annotate('ORIGAMI', xy=(origami_bpp, origami_psnr),
                    xytext=(origami_bpp + 0.5, origami_psnr - 1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontweight='bold', color='red')

    # Combined JPEG curve
    jpeg_by_q = {}
    for r in jpeg_results:
        q = int(r.quality_param)
        if q not in jpeg_by_q:
            jpeg_by_q[q] = []
        jpeg_by_q[q].append(r)

    jpeg_points = []
    for q in sorted(jpeg_by_q.keys(), reverse=True):
        results_q = jpeg_by_q[q]
        jpeg_points.append({
            'quality': q,
            'bpp': np.mean([r.bits_per_pixel for r in results_q]),
            'psnr': np.mean([r.psnr_db for r in results_q]),
            'ssim': np.mean([r.ms_ssim for r in results_q])
        })

    if jpeg_points:
        jpeg_bpp = [p['bpp'] for p in jpeg_points]
        jpeg_psnr = [p['psnr'] for p in jpeg_points]
        jpeg_ssim = [p['ssim'] for p in jpeg_points]

        ax1.plot(jpeg_bpp, jpeg_psnr, 'b-', marker='s', label='JPEG', markersize=8)
        ax2.plot(jpeg_bpp, jpeg_ssim, 'b-', marker='s', label='JPEG', markersize=8)

    ax1.set_xlabel('Bits Per Pixel', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Bits Per Pixel', fontsize=12)
    ax2.set_ylabel('MS-SSIM', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rd_curves_combined.png', dpi=300, bbox_inches='tight')
    print(f"Saved combined R-D curves to {output_dir / 'rd_curves_combined.png'}")

def main():
    """Run the evaluation"""

    # Configuration
    SLIDE_ID = "demo_out"
    DATA_DIR = Path("/Users/andrewluetgers/projects/dev/ORIGAMI/data") / SLIDE_ID
    SERVER_URL = "http://localhost:3007"
    OUTPUT_DIR = Path("evaluation_results")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*80)
    print("ORIGAMI COMPRESSION EVALUATION")
    print("="*80)
    print(f"\nSlide: {SLIDE_ID}")
    print(f"Server: {SERVER_URL}\n")

    # Check server
    try:
        response = requests.get(f"{SERVER_URL}/healthz")
        print(f"✓ Server health check: {response.status_code}\n")
    except:
        print("✗ Server not responding. Please start it with:")
        print("  cargo run --manifest-path server/Cargo.toml -- --slides-root data --port 3007")
        return

    all_results = []

    # Test each level
    for level in [16, 15, 14]:  # L0, L1, L2 in ORIGAMI terms
        origami_name = {16: 'L0', 15: 'L1', 14: 'L2'}[level]
        print(f"Testing Level {level} ({origami_name} in ORIGAMI)...")

        # Get test tiles
        test_tiles = get_test_tiles(DATA_DIR, level, count=30)
        print(f"  Found {len(test_tiles)} test tiles")

        # Test ORIGAMI
        print(f"  Testing ORIGAMI reconstruction...")
        origami_count = 0
        for x, y in test_tiles:
            result = test_origami_compression(SLIDE_ID, level, x, y, DATA_DIR, SERVER_URL)
            if result:
                all_results.append(result)
                origami_count += 1

        print(f"    Processed {origami_count} tiles")

        # Test JPEG recompression (subset for speed)
        print(f"  Testing JPEG recompression...")
        jpeg_qualities = [95, 90, 85, 80, 70, 60]

        for quality in jpeg_qualities:
            jpeg_count = 0
            for x, y in test_tiles[:10]:  # Test subset
                tile_path = DATA_DIR / "baseline_pyramid_files" / str(level) / f"{x}_{y}.jpg"
                if tile_path.exists():
                    result = test_jpeg_recompression(tile_path, quality)
                    if result:
                        all_results.append(result)
                        jpeg_count += 1

            print(f"    Q{quality}: {jpeg_count} tiles")

        print()

    # Calculate summary statistics
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Group by method
    origami_results = [r for r in all_results if r.method == "ORIGAMI"]
    jpeg_results = [r for r in all_results if r.method == "JPEG"]

    if origami_results:
        print(f"\nORIGAMI (n={len(origami_results)}):")
        print(f"  Average file size: {np.mean([r.file_size_bytes for r in origami_results])/1024:.1f} KB")
        print(f"  Average PSNR: {np.mean([r.psnr_db for r in origami_results]):.2f} dB")
        print(f"  Average MS-SSIM: {np.mean([r.ms_ssim for r in origami_results]):.4f}")
        print(f"  Average BPP: {np.mean([r.bits_per_pixel for r in origami_results]):.3f}")
        print(f"  Average decode time: {np.mean([r.decode_time_ms for r in origami_results]):.2f} ms")

    # JPEG by quality
    jpeg_by_q = {}
    for r in jpeg_results:
        q = int(r.quality_param)
        if q not in jpeg_by_q:
            jpeg_by_q[q] = []
        jpeg_by_q[q].append(r)

    print("\nJPEG Recompression:")
    for q in sorted(jpeg_by_q.keys(), reverse=True):
        results_q = jpeg_by_q[q]
        print(f"  Q{q} (n={len(results_q)}):")
        print(f"    File size: {np.mean([r.file_size_bytes for r in results_q])/1024:.1f} KB")
        print(f"    PSNR: {np.mean([r.psnr_db for r in results_q]):.2f} dB")
        print(f"    MS-SSIM: {np.mean([r.ms_ssim for r in results_q]):.4f}")
        print(f"    BPP: {np.mean([r.bits_per_pixel for r in results_q]):.3f}")

    # Save results
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Generate plots
    print("\nGenerating plots...")
    plot_results(all_results, OUTPUT_DIR)

    print(f"\n✓ Evaluation complete! Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()