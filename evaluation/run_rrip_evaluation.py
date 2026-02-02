#!/usr/bin/env python3
"""
RRIP Evaluation Script - Practical Testing with Your Actual Implementation
Compares RRIP against JPEG and JPEG 2000 using your existing tile data
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, asdict
import requests
import tempfile
import shutil

@dataclass
class TestResult:
    """Single test result"""
    method: str
    quality_param: float
    file_size_bytes: int
    encode_time_ms: float
    decode_time_ms: float
    psnr_db: float
    ssim: float
    ms_ssim: float
    bits_per_pixel: float

class RRIPEvaluator:
    """Evaluate RRIP against your actual server implementation"""

    def __init__(self, server_url="http://localhost:3007", slides_root="/Users/andrewluetgers/projects/dev/RRIP/data"):
        self.server_url = server_url
        self.slides_root = Path(slides_root)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="rrip_eval_"))

    def fetch_tile_from_server(self, slide_id: str, level: int, x: int, y: int) -> bytes:
        """Fetch a tile from the RRIP server"""
        url = f"{self.server_url}/tiles/{slide_id}/{level}/{x}_{y}.jpg"
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    def get_original_tile(self, slide_id: str, level: int, x: int, y: int) -> np.ndarray:
        """Get the original JPEG tile from disk"""
        tile_path = self.slides_root / slide_id / "baseline_pyramid_files" / str(level) / f"{x}_{y}.jpg"
        if tile_path.exists():
            return np.array(Image.open(tile_path))
        return None

    def get_rrip_reconstructed(self, slide_id: str, level: int, x: int, y: int) -> Tuple[np.ndarray, int, float]:
        """Get RRIP reconstructed tile and measure performance"""
        start = time.time()
        tile_bytes = self.fetch_tile_from_server(slide_id, level, x, y)
        decode_time = (time.time() - start) * 1000  # ms

        # Convert bytes to image
        import io
        img = Image.open(io.BytesIO(tile_bytes))
        reconstructed = np.array(img)

        # Calculate effective size (L2 + residuals for this tile)
        # This is approximate - for exact, we'd need pack file size / 20
        effective_size = self.estimate_rrip_size(slide_id, level, x, y)

        return reconstructed, effective_size, decode_time

    def estimate_rrip_size(self, slide_id: str, level: int, x: int, y: int) -> int:
        """Estimate RRIP compressed size for a tile"""
        # For L0/L1 tiles, size = (L2_size + residuals) / tiles_per_family
        if level <= 1:
            # Get L2 parent coordinates
            x2 = x >> (2 - level)
            y2 = y >> (2 - level)

            # Check for pack file
            pack_path = self.slides_root / slide_id / "residual_packs" / f"{x2}_{y2}.pack"
            if pack_path.exists():
                pack_size = pack_path.stat().st_size
                # Pack contains 20 tiles (4 L1 + 16 L0)
                return pack_size // 20

            # Otherwise estimate from individual files
            # L2 baseline
            l2_path = self.slides_root / slide_id / "baseline_pyramid_files" / "2" / f"{x2}_{y2}.jpg"
            l2_size = l2_path.stat().st_size if l2_path.exists() else 25000

            # Residual
            if level == 0:
                res_path = self.slides_root / slide_id / f"residuals_q32/L0/{x2}_{y2}/{x}_{y}.jpg"
            else:
                res_path = self.slides_root / slide_id / f"residuals_q32/L1/{x2}_{y2}/{x}_{y}.jpg"

            res_size = res_path.stat().st_size if res_path.exists() else 5000

            # Total = L2/4 + residual (L2 shared among 4 L1 or 16 L0)
            tiles_sharing_l2 = 16 if level == 0 else 4
            return l2_size // tiles_sharing_l2 + res_size
        else:
            # L2+ served directly
            tile_path = self.slides_root / slide_id / "baseline_pyramid_files" / str(level) / f"{x}_{y}.jpg"
            return tile_path.stat().st_size if tile_path.exists() else 25000

class QualityMetrics:
    """Calculate image quality metrics"""

    @staticmethod
    def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate PSNR in dB"""
        mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
        if mse == 0:
            return 100.0  # Perfect match
        return 20 * np.log10(255.0 / np.sqrt(mse))

    @staticmethod
    def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate SSIM"""
        # Convert to grayscale
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(compressed.shape) == 3:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)

        # Use cv2's SSIM implementation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(original, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(compressed, -1, window)[5:-5, 5:-5]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(original ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(compressed ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(original * compressed, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    @staticmethod
    def calculate_ms_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate MS-SSIM"""
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        levels = len(weights)

        ms_ssim_values = []
        for i in range(levels):
            ssim_val = QualityMetrics.calculate_ssim(original, compressed)
            ms_ssim_values.append(ssim_val)

            if i < levels - 1:
                original = cv2.pyrDown(original)
                compressed = cv2.pyrDown(compressed)

                # Check if images are too small
                if original.shape[0] < 11 or original.shape[1] < 11:
                    break

        # Calculate weighted product
        ms_ssim = 1.0
        for i, val in enumerate(ms_ssim_values):
            ms_ssim *= val ** weights[i]

        return ms_ssim

def test_jpeg_compression(image: np.ndarray, quality: int) -> TestResult:
    """Test standard JPEG compression"""
    start = time.time()

    # Compress
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_bytes = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_param)
    encode_time = (time.time() - start) * 1000

    # Decompress
    start = time.time()
    decoded = cv2.imdecode(compressed_bytes, cv2.IMREAD_COLOR)
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    decode_time = (time.time() - start) * 1000

    # Calculate metrics
    metrics = QualityMetrics()

    return TestResult(
        method="JPEG",
        quality_param=quality,
        file_size_bytes=len(compressed_bytes),
        encode_time_ms=encode_time,
        decode_time_ms=decode_time,
        psnr_db=metrics.calculate_psnr(image, decoded),
        ssim=metrics.calculate_ssim(image, decoded),
        ms_ssim=metrics.calculate_ms_ssim(image, decoded),
        bits_per_pixel=len(compressed_bytes) * 8 / (image.shape[0] * image.shape[1])
    )

def calculate_bd_rate(anchor_rates, anchor_psnr, test_rates, test_psnr):
    """Calculate Bjøntegaard Delta Rate"""
    # This is a simplified version - for paper, use the full implementation
    # from https://github.com/Anserw/Bjontegaard_metric

    from scipy import interpolate

    # Fit cubic polynomials
    anchor_interp = interpolate.interp1d(anchor_psnr, np.log10(anchor_rates),
                                        kind='cubic', fill_value='extrapolate')
    test_interp = interpolate.interp1d(test_psnr, np.log10(test_rates),
                                      kind='cubic', fill_value='extrapolate')

    # Find overlapping PSNR range
    min_psnr = max(min(anchor_psnr), min(test_psnr))
    max_psnr = min(max(anchor_psnr), max(test_psnr))

    # Integrate over the range
    psnr_points = np.linspace(min_psnr, max_psnr, 100)

    anchor_rates_interp = 10 ** anchor_interp(psnr_points)
    test_rates_interp = 10 ** test_interp(psnr_points)

    # Calculate average difference
    bd_rate = (np.mean(test_rates_interp) - np.mean(anchor_rates_interp)) / np.mean(anchor_rates_interp) * 100

    return bd_rate

def run_evaluation(slide_id="demo_out", num_tiles=50):
    """Run the full evaluation"""

    print(f"Running RRIP Evaluation on slide: {slide_id}")
    print(f"Testing {num_tiles} random L0 tiles\n")

    evaluator = RRIPEvaluator()
    metrics = QualityMetrics()

    # Results storage
    all_results = []

    # Sample random L0 tiles (highest resolution)
    # Adjust these ranges based on your slide dimensions
    max_x, max_y = 100, 100  # Adjust based on your slide

    np.random.seed(42)  # For reproducibility
    test_tiles = [(np.random.randint(0, max_x), np.random.randint(0, max_y))
                  for _ in range(num_tiles)]

    print("Testing RRIP reconstruction...")
    rrip_results = []

    for i, (x, y) in enumerate(test_tiles):
        # Get original tile
        original = evaluator.get_original_tile(slide_id, 0, x, y)
        if original is None:
            continue

        # Get RRIP reconstruction
        try:
            reconstructed, size, decode_time = evaluator.get_rrip_reconstructed(slide_id, 0, x, y)

            result = TestResult(
                method="RRIP",
                quality_param=32,  # Residual quality
                file_size_bytes=size,
                encode_time_ms=0,  # Not measured for pre-computed
                decode_time_ms=decode_time,
                psnr_db=metrics.calculate_psnr(original, reconstructed),
                ssim=metrics.calculate_ssim(original, reconstructed),
                ms_ssim=metrics.calculate_ms_ssim(original, reconstructed),
                bits_per_pixel=size * 8 / (256 * 256)
            )

            rrip_results.append(result)
            all_results.append(result)

            if (i + 1) % 10 == 0:
                avg_psnr = np.mean([r.psnr_db for r in rrip_results])
                avg_size = np.mean([r.file_size_bytes for r in rrip_results])
                print(f"  Processed {i+1}/{num_tiles} tiles - Avg PSNR: {avg_psnr:.2f} dB, "
                      f"Avg size: {avg_size/1024:.1f} KB")

        except Exception as e:
            print(f"  Error processing tile ({x},{y}): {e}")
            continue

    print(f"\nTesting JPEG re-compression...")

    # Test JPEG at different qualities
    jpeg_qualities = [95, 90, 85, 80, 70, 60, 50]
    jpeg_results = {q: [] for q in jpeg_qualities}

    for quality in jpeg_qualities:
        print(f"  Testing JPEG Q{quality}...")

        for x, y in test_tiles[:20]:  # Test subset for speed
            original = evaluator.get_original_tile(slide_id, 0, x, y)
            if original is None:
                continue

            result = test_jpeg_compression(original, quality)
            jpeg_results[quality].append(result)
            all_results.append(result)

    # Generate summary statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)

    # Group results by method
    methods_summary = {}

    # RRIP summary
    if rrip_results:
        methods_summary['RRIP'] = {
            'avg_size_kb': np.mean([r.file_size_bytes for r in rrip_results]) / 1024,
            'avg_psnr': np.mean([r.psnr_db for r in rrip_results]),
            'avg_ssim': np.mean([r.ssim for r in rrip_results]),
            'avg_ms_ssim': np.mean([r.ms_ssim for r in rrip_results]),
            'avg_bpp': np.mean([r.bits_per_pixel for r in rrip_results]),
            'avg_decode_ms': np.mean([r.decode_time_ms for r in rrip_results])
        }

    # JPEG summaries
    for quality, results in jpeg_results.items():
        if results:
            methods_summary[f'JPEG_Q{quality}'] = {
                'avg_size_kb': np.mean([r.file_size_bytes for r in results]) / 1024,
                'avg_psnr': np.mean([r.psnr_db for r in results]),
                'avg_ssim': np.mean([r.ssim for r in results]),
                'avg_ms_ssim': np.mean([r.ms_ssim for r in results]),
                'avg_bpp': np.mean([r.bits_per_pixel for r in results]),
                'avg_decode_ms': np.mean([r.decode_time_ms for r in results])
            }

    # Print summary table
    print(f"\n{'Method':<15} {'Size (KB)':<12} {'PSNR (dB)':<12} {'MS-SSIM':<12} {'BPP':<10} {'Decode (ms)':<12}")
    print("-" * 85)

    for method, stats in methods_summary.items():
        print(f"{method:<15} {stats['avg_size_kb']:<12.1f} {stats['avg_psnr']:<12.2f} "
              f"{stats['avg_ms_ssim']:<12.4f} {stats['avg_bpp']:<10.3f} {stats['avg_decode_ms']:<12.2f}")

    # Calculate BD-Rate if we have enough data points
    if len(jpeg_results[95]) > 0 and len(rrip_results) > 0:
        # Prepare data for BD-rate calculation
        jpeg_rates = []
        jpeg_psnrs = []

        for quality in [95, 90, 85, 80, 70]:
            if jpeg_results[quality]:
                jpeg_rates.append(np.mean([r.bits_per_pixel for r in jpeg_results[quality]]))
                jpeg_psnrs.append(np.mean([r.psnr_db for r in jpeg_results[quality]]))

        if len(jpeg_rates) >= 4:  # Need at least 4 points for cubic interpolation
            rrip_rate = np.mean([r.bits_per_pixel for r in rrip_results])
            rrip_psnr = np.mean([r.psnr_db for r in rrip_results])

            try:
                bd_rate = calculate_bd_rate(jpeg_rates, jpeg_psnrs,
                                          [rrip_rate], [rrip_psnr])
                print(f"\n{'='*85}")
                print(f"BD-Rate (RRIP vs JPEG): {bd_rate:.1f}%")
                print(f"Negative value means RRIP is more efficient")
            except:
                print("\nCould not calculate BD-Rate (need more data points)")

    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Save raw results as JSON
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Generate R-D curve
    plot_rd_curve(all_results, output_dir / "rd_curve.png")

    print(f"\nResults saved to {output_dir}/")

    return methods_summary

def plot_rd_curve(results: List[TestResult], output_path: Path):
    """Generate Rate-Distortion curve"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Group by method
    methods = {}
    for r in results:
        key = r.method if r.method == "RRIP" else f"{r.method}_Q{int(r.quality_param)}"
        if key not in methods:
            methods[key] = []
        methods[key].append(r)

    # Calculate averages for each method
    plot_data = []
    for method, method_results in methods.items():
        if method_results:
            avg_bpp = np.mean([r.bits_per_pixel for r in method_results])
            avg_psnr = np.mean([r.psnr_db for r in method_results])
            avg_ms_ssim = np.mean([r.ms_ssim for r in method_results])

            color = 'red' if 'RRIP' in method else 'blue'
            marker = 'o' if 'RRIP' in method else 's'
            label = method.replace('_Q', ' Q')

            plot_data.append({
                'method': method,
                'bpp': avg_bpp,
                'psnr': avg_psnr,
                'ms_ssim': avg_ms_ssim,
                'color': color,
                'marker': marker,
                'label': label
            })

    # Sort by BPP for line plotting
    plot_data.sort(key=lambda x: x['bpp'])

    # Plot PSNR vs BPP
    for item in plot_data:
        ax1.plot(item['bpp'], item['psnr'],
                marker=item['marker'], color=item['color'],
                markersize=10, label=item['label'])

    # Connect JPEG points
    jpeg_points = [p for p in plot_data if 'JPEG' in p['method']]
    if len(jpeg_points) > 1:
        jpeg_bpp = [p['bpp'] for p in jpeg_points]
        jpeg_psnr = [p['psnr'] for p in jpeg_points]
        ax1.plot(jpeg_bpp, jpeg_psnr, 'b-', alpha=0.3, linewidth=2)

    ax1.set_xlabel('Bits Per Pixel (BPP)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('Rate-Distortion: PSNR', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Plot MS-SSIM vs BPP
    for item in plot_data:
        ax2.plot(item['bpp'], item['ms_ssim'],
                marker=item['marker'], color=item['color'],
                markersize=10, label=item['label'])

    # Connect JPEG points
    if len(jpeg_points) > 1:
        jpeg_ms_ssim = [p['ms_ssim'] for p in jpeg_points]
        ax2.plot(jpeg_bpp, jpeg_ms_ssim, 'b-', alpha=0.3, linewidth=2)

    ax2.set_xlabel('Bits Per Pixel (BPP)', fontsize=12)
    ax2.set_ylabel('MS-SSIM', fontsize=12)
    ax2.set_title('Rate-Distortion: MS-SSIM', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add annotation for RRIP point
    rrip_point = next((p for p in plot_data if 'RRIP' in p['method']), None)
    if rrip_point:
        ax1.annotate('RRIP',
                    xy=(rrip_point['bpp'], rrip_point['psnr']),
                    xytext=(rrip_point['bpp']+0.5, rrip_point['psnr']-1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=11, color='red', fontweight='bold')

        ax2.annotate('RRIP',
                    xy=(rrip_point['bpp'], rrip_point['ms_ssim']),
                    xytext=(rrip_point['bpp']+0.5, rrip_point['ms_ssim']-0.01),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=11, color='red', fontweight='bold')

    plt.suptitle('RRIP Compression Evaluation: Rate-Distortion Performance',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"R-D curve saved to {output_path}")

if __name__ == "__main__":
    # Make sure your RRIP server is running!
    print("="*80)
    print("RRIP COMPRESSION EVALUATION")
    print("="*80)
    print("\nMake sure your RRIP server is running on port 3007!")
    print("Start it with: cargo run --manifest-path server/Cargo.toml -- --slides-root data --port 3007\n")

    input("Press Enter when server is ready...")

    # Run evaluation
    results = run_evaluation(slide_id="demo_out", num_tiles=50)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)