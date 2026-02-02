#!/usr/bin/env python3
"""
ORIGAMI Compression Evaluation Framework
Compares ORIGAMI against JPEG and JPEG 2000 using JPEG source images

Usage: python compare_compression.py --input-dir /path/to/jpeg/tiles --output-dir results/
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
import time

# For JPEG 2000 support
try:
    import glymur
    HAS_JPEG2000 = True
except ImportError:
    print("Warning: glymur not installed, JPEG 2000 comparison disabled")
    print("Install with: pip install glymur")
    HAS_JPEG2000 = False

@dataclass
class CompressionResult:
    """Results from a single compression test"""
    method: str
    quality: int
    file_size: int
    encode_time: float
    decode_time: float
    psnr: float
    ssim: float
    ms_ssim: float

class QualityMetrics:
    """Calculate image quality metrics"""

    @staticmethod
    def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    @staticmethod
    def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        if len(compressed.shape) == 3:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)

        # Use OpenCV's SSIM implementation
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
    def calculate_ms_ssim(original: np.ndarray, compressed: np.ndarray,
                         weights: List[float] = None) -> float:
        """Calculate Multi-Scale SSIM"""
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        levels = len(weights)
        ms_ssim = []

        for i in range(levels):
            ssim_val = QualityMetrics.calculate_ssim(original, compressed)
            ms_ssim.append(ssim_val)

            # Downsample for next level
            if i < levels - 1:
                original = cv2.pyrDown(original)
                compressed = cv2.pyrDown(compressed)

        # Combine with weights
        ms_ssim_val = 1.0
        for i, weight in enumerate(weights[:len(ms_ssim)]):
            ms_ssim_val *= ms_ssim[i] ** weight

        return ms_ssim_val

class CompressionTester:
    """Test different compression methods"""

    def __init__(self, reference_image: np.ndarray):
        self.reference = reference_image
        self.height, self.width = reference_image.shape[:2]

    def compress_jpeg(self, quality: int) -> Tuple[bytes, float, float]:
        """Compress using standard JPEG"""
        # Encode
        start = time.time()
        img_pil = Image.fromarray(self.reference)
        import io
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality, optimize=True)
        compressed_bytes = buffer.getvalue()
        encode_time = time.time() - start

        # Decode
        start = time.time()
        buffer.seek(0)
        decoded = np.array(Image.open(buffer))
        decode_time = time.time() - start

        return compressed_bytes, encode_time, decode_time, decoded

    def compress_jpeg2000(self, rate: float) -> Tuple[bytes, float, float]:
        """Compress using JPEG 2000"""
        if not HAS_JPEG2000:
            return None, 0, 0, self.reference

        # Save to temp file (glymur requires file)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Encode
            start = time.time()
            jp2 = glymur.Jp2k(tmp_path)
            jp2[:] = self.reference
            jp2.layer = [rate]  # Compression rate
            encode_time = time.time() - start

            # Get compressed size
            compressed_size = os.path.getsize(tmp_path)
            with open(tmp_path, 'rb') as f:
                compressed_bytes = f.read()

            # Decode
            start = time.time()
            decoded = jp2[:]
            decode_time = time.time() - start

            return compressed_bytes, encode_time, decode_time, decoded

        finally:
            os.unlink(tmp_path)

    def simulate_origami(self, residual_quality: int = 30) -> Tuple[int, float, float]:
        """
        Simulate ORIGAMI compression
        This is a simplified version - replace with actual ORIGAMI implementation
        """
        # For now, we'll simulate ORIGAMI by:
        # 1. Downsampling by 4x (to simulate L2)
        # 2. Upsampling back
        # 3. Computing residual
        # 4. Compressing residual as grayscale JPEG

        start_total = time.time()

        # Simulate L2 (downsample by 4x)
        l2_size = (self.width // 4, self.height // 4)
        l2 = cv2.resize(self.reference, l2_size, interpolation=cv2.INTER_AREA)

        # Simulate L2 compression
        l2_pil = Image.fromarray(l2)
        import io
        l2_buffer = io.BytesIO()
        l2_pil.save(l2_buffer, format='JPEG', quality=90)
        l2_size_bytes = len(l2_buffer.getvalue())

        # Upsample to create prediction
        prediction = cv2.resize(l2, (self.width, self.height),
                               interpolation=cv2.INTER_LINEAR)

        # Compute residual (luma only)
        reference_gray = cv2.cvtColor(self.reference, cv2.COLOR_RGB2GRAY)
        prediction_gray = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)

        # Residual with bias (to make it unsigned)
        residual = (reference_gray.astype(int) - prediction_gray.astype(int) + 128)
        residual = np.clip(residual, 0, 255).astype(np.uint8)

        # Compress residual as JPEG
        residual_pil = Image.fromarray(residual, mode='L')
        residual_buffer = io.BytesIO()
        residual_pil.save(residual_buffer, format='JPEG', quality=residual_quality)
        residual_size_bytes = len(residual_buffer.getvalue())

        # Total size (L2 + residual)
        total_size = l2_size_bytes + residual_size_bytes

        encode_time = time.time() - start_total

        # Simulate decode
        start = time.time()

        # Decode residual
        residual_buffer.seek(0)
        decoded_residual = np.array(Image.open(residual_buffer))

        # Reconstruct
        reconstructed_gray = np.clip(
            prediction_gray.astype(int) + decoded_residual.astype(int) - 128,
            0, 255
        ).astype(np.uint8)

        # Combine with chroma from prediction
        reconstructed = prediction.copy()
        reconstructed_ycrcb = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2YCrCb)
        reconstructed_ycrcb[:, :, 0] = reconstructed_gray
        reconstructed = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2RGB)

        decode_time = time.time() - start

        return total_size, encode_time, decode_time, reconstructed

def run_compression_comparison(image_path: str, output_dir: str):
    """Run full compression comparison on a single image"""

    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    tester = CompressionTester(image)
    metrics = QualityMetrics()

    results = []

    # Test JPEG at different qualities
    print("\nTesting JPEG compression...")
    for quality in [95, 90, 85, 80, 70, 60, 50]:
        compressed, enc_time, dec_time, decoded = tester.compress_jpeg(quality)

        result = CompressionResult(
            method='JPEG',
            quality=quality,
            file_size=len(compressed),
            encode_time=enc_time,
            decode_time=dec_time,
            psnr=metrics.calculate_psnr(image, decoded),
            ssim=metrics.calculate_ssim(image, decoded),
            ms_ssim=metrics.calculate_ms_ssim(image, decoded)
        )
        results.append(result)
        print(f"  Q{quality}: {result.file_size} bytes, "
              f"PSNR={result.psnr:.2f}dB, MS-SSIM={result.ms_ssim:.4f}")

    # Test JPEG 2000 if available
    if HAS_JPEG2000:
        print("\nTesting JPEG 2000 compression...")
        for rate in [0.5, 0.3, 0.2, 0.1, 0.05, 0.03]:
            compressed, enc_time, dec_time, decoded = tester.compress_jpeg2000(rate)
            if compressed:
                result = CompressionResult(
                    method='JPEG2000',
                    quality=int(rate * 100),
                    file_size=len(compressed),
                    encode_time=enc_time,
                    decode_time=dec_time,
                    psnr=metrics.calculate_psnr(image, decoded),
                    ssim=metrics.calculate_ssim(image, decoded),
                    ms_ssim=metrics.calculate_ms_ssim(image, decoded)
                )
                results.append(result)
                print(f"  Rate {rate}: {result.file_size} bytes, "
                      f"PSNR={result.psnr:.2f}dB, MS-SSIM={result.ms_ssim:.4f}")

    # Test ORIGAMI simulation
    print("\nTesting ORIGAMI compression (simulated)...")
    for residual_q in [50, 40, 30, 20, 10]:
        size, enc_time, dec_time, decoded = tester.simulate_origami(residual_q)

        result = CompressionResult(
            method='ORIGAMI',
            quality=residual_q,
            file_size=size,
            encode_time=enc_time,
            decode_time=dec_time,
            psnr=metrics.calculate_psnr(image, decoded),
            ssim=metrics.calculate_ssim(image, decoded),
            ms_ssim=metrics.calculate_ms_ssim(image, decoded)
        )
        results.append(result)
        print(f"  ResQ{residual_q}: {result.file_size} bytes, "
              f"PSNR={result.psnr:.2f}dB, MS-SSIM={result.ms_ssim:.4f}")

    return results

def plot_rd_curves(results: List[CompressionResult], output_path: str):
    """Generate Rate-Distortion curves"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Group by method
    methods = {}
    for r in results:
        if r.method not in methods:
            methods[r.method] = []
        methods[r.method].append(r)

    # Plot PSNR vs File Size
    ax = axes[0]
    for method, method_results in methods.items():
        sizes = [r.file_size / 1024 for r in method_results]  # KB
        psnrs = [r.psnr for r in method_results]
        ax.plot(sizes, psnrs, 'o-', label=method, markersize=8)

    ax.set_xlabel('File Size (KB)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Rate-Distortion: PSNR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot MS-SSIM vs File Size
    ax = axes[1]
    for method, method_results in methods.items():
        sizes = [r.file_size / 1024 for r in method_results]
        ms_ssims = [r.ms_ssim for r in method_results]
        ax.plot(sizes, ms_ssims, 'o-', label=method, markersize=8)

    ax.set_xlabel('File Size (KB)')
    ax.set_ylabel('MS-SSIM')
    ax.set_title('Rate-Distortion: MS-SSIM')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nR-D curves saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare compression methods for ORIGAMI evaluation')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory of images')
    parser.add_argument('--output-dir', type=str, default='compression_results',
                       help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    if os.path.isfile(args.input):
        images = [args.input]
    else:
        # Find all JPEG images in directory
        images = list(Path(args.input).glob('*.jpg')) + \
                list(Path(args.input).glob('*.jpeg'))

    all_results = []

    for img_path in images:
        print(f"\nProcessing: {img_path}")
        results = run_compression_comparison(str(img_path), str(output_dir))
        all_results.extend(results)

        # Save individual results
        img_name = Path(img_path).stem
        with open(output_dir / f'{img_name}_results.json', 'w') as f:
            json.dump([r.__dict__ for r in results], f, indent=2)

    # Generate combined R-D curves
    if all_results:
        plot_rd_curves(all_results, str(output_dir / 'rd_curves.png'))

        # Generate summary table
        print("\n" + "="*80)
        print("COMPRESSION COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Method':<12} {'Quality':<10} {'Size (KB)':<12} "
              f"{'PSNR (dB)':<12} {'MS-SSIM':<10}")
        print("-"*80)

        for r in sorted(all_results, key=lambda x: (x.method, -x.quality)):
            print(f"{r.method:<12} {r.quality:<10} {r.file_size/1024:<12.1f} "
                  f"{r.psnr:<12.2f} {r.ms_ssim:<10.4f}")

if __name__ == '__main__':
    main()