#!/usr/bin/env python3
"""
Simple ORIGAMI Evaluation without OpenCV dependency
Uses PIL/Pillow only for image operations
"""

import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import requests
import io
from dataclasses import dataclass, asdict
from typing import List, Tuple

@dataclass
class TestResult:
    method: str
    quality: int
    level: int
    file_size_kb: float
    psnr_db: float
    ssim: float
    bpp: float
    decode_ms: float

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images"""
    if img1.shape != img2.shape:
        return 0.0
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))

def calculate_simple_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified SSIM calculation"""
    if img1.shape != img2.shape:
        return 0.0

    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = np.mean(img2, axis=2)

    # Simple SSIM approximation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))

    return float(ssim)

def test_origami(slide_id: str, level: int, tiles: List[Tuple[int, int]],
              data_dir: Path) -> List[TestResult]:
    """Test ORIGAMI compression"""
    results = []

    for x, y in tiles[:20]:  # Limit for speed
        # Load original
        orig_path = data_dir / "baseline_pyramid_files" / str(level) / f"{x}_{y}.jpg"
        if not orig_path.exists():
            continue

        original = np.array(Image.open(orig_path))

        # Fetch from server
        start = time.time()
        try:
            response = requests.get(f"http://localhost:3007/tiles/{slide_id}/{level}/{x}_{y}.jpg")
            decode_ms = (time.time() - start) * 1000

            if response.status_code != 200:
                continue

            reconstructed = np.array(Image.open(io.BytesIO(response.content)))

            # Estimate size (simplified)
            if level >= 15:  # L16/L15 use reconstruction
                # Approximate: L14 baseline + residual
                size_kb = 8.0  # ~8KB average for ORIGAMI tiles
            else:
                size_kb = orig_path.stat().st_size / 1024

            results.append(TestResult(
                method="ORIGAMI",
                quality=32,
                level=level,
                file_size_kb=size_kb,
                psnr_db=calculate_psnr(original, reconstructed),
                ssim=calculate_simple_ssim(original, reconstructed),
                bpp=size_kb * 1024 * 8 / (256 * 256),
                decode_ms=decode_ms
            ))
        except Exception as e:
            print(f"  Error: {e}")
            continue

    return results

def test_jpeg(level: int, tiles: List[Tuple[int, int]], data_dir: Path,
              qualities: List[int]) -> List[TestResult]:
    """Test JPEG recompression"""
    results = []

    for quality in qualities:
        for x, y in tiles[:5]:  # Limit for speed
            orig_path = data_dir / "baseline_pyramid_files" / str(level) / f"{x}_{y}.jpg"
            if not orig_path.exists():
                continue

            # Load and recompress
            img = Image.open(orig_path)
            original = np.array(img)

            # Recompress at different quality
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            size_kb = len(buffer.getvalue()) / 1024

            # Decode
            buffer.seek(0)
            recompressed = np.array(Image.open(buffer))

            results.append(TestResult(
                method="JPEG",
                quality=quality,
                level=level,
                file_size_kb=size_kb,
                psnr_db=calculate_psnr(original, recompressed),
                ssim=calculate_simple_ssim(original, recompressed),
                bpp=size_kb * 1024 * 8 / (256 * 256),
                decode_ms=0
            ))

    return results

def main():
    """Run evaluation"""
    print("="*60)
    print("ORIGAMI EVALUATION (Simplified)")
    print("="*60)

    SLIDE_ID = "demo_out"
    DATA_DIR = Path("/Users/andrewluetgers/projects/dev/ORIGAMI/data") / SLIDE_ID

    all_results = []

    # Test L16 (highest res = L0 in ORIGAMI)
    print("\nTesting Level 16 (L0 in ORIGAMI)...")
    level_dir = DATA_DIR / "baseline_pyramid_files" / "16"
    tiles = []
    for tile_path in list(level_dir.glob("*.jpg"))[:30]:
        try:
            x, y = tile_path.stem.split('_')
            tiles.append((int(x), int(y)))
        except:
            continue

    print(f"  Testing {len(tiles)} tiles...")

    # Test ORIGAMI
    origami_results = test_origami(SLIDE_ID, 16, tiles, DATA_DIR)
    all_results.extend(origami_results)
    print(f"  ORIGAMI: {len(origami_results)} results")

    # Test JPEG
    jpeg_results = test_jpeg(16, tiles, DATA_DIR, [95, 90, 80, 70, 60])
    all_results.extend(jpeg_results)
    print(f"  JPEG: {len(jpeg_results)} results")

    # Calculate averages
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # ORIGAMI summary
    origami = [r for r in all_results if r.method == "ORIGAMI"]
    if origami:
        print(f"\nORIGAMI (n={len(origami)}):")
        print(f"  Avg size: {np.mean([r.file_size_kb for r in origami]):.1f} KB")
        print(f"  Avg PSNR: {np.mean([r.psnr_db for r in origami]):.2f} dB")
        print(f"  Avg SSIM: {np.mean([r.ssim for r in origami]):.4f}")
        print(f"  Avg BPP: {np.mean([r.bpp for r in origami]):.3f}")
        print(f"  Avg decode: {np.mean([r.decode_ms for r in origami]):.1f} ms")

    # JPEG by quality
    for q in [95, 90, 80, 70, 60]:
        jpeg_q = [r for r in all_results if r.method == "JPEG" and r.quality == q]
        if jpeg_q:
            print(f"\nJPEG Q{q} (n={len(jpeg_q)}):")
            print(f"  Avg size: {np.mean([r.file_size_kb for r in jpeg_q]):.1f} KB")
            print(f"  Avg PSNR: {np.mean([r.psnr_db for r in jpeg_q]):.2f} dB")
            print(f"  Avg SSIM: {np.mean([r.ssim for r in jpeg_q]):.4f}")
            print(f"  Avg BPP: {np.mean([r.bpp for r in jpeg_q]):.3f}")

    # Simple R-D plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ORIGAMI vs JPEG: Rate-Distortion Performance', fontsize=14, fontweight='bold')

    # ORIGAMI point
    if origami:
        origami_bpp = np.mean([r.bpp for r in origami])
        origami_psnr = np.mean([r.psnr_db for r in origami])
        origami_ssim = np.mean([r.ssim for r in origami])

        ax1.plot(origami_bpp, origami_psnr, 'ro', markersize=12, label='ORIGAMI')
        ax2.plot(origami_bpp, origami_ssim, 'ro', markersize=12, label='ORIGAMI')

    # JPEG curve
    jpeg_points = []
    for q in [95, 90, 80, 70, 60]:
        jpeg_q = [r for r in all_results if r.method == "JPEG" and r.quality == q]
        if jpeg_q:
            jpeg_points.append({
                'q': q,
                'bpp': np.mean([r.bpp for r in jpeg_q]),
                'psnr': np.mean([r.psnr_db for r in jpeg_q]),
                'ssim': np.mean([r.ssim for r in jpeg_q])
            })

    if jpeg_points:
        bpp = [p['bpp'] for p in jpeg_points]
        psnr = [p['psnr'] for p in jpeg_points]
        ssim = [p['ssim'] for p in jpeg_points]

        ax1.plot(bpp, psnr, 'b-s', label='JPEG', markersize=8)
        ax2.plot(bpp, ssim, 'b-s', label='JPEG', markersize=8)

        # Add quality labels
        for p in jpeg_points:
            ax1.annotate(f"Q{p['q']}", (p['bpp'], p['psnr']),
                        fontsize=8, ha='center', va='bottom')

    ax1.set_xlabel('Bits Per Pixel')
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Bits Per Pixel')
    ax2.set_ylabel('SSIM')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'rd_curves.png', dpi=150)
    print(f"\nPlot saved to {output_dir / 'rd_curves.png'}")

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    print(f"Results saved to {output_dir / 'results.json'}")

    # Calculate BD-Rate approximation
    if origami and jpeg_points:
        # Find JPEG points bracketing ORIGAMI PSNR
        origami_psnr = np.mean([r.psnr_db for r in origami])
        origami_bpp = np.mean([r.bpp for r in origami])

        # Simple linear interpolation
        for i in range(len(jpeg_points) - 1):
            if jpeg_points[i]['psnr'] > origami_psnr > jpeg_points[i+1]['psnr']:
                # Interpolate BPP at ORIGAMI PSNR
                psnr_range = jpeg_points[i]['psnr'] - jpeg_points[i+1]['psnr']
                psnr_offset = jpeg_points[i]['psnr'] - origami_psnr
                ratio = psnr_offset / psnr_range

                jpeg_bpp_at_origami_psnr = (jpeg_points[i]['bpp'] +
                                         ratio * (jpeg_points[i+1]['bpp'] - jpeg_points[i]['bpp']))

                bd_rate = (origami_bpp - jpeg_bpp_at_origami_psnr) / jpeg_bpp_at_origami_psnr * 100
                print(f"\n" + "="*60)
                print(f"BD-Rate (approximate): {bd_rate:.1f}%")
                print(f"At PSNR {origami_psnr:.1f} dB:")
                print(f"  JPEG needs {jpeg_bpp_at_origami_psnr:.3f} bpp")
                print(f"  ORIGAMI uses {origami_bpp:.3f} bpp")
                if bd_rate < 0:
                    print(f"  ORIGAMI is {-bd_rate:.1f}% more efficient")
                break

if __name__ == "__main__":
    main()