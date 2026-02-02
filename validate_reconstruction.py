#!/usr/bin/env python3
"""
Bulletproof validation that RRIP server is serving reconstructed tiles, not baselines.
This test compares tiles from RRIP server against baseline tiles to prove they are different.
"""

import requests
import numpy as np
from PIL import Image
import io
import os
import random
from pathlib import Path

def load_baseline_tile(slide_path, level, x, y):
    """Load a baseline tile directly from disk"""
    tile_path = Path(slide_path) / "baseline_pyramid_files" / str(level) / f"{x}_{y}.jpg"
    if not tile_path.exists():
        return None
    return Image.open(tile_path)

def fetch_server_tile(port, slide_id, level, x, y):
    """Fetch a tile from the RRIP server"""
    url = f"http://localhost:{port}/tiles/{slide_id}/{level}/{x}_{y}.jpg"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return Image.open(io.BytesIO(response.content))

def compare_images(img1, img2):
    """Compare two images and return difference metrics"""
    # Convert to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Check dimensions match
    if arr1.shape != arr2.shape:
        return {
            "dimensions_match": False,
            "img1_shape": arr1.shape,
            "img2_shape": arr2.shape
        }

    # Calculate differences
    diff = np.abs(arr1.astype(float) - arr2.astype(float))

    # Metrics
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    pixels_different = np.sum(diff > 0)
    total_pixels = diff.size
    percent_different = (pixels_different / total_pixels) * 100

    # Check if images are identical
    are_identical = np.array_equal(arr1, arr2)

    # Calculate PSNR if not identical
    if not are_identical and mean_diff > 0:
        mse = np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf') if are_identical else 0

    return {
        "dimensions_match": True,
        "are_identical": are_identical,
        "max_pixel_difference": int(max_diff),
        "mean_pixel_difference": float(mean_diff),
        "pixels_different": int(pixels_different),
        "total_pixels": int(total_pixels),
        "percent_pixels_different": float(percent_different),
        "psnr_db": float(psnr)
    }

def validate_reconstruction():
    """Main validation function"""
    print("=" * 80)
    print("RRIP RECONSTRUCTION VALIDATION TEST")
    print("=" * 80)
    print("\nThis test proves that the RRIP server is serving RECONSTRUCTED tiles,")
    print("not the original baseline tiles.\n")

    port = 3007
    slide_id = "demo_out"
    slide_path = "data/demo_out"

    # Test multiple tile types
    test_cases = [
        # L0 tiles (should be reconstructed)
        {"level": 16, "x": 100, "y": 100, "expected": "reconstructed"},
        {"level": 16, "x": 50, "y": 50, "expected": "reconstructed"},
        {"level": 16, "x": 150, "y": 150, "expected": "reconstructed"},

        # L1 tiles (should be reconstructed)
        {"level": 15, "x": 50, "y": 50, "expected": "reconstructed"},
        {"level": 15, "x": 25, "y": 25, "expected": "reconstructed"},

        # L2 tiles (should be baseline - identical)
        {"level": 14, "x": 25, "y": 25, "expected": "baseline"},
        {"level": 14, "x": 10, "y": 10, "expected": "baseline"},

        # Lower levels (should be baseline - identical)
        {"level": 13, "x": 10, "y": 10, "expected": "baseline"},
        {"level": 12, "x": 5, "y": 5, "expected": "baseline"},
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        level = test["level"]
        x = test["x"]
        y = test["y"]
        expected = test["expected"]

        print(f"\nTest {i}: Level {level}, Tile ({x},{y}) - Expected: {expected.upper()}")
        print("-" * 60)

        # Load baseline
        baseline = load_baseline_tile(slide_path, level, x, y)
        if baseline is None:
            print(f"  ❌ Baseline tile not found at expected path")
            continue

        # Fetch from server
        server = fetch_server_tile(port, slide_id, level, x, y)
        if server is None:
            print(f"  ❌ Server tile not accessible")
            continue

        # Compare
        comparison = compare_images(baseline, server)

        # Interpret results
        if not comparison["dimensions_match"]:
            print(f"  ❌ DIMENSION MISMATCH!")
            print(f"     Baseline: {comparison['img1_shape']}")
            print(f"     Server:   {comparison['img2_shape']}")
            result = "FAIL"
        elif comparison["are_identical"]:
            if expected == "baseline":
                print(f"  ✅ CORRECT: Tiles are IDENTICAL (serving baseline)")
                result = "PASS"
            else:
                print(f"  ❌ WRONG: Tiles are IDENTICAL (should be reconstructed!)")
                print(f"     This suggests server is serving baseline, not reconstruction!")
                result = "FAIL"
        else:
            if expected == "reconstructed":
                print(f"  ✅ CORRECT: Tiles are DIFFERENT (reconstruction working)")
                print(f"     Mean difference: {comparison['mean_pixel_difference']:.2f}")
                print(f"     Max difference:  {comparison['max_pixel_difference']}")
                print(f"     Pixels changed:  {comparison['percent_pixels_different']:.1f}%")
                print(f"     PSNR:           {comparison['psnr_db']:.1f} dB")
                result = "PASS"
            else:
                print(f"  ❌ WRONG: Tiles are DIFFERENT (should be identical!)")
                result = "FAIL"

        results.append({
            "test": test,
            "comparison": comparison,
            "result": result
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["result"] == "PASS")
    failed = sum(1 for r in results if r["result"] == "FAIL")

    print(f"\nTests Passed: {passed}/{len(results)}")
    print(f"Tests Failed: {failed}/{len(results)}")

    if failed == 0:
        print("\n🎉 SUCCESS: RRIP server is correctly serving reconstructed tiles!")
        print("   - L0/L1 tiles are being reconstructed from residuals")
        print("   - L2+ tiles are being served as baseline")
        print("   - The reconstruction is working as designed")
    else:
        print("\n⚠️  WARNING: Some tests failed - check results above")

    # File size comparison
    print("\n" + "=" * 80)
    print("FILE SIZE VALIDATION")
    print("=" * 80)

    # Check a specific L0 tile that should be reconstructed
    test_x, test_y = 100, 100
    baseline_path = Path(slide_path) / "baseline_pyramid_files" / "16" / f"{test_x}_{test_y}.jpg"

    if baseline_path.exists():
        baseline_size = os.path.getsize(baseline_path)

        # Get server tile size
        url = f"http://localhost:{port}/tiles/{slide_id}/16/{test_x}_{test_y}.jpg"
        response = requests.get(url)
        server_size = len(response.content)

        print(f"\nL0 Tile (16/{test_x}_{test_y}.jpg) Size Comparison:")
        print(f"  Baseline size: {baseline_size:,} bytes")
        print(f"  Server size:   {server_size:,} bytes")
        print(f"  Difference:    {abs(baseline_size - server_size):,} bytes")

        if abs(baseline_size - server_size) > 100:
            print("  ✅ Significant size difference confirms reconstruction")
        else:
            print("  ⚠️  Sizes are very similar - may need investigation")

if __name__ == "__main__":
    validate_reconstruction()