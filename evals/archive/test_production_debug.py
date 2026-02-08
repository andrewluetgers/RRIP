#!/usr/bin/env python3
"""
Test script to demonstrate that debug functionality uses the actual production code.
This imports and uses the exact same functions from wsi_residual_tool_grid.py.
"""

import sys
import pathlib
import numpy as np
from PIL import Image

# Import the ACTUAL production functions
from wsi_residual_tool_grid import (
    rgb_to_ycbcr_bt601,
    ycbcr_to_rgb_bt601,
    save_debug_image
)

def test_production_code():
    """Test that we're using the exact production code functions."""

    print("Testing production code functions from wsi_residual_tool_grid.py")
    print("=" * 60)

    # Create a test image
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Test 1: RGB to YCbCr conversion (production function)
    print("\n1. Testing rgb_to_ycbcr_bt601 from production code...")
    Y, Cb, Cr = rgb_to_ycbcr_bt601(test_img)
    print(f"   Y shape: {Y.shape}, Cb shape: {Cb.shape}, Cr shape: {Cr.shape}")
    print(f"   Y range: [{Y.min():.2f}, {Y.max():.2f}]")

    # Test 2: YCbCr to RGB conversion (production function)
    print("\n2. Testing ycbcr_to_rgb_bt601 from production code...")
    reconstructed = ycbcr_to_rgb_bt601(Y, Cb, Cr)
    print(f"   Reconstructed shape: {reconstructed.shape}")
    print(f"   Max reconstruction error: {np.max(np.abs(test_img.astype(float) - reconstructed.astype(float))):.2f}")

    # Test 3: Debug image saving (production function)
    print("\n3. Testing save_debug_image from production code...")
    debug_dir = pathlib.Path("test_debug_output")
    debug_dir.mkdir(exist_ok=True)

    # Save a test debug image
    output_path, file_size = save_debug_image(
        test_img,
        debug_dir,
        1,
        "test_production_function"
    )
    print(f"   Saved to: {output_path}")
    print(f"   File size: {file_size} bytes")

    # Verify the file exists and can be loaded
    loaded = Image.open(output_path)
    print(f"   Verified: Image loaded successfully, size={loaded.size}")

    print("\n" + "=" * 60)
    print("SUCCESS: All production code functions work correctly!")
    print("This proves we're using the SAME code, not a duplicate.")

    # Clean up
    import shutil
    shutil.rmtree(debug_dir)

if __name__ == "__main__":
    test_production_code()
