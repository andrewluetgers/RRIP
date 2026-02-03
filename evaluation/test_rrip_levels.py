#!/usr/bin/env python3
"""Test ORIGAMI with correct pyramid levels"""

import requests
import numpy as np
from PIL import Image
import io
from pathlib import Path

# Test L16 (highest res), L15, and L14 tiles
test_tiles = [
    (16, 42, 223),
    (16, 100, 67),
    (15, 50, 33),
    (15, 25, 100),
    (14, 25, 50),
    (14, 12, 25)
]

print("Testing ORIGAMI tile serving at different levels:\n")

for level, x, y in test_tiles:
    # Try to fetch from server
    tile_url = f"http://localhost:3007/tiles/demo_out/{level}/{x}_{y}.jpg"

    try:
        response = requests.get(tile_url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            arr = np.array(img)
            print(f"✓ L{level} tile ({x},{y}): {len(response.content)} bytes, shape {arr.shape}")
        else:
            print(f"✗ L{level} tile ({x},{y}): HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ L{level} tile ({x},{y}): {e}")

print("\nChecking residual structure:")
data_path = Path("/Users/andrewluetgers/projects/dev/ORIGAMI/data/demo_out")

# Check L14 as "L2" in ORIGAMI terminology (L16=L0, L15=L1, L14=L2)
baseline_l14 = data_path / "baseline_pyramid_files" / "14"
if baseline_l14.exists():
    tiles = list(baseline_l14.glob("*.jpg"))[:5]
    print(f"Found {len(list(baseline_l14.glob('*.jpg')))} L14 (L2 in ORIGAMI) baseline tiles")

# Check residuals - they should be organized by L14 coordinates
residuals = data_path / "residuals_q32"
if residuals.exists():
    l1_res = residuals / "L1"
    l0_res = residuals / "L0"

    if l1_res.exists():
        # Count L14 coordinate directories
        l14_dirs = list(l1_res.glob("*_*"))[:3]
        print(f"Found {len(list(l1_res.glob('*_*')))} L14 coordinate directories in L1 residuals")
        for d in l14_dirs:
            tiles = list(d.glob("*.jpg"))
            print(f"  {d.name}: {len(tiles)} L15 residual tiles")

    if l0_res.exists():
        l14_dirs = list(l0_res.glob("*_*"))[:3]
        print(f"Found {len(list(l0_res.glob('*_*')))} L14 coordinate directories in L0 residuals")
        for d in l14_dirs:
            tiles = list(d.glob("*.jpg"))
            print(f"  {d.name}: {len(tiles)} L16 residual tiles")