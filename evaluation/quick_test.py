#!/usr/bin/env python3
"""Quick test to verify the evaluation setup works"""

import requests
import numpy as np
from PIL import Image
import io
from pathlib import Path

# Test server connection
try:
    response = requests.get("http://localhost:3007/healthz")
    print(f"✓ Server health check: {response.status_code}")
except:
    print("✗ Server not responding on port 3007")
    exit(1)

# Test fetching a tile
try:
    # Try to get a L0 tile
    tile_url = "http://localhost:3007/tiles/demo_out/0/10_10.jpg"
    response = requests.get(tile_url)
    if response.status_code == 200:
        print(f"✓ Successfully fetched L0 tile: {len(response.content)} bytes")

        # Convert to image
        img = Image.open(io.BytesIO(response.content))
        arr = np.array(img)
        print(f"✓ Tile dimensions: {arr.shape}")
    else:
        print(f"✗ Failed to fetch tile: HTTP {response.status_code}")
except Exception as e:
    print(f"✗ Error fetching tile: {e}")

# Check for original tiles
data_path = Path("/Users/andrewluetgers/projects/dev/RRIP/data/demo_out")
baseline_path = data_path / "baseline_pyramid_files" / "0"
if baseline_path.exists():
    tiles = list(baseline_path.glob("*.jpg"))
    print(f"✓ Found {len(tiles)} baseline L0 tiles")

    if tiles:
        # Load first tile
        first_tile = np.array(Image.open(tiles[0]))
        print(f"✓ Baseline tile shape: {first_tile.shape}")
else:
    print("✗ No baseline tiles found")

# Check for residual packs
pack_path = data_path / "residual_packs"
if pack_path.exists():
    packs = list(pack_path.glob("*.pack"))
    print(f"✓ Found {len(packs)} residual packs")
else:
    print("✗ No residual packs found")

print("\n✅ Setup looks good! Ready to run full evaluation.")