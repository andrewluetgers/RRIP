"""
Example: Tile-based compression and serving
"""

from PIL import Image
import numpy as np
from rrip import TileManager

# Create a large sample image
print("Creating large sample image...")
width, height = 1024, 1024
image_array = np.zeros((height, width, 3), dtype=np.uint8)

# Create a more complex pattern
for y in range(height):
    for x in range(width):
        r = int(255 * np.sin(x / 50) ** 2)
        g = int(255 * np.sin(y / 50) ** 2)
        b = int(255 * np.sin((x + y) / 70) ** 2)
        image_array[y, x] = [r, g, b]

large_image = Image.fromarray(image_array)
large_image.save('/tmp/large_sample.png')
print(f"Large sample image saved to /tmp/large_sample.png")

# Store the image using TileManager
print("\nStoring image with TileManager...")
manager = TileManager('/tmp/rrip_storage')

config = {
    'downsample_factor': 4,
    'quality': 60,
    'interpolation': 'bicubic',
    'tile_size': 256
}

metadata = manager.store_image('large_image', large_image, encoder_config=config)

print(f"Image stored successfully:")
print(f"  Image ID: {metadata['image_id']}")
print(f"  Size: {metadata['image_size']}")
print(f"  Number of tiles: {metadata['num_tiles']}")
print(f"  Tile size: {metadata['tile_size']}")

# Retrieve individual tiles
print("\nRetrieving individual tiles...")
for i in range(min(3, metadata['num_tiles'])):
    tile = manager.get_tile('large_image', i)
    tile.save(f'/tmp/tile_{i}.png')
    print(f"  Tile {i} saved to /tmp/tile_{i}.png")

# Retrieve tile by position
print("\nRetrieving tile at position (300, 300)...")
tile_at_pos = manager.get_tile_by_position('large_image', 300, 300)
tile_at_pos.save('/tmp/tile_at_position.png')
print(f"  Tile saved to /tmp/tile_at_position.png")

# Reconstruct full image
print("\nReconstructing full image...")
reconstructed = manager.get_full_image('large_image')
reconstructed.save('/tmp/large_reconstructed.png')
print(f"  Reconstructed image saved to /tmp/large_reconstructed.png")

# List all stored images
print("\nListing all stored images...")
images = manager.list_images()
for img_id in images:
    info = manager.get_image_info(img_id)
    print(f"  {img_id}: {info['image_size']}, {info['num_tiles']} tiles")

print("\nExample completed!")
print("\nTo start the tile server, run:")
print("  rrip serve /tmp/rrip_storage")
print("\nThen access tiles via HTTP:")
print("  http://localhost:5000/images")
print("  http://localhost:5000/images/large_image")
print("  http://localhost:5000/images/large_image/tile/0")
print("  http://localhost:5000/images/large_image/full")
