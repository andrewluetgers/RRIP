"""
Example: Basic RRIP compression and decompression
"""

from PIL import Image
import numpy as np
from rrip import RRIPEncoder, RRIPDecoder

# Create a sample image
print("Creating sample image...")
width, height = 512, 512
image_array = np.zeros((height, width, 3), dtype=np.uint8)

# Create a gradient pattern
for y in range(height):
    for x in range(width):
        image_array[y, x] = [x % 256, y % 256, (x + y) % 256]

sample_image = Image.fromarray(image_array)
sample_image.save('/tmp/sample_original.png')
print(f"Sample image saved to /tmp/sample_original.png")

# Compress the image
print("\nCompressing image...")
encoder = RRIPEncoder(
    downsample_factor=4,  # Downsample by 4x for priors
    quality=70,           # Quality 0-100
    interpolation='bicubic'
)

encoded_data = encoder.encode_tile(sample_image)
print(f"Image encoded successfully")
print(f"Prior shape: {encoded_data['prior'].shape}")
print(f"Original shape: {encoded_data['shape']}")

# Decompress the image
print("\nDecompressing image...")
decoder = RRIPDecoder()
reconstructed = decoder.decode_tile(encoded_data)
reconstructed.save('/tmp/sample_reconstructed.png')
print(f"Reconstructed image saved to /tmp/sample_reconstructed.png")

# Compare quality
original_array = np.array(sample_image)
reconstructed_array = np.array(reconstructed)
mse = np.mean((original_array.astype(float) - reconstructed_array.astype(float)) ** 2)
psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

print(f"\nQuality Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Peak Signal-to-Noise Ratio: {psnr:.2f} dB")

# Test with different quality settings
print("\n" + "=" * 50)
print("Testing different quality settings:")
print("=" * 50)

for quality in [30, 50, 70, 90]:
    encoder = RRIPEncoder(downsample_factor=4, quality=quality, interpolation='bicubic')
    encoded = encoder.encode_tile(sample_image)
    decoded = decoder.decode_tile(encoded)
    
    decoded_array = np.array(decoded)
    mse = np.mean((original_array.astype(float) - decoded_array.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
    
    # Estimate compression ratio (rough)
    import pickle
    compressed_size = len(pickle.dumps(encoded['compressed_residuals']))
    original_size = original_array.nbytes
    ratio = original_size / compressed_size
    
    print(f"\nQuality {quality}:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Approx compression ratio: {ratio:.2f}x")
