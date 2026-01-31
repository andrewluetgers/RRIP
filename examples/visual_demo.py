"""
Visual demonstration of RRIP compression quality
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from rrip import RRIPEncoder, RRIPDecoder
import os

print("Creating demonstration image with various features...")

# Create a test image with different features
width, height = 512, 512
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Draw some patterns
# Gradient background
for y in range(height):
    color = int(255 * (y / height))
    draw.line([(0, y), (width, y)], fill=(color, 50, 255-color))

# Draw some shapes
draw.ellipse([50, 50, 200, 200], fill=(255, 0, 0), outline=(0, 0, 0), width=3)
draw.rectangle([250, 50, 450, 200], fill=(0, 255, 0), outline=(0, 0, 0), width=3)
draw.polygon([(100, 300), (200, 250), (300, 300), (300, 450), (100, 450)], 
             fill=(0, 0, 255), outline=(0, 0, 0), width=3)

# Add some text
try:
    draw.text((150, 370), "RRIP Demo", fill=(255, 255, 255))
except:
    pass  # Font might not be available

# Save original
image.save('/tmp/demo_original.png')
original_size = os.path.getsize('/tmp/demo_original.png')
print(f"Original image saved: /tmp/demo_original.png ({original_size/1024:.1f} KB)")

# Test different quality levels
qualities = [30, 50, 70, 90]
results = []

print("\nTesting compression at different quality levels...")
print("-" * 70)

for quality in qualities:
    encoder = RRIPEncoder(downsample_factor=4, quality=quality, interpolation='bicubic')
    decoder = RRIPDecoder()
    
    # Encode
    encoded = encoder.encode_tile(image)
    
    # Decode
    reconstructed = decoder.decode_tile(encoded)
    
    # Save
    output_path = f'/tmp/demo_quality_{quality}.png'
    reconstructed.save(output_path)
    output_size = os.path.getsize(output_path)
    
    # Calculate metrics
    original_array = np.array(image)
    reconstructed_array = np.array(reconstructed)
    mse = np.mean((original_array.astype(float) - reconstructed_array.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
    
    # Estimate compression for encoded data
    import pickle
    encoded_file = f'/tmp/demo_encoded_{quality}.rrip'
    encoder.save_encoded({'tiles': [encoded], 'positions': [(0, 0)], 
                          'image_size': (width, height), 'tile_size': width}, encoded_file)
    compressed_size = os.path.getsize(encoded_file)
    ratio = original_size / compressed_size
    
    print(f"Quality {quality:2d}: PSNR={psnr:5.2f} dB, Ratio={ratio:5.2f}x, "
          f"Size={compressed_size/1024:5.1f} KB, Output={output_size/1024:5.1f} KB")
    
    results.append({
        'quality': quality,
        'psnr': psnr,
        'ratio': ratio,
        'compressed_size': compressed_size,
        'output_path': output_path
    })

print("-" * 70)
print("\nCompression demonstration complete!")
print("\nFiles created:")
print("  - /tmp/demo_original.png (original)")
for r in results:
    print(f"  - /tmp/demo_quality_{r['quality']}.png (quality {r['quality']})")
    print(f"  - /tmp/demo_encoded_{r['quality']}.rrip (compressed)")

print("\nVisual comparison:")
print("Compare the original with reconstructed images to see the quality trade-offs.")
print("Higher quality preserves more detail but with less compression.")
