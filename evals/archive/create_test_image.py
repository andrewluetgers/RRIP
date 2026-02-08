#!/usr/bin/env python3
"""Create a test 1024x1024 image for debugging ORIGAMI compression."""
import numpy as np
from PIL import Image, ImageDraw
import pathlib

def create_test_image(output_path, size=1024):
    """Create a test image with gradients and patterns."""
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)

    # Create some patterns
    # Gradient background
    for y in range(size):
        color = int((y / size) * 128)
        draw.rectangle([(0, y), (size, y+1)], fill=(color, color+64, color+32))

    # Add some geometric shapes for detail
    # Circles at different scales
    for i in range(5):
        x = (i + 1) * 170
        y = 200
        radius = 30 + i * 10
        draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                     fill=(255-i*40, i*40, 128), outline='black', width=2)

    # Add diagonal lines for texture
    for i in range(0, size, 50):
        draw.line([(i, 0), (0, i)], fill='darkblue', width=1)
        draw.line([(size-i, size), (size, size-i)], fill='darkred', width=1)

    # Add some text-like patterns
    for y in range(400, 600, 30):
        for x in range(100, 900, 100):
            draw.rectangle([x, y, x+60, y+20], fill='darkgreen', outline='black')

    # Add fine details in corners for testing compression quality
    # Top-left: fine checkerboard
    for y in range(0, 100, 4):
        for x in range(0, 100, 4):
            if (x//4 + y//4) % 2 == 0:
                draw.rectangle([x, y, x+4, y+4], fill='black')

    # Bottom-right: gradual gradient
    for y in range(924, 1024):
        for x in range(924, 1024):
            gray = int(((x-924) + (y-924)) * 1.28)
            draw.point((x, y), fill=(gray, gray, gray))

    img.save(output_path, quality=95)
    print(f"Created test image: {output_path}")

if __name__ == "__main__":
    output_path = pathlib.Path("../paper/L0-1024.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_test_image(output_path)