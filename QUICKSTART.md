# RRIP Quick Start Guide

Get started with RRIP in 5 minutes!

## Installation

```bash
git clone https://github.com/andrewluetgers/RRIP.git
cd RRIP
pip install -r requirements.txt
pip install -e .
```

## Quick Demo

### 1. Create a Test Image

```bash
python examples/basic_example.py
```

This creates sample images and tests compression at different quality levels.

### 2. Compress an Image

```bash
# Compress with quality 70 (good balance)
rrip compress examples/sample.jpg compressed.rrip --quality 70

# Or try different quality levels
rrip compress examples/sample.jpg low_quality.rrip --quality 30
rrip compress examples/sample.jpg high_quality.rrip --quality 90
```

### 3. Decompress

```bash
rrip decompress compressed.rrip output.png
```

### 4. Store and Serve Tiles

```bash
# Store a large image as tiles
rrip store my_large_image.jpg ./storage my_image --quality 60 --tile-size 256

# List stored images
rrip list-images ./storage

# Retrieve full image
rrip retrieve ./storage my_image output.png

# Start tile server
rrip serve ./storage --port 5000
```

### 5. Access via HTTP

With the server running, you can access:

```bash
# List all images
curl http://localhost:5000/images

# Get image info
curl http://localhost:5000/images/my_image

# Get a specific tile (as PNG)
curl http://localhost:5000/images/my_image/tile/0 > tile.png

# Get tile at position
curl "http://localhost:5000/images/my_image/tile_at?x=500&y=300" > tile_at_pos.png

# Get full image
curl http://localhost:5000/images/my_image/full > full_image.png
```

## Python API Quick Start

```python
from PIL import Image
from rrip import RRIPEncoder, RRIPDecoder, TileManager

# === Basic Compression ===
image = Image.open('input.jpg')

# Create encoder
encoder = RRIPEncoder(
    downsample_factor=4,    # 4x downsampling for priors
    quality=70,             # Quality 0-100
    interpolation='bicubic' # bicubic, bilinear, or nearest
)

# Encode
encoded = encoder.encode_tile(image)

# Decode
decoder = RRIPDecoder()
reconstructed = decoder.decode_tile(encoded)

# Save
reconstructed.save('output.png')

# === Tile-based Storage ===
manager = TileManager('./storage')

# Store image
config = {
    'downsample_factor': 4,
    'quality': 60,
    'tile_size': 256
}
metadata = manager.store_image('my_image', image, encoder_config=config)

# Retrieve specific tile
tile = manager.get_tile('my_image', tile_index=0)

# Retrieve tile by position
tile = manager.get_tile_by_position('my_image', x=500, y=300)

# Retrieve full image
full_image = manager.get_full_image('my_image')

# List all images
images = manager.list_images()
```

## Quality Guidelines

Choose quality based on your needs:

| Quality | Use Case | Compression | PSNR |
|---------|----------|-------------|------|
| 20-40 | Thumbnails, previews | 30-50x | 35-40 dB |
| 50-60 | General use, web serving | 15-25x | 38-42 dB |
| 70-80 | High quality images | 10-15x | 42-46 dB |
| 90+ | Near-lossless | 5-10x | 46-50 dB |

## Parameter Guide

### downsample_factor
- **2**: Minimal compression, very high quality
- **4**: Good balance (recommended)
- **8**: High compression, lower quality

### interpolation
- **bicubic**: Best quality (recommended)
- **bilinear**: Faster, good quality
- **nearest**: Fastest, sharp edges

### tile_size
- **128**: Small tiles, more overhead
- **256**: Good balance (recommended)
- **512**: Large tiles, less overhead

## Benchmarking

Compare settings:

```bash
rrip benchmark input.jpg ./output --quality 70 --downsample-factor 4
```

This shows:
- Compression ratio
- Encoding/decoding time
- Quality metrics (PSNR)
- File sizes

## Common Workflows

### Workflow 1: Simple File Compression
```bash
rrip compress input.jpg output.rrip --quality 70
rrip decompress output.rrip reconstructed.png
```

### Workflow 2: Tile Server for Web App
```bash
# Store images
rrip store image1.jpg ./storage img1 --quality 60
rrip store image2.jpg ./storage img2 --quality 60

# Start server
rrip serve ./storage --port 5000

# Access from web app via HTTP API
```

### Workflow 3: Batch Processing
```python
import os
from pathlib import Path
from PIL import Image
from rrip import RRIPEncoder, RRIPDecoder

encoder = RRIPEncoder(quality=70)
decoder = RRIPDecoder()

# Compress all images in directory
for img_file in Path('input_images').glob('*.jpg'):
    image = Image.open(img_file)
    encoded = encoder.encode_tile(image)
    
    output_file = f'compressed/{img_file.stem}.rrip'
    encoder.save_encoded(encoded, output_file)
    print(f'Compressed {img_file} -> {output_file}')
```

## Tips

1. **Start with quality 70**: Good balance for most use cases
2. **Use tile_size=256**: Works well for most images
3. **Bicubic interpolation**: Best quality in most cases
4. **Benchmark first**: Test different settings on your data
5. **Consider content**: Photos compress better than diagrams/text

## Troubleshooting

### Server won't start
- Check if port is available: `lsof -i :5000`
- Try a different port: `rrip serve ./storage --port 8080`

### Poor compression ratio
- Increase downsample_factor
- Decrease quality
- Check if input is already compressed

### Low quality output
- Increase quality parameter
- Decrease downsample_factor
- Try bicubic interpolation

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/](examples/) for more usage patterns
- See [IMPLEMENTATION.md](IMPLEMENTATION.md) for technical details
- Run the tests: `python -m unittest discover tests`

## Getting Help

If you encounter issues:
1. Check the examples in `examples/`
2. Run tests to verify installation: `python -m unittest discover tests`
3. Check the API documentation in README.md
4. File an issue on GitHub

## License

MIT License - see LICENSE file for details
