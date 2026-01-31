"""
Command-line interface for RRIP
"""

import click
from PIL import Image
from pathlib import Path
from .encoder import RRIPEncoder
from .decoder import RRIPDecoder
from .tile_manager import TileManager
from .server import run_server


@click.group()
def main():
    """RRIP - Residual Reconstruction from Interpolated Priors"""
    pass


@main.command()
@click.argument('input_image', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--downsample-factor', '-d', default=4, type=int,
              help='Downsampling factor for priors (default: 4)')
@click.option('--quality', '-q', default=50, type=int,
              help='Quality level 0-100 (default: 50)')
@click.option('--interpolation', '-i', default='bicubic',
              type=click.Choice(['bilinear', 'bicubic', 'nearest']),
              help='Interpolation method (default: bicubic)')
@click.option('--tile-size', '-t', default=256, type=int,
              help='Tile size (default: 256)')
def compress(input_image, output_file, downsample_factor, quality, interpolation, tile_size):
    """Compress an image using RRIP."""
    click.echo(f"Compressing {input_image}...")
    
    # Load image
    image = Image.open(input_image)
    
    # Create encoder
    encoder = RRIPEncoder(
        downsample_factor=downsample_factor,
        quality=quality,
        interpolation=interpolation
    )
    
    # Encode image
    encoded = encoder.encode_image(image, tile_size=tile_size)
    
    # Save
    encoder.save_encoded(encoded, output_file)
    
    click.echo(f"Compressed image saved to {output_file}")
    click.echo(f"Settings: downsample={downsample_factor}, quality={quality}, "
               f"interpolation={interpolation}, tile_size={tile_size}")


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_image', type=click.Path())
def decompress(input_file, output_image):
    """Decompress a RRIP-compressed image."""
    click.echo(f"Decompressing {input_file}...")
    
    # Create decoder
    decoder = RRIPDecoder()
    
    # Load and decode
    encoded = decoder.load_encoded(input_file)
    image = decoder.decode_image(encoded)
    
    # Save
    image.save(output_image)
    
    click.echo(f"Decompressed image saved to {output_image}")


@main.command()
@click.argument('input_image', type=click.Path(exists=True))
@click.argument('storage_dir', type=click.Path())
@click.argument('image_id')
@click.option('--downsample-factor', '-d', default=4, type=int,
              help='Downsampling factor for priors (default: 4)')
@click.option('--quality', '-q', default=50, type=int,
              help='Quality level 0-100 (default: 50)')
@click.option('--interpolation', '-i', default='bicubic',
              type=click.Choice(['bilinear', 'bicubic', 'nearest']),
              help='Interpolation method (default: bicubic)')
@click.option('--tile-size', '-t', default=256, type=int,
              help='Tile size (default: 256)')
def store(input_image, storage_dir, image_id, downsample_factor, quality, interpolation, tile_size):
    """Store an image as tiles in the tile manager."""
    click.echo(f"Storing {input_image} as {image_id}...")
    
    # Load image
    image = Image.open(input_image)
    
    # Create tile manager
    manager = TileManager(storage_dir)
    
    # Store image
    config = {
        'downsample_factor': downsample_factor,
        'quality': quality,
        'interpolation': interpolation,
        'tile_size': tile_size
    }
    
    metadata = manager.store_image(image_id, image, encoder_config=config)
    
    click.echo(f"Image stored successfully")
    click.echo(f"Tiles: {metadata['num_tiles']}")
    click.echo(f"Image size: {metadata['image_size']}")


@main.command()
@click.argument('storage_dir', type=click.Path(exists=True))
@click.argument('image_id')
@click.argument('output_image', type=click.Path())
def retrieve(storage_dir, image_id, output_image):
    """Retrieve and reconstruct an image from tiles."""
    click.echo(f"Retrieving {image_id}...")
    
    # Create tile manager
    manager = TileManager(storage_dir)
    
    # Get image
    image = manager.get_full_image(image_id)
    
    # Save
    image.save(output_image)
    
    click.echo(f"Image saved to {output_image}")


@main.command()
@click.argument('storage_dir', type=click.Path(exists=True))
@click.option('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
@click.option('--port', default=5000, type=int, help='Port to bind to (default: 5000)')
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
def serve(storage_dir, host, port, debug):
    """Start the tile server for efficient tile serving."""
    click.echo(f"Starting RRIP tile server on {host}:{port}...")
    click.echo(f"Storage directory: {storage_dir}")
    
    run_server(storage_dir, host=host, port=port, debug=debug)


@main.command()
@click.argument('storage_dir', type=click.Path(exists=True))
def list_images(storage_dir):
    """List all stored images."""
    manager = TileManager(storage_dir)
    images = manager.list_images()
    
    if not images:
        click.echo("No images stored")
        return
    
    click.echo(f"Stored images ({len(images)}):")
    for image_id in images:
        info = manager.get_image_info(image_id)
        click.echo(f"  {image_id}: {info['image_size']}, {info['num_tiles']} tiles")


@main.command()
@click.argument('input_image', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--downsample-factor', '-d', default=4, type=int,
              help='Downsampling factor for priors (default: 4)')
@click.option('--quality', '-q', default=50, type=int,
              help='Quality level 0-100 (default: 50)')
def benchmark(input_image, output_dir, downsample_factor, quality):
    """Benchmark compression with different settings."""
    import time
    import os
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Benchmarking {input_image}...")
    
    # Load image
    image = Image.open(input_image)
    original_size = os.path.getsize(input_image)
    
    click.echo(f"Original size: {original_size / 1024:.2f} KB")
    click.echo()
    
    # Test different settings
    encoder = RRIPEncoder(
        downsample_factor=downsample_factor,
        quality=quality,
        interpolation='bicubic'
    )
    
    start_time = time.time()
    encoded = encoder.encode_image(image, tile_size=256)
    encode_time = time.time() - start_time
    
    # Save to measure compressed size
    test_file = output_dir / "test.rrip"
    encoder.save_encoded(encoded, test_file)
    compressed_size = os.path.getsize(test_file)
    
    # Test decoding
    decoder = RRIPDecoder()
    start_time = time.time()
    decoded = decoder.decode_image(encoded)
    decode_time = time.time() - start_time
    
    # Save decoded
    decoded.save(output_dir / "decoded.png")
    
    # Results
    click.echo(f"Downsample factor: {downsample_factor}")
    click.echo(f"Quality: {quality}")
    click.echo(f"Compressed size: {compressed_size / 1024:.2f} KB")
    click.echo(f"Compression ratio: {original_size / compressed_size:.2f}x")
    click.echo(f"Encoding time: {encode_time:.3f}s")
    click.echo(f"Decoding time: {decode_time:.3f}s")


if __name__ == '__main__':
    main()
