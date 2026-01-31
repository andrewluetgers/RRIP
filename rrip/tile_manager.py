"""
Tile Manager - Efficient tile storage and retrieval for whole slide images
"""

import os
import pickle
import json
from pathlib import Path
from .encoder import RRIPEncoder
from .decoder import RRIPDecoder


class TileManager:
    """
    Manages compressed tiles for efficient storage and retrieval.
    
    Provides:
    - Tile-based storage with metadata
    - Fast tile lookup and loading
    - Memory-efficient serving
    """
    
    def __init__(self, storage_dir):
        """
        Initialize tile manager.
        
        Args:
            storage_dir: Directory to store compressed tiles
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.storage_dir / "metadata.json"
        self.tiles_dir = self.storage_dir / "tiles"
        self.tiles_dir.mkdir(exist_ok=True)
        
        self.decoder = RRIPDecoder()
        
    def store_image(self, image_id, image, encoder_config=None):
        """
        Store an image as compressed tiles.
        
        Args:
            image_id: Unique identifier for the image
            image: PIL Image or numpy array
            encoder_config: dict with encoder settings (downsample_factor, quality, etc.)
            
        Returns:
            dict: Storage metadata
        """
        # Create encoder with config
        if encoder_config is None:
            encoder_config = {}
        
        encoder = RRIPEncoder(
            downsample_factor=encoder_config.get('downsample_factor', 4),
            quality=encoder_config.get('quality', 50),
            interpolation=encoder_config.get('interpolation', 'bicubic')
        )
        
        tile_size = encoder_config.get('tile_size', 256)
        
        # Encode image
        encoded_data = encoder.encode_image(image, tile_size=tile_size)
        
        # Create image directory
        image_dir = self.tiles_dir / image_id
        image_dir.mkdir(exist_ok=True)
        
        # Store tiles individually
        tile_metadata = []
        for idx, (tile_data, position) in enumerate(
            zip(encoded_data['tiles'], encoded_data['positions'])
        ):
            tile_file = image_dir / f"tile_{idx}.rrip"
            with open(tile_file, 'wb') as f:
                pickle.dump(tile_data, f)
            
            tile_metadata.append({
                'index': idx,
                'position': position,
                'file': str(tile_file.relative_to(self.storage_dir))
            })
        
        # Store metadata
        metadata = {
            'image_id': image_id,
            'image_size': encoded_data['image_size'],
            'tile_size': tile_size,
            'num_tiles': len(tile_metadata),
            'tiles': tile_metadata,
            'encoder_config': encoder_config
        }
        
        # Update global metadata
        self._update_metadata(image_id, metadata)
        
        return metadata
    
    def get_tile(self, image_id, tile_index):
        """
        Retrieve and decode a specific tile.
        
        Args:
            image_id: Image identifier
            tile_index: Tile index
            
        Returns:
            PIL Image: Decoded tile
        """
        # Load metadata
        metadata = self._load_metadata()
        
        if image_id not in metadata:
            raise ValueError(f"Image {image_id} not found")
        
        image_meta = metadata[image_id]
        
        if tile_index >= image_meta['num_tiles']:
            raise ValueError(f"Tile index {tile_index} out of range")
        
        # Load tile
        tile_info = image_meta['tiles'][tile_index]
        tile_file = self.storage_dir / tile_info['file']
        
        with open(tile_file, 'rb') as f:
            tile_data = pickle.load(f)
        
        # Decode tile
        return self.decoder.decode_tile(tile_data)
    
    def get_tile_by_position(self, image_id, x, y):
        """
        Get tile at specific pixel position.
        
        Args:
            image_id: Image identifier
            x: X coordinate
            y: Y coordinate
            
        Returns:
            PIL Image: Decoded tile containing the position
        """
        metadata = self._load_metadata()
        
        if image_id not in metadata:
            raise ValueError(f"Image {image_id} not found")
        
        image_meta = metadata[image_id]
        tile_size = image_meta['tile_size']
        
        # Find tile containing position
        for tile_info in image_meta['tiles']:
            tx, ty = tile_info['position']
            if tx <= x < tx + tile_size and ty <= y < ty + tile_size:
                return self.get_tile(image_id, tile_info['index'])
        
        raise ValueError(f"No tile found at position ({x}, {y})")
    
    def get_full_image(self, image_id):
        """
        Reconstruct full image from tiles.
        
        Args:
            image_id: Image identifier
            
        Returns:
            PIL Image: Reconstructed full image
        """
        metadata = self._load_metadata()
        
        if image_id not in metadata:
            raise ValueError(f"Image {image_id} not found")
        
        image_meta = metadata[image_id]
        
        # Load all tiles
        tiles = []
        positions = []
        
        for tile_info in image_meta['tiles']:
            tile_file = self.storage_dir / tile_info['file']
            with open(tile_file, 'rb') as f:
                tile_data = pickle.load(f)
            tiles.append(tile_data)
            positions.append(tile_info['position'])
        
        # Reconstruct image
        encoded_data = {
            'tiles': tiles,
            'positions': positions,
            'image_size': tuple(image_meta['image_size']),
            'tile_size': image_meta['tile_size']
        }
        
        return self.decoder.decode_image(encoded_data)
    
    def list_images(self):
        """List all stored images."""
        metadata = self._load_metadata()
        return list(metadata.keys())
    
    def get_image_info(self, image_id):
        """Get metadata for an image."""
        metadata = self._load_metadata()
        return metadata.get(image_id)
    
    def _update_metadata(self, image_id, image_metadata):
        """Update global metadata file."""
        metadata = self._load_metadata()
        metadata[image_id] = image_metadata
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load global metadata."""
        if not self.metadata_file.exists():
            return {}
        
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
