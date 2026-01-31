"""
Tests for TileManager
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from rrip.tile_manager import TileManager


class TestTileManager(unittest.TestCase):
    """Test tile manager functionality."""
    
    def setUp(self):
        """Create temporary storage directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TileManager(self.temp_dir)
        
        # Create test image
        self.test_image = Image.new('RGB', (512, 512))
        pixels = self.test_image.load()
        for i in range(512):
            for j in range(512):
                pixels[j, i] = (i % 256, j % 256, 128)
    
    def tearDown(self):
        """Clean up temporary directory."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_store_image(self):
        """Test storing an image."""
        metadata = self.manager.store_image('test_image', self.test_image)
        
        # Check metadata
        self.assertEqual(metadata['image_id'], 'test_image')
        self.assertEqual(metadata['image_size'], (512, 512))
        self.assertGreater(metadata['num_tiles'], 0)
    
    def test_get_tile(self):
        """Test retrieving a tile."""
        self.manager.store_image('test_image', self.test_image)
        
        # Get first tile
        tile = self.manager.get_tile('test_image', 0)
        
        # Should be a PIL Image
        self.assertIsInstance(tile, Image.Image)
    
    def test_get_tile_by_position(self):
        """Test retrieving tile by position."""
        self.manager.store_image('test_image', self.test_image, 
                                encoder_config={'tile_size': 256})
        
        # Get tile at position
        tile = self.manager.get_tile_by_position('test_image', 100, 100)
        
        self.assertIsInstance(tile, Image.Image)
    
    def test_get_full_image(self):
        """Test reconstructing full image."""
        self.manager.store_image('test_image', self.test_image)
        
        # Retrieve full image
        reconstructed = self.manager.get_full_image('test_image')
        
        # Check dimensions
        self.assertEqual(reconstructed.size, self.test_image.size)
    
    def test_list_images(self):
        """Test listing images."""
        self.manager.store_image('image1', self.test_image)
        self.manager.store_image('image2', self.test_image)
        
        images = self.manager.list_images()
        
        self.assertEqual(len(images), 2)
        self.assertIn('image1', images)
        self.assertIn('image2', images)
    
    def test_get_image_info(self):
        """Test getting image info."""
        self.manager.store_image('test_image', self.test_image)
        
        info = self.manager.get_image_info('test_image')
        
        self.assertIsNotNone(info)
        self.assertEqual(info['image_id'], 'test_image')
    
    def test_nonexistent_image(self):
        """Test error handling for nonexistent image."""
        with self.assertRaises(ValueError):
            self.manager.get_tile('nonexistent', 0)
    
    def test_different_encoder_configs(self):
        """Test storing with different encoder configurations."""
        config1 = {'downsample_factor': 2, 'quality': 80, 'tile_size': 128}
        config2 = {'downsample_factor': 8, 'quality': 30, 'tile_size': 512}
        
        self.manager.store_image('image1', self.test_image, encoder_config=config1)
        self.manager.store_image('image2', self.test_image, encoder_config=config2)
        
        info1 = self.manager.get_image_info('image1')
        info2 = self.manager.get_image_info('image2')
        
        self.assertEqual(info1['tile_size'], 128)
        self.assertEqual(info2['tile_size'], 512)


if __name__ == '__main__':
    unittest.main()
