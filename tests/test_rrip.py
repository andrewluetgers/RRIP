"""
Tests for RRIP encoder and decoder
"""

import unittest
import numpy as np
from PIL import Image
import tempfile
import os

from rrip.encoder import RRIPEncoder
from rrip.decoder import RRIPDecoder


class TestRRIPEncoder(unittest.TestCase):
    """Test RRIP encoder functionality."""
    
    def setUp(self):
        """Create test images."""
        # Create a simple test image
        self.test_image_rgb = Image.new('RGB', (256, 256), color=(100, 150, 200))
        self.test_image_gray = Image.new('L', (256, 256), color=128)
        
    def test_encoder_initialization(self):
        """Test encoder initialization with different parameters."""
        encoder = RRIPEncoder(downsample_factor=4, quality=50)
        self.assertEqual(encoder.downsample_factor, 4)
        self.assertEqual(encoder.quality, 50)
        self.assertEqual(encoder.interpolation, 'bicubic')
        
        encoder2 = RRIPEncoder(downsample_factor=8, quality=80, interpolation='bilinear')
        self.assertEqual(encoder2.downsample_factor, 8)
        self.assertEqual(encoder2.quality, 80)
        self.assertEqual(encoder2.interpolation, 'bilinear')
    
    def test_encode_tile_rgb(self):
        """Test encoding an RGB tile."""
        encoder = RRIPEncoder(downsample_factor=4, quality=50)
        encoded = encoder.encode_tile(self.test_image_rgb)
        
        # Check structure
        self.assertIn('prior', encoded)
        self.assertIn('compressed_residuals', encoded)
        self.assertIn('shape', encoded)
        self.assertIn('downsample_factor', encoded)
        
        # Check prior shape
        prior = encoded['prior']
        self.assertEqual(prior.shape, (64, 64, 3))  # 256/4 = 64
        
        # Check original shape preserved
        self.assertEqual(encoded['shape'], (256, 256, 3))
    
    def test_encode_tile_grayscale(self):
        """Test encoding a grayscale tile."""
        encoder = RRIPEncoder(downsample_factor=4, quality=50)
        encoded = encoder.encode_tile(self.test_image_gray)
        
        # Check prior shape
        prior = encoded['prior']
        self.assertEqual(prior.shape, (64, 64))  # 256/4 = 64
        
        # Check original shape preserved
        self.assertEqual(encoded['shape'], (256, 256))
    
    def test_encode_with_numpy_array(self):
        """Test encoding with numpy array input."""
        encoder = RRIPEncoder(downsample_factor=4, quality=50)
        array = np.array(self.test_image_rgb)
        encoded = encoder.encode_tile(array)
        
        self.assertIn('prior', encoded)
        self.assertIn('compressed_residuals', encoded)
    
    def test_quality_affects_quantization(self):
        """Test that quality setting affects quantization."""
        encoder_low = RRIPEncoder(downsample_factor=4, quality=10)
        encoder_high = RRIPEncoder(downsample_factor=4, quality=90)
        
        encoded_low = encoder_low.encode_tile(self.test_image_rgb)
        encoded_high = encoder_high.encode_tile(self.test_image_rgb)
        
        # Higher quality should have smaller quantization step
        quant_low = encoded_low['compressed_residuals']['quantization_step']
        quant_high = encoded_high['compressed_residuals']['quantization_step']
        
        self.assertGreater(quant_low, quant_high)
    
    def test_encode_full_image(self):
        """Test encoding a full image with tiles."""
        encoder = RRIPEncoder(downsample_factor=4, quality=50)
        
        # Create larger test image
        test_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
        
        encoded = encoder.encode_image(test_image, tile_size=256)
        
        # Check structure
        self.assertIn('tiles', encoded)
        self.assertIn('positions', encoded)
        self.assertIn('image_size', encoded)
        self.assertIn('tile_size', encoded)
        
        # Should have 4 tiles (2x2)
        self.assertEqual(len(encoded['tiles']), 4)
        self.assertEqual(len(encoded['positions']), 4)
    
    def test_save_and_load(self):
        """Test saving and loading encoded data."""
        encoder = RRIPEncoder(downsample_factor=4, quality=50)
        encoded = encoder.encode_tile(self.test_image_rgb)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.rrip') as f:
            temp_file = f.name
        
        try:
            encoder.save_encoded(encoded, temp_file)
            self.assertTrue(os.path.exists(temp_file))
            
            decoder = RRIPDecoder()
            loaded = decoder.load_encoded(temp_file)
            
            self.assertEqual(encoded['shape'], loaded['shape'])
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestRRIPDecoder(unittest.TestCase):
    """Test RRIP decoder functionality."""
    
    def setUp(self):
        """Create test data."""
        self.encoder = RRIPEncoder(downsample_factor=4, quality=50)
        self.decoder = RRIPDecoder()
        
        # Create test image
        self.test_image = Image.new('RGB', (256, 256), color=(100, 150, 200))
        self.encoded = self.encoder.encode_tile(self.test_image)
    
    def test_decode_tile(self):
        """Test decoding a tile."""
        decoded = self.decoder.decode_tile(self.encoded)
        
        # Check it's a PIL Image
        self.assertIsInstance(decoded, Image.Image)
        
        # Check dimensions match
        self.assertEqual(decoded.size, (256, 256))
    
    def test_decode_preserves_content(self):
        """Test that decoding preserves content reasonably."""
        # Use a more interesting test image
        test_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        
        encoded = self.encoder.encode_tile(test_image)
        decoded = self.decoder.decode_tile(encoded)
        
        # Compare
        original_array = np.array(test_image)
        decoded_array = np.array(decoded)
        
        # Should be similar but not exact (lossy compression)
        mean_error = np.mean(np.abs(original_array.astype(float) - decoded_array.astype(float)))
        
        # Error should be reasonable
        self.assertLess(mean_error, 50)  # Mean error < 50 pixel values
    
    def test_decode_full_image(self):
        """Test decoding a full image."""
        # Create and encode full image
        test_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
        encoded = self.encoder.encode_image(test_image, tile_size=256)
        
        # Decode
        decoded = self.decoder.decode_image(encoded)
        
        # Check dimensions
        self.assertEqual(decoded.size, (512, 512))
    
    def test_higher_quality_better_reconstruction(self):
        """Test that higher quality gives better reconstruction."""
        # Create test image with more detail
        test_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        
        # Encode with low quality
        encoder_low = RRIPEncoder(downsample_factor=4, quality=10)
        encoded_low = encoder_low.encode_tile(test_image)
        decoded_low = self.decoder.decode_tile(encoded_low)
        
        # Encode with high quality
        encoder_high = RRIPEncoder(downsample_factor=4, quality=90)
        encoded_high = encoder_high.encode_tile(test_image)
        decoded_high = self.decoder.decode_tile(encoded_high)
        
        # Compare errors
        original_array = np.array(test_image)
        error_low = np.mean(np.abs(original_array.astype(float) - np.array(decoded_low).astype(float)))
        error_high = np.mean(np.abs(original_array.astype(float) - np.array(decoded_high).astype(float)))
        
        # High quality should have lower error
        self.assertLess(error_high, error_low)


class TestRoundTrip(unittest.TestCase):
    """Test complete encode-decode round trips."""
    
    def test_round_trip_rgb(self):
        """Test encoding and decoding RGB image."""
        encoder = RRIPEncoder(downsample_factor=4, quality=70)
        decoder = RRIPDecoder()
        
        # Create test image
        original = Image.new('RGB', (256, 256))
        pixels = original.load()
        for i in range(256):
            for j in range(256):
                pixels[j, i] = (i, j, 128)
        
        # Encode and decode
        encoded = encoder.encode_tile(original)
        decoded = decoder.decode_tile(encoded)
        
        # Should maintain dimensions
        self.assertEqual(decoded.size, original.size)
        
        # Should maintain mode
        self.assertEqual(decoded.mode, original.mode)
    
    def test_round_trip_full_image(self):
        """Test encoding and decoding full image with tiles."""
        encoder = RRIPEncoder(downsample_factor=4, quality=70)
        decoder = RRIPDecoder()
        
        # Create larger test image
        original = Image.new('RGB', (512, 512))
        pixels = original.load()
        for i in range(512):
            for j in range(512):
                pixels[j, i] = (i % 256, j % 256, 128)
        
        # Encode and decode
        encoded = encoder.encode_image(original, tile_size=256)
        decoded = decoder.decode_image(encoded)
        
        # Should maintain dimensions
        self.assertEqual(decoded.size, original.size)


if __name__ == '__main__':
    unittest.main()
