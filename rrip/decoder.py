"""
RRIP Decoder - Reconstructs tiles from interpolated priors and residuals
"""

import numpy as np
from PIL import Image
import pickle
import zlib


class RRIPDecoder:
    """
    Decodes image tiles compressed with RRIP encoding.
    
    The decoder:
    1. Loads the low-resolution prior
    2. Interpolates prior to target resolution
    3. Decompresses and dequantizes residuals
    4. Adds residuals to interpolated prior
    """
    
    def decode_tile(self, encoded_tile):
        """
        Decode a single tile from encoded data.
        
        Args:
            encoded_tile: dict containing prior and compressed residuals
            
        Returns:
            PIL Image: Reconstructed tile
        """
        # Extract metadata
        prior = encoded_tile['prior']
        compressed_residuals = encoded_tile['compressed_residuals']
        original_shape = encoded_tile['shape']
        interpolation = encoded_tile['interpolation']
        
        # Step 1: Interpolate prior to original resolution
        interpolated_prior = self._interpolate_prior(
            prior, original_shape, interpolation
        )
        
        # Step 2: Decompress residuals
        residuals = self._decompress_residuals(compressed_residuals)
        
        # Step 3: Add residuals to prior
        reconstructed = interpolated_prior.astype(np.int16) + residuals
        
        # Clip to valid range
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return Image.fromarray(reconstructed)
    
    def _get_resample_method(self, interpolation):
        """
        Get PIL resample method from interpolation string.
        
        Args:
            interpolation: Interpolation method string
            
        Returns:
            PIL resample constant
        """
        resample_map = {
            'bicubic': Image.BICUBIC,
            'bilinear': Image.BILINEAR,
            'nearest': Image.NEAREST
        }
        return resample_map.get(interpolation, Image.BICUBIC)
    
    def _interpolate_prior(self, prior, target_shape, interpolation):
        """Interpolate prior to target resolution."""
        resample = self._get_resample_method(interpolation)
        
        if len(target_shape) == 3:
            # RGB image
            h, w, c = target_shape
            img = Image.fromarray(prior)
            img_large = img.resize((w, h), resample)
            return np.array(img_large)
        else:
            # Grayscale image
            h, w = target_shape
            img = Image.fromarray(prior)
            img_large = img.resize((w, h), resample)
            return np.array(img_large)
    
    def _decompress_residuals(self, compressed_residuals):
        """Decompress and dequantize residuals."""
        # Decompress
        compressed_data = compressed_residuals['data']
        quantization_step = compressed_residuals['quantization_step']
        shape = compressed_residuals['shape']
        
        decompressed = zlib.decompress(compressed_data)
        quantized = np.frombuffer(decompressed, dtype=np.int8).reshape(shape)
        
        # Dequantize
        residuals = quantized.astype(np.int16) * quantization_step
        
        return residuals
    
    def decode_image(self, encoded_data):
        """
        Decode an entire image from encoded tiles.
        
        Args:
            encoded_data: dict containing tiles, positions, and metadata
            
        Returns:
            PIL Image: Reconstructed image
        """
        tiles = encoded_data['tiles']
        positions = encoded_data['positions']
        image_size = encoded_data['image_size']
        tile_size = encoded_data['tile_size']
        
        # Create output image
        width, height = image_size
        
        # Determine number of channels from first tile
        first_tile_shape = tiles[0]['shape']
        if len(first_tile_shape) == 3:
            channels = first_tile_shape[2]
            output = Image.new('RGB', (width, height))
        else:
            output = Image.new('L', (width, height))
        
        # Decode and paste tiles
        for tile_data, (x, y) in zip(tiles, positions):
            decoded_tile = self.decode_tile(tile_data)
            output.paste(decoded_tile, (x, y))
        
        return output
    
    def load_encoded(self, filepath):
        """Load encoded data from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
