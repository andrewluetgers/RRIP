"""
RRIP Encoder - Compresses tiles using interpolated priors and residuals
"""

import numpy as np
from scipy import interpolate
from PIL import Image
import io
import pickle
import zlib


class RRIPEncoder:
    """
    Encodes image tiles using interpolated priors and residual compression.
    
    The encoder:
    1. Downsamples the tile to create a low-resolution prior
    2. Upsamples the prior using interpolation
    3. Computes residuals (difference between original and interpolated)
    4. Compresses residuals with lossy quantization
    """
    
    def __init__(self, downsample_factor=4, quality=50, interpolation='bicubic'):
        """
        Initialize the RRIP encoder.
        
        Args:
            downsample_factor: Factor to downsample for creating priors (higher = more compression)
            quality: Quality level 0-100 (higher = better quality, less compression)
            interpolation: Interpolation method ('bilinear', 'bicubic', 'nearest')
        """
        self.downsample_factor = downsample_factor
        self.quality = quality
        self.interpolation = interpolation
        
    def encode_tile(self, tile):
        """
        Encode a single tile using interpolated prior and residuals.
        
        Args:
            tile: PIL Image or numpy array
            
        Returns:
            dict: Encoded data containing prior and compressed residuals
        """
        # Convert to numpy array if needed
        if isinstance(tile, Image.Image):
            tile_array = np.array(tile)
        else:
            tile_array = tile
            
        original_shape = tile_array.shape
        
        # Step 1: Create low-resolution prior by downsampling
        prior = self._create_prior(tile_array)
        
        # Step 2: Interpolate prior back to original resolution
        interpolated_prior = self._interpolate_prior(prior, original_shape)
        
        # Step 3: Compute residuals
        residuals = tile_array.astype(np.int16) - interpolated_prior.astype(np.int16)
        
        # Step 4: Quantize and compress residuals
        compressed_residuals = self._compress_residuals(residuals)
        
        # Package the encoded data
        encoded = {
            'prior': prior,
            'compressed_residuals': compressed_residuals,
            'shape': original_shape,
            'downsample_factor': self.downsample_factor,
            'quality': self.quality,
            'interpolation': self.interpolation
        }
        
        return encoded
    
    def _create_prior(self, tile_array):
        """Create low-resolution prior by downsampling."""
        if len(tile_array.shape) == 3:
            # RGB image
            h, w, c = tile_array.shape
            new_h = h // self.downsample_factor
            new_w = w // self.downsample_factor
            
            # Use PIL for high-quality downsampling
            img = Image.fromarray(tile_array)
            img_small = img.resize((new_w, new_h), Image.LANCZOS)
            return np.array(img_small)
        else:
            # Grayscale image
            h, w = tile_array.shape
            new_h = h // self.downsample_factor
            new_w = w // self.downsample_factor
            
            img = Image.fromarray(tile_array)
            img_small = img.resize((new_w, new_h), Image.LANCZOS)
            return np.array(img_small)
    
    def _interpolate_prior(self, prior, target_shape):
        """Interpolate prior to target resolution."""
        if len(target_shape) == 3:
            # RGB image
            h, w, c = target_shape
            img = Image.fromarray(prior)
            
            if self.interpolation == 'bicubic':
                resample = Image.BICUBIC
            elif self.interpolation == 'bilinear':
                resample = Image.BILINEAR
            else:
                resample = Image.NEAREST
                
            img_large = img.resize((w, h), resample)
            return np.array(img_large)
        else:
            # Grayscale image
            h, w = target_shape
            img = Image.fromarray(prior)
            
            if self.interpolation == 'bicubic':
                resample = Image.BICUBIC
            elif self.interpolation == 'bilinear':
                resample = Image.BILINEAR
            else:
                resample = Image.NEAREST
                
            img_large = img.resize((w, h), resample)
            return np.array(img_large)
    
    def _compress_residuals(self, residuals):
        """Quantize and compress residuals based on quality setting."""
        # Quality-based quantization
        # Higher quality = less quantization = more detail preserved
        # Quality 100: step=1 (minimal quantization)
        # Quality 50: step=25 (moderate quantization)
        # Quality 0: step=50 (high quantization)
        quantization_step = self._calculate_quantization_step(self.quality)
        
        # Quantize residuals
        quantized = (residuals / quantization_step).astype(np.int8)
        
        # Compress using zlib
        compressed = zlib.compress(quantized.tobytes(), level=9)
        
        return {
            'data': compressed,
            'quantization_step': quantization_step,
            'dtype': str(quantized.dtype),
            'shape': residuals.shape
        }
    
    def _calculate_quantization_step(self, quality):
        """
        Calculate quantization step based on quality setting.
        
        Args:
            quality: Quality value 0-100
            
        Returns:
            int: Quantization step size (1-50)
        """
        # Maps quality 0-100 to quantization step 50-1
        # Linear mapping: higher quality = smaller step = less quantization
        return max(1, int(101 - quality) // 2)
    
    def encode_image(self, image, tile_size=256):
        """
        Encode an entire image by splitting into tiles.
        
        Args:
            image: PIL Image or numpy array
            tile_size: Size of tiles (tile_size x tile_size)
            
        Returns:
            dict: Encoded tiles and metadata
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        width, height = image.size
        
        tiles = []
        positions = []
        
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Extract tile
                box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                tile = image.crop(box)
                
                # Encode tile
                encoded_tile = self.encode_tile(tile)
                tiles.append(encoded_tile)
                positions.append((x, y))
        
        return {
            'tiles': tiles,
            'positions': positions,
            'image_size': (width, height),
            'tile_size': tile_size
        }
    
    def save_encoded(self, encoded_data, filepath):
        """Save encoded data to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(encoded_data, f)
