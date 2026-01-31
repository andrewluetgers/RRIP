"""
RRIP - Residual Reconstruction from Interpolated Priors
Optimized whole slide lossy tile compression and efficient serving
"""

__version__ = "0.1.0"

from .encoder import RRIPEncoder
from .decoder import RRIPDecoder
from .tile_manager import TileManager

__all__ = ["RRIPEncoder", "RRIPDecoder", "TileManager"]
