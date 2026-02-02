#!/usr/bin/env python3
"""
Generate JPEG Baseline Pyramids at Different Quality Levels

This script takes an existing DZI pyramid and re-encodes all tiles at specified
JPEG quality levels for compression comparison and quality analysis.

Usage:
    python generate_jpeg_baselines.py --input data/demo_out \
                                       --output results/jpeg_baselines \
                                       --qualities 60 80 90 \
                                       --reference-quality 90

Features:
- Re-encodes pyramid tiles at multiple JPEG quality levels
- Calculates storage statistics (size, compression ratio)
- Computes quality metrics (PSNR, SSIM) vs reference quality
- Generates structured JSON reports
- Supports level-specific and pyramid-wide analysis
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil
import io

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TileStats:
    """Statistics for a single tile"""
    level: int
    x: int
    y: int
    original_size_bytes: int
    reencoded_size_bytes: int
    compression_ratio: float
    psnr_db: float
    ssim: float
    encode_time_ms: float


@dataclass
class LevelStats:
    """Statistics for a pyramid level"""
    level: int
    quality: int
    num_tiles: int
    total_size_bytes: int
    avg_tile_size_bytes: float
    min_tile_size_bytes: int
    max_tile_size_bytes: int
    avg_compression_ratio: float
    avg_psnr_db: float
    avg_ssim: float
    avg_encode_time_ms: float


@dataclass
class PyramidStats:
    """Statistics for entire pyramid"""
    quality: int
    num_levels: int
    total_tiles: int
    total_size_bytes: int
    total_size_mb: float
    avg_tile_size_bytes: float
    raw_pyramid_size_bytes: int  # Assuming 256x256x3 RGB per tile
    compression_ratio: float
    avg_psnr_db: float
    avg_ssim: float
    level_stats: List[LevelStats]


# ============================================================================
# Quality Metrics
# ============================================================================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array

    Returns:
        PSNR in decibels (dB). Higher is better. Returns inf if images are identical.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray,
                   window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Simplified implementation using sliding window approach.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array
        window_size: Size of sliding window (default: 11)
        k1, k2: SSIM constants (default: 0.01, 0.03)

    Returns:
        SSIM value between -1 and 1. Higher is better (1 = identical).
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

    # Convert to grayscale if color
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = np.mean(img2, axis=2)

    # Constants
    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2

    # Create Gaussian window
    sigma = 1.5
    x = np.arange(window_size) - window_size // 2
    gauss = np.exp(-(x ** 2) / (2 * sigma ** 2))
    window = np.outer(gauss, gauss)
    window = window / window.sum()

    # Apply window convolution
    def convolve(img, window):
        from scipy import signal
        return signal.convolve2d(img, window, mode='valid')

    # Calculate statistics
    mu1 = convolve(img1, window)
    mu2 = convolve(img2, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1 ** 2, window) - mu1_sq
    sigma2_sq = convolve(img2 ** 2, window) - mu2_sq
    sigma12 = convolve(img1 * img2, window) - mu1_mu2

    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def calculate_simple_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simplified SSIM calculation without convolution (faster but less accurate).

    Uses global statistics instead of local windows.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array

    Returns:
        SSIM-like value between -1 and 1
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) == 3:
        img2 = np.mean(img2, axis=2)

    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Global statistics
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))

    return float(ssim)


# ============================================================================
# Pyramid Processing
# ============================================================================

class PyramidProcessor:
    """Process DZI pyramids and re-encode at different quality levels"""

    def __init__(self, input_dir: Path, output_dir: Path, use_simple_ssim: bool = False):
        """
        Initialize pyramid processor.

        Args:
            input_dir: Directory containing input DZI pyramid
            output_dir: Directory for output pyramids
            use_simple_ssim: Use simplified SSIM calculation (faster)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_simple_ssim = use_simple_ssim

        # Find DZI file
        dzi_files = list(input_dir.glob("*.dzi"))
        if not dzi_files:
            raise FileNotFoundError(f"No .dzi file found in {input_dir}")

        self.dzi_path = dzi_files[0]
        self.pyramid_name = self.dzi_path.stem
        self.tiles_dir = input_dir / f"{self.pyramid_name}_files"

        if not self.tiles_dir.exists():
            raise FileNotFoundError(f"Tiles directory not found: {self.tiles_dir}")

        # Calculate SSIM function
        self.calc_ssim = calculate_simple_ssim if use_simple_ssim else calculate_ssim

    def get_pyramid_levels(self) -> List[int]:
        """Get list of available pyramid levels"""
        levels = []
        for level_dir in sorted(self.tiles_dir.iterdir()):
            if level_dir.is_dir() and level_dir.name.isdigit():
                levels.append(int(level_dir.name))
        return sorted(levels)

    def get_level_tiles(self, level: int) -> List[Tuple[int, int]]:
        """
        Get list of tile coordinates for a level.

        Args:
            level: Pyramid level number

        Returns:
            List of (x, y) tile coordinates
        """
        level_dir = self.tiles_dir / str(level)
        tiles = []

        for tile_path in level_dir.glob("*.jpg"):
            try:
                x, y = tile_path.stem.split('_')
                tiles.append((int(x), int(y)))
            except ValueError:
                continue

        return tiles

    def process_tile(self, level: int, x: int, y: int,
                     quality: int, reference_img: Optional[np.ndarray] = None) -> TileStats:
        """
        Process a single tile: re-encode at specified quality and calculate metrics.

        Args:
            level: Pyramid level
            x, y: Tile coordinates
            quality: JPEG quality level (0-100)
            reference_img: Reference image for quality comparison (optional)

        Returns:
            TileStats object with processing results
        """
        tile_path = self.tiles_dir / str(level) / f"{x}_{y}.jpg"

        # Load original tile
        original_img = Image.open(tile_path)
        original_array = np.array(original_img)
        original_size = tile_path.stat().st_size

        # Re-encode at specified quality
        start_time = time.time()
        buffer = io.BytesIO()
        original_img.save(buffer, format='JPEG', quality=quality, optimize=True)
        encode_time = (time.time() - start_time) * 1000  # ms

        reencoded_bytes = buffer.getvalue()
        reencoded_size = len(reencoded_bytes)

        # Decode for quality comparison
        buffer.seek(0)
        reencoded_img = Image.open(buffer)
        reencoded_array = np.array(reencoded_img)

        # Calculate metrics vs reference (or original if no reference)
        ref_array = reference_img if reference_img is not None else original_array

        psnr = calculate_psnr(ref_array, reencoded_array)
        ssim = self.calc_ssim(ref_array, reencoded_array)

        compression_ratio = original_size / reencoded_size if reencoded_size > 0 else 0

        return TileStats(
            level=level,
            x=x,
            y=y,
            original_size_bytes=original_size,
            reencoded_size_bytes=reencoded_size,
            compression_ratio=compression_ratio,
            psnr_db=psnr,
            ssim=ssim,
            encode_time_ms=encode_time
        )

    def process_level(self, level: int, quality: int,
                     reference_quality: Optional[int] = None,
                     save_tiles: bool = True) -> Tuple[LevelStats, List[TileStats]]:
        """
        Process all tiles in a pyramid level.

        Args:
            level: Pyramid level number
            quality: JPEG quality to encode at
            reference_quality: Quality level to compare against (None = original)
            save_tiles: Whether to save re-encoded tiles to disk

        Returns:
            Tuple of (LevelStats, list of TileStats)
        """
        tiles = self.get_level_tiles(level)

        if not tiles:
            raise ValueError(f"No tiles found for level {level}")

        # Create output directory if saving
        if save_tiles:
            output_level_dir = self.output_dir / f"q{quality}" / f"{self.pyramid_name}_files" / str(level)
            output_level_dir.mkdir(parents=True, exist_ok=True)

        # Load reference images if comparing against different quality
        reference_images = {}
        if reference_quality is not None and reference_quality != quality:
            print(f"    Loading reference tiles at Q{reference_quality}...")
            for x, y in tqdm(tiles, desc="    Loading refs", leave=False):
                tile_path = self.tiles_dir / str(level) / f"{x}_{y}.jpg"
                img = Image.open(tile_path)

                # Re-encode at reference quality
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=reference_quality, optimize=True)
                buffer.seek(0)

                reference_images[(x, y)] = np.array(Image.open(buffer))

        # Process tiles
        tile_stats = []

        for x, y in tqdm(tiles, desc=f"    Level {level}", leave=False):
            ref_img = reference_images.get((x, y))
            stats = self.process_tile(level, x, y, quality, ref_img)
            tile_stats.append(stats)

            # Save re-encoded tile if requested
            if save_tiles:
                tile_path = self.tiles_dir / str(level) / f"{x}_{y}.jpg"
                img = Image.open(tile_path)

                output_path = output_level_dir / f"{x}_{y}.jpg"
                img.save(output_path, format='JPEG', quality=quality, optimize=True)

        # Calculate level statistics
        level_stats = LevelStats(
            level=level,
            quality=quality,
            num_tiles=len(tile_stats),
            total_size_bytes=sum(s.reencoded_size_bytes for s in tile_stats),
            avg_tile_size_bytes=np.mean([s.reencoded_size_bytes for s in tile_stats]),
            min_tile_size_bytes=min(s.reencoded_size_bytes for s in tile_stats),
            max_tile_size_bytes=max(s.reencoded_size_bytes for s in tile_stats),
            avg_compression_ratio=np.mean([s.compression_ratio for s in tile_stats]),
            avg_psnr_db=np.mean([s.psnr_db for s in tile_stats]),
            avg_ssim=np.mean([s.ssim for s in tile_stats]),
            avg_encode_time_ms=np.mean([s.encode_time_ms for s in tile_stats])
        )

        return level_stats, tile_stats

    def process_pyramid(self, quality: int,
                       reference_quality: Optional[int] = None,
                       save_tiles: bool = True,
                       levels: Optional[List[int]] = None) -> PyramidStats:
        """
        Process entire pyramid at specified quality.

        Args:
            quality: JPEG quality to encode at
            reference_quality: Quality level to compare against (None = original)
            save_tiles: Whether to save re-encoded tiles to disk
            levels: Specific levels to process (None = all levels)

        Returns:
            PyramidStats object with complete statistics
        """
        if levels is None:
            levels = self.get_pyramid_levels()

        print(f"\nProcessing pyramid at Q{quality} (reference: Q{reference_quality or 'original'})...")
        print(f"  Levels: {levels}")

        # Copy DZI manifest if saving
        if save_tiles:
            output_dzi_dir = self.output_dir / f"q{quality}"
            output_dzi_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.dzi_path, output_dzi_dir / self.dzi_path.name)

        # Process each level
        level_stats_list = []
        all_tile_stats = []

        for level in levels:
            level_stats, tile_stats = self.process_level(
                level, quality, reference_quality, save_tiles
            )
            level_stats_list.append(level_stats)
            all_tile_stats.extend(tile_stats)

        # Calculate pyramid statistics
        total_tiles = sum(ls.num_tiles for ls in level_stats_list)
        total_size = sum(ls.total_size_bytes for ls in level_stats_list)

        # Raw size assumes 256x256x3 bytes per tile (RGB)
        raw_tile_size = 256 * 256 * 3
        raw_pyramid_size = total_tiles * raw_tile_size

        pyramid_stats = PyramidStats(
            quality=quality,
            num_levels=len(level_stats_list),
            total_tiles=total_tiles,
            total_size_bytes=total_size,
            total_size_mb=total_size / (1024 * 1024),
            avg_tile_size_bytes=total_size / total_tiles if total_tiles > 0 else 0,
            raw_pyramid_size_bytes=raw_pyramid_size,
            compression_ratio=raw_pyramid_size / total_size if total_size > 0 else 0,
            avg_psnr_db=np.mean([s.psnr_db for s in all_tile_stats]),
            avg_ssim=np.mean([s.ssim for s in all_tile_stats]),
            level_stats=level_stats_list
        )

        return pyramid_stats


# ============================================================================
# Report Generation
# ============================================================================

def generate_comparison_report(pyramid_stats_list: List[PyramidStats],
                              output_path: Path):
    """
    Generate comparison report across multiple quality levels.

    Args:
        pyramid_stats_list: List of PyramidStats for different qualities
        output_path: Path to save JSON report
    """
    report = {
        "summary": {
            "num_qualities": len(pyramid_stats_list),
            "qualities": [ps.quality for ps in pyramid_stats_list],
            "total_tiles": pyramid_stats_list[0].total_tiles if pyramid_stats_list else 0,
            "num_levels": pyramid_stats_list[0].num_levels if pyramid_stats_list else 0
        },
        "quality_comparison": []
    }

    for ps in sorted(pyramid_stats_list, key=lambda x: x.quality, reverse=True):
        report["quality_comparison"].append({
            "quality": ps.quality,
            "total_size_mb": round(ps.total_size_mb, 2),
            "avg_tile_size_kb": round(ps.avg_tile_size_bytes / 1024, 2),
            "compression_ratio": round(ps.compression_ratio, 2),
            "avg_psnr_db": round(ps.avg_psnr_db, 2),
            "avg_ssim": round(ps.avg_ssim, 4),
            "level_breakdown": [
                {
                    "level": ls.level,
                    "num_tiles": ls.num_tiles,
                    "avg_tile_size_kb": round(ls.avg_tile_size_bytes / 1024, 2),
                    "avg_psnr_db": round(ls.avg_psnr_db, 2),
                    "avg_ssim": round(ls.avg_ssim, 4)
                }
                for ls in ps.level_stats
            ]
        })

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nComparison report saved to: {output_path}")


def print_summary_table(pyramid_stats_list: List[PyramidStats]):
    """Print formatted summary table to console"""
    print("\n" + "="*80)
    print("JPEG BASELINE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Quality':<10} {'Size (MB)':<12} {'Avg Tile (KB)':<15} "
          f"{'Comp Ratio':<12} {'PSNR (dB)':<12} {'SSIM':<8}")
    print("-"*80)

    for ps in sorted(pyramid_stats_list, key=lambda x: x.quality, reverse=True):
        print(f"Q{ps.quality:<8} {ps.total_size_mb:<12.2f} "
              f"{ps.avg_tile_size_bytes/1024:<15.2f} "
              f"{ps.compression_ratio:<12.2f} "
              f"{ps.avg_psnr_db:<12.2f} {ps.avg_ssim:<8.4f}")

    print("="*80)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate JPEG baseline pyramids at different quality levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate pyramids at Q60, Q80, Q90
  python generate_jpeg_baselines.py --input data/demo_out --qualities 60 80 90

  # Compare against Q90 reference
  python generate_jpeg_baselines.py --input data/demo_out --qualities 60 80 \\
                                     --reference-quality 90

  # Process specific levels only
  python generate_jpeg_baselines.py --input data/demo_out --qualities 60 80 90 \\
                                     --levels 14 15 16

  # Skip saving tiles (stats only)
  python generate_jpeg_baselines.py --input data/demo_out --qualities 60 80 90 \\
                                     --no-save-tiles
        """
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing DZI pyramid')
    parser.add_argument('--output', type=str, default='results/jpeg_baselines',
                       help='Output directory for re-encoded pyramids (default: results/jpeg_baselines)')
    parser.add_argument('--qualities', type=int, nargs='+', default=[60, 80, 90],
                       help='JPEG quality levels to generate (default: 60 80 90)')
    parser.add_argument('--reference-quality', type=int, default=None,
                       help='Reference quality for comparison (default: original tiles)')
    parser.add_argument('--levels', type=int, nargs='+', default=None,
                       help='Specific levels to process (default: all levels)')
    parser.add_argument('--no-save-tiles', action='store_true',
                       help='Skip saving re-encoded tiles (stats only)')
    parser.add_argument('--simple-ssim', action='store_true',
                       help='Use simplified SSIM calculation (faster)')

    args = parser.parse_args()

    # Validate inputs
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = PyramidProcessor(input_dir, output_dir, args.simple_ssim)

    print("="*80)
    print("JPEG BASELINE PYRAMID GENERATOR")
    print("="*80)
    print(f"Input pyramid: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Quality levels: {args.qualities}")
    print(f"Reference quality: {args.reference_quality or 'original'}")
    print(f"Saving tiles: {not args.no_save_tiles}")
    print(f"SSIM method: {'simplified' if args.simple_ssim else 'full'}")

    # Process pyramids at each quality level
    pyramid_stats_list = []

    for quality in sorted(args.qualities, reverse=True):
        try:
            stats = processor.process_pyramid(
                quality=quality,
                reference_quality=args.reference_quality,
                save_tiles=not args.no_save_tiles,
                levels=args.levels
            )
            pyramid_stats_list.append(stats)

            # Save individual quality report
            quality_report_path = output_dir / f"q{quality}_stats.json"
            with open(quality_report_path, 'w') as f:
                json.dump(asdict(stats), f, indent=2)

            print(f"  Saved stats to: {quality_report_path}")

        except Exception as e:
            print(f"Error processing Q{quality}: {e}")
            continue

    # Generate comparison report
    if pyramid_stats_list:
        comparison_path = output_dir / "quality_comparison.json"
        generate_comparison_report(pyramid_stats_list, comparison_path)

        # Print summary table
        print_summary_table(pyramid_stats_list)

        # Save detailed summary
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("JPEG BASELINE PYRAMID COMPARISON\n")
            f.write("="*80 + "\n\n")

            for ps in sorted(pyramid_stats_list, key=lambda x: x.quality, reverse=True):
                f.write(f"\nQuality Level: {ps.quality}\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Size: {ps.total_size_mb:.2f} MB\n")
                f.write(f"Total Tiles: {ps.total_tiles}\n")
                f.write(f"Number of Levels: {ps.num_levels}\n")
                f.write(f"Average Tile Size: {ps.avg_tile_size_bytes/1024:.2f} KB\n")
                f.write(f"Compression Ratio: {ps.compression_ratio:.2f}x\n")
                f.write(f"Average PSNR: {ps.avg_psnr_db:.2f} dB\n")
                f.write(f"Average SSIM: {ps.avg_ssim:.4f}\n")

                f.write(f"\nPer-Level Breakdown:\n")
                for ls in ps.level_stats:
                    f.write(f"  Level {ls.level}: {ls.num_tiles} tiles, "
                           f"{ls.avg_tile_size_bytes/1024:.2f} KB avg, "
                           f"PSNR {ls.avg_psnr_db:.2f} dB, "
                           f"SSIM {ls.avg_ssim:.4f}\n")

        print(f"\nDetailed summary saved to: {summary_path}")
        print("\nDone!")
        return 0
    else:
        print("\nError: No pyramids were successfully processed")
        return 1


if __name__ == '__main__':
    exit(main())
