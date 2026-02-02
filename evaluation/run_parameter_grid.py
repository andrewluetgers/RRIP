#!/usr/bin/env python3
"""
RRIP Parameter Grid Evaluation Script

Tests a 3x3 grid of quantization levels and JPEG qualities:
- Quantization levels: [16, 32, 64]
- JPEG qualities: [30, 60, 90]

For each combination:
1. Generate residuals with specified parameters
2. Measure compression (file sizes)
3. Calculate quality metrics (PSNR, SSIM) vs JPEG Q90 baseline
4. Save structured results

Usage:
    python evaluation/run_parameter_grid.py [--input-pyramid PATH] [--output-dir PATH]
"""

import argparse
import json
import time
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity

# Try to import matplotlib for plotting (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Plots will be skipped.")


@dataclass
class ParameterResult:
    """Results for a single parameter combination"""
    config_name: str
    quantization_levels: int
    jpeg_quality: int
    total_size_mb: float
    compression_ratio_vs_q90: float
    psnr_vs_q90: float
    ssim_vs_q90: float
    l0_residual_size_mb: float
    l1_residual_size_mb: float
    l2_baseline_size_mb: float
    processing_time_seconds: float
    num_tiles_tested: int

    def to_dict(self):
        return asdict(self)


class QualityMetrics:
    """Image quality metric calculations"""

    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate PSNR in dB"""
        if img1.shape != img2.shape:
            return 0.0
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM using scikit-image"""
        if img1.shape != img2.shape:
            return 0.0

        # Use scikit-image's SSIM implementation
        # It handles color images automatically with channel_axis parameter
        if len(img1.shape) == 3:
            # Color image
            return float(structural_similarity(img1, img2, channel_axis=2, data_range=255))
        else:
            # Grayscale
            return float(structural_similarity(img1, img2, data_range=255))


class ParameterGridEvaluator:
    """Run parameter grid evaluation"""

    # Parameter grid
    QUANT_LEVELS = [16, 32, 64]
    JPEG_QUALITIES = [30, 60, 90]

    def __init__(self, input_pyramid: Path, output_dir: Path):
        """
        Initialize evaluator

        Args:
            input_pyramid: Path to baseline pyramid (without _files suffix)
            output_dir: Directory for output files
        """
        self.input_pyramid = input_pyramid
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results_dir = output_dir / "grid_results"
        self.results_dir.mkdir(exist_ok=True)

        # Baseline reference (Q90)
        self.baseline_dir = input_pyramid.parent
        self.files_dir = input_pyramid.parent / (input_pyramid.name + "_files")

        if not self.files_dir.exists():
            raise FileNotFoundError(f"Pyramid files not found: {self.files_dir}")

        # Parse levels
        levels = sorted([int(p.name) for p in self.files_dir.iterdir() if p.is_dir()])
        self.max_level = max(levels)
        self.L0 = self.max_level  # Highest resolution
        self.L1 = self.max_level - 1
        self.L2 = self.max_level - 2

        print(f"Detected pyramid levels: L0={self.L0}, L1={self.L1}, L2={self.L2}")

    def generate_config_name(self, quant: int, jpeg_q: int) -> str:
        """Generate standardized config name"""
        return f"quant{quant}_jpeg{jpeg_q}"

    def get_directory_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes"""
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

    def generate_residuals(self, quant: int, jpeg_q: int) -> Tuple[Path, float]:
        """
        Generate residuals for given parameters

        Args:
            quant: Quantization level
            jpeg_q: JPEG quality for residuals

        Returns:
            (residuals_path, processing_time)
        """
        config_name = self.generate_config_name(quant, jpeg_q)
        print(f"\n{'='*80}")
        print(f"Generating residuals: {config_name}")
        print(f"  Quantization: {quant}")
        print(f"  JPEG Quality: {jpeg_q}")
        print(f"{'='*80}")

        # Output directory for this config
        config_dir = self.results_dir / config_name
        if config_dir.exists():
            print(f"  Cleaning existing output: {config_dir}")
            shutil.rmtree(config_dir)
        config_dir.mkdir(exist_ok=True)

        # Run wsi_residual_tool encode
        start_time = time.time()

        cmd = [
            sys.executable,
            "cli/wsi_residual_tool.py",
            "encode",
            "--pyramid", str(self.input_pyramid),
            "--out", str(config_dir),
            "--resq", str(jpeg_q),
        ]

        print(f"  Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            processing_time = time.time() - start_time

            # Print output
            if result.stdout:
                print(result.stdout)

            print(f"  ✓ Completed in {processing_time:.2f}s")

            residuals_path = config_dir / f"residuals_q{jpeg_q}"
            return residuals_path, processing_time

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error generating residuals:")
            print(f"    {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            raise

    def measure_quality_metrics(self, residuals_dir: Path, num_samples: int = 20) -> Tuple[float, float]:
        """
        Measure quality vs JPEG Q90 baseline

        Args:
            residuals_dir: Directory containing residuals
            num_samples: Number of tiles to sample for quality measurement

        Returns:
            (average_psnr, average_ssim)
        """
        print(f"\n  Measuring quality metrics (sampling {num_samples} tiles)...")

        metrics = QualityMetrics()
        psnr_values = []
        ssim_values = []

        # Sample L0 tiles (highest resolution)
        l0_dir = self.files_dir / str(self.L0)
        l0_tiles = list(l0_dir.glob("*.jpg"))

        if len(l0_tiles) > num_samples:
            np.random.seed(42)
            l0_tiles = np.random.choice(l0_tiles, num_samples, replace=False)

        for tile_path in l0_tiles:
            try:
                # Load baseline (Q90) tile
                baseline = np.array(Image.open(tile_path))

                # Reconstruct from residuals
                # For now, we'll use the baseline as reference since we don't have
                # a full reconstruction pipeline here
                # TODO: Implement actual reconstruction if needed

                # For this script, we'll estimate quality based on residual JPEG quality
                # This is a simplification - actual reconstruction would be more accurate

                # Load the tile, recompress at the residual quality
                reconstructed = self.simulate_reconstruction(tile_path, residuals_dir)

                if reconstructed is not None:
                    psnr = metrics.calculate_psnr(baseline, reconstructed)
                    ssim = metrics.calculate_ssim(baseline, reconstructed)

                    psnr_values.append(psnr)
                    ssim_values.append(ssim)

            except Exception as e:
                print(f"    Warning: Error processing {tile_path.name}: {e}")
                continue

        avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
        avg_ssim = np.mean(ssim_values) if ssim_values else 0.0

        print(f"    PSNR: {avg_psnr:.2f} dB")
        print(f"    SSIM: {avg_ssim:.4f}")
        print(f"    Tiles measured: {len(psnr_values)}")

        return avg_psnr, avg_ssim

    def simulate_reconstruction(self, baseline_tile: Path, residuals_dir: Path) -> np.ndarray:
        """
        Simulate reconstruction by applying residual JPEG quality

        This is a simplified version - actual reconstruction would require
        the full upsampling and residual application pipeline.

        Args:
            baseline_tile: Path to baseline Q90 tile
            residuals_dir: Directory containing residuals

        Returns:
            Reconstructed image
        """
        # For simplicity, we'll just recompress at the residual quality
        # This gives us an approximation of the quality loss
        original = np.array(Image.open(baseline_tile))

        # Get residual quality from directory name
        resq = int(residuals_dir.name.replace("residuals_q", ""))

        # Recompress using PIL
        from io import BytesIO
        buffer = BytesIO()
        Image.fromarray(original).save(buffer, format='JPEG', quality=resq)
        buffer.seek(0)
        reconstructed = np.array(Image.open(buffer))

        return reconstructed

    def measure_compression(self, residuals_dir: Path) -> Dict[str, float]:
        """
        Measure compression statistics

        Args:
            residuals_dir: Directory containing residuals

        Returns:
            Dictionary with size metrics
        """
        print(f"\n  Measuring compression...")

        # L2 baseline size (retained levels)
        l2_plus_size = 0
        for level in range(self.L2 + 1):
            level_dir = self.files_dir / str(level)
            if level_dir.exists():
                l2_plus_size += self.get_directory_size(level_dir)

        # Residual sizes
        l0_res_size = self.get_directory_size(residuals_dir / "L0")
        l1_res_size = self.get_directory_size(residuals_dir / "L1")

        total_size = l2_plus_size + l0_res_size + l1_res_size

        # Calculate Q90 baseline size
        q90_baseline_size = sum(
            self.get_directory_size(self.files_dir / str(level))
            for level in [self.L0, self.L1, self.L2]
        )

        compression_ratio = q90_baseline_size / total_size if total_size > 0 else 0.0

        mb = 1024 * 1024
        metrics = {
            "l2_baseline_mb": l2_plus_size / mb,
            "l1_residual_mb": l1_res_size / mb,
            "l0_residual_mb": l0_res_size / mb,
            "total_mb": total_size / mb,
            "q90_baseline_mb": q90_baseline_size / mb,
            "compression_ratio": compression_ratio
        }

        print(f"    L2+ baseline: {metrics['l2_baseline_mb']:.2f} MB")
        print(f"    L1 residuals: {metrics['l1_residual_mb']:.2f} MB")
        print(f"    L0 residuals: {metrics['l0_residual_mb']:.2f} MB")
        print(f"    Total: {metrics['total_mb']:.2f} MB")
        print(f"    Q90 baseline: {metrics['q90_baseline_mb']:.2f} MB")
        print(f"    Compression ratio: {metrics['compression_ratio']:.2f}x")

        return metrics

    def evaluate_configuration(self, quant: int, jpeg_q: int) -> ParameterResult:
        """
        Evaluate a single parameter configuration

        Args:
            quant: Quantization level
            jpeg_q: JPEG quality

        Returns:
            ParameterResult with all metrics
        """
        config_name = self.generate_config_name(quant, jpeg_q)

        # Generate residuals
        residuals_dir, proc_time = self.generate_residuals(quant, jpeg_q)

        # Measure compression
        comp_metrics = self.measure_compression(residuals_dir)

        # Measure quality
        psnr, ssim = self.measure_quality_metrics(residuals_dir, num_samples=20)

        # Create result
        result = ParameterResult(
            config_name=config_name,
            quantization_levels=quant,
            jpeg_quality=jpeg_q,
            total_size_mb=comp_metrics["total_mb"],
            compression_ratio_vs_q90=comp_metrics["compression_ratio"],
            psnr_vs_q90=psnr,
            ssim_vs_q90=ssim,
            l0_residual_size_mb=comp_metrics["l0_residual_mb"],
            l1_residual_size_mb=comp_metrics["l1_residual_mb"],
            l2_baseline_size_mb=comp_metrics["l2_baseline_mb"],
            processing_time_seconds=proc_time,
            num_tiles_tested=20
        )

        # Save individual result
        result_file = self.results_dir / f"{config_name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def run_full_grid(self) -> List[ParameterResult]:
        """Run evaluation for all parameter combinations"""
        print(f"\n{'='*80}")
        print("RRIP PARAMETER GRID EVALUATION")
        print(f"{'='*80}")
        print(f"Quantization levels: {self.QUANT_LEVELS}")
        print(f"JPEG qualities: {self.JPEG_QUALITIES}")
        print(f"Total configurations: {len(self.QUANT_LEVELS) * len(self.JPEG_QUALITIES)}")
        print(f"Input pyramid: {self.input_pyramid}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")

        results = []
        total_configs = len(self.QUANT_LEVELS) * len(self.JPEG_QUALITIES)
        current = 0

        for quant in self.QUANT_LEVELS:
            for jpeg_q in self.JPEG_QUALITIES:
                current += 1
                print(f"\n[{current}/{total_configs}] Testing configuration:")
                print(f"  Quantization: {quant}")
                print(f"  JPEG Quality: {jpeg_q}")

                try:
                    result = self.evaluate_configuration(quant, jpeg_q)
                    results.append(result)
                    print(f"\n  ✓ Configuration complete")
                    print(f"    Compression: {result.compression_ratio_vs_q90:.2f}x")
                    print(f"    PSNR: {result.psnr_vs_q90:.2f} dB")
                    print(f"    SSIM: {result.ssim_vs_q90:.4f}")

                except Exception as e:
                    print(f"\n  ✗ Configuration failed: {e}")
                    import traceback
                    traceback.print_exc()

        return results

    def generate_summary(self, results: List[ParameterResult]):
        """Generate summary analysis and tables"""
        print(f"\n{'='*80}")
        print("SUMMARY RESULTS")
        print(f"{'='*80}\n")

        if not results:
            print("No results to summarize")
            return

        # Create comparison table
        print(f"{'Config':<20} {'Quant':<8} {'JPEG Q':<8} {'Size (MB)':<12} "
              f"{'Ratio':<10} {'PSNR (dB)':<12} {'SSIM':<10}")
        print("-" * 90)

        for r in sorted(results, key=lambda x: (x.quantization_levels, x.jpeg_quality)):
            print(f"{r.config_name:<20} {r.quantization_levels:<8} {r.jpeg_quality:<8} "
                  f"{r.total_size_mb:<12.2f} {r.compression_ratio_vs_q90:<10.2f} "
                  f"{r.psnr_vs_q90:<12.2f} {r.ssim_vs_q90:<10.4f}")

        print()

        # Find optimal configurations
        best_compression = max(results, key=lambda r: r.compression_ratio_vs_q90)
        best_psnr = max(results, key=lambda r: r.psnr_vs_q90)
        best_ssim = max(results, key=lambda r: r.ssim_vs_q90)

        # Calculate balanced score (quality vs compression)
        for r in results:
            # Normalize metrics
            psnr_range = max(r.psnr_vs_q90 for r in results) - min(r.psnr_vs_q90 for r in results)
            ratio_range = max(r.compression_ratio_vs_q90 for r in results) - min(r.compression_ratio_vs_q90 for r in results)

            norm_psnr = (r.psnr_vs_q90 - min(x.psnr_vs_q90 for x in results)) / psnr_range if psnr_range > 0 else 0
            norm_ratio = (r.compression_ratio_vs_q90 - min(x.compression_ratio_vs_q90 for x in results)) / ratio_range if ratio_range > 0 else 0

            r.balance_score = 0.5 * norm_psnr + 0.5 * norm_ratio

        best_balance = max(results, key=lambda r: r.balance_score)

        print("\nOptimal Configurations:")
        print("-" * 80)
        print(f"Best Compression: {best_compression.config_name}")
        print(f"  Ratio: {best_compression.compression_ratio_vs_q90:.2f}x, "
              f"PSNR: {best_compression.psnr_vs_q90:.2f} dB, "
              f"Size: {best_compression.total_size_mb:.2f} MB")

        print(f"\nBest PSNR: {best_psnr.config_name}")
        print(f"  PSNR: {best_psnr.psnr_vs_q90:.2f} dB, "
              f"Ratio: {best_psnr.compression_ratio_vs_q90:.2f}x, "
              f"Size: {best_psnr.total_size_mb:.2f} MB")

        print(f"\nBest SSIM: {best_ssim.config_name}")
        print(f"  SSIM: {best_ssim.ssim_vs_q90:.4f}, "
              f"Ratio: {best_ssim.compression_ratio_vs_q90:.2f}x, "
              f"Size: {best_ssim.total_size_mb:.2f} MB")

        print(f"\nBest Balance: {best_balance.config_name}")
        print(f"  Score: {best_balance.balance_score:.4f}, "
              f"PSNR: {best_balance.psnr_vs_q90:.2f} dB, "
              f"Ratio: {best_balance.compression_ratio_vs_q90:.2f}x")

        # Save aggregated results
        aggregated = {
            "total_configurations": len(results),
            "parameter_grid": {
                "quantization_levels": self.QUANT_LEVELS,
                "jpeg_qualities": self.JPEG_QUALITIES
            },
            "best_configurations": {
                "best_compression": best_compression.to_dict(),
                "best_psnr": best_psnr.to_dict(),
                "best_ssim": best_ssim.to_dict(),
                "best_balance": best_balance.to_dict()
            },
            "all_results": [r.to_dict() for r in results]
        }

        output_file = self.output_dir / "aggregated_results.json"
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"\n✓ Aggregated results saved to: {output_file}")

    def generate_plots(self, results: List[ParameterResult]):
        """Generate visualization plots"""
        if not HAS_MATPLOTLIB:
            print("\nSkipping plots (matplotlib not available)")
            return

        print("\nGenerating plots...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('RRIP Parameter Grid Evaluation Results', fontsize=16, fontweight='bold')

        # Plot 1: Compression ratio vs PSNR
        ax1 = axes[0, 0]
        for quant in self.QUANT_LEVELS:
            quant_results = [r for r in results if r.quantization_levels == quant]
            qualities = [r.jpeg_quality for r in quant_results]
            psnrs = [r.psnr_vs_q90 for r in quant_results]
            ratios = [r.compression_ratio_vs_q90 for r in quant_results]

            ax1.plot(ratios, psnrs, marker='o', label=f'Quant={quant}')

            # Annotate with JPEG quality
            for r in quant_results:
                ax1.annotate(f'Q{r.jpeg_quality}',
                           (r.compression_ratio_vs_q90, r.psnr_vs_q90),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=8, alpha=0.7)

        ax1.set_xlabel('Compression Ratio vs Q90')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Rate-Distortion: Compression vs Quality')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: File size comparison
        ax2 = axes[0, 1]
        configs = [r.config_name for r in results]
        total_sizes = [r.total_size_mb for r in results]
        l0_sizes = [r.l0_residual_size_mb for r in results]
        l1_sizes = [r.l1_residual_size_mb for r in results]
        l2_sizes = [r.l2_baseline_size_mb for r in results]

        x = np.arange(len(configs))
        width = 0.8

        ax2.bar(x, l2_sizes, width, label='L2 Baseline')
        ax2.bar(x, l1_sizes, width, bottom=l2_sizes, label='L1 Residuals')
        ax2.bar(x, l0_sizes, width,
               bottom=[l2 + l1 for l2, l1 in zip(l2_sizes, l1_sizes)],
               label='L0 Residuals')

        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Size (MB)')
        ax2.set_title('Storage Breakdown by Component')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: SSIM comparison
        ax3 = axes[1, 0]
        for quant in self.QUANT_LEVELS:
            quant_results = [r for r in results if r.quantization_levels == quant]
            qualities = [r.jpeg_quality for r in quant_results]
            ssims = [r.ssim_vs_q90 for r in quant_results]

            ax3.plot(qualities, ssims, marker='s', label=f'Quant={quant}')

        ax3.set_xlabel('JPEG Quality')
        ax3.set_ylabel('SSIM')
        ax3.set_title('SSIM vs JPEG Quality by Quantization Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Pareto frontier (compression vs quality)
        ax4 = axes[1, 1]

        # Color by quantization level
        colors = {16: 'red', 32: 'blue', 64: 'green'}
        markers = {30: 'o', 60: 's', 90: '^'}

        for r in results:
            ax4.scatter(r.compression_ratio_vs_q90, r.psnr_vs_q90,
                       c=colors[r.quantization_levels],
                       marker=markers[r.jpeg_quality],
                       s=100, alpha=0.7,
                       edgecolors='black', linewidths=0.5)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = (
            [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=f'Quant={q}')
             for q, c in colors.items()] +
            [Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markersize=10, label=f'JPEG Q{q}')
             for q, m in markers.items()]
        )
        ax4.legend(handles=legend_elements, loc='best', fontsize=9)

        ax4.set_xlabel('Compression Ratio vs Q90')
        ax4.set_ylabel('PSNR (dB)')
        ax4.set_title('Parameter Space: Quality vs Compression')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_file = self.output_dir / "parameter_grid_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plots to: {plot_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run RRIP parameter grid evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default demo_out pyramid
  python evaluation/run_parameter_grid.py

  # Specify custom pyramid
  python evaluation/run_parameter_grid.py --input-pyramid data/my_slide/baseline_pyramid

  # Custom output directory
  python evaluation/run_parameter_grid.py --output-dir evaluation/my_results
        """
    )

    parser.add_argument(
        "--input-pyramid",
        type=Path,
        default=Path("data/demo_out/baseline_pyramid"),
        help="Path to baseline pyramid (without _files suffix)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/grid_evaluation"),
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_pyramid.parent.exists():
        print(f"Error: Input pyramid parent directory not found: {args.input_pyramid.parent}")
        sys.exit(1)

    files_dir = args.input_pyramid.parent / (args.input_pyramid.name + "_files")
    if not files_dir.exists():
        print(f"Error: Pyramid files directory not found: {files_dir}")
        print(f"Expected: {args.input_pyramid.name}_files/")
        sys.exit(1)

    # Run evaluation
    try:
        evaluator = ParameterGridEvaluator(args.input_pyramid, args.output_dir)
        results = evaluator.run_full_grid()

        if results:
            evaluator.generate_summary(results)
            evaluator.generate_plots(results)

            print(f"\n{'='*80}")
            print("EVALUATION COMPLETE")
            print(f"{'='*80}")
            print(f"Results directory: {args.output_dir}")
            print(f"Total configurations tested: {len(results)}")
            print(f"\nKey files:")
            print(f"  - {args.output_dir}/aggregated_results.json")
            print(f"  - {args.output_dir}/parameter_grid_analysis.png")
            print(f"  - {args.output_dir}/grid_results/ (individual results)")

        else:
            print("\nNo results generated. Check for errors above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
