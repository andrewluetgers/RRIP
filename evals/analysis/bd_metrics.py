"""
BD-Rate and BD-PSNR calculation for ORIGAMI compression analysis.

Based on Bjøntegaard Delta calculations for comparing rate-distortion curves.
Reference: G. Bjøntegaard, "Calculation of average PSNR differences between RD curves"
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class RDPoint:
    """Single point on a Rate-Distortion curve."""
    bitrate: float  # bits per pixel or total bytes
    psnr: float     # PSNR in dB
    quality: Optional[int] = None  # JPEG quality parameter if applicable


class BDMetrics:
    """Calculate Bjøntegaard Delta metrics between rate-distortion curves."""

    @staticmethod
    def piecewise_cubic_interpolation(x_points: np.ndarray, y_points: np.ndarray) -> interp.PchipInterpolator:
        """
        Create piecewise cubic hermite interpolating polynomial.
        This is more stable than regular cubic splines for RD curves.
        """
        # Sort points by x coordinate
        sorted_indices = np.argsort(x_points)
        x_sorted = x_points[sorted_indices]
        y_sorted = y_points[sorted_indices]

        # Remove duplicates
        unique_mask = np.append(True, np.diff(x_sorted) > 0)
        x_unique = x_sorted[unique_mask]
        y_unique = y_sorted[unique_mask]

        if len(x_unique) < 2:
            raise ValueError("Need at least 2 unique points for interpolation")

        return interp.PchipInterpolator(x_unique, y_unique)

    @staticmethod
    def integrate_interpolated(interpolator: interp.PchipInterpolator,
                              x_min: float, x_max: float,
                              num_samples: int = 1000) -> float:
        """Integrate an interpolated function using Simpson's rule."""
        x_samples = np.linspace(x_min, x_max, num_samples)
        y_samples = interpolator(x_samples)
        return np.trapz(y_samples, x_samples)

    @classmethod
    def calculate_bd_rate(cls,
                         reference_rates: List[float],
                         reference_psnrs: List[float],
                         test_rates: List[float],
                         test_psnrs: List[float]) -> float:
        """
        Calculate BD-Rate (Bjøntegaard Delta Rate).

        Args:
            reference_rates: Bitrates for reference codec (e.g., standard JPEG)
            reference_psnrs: PSNR values for reference codec
            test_rates: Bitrates for test codec (e.g., ORIGAMI)
            test_psnrs: PSNR values for test codec

        Returns:
            BD-Rate as percentage. Negative means test codec is better.
        """
        # Convert to numpy arrays and log domain for rates
        ref_rates = np.log10(np.array(reference_rates))
        ref_psnrs = np.array(reference_psnrs)
        test_rates = np.log10(np.array(test_rates))
        test_psnrs = np.array(test_psnrs)

        # Find common PSNR range
        psnr_min = max(ref_psnrs.min(), test_psnrs.min())
        psnr_max = min(ref_psnrs.max(), test_psnrs.max())

        if psnr_min >= psnr_max:
            raise ValueError("No overlapping PSNR range between curves")

        # Interpolate rate as function of PSNR
        ref_interp = cls.piecewise_cubic_interpolation(ref_psnrs, ref_rates)
        test_interp = cls.piecewise_cubic_interpolation(test_psnrs, test_rates)

        # Calculate average difference in log domain
        ref_integral = cls.integrate_interpolated(ref_interp, psnr_min, psnr_max)
        test_integral = cls.integrate_interpolated(test_interp, psnr_min, psnr_max)

        avg_diff_log = (test_integral - ref_integral) / (psnr_max - psnr_min)

        # Convert back from log domain to percentage
        bd_rate = (10**avg_diff_log - 1) * 100

        return bd_rate

    @classmethod
    def calculate_bd_psnr(cls,
                         reference_rates: List[float],
                         reference_psnrs: List[float],
                         test_rates: List[float],
                         test_psnrs: List[float]) -> float:
        """
        Calculate BD-PSNR (Bjøntegaard Delta PSNR).

        Args:
            reference_rates: Bitrates for reference codec
            reference_psnrs: PSNR values for reference codec
            test_rates: Bitrates for test codec
            test_psnrs: PSNR values for test codec

        Returns:
            BD-PSNR in dB. Positive means test codec is better.
        """
        # Convert to numpy arrays and log domain for rates
        ref_rates = np.log10(np.array(reference_rates))
        ref_psnrs = np.array(reference_psnrs)
        test_rates = np.log10(np.array(test_rates))
        test_psnrs = np.array(test_psnrs)

        # Find common rate range (in log domain)
        rate_min = max(ref_rates.min(), test_rates.min())
        rate_max = min(ref_rates.max(), test_rates.max())

        if rate_min >= rate_max:
            raise ValueError("No overlapping rate range between curves")

        # Interpolate PSNR as function of log(rate)
        ref_interp = cls.piecewise_cubic_interpolation(ref_rates, ref_psnrs)
        test_interp = cls.piecewise_cubic_interpolation(test_rates, test_psnrs)

        # Calculate average difference
        ref_integral = cls.integrate_interpolated(ref_interp, rate_min, rate_max)
        test_integral = cls.integrate_interpolated(test_interp, rate_min, rate_max)

        bd_psnr = (test_integral - ref_integral) / (rate_max - rate_min)

        return bd_psnr


class RDCurveAnalyzer:
    """Analyze and visualize Rate-Distortion curves for ORIGAMI vs baseline."""

    def __init__(self):
        self.reference_points: List[RDPoint] = []
        self.test_points: List[RDPoint] = []

    def add_reference_point(self, bitrate: float, psnr: float, quality: Optional[int] = None):
        """Add a point to the reference (baseline) RD curve."""
        self.reference_points.append(RDPoint(bitrate, psnr, quality))

    def add_test_point(self, bitrate: float, psnr: float, quality: Optional[int] = None):
        """Add a point to the test (ORIGAMI) RD curve."""
        self.test_points.append(RDPoint(bitrate, psnr, quality))

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate BD-Rate and BD-PSNR between the curves."""
        if len(self.reference_points) < 4 or len(self.test_points) < 4:
            raise ValueError("Need at least 4 points per curve for BD calculations")

        ref_rates = [p.bitrate for p in self.reference_points]
        ref_psnrs = [p.psnr for p in self.reference_points]
        test_rates = [p.bitrate for p in self.test_points]
        test_psnrs = [p.psnr for p in self.test_points]

        bd_rate = BDMetrics.calculate_bd_rate(ref_rates, ref_psnrs, test_rates, test_psnrs)
        bd_psnr = BDMetrics.calculate_bd_psnr(ref_rates, ref_psnrs, test_rates, test_psnrs)

        return {
            'bd_rate': bd_rate,
            'bd_psnr': bd_psnr
        }

    def plot_rd_curves(self, title: str = "Rate-Distortion Curves",
                      save_path: Optional[str] = None,
                      show_bd_metrics: bool = True) -> plt.Figure:
        """Plot RD curves with optional BD metrics annotation."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort points by bitrate
        ref_sorted = sorted(self.reference_points, key=lambda p: p.bitrate)
        test_sorted = sorted(self.test_points, key=lambda p: p.bitrate)

        # Extract data for plotting
        ref_rates = [p.bitrate for p in ref_sorted]
        ref_psnrs = [p.psnr for p in ref_sorted]
        test_rates = [p.bitrate for p in test_sorted]
        test_psnrs = [p.psnr for p in test_sorted]

        # Plot curves
        ax.plot(ref_rates, ref_psnrs, 'o-', label='Baseline (Standard JPEG)',
                linewidth=2, markersize=8, color='blue')
        ax.plot(test_rates, test_psnrs, 's-', label='ORIGAMI (Residual Pyramid)',
                linewidth=2, markersize=8, color='red')

        # Add quality labels if available
        for p in ref_sorted:
            if p.quality is not None:
                ax.annotate(f'Q{p.quality}', (p.bitrate, p.psnr),
                          textcoords="offset points", xytext=(5, 5),
                          fontsize=8, color='blue', alpha=0.7)

        for p in test_sorted:
            if p.quality is not None:
                ax.annotate(f'Q{p.quality}', (p.bitrate, p.psnr),
                          textcoords="offset points", xytext=(5, -10),
                          fontsize=8, color='red', alpha=0.7)

        # Add BD metrics if requested
        if show_bd_metrics and len(ref_sorted) >= 4 and len(test_sorted) >= 4:
            try:
                metrics = self.calculate_metrics()
                textstr = f'BD-Rate: {metrics["bd_rate"]:.1f}%\nBD-PSNR: {metrics["bd_psnr"]:.2f} dB'
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except Exception as e:
                print(f"Could not calculate BD metrics: {e}")

        ax.set_xlabel('Bitrate (bytes or bpp)', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Use log scale for bitrate if range is large
        if max(ref_rates + test_rates) / min(ref_rates + test_rates) > 10:
            ax.set_xscale('log')
            ax.set_xlabel('Bitrate (bytes or bpp) [log scale]', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig


def example_usage():
    """Example of how to use BD metrics for ORIGAMI analysis."""

    # Create analyzer
    analyzer = RDCurveAnalyzer()

    # Add baseline JPEG points (quality -> bitrate, PSNR)
    # These are example values - replace with actual measurements
    baseline_data = [
        (95, 1000000, 42.5),  # (quality, bytes, PSNR)
        (90, 750000, 40.8),
        (85, 600000, 39.2),
        (80, 500000, 37.8),
        (75, 420000, 36.5),
        (70, 350000, 35.2),
    ]

    for quality, bitrate, psnr in baseline_data:
        analyzer.add_reference_point(bitrate, psnr, quality)

    # Add ORIGAMI points (residual quality -> bitrate, PSNR)
    origami_data = [
        (64, 650000, 41.2),  # Higher quality residuals
        (48, 480000, 39.8),
        (32, 350000, 38.1),  # Default quality
        (24, 280000, 36.5),
        (16, 220000, 34.8),
        (8, 180000, 32.5),   # Lower quality residuals
    ]

    for quality, bitrate, psnr in origami_data:
        analyzer.add_test_point(bitrate, psnr, quality)

    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    print(f"BD-Rate: {metrics['bd_rate']:.2f}%")
    print(f"  Interpretation: ORIGAMI uses {abs(metrics['bd_rate']):.1f}% {'less' if metrics['bd_rate'] < 0 else 'more'} bitrate on average")
    print(f"BD-PSNR: {metrics['bd_psnr']:.2f} dB")
    print(f"  Interpretation: ORIGAMI provides {abs(metrics['bd_psnr']):.2f} dB {'better' if metrics['bd_psnr'] > 0 else 'worse'} quality on average")

    # Plot curves
    analyzer.plot_rd_curves(
        title="ORIGAMI vs Standard JPEG: Rate-Distortion Performance",
        save_path="rd_curves.png"
    )
    plt.show()


if __name__ == "__main__":
    example_usage()