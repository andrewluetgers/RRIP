#!/usr/bin/env python3
"""
wsi_residual_analyze.py

Analysis tool that processes debug capture output from wsi_residual_debug_runner.py.
This script handles all the expensive analysis operations separately from capture:
- Calculate PSNR/SSIM/MSE metrics
- Generate detailed manifests
- Create summary reports
- Build PAC files for serving
- Generate paper-ready visualizations

Usage:
  # Basic analysis with metrics
  python wsi_residual_analyze.py --capture-dir debug_output

  # Full analysis with PAC file generation
  python wsi_residual_analyze.py --capture-dir debug_output --pac

  # Skip expensive metrics calculation
  python wsi_residual_analyze.py --capture-dir debug_output --no-metrics

  # Compare multiple captures
  python wsi_residual_analyze.py --compare debug_output1 debug_output2
"""

import argparse
import pathlib
import numpy as np
from PIL import Image
import json
from typing import Dict, Any, List, Optional
import os
from datetime import datetime

# Optional: Import metrics if available
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("Warning: scikit-image not installed. Quality metrics will not be calculated.")

# Try to import additional metrics
try:
    from skimage.color import deltaE_ciede2000, rgb2lab
    HAS_DELTA_E = True
except ImportError:
    HAS_DELTA_E = False
    print("Warning: Delta E metrics not available. Requires scikit-image >= 0.16")

# Try to import VIF
try:
    from scipy.signal import convolve2d
    from scipy.ndimage import gaussian_filter
    import scipy.special
    HAS_VIF = True
except ImportError:
    HAS_VIF = False
    print("Warning: VIF metric not available. Requires scipy.")


def calculate_vif(ref_image, dist_image):
    """
    Calculate Visual Information Fidelity (VIF) metric.
    Higher values indicate better quality (max = 1).
    """
    if not HAS_VIF:
        return None

    # Convert to grayscale if needed
    if ref_image.ndim == 3:
        ref_image = 0.299 * ref_image[:,:,0] + 0.587 * ref_image[:,:,1] + 0.114 * ref_image[:,:,2]
    if dist_image.ndim == 3:
        dist_image = 0.299 * dist_image[:,:,0] + 0.587 * dist_image[:,:,1] + 0.114 * dist_image[:,:,2]

    sigma_nsq = 2  # Noise variance
    eps = 1e-10

    # Apply Gaussian filter to get local statistics
    ref_mu = gaussian_filter(ref_image.astype(np.float64), 1.5, mode='reflect')
    dist_mu = gaussian_filter(dist_image.astype(np.float64), 1.5, mode='reflect')
    ref_sq = gaussian_filter(ref_image.astype(np.float64)**2, 1.5, mode='reflect')
    dist_sq = gaussian_filter(dist_image.astype(np.float64)**2, 1.5, mode='reflect')
    ref_dist = gaussian_filter(ref_image.astype(np.float64) * dist_image.astype(np.float64), 1.5, mode='reflect')

    ref_sigma_sq = np.maximum(ref_sq - ref_mu**2, 0)
    dist_sigma_sq = np.maximum(dist_sq - dist_mu**2, 0)
    ref_dist_sigma = ref_dist - ref_mu * dist_mu

    # Calculate VIF
    g = ref_dist_sigma / (ref_sigma_sq + eps)
    sv_sq = dist_sigma_sq - g * ref_dist_sigma

    g[ref_sigma_sq < eps] = 0
    sv_sq[ref_sigma_sq < eps] = dist_sigma_sq[ref_sigma_sq < eps]

    g[dist_sigma_sq < eps] = 0
    sv_sq[ref_sigma_sq < eps] = 0

    # Information fidelity
    num = np.sum(np.log10(1 + g**2 * ref_sigma_sq / (sv_sq + sigma_nsq)))
    den = np.sum(np.log10(1 + ref_sigma_sq / sigma_nsq))

    if den == 0:
        return 0.0

    return float(num / den)


def calculate_delta_e(ref_image, dist_image):
    """
    Calculate Delta E (CIE DE2000) color difference.
    Returns mean and percentile statistics.
    Lower values are better (0 = identical).
    """
    if not HAS_DELTA_E:
        return None

    # Ensure RGB images
    if ref_image.ndim != 3 or dist_image.ndim != 3:
        return None

    # Convert to Lab color space
    ref_lab = rgb2lab(ref_image / 255.0)
    dist_lab = rgb2lab(dist_image / 255.0)

    # Calculate Delta E for each pixel
    delta_e = deltaE_ciede2000(ref_lab, dist_lab)

    return {
        "mean": float(np.mean(delta_e)),
        "median": float(np.median(delta_e)),
        "p95": float(np.percentile(delta_e, 95)),
        "p99": float(np.percentile(delta_e, 99)),
        "max": float(np.max(delta_e))
    }


class DebugAnalyzer:
    """Analyzes debug capture data to generate metrics and manifests."""

    def __init__(self, capture_dir: pathlib.Path):
        self.capture_dir = pathlib.Path(capture_dir)
        self.arrays_dir = self.capture_dir / "arrays"
        self.images_dir = self.capture_dir / "images"
        self.residuals_dir = self.capture_dir / "residuals"

        # Load capture metadata
        metadata_path = self.capture_dir / "capture_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"No capture metadata found at {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.manifest = {
            "configuration": self.metadata.get("configuration", {}),
            "input": self.metadata.get("input", {}),
            "compression_phase": {},
            "decompression_phase": {},
            "statistics": self.metadata.get("statistics", {}),
            "metrics": {}
        }

    def load_array(self, name: str) -> Optional[np.ndarray]:
        """Load a saved numpy array by name."""
        array_path = self.arrays_dir / f"{name}.npy"
        if array_path.exists():
            return np.load(array_path)
        return None

    def calculate_metrics(self, name1: str, name2: str) -> Dict[str, Any]:
        """Calculate quality metrics between two saved arrays."""
        if not HAS_METRICS:
            return {"error": "scikit-image not installed"}

        arr1 = self.load_array(name1)
        arr2 = self.load_array(name2)

        if arr1 is None or arr2 is None:
            return {"error": f"Could not load arrays: {name1}, {name2}"}

        if arr1.shape != arr2.shape:
            return {"error": f"Shape mismatch: {arr1.shape} vs {arr2.shape}"}

        metrics = {}

        # Basic metrics (work on grayscale or RGB)
        if arr1.ndim == 3:
            # For RGB, calculate PSNR/SSIM on luminance
            arr1_gray = 0.299 * arr1[:,:,0] + 0.587 * arr1[:,:,1] + 0.114 * arr1[:,:,2]
            arr2_gray = 0.299 * arr2[:,:,0] + 0.587 * arr2[:,:,1] + 0.114 * arr2[:,:,2]
        else:
            arr1_gray = arr1
            arr2_gray = arr2

        metrics["psnr"] = float(psnr(arr1_gray, arr2_gray, data_range=255))
        metrics["ssim"] = float(ssim(arr1_gray, arr2_gray, data_range=255))
        metrics["mse"] = float(np.mean((arr1_gray.astype(np.float32) - arr2_gray.astype(np.float32)) ** 2))

        # VIF metric
        if HAS_VIF:
            vif_score = calculate_vif(arr1, arr2)
            if vif_score is not None:
                metrics["vif"] = vif_score

        # Delta E metric (only for RGB images)
        if HAS_DELTA_E and arr1.ndim == 3 and arr2.ndim == 3:
            delta_e_stats = calculate_delta_e(arr1, arr2)
            if delta_e_stats is not None:
                metrics["delta_e"] = delta_e_stats

        return metrics

    def analyze_compression_phase(self, calculate_metrics: bool = True):
        """Analyze the compression phase data."""
        print("Analyzing compression phase...")

        # Analyze L2 tiles
        for event in self.metadata.get("capture_events", []):
            if event["event"] == "l2_loaded":
                x2, y2 = event["x2"], event["y2"]
                tile_key = f"L2_{x2}_{y2}"

                tile_info = {
                    "type": "L2",
                    "coordinates": [x2, y2]
                }

                # Add array statistics
                for suffix in ["original", "luma", "chroma_cb", "chroma_cr"]:
                    arr_name = f"{tile_key}_{suffix}"
                    arr = self.load_array(arr_name)
                    if arr is not None:
                        tile_info[suffix] = {
                            "shape": arr.shape,
                            "range": [float(arr.min()), float(arr.max())],
                            "mean": float(arr.mean()),
                            "std": float(arr.std())
                        }

                self.manifest["compression_phase"][tile_key] = tile_info

        # Analyze L1 tiles and calculate metrics
        l1_tiles = set()
        for f in self.arrays_dir.glob("L1_*_*_original.npy"):
            parts = f.stem.split("_")
            if len(parts) >= 4:
                x1, y1 = parts[1], parts[2]
                l1_tiles.add((x1, y1))

        for x1, y1 in sorted(l1_tiles):
            tile_key = f"L1_{x1}_{y1}"
            tile_info = {
                "type": "L1",
                "coordinates": [x1, y1]
            }

            # Calculate metrics if requested
            if calculate_metrics and HAS_METRICS:
                # Original vs prediction
                metrics = self.calculate_metrics(
                    f"L1_{x1}_{y1}_original",
                    f"L1_{x1}_{y1}_prediction"
                )
                tile_info["prediction_metrics"] = metrics

                # Original vs reconstructed (if available)
                if self.load_array(f"L1_{x1}_{y1}_reconstructed") is not None:
                    metrics = self.calculate_metrics(
                        f"L1_{x1}_{y1}_original",
                        f"L1_{x1}_{y1}_reconstructed"
                    )
                    tile_info["reconstruction_metrics"] = metrics

            # Add residual statistics
            residual_raw = self.load_array(f"L1_{x1}_{y1}_residual_raw")
            if residual_raw is not None:
                tile_info["residual_raw"] = {
                    "range": [float(residual_raw.min()), float(residual_raw.max())],
                    "mean": float(residual_raw.mean()),
                    "std": float(residual_raw.std()),
                    "sparsity": float(np.mean(np.abs(residual_raw) < 1))  # Percentage near zero
                }

            self.manifest["compression_phase"][tile_key] = tile_info

        # Similar analysis for L0 tiles
        l0_tiles = set()
        for f in self.arrays_dir.glob("L0_*_*_original.npy"):
            parts = f.stem.split("_")
            if len(parts) >= 4:
                x0, y0 = parts[1], parts[2]
                l0_tiles.add((x0, y0))

        for x0, y0 in sorted(l0_tiles):
            tile_key = f"L0_{x0}_{y0}"
            tile_info = {
                "type": "L0",
                "coordinates": [x0, y0]
            }

            # Calculate metrics if requested
            if calculate_metrics and HAS_METRICS:
                # Original vs prediction
                metrics = self.calculate_metrics(
                    f"L0_{x0}_{y0}_original",
                    f"L0_{x0}_{y0}_prediction"
                )
                tile_info["prediction_metrics"] = metrics

            # Add residual statistics
            residual_raw = self.load_array(f"L0_{x0}_{y0}_residual_raw")
            if residual_raw is not None:
                tile_info["residual_raw"] = {
                    "range": [float(residual_raw.min()), float(residual_raw.max())],
                    "mean": float(residual_raw.mean()),
                    "std": float(residual_raw.std()),
                    "sparsity": float(np.mean(np.abs(residual_raw) < 1))
                }

            self.manifest["compression_phase"][tile_key] = tile_info

    def generate_summary(self) -> str:
        """Generate a human-readable summary of the analysis."""
        lines = ["ORIGAMI COMPRESSION ANALYSIS SUMMARY", "=" * 50, ""]

        # Configuration
        lines.append("Configuration:")
        for key, value in self.manifest["input"].items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Compression statistics
        if self.manifest.get("statistics"):
            lines.append("Compression Statistics:")
            stats = self.manifest["statistics"]
            for key, value in stats.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        # Metrics summary
        if self.manifest.get("compression_phase"):
            lines.append("Quality Metrics:")

            # Collect all metrics
            l1_psnrs = []
            l1_ssims = []
            l1_vifs = []
            l1_delta_es = []
            l0_psnrs = []
            l0_ssims = []
            l0_vifs = []
            l0_delta_es = []

            for key, info in self.manifest["compression_phase"].items():
                if "prediction_metrics" in info:
                    m = info["prediction_metrics"]
                    if key.startswith("L1_"):
                        if "psnr" in m:
                            l1_psnrs.append(m["psnr"])
                        if "ssim" in m:
                            l1_ssims.append(m["ssim"])
                        if "vif" in m:
                            l1_vifs.append(m["vif"])
                        if "delta_e" in m and isinstance(m["delta_e"], dict):
                            l1_delta_es.append(m["delta_e"]["mean"])
                    elif key.startswith("L0_"):
                        if "psnr" in m:
                            l0_psnrs.append(m["psnr"])
                        if "ssim" in m:
                            l0_ssims.append(m["ssim"])
                        if "vif" in m:
                            l0_vifs.append(m["vif"])
                        if "delta_e" in m and isinstance(m["delta_e"], dict):
                            l0_delta_es.append(m["delta_e"]["mean"])

            # L1 metrics
            if l1_psnrs:
                lines.append(f"  L1 Prediction PSNR: {np.mean(l1_psnrs):.2f} dB (avg)")
            if l1_ssims:
                lines.append(f"  L1 Prediction SSIM: {np.mean(l1_ssims):.4f} (avg)")
            if l1_vifs:
                lines.append(f"  L1 Prediction VIF: {np.mean(l1_vifs):.4f} (avg)")
            if l1_delta_es:
                lines.append(f"  L1 Delta E (mean): {np.mean(l1_delta_es):.2f} (avg)")

            # L0 metrics
            if l0_psnrs:
                lines.append(f"  L0 Prediction PSNR: {np.mean(l0_psnrs):.2f} dB (avg)")
            if l0_ssims:
                lines.append(f"  L0 Prediction SSIM: {np.mean(l0_ssims):.4f} (avg)")
            if l0_vifs:
                lines.append(f"  L0 Prediction VIF: {np.mean(l0_vifs):.4f} (avg)")
            if l0_delta_es:
                lines.append(f"  L0 Delta E (mean): {np.mean(l0_delta_es):.2f} (avg)")

            lines.append("")

        # Residual analysis
        lines.append("Residual Analysis:")
        l1_sparsities = []
        l0_sparsities = []

        for key, info in self.manifest["compression_phase"].items():
            if "residual_raw" in info:
                sparsity = info["residual_raw"]["sparsity"]
                if key.startswith("L1_"):
                    l1_sparsities.append(sparsity)
                elif key.startswith("L0_"):
                    l0_sparsities.append(sparsity)

        if l1_sparsities:
            lines.append(f"  L1 Residual Sparsity: {np.mean(l1_sparsities)*100:.1f}% near zero")
        if l0_sparsities:
            lines.append(f"  L0 Residual Sparsity: {np.mean(l0_sparsities)*100:.1f}% near zero")

        return "\n".join(lines)

    def create_pac_file(self):
        """Create a PAC file from the captured data for serving."""
        pac_path = self.capture_dir / "tiles.pac"

        print(f"Creating PAC file: {pac_path}")

        # Simplified PAC structure
        pac_data = {
            "version": 1,
            "metadata": self.metadata,
            "tiles": {}
        }

        # Add residual tiles
        residuals_dir = self.capture_dir / "residuals"
        if residuals_dir.exists():
            for jpeg_file in residuals_dir.glob("**/*.jpg"):
                rel_path = jpeg_file.relative_to(residuals_dir)
                with open(jpeg_file, 'rb') as f:
                    pac_data["tiles"][str(rel_path)] = f.read()

        # Save PAC file (simplified - in production would use proper compression)
        import pickle
        with open(pac_path, 'wb') as f:
            pickle.dump(pac_data, f)

        size_mb = pac_path.stat().st_size / (1024 * 1024)
        print(f"PAC file created: {size_mb:.2f} MB")
        return pac_path

    def save_manifest(self):
        """Save the analysis manifest."""
        manifest_path = self.capture_dir / "analysis_manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, indent=2))
        return manifest_path

    def save_summary(self):
        """Save the analysis summary."""
        summary = self.generate_summary()
        summary_path = self.capture_dir / "analysis_summary.txt"
        summary_path.write_text(summary)
        return summary_path


def compare_captures(capture_dirs: List[pathlib.Path]):
    """Compare multiple debug captures."""
    print(f"Comparing {len(capture_dirs)} captures...")

    comparisons = []
    for capture_dir in capture_dirs:
        analyzer = DebugAnalyzer(capture_dir)
        analyzer.analyze_compression_phase(calculate_metrics=True)

        # Extract key metrics
        stats = analyzer.manifest.get("statistics", {})
        comparison = {
            "capture": str(capture_dir),
            "configuration": analyzer.manifest.get("input", {}),
            "compression_ratio": stats.get("compression_ratio", 0),
            "savings_pct": stats.get("savings_pct", 0)
        }

        # Add average metrics
        l1_psnrs = []
        l0_psnrs = []
        l0_vifs = []
        l0_delta_es = []
        for key, info in analyzer.manifest["compression_phase"].items():
            if "prediction_metrics" in info:
                m = info["prediction_metrics"]
                if key.startswith("L1_"):
                    if "psnr" in m:
                        l1_psnrs.append(m["psnr"])
                elif key.startswith("L0_"):
                    if "psnr" in m:
                        l0_psnrs.append(m["psnr"])
                    if "vif" in m:
                        l0_vifs.append(m["vif"])
                    if "delta_e" in m and isinstance(m["delta_e"], dict):
                        l0_delta_es.append(m["delta_e"]["mean"])

        if l1_psnrs:
            comparison["l1_psnr_avg"] = np.mean(l1_psnrs)
        if l0_psnrs:
            comparison["l0_psnr_avg"] = np.mean(l0_psnrs)
        if l0_vifs:
            comparison["l0_vif_avg"] = np.mean(l0_vifs)
        if l0_delta_es:
            comparison["l0_delta_e_avg"] = np.mean(l0_delta_es)

        comparisons.append(comparison)

    # Print comparison table
    print("\nComparison Results:")
    print("-" * 100)
    print(f"{'Capture':<30} {'Q':<5} {'J':<5} {'Ratio':<8} {'L0 PSNR':<9} {'L0 VIF':<8} {'L0 ΔE':<8}")
    print("-" * 100)

    for comp in comparisons:
        capture_name = pathlib.Path(comp["capture"]).name
        if len(capture_name) > 29:
            capture_name = capture_name[:26] + "..."
        config = comp["configuration"]
        print(f"{capture_name:<30} "
              f"{config.get('quantization', 0):<5} "
              f"{config.get('jpeg_quality', 0):<5} "
              f"{comp.get('compression_ratio', 0):<8.2f} "
              f"{comp.get('l0_psnr_avg', 0):<9.2f} "
              f"{comp.get('l0_vif_avg', 0):<8.3f} "
              f"{comp.get('l0_delta_e_avg', 0):<8.2f}")

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Analyze ORIGAMI debug capture data")
    parser.add_argument("--capture-dir", help="Path to debug capture directory")
    parser.add_argument("--compare", nargs="+", help="Compare multiple captures")
    parser.add_argument("--no-metrics", action="store_true",
                       help="Skip expensive metrics calculation")
    parser.add_argument("--pac", action="store_true",
                       help="Create PAC file for serving")
    parser.add_argument("--output-format", choices=["json", "text", "both"],
                       default="both", help="Output format for analysis")

    args = parser.parse_args()

    if args.compare:
        # Compare mode
        capture_dirs = [pathlib.Path(d) for d in args.compare]
        comparisons = compare_captures(capture_dirs)

        # Save comparison results
        comparison_file = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparisons, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")

    elif args.capture_dir:
        # Single capture analysis
        capture_dir = pathlib.Path(args.capture_dir)
        if not capture_dir.exists():
            print(f"Error: Capture directory not found: {capture_dir}")
            return

        analyzer = DebugAnalyzer(capture_dir)

        # Run analysis
        print(f"Analyzing capture: {capture_dir}")
        analyzer.analyze_compression_phase(calculate_metrics=not args.no_metrics)

        # Save results
        if args.output_format in ["json", "both"]:
            manifest_path = analyzer.save_manifest()
            print(f"✓ Manifest saved: {manifest_path}")

        if args.output_format in ["text", "both"]:
            summary_path = analyzer.save_summary()
            print(f"✓ Summary saved: {summary_path}")

            # Print summary to console
            print("\n" + "=" * 50)
            print(analyzer.generate_summary())
            print("=" * 50)

        # Create PAC file if requested
        if args.pac:
            pac_path = analyzer.create_pac_file()
            print(f"✓ PAC file created: {pac_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()