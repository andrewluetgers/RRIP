#!/usr/bin/env python3
"""
Quick test of a single parameter configuration to verify the evaluation pipeline works.

This is useful for testing before running the full 3x3 grid.
"""

import sys
from pathlib import Path

# Add parent directory to path to import the evaluator
sys.path.insert(0, str(Path(__file__).parent))

from run_parameter_grid import ParameterGridEvaluator


def main():
    """Test a single configuration"""
    print("="*80)
    print("Testing Single Configuration")
    print("="*80)

    # Setup
    input_pyramid = Path("data/demo_out/baseline_pyramid")
    output_dir = Path("evaluation/test_single_config")

    # Create evaluator
    evaluator = ParameterGridEvaluator(input_pyramid, output_dir)

    # Test one configuration: quant=32, jpeg_q=60
    print("\nTesting: quant=32, jpeg_quality=60")
    try:
        result = evaluator.evaluate_configuration(quant=32, jpeg_q=60)

        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"Configuration: {result.config_name}")
        print(f"Total size: {result.total_size_mb:.2f} MB")
        print(f"Compression ratio: {result.compression_ratio_vs_q90:.2f}x")
        print(f"PSNR: {result.psnr_vs_q90:.2f} dB")
        print(f"SSIM: {result.ssim_vs_q90:.4f}")
        print(f"Processing time: {result.processing_time_seconds:.2f}s")
        print(f"\nResult saved to: {output_dir}/grid_results/quant32_jpeg60_result.json")

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print("FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
