#!/usr/bin/env python3
"""
optl2_compress.py

Wrapper around wsi_residual_debug_with_manifest.py that first optimizes the L2
tile to minimize prediction error, then runs the standard compression pipeline.

The optimized L2 produces better bilinear predictions → smaller residuals →
better compression at the same quality, or same compression at better quality.

Decode is UNCHANGED — the decoder just sees an L2 tile and residuals.

Output directories use 'optl2_' prefix so the viewer can identify them.
"""
import argparse
import numpy as np
from PIL import Image
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from jpeg_encoder import JpegEncoder, parse_encoder_arg, is_jxl_encoder, is_webp_encoder
from optimize_downsample import optimize_for_upsample
from wsi_residual_debug_with_manifest import (
    tile_image, compress_with_debug, decompress_with_debug,
    create_pac_file, rgb_to_ycbcr_bt601, ycbcr_to_rgb_bt601,
    psnr, get_file_size,
)
import json
import os


def optimize_l2_for_prediction(l2_rgb, l1_tiles, max_delta=15, n_iterations=500, lr=0.3):
    """
    Optimize L2 tile pixels (all RGB channels) so bilinear upsampling best
    predicts L1 tiles. Uses the generic optimize_for_upsample function.
    """
    tile_size = 256

    # Build L1 target mosaic (512x512 RGB)
    l1_mosaic = np.zeros((tile_size*2, tile_size*2, 3), dtype=np.uint8)
    for (dx, dy), l1_gt in l1_tiles.items():
        l1_mosaic[dy*tile_size:(dy+1)*tile_size,
                  dx*tile_size:(dx+1)*tile_size] = l1_gt

    # Optimize
    l2_opt = optimize_for_upsample(
        l2_rgb, l1_mosaic,
        max_delta=max_delta, n_iterations=n_iterations, lr=lr
    )

    # Report
    orig_pred = np.array(Image.fromarray(l2_rgb).resize(
        (tile_size*2, tile_size*2), resample=Image.Resampling.BILINEAR))
    opt_pred = np.array(Image.fromarray(l2_opt).resize(
        (tile_size*2, tile_size*2), resample=Image.Resampling.BILINEAR))
    orig_energy = np.sum((l1_mosaic.astype(np.float64) - orig_pred.astype(np.float64))**2)
    opt_energy = np.sum((l1_mosaic.astype(np.float64) - opt_pred.astype(np.float64))**2)

    reduction = 1.0 - opt_energy / orig_energy
    l2_change_psnr = psnr(l2_rgb, l2_opt, data_range=255)
    l2_diff = np.abs(l2_opt.astype(np.float64) - l2_rgb.astype(np.float64))
    actual_mean_delta = np.mean(l2_diff)

    print(f"  L2 optimization: energy reduction={reduction:.1%}, "
          f"L2 PSNR vs original={l2_change_psnr:.1f}dB, "
          f"mean pixel change={actual_mean_delta:.1f}")

    return l2_opt


def main():
    parser = argparse.ArgumentParser(description="Optimized-L2 ORIGAMI compression")
    parser.add_argument("--image", required=True, help="Path to input image (1024x1024)")
    parser.add_argument("--out", help="Output directory (auto-generated if not specified)")
    parser.add_argument("--tile", type=int, default=256, help="Tile size")
    parser.add_argument("--resq", type=int, default=75, help="JPEG quality for residuals")
    parser.add_argument("--l1q", type=int, default=None, help="Override L1 residual quality")
    parser.add_argument("--l0q", type=int, default=None, help="Override L0 residual quality")
    parser.add_argument("--baseq", type=int, default=95, help="JPEG quality for baseline tiles")
    parser.add_argument("--pac", action="store_true", help="Create PAC file")
    parser.add_argument("--encoder", default="libjpeg-turbo", help="Encoder")
    parser.add_argument("--max-delta", type=int, default=15, help="Max L2 pixel perturbation")
    parser.add_argument("--iterations", type=int, default=500, help="Optimization iterations")

    args = parser.parse_args()
    encoder = parse_encoder_arg(args.encoder)

    # Generate output directory name
    if args.out is None:
        if encoder == JpegEncoder.LIBJPEG_TURBO:
            prefix = "optl2_"
        else:
            prefix = f"optl2_{encoder.value}_"

        if args.l1q is not None or args.l0q is not None:
            l1q = args.l1q if args.l1q is not None else args.resq
            l0q = args.l0q if args.l0q is not None else args.resq
            dir_parts = [f"{prefix}debug", f"l1q{l1q}", f"l0q{l0q}"]
        else:
            dir_parts = [f"{prefix}debug", f"j{args.resq}"]
        if args.pac:
            dir_parts.append("pac")
        args.out = "evals/runs/" + "_".join(dir_parts)

    print(f"Output directory: {args.out}")
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    # Tile the input image
    print(f"Processing image: {args.image}")
    l2_tile, l1_tiles, l0_tiles = tile_image(args.image, args.tile)

    # OPTIMIZE L2
    print(f"\nOptimizing L2 (max_delta=±{args.max_delta}, iterations={args.iterations})...")
    l2_optimized = optimize_l2_for_prediction(
        l2_tile, l1_tiles,
        max_delta=args.max_delta,
        n_iterations=args.iterations,
        lr=0.3
    )

    # Run standard pipeline with optimized L2
    print(f"\nRunning compression pipeline with optimized L2...")
    manifest = compress_with_debug(
        l2_optimized, l1_tiles, l0_tiles, args.out,
        args.tile, args.resq, args.baseq, encoder,
        l1_quality=args.l1q, l0_quality=args.l0q
    )

    # Decompress
    manifest = decompress_with_debug(args.out, manifest, args.tile)

    # PAC file
    if args.pac:
        pac_path, pac_size = create_pac_file(
            args.out, l2_optimized, l1_tiles, l0_tiles,
            args.tile, args.resq, args.baseq, encoder,
            l1_quality=args.l1q, l0_quality=args.l0q
        )
        manifest["pac_file"] = {"path": str(pac_path), "size": pac_size}

    # Add metadata
    manifest["input_image"] = str(args.image)
    manifest["l2_optimization"] = {
        "enabled": True,
        "max_delta": args.max_delta,
        "iterations": args.iterations,
    }

    # Save manifest
    manifest_path = pathlib.Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write summary
    summary_path = pathlib.Path(args.out) / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ORIGAMI COMPRESSION DEBUG SUMMARY (Optimized L2)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input Image: {args.image}\n")
        f.write(f"Tile Size: {args.tile}x{args.tile}\n")
        f.write(f"L2 Optimization: max_delta=±{args.max_delta}, iters={args.iterations}\n")
        l1q_eff = args.l1q if args.l1q is not None else args.resq
        l0q_eff = args.l0q if args.l0q is not None else args.resq
        if l1q_eff != l0q_eff:
            f.write(f"L1 Residual Quality: {l1q_eff}\n")
            f.write(f"L0 Residual Quality: {l0q_eff}\n")
        else:
            f.write(f"Residual JPEG Quality: {args.resq}\n")
        f.write(f"Baseline JPEG Quality: {args.baseq}\n\n")

        sc = manifest["size_comparison"]
        f.write("SIZE COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline Total: {sc['baseline_total']:,} bytes\n")
        f.write(f"  - L2: {sc['baseline_L2']:,} bytes\n")
        f.write(f"  - L1: {sc['baseline_L1_total']:,} bytes\n")
        f.write(f"  - L0: {sc['baseline_L0_total']:,} bytes\n\n")
        f.write(f"ORIGAMI Total: {sc['origami_total']:,} bytes\n")
        f.write(f"  - L2 (baseline): {sc['origami_L2_baseline']:,} bytes\n")
        f.write(f"  - L1 residuals: {sc['origami_L1_residuals']:,} bytes\n")
        f.write(f"  - L0 residuals: {sc['origami_L0_residuals']:,} bytes\n\n")
        f.write(f"Overall Compression Ratio: {sc['overall_compression_ratio']:.2f}x\n")
        f.write(f"Overall Space Savings: {sc['overall_space_savings_pct']:.1f}%\n")
        f.write(f"L1 Compression Ratio: {sc['L1_compression_ratio']:.2f}x\n")
        f.write(f"L0 Compression Ratio: {sc['L0_compression_ratio']:.2f}x\n\n")

        # Average quality metrics
        f.write("AVERAGE QUALITY METRICS\n")
        f.write("-" * 30 + "\n")

        l1_psnrs = []
        for tile_key in manifest["decompression_phase"]["L1"]:
            if tile_key.startswith("tile_"):
                tile_data = manifest["decompression_phase"]["L1"][tile_key]
                if "final_psnr" in tile_data:
                    l1_psnrs.append(tile_data["final_psnr"])
        if l1_psnrs:
            f.write(f"L1 Average PSNR: {np.mean(l1_psnrs):.2f} dB\n")

        l0_psnrs = []
        for tile_key in manifest["decompression_phase"]["L0"]:
            if tile_key.startswith("tile_"):
                tile_data = manifest["decompression_phase"]["L0"][tile_key]
                if "final_psnr" in tile_data:
                    l0_psnrs.append(tile_data["final_psnr"])
        if l0_psnrs:
            f.write(f"L0 Average PSNR: {np.mean(l0_psnrs):.2f} dB\n")

    print(f"\nDone! Output: {args.out}")
    print(f"Baseline: {sc['baseline_total']:,} bytes → ORIGAMI: {sc['origami_total']:,} bytes")
    print(f"Compression: {sc['overall_compression_ratio']:.2f}x, Savings: {sc['overall_space_savings_pct']:.1f}%")


if __name__ == "__main__":
    main()
