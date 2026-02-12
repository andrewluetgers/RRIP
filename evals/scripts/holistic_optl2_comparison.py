#!/usr/bin/env python3
"""
holistic_optl2_comparison.py — Full-pipeline comparison of OptL2 approaches.

Runs the COMPLETE ORIGAMI compression pipeline for each optimization strategy
and measures what actually matters:
  - Total file size (L2 baseline + L1 residuals + L0 residuals)
  - L2 tile size (the cost of optimization)
  - L1 and L0 residual sizes (the benefit of optimization)
  - Reconstructed tile quality (L1 and L0 PSNR — Y, Cb, Cr, RGB)
  - Encode time (optimization + compression)

Strategies compared:
  A. No optimization (standard ORIGAMI)
  B. Y-only OptL2 (optimize luma channel only)
  C. RGB open-loop OptL2 (optimize all channels, current approach)
  D. RGB JPEG-in-loop OptL2 (optimize all channels with JPEG round-trip)

Each strategy is tested across residual quality levels (resq) to show
rate-distortion tradeoffs. L2 baseline quality is fixed at 95.
"""

import argparse
import io
import time
import json
import numpy as np
from PIL import Image
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from wsi_residual_debug_with_manifest import (
    tile_image, compress_with_debug, decompress_with_debug,
    rgb_to_ycbcr_bt601, ycbcr_to_rgb_bt601, psnr,
)
from optimize_downsample import optimize_for_upsample, optimize_for_upsample_grayscale
from jpeg_encoder import JpegEncoder


def jpeg_roundtrip(rgb_uint8: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode then decode an RGB image."""
    img = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def optimize_yonly(l2_rgb, l1_mosaic, max_delta=15, n_iterations=100, lr=0.3):
    """Y-only optimization: optimize luma, leave chroma unchanged."""
    tile_size_h, tile_size_w = l1_mosaic.shape[:2]
    source_h, source_w = l2_rgb.shape[:2]

    Y_src, Cb_src, Cr_src = rgb_to_ycbcr_bt601(l2_rgb)
    Y_tgt, _, _ = rgb_to_ycbcr_bt601(l1_mosaic)

    Y_opt = optimize_for_upsample_grayscale(
        Y_src, Y_tgt, max_delta=max_delta, n_iterations=n_iterations, lr=lr
    )

    Y_opt_u8 = np.clip(np.round(Y_opt), 0, 255).astype(np.uint8)
    Cb_u8 = np.clip(np.round(Cb_src), 0, 255).astype(np.uint8)
    Cr_u8 = np.clip(np.round(Cr_src), 0, 255).astype(np.uint8)

    return ycbcr_to_rgb_bt601(Y_opt_u8.astype(np.float32),
                               Cb_u8.astype(np.float32),
                               Cr_u8.astype(np.float32))


def optimize_rgb_open(l2_rgb, l1_mosaic, max_delta=15, n_iterations=100, lr=0.3):
    """RGB open-loop optimization (current approach)."""
    return optimize_for_upsample(
        l2_rgb, l1_mosaic, max_delta=max_delta, n_iterations=n_iterations, lr=lr
    )


def optimize_rgb_jpeg_loop(l2_rgb, l1_mosaic, jpeg_quality, max_delta=15,
                           n_iterations=100, lr=0.3):
    """RGB JPEG-in-the-loop optimization."""
    target_h, target_w = l1_mosaic.shape[:2]
    source_h, source_w = l2_rgb.shape[:2]

    target_f = l1_mosaic.astype(np.float64)
    source_f = l2_rgb.astype(np.float64)
    source_orig = source_f.copy()
    best_energy = float("inf")
    best_source = source_f.copy()

    for _ in range(n_iterations):
        cur = np.clip(source_f, 0, 255).astype(np.uint8)
        jpeg_decoded = jpeg_roundtrip(cur, jpeg_quality)

        pred = np.array(
            Image.fromarray(jpeg_decoded).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
        ).astype(np.float64)

        residual = target_f - pred
        energy = np.sum(residual * residual)

        if energy < best_energy:
            best_energy = energy
            best_source = source_f.copy()

        grad = np.empty_like(source_f)
        for c in range(source_f.shape[2]):
            grad[:, :, c] = np.array(
                Image.fromarray(residual[:, :, c].astype(np.float32)).resize(
                    (source_w, source_h), Image.Resampling.BILINEAR
                )
            )

        source_f += lr * grad
        np.clip(source_f, source_orig - max_delta, source_orig + max_delta, out=source_f)
        np.clip(source_f, 0, 255, out=source_f)

    return np.clip(best_source, 0, 255).astype(np.uint8)


def run_pipeline(l2_tile, l1_tiles, l0_tiles, tile_size, resq, baseq, out_dir, encoder):
    """Run the full ORIGAMI pipeline and extract metrics from the manifest."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = compress_with_debug(
        l2_tile, l1_tiles, l0_tiles, out_dir,
        tile_size, resq, baseq, encoder,
    )
    manifest = decompress_with_debug(out_dir, manifest, tile_size)

    sc = manifest["size_comparison"]

    # Extract L1 reconstructed PSNR values
    l1_psnrs = []
    for tile_key, tile_data in manifest["decompression_phase"]["L1"].items():
        if tile_key.startswith("tile_") and "final_psnr" in tile_data:
            l1_psnrs.append(tile_data["final_psnr"])

    # Extract L0 reconstructed PSNR values
    l0_psnrs = []
    for tile_key, tile_data in manifest["decompression_phase"]["L0"].items():
        if tile_key.startswith("tile_") and "final_psnr" in tile_data:
            l0_psnrs.append(tile_data["final_psnr"])

    return {
        "l2_size": sc["origami_L2_baseline"],
        "l1_residual_size": sc["origami_L1_residuals"],
        "l0_residual_size": sc["origami_L0_residuals"],
        "total_size": sc["origami_total"],
        "baseline_total": sc["baseline_total"],
        "compression_ratio": sc["overall_compression_ratio"],
        "savings_pct": sc["overall_space_savings_pct"],
        "l1_avg_psnr": float(np.mean(l1_psnrs)) if l1_psnrs else 0,
        "l0_avg_psnr": float(np.mean(l0_psnrs)) if l0_psnrs else 0,
    }


def measure_per_channel_psnr(l2_tile, l1_tiles, l0_tiles, tile_size, baseq):
    """Measure per-channel prediction quality through the real JPEG pipeline.

    This simulates what the decoder sees: JPEG-encode L2, decode, upsample,
    then measure Y/Cb/Cr PSNR of the prediction vs ground truth.
    """
    # JPEG round-trip the L2 tile (what the decoder actually gets)
    l2_decoded = jpeg_roundtrip(l2_tile, baseq)

    # Build L1 ground truth mosaic
    l1_mosaic = np.zeros((tile_size * 2, tile_size * 2, 3), dtype=np.uint8)
    for (dx, dy), l1_gt in l1_tiles.items():
        l1_mosaic[dy * tile_size:(dy + 1) * tile_size,
                  dx * tile_size:(dx + 1) * tile_size] = l1_gt

    # Upsample prediction
    pred = np.array(Image.fromarray(l2_decoded).resize(
        (tile_size * 2, tile_size * 2), Image.Resampling.BILINEAR))

    # Per-channel PSNR
    tY, tCb, tCr = rgb_to_ycbcr_bt601(l1_mosaic)
    pY, pCb, pCr = rgb_to_ycbcr_bt601(pred)

    return {
        "pred_psnr_y": float(psnr(tY, pY, data_range=255)),
        "pred_psnr_cb": float(psnr(tCb, pCb, data_range=255)),
        "pred_psnr_cr": float(psnr(tCr, pCr, data_range=255)),
        "pred_psnr_rgb": float(psnr(l1_mosaic, pred, data_range=255)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Holistic OptL2 comparison across full ORIGAMI pipeline")
    parser.add_argument("--image", required=True, help="1024x1024 test image")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--baseq", type=int, default=95, help="L2 baseline JPEG quality")
    parser.add_argument("--max-delta", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--resq-list", type=str, default="30,50,70,80,90",
                        help="Comma-separated residual quality levels")
    args = parser.parse_args()

    resq_list = [int(q) for q in args.resq_list.split(",")]
    encoder = JpegEncoder.LIBJPEG_TURBO
    base_out = pathlib.Path("evals/runs/_holistic_comparison")

    print(f"Image: {args.image}")
    print(f"L2 baseline quality: {args.baseq}")
    print(f"Optimization: max_delta=±{args.max_delta}, iterations={args.iterations}")
    print(f"Residual qualities: {resq_list}")
    print()

    # Tile the image (once)
    l2_tile, l1_tiles, l0_tiles = tile_image(args.image, args.tile)

    # Build L1 mosaic for optimization target
    tile_size = args.tile
    l1_mosaic = np.zeros((tile_size * 2, tile_size * 2, 3), dtype=np.uint8)
    for (dx, dy), l1_gt in l1_tiles.items():
        l1_mosaic[dy * tile_size:(dy + 1) * tile_size,
                  dx * tile_size:(dx + 1) * tile_size] = l1_gt

    # ---- Pre-compute all optimized L2 variants ----
    strategies = {}

    print("=== Optimizing L2 tiles ===")

    # A. No optimization
    print("  [A] No optimization")
    strategies["none"] = {"l2": l2_tile, "opt_time": 0.0, "label": "No optimization"}

    # B. Y-only
    print("  [B] Y-only optimization...")
    t0 = time.time()
    l2_yonly = optimize_yonly(l2_tile, l1_mosaic,
                             max_delta=args.max_delta, n_iterations=args.iterations,
                             lr=args.lr)
    t_yonly = time.time() - t0
    strategies["yonly"] = {"l2": l2_yonly, "opt_time": t_yonly, "label": "Y-only OptL2"}
    print(f"      {t_yonly:.1f}s")

    # C. RGB open-loop
    print("  [C] RGB open-loop optimization...")
    t0 = time.time()
    l2_rgb_open = optimize_rgb_open(l2_tile, l1_mosaic,
                                    max_delta=args.max_delta, n_iterations=args.iterations,
                                    lr=args.lr)
    t_rgb_open = time.time() - t0
    strategies["rgb_open"] = {"l2": l2_rgb_open, "opt_time": t_rgb_open,
                              "label": "RGB open-loop"}
    print(f"      {t_rgb_open:.1f}s")

    # D. RGB JPEG-in-loop
    print(f"  [D] RGB JPEG-in-loop (q={args.baseq}) optimization...")
    t0 = time.time()
    l2_rgb_jloop = optimize_rgb_jpeg_loop(l2_tile, l1_mosaic, jpeg_quality=args.baseq,
                                          max_delta=args.max_delta, n_iterations=args.iterations,
                                          lr=args.lr)
    t_rgb_jloop = time.time() - t0
    strategies["rgb_jloop"] = {"l2": l2_rgb_jloop, "opt_time": t_rgb_jloop,
                               "label": "RGB JPEG-in-loop"}
    print(f"      {t_rgb_jloop:.1f}s")

    # ---- Measure L1 prediction quality per-channel (through JPEG) ----
    print("\n=== L1 Prediction Quality (through L2 JPEG @ q={}) ===".format(args.baseq))
    print(f"{'Strategy':<22}  {'Y':>8}  {'Cb':>8}  {'Cr':>8}  {'RGB':>8}")
    print("-" * 62)
    for key, strat in strategies.items():
        pq = measure_per_channel_psnr(strat["l2"], l1_tiles, l0_tiles, tile_size, args.baseq)
        strat["pred_quality"] = pq
        print(f"{strat['label']:<22}  {pq['pred_psnr_y']:8.2f}  {pq['pred_psnr_cb']:8.2f}  "
              f"{pq['pred_psnr_cr']:8.2f}  {pq['pred_psnr_rgb']:8.2f}")

    # ---- Run full pipeline for each strategy × residual quality ----
    all_results = []

    print(f"\n=== Full Pipeline Results ===")
    header = (f"{'ResQ':>4}  {'Strategy':<22}  {'OptTime':>7}  {'L2 KB':>7}  "
              f"{'L1res KB':>8}  {'L0res KB':>8}  {'Total KB':>8}  {'Base KB':>8}  "
              f"{'Ratio':>6}  {'L1 PSNR':>8}  {'L0 PSNR':>8}")
    print(header)
    print("-" * len(header))

    for resq in resq_list:
        for key, strat in strategies.items():
            out_dir = str(base_out / f"{key}_resq{resq}")

            # Suppress print output from the pipeline
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                t0 = time.time()
                metrics = run_pipeline(strat["l2"], l1_tiles, l0_tiles,
                                       tile_size, resq, args.baseq, out_dir, encoder)
                t_pipeline = time.time() - t0

            total_time = strat["opt_time"] + t_pipeline

            row = {
                "resq": resq,
                "strategy": key,
                "label": strat["label"],
                "opt_time": strat["opt_time"],
                "pipeline_time": t_pipeline,
                "total_time": total_time,
                **metrics,
                **strat.get("pred_quality", {}),
            }
            all_results.append(row)

            print(f"{resq:4d}  {strat['label']:<22}  {total_time:6.1f}s  "
                  f"{metrics['l2_size']/1024:7.1f}  {metrics['l1_residual_size']/1024:8.1f}  "
                  f"{metrics['l0_residual_size']/1024:8.1f}  {metrics['total_size']/1024:8.1f}  "
                  f"{metrics['baseline_total']/1024:8.1f}  {metrics['compression_ratio']:6.2f}  "
                  f"{metrics['l1_avg_psnr']:8.2f}  {metrics['l0_avg_psnr']:8.2f}")

        print()  # blank line between quality groups

    # ---- Summary: deltas vs no-optimization baseline ----
    print("\n=== DELTAS vs No Optimization ===")
    print(f"{'ResQ':>4}  {'Strategy':<22}  {'dL2 KB':>7}  {'dL1res':>7}  "
          f"{'dL0res':>7}  {'dTotal':>7}  {'dL1 PSNR':>9}  {'dL0 PSNR':>9}")
    print("-" * 90)

    for resq in resq_list:
        baseline = next(r for r in all_results if r["resq"] == resq and r["strategy"] == "none")
        for row in all_results:
            if row["resq"] != resq or row["strategy"] == "none":
                continue
            dl2 = (row["l2_size"] - baseline["l2_size"]) / 1024
            dl1 = (row["l1_residual_size"] - baseline["l1_residual_size"]) / 1024
            dl0 = (row["l0_residual_size"] - baseline["l0_residual_size"]) / 1024
            dt = (row["total_size"] - baseline["total_size"]) / 1024
            dp_l1 = row["l1_avg_psnr"] - baseline["l1_avg_psnr"]
            dp_l0 = row["l0_avg_psnr"] - baseline["l0_avg_psnr"]
            print(f"{resq:4d}  {row['label']:<22}  {dl2:+7.1f}  {dl1:+7.1f}  "
                  f"{dl0:+7.1f}  {dt:+7.1f}  {dp_l1:+9.2f}  {dp_l0:+9.2f}")
        print()

    # Save raw results
    results_path = base_out / "results.json"
    base_out.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nRaw results saved to: {results_path}")


if __name__ == "__main__":
    main()
