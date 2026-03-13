#!/usr/bin/env python3
"""Compute H&E stain fidelity metrics for image compression evaluation.

Separates images into Hematoxylin and Eosin optical density channels using
Ruifrok-Johnston deconvolution, then computes per-channel error metrics.

Metrics produced per stain channel (H and E):
  - PSNR: Peak signal-to-noise ratio on the OD map
  - MSE: Mean squared error in OD units
  - MAE: Mean absolute error in OD units (more interpretable)
  - P95_AE: 95th percentile absolute error (worst 5% of pixels)
  - P99_AE: 99th percentile absolute error (worst 1%)
  - MAX_AE: Maximum absolute error (single worst pixel)

Usage:
    # Compare two images directly
    python evals/scripts/stain_metrics.py --ref original.jpg --test compressed.jpg

    # Compute stain metrics for an ORIGAMI run (updates manifest.json)
    python evals/scripts/stain_metrics.py --run evals/runs/v2_b90_l0q75_ss256_nooptl2

    # Compute for multiple runs
    python evals/scripts/stain_metrics.py --run evals/runs/v2_b90_l0q75_ss256_nooptl2 evals/runs/v2_b90_l0q70_ss256_nooptl2

    # Compute for a JXL file (decodes, slices into tiles, compares to source)
    python evals/scripts/stain_metrics.py --jxl /tmp/jxl_full_q76.jxl --source evals/test-images/L0-1024.jpg --serve-q 90
"""
import argparse
import json
import os
import sys
import io
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import separate_stains, hed_from_rgb


def stain_separate(img_np):
    """Separate RGB image into H, E, DAB optical density channels.

    Returns (H_od, E_od, DAB_od) as float64 arrays, clipped to [0, max].
    Negative OD values (artifacts) are clipped to 0.
    """
    stains = separate_stains(img_np, hed_from_rgb)
    h_od = np.clip(stains[:, :, 0], 0, None)
    e_od = np.clip(stains[:, :, 1], 0, None)
    dab_od = np.clip(stains[:, :, 2], 0, None)
    return h_od, e_od, dab_od


def compute_stain_metrics(ref_img, test_img):
    """Compute stain fidelity metrics between reference and test images.

    Both inputs are numpy arrays (H, W, 3) in uint8 RGB.
    Returns dict with per-channel metrics.
    """
    ref_h, ref_e, ref_dab = stain_separate(ref_img)
    test_h, test_e, test_dab = stain_separate(test_img)

    results = {}
    for name, ref_ch, test_ch in [('H', ref_h, test_h), ('E', ref_e, test_e)]:
        diff = test_ch - ref_ch
        abs_diff = np.abs(diff)

        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(abs_diff))

        # PSNR using the max OD value in the reference as the signal range
        max_od = float(ref_ch.max())
        if max_od > 0 and mse > 0:
            psnr = 10 * np.log10(max_od ** 2 / mse)
        elif mse == 0:
            psnr = float('inf')
        else:
            psnr = 0.0

        results[f'{name}_psnr'] = round(psnr, 2)
        results[f'{name}_mse'] = round(mse, 6)
        results[f'{name}_mae'] = round(mae, 6)
        results[f'{name}_p50_ae'] = round(float(np.percentile(abs_diff, 50)), 6)
        results[f'{name}_p95_ae'] = round(float(np.percentile(abs_diff, 95)), 6)
        results[f'{name}_p99_ae'] = round(float(np.percentile(abs_diff, 99)), 6)
        results[f'{name}_max_ae'] = round(float(abs_diff.max()), 6)

        # Signed stats — is compression shifting stain up or down?
        results[f'{name}_mean_error'] = round(float(np.mean(diff)), 6)
        results[f'{name}_std_error'] = round(float(np.std(diff)), 6)

    return results


def jpeg_roundtrip(tile_np, quality):
    """JPEG encode+decode roundtrip."""
    img = Image.fromarray(tile_np)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))


def process_run(run_dir, source_path=None):
    """Compute stain metrics for an ORIGAMI run directory.

    Reads manifest.json to find source image, then compares
    decompressed tiles against source tiles.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.exists():
        print(f"  No manifest.json in {run_dir}")
        return None

    manifest = json.loads(manifest_path.read_text())
    dp = manifest.get('decompression_phase', {})
    l0 = dp.get('L0', {})

    if not l0:
        print(f"  No decompression_phase/L0 in {run_dir}")
        return None

    # Find source image
    src_path = source_path or manifest.get('source')
    if not src_path or not os.path.exists(src_path):
        print(f"  Source image not found: {src_path}")
        return None

    src = np.array(Image.open(src_path))
    tile_size = manifest.get('tile_size', 256)

    # Compute stain metrics per L0 tile
    h, w = src.shape[:2]
    all_tile_metrics = {}

    for tile_key, tile_data in sorted(l0.items()):
        # Parse tile coordinates from key like "tile_0_0"
        parts = tile_key.replace('tile_', '').split('_')
        tx, ty = int(parts[0]), int(parts[1])

        # Extract source tile
        y0, x0 = ty * tile_size, tx * tile_size
        y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
        src_tile = src[y0:y1, x0:x1]

        # Find the reconstructed tile
        # compute_metrics.py saves tiles in decompress/ but we deleted those
        # Instead, reconstruct from the manifest's compress/decompress paths
        # Actually, we need the reconstructed tile image
        decompress_dir = run_dir / 'decompress'
        tile_path = decompress_dir / f'L0_{tx}_{ty}.png'
        if not tile_path.exists():
            # Try alternate naming
            tile_path = decompress_dir / f'tile_{tx}_{ty}.png'
        if not tile_path.exists():
            continue

        test_tile = np.array(Image.open(tile_path))
        if src_tile.shape != test_tile.shape:
            continue

        metrics = compute_stain_metrics(src_tile, test_tile)
        all_tile_metrics[tile_key] = metrics

    if not all_tile_metrics:
        return None

    # Average across tiles
    avg_metrics = {}
    keys = list(list(all_tile_metrics.values())[0].keys())
    for k in keys:
        vals = [m[k] for m in all_tile_metrics.values()]
        avg_metrics[k] = round(sum(vals) / len(vals), 6)

    return avg_metrics, all_tile_metrics


def process_image_pair(ref_path, test_path, tile_size=256):
    """Compare two full images by tiling and computing stain metrics."""
    ref = np.array(Image.open(ref_path))
    test = np.array(Image.open(test_path))

    if ref.shape != test.shape:
        print(f"Shape mismatch: {ref.shape} vs {test.shape}")
        return None

    # Full image metrics
    full_metrics = compute_stain_metrics(ref, test)

    # Per-tile metrics
    h, w = ref.shape[:2]
    tile_metrics = {}
    for ty in range(h // tile_size):
        for tx in range(w // tile_size):
            y0, x0 = ty * tile_size, tx * tile_size
            ref_tile = ref[y0:y0+tile_size, x0:x0+tile_size]
            test_tile = test[y0:y0+tile_size, x0:x0+tile_size]
            tile_metrics[f'tile_{tx}_{ty}'] = compute_stain_metrics(ref_tile, test_tile)

    return full_metrics, tile_metrics


def process_jxl(jxl_path, source_path, serve_q=90, tile_size=256):
    """Simulate JXL tile server: decode JXL, slice, JPEG re-encode, measure stain metrics."""
    import subprocess

    src = np.array(Image.open(source_path))

    # Decode JXL
    png_path = '/tmp/_stain_jxl_decoded.png'
    subprocess.run(['djxl', jxl_path, png_path], capture_output=True, check=True)
    jxl_decoded = np.array(Image.open(png_path))

    h, w = src.shape[:2]
    tile_metrics = {}

    for ty in range(h // tile_size):
        for tx in range(w // tile_size):
            y0, x0 = ty * tile_size, tx * tile_size
            src_tile = src[y0:y0+tile_size, x0:x0+tile_size]
            jxl_tile = jxl_decoded[y0:y0+tile_size, x0:x0+tile_size]

            # JPEG re-encode (what the viewer actually sees)
            served_tile = jpeg_roundtrip(jxl_tile, serve_q)

            tile_metrics[f'tile_{tx}_{ty}'] = compute_stain_metrics(src_tile, served_tile)

    # Average
    avg = {}
    keys = list(list(tile_metrics.values())[0].keys())
    for k in keys:
        vals = [m[k] for m in tile_metrics.values()]
        avg[k] = round(sum(vals) / len(vals), 6)

    kb = os.path.getsize(jxl_path) / 1024
    return avg, tile_metrics, kb


def print_metrics(label, metrics, kb=None):
    """Pretty-print stain metrics."""
    kb_str = f" ({kb:.1f} KB)" if kb else ""
    print(f"\n{label}{kb_str}")
    print(f"  {'Stain':<6} {'PSNR':>7} {'MSE':>10} {'MAE':>10} {'P50_AE':>10} {'P95_AE':>10} {'P99_AE':>10} {'MAX_AE':>10} {'MeanErr':>10} {'StdErr':>10}")
    print(f"  {'-'*100}")
    for ch in ['H', 'E']:
        print(f"  {ch:<6} {metrics[f'{ch}_psnr']:>7.2f} {metrics[f'{ch}_mse']:>10.6f} {metrics[f'{ch}_mae']:>10.6f} "
              f"{metrics[f'{ch}_p50_ae']:>10.6f} {metrics[f'{ch}_p95_ae']:>10.6f} {metrics[f'{ch}_p99_ae']:>10.6f} "
              f"{metrics[f'{ch}_max_ae']:>10.6f} {metrics[f'{ch}_mean_error']:>10.6f} {metrics[f'{ch}_std_error']:>10.6f}")


def main():
    parser = argparse.ArgumentParser(description='H&E stain fidelity metrics')
    parser.add_argument('--ref', help='Reference image path')
    parser.add_argument('--test', help='Test image path')
    parser.add_argument('--run', nargs='+', help='ORIGAMI run directory/directories')
    parser.add_argument('--jxl', help='JXL file path (simulates tile server)')
    parser.add_argument('--source', default='evals/test-images/L0-1024.jpg', help='Source image for JXL comparison')
    parser.add_argument('--serve-q', type=int, default=90, help='JPEG quality for tile serving')
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    if args.ref and args.test:
        full, tiles = process_image_pair(args.ref, args.test, args.tile_size)
        if args.json:
            print(json.dumps({'full': full, 'tiles': tiles}, indent=2))
        else:
            print_metrics(f"Stain metrics: {args.test} vs {args.ref}", full)

    elif args.jxl:
        avg, tiles, kb = process_jxl(args.jxl, args.source, args.serve_q, args.tile_size)
        if args.json:
            print(json.dumps({'average': avg, 'tiles': tiles, 'storage_kb': kb}, indent=2))
        else:
            print_metrics(f"JXL tile server: {args.jxl}", avg, kb)

    elif args.run:
        for run_dir in args.run:
            result = process_run(run_dir, args.source)
            if result:
                avg, tiles = result
                manifest = json.loads(Path(run_dir, 'manifest.json').read_text())
                kb = (manifest.get('l2_bytes', 0) + manifest.get('fused_l0_bytes', 0)) / 1024
                print_metrics(f"ORIGAMI: {Path(run_dir).name}", avg, kb)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
