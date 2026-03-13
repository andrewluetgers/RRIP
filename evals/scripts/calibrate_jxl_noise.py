#!/usr/bin/env python3
"""Calibrate JXL photon_noise_iso to wavelet noise sigma.

Encodes a test image at JXL Q40 with various ISO values, decodes, and measures
the resulting noise sigma (via wavelet MAD estimator). Builds a lookup table
mapping measured sigma → ISO so we can auto-set the ISO to match an original
image's noise characteristics.

Usage:
    uv run python evals/scripts/calibrate_jxl_noise.py [--image PATH] [--quality 40]
"""

import argparse
import json
import subprocess
import tempfile
import pathlib
import numpy as np
from PIL import Image
import pywt


def estimate_sigma_mad(img_gray):
    """Estimate noise sigma via MAD of finest wavelet detail coefficients.
    Matches the Rust wavelet.rs estimate_sigma_mad() implementation."""
    # 2-level DWT decomposition using db4 (matches our Rust code)
    coeffs = pywt.wavedec2(img_gray.astype(np.float64), 'db4', level=2)
    # Finest detail: last element of coeffs list, horizontal subband
    finest_detail = coeffs[-1][0]  # (cH at finest level)
    sigma = np.median(np.abs(finest_detail)) / 0.6745
    return sigma


def measure_subband_sigmas(img_gray):
    """Measure per-subband noise sigmas (matches SynthesisParams structure)."""
    coeffs = pywt.wavedec2(img_gray.astype(np.float64), 'db4', level=2)
    # coeffs = [cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]
    approx_sigma = np.std(coeffs[0])
    subband_sigmas = []
    for level_detail in coeffs[1:]:  # level 2 (coarse) then level 1 (fine)
        h_sigma = np.std(level_detail[0])
        v_sigma = np.std(level_detail[1])
        d_sigma = np.std(level_detail[2])
        subband_sigmas.append([h_sigma, v_sigma, d_sigma])
    return approx_sigma, subband_sigmas


def encode_decode_jxl(image_path, quality, iso, tmp_dir):
    """Encode image to JXL with given quality and ISO, decode back."""
    jxl_path = tmp_dir / f"q{quality}_iso{iso}.jxl"
    png_path = tmp_dir / f"q{quality}_iso{iso}.png"

    cmd_encode = ['cjxl', str(image_path), str(jxl_path),
                  '-q', str(quality), '--lossless_jpeg=0']
    if iso > 0:
        cmd_encode.append(f'--photon_noise_iso={iso}')

    subprocess.run(cmd_encode, capture_output=True, check=True)
    jxl_size = jxl_path.stat().st_size

    cmd_decode = ['djxl', str(jxl_path), str(png_path)]
    subprocess.run(cmd_decode, capture_output=True, check=True)

    decoded = np.array(Image.open(png_path).convert('L')).astype(np.float64)
    return decoded, jxl_size


def main():
    parser = argparse.ArgumentParser(description='Calibrate JXL noise ISO to sigma')
    parser.add_argument('--image', default='evals/test-images/L0-1024.jpg',
                        help='Test image path')
    parser.add_argument('--quality', type=int, default=40,
                        help='JXL quality level')
    parser.add_argument('--output', default='evals/analysis/jxl_noise_calibration.json',
                        help='Output calibration file')
    args = parser.parse_args()

    image_path = pathlib.Path(args.image)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original and measure its noise
    original = np.array(Image.open(image_path).convert('L')).astype(np.float64)
    orig_sigma = estimate_sigma_mad(original)
    orig_approx, orig_subbands = measure_subband_sigmas(original)

    print(f"Original image: {image_path}")
    print(f"  MAD sigma: {orig_sigma:.4f}")
    print(f"  Approx sigma: {orig_approx:.4f}")
    print(f"  Subband sigmas (coarse→fine):")
    for i, sb in enumerate(orig_subbands):
        print(f"    Level {i+1}: H={sb[0]:.4f} V={sb[1]:.4f} D={sb[2]:.4f}")
    print()

    # ISO sweep
    iso_values = [0, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3200,
                  4800, 6400, 9600, 12800, 19200, 25600, 38400, 51200]

    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = pathlib.Path(tmp)

        # First: encode without noise to get the "clean" compressed baseline
        print(f"JXL Q{args.quality} sweep:")
        print(f"  {'ISO':>6}  {'Size':>7}  {'MAD σ':>7}  {'Δσ orig':>8}  "
              f"{'Fine H':>7}  {'Fine V':>7}  {'Fine D':>7}")
        print(f"  {'-'*65}")

        for iso in iso_values:
            decoded, jxl_size = encode_decode_jxl(image_path, args.quality, iso, tmp_dir)
            sigma = estimate_sigma_mad(decoded)
            approx, subbands = measure_subband_sigmas(decoded)

            # Difference from original
            delta_sigma = sigma - orig_sigma
            fine = subbands[-1]  # finest level

            print(f"  {iso:>6}  {jxl_size:>6}B  {sigma:>7.4f}  {delta_sigma:>+8.4f}  "
                  f"{fine[0]:>7.4f}  {fine[1]:>7.4f}  {fine[2]:>7.4f}")

            results.append({
                'iso': iso,
                'jxl_size': jxl_size,
                'mad_sigma': round(sigma, 6),
                'delta_sigma_from_original': round(delta_sigma, 6),
                'approx_sigma': round(approx, 6),
                'subband_sigmas': [[round(s, 6) for s in sb] for sb in subbands],
            })

    # Find best ISO match for original sigma
    clean_sigma = results[0]['mad_sigma']  # ISO=0
    print(f"\n  Clean (ISO=0) sigma: {clean_sigma:.4f}")
    print(f"  Original sigma:     {orig_sigma:.4f}")
    print(f"  Noise removed by JXL Q{args.quality}: {orig_sigma - clean_sigma:.4f}")

    # Find ISO that best matches original sigma
    best_iso = 0
    best_diff = abs(results[0]['mad_sigma'] - orig_sigma)
    for r in results:
        diff = abs(r['mad_sigma'] - orig_sigma)
        if diff < best_diff:
            best_diff = diff
            best_iso = r['iso']

    print(f"\n  Best ISO match for original: ISO {best_iso} "
          f"(sigma={next(r['mad_sigma'] for r in results if r['iso']==best_iso):.4f}, "
          f"diff={best_diff:.4f})")

    # Build calibration table: sigma → ISO (for interpolation)
    calibration = {
        'image': str(image_path),
        'quality': args.quality,
        'original_sigma': round(orig_sigma, 6),
        'original_approx_sigma': round(orig_approx, 6),
        'original_subband_sigmas': [[round(s, 6) for s in sb] for sb in orig_subbands],
        'clean_sigma': clean_sigma,
        'best_iso_match': best_iso,
        'sigma_to_iso_table': [
            {'sigma': r['mad_sigma'], 'iso': r['iso'], 'size': r['jxl_size']}
            for r in results
        ],
        'full_results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to: {output_path}")


if __name__ == '__main__':
    main()
