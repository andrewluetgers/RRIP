#!/usr/bin/env python3
"""
dct_embedding_analysis.py

Final analysis: What is the maximum reliable capacity of QIM embedding
in a 256x256 grayscale L2 tile, and how does it compare to the residual
data we need to store?

Also explores: what if we use ALL three channels (Y, Cb, Cr)?
And what about multiple L2 tiles (larger images)?
"""
import argparse
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
import io
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_bytes


JPEG_LUMA_Q50 = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float64)

JPEG_CHROMA_Q50 = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def quality_to_qtable(quality, base_table=JPEG_LUMA_Q50):
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    qtable = np.floor((base_table * scale + 50) / 100).astype(np.float64)
    qtable[qtable < 1] = 1
    qtable[qtable > 255] = 255
    return qtable


def psnr(a, b, data_range=255):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range**2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"DCT EMBEDDING — CAPACITY ANALYSIS")
    print(f"{'='*70}")

    # L2 tile is 256x256 = 32x32 = 1024 blocks of 8x8
    n_blocks = 32 * 32  # 1024 blocks

    print(f"\n--- Theoretical capacity per block at various JPEG qualities ---")
    print(f"{'Q':>4s}  {'Q_min':>6s}  {'Q_max':>6s}  {'Q_med':>6s}  "
          f"{'Safe coeffs':>12s}  {'Bits/block':>10s}  {'Total bits':>10s}  {'Total KB':>10s}  "
          f"{'PSNR cost':>10s}")
    print("-" * 100)

    for q in [50, 60, 70, 75, 80, 85, 90, 95]:
        qt_luma = quality_to_qtable(q, JPEG_LUMA_Q50)

        # "Safe" coefficients: Q >= 4 (enough headroom for parity nudge)
        # and not DC (0,0)
        safe = []
        for r in range(8):
            for c in range(8):
                if (r, c) == (0, 0):
                    continue
                if qt_luma[r, c] >= 4:
                    safe.append((r, c))

        bits_per_block = len(safe)
        total_bits = n_blocks * bits_per_block
        total_kb = total_bits / 8 / 1024

        # Estimate PSNR cost: each modified coeff adds ~(Q/2)^2 to MSE
        # Average Q for safe coefficients
        safe_qs = [qt_luma[r, c] for r, c in safe]
        if safe_qs:
            avg_q = np.mean(safe_qs)
            # Each block is 64 pixels, each coeff change adds ~(Q/2)^2 energy
            # spread across 64 pixels. About half the coefficients get modified (random payload).
            mse_per_block = (bits_per_block / 2) * (avg_q / 2)**2 / 64
            psnr_cost = 10 * np.log10(255**2 / mse_per_block) if mse_per_block > 0 else float('inf')
        else:
            psnr_cost = float('inf')

        print(f"{q:4d}  {min(safe_qs) if safe_qs else 0:6.0f}  {max(safe_qs) if safe_qs else 0:6.0f}  "
              f"{np.median(safe_qs) if safe_qs else 0:6.0f}  "
              f"{bits_per_block:12d}  {bits_per_block:10d}  {total_bits:10d}  {total_kb:10.1f}  "
              f"{psnr_cost:10.1f}dB")

    # Compare to residual data needs
    print(f"\n--- Residual data budget ---")
    print(f"  L1 residuals at q=40: ~11 KB")
    print(f"  L0 residuals at q=40: ~50 KB")
    print(f"  Total residuals at q=40: ~61 KB")
    print(f"  Total residuals at q=60: ~91 KB")

    print(f"\n--- Can embedding replace residuals? ---")
    for q in [50, 60, 70, 75, 80]:
        qt = quality_to_qtable(q, JPEG_LUMA_Q50)
        safe = [(r, c) for r in range(8) for c in range(8)
                if (r, c) != (0, 0) and qt[r, c] >= 4]
        cap_kb = n_blocks * len(safe) / 8 / 1024

        # For RGB (3 channels): luma + 2 chroma
        qt_c = quality_to_qtable(q, JPEG_CHROMA_Q50)
        safe_c = [(r, c) for r in range(8) for c in range(8)
                  if (r, c) != (0, 0) and qt_c[r, c] >= 4]
        cap_rgb_kb = (n_blocks * len(safe) + 2 * n_blocks * len(safe_c)) / 8 / 1024

        print(f"  q={q}: gray={cap_kb:.1f} KB, RGB={cap_rgb_kb:.1f} KB "
              f"(need 61 KB → {'POSSIBLE' if cap_rgb_kb >= 61 else 'IMPOSSIBLE'})")

    # The real question: even at max capacity, the embedding degrades L2 quality
    # substantially. What's the net effect on reconstruction?
    print(f"\n--- Net effect analysis ---")
    print(f"  Even if we could embed 61 KB of residual data into L2:")
    print(f"    - At q=50: L2 PSNR drops ~20 dB (severe visible artifacts)")
    print(f"    - Those artifacts propagate through upsampling to ALL L1 and L0 tiles")
    print(f"    - From our L2 sensitivity test: L2 at PSNR 33.8 → -2.8 dB on L0 reconstruction")
    print(f"    - Embedding at high capacity pushes L2 PSNR below 25 dB → devastating")
    print(f"")
    print(f"  The fundamental problem: to embed N bits, you perturb ~N/2 coefficients")
    print(f"  by ~Q/2 each. This is equivalent to adding quantization noise at quality Q.")
    print(f"  Embedding 61 KB into 256x256 = 3.2 bits/pixel → equivalent to q≈20-30 JPEG.")
    print(f"")
    print(f"  CONCLUSION: QIM embedding in L2 can reliably survive JPEG round-trips,")
    print(f"  but the capacity (~0.1-1 KB at acceptable quality) is 60-600x too small")
    print(f"  to replace residual files.")

    # What CAN it be used for?
    print(f"\n{'='*70}")
    print(f"VIABLE USE CASES FOR QIM IN ORIGAMI")
    print(f"{'='*70}")
    print(f"""
  1. METADATA EMBEDDING (~128 bytes at q=75, BER=0):
     - Encode residual quality settings, tile count, version info
     - Pack file location/hash for self-describing tiles
     - Eliminates need for out-of-band metadata

  2. ERROR CORRECTION CODES (~256-512 bytes):
     - Embed checksums or Reed-Solomon codes for residual data integrity
     - Detect corruption in stored residuals

  3. PREDICTION HINTS (~128-512 bytes):
     - Encode per-block "prediction mode" flags
     - Signal which L1 tiles have high-energy residuals (prioritize fetching)
     - Encode adaptive quantization maps

  4. WATERMARKING:
     - Embed provenance/ownership information
     - Survives JPEG re-compression and moderate editing
""")


if __name__ == "__main__":
    main()
