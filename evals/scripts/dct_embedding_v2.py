#!/usr/bin/env python3
"""
dct_embedding_v2.py

JPEG-native QIM embedding: instead of computing our own DCT (which may differ
from libjpeg's implementation), we use the JPEG round-trip itself as the
quantization step, then nudge pixels to flip specific coefficient parities.

Strategy:
1. JPEG-compress the image at quality Q → get stable pixel values
2. These pixels represent the quantized DCT coefficients (the JPEG grid)
3. To embed a bit, perturb a pixel group to nudge a DCT coefficient by ±Q
4. The perturbation is "JPEG-stable" because it moves to an adjacent grid point

Simpler alternative tested here:
- Encode image as JPEG at quality Q, decode back → "stabilized" image
- The stabilized pixel values map to exact quantized DCT coefficients
- Read the parity of those coefficients to extract a message
- To embed: perturb the pre-JPEG image so that after JPEG, coefficients have desired parity

Even simpler (what we actually test):
- Use jpegio/Pillow to directly read/write JPEG DCT coefficients
- This gives us direct access to the quantized coefficients
- Modify parity, re-save → perfect embedding that survives any re-compression at same Q

Usage:
  python dct_embedding_v2.py --image evals/test-images/L0-1024.jpg --quality 85
"""
import argparse
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
import io
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_bytes


# Standard JPEG luminance quantization table (quality 50 baseline)
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


def quality_to_qtable(quality):
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    qtable = np.floor((JPEG_LUMA_Q50 * scale + 50) / 100).astype(np.float64)
    qtable[qtable < 1] = 1
    qtable[qtable > 255] = 255
    return qtable


def psnr(a, b, data_range=255):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range**2 / mse)


def jpeg_roundtrip(gray_u8, quality):
    """JPEG encode+decode a grayscale image, return decoded pixels and size."""
    img = Image.fromarray(gray_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    size = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert("L"))
    return decoded, size


def get_dct_coeffs(gray_u8):
    """Compute DCT coefficients for all 8x8 blocks."""
    h, w = gray_u8.shape
    img = gray_u8.astype(np.float64) - 128.0

    bh, bw = h // 8, w // 8
    coeffs = np.zeros((bh, bw, 8, 8), dtype=np.float64)

    for by in range(bh):
        for bx in range(bw):
            block = img[by*8:(by+1)*8, bx*8:(bx+1)*8]
            coeffs[by, bx] = dctn(block, type=2, norm='ortho')

    return coeffs


def set_dct_coeffs(coeffs):
    """Convert DCT coefficients back to pixel image."""
    bh, bw = coeffs.shape[:2]
    h, w = bh * 8, bw * 8
    img = np.zeros((h, w), dtype=np.float64)

    for by in range(bh):
        for bx in range(bw):
            img[by*8:(by+1)*8, bx*8:(bx+1)*8] = idctn(coeffs[by, bx], type=2, norm='ortho')

    return np.clip(np.round(img + 128.0), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--quality", type=int, default=85)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    img_array = np.array(img)
    l2 = np.array(Image.fromarray(img_array[:1024, :1024]).resize((256, 256), Image.LANCZOS))
    gray = np.mean(l2.astype(np.float64), axis=2).astype(np.uint8)

    print(f"{'='*70}")
    print(f"JPEG-STABLE EMBEDDING v2")
    print(f"{'='*70}")

    # Step 1: JPEG-stabilize the image
    # After one JPEG round-trip, the pixel values represent the quantized grid.
    # A second round-trip at the same quality should be (nearly) idempotent.
    print(f"\n--- JPEG idempotency test ---")
    for q in [75, 80, 85, 90, 95]:
        rt1, s1 = jpeg_roundtrip(gray, q)
        rt2, s2 = jpeg_roundtrip(rt1, q)
        rt3, s3 = jpeg_roundtrip(rt2, q)
        diff_12 = np.sum(rt1 != rt2)
        diff_23 = np.sum(rt2 != rt3)
        max_diff_12 = np.max(np.abs(rt1.astype(int) - rt2.astype(int)))
        psnr_orig = psnr(gray, rt1)
        print(f"  q={q}: orig→rt1 PSNR={psnr_orig:.1f}dB | "
              f"rt1→rt2: {diff_12} pixels differ (max diff={max_diff_12}) | "
              f"rt2→rt3: {diff_23} pixels differ | "
              f"size: {s1}→{s2}→{s3}")

    # Step 2: QIM on JPEG-stabilized image
    # Strategy: stabilize first, then read/modify DCT coefficients
    print(f"\n--- QIM embedding on JPEG-stabilized image at q={args.quality} ---")

    # Stabilize
    stabilized, _ = jpeg_roundtrip(gray, args.quality)
    # Verify idempotency
    restabilized, _ = jpeg_roundtrip(stabilized, args.quality)
    n_unstable = np.sum(stabilized != restabilized)
    print(f"  Stabilized: {n_unstable} pixels differ after 2nd round-trip")

    # Get DCT coefficients of stabilized image
    qtable = quality_to_qtable(args.quality)
    coeffs = get_dct_coeffs(stabilized)
    bh, bw = coeffs.shape[:2]

    # Which coefficients to use? Those with Q > 1 (otherwise parity is meaningless)
    # and not DC (position 0,0) which affects overall brightness
    usable_positions = []
    for r in range(8):
        for c in range(8):
            if (r, c) == (0, 0):
                continue
            if qtable[r, c] >= 2:  # Q must be at least 2 for meaningful parity
                usable_positions.append((r, c))

    print(f"  Quantization table at q={args.quality}:")
    print(f"    {qtable.astype(int)}")
    print(f"  Usable coeff positions (Q >= 2, non-DC): {len(usable_positions)}")
    capacity_bits = bh * bw * len(usable_positions)
    print(f"  Total capacity: {capacity_bits} bits = {capacity_bits/8/1024:.1f} KB")

    # Test with different numbers of coefficients
    for n_coeffs in [1, 3, 5, 10, len(usable_positions)]:
        positions = usable_positions[:n_coeffs]
        cap = bh * bw * n_coeffs

        # Generate payload
        np.random.seed(42)
        payload = np.random.randint(0, 2, size=cap)

        # Embed by modifying quantized coefficient indices
        modified_coeffs = coeffs.copy()
        bit_idx = 0
        changes_made = 0

        for by in range(bh):
            for bx in range(bw):
                for (cr, cc) in positions:
                    if bit_idx >= cap:
                        break
                    q = qtable[cr, cc]
                    c = modified_coeffs[by, bx, cr, cc]
                    idx = round(c / q)

                    target = payload[bit_idx]
                    if (idx % 2) != target:
                        # Nudge to adjacent grid point
                        if c > idx * q:
                            new_idx = idx + 1
                        else:
                            new_idx = idx - 1
                        modified_coeffs[by, bx, cr, cc] = new_idx * q
                        changes_made += 1

                    bit_idx += 1

        # Reconstruct image from modified coefficients
        embedded_img = set_dct_coeffs(modified_coeffs)
        embed_psnr = psnr(gray, embedded_img)
        stab_psnr = psnr(stabilized, embedded_img)

        # Extract bits directly from modified image
        check_coeffs = get_dct_coeffs(embedded_img)
        extracted_direct = []
        bit_idx = 0
        for by in range(bh):
            for bx in range(bw):
                for (cr, cc) in positions:
                    if bit_idx >= cap:
                        break
                    q = qtable[cr, cc]
                    c = check_coeffs[by, bx, cr, cc]
                    idx = round(c / q)
                    extracted_direct.append(idx % 2)
                    bit_idx += 1

        extracted_direct = np.array(extracted_direct[:cap])
        ber_direct = np.mean(extracted_direct != payload)

        # JPEG round-trip and extract
        rt_img, rt_size = jpeg_roundtrip(embedded_img, args.quality)
        rt_coeffs = get_dct_coeffs(rt_img)
        extracted_rt = []
        bit_idx = 0
        for by in range(bh):
            for bx in range(bw):
                for (cr, cc) in positions:
                    if bit_idx >= cap:
                        break
                    q = qtable[cr, cc]
                    c = rt_coeffs[by, bx, cr, cc]
                    idx = round(c / q)
                    extracted_rt.append(idx % 2)
                    bit_idx += 1

        extracted_rt = np.array(extracted_rt[:cap])
        ber_rt = np.mean(extracted_rt != payload)

        # Also test at different JPEG qualities
        ber_others = {}
        for test_q in [args.quality - 10, args.quality + 5, 99]:
            if test_q < 1 or test_q > 100:
                continue
            rt2, _ = jpeg_roundtrip(embedded_img, test_q)
            c2 = get_dct_coeffs(rt2)
            ext2 = []
            bi = 0
            for by in range(bh):
                for bx in range(bw):
                    for (cr, cc) in positions:
                        if bi >= cap:
                            break
                        q_val = qtable[cr, cc]
                        idx2 = round(c2[by, bx, cr, cc] / q_val)
                        ext2.append(idx2 % 2)
                        bi += 1
            ber_others[test_q] = np.mean(np.array(ext2[:cap]) != payload)

        print(f"\n  {n_coeffs} coeff(s) — {cap} bits ({cap/8/1024:.2f} KB), {changes_made} coeffs modified:")
        print(f"    Image PSNR: {embed_psnr:.1f} dB (vs original), {stab_psnr:.1f} dB (vs stabilized)")
        print(f"    Direct BER: {ber_direct:.6f} ({int(ber_direct*cap)} errors)")
        print(f"    Same-Q JPEG BER: {ber_rt:.6f} ({int(ber_rt*cap)} errors)")
        for tq, ber_o in sorted(ber_others.items()):
            label = "SAME" if tq == args.quality else f"q={tq}"
            print(f"    JPEG q={tq:3d} BER: {ber_o:.6f} ({int(ber_o*cap)} errors)")


if __name__ == "__main__":
    main()
