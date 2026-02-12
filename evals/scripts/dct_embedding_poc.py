#!/usr/bin/env python3
"""
dct_embedding_poc.py

Proof-of-concept: embed arbitrary bits into an image such that they survive
JPEG compression at a known quality level.

Method: Quantization Index Modulation (QIM)
- Transform image into 8x8 DCT blocks (matching JPEG's internal transform)
- For each selected DCT coefficient, force its quantized index to encode a bit:
    - bit=0 → round to nearest EVEN multiple of Q
    - bit=1 → round to nearest ODD multiple of Q
- The decoder reads the parity of floor(coeff / Q) to extract the bit
- This survives JPEG re-compression at the same (or higher) quality because
  the coefficient is already at a quantization grid point

Usage:
  python dct_embedding_poc.py --image evals/test-images/L0-1024.jpg --quality 95
"""
import argparse
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
import io
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_bytes, parse_encoder_arg


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


def quality_to_qtable(quality, base_table=JPEG_LUMA_Q50):
    """Convert JPEG quality (1-100) to quantization table, matching libjpeg's formula."""
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    qtable = np.floor((base_table * scale + 50) / 100).astype(np.float64)
    qtable[qtable < 1] = 1
    qtable[qtable > 255] = 255
    return qtable


def dct_8x8(block):
    """2D DCT of an 8x8 block (type-II, matching JPEG)."""
    return dctn(block, type=2, norm='ortho')


def idct_8x8(block):
    """2D inverse DCT of an 8x8 block."""
    return idctn(block, type=2, norm='ortho')


def embed_bits(image_gray, bits, quality, coeff_positions=None):
    """
    Embed bits into grayscale image using QIM in DCT domain.

    Args:
        image_gray: HxW uint8 grayscale image
        bits: array of 0s and 1s to embed
        quality: JPEG quality level (determines quantization table)
        coeff_positions: list of (row, col) DCT coefficient positions to use
                        Default: [(1,2), (2,1), (2,2)] — mid-frequency coefficients

    Returns:
        modified image (uint8), number of bits embedded
    """
    if coeff_positions is None:
        # Mid-frequency coefficients: enough energy to be robust,
        # not so low-frequency that they cause visible artifacts
        coeff_positions = [(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]

    qtable = quality_to_qtable(quality)
    img = image_gray.astype(np.float64) - 128.0  # JPEG centers at 128
    h, w = img.shape

    # Pad to multiple of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')

    h_pad, w_pad = img.shape
    n_blocks_h = h_pad // 8
    n_blocks_w = w_pad // 8
    total_blocks = n_blocks_h * n_blocks_w
    bits_per_block = len(coeff_positions)
    capacity = total_blocks * bits_per_block

    n_embed = min(len(bits), capacity)
    embedded = np.zeros(img.shape, dtype=np.float64)

    bit_idx = 0
    for by in range(n_blocks_h):
        for bx in range(n_blocks_w):
            block = img[by*8:(by+1)*8, bx*8:(bx+1)*8]
            coeffs = dct_8x8(block)

            for ci, (cr, cc) in enumerate(coeff_positions):
                if bit_idx >= n_embed:
                    break

                q = qtable[cr, cc]
                c = coeffs[cr, cc]

                # Quantized index
                idx = c / q

                # QIM: force to even (bit=0) or odd (bit=1)
                target_bit = bits[bit_idx]
                rounded = round(idx)

                if (rounded % 2) != target_bit:
                    # Need to adjust: pick nearest index with correct parity
                    if idx > rounded:
                        rounded += 1  # go up
                    else:
                        rounded -= 1  # go down

                coeffs[cr, cc] = rounded * q
                bit_idx += 1

            embedded[by*8:(by+1)*8, bx*8:(bx+1)*8] = idct_8x8(coeffs)

    # Undo centering and clip
    result = np.clip(np.round(embedded + 128.0), 0, 255).astype(np.uint8)
    return result[:h, :w], n_embed


def extract_bits(image_gray, n_bits, quality, coeff_positions=None):
    """
    Extract embedded bits from grayscale image using QIM in DCT domain.

    Args:
        image_gray: HxW uint8 grayscale image
        n_bits: number of bits to extract
        quality: JPEG quality level (must match embedding quality)
        coeff_positions: must match embedding positions

    Returns:
        array of extracted bits
    """
    if coeff_positions is None:
        coeff_positions = [(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]

    qtable = quality_to_qtable(quality)
    img = image_gray.astype(np.float64) - 128.0
    h, w = img.shape

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')

    h_pad, w_pad = img.shape
    n_blocks_h = h_pad // 8
    n_blocks_w = w_pad // 8

    extracted = []
    bit_idx = 0

    for by in range(n_blocks_h):
        for bx in range(n_blocks_w):
            if bit_idx >= n_bits:
                break

            block = img[by*8:(by+1)*8, bx*8:(bx+1)*8]
            coeffs = dct_8x8(block)

            for ci, (cr, cc) in enumerate(coeff_positions):
                if bit_idx >= n_bits:
                    break

                q = qtable[cr, cc]
                c = coeffs[cr, cc]
                idx = round(c / q)
                extracted.append(idx % 2)
                bit_idx += 1

    return np.array(extracted[:n_bits])


def psnr(a, b, data_range=255):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range**2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for embedding")
    args = parser.parse_args()

    # Load image
    img = Image.open(args.image).convert("RGB")
    img_array = np.array(img)

    # Use the L2 tile (256x256)
    l2 = np.array(Image.fromarray(img_array[:1024, :1024]).resize((256, 256), Image.LANCZOS))
    gray = np.mean(l2.astype(np.float64), axis=2).astype(np.uint8)

    print(f"{'='*70}")
    print(f"DCT DOMAIN EMBEDDING — Proof of Concept")
    print(f"{'='*70}")
    print(f"Image: {args.image}")
    print(f"Tile: 256x256 grayscale (L2)")
    print(f"Embedding quality: {args.quality}")

    # Test with different coefficient sets and quality levels
    coeff_sets = {
        "1 coeff (2,1)": [(2, 1)],
        "3 coeffs": [(1, 2), (2, 1), (2, 2)],
        "5 coeffs": [(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)],
        "10 coeffs": [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1)],
    }

    for coeff_name, coeff_pos in coeff_sets.items():
        n_blocks = (256 // 8) * (256 // 8)  # 1024 blocks
        capacity_bits = n_blocks * len(coeff_pos)
        capacity_kb = capacity_bits / 8 / 1024

        print(f"\n--- {coeff_name} ({capacity_bits} bits = {capacity_kb:.1f} KB capacity) ---")

        # Generate random payload
        np.random.seed(42)
        payload = np.random.randint(0, 2, size=capacity_bits)

        # Embed
        embedded, n_embedded = embed_bits(gray, payload, args.quality, coeff_pos)
        embed_psnr = psnr(gray, embedded)
        print(f"  Embedded {n_embedded} bits, image PSNR: {embed_psnr:.1f} dB")

        # Extract directly (no JPEG round-trip)
        extracted_direct = extract_bits(embedded, n_embedded, args.quality, coeff_pos)
        ber_direct = np.mean(extracted_direct != payload[:n_embedded])
        print(f"  Direct extraction BER: {ber_direct:.4f} ({int(ber_direct*n_embedded)} errors / {n_embedded})")

        # Test JPEG round-trip at various qualities
        for jpeg_q in [args.quality, args.quality - 5, args.quality + 5,
                       args.quality - 10, 80, 90, 95, 99]:
            if jpeg_q < 1 or jpeg_q > 100 or jpeg_q == args.quality - 5 == args.quality + 5:
                continue

            # JPEG round-trip
            pil_img = Image.fromarray(embedded, mode="L")
            jpeg_bytes = encode_jpeg_to_bytes(pil_img, jpeg_q)
            decoded = np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("L"))

            # Extract from JPEG-decoded image
            extracted = extract_bits(decoded, n_embedded, args.quality, coeff_pos)
            ber = np.mean(extracted != payload[:n_embedded])
            decoded_psnr = psnr(gray, decoded)
            marker = " ← EMBEDDING QUALITY" if jpeg_q == args.quality else ""
            print(f"  JPEG q={jpeg_q:3d}: BER={ber:.4f} ({int(ber*n_embedded):5d}/{n_embedded}) "
                  f"PSNR={decoded_psnr:.1f}dB  size={len(jpeg_bytes)/1024:.1f}KB{marker}")

    # Capacity vs quality analysis
    print(f"\n{'='*70}")
    print(f"CAPACITY vs IMAGE QUALITY vs JPEG SURVIVAL")
    print(f"{'='*70}")

    coeff_pos = [(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]
    n_blocks = (256 // 8) * (256 // 8)
    capacity = n_blocks * len(coeff_pos)

    np.random.seed(42)
    payload = np.random.randint(0, 2, size=capacity)

    for embed_q in [75, 80, 85, 90, 95]:
        embedded, n = embed_bits(gray, payload, embed_q, coeff_pos)
        ep = psnr(gray, embedded)

        # JPEG at same quality
        pil_img = Image.fromarray(embedded, mode="L")
        jpeg_bytes = encode_jpeg_to_bytes(pil_img, embed_q)
        decoded = np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("L"))
        extracted = extract_bits(decoded, n, embed_q, coeff_pos)
        ber = np.mean(extracted != payload[:n])

        # JPEG at higher quality (should preserve better)
        jpeg_bytes_hq = encode_jpeg_to_bytes(pil_img, min(embed_q + 10, 100))
        decoded_hq = np.array(Image.open(io.BytesIO(jpeg_bytes_hq)).convert("L"))
        extracted_hq = extract_bits(decoded_hq, n, embed_q, coeff_pos)
        ber_hq = np.mean(extracted_hq != payload[:n])

        print(f"  Embed q={embed_q}: PSNR={ep:.1f}dB | same-q BER={ber:.4f} | +10q BER={ber_hq:.4f}")


if __name__ == "__main__":
    main()
