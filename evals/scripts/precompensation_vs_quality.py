#!/usr/bin/env python3
"""
precompensation_vs_quality.py

Compares: Is scaling residuals before JPEG encoding better than just using a higher
quality setting that produces the same file size?

If scale=1.2 at q=40 produces 71.7 KB, what quality setting produces ~71.7 KB
without scaling? And which gives better PSNR?

This is the fair comparison — matching total bytes.
"""
import argparse
import numpy as np
from PIL import Image
import io
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_bytes, parse_encoder_arg


def encode_decode_gray(pixels_u8, quality, encoder=JpegEncoder.LIBJPEG_TURBO):
    img = Image.fromarray(pixels_u8, mode="L")
    data = encode_jpeg_to_bytes(img, quality, encoder)
    decoded = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return decoded, len(data)


def rgb_to_ycbcr_bt601(rgb_u8):
    rgb = rgb_u8.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128.0
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 128.0
    return Y, Cb, Cr


def ycbcr_to_rgb_bt601(Y, Cb, Cr):
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0
    R = Y + 1.402*Cr
    G = Y - 0.344136*Cb - 0.714136*Cr
    B = Y + 1.772*Cb
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)


def psnr(a, b, data_range=255):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range**2 / mse)


def full_pipeline(image_path, quality, scale=1.0, encoder=JpegEncoder.LIBJPEG_TURBO):
    """Run full L2→L1→L0 pipeline with optional residual scaling."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    tile_size = 256
    UPSAMPLE = Image.Resampling.BILINEAR

    l2 = np.array(Image.fromarray(img_array[:1024, :1024]).resize((256, 256), Image.LANCZOS))
    l1_source = np.array(Image.fromarray(img_array[:1024, :1024]).resize((512, 512), Image.LANCZOS))

    l0_tiles = {}
    for dy in range(4):
        for dx in range(4):
            l0_tiles[(dx, dy)] = img_array[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    l1_tiles = {}
    for dy in range(2):
        for dx in range(2):
            l1_tiles[(dx, dy)] = l1_source[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    l1_pred_mosaic = np.array(Image.fromarray(l2).resize((512, 512), resample=UPSAMPLE))

    l1_psnrs = []
    total_l1_bytes = 0
    total_l0_bytes = 0
    l1_reconstructed = {}

    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)

        residual_raw = Y_gt - Y_pred
        scaled = residual_raw * scale
        centered = np.clip(np.round(scaled + 128.0), 0, 255).astype(np.uint8)
        decoded, size = encode_decode_gray(centered, quality, encoder)
        decoded_residual = (decoded.astype(np.float32) - 128.0) / scale
        Y_recon = np.clip(Y_pred + decoded_residual, 0, 255)

        total_l1_bytes += size
        rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
        l1_reconstructed[(dx, dy)] = rgb_recon
        l1_psnrs.append(psnr(l1_gt, rgb_recon))

    l1_mosaic = np.zeros((512, 512, 3), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            l1_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256] = l1_reconstructed[(dx, dy)]

    l0_pred_mosaic = np.array(Image.fromarray(l1_mosaic).resize((1024, 1024), resample=UPSAMPLE))

    l0_psnrs = []
    for (dx, dy), l0_gt in l0_tiles.items():
        pred = l0_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l0_gt)

        residual_raw = Y_gt - Y_pred
        scaled = residual_raw * scale
        centered = np.clip(np.round(scaled + 128.0), 0, 255).astype(np.uint8)
        decoded, size = encode_decode_gray(centered, quality, encoder)
        decoded_residual = (decoded.astype(np.float32) - 128.0) / scale
        Y_recon = np.clip(Y_pred + decoded_residual, 0, 255)

        total_l0_bytes += size
        rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
        l0_psnrs.append(psnr(l0_gt, rgb_recon))

    total = total_l1_bytes + total_l0_bytes
    return {
        "l1_psnr": np.mean(l1_psnrs),
        "l0_psnr": np.mean(l0_psnrs),
        "min_psnr": min(min(l1_psnrs), min(l0_psnrs)),
        "total_bytes": total,
        "l1_bytes": total_l1_bytes,
        "l0_bytes": total_l0_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--encoder", default="libjpeg-turbo")
    args = parser.parse_args()

    encoder = parse_encoder_arg(args.encoder)

    print(f"{'='*80}")
    print(f"PRECOMPENSATION vs QUALITY — Fair comparison at matched byte counts")
    print(f"{'='*80}")

    # First, sweep quality levels to build a bytes→PSNR curve
    print(f"\n--- Quality sweep (no scaling) ---")
    quality_results = {}
    for q in range(25, 86, 5):
        r = full_pipeline(args.image, q, scale=1.0, encoder=encoder)
        quality_results[q] = r
        print(f"  q={q:2d}: L1={r['l1_psnr']:.2f} dB, L0={r['l0_psnr']:.2f} dB, "
              f"min={r['min_psnr']:.2f} dB, size={r['total_bytes']/1024:.1f} KB")

    # Now test scaling at base qualities
    print(f"\n--- Scale sweep at various base qualities ---")
    for base_q in [30, 40, 50, 60]:
        print(f"\n  Base quality = {base_q}:")
        for scale in [1.0, 1.1, 1.2, 1.3, 1.5]:
            r = full_pipeline(args.image, base_q, scale=scale, encoder=encoder)

            # Find the quality level with closest byte count (no scaling)
            closest_q = min(quality_results.keys(),
                           key=lambda q: abs(quality_results[q]["total_bytes"] - r["total_bytes"]))
            matched = quality_results[closest_q]

            delta_min = r["min_psnr"] - matched["min_psnr"]
            delta_l1 = r["l1_psnr"] - matched["l1_psnr"]
            delta_l0 = r["l0_psnr"] - matched["l0_psnr"]

            print(f"    scale={scale:.1f}: min={r['min_psnr']:.2f}dB @ {r['total_bytes']/1024:.1f}KB "
                  f"  vs  q={closest_q} min={matched['min_psnr']:.2f}dB @ {matched['total_bytes']/1024:.1f}KB "
                  f"  Δmin={delta_min:+.2f}dB  ΔL1={delta_l1:+.2f}dB  ΔL0={delta_l0:+.2f}dB")


if __name__ == "__main__":
    main()
