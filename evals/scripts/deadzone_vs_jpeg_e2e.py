#!/usr/bin/env python3
"""
deadzone_vs_jpeg_e2e.py

End-to-end comparison: JPEG residual coding vs deadzone+zlib residual coding
through the full L2→L1→L0 reconstruction pipeline.

The key question: at the same total byte count, which gives better final tile quality?
"""
import argparse
import numpy as np
from PIL import Image
import io
import zlib
import struct
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_bytes


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


def encode_jpeg(residual_float, quality):
    """JPEG encode/decode residual. Returns (reconstructed_residual, compressed_size)."""
    centered = np.clip(np.round(residual_float + 128.0), 0, 255).astype(np.uint8)
    img = Image.fromarray(centered, mode="L")
    data = encode_jpeg_to_bytes(img, quality)
    decoded = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return decoded.astype(np.float32) - 128.0, len(data)


def encode_deadzone(residual_float, step_size):
    """Deadzone quantize + zlib compress residual. Returns (reconstructed_residual, compressed_size)."""
    quantized = np.round(residual_float / step_size).astype(np.int8)
    header = struct.pack('<HHf', residual_float.shape[0], residual_float.shape[1], step_size)
    compressed = zlib.compress(header + quantized.tobytes(), 9)
    recon = quantized.astype(np.float32) * step_size
    return recon, len(compressed)


def encode_deadzone_int16(residual_float, step_size):
    """Deadzone with int16 for larger dynamic range."""
    quantized = np.round(residual_float / step_size).astype(np.int16)
    header = struct.pack('<HHf', residual_float.shape[0], residual_float.shape[1], step_size)
    compressed = zlib.compress(header + quantized.tobytes(), 9)
    recon = quantized.astype(np.float32) * step_size
    return recon, len(compressed)


def full_pipeline(image_path, encode_fn, encode_params_l1, encode_params_l0):
    """
    Run full L2→L1→L0 pipeline with arbitrary residual encoding function.
    encode_fn(residual_float, params) → (recon_residual, size_bytes)
    """
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    UPSAMPLE = Image.Resampling.BILINEAR

    l2 = np.array(Image.fromarray(img_array[:1024, :1024]).resize((256, 256), Image.LANCZOS))
    l1_source = np.array(Image.fromarray(img_array[:1024, :1024]).resize((512, 512), Image.LANCZOS))

    l1_tiles = {}
    for dy in range(2):
        for dx in range(2):
            l1_tiles[(dx, dy)] = l1_source[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    l0_tiles = {}
    for dy in range(4):
        for dx in range(4):
            l0_tiles[(dx, dy)] = img_array[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    l1_pred_mosaic = np.array(Image.fromarray(l2).resize((512, 512), resample=UPSAMPLE))

    l1_psnrs = []
    total_l1_bytes = 0
    total_l0_bytes = 0
    l1_reconstructed = {}

    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)

        residual = Y_gt - Y_pred
        recon_residual, size = encode_fn(residual, encode_params_l1)
        total_l1_bytes += size

        Y_recon = np.clip(Y_pred + recon_residual, 0, 255)
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

        residual = Y_gt - Y_pred
        recon_residual, size = encode_fn(residual, encode_params_l0)
        total_l0_bytes += size

        Y_recon = np.clip(Y_pred + recon_residual, 0, 255)
        rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
        l0_psnrs.append(psnr(l0_gt, rgb_recon))

    return {
        "l1_psnr": np.mean(l1_psnrs),
        "l0_psnr": np.mean(l0_psnrs),
        "min_psnr": min(min(l1_psnrs), min(l0_psnrs)),
        "total_bytes": total_l1_bytes + total_l0_bytes,
        "l1_bytes": total_l1_bytes,
        "l0_bytes": total_l0_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    print(f"{'='*90}")
    print(f"END-TO-END: JPEG vs DEADZONE+ZLIB RESIDUAL CODING")
    print(f"{'='*90}")

    # JPEG sweep
    print(f"\n--- JPEG baseline (flat quality) ---")
    jpeg_results = {}
    for q in range(20, 96, 5):
        r = full_pipeline(args.image, encode_jpeg, q, q)
        jpeg_results[q] = r
        print(f"  JPEG q={q:2d}: L1={r['l1_psnr']:.2f}dB  L0={r['l0_psnr']:.2f}dB  "
              f"min={r['min_psnr']:.2f}dB  total={r['total_bytes']/1024:.1f}KB  "
              f"L1={r['l1_bytes']/1024:.1f}KB  L0={r['l0_bytes']/1024:.1f}KB")

    # JPEG +20 split
    print(f"\n--- JPEG +20 split ---")
    for q in range(20, 76, 5):
        r = full_pipeline(args.image, encode_jpeg, q+20, q)
        print(f"  JPEG L1={q+20:2d}/L0={q:2d}: L1={r['l1_psnr']:.2f}dB  L0={r['l0_psnr']:.2f}dB  "
              f"min={r['min_psnr']:.2f}dB  total={r['total_bytes']/1024:.1f}KB")

    # Deadzone sweep
    print(f"\n--- Deadzone + zlib (flat step) ---")
    dz_results = {}
    for step in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]:
        r = full_pipeline(args.image, encode_deadzone, step, step)
        dz_results[step] = r
        print(f"  DZ s={step:5.1f}: L1={r['l1_psnr']:.2f}dB  L0={r['l0_psnr']:.2f}dB  "
              f"min={r['min_psnr']:.2f}dB  total={r['total_bytes']/1024:.1f}KB  "
              f"L1={r['l1_bytes']/1024:.1f}KB  L0={r['l0_bytes']/1024:.1f}KB")

    # Deadzone split (smaller step for L1, larger for L0)
    print(f"\n--- Deadzone split (L1 fine, L0 coarse) ---")
    for l0_step in [4.0, 6.0, 8.0]:
        for l1_step in [2.0, 3.0, 4.0]:
            if l1_step >= l0_step:
                continue
            r = full_pipeline(args.image, encode_deadzone, l1_step, l0_step)
            print(f"  DZ L1={l1_step}/L0={l0_step}: L1={r['l1_psnr']:.2f}dB  L0={r['l0_psnr']:.2f}dB  "
                  f"min={r['min_psnr']:.2f}dB  total={r['total_bytes']/1024:.1f}KB")

    # === FAIR COMPARISON: iso-size ===
    print(f"\n{'='*90}")
    print(f"ISO-SIZE COMPARISON: Best method at matched byte counts")
    print(f"{'='*90}")

    # Build interpolation curves
    jpeg_pts = sorted([(r['total_bytes'], r['min_psnr'], r['l1_psnr'], r['l0_psnr']) for r in jpeg_results.values()])
    dz_pts = sorted([(r['total_bytes'], r['min_psnr'], r['l1_psnr'], r['l0_psnr']) for r in dz_results.values()])

    # For each deadzone size point, find closest JPEG size and compare
    print(f"\n{'DZ step':>8s}  {'DZ size':>8s}  {'DZ min':>8s}  {'JPEG q':>7s}  {'JPEG size':>9s}  {'JPEG min':>8s}  {'Δmin':>8s}")
    print("-" * 75)

    for step, r in sorted(dz_results.items()):
        target_size = r['total_bytes']
        # Find closest JPEG quality
        closest_q = min(jpeg_results.keys(),
                       key=lambda q: abs(jpeg_results[q]['total_bytes'] - target_size))
        jr = jpeg_results[closest_q]

        delta = r['min_psnr'] - jr['min_psnr']
        print(f"  s={step:5.1f}  {r['total_bytes']/1024:6.1f}KB  {r['min_psnr']:.2f}dB  "
              f"  q={closest_q:2d}  {jr['total_bytes']/1024:7.1f}KB  {jr['min_psnr']:.2f}dB  "
              f"{delta:+.2f}dB")


if __name__ == "__main__":
    main()
