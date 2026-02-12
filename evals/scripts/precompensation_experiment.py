#!/usr/bin/env python3
"""
precompensation_experiment.py

Tests whether pre-compensating residuals before JPEG encoding can improve
reconstruction quality by counteracting JPEG's systematic attenuation bias.

The idea: JPEG quantization systematically attenuates residuals toward zero
(correlation(R, E) = 0.41). By scaling residuals up before encoding,
we can partially counteract this shrinkage.

We test two strategies:
1. Linear scaling: R_enc = clip(128 + (R * scale_factor), 0, 255)
   Decode: R_dec = (decoded - 128) / scale_factor
2. Iterative correction: Encode R, decode to get R_q, compute E = R - R_q,
   then encode R + alpha*E as the corrected residual.

Usage:
  python precompensation_experiment.py --image evals/test-images/L0-1024.jpg --quality 40
"""
import argparse
import numpy as np
from PIL import Image
import io
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from jpeg_encoder import JpegEncoder, encode_jpeg_to_bytes, parse_encoder_arg


def encode_decode_gray(pixels_u8, quality, encoder=JpegEncoder.LIBJPEG_TURBO):
    """Encode grayscale pixels as JPEG and decode back. Returns (decoded_u8, size_bytes)."""
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


def run_experiment(image_path, quality, encoder_name="libjpeg-turbo"):
    """Run the full precompensation experiment."""
    encoder = parse_encoder_arg(encoder_name)

    # Load and tile image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    tile_size = 256

    # Create L2 tile (downsampled 4x)
    l2 = np.array(Image.fromarray(img_array[:1024, :1024]).resize((256, 256), Image.LANCZOS))

    # Create L1 tiles (downsampled 2x)
    l1_source = np.array(Image.fromarray(img_array[:1024, :1024]).resize((512, 512), Image.LANCZOS))

    # Create L0 tiles (original resolution)
    l0_tiles = {}
    for dy in range(4):
        for dx in range(4):
            l0_tiles[(dx, dy)] = img_array[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    l1_tiles = {}
    for dy in range(2):
        for dx in range(2):
            l1_tiles[(dx, dy)] = l1_source[dy*256:(dy+1)*256, dx*256:(dx+1)*256]

    UPSAMPLE = Image.Resampling.BILINEAR

    # --- Strategy 0: Baseline (no precompensation) ---
    print(f"\n{'='*60}")
    print(f"PRECOMPENSATION EXPERIMENT — q={quality}, encoder={encoder_name}")
    print(f"{'='*60}")

    # Build L1 prediction from L2
    l1_pred_mosaic = np.array(Image.fromarray(l2).resize((512, 512), resample=UPSAMPLE))

    strategies = {}

    for strategy_name, params in [
        ("baseline", {"scale": 1.0, "iterations": 0}),
        ("scale_1.1", {"scale": 1.1, "iterations": 0}),
        ("scale_1.2", {"scale": 1.2, "iterations": 0}),
        ("scale_1.3", {"scale": 1.3, "iterations": 0}),
        ("scale_1.5", {"scale": 1.5, "iterations": 0}),
        ("iterative_1", {"scale": 1.0, "iterations": 1}),
        ("iterative_2", {"scale": 1.0, "iterations": 2}),
        ("iterative_3", {"scale": 1.0, "iterations": 3}),
    ]:
        scale = params["scale"]
        iterations = params["iterations"]

        l1_psnrs = []
        l0_psnrs = []
        total_l1_bytes = 0
        total_l0_bytes = 0
        l1_reconstructed = {}

        # Process L1 tiles
        for (dx, dy), l1_gt in l1_tiles.items():
            pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)

            residual_raw = Y_gt - Y_pred  # float, centered at 0

            if iterations > 0:
                # Iterative correction: encode, measure error, add correction
                current_residual = residual_raw.copy()
                for it in range(iterations):
                    centered = np.clip(np.round(current_residual + 128.0), 0, 255).astype(np.uint8)
                    decoded, _ = encode_decode_gray(centered, quality, encoder)
                    decoded_residual = decoded.astype(np.float32) - 128.0
                    error = residual_raw - decoded_residual
                    current_residual = residual_raw + error  # Add correction

                # Final encode with corrected residual
                centered = np.clip(np.round(current_residual + 128.0), 0, 255).astype(np.uint8)
                decoded, size = encode_decode_gray(centered, quality, encoder)
                Y_recon = np.clip(Y_pred + (decoded.astype(np.float32) - 128.0), 0, 255)
            else:
                # Linear scaling
                scaled_residual = residual_raw * scale
                centered = np.clip(np.round(scaled_residual + 128.0), 0, 255).astype(np.uint8)
                decoded, size = encode_decode_gray(centered, quality, encoder)
                # Undo scale on decode
                decoded_residual = (decoded.astype(np.float32) - 128.0) / scale
                Y_recon = np.clip(Y_pred + decoded_residual, 0, 255)

            total_l1_bytes += size
            rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
            l1_reconstructed[(dx, dy)] = rgb_recon
            l1_psnrs.append(psnr(l1_gt, rgb_recon))

        # Build L1 mosaic for L0 prediction
        l1_mosaic = np.zeros((512, 512, 3), dtype=np.uint8)
        for dy in range(2):
            for dx in range(2):
                l1_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256] = l1_reconstructed[(dx, dy)]

        l0_pred_mosaic = np.array(Image.fromarray(l1_mosaic).resize((1024, 1024), resample=UPSAMPLE))

        # Process L0 tiles
        for (dx, dy), l0_gt in l0_tiles.items():
            pred = l0_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(l0_gt)

            residual_raw = Y_gt - Y_pred

            if iterations > 0:
                current_residual = residual_raw.copy()
                for it in range(iterations):
                    centered = np.clip(np.round(current_residual + 128.0), 0, 255).astype(np.uint8)
                    decoded, _ = encode_decode_gray(centered, quality, encoder)
                    decoded_residual = decoded.astype(np.float32) - 128.0
                    error = residual_raw - decoded_residual
                    current_residual = residual_raw + error

                centered = np.clip(np.round(current_residual + 128.0), 0, 255).astype(np.uint8)
                decoded, size = encode_decode_gray(centered, quality, encoder)
                Y_recon = np.clip(Y_pred + (decoded.astype(np.float32) - 128.0), 0, 255)
            else:
                scaled_residual = residual_raw * scale
                centered = np.clip(np.round(scaled_residual + 128.0), 0, 255).astype(np.uint8)
                decoded, size = encode_decode_gray(centered, quality, encoder)
                decoded_residual = (decoded.astype(np.float32) - 128.0) / scale
                Y_recon = np.clip(Y_pred + decoded_residual, 0, 255)

            total_l0_bytes += size
            rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
            l0_psnrs.append(psnr(l0_gt, rgb_recon))

        total_bytes = total_l1_bytes + total_l0_bytes
        l1_avg = np.mean(l1_psnrs)
        l0_avg = np.mean(l0_psnrs)
        min_psnr = min(min(l1_psnrs), min(l0_psnrs))

        strategies[strategy_name] = {
            "l1_psnr": l1_avg,
            "l0_psnr": l0_avg,
            "min_psnr": min_psnr,
            "total_bytes": total_bytes,
            "l1_bytes": total_l1_bytes,
            "l0_bytes": total_l0_bytes,
        }

    # Print results
    print(f"\n{'Strategy':<20} {'L1 PSNR':>10} {'L0 PSNR':>10} {'Min PSNR':>10} {'Total KB':>10} {'L1 KB':>8} {'L0 KB':>8}")
    print("-" * 86)

    baseline = strategies["baseline"]
    for name, s in strategies.items():
        l1_delta = s["l1_psnr"] - baseline["l1_psnr"]
        l0_delta = s["l0_psnr"] - baseline["l0_psnr"]
        min_delta = s["min_psnr"] - baseline["min_psnr"]
        size_pct = (s["total_bytes"] - baseline["total_bytes"]) / baseline["total_bytes"] * 100

        print(f"{name:<20} {s['l1_psnr']:>8.2f}dB {s['l0_psnr']:>8.2f}dB {s['min_psnr']:>8.2f}dB "
              f"{s['total_bytes']/1024:>8.1f}K {s['l1_bytes']/1024:>6.1f}K {s['l0_bytes']/1024:>6.1f}K")

    print(f"\n{'Strategy':<20} {'ΔL1':>10} {'ΔL0':>10} {'ΔMin':>10} {'Δ Size':>10}")
    print("-" * 60)
    for name, s in strategies.items():
        l1_delta = s["l1_psnr"] - baseline["l1_psnr"]
        l0_delta = s["l0_psnr"] - baseline["l0_psnr"]
        min_delta = s["min_psnr"] - baseline["min_psnr"]
        size_pct = (s["total_bytes"] - baseline["total_bytes"]) / baseline["total_bytes"] * 100

        print(f"{name:<20} {l1_delta:>+8.2f}dB {l0_delta:>+8.2f}dB {min_delta:>+8.2f}dB {size_pct:>+8.1f}%")

    return strategies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test residual precompensation strategies")
    parser.add_argument("--image", required=True, help="Path to 1024x1024 test image")
    parser.add_argument("--quality", type=int, default=40, help="JPEG quality for residuals")
    parser.add_argument("--encoder", default="libjpeg-turbo", help="Encoder to test")
    args = parser.parse_args()

    # Test at the specified quality
    strategies = run_experiment(args.image, args.quality, args.encoder)

    # Also test at q=60 for comparison
    if args.quality != 60:
        strategies_60 = run_experiment(args.image, 60, args.encoder)
