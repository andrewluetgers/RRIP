#!/usr/bin/env python3
"""
single_image_compress.py

Single-image ORIGAMI compression: downscale to create a prior, upscale back to
produce a prediction, compute a luma-only residual, and store prior + residual.

Also supports a YCbCr subsampling baseline mode that stores full-resolution Y
plus half-resolution Cb and Cr channels independently.

Usage:
  # ORIGAMI mode (default)
  uv run python evals/scripts/single_image_compress.py \
      --image evals/test-images/L0-1024.jpg \
      --prior-quality 70 --residual-quality 50

  # YCbCr subsample baseline
  uv run python evals/scripts/single_image_compress.py \
      --image evals/test-images/L0-1024.jpg \
      --mode ycbcr-subsample \
      --luma-quality 70 --chroma-quality 50
"""

import argparse
import json
import os
import pathlib
import numpy as np
from PIL import Image
from datetime import datetime
from skimage.metrics import structural_similarity as ssim_metric, mean_squared_error
from skimage.color import rgb2lab, deltaE_cie76
from jpeg_encoder import (
    JpegEncoder, encode_jpeg_to_file, encode_jpeg_to_bytes,
    parse_encoder_arg, is_jxl_encoder, is_webp_encoder, decode_jxl_to_image, file_extension,
)

# Try to import lz4 for pack compression
try:
    import lz4.block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

# Try to import sewar for VIF
try:
    from sewar.full_ref import vifp
    HAS_VIF = True
except ImportError:
    HAS_VIF = False

# Try to import lpips for perceptual similarity
try:
    import torch
    import lpips as lpips_lib
    _lpips_net = None
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


# ---------------------------------------------------------------------------
# YCbCr conversion (BT.601, matching Rust implementation)
# ---------------------------------------------------------------------------

def rgb_to_ycbcr_bt601(rgb_u8):
    """Convert RGB to YCbCr using BT.601."""
    rgb = rgb_u8.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0
    return Y, Cb, Cr


def ycbcr_to_rgb_bt601(Y, Cb, Cr):
    """Convert YCbCr to RGB using BT.601."""
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = mean_squared_error(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (RGB or grayscale)."""
    if img1.ndim == 3:
        return float(ssim_metric(img1, img2, channel_axis=2, data_range=255))
    return float(ssim_metric(img1, img2, data_range=255))


def calculate_vif(img1, img2):
    """Calculate Visual Information Fidelity between two images."""
    if not HAS_VIF:
        return None
    try:
        img1_u8 = np.clip(img1, 0, 255).astype(np.uint8)
        img2_u8 = np.clip(img2, 0, 255).astype(np.uint8)
        return float(vifp(img1_u8, img2_u8))
    except Exception as e:
        print(f"VIF calculation error: {e}")
        return None


def calculate_delta_e(img1, img2):
    """Calculate Delta E (CIE76) color difference between two RGB images."""
    try:
        if img1.ndim == 3 and img1.shape[2] == 3:
            img1_norm = np.clip(img1 / 255.0, 0, 1)
            img2_norm = np.clip(img2 / 255.0, 0, 1)
            lab1 = rgb2lab(img1_norm)
            lab2 = rgb2lab(img2_norm)
            return float(np.mean(deltaE_cie76(lab1, lab2)))
        return None
    except Exception as e:
        print(f"Delta E calculation error: {e}")
        return None


def calculate_lpips(img1, img2):
    """Calculate LPIPS perceptual similarity between two RGB images."""
    if not HAS_LPIPS:
        return None
    try:
        global _lpips_net
        if _lpips_net is None:
            _lpips_net = lpips_lib.LPIPS(net='alex', verbose=False)
        img1_t = torch.from_numpy(img1.copy()).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        img2_t = torch.from_numpy(img2.copy()).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            d = _lpips_net(img1_t, img2_t)
        return float(d.item())
    except Exception as e:
        print(f"LPIPS calculation error: {e}")
        return None


def compute_all_metrics(original, reconstructed):
    """Compute all metrics between original and reconstructed RGB images."""
    metrics = {
        "psnr": float(calculate_psnr(original, reconstructed)),
        "ssim": float(calculate_ssim(original, reconstructed)),
        "mse": float(mean_squared_error(original, reconstructed)),
    }
    vif_val = calculate_vif(original, reconstructed)
    if vif_val is not None:
        metrics["vif"] = vif_val
    delta_e_val = calculate_delta_e(original, reconstructed)
    if delta_e_val is not None:
        metrics["delta_e"] = delta_e_val
    lpips_val = calculate_lpips(original, reconstructed)
    if lpips_val is not None:
        metrics["lpips"] = lpips_val
    return metrics


# ---------------------------------------------------------------------------
# Helper: save image with encoder support (JPEG/JXL)
# ---------------------------------------------------------------------------

def save_compressed(image_pil, output_path, quality, encoder, mode="RGB"):
    """Compress and save an image. Returns (actual_path, file_size).
    For JXL, writes .jxl source and .png display copy, returns PNG path."""
    output_path = pathlib.Path(output_path)
    use_jxl = is_jxl_encoder(encoder)

    if use_jxl:
        jxl_path = output_path.with_suffix('.jxl')
        file_size = encode_jpeg_to_file(image_pil, jxl_path, quality, encoder)
        decoded = decode_jxl_to_image(jxl_path)
        png_path = output_path.with_suffix('.png')
        decoded.save(str(png_path), format="PNG")
        return png_path, file_size
    else:
        file_size = encode_jpeg_to_file(image_pil, output_path, quality, encoder)
        return output_path, file_size


def decode_compressed(path, mode="L"):
    """Load a compressed image back as a numpy array."""
    return np.array(Image.open(path).convert(mode))


# ---------------------------------------------------------------------------
# ORIGAMI single-image compression
# ---------------------------------------------------------------------------

def run_origami_mode(img_array, out_dir, prior_quality, residual_quality, encoder, original_file_size):
    """Run ORIGAMI prior+residual compression on a single image."""
    out_dir = pathlib.Path(out_dir)
    compress_dir = out_dir / "compress"
    compress_dir.mkdir(parents=True, exist_ok=True)

    h, w = img_array.shape[:2]
    ext = file_extension(encoder)

    print("\n=== ORIGAMI Single-Image Compression ===")
    print(f"  Image: {w}x{h}, prior_q={prior_quality}, residual_q={residual_quality}")

    images = {}

    # 001: Save original
    orig_path = compress_dir / "001_original.png"
    Image.fromarray(img_array).save(str(orig_path), format="PNG")
    images["original"] = "compress/001_original.png"
    print(f"  Saved: 001_original.png")

    # 002: Downscale to half size (prior)
    prior_h, prior_w = h // 2, w // 2
    prior_pil = Image.fromarray(img_array).resize((prior_w, prior_h), Image.LANCZOS)
    prior_arr = np.array(prior_pil)
    prior_png_path = compress_dir / "002_prior_half.png"
    prior_pil.save(str(prior_png_path), format="PNG")
    images["prior_half"] = "compress/002_prior_half.png"
    print(f"  Saved: 002_prior_half.png ({prior_w}x{prior_h})")

    # 003: Compress prior as RGB JPEG
    prior_jpg_name = f"003_prior_compressed{ext}"
    prior_jpg_path = compress_dir / prior_jpg_name
    prior_saved_path, prior_size = save_compressed(prior_pil, prior_jpg_path, prior_quality, encoder, mode="RGB")
    images["prior_compressed"] = f"compress/{prior_saved_path.name}"
    print(f"  Saved: {prior_saved_path.name} ({prior_size:,} bytes)")

    # 004: Decode prior back (lossy roundtrip)
    if is_jxl_encoder(encoder):
        prior_decoded_pil = decode_jxl_to_image(compress_dir / f"003_prior_compressed.jxl")
    else:
        prior_decoded_pil = Image.open(prior_saved_path).convert("RGB")
    prior_decoded_arr = np.array(prior_decoded_pil)
    prior_decoded_png = compress_dir / "004_prior_decoded.png"
    prior_decoded_pil.save(str(prior_decoded_png), format="PNG")
    images["prior_decoded"] = "compress/004_prior_decoded.png"
    print(f"  Saved: 004_prior_decoded.png")

    # 005: Upscale decoded prior to full size (bilinear prediction)
    prediction_pil = prior_decoded_pil.resize((w, h), Image.Resampling.BILINEAR)
    prediction_arr = np.array(prediction_pil)
    prediction_png = compress_dir / "005_prediction_upscaled.png"
    prediction_pil.save(str(prediction_png), format="PNG")
    images["prediction"] = "compress/005_prediction_upscaled.png"
    print(f"  Saved: 005_prediction_upscaled.png")

    # 006: Y channel of original
    Y_orig, Cb_orig, Cr_orig = rgb_to_ycbcr_bt601(img_array)
    y_orig_png = compress_dir / "006_Y_original.png"
    Image.fromarray(np.clip(Y_orig, 0, 255).astype(np.uint8), mode="L").save(str(y_orig_png))
    images["Y_original"] = "compress/006_Y_original.png"
    print(f"  Saved: 006_Y_original.png")

    # 007: Y channel of prediction
    Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(prediction_arr)
    y_pred_png = compress_dir / "007_Y_prediction.png"
    Image.fromarray(np.clip(Y_pred, 0, 255).astype(np.uint8), mode="L").save(str(y_pred_png))
    images["Y_prediction"] = "compress/007_Y_prediction.png"
    print(f"  Saved: 007_Y_prediction.png")

    # 008: Raw residual (normalized visualization)
    residual_raw = Y_orig - Y_pred
    residual_raw_norm = ((residual_raw - residual_raw.min()) /
                         (residual_raw.max() - residual_raw.min() + 1e-8) * 255).astype(np.uint8)
    res_raw_png = compress_dir / "008_residual_raw.png"
    Image.fromarray(residual_raw_norm, mode="L").save(str(res_raw_png))
    images["residual_raw"] = "compress/008_residual_raw.png"
    print(f"  Saved: 008_residual_raw.png (range: [{residual_raw.min():.1f}, {residual_raw.max():.1f}])")

    # 009: Centered residual
    residual_centered = np.clip(np.round(residual_raw + 128.0), 0, 255).astype(np.uint8)
    res_cent_png = compress_dir / "009_residual_centered.png"
    Image.fromarray(residual_centered, mode="L").save(str(res_cent_png))
    images["residual_centered"] = "compress/009_residual_centered.png"
    print(f"  Saved: 009_residual_centered.png")

    # 010: Compress residual as grayscale JPEG
    residual_pil = Image.fromarray(residual_centered, mode="L")
    res_jpg_name = f"010_residual_compressed{ext}"
    res_jpg_path = compress_dir / res_jpg_name
    res_saved_path, residual_size = save_compressed(residual_pil, res_jpg_path, residual_quality, encoder, mode="L")
    images["residual_compressed"] = f"compress/{res_saved_path.name}"
    print(f"  Saved: {res_saved_path.name} ({residual_size:,} bytes)")

    # 011: Decode residual
    if is_jxl_encoder(encoder):
        residual_decoded_arr = np.array(decode_jxl_to_image(compress_dir / f"010_residual_compressed.jxl").convert("L")).astype(np.float32)
    else:
        residual_decoded_arr = np.array(Image.open(res_saved_path).convert("L")).astype(np.float32)
    res_dec_png = compress_dir / "011_residual_decoded.png"
    Image.fromarray(np.clip(residual_decoded_arr, 0, 255).astype(np.uint8), mode="L").save(str(res_dec_png))
    images["residual_decoded"] = "compress/011_residual_decoded.png"
    print(f"  Saved: 011_residual_decoded.png")

    # 012: Reconstruct Y
    Y_recon = np.clip(Y_pred + (residual_decoded_arr - 128.0), 0, 255)
    y_recon_png = compress_dir / "012_Y_reconstructed.png"
    Image.fromarray(np.clip(Y_recon, 0, 255).astype(np.uint8), mode="L").save(str(y_recon_png))
    images["Y_reconstructed"] = "compress/012_Y_reconstructed.png"
    print(f"  Saved: 012_Y_reconstructed.png")

    # 013: Final RGB reconstruction (Y_recon + predicted Cb/Cr)
    reconstructed = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
    recon_png = compress_dir / "013_reconstructed_rgb.png"
    Image.fromarray(reconstructed).save(str(recon_png), format="PNG")
    images["reconstructed"] = "compress/013_reconstructed_rgb.png"
    print(f"  Saved: 013_reconstructed_rgb.png")

    # Compute metrics
    print("\n--- Metrics ---")
    recon_metrics = compute_all_metrics(img_array, reconstructed)
    print(f"  Reconstruction: PSNR={recon_metrics['psnr']:.2f} dB, SSIM={recon_metrics['ssim']:.4f}")

    pred_metrics = compute_all_metrics(img_array, prediction_arr)
    print(f"  Prediction only: PSNR={pred_metrics['psnr']:.2f} dB, SSIM={pred_metrics['ssim']:.4f}")

    # Sizes
    total_compressed = prior_size + residual_size
    compression_ratio = original_file_size / total_compressed if total_compressed > 0 else 0
    space_savings = 100 * (1 - total_compressed / original_file_size) if original_file_size > 0 else 0

    print(f"\n--- Sizes ---")
    print(f"  Original file: {original_file_size:,} bytes")
    print(f"  Prior: {prior_size:,} bytes")
    print(f"  Residual: {residual_size:,} bytes")
    print(f"  Total: {total_compressed:,} bytes")
    print(f"  Ratio: {compression_ratio:.2f}x, Savings: {space_savings:.1f}%")

    # LZ4 pack
    lz4_packed_size = None
    if HAS_LZ4:
        prior_bytes = pathlib.Path(prior_saved_path).read_bytes() if not is_jxl_encoder(encoder) else (compress_dir / "003_prior_compressed.jxl").read_bytes()
        residual_bytes = pathlib.Path(res_saved_path).read_bytes() if not is_jxl_encoder(encoder) else (compress_dir / "010_residual_compressed.jxl").read_bytes()
        raw_pack = prior_bytes + residual_bytes
        compressed_pack = len(raw_pack).to_bytes(4, 'little') + lz4.block.compress(
            raw_pack, mode='fast', compression=0, store_size=False
        )
        pac_path = out_dir / "pair.pac"
        pac_path.write_bytes(compressed_pack)
        lz4_packed_size = len(compressed_pack)
        print(f"  LZ4 pack: {lz4_packed_size:,} bytes")

    # Build manifest
    sizes = {
        "original_file_size": original_file_size,
        "prior_compressed_size": prior_size,
        "residual_compressed_size": residual_size,
        "total_compressed_size": total_compressed,
        "compression_ratio": round(compression_ratio, 2),
        "space_savings_pct": round(space_savings, 1),
    }
    if lz4_packed_size is not None:
        sizes["lz4_packed_size"] = lz4_packed_size

    manifest = {
        "type": "single_origami",
        "configuration": {
            "input_image": str(args.image),
            "image_dimensions": [w, h],
            "prior_quality": prior_quality,
            "residual_quality": residual_quality,
            "encoder": encoder.value,
        },
        "sizes": sizes,
        "images": images,
        "metrics": {
            "reconstruction": recon_metrics,
            "prediction_only": pred_metrics,
        },
        "data_ranges": {
            "residual_raw": {
                "min": float(residual_raw.min()),
                "max": float(residual_raw.max()),
                "mean": float(residual_raw.mean()),
                "std": float(residual_raw.std()),
            }
        },
    }

    return manifest


# ---------------------------------------------------------------------------
# YCbCr subsampling baseline
# ---------------------------------------------------------------------------

def run_ycbcr_mode(img_array, out_dir, luma_quality, chroma_quality, encoder, original_file_size):
    """Run YCbCr subsampling baseline compression."""
    out_dir = pathlib.Path(out_dir)
    compress_dir = out_dir / "compress"
    compress_dir.mkdir(parents=True, exist_ok=True)

    h, w = img_array.shape[:2]
    ext = file_extension(encoder)

    print("\n=== YCbCr Subsampling Baseline ===")
    print(f"  Image: {w}x{h}, luma_q={luma_quality}, chroma_q={chroma_quality}")

    images = {}

    # 001: Save original
    orig_path = compress_dir / "001_original.png"
    Image.fromarray(img_array).save(str(orig_path), format="PNG")
    images["original"] = "compress/001_original.png"
    print(f"  Saved: 001_original.png")

    # Convert to YCbCr
    Y, Cb, Cr = rgb_to_ycbcr_bt601(img_array)
    Y_u8 = np.clip(Y, 0, 255).astype(np.uint8)
    Cb_u8 = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr_u8 = np.clip(Cr, 0, 255).astype(np.uint8)

    # 002: Y original
    y_png = compress_dir / "002_Y_original.png"
    Image.fromarray(Y_u8, mode="L").save(str(y_png))
    images["Y_original"] = "compress/002_Y_original.png"
    print(f"  Saved: 002_Y_original.png")

    # 003: Compress Y (full resolution) as grayscale JPEG
    y_pil = Image.fromarray(Y_u8, mode="L")
    y_jpg_path = compress_dir / f"003_Y_compressed{ext}"
    y_saved_path, y_size = save_compressed(y_pil, y_jpg_path, luma_quality, encoder, mode="L")
    images["Y_compressed"] = f"compress/{y_saved_path.name}"
    print(f"  Saved: {y_saved_path.name} ({y_size:,} bytes)")

    # Downscale Cb, Cr to half size
    half_h, half_w = h // 2, w // 2

    # 004: Cb original at half size
    Cb_half_pil = Image.fromarray(Cb_u8, mode="L").resize((half_w, half_h), Image.LANCZOS)
    cb_half_png = compress_dir / "004_Cb_original_half.png"
    Cb_half_pil.save(str(cb_half_png))
    images["Cb_original_half"] = "compress/004_Cb_original_half.png"
    print(f"  Saved: 004_Cb_original_half.png ({half_w}x{half_h})")

    # 005: Compress Cb
    cb_jpg_path = compress_dir / f"005_Cb_compressed{ext}"
    cb_saved_path, cb_size = save_compressed(Cb_half_pil, cb_jpg_path, chroma_quality, encoder, mode="L")
    images["Cb_compressed"] = f"compress/{cb_saved_path.name}"
    print(f"  Saved: {cb_saved_path.name} ({cb_size:,} bytes)")

    # 006: Cr original at half size
    Cr_half_pil = Image.fromarray(Cr_u8, mode="L").resize((half_w, half_h), Image.LANCZOS)
    cr_half_png = compress_dir / "006_Cr_original_half.png"
    Cr_half_pil.save(str(cr_half_png))
    images["Cr_original_half"] = "compress/006_Cr_original_half.png"
    print(f"  Saved: 006_Cr_original_half.png ({half_w}x{half_h})")

    # 007: Compress Cr
    cr_jpg_path = compress_dir / f"007_Cr_compressed{ext}"
    cr_saved_path, cr_size = save_compressed(Cr_half_pil, cr_jpg_path, chroma_quality, encoder, mode="L")
    images["Cr_compressed"] = f"compress/{cr_saved_path.name}"
    print(f"  Saved: {cr_saved_path.name} ({cr_size:,} bytes)")

    # Reconstruct
    # Decode Y full-res
    if is_jxl_encoder(encoder):
        Y_dec = np.array(decode_jxl_to_image(compress_dir / f"003_Y_compressed.jxl").convert("L")).astype(np.float32)
    else:
        Y_dec = np.array(Image.open(y_saved_path).convert("L")).astype(np.float32)

    # 008: Y decoded
    y_dec_png = compress_dir / "008_Y_decoded.png"
    Image.fromarray(np.clip(Y_dec, 0, 255).astype(np.uint8), mode="L").save(str(y_dec_png))
    images["Y_decoded"] = "compress/008_Y_decoded.png"
    print(f"  Saved: 008_Y_decoded.png")

    # Decode Cb half-res, upscale
    if is_jxl_encoder(encoder):
        Cb_dec_half = np.array(decode_jxl_to_image(compress_dir / f"005_Cb_compressed.jxl").convert("L"))
    else:
        Cb_dec_half = np.array(Image.open(cb_saved_path).convert("L"))
    Cb_dec = np.array(Image.fromarray(Cb_dec_half, mode="L").resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32)

    # 009: Cb decoded upscaled
    cb_dec_png = compress_dir / "009_Cb_decoded_upscaled.png"
    Image.fromarray(np.clip(Cb_dec, 0, 255).astype(np.uint8), mode="L").save(str(cb_dec_png))
    images["Cb_decoded_upscaled"] = "compress/009_Cb_decoded_upscaled.png"
    print(f"  Saved: 009_Cb_decoded_upscaled.png")

    # Decode Cr half-res, upscale
    if is_jxl_encoder(encoder):
        Cr_dec_half = np.array(decode_jxl_to_image(compress_dir / f"007_Cr_compressed.jxl").convert("L"))
    else:
        Cr_dec_half = np.array(Image.open(cr_saved_path).convert("L"))
    Cr_dec = np.array(Image.fromarray(Cr_dec_half, mode="L").resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32)

    # 010: Cr decoded upscaled
    cr_dec_png = compress_dir / "010_Cr_decoded_upscaled.png"
    Image.fromarray(np.clip(Cr_dec, 0, 255).astype(np.uint8), mode="L").save(str(cr_dec_png))
    images["Cr_decoded_upscaled"] = "compress/010_Cr_decoded_upscaled.png"
    print(f"  Saved: 010_Cr_decoded_upscaled.png")

    # Convert back to RGB
    reconstructed = ycbcr_to_rgb_bt601(Y_dec, Cb_dec, Cr_dec)

    # 011: Reconstructed RGB
    recon_png = compress_dir / "011_reconstructed_rgb.png"
    Image.fromarray(reconstructed).save(str(recon_png), format="PNG")
    images["reconstructed"] = "compress/011_reconstructed_rgb.png"
    print(f"  Saved: 011_reconstructed_rgb.png")

    # Metrics
    print("\n--- Metrics ---")
    recon_metrics = compute_all_metrics(img_array, reconstructed)
    print(f"  Reconstruction: PSNR={recon_metrics['psnr']:.2f} dB, SSIM={recon_metrics['ssim']:.4f}")

    # Sizes
    total_compressed = y_size + cb_size + cr_size
    compression_ratio = original_file_size / total_compressed if total_compressed > 0 else 0
    space_savings = 100 * (1 - total_compressed / original_file_size) if original_file_size > 0 else 0

    print(f"\n--- Sizes ---")
    print(f"  Original file: {original_file_size:,} bytes")
    print(f"  Y: {y_size:,} bytes")
    print(f"  Cb: {cb_size:,} bytes")
    print(f"  Cr: {cr_size:,} bytes")
    print(f"  Total: {total_compressed:,} bytes")
    print(f"  Ratio: {compression_ratio:.2f}x, Savings: {space_savings:.1f}%")

    manifest = {
        "type": "single_ycbcr",
        "configuration": {
            "input_image": str(args.image),
            "image_dimensions": [w, h],
            "luma_quality": luma_quality,
            "chroma_quality": chroma_quality,
            "encoder": encoder.value,
        },
        "sizes": {
            "original_file_size": original_file_size,
            "y_compressed_size": y_size,
            "cb_compressed_size": cb_size,
            "cr_compressed_size": cr_size,
            "total_compressed_size": total_compressed,
            "compression_ratio": round(compression_ratio, 2),
            "space_savings_pct": round(space_savings, 1),
        },
        "images": images,
        "metrics": {
            "reconstruction": recon_metrics,
        },
    }

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global args
    parser = argparse.ArgumentParser(
        description="Single-image ORIGAMI compression or YCbCr subsampling baseline"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mode", choices=["origami", "ycbcr-subsample"], default="origami",
                        help="Compression mode (default: origami)")
    parser.add_argument("--output", help="Output directory (auto-generated if not specified)")
    parser.add_argument("--encoder", default="libjpeg-turbo",
                        help="Encoder: libjpeg-turbo, jpegli, mozjpeg, jpegxl, or webp")

    # ORIGAMI mode args
    parser.add_argument("--prior-quality", type=int, default=70,
                        help="JPEG quality for prior (origami mode)")
    parser.add_argument("--residual-quality", type=int, default=50,
                        help="JPEG quality for residual (origami mode)")

    # YCbCr mode args
    parser.add_argument("--luma-quality", type=int, default=70,
                        help="JPEG quality for Y channel (ycbcr-subsample mode)")
    parser.add_argument("--chroma-quality", type=int, default=50,
                        help="JPEG quality for Cb/Cr channels (ycbcr-subsample mode)")

    args = parser.parse_args()
    encoder = parse_encoder_arg(args.encoder)

    # Encoder prefix for directory naming
    if encoder == JpegEncoder.LIBJPEG_TURBO:
        encoder_prefix = ""
    else:
        encoder_prefix = encoder.value + "_"

    # Auto-generate output directory
    if args.output is None:
        if args.mode == "origami":
            args.output = f"evals/runs/{encoder_prefix}single_origami_p{args.prior_quality}_r{args.residual_quality}"
        else:
            args.output = f"evals/runs/{encoder_prefix}single_ycbcr_y{args.luma_quality}_c{args.chroma_quality}"

    print(f"Mode: {args.mode}")
    print(f"Encoder: {encoder.value}")
    print(f"Output: {args.output}")

    # Load image
    img = Image.open(args.image).convert("RGB")
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # Ensure even dimensions
    if h % 2 != 0:
        h -= 1
    if w % 2 != 0:
        w -= 1
    if h != img_array.shape[0] or w != img_array.shape[1]:
        img_array = img_array[:h, :w]
        print(f"  Trimmed to even dimensions: {w}x{h}")

    # Original file size
    original_file_size = os.path.getsize(args.image)

    # Create output directory
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run compression
    if args.mode == "origami":
        manifest = run_origami_mode(
            img_array, out_dir,
            args.prior_quality, args.residual_quality,
            encoder, original_file_size,
        )
    else:
        manifest = run_ycbcr_mode(
            img_array, out_dir,
            args.luma_quality, args.chroma_quality,
            encoder, original_file_size,
        )

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved: {manifest_path}")

    # Write summary
    summary_path = out_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        if args.mode == "origami":
            f.write(f"ORIGAMI SINGLE-IMAGE COMPRESSION\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input Image: {args.image}\n")
            f.write(f"Dimensions: {w}x{h}\n")
            f.write(f"Encoder: {encoder.value}\n")
            f.write(f"Prior Quality: {args.prior_quality}\n")
            f.write(f"Residual Quality: {args.residual_quality}\n\n")
            f.write(f"SIZES\n")
            f.write("-" * 30 + "\n")
            s = manifest["sizes"]
            f.write(f"Original: {s['original_file_size']:,} bytes\n")
            f.write(f"Prior: {s['prior_compressed_size']:,} bytes\n")
            f.write(f"Residual: {s['residual_compressed_size']:,} bytes\n")
            f.write(f"Total: {s['total_compressed_size']:,} bytes\n")
            f.write(f"Ratio: {s['compression_ratio']:.2f}x\n")
            f.write(f"Savings: {s['space_savings_pct']:.1f}%\n\n")
            f.write(f"METRICS\n")
            f.write("-" * 30 + "\n")
            m = manifest["metrics"]["reconstruction"]
            f.write(f"PSNR: {m['psnr']:.2f} dB\n")
            f.write(f"SSIM: {m['ssim']:.4f}\n")
            if 'vif' in m:
                f.write(f"VIF: {m['vif']:.4f}\n")
            if 'delta_e' in m:
                f.write(f"Delta E: {m['delta_e']:.2f}\n")
            if 'lpips' in m:
                f.write(f"LPIPS: {m['lpips']:.4f}\n")
            f.write(f"\nPrediction-only PSNR: {manifest['metrics']['prediction_only']['psnr']:.2f} dB\n")
            f.write(f"Prediction-only SSIM: {manifest['metrics']['prediction_only']['ssim']:.4f}\n")
        else:
            f.write(f"YCbCr SUBSAMPLING BASELINE\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input Image: {args.image}\n")
            f.write(f"Dimensions: {w}x{h}\n")
            f.write(f"Encoder: {encoder.value}\n")
            f.write(f"Luma Quality: {args.luma_quality}\n")
            f.write(f"Chroma Quality: {args.chroma_quality}\n\n")
            f.write(f"SIZES\n")
            f.write("-" * 30 + "\n")
            s = manifest["sizes"]
            f.write(f"Original: {s['original_file_size']:,} bytes\n")
            f.write(f"Y: {s['y_compressed_size']:,} bytes\n")
            f.write(f"Cb: {s['cb_compressed_size']:,} bytes\n")
            f.write(f"Cr: {s['cr_compressed_size']:,} bytes\n")
            f.write(f"Total: {s['total_compressed_size']:,} bytes\n")
            f.write(f"Ratio: {s['compression_ratio']:.2f}x\n")
            f.write(f"Savings: {s['space_savings_pct']:.1f}%\n\n")
            f.write(f"METRICS\n")
            f.write("-" * 30 + "\n")
            m = manifest["metrics"]["reconstruction"]
            f.write(f"PSNR: {m['psnr']:.2f} dB\n")
            f.write(f"SSIM: {m['ssim']:.4f}\n")
            if 'vif' in m:
                f.write(f"VIF: {m['vif']:.4f}\n")
            if 'delta_e' in m:
                f.write(f"Delta E: {m['delta_e']:.2f}\n")
            if 'lpips' in m:
                f.write(f"LPIPS: {m['lpips']:.4f}\n")

    print(f"Summary saved: {summary_path}")

    # Final summary
    print("\n" + "=" * 50)
    s = manifest["sizes"]
    print(f"Total compressed: {s['total_compressed_size']:,} bytes")
    print(f"Compression ratio: {s['compression_ratio']:.2f}x")
    m = manifest["metrics"]["reconstruction"]
    print(f"PSNR: {m['psnr']:.2f} dB")
    print(f"SSIM: {m['ssim']:.4f}")


if __name__ == "__main__":
    main()
