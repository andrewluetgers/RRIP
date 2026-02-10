#!/usr/bin/env python3
"""
wsi_residual_debug_with_manifest.py

Enhanced debug version that saves baseline JPEG tiles at quality 95 and creates
a detailed manifest with statistics at each stage.

Usage:
  python wsi_residual_debug_with_manifest.py --image evals/test-images/L0-1024.jpg --out debug_output --tile 256 --resq 75
"""
import argparse
import pathlib
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_cie76
from jpeg_encoder import (
    JpegEncoder, encode_jpeg_to_file, encode_jpeg_to_bytes,
    parse_encoder_arg, is_jxl_encoder, decode_jxl_to_image, file_extension,
)

try:
    from sewar.full_ref import vifp
    HAS_VIF = True
except ImportError:
    HAS_VIF = False
    print("Warning: VIF metric not available. Install with: pip install sewar")

# Try to import lpips for perceptual similarity
try:
    import torch
    import lpips as lpips_lib
    _lpips_net = None
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips/torch not available, LPIPS metrics will be skipped")


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def calculate_vif(img1, img2):
    """Calculate Visual Information Fidelity between two images."""
    if not HAS_VIF:
        return None
    try:
        # VIF expects uint8 images
        img1_uint8 = np.clip(img1, 0, 255).astype(np.uint8)
        img2_uint8 = np.clip(img2, 0, 255).astype(np.uint8)

        if img1_uint8.ndim == 3 and img1_uint8.shape[2] == 3:
            return float(vifp(img1_uint8, img2_uint8))
        else:
            # For grayscale, expand to 3 channels
            img1_3ch = np.stack([img1_uint8, img1_uint8, img1_uint8], axis=2)
            img2_3ch = np.stack([img2_uint8, img2_uint8, img2_uint8], axis=2)
            return float(vifp(img1_3ch, img2_3ch))
    except Exception as e:
        print(f"VIF calculation error: {e}")
        return None

def calculate_delta_e(img1, img2):
    """Calculate Delta E (CIE76) color difference between two images."""
    try:
        # Only for RGB images
        if img1.ndim == 3 and img1.shape[2] == 3 and img2.ndim == 3 and img2.shape[2] == 3:
            # Convert to LAB color space (expects values in [0,1] range)
            img1_norm = np.clip(img1 / 255.0, 0, 1)
            img2_norm = np.clip(img2 / 255.0, 0, 1)

            lab1 = rgb2lab(img1_norm)
            lab2 = rgb2lab(img2_norm)

            # Calculate per-pixel Delta E and return mean
            delta_e = deltaE_cie76(lab1, lab2)
            return float(np.mean(delta_e))
        else:
            # For grayscale, Delta E doesn't make sense
            return None
    except Exception as e:
        print(f"Delta E calculation error: {e}")
        return None

def calculate_lpips(img1, img2):
    """Calculate LPIPS perceptual similarity between two RGB images.
    Returns a float where lower = more similar (0 = identical)."""
    if not HAS_LPIPS:
        return None
    try:
        global _lpips_net
        if _lpips_net is None:
            _lpips_net = lpips_lib.LPIPS(net='alex', verbose=False)
        # LPIPS expects tensors in [-1, 1] range, shape (N, 3, H, W)
        img1_t = torch.from_numpy(img1.copy()).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        img2_t = torch.from_numpy(img2.copy()).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            d = _lpips_net(img1_t, img2_t)
        return float(d.item())
    except Exception as e:
        print(f"LPIPS calculation error: {e}")
        return None

def get_file_size(path):
    """Get file size in bytes."""
    return os.path.getsize(path) if os.path.exists(path) else 0

def save_image_with_stats(arr, path, step_num, name_suffix, normalize=False,
                          quality=None, reference=None, manifest_entry=None,
                          encoder=JpegEncoder.LIBJPEG_TURBO):
    """Save an image and return statistics about it."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Prepare the image
    if arr.ndim == 3 and arr.shape[2] == 3:
        # RGB image
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        is_grayscale = False
    else:
        # Grayscale
        if normalize:
            # Normalize to 0-255 range for visualization
            arr_display = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255)
            arr_norm = arr_display.astype(np.uint8)
        else:
            arr_norm = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_norm, mode="L")
        is_grayscale = True

    # Determine filename and format
    filename = f"{step_num:03d}_{name_suffix}"
    if not (filename.endswith('.jpg') or filename.endswith('.png')):
        filename += '.png'

    # Save with appropriate quality
    full_path = path / filename
    use_jxl = is_jxl_encoder(encoder)

    if filename.endswith('.jpg'):
        if use_jxl:
            # Encode as JXL, keep source, decode to PNG (lossless) for viewer
            jxl_filename = filename.replace('.jpg', '.jxl')
            jxl_path = path / jxl_filename
            jxl_size = encode_jpeg_to_file(img, jxl_path, quality or 75, encoder)
            decoded_img = decode_jxl_to_image(jxl_path)
            png_path = path / filename.replace('.jpg', '.png')
            decoded_img.save(str(png_path), format="PNG")
            # Keep .jxl source; viewer/reconstruction uses .png
            encoded_size = jxl_size
            full_path = png_path  # return png path for reconstruction
            filename = filename.replace('.jpg', '.png')
        else:
            encode_jpeg_to_file(img, full_path, quality or 75, encoder)
            encoded_size = get_file_size(full_path)
    else:
        img.save(full_path, format="PNG")
        encoded_size = get_file_size(full_path)

    # Calculate statistics
    stats = {
        "filename": filename,
        "file_size": encoded_size,
        "dimensions": f"{arr.shape[1]}x{arr.shape[0]}",
        "is_grayscale": is_grayscale,
        "data_range": {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }
    }

    # If we have a reference image, calculate quality metrics
    if reference is not None:
        if arr.ndim == 3 and reference.ndim == 3:
            # For RGB images, calculate metrics on luminance channel for some metrics
            Y_arr = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
            Y_ref = 0.299 * reference[:,:,0] + 0.587 * reference[:,:,1] + 0.114 * reference[:,:,2]

            metrics = {
                "psnr": float(psnr(Y_ref, Y_arr, data_range=255)),
                "ssim": float(ssim(Y_ref, Y_arr, data_range=255)),
                "mse": float(calculate_mse(Y_ref, Y_arr))
            }

            # Calculate VIF if available
            vif_val = calculate_vif(reference, arr)
            if vif_val is not None:
                metrics["vif"] = vif_val

            # Calculate Delta E for RGB images
            delta_e_val = calculate_delta_e(reference, arr)
            if delta_e_val is not None:
                metrics["delta_e"] = delta_e_val

            # Calculate LPIPS for RGB images
            lpips_val = calculate_lpips(reference, arr)
            if lpips_val is not None:
                metrics["lpips"] = lpips_val

            stats["metrics_vs_reference"] = metrics

        elif arr.ndim == 2 and reference.ndim == 2:
            # Both grayscale
            metrics = {
                "psnr": float(psnr(reference, arr, data_range=255)),
                "ssim": float(ssim(reference, arr, data_range=255)),
                "mse": float(calculate_mse(reference, arr))
            }

            # Calculate VIF if available (will convert to RGB internally)
            vif_val = calculate_vif(reference, arr)
            if vif_val is not None:
                metrics["vif"] = vif_val

            stats["metrics_vs_reference"] = metrics

    print(f"  Saved: {filename} ({stats['file_size']:,} bytes)")

    # Add to manifest if provided
    if manifest_entry is not None:
        manifest_entry[filename] = stats

    return full_path, stats

def save_baseline_jpeg(arr, path, name, quality=95, encoder=JpegEncoder.LIBJPEG_TURBO):
    """Save a baseline JPEG at specified quality."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    filename = f"{name}_baseline_q{quality}.jpg"
    full_path = path / filename

    if is_jxl_encoder(encoder):
        jxl_path = path / filename.replace('.jpg', '.jxl')
        jxl_size = encode_jpeg_to_file(img, jxl_path, quality, encoder)
        decoded_img = decode_jxl_to_image(jxl_path)
        png_path = path / filename.replace('.jpg', '.png')
        decoded_img.save(str(png_path), format="PNG")
        # Keep .jxl source; return png path for viewer
        return png_path, jxl_size
    else:
        encode_jpeg_to_file(img, full_path, quality, encoder)
        return full_path, get_file_size(full_path)

def rgb_to_ycbcr_bt601(rgb_u8):
    """Convert RGB to YCbCr using BT.601."""
    rgb = rgb_u8.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128.0
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 128.0
    return Y, Cb, Cr

def ycbcr_to_rgb_bt601(Y, Cb, Cr):
    """Convert YCbCr to RGB using BT.601."""
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0
    R = Y + 1.402*Cr
    G = Y - 0.344136*Cb - 0.714136*Cr
    B = Y + 1.772*Cb
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)

def tile_image(image_path, tile_size=256):
    """Tile a large image into a pyramid structure."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Create tiles
    h, w = img_array.shape[:2]

    # L0: 4x4 grid of 256x256 tiles (1024x1024 total)
    l0_tiles = {}
    for dy in range(4):
        for dx in range(4):
            y_start = dy * tile_size
            x_start = dx * tile_size
            l0_tiles[(dx, dy)] = img_array[y_start:y_start+tile_size,
                                          x_start:x_start+tile_size]

    # L1: 2x2 grid of 256x256 tiles (512x512 total)
    # These should be downsampled from the corresponding L0 region
    l1_source = img_array[:512*2, :512*2]  # Top-left 1024x1024
    l1_downsampled = Image.fromarray(l1_source).resize((512, 512), Image.LANCZOS)
    l1_array = np.array(l1_downsampled)

    l1_tiles = {}
    for dy in range(2):
        for dx in range(2):
            y_start = dy * tile_size
            x_start = dx * tile_size
            l1_tiles[(dx, dy)] = l1_array[y_start:y_start+tile_size,
                                          x_start:x_start+tile_size]

    # L2: Single 256x256 tile
    # Should be downsampled from the same L0 region
    l2_source = img_array[:1024, :1024]  # Top-left 1024x1024
    l2_downsampled = Image.fromarray(l2_source).resize((256, 256), Image.LANCZOS)
    l2_tile = np.array(l2_downsampled)

    return l2_tile, l1_tiles, l0_tiles

def compress_with_debug(l2_tile, l1_tiles, l0_tiles, out_dir, tile_size=256,
                        jpeg_quality=75, baseline_quality=95, encoder=JpegEncoder.LIBJPEG_TURBO):
    """Compress tiles with debug output and detailed statistics."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== COMPRESSION PHASE ===")

    # Initialize manifest
    config = {
            "tile_size": tile_size,
            "residual_jpeg_quality": jpeg_quality,
            "baseline_jpeg_quality": baseline_quality,
            "encoder": encoder.value
    }
    manifest = {
        "configuration": config,
        "compression_phase": {
            "L2": {},
            "L1": {},
            "L0": {}
        },
        "baseline_tiles": {},
        "size_comparison": {}
    }

    # Save L2 baseline JPEG
    l2_baseline_path, l2_baseline_size = save_baseline_jpeg(
        l2_tile, out_dir / "baseline", "L2", baseline_quality, encoder)
    manifest["baseline_tiles"]["L2"] = {
        "path": str(l2_baseline_path),
        "size": l2_baseline_size
    }

    # Save L2 tile and its channels
    l2_path, l2_stats = save_image_with_stats(
        l2_tile, out_dir / "compress", 1, "L2_original.png",
        manifest_entry=manifest["compression_phase"]["L2"])

    Y_l2, Cb_l2, Cr_l2 = rgb_to_ycbcr_bt601(l2_tile)
    save_image_with_stats(Y_l2, out_dir / "compress", 2, "L2_luma.png",
                         manifest_entry=manifest["compression_phase"]["L2"])
    save_image_with_stats(Cb_l2, out_dir / "compress", 3, "L2_chroma_cb.png",
                         manifest_entry=manifest["compression_phase"]["L2"])
    save_image_with_stats(Cr_l2, out_dir / "compress", 4, "L2_chroma_cr.png",
                         manifest_entry=manifest["compression_phase"]["L2"])

    # Upsample L2 for L1 prediction
    UPSAMPLE = Image.Resampling.BILINEAR
    l1_pred_mosaic = np.array(Image.fromarray(l2_tile).resize(
        (tile_size*2, tile_size*2), resample=UPSAMPLE))

    # Track total sizes
    total_baseline_L1 = 0
    total_residual_L1 = 0
    total_baseline_L0 = 0
    total_residual_L0 = 0

    # Process L1 tiles
    l1_reconstructed = {}
    step = 10
    for (dx, dy), l1_gt in l1_tiles.items():
        print(f"\n  Processing L1 tile ({dx}, {dy})")
        tile_manifest = {}

        # Save baseline JPEG for this L1 tile
        l1_baseline_path, l1_baseline_size = save_baseline_jpeg(
            l1_gt, out_dir / "baseline", f"L1_{dx}_{dy}", baseline_quality, encoder)
        manifest["baseline_tiles"][f"L1_{dx}_{dy}"] = {
            "path": str(l1_baseline_path),
            "size": l1_baseline_size
        }
        total_baseline_L1 += l1_baseline_size

        # Save original and channels
        save_image_with_stats(l1_gt, out_dir / "compress", step,
                             f"L1_{dx}_{dy}_original.png", manifest_entry=tile_manifest)

        Y_gt, Cb_gt, Cr_gt = rgb_to_ycbcr_bt601(l1_gt)
        save_image_with_stats(Y_gt, out_dir / "compress", step+1,
                             f"L1_{dx}_{dy}_luma.png", manifest_entry=tile_manifest)
        save_image_with_stats(Cb_gt, out_dir / "compress", step+2,
                             f"L1_{dx}_{dy}_chroma_cb.png", manifest_entry=tile_manifest)
        save_image_with_stats(Cr_gt, out_dir / "compress", step+3,
                             f"L1_{dx}_{dy}_chroma_cr.png", manifest_entry=tile_manifest)

        # Get prediction
        pred = l1_pred_mosaic[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
        save_image_with_stats(pred, out_dir / "compress", step+4,
                             f"L1_{dx}_{dy}_prediction.png",
                             reference=l1_gt, manifest_entry=tile_manifest)

        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

        # Denoise ground truth luma before residual computation

        # Compute residual
        residual_raw = Y_gt - Y_pred
        save_image_with_stats(residual_raw, out_dir / "compress", step+5,
                             f"L1_{dx}_{dy}_residual_raw.png",
                             normalize=True, manifest_entry=tile_manifest)

        # Center (+128)
        residual_centered = np.clip(np.round(residual_raw + 128.0), 0, 255).astype(np.uint8)
        save_image_with_stats(residual_centered, out_dir / "compress", step+7,
                             f"L1_{dx}_{dy}_residual_centered.png", manifest_entry=tile_manifest)

        # Save as JPEG
        jpeg_path, jpeg_stats = save_image_with_stats(
            residual_centered, out_dir / "compress", step+8,
            f"L1_{dx}_{dy}_residual_jpeg.jpg", quality=jpeg_quality, manifest_entry=tile_manifest,
            encoder=encoder)

        total_residual_L1 += jpeg_stats["file_size"]

        # Calculate compression ratio for this tile
        tile_manifest["compression_ratio"] = l1_baseline_size / jpeg_stats["file_size"]
        tile_manifest["space_savings_pct"] = 100 * (1 - jpeg_stats["file_size"] / l1_baseline_size)

        # Simulate reconstruction for L0 prediction
        r_dec = np.array(Image.open(jpeg_path).convert("L")).astype(np.float32) - 128.0
        Y_recon = np.clip(Y_pred + r_dec, 0, 255)
        l1_reconstructed[(dx, dy)] = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)

        # Save reconstructed image with metrics against original
        save_image_with_stats(l1_reconstructed[(dx, dy)], out_dir / "compress", step+9,
                            f"L1_{dx}_{dy}_reconstructed.png",
                            reference=l1_gt, manifest_entry=tile_manifest)

        # Save reconstruction quality metrics (legacy, for backward compatibility)
        tile_manifest["reconstruction_psnr"] = float(
            psnr(l1_gt, l1_reconstructed[(dx, dy)], data_range=255))

        manifest["compression_phase"]["L1"][f"tile_{dx}_{dy}"] = tile_manifest
        step += 10

    # Build L1 mosaic for L0 prediction
    l1_mosaic = np.zeros((tile_size*2, tile_size*2, 3), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            if (dx, dy) in l1_reconstructed:
                l1_mosaic[dy*tile_size:(dy+1)*tile_size,
                         dx*tile_size:(dx+1)*tile_size] = l1_reconstructed[(dx, dy)]

    # Upsample for L0 prediction
    l0_pred_mosaic = np.array(Image.fromarray(l1_mosaic).resize(
        (tile_size*4, tile_size*4), resample=UPSAMPLE))

    # Process L0 tiles
    step = 20
    for (dx, dy), l0_gt in l0_tiles.items():
        print(f"  Processing L0 tile ({dx}, {dy})")
        tile_manifest = {}

        # Save baseline JPEG for this L0 tile
        l0_baseline_path, l0_baseline_size = save_baseline_jpeg(
            l0_gt, out_dir / "baseline", f"L0_{dx}_{dy}", baseline_quality, encoder)
        manifest["baseline_tiles"][f"L0_{dx}_{dy}"] = {
            "path": str(l0_baseline_path),
            "size": l0_baseline_size
        }
        total_baseline_L0 += l0_baseline_size

        # Save original and channels
        save_image_with_stats(l0_gt, out_dir / "compress", step,
                             f"L0_{dx}_{dy}_original.png", manifest_entry=tile_manifest)

        Y_gt, Cb_gt, Cr_gt = rgb_to_ycbcr_bt601(l0_gt)
        save_image_with_stats(Y_gt, out_dir / "compress", step+1,
                             f"L0_{dx}_{dy}_luma.png", manifest_entry=tile_manifest)
        save_image_with_stats(Cb_gt, out_dir / "compress", step+2,
                             f"L0_{dx}_{dy}_chroma_cb.png", manifest_entry=tile_manifest)
        save_image_with_stats(Cr_gt, out_dir / "compress", step+3,
                             f"L0_{dx}_{dy}_chroma_cr.png", manifest_entry=tile_manifest)

        # Get prediction
        pred = l0_pred_mosaic[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
        save_image_with_stats(pred, out_dir / "compress", step+4,
                             f"L0_{dx}_{dy}_prediction.png",
                             reference=l0_gt, manifest_entry=tile_manifest)

        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

        # Denoise ground truth luma before residual computation

        # Compute residual
        residual_raw = Y_gt - Y_pred
        save_image_with_stats(residual_raw, out_dir / "compress", step+5,
                             f"L0_{dx}_{dy}_residual_raw.png",
                             normalize=True, manifest_entry=tile_manifest)

        # Center (+128)
        residual_centered = np.clip(np.round(residual_raw + 128.0), 0, 255).astype(np.uint8)
        save_image_with_stats(residual_centered, out_dir / "compress", step+7,
                             f"L0_{dx}_{dy}_residual_centered.png", manifest_entry=tile_manifest)

        # Save as JPEG
        jpeg_path, jpeg_stats = save_image_with_stats(
            residual_centered, out_dir / "compress", step+8,
            f"L0_{dx}_{dy}_residual_jpeg.jpg", quality=jpeg_quality, manifest_entry=tile_manifest,
            encoder=encoder)

        total_residual_L0 += jpeg_stats["file_size"]

        # Calculate compression ratio for this tile
        tile_manifest["compression_ratio"] = l0_baseline_size / jpeg_stats["file_size"]
        tile_manifest["space_savings_pct"] = 100 * (1 - jpeg_stats["file_size"] / l0_baseline_size)

        # Simulate reconstruction
        r_dec = np.array(Image.open(jpeg_path).convert("L")).astype(np.float32) - 128.0
        Y_recon = np.clip(Y_pred + r_dec, 0, 255)
        l0_reconstructed = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)

        # Save reconstructed image with metrics against original
        save_image_with_stats(l0_reconstructed, out_dir / "compress", step+9,
                            f"L0_{dx}_{dy}_reconstructed.png",
                            reference=l0_gt, manifest_entry=tile_manifest)

        manifest["compression_phase"]["L0"][f"tile_{dx}_{dy}"] = tile_manifest

        if step < 40:  # Only increment for first few to avoid too many files
            step += 10

    # Add overall statistics
    total_baseline = l2_baseline_size + total_baseline_L1 + total_baseline_L0
    total_residual = total_residual_L1 + total_residual_L0 + l2_baseline_size  # L2 is kept

    manifest["size_comparison"] = {
        "baseline_total": total_baseline,
        "baseline_L2": l2_baseline_size,
        "baseline_L1_total": total_baseline_L1,
        "baseline_L0_total": total_baseline_L0,
        "origami_total": total_residual,
        "origami_L2_baseline": l2_baseline_size,
        "origami_L1_residuals": total_residual_L1,
        "origami_L0_residuals": total_residual_L0,
        "overall_compression_ratio": total_baseline / total_residual,
        "overall_space_savings_pct": 100 * (1 - total_residual / total_baseline),
        "L1_compression_ratio": total_baseline_L1 / total_residual_L1 if total_residual_L1 > 0 else 0,
        "L0_compression_ratio": total_baseline_L0 / total_residual_L0 if total_residual_L0 > 0 else 0
    }

    return manifest

def decompress_with_debug(out_dir, manifest, tile_size=256):
    """Decompress tiles with debug output and statistics."""
    compress_dir = pathlib.Path(out_dir) / "compress"
    decompress_dir = pathlib.Path(out_dir) / "decompress"
    decompress_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== DECOMPRESSION PHASE ===")

    manifest["decompression_phase"] = {
        "L2": {},
        "L1": {},
        "L0": {}
    }

    # Load L2 tile
    l2_tile = np.array(Image.open(compress_dir / "001_L2_original.png").convert("RGB"))
    save_image_with_stats(l2_tile, decompress_dir, 50, "L2_decode.png",
                         manifest_entry=manifest["decompression_phase"]["L2"])

    # Upsample for L1 prediction
    UPSAMPLE = Image.Resampling.BILINEAR
    l1_pred = np.array(Image.fromarray(l2_tile).resize(
        (tile_size*2, tile_size*2), resample=UPSAMPLE))
    save_image_with_stats(l1_pred, decompress_dir, 51, "L1_mosaic_prediction.png",
                         manifest_entry=manifest["decompression_phase"]["L1"])

    # Reconstruct L1 tiles
    l1_reconstructed = {}
    step = 60
    for dy in range(2):
        for dx in range(2):
            print(f"  Reconstructing L1 tile ({dx}, {dy})")
            tile_manifest = {}

            # Get prediction slice
            pred = l1_pred[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

            # Load residual - use tile index to find correct file (.png for JXL, .jpg otherwise)
            tile_idx = dy * 2 + dx
            residual_file = f"{10 + tile_idx * 10 + 8:03d}_L1_{dx}_{dy}_residual_jpeg.jpg"
            residual_path = compress_dir / residual_file
            # For JXL runs, the .jpg was replaced with .png
            if not residual_path.exists():
                residual_path = compress_dir / residual_file.replace('.jpg', '.png')
            if residual_path.exists():
                residual_loaded = np.array(Image.open(residual_path).convert("L"))
                save_image_with_stats(residual_loaded, decompress_dir, step,
                                     f"L1_{dx}_{dy}_residual_loaded.png", manifest_entry=tile_manifest)

                # Decenter (-128)
                residual_decentered = residual_loaded.astype(np.float32) - 128.0
                save_image_with_stats(residual_decentered, decompress_dir, step+1,
                                     f"L1_{dx}_{dy}_residual_decentered.png",
                                     normalize=True, manifest_entry=tile_manifest)

                # Reconstruct luma
                Y_recon = np.clip(Y_pred + residual_decentered, 0, 255)
                save_image_with_stats(Y_recon, decompress_dir, step+2,
                                     f"L1_{dx}_{dy}_luma_reconstructed.png", manifest_entry=tile_manifest)

                # Full RGB reconstruction
                rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
                save_image_with_stats(rgb_recon, decompress_dir, step+3,
                                     f"L1_{dx}_{dy}_reconstructed.png", manifest_entry=tile_manifest)

                l1_reconstructed[(dx, dy)] = rgb_recon

                # Calculate final quality vs original
                original_path = compress_dir / f"{10 + tile_idx * 10:03d}_L1_{dx}_{dy}_original.png"
                if original_path.exists():
                    original = np.array(Image.open(original_path).convert("RGB"))
                    tile_manifest["final_psnr"] = float(psnr(original, rgb_recon, data_range=255))
                    tile_manifest["final_ssim"] = float(ssim(original.mean(axis=2),
                                                            rgb_recon.mean(axis=2), data_range=255))
                    tile_manifest["final_mse"] = float(calculate_mse(original, rgb_recon))
                    vif_val = calculate_vif(original, rgb_recon)
                    if vif_val is not None:
                        tile_manifest["final_vif"] = vif_val
                    delta_e_val = calculate_delta_e(original, rgb_recon)
                    if delta_e_val is not None:
                        tile_manifest["final_delta_e"] = delta_e_val
                    lpips_val = calculate_lpips(original, rgb_recon)
                    if lpips_val is not None:
                        tile_manifest["final_lpips"] = lpips_val

                manifest["decompression_phase"]["L1"][f"tile_{dx}_{dy}"] = tile_manifest

    # Build L1 mosaic
    l1_mosaic = np.zeros((tile_size*2, tile_size*2, 3), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            if (dx, dy) in l1_reconstructed:
                l1_mosaic[dy*tile_size:(dy+1)*tile_size,
                         dx*tile_size:(dx+1)*tile_size] = l1_reconstructed[(dx, dy)]
    save_image_with_stats(l1_mosaic, decompress_dir, 64, "L1_mosaic_full.png",
                         manifest_entry=manifest["decompression_phase"]["L1"])

    # Upsample for L0 prediction
    l0_pred = np.array(Image.fromarray(l1_mosaic).resize(
        (tile_size*4, tile_size*4), resample=UPSAMPLE))
    save_image_with_stats(l0_pred, decompress_dir, 65, "L0_mosaic_prediction.png",
                         manifest_entry=manifest["decompression_phase"]["L0"])

    # Reconstruct L0 tiles
    step = 70
    for dy in range(4):
        for dx in range(4):
            print(f"  Reconstructing L0 tile ({dx}, {dy})")
            tile_manifest = {}

            # Get prediction slice
            pred = l0_pred[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

            # Load residual - find the correct file with dynamic prefix (.jpg or .png for JXL)
            residual_files = list(compress_dir.glob(f"*_L0_{dx}_{dy}_residual_jpeg.jpg"))
            if not residual_files:
                residual_files = list(compress_dir.glob(f"*_L0_{dx}_{dy}_residual_jpeg.png"))
            if residual_files:
                residual_path = residual_files[0]
                residual_loaded = np.array(Image.open(residual_path).convert("L"))
                save_image_with_stats(residual_loaded, decompress_dir, step,
                                     f"L0_{dx}_{dy}_residual_loaded.png", manifest_entry=tile_manifest)

                # Decenter (-128)
                residual_decentered = residual_loaded.astype(np.float32) - 128.0
                save_image_with_stats(residual_decentered, decompress_dir, step+1,
                                     f"L0_{dx}_{dy}_residual_decentered.png",
                                     normalize=True, manifest_entry=tile_manifest)

                # Reconstruct luma
                Y_recon = np.clip(Y_pred + residual_decentered, 0, 255)
                save_image_with_stats(Y_recon, decompress_dir, step+2,
                                     f"L0_{dx}_{dy}_luma_reconstructed.png", manifest_entry=tile_manifest)

                # Full RGB reconstruction
                rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
                save_image_with_stats(rgb_recon, decompress_dir, step+3,
                                     f"L0_{dx}_{dy}_reconstructed.png", manifest_entry=tile_manifest)

                # Calculate final quality vs original
                # Find the correct original file
                for step_num in [20, 30, 40]:  # Check possible step numbers
                    original_path = compress_dir / f"{step_num:03d}_L0_{dx}_{dy}_original.png"
                    if original_path.exists():
                        original = np.array(Image.open(original_path).convert("RGB"))
                        tile_manifest["final_psnr"] = float(psnr(original, rgb_recon, data_range=255))
                        tile_manifest["final_ssim"] = float(ssim(original.mean(axis=2),
                                                                rgb_recon.mean(axis=2), data_range=255))
                        tile_manifest["final_mse"] = float(calculate_mse(original, rgb_recon))
                        vif_val = calculate_vif(original, rgb_recon)
                        if vif_val is not None:
                            tile_manifest["final_vif"] = vif_val
                        delta_e_val = calculate_delta_e(original, rgb_recon)
                        if delta_e_val is not None:
                            tile_manifest["final_delta_e"] = delta_e_val
                        lpips_val = calculate_lpips(original, rgb_recon)
                        if lpips_val is not None:
                            tile_manifest["final_lpips"] = lpips_val
                        break

                manifest["decompression_phase"]["L0"][f"tile_{dx}_{dy}"] = tile_manifest

            if step < 90:  # Limit number of files
                step += 10

    return manifest

def create_pac_file(out_dir, l2_tile, l1_tiles, l0_tiles, tile_size, jpeg_quality, baseline_quality, encoder=JpegEncoder.LIBJPEG_TURBO):
    """Create a PAC file with all tiles needed for serving."""
    import struct
    import io

    pac_path = pathlib.Path(out_dir) / "tiles.pac"

    print("\n=== CREATING PAC FILE ===")

    with open(pac_path, 'wb') as pac:
        # Write PAC header
        pac.write(b'PAC1')  # Magic number

        # Count total entries: 1 L2 + 4 L1 residuals + 16 L0 residuals = 21
        num_entries = 21
        pac.write(struct.pack('<I', num_entries))

        # Prepare to write index
        entries = []
        data_offset = 8 + num_entries * 24  # Header + index size

        # Helper to add entry
        def add_entry(level, x, y, data):
            nonlocal data_offset
            entry_id = f"L{level}_{x}_{y}"
            entries.append({
                'id': entry_id,
                'level': level,
                'x': x,
                'y': y,
                'offset': data_offset,
                'size': len(data)
            })
            data_offset += len(data)
            return data

        # Compress L2 baseline
        l2_img = Image.fromarray(np.clip(l2_tile, 0, 255).astype(np.uint8), mode="RGB")
        l2_bytes = encode_jpeg_to_bytes(l2_img, baseline_quality, encoder)
        l2_data = add_entry(2, 0, 0, l2_bytes)
        use_jxl = is_jxl_encoder(encoder)

        # Process L1 residuals
        l1_data_list = []
        Y_l2, Cb_l2, Cr_l2 = rgb_to_ycbcr_bt601(l2_tile)
        UPSAMPLE = Image.Resampling.BILINEAR
        l1_pred_mosaic = np.array(Image.fromarray(l2_tile).resize(
            (tile_size*2, tile_size*2), resample=UPSAMPLE))

        l1_reconstructed = {}
        for (dx, dy), l1_gt in l1_tiles.items():
            # Get prediction
            pred = l1_pred_mosaic[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

            # Compute residual
            Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)
            residual_raw = Y_gt - Y_pred

            # Center and save as JPEG
            residual_centered = np.clip(np.round(residual_raw + 128.0), 0, 255).astype(np.uint8)
            residual_img = Image.fromarray(residual_centered, mode="L")
            l1_encoded_bytes = encode_jpeg_to_bytes(residual_img, jpeg_quality, encoder)
            l1_data = add_entry(1, dx, dy, l1_encoded_bytes)
            l1_data_list.append(l1_data)

            # Reconstruct for L0 prediction
            if is_jxl_encoder(encoder):
                r_dec = np.array(decode_jxl_to_image(l1_data).convert("L")).astype(np.float32) - 128.0
            else:
                r_dec = np.array(Image.open(io.BytesIO(l1_data)).convert("L")).astype(np.float32) - 128.0
            Y_recon = np.clip(Y_pred + r_dec, 0, 255)
            l1_reconstructed[(dx, dy)] = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)

        # Build L1 mosaic for L0 prediction
        l1_mosaic = np.zeros((tile_size*2, tile_size*2, 3), dtype=np.uint8)
        for dy in range(2):
            for dx in range(2):
                if (dx, dy) in l1_reconstructed:
                    l1_mosaic[dy*tile_size:(dy+1)*tile_size,
                             dx*tile_size:(dx+1)*tile_size] = l1_reconstructed[(dx, dy)]

        # Process L0 residuals
        l0_pred_mosaic = np.array(Image.fromarray(l1_mosaic).resize(
            (tile_size*4, tile_size*4), resample=UPSAMPLE))

        l0_data_list = []
        for (dx, dy), l0_gt in l0_tiles.items():
            # Get prediction
            pred = l0_pred_mosaic[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

            # Compute residual
            Y_gt, _, _ = rgb_to_ycbcr_bt601(l0_gt)
            residual_raw = Y_gt - Y_pred

            # Center and save as JPEG
            residual_centered = np.clip(np.round(residual_raw + 128.0), 0, 255).astype(np.uint8)
            residual_img = Image.fromarray(residual_centered, mode="L")
            l0_encoded_bytes = encode_jpeg_to_bytes(residual_img, jpeg_quality, encoder)
            l0_data = add_entry(0, dx, dy, l0_encoded_bytes)
            l0_data_list.append(l0_data)

        # Write index
        for entry in entries:
            # Write entry: level (1 byte), x (2 bytes), y (2 bytes), offset (4 bytes), size (4 bytes)
            pac.write(struct.pack('<B', entry['level']))
            pac.write(struct.pack('<H', entry['x']))
            pac.write(struct.pack('<H', entry['y']))
            pac.write(struct.pack('<I', entry['offset']))
            pac.write(struct.pack('<I', entry['size']))
            # Pad to 24 bytes per entry
            pac.write(b'\x00' * 11)

        # Write data
        pac.write(l2_data)
        for data in l1_data_list:
            pac.write(data)
        for data in l0_data_list:
            pac.write(data)

    pac_size = os.path.getsize(pac_path)
    print(f"PAC file created: {pac_path} ({pac_size:,} bytes)")

    return pac_path, pac_size

def main():
    parser = argparse.ArgumentParser(description="Debug ORIGAMI compression with detailed manifest")
    parser.add_argument("--image", required=True, help="Path to input image (1024x1024)")
    parser.add_argument("--out", help="Output directory for debug images (auto-generated if not specified)")
    parser.add_argument("--tile", type=int, default=256, help="Tile size")
    parser.add_argument("--resq", type=int, default=75, help="JPEG quality for residuals")
    parser.add_argument("--baseq", type=int, default=95, help="JPEG quality for baseline tiles")
    parser.add_argument("--pac", action="store_true", help="Create PAC file for serving")
    parser.add_argument("--encoder", default="libjpeg-turbo",
                       help="Encoder: libjpeg-turbo, jpegli, mozjpeg, or jpegxl")

    args = parser.parse_args()
    encoder = parse_encoder_arg(args.encoder)

    # Generate output directory name if not specified
    if args.out is None:
        # Auto-generate: {encoder}_debug_j{J}_pac
        # libjpeg-turbo is the default, no prefix needed
        if encoder == JpegEncoder.LIBJPEG_TURBO:
            prefix = ""
        else:
            prefix = encoder.value + "_"
        dir_parts = [f"{prefix}debug", f"j{args.resq}"]
        if args.pac:
            dir_parts.append("pac")
        args.out = "evals/runs/" + "_".join(dir_parts)

        print(f"Output directory: {args.out}")

    # Create output directory
    output_dir = pathlib.Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tile the input image
    print(f"Processing image: {args.image}")
    l2_tile, l1_tiles, l0_tiles = tile_image(args.image, args.tile)

    # Compress with debug output
    manifest = compress_with_debug(l2_tile, l1_tiles, l0_tiles, args.out,
                                  args.tile, args.resq, args.baseq, encoder)

    # Decompress with debug output
    manifest = decompress_with_debug(args.out, manifest, args.tile)

    # Create PAC file if requested
    if args.pac:
        pac_path, pac_size = create_pac_file(args.out, l2_tile, l1_tiles, l0_tiles,
                                            args.tile, args.resq, args.baseq, encoder)
        manifest["pac_file"] = {
            "path": str(pac_path),
            "size": pac_size
        }

    # Add configuration to manifest
    manifest["input_image"] = str(args.image)

    # Save manifest
    manifest_path = pathlib.Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write summary text file
    summary_path = pathlib.Path(args.out) / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ORIGAMI COMPRESSION DEBUG SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input Image: {args.image}\n")
        f.write(f"Tile Size: {args.tile}x{args.tile}\n")
        f.write(f"Residual JPEG Quality: {args.resq}\n")
        f.write(f"Baseline JPEG Quality: {args.baseq}\n\n")

        f.write("SIZE COMPARISON\n")
        f.write("-" * 30 + "\n")
        sc = manifest["size_comparison"]
        f.write(f"Baseline Total: {sc['baseline_total']:,} bytes\n")
        f.write(f"  - L2: {sc['baseline_L2']:,} bytes\n")
        f.write(f"  - L1: {sc['baseline_L1_total']:,} bytes\n")
        f.write(f"  - L0: {sc['baseline_L0_total']:,} bytes\n\n")

        f.write(f"ORIGAMI Total: {sc['origami_total']:,} bytes\n")
        f.write(f"  - L2 (baseline): {sc['origami_L2_baseline']:,} bytes\n")
        f.write(f"  - L1 residuals: {sc['origami_L1_residuals']:,} bytes\n")
        f.write(f"  - L0 residuals: {sc['origami_L0_residuals']:,} bytes\n\n")

        f.write(f"Overall Compression Ratio: {sc['overall_compression_ratio']:.2f}x\n")
        f.write(f"Overall Space Savings: {sc['overall_space_savings_pct']:.1f}%\n")
        f.write(f"L1 Compression Ratio: {sc['L1_compression_ratio']:.2f}x\n")
        f.write(f"L0 Compression Ratio: {sc['L0_compression_ratio']:.2f}x\n\n")

        # Add average quality metrics
        f.write("AVERAGE QUALITY METRICS\n")
        f.write("-" * 30 + "\n")

        # Calculate L1 averages
        l1_psnrs = []
        for tile_key in manifest["decompression_phase"]["L1"]:
            if tile_key.startswith("tile_"):
                tile_data = manifest["decompression_phase"]["L1"][tile_key]
                if "final_psnr" in tile_data:
                    l1_psnrs.append(tile_data["final_psnr"])

        if l1_psnrs:
            f.write(f"L1 Average PSNR: {np.mean(l1_psnrs):.2f} dB\n")

        # Calculate L0 averages
        l0_psnrs = []
        for tile_key in manifest["decompression_phase"]["L0"]:
            if tile_key.startswith("tile_"):
                tile_data = manifest["decompression_phase"]["L0"][tile_key]
                if "final_psnr" in tile_data:
                    l0_psnrs.append(tile_data["final_psnr"])

        if l0_psnrs:
            f.write(f"L0 Average PSNR: {np.mean(l0_psnrs):.2f} dB\n")

    print(f"\nDebug images saved to: {args.out}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Summary saved to: {summary_path}")

    # Print quick summary
    print("\n" + "=" * 50)
    print(f"Baseline total: {manifest['size_comparison']['baseline_total']:,} bytes")
    print(f"ORIGAMI total: {manifest['size_comparison']['origami_total']:,} bytes")
    print(f"Compression ratio: {manifest['size_comparison']['overall_compression_ratio']:.2f}x")
    print(f"Space savings: {manifest['size_comparison']['overall_space_savings_pct']:.1f}%")

if __name__ == "__main__":
    main()