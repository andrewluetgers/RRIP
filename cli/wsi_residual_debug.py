#!/usr/bin/env python3
"""
wsi_residual_debug.py

Debug version of wsi_residual_tool that saves intermediate images for visualization.
This tool processes a single L2 tile and its family, saving all intermediate steps
as grayscale PNG images for paper visualizations.

Usage:
  python wsi_residual_debug.py --image paper/L0-1024.jpg --out debug_output --tile 256 --quant 32 --resq 75

The tool saves the following images:
Compression:
  001_L2_original.png - Original L2 tile
  002_L2_luma.png - Luminance channel of L2
  003_L2_chroma_cb.png - Cb channel of L2
  004_L2_chroma_cr.png - Cr channel of L2

  For each L1 tile (2x2 grid):
    010_L1_{x}_{y}_original.png - Original L1 tile
    011_L1_{x}_{y}_luma.png - Luminance channel
    012_L1_{x}_{y}_chroma_cb.png - Cb channel
    013_L1_{x}_{y}_chroma_cr.png - Cr channel
    014_L1_{x}_{y}_prediction.png - Upsampled prediction from L2
    015_L1_{x}_{y}_residual_raw.png - Raw residual (before quantization)
    016_L1_{x}_{y}_residual_quantized.png - After quantization
    017_L1_{x}_{y}_residual_centered.png - After +128 centering
    018_L1_{x}_{y}_residual_jpeg.jpg - After JPEG compression

  For each L0 tile (4x4 grid):
    020_L0_{x}_{y}_original.png - Original L0 tile
    021_L0_{x}_{y}_luma.png - Luminance channel
    022_L0_{x}_{y}_chroma_cb.png - Cb channel
    023_L0_{x}_{y}_chroma_cr.png - Cr channel
    024_L0_{x}_{y}_prediction.png - Upsampled prediction from L1 mosaic
    025_L0_{x}_{y}_residual_raw.png - Raw residual
    026_L0_{x}_{y}_residual_quantized.png - After quantization
    027_L0_{x}_{y}_residual_centered.png - After +128 centering
    028_L0_{x}_{y}_residual_jpeg.jpg - After JPEG compression

Decompression:
  050_L2_decode.png - L2 tile used for reconstruction
  051_L1_mosaic_prediction.png - Upsampled L2 for L1 prediction

  For each L1 tile:
    060_L1_{x}_{y}_residual_loaded.png - Loaded JPEG residual
    061_L1_{x}_{y}_residual_decentered.png - After -128 decentering
    062_L1_{x}_{y}_luma_reconstructed.png - Reconstructed luminance
    063_L1_{x}_{y}_reconstructed.png - Full RGB reconstruction

  064_L1_mosaic_full.png - Complete L1 mosaic
  065_L0_mosaic_prediction.png - Upsampled L1 for L0 prediction

  For each L0 tile:
    070_L0_{x}_{y}_residual_loaded.png - Loaded JPEG residual
    071_L0_{x}_{y}_residual_decentered.png - After -128 decentering
    072_L0_{x}_{y}_luma_reconstructed.png - Reconstructed luminance
    073_L0_{x}_{y}_reconstructed.png - Full RGB reconstruction
"""
import argparse
import pathlib
import numpy as np
from PIL import Image
import json

def save_debug_image(arr, path, step_num, name_suffix, normalize=False):
    """Save a debug image with sequential numbering."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if arr.ndim == 3 and arr.shape[2] == 3:
        # RGB image
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    else:
        # Grayscale
        if normalize:
            # Normalize to 0-255 range for visualization
            arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        else:
            arr_norm = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_norm, mode="L")

    filename = f"{step_num:03d}_{name_suffix}"
    if filename.endswith('.jpg'):
        img.save(path / filename, format="JPEG", quality=75)
    else:
        if not filename.endswith('.png'):
            filename += '.png'
        img.save(path / filename, format="PNG")

    print(f"  Saved: {filename}")
    return path / filename

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

def quantize_residual(residual, num_levels):
    """Quantize residual values to specified levels."""
    if num_levels >= 256:
        return residual

    min_val = -255.0
    max_val = 255.0
    normalized = (residual - min_val) / (max_val - min_val)
    quantized = np.round(normalized * (num_levels - 1)) / (num_levels - 1)
    return quantized * (max_val - min_val) + min_val

def tile_image(image_path, tile_size=256):
    """Tile a large image into a pyramid structure."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Create tiles
    h, w = img_array.shape[:2]

    # L2: Single 256x256 center crop
    l2_tile = img_array[h//2-tile_size//2:h//2+tile_size//2,
                        w//2-tile_size//2:w//2+tile_size//2]

    # L1: 2x2 grid of 256x256 tiles (512x512 total)
    l1_tiles = {}
    for dy in range(2):
        for dx in range(2):
            y_start = dy * tile_size
            x_start = dx * tile_size
            l1_tiles[(dx, dy)] = img_array[y_start:y_start+tile_size,
                                          x_start:x_start+tile_size]

    # L0: 4x4 grid of 256x256 tiles (1024x1024 total)
    l0_tiles = {}
    for dy in range(4):
        for dx in range(4):
            y_start = dy * tile_size
            x_start = dx * tile_size
            l0_tiles[(dx, dy)] = img_array[y_start:y_start+tile_size,
                                          x_start:x_start+tile_size]

    return l2_tile, l1_tiles, l0_tiles

def compress_with_debug(l2_tile, l1_tiles, l0_tiles, out_dir, tile_size=256,
                        quant_levels=32, jpeg_quality=75):
    """Compress tiles with debug output at each step."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== COMPRESSION PHASE ===")

    # Track saved residual paths for decompression phase
    residual_paths = {'L1': {}, 'L0': {}}

    # Save L2 tile and its channels
    save_debug_image(l2_tile, out_dir / "compress", 1, "L2_original.png")
    Y_l2, Cb_l2, Cr_l2 = rgb_to_ycbcr_bt601(l2_tile)
    save_debug_image(Y_l2, out_dir / "compress", 2, "L2_luma.png")
    save_debug_image(Cb_l2, out_dir / "compress", 3, "L2_chroma_cb.png")
    save_debug_image(Cr_l2, out_dir / "compress", 4, "L2_chroma_cr.png")

    # Upsample L2 for L1 prediction
    UPSAMPLE = Image.Resampling.BILINEAR
    l1_pred = np.array(Image.fromarray(l2_tile).resize(
        (tile_size*2, tile_size*2), resample=UPSAMPLE))

    # Process L1 tiles
    l1_reconstructed = {}
    step = 10
    for (dx, dy), l1_gt in l1_tiles.items():
        print(f"\n  Processing L1 tile ({dx}, {dy})")

        # Save original and channels
        save_debug_image(l1_gt, out_dir / "compress", step, f"L1_{dx}_{dy}_original.png")
        Y_gt, Cb_gt, Cr_gt = rgb_to_ycbcr_bt601(l1_gt)
        save_debug_image(Y_gt, out_dir / "compress", step+1, f"L1_{dx}_{dy}_luma.png")
        save_debug_image(Cb_gt, out_dir / "compress", step+2, f"L1_{dx}_{dy}_chroma_cb.png")
        save_debug_image(Cr_gt, out_dir / "compress", step+3, f"L1_{dx}_{dy}_chroma_cr.png")

        # Get prediction
        pred = l1_pred[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
        save_debug_image(pred, out_dir / "compress", step+4, f"L1_{dx}_{dy}_prediction.png")
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

        # Compute residual
        residual_raw = Y_gt - Y_pred
        save_debug_image(residual_raw, out_dir / "compress", step+5,
                        f"L1_{dx}_{dy}_residual_raw.png", normalize=True)

        # Quantize
        residual_quant = quantize_residual(residual_raw, quant_levels)
        save_debug_image(residual_quant, out_dir / "compress", step+6,
                        f"L1_{dx}_{dy}_residual_quantized.png", normalize=True)

        # Center (+128)
        residual_centered = np.clip(np.round(residual_quant + 128.0), 0, 255).astype(np.uint8)
        save_debug_image(residual_centered, out_dir / "compress", step+7,
                        f"L1_{dx}_{dy}_residual_centered.png")

        # Save as JPEG
        jpeg_path = save_debug_image(residual_centered, out_dir / "compress",
                                    step+8, f"L1_{dx}_{dy}_residual_jpeg.jpg")

        # Store path for decompression
        residual_paths['L1'][(dx, dy)] = f"{step+8:03d}_L1_{dx}_{dy}_residual_jpeg.jpg"

        # Simulate reconstruction for L0 prediction
        r_dec = np.array(Image.open(jpeg_path).convert("L")).astype(np.float32) - 128.0
        Y_recon = np.clip(Y_pred + r_dec, 0, 255)
        l1_reconstructed[(dx, dy)] = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)

        step += 10

    # Build L1 mosaic for L0 prediction
    l1_mosaic = np.zeros((tile_size*2, tile_size*2, 3), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            if (dx, dy) in l1_reconstructed:
                l1_mosaic[dy*tile_size:(dy+1)*tile_size,
                         dx*tile_size:(dx+1)*tile_size] = l1_reconstructed[(dx, dy)]

    # Upsample for L0 prediction
    l0_pred = np.array(Image.fromarray(l1_mosaic).resize(
        (tile_size*4, tile_size*4), resample=UPSAMPLE))

    # Process L0 tiles
    step = 20
    for (dx, dy), l0_gt in l0_tiles.items():
        print(f"  Processing L0 tile ({dx}, {dy})")

        # Save original and channels
        save_debug_image(l0_gt, out_dir / "compress", step, f"L0_{dx}_{dy}_original.png")
        Y_gt, Cb_gt, Cr_gt = rgb_to_ycbcr_bt601(l0_gt)
        save_debug_image(Y_gt, out_dir / "compress", step+1, f"L0_{dx}_{dy}_luma.png")
        save_debug_image(Cb_gt, out_dir / "compress", step+2, f"L0_{dx}_{dy}_chroma_cb.png")
        save_debug_image(Cr_gt, out_dir / "compress", step+3, f"L0_{dx}_{dy}_chroma_cr.png")

        # Get prediction
        pred = l0_pred[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
        save_debug_image(pred, out_dir / "compress", step+4, f"L0_{dx}_{dy}_prediction.png")
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

        # Compute residual
        residual_raw = Y_gt - Y_pred
        save_debug_image(residual_raw, out_dir / "compress", step+5,
                        f"L0_{dx}_{dy}_residual_raw.png", normalize=True)

        # Quantize
        residual_quant = quantize_residual(residual_raw, quant_levels)
        save_debug_image(residual_quant, out_dir / "compress", step+6,
                        f"L0_{dx}_{dy}_residual_quantized.png", normalize=True)

        # Center (+128)
        residual_centered = np.clip(np.round(residual_quant + 128.0), 0, 255).astype(np.uint8)
        save_debug_image(residual_centered, out_dir / "compress", step+7,
                        f"L0_{dx}_{dy}_residual_centered.png")

        # Save as JPEG
        save_debug_image(residual_centered, out_dir / "compress",
                        step+8, f"L0_{dx}_{dy}_residual_jpeg.jpg")

        if step < 40:  # Only increment for first few to avoid too many files
            step += 10

def decompress_with_debug(out_dir, tile_size=256):
    """Decompress tiles with debug output at each step."""
    compress_dir = pathlib.Path(out_dir) / "compress"
    decompress_dir = pathlib.Path(out_dir) / "decompress"
    decompress_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== DECOMPRESSION PHASE ===")

    # Load L2 tile
    l2_tile = np.array(Image.open(compress_dir / "001_L2_original.png").convert("RGB"))
    save_debug_image(l2_tile, decompress_dir, 50, "L2_decode.png")

    # Upsample for L1 prediction
    UPSAMPLE = Image.Resampling.BILINEAR
    l1_pred = np.array(Image.fromarray(l2_tile).resize(
        (tile_size*2, tile_size*2), resample=UPSAMPLE))
    save_debug_image(l1_pred, decompress_dir, 51, "L1_mosaic_prediction.png")

    # Reconstruct L1 tiles
    l1_reconstructed = {}
    step = 60
    for dy in range(2):
        for dx in range(2):
            print(f"  Reconstructing L1 tile ({dx}, {dy})")

            # Get prediction slice
            pred = l1_pred[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

            # Load residual JPEG - use tile index to find correct file
            tile_idx = dy * 2 + dx
            residual_file = f"{10 + tile_idx * 10 + 8:03d}_L1_{dx}_{dy}_residual_jpeg.jpg"
            residual_path = compress_dir / residual_file
            if residual_path.exists():
                residual_loaded = np.array(Image.open(residual_path).convert("L"))
                save_debug_image(residual_loaded, decompress_dir, step,
                               f"L1_{dx}_{dy}_residual_loaded.png")

                # Decenter (-128)
                residual_decentered = residual_loaded.astype(np.float32) - 128.0
                save_debug_image(residual_decentered, decompress_dir, step+1,
                               f"L1_{dx}_{dy}_residual_decentered.png", normalize=True)

                # Reconstruct luma
                Y_recon = np.clip(Y_pred + residual_decentered, 0, 255)
                save_debug_image(Y_recon, decompress_dir, step+2,
                               f"L1_{dx}_{dy}_luma_reconstructed.png")

                # Full RGB reconstruction
                rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
                save_debug_image(rgb_recon, decompress_dir, step+3,
                               f"L1_{dx}_{dy}_reconstructed.png")

                l1_reconstructed[(dx, dy)] = rgb_recon

    # Build L1 mosaic
    l1_mosaic = np.zeros((tile_size*2, tile_size*2, 3), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            if (dx, dy) in l1_reconstructed:
                l1_mosaic[dy*tile_size:(dy+1)*tile_size,
                         dx*tile_size:(dx+1)*tile_size] = l1_reconstructed[(dx, dy)]
    save_debug_image(l1_mosaic, decompress_dir, 64, "L1_mosaic_full.png")

    # Upsample for L0 prediction
    l0_pred = np.array(Image.fromarray(l1_mosaic).resize(
        (tile_size*4, tile_size*4), resample=UPSAMPLE))
    save_debug_image(l0_pred, decompress_dir, 65, "L0_mosaic_prediction.png")

    # Reconstruct L0 tiles
    step = 70
    for dy in range(4):
        for dx in range(4):
            print(f"  Reconstructing L0 tile ({dx}, {dy})")

            # Get prediction slice
            pred = l0_pred[dy*tile_size:(dy+1)*tile_size, dx*tile_size:(dx+1)*tile_size]
            Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)

            # Load residual JPEG
            residual_path = compress_dir / f"028_L0_{dx}_{dy}_residual_jpeg.jpg"
            if residual_path.exists():
                residual_loaded = np.array(Image.open(residual_path).convert("L"))
                save_debug_image(residual_loaded, decompress_dir, step,
                               f"L0_{dx}_{dy}_residual_loaded.png")

                # Decenter (-128)
                residual_decentered = residual_loaded.astype(np.float32) - 128.0
                save_debug_image(residual_decentered, decompress_dir, step+1,
                               f"L0_{dx}_{dy}_residual_decentered.png", normalize=True)

                # Reconstruct luma
                Y_recon = np.clip(Y_pred + residual_decentered, 0, 255)
                save_debug_image(Y_recon, decompress_dir, step+2,
                               f"L0_{dx}_{dy}_luma_reconstructed.png")

                # Full RGB reconstruction
                rgb_recon = ycbcr_to_rgb_bt601(Y_recon, Cb_pred, Cr_pred)
                save_debug_image(rgb_recon, decompress_dir, step+3,
                               f"L0_{dx}_{dy}_reconstructed.png")

            if step < 90:  # Limit number of files
                step += 10

def main():
    parser = argparse.ArgumentParser(description="Debug version of ORIGAMI compression")
    parser.add_argument("--image", required=True, help="Path to input image (1024x1024)")
    parser.add_argument("--out", required=True, help="Output directory for debug images")
    parser.add_argument("--tile", type=int, default=256, help="Tile size")
    parser.add_argument("--quant", type=int, default=32, help="Quantization levels")
    parser.add_argument("--resq", type=int, default=75, help="JPEG quality for residuals")

    args = parser.parse_args()

    # Tile the input image
    print(f"Processing image: {args.image}")
    l2_tile, l1_tiles, l0_tiles = tile_image(args.image, args.tile)

    # Compress with debug output
    compress_with_debug(l2_tile, l1_tiles, l0_tiles, args.out,
                       args.tile, args.quant, args.resq)

    # Decompress with debug output
    decompress_with_debug(args.out, args.tile)

    # Save configuration
    config = {
        "input_image": str(args.image),
        "tile_size": args.tile,
        "quantization_levels": args.quant,
        "jpeg_quality": args.resq,
        "output_directory": str(args.out)
    }
    config_path = pathlib.Path(args.out) / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print(f"\nDebug images saved to: {args.out}")
    print(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    main()