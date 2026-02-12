#!/usr/bin/env python
"""
Create JPEG baseline captures - simple JPEG recompression at various qualities.
This provides a baseline to compare against ORIGAMI compression.
"""

import argparse
import json
import pathlib
import struct
import numpy as np
from PIL import Image
from datetime import datetime
from skimage.metrics import structural_similarity as ssim, mean_squared_error
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
    print("Warning: lz4 library not available, pack files will not be compressed")

# Try to import sewar for VIF
try:
    from sewar.full_ref import vifp
    HAS_VIF = True
except ImportError:
    HAS_VIF = False
    print("Warning: sewar library not available, VIF metrics will be skipped")

# Try to import lpips for perceptual similarity
try:
    import torch
    import lpips as lpips_lib
    _lpips_net = None
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips/torch not available, LPIPS metrics will be skipped")

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = mean_squared_error(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)

def calculate_vif(img1, img2):
    """Calculate Visual Information Fidelity between two images."""
    if not HAS_VIF:
        return None

    try:
        # VIF requires uint8 input
        img1_uint8 = np.clip(img1, 0, 255).astype(np.uint8)
        img2_uint8 = np.clip(img2, 0, 255).astype(np.uint8)

        # Check if images are RGB
        if img1_uint8.ndim == 3 and img1_uint8.shape[2] == 3:
            # For RGB images, use the full image
            return float(vifp(img1_uint8, img2_uint8))
        elif img1_uint8.ndim == 2:
            # For grayscale images
            return float(vifp(img1_uint8, img2_uint8))
        else:
            return None

    except Exception as e:
        print(f"VIF calculation error: {e}")
        return None

def calculate_delta_e(img1, img2):
    """Calculate Delta E (CIE76) color difference between two images."""
    try:
        # Only calculate for RGB images
        if img1.ndim == 3 and img1.shape[2] == 3 and img2.ndim == 3 and img2.shape[2] == 3:
            # Normalize to 0-1 range for skimage
            img1_norm = np.clip(img1 / 255.0, 0, 1)
            img2_norm = np.clip(img2 / 255.0, 0, 1)

            # Convert to LAB color space
            lab1 = rgb2lab(img1_norm)
            lab2 = rgb2lab(img2_norm)

            # Calculate Delta E for each pixel
            delta_e = deltaE_cie76(lab1, lab2)

            # Return mean Delta E
            return float(np.mean(delta_e))
        else:
            # For non-RGB images
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
        # Input images are uint8 [0, 255] RGB (H, W, 3)
        img1_t = torch.from_numpy(img1.copy()).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        img2_t = torch.from_numpy(img2.copy()).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            d = _lpips_net(img1_t, img2_t)
        return float(d.item())
    except Exception as e:
        print(f"LPIPS calculation error: {e}")
        return None

def create_family_pack(tiles_dir, output_dir, tile_size=256):
    """Bundle 20 L0+L1 JPEG tiles into a single .pac file with LZ4 compression.

    Same binary layout as ORIGAMI's pack_residuals():
    - 24-byte header (magic, version, tile_size, entry_count, index_offset, data_offset)
    - 16-byte index entries per tile (level_kind, idx_in_parent, offset, length)
    - Concatenated raw JPEG data
    - LZ4 compressed with 4-byte size prefix
    """
    magic = b"JPGB"
    version = 1
    entries = []  # (level_kind, idx_in_parent, jpeg_bytes)

    # L1: 4 tiles (level_kind=1)
    for dy in range(2):
        for dx in range(2):
            for tile_ext in ['.jpg', '.webp', '.jxl']:
                p = tiles_dir / f"L1_{dx}_{dy}{tile_ext}"
                if p.exists():
                    entries.append((1, dy * 2 + dx, p.read_bytes()))
                    break

    # L0: 16 tiles (level_kind=0)
    for dy in range(4):
        for dx in range(4):
            for tile_ext in ['.jpg', '.webp', '.jxl']:
                p = tiles_dir / f"L0_{dx}_{dy}{tile_ext}"
                if p.exists():
                    entries.append((0, dy * 4 + dx, p.read_bytes()))
                    break

    if not entries:
        return None

    index_offset = 24
    entry_size = 16
    index_size = len(entries) * entry_size
    data_offset = index_offset + index_size
    data = b"".join([e[2] for e in entries])

    # 24-byte header: magic(4) + version(2) + tile_size(2) + entry_count(4) + index_offset(4) + data_offset(4) + reserved(4)
    header = b"".join([
        magic,
        version.to_bytes(2, "little"),
        tile_size.to_bytes(2, "little"),
        len(entries).to_bytes(4, "little"),
        index_offset.to_bytes(4, "little"),
        data_offset.to_bytes(4, "little"),
        (0).to_bytes(4, "little"),
    ])

    # 16-byte index entries: level_kind(1) + idx(1) + pad(2) + offset(4) + length(4) + reserved(4)
    index = []
    cursor = 0
    for level_kind, idx, blob in entries:
        index.append(bytes([level_kind, idx, 0, 0]))
        index.append(cursor.to_bytes(4, "little"))
        index.append(len(blob).to_bytes(4, "little"))
        index.append((0).to_bytes(4, "little"))
        cursor += len(blob)

    pack_data = header + b"".join(index) + data
    uncompressed_size = len(pack_data)

    # LZ4 compress with 4-byte size prefix (same as ORIGAMI)
    if HAS_LZ4:
        compressed_data = uncompressed_size.to_bytes(4, 'little') + lz4.block.compress(
            pack_data, mode='fast', compression=0, store_size=False
        )
    else:
        compressed_data = pack_data

    out_path = output_dir / "tiles.pac"
    out_path.write_bytes(compressed_data)

    ratio = uncompressed_size / len(compressed_data) if HAS_LZ4 else 1.0
    print(f"  Family pack: {uncompressed_size // 1024}KB → {len(compressed_data) // 1024}KB "
          f"(LZ4 ratio: {ratio:.2f}x, {len(entries)} entries)")

    return {
        "path": str(out_path),
        "size": len(compressed_data),
        "uncompressed_size": uncompressed_size,
        "entry_count": len(entries)
    }


def tile_and_compress(image_path, output_dir, jpeg_quality=90, tile_size=256, reference_quality=95, encoder=JpegEncoder.LIBJPEG_TURBO):
    """Create JPEG compressed tiles at specified quality."""

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    # Create output directories
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    manifest = {
        "type": "jpeg_baseline",
        "configuration": {
            "tile_size": tile_size,
            "jpeg_quality": jpeg_quality,
            "reference_quality": reference_quality,
            "input_image": str(image_path),
            "encoder": encoder.value
        },
        "tiles": {},
        "statistics": {
            "total_bytes": 0,
            "tile_count": 0,
            "average_psnr": 0,
            "average_ssim": 0
        }
    }

    use_jxl = is_jxl_encoder(encoder)
    ext = file_extension(encoder)

    # Process L0 tiles (4x4 grid from top-left 1024x1024)
    l0_stats = []
    vif_stats = []
    delta_e_stats = []
    lpips_stats = []
    print(f"Processing L0 tiles (4x4 grid)...")
    for dy in range(4):
        for dx in range(4):
            y_start = dy * tile_size
            x_start = dx * tile_size
            tile = img_array[y_start:y_start+tile_size, x_start:x_start+tile_size]

            tile_img = Image.fromarray(tile)

            if use_jxl:
                # Encode to JXL, measure size, decode to PNG (lossless) for viewer
                jxl_path = tiles_dir / f"L0_{dx}_{dy}.jxl"
                file_size = encode_jpeg_to_file(tile_img, jxl_path, jpeg_quality, encoder)
                decoded_img = decode_jxl_to_image(jxl_path)
                png_path = tiles_dir / f"L0_{dx}_{dy}.png"
                decoded_img.save(str(png_path), format="PNG")
                compressed = np.array(decoded_img.convert("RGB"))
                # Keep both .jxl source and .png for viewer
            else:
                tile_path = tiles_dir / f"L0_{dx}_{dy}{ext}"
                file_size = encode_jpeg_to_file(tile_img, tile_path, jpeg_quality, encoder)
                compressed = np.array(Image.open(tile_path).convert("RGB"))

            # Calculate metrics vs original
            psnr_val = calculate_psnr(tile, compressed)
            ssim_val = ssim(tile, compressed, channel_axis=2, data_range=255)
            mse_val = mean_squared_error(tile, compressed)
            vif_val = calculate_vif(tile, compressed)
            delta_e_val = calculate_delta_e(tile, compressed)
            lpips_val = calculate_lpips(tile, compressed)

            display_ext = ".png" if use_jxl else ext
            tile_info = {
                "file": f"L0_{dx}_{dy}{display_ext}",
                "size_bytes": file_size,
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "mse": float(mse_val)
            }

            if vif_val is not None:
                tile_info["vif"] = vif_val
            if delta_e_val is not None:
                tile_info["delta_e"] = delta_e_val
            if lpips_val is not None:
                tile_info["lpips"] = lpips_val

            manifest["tiles"][f"L0_{dx}_{dy}"] = tile_info
            manifest["statistics"]["total_bytes"] += file_size
            l0_stats.append((psnr_val, ssim_val))
            if vif_val is not None:
                vif_stats.append(vif_val)
            if delta_e_val is not None:
                delta_e_stats.append(delta_e_val)
            if lpips_val is not None:
                lpips_stats.append(lpips_val)

    # Process L1 tiles (2x2 grid, downsampled from 1024x1024 to 512x512)
    print(f"Processing L1 tiles (2x2 grid)...")
    l1_source = img_array[:1024, :1024]
    l1_downsampled = Image.fromarray(l1_source).resize((512, 512), Image.LANCZOS)
    l1_array = np.array(l1_downsampled)

    for dy in range(2):
        for dx in range(2):
            y_start = dy * tile_size
            x_start = dx * tile_size
            tile = l1_array[y_start:y_start+tile_size, x_start:x_start+tile_size]

            tile_img = Image.fromarray(tile)

            if use_jxl:
                jxl_path = tiles_dir / f"L1_{dx}_{dy}.jxl"
                file_size = encode_jpeg_to_file(tile_img, jxl_path, jpeg_quality, encoder)
                decoded_img = decode_jxl_to_image(jxl_path)
                png_path = tiles_dir / f"L1_{dx}_{dy}.png"
                decoded_img.save(str(png_path), format="PNG")
                compressed = np.array(decoded_img.convert("RGB"))
            else:
                tile_path = tiles_dir / f"L1_{dx}_{dy}{ext}"
                file_size = encode_jpeg_to_file(tile_img, tile_path, jpeg_quality, encoder)
                compressed = np.array(Image.open(tile_path).convert("RGB"))

            # Calculate metrics vs original
            psnr_val = calculate_psnr(tile, compressed)
            ssim_val = ssim(tile, compressed, channel_axis=2, data_range=255)
            mse_val = mean_squared_error(tile, compressed)
            vif_val = calculate_vif(tile, compressed)
            delta_e_val = calculate_delta_e(tile, compressed)
            lpips_val = calculate_lpips(tile, compressed)

            display_ext = ".png" if use_jxl else ext
            tile_info = {
                "file": f"L1_{dx}_{dy}{display_ext}",
                "size_bytes": file_size,
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "mse": float(mse_val)
            }

            if vif_val is not None:
                tile_info["vif"] = vif_val
            if delta_e_val is not None:
                tile_info["delta_e"] = delta_e_val
            if lpips_val is not None:
                tile_info["lpips"] = lpips_val

            manifest["tiles"][f"L1_{dx}_{dy}"] = tile_info
            manifest["statistics"]["total_bytes"] += file_size
            l0_stats.append((psnr_val, ssim_val))
            if vif_val is not None:
                vif_stats.append(vif_val)
            if delta_e_val is not None:
                delta_e_stats.append(delta_e_val)
            if lpips_val is not None:
                lpips_stats.append(lpips_val)

    # Process L2 tile (single tile, downsampled from 1024x1024 to 256x256)
    print(f"Processing L2 tile...")
    l2_source = img_array[:1024, :1024]
    l2_downsampled = Image.fromarray(l2_source).resize((256, 256), Image.LANCZOS)
    l2_array = np.array(l2_downsampled)

    # Save at specified quality
    tile_img = Image.fromarray(l2_array)

    if use_jxl:
        jxl_path = tiles_dir / "L2_0_0.jxl"
        file_size = encode_jpeg_to_file(tile_img, jxl_path, jpeg_quality, encoder)
        decoded_img = decode_jxl_to_image(jxl_path)
        png_path = tiles_dir / "L2_0_0.png"
        decoded_img.save(str(png_path), format="PNG")
        compressed = np.array(decoded_img.convert("RGB"))
    else:
        tile_path = tiles_dir / f"L2_0_0{ext}"
        file_size = encode_jpeg_to_file(tile_img, tile_path, jpeg_quality, encoder)
        compressed = np.array(Image.open(tile_path).convert("RGB"))

    # Calculate metrics vs original
    psnr_val = calculate_psnr(l2_array, compressed)
    ssim_val = ssim(l2_array, compressed, channel_axis=2, data_range=255)
    mse_val = mean_squared_error(l2_array, compressed)
    vif_val = calculate_vif(l2_array, compressed)
    delta_e_val = calculate_delta_e(l2_array, compressed)
    lpips_val = calculate_lpips(l2_array, compressed)

    display_ext = ".png" if use_jxl else ext
    tile_info = {
        "file": f"L2_0_0{display_ext}",
        "size_bytes": file_size,
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "mse": float(mse_val)
    }

    if vif_val is not None:
        tile_info["vif"] = vif_val
    if delta_e_val is not None:
        tile_info["delta_e"] = delta_e_val
    if lpips_val is not None:
        tile_info["lpips"] = lpips_val

    manifest["tiles"]["L2_0_0"] = tile_info
    manifest["statistics"]["total_bytes"] += file_size
    l0_stats.append((psnr_val, ssim_val))
    if vif_val is not None:
        vif_stats.append(vif_val)
    if delta_e_val is not None:
        delta_e_stats.append(delta_e_val)
    if lpips_val is not None:
        lpips_stats.append(lpips_val)

    # Calculate statistics
    manifest["statistics"]["tile_count"] = len(manifest["tiles"])

    # Average metrics
    if l0_stats:
        avg_psnr = np.mean([p for p, s in l0_stats])
        avg_ssim = np.mean([s for p, s in l0_stats])
        manifest["statistics"]["average_psnr"] = float(avg_psnr)
        manifest["statistics"]["average_ssim"] = float(avg_ssim)

    if vif_stats:
        manifest["statistics"]["average_vif"] = float(np.mean(vif_stats))

    if delta_e_stats:
        manifest["statistics"]["average_delta_e"] = float(np.mean(delta_e_stats))

    if lpips_stats:
        manifest["statistics"]["average_lpips"] = float(np.mean(lpips_stats))

    # Calculate reference size (at quality 95)
    reference_bytes = 0
    for dy in range(4):
        for dx in range(4):
            reference_bytes += 27 * 1024  # ~27KB per L0 tile at Q95
    for dy in range(2):
        for dx in range(2):
            reference_bytes += 27.5 * 1024  # ~27.5KB per L1 tile at Q95
    reference_bytes += 28.5 * 1024  # ~28.5KB for L2 tile at Q95

    manifest["statistics"]["reference_bytes"] = int(reference_bytes)
    manifest["statistics"]["compression_ratio"] = reference_bytes / manifest["statistics"]["total_bytes"]
    manifest["statistics"]["space_savings_pct"] = 100 * (1 - manifest["statistics"]["total_bytes"] / reference_bytes)

    # Create family pack (20 tiles: 4 L1 + 16 L0, bundled with LZ4)
    print("Creating family pack...")
    pack_info = create_family_pack(tiles_dir, output_dir, tile_size)
    if pack_info:
        manifest["pack"] = pack_info

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Create summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"JPEG BASELINE COMPRESSION (Q{jpeg_quality})\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input Image: {image_path}\n")
        f.write(f"JPEG Quality: {jpeg_quality}\n")
        f.write(f"Total Size: {manifest['statistics']['total_bytes']:,} bytes\n")
        f.write(f"Reference Size (Q95): {int(reference_bytes):,} bytes\n")
        f.write(f"Compression Ratio: {manifest['statistics']['compression_ratio']:.2f}x\n")
        f.write(f"Space Savings: {manifest['statistics']['space_savings_pct']:.1f}%\n")
        f.write(f"Average PSNR: {manifest['statistics']['average_psnr']:.2f} dB\n")
        f.write(f"Average SSIM: {manifest['statistics']['average_ssim']:.4f}\n")
        if 'average_vif' in manifest['statistics']:
            f.write(f"Average VIF: {manifest['statistics']['average_vif']:.4f}\n")
        if 'average_delta_e' in manifest['statistics']:
            f.write(f"Average Delta E: {manifest['statistics']['average_delta_e']:.2f}\n")
        if 'average_lpips' in manifest['statistics']:
            f.write(f"Average LPIPS: {manifest['statistics']['average_lpips']:.4f}\n")

    print(f"\n{'-' * 50}")
    print(f"JPEG Q{jpeg_quality} Baseline Complete")
    print(f"Total Size: {manifest['statistics']['total_bytes']:,} bytes")
    print(f"Compression Ratio: {manifest['statistics']['compression_ratio']:.2f}x")
    print(f"Average PSNR: {manifest['statistics']['average_psnr']:.2f} dB")
    print(f"Average SSIM: {manifest['statistics']['average_ssim']:.4f}")
    if 'average_vif' in manifest['statistics']:
        print(f"Average VIF: {manifest['statistics']['average_vif']:.4f}")
    if 'average_delta_e' in manifest['statistics']:
        print(f"Average Delta E: {manifest['statistics']['average_delta_e']:.2f}")
    if 'average_lpips' in manifest['statistics']:
        print(f"Average LPIPS: {manifest['statistics']['average_lpips']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Create JPEG baseline captures")
    parser.add_argument("--image", required=True, help="Path to input image (1024x1024)")
    parser.add_argument("--quality", type=int, required=True, help="JPEG quality (1-100)")
    parser.add_argument("--output", help="Output directory (auto-generated if not specified)")
    parser.add_argument("--encoder", default="libjpeg-turbo",
                       help="Encoder: libjpeg-turbo, jpegli, mozjpeg, jpegxl, or webp")

    args = parser.parse_args()
    encoder = parse_encoder_arg(args.encoder)

    if args.output is None:
        # Auto-generate: {encoder}_jpeg_baseline_q{Q}
        # libjpeg-turbo is the default, no prefix needed
        if encoder == JpegEncoder.LIBJPEG_TURBO:
            prefix = ""
        else:
            prefix = encoder.value + "_"
        args.output = f"evals/runs/{prefix}jpeg_baseline_q{args.quality}"

    print(f"Creating baseline at quality {args.quality} with {encoder.value}")
    print(f"Output directory: {args.output}")

    tile_and_compress(args.image, args.output, jpeg_quality=args.quality, encoder=encoder)

if __name__ == "__main__":
    main()