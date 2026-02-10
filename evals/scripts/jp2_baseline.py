#!/usr/bin/env python
"""
Create JPEG2000 baseline captures for comparison against ORIGAMI compression.

Unlike JPEG where each tile is independently encoded, JPEG2000's wavelet-based
compression stores multi-resolution data in a single file. A single J2K file
can serve all 3 resolution levels (L0 full-res, L1 half-res, L2 quarter-res).

For fair comparison, per-tile byte cost = total J2K file size / 21
(the number of tiles: 16 L0 + 4 L1 + 1 L2).

Encoding uses opj_compress (OpenJPEG) when available, with Pillow fallback.
Multi-resolution decoding uses opj_decompress -r (reduce) to leverage J2K's
native DWT levels, falling back to Pillow resize.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim, mean_squared_error

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

# Check for OpenJPEG tools
HAS_OPJ_COMPRESS = shutil.which("opj_compress") is not None
HAS_OPJ_DECOMPRESS = shutil.which("opj_decompress") is not None

TILES_IN_FAMILY = 21  # 16 L0 + 4 L1 + 1 L2


def quality_to_rate(quality):
    """Map JPEG-style quality (1-100) to JPEG2000 compression rate.

    Higher quality → lower rate (less compression).
    Rate is the target compression ratio (e.g. rate=20 means 20:1).

    Mapping:
      Q100 → rate 1 (lossless-ish)
      Q90  → rate 5
      Q75  → rate 12
      Q60  → rate 25
      Q50  → rate 35
      Q30  → rate 55
      Q10  → rate 80
      Q1   → rate 100
    """
    # Exponential mapping: rate = a * exp(-b * quality)
    # Tuned so Q90→~5, Q60→~25, Q30→~55
    import math
    if quality >= 100:
        return 1.0
    if quality <= 1:
        return 100.0
    rate = 105.0 * math.exp(-0.032 * quality)
    return max(1.0, round(rate, 1))


def encode_jp2(image_path, output_path, quality):
    """Encode image to JPEG2000. Returns file size in bytes.

    Uses opj_compress when available, falls back to Pillow.
    """
    rate = quality_to_rate(quality)

    if HAS_OPJ_COMPRESS:
        # opj_compress -r <rate> provides direct compression ratio control
        # Input must be a format opj_compress can read (BMP, PGM, PPM, PNM, PAM, PGX, PNG, TIF)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # Ensure input is saved as PNG for opj_compress
            img = Image.open(image_path).convert("RGB")
            img.save(tmp_path, format="PNG")

            result = subprocess.run(
                ["opj_compress", "-i", tmp_path, "-o", str(output_path), "-r", str(rate)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"opj_compress failed: {result.stderr}")
            return pathlib.Path(output_path).stat().st_size
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)
    else:
        # Pillow fallback: quality_mode='rates', quality_layers=[rate]
        img = Image.open(image_path).convert("RGB")
        img.save(str(output_path), format="JPEG2000",
                 quality_mode="rates", quality_layers=[rate])
        return pathlib.Path(output_path).stat().st_size


def encode_jp2_from_image(img, output_path, quality):
    """Encode a PIL Image to JPEG2000. Returns file size in bytes."""
    rate = quality_to_rate(quality)

    if HAS_OPJ_COMPRESS:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img.save(tmp_path, format="PNG")
            result = subprocess.run(
                ["opj_compress", "-i", tmp_path, "-o", str(output_path), "-r", str(rate)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"opj_compress failed: {result.stderr}")
            return pathlib.Path(output_path).stat().st_size
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)
    else:
        img.save(str(output_path), format="JPEG2000",
                 quality_mode="rates", quality_layers=[rate])
        return pathlib.Path(output_path).stat().st_size


def decode_jp2_at_resolution(jp2_path, target_size):
    """Decode a JPEG2000 file at a target resolution.

    Uses opj_decompress -r (reduce factor) when available to leverage
    J2K's native DWT levels. Falls back to Pillow resize.

    Args:
        jp2_path: Path to .jp2 file
        target_size: (width, height) tuple

    Returns:
        numpy array (H, W, 3) RGB uint8
    """
    jp2_path = str(jp2_path)

    # Determine reduce factor from size ratio
    full_img = Image.open(jp2_path)
    full_w, full_h = full_img.size
    target_w, target_h = target_size

    if full_w == target_w and full_h == target_h:
        return np.array(full_img.convert("RGB"))

    # Calculate reduce level: each level halves resolution
    reduce = 0
    w, h = full_w, full_h
    while w > target_w and h > target_h:
        w //= 2
        h //= 2
        reduce += 1

    if HAS_OPJ_DECOMPRESS and reduce > 0:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["opj_decompress", "-i", jp2_path, "-o", tmp_path, "-r", str(reduce)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                decoded = Image.open(tmp_path).convert("RGB")
                # opj_decompress may not produce exact target size, resize if needed
                if decoded.size != (target_w, target_h):
                    decoded = decoded.resize((target_w, target_h), Image.LANCZOS)
                return np.array(decoded)
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    # Fallback: decode full and resize
    return np.array(full_img.convert("RGB").resize(target_size, Image.LANCZOS))


# --- Metric functions (same as jpeg_baseline.py) ---

def calculate_psnr(img1, img2):
    mse = mean_squared_error(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_vif(img1, img2):
    if not HAS_VIF:
        return None
    try:
        img1_uint8 = np.clip(img1, 0, 255).astype(np.uint8)
        img2_uint8 = np.clip(img2, 0, 255).astype(np.uint8)
        if img1_uint8.ndim == 3 and img1_uint8.shape[2] == 3:
            return float(vifp(img1_uint8, img2_uint8))
        elif img1_uint8.ndim == 2:
            return float(vifp(img1_uint8, img2_uint8))
        return None
    except Exception as e:
        print(f"VIF calculation error: {e}")
        return None


def calculate_delta_e(img1, img2):
    try:
        if img1.ndim == 3 and img1.shape[2] == 3 and img2.ndim == 3 and img2.shape[2] == 3:
            from skimage.color import rgb2lab, deltaE_cie76
            img1_norm = np.clip(img1 / 255.0, 0, 1)
            img2_norm = np.clip(img2 / 255.0, 0, 1)
            lab1 = rgb2lab(img1_norm)
            lab2 = rgb2lab(img2_norm)
            delta_e = deltaE_cie76(lab1, lab2)
            return float(np.mean(delta_e))
        return None
    except Exception as e:
        print(f"Delta E calculation error: {e}")
        return None


def calculate_lpips(img1, img2):
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


def compute_tile_metrics(reference, compressed):
    """Compute all metrics for a tile pair. Returns dict of metric values."""
    metrics = {
        "psnr": float(calculate_psnr(reference, compressed)),
        "ssim": float(ssim(reference, compressed, channel_axis=2, data_range=255)),
        "mse": float(mean_squared_error(reference, compressed)),
    }
    vif_val = calculate_vif(reference, compressed)
    if vif_val is not None:
        metrics["vif"] = vif_val
    delta_e_val = calculate_delta_e(reference, compressed)
    if delta_e_val is not None:
        metrics["delta_e"] = delta_e_val
    lpips_val = calculate_lpips(reference, compressed)
    if lpips_val is not None:
        metrics["lpips"] = lpips_val
    return metrics


def jp2_baseline(image_path, output_dir, quality=60, tile_size=256):
    """Create JPEG2000 baseline: encode full image as single J2K, extract tiles for metrics."""

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    # Step 1: Encode full 1024x1024 image as single JPEG2000 file
    jp2_path = output_dir / "full_image.jp2"
    print(f"Encoding full image as JPEG2000 (quality={quality}, rate={quality_to_rate(quality)})...")
    total_file_size = encode_jp2_from_image(img, jp2_path, quality)
    per_tile_size = total_file_size / TILES_IN_FAMILY
    print(f"  Total J2K size: {total_file_size:,} bytes ({total_file_size/1024:.1f} KB)")
    print(f"  Per-tile cost (/{TILES_IN_FAMILY}): {per_tile_size:,.0f} bytes ({per_tile_size/1024:.1f} KB)")

    manifest = {
        "type": "jp2_baseline",
        "configuration": {
            "tile_size": tile_size,
            "jp2_quality": quality,
            "jp2_rate": quality_to_rate(quality),
            "input_image": str(image_path),
            "total_file_size": total_file_size,
            "tiles_in_family": TILES_IN_FAMILY,
            "encoder": "opj_compress" if HAS_OPJ_COMPRESS else "pillow",
        },
        "tiles": {},
        "statistics": {
            "total_bytes": 0,
            "tile_count": 0,
        }
    }

    all_psnr = []
    all_ssim = []
    all_vif = []
    all_delta_e = []
    all_lpips = []

    # Step 2: Decode at full resolution → extract 16 L0 tiles
    print("Processing L0 tiles (4x4 grid)...")
    l0_decoded = decode_jp2_at_resolution(jp2_path, (1024, 1024))

    for dy in range(4):
        for dx in range(4):
            y_start = dy * tile_size
            x_start = dx * tile_size
            ref_tile = img_array[y_start:y_start+tile_size, x_start:x_start+tile_size]
            dec_tile = l0_decoded[y_start:y_start+tile_size, x_start:x_start+tile_size]

            # Save decoded tile as PNG (lossless, no re-encoding artifacts)
            tile_img = Image.fromarray(dec_tile)
            tile_path = tiles_dir / f"L0_{dx}_{dy}.png"
            tile_img.save(str(tile_path), format="PNG")

            metrics = compute_tile_metrics(ref_tile, dec_tile)
            tile_info = {
                "file": f"L0_{dx}_{dy}.png",
                "size_bytes": int(round(per_tile_size)),
                **metrics,
            }
            manifest["tiles"][f"L0_{dx}_{dy}"] = tile_info
            manifest["statistics"]["total_bytes"] += int(round(per_tile_size))
            all_psnr.append(metrics["psnr"])
            all_ssim.append(metrics["ssim"])
            if "vif" in metrics:
                all_vif.append(metrics["vif"])
            if "delta_e" in metrics:
                all_delta_e.append(metrics["delta_e"])
            if "lpips" in metrics:
                all_lpips.append(metrics["lpips"])

    # Step 3: Decode at half resolution → extract 4 L1 tiles
    print("Processing L1 tiles (2x2 grid)...")
    l1_decoded = decode_jp2_at_resolution(jp2_path, (512, 512))

    # Reference: downsample original to 512x512
    l1_ref = np.array(Image.fromarray(img_array[:1024, :1024]).resize((512, 512), Image.LANCZOS))

    for dy in range(2):
        for dx in range(2):
            y_start = dy * tile_size
            x_start = dx * tile_size
            ref_tile = l1_ref[y_start:y_start+tile_size, x_start:x_start+tile_size]
            dec_tile = l1_decoded[y_start:y_start+tile_size, x_start:x_start+tile_size]

            tile_img = Image.fromarray(dec_tile)
            tile_path = tiles_dir / f"L1_{dx}_{dy}.png"
            tile_img.save(str(tile_path), format="PNG")

            metrics = compute_tile_metrics(ref_tile, dec_tile)
            tile_info = {
                "file": f"L1_{dx}_{dy}.png",
                "size_bytes": int(round(per_tile_size)),
                **metrics,
            }
            manifest["tiles"][f"L1_{dx}_{dy}"] = tile_info
            manifest["statistics"]["total_bytes"] += int(round(per_tile_size))
            all_psnr.append(metrics["psnr"])
            all_ssim.append(metrics["ssim"])
            if "vif" in metrics:
                all_vif.append(metrics["vif"])
            if "delta_e" in metrics:
                all_delta_e.append(metrics["delta_e"])
            if "lpips" in metrics:
                all_lpips.append(metrics["lpips"])

    # Step 4: Decode at quarter resolution → 1 L2 tile
    print("Processing L2 tile...")
    l2_decoded = decode_jp2_at_resolution(jp2_path, (256, 256))

    # Reference: downsample original to 256x256
    l2_ref = np.array(Image.fromarray(img_array[:1024, :1024]).resize((256, 256), Image.LANCZOS))

    tile_img = Image.fromarray(l2_decoded)
    tile_path = tiles_dir / "L2_0_0.png"
    tile_img.save(str(tile_path), format="PNG")

    metrics = compute_tile_metrics(l2_ref, l2_decoded)
    tile_info = {
        "file": "L2_0_0.png",
        "size_bytes": int(round(per_tile_size)),
        **metrics,
    }
    manifest["tiles"]["L2_0_0"] = tile_info
    manifest["statistics"]["total_bytes"] += int(round(per_tile_size))
    all_psnr.append(metrics["psnr"])
    all_ssim.append(metrics["ssim"])
    if "vif" in metrics:
        all_vif.append(metrics["vif"])
    if "delta_e" in metrics:
        all_delta_e.append(metrics["delta_e"])
    if "lpips" in metrics:
        all_lpips.append(metrics["lpips"])

    # Compute aggregate statistics
    manifest["statistics"]["tile_count"] = len(manifest["tiles"])
    manifest["statistics"]["average_psnr"] = float(np.mean(all_psnr))
    manifest["statistics"]["average_ssim"] = float(np.mean(all_ssim))
    if all_vif:
        manifest["statistics"]["average_vif"] = float(np.mean(all_vif))
    if all_delta_e:
        manifest["statistics"]["average_delta_e"] = float(np.mean(all_delta_e))
    if all_lpips:
        manifest["statistics"]["average_lpips"] = float(np.mean(all_lpips))

    # Reference size estimate (same as jpeg_baseline.py)
    reference_bytes = 16 * 27 * 1024 + 4 * 27.5 * 1024 + 28.5 * 1024
    manifest["statistics"]["reference_bytes"] = int(reference_bytes)
    manifest["statistics"]["compression_ratio"] = reference_bytes / total_file_size
    manifest["statistics"]["space_savings_pct"] = 100 * (1 - total_file_size / reference_bytes)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print(f"\n{'-' * 50}")
    print(f"JP2 Q{quality} Baseline Complete")
    print(f"Total J2K Size: {total_file_size:,} bytes")
    print(f"Per-tile Cost: {per_tile_size:,.0f} bytes")
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
    parser = argparse.ArgumentParser(description="Create JPEG2000 baseline captures")
    parser.add_argument("--image", required=True, help="Path to input image (1024x1024)")
    parser.add_argument("--quality", type=int, required=True,
                        help="JPEG-equivalent quality (1-100), mapped to J2K compression rate")
    parser.add_argument("--output", help="Output directory (auto-generated if not specified)")

    args = parser.parse_args()

    if args.output is None:
        args.output = f"evals/runs/jp2_baseline_q{args.quality}"

    encoder_name = "opj_compress" if HAS_OPJ_COMPRESS else "Pillow"
    decoder_name = "opj_decompress" if HAS_OPJ_DECOMPRESS else "Pillow"
    print(f"JPEG2000 baseline at quality {args.quality} (rate={quality_to_rate(args.quality)})")
    print(f"Encoder: {encoder_name}, Decoder: {decoder_name}")
    print(f"Output directory: {args.output}")

    jp2_baseline(args.image, args.output, quality=args.quality)


if __name__ == "__main__":
    main()
