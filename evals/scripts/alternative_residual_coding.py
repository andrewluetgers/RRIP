#!/usr/bin/env python3
"""
alternative_residual_coding.py

Exploring fundamentally different ways to encode the residual signal.

The insight: JPEG treats the residual as a "generic image" but it's NOT one.
It's a centered-at-128, mostly-zero signal with sparse edge features.
What representations exploit this structure?

Approaches tested:

1. SPARSE REPRESENTATION: The residual is mostly near zero. Instead of encoding
   all 65,536 pixels, encode only the significant ones (position + value).
   Like run-length encoding on steroids.

2. EDGE-AWARE: The residual is large only at edges. If we transmit edge positions
   (from L2, which the decoder already has), we only need values at those positions.

3. WAVELET: DWT is better than DCT for sparse, edge-like signals. Test actual
   wavelet compression of the residual.

4. LEARNED CODEBOOK: Cluster 8x8 residual blocks into a codebook. Each block is
   then just an index (1-2 bytes) into the codebook. Like vector quantization.

5. PREDICTIVE CODING OF THE RESIDUAL: The residual itself has structure — nearby
   pixels have correlated residuals. Delta-encode the residual (residual of residual).

6. BINARY MASK + MAGNITUDE: Separate the sign/position of significant residuals
   from their magnitudes. The mask compresses very well (run-length), and magnitudes
   can be quantized more aggressively.

Usage:
  python alternative_residual_coding.py --image evals/test-images/L0-1024.jpg
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

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("Warning: pywt not available. Install: pip install PyWavelets")


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


def encode_decode_gray(pixels_u8, quality, encoder=JpegEncoder.LIBJPEG_TURBO):
    img = Image.fromarray(pixels_u8, mode="L")
    data = encode_jpeg_to_bytes(img, quality, encoder)
    decoded = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return decoded, len(data)


def get_residuals(image_path):
    """Get L1 residuals from the standard pipeline."""
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

    residuals = {}
    preds = {}
    gts = {}
    for (dx, dy), l1_gt in l1_tiles.items():
        pred = l1_pred_mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256]
        Y_pred, Cb_pred, Cr_pred = rgb_to_ycbcr_bt601(pred)
        Y_gt, _, _ = rgb_to_ycbcr_bt601(l1_gt)
        residuals[(dx, dy)] = Y_gt - Y_pred
        preds[(dx, dy)] = (Y_pred, Cb_pred, Cr_pred)
        gts[(dx, dy)] = l1_gt

    return residuals, preds, gts, l1_tiles, l0_tiles, l2


# ========== ENCODING STRATEGIES ==========

def encode_jpeg_baseline(residual, quality):
    """Standard: center at 128, JPEG encode."""
    centered = np.clip(np.round(residual + 128.0), 0, 255).astype(np.uint8)
    decoded, size = encode_decode_gray(centered, quality)
    recon = decoded.astype(np.float32) - 128.0
    return recon, size


def encode_sparse_threshold(residual, threshold, quantize_bits=5):
    """
    Only encode pixels where |residual| > threshold.
    Encode as: (count, [(row, col, quantized_value), ...])
    Compress with zlib.
    """
    h, w = residual.shape
    mask = np.abs(residual) > threshold

    # Positions and values of significant pixels
    ys, xs = np.where(mask)
    values = residual[mask]

    # Quantize values to N bits (range: -max_val to +max_val)
    max_val = max(np.abs(values).max(), 1)
    n_levels = 2**quantize_bits
    quantized = np.round(values / max_val * (n_levels // 2)).astype(np.int8)

    # Pack: header (4 bytes: h, w, max_val as float16, n_significant) + data
    data = struct.pack('<HHe', h, w, np.float16(max_val))
    data += struct.pack('<I', len(ys))

    for i in range(len(ys)):
        # 2 bytes position (row * w + col), 1 byte value
        pos = ys[i] * w + xs[i]
        data += struct.pack('<Hb', pos, quantized[i])

    compressed = zlib.compress(data, 9)

    # Reconstruct
    recon = np.zeros_like(residual)
    dequantized = quantized.astype(np.float32) / (n_levels // 2) * max_val
    recon[ys, xs] = dequantized

    return recon, len(compressed)


def encode_deadzone_quantize_zlib(residual, step_size):
    """
    Dead-zone quantization + zlib compression.
    Like JPEG quantization but without the DCT — directly quantize spatial pixels.
    Dead zone: values in [-step/2, +step/2] → 0 (most pixels)
    The resulting array is very sparse → compresses well with zlib.
    """
    # Quantize with dead zone
    quantized = np.round(residual / step_size).astype(np.int8)

    # Pack as raw bytes + zlib
    header = struct.pack('<HHf', residual.shape[0], residual.shape[1], step_size)
    data = header + quantized.tobytes()
    compressed = zlib.compress(data, 9)

    # Reconstruct
    recon = quantized.astype(np.float32) * step_size

    return recon, len(compressed)


def encode_wavelet(residual, threshold_pct, wavelet='bior4.4', level=4):
    """
    Wavelet transform → threshold small coefficients → compress.
    Wavelets are much better than DCT for sparse, edge-like signals.
    """
    if not HAS_PYWT:
        return np.zeros_like(residual), float('inf')

    coeffs = pywt.wavedec2(residual, wavelet, level=level)

    # Flatten all coefficients
    all_coeffs = [coeffs[0].ravel()]
    for detail_level in coeffs[1:]:
        for subband in detail_level:
            all_coeffs.append(subband.ravel())
    flat = np.concatenate(all_coeffs)

    # Threshold: keep top N% by magnitude
    threshold = np.percentile(np.abs(flat), 100 - threshold_pct)
    total_coeffs = len(flat)
    kept = np.sum(np.abs(flat) > threshold)

    # Zero out small coefficients
    thresholded_coeffs = [coeffs[0].copy()]
    for detail_level in coeffs[1:]:
        thresholded = []
        for subband in detail_level:
            s = subband.copy()
            s[np.abs(s) <= threshold] = 0
            thresholded.append(s)
        thresholded_coeffs.append(tuple(thresholded))

    # Reconstruct
    recon = pywt.waverec2(thresholded_coeffs, wavelet)
    recon = recon[:residual.shape[0], :residual.shape[1]]

    # Estimate compressed size: quantize kept coefficients + positions
    # Non-zero coefficients: position (2-3 bytes) + value (1-2 bytes)
    # Then zlib compress
    sparse_data = []
    idx = 0
    for i, c in enumerate(all_coeffs):
        nz = np.nonzero(c if i == 0 else
                        (lambda: (thresholded_coeffs[0] if i == 0 else
                                  [s for level in thresholded_coeffs[1:] for s in level][i-1]).ravel())()
                        )[0]

    # Simpler: just quantize the whole coefficient array and zlib it
    quantized = np.round(flat / max(threshold, 0.1)).astype(np.int8)
    # Zero out thresholded coefficients
    quantized[np.abs(flat) <= threshold] = 0
    compressed = zlib.compress(quantized.tobytes(), 9)

    return recon.astype(np.float64), len(compressed)


def encode_vq_codebook(residual, block_size=4, n_clusters=256):
    """
    Vector quantization: divide residual into small blocks, cluster them,
    encode each block as a codebook index (1 byte for 256 clusters).
    """
    from sklearn.cluster import MiniBatchKMeans

    h, w = residual.shape
    bh, bw = h // block_size, w // block_size

    # Extract blocks
    blocks = []
    for by in range(bh):
        for bx in range(bw):
            block = residual[by*block_size:(by+1)*block_size,
                           bx*block_size:(bx+1)*block_size]
            blocks.append(block.ravel())
    blocks = np.array(blocks)

    # Cluster
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
    labels = kmeans.fit_predict(blocks)
    centroids = kmeans.cluster_centers_

    # Size: codebook (n_clusters * block_size^2 * 2 bytes) + indices (n_blocks * 1 byte)
    codebook_size = n_clusters * block_size * block_size * 2  # float16
    index_data = labels.astype(np.uint8).tobytes()
    index_compressed = zlib.compress(index_data, 9)
    total_size = codebook_size + len(index_compressed)

    # Reconstruct
    recon = np.zeros_like(residual)
    idx = 0
    for by in range(bh):
        for bx in range(bw):
            recon[by*block_size:(by+1)*block_size,
                  bx*block_size:(bx+1)*block_size] = centroids[labels[idx]].reshape(block_size, block_size)
            idx += 1

    return recon, total_size


def encode_delta_residual_zlib(residual, step_size):
    """
    Delta-encode the residual: encode differences between adjacent pixels.
    The residual has spatial correlation — adjacent pixels often have similar
    residual values (both near edges). Delta coding exploits this.
    Then quantize + zlib.
    """
    h, w = residual.shape

    # Predict each pixel from its left and top neighbors (median predictor, like PNG)
    predicted = np.zeros_like(residual)
    predicted[0, :] = 0  # first row: predict 0
    predicted[:, 0] = 0  # first col: predict 0
    predicted[1:, 1:] = residual[:-1, 1:]  # predict from above (simple)

    delta = residual - predicted

    # Quantize with dead zone
    quantized = np.round(delta / step_size).astype(np.int8)

    # Compress
    header = struct.pack('<HHf', h, w, step_size)
    compressed = zlib.compress(header + quantized.tobytes(), 9)

    # Reconstruct (sequential, must match encoder)
    recon_delta = quantized.astype(np.float32) * step_size
    recon = np.zeros_like(residual)
    recon[0, :] = recon_delta[0, :]
    recon[:, 0] = recon_delta[:, 0]
    for y in range(1, h):
        for x in range(1, w):
            recon[y, x] = recon[y-1, x] + recon_delta[y, x]

    return recon, len(compressed)


def encode_edge_guided(residual, l2_tile, threshold_multiplier=1.0):
    """
    Use edge information from L2 (which the decoder already has) to guide
    residual encoding. Spend bits only where L2 has edges (= where prediction fails).
    """
    # Detect edges in L2 luma
    Y_l2, _, _ = rgb_to_ycbcr_bt601(l2_tile)

    # Upsample edge map to match residual size
    from scipy.ndimage import sobel
    edge_x = sobel(Y_l2, axis=1)
    edge_y = sobel(Y_l2, axis=0)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)

    # Upsample to residual resolution
    edge_up = np.array(Image.fromarray(edge_mag.astype(np.float32)).resize(
        (residual.shape[1], residual.shape[0]), Image.Resampling.BILINEAR))

    # Threshold: only encode residual where edge strength > threshold
    threshold = np.mean(edge_mag) * threshold_multiplier
    mask = edge_up > threshold
    n_active = np.sum(mask)

    # Encode active pixels with dead-zone quantization
    step_size = 2.0
    quantized = np.round(residual / step_size).astype(np.int8)
    quantized[~mask] = 0  # zero out non-edge regions

    # Compress
    header = struct.pack('<HHf', residual.shape[0], residual.shape[1], step_size)
    # Pack mask as bitfield + quantized values at active positions
    mask_bytes = np.packbits(mask.ravel()).tobytes()
    active_values = quantized[mask].tobytes()
    data = header + struct.pack('<I', len(mask_bytes)) + mask_bytes + active_values
    compressed = zlib.compress(data, 9)

    # Reconstruct
    recon = quantized.astype(np.float32) * step_size

    return recon, len(compressed), n_active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    residuals, preds, gts, l1_tiles, l0_tiles, l2 = get_residuals(args.image)

    print(f"{'='*80}")
    print(f"ALTERNATIVE RESIDUAL CODING — Searching for better representations")
    print(f"{'='*80}")

    # Test on each L1 tile, report averages
    results = {}

    # JPEG baselines at various qualities
    for q in [30, 40, 50, 60, 70, 80]:
        sizes = []
        psnrs = []
        for key, res in residuals.items():
            recon, size = encode_jpeg_baseline(res, q)
            Y_pred = preds[key][0]
            Y_recon = np.clip(Y_pred + recon, 0, 255)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(gts[key])
            p = psnr(Y_gt, Y_recon)
            sizes.append(size)
            psnrs.append(p)
        results[f"JPEG q={q}"] = (np.mean(sizes), np.mean(psnrs))

    # Dead-zone quantize + zlib at various step sizes
    for step in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        sizes = []
        psnrs = []
        for key, res in residuals.items():
            recon, size = encode_deadzone_quantize_zlib(res, step)
            Y_pred = preds[key][0]
            Y_recon = np.clip(Y_pred + recon, 0, 255)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(gts[key])
            p = psnr(Y_gt, Y_recon)
            sizes.append(size)
            psnrs.append(p)
        results[f"Deadzone s={step}"] = (np.mean(sizes), np.mean(psnrs))

    # Sparse threshold
    for thresh in [1, 2, 3, 4, 6, 8, 10]:
        sizes = []
        psnrs = []
        for key, res in residuals.items():
            recon, size = encode_sparse_threshold(res, thresh)
            Y_pred = preds[key][0]
            Y_recon = np.clip(Y_pred + recon, 0, 255)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(gts[key])
            p = psnr(Y_gt, Y_recon)
            sizes.append(size)
            psnrs.append(p)
        results[f"Sparse t={thresh}"] = (np.mean(sizes), np.mean(psnrs))

    # Wavelet
    if HAS_PYWT:
        for keep_pct in [5, 10, 20, 30, 50, 75]:
            sizes = []
            psnrs = []
            for key, res in residuals.items():
                recon, size = encode_wavelet(res, keep_pct)
                Y_pred = preds[key][0]
                Y_recon = np.clip(Y_pred + recon, 0, 255)
                Y_gt, _, _ = rgb_to_ycbcr_bt601(gts[key])
                p = psnr(Y_gt, Y_recon)
                sizes.append(size)
                psnrs.append(p)
            results[f"Wavelet {keep_pct}%"] = (np.mean(sizes), np.mean(psnrs))

    # Vector quantization
    try:
        for bs, nc in [(4, 64), (4, 128), (4, 256), (8, 256), (8, 512)]:
            sizes = []
            psnrs = []
            for key, res in residuals.items():
                recon, size = encode_vq_codebook(res, block_size=bs, n_clusters=nc)
                Y_pred = preds[key][0]
                Y_recon = np.clip(Y_pred + recon, 0, 255)
                Y_gt, _, _ = rgb_to_ycbcr_bt601(gts[key])
                p = psnr(Y_gt, Y_recon)
                sizes.append(size)
                psnrs.append(p)
            results[f"VQ {bs}x{bs}/{nc}"] = (np.mean(sizes), np.mean(psnrs))
    except ImportError:
        print("  (sklearn not available, skipping VQ)")

    # Edge-guided
    for thresh_mult in [0.5, 1.0, 1.5, 2.0]:
        sizes = []
        psnrs = []
        for key, res in residuals.items():
            recon, size, n_active = encode_edge_guided(res, l2, thresh_mult)
            Y_pred = preds[key][0]
            Y_recon = np.clip(Y_pred + recon, 0, 255)
            Y_gt, _, _ = rgb_to_ycbcr_bt601(gts[key])
            p = psnr(Y_gt, Y_recon)
            sizes.append(size)
            psnrs.append(p)
        results[f"Edge-guided m={thresh_mult}"] = (np.mean(sizes), np.mean(psnrs))

    # Sort by size and display
    print(f"\n{'Method':<25s}  {'Avg Size':>10s}  {'Avg PSNR':>10s}")
    print("-" * 50)
    for name, (size, p) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"{name:<25s}  {size/1024:>8.1f}KB  {p:>8.2f}dB")

    # Rate-distortion comparison
    print(f"\n{'='*80}")
    print(f"RATE-DISTORTION: bytes vs PSNR (sorted by PSNR)")
    print(f"{'='*80}")
    print(f"\n{'Method':<25s}  {'Avg Size':>10s}  {'Avg PSNR':>10s}")
    print("-" * 50)
    for name, (size, p) in sorted(results.items(), key=lambda x: -x[1][1]):
        print(f"{name:<25s}  {size/1024:>8.1f}KB  {p:>8.2f}dB")

    # Find Pareto frontier (best PSNR at each size range)
    print(f"\n{'='*80}")
    print(f"PARETO FRONTIER — Best method at each size bracket")
    print(f"{'='*80}")

    sorted_by_size = sorted(results.items(), key=lambda x: x[1][0])
    frontier = []
    best_psnr = -999
    for name, (size, p) in sorted_by_size:
        if p > best_psnr:
            frontier.append((name, size, p))
            best_psnr = p

    print(f"\n{'Method':<25s}  {'Avg Size':>10s}  {'Avg PSNR':>10s}")
    print("-" * 50)
    for name, size, p in frontier:
        print(f"{name:<25s}  {size/1024:>8.1f}KB  {p:>8.2f}dB")


if __name__ == "__main__":
    main()
