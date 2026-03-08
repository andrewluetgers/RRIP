"""
Utility helpers for the "base codec + luma residual" and "latent-predicted residual" experiments.

Design choices (prototype-friendly):
- Luma: Rec.601 (Y = 0.299 R + 0.587 G + 0.114 B), float32 in [0,1]
- Residual quantization for storage as an "image-like" map:
    q = clamp(round(residual * 255), -127..127)  # signed int8-ish in units of 1/255
    u8_centered = q + 128 in [0..255]
- Residual/correction maps can be stored as JPEG90/95 for apples-to-apples with user request.
- Base codec is JPEG quality=Q (stand-in for JPEG XL). Swap later.
"""

from __future__ import annotations

import io
import math
import zlib
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image


def rgb_to_y(rgb_u8: np.ndarray) -> np.ndarray:
    """Rec.601 luma from uint8 RGB -> float32 [0,1]."""
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def jpeg_encode_decode_rgb(rgb_u8: np.ndarray, quality: int) -> np.ndarray:
    """Encode RGB to JPEG in-memory and decode back to uint8 RGB."""
    buf = io.BytesIO()
    Image.fromarray(rgb_u8, mode="RGB").save(
        buf, format="JPEG", quality=int(quality), subsampling=0, optimize=True
    )
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def psnr01(a01: np.ndarray, b01: np.ndarray) -> float:
    mse = float(np.mean((a01 - b01) ** 2, dtype=np.float64))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def mae01(a01: np.ndarray, b01: np.ndarray) -> float:
    return float(np.mean(np.abs(a01 - b01), dtype=np.float64))


def quantize_residual_int8(residual_float: np.ndarray) -> np.ndarray:
    """
    Quantize float residual in Y domain to int16 in [-127,127] (effectively int8 values).
    Units are 1/255 in Y.
    """
    q = np.round(residual_float * 255.0).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int16)
    return q


def dequantize_residual_int8(q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) / 255.0).astype(np.float32)


def signed_to_u8_centered(q: np.ndarray) -> np.ndarray:
    """Map signed [-127,127] to uint8 centered at 128."""
    u = (q.astype(np.int16) + 128)
    return np.clip(u, 0, 255).astype(np.uint8)


def u8_centered_to_signed(u: np.ndarray) -> np.ndarray:
    """Map centered uint8 [0,255] back to signed [-127,127]."""
    q = (u.astype(np.int16) - 128)
    return np.clip(q, -127, 127).astype(np.int16)


def save_gray01_png(path: str, arr01: np.ndarray) -> None:
    u8 = (np.clip(arr01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def save_rgb_png(path: str, rgb_u8: np.ndarray) -> None:
    Image.fromarray(rgb_u8, mode="RGB").save(path)


def save_centered_int8_as_jpeg(path: str, q_int8: np.ndarray, quality: int) -> None:
    u8 = signed_to_u8_centered(q_int8)
    Image.fromarray(u8, mode="L").save(path, quality=int(quality), subsampling=0, optimize=True)


def load_centered_jpeg_as_int8(path: str) -> np.ndarray:
    u8 = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return u8_centered_to_signed(u8)


def residual_visual(res: np.ndarray, ref_scale: float) -> np.ndarray:
    """Visualize signed residual around 0.5 using maxabs ref_scale."""
    s = float(ref_scale) if ref_scale > 1e-12 else 1e-12
    return np.clip((res / (2.0 * s)) + 0.5, 0.0, 1.0).astype(np.float32)


def error_vis_centered(err: np.ndarray, gain: float = 6.0) -> np.ndarray:
    """
    Visualize signed error as centered u8 with gain.
    Returns uint8 [0..255] (centered at 128).
    """
    q = np.clip(np.round(err * 255.0 * gain), -127, 127).astype(np.int16)
    return signed_to_u8_centered(q)


def pack_latents_int8(z_list: np.ndarray, scale: float, patch: int, H: int, W: int) -> bytes:
    """
    Pack float32 latents z_list (num_patches,zdim) into a simple binary format:
      [4 bytes header_len LE][header JSON][int8 payload ...]
    Header includes scale and patch geometry.
    """
    zq = np.clip(np.round(z_list / scale), -127, 127).astype(np.int8)
    header = {
        "patch": int(patch),
        "H": int(H),
        "W": int(W),
        "zdim": int(z_list.shape[1]),
        "scale": float(scale),
        "dtype": "int8",
        "order": "row-major patches (yy major, xx minor)",
    }
    hb = json.dumps(header).encode("utf-8")
    return len(hb).to_bytes(4, "little") + hb + zq.tobytes()


def zlib_size(data: bytes, level: int = 9) -> int:
    return len(zlib.compress(data, level=level))
