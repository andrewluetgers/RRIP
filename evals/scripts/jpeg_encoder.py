"""
jpeg_encoder.py

Shared JPEG encoder module supporting libjpeg-turbo (via Pillow) and jpegli (via cjpegli CLI).

Usage:
    from jpeg_encoder import JpegEncoder, encode_jpeg_to_file, encode_jpeg_to_bytes

    # Default: libjpeg-turbo via Pillow
    encode_jpeg_to_file(image, "output.jpg", quality=75)

    # With jpegli encoder
    encode_jpeg_to_file(image, "output.jpg", quality=75, encoder=JpegEncoder.JPEGLI)

    # Get JPEG bytes in memory
    data = encode_jpeg_to_bytes(image, quality=75, encoder=JpegEncoder.JPEGLI)
"""

import enum
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


class JpegEncoder(enum.Enum):
    LIBJPEG_TURBO = "libjpeg-turbo"
    JPEGLI = "jpegli"


def _check_cjpegli():
    """Check if cjpegli binary is available."""
    return shutil.which("cjpegli") is not None


def _encode_libjpeg(image: Image.Image, output_path: str, quality: int) -> int:
    """Encode JPEG using Pillow (libjpeg-turbo). Returns file size in bytes."""
    image.save(output_path, format="JPEG", quality=quality, optimize=True)
    return Path(output_path).stat().st_size


def _encode_jpegli_to_file(image: Image.Image, output_path: str, quality: int) -> int:
    """Encode JPEG using cjpegli CLI. Returns file size in bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pnm", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Pillow writes PGM for mode "L", PPM for mode "RGB"
        image.save(tmp_path, format="PPM")
        result = subprocess.run(
            ["cjpegli", tmp_path, output_path, "-q", str(quality)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cjpegli failed: {result.stderr}")
        return Path(output_path).stat().st_size
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _encode_jpegli_to_bytes(image: Image.Image, quality: int) -> bytes:
    """Encode JPEG using cjpegli CLI. Returns JPEG bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pnm", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        tmp_in_path = tmp_in.name
        tmp_out_path = tmp_out.name
    try:
        image.save(tmp_in_path, format="PPM")
        result = subprocess.run(
            ["cjpegli", tmp_in_path, tmp_out_path, "-q", str(quality)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cjpegli failed: {result.stderr}")
        return Path(tmp_out_path).read_bytes()
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


def encode_jpeg_to_file(
    image: Image.Image,
    output_path,
    quality: int,
    encoder: JpegEncoder = JpegEncoder.LIBJPEG_TURBO,
) -> int:
    """Encode an image to a JPEG file. Returns file size in bytes."""
    output_path = str(output_path)
    if encoder == JpegEncoder.JPEGLI:
        if not _check_cjpegli():
            raise RuntimeError(
                "cjpegli not found. Install with: ./scripts/install_jpegli.sh"
            )
        return _encode_jpegli_to_file(image, output_path, quality)
    else:
        return _encode_libjpeg(image, output_path, quality)


def encode_jpeg_to_bytes(
    image: Image.Image,
    quality: int,
    encoder: JpegEncoder = JpegEncoder.LIBJPEG_TURBO,
) -> bytes:
    """Encode an image to JPEG bytes in memory."""
    if encoder == JpegEncoder.JPEGLI:
        if not _check_cjpegli():
            raise RuntimeError(
                "cjpegli not found. Install with: ./scripts/install_jpegli.sh"
            )
        return _encode_jpegli_to_bytes(image, quality)
    else:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def parse_encoder_arg(value: str) -> JpegEncoder:
    """Parse --encoder argument string to JpegEncoder enum."""
    mapping = {
        "libjpeg-turbo": JpegEncoder.LIBJPEG_TURBO,
        "libjpeg": JpegEncoder.LIBJPEG_TURBO,
        "turbo": JpegEncoder.LIBJPEG_TURBO,
        "jpegli": JpegEncoder.JPEGLI,
    }
    key = value.lower().strip()
    if key not in mapping:
        raise ValueError(
            f"Unknown encoder: {value}. Choose from: libjpeg-turbo, jpegli"
        )
    return mapping[key]
