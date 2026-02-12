"""
jpeg_encoder.py

Shared encoder module supporting libjpeg-turbo (Pillow), jpegli (cjpegli CLI),
mozjpeg (vendor/mozjpeg/bin/cjpeg), JPEG XL (cjxl/djxl), and WebP (Pillow).

Usage:
    from jpeg_encoder import JpegEncoder, encode_jpeg_to_file, encode_jpeg_to_bytes

    # Default: libjpeg-turbo via Pillow
    encode_jpeg_to_file(image, "output.jpg", quality=75)

    # With jpegli encoder
    encode_jpeg_to_file(image, "output.jpg", quality=75, encoder=JpegEncoder.JPEGLI)

    # With mozjpeg encoder
    encode_jpeg_to_file(image, "output.jpg", quality=75, encoder=JpegEncoder.MOZJPEG)

    # With JPEG XL encoder (output is .jxl, not .jpg)
    encode_jpeg_to_file(image, "output.jxl", quality=75, encoder=JpegEncoder.JPEGXL)

    # With WebP encoder (output is .webp)
    encode_jpeg_to_file(image, "output.webp", quality=75, encoder=JpegEncoder.WEBP)

    # Get encoded bytes in memory
    data = encode_jpeg_to_bytes(image, quality=75, encoder=JpegEncoder.JPEGXL)

    # Decode JXL back to a PIL Image (for viewer display)
    pil_img = decode_jxl_to_image(jxl_bytes_or_path)
"""

import enum
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

# Resolve mozjpeg cjpeg path relative to this file's repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # evals/scripts -> evals -> repo root
_MOZJPEG_CJPEG = _REPO_ROOT / "vendor" / "mozjpeg" / "bin" / "cjpeg"


class JpegEncoder(enum.Enum):
    LIBJPEG_TURBO = "libjpeg-turbo"
    JPEGLI = "jpegli"
    MOZJPEG = "mozjpeg"
    JPEGXL = "jpegxl"
    WEBP = "webp"


def is_jxl_encoder(encoder: JpegEncoder) -> bool:
    """Return True if the encoder produces JXL output instead of JPEG."""
    return encoder == JpegEncoder.JPEGXL


def is_webp_encoder(encoder: JpegEncoder) -> bool:
    """Return True if the encoder produces WebP output instead of JPEG."""
    return encoder == JpegEncoder.WEBP


# ---------------------------------------------------------------------------
# libjpeg-turbo (Pillow)
# ---------------------------------------------------------------------------

def _encode_libjpeg(image: Image.Image, output_path: str, quality: int) -> int:
    """Encode JPEG using Pillow (libjpeg-turbo). Returns file size in bytes."""
    image.save(output_path, format="JPEG", quality=quality, optimize=True)
    return Path(output_path).stat().st_size


# ---------------------------------------------------------------------------
# jpegli (cjpegli CLI)
# ---------------------------------------------------------------------------

def _check_cjpegli():
    """Check if cjpegli binary is available."""
    return shutil.which("cjpegli") is not None


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


# ---------------------------------------------------------------------------
# mozjpeg (vendor/mozjpeg/bin/cjpeg)
# ---------------------------------------------------------------------------

def _check_mozjpeg():
    """Check if mozjpeg cjpeg binary is available."""
    return _MOZJPEG_CJPEG.exists()


def _encode_mozjpeg_to_file(image: Image.Image, output_path: str, quality: int) -> int:
    """Encode JPEG using mozjpeg cjpeg CLI. Returns file size in bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pnm", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path, format="PPM")
        result = subprocess.run(
            [str(_MOZJPEG_CJPEG), "-quality", str(quality), "-outfile", output_path, tmp_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"mozjpeg cjpeg failed: {result.stderr}")
        return Path(output_path).stat().st_size
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _encode_mozjpeg_to_bytes(image: Image.Image, quality: int) -> bytes:
    """Encode JPEG using mozjpeg cjpeg CLI. Returns JPEG bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pnm", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        tmp_in_path = tmp_in.name
        tmp_out_path = tmp_out.name
    try:
        image.save(tmp_in_path, format="PPM")
        result = subprocess.run(
            [str(_MOZJPEG_CJPEG), "-quality", str(quality), "-outfile", tmp_out_path, tmp_in_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"mozjpeg cjpeg failed: {result.stderr}")
        return Path(tmp_out_path).read_bytes()
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# JPEG XL (cjxl / djxl)
# ---------------------------------------------------------------------------

def _check_cjxl():
    """Check if cjxl binary is available."""
    return shutil.which("cjxl") is not None


def _check_djxl():
    """Check if djxl binary is available."""
    return shutil.which("djxl") is not None


def _encode_jxl_to_file(image: Image.Image, output_path: str, quality: int) -> int:
    """Encode image to JPEG XL using cjxl CLI. Returns file size in bytes."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path, format="PNG")
        result = subprocess.run(
            ["cjxl", tmp_path, output_path, "-q", str(quality)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cjxl failed: {result.stderr}")
        return Path(output_path).stat().st_size
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _encode_jxl_to_bytes(image: Image.Image, quality: int) -> bytes:
    """Encode image to JPEG XL using cjxl CLI. Returns JXL bytes."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as tmp_out:
        tmp_in_path = tmp_in.name
        tmp_out_path = tmp_out.name
    try:
        image.save(tmp_in_path, format="PNG")
        result = subprocess.run(
            ["cjxl", tmp_in_path, tmp_out_path, "-q", str(quality)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cjxl failed: {result.stderr}")
        return Path(tmp_out_path).read_bytes()
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


def decode_jxl_to_image(source) -> Image.Image:
    """Decode a JXL file or bytes to a PIL Image.

    Args:
        source: file path (str/Path) or bytes of JXL data.
    Returns:
        PIL Image (RGB or L depending on input).
    """
    if not _check_djxl():
        raise RuntimeError("djxl not found. Install libjxl.")

    cleanup_input = False
    if isinstance(source, (bytes, bytearray)):
        tmp_in = tempfile.NamedTemporaryFile(suffix=".jxl", delete=False)
        tmp_in.write(source)
        tmp_in.close()
        input_path = tmp_in.name
        cleanup_input = True
    else:
        input_path = str(source)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        result = subprocess.run(
            ["djxl", input_path, tmp_out_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"djxl failed: {result.stderr}")
        return Image.open(tmp_out_path).copy()
    finally:
        if cleanup_input:
            Path(input_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# WebP (Pillow — native support, no external tools needed)
# ---------------------------------------------------------------------------

def _encode_webp_to_file(image: Image.Image, output_path: str, quality: int) -> int:
    """Encode WebP using Pillow. Returns file size in bytes."""
    image.save(output_path, format="WEBP", quality=quality)
    return Path(output_path).stat().st_size


def _encode_webp_to_bytes(image: Image.Image, quality: int) -> bytes:
    """Encode WebP using Pillow. Returns WebP bytes."""
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_jpeg_to_file(
    image: Image.Image,
    output_path,
    quality: int,
    encoder: JpegEncoder = JpegEncoder.LIBJPEG_TURBO,
) -> int:
    """Encode an image to a file. Returns file size in bytes.

    For JPEGXL encoder, output_path should end in .jxl.
    """
    output_path = str(output_path)
    if encoder == JpegEncoder.JPEGLI:
        if not _check_cjpegli():
            raise RuntimeError(
                "cjpegli not found. Install with: ./scripts/build_jpegli.sh"
            )
        return _encode_jpegli_to_file(image, output_path, quality)
    elif encoder == JpegEncoder.MOZJPEG:
        if not _check_mozjpeg():
            raise RuntimeError(
                f"mozjpeg cjpeg not found at {_MOZJPEG_CJPEG}. "
                "Build with: ./scripts/build_mozjpeg.sh"
            )
        return _encode_mozjpeg_to_file(image, output_path, quality)
    elif encoder == JpegEncoder.JPEGXL:
        if not _check_cjxl():
            raise RuntimeError("cjxl not found. Install libjxl.")
        return _encode_jxl_to_file(image, output_path, quality)
    elif encoder == JpegEncoder.WEBP:
        return _encode_webp_to_file(image, output_path, quality)
    else:
        return _encode_libjpeg(image, output_path, quality)


def encode_jpeg_to_bytes(
    image: Image.Image,
    quality: int,
    encoder: JpegEncoder = JpegEncoder.LIBJPEG_TURBO,
) -> bytes:
    """Encode an image to bytes in memory.

    For JPEGXL encoder, returns JXL bytes (not JPEG).
    """
    if encoder == JpegEncoder.JPEGLI:
        if not _check_cjpegli():
            raise RuntimeError(
                "cjpegli not found. Install with: ./scripts/build_jpegli.sh"
            )
        return _encode_jpegli_to_bytes(image, quality)
    elif encoder == JpegEncoder.MOZJPEG:
        if not _check_mozjpeg():
            raise RuntimeError(
                f"mozjpeg cjpeg not found at {_MOZJPEG_CJPEG}. "
                "Build with: ./scripts/build_mozjpeg.sh"
            )
        return _encode_mozjpeg_to_bytes(image, quality)
    elif encoder == JpegEncoder.JPEGXL:
        if not _check_cjxl():
            raise RuntimeError("cjxl not found. Install libjxl.")
        return _encode_jxl_to_bytes(image, quality)
    elif encoder == JpegEncoder.WEBP:
        return _encode_webp_to_bytes(image, quality)
    else:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def file_extension(encoder: JpegEncoder) -> str:
    """Return the file extension for the given encoder (.jpg, .jxl, or .webp)."""
    if encoder == JpegEncoder.JPEGXL:
        return ".jxl"
    if encoder == JpegEncoder.WEBP:
        return ".webp"
    return ".jpg"


def parse_encoder_arg(value: str) -> JpegEncoder:
    """Parse --encoder argument string to JpegEncoder enum."""
    mapping = {
        "libjpeg-turbo": JpegEncoder.LIBJPEG_TURBO,
        "libjpeg": JpegEncoder.LIBJPEG_TURBO,
        "turbo": JpegEncoder.LIBJPEG_TURBO,
        "jpegli": JpegEncoder.JPEGLI,
        "mozjpeg": JpegEncoder.MOZJPEG,
        "jpegxl": JpegEncoder.JPEGXL,
        "jxl": JpegEncoder.JPEGXL,
        "webp": JpegEncoder.WEBP,
    }
    key = value.lower().strip()
    if key not in mapping:
        raise ValueError(
            f"Unknown encoder: {value}. "
            "Choose from: libjpeg-turbo, jpegli, mozjpeg, jpegxl, webp"
        )
    return mapping[key]
