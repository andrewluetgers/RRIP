#!/bin/bash
set -euo pipefail

# Build mozjpeg as a static library and install to vendor/mozjpeg/
#
# Usage:
#   ./scripts/build_mozjpeg.sh
#
# Produces:
#   vendor/mozjpeg/lib/libjpeg.a
#   vendor/mozjpeg/include/jpeglib.h (+ other headers)
#
# Prerequisites (macOS):
#   brew install cmake nasm
#
# Prerequisites (Linux):
#   apt-get install cmake nasm build-essential

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$SCRIPT_DIR/../vendor/mozjpeg"
BUILD_DIR="/tmp/mozjpeg-build"

echo "=== Building mozjpeg static library ==="
echo "Install prefix: $PREFIX"

# Detect platform
OS="$(uname -s)"
case "$OS" in
    Darwin)
        NPROC=$(sysctl -n hw.ncpu)
        ;;
    Linux)
        NPROC=$(nproc)
        ;;
    *)
        echo "Unsupported platform: $OS"
        exit 1
        ;;
esac

# Clone
echo "Cloning mozjpeg..."
rm -rf "$BUILD_DIR"
git clone --depth 1 https://github.com/mozilla/mozjpeg.git "$BUILD_DIR"

# Build
echo "Building..."
cd "$BUILD_DIR"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_SHARED_LIBS=OFF \
    -DPNG_SUPPORTED=OFF \
    -DWITH_TURBOJPEG=OFF

cmake --build . -- -j"$NPROC"
cmake --install .

# Cleanup
rm -rf "$BUILD_DIR"

echo ""
echo "=== mozjpeg build complete ==="
echo "Static lib: $PREFIX/lib/libjpeg.a"
echo "Headers:    $PREFIX/include/jpeglib.h"
echo ""
echo "To build origami with mozjpeg:"
echo "  cargo build --features mozjpeg"
