#!/bin/bash
set -euo pipefail

# Install jpegli (cjpegli/djpegli) from Google's jpegli repository.
# jpegli achieves ~35% better JPEG compression than libjpeg-turbo at equivalent quality.
#
# Usage:
#   ./scripts/install_jpegli.sh
#
# Prerequisites (macOS):
#   brew install cmake giflib jpeg-turbo libpng zlib
#
# Prerequisites (Linux):
#   apt-get install cmake build-essential libgif-dev libjpeg-turbo8-dev libpng-dev zlib1g-dev

INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
BUILD_DIR="${BUILD_DIR:-/tmp/jpegli}"

echo "=== Installing jpegli (cjpegli/djpegli) ==="

# Check for existing installation
if command -v cjpegli &>/dev/null; then
    echo "cjpegli is already installed at $(which cjpegli)"
    cjpegli --help 2>&1 | head -1 || true
    echo ""
    read -p "Reinstall? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping install."
        exit 0
    fi
fi

# Detect platform
OS="$(uname -s)"
case "$OS" in
    Darwin)
        echo "Platform: macOS"

        # Check for Homebrew dependencies
        for dep in cmake giflib jpeg-turbo libpng zlib; do
            if ! brew list "$dep" &>/dev/null; then
                echo "Missing dependency: $dep"
                echo "Install with: brew install $dep"
                exit 1
            fi
        done

        export CMAKE_PREFIX_PATH="$(brew --prefix giflib):$(brew --prefix jpeg-turbo):$(brew --prefix libpng):$(brew --prefix zlib)"
        NPROC=$(sysctl -n hw.ncpu)
        ;;
    Linux)
        echo "Platform: Linux"
        NPROC=$(nproc)
        ;;
    *)
        echo "Unsupported platform: $OS"
        exit 1
        ;;
esac

# Clone and build
echo "Cloning jpegli..."
rm -rf "$BUILD_DIR"
git clone --depth 1 https://github.com/google/jpegli.git "$BUILD_DIR"

echo "Building jpegli..."
cd "$BUILD_DIR"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DJPEGXL_ENABLE_BENCHMARK=OFF \
    -DJPEGXL_ENABLE_EXAMPLES=OFF \
    -DJPEGXL_ENABLE_MANPAGES=OFF

cmake --build . --target cjpegli djpegli -- -j"$NPROC"

# Install
echo "Installing to $INSTALL_DIR..."
if [ -w "$INSTALL_DIR" ]; then
    cp tools/cjpegli tools/djpegli "$INSTALL_DIR/"
else
    sudo cp tools/cjpegli tools/djpegli "$INSTALL_DIR/"
fi

# Cleanup
rm -rf "$BUILD_DIR"

echo ""
echo "=== Installation complete ==="
echo "cjpegli: $(which cjpegli)"
echo "djpegli: $(which djpegli)"
cjpegli --help 2>&1 | head -3 || true
