#!/bin/bash
set -euo pipefail

# Build jpegli as a static library and install to vendor/jpegli/
#
# Usage:
#   ./scripts/build_jpegli.sh
#
# Produces:
#   vendor/jpegli/lib/libjpegli-static.a
#   vendor/jpegli/lib/libhwy.a
#   vendor/jpegli/include/jpegli/jpeglib.h (+ other headers)
#
# Prerequisites (macOS):
#   brew install llvm coreutils cmake giflib jpeg-turbo libpng ninja zlib
#
# Prerequisites (Linux):
#   apt-get install cmake build-essential libgif-dev libjpeg-turbo8-dev \
#       libpng-dev zlib1g-dev

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$SCRIPT_DIR/../vendor/jpegli"
BUILD_DIR="/tmp/jpegli-build"

echo "=== Building jpegli static library ==="
echo "Install prefix: $PREFIX"

# Detect platform
OS="$(uname -s)"
case "$OS" in
    Darwin)
        NPROC=$(sysctl -n hw.ncpu)

        # Check for Homebrew dependencies
        for dep in llvm coreutils cmake giflib jpeg-turbo libpng ninja zlib; do
            if ! brew list "$dep" &>/dev/null; then
                echo "Missing dependency: $dep"
                echo "Install with: brew install $dep"
                exit 1
            fi
        done

        # Use Homebrew LLVM (Apple clang is not officially supported)
        LLVM_PREFIX="$(brew --prefix llvm)"
        export CC="${LLVM_PREFIX}/bin/clang"
        export CXX="${LLVM_PREFIX}/bin/clang++"
        export CMAKE_PREFIX_PATH="$(brew --prefix giflib):$(brew --prefix jpeg-turbo):$(brew --prefix libpng):$(brew --prefix zlib)"
        ;;
    Linux)
        NPROC=$(nproc)
        ;;
    *)
        echo "Unsupported platform: $OS"
        exit 1
        ;;
esac

# Clone with submodules (highway is a submodule)
echo "Cloning jpegli..."
rm -rf "$BUILD_DIR"
git clone --depth 1 --recursive --shallow-submodules \
    https://github.com/google/jpegli.git "$BUILD_DIR"

# Build
echo "Building..."
cd "$BUILD_DIR"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBUILD_SHARED_LIBS=OFF \
    -DJPEGXL_STATIC=ON \
    -DBUILD_TESTING=OFF \
    -DJPEGXL_ENABLE_TOOLS=OFF \
    -DJPEGXL_ENABLE_BENCHMARK=OFF \
    -DJPEGXL_ENABLE_DEVTOOLS=OFF \
    -DJPEGXL_ENABLE_FUZZERS=OFF \
    -DJPEGXL_ENABLE_MANPAGES=OFF \
    -DJPEGXL_ENABLE_DOXYGEN=OFF \
    -G Ninja

cmake --build . -- -j"$NPROC"
cmake --install .

# Also copy highway static lib if not installed by cmake
if [ ! -f "$PREFIX/lib/libhwy.a" ]; then
    find . -name "libhwy.a" -exec cp {} "$PREFIX/lib/" \;
fi

# Copy libjpegli-static.a if not installed by cmake
if [ ! -f "$PREFIX/lib/libjpegli-static.a" ]; then
    echo "libjpegli-static.a not installed by cmake, copying manually..."
    find . -name "libjpegli-static.a" -exec cp {} "$PREFIX/lib/" \;
fi

# Copy jpegli headers if not installed by cmake
if [ ! -f "$PREFIX/include/jpegli/encode.h" ]; then
    echo "jpegli headers not installed by cmake, copying manually..."
    mkdir -p "$PREFIX/include/jpegli"
    find "$BUILD_DIR" -path "*/jpegli/*.h" -not -path "*/CMakeFiles/*" | while read h; do
        cp "$h" "$PREFIX/include/jpegli/"
    done
fi

# Cleanup
rm -rf "$BUILD_DIR"

echo ""
echo "=== jpegli build complete ==="
echo "Static lib: $PREFIX/lib/libjpegli-static.a"
echo "Highway:    $PREFIX/lib/libhwy.a"
echo "Headers:    $PREFIX/include/"
echo ""
echo "To build origami with jpegli:"
echo "  cargo build --features jpegli"
