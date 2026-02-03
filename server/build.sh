#!/usr/bin/env bash

# ORIGAMI Tile Server Build Script
# Usage: ./build.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect platform and set TurboJPEG paths
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        TURBOJPEG_LIB_DIR="${TURBOJPEG_LIB_DIR:-$(brew --prefix jpeg-turbo 2>/dev/null || echo /opt/homebrew)/lib}"
        TURBOJPEG_INCLUDE_DIR="${TURBOJPEG_INCLUDE_DIR:-$(brew --prefix jpeg-turbo 2>/dev/null || echo /opt/homebrew)/include}"
    else
        TURBOJPEG_LIB_DIR="${TURBOJPEG_LIB_DIR:-/opt/homebrew/lib}"
        TURBOJPEG_INCLUDE_DIR="${TURBOJPEG_INCLUDE_DIR:-/opt/homebrew/include}"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    TURBOJPEG_LIB_DIR="${TURBOJPEG_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"
    TURBOJPEG_INCLUDE_DIR="${TURBOJPEG_INCLUDE_DIR:-/usr/include}"
fi

# Default settings
BUILD_MODE="release"
RUN_TESTS=false
RUN_CLIPPY=true
CLEAN=false
VERBOSE=false
TARGET=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_MODE="debug"
            shift
            ;;
        --release)
            BUILD_MODE="release"
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --no-clippy)
            RUN_CLIPPY=false
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --docker)
            BUILD_DOCKER=true
            shift
            ;;
        --help)
            echo "ORIGAMI Tile Server Build Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug          Build in debug mode"
            echo "  --release        Build in release mode (default)"
            echo "  --test           Run tests after building"
            echo "  --no-clippy      Skip clippy linting"
            echo "  --clean          Clean before building"
            echo "  --verbose, -v    Verbose output"
            echo "  --target TARGET  Cross-compile for target"
            echo "  --docker         Build Docker image after compilation"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Build release version"
            echo "  $0 --debug --test     # Build debug and run tests"
            echo "  $0 --clean --release  # Clean build in release mode"
            echo "  $0 --docker           # Build release and create Docker image"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Export environment variables
export TURBOJPEG_SOURCE=explicit
export TURBOJPEG_DYNAMIC=1
export TURBOJPEG_LIB_DIR
export TURBOJPEG_INCLUDE_DIR

# Set Rust flags for optimization
if [ "$BUILD_MODE" = "release" ]; then
    # Use conservative optimization flags to avoid conflicts
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
    export CARGO_PROFILE_RELEASE_DEBUG=true
    BUILD_FLAGS="--release"
else
    BUILD_FLAGS=""
fi

if [ "$VERBOSE" = true ]; then
    BUILD_FLAGS="$BUILD_FLAGS --verbose"
fi

if [ -n "$TARGET" ]; then
    BUILD_FLAGS="$BUILD_FLAGS --target $TARGET"
fi

# Function to print colored output
print_step() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}► $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Start build process
echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}   ${GREEN}ORIGAMI Tile Server Build${NC}          ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
echo ""

# Check TurboJPEG installation
if [ ! -d "$TURBOJPEG_LIB_DIR" ]; then
    print_error "TurboJPEG library not found at $TURBOJPEG_LIB_DIR"
    echo "Please install libjpeg-turbo:"
    echo "  macOS: brew install jpeg-turbo"
    echo "  Linux: apt-get install libturbojpeg0-dev"
    exit 1
fi

print_success "TurboJPEG found at $TURBOJPEG_LIB_DIR"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    print_step "Cleaning build artifacts"
    cargo clean
    print_success "Clean complete"
    echo ""
fi

# Run clippy if enabled
if [ "$RUN_CLIPPY" = true ]; then
    print_step "Running clippy linter"
    if cargo clippy -- -D warnings 2>/dev/null; then
        print_success "Clippy passed"
    else
        print_warning "Clippy found some warnings"
    fi
    echo ""
fi

# Build the project
print_step "Building in $BUILD_MODE mode"
echo "Build flags: $BUILD_FLAGS"
echo "Rust flags: $RUSTFLAGS"
echo ""

if cargo build $BUILD_FLAGS; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi
echo ""

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    print_step "Running tests"
    if cargo test $BUILD_FLAGS; then
        print_success "All tests passed"
    else
        print_error "Some tests failed"
        exit 1
    fi
    echo ""
fi

# Build Docker image if requested
if [ "$BUILD_DOCKER" = true ]; then
    print_step "Building Docker image"
    docker build -t origami-tile-server:latest .
    print_success "Docker image built: origami-tile-server:latest"
    echo ""
fi

# Print binary location
if [ "$BUILD_MODE" = "release" ]; then
    BINARY_PATH="target/release/origami-tile-server"
else
    BINARY_PATH="target/debug/origami-tile-server"
fi

if [ -n "$TARGET" ]; then
    BINARY_PATH="target/$TARGET/$BUILD_MODE/origami-tile-server"
fi

# Get binary size
if [ -f "$BINARY_PATH" ]; then
    BINARY_SIZE=$(ls -lh "$BINARY_PATH" | awk '{print $5}')
    echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}   ${GREEN}Build Summary${NC}                   ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Mode:   ${GREEN}$BUILD_MODE${NC}"
    echo -e "  Binary: ${GREEN}$BINARY_PATH${NC}"
    echo -e "  Size:   ${GREEN}$BINARY_SIZE${NC}"
    echo ""
    print_success "Build complete! Run with: ./run-server.sh"
else
    print_warning "Binary not found at expected location: $BINARY_PATH"
fi