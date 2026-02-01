#!/usr/bin/env bash

# RRIP Tile Server Launch Script
# Usage: ./run-server.sh [options]

set -e

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

# Default configuration
PORT="${PORT:-3007}"
SLIDES_ROOT="${SLIDES_ROOT:-../data}"
RUST_LOG="${RUST_LOG:-info}"
BUILD_MODE="${BUILD_MODE:-release}"  # Can be 'debug' or 'release'

# Performance settings
RAYON_THREADS="${RAYON_THREADS:-8}"
TOKIO_WORKERS="${TOKIO_WORKERS:-4}"
TOKIO_BLOCKING_THREADS="${TOKIO_BLOCKING_THREADS:-24}"
MAX_INFLIGHT_FAMILIES="${MAX_INFLIGHT_FAMILIES:-16}"
CACHE_ENTRIES="${CACHE_ENTRIES:-4096}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --slides-root)
            SLIDES_ROOT="$2"
            shift 2
            ;;
        --debug)
            BUILD_MODE="debug"
            shift
            ;;
        --release)
            BUILD_MODE="release"
            shift
            ;;
        --timing)
            TIMING_FLAG="--timing-breakdown"
            shift
            ;;
        --help)
            echo "RRIP Tile Server"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT          Port to listen on (default: 3007)"
            echo "  --slides-root PATH   Path to slides data (default: ../data)"
            echo "  --debug              Build in debug mode"
            echo "  --release            Build in release mode (default)"
            echo "  --timing             Enable timing breakdown logging"
            echo "  --help               Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  PORT                 Port to listen on"
            echo "  SLIDES_ROOT          Path to slides data"
            echo "  RUST_LOG             Log level (error, warn, info, debug, trace)"
            echo "  BUILD_MODE           Build mode (debug or release)"
            echo "  RAYON_THREADS        Number of Rayon threads (default: 8)"
            echo "  TOKIO_WORKERS        Number of Tokio workers (default: 4)"
            echo "  CACHE_ENTRIES        Number of cache entries (default: 4096)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "🚀 Starting RRIP Tile Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Slides root: $SLIDES_ROOT"
echo "🌐 Port: $PORT"
echo "🔧 Build mode: $BUILD_MODE"
echo "📊 Log level: $RUST_LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Build flags
if [ "$BUILD_MODE" = "release" ]; then
    BUILD_FLAGS="--release"
    PROFILE_DEBUG="CARGO_PROFILE_RELEASE_DEBUG=true"
else
    BUILD_FLAGS=""
    PROFILE_DEBUG=""
fi

# Export environment variables
export TURBOJPEG_SOURCE=explicit
export TURBOJPEG_DYNAMIC=1
export TURBOJPEG_LIB_DIR
export TURBOJPEG_INCLUDE_DIR
export RUST_LOG
if [ -n "$PROFILE_DEBUG" ]; then
    export $PROFILE_DEBUG
fi

# Check if TurboJPEG is installed
if [ ! -d "$TURBOJPEG_LIB_DIR" ]; then
    echo "⚠️  Warning: TurboJPEG library directory not found at $TURBOJPEG_LIB_DIR"
    echo "   Please install libjpeg-turbo:"
    echo "   macOS: brew install jpeg-turbo"
    echo "   Linux: apt-get install libturbojpeg0-dev"
    exit 1
fi

# Run the server
exec cargo run $BUILD_FLAGS -- \
    --slides-root "$SLIDES_ROOT" \
    --port "$PORT" \
    --rayon-threads "$RAYON_THREADS" \
    --tokio-workers "$TOKIO_WORKERS" \
    --tokio-blocking-threads "$TOKIO_BLOCKING_THREADS" \
    --max-inflight-families "$MAX_INFLIGHT_FAMILIES" \
    --cache-entries "$CACHE_ENTRIES" \
    $TIMING_FLAG