#!/bin/bash

# Performance test for Moka cache optimization
echo "🚀 ORIGAMI Server Performance Test"
echo "================================"
echo ""
echo "This test compares cache performance with the new lock-free Moka cache."
echo ""

# Build with optimization
echo "Building optimized server..."
export TURBOJPEG_SOURCE=explicit
export TURBOJPEG_DYNAMIC=1
export TURBOJPEG_LIB_DIR=/opt/homebrew/lib
export TURBOJPEG_INCLUDE_DIR=/opt/homebrew/include

cd "$(dirname "$0")"
cargo build --release 2>&1 | tail -5

echo ""
echo "Starting server with performance settings for M5..."
echo ""
echo "Configuration:"
echo "  --rayon-threads 8      (CPU workers)"
echo "  --tokio-workers 4      (Async workers)"
echo "  --tokio-blocking-threads 24"
echo "  --max-inflight-families 16"
echo "  --buffer-pool-size 256"
echo "  --cache-entries 4096   (Moka cache size)"
echo ""

# Run the server
cargo run --release -- \
  --slides-root ../data \
  --port 3007 \
  --rayon-threads 8 \
  --tokio-workers 4 \
  --tokio-blocking-threads 24 \
  --max-inflight-families 16 \
  --buffer-pool-size 256 \
  --cache-entries 4096 \
  --timing-breakdown