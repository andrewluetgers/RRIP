#!/bin/bash
# Script to generate a flamegraph of the RRIP server under load

echo "Setting up environment..."
export TURBOJPEG_SOURCE=explicit
export TURBOJPEG_DYNAMIC=1
export TURBOJPEG_LIB_DIR=/opt/homebrew/lib
export TURBOJPEG_INCLUDE_DIR=/opt/homebrew/include
export RUST_LOG=warn
export CARGO_PROFILE_RELEASE_DEBUG=true

echo "Building release binary with debug symbols..."
cargo build --release

echo "Starting server with flamegraph..."
# Start flamegraph in background
sudo -E cargo flamegraph --root --bin origami-tile-server -- --slides-root ../data --port 3007 &
FLAMEGRAPH_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

# Run load test to generate samples
echo "Running load test to generate profiling data..."
for run in {1..10}; do
    echo "  Load run $run/10"
    # Request L1 tiles to trigger family generation (the expensive path)
    for i in {1..10}; do
        curl -s "http://localhost:3007/tiles/demo_out/15/10${i}_10${i}.jpg" > /dev/null &
    done
    wait
    sleep 1
done

echo "Load test complete, stopping server..."
# Stop the flamegraph recording
sudo kill -INT $FLAMEGRAPH_PID
wait $FLAMEGRAPH_PID 2>/dev/null

echo "Flamegraph should be saved as flamegraph.svg"
echo "Check if we got sufficient samples:"
if [ -f cargo-flamegraph.trace ]; then
    SAMPLES=$(wc -l < cargo-flamegraph.trace)
    echo "  Captured $SAMPLES samples"
fi