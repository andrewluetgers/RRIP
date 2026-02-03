#!/bin/bash
# Load test script for flamegraph profiling
# This script exercises key paths in the ORIGAMI server

PORT=${1:-3007}
HOST="http://localhost:$PORT"

echo "Starting load test against $HOST..."
echo "This will generate various tile requests to exercise:"
echo "  - L2 baseline tiles (no reconstruction)"
echo "  - L1 tiles (require family generation)"
echo "  - L0 tiles (require full reconstruction)"
echo "  - Cache hits and misses"
echo ""

# Function to make concurrent requests
make_requests() {
    local level=$1
    local start_x=$2
    local start_y=$3
    local count=$4

    echo "Requesting $count tiles at level $level..."

    for ((i=0; i<$count; i++)); do
        x=$((start_x + i))
        y=$((start_y + i))
        curl -s "$HOST/tiles/demo_out/$level/${x}_${y}.jpg" > /dev/null &
    done
    wait
}

# Test 1: L2 baseline tiles (no reconstruction needed)
echo "Test 1: L2 baseline tiles (direct serve)..."
make_requests 14 50 50 20

# Small delay
sleep 1

# Test 2: L1 tiles (triggers family generation with residuals)
echo "Test 2: L1 tiles (family generation)..."
make_requests 15 100 100 10

# Small delay
sleep 1

# Test 3: L0 tiles (full reconstruction path)
echo "Test 3: L0 tiles (full reconstruction)..."
make_requests 16 200 200 10

# Test 4: Mixed workload (concurrent different levels)
echo "Test 4: Mixed workload..."
make_requests 14 60 60 5 &
make_requests 15 120 120 5 &
make_requests 16 240 240 5 &
wait

# Test 5: Cache hits (repeat some requests)
echo "Test 5: Cache hits (repeating requests)..."
make_requests 15 100 100 10
make_requests 16 200 200 10

echo "Load test complete!"