#!/bin/bash
# Simple load test for flamegraph - focuses on L1 tiles that trigger family generation

PORT=${1:-3007}

# Just request 5 L1 tiles to trigger family generation
echo "Requesting L1 tiles to trigger family generation..."

for i in {1..5}; do
    curl -s "http://localhost:$PORT/tiles/demo_out/15/10${i}_10${i}.jpg" > /dev/null &
done

wait
echo "Done"