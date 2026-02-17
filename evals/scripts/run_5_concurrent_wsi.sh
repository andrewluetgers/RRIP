#!/bin/bash
# Run 5 sequential WSI encodes with pyramid generation in background
# This tests GPU+CPU overlap: GPU does families, CPU does pyramid concurrently

set -e

SLIDE="/workspace/data/3DHISTECH-2-256/4_1"
OUT_BASE="/workspace/RRIP/evals/runs"
ENCODER="/workspace/RRIP/gpu-encode/target/release/origami-gpu-encode"

# Clean old runs
rm -rf "${OUT_BASE}/sequential_pyramid_"*

echo "=== Testing GPU+CPU Overlap: 5 Sequential WSI Encodes ==="
echo "Each encode: GPU does families, then spawns CPU pyramid in background"
echo "Expectation: Pyramid runs on CPU while GPU moves to next slide"
echo ""

total_start=$(date +%s)

for i in {1..5}; do
    out_dir="${OUT_BASE}/sequential_pyramid_${i}"

    echo "--- Slide $i/5 ---"
    time "$ENCODER" encode \
        --slide "$SLIDE" \
        --out "$out_dir" \
        --baseq 80 --l1q 70 --l0q 60 --subsamp 444 \
        --generate-pyramid --pyramid-sharpen 0.25 \
        --max-parents 200 \
        2>&1 | grep -E '(INFO|Spawning|Waiting|complete|elapsed|throughput)'

    echo ""
done

total_end=$(date +%s)
total_elapsed=$((total_end - total_start))

echo "=== Summary ==="
echo "Total time for 5 slides: ${total_elapsed}s"
echo "Average per slide: $((total_elapsed / 5))s"
