#!/bin/bash
# Run 5 WSI encodes with max concurrency of 2 (based on ~50% GPU usage per slide)
# This should fully saturate the GPU without oversubscription

set -e

SLIDE="/workspace/data/3DHISTECH-2-256/4_1"
OUT_BASE="/workspace/RRIP/evals/runs"
ENCODER="/workspace/RRIP/gpu-encode/target/release/origami-gpu-encode"

# Clean old runs
rm -rf "${OUT_BASE}/concurrent2_pyramid_"*

echo "=== Testing Optimal GPU Saturation: 5 Slides with Concurrency=2 ==="
echo "Based on ~50% GPU usage per slide, run max 2 slides at a time"
echo "Expectation: 100% GPU utilization, minimal queueing"
echo ""

total_start=$(date +%s)

# Function to run a single encode
run_encode() {
    local i=$1
    local out_dir="${OUT_BASE}/concurrent2_pyramid_${i}"

    echo "--- Starting slide $i/5 ($(date +%H:%M:%S)) ---"
    start=$(date +%s)

    "$ENCODER" encode \
        --slide "$SLIDE" \
        --out "$out_dir" \
        --baseq 80 --l1q 70 --l0q 60 --subsamp 444 \
        --generate-pyramid --pyramid-sharpen 0.25 \
        --max-parents 200 \
        2>&1 | grep -E '(INFO|Spawning|continues in background|complete|elapsed|throughput)'

    end=$(date +%s)
    elapsed=$((end - start))
    echo "--- Slide $i complete: ${elapsed}s ($(date +%H:%M:%S)) ---"
    echo ""
}

# Run slides in pairs (max 2 concurrent)
# Slide 1 & 2 in parallel
run_encode 1 &
run_encode 2 &
wait

# Slide 3 & 4 in parallel
run_encode 3 &
run_encode 4 &
wait

# Slide 5 alone
run_encode 5

total_end=$(date +%s)
total_elapsed=$((total_end - total_start))

echo "=== Summary ==="
echo "Total wall-clock time for 5 slides (concurrency=2): ${total_elapsed}s"
echo ""
echo "Expected timings:"
echo "  - Sequential (concurrency=1): ~85s (5 × 5s families + 60s pyramid overlap)"
echo "  - Pairs (concurrency=2): ~130s (3 pairs × ~65s, if no speedup from GPU overlap)"
echo "  - Pairs (concurrency=2): ~70s (3 pairs, if GPU parallelism cuts time in half)"
echo ""
echo "Actual breakdown:"
echo "  - Pair 1 (slides 1&2): should take max(slide1_time, slide2_time)"
echo "  - Pair 2 (slides 3&4): should take max(slide3_time, slide4_time)"
echo "  - Slide 5: runs alone"
echo ""

# Calculate theoretical best case
if [ $total_elapsed -lt 100 ]; then
    echo "Result: GPU parallelism is WORKING! Both slides in a pair ran simultaneously."
    echo "        GPU was underutilized with 1 slide. Concurrency=2 is optimal."
elif [ $total_elapsed -gt 180 ]; then
    echo "Result: GPU is BOTTLENECKED. Slides in a pair queued, not parallel."
    echo "        One slide already saturates GPU. Concurrency=1 is optimal."
else
    echo "Result: Partial GPU overlap. Some benefit from concurrency=2."
fi
