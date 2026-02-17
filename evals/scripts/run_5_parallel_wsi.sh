#!/bin/bash
# Run 5 WSI encodes in PARALLEL to test GPU saturation
# Each process gets its own GPU stream and competes for GPU resources

set -e

SLIDE="/workspace/data/3DHISTECH-2-256/4_1"
OUT_BASE="/workspace/RRIP/evals/runs"
ENCODER="/workspace/RRIP/gpu-encode/target/release/origami-gpu-encode"

# Clean old runs
rm -rf "${OUT_BASE}/parallel_pyramid_"*

echo "=== Testing GPU Saturation: 5 PARALLEL WSI Encodes ==="
echo "Each process runs simultaneously, sharing GPU resources"
echo "Expectation: GPU utilization stays high, processes compete for GPU"
echo ""

total_start=$(date +%s)

# Launch all 5 processes in parallel (backgrounded)
for i in {1..5}; do
    out_dir="${OUT_BASE}/parallel_pyramid_${i}"

    echo "--- Launching slide $i/5 in background ---"
    (
        time "$ENCODER" encode \
            --slide "$SLIDE" \
            --out "$out_dir" \
            --baseq 80 --l1q 70 --l0q 60 --subsamp 444 \
            --generate-pyramid --pyramid-sharpen 0.25 \
            --max-parents 200 \
            2>&1 | grep -E '(INFO|Spawning|Waiting|complete|elapsed|throughput)'
    ) &
done

# Wait for all background processes to complete
echo ""
echo "Waiting for all 5 parallel encodes to complete..."
wait

total_end=$(date +%s)
total_elapsed=$((total_end - total_start))

echo ""
echo "=== Summary ==="
echo "Total wall-clock time for 5 parallel slides: ${total_elapsed}s"
echo "Average per slide (if perfectly parallel): $((total_elapsed))s"
echo ""
echo "Compare to sequential (with background pyramids): expected ~325s (5 × 65s)"
echo "If GPU is bottleneck: time ≈ 5 × (GPU time per slide)"
echo "If CPU is bottleneck: time ≈ max(5 × GPU time, pyramid time)"
