#!/bin/bash
# Compare sequential (GPU+CPU overlap) vs parallel (GPU saturation) approaches
set -e

echo "=========================================="
echo "WSI Encoding: Sequential vs Parallel Test"
echo "=========================================="
echo ""
echo "This script runs two tests:"
echo "  1. Sequential: One slide at a time on GPU, CPU pyramids overlap"
echo "  2. Parallel: All 5 slides run simultaneously, competing for GPU"
echo ""
echo "Goal: Determine which approach maximizes throughput"
echo ""

# Test 1: Sequential with background pyramids
echo "TEST 1: Sequential encoding (GPU+CPU overlap)"
echo "=============================================="
seq_start=$(date +%s)
bash /workspace/RRIP/evals/scripts/run_5_concurrent_wsi.sh 2>&1 | tee /tmp/sequential_test.log
seq_end=$(date +%s)
seq_elapsed=$((seq_end - seq_start))

echo ""
echo "Sequential test complete: ${seq_elapsed}s"
echo ""
echo "Press Enter to continue to parallel test (or Ctrl+C to stop)..."
read

# Test 2: Parallel encoding
echo ""
echo "TEST 2: Parallel encoding (GPU saturation)"
echo "==========================================="
par_start=$(date +%s)
bash /workspace/RRIP/evals/scripts/run_5_parallel_wsi.sh 2>&1 | tee /tmp/parallel_test.log
par_end=$(date +%s)
par_elapsed=$((par_end - par_start))

echo ""
echo "========================================"
echo "FINAL COMPARISON"
echo "========================================"
echo "Sequential (GPU+CPU overlap): ${seq_elapsed}s"
echo "Parallel (GPU saturation):    ${par_elapsed}s"
echo ""

if [ $seq_elapsed -lt $par_elapsed ]; then
    speedup=$(echo "scale=1; $par_elapsed * 100 / $seq_elapsed - 100" | bc)
    echo "Winner: Sequential is ${speedup}% faster"
    echo "Interpretation: GPU can't handle 5 parallel streams efficiently,"
    echo "                or pyramid CPU work is the bottleneck."
elif [ $par_elapsed -lt $seq_elapsed ]; then
    speedup=$(echo "scale=1; $seq_elapsed * 100 / $par_elapsed - 100" | bc)
    echo "Winner: Parallel is ${speedup}% faster"
    echo "Interpretation: GPU has unused capacity when running one slide,"
    echo "                parallel streams better utilize GPU resources."
else
    echo "Tie: Both approaches take the same time"
fi

echo ""
echo "Logs saved to:"
echo "  Sequential: /tmp/sequential_test.log"
echo "  Parallel:   /tmp/parallel_test.log"
