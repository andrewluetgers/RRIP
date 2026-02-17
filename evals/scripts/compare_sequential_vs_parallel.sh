#!/bin/bash
# Compare sequential vs concurrent approaches for WSI encoding
set -e

echo "========================================================="
echo "WSI Encoding: Throughput Optimization Test"
echo "========================================================="
echo ""
echo "This script runs three tests:"
echo "  1. Sequential: One slide at a time, CPU pyramids overlap"
echo "  2. Concurrent-2: Max 2 slides at once (based on ~50% GPU/slide)"
echo "  3. Parallel-5: All 5 slides simultaneously (stress test)"
echo ""
echo "Goal: Find optimal concurrency for maximum throughput"
echo ""

# Test 1: Sequential with background pyramids
echo "TEST 1: Sequential encoding (concurrency=1)"
echo "============================================"
seq_start=$(date +%s)
bash /workspace/RRIP/evals/scripts/run_5_concurrent_wsi.sh 2>&1 | tee /tmp/sequential_test.log
seq_end=$(date +%s)
seq_elapsed=$((seq_end - seq_start))

echo ""
echo "Sequential test complete: ${seq_elapsed}s"
echo ""
echo "Press Enter to continue to concurrent-2 test (or Ctrl+C to stop)..."
read

# Test 2: Concurrent pairs (optimal for ~50% GPU usage per slide)
echo ""
echo "TEST 2: Concurrent-2 encoding (concurrency=2)"
echo "=============================================="
conc2_start=$(date +%s)
bash /workspace/RRIP/evals/scripts/run_5_slides_2concurrent.sh 2>&1 | tee /tmp/concurrent2_test.log
conc2_end=$(date +%s)
conc2_elapsed=$((conc2_end - conc2_start))

echo ""
echo "Concurrent-2 test complete: ${conc2_elapsed}s"
echo ""
echo "Press Enter to continue to parallel-5 test (or Ctrl+C to stop)..."
read

# Test 3: Full parallel (stress test)
echo ""
echo "TEST 3: Parallel-5 encoding (concurrency=5)"
echo "============================================"
par_start=$(date +%s)
bash /workspace/RRIP/evals/scripts/run_5_parallel_wsi.sh 2>&1 | tee /tmp/parallel5_test.log
par_end=$(date +%s)
par_elapsed=$((par_end - par_start))

echo ""
echo "========================================"
echo "FINAL COMPARISON"
echo "========================================"
echo "Sequential (concurrency=1):  ${seq_elapsed}s"
echo "Concurrent-2 (concurrency=2): ${conc2_elapsed}s"
echo "Parallel-5 (concurrency=5):   ${par_elapsed}s"
echo ""

# Find winner
min_time=$seq_elapsed
winner="Sequential"

if [ $conc2_elapsed -lt $min_time ]; then
    min_time=$conc2_elapsed
    winner="Concurrent-2"
fi

if [ $par_elapsed -lt $min_time ]; then
    min_time=$par_elapsed
    winner="Parallel-5"
fi

echo "Winner: $winner (${min_time}s)"
echo ""

if [ "$winner" = "Sequential" ]; then
    echo "Interpretation: GPU is already saturated with 1 slide, or"
    echo "                pyramid CPU work is the bottleneck."
elif [ "$winner" = "Concurrent-2" ]; then
    echo "Interpretation: GPU has ~50% unused capacity with 1 slide."
    echo "                Concurrency=2 achieves optimal GPU utilization."
else
    echo "Interpretation: GPU can handle >2 parallel streams efficiently."
    echo "                Higher concurrency better utilizes GPU resources."
fi

echo ""
echo "Logs saved to:"
echo "  Sequential:   /tmp/sequential_test.log"
echo "  Concurrent-2: /tmp/concurrent2_test.log"
echo "  Parallel-5:   /tmp/parallel5_test.log"
