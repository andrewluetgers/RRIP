#!/usr/bin/env bash
# Generate optimized-L2 ORIGAMI runs for comparison viewer
# Uses the same pipeline as wsi_residual_debug_with_manifest.py but with
# L2 pixels optimized at encode time to minimize prediction error.

set -e
cd "$(dirname "$0")/../.."

IMAGE="evals/test-images/L0-1024.jpg"

# Flat quality runs with optimized L2 (q=10 to q=90, step 5)
for Q in 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90; do
    DIR="evals/runs/optl2_debug_j${Q}_pac"
    if [ -d "$DIR" ]; then
        echo "=== Skipping OptL2 flat q=$Q (already exists) ==="
        continue
    fi
    echo "=== OptL2 flat q=$Q ==="
    uv run python evals/scripts/optl2_compress.py \
        --image "$IMAGE" --resq "$Q" --pac --max-delta 15
done

# +20 split quality runs with optimized L2
for L0Q in 10 20 30 40 50 60 70 80; do
    L1Q=$((L0Q + 20))
    if [ $L1Q -gt 90 ]; then
        L1Q=90
    fi
    DIR="evals/runs/optl2_debug_l1q${L1Q}_l0q${L0Q}_pac"
    if [ -d "$DIR" ]; then
        echo "=== Skipping OptL2 split L1=$L1Q L0=$L0Q (already exists) ==="
        continue
    fi
    echo "=== OptL2 split L1=$L1Q L0=$L0Q ==="
    uv run python evals/scripts/optl2_compress.py \
        --image "$IMAGE" --l1q "$L1Q" --l0q "$L0Q" --pac --max-delta 15
done

echo "Done! All OptL2 runs generated."
