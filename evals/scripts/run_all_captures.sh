#!/bin/bash

# Run all JPEG quality levels for ORIGAMI debug captures

IMAGE="evals/test-images/L0-1024.jpg"

# JPEG qualities
JPEG_QUALITIES=(10 20 30 40 50 60 70 80 90)

echo "Starting ORIGAMI debug captures..."
echo "Total runs: ${#JPEG_QUALITIES[@]}"
echo ""

count=0
total=${#JPEG_QUALITIES[@]}

for J in "${JPEG_QUALITIES[@]}"; do
    count=$((count + 1))
    echo "[$count/$total] Running J$J..."

    python evals/scripts/wsi_residual_debug_with_manifest.py \
        --image "$IMAGE" \
        --resq $J \
        --pac \
        --out "evals/runs/debug_j${J}_pac"

    if [ $? -eq 0 ]; then
        echo "✓ J$J completed"
    else
        echo "✗ J$J failed"
    fi
    echo ""
done

echo "All captures complete!"
echo "View results at http://localhost:8099"
