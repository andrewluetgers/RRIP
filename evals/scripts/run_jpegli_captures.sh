#!/bin/bash

# Run all jpegli eval captures: JPEG baselines + ORIGAMI residuals
# Requires: cjpegli installed (see scripts/install_jpegli.sh)

set -euo pipefail

IMAGE="evals/test-images/L0-1024.jpg"

# Check cjpegli is available
if ! command -v cjpegli &>/dev/null; then
    echo "Error: cjpegli not found. Install with: ./scripts/install_jpegli.sh"
    exit 1
fi

echo "=== Jpegli Eval Captures ==="
echo ""

# JPEG baselines with jpegli
JPEG_QUALITIES=(30 40 50 60 70 80 90)

echo "--- JPEG Baselines (jpegli) ---"
count=0
total=${#JPEG_QUALITIES[@]}

for Q in "${JPEG_QUALITIES[@]}"; do
    count=$((count + 1))
    echo "[$count/$total] JPEG baseline Q$Q (jpegli)..."

    uv run python evals/scripts/jpeg_baseline.py \
        --image "$IMAGE" --quality $Q --encoder jpegli \
        --output "evals/runs/jpegli_jpeg_baseline_q${Q}"

    if [ $? -eq 0 ]; then
        echo "  done"
    else
        echo "  FAILED"
    fi
done

echo ""
echo "--- ORIGAMI with jpegli residual encoding ---"
count=0

for J in "${JPEG_QUALITIES[@]}"; do
    count=$((count + 1))
    echo "[$count/$total] ORIGAMI J$J (jpegli)..."

    uv run python evals/scripts/wsi_residual_debug_with_manifest.py \
        --image "$IMAGE" --resq $J --pac --encoder jpegli \
        --out "evals/runs/jpegli_debug_j${J}_pac"

    if [ $? -eq 0 ]; then
        echo "  done"
    else
        echo "  FAILED"
    fi
done

echo ""
echo "=== All jpegli captures complete ==="
echo "View results at http://localhost:8099"
