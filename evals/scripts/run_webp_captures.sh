#!/bin/bash

# Run WebP vs JPEG comparison evals:
#   1. JPEG baselines (libjpeg-turbo) — for comparison
#   2. WebP baselines — direct WebP compression
#   3. ORIGAMI with JPEG residuals (libjpeg-turbo) — for comparison
#   4. ORIGAMI with WebP residuals
#
# All at quality steps: 30, 40, 50, 60, 90

set -euo pipefail

IMAGE="evals/test-images/L0-1024.jpg"
QUALITIES=(30 40 50 60 90)

echo "=== WebP vs JPEG Comparison Evals ==="
echo "Qualities: ${QUALITIES[*]}"
echo ""

# --- 1. JPEG Baselines (libjpeg-turbo) ---
echo "--- [1/4] JPEG Baselines (libjpeg-turbo) ---"
count=0
total=${#QUALITIES[@]}

for Q in "${QUALITIES[@]}"; do
    count=$((count + 1))
    OUT="evals/runs/jpeg_baseline_q${Q}"
    if [ -d "$OUT" ] && [ -f "$OUT/manifest.json" ]; then
        echo "[$count/$total] JPEG baseline Q$Q — already exists, skipping"
        continue
    fi
    echo "[$count/$total] JPEG baseline Q$Q..."

    uv run python evals/scripts/jpeg_baseline.py \
        --image "$IMAGE" --quality $Q \
        --output "$OUT"

    if [ $? -eq 0 ]; then
        echo "  done"
    else
        echo "  FAILED"
    fi
done

echo ""

# --- 2. WebP Baselines ---
echo "--- [2/4] WebP Baselines ---"
count=0

for Q in "${QUALITIES[@]}"; do
    count=$((count + 1))
    OUT="evals/runs/webp_jpeg_baseline_q${Q}"
    if [ -d "$OUT" ] && [ -f "$OUT/manifest.json" ]; then
        echo "[$count/$total] WebP baseline Q$Q — already exists, skipping"
        continue
    fi
    echo "[$count/$total] WebP baseline Q$Q..."

    uv run python evals/scripts/jpeg_baseline.py \
        --image "$IMAGE" --quality $Q --encoder webp \
        --output "$OUT"

    if [ $? -eq 0 ]; then
        echo "  done"
    else
        echo "  FAILED"
    fi
done

echo ""

# --- 3. ORIGAMI with JPEG residuals (libjpeg-turbo) ---
echo "--- [3/4] ORIGAMI with JPEG residuals ---"
count=0

for J in "${QUALITIES[@]}"; do
    count=$((count + 1))
    OUT="evals/runs/debug_j${J}_pac"
    if [ -d "$OUT" ] && [ -f "$OUT/manifest.json" ]; then
        echo "[$count/$total] ORIGAMI J$J (jpeg) — already exists, skipping"
        continue
    fi
    echo "[$count/$total] ORIGAMI J$J (jpeg)..."

    uv run python evals/scripts/wsi_residual_debug_with_manifest.py \
        --image "$IMAGE" --resq $J --pac \
        --out "$OUT"

    if [ $? -eq 0 ]; then
        echo "  done"
    else
        echo "  FAILED"
    fi
done

echo ""

# --- 4. ORIGAMI with WebP residuals ---
echo "--- [4/4] ORIGAMI with WebP residuals ---"
count=0

for J in "${QUALITIES[@]}"; do
    count=$((count + 1))
    OUT="evals/runs/webp_debug_j${J}_pac"
    if [ -d "$OUT" ] && [ -f "$OUT/manifest.json" ]; then
        echo "[$count/$total] ORIGAMI J$J (webp) — already exists, skipping"
        continue
    fi
    echo "[$count/$total] ORIGAMI J$J (webp)..."

    uv run python evals/scripts/wsi_residual_debug_with_manifest.py \
        --image "$IMAGE" --resq $J --pac --encoder webp \
        --out "$OUT"

    if [ $? -eq 0 ]; then
        echo "  done"
    else
        echo "  FAILED"
    fi
done

echo ""
echo "=== All captures complete ==="
echo "View results at http://localhost:8084"
