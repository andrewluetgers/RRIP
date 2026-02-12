#!/bin/bash

# Sweep a grid of quality parameters for single-image ORIGAMI and YCbCr subsample modes.
#
# Usage:
#   bash evals/scripts/run_single_image_sweep.sh [encoder] [image]
#
# Examples:
#   bash evals/scripts/run_single_image_sweep.sh
#   bash evals/scripts/run_single_image_sweep.sh jpegli
#   bash evals/scripts/run_single_image_sweep.sh libjpeg-turbo evals/test-images/some_image.jpg

ENCODER="${1:-libjpeg-turbo}"
IMAGE="${2:-evals/test-images/L0-1024.jpg}"

PRIOR_QUALITIES=(50 60 70 80 90)
RESIDUAL_QUALITIES=(30 40 50 60 70 80)

LUMA_QUALITIES=(50 60 70 80 90)
CHROMA_QUALITIES=(30 40 50 60 70 80)

origami_total=$(( ${#PRIOR_QUALITIES[@]} * ${#RESIDUAL_QUALITIES[@]} ))
ycbcr_total=$(( ${#LUMA_QUALITIES[@]} * ${#CHROMA_QUALITIES[@]} ))
grand_total=$(( origami_total + ycbcr_total ))

echo "Single-Image Compression Sweep"
echo "==============================="
echo "Encoder: $ENCODER"
echo "Image: $IMAGE"
echo "ORIGAMI runs: $origami_total (${#PRIOR_QUALITIES[@]} prior x ${#RESIDUAL_QUALITIES[@]} residual)"
echo "YCbCr runs: $ycbcr_total (${#LUMA_QUALITIES[@]} luma x ${#CHROMA_QUALITIES[@]} chroma)"
echo "Total runs: $grand_total"
echo ""

count=0

# --- ORIGAMI mode ---
echo "=== ORIGAMI Mode ==="
for PQ in "${PRIOR_QUALITIES[@]}"; do
    for RQ in "${RESIDUAL_QUALITIES[@]}"; do
        count=$((count + 1))
        echo "[$count/$grand_total] ORIGAMI p${PQ} r${RQ}..."

        uv run python evals/scripts/single_image_compress.py \
            --image "$IMAGE" \
            --mode origami \
            --prior-quality $PQ \
            --residual-quality $RQ \
            --encoder "$ENCODER"

        if [ $? -eq 0 ]; then
            echo "  Done"
        else
            echo "  FAILED"
        fi
    done
done

echo ""

# --- YCbCr subsample mode ---
echo "=== YCbCr Subsample Mode ==="
for LQ in "${LUMA_QUALITIES[@]}"; do
    for CQ in "${CHROMA_QUALITIES[@]}"; do
        count=$((count + 1))
        echo "[$count/$grand_total] YCbCr y${LQ} c${CQ}..."

        uv run python evals/scripts/single_image_compress.py \
            --image "$IMAGE" \
            --mode ycbcr-subsample \
            --luma-quality $LQ \
            --chroma-quality $CQ \
            --encoder "$ENCODER"

        if [ $? -eq 0 ]; then
            echo "  Done"
        else
            echo "  FAILED"
        fi
    done
done

echo ""
echo "All $grand_total runs complete!"
echo "View results: cd evals/viewer && node viewer-server.js"
