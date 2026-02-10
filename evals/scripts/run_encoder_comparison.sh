#!/bin/bash

# Run all encoder comparison captures: 4 encoders x 7 qualities x 2 modes = 56 runs
#
# Encoders: libjpeg-turbo, mozjpeg, jpegli, jpegxl
# Modes: JPEG baseline recompression + ORIGAMI residual encoding
#
# Prerequisites:
#   - vendor/mozjpeg/bin/cjpeg   (./scripts/build_mozjpeg.sh)
#   - cjpegli                     (./scripts/build_jpegli.sh)
#   - cjxl / djxl                 (brew install jpeg-xl)

set -euo pipefail

IMAGE="evals/test-images/L0-1024.jpg"
QUALITIES=(30 40 50 60 70 80 90)
ENCODERS=(libjpeg-turbo mozjpeg jpegli jpegxl)

# --- Preflight checks ---

if [ ! -f "$IMAGE" ]; then
    echo "Error: Test image not found: $IMAGE"
    exit 1
fi

SKIP_ENCODERS=()

if [ ! -f "vendor/mozjpeg/bin/cjpeg" ]; then
    echo "Warning: mozjpeg not found at vendor/mozjpeg/bin/cjpeg — skipping mozjpeg runs"
    echo "  Build with: ./scripts/build_mozjpeg.sh"
    SKIP_ENCODERS+=(mozjpeg)
fi

if ! command -v cjpegli &>/dev/null; then
    echo "Warning: cjpegli not found — skipping jpegli runs"
    echo "  Build with: ./scripts/build_jpegli.sh"
    SKIP_ENCODERS+=(jpegli)
fi

if ! command -v cjxl &>/dev/null || ! command -v djxl &>/dev/null; then
    echo "Warning: cjxl/djxl not found — skipping jpegxl runs"
    echo "  Install with: brew install jpeg-xl"
    SKIP_ENCODERS+=(jpegxl)
fi

should_skip() {
    local enc="$1"
    for skip in "${SKIP_ENCODERS[@]+"${SKIP_ENCODERS[@]}"}"; do
        if [ "$enc" = "$skip" ]; then
            return 0
        fi
    done
    return 1
}

# --- Count total runs ---
total=0
for ENC in "${ENCODERS[@]}"; do
    if should_skip "$ENC"; then continue; fi
    for Q in "${QUALITIES[@]}"; do
        total=$((total + 2))  # baseline + origami
    done
done

echo ""
echo "=== Multi-Encoder Comparison ==="
echo "Encoders: ${ENCODERS[*]}"
echo "Qualities: ${QUALITIES[*]}"
echo "Total runs: $total"
echo ""

count=0

for ENC in "${ENCODERS[@]}"; do
    if should_skip "$ENC"; then
        echo "--- Skipping $ENC (not available) ---"
        echo ""
        continue
    fi

    echo "--- $ENC ---"

    for Q in "${QUALITIES[@]}"; do
        # JPEG baseline
        count=$((count + 1))
        echo "[$count/$total] $ENC baseline Q$Q..."
        uv run python evals/scripts/jpeg_baseline.py \
            --image "$IMAGE" --quality "$Q" --encoder "$ENC"
        echo "  done"

        # ORIGAMI residual
        count=$((count + 1))
        echo "[$count/$total] $ENC ORIGAMI J$Q..."
        uv run python evals/scripts/wsi_residual_debug_with_manifest.py \
            --image "$IMAGE" --resq "$Q" --pac --encoder "$ENC"
        echo "  done"
    done

    echo ""
done

echo "=== All encoder comparison runs complete ==="
echo "View results: cd evals/viewer && pnpm start  (http://localhost:8084)"
