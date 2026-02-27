#!/bin/bash
# Residual downscale sweep (bicubic): test L0 residual downscaling at various scales and quality levels.
# Generates runs with naming: bc_rs_444_optl2_d20_l1q{L1Q}_l0q{L0Q}_l0s{SCALE}
set -e
cd "$(dirname "$0")/../.."

BIN=server/target2/release/origami
IMG=evals/test-images/L0-1024.jpg

# Fixed parameters
SUBSAMP=444
DELTA=20
BASEQ=95

# Sweep parameters
L1_QUALITIES=(80 90)
L0_QUALITIES=(60 70 80 90)
L0_SCALES=(100 85 67 50)

echo "=== Residual Downscale Sweep (bicubic) ==="
echo "Image: $IMG"
echo "Base: Q${BASEQ} ${SUBSAMP} optL2 ±${DELTA}"
echo "L1Q: ${L1_QUALITIES[*]}"
echo "L0Q: ${L0_QUALITIES[*]}"
echo "L0 scales: ${L0_SCALES[*]}"
echo ""

TOTAL=$(( ${#L1_QUALITIES[@]} * ${#L0_QUALITIES[@]} * ${#L0_SCALES[@]} ))
COUNT=0

for L1Q in "${L1_QUALITIES[@]}"; do
    for L0Q in "${L0_QUALITIES[@]}"; do
        for SCALE in "${L0_SCALES[@]}"; do
            COUNT=$((COUNT + 1))

            if [ "$SCALE" -eq 100 ]; then
                # No scale suffix for 100%
                DIR="evals/runs/bc_rs_${SUBSAMP}_optl2_d${DELTA}_l1q${L1Q}_l0q${L0Q}"
            else
                DIR="evals/runs/bc_rs_${SUBSAMP}_optl2_d${DELTA}_l1q${L1Q}_l0q${L0Q}_l0s${SCALE}"
            fi

            if [ -d "$DIR" ]; then
                echo "  [$COUNT/$TOTAL] SKIP (exists): $DIR"
                continue
            fi

            echo "  [$COUNT/$TOTAL] L1Q=${L1Q} L0Q=${L0Q} L0scale=${SCALE}% → $DIR"
            $BIN encode --image "$IMG" --out "$DIR" \
                --baseq "$BASEQ" --l1q "$L1Q" --l0q "$L0Q" \
                --subsamp "$SUBSAMP" --optl2 --max-delta "$DELTA" \
                --l0-scale "$SCALE" \
                --manifest --debug-images 2>&1 | tail -1
        done
    done
done

echo ""
echo "=== Sweep complete: $COUNT runs ==="
echo "Start viewer: cd evals/viewer && pnpm start"
