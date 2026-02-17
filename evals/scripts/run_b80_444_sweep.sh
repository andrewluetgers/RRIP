#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

BIN=server/target2/release/origami
IMG=evals/test-images/L0-1024.jpg

# Sweep: baseq 80, 444, L1Q/L0Q split with 20-point gap
# Two variants: plain (no optl2) and optl2 with delta 15
SPLITS=(
    "30:10"
    "40:20"
    "50:30"
    "60:40"
    "70:50"
    "80:60"
    "90:70"
)

# ============================================================================
# SECTION 1: baseq 80, 444, no optl2
# ============================================================================
echo "=== B80 444 (no opt) — quality sweep ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/rs_444_b80_l1q${L1Q}_l0q${L0Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  444 b80 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --baseq 80 --subsamp 444 --l1q "$L1Q" --l0q "$L0Q" \
        --manifest --debug-images 2>&1 | tail -1
done

# ============================================================================
# SECTION 2: baseq 80, 444, optl2 delta 15
# ============================================================================
echo ""
echo "=== B80 444 + OptL2 d15 — quality sweep ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/rs_444_b80_optl2_d15_l1q${L1Q}_l0q${L0Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  444 b80 optl2 d15 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --baseq 80 --subsamp 444 --optl2 --max-delta 15 \
        --l1q "$L1Q" --l0q "$L0Q" \
        --manifest --debug-images 2>&1 | tail -1
done

echo ""
echo "=== ALL ENCODING DONE ==="
echo ""

# ============================================================================
# SECTION 3: Compute metrics for all new runs
# ============================================================================
echo "=== Computing metrics ==="
RUNS=""
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    RUNS="$RUNS evals/runs/rs_444_b80_l1q${L1Q}_l0q${L0Q}"
    RUNS="$RUNS evals/runs/rs_444_b80_optl2_d15_l1q${L1Q}_l0q${L0Q}"
done
uv run python evals/scripts/compute_metrics.py $RUNS

echo ""
echo "=== SWEEP COMPLETE ==="
