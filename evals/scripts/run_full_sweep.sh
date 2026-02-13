#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

BIN=server/target2/release/origami
IMG=evals/test-images/L0-1024.jpg
QUALITIES=(10 20 30 40 50 60 70 80 90)

# ============================================================================
# SECTION 1: JPEG Baselines (external pipeline — Python/Pillow for comparison)
# ============================================================================
echo "=== JPEG Baselines (Pillow) ==="
for Q in "${QUALITIES[@]}"; do
    DIR="evals/runs/jpeg_baseline_q${Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  JPEG baseline Q${Q}..."
    uv run python evals/scripts/jpeg_baseline.py \
        --image "$IMG" --quality "$Q" \
        --out "$DIR" 2>&1 | tail -1
done

# ============================================================================
# SECTION 2: Origami 444 (plain, no optimizations) — quality sweep
# ============================================================================
echo ""
echo "=== Origami 444 (no opt) ==="
for Q in "${QUALITIES[@]}"; do
    DIR="evals/runs/rs_444_j${Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  444 j${Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --resq "$Q" --subsamp 444 --l2resq 0 \
        --manifest --debug-images 2>&1 | tail -1
done

# ============================================================================
# SECTION 3: Origami 444 + OptL2 — quality sweep
# ============================================================================
echo ""
echo "=== Origami 444 + OptL2 ==="
for Q in "${QUALITIES[@]}"; do
    DIR="evals/runs/rs_444_optl2_j${Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  444+optL2 j${Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --resq "$Q" --subsamp 444 --optl2 --l2resq 0 \
        --manifest --debug-images 2>&1 | tail -1
done

# ============================================================================
# SECTION 4: Origami 420 + OptL2 — quality sweep
# ============================================================================
echo ""
echo "=== Origami 420 + OptL2 ==="
for Q in "${QUALITIES[@]}"; do
    DIR="evals/runs/rs_420_optl2_j${Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  420+optL2 j${Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --resq "$Q" --subsamp 420 --optl2 --l2resq 0 \
        --manifest --debug-images 2>&1 | tail -1
done

# ============================================================================
# SECTION 5: Origami 420opt + OptL2 — quality sweep (both optimizations combined)
# ============================================================================
echo ""
echo "=== Origami 420opt + OptL2 ==="
for Q in "${QUALITIES[@]}"; do
    DIR="evals/runs/rs_420opt_optl2_j${Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  420opt+optL2 j${Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --resq "$Q" --subsamp 420opt --optl2 --l2resq 0 \
        --manifest --debug-images 2>&1 | tail -1
done

# ============================================================================
# SECTION 6: Split quality variants (l1q higher than l0q)
#   Key split points that showed promise in earlier experiments
# ============================================================================
SPLITS=(
    "40:20"
    "50:30"
    "60:40"
    "70:50"
    "80:60"
    "90:70"
)

echo ""
echo "=== Split quality: 444 (no opt) ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/rs_444_l1q${L1Q}_l0q${L0Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  444 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --l1q "$L1Q" --l0q "$L0Q" --subsamp 444 --l2resq 0 \
        --manifest --debug-images 2>&1 | tail -1
done

echo ""
echo "=== Split quality: 444 + OptL2 ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/rs_444_optl2_l1q${L1Q}_l0q${L0Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  444+optL2 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --l1q "$L1Q" --l0q "$L0Q" --subsamp 444 --optl2 --l2resq 0 \
        --manifest --debug-images 2>&1 | tail -1
done

echo ""
echo "=== Split quality: 420 + OptL2 ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/rs_420_optl2_l1q${L1Q}_l0q${L0Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  420+optL2 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --l1q "$L1Q" --l0q "$L0Q" --subsamp 420 --optl2 --l2resq 0 \
        --manifest --debug-images 2>&1 | tail -1
done

echo ""
echo "=== Split quality: 420opt + OptL2 ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/rs_420opt_optl2_l1q${L1Q}_l0q${L0Q}"
    [ -d "$DIR" ] && rm -rf "$DIR"
    echo "  420opt+optL2 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --l1q "$L1Q" --l0q "$L0Q" --subsamp 420opt --optl2 --l2resq 0 \
        --manifest --debug-images 2>&1 | tail -1
done

echo ""
echo "=== ALL ENCODING DONE ==="
echo ""

# ============================================================================
# SECTION 7: Compute metrics for all new runs
# ============================================================================
echo "=== Computing metrics ==="
RUNS=$(find evals/runs -maxdepth 1 -mindepth 1 -type d \
    ! -name '_archive' ! -name '_holistic_comparison' ! -name 'optl2_*' \
    | sort)
echo "Found $(echo "$RUNS" | wc -l | tr -d ' ') runs to process"
uv run python evals/scripts/compute_metrics.py $RUNS

echo ""
echo "=== FULL SWEEP COMPLETE ==="
