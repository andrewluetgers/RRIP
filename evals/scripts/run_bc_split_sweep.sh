#!/bin/bash
# Bicubic split quality sweep — rerun the core split-quality configurations
# with bicubic upsample. Run names prefixed with bc_ to distinguish from
# the old bilinear runs.
set -e
cd "$(dirname "$0")/../.."

BIN=server/target2/release/origami
IMG=evals/test-images/L0-1024.jpg

SPLITS=(
    "40:20"
    "50:30"
    "60:40"
    "70:50"
    "80:60"
    "90:70"
)

COUNT=0
TOTAL=$(( ${#SPLITS[@]} * 4 ))  # 4 configs per split point

# ============================================================================
# SECTION 1: 444 (no opt)
# ============================================================================
echo "=== Bicubic split quality: 444 (no opt) ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/bc_rs_444_l1q${L1Q}_l0q${L0Q}"
    COUNT=$((COUNT + 1))
    if [ -d "$DIR" ]; then
        echo "  [$COUNT/$TOTAL] SKIP (exists): $DIR"
        continue
    fi
    echo "  [$COUNT/$TOTAL] 444 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --l1q "$L1Q" --l0q "$L0Q" --subsamp 444 \
        --manifest --debug-images 2>&1 | tail -1
done

# ============================================================================
# SECTION 2: 444 + OptL2
# ============================================================================
echo ""
echo "=== Bicubic split quality: 444 + OptL2 ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/bc_rs_444_optl2_l1q${L1Q}_l0q${L0Q}"
    COUNT=$((COUNT + 1))
    if [ -d "$DIR" ]; then
        echo "  [$COUNT/$TOTAL] SKIP (exists): $DIR"
        continue
    fi
    echo "  [$COUNT/$TOTAL] 444+optL2 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --l1q "$L1Q" --l0q "$L0Q" --subsamp 444 --optl2 \
        --manifest --debug-images 2>&1 | tail -1
done

# ============================================================================
# SECTION 3: 420 + OptL2
# ============================================================================
echo ""
echo "=== Bicubic split quality: 420 + OptL2 ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/bc_rs_420_optl2_l1q${L1Q}_l0q${L0Q}"
    COUNT=$((COUNT + 1))
    if [ -d "$DIR" ]; then
        echo "  [$COUNT/$TOTAL] SKIP (exists): $DIR"
        continue
    fi
    echo "  [$COUNT/$TOTAL] 420+optL2 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --l1q "$L1Q" --l0q "$L0Q" --subsamp 420 --optl2 \
        --manifest --debug-images 2>&1 | tail -1
done

# ============================================================================
# SECTION 4: 420opt + OptL2
# ============================================================================
echo ""
echo "=== Bicubic split quality: 420opt + OptL2 ==="
for S in "${SPLITS[@]}"; do
    L1Q="${S%%:*}"
    L0Q="${S##*:}"
    DIR="evals/runs/bc_rs_420opt_optl2_l1q${L1Q}_l0q${L0Q}"
    COUNT=$((COUNT + 1))
    if [ -d "$DIR" ]; then
        echo "  [$COUNT/$TOTAL] SKIP (exists): $DIR"
        continue
    fi
    echo "  [$COUNT/$TOTAL] 420opt+optL2 l1q${L1Q} l0q${L0Q}..."
    $BIN encode --image "$IMG" --out "$DIR" \
        --l1q "$L1Q" --l0q "$L0Q" --subsamp 420opt --optl2 \
        --manifest --debug-images 2>&1 | tail -1
done

echo ""
echo "=== Bicubic split sweep complete: $COUNT runs ==="
echo "Start viewer: cd evals/viewer && pnpm start"
