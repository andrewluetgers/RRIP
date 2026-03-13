#!/bin/bash
# V4 split luma/chroma seed sweep — quality-focused.
#
# Thesis: Luma needs high quality (human vision is sharp for luminance),
# chroma can be crushed aggressively (human vision is blurry for color).
# The win is reallocating bytes from chroma → luma/residual.
#
# Key insight: a LARGE chroma seed at very low Q may be nearly free in bytes
# but give better color predictions than a tiny one. So chroma sizes go up to 512.
# Meanwhile a smaller luma seed may be fine if the residual carries the detail.
#
# Fixed params: 444 subsamp, optL2 ±20, lanczos3, jpegxl encoder.
#
# Sweep axes:
#   - Luma seed size:     128, 256, 384, 512
#   - Luma seed quality:  60, 70, 80, 90, 95
#   - Chroma seed size:   64, 128, 256, 512
#   - Chroma seed quality: 20, 30, 40, 50
#   - L0 residual quality: 60, 70, 80, 90
#
# No --debug-images to save disk. Manifests have Y-PSNR per tile.
set -euo pipefail

cd "$(dirname "$0")/../.."  # cd to RRIP root

ORIGAMI="server/target2/release/origami"
IMAGE="evals/test-images/L0-1024.jpg"
RUNS="evals/runs"

# Ensure binary exists
if [ ! -f "$ORIGAMI" ]; then
  echo "Binary not found at $ORIGAMI — building..."
  (cd server && CMAKE_POLICY_VERSION_MINIMUM=3.5 CARGO_TARGET_DIR=target2 cargo build --release)
fi

# ─── Phase 0: V2 baselines for comparison ───
echo "══════════════════════════════════════════"
echo "  Phase 0: V2 baselines (Q60-90)"
echo "══════════════════════════════════════════"

for L0Q in 60 70 80 90; do
  name="v2_b90_l0q${L0Q}_optl2_d20"
  outdir="${RUNS}/${name}"
  if [ -d "$outdir" ]; then
    echo "  [skip] ${name} (exists)"
  else
    echo "  → ${name}"
    $ORIGAMI encode \
      --image "$IMAGE" --out "$outdir" \
      --baseq 90 --l0q "$L0Q" \
      --subsamp 444 --optl2 --max-delta 20 \
      --upsample-filter lanczos3 --downsample-filter lanczos3 \
      --manifest --pack 2>&1 | tail -1
  fi
done

# ─── Phase 1: V4 split-seed sweep ───
echo ""
echo "══════════════════════════════════════════"
echo "  Phase 1: V4 split-seed sweep"
echo "══════════════════════════════════════════"

COUNT=0
TOTAL=0

# Count total runs
for LS in 128 256 384 512; do
  for LQ in 60 70 80 90 95; do
    for CS in 64 128 256 512; do
      for CQ in 20 30 40 50; do
        for L0Q in 60 70 80 90; do
          TOTAL=$((TOTAL + 1))
        done
      done
    done
  done
done

echo "  Total V4 runs to evaluate: ${TOTAL}"
echo ""

for LS in 128 256 384 512; do
  for LQ in 60 70 80 90 95; do
    for CS in 64 128 256 512; do
      for CQ in 20 30 40 50; do
        for L0Q in 60 70 80 90; do
          COUNT=$((COUNT + 1))
          name="v4_sl${LQ}x${LS}_sc${CQ}x${CS}_l0q${L0Q}_optl2"
          outdir="${RUNS}/${name}"

          if [ -d "$outdir" ]; then
            echo "  [${COUNT}/${TOTAL}] [skip] ${name} (exists)"
            continue
          fi

          echo "  [${COUNT}/${TOTAL}] → ${name}"
          $ORIGAMI encode \
            --image "$IMAGE" --out "$outdir" \
            --seed-luma-size "$LS" --seed-luma-q "$LQ" \
            --seed-chroma-size "$CS" --seed-chroma-q "$CQ" \
            --l0q "$L0Q" \
            --subsamp 444 --optl2 --max-delta 20 \
            --upsample-filter lanczos3 --downsample-filter lanczos3 \
            --manifest --pack 2>&1 | tail -1
        done
      done
    done
  done
done

# ─── Phase 2: Summary ───
echo ""
echo "══════════════════════════════════════════"
echo "  All encodes done!"
echo "══════════════════════════════════════════"
echo ""
echo "V4 runs: $(ls -d ${RUNS}/v4_* 2>/dev/null | wc -l)"
echo ""
echo "Analyze with: python3 evals/scripts/analyze_v4_sweep.py"
echo "View in browser: cd evals/viewer && pnpm start"
