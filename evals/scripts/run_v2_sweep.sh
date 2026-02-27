#!/bin/bash
# Run v2 fused pipeline sweep: 4 variants across (baseq, l0q) combinations.
# All use: 444 subsamp, optL2 ±20, lanczos3 upsample.
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

# Variants: v2_b{baseq}_l0q{l0q}_optl2_d20
declare -a VARIANTS=(
  "90 70"
  "90 80"
  "80 70"
  "80 80"
)

for variant in "${VARIANTS[@]}"; do
  read -r baseq l0q <<< "$variant"
  name="v2_b${baseq}_l0q${l0q}_optl2_d20"
  outdir="${RUNS}/${name}"

  echo "=========================================="
  echo "  Encoding: ${name}"
  echo "=========================================="

  rm -rf "$outdir"
  $ORIGAMI encode \
    --image "$IMAGE" \
    --out "$outdir" \
    --baseq "$baseq" \
    --l0q "$l0q" \
    --subsamp 444 \
    --optl2 \
    --max-delta 20 \
    --upsample-filter lanczos3 \
    --downsample-filter lanczos3 \
    --manifest \
    --debug-images \
    --pack

  echo ""
done

echo "=========================================="
echo "  All v2 encodes complete!"
echo "=========================================="
echo ""
echo "Run dirs:"
ls -d ${RUNS}/v2_* 2>/dev/null || echo "(none found)"
echo ""
echo "Computing metrics..."
for variant in "${VARIANTS[@]}"; do
  read -r baseq l0q <<< "$variant"
  name="v2_b${baseq}_l0q${l0q}_optl2_d20"
  echo "  → ${name}"
  uv run python evals/scripts/compute_metrics.py "evals/runs/${name}" 2>&1 || echo "  (metrics failed)"
done

echo ""
echo "Done! Start viewer with: cd evals/viewer && pnpm start"
