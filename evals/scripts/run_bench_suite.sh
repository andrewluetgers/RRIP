#!/usr/bin/env bash
#
# Run the full benchmark suite: JXL Q80 and Q40 separately, then generate report.
#
# Usage:
#   pnpm bench:all
#   bash evals/scripts/run_bench_suite.sh [--duration N] [--rates R1,R2,...]
#
set -uo pipefail

DURATION="${1:-15}"
RATES="${2:-50,100,200,400}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUITE_DIR="evals/runs/bench_suite_${TIMESTAMP}"
mkdir -p "$SUITE_DIR"

echo "═══════════════════════════════════════════════════"
echo "  ORIGAMI Tile Server Benchmark Suite"
echo "  Duration: ${DURATION}s per rate  Rates: ${RATES}"
echo "  Output: ${SUITE_DIR}"
echo "═══════════════════════════════════════════════════"

for variant in jxl80 jxl40; do
  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "║  Benchmarking: ${variant}"
  echo "╚══════════════════════════════════════════════════╝"
  bash "$SCRIPT_DIR/bench_vegeta.sh" \
    --duration "$DURATION" \
    --rates "$RATES" \
    --slide-filter "$variant" 2>&1 | tee "$SUITE_DIR/${variant}.log"
done

# Copy the latest vegeta results into suite dir
for variant in jxl80 jxl40; do
  latest=$(ls -td evals/runs/bench_vegeta_* 2>/dev/null | head -1)
  if [ -d "$latest" ]; then
    cp "$latest"/*.bin "$SUITE_DIR/" 2>/dev/null || true
    cp "$latest"/*.txt "$SUITE_DIR/" 2>/dev/null || true
  fi
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Generating report..."
echo "═══════════════════════════════════════════════════"

uv run python "$SCRIPT_DIR/bench_report.py" "$SUITE_DIR"

echo ""
echo "Suite complete: $SUITE_DIR"
