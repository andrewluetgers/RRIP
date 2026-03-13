#!/usr/bin/env bash
#
# Benchmark tile server: Lanczos3 (baseline) vs SR Model decode performance.
#
# Tests cold-miss family reconstruction throughput at various concurrency levels.
# No cache (RocksDB disabled) to measure raw decode performance.
#
# Usage:
#   bash evals/scripts/bench_sr_model.sh [options]
#
# Options:
#   --slide SLIDE_ID        Slide to test (default: v2_b90_l0q80)
#   --port PORT             Server port (default: 3007)
#   --slides-root DIR       Slides directory (default: data/dzi)
#   --families N            Number of L2 families to test (default: 50)
#   --concurrency C1,C2,..  Concurrency levels (default: 1,4,8,16,32)
#   --sr-model PATH         SR ONNX model path (default: models/model_sr.onnx)
#
set -uo pipefail

SLIDE_ID="v2_b90_l0q80"
PORT=3007
SLIDES_ROOT="data/dzi"
NUM_FAMILIES=50
CONCURRENCY_LEVELS="1,4,8,16,32"
SR_MODEL="models/model_sr.onnx"
RESULTS_DIR="evals/runs/sr_bench_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slide)       SLIDE_ID="$2"; shift 2;;
    --port)        PORT="$2"; shift 2;;
    --slides-root) SLIDES_ROOT="$2"; shift 2;;
    --families)    NUM_FAMILIES="$2"; shift 2;;
    --concurrency) CONCURRENCY_LEVELS="$2"; shift 2;;
    --sr-model)    SR_MODEL="$2"; shift 2;;
    *)             echo "Unknown arg: $1"; exit 1;;
  esac
done

# Find origami binary (must be built with --features sr-model)
ORIGAMI=""
for candidate in server/target2/release/origami server/target/release/origami; do
  if [[ -x "$candidate" ]]; then
    ORIGAMI="$candidate"
    break
  fi
done
if [[ -z "$ORIGAMI" ]]; then
  echo "ERROR: origami binary not found. Run: cd server && cargo build --release --features sr-model"
  exit 1
fi

if [[ ! -f "$SR_MODEL" ]]; then
  echo "ERROR: SR model not found at $SR_MODEL"
  exit 1
fi

# Kill any existing server on the port
lsof -i :"$PORT" -t 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

# Cleanup on exit
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 0.5
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PACK_DIR="$SLIDES_ROOT/$SLIDE_ID/residual_packs"
if [[ ! -d "$PACK_DIR" ]]; then
  echo "ERROR: No residual_packs at $PACK_DIR"
  exit 1
fi

# Find L0 level from pyramid
FILES_DIR="$SLIDES_ROOT/$SLIDE_ID/baseline_pyramid_files"
L0=$(ls "$FILES_DIR" | sort -n | tail -1)
L1=$((L0 - 1))

mkdir -p "$RESULTS_DIR"

echo "Binary:       $ORIGAMI"
echo "Slide:        $SLIDE_ID"
echo "SR Model:     $SR_MODEL"
echo "L0=$L0  L1=$L1"
echo "Results:      $RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Generate URL files
# ---------------------------------------------------------------------------

# Collect L2 parent coords from pack files, randomized
ALL_PARENTS="$RESULTS_DIR/all_parents.txt"
for pack in "$PACK_DIR"/*.pack; do
  basename "$pack" .pack
done | sort -R > "$ALL_PARENTS"

TOTAL_PARENTS=$(wc -l < "$ALL_PARENTS" | tr -d ' ')
echo "Total L2 families available: $TOTAL_PARENTS"

PARENTS_FILE="$RESULTS_DIR/parents.txt"
head -n "$NUM_FAMILIES" "$ALL_PARENTS" > "$PARENTS_FILE"
ACTUAL_FAMILIES=$(wc -l < "$PARENTS_FILE" | tr -d ' ')
echo "Using $ACTUAL_FAMILIES families"

# One L0 tile per family (worst case — forces family reconstruction)
WORST_URLS="$RESULTS_DIR/worst_case_urls.txt"
> "$WORST_URLS"
while IFS=_ read -r x2 y2; do
  dx=$(( RANDOM % 4 ))
  dy=$(( RANDOM % 4 ))
  x0=$(( x2 * 4 + dx ))
  y0=$(( y2 * 4 + dy ))
  echo "http://localhost:${PORT}/tiles/${SLIDE_ID}/${L0}/${x0}_${y0}.jpg"
done < "$PARENTS_FILE" >> "$WORST_URLS"

echo "Worst-case URLs: $ACTUAL_FAMILIES (one L0 tile per distinct family)"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

kill_server() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  fi
}

get_rss_mb() {
  ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f", $1/1024}'
}

now_ms() {
  python3 -c "import time; print(int(time.time()*1000))"
}

run_curl_bench() {
  local url_file="$1"
  local conc="$2"
  local out_file="$3"
  cat "$url_file" | xargs -P "$conc" -I{} \
    curl -s -o /dev/null -w '%{http_code} %{time_total} %{size_download} %{url_effective}\n' '{}' \
    > "$out_file" 2>&1
}

summarize() {
  local file="$1"
  local label="$2"
  python3 - "$file" "$label" <<'PYEOF'
import sys, statistics
times, sizes, ok = [], [], 0
for line in open(sys.argv[1]):
    parts = line.strip().split()
    if len(parts) < 3: continue
    try:
        code, t, sz = int(parts[0]), float(parts[1]) * 1000, int(parts[2])
    except: continue
    times.append(t)
    sizes.append(sz)
    if code == 200: ok += 1
if not times:
    print(f"    {sys.argv[2]}: no results"); sys.exit(0)
times.sort()
n = len(times)
wall_s = max(times) / 1000  # wall clock is the longest request
p50 = times[n // 2]
p95 = times[int(n * 0.95)] if n > 1 else times[0]
p99 = times[int(n * 0.99)] if n > 1 else times[0]
families_sec = n / wall_s if wall_s > 0 else 0
print(f"    {sys.argv[2]:>6s}: {ok}/{n} ok | {families_sec:6.1f} families/s | "
      f"p50={p50:6.0f}ms  p95={p95:6.0f}ms  p99={p99:6.0f}ms  max={times[-1]:6.0f}ms")
PYEOF
}

# ---------------------------------------------------------------------------
# Benchmark: Lanczos3 (baseline)
# ---------------------------------------------------------------------------

IFS=',' read -ra CONC_ARRAY <<< "$CONCURRENCY_LEVELS"

echo "================================================================"
echo "  LANCZOS3 (baseline, no cache)"
echo "================================================================"

RUST_LOG=warn "$ORIGAMI" serve \
  --slides-root "$SLIDES_ROOT" \
  --port "$PORT" \
  --upsample-filter lanczos3 \
  --metrics-interval-secs 5 \
  > "$RESULTS_DIR/server_lanczos3.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 30); do
  curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1 && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  ERROR: Server exited"; tail -10 "$RESULTS_DIR/server_lanczos3.log"; exit 1
  fi
  sleep 1
done

# Warmup (3 families)
head -3 "$WORST_URLS" | xargs -I{} curl -s -o /dev/null '{}' 2>/dev/null
sleep 0.5

RSS_START=$(get_rss_mb "$SERVER_PID")
for conc in "${CONC_ARRAY[@]}"; do
  OUT="$RESULTS_DIR/lanczos3_c${conc}.txt"
  run_curl_bench "$WORST_URLS" "$conc" "$OUT"
  summarize "$OUT" "c=$conc"
done
RSS_END=$(get_rss_mb "$SERVER_PID")
echo "  RSS: ${RSS_START}MB -> ${RSS_END}MB"

# Grab server metrics
curl -s "http://localhost:${PORT}/metrics" > "$RESULTS_DIR/metrics_lanczos3.txt" 2>/dev/null || true

kill_server "$SERVER_PID"
sleep 1

# ---------------------------------------------------------------------------
# Benchmark: SR Model (no cache)
# ---------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  SR MODEL (no cache)"
echo "================================================================"

RUST_LOG=warn "$ORIGAMI" serve \
  --slides-root "$SLIDES_ROOT" \
  --port "$PORT" \
  --upsample-filter lanczos3 \
  --sr-model "$SR_MODEL" \
  --metrics-interval-secs 5 \
  > "$RESULTS_DIR/server_sr.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 30); do
  curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1 && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  ERROR: Server exited"; tail -10 "$RESULTS_DIR/server_sr.log"; exit 1
  fi
  sleep 1
done

# Warmup
head -3 "$WORST_URLS" | xargs -I{} curl -s -o /dev/null '{}' 2>/dev/null
sleep 0.5

RSS_START=$(get_rss_mb "$SERVER_PID")
for conc in "${CONC_ARRAY[@]}"; do
  OUT="$RESULTS_DIR/sr_c${conc}.txt"
  run_curl_bench "$WORST_URLS" "$conc" "$OUT"
  summarize "$OUT" "c=$conc"
done
RSS_END=$(get_rss_mb "$SERVER_PID")
echo "  RSS: ${RSS_START}MB -> ${RSS_END}MB"

curl -s "http://localhost:${PORT}/metrics" > "$RESULTS_DIR/metrics_sr.txt" 2>/dev/null || true

kill_server "$SERVER_PID"
sleep 1

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""

# Generate comparison table
python3 - "$RESULTS_DIR" "$CONCURRENCY_LEVELS" <<'PYEOF'
import sys, os, statistics

results_dir = sys.argv[1]
conc_levels = sys.argv[2].split(',')

def parse_results(path):
    times = []
    ok = 0
    for line in open(path):
        parts = line.strip().split()
        if len(parts) < 3: continue
        try:
            code, t = int(parts[0]), float(parts[1]) * 1000
        except: continue
        times.append(t)
        if code == 200: ok += 1
    if not times:
        return None
    times.sort()
    n = len(times)
    return {
        'n': n, 'ok': ok,
        'p50': times[n // 2],
        'p95': times[int(n * 0.95)] if n > 1 else times[0],
        'max': times[-1],
        'wall': max(times) / 1000,
        'fam_sec': n / (max(times) / 1000) if max(times) > 0 else 0,
    }

print(f"{'Conc':>6s}  {'─── Lanczos3 ───':>32s}  {'──── SR Model ────':>32s}  {'Δ fam/s':>8s}")
print(f"{'':>6s}  {'fam/s':>8s} {'p50ms':>7s} {'p95ms':>7s} {'maxms':>7s}  {'fam/s':>8s} {'p50ms':>7s} {'p95ms':>7s} {'maxms':>7s}  {'':>8s}")
print("─" * 90)

for c in conc_levels:
    l3_file = os.path.join(results_dir, f'lanczos3_c{c}.txt')
    sr_file = os.path.join(results_dir, f'sr_c{c}.txt')

    l3 = parse_results(l3_file) if os.path.exists(l3_file) else None
    sr = parse_results(sr_file) if os.path.exists(sr_file) else None

    if l3 and sr:
        delta = sr['fam_sec'] - l3['fam_sec']
        print(f"  c={c:>2s}  {l3['fam_sec']:8.1f} {l3['p50']:7.0f} {l3['p95']:7.0f} {l3['max']:7.0f}"
              f"  {sr['fam_sec']:8.1f} {sr['p50']:7.0f} {sr['p95']:7.0f} {sr['max']:7.0f}"
              f"  {delta:+8.1f}")
    elif l3:
        print(f"  c={c:>2s}  {l3['fam_sec']:8.1f} {l3['p50']:7.0f} {l3['p95']:7.0f} {l3['max']:7.0f}"
              f"  {'(no data)':>32s}  {'N/A':>8s}")

PYEOF
