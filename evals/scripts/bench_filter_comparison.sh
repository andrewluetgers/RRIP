#!/usr/bin/env bash
#
# Benchmark tile server performance across upsample filters (bilinear, bicubic, lanczos3).
#
# Two test scenarios per filter:
#
#   1. WORST CASE: Each request hits a different L2 family (max family gen pressure)
#      No --cache-dir (RocksDB disabled). Measures raw family generation throughput.
#
#   2. REALISTIC: Bursts of tiles from the SAME family arrive concurrently.
#      Tests whether singleflight / caching prevents redundant family decodes.
#      20 tiles per family (4 L1 + 16 L0), fired simultaneously.
#
# Usage:
#   pnpm bench
#   bash evals/scripts/bench_filter_comparison.sh [options]
#
# Options:
#   --slide SLIDE_ID        Slide to test (default: origami_90_80_70)
#   --port PORT             Server port (default: 3007)
#   --slides-root DIR       Slides directory (default: data/dzi)
#   --families N            Number of L2 families to test (default: 50)
#   --concurrency C1,C2,..  Concurrency levels for worst-case (default: 1,4,8,16,32)
#
set -uo pipefail

SLIDE_ID="origami_90_80_70"
PORT=3007
SLIDES_ROOT="data/dzi"
NUM_FAMILIES=50
CONCURRENCY_LEVELS="1,4,8,16,32,64,128"
RESULTS_DIR="evals/runs/filter_bench_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slide)       SLIDE_ID="$2"; shift 2;;
    --port)        PORT="$2"; shift 2;;
    --slides-root) SLIDES_ROOT="$2"; shift 2;;
    --families)    NUM_FAMILIES="$2"; shift 2;;
    --concurrency) CONCURRENCY_LEVELS="$2"; shift 2;;
    *)             echo "Unknown arg: $1"; exit 1;;
  esac
done

# Find origami binary
ORIGAMI=""
for candidate in server/target2/release/origami server/target/release/origami; do
  if [[ -x "$candidate" ]]; then
    ORIGAMI="$candidate"
    break
  fi
done
if [[ -z "$ORIGAMI" ]]; then
  echo "ERROR: origami binary not found. Run: cd server && cargo build --release"
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
echo "L0=$L0  L1=$L1"
echo "Results:      $RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Generate URL files
# ---------------------------------------------------------------------------

# Collect all L2 parent coords from pack files
ALL_PARENTS="$RESULTS_DIR/all_parents.txt"
for pack in "$PACK_DIR"/*.pack; do
  basename "$pack" .pack
done | sort -R > "$ALL_PARENTS"

TOTAL_PARENTS=$(wc -l < "$ALL_PARENTS" | tr -d ' ')
echo "Total L2 families available: $TOTAL_PARENTS"

# Limit to NUM_FAMILIES
PARENTS_FILE="$RESULTS_DIR/parents.txt"
head -n "$NUM_FAMILIES" "$ALL_PARENTS" > "$PARENTS_FILE"
ACTUAL_FAMILIES=$(wc -l < "$PARENTS_FILE" | tr -d ' ')
echo "Using $ACTUAL_FAMILIES families"

# --- Worst case: one random L0 tile per family ---
WORST_URLS="$RESULTS_DIR/worst_case_urls.txt"
> "$WORST_URLS"
while IFS=_ read -r x2 y2; do
  dx=$(( RANDOM % 4 ))
  dy=$(( RANDOM % 4 ))
  x0=$(( x2 * 4 + dx ))
  y0=$(( y2 * 4 + dy ))
  echo "http://localhost:${PORT}/tiles/${SLIDE_ID}/${L0}/${x0}_${y0}.jpg"
done < "$PARENTS_FILE" >> "$WORST_URLS"

# --- Realistic: ALL 20 tiles per family (4 L1 + 16 L0), grouped ---
# Each family's 20 URLs go on consecutive lines so xargs -P fires them together
REALISTIC_URLS="$RESULTS_DIR/realistic_urls.txt"
> "$REALISTIC_URLS"
while IFS=_ read -r x2 y2; do
  # 4 L1 tiles
  for dx in 0 1; do
    for dy in 0 1; do
      x1=$(( x2 * 2 + dx ))
      y1=$(( y2 * 2 + dy ))
      echo "http://localhost:${PORT}/tiles/${SLIDE_ID}/${L1}/${x1}_${y1}.jpg"
    done
  done
  # 16 L0 tiles
  for dx in 0 1 2 3; do
    for dy in 0 1 2 3; do
      x0=$(( x2 * 4 + dx ))
      y0=$(( y2 * 4 + dy ))
      echo "http://localhost:${PORT}/tiles/${SLIDE_ID}/${L0}/${x0}_${y0}.jpg"
    done
  done
done < "$PARENTS_FILE" >> "$REALISTIC_URLS"

REALISTIC_COUNT=$(wc -l < "$REALISTIC_URLS" | tr -d ' ')
echo "Worst-case URLs: $ACTUAL_FAMILIES (one per family)"
echo "Realistic URLs:  $REALISTIC_COUNT (20 per family x $ACTUAL_FAMILIES families)"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FILTERS=("bilinear" "bicubic" "lanczos3")

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

# Run concurrent curl requests, output: "http_code time_total_secs size url"
run_curl_bench() {
  local url_file="$1"
  local conc="$2"
  local out_file="$3"
  cat "$url_file" | xargs -P "$conc" -I{} \
    curl -s -o /dev/null -w '%{http_code} %{time_total} %{size_download} %{url_effective}\n' '{}' \
    > "$out_file" 2>&1
}

# Print summary stats from curl output file
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
total_s = sum(times) / 1000
p95 = times[int(n * 0.95)] if n > 1 else times[0]
p99 = times[int(n * 0.99)] if n > 1 else times[0]
rps = n / total_s if total_s > 0 else 0
print(f"    {sys.argv[2]}: {ok}/{n} ok | {rps:.1f} req/s | "
      f"med={statistics.median(times):.0f}ms mean={statistics.mean(times):.0f}ms "
      f"p95={p95:.0f}ms p99={p99:.0f}ms max={times[-1]:.0f}ms")
PYEOF
}

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

IFS=',' read -ra CONC_ARRAY <<< "$CONCURRENCY_LEVELS"

for filter in "${FILTERS[@]}"; do
  echo "================================================================"
  echo "  FILTER: $filter"
  echo "================================================================"

  # --- WORST CASE: no cache ---
  echo ""
  echo "  [WORST CASE] no cache, one tile per family"
  RUST_LOG=warn "$ORIGAMI" serve \
    --slides-root "$SLIDES_ROOT" \
    --port "$PORT" \
    --upsample-filter "$filter" \
    --timing-breakdown \
    --metrics-interval-secs 5 \
    > "$RESULTS_DIR/server_worst_${filter}.log" 2>&1 &
  SERVER_PID=$!

  for i in $(seq 1 30); do
    curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  ERROR: Server exited"; tail -10 "$RESULTS_DIR/server_worst_${filter}.log"; exit 1
    fi
    sleep 1
  done

  # Warmup
  head -3 "$WORST_URLS" | xargs -I{} curl -s -o /dev/null '{}' 2>/dev/null
  sleep 0.5

  RSS_START=$(get_rss_mb "$SERVER_PID")
  for conc in "${CONC_ARRAY[@]}"; do
    OUT="$RESULTS_DIR/worst_${filter}_c${conc}.txt"
    WALL_START=$(now_ms)
    run_curl_bench "$WORST_URLS" "$conc" "$OUT"
    WALL_END=$(now_ms)
    WALL_MS=$(( WALL_END - WALL_START ))
    echo "    c=$conc  wall=${WALL_MS}ms"
    summarize "$OUT" "c=$conc"
  done
  RSS_END=$(get_rss_mb "$SERVER_PID")
  echo "  RSS: ${RSS_START}MB -> ${RSS_END}MB"

  # Save server metrics
  grep "metrics " "$RESULTS_DIR/server_worst_${filter}.log" | tail -1 \
    > "$RESULTS_DIR/metrics_worst_${filter}.txt" 2>/dev/null || true

  kill_server "$SERVER_PID"
  sleep 1

  # --- REALISTIC: cache enabled, 20 tiles per family fired concurrently ---
  echo ""
  CACHE_DIR="$RESULTS_DIR/rocksdb_${filter}"
  rm -rf "$CACHE_DIR"
  echo "  [REALISTIC] RocksDB cache enabled, 20 tiles/family burst"
  RUST_LOG=info "$ORIGAMI" serve \
    --slides-root "$SLIDES_ROOT" \
    --port "$PORT" \
    --upsample-filter "$filter" \
    --timing-breakdown \
    --metrics-interval-secs 5 \
    --cache-dir "$CACHE_DIR" \
    > "$RESULTS_DIR/server_real_${filter}.log" 2>&1 &
  SERVER_PID=$!

  for i in $(seq 1 30); do
    curl -sf "http://localhost:${PORT}/healthz" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  ERROR: Server exited"; tail -10 "$RESULTS_DIR/server_real_${filter}.log"; exit 1
    fi
    sleep 1
  done

  # Realistic test: fire all 20 tiles per family with concurrency=20
  # This simulates a viewer loading a viewport (many tiles from nearby families)
  for conc in 20 40 80; do
    OUT="$RESULTS_DIR/real_${filter}_c${conc}.txt"
    WALL_START=$(now_ms)
    run_curl_bench "$REALISTIC_URLS" "$conc" "$OUT"
    WALL_END=$(now_ms)
    WALL_MS=$(( WALL_END - WALL_START ))
    echo "    c=$conc  wall=${WALL_MS}ms"
    summarize "$OUT" "c=$conc"
  done

  # Count how many families were actually generated vs served from cache
  FAMILIES_GEN=$(grep -c "family_breakdown" "$RESULTS_DIR/server_real_${filter}.log" 2>/dev/null || true)
  FAMILIES_GEN=${FAMILIES_GEN:-0}
  CACHE_HITS=$(grep -c "cache_hit" "$RESULTS_DIR/server_real_${filter}.log" 2>/dev/null || true)
  CACHE_HITS=${CACHE_HITS:-0}
  GENERATED=$(grep -c "tile generated" "$RESULTS_DIR/server_real_${filter}.log" 2>/dev/null || true)
  GENERATED=${GENERATED:-0}
  echo "  Families generated: $FAMILIES_GEN (expected ~$ACTUAL_FAMILIES)"
  echo "  Tiles from cache: $CACHE_HITS  |  Tiles generated: $GENERATED"

  grep "metrics " "$RESULTS_DIR/server_real_${filter}.log" | tail -1 \
    > "$RESULTS_DIR/metrics_real_${filter}.txt" 2>/dev/null || true

  kill_server "$SERVER_PID"
  sleep 1
  echo ""
done

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

echo "================================================================"
echo "  COMPARISON SUMMARY"
echo "================================================================"

python3 - "$RESULTS_DIR" "$CONCURRENCY_LEVELS" "$ACTUAL_FAMILIES" "${FILTERS[@]}" <<'PYEOF'
import sys, os, statistics

results_dir = sys.argv[1]
conc_levels = sys.argv[2].split(",")
num_families = int(sys.argv[3])
filters = sys.argv[4:]

def parse_curl(path):
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
    if not times: return None
    times.sort()
    n = len(times)
    return {
        "n": n, "ok": ok,
        "total_s": sum(times) / 1000,
        "rps": n / (sum(times) / 1000) if sum(times) > 0 else 0,
        "med": statistics.median(times),
        "mean": statistics.mean(times),
        "p95": times[int(n * 0.95)] if n > 1 else times[0],
        "p99": times[int(n * 0.99)] if n > 1 else times[0],
        "max": times[-1],
    }

# WORST CASE table
print("\n--- WORST CASE (no cache, 1 tile per family) ---\n")
col = 16
hdr = f"{'':>{col}}"
for f in filters: hdr += f"  {f:>16}"
print(hdr)
print("-" * len(hdr))

for c in conc_levels:
    print(f"\n  c={c}")
    for metric, unit in [("rps","req/s"),("med","ms"),("p95","ms"),("p99","ms"),("max","ms")]:
        row = f"    {metric+' '+unit:>{col}}"
        for f in filters:
            path = os.path.join(results_dir, f"worst_{f}_c{c}.txt")
            d = parse_curl(path) if os.path.exists(path) else None
            val = f"{d[metric]:.0f}" if d else "—"
            row += f"  {val:>16}"
        print(row)

# REALISTIC table
print("\n\n--- REALISTIC (cache on, 20 tiles/family burst) ---\n")
hdr = f"{'':>{col}}"
for f in filters: hdr += f"  {f:>16}"
print(hdr)
print("-" * len(hdr))

for c in ["20", "40", "80"]:
    print(f"\n  c={c}")
    for metric, unit in [("rps","req/s"),("med","ms"),("p95","ms"),("max","ms")]:
        row = f"    {metric+' '+unit:>{col}}"
        for f in filters:
            path = os.path.join(results_dir, f"real_{f}_c{c}.txt")
            d = parse_curl(path) if os.path.exists(path) else None
            val = f"{d[metric]:.0f}" if d else "—"
            row += f"  {val:>16}"
        print(row)

# Duplicate family gen
print("\n\n--- SINGLEFLIGHT EFFICIENCY (realistic test) ---\n")
row_hdr = f"{'':>{col}}"
for f in filters: row_hdr += f"  {f:>16}"
print(row_hdr)
print("-" * len(row_hdr))
row = f"  {'families gen':>{col}}"
for f in filters:
    logpath = os.path.join(results_dir, f"server_real_{f}.log")
    count = 0
    if os.path.exists(logpath):
        for line in open(logpath):
            if "family_breakdown" in line: count += 1
    row += f"  {count:>16}"
print(row)
row = f"  {'expected':>{col}}"
for f in filters: row += f"  {num_families:>16}"
print(row)
row = f"  {'duplicates':>{col}}"
for f in filters:
    logpath = os.path.join(results_dir, f"server_real_{f}.log")
    count = 0
    if os.path.exists(logpath):
        for line in open(logpath):
            if "family_breakdown" in line: count += 1
    dup = count - num_families
    row += f"  {dup:>16}"
print(row)

print(f"\nResults: {results_dir}/")
PYEOF
