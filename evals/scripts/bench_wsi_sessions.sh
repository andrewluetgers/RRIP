#!/usr/bin/env bash
#
# Comprehensive tile server load test simulating real viewer sessions.
#
# Simulates multiple concurrent users panning/zooming across different slides.
# Measures latency, throughput, cache behavior, CPU, and memory at various
# concurrency levels.
#
# Test scenarios:
#   1. COLD START: All families are cache misses (measures raw generation speed)
#   2. SESSION SIM: Realistic pan/zoom patterns with locality (tests cache effectiveness)
#   3. MULTI-SLIDE: Concurrent users on different slides (tests memory/contention)
#   4. SUSTAINED: Fixed load over 60s to measure stability and resource growth
#
# Usage:
#   bash evals/scripts/bench_wsi_sessions.sh [options]
#
# Options:
#   --port PORT             Server port (default: 3007)
#   --slides-root DIR       Slides directory (default: ~/dev/data/WSI/slides)
#   --families N            Families per test (default: 100)
#   --concurrency C1,C2,..  Concurrency levels (default: 1,4,8,16,32,64)
#   --duration SECS         Duration for sustained test (default: 60)
#   --slide-filter PATTERN  Only test slides matching pattern (e.g. "jxl40")
#
set -uo pipefail

PORT=3007
SLIDES_ROOT="${SLIDES_ROOT:-$HOME/dev/data/WSI/slides}"
NUM_FAMILIES=100
CONCURRENCY_LEVELS="1,4,8,16,32,64,128"
SUSTAINED_SECS=60
SLIDE_FILTER=""
RESULTS_DIR="evals/runs/bench_sessions_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)         PORT="$2"; shift 2;;
    --slides-root)  SLIDES_ROOT="$2"; shift 2;;
    --families)     NUM_FAMILIES="$2"; shift 2;;
    --concurrency)  CONCURRENCY_LEVELS="$2"; shift 2;;
    --duration)     SUSTAINED_SECS="$2"; shift 2;;
    --slide-filter) SLIDE_FILTER="$2"; shift 2;;
    *)              echo "Unknown arg: $1"; exit 1;;
  esac
done

BASE_URL="http://localhost:${PORT}"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Discover available slides
# ---------------------------------------------------------------------------

echo "=== Discovering slides ==="
SLIDES_JSON=$(curl -sf "${BASE_URL}/slides.json" 2>/dev/null)
if [[ -z "$SLIDES_JSON" ]]; then
  echo "ERROR: Cannot reach tile server at $BASE_URL"
  echo "Start with: origami serve --slides-root $SLIDES_ROOT --port $PORT"
  exit 1
fi

# Parse slide IDs and labels
SLIDE_IDS=$(echo "$SLIDES_JSON" | python3 -c "
import sys, json
slides = json.load(sys.stdin)
for s in slides:
    sid = s.get('id', '')
    label = s.get('label', sid)
    print(f\"{sid}|{label}\")
")

echo "$SLIDE_IDS" | head -20
TOTAL_SLIDES=$(echo "$SLIDE_IDS" | wc -l | tr -d ' ')
echo "Total slides: $TOTAL_SLIDES"

# Apply filter
if [[ -n "$SLIDE_FILTER" ]]; then
  SLIDE_IDS=$(echo "$SLIDE_IDS" | grep -i "$SLIDE_FILTER")
  TOTAL_SLIDES=$(echo "$SLIDE_IDS" | wc -l | tr -d ' ')
  echo "After filter '$SLIDE_FILTER': $TOTAL_SLIDES slides"
fi

echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

get_server_pid() {
  lsof -i :"$PORT" -t 2>/dev/null | head -1
}

get_rss_mb() {
  local pid="$1"
  ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1f", $1/1024}'
}

get_cpu_pct() {
  local pid="$1"
  ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' '
}

get_metrics() {
  curl -sf "${BASE_URL}/metrics" 2>/dev/null || echo "{}"
}

# Run concurrent curl requests, collect timing
# Output per line: "http_code time_total_ms size_bytes url"
run_curl_bench() {
  local url_file="$1"
  local conc="$2"
  cat "$url_file" | xargs -P "$conc" -I{} \
    curl -sf -o /dev/null -w '%{http_code} %{time_total} %{size_download} {}\n' '{}' 2>/dev/null
}

# Compute latency stats from timing output
compute_stats() {
  local input="$1"
  python3 -c "
import sys
times = []
codes = {}
total_bytes = 0
for line in open('$input'):
    parts = line.strip().split()
    if len(parts) < 4: continue
    code = parts[0]
    t_sec = float(parts[1])
    size = int(float(parts[2]))
    times.append(t_sec * 1000)
    codes[code] = codes.get(code, 0) + 1
    total_bytes += size

if not times:
    print('No data')
    sys.exit(0)

times.sort()
n = len(times)
print(f'Requests:  {n}')
print(f'HTTP codes: {codes}')
print(f'Total data: {total_bytes/1e6:.1f} MB')
print(f'Latency (ms):')
print(f'  min={times[0]:.0f}  median={times[n//2]:.0f}  mean={sum(times)/n:.0f}  p95={times[int(n*0.95)]:.0f}  p99={times[int(n*0.99)]:.0f}  max={times[-1]:.0f}')
if times[-1] > 0:
    print(f'Throughput: {n/(sum(times)/1000):.0f} req/s (wall), {n*1000/times[-1]:.0f} req/s (elapsed)')
"
}

# ---------------------------------------------------------------------------
# Generate URL files for each scenario
# ---------------------------------------------------------------------------

echo "=== Generating test URLs ==="

# For each slide, discover L0 level and tile grid
generate_slide_urls() {
  local slide_id="$1"
  local dzi_url="${BASE_URL}/dzi/${slide_id}.dzi"

  # Parse DZI to get tile size and dimensions
  local dzi=$(curl -sf "$dzi_url" 2>/dev/null)
  if [[ -z "$dzi" ]]; then
    echo "  SKIP $slide_id (no DZI)" >&2
    return
  fi

  local dims=$(echo "$dzi" | python3 -c "
import sys, xml.etree.ElementTree as ET
tree = ET.parse(sys.stdin)
root = tree.getroot()
ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
ts = int(root.get('TileSize', 256))
el = root.find(f'{ns}Size')
w, h = int(el.get('Width')), int(el.get('Height'))
import math
max_level = math.ceil(math.log2(max(w, h)))
tiles_x = math.ceil(w / ts)
tiles_y = math.ceil(h / ts)
print(f'{max_level} {tiles_x} {tiles_y} {ts}')
" 2>/dev/null)

  if [[ -z "$dims" ]]; then
    return
  fi

  local l0=$(echo "$dims" | cut -d' ' -f1)
  local tx=$(echo "$dims" | cut -d' ' -f2)
  local ty=$(echo "$dims" | cut -d' ' -f3)
  local l1=$((l0 - 1))
  local l2=$((l0 - 2))

  echo "$slide_id $l0 $l1 $l2 $tx $ty"
}

SLIDE_INFO_FILE="$RESULTS_DIR/slide_info.txt"
> "$SLIDE_INFO_FILE"

while IFS='|' read -r sid label; do
  info=$(generate_slide_urls "$sid")
  if [[ -n "$info" ]]; then
    echo "$info" >> "$SLIDE_INFO_FILE"
  fi
done <<< "$SLIDE_IDS"

NUM_TESTABLE=$(wc -l < "$SLIDE_INFO_FILE" | tr -d ' ')
echo "Testable slides: $NUM_TESTABLE"

if [[ "$NUM_TESTABLE" -eq 0 ]]; then
  echo "ERROR: No slides available for testing"
  exit 1
fi

# Generate all URLs via a single Python script that models realistic behavior
# Write the URL generator as a temp script to avoid heredoc escaping issues
GEN_SCRIPT="$RESULTS_DIR/gen_urls.py"
cat > "$GEN_SCRIPT" << 'PYSCRIPT'
import random, math, json, os, sys

random.seed(42)
BASE_URL = sys.argv[1]
SLIDE_INFO_FILE = sys.argv[2]
OUT_DIR = sys.argv[3]

NUM_SESSIONS = 10
VIEWPORT_W = 6
VIEWPORT_H = 4
PAN_STEPS = 8
NUM_REGIONS = 3

with open(SLIDE_INFO_FILE) as f:
    slides = [line.strip().split() for line in f]

if not slides:
    print("ERROR: no slides", file=sys.stderr)
    sys.exit(1)

slide_tissue = {}
dzi_dir = os.path.expanduser("~/dev/data/WSI/dzi")
for parts in slides:
    sid = parts[0]
    base_id = sid
    for suffix in ['_original', '_jxl80', '_jxl40', '_jxl40_nonoise', '_jpeg80', '_jpeg40']:
        if sid.endswith(suffix):
            base_id = sid[:-len(suffix)]
            break
    tj = os.path.join(dzi_dir, f"{base_id}.tissue.json")
    if os.path.exists(tj) and sid not in slide_tissue:
        with open(tj) as fh:
            meta = json.load(fh)
        included = meta.get('included_detect_tiles', [])
        scale = meta.get('scale', 1)
        centers = [(tx * scale + scale // 2, ty * scale + scale // 2)
                   for tx, ty in included]
        slide_tissue[sid] = centers

session_urls = []
cold_urls = []

for session_idx in range(NUM_SESSIONS):
    sid, l0s, l1s, l2s, txs, tys = random.choice(slides)
    l0, l1, l2, tx, ty = int(l0s), int(l1s), int(l2s), int(txs), int(tys)
    l3 = l0 - 3
    l4 = l0 - 4

    centers = slide_tissue.get(sid, [])
    if centers:
        cx, cy = random.choice(centers)
        cx, cy = min(cx, tx - 1), min(cy, ty - 1)
    else:
        cx = random.randint(tx // 4, 3 * tx // 4)
        cy = random.randint(ty // 4, 3 * ty // 4)

    for region_idx in range(NUM_REGIONS):
        def viewport(level, base_x, base_y, vw=VIEWPORT_W, vh=VIEWPORT_H):
            for dy in range(vh):
                for dx in range(vw):
                    x = min(base_x + dx, tx - 1)
                    y = min(base_y + dy, ty - 1)
                    session_urls.append(f"{BASE_URL}/tiles/{sid}/{level}/{x}_{y}.jpg")

        # Phase 1: Overview L4
        viewport(l4, max(0, cx // 16 - VIEWPORT_W // 2), max(0, cy // 16 - VIEWPORT_H // 2))
        # Phase 2: L3
        viewport(l3, max(0, cx // 8 - VIEWPORT_W // 2), max(0, cy // 8 - VIEWPORT_H // 2))
        # Phase 3: L2
        viewport(l2, max(0, cx // 4 - VIEWPORT_W // 2), max(0, cy // 4 - VIEWPORT_H // 2))
        # Phase 4: L1
        viewport(l1, max(0, cx // 2 - VIEWPORT_W // 2), max(0, cy // 2 - VIEWPORT_H // 2))
        # Phase 5: L0 initial
        base_x0 = max(0, cx - VIEWPORT_W // 2)
        base_y0 = max(0, cy - VIEWPORT_H // 2)
        viewport(l0, base_x0, base_y0)

        # Phase 6: Pan at L0
        pan_x, pan_y = base_x0, base_y0
        for step in range(PAN_STEPS):
            if step % 2 == 0:
                pan_x = min(pan_x + VIEWPORT_W, tx - VIEWPORT_W)
            else:
                pan_y = min(pan_y + VIEWPORT_H, ty - VIEWPORT_H)
            viewport(l0, pan_x, pan_y)

        # Phase 7: Zoom back out to L2
        mid_x = (base_x0 + pan_x) // 2
        mid_y = (base_y0 + pan_y) // 2
        viewport(l2, max(0, mid_x // 4 - VIEWPORT_W // 2), max(0, mid_y // 4 - VIEWPORT_H // 2))

        # Move to next region
        if centers and region_idx < NUM_REGIONS - 1:
            cx, cy = random.choice(centers)
            cx, cy = min(cx, tx - 1), min(cy, ty - 1)
        else:
            cx = min(cx + tx // 6, tx - 1)
            cy = min(cy + ty // 6, ty - 1)

    # Cold-start URLs
    for _ in range(min(20, tx)):
        cold_urls.append(f"{BASE_URL}/tiles/{sid}/{l0}/{random.randint(0,tx-1)}_{random.randint(0,ty-1)}.jpg")

with open(os.path.join(OUT_DIR, "session_urls.txt"), "w") as f:
    f.write("\n".join(session_urls) + "\n")
with open(os.path.join(OUT_DIR, "cold_urls.txt"), "w") as f:
    f.write("\n".join(cold_urls) + "\n")

print(f"Sessions: {NUM_SESSIONS}, regions/session: {NUM_REGIONS}")
print(f"Session URLs: {len(session_urls)} ({len(session_urls) / NUM_SESSIONS:.0f}/session)")
print(f"Cold start URLs: {len(cold_urls)}")
PYSCRIPT

python3 "$GEN_SCRIPT" "$BASE_URL" "$SLIDE_INFO_FILE" "$RESULTS_DIR"

COLD_URLS="$RESULTS_DIR/cold_urls.txt"
SESSION_URLS="$RESULTS_DIR/session_urls.txt"
COLD_COUNT=$(wc -l < "$COLD_URLS" | tr -d ' ')
SESSION_COUNT=$(wc -l < "$SESSION_URLS" | tr -d ' ')
echo "Cold start URLs: $COLD_COUNT"
echo "Session sim URLs: $SESSION_COUNT"
echo ""

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

SERVER_PID=$(get_server_pid)
if [[ -z "$SERVER_PID" ]]; then
  echo "WARNING: Cannot find server PID for resource monitoring"
fi

echo "Server PID: $SERVER_PID"
echo ""

# Save server config
echo "Port: $PORT" > "$RESULTS_DIR/config.txt"
echo "Slides root: $SLIDES_ROOT" >> "$RESULTS_DIR/config.txt"
echo "Slides: $NUM_TESTABLE" >> "$RESULTS_DIR/config.txt"
echo "Families: $NUM_FAMILIES" >> "$RESULTS_DIR/config.txt"
echo "Concurrency: $CONCURRENCY_LEVELS" >> "$RESULTS_DIR/config.txt"
echo "Date: $(date)" >> "$RESULTS_DIR/config.txt"

# ── Test 1: Cold start (no cache, random tiles) ──
echo "════════════════════════════════════════════════════"
echo "  TEST 1: COLD START (random L0 tiles, no locality)"
echo "════════════════════════════════════════════════════"

IFS=',' read -ra CONC_ARRAY <<< "$CONCURRENCY_LEVELS"
for conc in "${CONC_ARRAY[@]}"; do
  echo ""
  echo "--- Concurrency: $conc ---"

  rss_before=""
  [[ -n "$SERVER_PID" ]] && rss_before=$(get_rss_mb "$SERVER_PID")

  out="$RESULTS_DIR/cold_c${conc}.txt"
  start=$(python3 -c "import time; print(time.time())")
  run_curl_bench "$COLD_URLS" "$conc" > "$out"
  elapsed=$(python3 -c "import time; print(f'{time.time() - $start:.1f}')")

  rss_after=""
  cpu=""
  [[ -n "$SERVER_PID" ]] && rss_after=$(get_rss_mb "$SERVER_PID") && cpu=$(get_cpu_pct "$SERVER_PID")

  echo "Elapsed: ${elapsed}s  RSS: ${rss_before:-?}→${rss_after:-?} MB  CPU: ${cpu:-?}%"
  compute_stats "$out"
done

# ── Test 2: Session simulation (pan/zoom with locality) ──
echo ""
echo "════════════════════════════════════════════════════"
echo "  TEST 2: SESSION SIMULATION (pan/zoom, 10 sessions)"
echo "════════════════════════════════════════════════════"

for conc in "${CONC_ARRAY[@]}"; do
  echo ""
  echo "--- Concurrency: $conc ---"

  rss_before=""
  [[ -n "$SERVER_PID" ]] && rss_before=$(get_rss_mb "$SERVER_PID")

  out="$RESULTS_DIR/session_c${conc}.txt"
  start=$(python3 -c "import time; print(time.time())")
  run_curl_bench "$SESSION_URLS" "$conc" > "$out"
  elapsed=$(python3 -c "import time; print(f'{time.time() - $start:.1f}')")

  rss_after=""
  cpu=""
  [[ -n "$SERVER_PID" ]] && rss_after=$(get_rss_mb "$SERVER_PID") && cpu=$(get_cpu_pct "$SERVER_PID")

  echo "Elapsed: ${elapsed}s  RSS: ${rss_before:-?}→${rss_after:-?} MB  CPU: ${cpu:-?}%"
  compute_stats "$out"
done

# ── Test 3: Sustained load with growing concurrency ──
echo ""
echo "════════════════════════════════════════════════════"
echo "  TEST 3: SUSTAINED LOAD (${SUSTAINED_SECS}s per concurrency level)"
echo "════════════════════════════════════════════════════"

# Use the session URLs (realistic pattern) for sustained test
# They have locality and real zoom/pan behavior

# Monitor resources during all sustained tests
MONITOR_LOG="$RESULTS_DIR/resource_monitor.txt"
echo "timestamp_ms,rss_mb,cpu_pct,concurrency" > "$MONITOR_LOG"

SUSTAINED_CONC_LEVELS="4,8,16,32,64,128"
IFS=',' read -ra SUSTAINED_CONC <<< "$SUSTAINED_CONC_LEVELS"

for conc in "${SUSTAINED_CONC[@]}"; do
  echo ""
  echo "--- Sustained concurrency: $conc (${SUSTAINED_SECS}s) ---"

  # Start resource monitor
  (
    while true; do
      ts=$(python3 -c "import time; print(int(time.time()*1000))")
      rss=$(get_rss_mb "$SERVER_PID" 2>/dev/null || echo "0")
      cpu=$(get_cpu_pct "$SERVER_PID" 2>/dev/null || echo "0")
      echo "$ts,$rss,$cpu,$conc" >> "$MONITOR_LOG"
      sleep 1
    done
  ) &
  MON_PID=$!

  rss_before=""
  [[ -n "$SERVER_PID" ]] && rss_before=$(get_rss_mb "$SERVER_PID")

  out="$RESULTS_DIR/sustained_c${conc}.txt"
  start=$(python3 -c "import time; print(time.time())")

  # Feed session URLs repeatedly for SUSTAINED_SECS duration
  timeout "${SUSTAINED_SECS}s" bash -c "
    while true; do
      cat '$SESSION_URLS'
    done | xargs -P $conc -I{} curl -sf -o /dev/null -w '%{http_code} %{time_total} %{size_download} {}\n' '{}' 2>/dev/null
  " > "$out" 2>/dev/null || true

  elapsed=$(python3 -c "import time; print(f'{time.time() - $start:.1f}')")

  kill "$MON_PID" 2>/dev/null || true

  rss_after=""
  cpu=""
  [[ -n "$SERVER_PID" ]] && rss_after=$(get_rss_mb "$SERVER_PID") && cpu=$(get_cpu_pct "$SERVER_PID")

  echo "Elapsed: ${elapsed}s  RSS: ${rss_before:-?}→${rss_after:-?} MB  CPU: ${cpu:-?}%"
  compute_stats "$out"
done

# ── Final metrics snapshot ──
echo ""
echo "════════════════════════════════════════════════════"
echo "  SERVER METRICS"
echo "════════════════════════════════════════════════════"
get_metrics | python3 -c "
import sys
for line in sys.stdin:
    line = line.strip()
    if line and not line.startswith('#') and ('cache' in line.lower() or 'tile' in line.lower() or 'family' in line.lower()):
        print(f'  {line}')
" 2>/dev/null || echo "  (metrics endpoint not available)"

echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Resource monitor: $MONITOR_LOG"
