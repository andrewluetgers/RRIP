#!/usr/bin/env bash
#
# WSI tile server load test using vegeta.
#
# Simulates realistic viewer sessions (zoom, pan, zoom out) across multiple
# slides using tissue-aware paths, and measures latency/throughput at
# increasing concurrency with proper cache hit/miss breakdowns.
#
# Usage:
#   bash evals/scripts/bench_vegeta.sh [options]
#
# Options:
#   --port PORT             Server port (default: 3007)
#   --duration SECS         Duration per rate level (default: 30)
#   --rates R1,R2,...       Requests/sec levels (default: 50,100,200,400,800,1200)
#   --slide-filter PATTERN  Only test slides matching pattern
#
set -uo pipefail

PORT=3007
DURATION=30
RATES="50,100,200,400,800,1200"
SLIDE_FILTER=""
BASE_URL="${BASE_URL:-http://localhost:${PORT}}"
RESULTS_DIR="evals/runs/bench_vegeta_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)         PORT="$2"; shift 2;;
    --duration)     DURATION="$2"; shift 2;;
    --rates)        RATES="$2"; shift 2;;
    --slide-filter) SLIDE_FILTER="$2"; shift 2;;
    *)              echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Verify server is up
# ---------------------------------------------------------------------------
if ! curl -sfk "${BASE_URL}/healthz" > /dev/null 2>&1; then
  echo "ERROR: Tile server not responding at $BASE_URL"
  exit 1
fi

SERVER_PID=$(lsof -i :"$PORT" -t 2>/dev/null | head -1)
echo "Server PID: ${SERVER_PID:-unknown}"

# ---------------------------------------------------------------------------
# Discover slides and generate session URLs
# ---------------------------------------------------------------------------
echo "=== Discovering slides ==="

SLIDES_JSON=$(curl -sfk "${BASE_URL}/slides.json")
SLIDE_IDS=$(echo "$SLIDES_JSON" | python3 -c "
import sys, json
slides = json.load(sys.stdin)
for s in slides:
    print(s['id'])
")

if [[ -n "$SLIDE_FILTER" ]]; then
  SLIDE_IDS=$(echo "$SLIDE_IDS" | grep -i "$SLIDE_FILTER")
fi

TOTAL_SLIDES=$(echo "$SLIDE_IDS" | wc -l | tr -d ' ')
echo "Slides: $TOTAL_SLIDES"

# Build slide info (level, grid size)
SLIDE_INFO="$RESULTS_DIR/slide_info.txt"
> "$SLIDE_INFO"
while read -r sid; do
  dzi=$(curl -sfk "${BASE_URL}/dzi/${sid}.dzi" 2>/dev/null)
  [[ -z "$dzi" ]] && continue
  dims=$(echo "$dzi" | python3 -c "
import sys, xml.etree.ElementTree as ET, math
tree = ET.parse(sys.stdin)
root = tree.getroot()
ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
ts = int(root.get('TileSize', 256))
el = root.find(f'{ns}Size')
w, h = int(el.get('Width')), int(el.get('Height'))
ml = math.ceil(math.log2(max(w, h)))
print(f'{ml} {math.ceil(w/ts)} {math.ceil(h/ts)}')
" 2>/dev/null)
  [[ -n "$dims" ]] && echo "$sid $dims" >> "$SLIDE_INFO"
done <<< "$SLIDE_IDS"

NUM_TESTABLE=$(wc -l < "$SLIDE_INFO" | tr -d ' ')
echo "Testable: $NUM_TESTABLE"

# Generate session URLs via Python
python3 "$( dirname "$0" )/../scripts/bench_wsi_sessions.sh" 2>/dev/null || true  # ignore if old script

# Use the URL generator from bench_wsi_sessions
GEN_SCRIPT="$RESULTS_DIR/gen_urls.py"
cat > "$GEN_SCRIPT" << 'PYSCRIPT'
import random, json, os, sys

random.seed(42)
BASE_URL, SLIDE_INFO, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]

VIEWPORT_W, VIEWPORT_H = 6, 4
PAN_STEPS, NUM_REGIONS, NUM_SESSIONS = 8, 3, 20

with open(SLIDE_INFO) as f:
    slides = [line.strip().split() for line in f if line.strip()]

# Load tissue centers
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
        scale = meta.get('scale', 1)
        centers = [(tx * scale + scale // 2, ty * scale + scale // 2)
                   for tx, ty in meta.get('included_detect_tiles', [])]
        slide_tissue[sid] = centers

urls = []
for _ in range(NUM_SESSIONS):
    sid, l0s, txs, tys = random.choice(slides)[0], *[int(x) for x in random.choice(slides)[1:]]
    # Re-read from the chosen slide
    chosen = random.choice(slides)
    sid = chosen[0]
    l0, tx, ty = int(chosen[1]), int(chosen[2]), int(chosen[3])
    l1, l2, l3, l4 = l0-1, l0-2, l0-3, l0-4

    centers = slide_tissue.get(sid, [])
    cx = min(random.choice(centers)[0], tx-1) if centers else random.randint(tx//4, 3*tx//4)
    cy = min(random.choice(centers)[1], ty-1) if centers else random.randint(ty//4, 3*ty//4)

    for _ in range(NUM_REGIONS):
        def vp(level, bx, by):
            for dy in range(VIEWPORT_H):
                for dx in range(VIEWPORT_W):
                    urls.append(f"GET {BASE_URL}/tiles/{sid}/{level}/{min(bx+dx,tx-1)}_{min(by+dy,ty-1)}.jpg")

        vp(l4, max(0, cx//16 - 3), max(0, cy//16 - 2))
        vp(l3, max(0, cx//8 - 3), max(0, cy//8 - 2))
        vp(l2, max(0, cx//4 - 3), max(0, cy//4 - 2))
        vp(l1, max(0, cx//2 - 3), max(0, cy//2 - 2))
        bx0, by0 = max(0, cx - 3), max(0, cy - 2)
        vp(l0, bx0, by0)

        px, py = bx0, by0
        for step in range(PAN_STEPS):
            if step % 2 == 0: px = min(px + VIEWPORT_W, tx - VIEWPORT_W)
            else: py = min(py + VIEWPORT_H, ty - VIEWPORT_H)
            vp(l0, px, py)

        vp(l2, max(0, (bx0+px)//8 - 3), max(0, (by0+py)//8 - 2))

        if centers:
            cx, cy = random.choice(centers)
            cx, cy = min(cx, tx-1), min(cy, ty-1)
        else:
            cx, cy = min(cx + tx//6, tx-1), min(cy + ty//6, ty-1)

with open(os.path.join(OUT_DIR, "targets.txt"), "w") as f:
    f.write("\n".join(urls) + "\n")

# --- Cold-miss targets: unique L1/L2 tiles, never repeated ---
# Request at L1 and L2 to force real decode+downsample work (not static reads).
# Each URL is unique — guarantees 0% cache hit rate.
cold_urls = []
seen = set()
for parts in slides * 200:
    sid = parts[0]
    l0, tx, ty = int(parts[1]), int(parts[2]), int(parts[3])
    l1, l2 = l0 - 1, l0 - 2
    centers = slide_tissue.get(sid, [])
    if not centers:
        continue
    cx, cy = random.choice(centers)
    cx, cy = min(cx, tx-1), min(cy, ty-1)
    # Pick L1 or L2 level (these force server-side decode+downsample)
    level = random.choice([l1, l1, l2])  # bias toward L1 (more tiles)
    if level == l1:
        x = max(0, min(cx // 2 + random.randint(-8, 8), tx // 2 - 1))
        y = max(0, min(cy // 2 + random.randint(-8, 8), ty // 2 - 1))
    else:
        x = max(0, min(cx // 4 + random.randint(-4, 4), tx // 4 - 1))
        y = max(0, min(cy // 4 + random.randint(-4, 4), ty // 4 - 1))
    key = f"{sid}/{level}/{x}_{y}"
    if key in seen:
        continue
    seen.add(key)
    cold_urls.append(f"GET {BASE_URL}/tiles/{key}.jpg")
    if len(cold_urls) >= 50000:
        break

with open(os.path.join(OUT_DIR, "cold_targets.txt"), "w") as f:
    f.write("\n".join(cold_urls) + "\n")

# --- Blended targets at various cache hit ratios ---
# Mix cold (unique) URLs with warm (repeated) URLs
for hit_pct in [0, 25, 50, 75]:
    miss_pct = 100 - hit_pct
    blended = []
    cold_idx = 0
    warm_pool = urls[:2000]  # first 2000 session URLs as the "warm" pool
    for i in range(20000):
        if random.randint(0, 99) < miss_pct and cold_idx < len(cold_urls):
            blended.append(cold_urls[cold_idx])
            cold_idx += 1
        else:
            blended.append(random.choice(warm_pool))
    fname = os.path.join(OUT_DIR, f"blend_{hit_pct}hit_targets.txt")
    with open(fname, "w") as f:
        f.write("\n".join(blended) + "\n")

print(f"Generated {len(urls)} session URLs, {len(cold_urls)} cold URLs")
print(f"Blended targets: 0%/25%/50%/75% hit rate × 20K URLs each")
PYSCRIPT

python3 "$GEN_SCRIPT" "$BASE_URL" "$SLIDE_INFO" "$RESULTS_DIR"
TARGETS="$RESULTS_DIR/targets.txt"
TOTAL_URLS=$(wc -l < "$TARGETS" | tr -d ' ')
echo "Total URLs: $TOTAL_URLS"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

get_rss_mb() {
  ps -o rss= -p "$1" 2>/dev/null | awk '{printf "%.1f", $1/1024}'
}

snapshot_metrics() {
  curl -sfk "${BASE_URL}/metrics" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Run vegeta at increasing rates
# ---------------------------------------------------------------------------

IFS=',' read -ra RATE_ARRAY <<< "$RATES"

echo "════════════════════════════════════════════════════════════════"
echo "  VEGETA LOAD TEST: ${DURATION}s per rate, session-based URLs"
echo "════════════════════════════════════════════════════════════════"
echo ""

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
printf "%-8s %8s %8s %8s %8s %8s %8s %8s %10s %8s %6s\n" \
  "Rate" "Total" "200s" "404s" "Errs" "Med(ms)" "P95(ms)" "P99(ms)" "Max(ms)" "RSS(MB)" "CPU%" > "$SUMMARY_FILE"
echo "--------------------------------------------------------------------------------------------" >> "$SUMMARY_FILE"

# Capture baseline metrics
METRICS_BEFORE=$(snapshot_metrics)

for rate in "${RATE_ARRAY[@]}"; do
  echo "--- Rate: ${rate} req/s for ${DURATION}s ---"

  rss_before=""
  [[ -n "$SERVER_PID" ]] && rss_before=$(get_rss_mb "$SERVER_PID")

  # Run vegeta attack
  RESULT_BIN="$RESULTS_DIR/result_r${rate}.bin"
  RESULT_TXT="$RESULTS_DIR/result_r${rate}.txt"
  RESULT_HIST="$RESULTS_DIR/hist_r${rate}.txt"

  vegeta attack \
    -targets="$TARGETS" \
    -rate="${rate}/1s" \
    -duration="${DURATION}s" \
    -timeout=10s \
    -max-workers=256 \
    -insecure \
    -http2 \
    > "$RESULT_BIN"

  # Generate reports
  vegeta report "$RESULT_BIN" > "$RESULT_TXT"
  vegeta report -type=hist "$RESULT_BIN" > "$RESULT_HIST" 2>/dev/null || true

  rss_after=""
  cpu=""
  [[ -n "$SERVER_PID" ]] && rss_after=$(get_rss_mb "$SERVER_PID") && cpu=$(ps -o %cpu= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')

  # Parse vegeta report
  total=$(grep "Requests" "$RESULT_TXT" | head -1 | awk '{print $3}' | tr -d '[]')
  success=$(grep "Success" "$RESULT_TXT" | awk '{print $3}')
  codes=$(grep -A20 "Status Codes" "$RESULT_TXT" | grep -E "^\s+\[" || true)
  code_200=$(echo "$codes" | grep "200" | awk '{print $2}' || echo "0")
  code_404=$(echo "$codes" | grep "404" | awk '{print $2}' || echo "0")
  code_0=$(echo "$codes" | grep "\[0\]" | awk '{print $2}' || echo "0")

  latencies=$(grep -A10 "Latencies" "$RESULT_TXT" | head -2)

  echo "  $(cat "$RESULT_TXT" | head -20)"
  echo "  RSS: ${rss_before:-?} → ${rss_after:-?} MB  CPU: ${cpu:-?}%"

  # Extract latency percentiles for summary
  med=$(echo "$latencies" | grep -oP 'mean=\K[0-9.]+[µm]?s' || echo "?")
  p50=$(echo "$latencies" | grep -oP '50th=\K[0-9.]+[µm]?s' || echo "?")
  p95=$(echo "$latencies" | grep -oP '95th=\K[0-9.]+[µm]?s' || echo "?")
  p99=$(echo "$latencies" | grep -oP '99th=\K[0-9.]+[µm]?s' || echo "?")
  pmax=$(echo "$latencies" | grep -oP 'max=\K[0-9.]+[µm]?s' || echo "?")

  printf "%-8s %8s %8s %8s %8s %8s %8s %8s %10s %8s %6s\n" \
    "${rate}/s" "${total:-?}" "${code_200:-0}" "${code_404:-0}" "${code_0:-0}" \
    "$p50" "$p95" "$p99" "$pmax" "${rss_after:-?}" "${cpu:-?}" >> "$SUMMARY_FILE"

  echo ""
done

# Capture after metrics
METRICS_AFTER=$(snapshot_metrics)

echo "════════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════════════════"
cat "$SUMMARY_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SERVER METRICS DELTA"
echo "════════════════════════════════════════════════════════════════"

# Show metrics diff if available
if [[ -n "$METRICS_BEFORE" ]] && [[ -n "$METRICS_AFTER" ]]; then
  python3 -c "
before = '''$METRICS_BEFORE'''
after = '''$METRICS_AFTER'''
bvals = {}
avals = {}
for line in before.strip().split('\n'):
    if line and not line.startswith('#'):
        parts = line.split()
        if len(parts) >= 2:
            try: bvals[parts[0]] = float(parts[1])
            except: pass
for line in after.strip().split('\n'):
    if line and not line.startswith('#'):
        parts = line.split()
        if len(parts) >= 2:
            try: avals[parts[0]] = float(parts[1])
            except: pass
for k in sorted(avals):
    if k in bvals and avals[k] != bvals[k]:
        delta = avals[k] - bvals[k]
        if delta != 0:
            print(f'  {k}: {bvals[k]:.0f} → {avals[k]:.0f} (+{delta:.0f})')
" 2>/dev/null || echo "  (could not parse metrics)"
fi

# ── Cache stress tests: various hit ratios ──
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  CACHE STRESS TEST: 200 req/s at 0%/25%/50%/75% cache hit rate"
echo "════════════════════════════════════════════════════════════════"
echo ""

STRESS_RATE=200
for hit_pct in 0 25 50 75; do
  target_file="$RESULTS_DIR/blend_${hit_pct}hit_targets.txt"

  # For 0% hit, use the cold targets directly (all unique L1/L2)
  if [[ "$hit_pct" -eq 0 ]]; then
    target_file="$RESULTS_DIR/cold_targets.txt"
  fi

  if [[ ! -f "$target_file" ]]; then
    echo "--- ${hit_pct}% hit: SKIP (no target file) ---"
    continue
  fi

  num_targets=$(wc -l < "$target_file" | tr -d ' ')
  echo "--- ${hit_pct}% cache hit target (${num_targets} unique URLs, ${STRESS_RATE} req/s, ${DURATION}s) ---"

  rss_before=""
  [[ -n "$SERVER_PID" ]] && rss_before=$(get_rss_mb "$SERVER_PID")

  RESULT_BIN="$RESULTS_DIR/stress_${hit_pct}hit.bin"
  vegeta attack \
    -targets="$target_file" \
    -rate="${STRESS_RATE}/1s" \
    -duration="${DURATION}s" \
    -timeout=10s \
    -max-workers=256 \
    -insecure \
    -http2 \
    > "$RESULT_BIN"

  rss_after=""
  cpu=""
  [[ -n "$SERVER_PID" ]] && rss_after=$(get_rss_mb "$SERVER_PID") && cpu=$(ps -o %cpu= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')

  echo "  RSS: ${rss_before:-?} → ${rss_after:-?} MB  CPU: ${cpu:-?}%"
  vegeta report "$RESULT_BIN" | head -8
  echo "  Latency buckets:"
  vegeta report -type=hist -buckets '[0,1ms,2ms,5ms,10ms,20ms,50ms,100ms,500ms]' "$RESULT_BIN"
  echo ""
done

echo ""
echo "Results: $RESULTS_DIR"
echo "Per-rate reports: result_r*.txt"
echo "Cache stress: stress_*hit.bin"
echo "Vegeta binary: use 'vegeta report -type=hist <file>.bin' for histograms"
