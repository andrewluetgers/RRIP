#!/usr/bin/env bash
# profile-wsi.sh — Build, sync, and run profiling encodes on RunPod.
#
# Usage:
#   ./gpu-encode/scripts/profile-wsi.sh [--dataset 3DHISTECH-1|3DHISTECH-2|all]
#   ./gpu-encode/scripts/profile-wsi.sh --skip-build --dataset 3DHISTECH-1
#
# This script:
#   1. Syncs code to RunPod pod
#   2. Builds the GPU encoder on the pod
#   3. Downloads DICOM test data (if not present)
#   4. Starts background resource monitoring (nvidia-smi + CPU/RAM)
#   5. Runs the GPU encoder with --profile --pack
#   6. Stops monitoring and downloads results
#
# Output: evals/runs/gpu_profile_<dataset>/ on the pod + downloaded locally

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
DATASET="all"
BASEQ=80
L1Q=60
L0Q=40
SUBSAMP=444
MAX_DELTA=20
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --baseq) BASEQ="$2"; shift 2 ;;
        --l1q) L1Q="$2"; shift 2 ;;
        --l0q) L0Q="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Get pod SSH info ---
echo "Querying RunPod for active pod..."
POD_INFO=$(/usr/bin/curl -s \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{"query":"{ myself { pods { id name desiredStatus machine { gpuDisplayName } runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"}' \
  "https://api.runpod.io/graphql")

read -r IP PORT GPU_NAME <<< "$(echo "$POD_INFO" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for pod in data['data']['myself']['pods']:
    if pod['desiredStatus'] != 'RUNNING' or not pod.get('runtime'):
        continue
    for port in pod['runtime']['ports']:
        if port['privatePort'] == 22 and port['isIpPublic']:
            gpu = pod['machine']['gpuDisplayName']
            print(f\"{port['ip']} {port['publicPort']} {gpu}\")
            sys.exit(0)
print('NONE NONE NONE')
")"

if [ "$IP" = "NONE" ]; then
    echo "ERROR: No running RunPod pod found."
    exit 1
fi

SSH_CMD="ssh -i ~/.ssh/id_runpod -o StrictHostKeyChecking=no root@$IP -p $PORT"
SCP_CMD="scp -i ~/.ssh/id_runpod -o StrictHostKeyChecking=no -P $PORT"

echo "Pod: $IP:$PORT ($GPU_NAME)"
echo "Dataset: $DATASET"
echo "Settings: baseq=$BASEQ l1q=$L1Q l0q=$L0Q subsamp=$SUBSAMP max_delta=$MAX_DELTA"
echo ""

# --- Sync code ---
echo "=== Syncing code to pod ==="
rsync -avz --progress \
    --exclude 'target' --exclude 'target2' --exclude 'node_modules' \
    --exclude '.git' --exclude 'evals/runs' --exclude 'evals/test-images' \
    --exclude 'data' --exclude '.venv' --exclude '__pycache__' \
    -e "ssh -i ~/.ssh/id_runpod -p $PORT" \
    "$REPO_ROOT/" \
    "root@$IP:/workspace/RRIP/"
echo ""

# --- Build ---
if [ "$SKIP_BUILD" = false ]; then
    echo "=== Building GPU encoder on pod ==="
    $SSH_CMD "source /root/.cargo/env && cd /workspace/RRIP/gpu-encode && cargo build --release 2>&1" | tail -5
    echo "Build complete."
    echo ""
fi

# --- Download datasets ---
download_dataset() {
    local name="$1"
    local url="$2"
    local data_dir="$3"
    local check_file="$4"

    echo "Checking $name..."
    if $SSH_CMD "test -f $check_file" 2>/dev/null; then
        echo "  Already present: $check_file"
        return
    fi

    echo "  Downloading $name..."
    $SSH_CMD "mkdir -p $data_dir && cd $data_dir && wget -q --show-progress '$url' -O $name.zip && unzip -o $name.zip && rm $name.zip" 2>&1
    echo "  Done."
}

# --- Run profiling encode ---
encode_profile() {
    local name="$1"
    local slide_path="$2"
    local tile_size="$3"
    local run_name="gpu_profile_${name}_b${BASEQ}_l1q${L1Q}_l0q${L0Q}"
    local remote_out="/workspace/RRIP/evals/runs/$run_name"
    local local_out="$REPO_ROOT/evals/runs/$run_name"

    echo ""
    echo "============================================="
    echo "=== Profiling $name ==="
    echo "============================================="
    echo "  Slide: $slide_path (tile=${tile_size})"
    echo "  Output: $remote_out"
    echo ""

    # Clean previous run
    $SSH_CMD "rm -rf $remote_out" 2>/dev/null || true

    # Start background resource monitoring on pod
    echo "Starting background resource monitoring..."
    $SSH_CMD "mkdir -p $remote_out && bash /workspace/RRIP/gpu-encode/scripts/resource-monitor.sh $remote_out start" 2>&1

    # Run encode with --profile --pack
    echo ""
    echo "Running GPU encode with --profile..."
    $SSH_CMD "source /root/.cargo/env && \
      time /workspace/RRIP/gpu-encode/target/release/origami-gpu-encode encode \
        --slide $slide_path \
        --out $remote_out \
        --baseq $BASEQ --l1q $L1Q --l0q $L0Q --subsamp $SUBSAMP \
        --optl2 --max-delta $MAX_DELTA --tile $tile_size \
        --pack --profile 2>&1"

    # Stop background resource monitoring
    echo ""
    echo "Stopping resource monitoring..."
    $SSH_CMD "bash /workspace/RRIP/gpu-encode/scripts/resource-monitor.sh $remote_out stop" 2>&1

    # Download results
    echo ""
    echo "Downloading results..."
    mkdir -p "$local_out"
    $SCP_CMD "root@$IP:$remote_out/timing_report.json" "$local_out/" 2>/dev/null || echo "  (no timing_report.json)"
    $SCP_CMD "root@$IP:$remote_out/summary.json" "$local_out/" 2>/dev/null || echo "  (no summary.json)"
    $SCP_CMD "root@$IP:$remote_out/gpu_monitor.csv" "$local_out/" 2>/dev/null || echo "  (no gpu_monitor.csv)"
    $SCP_CMD "root@$IP:$remote_out/cpu_monitor.csv" "$local_out/" 2>/dev/null || echo "  (no cpu_monitor.csv)"

    echo ""
    echo "--- Results ($name) ---"
    echo "summary.json:"
    cat "$local_out/summary.json" 2>/dev/null || echo "  (not available)"
    echo ""
    echo "timing_report.json:"
    cat "$local_out/timing_report.json" 2>/dev/null || echo "  (not available)"
    echo ""
}

# --- 3DHISTECH-1 ---
if [ "$DATASET" = "3DHISTECH-1" ] || [ "$DATASET" = "all" ]; then
    download_dataset "3DHISTECH-1" \
        "https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-1.zip" \
        "/workspace/data/3DHISTECH-1" \
        "/workspace/data/3DHISTECH-1/000005.dcm"

    SLIDE_1=$($SSH_CMD "test -f /workspace/data/3DHISTECH-1/000005.dcm && echo /workspace/data/3DHISTECH-1/000005.dcm || echo /workspace/data/000005.dcm")
    encode_profile "3dh1" "$SLIDE_1" 1024
fi

# --- 3DHISTECH-2 ---
if [ "$DATASET" = "3DHISTECH-2" ] || [ "$DATASET" = "all" ]; then
    download_dataset "3DHISTECH-2" \
        "https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-2.zip" \
        "/workspace/data/3DHISTECH-2" \
        "/workspace/data/3DHISTECH-2/4_1"

    encode_profile "3dh2" "/workspace/data/3DHISTECH-2/4_1" 512
fi

echo ""
echo "============================================="
echo "=== All profiling runs complete ==="
echo "============================================="
echo ""
echo "Local results in:"
ls -d "$REPO_ROOT"/evals/runs/gpu_profile_* 2>/dev/null || echo "  (none)"
