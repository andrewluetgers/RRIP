#!/usr/bin/env bash
# encode-wsi.sh — Download DICOM WSI test data and run GPU WSI encode on RunPod.
#
# Usage:
#   ./gpu-encode/scripts/encode-wsi.sh [--dataset 3DHISTECH-1|3DHISTECH-2|all]
#
# This script:
#   1. Downloads DICOM WSI test datasets from openslide.cs.cmu.edu (if not present)
#   2. Runs the GPU encoder on the highest-resolution level
#   3. Reports timing and throughput
#
# Datasets:
#   3DHISTECH-1: 57344x60416, 1024x1024 tiles, 3304 frames, ~270MB
#   3DHISTECH-2: 267776x370688, 512x512 tiles, 25264+16118 frames, ~1.6GB (split across 2 files)

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
EXTRA_ARGS="--pack"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --baseq) BASEQ="$2"; shift 2 ;;
        --l1q) L1Q="$2"; shift 2 ;;
        --l0q) L0Q="$2"; shift 2 ;;
        --max-parents) EXTRA_ARGS="$EXTRA_ARGS --max-parents $2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# --- Get pod SSH info ---
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

echo "Pod: $IP:$PORT ($GPU_NAME)"
echo "Dataset: $DATASET"
echo "Settings: baseq=$BASEQ l1q=$L1Q l0q=$L0Q subsamp=$SUBSAMP max_delta=$MAX_DELTA"
echo ""

# --- Download and extract test data ---
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

# --- Encode a WSI ---
encode_wsi() {
    local name="$1"
    local slide_path="$2"
    local tile_size="$3"
    local run_name="gpu_dicom_${name}_b${BASEQ}_l1q${L1Q}_l0q${L0Q}"

    echo ""
    echo "=== Encoding $name ==="
    echo "  Slide: $slide_path (tile=${tile_size})"
    echo "  Output: evals/runs/$run_name"
    echo ""

    $SSH_CMD "source /root/.cargo/env && \
      rm -rf /workspace/RRIP/evals/runs/$run_name && \
      time /workspace/RRIP/gpu-encode/target/release/origami-gpu-encode encode \
        --slide $slide_path \
        --out /workspace/RRIP/evals/runs/$run_name \
        --baseq $BASEQ --l1q $L1Q --l0q $L0Q --subsamp $SUBSAMP \
        --optl2 --max-delta $MAX_DELTA --tile $tile_size $EXTRA_ARGS 2>&1"

    echo ""
    echo "--- Summary ($name) ---"
    $SSH_CMD "cat /workspace/RRIP/evals/runs/$run_name/summary.json" 2>/dev/null
    echo ""
}

# --- 3DHISTECH-1 ---
if [ "$DATASET" = "3DHISTECH-1" ] || [ "$DATASET" = "all" ]; then
    download_dataset "3DHISTECH-1" \
        "https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-1.zip" \
        "/workspace/data/3DHISTECH-1" \
        "/workspace/data/3DHISTECH-1/000005.dcm"

    # If the files were extracted to /workspace/data/ directly (flat), handle both cases
    SLIDE_1=$($SSH_CMD "test -f /workspace/data/3DHISTECH-1/000005.dcm && echo /workspace/data/3DHISTECH-1/000005.dcm || echo /workspace/data/000005.dcm")
    encode_wsi "3dh1" "$SLIDE_1" 1024
fi

# --- 3DHISTECH-2 ---
if [ "$DATASET" = "3DHISTECH-2" ] || [ "$DATASET" = "all" ]; then
    download_dataset "3DHISTECH-2" \
        "https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-2.zip" \
        "/workspace/data/3DHISTECH-2" \
        "/workspace/data/3DHISTECH-2/4_1"

    # 3DHISTECH-2 highest-res level is split: 4_1 (25264 frames) + 4_2 (16118 frames)
    # Encode 4_1 (the larger partition)
    encode_wsi "3dh2" "/workspace/data/3DHISTECH-2/4_1" 512
fi

echo ""
echo "=== All encodes complete ==="
