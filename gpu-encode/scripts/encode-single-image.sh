#!/usr/bin/env bash
# encode-single-image.sh — Run GPU single-image encode on RunPod and download results.
#
# Usage:
#   ./gpu-encode/scripts/encode-single-image.sh [--image PATH] [--baseq N] [--l1q N] [--l0q N]
#
# Defaults encode evals/test-images/L0-1024.jpg with standard eval settings.
# Results are downloaded to evals/runs/gpu_<params>/ for viewer comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
IMAGE="evals/test-images/L0-1024.jpg"
BASEQ=95
L1Q=60
L0Q=40
SUBSAMP=444
OPTL2="--optl2"
MAX_DELTA=20
EXTRA_ARGS="--manifest --debug-images"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --image) IMAGE="$2"; shift 2 ;;
        --baseq) BASEQ="$2"; shift 2 ;;
        --l1q) L1Q="$2"; shift 2 ;;
        --l0q) L0Q="$2"; shift 2 ;;
        --subsamp) SUBSAMP="$2"; shift 2 ;;
        --no-optl2) OPTL2=""; shift ;;
        --max-delta) MAX_DELTA="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Determine run name
BASENAME=$(basename "$IMAGE" | sed 's/\.[^.]*$//')
RUN_NAME="gpu_${SUBSAMP}_b${BASEQ}_optl2_d${MAX_DELTA}_l1q${L1Q}_l0q${L0Q}"

# --- Get pod SSH info ---
POD_INFO=$(/usr/bin/curl -s \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{"query":"{ myself { pods { id name desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"}' \
  "https://api.runpod.io/graphql")

read -r IP PORT <<< "$(echo "$POD_INFO" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for pod in data['data']['myself']['pods']:
    if pod['desiredStatus'] != 'RUNNING' or not pod.get('runtime'):
        continue
    for port in pod['runtime']['ports']:
        if port['privatePort'] == 22 and port['isIpPublic']:
            print(f\"{port['ip']} {port['publicPort']}\")
            sys.exit(0)
print('NONE NONE')
")"

if [ "$IP" = "NONE" ]; then
    echo "ERROR: No running RunPod pod found."
    exit 1
fi

SSH_CMD="ssh -i ~/.ssh/id_runpod -o StrictHostKeyChecking=no root@$IP -p $PORT"
SCP_CMD="scp -i ~/.ssh/id_runpod -o StrictHostKeyChecking=no -P $PORT"

echo "Pod: $IP:$PORT"
echo "Image: $IMAGE"
echo "Run: $RUN_NAME"
echo "Settings: baseq=$BASEQ l1q=$L1Q l0q=$L0Q subsamp=$SUBSAMP max_delta=$MAX_DELTA"
echo ""

# Ensure test image exists on pod
echo "Uploading test image..."
$SSH_CMD "mkdir -p /workspace/RRIP/evals/test-images"
$SCP_CMD "$REPO_ROOT/$IMAGE" "root@$IP:/workspace/RRIP/$IMAGE" 2>/dev/null || true

# Run encode
echo "Running GPU encode..."
$SSH_CMD "source /root/.cargo/env && \
  rm -rf /workspace/RRIP/evals/runs/$RUN_NAME && \
  time /workspace/RRIP/gpu-encode/target/release/origami-gpu-encode encode \
    --image /workspace/RRIP/$IMAGE \
    --out /workspace/RRIP/evals/runs/$RUN_NAME \
    --baseq $BASEQ --l1q $L1Q --l0q $L0Q --subsamp $SUBSAMP \
    $OPTL2 --max-delta $MAX_DELTA $EXTRA_ARGS 2>&1"

# Download results
echo ""
echo "Downloading results..."
mkdir -p "$REPO_ROOT/evals/runs/$RUN_NAME"
rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_runpod -p $PORT" \
  "root@$IP:/workspace/RRIP/evals/runs/$RUN_NAME/" \
  "$REPO_ROOT/evals/runs/$RUN_NAME/"

echo ""
echo "Done! Results in evals/runs/$RUN_NAME/"
echo "Start the viewer: cd evals/viewer && pnpm start"
