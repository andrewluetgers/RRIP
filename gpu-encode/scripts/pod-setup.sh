#!/usr/bin/env bash
# pod-setup.sh — Set up the RunPod GPU pod for ORIGAMI GPU encoding.
#
# Usage: ./gpu-encode/scripts/pod-setup.sh
#
# Prerequisites:
#   - RUNPOD_API_KEY env var set
#   - SSH key at ~/.ssh/id_runpod
#   - A running RunPod pod (use RunPod web UI to create one)
#
# This script:
#   1. Queries the RunPod API for the running pod's SSH details
#   2. Syncs the RRIP codebase to the pod
#   3. Installs Rust if needed
#   4. Builds the GPU encoder (release mode)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Query RunPod for pod SSH details ---
echo "Querying RunPod API for running pods..."
POD_INFO=$(/usr/bin/curl -s \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{"query":"{ myself { pods { id name desiredStatus machine { gpuDisplayName } runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"}' \
  "https://api.runpod.io/graphql")

# Parse the first running pod with SSH access
SSH_IP=$(echo "$POD_INFO" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for pod in data['data']['myself']['pods']:
    if pod['desiredStatus'] != 'RUNNING' or not pod.get('runtime'):
        continue
    for port in pod['runtime']['ports']:
        if port['privatePort'] == 22 and port['isIpPublic']:
            print(f\"{port['ip']}:{port['publicPort']}:{pod['name']}:{pod['machine']['gpuDisplayName']}\")
            sys.exit(0)
print('NONE')
")

if [ "$SSH_IP" = "NONE" ]; then
    echo "ERROR: No running RunPod pod found with SSH access."
    echo "Create a pod at https://www.runpod.io/console/pods"
    echo "Recommended: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    exit 1
fi

IP=$(echo "$SSH_IP" | cut -d: -f1)
PORT=$(echo "$SSH_IP" | cut -d: -f2)
POD_NAME=$(echo "$SSH_IP" | cut -d: -f3)
GPU=$(echo "$SSH_IP" | cut -d: -f4)

echo "Found pod: $POD_NAME ($GPU)"
echo "  SSH: $IP:$PORT"

SSH_CMD="ssh -i ~/.ssh/id_runpod -o StrictHostKeyChecking=no root@$IP -p $PORT"

# --- Sync codebase ---
echo ""
echo "Syncing codebase to pod..."
rsync -avz --progress \
  --exclude 'target' --exclude 'target2' --exclude 'node_modules' \
  --exclude '.git' --exclude 'evals/runs' --exclude 'evals/test-images' \
  --exclude 'data' --exclude '.venv' --exclude '__pycache__' \
  -e "ssh -i ~/.ssh/id_runpod -p $PORT" \
  "$REPO_ROOT/" \
  "root@$IP:/workspace/RRIP/"

# --- Install Rust if needed ---
echo ""
echo "Checking Rust installation..."
$SSH_CMD "source /root/.cargo/env 2>/dev/null && rustc --version || (echo 'Installing Rust...' && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)"

# --- Build GPU encoder ---
echo ""
echo "Building GPU encoder (release)..."
$SSH_CMD "source /root/.cargo/env && cd /workspace/RRIP/gpu-encode && cargo build --release 2>&1"

echo ""
echo "Setup complete! Pod is ready for GPU encoding."
echo ""
echo "SSH into the pod:"
echo "  ssh -i ~/.ssh/id_runpod root@$IP -p $PORT"
echo ""
echo "Or run encode scripts directly:"
echo "  ./gpu-encode/scripts/encode-single-image.sh"
echo "  ./gpu-encode/scripts/encode-wsi.sh"
