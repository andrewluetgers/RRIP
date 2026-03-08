#!/bin/bash
set -e

# ============================================================================
# WSI SR Training on RunPod
#
# Prerequisites:
#   - RUNPOD_API_KEY env var set
#   - SSH key at ~/.ssh/id_runpod
#   - This repo committed and pushed (we git clone on the pod)
#
# Usage:
#   # 1. Push your code first
#   git add wsi_sr/ && git commit -m "wsi_sr training code" && git push
#
#   # 2. Run this script
#   ./wsi_sr/runpod_train.sh
#
#   # Or manually: create pod, upload data, run training
# ============================================================================

echo "=== WSI SR Training Setup ==="

# ---- Step 1: Query existing pods or create a new one ----
echo ""
echo "Querying RunPod for existing pods..."
PODS=$(curl -s -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  --data '{"query":"{ myself { pods { id name desiredStatus machine { gpuDisplayName } runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"}' \
  https://api.runpod.io/graphql)

echo "$PODS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pods = data.get('data', {}).get('myself', {}).get('pods', [])
if not pods:
    print('No existing pods found.')
    print('Create a cheap GPU pod manually at https://www.runpod.io/console/pods')
    print('Recommended: RTX 4090 or A40, community cloud (~\$0.30-0.50/hr)')
    print('Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04')
else:
    for p in pods:
        gpu = p.get('machine', {}).get('gpuDisplayName', '?')
        status = p.get('desiredStatus', '?')
        ports = p.get('runtime', {}).get('ports', []) if p.get('runtime') else []
        ssh = next((port for port in ports if port.get('privatePort') == 22), None)
        ssh_str = f'{ssh[\"ip\"]}:{ssh[\"publicPort\"]}' if ssh else 'no SSH'
        print(f'  {p[\"name\"]} ({p[\"id\"]}): {gpu}, {status}, SSH={ssh_str}')
" 2>/dev/null || echo "Failed to query API. Is RUNPOD_API_KEY set?"

echo ""
echo "Once you have a running pod with SSH access, run:"
echo ""
echo "  # Set connection info from API query above"
echo "  POD_IP=<ip>"
echo "  POD_PORT=<port>"
echo ""
echo "  # Upload the DICOM data"
echo "  scp -i ~/.ssh/id_runpod -P \$POD_PORT data/3DHISTECH-1-extract/000005.dcm root@\$POD_IP:/workspace/"
echo ""
echo "  # Clone repo and start training"
echo "  ssh -i ~/.ssh/id_runpod root@\$POD_IP -p \$POD_PORT 'bash -s' << 'REMOTE_EOF'"
echo "    cd /workspace"
echo "    git clone https://github.com/<your-repo>/RRIP.git 2>/dev/null || (cd RRIP && git pull)"
echo "    cd RRIP/wsi_sr"
echo "    pip install -r requirements.txt pydicom"
echo ""
echo "    # Extract tiles (skip blank background)"
echo "    python prepare_tiles.py --dcm /workspace/000005.dcm --outdir /workspace/tiles --skip-blank 0.85"
echo ""
echo "    # Train SR model"
echo "    python train.py --tiles /workspace/tiles --mode sr --epochs 200 --batch 16 --jpeg-quality 95 --crop 256"
echo ""
echo "    # Export"
echo "    python export.py --checkpoint checkpoints/best.pt --output model_sr.onnx --benchmark"
echo "  REMOTE_EOF"
