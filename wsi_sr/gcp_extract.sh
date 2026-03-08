#!/bin/bash
# =================================================================
# Stage 1: Create GCP VM and extract tiles from TCGA
#
# Creates a spot VM in us-central1, installs deps, runs extraction,
# uploads tiles to GCS, then deletes the VM.
#
# Usage:
#   # Stage 1 (small, 35 slides):
#   bash gcp_extract.sh stage1
#
#   # Stage 2 (full, 800 slides):
#   bash gcp_extract.sh stage2
#
#   # Just create VM (don't auto-run extraction):
#   bash gcp_extract.sh create-only
#
#   # Delete VM when done:
#   bash gcp_extract.sh delete
# =================================================================

set -euo pipefail

PROJECT="wsi-1-480715"
ZONE="${GCP_ZONE:-us-central1-b}"
BUCKET="gs://wsi-1-480715-tcga-tiles"

# VM config — adjust based on stage
STAGE="${1:-stage1}"

case "$STAGE" in
  stage1)
    VM_NAME="tcga-extract-s1"
    MACHINE_TYPE="e2-highcpu-8"  # 8 vCPU, plenty for 35 slides
    MANIFEST="tcga_stage1_manifest.json"
    GCS_PREFIX="stage1"
    ;;
  stage2)
    VM_NAME="tcga-extract-s2"
    MACHINE_TYPE="c3-highcpu-88"  # 88 vCPU for 800 slides
    MANIFEST="tcga_training_manifest.json"
    GCS_PREFIX="stage2"
    ;;
  create-only)
    VM_NAME="tcga-extract"
    MACHINE_TYPE="c3-highcpu-22"
    ;;
  delete)
    echo "Deleting VM..."
    gcloud compute instances delete tcga-extract-s1 \
      --project="$PROJECT" --zone="$ZONE" --quiet 2>/dev/null || true
    gcloud compute instances delete tcga-extract-s2 \
      --project="$PROJECT" --zone="$ZONE" --quiet 2>/dev/null || true
    echo "Done."
    exit 0
    ;;
  *)
    echo "Usage: $0 {stage1|stage2|create-only|delete}"
    exit 1
    ;;
esac

echo "=== Stage: $STAGE ==="
echo "VM: $VM_NAME ($MACHINE_TYPE) in $ZONE"
echo "Bucket: $BUCKET"
echo ""

# --- Create VM ---
echo "Creating VM..."
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --scopes=storage-rw \
  --metadata=startup-script='#!/bin/bash
    apt-get update -qq
    apt-get install -y -qq python3-pip python3-venv openslide-tools curl
    python3 -m venv /opt/extract_env
    /opt/extract_env/bin/pip install openslide-python Pillow numpy
    echo "VM ready" > /tmp/vm_ready
  '

echo "Waiting for VM to boot and install deps..."
for i in $(seq 1 60); do
  if gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" \
    --command="test -f /tmp/vm_ready && echo ready" 2>/dev/null | grep -q ready; then
    echo "VM is ready!"
    break
  fi
  echo "  waiting... ($i/60)"
  sleep 10
done

if [ "$STAGE" = "create-only" ]; then
  echo ""
  echo "VM created. SSH in with:"
  echo "  gcloud compute ssh $VM_NAME --project=$PROJECT --zone=$ZONE"
  exit 0
fi

# --- Upload manifest and extraction script ---
echo ""
echo "Uploading scripts and manifest..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

gcloud compute scp \
  "$SCRIPT_DIR/extract_tiles_tcga.py" \
  "$SCRIPT_DIR/$MANIFEST" \
  "$VM_NAME:/tmp/" \
  --project="$PROJECT" --zone="$ZONE"

# --- Run extraction ---
echo ""
echo "Starting extraction on VM..."

gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" -- bash -s <<REMOTE_SCRIPT
set -euo pipefail

export PATH="/opt/extract_env/bin:\$PATH"

echo "=== Running extraction ==="
echo "Manifest: /tmp/$MANIFEST"
echo "Output: /tmp/tiles"
echo "Bucket: $BUCKET/$GCS_PREFIX"

cd /tmp
python3 extract_tiles_tcga.py \
  --manifest "$MANIFEST" \
  --output /tmp/tiles \
  --bucket "$BUCKET/$GCS_PREFIX" \
  --quality 95

echo ""
echo "=== Extraction complete ==="
echo "Uploading results CSV..."
gsutil cp /tmp/tiles/extraction_results.csv "$BUCKET/$GCS_PREFIX/extraction_results.csv"
echo "Done!"
REMOTE_SCRIPT

# --- Download results CSV locally ---
echo ""
echo "Downloading results..."
gsutil cp "$BUCKET/$GCS_PREFIX/extraction_results.csv" "$SCRIPT_DIR/extraction_results_${STAGE}.csv"

echo ""
echo "=== STAGE $STAGE COMPLETE ==="
echo "Tiles in: $BUCKET/$GCS_PREFIX/"
echo "Results:  $SCRIPT_DIR/extraction_results_${STAGE}.csv"
echo ""
echo "To check tile counts:"
echo "  gsutil ls -r $BUCKET/$GCS_PREFIX/ | wc -l"
echo ""
echo "To delete the VM (save money!):"
echo "  gcloud compute instances delete $VM_NAME --project=$PROJECT --zone=$ZONE --quiet"
echo ""
echo "Or run: bash gcp_extract.sh delete"
