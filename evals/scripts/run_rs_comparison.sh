#!/bin/bash
set -e
cd /Users/andrewluetgers/projects/dev/RRIP

BIN=server/target2/release/origami
IMG=evals/test-images/L0-1024.jpg

# --- Uniform quality (resq=40) ---

echo "2. Rs 444 j40"
$BIN encode --image $IMG --out evals/runs/rs_debug_j40_pac \
  --resq 40 --subsamp 444 --manifest --pack --debug-images 2>&1 | tail -1

echo "3. Rs 420 j40"
$BIN encode --image $IMG --out evals/runs/rs_debug_j40_s420_pac \
  --resq 40 --subsamp 420 --manifest --pack --debug-images 2>&1 | tail -1

echo "4. Rs 420opt j40"
$BIN encode --image $IMG --out evals/runs/rs_debug_j40_s420opt_pac \
  --resq 40 --subsamp 420opt --manifest --pack --debug-images 2>&1 | tail -1

echo "5. Rs 444+optL2 j40"
$BIN encode --image $IMG --out evals/runs/rs_debug_j40_optl2_pac \
  --resq 40 --subsamp 444 --optl2 --manifest --pack --debug-images 2>&1 | tail -1

echo "6. Rs 420+optL2 j40"
$BIN encode --image $IMG --out evals/runs/rs_debug_j40_s420_optl2_pac \
  --resq 40 --subsamp 420 --optl2 --manifest --pack --debug-images 2>&1 | tail -1

echo "7. Rs 420opt+optL2 j40"
$BIN encode --image $IMG --out evals/runs/rs_debug_j40_s420opt_optl2_pac \
  --resq 40 --subsamp 420opt --optl2 --manifest --pack --debug-images 2>&1 | tail -1

# --- Split quality (l1q=60, l0q=40) ---

echo "8. Py split l1q60 l0q40"
uv run python evals/scripts/wsi_residual_debug_with_manifest.py \
  --image $IMG --l1q 60 --l0q 40 --pac \
  --out evals/runs/py_debug_l1q60_l0q40_pac 2>&1 | tail -1

echo "9. Rs 444 split l1q60 l0q40"
$BIN encode --image $IMG --out evals/runs/rs_debug_l1q60_l0q40_pac \
  --l1q 60 --l0q 40 --subsamp 444 --manifest --pack --debug-images 2>&1 | tail -1

echo "10. Rs 420 split l1q60 l0q40"
$BIN encode --image $IMG --out evals/runs/rs_debug_l1q60_l0q40_s420_pac \
  --l1q 60 --l0q 40 --subsamp 420 --manifest --pack --debug-images 2>&1 | tail -1

echo "11. Rs 420opt split l1q60 l0q40"
$BIN encode --image $IMG --out evals/runs/rs_debug_l1q60_l0q40_s420opt_pac \
  --l1q 60 --l0q 40 --subsamp 420opt --manifest --pack --debug-images 2>&1 | tail -1

echo "12. Rs 444+optL2 split l1q60 l0q40"
$BIN encode --image $IMG --out evals/runs/rs_debug_l1q60_l0q40_optl2_pac \
  --l1q 60 --l0q 40 --subsamp 444 --optl2 --manifest --pack --debug-images 2>&1 | tail -1

echo "13. Rs 420+optL2 split l1q60 l0q40"
$BIN encode --image $IMG --out evals/runs/rs_debug_l1q60_l0q40_s420_optl2_pac \
  --l1q 60 --l0q 40 --subsamp 420 --optl2 --manifest --pack --debug-images 2>&1 | tail -1

echo "14. Rs 420opt+optL2 split l1q60 l0q40"
$BIN encode --image $IMG --out evals/runs/rs_debug_l1q60_l0q40_s420opt_optl2_pac \
  --l1q 60 --l0q 40 --subsamp 420opt --optl2 --manifest --pack --debug-images 2>&1 | tail -1

echo "=== ALL DONE ==="
