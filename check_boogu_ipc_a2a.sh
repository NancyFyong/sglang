#!/bin/bash
# The IPC all-to-all fast path was disabled in every benchmark so far -- by this
# harness, not by the model. Every script exported
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, and CUDA IPC tensor sharing
# under expandable segments needs the pidfd_getfd syscall (Linux >= 5.6). This
# host runs 5.4, so IPC_A2A.init died in _share and usp.py silently fell back to
# NCCL for the 2-rank ulysses all-to-all.
#
# Two predictions to separate the two causes:
#   sp2      devices 0,1 -> dropping expandable_segments should let IPC engage.
#   cfg2sp2  the second sp group sits on devices 2,3, and ipc_a2a.py:140 computes
#            the peer as `1 - dev` -> -1 -> "Invalid peer device id". So this one
#            should STILL fall back even with the allocator fixed.
# If both hold, the remaining win needs the peer computation fixed too.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
REF=outputs/boogu_cfg_check/img_sp1_r1
OUT=outputs/boogu_ipc
PROMPT="A futuristic cyberpunk city at night, neon reflections on wet streets"

export PYTHONPATH="$PWD/python"
export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT"

run() {
  local label="$1" gpus="$2" port="$3" alloc="$4"; shift 4
  PYTORCH_CUDA_ALLOC_CONF="$alloc" \
  CUDA_VISIBLE_DEVICES="$gpus" $PY -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --backend=sglang \
    --model-path "$MODEL" \
    --prompt "$PROMPT" \
    --width 1024 --height 1024 \
    --num-inference-steps 50 --guidance-scale 4.0 --seed 42 \
    --dit-cpu-offload false \
    --master-port "$port" \
    --save-output --warmup \
    --output-path "$OUT/img_$label" \
    "$@" > "$OUT/$label.log" 2>&1
  local t ipc
  t=$(grep -oE "average time per step: [0-9.]+" "$OUT/$label.log" | tail -1 | grep -oE "[0-9.]+$")
  if grep -q "IPC all-to-all init failed\|IPC all-to-all unavailable" "$OUT/$label.log"; then
    ipc="NCCL fallback"
  else
    ipc="IPC engaged"
  fi
  printf "  %-22s %-9s s/step   [%s]\n" "$label" "${t:-FAILED}" "$ipc"
}

echo "=== sp2 (peer computation is valid: CUDA_VISIBLE_DEVICES maps these to 0,1) ==="
run sp2_expandable 0,2 30271 expandable_segments:True  --num-gpus 2 --ulysses-degree 2 --ring-degree 1
run sp2_ipc        0,2 30272 ""                        --num-gpus 2 --ulysses-degree 2 --ring-degree 1

# The 4-GPU cfg2 x sp2 leg is deliberately left out: GPU 1 is carrying the user's
# own torch.compile benchmark and GPU 3 this script's sibling, so a 4-GPU timing
# here would measure contention, not the transport. The prediction to test later is
# that it still falls back, because its second sp group lands on devices 2,3 and
# ipc_a2a.py:140 computes that peer as 1 - dev = -1.

echo
echo "=== still bitwise-exact vs stored sp1 reference? ==="
$PY - "$REF" "$OUT" <<'PYEOF'
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ref_dir, out = sys.argv[1], Path(sys.argv[2])


def load(d):
    p = [x for x in sorted(glob.glob(f"{d}/**", recursive=True))
         if x.lower().endswith((".png", ".jpg", ".jpeg"))]
    return np.asarray(Image.open(p[0]).convert("RGB"), np.float64) if p else None


ref = load(ref_dir)
if ref is None:
    sys.exit(f"missing reference image in {ref_dir}")

for label in ("sp2_expandable", "sp2_ipc", "cfg2sp2_ipc"):
    o = load(out / f"img_{label}")
    if o is None:
        print(f"  {label:22s} MISSING")
        continue
    mse = float(np.mean((ref - o) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    verdict = "BITWISE-EXACT" if mse == 0 else "differs"
    print(f"  {label:22s} psnr={psnr:>8.2f} dB  max|diff|={np.abs(ref-o).max():>4.0f}  {verdict}")
PYEOF
