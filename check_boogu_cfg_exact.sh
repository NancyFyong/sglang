#!/bin/bash
# Does exact_parallel_combine make CFG parallel bitwise-identical to 1 GPU?
#
# Before the change, --cfg-parallel-size 2 scored 24.42 dB / max|diff|=232 against
# sp1 -- not a bug, but the deliberately re-associated combine (cfg_policy.py:86)
# kept for WAN's bf16 baseline: 4*p - 3*n instead of n + 4*(p-n), which is ~15x
# noisier per step at cfg_scale=4 and compounds over 50 sampler steps.
#
# sp1_ctl re-runs the single-GPU reference to confirm the pipeline is deterministic
# run-to-run, so comparing the new CFG-parallel images against the *stored* sp1
# reference from the previous session is valid.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
REF=outputs/boogu_cfg_check/img_sp1_r1
OUT=outputs/boogu_cfg_exact
PROMPT="A futuristic cyberpunk city at night, neon reflections on wet streets"

export PYTHONPATH="$PWD/python"
export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

run() {
  local label="$1" gpus="$2" port="$3"; shift 3
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
  local t
  t=$(grep -oE "average time per step: [0-9.]+" "$OUT/$label.log" | tail -1 | grep -oE "[0-9.]+$")
  printf "  %-12s %s s/step\n" "$label" "${t:-FAILED}"
}

# GPU 0 is carrying another tenant's 42 GB / 100% util, so keep the 1- and 2-GPU
# configs off it; only the 4-GPU config has to touch it.
echo "=== runs ==="
run sp1_ctl    1       30251
run cfg2_fix   1,2     30252 --num-gpus 2 --cfg-parallel-size 2
run cfg2sp2_fix 0,1,2,3 30253 --num-gpus 4 --cfg-parallel-size 2 \
                               --ulysses-degree 2 --ring-degree 1

echo
echo "=== bitwise comparison vs stored sp1 reference ==="
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

for label in ("sp1_ctl", "cfg2_fix", "cfg2sp2_fix"):
    o = load(out / f"img_{label}")
    if o is None:
        print(f"  {label:12s} MISSING")
        continue
    mse = float(np.mean((ref - o) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    verdict = "BITWISE-EXACT" if mse == 0 else "differs"
    print(f"  {label:12s} psnr={psnr:>8.2f} dB  max|diff|={np.abs(ref-o).max():>4.0f}  {verdict}")
PYEOF
