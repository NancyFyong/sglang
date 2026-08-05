#!/bin/bash
# Is --enable-torch-compile a free win for Boogu, and is it bitwise-exact?
#
# The last untested lossless lever. Base commit a8a0d23431 made the DiT
# torch.compile-compatible (only the mask/rope index builder is
# @torch.compiler.disable'd, which is data-dependent Python and has to be), but no
# benchmark has ever been run with it on. Attention is only ~13% of a Boogu step,
# so the model is GEMM/elementwise-bound -- exactly what compile's fusion targets.
#
# Compared against the same stored sp1 reference as check_boogu_cfg_exact.sh.
# Runs on GPU 3: the user is benchmarking torch.compile on GPU 1 from the main
# checkout, and GPUs 0/2 carry the usual ~80% background load.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
REF=outputs/boogu_cfg_check/img_sp1_r1
OUT=outputs/boogu_compile
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
  printf "  %-14s %s s/step\n" "$label" "${t:-FAILED}"
}

echo "=== runs (warmup pays the compile, so it is excluded from s/step) ==="
run eager   3 30281
run compile 3 30282 --enable-torch-compile

echo
echo "=== vs stored sp1 reference ==="
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

for label in ("eager", "compile"):
    o = load(out / f"img_{label}")
    if o is None:
        print(f"  {label:14s} MISSING")
        continue
    mse = float(np.mean((ref - o) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    verdict = "BITWISE-EXACT" if mse == 0 else "differs"
    print(f"  {label:14s} psnr={psnr:>8.2f} dB  max|diff|={np.abs(ref-o).max():>4.0f}  {verdict}")
PYEOF
