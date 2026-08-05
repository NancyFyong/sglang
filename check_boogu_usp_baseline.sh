#!/bin/bash
# Fills in the two runs `check_boogu_usp.sh` could not produce: the sp=1
# baseline (its default master port was already bound) and a second sp=2 run,
# because the first one timed 3.6x slower than sp=4 -- a 2x degree change cannot
# do that, so one of the two numbers came from a contended GPU.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
OUT=outputs/usp_check
PROMPT="A futuristic cyberpunk city at night, neon reflections on wet streets"

export PYTHONPATH="$PWD/python"
export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

run() {
  local label="$1" gpus="$2" port="$3"; shift 3
  echo "=== $label  (CUDA_VISIBLE_DEVICES=$gpus  master_port=$port) ==="
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
  local rc=$?
  grep -E "warmup excluded|average time per step" "$OUT/$label.log" | tail -2 | sed 's/^/  /'
  echo "  exit=$rc"
}

run baseline_sp1 1 30071
run ulysses_sp2b 1,2 30072 --num-gpus 2 --ulysses-degree 2 --ring-degree 1

echo "=== PSNR vs the sp=1 baseline ==="
$PY - "$OUT" <<'PYEOF'
import glob
import sys

import numpy as np
from PIL import Image


def load(label):
    paths = sorted(glob.glob(f"{sys.argv[1]}/img_{label}/**", recursive=True))
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not paths:
        return None
    print(f"  {label}: {paths[0].split('/')[-1]}")
    return np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.float64)


baseline = load("baseline_sp1")
if baseline is None:
    raise SystemExit("no baseline image")
for label in ("ulysses_sp2", "ulysses_sp2b", "ulysses_sp4"):
    other = load(label)
    if other is None:
        print(f"  {label}: MISSING")
        continue
    mse = float(np.mean((baseline - other) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    print(
        f"  {label}: psnr={psnr:.2f} dB  "
        f"max_abs_diff={np.abs(baseline - other).max():.0f}"
    )
PYEOF
