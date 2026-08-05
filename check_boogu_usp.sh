#!/bin/bash
# End-to-end check for Boogu-Image Ulysses sequence parallelism.
#
# Same prompt / seed / steps at sp=1, sp=2 and sp=4. Sequence parallelism is a
# pure work split, so the images must match the single-GPU baseline to within
# bf16 reduction noise -- a PSNR in the 40s, not the 20s that a real numerical
# divergence produces. Wall clock comes from the "warmup excluded" line so model
# load and the first-call compile are not counted.
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
  local label="$1" gpus="$2"; shift 2
  echo "=== $label  (CUDA_VISIBLE_DEVICES=$gpus) ==="
  CUDA_VISIBLE_DEVICES="$gpus" $PY -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --backend=sglang \
    --model-path "$MODEL" \
    --prompt "$PROMPT" \
    --width 1024 --height 1024 \
    --num-inference-steps 50 --guidance-scale 4.0 --seed 42 \
    --dit-cpu-offload false \
    --save-output --warmup \
    --output-path "$OUT/img_$label" \
    "$@" > "$OUT/$label.log" 2>&1
  local rc=$?
  grep -E "warmup excluded|enable_sequence_shard|Ulysses|sequence parallelism" "$OUT/$label.log" \
    | tail -4 | sed 's/^/  /'
  echo "  exit=$rc"
  [ "$rc" = 0 ]
}

run baseline_sp1 1
run ulysses_sp2 1,2 --num-gpus 2 --ulysses-degree 2 --ring-degree 1
run ulysses_sp4 0,1,2,3 --num-gpus 4 --ulysses-degree 4 --ring-degree 1

echo "=== ring guard must refuse (ulysses 2 x ring 2) ==="
run ring_rejected 0,1,2,3 --num-gpus 4 --ulysses-degree 2 --ring-degree 2
grep -c "Ulysses-only" "$OUT/ring_rejected.log" | sed 's/^/  matches: /'

echo "=== PSNR vs the sp=1 baseline ==="
$PY - "$OUT" <<'PYEOF'
import glob
import sys

import numpy as np
from PIL import Image


def load(label):
    # --output-path is a directory; the png lands inside it. Glob into the dir
    # and take the newest, so stale images from an earlier run are ignored.
    paths = sorted(
        p
        for p in glob.glob(f"{sys.argv[1]}/img_{label}/*")
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not paths:
        return None
    print(f"  {label}: {paths[-1]}")
    return np.asarray(Image.open(paths[-1]).convert("RGB"), dtype=np.float64)


baseline = load("baseline_sp1")
if baseline is None:
    raise SystemExit("no baseline image")
for label in ("ulysses_sp2", "ulysses_sp4"):
    other = load(label)
    if other is None:
        print(f"  {label}: MISSING")
        continue
    mse = float(np.mean((baseline - other) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    print(f"  {label}: psnr={psnr:.2f} dB  max_abs_diff={np.abs(baseline - other).max():.0f}")
PYEOF
