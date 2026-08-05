#!/bin/bash
# Escape-hatch verification for Boogu-Image text sharding.
#
# The default sp2/sp4 runs now take the sp_shard_utils text-shard path, which is
# lossy (a valid-but-different sample; Boogu saturates at ~21-25 dB for any bf16
# reorder, so PSNR alone cannot tell "correct sharding" from "bug"). This isolates
# the two: forcing the replicated-prefix path with SGLANG_SP_TEXT_SHARD_MIN=99999
# must reproduce the sp=1 baseline BITWISE (psnr=inf, max_abs_diff=0). If it does,
# the text-shard divergence is attributable solely to sharding and the merge did
# not break the exact path.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
OUT=outputs/usp_escape
PROMPT="A futuristic cyberpunk city at night, neon reflections on wet streets"

export PYTHONPATH="$PWD/python"
export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

run() {
  local label="$1" gpus="$2"; shift 2
  echo "=== $label  (CUDA_VISIBLE_DEVICES=$gpus  SGLANG_SP_TEXT_SHARD_MIN=${SGLANG_SP_TEXT_SHARD_MIN:-0}) ==="
  rm -rf "$OUT/img_$label"
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
  echo "  exit=$?"
}

SGLANG_SP_TEXT_SHARD_MIN=0     run baseline_sp1 1
export SGLANG_SP_TEXT_SHARD_MIN=99999
run replicate_sp2 1,2 --num-gpus 2 --ulysses-degree 2 --ring-degree 1
run replicate_sp4 0,1,2,3 --num-gpus 4 --ulysses-degree 4 --ring-degree 1
unset SGLANG_SP_TEXT_SHARD_MIN

echo "=== PSNR vs the sp=1 baseline (forced-replicate must be bitwise) ==="
$PY - "$OUT" <<'PYEOF'
import glob
import sys

import numpy as np
from PIL import Image


def load(label):
    # each run writes into a per-label directory; take the newest png in it
    paths = sorted(glob.glob(f"{sys.argv[1]}/img_{label}/*.png"))
    if not paths:
        return None
    print(f"  {label}: {paths[-1].split('/')[-1]}")
    return np.asarray(Image.open(paths[-1]).convert("RGB"), dtype=np.float64)


baseline = load("baseline_sp1")
if baseline is None:
    raise SystemExit("no baseline image")
for label in ("replicate_sp2", "replicate_sp4"):
    other = load(label)
    if other is None:
        print(f"  {label}: MISSING")
        continue
    mse = float(np.mean((baseline - other) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    verdict = "BITWISE" if mse == 0 else "NOT bitwise"
    print(
        f"  {label}: psnr={psnr:.2f} dB  max_abs_diff="
        f"{np.abs(baseline - other).max():.0f}  -> {verdict}"
    )
PYEOF
