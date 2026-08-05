#!/bin/bash
# FLUX.2-klein-4B USP scaling, same protocol as the Boogu runs (1024x1024,
# 50 steps, seed 42, warmup excluded) so the two curves are comparable.
#
# The point of the extra `sp4_replicate` run: FLUX shards its text stream when
# sp>1 (should_shard_text -> num_replicated_prefix=0), while Boogu replicates
# it. Forcing SGLANG_SP_TEXT_SHARD_MIN above the 512-token text length flips
# FLUX onto Boogu's replicated-prefix route, which measures the prefix tax
# directly instead of inferring it from an Amdahl fit.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/black-forest-labs/FLUX.2-klein-4B
OUT=outputs/flux_usp
PROMPT="A futuristic cyberpunk city at night, neon reflections on wet streets"

export PYTHONPATH="$PWD/python"
export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

run() {
  local label="$1" gpus="$2" port="$3"; shift 3
  echo "=== $label  (CUDA_VISIBLE_DEVICES=$gpus  master_port=$port  TEXT_SHARD_MIN=${SGLANG_SP_TEXT_SHARD_MIN:-0}) ==="
  CUDA_VISIBLE_DEVICES="$gpus" $PY -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --backend=sglang \
    --model-path "$MODEL" \
    --prompt "$PROMPT" \
    --width 1024 --height 1024 \
    --num-inference-steps 50 --seed 42 \
    --dit-cpu-offload false \
    --master-port "$port" \
    --save-output --warmup \
    --output-path "$OUT/img_$label" \
    "$@" > "$OUT/$label.log" 2>&1
  local rc=$?
  grep -E "warmup excluded|average time per step" "$OUT/$label.log" | tail -2 | sed 's/^/  /'
  echo "  exit=$rc"
}

run flux2_sp1 0 30101

run flux2_sp2  0,1 30103 --num-gpus 2 --ulysses-degree 2 --ring-degree 1
run flux2_sp4  0,1,2,3 30104 --num-gpus 4 --ulysses-degree 4 --ring-degree 1

# Same sp=4 geometry, but the 512-token text stream stays replicated on every
# rank -- i.e. exactly what Boogu does today.
SGLANG_SP_TEXT_SHARD_MIN=99999 \
  run flux2_sp4_replicate 0,1,2,3 30105 --num-gpus 4 --ulysses-degree 4 --ring-degree 1

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


baseline = load("flux2_sp1")
if baseline is None:
    raise SystemExit("no baseline image")
for label in ("flux2_sp2", "flux2_sp4", "flux2_sp4_replicate"):
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
