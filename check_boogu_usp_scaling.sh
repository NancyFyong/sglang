#!/bin/bash
# Does the replicated instruct prefix explain the sub-linear scaling?
# The prefix is a fixed 256 tokens, so quadrupling the image stream should
# quarter its share and push sp=4 efficiency up from 74%.
set -u
cd "$(dirname "$0")"
PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
OUT=outputs/usp_scaling
export PYTHONPATH="$PWD/python"
export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

run() {
  local label="$1" gpus="$2" port="$3" res="$4"; shift 4
  CUDA_VISIBLE_DEVICES="$gpus" $PY -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
    --backend=sglang --model-path "$MODEL" \
    --prompt "A futuristic cyberpunk city at night, neon reflections on wet streets" \
    --width "$res" --height "$res" \
    --num-inference-steps 20 --guidance-scale 4.0 --seed 42 \
    --dit-cpu-offload false --master-port "$port" --warmup \
    --output-path "$OUT/img_$label" "$@" > "$OUT/$label.log" 2>&1
  printf '  %-22s exit=%s  %s\n' "$label" "$?" \
    "$(grep -oE 'average time per step: [0-9.]+' "$OUT/$label.log" | tail -1)"
}

echo "=== 2048x2048 (16384 image tokens vs 4096) ==="
run res2048_sp1 1       30081 2048
run res2048_sp4 0,1,2,3 30082 2048 --num-gpus 4 --ulysses-degree 4 --ring-degree 1
