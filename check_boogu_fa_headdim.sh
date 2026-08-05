#!/bin/bash
# Boogu's DiT runs on torch_sdpa, not FlashAttention, and nobody chose that:
# FA is the default backend on CUDA, but `_resolve_flash_attention_backend_cls_str`
# downgrades to sdpa when head_size is absent from
# `FlashAttentionBackend.get_supported_head_sizes()` -- a list of multiples of 32
# inherited from vLLM's paged-attention constraint. Boogu's head_dim is
# 3360 / 28 = 120, so the log says "Cannot use FlashAttention backend for head
# size 120." The text encoder, at a multiple of 32, gets `fa`.
#
# Both kernels accept every multiple of 8 up to 256 (swept: fa_ver 3 on H20 and
# flash_attn 2.8.3, 32/32 sizes pass), so the list was simply stale. This measures
# what widening it buys, and what it costs in output equality.
#
# Interleaved sdpa/fa passes, min-of-N: this host has another tenant at ~80% on
# every GPU, so a single pass can absorb a contention burst.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
REF=outputs/boogu_cfg_check/img_sp1_r1
OUT=outputs/boogu_fa
PROMPT="A futuristic cyberpunk city at night, neon reflections on wet streets"

export PYTHONPATH="$PWD/python"
export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1
mkdir -p "$OUT"

run() {
  local label="$1" port="$2"; shift 2
  CUDA_VISIBLE_DEVICES=3 $PY -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
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
  local t backend
  t=$(grep -oE "average time per step: [0-9.]+" "$OUT/$label.log" | tail -1 | grep -oE "[0-9.]+$")
  backend=$(grep -oE "Attention backends for transformer: .*" "$OUT/$label.log" | tail -1 | sed 's/.*: //')
  printf "  %-14s %-9s s/step   transformer=%s\n" "$label" "${t:-FAILED}" "${backend:-?}"
}

# Pass 1 and 2 interleaved so a slow stretch cannot land on one config only.
echo "=== pass 1 ==="
run sdpa_p1 30291 --attention-backend torch_sdpa
run fa_p1   30292 --attention-backend fa
echo "=== pass 2 ==="
run sdpa_p2 30293 --attention-backend torch_sdpa
run fa_p2   30294 --attention-backend fa
echo "=== fa + torch.compile (the two levers together) ==="
run fa_compile 30295 --attention-backend fa --enable-torch-compile

echo
echo "=== vs stored single-GPU sdpa reference ==="
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

for label in ("sdpa_p1", "fa_p1", "sdpa_p2", "fa_p2", "fa_compile"):
    o = load(out / f"img_{label}")
    if o is None:
        print(f"  {label:14s} MISSING")
        continue
    mse = float(np.mean((ref - o) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    verdict = "BITWISE-EXACT" if mse == 0 else "differs"
    print(f"  {label:14s} psnr={psnr:>8.2f} dB  max|diff|={np.abs(ref-o).max():>4.0f}  {verdict}")
PYEOF
