#!/bin/bash
# A/B benchmark: packed q/k/v Ulysses all-to-all vs. three separate collectives.
#
# The toggle (SGLANG_BOOGU_PACKED_QKV_A2A) is a construction-time flag on
# USPAttention, so the two variants cannot be interleaved inside one process the
# way the projection-fusion bench did -- we launch separate processes. This box
# has other tenants (a prior bench saw a +42% baseline swing across two reps), so
# we (a) INTERLEAVE baseline/packed within each rep to expose both to the same
# contention, (b) take the MIN over reps for the DenoisingStage per-step time
# (min is the least-contended sample), and (c) assert the two variants produce the
# SAME image (packed a2a is a pure data permutation -> must be bitwise identical),
# which is contention-proof.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
OUT=outputs/packed_a2a
PROMPT="A futuristic cyberpunk city at night, neon reflections on wet streets"
REPS=3

export PYTHONPATH="$PWD/python"
export FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# One generate run. $1 label, $2 CUDA_VISIBLE_DEVICES, $3 packed(0/1), rest = extra flags.
run() {
  local label="$1" gpus="$2" packed="$3"; shift 3
  SGLANG_BOOGU_PACKED_QKV_A2A="$packed" CUDA_VISIBLE_DEVICES="$gpus" \
    $PY -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate \
      --backend=sglang \
      --model-path "$MODEL" \
      --prompt "$PROMPT" \
      --width 1024 --height 1024 \
      --num-inference-steps 50 --guidance-scale 4.0 --seed 42 \
      --dit-cpu-offload false \
      --save-output --warmup \
      --output-path "$OUT/img_$label" \
      "$@" > "$OUT/$label.log" 2>&1
  # DenoisingStage per-step time isolates the DiT loop (excludes VAE/encode/load).
  awk '/average time per step/ {t=$(NF-1)} END{if(t!="")print t}' "$OUT/$label.log"
}

# For an SP degree: interleave baseline/packed for REPS reps, print per-step times.
sweep() {
  local sp="$1" gpus="$2"; shift 2
  echo "=== ulysses sp=$sp (CUDA_VISIBLE_DEVICES=$gpus) ==="
  local base_times="" pack_times=""
  for r in $(seq 1 "$REPS"); do
    local b p
    b=$(run "sp${sp}_base_r${r}" "$gpus" 0 "$@")
    p=$(run "sp${sp}_pack_r${r}" "$gpus" 1 "$@")
    echo "  rep$r: baseline=${b:-FAIL}  packed=${p:-FAIL} s/step"
    base_times="$base_times $b"; pack_times="$pack_times $p"
  done
  echo "  baseline s/step:$base_times"
  echo "  packed   s/step:$pack_times"
  # engagement sanity: packed run must log async a2a path selection if any.
  grep -c "Attention backends for transformer: fa" "$OUT/sp${sp}_pack_r1.log" \
    | sed 's/^/  transformer-fa lines(pack): /'
}

sweep 2 1,2 --num-gpus 2 --ulysses-degree 2 --ring-degree 1
sweep 4 0,1,2,3 --num-gpus 4 --ulysses-degree 4 --ring-degree 1

echo "=== min-of-reps s/step and equality (packed must == baseline bitwise) ==="
$PY - "$OUT" "$REPS" <<'PYEOF'
import glob
import re
import sys

import numpy as np
from PIL import Image

out, reps = sys.argv[1], int(sys.argv[2])


def per_step(label):
    try:
        txt = open(f"{out}/{label}.log").read()
    except FileNotFoundError:
        return None
    m = re.findall(r"average time per step:\s*([\d.]+)", txt)
    return float(m[-1]) if m else None


def image(label):
    paths = sorted(
        p
        for p in glob.glob(f"{out}/img_{label}/*")
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not paths:
        return None
    return np.asarray(Image.open(paths[-1]).convert("RGB"), dtype=np.float64)


for sp in (2, 4):
    base = [per_step(f"sp{sp}_base_r{r}") for r in range(1, reps + 1)]
    pack = [per_step(f"sp{sp}_pack_r{r}") for r in range(1, reps + 1)]
    base = [t for t in base if t]
    pack = [t for t in pack if t]
    if not base or not pack:
        print(f"  sp{sp}: MISSING timings base={base} pack={pack}")
        continue
    bmin, pmin = min(base), min(pack)
    speedup = bmin / pmin
    print(
        f"  sp{sp}: baseline_min={bmin:.4f}  packed_min={pmin:.4f}  "
        f"s/step  speedup={speedup:.4f}x  ({(speedup-1)*100:+.2f}%)"
    )
    # bitwise equality of the two variants at this sp (rep 1 images)
    b_img, p_img = image(f"sp{sp}_base_r1"), image(f"sp{sp}_pack_r1")
    if b_img is None or p_img is None:
        print(f"    equality: MISSING image(s)")
        continue
    mse = float(np.mean((b_img - p_img) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    tag = "BITWISE-EQUAL" if mse == 0 else "DIFFERS"
    print(
        f"    equality(packed vs baseline): psnr={psnr:.2f} dB "
        f"max_abs_diff={np.abs(b_img - p_img).max():.0f}  [{tag}]"
    )
PYEOF
