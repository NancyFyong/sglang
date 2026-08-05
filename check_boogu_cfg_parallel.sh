#!/bin/bash
# Which 4-GPU layout is actually fastest for Boogu -- and is CFG parallel a free
# lossless win we left on the table?
#
# Boogu's pipeline config sets should_use_guidance=True, so every step runs two
# forwards (cond + uncond). CFG parallel puts them on separate ranks: exactly
# parallel, no shared math, so it should be near-2x AND bitwise-exact. Crucially
# server_args auto-enables it only when no sp/ulysses/ring flag is given -- every
# benchmark so far passed --ulysses-degree and therefore silently disabled it.
#
# So sp4 (measured 2.96x) may not be the best 4-GPU layout: cfg2 x sp2 should be
# ~2 x 1.77 = 3.5x. Same discard-first + min-of-2 protocol as the FLUX runs.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/Boogu/Boogu-Image-0.1-Base
OUT=outputs/boogu_cfg_check
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
  local t cfg
  t=$(grep -oE "average time per step: [0-9.]+" "$OUT/$label.log" | tail -1 | grep -oE "[0-9.]+$")
  cfg=$(grep -oE '"enable_cfg_parallel": [a-z]+' "$OUT/$label.log" | head -1)
  printf "  %-14s %-10s s/step   [%s]\n" "$label" "${t:-FAILED}" "$cfg"
}

sp1()      { run "$1" 0       "$2"; }
cfg2()     { run "$1" 0,1     "$2" --num-gpus 2 --cfg-parallel-size 2; }
sp4()      { run "$1" 0,1,2,3 "$2" --num-gpus 4 --ulysses-degree 4 --ring-degree 1; }
cfg2_sp2() { run "$1" 0,1,2,3 "$2" --num-gpus 4 --cfg-parallel-size 2 \
                                   --ulysses-degree 2 --ring-degree 1; }

echo "=== throwaway clock-ramp run (discarded) ==="
sp1 discard 30150

for rep in 1 2; do
  echo "=== repeat $rep ==="
  sp1      "sp1_r$rep"      "3015$rep"
  cfg2     "cfg2_r$rep"     "3016$rep"
  sp4      "sp4_r$rep"      "3017$rep"
  cfg2_sp2 "cfg2sp2_r$rep"  "3018$rep"
done

echo "=== min-of-2, and GPUs used ==="
$PY - "$OUT" <<'PYEOF'
import glob
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

out = Path(sys.argv[1])
pat = re.compile(r"average time per step: ([0-9.]+)")
gpus = {"sp1": 1, "cfg2": 2, "sp4": 4, "cfg2sp2": 4}
res = {}
for cfg in gpus:
    vals = []
    for rep in (1, 2):
        log = out / f"{cfg}_r{rep}.log"
        if log.exists():
            m = pat.findall(log.read_text(errors="ignore"))
            if m:
                vals.append(float(m[-1]))
    if vals:
        res[cfg] = min(vals)
        print(f"  {cfg:9s} {[f'{v:.4f}' for v in vals]} min={min(vals):.4f}")

if "sp1" in res:
    base = res["sp1"]
    print()
    for cfg, n in gpus.items():
        if cfg in res and cfg != "sp1":
            print(f"  {cfg:9s} {n} GPU  {base/res[cfg]:.2f}x   "
                  f"({100*base/res[cfg]/n:.0f}% of linear)")


def load(label):
    paths = sorted(glob.glob(f"{out}/img_{label}_r1/**", recursive=True))
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    return np.asarray(Image.open(paths[0]).convert("RGB"), np.float64) if paths else None


print("\n=== PSNR vs sp1 (is CFG parallel bitwise-exact too?) ===")
ref = load("sp1")
if ref is not None:
    for cfg in ("cfg2", "sp4", "cfg2sp2"):
        o = load(cfg)
        if o is None:
            print(f"  {cfg:9s} MISSING")
            continue
        mse = float(np.mean((ref - o) ** 2))
        psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
        print(f"  {cfg:9s} psnr={psnr:.2f} dB  max|diff|={np.abs(ref - o).max():.0f}")
PYEOF
