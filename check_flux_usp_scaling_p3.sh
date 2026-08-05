#!/bin/bash
# Pass 3: the first two passes showed this host is too noisy for single
# measurements -- an identical sp2 config timed 0.2324 then 0.1475 s/step, and
# the first run of every batch is reproducibly slow (0.5865 twice) because the
# GPUs ramp clocks. Another tenant also holds all four GPUs at ~80% util.
#
# So: one throwaway run to spin the clocks up, then interleave the four configs
# three times and report min-of-3 per config. Contention and clock ramp only
# ever *add* time, so the minimum is the right estimator; interleaving keeps a
# slow stretch from landing entirely on one config.
set -u
cd "$(dirname "$0")"

PY=/group/40173/zionyfeng/uv_venv/sglang-boogu/bin/python
MODEL=/group/40173/zionyfeng/models/black-forest-labs/FLUX.2-klein-4B
OUT=outputs/flux_usp_p3
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
    --num-inference-steps 50 --seed 42 \
    --dit-cpu-offload false \
    --master-port "$port" \
    --save-output --warmup \
    --output-path "$OUT/img_$label" \
    "$@" > "$OUT/$label.log" 2>&1
  local t
  t=$(grep -oE "average time per step: [0-9.]+" "$OUT/$label.log" | tail -1 | grep -oE "[0-9.]+$")
  printf "  %-24s %s s/step\n" "$label" "${t:-FAILED}"
}

sp1()  { run "$1" 0       "$2"; }
sp2()  { run "$1" 0,1     "$2" --num-gpus 2 --ulysses-degree 2 --ring-degree 1; }
sp4()  { run "$1" 0,1,2,3 "$2" --num-gpus 4 --ulysses-degree 4 --ring-degree 1; }
sp4r() { SGLANG_SP_TEXT_SHARD_MIN=99999 run "$1" 0,1,2,3 "$2" --num-gpus 4 --ulysses-degree 4 --ring-degree 1; }

echo "=== throwaway clock-ramp run (discarded) ==="
sp1 discard 30110

for rep in 1 2 3; do
  echo "=== repeat $rep ==="
  sp1  "sp1_r$rep"  "3011$rep"
  sp2  "sp2_r$rep"  "3012$rep"
  sp4  "sp4_r$rep"  "3013$rep"
  sp4r "sp4rep_r$rep" "3014$rep"
done

echo "=== min-of-3 per config ==="
$PY - "$OUT" <<'PYEOF'
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
pat = re.compile(r"average time per step: ([0-9.]+)")
res = {}
for cfg in ("sp1", "sp2", "sp4", "sp4rep"):
    vals = []
    for rep in (1, 2, 3):
        log = out / f"{cfg}_r{rep}.log"
        if not log.exists():
            continue
        m = pat.findall(log.read_text(errors="ignore"))
        if m:
            vals.append(float(m[-1]))
    if vals:
        res[cfg] = vals
        print(f"  {cfg:8s} runs={[f'{v:.4f}' for v in vals]} min={min(vals):.4f}")

if "sp1" in res:
    base = min(res["sp1"])
    print()
    for cfg in ("sp2", "sp4", "sp4rep"):
        if cfg in res:
            t = min(res[cfg])
            n = 2 if cfg == "sp2" else 4
            print(f"  {cfg:8s} {base/t:.2f}x vs sp1   ({100*base/t/n:.0f}% of linear)")
    if "sp4" in res and "sp4rep" in res:
        print(
            f"\n  prefix tax at sp4: replicated/sharded = "
            f"{min(res['sp4rep'])/min(res['sp4']):.3f}x"
        )
PYEOF
