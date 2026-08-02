"""Calibrate Boogu-Image TeaCache polynomial coefficients.

TeaCache decides whether to reuse a cached residual by accumulating a *rescaled*
relative-L1 distance of the modulated input (`temb`) and comparing it against a
threshold. The rescaling polynomial is what turns the cheap `temb` signal into a
usable proxy for the expensive quantity we actually care about: how much the
stream-layer residual changed since the last computed step.

This script profiles both quantities over full uncached denoising runs and fits
`np.polyfit(x, y, 4)`, the procedure the TeaCache authors describe:

    x_i = ||temb_i     - temb_{i-1}||_1     / ||temb_{i-1}||_1
    y_i = ||residual_i - residual_{i-1}||_1 / ||residual_{i-1}||_1

Skipping is forced off during profiling (`start_skipping` is pushed past the end
of the schedule), so every step is a real forward pass and every (x, y) pair is
measured against its true predecessor.

The pipeline is built in-process via `GPUWorker` rather than through
`DiffGenerator`, because the latter spawns the model into a subprocess where the
probe's monkeypatches would not apply.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m \
        sglang.multimodal_gen.tools.calibrate_boogu_teacache \
        --model-path <path/to/Boogu-Image> --steps 50

The printed `coefficients` and a threshold from the printed skip-rate table go
into `BooguImageSamplingParams.teacache_params`.
"""

import argparse
import json
import socket

import numpy as np
import torch

DEFAULT_PROMPTS = [
    "A photorealistic portrait of an elderly fisherman mending a net at dawn, "
    "soft rim light, shallow depth of field",
    "An intricate art-nouveau poster of a mechanical hummingbird, gold foil, "
    "flat vector shapes, high contrast",
    "A wide landscape of terraced rice fields in heavy fog, muted palette, "
    "fine mist detail, cinematic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--dump-path", default="boogu_teacache_calibration.json")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Repeatable. Defaults to a small spread of subject/style/detail prompts.",
    )
    return parser.parse_args()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Relative L1 distance, the same reduction TeaCache uses at inference."""
    diff = (current - previous).abs().mean()
    base = previous.abs().mean()
    return (diff / base).float().cpu().item()


class StreamResidualProbe:
    """Records paired (temb, residual) relative-L1 distances per CFG branch.

    Wraps the two TeaCache hooks on the live transformer instead of duplicating
    the interleave/residual math, so what gets profiled is exactly what the model
    caches at inference time.
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self.samples: list[dict[str, float]] = []
        self.run_index = -1
        self._prev_temb: dict[bool, torch.Tensor] = {}
        self._prev_residual: dict[bool, torch.Tensor] = {}
        self._pending_x: dict[bool, float] = {}
        self._orig_should_skip = transformer.should_skip_forward_for_cached_states
        self._orig_maybe_cache = transformer.maybe_cache_states

    def install(self) -> None:
        self.transformer.should_skip_forward_for_cached_states = self._should_skip
        self.transformer.maybe_cache_states = self._maybe_cache

    def remove(self) -> None:
        self.transformer.should_skip_forward_for_cached_states = self._orig_should_skip
        self.transformer.maybe_cache_states = self._orig_maybe_cache

    def start_run(self) -> None:
        """Drop the cross-step baselines before a new prompt.

        Instruction lengths differ per prompt, so the interleaved sequence -- and
        therefore the residual -- changes shape between runs. Only within one run
        is a step-to-step comparison meaningful.
        """
        self.run_index += 1
        self._prev_temb.clear()
        self._prev_residual.clear()
        self._pending_x.clear()

    def _should_skip(self, **kwargs) -> bool:
        should_skip = self._orig_should_skip(**kwargs)
        if should_skip:
            raise AssertionError(
                "profiling run skipped a step; push start_skipping past the end "
                "of the schedule so every measured pair has a true predecessor"
            )
        branch = self.transformer.is_cfg_negative
        temb = kwargs["temb"].detach().float()
        prev_temb = self._prev_temb.get(branch)
        if prev_temb is not None:
            self._pending_x[branch] = _rel_l1(temb, prev_temb)
        self._prev_temb[branch] = temb
        return should_skip

    def _maybe_cache(self, hidden_states, original_hidden_states) -> None:
        self._orig_maybe_cache(hidden_states, original_hidden_states)
        branch = self.transformer.is_cfg_negative
        residual = (hidden_states - original_hidden_states).detach().float()
        prev_residual = self._prev_residual.get(branch)
        pending_x = self._pending_x.pop(branch, None)
        if prev_residual is not None and pending_x is not None:
            self.samples.append(
                {
                    "run": self.run_index,
                    "step": self.transformer.cnt,
                    "is_cfg_negative": branch,
                    "x_temb_rel_l1": pending_x,
                    "y_residual_rel_l1": _rel_l1(residual, prev_residual),
                }
            )
        self._prev_residual[branch] = residual


def build_worker(model_path: str):
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        set_global_server_args,
    )

    server_args = ServerArgs.from_kwargs(
        model_path=model_path,
        num_gpus=1,
        dit_cpu_offload=False,
        warmup_mode="off",
    )
    set_global_server_args(server_args)
    worker = GPUWorker(
        local_rank=0, rank=0, master_port=_free_port(), server_args=server_args
    )
    return worker, server_args


def run_prompt(worker, server_args, prompt: str, args: argparse.Namespace, seed: int):
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    sampling_params = SamplingParams.from_user_sampling_params_args(
        server_args.model_path,
        server_args=server_args,
        prompt=prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        enable_teacache=True,
        seed=seed,
        save_output=False,
    )
    sampling_params.teacache_params.start_skipping = 10 * args.steps
    req = prepare_request(server_args=server_args, sampling_params=sampling_params)
    worker.execute_forward([req])


def simulate_thresholds(samples, coefficients, thresholds) -> list[dict[str, float]]:
    """Replay the accumulator over the profiled runs to price each threshold.

    The profiled `x` sequence is what inference would see (`temb` depends only on
    the timestep), so accumulating `poly(x)` per run reproduces the real skip
    pattern for a given threshold -- without paying for a generation per
    candidate.
    """
    rescale = np.poly1d(coefficients)
    runs: dict[tuple, list[float]] = {}
    for s in samples:
        runs.setdefault((s["run"], s["is_cfg_negative"]), []).append(s["x_temb_rel_l1"])

    report = []
    for threshold in thresholds:
        skipped = total = 0
        for xs in runs.values():
            accumulated = 0.0
            for x in xs:
                total += 1
                accumulated += rescale(x)
                if accumulated >= threshold:
                    accumulated = 0.0
                else:
                    skipped += 1
        report.append({"threshold": threshold, "skip_ratio": skipped / total})
    return report


def fit_and_report(samples, args, num_prompts: int) -> None:
    x = np.array([s["x_temb_rel_l1"] for s in samples])
    y = np.array([s["y_residual_rel_l1"] for s in samples])
    coefficients = np.polyfit(x, y, args.degree)
    fit_rmse = float(np.sqrt(np.mean((np.poly1d(coefficients)(x) - y) ** 2)))
    thresholds = simulate_thresholds(
        samples, coefficients, [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4]
    )

    print(f"\nsamples: {len(samples)}")
    print(f"x (temb rel-L1)     range: [{x.min():.6g}, {x.max():.6g}]")
    print(f"y (residual rel-L1) range: [{y.min():.6g}, {y.max():.6g}]")
    print(f"mean sum(y) per run: {y.sum() / num_prompts:.6g}")
    print(f"fit rmse: {fit_rmse:.6g}")
    print("\nthreshold -> fraction of stream-layer passes skipped")
    for row in thresholds:
        print(f"  {row['threshold']:<6} {row['skip_ratio']:.3f}")
    print("\ncoefficients=[")
    for c in coefficients:
        print(f"    {c:.8e},")
    print("]")

    with open(args.dump_path, "w") as f:
        json.dump(
            {
                "steps": args.steps,
                "degree": args.degree,
                "coefficients": coefficients.tolist(),
                "fit_rmse": fit_rmse,
                "mean_total_y_per_run": float(y.sum() / num_prompts),
                "threshold_simulation": thresholds,
                "samples": samples,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {args.dump_path}")


def main() -> None:
    args = parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS

    worker, server_args = build_worker(args.model_path)
    transformer = worker.pipeline.get_module("transformer")
    probe = StreamResidualProbe(transformer)
    probe.install()

    try:
        for i, prompt in enumerate(prompts):
            print(f"[{i + 1}/{len(prompts)}] {prompt[:70]}...", flush=True)
            probe.start_run()
            run_prompt(worker, server_args, prompt, args, seed=1234 + i)
    finally:
        probe.remove()

    if not probe.samples:
        raise SystemExit(
            "no samples collected -- check that enable_teacache reached the batch"
        )
    fit_and_report(probe.samples, args, len(prompts))


if __name__ == "__main__":
    main()
