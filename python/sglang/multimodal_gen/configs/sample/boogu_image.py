# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class BooguImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50

    num_frames: int = 1
    height: int = 1024
    width: int = 1024

    guidance_scale: float = 4.0
    negative_prompt: str = ""
    max_sequence_length: int = 1280

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.15,
            coefficients=[
                -3.08045179e00,
                1.27824659e01,
                -1.07070552e01,
                3.88755761e00,
                -2.41140220e-01,
            ],
        )
    )
