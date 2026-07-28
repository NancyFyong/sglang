# SPDX-License-Identifier: Apache-2.0
"""VAE encoding of the LingBot-Video MoE TI2V condition frame.

This is a separate stage rather than a step inside ``DenoisingStage`` because of
generator ordering: the reference pipeline samples the VAE posterior of the
condition frame *before* it draws the initial latent noise, and both draws come
from the same seeded generator. Encoding inside the denoising loop would leave
the noise as the first draw and produce a completely different trajectory for
the same seed.
"""

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.ti2v import (
    encode_condition_latent,
    get_condition_pil_image,
    preprocess_condition_pixels,
    should_apply_lingbot_ti2v,
    single_generator,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision


class LingBotVideoConditionLatentStage(PipelineStage):
    """Encode the TI2V condition frame into ``batch.condition_latent``.

    Must be mounted *before* the latent preparation stage; see the module
    docstring for why the ordering is load-bearing.
    """

    deduplicated_output_fields = ("condition_latent",)

    def __init__(self, vae, component_name: str = "vae") -> None:
        super().__init__()
        self.vae = vae
        self.component_name = component_name

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                self.component_name,
                target_dtype=vae_dtype,
            )
        ]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not should_apply_lingbot_ti2v(batch, server_args):
            return batch

        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        pixels = preprocess_condition_pixels(
            get_condition_pil_image(batch),
            height=int(batch.height),
            width=int(batch.width),
        )
        scale, shift = server_args.pipeline_config.get_decode_scale_and_shift(
            get_local_torch_device(), torch.float32, self.vae
        )
        with self.use_declared_component(
            component_name=self.component_name,
            module=self.vae,
            target_dtype=vae_dtype,
        ) as vae:
            assert vae is not None
            self.vae = vae
            batch.condition_latent = encode_condition_latent(
                vae=vae,
                pixels=pixels,
                generator=single_generator(batch),
                scale=scale,
                shift=shift,
            )
        return batch
