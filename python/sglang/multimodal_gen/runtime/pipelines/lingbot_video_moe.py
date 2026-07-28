# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoETI2VConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    LingBotVideoMoETI2VSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoConditionLatentStage,
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _flow_shift_kwarg(batch, server_args: ServerArgs) -> tuple[str, float | None]:
    shift = (
        batch.flow_shift
        if batch.flow_shift is not None
        else server_args.pipeline_config.flow_shift
    )
    return ("shift", shift)


class LingBotVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "LingBotVideoPipeline"
    is_video_pipeline = True

    _required_config_modules = (
        "text_encoder",
        "processor",
        "vae",
        "transformer",
        "scheduler",
    )

    def _add_condition_latent_stage(self) -> None:
        """T2V has no condition frame; the TI2V subclass overrides this."""

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_stage(
            LingBotVideoTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("processor")],
                transformer=self.get_module("transformer"),
            ),
        )
        # Ordering is load-bearing: the condition frame's VAE posterior is
        # sampled off the same generator that draws the initial noise, and the
        # reference pipeline does it in this order.
        self._add_condition_latent_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[_flow_shift_kwarg],
        )
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


class LingBotVideoImageToVideoPipeline(LingBotVideoPipeline):
    """TI2V variant: same stages, plus first-frame conditioning.

    No ``ImageVAEEncodingStage`` is mounted on purpose: LingBot replaces the
    first latent frame with a clean condition latent instead of concatenating an
    image latent along the channel dim, and DenoisingStage rejects
    ``batch.image_latent`` for TI2V. ``LingBotVideoConditionLatentStage``
    produces that condition latent, and DenoisingStage pins it before the loop
    and re-pins it after every scheduler step.
    """

    pipeline_name = "LingBotVideoImageToVideoPipeline"
    pipeline_config_cls = LingBotVideoMoETI2VConfig
    sampling_params_cls = LingBotVideoMoETI2VSamplingParams

    def _add_condition_latent_stage(self) -> None:
        self.add_stage(LingBotVideoConditionLatentStage(vae=self.get_module("vae")))


EntryClass = [LingBotVideoPipeline, LingBotVideoImageToVideoPipeline]
