# SPDX-License-Identifier: Apache-2.0

import torch
from PIL import Image

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.ti2v import (
    build_vlm_image,
    get_condition_pil_image,
    preprocess_condition_pixels,
    should_apply_lingbot_ti2v,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0

PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PROMPT_TEMPLATE = "<|vision_start|><|video_pad|><|vision_end|>"


class LingBotVideoTextEncodingStage(TextEncodingStage):
    """Qwen3-VL prompt/negative encoding for LingBot-Video MoE (T2V and TI2V).

    For TI2V the condition frame is fed to Qwen3-VL as a visual token block
    (``IMG_PROMPT_TEMPLATE`` prepended to the user text) for *both* the positive
    and the negative prompt, matching the reference implementation.
    """

    def __init__(self, text_encoders, tokenizers, transformer):
        super().__init__(text_encoders, tokenizers)
        self.transformer = transformer
        self.token_length = TOKEN_LENGTH
        self.hidden_state_skip_layer = HIDDEN_STATE_SKIP_LAYER
        self.prompt_template = PROMPT_TEMPLATE
        self._crop_start: int | None = None

    @staticmethod
    def check_inputs(height: int, width: int, num_frames: int) -> None:
        if num_frames != 1 and (num_frames - 1) % 4 != 0:
            raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` must be multiples of 16, got {height}x{width}."
            )

    @staticmethod
    def apply_text_to_template(text: str, template: str = PROMPT_TEMPLATE) -> str:
        return template.format(text)

    def _compute_crop_start(self) -> int:
        processor = self.tokenizers[0]
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            marked = self.prompt_template.format(marker)
            marker_pos = marked.find(marker)
            if marker_pos < 0:
                self._crop_start = 0
            else:
                prefix = processor(
                    text=marked[:marker_pos],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    def _build_prompt_inputs(
        self,
        prompt: str | list[str],
        images: list[Image.Image] | None = None,
    ):
        processor = self.tokenizers[0]
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        # The visual marker sits in front of the user text, inside the template.
        visual_template = IMG_PROMPT_TEMPLATE if images is not None else ""
        texts = [
            self.apply_text_to_template(visual_template + text, self.prompt_template)
            for text in prompts
        ]
        return processor(
            text=texts,
            images=images,
            videos=None,
            video_metadata=None,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
        )

    @torch.no_grad()
    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
        images: list[Image.Image] | None = None,
    ):
        text_encoder = self.text_encoders[0]
        if text_encoder is None or self.tokenizers[0] is None:
            raise ValueError(
                "`text_encoder` and `processor` are required for encode_prompt()."
            )

        inputs = self._build_prompt_inputs(prompt, images=images)
        inputs = inputs.to(device)
        # SGLang's native Qwen3-VL builds its attention out of `LocalAttention`,
        # which reads the forward context; the transformers fallback does not.
        # Without this the native encoder dies with "Forward context is not set".
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = text_encoder(
                **inputs,
                output_hidden_states=self.hidden_state_skip_layer is not None,
                # Only hidden states are used; keeping one token's logits avoids a
                # `seq_len x 151936` projection whose result is thrown away.
                logits_to_keep=1,
            )
        if self.hidden_state_skip_layer is not None:
            prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        else:
            prompt_embeds = outputs.last_hidden_state

        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        # B=1: drop right padding before DiT inference.
        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]

        return prompt_embeds.to(dtype=dtype), prompt_mask

    def _build_vlm_images(
        self, batch: Req, server_args: ServerArgs
    ) -> list[Image.Image] | None:
        """Return the TI2V condition frame sized for Qwen3-VL, or None for T2V."""

        if not should_apply_lingbot_ti2v(batch, server_args):
            return None
        pixels = preprocess_condition_pixels(
            get_condition_pil_image(batch),
            height=int(batch.height),
            width=int(batch.width),
        )
        # Read the patch size off the processor, not off the text encoder: the
        # encoder is either SGLang's native Qwen3-VL or the transformers one
        # (loader fallback), and only the processor exposes it the same way in
        # both cases. It is also the value the processor itself patchifies with,
        # which is what matters when we pass ``do_resize=False``.
        image_processor = self.tokenizers[0].image_processor
        return [build_vlm_image(pixels, image_processor.patch_size)]

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dtype = next(self.transformer.parameters(), torch.tensor([])).dtype
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            dtype = torch.bfloat16

        self.check_inputs(int(batch.height), int(batch.width), int(batch.num_frames))

        # The same condition frame conditions both branches of CFG.
        images = self._build_vlm_images(batch, server_args)

        prompt_embeds, prompt_mask = self._encode_prompt(
            batch.prompt, device, dtype, images=images
        )
        batch.prompt_embeds = [prompt_embeds]
        batch.prompt_attention_mask = prompt_mask

        if batch.do_classifier_free_guidance:
            negative_embeds, negative_mask = self._encode_prompt(
                batch.negative_prompt, device, dtype, images=images
            )
            batch.negative_prompt_embeds = [negative_embeds]
            batch.negative_attention_mask = negative_mask
        return batch
