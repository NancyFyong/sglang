# SPDX-License-Identifier: Apache-2.0
"""LingBot-Video MoE TI2V-specific helpers shared by the generic denoising stage."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoETI2VConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2


def should_apply_lingbot_ti2v(batch: Req, server_args: ServerArgs) -> bool:
    """Return whether the request should use the LingBot TI2V latent path."""

    return bool(
        server_args.pipeline_config.task_type == ModelTaskType.TI2V
        and batch.condition_image is not None
        and type(server_args.pipeline_config) is LingBotVideoMoETI2VConfig
    )


def get_condition_pil_image(batch: Req) -> Image.Image:
    """Return the single un-preprocessed condition frame as a PIL image."""

    image = batch.condition_image
    if isinstance(image, list):
        assert len(image) >= 1, "TI2V requires a condition image"
        image = image[0]
    assert isinstance(
        image, Image.Image
    ), f"LingBot TI2V expects an un-preprocessed PIL condition image, got {type(image)}"
    return image


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """Round ``(height, width)`` to a multiple of ``factor`` within a pixel budget."""

    if max_pixels is None:
        max_pixels = IMAGE_MAX_TOKEN_NUM * factor**2
    if min_pixels is None:
        min_pixels = IMAGE_MIN_TOKEN_NUM * factor**2
    if max_pixels < min_pixels:
        raise ValueError(
            f"max_pixels ({max_pixels}) must be >= min_pixels ({min_pixels})."
        )
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"Condition image aspect ratio must be <= {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}."
        )

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def preprocess_condition_pixels(
    image: Image.Image, height: int, width: int
) -> torch.Tensor:
    """Scale-to-cover then center-crop the condition frame to ``(1, 3, 1, H, W)`` in [0, 1]."""

    # Interpolate on uint8, not on a float copy: the uint8 bilinear kernel rounds
    # every output pixel back to an integer, so promoting first shifts ~all pixels.
    raw = (
        torch.from_numpy(np.array(image.convert("RGB")))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    old_height, old_width = raw.shape[-2:]
    scale = max(height / old_height, width / old_width)
    new_height = max(math.ceil(old_height * scale), height)
    new_width = max(math.ceil(old_width * scale), width)
    resized = F.interpolate(
        raw, size=(new_height, new_width), mode="bilinear", align_corners=False
    )
    top = int(round((new_height - height) / 2.0))
    left = int(round((new_width - width) / 2.0))
    cropped = resized[:, :, top : top + height, left : left + width].float() / 255.0
    return cropped.unsqueeze(2)


def build_vlm_image(pixels: torch.Tensor, vision_patch_size: int) -> Image.Image:
    """Convert condition pixels back to a PIL image sized for Qwen3-VL."""

    assert pixels.ndim == 5 and pixels.shape[2] == 1, f"got shape {tuple(pixels.shape)}"
    frame = pixels[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    image = Image.fromarray(array, mode="RGB")

    patch_factor = vision_patch_size * SPATIAL_MERGE_SIZE
    resized_height, resized_width = smart_resize(
        height=image.height, width=image.width, factor=patch_factor
    )
    return image.resize((resized_width, resized_height))


def single_generator(batch: Req) -> torch.Generator | None:
    """Return the single generator of a batch-of-one request."""

    if isinstance(batch.generator, list):
        assert len(batch.generator) == 1
        return batch.generator[0]
    return batch.generator


def encode_condition_latent(
    *,
    vae: object,
    pixels: torch.Tensor,
    generator: torch.Generator | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """VAE-encode the single condition frame into a normalized clean latent."""

    device = get_local_torch_device()
    pixels = pixels.to(device=device, dtype=torch.float32)
    norm_pixels = (pixels - 0.5) / 0.5
    # The bf16 autocast makes `sample()` draw bf16 noise, which fixes both its
    # values and how many generator offsets it consumes before the initial noise.
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        latents = vae.encode(norm_pixels).sample(generator)
    return (latents.float() - shift) * scale


def apply_condition_latent(
    latents: torch.Tensor, condition_latent: torch.Tensor
) -> torch.Tensor:
    """Pin ``condition_latent`` to the leading latent frames, out of place."""

    assert latents.ndim == 5 and condition_latent.ndim == 5
    condition_frames = condition_latent.shape[2]
    assert condition_frames <= latents.shape[2]
    return torch.cat(
        [
            condition_latent.to(device=latents.device, dtype=latents.dtype),
            latents[:, :, condition_frames:],
        ],
        dim=2,
    )


def pin_lingbot_ti2v_condition(*, latents: torch.Tensor, batch: Req) -> torch.Tensor:
    """Pin the pre-encoded condition latent to the first latent frame."""

    # LingBot replaces the first latent frame instead of concatenating along the
    # channel dim, so an image latent from ImageVAEEncodingStage would be wrong.
    assert batch.image_latent is None, "TI2V task should not have image latents"
    condition_latent = batch.condition_latent
    assert (
        condition_latent is not None
    ), "LingBotVideoConditionLatentStage must run before the denoising loop"
    if get_sp_world_size() > 1:
        raise NotImplementedError(
            "LingBot-Video MoE TI2V does not support sequence parallelism yet: "
            "the conditioned first latent frame would have to be re-pinned on "
            "the owning SP rank only. Run with --ulysses-degree 1 --ring-degree 1."
        )

    batch.latents = apply_condition_latent(latents, condition_latent).to(
        get_local_torch_device()
    )
    return condition_latent
