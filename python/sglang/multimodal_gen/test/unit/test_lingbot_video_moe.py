# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
from types import SimpleNamespace

import torch
from PIL import Image

from sglang.multimodal_gen.configs.models.dits.lingbot_video_moe import (
    LingBotVideoMoEArchConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
    LingBotVideoMoETI2VConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    LingBotVideoMoESamplingParams,
    LingBotVideoMoETI2VSamplingParams,
)
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_model_info,
    get_pipeline_config_classes,
)
from sglang.multimodal_gen.runtime.layers.moe import (
    LingBotVideoGroupedExperts,
    LingBotVideoRouter,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.models.dits import (
    lingbot_video_moe as dits_lingbot_video_moe,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_video_moe import (
    LingBotVideoAttention,
    LingBotVideoTransformer3DModel,
    _joint_position_ids,
    make_joint_position_ids,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    IMG_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE,
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.ti2v import (
    apply_condition_latent,
    encode_condition_latent,
    pin_lingbot_ti2v_condition,
    preprocess_condition_pixels,
    should_apply_lingbot_ti2v,
)

_LINGBOT_MODULE_SUBDIRS = (
    "scheduler",
    "text_encoder",
    "processor",
    "transformer",
    "vae",
)


def test_moe_path_resolves_moe_configs():
    get_model_info.cache_clear()
    _get_config_info.cache_clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "lingbot-video-moe-30b-a3b")
        os.makedirs(model_dir)
        with open(
            os.path.join(model_dir, "model_index.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"_class_name": "LingBotVideoPipeline", "_diffusers_version": "0.37.1"},
                f,
            )
        for subdir in _LINGBOT_MODULE_SUBDIRS:
            os.mkdir(os.path.join(model_dir, subdir))
        info = get_model_info(model_dir, backend="sglang")

    assert info.pipeline_cls.__name__ == "LingBotVideoPipeline"
    assert info.pipeline_config_cls is LingBotVideoMoEPipelineConfig
    assert info.sampling_param_cls is LingBotVideoMoESamplingParams


def test_arch_config_defaults_without_mlp_only_layers():
    arch = LingBotVideoMoEArchConfig()
    assert arch.num_experts == 128
    assert arch.mlp_only_layers == ()


def test_router_bias_shifts_selection_but_not_gate_weights():
    router = LingBotVideoRouter(
        hidden_size=4,
        num_experts=4,
        top_k=2,
        score_func="sigmoid",
        norm_topk_prob=False,
        n_group=None,
        topk_group=None,
        route_scale=1.0,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [-2.0, 0.0, 0.0, 0.0],
                    [-4.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))

    top_indices, top_scores = router(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    assert set(top_indices[0].tolist()) == {0, 3}
    raw = torch.sigmoid(torch.tensor([4.0, -4.0]))
    picked = {
        int(idx): float(score.detach())
        for idx, score in zip(top_indices[0], top_scores[0])
    }
    assert abs(picked[0] - float(raw[0])) < 1e-5
    assert abs(picked[3] - float(raw[1])) < 1e-5


def _sdpa(q, k, v, attn_mask=None, attn_mask_meta=None):
    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if attn_mask is not None and attn_mask.dim() == 2:
        attn_mask = attn_mask[:, None, None, :]
    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=attn_mask
    )
    return out.transpose(1, 2)


def _real_attention(num_heads, head_dim):
    attn = object.__new__(LingBotVideoAttention)
    attn.local_num_heads = num_heads
    attn.head_dim = head_dim
    attn.to_q = attn.to_k = attn.to_v = attn.to_out = lambda x: (x, None)
    attn.norm_q = attn.norm_k = lambda t: t
    attn.attn = _sdpa
    return attn


def test_attention_isolates_samples_across_batch(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, batch, seq_len = 4, 8, 3, 8
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, hidden)
    freqs = torch.zeros(batch * seq_len, head_dim // 2)

    valid = [seq_len, seq_len - 2, seq_len - 5]
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for i, length in enumerate(valid):
        mask[i, :length] = True

    batched = attn.forward(x, (freqs, freqs), mask)

    for i, length in enumerate(valid):
        solo = attn.forward(
            x[i : i + 1],
            (freqs[i * seq_len : (i + 1) * seq_len],) * 2,
            mask[i : i + 1],
        )
        torch.testing.assert_close(batched[i : i + 1, :length], solo[:, :length])

    # Flattening the batch into one sequence lets sample 0 attend across the
    # boundary; its output must differ from the isolated per-sample result.
    flat = attn.forward(x.reshape(1, batch * seq_len, hidden), (freqs, freqs), None)
    flat = flat.reshape(batch, seq_len, hidden)
    assert (flat[0, : valid[0]] - batched[0, : valid[0]]).abs().max() > 1e-3


def test_attention_forwards_2d_mask_and_varlen_metadata(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, batch, seq_len = 4, 8, 2, 6
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    captured = {}

    def capture_attention(q, k, v, attn_mask=None, attn_mask_meta=None):
        captured["mask"] = attn_mask
        captured["meta"] = attn_mask_meta
        return _sdpa(q, k, v, attn_mask=attn_mask)

    attn.attn = capture_attention
    x = torch.randn(batch, seq_len, hidden)
    freqs = torch.zeros(batch * seq_len, head_dim // 2)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    metadata = {"max_seqlen": seq_len}

    attn.forward(x, (freqs, freqs), mask, metadata)

    assert captured["mask"] is mask
    assert captured["meta"] is metadata


def test_attention_single_sample_matches_direct_attention(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, seq_len = 4, 8, 6
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, hidden)
    freqs = torch.zeros(seq_len, head_dim // 2)

    out = attn.forward(x, (freqs, freqs), attention_mask=None)

    qkv = x.unflatten(2, (num_heads, head_dim))
    expected = _sdpa(qkv, qkv, qkv).flatten(2)
    torch.testing.assert_close(out, expected)


class _FakeBatchEncoding(dict):
    def to(self, _device):
        return self


class _FakeQwenProcessor:
    def __init__(self, prompt_width, prefix_width, true_len):
        self.prompt_width = prompt_width
        self.prefix_width = prefix_width
        self.true_len = true_len

    def __call__(self, **kwargs):
        if "max_length" in kwargs:
            width = self.prompt_width
            mask = torch.zeros(1, width, dtype=torch.long)
            mask[0, : self.true_len] = 1
        else:
            width = self.prefix_width
            mask = torch.ones(1, width, dtype=torch.long)
        return _FakeBatchEncoding(
            input_ids=torch.zeros(1, width, dtype=torch.long),
            attention_mask=mask,
        )


def _text_encoding_stage(processor, encoder):
    stage = object.__new__(LingBotVideoTextEncodingStage)
    stage.text_encoders = [encoder]
    stage.tokenizers = [processor]
    stage.token_length = 128
    stage.hidden_state_skip_layer = 0
    stage.prompt_template = PROMPT_TEMPLATE
    stage._crop_start = None
    return stage


def test_text_encoding_crops_template_then_trims_padding():
    prompt_width, prefix_width, true_len, channels = 10, 3, 8, 4
    hidden = torch.arange(prompt_width, dtype=torch.float32)
    hidden = hidden.view(1, prompt_width, 1).expand(1, prompt_width, channels)

    def encoder(**kwargs):
        return SimpleNamespace(hidden_states=[hidden])

    stage = _text_encoding_stage(
        _FakeQwenProcessor(prompt_width, prefix_width, true_len), encoder
    )
    embeds, mask = stage._encode_prompt(
        "a structured caption", torch.device("cpu"), torch.float32
    )

    assert tuple(embeds.shape) == (1, true_len - prefix_width, channels)
    torch.testing.assert_close(embeds, hidden[:, prefix_width:true_len])
    assert int(mask.sum()) == true_len - prefix_width
    assert stage._compute_crop_start() == prefix_width


def test_check_inputs_enforces_frame_and_size_contract():
    check = LingBotVideoTextEncodingStage.check_inputs
    check(480, 832, 1)
    check(480, 832, 81)
    try:
        check(480, 832, 82)
        raise AssertionError("expected ValueError for num_frames=82")
    except ValueError:
        pass
    try:
        check(480, 830, 81)
        raise AssertionError("expected ValueError for width=830")
    except ValueError:
        pass


def test_decode_scale_and_shift_invert_vae_normalization():
    config = LingBotVideoMoEPipelineConfig()
    scale, shift = config.get_decode_scale_and_shift(
        torch.device("cpu"), torch.float32, vae=None
    )
    arch = config.vae_config.arch_config
    std = torch.tensor(arch.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    mean = torch.tensor(arch.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    torch.testing.assert_close(scale, 1.0 / std)
    torch.testing.assert_close(shift, mean)


def test_latents_stay_fp32_under_bf16_precision():
    config = LingBotVideoMoEPipelineConfig()
    assert config.get_latent_dtype(torch.bfloat16) == torch.float32


def test_grouped_experts_store_packed_w13_weight():
    experts = LingBotVideoGroupedExperts(
        num_experts=2, hidden_size=4, intermediate_size=3
    )
    names = {n for n, _ in experts.named_parameters()}
    assert "w13_weight" in names and "w2" in names
    assert "w1" not in names and "w3" not in names
    assert tuple(experts.w13_weight.shape) == (2, 6, 4)  # [E, 2I, H]


def test_preprocess_packs_w1_w3_into_w13_weight():
    pack = LingBotVideoTransformer3DModel.preprocess_loaded_state_dict
    E, I, H = 2, 3, 4
    w1 = torch.arange(E * I * H, dtype=torch.float32).reshape(E, I, H)
    w2 = torch.arange(E * H * I, dtype=torch.float32).reshape(E, H, I)
    w3 = torch.arange(E * I * H, dtype=torch.float32).reshape(E, I, H) + 100.0
    # block 0: w1 before w3; block 1: w3 before w1 (order-independence).
    src = [
        ("blocks.0.ffn.experts.w1", w1),
        ("blocks.0.ffn.experts.w2", w2),
        ("blocks.0.ffn.experts.w3", w3),
        ("blocks.0.ffn.router.weight", torch.zeros(E, H)),
        ("blocks.1.ffn.experts.w3", w3.clone()),
        ("blocks.1.ffn.experts.w2", w2.clone()),
        ("blocks.1.ffn.experts.w1", w1.clone()),
    ]
    out = dict(pack(None, iter(src)))
    assert set(out.keys()) == {
        "blocks.0.ffn.experts.w13_weight",
        "blocks.0.ffn.experts.w2",
        "blocks.0.ffn.router.weight",
        "blocks.1.ffn.experts.w13_weight",
        "blocks.1.ffn.experts.w2",
    }
    packed = torch.cat((w1, w3), dim=1)  # gate then up, dim-1
    torch.testing.assert_close(out["blocks.0.ffn.experts.w13_weight"], packed)
    torch.testing.assert_close(out["blocks.1.ffn.experts.w13_weight"], packed)
    torch.testing.assert_close(out["blocks.0.ffn.experts.w2"], w2)


def test_joint_position_ids_match_reference_and_cover_padding():
    dev = torch.device("cpu")
    gt, gh, gw = 2, 3, 4
    n_video = gt * gh * gw

    # B==1, no padding: byte-identical to the per-sample reference.
    vec = _joint_position_ids(torch.tensor([5]), gt, gh, gw, 5, dev)
    torch.testing.assert_close(vec, make_joint_position_ids(5, gt, gh, gw, dev))

    # B==1 with padding: real tokens match the text_len=4 reference; the extra
    # padding row is (0,0,0). vec has n_video+L rows (matches q for B*S).
    vec_p = _joint_position_ids(torch.tensor([4]), gt, gh, gw, 5, dev)
    torch.testing.assert_close(
        vec_p[: n_video + 4], make_joint_position_ids(4, gt, gh, gw, dev)
    )
    torch.testing.assert_close(
        vec_p[n_video + 4 :], torch.zeros((1, 3), dtype=torch.int32)
    )

    # B>1 with padding: covers B*S rows; each sample's real tokens match its ref.
    text_lens = [5, 3, 6]
    B, L = len(text_lens), 6
    vec_b = _joint_position_ids(torch.tensor(text_lens), gt, gh, gw, L, dev)
    assert vec_b.shape[0] == B * (n_video + L)
    for i, t in enumerate(text_lens):
        start = i * (n_video + L)
        real = n_video + t
        torch.testing.assert_close(
            vec_b[start : start + real], make_joint_position_ids(t, gt, gh, gw, dev)
        )


# --- TI2V (first-frame conditioned) ---


def test_ti2v_class_name_wins_over_t2v_path_detector():
    get_model_info.cache_clear()
    _get_config_info.cache_clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "lingbot-video-moe-30b-a3b")
        os.makedirs(model_dir)
        with open(
            os.path.join(model_dir, "model_index.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "_class_name": "LingBotVideoImageToVideoPipeline",
                    "_diffusers_version": "0.37.1",
                },
                f,
            )
        for subdir in _LINGBOT_MODULE_SUBDIRS:
            os.mkdir(os.path.join(model_dir, subdir))
        info = get_model_info(model_dir, backend="sglang")
    get_model_info.cache_clear()
    _get_config_info.cache_clear()

    assert info.pipeline_cls.__name__ == "LingBotVideoImageToVideoPipeline"
    assert info.pipeline_config_cls is LingBotVideoMoETI2VConfig
    assert info.sampling_param_cls is LingBotVideoMoETI2VSamplingParams


def test_ti2v_config_loads_vae_encoder_and_keeps_raw_condition_image():
    config = LingBotVideoMoETI2VConfig()
    assert config.vae_config.load_encoder
    assert config.vae_config.load_decoder
    assert config.skip_input_image_preprocess


def test_should_apply_lingbot_ti2v_needs_ti2v_config_and_image():
    image = Image.new("RGB", (8, 8))
    ti2v_args = SimpleNamespace(pipeline_config=LingBotVideoMoETI2VConfig())
    t2v_args = SimpleNamespace(pipeline_config=LingBotVideoMoEPipelineConfig())

    assert should_apply_lingbot_ti2v(SimpleNamespace(condition_image=image), ti2v_args)
    assert not should_apply_lingbot_ti2v(
        SimpleNamespace(condition_image=None), ti2v_args
    )
    assert not should_apply_lingbot_ti2v(
        SimpleNamespace(condition_image=image), t2v_args
    )


def test_condition_pixels_honor_requested_resolution():
    pixels = preprocess_condition_pixels(
        Image.new("RGB", (1280, 704), color=(255, 128, 0)), height=480, width=832
    )

    assert tuple(pixels.shape) == (1, 3, 1, 480, 832)
    assert pixels.dtype == torch.float32
    torch.testing.assert_close(
        pixels[0, :, 0, 0, 0], torch.tensor([1.0, 128.0 / 255.0, 0.0])
    )
    assert float(pixels.min()) >= 0.0 and float(pixels.max()) <= 1.0


def test_condition_pixels_resize_in_uint8_like_the_reference():
    generator = torch.Generator().manual_seed(0)
    noise = torch.randint(
        0, 256, (67, 121, 3), dtype=torch.uint8, generator=generator
    ).numpy()

    pixels = preprocess_condition_pixels(Image.fromarray(noise), height=96, width=160)

    scaled = pixels * 255.0
    torch.testing.assert_close(scaled, scaled.round(), rtol=0, atol=1e-4)


def test_condition_pixels_center_crop_keeps_the_middle():
    array = torch.zeros(3, 96, 288, dtype=torch.uint8)
    array[:, :, 96:192] = 255
    image = Image.fromarray(array.permute(1, 2, 0).numpy())

    pixels = preprocess_condition_pixels(image, height=96, width=96)

    assert tuple(pixels.shape) == (1, 3, 1, 96, 96)
    torch.testing.assert_close(pixels, torch.ones_like(pixels))


def test_encode_condition_latent_applies_vae_normalization(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
        ti2v,
    )

    monkeypatch.setattr(ti2v, "get_local_torch_device", lambda: torch.device("cpu"))

    raw_latent = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1, 1)
    captured = {}

    class _Dist:
        def sample(self, generator=None):
            captured["generator"] = generator
            return raw_latent

    class _VAE:
        def encode(self, x):
            captured["pixels"] = x
            return _Dist()

    config = LingBotVideoMoETI2VConfig()
    scale, shift = config.get_decode_scale_and_shift(
        torch.device("cpu"), torch.float32, vae=None
    )
    generator = torch.Generator().manual_seed(0)

    latent = encode_condition_latent(
        vae=_VAE(),
        pixels=torch.full((1, 3, 1, 2, 2), 0.75),
        generator=generator,
        scale=scale,
        shift=shift,
    )

    torch.testing.assert_close(
        captured["pixels"], torch.full((1, 3, 1, 2, 2), 0.5), rtol=0, atol=0
    )
    assert captured["generator"] is generator
    torch.testing.assert_close(latent, (raw_latent - shift) * scale)
    torch.testing.assert_close(latent / scale + shift, raw_latent)


def test_apply_condition_latent_rebinds_instead_of_writing_in_place():
    latents = torch.randn(1, 16, 21, 4, 4)
    original = latents.clone()
    condition = torch.randn(1, 16, 1, 4, 4)

    out = apply_condition_latent(latents, condition)

    assert out is not latents
    torch.testing.assert_close(latents, original)
    torch.testing.assert_close(out[:, :, :1], condition)
    torch.testing.assert_close(out[:, :, 1:], original[:, :, 1:])


def test_pin_condition_requires_the_condition_latent_from_its_own_stage(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
        ti2v,
    )

    monkeypatch.setattr(ti2v, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ti2v, "get_sp_world_size", lambda: 1)

    latents = torch.randn(1, 16, 21, 4, 4)
    condition = torch.randn(1, 16, 1, 4, 4)
    batch = SimpleNamespace(image_latent=None, condition_latent=condition, latents=None)

    returned = pin_lingbot_ti2v_condition(latents=latents, batch=batch)

    assert returned is condition
    torch.testing.assert_close(batch.latents[:, :, :1], condition)
    torch.testing.assert_close(batch.latents[:, :, 1:], latents[:, :, 1:])

    batch.condition_latent = None
    try:
        pin_lingbot_ti2v_condition(latents=latents, batch=batch)
    except AssertionError:
        pass
    else:
        raise AssertionError("a missing condition latent must not pass silently")


def test_ti2v_pipeline_encodes_the_condition_before_drawing_noise(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines import lingbot_video_moe as pipelines
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )

    order = []

    for attr in (
        "InputValidationStage",
        "LingBotVideoTextEncodingStage",
        "LingBotVideoConditionLatentStage",
        "DenoisingStage",
    ):
        monkeypatch.setattr(pipelines, attr, lambda *_, _name=attr, **__: _name)
    monkeypatch.setattr(ComposedPipelineBase, "get_module", lambda self, name: name)
    monkeypatch.setattr(
        ComposedPipelineBase, "add_stage", lambda self, stage: order.append(stage)
    )
    for method in (
        "add_standard_latent_preparation_stage",
        "add_standard_timestep_preparation_stage",
        "add_standard_decoding_stage",
    ):
        monkeypatch.setattr(
            ComposedPipelineBase,
            method,
            lambda self, _name=method, **__: order.append(_name),
        )

    t2v = object.__new__(pipelines.LingBotVideoPipeline)
    t2v.create_pipeline_stages(server_args=SimpleNamespace())
    assert "LingBotVideoConditionLatentStage" not in order

    order.clear()
    ti2v_pipeline = object.__new__(pipelines.LingBotVideoImageToVideoPipeline)
    ti2v_pipeline.create_pipeline_stages(server_args=SimpleNamespace())

    assert order.index("LingBotVideoConditionLatentStage") < order.index(
        "add_standard_latent_preparation_stage"
    )
    assert order.index("add_standard_latent_preparation_stage") < order.index(
        "DenoisingStage"
    )


class _RecordingQwenProcessor(_FakeQwenProcessor):
    def __init__(self, *args):
        super().__init__(*args)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return super().__call__(**kwargs)


def _encode_with_images(images):
    prompt_width, prefix_width, true_len, channels = 10, 3, 8, 4
    hidden = torch.zeros(1, prompt_width, channels)
    processor = _RecordingQwenProcessor(prompt_width, prefix_width, true_len)
    stage = _text_encoding_stage(
        processor, lambda **kwargs: SimpleNamespace(hidden_states=[hidden])
    )
    stage._encode_prompt(
        "a structured caption", torch.device("cpu"), torch.float32, images=images
    )
    return next(call for call in processor.calls if "max_length" in call)


def test_text_encoding_passes_condition_image_with_marker():
    call = _encode_with_images([Image.new("RGB", (32, 32))])
    assert call["images"] is not None and len(call["images"]) == 1
    assert IMG_PROMPT_TEMPLATE in call["text"][0]
    assert call["text"][0].index(IMG_PROMPT_TEMPLATE) < call["text"][0].index(
        "a structured caption"
    )


def test_text_encoding_stays_text_only_without_condition_image():
    call = _encode_with_images(None)
    assert call["images"] is None
    assert IMG_PROMPT_TEMPLATE not in call["text"][0]


def test_explicit_pipeline_class_name_refines_t2v_config_to_ti2v():
    config_classes = get_pipeline_config_classes("LingBotVideoImageToVideoPipeline")

    assert config_classes == (
        LingBotVideoMoETI2VConfig,
        LingBotVideoMoETI2VSamplingParams,
    )
    assert issubclass(LingBotVideoMoETI2VConfig, LingBotVideoMoEPipelineConfig)
    assert LingBotVideoMoETI2VConfig is not LingBotVideoMoEPipelineConfig


def test_lingbot_remote_code_modules_load_through_diffusers_loaders():
    assert (
        ComponentLoader.resolve_transformers_or_diffusers(
            "lingbot_video.transformer_lingbot_video", "transformer"
        )
        == "diffusers"
    )
    assert (
        ComponentLoader.resolve_transformers_or_diffusers(
            "lingbot_video_diffusers.scheduling_flow_unipc", "scheduler"
        )
        == "diffusers"
    )
    assert (
        ComponentLoader.resolve_transformers_or_diffusers(
            "transformers", "text_encoder"
        )
        == "transformers"
    )


def test_vlm_image_patch_size_comes_from_the_processor():
    processor = _FakeQwenProcessor(10, 3, 8)
    processor.image_processor = SimpleNamespace(patch_size=16)
    stage = _text_encoding_stage(processor, encoder=None)

    batch = SimpleNamespace(
        condition_image=Image.new("RGB", (1280, 704)), height=480, width=832
    )
    server_args = SimpleNamespace(pipeline_config=LingBotVideoMoETI2VConfig())

    images = stage._build_vlm_images(batch, server_args)

    assert images is not None and len(images) == 1
    assert images[0].width % 32 == 0 and images[0].height % 32 == 0
    assert (
        stage._build_vlm_images(
            batch, SimpleNamespace(pipeline_config=LingBotVideoMoEPipelineConfig())
        )
        is None
    )


def test_qwen3vl_text_attention_honors_the_explicit_head_dim(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    from sglang.multimodal_gen.runtime.models.encoders import qwen3vl

    monkeypatch.setattr(qwen3vl, "LocalAttention", lambda **kwargs: kwargs)
    config = Qwen3VLTextConfig(
        hidden_size=2560,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=1,
    )

    attention = qwen3vl.Qwen3VLTextAttention(config, layer_idx=0)

    assert attention.head_dim == 128
    assert tuple(attention.q_proj.weight.shape) == (4096, 2560)
    assert tuple(attention.k_proj.weight.shape) == (1024, 2560)
    assert tuple(attention.v_proj.weight.shape) == (1024, 2560)
    assert tuple(attention.o_proj.weight.shape) == (2560, 4096)


def _tied_head_encoder(tie_word_embeddings: bool):
    from sglang.multimodal_gen.runtime.models.encoders import qwen3vl

    class _Encoder(torch.nn.Module):
        load_weights = qwen3vl.Qwen3VLForConditionalGeneration.load_weights

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.embed_tokens = torch.nn.Embedding(6, 4)
            self.lm_head = torch.nn.Linear(4, 6, bias=False)
            self.config = SimpleNamespace(
                arch_config=SimpleNamespace(
                    text_config=SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
                )
            )

    return _Encoder()


def test_qwen3vl_ties_the_lm_head_when_the_checkpoint_omits_it():
    encoder = _tied_head_encoder(tie_word_embeddings=True)
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    loaded = encoder.load_weights(
        [("model.language_model.embed_tokens.weight", weight)]
    )

    assert "lm_head.weight" in loaded
    assert encoder.lm_head.weight is encoder.model.language_model.embed_tokens.weight
    assert torch.equal(encoder.lm_head.weight, weight)


def test_qwen3vl_leaves_an_untied_lm_head_to_the_checkpoint():
    encoder = _tied_head_encoder(tie_word_embeddings=False)
    head = encoder.lm_head.weight

    loaded = encoder.load_weights(
        [("model.language_model.embed_tokens.weight", torch.zeros(6, 4))]
    )

    assert "lm_head.weight" not in loaded
    assert encoder.lm_head.weight is head


def test_qwen3vl_hidden_states_end_with_the_final_norm(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    from sglang.multimodal_gen.runtime.models.encoders import qwen3vl

    class _PassThroughAttention(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, query, key, value):
            return query

    monkeypatch.setattr(qwen3vl, "LocalAttention", _PassThroughAttention)
    config = Qwen3VLTextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=3,
        intermediate_size=64,
        vocab_size=64,
    )
    model = qwen3vl.Qwen3VLTextModel(config).eval()

    with torch.no_grad():
        outputs = model(
            input_ids=torch.randint(0, 64, (1, 5)), output_hidden_states=True
        )

    assert len(outputs.hidden_states) == config.num_hidden_layers + 1
    assert torch.equal(outputs.hidden_states[-1], outputs.last_hidden_state)
