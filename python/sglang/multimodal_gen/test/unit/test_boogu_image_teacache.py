# SPDX-License-Identifier: Apache-2.0
"""TeaCache bookkeeping for the Boogu-Image DiT.

These tests exercise the cache accounting on a bare instance (no weights, no
GPU): the parts that silently produce wrong pixels rather than crashing if they
regress.
"""

import inspect
import unittest

import torch

from sglang.multimodal_gen.configs.models.dits.boogu_image import BooguImageDitConfig
from sglang.multimodal_gen.configs.sample.boogu_image import BooguImageSamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.boogu_image import (
    BooguDoubleStreamBlock,
    BooguImageTransformer2DModel,
)


class _FakeBatch:
    """The five fields `_get_teacache_context` reads off the request."""

    def __init__(
        self,
        num_inference_steps: int = 4,
        do_cfg: bool = False,
        is_cfg_negative: bool = False,
        teacache_params: TeaCacheParams | None = None,
    ):
        self.enable_teacache = True
        self.teacache_params = teacache_params or TeaCacheParams(
            teacache_thresh=1e9, coefficients=[1.0, 0.0], start_skipping=1
        )
        self.num_inference_steps = num_inference_steps
        self.do_classifier_free_guidance = do_cfg
        self.is_cfg_negative = is_cfg_negative


def _bare_transformer() -> BooguImageTransformer2DModel:
    """A transformer with only TeaCache state -- `__init__` would build 40 layers."""
    transformer = BooguImageTransformer2DModel.__new__(BooguImageTransformer2DModel)
    transformer.config = BooguImageDitConfig()
    transformer._init_teacache_state()
    return transformer


class TestBooguTeaCacheCfgSeparation(unittest.TestCase):
    def test_cfg_branches_keep_independent_residuals(self):
        transformer = _bare_transformer()
        positive_residual = torch.full((2, 3), 1.0)
        negative_residual = torch.full((2, 3), -5.0)
        hidden_states = torch.zeros((2, 3))

        transformer.is_cfg_negative = False
        transformer.maybe_cache_states(positive_residual, torch.zeros((2, 3)))
        transformer.is_cfg_negative = True
        transformer.maybe_cache_states(negative_residual, torch.zeros((2, 3)))

        transformer.is_cfg_negative = False
        torch.testing.assert_close(
            transformer.retrieve_cached_states(hidden_states), positive_residual
        )
        transformer.is_cfg_negative = True
        torch.testing.assert_close(
            transformer.retrieve_cached_states(hidden_states), negative_residual
        )

    def test_boogu_prefix_allocates_the_negative_cache_slots(self):
        transformer = _bare_transformer()

        self.assertTrue(transformer._supports_cfg_cache)
        self.assertIsNone(transformer.previous_residual_negative)

    def test_residual_is_the_difference_across_the_stream_layers(self):
        transformer = _bare_transformer()
        original = torch.tensor([[1.0, 2.0]])
        refined = torch.tensor([[4.0, 6.0]])

        transformer.maybe_cache_states(refined, original)
        later_input = torch.tensor([[10.0, 20.0]])
        torch.testing.assert_close(
            transformer.retrieve_cached_states(later_input),
            later_input + (refined - original),
        )


class TestBooguTeaCacheSkipDecision(unittest.TestCase):
    def _decide(self, transformer, batch, current_timestep, temb):
        with set_forward_context(
            current_timestep=current_timestep, attn_metadata=None, forward_batch=batch
        ):
            return transformer.should_skip_forward_for_cached_states(temb=temb)

    def test_first_step_never_skips(self):
        transformer = _bare_transformer()
        transformer.enable_teacache = True

        self.assertFalse(self._decide(transformer, _FakeBatch(), 0, torch.ones((1, 8))))

    def test_identical_temb_below_threshold_skips(self):
        transformer = _bare_transformer()
        transformer.enable_teacache = True
        batch = _FakeBatch(num_inference_steps=8)
        temb = torch.ones((1, 8))

        self._decide(transformer, batch, 0, temb)
        transformer.cnt = 2
        self.assertTrue(self._decide(transformer, batch, 2, temb))

    def test_boundary_steps_always_compute(self):
        transformer = _bare_transformer()
        transformer.enable_teacache = True
        batch = _FakeBatch(num_inference_steps=8)
        temb = torch.ones((1, 8))

        self._decide(transformer, batch, 0, temb)
        transformer.cnt = batch.num_inference_steps - 1
        self.assertFalse(self._decide(transformer, batch, 7, temb))

    def test_disabled_teacache_never_skips(self):
        transformer = _bare_transformer()
        transformer.enable_teacache = False

        self.assertFalse(self._decide(transformer, _FakeBatch(), 3, torch.ones((1, 8))))


class TestBooguCacheDitContract(unittest.TestCase):
    def test_double_stream_block_uses_forward_pattern_0_names(self):
        params = list(inspect.signature(BooguDoubleStreamBlock.forward).parameters)

        self.assertEqual(params[:3], ["self", "hidden_states", "encoder_hidden_states"])

    def test_sampling_params_ship_calibrated_teacache_coefficients(self):
        params = BooguImageSamplingParams()

        self.assertGreater(params.teacache_params.teacache_thresh, 0.0)
        self.assertNotEqual(params.teacache_params.get_coefficients(), [])


if __name__ == "__main__":
    unittest.main()
