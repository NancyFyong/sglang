import unittest
from unittest.mock import MagicMock

import torch

from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy


def _combine_env():
    req = MagicMock()
    req.cfg_normalization = 0
    req.guidance_rescale = 0

    pipeline_config = MagicMock()
    pipeline_config.postprocess_cfg_noise.side_effect = lambda _, noise, __: noise
    return req, pipeline_config


class TestCFGPolicyCombine(unittest.TestCase):
    def test_cfg_parallel_uses_parallel_arithmetic_order(self):
        policy = CFGPolicy()
        req, pipeline_config = _combine_env()

        pos = torch.tensor([1.0], dtype=torch.bfloat16)
        neg = torch.tensor([0.1], dtype=torch.bfloat16)

        serial = policy.combine([pos, neg], req, 7.0, pipeline_config)
        parallel = policy.combine(
            [pos, neg], req, 7.0, pipeline_config, cfg_parallel=True
        )

        self.assertTrue(torch.equal(serial, neg + 7.0 * (pos - neg)))
        self.assertTrue(torch.equal(parallel, 7.0 * pos + (1 - 7.0) * neg))
        self.assertFalse(torch.equal(serial, parallel))

    def test_exact_parallel_combine_matches_serial_bitwise(self):
        """A model opting in must get identical output with and without CFG parallel.

        This is what makes ``--cfg-parallel-size`` lossless for Boogu, so it has to
        hold bitwise, not approximately: the two paths feed the same sampler state
        and any per-step difference compounds over the denoising loop.
        """
        policy = CFGPolicy(exact_parallel_combine=True)
        req, pipeline_config = _combine_env()

        # Values close to each other, as cond/uncond predictions are: that is the
        # regime where re-associating into cfg_scale * p + (1 - cfg_scale) * n
        # loses precision, so equality here is a real constraint.
        pos = torch.tensor([0.5312, -1.25, 0.0078], dtype=torch.bfloat16)
        neg = torch.tensor([0.5273, -1.24, 0.0079], dtype=torch.bfloat16)

        serial = policy.combine([pos, neg], req, 4.0, pipeline_config)
        parallel = policy.combine(
            [pos, neg], req, 4.0, pipeline_config, cfg_parallel=True
        )

        self.assertTrue(torch.equal(serial, parallel))
        self.assertFalse(
            torch.equal(parallel, 4.0 * pos + (1 - 4.0) * neg),
            "opting in must not fall back to the re-associated formula",
        )


if __name__ == "__main__":
    unittest.main()
