# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel bookkeeping for the Boogu-Image DiT.

Ulysses moves tokens between ranks and heads between tensors. Get any of that
arithmetic wrong and the model still runs -- it just attends to the wrong keys
and returns a plausible, wrong image. These tests pin the parts that fail
silently, on CPU with no distributed group.
"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.boogu_image import BooguImageDitConfig
from sglang.multimodal_gen.runtime.distributed import sp_shard_utils
from sglang.multimodal_gen.runtime.models.dits import boogu_image
from sglang.multimodal_gen.runtime.models.dits.boogu_image import (
    BooguImageTransformer2DModel,
    BooguRopeBundle,
    chunk_freqs_cis,
    expand_kv_heads,
    interleave_instruct_image,
    pad_freqs_cis,
    shard_sequence,
    ulysses_kv_head_repeats,
    validate_sequence_parallel_config,
)

NUM_HEADS = 28
NUM_KV_HEADS = 7


class TestUlyssesKvHeadRepeats(unittest.TestCase):
    def test_boogu_head_counts_across_supported_degrees(self):
        self.assertEqual(
            ulysses_kv_head_repeats(NUM_HEADS, NUM_KV_HEADS, ulysses_size=1), 1
        )
        self.assertEqual(
            ulysses_kv_head_repeats(NUM_HEADS, NUM_KV_HEADS, ulysses_size=2), 2
        )
        self.assertEqual(
            ulysses_kv_head_repeats(NUM_HEADS, NUM_KV_HEADS, ulysses_size=4), 4
        )
        self.assertEqual(
            ulysses_kv_head_repeats(NUM_HEADS, NUM_KV_HEADS, ulysses_size=7), 1
        )

    def test_degree_that_does_not_divide_query_heads_is_rejected(self):
        with self.assertRaises(ValueError):
            ulysses_kv_head_repeats(NUM_HEADS, NUM_KV_HEADS, ulysses_size=8)

    def test_widened_kv_heads_stay_a_whole_number_of_query_groups(self):
        for ulysses_size in (1, 2, 4, 7, 14, 28):
            repeats = ulysses_kv_head_repeats(NUM_HEADS, NUM_KV_HEADS, ulysses_size)
            self.assertEqual(NUM_HEADS % (NUM_KV_HEADS * repeats), 0)


class TestExpandKvHeads(unittest.TestCase):
    def _kv_head_of_query(self, num_kv_heads: int, query_head: int) -> int:
        return query_head // (NUM_HEADS // num_kv_heads)

    def test_duplication_preserves_the_gqa_pairing(self):
        for ulysses_size in (2, 4):
            repeats = ulysses_kv_head_repeats(NUM_HEADS, NUM_KV_HEADS, ulysses_size)
            kv = torch.arange(NUM_KV_HEADS, dtype=torch.float32).view(1, 1, -1, 1)
            widened = expand_kv_heads(kv, repeats)

            self.assertEqual(widened.shape[2], NUM_KV_HEADS * repeats)
            for query_head in range(NUM_HEADS):
                widened_kv_head = self._kv_head_of_query(
                    NUM_KV_HEADS * repeats, query_head
                )
                self.assertEqual(
                    widened[0, 0, widened_kv_head, 0].item(),
                    float(self._kv_head_of_query(NUM_KV_HEADS, query_head)),
                )

    def test_per_rank_head_shards_keep_queries_with_their_keys(self):
        ulysses_size = 4
        repeats = ulysses_kv_head_repeats(NUM_HEADS, NUM_KV_HEADS, ulysses_size)
        widened_kv_heads = NUM_KV_HEADS * repeats
        heads_per_rank = NUM_HEADS // ulysses_size
        kv_heads_per_rank = widened_kv_heads // ulysses_size

        for rank in range(ulysses_size):
            for local_query_head in range(heads_per_rank):
                global_query_head = rank * heads_per_rank + local_query_head
                local_kv_head = local_query_head // (
                    heads_per_rank // kv_heads_per_rank
                )
                global_kv_head = rank * kv_heads_per_rank + local_kv_head
                self.assertEqual(
                    global_kv_head // repeats,
                    self._kv_head_of_query(NUM_KV_HEADS, global_query_head),
                )


class TestSequenceSharding(unittest.TestCase):
    def test_chunks_concatenate_back_into_the_input(self):
        tensor = torch.arange(2 * 12 * 3, dtype=torch.float32).view(2, 12, 3)
        chunks = [shard_sequence(tensor, num_chunks=4, chunk_index=i) for i in range(4)]

        for chunk in chunks:
            self.assertEqual(chunk.shape, (2, 3, 3))
        torch.testing.assert_close(torch.cat(chunks, dim=1), tensor)

    def test_freqs_chunks_track_the_token_chunks(self):
        tokens = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
        cos = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
        sin = -cos

        for chunk_index in range(2):
            token_chunk = shard_sequence(tokens, num_chunks=2, chunk_index=chunk_index)
            chunk_cos, chunk_sin = chunk_freqs_cis(
                (cos, sin), num_chunks=2, chunk_index=chunk_index
            )
            torch.testing.assert_close(chunk_cos, token_chunk)
            torch.testing.assert_close(chunk_sin, -token_chunk)

    def test_padding_freqs_are_identity_rotations(self):
        cos = torch.full((1, 3, 4), 0.5)
        sin = torch.full((1, 3, 4), 0.25)

        padded_cos, padded_sin = pad_freqs_cis((cos, sin), pad=2)

        self.assertEqual(padded_cos.shape, (1, 5, 4))
        torch.testing.assert_close(padded_cos[:, 3:], torch.ones((1, 2, 4)))
        torch.testing.assert_close(padded_sin[:, 3:], torch.zeros((1, 2, 4)))

    def test_padding_is_a_no_op_when_the_stream_already_divides(self):
        freqs = (torch.zeros((1, 4, 2)), torch.zeros((1, 4, 2)))

        self.assertIs(pad_freqs_cis(freqs, pad=0), freqs)


class TestUniformJointLayout(unittest.TestCase):
    def test_uniform_lengths_pack_as_plain_concatenation(self):
        instruct = torch.arange(2 * 3 * 2, dtype=torch.float32).view(2, 3, 2)
        img = torch.arange(2 * 5 * 2, dtype=torch.float32).view(2, 5, 2) + 100

        packed = interleave_instruct_image(
            instruct=instruct,
            img=img,
            encoder_seq_lengths=[3, 3],
            seq_lengths=[8, 8],
        )

        torch.testing.assert_close(packed, torch.cat([instruct, img], dim=1))


def _rope_bundle(
    instruction_seq_lengths, combined_img_seq_lengths, dim=2
) -> BooguRopeBundle:
    batch_size = len(instruction_seq_lengths)
    seq_lengths = [
        enc + img for enc, img in zip(instruction_seq_lengths, combined_img_seq_lengths)
    ]

    def freqs(length):
        return (
            torch.zeros((batch_size, length, dim)),
            torch.zeros((batch_size, length, dim)),
        )

    return BooguRopeBundle(
        joint=freqs(max(seq_lengths)),
        context=freqs(max(instruction_seq_lengths)),
        noise=freqs(max(combined_img_seq_lengths)),
        combined_img=freqs(max(combined_img_seq_lengths)),
        instruction_seq_lengths=instruction_seq_lengths,
        seq_lengths=seq_lengths,
        combined_img_seq_lengths=combined_img_seq_lengths,
    )


def _bare_transformer(sp_size: int) -> BooguImageTransformer2DModel:
    """A transformer with only the layout state -- `__init__` builds 40 layers."""
    transformer = BooguImageTransformer2DModel.__new__(BooguImageTransformer2DModel)
    transformer.sp_size = sp_size
    transformer.dim = 4
    return transformer


def _instruct(seq_len: int, batch_size: int = 1, dim: int = 4) -> torch.Tensor:
    return torch.arange(batch_size * seq_len * dim, dtype=torch.float32).view(
        batch_size, seq_len, dim
    )


class TestLayoutStreamInputs(unittest.TestCase):
    def test_unsharded_layout_is_the_global_packing(self):
        transformer = _bare_transformer(sp_size=1)
        rope = _rope_bundle([3, 2], [5, 5])
        img = torch.zeros((2, 5, 4))
        instruct = _instruct(3, batch_size=2)

        local_img, local_instruct, layout = transformer._layout_stream_inputs(
            img_hidden_states=img,
            instruct_hidden_states=instruct,
            rope=rope,
            sequence_shard_enabled=False,
            text_shard_enabled=False,
        )

        self.assertIs(local_img, img)
        self.assertIs(local_instruct, instruct)
        self.assertFalse(layout.text_sharded)
        self.assertEqual(layout.num_replicated_prefix, 0)
        self.assertEqual(layout.encoder_seq_lengths, [3, 2])
        self.assertEqual(layout.seq_lengths, [8, 7])
        self.assertEqual(layout.img_seq_lengths, [5, 5])

    def test_sharded_layout_splits_only_the_image_stream(self):
        transformer = _bare_transformer(sp_size=2)
        rope = _rope_bundle([3], [8])
        img = torch.arange(8 * 4, dtype=torch.float32).view(1, 8, 4)
        instruct = _instruct(3)

        shards = []
        for rank in range(2):
            with patch.object(
                boogu_image,
                "get_sp_group",
                return_value=type("G", (), {"rank_in_group": rank})(),
            ):
                local_img, local_instruct, layout = transformer._layout_stream_inputs(
                    img_hidden_states=img,
                    instruct_hidden_states=instruct,
                    rope=rope,
                    sequence_shard_enabled=True,
                    text_shard_enabled=False,
                )
            shards.append(local_img)
            self.assertIs(local_instruct, instruct)
            self.assertFalse(layout.text_sharded)
            self.assertEqual(layout.num_replicated_prefix, 3)
            self.assertEqual(layout.encoder_seq_lengths, [3])
            self.assertEqual(layout.img_seq_lengths, [4])
            self.assertEqual(layout.seq_lengths, [7])
            self.assertEqual(layout.global_img_seq_len, 8)
            self.assertEqual(layout.joint_freqs_cis[0].shape, (1, 7, 2))

        torch.testing.assert_close(torch.cat(shards, dim=1), img)

    def test_sharded_layout_pads_an_indivisible_image_stream(self):
        transformer = _bare_transformer(sp_size=2)
        rope = _rope_bundle([3], [9])
        img = torch.ones((1, 9, 4))
        instruct = _instruct(3)

        with patch.object(
            boogu_image,
            "get_sp_group",
            return_value=type("G", (), {"rank_in_group": 1})(),
        ):
            local_img, _, layout = transformer._layout_stream_inputs(
                img_hidden_states=img,
                instruct_hidden_states=instruct,
                rope=rope,
                sequence_shard_enabled=True,
                text_shard_enabled=False,
            )

        self.assertEqual(local_img.shape, (1, 5, 4))
        self.assertEqual(layout.global_img_seq_len, 9)
        torch.testing.assert_close(local_img[0, 4], torch.zeros(4))
        torch.testing.assert_close(layout.img_freqs_cis[0][0, 4], torch.ones(2))

    def test_text_sharded_layout_splits_both_streams(self):
        transformer = _bare_transformer(sp_size=2)
        rope = _rope_bundle([4], [8])
        img = torch.arange(8 * 4, dtype=torch.float32).view(1, 8, 4)
        instruct = _instruct(4)

        img_shards, instruct_shards = [], []
        for rank in range(2):
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        boogu_image,
                        "get_sp_group",
                        return_value=type("G", (), {"rank_in_group": rank})(),
                    )
                )
                stack.enter_context(
                    patch.object(sp_shard_utils, "get_sp_world_size", return_value=2)
                )
                stack.enter_context(
                    patch.object(
                        sp_shard_utils, "get_sp_parallel_rank", return_value=rank
                    )
                )
                local_img, local_instruct, layout = transformer._layout_stream_inputs(
                    img_hidden_states=img,
                    instruct_hidden_states=instruct,
                    rope=rope,
                    sequence_shard_enabled=True,
                    text_shard_enabled=True,
                )
            img_shards.append(local_img)
            instruct_shards.append(local_instruct)
            self.assertTrue(layout.text_sharded)
            self.assertEqual(layout.num_replicated_prefix, 0)
            self.assertEqual(layout.encoder_seq_lengths, [2])
            self.assertEqual(layout.img_seq_lengths, [4])
            self.assertEqual(layout.seq_lengths, [6])
            self.assertEqual(layout.global_img_seq_len, 8)
            self.assertEqual(layout.global_instruct_len, 4)
            self.assertEqual(layout.joint_freqs_cis[0].shape, (1, 6, 2))

        torch.testing.assert_close(torch.cat(img_shards, dim=1), img)
        torch.testing.assert_close(torch.cat(instruct_shards, dim=1), instruct)


class TestSequenceParallelConfigValidation(unittest.TestCase):
    def test_ring_degree_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_sequence_parallel_config(sp_size=4, ring_size=2)

    def test_ulysses_only_and_single_gpu_are_accepted(self):
        validate_sequence_parallel_config(sp_size=4, ring_size=1)
        validate_sequence_parallel_config(sp_size=1, ring_size=1)

    def test_packed_qkv_input_a2a_defaults_on(self):
        self.assertTrue(BooguImageDitConfig().arch_config.enable_packed_qkv_input_a2a)


if __name__ == "__main__":
    unittest.main()
