"""CPU checks with the actual Qwen3 layer implementation and paper's window."""
from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.adapter_looped_qwen import LoopedQwenAdapter


class LoopedQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(4961)
        config = Qwen3Config(
            vocab_size=47, hidden_size=32, intermediate_size=64,
            num_hidden_layers=36, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, max_position_embeddings=128,
            attention_dropout=0.0, pad_token_id=0, eos_token_id=1,
        )
        config._attn_implementation = "sdpa"
        cls.model = Qwen3ForCausalLM(config).eval()
        # Nonuniform norm weights make a missing/duplicate final norm observable.
        with torch.no_grad():
            cls.model.model.norm.weight.copy_(torch.linspace(0.7, 1.4, 32))
        cls.adapter = LoopedQwenAdapter.from_model(
            cls.model, SimpleNamespace(pad_token_id=0, eos_token_id=1), loops=8
        )
        cls.ids = torch.tensor([[3, 4, 5, 6, 7, 0], [9, 10, 11, 12, 13, 14]])
        cls.mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
        cls.rows = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        cls.positions = torch.tensor([2, 3, 4, 2, 3, 4, 5])

    @torch.inference_mode()
    def test_complete_validation_and_native_model_restoration(self):
        original = self.model.model.layers
        result = self.adapter.validate(self.ids, self.mask, self.rows, self.positions)
        self.assertIs(self.model.model.layers, original)
        self.assertEqual(len(original), 36)
        self.assertTrue(result["native_damped_wrapper_replay_passed"])
        self.assertTrue(result["k1_dense_logits_match_exactly"])
        self.assertEqual(result["k1_native_dense_all_hidden_max_abs_error"], 0.0)
        self.assertGreater(result["early_final_mean_abs_logit_difference"], 0)
        self.assertGreater(result["looped_vs_dense_mean_abs_logit_difference"], 0)

    @torch.inference_mode()
    def test_every_layer_count_and_readout_position(self):
        counts = [0] * 36
        handles = []
        for index, layer in enumerate(self.model.model.layers):
            def record(module, args, output, index=index):
                counts[index] += 1
            handles.append(layer.register_forward_hook(record))
        try:
            values = self.adapter.forward_logits(
                self.ids, self.mask, self.rows, self.positions
            )
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts[:15], [1] * 15)
        self.assertEqual(counts[15:19], [8] * 4)
        # Each of the two outputs must traverse the entire tail.
        self.assertEqual(counts[19:], [2] * 17)
        for value in values.values():
            self.assertEqual(value.shape, (len(self.rows), 47))
            self.assertEqual(value.dtype, torch.float32)
        # Read every valid position, then gather our answer positions independently.
        all_rows, all_positions = self.mask.nonzero(as_tuple=True)
        full = self.adapter.forward_logits(
            self.ids, self.mask, all_rows, all_positions
        )
        mapping = {
            (int(row), int(position)): i
            for i, (row, position) in enumerate(zip(all_rows, all_positions))
        }
        indices = [mapping[int(row), int(pos)] for row, pos in zip(self.rows, self.positions)]
        for key in values:
            torch.testing.assert_close(values[key], full[key][indices], rtol=0, atol=0)

    @torch.inference_mode()
    def test_k1_complete_logits_equal_unmodified_dense(self):
        adapter = LoopedQwenAdapter.from_model(
            self.model, SimpleNamespace(pad_token_id=0, eos_token_id=1), loops=1
        )
        rows, positions = self.mask.nonzero(as_tuple=True)
        actual = adapter.forward_logits(self.ids, self.mask, rows, positions)
        dense = self.model(
            input_ids=self.ids, attention_mask=self.mask, use_cache=False
        ).logits.float()[rows, positions]
        for values in actual.values():
            torch.testing.assert_close(values, dense, rtol=0, atol=0)

    @torch.inference_mode()
    def test_reference_is_first_damped_substep_not_natural_block(self):
        actual = self.adapter.forward_logits(self.ids, self.mask, self.rows, self.positions)
        unmodified = self.model(
            input_ids=self.ids, attention_mask=self.mask, use_cache=False
        ).logits.float()[self.rows, self.positions]
        # Passing the unblended first g(h) through the tail would produce dense.
        self.assertFalse(torch.allclose(actual["reference"], unmodified, atol=1e-6, rtol=1e-6))
        # This checks both reference semantics and that ordinary dense predictions
        # have not been mislabeled as the eight-step looped baseline.
        self.assertFalse(torch.allclose(actual["baseline"], unmodified, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
