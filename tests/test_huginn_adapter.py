"""CPU tests using a small model built from the exact released Huginn code."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from src.adapter_huginn import HuginnAdapter


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/Huginn-0125"


class HuginnAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(761)
        model_class = get_class_from_dynamic_module(
            "raven_modeling_minimal.RavenForCausalLM", str(CHECKPOINT),
            local_files_only=True,
        )
        config = model_class.config_class(
            n_embd=32, n_heads=2, n_layers=5, intermediate_size=64,
            n_layers_in_prelude=1, n_layers_in_recurrent_block=2,
            n_layers_in_coda=2, mean_recurrence=32, mean_backprop_depth=0,
            block_size=256, vocab_size=64, padding_multiple=1,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            torch_dtype="float32",
        )
        cls.model = model_class(config).eval()
        # Nonuniform final norm weights expose missing or repeated normalization.
        with torch.no_grad():
            cls.model.transformer.ln_f.weight.copy_(torch.linspace(0.6, 1.4, 32))
        cls.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)
        cls.adapter = HuginnAdapter.from_model(cls.model, cls.tokenizer, loops=16)
        cls.inputs = torch.tensor([[1, 3, 4, 5, 6, 7, 0, 0], [1, 3, 4, 5, 8, 9, 10, 11]])
        cls.mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
        cls.rows = torch.tensor([0, 0, 1, 1, 1])
        cls.positions = torch.tensor([3, 5, 3, 5, 7])
        cls.keys = ["question-A", "question-A"]

    @torch.inference_mode()
    def test_reference_is_complete_first_update_and_full_coda(self):
        initial = self.adapter.initial_states(self.keys, self.inputs.shape[1])
        native_first = self.model(
            self.inputs, input_states=initial, num_steps=1, use_cache=False,
        ).logits[self.rows, self.positions]
        actual = self.adapter.forward_logits(
            self.inputs, self.mask, self.rows, self.positions, self.keys
        )
        torch.testing.assert_close(actual["reference"], native_first, atol=2e-5, rtol=2e-5)
        frequencies = self.model.freqs_cis[:, : self.inputs.shape[1]]
        noise_readout = self.model.lm_head(
            self.adapter._readout_hidden(initial, frequencies)[self.rows, self.positions]
        ).float()
        self.assertFalse(torch.allclose(actual["reference"], noise_readout))
        self.assertEqual(actual["baseline"].dtype, torch.float32)

    def test_native_final_and_zero_guidance_for_both_paper_depths(self):
        for depth in (16, 32):
            adapter = HuginnAdapter.from_model(self.model, self.tokenizer, loops=depth)
            evidence = adapter.validate(self.inputs, self.mask, self.rows, self.positions, self.keys)
            self.assertEqual(evidence["native_final_hidden_max_abs_error"], 0.0)
            self.assertEqual(evidence["native_same_shape_logits_max_abs_error"], 0.0)
            self.assertEqual(evidence["native_reference_hidden_max_abs_error"], 0.0)
            self.assertEqual(evidence["native_reference_same_shape_logits_max_abs_error"], 0.0)
            self.assertTrue(evidence["zero_guidance_matches_native_same_shape_exactly"])
            json.dumps(evidence)

    @torch.inference_mode()
    def test_reduced_depth_reference_matches_native_fifth_update(self):
        adapter = HuginnAdapter.from_model(
            self.model, self.tokenizer, loops=16, reference_step=5
        )
        evidence = adapter.validate(self.inputs, self.mask, self.rows, self.positions, self.keys)
        self.assertEqual(evidence["reference_recurrent_update"], 5)
        self.assertEqual(evidence["native_reference_hidden_max_abs_error"], 0.0)
        self.assertEqual(evidence["native_reference_same_shape_logits_max_abs_error"], 0.0)
        initial = adapter.initial_states(self.keys, self.inputs.shape[1])
        native_reference = self.model(
            self.inputs, input_states=initial, num_steps=5, use_cache=False,
        ).logits[self.rows, self.positions].float()
        result = adapter.forward_logits(
            self.inputs, self.mask, self.rows, self.positions, self.keys,
            initial_states=initial,
        )
        torch.testing.assert_close(result["reference"], native_reference, atol=2e-5, rtol=2e-5)
        first_reference = self.adapter.forward_logits(
            self.inputs, self.mask, self.rows, self.positions, self.keys,
            initial_states=initial,
        )
        torch.testing.assert_close(result["baseline"], first_reference["baseline"], atol=0, rtol=0)
        self.assertFalse(torch.equal(result["reference"], first_reference["reference"]))

    def test_baseline_only_has_one_readout_and_projection(self):
        adapter = HuginnAdapter.from_model(
            self.model, self.tokenizer, loops=16, reference_step=5
        )
        initial = adapter.initial_states(self.keys, self.inputs.shape[1])
        original = initial.clone()
        guided = adapter.forward_logits(
            self.inputs, self.mask, self.rows, self.positions, self.keys,
            initial_states=initial,
        )
        projections = []
        handle = self.model.lm_head.register_forward_hook(
            lambda module, inputs, output: projections.append(output.shape)
        )
        try:
            with patch.object(adapter, "_readout_hidden", wraps=adapter._readout_hidden) as readout:
                with patch.object(adapter, "initial_states", side_effect=AssertionError("unexpected RNG")):
                    baseline = adapter.forward_baseline_logits(
                        self.inputs, self.mask, self.rows, self.positions, self.keys,
                        initial_states=initial,
                    )
                self.assertEqual(readout.call_count, 1)
            self.assertEqual(len(projections), 1)
        finally:
            handle.remove()
        torch.testing.assert_close(baseline, guided["baseline"], atol=0, rtol=0)
        torch.testing.assert_close(initial, original, atol=0, rtol=0)

    def test_reference_step_bounds_and_final_step(self):
        for step in (0, -1, 17, True, 5.0, "5", None):
            with self.subTest(reference_step=step):
                with self.assertRaisesRegex(ValueError, "reference_step"):
                    HuginnAdapter.from_model(self.model, self.tokenizer, loops=16, reference_step=step)
        adapter = HuginnAdapter.from_model(self.model, self.tokenizer, loops=5, reference_step=5)
        evidence = adapter.validate(self.inputs, self.mask, self.rows, self.positions, self.keys)
        self.assertEqual(evidence["early_final_mean_abs_logit_difference"], 0.0)
        self.assertEqual(evidence["reference_recurrent_update"], 5)

    def test_supplied_initial_states_must_match_shape_and_dtype(self):
        initial = self.adapter.initial_states(self.keys, self.inputs.shape[1])
        for method in (self.adapter.forward_logits, self.adapter.forward_baseline_logits):
            with self.assertRaisesRegex(ValueError, "shape"):
                method(
                    self.inputs, self.mask, self.rows, self.positions, self.keys,
                    initial_states=initial[:, :-1],
                )
            with self.assertRaisesRegex(ValueError, "device and dtype"):
                method(
                    self.inputs, self.mask, self.rows, self.positions, self.keys,
                    initial_states=initial.double(),
                )

    def test_noise_prefix_is_invariant_to_length_batch_order_and_global_rng(self):
        state = self.adapter.initial_states(["A", "B", "A"], 131)
        torch.manual_seed(9999)
        shorter = self.adapter.initial_states(["B", "A"], 63)
        torch.testing.assert_close(state[0], state[2], atol=0, rtol=0)
        torch.testing.assert_close(state[0, :63], shorter[1], atol=0, rtol=0)
        torch.testing.assert_close(state[1, :63], shorter[0], atol=0, rtol=0)
        self.assertFalse(torch.equal(state[0], state[1]))
        bound = 3 * self.model.config.init_values["std"] * self.model.emb_scale
        self.assertLessEqual(state.abs().max().item(), bound + 1e-6)

    def test_right_padding_and_option_batching_preserve_scored_prefix(self):
        batch = self.adapter.forward_logits(self.inputs, self.mask, self.rows, self.positions, self.keys)
        first_rows = torch.zeros(2, dtype=torch.long)
        single = self.adapter.forward_logits(
            self.inputs[:1, :6], self.mask[:1, :6], first_rows, self.positions[:2], self.keys[:1]
        )
        for mode in ("baseline", "reference"):
            torch.testing.assert_close(batch[mode][:2], single[mode], atol=2e-5, rtol=2e-5)
        changed_padding = self.inputs.clone()
        changed_padding[0, 6:] = torch.tensor([30, 31])
        changed = self.adapter.forward_logits(
            changed_padding, self.mask, self.rows, self.positions, self.keys
        )
        for mode in ("baseline", "reference"):
            torch.testing.assert_close(batch[mode], changed[mode], atol=0, rtol=0)

    def test_rejects_left_padding_and_missing_question_keys(self):
        left_padded_mask = self.mask.clone()
        left_padded_mask[0] = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "right padding"):
            self.adapter.forward_logits(self.inputs, left_padded_mask, self.rows, self.positions, self.keys)
        with self.assertRaisesRegex(ValueError, "sample_key"):
            self.adapter.forward_logits(self.inputs, self.mask, self.rows, self.positions)

    def test_official_tokenizer_prepends_exactly_one_bos(self):
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        text = "Question: What moves?\nAnswer:"
        plain = tokenizer.encode(text, add_special_tokens=False)
        with_specials = tokenizer.encode(text, add_special_tokens=True)
        self.assertEqual(with_specials, [tokenizer.bos_token_id] + plain)
        self.assertEqual(tokenizer.bos_token_id, 65504)
        self.assertTrue(self.adapter.add_bos)


if __name__ == "__main__":
    unittest.main()
