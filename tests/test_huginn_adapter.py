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

    def test_hidden_guidance_matches_native_states_and_one_output_pass(self):
        for loops, reference in ((16, 7), (32, 6)):
            with self.subTest(loops=loops, reference=reference):
                adapter = HuginnAdapter.from_model(
                    self.model, self.tokenizer, loops=loops, reference_step=reference,
                )
                metadata = adapter.metadata()
                evidence = adapter.validate_hidden(
                    self.inputs, self.mask, self.rows, self.positions, self.keys, strength=0.5,
                )
                for field in ("hidden_native_final_raw_max_abs_error",
                              "hidden_native_reference_raw_max_abs_error",
                              "hidden_manual_readout_max_abs_error"):
                    self.assertEqual(evidence[field], 0.0)
                self.assertTrue(evidence["hidden_zero_strength_matches_native_exactly"])
                self.assertTrue(evidence["hidden_paired_matches_standalone_exactly"])
                self.assertTrue(evidence["hidden_initial_states_unchanged"])
                self.assertEqual(evidence["hidden_standalone_recurrent_updates"], loops)
                self.assertEqual(evidence["hidden_standalone_coda_layer_calls"], [1, 1])
                self.assertEqual(evidence["hidden_standalone_head_passes"], 1)
                self.assertEqual(adapter.metadata(), metadata)
                json.dumps(evidence)

    @torch.inference_mode()
    def test_hidden_blend_precedes_normalization_and_coda_for_every_context_position(self):
        adapter = HuginnAdapter.from_model(self.model, self.tokenizer, loops=16, reference_step=7)
        initial = adapter.initial_states(self.keys, self.inputs.shape[1])
        final, reference, frequencies = adapter._recurrent_states(self.inputs, initial)
        actual = adapter.forward_hidden_logits(
            self.inputs, self.mask, self.rows, self.positions, self.keys,
            strength=0.5, initial_states=initial,
        )
        manual_raw = final + 0.5 * (final - reference)
        manual = self.model.lm_head(
            adapter._readout_hidden(manual_raw, frequencies)[self.rows, self.positions]
        ).float()
        torch.testing.assert_close(actual, manual, atol=0, rtol=0)
        partial = final.clone()
        partial[self.rows, self.positions] = manual_raw[self.rows, self.positions]
        answer_only = self.model.lm_head(
            adapter._readout_hidden(partial, frequencies)[self.rows, self.positions]
        ).float()
        self.assertGreater((actual - answer_only).abs().max().item(), 1e-6)
        final_after_coda = adapter._readout_hidden(final, frequencies)
        reference_after_coda = adapter._readout_hidden(reference, frequencies)
        after_coda = self.model.lm_head((
            final_after_coda + 0.5 * (final_after_coda - reference_after_coda)
        )[self.rows, self.positions]).float()
        self.assertGreater((actual - after_coda).abs().max().item(), 1e-6)
        final_norm = self.model.transformer.ln_f(final)
        reference_norm = self.model.transformer.ln_f(reference)
        wrong_normalized_input = self.model.lm_head(
            adapter._readout_hidden(
                final_norm + 0.5 * (final_norm - reference_norm), frequencies,
            )[self.rows, self.positions]
        ).float()
        self.assertGreater((actual - wrong_normalized_input).abs().max().item(), 1e-6)

    @torch.inference_mode()
    def test_hidden_comparison_preserves_states_and_uses_one_recurrent_trajectory(self):
        adapter = HuginnAdapter.from_model(self.model, self.tokenizer, loops=16, reference_step=7)
        initial = adapter.initial_states(self.keys, self.inputs.shape[1])
        original_initial = initial.clone()
        recurrent = adapter._recurrent_states
        tracked = {}

        def track_states(*args, **kwargs):
            final, reference, frequencies = recurrent(*args, **kwargs)
            tracked.update(final=final, reference=reference,
                           original_final=final.clone(), original_reference=reference.clone())
            return final, reference, frequencies

        with patch.object(adapter, "_recurrent_states", side_effect=track_states) as trajectory:
            with patch.object(adapter, "_readout_hidden", wraps=adapter._readout_hidden) as readout:
                with patch.object(adapter, "initial_states", side_effect=AssertionError("unexpected RNG")):
                    paired = adapter.forward_hidden_comparison_logits(
                        self.inputs, self.mask, self.rows, self.positions, self.keys,
                        initial_states=initial,
                    )
            self.assertEqual(trajectory.call_count, 1)
            self.assertEqual(readout.call_count, 2)
        for name in ("final", "reference"):
            torch.testing.assert_close(tracked[name], tracked[f"original_{name}"], atol=0, rtol=0)
        torch.testing.assert_close(initial, original_initial, atol=0, rtol=0)
        baseline = adapter.forward_baseline_logits(
            self.inputs, self.mask, self.rows, self.positions, self.keys, initial_states=initial,
        )
        torch.testing.assert_close(paired["baseline"], baseline, atol=0, rtol=0)

    def test_hidden_padding_and_option_batching_preserve_scored_prefix(self):
        adapter = HuginnAdapter.from_model(self.model, self.tokenizer, loops=16, reference_step=7)
        batch = adapter.forward_hidden_logits(self.inputs, self.mask, self.rows, self.positions, self.keys)
        single = adapter.forward_hidden_logits(
            self.inputs[:1, :6], self.mask[:1, :6], torch.zeros(2, dtype=torch.long),
            self.positions[:2], self.keys[:1],
        )
        torch.testing.assert_close(batch[:2], single, atol=2e-5, rtol=2e-5)
        changed_padding = self.inputs.clone()
        changed_padding[0, 6:] = torch.tensor([30, 31])
        changed = adapter.forward_hidden_logits(
            changed_padding, self.mask, self.rows, self.positions, self.keys,
        )
        torch.testing.assert_close(batch, changed, atol=0, rtol=0)

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
        for method in (self.adapter.forward_logits, self.adapter.forward_baseline_logits,
                       self.adapter.forward_hidden_logits, self.adapter.forward_hidden_comparison_logits):
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
