"""CPU checks against tiny models constructed from the pinned official source."""
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers import PreTrainedTokenizerFast

from src.adapter_parcae import ParcaeAdapter, native_classes


class NativeParcaeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(197)
        config_class, model_class = native_classes()
        config = config_class(
            name="tiny-parcae", block_size=32, n_embd=64, intermediate_size=128,
            num_attention_heads=2, num_key_value_heads=2,
            vocab_size=64, padded_vocab_size=64, padding_multiple=1,
            n_layers_in_prelude=1, n_layers_in_recurrent_block=2, n_layers_in_coda=1,
            recurrent_embedding_dimension=64, recurrent_intermediation_embedding_dimension=128,
            mean_recurrence=8, mean_backprop_depth=2, init_strategy="scaled-zero",
            init_orthogonal=False, state_init="like-init", prelude_norm=True,
            norm_class_name="RMSNorm", mlp_class_name="BaseMLP", nonlin_name="ReLU2",
            qk_norm=True, tie_embeddings=True, use_fused_head="pytorch",
            skip_initialization=True,
        )
        model = model_class(config).eval()
        # Initialize the fixture directly: upstream scaled-zero initialization
        # asks torch.trunc_normal_ for std=0, unsupported by the installed 2.6.
        # Nonzero residuals make every recurrence, coda, and value embedding
        # observable in the independent native-forward comparison.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("A_log"):
                    parameter.zero_()
                elif name.endswith("dt_bias"):
                    parameter.fill_(-0.5)
                elif name.endswith("adapter.B"):
                    torch.nn.init.eye_(parameter)
                elif parameter.ndim == 2:
                    parameter.normal_(std=0.04)
                else:
                    parameter.fill_(1.0)
        cls.adapter = ParcaeAdapter.__new__(ParcaeAdapter)
        cls.adapter._configure(model, None, 8)
        cls.ids = torch.tensor([[4, 5, 6, 7, 8, 9], [4, 5, 6, 1, 0, 0]])
        cls.mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        cls.rows = torch.tensor([0, 0, 0, 1, 1])
        cls.positions = torch.tensor([2, 3, 5, 2, 3])
        cls.keys = ["same-question", "same-question"]

    def test_native_final_first_loop_and_all_selected_positions(self):
        report = self.adapter.validate(
            self.ids, self.mask, self.rows, self.positions, self.keys
        )
        self.assertEqual(report["recurrent_updates_in_forward"], 8)
        self.assertLess(report["native_final_logits_max_abs_error"], 3e-4)
        self.assertLess(report["native_first_loop_logits_max_abs_error"], 3e-4)
        self.assertTrue(report["zero_guidance_matches_baseline_exactly"])
        self.assertTrue(report["latent_prefix_invariant_to_length_and_batch"])
        self.assertGreater(report["early_final_mean_abs_logit_difference"], 1e-5)

    def test_readouts_execute_projection_and_complete_coda_without_extra_loops(self):
        model = self.adapter.model
        with patch.object(model, "update_recurrent_state", wraps=model.update_recurrent_state) as recurrent:
            with patch.object(model.transformer.C, "forward", wraps=model.transformer.C.forward) as project:
                with patch.object(
                    model.transformer.coda[0], "forward", wraps=model.transformer.coda[0].forward
                ) as coda:
                    self.adapter.forward_logits(
                        self.ids, self.mask, self.rows, self.positions, self.keys
                    )
        self.assertEqual(recurrent.call_count, 8)
        self.assertEqual(project.call_count, 2)
        self.assertEqual(coda.call_count, 2)

    def test_initialization_is_prefix_stable_and_has_native_distribution(self):
        a = self.adapter
        small = a.initial_state((1, 5, 64), ["q"])
        large = a.initial_state((3, 13, 64), ["other", "q", "q"])
        torch.testing.assert_close(small[0], large[1, :5], rtol=0, atol=0)
        torch.testing.assert_close(small[0], large[2, :5], rtol=0, atol=0)
        self.assertFalse(torch.equal(large[0], large[1]))
        samples = a.initial_state((1, 2048, 64), ["distribution"]).float()
        standardized = samples / (a.initial_std * a.model.emb_scale)
        self.assertLessEqual(standardized.abs().max().item(), 3.0)
        self.assertLess(abs(standardized.mean().item()), 0.015)
        normal_pdf_at_three = math.exp(-4.5) / math.sqrt(2 * math.pi)
        normal_mass = math.erf(3 / math.sqrt(2))
        expected_variance = 1 - 6 * normal_pdf_at_three / normal_mass
        self.assertLess(abs(standardized.var().item() - expected_variance), 0.02)

    def test_mask_validation_prevents_silent_left_padding_or_padded_scoring(self):
        with self.assertRaisesRegex(ValueError, "right-padded"):
            self.adapter.forward_logits(
                self.ids, torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
                self.rows, self.positions, self.keys,
            )
        with self.assertRaisesRegex(ValueError, "padded"):
            self.adapter.forward_logits(self.ids, self.mask, [1], [5], self.keys)
        with self.assertRaisesRegex(ValueError, "sample_keys"):
            self.adapter.forward_logits(self.ids, self.mask, self.rows, self.positions)

    def test_native_like_init_is_really_stochastic(self):
        model = self.adapter.model
        shape = (1, 5, model.config.n_embd)
        first = model.initialize_state(torch.zeros(shape))
        second = model.initialize_state(torch.zeros(shape))
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(self.adapter.stochastic)


class ParcaeTokenizerTests(unittest.TestCase):
    def test_official_tokenizer_has_no_automatic_bos_and_keeps_answer_boundary(self):
        root = Path(__file__).resolve().parents[1]
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(root / "models/Parcae-370M/tokenizer.json"),
            add_bos_token=False, add_eos_token=False,
        )
        text = "Question: What is frozen water called?\nAnswer:"
        prefix = tokenizer.encode(text, add_special_tokens=False)
        self.assertEqual(tokenizer.encode(text), prefix)
        whole = tokenizer.encode(text + " ice", add_special_tokens=False)
        self.assertEqual(whole[:len(prefix)], prefix)
        self.assertGreater(len(whole), len(prefix))
        self.assertEqual(tokenizer.vocab_size, 32768)


if __name__ == "__main__":
    unittest.main()
