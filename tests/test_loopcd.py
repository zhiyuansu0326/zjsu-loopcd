"""Independent CPU checks against the checkpoint's actual Ouro implementation."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.loopcd import Candidate, MODES, OuroScorer, encode_candidate, guided_logits


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/Ouro-1.4B"


class GuidanceTests(unittest.TestCase):
    def test_fixed_and_adaptive_use_float32_and_full_vocabulary(self):
        final = torch.tensor([[0, 1, 2, 3], [4, 0, 0, 0]], dtype=torch.bfloat16)
        early = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]], dtype=torch.bfloat16)
        final64, early64 = final.double(), early.double()
        fixed = final64 + 0.5 * (final64 - early64)
        probabilities = final64.exp() / final64.exp().sum(-1, keepdim=True)
        sorted_probabilities = probabilities.sort(-1, descending=True).values
        omega = 1 - (sorted_probabilities[:, 0] - sorted_probabilities[:, 1])
        adaptive = final64 + omega[:, None] * (final64 - early64)
        for expected, actual in (
            (fixed, guided_logits(final, early, strength=0.5)),
            (adaptive, guided_logits(final, early, strength=1, adaptive=True)),
        ):
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            guided_logits(final, early, strength=0), final.float(), atol=0, rtol=0
        )


class TokenBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT, local_files_only=True, trust_remote_code=True
        )

    def test_real_tokenizer_continuation_and_teacher_forcing(self):
        for context in ("Question: What moves?\nAnswer:", "Question: What moves?\nAnswer: \n"):
            choice = "the rolling red ball."
            candidate = encode_candidate(self.tokenizer, context, choice, 8, 2)
            stripped = context.rstrip()
            moved_whitespace = context[len(stripped):]
            full = self.tokenizer.encode(
                stripped + moved_whitespace + " " + choice, add_special_tokens=False
            )
            prefix = self.tokenizer.encode(stripped, add_special_tokens=False)
            self.assertEqual(full[: len(prefix)], prefix)
            self.assertEqual(candidate.targets, full[len(prefix):])
            self.assertEqual(candidate.input_ids, full[:-1])
            self.assertEqual(candidate.positions[0], len(prefix) - 1)
            self.assertEqual(candidate.positions[-1], len(full) - 2)
            self.assertGreater(len(candidate.targets), 1)
            for position, target in zip(candidate.positions, candidate.targets, strict=True):
                self.assertEqual(full[position + 1], target)
            self.assertEqual(self.tokenizer.encode(stripped), prefix)


class NativeOuroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(739)
        config = AutoConfig.from_pretrained(
            CHECKPOINT, local_files_only=True, trust_remote_code=True
        )
        config.update({
            "hidden_size": 32,
            "intermediate_size": 64,
            "head_dim": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "torch_dtype": "float32",
        })
        cls.model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, attn_implementation="sdpa"
        ).eval()
        # Nonuniform trained-like RMSNorm weights make accidental renormalization observable.
        with torch.no_grad():
            cls.model.model.norm.weight.copy_(torch.linspace(0.6, 1.5, config.hidden_size))
        cls.scorer = OuroScorer(
            cls.model, SimpleNamespace(pad_token_id=0, eos_token_id=0)
        )
        # Each sequence contains a known context followed by a multi-token continuation.
        cls.full_sequences = ([3, 4, 5, 7, 9, 11], [2, 6, 8, 10, 12, 14, 16, 18])
        cls.context_lengths = (3, 4)
        cls.candidates = [
            Candidate(i, i + 1, list(full[:-1]), list(range(n - 1, len(full) - 1)), list(full[n:]))
            for i, (full, n) in enumerate(zip(cls.full_sequences, cls.context_lengths))
        ]

    @torch.inference_mode()
    def test_every_gathered_answer_position_matches_native_early_and_final(self):
        actual = self.scorer.score(self.candidates)
        for row, full, context_length in zip(
            actual, self.full_sequences, self.context_lengths, strict=True
        ):
            inputs = torch.tensor([full[:-1]])
            kwargs = dict(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                use_cache=False, use_weighted_exit=False,
            )
            final = self.model(**kwargs, exit_at_step=3).logits[0].float()
            early = self.model(**kwargs, exit_at_step=0).logits[0].float()
            probs = final.softmax(-1).sort(-1, descending=True).values
            omega = 1 - (probs[:, 0] - probs[:, 1])
            logits = {
                "baseline": final,
                "reference": early,
                "loopcd_fixed": final + 0.5 * (final - early),
                "loopcd_adaptive": final + omega[:, None] * (final - early),
            }
            for mode, values in logits.items():
                log_probs = values.log_softmax(-1)
                # Derive positions from the original context, independently of Candidate.positions.
                expected = torch.stack([
                    log_probs[target_position - 1, full[target_position]]
                    for target_position in range(context_length, len(full))
                ])
                observed = torch.tensor(row["token_logprobs"][mode])
                torch.testing.assert_close(observed, expected, atol=2e-6, rtol=1e-6)
                self.assertAlmostEqual(row["scores"][mode], expected.double().sum().item(), places=5)
            self.assertFalse(torch.equal(final, early))

    def test_right_padding_does_not_change_any_policy_score(self):
        batched = self.scorer.score(self.candidates)
        for candidate, batch_row in zip(self.candidates, batched, strict=True):
            single_row = self.scorer.score([candidate])[0]
            for mode in MODES:
                torch.testing.assert_close(
                    torch.tensor(batch_row["token_logprobs"][mode]),
                    torch.tensor(single_row["token_logprobs"][mode]),
                    atol=2e-6, rtol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
