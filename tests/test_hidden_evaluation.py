"""CPU integration of Hidden scoring, native readout, resume, and saved audits."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from scripts import analyze
from scripts.evaluate import summarize
from src.adapter_huginn import HuginnAdapter
from src.evaluation import build_candidates, validate_documents
from src.loopcd import HIDDEN_MODES, MODES, guided_hidden
from src.registry import load_registry, resolve_run
from src.run_integrity import compare_resume_manifests, load_completed_scores
from src.suite_scoring import SuiteScorer


ROOT = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def __len__(self):
        return 64

    def encode(self, text, add_special_tokens=False):
        tokens = [3 + ord(character) % 61 for character in text]
        return [self.bos_token_id, *tokens] if add_special_tokens else tokens


class HiddenEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(8241)
        model_class = get_class_from_dynamic_module(
            "raven_modeling_minimal.RavenForCausalLM", str(ROOT / "models/Huginn-0125"),
            local_files_only=True,
        )
        config = model_class.config_class(
            n_embd=32, n_heads=2, n_layers=5, intermediate_size=64,
            n_layers_in_prelude=1, n_layers_in_recurrent_block=2, n_layers_in_coda=2,
            mean_recurrence=32, mean_backprop_depth=0, block_size=128,
            vocab_size=64, padding_multiple=1, bos_token_id=1, eos_token_id=2, pad_token_id=0,
            torch_dtype="float32",
        )
        cls.model = model_class(config).eval()
        with torch.no_grad():
            cls.model.transformer.ln_f.weight.copy_(torch.linspace(0.5, 1.5, 32))
        cls.adapter = HuginnAdapter.from_model(cls.model, TinyTokenizer(), loops=16, reference_step=7)
        cls.documents = [
            {"id": "q0", "question": "Wet?", "context": "Question: Wet?\nAnswer:",
             "choices": ["rain", "dry"], "labels": ["A", "B"], "gold": 0, "fewshot_ids": []},
            {"id": "q1", "question": "Hot?", "context": "Question: Hot?\nAnswer:",
             "choices": ["ice", "fire"], "labels": ["A", "B"], "gold": 1, "fewshot_ids": []},
        ]
        validate_documents(cls.documents, 2)
        cls.candidates, _ = build_candidates(
            cls.adapter.tokenizer, cls.documents, cls.adapter.max_length,
            bos_token_id=cls.adapter.bos_token_id,
        )
        cls.scorer = SuiteScorer(cls.adapter, [row["id"] for row in cls.documents], guidance_space="hidden")
        cls.tokenized = {
            (row.question_index, row.choice_index): (row.targets, len(row.input_ids))
            for row in cls.candidates
        }

    def write_scores(self, output, scores):
        path = output / "candidate_scores.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in scores))
        return path

    def summarized(self, output, scores, modes=HIDDEN_MODES):
        path = self.write_scores(output, scores)
        completed = load_completed_scores(path, self.candidates, modes=modes)
        with redirect_stdout(io.StringIO()):
            summarize(self.documents, completed, output, {}, modes=modes)
        return analyze.read_json(output / "summary.json")

    @torch.inference_mode()
    def test_hidden_candidate_scores_match_independent_native_raw_state_readout(self):
        # The Hidden branch must not obtain its result from logit extrapolation.
        with patch.object(self.adapter, "forward_logits", side_effect=AssertionError("wrong policy")):
            actual = self.scorer.score(self.candidates)
        inputs, mask, rows, positions, targets, keys = self.scorer.tensors(self.candidates)
        initial = self.adapter.initial_states(keys, inputs.shape[1])
        captured, updates = {}, 0

        def capture(module, args, output):
            nonlocal updates
            updates += 1
            if updates in (7, 16):
                captured[updates] = output.clone()

        handle = self.model.transformer.core_block[-1].register_forward_hook(capture)
        try:
            native = self.model(
                inputs, input_states=initial, num_steps=16, use_cache=False,
                output_details={"return_logits": False, "return_latents": False,
                                "return_head": True, "return_stats": False},
            )
        finally:
            handle.remove()
        self.assertEqual(updates, 16)
        hidden = self.model.transformer.ln_f(captured[16] + 0.5 * (captured[16] - captured[7]))
        frequencies = self.model.freqs_cis[:, :inputs.shape[1]]
        block_index = torch.tensor(0, dtype=torch.long)
        for block in self.model.transformer.coda:
            block_index -= 1
            hidden = block(hidden, frequencies, block_index, None, None)
        hidden = self.model.transformer.ln_f(hidden)
        hidden_logits = self.model.lm_head(hidden[rows, positions]).float()
        baseline_logits = self.model.lm_head(native.hidden_states[rows, positions]).float()
        independently_scored = {
            mode: logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1).tolist()
            for mode, logits in (("baseline", baseline_logits), ("loopcd_hidden", hidden_logits))
        }
        offset = 0
        for candidate, saved in zip(self.candidates, actual):
            count = len(candidate.targets)
            self.assertEqual(set(saved["scores"]), set(HIDDEN_MODES))
            for mode in HIDDEN_MODES:
                expected = independently_scored[mode][offset:offset + count]
                self.assertEqual(saved["token_logprobs"][mode], expected)
                self.assertEqual(saved["scores"][mode], sum(expected))
            offset += count
        self.assertGreater((hidden_logits - baseline_logits).abs().max().item(), 1e-5)
        self.assertNotEqual(independently_scored["loopcd_hidden"], independently_scored["baseline"])

    def test_hidden_saved_results_round_trip_and_corruption_rejection(self):
        scores = self.scorer.score(self.candidates)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = self.summarized(output, scores)
            audited = analyze.audit_scores(
                self.documents, self.tokenized, output, summary,
                modes=HIDDEN_MODES, guided=("loopcd_hidden",),
            )
            self.assertEqual(audited["candidate_count"], 4)
            self.assertEqual(audited["correct"], summary["correct"])
            self.assertEqual(set(audited["paired"]["acc"]), {"loopcd_hidden"})
            with self.assertRaisesRegex(ValueError, "score modes"):
                load_completed_scores(output / "candidate_scores.jsonl", self.candidates)
            damaged = copy.deepcopy(scores)
            damaged[0]["token_logprobs"]["loopcd_hidden"][0] -= 0.25
            self.write_scores(output, damaged)
            with self.assertRaisesRegex(ValueError, "token sum"):
                load_completed_scores(output / "candidate_scores.jsonl", self.candidates, modes=HIDDEN_MODES)
            with self.assertRaisesRegex(ValueError, "Token sum"):
                analyze.audit_scores(self.documents, self.tokenized, output, summary,
                                     modes=HIDDEN_MODES, guided=("loopcd_hidden",))
            self.write_scores(output, scores)
            predictions = analyze.read_jsonl(output / "predictions.jsonl")
            predictions[0]["predictions"]["acc_norm"]["loopcd_hidden"] ^= 1
            (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
            with self.assertRaisesRegex(ValueError, "argmax mismatch"):
                analyze.audit_scores(self.documents, self.tokenized, output, summary,
                                     modes=HIDDEN_MODES, guided=("loopcd_hidden",))

    def test_original_four_policy_evaluation_and_resume_remain_distinct(self):
        old_adapter = HuginnAdapter.from_model(self.model, self.adapter.tokenizer, loops=16)
        old_scorer = SuiteScorer(old_adapter, [doc["id"] for doc in self.documents])
        self.assertEqual(old_scorer.modes, MODES)
        scores = old_scorer.score(self.candidates)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = self.summarized(output, scores, modes=MODES)
            result = analyze.audit_scores(self.documents, self.tokenized, output, summary)
            self.assertEqual(result["candidate_count"], 4)
            self.assertEqual(set(summary["correct"]["acc"]), set(MODES))
            with self.assertRaisesRegex(ValueError, "score modes"):
                load_completed_scores(output / "candidate_scores.jsonl", self.candidates, modes=HIDDEN_MODES)

    def test_hidden_validation_evidence_integrates_with_independent_audits(self):
        validation = self.scorer.validate(self.candidates)
        config = resolve_run("arc_challenge", "huginn_16", guidance_space="hidden")["model"]
        evidence = analyze.audit_hidden_checks(validation, config)
        self.assertEqual(evidence["hidden_standalone_head_passes"], 1)
        native = analyze.legacy.validate_native_checks(validation, False, config, modes=HIDDEN_MODES)
        self.assertTrue(native["zero_guidance_identity"])
        for field, value, message in (
            ("hidden_standalone_coda_layer_calls", [2, 2], "output pass"),
            ("hidden_guidance_positions", "answer positions", "complete input"),
            ("hidden_manual_readout_max_abs_error", 0.01, "state mismatch"),
        ):
            damaged = copy.deepcopy(validation)
            damaged["hidden_checks"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                analyze.audit_hidden_checks(damaged, config)

    def test_registry_separates_hidden_standard_logits_and_half_depth_protocols(self):
        registry_before = load_registry()
        for task in ("arc_easy", "arc_challenge", "sciq", "piqa"):
            for model, reference in (("huginn_16", 7), ("huginn_32", 6)):
                normal = resolve_run(task, model)
                hidden = resolve_run(task, model, guidance_space="hidden")
                self.assertNotIn("reference_step", normal["model"])
                self.assertEqual(hidden["model"]["reference_step"], reference)
                self.assertEqual(hidden["model"]["fixed"], 0.5)
                self.assertEqual(hidden["output"], ROOT / "results/hidden" / task / model)
                self.assertEqual(normal, resolve_run(task, model))
            reduced = resolve_run(task, "huginn_16", "huginn_half_depth")
            self.assertEqual(reduced["model"]["reference_step"], 5)
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                resolve_run(task, "huginn_16", "huginn_half_depth", guidance_space="hidden")
        self.assertEqual(load_registry(), registry_before)
        with self.assertRaisesRegex(ValueError, "configured for"):
            resolve_run("arc_challenge", "ouro_14", guidance_space="hidden")
        with self.assertRaisesRegex(ValueError, "guidance_space"):
            resolve_run("arc_challenge", "huginn_16", guidance_space="unknown")

    def test_hidden_guidance_helper_is_immutable_and_preserves_floating_dtype(self):
        for dtype in (torch.float32, torch.bfloat16):
            final = torch.arange(8, dtype=dtype).reshape(1, 2, 4)
            reference = final.flip(-1)
            old_final, old_reference = final.clone(), reference.clone()
            result = guided_hidden(final, reference, 0.5)
            self.assertEqual(result.dtype, dtype)
            torch.testing.assert_close(result, final + 0.5 * (final - reference), atol=0, rtol=0)
            torch.testing.assert_close(final, old_final, atol=0, rtol=0)
            torch.testing.assert_close(reference, old_reference, atol=0, rtol=0)
        for strength in (-1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, "strength"):
                guided_hidden(final, reference, strength)
        for incompatible in (reference[:, :, :-1], reference.float(), torch.empty_like(reference, device="meta")):
            with self.assertRaisesRegex(ValueError, "shape, dtype, and device"):
                guided_hidden(final, incompatible, 0.5)

    def test_resume_manifest_rejects_changed_hidden_policy_and_source(self):
        manifest = {
            "name": "huginn_16", "configuration": {"loops": 16, "reference_step": 7},
            "documents": 2, "candidates": 4, "data_sha256": "data", "model_manifest_sha256": "model",
            "code_hashes": {"src/adapter_huginn.py": "adapter-v1"},
            "native_source_hashes": {"official.py": "native-v1"},
            "dtype": "float32", "float32_matmul_precision": "highest", "batch_size": 4,
            "token_budget": 128, "seed": 1234, "add_bos": True, "chat_template": False,
            "max_input_tokens": 64, "truncation": False, "torch": "test", "transformers": "test",
            "adapter_metadata": {"reference_recurrent_update": 7}, "guidance_space": "hidden",
            "scored_modes": list(HIDDEN_MODES),
        }
        compare_resume_manifests(manifest, copy.deepcopy(manifest))
        for field, changed in (("guidance_space", "logits"), ("scored_modes", list(MODES)),
                               ("code_hashes", {"src/adapter_huginn.py": "adapter-v2"})):
            current = copy.deepcopy(manifest)
            current[field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                compare_resume_manifests(manifest, current)

    def test_source_audit_rejects_resealed_wrong_hidden_reference_and_changed_adapter(self):
        resolved = resolve_run("arc_challenge", "huginn_16", guidance_space="hidden")
        _, tasks = load_registry()
        paths = (
            "scripts/evaluate.py", "src/suite_scoring.py", "src/loopcd.py",
            "src/adapter_huginn.py", "src/run_integrity.py", "src/registry.py",
            "src/evaluation.py", "configs/models.json", "configs/tasks.json", "configs/hidden.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            snapshot = output / "source_snapshot"
            snapshot.mkdir(parents=True)
            for path in paths:
                (snapshot / Path(path).name).write_bytes((ROOT / path).read_bytes())
            manifest = {
                "context_policy": tasks["context_policy"],
                "code_hashes": {path: analyze.digest(snapshot / Path(path).name) for path in paths},
            }
            with patch.object(analyze, "ROOT", root):
                verified = analyze.audit_sources(output, manifest, "huginn_16", "arc_challenge", resolved)
                self.assertEqual(len(verified), len(paths))
                hidden_path = snapshot / "hidden.json"
                original = hidden_path.read_bytes()
                hidden = json.loads(original)
                hidden["models"]["huginn_16"]["reference_step"] = 6
                hidden_path.write_text(json.dumps(hidden))
                manifest["code_hashes"]["configs/hidden.json"] = analyze.digest(hidden_path)
                with self.assertRaisesRegex(ValueError, "Snapshot Hidden settings"):
                    analyze.audit_sources(output, manifest, "huginn_16", "arc_challenge", resolved)
                hidden_path.write_bytes(original)
                manifest["code_hashes"]["configs/hidden.json"] = analyze.digest(hidden_path)
                (snapshot / "adapter_huginn.py").write_text("modified adapter\n")
                with self.assertRaisesRegex(ValueError, "changed source snapshot"):
                    analyze.audit_sources(output, manifest, "huginn_16", "arc_challenge", resolved)


if __name__ == "__main__":
    unittest.main()
