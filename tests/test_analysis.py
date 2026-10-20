"""Audit failure cases and independent statistical/normalization checks."""

import copy
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze


class CharacterTokenizer:
    """Predictable Unicode-character tokens for causal-boundary tests."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[ord(character) for character in text])


class PrecisionDecisionTests(unittest.TestCase):
    def test_only_evidenced_pre_scoring_precision_upgrade_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "results/multitask"
            logs = directory / "logs"
            logs.mkdir(parents=True)
            log = logs / "failed.err"
            log.write_text("AssertionError: Batch/single numeric mismatch\n")
            entry = {
                "default": "high", "selected": "highest",
                "failure_log": str(log.relative_to(root)),
                "failure_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                "formal_candidates_before_retry": 0,
                "validation_tolerances_unchanged": True,
            }
            ledger = directory / "precision_decisions.json"
            with patch.object(analyze, "ROOT", root):
                self.assertEqual(analyze.precision_decision("piqa", "ouro_14",
                                 {"matmul_precision": "high"}), ("high", None))
                ledger.write_text(json.dumps({"overrides": {"piqa/ouro_14": entry}}))
                self.assertEqual(analyze.precision_decision("piqa", "ouro_14",
                                 {"matmul_precision": "high"})[0], "highest")
                for field, value in (("selected", "medium"), ("formal_candidates_before_retry", 1),
                                     ("validation_tolerances_unchanged", False),
                                     ("failure_log_sha256", "bad")):
                    broken = dict(entry)
                    broken[field] = value
                    ledger.write_text(json.dumps({"overrides": {"piqa/ouro_14": broken}}))
                    with self.assertRaises((AssertionError, ValueError)):
                        analyze.precision_decision("piqa", "ouro_14", {"matmul_precision": "high"})


def fixture():
    # Unicode length, token count and a prefix space must not be conflated.
    documents = [{
        "id": "q0", "question": "?", "choices": ["é", "ab"],
        "labels": ["A", "B"], "gold": 1, "context": "Question: ?\nAnswer:",
    }]
    tokens = {(0, 0): ([17, 18], 20), (0, 1): ([27, 28], 20)}
    records = []
    for choice, values in enumerate([[-1., -1.], [-1., -2.]]):
        records.append({
            "question_index": 0, "choice_index": choice,
            "target_ids": tokens[0, choice][0][:], "input_length": 20,
            "token_logprobs": {mode: values[:] for mode in analyze.MODES},
            "scores": {mode: sum(values) for mode in analyze.MODES},
        })
    predictions = [{
        **{key: documents[0][key] for key in ("id", "question", "choices", "labels", "gold")},
        "scores": {mode: [-2., -3.] for mode in analyze.MODES},
        "scores_per_character": {mode: [-2., -1.5] for mode in analyze.MODES},
        "predictions": {
            "acc": {mode: 0 for mode in analyze.MODES},
            "acc_norm": {mode: 1 for mode in analyze.MODES},
        },
    }]
    summary = {
        "n": 1,
        "correct": {
            "acc": {mode: 0 for mode in analyze.MODES},
            "acc_norm": {mode: 1 for mode in analyze.MODES},
        },
        "accuracy_percent": {
            "acc": {mode: 0. for mode in analyze.MODES},
            "acc_norm": {mode: 100. for mode in analyze.MODES},
        },
    }
    return documents, tokens, records, predictions, summary


def write_records(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class ScoreAuditTests(unittest.TestCase):
    def audit(self, change=None):
        docs, tokens, records, predictions, summary = fixture()
        if change is not None:
            change(docs, tokens, records, predictions, summary)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_records(output / "candidate_scores.jsonl", records)
            write_records(output / "predictions.jsonl", predictions)
            return analyze.audit_scores(docs, tokens, output, summary)

    def test_unicode_character_normalization_can_change_winner(self):
        result = self.audit()
        self.assertEqual(result["correct"]["acc"]["baseline"], 0)
        self.assertEqual(result["correct"]["acc_norm"]["baseline"], 1)
        self.assertEqual(result["answer_token_count_per_policy"], 4)
        self.assertEqual(result["paired"]["acc"]["loopcd_fixed"]["mcnemar_exact_two_sided_p"], 1.)

    def test_rejects_incomplete_duplicate_or_damaged_candidate_records(self):
        changes = {
            "missing": lambda d, t, r, p, s: r.pop(),
            "duplicate": lambda d, t, r, p, s: r.append(copy.deepcopy(r[0])),
            "target": lambda d, t, r, p, s: r[0]["target_ids"].__setitem__(0, 999),
            "length": lambda d, t, r, p, s: r[0].__setitem__("input_length", 19),
            "sum": lambda d, t, r, p, s: r[0]["scores"].__setitem__("baseline", -1.),
            "missing_token": lambda d, t, r, p, s: r[0]["token_logprobs"]["reference"].pop(),
            "nan": lambda d, t, r, p, s: r[0]["token_logprobs"]["baseline"].__setitem__(0, float("nan")),
            "bool_token": lambda d, t, r, p, s: r[0]["target_ids"].__setitem__(0, True),
            "bool_logp": lambda d, t, r, p, s: r[0]["token_logprobs"]["baseline"].__setitem__(0, False),
            "positive_logp": lambda d, t, r, p, s: r[0]["token_logprobs"]["baseline"].__setitem__(0, .1),
            "wrong_answer": lambda d, t, r, p, s: p[0]["predictions"]["acc_norm"].__setitem__("baseline", 0),
            "wrong_summary": lambda d, t, r, p, s: s["correct"]["acc_norm"].__setitem__("baseline", 0),
        }
        for name, change in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.audit(change)

    def test_ties_choose_first_option(self):
        def tie(docs, tokens, records, predictions, summary):
            docs[0]["choices"] = ["a", "b"]
            predictions[0]["choices"] = ["a", "b"]
            for row in records:
                row["token_logprobs"] = {mode: [-1., -1.] for mode in analyze.MODES}
                row["scores"] = {mode: -2. for mode in analyze.MODES}
            for field in ("scores", "scores_per_character"):
                predictions[0][field] = {mode: [-2., -2.] for mode in analyze.MODES}
            for metric in analyze.METRICS:
                predictions[0]["predictions"][metric] = {mode: 0 for mode in analyze.MODES}
                summary["correct"][metric] = {mode: 0 for mode in analyze.MODES}
                summary["accuracy_percent"][metric] = {mode: 0. for mode in analyze.MODES}
        result = self.audit(tie)
        self.assertEqual(result["correct"]["acc_norm"]["baseline"], 0)


class EncodingAuditTests(unittest.TestCase):
    def test_bos_whitespace_and_left_truncation_preserve_every_target(self):
        docs = [{"context": "Question: ?\nAnswer:  ", "choices": ["é", "long"]}]
        full, no_drop = analyze.retokenize(docs, CharacterTokenizer(), True, 100)
        short, drop = analyze.retokenize(docs, CharacterTokenizer(), True, 8)
        self.assertEqual(full[0, 0][0], [ord(c) for c in "   é"])
        self.assertEqual(full[0, 0][0], short[0, 0][0])
        self.assertEqual(full[0, 1][0], short[0, 1][0])
        self.assertEqual(short[0, 0][1], 8)
        self.assertEqual(short[0, 1][1], 8)
        self.assertEqual(no_drop["candidates_truncated"], 0)
        self.assertEqual(drop["questions_truncated"], 1)
        self.assertEqual(drop["candidates_truncated"], 2)
        self.assertEqual(drop["total_removed_context_tokens"],
                         sum(length - 8 for targets, length in full.values()))

    def test_answer_overflow_is_rejected_instead_of_losing_tokens(self):
        with self.assertRaisesRegex(ValueError, "causal prediction"):
            analyze.retokenize([{"context": "Q:", "choices": ["answer"]}],
                               CharacterTokenizer(), False, 3)


class StatisticalTests(unittest.TestCase):
    def test_exact_mcnemar_matches_enumerated_conditional_null(self):
        for corrected, regressed in ((0, 0), (4, 0), (3, 1), (2, 2)):
            rows = []
            for kind in ["corrected"] * corrected + ["regressed"] * regressed + ["same"]:
                base, guided = (1, 0) if kind == "corrected" else (0, 1) if kind == "regressed" else (0, 0)
                rows.append({"gold": 0, "predictions": {"acc": {
                    "baseline": base, "loopcd_fixed": guided,
                }}})
            result = analyze.legacy.paired_stats(rows, "loopcd_fixed", "acc")
            discordant = corrected + regressed
            probabilities = [sum(bits) for bits in itertools.product((0, 1), repeat=discordant)]
            tail = sum(count <= min(corrected, regressed) for count in probabilities) / len(probabilities)
            self.assertEqual(result["mcnemar_exact_two_sided_p"], min(1., 2 * tail))
            self.assertEqual(result["net_correct"], corrected - regressed)
            low, high = result["paired_bootstrap_95ci_pp"]
            self.assertLessEqual(low, high)

    def test_holm_uses_all_fourteen_comparisons_separately_per_metric(self):
        pvalues = [0.001, .003, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1, .2, .9]
        models = {}
        for index in range(7):
            models[str(index)] = {"paired": {
                metric: {
                    mode: {"mcnemar_exact_two_sided_p": pvalues[2 * index + offset] if metric == "acc" else 1.}
                    for offset, mode in enumerate(analyze.GUIDED)
                } for metric in analyze.METRICS
            }}
        analyze.legacy.holm_adjust(models)
        adjusted = [models[str(index)]["paired"]["acc"][mode]["mcnemar_holm_adjusted_p"]
                    for index in range(7) for mode in analyze.GUIDED]
        expected = np.minimum(1., np.maximum.accumulate(
            np.array(pvalues) * np.arange(14, 0, -1),
        ))
        np.testing.assert_allclose(adjusted, expected)
        self.assertEqual(models["0"]["paired"]["acc_norm"]["loopcd_fixed"]["mcnemar_holm_adjusted_p"], 1.)


class SnapshotAuditTests(unittest.TestCase):
    def test_required_snapshot_cannot_fall_back_to_current_source(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "inventory"):
                analyze.audit_sources(
                    Path(directory), {"code_hashes": {}}, "tiny", "sciq",
                    {"model": {"family": "ouro"}},
                )

    def test_partial_cli_never_writes_final_artifacts(self):
        with patch.object(analyze, "audit_task", return_value={"complete": False}) as audit:
            with patch.object(Path, "write_text", side_effect=AssertionError("Unexpected write")):
                with patch.object(analyze.sys, "argv", ["analyze.py", "--task", "sciq", "--partial"]):
                    with patch("builtins.print"):
                        analyze.main()
            self.assertEqual(audit.call_args.args[0], "sciq")
            self.assertTrue(audit.call_args.kwargs["partial"])


class RuntimeAndCheckpointAuditTests(unittest.TestCase):
    def runtime(self):
        from tests.test_run_integrity import manifest
        result = manifest()
        result.update({
            "task": "sciq", "task_configuration": {}, "shots": 0, "primary_metric": "acc",
            "protocol_sha256": "a" * 64, "context_policy": "left", "truncation_details": {},
        })
        return result

    def test_runtime_fields_are_required_and_types_are_checked(self):
        valid = self.runtime()
        # Package versions are historical provenance, not required to match the
        # current interpreter's installed packages.
        valid["torch"] = "historical-torch-version"
        analyze.audit_runtime_metadata(valid)
        for field in sorted(analyze.MULTITASK_MANIFEST_KEYS):
            broken = copy.deepcopy(valid)
            del broken[field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, field):
                analyze.audit_runtime_metadata(broken)
        for field, bad in [
            ("batch_size", True), ("batch_size", 0), ("token_budget", -1),
            ("token_budget", 8.), ("torch", None), ("transformers", ""),
            ("adapter_metadata", []), ("max_input_tokens", False),
            ("shots", False), ("truncation", 0), ("native_source_hashes", {}),
        ]:
            broken = copy.deepcopy(valid)
            broken[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaisesRegex(ValueError, field):
                analyze.audit_runtime_metadata(broken)

    def checkpoint(self, directory, change=None):
        root = Path(directory)
        model_dir = root / "models/Tiny"
        model_dir.mkdir(parents=True)
        contents = {
            "config.json": b"{}", "tokenizer.json": b"{}",
            "model-00001-of-00002.safetensors": b"first weights",
            "model-00002-of-00002.safetensors": b"second weights",
            "model.safetensors.index.json": json.dumps({"weight_map": {
                "first": "model-00001-of-00002.safetensors",
                "second": "model-00002-of-00002.safetensors",
            }}).encode(),
        }
        files = []
        for name, content in contents.items():
            (model_dir / name).write_bytes(content)
            files.append({"path": f"models/Tiny/{name}", "bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest()})
        manifest = {"revision": "a" * 40, "complete": True, "files": files}
        if change is not None:
            change(manifest, model_dir)
        (root / "models/manifest.json").write_text(json.dumps(manifest))
        config = {"model_manifest": "models/manifest.json", "model_path": "models/Tiny", "family": "ouro"}
        with patch.object(analyze, "ROOT", root), patch.object(
            analyze, "native_source_hashes", return_value={"native.py": "b" * 64},
        ):
            return analyze.audit_checkpoint(config, {})

    def test_checkpoint_requires_complete_nonempty_weights_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.checkpoint(directory)
            self.assertEqual(result["revision"], "a" * 40)
        changes = {
            "incomplete": lambda m, p: m.__setitem__("complete", False),
            "weights_removed": lambda m, p: m.__setitem__("files", m["files"][:2]),
            "index_removed": lambda m, p: m["files"].pop(),
            "shard_unlisted": lambda m, p: m["files"].pop(2),
            "shard_missing": lambda m, p: (p / "model-00001-of-00002.safetensors").unlink(),
            "wrong_size": lambda m, p: m["files"][0].__setitem__("bytes", 999),
            "wrong_hash": lambda m, p: m["files"][0].__setitem__("sha256", "c" * 64),
            "duplicate_alias": lambda m, p: m["files"].append({
                **m["files"][0], "path": "models/Tiny/../Tiny/config.json",
            }),
        }
        for name, change in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.checkpoint(directory, change)


class FinalArtifactsTests(unittest.TestCase):
    def result(self):
        _, _, _, _, summary = fixture()
        model = {
            "display": "Tiny, Unicode modèle", "shots": 0, "n": 1,
            "accuracy_percent": summary["accuracy_percent"], "correct": summary["correct"],
            "paper_table2": {"baseline": 10., "loopcd_fixed": 20., "loopcd_adaptive": 30.},
            "paired": {
                metric: {
                    mode: {
                        "delta_percentage_points": 0., "mcnemar_holm_adjusted_p": 1.,
                        "mcnemar_exact_two_sided_p": 1., "paired_bootstrap_95ci_pp": [0., 0.],
                        "corrected": 0, "regressed": 0, "net_correct": 0, "changed_answers": 0,
                    } for mode in analyze.GUIDED
                } for metric in analyze.METRICS
            },
            "run_manifest": {"dtype": "float32", "float32_matmul_precision": "highest",
                             "max_input_tokens": 20},
            "audit": {"unique_candidates": 2, "validation": {"batch_single_max_error": 0.}},
            "results_path": "results/multitask/sciq/tiny",
        }
        return {
            "task": "sciq", "display": "SciQ", "complete": True, "primary_metric": "acc",
            "task_configuration": {"split": "test", "count": 1},
            "expected_documents_per_configuration": 1,
            "models": {name: copy.deepcopy(model) for name in sorted(analyze.MODEL_NAMES)},
        }

    def test_csv_has_both_metrics_and_only_primary_has_paper_values(self):
        rows = list(csv.DictReader(io.StringIO(analyze.comparison_csv({"sciq": self.result()}))))
        self.assertEqual(len(rows), 14)
        self.assertEqual({row["model"] for row in rows}, analyze.MODEL_NAMES)
        for row in rows:
            self.assertEqual(row["display"], "Tiny, Unicode modèle")
            if row["metric"] == "acc":
                self.assertEqual(row["is_primary"], "True")
                self.assertEqual(float(row["paper_baseline_percent"]), 10.)
            else:
                self.assertEqual(row["is_primary"], "False")
                self.assertEqual(row["paper_baseline_percent"], "")
            self.assertEqual(float(row["fixed_holm_p"]), 1.)

    def test_sciq_report_keeps_preselected_metric_and_protocol_limit(self):
        text = analyze.report_task(self.result())
        self.assertIn("The primary metric was fixed as `acc`", text)
        self.assertIn("the primary metric has not been reassigned based on agreement with the paper", text)
        self.assertIn("not confirmation of their evaluation protocol", text)
        self.assertIn("unnormalized scores are closer to the paper", text)
        self.assertIn("no cross-task correction", text)

    def test_final_artifacts_reject_incomplete_task_even_when_called_directly(self):
        result = self.result()
        result["complete"] = False
        for fn, value in (
            (analyze.report_task, result),
            (analyze.report_all, {"sciq": result}),
            (analyze.comparison_csv, {"sciq": result}),
        ):
            with self.subTest(function=fn.__name__), self.assertRaises(ValueError):
                fn(value)


if __name__ == "__main__":
    unittest.main()
