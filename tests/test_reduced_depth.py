"""Cross-depth identity and paired-analysis invariants."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze
from scripts.analyze_reduced_depth import paired_comparison, verify_reused_baseline
from scripts.analyze_reduced_depth import audit_historical_arc, write_report
from src.registry import resolve_run


class ReducedDepthTests(unittest.TestCase):
    def test_arc_easy_data_audit_accepts_original_zero_shot_schema(self):
        resolved = resolve_run("arc_easy", "huginn_16", "huginn_half_depth")
        documents, evidence = analyze.audit_data("arc_easy", resolved, {})
        self.assertEqual(len(documents), 2376)
        self.assertTrue(evidence["canonical_raw_and_complete_prompts_verified"])
        self.assertEqual(evidence["validation"]["protocol_schema"], "historical_arc_easy_0shot")
        self.assertIn("data/arc_easy_8shot.jsonl", evidence["raw_and_harness_source_sha256"])
        original_read = analyze.read_json

        def damaged_protocol(path):
            value = original_read(path)
            if Path(path) == resolved["protocol_path"]:
                value["output_sha256"] = "0" * 64
            return value

        with patch.object(analyze, "read_json", side_effect=damaged_protocol):
            with self.assertRaisesRegex(ValueError, "zero-shot output"):
                analyze.audit_data("arc_easy", resolved, {})

    def test_historical_arc_alignment_preserves_original_manifest(self):
        resolved = resolve_run("arc_easy", "huginn_16")
        path = analyze.ROOT / "results/arc_all_models/huginn_16/run_manifest.json"
        original = analyze.read_json(path)
        frozen_digest = analyze.digest(path)
        result = audit_historical_arc("huginn_16", resolved, {})
        self.assertEqual(result["run_manifest"], original)
        self.assertEqual(analyze.digest(path), frozen_digest)
        self.assertNotIn("native_source_hashes", original)
        self.assertEqual(result["comparable_manifest"]["native_source_hashes"],
                         result["checkpoint"]["native_source_hashes"])
        self.assertEqual(result["comparable_manifest"]["shots"], 0)
        self.assertTrue(result["historical_schema_alignment"]["original_manifest_preserved"])
        self.assertEqual(resolved["output"], path.parent)

    def test_report_uses_task_split_count_and_frozen_primary_metric(self):
        contrasts = ("reduced_baseline_vs_full", "guided_reduced_vs_full", "guided_vs_reduced_baseline")
        paired = {key: {
            "delta_percentage_points": 1.0, "paired_bootstrap_95ci_pp": [-1.0, 2.0],
            "corrected": 3, "regressed": 2, "mcnemar_exact_two_sided_p": 1.0,
            "mcnemar_holm_adjusted_p": 1.0,
        } for key in contrasts}
        for task in ("sciq", "piqa"):
            resolved = resolve_run(task, "huginn_16", "huginn_half_depth")
            task_config = resolved["task"]
            n, metric = task_config["count"], task_config["primary_metric"]
            policies = {key: {
                "display": key, "n": n, "correct": {"acc": 930, "acc_norm": 800},
                "accuracy_percent": {"acc": 93.0, "acc_norm": 80.0},
            } for key in ("baseline_r32", "baseline_r16", "loopcd_r16_ref5")}
            flops = {key: {
                "total_tflops": 1.0, "fraction_of_baseline_r32": 0.55, "removed_percent": 45.0,
            } for key in policies}
            result = {
                "task": task, "task_configuration": task_config, "primary_metric": metric,
                "policies": policies, "paired": {"acc": paired, "acc_norm": paired},
                "audited_runs": {"reduced": {
                    "audit": {"unique_candidates": n * 4}, "run_manifest": {"shots": 0},
                }},
                "baseline_replay": {"maximum_token_logprob_difference": 0},
                "costs": {
                    "flops_512_token_prefill": {
                        "including_recurrent_injection": flops, "matching_paper_table_convention": flops,
                    },
                    "timing": {key: {
                        "aggregate_seconds": {"median": 1.0, "iqr": 0.1},
                        "speedup_vs_baseline_r32": 1.8,
                    } for key in policies},
                },
            }
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                write_report(result, output)
                report = (output / "REPORT.md").read_text()
                self.assertIn(f"Huginn × {task_config['display']}", report)
                self.assertIn(f"Complete {task_config['split']} split: {n:,} questions", report)
                self.assertIn(f"Correct / {n:,} ({metric})", report)
                self.assertIn(f"Comparison ({metric})", report)
                self.assertIn(f"TASK={task}", report)
                self.assertNotIn("ARC-Challenge", report)
                with (output / "comparison.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["primary_metric"], metric)
                self.assertEqual(int(rows[0]["correct_primary"]), policies["baseline_r32"]["correct"][metric])

    def test_cross_run_pairing_rejects_order_and_label_mismatch(self):
        first = [{"id": str(i), "gold": 0,
                  "predictions": {"acc_norm": {"baseline": 1}}} for i in range(4)]
        second = [{"id": str(i), "gold": 0,
                   "predictions": {"acc_norm": {"loopcd_fixed": 0}}} for i in range(4)]
        result = paired_comparison(first, "baseline", second, "loopcd_fixed", "acc_norm")
        self.assertEqual(result["corrected"], 4)
        self.assertEqual(result["regressed"], 0)
        self.assertEqual(result["delta_percentage_points"], 100)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 0.125)
        with self.assertRaisesRegex(ValueError, "document order"):
            paired_comparison(first, "baseline", list(reversed(second)), "loopcd_fixed", "acc_norm")
        second[0]["gold"] = 1
        with self.assertRaisesRegex(ValueError, "labels"):
            paired_comparison(first, "baseline", second, "loopcd_fixed", "acc_norm")

    def test_baseline_replay_checks_tokens_and_predictions(self):
        record = {
            "question_index": 0, "choice_index": 0, "target_ids": [7],
            "input_length": 4, "token_logprobs": {"baseline": [-2.0]},
        }
        prediction = {
            "id": "q", "gold": 0,
            "predictions": {"acc": {"baseline": 0}, "acc_norm": {"baseline": 0}},
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(analyze, "ROOT", Path(directory)):
            old, new = Path(directory) / "old", Path(directory) / "new"
            old.mkdir()
            new.mkdir()
            for path in (old, new):
                (path / "candidate_scores.jsonl").write_text(json.dumps(record) + "\n")
                (path / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
            self.assertEqual(verify_reused_baseline(old, new)["maximum_token_logprob_difference"], 0)
            damaged = copy.deepcopy(record)
            damaged["token_logprobs"]["baseline"][0] -= 0.01
            (new / "candidate_scores.jsonl").write_text(json.dumps(damaged) + "\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_reused_baseline(old, new)
            (new / "candidate_scores.jsonl").write_text(json.dumps(record) + "\n")
            prediction["predictions"]["acc_norm"]["baseline"] = 1
            (new / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
            with self.assertRaisesRegex(ValueError, "prediction changed"):
                verify_reused_baseline(old, new)


if __name__ == "__main__":
    unittest.main()
