"""Frozen Hidden settings, paired families, replay provenance, and report integrity."""

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_hidden as hidden
from scripts import analyze
from src.registry import load_registry


ROOT = Path(__file__).resolve().parents[1]


def plan_fixture():
    paths = ("configs/models.json", "configs/tasks.json", "configs/hidden.json")
    return {
        "task_order": list(hidden.TASKS), "model_order": list(hidden.MODELS),
        "primary_metrics": dict(hidden.PRIMARY_METRICS),
        "fixed_parameters": copy.deepcopy(hidden.FIXED_PARAMETERS),
        "bootstrap": {"samples": 20000, "seed": 1234},
        "permutation": {"samples": 200000, "seed": 1234},
        "multiplicity": {"primary_tasks": 16, "primary_macro": 4},
        "timing": {"batches_per_task": 3, "candidates_per_batch": 8,
                   "warmups_per_policy_per_batch": 2, "repetitions_per_policy_per_batch": 6},
        "configuration_sha256": {path: analyze.digest(ROOT / path) for path in paths},
    }


def complete_fixture():
    _, registry = load_registry()
    reference = json.loads(hidden.PAPER_PATH.read_text())
    results = {}
    for index, task in enumerate(hidden.TASKS):
        n = registry["tasks"][task]["count"]
        results[task] = {}
        for model in hidden.MODELS:
            loops = hidden.FIXED_PARAMETERS["loops_by_model"][model]
            # Whole-task patterns deliberately have an equal-task mean differing
            # from a pooled/question-weighted mean.
            counts = {"baseline": n * (index in (1, 3)),
                      "loopcd_logits": n * (index in (0, 2)),
                      "loopcd_hidden": n * (index in (0, 1, 2))}
            paired = {}
            for name, (left, right, _) in hidden.CONTRASTS.items():
                change = counts[right] - counts[left]
                paired[name] = {
                    "corrected": max(change, 0), "regressed": max(-change, 0),
                    "delta_percentage_points": 100 * change / n,
                    "paired_bootstrap_95ci_pp": [100 * change / n] * 2,
                    # A synthetic audit output used to verify multiplicity wiring.
                    "mcnemar_exact_two_sided_p": 0.01,
                }
            manifest = {
                "configuration": {"family": "huginn", "loops": loops, "fixed": 0.5,
                                  "reference_step": hidden.REFERENCES[model]},
                "adapter_metadata": {"loops": loops, "reference_recurrent_update": hidden.REFERENCES[model],
                                     "model_revision": "revision"},
                "guidance_space": "hidden", "dtype": "float32", "float32_matmul_precision": "highest",
                "seed": 1234, "shots": 0, "model_manifest_sha256": "checkpoint",
                "native_source_hashes": {"native": "digest"},
            }
            timing = {
                policy: {"aggregate_seconds": {"median": seconds}, "latency_ratio_to_baseline": seconds / 2}
                for policy, seconds in (("baseline", 2), ("loopcd_logits", 2.2), ("loopcd_hidden", 2.1))
            }
            flops = {
                "including_recurrent_injection": {
                    policy: {"total_tflops": 1.0 if policy != "loopcd_logits" else 1.04,
                             "guidance_elementwise_operations": 123 if policy == "loopcd_hidden" else 0}
                    for policy in hidden.POLICIES
                }
            }
            results[task][model] = {
                "complete": True, "task": task, "model": model,
                "display": registry["tasks"][task]["display"], "primary_metric": hidden.PRIMARY_METRICS[task],
                "paper_table3": reference["scores_percent"][task][model],
                "audited_runs": {"original": {"audit": {"passed": True}},
                                 "hidden": {"audit": {"passed": True, "unique_candidates": {
                                     "arc_easy": 9501, "arc_challenge": 4687, "sciq": 4000, "piqa": 3676,
                                 }[task]}, "run_manifest": manifest}},
                "policies": {
                    policy: {"n": n, "correct": {metric: count for metric in analyze.METRICS},
                             "accuracy_percent": {metric: 100 * count / n for metric in analyze.METRICS}}
                    for policy, count in counts.items()
                },
                "paired": {metric: copy.deepcopy(paired) for metric in analyze.METRICS},
                "baseline_replay": {"identical_predictions_both_metrics": True,
                                    "maximum_token_logprob_difference": 0, "tolerance": 1e-3},
                "cost_audit": {"passed": True},
                "costs": {"complete": True, "state": "complete", "task": task, "model": model,
                          "loops": loops, "hidden_reference_step": hidden.REFERENCES[model],
                          "fixed_strength": 0.5, "timing": timing, "flops_512_token_prefill": flops,
                          "batches": [{"candidate_count": 8}] * 3,
                          "methodology": {"warmup_per_policy_per_batch": 2, "repetitions_per_policy_per_batch": 6}},
            }
    return results


class HiddenAnalysisTests(unittest.TestCase):
    def test_paper_table3_numbers_and_distinct_full_depth_references(self):
        paper = json.loads(hidden.PAPER_PATH.read_text())
        self.assertEqual(paper["settings"]["huginn_16"]["reference_step"], 7)
        self.assertEqual(paper["settings"]["huginn_32"]["reference_step"], 6)
        self.assertEqual(paper["scores_percent"]["piqa"]["huginn_16"],
                         {"baseline": 75.19, "loopcd_hidden": 75.08})
        self.assertEqual(paper["scores_percent"]["arc_challenge"]["huginn_32"],
                         {"baseline": 37.97, "loopcd_hidden": 38.74})

    def test_plan_rejects_metric_reference_timing_and_hash_changes(self):
        hidden.validate_plan(plan_fixture())
        for key in ("metric", "reference", "timing", "hash"):
            plan = plan_fixture()
            if key == "metric":
                plan["primary_metrics"]["sciq"] = "acc_norm"
            elif key == "reference":
                plan["fixed_parameters"]["reference_steps"]["huginn_16"] = 6
            elif key == "timing":
                plan["timing"]["repetitions_per_policy_per_batch"] = 5
            else:
                plan["configuration_sha256"]["configs/hidden.json"] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                hidden.validate_plan(plan)

    def test_complete_suite_means_and_multiplicity_use_all_eight_settings(self):
        result = hidden.build_suite(complete_fixture(), plan_fixture())
        for model in hidden.MODELS:
            means = result["means"][model]
            self.assertEqual(set(means), {"frozen_primary_metrics", "all_acc", "all_acc_norm"})
            primary = means["frozen_primary_metrics"]
            self.assertEqual(primary["accuracy_percent"],
                             {"baseline": 50, "loopcd_logits": 50, "loopcd_hidden": 75})
            self.assertEqual(primary["contrasts"]["hidden_vs_baseline"]["delta_percentage_points"], 25)
            self.assertEqual(primary["contrasts"]["hidden_vs_baseline"]["wins"], 2)
            self.assertEqual(primary["contrasts"]["hidden_vs_baseline"]["losses"], 1)
            self.assertEqual(len(primary["contrasts"]["hidden_vs_logits"]["task_discordances"]), 4)
            for task in hidden.TASKS:
                row = result["tasks"][task][model]
                for contrast in hidden.CONTRASTS:
                    self.assertAlmostEqual(row["paired"][hidden.PRIMARY_METRICS[task]][contrast]
                                           ["mcnemar_holm_across_sixteen_p"], 0.16)
            self.assertAlmostEqual(result["sampled_latency_ratio_geometric_mean"][model]
                                   ["baseline"]["loopcd_hidden"], 1.05)
            self.assertAlmostEqual(result["sampled_latency_ratio_geometric_mean"][model]
                                   ["loopcd_logits"]["loopcd_hidden"], 2.1 / 2.2)
        macro = [result["means"][model]["frozen_primary_metrics"]["contrasts"][name]
                 for model in hidden.MODELS for name in hidden.CONTRASTS]
        ordered = sorted(macro, key=lambda row: row["paired_signflip_two_sided_p"])
        self.assertAlmostEqual(ordered[0]["paired_signflip_holm_across_four_p"],
                               min(1, 4 * ordered[0]["paired_signflip_two_sided_p"]))

    def test_missing_task_wrong_depth_reference_replay_and_costs_are_rejected(self):
        for damage in ("missing", "reference", "depth", "metric", "replay", "cost", "count", "pair"):
            result = complete_fixture()
            row = result["sciq"]["huginn_16"]
            if damage == "missing":
                del result["piqa"]["huginn_32"]
            elif damage == "reference":
                row["audited_runs"]["hidden"]["run_manifest"]["configuration"]["reference_step"] = 6
            elif damage == "depth":
                row["audited_runs"]["hidden"]["run_manifest"]["adapter_metadata"]["loops"] = 32
            elif damage == "metric":
                row["primary_metric"] = "acc_norm"
            elif damage == "replay":
                row["baseline_replay"]["maximum_token_logprob_difference"] = .002
            elif damage == "cost":
                row["costs"]["complete"] = False
            elif damage == "count":
                row["policies"]["baseline"]["n"] -= 1
            else:
                row["paired"]["acc"]["hidden_vs_baseline"]["regressed"] = 1
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                hidden.validate_results(result)

    def test_shared_provenance_uses_snapshot_alignment_and_rejects_changes(self):
        old = {field: f"{field}_value" for field in hidden.SHARED_MANIFEST_FIELDS}
        old["configuration"] = {"family": "huginn", "loops": 16, "fixed": .5}
        old["adapter_metadata"] = {"reference_recurrent_update": 1, "noise_seed": 1234,
                                   "noise_key": "question", "noise_coupling": "shared"}
        new = copy.deepcopy(old)
        new["configuration"].update(reference_step=7, matmul_precision="highest")
        new["adapter_metadata"]["reference_recurrent_update"] = 7
        original = {"run_manifest": {"kept": "unchanged"}, "comparable_manifest": old}
        result = hidden.validate_shared_provenance(original, {"run_manifest": new})
        self.assertTrue(result["original_source_snapshots_preserved"])
        self.assertEqual(original["run_manifest"], {"kept": "unchanged"})
        for field in ("data_sha256", "native_source_hashes", "noise_seed"):
            changed = copy.deepcopy(new)
            if field == "noise_seed":
                changed["adapter_metadata"][field] = 1235
            else:
                changed[field] = "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                hidden.validate_shared_provenance(original, {"run_manifest": changed})

    def test_report_and_csv_keep_declines_secondary_metrics_and_runtime_limits(self):
        result = hidden.build_suite(complete_fixture(), plan_fixture())
        report = hidden.render_report(result)
        self.assertIn("75.19→75.08", report)
        self.assertIn("Table A8", report)
        self.assertIn("h7", report)
        self.assertIn("not full-test end-to-end", report)
        self.assertIn("Holm p (16 tests)", report)
        self.assertIn("All acc_norm (descriptive)", report)
        rows = list(csv.DictReader(io.StringIO(hidden.render_csv(result))))
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(row["primary_metric"], hidden.PRIMARY_METRICS[row["task"]])
            if row["metric"] == row["primary_metric"]:
                self.assertEqual(float(row["primary_hidden_vs_baseline_holm_p"]), .16)
            else:
                self.assertEqual(row["primary_hidden_vs_baseline_holm_p"], "")
                self.assertEqual(row["paper_baseline_percent"], "")

    def test_historical_arc_configuration_aligns_only_verified_shots_and_paper(self):
        old = {field: f"{field}_value" for field in hidden.SHARED_MANIFEST_FIELDS}
        old["shots"] = 0
        old["configuration"] = {
            "family": "huginn", "display": "Huginn-0125 R=32",
            "model_path": "models/Huginn-0125", "model_manifest": "models/huginn_manifest.json",
            "loops": 32, "shots": 0, "fixed": .5, "cap": .5, "batch_size": 8,
            "paper": [64.31, 64.10, 64.69],
        }
        old["adapter_metadata"] = {"reference_recurrent_update": 1, "noise_seed": 1234,
                                   "noise_key": "question", "noise_coupling": "shared"}
        new = copy.deepcopy(old)
        del new["configuration"]["shots"]
        del new["configuration"]["paper"]
        new["configuration"].update(reference_step=6, matmul_precision="highest")
        new["adapter_metadata"]["reference_recurrent_update"] = 6
        original = {"run_manifest": copy.deepcopy(old), "comparable_manifest": old,
                    "historical_schema_alignment": {"original_manifest_preserved": True}}
        before = copy.deepcopy(original)
        result = hidden.validate_shared_provenance(original, {"run_manifest": new})
        self.assertEqual(result["historical_configuration_alignment"],
                         {"verified_shots": 0, "ignored_informational_fields": ["paper"]})
        self.assertEqual(original, before)
        for damage in ("shots", "loops", "model_path", "cap", "unknown", "new_paper", "no_alignment", "unverified"):
            previous, current = copy.deepcopy(original), copy.deepcopy(new)
            if damage == "shots":
                previous["comparable_manifest"]["configuration"]["shots"] = 8
            elif damage == "loops":
                previous["comparable_manifest"]["configuration"]["loops"] = 16
            elif damage == "model_path":
                previous["comparable_manifest"]["configuration"]["model_path"] = "models/different"
            elif damage == "cap":
                previous["comparable_manifest"]["configuration"]["cap"] = 1.0
            elif damage == "unknown":
                previous["comparable_manifest"]["configuration"]["extra_parameter"] = True
            elif damage == "new_paper":
                current["configuration"]["paper"] = [64.31, 64.10, 64.69]
            elif damage == "no_alignment":
                del previous["historical_schema_alignment"]
            else:
                previous["historical_schema_alignment"]["original_manifest_preserved"] = False
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                hidden.validate_shared_provenance(previous, {"run_manifest": current})

    def test_partial_mode_is_read_only_and_final_refuses_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plan = output / "analysis_plan.json"
            plan.write_text(json.dumps(plan_fixture()))
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with patch("sys.stdout", io.StringIO()):
                status = hidden.run(output, partial=True)
            self.assertFalse(status["complete"])
            self.assertTrue(status["progress_only"])
            self.assertEqual(len(status["settings"]), 8)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)
            with self.assertRaisesRegex(ValueError, "Eight complete"):
                hidden.run(output)
            self.assertFalse((output / "completion.json").exists())

    def test_repeat_without_local_plan_uses_frozen_root_plan_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = plan_fixture()
            for relative in plan["configuration_sha256"]:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / relative).read_bytes())
            (root / "configs/paper_mc_reference.json").write_text("{}")
            frozen = root / "results/hidden/analysis_plan.json"
            frozen.parent.mkdir(parents=True)
            frozen.write_text(json.dumps(plan))
            output = root / "results/repeats/hidden"
            before = {str(path.relative_to(root)): path.read_bytes()
                      for path in root.rglob("*") if path.is_file()}
            with patch.object(hidden, "ROOT", root), patch("sys.stdout", io.StringIO()):
                status = hidden.run(output, partial=True)
                self.assertFalse(status["complete"])
                self.assertEqual(len(status["settings"]), 8)
                with self.assertRaisesRegex(ValueError, "Eight complete"):
                    hidden.run(output)
                explicit = root / "explicit.json"
                changed = copy.deepcopy(plan)
                changed["fixed_parameters"]["reference_steps"]["huginn_16"] = 6
                explicit.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "Frozen Hidden settings"):
                    hidden.run(output, explicit, partial=True)
                explicit.unlink()
            self.assertFalse(output.exists())
            self.assertEqual({str(path.relative_to(root)): path.read_bytes()
                              for path in root.rglob("*") if path.is_file()}, before)

    def test_incomplete_report_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                hidden.write_artifacts({"complete": False}, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_completion_counts_new_evaluations_and_hashes_written_report(self):
        result = hidden.build_suite(complete_fixture(), plan_fixture())
        with tempfile.TemporaryDirectory() as directory, patch.object(analyze, "ROOT", Path(directory)):
            output = Path(directory) / "hidden"
            hidden.write_artifacts(result, output)
            completion = json.loads((output / "completion.json").read_text())
            self.assertEqual(completion["new_evaluation_runs"], 8)
            self.assertEqual(completion["new_question_evaluations"], 12772)
            self.assertEqual(completion["new_scored_candidates"], 43728)
            self.assertEqual(completion["questions_per_depth"], 6386)
            inventory = json.loads((output / "artifact_manifest.json").read_text())["files"]
            self.assertEqual(inventory["hidden/REPORT.md"]["sha256"], analyze.digest(output / "REPORT.md"))
            self.assertEqual(completion["artifact_manifest_sha256"], analyze.digest(output / "artifact_manifest.json"))


if __name__ == "__main__":
    unittest.main()
