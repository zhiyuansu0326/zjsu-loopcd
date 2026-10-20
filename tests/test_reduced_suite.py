"""Equal-task inference and frozen-protocol checks for the reduced-depth suite."""

import copy
import json
from pathlib import Path
import unittest

from scripts.reduced_suite import (
    CONTRASTS, PRIMARY_METRICS, TASKS, build_suite, holm_adjust,
    macro_paired_stats, validate_task_results,
)
from src.huginn_costs import paper_prefill_comparison
from src.registry import load_registry


ROOT = Path(__file__).resolve().parents[1]


def complete_fixture():
    """Use whole-task correct/incorrect patterns with exact known mean changes."""
    _, registry = load_registry()
    config = json.loads((ROOT / "models/Huginn-0125/config.json").read_text())
    flops = paper_prefill_comparison(config)
    full = (False, True, False, True)
    reduced = (True, False, True, False)
    guided = (True, True, True, False)
    results = {}
    for index, task in enumerate(TASKS):
        n = registry["tasks"][task]["count"]
        counts = {"baseline_r32": n * full[index], "baseline_r16": n * reduced[index],
                  "loopcd_r16_ref5": n * guided[index]}
        paired = {}
        for name, (left, right, _) in CONTRASTS.items():
            delta = counts[right] - counts[left]
            paired[name] = {
                "corrected": max(delta, 0), "regressed": max(-delta, 0),
                "delta_percentage_points": 100 * delta / n,
                "mcnemar_exact_two_sided_p": min(1.0, 2.0**(1 - abs(delta))),
            }
        timing = {
            policy: {"aggregate_seconds": {"median": seconds}, "speedup_vs_baseline_r32": 2 / seconds}
            for policy, seconds in (("baseline_r32", 2), ("baseline_r16", 1), ("loopcd_r16_ref5", 1.1))
        }
        results[task] = {
            "complete": True, "task": task, "display": task,
            "primary_metric": PRIMARY_METRICS[task],
            "experiment": {
                "name": "huginn_half_depth", "model": "huginn_16", "reported_policy": "loopcd_fixed",
                "full_depth_model": "huginn_32", "model_overrides": {"reference_step": 5, "fixed": 0.5},
            },
            "audited_runs": {"full": {"audit": {"passed": True}},
                             "original_reduced": {"audit": {"passed": True}},
                             "reduced": {"audit": {"passed": True}, "run_manifest": {
                "configuration": {"family": "huginn", "loops": 16, "reference_step": 5, "fixed": 0.5},
                "adapter_metadata": {
                    "loops": 16, "reference_recurrent_update": 5, "model_revision": "revision",
                    "noise_seed": 1234, "noise_key": "question_id", "noise_generation": "shared",
                    "noise_coupling": "shared",
                },
                "dtype": "float32", "float32_matmul_precision": "highest", "seed": 1234, "shots": 0,
                "model_manifest_sha256": "checkpoint", "native_source_hashes": {"source": "digest"},
            }}},
            "policies": {
                policy: {"n": n, "correct": {"acc": correct, "acc_norm": correct},
                         "accuracy_percent": {"acc": 100 * correct / n, "acc_norm": 100 * correct / n}}
                for policy, correct in counts.items()
            },
            "paired": {metric: copy.deepcopy(paired) for metric in ("acc", "acc_norm")},
            "costs": {"complete": True, "state": "complete", "task": task,
                      "batches": [{"candidate_count": 8}] * 3,
                      "methodology": {"warmup_per_policy_per_batch": 2, "repetitions_per_policy_per_batch": 6},
                      "flops_512_token_prefill": flops, "timing": timing},
            "cost_audit": {"passed": True},
            "costs_file": f"{task}/huginn_costs.json", "costs_sha256": "cost_digest",
            "baseline_replay": {"identical_predictions_both_metrics": True},
        }
    return results


class ReducedSuiteTests(unittest.TestCase):
    def test_bootstrap_is_paired_stratified_equal_weight_and_deterministic(self):
        strata = [(10, 5, 0), (1000, 0, 100)]
        first = macro_paired_stats(strata)
        second = macro_paired_stats(strata)
        self.assertEqual(first, second)
        self.assertEqual(first["delta_percentage_points"], 20.0)
        self.assertNotAlmostEqual(first["delta_percentage_points"], 100 * (5 - 100) / 1010)
        low, high = first["stratified_paired_bootstrap_95ci_pp"]
        self.assertLess(low, 20)
        self.assertGreater(high, 20)

    def test_permutation_is_two_sided_with_plus_one_and_correct_null(self):
        positive = macro_paired_stats([(10, 10, 0)])
        negative = macro_paired_stats([(10, 0, 10)])
        self.assertEqual(positive["paired_signflip_two_sided_p"], negative["paired_signflip_two_sided_p"])
        self.assertAlmostEqual(positive["paired_signflip_two_sided_p"], 2 / 2**10, delta=0.0005)
        self.assertEqual(positive["paired_signflip_two_sided_p"],
                         (positive["permutation_extreme_samples"] + 1) / 200001)
        rare = macro_paired_stats([(100, 100, 0)], bootstrap_samples=10, permutation_samples=1000)
        self.assertEqual(rare["paired_signflip_two_sided_p"], 1 / 1001)
        identical = macro_paired_stats([(10, 0, 0)], bootstrap_samples=10, permutation_samples=10)
        self.assertEqual(identical["paired_signflip_two_sided_p"], 1)
        self.assertEqual(identical["stratified_paired_bootstrap_95ci_pp"], [0, 0])

    def test_holm_uses_the_whole_explicit_family_and_preserves_ties(self):
        self.assertEqual(holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}),
                         {"a": 0.03, "b": 0.06, "c": 0.06})
        family = {str(index): 0.01 for index in range(12)}
        self.assertTrue(all(value == 0.12 for value in holm_adjust(family).values()))

    def test_suite_means_are_not_sample_weighted_and_keep_both_secondary_views(self):
        results = complete_fixture()
        plan = json.loads((ROOT / "results/reduced_depth/analysis_plan.json").read_text())
        suite = build_suite(results, plan)
        means = suite["means"]["frozen_primary_metrics"]
        self.assertEqual(means["accuracy_percent"], {
            "baseline_r32": 50, "baseline_r16": 50, "loopcd_r16_ref5": 75,
        })
        self.assertEqual(means["contrasts"]["guided_reduced_vs_full"]["delta_percentage_points"], 25)
        self.assertEqual(means["contrasts"]["guided_reduced_vs_full"]["wins"], 2)
        self.assertEqual(means["contrasts"]["guided_reduced_vs_full"]["ties"], 1)
        self.assertEqual(means["contrasts"]["guided_reduced_vs_full"]["losses"], 1)
        self.assertEqual(set(suite["means"]), {"frozen_primary_metrics", "all_acc", "all_acc_norm"})
        self.assertTrue(all("mcnemar_holm_across_twelve_p" in row
                            for task in suite["tasks"].values() for row in task["primary_paired"].values()))
        self.assertAlmostEqual(suite["sampled_speedup_geometric_mean"]["loopcd_r16_ref5"], 2 / 1.1)

    def test_missing_task_or_changed_executed_configuration_is_rejected(self):
        missing = complete_fixture()
        del missing["sciq"]
        with self.assertRaisesRegex(ValueError, "four fixed tasks"):
            validate_task_results(missing)
        changed = complete_fixture()
        changed["sciq"]["audited_runs"]["reduced"]["run_manifest"]["configuration"]["fixed"] = 0.75
        with self.assertRaisesRegex(ValueError, "Executed half-depth configuration"):
            validate_task_results(changed)
        changed = complete_fixture()
        changed["sciq"]["primary_metric"] = "acc_norm"
        with self.assertRaisesRegex(ValueError, "Frozen primary metric"):
            validate_task_results(changed)
        changed = complete_fixture()
        changed["piqa"]["costs"]["complete"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete cost"):
            validate_task_results(changed)


if __name__ == "__main__":
    unittest.main()
