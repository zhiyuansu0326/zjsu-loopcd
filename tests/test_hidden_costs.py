"""Hidden guidance arithmetic and standalone benchmark evidence checks."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import hidden_cost_audit as audit
from scripts.benchmark_hidden_huginn import summarize_timing
from src.hidden_costs import POLICIES, comparison, prefill_comparison
from src.huginn_costs import sample_statistics
from src.run_integrity import sha256_file
from tests import test_reduced_cost_audit as reduced_fixture


class HiddenArithmeticTests(unittest.TestCase):
    def test_hidden_has_one_matrix_readout_and_separate_full_sequence_vector_work(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "models/Huginn-0125/config.json").read_text())
        for loops in (16, 32):
            rows = comparison(config, loops, sequence_length=73, batch_size=8, head_positions=39)
            baseline, logits, hidden = (rows[key] for key in POLICIES)
            self.assertEqual(hidden["components_flops"], baseline["components_flops"])
            self.assertEqual(hidden["fraction_of_baseline"], 1)
            self.assertEqual(hidden["guidance_elementwise_operations"], 3 * 8 * 73 * 5280)
            self.assertEqual(logits["guidance_elementwise_operations"], 3 * 39 * 65536)
            self.assertEqual(logits["total_flops"] - hidden["total_flops"],
                             baseline["components_flops"]["final_coda"] + baseline["components_flops"]["final_head"])
            paper = prefill_comparison(config, loops)["matching_paper_table_convention"]
            self.assertEqual(round(paper["baseline"]["total_tflops"], 3), 28.261 if loops == 16 else 54.526)
            self.assertEqual(round(paper["loopcd_logits"]["fraction_of_baseline"], 3), 1.042 if loops == 16 else 1.022)


class HiddenCostAuditTests(unittest.TestCase):
    def setUp(self):
        # Reuse the earlier benchmark's small filesystem fixture, then construct
        # this experiment's distinct policies, sources, and fixed 3x8 workload.
        reduced_fixture.ReducedCostAuditTests.setUp(self)
        for name in ("scripts/benchmark_hidden_huginn.py", "src/hidden_costs.py", "configs/hidden.json"):
            path, snapshot = self.root / name, self.snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture {name}\n")
            snapshot.write_bytes(path.read_bytes())
            self.report["provenance"]["code_hashes"][name] = sha256_file(path)
        config = self.report["checkpoint_config"]
        self.manifest.update(name="huginn_16", guidance_space="hidden", hidden_setting={"method": "LoopCD-Hidden"},
                             documents=16, candidates=32)
        self.manifest["configuration"]["reference_step"] = 7
        self.manifest["adapter_metadata"].update(
            reference_recurrent_update=7, model_revision="revision", noise_seed=1234,
            noise_key="question", noise_generation="shared", noise_coupling="shared",
        )
        self.documents = [{"id": f"q{index}"} for index in range(16)]
        self.encoded = {(index // 2, index % 2): ([index % 16 + 1], 4 + index // 4) for index in range(32)}
        self.report.update(model="huginn_16", loops=16, hidden_reference_step=7, fixed_strength=0.5,
                           hidden_setting=self.manifest["hidden_setting"],
                           flops_512_token_prefill=prefill_comparison(config, 16))
        self.report["provenance"]["hidden_configuration_sha256"] = sha256_file(self.root / "configs/hidden.json")
        self.report["adapter_metadata"] = {
            "hidden": copy.deepcopy(self.manifest["adapter_metadata"]),
            "logits": {**self.manifest["adapter_metadata"], "reference_recurrent_update": 1},
        }
        self.report["methodology"].update(
            warmup_per_policy_per_batch=2, repetitions_per_policy_per_batch=6,
            policies={
                "baseline": {"loops": 16, "reference_step": None, "coda_passes": 1},
                "loopcd_logits": {"loops": 16, "reference_step": 1, "fixed_strength": 0.5, "coda_passes": 2},
                "loopcd_hidden": {"loops": 16, "reference_step": 7, "fixed_strength": 0.5, "coda_passes": 1},
            },
        )
        self.report["validation"] = {
            **{policy: {"standalone_matches_shared_comparison_exactly": True,
                        "max_absolute_token_logprob_difference": 0.0,
                        "normal_initial_state_generation_matches_precomputed": True}
               for policy in ("baseline", "loopcd_hidden")},
            "loopcd_logits": {"matches_shared_evaluator_token_logprobs_exactly": True,
                              "max_absolute_token_logprob_difference": 0.0,
                              "normal_initial_state_generation_matches_precomputed": True},
        }
        ordered = sorted(self.encoded, key=lambda key: (self.encoded[key][1], *key))
        self.report["batches"] = []
        for index in range(3):
            quantile = (index + 1) / 4
            start = int(quantile * 31) - 4
            selected = ordered[start:start + 8]
            length = max(self.encoded[key][1] for key in selected)
            self.report["batches"].append({
                "batch_index": index, "candidate_count": 8, "length_quantile": quantile,
                "start_in_length_sorted_candidates": start, "padded_length": length,
                "answer_prediction_positions": 8,
                "input_tokens_without_padding": sum(self.encoded[key][1] for key in selected),
                "candidates": [{"question_index": question, "question_id": f"q{question}",
                                "choice_index": choice, "input_length": self.encoded[question, choice][1],
                                "target_ids": self.encoded[question, choice][0]} for question, choice in selected],
                "flops": comparison(config, 16, sequence_length=length, batch_size=8, head_positions=8),
                "initial_state_generation_raw_seconds": [0.01] * 6,
                "initial_state_generation_seconds": sample_statistics([0.01] * 6),
            })
        self.report["raw_samples"] = [{
            "repetition": rep, "batch_index": batch, "policy": policy,
            "policy_order_in_batch": (index - rep - batch) % 3,
            "seconds": (index + 1) * 0.1 + rep * 0.01 + batch * 0.01,
            "peak_gpu_allocated_gib": 2.0, "peak_incremental_gpu_allocated_gib": 1.0,
            "peak_gpu_reserved_gib": 3.0,
        } for rep in range(6) for batch in range(3) for index, policy in enumerate(POLICIES)]
        self.report["timing"] = summarize_timing(self.report["raw_samples"], 3, 6)
        root_patch = patch.object(audit, "ROOT", self.root)
        token_patch = patch.object(audit, "retokenized_candidates",
                                  return_value=(self.documents, self.encoded, self.truncation))
        root_patch.start()
        token_patch.start()
        self.addCleanup(root_patch.stop)
        self.addCleanup(token_patch.stop)

    def run_audit(self, report=None):
        self.cost_path.write_text(json.dumps(self.report if report is None else report))
        return audit.audit_costs(self.cost_path, self.manifest)

    def test_complete_evidence_reconstructs_all_policies_and_samples(self):
        evidence = self.run_audit()
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["verified_sampled_candidates"], 24)
        self.assertEqual(evidence["verified_raw_timing_samples"], 54)
        self.assertEqual(evidence["verified_source_files"], 14)

    def test_r32_uses_reference_six_and_reconstructs_its_own_compute(self):
        self.manifest["name"] = "huginn_32"
        self.manifest["configuration"].update(loops=32, reference_step=6)
        self.manifest["adapter_metadata"].update(loops=32, reference_recurrent_update=6)
        self.report.update(model="huginn_32", loops=32, hidden_reference_step=6)
        self.report["adapter_metadata"]["hidden"].update(loops=32, reference_recurrent_update=6)
        self.report["adapter_metadata"]["logits"]["loops"] = 32
        for policy in self.report["methodology"]["policies"].values():
            policy["loops"] = 32
        self.report["methodology"]["policies"]["loopcd_hidden"]["reference_step"] = 6
        config = self.report["checkpoint_config"]
        self.report["flops_512_token_prefill"] = prefill_comparison(config, 32)
        for batch in self.report["batches"]:
            batch["flops"] = comparison(config, 32, sequence_length=batch["padded_length"],
                                        batch_size=8, head_positions=batch["answer_prediction_positions"])
        self.assertTrue(self.run_audit()["passed"])

    def test_wrong_hidden_setting_counts_and_readout_policy_are_rejected(self):
        changes = [
            (("hidden_reference_step",), 1),
            (("methodology", "policies", "loopcd_hidden", "coda_passes"), 2),
            (("methodology", "repetitions_per_policy_per_batch"), 3),
            (("provenance", "slurm", "Account"), "other"),
            (("validation", "loopcd_hidden", "standalone_matches_shared_comparison_exactly"), False),
        ]
        for path, value in changes:
            report = copy.deepcopy(self.report)
            target = report
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run_audit(report)

    def test_timing_candidates_flops_and_sources_are_independently_checked(self):
        report = copy.deepcopy(self.report)
        report["timing"]["loopcd_hidden"]["speedup_vs_logits"] *= 2
        with self.assertRaisesRegex(ValueError, "Reconstructed timing"):
            self.run_audit(report)
        report = copy.deepcopy(self.report)
        report["batches"][0]["candidates"][0]["target_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "Retokenized benchmark"):
            self.run_audit(report)
        report = copy.deepcopy(self.report)
        report["flops_512_token_prefill"]["including_recurrent_injection"]["loopcd_hidden"]["total_flops"] += 100
        with self.assertRaisesRegex(ValueError, "Prefill FLOPs"):
            self.run_audit(report)
        (self.snapshot / "src/hidden_costs.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "snapshot SHA"):
            self.run_audit()

    def test_summarizer_rejects_duplicate_or_missing_measurements(self):
        samples = copy.deepcopy(self.report["raw_samples"])
        samples[-1] = samples[0]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            summarize_timing(samples, 3, 6)


if __name__ == "__main__":
    unittest.main()
