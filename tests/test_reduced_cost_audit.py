"""Cost-report tampering tests with a small filesystem evidence fixture."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import reduced_cost_audit as audit
from src.huginn_costs import comparison, paper_prefill_comparison, sample_statistics
from src.run_integrity import sha256_file


class ReducedCostAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cost_path = self.root / "results/huginn_costs.json"
        self.snapshot = self.root / "results/huginn_costs_source_snapshot"
        config = {
            "n_embd": 8, "intermediate_size": 16, "n_heads": 2,
            "num_key_value_heads": 2, "head_dim": 4, "padded_vocab_size": 32,
            "n_layers_in_prelude": 1, "n_layers_in_recurrent_block": 2,
            "n_layers_in_coda": 1, "block_size": 32,
        }
        paths = [
            "scripts/benchmark_huginn.py", "src/huginn_costs.py", "src/adapter_huginn.py",
            "src/suite_scoring.py", "src/loopcd.py", "src/evaluation.py",
            "src/run_integrity.py", "src/registry.py", "configs/models.json", "configs/tasks.json",
            "models/huginn/config.json",
        ]
        for name in paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(config) if name.endswith("/config.json") else f"fixture {name}\n"
            path.write_text(text)
            destination = self.snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)
        for name in ("data/0shot.jsonl", "data/protocol.json", "models/manifest.json",
                     "models/huginn/native.py", "sources/looped_qwen/loopcd.html"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name + "\n")
        self.documents = [{"id": "q0"}, {"id": "q1"}]
        self.encoded = {(0, 0): ([7], 4), (0, 1): ([8], 5),
                        (1, 0): ([9], 6), (1, 1): ([10], 7)}
        self.truncation = {"records": [], "candidates_truncated": 0}
        hashes = {name: sha256_file(self.root / name) for name in paths}
        self.manifest = {
            "task": "arc_challenge", "shots": 0, "documents": 2, "candidates": 4,
            "configuration": {
                "family": "huginn", "loops": 16, "reference_step": 5, "fixed": 0.5,
                "model_path": "models/huginn", "model_manifest": "models/manifest.json",
            },
            "task_configuration": {
                "data_path": "data/{shots}shot.jsonl", "protocol_path": "data/protocol.json",
            },
            "data_sha256": sha256_file(self.root / "data/0shot.jsonl"),
            "protocol_sha256": sha256_file(self.root / "data/protocol.json"),
            "model_manifest_sha256": sha256_file(self.root / "models/manifest.json"),
            "native_source_hashes": {
                "models/huginn/native.py": sha256_file(self.root / "models/huginn/native.py"),
            },
            "torch": "test", "transformers": "test", "gpu": "NVIDIA H200", "seed": 1234,
            "dtype": "float32", "float32_matmul_precision": "highest",
            "adapter_metadata": {"loops": 16, "reference_recurrent_update": 5},
            "code_hashes": {name: value for name, value in hashes.items() if name.startswith("src/")},
        }
        provenance = {key: self.manifest[key] for key in (
            "data_sha256", "protocol_sha256", "model_manifest_sha256",
            "native_source_hashes", "torch", "transformers", "gpu", "seed",
        )}
        provenance.update(
            slurm={"Account": "scavenger-h200", "Partition": "scavenger-h200", "JobId": "1"},
            code_hashes=hashes,
            paper_source_sha256=sha256_file(self.root / "sources/looped_qwen/loopcd.html"),
        )
        self.report = {
            "task": "arc_challenge", "shots": 0, "state": "complete", "complete": True,
            "provenance": provenance, "checkpoint_config": config,
            "checkpoint_config_sha256": sha256_file(self.root / "models/huginn/config.json"),
            "adapter_metadata": copy.deepcopy(self.manifest["adapter_metadata"]),
            "truncation_details": self.truncation,
            "flops_512_token_prefill": paper_prefill_comparison(config),
            "methodology": {
                "dtype": "float32", "float32_matmul_precision": "highest", "cache": False,
                "warmup_per_policy_per_batch": 1, "repetitions_per_policy_per_batch": 3,
                "policies": {
                    "baseline_r32": {"loops": 32, "reference_step": None, "coda_passes": 1},
                    "baseline_r16": {"loops": 16, "reference_step": None, "coda_passes": 1},
                    "loopcd_r16_ref5": {
                        "loops": 16, "reference_step": 5, "fixed_strength": 0.5, "coda_passes": 2,
                    },
                },
            },
            "validation": {
                **{policy: {
                    "standalone_matches_joint_baseline_exactly": True,
                    "max_absolute_logit_difference": 0.0,
                } for policy in audit.POLICIES[:2]},
                "loopcd_r16_ref5": {
                    "matches_shared_evaluator_token_logprobs_exactly": True,
                    "max_absolute_token_logprob_difference": 0.0,
                    "precomputed_noise_matches_normal_state_generation": True,
                },
            },
            "batches": [{
                "batch_index": 0, "candidate_count": 2, "length_quantile": 0.5,
                "start_in_length_sorted_candidates": 0, "padded_length": 5,
                "answer_prediction_positions": 2, "input_tokens_without_padding": 9,
                "candidates": [{
                    "question_index": 0, "question_id": "q0", "choice_index": choice,
                    "input_length": self.encoded[(0, choice)][1],
                    "target_ids": self.encoded[(0, choice)][0],
                } for choice in (0, 1)],
                "flops": comparison(config, sequence_length=5, batch_size=2, head_positions=2),
                "initial_state_generation_raw_seconds": [0.01, 0.02, 0.03],
                "initial_state_generation_seconds": sample_statistics([0.01, 0.02, 0.03]),
            }],
            "raw_samples": [], "timing": {},
        }
        for rep in range(3):
            for index, policy in enumerate(audit.POLICIES):
                self.report["raw_samples"].append({
                    "repetition": rep, "batch_index": 0, "policy": policy,
                    "policy_order_in_batch": (index - rep) % 3,
                    "seconds": (index + 1) * 0.1 + rep * 0.01,
                    "peak_gpu_allocated_gib": 2.0, "peak_incremental_gpu_allocated_gib": 1.0,
                    "peak_gpu_reserved_gib": 3.0,
                })
        for policy in audit.POLICIES:
            values = [row["seconds"] for row in self.report["raw_samples"] if row["policy"] == policy]
            self.report["timing"][policy] = {
                "aggregate_seconds": sample_statistics(values), "aggregate_raw_seconds": values,
                "peak_gpu_allocated_gib": 2.0, "peak_incremental_gpu_allocated_gib": 1.0,
                "by_batch": [{
                    "batch_index": 0, "seconds": sample_statistics(values),
                    "raw_seconds": values, "peak_gpu_allocated_gib": 2.0,
                }],
            }
        for row in self.report["timing"].values():
            row["speedup_vs_baseline_r32"] = (
                self.report["timing"]["baseline_r32"]["aggregate_seconds"]["median"]
                / row["aggregate_seconds"]["median"]
            )
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

    def test_complete_fixture_reconstructs_every_sample(self):
        evidence = self.run_audit()
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["verified_raw_timing_samples"], 9)
        self.assertEqual(evidence["verified_sampled_candidates"], 2)

    def test_rejects_incomplete_wrong_runtime_and_unauthorized_allocation(self):
        for path, value, message in (
            (("complete",), False, "incomplete"),
            (("provenance", "gpu"), "Another H200", "provenance gpu"),
            (("methodology", "float32_matmul_precision"), "high", "precision"),
            (("provenance", "slurm", "Account"), "other", "scavenger-h200"),
            (("validation", "baseline_r16", "standalone_matches_joint_baseline_exactly"), False,
             "Standalone validation"),
        ):
            report = copy.deepcopy(self.report)
            target = report
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, message):
                self.run_audit(report)

    def test_rejects_timing_summary_and_sample_tampering(self):
        report = copy.deepcopy(self.report)
        report["timing"]["baseline_r16"]["speedup_vs_baseline_r32"] *= 2
        with self.assertRaisesRegex(ValueError, "Reconstructed timing"):
            self.run_audit(report)
        report = copy.deepcopy(self.report)
        report["raw_samples"][-1] = copy.deepcopy(report["raw_samples"][0])
        with self.assertRaisesRegex(ValueError, "duplicate timing"):
            self.run_audit(report)
        report = copy.deepcopy(self.report)
        report["raw_samples"].pop()
        with self.assertRaisesRegex(ValueError, "Missing timing"):
            self.run_audit(report)

    def test_rejects_flops_candidate_and_source_tampering(self):
        report = copy.deepcopy(self.report)
        report["batches"][0]["flops"]["baseline_r32"]["total_flops"] += 100
        with self.assertRaisesRegex(ValueError, "Batch FLOPs"):
            self.run_audit(report)
        report = copy.deepcopy(self.report)
        report["batches"][0]["candidates"][0]["question_id"] = "another-question"
        with self.assertRaisesRegex(ValueError, "Retokenized benchmark candidates"):
            self.run_audit(report)
        (self.snapshot / "src/adapter_huginn.py").write_text("modified\n")
        with self.assertRaisesRegex(ValueError, "snapshot SHA"):
            self.run_audit()

    def test_rejects_snapshot_path_escape(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            audit.inside(self.snapshot, "../outside.py")


if __name__ == "__main__":
    unittest.main()
