"""Reconstruct the evidence behind the standalone Huginn cost comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.huginn_costs import comparison, paper_prefill_comparison, sample_statistics
from src.run_integrity import sha256_file

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("baseline_r32", "baseline_r16", "loopcd_r16_ref5")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match(actual, expected, label):
    """Compare rebuilt JSON, rejecting nonfinite or silently missing fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{label}: fields differ")
        for key, value in expected.items():
            match(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label}: lengths differ")
        for index, value in enumerate(expected):
            match(actual[index], value, f"{label}[{index}]")
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"{label}: number differs")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{label}: value differs")


def inside(directory, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Evidence path must be relative")
    path = (directory / name).resolve()
    require(path.is_relative_to(directory.resolve()), "Evidence path escapes its directory")
    return path


def retokenized_candidates(data_path, model_path, max_length):
    from tokenizers import Tokenizer
    from scripts.analyze import retokenize

    documents = [json.loads(line) for line in data_path.read_text().splitlines()]
    tokens, truncation = retokenize(
        documents, Tokenizer.from_file(str(model_path / "tokenizer.json")), True, max_length,
    )
    return documents, tokens, truncation


def audit_costs(costs_path, run_manifest):
    """Verify a completed benchmark against its evaluated run and raw samples."""
    costs_path = Path(costs_path)
    costs = json.loads(costs_path.read_text())
    require(costs.get("complete") is True and costs.get("state") == "complete",
            "Standalone benchmark is incomplete")
    require(costs["task"] == run_manifest["task"] and costs["shots"] == run_manifest["shots"],
            "Benchmark task or shots mismatch")
    provenance, methodology = costs["provenance"], costs["methodology"]
    for key in ("data_sha256", "protocol_sha256", "model_manifest_sha256",
                "native_source_hashes", "torch", "transformers", "gpu", "seed"):
        match(provenance[key], run_manifest[key], f"Benchmark provenance {key}")
    for key in ("dtype", "float32_matmul_precision"):
        match(methodology[key], run_manifest[key], f"Benchmark precision {key}")
    require(methodology["dtype"] == "float32"
            and methodology["float32_matmul_precision"] == "highest"
            and methodology["cache"] is False, "Wrong benchmark numeric protocol")
    require("H200" in provenance["gpu"], "Benchmark GPU is not H200")
    allocation = provenance["slurm"]
    require(allocation["Account"] == allocation["Partition"] == "scavenger-h200",
            "Benchmark allocation must use scavenger-h200")
    require(bool(allocation["JobId"]), "Benchmark lacks an allocated job ID")
    match(costs["adapter_metadata"], run_manifest["adapter_metadata"], "Benchmark adapter metadata")
    match(methodology["policies"], {
        "baseline_r32": {"loops": 32, "reference_step": None, "coda_passes": 1},
        "baseline_r16": {"loops": 16, "reference_step": None, "coda_passes": 1},
        "loopcd_r16_ref5": {
            "loops": 16, "reference_step": 5, "fixed_strength": 0.5, "coda_passes": 2,
        },
    }, "Benchmark policies")
    require(run_manifest["configuration"]["family"] == "huginn"
            and run_manifest["configuration"]["loops"] == 16
            and run_manifest["configuration"]["reference_step"] == 5
            and run_manifest["configuration"]["fixed"] == 0.5,
            "Cost evidence requires the R16 reference5 fixed0.5 run")

    model_path = inside(ROOT, run_manifest["configuration"]["model_path"])
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    require(sha256_file(config_path) == costs["checkpoint_config_sha256"],
            "Checkpoint configuration SHA mismatch")
    match(costs["checkpoint_config"], config, "Checkpoint configuration")
    model_manifest_path = inside(ROOT, run_manifest["configuration"]["model_manifest"])
    require(sha256_file(model_manifest_path) == provenance["model_manifest_sha256"],
            "Checkpoint manifest SHA mismatch")
    task = run_manifest["task_configuration"]
    data_path = inside(ROOT, task["data_path"].format(shots=run_manifest["shots"]))
    protocol_path = inside(ROOT, task["protocol_path"].format(shots=run_manifest["shots"]))
    require(sha256_file(data_path) == provenance["data_sha256"], "Benchmark data SHA mismatch")
    require(sha256_file(protocol_path) == provenance["protocol_sha256"], "Benchmark protocol SHA mismatch")
    require(sha256_file(ROOT / "sources/looped_qwen/loopcd.html") == provenance["paper_source_sha256"],
            "Paper source SHA mismatch")
    for name, digest in provenance["native_source_hashes"].items():
        require(sha256_file(inside(ROOT, name)) == digest, f"Native source SHA mismatch: {name}")
    required_sources = {
        "scripts/benchmark_huginn.py", "src/huginn_costs.py", "src/adapter_huginn.py",
        "src/suite_scoring.py", "src/loopcd.py", "src/evaluation.py",
        "src/run_integrity.py", "src/registry.py", "configs/models.json", "configs/tasks.json",
        str(config_path.relative_to(ROOT)),
    }
    hashes = provenance["code_hashes"]
    require(set(hashes) == required_sources, "Benchmark source inventory differs")
    snapshot = costs_path.parent / (costs_path.stem + "_source_snapshot")
    for name, digest in hashes.items():
        require(sha256_file(inside(snapshot, name)) == digest, f"Benchmark snapshot SHA mismatch: {name}")
        if name in run_manifest["code_hashes"]:
            require(digest == run_manifest["code_hashes"][name],
                    f"Benchmark/evaluation source differs: {name}")
    require(hashes[str(config_path.relative_to(ROOT))] == costs["checkpoint_config_sha256"],
            "Snapshot checkpoint configuration differs")

    validation = costs["validation"]
    for policy in POLICIES[:2]:
        match(validation[policy], {
            "standalone_matches_joint_baseline_exactly": True, "max_absolute_logit_difference": 0.0,
        }, f"Standalone validation {policy}")
    match(validation["loopcd_r16_ref5"], {
        "matches_shared_evaluator_token_logprobs_exactly": True,
        "max_absolute_token_logprob_difference": 0.0,
        "precomputed_noise_matches_normal_state_generation": True,
    }, "Guided validation")
    match(costs["flops_512_token_prefill"], paper_prefill_comparison(config), "Prefill FLOPs")

    documents, encoded, truncation = retokenized_candidates(data_path, model_path, config["block_size"])
    require(len(documents) == run_manifest["documents"]
            and len(encoded) == run_manifest["candidates"], "Benchmark candidate universe changed")
    match(costs["truncation_details"], truncation, "Benchmark truncation")
    ordered = sorted(encoded, key=lambda key: (encoded[key][1], *key))
    batches = costs["batches"]
    require(isinstance(batches, list) and bool(batches), "Benchmark has no batches")
    batch_size = batches[0]["candidate_count"]
    require(type(batch_size) is int and batch_size > 0
            and len(encoded) >= batch_size * len(batches), "Invalid benchmark batch size")
    sampled_keys = set()
    for index, batch in enumerate(batches):
        require(batch["batch_index"] == index and batch["candidate_count"] == batch_size,
                "Benchmark batch indices or sizes differ")
        quantile = (index + 1) / (len(batches) + 1)
        center = int(quantile * (len(ordered) - 1))
        start = min(max(0, center - batch_size // 2), len(ordered) - batch_size)
        selected = ordered[start:start + batch_size]
        match(batch["length_quantile"], quantile, "Batch quantile")
        require(batch["start_in_length_sorted_candidates"] == start, "Batch selection start changed")
        expected_candidates = [{
            "question_index": question, "question_id": documents[question]["id"],
            "choice_index": choice, "input_length": encoded[(question, choice)][1],
            "target_ids": encoded[(question, choice)][0],
        } for question, choice in selected]
        match(batch["candidates"], expected_candidates, "Retokenized benchmark candidates")
        require(not sampled_keys.intersection(selected), "Benchmark batches overlap")
        sampled_keys.update(selected)
        padded = max(encoded[key][1] for key in selected)
        positions = sum(len(encoded[key][0]) for key in selected)
        require(batch["padded_length"] == padded and batch["answer_prediction_positions"] == positions
                and batch["input_tokens_without_padding"] == sum(encoded[key][1] for key in selected),
                "Benchmark batch tensor shape mismatch")
        match(batch["flops"], comparison(
            config, sequence_length=padded, batch_size=batch_size, head_positions=positions,
        ), "Batch FLOPs")
        match(batch["initial_state_generation_seconds"],
              sample_statistics(batch["initial_state_generation_raw_seconds"]), "Initial-state timing")

    repetitions = methodology["repetitions_per_policy_per_batch"]
    warmup = methodology["warmup_per_policy_per_batch"]
    require(type(repetitions) is int and repetitions >= 3 and type(warmup) is int and warmup >= 1,
            "Insufficient benchmark repetition or warmup")
    raw = costs["raw_samples"]
    expected_keys = {(rep, batch, policy) for rep in range(repetitions)
                     for batch in range(len(batches)) for policy in POLICIES}
    actual_keys = set()
    for sample in raw:
        key = (sample["repetition"], sample["batch_index"], sample["policy"])
        require(key in expected_keys and key not in actual_keys, "Unexpected or duplicate timing sample")
        actual_keys.add(key)
        require(type(sample["seconds"]) in (int, float) and math.isfinite(sample["seconds"])
                and sample["seconds"] > 0, "Invalid raw timing")
        offset = (key[0] + key[1]) % len(POLICIES)
        require(sample["policy_order_in_batch"] == (POLICIES.index(key[2]) - offset) % len(POLICIES),
                "Timing policy rotation changed")
        for field in ("peak_gpu_allocated_gib", "peak_incremental_gpu_allocated_gib",
                      "peak_gpu_reserved_gib"):
            require(type(sample[field]) in (int, float) and math.isfinite(sample[field])
                    and sample[field] >= 0, "Invalid GPU memory measurement")
        require(sample["peak_incremental_gpu_allocated_gib"] <= sample["peak_gpu_allocated_gib"]
                <= sample["peak_gpu_reserved_gib"], "Inconsistent GPU memory measurements")
    require(actual_keys == expected_keys, "Missing timing samples")
    rebuilt = {}
    for policy in POLICIES:
        samples = [sample for sample in raw if sample["policy"] == policy]
        aggregates = [sum(sample["seconds"] for sample in samples if sample["repetition"] == rep)
                      for rep in range(repetitions)]
        rebuilt[policy] = {
            "aggregate_seconds": sample_statistics(aggregates), "aggregate_raw_seconds": aggregates,
            "peak_gpu_allocated_gib": max(sample["peak_gpu_allocated_gib"] for sample in samples),
            "peak_incremental_gpu_allocated_gib": max(
                sample["peak_incremental_gpu_allocated_gib"] for sample in samples),
            "by_batch": [{
                "batch_index": index,
                "seconds": sample_statistics(sample["seconds"] for sample in samples
                                             if sample["batch_index"] == index),
                "raw_seconds": [sample["seconds"] for sample in samples if sample["batch_index"] == index],
                "peak_gpu_allocated_gib": max(sample["peak_gpu_allocated_gib"] for sample in samples
                                              if sample["batch_index"] == index),
            } for index in range(len(batches))],
        }
    for policy in POLICIES:
        rebuilt[policy]["speedup_vs_baseline_r32"] = (
            rebuilt["baseline_r32"]["aggregate_seconds"]["median"]
            / rebuilt[policy]["aggregate_seconds"]["median"]
        )
    match(costs["timing"], rebuilt, "Reconstructed timing")
    return {
        "passed": True, "costs_sha256": sha256_file(costs_path),
        "verified_source_files": len(hashes), "verified_native_files": len(provenance["native_source_hashes"]),
        "independent_retokenization": True, "verified_sampled_candidates": len(sampled_keys),
        "verified_raw_timing_samples": len(raw), "flops_recomputed": True,
        "timing_statistics_recomputed": True, "shared_evaluation_provenance": True,
        "standalone_validation_passed": True,
    }
