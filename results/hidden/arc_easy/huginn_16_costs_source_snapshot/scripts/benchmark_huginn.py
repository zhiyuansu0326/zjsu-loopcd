#!/usr/bin/env python3
"""Compare standalone Huginn R32, R16, and R16 LoopCD on identical task batches."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.huginn_costs import comparison, paper_prefill_comparison, sample_statistics
from src.registry import load_registry, resolve_run
from src.run_integrity import native_source_hashes, sha256_file


POLICIES = ("baseline_r32", "baseline_r16", "loopcd_r16_ref5")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def select_batches(candidates, batch_size, count):
    """Fixed length-stratified contiguous slices; selection never uses labels."""
    ordered = sorted(candidates, key=lambda candidate: (
        len(candidate.input_ids), candidate.question_index, candidate.choice_index,
    ))
    if len(ordered) < batch_size * count:
        raise ValueError("Not enough candidates for distinct benchmark batches")
    batches = []
    for index in range(count):
        quantile = (index + 1) / (count + 1)
        center = int(quantile * (len(ordered) - 1))
        start = min(max(0, center - batch_size // 2), len(ordered) - batch_size)
        batches.append((quantile, start, ordered[start:start + batch_size]))
    keys = [(candidate.question_index, candidate.choice_index)
            for _, _, batch in batches for candidate in batch]
    if len(set(keys)) != len(keys):
        raise ValueError("Length-quantile slices overlap; use fewer batches")
    return batches


def slurm_allocation():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the benchmark in an allocated scavenger-h200 GPU job")
    if os.environ.get("SLURM_JOB_PARTITION") != "scavenger-h200":
        raise RuntimeError("Only the scavenger-h200 partition is authorized")
    command = ["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]]
    description = subprocess.check_output(command, text=True).strip()
    fields = dict(field.split("=", 1) for field in description.split() if "=" in field)
    if fields.get("Account") != "scavenger-h200" or fields.get("Partition") != "scavenger-h200":
        raise RuntimeError("Both Slurm account and partition must be scavenger-h200")
    return {name: fields.get(name) for name in (
        "JobId", "Account", "Partition", "QOS", "NodeList", "AllocTRES", "TresPerNode",
    )}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _, tasks = load_registry()
    parser.add_argument("--task", choices=list(tasks["tasks"]), default="arc_challenge")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--flops-only", action="store_true",
                        help="Compute 512-token analytic FLOPs without loading weights or using a GPU")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path; existing results are preserved: {args.output}")
    if min(args.batch_size, args.batches, args.warmup) < 1 or args.repetitions < 3:
        raise ValueError("Positive batch/warmup counts and at least three repetitions are required")
    resolved = resolve_run(args.task, "huginn_16")
    config_path = ROOT / resolved["model"]["model_path"] / "config.json"
    model_config = json.loads(config_path.read_text())
    report = {
        "started_utc": utc_now(), "state": "running", "complete": False,
        "task": args.task, "shots": resolved["shots"],
        "flops_512_token_prefill": paper_prefill_comparison(model_config),
        "checkpoint_config": model_config,
        "checkpoint_config_sha256": sha256_file(config_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.flops_only:
        report.update(state="analytic_only", complete=True, finished_utc=utc_now())
        save_json(args.output, report)
        print(json.dumps(report["flops_512_token_prefill"], indent=2))
        return

    allocation = slurm_allocation()
    import torch
    import transformers
    from src.adapter_huginn import HuginnAdapter
    from src.evaluation import build_candidates, validate_documents
    from src.loopcd import guided_logits
    from src.suite_scoring import SuiteScorer

    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name():
        raise RuntimeError("An allocated H200 is required for this benchmark")
    torch.set_num_threads(min(6, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(1234)
    data_path, protocol_path = resolved["data_path"], resolved["protocol_path"]
    documents = [json.loads(line) for line in data_path.read_text().splitlines()]
    validate_documents(documents, resolved["task"]["count"])
    protocol = json.loads(protocol_path.read_text())
    recorded = protocol.get("outputs", {}).get(str(resolved["shots"]), protocol.get("output"))
    expected_hash = recorded["sha256"] if isinstance(recorded, dict) else protocol.get("output_sha256")
    if expected_hash is None or sha256_file(data_path) != expected_hash:
        raise ValueError("Prepared task data do not match the recorded protocol")

    paths = [
        Path(__file__), ROOT / "src/huginn_costs.py", ROOT / "src/adapter_huginn.py",
        ROOT / "src/suite_scoring.py", ROOT / "src/loopcd.py", ROOT / "src/evaluation.py",
        ROOT / "src/run_integrity.py", ROOT / "src/registry.py",
        ROOT / "configs/models.json", ROOT / "configs/tasks.json", config_path,
    ]
    source_snapshot = args.output.parent / (args.output.stem + "_source_snapshot")
    for path in paths:
        destination = source_snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    report.update({
        "methodology": {
            "policies": {
                "baseline_r32": {"loops": 32, "reference_step": None, "coda_passes": 1},
                "baseline_r16": {"loops": 16, "reference_step": None, "coda_passes": 1},
                "loopcd_r16_ref5": {
                    "loops": 16, "reference_step": 5, "fixed_strength": 0.5, "coda_passes": 2,
                },
            },
            "selection": (
                "Sort all task candidates by input length, question index, and choice index. "
                "Use contiguous fixed-size slices centered on i/(batches+1) length quantiles."
            ),
            "scope": "Standalone teacher-forced answer-token log probabilities on sampled batches.",
            "timed_operations": (
                "Input validation, initial-state clone, embedding, prelude, recurrence, "
                "policy-specific coda/head, fixed guidance if used, log_softmax and target gather. "
                "CUDA is synchronized before and after each perf_counter measurement."
            ),
            "excluded_from_policy_timing": (
                "Checkpoint loading, tokenizer/data preparation, tensor construction/transfer, "
                "initial random-state generation and its transfer, warmup, and result serialization."
            ),
            "initial_state_handling": (
                "One stable CPU truncated-normal state per input batch is transferred before "
                "timing and reused unchanged for every policy and repetition. Separate timings "
                "record the actual adapter state generation plus GPU transfer."
            ),
            "initial_state_storage": "Precomputed initial states for all sampled batches remain on GPU.",
            "output_projection": "All vocabulary logits at answer-predicting positions only.",
            "cache": False, "dtype": "float32", "float32_matmul_precision": "highest",
            "warmup_per_policy_per_batch": args.warmup,
            "repetitions_per_policy_per_batch": args.repetitions,
            "order": (
                "Repetition r rotates batch order by r; within batch b it rotates policy order "
                "by r+b. Each measurement uses an independent forward pass."
            ),
            "aggregate": (
                "Sum one measured latency from each sampled batch within a repetition; "
                "report median, quartiles and MAD across repetitions. Speedup is the ratio "
                "of baseline-R32 median aggregate latency to policy median aggregate latency."
            ),
            "interpretation": "Sampled inference latency; full-task evaluation throughput is not measured.",
        },
        "provenance": {
            "data_sha256": sha256_file(data_path), "protocol_sha256": sha256_file(protocol_path),
            "model_manifest_sha256": sha256_file(ROOT / resolved["model"]["model_manifest"]),
            "code_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
            "native_source_hashes": native_source_hashes("huginn", ROOT / resolved["model"]["model_path"]),
            "paper_source_sha256": sha256_file(ROOT / "sources/looped_qwen/loopcd.html"),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "host": socket.gethostname(),
            "slurm": allocation, "gpu": torch.cuda.get_device_name(),
            "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "seed": 1234,
        },
        "raw_samples": [], "batches": [],
    })
    save_json(args.output, report)
    print("LOADING Huginn float32 highest", flush=True)
    load_start = time.perf_counter()
    half = HuginnAdapter(
        ROOT / resolved["model"]["model_path"], device="cuda:0",
        dtype=torch.float32, loops=16, reference_step=5,
    )
    full = HuginnAdapter.from_model(half.model, half.tokenizer, loops=32, reference_step=5)
    torch.cuda.synchronize()
    report["checkpoint_load_seconds"] = time.perf_counter() - load_start
    report["adapter_metadata"] = half.metadata()
    candidates, truncation = build_candidates(
        half.tokenizer, documents, half.max_length, bos_token_id=half.bos_token_id,
    )
    report["truncation_details"] = truncation
    scorer = SuiteScorer(half, [document["id"] for document in documents], fixed=0.5, cap=0.5)
    prepared = []
    selected_batches = select_batches(candidates, args.batch_size, args.batches)
    for batch_index, (quantile, start, batch) in enumerate(selected_batches):
        inputs, mask, rows, positions, targets, keys = scorer.tensors(batch)
        initial = half.initial_states(keys, inputs.shape[1])
        torch.cuda.synchronize()
        noise_seconds = []
        for _ in range(args.repetitions):
            before = time.perf_counter()
            repeated_initial = half.initial_states(keys, inputs.shape[1])
            torch.cuda.synchronize()
            noise_seconds.append(time.perf_counter() - before)
            torch.testing.assert_close(initial, repeated_initial, atol=0, rtol=0)
            del repeated_initial
        prepared.append((inputs, mask, rows, positions, targets, keys, initial))
        report["batches"].append({
            "batch_index": batch_index, "length_quantile": quantile,
            "start_in_length_sorted_candidates": start,
            "candidate_count": len(batch), "padded_length": inputs.shape[1],
            "input_tokens_without_padding": sum(len(candidate.input_ids) for candidate in batch),
            "answer_prediction_positions": targets.numel(),
            "candidates": [{
                "question_index": candidate.question_index, "question_id": documents[candidate.question_index]["id"],
                "choice_index": candidate.choice_index, "input_length": len(candidate.input_ids),
                "target_ids": candidate.targets,
            } for candidate in batch],
            "initial_state_generation_seconds": sample_statistics(noise_seconds),
            "initial_state_generation_raw_seconds": noise_seconds,
            "flops": comparison(
                model_config, sequence_length=inputs.shape[1],
                batch_size=inputs.shape[0], head_positions=targets.numel(),
            ),
        })

    @torch.inference_mode()
    def forward(policy, batch):
        inputs, mask, rows, positions, targets, keys, initial = batch
        adapter = full if policy == "baseline_r32" else half
        if policy == "loopcd_r16_ref5":
            outputs = adapter.forward_logits(
                inputs, mask, rows, positions, sample_keys=keys, initial_states=initial,
            )
            logits = guided_logits(outputs["baseline"], outputs["reference"], 0.5)
        else:
            logits = adapter.forward_baseline_logits(
                inputs, mask, rows, positions, sample_keys=keys, initial_states=initial,
            )
        return logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)

    # Check that the timed baseline route preserves the ordinary evaluator output.
    inputs, mask, rows, positions, _, keys, initial = prepared[0]
    report["validation"] = {}
    with torch.inference_mode():
        for name, adapter in (("baseline_r32", full), ("baseline_r16", half)):
            pair = adapter.forward_logits(
                inputs, mask, rows, positions, sample_keys=keys, initial_states=initial,
            )
            baseline = adapter.forward_baseline_logits(
                inputs, mask, rows, positions, sample_keys=keys, initial_states=initial,
            )
            torch.testing.assert_close(baseline, pair["baseline"], atol=0, rtol=0)
            report["validation"][name] = {
                "standalone_matches_joint_baseline_exactly": True,
                "max_absolute_logit_difference": (baseline - pair["baseline"]).abs().max().item(),
            }
            del pair, baseline
        ordinary_scores = scorer.score(selected_batches[0][2])
        ordinary_fixed = torch.tensor([
            value for row in ordinary_scores for value in row["token_logprobs"]["loopcd_fixed"]
        ], dtype=torch.float32, device=half.device)
        benchmark_fixed = forward("loopcd_r16_ref5", prepared[0])
        torch.testing.assert_close(benchmark_fixed, ordinary_fixed, atol=0, rtol=0)
        report["validation"]["loopcd_r16_ref5"] = {
            "matches_shared_evaluator_token_logprobs_exactly": True,
            "max_absolute_token_logprob_difference": (benchmark_fixed - ordinary_fixed).abs().max().item(),
            "precomputed_noise_matches_normal_state_generation": True,
        }
        del ordinary_fixed, benchmark_fixed
    for batch in prepared:
        for _ in range(args.warmup):
            for policy in POLICIES:
                result = forward(policy, batch)
                torch.cuda.synchronize()
                if not bool(torch.isfinite(result).all()):
                    raise ValueError(f"Nonfinite outputs in {policy}")
                del result
    gc.collect()
    print("WARMUP_COMPLETE", json.dumps({
        "batch_shapes": [[row["candidate_count"], row["padded_length"]] for row in report["batches"]],
        "repetitions": args.repetitions,
    }), flush=True)
    for repetition in range(args.repetitions):
        batch_order = list(range(args.batches))
        batch_order = batch_order[repetition % args.batches:] + batch_order[:repetition % args.batches]
        for batch_index in batch_order:
            offset = (repetition + batch_index) % len(POLICIES)
            policy_order = POLICIES[offset:] + POLICIES[:offset]
            for order, policy in enumerate(policy_order):
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                result = forward(policy, prepared[batch_index])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                sample = {
                    "repetition": repetition, "batch_index": batch_index,
                    "policy": policy, "policy_order_in_batch": order,
                    "seconds": elapsed,
                    "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_incremental_gpu_allocated_gib": (
                        torch.cuda.max_memory_allocated() - resident
                    ) / 2**30,
                    "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
                if not bool(torch.isfinite(result).all()):
                    raise ValueError(f"Nonfinite outputs in {policy}")
                del result
                report["raw_samples"].append(sample)
        save_json(args.output, report)
        print("TIMING_REPETITION", repetition + 1, "/", args.repetitions, flush=True)
    timing = {}
    for policy in POLICIES:
        samples = [sample for sample in report["raw_samples"] if sample["policy"] == policy]
        aggregates = [
            sum(sample["seconds"] for sample in samples if sample["repetition"] == repetition)
            for repetition in range(args.repetitions)
        ]
        timing[policy] = {
            "aggregate_seconds": sample_statistics(aggregates),
            "aggregate_raw_seconds": aggregates,
            "peak_gpu_allocated_gib": max(sample["peak_gpu_allocated_gib"] for sample in samples),
            "peak_incremental_gpu_allocated_gib": max(
                sample["peak_incremental_gpu_allocated_gib"] for sample in samples
            ),
            "by_batch": [{
                "batch_index": index,
                "seconds": sample_statistics(sample["seconds"] for sample in samples if sample["batch_index"] == index),
                "raw_seconds": [sample["seconds"] for sample in samples if sample["batch_index"] == index],
                "peak_gpu_allocated_gib": max(
                    sample["peak_gpu_allocated_gib"] for sample in samples if sample["batch_index"] == index
                ),
            } for index in range(args.batches)],
        }
    for result in timing.values():
        result["speedup_vs_baseline_r32"] = (
            timing["baseline_r32"]["aggregate_seconds"]["median"] / result["aggregate_seconds"]["median"]
        )
    report.update(timing=timing, state="complete", complete=True, finished_utc=utc_now())
    save_json(args.output, report)
    print("BENCHMARK_COMPLETE", json.dumps({
        policy: {"median_seconds": result["aggregate_seconds"]["median"],
                 "speedup": result["speedup_vs_baseline_r32"]}
        for policy, result in timing.items()
    }), flush=True)


if __name__ == "__main__":
    main()
