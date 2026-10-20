#!/usr/bin/env python3
"""Time baseline, LoopCD-Logits, and LoopCD-Hidden at one fixed Huginn depth."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_huginn import save_json, select_batches, slurm_allocation, utc_now
from src.hidden_costs import POLICIES, comparison, prefill_comparison
from src.huginn_costs import sample_statistics
from src.registry import load_registry, resolve_run
from src.run_integrity import native_source_hashes, sha256_file


def summarize_timing(samples, batches, repetitions):
    expected = {(policy, batch, repetition) for policy in POLICIES
                for batch in range(batches) for repetition in range(repetitions)}
    observed = [(row["policy"], row["batch_index"], row["repetition"]) for row in samples]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Timing must contain every policy/batch/repetition exactly once")
    timing = {}
    for policy in POLICIES:
        rows = [row for row in samples if row["policy"] == policy]
        aggregates = [sum(row["seconds"] for row in rows if row["repetition"] == repeat)
                      for repeat in range(repetitions)]
        timing[policy] = {
            "aggregate_seconds": sample_statistics(aggregates), "aggregate_raw_seconds": aggregates,
            "peak_gpu_allocated_gib": max(row["peak_gpu_allocated_gib"] for row in rows),
            "peak_incremental_gpu_allocated_gib": max(row["peak_incremental_gpu_allocated_gib"] for row in rows),
            "by_batch": [{
                "batch_index": index,
                "seconds": sample_statistics(row["seconds"] for row in rows if row["batch_index"] == index),
                "raw_seconds": [row["seconds"] for row in rows if row["batch_index"] == index],
                "peak_gpu_allocated_gib": max(row["peak_gpu_allocated_gib"] for row in rows if row["batch_index"] == index),
            } for index in range(batches)],
        }
    baseline = timing["baseline"]["aggregate_seconds"]["median"]
    logits = timing["loopcd_logits"]["aggregate_seconds"]["median"]
    for row in timing.values():
        median = row["aggregate_seconds"]["median"]
        if median <= 0:
            raise ValueError("Standalone timing must be positive")
        row["latency_ratio_to_baseline"] = median / baseline
        row["speedup_vs_baseline"] = baseline / median
        row["latency_ratio_to_logits"] = median / logits
        row["speedup_vs_logits"] = logits / median
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _, tasks = load_registry()
    parser.add_argument("--task", choices=list(tasks["tasks"]), required=True)
    parser.add_argument("--model", choices=("huginn_16", "huginn_32"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flops-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Existing results are preserved; choose a new output: {args.output}")
    batch_size, batch_count, warmup, repetitions = 8, 3, 2, 6
    resolved = resolve_run(args.task, args.model, guidance_space="hidden")
    model = resolved["model"]
    loops, reference_step, strength = model["loops"], model["reference_step"], model["fixed"]
    if reference_step != {16: 7, 32: 6}[loops] or strength != 0.5:
        raise ValueError("Use the fixed paper setting: R16/h7 or R32/h6, strength 0.5")
    config_path = ROOT / model["model_path"] / "config.json"
    model_config = json.loads(config_path.read_text())
    report = {
        "started_utc": utc_now(), "state": "running", "complete": False,
        "task": args.task, "model": args.model, "loops": loops, "shots": resolved["shots"],
        "hidden_reference_step": reference_step, "fixed_strength": strength,
        "hidden_setting": resolved["hidden_setting"],
        "flops_512_token_prefill": prefill_comparison(model_config, loops),
        "checkpoint_config": model_config, "checkpoint_config_sha256": sha256_file(config_path),
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
        raise RuntimeError("An allocated scavenger-h200 H200 is required")
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
        raise ValueError("Prepared task data do not match their recorded protocol")
    paths = [
        Path(__file__), ROOT / "scripts/benchmark_huginn.py", ROOT / "src/hidden_costs.py",
        ROOT / "src/huginn_costs.py", ROOT / "src/adapter_huginn.py", ROOT / "src/suite_scoring.py",
        ROOT / "src/loopcd.py", ROOT / "src/evaluation.py", ROOT / "src/run_integrity.py",
        ROOT / "src/registry.py", ROOT / "configs/models.json", ROOT / "configs/tasks.json",
        ROOT / "configs/hidden.json", config_path,
    ]
    snapshot = args.output.parent / (args.output.stem + "_source_snapshot")
    for path in paths:
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    report.update({
        "methodology": {
            "policies": {
                "baseline": {"loops": loops, "reference_step": None, "coda_passes": 1},
                "loopcd_logits": {"loops": loops, "reference_step": 1, "fixed_strength": strength, "coda_passes": 2},
                "loopcd_hidden": {"loops": loops, "reference_step": reference_step, "fixed_strength": strength, "coda_passes": 1},
            },
            "selection": "Sort all task candidates by input length, question index, and choice index; use eight-candidate slices centered at length quantiles 0.25, 0.50, and 0.75.",
            "scope": "Independent standalone teacher-forced answer-token likelihood scoring at the same recurrent depth.",
            "hidden_intervention": "Blend full-sequence raw recurrent states before pre-coda normalization; perform one complete coda and vocabulary head.",
            "timed_operations": "Input validation, initial-state clone, embedding, prelude, recurrence, policy-specific readout and guidance, log_softmax, and target gather; perf_counter with CUDA synchronization before and after.",
            "excluded_from_policy_timing": "Checkpoint loading, tokenizer/data preparation, tensor construction/transfer, initial random-state generation/transfer, warmup, and serialization.",
            "initial_state_handling": "Identical precomputed per-question initial states supplied to every independent policy forward; normal CPU generation plus GPU transfer is separately timed and checked.",
            "initial_state_storage": "All three precomputed batch states remain on GPU during measurements.",
            "output_projection": "Full vocabulary at answer-predicting positions only.",
            "cache": False, "dtype": "float32", "float32_matmul_precision": "highest",
            "warmup_per_policy_per_batch": warmup, "repetitions_per_policy_per_batch": repetitions,
            "order": "Repetition r rotates batch order by r; batch b rotates policy order by r+b. Each call performs a separate forward.",
            "aggregate": "Sum one latency from each sampled batch per repetition, then report median, quartiles and MAD. Ratios compare policy aggregate medians.",
            "interpretation": "Sampled scoring latency; Hidden has unchanged matrix-product FLOPs but additional reference capture and vector operations.",
        },
        "provenance": {
            "data_sha256": sha256_file(data_path), "protocol_sha256": sha256_file(protocol_path),
            "model_manifest_sha256": sha256_file(ROOT / model["model_manifest"]),
            "hidden_configuration_sha256": sha256_file(ROOT / "configs/hidden.json"),
            "code_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
            "native_source_hashes": native_source_hashes("huginn", ROOT / model["model_path"]),
            "paper_source_sha256": sha256_file(ROOT / "sources/looped_qwen/loopcd.html"),
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "host": socket.gethostname(), "slurm": allocation, "gpu": torch.cuda.get_device_name(),
            "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30, "seed": 1234,
        },
        "raw_samples": [], "batches": [],
    })
    save_json(args.output, report)
    print("LOADING", args.model, "float32 highest", flush=True)
    load_start = time.perf_counter()
    hidden = HuginnAdapter(ROOT / model["model_path"], device="cuda:0", dtype=torch.float32,
                           loops=loops, reference_step=reference_step)
    logits_adapter = HuginnAdapter.from_model(hidden.model, hidden.tokenizer, loops=loops, reference_step=1)
    torch.cuda.synchronize()
    report["checkpoint_load_seconds"] = time.perf_counter() - load_start
    report["adapter_metadata"] = {"hidden": hidden.metadata(), "logits": logits_adapter.metadata()}
    candidates, truncation = build_candidates(hidden.tokenizer, documents, hidden.max_length,
                                               bos_token_id=hidden.bos_token_id)
    report["truncation_details"] = truncation
    scorer = SuiteScorer(logits_adapter, [row["id"] for row in documents], fixed=strength, cap=0.5)
    selected = select_batches(candidates, batch_size, batch_count)
    prepared = []
    for batch_index, (quantile, start, batch) in enumerate(selected):
        inputs, mask, rows, positions, targets, keys = scorer.tensors(batch)
        initial = hidden.initial_states(keys, inputs.shape[1])
        torch.cuda.synchronize()
        noise_times = []
        for _ in range(repetitions):
            before = time.perf_counter()
            repeated = hidden.initial_states(keys, inputs.shape[1])
            torch.cuda.synchronize()
            noise_times.append(time.perf_counter() - before)
            torch.testing.assert_close(initial, repeated, atol=0, rtol=0)
            del repeated
        prepared.append((inputs, mask, rows, positions, targets, keys, initial))
        report["batches"].append({
            "batch_index": batch_index, "length_quantile": quantile, "start_in_length_sorted_candidates": start,
            "candidate_count": len(batch), "padded_length": inputs.shape[1],
            "input_tokens_without_padding": sum(len(candidate.input_ids) for candidate in batch),
            "answer_prediction_positions": targets.numel(),
            "candidates": [{"question_index": c.question_index, "question_id": documents[c.question_index]["id"],
                            "choice_index": c.choice_index, "input_length": len(c.input_ids), "target_ids": c.targets}
                           for c in batch],
            "initial_state_generation_seconds": sample_statistics(noise_times),
            "initial_state_generation_raw_seconds": noise_times,
            "flops": comparison(model_config, loops, sequence_length=inputs.shape[1],
                                batch_size=inputs.shape[0], head_positions=targets.numel()),
        })

    @torch.inference_mode()
    def forward(policy, batch):
        inputs, mask, rows, positions, targets, keys, initial = batch
        if policy == "loopcd_hidden":
            values = hidden.forward_hidden_logits(inputs, mask, rows, positions, sample_keys=keys,
                                                   strength=strength, initial_states=initial)
        elif policy == "loopcd_logits":
            pair = logits_adapter.forward_logits(inputs, mask, rows, positions, sample_keys=keys,
                                                   initial_states=initial)
            values = guided_logits(pair["baseline"], pair["reference"], strength)
        else:
            values = hidden.forward_baseline_logits(inputs, mask, rows, positions, sample_keys=keys,
                                                     initial_states=initial)
        return values.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)

    inputs, mask, rows, positions, targets, keys, initial = prepared[0]
    with torch.inference_mode():
        pair = hidden.forward_hidden_comparison_logits(inputs, mask, rows, positions, sample_keys=keys,
                                                       strength=strength)
        report["validation"] = {}
        for policy in ("baseline", "loopcd_hidden"):
            expected = pair[policy].log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            actual = forward(policy, prepared[0])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            report["validation"][policy] = {
                "standalone_matches_shared_comparison_exactly": True,
                "max_absolute_token_logprob_difference": (actual - expected).abs().max().item(),
                "normal_initial_state_generation_matches_precomputed": True,
            }
            del expected, actual
        del pair
        ordinary = scorer.score(selected[0][2])
        expected = torch.tensor([value for row in ordinary for value in row["token_logprobs"]["loopcd_fixed"]],
                                dtype=torch.float32, device=hidden.device)
        actual = forward("loopcd_logits", prepared[0])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        report["validation"]["loopcd_logits"] = {
            "matches_shared_evaluator_token_logprobs_exactly": True,
            "max_absolute_token_logprob_difference": (actual - expected).abs().max().item(),
            "normal_initial_state_generation_matches_precomputed": True,
        }
        del actual, expected
    for batch in prepared:
        for _ in range(warmup):
            for policy in POLICIES:
                result = forward(policy, batch)
                torch.cuda.synchronize()
                if not bool(torch.isfinite(result).all()):
                    raise ValueError(f"Nonfinite warmup scores: {policy}")
                del result
    gc.collect()
    print("WARMUP_COMPLETE", json.dumps({"batch_shapes": [[r["candidate_count"], r["padded_length"]]
                                                           for r in report["batches"]]}), flush=True)
    for repetition in range(repetitions):
        batch_order = list(range(batch_count))
        offset = repetition % batch_count
        batch_order = batch_order[offset:] + batch_order[:offset]
        for batch_index in batch_order:
            offset = (repetition + batch_index) % len(POLICIES)
            for order, policy in enumerate(POLICIES[offset:] + POLICIES[:offset]):
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                result = forward(policy, prepared[batch_index])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                sample = {
                    "repetition": repetition, "batch_index": batch_index,
                    "policy": policy, "policy_order_in_batch": order, "seconds": elapsed,
                    "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_incremental_gpu_allocated_gib": (torch.cuda.max_memory_allocated() - resident) / 2**30,
                    "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
                if not bool(torch.isfinite(result).all()):
                    raise ValueError(f"Nonfinite timed scores: {policy}")
                del result
                report["raw_samples"].append(sample)
        save_json(args.output, report)
        print("TIMING_REPETITION", repetition + 1, "/", repetitions, flush=True)
    report.update(timing=summarize_timing(report["raw_samples"], batch_count, repetitions),
                  state="complete", complete=True, finished_utc=utc_now())
    save_json(args.output, report)
    print("BENCHMARK_COMPLETE", json.dumps({policy: row["aggregate_seconds"]["median"]
                                           for policy, row in report["timing"].items()}), flush=True)


if __name__ == "__main__":
    main()
