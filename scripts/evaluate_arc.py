#!/usr/bin/env python3
"""Score every ARC-Easy answer with baseline and paper-fixed LoopCD settings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from src.loopcd import MODES, OuroScorer, encode_candidate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/Ouro-1.4B")
    parser.add_argument("--data", type=Path, default=ROOT / "data/arc_easy_8shot.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/arc_easy")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="high")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(1234)
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.set_num_threads(min(6, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    if not torch.cuda.is_available():
        raise RuntimeError("Run this script inside a Slurm job with an allocated GPU.")
    started = time.perf_counter()
    print(f"Loading Ouro on {torch.cuda.get_device_name()} with {args.dtype}, matmul={args.matmul_precision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        dtype=getattr(torch, args.dtype), attn_implementation="sdpa", device_map={"": "cuda:0"}
    ).eval()
    scorer = OuroScorer(model, tokenizer)
    documents = [json.loads(line) for line in args.data.read_text().splitlines()]
    if args.limit:
        documents = documents[:args.limit]
    candidates = [
        encode_candidate(tokenizer, doc["context"], choice, index, choice_index)
        for index, doc in enumerate(documents)
        for choice_index, choice in enumerate(doc["choices"])
    ]
    validation = scorer.validate(candidates)
    print("VALIDATION", json.dumps(validation), flush=True)
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = {
        "model_path": str(args.model), "data_sha256": sha256(args.data),
        "loopcd_source_sha256": sha256(ROOT / "src/loopcd.py"),
        "evaluator_source_sha256": sha256(Path(__file__)),
        "model_manifest_sha256": sha256(ROOT / "models/model_manifest.json"),
        "documents": len(documents), "candidates": len(candidates),
        "seed": 1234, "recurrent_passes": 4, "reference_pass": 1,
        "fixed_strength": 0.5, "adaptive_cap": 1.0,
        "dtype": args.dtype, "logit_arithmetic": "float32",
        "float32_matmul_precision": args.matmul_precision,
        "tokenizer_add_special_tokens": False, "add_bos_token": False,
        "max_input_tokens": max(len(c.input_ids) for c in candidates),
        "attention": "sdpa", "batch_size": args.batch_size,
        "token_budget": args.token_budget,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(), "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "generation": False, "cache": False, "truncation": False,
    }
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("data_sha256", "loopcd_source_sha256", "evaluator_source_sha256",
                    "model_manifest_sha256", "documents", "batch_size", "token_budget", "dtype",
                    "float32_matmul_precision"):
            if previous[key] != manifest[key]:
                raise RuntimeError(f"Cannot resume incompatible run: {key}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result_path = args.output / "candidate_scores.jsonl"
    completed = {}
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            row = json.loads(line)
            key = (row["question_index"], row["choice_index"])
            if key in completed:
                raise RuntimeError("Duplicate saved candidate")
            completed[key] = row
    pending = [c for c in candidates if (c.question_index, c.choice_index) not in completed]
    pending.sort(key=lambda candidate: len(candidate.input_ids))
    load_seconds = time.perf_counter() - started
    scoring_started = time.perf_counter()
    cursor = 0
    with result_path.open("a", buffering=1) as handle:
        while cursor < len(pending):
            end = min(cursor + args.batch_size, len(pending))
            while (end - cursor) * len(pending[end - 1].input_ids) > args.token_budget and end > cursor + 1:
                end -= 1
            batch = pending[cursor:end]
            outputs = scorer.score(batch)
            for row in outputs:
                handle.write(json.dumps(row) + "\n")
                completed[(row["question_index"], row["choice_index"])] = row
            cursor = end
            if cursor % (args.batch_size * 10) == 0 or cursor == len(pending):
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - scoring_started
                print(json.dumps({
                    "completed_candidates": len(completed), "total_candidates": len(candidates),
                    "scoring_seconds": round(elapsed, 2),
                    "candidates_per_second": round(cursor / elapsed, 2),
                    "estimated_seconds_remaining": round((len(pending) - cursor) * elapsed / cursor, 1),
                    "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                }), flush=True)
    torch.cuda.synchronize()
    correct = {metric: {mode: 0 for mode in MODES} for metric in ("acc", "acc_norm")}
    predictions = []
    for index, doc in enumerate(documents):
        option_rows = [completed[(index, ci)] for ci in range(len(doc["choices"]))]
        scores = {mode: [row["scores"][mode] for row in option_rows] for mode in MODES}
        normalized = {
            mode: [value / len(choice) for value, choice in zip(values, doc["choices"])]
            for mode, values in scores.items()
        }
        predictions_by_metric = {}
        for metric, values in (("acc", scores), ("acc_norm", normalized)):
            predictions_by_metric[metric] = {}
            for mode in MODES:
                prediction = max(range(len(doc["choices"])), key=lambda ci: values[mode][ci])
                predictions_by_metric[metric][mode] = prediction
                correct[metric][mode] += int(prediction == doc["gold"])
        predictions.append({
            "id": doc["id"], "question": doc["question"], "choices": doc["choices"],
            "labels": doc["labels"], "gold": doc["gold"],
            "scores": scores, "scores_per_character": normalized,
            "predictions": predictions_by_metric,
        })
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    summary = {
        "n": len(documents),
        "correct": correct,
        "accuracy_percent": {
            metric: {mode: 100 * count / len(documents) for mode, count in counts.items()}
            for metric, counts in correct.items()
        },
        "load_and_validation_seconds": load_seconds,
        "this_invocation_scoring_seconds": time.perf_counter() - scoring_started,
        "this_invocation_total_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "note": "All policies share the same four-pass trajectory. Joint runtime is not a policy speedup measurement.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
