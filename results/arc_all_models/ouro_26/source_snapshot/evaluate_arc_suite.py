#!/usr/bin/env python3
"""One frozen Table 2 model configuration on the full ARC-Easy test split."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import transformers

from src.loopcd import MODES, encode_candidate
from src.suite_scoring import SuiteScorer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(documents, completed, output, timing):
    correct = {metric: {mode: 0 for mode in MODES} for metric in ("acc", "acc_norm")}
    predictions = []
    for index, doc in enumerate(documents):
        candidates = [completed[(index, choice)] for choice in range(len(doc["choices"]))]
        scores = {mode: [row["scores"][mode] for row in candidates] for mode in MODES}
        normalized = {
            mode: [score / len(choice) for score, choice in zip(values, doc["choices"])]
            for mode, values in scores.items()
        }
        verdicts = {}
        for metric, by_mode in (("acc", scores), ("acc_norm", normalized)):
            verdicts[metric] = {}
            for mode, values in by_mode.items():
                chosen = max(range(len(values)), key=values.__getitem__)
                verdicts[metric][mode] = chosen
                correct[metric][mode] += int(chosen == doc["gold"])
        predictions.append({
            "id": doc["id"], "question": doc["question"], "choices": doc["choices"],
            "labels": doc["labels"], "gold": doc["gold"], "scores": scores,
            "scores_per_character": normalized, "predictions": verdicts,
        })
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    result = {
        "n": len(documents), "correct": correct,
        "accuracy_percent": {
            metric: {mode: 100*count/len(documents) for mode,count in values.items()}
            for metric,values in correct.items()
        },
        **timing,
        "note": "Joint readouts share a recurrent trajectory; runtime is not a policy speedup measurement.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--matmul-precision", choices=("high", "highest"), default="high")
    args = parser.parse_args()
    registry = json.loads((ROOT / "configs/arc_all_models.json").read_text())
    config = registry["models"][args.name]
    if args.limit and args.output is None:
        raise ValueError("Pilot runs require a separate --output directory.")
    output = args.output or ROOT / "results/arc_all_models" / args.name
    output.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or config["batch_size"]
    torch.set_num_threads(min(6, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(registry["seed"])
    random.seed(registry["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA GPU is required; do not load these models on the login process.")
    family = config["family"]
    module = importlib.import_module("src.adapter_" + family)
    cls = getattr(module, {
        "ouro": "OuroAdapter", "huginn": "HuginnAdapter",
        "parcae": "ParcaeAdapter", "looped_qwen": "LoopedQwenAdapter",
    }[family])
    start = time.perf_counter()
    print("LOADING", args.name, args.dtype, torch.cuda.get_device_name(), flush=True)
    adapter = cls(ROOT / config["model_path"], device="cuda:0", dtype=getattr(torch, args.dtype), loops=config["loops"])
    data_path = ROOT / f"data/arc_easy_{config['shots']}shot.jsonl"
    documents = [json.loads(line) for line in data_path.read_text().splitlines()]
    if args.limit:
        documents = documents[:args.limit]
    candidates = [
        encode_candidate(adapter.tokenizer, doc["context"], choice, index, choice_index)
        for index, doc in enumerate(documents) for choice_index, choice in enumerate(doc["choices"])
    ]
    add_bos = bool(getattr(adapter, "add_bos", False))
    if add_bos:
        bos = getattr(adapter, "bos_token_id", adapter.tokenizer.bos_token_id)
        assert bos is not None
        for candidate in candidates:
            candidate.input_ids.insert(0, bos)
            candidate.positions = [position + 1 for position in candidate.positions]
    scorer = SuiteScorer(adapter, [row["id"] for row in documents], config["fixed"], config["cap"])
    validation = scorer.validate(candidates)
    print("VALIDATION", json.dumps(validation), flush=True)
    code_paths = [
        Path(__file__), ROOT / "src/suite_scoring.py", ROOT / "src/loopcd.py", Path(module.__file__),
    ]
    adapter_metadata = getattr(adapter, "metadata", {})
    if callable(adapter_metadata):
        adapter_metadata = adapter_metadata()
    manifest = {
        "name": args.name, "configuration": config,
        "documents": len(documents), "candidates": len(candidates),
        "data_sha256": digest(data_path),
        "model_manifest_sha256": digest(ROOT / config["model_manifest"]),
        "code_hashes": {str(path.relative_to(ROOT)): digest(path) for path in code_paths},
        "dtype": args.dtype, "float32_matmul_precision": args.matmul_precision,
        "batch_size": batch_size, "token_budget": args.token_budget,
        "seed": registry["seed"], "add_bos": add_bos, "chat_template": False,
        "max_input_tokens": max(len(candidate.input_ids) for candidate in candidates),
        "truncation": False, "torch": torch.__version__, "transformers": transformers.__version__,
        "adapter_metadata": adapter_metadata,
        "gpu": torch.cuda.get_device_name(), "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("configuration", "documents", "data_sha256", "model_manifest_sha256", "code_hashes",
                    "dtype", "float32_matmul_precision", "batch_size", "token_budget", "add_bos"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume configuration changed: {key}")
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in code_paths:
        shutil.copy2(path, snapshot / path.name)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    score_path = output / "candidate_scores.jsonl"
    completed = {}
    if score_path.exists():
        for line in score_path.read_text().splitlines():
            row = json.loads(line)
            key = (row["question_index"], row["choice_index"])
            if key in completed:
                raise ValueError("Duplicate saved candidate")
            completed[key] = row
    pending = [candidate for candidate in candidates if (candidate.question_index,candidate.choice_index) not in completed]
    pending.sort(key=lambda candidate: len(candidate.input_ids))
    init_seconds = time.perf_counter() - start
    score_start = time.perf_counter()
    cursor, steps = 0, 0
    with score_path.open("a", buffering=1) as handle:
        while cursor < len(pending):
            end = min(cursor + batch_size, len(pending))
            while (end-cursor) * len(pending[end-1].input_ids) > args.token_budget and end > cursor+1:
                end -= 1
            for row in scorer.score(pending[cursor:end]):
                handle.write(json.dumps(row) + "\n")
                completed[(row["question_index"],row["choice_index"])] = row
            cursor, steps = end, steps+1
            if steps % 10 == 0 or cursor == len(pending):
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - score_start
                print(json.dumps({
                    "completed": len(completed), "total": len(candidates),
                    "seconds": round(elapsed,2), "candidates_per_second": round(cursor/elapsed,2),
                    "remaining_seconds": round((len(pending)-cursor)*elapsed/cursor,1),
                    "peak_gpu_gib": round(torch.cuda.max_memory_allocated()/2**30,3),
                }), flush=True)
    torch.cuda.synchronize()
    summarize(documents, completed, output, {
        "load_and_validation_seconds": init_seconds,
        "this_invocation_scoring_seconds": time.perf_counter()-score_start,
        "this_invocation_total_seconds": time.perf_counter()-start,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
    })


if __name__ == "__main__":
    main()
