#!/usr/bin/env python3
"""Independently audit saved multiple-choice runs, without loading model weights.

--partial only prints JSON for complete runs. A final report requires every model
in each selected task. Historical ARC-Easy runs are audited in place and reused.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

from scripts import analyze_arc_suite as legacy
from src.registry import load_registry, resolve_run
from src.run_integrity import REQUIRED_MANIFEST_KEYS, native_source_hashes, sha256_file

MODES = legacy.MODES
GUIDED = legacy.GUIDED
METRICS = legacy.METRICS
REQUIRED = (*legacy.REQUIRED, "data_protocol.json")
MODEL_NAMES = {
    "ouro_14", "ouro_26", "huginn_16", "huginn_32",
    "parcae_370", "parcae_13", "looped_qwen",
}
TASK_NAMES = ("arc_easy", "arc_challenge", "sciq", "piqa")
MULTITASK_MANIFEST_KEYS = REQUIRED_MANIFEST_KEYS | {
    "task", "task_configuration", "shots", "primary_metric", "protocol_sha256",
    "context_policy", "truncation_details",
}
MODE_NAMES = {"loopcd_fixed": "Fixed", "loopcd_adaptive": "Adaptive"}
require = legacy.require
read_json = legacy.read_json
read_jsonl = legacy.read_jsonl
digest = sha256_file


def close(actual, expected, message, tolerance=1e-10):
    require(type(actual) in (int, float) and math.isfinite(actual),
            message + ": invalid or nonfinite number")
    require(abs(actual - expected) <= tolerance,
            f"{message}: {actual!r} != {expected!r}")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def audit_checkpoint(config, cache):
    """Hash metadata/tokenizers/native code; retain prior large-weight hashes."""
    path = ROOT / config["model_manifest"]
    key = ("checkpoint", str(path))
    if key in cache:
        return cache[key]
    manifest = read_json(path)
    require(isinstance(manifest.get("revision"), str) and len(manifest["revision"]) == 40
            and all(character in "0123456789abcdef" for character in manifest["revision"]),
            "Checkpoint revision is not a pinned commit")
    require("complete" not in manifest or manifest["complete"] is True,
            "Checkpoint manifest is explicitly incomplete")
    files = manifest["files"]
    require(bool(files), "Checkpoint manifest has no files")
    checked, sized = {}, {}
    seen = set()
    model_dir = (ROOT / config["model_path"]).resolve()
    for entry in files:
        name = entry.get("path", entry.get("filename"))
        require(isinstance(name, str) and name, "Checkpoint entry has no file path")
        file_path = ROOT / name if name.startswith("models/") else model_dir / name
        file_path = file_path.resolve()
        require(file_path.is_relative_to(model_dir),
                "Checkpoint file escapes the selected model directory")
        require(file_path not in seen, "Duplicate checkpoint manifest file")
        seen.add(file_path)
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0
                and file_path.is_file() and file_path.stat().st_size == entry["bytes"],
                f"Checkpoint file missing or changed size: {file_path}")
        require(isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64
                and all(character in "0123456789abcdef" for character in entry["sha256"]),
                f"Missing recorded checkpoint digest: {file_path}")
        if entry["bytes"] < 20 * 1024 * 1024:
            actual = digest(file_path)
            require(actual == entry["sha256"], f"Checkpoint metadata changed: {file_path}")
            checked[relative(file_path)] = actual
        else:
            sized[relative(file_path)] = {
                "bytes": entry["bytes"], "download_verified_sha256": entry["sha256"],
            }
    require(model_dir / "tokenizer.json" in seen and model_dir / "config.json" in seen,
            "Pinned checkpoint does not include its tokenizer/config")
    weights = {
        item for item in seen
        if (item.name == "model.safetensors"
            or (item.name.startswith("model-") and item.suffix == ".safetensors")
            or (item.name.startswith("pytorch_model") and item.suffix == ".bin"))
    }
    require(bool(weights), "Checkpoint manifest lists no supported weight files")
    require(all(path.stat().st_size > 0 for path in weights), "Empty checkpoint weight file")
    indices = {item for item in seen
               if item.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}}
    if len(weights) > 1:
        require(bool(indices), "Sharded checkpoint has no recorded weight index")
    for index_path in indices:
        weight_map = read_json(index_path)["weight_map"]
        require(isinstance(weight_map, dict) and bool(weight_map), "Empty checkpoint weight map")
        shards = {(model_dir / filename).resolve() for filename in weight_map.values()}
        require(shards == weights, "Checkpoint weight index/shard inventory mismatch")
    native = native_source_hashes(config["family"], model_dir)
    result = {
        "repository": manifest.get("repository"), "revision": manifest["revision"],
        "manifest_sha256": digest(path),
        "metadata_sha256_verified": checked, "native_source_hashes": native,
        "large_weights": sized,
        "large_weight_audit_scope": (
            "Current file sizes checked; SHA256 values are the pinned download verification "
            "records. This analysis does not rehash large weight files or prove which bytes "
            "were loaded by a past process."
        ),
    }
    cache[key] = result
    return result


def model_max_length(config):
    native = read_json(ROOT / config["model_path"] / "config.json")
    field = "block_size" if config["family"] in {"huginn", "parcae"} else "max_position_embeddings"
    value = native[field]
    require(type(value) is int and value > 0, "Invalid pinned model context window")
    return value


def retokenize(documents, tokenizer, add_bos, max_length):
    """Independent encoding and harness left-context truncation, without runner code."""
    encoded, records = {}, []
    original_max = 0
    for question_index, doc in enumerate(documents):
        context = doc["context"]
        trimmed = context.rstrip()
        whitespace = context[len(trimmed):]
        context_ids = tokenizer.encode(trimmed, add_special_tokens=False).ids
        require(bool(context_ids), "Empty tokenized context")
        for choice_index, choice in enumerate(doc["choices"]):
            whole = tokenizer.encode(
                trimmed + whitespace + " " + choice, add_special_tokens=False,
            ).ids
            require(whole[:len(context_ids)] == context_ids, "Unstable context token boundary")
            targets = whole[len(context_ids):]
            require(bool(targets), "Empty answer continuation")
            original_length = len(whole) - 1 + int(add_bos)
            drop = max(0, original_length - max_length)
            first_prediction_position = len(context_ids) - 1 + int(add_bos) - drop
            require(first_prediction_position >= 0,
                    "Answer would lose a causal prediction position under truncation")
            require(len(targets) <= original_length - drop,
                    "Full answer does not fit within the context window")
            key = (question_index, choice_index)
            encoded[key] = (targets, original_length - drop)
            original_max = max(original_max, original_length)
            if drop:
                records.append({
                    "question_index": question_index, "choice_index": choice_index,
                    "removed_context_tokens": drop,
                })
    return encoded, {
        "model_max_length": max_length, "original_max_input_tokens": original_max,
        "candidates_truncated": len(records),
        "questions_truncated": len({row["question_index"] for row in records}),
        "total_removed_context_tokens": sum(row["removed_context_tokens"] for row in records),
        "records": records,
    }


def audit_data(task, resolved, cache):
    data_path, shots = resolved["data_path"], resolved["shots"]
    key = ("data", task, shots)
    if key in cache:
        return cache[key]
    require(data_path.resolve() == (ROOT / f"data/tasks/{task}/{shots}shot.jsonl").resolve(),
            "Unexpected prepared-data path")
    if task == "arc_challenge":
        from src.data_arc_challenge import audit_prepared
        evidence = audit_prepared(shots, seed=resolved["seed"])
    else:
        from src.data_external import audit_prepared
        evidence = audit_prepared(task, shots=shots, seed=resolved["seed"])
    documents = read_jsonl(data_path)
    require(len(documents) == resolved["task"]["count"], "Wrong full evaluation count")
    require(len({doc["id"] for doc in documents}) == len(documents), "Duplicate document IDs")
    require(all(type(doc["gold"]) is int and 0 <= doc["gold"] < len(doc["choices"])
                and len(doc["choices"]) == len(doc["labels"])
                and len(set(doc["labels"])) == len(doc["labels"])
                and all(isinstance(choice, str) and choice for choice in doc["choices"])
                and len(doc["fewshot_ids"]) == shots
                for doc in documents), "Invalid documents or few-shot count")
    protocol_path = resolved["protocol_path"]
    protocol = read_json(protocol_path)
    output = protocol.get("outputs", {}).get(str(shots), protocol.get("output"))
    require(isinstance(output, dict) and output["sha256"] == digest(data_path),
            "Protocol/prepared-data hash mismatch")
    require(protocol["task"] == task and protocol["split"] == resolved["task"]["split"],
            "Protocol task/evaluation split mismatch")
    require(protocol.get("seed", protocol.get("fewshot_seed")) == resolved["seed"],
            "Protocol seed mismatch")
    require(output["rows"] == len(documents)
            and (ROOT / output["path"]).resolve() == data_path.resolve(),
            "Protocol output identity/count mismatch")
    source_hashes = {}
    for entry in [*protocol.get("raw_files", []), *protocol["sources"]]:
        source_path = ROOT / entry["path"]
        require(source_path.resolve().is_relative_to(ROOT.resolve()),
                "Data provenance path escapes repository")
        require(source_path.is_file() and digest(source_path) == entry["sha256"],
                f"Data/harness provenance hash mismatch: {source_path}")
        source_hashes[entry["path"]] = entry["sha256"]
    result = (documents, {
        "canonical_raw_and_complete_prompts_verified": True,
        "validation": evidence,
        "raw_and_harness_source_sha256": source_hashes,
        "data_path": relative(data_path), "data_sha256": digest(data_path),
        "protocol_path": relative(protocol_path), "protocol_sha256": digest(protocol_path),
    })
    cache[key] = result
    return result


def audit_sources(output, manifest, name, task, resolved):
    required_paths = {
        "scripts/evaluate.py", "src/suite_scoring.py", "src/loopcd.py",
        f"src/adapter_{resolved['model']['family']}.py", "src/run_integrity.py",
        "src/registry.py", "src/evaluation.py", "configs/models.json", "configs/tasks.json",
    }
    hashes = manifest["code_hashes"]
    require(set(hashes) == required_paths, "Incomplete or unexpected source snapshot inventory")
    require(len({Path(path).name for path in hashes}) == len(hashes),
            "Source snapshot basename collision")
    verified = {}
    for path, expected in hashes.items():
        snapshot = output / "source_snapshot" / Path(path).name
        require(snapshot.is_file() and digest(snapshot) == expected,
                f"Missing/changed source snapshot: {path}")
        verified[path] = {"sha256": expected, "verified_file": relative(snapshot)}
    frozen_models = read_json(output / "source_snapshot/models.json")
    frozen_tasks = read_json(output / "source_snapshot/tasks.json")
    require(frozen_models["seed"] == resolved["seed"]
            and frozen_models["models"][name] == resolved["model"],
            "Snapshot model registry differs from the run")
    require(frozen_tasks["tasks"][task] == resolved["task"]
            and frozen_tasks["context_policy"] == manifest["context_policy"],
            "Snapshot task registry differs from the run")
    return verified


def audit_scores(documents, tokenized, output, summary):
    """Recompute every likelihood, prediction and summary from per-token scores."""
    candidates, token_count, maximum = {}, 0, 0
    with (output / "candidate_scores.jsonl").open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = row["question_index"], row["choice_index"]
            require(all(type(value) is int for value in key), "Candidate indices must be integers")
            require(key in tokenized and key not in candidates,
                    f"Invalid/duplicate candidate {key} on line {line_number}")
            targets, input_length = tokenized[key]
            require(isinstance(row["target_ids"], list)
                    and all(type(token) is int for token in row["target_ids"])
                    and row["target_ids"] == targets, f"Target token mismatch at {key}")
            require(type(row["input_length"]) is int and row["input_length"] == input_length,
                    f"Input/truncation length mismatch at {key}")
            require(set(row["scores"]) == set(MODES)
                    and set(row["token_logprobs"]) == set(MODES), f"Policy mismatch at {key}")
            recomputed = {}
            for mode in MODES:
                values = row["token_logprobs"][mode]
                require(isinstance(values, list) and len(values) == len(targets),
                        f"Missing answer token scores at {key}/{mode}")
                require(all(type(value) in (int, float) and math.isfinite(value)
                            and value <= 1e-6 for value in values),
                        f"Invalid log probability at {key}/{mode}")
                recomputed[mode] = sum(values)
                close(row["scores"][mode], recomputed[mode], f"Token sum {key}/{mode}")
            candidates[key] = recomputed
            token_count += len(targets)
            maximum = max(maximum, input_length)
    require(set(candidates) == set(tokenized), "Missing candidates")
    saved_predictions = read_jsonl(output / "predictions.jsonl")
    require(len(saved_predictions) == len(documents), "Missing/extra predictions")
    correct = {metric: {mode: 0 for mode in MODES} for metric in METRICS}
    reconstructed = []
    for index, (doc, saved) in enumerate(zip(documents, saved_predictions)):
        for field in ("id", "question", "choices", "labels", "gold"):
            require(saved[field] == doc[field], f"Prediction/data mismatch {index}/{field}")
        scores = {
            mode: [candidates[index, choice][mode] for choice in range(len(doc["choices"]))]
            for mode in MODES
        }
        normalized = {
            mode: [value / len(choice) for value, choice in zip(values, doc["choices"])]
            for mode, values in scores.items()
        }
        for field, expected in (("scores", scores), ("scores_per_character", normalized)):
            require(set(saved[field]) == set(MODES), f"Prediction policy mismatch {index}/{field}")
            for mode, values in expected.items():
                require(len(saved[field][mode]) == len(values), "Prediction option count mismatch")
                for choice_index, value in enumerate(values):
                    close(saved[field][mode][choice_index], value,
                          f"Prediction score {index}/{field}/{mode}/{choice_index}")
        require(set(saved["predictions"]) == set(METRICS), "Prediction metric mismatch")
        verdicts = {}
        for metric, by_mode in (("acc", scores), ("acc_norm", normalized)):
            require(set(saved["predictions"][metric]) == set(MODES), "Prediction modes mismatch")
            verdicts[metric] = {}
            for mode, values in by_mode.items():
                chosen = max(range(len(values)), key=values.__getitem__)
                require(type(saved["predictions"][metric][mode]) is int
                        and saved["predictions"][metric][mode] == chosen,
                        f"Prediction argmax mismatch {index}/{metric}/{mode}")
                verdicts[metric][mode] = chosen
                correct[metric][mode] += int(chosen == doc["gold"])
        reconstructed.append({"id": doc["id"], "gold": doc["gold"], "predictions": verdicts})
    require(type(summary["n"]) is int and summary["n"] == len(documents), "Summary count mismatch")
    require(summary["correct"] == correct, "Summary correct counts differ from token reconstruction")
    accuracy = {
        metric: {mode: 100 * count / len(documents) for mode, count in counts.items()}
        for metric, counts in correct.items()
    }
    require(set(summary["accuracy_percent"]) == set(METRICS), "Summary metrics mismatch")
    for metric in METRICS:
        require(set(summary["accuracy_percent"][metric]) == set(MODES), "Summary modes mismatch")
        for mode in MODES:
            close(summary["accuracy_percent"][metric][mode], accuracy[metric][mode],
                  f"Summary accuracy {metric}/{mode}")
    return {
        "correct": correct, "accuracy_percent": accuracy,
        "paired": {metric: {mode: legacy.paired_stats(reconstructed, mode, metric) for mode in GUIDED}
                   for metric in METRICS},
        "candidate_count": len(candidates), "answer_token_count_per_policy": token_count,
        "max_input_tokens": maximum,
    }


def precision_decision(task, name, config):
    """Allow only documented, pre-scoring upgrades from high to highest."""
    path = ROOT / "results/multitask/precision_decisions.json"
    if not path.exists():
        return config["matmul_precision"], None
    entry = read_json(path)["overrides"].get(f"{task}/{name}")
    if entry is None:
        return config["matmul_precision"], None
    require(entry["default"] == config["matmul_precision"] == "high"
            and entry["selected"] == "highest", "Invalid precision override")
    require(type(entry["formal_candidates_before_retry"]) is int
            and entry["formal_candidates_before_retry"] == 0
            and entry["validation_tolerances_unchanged"] is True,
            "Precision must be chosen before formal scoring without relaxed validation")
    log = (ROOT / entry["failure_log"]).resolve()
    require(log.is_relative_to((ROOT / "results/multitask/logs").resolve())
            and digest(log) == entry["failure_log_sha256"]
            and "AssertionError" in log.read_text(), "Missing precision failure evidence")
    return entry["selected"], {"decision": entry, "ledger_sha256": digest(path)}


def audit_model(task, name, resolved, tasks_registry, paper, cache):
    config, task_config, output = resolved["model"], resolved["task"], resolved["output"]
    documents, data_evidence = audit_data(task, resolved, cache)
    manifest, validation, summary = (
        read_json(output / filename)
        for filename in ("run_manifest.json", "validation.json", "summary.json")
    )
    checkpoint = audit_checkpoint(config, cache)
    audit_runtime_metadata(manifest)
    expected_precision, precision_evidence = precision_decision(task, name, config)
    expected_candidates = sum(len(doc["choices"]) for doc in documents)
    expected_fields = {
        "name": name, "configuration": config, "task": task, "task_configuration": task_config,
        "seed": resolved["seed"], "shots": resolved["shots"],
        "primary_metric": task_config["primary_metric"],
        "documents": len(documents), "candidates": expected_candidates,
        "data_sha256": data_evidence["data_sha256"],
        "protocol_sha256": data_evidence["protocol_sha256"],
        "model_manifest_sha256": checkpoint["manifest_sha256"],
        "native_source_hashes": checkpoint["native_source_hashes"],
        "context_policy": tasks_registry["context_policy"], "chat_template": False,
        "add_bos": config["family"] == "huginn",
        "float32_matmul_precision": expected_precision, "dtype": "float32",
    }
    for field, value in expected_fields.items():
        require(manifest[field] == value, f"{task}/{name}: manifest mismatch: {field}")
    require(digest(output / "data_protocol.json") == manifest["protocol_sha256"],
            "Run protocol snapshot hash mismatch")
    for field, expected in {
        "task": task, "model": name, "primary_metric": task_config["primary_metric"],
        "shots": resolved["shots"], "split": task_config["split"],
    }.items():
        require(summary[field] == expected, f"Summary configuration mismatch: {field}")
    source_evidence = audit_sources(output, manifest, name, task, resolved)
    native_evidence = legacy.validate_native_checks(validation, False, config)
    tokenizer_path = ROOT / config["model_path"] / "tokenizer.json"
    token_key = ("tokens", str(resolved["data_path"]), digest(tokenizer_path),
                 manifest["add_bos"], model_max_length(config))
    if token_key not in cache:
        cache[token_key] = retokenize(
            documents, Tokenizer.from_file(str(tokenizer_path)),
            manifest["add_bos"], model_max_length(config),
        )
    tokenized, truncation = cache[token_key]
    require(manifest["truncation_details"] == truncation, "Context truncation inventory mismatch")
    require(type(manifest["truncation"]) is bool
            and manifest["truncation"] == bool(truncation["candidates_truncated"]),
            "Context truncation flag mismatch")
    scores = audit_scores(documents, tokenized, output, summary)
    require(scores["max_input_tokens"] == manifest["max_input_tokens"], "Maximum input length mismatch")
    reference = dict(zip(MODES[:3], paper[task][name]))
    return {
        "display": config["display"], "configuration": config, "shots": resolved["shots"],
        "results_path": relative(output), "n": len(documents),
        "correct": scores["correct"], "accuracy_percent": scores["accuracy_percent"],
        "paired": scores["paired"], "paper_table2": reference,
        "actual_minus_paper_pp": {
            mode: scores["accuracy_percent"][task_config["primary_metric"]][mode] - value
            for mode, value in reference.items()
        },
        "checkpoint": checkpoint,
        "precision_decision": precision_evidence,
        "audit": {
            "passed": True, "unique_documents": len(documents),
            "unique_candidates": scores["candidate_count"],
            "answer_token_count_per_policy": scores["answer_token_count_per_policy"],
            "data": data_evidence, "independent_retokenization": True,
            "all_token_sums_predictions_and_summary_match": True,
            "truncation_details": truncation, "verified_code": source_evidence,
            "validation": native_evidence,
            "result_file_sha256": {filename: digest(output / filename) for filename in REQUIRED},
        },
        "run_manifest": manifest, "native_validation": validation,
        "runtime": {key: value for key, value in summary.items()
                    if key.endswith("_seconds") or key == "peak_gpu_allocated_gib"},
    }


def audit_runtime_metadata(manifest):
    """Require historical runtime provenance without imposing today's versions."""
    require(isinstance(manifest, dict), "Run manifest must be an object")
    missing = MULTITASK_MANIFEST_KEYS - manifest.keys()
    require(not missing, f"Run manifest lacks required fields: {sorted(missing)}")
    for field in ("torch", "transformers"):
        require(isinstance(manifest.get(field), str) and bool(manifest[field]),
                f"Missing/invalid historical package version: {field}")
    for field in ("batch_size", "token_budget", "documents", "candidates", "max_input_tokens"):
        require(type(manifest.get(field)) is int and manifest[field] > 0,
                f"Missing/invalid runtime setting: {field}")
    for field in ("seed", "shots"):
        require(type(manifest[field]) is int and manifest[field] >= 0,
                f"Missing/invalid runtime setting: {field}")
    for field in ("add_bos", "chat_template", "truncation"):
        require(type(manifest[field]) is bool, f"Invalid boolean runtime setting: {field}")
    for field in ("adapter_metadata", "configuration", "task_configuration", "truncation_details"):
        require(isinstance(manifest[field], dict), f"Missing/invalid {field}")
    for field in ("code_hashes", "native_source_hashes"):
        require(isinstance(manifest[field], dict) and bool(manifest[field]),
                f"Missing/invalid {field}")
    json.dumps(manifest["adapter_metadata"], allow_nan=False)


def audit_task(task, partial=False, cache=None):
    models_registry, tasks_registry = load_registry()
    require(set(models_registry["models"]) == MODEL_NAMES, "Expected all seven model configurations")
    require(task in TASK_NAMES, "Unsupported task")
    task_config = tasks_registry["tasks"][task]
    require(task_config["primary_metric"] in METRICS, "Unknown primary metric")
    cache = {} if cache is None else cache
    paper_path = ROOT / "configs/paper_mc_reference.json"
    paper = read_json(paper_path)
    old = read_json(ROOT / "configs/arc_all_models.json") if task == "arc_easy" else None
    models, incomplete = {}, {}
    for name, config in models_registry["models"].items():
        resolved = resolve_run(task, name)
        if old is not None:
            old_config = old["models"][name]
            require({k: v for k, v in old_config.items() if k not in {"shots", "paper", "reuse"}}
                    == {k: v for k, v in config.items() if k != "matmul_precision"},
                    "Historical ARC-E model configuration changed")
            require(old_config["shots"] == resolved["shots"]
                    and old["dataset_size"] == task_config["count"]
                    and old["seed"] == resolved["seed"]
                    and task_config["primary_metric"] == "acc_norm",
                    "Historical ARC-E task protocol changed")
            output = ROOT / old_config.get("reuse", f"results/arc_all_models/{name}")
            required = legacy.REQUIRED
        else:
            output, required = resolved["output"], REQUIRED
        if output.exists():
            list(output.iterdir())  # Refresh NFS entries before testing completion markers.
        missing = [filename for filename in required if not (output / filename).is_file()]
        if missing:
            incomplete[name] = missing
            continue
        print(f"Auditing {task}/{name}: all candidates and token probabilities", file=sys.stderr, flush=True)
        if old is not None:
            result = legacy.audit_model(name, old_config, cache)
            result["paper_table2"] = result["paper_table2_arc_easy"]
            result["shots"] = old_config["shots"]
            result["reused_historical_result"] = True
            result["checkpoint"] = audit_checkpoint(config, cache)
            models[name] = result
        else:
            models[name] = audit_model(task, name, resolved, tasks_registry, paper, cache)
    if not incomplete:
        legacy.holm_adjust(models)
    result = {
        "task": task, "display": task_config["display"], "complete": not incomplete,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_configuration": task_config, "primary_metric": task_config["primary_metric"],
        "expected_documents_per_configuration": task_config["count"],
        "models": models, "incomplete": incomplete,
        "source_hashes": {
            relative(path): digest(path)
            for path in (Path(__file__), ROOT / "scripts/analyze_arc_suite.py",
                         ROOT / "configs/models.json", ROOT / "configs/tasks.json", paper_path)
        },
        "statistical_protocol": {
            "bootstrap": "20000 paired question resamples; seed 1234; pointwise percentile 95% CI",
            "mcnemar": "Exact two-sided conditional binomial test",
            "holm": "14 comparisons (7 models × 2 guidance policies), separately per task and metric; only when task complete",
            "cross_task_correction": False,
            "uncertainty_excludes": "Between-seed/model-run variation and protocol differences",
        },
        "metric_note": (
            "Both acc and acc_norm are reconstructed. acc_norm uses Python len(original choice), "
            "excluding the added delimiter. Primary metric is a frozen reproduction assumption; "
            "Table 2 and the pinned task YAMLs do not specify a primary for every task."
        ),
    }
    if not partial:
        require(not incomplete, f"{task}: final report requires all seven models; missing {incomplete}")
    return result


def report_task(result):
    require(result["complete"] and set(result["models"]) == MODEL_NAMES,
            "A final task report requires seven audited configurations")
    primary, models = result["primary_metric"], result["models"]
    lines = [
        f"# {result['display']}: LoopCD-Logits across all seven model configurations", "",
        f"The full {result['task_configuration']['split']} split contains "
        f"{result['expected_documents_per_configuration']} questions per configuration.", "",
        f"The primary metric is `{primary}`; both metrics are reported. "
        "`acc` uses the sum of conditional log likelihoods over the complete answer; "
        "`acc_norm` divides this sum by the Unicode character count of the original answer, "
        "excluding the added leading space.", "",
        "## Reproduction / paper (%)", "",
        "| Configuration | Shots | Baseline | Fixed | Adaptive | Fixed change (pp) | Adaptive change (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models.values():
        values = [
            f"{model['accuracy_percent'][primary][mode]:.2f} / {model['paper_table2'][mode]:.2f}"
            for mode in MODES[:3]
        ]
        changes = [f"{model['paired'][primary][mode]['delta_percentage_points']:+.2f}" for mode in GUIDED]
        lines.append("| " + " | ".join([model["display"], str(model["shots"]), *values, *changes]) + " |")
    lines.extend([
        "", f"## Paired question-level statistics ({primary})", "",
        "| Configuration | Policy | Corrected | Regressed | Answers changed | Net correct | Paired 95% CI (pp) | McNemar p | Holm p |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ])
    for model in models.values():
        for mode in GUIDED:
            values = model["paired"][primary][mode]
            low, high = values["paired_bootstrap_95ci_pp"]
            lines.append(
                f"| {model['display']} | {MODE_NAMES[mode]} | {values['corrected']} | "
                f"{values['regressed']} | {values['changed_answers']} | {values['net_correct']:+d} | "
                f"[{low:+.2f}, {high:+.2f}] | {values['mcnemar_exact_two_sided_p']:.4g} | "
                f"{values['mcnemar_holm_adjusted_p']:.4g} |"
            )
    significant = [
        f"{model['display']} {MODE_NAMES[mode]} ("
        f"{model['paired'][primary][mode]['delta_percentage_points']:+.2f} pp)"
        for model in models.values() for mode in GUIDED
        if model["paired"][primary][mode]["mcnemar_holm_adjusted_p"] < .05
    ]
    lines.extend([
        "",
        ("Primary-metric changes with Holm p<0.05 for this task: " + "; ".join(significant) + ".")
        if significant else "No primary-metric comparison for this task has Holm p<0.05.",
        "",
        "Pointwise 95% confidence intervals use 20,000 paired question-level bootstrap resamples "
        "(seed=1234). Exact two-sided McNemar tests receive Holm correction separately for each "
        "task and metric, across 14 comparisons per family.", "",
        "## Both metrics and the reference state", "",
        "| Configuration | Metric | Baseline | Fixed | Adaptive | First full-loop reference |",
        "|---|---|---:|---:|---:|---:|",
    ])
    sensitive = []
    for model in models.values():
        for metric in METRICS:
            cells = [f"{model['correct'][metric][mode]} ({model['accuracy_percent'][metric][mode]:.2f}%)"
                     for mode in MODES]
            lines.append("| " + " | ".join([model["display"], metric, *cells]) + " |")
        for mode in GUIDED:
            if (model["paired"]["acc"][mode]["delta_percentage_points"]
                    * model["paired"]["acc_norm"][mode]["delta_percentage_points"] < 0):
                sensitive.append(f"{model['display']} {MODE_NAMES[mode]}")
    if sensitive:
        lines.extend(["", "Comparisons with opposite directions of change between the two metrics: "
                      + "; ".join(sensitive) + "."])
    if result["task"] == "sciq":
        distances = {
            metric: sum(abs(model["accuracy_percent"][metric][mode] - model["paper_table2"][mode])
                        for model in models.values() for mode in MODES[:3]) / (len(models) * 3)
            for metric in METRICS
        }
        ouro = models["ouro_14"]
        ouro_raw = " / ".join(f"{ouro['accuracy_percent']['acc'][mode]:.2f}" for mode in MODES[:3])
        ouro_norm = " / ".join(f"{ouro['accuracy_percent']['acc_norm'][mode]:.2f}" for mode in MODES[:3])
        ouro_paper = " / ".join(f"{ouro['paper_table2'][mode]:.2f}" for mode in MODES[:3])
        ouro_distances = {
            metric: sum(abs(ouro["accuracy_percent"][metric][mode] - ouro["paper_table2"][mode])
                        for mode in MODES[:3]) / 3
            for metric in METRICS
        }
        if ouro_distances["acc_norm"] < ouro_distances["acc"]:
            proximity = "For this configuration, character-normalized scores are closer to the paper"
        elif ouro_distances["acc"] < ouro_distances["acc_norm"]:
            proximity = "For this configuration, unnormalized scores are closer to the paper"
        else:
            proximity = "For this configuration, both metrics have the same mean absolute difference from the paper"
        lines.extend([
            "", "### SciQ metric comparison", "",
            f"Ouro-1.4B baseline / fixed / adaptive: `acc` is {ouro_raw}%, "
            f"`acc_norm` is {ouro_norm}%, and the paper reports {ouro_paper}%. "
            f"{proximity}.",
            f"Across all 21 published scores for the seven configurations, the mean absolute "
            f"difference is {distances['acc']:.3f} pp for `acc` and "
            f"{distances['acc_norm']:.3f} pp for `acc_norm`. "
            "SciQ uses `acc` as its primary metric and reports `acc_norm` as a supplementary metric.",
        ])
    lines.extend([
        "", "## Evaluation protocol", "",
        "- Fixed guidance strength is 0.5. The adaptive cap is 0.5 for Huginn and 1.0 for the other "
        "models. Adaptive strength uses the top-two probability gap over the full vocabulary. The reference is the "
        "first complete update followed by the full coda/head; all policies share one loop trajectory.",
        "- Shot counts follow paper Table A2, with seed 1234. Nonzero-shot examples are sampled "
        "from the training set as questions are processed. SciQ prompts include support, "
        "and PIQA uses the labeled validation split.",
        "- Huginn receives one BOS token through its official interface; the other models receive "
        "no extra BOS token. Prompts use plain task text. Huginn/Parcae random initial states are "
        "fixed by question ID, with positional noise shared across candidates.",
        "- For the new tasks, context is truncated from the left only when the model window is "
        "insufficient, preserving all answer tokens and prediction positions. The table "
        "below reports truncation counts for each configuration.",
        "- Checkpoint manifests, small model files, tokenizers, and native source files are "
        "verified with SHA256. Large weights are checked by file size in this audit, with hashes "
        "referenced from download-time verification.",
        "- Evaluations use full recurrent depth. Recorded execution time covers all four scoring "
        "policies sharing one loop trajectory.",
        "", "| Configuration | dtype / matmul | Truncated questions / candidates | Maximum input tokens | Maximum batch/single token logp difference |",
        "|---|---|---:|---:|---:|",
    ])
    for model in models.values():
        manifest = model["run_manifest"]
        trunc = manifest.get("truncation_details", {})
        validation = model["audit"]["validation"]
        lines.append(
            f"| {model['display']} | {manifest.get('dtype')} / "
            f"{manifest.get('float32_matmul_precision')} | "
            f"{trunc.get('questions_truncated', 0)} / {trunc.get('candidates_truncated', 0)} | "
            f"{manifest['max_input_tokens']} | {validation['batch_single_max_error']:.3g} |"
        )
    if result["task"] == "arc_easy":
        lines.extend(["", "ARC-Easy analysis reads the original candidate scores. "
                      "The full historical protocol is documented in the "
                      "[ARC-Easy report](../../arc_all_models/REPORT.md)."])
    lines.extend([
        "", "See comparison.json for source paths, candidate-level truncation inventories, paired "
        "statistics for both metrics, and full audit evidence for each configuration. "
        f"Regenerate with `venv/bin/python scripts/analyze.py --task {result['task']}`.", "",
        "Sources: [paper Tables 2 / A2 / A4](https://arxiv.org/html/2610.02185v1) and the "
        "[pinned lm-eval implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/"
        "d6de81643928d653435c431bae19945d41d32520). Paper reference values and supporting evidence "
        "for the new tasks are stored in configs/paper_mc_reference.json; ARC-Easy reference "
        "values remain in configs/arc_all_models.json.", "",
    ])
    return "\n".join(lines)


def report_all(results):
    require(bool(results) and all(result["complete"] and set(result["models"]) == MODEL_NAMES
                                 for result in results.values()),
            "A final combined report requires complete audited tasks")
    lines = [
        "# Multitask LoopCD-Logits reproduction", "",
        "Each task evaluates seven model configurations with baseline, fixed, and adaptive guidance. "
        "Reports include both primary and supplementary metrics.", "",
        "| Task | Split / questions | Primary metric | Configurations | Detailed report |",
        "|---|---|---|---:|---|",
    ]
    for task, result in results.items():
        config = result["task_configuration"]
        lines.append(
            f"| {result['display']} | {config['split']} / {config['count']} | "
            f"{result['primary_metric']} | {len(result['models'])} | [{task}]({task}/REPORT.md) |"
        )
    lines.extend([
        "", "Accuracy and paired statistics are reported separately for each task and metric. "
        "Holm correction applies to 14 comparisons within each task/metric family.",
        "", "This combined report includes only the fully audited tasks selected by `--task`; "
        "`--task all` includes all four tasks. Original ARC-Easy scores are preserved. See "
        "comparison.csv for a machine-readable table by task, model, and metric. Paper reference "
        "values appear only in primary-metric rows; the paper columns in supplementary-metric "
        "rows are left empty.", "",
    ])
    return "\n".join(lines)


def comparison_csv(results):
    """One row per task/model/metric; paper cells only for the selected primary."""
    require(bool(results) and all(result["complete"] and set(result["models"]) == MODEL_NAMES
                                 for result in results.values()),
            "A final comparison CSV requires complete audited tasks")
    stream = io.StringIO(newline="")
    fields = [
        "task", "model", "display", "split", "n", "shots", "metric", "is_primary",
        "baseline_percent", "fixed_percent", "adaptive_percent", "reference_percent",
        "fixed_delta_pp", "adaptive_delta_pp", "fixed_holm_p", "adaptive_holm_p",
        "paper_baseline_percent", "paper_fixed_percent", "paper_adaptive_percent",
        "candidates", "truncated_questions", "truncated_candidates", "results_path",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for task, result in results.items():
        for name, model in result["models"].items():
            for metric in METRICS:
                accuracy, paired = model["accuracy_percent"][metric], model["paired"][metric]
                primary = metric == result["primary_metric"]
                paper = model["paper_table2"] if primary else {}
                truncation = model["run_manifest"].get("truncation_details", {})
                writer.writerow({
                    "task": task, "model": name, "display": model["display"],
                    "split": result["task_configuration"]["split"], "n": model["n"],
                    "shots": model["shots"], "metric": metric, "is_primary": primary,
                    "baseline_percent": accuracy["baseline"],
                    "fixed_percent": accuracy["loopcd_fixed"],
                    "adaptive_percent": accuracy["loopcd_adaptive"],
                    "reference_percent": accuracy["reference"],
                    "fixed_delta_pp": paired["loopcd_fixed"]["delta_percentage_points"],
                    "adaptive_delta_pp": paired["loopcd_adaptive"]["delta_percentage_points"],
                    "fixed_holm_p": paired["loopcd_fixed"]["mcnemar_holm_adjusted_p"],
                    "adaptive_holm_p": paired["loopcd_adaptive"]["mcnemar_holm_adjusted_p"],
                    "paper_baseline_percent": paper.get("baseline", ""),
                    "paper_fixed_percent": paper.get("loopcd_fixed", ""),
                    "paper_adaptive_percent": paper.get("loopcd_adaptive", ""),
                    "candidates": model["audit"]["unique_candidates"],
                    "truncated_questions": truncation.get("questions_truncated", 0),
                    "truncated_candidates": truncation.get("candidates_truncated", 0),
                    "results_path": model["results_path"],
                })
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", *TASK_NAMES), default="all")
    parser.add_argument("--partial", action="store_true",
                        help="Read-only JSON audit of completed configurations; no report writes")
    args = parser.parse_args()
    selected = TASK_NAMES if args.task == "all" else (args.task,)
    cache = {}
    results = {task: audit_task(task, partial=args.partial, cache=cache) for task in selected}
    combined = {
        "complete": all(result["complete"] for result in results.values()),
        "selected_tasks": list(selected), "tasks": results,
    }
    if args.partial:
        print(json.dumps(combined, indent=2, ensure_ascii=False, allow_nan=False))
        return
    # Audit and render every selected task before writing any final artifact.
    artifacts = []
    base = ROOT / "results/multitask"
    for task, result in results.items():
        artifacts.extend([
            (base / task / "comparison.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
            (base / task / "REPORT.md", report_task(result)),
        ])
    artifacts.extend([
        (base / "comparison.json", json.dumps(combined, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
        (base / "comparison.csv", comparison_csv(results)),
        (base / "REPORT.md", report_all(results)),
    ])
    for path, text in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
    print(json.dumps({"complete": True, "selected_tasks": list(selected),
                      "report": str(base / "REPORT.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
