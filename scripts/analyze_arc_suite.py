#!/usr/bin/env python3
"""Audit saved ARC-Easy token scores and report the seven frozen configurations.

No models are loaded. Local tokenizer.json files independently reproduce candidate
token boundaries. --partial audits only configurations with a summary completion
marker and prints JSON; it never creates a final comparison or report.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "loopcd_fixed", "loopcd_adaptive", "reference")
GUIDED = ("loopcd_fixed", "loopcd_adaptive")
METRICS = ("acc", "acc_norm")
MODE_NAMES = {"loopcd_fixed": "Fixed", "loopcd_adaptive": "Adaptive"}
N_DOCUMENTS, N_CANDIDATES = 2376, 9501
REQUIRED = ("summary.json", "predictions.jsonl", "candidate_scores.jsonl",
            "run_manifest.json", "validation.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, message, tolerance=1e-10):
    require(isinstance(actual, (int, float)) and math.isfinite(actual),
            message + ": nonfinite/non-numeric value")
    require(abs(actual - expected) <= tolerance,
            f"{message}: {actual!r} != {expected!r}")


def paired_stats(rows, mode, metric):
    base = np.array([r["predictions"][metric]["baseline"] == r["gold"] for r in rows])
    guided = np.array([r["predictions"][metric][mode] == r["gold"] for r in rows])
    improved, regressed = int((~base & guided).sum()), int((base & ~guided).sum())
    discordant = improved + regressed
    # Conditional on discordance, the exact null distribution is Binomial(d, .5).
    tail_count = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
    p = min(1.0, 2 * tail_count / 2**discordant)
    rng = np.random.default_rng(1234)
    counts = rng.multinomial(
        len(rows), [improved / len(rows), regressed / len(rows),
                    1 - discordant / len(rows)], size=20000,
    )
    deltas = 100 * (counts[:, 0] - counts[:, 1]) / len(rows)
    changed = sum(r["predictions"][metric]["baseline"] != r["predictions"][metric][mode]
                  for r in rows)
    return {
        "corrected": improved, "regressed": regressed, "net_correct": improved - regressed,
        "changed_answers": changed,
        "delta_percentage_points": 100 * (improved - regressed) / len(rows),
        "paired_bootstrap_95ci_pp": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "mcnemar_exact_two_sided_p": p,
    }


def holm_adjust(models):
    """Correct all 14 comparisons per metric; no model independence is assumed."""
    for metric in METRICS:
        tests = [model["paired"][metric][mode] for model in models.values() for mode in GUIDED]
        previous = 0.0
        for rank, test in enumerate(sorted(tests, key=lambda x: x["mcnemar_exact_two_sided_p"])):
            previous = max(previous, min(1.0, (len(tests) - rank) * test["mcnemar_exact_two_sided_p"]))
            test["mcnemar_holm_adjusted_p"] = previous


def validate_native_checks(validation, legacy, config):
    identity = "zero_guidance_matches_native_exactly" if legacy else "zero_guidance_matches_baseline_exactly"
    require(validation.get(identity) is True, f"Missing/failed {identity}")
    errors_key = "batch_single_errors_by_policy" if legacy else "batch_single_token_logprob_errors"
    tolerance_key = "batch_single_absolute_tolerance" if legacy else "batch_single_tolerance"
    errors, tolerance = validation[errors_key], validation[tolerance_key]
    require(set(errors) == set(MODES), "Validation must check all four policies")
    require(0 < tolerance <= 0.001, "Unexpectedly loose batch consistency threshold")
    require(all(math.isfinite(x) and 0 <= x <= tolerance for x in errors.values()),
            "Batch/single consistency failed")
    if not legacy:
        require(0 <= validation["masked_padding_token_change_max_abs_error"] <= 1e-5,
                "Masked padding invariance failed")
    difference = validation.get("early_final_mean_abs_difference",
                                validation.get("early_final_mean_abs_logit_difference"))
    require(difference is not None and math.isfinite(difference) and difference > 0,
            "Missing distinct early/final readout check")
    native_errors = {}
    if legacy:
        native_errors["final_logits"] = validation["native_final_logits_max_abs_error"]
        require(validation["recurrent_passes"] == config["loops"], "Wrong native recurrent depth")
    elif config["family"] == "ouro":
        native_errors = validation["native_full_head_max_abs_error"]
        require(set(native_errors) == {"baseline", "reference"}, "Missing native Ouro readout")
        require(validation["complete_recurrent_passes"] == config["loops"] and validation["reference_pass"] == 1,
                "Wrong Ouro native recurrent depth")
        require(validation["zero_guidance_identity"] is True, "Ouro native identity failed")
    elif config["family"] == "huginn":
        for field in ("native_final_hidden_max_abs_error", "native_same_shape_logits_max_abs_error"):
            close(validation[field], 0, f"Huginn exact native check {field}", tolerance=0)
            native_errors[field] = validation[field]
        native_errors["full_logits"] = validation["native_full_logits_max_abs_error"]
        require(validation["zero_guidance_matches_native_same_shape_exactly"] is True
                and validation["same_initial_states_supplied_to_native"] is True,
                "Missing native Huginn identity/initial-state check")
        require(validation["loops"] == config["loops"] and validation["reference_recurrent_update"] == 1
                and validation["reference_is_initial_noise"] is False, "Wrong Huginn readout semantics")
    elif config["family"] == "parcae":
        native_errors = {field: validation[field] for field in
                         ("native_final_logits_max_abs_error", "native_first_loop_logits_max_abs_error")}
        require(validation["zero_guidance_matches_native"] is True
                and validation["latent_prefix_invariant_to_length_and_batch"] is True
                and validation["same_keys_repeat_exactly"] is True, "Parcae native/latent checks failed")
        require(validation["recurrent_updates_in_forward"] == config["loops"]
                and validation["reference_after_complete_updates"] == 1, "Wrong Parcae recurrent depth")
    elif config["family"] == "looped_qwen":
        require(validation["native_damped_wrapper_replay_passed"] is True,
                "Looped Qwen wrapper replay failed")
        require(validation["k1_dense_logits_match_exactly"] is True,
                "Looped Qwen K=1 dense identity failed")
        close(validation["k1_native_dense_all_hidden_max_abs_error"], 0, "Qwen K=1 hidden identity", tolerance=0)
        native_errors = validation["native_damped_wrapper_replay_max_logit_errors"]
        require(set(native_errors) == {"baseline", "reference"}, "Missing Qwen native readout")
        require(validation["loops"] == 8 and validation["reference_pass"] == 1
                and validation["window_zero_based"] == [15, 16, 17, 18]
                and validation["anchor_weight"] == 0
                and validation["damping_step_size"] == 1 / 8, "Wrong Looped Qwen recurrence")
    else:
        raise ValueError("Unrecognized model family")
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= .001
                for x in native_errors.values()), "Native/independent replay exceeds conservative 0.001 absolute audit bound")
    return {"zero_guidance_identity": True, "batch_single_max_error": max(errors.values()),
            "batch_single_tolerance": tolerance, "early_final_mean_abs_difference": difference,
            "native_readout_max_abs_error": max(native_errors.values())}


def audit_prepared_data(documents, shots):
    """Check the canonical held-out split and independently rebuild every prompt."""
    raw_dir = ROOT / "data/raw/ARC-Easy"
    expected_hashes = {
        "Train": "c62e5482d8c01f2d5a7f24239ac9db3b12f9e47267c4e3f088c6ffe41d5545ef",
        "Test": "ce32a0774b4beb1977d2c6b16dda5dd2b24fdc9c7f506882c452678676bd09be",
    }
    splits = {}
    for split, expected in expected_hashes.items():
        path = raw_dir / f"ARC-Easy-{split}.jsonl"
        require(digest(path) == expected, f"Canonical {split} source checksum mismatch")
        splits[split] = read_jsonl(path)
    require(len(splits["Train"]) == 2251 and len(splits["Test"]) == N_DOCUMENTS,
            "Canonical split counts mismatch")
    rng = random.Random(1234)
    for index, (doc, raw) in enumerate(zip(documents, splits["Test"])):
        question = raw["question"]
        labels = [choice["label"] for choice in question["choices"]]
        expected = {"id": raw["id"], "question": question["stem"],
                    "choices": [choice["text"] for choice in question["choices"]],
                    "labels": labels, "gold": labels.index(raw["answerKey"])}
        for field, value in expected.items():
            require(doc[field] == value, f"Canonical test mismatch {index}/{field}")
        examples = rng.sample(splits["Train"], shots)
        require(doc["fewshot_ids"] == [example["id"] for example in examples],
                f"Training example sampling mismatch at question {index}")
        prefix = ""
        for example in examples:
            answer = next(choice["text"] for choice in example["question"]["choices"]
                          if choice["label"] == example["answerKey"])
            prefix += f"Question: {example['question']['stem']}\nAnswer: {answer}\n\n"
        expected_context = prefix + f"Question: {question['stem']}\nAnswer:"
        require(doc["context"] == expected_context, f"Few-shot context mismatch at question {index}")


def validate_manifest(name, config, manifest, data_path, output):
    require(manifest["documents"] == N_DOCUMENTS and manifest["candidates"] == N_CANDIDATES,
            f"{name}: manifest is not a full test run")
    require(manifest["data_sha256"] == digest(data_path), f"{name}: data checksum changed")
    require(manifest["model_manifest_sha256"] == digest(ROOT / config["model_manifest"]),
            f"{name}: checkpoint manifest checksum changed")
    require(manifest["seed"] == 1234 and manifest["truncation"] is False,
            f"{name}: unexpected seed/truncation")
    legacy = "configuration" not in manifest
    if legacy:
        require(name == "ouro_14", "Only original Ouro-1.4B result may use the legacy schema")
        require(manifest["recurrent_passes"] == config["loops"] and manifest["reference_pass"] == 1,
                "Original Ouro loop protocol mismatch")
        require(manifest["fixed_strength"] == config["fixed"] and manifest["adaptive_cap"] == config["cap"],
                "Original Ouro guidance parameters mismatch")
        code_hashes = {
            "src/loopcd.py": manifest["loopcd_source_sha256"],
            "scripts/evaluate_arc.py": manifest["evaluator_source_sha256"],
        }
        add_bos = manifest["add_bos_token"]
    else:
        require(manifest["name"] == name and manifest["configuration"] == config,
                f"{name}: frozen configuration mismatch")
        require(manifest["chat_template"] is False, f"{name}: chat template unexpectedly applied")
        code_hashes, add_bos = manifest["code_hashes"], manifest["add_bos"]
    require(add_bos == (config["family"] == "huginn"), f"{name}: unexpected BOS policy")
    verified_code = {}
    for path, expected in code_hashes.items():
        snapshot = output / "source_snapshot" / Path(path).name
        source = snapshot if snapshot.exists() else ROOT / path
        require(digest(source) == expected, f"{name}: source checksum mismatch: {source}")
        verified_code[path] = {"sha256": expected, "verified_file": str(source.relative_to(ROOT))}
    return legacy, add_bos, verified_code


def tokenize_candidates(documents, tokenizer_path, add_bos):
    """Re-encode without importing the evaluator or its candidate helper."""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    targets = {}
    for index, doc in enumerate(documents):
        context = doc["context"]
        trimmed = context.rstrip()
        whitespace = context[len(trimmed):]
        context_ids = tokenizer.encode(trimmed, add_special_tokens=False).ids
        require(bool(context_ids), "Empty encoded context")
        for choice_index, choice in enumerate(doc["choices"]):
            whole = tokenizer.encode(trimmed + whitespace + " " + choice,
                                     add_special_tokens=False).ids
            require(whole[:len(context_ids)] == context_ids, "Unstable context token boundary")
            continuation = whole[len(context_ids):]
            require(bool(continuation), "Empty continuation")
            targets[index, choice_index] = (continuation, len(whole) - 1 + int(add_bos))
    return targets


def audit_model(name, config, cache):
    output = ROOT / config.get("reuse", f"results/arc_all_models/{name}")
    data_path = ROOT / f"data/arc_easy_{config['shots']}shot.jsonl"
    if str(data_path) not in cache:
        cache[str(data_path)] = read_jsonl(data_path)
    documents = cache[str(data_path)]
    require(len(documents) == N_DOCUMENTS and len({row["id"] for row in documents}) == N_DOCUMENTS,
            f"{name}: test IDs are missing/duplicated")
    expected_keys = {(index, choice) for index, doc in enumerate(documents)
                     for choice in range(len(doc["choices"]))}
    require(len(expected_keys) == N_CANDIDATES, f"{name}: unexpected number of choices")
    require(Counter(len(d["choices"]) for d in documents) == {3: 7, 4: 2365, 5: 4},
            f"{name}: unexpected choice count distribution")
    require(all(len(d["fewshot_ids"]) == config["shots"] for d in documents),
            f"{name}: few-shot count mismatch")
    data_audit_key = ("canonical_data_verified", str(data_path))
    if data_audit_key not in cache:
        audit_prepared_data(documents, config["shots"])
        cache[data_audit_key] = True
    summary, manifest, validation = [read_json(output / filename)
                                     for filename in ("summary.json", "run_manifest.json", "validation.json")]
    legacy, add_bos, verified_code = validate_manifest(name, config, manifest, data_path, output)
    validation_audit = validate_native_checks(validation, legacy, config)
    tokenizer_path = ROOT / config["model_path"] / "tokenizer.json"
    token_cache_key = (str(tokenizer_path), config["shots"], add_bos)
    if token_cache_key not in cache:
        cache[token_cache_key] = tokenize_candidates(documents, tokenizer_path, add_bos)
    tokenized = cache[token_cache_key]
    candidates = {}
    max_input = total_answer_tokens = 0
    with (output / "candidate_scores.jsonl").open() as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            key = (row["question_index"], row["choice_index"])
            require(all(type(x) is int for x in key), "Candidate indices must be integers")
            require(key in expected_keys and key not in candidates,
                    f"{name}: invalid/duplicate candidate {key}, line {line_no}")
            target_ids, input_length = tokenized[key]
            require(row["target_ids"] == target_ids and row["input_length"] == input_length,
                    f"{name}: saved target token alignment/input length mismatch at {key}")
            require(set(row["scores"]) == set(MODES) and set(row["token_logprobs"]) == set(MODES),
                    f"{name}: missing/extra policy at {key}")
            recomputed = {}
            for mode in MODES:
                values = row["token_logprobs"][mode]
                require(len(values) == len(target_ids), f"{name}: missing token score at {key}/{mode}")
                require(all(isinstance(v, (int, float)) and math.isfinite(v) and v <= 1e-6 for v in values),
                        f"{name}: invalid log probability at {key}/{mode}")
                recomputed[mode] = sum(values)
                close(row["scores"][mode], recomputed[mode], f"{name}: candidate score {key}/{mode}")
            candidates[key] = recomputed
            total_answer_tokens += len(target_ids)
            max_input = max(max_input, input_length)
    require(set(candidates) == expected_keys, f"{name}: missing candidates")
    require(max_input == manifest["max_input_tokens"], f"{name}: max context length mismatch")
    predictions = read_jsonl(output / "predictions.jsonl")
    require(len(predictions) == N_DOCUMENTS and len({row["id"] for row in predictions}) == N_DOCUMENTS,
            f"{name}: missing/duplicate predictions")
    reconstructed = []
    correct = {metric: {mode: 0 for mode in MODES} for metric in METRICS}
    for index, (doc, saved) in enumerate(zip(documents, predictions)):
        for field in ("id", "question", "choices", "labels", "gold"):
            require(saved[field] == doc[field], f"{name}: prediction/source mismatch {index}/{field}")
        require(type(doc["gold"]) is int and 0 <= doc["gold"] < len(doc["choices"]), "Invalid gold index")
        require(len(doc["labels"]) == len(doc["choices"]) and all(doc["choices"]), "Invalid answer choices")
        scores = {mode: [candidates[index, choice][mode] for choice in range(len(doc["choices"]))]
                  for mode in MODES}
        norm = {mode: [value / len(choice) for value, choice in zip(values, doc["choices"])]
                for mode, values in scores.items()}
        verdicts = {}
        require(set(saved["predictions"]) == set(METRICS), f"{name}: prediction metrics mismatch")
        for field, by_mode in (("scores", scores), ("scores_per_character", norm)):
            require(set(saved[field]) == set(MODES), f"{name}: prediction score modes mismatch")
            for mode, expected in by_mode.items():
                require(len(saved[field][mode]) == len(expected), f"{name}: prediction choices mismatch")
                for choice_index, value in enumerate(expected):
                    close(saved[field][mode][choice_index], value,
                          f"{name}: saved {field}/{index}/{mode}/{choice_index}")
        for metric, by_mode in (("acc", scores), ("acc_norm", norm)):
            require(set(saved["predictions"][metric]) == set(MODES), f"{name}: prediction modes mismatch")
            verdicts[metric] = {}
            for mode, values in by_mode.items():
                chosen = max(range(len(values)), key=values.__getitem__)
                require(saved["predictions"][metric][mode] == chosen,
                        f"{name}: argmax mismatch at {index}/{metric}/{mode}")
                verdicts[metric][mode] = chosen
                correct[metric][mode] += int(chosen == doc["gold"])
        reconstructed.append({"id": doc["id"], "gold": doc["gold"], "predictions": verdicts})
    require(summary["n"] == N_DOCUMENTS and summary["correct"] == correct,
            f"{name}: independently recomputed correct counts differ from summary")
    accuracy = {metric: {mode: 100 * count / N_DOCUMENTS for mode, count in values.items()}
                for metric, values in correct.items()}
    for metric in METRICS:
        require(set(summary["accuracy_percent"][metric]) == set(MODES), f"{name}: summary modes mismatch")
        for mode in MODES:
            close(summary["accuracy_percent"][metric][mode], accuracy[metric][mode],
                  f"{name}: summary percentage {metric}/{mode}")
    paper = dict(zip(MODES[:3], config["paper"]))
    model_manifest = read_json(ROOT / config["model_manifest"])
    return {
        "display": config["display"], "configuration": config,
        "results_path": str(output.relative_to(ROOT)), "n": N_DOCUMENTS,
        "correct": correct, "accuracy_percent": accuracy, "paper_table2_arc_easy": paper,
        "actual_minus_paper_pp": {mode: accuracy["acc_norm"][mode] - value for mode, value in paper.items()},
        "paired": {metric: {mode: paired_stats(reconstructed, mode, metric) for mode in GUIDED}
                   for metric in METRICS},
        "checkpoint": {"repository": model_manifest.get("repository"), "revision": model_manifest.get("revision")},
        "audit": {
            "passed": True, "unique_documents": N_DOCUMENTS, "unique_candidates": len(candidates),
            "answer_token_count_per_policy": total_answer_tokens,
            "canonical_test_and_training_fewshot_context_verified": True,
            "independent_retokenization": True, "all_token_sums_predictions_and_summary_match": True,
            "tokenizer_sha256": digest(tokenizer_path), "verified_code": verified_code,
            "result_file_sha256": {filename: digest(output / filename) for filename in REQUIRED},
            "validation": validation_audit,
        },
        "run_manifest": manifest, "native_validation": validation,
        "runtime": {key: value for key, value in summary.items()
                    if key.endswith("_seconds") or key == "peak_gpu_allocated_gib"},
    }


def report(result):
    models = result["models"]
    lines = [
        "# ARC-Easy: LoopCD-Logits reproduction for all paper model configurations",
        "",
        "This report covers LoopCD-Logits on ARC-Easy for every model configuration in Table 2 of the paper. "
        "It excludes Thinking, hidden-state guidance, and speed extensions. "
        "Four model families, six checkpoints, and seven loop-depth configurations are complete, each using all 2,376 questions and 9,501 candidate answers. "
        "Scores below were independently recomputed from saved answer-token log probabilities and checked against retokenization, per-question predictions, summaries, and file hashes.",
        "",
        "## Reproduction and paper results",
        "",
        "The predetermined display metric is acc_norm: the sum of conditional log likelihoods over every candidate-answer token, divided by the original answer text's Unicode character count, "
        "excluding any added leading space. The paper does not explicitly distinguish acc from acc_norm, so unnormalized acc is also reported. "
        "These results do not establish exact agreement with the authors' unreleased evaluation program. Cells show reproduced / paper percentages.",
        "",
        "| Configuration | Baseline | LoopCD fixed | LoopCD adaptive | Fixed net correct / pp | Adaptive net correct / pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models.values():
        accuracy, paper, paired = model["accuracy_percent"]["acc_norm"], model["paper_table2_arc_easy"], model["paired"]["acc_norm"]
        values = [f"{accuracy[mode]:.2f} / {paper[mode]:.2f}" for mode in MODES[:3]]
        changes = [f"{paired[mode]['net_correct']:+d} / {paired[mode]['delta_percentage_points']:+.2f}" for mode in GUIDED]
        lines.append("| " + " | ".join([model["display"], *values, *changes]) + " |")
    opposite_directions = []
    for model in models.values():
        for mode in GUIDED:
            normalized = model["paired"]["acc_norm"][mode]["delta_percentage_points"]
            raw = model["paired"]["acc"][mode]["delta_percentage_points"]
            if normalized * raw < 0:
                opposite_directions.append(
                    f"{model['display']} {MODE_NAMES[mode]}: acc_norm {normalized:+.2f} pp, acc {raw:+.2f} pp"
                )
    if opposite_directions:
        lines.extend([
            "",
            "**Conclusions depend on the scoring metric.** The following comparisons change direction between metrics and do not support a claim of improvement across the board: "
            + "; ".join(opposite_directions) + ". Both metrics were retained before the experiments and were not selected from the results.",
        ])
    lines.extend([
        "",
        "Baseline denotes each model's native final output at its specified depth; the Looped-Qwen baseline is the output after eight damped loop updates. "
        "Each comparison shares a single loop trajectory. The weak reference uses the state after the first complete update, passed through the model's full output path to obtain logits.",
        "",
        "## Paired changes on the same questions (acc_norm)",
        "",
        "| Configuration | Policy | Corrected | Regressed | Changed answers | Net correct | Paired 95% CI (pp) | McNemar p | Holm p |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ])
    for model in models.values():
        for mode, values in model["paired"]["acc_norm"].items():
            low, high = values["paired_bootstrap_95ci_pp"]
            lines.append(
                f"| {model['display']} | {MODE_NAMES[mode]} | {values['corrected']} | {values['regressed']} | "
                f"{values['changed_answers']} | {values['net_correct']:+d} | [{low:+.2f}, {high:+.2f}] | "
                f"{values['mcnemar_exact_two_sided_p']:.4f} | {values['mcnemar_holm_adjusted_p']:.4f} |"
            )
    significant = [
        f"{m['display']} {MODE_NAMES[mode]} ({'increase' if values['net_correct'] > 0 else 'decrease'})"
        for m in models.values() for mode, values in m["paired"]["acc_norm"].items()
        if values["mcnemar_holm_adjusted_p"] < .05
    ]
    lines.extend([
        "",
        "Corrected and regressed counts compare the same questions against baseline; changed answers also include switches between incorrect options. "
        "The 95% intervals use 20,000 paired bootstrap resamples (seed=1234) and describe uncertainty from sampling questions. "
        "They exclude uncertainty from random initial states, repeated model runs, and protocol selection. McNemar tests are exact and two-sided. "
        "Holm correction applies across the 14 primary-metric comparisons; the intervals themselves are not adjusted for multiple comparisons.",
        "",
        ("Under the primary acc_norm metric, the following changes meet the Holm p<0.05 criterion: " + "; ".join(significant) + ".")
        if significant else "Under the primary acc_norm metric, no change has sufficient evidence of significance at Holm p<0.05.",
        "",
        "## Complete accuracy results for both metrics",
        "",
        "Each cell shows the correct count (percentage). The weak reference is diagnostic only and has no corresponding column in the paper's Table 2.",
        "",
        "| Configuration | Metric | Baseline | Fixed | Adaptive | First-loop weak reference |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for model in models.values():
        for metric in METRICS:
            cells = [f"{model['correct'][metric][mode]} ({model['accuracy_percent'][metric][mode]:.2f}%)" for mode in MODES]
            lines.append("| " + " | ".join([model["display"], metric, *cells]) + " |")
    raw_significant = [
        f"{model['display']} {MODE_NAMES[mode]} ({values['delta_percentage_points']:+.2f} pp, "
        f"Holm p={values['mcnemar_holm_adjusted_p']:.4f})"
        for model in models.values() for mode, values in model["paired"]["acc"].items()
        if values["mcnemar_holm_adjusted_p"] < .05
    ]
    if raw_significant:
        lines.extend(["", "For the supplementary acc metric, the following changes meet Holm p<0.05 across that metric's 14 comparisons: "
                      + "; ".join(raw_significant) + "."])
    lines.extend([
        "",
        "Corrected and regressed counts, net changes, bootstrap intervals, and McNemar tests for acc are also saved in comparison.json. "
        "Its Holm correction is performed separately across the 14 acc comparisons as a supplementary metric.",
        "",
        "## Fixed protocol and comparability limitations",
        "",
        "- Ouro uses 8-shot examples from the training set; other configurations are 0-shot. Questions retain their original order, with prompts formatted as `Question: ...\\nAnswer:`. "
        "Scoring uses complete answer text rather than option letters, without chat templates, generated answers, or context truncation. "
        "The 8-shot procedure advances one `random.Random(1234)` instance with `.sample(train, 8)`, resampling examples for each question without selecting against test labels.",
        "- The formula is `z′ = z_R + ω(z_R − z_1)`. Fixed ω=0.5; adaptive "
        "`ω = cap × [1 − (p_R,top1 − p_R,top2)]`, with probabilities computed over the full vocabulary. "
        "Huginn uses cap=0.5; all others use cap=1.0, following Table A4. No accuracy-driven strength or seed search was performed.",
        "- All four Ouro loops traverse the complete stack and use native normalization. Huginn performs 16 or 32 complete updates. "
        "Its first-update state passes through the full coda and output normalization; the official tokenizer/evaluator requires exactly one BOS.",
        "- Huginn and the published Parcae checkpoints have stochastic loop initialization. To make candidate answers, batch lengths, and policies comparable, "
        "the native initialization distribution is retained, with shared position states determined by seed=1234 and question ID. Answer text does not determine the seed. "
        "Native forward checks use the same explicit initial state. The paper does not specify seeds or their coupling, so this is an explicit reproduction assumption.",
        "- Looped-Qwen uses the original official Qwen/Qwen3-4B release weights, not the separate Qwen3-4B-Base checkpoint. "
        "It uses a 15-layer prefix, the zero-indexed loop window 15:19, and a 17-layer tail. Eight updates follow "
        "`u_next = (7/8)u + (1/8)g(u)`, without an additional anchor. The early output passes through the complete tail/head after the first damped update. "
        "K=8 follows the LoopCD appendix; the default K=6/anchor recipe in the upstream Training-Free project index README is not used. "
        "Downloading the authors' GitHub source returned 404, with evidence retained. The wrapper follows the paper and is checked by independent eight-step replay and recovery of the ordinary dense output at K=1.",
        "- The paper does not fix complete model revisions, a harness commit, few-shot examples, random states, or every numerical implementation detail. "
        "This project freezes local official weights and source hashes and uses fixed lm-eval prompt rules. Both agreement with and deviations from the paper are reported. "
        "These ARC-Easy configurations alone cannot establish results for other tasks, hidden-state guidance, or speed gains from fewer loops.",
        "",
        "## Numerical validation and joint-run resources",
        "",
        "Each configuration checks zero-guidance identity, agreement with native or independently replayed loop outputs, distinct early and final readouts, and agreement between single-example and padded-batch scoring. "
        "The new unified evaluator also checks that changing masked padding tokens leaves valid outputs unchanged. The earlier Ouro-1.4B run reuses its completed native-output and batch-consistency checks. "
        "GPU runs use the dtype and matmul settings below; logit guidance and log_softmax use FP32. "
        "`high` permits faster internal matrix multiplication, while `highest` uses the highest precision. Precision changes follow numerical consistency checks before scoring, without accuracy-based tuning.",
        "",
        "Huginn and Looped-Qwen initially failed numerical preflight with `high` and switched to `highest` before formal scoring, "
        "retaining the formulas and validation tolerances. Failed error measurements, job records, and precision decisions are recorded in precision_decisions.json.",
        "",
        "| Configuration | dtype / matmul | Maximum single/batch token logp difference | Scoring seconds in this invocation | Peak GPU memory (GiB) |",
        "|---|---|---:|---:|---:|",
    ])
    for model in models.values():
        manifest, runtime = model["run_manifest"], model["runtime"]
        lines.append(
            f"| {model['display']} | {manifest['dtype']} / {manifest['float32_matmul_precision']} | "
            f"{model['audit']['validation']['batch_single_max_error']:.3g} | "
            f"{runtime['this_invocation_scoring_seconds']:.1f} | {runtime['peak_gpu_allocated_gib']:.2f} |"
        )
    lines.extend([
        "",
        "These resources cover the joint experiment in which baseline, both LoopCD policies, and the weak reference share loop trajectories. "
        "They cannot establish standalone policy speedups or validate the paper's 'almost free' throughput claim. "
        "Timing covers the current invocation and excludes earlier invocations when resuming saved candidates. Model loading/validation time and GPU/job information are recorded in comparison.json.",
        "",
        "## Auditable artifacts and sources",
        "",
        "Each configuration directory retains candidate_scores.jsonl, predictions.jsonl, summary.json, run_manifest.json, and validation.json. "
        "New runs additionally save source_snapshot; Ouro-1.4B reuses ../arc_easy. comparison.json contains all audit counts, file hashes, "
        "run parameters, and statistics for both metrics. Regenerate this report with `venv/bin/python scripts/analyze_arc_suite.py`.",
        "",
        "Independent data reconstruction is recorded in data_audit.json; checks of official model implementations and metadata are recorded in source_integrity.json. "
        "The aggregation script also verifies the original Train/Test file hashes, reconstructs all original questions and few-shot contexts, "
        "and requires native-output or independent-replay validation fields for each model family. Zero-guidance algebraic identity is not a substitute for native-output consistency.",
        "",
        "Sources: [LoopCD Tables 2, A1, A2, and A4](https://arxiv.org/html/2610.02185v1), "
        "[Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1), and "
        "[pinned lm-eval source](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). "
        "Official model repositories and exact revisions are recorded in models/*manifest.json; data sources, character normalization, and sampling references are recorded in data/protocol.json.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true",
                        help="Audit completed configurations and print progress; do not write final artifacts.")
    args = parser.parse_args()
    registry_path = ROOT / "configs/arc_all_models.json"
    registry = read_json(registry_path)
    require(registry["dataset_size"] == N_DOCUMENTS and registry["seed"] == 1234, "Registry protocol changed")
    require(len(registry["models"]) == 7, "Expected all seven paper configurations")
    list((ROOT / "results/arc_all_models").iterdir())
    models, incomplete, cache = {}, {}, {}
    for name, config in registry["models"].items():
        output = ROOT / config.get("reuse", f"results/arc_all_models/{name}")
        # Listing refreshes newly created NFS directory entries before stat/read.
        if output.exists():
            list(output.iterdir())
        missing = [filename for filename in REQUIRED if not (output / filename).exists()]
        if missing:
            incomplete[name] = missing
            continue
        print(f"Auditing {name}: token alignment, all four policies, all 2376 questions", file=sys.stderr, flush=True)
        models[name] = audit_model(name, config, cache)
    result = {
        "complete": not incomplete, "scope": registry["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": digest(registry_path),
        "analysis_script_sha256": digest(Path(__file__)),
        "expected_documents_per_configuration": N_DOCUMENTS,
        "expected_candidates_per_configuration": N_CANDIDATES,
        "metric_note": "acc_norm primary: summed conditional log likelihood / Python len(original answer); acc unnormalized also reported.",
        "statistical_protocol": {
            "bootstrap": "20000 paired question resamples, seed 1234; percentile pointwise 95% CI",
            "mcnemar": "Exact two-sided conditional binomial test",
            "holm": "14 comparisons across all 7 configurations per metric; only computed once all complete",
            "uncertainty_excludes": "Between-seed/model-run variation and evaluation protocol differences",
        },
        "supporting_evidence": {
            filename: {"path": f"results/arc_all_models/{filename}",
                       "sha256": digest(ROOT / "results/arc_all_models" / filename)}
            for filename in ("data_audit.json", "source_integrity.json", "precision_decisions.json", "cpu_tests.json")
            if (ROOT / "results/arc_all_models" / filename).exists()
        },
        "models": models, "incomplete": incomplete,
    }
    if not incomplete:
        holm_adjust(models)
    if args.partial:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    require(not incomplete, f"Final report requires seven complete results; missing: {incomplete}. Use --partial for progress.")
    output = ROOT / "results/arc_all_models"
    # All audits finish before any final artifact is written.
    json_text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    markdown = report(result)
    for filename, text in (("comparison.json", json_text), ("REPORT.md", markdown)):
        temporary = output / (filename + ".tmp")
        temporary.write_text(text)
        temporary.replace(output / filename)
    print(json.dumps({"complete": True, "audited_configurations": len(models),
                      "report": str(output / "REPORT.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
