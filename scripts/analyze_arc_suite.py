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
MODE_NAMES = {"loopcd_fixed": "固定", "loopcd_adaptive": "自适应"}
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
        "# ARC-Easy：论文全部模型配置的 LoopCD-Logits 复现",
        "",
        "本报告仅覆盖论文 Table 2 的 ARC-Easy 全部模型配置及 LoopCD-Logits，"
        "不包括 Thinking、隐藏态引导或速度扩展。"
        "完成4个模型家族、6份权重、7个循环深度配置；每项均使用完整2376题、9501个候选答案。"
        "下表的分数均由保存的答案 token 对数概率独立重算，且通过重新分词、逐题预测、汇总结果及文件哈希核对。",
        "",
        "## 实测与论文",
        "",
        "预先固定以 acc_norm 展示：候选答案全部 token 的条件对数似然之和，除以原答案文本的 Unicode 字符数，"
        "不计额外添加的前导空格。论文没有明确区分 acc 与 acc_norm，故同时报告未归一化 acc，"
        "不能据此认定与作者未公开的评测程序完全一致。单元格为“本次 / 论文”，单位为%。",
        "",
        "| 配置 | 普通解码 | LoopCD 固定 | LoopCD 自适应 | 固定净增答对 / pp | 自适应净增答对 / pp |",
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
                    f"{model['display']} {MODE_NAMES[mode]}：acc_norm {normalized:+.2f} pp，acc {raw:+.2f} pp"
                )
    if opposite_directions:
        lines.extend([
            "",
            "**结论对评分口径敏感。** 以下比较在两种口径下方向相反，不能概括为全面提升："
            + "；".join(opposite_directions) + "。两种口径在实验前均已固定保留，没有按结果选择。",
        ])
    lines.extend([
        "",
        "普通解码指各模型指定深度的原生最终输出；Looped-Qwen 的基线是8步阻尼循环后的输出。"
        "所有比较均使用各自同一次循环轨迹；弱参考取第1次完整更新后的状态，经该模型完整输出路径得到 logits。",
        "",
        "## 同题配对变化（acc_norm）",
        "",
        "| 配置 | 策略 | 改对 | 改错 | 答案改变 | 净增 | 95%配对区间（pp） | McNemar p | Holm p |",
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
        f"{m['display']} {MODE_NAMES[mode]}（{'提升' if values['net_correct'] > 0 else '下降'}）"
        for m in models.values() for mode, values in m["paired"]["acc_norm"].items()
        if values["mcnemar_holm_adjusted_p"] < .05
    ]
    lines.extend([
        "",
        "改对/改错按同题相对于普通解码计算；“答案改变”还包含错误选项之间的变化。"
        "95%区间使用20000次配对 bootstrap（seed=1234），反映题目抽样不确定性，"
        "不包含随机初态、多次模型运行或协议选择的不确定性。McNemar 为精确双侧检验；"
        "Holm 在14个主口径比较内校正多重检验，区间本身未经多重校正。",
        "",
        ("在 acc_norm 主口径下，按 Holm p<0.05 的判据，本次有：" + "；".join(significant) + "。")
        if significant else "在 acc_norm 主口径下，按 Holm p<0.05 的判据，本次没有足够证据认定任何一项变化显著。",
        "",
        "## 两种口径的完整正确率",
        "",
        "各项以“答对数（百分比）”列出；弱参考仅用于诊断，没有对应论文 Table 2 列。",
        "",
        "| 配置 | 口径 | 普通解码 | 固定 | 自适应 | 第1循环弱参考 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for model in models.values():
        for metric in METRICS:
            cells = [f"{model['correct'][metric][mode]}（{model['accuracy_percent'][metric][mode]:.2f}%）" for mode in MODES]
            lines.append("| " + " | ".join([model["display"], metric, *cells]) + " |")
    raw_significant = [
        f"{model['display']} {MODE_NAMES[mode]}（{values['delta_percentage_points']:+.2f} pp，"
        f"Holm p={values['mcnemar_holm_adjusted_p']:.4f}）"
        for model in models.values() for mode, values in model["paired"]["acc"].items()
        if values["mcnemar_holm_adjusted_p"] < .05
    ]
    if raw_significant:
        lines.extend(["", "补充 acc 口径中，通过该口径14项比较的 Holm p<0.05 判据的变化为："
                      + "；".join(raw_significant) + "。"])
    lines.extend([
        "",
        "acc 的改对/改错、净增、bootstrap 和 McNemar 检验也完整保存在 comparison.json；"
        "其 Holm 校正在独立的14个 acc 比较内进行，属于补充口径。",
        "",
        "## 固定协议与可比性限制",
        "",
        "- Ouro 使用训练集8-shot；其余配置0-shot。所有题目保留原顺序，提示为 `Question: ...\\nAnswer:`，"
        "评分完整答案文本，不评分选项字母，不使用聊天模板，不生成答案、不截断上下文。"
        "8-shot 用 advancing `random.Random(1234).sample(train, 8)`，每题重新抽示例；不依据测试标签挑示例。",
        "- 公式为 `z′ = z_R + ω(z_R − z_1)`。固定 ω=0.5；自适应 "
        "`ω = cap × [1 − (p_R,top1 − p_R,top2)]`，概率在完整词表上计算。"
        "Huginn 的 cap=0.5，其余 cap=1.0；均来自论文 Table A4，无准确率驱动的强度或种子搜索。",
        "- Ouro 的4轮均走完整堆栈，每轮使用原生归一化。Huginn 使用16/32次完整更新，"
        "第1更新状态经过完整 coda 和输出归一化；其官方 tokenizer/evaluator 要求恰好一个 BOS。",
        "- Huginn 与已发布 Parcae 检查点的循环初态含随机量。为使候选答案、批量长度和策略之间可比较，"
        "保持原生初态分布，按 seed=1234 与问题 ID 确定共享的位置初态；不以答案文本产生种子。"
        "原生前向验证使用同一显式初态。论文未说明随机种子/耦合方式，此处是明确的复现假设。",
        "- Looped-Qwen 采用官方 Qwen/Qwen3-4B 原始发布权重（不是单独的 Qwen3-4B-Base），"
        "前15层、循环窗口零起始索引15:19、后17层；8次更新为 "
        "`u_next = (7/8)u + (1/8)g(u)`，无额外 anchor。早期输出在第1次阻尼更新后继续完整 tail/head。"
        "遵循 LoopCD 附录的 K=8；上游 Training-Free 项目索引 README 的默认 K=6/anchor 配方不用于本次。"
        "作者 GitHub 源码下载返回404，证据留档；wrapper 根据论文实现，并检查独立8步重放及 K=1 恢复普通 dense 输出。",
        "- 论文未固定完整模型 revision、harness commit、few-shot实例、随机状态及所有数值实现细节。"
        "本项目冻结本地官方权重和源码哈希，采用固定 lm-eval 提示规则；分数接近或偏离论文均如实报告。"
        "这些配置在同一 ARC-Easy 任务上的结果不支持推断其他任务、隐藏态引导或减循环速度结论。",
        "",
        "## 数值验证和联合运行资源",
        "",
        "每项检查零强度恒等、原生/独立循环输出一致性、早晚读出不同、单条与补齐批量评分一致性；"
        "新统一评测器还检查改变被掩码的补齐 token 不影响有效输出。旧 Ouro-1.4B 复用先前完成的原生和批量一致性验证。"
        "GPU 运行时采用下表记录的 dtype 与 matmul 设置；logits 引导和 log_softmax 为 FP32。"
        "`high` 允许较快的内部矩阵乘法，`highest` 使用最高精度。精度切换依据评分前的数值一致性检查，未查看准确率调参。",
        "",
        "Huginn 与 Looped-Qwen 的 `high` 预检曾未通过，随后在正式评分前改用 `highest`，"
        "保留原有公式和校验容差；具体失败误差、作业及精度决策见 precision_decisions.json。",
        "",
        "| 配置 | dtype / matmul | 单条-批量最大 token logp 差 | 本次评分调用秒数 | 峰值显存 GiB |",
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
        "以上是普通解码、两种 LoopCD 和弱参考共享循环轨迹的联合实验资源，"
        "不能用它计算独立策略加速比或验证论文“几乎免费”的吞吐结论。"
        "计时字段记录本次调用；如恢复已有候选，不包含此前调用的时间。模型加载/验证时间及 GPU/作业信息见 comparison.json。",
        "",
        "## 可复核产物与来源",
        "",
        "每项子目录保留 candidate_scores.jsonl、predictions.jsonl、summary.json、run_manifest.json、validation.json，"
        "新运行另保存 source_snapshot；Ouro-1.4B 复用 ../arc_easy。comparison.json 包含全部审计计数、文件哈希、"
        "运行参数和两种口径统计，可用 `venv/bin/python scripts/analyze_arc_suite.py` 重新生成本报告。",
        "",
        "独立的数据重建记录见 data_audit.json；官方模型实现和元数据核验见 source_integrity.json。"
        "聚合脚本还自行核验原始 Train/Test 文件哈希，重建全部原题和 few-shot 上下文，"
        "按模型家族强制检查原生输出/独立重放验证字段，不能以零强度代数恒等代替原生一致性。",
        "",
        "依据：[LoopCD 论文 Table 2、A1、A2、A4](https://arxiv.org/html/2610.02185v1)、"
        "[Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1)、"
        "[固定 lm-eval 源码](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520)。"
        "模型官方仓库及精确版本见各 models/*manifest.json；数据来源、字符归一化和采样依据见 data/protocol.json。",
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
