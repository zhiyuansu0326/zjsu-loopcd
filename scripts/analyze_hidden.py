#!/usr/bin/env python3
"""Audit the fixed Huginn R16/R32 × four-task LoopCD-Hidden experiment.

Partial mode audits completed runs and prints progress only. Final artifacts
require all eight full datasets and their standalone cost measurements.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze
from scripts.analyze_reduced_depth import (
    audit_historical_arc, paired_comparison, verify_reused_baseline,
)
from scripts.reduced_suite import (
    BOOTSTRAP_SAMPLES, PERMUTATION_SAMPLES, PRIMARY_METRICS, SEED, TASKS,
    holm_adjust, macro_paired_stats,
)
from src.registry import load_registry, resolve_run

MODELS = ("huginn_16", "huginn_32")
REFERENCES = {"huginn_16": 7, "huginn_32": 6}
POLICIES = ("baseline", "loopcd_logits", "loopcd_hidden")
CONTRASTS = {
    "hidden_vs_baseline": ("baseline", "loopcd_hidden", "Hidden − Baseline"),
    "hidden_vs_logits": ("loopcd_logits", "loopcd_hidden", "Hidden − Logits"),
}
FIXED_PARAMETERS = {
    "checkpoint": "tomg-group-umd/huginn-0125",
    "loops_by_model": {"huginn_16": 16, "huginn_32": 32},
    "reference_steps": REFERENCES, "fixed_strength": 0.5,
    "logits_reference_step": 1, "seed": SEED,
    "dtype": "float32", "matmul_precision": "highest", "shots": 0,
}
SHARED_MANIFEST_FIELDS = (
    "data_sha256", "protocol_sha256", "model_manifest_sha256", "native_source_hashes",
    "dtype", "float32_matmul_precision", "seed", "shots", "add_bos", "chat_template",
    "batch_size", "token_budget", "torch", "transformers", "gpu",
)
PAPER_PATH = ROOT / "configs/paper_hidden_reference.json"


def validate_plan(plan):
    analyze.require(plan["task_order"] == list(TASKS), "Frozen task order changed")
    analyze.require(plan["model_order"] == list(MODELS), "Frozen model order changed")
    analyze.require(plan["primary_metrics"] == PRIMARY_METRICS, "Frozen primary metrics changed")
    analyze.require(plan["fixed_parameters"] == FIXED_PARAMETERS, "Frozen Hidden settings changed")
    for name, count in (("bootstrap", BOOTSTRAP_SAMPLES), ("permutation", PERMUTATION_SAMPLES)):
        analyze.require(plan[name]["samples"] == count and plan[name]["seed"] == SEED,
                        f"Frozen {name} settings changed")
    analyze.require(plan["multiplicity"]["primary_tasks"] == 16
                    and plan["multiplicity"]["primary_macro"] == 4,
                    "Frozen multiple-testing families changed")
    hashes = plan["configuration_sha256"]
    analyze.require({"configs/models.json", "configs/tasks.json", "configs/hidden.json"} <= set(hashes),
                    "Incomplete frozen configuration inventory")
    for path, expected in hashes.items():
        analyze.require(analyze.digest(ROOT / path) == expected,
                        f"Frozen configuration changed: {path}")
    timing = plan["timing"]
    analyze.require(timing["batches_per_task"] == 3 and timing["candidates_per_batch"] == 8
                    and timing["warmups_per_policy_per_batch"] == 2
                    and timing["repetitions_per_policy_per_batch"] == 6,
                    "Frozen timing sample/repetition counts changed")


def validate_shared_provenance(original, hidden):
    """Compare experimental inputs without requiring today's source versions."""
    old = original.get("comparable_manifest", original["run_manifest"])
    new = hidden["run_manifest"]
    for field in SHARED_MANIFEST_FIELDS:
        analyze.require(old[field] == new[field], f"Original/Hidden setting differs: {field}")
    old_config, new_config = dict(old["configuration"]), new["configuration"]
    historical_alignment = None
    if "historical_schema_alignment" in original:
        alignment = original["historical_schema_alignment"]
        analyze.require(isinstance(alignment, dict)
                        and alignment.get("original_manifest_preserved") is True
                        and "comparable_manifest" in original,
                        "Historical configuration alignment requires audited original provenance")
        analyze.require(type(old_config.get("shots")) is int
                        and old_config["shots"] == old["shots"] == new["shots"],
                        "Historical configuration shots differ from matched manifest shots")
        old_config.pop("shots")
        dropped = ["paper"] if "paper" in old_config else []
        old_config.pop("paper", None)
        historical_alignment = {"verified_shots": old["shots"],
                                "ignored_informational_fields": dropped}
    analyze.require(old_config.get("reference_step", 1) == 1 and old_config["fixed"] == 0.5,
                    "Original fixed Logits comparison must use h1 and strength 0.5")
    # Older ARC-Easy snapshots did not record matmul precision inside configuration.
    ignored = {"reference_step", "matmul_precision"}
    analyze.require({key: value for key, value in old_config.items() if key not in ignored}
                    == {key: value for key, value in new_config.items() if key not in ignored},
                    "Original/Hidden model configurations differ beyond reference selection")
    old_metadata, new_metadata = old["adapter_metadata"], new["adapter_metadata"]
    noise_keys = {key for key in old_metadata if key.startswith("noise_")}
    analyze.require(noise_keys and noise_keys == {key for key in new_metadata if key.startswith("noise_")},
                    "Initial-state metadata inventory changed")
    for key in sorted(noise_keys):
        analyze.require(old_metadata[key] == new_metadata[key],
                        f"Initial-state protocol changed: {key}")
    analyze.require(old_metadata.get("reference_recurrent_update", 1) == 1,
                    "Original Logits adapter does not use h1")
    return {"matched_manifest_fields": list(SHARED_MANIFEST_FIELDS),
            "matched_noise_fields": sorted(noise_keys), "original_fixed_logits_reference_step": 1,
            "original_source_snapshots_preserved": True,
            "historical_configuration_alignment": historical_alignment}


def audit_setting(task, model, output, tasks_registry, paper_logits, cache, *, include_costs=True):
    original_run = resolve_run(task, model)
    hidden_run = resolve_run(task, model, guidance_space="hidden")
    hidden_run["output"] = Path(output) / task / model
    if task == "arc_easy":
        original = audit_historical_arc(model, original_run, cache)
    else:
        original = analyze.audit_model(task, model, original_run, tasks_registry, paper_logits, cache)
    hidden = analyze.audit_model(task, model, hidden_run, tasks_registry, paper_logits, cache)
    provenance = validate_shared_provenance(original, hidden)
    replay = verify_reused_baseline(original_run["output"], hidden_run["output"])
    replay["bitwise_equal_saved_token_logprobs"] = replay["maximum_token_logprob_difference"] == 0
    predictions = {
        "original": analyze.read_jsonl(original_run["output"] / "predictions.jsonl"),
        "hidden": analyze.read_jsonl(hidden_run["output"] / "predictions.jsonl"),
    }
    modes = {"baseline": (hidden, "baseline"), "loopcd_logits": (original, "loopcd_fixed"),
             "loopcd_hidden": (hidden, "loopcd_hidden")}
    policies = {
        policy: {
            "n": row["n"], "results_path": row["results_path"], "saved_policy": mode,
            "correct": {metric: row["correct"][metric][mode] for metric in analyze.METRICS},
            "accuracy_percent": {
                metric: row["accuracy_percent"][metric][mode] for metric in analyze.METRICS
            },
        } for policy, (row, mode) in modes.items()
    }
    paired = {
        metric: {
            "hidden_vs_baseline": paired_comparison(
                predictions["hidden"], "baseline", predictions["hidden"], "loopcd_hidden", metric,
            ),
            "hidden_vs_logits": paired_comparison(
                predictions["original"], "loopcd_fixed", predictions["hidden"], "loopcd_hidden", metric,
            ),
        } for metric in analyze.METRICS
    }
    paper = analyze.read_json(PAPER_PATH)
    result = {
        "complete": include_costs, "task": task, "model": model,
        "display": hidden_run["task"]["display"], "primary_metric": PRIMARY_METRICS[task],
        "task_configuration": hidden_run["task"], "policies": policies, "paired": paired,
        "paper_table3": paper["scores_percent"][task][model],
        "baseline_replay": replay, "shared_provenance": provenance,
        "audited_runs": {"original": original, "hidden": hidden},
    }
    if include_costs:
        from scripts.hidden_cost_audit import audit_costs

        path = Path(output) / task / f"{model}_costs.json"
        result["cost_audit"] = audit_costs(path, hidden["run_manifest"])
        result["costs"] = analyze.read_json(path)
        result["costs_file"] = analyze.relative(path)
        result["costs_sha256"] = analyze.digest(path)
    return result


def validate_results(results):
    analyze.require(set(results) == set(TASKS)
                    and all(set(results[task]) == set(MODELS) for task in TASKS),
                    "All eight fixed task/depth settings are required")
    _, registry = load_registry()
    common_checkpoint = None
    for task in TASKS:
        for model in MODELS:
            row = results[task][model]
            analyze.require(row.get("complete") is True and row["task"] == task and row["model"] == model,
                            f"Incomplete or mismatched setting: {task}/{model}")
            analyze.require(row["primary_metric"] == PRIMARY_METRICS[task], "Frozen primary metric changed")
            analyze.require(set(row["audited_runs"]) == {"original", "hidden"}
                            and all(run["audit"]["passed"] is True for run in row["audited_runs"].values()),
                            "Missing successful independent score audits")
            manifest = row["audited_runs"]["hidden"]["run_manifest"]
            config, metadata = manifest["configuration"], manifest["adapter_metadata"]
            loops = FIXED_PARAMETERS["loops_by_model"][model]
            analyze.require(config["family"] == "huginn" and config["loops"] == loops
                            and config["fixed"] == 0.5 and config["reference_step"] == REFERENCES[model]
                            and metadata["loops"] == loops
                            and metadata["reference_recurrent_update"] == REFERENCES[model]
                            and manifest["guidance_space"] == "hidden",
                            "Executed Hidden configuration differs from Table 3")
            analyze.require(manifest["dtype"] == "float32"
                            and manifest["float32_matmul_precision"] == "highest"
                            and manifest["seed"] == SEED and manifest["shots"] == 0,
                            "Frozen precision/seed/shot protocol changed")
            identity = (manifest["model_manifest_sha256"], manifest["native_source_hashes"],
                        metadata["model_revision"])
            if common_checkpoint is None:
                common_checkpoint = identity
            analyze.require(identity == common_checkpoint, "Checkpoint changed across settings")
            n = registry["tasks"][task]["count"]
            analyze.require(set(row["policies"]) == set(POLICIES), "Missing comparison policy")
            for policy, values in row["policies"].items():
                analyze.require(values["n"] == n, "Incomplete task split")
                for metric in analyze.METRICS:
                    count = values["correct"][metric]
                    analyze.require(type(count) is int and 0 <= count <= n, "Invalid correct count")
                    analyze.close(values["accuracy_percent"][metric], 100 * count / n,
                                  "Correct count and accuracy disagree")
            for metric in analyze.METRICS:
                for name, (left, right, _) in CONTRASTS.items():
                    statistics = row["paired"][metric][name]
                    corrected, regressed = statistics["corrected"], statistics["regressed"]
                    analyze.require(type(corrected) is int and type(regressed) is int
                                    and min(corrected, regressed) >= 0 and corrected + regressed <= n,
                                    "Invalid paired correctness counts")
                    difference = row["policies"][right]["correct"][metric] - row["policies"][left]["correct"][metric]
                    analyze.require(corrected - regressed == difference, "Paired counts disagree with accuracy")
                    analyze.close(statistics["delta_percentage_points"], 100 * difference / n,
                                  "Paired percentage-point change disagrees")
            replay = row["baseline_replay"]
            analyze.require(replay["identical_predictions_both_metrics"] is True
                            and math.isfinite(replay["maximum_token_logprob_difference"])
                            and 0 <= replay["maximum_token_logprob_difference"] <= replay["tolerance"] <= 1e-3,
                            "New baseline does not reproduce saved original baseline")
            costs = row["costs"]
            analyze.require(row["cost_audit"]["passed"] is True
                            and costs.get("complete") is True and costs.get("state") == "complete"
                            and costs["task"] == task and costs["model"] == model
                            and costs["loops"] == loops and costs["hidden_reference_step"] == REFERENCES[model]
                            and costs["fixed_strength"] == 0.5, "Incomplete or mismatched cost audit")
            analyze.require(len(costs["batches"]) == 3
                            and all(batch["candidate_count"] == 8 for batch in costs["batches"])
                            and costs["methodology"]["warmup_per_policy_per_batch"] == 2
                            and costs["methodology"]["repetitions_per_policy_per_batch"] == 6,
                            "Frozen standalone cost workload changed")
            for policy in POLICIES:
                timing = costs["timing"][policy]
                seconds = timing["aggregate_seconds"]["median"]
                analyze.require(math.isfinite(seconds) and seconds > 0, "Missing positive standalone timing")
                baseline = costs["timing"]["baseline"]["aggregate_seconds"]["median"]
                analyze.close(timing["latency_ratio_to_baseline"], seconds / baseline,
                              "Standalone latency ratio mismatch")


def descriptive_view(results, model, metrics):
    means = {
        policy: math.fsum(results[task][model]["policies"][policy]["accuracy_percent"][metrics[task]]
                          for task in TASKS) / len(TASKS)
        for policy in POLICIES
    }
    comparisons = {}
    for name, (left, right, _) in CONTRASTS.items():
        deltas = {task: results[task][model]["paired"][metrics[task]][name]["delta_percentage_points"]
                  for task in TASKS}
        comparisons[name] = {
            "delta_percentage_points": means[right] - means[left], "task_deltas_pp": deltas,
            "wins": sum(value > 0 for value in deltas.values()),
            "ties": sum(value == 0 for value in deltas.values()),
            "losses": sum(value < 0 for value in deltas.values()),
        }
    return {"metrics_by_task": metrics, "accuracy_percent": means, "contrasts": comparisons}


def build_suite(results, plan):
    """Infer only from the frozen primary metrics; retain two descriptive views."""
    validate_plan(plan)
    validate_results(results)
    results = copy.deepcopy(results)
    means = {}
    for model in MODELS:
        means[model] = {
            "frozen_primary_metrics": descriptive_view(results, model, dict(PRIMARY_METRICS)),
            "all_acc": descriptive_view(results, model, {task: "acc" for task in TASKS}),
            "all_acc_norm": descriptive_view(results, model, {task: "acc_norm" for task in TASKS}),
        }
        for name, statistics in means[model]["frozen_primary_metrics"]["contrasts"].items():
            strata = [(results[task][model]["policies"]["baseline"]["n"],
                       results[task][model]["paired"][PRIMARY_METRICS[task]][name]["corrected"],
                       results[task][model]["paired"][PRIMARY_METRICS[task]][name]["regressed"])
                      for task in TASKS]
            inference = macro_paired_stats(strata)
            analyze.close(statistics["delta_percentage_points"], inference["delta_percentage_points"],
                          "Equal-task mean and paired macro delta disagree")
            statistics.update(inference)
    macro_p = holm_adjust({
        f"{model}/{name}": means[model]["frozen_primary_metrics"]["contrasts"][name]["paired_signflip_two_sided_p"]
        for model in MODELS for name in CONTRASTS
    })
    task_p = holm_adjust({
        f"{task}/{model}/{name}": results[task][model]["paired"][PRIMARY_METRICS[task]][name]["mcnemar_exact_two_sided_p"]
        for task in TASKS for model in MODELS for name in CONTRASTS
    })
    for model in MODELS:
        for name in CONTRASTS:
            means[model]["frozen_primary_metrics"]["contrasts"][name]["paired_signflip_holm_across_four_p"] = macro_p[f"{model}/{name}"]
            for task in TASKS:
                results[task][model]["paired"][PRIMARY_METRICS[task]][name]["mcnemar_holm_across_sixteen_p"] = task_p[f"{task}/{model}/{name}"]
    return {
        "complete": True, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "huginn_full_depth_hidden", "task_order": list(TASKS), "model_order": list(MODELS),
        "fixed_parameters": copy.deepcopy(FIXED_PARAMETERS), "tasks": results, "means": means,
        "sampled_latency_ratio_geometric_mean": {
            model: {
                denominator: {
                    policy: math.exp(math.fsum(math.log(
                        results[task][model]["costs"]["timing"][policy]["aggregate_seconds"]["median"]
                        / results[task][model]["costs"]["timing"][denominator]["aggregate_seconds"]["median"]
                    ) for task in TASKS) / len(TASKS))
                    for policy in POLICIES
                } for denominator in ("baseline", "loopcd_logits")
            } for model in MODELS
        },
        "paper_reference": analyze.read_json(PAPER_PATH),
        "statistical_protocol": {
            "primary_metrics": PRIMARY_METRICS, "task_weight": 0.25,
            "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": SEED,
                          "task_ci": "Paired question bootstrap, percentile 95% CI.",
                          "macro_ci": "Independent within-task paired resampling, equal-task mean delta, percentile 95% CI."},
            "permutation": {"samples": PERMUTATION_SAMPLES, "seed": SEED,
                            "method": "Paired sign flips conditional on within-task correctness discordances; two-sided plus-one Monte Carlo p."},
            "multiplicity": {"primary_tasks": "Holm across 16 exact McNemar tests: four tasks × two depths × two contrasts.",
                             "primary_macro": "Holm across four sign-flip tests: two depths × two contrasts."},
            "secondary_views": "All-acc and all-acc_norm means are descriptive. Primary inference uses only pre-specified task metrics.",
            "limitations": "Four fixed tasks and one deterministic initial-state seed; confidence intervals capture within-task question sampling, not checkpoint or initialization uncertainty.",
        },
    }


def render_report(result):
    lines = [
        "# Huginn LoopCD-Hidden at full depth across four tasks", "",
        "Eight complete evaluations use Huginn R16 with h7 and R32 with h6, both at fixed strength 0.5. "
        "All runs use 0-shot prompts, FP32/highest, and the original per-question initial states. "
        "Hidden blends raw recurrent states over the complete sequence before the native "
        "ln_f → coda → ln_f → lm_head readout. Comparisons use the same-depth baseline and "
        "original fixed Logits policy (h1, 0.5).", "",
        "| Task | Depth | Primary metric | Questions | Baseline % | Logits % | Hidden % | Hidden−baseline pp | Hidden−Logits pp | Paper baseline→Hidden % |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for model in MODELS:
            row = result["tasks"][task][model]
            metric = row["primary_metric"]
            scores = [row["policies"][policy]["accuracy_percent"][metric] for policy in POLICIES]
            pairs = row["paired"][metric]
            paper = row["paper_table3"]
            lines.append(f"| {row['display']} | {FIXED_PARAMETERS['loops_by_model'][model]} | {metric} | "
                         f"{row['policies']['baseline']['n']:,} | {scores[0]:.2f} | {scores[1]:.2f} | {scores[2]:.2f} | "
                         f"{pairs['hidden_vs_baseline']['delta_percentage_points']:+.2f} | "
                         f"{pairs['hidden_vs_logits']['delta_percentage_points']:+.2f} | "
                         f"{paper['baseline']:.2f}→{paper['loopcd_hidden']:.2f} |")
    lines += [
        "", "Paper scores come from [Table 3(a), PDF page 7](https://arxiv.org/html/2610.02185v1#S4.T3), "
        "with references and strengths from [Table A4, page 18](https://arxiv.org/html/2610.02185v1#A2.T4). "
        "These are full-budget settings. Table A8's separate R32→R16 experiment uses h6 at R16; "
        "the present full-budget R16 setting uses h7.",
        "", "Primary metrics were fixed before scoring: acc for SciQ and acc_norm for ARC-Easy, "
        "ARC-Challenge, and PIQA. The paper does not unambiguously specify these metrics for every task. "
        "SciQ prompts include support and retain all "
        "116 empty-support cases. PIQA uses its 1,838 labeled validation examples; the other three tasks "
        "use complete test splits. acc_norm divides by the original answer's character count.", "",
        "SciQ's paper values are around 90%. The next table compares them with both the "
        "pre-specified primary acc and the descriptive acc_norm results.", "",
        "| SciQ depth | Actual acc baseline→Hidden % | Actual acc_norm baseline→Hidden % | Paper baseline→Hidden % |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        row = result["tasks"]["sciq"][model]
        pairs = []
        for metric in ("acc", "acc_norm"):
            baseline = row["policies"]["baseline"]["accuracy_percent"][metric]
            guided = row["policies"]["loopcd_hidden"]["accuracy_percent"][metric]
            pairs.append(f"{baseline:.2f}→{guided:.2f}")
        paper = row["paper_table3"]
        lines.append(f"| {FIXED_PARAMETERS['loops_by_model'][model]} | {pairs[0]} | {pairs[1]} | "
                     f"{paper['baseline']:.2f}→{paper['loopcd_hidden']:.2f} |")
    lines += ["",
        "| Depth | Mean definition | Baseline % | Logits % | Hidden % | Hidden−baseline pp | Hidden−Logits pp |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for view, label in (("frozen_primary_metrics", "Frozen primary metrics"), ("all_acc", "All acc (descriptive)"),
                            ("all_acc_norm", "All acc_norm (descriptive)")):
            row = result["means"][model][view]
            values = [row["accuracy_percent"][policy] for policy in POLICIES]
            lines.append(f"| {FIXED_PARAMETERS['loops_by_model'][model]} | {label} | "
                         f"{values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | "
                         f"{row['contrasts']['hidden_vs_baseline']['delta_percentage_points']:+.2f} | "
                         f"{row['contrasts']['hidden_vs_logits']['delta_percentage_points']:+.2f} |")
    lines += ["", "Each task receives one quarter of the mean regardless of question count. "
              "The following inference uses only the frozen primary metrics.", "",
              "| Depth | Comparison | Mean change pp | Paired 95% CI | Task wins/ties/losses | Sign-flip p | Holm p (4 tests) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for name, (_, _, label) in CONTRASTS.items():
            row = result["means"][model]["frozen_primary_metrics"]["contrasts"][name]
            low, high = row["stratified_paired_bootstrap_95ci_pp"]
            lines.append(f"| {FIXED_PARAMETERS['loops_by_model'][model]} | {label} | {row['delta_percentage_points']:+.2f} | "
                         f"[{low:+.2f}, {high:+.2f}] | {row['wins']}/{row['ties']}/{row['losses']} | "
                         f"{row['paired_signflip_two_sided_p']:.4g} | {row['paired_signflip_holm_across_four_p']:.4g} |")
    lines += ["", "| Task/depth | Comparison | Change pp | Paired 95% CI | Corrected/regressed | Exact McNemar p | Holm p (16 tests) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for model in MODELS:
            setting = result["tasks"][task][model]
            for name, (_, _, label) in CONTRASTS.items():
                row = setting["paired"][PRIMARY_METRICS[task]][name]
                low, high = row["paired_bootstrap_95ci_pp"]
                lines.append(f"| {setting['display']}/R{FIXED_PARAMETERS['loops_by_model'][model]} | {label} | "
                             f"{row['delta_percentage_points']:+.2f} | [{low:+.2f}, {high:+.2f}] | "
                             f"{row['corrected']}/{row['regressed']} | {row['mcnemar_exact_two_sided_p']:.4g} | "
                             f"{row['mcnemar_holm_across_sixteen_p']:.4g} |")
    lines += [
        "", "Paired bootstrap uses 20,000 question resamples, independently within each task for macro means. "
        "Macro tests use 200,000 paired sign-flip samples and plus-one two-sided p-values. "
        "Each method and contrast starts from seed 1234. Separate Holm families contain 16 task tests "
        "and four macro tests. Confidence intervals are pointwise. The four tasks and one initial-state "
        "seed are fixed; uncertainty reflects question sampling within those tasks.", "",
        "| Task/depth | Baseline seconds | Logits seconds | Hidden seconds | Hidden/baseline latency | Hidden/Logits latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for model in MODELS:
            row = result["tasks"][task][model]
            timing = row["costs"]["timing"]
            values = [timing[policy]["aggregate_seconds"]["median"] for policy in POLICIES]
            lines.append(f"| {row['display']}/R{FIXED_PARAMETERS['loops_by_model'][model]} | "
                         f"{values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | "
                         f"{values[2] / values[0]:.4f}× | {values[2] / values[1]:.4f}× |")
    lines += [
        "", "Timing measures each policy independently on the same sampled workload with CUDA synchronization "
        "and repeated measurements. Values are medians of aggregate policy seconds, not full-test end-to-end "
        "latencies. Joint baseline/Hidden scoring time is excluded from speed comparisons. Hidden keeps one "
        "complete coda/head and the baseline's matrix-product FLOPs; storing reference states, sequence-wide "
        "vector arithmetic, and memory traffic can still affect measured latency.", "",
        "Each setting selects three batches of eight candidates, with two warmups and six measurements "
        "per policy per batch. Policy timing excludes checkpoint loading, preparation, transfers, "
        "precomputed initial-state generation, and serialization. Initial-state generation times are "
        "recorded separately.", "",
    ]
    for model in MODELS:
        ratios = result["sampled_latency_ratio_geometric_mean"][model]
        lines.append(f"R{FIXED_PARAMETERS['loops_by_model'][model]} geometric mean of four task latency ratios: "
                     f"Hidden/baseline {ratios['baseline']['loopcd_hidden']:.4f}×; "
                     f"Hidden/Logits {ratios['loopcd_logits']['loopcd_hidden']:.4f}×.")
    lines += ["",
        "| Depth | Baseline TFLOPs | Logits TFLOPs | Hidden TFLOPs | Hidden vector operations |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        flops = result["tasks"][TASKS[0]][model]["costs"]["flops_512_token_prefill"]["including_recurrent_injection"]
        lines.append(f"| {FIXED_PARAMETERS['loops_by_model'][model]} | {flops['baseline']['total_tflops']:.3f} | "
                     f"{flops['loopcd_logits']['total_tflops']:.3f} | {flops['loopcd_hidden']['total_tflops']:.3f} | "
                     f"{flops['loopcd_hidden']['guidance_elementwise_operations']:,} |")
    maximum = max(result["tasks"][task][model]["baseline_replay"]["maximum_token_logprob_difference"]
                  for task in TASKS for model in MODELS)
    lines += [
        "", "Analytic FLOPs count matrix arithmetic for a 512-token prefill, batch size one, and readout "
        "at every position, including recurrent injection. The costs also retain the paper's counting "
        "convention. Vector operations are listed separately.",
        "", f"The maximum old/new baseline answer-token log-probability difference is {maximum:.9g} "
        "(audit tolerance 0.001). Every baseline prediction matches under both metrics. Candidate sums, "
        "character normalization, argmax decisions, and summaries were recomputed from saved token scores. "
        "Audits verify raw data, tokenization, source snapshots, checkpoint revision, run parameters, "
        "and independent costs. Original Logits artifacts remain unchanged; historical ARC-Easy metadata "
        "is aligned using recorded evidence without rewriting its manifest.",
        "", "comparison.json retains both metrics, detailed audits, and source hashes. "
        "comparison.csv contains filterable task results. The frozen protocol is in "
        "[analysis_plan.json](analysis_plan.json).", "",
    ]
    sensitive = []
    for task in TASKS:
        for model in MODELS:
            for name in CONTRASTS:
                paired = result["tasks"][task][model]["paired"]
                if paired["acc"][name]["delta_percentage_points"] * paired["acc_norm"][name]["delta_percentage_points"] < 0:
                    sensitive.append(f"{task}/{model}/{name}")
    if sensitive:
        lines += ["The two metrics disagree on the direction of change for: " + "; ".join(sensitive)
                  + ".", ""]
    return "\n".join(lines)


def render_csv(result):
    stream = io.StringIO()
    fields = ["task", "model", "primary_metric", "n", "metric", "baseline_percent", "logits_percent",
              "hidden_percent", "hidden_vs_baseline_pp", "hidden_vs_logits_pp", "paper_baseline_percent",
              "paper_hidden_percent", "primary_hidden_vs_baseline_holm_p", "primary_hidden_vs_logits_holm_p"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for task in TASKS:
        for model in MODELS:
            row = result["tasks"][task][model]
            for metric in analyze.METRICS:
                primary = metric == PRIMARY_METRICS[task]
                writer.writerow({
                    "task": task, "model": model, "primary_metric": PRIMARY_METRICS[task],
                    "n": row["policies"]["baseline"]["n"], "metric": metric,
                    "baseline_percent": row["policies"]["baseline"]["accuracy_percent"][metric],
                    "logits_percent": row["policies"]["loopcd_logits"]["accuracy_percent"][metric],
                    "hidden_percent": row["policies"]["loopcd_hidden"]["accuracy_percent"][metric],
                    "hidden_vs_baseline_pp": row["paired"][metric]["hidden_vs_baseline"]["delta_percentage_points"],
                    "hidden_vs_logits_pp": row["paired"][metric]["hidden_vs_logits"]["delta_percentage_points"],
                    "paper_baseline_percent": row["paper_table3"]["baseline"] if primary else "",
                    "paper_hidden_percent": row["paper_table3"]["loopcd_hidden"] if primary else "",
                    **{f"primary_{name}_holm_p": row["paired"][metric][name]["mcnemar_holm_across_sixteen_p"] if primary else ""
                       for name in CONTRASTS},
                })
    return stream.getvalue()


def write_artifacts(result, output):
    """No completion marker is emitted until all audited report files exist."""
    analyze.require(result.get("complete") is True, "Cannot publish an incomplete Hidden report")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"REPORT.md": render_report(result), "comparison.csv": render_csv(result),
                 "comparison.json": json.dumps(result, indent=2, ensure_ascii=False) + "\n"}
    for name, content in artifacts.items():
        (output / name).write_text(content)
    files = {
        analyze.relative(path): {"bytes": path.stat().st_size, "sha256": analyze.digest(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"completion.json", "artifact_manifest.json"}
        and "__pycache__" not in path.parts and path.suffix not in {".tmp", ".log", ".out", ".err"}
    }
    artifact_path = output / "artifact_manifest.json"
    artifact_path.write_text(json.dumps({"scope": "Huginn full-depth Hidden: two depths × four tasks",
                                        "files": files}, indent=2) + "\n")
    (output / "completion.json").write_text(json.dumps({
        "complete": True, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "new_evaluation_runs": 8, "reused_logits_runs": 8,
        "questions_per_depth": sum(result["tasks"][task][MODELS[0]]["policies"]["baseline"]["n"] for task in TASKS),
        "new_question_evaluations": sum(result["tasks"][task][model]["policies"]["baseline"]["n"]
                                        for task in TASKS for model in MODELS),
        "new_scored_candidates": sum(result["tasks"][task][model]["audited_runs"]["hidden"]["audit"]["unique_candidates"]
                                     for task in TASKS for model in MODELS),
        "artifact_manifest_sha256": analyze.digest(artifact_path),
        "report": analyze.relative(output / "REPORT.md"),
    }, indent=2) + "\n")


def run(output=None, plan_path=None, *, partial=False):
    output = Path(output) if output is not None else ROOT / "results/hidden"
    if plan_path is None:
        local_plan = output / "analysis_plan.json"
        plan_path = local_plan if local_plan.is_file() else ROOT / "results/hidden/analysis_plan.json"
    else:
        plan_path = Path(plan_path)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    validate_plan(plan)
    _, tasks_registry = load_registry()
    paper_logits = analyze.read_json(ROOT / "configs/paper_mc_reference.json")
    results, cache, progress = {task: {} for task in TASKS}, {}, []
    for task in TASKS:
        for model in MODELS:
            folder = output / task / model
            if folder.exists():
                list(folder.iterdir())  # Refresh NFS metadata before checking summary markers.
            missing = [name for name in analyze.REQUIRED if not (folder / name).is_file()]
            label = f"{task}/{model}"
            if missing:
                progress.append({"setting": label, "state": "evaluation_incomplete", "missing": missing})
                continue
            cost_path = output / task / f"{model}_costs.json"
            cost_ready = cost_path.is_file() and analyze.read_json(cost_path).get("complete") is True
            print(f"Auditing {label}", file=sys.stderr, flush=True)
            row = audit_setting(task, model, output, tasks_registry, paper_logits, cache,
                                include_costs=cost_ready)
            progress.append({"setting": label, "state": "complete" if cost_ready else "accuracy_audited_costs_pending",
                             "accuracy_percent": {policy: row["policies"][policy]["accuracy_percent"] for policy in POLICIES},
                             "baseline_replay": row["baseline_replay"]})
            if cost_ready:
                results[task][model] = row
    ready = all(len(results[task]) == len(MODELS) for task in TASKS)
    if partial:
        status = {"complete": ready, "progress_only": True, "settings": progress}
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return status
    analyze.require(ready, "Eight complete evaluations and standalone cost audits are required; use --partial for progress")
    result = build_suite(results, plan)
    result["analysis_plan"] = plan
    result["analysis_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    result["analysis_plan_file"] = "analysis_plan.json"
    result["analysis_source_sha256"] = {
        analyze.relative(path): analyze.digest(path) for path in (
            Path(__file__), ROOT / "scripts/analyze.py", ROOT / "scripts/analyze_arc_suite.py",
            ROOT / "scripts/analyze_reduced_depth.py", ROOT / "scripts/reduced_suite.py",
            ROOT / "scripts/hidden_cost_audit.py", PAPER_PATH,
        )
    }
    # Repeats inherit the verified frozen plan. Partial mode never creates it,
    # and even full mode waits until every evaluation and cost audit succeeds.
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "analysis_plan.json"
    if not destination.is_file() or destination.read_bytes() != plan_bytes:
        destination.write_bytes(plan_bytes)
    write_artifacts(result, output)
    print(json.dumps({"complete": True, "report": analyze.relative(output / "REPORT.md"),
                      "means": {model: result["means"][model]["frozen_primary_metrics"]["accuracy_percent"]
                                for model in MODELS}}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/hidden")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    run(args.output, args.plan, partial=args.partial)


if __name__ == "__main__":
    main()
