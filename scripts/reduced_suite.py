#!/usr/bin/env python3
"""Aggregate the fixed four-task Huginn half-depth experiment."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze
from scripts.analyze_reduced_depth import POLICIES
from src.registry import load_registry

TASKS = ("arc_easy", "arc_challenge", "sciq", "piqa")
PRIMARY_METRICS = {
    "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "sciq": "acc", "piqa": "acc_norm",
}
CONTRASTS = {
    "reduced_baseline_vs_full": ("baseline_r32", "baseline_r16", "R16 baseline minus R32 baseline"),
    "guided_reduced_vs_full": ("baseline_r32", "loopcd_r16_ref5", "R16 LoopCD minus R32 baseline"),
    "guided_vs_reduced_baseline": ("baseline_r16", "loopcd_r16_ref5", "R16 LoopCD minus R16 baseline"),
}
BOOTSTRAP_SAMPLES = 20000
PERMUTATION_SAMPLES = 200000
SEED = 1234


def holm_adjust(pvalues):
    """Return adjusted p-values for one explicitly supplied hypothesis family."""
    if not pvalues or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("Expected a nonempty family of finite p-values between zero and one")
    adjusted, previous = {}, 0.0
    for rank, (name, pvalue) in enumerate(sorted(pvalues.items(), key=lambda item: (item[1], item[0]))):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalue))
        adjusted[name] = previous
    return {name: adjusted[name] for name in pvalues}


def macro_paired_stats(strata, *, bootstrap_samples=BOOTSTRAP_SAMPLES,
                       permutation_samples=PERMUTATION_SAMPLES, seed=SEED):
    """Equal-weight task differences with within-task paired resampling.

    Each stratum is ``(n, corrected, regressed)``. Multinomial counts are exactly
    equivalent to resampling paired question differences (+1, -1, 0). Under
    paired sign flips, the number of positive discordances is Binomial(d, .5).
    """
    strata = list(strata)
    if not strata or min(bootstrap_samples, permutation_samples) < 1:
        raise ValueError("Nonempty task strata and positive resampling counts are required")
    for n, corrected, regressed in strata:
        if any(type(value) is not int for value in (n, corrected, regressed)) or n < 1:
            raise ValueError("Stratum counts must be integers with positive n")
        if min(corrected, regressed) < 0 or corrected + regressed > n:
            raise ValueError("Invalid correctness discordance counts")
    observed = 100 * math.fsum((corrected - regressed) / n for n, corrected, regressed in strata) / len(strata)
    bootstrap_rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(seed)
    bootstrap = np.zeros(bootstrap_samples, dtype=np.float64)
    permutation = np.zeros(permutation_samples, dtype=np.float64)
    for n, corrected, regressed in strata:
        discordant = corrected + regressed
        counts = bootstrap_rng.multinomial(
            n, [corrected / n, regressed / n, 1 - discordant / n], size=bootstrap_samples,
        )
        bootstrap += (counts[:, 0] - counts[:, 1]) / n
        positives = permutation_rng.binomial(discordant, 0.5, size=permutation_samples)
        permutation += (2 * positives - discordant) / n
    bootstrap *= 100 / len(strata)
    permutation *= 100 / len(strata)
    # Floating arithmetic can differ at tied outcomes after unequal task weights.
    extreme = int(np.count_nonzero(np.abs(permutation) >= abs(observed) - 1e-12))
    return {
        "delta_percentage_points": observed,
        "stratified_paired_bootstrap_95ci_pp": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "paired_signflip_two_sided_p": (extreme + 1) / (permutation_samples + 1),
        "permutation_extreme_samples": extreme,
        "permutation_samples": permutation_samples,
        "bootstrap_samples": bootstrap_samples, "seed": seed,
        "task_discordances": [{"n": n, "corrected": corrected, "regressed": regressed}
                              for n, corrected, regressed in strata],
    }


def validate_plan(plan):
    analyze.require(plan["task_order"] == list(TASKS), "Analysis plan task order changed")
    analyze.require(plan["primary_metrics"] == PRIMARY_METRICS, "Analysis plan primary metrics changed")
    expected = {
        "checkpoint": "tomg-group-umd/huginn-0125", "reduced_loops": 16,
        "full_baseline_loops": 32, "reference_step": 5, "fixed_strength": 0.5,
        "seed": SEED, "dtype": "float32", "matmul_precision": "highest",
    }
    analyze.require(plan["fixed_parameters"] == expected, "Analysis plan fixed parameters changed")
    for name, count in (("bootstrap", BOOTSTRAP_SAMPLES), ("permutation", PERMUTATION_SAMPLES)):
        analyze.require(plan[name]["samples"] == count and plan[name]["seed"] == SEED,
                        f"Analysis plan {name} settings changed")
    for path, digest in plan["configuration_sha256"].items():
        analyze.require(analyze.digest(ROOT / path) == digest, f"Frozen configuration changed: {path}")


def validate_task_results(results):
    analyze.require(set(results) == set(TASKS), "Exactly the four fixed tasks must be complete")
    _, registry = load_registry()
    expected_flops = None
    expected_checkpoint = None
    for task in TASKS:
        result = results[task]
        analyze.require(result.get("complete") is True and result.get("task") == task,
                        f"Incomplete or mismatched task: {task}")
        analyze.require(result["primary_metric"] == PRIMARY_METRICS[task],
                        f"Frozen primary metric changed: {task}")
        analyze.require(set(result["audited_runs"]) == {"full", "original_reduced", "reduced"}
                        and all(run["audit"].get("passed") is True for run in result["audited_runs"].values()),
                        f"Missing successful evaluation audit: {task}")
        analyze.require(result["cost_audit"].get("passed") is True
                        and result["baseline_replay"].get("identical_predictions_both_metrics") is True,
                        f"Missing cost or baseline replay audit: {task}")
        experiment = result["experiment"]
        overrides = experiment["model_overrides"]
        analyze.require(experiment["name"] == "huginn_half_depth"
                        and experiment["model"] == "huginn_16"
                        and experiment["reported_policy"] == "loopcd_fixed"
                        and experiment["full_depth_model"] == "huginn_32"
                        and overrides["reference_step"] == 5 and overrides["fixed"] == 0.5,
                        f"Half-depth configuration changed: {task}")
        manifest = result["audited_runs"]["reduced"]["run_manifest"]
        config, adapter = manifest["configuration"], manifest["adapter_metadata"]
        analyze.require(config["family"] == "huginn" and config["loops"] == 16
                        and config["reference_step"] == 5 and config["fixed"] == 0.5
                        and adapter["loops"] == 16 and adapter["reference_recurrent_update"] == 5,
                        f"Executed half-depth configuration changed: {task}")
        analyze.require(manifest["dtype"] == "float32" and manifest["float32_matmul_precision"] == "highest"
                        and manifest["seed"] == SEED and manifest["shots"] == 0,
                        f"Numerical or prompting protocol changed: {task}")
        checkpoint = (manifest["model_manifest_sha256"], manifest["native_source_hashes"],
                      adapter["model_revision"], adapter["noise_seed"], adapter["noise_key"],
                      adapter["noise_generation"], adapter["noise_coupling"])
        if expected_checkpoint is None:
            expected_checkpoint = checkpoint
        analyze.require(checkpoint == expected_checkpoint, "Checkpoint or initial-state protocol differs across tasks")
        n = registry["tasks"][task]["count"]
        analyze.require(set(result["policies"]) == set(POLICIES), f"Missing policy: {task}")
        for policy, row in result["policies"].items():
            analyze.require(row["n"] == n, f"Incomplete question count: {task}/{policy}")
            for metric in ("acc", "acc_norm"):
                correct = row["correct"][metric]
                analyze.require(type(correct) is int and 0 <= correct <= n, "Invalid correct count")
                analyze.close(row["accuracy_percent"][metric], 100 * correct / n,
                              f"Accuracy differs from count: {task}/{policy}/{metric}")
        for metric in ("acc", "acc_norm"):
            analyze.require(set(result["paired"][metric]) == set(CONTRASTS), "Missing paired comparison")
            for name, (left, right, _) in CONTRASTS.items():
                paired = result["paired"][metric][name]
                corrected, regressed = paired["corrected"], paired["regressed"]
                analyze.require(type(corrected) is int and type(regressed) is int
                                and min(corrected, regressed) >= 0 and corrected + regressed <= n,
                                f"Invalid paired counts: {task}/{name}")
                difference = result["policies"][right]["correct"][metric] - result["policies"][left]["correct"][metric]
                analyze.require(corrected - regressed == difference, "Paired counts disagree with accuracies")
                analyze.close(paired["delta_percentage_points"], 100 * difference / n, "Paired delta mismatch")
        costs = result["costs"]
        analyze.require(costs.get("complete") is True and costs.get("state") == "complete"
                        and costs.get("task") == task, f"Incomplete cost evidence: {task}")
        analyze.require(len(costs["batches"]) == 3
                        and all(batch["candidate_count"] == 8 for batch in costs["batches"])
                        and costs["methodology"]["warmup_per_policy_per_batch"] == 2
                        and costs["methodology"]["repetitions_per_policy_per_batch"] == 6,
                        f"Frozen benchmark sample/repetition counts changed: {task}")
        flops = costs["flops_512_token_prefill"]
        analyze.require(flops["protocol"]["sequence_length"] == 512
                        and flops["protocol"]["batch_size"] == 1
                        and flops["protocol"]["head_positions"] == 512,
                        "Expected the fixed 512-token analytic prefill")
        if expected_flops is None:
            expected_flops = flops
        analyze.require(flops == expected_flops, "Analytic model FLOPs differ across tasks")
        for policy in POLICIES:
            timing = costs["timing"][policy]
            value = timing["aggregate_seconds"]["median"]
            analyze.require(math.isfinite(value) and value > 0, "Missing positive standalone timing")
            analyze.close(timing["speedup_vs_baseline_r32"],
                          costs["timing"]["baseline_r32"]["aggregate_seconds"]["median"] / value,
                          "Timing speedup mismatch")


def descriptive_view(results, metrics):
    means = {
        policy: math.fsum(results[task]["policies"][policy]["accuracy_percent"][metrics[task]]
                          for task in TASKS) / len(TASKS)
        for policy in POLICIES
    }
    contrasts = {}
    for name, (left, right, _) in CONTRASTS.items():
        deltas = {
            task: results[task]["paired"][metrics[task]][name]["delta_percentage_points"]
            for task in TASKS
        }
        contrasts[name] = {
            "delta_percentage_points": means[right] - means[left],
            "task_deltas_pp": deltas,
            "wins": sum(value > 0 for value in deltas.values()),
            "ties": sum(value == 0 for value in deltas.values()),
            "losses": sum(value < 0 for value in deltas.values()),
        }
    return {"metrics_by_task": metrics, "accuracy_percent": means, "contrasts": contrasts}


def build_suite(results, plan):
    """Aggregate already audited task results under the frozen analysis plan."""
    validate_plan(plan)
    validate_task_results(results)
    views = {
        "frozen_primary_metrics": descriptive_view(results, dict(PRIMARY_METRICS)),
        "all_acc": descriptive_view(results, {task: "acc" for task in TASKS}),
        "all_acc_norm": descriptive_view(results, {task: "acc_norm" for task in TASKS}),
    }
    primary = views["frozen_primary_metrics"]
    for name, row in primary["contrasts"].items():
        strata = [
            (results[task]["policies"]["baseline_r32"]["n"],
             results[task]["paired"][PRIMARY_METRICS[task]][name]["corrected"],
             results[task]["paired"][PRIMARY_METRICS[task]][name]["regressed"])
            for task in TASKS
        ]
        inference = macro_paired_stats(strata)
        analyze.close(row["delta_percentage_points"], inference["delta_percentage_points"],
                      "Equal-task mean and paired macro difference disagree")
        row.update(inference)
    macro_holm = holm_adjust({
        name: row["paired_signflip_two_sided_p"] for name, row in primary["contrasts"].items()
    })
    for name, adjusted in macro_holm.items():
        primary["contrasts"][name]["paired_signflip_holm_across_three_p"] = adjusted
    task_holm = holm_adjust({
        f"{task}/{name}": results[task]["paired"][PRIMARY_METRICS[task]][name]["mcnemar_exact_two_sided_p"]
        for task in TASKS for name in CONTRASTS
    })
    task_rows = {}
    for task in TASKS:
        row = results[task]
        primary_paired = copy.deepcopy(row["paired"][PRIMARY_METRICS[task]])
        for name, comparison in primary_paired.items():
            comparison["mcnemar_holm_across_twelve_p"] = task_holm[f"{task}/{name}"]
        task_rows[task] = {
            "display": row.get("display", task), "primary_metric": PRIMARY_METRICS[task],
            "n": row["policies"]["baseline_r32"]["n"], "policies": row["policies"],
            "primary_paired": primary_paired, "timing": row["costs"]["timing"],
            "costs_file": row["costs_file"], "costs_sha256": row["costs_sha256"],
            "baseline_replay": row["baseline_replay"],
        }
    return {
        "complete": True, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "huginn_half_depth", "task_order": list(TASKS),
        "fixed_parameters": copy.deepcopy(plan["fixed_parameters"]),
        "tasks": task_rows, "means": views,
        "flops_512_token_prefill": results[TASKS[0]]["costs"]["flops_512_token_prefill"],
        "sampled_speedup_geometric_mean": {
            policy: math.exp(math.fsum(math.log(results[task]["costs"]["timing"][policy]["speedup_vs_baseline_r32"])
                                      for task in TASKS) / len(TASKS))
            for policy in POLICIES
        },
        "statistical_protocol": {
            "primary_metrics": PRIMARY_METRICS, "task_weight": 0.25,
            "bootstrap": {
                "samples": BOOTSTRAP_SAMPLES, "seed": SEED,
                "method": "Independent within-task paired multinomial resampling of corrected/regressed/unchanged questions; percentile 95% CI for equal-task mean delta.",
            },
            "permutation": {
                "samples": PERMUTATION_SAMPLES, "seed": SEED,
                "method": "Independent paired sign flips conditional on each task's correctness discordances, Binomial(discordant,0.5); statistic 100 times the equal-task mean delta; plus-one two-sided Monte Carlo p.",
                "rng": "Each contrast and resampling method starts a separate NumPy default_rng(1234).",
                "tie_tolerance_pp": 1e-12,
            },
            "multiplicity": {
                "primary_macro": "Holm over the three primary macro sign-flip tests.",
                "primary_tasks": "Holm over the twelve primary exact McNemar tests: four tasks times three contrasts.",
                "individual_reports": "The original within-task Holm adjustment over three contrasts is also retained under mcnemar_holm_adjusted_p.",
            },
            "secondary_views": "All-acc and all-acc_norm means and task differences are descriptive; the frozen primary metrics determine primary inference.",
            "stability": "Wins/ties/losses count task-level accuracy changes, without treating task count as a separate significance test.",
        },
    }


def write_report(result, output):
    primary = result["means"]["frozen_primary_metrics"]
    primary_contrast = primary["contrasts"]["guided_reduced_vs_full"]
    recovery = primary["contrasts"]["guided_vs_reduced_baseline"]
    flops = result["flops_512_token_prefill"]["including_recurrent_injection"]
    lines = [
        "# Huginn half-depth LoopCD across four tasks", "",
        "One checkpoint and one fixed setting across ARC-Easy, ARC-Challenge, SciQ, and PIQA: "
        "Huginn-0125, R=16, reference h5, fixed strength 0.5. The baselines use R=32 and R=16. "
        "All complete task splits use 0-shot prompts, FP32 with highest matrix precision, "
        "and identical per-question initial states across policies.", "",
        "## Complete task results", "",
        "| Task | Primary metric | Questions | R32 baseline (%) | R16 baseline (%) | R16 LoopCD (%) | Change vs R32 (pp) | Change vs R16 (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = result["tasks"][task]
        metric = row["primary_metric"]
        values = [row["policies"][policy]["accuracy_percent"][metric] for policy in POLICIES]
        lines.append(f"| [{row['display']}]({task}/REPORT.md) | {metric} | {row['n']:,} | "
                     f"{values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | "
                     f"{row['primary_paired']['guided_reduced_vs_full']['delta_percentage_points']:+.2f} | "
                     f"{row['primary_paired']['guided_vs_reduced_baseline']['delta_percentage_points']:+.2f} |")
    values = [primary["accuracy_percent"][policy] for policy in POLICIES]
    lines += [
        f"| **Equal-task mean** | Frozen primary metrics | — | **{values[0]:.2f}** | "
        f"**{values[1]:.2f}** | **{values[2]:.2f}** | "
        f"**{primary_contrast['delta_percentage_points']:+.2f}** | **{recovery['delta_percentage_points']:+.2f}** |",
        "", "Each task receives one quarter of the mean, regardless of split size. "
        "Primary metrics remain acc_norm for ARC-Easy, ARC-Challenge, and PIQA, and acc for SciQ. "
        "ARC-Easy, ARC-Challenge, and SciQ use their test splits; PIQA uses its labeled validation split.",
        "", f"Against R32, guided R16 improves on {primary_contrast['wins']} tasks, "
        f"ties on {primary_contrast['ties']}, and declines on {primary_contrast['losses']}. "
        f"Against R16, the corresponding counts are {recovery['wins']}, {recovery['ties']}, "
        f"and {recovery['losses']}. All four task results contribute to the reported mean.",
        "", "## Mean comparisons and uncertainty", "",
        "| Comparison | Equal-task change (pp) | Stratified paired 95% CI | Task wins / ties / losses | Sign-flip p | Holm p (3 tests) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, (_, _, display) in CONTRASTS.items():
        row = primary["contrasts"][name]
        low, high = row["stratified_paired_bootstrap_95ci_pp"]
        lines.append(f"| {display} | {row['delta_percentage_points']:+.2f} | [{low:+.2f}, {high:+.2f}] | "
                     f"{row['wins']} / {row['ties']} / {row['losses']} | {row['paired_signflip_two_sided_p']:.4g} | "
                     f"{row['paired_signflip_holm_across_three_p']:.4g} |")
    lines += [
        "", "Intervals use 20,000 paired question resamples independently within each task. "
        "Two-sided permutation tests use 200,000 sign-flip samples on correctness discordances, "
        "with the same equal-task statistic and the plus-one Monte Carlo correction. "
        "Each contrast and method uses seed 1234. Holm correction covers the three primary mean contrasts. "
        "The tasks are held fixed; uncertainty reflects resampling questions within these tasks.",
        "", "## Secondary metric views", "",
        "| Mean definition | R32 baseline (%) | R16 baseline (%) | R16 LoopCD (%) | Guided change vs R32 (pp) | Guided change vs R16 (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("frozen_primary_metrics", "Frozen primary metrics"),
                       ("all_acc", "All four tasks: acc"), ("all_acc_norm", "All four tasks: acc_norm")):
        view = result["means"][key]
        values = [view["accuracy_percent"][policy] for policy in POLICIES]
        lines.append(f"| {label} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | "
                     f"{view['contrasts']['guided_reduced_vs_full']['delta_percentage_points']:+.2f} | "
                     f"{view['contrasts']['guided_vs_reduced_baseline']['delta_percentage_points']:+.2f} |")
    lines += [
        "", "Both secondary metric views are descriptive. The primary metrics were frozen "
        "before the task extension and determine the primary comparisons.",
        "", "## Per-task primary comparisons", "",
        "| Task | Comparison | Change (pp) | Corrected / regressed | Exact McNemar p | Holm p (12 tests) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for name, (_, _, display) in CONTRASTS.items():
            row = result["tasks"][task]["primary_paired"][name]
            lines.append(f"| {result['tasks'][task]['display']} | {display} | {row['delta_percentage_points']:+.2f} | "
                         f"{row['corrected']} / {row['regressed']} | {row['mcnemar_exact_two_sided_p']:.4g} | "
                         f"{row['mcnemar_holm_across_twelve_p']:.4g} |")
    lines += [
        "", "This table applies one Holm family across all twelve primary task contrasts. "
        "Individual task reports retain their within-task three-contrast adjustment.",
        "", "## Compute and sampled timing", "",
        "| Policy | Forward TFLOPs (512 tokens) | Fraction of R32 |",
        "|---|---:|---:|",
    ]
    for policy, (_, _, display) in POLICIES.items():
        cost = flops[policy]
        lines.append(f"| {display} | {cost['total_tflops']:.3f} | {cost['fraction_of_baseline_r32']:.4f}× |")
    lines += [
        "", "The same analytic cost applies to every task: batch size one, 512-token prefill, "
        "all positions projected, two FLOPs per multiply-accumulate, dense attention products, "
        "and the recurrent injection matrix included. Guided R16 includes its second coda and head. "
        f"It uses {flops['loopcd_r16_ref5']['removed_percent']:.2f}% fewer matrix-product FLOPs than R32. "
        "The separate paper-convention ledger is retained in the JSON.", "",
        "| Task | R32 median (ms) | R16 median (ms) | Guided R16 median (ms) | R16 speedup | Guided R16 speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        timing = result["tasks"][task]["timing"]
        times = [1000 * timing[policy]["aggregate_seconds"]["median"] for policy in POLICIES]
        lines.append(f"| {result['tasks'][task]['display']} | {times[0]:.2f} | {times[1]:.2f} | {times[2]:.2f} | "
                     f"{timing['baseline_r16']['speedup_vs_baseline_r32']:.2f}× | "
                     f"{timing['loopcd_r16_ref5']['speedup_vs_baseline_r32']:.2f}× |")
    speedup = result["sampled_speedup_geometric_mean"]
    lines += [
        f"| **Geometric mean speedup** | — | — | — | **{speedup['baseline_r16']:.2f}×** | "
        f"**{speedup['loopcd_r16_ref5']:.2f}×** |",
        "", "Each task measures three fixed length-quantile batches of eight answer candidates. "
        "The three policies run independently on identical inputs within the same allocated H200 job, "
        "with two warmups and six synchronized repetitions per batch and policy. Reported latency "
        "is the median summed time for the three batches. Per-task JSON files include dispersion, "
        "memory measurements, and individual samples. Timings include forward scoring, guidance, "
        "and target log probabilities; they exclude model loading, tokenization, input preparation, "
        "and CPU random-state generation/transfer. These sampled timings use answer-position "
        "projections and task-specific lengths, distinct from the analytic 512-token prefill.",
        "", "## Reproduction evidence", "",
        "Every task report independently audits saved candidate likelihoods, predictions, paired "
        "document identities, source snapshots, model/data provenance, reference h5, and standalone "
        "timing. The new R16 baselines retain the original predictions. The existing ARC-Challenge "
        "evaluation and timing records are reused.",
        "", "[Frozen analysis plan](analysis_plan.json) · [Comparison JSON](comparison.json) · "
        "[Comparison CSV](comparison.csv)", "",
        "Rebuild all audited reports with `venv/bin/python -B scripts/analyze_reduced_depth.py --task all`.", "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text("\n".join(lines))
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["scope", "task_or_mean", "metric", "policy", "n", "correct", "accuracy_percent",
                     "prefill_tflops", "relative_flops", "sampled_median_ms", "sampled_speedup"])
    for task in TASKS:
        row = result["tasks"][task]
        for metric in ("acc", "acc_norm"):
            for policy in POLICIES:
                writer.writerow(["task", task, metric, policy, row["n"], row["policies"][policy]["correct"][metric],
                                 row["policies"][policy]["accuracy_percent"][metric], flops[policy]["total_tflops"],
                                 flops[policy]["fraction_of_baseline_r32"],
                                 1000 * row["timing"][policy]["aggregate_seconds"]["median"],
                                 row["timing"][policy]["speedup_vs_baseline_r32"]])
    for name, view in result["means"].items():
        for policy in POLICIES:
            writer.writerow(["equal_task_mean", name, name, policy, "", "", view["accuracy_percent"][policy],
                             flops[policy]["total_tflops"], flops[policy]["fraction_of_baseline_r32"], "",
                             speedup[policy]])
    (output / "comparison.csv").write_text(stream.getvalue())


def seal_suite(result, output):
    """Seal the suite and its already sealed task reports after a complete audit."""
    analyze.require(result.get("complete") is True and set(result["tasks"]) == set(TASKS),
                    "Only a complete four-task suite can be sealed")
    artifact_path, completion_path = output / "artifact_manifest.json", output / "completion.json"
    inventory = {
        analyze.relative(path): {"bytes": path.stat().st_size, "sha256": analyze.digest(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path not in (artifact_path, completion_path)
        and "__pycache__" not in path.parts and path.suffix not in {".tmp", ".log", ".out", ".err", ".pyc"}
    }
    artifact_path.write_text(json.dumps({
        "scope": "Four-task fixed-parameter Huginn half-depth evaluation, paired accuracy, FLOPs, and standalone timings",
        "files": inventory,
    }, indent=2) + "\n")
    completion_path.write_text(json.dumps({
        "complete": True, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": list(TASKS), "reduced_evaluation_runs": len(TASKS),
        "audited_historical_baseline_runs": 2 * len(TASKS),
        "reduced_question_evaluations": sum(result["tasks"][task]["n"] for task in TASKS),
        "reduced_scored_candidates": sum(
            analyze.read_json(output / task / "completion.json")["new_candidate_scores"] for task in TASKS
        ),
        "analysis_plan_sha256": result["analysis_plan"]["sha256"],
        "artifact_manifest_sha256": analyze.digest(artifact_path),
        "report": analyze.relative(output / "REPORT.md"),
    }, indent=2) + "\n")


def run_suite(output=None):
    from scripts.analyze_reduced_depth import build_result, seal_result, write_report as write_task_report

    output = (Path(output) if output is not None else ROOT / "results/reduced_depth").resolve()
    plan_path = ROOT / "results/reduced_depth/analysis_plan.json"
    plan = analyze.read_json(plan_path)
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=True)
    output_plan = output / "analysis_plan.json"
    if output_plan != plan_path.resolve():
        if output_plan.exists():
            analyze.require(analyze.digest(output_plan) == analyze.digest(plan_path),
                            "Repeat directory contains a different analysis plan")
        else:
            output_plan.write_bytes(plan_path.read_bytes())
        plan_path = output_plan
    results = {}
    for task in TASKS:
        task_output = output / task
        results[task] = build_result(task, output=task_output)
    result = build_suite(results, plan)
    for task, row in results.items():
        task_output = output / task
        write_task_report(row, task_output)
        seal_result(row, task_output)
    result["analysis_plan"] = {"path": analyze.relative(plan_path), "sha256": analyze.digest(plan_path),
                               "created_utc": plan["created_utc"]}
    result["task_comparison_sha256"] = {
        task: analyze.digest(output / task / "comparison.json") for task in TASKS
    }
    result["analysis_source_sha256"] = {
        analyze.relative(path): analyze.digest(path) for path in (
            Path(__file__), ROOT / "scripts/analyze_reduced_depth.py", ROOT / "scripts/analyze.py",
            ROOT / "scripts/analyze_arc_suite.py", ROOT / "scripts/reduced_cost_audit.py",
        )
    }
    result["numpy_version"] = np.__version__
    write_report(result, output)
    seal_suite(result, output)
    print(json.dumps({"complete": True, "report": analyze.relative(output / "REPORT.md"),
                      "primary_means": result["means"]["frozen_primary_metrics"]["accuracy_percent"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory containing the four task subdirectories")
    args = parser.parse_args()
    run_suite(args.output)


if __name__ == "__main__":
    main()
