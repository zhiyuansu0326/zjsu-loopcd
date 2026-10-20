#!/usr/bin/env python3
"""Audit Huginn half-depth accuracy and independent compute measurements."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze
from scripts.reduced_cost_audit import audit_costs
from src.registry import load_registry, resolve_run

EXPERIMENT = "huginn_half_depth"
POLICIES = {
    "baseline_r32": ("full", "baseline", "Baseline R=32"),
    "baseline_r16": ("reduced", "baseline", "Baseline R=16"),
    "loopcd_r16_ref5": ("reduced", "loopcd_fixed", "LoopCD R=16, reference=5"),
}


def paired_comparison(first, first_mode, second, second_mode, metric):
    """Pair by identity and gold answer, then use the shared statistical protocol."""
    analyze.require(len(first) == len(second) and len(first) > 0, "Unequal paired sample counts")
    rows = []
    for base, guided in zip(first, second):
        analyze.require(base["id"] == guided["id"] and base["gold"] == guided["gold"],
                        "Paired runs differ in document order or labels")
        rows.append({
            "gold": base["gold"],
            "predictions": {metric: {
                "baseline": base["predictions"][metric][first_mode],
                "loopcd_fixed": guided["predictions"][metric][second_mode],
            }},
        })
    return analyze.legacy.paired_stats(rows, "loopcd_fixed", metric)


def verify_reused_baseline(previous, current):
    """Compare every baseline answer token, not just the final accuracy."""
    old = {
        (row["question_index"], row["choice_index"]): row
        for row in analyze.read_jsonl(previous / "candidate_scores.jsonl")
    }
    new = {
        (row["question_index"], row["choice_index"]): row
        for row in analyze.read_jsonl(current / "candidate_scores.jsonl")
    }
    analyze.require(old.keys() == new.keys(), "Baseline candidate sets differ")
    maximum = 0.0
    total = 0
    for key, row in old.items():
        other = new[key]
        analyze.require(row["target_ids"] == other["target_ids"]
                        and row["input_length"] == other["input_length"],
                        f"Baseline input/target mismatch at {key}")
        left, right = row["token_logprobs"]["baseline"], other["token_logprobs"]["baseline"]
        analyze.require(len(left) == len(right), f"Baseline token count mismatch at {key}")
        maximum = max(maximum, max(abs(a - b) for a, b in zip(left, right)))
        total += len(left)
    analyze.require(maximum <= 1e-3, f"New R16 baseline differs by {maximum}")
    old_predictions = analyze.read_jsonl(previous / "predictions.jsonl")
    new_predictions = analyze.read_jsonl(current / "predictions.jsonl")
    analyze.require(len(old_predictions) == len(new_predictions), "Baseline prediction count changed")
    for left, right in zip(old_predictions, new_predictions):
        analyze.require(left["id"] == right["id"] and left["gold"] == right["gold"],
                        "Baseline document identity changed")
        for metric in analyze.METRICS:
            analyze.require(left["predictions"][metric]["baseline"]
                            == right["predictions"][metric]["baseline"],
                            f"Baseline prediction changed at {left['id']}/{metric}")
    return {
        "candidates": len(old), "answer_tokens": total,
        "maximum_token_logprob_difference": maximum, "tolerance": 1e-3,
        "identical_predictions_both_metrics": True,
        "original_path": analyze.relative(previous),
    }


def audit_historical_arc(name, resolved, cache):
    """Audit frozen ARC-Easy scores while retaining their original manifest."""
    registry = analyze.read_json(ROOT / "configs/arc_all_models.json")
    config = registry["models"][name]
    analyze.require(registry["seed"] == resolved["seed"]
                    and registry["dataset_size"] == resolved["task"]["count"]
                    and config["shots"] == resolved["shots"], "Historical ARC-Easy task changed")
    analyze.require(
        {key: value for key, value in config.items() if key not in {"shots", "paper", "reuse"}}
        == {key: value for key, value in resolved["model"].items() if key != "matmul_precision"},
        "Historical ARC-Easy model configuration changed",
    )
    result = analyze.legacy.audit_model(name, config, cache)
    checkpoint = analyze.audit_checkpoint(resolved["model"], cache)
    _, data = analyze.audit_data("arc_easy", resolved, cache)
    manifest = result["run_manifest"]
    analyze.require(set(manifest["code_hashes"]) == {
        "scripts/evaluate_arc_suite.py", "src/suite_scoring.py", "src/loopcd.py", "src/adapter_huginn.py",
    }, "Historical Huginn source inventory differs")
    analyze.require(manifest["data_sha256"] == data["data_sha256"]
                    and manifest["model_manifest_sha256"] == checkpoint["manifest_sha256"]
                    and manifest["adapter_metadata"]["model_revision"] == checkpoint["revision"],
                    "Historical ARC-Easy data/checkpoint identity changed")
    result["checkpoint"] = checkpoint
    result["audit"]["data"] = data
    result["comparable_manifest"] = {
        **manifest,
        "shots": config["shots"], "protocol_sha256": data["protocol_sha256"],
        "native_source_hashes": checkpoint["native_source_hashes"],
    }
    result["historical_schema_alignment"] = {
        "original_manifest_preserved": True,
        "shots": "Read from the original frozen configuration.",
        "protocol_sha256": "Derived from the verified protocol copy and identical prepared data; not recorded by the original runner.",
        "native_source_hashes": "Resolved from the verified pinned checkpoint whose manifest SHA matches the original run; not recorded by the original runner.",
    }
    resolved["output"] = ROOT / result["results_path"]
    return result


def build_result(task, costs_path=None, output=None):
    _, tasks = load_registry()
    cache = {}
    paper = analyze.read_json(ROOT / "configs/paper_mc_reference.json")
    resolved = {
        "full": resolve_run(task, "huginn_32"),
        "original_reduced": resolve_run(task, "huginn_16"),
        "reduced": resolve_run(task, "huginn_16", EXPERIMENT),
    }
    output = Path(output) if output is not None else ROOT / "results/reduced_depth" / task
    resolved["reduced"]["output"] = output / "huginn_16_ref5"
    audited = {}
    for key, run in resolved.items():
        name = "huginn_32" if key == "full" else "huginn_16"
        print(f"Auditing {task}/{key}", file=sys.stderr, flush=True)
        if task == "arc_easy" and key != "reduced":
            audited[key] = audit_historical_arc(name, run, cache)
        else:
            audited[key] = analyze.audit_model(task, name, run, tasks, paper, cache)
    new_manifest = audited["reduced"]["run_manifest"]
    shared_fields = (
        "data_sha256", "protocol_sha256", "model_manifest_sha256", "native_source_hashes",
        "dtype", "float32_matmul_precision", "seed", "shots", "add_bos", "chat_template",
        "batch_size", "token_budget", "torch", "transformers", "gpu",
    )
    comparable = [row.get("comparable_manifest", row["run_manifest"]) for row in audited.values()]
    for field in shared_fields:
        analyze.require(all(manifest[field] == new_manifest[field] for manifest in comparable),
                        f"Unmatched setting across depths: {field}")
    noise_keys = {key for key in new_manifest["adapter_metadata"] if key.startswith("noise_")}
    analyze.require(all({key for key in row["run_manifest"]["adapter_metadata"] if key.startswith("noise_")}
                        == noise_keys for row in audited.values()), "Random initial-state metadata changed")
    for key in sorted(noise_keys):
        analyze.require(all(row["run_manifest"]["adapter_metadata"][key]
                            == new_manifest["adapter_metadata"][key] for row in audited.values()),
                        f"Random initial-state protocol changed: {key}")
    identity = verify_reused_baseline(resolved["original_reduced"]["output"],
                                      resolved["reduced"]["output"])
    predictions = {
        key: analyze.read_jsonl(run["output"] / "predictions.jsonl")
        for key, run in resolved.items()
    }
    results = {}
    for policy, (key, mode, display) in POLICIES.items():
        row = audited[key]
        results[policy] = {
            "display": display, "n": row["n"],
            "correct": {metric: row["correct"][metric][mode] for metric in analyze.METRICS},
            "accuracy_percent": {
                metric: row["accuracy_percent"][metric][mode] for metric in analyze.METRICS
            },
            "results_path": row["results_path"], "policy": mode,
        }
    contrasts = {
        "reduced_baseline_vs_full": ("full", "baseline", "reduced", "baseline"),
        "guided_reduced_vs_full": ("full", "baseline", "reduced", "loopcd_fixed"),
        "guided_vs_reduced_baseline": ("reduced", "baseline", "reduced", "loopcd_fixed"),
    }
    paired = {
        metric: {
            label: paired_comparison(predictions[a], amode, predictions[b], bmode, metric)
            for label, (a, amode, b, bmode) in contrasts.items()
        }
        for metric in analyze.METRICS
    }
    for metric, comparisons in paired.items():
        previous = 0.0
        ordered = sorted(comparisons.values(), key=lambda row: row["mcnemar_exact_two_sided_p"])
        for rank, row in enumerate(ordered):
            previous = max(previous, min(1.0, (len(ordered) - rank) * row["mcnemar_exact_two_sided_p"]))
            row["mcnemar_holm_adjusted_p"] = previous
    costs_path = costs_path or output / "huginn_costs.json"
    cost_audit = audit_costs(costs_path, new_manifest)
    costs = analyze.read_json(costs_path)
    analyze.require(costs.get("task") == task, "Benchmark task mismatch")
    for policy in POLICIES:
        analyze.require(costs["timing"][policy]["aggregate_seconds"]["median"] > 0,
                        "Missing positive standalone timing")
    task_config = resolved["reduced"]["task"]
    return {
        "complete": True, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task, "primary_metric": resolved["reduced"]["task"]["primary_metric"],
        "experiment": resolved["reduced"]["experiment"],
        "display": task_config["display"], "task_configuration": task_config,
        "scope": f"Huginn-0125 on the complete {task_config['display']} {task_config['split']} split, {resolved['reduced']['shots']}-shot.",
        "policies": results, "paired": paired, "baseline_replay": identity,
        "costs_file": analyze.relative(costs_path), "costs_sha256": analyze.digest(costs_path),
        "costs": costs, "cost_audit": cost_audit, "audited_runs": audited,
        "statistical_protocol": {
            "bootstrap": "20000 paired question resamples, seed 1234, percentile 95% CI",
            "mcnemar": "Exact two-sided; Holm correction across three contrasts separately per metric",
            "primary_comparison": f"R16 fixed LoopCD reference5 against R32 baseline, {task_config['primary_metric']}",
            "paper_comparison": "Table A8 configuration; paper Figure 4 reports a seven-task mean",
            "randomness": "One deterministic per-question initial-state seed, shared across depths",
        },
        "analysis_source_sha256": {
            analyze.relative(path): analyze.digest(path)
            for path in (Path(__file__), ROOT / "scripts/analyze.py",
                         ROOT / "scripts/analyze_arc_suite.py", ROOT / "scripts/reduced_cost_audit.py")
        },
    }


def write_report(result, output):
    metric = result["primary_metric"]
    task = result["task"]
    task_config = result["task_configuration"]
    display = task_config["display"]
    n = result["policies"]["baseline_r32"]["n"]
    candidate_count = result["audited_runs"]["reduced"]["audit"]["unique_candidates"]
    shots = result["audited_runs"]["reduced"]["run_manifest"]["shots"]
    secondary_metric = "acc" if metric == "acc_norm" else "acc_norm"
    costs = result["costs"]
    flops = costs["flops_512_token_prefill"]["including_recurrent_injection"]
    paper_flops = costs["flops_512_token_prefill"]["matching_paper_table_convention"]
    lines = [
        f"# Huginn × {display}: half-depth LoopCD", "",
        f"Complete {task_config['split']} split: {n:,} questions, {candidate_count:,} answer candidates, {shots}-shot. "
        "The checkpoint, prompts, FP32 precision, and per-question initial states are shared. "
        "The reduced-depth setting follows Table A8: R=16, reference h5, fixed strength 0.5. "
        f"The primary metric is {metric}.", "",
        "## Accuracy and compute", "",
        f"| Setting | Correct / {n:,} ({metric}) | {metric} (%) | {secondary_metric} (%) | Forward TFLOPs | Relative FLOPs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, row in result["policies"].items():
        cost = flops[policy]
        lines.append(
            f"| {row['display']} | {row['correct'][metric]} | "
            f"{row['accuracy_percent'][metric]:.2f} | {row['accuracy_percent'][secondary_metric]:.2f} | "
            f"{cost['total_tflops']:.3f} | {cost['fraction_of_baseline_r32']:.3f}× |"
        )
    primary = result["paired"][metric]["guided_reduced_vs_full"]
    recovery = result["paired"][metric]["guided_vs_reduced_baseline"]
    lines += [
        "",
        f"Half-depth LoopCD changes accuracy by **{primary['delta_percentage_points']:+.2f} pp** "
        f"against the full-depth baseline and **{recovery['delta_percentage_points']:+.2f} pp** "
        f"against the half-depth baseline. Its analytic matrix-product FLOPs are "
        f"**{flops['loopcd_r16_ref5']['removed_percent']:.2f}% lower** than the full-depth baseline.",
        "",
        "FLOPs use a batch of one 512-token prefill with all positions projected, two operations "
        "per multiply-accumulate, and dense attention products. Counts include the recurrent "
        "injection projection and the extra coda/head for logit guidance. Embedding lookup and "
        "elementwise operations are excluded.",
        "",
        "The paper's printed Huginn values are reproduced by omitting the recurrent injection "
        "projection from this ledger: "
        f"{paper_flops['baseline_r32']['total_tflops']:.3f} TFLOPs for R=32, "
        f"{paper_flops['baseline_r16']['total_tflops']:.3f} for R=16, and "
        f"{paper_flops['loopcd_r16_ref5']['fraction_of_baseline_r32']:.4f}× for guided R=16. "
        "Both ledgers are saved in the cost JSON.",
        "",
        "## Paired comparisons", "",
        f"| Comparison ({metric}) | Change (pp) | 95% paired CI | Corrected / regressed | Exact p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "reduced_baseline_vs_full": "R16 baseline − R32 baseline",
        "guided_reduced_vs_full": "R16 LoopCD − R32 baseline",
        "guided_vs_reduced_baseline": "R16 LoopCD − R16 baseline",
    }
    for key, row in result["paired"][metric].items():
        low, high = row["paired_bootstrap_95ci_pp"]
        lines.append(f"| {labels[key]} | {row['delta_percentage_points']:+.2f} | "
                     f"[{low:+.2f}, {high:+.2f}] | {row['corrected']} / {row['regressed']} | "
                     f"{row['mcnemar_exact_two_sided_p']:.4g} | {row['mcnemar_holm_adjusted_p']:.4g} |")
    lines += [
        "", "Intervals use 20,000 paired question resamples with seed 1234. "
        "Holm correction covers these three contrasts separately for each metric. "
        "This experiment tests one task; the paper's Figure 4 aggregates seven tasks.",
        "", "## Standalone timing", "",
        "| Setting | Median time for sampled batches (ms) | IQR (ms) | Speedup vs R32 |",
        "|---|---:|---:|---:|",
    ]
    for policy, row in result["policies"].items():
        timing = costs["timing"][policy]
        stats = timing["aggregate_seconds"]
        lines.append(f"| {row['display']} | {1000*stats['median']:.2f} | "
                     f"{1000*stats['iqr']:.2f} | {timing['speedup_vs_baseline_r32']:.2f}× |")
    lines += [
        "", "Policies run independently on the same allocated H200 and identical sampled "
        f"{display} batches. Baselines perform one readout; guided scoring performs two. "
        "Timing uses warmup, CUDA synchronization, and repeated measurements. Model loading, "
        "tokenization, and initial-state construction are excluded; the same prepared initial "
        "states are supplied to all policies. These are sampled scoring measurements; "
        "the analytic 512-token prefill uses a different input shape.",
        "", "## Verification", "",
        "Saved candidate tokens, likelihood sums, predictions, summaries, source snapshots, "
        "checkpoint metadata, and data provenance pass independent audits. "
        "The h5 reference and final output match native Huginn calls with identical initial states. "
        f"The new R16 baseline has identical predictions to the prior run; maximum per-token "
        f"log-probability difference: {result['baseline_replay']['maximum_token_logprob_difference']:.3g}.",
        "", "[Comparison JSON](comparison.json) · [CSV](comparison.csv) · "
        "[Cost and timing evidence](huginn_costs.json)", "",
        f"Reproduce with `make submit-reduced TASK={task} "
        f"REDUCED_OUTPUT=results/repeats/reduced_depth/{task}`; "
        f"rebuild this report with `make report-reduced TASK={task}`.", "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text("\n".join(lines))
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["policy", "n", "primary_metric", "correct_primary", "primary_accuracy_percent",
                     "correct_acc_norm", "correct_acc", "acc_norm_percent", "acc_percent",
                     "prefill_tflops", "relative_flops", "sampled_median_ms", "sampled_speedup"])
    for policy, row in result["policies"].items():
        writer.writerow([policy, row["n"], metric, row["correct"][metric], row["accuracy_percent"][metric],
                         row["correct"]["acc_norm"], row["correct"]["acc"],
                         row["accuracy_percent"]["acc_norm"], row["accuracy_percent"]["acc"],
                         flops[policy]["total_tflops"], flops[policy]["fraction_of_baseline_r32"],
                         costs["timing"][policy]["aggregate_seconds"]["median"] * 1000,
                         costs["timing"][policy]["speedup_vs_baseline_r32"]])
    (output / "comparison.csv").write_text(stream.getvalue())


def seal_result(result, output):
    """Write completion and file hashes after a per-task report has been rebuilt."""
    output = Path(output)
    inventory = {
        analyze.relative(path): {"bytes": path.stat().st_size, "sha256": analyze.digest(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"artifact_manifest.json", "completion.json"}
        and "__pycache__" not in path.parts and path.suffix not in {".tmp", ".log", ".out", ".err"}
    }
    artifact_path = output / "artifact_manifest.json"
    artifact_path.write_text(json.dumps({
        "scope": f"Huginn {result['display']} half-depth evaluation, standalone timing, and paired comparisons",
        "files": inventory,
    }, indent=2) + "\n")
    (output / "completion.json").write_text(json.dumps({
        "complete": True, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "new_evaluation_runs": 1, "reused_baseline_runs": 2,
        "questions_per_run": result["policies"]["baseline_r32"]["n"],
        "new_candidate_scores": result["audited_runs"]["reduced"]["audit"]["unique_candidates"],
        "artifact_manifest_sha256": analyze.digest(artifact_path),
        "report": analyze.relative(output / "REPORT.md"),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *analyze.TASK_NAMES], default="arc_challenge")
    parser.add_argument("--output", type=Path, help="Comparison directory, or four-task root for --task all")
    args = parser.parse_args()
    if args.task == "all":
        from scripts.reduced_suite import run_suite
        run_suite(output=args.output)
        return
    output = args.output or ROOT / "results/reduced_depth" / args.task
    result = build_result(args.task, output=output)
    write_report(result, output)
    seal_result(result, output)
    print(json.dumps({"complete": True, "report": analyze.relative(output / "REPORT.md"),
                      "policies": result["policies"]}), flush=True)


if __name__ == "__main__":
    main()
