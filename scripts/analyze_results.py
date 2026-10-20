#!/usr/bin/env python3
"""Paired analysis of the fixed experiment; no parameter selection."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = {"baseline": 84.09, "loopcd_fixed": 84.39, "loopcd_adaptive": 84.26}
NAMES = {
    "baseline": "Baseline (loop 4)",
    "loopcd_fixed": "LoopCD fixed strength ω=0.5",
    "loopcd_adaptive": "LoopCD adaptive cap=1.0",
    "reference": "Weak reference (loop 1)",
}


def paired_stats(rows, mode, metric):
    base = np.array([r["predictions"][metric]["baseline"] == r["gold"] for r in rows])
    guided = np.array([r["predictions"][metric][mode] == r["gold"] for r in rows])
    improved = int((~base & guided).sum())
    regressed = int((base & ~guided).sum())
    changed = sum(
        r["predictions"][metric]["baseline"] != r["predictions"][metric][mode] for r in rows
    )
    discordant = improved + regressed
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2**discordant)
    rng = np.random.default_rng(1234)
    counts = rng.multinomial(
        len(rows), [improved / len(rows), regressed / len(rows), 1 - discordant / len(rows)],
        size=20000,
    )
    deltas = 100 * (counts[:, 0] - counts[:, 1]) / len(rows)
    return {
        "corrected": improved, "regressed": regressed, "changed_answers": changed,
        "net_correct": improved - regressed,
        "delta_percentage_points": 100 * (improved - regressed) / len(rows),
        "paired_bootstrap_95ci_pp": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "mcnemar_exact_two_sided_p": p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/arc_easy")
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    manifest = json.loads((args.results / "run_manifest.json").read_text())
    validation = json.loads((args.results / "validation.json").read_text())
    assert validation["zero_guidance_matches_native_exactly"]
    rows = [json.loads(line) for line in (args.results / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == summary["n"]
    assert len({row["id"] for row in rows}) == len(rows)
    paired = {
        metric: {mode: paired_stats(rows, mode, metric) for mode in ("loopcd_fixed", "loopcd_adaptive")}
        for metric in ("acc", "acc_norm")
    }
    result = {
        "n": len(rows), "paper_table2_arc_easy": PAPER,
        "paired": paired,
        "metric_note": "acc_norm is per-character normalized candidate log likelihood; acc is unnormalized. Both are retained because the paper does not explicitly name the ARC-E headline metric.",
    }
    (args.results / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    examples = []
    for mode in ("loopcd_fixed", "loopcd_adaptive"):
        for category in ("corrected", "regressed"):
            selected = []
            for row in rows:
                baseline_correct = row["predictions"]["acc_norm"]["baseline"] == row["gold"]
                guided_correct = row["predictions"]["acc_norm"][mode] == row["gold"]
                qualifies = (not baseline_correct and guided_correct) if category == "corrected" else (baseline_correct and not guided_correct)
                if qualifies:
                    selected.append({"policy": mode, "category": category, **row})
                if len(selected) == 2:
                    break
            examples.extend(selected)
    (args.results / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Ouro-1.4B × ARC-Easy: Functional reproduction of LoopCD",
        "",
        f"{'Full test set' if len(rows) == 2376 else 'Test subset'}: {len(rows)} questions. The sole model is ByteDance/Ouro-1.4B base, with 8-shot examples from the training set, seed=1234, and four complete loops.",
        "Fixed ω=0.5 and adaptive cap=1.0 follow Table A4 of the paper. All policies share the same questions, examples, weights, and loop trajectories.",
        "",
        "## Reproduction and paper results",
        "",
        "The primary metric is acc_norm: conditional answer log likelihood divided by the answer text's character count. "
        "The table also reports unnormalized acc.",
        "",
        "| Policy | Correct | Reproduced acc_norm | Paper Table 2 | Reproduced acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in NAMES:
        lines.append(
            f"| {NAMES[mode]} | {summary['correct']['acc_norm'][mode]}/{len(rows)} | "
            f"{summary['accuracy_percent']['acc_norm'][mode]:.2f}% | "
            f"{str(PAPER[mode]) + '%' if mode in PAPER else 'Not compared'} | "
            f"{summary['accuracy_percent']['acc'][mode]:.2f}% |"
        )
    lines.extend([
        "",
        "## Paired changes on the same questions",
        "",
        "| Policy | Corrected | Regressed | Net correct | Change (pp) | Paired bootstrap 95% CI | McNemar p |",
        "|---|---:|---:|---:|---:|---|---:|",
    ])
    for mode, values in paired["acc_norm"].items():
        low, high = values["paired_bootstrap_95ci_pp"]
        lines.append(
            f"| {NAMES[mode]} | {values['corrected']} | {values['regressed']} | "
            f"{values['net_correct']:+d} | {values['delta_percentage_points']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | {values['mcnemar_exact_two_sided_p']:.4f} |"
        )
    lines.extend([
        "",
        "Intervals use 20,000 paired question-level bootstrap resamples with seed=1234.",
        "",
    ])
    if all(values["paired_bootstrap_95ci_pp"][0] <= 0 <= values["paired_bootstrap_95ci_pp"][1]
           for values in paired["acc_norm"].values()):
        lines.extend([
            "Both LoopCD acc_norm improvement intervals include zero.",
            "",
        ])
    lines.extend([
        "## Implementation and validation",
        "",
        "Official Ouro applies RMSNorm after each complete 24-layer loop before continuing to the next loop. "
        "This implementation reads the first and fourth loop states returned by the official model and applies the shared lm_head to obtain z₁ and z₄. "
        "Scores use z′=z₄+ω(z₄−z₁). Adaptive ω=1−(p₄,top1−p₄,top2), with probabilities computed over the full vocabulary.",
        f"Model computation uses {manifest['dtype']} (FP32 matmul precision={manifest.get('float32_matmul_precision', 'default')}), "
        "with FP32 logit arithmetic and log_softmax. Every token of each candidate answer receives teacher-forced conditional likelihood scoring with the full context.",
        "",
        "Checks verify that ω=0 matches official fourth-loop logits, early and final loop outputs differ, and single-example and padded-batch inputs agree. "
        "A tiny official Ouro model also has tests covering every answer position. See validation.json for full-model checks and ../../tests/test_loopcd.py for test source.",
        "",
        "During numerical preflight, BF16 had a maximum single-example/batch answer-token log-probability difference of 0.28756 and was excluded from formal results. "
        "The formal configuration had zero difference on the same check. FP32 matmul precision=high permits faster internal matrix multiplication "
        "and differs from highest, which requests the highest precision.",
        "",
        "All three policies and the weak reference share one four-loop computation. Recorded elapsed time covers this joint evaluation.",
        "",
        f"- Model loading and validation: {summary['load_and_validation_seconds']:.1f} seconds.",
        f"- Scoring in this invocation: {summary['this_invocation_scoring_seconds']:.1f} seconds.",
        f"- Peak allocated GPU memory: {summary['peak_gpu_allocated_gib']:.2f} GiB.",
        "",
        "## Illustrative examples",
        "",
        "For each policy and each category, the examples below are the first two corrected or regressed cases in the original test-set order. Complete per-question results are saved in predictions.jsonl.",
    ])
    for ex in examples:
        lines.extend([
            "",
            f"### {ex['id']} · {NAMES[ex['policy']]} · {'Corrected' if ex['category'] == 'corrected' else 'Regressed'}",
            "",
            ex["question"],
            "",
        ])
        for label, choice in zip(ex["labels"], ex["choices"]):
            lines.append(f"- {label}. {choice}")
        baseline = ex["predictions"]["acc_norm"]["baseline"]
        guided = ex["predictions"]["acc_norm"][ex["policy"]]
        lines.extend([
            "",
            f"Correct answer: {ex['labels'][ex['gold']]}; baseline: {ex['labels'][baseline]}; "
            f"LoopCD: {ex['labels'][guided]}.",
        ])
    lines.extend([
        "",
        "## Reproduction settings",
        "",
        "The experiment uses a pinned official model revision, lm-eval prompt construction and sampling rules, seed=1234, and the guidance strengths listed above.",
        "",
        "Sources: Table 2 on page 6, Table A2 on page 17, and Table A4 on page 18 of the local paper; "
        "[paper](https://arxiv.org/html/2610.02185v1), "
        "[official model](https://huggingface.co/ByteDance/Ouro-1.4B), and "
        "[official lm-eval](https://github.com/EleutherAI/lm-evaluation-harness). "
        "Exact revisions and file hashes are recorded in ../../models/model_manifest.json, ../../data/protocol.json, and run_manifest.json.",
        "",
    ])
    (args.results / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
