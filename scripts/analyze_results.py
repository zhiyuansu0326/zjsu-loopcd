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
    "baseline": "普通解码（第4次循环）",
    "loopcd_fixed": "LoopCD 固定强度 ω=0.5",
    "loopcd_adaptive": "LoopCD 自适应上限=1.0",
    "reference": "弱参考（第1次循环）",
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
        "# Ouro-1.4B × ARC-Easy：LoopCD 功能复现",
        "",
        f"{'完整测试集' if len(rows) == 2376 else '测试子集'} {len(rows)} 题。唯一模型为 ByteDance/Ouro-1.4B base；8-shot 示例来自训练集，seed=1234，4次完整循环。",
        "无训练、无参数搜索；固定采用论文 Table A4 的 ω=0.5、自适应上限1.0。所有策略使用相同问题、示例、权重和循环轨迹。",
        "",
        "## 实测与论文",
        "",
        "下表预先选择 acc_norm 作为展示口径（答案条件对数似然除以答案文本字符数）。"
        "论文只明确 likelihood scoring，未明确写出 ARC-E 列的归一化口径，因此同时保留未归一化 acc，不能声称是作者原评测代码的逐位复现。",
        "",
        "| 策略 | 答对数 | 本次 acc_norm | 论文 Table 2 | 本次 acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in NAMES:
        lines.append(
            f"| {NAMES[mode]} | {summary['correct']['acc_norm'][mode]}/{len(rows)} | "
            f"{summary['accuracy_percent']['acc_norm'][mode]:.2f}% | "
            f"{str(PAPER[mode]) + '%' if mode in PAPER else '未比较'} | "
            f"{summary['accuracy_percent']['acc'][mode]:.2f}% |"
        )
    lines.extend([
        "",
        "## 同题配对变化",
        "",
        "| 策略 | 改对 | 改错 | 净增答对 | 提升百分点 | 配对 bootstrap 95%区间 | McNemar p |",
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
        "区间使用固定种子1234、20000次同题配对重采样；它描述测试题抽样不确定性，不是多次模型运行的波动。"
        "单模型单任务的结果不能推广为整篇论文所有模型、生成任务或降循环速度结论。",
        "",
    ])
    if all(values["paired_bootstrap_95ci_pp"][0] <= 0 <= values["paired_bootstrap_95ci_pp"][1]
           for values in paired["acc_norm"].values()):
        lines.extend([
            "本次两种 LoopCD 的 acc_norm 改善区间都包含零；这次实测提升较小，不能据此认定存在稳定、显著的准确率提升。",
            "",
        ])
    lines.extend([
        "## 实现和验证",
        "",
        "官方 Ouro 在每次完整24层循环后执行 RMSNorm，再继续下一循环。"
        "这里直接读取官方模型返回的第1/第4次循环状态，通过共享 lm_head 得到 z₁、z₄，"
        "以 z′=z₄+ω(z₄−z₁) 评分。自适应 ω=1−(p₄,top1−p₄,top2)，概率在完整词表上计算。",
        f"模型计算使用 {manifest['dtype']}（FP32 matmul precision={manifest.get('float32_matmul_precision', 'default')}），"
        "FP32 logits 运算和 log_softmax；对每个候选答案的全部 token 做 teacher-forced 条件似然评分。"
        "不使用选项字母打分，不截断上下文，不抽样生成。",
        "",
        "已检查 ω=0 与官方第4循环 logits 一致、早晚循环输出不同、单条/补齐批量输入一致性，"
        "另有微型官方 Ouro 的全答案位置测试。真实模型检查见 validation.json，测试源码见 ../../tests/test_loopcd.py。",
        "",
        "数值预检中，BF16 的单条/批量答案 token 对数概率最大差异为0.28756，未用于正式结果。"
        "正式配置的同项检查误差为零。FP32 matmul precision=high 允许较快的内部矩阵乘法实现，"
        "不等同于强制最高精度的 highest；这里选择精度依据是数值检查，没有按测试准确率调参。",
        "",
        "三种策略和弱参考共用一次4循环计算，因此记录的总耗时是联合实验耗时，不能据此宣称 LoopCD 的单独加速比。",
        "",
        f"- 模型加载和验证：{summary['load_and_validation_seconds']:.1f} 秒。",
        f"- 本次评分调用：{summary['this_invocation_scoring_seconds']:.1f} 秒。",
        f"- 峰值已分配 GPU 显存：{summary['peak_gpu_allocated_gib']:.2f} GiB。",
        "",
        "## 直观例子",
        "",
        "以下按原测试集顺序分别取每种策略最早两条改对/改错案例；完整逐题结果保存在 predictions.jsonl。",
    ])
    for ex in examples:
        lines.extend([
            "",
            f"### {ex['id']} · {NAMES[ex['policy']]} · {'改对' if ex['category'] == 'corrected' else '改错'}",
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
            f"正确答案：{ex['labels'][ex['gold']]}；普通解码：{ex['labels'][baseline]}；"
            f"LoopCD：{ex['labels'][guided]}。",
        ])
    lines.extend([
        "",
        "## 可比性限制",
        "",
        "论文没有公布完整 few-shot 实例/随机种子、模型精确 revision 和所有数值实现细节。"
        "本次采用冻结的官方模型版本、固定的官方 lm-eval 提示构造与采样规则，未以测试结果选择种子或强度。"
        "绝对分数的差异可能来自这些未公开细节、数值精度/attention 实现和归一化口径。"
        "即便接近论文分数，也只代表这个设置的功能复现。",
        "",
        "来源：本地论文第6页 Table 2、第17页 Table A2、第18页 Table A4；"
        "[论文](https://arxiv.org/html/2610.02185v1)、"
        "[官方模型](https://huggingface.co/ByteDance/Ouro-1.4B)、"
        "[官方 lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)。"
        "精确版本和文件哈希见 ../../models/model_manifest.json、../../data/protocol.json 和 run_manifest.json。",
        "",
    ])
    (args.results / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
