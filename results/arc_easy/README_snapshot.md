# Ouro-1.4B × ARC-Easy：LoopCD 功能复现

本项目复现 *Decoding Looped Transformers Better for (Almost) Free* 的一个设置：使用官方 **ByteDance/Ouro-1.4B base** 模型，评测 **ARC-Easy 完整测试集 2,376 题**。任务是从给定候选答案中选出正确答案，便于逐题检查 LoopCD 如何改变预测。

这里实现论文的 **LoopCD-Logits**：使用同一模型第 1 次循环的输出作为弱参考，调整第 4 次循环的 logits。只做候选答案全文的条件似然评分，不训练模型、不生成答案，也不覆盖整篇论文的其他模型、任务或加速实验。

## 已完成的结果

2026-10-20 在单张 H200 上完成全部 2,376 题。以下使用预先选定的 `acc_norm`；未归一化 `acc` 也完整保存在报告中。

| 方法 | 本次准确率 | 论文 Table 2 |
|---|---:|---:|
| 普通解码 | 84.09%（1998题） | 84.09% |
| LoopCD 固定强度 | 84.39%（2005题） | 84.39% |
| LoopCD 自适应 | 84.13%（1999题） | 84.26% |

**固定版的汇总准确率与论文一致到小数点后两位**：改对38题、改错31题，净多答对7题。自适应版净多答对1题，低于论文约0.13个百分点。改善幅度小，配对95%区间包含零，不能把本次结果解释为稳定、显著的提升。

全量评分约148秒，峰值已分配显存6.66 GiB；这是各策略共享循环计算的联合耗时，不是加速比。4项独立测试和真实模型检查均通过。详见[完整中文报告](results/arc_easy/REPORT.md)，其中同时展示成功和失败例子。

## 固定实验设置

| 设置 | 本项目采用的值 |
|---|---|
| 模型 | `ByteDance/Ouro-1.4B`，base 版本 |
| 模型 revision | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| 循环次数 / 参考循环 | `R=4` / `h=1` |
| 提示示例 | 8-shot，来自训练集，固定 seed `1234` |
| 比较方法 | 第 4 次循环基线、固定强度 `ω=0.5`、自适应强度上限 `1.0` |
| 辅助检查 | 第 1 次循环弱参考的准确率 |
| 评分 | 同时保留 `acc` 与按答案字符数归一化的 `acc_norm` |
| 数值设置 | FP32，PyTorch `matmul_precision=high`，SDPA；不截断输入 |

调整公式为 `z′ = z₄ + ω(z₄ − z₁)`。自适应版本在每个预测位置使用 `ω = 1 − (p₄,top1 − p₄,top2)`，其中概率来自第 4 次循环的完整词表分布。各方法共享同一次四循环计算；联合运行耗时不能用作方法间的加速比。

预检查发现 BF16 在单条与不同长度批量输入之间有数值敏感性，因此正式实验使用 FP32；`high` 允许 PyTorch 使用较快的内部矩阵乘法精度，不等同于强制 IEEE FP32 的 `highest`。正式配置在真实题目的单条/批量检查中误差为零，零强度输出与官方模型一致。检查针对数值一致性，不根据准确率选择精度或参数。

## 运行

在当前项目目录中直接使用现有 `venv-stage3c-h`，无需重新安装依赖。保留的 `venv-cpu` 未用于本次实验。

```bash
cd /work/zs220/zjsu-loopcd
venv-stage3c-h/bin/python -B scripts/download_model.py
venv-stage3c-h/bin/python -B scripts/prepare_arc.py --seed 1234
mkdir -p results
sbatch scripts/run_arc.slurm
```

下载脚本使用冻结版本的官方模型，校验权重及配置文件；已有文件会校验后复用。数据准备脚本从 AI2 官方归档提取 ARC-Easy 数据，并按固定的 lm-eval 规则生成 `data/arc_easy_8shot.jsonl`。

Slurm 脚本默认使用 `scavenger-h200` 分区和账户，申请 **1 张 H200、6 个 CPU、32 GB 内存、45 分钟**。模型和数据准备完成后，GPU 作业离线运行；日志位于 `results/slurm-作业号.out` 和 `.err`。默认不限制题数，评测完整测试集。

作业成功结束后生成中文报告：

```bash
venv-stage3c-h/bin/python -B scripts/analyze_results.py
```

查看 [实验报告](results/arc_easy/REPORT.md) 和 [结果汇总](results/arc_easy/summary.json)。报告比较实测与论文，展示改对和改错的例子；不根据测试结果选择强度或示例种子。

如需另存一次运行，可传入独立输出目录，并在分析时使用相同目录：

```bash
sbatch scripts/run_arc.slurm --output results/arc_easy_repeat
venv-stage3c-h/bin/python -B scripts/analyze_results.py --results results/arc_easy_repeat
```

`evaluate_arc.py` 还接受 `--model`、`--data`、`--limit`、`--batch-size`、`--token-budget`、`--dtype` 和 `--matmul-precision`。其中 `--limit 0` 表示完整测试集，也是默认值；小样本试跑应使用独立输出目录。相同代码、模型、数据及配置再次提交到同一目录会复用已经保存的候选评分。

## 文件与可比性

- `src/loopcd.py`：循环输出读取、LoopCD 公式、候选评分和模型一致性检查。
- `models/model_manifest.json`、`data/protocol.json`：模型及数据来源、冻结版本、文件哈希和提示规则。
- `results/arc_easy/predictions.jsonl`：逐题答案与各方法评分。
- `results/arc_easy/candidate_scores.jsonl`：候选答案的 token 级分数，支持追溯和续跑。
- `results/arc_easy/run_manifest.json`、`validation.json`：运行环境、配置和验证记录。
- `results/arc_easy/comparison.json`、`examples.json`：同题配对统计和直观案例。

论文未完整指定 few-shot 随机种子、模型 revision 及 ARC-Easy 列的归一化口径。本项目冻结这些选择并同时报告两种评分口径，因此目标是可检查、可重跑的功能复现，实测差异会如实保留在报告中。
