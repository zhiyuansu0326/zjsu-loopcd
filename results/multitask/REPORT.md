# 多任务 LoopCD-Logits 复现

下列任务全部通过七个模型配置的完整候选与逐题审计；各项均报告基线、固定和自适应引导。主指标在评分前确定，补充指标完整保留。

| 任务 | 划分 / 题数 | 主指标 | 配置数 | 详细报告 |
|---|---|---|---:|---|
| ARC-Easy | test / 2376 | acc_norm | 7 | [arc_easy](arc_easy/REPORT.md) |
| ARC-Challenge | test / 1172 | acc_norm | 7 | [arc_challenge](arc_challenge/REPORT.md) |
| SciQ | test / 1000 | acc | 7 | [sciq](sciq/REPORT.md) |
| PIQA | validation / 1838 | acc_norm | 7 | [piqa](piqa/REPORT.md) |

每个任务分别在 14 个比较内做 Holm 校正，没有合并成跨任务显著性结论。未计算不同提示协议、不同指标之间的合并准确率，也不以结果选择最佳策略。论文未公开全部评测细节，因此这是固定协议的独立复现，不能把接近的分数视为逐位重现。

本总报告只包含本次 `--task` 选择且完整审计的任务；`--task all` 包含全部四项。原始 ARC-Easy 评分和报告保留不变。逐任务、模型、指标的机器可读表见 comparison.csv；该表只在主指标行填论文引用值，补充指标行的论文列留空。
