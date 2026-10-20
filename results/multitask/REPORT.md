# Multitask LoopCD-Logits reproduction

Each task evaluates seven model configurations with baseline, fixed, and adaptive guidance. Reports include both primary and supplementary metrics.

| Task | Split / questions | Primary metric | Configurations | Detailed report |
|---|---|---|---:|---|
| ARC-Easy | test / 2376 | acc_norm | 7 | [arc_easy](arc_easy/REPORT.md) |
| ARC-Challenge | test / 1172 | acc_norm | 7 | [arc_challenge](arc_challenge/REPORT.md) |
| SciQ | test / 1000 | acc | 7 | [sciq](sciq/REPORT.md) |
| PIQA | validation / 1838 | acc_norm | 7 | [piqa](piqa/REPORT.md) |

Accuracy and paired statistics are reported separately for each task and metric. Holm correction applies to 14 comparisons within each task/metric family.

This combined report includes only the fully audited tasks selected by `--task`; `--task all` includes all four tasks. Original ARC-Easy scores are preserved. See comparison.csv for a machine-readable table by task, model, and metric. Paper reference values appear only in primary-metric rows; the paper columns in supplementary-metric rows are left empty.
