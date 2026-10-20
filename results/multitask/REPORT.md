# Multitask LoopCD-Logits reproduction

All tasks below passed complete candidate-level and question-level audits across seven model configurations. Each task reports baseline, fixed, and adaptive guidance. Primary metrics were determined before scoring; supplementary metrics are retained in full.

| Task | Split / questions | Primary metric | Configurations | Detailed report |
|---|---|---|---:|---|
| ARC-Easy | test / 2376 | acc_norm | 7 | [arc_easy](arc_easy/REPORT.md) |
| ARC-Challenge | test / 1172 | acc_norm | 7 | [arc_challenge](arc_challenge/REPORT.md) |
| SciQ | test / 1000 | acc | 7 | [sciq](sciq/REPORT.md) |
| PIQA | validation / 1838 | acc_norm | 7 | [piqa](piqa/REPORT.md) |

Each task applies Holm correction within its own family of 14 comparisons; results are not combined into a cross-task significance claim. Accuracy is not pooled across different prompt protocols or metrics, and results are not used to choose a best policy. Because the paper does not publish every evaluation detail, this is an independent reproduction under a fixed protocol; close scores do not establish bit-for-bit replication.

This combined report includes only the fully audited tasks selected by `--task`; `--task all` includes all four tasks. Original ARC-Easy scores are preserved. See comparison.csv for a machine-readable table by task, model, and metric. Paper reference values appear only in primary-metric rows; the paper columns in supplementary-metric rows are left empty.
