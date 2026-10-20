# ARC-Easy: LoopCD-Logits across all seven model configurations

The full test split contains 2376 questions per configuration.

The primary metric is `acc_norm`; both metrics are reported. `acc` uses the sum of conditional log likelihoods over the complete answer; `acc_norm` divides this sum by the Unicode character count of the original answer, excluding the added leading space.

## Reproduction / paper (%)

| Configuration | Shots | Baseline | Fixed | Adaptive | Fixed change (pp) | Adaptive change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Ouro-1.4B R=4 | 8 | 84.09 / 84.09 | 84.39 / 84.39 | 84.13 / 84.26 | +0.29 | +0.04 |
| Ouro-2.6B R=4 | 8 | 88.17 / 88.01 | 88.93 / 89.10 | 88.72 / 89.02 | +0.76 | +0.55 |
| Huginn-0125 R=16 | 0 | 64.39 / 64.31 | 66.33 / 64.86 | 66.25 / 64.94 | +1.94 | +1.85 |
| Huginn-0125 R=32 | 0 | 64.27 / 64.31 | 65.28 / 64.10 | 65.70 / 64.69 | +1.01 | +1.43 |
| Parcae-370M R=8 | 0 | 54.59 / 54.42 | 54.59 / 55.05 | 54.50 / 54.63 | +0.00 | -0.08 |
| Parcae-1.3B R=8 | 0 | 64.90 / 65.07 | 66.16 / 65.91 | 65.49 / 66.16 | +1.26 | +0.59 |
| Looped-Qwen3-4B R=8 | 0 | 79.38 / 79.25 | 79.25 / 79.00 | 79.00 / 78.58 | -0.13 | -0.38 |

## Paired question-level statistics (acc_norm)

| Configuration | Policy | Corrected | Regressed | Answers changed | Net correct | Paired 95% CI (pp) | McNemar p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Ouro-1.4B R=4 | Fixed | 38 | 31 | 88 | +7 | [-0.38, +0.97] | 0.4704 | 1 |
| Ouro-1.4B R=4 | Adaptive | 48 | 47 | 117 | +1 | [-0.76, +0.84] | 1 | 1 |
| Ouro-2.6B R=4 | Fixed | 39 | 21 | 68 | +18 | [+0.13, +1.39] | 0.02734 | 0.3007 |
| Ouro-2.6B R=4 | Adaptive | 54 | 41 | 107 | +13 | [-0.25, +1.35] | 0.2181 | 1 |
| Huginn-0125 R=16 | Fixed | 168 | 122 | 364 | +46 | [+0.55, +3.32] | 0.008119 | 0.1055 |
| Huginn-0125 R=16 | Adaptive | 146 | 102 | 309 | +44 | [+0.55, +3.16] | 0.00621 | 0.08694 |
| Huginn-0125 R=32 | Fixed | 156 | 132 | 367 | +24 | [-0.42, +2.40] | 0.1752 | 1 |
| Huginn-0125 R=32 | Adaptive | 145 | 111 | 329 | +34 | [+0.13, +2.74] | 0.03895 | 0.3895 |
| Parcae-370M R=8 | Fixed | 57 | 57 | 158 | +0 | [-0.88, +0.88] | 1 | 1 |
| Parcae-370M R=8 | Adaptive | 91 | 93 | 259 | -2 | [-1.22, +1.01] | 0.9413 | 1 |
| Parcae-1.3B R=8 | Fixed | 91 | 61 | 200 | +30 | [+0.21, +2.27] | 0.01837 | 0.2204 |
| Parcae-1.3B R=8 | Adaptive | 131 | 117 | 320 | +14 | [-0.72, +1.89] | 0.4091 | 1 |
| Looped-Qwen3-4B R=8 | Fixed | 45 | 48 | 116 | -3 | [-0.93, +0.67] | 0.8358 | 1 |
| Looped-Qwen3-4B R=8 | Adaptive | 44 | 53 | 130 | -9 | [-1.22, +0.42] | 0.4168 | 1 |

No primary-metric comparison for this task has Holm p<0.05.

Pointwise 95% confidence intervals use 20,000 paired question-level bootstrap resamples (seed=1234). Exact two-sided McNemar tests receive Holm correction separately for each task and metric, across 14 comparisons per family.

## Both metrics and the reference state

| Configuration | Metric | Baseline | Fixed | Adaptive | First full-loop reference |
|---|---|---:|---:|---:|---:|
| Ouro-1.4B R=4 | acc | 1991 (83.80%) | 2006 (84.43%) | 2009 (84.55%) | 1527 (64.27%) |
| Ouro-1.4B R=4 | acc_norm | 1998 (84.09%) | 2005 (84.39%) | 1999 (84.13%) | 1386 (58.33%) |
| Ouro-2.6B R=4 | acc | 2071 (87.16%) | 2086 (87.79%) | 2088 (87.88%) | 1713 (72.10%) |
| Ouro-2.6B R=4 | acc_norm | 2095 (88.17%) | 2113 (88.93%) | 2108 (88.72%) | 1630 (68.60%) |
| Huginn-0125 R=16 | acc | 1647 (69.32%) | 1595 (67.13%) | 1615 (67.97%) | 859 (36.15%) |
| Huginn-0125 R=16 | acc_norm | 1530 (64.39%) | 1576 (66.33%) | 1574 (66.25%) | 786 (33.08%) |
| Huginn-0125 R=32 | acc | 1642 (69.11%) | 1585 (66.71%) | 1604 (67.51%) | 859 (36.15%) |
| Huginn-0125 R=32 | acc_norm | 1527 (64.27%) | 1551 (65.28%) | 1561 (65.70%) | 786 (33.08%) |
| Parcae-370M R=8 | acc | 1459 (61.41%) | 1464 (61.62%) | 1437 (60.48%) | 1321 (55.60%) |
| Parcae-370M R=8 | acc_norm | 1297 (54.59%) | 1297 (54.59%) | 1295 (54.50%) | 1184 (49.83%) |
| Parcae-1.3B R=8 | acc | 1669 (70.24%) | 1649 (69.40%) | 1627 (68.48%) | 1497 (63.01%) |
| Parcae-1.3B R=8 | acc_norm | 1542 (64.90%) | 1572 (66.16%) | 1556 (65.49%) | 1320 (55.56%) |
| Looped-Qwen3-4B R=8 | acc | 1915 (80.60%) | 1920 (80.81%) | 1923 (80.93%) | 1806 (76.01%) |
| Looped-Qwen3-4B R=8 | acc_norm | 1886 (79.38%) | 1883 (79.25%) | 1877 (79.00%) | 1752 (73.74%) |

Comparisons with opposite directions of change between the two metrics: Huginn-0125 R=16 Fixed; Huginn-0125 R=16 Adaptive; Huginn-0125 R=32 Fixed; Huginn-0125 R=32 Adaptive; Parcae-1.3B R=8 Fixed; Parcae-1.3B R=8 Adaptive; Looped-Qwen3-4B R=8 Fixed; Looped-Qwen3-4B R=8 Adaptive.

## Evaluation protocol

- Fixed guidance strength is 0.5. The adaptive cap is 0.5 for Huginn and 1.0 for the other models. Adaptive strength uses the top-two probability gap over the full vocabulary. The reference is the first complete update followed by the full coda/head; all policies share one loop trajectory.
- Shot counts follow paper Table A2, with seed 1234. Nonzero-shot examples are sampled from the training set as questions are processed. SciQ prompts include support, and PIQA uses the labeled validation split.
- Huginn receives one BOS token through its official interface; the other models receive no extra BOS token. Prompts use plain task text. Huginn/Parcae random initial states are fixed by question ID, with positional noise shared across candidates.
- For the new tasks, context is truncated from the left only when the model window is insufficient, preserving all answer tokens and prediction positions. The table below reports truncation counts for each configuration.
- Checkpoint manifests, small model files, tokenizers, and native source files are verified with SHA256. Large weights are checked by file size in this audit, with hashes referenced from download-time verification.
- Evaluations use full recurrent depth. Recorded execution time covers all four scoring policies sharing one loop trajectory.

| Configuration | dtype / matmul | Truncated questions / candidates | Maximum input tokens | Maximum batch/single token logp difference |
|---|---|---:|---:|---:|
| Ouro-1.4B R=4 | float32 / high | 0 / 0 | 524 | 0 |
| Ouro-2.6B R=4 | float32 / high | 0 / 0 | 524 | 0 |
| Huginn-0125 R=16 | float32 / highest | 0 / 0 | 164 | 2.01e-05 |
| Huginn-0125 R=32 | float32 / highest | 0 / 0 | 164 | 1.72e-05 |
| Parcae-370M R=8 | float32 / high | 0 / 0 | 159 | 0 |
| Parcae-1.3B R=8 | float32 / high | 0 / 0 | 159 | 0 |
| Looped-Qwen3-4B R=8 | float32 / highest | 0 / 0 | 166 | 6.1e-05 |

ARC-Easy analysis reads the original candidate scores. The full historical protocol is documented in the [ARC-Easy report](../../arc_all_models/REPORT.md).

See comparison.json for source paths, candidate-level truncation inventories, paired statistics for both metrics, and full audit evidence for each configuration. Regenerate with `venv/bin/python scripts/analyze.py --task arc_easy`.

Sources: [paper Tables 2 / A2 / A4](https://arxiv.org/html/2610.02185v1) and the [pinned lm-eval implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). Paper reference values and supporting evidence for the new tasks are stored in configs/paper_mc_reference.json; ARC-Easy reference values remain in configs/arc_all_models.json.
