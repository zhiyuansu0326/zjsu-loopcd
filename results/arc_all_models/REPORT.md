# ARC-Easy: LoopCD-Logits reproduction for all paper model configurations

This report covers LoopCD-Logits on ARC-Easy for every model configuration in Table 2 of the paper. Four model families, six checkpoints, and seven loop-depth configurations are complete, each using all 2,376 questions and 9,501 candidate answers. Scores are computed from saved answer-token log probabilities.

## Reproduction and paper results

The primary metric is acc_norm: the sum of conditional log likelihoods over every candidate-answer token, divided by the original answer text's Unicode character count, excluding any added leading space. Unnormalized acc is also reported. Cells show reproduced / paper percentages.

| Configuration | Baseline | LoopCD fixed | LoopCD adaptive | Fixed net correct / pp | Adaptive net correct / pp |
|---|---:|---:|---:|---:|---:|
| Ouro-1.4B R=4 | 84.09 / 84.09 | 84.39 / 84.39 | 84.13 / 84.26 | +7 / +0.29 | +1 / +0.04 |
| Ouro-2.6B R=4 | 88.17 / 88.01 | 88.93 / 89.10 | 88.72 / 89.02 | +18 / +0.76 | +13 / +0.55 |
| Huginn-0125 R=16 | 64.39 / 64.31 | 66.33 / 64.86 | 66.25 / 64.94 | +46 / +1.94 | +44 / +1.85 |
| Huginn-0125 R=32 | 64.27 / 64.31 | 65.28 / 64.10 | 65.70 / 64.69 | +24 / +1.01 | +34 / +1.43 |
| Parcae-370M R=8 | 54.59 / 54.42 | 54.59 / 55.05 | 54.50 / 54.63 | +0 / +0.00 | -2 / -0.08 |
| Parcae-1.3B R=8 | 64.90 / 65.07 | 66.16 / 65.91 | 65.49 / 66.16 | +30 / +1.26 | +14 / +0.59 |
| Looped-Qwen3-4B R=8 | 79.38 / 79.25 | 79.25 / 79.00 | 79.00 / 78.58 | -3 / -0.13 | -9 / -0.38 |

The following comparisons change direction between metrics: Huginn-0125 R=16 Fixed: acc_norm +1.94 pp, acc -2.19 pp; Huginn-0125 R=16 Adaptive: acc_norm +1.85 pp, acc -1.35 pp; Huginn-0125 R=32 Fixed: acc_norm +1.01 pp, acc -2.40 pp; Huginn-0125 R=32 Adaptive: acc_norm +1.43 pp, acc -1.60 pp; Parcae-1.3B R=8 Fixed: acc_norm +1.26 pp, acc -0.84 pp; Parcae-1.3B R=8 Adaptive: acc_norm +0.59 pp, acc -1.77 pp; Looped-Qwen3-4B R=8 Fixed: acc_norm -0.13 pp, acc +0.21 pp; Looped-Qwen3-4B R=8 Adaptive: acc_norm -0.38 pp, acc +0.34 pp.

Baseline denotes each model's native final output at its specified depth; the Looped-Qwen baseline is the output after eight damped loop updates. Each comparison shares a single loop trajectory. The weak reference uses the state after the first complete update, passed through the model's full output path to obtain logits.

## Paired changes on the same questions (acc_norm)

| Configuration | Policy | Corrected | Regressed | Changed answers | Net correct | Paired 95% CI (pp) | McNemar p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Ouro-1.4B R=4 | Fixed | 38 | 31 | 88 | +7 | [-0.38, +0.97] | 0.4704 | 1.0000 |
| Ouro-1.4B R=4 | Adaptive | 48 | 47 | 117 | +1 | [-0.76, +0.84] | 1.0000 | 1.0000 |
| Ouro-2.6B R=4 | Fixed | 39 | 21 | 68 | +18 | [+0.13, +1.39] | 0.0273 | 0.3007 |
| Ouro-2.6B R=4 | Adaptive | 54 | 41 | 107 | +13 | [-0.25, +1.35] | 0.2181 | 1.0000 |
| Huginn-0125 R=16 | Fixed | 168 | 122 | 364 | +46 | [+0.55, +3.32] | 0.0081 | 0.1055 |
| Huginn-0125 R=16 | Adaptive | 146 | 102 | 309 | +44 | [+0.55, +3.16] | 0.0062 | 0.0869 |
| Huginn-0125 R=32 | Fixed | 156 | 132 | 367 | +24 | [-0.42, +2.40] | 0.1752 | 1.0000 |
| Huginn-0125 R=32 | Adaptive | 145 | 111 | 329 | +34 | [+0.13, +2.74] | 0.0390 | 0.3895 |
| Parcae-370M R=8 | Fixed | 57 | 57 | 158 | +0 | [-0.88, +0.88] | 1.0000 | 1.0000 |
| Parcae-370M R=8 | Adaptive | 91 | 93 | 259 | -2 | [-1.22, +1.01] | 0.9413 | 1.0000 |
| Parcae-1.3B R=8 | Fixed | 91 | 61 | 200 | +30 | [+0.21, +2.27] | 0.0184 | 0.2204 |
| Parcae-1.3B R=8 | Adaptive | 131 | 117 | 320 | +14 | [-0.72, +1.89] | 0.4091 | 1.0000 |
| Looped-Qwen3-4B R=8 | Fixed | 45 | 48 | 116 | -3 | [-0.93, +0.67] | 0.8358 | 1.0000 |
| Looped-Qwen3-4B R=8 | Adaptive | 44 | 53 | 130 | -9 | [-1.22, +0.42] | 0.4168 | 1.0000 |

Corrected and regressed counts compare the same questions against baseline; changed answers also include switches between incorrect options. Pointwise 95% intervals use 20,000 paired question-level bootstrap resamples (seed=1234). McNemar tests are exact and two-sided. Holm correction applies across the 14 primary-metric comparisons.

No acc_norm comparison has Holm p<0.05.

## Complete accuracy results for both metrics

Each cell shows the correct count (percentage), including the first-loop reference.

| Configuration | Metric | Baseline | Fixed | Adaptive | First-loop weak reference |
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

For the supplementary acc metric, the following changes meet Holm p<0.05 across that metric's 14 comparisons: Huginn-0125 R=16 Fixed (-2.19 pp, Holm p=0.0197); Huginn-0125 R=32 Fixed (-2.40 pp, Holm p=0.0046).

Corrected and regressed counts, net changes, bootstrap intervals, and McNemar tests for acc are also saved in comparison.json. Its Holm correction is performed separately across the 14 acc comparisons as a supplementary metric.

## Evaluation protocol

- Ouro uses 8-shot examples from the training set; other configurations are 0-shot. Questions retain their original order, with prompts formatted as `Question: ...\nAnswer:`. Scoring uses the complete answer text and context. The 8-shot procedure advances one `random.Random(1234)` instance with `.sample(train, 8)`, resampling examples for each question.
- The formula is `z′ = z_R + ω(z_R − z_1)`. Fixed ω=0.5; adaptive `ω = cap × [1 − (p_R,top1 − p_R,top2)]`, with probabilities computed over the full vocabulary. Huginn uses cap=0.5; all others use cap=1.0, following Table A4.
- All four Ouro loops traverse the complete stack and use native normalization. Huginn performs 16 or 32 complete updates. Its first-update state passes through the full coda and output normalization; the official tokenizer/evaluator requires exactly one BOS.
- Huginn and the published Parcae checkpoints have stochastic loop initialization. To make candidate answers, batch lengths, and policies comparable, the native initialization distribution is retained, with shared position states determined by seed=1234 and question ID. Native forward checks use the same explicit initial state.
- Looped-Qwen uses the original official Qwen/Qwen3-4B release weights. It uses a 15-layer prefix, the zero-indexed loop window 15:19, and a 17-layer tail. Eight updates follow `u_next = (7/8)u + (1/8)g(u)`. The early output passes through the complete tail/head after the first damped update. The local wrapper implements K=8 from the LoopCD appendix and is checked by independent eight-step replay and recovery of the ordinary dense output at K=1.
- Model revisions, source hashes, and lm-eval prompt rules are pinned in the repository manifests.

## Numerical validation and joint-run resources

Each configuration checks zero-guidance identity, agreement with native or independently replayed loop outputs, distinct early and final readouts, and agreement between single-example and padded-batch scoring. The new unified evaluator also checks that changing masked padding tokens leaves valid outputs unchanged. The earlier Ouro-1.4B run reuses its completed native-output and batch-consistency checks. GPU runs use the dtype and matmul settings below; logit guidance and log_softmax use FP32. `high` permits faster internal matrix multiplication, while `highest` uses the highest precision. Precision settings follow numerical consistency checks.

Huginn and Looped-Qwen use `highest` to meet the numerical consistency thresholds. Error measurements, job records, and precision settings are recorded in precision_decisions.json.

| Configuration | dtype / matmul | Maximum single/batch token logp difference | Scoring seconds in this invocation | Peak GPU memory (GiB) |
|---|---|---:|---:|---:|
| Ouro-1.4B R=4 | float32 / high | 0 | 147.9 | 6.66 |
| Ouro-2.6B R=4 | float32 / high | 0 | 298.7 | 10.83 |
| Huginn-0125 R=16 | float32 / highest | 2.01e-05 | 465.6 | 14.57 |
| Huginn-0125 R=32 | float32 / highest | 1.72e-05 | 882.9 | 14.57 |
| Parcae-370M R=8 | float32 / high | 0 | 10.0 | 1.85 |
| Parcae-1.3B R=8 | float32 / high | 0 | 22.7 | 5.42 |
| Looped-Qwen3-4B R=8 | float32 / highest | 6.1e-05 | 131.5 | 15.57 |

These resources cover the joint experiment in which baseline, both LoopCD policies, and the weak reference share loop trajectories. Timing covers the current invocation. Model loading/validation time and GPU/job information are recorded in comparison.json.

## Auditable artifacts and sources

Each configuration directory retains candidate_scores.jsonl, predictions.jsonl, summary.json, run_manifest.json, and validation.json. New runs additionally save source_snapshot; Ouro-1.4B reuses ../arc_easy. comparison.json contains all audit counts, file hashes, run parameters, and statistics for both metrics. Regenerate this report with `venv/bin/python scripts/analyze_arc_suite.py`.

Independent data reconstruction is recorded in data_audit.json; checks of official model implementations and metadata are recorded in source_integrity.json. The aggregation script also verifies the original Train/Test file hashes, reconstructs all original questions and few-shot contexts, and checks native-output or independent-replay validation for each model family.

Sources: [LoopCD Tables 2, A1, A2, and A4](https://arxiv.org/html/2610.02185v1), [Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1), and [pinned lm-eval source](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). Official model repositories and exact revisions are recorded in models/*manifest.json; data sources, character normalization, and sampling references are recorded in data/protocol.json.
