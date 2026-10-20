# Huginn half-depth LoopCD across four tasks

One checkpoint and one fixed setting across ARC-Easy, ARC-Challenge, SciQ, and PIQA: Huginn-0125, R=16, reference h5, fixed strength 0.5. The baselines use R=32 and R=16. All complete task splits use 0-shot prompts, FP32 with highest matrix precision, and identical per-question initial states across policies.

## Complete task results

| Task | Primary metric | Questions | R32 baseline (%) | R16 baseline (%) | R16 LoopCD (%) | Change vs R32 (pp) | Change vs R16 (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| [ARC-Easy](arc_easy/REPORT.md) | acc_norm | 2,376 | 64.27 | 64.39 | 65.32 | +1.05 | +0.93 |
| [ARC-Challenge](arc_challenge/REPORT.md) | acc_norm | 1,172 | 37.88 | 37.03 | 39.25 | +1.37 | +2.22 |
| [SciQ](sciq/REPORT.md) | acc | 1,000 | 93.10 | 93.20 | 93.40 | +0.30 | +0.20 |
| [PIQA](piqa/REPORT.md) | acc_norm | 1,838 | 75.84 | 75.84 | 76.28 | +0.44 | +0.44 |
| **Equal-task mean** | Frozen primary metrics | — | **67.77** | **67.62** | **68.56** | **+0.79** | **+0.94** |

Each task receives one quarter of the mean, regardless of split size. Primary metrics remain acc_norm for ARC-Easy, ARC-Challenge, and PIQA, and acc for SciQ. ARC-Easy, ARC-Challenge, and SciQ use their test splits; PIQA uses its labeled validation split.

Against R32, guided R16 improves on 4 tasks, ties on 0, and declines on 0. Against R16, the corresponding counts are 4, 0, and 0. All four task results contribute to the reported mean.

## Mean comparisons and uncertainty

| Comparison | Equal-task change (pp) | Stratified paired 95% CI | Task wins / ties / losses | Sign-flip p | Holm p (3 tests) |
|---|---:|---:|---:|---:|---:|
| R16 baseline minus R32 baseline | -0.16 | [-0.59, +0.27] | 2 / 1 / 1 | 0.4735 | 0.4735 |
| R16 LoopCD minus R32 baseline | +0.79 | [+0.15, +1.43] | 4 / 0 / 0 | 0.01725 | 0.03451 |
| R16 LoopCD minus R16 baseline | +0.94 | [+0.34, +1.56] | 4 / 0 / 0 | 0.002825 | 0.008475 |

Intervals use 20,000 paired question resamples independently within each task. Two-sided permutation tests use 200,000 sign-flip samples on correctness discordances, with the same equal-task statistic and the plus-one Monte Carlo correction. Each contrast and method uses seed 1234. Holm correction covers the three primary mean contrasts. The tasks are held fixed; uncertainty reflects resampling questions within these tasks.

## Secondary metric views

| Mean definition | R32 baseline (%) | R16 baseline (%) | R16 LoopCD (%) | Guided change vs R32 (pp) | Guided change vs R16 (pp) |
|---|---:|---:|---:|---:|---:|
| Frozen primary metrics | 67.77 | 67.62 | 68.56 | +0.79 | +0.94 |
| All four tasks: acc | 68.19 | 67.62 | 68.11 | -0.08 | +0.49 |
| All four tasks: acc_norm | 67.07 | 66.84 | 67.96 | +0.89 | +1.12 |

Both secondary metric views are descriptive. The primary metrics were frozen before the task extension and determine the primary comparisons.

## Per-task primary comparisons

| Task | Comparison | Change (pp) | Corrected / regressed | Exact McNemar p | Holm p (12 tests) |
|---|---|---:|---:|---:|---:|
| ARC-Easy | R16 baseline minus R32 baseline | +0.13 | 49 / 46 | 0.8376 | 1 |
| ARC-Easy | R16 LoopCD minus R32 baseline | +1.05 | 117 / 92 | 0.09666 | 1 |
| ARC-Easy | R16 LoopCD minus R16 baseline | +0.93 | 102 / 80 | 0.1193 | 1 |
| ARC-Challenge | R16 baseline minus R32 baseline | -0.85 | 20 / 30 | 0.2026 | 1 |
| ARC-Challenge | R16 LoopCD minus R32 baseline | +1.37 | 69 / 53 | 0.1742 | 1 |
| ARC-Challenge | R16 LoopCD minus R16 baseline | +2.22 | 70 / 44 | 0.01881 | 0.2257 |
| SciQ | R16 baseline minus R32 baseline | +0.10 | 6 / 5 | 1 | 1 |
| SciQ | R16 LoopCD minus R32 baseline | +0.30 | 12 / 9 | 0.6636 | 1 |
| SciQ | R16 LoopCD minus R16 baseline | +0.20 | 12 / 10 | 0.8318 | 1 |
| PIQA | R16 baseline minus R32 baseline | +0.00 | 22 / 22 | 1 | 1 |
| PIQA | R16 LoopCD minus R32 baseline | +0.44 | 51 / 43 | 0.4705 | 1 |
| PIQA | R16 LoopCD minus R16 baseline | +0.44 | 43 / 35 | 0.4282 | 1 |

This table applies one Holm family across all twelve primary task contrasts. Individual task reports retain their within-task three-contrast adjustment.

## Compute and sampled timing

| Policy | Forward TFLOPs (512 tokens) | Fraction of R32 |
|---|---:|---:|
| Baseline R=32 | 56.353 | 1.0000× |
| Baseline R=16 | 29.174 | 0.5177× |
| LoopCD R=16, reference=5 | 30.350 | 0.5386× |

The same analytic cost applies to every task: batch size one, 512-token prefill, all positions projected, two FLOPs per multiply-accumulate, dense attention products, and the recurrent injection matrix included. Guided R16 includes its second coda and head. It uses 46.14% fewer matrix-product FLOPs than R32. The separate paper-convention ledger is retained in the JSON.

| Task | R32 median (ms) | R16 median (ms) | Guided R16 median (ms) | R16 speedup | Guided R16 speedup |
|---|---:|---:|---:|---:|---:|
| ARC-Easy | 1951.27 | 1013.03 | 1043.91 | 1.93× | 1.87× |
| ARC-Challenge | 2280.79 | 1175.26 | 1215.54 | 1.94× | 1.88× |
| SciQ | 5447.37 | 2804.96 | 2887.40 | 1.94× | 1.89× |
| PIQA | 2133.09 | 1103.79 | 1142.15 | 1.93× | 1.87× |
| **Geometric mean speedup** | — | — | — | **1.94×** | **1.87×** |

Each task measures three fixed length-quantile batches of eight answer candidates. The three policies run independently on identical inputs within the same allocated H200 job, with two warmups and six synchronized repetitions per batch and policy. Reported latency is the median summed time for the three batches. Per-task JSON files include dispersion, memory measurements, and individual samples. Timings include forward scoring, guidance, and target log probabilities; they exclude model loading, tokenization, input preparation, and CPU random-state generation/transfer. These sampled timings use answer-position projections and task-specific lengths, distinct from the analytic 512-token prefill.

## Reproduction evidence

Every task report independently audits saved candidate likelihoods, predictions, paired document identities, source snapshots, model/data provenance, reference h5, and standalone timing. The new R16 baselines retain the original predictions. The existing ARC-Challenge evaluation and timing records are reused.

[Frozen analysis plan](analysis_plan.json) · [Comparison JSON](comparison.json) · [Comparison CSV](comparison.csv)

Rebuild all audited reports with `venv/bin/python -B scripts/analyze_reduced_depth.py --task all`.
