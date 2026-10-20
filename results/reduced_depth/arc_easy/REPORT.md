# Huginn × ARC-Easy: half-depth LoopCD

Complete test split: 2,376 questions, 9,501 answer candidates, 0-shot. The checkpoint, prompts, FP32 precision, and per-question initial states are shared. The reduced-depth setting follows Table A8: R=16, reference h5, fixed strength 0.5. The primary metric is acc_norm.

## Accuracy and compute

| Setting | Correct / 2,376 (acc_norm) | acc_norm (%) | acc (%) | Forward TFLOPs | Relative FLOPs |
|---|---:|---:|---:|---:|---:|
| Baseline R=32 | 1527 | 64.27 | 69.11 | 56.353 | 1.000× |
| Baseline R=16 | 1530 | 64.39 | 69.32 | 29.174 | 0.518× |
| LoopCD R=16, reference=5 | 1552 | 65.32 | 68.52 | 30.350 | 0.539× |

Half-depth LoopCD changes accuracy by **+1.05 pp** against the full-depth baseline and **+0.93 pp** against the half-depth baseline. Its analytic matrix-product FLOPs are **46.14% lower** than the full-depth baseline.

FLOPs use a batch of one 512-token prefill with all positions projected, two operations per multiply-accumulate, and dense attention products. Counts include the recurrent injection projection and the extra coda/head for logit guidance. Embedding lookup and elementwise operations are excluded.

The paper's printed Huginn values are reproduced by omitting the recurrent injection projection from this ledger: 54.526 TFLOPs for R=32, 28.261 for R=16, and 0.5399× for guided R=16. Both ledgers are saved in the cost JSON.

## Paired comparisons

| Comparison (acc_norm) | Change (pp) | 95% paired CI | Corrected / regressed | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| R16 baseline − R32 baseline | +0.13 | [-0.67, +0.93] | 49 / 46 | 0.8376 | 0.8376 |
| R16 LoopCD − R32 baseline | +1.05 | [-0.13, +2.23] | 117 / 92 | 0.09666 | 0.29 |
| R16 LoopCD − R16 baseline | +0.93 | [-0.21, +2.02] | 102 / 80 | 0.1193 | 0.29 |

Intervals use 20,000 paired question resamples with seed 1234. Holm correction covers these three contrasts separately for each metric. This experiment tests one task; the paper's Figure 4 aggregates seven tasks.

## Standalone timing

| Setting | Median time for sampled batches (ms) | IQR (ms) | Speedup vs R32 |
|---|---:|---:|---:|
| Baseline R=32 | 1951.27 | 10.76 | 1.00× |
| Baseline R=16 | 1013.03 | 5.22 | 1.93× |
| LoopCD R=16, reference=5 | 1043.91 | 2.75 | 1.87× |

Policies run independently on the same allocated H200 and identical sampled ARC-Easy batches. Baselines perform one readout; guided scoring performs two. Timing uses warmup, CUDA synchronization, and repeated measurements. Model loading, tokenization, and initial-state construction are excluded; the same prepared initial states are supplied to all policies. These are sampled scoring measurements; the analytic 512-token prefill uses a different input shape.

## Verification

Saved candidate tokens, likelihood sums, predictions, summaries, source snapshots, checkpoint metadata, and data provenance pass independent audits. The h5 reference and final output match native Huginn calls with identical initial states. The new R16 baseline has identical predictions to the prior run; maximum per-token log-probability difference: 0.

[Comparison JSON](comparison.json) · [CSV](comparison.csv) · [Cost and timing evidence](huginn_costs.json)

Reproduce with `make submit-reduced TASK=arc_easy REDUCED_OUTPUT=results/repeats/reduced_depth/arc_easy`; rebuild this report with `make report-reduced TASK=arc_easy`.
