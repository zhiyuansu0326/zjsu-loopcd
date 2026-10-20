# Huginn × ARC-Challenge: half-depth LoopCD

Complete test split: 1,172 questions, 4,687 answer candidates, 0-shot. The checkpoint, prompts, FP32 precision, and per-question initial states are shared. The reduced-depth setting follows Table A8: R=16, reference h5, fixed strength 0.5.

## Accuracy and compute

| Setting | Correct / 1,172 | acc_norm (%) | acc (%) | Forward TFLOPs | Relative FLOPs |
|---|---:|---:|---:|---:|---:|
| Baseline R=32 | 444 | 37.88 | 35.15 | 56.353 | 1.000× |
| Baseline R=16 | 434 | 37.03 | 32.76 | 29.174 | 0.518× |
| LoopCD R=16, reference=5 | 460 | 39.25 | 35.32 | 30.350 | 0.539× |

Half-depth LoopCD changes accuracy by **+1.37 pp** against the full-depth baseline and **+2.22 pp** against the half-depth baseline. Its analytic matrix-product FLOPs are **46.14% lower** than the full-depth baseline.

FLOPs use a batch of one 512-token prefill with all positions projected, two operations per multiply-accumulate, and dense attention products. Counts include the recurrent injection projection and the extra coda/head for logit guidance. Embedding lookup and elementwise operations are excluded.

The paper's printed Huginn values are reproduced by omitting the recurrent injection projection from this ledger: 54.526 TFLOPs for R=32, 28.261 for R=16, and 0.5399× for guided R=16. Both ledgers are saved in the cost JSON.

## Paired comparisons

| Comparison (acc_norm) | Change (pp) | 95% paired CI | Corrected / regressed | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| R16 baseline − R32 baseline | -0.85 | [-2.05, +0.34] | 20 / 30 | 0.2026 | 0.3484 |
| R16 LoopCD − R32 baseline | +1.37 | [-0.51, +3.24] | 69 / 53 | 0.1742 | 0.3484 |
| R16 LoopCD − R16 baseline | +2.22 | [+0.51, +4.01] | 70 / 44 | 0.01881 | 0.05643 |

Intervals use 20,000 paired question resamples with seed 1234. Holm correction covers these three contrasts separately for each metric. This experiment tests one task; the paper's Figure 4 aggregates seven tasks.

## Standalone timing

| Setting | Median time for sampled batches (ms) | IQR (ms) | Speedup vs R32 |
|---|---:|---:|---:|
| Baseline R=32 | 2280.79 | 7.15 | 1.00× |
| Baseline R=16 | 1175.26 | 6.27 | 1.94× |
| LoopCD R=16, reference=5 | 1215.54 | 5.37 | 1.88× |

Policies run independently on the same allocated H200 and identical sampled ARC-Challenge batches. Baselines perform one readout; guided scoring performs two. Timing uses warmup, CUDA synchronization, and repeated measurements. Model loading, tokenization, and initial-state construction are excluded; the same prepared initial states are supplied to all policies. These are sampled scoring measurements; the analytic 512-token prefill uses a different input shape.

## Verification

Saved candidate tokens, likelihood sums, predictions, summaries, source snapshots, checkpoint metadata, and data provenance pass independent audits. The h5 reference and final output match native Huginn calls with identical initial states. The new R16 baseline has identical predictions to the prior run; maximum per-token log-probability difference: 0.

[Comparison JSON](comparison.json) · [CSV](comparison.csv) · [Cost and timing evidence](huginn_costs.json)

Reproduce with `make submit-reduced TASK=arc_challenge REDUCED_OUTPUT=results/repeats/huginn_half_depth`; rebuild this report with `make report-reduced TASK=arc_challenge`.
