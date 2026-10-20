# Huginn LoopCD-Hidden at full depth across four tasks

Eight complete evaluations use Huginn R16 with h7 and R32 with h6, both at fixed strength 0.5. All runs use 0-shot prompts, FP32/highest, and the original per-question initial states. Hidden blends raw recurrent states over the complete sequence before the native ln_f → coda → ln_f → lm_head readout. Comparisons use the same-depth baseline and original fixed Logits policy (h1, 0.5).

| Task | Depth | Primary metric | Questions | Baseline % | Logits % | Hidden % | Hidden−baseline pp | Hidden−Logits pp | Paper baseline→Hidden % |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ARC-Easy | 16 | acc_norm | 2,376 | 64.39 | 66.33 | 64.65 | +0.25 | -1.68 | 64.14→64.35 |
| ARC-Easy | 32 | acc_norm | 2,376 | 64.27 | 65.28 | 64.18 | -0.08 | -1.09 | 64.18→64.52 |
| ARC-Challenge | 16 | acc_norm | 1,172 | 37.03 | 40.02 | 38.40 | +1.37 | -1.62 | 37.37→37.97 |
| ARC-Challenge | 32 | acc_norm | 1,172 | 37.88 | 39.85 | 38.57 | +0.68 | -1.28 | 37.97→38.74 |
| SciQ | 16 | acc | 1,000 | 93.20 | 93.10 | 93.20 | +0.00 | +0.10 | 90.00→90.70 |
| SciQ | 32 | acc | 1,000 | 93.10 | 92.70 | 93.00 | -0.10 | +0.30 | 90.30→91.00 |
| PIQA | 16 | acc_norm | 1,838 | 75.84 | 76.17 | 75.14 | -0.71 | -1.03 | 75.19→75.08 |
| PIQA | 32 | acc_norm | 1,838 | 75.84 | 76.44 | 75.46 | -0.38 | -0.98 | 75.52→75.57 |

Paper scores come from [Table 3(a), PDF page 7](https://arxiv.org/html/2610.02185v1#S4.T3), with references and strengths from [Table A4, page 18](https://arxiv.org/html/2610.02185v1#A2.T4). These are full-budget settings. Table A8's separate R32→R16 experiment uses h6 at R16; the present full-budget R16 setting uses h7.

Primary metrics were fixed before scoring: acc for SciQ and acc_norm for ARC-Easy, ARC-Challenge, and PIQA. The paper does not unambiguously specify these metrics for every task. SciQ prompts include support and retain all 116 empty-support cases. PIQA uses its 1,838 labeled validation examples; the other three tasks use complete test splits. acc_norm divides by the original answer's character count.

SciQ's paper values are around 90%. The next table compares them with both the pre-specified primary acc and the descriptive acc_norm results.

| SciQ depth | Actual acc baseline→Hidden % | Actual acc_norm baseline→Hidden % | Paper baseline→Hidden % |
|---|---:|---:|---:|
| 16 | 93.20→93.20 | 90.10→90.50 | 90.00→90.70 |
| 32 | 93.10→93.00 | 90.30→90.80 | 90.30→91.00 |

| Depth | Mean definition | Baseline % | Logits % | Hidden % | Hidden−baseline pp | Hidden−Logits pp |
|---|---|---:|---:|---:|---:|---:|
| 16 | Frozen primary metrics | 67.62 | 68.90 | 67.84 | +0.23 | -1.06 |
| 16 | All acc (descriptive) | 67.62 | 67.64 | 67.93 | +0.31 | +0.29 |
| 16 | All acc_norm (descriptive) | 66.84 | 68.28 | 67.17 | +0.33 | -1.11 |
| 32 | Frozen primary metrics | 67.77 | 68.57 | 67.80 | +0.03 | -0.76 |
| 32 | All acc (descriptive) | 68.19 | 67.64 | 67.64 | -0.55 | +0.00 |
| 32 | All acc_norm (descriptive) | 67.07 | 68.07 | 67.25 | +0.18 | -0.81 |

Each task receives one quarter of the mean regardless of question count. The following inference uses only the frozen primary metrics.

| Depth | Comparison | Mean change pp | Paired 95% CI | Task wins/ties/losses | Sign-flip p | Holm p (4 tests) |
|---|---|---:|---:|---:|---:|---:|
| 16 | Hidden − Baseline | +0.23 | [-0.29, +0.76] | 2/1/1 | 0.3888 | 0.7776 |
| 16 | Hidden − Logits | -1.06 | [-1.84, -0.28] | 1/0/3 | 0.007665 | 0.03066 |
| 32 | Hidden − Baseline | +0.03 | [-0.51, +0.57] | 1/0/3 | 0.9159 | 0.9159 |
| 32 | Hidden − Logits | -0.76 | [-1.51, -0.01] | 1/0/3 | 0.04431 | 0.1329 |

| Task/depth | Comparison | Change pp | Paired 95% CI | Corrected/regressed | Exact McNemar p | Holm p (16 tests) |
|---|---|---:|---:|---:|---:|---:|
| ARC-Easy/R16 | Hidden − Baseline | +0.25 | [-0.72, +1.22] | 70/64 | 0.666 | 1 |
| ARC-Easy/R16 | Hidden − Logits | -1.68 | [-3.16, -0.21] | 142/182 | 0.0301 | 0.4816 |
| ARC-Easy/R32 | Hidden − Baseline | -0.08 | [-1.14, +0.93] | 77/79 | 0.9362 | 1 |
| ARC-Easy/R32 | Hidden − Logits | -1.09 | [-2.53, +0.29] | 136/162 | 0.1474 | 1 |
| ARC-Challenge/R16 | Hidden − Baseline | +1.37 | [-0.09, +2.82] | 44/28 | 0.07637 | 1 |
| ARC-Challenge/R16 | Hidden − Logits | -1.62 | [-3.84, +0.60] | 81/100 | 0.1808 | 1 |
| ARC-Challenge/R32 | Hidden − Baseline | +0.68 | [-0.77, +2.13] | 44/36 | 0.434 | 1 |
| ARC-Challenge/R32 | Hidden − Logits | -1.28 | [-3.41, +0.77] | 70/85 | 0.2607 | 1 |
| SciQ/R16 | Hidden − Baseline | +0.00 | [-0.80, +0.80] | 9/9 | 1 | 1 |
| SciQ/R16 | Hidden − Logits | +0.10 | [-0.90, +1.10] | 14/13 | 1 | 1 |
| SciQ/R32 | Hidden − Baseline | -0.10 | [-0.90, +0.70] | 7/8 | 1 | 1 |
| SciQ/R32 | Hidden − Logits | +0.30 | [-0.70, +1.30] | 14/11 | 0.69 | 1 |
| PIQA/R16 | Hidden − Baseline | -0.71 | [-1.52, +0.05] | 22/35 | 0.1112 | 1 |
| PIQA/R16 | Hidden − Logits | -1.03 | [-2.29, +0.16] | 56/75 | 0.1155 | 1 |
| PIQA/R32 | Hidden − Baseline | -0.38 | [-1.25, +0.49] | 31/38 | 0.4704 | 1 |
| PIQA/R32 | Hidden − Logits | -0.98 | [-2.23, +0.27] | 59/77 | 0.1446 | 1 |

Paired bootstrap uses 20,000 question resamples, independently within each task for macro means. Macro tests use 200,000 paired sign-flip samples and plus-one two-sided p-values. Each method and contrast starts from seed 1234. Separate Holm families contain 16 task tests and four macro tests. Confidence intervals are pointwise. The four tasks and one initial-state seed are fixed; uncertainty reflects question sampling within those tasks.

| Task/depth | Baseline seconds | Logits seconds | Hidden seconds | Hidden/baseline latency | Hidden/Logits latency |
|---|---:|---:|---:|---:|---:|
| ARC-Easy/R16 | 1.0100 | 1.0438 | 1.0123 | 1.0023× | 0.9698× |
| ARC-Easy/R32 | 1.9807 | 2.0124 | 1.9768 | 0.9981× | 0.9823× |
| ARC-Challenge/R16 | 1.1703 | 1.2096 | 1.1690 | 0.9988× | 0.9664× |
| ARC-Challenge/R32 | 2.2716 | 2.3096 | 2.2715 | 1.0000× | 0.9835× |
| SciQ/R16 | 2.7947 | 2.8773 | 2.7939 | 0.9997× | 0.9710× |
| SciQ/R32 | 5.3797 | 5.4583 | 5.3772 | 0.9995× | 0.9851× |
| PIQA/R16 | 1.1041 | 1.1436 | 1.1006 | 0.9968× | 0.9624× |
| PIQA/R32 | 2.1561 | 2.1990 | 2.1579 | 1.0009× | 0.9813× |

Timing measures each policy independently on the same sampled workload with CUDA synchronization and repeated measurements. Values are medians of aggregate policy seconds, not full-test end-to-end latencies. Joint baseline/Hidden scoring time is excluded from speed comparisons. Hidden keeps one complete coda/head and the baseline's matrix-product FLOPs; storing reference states, sequence-wide vector arithmetic, and memory traffic can still affect measured latency.

Each setting selects three batches of eight candidates, with two warmups and six measurements per policy per batch. Policy timing excludes checkpoint loading, preparation, transfers, precomputed initial-state generation, and serialization. Initial-state generation times are recorded separately.

R16 geometric mean of four task latency ratios: Hidden/baseline 0.9994×; Hidden/Logits 0.9674×.
R32 geometric mean of four task latency ratios: Hidden/baseline 0.9996×; Hidden/Logits 0.9831×.

| Depth | Baseline TFLOPs | Logits TFLOPs | Hidden TFLOPs | Hidden vector operations |
|---|---:|---:|---:|---:|
| 16 | 29.174 | 30.350 | 29.174 | 8,110,080 |
| 32 | 56.353 | 57.528 | 56.353 | 8,110,080 |

Analytic FLOPs count matrix arithmetic for a 512-token prefill, batch size one, and readout at every position, including recurrent injection. The costs also retain the paper's counting convention. Vector operations are listed separately.

The maximum old/new baseline answer-token log-probability difference is 0 (audit tolerance 0.001). Every baseline prediction matches under both metrics. Candidate sums, character normalization, argmax decisions, and summaries were recomputed from saved token scores. Audits verify raw data, tokenization, source snapshots, checkpoint revision, run parameters, and independent costs. Original Logits artifacts remain unchanged; historical ARC-Easy metadata is aligned using recorded evidence without rewriting its manifest.

comparison.json retains both metrics, detailed audits, and source hashes. comparison.csv contains filterable task results. The frozen protocol is in [analysis_plan.json](analysis_plan.json).

The two metrics disagree on the direction of change for: arc_easy/huginn_16/hidden_vs_baseline; arc_easy/huginn_16/hidden_vs_logits; arc_easy/huginn_32/hidden_vs_logits; arc_challenge/huginn_32/hidden_vs_baseline; sciq/huginn_16/hidden_vs_logits; sciq/huginn_32/hidden_vs_baseline; piqa/huginn_16/hidden_vs_logits; piqa/huginn_32/hidden_vs_logits.
