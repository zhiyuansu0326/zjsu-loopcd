# PIQA: LoopCD-Logits across all seven model configurations

The full validation split contains 1838 questions per configuration. Token probabilities and tokenization boundaries for every candidate, question-level predictions, and aggregate results have been rechecked. Raw data, complete prompts, source snapshots, and native forward-pass validation records also passed the audit.

The primary metric was fixed as `acc_norm` before scoring; both metrics are retained. `acc` uses the sum of conditional log likelihoods over the complete answer; `acc_norm` divides this sum by the Unicode character count of the original answer, excluding the added leading space. Paper Table 2 does not explicitly distinguish these metrics for each task, so absolute comparisons with the paper retain protocol uncertainty.

## Reproduction / paper (%)

| Configuration | Shots | Baseline | Fixed | Adaptive | Fixed change (pp) | Adaptive change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Ouro-1.4B R=4 | 0 | 78.45 / 78.51 | 78.94 / 79.05 | 78.94 / 78.73 | +0.49 | +0.49 |
| Ouro-2.6B R=4 | 0 | 80.14 / 80.25 | 80.63 / 80.74 | 80.79 / 80.69 | +0.49 | +0.65 |
| Huginn-0125 R=16 | 0 | 75.84 / 75.52 | 76.17 / 74.54 | 75.68 / 74.86 | +0.33 | -0.16 |
| Huginn-0125 R=32 | 0 | 75.84 / 75.79 | 76.44 / 74.97 | 76.39 / 75.79 | +0.60 | +0.54 |
| Parcae-370M R=8 | 0 | 67.14 / 66.97 | 67.90 / 68.01 | 67.68 / 67.41 | +0.76 | +0.54 |
| Parcae-1.3B R=8 | 0 | 73.01 / 72.03 | 72.80 / 71.44 | 72.03 / 70.35 | -0.22 | -0.98 |
| Looped-Qwen3-4B R=8 | 0 | 75.30 / 75.19 | 74.81 / 74.65 | 75.08 / 75.14 | -0.49 | -0.22 |

## Paired question-level statistics (acc_norm)

| Configuration | Policy | Corrected | Regressed | Answers changed | Net correct | Paired 95% CI (pp) | McNemar p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Ouro-1.4B R=4 | Fixed | 32 | 23 | 55 | +9 | [-0.33, +1.31] | 0.2806 | 1 |
| Ouro-1.4B R=4 | Adaptive | 45 | 36 | 81 | +9 | [-0.49, +1.47] | 0.3742 | 1 |
| Ouro-2.6B R=4 | Fixed | 31 | 22 | 53 | +9 | [-0.27, +1.25] | 0.2717 | 1 |
| Ouro-2.6B R=4 | Adaptive | 51 | 39 | 90 | +12 | [-0.38, +1.69] | 0.2461 | 1 |
| Huginn-0125 R=16 | Fixed | 62 | 56 | 118 | +6 | [-0.82, +1.47] | 0.6455 | 1 |
| Huginn-0125 R=16 | Adaptive | 53 | 56 | 109 | -3 | [-1.31, +0.92] | 0.8482 | 1 |
| Huginn-0125 R=32 | Fixed | 68 | 57 | 125 | +11 | [-0.60, +1.80] | 0.3712 | 1 |
| Huginn-0125 R=32 | Adaptive | 57 | 47 | 104 | +10 | [-0.54, +1.63] | 0.3776 | 1 |
| Parcae-370M R=8 | Fixed | 46 | 32 | 78 | +14 | [-0.16, +1.69] | 0.1405 | 1 |
| Parcae-370M R=8 | Adaptive | 67 | 57 | 124 | +10 | [-0.65, +1.74] | 0.4191 | 1 |
| Parcae-1.3B R=8 | Fixed | 36 | 40 | 76 | -4 | [-1.14, +0.71] | 0.731 | 1 |
| Parcae-1.3B R=8 | Adaptive | 53 | 71 | 124 | -18 | [-2.18, +0.22] | 0.1265 | 1 |
| Looped-Qwen3-4B R=8 | Fixed | 24 | 33 | 57 | -9 | [-1.31, +0.33] | 0.2892 | 1 |
| Looped-Qwen3-4B R=8 | Adaptive | 38 | 42 | 80 | -4 | [-1.14, +0.76] | 0.7376 | 1 |

No primary-metric comparison for this task has Holm p<0.05.

Confidence intervals use 20,000 paired question-level bootstrap resamples (seed=1234) without a multiple-comparison adjustment. Exact two-sided McNemar tests receive Holm correction across the 14 primary-metric comparisons within this task. The supplementary metric has a separate family of 14 comparisons. Tasks are separate families, with no cross-task correction. Intervals reflect question-sampling variation only; they exclude uncertainty from initialization seeds, repeated model runs, and evaluation protocols.

## Both metrics and the reference state

| Configuration | Metric | Baseline | Fixed | Adaptive | First full-loop reference |
|---|---|---:|---:|---:|---:|
| Ouro-1.4B R=4 | acc | 1444 (78.56%) | 1450 (78.89%) | 1448 (78.78%) | 1238 (67.36%) |
| Ouro-1.4B R=4 | acc_norm | 1442 (78.45%) | 1451 (78.94%) | 1451 (78.94%) | 1218 (66.27%) |
| Ouro-2.6B R=4 | acc | 1468 (79.87%) | 1464 (79.65%) | 1465 (79.71%) | 1323 (71.98%) |
| Ouro-2.6B R=4 | acc_norm | 1473 (80.14%) | 1482 (80.63%) | 1485 (80.79%) | 1319 (71.76%) |
| Huginn-0125 R=16 | acc | 1382 (75.19%) | 1366 (74.32%) | 1374 (74.76%) | 1077 (58.60%) |
| Huginn-0125 R=16 | acc_norm | 1394 (75.84%) | 1400 (76.17%) | 1391 (75.68%) | 1029 (55.98%) |
| Huginn-0125 R=32 | acc | 1386 (75.41%) | 1370 (74.54%) | 1379 (75.03%) | 1077 (58.60%) |
| Huginn-0125 R=32 | acc_norm | 1394 (75.84%) | 1405 (76.44%) | 1404 (76.39%) | 1029 (55.98%) |
| Parcae-370M R=8 | acc | 1244 (67.68%) | 1238 (67.36%) | 1241 (67.52%) | 1195 (65.02%) |
| Parcae-370M R=8 | acc_norm | 1234 (67.14%) | 1248 (67.90%) | 1244 (67.68%) | 1187 (64.58%) |
| Parcae-1.3B R=8 | acc | 1316 (71.60%) | 1315 (71.55%) | 1319 (71.76%) | 1276 (69.42%) |
| Parcae-1.3B R=8 | acc_norm | 1342 (73.01%) | 1338 (72.80%) | 1324 (72.03%) | 1288 (70.08%) |
| Looped-Qwen3-4B R=8 | acc | 1377 (74.92%) | 1361 (74.05%) | 1353 (73.61%) | 1349 (73.39%) |
| Looped-Qwen3-4B R=8 | acc_norm | 1384 (75.30%) | 1375 (74.81%) | 1380 (75.08%) | 1371 (74.59%) |

Comparisons with opposite directions of change between the two metrics: Ouro-2.6B R=4 Fixed; Ouro-2.6B R=4 Adaptive; Huginn-0125 R=16 Fixed; Huginn-0125 R=32 Fixed; Huginn-0125 R=32 Adaptive; Parcae-370M R=8 Fixed; Parcae-370M R=8 Adaptive; Parcae-1.3B R=8 Adaptive. These results do not support a claim of consistent improvement.

## Protocol and limitations

- Fixed guidance strength is 0.5. The adaptive cap is 0.5 for Huginn and 1.0 for the other models. Adaptive uncertainty is computed over the full vocabulary. The reference is the first complete update followed by the full coda/head; all policies share one loop trajectory.
- Shot counts follow paper Table A2, with seed 1234. Nonzero-shot examples are sampled from the training set as questions are processed, without using test labels to select examples. SciQ prompts include support, and PIQA uses the labeled validation split.
- Huginn receives one BOS token through its official interface; the other models receive no extra BOS token. No chat templates are used. Huginn/Parcae random initial states are fixed by question ID, with positional noise shared across candidates. The paper does not publish an identical seed policy.
- For the new tasks, context is truncated from the left only when the model window is insufficient; all answer tokens and prediction positions must be preserved. The table below reports the effect for each configuration. For long prompts, truncation may retain fewer examples than the nominal shot count.
- Checkpoint manifests, small model files, tokenizers, and native source files are verified with SHA256. Large weights are checked by file size in this audit, with hashes referenced from download-time verification; they are not claimed to have been fully rehashed.
- Joint execution time cannot estimate the throughput benefit of running a policy alone. These results cover full-depth LoopCD-Logits on the evaluated tasks only; they do not validate hidden-state guidance, reduced-depth evaluation, or speed claims.

| Configuration | dtype / matmul | Truncated questions / candidates | Maximum input tokens | Maximum batch/single token logp difference |
|---|---|---:|---:|---:|
| Ouro-1.4B R=4 | float32 / highest | 0 / 0 | 241 | 2.86e-05 |
| Ouro-2.6B R=4 | float32 / highest | 0 / 0 | 241 | 1.53e-05 |
| Huginn-0125 R=16 | float32 / highest | 0 / 0 | 254 | 2.36e-05 |
| Huginn-0125 R=32 | float32 / highest | 0 / 0 | 254 | 4.1e-05 |
| Parcae-370M R=8 | float32 / highest | 0 / 0 | 259 | 2.17e-05 |
| Parcae-1.3B R=8 | float32 / highest | 0 / 0 | 259 | 6.59e-05 |
| Looped-Qwen3-4B R=8 | float32 / highest | 0 / 0 | 243 | 0.000142 |

See comparison.json for source paths, candidate-level truncation inventories, paired statistics for both metrics, and full audit evidence for each configuration. Regenerate with `venv/bin/python scripts/analyze.py --task piqa`.

Sources: [paper Tables 2 / A2 / A4](https://arxiv.org/html/2610.02185v1) and the [pinned lm-eval implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). Paper reference values and supporting evidence for the new tasks are stored in configs/paper_mc_reference.json; ARC-Easy reference values remain in configs/arc_all_models.json.
