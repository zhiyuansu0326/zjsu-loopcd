# ARC-Challenge: LoopCD-Logits across all seven model configurations

The full test split contains 1172 questions per configuration. Token probabilities and tokenization boundaries for every candidate, question-level predictions, and aggregate results have been rechecked. Raw data, complete prompts, source snapshots, and native forward-pass validation records also passed the audit.

The primary metric was fixed as `acc_norm` before scoring; both metrics are retained. `acc` uses the sum of conditional log likelihoods over the complete answer; `acc_norm` divides this sum by the Unicode character count of the original answer, excluding the added leading space. Paper Table 2 does not explicitly distinguish these metrics for each task, so absolute comparisons with the paper retain protocol uncertainty.

## Reproduction / paper (%)

| Configuration | Shots | Baseline | Fixed | Adaptive | Fixed change (pp) | Adaptive change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Ouro-1.4B R=4 | 25 | 60.24 / 60.41 | 61.86 / 62.29 | 61.60 / 61.69 | +1.62 | +1.37 |
| Ouro-2.6B R=4 | 25 | 66.55 / 66.21 | 69.03 / 68.60 | 69.37 / 69.71 | +2.47 | +2.82 |
| Huginn-0125 R=16 | 0 | 37.03 / 37.63 | 40.02 / 39.76 | 39.76 / 39.33 | +2.99 | +2.73 |
| Huginn-0125 R=32 | 0 | 37.88 / 37.63 | 39.85 / 39.51 | 38.82 / 39.42 | +1.96 | +0.94 |
| Parcae-370M R=8 | 25 | 32.42 / 32.59 | 35.58 / 35.15 | 36.26 / 36.95 | +3.16 | +3.84 |
| Parcae-1.3B R=8 | 25 | 40.36 / 41.21 | 43.77 / 43.86 | 44.45 / 45.31 | +3.41 | +4.10 |
| Looped-Qwen3-4B R=8 | 25 | 62.46 / 62.12 | 62.80 / 62.54 | 63.82 / 64.08 | +0.34 | +1.37 |

## Paired question-level statistics (acc_norm)

| Configuration | Policy | Corrected | Regressed | Answers changed | Net correct | Paired 95% CI (pp) | McNemar p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Ouro-1.4B R=4 | Fixed | 43 | 24 | 88 | +19 | [+0.26, +2.99] | 0.02712 | 0.1627 |
| Ouro-1.4B R=4 | Adaptive | 51 | 35 | 122 | +16 | [-0.17, +2.90] | 0.1052 | 0.3201 |
| Ouro-2.6B R=4 | Fixed | 50 | 21 | 87 | +29 | [+1.11, +3.92] | 0.0007668 | 0.007668 |
| Ouro-2.6B R=4 | Adaptive | 65 | 32 | 122 | +33 | [+1.19, +4.44] | 0.00105 | 0.009446 |
| Huginn-0125 R=16 | Fixed | 103 | 68 | 258 | +35 | [+0.77, +5.12] | 0.009122 | 0.07298 |
| Huginn-0125 R=16 | Adaptive | 90 | 58 | 219 | +32 | [+0.68, +4.78] | 0.01058 | 0.07408 |
| Huginn-0125 R=32 | Fixed | 88 | 65 | 245 | +23 | [-0.17, +4.01] | 0.07496 | 0.3201 |
| Huginn-0125 R=32 | Adaptive | 70 | 59 | 211 | +11 | [-0.94, +2.82] | 0.3787 | 0.7574 |
| Parcae-370M R=8 | Fixed | 62 | 25 | 145 | +37 | [+1.62, +4.69] | 9.061e-05 | 0.0009967 |
| Parcae-370M R=8 | Adaptive | 84 | 39 | 213 | +45 | [+2.05, +5.63] | 6.1e-05 | 0.000732 |
| Parcae-1.3B R=8 | Fixed | 64 | 24 | 135 | +40 | [+1.88, +4.95] | 2.373e-05 | 0.0003322 |
| Parcae-1.3B R=8 | Adaptive | 90 | 42 | 202 | +48 | [+2.22, +5.97] | 3.574e-05 | 0.0004646 |
| Looped-Qwen3-4B R=8 | Fixed | 28 | 24 | 72 | +4 | [-0.85, +1.54] | 0.6778 | 0.7574 |
| Looped-Qwen3-4B R=8 | Adaptive | 41 | 25 | 87 | +16 | [+0.00, +2.73] | 0.06402 | 0.3201 |

Primary-metric changes with Holm p<0.05 for this task: Ouro-2.6B R=4 Fixed (+2.47 pp); Ouro-2.6B R=4 Adaptive (+2.82 pp); Parcae-370M R=8 Fixed (+3.16 pp); Parcae-370M R=8 Adaptive (+3.84 pp); Parcae-1.3B R=8 Fixed (+3.41 pp); Parcae-1.3B R=8 Adaptive (+4.10 pp).

Confidence intervals use 20,000 paired question-level bootstrap resamples (seed=1234) without a multiple-comparison adjustment. Exact two-sided McNemar tests receive Holm correction across the 14 primary-metric comparisons within this task. The supplementary metric has a separate family of 14 comparisons. Tasks are separate families, with no cross-task correction. Intervals reflect question-sampling variation only; they exclude uncertainty from initialization seeds, repeated model runs, and evaluation protocols.

## Both metrics and the reference state

| Configuration | Metric | Baseline | Fixed | Adaptive | First full-loop reference |
|---|---|---:|---:|---:|---:|
| Ouro-1.4B R=4 | acc | 657 (56.06%) | 674 (57.51%) | 674 (57.51%) | 434 (37.03%) |
| Ouro-1.4B R=4 | acc_norm | 706 (60.24%) | 725 (61.86%) | 722 (61.60%) | 451 (38.48%) |
| Ouro-2.6B R=4 | acc | 727 (62.03%) | 760 (64.85%) | 765 (65.27%) | 530 (45.22%) |
| Ouro-2.6B R=4 | acc_norm | 780 (66.55%) | 809 (69.03%) | 813 (69.37%) | 560 (47.78%) |
| Huginn-0125 R=16 | acc | 384 (32.76%) | 422 (36.01%) | 417 (35.58%) | 220 (18.77%) |
| Huginn-0125 R=16 | acc_norm | 434 (37.03%) | 469 (40.02%) | 466 (39.76%) | 283 (24.15%) |
| Huginn-0125 R=32 | acc | 412 (35.15%) | 429 (36.60%) | 427 (36.43%) | 220 (18.77%) |
| Huginn-0125 R=32 | acc_norm | 444 (37.88%) | 467 (39.85%) | 455 (38.82%) | 283 (24.15%) |
| Parcae-370M R=8 | acc | 360 (30.72%) | 384 (32.76%) | 399 (34.04%) | 266 (22.70%) |
| Parcae-370M R=8 | acc_norm | 380 (32.42%) | 417 (35.58%) | 425 (36.26%) | 331 (28.24%) |
| Parcae-1.3B R=8 | acc | 449 (38.31%) | 485 (41.38%) | 501 (42.75%) | 346 (29.52%) |
| Parcae-1.3B R=8 | acc_norm | 473 (40.36%) | 513 (43.77%) | 521 (44.45%) | 370 (31.57%) |
| Looped-Qwen3-4B R=8 | acc | 679 (57.94%) | 689 (58.79%) | 701 (59.81%) | 581 (49.57%) |
| Looped-Qwen3-4B R=8 | acc_norm | 732 (62.46%) | 736 (62.80%) | 748 (63.82%) | 627 (53.50%) |

## Protocol and limitations

- Fixed guidance strength is 0.5. The adaptive cap is 0.5 for Huginn and 1.0 for the other models. Adaptive uncertainty is computed over the full vocabulary. The reference is the first complete update followed by the full coda/head; all policies share one loop trajectory.
- Shot counts follow paper Table A2, with seed 1234. Nonzero-shot examples are sampled from the training set as questions are processed, without using test labels to select examples. SciQ prompts include support, and PIQA uses the labeled validation split.
- Huginn receives one BOS token through its official interface; the other models receive no extra BOS token. No chat templates are used. Huginn/Parcae random initial states are fixed by question ID, with positional noise shared across candidates. The paper does not publish an identical seed policy.
- For the new tasks, context is truncated from the left only when the model window is insufficient; all answer tokens and prediction positions must be preserved. The table below reports the effect for each configuration. For long prompts, truncation may retain fewer examples than the nominal shot count.
- Checkpoint manifests, small model files, tokenizers, and native source files are verified with SHA256. Large weights are checked by file size in this audit, with hashes referenced from download-time verification; they are not claimed to have been fully rehashed.
- Joint execution time cannot estimate the throughput benefit of running a policy alone. These results cover full-depth LoopCD-Logits on the evaluated tasks only; they do not validate hidden-state guidance, reduced-depth evaluation, or speed claims.

| Configuration | dtype / matmul | Truncated questions / candidates | Maximum input tokens | Maximum batch/single token logp difference |
|---|---|---:|---:|---:|
| Ouro-1.4B R=4 | float32 / high | 0 / 0 | 1311 | 0 |
| Ouro-2.6B R=4 | float32 / high | 0 / 0 | 1311 | 0 |
| Huginn-0125 R=16 | float32 / highest | 0 / 0 | 176 | 9.06e-05 |
| Huginn-0125 R=32 | float32 / highest | 0 / 0 | 176 | 6.53e-05 |
| Parcae-370M R=8 | float32 / high | 0 / 0 | 1311 | 0 |
| Parcae-1.3B R=8 | float32 / highest | 0 / 0 | 1311 | 0.00013 |
| Looped-Qwen3-4B R=8 | float32 / highest | 0 / 0 | 1279 | 8.11e-05 |

See comparison.json for source paths, candidate-level truncation inventories, paired statistics for both metrics, and full audit evidence for each configuration. Regenerate with `venv/bin/python scripts/analyze.py --task arc_challenge`.

Sources: [paper Tables 2 / A2 / A4](https://arxiv.org/html/2610.02185v1) and the [pinned lm-eval implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). Paper reference values and supporting evidence for the new tasks are stored in configs/paper_mc_reference.json; ARC-Easy reference values remain in configs/arc_all_models.json.
