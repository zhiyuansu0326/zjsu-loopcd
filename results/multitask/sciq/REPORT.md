# SciQ: LoopCD-Logits across all seven model configurations

The full test split contains 1000 questions per configuration. Token probabilities and tokenization boundaries for every candidate, question-level predictions, and aggregate results have been rechecked. Raw data, complete prompts, source snapshots, and native forward-pass validation records also passed the audit.

The primary metric was fixed as `acc` before scoring; both metrics are retained. `acc` uses the sum of conditional log likelihoods over the complete answer; `acc_norm` divides this sum by the Unicode character count of the original answer, excluding the added leading space. Paper Table 2 does not explicitly distinguish these metrics for each task, so absolute comparisons with the paper retain protocol uncertainty.

## Reproduction / paper (%)

| Configuration | Shots | Baseline | Fixed | Adaptive | Fixed change (pp) | Adaptive change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Ouro-1.4B R=4 | 0 | 95.90 / 94.70 | 96.20 / 95.30 | 96.10 / 95.20 | +0.30 | +0.20 |
| Ouro-2.6B R=4 | 0 | 96.10 / 94.20 | 96.40 / 94.60 | 96.50 / 94.30 | +0.30 | +0.40 |
| Huginn-0125 R=16 | 0 | 93.20 / 90.30 | 93.10 / 90.90 | 93.40 / 91.20 | -0.10 | +0.20 |
| Huginn-0125 R=32 | 0 | 93.10 / 90.40 | 92.70 / 91.10 | 92.90 / 91.20 | -0.40 | -0.20 |
| Parcae-370M R=8 | 0 | 86.00 / 78.60 | 85.20 / 79.40 | 84.80 / 79.80 | -0.80 | -1.20 |
| Parcae-1.3B R=8 | 0 | 88.10 / 83.70 | 88.90 / 85.80 | 88.50 / 85.30 | +0.80 | +0.40 |
| Looped-Qwen3-4B R=8 | 0 | 96.30 / 95.30 | 96.00 / 95.00 | 96.20 / 95.10 | -0.30 | -0.10 |

## Paired question-level statistics (acc)

| Configuration | Policy | Corrected | Regressed | Answers changed | Net correct | Paired 95% CI (pp) | McNemar p | Holm p |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Ouro-1.4B R=4 | Fixed | 5 | 2 | 7 | +3 | [-0.20, +0.80] | 0.4531 | 1 |
| Ouro-1.4B R=4 | Adaptive | 5 | 3 | 8 | +2 | [-0.30, +0.80] | 0.7266 | 1 |
| Ouro-2.6B R=4 | Fixed | 4 | 1 | 5 | +3 | [-0.10, +0.80] | 0.375 | 1 |
| Ouro-2.6B R=4 | Adaptive | 5 | 1 | 7 | +4 | [+0.00, +0.90] | 0.2188 | 1 |
| Huginn-0125 R=16 | Fixed | 11 | 12 | 26 | -1 | [-1.00, +0.80] | 1 | 1 |
| Huginn-0125 R=16 | Adaptive | 13 | 11 | 26 | +2 | [-0.70, +1.20] | 0.8388 | 1 |
| Huginn-0125 R=32 | Fixed | 10 | 14 | 26 | -4 | [-1.40, +0.60] | 0.5413 | 1 |
| Huginn-0125 R=32 | Adaptive | 11 | 13 | 28 | -2 | [-1.20, +0.80] | 0.8388 | 1 |
| Parcae-370M R=8 | Fixed | 16 | 24 | 47 | -8 | [-2.00, +0.40] | 0.2682 | 1 |
| Parcae-370M R=8 | Adaptive | 20 | 32 | 57 | -12 | [-2.60, +0.20] | 0.1263 | 1 |
| Parcae-1.3B R=8 | Fixed | 16 | 8 | 31 | +8 | [-0.10, +1.80] | 0.1516 | 1 |
| Parcae-1.3B R=8 | Adaptive | 21 | 17 | 48 | +4 | [-0.80, +1.60] | 0.6271 | 1 |
| Looped-Qwen3-4B R=8 | Fixed | 0 | 3 | 5 | -3 | [-0.70, +0.00] | 0.25 | 1 |
| Looped-Qwen3-4B R=8 | Adaptive | 1 | 2 | 4 | -1 | [-0.50, +0.20] | 1 | 1 |

No primary-metric comparison for this task has Holm p<0.05.

Confidence intervals use 20,000 paired question-level bootstrap resamples (seed=1234) without a multiple-comparison adjustment. Exact two-sided McNemar tests receive Holm correction across the 14 primary-metric comparisons within this task. The supplementary metric has a separate family of 14 comparisons. Tasks are separate families, with no cross-task correction. Intervals reflect question-sampling variation only; they exclude uncertainty from initialization seeds, repeated model runs, and evaluation protocols.

## Both metrics and the reference state

| Configuration | Metric | Baseline | Fixed | Adaptive | First full-loop reference |
|---|---|---:|---:|---:|---:|
| Ouro-1.4B R=4 | acc | 959 (95.90%) | 962 (96.20%) | 961 (96.10%) | 583 (58.30%) |
| Ouro-1.4B R=4 | acc_norm | 947 (94.70%) | 953 (95.30%) | 951 (95.10%) | 524 (52.40%) |
| Ouro-2.6B R=4 | acc | 961 (96.10%) | 964 (96.40%) | 965 (96.50%) | 837 (83.70%) |
| Ouro-2.6B R=4 | acc_norm | 943 (94.30%) | 946 (94.60%) | 943 (94.30%) | 737 (73.70%) |
| Huginn-0125 R=16 | acc | 932 (93.20%) | 931 (93.10%) | 934 (93.40%) | 577 (57.70%) |
| Huginn-0125 R=16 | acc_norm | 901 (90.10%) | 906 (90.60%) | 909 (90.90%) | 512 (51.20%) |
| Huginn-0125 R=32 | acc | 931 (93.10%) | 927 (92.70%) | 929 (92.90%) | 577 (57.70%) |
| Huginn-0125 R=32 | acc_norm | 903 (90.30%) | 907 (90.70%) | 913 (91.30%) | 512 (51.20%) |
| Parcae-370M R=8 | acc | 860 (86.00%) | 852 (85.20%) | 848 (84.80%) | 817 (81.70%) |
| Parcae-370M R=8 | acc_norm | 786 (78.60%) | 794 (79.40%) | 799 (79.90%) | 729 (72.90%) |
| Parcae-1.3B R=8 | acc | 881 (88.10%) | 889 (88.90%) | 885 (88.50%) | 800 (80.00%) |
| Parcae-1.3B R=8 | acc_norm | 831 (83.10%) | 856 (85.60%) | 854 (85.40%) | 729 (72.90%) |
| Looped-Qwen3-4B R=8 | acc | 963 (96.30%) | 960 (96.00%) | 962 (96.20%) | 960 (96.00%) |
| Looped-Qwen3-4B R=8 | acc_norm | 953 (95.30%) | 950 (95.00%) | 950 (95.00%) | 935 (93.50%) |

Comparisons with opposite directions of change between the two metrics: Huginn-0125 R=16 Fixed; Huginn-0125 R=32 Fixed; Huginn-0125 R=32 Adaptive; Parcae-370M R=8 Fixed; Parcae-370M R=8 Adaptive. These results do not support a claim of consistent improvement.

### The paper's SciQ scoring metric remains uncertain

Ouro-1.4B baseline / fixed / adaptive: `acc` is 95.90 / 96.20 / 96.10%, `acc_norm` is 94.70 / 95.30 / 95.10%, and the paper reports 94.70 / 95.30 / 95.20%. For this configuration, character-normalized scores are closer to the paper. Numerical proximity is only a clue to the authors' possible scoring metric, not confirmation of their evaluation protocol.
Across all 21 published scores for the seven configurations, the mean absolute difference is 2.581 pp for `acc` and 0.129 pp for `acc_norm`. These distances only describe agreement with the paper; they are not used to select a metric or infer significance. The primary metric remains `acc`, as fixed before scoring. Both metrics are reported in full, and the primary metric has not been reassigned based on agreement with the paper.

## Protocol and limitations

- Fixed guidance strength is 0.5. The adaptive cap is 0.5 for Huginn and 1.0 for the other models. Adaptive uncertainty is computed over the full vocabulary. The reference is the first complete update followed by the full coda/head; all policies share one loop trajectory.
- Shot counts follow paper Table A2, with seed 1234. Nonzero-shot examples are sampled from the training set as questions are processed, without using test labels to select examples. SciQ prompts include support, and PIQA uses the labeled validation split.
- Huginn receives one BOS token through its official interface; the other models receive no extra BOS token. No chat templates are used. Huginn/Parcae random initial states are fixed by question ID, with positional noise shared across candidates. The paper does not publish an identical seed policy.
- For the new tasks, context is truncated from the left only when the model window is insufficient; all answer tokens and prediction positions must be preserved. The table below reports the effect for each configuration. For long prompts, truncation may retain fewer examples than the nominal shot count.
- Checkpoint manifests, small model files, tokenizers, and native source files are verified with SHA256. Large weights are checked by file size in this audit, with hashes referenced from download-time verification; they are not claimed to have been fully rehashed.
- Joint execution time cannot estimate the throughput benefit of running a policy alone. These results cover full-depth LoopCD-Logits on the evaluated tasks only; they do not validate hidden-state guidance, reduced-depth evaluation, or speed claims.

| Configuration | dtype / matmul | Truncated questions / candidates | Maximum input tokens | Maximum batch/single token logp difference |
|---|---|---:|---:|---:|
| Ouro-1.4B R=4 | float32 / high | 0 / 0 | 633 | 0.00077 |
| Ouro-2.6B R=4 | float32 / high | 0 / 0 | 633 | 0.000727 |
| Huginn-0125 R=16 | float32 / highest | 0 / 0 | 668 | 1.14e-05 |
| Huginn-0125 R=32 | float32 / highest | 0 / 0 | 668 | 1.24e-05 |
| Parcae-370M R=8 | float32 / highest | 0 / 0 | 630 | 8.11e-06 |
| Parcae-1.3B R=8 | float32 / highest | 0 / 0 | 630 | 1.48e-05 |
| Looped-Qwen3-4B R=8 | float32 / highest | 0 / 0 | 656 | 1.24e-05 |

See comparison.json for source paths, candidate-level truncation inventories, paired statistics for both metrics, and full audit evidence for each configuration. Regenerate with `venv/bin/python scripts/analyze.py --task sciq`.

Sources: [paper Tables 2 / A2 / A4](https://arxiv.org/html/2610.02185v1) and the [pinned lm-eval implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). Paper reference values and supporting evidence for the new tasks are stored in configs/paper_mc_reference.json; ARC-Easy reference values remain in configs/arc_all_models.json.
