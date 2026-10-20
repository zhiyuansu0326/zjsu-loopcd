# LoopCD: Seven Model Configurations × Four Tasks

A reproduction of **LoopCD-Logits** from Liu et al., [*Decoding Looped Transformers Better for (Almost) Free*](https://arxiv.org/html/2610.02185v1).

All **28 task/model configurations are complete and independently audited**: 7 configurations across ARC-Easy, ARC-Challenge, SciQ, and PIQA. See the [four-task report](results/multitask/REPORT.md), [JSON](results/multitask/comparison.json), and [CSV](results/multitask/comparison.csv). Original scores and execution source snapshots are preserved.

Each full-depth run evaluates baseline decoding, fixed-strength LoopCD, and adaptive LoopCD using the same model trajectory. The first-loop reference is also saved.

The full-depth evaluation covers **44,702 question evaluations and 153,048 candidate answers**. Each run includes scores, predictions, configuration, and validation records.

The main matrix evaluates multiple-choice answer scoring at each model's full recurrent depth.

## Reproduction checklist

- [x] LoopCD-Logits with fixed and adaptive guidance.
- [x] Full-depth evaluation: 7 configurations × ARC-Easy, ARC-Challenge, SciQ, and PIQA (28 runs).
- [x] Unified data preparation, evaluation, and auditing; saved scores and statistics.
- [x] Huginn × all four tasks at half depth: fixed paper parameters, paired task/mean comparisons, FLOPs, and standalone H200 timing.
- [ ] LoopCD-Hidden.
- [ ] Remaining multiple-choice benchmarks, math/code generation, and Thinking models.
- [ ] Reduced-depth accuracy and FLOPs comparisons across the remaining model/task settings.
- [ ] Standalone inference overhead and speed benchmarks for the other model families.

## Results

Each cell is **baseline / fixed / adaptive**, in accuracy percent. The primary metric is `acc_norm` for ARC-Easy, ARC-Challenge, and PIQA, and `acc` for SciQ. Detailed results include both metrics.

| Model configuration | ARC-Easy | ARC-Challenge | SciQ | PIQA |
|---|---:|---:|---:|---:|
| Ouro-1.4B R=4 | 84.09 / 84.39 / 84.13 | 60.24 / 61.86 / 61.60 | 95.90 / 96.20 / 96.10 | 78.45 / 78.94 / 78.94 |
| Ouro-2.6B R=4 | 88.17 / 88.93 / 88.72 | 66.55 / 69.03 / 69.37 | 96.10 / 96.40 / 96.50 | 80.14 / 80.63 / 80.79 |
| Huginn-0125 R=16 | 64.39 / 66.33 / 66.25 | 37.03 / 40.02 / 39.76 | 93.20 / 93.10 / 93.40 | 75.84 / 76.17 / 75.68 |
| Huginn-0125 R=32 | 64.27 / 65.28 / 65.70 | 37.88 / 39.85 / 38.82 | 93.10 / 92.70 / 92.90 | 75.84 / 76.44 / 76.39 |
| Parcae-370M R=8 | 54.59 / 54.59 / 54.50 | 32.42 / 35.58 / 36.26 | 86.00 / 85.20 / 84.80 | 67.14 / 67.90 / 67.68 |
| Parcae-1.3B R=8 | 64.90 / 66.16 / 65.49 | 40.36 / 43.77 / 44.45 | 88.10 / 88.90 / 88.50 | 73.01 / 72.80 / 72.03 |
| Looped-Qwen3-4B R=8 | 79.38 / 79.25 / 79.00 | 62.46 / 62.80 / 63.82 | 96.30 / 96.00 / 96.20 | 75.30 / 74.81 / 75.08 |

On ARC-Challenge, both guidance methods improve every configuration's primary metric. Improvements are mixed on the other tasks. The reports include paired confidence intervals and exact McNemar tests, with Holm correction across 14 comparisons within each task and metric.

Full comparisons with the paper, both metrics, and statistical results:

- [ARC-Easy](results/multitask/arc_easy/REPORT.md)
- [ARC-Challenge](results/multitask/arc_challenge/REPORT.md)
- [SciQ](results/multitask/sciq/REPORT.md)
- [PIQA](results/multitask/piqa/REPORT.md)
- [All results as JSON](results/multitask/comparison.json) · [CSV](results/multitask/comparison.csv)

### SciQ character-normalized results

SciQ uses `acc` as its primary metric. The supplementary `acc_norm` scores below are closer to the paper's reported values: their mean absolute difference is 0.129 percentage points, compared with 2.581 for `acc`.

| Model configuration | SciQ `acc_norm`: baseline / fixed / adaptive |
|---|---:|
| Ouro-1.4B R=4 | 94.70 / 95.30 / 95.10 |
| Ouro-2.6B R=4 | 94.30 / 94.60 / 94.30 |
| Huginn-0125 R=16 | 90.10 / 90.60 / 90.90 |
| Huginn-0125 R=32 | 90.30 / 90.70 / 91.30 |
| Parcae-370M R=8 | 78.60 / 79.40 / 79.90 |
| Parcae-1.3B R=8 | 83.10 / 85.60 / 85.40 |
| Looped-Qwen3-4B R=8 | 95.30 / 95.00 / 95.00 |

## Half-depth results: Huginn × four tasks

The Table A8 setting uses **R=16, reference iteration 5, and fixed strength 0.5**
on all four tasks. The standard R=16 row above uses reference iteration 1.
The extension reuses the completed ARC-Challenge result and adds the other three
tasks without tuning. It covers all **6,386 questions**, with the same 0-shot
prompts, FP32 precision, and per-question initial states as the saved baselines.

| Task | R32 baseline (%) | R16 baseline (%) | R16 LoopCD (%) | Change vs R32 (pp) |
|---|---:|---:|---:|---:|
| ARC-Easy | 64.27 | 64.39 | 65.32 | +1.05 |
| ARC-Challenge | 37.88 | 37.03 | 39.25 | +1.37 |
| SciQ | 93.10 | 93.20 | 93.40 | +0.30 |
| PIQA | 75.84 | 75.84 | 76.28 | +0.44 |
| **Equal-task mean** | **67.77** | **67.62** | **68.56** | **+0.79** |

The primary metrics remain `acc_norm` for ARC-Easy, ARC-Challenge, and PIQA,
and `acc` for SciQ. Each task has equal weight in the mean. Guided R16 improves
all four primary scores against both baselines. Against R32, the mean gain is
**+0.79 pp**, with a stratified paired 95% confidence interval of **[+0.15, +1.43]
pp** and a two-sided sign-flip **Holm-adjusted p = 0.0345**. Against R16, the gain
is **+0.94 pp**, with **[+0.34, +1.56] pp** and **p = 0.0085**. Holm correction
covers the three mean contrasts; no individual task contrast passes the separate
12-comparison Holm correction. These intervals resample questions within the
four fixed tasks.

The gain depends on the scoring metric: using `acc` for every task gives a mean
change of **−0.08 pp** against R32, while using `acc_norm` throughout gives
**+0.89 pp**. Both secondary views and every task result are included in the
[four-task half-depth report](results/reduced_depth/REPORT.md).

Guided R16 uses **46.14% fewer analytic matrix-product FLOPs** than R32, including
its extra readout and the recurrent injection projection, at a 512-token prefill.
Standalone H200 scoring gives **1.87×–1.89×** per-task speedups and a **1.87×**
geometric mean. Each task measures three fixed batches of eight candidates over
six repetitions; loading, tokenization, and initial-state generation are excluded.
The report includes both FLOP accounting conventions, sampled timings, the exact
baseline replay checks, and the [frozen analysis plan](results/reduced_depth/analysis_plan.json).

## Models and evaluation protocol

“Seven models” means **seven configurations across four architecture families and six checkpoints**. Huginn uses the same checkpoint at two recurrent depths. Looped-Qwen uses the original `Qwen/Qwen3-4B` checkpoint with the loop construction described below.

| CLI model ID | Checkpoint | Recurrent depth |
|---|---|---:|
| `ouro_14` | `ByteDance/Ouro-1.4B` | 4 |
| `ouro_26` | `ByteDance/Ouro-2.6B` | 4 |
| `huginn_16` | `tomg-group-umd/huginn-0125` | 16 |
| `huginn_32` | `tomg-group-umd/huginn-0125` | 32 |
| `parcae_370` | `SandyResearch/parcae-370m` | 8 |
| `parcae_13` | `SandyResearch/parcae-1.3b` | 8 |
| `looped_qwen` | `Qwen/Qwen3-4B` | 8 |

| CLI task ID | Split | Questions | Few-shot examples | Primary metric |
|---|---|---:|---|---|
| `arc_easy` | test | 2,376 | Ouro: 8; others: 0 | `acc_norm` |
| `arc_challenge` | test | 1,172 | Huginn: 0; others: 25 | `acc_norm` |
| `sciq` | test | 1,000 | 0, with the original support paragraph | `acc` |
| `piqa` | validation | 1,838 | 0 | `acc_norm` |

The shared [model registry](configs/models.json) and [task registry](configs/tasks.json) drive data preparation, evaluation, and analysis.

- **Answer scoring:** score the entire candidate answer, preceded by one space, using teacher-forced conditional log probabilities and the task's plain-text prompt.
- **Metrics:** `acc` selects the largest sum of answer-token log probabilities. `acc_norm` divides that sum by the original answer's Unicode character count (`len(answer)`), excluding the added space.
- **Guidance:** with final-loop logits `zR` and first-complete-loop logits `z1`, compute `zR + ω × (zR − z1)`. Fixed guidance uses `ω = 0.5`. Adaptive guidance uses `ω = cap × (1 − (p_top1 − p_top2))`, where the probabilities come from the final-loop distribution over the full vocabulary. The cap is `0.5` for Huginn and `1.0` otherwise.
- **Reproducibility:** seed `1234`, pinned checkpoint revisions, pinned data hashes, and task definitions from lm-evaluation-harness commit `d6de81643928d653435c431bae19945d41d32520`. Few-shot examples come from training data.
- **Numerics:** FP32 inference and scoring, with matrix-multiplication precision recorded per run. Seven new task/model pairs use `highest` to meet the batch/single consistency threshold. See the [precision decision record](results/multitask/precision_decisions.json).

All completed runs fit the model context windows without truncation. The general evaluator supports left truncation of context while preserving the complete answer.

## Installation and data

The reports and completed run records are available directly in the repository. The Python environment, full checkpoint weights, raw/prepared datasets, and download caches are not committed. Recreate those assets before rerunning evaluations or performing a full local audit.

The original environment used Python **3.12.9**, PyTorch **2.6.0+cu124**, and Transformers **4.56.2**. On the original workspace, use the existing `venv/`. For a fresh clone on a compatible Linux/NVIDIA system:

```bash
git clone https://github.com/zhiyuansu0326/zjsu-loopcd.git
cd zjsu-loopcd

python3.12 -m venv venv
venv/bin/python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
venv/bin/python -m pip install -r requirements.txt \
  -c requirements.freeze.txt

make check-env
make prepare-data TASK=all
make download-models
make test
```

The model downloads require approximately **37 GB**, in addition to environment and data storage. Download scripts verify pinned model revisions and file hashes. Preparation verifies dataset hashes and creates the prompts used by the shared evaluator. Downloads require network access; prepared evaluations can run offline.

`make check-env` and `make test` run on CPU. GPU evaluation runs inside an allocated GPU job.

## Run an evaluation

Use a separate output directory for each repeat.

### Slurm on the original cluster

Run these commands from the repository root:

```bash
make submit TASK=sciq MODEL=ouro_26 \
  OUTPUT=results/repeats/sciq/ouro_26

make submit TASK=piqa MODEL=parcae_13 \
  OUTPUT=results/repeats/piqa/parcae_13
```

`make submit` selects the recorded task/model matrix-multiplication precision, including the validated `highest` overrides. To set it explicitly, add `MATMUL_PRECISION=highest`.

Each job requests one H200, six CPUs, 64 GB of host memory, and up to 60 minutes on partition and account **`scavenger-h200`**, the authorized GPU partition for this workspace. `TASK=all` is supported for preparation and reporting; each GPU submission takes one task and model.

For an additional evaluator option, invoke the wrapper directly and specify the precision:

```bash
mkdir -p results/multitask/logs
sbatch scripts/run_suite.slurm \
  --task piqa --model parcae_13 \
  --matmul-precision highest \
  --output results/repeats/piqa/parcae_13
```

### Direct Python entry point

Inside an allocated GPU job, or on a separate machine with a suitable CUDA GPU:

```bash
venv/bin/python -B scripts/evaluate.py \
  --task sciq --model ouro_26 \
  --matmul-precision high \
  --output results/repeats/sciq/ouro_26
```

The direct evaluator uses the default precision in `configs/models.json` unless explicitly overridden. To match the published runs, pass `--matmul-precision highest` for Huginn and Looped-Qwen, and for these additional pairs:

- `piqa` with `ouro_14` or `ouro_26`
- `arc_challenge`, `sciq`, or `piqa` with `parcae_13`
- `sciq` or `piqa` with `parcae_370`

The remaining pairs use `high`.

For a small smoke run, add `--limit 10` and choose a separate output directory. A positive limit requires an explicit output path. Run `venv/bin/python scripts/evaluate.py --help` for all options.

To resume a partial run, reuse its output directory and original arguments. The evaluator validates the manifest, source hashes, data, and saved candidates before continuing.

### Huginn half-depth experiment

The named `huginn_half_depth` experiment uses the same evaluator and task data.
It selects R=16, reference iteration 5, and fixed guidance strength 0.5 from
paper Table A8. The seven standard configurations remain in `configs/models.json`;
experiment settings are in `configs/experiments.json`.

Run each task's evaluation, independent timing, and report on `scavenger-h200`
with new output directories:

```bash
for task in arc_easy arc_challenge sciq piqa; do
  make submit-reduced TASK="$task" \
    REDUCED_OUTPUT="results/repeats/reduced_depth/$task"
done
```

This compares against the saved R=32 and R=16 baselines. To run only the new
accuracy evaluation inside an allocated GPU job:

```bash
venv/bin/python -B scripts/evaluate.py \
  --task arc_challenge --model huginn_16 --experiment huginn_half_depth \
  --output results/repeats/reduced_depth/arc_challenge/huginn_16_ref5
```

After all four jobs complete, build the repeat suite with:

```bash
make report-reduced TASK=all REDUCED_OUTPUT=results/repeats/reduced_depth
```

Rebuild the published suite with `make report-reduced TASK=all`; use a specific
`TASK` for one task's comparison. The analysis keeps the registered task metrics
and gives each task equal weight. The [analysis plan](results/reduced_depth/analysis_plan.json)
records the fixed parameters, resampling methods, and multiple-comparison families.

## Audit and report

After installing dependencies and restoring data and model assets:

```bash
make audit TASK=all

# Inspect progress without requiring all runs to be complete:
venv/bin/python -B scripts/analyze.py --task all --partial

# Regenerate the four-task reports from the registered saved runs:
make report TASK=all
```

The CPU audit requires the checkpoint files. It checks weight-file sizes against their download manifests and hashes metadata, tokenizers, and source files. It reconstructs prompts, re-tokenizes candidates, verifies saved token scores and sums, and recomputes predictions and accuracy.

`make audit` writes a maintenance audit record without changing completed scores or reports. `make report` rewrites generated reports. A single-task report command also rewrites the combined index to contain only that task; use `TASK=all` to restore the four-task index.

The analyzer reads the registered published result locations. Each run under `results/repeats/` has its own scores, predictions, summary, and manifest.

## Repository layout

| Path | Purpose |
|---|---|
| `configs/` | Shared model/task registries and paper reference scores |
| `src/` | Architecture adapters, shared LoopCD scoring, data preparation, and integrity checks |
| `scripts/prepare_tasks.py` | Unified data preparation |
| `scripts/evaluate.py` | Unified GPU evaluation |
| `scripts/analyze.py` | Independent CPU audit and comparison reports |
| `tests/` | Unit and protocol checks |
| `sources/` | Pinned upstream implementation and task-definition evidence |
| `models/` | Checkpoint manifests; downloaded weights and model assets |
| `data/` | Downloaded source datasets and generated prompts |
| `results/arc_easy/` | Original Ouro-1.4B ARC-Easy evaluation |
| `results/arc_all_models/` | Other six original ARC-Easy evaluations |
| `results/multitask/` | Three additional tasks, all-task comparisons, and completion evidence |
| `results/reduced_depth/` | Huginn half-depth accuracy, standalone timing, FLOPs, and paired comparisons |

Each completed run retains `candidate_scores.jsonl`, `predictions.jsonl`, `summary.json`, `run_manifest.json`, and validation evidence. The original Ouro-1.4B ARC-Easy run retains its historical provenance format; the other 27 runs also include a `source_snapshot/`. Historical entry points remain for compatibility; new work should use the unified scripts above.

## Implementation details

Evaluation uses seed `1234`. Confidence intervals come from paired resampling of evaluation questions.

Huginn and Parcae use question-keyed random initial states shared across candidates and guidance methods.

Looped-Qwen3 uses a local implementation of the paper's loop construction: Qwen3-4B's layer window `15:19`, eight updates of `h ← (7/8)h + (1/8)g(h)`, and the full remaining output path. See the [implementation record](sources/looped_qwen/evidence.json).

The original 28 runs record joint evaluation time for four scoring policies on one
recurrent trajectory. The Huginn half-depth experiment also records independent
policy timing with warmed, synchronized forwards on identical sampled batches.

See the [combined report](results/multitask/REPORT.md) and [machine-readable results](results/multitask/comparison.json) for additional provenance. Documentation and generated reports are in English. Upstream models, datasets, and vendored sources retain their respective licenses.
