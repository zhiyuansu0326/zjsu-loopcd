# Ouro-1.4B × ARC-Easy: Functional reproduction of LoopCD

Full test set: 2376 questions. The sole model is ByteDance/Ouro-1.4B base, with 8-shot examples from the training set, seed=1234, and four complete loops.
No training or parameter search was performed. Fixed ω=0.5 and adaptive cap=1.0 follow Table A4 of the paper. All policies share the same questions, examples, weights, and loop trajectories.

## Reproduction and paper results

The table uses the predetermined acc_norm metric: conditional answer log likelihood divided by the answer text's character count. The paper specifies likelihood scoring but does not explicitly identify the normalization used for ARC-E. Unnormalized acc is therefore retained as well; these results do not establish a bitwise reproduction of the authors' evaluation code.

| Policy | Correct | Reproduced acc_norm | Paper Table 2 | Reproduced acc |
|---|---:|---:|---:|---:|
| Baseline (loop 4) | 1998/2376 | 84.09% | 84.09% | 83.80% |
| LoopCD fixed strength ω=0.5 | 2005/2376 | 84.39% | 84.39% | 84.43% |
| LoopCD adaptive cap=1.0 | 1999/2376 | 84.13% | 84.26% | 84.55% |
| Weak reference (loop 1) | 1386/2376 | 58.33% | Not compared | 64.27% |

## Paired changes on the same questions

| Policy | Corrected | Regressed | Net correct | Change (pp) | Paired bootstrap 95% CI | McNemar p |
|---|---:|---:|---:|---:|---|---:|
| LoopCD fixed strength ω=0.5 | 38 | 31 | +7 | +0.29 | [-0.38, +0.97] | 0.4704 |
| LoopCD adaptive cap=1.0 | 48 | 47 | +1 | +0.04 | [-0.76, +0.84] | 1.0000 |

Intervals use 20,000 paired bootstrap resamples with seed=1234. They describe uncertainty from sampling test questions, not variation across repeated model runs. Results for one model and one task cannot establish the paper's conclusions for other models, generation tasks, or speed gains from fewer loops.

Both LoopCD acc_norm improvement intervals include zero. The observed gains are small and do not establish a stable, statistically significant accuracy improvement.

## Implementation and validation

Official Ouro applies RMSNorm after each complete 24-layer loop before continuing to the next loop. This implementation reads the first and fourth loop states returned by the official model and applies the shared lm_head to obtain z₁ and z₄. Scores use z′=z₄+ω(z₄−z₁). Adaptive ω=1−(p₄,top1−p₄,top2), with probabilities computed over the full vocabulary.
Model computation uses float32 (FP32 matmul precision=high), with FP32 logit arithmetic and log_softmax. Every token of each candidate answer receives teacher-forced conditional likelihood scoring. Option letters are not scored; contexts are not truncated, and answers are not sampled.

Checks verify that ω=0 matches official fourth-loop logits, early and final loop outputs differ, and single-example and padded-batch inputs agree. A tiny official Ouro model also has tests covering every answer position. See validation.json for full-model checks and ../../tests/test_loopcd.py for test source.

During numerical preflight, BF16 had a maximum single-example/batch answer-token log-probability difference of 0.28756 and was excluded from formal results. The formal configuration had zero difference on the same check. FP32 matmul precision=high permits faster internal matrix multiplication and differs from highest, which requests the highest precision. Precision was selected from numerical checks, without tuning against test accuracy.

All three policies and the weak reference share one four-loop computation. Recorded elapsed time therefore covers the joint experiment and cannot establish a standalone LoopCD speedup.

- Model loading and validation: 13.0 seconds.
- Scoring in this invocation: 147.9 seconds.
- Peak allocated GPU memory: 6.66 GiB.

## Illustrative examples

For each policy and each category, the examples below are the first two corrected or regressed cases in the original test-set order. Complete per-question results are saved in predictions.jsonl.

### Mercury_7239733 · LoopCD fixed strength ω=0.5 · Corrected

Meiosis is a type of cell division in which germ cells divide to produce haploid cells. Where does meiosis occur?

- A. brain cells
- B. bone cells
- C. muscle cells
- D. ovary cells

Correct answer: D; baseline: C; LoopCD: D.

### Mercury_7166145 · LoopCD fixed strength ω=0.5 · Corrected

The circulatory system and the endocrine system work together in the human body. Which describes one way in which these systems interact?

- A. taking in oxygen and transporting it to cells of the body
- B. releasing hormones and transporting them to cells of the body
- C. absorbing nutrients from food and transporting them to cells in the body
- D. collecting waste products from the cells and transporting it out of the body

Correct answer: B; baseline: C; LoopCD: B.

### Mercury_SC_405199 · LoopCD fixed strength ω=0.5 · Regressed

Which of these will most likely harm a habitat?

- A. planting trees
- B. water pollution
- C. rainfall
- D. sunlight

Correct answer: B; baseline: B; LoopCD: A.

### Mercury_7146195 · LoopCD fixed strength ω=0.5 · Regressed

A research scientist writes a paper on the initial regrowth of a forest after a fire has damaged the entire ecosystem. Which title would be best for the paper?

- A. Primary Succession
- B. Stable Communities
- C. Climax Communities
- D. Secondary Succession

Correct answer: D; baseline: D; LoopCD: A.

### Mercury_7166145 · LoopCD adaptive cap=1.0 · Corrected

The circulatory system and the endocrine system work together in the human body. Which describes one way in which these systems interact?

- A. taking in oxygen and transporting it to cells of the body
- B. releasing hormones and transporting them to cells of the body
- C. absorbing nutrients from food and transporting them to cells in the body
- D. collecting waste products from the cells and transporting it out of the body

Correct answer: B; baseline: C; LoopCD: B.

### Mercury_SC_402980 · LoopCD adaptive cap=1.0 · Corrected

During which activity should a student wear goggles?

- A. writing a science report
- B. mixing baking soda with water
- C. measuring the length of a shadow
- D. examining a leaf with a microscope

Correct answer: B; baseline: D; LoopCD: B.

### Mercury_SC_405199 · LoopCD adaptive cap=1.0 · Regressed

Which of these will most likely harm a habitat?

- A. planting trees
- B. water pollution
- C. rainfall
- D. sunlight

Correct answer: B; baseline: B; LoopCD: A.

### VASoL_2011_5_34 · LoopCD adaptive cap=1.0 · Regressed

Which layer of Earth is the thinnest?

- A. Inner core
- B. Crust
- C. Outer core
- D. Mantle

Correct answer: B; baseline: B; LoopCD: C.

## Comparability limitations

The paper does not disclose the complete few-shot examples and random seeds, exact model revision, or every numerical implementation detail. This experiment uses a frozen official model revision and fixed official lm-eval prompt construction and sampling rules. Seeds and guidance strength were not selected from test results. Absolute score differences may arise from undisclosed details, numerical precision, attention implementations, or normalization. Even close agreement with the paper establishes only a functional reproduction of this setting.

Sources: Table 2 on page 6, Table A2 on page 17, and Table A4 on page 18 of the local paper; [paper](https://arxiv.org/html/2610.02185v1), [official model](https://huggingface.co/ByteDance/Ouro-1.4B), and [official lm-eval](https://github.com/EleutherAI/lm-evaluation-harness). Exact revisions and file hashes are recorded in ../../models/model_manifest.json, ../../data/protocol.json, and run_manifest.json.
