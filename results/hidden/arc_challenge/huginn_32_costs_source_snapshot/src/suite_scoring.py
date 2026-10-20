"""Shared ARC scoring, without model-specific assumptions about the recurrent block."""

import torch

from src.loopcd import HIDDEN_MODES, MODES, guided_logits


class SuiteScorer:
    def __init__(self, adapter, question_ids, fixed=0.5, cap=1.0, *, guidance_space="logits"):
        self.adapter = adapter
        self.question_ids = question_ids
        self.fixed = fixed
        self.cap = cap
        if guidance_space not in ("logits", "hidden"):
            raise ValueError("guidance_space must be logits or hidden")
        self.guidance_space = guidance_space
        self.modes = HIDDEN_MODES if guidance_space == "hidden" else MODES

    def forward(self, inputs, mask, rows, positions, keys):
        if self.guidance_space == "hidden":
            return self.adapter.forward_hidden_comparison_logits(
                inputs, mask, rows, positions, sample_keys=keys, strength=self.fixed,
            )
        return self.adapter.forward_logits(inputs, mask, rows, positions, sample_keys=keys)

    def tensors(self, candidates):
        length = max(len(c.input_ids) for c in candidates)
        if length > self.adapter.max_length:
            raise ValueError("Context overflow; truncation is forbidden.")
        device = self.adapter.device
        inputs = torch.full((len(candidates), length), self.adapter.pad_id, device=device, dtype=torch.long)
        attention = torch.zeros_like(inputs)
        rows, positions, targets = [], [], []
        for row, c in enumerate(candidates):
            inputs[row, :len(c.input_ids)] = torch.tensor(c.input_ids, device=device)
            attention[row, :len(c.input_ids)] = 1
            rows.extend([row] * len(c.targets))
            positions.extend(c.positions)
            targets.extend(c.targets)
        return (
            inputs, attention, torch.tensor(rows, device=device),
            torch.tensor(positions, device=device), torch.tensor(targets, device=device),
            [self.question_ids[c.question_index] for c in candidates],
        )

    @torch.inference_mode()
    def score(self, candidates):
        inputs, mask, rows, positions, labels, keys = self.tensors(candidates)
        outputs = self.forward(inputs, mask, rows, positions, keys)
        final = outputs["baseline"].float()
        assert all(value.shape == final.shape and value.shape[0] == len(labels)
                   and bool(torch.isfinite(value).all()) for value in outputs.values())
        values = {}
        for mode in self.modes:
            if mode in outputs:
                logits = outputs[mode].float()
            else:
                logits = guided_logits(
                    final, outputs["reference"].float(),
                    self.cap if mode == "loopcd_adaptive" else self.fixed,
                    adaptive=(mode == "loopcd_adaptive"),
                )
            values[mode] = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1).cpu().tolist()
        results, offset = [], 0
        for c in candidates:
            n = len(c.targets)
            token_scores = {mode: v[offset:offset+n] for mode, v in values.items()}
            results.append({
                "question_index": c.question_index, "choice_index": c.choice_index,
                "scores": {mode: sum(v) for mode, v in token_scores.items()},
                "token_logprobs": token_scores, "target_ids": c.targets,
                "input_length": len(c.input_ids),
            })
            offset += n
        return results

    @torch.inference_mode()
    def validate(self, candidates):
        pair = candidates[:2]
        tensors = self.tensors(pair)
        inputs, mask, rows, positions, _, keys = tensors
        validation = self.adapter.validate(inputs, mask, rows, positions, sample_keys=keys)
        outputs = self.forward(inputs, mask, rows, positions, keys)
        if self.guidance_space == "hidden":
            validation["hidden_checks"] = self.adapter.validate_hidden(
                inputs, mask, rows, positions, sample_keys=keys, strength=self.fixed,
            )
            zero = self.adapter.forward_hidden_logits(
                inputs, mask, rows, positions, sample_keys=keys, strength=0.0,
            )
        else:
            zero = guided_logits(outputs["baseline"], outputs["reference"], 0.0)
        torch.testing.assert_close(zero, outputs["baseline"].float(), rtol=0, atol=0)
        batched = self.score(pair)
        singles = [self.score([candidate])[0] for candidate in pair]
        errors = {
            mode: max(
                abs(a-b)
                for bscore, sscore in zip(batched, singles)
                for a,b in zip(bscore["token_logprobs"][mode], sscore["token_logprobs"][mode])
            )
            for mode in self.modes
        }
        # Fixed shape test separates padding leakage from different GEMM kernels.
        changed = inputs.clone()
        changed[mask == 0] = (self.adapter.pad_id + 1) % len(self.adapter.tokenizer)
        modified = self.forward(changed, mask, rows, positions, keys)
        padding_error = max((outputs[m] - modified[m]).abs().max().item() for m in outputs)
        if padding_error > 1e-5:
            raise AssertionError(f"Masked padding changes valid logits: {padding_error}")
        if max(errors.values()) > 1e-3:
            raise AssertionError(f"Batch/single numeric mismatch: {errors}")
        validation.update({
            "zero_guidance_matches_baseline_exactly": True,
            "batch_single_token_logprob_errors": errors,
            "batch_single_tolerance": 1e-3,
            "masked_padding_token_change_max_abs_error": padding_error,
            "early_final_mean_abs_difference": (
                validation["early_final_mean_abs_logit_difference"] if self.guidance_space == "hidden"
                else (outputs["baseline"]-outputs["reference"]).abs().mean().item()
            ),
            "stochastic": self.adapter.stochastic,
        })
        return validation
