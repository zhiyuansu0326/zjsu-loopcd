"""LoopCD-Logits for the official Ouro checkpoint; no training or extra loops."""

from dataclasses import dataclass

import torch


MODES = ("baseline", "loopcd_fixed", "loopcd_adaptive", "reference")


def guided_logits(final, reference, strength=0.5, adaptive=False):
    """Paper Eq. (3), with Eq. (4) computed over the entire vocabulary."""
    final, reference = final.float(), reference.float()
    if adaptive:
        top2 = final.softmax(dim=-1).topk(2, dim=-1).values
        omega = strength * (1.0 - (top2[..., 0] - top2[..., 1]))
        omega = omega.unsqueeze(-1)
    else:
        omega = strength
    return final + omega * (final - reference)


@dataclass
class Candidate:
    question_index: int
    choice_index: int
    input_ids: list[int]
    positions: list[int]
    targets: list[int]


def encode_candidate(tokenizer, context, choice, question_index, choice_index):
    # Match lm-eval's causal _encode_pair: move trailing context whitespace.
    spaces = len(context) - len(context.rstrip())
    continuation = context[len(context) - spaces:] + " " + choice if spaces else " " + choice
    context = context.rstrip() if spaces else context
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    whole_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not context_ids or whole_ids[:len(context_ids)] != context_ids:
        raise ValueError("Tokenizer boundary changes the context; do not silently mis-score.")
    targets = whole_ids[len(context_ids):]
    if not targets:
        raise ValueError("Empty answer continuation")
    return Candidate(
        question_index, choice_index, whole_ids[:-1],
        list(range(len(context_ids) - 1, len(whole_ids) - 1)), targets,
    )


class OuroScorer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if model.model.total_ut_steps != 4:
            raise ValueError("This reproduction requires the paper's four recurrent passes.")

    @torch.inference_mode()
    def score(self, candidates):
        lengths = [len(c.input_ids) for c in candidates]
        maximum = max(lengths)
        if maximum > self.model.config.max_position_embeddings:
            raise ValueError("Input exceeds model context; truncation is forbidden.")
        inputs = torch.full(
            (len(candidates), maximum), self.pad_id, dtype=torch.long, device=self.device
        )
        attention = torch.zeros_like(inputs)
        row_indices, positions, targets = [], [], []
        for row, candidate in enumerate(candidates):
            inputs[row, :lengths[row]] = torch.tensor(candidate.input_ids, device=self.device)
            attention[row, :lengths[row]] = 1
            row_indices.extend([row] * len(candidate.targets))
            positions.extend(candidate.positions)
            targets.extend(candidate.targets)
        # Native Ouro applies its final RMSNorm after EVERY complete stack pass.
        # The returned list contains those normalized, recurrently reused states.
        _, states, _ = self.model.model(
            input_ids=inputs, attention_mask=attention, use_cache=False
        )
        if len(states) != 4:
            raise AssertionError("Expected four complete recurrent states")
        rows = torch.tensor(row_indices, device=self.device)
        cols = torch.tensor(positions, device=self.device)
        labels = torch.tensor(targets, device=self.device)
        # Only answer-predicting positions need a vocabulary projection.
        final = self.model.lm_head(states[-1][rows, cols]).float()
        reference = self.model.lm_head(states[0][rows, cols]).float()
        del states
        token_scores = {}
        for mode in MODES:
            if mode == "baseline":
                logits = final
            elif mode == "reference":
                logits = reference
            else:
                logits = guided_logits(
                    final, reference,
                    strength=1.0 if mode == "loopcd_adaptive" else 0.5,
                    adaptive=(mode == "loopcd_adaptive"),
                )
            log_probs = logits.log_softmax(dim=-1)
            token_scores[mode] = log_probs.gather(-1, labels[:, None]).squeeze(-1).cpu().tolist()
        results, offset = [], 0
        for candidate in candidates:
            n = len(candidate.targets)
            scores = {mode: sum(values[offset:offset + n]) for mode, values in token_scores.items()}
            results.append({
                "question_index": candidate.question_index,
                "choice_index": candidate.choice_index,
                "scores": scores,
                "token_logprobs": {mode: values[offset:offset + n] for mode, values in token_scores.items()},
                "target_ids": candidate.targets,
                "input_length": len(candidate.input_ids),
            })
            offset += n
        return results

    @torch.inference_mode()
    def validate(self, candidates):
        """Check actual checkpoint output, padding, and the zero-guidance identity."""
        pair = candidates[:2]
        batched = self.score(pair)
        singles = [self.score([candidate])[0] for candidate in pair]
        padding_errors = {
            mode: max(
                abs(a - b) for batch, single in zip(batched, singles)
                for a, b in zip(batch["token_logprobs"][mode], single["token_logprobs"][mode])
            ) for mode in MODES
        }
        max_padding_error = max(padding_errors.values())
        # The production run uses FP32; require agreement to 1e-4 natural-log units.
        padding_tolerance = 1e-4 if next(self.model.parameters()).dtype == torch.float32 else 0.15
        if max_padding_error > padding_tolerance:
            raise AssertionError(f"Unexpected batched/single token difference: {max_padding_error}")
        inputs = torch.tensor([pair[0].input_ids], device=self.device)
        mask = torch.ones_like(inputs)
        _, states, _ = self.model.model(input_ids=inputs, attention_mask=mask, use_cache=False)
        final = self.model.lm_head(states[-1][:, -1:]).float()
        reference = self.model.lm_head(states[0][:, -1:]).float()
        official = self.model(
            input_ids=inputs, attention_mask=mask, use_cache=False,
            logits_to_keep=1, exit_at_step=3, use_weighted_exit=False
        ).logits.float()
        native_max_error = (final - official).abs().max().item()
        torch.testing.assert_close(final, official, rtol=0, atol=0)
        torch.testing.assert_close(guided_logits(final, reference, 0), official, rtol=0, atol=0)
        if torch.equal(final, reference):
            raise AssertionError("Early and final predictions unexpectedly identical")
        return {
            "native_final_logits_max_abs_error": native_max_error,
            "zero_guidance_matches_native_exactly": True,
            "batch_single_max_token_logprob_error": max_padding_error,
            "batch_single_errors_by_policy": padding_errors,
            "batch_single_absolute_tolerance": padding_tolerance,
            "early_final_mean_abs_logit_difference": (final - reference).abs().mean().item(),
            "recurrent_passes": len(states),
        }
