"""LoopCD's deterministic K=8 damped wrapper over official Qwen3-4B.

Evidence: LoopCD Appendix A.2/Table A1; Training-Free Looped Transformers
Eq. (10). Zero-based layers 15:19 are the window, with 15 prelude and 17
tail layers. This implements the explicit LoopCD recurrence without an anchor.
The upstream training-free repository's default anchored recipe is different.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask


class _NativeReplayWindow(nn.Module):
    """Independent residual-form Euler block used only inside native forward."""

    def __init__(self, layers, steps: int, denominator: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.denominator = denominator
        self.attention_type = "full_attention"

    def forward(self, hidden_states, **kwargs):
        for _ in range(self.steps):
            before = hidden_states
            proposed = before
            for layer in self.layers:
                proposed = layer(proposed, **kwargs)
            # Algebraically equivalent to the production convex combination.
            # K=1 takes the unmodified block output, avoiding cancellation error.
            hidden_states = (
                proposed if self.denominator == 1
                else before + (proposed - before) / self.denominator
            )
        return hidden_states


class LoopedQwenAdapter:
    stochastic = False
    window_start = 15
    window_stop = 19

    def __init__(
        self, model_path, device="cuda:0", dtype=torch.float32, loops=8
    ):
        model_path = Path(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True,
        ).to(device).eval()
        self._initialize(model, tokenizer, loops)

    @classmethod
    def from_model(cls, model, tokenizer, loops=8):
        """Use an already constructed official Qwen3 model (including tiny tests)."""
        adapter = cls.__new__(cls)
        adapter._initialize(model, tokenizer, loops)
        return adapter

    def _initialize(self, model, tokenizer, loops):
        if not isinstance(loops, int) or loops < 1:
            raise ValueError("loops must be a positive integer")
        if model.config.model_type != "qwen3" or len(model.model.layers) != 36:
            raise ValueError("Expected the 36-layer dense Qwen3 architecture")
        if any(layer.attention_type != "full_attention" for layer in model.model.layers):
            raise ValueError("This adapter is specified for full-attention Qwen3-4B")
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.loops = loops
        self.max_length = int(model.config.max_position_embeddings)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def _prepare(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Expected matching [batch, sequence] inputs and mask")
        if input_ids.shape[1] > self.max_length:
            raise ValueError("Context truncation is forbidden")
        if input_ids.device != self.device or attention_mask.device != self.device:
            raise ValueError("Inputs must already reside on the adapter device")
        body = self.model.model
        hidden = body.embed_tokens(input_ids)
        cache_position = torch.arange(input_ids.shape[1], device=self.device)
        position_ids = cache_position.unsqueeze(0)
        mask = create_causal_mask(
            config=body.config, input_embeds=hidden,
            attention_mask=attention_mask, cache_position=cache_position,
            past_key_values=None, position_ids=position_ids,
        )
        kwargs = {
            "attention_mask": mask,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": False,
            "cache_position": cache_position,
            "position_embeddings": body.rotary_emb(hidden, position_ids),
        }
        return hidden, kwargs

    def _apply_layers(self, hidden, start, stop, kwargs):
        for layer in self.model.model.layers[start:stop]:
            hidden = layer(hidden, **kwargs)
        return hidden

    def _trajectory(self, input_ids, attention_mask, loops=None):
        loops = self.loops if loops is None else loops
        hidden, kwargs = self._prepare(input_ids, attention_mask)
        hidden = self._apply_layers(hidden, 0, self.window_start, kwargs)
        first = None
        alpha = 1.0 / loops
        for iteration in range(loops):
            proposed = self._apply_layers(
                hidden, self.window_start, self.window_stop, kwargs
            )
            # Full four-layer block first, then one damped update. Do not damp
            # each layer individually or normalize between completed substeps.
            hidden = (1.0 - alpha) * hidden + alpha * proposed
            if iteration == 0:
                first = hidden
        return hidden, first, kwargs

    def _readout(self, hidden, kwargs, rows, positions):
        hidden = self._apply_layers(hidden, self.window_stop, 36, kwargs)
        selected = self.model.model.norm(hidden[rows, positions])
        return self.model.lm_head(selected).float()

    def _indices(self, input_ids, attention_mask, rows, positions):
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        positions = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        if rows.ndim != 1 or rows.shape != positions.shape or rows.numel() == 0:
            raise ValueError("rows and positions must be nonempty matching 1D indices")
        if (
            torch.any(rows < 0) or torch.any(rows >= input_ids.shape[0])
            or torch.any(positions < 0) or torch.any(positions >= input_ids.shape[1])
        ):
            raise ValueError("Readout position outside input")
        if not torch.all(attention_mask[rows, positions] == 1):
            raise ValueError("Cannot score a padding position")
        return rows, positions

    @torch.inference_mode()
    def forward_logits(
        self, input_ids, attention_mask, rows, positions, sample_keys=None
    ):
        rows, positions = self._indices(input_ids, attention_mask, rows, positions)
        final, reference, kwargs = self._trajectory(input_ids, attention_mask)
        baseline = self._readout(final, kwargs, rows, positions)
        earlier = self._readout(reference, kwargs, rows, positions)
        return {"baseline": baseline, "reference": earlier}

    @contextmanager
    def _native_window(self, steps, denominator):
        """Keep native Qwen3 forward, replacing its window by a replay module."""
        body = self.model.model
        original = body.layers
        window = _NativeReplayWindow(
            list(original[self.window_start:self.window_stop]), steps, denominator
        )
        body.layers = nn.ModuleList(
            list(original[:self.window_start]) + [window]
            + list(original[self.window_stop:])
        )
        try:
            yield
        finally:
            body.layers = original

    def _native_replay_logits(
        self, input_ids, attention_mask, rows, positions, steps, denominator
    ):
        with self._native_window(steps, denominator):
            hidden = self.model.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
        return self.model.lm_head(hidden[rows, positions]).float()

    @torch.inference_mode()
    def validate(
        self, input_ids, attention_mask, rows, positions, sample_keys=None
    ):
        rows, positions = self._indices(input_ids, attention_mask, rows, positions)
        observed = self.forward_logits(input_ids, attention_mask, rows, positions)
        # Replay the explicit wrapper inside native Qwen3 forward, rather than
        # incorrectly treating the unlooped dense model as the looped baseline.
        replay = {
            "baseline": self._native_replay_logits(
                input_ids, attention_mask, rows, positions, self.loops, self.loops
            ),
            "reference": self._native_replay_logits(
                input_ids, attention_mask, rows, positions, 1, self.loops
            ),
        }
        tolerance = 5e-4 if self.dtype == torch.float32 else 0.3
        errors = {}
        for key in observed:
            errors[key] = (observed[key] - replay[key]).abs().max().item()
            torch.testing.assert_close(
                observed[key], replay[key], rtol=1e-5, atol=tolerance
            )
        # K=1 must reduce to the ordinary dense model at every input position.
        dense_hidden = self.model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        one, _, kwargs = self._trajectory(input_ids, attention_mask, loops=1)
        one = self._apply_layers(one, self.window_stop, 36, kwargs)
        one = self.model.model.norm(one)
        dense_identity_error = (one - dense_hidden).abs().max().item()
        torch.testing.assert_close(one, dense_hidden, rtol=0, atol=0)
        dense_logits = self.model.lm_head(dense_hidden[rows, positions]).float()
        one_logits = self.model.lm_head(one[rows, positions]).float()
        torch.testing.assert_close(one_logits, dense_logits, rtol=0, atol=0)
        single_errors = {}
        for key in observed:
            single_errors[key] = 0.0
        for row in rows.unique().tolist():
            selected = rows == row
            valid_positions = attention_mask[row].nonzero().flatten()
            length = int(valid_positions[-1]) + 1
            single = self.forward_logits(
                input_ids[row:row + 1, :length], attention_mask[row:row + 1, :length],
                torch.zeros(int(selected.sum()), dtype=torch.long, device=self.device),
                positions[selected],
            )
            for key in observed:
                error = (
                    observed[key][selected].log_softmax(-1)
                    - single[key].log_softmax(-1)
                ).abs().max().item()
                single_errors[key] = max(single_errors[key], error)
                torch.testing.assert_close(
                    observed[key][selected].log_softmax(-1),
                    single[key].log_softmax(-1), rtol=1e-5, atol=tolerance,
                )
        difference = (observed["baseline"] - observed["reference"]).abs().mean().item()
        if self.loops > 1 and difference == 0:
            raise AssertionError("Early and final recurrent outputs unexpectedly identical")
        return {
            "loops": self.loops,
            "reference_pass": 1,
            "window_zero_based": list(range(self.window_start, self.window_stop)),
            "damping_step_size": 1 / self.loops,
            "anchor_weight": 0,
            "native_damped_wrapper_replay_max_logit_errors": errors,
            "native_damped_wrapper_replay_passed": True,
            "k1_native_dense_all_hidden_max_abs_error": dense_identity_error,
            "k1_dense_logits_match_exactly": True,
            "batch_single_max_full_vocab_logprob_errors": single_errors,
            "early_final_mean_abs_logit_difference": difference,
            "looped_vs_dense_mean_abs_logit_difference": (
                observed["baseline"] - dense_logits
            ).abs().mean().item(),
            "absolute_tolerance": tolerance,
            "dtype": str(self.dtype),
            "stochastic": False,
        }
