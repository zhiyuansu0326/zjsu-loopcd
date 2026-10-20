"""LoopCD readouts from one unmodified, pinned Huginn recurrence trajectory."""

import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.loopcd import guided_hidden


REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"


class HuginnAdapter:
    stochastic = True
    # The official tokenizer and checkpoint likelihood evaluator both prepend BOS.
    add_bos = True

    def __init__(self, model_path, device="cuda:0", dtype=torch.float32, loops=32, reference_step=1):
        model_path = Path(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=dtype, device_map={"": str(device)},
        ).eval()
        self._configure(model, tokenizer, loops, reference_step)

    @classmethod
    def from_model(cls, model, tokenizer, loops=32, reference_step=1):
        """Wrap an already loaded official model (also supports small CPU tests)."""
        instance = cls.__new__(cls)
        instance._configure(model.eval(), tokenizer, loops, reference_step)
        return instance

    def _configure(self, model, tokenizer, loops, reference_step):
        if isinstance(loops, bool) or not isinstance(loops, int) or loops < 1:
            raise ValueError("loops must be a positive integer")
        if (
            isinstance(reference_step, bool)
            or not isinstance(reference_step, int)
            or not 1 <= reference_step <= loops
        ):
            raise ValueError("reference_step must be an integer between 1 and loops")
        if model.config.state_init != "like-init":
            raise ValueError("This adapter implements Huginn's released like-init state.")
        if model.config.test_time_noise != 0:
            raise ValueError("Extra recurrent noise is outside the paper's configuration.")
        self.model, self.tokenizer, self.loops = model, tokenizer, loops
        self.reference_step = reference_step
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        self.max_length = model.config.block_size
        self.noise_seed = 1234

    def metadata(self):
        return {
            "model_revision": REVISION,
            "loops": self.loops,
            "reference_recurrent_update": self.reference_step,
            "reference_is_initial_noise": False,
            "coda_layers": len(self.model.transformer.coda),
            "readout": "ln_f -> complete coda -> ln_f -> lm_head",
            "noise_seed": self.noise_seed,
            "noise_key": "SHA256(seed, question ID), independent of answer text",
            "noise_generation": "CPU float32 truncated normal, fixed 64-token blocks; then cast to model dtype",
            "noise_std_before_embedding_scale": float(self.model.config.init_values["std"]),
            "noise_truncation_standard_deviations": [-3, 3],
            "noise_embedding_scale": float(self.model.emb_scale),
            "noise_coupling": "same question shares identical positional initial states across options and depths",
            "paper_rng_difference": "Paper seed and sample ordering are not specified; this fixes an explicit reproducible RNG protocol.",
            "padding": "right padding only; native causal attention makes suffix padding invisible to scored tokens",
            "add_bos_token": True,
            "bos_token_id": self.bos_token_id,
            "bos_policy": "Prepend exactly one BOS to context; no special tokens in continuation.",
            "bos_evidence": [
                "models/Huginn-0125/tokenizer.json: post_processor.single prepends <|begin_text|>",
                "sources/huginn/evaluate_raven/quick_checkpoint_eval.py:139 explicitly prepends BOS",
                "https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/evaluate_raven/quick_checkpoint_eval.py#L139",
            ],
        }

    def _validate_inputs(self, input_ids, attention_mask, rows, positions, sample_keys):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, length] shapes")
        if input_ids.shape[1] > self.max_length:
            raise ValueError("Input exceeds Huginn context length; truncation is forbidden.")
        if sample_keys is None or len(sample_keys) != input_ids.shape[0]:
            raise ValueError("Huginn requires one stable question-ID sample_key per input row.")
        mask = attention_mask.to(torch.bool)
        lengths = mask.sum(-1)
        right_padding_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None] < lengths[:, None]
        if not torch.equal(mask, right_padding_mask) or not bool((lengths > 0).all()):
            raise ValueError("Only nonempty inputs with contiguous right padding are supported.")
        rows = torch.as_tensor(rows, dtype=torch.long, device=input_ids.device)
        positions = torch.as_tensor(positions, dtype=torch.long, device=input_ids.device)
        if rows.ndim != 1 or rows.shape != positions.shape or not rows.numel():
            raise ValueError("rows and positions must be equally sized, nonempty 1D indices.")
        if not bool(((rows >= 0) & (rows < input_ids.shape[0])).all()):
            raise ValueError("Answer row index is out of range.")
        if not bool(((positions >= 0) & (positions < lengths[rows])).all()):
            raise ValueError("Every answer-predicting position must be an actual input token.")
        return rows, positions

    def initial_states(self, sample_keys, sequence_length):
        """Native state distribution, with a prefix-stable RNG stream per question.

        Native initialize_state first draws unused randn values, then overwrites
        them with trunc_normal_. We omit that unused draw and explicitly supply
        the resulting state to native validation. Fixed-size blocks also make
        uniform/erfinv dispatch independent of candidate length and batch size.
        """
        hidden_size = self.model.config.n_embd
        std = float(self.model.config.init_values["std"])
        batch = torch.empty(
            len(sample_keys), sequence_length, hidden_size,
            dtype=self.dtype, device=self.device,
        )
        cached_rows = {}
        for index, key in enumerate(sample_keys):
            stable_key = str(key)
            if stable_key in cached_rows:
                batch[index].copy_(batch[cached_rows[stable_key]])
                continue
            seed_material = f"loopcd-huginn:{self.noise_seed}:{stable_key}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") % (2**63)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            cpu_state = torch.empty(sequence_length, hidden_size, dtype=torch.float32)
            for start in range(0, sequence_length, 64):
                block = torch.empty(64, hidden_size, dtype=torch.float32)
                torch.nn.init.trunc_normal_(
                    block, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator
                )
                stop = min(start + 64, sequence_length)
                cpu_state[start:stop] = block[: stop - start]
            cpu_state.mul_(self.model.emb_scale)
            batch[index].copy_(cpu_state.to(device=self.device, dtype=self.dtype))
            cached_rows[stable_key] = index
        return batch

    def _readout_hidden(self, latents, frequencies):
        # Both normalizations are native: one after recurrence and one after coda.
        hidden = self.model.transformer.ln_f(latents)
        block_index = torch.tensor(0, dtype=torch.long, device="cpu")
        for block in self.model.transformer.coda:
            block_index -= 1
            hidden = block(hidden, frequencies, block_index, None, None)
        return self.model.transformer.ln_f(hidden)

    def _recurrent_states(self, input_ids, initial_states, capture_reference=True):
        """Return raw completed recurrent states before either readout normalization."""
        frequencies = self.model.freqs_cis[:, : input_ids.shape[1]]
        embedded = self.model.transformer.wte(input_ids)
        if self.model.emb_scale != 1:
            embedded = embedded * self.model.emb_scale
        block_index = torch.tensor(-1, dtype=torch.long, device="cpu")
        for block in self.model.transformer.prelude:
            block_index += 1
            embedded = block(embedded, frequencies, block_index, None, None)
        hidden = initial_states.clone()
        early = None
        for step in range(self.loops):
            hidden, block_index = self.model.core_block_forward(
                hidden, embedded, frequencies, None, None, block_index, step
            )
            if capture_reference and step + 1 == self.reference_step:
                # Capture the complete selected update before readout normalization.
                early = hidden.clone()
        return hidden, early, frequencies

    def _trajectory(self, input_ids, initial_states, capture_reference=True):
        hidden, early, frequencies = self._recurrent_states(
            input_ids, initial_states, capture_reference=capture_reference,
        )
        final_hidden = self._readout_hidden(hidden, frequencies)
        reference_hidden = self._readout_hidden(early, frequencies) if capture_reference else None
        return final_hidden, reference_hidden

    def _resolve_initial_states(self, input_ids, sample_keys, initial_states):
        if initial_states is None:
            return self.initial_states(sample_keys, input_ids.shape[1])
        expected_shape = (*input_ids.shape, self.model.config.n_embd)
        if not isinstance(initial_states, torch.Tensor) or initial_states.shape != expected_shape:
            raise ValueError(f"initial_states must have shape {expected_shape}")
        if initial_states.device != self.device or initial_states.dtype != self.dtype:
            raise ValueError("initial_states must match the model device and dtype")
        return initial_states

    @torch.inference_mode()
    def forward_logits(
        self, input_ids, attention_mask, rows, positions, sample_keys=None, *, initial_states=None
    ):
        rows, positions = self._validate_inputs(input_ids, attention_mask, rows, positions, sample_keys)
        initial = self._resolve_initial_states(input_ids, sample_keys, initial_states)
        final_hidden, early_hidden = self._trajectory(input_ids, initial)
        return {
            "baseline": self.model.lm_head(final_hidden[rows, positions]).float(),
            "reference": self.model.lm_head(early_hidden[rows, positions]).float(),
        }

    @torch.inference_mode()
    def forward_baseline_logits(
        self, input_ids, attention_mask, rows, positions, sample_keys=None, *, initial_states=None
    ):
        """Run only the final readout, for independently timed baseline inference.

        Supplying precomputed initial states excludes CPU noise generation and
        transfer from repeated inference timing. The supplied tensor is not modified.
        """
        rows, positions = self._validate_inputs(input_ids, attention_mask, rows, positions, sample_keys)
        initial = self._resolve_initial_states(input_ids, sample_keys, initial_states)
        final_hidden, _ = self._trajectory(input_ids, initial, capture_reference=False)
        return self.model.lm_head(final_hidden[rows, positions]).float()

    @torch.inference_mode()
    def forward_hidden_logits(
        self, input_ids, attention_mask, rows, positions, sample_keys=None, *,
        strength=0.5, initial_states=None,
    ):
        """Fixed hidden guidance over the full sequence, with one coda and head."""
        rows, positions = self._validate_inputs(input_ids, attention_mask, rows, positions, sample_keys)
        initial = self._resolve_initial_states(input_ids, sample_keys, initial_states)
        final, reference, frequencies = self._recurrent_states(input_ids, initial)
        guided = guided_hidden(final, reference, strength)
        hidden = self._readout_hidden(guided, frequencies)
        return self.model.lm_head(hidden[rows, positions]).float()

    @torch.inference_mode()
    def forward_hidden_comparison_logits(
        self, input_ids, attention_mask, rows, positions, sample_keys=None, *,
        strength=0.5, initial_states=None,
    ):
        """Paired evaluation readouts sharing one unmodified recurrent trajectory."""
        rows, positions = self._validate_inputs(input_ids, attention_mask, rows, positions, sample_keys)
        initial = self._resolve_initial_states(input_ids, sample_keys, initial_states)
        final, reference, frequencies = self._recurrent_states(input_ids, initial)
        guided = guided_hidden(final, reference, strength)
        baseline_hidden = self._readout_hidden(final, frequencies)
        guided_hidden_states = self._readout_hidden(guided, frequencies)
        return {
            "baseline": self.model.lm_head(baseline_hidden[rows, positions]).float(),
            "loopcd_hidden": self.model.lm_head(guided_hidden_states[rows, positions]).float(),
        }

    @torch.inference_mode()
    def validate_hidden(
        self, input_ids, attention_mask, rows, positions, sample_keys=None, *, strength=0.5,
    ):
        """Check raw states against native hooks and independently rebuild guidance."""
        rows, positions = self._validate_inputs(input_ids, attention_mask, rows, positions, sample_keys)
        initial = self.initial_states(sample_keys, input_ids.shape[1])
        original_initial = initial.clone()
        captured, native_updates = {}, 0

        def capture_native(module, args, output):
            nonlocal native_updates
            native_updates += 1
            if native_updates in {self.reference_step, self.loops}:
                captured[native_updates] = output.clone()

        handle = self.model.transformer.core_block[-1].register_forward_hook(capture_native)
        try:
            native = self.model(
                input_ids=input_ids, input_states=initial, num_steps=self.loops,
                use_cache=False, attention_mask=None,
                output_details={
                    "return_logits": False, "return_latents": False,
                    "return_head": True, "return_stats": False,
                },
            )
        finally:
            handle.remove()
        if native_updates != self.loops:
            raise AssertionError("Native Huginn recurrent update count changed")
        final, reference, frequencies = self._recurrent_states(input_ids, initial)
        torch.testing.assert_close(final, captured[self.loops], atol=0, rtol=0)
        torch.testing.assert_close(reference, captured[self.reference_step], atol=0, rtol=0)

        # Independent full-sequence readout: do not call the adapter readout or
        # guidance helper here, so normalization/placement mistakes are exposed.
        manual_raw = captured[self.loops] + strength * (
            captured[self.loops] - captured[self.reference_step]
        )
        manual_hidden = self.model.transformer.ln_f(manual_raw)
        block_index = torch.tensor(0, dtype=torch.long, device="cpu")
        for block in self.model.transformer.coda:
            block_index -= 1
            manual_hidden = block(manual_hidden, frequencies, block_index, None, None)
        manual_hidden = self.model.transformer.ln_f(manual_hidden)
        manual_logits = self.model.lm_head(manual_hidden[rows, positions]).float()
        native_same_shape = self.model.lm_head(native.hidden_states[rows, positions]).float()
        zero = self.forward_hidden_logits(
            input_ids, attention_mask, rows, positions, sample_keys,
            strength=0.0, initial_states=initial,
        )
        torch.testing.assert_close(zero, native_same_shape, atol=0, rtol=0)

        counts = {"recurrent_updates": 0, "head_passes": 0}
        coda_calls = [0] * len(self.model.transformer.coda)

        def count_updates(module, args, output):
            counts["recurrent_updates"] += 1

        def count_head(module, args, output):
            counts["head_passes"] += 1

        handles = [
            self.model.transformer.core_block[-1].register_forward_hook(count_updates),
            self.model.lm_head.register_forward_hook(count_head),
        ]
        for index, block in enumerate(self.model.transformer.coda):
            def count_coda(module, args, output, index=index):
                coda_calls[index] += 1
            handles.append(block.register_forward_hook(count_coda))
        try:
            standalone = self.forward_hidden_logits(
                input_ids, attention_mask, rows, positions, sample_keys,
                strength=strength, initial_states=initial,
            )
        finally:
            for hook in handles:
                hook.remove()
        if counts != {"recurrent_updates": self.loops, "head_passes": 1} or any(
            count != 1 for count in coda_calls
        ):
            raise AssertionError("Standalone hidden guidance must use one trajectory and one output pass")
        torch.testing.assert_close(standalone, manual_logits, atol=0, rtol=0)
        paired = self.forward_hidden_comparison_logits(
            input_ids, attention_mask, rows, positions, sample_keys,
            strength=strength, initial_states=initial,
        )
        torch.testing.assert_close(paired["loopcd_hidden"], standalone, atol=0, rtol=0)
        torch.testing.assert_close(paired["baseline"], native_same_shape, atol=0, rtol=0)
        torch.testing.assert_close(initial, original_initial, atol=0, rtol=0)
        if not bool(torch.isfinite(standalone).all()):
            raise AssertionError("Nonfinite hidden-guided Huginn logits")
        return {
            "hidden_guidance_strength": strength,
            "hidden_guidance_location": "raw recurrent state before native pre-coda ln_f",
            "hidden_guidance_positions": "all input and context positions before complete coda",
            "hidden_native_final_raw_max_abs_error": (final - captured[self.loops]).abs().max().item(),
            "hidden_native_reference_raw_max_abs_error": (
                reference - captured[self.reference_step]
            ).abs().max().item(),
            "hidden_manual_readout_max_abs_error": (standalone - manual_logits).abs().max().item(),
            "hidden_zero_strength_matches_native_exactly": True,
            "hidden_paired_matches_standalone_exactly": True,
            "hidden_initial_states_unchanged": True,
            "hidden_standalone_recurrent_updates": counts["recurrent_updates"],
            "hidden_standalone_coda_layer_calls": coda_calls,
            "hidden_standalone_head_passes": counts["head_passes"],
        }

    @torch.inference_mode()
    def validate(self, input_ids, attention_mask, rows, positions, sample_keys=None):
        rows, positions = self._validate_inputs(input_ids, attention_mask, rows, positions, sample_keys)
        initial = self.initial_states(sample_keys, input_ids.shape[1])
        final_hidden, early_hidden = self._trajectory(input_ids, initial)
        baseline = self.model.lm_head(final_hidden[rows, positions]).float()
        reference = self.model.lm_head(early_hidden[rows, positions]).float()
        native = self.model(
            input_ids=input_ids, input_states=initial, num_steps=self.loops,
            use_cache=False, attention_mask=None,
            output_details={
                "return_logits": True, "return_latents": False,
                "return_head": True, "return_stats": False,
            },
        )
        native_hidden = native.hidden_states[rows, positions]
        # The exact hidden-state comparison isolates recurrence/coda correctness
        # from different GEMM shapes when projecting all tokens versus selected tokens.
        torch.testing.assert_close(final_hidden[rows, positions], native_hidden, atol=0, rtol=0)
        native_same_shape_logits = self.model.lm_head(native_hidden).float()
        torch.testing.assert_close(baseline, native_same_shape_logits, atol=0, rtol=0)
        zero_guidance = baseline + 0.0 * (baseline - reference)
        torch.testing.assert_close(zero_guidance, native_same_shape_logits, atol=0, rtol=0)
        native_full_logits = native.logits[rows, positions].float()
        if self.dtype == torch.float32:
            torch.testing.assert_close(baseline, native_full_logits, atol=2e-5, rtol=2e-5)
        native_reference = self.model(
            input_ids=input_ids, input_states=initial, num_steps=self.reference_step,
            use_cache=False, attention_mask=None,
            output_details={
                "return_logits": True, "return_latents": False,
                "return_head": True, "return_stats": False,
            },
        )
        native_reference_hidden = native_reference.hidden_states[rows, positions]
        torch.testing.assert_close(
            early_hidden[rows, positions], native_reference_hidden, atol=0, rtol=0
        )
        native_reference_same_shape_logits = self.model.lm_head(native_reference_hidden).float()
        torch.testing.assert_close(reference, native_reference_same_shape_logits, atol=0, rtol=0)
        native_reference_full_logits = native_reference.logits[rows, positions].float()
        if self.dtype == torch.float32:
            torch.testing.assert_close(reference, native_reference_full_logits, atol=2e-5, rtol=2e-5)
        if self.loops > self.reference_step and torch.equal(baseline, reference):
            raise AssertionError("The configured reference and final recurrent readouts unexpectedly coincide.")
        if not bool(torch.isfinite(baseline).all() and torch.isfinite(reference).all()):
            raise AssertionError("Nonfinite Huginn logits")
        return {
            **self.metadata(),
            "native_final_hidden_max_abs_error": (final_hidden[rows, positions] - native_hidden).abs().max().item(),
            "native_same_shape_logits_max_abs_error": (baseline - native_same_shape_logits).abs().max().item(),
            "native_full_logits_max_abs_error": (baseline - native_full_logits).abs().max().item(),
            "native_reference_hidden_max_abs_error": (
                early_hidden[rows, positions] - native_reference_hidden
            ).abs().max().item(),
            "native_reference_same_shape_logits_max_abs_error": (
                reference - native_reference_same_shape_logits
            ).abs().max().item(),
            "native_reference_full_logits_max_abs_error": (
                reference - native_reference_full_logits
            ).abs().max().item(),
            "zero_guidance_matches_native_same_shape_exactly": True,
            "early_final_mean_abs_logit_difference": (baseline - reference).abs().mean().item(),
            "same_initial_states_supplied_to_native": True,
        }
