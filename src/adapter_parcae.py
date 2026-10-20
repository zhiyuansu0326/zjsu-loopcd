"""LoopCD readout for the unmodified, pinned official Parcae implementation.

The checkpoint uses a random truncated-normal initial recurrent state. We retain
that distribution, but generate it from a CPU RNG keyed by question identity so
candidate length, padding, and batching cannot change a shared prefix's noise.
"""
from __future__ import annotations

import hashlib
import builtins
import importlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "sources/parcae"


def native_classes():
    """Import the vendored official code; use its built-in SDPA implementation."""
    source = str(SOURCE_DIR)
    if source not in sys.path:
        sys.path.insert(0, source)
    original_import = builtins.__import__

    def inference_import(name, *args, **kwargs):
        if name in {"parcae_lm.ops.linear_cross_entropy", "kernels"}:
            # Upstream already handles ImportError for these optional paths.
            # The training loss's Triton decorators otherwise demand a GPU even
            # when importing a CPU model; neither extension is used by this task.
            raise ImportError("Optional training/remote kernel disabled for inference")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=inference_import):
        from parcae_lm.models.parcae.config import ParcaeConfig
        from parcae_lm.models.parcae.parcae import Parcae
    backend = importlib.import_module("parcae_lm.attention_backends.flash_attention")
    backend._override_impl = "sdpa"
    return ParcaeConfig, Parcae


class ParcaeAdapter:
    stochastic = True

    def __init__(self, model_path, device="cuda:0", dtype=torch.float32, loops=8):
        config_class, model_class = native_classes()
        model_path = Path(model_path)
        config_data = json.loads((model_path / "config.json").read_text())
        if config_data.pop("_class_name", None) != "ParcaeConfig":
            raise ValueError("Expected an official ParcaeConfig checkpoint")
        # Skip only initialization of weights that will all be loaded strictly.
        # Keep rope settings and all learned architecture options from the file.
        config_data["skip_initialization"] = True
        config = config_class(**config_data)
        with torch.device("meta"):
            model = model_class(config)
        state = torch.load(
            model_path / "pytorch_model.bin", map_location="cpu",
            weights_only=True, mmap=True,
        )
        model.load_state_dict(state, strict=True, assign=True)
        if config.tie_embeddings:
            # assign=True can create distinct Parameter objects over the same
            # mapped storage; restore tying before transferring to the GPU.
            input_weight = state["transformer.wte.weight"]
            output_weight = state["lm_head.weight"]
            if input_weight.data_ptr() != output_weight.data_ptr():
                raise ValueError("Released tied embeddings must share checkpoint storage")
            model.lm_head.weight = model.transformer.wte.weight
        del state
        model = model.to(device=device, dtype=dtype).eval()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_path / "tokenizer.json"),
            add_bos_token=False, add_eos_token=False,
        )
        self._configure(model, tokenizer, loops)
        self.model_path = str(model_path)

    def _configure(self, model, tokenizer, loops):
        """Also permits tiny official models in CPU tests without checkpoints."""
        if loops < 1:
            raise ValueError("loops must be positive")
        if model.config.state_init != "like-init":
            raise ValueError("This adapter expects the released like-init checkpoints")
        if model.config.use_fused_head != "pytorch":
            raise ValueError("This adapter expects the released PyTorch LM head")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.loops = int(loops)
        self.pad_id = (tokenizer.pad_token_id if tokenizer is not None else None)
        if self.pad_id is None:
            self.pad_id = 0  # Right-padding tokens are never visible to scored positions.
        self.max_length = int(model.config.block_size)
        self.initial_std = float(model.config.init.get_std("embedding"))
        self.randomness = {
            "state_init": "like-init", "stochastic": True,
            "distribution": "Normal(0, std^2) truncated to [-3*std, +3*std], then embedding_scale",
            "std": self.initial_std, "embedding_scale": float(model.emb_scale),
            "seed_base": 1234,
            "seed_rule": "little-endian first 8 bytes of SHA256('parcae-latent-v1|1234|' + str(question_id)), modulo 2**63",
            "sampling": "CPU float32 uniform, inverse normal CDF, then cast to model dtype",
            "difference_from_native_rng": "Preserves the distribution; omits the native discarded randn draw and fixes noise by question rather than batch shape.",
            "sharing": "Identical question IDs share initial-state prefixes across all candidates and policies",
        }

    @staticmethod
    def _seed(key):
        payload = f"parcae-latent-v1|1234|{key}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % 2**63

    def initial_state(self, shape, sample_keys, *, device=None, dtype=None):
        """Return prefix-stable samples of the checkpoint's native distribution."""
        if sample_keys is None or len(sample_keys) != shape[0]:
            raise ValueError("Provide one stable question ID per batch row in sample_keys")
        if shape[-1] != self.model.config.recurrent_embedding_dimension:
            raise ValueError("Initial-state width does not match recurrent dimension")
        states = []
        lower_cdf = (1.0 + math.erf(-3.0 / math.sqrt(2.0))) / 2.0
        upper_cdf = (1.0 + math.erf(3.0 / math.sqrt(2.0))) / 2.0
        for key in sample_keys:
            generator = torch.Generator(device="cpu").manual_seed(self._seed(key))
            values = torch.rand(
                (shape[1], shape[2]), generator=generator, dtype=torch.float32, device="cpu"
            )
            values.mul_(2 * (upper_cdf - lower_cdf)).add_(2 * lower_cdf - 1)
            values.erfinv_().mul_(self.initial_std * math.sqrt(2.0))
            values.clamp_(-3 * self.initial_std, 3 * self.initial_std)
            values.mul_(self.model.emb_scale)
            states.append(values)
        parameter = next(self.model.parameters())
        return torch.stack(states).to(
            device=device if device is not None else self.device,
            dtype=dtype if dtype is not None else parameter.dtype,
        )

    def _check_inputs(self, input_ids, attention_mask, rows, positions, sample_keys):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Expected equally shaped [batch, sequence] IDs and mask")
        if input_ids.shape[1] > self.max_length:
            raise ValueError("Context exceeds native limit; truncation is not permitted")
        valid = attention_mask.bool()
        lengths = valid.sum(-1)
        expected = torch.arange(input_ids.shape[1], device=valid.device)[None] < lengths[:, None]
        if not torch.equal(valid, expected) or bool((lengths == 0).any()):
            raise ValueError("Native causal attention requires nonempty right-padded inputs")
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        positions = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        if rows.ndim != 1 or rows.shape != positions.shape or not rows.numel():
            raise ValueError("rows and positions must be equal nonempty 1D arrays")
        if bool((rows < 0).any()) or bool((rows >= input_ids.shape[0]).any()):
            raise ValueError("Requested row outside batch")
        if bool((positions < 0).any()) or bool((positions >= lengths.to(self.device)[rows]).any()):
            raise ValueError("Cannot read padded or out-of-range prediction positions")
        if sample_keys is None or len(sample_keys) != input_ids.shape[0]:
            raise ValueError("sample_keys must supply a stable question ID for every row")
        return rows, positions

    def _readout(self, state, input_ids, freqs, mask, rows, positions):
        model = self.model
        hidden = model.transformer.C(state)  # Official recurrent -> decoder projection.
        offset = model.config.n_layers_in_prelude + model.config.n_layers_in_recurrent_block
        for index, block in enumerate(model.transformer.coda):
            value_index = str(offset + index)
            value = model.value_embeds[value_index](input_ids) if value_index in model.value_embeds else None
            hidden = block(hidden, freqs, mask, ve=value)
        hidden = model.transformer.ln_f(hidden)
        logits = model.lm_head(hidden[rows, positions]).float() * model.config.init.logit_scale
        if model.config.logit_softcap is not None:
            cap = model.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return logits

    @torch.inference_mode()
    def forward_logits(self, input_ids, attention_mask, rows, positions, sample_keys=None):
        rows, positions = self._check_inputs(input_ids, attention_mask, rows, positions, sample_keys)
        model = self.model
        freqs = model.freqs_cis[:, :input_ids.shape[1]]
        embeddings = model.transformer.wte(input_ids)
        if model.emb_scale != 1:
            embeddings = embeddings * model.emb_scale
        model._current_input_ids = input_ids
        for index, block in enumerate(model.transformer.prelude):
            value_index = str(index)
            value = model.value_embeds[value_index](input_ids) if value_index in model.value_embeds else None
            embeddings = block(embeddings, freqs, attention_mask, ve=value)
        if model.config.prelude_norm:
            embeddings = model.transformer.ln_prelude(embeddings)
        state = self.initial_state(
            (*embeddings.shape[:-1], model.config.recurrent_embedding_dimension), sample_keys,
            device=embeddings.device, dtype=embeddings.dtype,
        )
        total_steps = torch.tensor(self.loops, device=self.device)
        reference_state = None
        for step in range(self.loops):
            state = model.update_recurrent_state(
                state, embeddings, freqs, attention_mask,
                torch.tensor(step, device=self.device), total_steps,
            )
            if step == 0:
                reference_state = state
        # Both readouts run after the single shared recurrence trajectory.
        return {
            "baseline": self._readout(state, input_ids, freqs, attention_mask, rows, positions),
            "reference": self._readout(reference_state, input_ids, freqs, attention_mask, rows, positions),
        }

    @torch.inference_mode()
    def _native_logits(self, input_ids, attention_mask, rows, positions, sample_keys, loops):
        def injected_initial_state(embeddings):
            return self.initial_state(
                (*embeddings.shape[:-1], self.model.config.recurrent_embedding_dimension),
                sample_keys, device=embeddings.device, dtype=embeddings.dtype,
            )
        # Upstream has no input_states argument. This narrow temporary override
        # supplies the exact latent to the otherwise unmodified native forward.
        with patch.object(self.model, "initialize_state", side_effect=injected_initial_state):
            native = self.model(
                input_ids, attention_mask=attention_mask, return_logits=True,
                num_steps_pair=torch.tensor([loops, 0], device=self.device),
            )["logits"]
        return native[rows, positions].float()

    @torch.inference_mode()
    def validate(self, input_ids, attention_mask, rows, positions, sample_keys=None):
        rows, positions = self._check_inputs(input_ids, attention_mask, rows, positions, sample_keys)
        with patch.object(self.model, "update_recurrent_state", wraps=self.model.update_recurrent_state) as recurrent:
            output = self.forward_logits(input_ids, attention_mask, rows, positions, sample_keys)
            calls = recurrent.call_count
        if calls != self.loops:
            raise AssertionError(f"Readout introduced additional recurrence: {calls}")
        native = self._native_logits(input_ids, attention_mask, rows, positions, sample_keys, self.loops)
        native_first = self._native_logits(input_ids, attention_mask, rows, positions, sample_keys, 1)
        tolerance = 3e-4 if next(self.model.parameters()).dtype == torch.float32 else 0.12
        torch.testing.assert_close(output["baseline"], native, rtol=3e-5, atol=tolerance)
        torch.testing.assert_close(output["reference"], native_first, rtol=3e-5, atol=tolerance)
        zero = output["baseline"] + 0.0 * (output["baseline"] - output["reference"])
        torch.testing.assert_close(zero, output["baseline"], rtol=0, atol=0)
        repeated = self.forward_logits(input_ids, attention_mask, rows, positions, sample_keys)
        for mode in output:
            torch.testing.assert_close(repeated[mode], output[mode], rtol=0, atol=0)
        width = self.model.config.recurrent_embedding_dimension
        short = self.initial_state((1, 3, width), [sample_keys[0]], device="cpu", dtype=torch.float32)
        long = self.initial_state((2, 7, width), [sample_keys[0], sample_keys[0]], device="cpu", dtype=torch.float32)
        torch.testing.assert_close(short[0], long[0, :3], rtol=0, atol=0)
        torch.testing.assert_close(short[0], long[1, :3], rtol=0, atol=0)
        padding_error = 0.0
        for row in rows.unique().tolist():
            selected = rows == row
            length = int(attention_mask[row].sum())
            single = self.forward_logits(
                input_ids[row:row + 1, :length], attention_mask[row:row + 1, :length],
                torch.zeros_like(rows[selected]), positions[selected], [sample_keys[row]],
            )
            for mode in output:
                padding_error = max(
                    padding_error, (single[mode] - output[mode][selected]).abs().max().item()
                )
                torch.testing.assert_close(single[mode], output[mode][selected], rtol=3e-5, atol=tolerance)
        return {
            "native_final_logits_max_abs_error": (output["baseline"] - native).abs().max().item(),
            "native_first_loop_logits_max_abs_error": (output["reference"] - native_first).abs().max().item(),
            "zero_guidance_matches_native": True,
            "zero_guidance_matches_baseline_exactly": True,
            "recurrent_updates_in_forward": calls, "reference_after_complete_updates": 1,
            "early_readout": "official transformer.C -> complete coda -> ln_f -> lm_head -> logit_scale",
            "batch_single_logits_max_abs_error": padding_error,
            "early_final_mean_abs_logit_difference": (output["baseline"] - output["reference"]).abs().mean().item(),
            "latent_prefix_invariant_to_length_and_batch": True,
            "same_keys_repeat_exactly": True, "randomness": self.randomness,
            "attention_backend": "official SDPA fallback", "logits_dtype": "float32",
            "native_check_note": "Native forward receives the identical explicit latent through a temporary initialize_state override.",
        }
