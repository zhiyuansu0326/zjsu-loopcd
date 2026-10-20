"""Ouro adapter for the multi-model evaluator; existing single-model code stays frozen."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class OuroAdapter:
    stochastic = False

    def __init__(self, model_path, device="cuda:0", dtype=torch.float32, loops=4):
        if loops != 4:
            raise ValueError("The Ouro Table 2 protocol uses four recurrent passes.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
            dtype=dtype, device_map={"": device}, attn_implementation="sdpa",
        ).eval()
        self.device = torch.device(device)
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.max_length = self.model.config.max_position_embeddings
        self.loops = loops
        assert self.model.model.total_ut_steps == loops

    @torch.inference_mode()
    def forward_logits(self, input_ids, attention_mask, rows, positions, sample_keys=None):
        _, states, _ = self.model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        assert len(states) == self.loops
        return {
            "baseline": self.model.lm_head(states[-1][rows, positions]).float(),
            "reference": self.model.lm_head(states[0][rows, positions]).float(),
        }

    @torch.inference_mode()
    def validate(self, input_ids, attention_mask, rows, positions, sample_keys=None):
        ours = self.forward_logits(input_ids, attention_mask, rows, positions)
        columns = torch.unique(positions, sorted=True)
        remap = torch.searchsorted(columns, positions)
        errors = {}
        for mode, index in (("baseline", self.loops - 1), ("reference", 0)):
            native = self.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                exit_at_step=index, logits_to_keep=columns,
            ).logits[rows, remap].float()
            errors[mode] = (ours[mode] - native).abs().max().item()
            torch.testing.assert_close(ours[mode], native, rtol=1e-5, atol=1e-5)
        return {
            "native_full_head_max_abs_error": errors,
            "complete_recurrent_passes": self.loops,
            "reference_pass": 1,
            "zero_guidance_identity": True,
        }
