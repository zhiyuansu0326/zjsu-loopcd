"""Analytic matrix-product FLOPs for the released Huginn architecture."""

from __future__ import annotations

import math
import statistics


def forward_flops(config, loops, *, sequence_length=512, batch_size=1,
                  head_positions=None, guided=False, include_injection=True):
    """Count two FLOPs per multiply-accumulate, including dense attention.

    ``head_positions`` is the total number of projected positions in the batch.
    The paper's 512-token prefill projects every position. Our multiple-choice
    scorer projects answer-predicting positions only.
    """
    for name, value in (("loops", loops), ("sequence_length", sequence_length),
                        ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if head_positions is None:
        head_positions = batch_size * sequence_length
    if type(head_positions) is not int or not 1 <= head_positions <= batch_size * sequence_length:
        raise ValueError("head_positions must be between one and batch_size * sequence_length")
    width = int(config["n_embd"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["n_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    vocab = int(config["padded_vocab_size"])
    if width != heads * head_dim:
        raise ValueError("Attention output width must equal n_embd")
    tokens = batch_size * sequence_length
    qkv = 2 * tokens * width * (heads + 2 * kv_heads) * head_dim
    attention_output = 2 * tokens * width * width
    # Both Q K^T and attention V use a full sequence_length-square matrix.
    # This conventional dense count does not apply a causal triangular discount.
    attention_products = 4 * batch_size * sequence_length**2 * heads * head_dim
    gated_mlp = 6 * tokens * width * intermediate
    block = qkv + attention_output + attention_products + gated_mlp
    injection_per_step = 4 * tokens * width * width
    prelude = int(config["n_layers_in_prelude"]) * block
    recurrent_blocks = loops * int(config["n_layers_in_recurrent_block"]) * block
    recurrent_injection = loops * injection_per_step if include_injection else 0
    coda = int(config["n_layers_in_coda"]) * block
    head = 2 * head_positions * width * vocab
    components = {
        "prelude": prelude,
        "recurrent_blocks": recurrent_blocks,
        "recurrent_injection": recurrent_injection,
        "final_coda": coda,
        "final_head": head,
        "reference_coda": coda if guided else 0,
        "reference_head": head if guided else 0,
    }
    total = sum(components.values())
    return {
        "loops": loops, "guided": guided, "sequence_length": sequence_length,
        "batch_size": batch_size, "head_positions": head_positions,
        "include_recurrent_injection": include_injection,
        "components_flops": components, "total_flops": total,
        "total_tflops": total / 1e12,
        "per_block_flops": {
            "qkv": qkv, "attention_output": attention_output,
            "attention_products": attention_products, "gated_mlp": gated_mlp,
        },
    }


def comparison(config, **shape):
    """Three policies, each normalized to the same full-depth baseline."""
    policies = {
        "baseline_r32": forward_flops(config, 32, **shape),
        "baseline_r16": forward_flops(config, 16, **shape),
        "loopcd_r16_ref5": forward_flops(config, 16, guided=True, **shape),
    }
    full = policies["baseline_r32"]["total_flops"]
    for value in policies.values():
        value["fraction_of_baseline_r32"] = value["total_flops"] / full
        value["removed_percent"] = 100 * (1 - value["total_flops"] / full)
    return policies


def paper_prefill_comparison(config):
    """Expose the actual injection matrix separately from the printed table."""
    return {
        "protocol": {
            "sequence_length": 512, "batch_size": 1, "head_positions": 512,
            "multiply_accumulate_flops": 2,
            "attention_count": "Dense QK and AV matrix products, no causal triangle discount.",
            "counted": "Attention and MLP matrices, recurrent injection, coda, and vocabulary head.",
            "excluded": "Embedding lookup, normalization, RoPE, activations, biases, residual additions, and guidance vector arithmetic.",
            "paper_source": "sources/looped_qwen/loopcd.html, Appendix D.1 and Tables A7-A9",
        },
        "including_recurrent_injection": comparison(config),
        "matching_paper_table_convention": comparison(config, include_injection=False),
        "paper_reported": {
            "baseline_r32_tflops": 54.526,
            "baseline_r16_tflops": 28.261,
            "loopcd_r16_fraction_of_baseline_r32": 0.54,
            "loopcd_r16_removed_percent": 46.0,
        },
        "accounting_difference": (
            "The printed Huginn TFLOPs match the Transformer blocks and head with "
            "dense attention. Including the released core_block_forward's 2d-to-d "
            "injection matrix adds one projection per recurrent step. Both ledgers "
            "are shown; the injection-inclusive ledger counts all forward matrix products."
        ),
    }


def sample_statistics(values):
    values = list(values)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Expected nonempty finite nonnegative measurements")
    ordered = sorted(values)

    def quantile(probability):
        index = (len(ordered) - 1) * probability
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    median = statistics.median(values)
    return {
        "n": len(values), "median": median,
        "mean": statistics.mean(values), "minimum": min(values), "maximum": max(values),
        "p25": quantile(0.25), "p75": quantile(0.75),
        "iqr": quantile(0.75) - quantile(0.25),
        "median_absolute_deviation": statistics.median(abs(value - median) for value in values),
    }
