"""Matrix-product and guidance-vector arithmetic for same-depth Huginn policies."""

from src.huginn_costs import forward_flops


POLICIES = ("baseline", "loopcd_logits", "loopcd_hidden")


def comparison(config, loops, *, sequence_length=512, batch_size=1,
               head_positions=None, include_injection=True):
    if loops not in (16, 32):
        raise ValueError("This experiment uses Huginn R16 and R32")
    shape = {"sequence_length": sequence_length, "batch_size": batch_size,
             "head_positions": head_positions, "include_injection": include_injection}
    result = {
        "baseline": forward_flops(config, loops, **shape),
        "loopcd_logits": forward_flops(config, loops, guided=True, **shape),
        "loopcd_hidden": forward_flops(config, loops, **shape),
    }
    baseline = result["baseline"]["total_flops"]
    for policy, row in result.items():
        row["guidance_space"] = {"baseline": None, "loopcd_logits": "logits", "loopcd_hidden": "hidden"}[policy]
        row["guided"] = policy != "baseline"
        row["fraction_of_baseline"] = row["total_flops"] / baseline
        row["matrix_flops_overhead_percent"] = 100 * (row["total_flops"] / baseline - 1)
        if policy == "loopcd_hidden":
            elements = batch_size * sequence_length * int(config["n_embd"])
        elif policy == "loopcd_logits":
            elements = row["head_positions"] * int(config["padded_vocab_size"])
        else:
            elements = 0
        row["guidance_elementwise_operations"] = 3 * elements
        row["guidance_elementwise_formula"] = "One subtraction, scalar multiplication, and addition per guided element."
    return result


def prefill_comparison(config, loops):
    return {
        "protocol": {
            "sequence_length": 512, "batch_size": 1, "head_positions": 512,
            "multiply_accumulate_flops": 2,
            "attention_count": "Dense QK and AV matrix products, no causal triangle discount.",
            "counted": "Attention and MLP matrices, recurrent injection, coda, and vocabulary head.",
            "excluded": "Embedding lookup, normalization, RoPE, activations, biases, residual additions, and guidance vector arithmetic.",
            "guidance_vector_arithmetic": "Reported separately; not added to the matrix-product FLOP total.",
            "paper_source": "sources/looped_qwen/loopcd.html, Appendix D.1 and Table A7",
        },
        "including_recurrent_injection": comparison(config, loops),
        "matching_paper_table_convention": comparison(config, loops, include_injection=False),
        "paper_reported": {
            "baseline_tflops": 28.261 if loops == 16 else 54.526,
            "logits_fraction_of_baseline": 1.042 if loops == 16 else 1.022,
            "hidden_fraction_of_baseline": 1.000,
        },
        "accounting_difference": (
            "The paper's printed Huginn TFLOPs match Transformer blocks and head with dense "
            "attention, excluding the recurrent 2d-to-d injection matrix. Both ledgers are retained."
        ),
        "hidden_guidance_cost": (
            "Hidden guidance has the same matrix products as baseline and one complete coda/head. "
            "It additionally captures the reference state and blends raw recurrent states over "
            "the full sequence; these operations can affect measured latency and memory."
        ),
    }
