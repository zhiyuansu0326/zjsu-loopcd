"""Cross-check analytic costs against the paper and released matrix dimensions."""

import json
from pathlib import Path
import unittest

from src.huginn_costs import forward_flops, paper_prefill_comparison, sample_statistics


ROOT = Path(__file__).resolve().parents[1]


class HuginnCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "models/Huginn-0125/config.json").read_text())

    def test_printed_paper_tflops_recovered_with_explicit_injection_difference(self):
        result = paper_prefill_comparison(self.config)
        printed = result["matching_paper_table_convention"]
        self.assertEqual(round(printed["baseline_r32"]["total_tflops"], 3), 54.526)
        self.assertEqual(round(printed["baseline_r16"]["total_tflops"], 3), 28.261)
        self.assertEqual(round(printed["loopcd_r16_ref5"]["fraction_of_baseline_r32"], 2), 0.54)
        inclusive = result["including_recurrent_injection"]["baseline_r32"]
        # Native adapter is nn.Linear(2 * n_embd, n_embd), called on every step.
        extra = 32 * 512 * (2 * 5280) * 5280 * 2
        self.assertEqual(inclusive["total_flops"] - printed["baseline_r32"]["total_flops"], extra)

    def test_guidance_adds_exactly_one_complete_coda_and_selected_position_head(self):
        base = forward_flops(self.config, 16, sequence_length=73, batch_size=8, head_positions=39)
        guided = forward_flops(self.config, 16, sequence_length=73, batch_size=8,
                               head_positions=39, guided=True)
        self.assertEqual(
            guided["total_flops"] - base["total_flops"],
            base["components_flops"]["final_coda"] + 2 * 39 * 5280 * 65536,
        )
        full_head = forward_flops(self.config, 16, sequence_length=73, batch_size=8)
        self.assertEqual(
            full_head["total_flops"] - base["total_flops"],
            2 * (73 * 8 - 39) * 5280 * 65536,
        )

    def test_invalid_shapes_and_nonfinite_measurements_are_rejected(self):
        for args in ({"sequence_length": 0}, {"batch_size": -1}, {"head_positions": 513}):
            with self.assertRaises(ValueError):
                forward_flops(self.config, 16, **args)
        with self.assertRaises(ValueError):
            sample_statistics([1, float("nan")])
        stats = sample_statistics([1, 2, 3, 4])
        self.assertEqual(stats["median"], 2.5)
        self.assertEqual(stats["iqr"], 1.5)


if __name__ == "__main__":
    unittest.main()
