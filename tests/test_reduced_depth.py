"""Cross-depth identity and paired-analysis invariants."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze
from scripts.analyze_reduced_depth import paired_comparison, verify_reused_baseline


class ReducedDepthTests(unittest.TestCase):
    def test_cross_run_pairing_rejects_order_and_label_mismatch(self):
        first = [{"id": str(i), "gold": 0,
                  "predictions": {"acc_norm": {"baseline": 1}}} for i in range(4)]
        second = [{"id": str(i), "gold": 0,
                   "predictions": {"acc_norm": {"loopcd_fixed": 0}}} for i in range(4)]
        result = paired_comparison(first, "baseline", second, "loopcd_fixed", "acc_norm")
        self.assertEqual(result["corrected"], 4)
        self.assertEqual(result["regressed"], 0)
        self.assertEqual(result["delta_percentage_points"], 100)
        self.assertEqual(result["mcnemar_exact_two_sided_p"], 0.125)
        with self.assertRaisesRegex(ValueError, "document order"):
            paired_comparison(first, "baseline", list(reversed(second)), "loopcd_fixed", "acc_norm")
        second[0]["gold"] = 1
        with self.assertRaisesRegex(ValueError, "labels"):
            paired_comparison(first, "baseline", second, "loopcd_fixed", "acc_norm")

    def test_baseline_replay_checks_tokens_and_predictions(self):
        record = {
            "question_index": 0, "choice_index": 0, "target_ids": [7],
            "input_length": 4, "token_logprobs": {"baseline": [-2.0]},
        }
        prediction = {
            "id": "q", "gold": 0,
            "predictions": {"acc": {"baseline": 0}, "acc_norm": {"baseline": 0}},
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(analyze, "ROOT", Path(directory)):
            old, new = Path(directory) / "old", Path(directory) / "new"
            old.mkdir()
            new.mkdir()
            for path in (old, new):
                (path / "candidate_scores.jsonl").write_text(json.dumps(record) + "\n")
                (path / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
            self.assertEqual(verify_reused_baseline(old, new)["maximum_token_logprob_difference"], 0)
            damaged = copy.deepcopy(record)
            damaged["token_logprobs"]["baseline"][0] -= 0.01
            (new / "candidate_scores.jsonl").write_text(json.dumps(damaged) + "\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_reused_baseline(old, new)
            (new / "candidate_scores.jsonl").write_text(json.dumps(record) + "\n")
            prediction["predictions"]["acc_norm"]["baseline"] = 1
            (new / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
            with self.assertRaisesRegex(ValueError, "prediction changed"):
                verify_reused_baseline(old, new)


if __name__ == "__main__":
    unittest.main()
