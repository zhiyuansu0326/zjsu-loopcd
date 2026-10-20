"""Finalization must never publish completion after an incomplete/failed audit."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import finalize_results as finalize


class FinalizationTests(unittest.TestCase):
    def invoke(self, root, run):
        base = root / "results/multitask"
        with patch.object(finalize, "ROOT", root), patch.object(finalize, "BASE", base), \
                patch("sys.argv", ["finalize_results.py"]), \
                patch.object(finalize.subprocess, "run", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()):
            finalize.main()

    def test_audit_failure_records_failure_without_publishing_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = 0

            def run(*args, **kwargs):
                nonlocal calls
                calls += 1
                status = json.loads((root / "results/multitask/completion.json").read_text())
                self.assertFalse(status["complete"])
                self.assertEqual(status["state"], "running")
                if calls == 2:
                    raise subprocess.CalledProcessError(1, "analyze.py")

            with self.assertRaises(subprocess.CalledProcessError):
                self.invoke(root, run)
            base = root / "results/multitask"
            self.assertFalse(json.loads((base / "completion.json").read_text())["complete"])
            self.assertFalse((base / "artifact_manifest.json").exists())

    def test_success_requires_all_28_configurations_and_records_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "results/multitask"
            base.mkdir(parents=True)
            (root / "README.md").write_text(finalize.PENDING + "\n")
            model = {"n": 3, "audit": {
                "unique_candidates": 12, "answer_token_count_per_policy": 20,
            }}
            fixture = {"complete": True, "tasks": {
                f"task{t}": {"models": {f"model{m}": model for m in range(7)}}
                for t in range(4)
            }}
            (base / "comparison.json").write_text(json.dumps(fixture))
            self.invoke(root, lambda *args, **kwargs: None)
            status = json.loads((base / "completion.json").read_text())
            artifact = json.loads((base / "artifact_manifest.json").read_text())
            self.assertTrue(status["complete"])
            self.assertEqual(status["configurations"], 28)
            self.assertEqual(status["question_evaluations"], 84)
            self.assertEqual(status["scored_candidates"], 336)
            self.assertEqual(status["answer_token_logprobs_all_four_policies"], 2240)
            self.assertIn("README.md", artifact["files"])
            self.assertNotIn("results/multitask/completion.json", artifact["files"])
            self.assertIn(finalize.FINISHED, (root / "README.md").read_text())
            fixture["tasks"]["task3"]["models"].pop("model6")
            (base / "comparison.json").write_text(json.dumps(fixture))
            with self.assertRaisesRegex(ValueError, "28 complete"):
                self.invoke(root, lambda *args, **kwargs: None)
            self.assertFalse(json.loads((base / "completion.json").read_text())["complete"])
            self.assertFalse((base / "artifact_manifest.json").exists())
            self.assertEqual(len(list((base / "finalization_history").glob("*/artifact_manifest.json"))), 1)
            self.assertNotIn(finalize.FINISHED, (root / "README.md").read_text())
            self.assertIn(finalize.PENDING, (root / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
