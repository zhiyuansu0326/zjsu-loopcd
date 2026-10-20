"""Resume integrity checks using temporary metadata and tiny synthetic records."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.run_integrity import (
    DEFAULT_MODES,
    compare_resume_manifests,
    load_completed_scores,
    native_source_hashes,
    sha256_file,
    validate_resume_manifest,
)


def manifest():
    return {
        "name": "test_model", "configuration": {"family": "parcae", "loops": 8},
        "documents": 2, "candidates": 8, "data_sha256": "data",
        "model_manifest_sha256": "model", "code_hashes": {"adapter.py": "code"},
        "native_source_hashes": {"native.py": "native"}, "dtype": "float32",
        "float32_matmul_precision": "high", "batch_size": 2, "token_budget": 128,
        "seed": 1234, "add_bos": False, "chat_template": False,
        "max_input_tokens": 12, "truncation": False, "torch": "2.6.0",
        "transformers": "4.56.2", "adapter_metadata": {"seed": 1234},
        "gpu": "H200", "host": "node-a", "slurm_job_id": "123",
    }


def candidates():
    return [
        SimpleNamespace(
            question_index=q, choice_index=c, targets=[10 + c, 20 + c],
            input_ids=[1, 2, 3, 10 + c],
        )
        for q, count in ((0, 3), (1, 5))
        for c in range(count)
    ]


def record(candidate):
    values = [-0.2, -0.4]
    return {
        "question_index": candidate.question_index, "choice_index": candidate.choice_index,
        "target_ids": candidate.targets, "input_length": len(candidate.input_ids),
        "scores": {mode: sum(values) for mode in DEFAULT_MODES},
        "token_logprobs": {mode: values[:] for mode in DEFAULT_MODES},
    }


class ManifestTests(unittest.TestCase):
    def test_every_recorded_numerical_setting_and_source_must_match(self):
        old = manifest()
        changed_values = {
            "seed": 4321, "torch": "2.7.0", "transformers": "4.57.0",
            "adapter_metadata": {"seed": 4321},
            "native_source_hashes": {"native.py": "changed"},
            "configuration": {"family": "parcae", "loops": 4},
            "batch_size": 4, "dtype": "bfloat16", "gpu": "H100",
        }
        for key, value in changed_values.items():
            with self.subTest(key=key):
                new = copy.deepcopy(old)
                new[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    compare_resume_manifests(old, new)
        newer = {**old, "new_numerical_option": 1}
        with self.assertRaisesRegex(ValueError, "new_numerical_option"):
            compare_resume_manifests(old, newer)

    def test_only_host_and_job_identifiers_may_change(self):
        old = manifest()
        new = {**old, "host": "node-b", "slurm_job_id": "999"}
        compare_resume_manifests(old, new)

    def test_json_boolean_is_not_equal_to_integer_in_nested_metadata(self):
        old, new = manifest(), manifest()
        old["adapter_metadata"] = {"flag": 1}
        new["adapter_metadata"] = {"flag": True}
        with self.assertRaisesRegex(ValueError, "adapter_metadata"):
            compare_resume_manifests(old, new)

    def test_legacy_missing_source_hashes_cannot_be_silently_upgraded(self):
        old = manifest()
        del old["native_source_hashes"]
        with self.assertRaisesRegex(ValueError, "native_source_hashes"):
            compare_resume_manifests(old, manifest())

    def test_orphan_scores_rejected_before_writing_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            metadata, scores = folder / "run_manifest.json", folder / "candidate_scores.jsonl"
            scores.write_text(json.dumps(record(candidates()[0])) + "\n")
            before = scores.read_bytes()
            with self.assertRaisesRegex(ValueError, "without a previous run manifest"):
                validate_resume_manifest(metadata, scores, manifest())
            self.assertFalse(metadata.exists())
            self.assertEqual(scores.read_bytes(), before)
            scores.write_text("")
            self.assertIsNone(validate_resume_manifest(metadata, scores, manifest()))
            self.assertFalse(metadata.exists())

    def test_valid_manifest_is_compared_without_rewriting_its_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            metadata, scores = folder / "run_manifest.json", folder / "candidate_scores.jsonl"
            original = json.dumps(manifest(), indent=4).encode()
            metadata.write_bytes(original)
            before = metadata.stat().st_mtime_ns
            old = validate_resume_manifest(metadata, scores, manifest())
            self.assertEqual(old, manifest())
            self.assertEqual(metadata.read_bytes(), original)
            self.assertEqual(metadata.stat().st_mtime_ns, before)


class SavedScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "candidate_scores.jsonl"
        self.candidates = candidates()

    def write_rows(self, rows):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_partial_and_complete_sets_support_three_and_five_choices(self):
        self.assertEqual(load_completed_scores(self.path, self.candidates), {})
        rows = [record(c) for c in reversed(self.candidates)]
        self.write_rows(rows[:3])
        completed = load_completed_scores(self.path, self.candidates)
        self.assertEqual(set(completed), {(1, 4), (1, 3), (1, 2)})
        self.write_rows(rows)
        completed = load_completed_scores(self.path, self.candidates)
        self.assertEqual(len(completed), 8)

    def test_wrong_indices_targets_length_and_duplicates_are_rejected(self):
        bad_cases = [
            ("choice_index", 3, "outside"),  # Question zero has only three choices.
            ("question_index", 2, "outside"),
            ("question_index", True, "integers"),
            ("target_ids", [11, 21], "target_ids"),
            ("target_ids", [True, 20], "target_ids"),
            ("input_length", 5, "input_length"),
            ("input_length", True, "input_length"),
        ]
        for key, value, expected in bad_cases:
            with self.subTest(key=key, value=value):
                row = record(self.candidates[0])
                row[key] = value
                self.write_rows([row])
                with self.assertRaisesRegex(ValueError, expected):
                    load_completed_scores(self.path, self.candidates)
        row = record(self.candidates[0])
        self.write_rows([row, row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_completed_scores(self.path, self.candidates)

    def test_mode_token_count_finiteness_and_aggregate_integrity(self):
        for problem in ("missing_mode", "token_count", "nan", "wrong_sum"):
            with self.subTest(problem=problem):
                row = record(self.candidates[0])
                if problem == "missing_mode":
                    del row["scores"]["reference"]
                elif problem == "token_count":
                    row["token_logprobs"]["baseline"] = [-0.6]
                elif problem == "nan":
                    row["scores"]["baseline"] = float("nan")
                else:
                    row["scores"]["baseline"] = 100.0
                self.write_rows([row])
                with self.assertRaises(ValueError):
                    load_completed_scores(self.path, self.candidates)

    def test_truncated_last_record_is_not_silently_discarded(self):
        valid = json.dumps(record(self.candidates[0])) + "\n"
        self.path.write_text(valid + '{"question_index":')
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, r":2:.*truncated"):
            load_completed_scores(self.path, self.candidates)
        self.assertEqual(self.path.read_bytes(), before)


class SourceHashTests(unittest.TestCase):
    def test_streaming_hash_does_not_use_read_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "large.bin"
            data = b"known-content" * 200000
            path.write_bytes(data)
            expected = hashlib.sha256(data).hexdigest()
            with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
                self.assertEqual(sha256_file(path), expected)

    def test_checkpoint_native_and_parcae_recursive_sources_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "models/test"
            model.mkdir(parents=True)
            (model / "modeling.py").write_text("x = 1\n")
            (model / "config.json").write_text("{}")
            for family in ("ouro", "huginn"):
                hashes = native_source_hashes(family, "models/test", root=root)
                self.assertEqual(set(hashes), {"models/test/modeling.py"})
            source = root / "sources/parcae/nested"
            source.mkdir(parents=True)
            native = source / "model.py"
            native.write_text("x = 1\n")
            first = native_source_hashes("parcae", model, root=root)
            native.write_text("x = 2\n")
            second = native_source_hashes("parcae", model, root=root)
            self.assertEqual(set(first), {"sources/parcae/nested/model.py"})
            self.assertNotEqual(first, second)
            with self.assertRaises(FileNotFoundError):
                native_source_hashes("ouro", "missing", root=root)

    def test_qwen_hashes_installed_package_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder) / "qwen3"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "modeling_qwen3.py").write_text("official_model = True\n")
            spec = SimpleNamespace(submodule_search_locations=[str(package)])
            with patch("src.run_integrity.importlib.util.find_spec", return_value=spec) as find:
                result = native_source_hashes("looped_qwen", "unused")
            find.assert_called_once_with("transformers.models.qwen3")
            self.assertEqual(set(result), {
                "transformers/models/qwen3/__init__.py",
                "transformers/models/qwen3/modeling_qwen3.py",
            })


if __name__ == "__main__":
    unittest.main()
