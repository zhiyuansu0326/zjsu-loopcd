"""Frozen templates, label alignment, full-split provenance, and cache integrity."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jinja2 import Environment, StrictUndefined
import yaml

from src import data_external as data


def sciq_fixture(support=" \tA support paragraph.  \n"):
    return {
        "question": "Which answer?", "support": support, "distractor1": "one",
        "distractor2": "two", "distractor3": "three", "correct_answer": "four",
    }


def piqa_fixture():
    return {"id": "source-uuid", "goal": "Reach the shelf.", "sol1": "Use a ladder.", "sol2": "Use a cup."}


class TemplateTests(unittest.TestCase):
    def test_sciq_preserves_support_trailing_whitespace_and_choice_order(self):
        row = data.normalize_sciq(sciq_fixture(), 7)
        self.assertEqual(row["context"], "A support paragraph.  \n\nQuestion: Which answer?\nAnswer:")
        self.assertEqual(row["choices"], ["one", "two", "three", "four"])
        self.assertEqual(row["gold"], 3)
        self.assertEqual(row["id"], "sciq:test:000007")
        self.assertEqual(row["support"], sciq_fixture()["support"])

    def test_sciq_empty_support_keeps_harness_leading_newline(self):
        row = data.normalize_sciq(sciq_fixture(""), 0)
        self.assertEqual(row["context"], "\nQuestion: Which answer?\nAnswer:")

    def test_piqa_label_does_not_change_prompt_or_order(self):
        first = data.normalize_piqa(piqa_fixture(), 0, 3)
        second = data.normalize_piqa(piqa_fixture(), 1, 3)
        self.assertEqual(first["context"], "Question: Reach the shelf.\nAnswer:")
        self.assertEqual(first["choices"], ["Use a ladder.", "Use a cup."])
        self.assertEqual(first["id"], "piqa:validation:source-uuid")
        self.assertEqual({k: v for k, v in first.items() if k != "gold"},
                         {k: v for k, v in second.items() if k != "gold"})

    def test_invalid_label_and_fewshot_rejected(self):
        for label in (-1, 2, True, "0"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                data.normalize_piqa(piqa_fixture(), label, 0)
        with self.assertRaisesRegex(ValueError, "zero-shot"):
            data.prepare_external("sciq", shots=8)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            data.prepare_external("unsupported")

    def test_frozen_harness_jinja_matches_both_templates(self):
        environment = Environment(undefined=StrictUndefined)
        for task, raw, normalizer in (
            ("sciq", sciq_fixture(), lambda r: data.normalize_sciq(r, 0)),
            ("piqa", {**piqa_fixture(), "label": 1}, lambda r: data.normalize_piqa(r, r["label"], 0)),
        ):
            with self.subTest(task=task):
                path = data.ROOT / f"sources/lm_eval/lm_eval/tasks/{task}/{task}.yaml"
                self.assertEqual(data.sha256_file(path), data.HARNESS_FILES[task][task + ".yaml"])
                config = yaml.safe_load(path.read_text())
                row = normalizer(raw)
                self.assertEqual(environment.from_string(config["doc_to_text"]).render(**raw), row["context"])
                rendered_choices = environment.from_string(config["doc_to_choice"]).render(**raw)
                self.assertEqual(ast.literal_eval(rendered_choices), row["choices"])
                self.assertEqual(config["doc_to_target"] if task == "sciq" else raw[config["doc_to_target"]], row["gold"])


class IntegrityTests(unittest.TestCase):
    def test_corrupt_cache_is_rejected_without_network_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cached"
            path.write_bytes(b"corrupt")
            with patch.object(data.urllib.request, "urlopen", side_effect=AssertionError("No network")):
                with self.assertRaisesRegex(ValueError, "Existing source checksum"):
                    data._download_verified(path, "https://example.invalid/data", hashlib.sha256(b"valid").hexdigest())
            self.assertEqual(path.read_bytes(), b"corrupt")

    def test_full_canonical_splits_and_pinned_harness_match(self):
        environment = Environment(undefined=StrictUndefined)
        for task in data.COUNTS:
            with self.subTest(task=task):
                protocol = json.loads((data.ROOT / f"data/tasks/{task}/protocol.json").read_text())
                output = data.ROOT / protocol["output"]["path"]
                self.assertEqual(data.sha256_file(output), protocol["output"]["sha256"])
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                audit = data.audit_external(task, rows)
                self.assertTrue(audit["passed"])
                self.assertEqual(audit["documents"], data.COUNTS[task])
                self.assertEqual(audit["candidates"], 4000 if task == "sciq" else 3676)
                for record in protocol["raw_files"] + protocol["sources"]:
                    self.assertEqual(data.sha256_file(data.ROOT / record["path"]), record["sha256"])
                raw, labels = data._load_raw(task)
                config = yaml.safe_load((data.ROOT / f"sources/lm_eval/lm_eval/tasks/{task}/{task}.yaml").read_text())
                template = environment.from_string(config["doc_to_text"])
                choice_template = environment.from_string(config["doc_to_choice"])
                for index, (source, row) in enumerate(zip(raw, rows)):
                    self.assertEqual(template.render(**source), row["context"])
                    self.assertEqual(ast.literal_eval(choice_template.render(**source)), row["choices"])
                    self.assertEqual(row["gold"], 3 if task == "sciq" else labels[index])
                if task == "sciq":
                    self.assertEqual(audit["support_empty_count"], 116)
                else:
                    self.assertTrue(audit["piqa_author_site_matches_archive"])

    def test_audit_rejects_missing_reordered_relabelled_or_modified_rows(self):
        for task in data.COUNTS:
            rows = [json.loads(line) for line in (data.ROOT / f"data/tasks/{task}/0shot.jsonl").read_text().splitlines()]
            modifications = [
                lambda r: r.pop(),
                lambda r: r.__setitem__(slice(0, 2), [r[1], r[0]]),
                lambda r: r[0].__setitem__("gold", (r[0]["gold"] + 1) % len(r[0]["choices"])),
                lambda r: r[0].__setitem__("context", "Question: wrong\nAnswer:"),
                lambda r: r[0].__setitem__("choices", r[0]["choices"][::-1]),
                lambda r: r[1].__setitem__("id", r[0]["id"]),
            ]
            for index, mutate in enumerate(modifications):
                with self.subTest(task=task, mutation=index):
                    changed = copy.deepcopy(rows)
                    mutate(changed)
                    with self.assertRaises(ValueError):
                        data.audit_external(task, changed)

    def test_offline_rebuild_preserves_output_and_protocol(self):
        for task in data.COUNTS:
            directory = data.ROOT / f"data/tasks/{task}"
            before = {name: (directory / name).read_bytes() for name in ("0shot.jsonl", "protocol.json")}
            with patch.object(data.urllib.request, "urlopen", side_effect=AssertionError("Unexpected network")):
                protocol = data.prepare_external(task)
            self.assertEqual(protocol["output"]["rows"], data.COUNTS[task])
            for name, content in before.items():
                self.assertEqual((directory / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
