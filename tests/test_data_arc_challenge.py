"""Data integrity checks; no model loading, downloads, or GPU required."""

import copy
import hashlib
import json
import random
import unittest

from src import data_arc_challenge as data


def fixture():
    train = [
        {
            "id": f"train-{i}", "question": f"Question number {i}?",
            "choices": [f"right {i}", f"wrong {i}"], "labels": ["A", "B"], "gold": 0,
        }
        for i in range(40)
    ]
    test = [
        {
            "id": f"test-{i}", "question": f"Test {i}?",
            "choices": ["one", "two", "three"] + (["four", "five"] if i else []),
            "labels": ["1", "2", "3"] + (["4", "5"] if i else []), "gold": i,
        }
        for i in range(2)
    ]
    return train, test


class ARCChallengeUnitTests(unittest.TestCase):
    def test_advancing_sampling_and_complete_answer_format(self):
        train, test = fixture()
        rows = data.build_rows(train, test, 25, 1234)
        rng = random.Random(1234)
        for row, doc in zip(rows, test):
            examples = rng.sample(train, 25)
            self.assertEqual(row["fewshot_ids"], [item["id"] for item in examples])
            self.assertTrue(row["context"].endswith(f"Question: {doc['question']}\nAnswer:"))
            self.assertEqual(row["context"].count("\nAnswer:"), 26)
        self.assertNotEqual(rows[0]["fewshot_ids"], rows[1]["fewshot_ids"])
        self.assertEqual(data.audit_rows(rows, train, test, 25, 1234)["candidate_count"], 8)

    def test_zero_shot_contains_no_training_answers(self):
        train, test = fixture()
        rows = data.build_rows(train, test, 0, 1234)
        for row in rows:
            self.assertEqual(row["fewshot_ids"], [])
            self.assertEqual(row["context"], f"Question: {row['question']}\nAnswer:")
        self.assertTrue(data.audit_rows(rows, train, test, 0, 1234)["test_label_invariance"])

    def test_detects_raw_fields_prompt_selection_and_order_tampering(self):
        train, test = fixture()
        rows = data.build_rows(train, test, 25, 1234)
        mutations = (
            ("question", "changed question"), ("choices", ["changed", "choices"]),
            ("labels", ["A", "B", "C"]), ("gold", 1),
            ("context", rows[0]["context"].replace("right", "wrong", 1)),
            ("fewshot_ids", rows[0]["fewshot_ids"][::-1]), ("source_index", 1),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                changed = copy.deepcopy(rows)
                changed[0][key] = value
                with self.assertRaises(ValueError):
                    data.audit_rows(changed, train, test, 25, 1234)
        with self.assertRaises(ValueError):
            data.audit_rows(rows[::-1], train, test, 25, 1234)
        with self.assertRaises(ValueError):
            data.audit_rows(rows[:-1], train, test, 25, 1234)

    def test_rejects_overlap_and_invalid_shots(self):
        train, test = fixture()
        test[0]["id"] = train[0]["id"]
        rows = data.build_rows(train, test, 0, 1234)
        with self.assertRaises(ValueError):
            data.audit_rows(rows, train, test, 0, 1234)
        with self.assertRaises(ValueError):
            data.build_rows(train, test, 5, 1234)
        with self.assertRaises(ValueError):
            data.prepare_arc_challenge(True)


@unittest.skipUnless(
    (data.ROOT / "data/tasks/arc_challenge/25shot.jsonl").exists()
    and (data.ROOT / "data/tasks/arc_challenge/0shot.jsonl").exists(),
    "Prepare official ARC-Challenge data to run full local integrity audit.",
)
class ARCChallengeOfficialDataTests(unittest.TestCase):
    def test_every_prepared_row_and_hash_matches_official_data(self):
        protocol_path = data.ROOT / "data/tasks/arc_challenge/protocol.json"
        protocol = json.loads(protocol_path.read_text())
        self.assertEqual(protocol["counts"], {"train": 1119, "validation": 299, "test": 1172})
        for shots in (0, 25):
            checks = data.audit_arc_challenge(shots)
            self.assertEqual(checks["rows_checked"], 1172)
            self.assertEqual(checks["candidate_count"], 4687)
            path = data.ROOT / protocol["outputs"][str(shots)]["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                protocol["outputs"][str(shots)]["sha256"],
            )

    def test_all_three_raw_splits_are_pinned(self):
        for split, count in data.COUNTS.items():
            self.assertEqual(len(data.read_split(split)), count)


if __name__ == "__main__":
    unittest.main()
