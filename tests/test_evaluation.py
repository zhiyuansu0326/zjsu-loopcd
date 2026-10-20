"""Input boundary cases that would otherwise silently mis-score new tasks."""
import copy
import unittest

from src.evaluation import fit_context, validate_documents
from src.loopcd import Candidate
from src.registry import load_registry, resolve_run


class EvaluationTests(unittest.TestCase):
    def test_left_truncation_preserves_complete_answer_and_alignment(self):
        # Full tokens: context [10,11,12,13,14], answer [20,21,22].
        source = Candidate(3, 1, [10, 11, 12, 13, 14, 20, 21], [4, 5, 6], [20, 21, 22])
        fitted, drop = fit_context(source, 4)
        self.assertEqual(drop, 3)
        self.assertEqual(fitted.input_ids, [13, 14, 20, 21])
        self.assertEqual(fitted.positions, [1, 2, 3])
        self.assertEqual(fitted.targets, [20, 21, 22])
        self.assertEqual(source.positions, [4, 5, 6])

    def test_answer_that_cannot_fit_fails_instead_of_silently_changing_metric(self):
        candidate = Candidate(0, 0, [10, 20, 21], [0, 1, 2], [20, 21, 22])
        with self.assertRaisesRegex(ValueError, "answer alone"):
            fit_context(candidate, 2)

    def test_untruncated_candidate_is_identical(self):
        candidate = Candidate(0, 0, [10, 20], [0, 1], [20, 21])
        fitted, drop = fit_context(candidate, 2)
        self.assertEqual(candidate, fitted)
        self.assertEqual(drop, 0)

    def test_invalid_documents_fail_before_scoring(self):
        doc = dict(id="x", choices=["a", "b"], labels=["A", "B"], gold=0, context="Question:")
        validate_documents([doc], 1)
        for change in [dict(choices=["", "b"]), dict(gold=2), dict(gold=True),
                       dict(labels=["A", "A"]), dict(context="")]:
            invalid = copy.deepcopy(doc)
            invalid.update(change)
            with self.assertRaises(ValueError):
                validate_documents([invalid], 1)
        with self.assertRaises(ValueError):
            validate_documents([doc, doc], 2)

    def test_every_task_model_combination_resolves(self):
        models, tasks = load_registry()
        for task in tasks["tasks"]:
            for model in models["models"]:
                run = resolve_run(task, model)
                self.assertEqual(run["seed"], 1234)
                self.assertIn(run["shots"], (0, 8, 25))
                self.assertIn(run["task"]["primary_metric"], ("acc", "acc_norm"))
        self.assertEqual(resolve_run("arc_challenge", "ouro_14")["shots"], 25)
        self.assertEqual(resolve_run("arc_challenge", "huginn_32")["shots"], 0)
        self.assertEqual(resolve_run("piqa", "ouro_14")["task"]["split"], "validation")


if __name__ == "__main__":
    unittest.main()
