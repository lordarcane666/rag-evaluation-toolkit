from datetime import datetime

from rag_eval_toolkit.core.errors import ConfigurationError, EvaluationError
from rag_eval_toolkit.core.models import EvaluationConfig, MetricDefinition, MetricResult
from rag_eval_toolkit.evaluators.engine import Evaluator
from rag_eval_toolkit.evaluators.statistics import aggregate
from rag_eval_toolkit.metrics.base import MetricContext
from rag_eval_toolkit.metrics.registry import default_registry
from rag_eval_toolkit.metrics.text import tokenize
from tests.support import FIXED_TIME, OfflineTestCase, dataset, sample


def score(name, item=None, k=None):
    return default_registry().get(name).evaluate(item or sample(), MetricContext(k=k))


class MetricTests(OfflineTestCase):
    def test_unicode_normalization(self):
        self.assertEqual(tokenize("CAFÉ café\u0301 ＲＡＧ Straße"), ["café", "café", "rag", "strasse"])

    def test_normalized_exact_match(self):
        self.assertEqual(score("exact_match", sample(answer=" RAG, combines RETRIEVAL and generation! ")).score, 1)

    def test_order_matters_for_exact_match(self):
        item = sample(answer="blue red", reference_answer="red blue")
        self.assertEqual(score("exact_match", item).score, 0)
        self.assertEqual(score("token_f1", item).score, 1)
        self.assertEqual(score("answer_reference_similarity", item).score, 0.5)

    def test_multiset_duplicates_are_penalized(self):
        item = sample(answer="a a b", reference_answer="a b")
        self.assertAlmostEqual(score("token_precision", item).score, 2 / 3)
        self.assertEqual(score("token_recall", item).score, 1)
        self.assertAlmostEqual(score("token_f1", item).score, 0.8)

    def test_empty_prediction_is_zero_with_valid_reference(self):
        for name in ("exact_match", "token_precision", "token_recall", "token_f1", "answer_reference_similarity"):
            with self.subTest(name=name):
                self.assertEqual(score(name, sample(answer="")).score, 0)

    def test_missing_or_empty_reference_skips(self):
        for reference in (None, "", "...", " "):
            for name in ("exact_match", "token_precision", "token_recall", "token_f1", "answer_reference_similarity"):
                with self.subTest(reference=reference, name=name):
                    result = score(name, sample(reference_answer=reference))
                    self.assertEqual(result.status, "skipped")
                    self.assertIsNone(result.score)

    def test_lexical_relevance_is_unique_token_coverage(self):
        item = sample(question="red red blue", answer="red")
        self.assertEqual(score("lexical_answer_relevance", item).score, 0.5)
        self.assertEqual(score("lexical_answer_relevance", sample(question="???")).status, "skipped")

    def test_context_at_k_and_missing_rank_slots(self):
        item = sample(contexts=[{"id": "d1", "text": "one"}, {"id": "d2", "text": "two"}], reference_context_ids=["d1", "d3"])
        self.assertEqual(score("context_precision", item, k=1).score, 1)
        self.assertEqual(score("context_recall", item, k=1).score, 0.5)
        self.assertAlmostEqual(score("context_precision", item, k=3).score, 1 / 3)
        self.assertAlmostEqual(score("context_f1", item, k=3).score, 0.4)
        self.assertEqual(score("context_precision", item).score, 0.5)

    def test_empty_retrieval_with_positive_labels_is_zero(self):
        for name in ("context_precision", "context_recall", "context_f1"):
            self.assertEqual(score(name, sample(contexts=[])).score, 0)

    def test_absent_or_empty_context_labels_skip(self):
        for refs in (None, []):
            for name in ("context_precision", "context_recall", "context_f1"):
                self.assertEqual(score(name, sample(reference_context_ids=refs)).status, "skipped")

    def test_unknown_and_duplicate_registration(self):
        registry = default_registry()
        with self.assertRaises(ConfigurationError):
            registry.get("not_real")
        with self.assertRaises(ConfigurationError):
            registry.register(registry.get("token_f1"))


class EngineTests(OfflineTestCase):
    def test_reproducible_report(self):
        first = Evaluator().evaluate(dataset(), timestamp=FIXED_TIME)
        second = Evaluator().evaluate(dataset(), timestamp=FIXED_TIME)
        self.assertEqual(first.model_dump_json(), second.model_dump_json())

    def test_naive_timestamp_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            Evaluator().evaluate(dataset(), timestamp=datetime(2026, 1, 1))

    def test_llm_requires_opt_in_and_judge(self):
        with self.assertRaises(ConfigurationError):
            Evaluator(EvaluationConfig(metrics=("faithfulness",)))
        with self.assertRaises(ConfigurationError):
            Evaluator(EvaluationConfig(metrics=("faithfulness",), enable_llm=True))

    def test_statistics_exclude_missing_and_use_population_stddev(self):
        results = [MetricResult(metric="x", score=value) for value in (0.0, 1.0)]
        results += [MetricResult(metric="x", status="skipped", explanation="none"), MetricResult(metric="x", status="error", explanation="failed")]
        stats = aggregate(results)
        self.assertEqual((stats.count, stats.skipped, stats.errors), (2, 1, 1))
        self.assertEqual((stats.mean, stats.median, stats.stddev, stats.minimum, stats.maximum), (0.5, 0.5, 0.5, 0, 1))
        self.assertIsNone(aggregate([]).mean)
        self.assertEqual(aggregate([results[0]]).stddev, 0)

    def test_custom_metric_and_safe_failures(self):
        class Broken:
            definition = MetricDefinition(name="broken", kind="deterministic", description="test")

            def evaluate(self, sample, context):
                raise RuntimeError("secret-must-not-appear")

        registry = default_registry()
        registry.register(Broken())
        with self.assertLogs("rag_eval_toolkit.evaluators.engine", level="WARNING") as logs:
            report = Evaluator(EvaluationConfig(metrics=("broken",)), registry=registry).evaluate(dataset())
        self.assertEqual(report.results[0].metrics[0].status, "error")
        self.assertEqual(report.aggregates["broken"].errors, 1)
        self.assertNotIn("secret-must-not-appear", report.model_dump_json() + str(logs.output))
        with self.assertRaises(EvaluationError):
            Evaluator(EvaluationConfig(metrics=("broken",), fail_fast=True), registry=registry).evaluate(dataset())

    def test_data_fingerprint_changes_with_answer_but_targets_do_not(self):
        first = Evaluator().evaluate(dataset(sample(answer="one")))
        second = Evaluator().evaluate(dataset(sample(answer="two")))
        self.assertNotEqual(first.dataset.sha256, second.dataset.sha256)
        self.assertEqual(first.results[0].comparison_fingerprint, second.results[0].comparison_fingerprint)

    def test_explicit_custom_error_obeys_fail_fast(self):
        class Failed:
            definition = MetricDefinition(name="failed", kind="deterministic", description="test")

            def evaluate(self, sample, context):
                return MetricResult(metric="failed", status="error", explanation="Unavailable resource.")

        registry = default_registry()
        registry.register(Failed())
        with self.assertRaises(EvaluationError):
            Evaluator(
                EvaluationConfig(metrics=("failed",), fail_fast=True), registry=registry,
            ).evaluate(dataset())
