import csv
import io
import json
import tempfile
from pathlib import Path

from pydantic import ValidationError

from rag_eval_toolkit.benchmarks.compare import compare_reports
from rag_eval_toolkit.benchmarks.runner import run_benchmark
from rag_eval_toolkit.core.errors import ConfigurationError, ReportError
from rag_eval_toolkit.core.models import BenchmarkConfig, EvaluationConfig, EvaluationReport
from rag_eval_toolkit.evaluators.engine import Evaluator
from rag_eval_toolkit.reporting.exporters import ExporterRegistry, export_report
from rag_eval_toolkit.reporting.io import write_output
from tests.support import FIXED_TIME, ROOT, OfflineTestCase, dataset, sample


def report(item=None, **configuration):
    config = {"metrics": ("token_f1", "context_precision"), **configuration}
    return Evaluator(EvaluationConfig(**config)).evaluate(dataset(item or sample()), timestamp=FIXED_TIME)


class ReportTests(OfflineTestCase):
    def test_json_round_trip_and_byte_reproducibility(self):
        original = report()
        content = export_report(original, "json")
        self.assertEqual(EvaluationReport.model_validate_json(content), original)
        self.assertEqual(content, export_report(report(), "json"))

    def test_csv_contains_configuration_scores_statistics_and_warnings(self):
        content = export_report(report(sample(reference_answer=None)), "csv")
        rows = list(csv.DictReader(io.StringIO(content)))
        self.assertEqual({row["record_type"] for row in rows}, {"metadata", "aggregate", "sample", "warning"})
        metadata = json.loads(next(row["details_json"] for row in rows if row["record_type"] == "metadata"))
        self.assertIn("configuration", metadata)
        skipped = next(row for row in rows if row["record_type"] == "sample" and row["metric"] == "token_f1")
        self.assertEqual((skipped["status"], skipped["score"]), ("skipped", ""))

    def test_csv_formula_injection_is_escaped(self):
        rows = list(csv.DictReader(io.StringIO(export_report(report(sample(id="=1+1")), "csv"))))
        self.assertEqual(next(row["sample_id"] for row in rows if row["record_type"] == "sample"), "'=1+1")

    def test_markdown_escapes_untrusted_identifiers(self):
        content = export_report(report(sample(id="<script>x</script>|[x](https://test)")), "markdown")
        self.assertNotIn("<script>", content)
        self.assertNotIn("[x](https://test)", content)
        self.assertIn("\\|", content)
        self.assertIn("Evaluation configuration", content)

    def test_skipped_markdown_shows_na(self):
        content = export_report(report(sample(reference_answer=None)), "markdown")
        self.assertIn("N/A", content)
        self.assertIn("No reference answer", content)

    def test_raw_answers_and_metadata_not_in_default_reports(self):
        result = report(sample(answer="private-answer-value", metadata={"api_key": "private-metadata-value"}))
        for format in ("json", "csv", "markdown"):
            self.assertNotIn("private-answer-value", export_report(result, format))
            self.assertNotIn("private-metadata-value", export_report(result, format))

    def test_custom_exporter_and_duplicate_rejection(self):
        registry = ExporterRegistry()
        registry.register("brief", lambda result: result.dataset.name)
        self.assertEqual(registry.export(report(), "brief"), "test")
        with self.assertRaises(ConfigurationError):
            registry.register("json", lambda result: "")
        with self.assertRaises(ConfigurationError):
            registry.export(report(), "missing")

    def test_output_protects_inputs_and_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            write_output(output, "first")
            with self.assertRaises(ReportError):
                write_output(output, "second")
            with self.assertRaises(ReportError):
                write_output(output, "second", overwrite=True, protected=(output,))
            self.assertEqual(output.read_text(), "first")
            write_output(output, "second", overwrite=True)
            self.assertEqual(output.read_text(), "second")

    def test_report_consistency_validation(self):
        raw = report().model_dump(mode="json")
        raw["dataset"]["sample_count"] = 2
        with self.assertRaises(ValidationError):
            EvaluationReport.model_validate(raw)
        raw = report().model_dump(mode="json")
        raw["aggregates"]["token_f1"]["count"] = 2
        with self.assertRaises(ValidationError):
            EvaluationReport.model_validate(raw)


class BenchmarkTests(OfflineTestCase):
    def test_regression_improvement_and_threshold(self):
        perfect = report()
        poor = report(sample(answer="totally unrelated"))
        regression = compare_reports(perfect, poor)
        self.assertTrue(regression.has_regression)
        self.assertEqual(regression.metrics[0].status, "regressed")
        self.assertFalse(compare_reports(poor, perfect).has_regression)
        self.assertEqual(compare_reports(perfect, perfect).metrics[0].status, "unchanged")
        self.assertFalse(compare_reports(perfect, poor, threshold=1.0).has_regression)

    def test_question_reference_and_id_mismatches_rejected(self):
        for changes in ({"id": "other"}, {"question": "Different question?"}, {"reference_answer": "different"}, {"reference_context_ids": ["other"]}):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                compare_reports(report(), report(sample(**changes)))

    def test_cutoff_mismatch_rejected(self):
        with self.assertRaises(ConfigurationError):
            compare_reports(report(k=1), report(k=2))

    def test_metric_mismatch_rejected(self):
        with self.assertRaises(ConfigurationError):
            compare_reports(report(), report(metrics=("exact_match",)))

    def test_nonfinite_and_out_of_range_thresholds_rejected(self):
        for threshold in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(threshold=threshold), self.assertRaises(ConfigurationError):
                compare_reports(report(), report(), threshold=threshold)

    def test_coverage_regression_cannot_hide_failures(self):
        raw = report().model_dump(mode="json")
        raw["results"][0]["metrics"][0].update(status="error", score=None, explanation="failed")
        raw["aggregates"]["token_f1"] = {"count": 0, "errors": 1}
        candidate = EvaluationReport.model_validate(raw)
        result = compare_reports(report(), candidate)
        self.assertTrue(result.has_regression)
        self.assertTrue(result.metrics[0].coverage_regression)
        self.assertIsNone(result.metrics[0].delta)

    def test_both_skipped_not_applicable(self):
        skipped = report(sample(reference_answer=None))
        result = compare_reports(skipped, skipped)
        self.assertEqual(result.metrics[0].status, "not_applicable")
        self.assertEqual(result.metrics[0].excluded_count, 1)

    def test_multidataset_manifest_and_micro_aggregates(self):
        config = BenchmarkConfig.model_validate_json((ROOT / "examples/benchmark.json").read_text())
        result = run_benchmark(config, base_directory=ROOT / "examples", timestamp=FIXED_TIME)
        self.assertEqual(len(result.runs), 4)
        self.assertEqual(len(result.comparisons), 2)
        self.assertEqual(result.aggregates["candidate"]["token_f1"].count, 5)
        self.assertEqual(result.aggregates["candidate"]["token_f1"].skipped, 1)
        self.assertFalse(result.has_regression)

    def test_manifest_duplicate_runs(self):
        data = json.loads((ROOT / "examples/benchmark.json").read_text())
        data["runs"].append(data["runs"][0])
        with self.assertRaises(ValidationError):
            BenchmarkConfig.model_validate(data)

    def test_partial_configuration_grid_is_rejected(self):
        data = json.loads((ROOT / "examples/benchmark.json").read_text())
        data["runs"][3]["configuration"] = "different-candidate"
        with self.assertRaises(ValidationError):
            BenchmarkConfig.model_validate(data)

    def test_lower_is_better_direction(self):
        from rag_eval_toolkit.judges.base import JudgeVerdict

        class RiskJudge:
            def evaluate(self, item, metric):
                return JudgeVerdict(status="ok", score=float(item.answer), explanation="Test fixture.")

            def provenance(self):
                return {"provider": "test-fixture"}

        evaluator = Evaluator(
            EvaluationConfig(metrics=("hallucination_risk",), enable_llm=True), judge=RiskJudge(),
        )
        high = evaluator.evaluate(dataset(sample(answer="0.8")))
        low = evaluator.evaluate(dataset(sample(answer="0.2")))
        self.assertFalse(compare_reports(high, low).has_regression)
        self.assertTrue(compare_reports(low, high).has_regression)

    def test_pairing_ignores_sample_order(self):
        evaluator = Evaluator(EvaluationConfig(metrics=("token_f1",)))
        first, second = sample(id="one"), sample(id="two")
        baseline = evaluator.evaluate(dataset(first, second))
        candidate = evaluator.evaluate(dataset(second, first))
        result = compare_reports(baseline, candidate)
        self.assertEqual(result.metrics[0].paired_count, 2)
        self.assertFalse(result.has_regression)

    def test_missing_baseline(self):
        data = json.loads((ROOT / "examples/benchmark.json").read_text())
        data["baseline"] = "absent"
        with self.assertRaises(ValidationError):
            BenchmarkConfig.model_validate(data)
