import json
import tempfile
from pathlib import Path

from pydantic import ValidationError

from rag_eval_toolkit.core.errors import DatasetError
from rag_eval_toolkit.core.models import (
    AggregateStatistics, EvaluationConfig, EvaluationDataset, EvaluationSample, MetricResult,
)
from rag_eval_toolkit.datasets.jsonl import load_dataset
from tests.support import OfflineTestCase, sample


class ModelsTests(OfflineTestCase):
    def test_valid_sample_and_empty_answer(self):
        self.assertEqual(sample(answer="").answer, "")

    def test_missing_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            EvaluationSample.model_validate({"id": "x"})

    def test_wrong_types_and_unknown_fields(self):
        for changes in ({"question": 2}, {"contexts": "bad"}, {"answer": None}, {"extra": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                sample(**changes)

    def test_blank_questions_contexts_ids(self):
        for changes in ({"id": " "}, {"question": "\t"}, {"contexts": [{"id": "d", "text": " "}]}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                sample(**changes)

    def test_duplicate_context_ids(self):
        with self.assertRaises(ValidationError):
            sample(contexts=[{"id": "x", "text": "a"}, {"id": "x", "text": "b"}])

    def test_duplicate_reference_ids(self):
        with self.assertRaises(ValidationError):
            sample(reference_context_ids=["x", "x"])

    def test_duplicate_sample_ids(self):
        with self.assertRaises(ValidationError):
            EvaluationDataset(name="x", samples=(sample(), sample()))

    def test_empty_dataset(self):
        with self.assertRaises(ValidationError):
            EvaluationDataset(name="x", samples=())

    def test_metric_score_bounds_and_types(self):
        for score in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(score=score), self.assertRaises(ValidationError):
                MetricResult(metric="x", score=score)

    def test_result_status_invariants(self):
        for data in ({"status": "ok"}, {"status": "skipped", "score": 0}, {"status": "error"}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                MetricResult(metric="x", **data)

    def test_skipped_score_is_null(self):
        self.assertIsNone(MetricResult(metric="x", status="skipped", explanation="No reference").score)

    def test_k_and_duplicate_metrics(self):
        for k in (0, -1, True, 1.2):
            with self.subTest(k=k), self.assertRaises(ValidationError):
                EvaluationConfig(metrics=("token_f1",), k=k)
        with self.assertRaises(ValidationError):
            EvaluationConfig(metrics=("token_f1", "token_f1"))

    def test_aggregate_invariants(self):
        with self.assertRaises(ValidationError):
            AggregateStatistics(count=0, mean=0.0)
        with self.assertRaises(ValidationError):
            AggregateStatistics(count=1)


class DatasetTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data.jsonl"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")
        return self.path

    def test_load_utf8_bom_and_blank_lines(self):
        content = "\ufeff" + sample(answer="Azərbaycan").model_dump_json() + "\n\n"
        loaded = load_dataset(self.write(content))
        self.assertEqual(loaded.samples[0].answer, "Azərbaycan")
        self.assertEqual(len(loaded.warnings), 1)

    def test_malformed_json_reports_line_without_content(self):
        with self.assertRaises(DatasetError) as error:
            load_dataset(self.write(sample().model_dump_json() + '\n{"secret-test-value"'))
        self.assertIn("Line 2", str(error.exception))
        self.assertNotIn("secret-test-value", str(error.exception))

    def test_invalid_field_path(self):
        data = sample().model_dump(mode="json")
        data["contexts"][0]["text"] = None
        with self.assertRaisesRegex(DatasetError, "contexts.0.text"):
            load_dataset(self.write(json.dumps(data)))

    def test_duplicate_sample_id_reports_line(self):
        with self.assertRaisesRegex(DatasetError, "Line 2: duplicate sample ID"):
            load_dataset(self.write((sample().model_dump_json() + "\n") * 2))

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(DatasetError):
            load_dataset(self.write('{"id":"one","id":"two"}'))

    def test_nonfinite_json_rejected(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), self.assertRaises(DatasetError):
                load_dataset(self.write('{"metadata":{"number":' + constant + '}}'))

    def test_empty_and_missing_files(self):
        with self.assertRaises(DatasetError):
            load_dataset(self.path)
        with self.assertRaises(DatasetError):
            load_dataset(self.write("\n  \n"))

    def test_bounded_line_and_invalid_encoding(self):
        with self.assertRaisesRegex(DatasetError, "exceeds"):
            load_dataset(self.write("a" * 100), max_line_bytes=20)
        self.path.write_bytes(b"\xff")
        with self.assertRaises(DatasetError):
            load_dataset(self.path)

    def test_nonobject_and_no_code_execution(self):
        for content in ("[]", "null", '__import__("os").system("echo forbidden")'):
            with self.subTest(content=content), self.assertRaises(DatasetError):
                load_dataset(self.write(content))
