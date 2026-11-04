import contextlib
import io
import json
import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch

from rag_eval_toolkit.cli.app import main
from rag_eval_toolkit.core.models import EvaluationReport
from tests.support import ROOT, OfflineTestCase, sample


class CLITests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "output.json"
        self.demo = str(ROOT / "examples/demo.jsonl")

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = main(list(arguments))
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_help_version_and_metric_listing(self):
        for command in (("--help",), ("evaluate", "--help"), ("version",), ("metrics",)):
            code, output, _ = self.invoke(*command)
            self.assertEqual(code, 0)
            self.assertTrue(output)
        self.assertIn("0.1.0", self.invoke("version")[1])

    def test_validate_example(self):
        code, output, _ = self.invoke("validate", self.demo)
        self.assertEqual(code, 0)
        self.assertIn("4 samples", output)

    def test_required_evaluation_command(self):
        code, output, _ = self.invoke("evaluate", self.demo, "--metrics", "exact_match,token_f1,context_precision,context_recall")
        self.assertEqual(code, 0)
        self.assertIn("RAG evaluation report", output)

    def test_json_stdout_is_parseable_and_contains_no_progress_text(self):
        code, output, _ = self.invoke("evaluate", self.demo, "--format", "json")
        self.assertEqual(code, 0)
        report = EvaluationReport.model_validate_json(output)
        self.assertEqual(report.dataset.sample_count, 4)

    def test_unicode_output_is_utf8_even_with_ascii_stream(self):
        self.output.write_text(sample(id="şəki").model_dump_json(), encoding="utf-8")
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="ascii")
        with patch("sys.stdout", stream):
            code = main(["evaluate", str(self.output), "--format", "json"])
        stream.flush()
        self.assertEqual(code, 0)
        self.assertIn("şəki", buffer.getvalue().decode("utf-8"))
        stream.detach()

    def test_all_export_formats_to_files(self):
        for format in ("json", "csv", "markdown"):
            output = self.output.with_suffix("." + format)
            code, stdout, _ = self.invoke("evaluate", self.demo, "--format", format, "--output", str(output))
            self.assertEqual(code, 0)
            self.assertEqual(stdout, "")
            self.assertTrue(output.read_text(encoding="utf-8"))

    def test_output_and_input_protection(self):
        self.output.write_text("keep", encoding="utf-8")
        self.assertEqual(self.invoke("evaluate", self.demo, "--output", str(self.output))[0], 2)
        self.assertEqual(self.output.read_text(), "keep")
        self.assertEqual(self.invoke("evaluate", self.demo, "--output", self.demo, "--overwrite")[0], 2)

    def test_invalid_metric_cutoff_and_timestamp(self):
        for options in (("--metrics", "nope"), ("--k", "0"), ("--timestamp", "2026-01-01"), ("--format", "xml"), ("--metrics", "token_f1,")):
            with self.subTest(options=options):
                self.assertEqual(self.invoke("evaluate", self.demo, *options)[0], 2)

    def test_invalid_file(self):
        self.output.write_text("not-json", encoding="utf-8")
        code, _, stderr = self.invoke("validate", str(self.output))
        self.assertEqual(code, 2)
        self.assertIn("Line 1", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_llm_never_enabled_implicitly(self):
        with patch("rag_eval_toolkit.providers.openai_compatible.UrllibTransport.send") as send:
            code, _, _ = self.invoke("evaluate", self.demo, "--metrics", "faithfulness")
            self.assertEqual(code, 2)
            send.assert_not_called()

    def test_llm_requires_endpoint_and_model(self):
        self.assertEqual(self.invoke("evaluate", self.demo, "--metrics", "faithfulness", "--enable-llm")[0], 2)

    def test_mocked_llm_failure_sets_exit_one_without_secret_leak(self):
        with patch("rag_eval_toolkit.providers.openai_compatible.UrllibTransport.send", return_value=b"malformed-test-secret"):
            code, output, stderr = self.invoke("evaluate", self.demo, "--metrics", "faithfulness", "--enable-llm", "--judge-base-url", "http://localhost:8000/v1", "--judge-model", "test", "--format", "json")
        self.assertEqual(code, 1)
        self.assertNotIn("malformed-test-secret", output + stderr)

    def test_fixed_timestamp_is_reproducible(self):
        arguments = ("evaluate", self.demo, "--format", "json", "--timestamp", "2026-01-01T00:00:00Z")
        self.assertEqual(self.invoke(*arguments)[1], self.invoke(*arguments)[1])

    def test_fail_fast_metric_error_exit_one(self):
        with patch("rag_eval_toolkit.providers.openai_compatible.UrllibTransport.send", return_value=b"bad"):
            code, output, stderr = self.invoke(
                "evaluate", self.demo, "--metrics", "faithfulness", "--enable-llm",
                "--judge-base-url", "http://localhost:8000/v1", "--judge-model", "test", "--fail-fast",
            )
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("Metric failed", stderr)

    def test_benchmark_cli(self):
        code, output, _ = self.invoke("benchmark", str(ROOT / "examples/benchmark.json"))
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(output)["runs"]), 4)

    def test_compare_cli_and_regression_exit_code(self):
        baseline = self.output.with_name("baseline.json")
        candidate = self.output.with_name("candidate.json")
        self.assertEqual(self.invoke("evaluate", self.demo, "--metrics", "token_f1", "--format", "json", "--output", str(baseline))[0], 0)
        self.assertEqual(self.invoke("evaluate", str(ROOT / "examples/candidate.jsonl"), "--metrics", "token_f1", "--format", "json", "--output", str(candidate))[0], 0)
        self.assertEqual(self.invoke("compare", str(baseline), str(candidate))[0], 0)
        self.assertEqual(self.invoke("compare", str(candidate), str(baseline))[0], 1)

    def test_documented_custom_metric_example(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(ROOT / "examples/custom_metric.py"), run_name="__main__")
        self.assertEqual(json.loads(output.getvalue())["mean"], 1)
