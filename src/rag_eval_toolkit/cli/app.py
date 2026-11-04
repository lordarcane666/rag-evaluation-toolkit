"""Dependency-light argparse CLI with machine-readable stdout and useful exit codes."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from rag_eval_toolkit import __version__
from rag_eval_toolkit.benchmarks.compare import compare_reports
from rag_eval_toolkit.benchmarks.runner import run_benchmark
from rag_eval_toolkit.core.errors import ConfigurationError, EvaluationError, RAGEvalError, ReportError
from rag_eval_toolkit.core.models import BenchmarkConfig, EvaluationConfig, EvaluationReport
from rag_eval_toolkit.datasets.jsonl import load_dataset
from rag_eval_toolkit.evaluators.engine import Evaluator
from rag_eval_toolkit.judges.base import Judge
from rag_eval_toolkit.metrics.registry import DEFAULT_METRICS, default_registry
from rag_eval_toolkit.reporting.exporters import export_report
from rag_eval_toolkit.reporting.io import write_output


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, help="Write to a local file instead of stdout")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")


def _add_judge(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-llm", action="store_true",
        help="Explicitly permit sending samples to the configured judge",
    )
    parser.add_argument(
        "--judge-base-url",
        help="Explicit HTTPS API root or loopback HTTP root, e.g. http://localhost:8000/v1",
    )
    parser.add_argument("--judge-model", help="Model identifier served by the configured endpoint")
    parser.add_argument(
        "--api-key-env", default="RAG_EVAL_API_KEY",
        help="Environment variable name containing the API key",
    )
    parser.add_argument("--judge-timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument("--judge-json-mode", action=argparse.BooleanOptionalAction, default=True,
                        help="Request JSON mode; disable only for servers without response_format support")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval", description="Evaluate RAG outputs locally; LLM judges require explicit opt-in.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="Print the toolkit version")
    commands.add_parser("metrics", help="List metrics, types, and score directions")
    validate = commands.add_parser("validate", help="Validate a UTF-8 JSONL dataset")
    validate.add_argument("dataset", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate a JSONL dataset")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated metric names")
    evaluate.add_argument("--k", type=_positive_int, help="Retrieval rank cutoff (default: all retrieved contexts)")
    evaluate.add_argument("--format", choices=("json", "csv", "markdown"), default="markdown")
    evaluate.add_argument("--timestamp", help="Fixed timezone-aware ISO timestamp for reproducible reports")
    evaluate.add_argument("--fail-fast", action="store_true", help="Stop on the first metric failure")
    _add_output(evaluate)
    _add_judge(evaluate)
    compare = commands.add_parser("compare", help="Compare two JSON evaluation reports on paired samples")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--threshold", type=float, default=0.0, help="Allowed absolute score decrease, 0..1")
    _add_output(compare)
    benchmark = commands.add_parser("benchmark", help="Evaluate a JSON benchmark manifest of precomputed outputs")
    benchmark.add_argument("configuration", type=Path)
    benchmark.add_argument("--timestamp", help="Fixed timezone-aware ISO timestamp")
    _add_output(benchmark)
    _add_judge(benchmark)
    return parser


def _timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError("timezone required")
        return result
    except ValueError:
        raise ConfigurationError("--timestamp must be ISO 8601 with a timezone, e.g. 2026-01-01T00:00:00Z") from None


def _judge(args: argparse.Namespace, configuration: EvaluationConfig) -> Judge | None:
    registry = default_registry()
    uses_llm = any(registry.get(name).definition.kind == "llm" for name in configuration.metrics)
    if uses_llm and not args.enable_llm:
        raise ConfigurationError("LLM metrics require --enable-llm; no data was sent")
    if not uses_llm:
        if args.enable_llm or args.judge_base_url or args.judge_model:
            raise ConfigurationError("Judge options require at least one LLM metric in --metrics/configuration")
        return None
    if not args.judge_base_url or not args.judge_model:
        raise ConfigurationError("LLM metrics require --judge-base-url and --judge-model")
    from rag_eval_toolkit.providers.openai_compatible import JudgeOptions, OpenAICompatibleJudge

    return OpenAICompatibleJudge(
        JudgeOptions(
            base_url=args.judge_base_url, model=args.judge_model, allow_network=True,
            timeout=args.judge_timeout, json_mode=args.judge_json_mode,
        ),
        api_key=os.environ.get(args.api_key_env),
    )


def _emit(content: str, args: argparse.Namespace, protected: tuple[Path, ...]) -> None:
    if args.output is not None:
        write_output(args.output, content, overwrite=args.overwrite, protected=protected)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(content)


def _model_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"), ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False,
    ) + "\n"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        raise ReportError("Cannot read UTF-8 JSON configuration/report file") from None


def _run(args: argparse.Namespace) -> int:
    if args.command == "version":
        print(f"rag-evaluation-toolkit {__version__}")
        return 0
    if args.command == "metrics":
        registry = default_registry()
        for name in registry.names():
            definition = registry.get(name).definition
            direction = "higher is better" if definition.higher_is_better else "lower is better"
            print(f"{name} [{definition.kind}; {direction}]: {definition.description}")
        return 0
    if args.command == "validate":
        dataset = load_dataset(args.dataset)
        print(f"Valid: {len(dataset.samples)} samples in {dataset.name}")
        for warning in dataset.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        return 0
    if args.command == "evaluate":
        names = tuple(name.strip() for name in args.metrics.split(","))
        configuration = EvaluationConfig(
            metrics=names, k=args.k, enable_llm=args.enable_llm, fail_fast=args.fail_fast,
        )
        judge = _judge(args, configuration)
        evaluator = Evaluator(configuration, judge=judge)
        timestamp = _timestamp(args.timestamp)
        dataset = load_dataset(args.dataset)
        report = evaluator.evaluate(dataset, timestamp=timestamp)
        _emit(export_report(report, args.format), args, (args.dataset,))
        return int(any(stats.errors for stats in report.aggregates.values()))
    if args.command == "compare":
        baseline = EvaluationReport.model_validate_json(_read(args.baseline))
        candidate = EvaluationReport.model_validate_json(_read(args.candidate))
        comparison = compare_reports(baseline, candidate, threshold=args.threshold)
        _emit(_model_json(comparison), args, (args.baseline, args.candidate))
        return int(comparison.has_regression)
    if args.command == "benchmark":
        configuration = BenchmarkConfig.model_validate_json(_read(args.configuration))
        evaluation = configuration.evaluation.model_copy(update={"enable_llm": args.enable_llm})
        judge = _judge(args, evaluation)
        configuration = configuration.model_copy(update={"evaluation": evaluation})
        result = run_benchmark(
            configuration, base_directory=args.configuration.parent,
            timestamp=_timestamp(args.timestamp), judge=judge,
        )
        protected = (args.configuration,) + tuple(args.configuration.parent / run.path for run in configuration.runs)
        _emit(_model_json(result), args, protected)
        has_errors = any(stats.errors for run in result.runs for stats in run.report.aggregates.values())
        return int(result.has_regression or has_errors)
    raise ConfigurationError("Unknown command")


def main(argv: list[str] | None = None) -> int:
    # Machine-readable output remains UTF-8 even in legacy Windows pipe locales.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) or "root" for error in exc.errors())
        print(f"Error: invalid configuration/report fields: {fields}", file=sys.stderr)
        return 2
    except EvaluationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RAGEvalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 1
