"""Evaluate named pipeline outputs over multiple datasets; never invent pipeline measurements."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rag_eval_toolkit.benchmarks.compare import compare_reports
from rag_eval_toolkit.core.models import (
    BenchmarkComparison, BenchmarkConfig, BenchmarkResult, BenchmarkRunResult,
)
from rag_eval_toolkit.datasets.jsonl import load_dataset
from rag_eval_toolkit.evaluators.engine import Evaluator
from rag_eval_toolkit.evaluators.statistics import aggregate
from rag_eval_toolkit.judges.base import Judge
from rag_eval_toolkit.metrics.registry import MetricRegistry


def run_benchmark(
    configuration: BenchmarkConfig, *, base_directory: str | Path = ".",
    timestamp: datetime | None = None, registry: MetricRegistry | None = None,
    judge: Judge | None = None,
) -> BenchmarkResult:
    created_at = timestamp or datetime.now(UTC)
    evaluator = Evaluator(configuration.evaluation, registry=registry, judge=judge)
    runs = []
    for run in configuration.runs:
        dataset = load_dataset(Path(base_directory) / run.path, name=run.dataset)
        runs.append(BenchmarkRunResult(
            dataset=run.dataset, configuration=run.configuration,
            report=evaluator.evaluate(dataset, timestamp=created_at),
        ))
    baselines = {run.dataset: run.report for run in runs if run.configuration == configuration.baseline}
    comparisons = tuple(
        BenchmarkComparison(
            dataset=run.dataset, configuration=run.configuration,
            comparison=compare_reports(baselines[run.dataset], run.report,
                                       threshold=configuration.regression_threshold),
        )
        for run in runs if run.configuration != configuration.baseline
    )
    aggregates = {
        label: {
            name: aggregate(
                metric for run in runs if run.configuration == label
                for result in run.report.results for metric in result.metrics if metric.metric == name
            )
            for name in configuration.evaluation.metrics
        }
        for label in sorted({run.configuration for run in runs})
    }
    return BenchmarkResult(
        name=configuration.name, created_at=created_at, configuration=configuration,
        runs=tuple(runs), comparisons=comparisons, aggregates=aggregates,
        has_regression=any(item.comparison.has_regression for item in comparisons),
    )
