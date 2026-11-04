"""JSON, long-form CSV, and Markdown without raw questions/answers by default."""

from __future__ import annotations

import csv
import html
import io
import json
import re
from collections.abc import Callable

from rag_eval_toolkit.core.errors import ConfigurationError
from rag_eval_toolkit.core.models import EvaluationReport


def json_report(report: EvaluationReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"), ensure_ascii=False,
        indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"


def _csv_safe(value: object) -> object:
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


def csv_report(report: EvaluationReport) -> str:
    output = io.StringIO(newline="")
    fields = [
        "record_type", "dataset", "timestamp", "sample_id", "metric", "status", "score",
        "count", "skipped", "errors", "mean", "median", "stddev", "minimum", "maximum", "details_json",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    def row(record_type: str, **values: object) -> None:
        data = {"record_type": record_type, "dataset": report.dataset.name,
                "timestamp": report.created_at.isoformat(), **values}
        writer.writerow({key: _csv_safe(value) for key, value in data.items()})

    metadata = report.model_dump(mode="json", exclude={"results", "aggregates", "warnings"})
    row("metadata", details_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    for warning in report.warnings:
        row("warning", details_json=json.dumps({"warning": warning}, ensure_ascii=False))
    for name, stats in report.aggregates.items():
        row("aggregate", metric=name, **stats.model_dump())
    for result in report.results:
        for metric in result.metrics:
            row(
                "sample", sample_id=result.sample_id, metric=metric.metric,
                status=metric.status, score=metric.score,
                details_json=json.dumps(
                    metric.model_dump(mode="json", exclude={"metric", "status", "score"}),
                    ensure_ascii=False, sort_keys=True,
                ),
            )
    return output.getvalue()


def _md(value: object) -> str:
    text = html.escape(str(value), quote=False).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{\[\]}()#+.!|>~-])", r"\\\1", text)


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def markdown_report(report: EvaluationReport) -> str:
    lines = [
        "# RAG evaluation report", "",
        f"Dataset: **{_md(report.dataset.name)}** ({report.dataset.sample_count} samples)",
        f"Timestamp: {_md(report.created_at.isoformat())}",
        f"Dataset SHA-256: `{report.dataset.sha256}`", "",
        "## Aggregate scores", "",
        "| Metric | OK | Skipped | Errors | Mean | Median | Stddev | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in report.aggregates.items():
        values = (stats.mean, stats.median, stats.stddev, stats.minimum, stats.maximum)
        lines.append(f"| {_md(name)} | {stats.count} | {stats.skipped} | {stats.errors} | "
                     + " | ".join(_number(value) for value in values) + " |")
    lines.extend(["", "## Individual scores", "",
                  "| Sample | Metric | Status | Score | Explanation / evidence |",
                  "| --- | --- | --- | ---: | --- |"])
    for result in report.results:
        for metric in result.metrics:
            detail = metric.explanation
            if metric.evidence or metric.unsupported_claims:
                detail += " " + json.dumps({
                    "evidence": [item.model_dump() for item in metric.evidence],
                    "unsupported_claims": metric.unsupported_claims,
                }, ensure_ascii=False)
            lines.append(f"| {_md(result.sample_id)} | {_md(metric.metric)} | {metric.status} | "
                         f"{_number(metric.score)} | {_md(detail)} |")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {_md(warning)}" for warning in report.warnings)
    config = json.dumps({
        "evaluation": report.configuration.model_dump(mode="json"),
        "metrics": [definition.model_dump() for definition in report.metric_definitions],
        "judge": report.judge, "toolkit_version": report.toolkit_version,
    }, ensure_ascii=False, indent=2, sort_keys=True)
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", config)), default=0))
    lines.extend(["", "## Evaluation configuration", "", fence + "json", config, fence, ""])
    return "\n".join(lines)


class ExporterRegistry:
    def __init__(self) -> None:
        self._exporters: dict[str, Callable[[EvaluationReport], str]] = {
            "json": json_report, "csv": csv_report, "markdown": markdown_report,
        }

    def register(self, name: str, exporter: Callable[[EvaluationReport], str]) -> None:
        if name in self._exporters:
            raise ConfigurationError(f"Exporter already registered: {name}")
        self._exporters[name] = exporter

    def export(self, report: EvaluationReport, format: str) -> str:
        try:
            exporter = self._exporters[format]
        except KeyError:
            raise ConfigurationError(f"Unknown report format: {format}") from None
        return exporter(report)


def export_report(report: EvaluationReport, format: str = "json") -> str:
    return ExporterRegistry().export(report, format)
