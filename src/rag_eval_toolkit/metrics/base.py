"""Minimal contracts for user-defined metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_eval_toolkit.core.models import EvaluationSample, MetricDefinition, MetricResult
from rag_eval_toolkit.judges.base import Judge


@dataclass(frozen=True)
class MetricContext:
    k: int | None = None
    judge: Judge | None = None


class Metric(Protocol):
    @property
    def definition(self) -> MetricDefinition: ...

    def evaluate(self, sample: EvaluationSample, context: MetricContext) -> MetricResult: ...


def skipped(name: str, reason: str) -> MetricResult:
    return MetricResult(metric=name, status="skipped", explanation=reason)
