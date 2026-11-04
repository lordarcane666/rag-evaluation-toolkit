"""Macro sample statistics with explicit missing-value counts."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import fmean, median, pstdev

from rag_eval_toolkit.core.models import AggregateStatistics, MetricResult


def aggregate(results: Iterable[MetricResult]) -> AggregateStatistics:
    items = list(results)
    scores = [item.score for item in items if item.status == "ok" and item.score is not None]
    return AggregateStatistics(
        count=len(scores), skipped=sum(item.status == "skipped" for item in items),
        errors=sum(item.status == "error" for item in items),
        mean=fmean(scores) if scores else None, median=median(scores) if scores else None,
        stddev=pstdev(scores) if scores else None, minimum=min(scores) if scores else None,
        maximum=max(scores) if scores else None,
    )
