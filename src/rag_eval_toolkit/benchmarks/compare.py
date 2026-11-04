"""Paired run comparison with direction-aware deltas and coverage regression detection."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Literal

from rag_eval_toolkit.core.errors import ConfigurationError
from rag_eval_toolkit.core.models import EvaluationReport, MetricComparison, RunComparison


def compare_reports(
    baseline: EvaluationReport, candidate: EvaluationReport, *, threshold: float = 0.0,
) -> RunComparison:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ConfigurationError("Regression threshold must be finite and between 0 and 1")
    left = {result.sample_id: result for result in baseline.results}
    right = {result.sample_id: result for result in candidate.results}
    if left.keys() != right.keys():
        raise ConfigurationError("Reports must have identical sample IDs for a paired comparison")
    if any(left[key].comparison_fingerprint != right[key].comparison_fingerprint for key in left):
        raise ConfigurationError("Questions or reference targets differ; reports are not comparable")
    definitions = {definition.name: definition for definition in baseline.metric_definitions}
    other_definitions = {definition.name: definition for definition in candidate.metric_definitions}
    if (definitions != other_definitions or baseline.configuration.k != candidate.configuration.k
            or baseline.judge != candidate.judge or baseline.toolkit_version != candidate.toolkit_version):
        raise ConfigurationError("Metric versions, cutoff, judge configuration, and toolkit version must match")
    comparisons: list[MetricComparison] = []
    warnings: list[str] = []
    for name, definition in definitions.items():
        pairs: list[tuple[float, float]] = []
        coverage_regression = False
        for key in left:
            before = next(metric for metric in left[key].metrics if metric.metric == name)
            after = next(metric for metric in right[key].metrics if metric.metric == name)
            if before.status == "ok" and after.status != "ok":
                coverage_regression = True
            if before.score is not None and after.score is not None:
                pairs.append((before.score, after.score))
        excluded = len(left) - len(pairs)
        if excluded:
            warnings.append(f"{name}: {excluded} sample(s) excluded from paired scores.")
        if coverage_regression:
            warnings.append(f"{name}: candidate lost successful evaluations; regression gate fails.")
        if pairs:
            before_mean = fmean(before for before, _ in pairs)
            after_mean = fmean(after for _, after in pairs)
            delta = after_mean - before_mean
            improvement = delta if definition.higher_is_better else -delta
            # Ignore only floating-point noise at the explicitly chosen threshold boundary.
            significant = abs(improvement) > threshold and not math.isclose(
                abs(improvement), threshold, abs_tol=1e-12, rel_tol=1e-12,
            )
            status: Literal["improved", "regressed", "unchanged"] = "improved" if significant and improvement > 0 else (
                "regressed" if significant else "unchanged"
            )
            comparisons.append(MetricComparison(
                metric=name, paired_count=len(pairs), excluded_count=excluded,
                coverage_regression=coverage_regression, baseline_mean=before_mean,
                candidate_mean=after_mean, delta=delta, improvement=improvement, status=status,
            ))
        else:
            comparisons.append(MetricComparison(
                metric=name, paired_count=0, excluded_count=excluded,
                coverage_regression=coverage_regression, status="not_applicable",
            ))
    return RunComparison(
        baseline_dataset=baseline.dataset.name, candidate_dataset=candidate.dataset.name,
        threshold=threshold, metrics=tuple(comparisons),
        has_regression=any(item.status == "regressed" or item.coverage_regression for item in comparisons),
        warnings=tuple(warnings),
    )
