"""Sequential, deterministic orchestration with explicit per-metric failure results."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from rag_eval_toolkit.core.errors import ConfigurationError, EvaluationError
from rag_eval_toolkit.core.models import (
    DatasetInfo, EvaluationConfig, EvaluationDataset, EvaluationReport, EvaluationResult, MetricResult,
)
from rag_eval_toolkit.core.serialization import comparison_fingerprint, dataset_digest
from rag_eval_toolkit.evaluators.statistics import aggregate
from rag_eval_toolkit.judges.base import Judge
from rag_eval_toolkit.metrics.base import MetricContext
from rag_eval_toolkit.metrics.registry import DEFAULT_METRICS, MetricRegistry, default_registry

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self, configuration: EvaluationConfig | None = None, *,
        registry: MetricRegistry | None = None, judge: Judge | None = None,
    ) -> None:
        self.configuration = configuration or EvaluationConfig(metrics=DEFAULT_METRICS)
        self.registry = registry if registry is not None else default_registry()
        self.judge = judge
        self.metrics = tuple(self.registry.get(name) for name in self.configuration.metrics)
        self.uses_llm = any(metric.definition.kind == "llm" for metric in self.metrics)
        if self.uses_llm and not self.configuration.enable_llm:
            raise ConfigurationError("LLM metrics require explicit enable_llm=True / --enable-llm")
        if self.uses_llm and self.judge is None:
            raise ConfigurationError("LLM metrics require a configured judge")

    def evaluate(
        self, dataset: EvaluationDataset, *, timestamp: datetime | None = None,
    ) -> EvaluationReport:
        created_at = timestamp or datetime.now(UTC)
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ConfigurationError("Evaluation timestamp must include a timezone")
        results: list[EvaluationResult] = []
        warnings = list(dataset.warnings)
        context = MetricContext(k=self.configuration.k, judge=self.judge if self.uses_llm else None)
        for sample in dataset.samples:
            metric_results = []
            for metric in self.metrics:
                try:
                    result = metric.evaluate(sample, context)
                    if result.metric != metric.definition.name:
                        raise ValueError("metric result name does not match its definition")
                except Exception as exc:
                    # Plugin/provider errors become visible error records, never raw exception dumps.
                    logger.warning("metric_failed", extra={
                        "metric": metric.definition.name, "error_type": type(exc).__name__,
                    })
                    reason = f"Metric failed ({type(exc).__name__}); inspect the metric/provider configuration."
                    if self.configuration.fail_fast:
                        raise EvaluationError(reason) from None
                    result = MetricResult(metric=metric.definition.name, status="error", explanation=reason)
                metric_results.append(result)
                if result.status == "error" and self.configuration.fail_fast:
                    raise EvaluationError(f"Metric returned an error status: {metric.definition.name}")
            results.append(EvaluationResult(
                sample_id=sample.id, comparison_fingerprint=comparison_fingerprint(sample),
                metrics=tuple(metric_results),
            ))
        aggregates = {
            name: aggregate(metric for result in results for metric in result.metrics if metric.metric == name)
            for name in self.configuration.metrics
        }
        if any(item.skipped for item in aggregates.values()):
            warnings.append("Skipped evaluations are excluded from aggregates; compare coverage counts.")
        if any(item.errors for item in aggregates.values()):
            warnings.append("Some metrics failed; error results are excluded from aggregate scores.")
        warnings.append("Lexical/ID metrics are heuristics, not validated semantic or factuality measures.")
        if self.uses_llm:
            warnings.append("LLM scores may vary; prompt isolation does not guarantee injection resistance.")
        return EvaluationReport(
            created_at=created_at.astimezone(UTC),
            dataset=DatasetInfo(name=dataset.name, sample_count=len(dataset.samples), sha256=dataset_digest(dataset)),
            configuration=self.configuration,
            metric_definitions=tuple(metric.definition for metric in self.metrics),
            judge=self.judge.provenance() if self.uses_llm and self.judge is not None else None,
            results=tuple(results), aggregates=aggregates, warnings=tuple(warnings),
        )
