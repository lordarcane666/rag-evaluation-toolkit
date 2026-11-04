"""Explicit metric registration; no import-time plugin discovery or code loading."""

from __future__ import annotations

from rag_eval_toolkit.core.errors import ConfigurationError
from rag_eval_toolkit.judges.base import JudgeMetric
from rag_eval_toolkit.metrics.answer import AnswerMetric, AnswerMode, LexicalAnswerRelevance
from rag_eval_toolkit.metrics.base import Metric
from rag_eval_toolkit.metrics.context import ContextMetric, ContextMode
from rag_eval_toolkit.metrics.llm import JudgeMetricEvaluator

DEFAULT_METRICS = (
    "exact_match", "token_precision", "token_recall", "token_f1", "answer_reference_similarity",
    "lexical_answer_relevance", "context_precision", "context_recall", "context_f1",
)


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        name = metric.definition.name
        if name in self._metrics:
            raise ConfigurationError(f"Metric already registered: {name}")
        self._metrics[name] = metric

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError:
            raise ConfigurationError(f"Unknown metric: {name}. Use 'rag-eval metrics'.") from None

    def names(self) -> tuple[str, ...]:
        return tuple(self._metrics)


def default_registry() -> MetricRegistry:
    registry = MetricRegistry()
    answer_modes: tuple[AnswerMode, ...] = (
        "exact_match", "token_precision", "token_recall", "token_f1", "answer_reference_similarity",
    )
    context_modes: tuple[ContextMode, ...] = ("context_precision", "context_recall", "context_f1")
    judge_names: tuple[JudgeMetric, ...] = (
        "faithfulness", "answer_relevance", "hallucination_risk", "unsupported_claims",
    )
    for mode in answer_modes:
        registry.register(AnswerMetric(mode))
    registry.register(LexicalAnswerRelevance())
    for context_mode in context_modes:
        registry.register(ContextMetric(context_mode))
    for judge_name in judge_names:
        registry.register(JudgeMetricEvaluator(judge_name))
    return registry
