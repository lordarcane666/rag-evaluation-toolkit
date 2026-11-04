"""Ranked retrieval metrics based exclusively on reference document IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag_eval_toolkit.core.models import EvaluationSample, MetricDefinition, MetricResult
from rag_eval_toolkit.metrics.base import MetricContext, skipped

ContextMode = Literal["context_precision", "context_recall", "context_f1"]


@dataclass(frozen=True)
class ContextMetric:
    mode: ContextMode

    @property
    def definition(self) -> MetricDefinition:
        return MetricDefinition(
            name=self.mode, kind="deterministic",
            description="Reference-ID retrieval metric; k limits retrieval ranks, not reference IDs.",
        )

    def evaluate(self, sample: EvaluationSample, context: MetricContext) -> MetricResult:
        if sample.reference_context_ids is None:
            return skipped(self.mode, "No reference context IDs provided.")
        if not sample.reference_context_ids:
            return skipped(self.mode, "Reference context ID set is empty; no positive relevance labels.")
        references = set(sample.reference_context_ids)
        retrieved = sample.contexts[:context.k]
        hits = len({item.id for item in retrieved} & references)
        denominator = context.k if context.k is not None else len(retrieved)
        precision = hits / denominator if denominator else 0.0
        recall = hits / len(references)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score = {"context_precision": precision, "context_recall": recall, "context_f1": f1}[self.mode]
        return MetricResult(
            metric=self.mode, score=score,
            explanation="ID-based relevance labels; unlabelled retrieved IDs count as non-relevant.",
            details={
                "hits": hits, "retrieved": len(retrieved), "references": len(references),
                "precision_denominator": denominator, "k": context.k,
            },
        )
