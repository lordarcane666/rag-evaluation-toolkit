"""Optional semantic judge metrics, kept separate from lexical heuristics."""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval_toolkit.core.errors import JudgeError
from rag_eval_toolkit.core.models import EvaluationSample, MetricDefinition, MetricResult
from rag_eval_toolkit.judges.base import JudgeMetric
from rag_eval_toolkit.metrics.base import MetricContext, skipped


@dataclass(frozen=True)
class JudgeMetricEvaluator:
    name: JudgeMetric

    @property
    def definition(self) -> MetricDefinition:
        return MetricDefinition(
            name=self.name, kind="llm",
            higher_is_better=self.name not in {"hallucination_risk", "unsupported_claims"},
            description="Model-based judgment; subject to model bias and prompt injection.",
        )

    def evaluate(self, sample: EvaluationSample, context: MetricContext) -> MetricResult:
        if context.judge is None:
            raise JudgeError("No judge configured")
        if not sample.answer.strip():
            if self.name == "answer_relevance":
                return MetricResult(metric=self.name, score=0.0, explanation="The answer is empty.")
            return skipped(self.name, "Empty answer; no claims to assess.")
        if self.name != "answer_relevance" and not sample.contexts:
            return skipped(self.name, "No context supplied; grounding cannot be assessed.")
        verdict = context.judge.evaluate(sample, self.name)
        documents = {item.id: item.text for item in sample.contexts}
        for evidence in verdict.evidence:
            if evidence.context_id not in documents or evidence.quote not in documents[evidence.context_id]:
                raise JudgeError("Judge returned evidence that is not an exact context quotation")
        return MetricResult(metric=self.name, **verdict.model_dump())
