"""Judge protocol and structured, concise verdicts without chain-of-thought."""

from __future__ import annotations

from typing import Literal, Protocol, Self

from pydantic import Field, JsonValue, model_validator

from rag_eval_toolkit.core.models import EvaluationSample, Evidence, Model, Score

JudgeMetric = Literal["faithfulness", "answer_relevance", "hallucination_risk", "unsupported_claims"]


class JudgeVerdict(Model):
    status: Literal["ok", "skipped"]
    score: Score | None
    explanation: str = Field(min_length=1, max_length=2000)
    evidence: tuple[Evidence, ...] = ()
    unsupported_claims: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if (self.status == "ok") != (self.score is not None):
            raise ValueError("only ok verdicts must contain a numeric score")
        return self


class Judge(Protocol):
    def evaluate(self, sample: EvaluationSample, metric: JudgeMetric) -> JudgeVerdict: ...

    def provenance(self) -> dict[str, JsonValue]:
        """Return non-secret settings only: provider, model, prompt version, etc."""
        ...
