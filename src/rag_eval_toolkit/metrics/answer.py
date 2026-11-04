"""Reference comparisons and explicitly lexical question-answer relevance."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from rag_eval_toolkit.core.models import EvaluationSample, MetricDefinition, MetricResult
from rag_eval_toolkit.metrics.base import MetricContext, skipped
from rag_eval_toolkit.metrics.text import overlap_scores, tokenize

AnswerMode = Literal[
    "exact_match", "token_precision", "token_recall", "token_f1", "answer_reference_similarity",
]


@dataclass(frozen=True)
class AnswerMetric:
    mode: AnswerMode

    @property
    def definition(self) -> MetricDefinition:
        descriptions = {
            "exact_match": "Equality of normalized token sequences (lexical).",
            "token_precision": "Multiset token overlap divided by answer token count.",
            "token_recall": "Multiset token overlap divided by reference token count.",
            "token_f1": "Harmonic mean of multiset token precision and recall.",
            "answer_reference_similarity": "Token sequence similarity using SequenceMatcher.",
        }
        return MetricDefinition(name=self.mode, kind="deterministic", description=descriptions[self.mode])

    def evaluate(self, sample: EvaluationSample, context: MetricContext) -> MetricResult:
        if sample.reference_answer is None:
            return skipped(self.mode, "No reference answer provided.")
        reference = tokenize(sample.reference_answer)
        if not reference:
            return skipped(self.mode, "Reference answer contains no lexical tokens.")
        prediction = tokenize(sample.answer)
        if self.mode == "exact_match":
            score = float(prediction == reference)
        elif self.mode == "answer_reference_similarity":
            score = SequenceMatcher(None, prediction, reference, autojunk=False).ratio()
        else:
            precision, recall, f1 = overlap_scores(prediction, reference)
            score = {"token_precision": precision, "token_recall": recall, "token_f1": f1}[self.mode]
        return MetricResult(
            metric=self.mode, score=score,
            explanation="Lexical comparison; does not measure factuality or semantic equivalence.",
            details={"answer_tokens": len(prediction), "reference_tokens": len(reference)},
        )


class LexicalAnswerRelevance:
    @property
    def definition(self) -> MetricDefinition:
        return MetricDefinition(
            name="lexical_answer_relevance", kind="deterministic",
            description="Fraction of unique question tokens present in the answer; lexical heuristic.",
        )

    def evaluate(self, sample: EvaluationSample, context: MetricContext) -> MetricResult:
        question = set(tokenize(sample.question))
        if not question:
            return skipped(self.definition.name, "Question contains no lexical tokens.")
        matched = question & set(tokenize(sample.answer))
        return MetricResult(
            metric=self.definition.name, score=len(matched) / len(question),
            explanation="Question-token coverage only; not a semantic relevance assessment.",
            details={"question_tokens": len(question), "matched_tokens": len(matched)},
        )
