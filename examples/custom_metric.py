"""Run with: python examples/custom_metric.py (after installation)."""

from rag_eval_toolkit import Evaluator, load_dataset
from rag_eval_toolkit.core.models import EvaluationConfig, EvaluationSample, MetricDefinition, MetricResult
from rag_eval_toolkit.metrics.base import MetricContext
from rag_eval_toolkit.metrics.registry import default_registry


class NonEmptyAnswer:
    @property
    def definition(self) -> MetricDefinition:
        return MetricDefinition(
            name="non_empty_answer", kind="deterministic", version="1",
            description="Checks presence of non-whitespace output; not answer quality.",
        )

    def evaluate(self, sample: EvaluationSample, context: MetricContext) -> MetricResult:
        return MetricResult(metric=self.definition.name, score=float(bool(sample.answer.strip())))


if __name__ == "__main__":
    from pathlib import Path

    registry = default_registry()
    registry.register(NonEmptyAnswer())
    report = Evaluator(
        EvaluationConfig(metrics=("token_f1", "non_empty_answer")), registry=registry,
    ).evaluate(load_dataset(Path(__file__).with_name("demo.jsonl")))
    print(report.aggregates["non_empty_answer"].model_dump_json())
