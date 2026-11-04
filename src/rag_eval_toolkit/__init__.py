"""RAG Evaluation Toolkit: explicit, reproducible evaluation."""

from rag_eval_toolkit._version import __version__
from rag_eval_toolkit.core.models import (
    BenchmarkResult,
    EvaluationDataset,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
    MetricResult,
    RetrievedContext,
)
from rag_eval_toolkit.datasets.jsonl import load_dataset
from rag_eval_toolkit.evaluators.engine import Evaluator

__all__ = [
    "BenchmarkResult", "EvaluationDataset", "EvaluationReport", "EvaluationResult",
    "EvaluationSample", "Evaluator", "MetricResult", "RetrievedContext", "load_dataset", "__version__",
]
