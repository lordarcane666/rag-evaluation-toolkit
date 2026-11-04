"""Reproducible evaluation of precomputed RAG pipeline outputs."""

from rag_eval_toolkit.benchmarks.compare import compare_reports
from rag_eval_toolkit.benchmarks.runner import run_benchmark

__all__ = ["compare_reports", "run_benchmark"]
