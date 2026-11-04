"""Provider-independent judge contracts; importing them does not enable networking."""

from rag_eval_toolkit.judges.base import Judge, JudgeVerdict

__all__ = ["Judge", "JudgeVerdict"]
