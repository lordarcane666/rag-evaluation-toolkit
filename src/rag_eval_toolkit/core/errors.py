"""Public exceptions with messages safe for command-line display."""


class RAGEvalError(Exception):
    """Base class for actionable toolkit errors."""


class DatasetError(RAGEvalError):
    """Invalid or unreadable evaluation data."""


class ConfigurationError(RAGEvalError):
    """Invalid evaluation or provider configuration."""


class EvaluationError(RAGEvalError):
    """An evaluation failed in fail-fast mode."""


class JudgeError(RAGEvalError):
    """Judge failure; messages must not include credentials or response bodies."""


class ReportError(RAGEvalError):
    """Invalid report or unsafe output operation."""
