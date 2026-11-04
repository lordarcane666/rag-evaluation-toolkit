"""Documented Unicode lexical normalization and multiset overlap primitives."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


def tokenize(text: str) -> list[str]:
    """NFKC, Unicode casefold, then runs of Unicode letters/digits; no stemming."""
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def overlap_scores(prediction: list[str], reference: list[str]) -> tuple[float, float, float]:
    """Multiset precision/recall/F1; callers decide whether an empty target is applicable."""
    matches = sum((Counter(prediction) & Counter(reference)).values())
    precision = matches / len(prediction) if prediction else 0.0
    recall = matches / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
