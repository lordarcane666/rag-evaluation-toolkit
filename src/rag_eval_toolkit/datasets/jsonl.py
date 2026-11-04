"""Bounded UTF-8 JSONL loading with useful, content-free validation errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

from pydantic import ValidationError

from rag_eval_toolkit.core.errors import DatasetError
from rag_eval_toolkit.core.models import EvaluationDataset, EvaluationSample


def _reject_constant(value: str) -> NoReturn:
    raise ValueError("non-finite JSON numbers are not allowed")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def load_dataset(
    path: str | Path, *, name: str | None = None, max_line_bytes: int = 2_000_000,
) -> EvaluationDataset:
    """Load JSONL; report line/field locations without printing dataset contents."""
    if max_line_bytes < 1:
        raise ValueError("max_line_bytes must be positive")
    source = Path(path)
    samples: list[EvaluationSample] = []
    seen: set[str] = set()
    blank_count = 0
    try:
        with source.open("rb") as stream:
            line_number = 0
            while line := stream.readline(max_line_bytes + 1):
                line_number += 1
                if len(line) > max_line_bytes:
                    raise DatasetError(f"Line {line_number}: exceeds max_line_bytes")
                try:
                    text = line.decode("utf-8-sig" if line_number == 1 else "utf-8")
                    if not text.strip():
                        blank_count += 1
                        continue
                    raw = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)
                    sample = EvaluationSample.model_validate(raw)
                except ValidationError as exc:
                    locations = "; ".join(
                        f"{'.'.join(map(str, err['loc'])) or 'sample'}: {err['type']}"
                        for err in exc.errors(include_input=False, include_url=False)
                    )
                    raise DatasetError(f"Line {line_number}: {locations}") from None
                except (ValueError, RecursionError):
                    raise DatasetError(
                        f"Line {line_number}: invalid UTF-8/JSON, duplicate key, or non-finite number"
                    ) from None
                if sample.id in seen:
                    raise DatasetError(f"Line {line_number}: duplicate sample ID")
                seen.add(sample.id)
                samples.append(sample)
    except OSError:
        raise DatasetError(f"Cannot read dataset file: {source.name}") from None
    if not samples:
        raise DatasetError("Dataset contains no samples")
    warnings = (f"Ignored {blank_count} blank JSONL line(s).",) if blank_count else ()
    return EvaluationDataset(name=name or source.stem, samples=tuple(samples), warnings=warnings)
