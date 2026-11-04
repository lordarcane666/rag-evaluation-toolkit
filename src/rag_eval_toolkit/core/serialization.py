"""Canonical JSON and content hashes (never hashes of provider credentials)."""

from __future__ import annotations

import hashlib
import json

from rag_eval_toolkit.core.models import EvaluationDataset, EvaluationSample


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dataset_digest(dataset: EvaluationDataset) -> str:
    return digest([sample.model_dump(mode="json") for sample in dataset.samples])


def comparison_fingerprint(sample: EvaluationSample) -> str:
    """Hash evaluation targets, excluding the pipeline's answer/retrieval outputs."""
    return digest({
        "id": sample.id,
        "question": sample.question,
        "reference_answer": sample.reference_answer,
        "reference_context_ids": (
            sorted(sample.reference_context_ids) if sample.reference_context_ids is not None else None
        ),
    })
