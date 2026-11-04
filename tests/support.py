"""Synthetic fixtures and a network-denying test base."""

from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from rag_eval_toolkit.core.models import EvaluationDataset, EvaluationSample

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def sample(**changes):
    data = {
        "id": "s1", "question": "What is RAG?", "answer": "RAG combines retrieval and generation",
        "contexts": [{"id": "d1", "text": "RAG combines retrieval and generation."}],
        "reference_answer": "RAG combines retrieval and generation", "reference_context_ids": ["d1"],
    }
    data.update(changes)
    return EvaluationSample.model_validate(data)


def dataset(*samples):
    return EvaluationDataset(name="test", samples=samples or (sample(),))


class OfflineTestCase(TestCase):
    def setUp(self):
        super().setUp()
        for name in ("socket.socket.connect", "socket.create_connection"):
            blocker = patch(name, side_effect=AssertionError("Network forbidden in tests"))
            blocker.start()
            self.addCleanup(blocker.stop)
