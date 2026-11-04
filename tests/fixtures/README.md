# Fixtures

Synthetic reusable objects are constructed in `tests/support.py`. Complete
JSONL fixtures live in `examples/` so tests exercise the same files users run.
Malformed file fixtures and provider responses are generated inside isolated
test temporary directories/in-memory transports. No external datasets or
credentials are required.
