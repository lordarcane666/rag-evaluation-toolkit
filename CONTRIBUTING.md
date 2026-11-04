# Contributing

Contributions should solve a concrete evaluation problem and include synthetic
regression tests. Open a focused issue or pull request describing the trigger,
expected behavior, limitations, and validation. Do not submit private datasets,
credentials, artificial adoption metrics, or third-party content without rights.

## Development

Use Python 3.11 or newer in a virtual environment, then run from the root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy
python -m build
rag-eval validate examples/demo.jsonl
```

The default tests require no API keys or internet. Tests subclass
`OfflineTestCase`, which blocks network connections. Use an injected transport
or mock urllib for HTTP tests. Tests also run with the standard-library runner:
`python -m unittest discover -s tests -t . -v`.

Keep validation, orchestration, metric logic, providers, and exporters separate.
Use type hints in library code and explain formulas and edge cases. Do not
replace an unavailable metric with a misleading zero. Metric behavior changes
must update its version and documentation. Check lower-is-better metrics,
missing references, duplicate tokens, Unicode, and paired sample coverage.

Do not add network activity to default commands or automatic provider/plugin
discovery. Do not log dataset text, keys, raw HTTP responses, or exception
messages that could contain secrets. Update the changelog and runnable examples
when public behavior changes. Contributions are submitted under Apache-2.0.
