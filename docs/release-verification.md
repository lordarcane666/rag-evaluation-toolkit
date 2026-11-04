# 0.1.0 release verification

Verification is recorded for this initial local release on 2026-11-04. This
file distinguishes executed checks from configured or unavailable checks.

The local environment supplies Windows, CPython 3.12.14, Pydantic 2.14.1,
setuptools 84.2.1, wheel 0.48.0, and Git 2.56.0.windows.2. Runtime dependencies
were reused from an existing local installation; no package index was contacted.

The test suite is pytest-compatible and also runs directly with unittest.
The executed unittest suite passed: **96 tests, 0 failures, 0 errors, 0 skips**.
The suite includes model and JSONL validation, metric formulas and edge cases,
judge transport/schema/evidence errors, privacy defaults, reporting, comparison
direction, coverage regressions, and CLI integration.

| Check | Actual local result |
| --- | --- |
| Standard-library unittest suite | 96 passed on Python 3.12.14 |
| README examples | 15 CLI executions, one Python API block, and custom metric example passed |
| Deterministic reproducibility | Repeated fixed-timestamp JSON matched byte for byte |
| Python 3.11 syntax | All library modules parsed with Python 3.11 grammar |
| Bytecode compilation | Source, tests, examples, and tools compiled |
| Wheel and source distribution | Built using installed setuptools PEP 517 backend |
| Wheel integrity | RECORD hashes, metadata, CLI entry point, and py.typed verified |
| Local wheel installation | Installed in a separate venv; import resolves to installed site-packages |
| Installed CLI | Version, validation, and JSON evaluation executed successfully |
| pytest / Ruff / mypy | Unavailable; installation and invocation attempts failed |
| Remote GitHub CI | Configured only; not executed |

The local venv reuses the existing Pydantic dependency installation through
system-site-packages; the toolkit itself is installed from its wheel. This
proves local packaging/import/CLI behavior, not a clean online dependency
resolution or execution on every supported Python/platform combination.

Normal reproduction commands (with development tools installed):

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy
python -m build
python tools/verify_wheel.py
python tools/verify_examples.py
```

Offline alternatives actually used are `python -m unittest discover -s tests
-t . -q` and `python tools/build_local.py`. Ruff and mypy still need to run in
a provisioned environment before declaring all quality gates green.

The environment has `PIP_NO_INDEX=1`, no cached pytest/Ruff/mypy/build wheels,
and an unavailable network proxy. Installing these development tools failed.
They are declared in the dev extra and configured in CI; they have not passed
locally. The PEP 517 backend can build offline through `tools/build_local.py`.

Python's mode-0700 temporary directories omit this sandbox's Windows SID.
The local verification launcher, outside the repository, uses inherited ACLs
only for new temporary directories under the authorized workspace. Test and
library code are not modified or bypassed by that environment accommodation.

GitHub account verification failed because the connected tool returned
`Unknown tool github.get_profile`. GitHub CLI was not installed and the API
connection failed at the configured proxy. The authenticated owner, repository
existence, and restriction status could not be established. No remote
repository was created or pushed; no account restriction was assumed or bypassed.

Remote CI on Python 3.11/3.12/3.13 and live LLM provider validation are not
claimed. No real external judge call or scientific benchmark was executed.
Synthetic example outputs demonstrate software behavior only.
