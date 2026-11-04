# Complete source tree

Tracked project source and configuration. Generated `.git/`, `.venv/`, `build/`, `dist/`, `reports/`, package metadata, and caches are excluded. The local distribution artifacts are in `dist/`.

```text
rag-evaluation-toolkit/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── workflows/
│   │   └── ci.yml
│   ├── dependabot.yml
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/
│   ├── architecture.md
│   ├── custom-metrics.md
│   ├── dataset-format.md
│   ├── metrics.md
│   ├── privacy.md
│   ├── project-tree.md
│   ├── publishing.md
│   ├── release-verification.md
│   └── roadmap.md
├── examples/
│   ├── benchmark.json
│   ├── candidate.jsonl
│   ├── custom_metric.py
│   ├── demo.jsonl
│   ├── operations-baseline.jsonl
│   └── operations-candidate.jsonl
├── src/
│   └── rag_eval_toolkit/
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   ├── compare.py
│       │   └── runner.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── app.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── errors.py
│       │   ├── models.py
│       │   └── serialization.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   └── jsonl.py
│       ├── evaluators/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   └── statistics.py
│       ├── judges/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── prompts.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── answer.py
│       │   ├── base.py
│       │   ├── context.py
│       │   ├── llm.py
│       │   ├── registry.py
│       │   └── text.py
│       ├── providers/
│       │   ├── __init__.py
│       │   └── openai_compatible.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── exporters.py
│       │   └── io.py
│       ├── __init__.py
│       ├── __main__.py
│       ├── _version.py
│       └── py.typed
├── tests/
│   ├── fixtures/
│   │   └── README.md
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_cli.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_judges.py
│   │   ├── test_metrics_engine.py
│   │   ├── test_models_datasets.py
│   │   └── test_reporting_benchmarks.py
│   ├── __init__.py
│   └── support.py
├── tools/
│   ├── build_local.py
│   ├── verify_examples.py
│   └── verify_wheel.py
├── .gitattributes
├── .gitignore
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── MANIFEST.in
├── NOTICE
├── pyproject.toml
├── README.md
└── SECURITY.md
```
