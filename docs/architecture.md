# Architecture

The package uses a `src` layout and Python 3.11+ syntax. Pydantic 2 is the only
runtime dependency. The standard-library CLI avoids introducing a framework
dependency for argument parsing. The optional HTTP adapter also uses the
standard library; importing or configuring deterministic evaluation does not
initiate a connection.

```text
JSONL --> validated EvaluationDataset --> Evaluator --> EvaluationReport
                                          |                  |
                                     MetricRegistry     ExporterRegistry
                                          |
                                    Metric protocol
                                          |
                              deterministic / optional Judge

BenchmarkConfig --> repeated Evaluator calls --> paired comparisons + aggregates
```

- `core`: frozen Pydantic models, public exceptions, canonical serialization,
  and target/content hashes. Freezing prevents field reassignment, not mutation
  of nested metadata dictionaries; custom metrics must not mutate input data.
- `datasets`: bounded JSONL parsing, UTF-8/BOM support, duplicate-key/ID checks,
  and content-free diagnostics. Dataset loading is in memory after line parsing.
- `metrics`: small metric implementations sharing normalization/overlap
  primitives, an explicit registry, and a `MetricContext` for cutoff/judge
  dependency injection.
- `evaluators`: sequential orchestration and status-aware descriptive
  statistics. Errors are visible per-metric records or fail-fast exceptions;
  raw exception messages are not logged or copied into reports.
- `judges`: protocol, verdict model, and versioned rubrics. This layer does not
  know about a specific API client.
- `providers`: opt-in network transport, explicit endpoint/model, bounded
  responses, timeouts, secret-bearing headers, no redirects or retries.
- `benchmarks`: precomputed configuration outputs, strict paired target checks,
  score direction, absolute regression thresholds, and coverage regressions.
- `reporting`: interchange JSON, long-form CSV, escaped Markdown, custom
  exporters, and protected file output.
- `cli`: validation/evaluation/benchmark/compare commands and exit-code policy.

## Design decisions

No RAG framework, embeddings model, vector database, model SDK, automatic
plugin loading, dataset execution, or hidden network lookup is required.
Custom metrics are ordinary trusted Python objects. Registration rejects
duplicate names; changing a metric's semantics requires changing its version.

Successful scores, unavailable scores, and execution errors remain distinct
through aggregation and comparison. Report schemas are versioned separately
from package versions. `comparison_fingerprint` covers ID, question, reference
answer, and sorted reference IDs, deliberately excluding generated answers
and retrieved outputs so pipeline changes can be compared.

## Current bounds

The engine is synchronous and processes samples sequentially. The full dataset
and report are kept in memory. JSONL lines are capped at 2 MB by default; total
dataset size is not capped. Token sequence comparison can be quadratic on
large repeated sequences. Judge input/response caps reject oversized samples
instead of silently truncating them. There is no judge cache, retry loop,
embedding similarity, claim-decomposition model, or plugin sandbox.
