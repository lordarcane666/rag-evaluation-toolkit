# RAG Evaluation Toolkit

An extensible open-source Python toolkit for evaluating Retrieval-Augmented
Generation systems through deterministic metrics, model-based evaluation,
reproducible benchmarks, and detailed evaluation reports.

**Version 0.1.0 · Python 3.11+ · Apache-2.0 · Initial alpha release**

Evaluate the outputs of your RAG pipeline without coupling evaluation to its
retriever, vector store, generation framework, or model vendor. Start offline;
enable a judge only when you deliberately choose where the data will go.

This project does not claim validated semantic performance, adoption, or results
on public research benchmarks. Included datasets are explicitly synthetic.

## Why use it?

Retrieval and generation failures need different measurements. A correct answer
can use different words from its reference; a fluent answer can be unsupported;
a higher average can hide missing evaluations. This toolkit exposes those
distinctions through metric definitions, per-sample results, coverage counts,
and paired comparisons.

## Features

- Validated JSONL samples, unique IDs, UTF-8 support, bounded line reads, and
  line/field diagnostics that do not print private input values.
- Nine deterministic lexical/ID metrics, independently callable and testable.
- Four optional LLM judge metrics with structured scores, concise explanations,
  verbatim context citations, and unsupported-claim lists.
- A provider-independent judge protocol and opt-in OpenAI-compatible Chat
  Completions adapter for remote APIs and loopback inference servers.
- JSON, long-form CSV, and Markdown reports with configuration, hashes,
  timestamps, sample scores, skipped/error counts, and descriptive statistics.
- Multi-dataset, multi-configuration benchmarks over precomputed RAG outputs;
  paired regressions respect score direction and flag lost evaluation coverage.
- Custom metric/judge/exporter registration without automatic plugin imports.
- No model downloads, API keys, network access, or HTTP packages required for
  deterministic evaluation. Pydantic is the only runtime dependency.

## Installation

From a checkout or unpacked source distribution:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
rag-eval version
```

This release is not assumed to be published on PyPI. To install a locally built
wheel instead, use `python -m pip install dist/rag_evaluation_toolkit-0.1.0-py3-none-any.whl`.

## Quick start

Run from the repository root after installation:

```bash
rag-eval validate examples/demo.jsonl
rag-eval evaluate examples/demo.jsonl --metrics exact_match,token_f1,context_precision,context_recall
rag-eval evaluate examples/demo.jsonl --format json --output reports/results.json
rag-eval evaluate examples/demo.jsonl --format markdown --output reports/results.md
rag-eval evaluate examples/demo.jsonl --format csv --output reports/results.csv
rag-eval metrics
rag-eval --help
```

Output paths are created as needed. Existing files require `--overwrite`; input
files are protected even with that flag. Evaluation JSON goes only to stdout
when `--output` is absent. File status messages go to stderr.

Python API:

```python
from rag_eval_toolkit import Evaluator, load_dataset
from rag_eval_toolkit.core.models import EvaluationConfig
from rag_eval_toolkit.reporting import export_report

dataset = load_dataset("examples/demo.jsonl")
evaluator = Evaluator(EvaluationConfig(metrics=("token_f1", "context_recall"), k=3))
report = evaluator.evaluate(dataset)
assert report.dataset.sample_count == 4
print(export_report(report, "markdown"))
```

## Dataset format

One JSON object per line; the order of `contexts` is retrieval rank order:

```json
{"id":"sample-001","question":"What is RAG?","answer":"RAG combines retrieval with generation.","contexts":[{"id":"doc-1","text":"Retrieval-Augmented Generation combines document retrieval with language generation."}],"reference_answer":"RAG combines retrieval and language generation.","reference_context_ids":["doc-1"]}
```

Required: `id`, `question`, `answer`, `contexts`. An empty generated answer or
empty retrieval list is valid. `reference_answer`, `reference_context_ids`, and
JSON `metadata` are optional. Contexts require `id` and nonblank `text` and may
also have metadata. Duplicate sample IDs, duplicate context/reference IDs,
unknown fields, malformed JSON, duplicate JSON keys, and non-finite JSON numbers
are rejected. See [dataset format](docs/dataset-format.md).

## Metrics and interpretation

All scores are in `[0, 1]`. Missing requirements yield `status="skipped"` and
`score=null`, not a zero. Failed evaluations yield `status="error"`. Means and
other statistics use successful scores only and always report coverage.

| Metric | Meaning | Requirements |
| --- | --- | --- |
| `exact_match` | Equal normalized token sequences | Nonempty tokenized reference answer |
| `token_precision` | Multiset overlap / answer tokens | Reference answer |
| `token_recall` | Multiset overlap / reference tokens | Reference answer |
| `token_f1` | Harmonic mean of token precision and recall | Reference answer |
| `answer_reference_similarity` | Order-sensitive token sequence similarity | Reference answer |
| `lexical_answer_relevance` | Unique question-token coverage in answer | Tokenized question |
| `context_precision` | Relevant retrieved IDs / requested rank slots | Positive reference IDs |
| `context_recall` | Relevant retrieved IDs / reference IDs | Positive reference IDs |
| `context_f1` | Harmonic mean of context precision and recall | Positive reference IDs |

Normalization uses Unicode NFKC, casefolding, and letter/digit tokens. It does
not perform stemming, synonym expansion, language-specific segmentation, or
semantic entailment. Punctuation-insensitive exact match is explicitly a
normalized token match, not byte equality.

Use `--k N` for retrieval metrics at K. With K set, precision's denominator is
**K even if fewer documents were returned**. Without K, all retrieved contexts
are used. Empty retrieval with positive reference IDs scores zero. An absent
or empty reference ID set is skipped. Unlabelled IDs count as non-relevant;
incomplete annotations can understate retrieval quality.

None of these deterministic metrics proves that an answer is true, supported,
or semantically relevant. See [formulas and limitations](docs/metrics.md).

## Optional LLM evaluation

| Metric | Direction | Meaning |
| --- | --- | --- |
| `faithfulness` | Higher is better | Judge-estimated fraction of grounded factual claims |
| `answer_relevance` | Higher is better | Judge-estimated directness and completeness |
| `hallucination_risk` | Lower is better | Unsupported-claim risk estimate, not a calibrated probability |
| `unsupported_claims` | Lower is better | Unsupported factual claim fraction plus identified claims |

LLM evaluation requires **both** explicit opt-in and an explicit endpoint/model.
For a running, compatible local server (replace the model ID with one it serves):

```bash
rag-eval evaluate examples/demo.jsonl --metrics faithfulness,answer_relevance,hallucination_risk,unsupported_claims --enable-llm --judge-base-url http://localhost:8000/v1 --judge-model your-served-model --format json
```

For remote services, supply their HTTPS API root. Put the API key in the
`RAG_EVAL_API_KEY` environment variable using your secret manager or shell;
never place the key in a command argument, dataset, URL, or committed file.
`--api-key-env` selects another environment variable **name**.

The adapter uses the standard library, so there is no HTTP dependency to
install. It requests JSON mode by default; `--no-judge-json-mode` supports
servers lacking `response_format`. Output schema validation still applies.
The Python `JudgeOptions` API also allows `temperature=None` and `seed=None`
for servers/models that reject those parameters. Parameter compatibility
varies by server; no particular live model has been certified by this release.

JSON mode alone does not ensure schema correctness; the adapter independently
validates the returned object. See the official
[structured-output documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

Grounding metrics skip empty answers and samples without contexts. Empty
answers get zero relevance. A judge can skip answers without factual claims.
Evidence with unknown context IDs or quotes not present verbatim becomes an
error. Explanations are concise conclusions; no hidden chain-of-thought is
requested. Prompt separation reduces injection exposure but cannot make an
LLM immune to instructions embedded in adversarial documents.

## Benchmarks and regression checks

The toolkit evaluates **precomputed pipeline outputs**. It does not run a
retriever/generator or measure its latency. Store each configuration's outputs
in JSONL, retaining the same sample IDs, questions, and reference targets.

```bash
rag-eval benchmark examples/benchmark.json --output reports/benchmark.json
rag-eval evaluate examples/demo.jsonl --metrics token_f1,context_recall --format json --output reports/baseline.json
rag-eval evaluate examples/candidate.jsonl --metrics token_f1,context_recall --format json --output reports/candidate.json
rag-eval compare reports/baseline.json reports/candidate.json --threshold 0.01 --output reports/comparison.json
```

The manifest records named configurations and datasets, metric selection,
cutoff, and absolute regression threshold. Paths resolve relative to the
manifest. Every dataset must contain the baseline and the same candidate
configuration set. A manifest cannot authorize network transmission:
`--enable-llm` is still required on the command line.

Comparisons require equal sample IDs and target fingerprints, metric versions,
cutoff, judge configuration, and toolkit version. Scores are paired only where
both runs succeeded. Losing a previously successful evaluation fails the
regression gate even if the remaining average improves. Differences within
`1e-12` of the threshold boundary are treated as floating-point noise.

Summary statistics are mean, median, population standard deviation, minimum,
and maximum. Benchmark configuration summaries pool successful sample scores
across datasets (micro aggregation); per-dataset reports remain available.
Descriptive changes are **not** significance tests, confidence intervals,
calibrated risk estimates, or evidence of general real-world improvement.

## Reports and reproducibility

JSON is the lossless interchange format. CSV uses `record_type` rows for
metadata, warnings, aggregates, and samples. Markdown includes readable tables
and the full evaluation configuration. CSV formula-like identifiers are
escaped; Markdown table content is escaped.

Reports contain SHA-256 dataset/target hashes, metric versions, software
version, and judge settings excluding credentials. Raw questions, answers,
contexts, and metadata are omitted from reports; optional judge evidence and
explanations can still contain source information. Treat reports as sensitive.

For deterministic byte-stable output with unchanged data, ordering, runtime,
and dependencies, fix the report timestamp:

```bash
rag-eval evaluate examples/demo.jsonl --format json --timestamp 2026-01-01T00:00:00Z
```

Record your environment with `python -m pip freeze`. LLM scores are not
guaranteed deterministic, even with temperature zero and a fixed seed.

Exit codes: **0** completed (possibly with explicitly skipped metrics), **1**
metric errors or detected regressions/coverage loss, **2** invalid inputs,
configuration, or output errors. A skipped metric is not evidence of success;
inspect coverage before accepting a run.

## Extend the toolkit

Implement the `Metric` protocol: a versioned `MetricDefinition` and
`evaluate(sample, context) -> MetricResult`. Register it on a `MetricRegistry`
and inject that registry into `Evaluator`. Run the complete example:

```bash
python examples/custom_metric.py
```

Implement `Judge.evaluate()` and `Judge.provenance()` for another model
provider. `ExporterRegistry.register(name, callable)` supports custom report
formats. Custom dataset loaders should return an `EvaluationDataset`.
Custom Python code is trusted code, not sandboxed; dataset text is never
imported or executed. See [custom metrics](docs/custom-metrics.md) and
[architecture](docs/architecture.md).

## Privacy and security

Default evaluation makes no network calls. Explicitly enabling an LLM judge
permits sending each sample's question, answer, and retrieved context IDs/text
to that endpoint; metadata and reference answers are not sent. Remote
providers have their own retention policies. A local server can itself forward
data elsewhere. Read [privacy](docs/privacy.md) and [security](SECURITY.md).

## Development and contributing

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy
python -m build
```

Tests are also standard-library `unittest` tests and can run without the
optional pytest runner: `python -m unittest discover -s tests -t . -v`.
The suite uses synthetic data and mocked transports, with network connections
denied in test cases. CI is configured for Python 3.11, 3.12, and 3.13 on Linux
and Windows. Configuration is not a claim that remote CI has run; see the
[release verification record](docs/release-verification.md) for actual results.

Read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
and the [roadmap](docs/roadmap.md). Priorities include human-annotated metric
validation, optional bounded concurrency, and uncertainty estimates. Apache-2.0
licensing is described in [LICENSE](LICENSE); citation metadata is in
[CITATION.cff](CITATION.cff).
