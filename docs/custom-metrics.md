# Extension interfaces

The full runnable metric example is `examples/custom_metric.py`:

```bash
python examples/custom_metric.py
```

Implement two protocol members: a `definition` property returning
`MetricDefinition(name, kind, higher_is_better, version, description)` and
`evaluate(sample, context)` returning `MetricResult`. Return `skipped` with a
null score and a reason when reference requirements are missing. Scores for
successful results must be finite and in `[0, 1]`. Version changes are required
when metric semantics change, so comparisons can reject incompatible reports.

Register explicitly using `registry.register(metric)` and pass
`Evaluator(configuration, registry=registry)`. Duplicate names are rejected.
Return the registered name in every result. `MetricContext` carries the rank
cutoff and optional judge. The engine records unexpected metric exceptions as
error results without their raw messages; `fail_fast=True` stops evaluation.

Implement `Judge.evaluate(sample, metric) -> JudgeVerdict` and
`Judge.provenance() -> dict[str, JsonValue]` for a provider. Provenance must
contain only non-secret settings. Return evidence as context ID plus exact
quotation; the metric layer checks that quotations exist even for custom
judges. If a provider does not support a metric, return an explicit skipped
verdict instead of fabricating a score. Explicit LLM opt-in still applies.

`OpenAICompatibleJudge` accepts a `ChatTransport` object; tests use this seam
to inspect requests and return synthetic responses without an internet
connection. A transport must enforce timeouts and bounds and must not log
private payloads. Returning a fixture verdict is useful for integration tests
but is not an actual model benchmark.

Custom dataset loaders return validated `EvaluationDataset` objects. Custom
report exporters are callables from `EvaluationReport` to `str`, registered
with `ExporterRegistry.register`. Use `write_output` for protected local
output or implement your own authorized storage layer. CLI formats are the
three built-ins; custom exporters are available through the Python API.

Extensions execute trusted Python in the caller's process. The toolkit does
not discover code from datasets, load arbitrary module paths, sandbox plugins,
or enforce a no-network policy on user-written plugins.
