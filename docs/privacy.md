# Privacy and untrusted content

## Default execution

Deterministic evaluation is local. It does not download models, discover
endpoints, read API keys, or make network requests. Dataset text is parsed as
data and never executed as Python, shell commands, templates, or plugins.
Installing dependencies is a separate package-manager operation that can
contact a package index; that is not evaluation-time transmission.

## Explicit judge enablement

The CLI requires `--enable-llm`, `--judge-base-url`, and `--judge-model` when
LLM metrics are selected. Python requires `EvaluationConfig(enable_llm=True)`
and an explicitly constructed judge. The built-in adapter additionally requires
`JudgeOptions(allow_network=True)`. A benchmark manifest alone cannot opt in.

Per selected LLM metric, a request can contain the sample's question, answer,
and retrieved context IDs/text. Sample metadata, context metadata, reference
answers, reference context IDs, and sample IDs are not in the judge payload.
Endpoint providers can log/store/process the transmitted fields. A local
inference server may proxy them to a remote service. Understand that service's
policies and data routing before enabling evaluation.

## Secrets and transport

The built-in adapter puts API keys in authorization headers only. Its key is
stored as `SecretStr` and omitted from provenance. Endpoint URLs cannot contain
embedded credentials, query strings, or fragments. HTTPS is mandatory except
for loopback HTTP. TLS verification is not disabled. Redirects are rejected
to prevent forwarding a dataset or authorization header to a new destination.
There are no automatic retries, which avoids silently repeating billable
requests. Environment/system proxy settings are respected by urllib.

Transport/provider exceptions are sanitized; raw bodies, headers, and exception
strings are not included in error reports or logs. Library logs identify a
failed metric and error type using logging `extra` fields. Do not enable
third-party HTTP debugging that records credentials or bodies. A custom judge
is responsible for safe provenance and logging.

## Prompt injection and evidence

Rubrics live in a system message. The user message contains serialized JSON
with untrusted data; it is explicitly marked as non-instructional. There are
no model tools, browsing, code execution, or agent loops. Responses must
validate against a schema, and context citations must exist verbatim.

These are mitigations, not an injection-proof boundary. A malicious context
can still influence an LLM's score or explanation while meeting the schema.
Use human review and adversarial testing for consequential decisions.

## Reports and storage

Raw source texts and metadata are omitted from default reports, but judge
quotes, claims, and explanations can disclose sensitive content. IDs,
configuration labels, and content hashes may also be sensitive. Hashes are
not anonymization. Review reports before sharing; store and delete datasets
and reports according to your own retention policy. There is no telemetry,
remote upload, or automatic data retention service in the toolkit.
