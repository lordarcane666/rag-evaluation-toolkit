# JSONL dataset contract

Encode datasets as UTF-8, one JSON object per line. A UTF-8 BOM is accepted only
at the start. Blank lines are ignored and reported as warnings. Files are
read locally; URLs are not dataset inputs. Each line is limited to 2,000,000
bytes by default (`load_dataset(..., max_line_bytes=...)` can adjust this).

| Field | Type | Rule |
| --- | --- | --- |
| id | string | Required, nonblank, unique within dataset, at most 256 characters |
| question | string | Required, nonblank |
| answer | string | Required; empty is a valid generated failure |
| contexts | array of objects | Required; empty array is valid |
| reference_answer | string or null | Optional; absent/empty-token references skip reference metrics |
| reference_context_ids | array of strings or null | Optional; unique IDs; absent/empty sets skip retrieval metrics |
| metadata | JSON object | Optional; not sent to judges or copied into reports |

Each context has a nonblank `id`, nonblank `text`, and optional JSON metadata.
Context order is retrieval order, with the first item at rank 1. Context IDs
must be unique within a sample. Reference context IDs may point to documents
not retrieved; that is how missing relevant documents are represented.

Unknown model fields are rejected. Numeric IDs are not coerced to strings.
ID whitespace is stripped. JSON duplicate keys and NaN/Infinity are invalid.
Diagnostics show line numbers and field locations, omitting actual field
values. Sample IDs are included in reports and should not contain secrets.

The examples in `examples/` are synthetic, not customer data. `demo.jsonl`
deliberately contains an incorrect answer, a retrieval miss, and a sample
without references. `candidate.jsonl` has matching evaluation targets but
different generated/retrieved outputs for a controlled comparison.

Custom loaders can construct `EvaluationSample` objects and wrap them in
`EvaluationDataset(name=..., samples=(...))`. Models validate independently
of the file parser. Nested metadata is user-controlled data; do not execute
it, use it as a path/command, or put credentials into it.
