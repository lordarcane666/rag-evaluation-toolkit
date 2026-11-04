# Metric definitions and limitations

Let `A` and `R` be normalized answer/reference token lists, `Q` the set of
question tokens, and `H = sum((Counter(A) & Counter(R)).values())`.
Normalization is NFKC, Unicode casefold, and `[^\W_]+` token extraction.
Punctuation and underscores separate tokens. Stop words are retained. This
is not a multilingual tokenizer; scripts without word boundaries need a
custom metric/tokenizer. No language-specific stemming is performed.

| Name | Formula / rule |
| --- | --- |
| exact_match | `float(A == R)` |
| token_precision | `H / len(A)`; zero if A is empty and R nonempty |
| token_recall | `H / len(R)` |
| token_f1 | `2PR/(P+R)`; zero if P+R is zero |
| answer_reference_similarity | `SequenceMatcher(None, A, R, autojunk=False).ratio()` |
| lexical_answer_relevance | `len(Q & set(A)) / len(Q)` |

Reference answer metrics skip absent references and references that normalize
to zero tokens. An empty prediction against a nonempty reference scores zero.
Lexical relevance skips questions with no tokens. Sequence similarity is
order-sensitive and may be asymmetric when arguments are swapped; the toolkit
always places the generated answer first. It is not semantic similarity.

For retrieval, let `T` be the nonempty reference ID set, `D` the retrieved IDs
through rank K, and `h = len(D & T)`.

- Precision at K: `h/K`; unfilled retrieval slots count as misses.
- With no K, precision uses `h / returned_document_count`, or zero for empty
  retrieval when positive reference IDs exist.
- Recall at K: `h/len(T)`. The denominator always includes all reference IDs.
- Context F1: harmonic mean of context precision and recall.

Reference IDs are exact case-sensitive labels. Duplicate retrieved IDs and
reference IDs are invalid input. If reference IDs are absent or an empty list,
all context metrics skip: the toolkit does not invent positive labels. These
metrics evaluate labelled retrieval, not whether passage text is meaningful,
truthful, sufficient, or correctly cited by the answer.

## Judge metrics

Faithfulness estimates the fraction of factual claims grounded in supplied
contexts. Answer relevance estimates whether the answer directly and fully
addresses the question. Hallucination risk estimates grounding risk;
unsupported-claims detection returns an estimated unsupported-claim fraction
and an explicit list. The latter two metrics are lower-is-better.

Grounding is not truth: an answer grounded in a false context can receive a
high faithfulness score. Lack of supplied evidence is not proof of falsehood.
Judges differ in claim boundaries and may produce inconsistent scores across
these metrics, which are evaluated separately. Evidence validation proves
only that a quotation is present, not that it entails the model's conclusion.

Judges must return a structured score, status, concise explanation, evidence,
and unsupported claims. Out-of-range/non-finite scores, unknown fields,
truncated/refused completions, and invented evidence become explicit errors.
The toolkit does not request hidden chain-of-thought. These metrics have not
been calibrated or validated against a human-annotated corpus in this release.

## Aggregation and comparisons

Use successful samples only for mean, median, population standard deviation,
minimum, and maximum. One score has standard deviation zero; zero scores
yield null statistics. Skipped/error counts are always separate.

Paired comparisons align exact sample IDs and target fingerprints, then use
the intersection of successful evaluations. They reject incompatible cutoff,
metric version, toolkit version, or judge provenance. `delta` is candidate
minus baseline; `improvement` flips the sign for lower-is-better metrics.
An improvement below the negative absolute threshold is a regression; loss
of successful coverage is also a regression even without any paired scores.

These are descriptive comparisons, with no significance claims or confidence
intervals. Inspect individual samples and evaluation coverage before decisions.
