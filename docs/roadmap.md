# Roadmap

The 0.1.0 release establishes validated data contracts, offline deterministic
metrics, opt-in structured judges, reporting, paired regression gates, and
benchmarks over saved pipeline outputs. It is an initial alpha API.

Next development priorities:

1. Validate judge scores against a licensed human-annotated RAG corpus; publish
   protocols, uncertainty, and disagreements rather than only correlations.
2. Add paired bootstrap confidence intervals and explicit minimum-coverage
   acceptance policies, with statistical tests reviewed independently.
3. Add streaming aggregation for large datasets and bounded optional judge
   concurrency with documented cancellation and billing behavior.
4. Add multilingual tokenization strategies as optional dependencies, keeping
   lexical and semantic claims distinct.
5. Add human-auditable claim segmentation, support/contradiction labels, and
   evidence provenance without requesting private chain-of-thought.
6. Consider content-addressed local judge caching, with opt-in retention and
   complete prompt/model version keys.

These are future work, not implemented features. Live-provider compatibility,
large-scale performance, multilingual effectiveness, and semantic accuracy
must be measured before making claims about them.
