# Publishing under the intended owner

The only intended GitHub destination is
`https://github.com/lordarcane666/rag-evaluation-toolkit`. A local remote URL
does not establish that the remote exists or that an account is authenticated.
Do not publish this repository under another account.

Before a first publication, in an environment with Git, GitHub CLI, and a
working authenticated network connection:

1. Run `gh api user --jq .login` and verify the exact result is
   `lordarcane666`. Stop on authentication errors or a different login.
2. Query `gh api repos/lordarcane666/rag-evaluation-toolkit` to check existence.
   If it exists, inspect it and reconcile explicitly; do not overwrite it or
   force-push. Only a confirmed 404 with successful authentication supports
   creating a new repository. Connectivity/permission failures are not 404s.
3. Complete the development checks in CONTRIBUTING.md. Review the tracked
   file list and commit content for secrets and generated private reports.
4. If GitHub reports a flagged, suspended, or restricted account, stop remote
   publishing. Do not switch owners, retry repeatedly, or bypass restrictions.
5. Once identity, availability, and checks are established, create the public
   repository and push the local main branch:

```bash
gh repo create lordarcane666/rag-evaluation-toolkit --public --description "An extensible open-source Python toolkit for evaluating RAG systems with deterministic metrics, LLM judges, and reproducible benchmarks."
git push -u origin main
gh repo edit lordarcane666/rag-evaluation-toolkit --add-topic python --add-topic rag --add-topic llm --add-topic evaluation --add-topic nlp --add-topic machine-learning --add-topic benchmarking --add-topic open-source
```

These are conditional instructions, not actions automatically run by the
toolkit. Inspect the rendered README and actual Actions results before tagging
a public release. No stars, download counts, contributors, benchmark scores,
or previous maintenance history should be invented for an application.
