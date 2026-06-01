## 2026-06-01 D1-123

- Suggestion: add a provider-cache repair mode that accepts a predicate such as
  `--partial-cache-hits-only` and avoids the unconditional
  `session_history_tool_activity` aggregate join.
- How it would help: this would reduce wall-clock time and DB churn for
  targeted reporting repairs. The general repair script was correct but too
  broad for a 21k-row xAI-only cache fix, forcing a manual SQL repair path.

## 2026-06-01 D1-107

- Suggestion: add a gated Grok Build smoke harness that can launch the real
  Grok CLI against a selected LiteLLM base URL, run an embedding request with
  the same auth/client headers, and emit the exact `session_history` row checks.
- How it would help: this would reduce time and token churn by replacing ad hoc
  curl/psql stitching with one repeatable command that proves client identity,
  route metadata, cost, and embedding persistence against the intended DB.
