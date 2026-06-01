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

## 2026-06-01 D1-172

- Suggestion: add a dev-only xAI OAuth smoke command that validates
  `LITELLM_XAI_OAUTH_AUTH_FILE`, redacts the credential payload, calls each
  enabled `oa_xai/*` model, and prints the matching `session_history` and
  provider-error metadata checks.
- How it would help: D1-172 could be closed with live evidence as soon as a
  LiteLLM-owned xAI OAuth credential is configured, without reassembling curl,
  docker, and database checks by hand.

## 2026-06-01 D1-173

- Suggestion: keep a small manifest of Grok Build fixture expectations next to
  the pass-through harness: required forwarded headers, required stripped
  headers, endpoint/body modes, expected session-history identity, and expected
  quota buckets.
- How it would help: this would reduce test churn and review time by making it
  obvious when a failing assertion is stale fixture metadata versus a routing or
  observability regression.
