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

## 2026-06-01 D1-174

- Suggestion: promote the gated `oa_xai/*` live smoke into a small reusable
  script that performs the request and prints the exact `session_history`,
  provider-error, and quota-observation rows for the generated session id.
- How it would help: this would make OAuth-managed xAI live verification a
  single repeatable command once credentials are configured, while keeping the
  default pytest harness secret-free and offline.

## 2026-06-01 D1-175

- Suggestion: add a small `scripts/disposition_investigate_codex.py` helper that
  inventories root `.analysis/investigate-codex-*.md` files, flags duplicate
  archived copies, emits a per-file disposition template, and optionally moves
  processed intake into `.analysis/investigation/`.
- How it would help: this would reduce time and token churn for large intake
  batches, prevent stale wrong-parent dispositions from surviving unnoticed, and
  make root-intake cleanup less error-prone when new investigation files appear
  mid-task.

## 2026-06-01 D1-176

- Suggestion: add one dev smoke command for `oa_xai/*` that migrates the Hermes
  xAI OAuth record to the LiteLLM-owned path, verifies the running dev proxy is
  using that path with a writable refresh mount, calls every enabled
  `oa_xai/*` chat model through `/v1/chat/completions`, and prints the matching
  `session_history` evidence with secrets redacted. The same command should
  enforce/verify `0600` permissions on the managed token file and query
  `session_history` by exact generated session IDs instead of broad JSON
  metadata predicates.
- How it would help: this would have caught the D1-172/D1-174 mock-only closure,
  the router-bypass bug, and the multi-agent chat-completions provider rejection
  in one repeatable gate before the item was marked complete. It would also
  reduce time/token churn from ad hoc slow DB probes and prevent a migrated
  OAuth credential from being left world-readable after a refresh.
