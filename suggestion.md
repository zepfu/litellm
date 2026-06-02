## 2026-06-02 workspace cleanup

- Suggestion: add a lightweight scratch-file triage command that lists
  untracked root `repro*`, `test_*`, and package-local one-off files, checks
  whether each is imported or matched by pytest discovery, and emits a
  keep/discard/archive recommendation.
- How it would help: this would reduce time and risk when cleaning up after
  investigation work by separating real repo artifacts from local repro scripts
  before staging or deleting anything.

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

## 2026-06-01 investigate-codex-019e857b-9e88-7f53-90f9-acef3062b597

- Suggestion: when cross-repo subagent failure files are emitted into this repo,
  include a small machine-readable header with `source_repo`, `failure_class`,
  `read_only_requested`, `files_modified`, and `main_thread_reverted`.
- How it would help: this would reduce investigation time and token churn by
  letting the LiteLLM intake pass distinguish context-only external subagent
  telemetry from LiteLLM implementation defects without re-reading the full
  prompt summary.

## 2026-06-01 investigate-codex-019e8590-7116-7cf2-8e47-52b5b64edd33

- Suggestion: add a small intake-disposition helper that stamps start/end time,
  moves `investigate-*.md` into `.analysis/investigation/`, and appends a
  templated `.analysis/investigations.md` entry.
- How it would help: this would reduce timestamp and ledger churn for recurring
  investigation intake work, saving time and lowering the chance of incomplete
  disposition evidence.

## 2026-06-01 investigate-codex-019e8598-eda4-7401-b419-bbf3276f0f3a

- Suggestion: when `aawm-codex-agent-auto` returns terminal `completed=null`,
  have the spawning tool surface the last provider status/error class, even if
  no final answer exists.
- How it would help: this would reduce triage time and token churn by making it
  clear whether the next action should be a same-alias retry for 429/capacity,
  a smaller-model redispatch, or a local fallback for a deterministic null
  completion.

## 2026-06-01 D1-177

- Suggestion: add one reusable `oa_xai` smoke script that can run both chat and
  Responses-backed aliases, restarts or verifies the target dev proxy is using
  the current worktree code, then prints exact `/health`, model-cost mode,
  client response, `session_history`, and quota-observation evidence for the
  generated session IDs.
- How it would help: this would have made the multi-agent follow-up impossible
  to miss after D1-176, because the script would fail if a model reaches xAI on
  the wrong endpoint or if the live proxy is still serving a stale loaded module
  after code changes.
