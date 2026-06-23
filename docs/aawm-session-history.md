# AAWM Session History Metadata

This fork stores AAWM-specific routing and observability details in
`aawm_tristore.public.session_history.metadata`. These keys are intended for
maintainer diagnostics and downstream reporting surfaces. They should not be
treated as public LiteLLM API guarantees.

## Inbound Alias Capture

`public.session_history` includes a nullable `inbound_model_alias` text column.
It stores the model request value exactly as LiteLLM received it before any alias
resolution.

- For AAWM alias requests, this is the inbound alias, for example
  `aawm-read`, `aawm-low`, or `aawm-code-anthropic`.
- For direct concrete requests, this is the concrete model string (and may equal
  `session_history.model`).

`session_history.model` remains the routed/selected concrete provider model and
must not be repurposed for inbound alias grouping.

Existing metadata fields `requested_model_alias` and `model_alias_label` remain
compatibility and trace metadata. Use `inbound_model_alias` as the canonical field
for reporting and grouping by requested alias.

Historical rows written before this field existed may be `NULL` unless they were
explicitly backfilled from prior metadata.

## Rate Limit And Billing Observations

`public.rate_limit_observations` stores provider quota, rate-limit, and billing
snapshots discovered during normal request logging and quota probes. The
canonical grouping fields are `provider`, `client`, `account_hash`, `model`,
`quota_key`, `quota_period`, `quota_type`, and `source`.

The normalized progress fields are:

- `expected_reset_at`: the provider reset or billing-period boundary when one
  is known.
- `remaining_pct`: bounded 0-100 remaining percentage used by dashboards and
  interval materializations.
- `quota_limit`, `quota_used`, and `quota_remaining`: provider-reported absolute
  amounts when the unit is known from `quota_type` or the provider payload.
- `billing_period_start_at` and `billing_period_end_at`: explicit billing-period
  boundaries when a provider reports them separately from short-window reset
  headers.

Provider-specific fields that do not fit a stable column live in
`raw_provider_fields` as sanitized JSONB. `evidence` records the bounded
signals, source fields, and interpretation notes that explain how the snapshot
was classified. These JSONB fields must not contain credentials, account ids,
authorization headers, prompt bodies, response text, or raw tool arguments.

Grok monthly billing payloads with absolute counters populate `quota_limit`,
`quota_used`, `quota_remaining`, both billing-period boundary columns, and a raw
copy of the sanitized `monthlyLimit`, `used`, `onDemandCap`, and period fields.
Newer Grok credit billing payloads may provide only `creditUsagePercent` and
`productUsage`; those snapshots populate `remaining_pct` and leave absolute
quota fields null rather than inventing counts. xAI OAuth rate-limit headers
populate absolute request/token amounts and carry billing period ends when the
provider config or managed subscription context exposes one.

## Session History Outage Spool

`session_history` rows are normally written directly to
`aawm_tristore.public.session_history`. If a batch flush keeps failing after the
configured retry budget, LiteLLM writes the exact batch payload to a local JSONL
spool for later replay instead of logging full tracebacks indefinitely.

The default spool directory is `/mnt/e/litellm/session_history`; override it with
`AAWM_SESSION_HISTORY_SPOOL_DIR` when a deployment needs a different local
volume. In dev compose, mount the host durable fallback path and set
`AAWM_SESSION_HISTORY_SPOOL_DIR=/mnt/e/litellm/session_history` explicitly so
outage replay survives container restarts. Production deployments must make the
same path host-backed in their owning compose/infrastructure repo; otherwise the
fallback file is only container-local and can be lost on recreate. Failed
flushes retry every `AAWM_SESSION_HISTORY_FAILED_FLUSH_RETRY_SECONDS` seconds,
and `AAWM_SESSION_HISTORY_FAILED_FLUSH_MAX_RETRIES` controls how many retry
attempts happen before the batch is atomically written to the spool.
Retryable asyncpg connection interruptions, such as `ConnectionDoesNotExistError`
or "connection was closed in the middle of operation" during pool acquire/init,
are treated as degraded persistence telemetry while the retry budget is still
active. LiteLLM drops the cached session-history pool before retrying those
failures, writes a temporary retry write-ahead spool file before the inline
retry, and marks the writer as `db_degraded_spooling` for a short bounded
window. While that degraded window is active, newly accepted records bypass the
normal in-memory-only enqueue path and are durably spooled for replay; the
enqueue path also drains a small batch of already queued records into the same
spool artifact when possible. This reduces the restart-loss window and avoids
creating one tiny overflow file per record during bursts. The degraded window
clears after a successful retry. If retries are exhausted, the write-ahead spool
file remains in place for replay. Recovered retryable disconnects should produce
warning/recovery telemetry with `retry_count`, `db_pool_reset`, `spooled`,
`spool_pending`, `spool_bytes`, `flush_recovered`, and a bounded failure
fingerprint rather than an active `*-error.jsonl` intake record.
`AAWM_SESSION_HISTORY_DEGRADED_SPOOL_SECONDS` controls the degraded enqueue
spooling window and defaults to 30 seconds.

If the synchronous flush helper is invoked from an already running event loop,
LiteLLM does not attempt to nest `run_until_complete`. It durably spools the
batch for replay, emits degraded telemetry with `spooled=true`, and leaves the
in-process drainer to recover the rows. Legacy or partially queued records may
also lack `provider_response_id`; payload construction keeps that column null
rather than failing the whole batch, while preserving any explicit provider
response id already present on the record or metadata.

Each `.jsonl` artifact contains one JSON object per line. The first line is a
metadata/header record with `type=metadata`, `format_version`, `spooled_at`,
`reason`, `retry_count`, and `record_count`. Each following line is a
`type=record` object containing one sanitized `session_history` record. Legacy
`.json` spool files from earlier releases are still loaded and drained when
present. Spool filenames include a UTC timestamp and a trace/session/call
identifier so operators can find the affected dataset without printing prompt,
tool, or metadata payloads into logs. Writes use a same-directory temporary file
followed by replacement, so completed `.jsonl` files are replayable and partial
writes do not look like ready datasets.

The in-process drainer checks for existing spool files when the callback starts
and after successful writes, so records left behind by a prior outage can replay
without waiting for a new failed batch. If the spool directory exists but
cannot be listed temporarily, startup and replay summaries report
`spool_pending=unknown` instead of `spool_pending=0`; future drainer triggers
retry the listing rather than treating the backlog as empty. Missing directories
still report `spool_pending=0`. Spool-write warnings include
`spool_write_succeeded=true`, the artifact path, record count, byte count, queue
depth, pending-file count, and oldest pending age when available. Queue-full
warnings include a terminal `queue_disposition` such as `queued_after_wait`,
`overflow_flush_started`, `spool_write_started`, `db_degraded_spooling`, or
`spool_write_failed`. Replay logs include `spool_replay_started`,
`spool_replay_recovered`, `spool_replay_failed`, attempted-file count,
removed-file count, replay duration, and the remaining spool summary so
operators do not need to infer whether `/mnt/e/litellm/session_history` was
actually used. If replay fails, the drainer stays active and retries the spool
batch with bounded backoff before leaving the files in place for a future
trigger. `AAWM_SESSION_HISTORY_SPOOL_REPLAY_BACKOFF_SECONDS` accepts a
comma-separated retry schedule and defaults to `5,15,30,60,120`. Replay is
at-least-once. The database insert path remains idempotent, so recovery tools
and the in-process drainer should assume duplicates are possible after a crash
or partial outage. The spool contains local sensitive session metadata and must
stay out of git, shared logs, and external artifacts unless it has been reviewed
and sanitized.

## Durable Alias Routing State (D1-323)

AAWM auto-agent alias selectors keep two kinds of process-local routing state:

- session affinity: pins a calling session to a selected provider/model/route family
- candidate cooldown: blocks retryable exhausted candidates for a bounded TTL

LiteLLM now mirrors that state into the proxy Redis-backed `DualCache` so
affinity and cooldown survive process restarts and can be shared across LiteLLM
workers that use the same routing namespace. When Redis is not configured or is
unavailable, selectors keep the existing process-local fallback and expose that
state source in metadata rather than labeling it durable.

Operational notes:

- Namespace env: `AAWM_ALIAS_ROUTING_STATE_NAMESPACE` (default: `aawm-routing-v1`).
  Prod and dev instances that should share routing state must use the same value.
- Durable key shape:
  `aawm:alias-routing:{namespace}:{family}:{kind}:{state_key}` where `family` is
  `codex` or `anthropic`, and `kind` is `affinity` or `cooldown`.
- Durable payloads store absolute wall-clock expiry (`expires_at_epoch` via
  `time.time()`). Process-local maps still use monotonic expiry for the fast path.
- Read order: memory first, then durable cache hydrate; if durable cache is absent
  or unavailable, selectors keep the existing in-memory fallback without failing
  the request.
- Session-history / alias metadata may include lightweight source fields such as
  `codex_auto_agent_affinity_state_source`, `codex_auto_agent_cooldown_state_source`,
  `anthropic_auto_agent_affinity_state_source`, and
  `anthropic_auto_agent_cooldown_state_source` with values `memory`,
  `durable_cache`, or `local_fallback`. These fields do not store prompts, tools,
  credentials, or raw session payloads.

## Alias Routing Audit Metadata

AAWM auto-agent aliases attach bounded routing metadata to selected requests so
operators can confirm ordered failover without replaying raw transcripts.

Common keys include:

- `requested_model_alias` / `model_alias_label`: the inbound alias, for example
  `aawm-code` or `aawm-code-anthropic`.
- `aawm_alias_routing_audit_events`: ordered events for skipped, failed, and
  selected candidates.
- `codex_auto_agent_attempts` or `anthropic_auto_agent_attempts`: candidate
  attempts made by the selected alias handler.
- `codex_auto_agent_skipped_candidates` or
  `anthropic_auto_agent_skipped_candidates`: candidates skipped because of
  cooldown or stateful session-affinity cooldown.

For retryable provider errors, the handler records a
`candidate_retryable_failure` event, cools down that candidate, and selects the
next configured usable candidate. For tool-bearing or stateful
`aawm-code-anthropic` requests, every declared candidate route is treated as a
Claude Code tool-contract route: if the alias declares OpenAI, xAI,
native Anthropic, or another provider/model target, that target must preserve
tool calls, tool-use ids, tool arguments, tool-result replay, and ordered
failover metadata for engineering-agent traffic. A declared candidate must not
be skipped as policy-incompatible for the same request class it is meant to
serve. If a declared candidate mishandles tools, that is an adapter translation
defect to fix or evidence for removing/reclassifying the candidate; it is not a
selector-side compatibility decision.

## Anthropic Adapter Usage Visibility

Anthropic-compatible adapter streams can expose client-visible usage estimates
for Claude Code without changing provider-truth reporting. When a non-Anthropic
Responses adapter has enough request context, the streamed `message_start`
event may include an estimated Anthropic-shaped `usage.input_tokens` value so
Claude Code can show activity before the terminal provider usage arrives.

The terminal `message_delta.usage` remains provider-reported when upstream
usage is available, and fallback usage remains marked in
`litellm_metadata.anthropic_adapter_client_visible_usage_source`. Estimated
message-start usage is marked separately with
`anthropic_adapter_message_start_usage_source=estimated` and
`anthropic_adapter_message_start_usage_estimated=true`. These client-visible
fields must not be treated as canonical provider token or cost truth for
`session_history`; canonical token and cost columns continue to come from the
normal upstream usage and cost calculation path.

## AAWM Alias Candidate Orders (D1-363)

`aawm-low`, `aawm-low-anthropic`, `aawm-code`, and `aawm-code-anthropic` do
not prepend or select Antigravity-backed candidates during normal alias
selection. These aliases follow their declared non-Antigravity failover order
below. Direct explicit Antigravity routes remain available separately and are
documented in the Antigravity OAuth Credentials section.

`aawm-low` and `aawm-low-anthropic` use this order:

1. OpenRouter North Mini (`openrouter/cohere/north-mini-code:free`)
2. OpenRouter Owl Alpha (`openrouter/owl-alpha`)
3. OpenCode Zen `deepseek-v4-flash`
4. OpenCode Zen `big-pickle`
5. `gpt-5.4-mini` as the OpenAI last-resort candidate for `aawm-low`
6. native Anthropic Haiku as the last-resort candidate for `aawm-low-anthropic`

`aawm-code` uses this order:

1. `gpt-5.3-codex-spark`
2. `grok-composer-2.5-fast`
3. `oa_xai/grok-build`
4. `gpt-5.5` as the OpenAI last-resort candidate with medium reasoning

`aawm-code-anthropic` uses this order:

1. `gpt-5.3-codex-spark`
2. `grok-composer-2.5-fast`
3. `oa_xai/grok-build`
4. native Anthropic `claude-sonnet-4-6` as the last-resort candidate

This is default alias behavior; no staging environment variable is required for
dev or production routing.

Alias-visible metadata (for example, route audit fields and session-history
model keys) must keep the provider-prefixed names (`openrouter/...`) so
operators can distinguish the declared OpenRouter candidates. The OpenRouter
completion adapter must continue to send the provider-stripped model slug
upstream (`cohere/north-mini-code:free`, `owl-alpha`) to match OpenRouter
ingress expectations.

The Codex `aawm-code` alias uses `gpt-5.5` as the OpenAI last-resort candidate,
not plain `gpt-5.3-codex`, because ChatGPT-account Codex passthrough rejects the
plain `gpt-5.3-codex` model. When that last-resort candidate is selected,
LiteLLM applies medium reasoning by default if the request did not already set a
reasoning effort. Stateful continuation requests keep the established candidate
through session affinity, including last-resort `gpt-5.5`, until that candidate
itself cools down or the session is redispatched.

## Grok Native OIDC Credentials

`xai/grok-composer-2.5-fast`, `xai/grok-build`, and
`xai/grok-build-0.1` use the Grok native OIDC credential path, not the managed
`oa_xai/*` OAuth credential file. `LITELLM_XAI_GROK_AUTH_FILE` should point
directly at the Grok CLI credential, normally `/home/zepfu/.grok/auth.json`.

LiteLLM is a read-only consumer of that file. It selects a valid access token
from the configured Grok OIDC credential and fails the Grok native candidate as
`candidate_unavailable` when the file is missing, lacks an access token, or is
expired/near expiry. LiteLLM must not copy, seed, refresh, atomically replace,
or otherwise mutate the Grok CLI credential during model requests.

The provider-status sidecar is the scheduled writer. It mounts
`/home/zepfu/.grok` writable, takes a file lock, refreshes the credential on the
configured cadence, and writes the updated JSON atomically with file mode
`0600`. The atomic write preserves existing file ownership/private mode unless
`AAWM_GROK_OIDC_AUTH_FILE_UID`, `AAWM_GROK_OIDC_AUTH_FILE_GID`, or
`AAWM_GROK_OIDC_AUTH_FILE_MODE` are set; group/other-readable or writable modes
are clamped back to `0600`. Dev compose sets those ownership
defaults so a prior container-owned `nobody:nogroup` credential is corrected on
the next sidecar cycle without giving LiteLLM write access. This metadata-only
repair runs every provider-status sidecar cycle when Grok OIDC refresh is
enabled; token refresh still follows the configured refresh interval. In dev
compose the sidecar runs with
`AAWM_GROK_OIDC_REFRESH_ENABLED=1`,
`AAWM_GROK_OIDC_AUTH_FILE=/home/zepfu/.grok/auth.json`, and a one-hour refresh
interval. The dev LiteLLM container mounts the same host directory read-only.

Each Grok OIDC refresh attempt writes sanitized provider-auth telemetry when
the sidecar runs with DB writes enabled. `provider_auth_observations` is the
append-only event table, and `provider_auth_current` is the latest-state view
keyed by environment, provider, auth family, credential scope, and auth-file
identity hash. Rows include refresh status, attempted/refreshed/skipped flags,
credential expiry, last successful validation time, source sidecar task, and
redacted failure class/message. They must never include access tokens, refresh
tokens, raw auth-file contents, or the raw auth-file path. Dashboard Provider
Health should use `provider_auth_current` for current expiry and stale/failed
auth display, then drill into `provider_auth_observations` for history.

Keep the Grok CLI/OIDC credential separate from the managed `oa_xai/*`
`oauth-auth.json` file. The explicit Grok billing poll should run from the same
sidecar context so quota snapshots use the credential that the sidecar keeps
fresh, not a second LiteLLM-managed copy.

## Managed xAI OAuth Credentials

Managed `oa_xai/*` routes read `LITELLM_XAI_OAUTH_AUTH_FILE`, normally
`/home/zepfu/.litellm/xai/oauth-auth.json`. LiteLLM is a read-only consumer of
that file. During request handling it selects a valid scoped credential and
fails with clear sidecar-refresh wording when the credential is missing, lacks
an access token, or is expired/near expiry. LiteLLM must not refresh or write
this managed credential during model requests.

The provider-status sidecar is the scheduled managed xAI OAuth writer. It
mounts `/home/zepfu/.litellm/xai` writable, locks the configured auth file,
refreshes using the managed credential's refresh token and OIDC client id, and
writes updated access token, refresh token, ID token, token type, and expiry
fields atomically. The dev LiteLLM container mounts the same host directory
read-only. In dev compose the sidecar runs with
`AAWM_XAI_OAUTH_REFRESH_ENABLED=1`,
`AAWM_XAI_OAUTH_AUTH_FILE=/home/zepfu/.litellm/xai/oauth-auth.json`, and a
one-hour refresh interval.

Each managed xAI OAuth refresh attempt writes sanitized provider-auth telemetry
into `provider_auth_observations` and `provider_auth_current`. Rows use provider
`xai`, auth family `xai_oauth`, source task `xai_oauth_refresh`, the credential
scope, the auth-file identity hash, attempted/refreshed/skipped flags, expiry,
last successful validation time, and redacted failure class/message. Rows must
never include access tokens, refresh tokens, raw auth-file contents, or the raw
auth-file path.

## Antigravity OAuth Credentials

Antigravity Code Assist routes use sidecar-managed OAuth token data. In prod,
LiteLLM should be configured with
`LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE` for the provider-status sidecar's
refreshed token copy and `LITELLM_ANTIGRAVITY_SEED_AUTH_FILE` for the
read-only Antigravity CLI login seed, normally
`/home/zepfu/.gemini/antigravity-cli/antigravity-oauth-token`. LiteLLM is a
read-only consumer of those files. During request handling it loads the first
valid candidate token and never attempts a direct OAuth refresh or invokes
`agy`. If all candidate tokens are expired or invalid, LiteLLM fails the
Antigravity candidate with a clear sidecar-refresh message.

For AAWM auto-agent aliases, stale or missing Antigravity token data is treated
as provider-auth degradation during candidate selection. The selector records the
Antigravity candidate under the `antigravity:auth_degraded` lane, marks it
skipped with `reason=auth_degraded`, applies a short candidate cooldown, and
continues to the next declared candidate. This expected degraded state logs a
bounded warning without traceback; unexpected Antigravity lane-resolution
exceptions still emit traceback-bearing error logs for intake.

The provider-status sidecar is the scheduled Antigravity writer. It mounts the
managed token directory writable, locks the configured token file, attempts
direct OAuth refresh using configured/token/CLI-extracted client values, and
falls back to `agy models` for Antigravity CLI silent refresh when the direct
client pair is rejected. It also needs access to the seed Antigravity CLI
directory and `agy` binary when CLI fallback is expected. In dev compose the
sidecar runs with `AAWM_ANTIGRAVITY_OAUTH_REFRESH_ENABLED=1`,
`AAWM_ANTIGRAVITY_AUTH_FILE=/home/zepfu/.gemini/antigravity-cli/antigravity-oauth-token`,
and a one-hour refresh interval; prod may instead point
`AAWM_ANTIGRAVITY_AUTH_FILE` at the managed auth copy while LiteLLM also has
`LITELLM_ANTIGRAVITY_SEED_AUTH_FILE` for read-only fallback.

Each Antigravity refresh attempt writes sanitized provider-auth telemetry into
the same `provider_auth_observations` table and `provider_auth_current` view
used by Grok OIDC. Rows use provider `antigravity`, auth family
`antigravity_oauth`, source task `antigravity_oauth_refresh`, the auth-file
identity hash, attempted/refreshed/skipped flags, expiry, last successful
validation time, and redacted failure class/message. Rows must never include
access tokens, refresh tokens, raw auth-file contents, or the raw auth-file
path.

## Codex OAuth Credentials

Codex/OpenAI adapter routes that use ChatGPT-account OAuth read the Codex auth
JSON file, normally `/home/zepfu/.codex/auth.json`, through
`LITELLM_CODEX_AUTH_FILE`. LiteLLM is a read-only consumer of that file. During
request handling it loads headers only when the access token is still valid. If
the token is expired or invalid, LiteLLM fails with a clear sidecar-refresh
message instead of attempting OAuth refresh or writing the auth JSON.

The provider-status sidecar is the scheduled Codex OAuth writer. It mounts
`/home/zepfu/.codex` writable, locks the configured auth file, refreshes with
the Codex OAuth refresh token, and writes the updated access token, refresh
token, optional ID token, account ID, expiry, and top-level `last_refresh`
timestamp atomically. The dev LiteLLM container mounts the same host directory
read-only. In dev compose the sidecar runs with
`AAWM_CODEX_OAUTH_REFRESH_ENABLED=1`,
`AAWM_CODEX_AUTH_FILE=/home/zepfu/.codex/auth.json`, host-user ownership
defaults `AAWM_CODEX_AUTH_FILE_UID=1000` and
`AAWM_CODEX_AUTH_FILE_GID=1000`, private mode
`AAWM_CODEX_AUTH_FILE_MODE=0o600`, and a one-hour refresh interval. The
refresh path repairs both `auth.json` and `auth.json.lock` metadata while it
holds the lock, including skipped refresh cycles, so root-owned or container
UID-owned files do not prevent Codex from launching.

Each Codex OAuth refresh attempt writes sanitized provider-auth telemetry into
the same `provider_auth_observations` table and `provider_auth_current` view.
Rows use provider `openai`, auth family `codex_oauth`, source task
`codex_oauth_refresh`, credential scope set to the sanitized ChatGPT account ID
when known, the auth-file identity hash, attempted/refreshed/skipped flags,
expiry, last successful validation time, and redacted failure class/message.
Rows must never include access tokens, refresh tokens, raw auth-file contents,
or the raw auth-file path.

Grok native and `oa_xai/*` Responses candidates remove request fields, hosted
tools, and `reasoning` input items that the selected Grok-family model declares
unsupported. This includes `reasoning` items that carry `encrypted_content` from
another provider's compacted Responses state; forwarding those blobs to Grok can
trigger provider errors such as `Could not decode the compaction blob`. Ordinary
non-reasoning continuation items, including `function_call` and
`function_call_output`, remain in the outbound request. If a Grok-family
upstream still rejects a compacted request with `Could not decode the compaction
blob`, alias-probe mode classifies that 400 as candidate-unavailable so the
declared failover sequence can continue instead of stranding the worker on the
rejecting candidate.

## Access Log Display Semantics

AAWM route logs emit a compact route line to the LiteLLM proxy logger after the
incoming request has enough local context to identify the route and, when
available, the egress target. Pass-through requests and general proxy requests
such as embeddings use the same formatter and native access-log suppression
path.
The display shape is:

```text
YYYYMMDD HH:MM:SS [ROUTE] [client-name/version] - [agent-name[#agent-id][@repository][.model(alias)]] METHOD ip:port incoming -> outgoing
```

The timestamp uses a 24-hour clock. Known TUI client products are normalized for
display, for example `claude-cli/*` and `claude-code/*` become `Claude/*`,
`codex-cli/*` and `codex-tui/*` become `Codex/*`, and `grok-*/*` becomes
`Grok/*`. Client name/version, agent, agent id, repository, model alias, and
client address segments are omitted when that metadata is unavailable. The
selected model is printed as `model(alias)` only when the inbound alias differs
from the selected model. When owner metadata is present, the route context is
composed as `agent#id@repository.model(alias)`; missing subfields are omitted
without inventing values.

These lines are for live container-log triage, not durable reporting. Durable
model and alias attribution still comes from `session_history.model` and
`session_history.inbound_model_alias`. Route-log identity fields are conservative
display tokens derived from normalized metadata or explicit headers such as
`x-aawm-client-name`, `x-aawm-client-version`, `user-agent`,
`x-aawm-agent-name`, and `x-aawm-repository`. Repository display can also come
from the same structured identity aliases used by `session_history`, including
`repo`, `workspace_root`, `project_root`, `working_directory`, `cwd_path`,
`cwd_uri`, and `aawm_claude_project`. LiteLLM omits prompt-like, sentence-like,
or punctuation-heavy identity values instead of printing raw request text.
LiteLLM intentionally does not inspect prompt text or raw tool arguments for the
route log.

Route logs must not include API keys, authorization headers, full request or
response bodies, prompt content, tool arguments, or arbitrary query strings.
Incoming endpoints preserve only known-safe routing query markers, and outgoing
targets are logged as host plus path.

For requests that emit this enriched `[ROUTE]` line, LiteLLM suppresses the
matching native Uvicorn/Gunicorn access record for that request. Unrelated
routes should continue to use the normal server access log. Successful
health-check access records for `/health`, `/health/readiness`,
`/health/liveliness`, and `/health/services` are also suppressed because they do
not carry model-routing context; failed health checks remain visible through the
native access log. Repeated enriched route lines with the same route type,
client product, owner/model context, method, incoming endpoint, and outgoing
target are coalesced for a short window so tight loops do not dominate Docker
logs. The matching native access record is still suppressed for each coalesced
request. Set `AAWM_ROUTE_LOG_DEDUP_WINDOW_SECONDS=0` to disable coalescing, or
increase the value for targeted noisy-window capture reduction.

When `AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS` is greater than zero (default `60`),
healthy per-request enriched `[ROUTE]` lines are not emitted immediately.
Instead, completed requests are grouped into periodic rollup blocks in the
process-local proxy logger. Set `AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS=0` to
restore immediate per-request `[ROUTE]` lines for debug windows. Rollup headers
use:

```text
YYYYMMDD HH:MM:SS [EARLY] repo@Client[version] /incoming -> outgoing
```

`[EARLY]` appears only when a bounded in-memory cap forces a flush before the
interval elapses. Each rollup subline uses ` - model(alias) - Turns: N` with an
optional trailing status tag (`[Degraded]`, `[Cooling Down]`, `[Failed]`, or
`[Exhausted]`). `Turns` counts completed requests only. Alias-route candidate
events that degrade, cool down, fail, or exhaust before a successful completion
emit an immediate compact status line:

```text
YYYYMMDD HH:MM:SS - <alias>: <model> Status: <status> - Message: <details>
```

Those events also contribute zero-turn rollup sublines so multiple failed
candidates remain visible in the same bucket. Rollups flush on the configured
interval and at process shutdown via a shutdown-safe flush helper.

AAWM alias routing audit events are still attached to request metadata for
session-history and diagnostic consumers. Container log emission is narrower:
failures, cooldowns, redispatches, no-candidate outcomes, and explicit warning
events remain logged, while healthy selected/session-affinity continuation
events are skipped unless `AAWM_ALIAS_ROUTE_LOG_HEALTHY=1` is set for a targeted
debug window.

## Langfuse Event Size Fitting

Before enqueueing a Langfuse generation, LiteLLM estimates the serialized event
size and tries to fit the whole event below the configured Langfuse event limit.
The fitter preserves the existing input truncation behavior for prompts, then
continues through oversized `output`, `metadata`, `model_parameters`,
`status_message`, `prompt`, and other non-core generation fields until the full
event fits or all safe fields have been reduced.

Structured metadata and model-parameter fields are replaced with compact
omission markers instead of partial raw values. Successful fitting that leaves
the event safely below the warning threshold is debug-only telemetry. Warnings
are reserved for raw near-limit events, events that remain near the warning
threshold after fitting, or events that still cannot fit after all safe fields
have been reduced. Any emitted size summary reports only identifiers, field
names, byte counts, omission counts, and whether fitting still failed; it must
not include prompt bodies, response text, tool arguments, credentials, or
oversized raw metadata values.

## Tool Definition Snapshots

Pass-through requests can advertise large tool definitions. LiteLLM records a
compact per-generation reference in `session_history.metadata` and stores the
full sanitized snapshot once per session/hash in
`aawm_tristore.public.session_history_tool_definition_snapshots`.

Compact metadata may include:

- `aawm_tool_definition_capture_version`: capture contract version, currently
  `v1`.
- `aawm_tool_definition_capture_source`: source of the captured definitions,
  currently `passthrough_request_body`.
- `aawm_tool_definition_count`: total advertised tool/function definitions
  seen on the request.
- `aawm_tool_definition_captured_count`: number of sanitized definitions
  captured in the bounded snapshot.
- `aawm_tool_definition_sources`: request fields that contributed definitions,
  such as `["tools"]` or `["functions"]`.
- `aawm_tool_definition_names`: bounded list of advertised tool/function names.
- `aawm_tool_definition_types`: bounded list of advertised tool/function types.
- `aawm_tool_definition_snapshot_hash`: SHA-256 hash of the sanitized snapshot.
- `aawm_tool_definition_snapshot_truncated`: `true` when the captured snapshot
  was bounded or any captured definition was truncated/redacted.
- `aawm_tool_definition_snapshot_storage`: durable lookup table name,
  `session_history_tool_definition_snapshots`.
- `aawm_tool_definition_snapshot_storage_key`: currently
  `session_id,aawm_tool_definition_snapshot_hash`.

The full snapshot is intentionally not stored in every
`session_history.metadata` row and is stripped from Langfuse generation
metadata before SDK enqueue. Consumers that need the full sanitized definition
payload should join:

```sql
SELECT s.sanitized_snapshot
FROM public.session_history h
JOIN public.session_history_tool_definition_snapshots s
  ON s.session_id = h.session_id
 AND s.snapshot_hash = h.metadata->>'aawm_tool_definition_snapshot_hash'
WHERE h.session_id = $1
  AND h.metadata ? 'aawm_tool_definition_snapshot_hash';
```

The durable table stores sanitized/redacted definitions only. It should be used
for drill-down and attribution evidence, not as proof that any tool was called.

Langfuse-only historical backfills cannot reconstruct a full snapshot once
generation metadata has been compacted. They preserve the compact hash/reference
fields when present, but the durable table is populated only by runtime
session-history ingestion or by older Langfuse rows that still carried the
inline `aawm_tool_definition_snapshot` value.

## Langfuse Metadata Compaction

Langfuse generation metadata is intentionally more compact than
`session_history.metadata`. Session history remains the operational drill-down
surface, while Langfuse keeps bounded summaries for high-cardinality fields that
otherwise recur on every generation event.

Before Langfuse SDK enqueue, LiteLLM strips
`aawm_tool_definition_snapshot` and compacts these metadata fields:

- `prompt_overhead_component_paths`: replaced with a stable hash, total count,
  bucket counts, original byte size, and bounded path samples by bucket.
- `prompt_overhead_excluded_component_paths`: replaced with a stable hash,
  total count, original byte size, and bounded path samples.
- `codex_response_headers`: replaced with a stable hash, source, header count,
  bounded header-name samples, rate-limit header names, and request-id
  presence. Header values are not kept inline in Langfuse metadata.
- `responses_stream_tool_state`: replaced with a stable hash, tool-call count,
  tool names, tool type counts, and bounded per-tool samples containing
  `type`, `name`, `call_id`, `arguments_hash`, and `arguments_size_bytes`.
  Raw tool arguments are not kept inline.
- `claude_tool_advertisement_compaction_events`: replaced with a stable hash,
  event count, tool names, statuses, Claude Code versions, and aggregate
  original/compacted/saved character counts. This does not re-expand or
  double-count the already-compacted Claude tool definitions.

These summaries reduce metadata size and avoid repeating raw headers, path
inventories, tool arguments, or compaction audit detail inside each Langfuse
event. They do not eliminate Langfuse size warnings where the request `input`
field itself is already close to the configured Langfuse event-size threshold.

The Langfuse payload-size regression tests include a post-D1-314 guardrail
fixture that classifies event pressure into `already_handled`,
`remaining_candidate`, and `no_op` buckets via
`_build_langfuse_compaction_savings_audit()`. That helper is bounded diagnostic
telemetry only: each entry reports `family`, `field`, `original_size_bytes`,
`final_size_bytes`, `saved_bytes`, `saved_ratio`, `mode`, `strategy`, and
`classification` without raw prompts, tool arguments, header values, or local
file content.

Read the audit as follows:

- `already_handled`: D1-238/D1-314 metadata compactors already reduced the field.
  These savings are expected and should not be reopened as new compaction TODOs.
- `remaining_candidate`: oversized generation fields still reduced by the
  event-size fitter, usually `input` after metadata compaction.
- `no_op`: unchanged or non-candidate fields with no measured savings.

When a Langfuse payload-size audit log is emitted, the JSON summary may include
`compaction_savings_audit` with `classification_counts`, `totals`, and bounded
`entries`. Use it to tell whether a warning is input-dominated or a metadata
regression, not to recover raw request content.

## Diagnostic Payload Capture Boundary

Opt-in pass-through diagnostic payload capture writes local JSON artifacts under
`/tmp/captures/diagnostic_payloads` by default. It is intentionally outside
`session_history`, provider-error observations, Langfuse metadata, and runtime
error JSONL intake.

The capture manifest records safe routing context, endpoint templates, byte
counts, hashes, and omitted-field descriptions for scoped investigations. Raw
headers, request bodies, response bodies, stream chunks, prompts, tool
arguments, OAuth tokens, API keys, cookies, concrete session ids, and local file
content must not be copied into `session_history.metadata`.

Google Code Assist / Antigravity bootstrap preflight calls in
`litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` already use
that same direct diagnostic capture path for `v1internal:loadCodeAssist` and the
prime preflight endpoints. Those artifacts are exact-scope gated, local-only,
and shape/hash-only unless the separate full-payload capture opt-in is enabled.
They must not be treated as `session_history` rows or copied into
`session_history.metadata`.

Native `/rerank` diagnostic manifests use route family `rerank` and endpoint
template `/rerank`. They are local scoped artifacts only; rerank query text,
input documents, returned document text, raw headers, and raw bodies must not be
copied into `session_history.metadata`.

AssemblyAI transcript polling and Vertex AI Live websocket diagnostics require
route-specific manifests before they can be durable enough for investigation.
Do not copy polling transcript text, websocket messages, or raw websocket
lifecycle payloads into `session_history.metadata`; only store explicit
shape/hash summaries after those route-specific capture surfaces exist.

## Codex Tool-Description Patches

AAWM Codex and Claude Code adapter paths may patch advertised tool descriptions
before provider egress when the route needs guidance that survives provider
transforms which can drop top-level instructions or metadata. These patches are
request-shape guidance only: they do not prove that a model called a tool
correctly.

Rows and traces affected by this path include the request tag
`codex-tool-description-patch` plus one tag per applied patch, for example
`codex-tool-description-patch:spawn-agent-fanout-policy` or
`codex-tool-description-patch:core-tool-guidance-edit`. Metadata may include:

- `codex_tool_description_patch_count`: number of applied tool-description
  patch events.
- `codex_tool_description_patch_replacement_count`: number of text replacements
  made by replacement-style patches.
- `codex_tool_description_patch_ids`: stable patch identifiers applied to the
  request.
- `codex_tool_description_patch_events`: bounded per-tool patch records with
  tool name, path, and patch id.

Core-tool guidance patches currently target Claude Code/Codex tools such as
`Bash`, `Edit`, `Read`, and `Write`. The intent is to make declared fallback
models, including xAI/Grok routes, receive the same operational cautions about
structured edits, stale `old_string` retries, bounded reads, and reading before
overwriting existing files even when the provider adapter removes unsupported
top-level fields.

## xAI Responses Sanitization

LiteLLM sanitizes xAI Responses request bodies for Codex/OpenAI passthrough and
native Grok Composer passthrough before forwarding to xAI. The sanitizer removes
unsupported top-level Responses fields and unsupported tool fields while
preserving bounded metadata that explains the adaptation. Native Grok and
managed xAI Responses requests also drop `tool_choice` when the outgoing payload
has no usable `tools` definitions, since xAI rejects `tool_choice` without
tools.

If the request carries a LiteLLM-encoded Responses `previous_response_id`, the
xAI/Grok sanitizer decodes it back to the original upstream response id before
egress. The encoded id is only for LiteLLM deployment affinity; xAI and Grok
must receive the raw upstream id, otherwise compacted continuations can fail
with provider errors such as `Could not decode the compaction blob`.

For AAWM Codex aliases, hosted-tool support is evaluated again after the alias
has selected a concrete xAI/Grok candidate such as `grok-composer-2.5-fast` or
`oa_xai/grok-build`. This catches provider-invalid Codex tool variants, including
`custom`, that could not be classified while the inbound request model was still
the abstract alias `aawm-code`.

OpenRouter completion-adapter candidates classify provider-wrapped 400 responses
with `metadata.provider_name` and `metadata.raw=ERROR` as terminal candidate
failures. Alias probes cool down only that OpenRouter candidate, record
`OPENROUTER_PROVIDER_RAW_ERROR` in attempt metadata, and continue to the next
declared candidate rather than surfacing an ASGI traceback to the client.

Grok Composer and Grok Build candidates also drop unsupported reasoning-effort
fields before egress, including `reasoning`, `reasoning_effort`,
`reasoningEffort`, and nested `output_config.effort`. If xAI still rejects an
alias-probe request with an unsupported reasoning-parameter 400, LiteLLM treats
that target as candidate-unavailable so the declared alias sequence can continue
to the next candidate instead of terminating the agent dispatch on the provider
400.

Grok Composer and Grok Build candidates also remove unsupported Responses
`reasoning` input items before egress, including encrypted compaction items from
another provider. The sanitizer records only bounded metadata about the removal,
for example the input index and whether `encrypted_content` was present; it does
not record the encrypted blob value. Non-reasoning continuation items stay in the
outbound `input` array so function-call state can continue without sending
provider-invalid compaction state.

Rows affected by this path may include:

- `codex_unsupported_hosted_tool_removed_count`,
  `codex_unsupported_hosted_tool_types_removed`, and
  `codex_unsupported_hosted_tools_removed`: Codex hosted-tool definitions removed
  because the selected concrete model marks them unsupported. For xAI/Grok
  Responses routes this commonly includes `custom`, `image_generation`,
  `namespace`, or `tool_search`.
- `codex_unsupported_hosted_tool_choice_removed`: the removed `tool_choice` when
  it referenced a hosted tool removed by the selected model policy.
- `codex_unsupported_input_item_removed_count`,
  `codex_unsupported_input_item_types_removed`, and
  `codex_unsupported_input_items_removed`: unsupported Responses input items
  removed before egress. For Grok-family routes this includes `reasoning` input
  items and records `encrypted_content=true` when an encrypted compaction blob
  was removed without logging the blob.
- `xai_responses_request_sanitized`: `true` when the outgoing request body was
  changed before xAI egress.
- `xai_responses_previous_response_id_decoded`: `true` when a LiteLLM-encoded
  `previous_response_id` was decoded back to the upstream response id before
  egress.
- `xai_responses_sanitized_removed_params`: normalized top-level field names
  removed from the request, for example `["instructions", "metadata"]`.
- `xai_responses_sanitized_tool_count`: number of tool definitions whose
  outbound shape changed.
- `xai_responses_sanitized_tool_types`: normalized tool types whose outbound
  shape changed, for example `["code_interpreter", "web_search", "x_search"]`.
- `xai_responses_sanitized_tools`: bounded detail records keyed by tool index,
  type, and removed fields where available.
- `xai_tool_choice_without_tools_removed`: the removed `tool_choice` value when
  no typed tool definitions were present in the outgoing payload.
- `xai_tool_choice_without_tools_removed_reason`: currently `missing_tools`
  when `tool_choice` was removed because no usable tools survived.

Related route metadata distinguishes Codex and native Grok traffic:

- Codex/OpenAI passthrough Responses traffic uses
  `openai_passthrough_route_family=openai_responses` and
  `passthrough_route_family=codex_responses`.
- Native Grok Composer passthrough traffic uses
  `openai_passthrough_route_family=openai_responses` and
  `passthrough_route_family=grok_cli_chat_proxy`.
- Anthropic Grok native Responses adapter traffic preserves the selected
  adapter route family, for example
  `passthrough_route_family=anthropic_grok_native_responses_adapter`, while
  retaining Grok transport/auth/quota metadata such as
  `grok_cli_chat_proxy=true`, `grok_cli_chat_proxy_used=true`, and
  `route:grok_cli_chat_proxy`.
- Managed `oa_xai/*` traffic should still be reported as provider `xai` while
  preserving the public `oa_xai/*` model/model_group label.

Native Grok passthrough session identity prefers an explicit
`x-grok-session-id` header. If that header is absent, LiteLLM uses the native
`x-grok-conv-id` header as the persisted `session_id` so usage-bearing Grok TUI
rows remain reportable under a stable conversation identifier.

Native Grok passthrough model attribution prefers `x-grok-model-override`.
When that header is absent but the JSON request body contains a supported
native Grok model such as `grok-composer-2.5-fast`, LiteLLM promotes that body
model into passthrough metadata and forwards it as `x-grok-model-override`.
Zero-token Grok side-channel rows that do not carry a per-request model remain
excluded from usage reporting, but should still persist a non-`unknown`
`model`/`model_group`; if no stronger model evidence exists, they are attributed
to the generic Grok TUI client model `grok-build`.

Native Grok storage uploads are a local-only side channel. LiteLLM authenticates
`/grok/*/storage` requests with the normal LiteLLM key path, records no raw
storage body, does not forward those uploads to `cli-chat-proxy.grok.com`, and
returns a benign local success response. These storage artifacts can contain
session replay material, terminal output, tool calls/results, local config, and
compressed session state, so they must not enter pass-through logs, Langfuse,
session history, or upstream xAI storage.

Native Grok coding-data-retention probes are also local-only. LiteLLM
authenticates `/grok/*/privacy/coding-data-retention` requests with the normal
LiteLLM key path, clears any parsed request body before auth/logging hooks, does
not forward the probe to `cli-chat-proxy.grok.com`, and returns a benign local
success response. This avoids upstream 422 noise for empty probes while keeping
privacy preference payloads out of logs, Langfuse, session history, and upstream
storage unless a future route explicitly implements a validated translation.

Grok session mutation side channels such as `/grok/v1/sessions/register` and
`/grok/v1/sessions/{id}/replicas/update` attach redacted request-shape metadata
to passthrough logging/session metadata. The fields include
`grok_side_channel_endpoint_type`, `grok_side_channel_endpoint_path_template`,
content type, canonical/raw body byte length, body SHA-256, digest source, JSON
container type, top-level JSON key/type names, and array length when applicable.
They deliberately omit raw body values, auth headers, credential payloads, and
concrete session ids from path templates. Use these fields to capture real
native side-channel payload shape for refresh-continuity debugging without
persisting the payload itself.

Sanitization metadata proves request adaptation only. It does not prove tool
execution, model tool-use quality, or upstream success by itself; combine it
with status, token, cost, and error fields when building reports.

## OpenCode Zen Codex Tool-Adjacency Sanitization

OpenCode Zen authentication is API-key based for this LiteLLM integration.
LiteLLM loads an explicit `LITELLM_OPENCODE_API_KEY` / `OPENCODE_API_KEY`, or a
provider-scoped `type: "api"` key from `LITELLM_OPENCODE_AUTH_FILE` /
`~/.local/share/opencode/auth.json`. Non-API auth types are rejected. Because
there is no refreshable OpenCode credential in this contract, provider-status
sidecar refresh, validation, and auth-expiry persistence do not apply to
OpenCode. Dev compose therefore mounts `/home/zepfu/.local/share/opencode`
read-only into LiteLLM and should not add an OpenCode sidecar writer unless a
new refreshable credential contract is introduced.

Codex/OpenAI Responses traffic that is adapted to OpenCode Zen chat completions
must satisfy OpenAI chat tool-call ordering before egress. If a Responses input
history contains an assistant `function_call` / chat `tool_calls` item without
the immediately following `tool` result messages required by DeepSeek-compatible
chat completion providers, LiteLLM removes the unmatched assistant tool-call
turn and any orphan tool-result messages before sending the OpenCode request.

Rows and traces affected by this path use
`passthrough_route_family=codex_opencode_zen_adapter` and should include the
request tag `opencode-zen-chat-tool-adjacency-sanitized`. Langfuse observations
also include an `opencode_zen.chat_tool_adjacency_sanitized` span with counts
for removed assistant, orphan tool, partial tool, and extra tool messages, plus
the before/after chat message counts. `session_history.metadata` may retain the
sanitizer tag even when the detailed count fields are only present on the
Langfuse observation.

Interpret this sanitizer as request-shape repair, not proof of model quality.
Successful closure still requires the final provider status, token usage, and
absence of provider error or rate-limit observation rows for the same
`trace_id`/`session_id`.

Codex/OpenCode and Codex/OpenRouter adapter probes call `litellm.acompletion`
directly instead of routing through the proxy's common `route_request` helper.
Those direct calls should still pass the proxy-owned shared aiohttp session when
one is available, including candidate-unavailable and rate-limited probe paths,
so failed candidate attempts do not create orphan per-request client sessions.
