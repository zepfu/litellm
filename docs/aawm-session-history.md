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


## Anthropic / Claude Context Window Selection

LiteLLM records **requested** Claude/Anthropic context-window mode in
`session_history.metadata`. This is evidence from request headers, retained
beta metadata, or model suffix labels—not an inference from observed input
token volume.

Persisted keys (allowlisted in `metadata`):

- `anthropic_context_window_mode`: `extended_1m`, `default_200k`, or `unknown`
- `anthropic_context_window_requested_tokens`: `1000000`, `200000`, or `null`
  when unavailable
- `anthropic_context_window_source`: how the mode was determined, for example
  `anthropic_beta_header`, `model_suffix_1m`, `no_extended_context_evidence`,
  or `unavailable`
- `anthropic_context_window_beta`: safe retained beta token when present (for
  example `context-1m-2025-08-07`); omitted when not captured
- `anthropic_context_window_classification`: `classified` or `unavailable`

Header evidence is read from direct `anthropic-beta` / `x-pass-anthropic-beta`
values and retained shapes such as `llm_provider-anthropic-beta` on forwarded
metadata. Model suffix evidence uses retained labels such as `claude-opus-4-7[1m]`
on `inbound_model_alias`, `requested_model_alias`, or related metadata before
normalization.

Live session-history writes may stamp `default_200k` when traffic is clearly
Anthropic but no extended-context evidence exists. Historical backfill only
classifies rows when retained header or suffix evidence exists; otherwise it
sets `unknown` / `unavailable` markers.

## Responses terminal stream semantics

OpenAI/Codex Responses passthrough streams now treat
`response.completed`, `response.failed`, and `response.incomplete` as the
shared terminal event set for both logging reconstruction and
`_collect_responses_response_from_stream`.

- Recognized failed/incomplete terminals are logged with standard logging
  `status=failure` and carry stable terminal diagnostics in `_hidden_params`:
  bounded/redacted `responses_terminal_error` and bounded/redacted
  `responses_terminal_incomplete_details` reason. The collector keeps the full
  terminal `response` payload (including status and incomplete details) for
  adapter behavior while only persisting bounded/redacted fields for
  diagnostics.
- Post-first-byte proxy terminal chunks that emit `response.failed` + `[DONE]`
  are consumed as first-class terminal events by both the logging handler and
  the Responses collector.
- The only synthesized success fallback is the explicit clean
  `response.output_text.done` + `[DONE]` shape with no formal terminal event;
  that path is marked in hidden params as estimated/synthesized.
- Streams with no recognized terminal event and no clean done-only fallback
  produce no reconstructed logging success; the collector closes the source
  iterator and raises HTTP 502.

### Repairing existing rows (`--repair-session-history`)

To stamp or refresh Anthropic context-window metadata on rows already stored in
`public.session_history`, use the backfill script in repair mode. Repairs call
`_enrich_backfill_anthropic_context_window_metadata`, which classifies from
retained beta headers and model suffix labels only—not from input token volume.

```bash
./.venv/bin/python scripts/backfill_session_history.py \
  --repair-session-history \
  --repair-anthropic-context-window \
  [--provider anthropic] [--model claude-opus-4-7] \
  [--request-id <litellm_call_id>] [--trace-id <id>] [--session-id <id>] \
  [--from-start-time <iso>] [--to-start-time <iso>] \
  [--limit N] [--batch-size N]
```

Omit `--apply` for a dry run. JSON output includes `scanned_rows`,
`rows_with_updates`, and `anthropic_context_window_updates`. With `--apply`, only
`metadata` JSONB is updated for context-window changes when
`--repair-anthropic-context-window` is used without `--repair-costs` or
`--repair-tenant-ids`. Combine flags to run multiple repair passes in one scan.

`unknown` / `unavailable` on repaired historical rows means retained evidence in
stored metadata (and `inbound_model_alias` when present) was insufficient to
classify extended context; it does not mean LiteLLM inferred a window from
observed token counts.

## Host Attribution

`public.session_history` includes nullable `client_ip` and `host_name` text
columns. They capture the incoming LiteLLM client address and the resolved
display host used by AAWM route logging.

- `client_ip` stores the canonical requester IP literal when one is available.
- `host_name` stores the resolved host label shown in route rollup headers, for
  example `thoth` from Tailscale/MagicDNS reverse lookup, the local machine's
  Tailscale MagicDNS short hostname when loopback or Docker gateway-adjacent
  traffic is attributed via lookup (`magicdns_local`), `localhost` when that
  local lookup fails, or the IP literal when DNS has no better label.
- Metadata may also retain `client_ip_source` and `host_name_source` for debugging
  (`request_client`, `x_forwarded_for`, `loopback`, `docker_bridge_gateway`,
  `tailscale_self`, `tailscale_self_cache`, `reverse_dns`, `magicdns_reverse`,
  `magicdns_local`, `magicdns_local_cache`, `reverse_dns_cache`,
  `magicdns_reverse_cache`, `ip_literal`, `ip_literal_cache`).
- For local display sources (loopback, unspecified, link-local, Docker bridge
  gateway), LiteLLM resolves the host label in this order:
  1. A sanitized Tailscale self identity snapshot (for example
     `Self.DNSName`, `Self.TailscaleIPs`, and `MagicDNSSuffix` only) from a
     read-only directory mounted at `/host/aawm` in `litellm-dev`, with the
     snapshot file at `/host/aawm/tailscale-self.json`.
  2. The existing hostname/MagicDNS discovery path: process-visible
     `100.64.0.0/10` addresses, hostnames/FQDNs from the process and optional
     read-only host hostname files such as `/host/etc/hostname`, tailnet search
     domains from resolver config, direct MagicDNS A lookups, and MagicDNS PTR
     for the final short hostname.
  3. `localhost` when neither path yields a label.
  The snapshot path is configurable with
  `AAWM_ROUTE_HOST_TAILSCALE_SELF_SNAPSHOT_PATH`; host operators can refresh it
  with `scripts/write_tailscale_self_snapshot.py` without mounting the Tailscale
  socket into the container. The dev compose mount uses
  `AAWM_ROUTE_HOST_TAILSCALE_SELF_SNAPSHOT_HOST_DIR` for the host-side snapshot
  directory. Results are cached under dedicated local-host cache keys so
  repeated local requests do not rescan discovery surfaces.
- For Tailscale CGNAT client IPs in `100.64.0.0/10`, LiteLLM tries normal reverse
  DNS first. If that misses, it queries the MagicDNS resolver at `100.100.100.100`
  directly for PTR and uses the returned short hostname when available.
- Successful hostname resolutions (including MagicDNS) use a longer cache TTL;
  IP-literal fallbacks use a shorter TTL so a later MagicDNS success can replace
  them without waiting for the full hostname cache window.

Route rollup headers use the exact display form
`repo#Client[version]@host`, for example
`aawm-infrastructure#Codex[0.142.5]@thoth /openai_passthrough/responses`.

## Repository And Tenant Attribution

`session_history.repository` identifies the workspace or project label used for
route grouping and drill-down. `session_history.tenant_id` is stricter: LiteLLM
only falls back from repository to tenant when metadata records a trusted
`repository_source`.

When request-time metadata includes a bounded route-rollup group header such as
`metadata.aawm_route_rollup_context.group_header_label='aegis@Claude[2.1.199]'`,
LiteLLM may recover the normalized repository prefix (`aegis`) for
`session_history.repository` and, when no stronger explicit tenant signal is
present, for `session_history.tenant_id`, but only when the prefix is a known
AAWM workspace repository. This fallback runs only after explicit tenant headers
and explicit repository metadata keys; it does not promote
`langfuse_trace_user_id`, `trace_user_id`, owner labels such as `zepfu`, or
other stale Codex trace-user values. Historical rows that never stored rollup
context in
`session_history.metadata` may require Langfuse observation metadata during
backfill or `repair_session_history_repository_identity.py` repair passes.

Trusted repository sources include explicit repository headers, explicit
metadata or `litellm_metadata` keys, and current workspace context such as
`<environment_context><cwd>...</cwd></environment_context>`,
`AGENTS.md instructions for ...`, or workspace-directory blocks. Codex traffic
uses a narrower tenant fallback rule: generic `metadata.repository` remains a
diagnostic repository label unless it names a known AAWM workspace repository.
Known workspace labels such as `aawm-tap`, `litellm`, or `dashboard-shell` may
be promoted to `tenant_id`; arbitrary labels, placeholder worktree names, owner
names, and file-like values are not promoted. Other trusted Codex sources remain
explicit headers, current `x-codex-turn-metadata` `project_path`, or current
workspace/cwd text. Recursive text scans
intentionally ignore assistant history and tool-output items such as
`function_call_output`, `custom_tool_call_output`, `tool_search_output`, and
reasoning blocks, because those can contain stale worktree paths or fixture
strings from earlier turns.

Rows where repository was inferred from untrusted text may keep a diagnostic
repository label, but file-like or cited-artifact candidates can remain
`repository=NULL` rather than being promoted to a misleading workspace. These
records include `metadata.repository_tenant_fallback_skipped=true` and
`metadata.tenant_id_source=repository_untrusted` so downstream reporting can
distinguish "untrusted/unresolved repository" from an omitted attribution field.
Claude CLI rows that arrive without a trusted `aawm_claude_project`, repository
header, or current workspace signal are also stamped as unresolved, with
`metadata.session_history_repository_unresolved_reason=no_trusted_claude_project_signal`,
rather than promoted from client name alone. `aawm_claude_project` must normalize
to a known AAWM workspace repository such as `dashboard-shell`; owner names such
as `zepfu`, placeholders, and file-like labels are retained only as diagnostic
metadata and are not used as `repository` or `tenant_id`.
The current missing-repository reason codes are:

- `no_trusted_repository_signal`: no trusted repository, tenant, workspace, or
  header signal was available for the row.
- `no_trusted_claude_project_signal`: Claude/Anthropic traffic had no trusted
  `aawm_claude_project` or equivalent current workspace signal.
- `no_trusted_grok_project_signal`: Grok/XAI traffic had no trusted project or
  current workspace signal.

Large groups of non-excluded null repositories are surfaced by the provider
status sidecar as `large_null_repository_cluster` anomalies until they are
repaired or stamped with an explicit durable unresolved classification such as
`metadata.session_history_repository_status=unresolved`.


Codex memory-writing workloads still record `metadata.source_repository`,
`metadata.workload_type`, and `metadata.workload_subtype` (and optional
`metadata.memory_workload_label` such as `litellm (memory)`) for diagnostics,
but `session_history.repository` and `session_history.tenant_id` group on the
base workspace repository (for example `litellm`) so dashboards do not split
memory traffic into a separate pseudo-repo.

Truncated or placeholder repository labels containing `...` are never accepted
as repository identities; repair jobs classify or resolve them instead of
preserving them on `repository` or `tenant_id`.

When a row already has a known workspace `repository` but
`metadata.tenant_id_source` is `repository_untrusted` or `trace_user_untrusted`
from an earlier stale `trace_user_id` or untrusted metadata pass, normalization
and `repair_session_history_repository_identity.py` may repair the row to
`repository=tenant_id=<known repo>` with `metadata.session_history_repository_status=repaired`
and `metadata.tenant_id_source=repository_repair`. Stale `trace_user_id` rejection
does not block that promotion when a valid current repository label is present.

Stale Codex `metadata.trace_user_id` values from compact/resume history are
diagnostic only. They may remain in `metadata.trace_user_id` for Langfuse and
session-history inspection, but LiteLLM does not promote them to
`session_history.tenant_id` unless the tenant came from an explicitly trusted
current source such as tenant headers, current `x-codex-turn-metadata`
`project_path`, or current workspace/cwd text. Rejected promotions are marked with
`metadata.tenant_id_source=trace_user_untrusted` and
`metadata.trace_user_tenant_fallback_skipped=true`.

Paths that appear only because prompt text, handoff notes, or completion output
referenced an artifact file, such as
`/home/zepfu/projects/<repo>/.analysis/suggestion.md`, proposal files, or other
`.analysis/` paths, do not establish `session_history.repository` or
`session_history.tenant_id` unless a trusted current-workspace source (headers,
current cwd/workspace blocks, or explicit metadata keys above) proves that repo is
the active workspace for the turn. Maintainers should treat those strings as
cited artifact locations, not as ownership signals.

Backfill or repair jobs may record the referenced artifact owner separately in
metadata (for example which repo owned the cited file) while keeping
`repository` and `tenant_id` null or setting them only from trusted
tenant/header/trace signals. That split keeps historical drill-down honest:
diagnostic artifact context without promoting stale prompt references into route
grouping.

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

Grok no-format `/v1/billing` monthly counter payloads (`monthlyLimit`,
`used`) populate `quota_limit`, `quota_used`, `quota_remaining`, billing-period
columns, and quota key `xai_grok_build_monthly_requests:requests` with
`quota_period=monthly`. They are separate from weekly credit telemetry.

Grok `/v1/billing?format=credits` weekly payloads use quota key
`xai_grok_build_weekly_credits:credits` with `quota_period=weekly` when
`currentPeriod.type` is `USAGE_PERIOD_TYPE_WEEKLY`. `creditUsagePercent` is used
percent; `remaining_pct` stores remaining percent. Fresh weekly periods without
`creditUsagePercent` persist `remaining_pct=100` and `used_percentage=0` with
explicit fresh-period evidence, not null/unknown. xAI OAuth rate-limit headers populate absolute request/token
amounts and carry billing period ends when the provider config or managed
subscription context exposes one.

## Provider Credit Observations


Anthropic Fable weekly overage-included headers (`anthropic-ratelimit-unified-7d_oi-*`)
persist as quota key `anthropic_unified_7d_oi:7d_oi` with `quota_period=seven_day`
and `window_minutes=10080`. This is distinct from the baseline `anthropic_unified_7d:7d`
bucket and from the retired `anthropic_unified_7d_sonnet:7d_sonnet` series. Prompt-side
token cache breakdowns remain sub-buckets of input tokens, not additional quota keys.

`public.provider_credit_observations` stores provider banked reset-credit
snapshots from scheduled sidecar polls (for example Codex usage-limit reset
credits polled from `/backend-api/wham/rate-limit-reset-credits`). Canonical
grouping fields are `environment`, `provider`, `account_hash`, `credit_family`,
`credit_identity`, `credit_type`, and `source`. There are no `client` or
`client_version` columns.

Normalized fields:

- `credit_identity`: stable per-credit key (provider id or derived hash).
- `available_count`: per-row availability (`1` or `0` for detail rows; aggregate
  semantics only when legacy aggregate payloads are parsed).
- `granted_at` / `expires_at`: credit grant and expiry from the detail payload;
  not the same as `/wham/usage` rate-limit window resets.
- `status`: `available`, `used`, or `expired` (including lifecycle inference when
  credits disappear or pass expiry).
- `redeem_started_at` / `redeemed_at`: provider redemption timestamps when present.
- `operator_annotation` / `source_url`: operator context for known promotions or
  backfill rows (no secrets or raw account ids).
- `raw_provider_fields` and `evidence`: sanitized JSONB interpretation notes.

`public.provider_credit_current` exposes the latest row per
`(environment, provider, account_hash, credit_family, credit_identity, source)`
for dashboards and investigations. These observations are distinct from
`public.rate_limit_observations`, which capture quota windows and billing
snapshots from request traffic.


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

### Flush retry telemetry semantics

Operators should treat session-history persistence logs as a small state machine,
not as implicit data-loss signals.

- **Transient retry (in budget):** Retryable failures such as asyncpg
  `TimeoutError`, connection drops, or pool acquire timeouts emit **WARNING**
  (not ERROR) with `degraded_telemetry=true`, `at_risk_of_loss=false`, and
  `retry_count` / `retry_budget_remaining` when the outer retry loop is active.
  Records are not considered lost while retries continue or a retry write-ahead
  spool exists.
- **Write-ahead protection:** Before an inline retry, LiteLLM may create a
  temporary retry write-ahead spool file. Logs include
  `retry_write_ahead_spooled=true` and `retry_write_ahead_spool_path_present=true`
  when that artifact exists. While protected, exhaustion logs state that the
  batch remains on disk for replay (`spooled=true`, `at_risk_of_loss=false`).
- **Recovery:** Successful flush after retry logs `flush_recovered=true` with
  `degraded_telemetry=false`, `spooled=false`, and optional
  `retry_spool_removed=true` when the write-ahead file is deleted.
- **Durable spool replay:** After retry budget exhaustion, batches move to the
  durable JSONL spool directory for the in-process drainer (`spool_replay_*`
  telemetry). That is replay protection, not loss. If a spool artifact
  disappears between listing and load because another recovery path already
  removed it, the drainer skips that stale path instead of quarantining it as a
  bad record.
- **True at-risk / error cases:** ERROR or `exception` logs with
  `at_risk_of_loss=true` indicate spool creation failed, deferred flush from a
  running loop could not spool, or retry exhaustion could not write any
  protection artifact. Non-retryable flush failures still use `exception` on
  the first failure in a window.

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

- Namespace env: `AAWM_ALIAS_ROUTING_STATE_NAMESPACE`.
  When it is omitted, `LITELLM_LANGFUSE_TRACE_ENVIRONMENT` (falling back to
  `LITELLM_AAWM_ERROR_LOG_ENV`) derives `aawm-routing-dev-v1` or
  `aawm-routing-prod-v1`. Use one explicit shared namespace only for
  intentionally shared routing planes.
- Redis timeout and write-retry behavior is controlled by:
  - `AAWM_ALIAS_ROUTING_REDIS_TIMEOUT_SECONDS` (`float`, default `10`, clamped
    to `1.0`–`60.0`): passed to the alias-routing Redis client
    `socket_timeout` and shared by both URL and host-based Redis
    configuration. Non-finite values (`nan`, `inf`, `-inf`) fall back to the
    default `10`.
  - `AAWM_ALIAS_ROUTING_REDIS_DURABLE_WRITE_RETRY_BACKOFF_SECONDS` (`float`,
    default `0.25`, clamped to `0.05`–`2.0`): wait time before one bounded retry
    for durable `SET` write failures when the failure is retryable. Non-finite
    values (`nan`, `inf`, `-inf`) fall back to the default `0.25`.
- Durable key shape:
  `aawm:alias-routing:{namespace}:{family}:{kind}:{state_key}` where `family` is
  `codex` or `anthropic`, and `kind` is `affinity` or `cooldown`.
- Durable payloads store absolute wall-clock expiry (`expires_at_epoch` via
  `time.time()`). Process-local maps still use monotonic expiry for the fast path.
- `AAWM_ALIAS_ROUTING_REDIS_*` settings only control alias-routing state
  persistence. They do not enable or alter LLM response caching.
- Durable writes are bounded:
  - The alias-routing writer opts into Redis `SET` exception propagation so it
    can retry accurately; unrelated `RedisCache.async_set_cache` callers retain
    the existing default write-error suppression behavior.
  - Alias-routing durability attempts one retry max when a write fails with
    connection/timeout-class exceptions.
  - Retriable write exceptions are limited to Redis `TimeoutError`,
    Redis `ConnectionError`, and Python timeout/connection-class failures.
  - Non-retryable errors fail fast and return `False` immediately.
  - If the retry attempt also fails, the final visible durable-write result is
    `False`; it must never be treated as success.
- Payload and TTL are preserved on retries:
  - The exact durable payload is written with unchanged content and stable positive
    TTL on each attempt, including the `expires_at_epoch` field calculated once for
    the request.
- Standard LiteLLM Router `failed_calls` counters remain process-local
  telemetry and are not mirrored by the durable alias-routing Redis manager.
- Read order: memory first, then durable cache hydrate; if durable cache is absent
  or unavailable, selectors keep the existing in-memory fallback without failing
  the request.
- Session-history / alias metadata may include lightweight source fields such as
  `codex_auto_agent_affinity_state_source`, `codex_auto_agent_cooldown_state_source`,
  `anthropic_auto_agent_affinity_state_source`, and
  `anthropic_auto_agent_cooldown_state_source` with values `memory`,
  `durable_cache`, `durable_quota`, or `local_fallback`. These fields do not
  store prompts, tools, credentials, or raw session payloads. `durable_quota`
  means the selector skipped a candidate because a separate quota observation,
  rather than candidate-specific cooldown state, showed the provider account or
  lane was exhausted until a known reset time.
- Readiness reporting should reflect whether routing state is coming from durable
  cache (`durable_cache`) or local fallback (`local_fallback` / in-memory), so
  operators can distinguish persistent state from process-local behavior.
- Shared PostgreSQL durable quota (`durable_quota`) remains separate and is not
  merged across routing namespaces.
- Bare transient upstream statuses (`500`, `502`, `503`, and `529`) that do not
  carry explicit capacity, quota, rate-limit, or usage-limit evidence are
  treated as request-local alias failures for most candidates. They are skipped
  for the current alias progression so failover can continue, but they do not
  write durable provider/model cooldown state. Native Grok 4.5 candidates on
  `codex_grok_native_responses_adapter` and
  `anthropic_grok_native_responses_adapter` use cooldown scope `none` for these
  bare transient classes: fresh requests exclude the failed native candidate for
  that request only, while in-flight continuations retry the **same affinity
  candidate** with a dedicated **request-scoped** total-attempt budget (default
  **8**, independent of alias candidate-pool length and preserved across outer
  candidate-selection re-entry within the same request; override
  `AAWM_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS`, clamped 6–16) and short
  exponential backoff capped near 1s between retries. These retries never switch
  provider/model, never write local or Redis candidate cooldown, and do not
  signal `redispatch_required`. Generic `candidate_unavailable` is deliberately
  excluded from this budget. Attempt metadata may include
  `native_grok_continuation_retry` with `scheduled_same_candidate_retry` or
  `same_candidate_retry_exhausted` status. Explicit capacity/rate-limit/usage-limit
  failures still use candidate cooldowns; malformed-output redispatch and
  non-native candidates are unchanged.
- Generic xAI `candidate_unavailable` failures (missing, expired, or refreshing
  credentials; broad probe unavailability) never write durable candidate
  cooldowns. Native Grok 4.5 keeps cooldown scope `none` for that class; all
  other xAI alias candidates (`oa_xai/grok-4.5`, Composer, Grok Build, managed
  OAuth lanes) use request-local exclusion only so transient credential gaps
  cannot leave multi-hour Redis keys. Explicit rate-limit, capacity, usage, and
  quota classes on xAI candidates still use durable candidate cooldowns.
- `malformed_tool_call_text` remains a hard reject with in-flight redispatch /
  failover. For native Grok 4.5 only, that class is request-local (no durable
  30-minute candidate cooldown). Composer, Grok Build, Spark, and other
  candidates keep durable candidate cooldowns for malformed tool-call text.
- For the `gpt-5.3-codex-spark` candidate only, durable capacity, rate-limit,
  usage-limit, malformed_tool_call_text, upstream-overloaded, and upstream-timeout
  cooldowns are capped at five minutes. Other candidates keep the default durable
  behavior: 3-hour caps for capacity/rate-limit/usage-limit and 30-minute
  candidate-cached malformed tool-call cooldowns (except native Grok 4.5
  malformed tool-call text, which is request-local), with request-local
  transient cooldowns remaining separate.

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
next configured usable candidate. Alias selection also honors process-local
adapter cooldown evidence for OpenRouter and Google Code Assist candidates so
recent adapter-level exhaustion suppresses those candidates before the next
dispatch. Declared OpenRouter free daily candidates are additionally checked
from `rate_limit_observations` through a short-timeout, short-TTL cache. When
the latest `openrouter_free_daily_requests:requests` observation reports
`remaining_pct <= 0` and a future `expected_reset_at`, those free daily
candidates are skipped before upstream attempts with
`reason=durable_quota_exhausted` and `cooldown_state_source=durable_quota`; if
the lookup fails or expires, selection fails open and preserves the declared
alias order.

Expected upstream provider `429` responses that are returned directly to the
requester are still durable provider-error observations. The
`provider_error_observations` row should preserve the requested `model_group`,
`provider`, `model`, `status_code`, and `error_class=rate_limited`. For
OpenRouter provider-capacity errors, metadata also records bounded
`upstream_provider_name`, `upstream_is_byok`, `upstream_error_raw`,
`litellm_retry_count`, `litellm_max_retries`, `litellm_retries_exhausted`,
`available_model_group_fallbacks`, `no_model_group_fallbacks`, and
`provider_error_fingerprint` fields. These fields are the maintainer-facing
record for expected provider errors that should not reopen traceback-style
runtime error intake.

For Grok native alias probes, the specific upstream 403
`permission-denied` response that says access to the chat endpoint is denied is
recorded as a candidate-unavailable condition, allowing the alias to progress to
the next declared model instead of surfacing repeated pass-through tracebacks.
In-flight redispatch-required 429 responses include bounded audit metadata such
as `failure_class`, `cooldown_scope`, `error_status_code`, `retry_after_seconds`,
`aawm_alias_routing_audit_events`, and the attempt/skipped-candidate summaries
so operators can correlate client-visible redispatch failures with the durable
cooldown that triggered them.

Terminal Codex and Anthropic auto-agent outcomes that never complete a normal
provider write path (all-candidates-unavailable / no-candidate, and in-flight
`redispatch_required` 429s) also emit **audit-only** persistence:

- events are enqueued with `_skip_session_history=true` so they do **not** insert
  a `session_history` row and do **not** double-write through a later success or
  normal fallback path;
- durable rows still land in `aawm_alias_routing_audit` via the existing
  best-effort audit insert;
- direct event context IDs (`session_id`, `trace_id`, `litellm_call_id`,
  `agent_id`) are attached on the event itself; when no upstream
  `litellm_call_id` is available, one request-scoped fallback ID is generated
  and reused across every terminal attempt event;
- structured agent/dispatch fields (`agent_name`, `agent_role`, `agent_profile`,
  `thread_source`, `dispatch_id`, `redispatch_ordinal`) are preferred from
  request metadata; when those are absent, the exact role-profile sentence
  `You are a '<role>' agent.` is used as a fallback for `explorer` /
  `worker` / `default`, and inferred profiles set `thread_source=subagent`;
- `actual_prior_tool_activity_summary` records a conservative count of prior
  tool calls/results already present on the request (continuation markers alone
  do not count), with separate `has_prior_file_edit_activity`,
  `prior_file_edit_tool_call_count`, and `prior_file_edit_tool_names` fields
  for recognized edit/write/`apply_patch` tools;
- terminal audit-only persistence carries the accumulated candidate/fallback
  attempt events, not only the final failed candidate;
- terminal events set `terminal_activity_status` to
  `failed_after_partial_activity` or `failed_no_activity` so parents can
  distinguish partial-edit failures from no-op capacity failures;
- `cooldown_state_source` is propagated from the selected candidate/selection
  state onto the audit event when known.

These fields are observability-only. They do not change candidate order,
affinity, cooldown duration, or redispatch thresholds. For tool-bearing or stateful
`aawm-code-anthropic` requests, every declared candidate route is treated as a
Claude Code tool-contract route: if the alias declares OpenAI, xAI,
native Anthropic, or another provider/model target, that target must preserve
tool calls, tool-use ids, tool arguments, tool-result replay, and ordered
failover metadata for engineering-agent traffic. A declared candidate must not
be skipped as policy-incompatible for the same request class it is meant to
serve. If a declared candidate mishandles tools, that is an adapter translation
defect to fix or evidence for removing/reclassifying the candidate; it is not a
selector-side compatibility decision.

## Anthropic Upstream Egress Boundary

The Anthropic/Claude routing boundary is determined by the selected upstream
provider and model, not by the request's wire format. An Anthropic-shaped
Claude Code request may use a supported adapter to reach a non-Anthropic model.
That adapter flow does not authorize an Anthropic/Claude model to use the same
cross-provider egress.

Any selected Anthropic/Claude model must use an Anthropic-native route and
Anthropic-native provider credential. It must never be sent through
Codex/ChatGPT OAuth, `chatgpt.com/backend-api/codex/responses`, an OpenAI/Codex
adapter, or another provider's transport. This prohibition applies to alias
candidate selection, fallbacks, retries, cooldown recovery, probes, smoke
tests, acceptance harnesses, and manual diagnostics.

If the Anthropic-native route or credential is unavailable, routing must fail
closed and retain explicit failure/audit metadata. It must not silently select
a Codex path for the Anthropic model. Cross-provider egress for a selected
Anthropic/Claude model is treated as a potential terms-of-service violation.

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

`aawm-sota-openai` mirrors `aawm-sota` and uses this order:

1. `gpt-5.6-sol`
2. `gpt-5.5` as the OpenAI last-resort candidate

`aawm-sota-xai` uses this order:

1. `oa_xai/grok-4.5` via the managed xAI OAuth Responses adapter
2. `grok-4.5` via the native Grok OIDC Responses adapter
3. `grok-build` as the xAI last-resort candidate

Grok 4.5 is treated as a live candidate. Generic
`aawm_codex_auto_agent_candidate_unavailable` probe failures do not apply a
durable Grok 4.5 cooldown: native Grok 4.5 uses cooldown scope `none`, and
other xAI alias candidates (Composer, Grok Build, managed OAuth Grok 4.5)
use request-local exclusion only. Explicit usage, quota, rate-limit, or
capacity signals still use the normal durable candidate cooldown/fallback
path. Native Grok 4.5 `malformed_tool_call_text` remains rejected and can
redispatch in-flight, but is request-local rather than a durable candidate
cooldown.

`aawm-sota` uses this order:

1. `gpt-5.6-sol`
2. `gpt-5.5` as the OpenAI last-resort candidate

`aawm-low` and `aawm-low-anthropic` use this order:

1. OpenRouter North Mini (`openrouter/cohere/north-mini-code:free`)
2. OpenRouter Owl Alpha (`openrouter/owl-alpha`)
3. OpenCode Zen `deepseek-v4-flash`
4. OpenCode Zen `big-pickle`
5. `gpt-5.6-luna`
6. `gpt-5.4-mini` as the OpenAI last-resort candidate for `aawm-low`
7. native Anthropic Haiku as the last-resort candidate for `aawm-low-anthropic`

`aawm-code` uses this order:

1. `gpt-5.3-codex-spark`
2. `xai/grok-4.5` via the native Grok OIDC Responses adapter
3. `grok-composer-2.5-fast`
4. `oa_xai/grok-build`
5. `gpt-5.6-terra`
6. `gpt-5.5` as the OpenAI last-resort candidate with medium reasoning

`aawm-orchestration` uses this order:

1. `gpt-5.6-terra`
2. `gpt-5.5` as the OpenAI last-resort candidate

`aawm-sota-anthropic` uses this order:

1. native Anthropic `claude-fable-5`
2. native Anthropic `claude-opus-4-8[1m]` as the last-resort candidate

`aawm-code-anthropic` uses this order:

1. `gpt-5.3-codex-spark`
2. `xai/grok-4.5` via the native Grok OIDC Responses adapter
3. `grok-composer-2.5-fast`
4. `oa_xai/grok-build`
5. native Anthropic `claude-sonnet-5[1m]` (1m context window)
6. native Anthropic `claude-sonnet-5`
7. native Anthropic `claude-sonnet-4-6` as the last-resort candidate

`aawm-orchestration-anthropic` uses this order:

1. native Anthropic `claude-opus-4-8[1m]` as the sole last-resort candidate

OpenAI `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna` pricing in
`model_prices_and_context_window.json` follows the official GPT-5.6 preview page
(`https://openai.com/index/previewing-gpt-5-6-sol/`): per-token input/output
costs, cache write at 1.25× uncached input, and cache read at 10% of uncached
input. Official GPT-5.6 API documentation also verifies reasoning efforts
`none`, `low`, `medium`, `high`, `xhigh`, and `max`; all three entries therefore
set `supports_reasoning`, `supports_none_reasoning_effort`,
`supports_xhigh_reasoning_effort`, and `supports_max_reasoning_effort`.
Unverified context-window and unrelated capability metadata remain omitted.

### Codex reasoning-effort reconciliation

Native OpenAI Codex Responses requests reconcile a recognized requested
reasoning effort after the concrete model is resolved and before provider
egress. The ordered vocabulary is `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, and `max`.

- The handler only maps downward when the requested tier is above the resolved
  model's catalog-advertised ceiling. It never raises a requested effort.
- `gpt-5.3-codex-spark` advertises reasoning through `xhigh`, so an inbound
  `reasoning.effort=max` is emitted as `xhigh`.
- GPT-5.6 Sol, Terra, and Luna advertise
  `supports_max_reasoning_effort=true`, so `max` remains `max`.
- Alias requests recalculate from the original inbound body for every candidate;
  a Spark attempt may emit `xhigh` while a later Terra attempt emits the
  original `max`.
- Direct concrete-model Codex passthrough requests use the same reconciliation
  after adapter and alias resolution. Provider routes without an explicit
  compatible capability contract are unchanged.
- Unknown effort strings and models without a known ceiling are not guessed or
  rewritten. Provider-native validation remains authoritative, and
  deterministic HTTP 400 invalid-effort responses are not classified as
  retryable alias failures.

Structured metadata records `reasoning_effort_requested`,
`reasoning_effort_native_value`, `reasoning_effort_supported_ceiling`,
`reasoning_effort_resolved_model`, `reasoning_effort_resolved_provider`,
`reasoning_effort_mapping_reason`, and, for aliases,
`reasoning_effort_candidate_attempt`. Downward mappings also record
`reasoning_effort_clamped_from` and `reasoning_effort_clamp_reason`. Tags include
the emitted effort, supported ceiling, mapping, and candidate attempt where
applicable.

The user-level Codex model catalog exposes `max` for Sol, Terra, and Luna and
`ultra` only for Sol. `ultra` is a Codex product intelligence mode that combines
maximum reasoning with proactive subagent delegation; it is not sent upstream
as an OpenAI API `reasoning.effort` value.

This is default alias behavior; no staging environment variable is required for
dev or production routing.

Alias-visible metadata (for example, route audit fields and session-history
model keys) must keep the provider-prefixed names (`openrouter/...`) so
operators can distinguish the declared OpenRouter candidates. The OpenRouter
completion adapter must continue to send the provider-stripped model slug
upstream (`cohere/north-mini-code:free`, `owl-alpha`) to match OpenRouter
ingress expectations.

The Codex `aawm-code` alias uses `gpt-5.5` as the OpenAI last-resort candidate after `gpt-5.3-codex-spark`, Grok adapter lanes, and `gpt-5.6-terra`,
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
`0600`. Scheduled endpoint probes cover Anthropic, OpenAI, OpenRouter, xAI, and
NVIDIA NIM front doors only; Google/Gemini control-plane hosts are outside
provider-status monitoring. The atomic write preserves existing file ownership/private mode unless
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

Antigravity Code Assist routes use OAuth token files on the host. In prod,
LiteLLM should be configured with `LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE` for
the refreshed token copy and `LITELLM_ANTIGRAVITY_SEED_AUTH_FILE` for the
read-only Antigravity CLI login seed, normally
`/home/zepfu/.gemini/antigravity-cli/antigravity-oauth-token`. LiteLLM is a
read-only consumer of those files. During request handling it loads the first
valid candidate token and never attempts a direct OAuth refresh or invokes
`agy`. If all candidate tokens are expired or invalid, LiteLLM fails the
Antigravity candidate with a clear refresh-required message.

For AAWM auto-agent aliases, stale or missing Antigravity token data is treated
as provider-auth degradation during candidate selection. The selector records the
Antigravity candidate under the `antigravity:auth_degraded` lane, marks it
skipped with `reason=auth_degraded`, applies a short candidate cooldown, and
continues to the next declared candidate. This expected degraded state logs a
bounded warning without traceback; unexpected Antigravity lane-resolution
exceptions still emit traceback-bearing error logs for intake.

The provider-status sidecar does not schedule Antigravity OAuth refresh, persist
Antigravity auth telemetry, or probe Google/Gemini front-door endpoints.
Antigravity token maintenance remains outside provider-status ownership (for
example `scripts/antigravity_oauth_refresh.py` for manual or non-sidecar use).
Historical `provider_auth_observations` rows for Antigravity may still exist in
the database.

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
`function_call_output`, remain in the outbound request. For Grok native
Responses passthrough, `function_call.arguments` is coerced to a JSON object
before forwarding: existing dicts are preserved, parseable JSON strings are
parsed, and missing/empty/invalid/non-object values default to `{}`. If a Grok-family
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
`cwd_uri`, and `aawm_claude_project`. For Claude CLI traffic,
`aawm_claude_project` is the preferred trusted project source; client product
strings such as `claude-cli` alone do not establish repository ownership.
LiteLLM omits prompt-like, sentence-like, or punctuation-heavy identity values
instead of printing raw request text. LiteLLM intentionally does not inspect
prompt text or raw tool arguments for the route log.

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
YYYYMMDD HH:MM:SS [EARLY] repo@Client[version] /incoming
```

Codex rollup headers prefer the current `x-codex-turn-metadata.project_path`
repository when available. Known placeholder or fixture-like repository labels
such as `x`, `wt`, and `wt-ops-xyz` are suppressed instead of being printed as
the route owner.

`[EARLY]` appears only when a bounded in-memory cap forces a flush before the
interval elapses. Each rollup subline uses ` - model(alias) - Turns: N` with an
optional trailing status tag (`[Degraded]`, `[Cooling Down]`, `[Failed]`, or
`[Exhausted]`). `Cooling Down` is reserved for actual candidate-scoped cooldown
or skipped-cooldown state; `retryable_no_cooldown`, scope `none`, and
request-local redispatch/failover failures render as `Failed` instead of
`Cooling Down`. Exhausted and Degraded labels are unchanged. The subline appends
` -> outgoing` only when the model's
destination differs from the incoming endpoint's default upstream target for
the group. For example, native Anthropic traffic under `/anthropic/v1/messages`
omits `api.anthropic.com/v1/messages`, and Codex passthrough traffic under
`/openai_passthrough/responses` omits
`chatgpt.com/backend-api/codex/responses`; mixed-provider routes such as Grok
or Google adapters remain explicit on their own sublines. If LiteLLM cannot
infer an endpoint default, it suppresses the destination only when every subline
in the bucket shares the same outgoing target. `Turns` counts completed
requests only. Alias-route candidate events that degrade, cool down, fail, or
exhaust before a successful completion emit an immediate compact status line:

```text
YYYYMMDD HH:MM:SS - <alias>: <model> Status: <status> - Message: <details>
```

Those events also contribute zero-turn rollup sublines so multiple failed
candidates remain visible in the same bucket. Rollups flush on the configured
interval and at process shutdown via a shutdown-safe flush helper.

Successful streaming and non-streaming Codex auto-agent requests adapted to
OpenCode Zen chat completions or OpenRouter chat completions also register
native access-log replacement and completed-turn rollups. Their rollup sublines
use the real outbound endpoint (`opencode.ai/zen/v1/chat/completions` or
`openrouter.ai/api/v1/chat/completions`) rather than the internal adapter route
family, so operators can distinguish provider traffic without also seeing raw
`adapted_to=...` Uvicorn access records for successful requests.

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

## OpenAI Responses Function-Name Sanitization

OpenAI Responses requests may contain the same function name in historical
`input` items, current `tools` declarations, and a function `tool_choice`.
Before direct Codex/OpenAI Responses egress, including
Anthropic-to-Responses adapter routes, LiteLLM builds one request-local mapping
for names longer than 64 characters and applies it consistently to:

- top-level `input` items with `type=function_call`;
- top-level `tools` entries with `type=function`;
- top-level function `tool_choice` values.

Existing names of 64 characters or fewer are reserved unchanged. Rewritten
names use a deterministic digest and collision fallback so distinct original
names cannot silently merge. The mapping exists only for the lifetime of the
request. Successful JSON and SSE Responses output restores the original
client-visible name on function-call items, function-argument completion
events, and terminal response payloads.

When rewriting occurs, bounded metadata may include:

- `responses_function_name_sanitization_algorithm`;
- `responses_function_name_sanitization_max_length`;
- `responses_function_name_sanitized_distinct_count`;
- `responses_function_name_sanitized_occurrence_count`;
- `responses_function_name_sanitized_surfaces`;
- `responses_function_name_sanitization_collision_fallback`.

These fields describe the adaptation only. They never contain the original
names, rewritten names, or the request-local mapping. Use them to explain why
the provider-bound shape differs from captured client input, not as evidence
that a function was selected or executed.

## Output Contract Metadata

Runtime session-history scoring records bounded output-contract classifications in
`session_history` columns and mirrored `usage_*` metadata fields. These checks are
metadata-only: they do not execute, rewrite, or convert assistant text into tool
calls.

When assistant output prints literal tool-invocation text instead of structured
`tool_use`, LiteLLM classifies the generation with
`output_contract_failure_class=literal_tool_call_text`,
`output_contract_compliance_score=0.0`, and
`agent_score_reasons.output_contract_compliance=["literal_tool_call_text"]`.
This includes serialized `composer_call` transcript text and Claude-style XML
literals such as line-start `<invoke name="Bash">` blocks with `<parameter>`
content. Benign prose that merely discusses those strings inline is not flagged.
Request-side `<tool_use_error>` content and normal structured `tool_use` blocks
remain distinct from this assistant-side literal-text failure class.

When spawned-agent output is reduced to a pseudo function wrapper such as
`<function=explorer>["/home/zepfu/projects/litellm"]</function>`, LiteLLM
classifies the generation with
`output_contract_failure_class=malformed_final_payload`,
`output_contract_compliance_score=0.0`, and
`agent_score_reasons.output_contract_compliance=["malformed_final_payload"]`.
This captures malformed final payloads that are not structured findings or a
normal final answer, while avoiding benign prose that quotes the wrapper while
explaining the incident.

Langfuse-only historical backfills cannot reconstruct a full snapshot once
generation metadata has been compacted. They preserve the compact hash/reference
fields when present, but the durable table is populated only by runtime
session-history ingestion or by older Langfuse rows that still carried the
inline `aawm_tool_definition_snapshot` value.

`scripts/backfill_session_history.py` in `--source-mode langfuse_clickhouse` reads
Langfuse ClickHouse `observations` in batches. By default it does **not** select
`o.input` or `o.output` (the query uses `NULL AS observation_input` /
`observation_output`) to avoid scanning large payload blobs across broad time
windows. Reconstruction still uses trace fields, observation metadata, usage/cost
columns, and `tool_calls` / `tool_call_names`. Pass `--clickhouse-include-payloads`
when you explicitly need full input/output hydration from ClickHouse.

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

## Worker Context Exhaustion Metadata (D1-412)

Upstream orchestrators may attach bounded worker failure context on
`litellm_metadata` when a worker subagent hits context-window exhaustion or
stops with partial telemetry. LiteLLM copies only the allowlisted keys into
`session_history.metadata` without treating partial model output as success.

Allowlisted keys:

- `worker_context_exhaustion_failure_class`
- `worker_context_exhaustion_failure_reason`
- `worker_context_exhaustion_partial_output_summary`
- `worker_context_exhaustion_changed_paths_hint` (string or bounded path list)
- `worker_context_exhaustion_attempted_patch_scope`
- `worker_context_exhaustion_last_visible_message`
- `worker_context_exhaustion_success` (when supplied as `false`, persisted as `false`)
- `worker_context_exhaustion_completed` (when supplied as `false`, persisted as `false`)

Values are truncated to secret-safe string limits; raw transcripts, prompts,
tool arguments, and API secrets must not be placed in these fields. LiteLLM does
not infer `worker_context_exhaustion_success` or `worker_context_exhaustion_completed`
from a non-empty assistant completion when the orchestrator supplied `false`.

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

Selected Grok Responses models declare
`custom_tool_function_adapters=["apply_patch"]` in the model metadata. For AAWM
Codex alias candidates only, LiteLLM converts the grammar-backed Codex
`apply_patch` custom-tool definition into an xAI-compatible function definition
with one required string field named `input`. Matching custom `tool_choice`,
`custom_tool_call`, and `custom_tool_call_output` continuation items are converted
to their function equivalents before provider-specific filtering and input-item
rewriting. Other custom tools remain unsupported and are still removed.

When the selected Grok candidate returns a matching `function_call`, LiteLLM
requires its arguments to be exactly one JSON string field named `input`, then
restores the client-visible item to `custom_tool_call` with the raw patch text.
Streaming responses are rebuilt with the restored item in
`response.output_item.done`, which is sufficient for Codex to dispatch the
client-hosted tool. Invalid or ambiguous arguments fail closed as a retryable
malformed candidate response so the alias can continue rather than exposing an
unusable function call to Codex.

OpenRouter completion-adapter candidates classify provider-wrapped 400 responses
with `metadata.provider_name` and `metadata.raw=ERROR` as terminal candidate
failures. Alias probes cool down only that OpenRouter candidate, record
`OPENROUTER_PROVIDER_RAW_ERROR` in attempt metadata, and continue to the next
declared candidate rather than surfacing an ASGI traceback to the client.
OpenRouter alias-probe 404 responses whose provider message indicates no
endpoint is available for the requested model (for example
`No endpoints found for openrouter/owl-alpha`) are classified as
candidate-unavailable during alias-probe dispatch. LiteLLM applies the normal
per-candidate cooldown, records the skipped attempt in alias metadata, and
continues to the next declared candidate instead of surfacing an ASGI traceback.
Non-alias OpenRouter 404s and unrelated not-found failures are not remapped and
still surface to the client normally.

Successful non-streaming OpenRouter completion-adapter requests also contribute
completed turns to the AAWM route rollup and suppress their matching native
`/openai_passthrough/responses?adapted_to=openrouter.ai/api/v1/chat/completions`
success access record.

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
- `codex_custom_tool_function_adapter_count`,
  `codex_custom_tool_function_adapter_names`,
  `codex_custom_tool_function_adapter_tools`,
  `codex_custom_tool_function_adapter_input_item_count`,
  `codex_custom_tool_function_adapter_input_items`, and
  `codex_custom_tool_function_adapter_tool_choice`: bounded evidence that an
  approved custom tool was converted to the provider-compatible function shape.
  These fields record names, indexes, item types, and counts, but never the raw
  patch contents.
- `codex_unsupported_input_item_removed_count`,
  `codex_unsupported_input_item_types_removed`, and
  `codex_unsupported_input_items_removed`: unsupported Responses input items
  removed before egress. For Grok-family routes this includes `reasoning` input
  items and records `encrypted_content=true` when an encrypted compaction blob
  was removed without logging the blob.
- `grok_native_input_item_rewrite_count`,
  `grok_native_input_item_rewrite_types`, and
  `grok_native_input_item_rewrites`: Grok-native Responses input items rewritten
  before egress because the selected concrete model rejects those typed
  Responses variants. For `grok-composer-2.5-fast` and Grok 4.5 native/OIDC
  routes, `function_call` and `function_call_output` continuation items are
  preserved as ordinary message text instead of being sent as raw Responses
  tool items. This avoids the Grok CLI proxy `ModelInput` deserialization
  failure while retaining the prior tool action and outcome as context. The
  preserved message text uses neutral context-note wording, not `Previous tool
  call`, `Name`, `Call ID`, or `Arguments` transcript fields, so Grok Composer
  is not primed to re-emit a non-executing tool-call template as assistant
  output.
- If Grok Composer still emits explicit literal tool-call blocks in assistant
  output, LiteLLM repairs only the bounded pattern with `Tool label:`,
  optional `Correlation ref:`, and JSON `Input payload:` fields whose arguments
  validate against the advertised tool schema. Repaired blocks become executable
  `function_call` output items, natural-language preface text remains assistant
  text, and ambiguous or invalid payloads still fail closed as malformed output.
- `anthropic_grok_native_prior_function_call_replay_dropped_count` and
  `anthropic_grok_native_prior_function_call_replay_dropped_items`: Anthropic
  Grok-native adapter requests drop prior assistant `tool_use` replay before
  Grok-native input-item rewriting. Paired tool-result observations remain
  available as user-side context, but prior tool calls are not sent as
  assistant-authored text because Grok Composer can copy that context forward as
  malformed tool-call output.
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

Known stale `/grok/v1/sessions/{id}/replicas/update` responses with upstream
`404 {"error":"Session not found or not owned"}` are treated as degraded
side-channel telemetry. LiteLLM returns the upstream status to the client, but
does not emit traceback-style active error intake for that known stale-session
shape.

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

Proxy-routed embedding requests should also retain the proxy-owned shared
aiohttp session when they reach `BaseLLMHTTPHandler` providers such as local
hosted embeddings, OpenRouter embeddings, Cohere-compatible embeddings, and
similar OpenAI-compatible embedding endpoints. The shared session remains
externally owned by the proxy; provider handlers may reuse it for async request
transport, but cleanup still happens through proxy shutdown and
`close_litellm_async_clients()`, not cache eviction.

`LLMClientCache` must not close HTTP or SDK clients when TTL eviction removes a
cache key; in-flight requests can still hold those clients. Evicted client-like
values are retained on the cache until `close_litellm_async_clients()` runs, so
they do not fall out of scope and emit unclosed aiohttp session warnings before
the explicit shutdown cleanup path can close them.

## Target Runtime Verifier

`scripts/verify_target_runtime_release.py` is a narrow target-aware verifier
for release and session-history correlation checks. It does not run sidecar,
Langfuse callback, or dashboard lanes.

- Choose `--target {dev,prod}` before any probe helpers run. Dev profiles
  describe the local bind-mounted `litellm-dev` runtime; prod profiles describe
  the released `aawm-litellm` image.
- `--target prod` is fail-closed: without `--release-runbook`, `--image-tag`,
  `--callback-wheel`, `--db-name`, and `source_mode=released_image`, the script
  refuses with a nonzero exit before Docker, SQL, HTTP, Langfuse, or `psycopg`
  import. Prod mutation is never allowed.
- The `session-history` lane (`--lane session-history`) proves persistence in
  `aawm_tristore.public.session_history`, not HTTP success alone. It requires
  `--marker-id` and at least one of `--session-id`, `--trace-id`, or
  `--litellm-call-id`, then fails if no matching row is found. Evidence reports
  row ids and attribution fields with raw `metadata` omitted/redacted.
- `--workspace-root` sets the active repository for provenance. Repeatable
  `--referenced-artifact-path` values (for example paths under
  `/home/zepfu/projects/<repo>/...`) are recorded as referenced artifact owners
  and do not override the active workspace repository.
