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

## Agent Identity Capture

`public.session_history` and `public.session_history_tool_activity` include a
nullable `agent_id` text column. This is an opaque client-provided dispatch,
task, or subagent identifier. It is separate from `agent_name`, which remains the
stable human-readable role or display name such as `orchestrator`, `researcher`,
or `Planck`.

The callback accepts explicit ID fields from bounded metadata/body/header sources
such as `agent_id`, `aawm_agent_id`, `subagent_id`, `task_id`,
`x-aawm-agent-id`, `x-grok-agent-id`, `x-litellm-agent-id`, and `x-agent-id`.
Values that match the session id, trace id, repository, tenant, or human
`agent_name` are rejected rather than copied into `agent_id`.

Codex main-session rows default `agent_name` to `orchestrator` when the request
is a native Codex passthrough and no explicit role/name is available. Child or
subagent rows should only use a child display name when the request or transcript
provides one. Rows without a reliable opaque id keep `agent_id = NULL`.

Historical rows written before this field existed may be `NULL` unless they are
backfilled from a trustworthy source. Synthetic Codex transcript rows use the
transcript session metadata id as `agent_id` when available and mark
`metadata.agent_id_source = codex_transcript.session_meta.id`.

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

Grok monthly billing payloads populate `quota_limit`, `quota_used`,
`quota_remaining`, both billing-period boundary columns, and a raw copy of the
sanitized `monthlyLimit`, `used`, `onDemandCap`, and period fields. xAI OAuth
rate-limit headers populate absolute request/token amounts and carry billing
period ends when the provider config or managed subscription context exposes
one.

## Session History Outage Spool

`session_history` rows are normally written directly to
`aawm_tristore.public.session_history`. If a batch flush keeps failing after the
configured retry budget, LiteLLM writes the exact batch payload to a local JSON
spool for later replay instead of logging full tracebacks indefinitely.

The default spool directory is `/mnt/e/litellm/session_history`; override it with
`AAWM_SESSION_HISTORY_SPOOL_DIR` when a deployment needs a different local
volume. Failed flushes retry every `AAWM_SESSION_HISTORY_FAILED_FLUSH_RETRY_SECONDS`
seconds, and `AAWM_SESSION_HISTORY_FAILED_FLUSH_MAX_RETRIES` controls how many
retry attempts happen before the batch is atomically written to the spool.

Spool filenames include a UTC timestamp and a trace/session/call identifier so
operators can find the affected dataset without printing prompt, tool, or
metadata payloads into logs. Writes use a same-directory temporary file followed
by replacement, so completed `.json` files are replayable and partial writes do
not look like ready datasets.

The in-process drainer checks for existing spool files when the callback starts
and after successful writes, so records left behind by a prior outage can replay
without waiting for a new failed batch. Replay is at-least-once. The database
insert path remains idempotent, so recovery tools and the in-process drainer
should assume duplicates are possible after a crash or partial outage. The spool
contains local sensitive session metadata and must stay out of git, shared logs,
and external artifacts unless it has been reviewed and sanitized.

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
next configured usable candidate. The Codex `aawm-code` alias keeps
`gpt-5.3-codex-spark` as its normal OpenAI/Codex Responses candidate and uses
`gpt-5.5` as the OpenAI last-resort candidate with medium reasoning applied by
default when the inbound request did not already provide a reasoning setting.
Plain `gpt-5.3-codex` is intentionally not an `aawm-code` ChatGPT-account
Codex Responses candidate because that account surface rejects it before work
can start. For tool-bearing or stateful
`aawm-code-anthropic` requests, every declared candidate route is treated as a
Claude Code tool-contract route: if the alias declares Antigravity, OpenAI, xAI,
native Anthropic, or another provider/model target, that target must preserve
tool calls, tool-use ids, tool arguments, tool-result replay, and ordered
failover metadata for engineering-agent traffic. A declared candidate must not
be skipped as policy-incompatible for the same request class it is meant to
serve. If a declared candidate mishandles tools, that is an adapter translation
defect to fix or evidence for removing/reclassifying the candidate; it is not a
selector-side compatibility decision.

## Antigravity Native OAuth Credentials

Antigravity Code Assist routes use a LiteLLM-managed OAuth credential file for
runtime access-token refreshes. The Antigravity CLI credential is treated as a
seed, not as the long-term refresh target.

The seed path comes from `LITELLM_ANTIGRAVITY_SEED_AUTH_FILE`, then legacy
`LITELLM_ANTIGRAVITY_AUTH_FILE`, then the Antigravity CLI default token path.
The managed path comes from `LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE` and defaults
to `~/.litellm/antigravity/antigravity-oauth-token`.

When the seed credential is newer than the managed credential, LiteLLM copies
the seed into the managed file and invalidates the cached access token. Direct
OAuth refreshes then write the new token data back to the managed file only. If
direct refresh fails because the local OAuth client values are stale, LiteLLM can
invoke `agy models` against the explicit seed path, reload the CLI-refreshed
seed credential, and persist that refreshed data back into the managed file on
the normal access-token load path.

Prod and dev containers should mount the seed credential read-only when possible
and mount the managed credential directory writable. Do not configure a single
read-only Antigravity token file as the managed refresh target; doing so can
leave the route pinned to stale credentials or make refresh persistence fail.

## Grok Native OIDC Credentials

`xai/grok-composer-2.5-fast`, `xai/grok-build`, and
`xai/grok-build-0.1` use the Grok native OIDC credential path, not the managed
`oa_xai/*` OAuth credential file. In `litellm-dev`, `LITELLM_XAI_GROK_AUTH_FILE`
defaults to `/home/zepfu/.litellm/xai/grok-auth.json`, while the personal Grok
CLI credential at `/home/zepfu/.grok/auth.json` is mounted read-only and exposed
only as `LITELLM_XAI_GROK_SEED_AUTH_FILE`.

The Grok native refresh path updates the configured credential by writing a
temporary file beside it and then atomically replacing the original file. The
configured refresh target must therefore live on writable LiteLLM-owned storage.
Mounting the configured target read-only makes Composer and other Grok native
candidates fail as `candidate_unavailable` during token refresh, which breaks the
declared alias failover order. Keep `grok-auth.json` separate from the managed
`oa_xai/*` `oauth-auth.json` file so the two credential families remain
auditable.

When the read-only Grok CLI seed credential is newer than the LiteLLM-owned
managed credential, LiteLLM replaces the managed Grok credential from the seed
before selecting or refreshing an access token. This lets a fresh Grok/OIDC login
take effect without writing back into `/home/zepfu/.grok`.

Grok native and `oa_xai/*` Responses candidates remove request fields and hosted
tools that the selected Grok-family model declares unsupported. Direct native
Grok passthrough preserves Responses `reasoning` input items that carry
`encrypted_content`; those encrypted items are treated as provider-owned
compacted session state and must round-trip unchanged rather than being removed
as ordinary unsupported reasoning summaries. Adapter and alias-failover routes
strip provider-bound compacted session state before xAI/Grok egress because a
`previous_response_id` or encrypted reasoning blob from another route/session
can trigger upstream `Could not decode the compaction blob` failures.
If a Grok-family upstream still rejects a compacted request with `Could not
decode the compaction blob`, alias-probe mode classifies that 400 as
candidate-unavailable so the declared failover sequence can continue instead of
stranding the worker on the rejecting candidate.

## Access Log Display Semantics

AAWM route logs emit a compact route line to the LiteLLM proxy logger after the
incoming request has enough local context to identify the route and, when
available, the egress target. Pass-through requests and general proxy requests
such as embeddings use the same formatter and native access-log suppression
path.
The display shape is:

```text
YYYYMMDD HH:MM:SS TYPE [Client/Version -] [agent[#agent_id]@repository.]model(alias) METHOD ip:port incoming -> outgoing
```

The timestamp uses a 24-hour clock. `TYPE` is normally `ROUTE`, with lighter
`EMBED` and `RERANK` forms for embedding/rerank traffic. TUI clients are
normalized to common product names such as `Claude`, `Codex`, and `Grok`; other
client products keep their parsed product name/version. Agent, agent id,
repository, model alias, and client address segments are omitted when that
metadata is unavailable or not relevant. Route logs print `#agent_id` only when
a real client-provided opaque id is available before egress; they do not
synthesize an agent id from `session_id`. The selected model is printed as
`model(alias)` only when the inbound alias differs from the selected model.
For TUI route traffic with a known repository but no explicit agent name,
LiteLLM uses the same `orchestrator` default as `session_history`, so those rows
display as `orchestrator@repository.model(alias)` instead of `repository.model`.
Embedding and rerank route logs intentionally omit agent/repository identity
unless the model label itself carries the needed operational context.

These lines are for live container-log triage, not durable reporting. Durable
model and alias attribution still comes from `session_history.model` and
`session_history.inbound_model_alias`. Route-log identity fields are conservative
display tokens derived from normalized metadata or explicit headers such as
`x-aawm-client-name`, `x-aawm-client-version`, `user-agent`,
`x-aawm-agent-name`, `x-aawm-agent-id`, and `x-aawm-repository`; LiteLLM omits
prompt-like, sentence-like, or punctuation-heavy identity values instead of
printing raw request text. Repository lookup intentionally mirrors the
`session_history` callback's safe source order where route logging has access to
the same data: explicit request metadata, nested `litellm_metadata`, explicit
headers, request-body workspace/cwd fields, and the bounded request-header
tenant fallback used for repository-like tenant ids. If no repository token is
available from those sources, LiteLLM may infer only the repository label from
bounded, known workspace fields in the parsed request body, including
`workspace_root`, `project_root`, `working_directory`, `cwd_path`, `cwdUri`,
`request.metadata.repository`, and Claude/Codex cwd markers such as
`<cwd>...</cwd>`. Body-derived repository values are normalized to a slug or
`owner/repo` label, and worktree paths are trimmed back to the repository root
before logging. LiteLLM does not print raw prompt text or raw tool arguments for
the route log.

Route logs must not include API keys, authorization headers, full request or
response bodies, prompt content, tool arguments, or arbitrary query strings.
Incoming endpoints preserve only known-safe routing query markers, and outgoing
targets are logged as host plus path.

For requests that emit this enriched route line, LiteLLM suppresses the
matching native Uvicorn/Gunicorn access record for that request. Unrelated
routes should continue to use the normal server access log.

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

For AAWM Codex aliases, hosted-tool support is evaluated again after the alias
has selected a concrete xAI/Grok candidate such as `grok-composer-2.5-fast` or
`oa_xai/grok-build`. This catches provider-invalid Codex tool variants, including
`custom`, that could not be classified while the inbound request model was still
the abstract alias `aawm-code`.

Grok Composer and Grok Build candidates also drop unsupported reasoning-effort
fields before egress, including `reasoning`, `reasoning_effort`,
`reasoningEffort`, and nested `output_config.effort`. If xAI still rejects an
alias-probe request with an unsupported reasoning-parameter 400, LiteLLM treats
that target as candidate-unavailable so the declared alias sequence can continue
to the next candidate instead of terminating the agent dispatch on the provider
400.

Adapter-managed xAI/Grok requests also remove provider-bound compacted
continuation state before egress. This covers `previous_response_id` and
Responses `reasoning` input items that contain `encrypted_content` when the
request is being prepared for managed `oa_xai/*` or Grok native OAuth adapter
routes. Direct `/grok` passthrough is excluded so native Grok clients can keep
their own valid compact response state.

Rows affected by this path may include:

- `codex_unsupported_hosted_tool_removed_count`,
  `codex_unsupported_hosted_tool_types_removed`, and
  `codex_unsupported_hosted_tools_removed`: Codex hosted-tool definitions removed
  because the selected concrete model marks them unsupported. For xAI/Grok
  Responses routes this commonly includes `custom`, `image_generation`,
  `namespace`, or `tool_search`.
- `codex_unsupported_hosted_tool_choice_removed`: the removed `tool_choice` when
  it referenced a hosted tool removed by the selected model policy.
- `xai_responses_request_sanitized`: `true` when the outgoing request body was
  changed before xAI egress.
- `xai_responses_sanitized_removed_params`: normalized top-level field names
  removed from the request, for example `["instructions", "metadata"]`.
- `xai_responses_sanitized_tool_count`: number of tool definitions whose
  outbound shape changed.
- `xai_responses_sanitized_tool_types`: normalized tool types whose outbound
  shape changed, for example `["code_interpreter", "web_search", "x_search"]`.
- `xai_responses_sanitized_tools`: bounded detail records keyed by tool index,
  type, and removed fields where available.
- `xai_adapter_compaction_state_removed`: `true` when adapter/failover
  preparation removed provider-bound compacted continuation state before
  xAI/Grok egress.
- `xai_adapter_compaction_state_removed_fields`: normalized field names removed
  from the outgoing request, for example `["input", "previous_response_id"]`.
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

The native Grok privacy side-channel endpoint
`/v1/privacy/coding-data-retention` requires `codingDataRetentionOptOut`.
LiteLLM preserves explicit client values and supplies `true` when the field is
omitted, so the CLI privacy probe does not fail open or produce recurring 422
noise.

Sanitization metadata proves request adaptation only. It does not prove tool
execution, model tool-use quality, or upstream success by itself; combine it
with status, token, cost, and error fields when building reports.

## OpenCode Zen Codex Tool-Adjacency Sanitization

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
