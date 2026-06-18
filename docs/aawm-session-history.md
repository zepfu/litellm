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
still report `spool_pending=0`. Replay is at-least-once. The database
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
next configured usable candidate. For tool-bearing or stateful
`aawm-code-anthropic` requests, every declared candidate route is treated as a
Claude Code tool-contract route: if the alias declares Antigravity, OpenAI, xAI,
native Anthropic, or another provider/model target, that target must preserve
tool calls, tool-use ids, tool arguments, tool-result replay, and ordered
failover metadata for engineering-agent traffic. A declared candidate must not
be skipped as policy-incompatible for the same request class it is meant to
serve. If a declared candidate mishandles tools, that is an adapter translation
defect to fix or evidence for removing/reclassifying the candidate; it is not a
selector-side compatibility decision.

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

Keep the Grok CLI/OIDC credential separate from the managed `oa_xai/*`
`oauth-auth.json` file. Managed `oa_xai/*` routes still use LiteLLM-owned OAuth
refresh/write behavior, while native Grok routes rely on the sidecar-maintained
CLI credential. The explicit Grok billing poll should run from the same
sidecar context so quota snapshots use the credential that the sidecar keeps
fresh, not a second LiteLLM-managed copy.

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
