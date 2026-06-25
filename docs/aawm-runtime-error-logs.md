# AAWM Runtime Error Logs

Managed AAWM LiteLLM environments can mirror `ERROR`-level LiteLLM log records
into the local repo analysis intake directory. This is an operator workflow aid:
the durable queue remains `.analysis/todo.md`, while `*-error.jsonl` files are
short-lived structured intake artifacts that should be converted into TODO items
and then cleaned up after resolution.

During the migration window, agents should also inspect legacy
`.analysis/*-error.log` files. Those plain-text files are retained for discovery
only until all writers and intake instructions have moved to JSONL.

The provider-status observations sidecar can also append
structured anomaly rows to the same `<environment>-error.jsonl` convention when
`AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED=1`. Those rows come from the scheduled
session-history/rate-limit anomaly scan in
`scripts/run_provider_status_observations_loop.py`, not from LiteLLM `ERROR`
logging. See `docs/aawm-provider-status-observations.md` for the sidecar env vars
and scan behavior.

## Enablement

Set `LITELLM_AAWM_ERROR_LOG_ENABLED=1` to enable the sink. The target directory
defaults to `.analysis` under the current working directory and can be overridden
with `LITELLM_AAWM_ERROR_LOG_DIR`. The environment name in the filename comes
from `LITELLM_AAWM_ERROR_LOG_ENV`, then `LITELLM_LANGFUSE_TRACE_ENVIRONMENT`,
then `LITELLM_ENV` or `ENVIRONMENT`; unsafe characters are normalized.

For example, the dev proxy writes `/app/.analysis/dev-error.jsonl` inside the
container, mounted to this repo's `.analysis/dev-error.jsonl`. The matching
legacy path is `/app/.analysis/dev-error.log`, mounted to
`.analysis/dev-error.log`.

Malformed tool-call detections use the same local intake directory and append
to `.analysis/malformed-error.jsonl`. This file is intentionally separate from
`<environment>-error.jsonl` so investigations can inspect the raw malformed
tool-call text or payload without scraping container logs. Enable it with
`LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED=1`, or by enabling the generic AAWM
error sink or setting `LITELLM_AAWM_ERROR_LOG_DIR`. The malformed-specific
enable flag is enough to default the path to the current working directory's
`.analysis/malformed-error.jsonl`.

When the proxy or provider-status sidecar runs as root inside a container, a
plain bind-mount append would create root-owned active intake files on the host.
After each successful append, the JSONL writers therefore make a best-effort
metadata repair:

- the file owner/group defaults to the owner/group of the mounted target
  directory;
- `LITELLM_AAWM_ERROR_LOG_FILE_UID` and
  `LITELLM_AAWM_ERROR_LOG_FILE_GID` can explicitly override that owner/group;
- file mode is left as created by the process umask unless
  `LITELLM_AAWM_ERROR_LOG_FILE_MODE` is set to an octal value such as `0640`.

Ownership or mode repair failures are swallowed after the JSONL line is written.
Error-intake logging must never fail a client request or the sidecar scan loop.

## JSONL intake shape

The sink writes append-safe newline-delimited JSON. Each line is one complete
JSON object, so repeated runtime failures can be appended without rewriting the
file.

`malformed-error.jsonl` follows the same append-only JSONL convention, with one
row per detected malformed tool call. Rows include route/session identity
fields such as `provider`, `model`, `model_alias`, `route_family`, `endpoint`,
`upstream_url`, `repository`, `agent_name`, `agent_id`, `session_id`,
`trace_id`, and `litellm_call_id`, plus the bounded
`malformed_tool_call_text` or `malformed_tool_call_payload` that triggered the
detection. Individual text and payload fields are truncated to bounded previews;
the file keeps appending by default. Operators can set
`LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES` as an explicit safety ceiling if a
runtime needs to stop appending after a known size.

Each event should include at least:

- `schema_version`
- `environment`
- `observed_at`
- `logger`
- `level`
- `message`
- `traceback`, `traceback_text`, and `traceback_lines`
- `raw_text` with the original redacted error text
- `fingerprint` for grouping repeated failures
- `context` with route/provider metadata when available

Practical `context` fields to inspect are:

- `source`
- `container`
- `endpoint`
- `upstream_url`
- `provider`
- `model`
- `model_alias`
- `route_family`
- `status_code`
- `trace_id`
- `litellm_call_id`
- `callback_name`
- `callback_phase`
- `handler_branch`
- `langfuse_failure_class`
- `event_type`
- `worker_timeout_seconds`
- `queue_depth`
- `queue_maxsize`
- `coroutine_name`
- `worker_delivery_state`
- `aawm_passthrough_inbound_content_type`
- `aawm_passthrough_json_egress_content_type_removed`
- `aawm_passthrough_json_egress_content_type_removed_value`
- `aawm_passthrough_body_container_type`
- `aawm_passthrough_body_top_level_keys`
- `aawm_passthrough_input_container_type`
- `aawm_passthrough_input_item_count`
- `aawm_passthrough_input_item_type_counts`
- `aawm_passthrough_tool_count`
- `aawm_passthrough_tool_type_counts`
- `grok_side_channel`
- `grok_side_channel_endpoint_type`
- `grok_side_channel_endpoint_path_template`
- `grok_side_channel_request_content_type`
- `grok_side_channel_request_body_byte_length`
- `grok_side_channel_request_body_digest_source`
- `grok_side_channel_request_json_container_type`
- `grok_side_channel_request_array_length`

Sidecar anomaly intake lines use the same append-safe JSONL file convention but a
different event shape. Each detected anomaly class is written as one line with
`event=aawm_observability_anomaly`. Practical fields to inspect are:

- `anomaly_class`
- `anomaly_source`
- `lookback_hours`
- `row_count`
- `expected`
- `examples`
- `recommended_todo`
- `cleanup_requirement`

These records summarize telemetry consistency findings from recent database rows.
They are not traceback-style LiteLLM runtime failures and may omit logger,
level, traceback, and `context` fields that proxy error mirroring normally
includes.

Pass-through upstream failures populate this context from safe route metadata.
For both main pass-through exceptions and streaming chunk-processor exceptions,
the error record should include endpoint, upstream URL without query secrets,
provider, selected/requested model label, model alias, route family, status
code when available, trace id, and LiteLLM call id. The pass-through sink
intentionally omits raw request bodies, response bodies, headers, prompts, and
credentials.

For JSON pass-through egress, LiteLLM removes any inbound `content-type` header
before calling `httpx` with `json=...`, allowing `httpx` to set the correct JSON
content type for the serialized provider-bound body. This normalization is
limited to JSON egress. Raw-body passthrough and multipart forwarding preserve
or rebuild their own content-type handling as appropriate. When JSON egress
removes an inbound content type, the runtime error JSONL context records only
safe shape metadata such as the removed content-type value, provider-bound
top-level keys, input container type/counts, and tool type counts. It must not
record raw prompts, tool arguments, request bodies, credentials, or concrete
session identifiers.

Pass-through `httpx.TimeoutException` failures during upstream setup are
reported as status code `504` so alias wrappers can classify them as upstream
timeouts. Pre-first-byte `httpx.ConnectError` failures, including DNS
resolution failures, are retried on the same schedule and reported as status
code `503` if the retry budget is exhausted. Midstream streaming timeouts are
logged with the same safe context but without exception traceback once response
bytes have already been emitted. They remain terminal because the response has
already started and the request cannot be safely replayed.

## Pass-through hidden upstream retry

AAWM pass-through routes use a fixed hidden retry schedule for retry-safe
upstream failures before the first response byte is returned to the client. The
schedule is `5s, 15s, 30s, 60s, 120s` and applies to upstream
`500`/`502`/`503`/`504`/`529` statuses plus pre-first-byte transport
connectivity failures such as DNS resolution errors and request timeouts.

`429` rate-limit responses are not hidden-retried. They should surface to the
client with available retry/reset metadata so alias selection and quota
observation can make the next decision without burning extra quota. Expected
provider `429` responses are logged at warning level and still run failure
callbacks with sanitized context, but they do not create active
`.analysis/*-error.jsonl` exception-intake rows unless a separate callback or
logging failure occurs.

The direct Grok billing passthrough endpoint has one additional degraded
telemetry class. When `/grok/v1/billing` receives the known upstream `400`
timeout/cancel body (`The operation was cancelled` / `Timeout expired`), the
proxy logs a warning with `failure_kind=degraded_grok_billing_timeout` and does
not create traceback-style active error intake. Unexpected Grok billing `400`
bodies, auth failures, and provider `429` responses remain visible through
their normal failure paths.

The retry wrapper intentionally stops at the pre-first-byte boundary. Once a
streaming response has been handed to the client, midstream failures remain
terminal for that stream and are recorded through the streaming error context
path instead of replaying the request.

Post-first-byte upstream read timeouts also emit an explicit terminal stream
event to the client after any bytes already forwarded. The proxy preserves the
partial stream, appends a route-family-specific terminal failure chunk, and does
not pass the partial byte stream through the normal success-callback pipeline.
It records `failure_kind=streaming_upstream_read_timeout`,
`stream_failure_stage=stream_interrupted_after_first_byte`,
`stream_hidden_retry_safe=false`, and stream progress counters in both runtime
error intake and `litellm_params.metadata`. AAWM route status/rollup surfaces
the interruption as `Status: Failed` instead of leaving the turn looking
successful.

Any runtime path that emits a traceback is active error intake, independent of
the logger level that produced it. Expected degraded states should avoid
`exc_info=True` and write structured degraded-provider metadata instead. If a
`WARNING` or ASGI path still emits a traceback, it should be captured in the
environment's `*-error.jsonl` intake or changed so the traceback is no longer
emitted for expected behavior.

Routes with their own retry, cooldown, or alias-candidate progression set
`caller_managed_hidden_retry=True` so the shared pass-through wrapper does not
double retry. This includes Google/Antigravity adapter calls, OpenRouter adapter
calls, Codex/auto-agent candidate probes, Cursor lifecycle calls, and Grok
session side-channel mutation routes.

Alias candidate probes also pass their caller-managed transient upstream status
set into the pass-through layer. For current AAWM Codex/Anthropic alias probes,
upstream `500`/`502`/`503`/`504`/`529` responses are owned by alias candidate
progression, not by generic pass-through exception intake. The pass-through
layer preserves the upstream status for the alias wrapper, skips generic
traceback-style `.analysis/*-error.jsonl` emission for those handled statuses,
and lets the alias layer record candidate failure, cooldown, skipped candidates,
and final redispatch/no-candidate metadata. Direct non-probe routes still use
their normal logging behavior for exhausted transient upstream failures.

Alias route failure logging uses one canonical warning by default: the route
status line written through the AAWM route access logger (for example
`Status: Cooling Down` / `Failed` / `Exhausted`). Rollup buckets still record
zero-turn status updates for the same failure. Full structured
`AAWM_ALIAS_ROUTE` JSON audit lines are investigation-only and require
`AAWM_ALIAS_ROUTE_VERBOSE_JSON=1` (also accepts `true`, `yes`, `debug`, or
`verbose`). Healthy alias selection events remain suppressed unless
`AAWM_ALIAS_ROUTE_LOG_HEALTHY=1`.

`provider_terminal_error` and `candidate_unavailable` alias cooldowns are
durable per-candidate cooldowns because those outcomes are reusable across
requests. They prevent repeated retry/log spam for the same terminal or
unavailable candidate until the cooldown expires. Bare transient upstream
`500`/`502`/`503`/`529` failures remain request-local so temporary provider
instability does not suppress a route longer than the current progression
requires.

Successful or exhausted hidden retries add bounded metadata under
`litellm_params.metadata`, including
`aawm_passthrough_hidden_retry_attempts`,
`aawm_passthrough_hidden_retry_count`,
`aawm_passthrough_hidden_retry_final_outcome`, and, for transport failures,
`aawm_passthrough_hidden_retry_failure_classification`. These fields contain
attempt numbers, status/failure class, and wait seconds only; they must not
contain raw request bodies, response bodies, headers, prompts, tool arguments,
or credentials.

When a hidden retry budget is exhausted, the runtime error JSONL context uses
`failure_kind=transient_provider_connectivity` for DNS/connectivity failures
and `failure_kind=expected_upstream_capacity_or_internal` for retryable upstream
HTTP capacity/internal statuses. This keeps active error intake visible while
making the expected transient class machine-readable.

Intermediate hidden-retry attempts are progress events, not terminal runtime
failures, and are logged at `INFO`. When all attempts are exhausted for an
expected capacity, internal, timeout, or connectivity failure, the final
pass-through error remains visible at `ERROR` but is logged as a concise
terminal upstream failure instead of a repeated internal traceback. The JSONL
context includes `failure_kind`, `hidden_retry_final_outcome`,
`hidden_retry_failure_classification`, and `hidden_retry_count` so operators can
distinguish retry exhaustion from unexpected LiteLLM exceptions.

Grok session side-channel failures keep their 401/404 responses visible instead
of retrying or suppressing them. Their runtime error JSONL context copies only
safe scalar side-channel descriptors from passthrough metadata, such as
`grok_side_channel`, `grok_side_channel_endpoint_type`,
`grok_side_channel_endpoint_path_template`,
`grok_side_channel_request_content_type`,
`grok_side_channel_request_body_byte_length`,
`grok_side_channel_request_body_digest_source`,
`grok_side_channel_request_json_container_type`, and
`grok_side_channel_request_array_length`. It deliberately omits raw body values,
body digests, top-level key maps, auth headers, credential payloads, and concrete
session ids so stale-session and no-auth-context errors remain diagnosable
without leaking client state.

When `grok_side_channel_endpoint_path_template` is available, generic
`endpoint` and `upstream_url` values in JSONL context use the template path
instead of the concrete Grok session URL. For example, a failed
`/grok/v1/sessions/<id>/signals` request records
`/grok/v1/sessions/{session_id}/signals` and
`https://cli-chat-proxy.grok.com/v1/sessions/{session_id}/signals`.

Known Grok Build `/signals` auth-context `401` responses are a
separate degraded telemetry class. When the upstream body contains all of:
`Invalid or expired credentials`, `x_xai_token_auth=xai-grok-cli`, and
`no auth context`, LiteLLM still returns the upstream `401` to the client but
logs a warning with `failure_kind=degraded_grok_signals_auth_context` instead of
emitting traceback-style active error intake. This is intentional operational
policy: the side-channel call is often stale or missing native Grok CLI auth
context and should not keep reopening `.analysis/*-error.jsonl` for the same
fingerprint.

Known stale Grok Build `/sessions/{session_id}/replicas/update` `404` responses
with upstream body `{"error":"Session not found or not owned"}` are also treated
as degraded side-channel telemetry. LiteLLM still returns the upstream `404` to
the client, but logs a warning with
`failure_kind=degraded_grok_replicas_update_not_owned` instead of emitting
traceback-style active error intake.

Unexpected `/signals` `401` bodies, unexpected `/replicas/update` `404` bodies,
other Grok side-channel `401`/`404` failures, and any data-bearing side-channel
error that does not match a known degraded shape still use the normal failure
logging path and remain visible for triage.

When the Langfuse SDK background ingestion consumer emits its generic support
message (`Unexpected error occurred. Please check your request and contact
support: https://langfuse.com/support.`), LiteLLM keeps the original message and
adds structured context for triage:

- `source=langfuse_sdk`
- `callback_name=langfuse`
- `callback_phase=sdk_background_ingestion_upload`
- `langfuse_failure_class=langfuse_sdk_background_ingestion_upload_failure`
- `failure_kind=degraded_langfuse_sdk_background_ingestion`
- `langfuse_payload_size_state` when recent enqueue audit context is available
- compact payload-size fields such as `langfuse_total_size_bytes`,
  `langfuse_max_event_size_bytes`, `langfuse_generation_id`, and
  `langfuse_call_type`
- `recommended_operator_action` with an operator-facing diagnostic hint

Treat that class as telemetry-only degradation. Client requests should not fail
because of this callback error. Check Langfuse web/worker/blob-storage logs near
the same timestamp and use `langfuse_payload_size_state` to distinguish
near-limit, over-limit-before-enqueue, fit-failed-before-enqueue, and
enqueued-but-sdk-failed cases.

Repeated Langfuse SDK support-string rows with the same active error fingerprint
and stable Langfuse context (trace/generation/payload-state fields) are
coalesced in `AawmErrorLogFileHandler` for a bounded TTL. The first row is
always written to `*-error.jsonl`; duplicates inside the TTL are suppressed.
Non-Langfuse `ERROR` rows are unaffected.

Configure the TTL with
`LITELLM_AAWM_ERROR_LOG_LANGFUSE_SUPPORT_STRING_COALESCE_TTL_SECONDS`. The
default is `300` seconds.


When the async logging worker times out or fails while delivering a queued
callback coroutine, the JSONL record uses `source=logging_worker`. The
`callback_name` and `callback_phase` fields identify the active callback or
phase when known, while `event_type`, `coroutine_name`,
`worker_timeout_seconds`, `queue_depth`, `queue_maxsize`, and
`worker_delivery_state` describe the worker state. These diagnostics are
bounded scalar fields only; the worker metadata path must not include prompts,
headers, request or response bodies, tool arguments, OAuth tokens, API keys, or
raw metadata blobs.

Treat `worker_delivery_state=timed_out` as degraded telemetry. It should make
the successful provider response fail only if a separate client-facing path
already failed. Use the callback fields to decide whether the timeout belongs
to Langfuse, session history, another built-in callback, or a custom logger.

Use the structured fields to group and triage failures, but keep
`.analysis/todo.md` as the source of truth for active work.

## Agent intake workflow

At TODO-driven work startup, inspect:

- `.analysis/*-error.jsonl`
- legacy `.analysis/*-error.log`
- sidecar anomaly intake lines with `event=aawm_observability_anomaly`


For every active intake file, add or update a `.analysis/todo.md` entry for the
underlying resolution. Capture the environment, error detail, traceback context,
intended fix, acceptance evidence, and a cleanup requirement for the source
`*-error.jsonl` or legacy `*-error.log` file.

Do not mark an error-intake-driven item complete until the underlying error is
fixed, verification evidence is recorded, and the corresponding active JSONL or
legacy text intake file has been deleted or archived out of active intake.

## Redaction and local-only handling

Managed prod deployments must mount a writable repo-local analysis directory
into the container and set the sink explicitly, for example:

```text
LITELLM_AAWM_ERROR_LOG_ENABLED=1
LITELLM_AAWM_ERROR_LOG_ENV=prod
LITELLM_AAWM_ERROR_LOG_DIR=/app/.analysis
```

Without that writable mount, prod tracebacks remain visible only in container
logs and never become local `.analysis/prod-error.log` intake. The prod compose
mount should point at this repo's `.analysis` directory; sibling infrastructure
repositories should not keep a second durable queue for LiteLLM runtime errors.

The sink reuses LiteLLM's secret-redaction filter and records traceback context
without adding request bodies, prompt payloads, or tool arguments. Error-intake
files are local sensitive artifacts and must not be committed or pushed.

## Opt-in Diagnostic Payload Capture

Pass-through diagnostic payload capture is disabled by default. It is separate
from runtime error JSONL, `session_history`, provider-error observations, and
Langfuse metadata.

## Opt-in Langfuse input shape/hash summary

Langfuse event fitting still uses budget-aware head/tail truncation for oversized
`input` by default. For stronger privacy on large request bodies, enable:

```text
AAWM_LANGFUSE_INPUT_SHAPE_HASH_ONLY=1
```

Accepted truthy values include `1`, `true`, `yes`, and `on`. When enabled and an
event must be fitted because it exceeds the Langfuse fit target, oversized
`input` is replaced with a bounded `litellm_langfuse_input_summary` object
instead of retained raw head/tail text. The summary records a stable hash,
original and final byte sizes, container type, item counts, role counts,
content-block type counts, bounded shape-only head/tail samples, omitted counts,
and `raw_reconstruction` status. Dict keys are summarized with bounded
non-reconstructive descriptors (`key_index`, `key_hash`, `key_length`, and
category) rather than raw user-controlled key names. Identifier-like values such
as `id`, `call_id`, `tool_call_id`, and `name` are shape-only or hashed/length
only, not previewed. Non-primitive or user-controlled scalar values are not
rendered via raw `str(value)` previews. It must not include raw prompts, tool
arguments, tool names, local file paths or content, source snippets, API keys,
cookies, authorization headers, or raw key names.

`raw_reconstruction.source` is one of:

- `not_available_by_default` when no durable raw-body lookup is configured.
- `cold_storage_object_key` when generation metadata includes
  `cold_storage_object_key` and cold-storage retrieval is configured.
- `full_payload_capture_required` when pass-through full-payload capture is
  enabled; that remains a deliberate short-lived investigation opt-in, not
  routine telemetry.

This mode is not the same as per-call `mask_input` (which redacts Langfuse input
entirely) or raw request/response logging knobs (which send raw material to
observability sinks when explicitly enabled). Metadata compactors from D1-238
and D1-314 are unchanged.

Enable the scoped diagnostic manifest writer with:

```text
AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE=1
AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_DIR=/tmp/captures/diagnostic_payloads
```

At least one exact-match scope must also be set, otherwise no artifact is
written:

```text
AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_ROUTE_FAMILIES=codex_responses
AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_ENDPOINT_TEMPLATES=/grok/v1/sessions/{session_id}/signals
AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_TRACE_IDS=<trace-id>
AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_LITELLM_CALL_IDS=<litellm-call-id>
```

When multiple scope dimensions are set, all configured dimensions must match.
Scope values are comma-separated exact strings; regex, substring, and wildcard
matching are intentionally not supported. Use route family for broad but still
bounded investigations, and trace id or LiteLLM call id for one-off captures.

Artifacts are written as local JSON files under
`/tmp/captures/diagnostic_payloads` by default and chmod'd to `0600`. Each
artifact contains a manifest with environment, route family, endpoint template,
trace id, LiteLLM call id, redaction mode, byte counts, hashes, and explicit
omitted-field descriptions. The artifact keeps sanitized request/response
shape data and selected safe header values only. It must not include raw
headers, raw request bodies, raw response bodies, stream bytes, prompts, tool
arguments, OAuth tokens, API keys, cookies, concrete Grok session ids, or local
file content.

The legacy full-payload pass-through capture remains a stronger separate
operator opt-in through `AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS` or its runtime
control file. That mode intentionally persists raw upstream request/response
headers and bodies for short-lived manual investigations and should not be used
as default telemetry.

Google Code Assist / Antigravity bootstrap preflight calls in
`litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` also call
`capture_passthrough_shape()` directly for `v1internal:loadCodeAssist` and the
`retrieveUserQuota` / `fetchAdminControls` / `listExperiments` prime path. Those
direct preflight captures use the same exact-scope gate as other diagnostic
manifest writes: they stay local-only under
`/tmp/captures/diagnostic_payloads` by default and persist shape/hash manifest
data only unless the separate full-payload capture opt-in is enabled.

Native `/rerank` proxy requests use a separate rerank diagnostic manifest
wrapper rather than the pass-through HTTP response helper. The route family is
`rerank`, the endpoint template is `/rerank`, and the artifact records only
request/response shape, byte counts, hashes, and omitted-field descriptions.
Rerank query text, input documents, and returned document text are private
content and must remain summarized as shapes unless a separate full-payload
capture mode is explicitly added and enabled for a short-lived investigation.

Diagnostic manifest coverage by route family:

- OpenAI, Anthropic, Gemini, Vertex, Cohere, and Cursor pass-through HTTP
  handlers use the shared pass-through capture hooks for nonstreaming,
  streaming, and error-shape manifests.
- Google Code Assist / Antigravity bootstrap preflight calls use direct
  `capture_passthrough_shape()` calls because they run before the normal shared
  pass-through response handler.
- Native `/rerank` proxy calls use the rerank wrapper described above because
  they flow through `base_process_llm_request(route_type="arerank")`, not the
  pass-through HTTP stack.
- AssemblyAI transcript polling is intentionally not captured per poll. The
  initial pass-through request remains covered by the shared hooks; terminal
  transcript capture should be added only with a transcript-specific manifest
  that emits once per completed transcript and keeps transcript text summarized.
- Vertex AI Live websocket sessions are intentionally not forced into the HTTP
  diagnostic manifest shape. They need a websocket lifecycle manifest with
  explicit open/message/close boundaries before capture should be enabled.

## ChatGPT Codex quota errors

When ChatGPT Codex passthrough returns HTTP 429 with
`error.type = usage_limit_reached`, LiteLLM reshapes the client-facing error
into a structured `rate_limit_error` instead of returning a raw upstream byte
string. The detail preserves upstream quota fields such as `plan_type`,
`resets_at`, `resets_in_seconds`, and `eligible_promo`, adds
`retry_after_seconds`, and sets the `Retry-After` header from the reset data.

Treat this class as upstream account quota exhaustion. It is distinct from
short-lived high-demand throttling, and alias handlers should classify it as
`usage_limit_reached` when deciding whether a fresh dispatch can advance to the
next declared candidate.

## ChatGPT Codex encrypted continuations

When ChatGPT Codex passthrough returns HTTP 400 with
`error.code = invalid_encrypted_content`, LiteLLM treats the response as a known
client/session continuation failure. This usually means the request carried
encrypted reasoning or continuation content that the ChatGPT Codex backend could
not decrypt or parse for that session.

LiteLLM does not retry this class and does not hide the upstream 400 from the
client. It logs a compact warning with
`failure_kind=openai_chatgpt_codex_invalid_encrypted_content`, sends
`traceback_str=None` to failure hooks, and avoids writing an active traceback
entry to the runtime error JSONL file for this known provider/client condition.
Unrelated ChatGPT Codex 400 responses still use the normal exception and error
intake path.
