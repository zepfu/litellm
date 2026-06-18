# AAWM Runtime Error Logs

Managed AAWM LiteLLM environments can mirror `ERROR`-level LiteLLM log records
into the local repo analysis intake directory. This is an operator workflow aid:
the durable queue remains `.analysis/todo.md`, while `*-error.jsonl` files are
short-lived structured intake artifacts that should be converted into TODO items
and then cleaned up after resolution.

During the migration window, agents should also inspect legacy
`.analysis/*-error.log` files. Those plain-text files are retained for discovery
only until all writers and intake instructions have moved to JSONL.

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

## JSONL intake shape

The sink writes append-safe newline-delimited JSON. Each line is one complete
JSON object, so repeated runtime failures can be appended without rewriting the
file.

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
- `grok_side_channel`
- `grok_side_channel_endpoint_type`
- `grok_side_channel_endpoint_path_template`
- `grok_side_channel_request_content_type`
- `grok_side_channel_request_body_byte_length`
- `grok_side_channel_request_body_digest_source`
- `grok_side_channel_request_json_container_type`
- `grok_side_channel_request_array_length`

Pass-through upstream failures populate this context from safe route metadata.
For both main pass-through exceptions and streaming chunk-processor exceptions,
the error record should include endpoint, upstream URL without query secrets,
provider, selected/requested model label, model alias, route family, status
code when available, trace id, and LiteLLM call id. The pass-through sink
intentionally omits raw request bodies, response bodies, headers, prompts, and
credentials.

Pass-through `httpx.TimeoutException` failures during upstream setup are
reported as status code `504` so alias wrappers can classify them as upstream
timeouts. Pre-first-byte `httpx.ConnectError` failures, including DNS
resolution failures, are retried on the same schedule and reported as status
code `503` if the retry budget is exhausted. Midstream streaming timeouts are
logged with the same safe context and traceback, but remain terminal because
response bytes may already have been sent to the client.

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

Routes with their own retry, cooldown, or alias-candidate progression set
`caller_managed_hidden_retry=True` so the shared pass-through wrapper does not
double retry. This includes Google/Antigravity adapter calls, OpenRouter adapter
calls, Codex/auto-agent candidate probes, Cursor lifecycle calls, and Grok
session side-channel mutation routes.

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

When the Langfuse SDK background ingestion consumer emits its generic support
message (`Unexpected error occurred. Please check your request and contact
support: https://langfuse.com/support.`), LiteLLM keeps the original message and
adds structured context for triage:

- `source=langfuse_sdk`
- `callback_name=langfuse`
- `callback_phase=sdk_background_ingestion_upload`
- `langfuse_failure_class=langfuse_sdk_background_ingestion_upload_failure`

Treat that class as a Langfuse ingestion/upload failure signal. Check the
Langfuse web/worker/blob-storage logs near the same timestamp before assuming
the LiteLLM callback wrapper itself failed.

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
