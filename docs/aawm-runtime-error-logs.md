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
