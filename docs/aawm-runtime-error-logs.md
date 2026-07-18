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

### Writer offload, rotation, and content-field policy

The generic LiteLLM ERROR sink (`AawmErrorLogFileHandler` in
`litellm/_logging.py`) is opt-in only. When enabled:

- **Offloaded writes**: caller threads build a redacted JSONL line and enqueue it
  on a bounded background writer so ERROR bursts do not block request handlers
  under a process-wide disk lock.
- **Queue capacity**: `LITELLM_AAWM_ERROR_LOG_QUEUE_MAXSIZE` (default `1024`).
  When the queue is full, additional records are **dropped** rather than
  blocking. Dropped totals are available via
  `AawmErrorLogFileHandler.dropped_record_count()` and
  `get_aawm_error_log_dropped_record_count()`. A throttled `LiteLLM` warning is
  emitted at most once per minute while drops continue.
- **Size rotation**: `LITELLM_AAWM_ERROR_LOG_MAX_BYTES` (default `10485760`,
  10 MiB) and `LITELLM_AAWM_ERROR_LOG_BACKUP_COUNT` (default `5`) control
  `RotatingFileHandler` behavior for `<environment>-error.jsonl`.
- **Default context is structural/metadata-only**. Body-preview and other
  content-bearing fields, currently
  `aawm_passthrough_request_shape_error_body_preview` and
  `grok_side_channel_request_body_digest_source`, are excluded by default.
  Set `LITELLM_AAWM_ERROR_LOG_INCLUDE_CONTENT_FIELDS=1` only when operators
  explicitly accept those fields on local intake disks. Regex secret redaction
  still applies, but it cannot guarantee removal of arbitrary prompt or PII
  content from previews.
- **Langfuse SDK support-string diagnostics** attach to the `langfuse` logger
  independent of callback registration order and whether JSON logging came from
  the `JSON_LOGS` environment variable or `litellm_settings.json_logs`.

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

Terminal agent-session errors use a dedicated structured writer that appends to
the normal `<environment>-error.jsonl` file under the same local intake
directory (`dev-error.jsonl` in dev and `prod-error.jsonl` in prod). Enable it
explicitly with `LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED=1` (or
`true`/`yes`/`on`). If that flag is unset, the writer defaults on when either
`LITELLM_AAWM_ERROR_LOG_DIR` is set or the generic sink is enabled via
`LITELLM_AAWM_ERROR_LOG_ENABLED=1`. Explicitly setting
`LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED=0` (or `false`/`no`/`off`)
disables terminal-agent intake even when the generic sink remains enabled.
Terminal agent rows honor the shared
`LITELLM_AAWM_ERROR_LOG_MAX_BYTES` ceiling (default `10485760` / 10 MiB when
unset or non-positive) and refuse an append when the projected encoded line
would cross that size. Malformed-tool rows use a separate ceiling via
`LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES` with the same 10 MiB default.
Neither writer truncates or rewrites existing intake to make room.

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

### Repo-local `.analysis` mounts for non-root sidecars

Some non-root third-party containers (for example Langfuse `web` and `worker`)
append runtime JSONL through the AAWM container wrapper with
`AAWM_ERROR_JSONL_DIR=/aawm-analysis`, bind-mounting this repo's `.analysis`
directory at `/aawm-analysis`. The mount must be writable by the container's
effective user or one of its groups, not only by the host checkout owner.
Root-owned LiteLLM metadata repair does not apply to these wrapper paths.

On hosts where the container user has supplemental group id `1000` (common when
the process runs as a fixed non-root uid with gid `1000` in the image), the
host `.analysis` directory typically needs group-write permission (for example
mode `2775` or `775` with group matching that supplemental gid) so append and
create of `<container-name>-error.jsonl` succeed.

Verify with an in-container probe: a non-destructive write test under
`/aawm-analysis`, or confirming the wrapper can open the intended `*-error.jsonl`
without `permission denied` in container logs.

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
detection. The text detector treats literal tool-call transcript blocks, such
as `composer_call` markers or `Tool label:` / `Input payload:` output, as
malformed when they appear in an assistant final response instead of an
executable tool call. Individual text and payload fields are truncated to
bounded previews, including string-style `message.content` and list-style
`message.content[].text` Responses outputs.

Malformed-tool intake bounds:

- `LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_ITEMS` defaults to `8`.
  `persist_malformed_tool_call_detection()` always applies this per-response
  evidence bound, and configured or direct-call values are hard-clamped to `64`.
- Multi-evidence responses are encoded and appended under one lock/file open
  through `append_malformed_tool_call_detections()`, rather than one filesystem
  cycle per evidence item.
- `LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES` defaults to 10 MiB. The entire
  pending batch is refused when it would cross the ceiling, preserving existing
  unresolved rows.
- Async request paths use
  `schedule_persist_malformed_tool_call_detection()`, which schedules
  `asyncio.to_thread(...)`; synchronous callers persist inline.

When literal tool-call blocks match the explicit Grok repair pattern and their
arguments validate against the advertised tool schema, LiteLLM repairs them into
executable `function_call` items instead of immediately logging a malformed
tool-call failure. The repair also strips provider tool-end markers such as
`<|tool_call_end|>` and fullwidth Grok marker variants before emitting the tool
call, so repaired rows no longer emit
`aawm_auto_agent_malformed_tool_call_text`.

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
- `aawm_passthrough_request_shape_error_class`
- `aawm_passthrough_request_shape_error_message_class`
- `aawm_passthrough_request_shape_error_body_preview`
- `aawm_passthrough_request_shape_summary`
- `aawm_passthrough_request_shape_fingerprint`
- `aawm_passthrough_request_shape_error_fingerprint`
- `grok_side_channel`
- `grok_side_channel_endpoint_type`
- `grok_side_channel_endpoint_path_template`
- `grok_side_channel_request_content_type`
- `grok_side_channel_request_body_byte_length`
- `grok_side_channel_request_body_digest_source`
- `grok_side_channel_request_json_container_type`
- `grok_side_channel_request_array_length`
- `aiohttp_owner_kind`
- `aiohttp_creation_site`
- `aiohttp_litellm_owns_session`
- `aiohttp_session_id`
- `aiohttp_connector_id`
- `aiohttp_event_loop_id`
- `aiohttp_pid`
- `aiohttp_container_hostname`
- `aiohttp_context_keys`
- `aiohttp_context_resource`

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

## Asyncio aiohttp lifecycle attribution

When the asyncio loop emits a non-exception context that includes an aiohttp
`client_session` or connector, LiteLLM logs the row as an `ERROR` level runtime
intake row with safe lifecycle-attribution fields in `context`.

LiteLLM-owned aiohttp sessions are tagged when they are created or provided to
the custom aiohttp handler. If a warning arrives for an untagged resource, the
row still records `aiohttp_owner_kind=unattributed` plus the resource id, pid,
hostname, loop-context keys, and whether the loop context named a
`client_session` or `connector`. This makes future leak attribution possible
without guessing ownership for older or third-party sessions.

Observed session fields are:

- `aiohttp_owner_kind`
- `aiohttp_creation_site`
- `aiohttp_litellm_owns_session`
- `aiohttp_session_id`
- `aiohttp_connector_id`
- `aiohttp_event_loop_id`
- `aiohttp_pid`
- `aiohttp_container_hostname`
- `aiohttp_context_keys` (the names of keys present in the loop context)
- `aiohttp_context_resource` (`client_session` or `connector`)

These lifecycle rows intentionally omit prompt text, headers, credentials,
connector internals, and raw tracebacks when no exception object is present.



## Pass-through Responses request-shape 422 instrumentation

When `/openai_passthrough/responses` returns HTTP `422` because the upstream
could not deserialize the provider-bound JSON body (for example `ModelInput` or
unknown tool/input variant errors), LiteLLM classifies the failure as
`failure_kind=request_shape_deserialization_failed` and adds sanitized
request-shape fields to the runtime error JSONL `context`.

The instrumentation is detection-only for the request body: LiteLLM still
returns the upstream `422` to the caller and does not retry or rewrite the
request body on this path. When the failure is tied to an AAWM alias or carries
agent/dispatch identity, the same classification also emits a structured
terminal-agent JSONL row (see `Terminal agent error JSONL intake` below) with
bounded request-shape fields and agent/session correlation when available.
Unattributed direct Responses `422` requests keep normal failure-hook handling
but do not create terminal-agent intake.

Recorded fields include:

- `aawm_passthrough_request_shape_error_class`
- `aawm_passthrough_request_shape_error_message_class` such as
  `model_input_deserialization_failed`, `unsupported_variant_deserialization_failed`,
  or `generic_deserialization_failed`
- `aawm_passthrough_request_shape_error_body_preview` (bounded upstream error text)
- `aawm_passthrough_request_shape_summary` (container/top-level keys, input/tool counts)
- `aawm_passthrough_input_item_shape_samples` (bounded head/tail samples containing
  only item index, container type, item type, and sorted key names)
- `aawm_passthrough_request_shape_fingerprint` (stable grouping key for request shape)
- `aawm_passthrough_request_shape_error_fingerprint` (stable grouping key for error class)

These records intentionally omit raw request bodies, prompts, tool arguments,
authorization headers, and API keys. Use the fingerprints to group repeated
same-profile worker dispatch failures before starting a payload rewrite or route
fix.

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

Adapted Google Code Assist streams also own cleanup of their nested event
parser and source body iterator. If a downstream client stops consuming the
stream early, both iterators are closed before teardown; this releases upstream
resources without synthesizing a terminal event or replaying the request.

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

Native Grok 4.5 is the exception for generic candidate-availability probe
failures: `aawm_codex_auto_agent_candidate_unavailable` does not create a timed
cooldown for that live route. Explicit usage, quota, rate-limit, or capacity
signals still use the normal cooldown and fallback path.

`malformed_tool_call_text` is also a durable per-candidate cooldown path. For
non-spark candidates it resolves to a 30-minute cooldown; Spark candidates keep
their existing five-minute durable override for this failure class.

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

Grok/xAI CLI passthrough `403` responses whose upstream body contains
`personal-team-blocked:spending-limit` are treated as upstream account quota
exhaustion, not generic permission failures. LiteLLM preserves the upstream `403`
response for callers, logs a warning with
`failure_kind=upstream_grok_account_quota_exhaustion`, and avoids repeated full
traceback logging for this known account-limit class. Unrelated Grok `403` bodies
still use the normal failure logging path.

Grok/xAI CLI passthrough `402` responses whose upstream body is exactly
`{"error":"Grok Build usage balance exhausted"}` (or equivalent JSON with that
error string) are treated the same way: upstream account/provider quota exhaustion
for xAI passthrough targets only. LiteLLM preserves the upstream `402` for callers,
logs a warning with `failure_kind=upstream_grok_account_quota_exhaustion` without a
traceback, and does not emit active `.analysis/*-error.jsonl` intake for this known
quota shape. Unrelated Grok `402` bodies still use the normal failure logging path.
For `aawm-code` and `aawm-code-anthropic` alias routes, the same known Grok Build
`402` body is also classified as retryable provider/account quota exhaustion: the
exhausted Grok candidate gets a durable cooldown, non-in-flight requests fall through
to the next alias candidate when one is available, and in-flight sessions return
`redispatch_required` with attempt metadata including `error_status_code`,
`failure_class`, and `error_tokens`.

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

## Terminal agent error JSONL intake

Some pass-through and AAWM alias failures terminate a spawned agent session
before useful work or a final response. Those outcomes can write structured
rows through `litellm.proxy.aawm_runtime_error_logging.persist_agent_terminal_error`
into `.analysis/<environment>-error.jsonl` under the configured error-log
directory, using the shared AAWM error-log environment resolver.

This path is best-effort local intake only. It does not replace
`.analysis/todo.md` as the durable queue, and it does not claim durable
`session_history` persistence for terminal-agent rows. Alias routing may still
emit separate audit/route events; those are distinct from this JSONL intake.

### Enablement and size control

- Opt-in flag: `LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED`
  - explicit truthy values: `1`, `true`, `yes`, `on`
  - explicit falsy values disable the writer even if the generic error sink is on
- Default when the flag is unset:
  - enabled if `LITELLM_AAWM_ERROR_LOG_DIR` is set, or
  - enabled if `LITELLM_AAWM_ERROR_LOG_ENABLED` is truthy
  - otherwise disabled
- Directory resolution reuses `_get_aawm_error_log_dir()` and falls back to the
  process working directory's `.analysis`
- Shared max-bytes control: `LITELLM_AAWM_ERROR_LOG_MAX_BYTES`
  - when unset or non-positive, appends continue without a file-size ceiling
  - when set to a positive integer, terminal-agent appends stop once
    the active `<environment>-error.jsonl` file reaches that size
- Ownership/mode repair after append reuses the same
  `LITELLM_AAWM_ERROR_LOG_FILE_UID` / `GID` / `MODE` helpers as the generic sink
- Append failures are swallowed; terminal intake must never fail the client
  request

### JSONL fields and semantics

Each terminal-agent row is one append-only JSON object. Top-level fields include:

- `schema_version` (currently `1`)
- `observed_at` (UTC ISO timestamp)
- `environment` (from the shared AAWM error-log environment resolver)
- `logger` = `litellm.proxy.agent_terminal`
- `level` = `ERROR`
- `message` = `Agent terminal error: <failure_kind>`
- `traceback`, `traceback_text`, `traceback_lines`, `raw_text` (structured
  terminal rows leave these empty/null; they are not traceback dumps)
- `fingerprint` for stable grouping of repeated failure classes
- `failure_kind` and `error_code`
- `status_code` (HTTP/upstream status when known)
- correlation identity: `agent_id`, `session_id`, `litellm_call_id`
- terminal semantics:
  - `terminal_outcome` — why the attempt ended (for example
    `request_rejected`, `agent_session_terminated`,
    `malformed_tool_call_rejected`)
  - `fallback_result` — what same-request fallback produced (for example
    `none`, `no_candidate_available`)
  - `redispatch_required` — whether the client/orchestrator should redispatch
    rather than treat the session as permanently killed by this row alone
  - `agent_session_killed` — whether this outcome terminated the agent session
- `context` — bounded nested metadata used for triage

Useful `context` identity and attempt fields (when available):

- route/provider: `source`, `container`, `endpoint` /
  `incoming_endpoint`, `upstream_url`, `outgoing_target`, `provider`, `model`,
  `model_alias` / `alias_model`, `alias_family`, `route_family`, `status_code` /
  `error_status_code`
- failure classification: `failure_kind`, `failure_class`, `error_code`,
  `event_type`, `candidate_status`, `failure_phase`, `attempted_provider_call`
- agent/session correlation: `repository`, `tenant_id`, `agent_name`,
  `agent_id`, `agent_role`, `agent_profile`, `thread_source`, `session_id`,
  `thread_id`, `trace_id`, `litellm_call_id`, `dispatch_id`,
  `redispatch_ordinal`
- attempt sequence: `attempt_count`, `attempts`, `candidate_count`,
  `candidates`, plus hidden-retry scalars such as
  `hidden_retry_final_outcome`, `hidden_retry_failure_classification`,
  `hidden_retry_count` when the failure exhausted hidden retries
- cooldown diagnostics when present: `cooldown_scope`, `cooldown_state_source`
- activity/status summaries: `terminal_activity_status`,
  `actual_prior_tool_activity_summary`
- bounded/redacted request-shape fields for Responses `422` killers:
  `aawm_passthrough_request_shape_summary`,
  `aawm_passthrough_request_shape_fingerprint`,
  `aawm_passthrough_request_shape_error_class`,
  `aawm_passthrough_request_shape_error_message_class`,
  `aawm_passthrough_request_shape_error_body_preview`,
  `aawm_passthrough_request_shape_error_fingerprint`

String context values are secret-redacted and truncated (about 2 KiB). Nested
attempt, candidate, request-shape, and prior-activity structures are rebuilt
from explicit field allowlists; unknown nested fields such as messages, bodies,
headers, tool arguments, and provider response detail are dropped rather than
serialized. Lists are item-bounded before append. The fingerprint is a SHA-256
over a stable lowercased join of failure class tokens, status code, provider,
model alias, route family, endpoint, and the request-shape error fingerprint
when present. Use `fingerprint` to group repeated terminal classes without
losing per-row agent/session/call identity fields.

Common emission paths:

- Responses request-shape `422` (`failure_kind=request_shape_deserialization_failed`)
  from the pass-through exception path, with
  `terminal_outcome=request_rejected`, `fallback_result=none`, and
  `agent_session_killed=true` when an AAWM alias or agent/dispatch identity is
  present; unattributed direct `422` responses are not written to this sink
- AAWM alias no-candidate terminal outcomes
  (`failure_kind=agent_alias_no_candidate`) with
  `terminal_outcome=agent_session_terminated` and
  `fallback_result=no_candidate_available`
- Malformed tool-call detections continue to use
  `.analysis/malformed-error.jsonl` with the same terminal semantic fields
  (`terminal_outcome`, `fallback_result`, `redispatch_required`,
  `agent_session_killed`) for correlation. New rows expose the stable grouping
  hash as `fingerprint`; `failure_fingerprint` is retained as a compatibility
  alias.

### xAI `SAFETY_CHECK_TYPE_CYBER` handling for AAWM aliases

Grok/xAI upstream `403` bodies that include all of `permission-denied`,
`Content violates usage guidelines`, and `SAFETY_CHECK_TYPE_CYBER` are classified
as `safety_policy_denied`.

For managed AAWM aliases (`model_alias` starting with `aawm-`):

- intermediate candidate denials are request-local only
- the denied candidate is excluded for the remainder of the same request so
  alias routing can fall through to the next declared candidate
- LiteLLM does **not** set a durable Redis/model cooldown for this class
- the pass-through layer defers generic failure hooks and does **not** write an
  intermediate terminal-agent JSONL row while same-request fallback can still
  advance
- one terminal-agent JSONL row is written only if fallback exhausts every
  remaining candidate and the alias ends in a no-candidate / agent-session
  terminated outcome

This prevents safety denials from being misread as provider unavailability or
durable cooldown evidence while still creating investigation intake when the
denial actually kills the agent session.

### Direct / non-alias `403` behavior

Direct Grok/xAI routes and non-AAWM aliases keep normal terminal failure
handling for the same `SAFETY_CHECK_TYPE_CYBER` body:

- generic pass-through failure logging and hooks still run
- active error intake is not deferred to alias fallback logic
- operators should treat these as ordinary upstream permission/safety failures
  for that direct route rather than as AAWM multi-candidate fallback events

### Privacy boundaries

Terminal-agent JSONL intake intentionally omits:

- prompts and assistant text
- tool arguments and tool result bodies
- credentials, OAuth tokens, API keys, cookies, and authorization headers
- raw request or response bodies

Only bounded/redacted route metadata, identity fields, request-shape summaries,
and failure classification tokens are recorded. Full-payload capture remains a
separate, stronger operator opt-in and is not enabled by terminal-agent intake.

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

## ClickHouse runtime auth intake (aawm-clickhouse)

Active errors in `.analysis/aawm-clickhouse-error.jsonl` that show `UNKNOWN_USER`
for user `default`, `Authentication failed: password is incorrect, or there is no
user with such name` for `default`, or similar default-user auth failures mean a
caller is probing or connecting with ClickHouse user `default` while
`aawm-clickhouse` exposes only configured users (for example `clickhouse`).

Resolve by identifying the caller (manual probe, script, Langfuse web/worker, or
other tooling) and pointing it at explicit credentials instead of implicit
`default`:

- `CLICKHOUSE_URL`, `CLICKHOUSE_USER`, and `CLICKHOUSE_PASSWORD`, or
- `LANGFUSE_CLICKHOUSE_URL`, `LANGFUSE_CLICKHOUSE_USER`, and
  `LANGFUSE_CLICKHOUSE_PASSWORD` for Langfuse-owned paths.

Do not enable or add a `default` ClickHouse user as the fix. Align client env
with the server's configured users.

Repo scripts that query Langfuse ClickHouse
(`scripts/backfill_session_history.py`,
`scripts/backfill_rate_limit_observations.py`,
`scripts/backfill_session_history_runtime_identity.py`,
`scripts/score_agent_trace_quality.py`) now:

- resolve auth from CLI flags or the env vars above (with bounded local defaults
  only when no env is set);
- run a `SELECT 1` preflight before ClickHouse-backed source modes such as
  `langfuse_clickhouse`;
- emit redacted `clickhouse_auth` diagnostics (normalized URL, redacted raw URL
  userinfo, user, and `*_source` fields) without passwords.

`scripts/backfill_session_history_runtime_identity.py` additionally resolves
ClickHouse auth once in `main()` (threaded into preflight/fetch), keyset-pages
GENERATION observations (`--clickhouse-page-size`, optional
`--clickhouse-max-pages` / `--clickhouse-resume-after-id`), and batch-loads
PostgreSQL temp identity tables (`--insert-batch-size`). See
`docs/aawm-session-history.md` § Runtime / Client Identity Backfill.

Archive active `aawm-clickhouse-error.jsonl` intake only after verification shows
no new default-user auth error signatures for the resolved caller class.


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
