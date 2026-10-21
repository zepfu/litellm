# OpenAI Passthrough

Pass-through endpoints for direct OpenAI API access

## Overview

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ❌ | Not supported |
| Logging | ✅ | Works across all integrations |
| Streaming | ✅ | Fully supported |

## Available Endpoints

### `/openai_passthrough` - Recommended
Dedicated passthrough endpoint that guarantees direct routing to OpenAI without conflicts.

**Use this for:**
- OpenAI Responses API (`/v1/responses`)
- Any endpoint where you need guaranteed passthrough
- When `/openai` routes are conflicting with LiteLLM's native implementations

### `/openai` - Legacy
Standard passthrough endpoint that may conflict with LiteLLM's native implementations.

**Note:** Some endpoints like `/openai/v1/responses` will be routed to LiteLLM's native implementation instead of OpenAI.

## Responses streaming lifecycle

Native OpenAI Responses streaming requires an authoritative terminal event:
`response.completed`, `response.failed`, or `response.incomplete`. LiteLLM
forwards the first valid terminal, emits exactly one `data: [DONE]`, and
discards incomplete trailing frames. A stream that ends without a valid
terminal is completed on the wire as `response.incomplete`; it is not treated
as a successful response.

For managed alias routing, the session-owner reservation and legacy session
affinity remain request-scoped while the stream is active. A completed terminal
can promote the owner and commit affinity. Failed, incomplete, cancelled, or
disconnected streams release the reservation and do not write new affinity.

Native OpenAI and alias-owned xAI Responses cleanup explicitly retires the
original chunk processor after response wrappers close, keeping its
finalization out of garbage collection.

For Codex OAuth, a text-only request with `tools: []` and retained
`web_search_call` history declares `web_search` with `tool_choice: "none"`.
The provider needs that declaration to read search history. Existing history
is preserved and new searches remain disabled. Nonempty tool lists, forced
tool choices and other provider routes retain their existing behavior.

### Responses stream limits

Native Responses streams share one attempt-scoped event pipeline across
precommit inspection, delivery, logging, usage and response policies. Configure
its limits through process environment variables. Byte limits use bytes;
the defaults below use binary MiB (`1 MiB = 1,048,576 bytes`).

| Environment variable | Default | Bounds |
| --- | --- | --- |
| `AAWM_RESPONSES_MAX_EVENT_BYTES` | 2 MiB | One SSE event, including an unfinished event |
| `AAWM_RESPONSES_MAX_STREAM_BYTES` | 128 MiB | Total admitted stream bytes |
| `AAWM_RESPONSES_MAX_STREAM_EVENTS` | 100,000 | Admitted events |
| `AAWM_RESPONSES_MAX_HELD_EVENTS` | 256 | Retained parsed events |
| `AAWM_RESPONSES_MAX_HELD_BYTES` | 8 MiB | Wire bytes plus JSON bytes of retained payloads |
| `AAWM_RESPONSES_MAX_OUTPUT_BYTES` | 8 MiB | Reconstructed output retained for summaries |
| `AAWM_RESPONSES_MAX_OUTPUT_ITEMS` | 128 | Reconstructed output items |
| `AAWM_RESPONSES_MAX_RAW_EVENTS` | 64 | Diagnostic raw-event history |
| `AAWM_RESPONSES_MAX_RAW_BYTES` | 2 MiB | Diagnostic raw-event history bytes |

Values must be integers. Held-event and raw-history limits accept zero; other
limits must be positive. Missing, invalid or below-minimum values use the
defaults. Raw-event retention is off unless
`AAWM_RESPONSES_EVENT_RAW_DIAGNOSTICS` is `1`, `true`, `yes` or `on`
(case-insensitive); either raw-history limit set to zero disables it.

Event, stream, retained-event and output-summary overflow rejects admission of
required state and follows the existing stream-failure handling. Already
admitted complete events retain their order; subsequent provider output is not
forwarded after the bound failure. Diagnostic raw-event history instead evicts
older entries within its limits without terminating delivery. These retention
budgets bound specific stream state, rather than the Python process's total
memory.

The accounting summary retains token counters and token-detail fields from
provider usage. Per-input `usage.attribution` remains on the forwarded provider
event and retained terminal response; it is excluded from the accounting
summary so request-history attribution cannot exhaust its 16 KiB usage budget.

Audit-only response policies process events incrementally. Watermark
`buffer_response` deliberately waits for the response and rejects buffering
above `general_settings.openai_passthrough_text_watermark.limits.max_sse_stream_bytes`
(128 MiB by default). Its text budget is separately controlled by
`limits.max_text_bytes_per_direction` (1 MiB by default).

### Capacity recovery before output

Native Responses routes keep explicitly allowlisted lifecycle events, empty
message and client-tool headers, and validated reasoning events private to the
current provider attempt. Reasoning content may be nonempty; client-tool
arguments or custom input must be exactly empty to qualify as a header.
Unknown events or payload extensions end replay eligibility.

This private-prefix behavior requires validated effective request capabilities
that exclude independent provider execution. When provider-executed actions
are possible, replay eligibility ends at dispatch. The absence of a tool event
or visible text does not prove that no provider-side action occurred.

A recognized overload while replay remains safe discards the failed attempt's
events, identities and reasoning, then retries through the existing capacity
coordinator. The surviving prefix is released in order at the first useful
assistant output or client-tool payload, without another upstream read or a
wait for completion. Text or tool payloads in a failure snapshot also prohibit
replay. Events after the attempt's first terminal event cannot change its
disposition, whether received together or in separate reads.

The shared stream owner enforces hard prefix and event bounds. Prefix
exhaustion ends replay eligibility and releases accepted events. Event
exhaustion preserves accepted complete events before terminating under the
normal malformed-stream policy. Neither condition permits overload recovery.
Prefix storage is released before consuming the live remainder.

The request-wide capacity deadline and progressive backoff remain in effect
across attempts. Cancellation ends buffering or backoff and prevents later
sends. Retry eligibility does not authorize an account change: direct and
alias routes retain their existing credential, endpoint and continuation
ownership rules. A valid `response.completed` releases reasoning-only or
empty-output responses. Output present only in a completion event is preserved.
`[DONE]` ends replay eligibility; the terminal owner determines validity, and
the marker alone does not establish success.

Diagnostics identify the event and reason that ended replay eligibility,
the retry disposition and actual send counts. Elapsed waiting is reported
separately from a timeout or exhausted retry budget; conservative commitment
does not mean the retry budget was consumed. Diagnostic event and reason labels
come from closed sets; unknown event names use a fixed label. Diagnostics
exclude raw reasoning, tool names, arguments, user payloads and credentials.

### Delivery and attribution

Route rollups attribute a managed request to the explicitly dispatched agent
name from structured `agent_name` metadata, including the canonical Codex name
inside the JSON string at `client_metadata["x-codex-turn-metadata"]`, or the
`x-aawm-agent-name` header. Structured names take precedence over the header
fallback. LiteLLM rejects unsafe or oversized labels and preserves the existing
header when no explicit name is supplied; a role declaration alone is not
treated as a dispatched agent name. All route-rollup producers apply this
attribution when present; absent-name behavior is unchanged.

Alpha can emit a bounded provenance companion for a matching emitted rollup
group. Observation is off unless
`general_settings.aawm_route_rollup_provenance_observation` is a mapping with
at least one exact `litellm_call_id`, `trace_id`, or `session_id`
selector. All configured selectors must match one cached request origin; any
invalid selector disables the mapping. A boolean setting or an empty mapping
does not enable observation. Select a known call ID, for example:

```yaml
general_settings:
  aawm_route_rollup_provenance_observation:
    litellm_call_id: "known-call-id"
```

`trace_id` matches an existing trace carrier. `session_id` is the cached logging
correlation field, with its existing client-field, header, and thread fallback
semantics; it does not create a session-owner identity. A missing configured
identity does not match.

Companions describe actual nonempty group emissions, including timer and manual
flushes. They record the cached winning name carrier, hashed request identity,
and rendered group attributes, including aggregated turns and each rendered
subline's model, target, and effort. Multiple or unknown request origins prevent
a claim of exclusive request attribution. Code fingerprints identify the listed
loaded functions and their embedded code constants; they do not attest
closures, globals, imported callees, or the complete deployment. Observation
does not change visible rollups, grouping, provider identity, or name precedence.

The final delivered disposition is also published for logging, rollup, and
session-transfer consumers. Consumers must use this disposition rather than
the upstream HTTP status: `completed` is the only successful outcome;
`failed`, `incomplete`, `cancelled`, and `disconnected` are non-success
outcomes. The disposition is selected once, after the terminal event and
`[DONE]` have been delivered for streams or after the response body has been
sent for non-streaming Responses.

Exit telemetry retains logical provider-call counts, transport connection
failure counts, and hidden retry counts alongside the wire-enriched send-ledger
snapshot. The OpenAI provider message `response protection is unavailable`,
with type `internal_error` and code `error`, null, or absent, follows the
ordinary retry budget and cannot receive additional capacity-retry
authorization.

The internal callback contract is
`on_disposition(disposition, trace)`. Consumers should read a copy from
`trace.snapshot()` containing the wire state, response/body commitment,
terminal and `[DONE]` markers, final disposition, and bounded duplicate,
partial-frame, and close-error fields. Detailed malformed-frame metadata
remains on `trace.metadata`; a malformed frame is converted to synthetic
`response.failed` before any later success can be selected. Consumers must
not promote ownership, write legacy affinity, or infer success independently
from the provider response.

## When to use this?

- For 90% of your use cases, you should use the [native LiteLLM OpenAI Integration](https://docs.litellm.ai/docs/providers/openai) (`/chat/completions`, `/embeddings`, `/completions`, `/images`, `/batches`, etc.)
- Use `/openai_passthrough` to call less popular or newer OpenAI endpoints that LiteLLM doesn't fully support yet, such as `/assistants`, `/threads`, `/vector_stores`, `/responses`

Simply replace `https://api.openai.com` with `LITELLM_PROXY_BASE_URL/openai_passthrough`

## Usage Examples

Requirements:
Set `OPENAI_API_KEY` in your environment variables.

> **Maintainer note:** Codex-native authenticated `GET /models` and `GET /v1/models` requests preserve inbound OpenAI auth and use `CHATGPT_API_BASE`. Generic or non-Codex requests retain the server-side `OPENAI_API_KEY` behavior.

### Assistants API

#### Create OpenAI Client

Make sure you do the following:
- Point `base_url` to your `LITELLM_PROXY_BASE_URL/openai`
- Use your `LITELLM_API_KEY` as the `api_key`

```python
import openai

client = openai.OpenAI(
    base_url="http://0.0.0.0:4000/openai_passthrough",  # <your-proxy-url>/openai_passthrough
    api_key="sk-anything"  # <your-proxy-api-key>
)
```

#### Create an Assistant

```python
# Create an assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a math tutor. Help solve equations.",
    model="gpt-4o",
)
```

#### Create a Thread
```python
# Create a thread
thread = client.beta.threads.create()
```

#### Add a Message to the Thread
```python
# Add a message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve 3x + 11 = 14",
)
```

#### Run the Assistant
```python
# Create a run to get the assistant's response
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Check run status
run_status = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
)
```

#### Retrieve Messages
```python
# List messages after the run completes
messages = client.beta.threads.messages.list(
    thread_id=thread.id
)
```

#### Delete the Assistant

```python
# Delete the assistant when done
client.beta.assistants.delete(assistant.id)
```

## Text watermark policy (optional)

`/openai_passthrough` can inspect visible Responses and Chat Completions text
for deterministic Unicode carriers (zero-width/format controls, noncharacters,
and configured exotic spaces). This is not a vendor-watermark detector and
does not rewrite prose with a second model.

Shipped default is off:

```yaml
general_settings:
  openai_passthrough_text_watermark:
    mode: off
    unicode:
      enabled: true
      policy: conservative
    removal:
      enabled: false
      stream_policy: audit_only
    statistical_detectors: []
```

Leave this disabled unless you explicitly opt in. `sanitize` / `enforce`
require `removal.enabled: true`. `enforce` plus streamed output also requires
`stream_policy: buffer_response`. Statistical detectors stay an empty disabled
registry; they report `unsupported` / `inconclusive` and never load
torch/transformers.
