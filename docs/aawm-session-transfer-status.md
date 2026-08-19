# AAWM Session Transfer Status

`GET /internal/aawm/session-transfer-status` is the authenticated, content-free
live-transfer contract for D1-617. Agent tooling can query whether a provider
response is still moving through LiteLLM without receiving prompts, deltas,
reasoning, tool arguments/results, credentials, raw payloads, raw headers, or
Redis keys.

This endpoint reports **transport and context evidence only**. LiteLLM does not
recommend cancellation or redispatch. A completed LiteLLM transfer does **not**
prove that the local agent finished parsing, persistence, or tool execution.

## Authentication

The route is private and always runs `user_api_key_auth`.

Authorized callers:

- proxy admin / proxy admin viewer
- a dedicated transcript service key whose `permissions` include
  `aawm_session_transfer_status`
- a virtual key whose `allowed_routes` includes
  `/internal/aawm/session-transfer-status`

All other authenticated callers receive HTTP 403. Unauthenticated callers fail
closed through the existing auth dependency.

The route is listed in `LiteLLMRoutes.self_managed_routes`. The handler
enforces the permission check above.

## Query parameters

At least one identity filter is required:

| Parameter | Meaning |
| --- | --- |
| `session_id` | Canonical or extracted session identity |
| `codex_session_id` | Exact Codex session ID accepted or extracted by LiteLLM |
| `canonical_session_id` | Server-side canonical session identity |
| `agent_id` | Agent / subagent identity |
| `litellm_call_id` | LiteLLM call identity |
| `active_only` | When true, omit stale and terminal records |
| `limit` | Result bound, default 20, maximum 50 |

Redis keys hash these identities. The response still returns the exact
authorized query identity fields. Consumers never reverse a hash.

## Response schema (`session-transfer-status.v1`)

```json
{
  "schema_version": "session-transfer-status.v1",
  "registry": {
    "state": "ok",
    "mode": "redis",
    "state_source": "durable_cache",
    "reachable": true,
    "error_class": null
  },
  "result_count": 1,
  "truncated": false,
  "transfers": [
    {
      "schema_version": "session-transfer-status.v1",
      "litellm_call_id": "...",
      "trace_id": "...",
      "canonical_session_id": "...",
      "session_id": "...",
      "codex_session_id": "...",
      "agent_id": "...",
      "agent_name": "...",
      "parent_agent_id": "...",
      "parent_session_id": "...",
      "provider": "openai",
      "model": "gpt-5.4",
      "route": "https://chatgpt.com/backend-api/codex/responses",
      "stream_path": "pass_through",
      "source_instance": "litellm-dev-1",
      "phase": "response_streaming",
      "active": true,
      "stale": false,
      "redis_degraded": false,
      "freshness": "live",
      "received_at": "2026-08-18T12:00:00Z",
      "preparing_at": "2026-08-18T12:00:00Z",
      "awaiting_upstream_at": "2026-08-18T12:00:01Z",
      "first_upstream_chunk_at": "2026-08-18T12:00:02Z",
      "first_downstream_chunk_at": "2026-08-18T12:00:02Z",
      "last_heartbeat_at": "2026-08-18T12:00:03Z",
      "finalized_at": null,
      "upstream_chunk_count": 4,
      "upstream_byte_count": 512,
      "downstream_chunk_count": 3,
      "downstream_byte_count": 480,
      "context": {
        "context_window": 128000,
        "estimated_input_tokens": 1200,
        "estimated_output_tokens": 40,
        "provider_input_tokens": 1180,
        "provider_output_tokens": 32,
        "remaining_tokens": 126820,
        "request_count": 3,
        "cumulative_input_tokens": 3600,
        "repeated_prefix_tokens": 800,
        "prompt_category_tokens": {
          "system": 200,
          "tool_advertisement": 300,
          "conversation": 500,
          "other": 100,
          "residual": 20,
          "system_behavior": 80,
          "system_safety": 40,
          "system_instructional": 60,
          "system_unclassified": 20
        }
      },
      "terminal_state": null,
      "disconnect_reason": null,
      "timeout_kind": null,
      "error_code": null,
      "error_class": null
    }
  ]
}
```

### Phases

Truthful phases only:

- `request_received`
- `request_preparing`
- `awaiting_upstream`
- `response_streaming`
- `finalizing`
- terminal: `completed`, `failed`, `cancelled`, `disconnected`, `timed_out`

Request bodies are buffered before provider dispatch. The endpoint never claims
progressive client-upload progress.

Upstream counters are provider-receipt counts. Downstream counters are client
emission counts. They can differ when the handler peeks, holds a `[DONE]`
suffix, or synthesizes a terminal event.

### Freshness and Redis degradation

- Heartbeats are throttled to at most one write every 750ms.
- Active records expire after 180 seconds.
- An active record with no heartbeat for 30 seconds becomes `stale`. Stale is
  not a terminal claim.
- Terminal summaries are retained for 45 seconds.
- Indexes are bounded to 64 call IDs per identity and 50 query results.
- A worker crash or Redis outage produces `unavailable` / `stale` /
  `redis_degraded=true`. It never invents `completed`.

The registry reuses the managed AAWM alias-routing Redis connection
(`AAWM_ALIAS_ROUTING_REDIS_*`) but stores records under
`aawm:session-transfer:<namespace>:...`. Cooldown and affinity keys are never
shared. Optional override: `AAWM_SESSION_TRANSFER_STATE_NAMESPACE`.

When Redis is not attached, the process falls back to a memory store and
reports `registry.state=unavailable` or `degraded`. Process-local state is
incomplete in multi-worker deployments.

## Stream-path coverage

First landing instruments:

- central pass-through request phases in
  `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- the shared stream generator in
  `litellm/proxy/pass_through_endpoints/streaming_handler.py`

Remaining adapter `StreamingResponse` paths should call
`publish_adapter_transfer_event()` from
`litellm.proxy.aawm_session_transfer.hooks` instead of creating a second
registry. Adapter coverage is incomplete in this landing and is documented as
such; the endpoint and registry are shared.

## Sanitization

Returned strings are printable, length-bounded, and reject credential-shaped
prefixes. Routes keep scheme/host/path only. Errors expose a short class name
and a closed `error_code` set (`timeout`, `disconnect`, `cancelled`,
`upstream_error`, `internal`). Redis keys, raw exceptions, headers, and request
or response bodies are never returned.

Context measurements are token counts and window sizes only. Prompt category
fields are estimated counts, not prompt text.

## Deployment

No extra container, SQL, or live Redis mutation is required to land the source.
Production multi-worker freshness requires the existing AAWM alias-routing Redis
sidecar. Create a dedicated transcript key with
`permissions=["aawm_session_transfer_status"]` only under separate operator
authorization.
