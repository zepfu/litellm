# v1/messages → /responses Parameter Mapping

When you send a request to `/v1/messages` targeting an OpenAI or Azure model, LiteLLM internally routes it through the OpenAI Responses API. This page documents exactly how every parameter gets translated in both directions.

The transformation lives in `litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py`.

Streaming Anthropic SSE reconstruction (`event: content_block_*`, `message_start` / `message_delta` / `message_stop`) is **not** part of this module. That path is owned by `responses_adapters/streaming_iterator.py`, which reuses shared emitter helpers from `adapters/streaming_iterator.py`. This mapping page only covers request/response **body** conversion.

### Prompt cache routing (`cache_control` → `prompt_cache_key`)

Anthropic `cache_control` markers (including Claude Code's moving per-turn breakpoints on recent messages) are translated into OpenAI Responses `prompt_cache_key` **only from the stable surface**: `system` and `tools`. Message-level `cache_control` is recognized for cache-intent metadata but does **not** change the key across turns. If only volatile message-level markers are present, LiteLLM omits `prompt_cache_key` rather than emitting a per-turn key that would defeat OpenAI cache affinity. An explicit request `prompt_cache_key` (when provided) is bounded to ≤ 64 characters by hashing when needed; it is not re-derived via the system/tools helper.

Shared derivation lives in `adapters/observability.derive_prompt_cache_key`; this Responses adapter resolves and bounds the outbound key in `_resolve_responses_prompt_cache_key` / `_bound_prompt_cache_key`.



## Request: Anthropic → Responses API

### Top-level parameters

| Anthropic (`/v1/messages`) | Responses API | Notes |
|---|---|---|
| `model` | `model` | Passed through as-is |
| `messages` | `input` | Structurally transformed — see the messages section below |
| `system` (string) | `instructions` | Passed as a plain string |
| `system` (list of content blocks) | `instructions` | Text blocks are joined with `\n`; non-text blocks are ignored |
| `max_tokens` | `max_output_tokens` | Renamed |
| `temperature` | `temperature` | Passed through as-is |
| `top_p` | `top_p` | Passed through as-is |
| `tools` | `tools` | Format-translated — see the tools section below |
| `tool_choice` | `tool_choice` | Type-remapped — see the tool_choice section below |
| `thinking` | `reasoning` | Budget tokens mapped to effort level — see the thinking section below |
| `output_format` or `output_config.format` | `text` | Wrapped as `{"format": {"type": "json_schema", "name": "structured_output", "schema": ..., "strict": true}}` |
| `context_management` | `context_management` | Converted from Anthropic dict to OpenAI array format — see the context_management section below |
| `metadata.user_id` | `user` | Extracted from the metadata object and truncated to 64 characters |
| `cache_control` (on `system` / `tools` / content blocks) | `prompt_cache_key` (+ litellm metadata) | Stable key from system/tools only; message-only markers omit the key — see prompt cache section above |
| `prompt_cache_key` (explicit) | `prompt_cache_key` | Passed through when ≤ 64 chars; longer values are hashed to an OpenAI-safe key |
| `stop_sequences` | ❌ Not mapped | Dropped silently |
| `top_k` | ❌ Not mapped | Dropped silently |
| `speed` | ❌ Not mapped | Only used to set Anthropic beta headers on the native path |


### How messages get converted

Each Anthropic message is expanded into one or more Responses API input items. The key difference is that `tool_result` and `tool_use` blocks become **top-level items** in the input array rather than being nested inside a message.

| Anthropic message | Responses API input item |
|---|---|
| `user` role, string content | `{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}` |
| `user` role, `{"type": "text"}` block | `{"type": "input_text", "text": "..."}` inside a user message |
| `user` role, `{"type": "image", "source": {"type": "base64"}}` | `{"type": "input_image", "image_url": "data:<media_type>;base64,<data>"}` inside a user message |
| `user` role, `{"type": "image", "source": {"type": "url"}}` | `{"type": "input_image", "image_url": "<url>"}` inside a user message |
| `user` role, `{"type": "tool_result"}` block | Top-level `{"type": "function_call_output", "call_id": "...", "output": "..."}` — pulled out of the message entirely |
| `assistant` role, string content | `{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}` |
| `assistant` role, `{"type": "text"}` block | `{"type": "output_text", "text": "..."}` inside an assistant message |
| `assistant` role, `{"type": "tool_use"}` block | Top-level `{"type": "function_call", "call_id": "<id>", "name": "...", "arguments": "<JSON string>"}` — pulled out of the message entirely |
| `assistant` role, `{"type": "thinking"}` block | `{"type": "output_text", "text": "<thinking text>"}` inside an assistant message |


### tools

| Anthropic tool | Responses API tool |
|---|---|
| Any tool where `type` starts with `"web_search"` or `name == "web_search"` | `{"type": "web_search_preview"}` |
| All other tools | `{"type": "function", "name": "...", "description": "...", "parameters": <input_schema>}` |


### tool_choice

| Anthropic `tool_choice.type` | Responses API `tool_choice` |
|---|---|
| `"auto"` | `{"type": "auto"}` |
| `"any"` | `{"type": "required"}` |
| `"tool"` | `{"type": "function", "name": "<tool name>"}` |


### thinking → reasoning

The `budget_tokens` value is mapped to a string effort level. `summary` is always set to `"detailed"`.

| `thinking.budget_tokens` | `reasoning.effort` |
|---|---|
| >= 10000 | `"high"` |
| >= 5000 | `"medium"` |
| >= 2000 | `"low"` |
| < 2000 | `"minimal"` |

If `thinking.type` is anything other than `"enabled"`, the `reasoning` field is not sent at all.


### context_management

Anthropic uses a nested dict with an `edits` array. OpenAI uses a flat array of compaction objects.

```
Anthropic input:
{
  "edits": [
    {
      "type": "compact_20260112",
      "trigger": {"type": "input_tokens", "value": 150000}
    }
  ]
}

Responses API output:
[
  {"type": "compaction", "compact_threshold": 150000}
]
```


## Response: Responses API → Anthropic

When the Responses API reply comes back, LiteLLM converts it into an Anthropic `AnthropicMessagesResponse`.

| Responses API field | Anthropic response field | Notes |
|---|---|---|
| `response.id` | `id` | |
| `response.model` | `model` | Falls back to `"unknown-model"` if missing |
| `ResponseReasoningItem` — `summary[*].text` | `content` block `{"type": "thinking", "thinking": "..."}` | Each non-empty summary text becomes a thinking block |
| `ResponseOutputMessage` — `content[*]` where `type == "output_text"` | `content` block `{"type": "text", "text": "..."}` | |
| `ResponseFunctionToolCall` — `{call_id, name, arguments}` | `content` block `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}` | `arguments` is JSON-parsed back into a dict |
| Any `function_call` present in output | `stop_reason: "tool_use"` | |
| `response.status == "incomplete"` | `stop_reason: "max_tokens"` | Takes precedence over the default |
| Everything else | `stop_reason: "end_turn"` | Default |
| `response.usage.input_tokens` | `usage.input_tokens` | |
| `response.usage.output_tokens` | `usage.output_tokens` | |
| *(hardcoded)* | `type: "message"` | Always set |
| *(hardcoded)* | `role: "assistant"` | Always set |
| *(hardcoded)* | `stop_sequence: null` | Always null on this path |

## Streaming: Responses events → Anthropic SSE

Streaming reconstruction lives in
`litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py`
(`AnthropicResponsesStreamWrapper`).

Anthropic event envelopes and byte framing are **not** duplicated here. The
Responses wrapper imports the shared emitter surface from
`litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py`
(the same helpers used by the Chat Completions adapter tree):

- `emit_message_start` / `emit_message_delta` / `emit_message_stop`
- `emit_content_block_start` / `emit_content_block_delta` / `emit_content_block_stop`
- `encode_anthropic_sse_chunk` (used by `async_anthropic_sse_wrapper()`)

| Responses API event | Anthropic SSE |
|---|---|
| `response.created` | `message_start` |
| `response.output_item.added` (`message`, `function_call`, `reasoning`, `mcp_call`) | `content_block_start` |
| `response.output_text.delta` | `content_block_delta` with `text_delta` |
| `response.reasoning_summary_text.delta` | `content_block_delta` with `thinking_delta` |
| `response.function_call_arguments.delta` / `.done` | `content_block_delta` with `input_json_delta` (empty argument deltas are not framed) |
| `response.output_item.done` | `content_block_stop` (and a late `input_json_delta` when arguments never streamed) |
| `response.completed` | `message_delta` + `message_stop` |

Only the **Responses → normalized delta** mapping is tree-local. Client-visible
Anthropic SSE shapes and framing stay on the shared emitter so Chat vs Responses
lanes cannot diverge.

### `prompt_cache_key` stability

When this adapter sets OpenAI `prompt_cache_key` for Anthropic-shaped traffic,
the key is produced by `adapters/observability.derive_prompt_cache_key()`. The
helper hashes stable `system` + `tools` roots only. Moving / per-turn
`cache_control` breakpoints on conversation turns are excluded so multi-turn
sessions reuse the same key instead of missing the server-side prompt cache on
every turn.
