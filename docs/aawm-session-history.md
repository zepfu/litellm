# AAWM Session History Metadata

This fork stores AAWM-specific routing and observability details in
`aawm_tristore.public.session_history.metadata`. These keys are intended for
maintainer diagnostics and downstream reporting surfaces. They should not be
treated as public LiteLLM API guarantees.

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

## xAI Responses Sanitization

LiteLLM sanitizes xAI Responses request bodies for Codex/OpenAI passthrough and
native Grok Composer passthrough before forwarding to xAI. The sanitizer removes
unsupported top-level Responses fields and unsupported tool fields while
preserving bounded metadata that explains the adaptation. Native Grok and
managed xAI Responses requests also drop `tool_choice` when the outgoing payload
has no usable `tools` definitions, since xAI rejects `tool_choice` without
tools.

Rows affected by this path may include:

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
