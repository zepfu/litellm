# AAWM Session History Metadata

This fork stores AAWM-specific routing and observability details in
`aawm_tristore.public.session_history.metadata`. These keys are intended for
maintainer diagnostics and downstream reporting surfaces. They should not be
treated as public LiteLLM API guarantees.

## xAI Responses Sanitization

LiteLLM sanitizes xAI Responses request bodies for Codex/OpenAI passthrough and
native Grok Composer passthrough before forwarding to xAI. The sanitizer removes
unsupported top-level Responses fields and unsupported tool fields while
preserving bounded metadata that explains the adaptation.

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
