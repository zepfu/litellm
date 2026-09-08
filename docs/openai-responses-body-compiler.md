# OpenAI Responses Body Compiler

OpenAI Responses egress uses
`litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/openai_responses_body.py`
as the canonical request-body boundary.

## Lifecycle

The compiler runs after pass-through schema normalization, guardrail metadata
attachment, and `pre_call_hook`. Those transforms operate on the mutable
observability body. The compiler then produces the exact provider payload and
binds that payload to the request. HTTPX serializes only `wire_body.body`.

The provider-bound payload is a recursively immutable, JSON-serializable
snapshot. Logging, encrypted-reasoning disposition, function-name restoration,
watermark audit, and the hook-visible body are separate diagnostics state.
Diagnostics must not be copied back into the provider payload.

## Transform Order

The compiler applies legacy function-history normalization, resolved-model
request shaping, function-name sanitization, request watermark egress, and
encrypted-reasoning preparation. Encrypted-reasoning provenance is validated
before server-owned sidecars are removed.

One scoped sanitation traversal then removes known top-level LiteLLM/AAWM
context and route/provenance sidecars from top-level `input` and `output`
items. It does not recurse into prompts, tool schemas, function arguments, or
other user data. OpenAI `metadata` remains a supported client field; the
server-owned `litellm_metadata` namespace does not.

Direct non-OpenAI egress retains route-identity cleanup at the request envelope
and top-level `input`/`output` items without traversing nested user/tool data.
