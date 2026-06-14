# AAWM Session History Metadata

This fork stores AAWM-specific routing and observability details in
`aawm_tristore.public.session_history.metadata`. These keys are intended for
maintainer diagnostics and downstream reporting surfaces. They should not be
treated as public LiteLLM API guarantees.

## Inbound Alias Capture

`public.session_history` includes a nullable `inbound_model_alias` text column.
It stores the model request value exactly as LiteLLM received it before any alias
resolution.

- For AAWM alias requests, this is the inbound alias, for example
  `aawm-read`, `aawm-low`, or `aawm-code-anthropic`.
- For direct concrete requests, this is the concrete model string (and may equal
  `session_history.model`).

`session_history.model` remains the routed/selected concrete provider model and
must not be repurposed for inbound alias grouping.

Existing metadata fields `requested_model_alias` and `model_alias_label` remain
compatibility and trace metadata. Use `inbound_model_alias` as the canonical field
for reporting and grouping by requested alias.

Historical rows written before this field existed may be `NULL` unless they were
explicitly backfilled from prior metadata.

## Alias Routing Audit Metadata

AAWM auto-agent aliases attach bounded routing metadata to selected requests so
operators can confirm ordered failover without replaying raw transcripts.

Common keys include:

- `requested_model_alias` / `model_alias_label`: the inbound alias, for example
  `aawm-code` or `aawm-code-anthropic`.
- `aawm_alias_routing_audit_events`: ordered events for skipped, failed, and
  selected candidates.
- `codex_auto_agent_attempts` or `anthropic_auto_agent_attempts`: candidate
  attempts made by the selected alias handler.
- `codex_auto_agent_skipped_candidates` or
  `anthropic_auto_agent_skipped_candidates`: candidates skipped because of
  cooldown or stateful session-affinity cooldown.

For retryable provider errors, the handler records a
`candidate_retryable_failure` event, cools down that candidate, and selects the
next configured usable candidate. For tool-bearing or stateful
`aawm-code-anthropic` requests, every declared candidate route is treated as a
Claude Code tool-contract route: if the alias declares Antigravity, OpenAI, xAI,
native Anthropic, or another provider/model target, that target must preserve
tool calls, tool-use ids, tool arguments, tool-result replay, and ordered
failover metadata for engineering-agent traffic. A declared candidate must not
be skipped as policy-incompatible for the same request class it is meant to
serve. If a declared candidate mishandles tools, that is an adapter translation
defect to fix or evidence for removing/reclassifying the candidate; it is not a
selector-side compatibility decision.

## Grok Native OIDC Credentials

`xai/grok-composer-2.5-fast`, `xai/grok-build`, and
`xai/grok-build-0.1` use the Grok native OIDC credential path, not the managed
`oa_xai/*` OAuth credential file. In `litellm-dev`, `LITELLM_XAI_GROK_AUTH_FILE`
defaults to `/home/zepfu/.litellm/xai/grok-auth.json`, while the personal Grok
CLI credential at `/home/zepfu/.grok/auth.json` is mounted read-only and exposed
only as `LITELLM_XAI_GROK_SEED_AUTH_FILE`.

The Grok native refresh path updates the configured credential by writing a
temporary file beside it and then atomically replacing the original file. The
configured refresh target must therefore live on writable LiteLLM-owned storage.
Mounting the configured target read-only makes Composer and other Grok native
candidates fail as `candidate_unavailable` during token refresh, which breaks the
declared alias failover order. Keep `grok-auth.json` separate from the managed
`oa_xai/*` `oauth-auth.json` file so the two credential families remain
auditable.

When the read-only Grok CLI seed credential is newer than the LiteLLM-owned
managed credential, LiteLLM replaces the managed Grok credential from the seed
before selecting or refreshing an access token. This lets a fresh Grok/OIDC login
take effect without writing back into `/home/zepfu/.grok`.

## Access Log Display Semantics

AAWM passthrough route logs emit a compact route line to the LiteLLM proxy logger
after the incoming request has been locally prepared and the egress target is
known.
The display shape is:

```text
ROUTE: [agent@repository -] [model(alias) -] [ip:port -] METHOD incoming -> outgoing PROTOCOL
```

Agent, repository, model alias, and client address segments are omitted when
that metadata is unavailable. The selected model is printed as `model(alias)`
only when the inbound alias differs from the selected model.

These lines are for live container-log triage, not durable reporting. Durable
model and alias attribution still comes from `session_history.model` and
`session_history.inbound_model_alias`. Route-log identity fields are conservative
display tokens derived from normalized metadata or explicit headers such as
`x-aawm-agent-name` and `x-aawm-repository`; LiteLLM omits prompt-like,
sentence-like, or punctuation-heavy agent/repository values instead of printing
raw request text. LiteLLM intentionally does not inspect prompt text or raw tool
arguments for the route log.

Route logs must not include API keys, authorization headers, full request or
response bodies, prompt content, tool arguments, or arbitrary query strings.
Incoming endpoints preserve only known-safe routing query markers, and outgoing
targets are logged as host plus path.

For pass-through requests that emit this enriched `ROUTE:` line, LiteLLM
suppresses the matching native Uvicorn/Gunicorn access record for that request.
Unrelated routes should continue to use the normal server access log.

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

## Codex Tool-Description Patches

AAWM Codex and Claude Code adapter paths may patch advertised tool descriptions
before provider egress when the route needs guidance that survives provider
transforms which can drop top-level instructions or metadata. These patches are
request-shape guidance only: they do not prove that a model called a tool
correctly.

Rows and traces affected by this path include the request tag
`codex-tool-description-patch` plus one tag per applied patch, for example
`codex-tool-description-patch:spawn-agent-fanout-policy` or
`codex-tool-description-patch:core-tool-guidance-edit`. Metadata may include:

- `codex_tool_description_patch_count`: number of applied tool-description
  patch events.
- `codex_tool_description_patch_replacement_count`: number of text replacements
  made by replacement-style patches.
- `codex_tool_description_patch_ids`: stable patch identifiers applied to the
  request.
- `codex_tool_description_patch_events`: bounded per-tool patch records with
  tool name, path, and patch id.

Core-tool guidance patches currently target Claude Code/Codex tools such as
`Bash`, `Edit`, `Read`, and `Write`. The intent is to make declared fallback
models, including xAI/Grok routes, receive the same operational cautions about
structured edits, stale `old_string` retries, bounded reads, and reading before
overwriting existing files even when the provider adapter removes unsupported
top-level fields.

## xAI Responses Sanitization

LiteLLM sanitizes xAI Responses request bodies for Codex/OpenAI passthrough and
native Grok Composer passthrough before forwarding to xAI. The sanitizer removes
unsupported top-level Responses fields and unsupported tool fields while
preserving bounded metadata that explains the adaptation. Native Grok and
managed xAI Responses requests also drop `tool_choice` when the outgoing payload
has no usable `tools` definitions, since xAI rejects `tool_choice` without
tools.

For AAWM Codex aliases, hosted-tool support is evaluated again after the alias
has selected a concrete xAI/Grok candidate such as `grok-composer-2.5-fast` or
`oa_xai/grok-build`. This catches provider-invalid Codex tool variants, including
`custom`, that could not be classified while the inbound request model was still
the abstract alias `aawm-code`.

Rows affected by this path may include:

- `codex_unsupported_hosted_tool_removed_count`,
  `codex_unsupported_hosted_tool_types_removed`, and
  `codex_unsupported_hosted_tools_removed`: Codex hosted-tool definitions removed
  because the selected concrete model marks them unsupported. For xAI/Grok
  Responses routes this commonly includes `custom`, `image_generation`,
  `namespace`, or `tool_search`.
- `codex_unsupported_hosted_tool_choice_removed`: the removed `tool_choice` when
  it referenced a hosted tool removed by the selected model policy.
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
