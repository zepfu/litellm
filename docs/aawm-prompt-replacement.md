# TUI prompt replacement

Prompt replacement follows the incoming API format independently of the
destination provider. A client using a supported format inherits replacement
support without a separate TUI implementation.

Eligible prompt fields are Responses `instructions` and system/developer input
messages, Chat Completions system/developer message content, Anthropic Messages
`system`, and Gemini `systemInstruction` text. Existing Claude control-plane
replacement surfaces retain their section matching and version compatibility.
Message roles and unrelated user, assistant and tool content are preserved.

Rules match source text before routing and format conversion. A global rule
does not require a recognized TUI or `cc_version`. Optional source-TUI, version
and prompt-field filters narrow applicability; they do not select a destination
provider. Eligible fields whose text does not match remain unchanged.

Shared JSON manifests define rules using `id`, `before` or `before_pattern`,
and `after`. Optional `source_tuis`, `cc_version_min`, and `prompt_fields`
restrict a rule. Source identity uses client metadata or headers, including
User-Agent product/version; unknown products remain usable as explicit filters.
Prompt fields are `instructions`, `system`, or `developer`; Gemini's
`systemInstruction` uses the `system` field filter. Without filters, a rule
applies globally. The built-in
memory rules live in
`context-replacement/shared/codex-memory-edit-instructions.json`.
Set `AAWM_PROMPT_REPLACEMENT_MANIFEST_DIR` to use another directory of JSON
manifests. Default discovery checks the repository directory, then the
installed control-plane package assets. Reload the proxy after editing cached
rules. Claude's compatibility facade and the generic hooks share the same
manifest and text caches, including the compiled Claude report-file matcher.
Gemini passthrough envelopes rewrite the selected `custom_body`, preserving
unselected outer instructions.

The Codex memory rule changes the developer instructions to:

```text
You can update the memories only when explicitly asked by the user. This must always come from a direct request from the user.

Do not try to edit the memory files yourself unless explicitly directed by user, only add one update note in [...].
```

`[...]` is the original client-provided update-note destination. Replacement
preserves that path and the surrounding instructions. It does not edit memory
files or change client configuration.

Matched rule IDs, replacement counts and failures are recorded with request
metadata. Retries and continuations preserve the rewritten text without
repeatedly altering it.

When the existing payload-capture callback receives a provider-transformed
`complete_input_dict`, it also saves `provider_request.json` and provenance in
`meta.json`. This is the logical body retained by the provider call path in
logging callback arguments, rather than final HTTP serialization. Its presence
and stream flags depend on that path. The existing
`system`, `messages` and `tools` captures remain separate input evidence.
Provider-body evidence is omitted when the provider path supplies none.
It is also omitted when message logging is redacted.

Source checks establish engine behavior; live acceptance requires correlated
provider-bound request evidence. Reconstructed callback inputs, health checks
and shape-only captures do not establish the exact outgoing instruction text.
For alpha, use the persistent repository-root deployment and the existing
capture controls described by the runtime configuration. Alpha evidence does
not establish deployment to development or production.
