# OpenAI Passthrough Text-Watermark Detection and Sanitation

Status: design proposal only
Date: 2026-08-20
Target: `/openai_passthrough`, primarily `/v1/responses` and
`/v1/chat/completions`

## Objective

Add an optional, local-only text-watermark policy to the OpenAI passthrough
path that can:

1. Inspect text received from the upstream LLM.
2. Inspect text the harness sends to the upstream LLM.
3. Record bounded detection and sanitation evidence in
   `session_history.metadata`.
4. Remove configured deterministic text carriers without calling another
   model.

This proposal does not claim that LiteLLM can generically detect or remove an
undocumented statistical vendor watermark. It distinguishes deterministic
Unicode carrier evidence from scheme-specific statistical detection and from
an unknown vendor watermark state.

## Recommendation

Implement a small, fork-owned module with a detector/remover registry, but ship
only the following enabled behavior initially:

- Protocol-aware extraction of user-visible text from OpenAI Responses and
  Chat Completions payloads.
- Conservative deterministic inspection for suspicious Unicode carriers.
- Optional conservative Unicode sanitation.
- Re-inspection after sanitation.
- Separate request and response audit objects in
  `session_history.metadata`.
- Detection-only streaming by default.
- Strict streamed-output removal only when the configured stream policy
  buffers text before emitting it.

Do not include model-based paraphrasing, self-information rewriting, or a
second-model verification pass. Do not label a statistical vendor watermark
as removed unless an authoritative detector for that exact scheme reports a
negative result.

## Research Review

The repositories were reviewed at these commits on 2026-08-20:

| Repository | Reviewed commit | Relevant approach | Fit for LiteLLM |
| --- | --- | --- | --- |
| [`guillaumemeyer/watermarks-remover`](https://github.com/guillaumemeyer/watermarks-remover/tree/d5563d2e129166cae737cc6ac604e75f5631c368) | `d5563d2e129166cae737cc6ac604e75f5631c368` | Context-aware Unicode inspection/sanitation, a common detector interface, pre/post detection, and explicit same-key limitations for research detectors | Best source of concepts for the deterministic scanner and audit contract. MIT licensed. Its broad file/media service and model rewrite machinery are out of scope. |
| [`mikiane/claude-watermark-cleaner`](https://github.com/mikiane/claude-watermark-cleaner/tree/63171ace22f27eff9985ae4ceaaf6029827fd189) | `63171ace22f27eff9985ae4ceaaf6029827fd189` | Unicode cleanup followed by Ollama or Codex paraphrasing, with protected literals and number preservation checks | Protected-literal ideas are useful, but its statistical-removal path requires a second model and violates this proposal's constraint. No license file was present in the reviewed snapshot, so do not copy source. |
| [`aloshdenny/claude-awm`](https://github.com/aloshdenny/claude-awm/tree/bd442b409a4efd55035c88a661b58e31c007876d) | `bd442b409a4efd55035c88a661b58e31c007876d` | SynthID research generation/detection, normalization experiments, and variation-selector/tokenizer-desynchronization analysis | Useful for threat modeling. Its detector is valid only with the matching research key, tokenizer, and scheme. It is too heavy for the request path and is not a production vendor detector. No license file was present in the reviewed snapshot, so do not copy source. |
| [`cuibitlabs/claude-watermark-remover`](https://github.com/cuibitlabs/claude-watermark-remover/tree/4bc80933096627c0d913b71cbb8d86ed23bc3f1d) | `4bc80933096627c0d913b71cbb8d86ed23bc3f1d` | Conservative/aggressive Unicode sanitation, protected-span auditing, bounded reports, and an explicit `unverified` provenance verdict | Good source of audit semantics and conservative defaults. MIT licensed. Its primary statistical-removal workflow uses another model and is out of scope. |
| [`Allencheng97/Self-information-Rewrite-Attack`](https://github.com/Allencheng97/Self-information-Rewrite-Attack/tree/eeae0b50bc64bed3e9730ef43d48da5a182983a0) | `eeae0b50bc64bed3e9730ef43d48da5a182983a0` | Paraphrase, score token self-information, blank selected tokens, and regenerate to attack multiple research watermark schemes | Useful evidence that statistical removal is a rewriting problem. It requires one or more local models and is therefore out of scope. The repository declares MIT in its README. |

### Conclusions from the review

1. Invisible or format Unicode can be inspected and changed
   deterministically.
2. A suspicious Unicode character is not, by itself, proof of a vendor
   watermark. Some joiners, bidi controls, variation selectors, and script
   fillers have legitimate uses.
3. Statistical token-choice detection generally needs the exact generation
   scheme, key, tokenizer, and detector parameters. A detector configured with
   a research key is not a detector for arbitrary OpenAI, Anthropic, Gemini, or
   other provider output.
4. Rewriting can disrupt statistical watermarks, but every reviewed practical
   rewrite approach uses another model. That is explicitly excluded here.
5. Variation selectors deserve explicit detection because they can alter
   tokenization while remaining visually unobtrusive. Blanket removal is not a
   safe default because variation selectors can be meaningful for emoji, CJK,
   and Mongolian text.
6. Detection must run both before and after any transformation. A successful
   local sanitation result proves only that the configured deterministic
   carrier classes were removed.

## Detection Semantics

The implementation should report one of these evidence classes:

| Evidence class | Meaning | Allowed claim |
| --- | --- | --- |
| `unicode_carrier` | A configured suspicious Unicode class was found | `signal_detected`; never claim vendor attribution |
| `same_key_statistical` | A configured detector matched using the expected scheme, key, tokenizer, and threshold | `watermark_detected` for that configured scheme only |
| `vendor_authoritative` | A future vendor detector reports a result | Use the vendor's exact result and limitations |
| `heuristic_style` | Stylometry or phrase/style heuristics fired | Informational only; do not call it a watermark |
| `unsupported` | The text may use a scheme for which no detector is configured | No detection conclusion |
| `inconclusive` | Too little text, detector error, truncation, or ambiguous evidence | No detection conclusion |

The initial implementation should enable only `unicode_carrier`. The registry
should leave room for `same_key_statistical` detectors, but those detectors
must be disabled unless all required scheme inputs are explicitly configured.

## Text Surfaces

Scanning and mutation must be path-aware. Do not recursively rewrite every
string in the JSON body.

### Request surfaces

Scan these visible text nodes:

- Responses `instructions`.
- Responses top-level string `input`.
- Responses message items under `input`, including string content and
  `input_text` or text content parts.
- Chat Completions `messages[*].content` strings and text content parts.
- Text tool results only when the configured scan scope explicitly includes
  them.

Record the role and JSON path for every scanned node. This allows a session
history query to distinguish current user text, prior assistant text, system
instructions, and tool output.

Do not mutate these request surfaces:

- Tool definitions or JSON schemas.
- Tool-call arguments.
- IDs, item references, URLs, file references, or model names.
- Encrypted reasoning state.
- Reasoning signatures or provider continuation state.
- Arbitrary metadata.

Detection may optionally inspect protected or code-like surfaces, but removal
must remain off for them unless a later, separately approved policy defines a
safe transformation.

### Response surfaces

For non-streaming Responses payloads, scan:

- Top-level `output_text`, when present.
- `output[*].content[*].text` for visible text/output-text parts.
- Visible refusal text if the response schema represents it as text.

For non-streaming Chat Completions payloads, scan:

- `choices[*].message.content` strings and text content parts.

Do not inspect or mutate:

- Tool-call arguments.
- Encrypted reasoning content.
- Signatures.
- IDs or provider state.
- Structured non-text output.

For streaming Responses payloads, recognize at least:

- `response.output_text.delta`.
- `response.output_text.done`.
- Text-bearing `response.content_part.added` and
  `response.content_part.done` events.
- Text in the terminal `response.completed`, `response.failed`, or
  `response.incomplete` response envelope when present.

The stream transformer must parse SSE frames and JSON. It must never run a
Unicode replacement over raw SSE bytes, because that could alter event names,
JSON syntax, IDs, usage, tool arguments, or encrypted state.

## Proposed Module

Add a focused module, for example:

```text
litellm/proxy/pass_through_endpoints/aawm_text_watermark/
  __init__.py
  config.py
  text_nodes.py
  unicode_detector.py
  policy.py
  responses_stream.py
```

Suggested responsibilities:

- `config.py`: validate the server policy and expose an immutable runtime
  configuration.
- `text_nodes.py`: yield typed text-node references and apply replacements to
  only supported paths.
- `unicode_detector.py`: inspect and conservatively sanitize Unicode text.
- `policy.py`: run detect, optional transform, re-detect, and build bounded
  audit objects.
- `responses_stream.py`: parse SSE frames, maintain per-output-item text state,
  and transform only visible text fields.

Keep the initial implementation dependency-free by using Python's standard
`unicodedata`, `codecs`, and `json` modules. Do not add Torch, Transformers,
MarkLLM, or a local model dependency to the proxy.

## Unicode Policy

Use two explicit policies.

### `conservative`

Suitable as the default programmatic removal mode:

- Remove noncharacters and reserved default-ignorable code points that have no
  interchange meaning.
- Remove clearly unsafe isolated zero-width and format controls.
- Normalize configured exotic spaces to U+0020.
- Preserve context-valid script joiners.
- Preserve valid emoji glue and flag tag sequences.
- Preserve context-valid CJK/Mongolian variation selectors.
- Preserve paired, context-valid directional formatting.
- Detect but do not replace confusable letters.
- Detect but do not blanket-remove all variation selectors.
- Use NFC only; do not enable NFKC by default.

### `aggressive`

Explicit opt-in only:

- Includes the conservative policy.
- Removes all configured format controls and variation selectors.
- Optionally applies NFKC.
- Optionally maps a configured confusable subset.

Aggressive mode can alter emoji presentation, CJK glyph selection, Mongolian
orthography, bidirectional text, compatibility characters, and identifiers. It
must never be the default and should not be available as an untrusted
per-request override.

## Configuration

Add a typed `ConfigGeneralSettings` field:

```yaml
general_settings:
  openai_passthrough_text_watermark:
    mode: off
    directions:
      request: true
      response: true
    endpoints:
      - responses
      - chat_completions

    unicode:
      enabled: true
      policy: conservative
      normalize_spaces: true
      nfkc: false
      detect_confusables: false

    removal:
      enabled: false
      stream_policy: audit_only
      on_unremovable: allow

    statistical_detectors: []

    limits:
      max_text_bytes_per_direction: 1048576
      max_text_nodes_per_direction: 256
      max_reported_paths: 32
      max_reported_hits_per_path: 16
```

### Modes

| Mode | Behavior |
| --- | --- |
| `off` | No scan, mutation, or metadata |
| `detect` | Scan request and response; never mutate |
| `sanitize` | Scan, sanitize enabled deterministic carrier classes, re-scan, and continue |
| `enforce` | Require configured disallowed signals to be absent after sanitation or reject the operation |

`removal.enabled` should be required for mutation even when `mode` is
`sanitize` or `enforce`. This makes the destructive setting explicit.

### Stream policies

| Policy | Behavior |
| --- | --- |
| `audit_only` | Preserve normal streaming; detect incrementally; never mutate response text |
| `safe_subset` | Stream while removing only code points whose decision does not require future context; report `best_effort` |
| `buffer_text_item` | Buffer each output-text item until its done event, sanitize it, then emit replacement events |
| `buffer_response` | Buffer the complete response stream, sanitize and verify, then emit; required for strict response enforcement |

`buffer_response` changes latency and time-to-first-token behavior. It is the
only simple policy that can promise no configured carrier was delivered before
the final detection result was known.

Do not permit `enforce` plus streamed response output unless
`stream_policy: buffer_response` is selected. Once bytes have been emitted,
LiteLLM cannot truthfully retract them.

### Statistical detector configuration

Keep the schema generic:

```yaml
    statistical_detectors:
      - name: internal_keyed_gumbel
        type: keyed_gumbel
        enabled: false
        tokenizer: internal-model-tokenizer
        key_secret_ref: os.environ/INTERNAL_WATERMARK_KEY
        threshold: 2.33
        minimum_tokens: 64
```

The audit must include the detector name and a non-secret configuration
version, but never the key. A detector must return `unsupported` when its
tokenizer, key, or scheme does not match the text's expected producer.

## Request Flow

Use two request checkpoints. The intake checkpoint belongs in
`BaseOpenAIPassThroughHandler._base_openai_pass_through_handler` immediately
after JSON parsing. The egress checkpoint must run in every actual outbound
request owner after route preparation and pre-call hooks, immediately before
the provider request is built or sent.

Recommended order:

1. Parse the original request body.
2. Resolve the watermark policy for this endpoint and request identity.
3. Extract and detect supported text nodes in the harness-original body.
4. Store a request-scoped draft audit and continue existing Codex/xAI request
   preparation, alias dispatch, observability preparation, and pre-call hooks.
5. Immediately before each upstream send, extract supported nodes from the
   final provider-bound body.
6. Sanitize configured node classes when enabled.
7. Re-detect the exact upstream-bound text and finalize
   `watermark_input_audit`.
8. Put the finalized audit in the callback-visible
   `kwargs["litellm_params"]["metadata"]` and, when the request body carries
   LiteLLM observability metadata, in `litellm_metadata`.
9. Send exactly the finalized provider-bound body.

The egress helper must cover generic `pass_through_request`, the
`try_dispatch_codex_request_fn` early-return path, and managed xAI/Grok request
owners. No outbound branch may bypass it. A request-scoped policy context can
carry the intake result across these branches without rescanning the original
body or storing raw text.

The request audit must distinguish:

- `harness_original`: what LiteLLM received.
- `upstream_sent`: what LiteLLM sent after configured sanitation.

If enforcement rejects a request, return a bounded policy error before upstream
egress. The current session-history failure builder does not create a primary
history row for every generic pre-egress rejection, so implementation must add
an explicit watermark-policy failure-record case. When a session ID is
available, that case should build the normal bounded session-history record
with `source_status: failure` and `watermark_input_audit.status: blocked`.

## Response Flow

### Non-streaming

The generic pass-through path already reads the body, parses JSON, performs
response repairs, and then schedules the success handler. Insert the policy
after existing schema/provenance repairs and before:

- `capture_passthrough_shape`.
- `pass_through_async_success_handler`.
- Construction of the final FastAPI `Response`.

Recommended order:

1. Parse and perform existing provider-specific response repairs.
2. Detect on supported provider-response text.
3. Sanitize when enabled.
4. Re-detect delivered text.
5. Add `watermark_output_audit` to the logging metadata.
6. Pass the transformed response body to success logging.
7. Serialize and return the same transformed body to the harness.
8. Remove `content-length` whenever serialization changed.

Apply the same contract to managed Codex/xAI response owners that return before
the generic pass-through finalizer. The policy helper must run before their
success callback and before their response is returned. Mutating a FastAPI
response later in `BaseOpenAIPassThroughHandler` would be too late because
generic non-stream success logging is already scheduled inside
`pass_through_request`.

The response audit must distinguish:

- `provider_received`: visible text received from upstream after existing
  protocol repair but before watermark sanitation.
- `client_delivered`: visible text returned to the harness.

### Streaming

Integrate with `PassThroughStreamingHandler.chunk_processor` after existing
encrypted-reasoning provenance stamping and before a chunk is recorded for
logging or yielded to the client.

Maintain separate state for:

- Provider-received text detection.
- Client-delivered text detection.
- Per-response/output-item UTF-8 and SSE framing.
- Bounded lookbehind/lookahead needed by the selected Unicode policy.
- Audit counters and truncation flags.

Logging reconstruction must consume the delivered chunks so session history,
Langfuse, and the harness agree about the visible output. The audit metadata
retains the fact that a provider-received signal existed before sanitation.
The finalized output audit must be attached to
`success_handler_kwargs["litellm_params"]["metadata"]` before streaming success
callbacks are scheduled.

## Session-History Representation

Use the existing `session_history.metadata` JSONB column. A schema migration is
not required for the initial implementation.

Add these keys to `_AAWM_SESSION_HISTORY_METADATA_KEYS`:

```text
watermark_input_audit
watermark_output_audit
```

Each object should be bounded and contain no raw prompt or response text:

```json
{
  "schema_version": 1,
  "direction": "request",
  "mode": "sanitize",
  "status": "sanitized",
  "signal_detected": true,
  "confirmed_watermark_detected": false,
  "vendor_attribution": "unknown",
  "scanned_text_nodes": 3,
  "scanned_text_bytes": 1482,
  "skipped_text_nodes": 1,
  "detectors": [
    {
      "name": "unicode_carrier",
      "version": 1,
      "status": "detected",
      "confidence": "probable",
      "hit_count": 4,
      "hit_kinds": ["zwj_family", "variation_selector"]
    }
  ],
  "transformation": {
    "attempted": true,
    "policy": "conservative",
    "removed_count": 2,
    "replaced_count": 0,
    "post_status": "signal_remaining",
    "result": "partially_sanitized"
  },
  "paths": [
    {
      "path": "input[2].content[0].text",
      "role": "user",
      "hit_count": 4,
      "hit_kinds": ["zwj_family", "variation_selector"]
    }
  ],
  "truncated": false,
  "errors": []
}
```

Recommended top-level `status` values:

- `clean`
- `detected`
- `sanitized`
- `partially_sanitized`
- `blocked`
- `unsupported`
- `inconclusive`
- `error`

Recommended post-transformation results:

- `not_requested`
- `removed_verified`
- `partially_sanitized`
- `detected_unremoved`
- `removal_failed`
- `unsupported`
- `inconclusive`

Use `removed_verified` only when re-running the same applicable detector proves
the configured signal is absent. For a statistical vendor watermark with no
authoritative detector, use `unsupported` or `inconclusive`, never
`removed_verified`.

The path list and detector details must be capped. Store aggregate counts after
the cap and set `truncated: true`. Do not store raw characters, surrounding
text, detector keys, full prompts, full responses, or unkeyed content hashes in
session history.

## Failure Behavior

Detection should be fail-soft in `detect` and `sanitize` modes:

- A detector error is recorded as `error`.
- Other configured detectors continue.
- The request or response continues unless the policy explicitly says
  otherwise.

`enforce` should fail closed only for configured evidence classes and
confidence levels:

- Request enforcement can reject before egress.
- Non-streaming response enforcement can reject before returning output.
- Streaming response enforcement requires complete buffering.

Policy rejections must enter the failure callback path with the relevant
bounded input/output audit already attached. Extend the session-history failure
builder to create a primary row for this recognized failure class; do not rely
on rate-limit, provider-error, or structured-output detection to make the row
exist.

An unavailable same-key detector must not silently become a clean result.
Report `unsupported` or `inconclusive`.

## Performance and Privacy Limits

- Scan Unicode in O(n) time.
- Keep reports bounded independently from scanned input size.
- Reject or mark `inconclusive` after configurable text-node and byte limits.
- Do not log raw matching text.
- Do not load model/tokenizer dependencies unless a statistical detector is
  explicitly configured.
- Cache immutable detector configuration, but do not cache prompt or response
  text.
- Record detection and sanitation duration in the audit object or existing
  latency metadata only when needed for operations.

## Proposed Implementation Surfaces

The minimum source changes for implementation are:

1. Add the `aawm_text_watermark` package.
2. Add the typed general setting in `litellm/proxy/_types.py`.
3. Apply request policy in
   `aawm_adapter_runtime/openai_passthrough_handler.py`.
4. Apply non-stream response policy in
   `pass_through_endpoints/pass_through_endpoints.py`.
5. Apply stream policy in
   `pass_through_endpoints/streaming_handler.py`.
6. Add the two audit keys to
   `integrations/aawm_agent_identity/constants.py`.
7. Add explicit policy-block handling to
   `integrations/aawm_session_history/record.py`.
8. Apply the shared request/response policy contract in managed Codex/xAI
   egress owners that bypass the generic pass-through finalizers.
9. Document the configuration in the OpenAI passthrough consumer
   documentation when implementation is authorized.

No new database table or column is needed for the initial version.

## Focused Acceptance Evidence

When implementation is authorized, the focused acceptance surface should
demonstrate:

1. A request containing a zero-width carrier is detected and recorded in
   `watermark_input_audit`.
2. In `sanitize` mode, the upstream-bound request contains the cleaned text and
   the audit distinguishes original from sent state.
3. A non-streaming response containing a configured carrier is detected,
   optionally sanitized, re-detected, logged, and returned with matching
   delivered text.
4. A streaming response records provider-received detection and, under the
   selected stream policy, either preserves the stream unchanged or emits only
   sanitized text.
5. Tool arguments, encrypted reasoning, IDs, URLs, code blocks, and structured
   data remain byte-identical.
6. Context-valid emoji/script joiners and variation selectors survive the
   conservative policy.
7. Aggressive mode removes configured variation selectors only when explicitly
   enabled.
8. Session history contains bounded audit metadata and no raw text.
9. An unavailable statistical detector produces `unsupported` or
   `inconclusive`, not `clean`.
10. Strict streamed-response enforcement is rejected at configuration time
    unless complete response buffering is enabled.

## Non-Goals

- Proving whether arbitrary text was generated by a particular vendor.
- Claiming human authorship after sanitation.
- Removing C2PA or media metadata.
- Rewriting prose with another model.
- Adding a general AI-writing detector.
- Mutating tool calls, code, schemas, encrypted state, signatures, or
  structured data.
- Enabling aggressive removal by default.

## Decision

The feasible no-second-model feature is deterministic carrier detection and
sanitation plus honest audit metadata. The design can also host exact
same-key statistical detectors, but those detectors are useful only for
schemes LiteLLM controls or for future authoritative vendor integrations.

A generic configuration switch cannot truthfully guarantee removal of an
unknown statistical token watermark. The system should say that explicitly in
both its configuration documentation and its `session_history` evidence.
