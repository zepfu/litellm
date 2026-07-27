# Pass-Through Endpoints Architecture

## Why Pass-Through Endpoints Transform Requests

Even "pass-through" endpoints must perform essential transformations. The request **body** passes through unchanged, but:

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as LiteLLM Proxy
    participant Provider as LLM Provider

    Client->>Proxy: POST /vertex_ai/v1/projects/.../generateContent
    Note over Client,Proxy: Headers: Authorization: Bearer sk-litellm-key
    Note over Client,Proxy: Body: { "contents": [...] }
    
    rect rgb(240, 240, 240)
        Note over Proxy: 1. URL Construction
        Note over Proxy: Build regional/provider-specific URL
    end
    
    rect rgb(240, 240, 240)
        Note over Proxy: 2. Auth Header Replacement
        Note over Proxy: LiteLLM key → provider credentials
    end

    rect rgb(240, 240, 240)
        Note over Proxy: 3. Extra Operations
        Note over Proxy: • x-pass-* headers (strip prefix, forward)
        Note over Proxy: • x-litellm-tags → metadata
        Note over Proxy: • Guardrails (opt-in)
        Note over Proxy: • Multipart form reconstruction
    end
    
    Proxy->>Provider: POST https://us-central1-aiplatform.googleapis.com/...
    Note over Proxy,Provider: Headers: Authorization: Bearer ya29.google-oauth...
    Note over Proxy,Provider: Body: { "contents": [...] } ← UNCHANGED
    
    Provider-->>Proxy: Response (streaming or non-streaming)

    rect rgb(240, 240, 240)
        Note over Proxy: 4. Response Handling (async)
        Note over Proxy: • Collect streaming chunks for logging
        Note over Proxy: • Cost injection (if enabled)
        Note over Proxy: • Parse response → calculate cost → log
    end
    
    Proxy-->>Client: Response (unchanged)
```

## Essential Transformations

- **URL Construction** - Build correct provider URL (e.g., regional endpoints for Vertex AI, Bedrock)
- **Auth Header Replacement** - Swap LiteLLM virtual key for actual provider credentials

## Extra Operations

| Operation | Description |
|-----------|-------------|
| `x-pass-*` headers | Strip prefix and forward (e.g., `x-pass-anthropic-beta` → `anthropic-beta`) |
| `x-litellm-tags` header | Extract tags and add to request metadata for logging |
| Streaming chunk collection | Collect chunks async for logging after stream completes |
| Multipart form handling | Reconstruct multipart/form-data requests for file uploads |
| Guardrails (opt-in) | Run content filtering when explicitly configured |
| Cost injection | Inject cost into streaming chunks when `include_cost_in_streaming_usage` enabled |

## What Does NOT Change

- Request body
- Response body
- Provider-specific parameters

## AAWM shared pass-through request engine (RR-056)

`pass_through_endpoints.py` is the shared HTTP/WebSocket pass-through core used
by provider routes. Operator-facing behaviors that are easy to miss:

### Hidden pre-first-byte retries (issue #1)

- Pre-first-byte upstream 5xx/429/timeout/connect failures may be retried with
  fixed backoff inside a single inbound request.
- Total wall-clock budget is bounded by
  `AAWM_PASSTHROUGH_HIDDEN_RETRY_BUDGET_SECONDS` (default = sum of backoff
  schedule, currently 230s). Set `0` to disable the wall-clock bound while
  keeping the attempt-count ceiling.
- This is independent of the per-attempt HTTP client timeout.

### xAI failure capture (issue #2)

- Direct `_direct_capture_xai_passthrough_failure` is a **fallback only**.
- When `AawmAgentIdentity` is already registered on `litellm.callbacks` /
  `failure_callback`, the direct path is skipped to avoid double
  session_history / failure bookkeeping.

### Provider failure classifiers (issue #3)

- Known vendor failure kinds live under
  `provider_failure_classifiers/` (per-provider modules + `registry.py`).
- The shared request engine imports
  `_run_passthrough_provider_failure_classifiers` from that package and runs a
  registry-style dispatch; new vendor quirks register in the package rather
  than growing the god-module exception path.

### WebSocket message buffer (issue #4)

- Upstream WS messages retained for success-handler cost/logging use a bounded
  ring buffer (`_WEBSOCKET_PASSTHROUGH_MESSAGE_BUFFER_MAX`, default 256).
- Long-lived realtime sessions (e.g. Vertex Live) must not grow unbounded.

### Failure-hook transformation (issue #5)

- `post_call_failure_hook` return values that are real `BaseException` instances
  are applied on the pass-through failure path (same contract as
  `common_request_processing` / auth exception handling).

### Non-stream SSE detection (issue #6)

- Non-GET "non-stream" sends use `stream=True` so content-type can be inspected
  before full-body buffering; true SSE hands off to the streaming handler.
- Non-SSE bodies (success and error) are drained with `aread()`.

### Tool-schema normalization gate (issue #9)

- OpenAI function-tool `type: object` `properties` fixes run only for
  OpenAI-like targets (provider/endpoint/host gate), not for every pass-through.

### Route registry lookup (issue #10)

- Registered pass-through routes maintain exact/subpath path indexes for
  per-request lookup; `is_registered_pass_through_route` reuses
  `get_registered_pass_through_route`.


### Agent-identity import (issue #7 / RR-003)

- Direct xAI capture uses a single canonical import:
  `litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance`.
- RR-003 packaging force-includes that module into the published
  `aawm_litellm_callbacks.agent_identity` wheel surface; the checkout wheel
  loader re-exports the same module. No dual runtime probe remains.
- Callback *registration* markers still recognize either package path so
  configs that list the wheel dotted name continue to skip double-capture.

### Inline imports (issue #8)

- Non-circular helpers (`all_litellm_params`, `get_end_user_id_from_request_body`,
  `persist_agent_terminal_error`) are module-scoped.
- `proxy_logging_obj` / `Logging` remain function-local because `proxy_server`
  imports this module at startup.

### Chat-completion body parse (issue #11)

- `chat_completion_pass_through_endpoint` parses bodies with `json.loads` only
  (no `ast.literal_eval` first attempt).

## AAWM Claude control-plane (fork overlay)

`aawm_claude_control_plane.py` owns Claude Code request rewrites and dynamic
injection for this fork. Operator-facing behavior that is easy to miss:

### Trust boundary

- Explicit AAWM HTML/`@@@` directives may appear anywhere in `system` /
  `messages`.
- `:#name.ctx#:` markers and SubagentStart dispatch backtick/acronym grabs run
  only on trusted surfaces: `system` and the first user message. Later tool /
  web / assistant text cannot trigger same-tenant content-store lookups.

### Lookup fan-out budget (RR-053)

Dispatch backtick/acronym expansion is intentionally bounded:

- **Per text node:** at most `_AAWM_DISPATCH_CONTEXT_REFERENCE_MAX` (24)
  distinct references.
- **Per request:** at most `_AAWM_DISPATCH_CONTEXT_REFERENCE_REQUEST_MAX` (48)
  distinct lookups across *all* trusted text blocks, with request-wide
  deduplication of names already resolved earlier in the walk.
- Common all-caps noise tokens (`SQL`, `JSON`, `API`, ...) are stopworded.
- Pool acquire uses a finite timeout
  (`AAWM_DYNAMIC_INJECTION_ACQUIRE_TIMEOUT_SECONDS`, default 10s).

Per-node caps alone are not enough: system + first-user content can contain many
blocks; the request-wide budget is what stops unbounded DB fan-out.

### Connection pool lifecycle

The control-plane module owns the canonical process-wide asyncpg pool used for
dynamic injection / context grabs **and** for sibling AAWM Postgres lookups that
share the same DSN (for example OpenRouter free-tier durable quota reads in
`llm_passthrough_endpoints.py`). `close_aawm_dynamic_injection_pool()` is
invoked from `proxy_shutdown_event()` so connections are released on clean
proxy shutdown.

`llm_passthrough_endpoints` re-exports the control-plane pool/DSN helpers
(`_get_aawm_dynamic_injection_pool`, `_build_aawm_dynamic_injection_dsn`, …)
for stable import compatibility, but it must not create a second asyncpg pool.

### Control-plane rewrite scope

`apply_claude_control_plane_rewrites_to_anthropic_request_body()` rewrites only
`system` and the first user message (auto-memory, prompt-patch manifest,
CommonMark identifier list). Full history is not re-scanned every turn.

## AAWM alias routing and adapter ownership (RR-054)

`llm_passthrough_endpoints.py` retains FastAPI route registration and thin
compatibility wrappers, plus runtime assembly. Provider preparation and
provider algorithms, along with shared policy, state, I/O, shaping, retry
sequencing, stream validation, and response finalization, are package-owned:

| Concern | Owner |
|--------|--------|
| Candidate tables, aliases, model allowlists, cooldown defaults | `aawm_alias_routing/policy.py` |
| Cooldown, affinity, OAuth, lane-cache, and candidate probe-lock state | `aawm_alias_routing/state.py` |
| Durable Redis keys, max-expiry writes, negative reads, DualCache coherency | `aawm_alias_routing/durable.py` |
| Alias-routing Redis connection, DualCache attachment, self-heal, write-retry policy, readiness status | `litellm/proxy/aawm_alias_routing_redis.py` |
| Google and Antigravity OAuth file/token I/O | `aawm_alias_routing/google_oauth.py`, `aawm_alias_routing/antigravity_oauth.py` |
| Config-driven nine-route execution plans | `aawm_alias_routing/adapter_config.py`, `aawm_alias_routing/adapter_driver.py` |
| Shared Anthropic-to-Responses/completion shaping orchestration | `litellm/llms/anthropic/experimental_pass_through/providers/common.py` |
| Provider request shaping, credential/target preparation order, and route-plan construction | `litellm/llms/anthropic/experimental_pass_through/providers/<provider>/adapter.py` |
| Google/Antigravity provider preparation and Google process-local project/prime cache algorithms | `providers/google/adapter.py`, `providers/google/process_cache.py`, `providers/antigravity/adapter.py` |
| Grok argument/input normalization and composer response repair | `providers/grok/normalization.py`, `providers/grok/composer_repair.py` |
| OpenCode Zen request and stream normalization | `providers/opencode_zen/normalization.py` |
| OpenCode Zen constants (base URL, provider ID, auth paths, free models) | `providers/opencode_zen/constants.py` |
| Antigravity constants (forward header allowlist) | `providers/antigravity/constants.py` |
| Model resolution and adapter model normalization | `aawm_adapter_runtime/model_resolution.py` |
| Response-body inspection, malformed intake context, Grok Composer repair, schema validation, custom-tool argument parsing | `aawm_adapter_runtime/request_build.py` |
| Provider-neutral Responses SSE framing, parsing, summaries, repaired IDs, and Anthropic stream wrappers | `aawm_adapter_runtime/sse.py` |
| Custom and namespace tool-call restoration for response bodies and SSE streams | `aawm_adapter_runtime/tool_call_restore.py` |
| Responses stream accumulation, output merging, finalization, and empty-success diagnostics | `aawm_adapter_runtime/stream_collect.py` |
| Responses payload validation, failure taxonomy, bounded replay, and malformed/empty-success rejection | `aawm_adapter_runtime/payload_validation.py` |
| Lane key generation and session affinity | `aawm_alias_routing/lane_keys.py` |
| Google environment policy and configuration | `providers/google/env_policy.py` |
| Google context window management | `providers/google/context_window.py` |
| Google error signal extraction | `providers/google/error_signals.py` |
| Google adapter retry, cooldown, transient failure handling, hidden retry metadata, and transient status constant | `providers/google/retry_runtime.py` |
| Google Code Assist request building, tool replay, tool-call caches, schema sanitization, response translation, and stream shaping | `providers/google/codex_code_assist.py` |
| Grok side channel endpoint classification | `providers/grok/side_channel.py` |
| OpenRouter retry and transport algorithms | `providers/openrouter/retry_transport.py` |
| Cross-provider text/JSON shaping primitives | `aawm_alias_routing/provider_shaping.py` |
| Bounded, lossless stream peeking | `aawm_alias_routing/streaming.py` |
| Responses-to-Anthropic finalization | `aawm_alias_routing/responses_finalize.py` |
| Shared retry attempt sequencing and cooldown waits | `aawm_alias_routing/retry.py` |
| Shared alias candidate retry loop and R3-1 single-flight cooldown publication | `aawm_alias_routing/candidate_loop.py` |
| Typed alias-route seam contracts (`AliasRouteServices`, cooldown publication plan) | `aawm_alias_routing/interfaces.py` |
| Structured task-state source selection | `aawm_alias_routing/task_state.py` |
| Snapshot ordering, distribution strategy, TUI/schedule gates, selection-context memoization, alias-candidate getters | `aawm_alias_routing/snapshot_select.py` |
| Config refresh handler body and YAML source loading (decorator stays in god module) | `aawm_alias_routing/config_refresh.py` |
| Codex auth-file discovery, JWT decode, token validation, Codex-native-auth request detection | `aawm_alias_routing/codex_oauth.py` |
| OpenRouter free-daily-quota probe, durable cooldown helpers, alias-probe cooldown gate | `aawm_alias_routing/openrouter_quota.py` |
| Failure token extraction, retryable-exhaustion classification, cooldown scope/duration derivation, Kimi safe metadata, Grok account-lane keys, native-Grok continuation retry planning | `aawm_alias_routing/error_signals.py` |
| Immutable cooldown publication-plan resolution, post-lock durable persistence, Codex/Anthropic cooldown application, read-pilot gated application | `aawm_alias_routing/cooldown_apply.py` |
| Attempt lifecycle mutation, read-pilot evidence recording, reasoning-effort normalization, Codex/Anthropic alias metadata composition | `aawm_alias_routing/attempt_records.py` |
| Route registration, process-local routing call sites, compatibility re-exports, and residual provider orchestration | `llm_passthrough_endpoints.py` |

The legacy `aawm_alias_routing_policy.py` module is a compatibility facade over
`aawm_alias_routing/policy.py`; it does not redeclare candidate tables or
policy constants. `aawm_alias_routing_policy.pyi` supplies its static public
contract. Google follows the same pattern: `providers/google/shaping.py`
re-exports functions owned by focused implementation modules and
`providers/google/shaping.pyi` supplies the type-checking contract. The facade
modules preserve import and monkeypatch compatibility without creating a second
policy or shaping owner.

### Managed Kimi native contract boundary

The local `/kimi/v1/models`, `/kimi/v1/usages`, and
`/kimi/v1/chat/completions` facade routes resolve to the exact managed upstream
base `https://api.kimi.com/coding/v1` through the shared Kimi native-contract
resolver. Managed `kimi_code/*` traffic is OAuth-only. The descriptor controls
the native header contract with service-owned, non-personal identity values;
caller authorization, user-agent, `X-Msh-*`, device, session, and endpoint
identity cannot override it.

Generic `moonshot/*` API-key routes and generic `/v1` behavior are separate
upstream LiteLLM functionality. They are not consumers of the managed Kimi
descriptor and are never a fallback for the local `/kimi/v1` facade.

The nine Anthropic adapter route entrypoints delegate provider preparation to
the corresponding `providers/<provider>/adapter.py` module, then delegate
execution to the shared package drivers. The provider modules own transform
ordering, provider policy application, credential/target preparation, and
route-plan construction. `llm_passthrough_endpoints.py` injects route-layer
services through immutable runtime contracts so it retains FastAPI request
objects, environment/credential access points, egress validation, and transport
callbacks. The provider algorithm extraction is complete; the god-file still
owns the integration call sites and runtime assembly for these modules, including
Google/Antigravity cache-lifecycle callbacks, provider normalization/repair
invocation, request/stream handoff, transport callbacks, route-layer error
mapping, and final response delivery. Those callbacks do not move substantive
provider algorithms back into the route module. Responses finalization is
configured with explicit runtime callbacks so that control flow is
package-owned without importing the FastAPI route module.

#### Wave 5A deferred state (resolved in Wave 5B)

The following process-local state moved onto `AliasRoutingStateManager` in
Wave 5B. God-module facades and Wave 5A modules access them through manager
methods or injected callbacks:

| State | Owner | Access seam |
|-------|-------|-------------|
| `_openrouter_free_daily_quota_cache` (tuple) | `AliasRoutingStateManager` | `get/set_openrouter_free_quota_cache()` methods; god-module `__getattr__` compat |
| `_openrouter_free_daily_quota_lock` (asyncio.Lock) | `AliasRoutingStateManager` | shared lock object via `openrouter_free_quota_lock` |
| `_round_robin_cursor_by_alias` (dict) | `AliasRoutingStateManager` | `round_robin_cursor` attribute; snapshot_select via injected reference |
| `_read_pilot_cooldown_gate` (CooldownEvidenceGate) | `AliasRoutingStateManager` | `read_pilot_gate` attribute with separate `AliasFamilyState` |
| FastAPI route registration (`@router.post`, etc.) | `llm_passthrough_endpoints.py` | decorators stay in god module; handler bodies in Wave 5A modules |

#### Wave 5B ownership

| Concern | Module |
|---------|--------|
| Codex/Anthropic active cooldown reads/writes, session affinity, merged-family state, R3-1 memory publication, state-source attachment | `aawm_alias_routing/cooldown_state.py` |
| Candidate lookup, state construction, availability shaping, request-local cooldown/exclusion, forced/adapter/Kimi/Grok lane application, Codex/Anthropic selectors, in-flight/redispatch errors | `aawm_alias_routing/selection.py` |
| Read-pilot evidence gate, round-robin cursor, OpenRouter quota cache/lock | `aawm_alias_routing/state.py` (`AliasRoutingStateManager`) |

#### Wave 5C ownership and baseline parity

Wave 5C moves 50 functions from `llm_passthrough_endpoints.py` at baseline
`79fc94c3a5` into `error_signals.py` (33), `cooldown_apply.py` (8), and
`attempt_records.py` (9). The god module exposes same-object compatibility
facades; the functions retain their owning module globals, while host services
are supplied through explicit runtime configuration callbacks. This preserves
the existing god-module monkeypatch surface used by `candidate_loop.py` without
making any Wave 5C module import the god module at module scope.

The normalized AST contract matches all 50 baseline bodies and signatures.
Raw source differences are limited to extraction mechanics:

| Difference | Scope | Reason |
|------------|-------|--------|
| Fail-fast `assert` guards on configured callbacks | 25 frozen functions | Surface incomplete runtime wiring at the owning module boundary instead of failing later with an opaque `NoneType` call |
| Renamed callback globals and direct `CooldownPublicationPlan` import | Seven `cooldown_apply.py` functions | Keep publication resolution, durable persistence, request-local application, family setters, read-pilot gate, and state manager explicit while preserving control flow and target keys |
| Classification/gate/model-catalog callback names | Three `attempt_records.py` functions | Keep Wave 5D audit services and model information as configured host dependencies |
| Local FastAPI/status imports in source-error summarization | `_get_codex_auto_agent_source_error_summary` | Avoid a module-scope god-module dependency while preserving the same `HTTPException` construction |
| Same-object `install()` facade publication | All three Wave 5C modules | Preserve module globals and explicit callbacks; avoid host-global rebinding that breaks module imports and cross-suite isolation |

`_apply_request_local_cooldown_from_plan` and
`_apply_codex_auto_agent_grok_account_lane_cooldown` remain defined by
`selection.py`. Synchronous memory publishers remain defined by
`cooldown_state.py`. The candidate loop resolves Wave 5C callables through the
god-module facades at invocation time so existing tests and operators can
monkeypatch the compatibility surface.

Codex continuation affinity fails closed when its pinned candidate is absent
from the active enumeration or its route family is incompatible. This applies
equally to memory and durable-cache affinity: the selector raises
redispatch-required before resolving a provider lane or selecting an alternate
upstream.

Google retry classification retains separate capacity, rate-limit, transient,
and request budgets through strategy callbacks. The shared retry driver owns
only identical attempt sequencing; it does not collapse Google's multi-budget
semantics into OpenRouter policy.

#### Wave 5D ownership and baseline parity

Wave 5D moves 23 symbols (14 functions, 1 TypedDict class, 6 constants, and 2
integrator-owned orchestration functions) from `llm_passthrough_endpoints.py`
at baseline `66963d07ce` into four modules:

| Concern | Module | Symbols |
|---------|--------|---------|
| Request context extraction, agent dispatch fields, role inference, prior-tool traversal, activity classification, request-call ID, terminal context attachment | `aawm_alias_routing/audit_context.py` | 14 functions + 1 TypedDict + 6 constants |
| Audit event construction, in-flight exception detection, continuation-state recursion | `aawm_alias_routing/audit_build.py` | 4 functions |
| Route-event emission filtering, audit-only spool/enqueue persistence | `aawm_alias_routing/audit_persist.py` | 3 functions |
| Cross-module terminal-event orchestration, attempt enrichment, no-candidate event emission | `aawm_alias_routing/audit_events.py` | 2 functions |

`audit_events.py` is the integrator-owned compatibility aggregator. It owns
`_enrich_auto_agent_alias_terminal_event_from_attempts` and
`_emit_auto_agent_alias_no_candidate_event`, which orchestrate across
audit_context, audit_build, and audit_persist through injected runtime seams.

The god module exposes same-object facades for every moved symbol. No Wave 5D
module imports `llm_passthrough_endpoints.py` at module scope. Host
dependencies are supplied through `configure_audit_context_runtime`,
`configure_audit_build_runtime`, `configure_audit_persist_runtime`, and
`configure_audit_events_runtime` with late-binding lambdas.

`selection.py` retains exclusive ownership of the five redispatch/in-flight
exception builders (`_raise_codex_auto_agent_in_flight_cooldown`,
`_raise_anthropic_auto_agent_in_flight_cooldown`,
`_build_auto_agent_redispatch_http_exception_detail`,
`_raise_codex_auto_agent_redispatch_required`,
`_raise_anthropic_auto_agent_redispatch_required`). Wave 5D does not create
`redispatch.py` or duplicate these functions.

Behavior-preservation invariants:

- All frozen function bodies and signatures are AST-identical to baseline
  `66963d07ce`, ignoring only explicit fail-fast seam assertions added by the
  author contract.
- `attempt_records.py` receives audit callbacks (`emit_route_event`,
  `build_audit_event`, `build_audit_events`, `persist_audit_only_events`)
  through its existing `configure_attempt_records_runtime` seam, resolving to
  the Wave 5D module functions via god-module facades.
- `candidate_loop.py` resolves `_emit_auto_agent_alias_no_candidate_event`,
  `_codex_auto_agent_request_has_continuation_state`, and
  `_is_auto_agent_alias_in_flight_cooldown_http_exception` through the
  god-module facade surface (`_lpe.<name>`), preserving the existing
  monkeypatch contract.
- Terminal audit context memoization on `request.state` is unchanged.
- Event ordering, omission rules, sanitization, persistence dispositions,
  and synchronous/async behavior are preserved exactly.

#### Wave 6A adapter-runtime ownership

Wave 6A moves exactly 70 functions from `llm_passthrough_endpoints.py` into
five focused `aawm_adapter_runtime` modules:

| Module | Owned functions |
|--------|-----------------|
| `request_build.py` | 22 |
| `sse.py` | 11 |
| `tool_call_restore.py` | 14 |
| `stream_collect.py` | 9 |
| `payload_validation.py` | 14 |

`aawm_adapter_runtime.__init__.install()` installs these modules in dependency
order: request building, SSE, tool restoration, stream collection, then payload
validation. `llm_passthrough_endpoints.py` calls that installer during normal
module initialization; callers and tests do not manually install Wave 6A.

Every moved name is a same-object compatibility facade on the god module. The
installed functions use the live `llm_passthrough_endpoints` globals dictionary,
so existing monkeypatches remain reachable at call time. The one
`lru_cache`-decorated runtime factory is reconstructed around a host-rebound
callable while preserving its cache configuration and same-object facade.

`sse.py` is the canonical owner of `_mapping_or_attr_get` and
`_responses_repaired_output_item_id`. Dependent Wave 6A modules resolve those
names through host-global lookup and do not define or import duplicate owners.
None of the five extracted modules imports `llm_passthrough_endpoints` at module
scope. Constants, route definitions, compatibility wrappers, and provider
runtime paths remain in their pre-Wave 6A owners.

#### Wave 6B provider consumer architecture (pending acceptance)

Wave 6B extracts provider-shaped route runtime and request-preparation
functions from `llm_passthrough_endpoints.py` into focused consumer modules
under `litellm/proxy/pass_through_endpoints/providers/`. Each module owns the
route-layer orchestration for one provider family while delegating retry
algorithms, normalization, constants, and OAuth to their existing upstream
owners. Host dependencies are injected through frozen `Runtime` dataclasses so
no Wave 6B module imports the god module at module scope.

| Concern | Module | Functions |
|---------|--------|-----------|
| OpenRouter route-layer runtime: credential resolution, rate-limit key/wait-key derivation, retry orchestration delegation, response assembly | `providers/openrouter/runtime.py` | 45 |
| NVIDIA adapter target, credential resolution, retryable-status policy, and retry execution | `providers/nvidia/runtime.py` | 15 |
| OpenCode Zen target/auth/header resolution, streaming normalization handoff, Responses SSE framing, and chat-completion sanitization | `providers/opencode_zen/runtime.py` | 28 |
| xAI and Grok-native request preparation: OAuth model detection, upstream model resolution, Codex unsupported-field drops, tool-choice normalization, and Grok passthrough target assembly | `providers/xai/request_prep.py` | 24 |
| Antigravity pass-through runtime: CLI binary discovery, OAuth refresh failure formatting, native header construction, endpoint normalization, and request body preparation | `providers/antigravity/runtime.py` | 14 |
| Shared candidate-unavailable error vocabulary: per-provider detail extraction and structured `ProxyException` raising | `providers/common.py` | 12 |

**Façade and delegate ownership.** Each Wave 6B module delegates substantive
algorithms to pre-existing owners and does not duplicate them:

| Delegate owner | Consumed by |
|---------------|-------------|
| `providers/openrouter/retry_transport.py` (retry mechanics, rate-limit keys, free-model classification) | `providers/openrouter/runtime.py` |
| `providers/opencode_zen/normalization.py` and `providers/opencode_zen/constants.py` | `providers/opencode_zen/runtime.py` |
| `providers/antigravity/adapter.py`, `providers/antigravity/constants.py`, `aawm_alias_routing/antigravity_oauth.py` | `providers/antigravity/runtime.py` |
| `litellm.llms.xai.oauth` (model detection, token acquisition, request preparation) and `providers/grok/normalization.py` | `providers/xai/request_prep.py` |
| `providers/common.py` candidate-unavailable raisers | `providers/opencode_zen/runtime.py`, `providers/antigravity/runtime.py`, `providers/xai/request_prep.py` |

**Configuration and monkeypatch behavior.** All five provider modules receive
host callbacks through explicit `configure_*_runtime()` functions with frozen
dataclass contracts:

- `configure_openrouter_runtime(Runtime)` -- OpenRouter
- `configure_nvidia_runtime(NvidiaRuntimeDependencies)` -- NVIDIA (ships
  `DEFAULT_NVIDIA_RUNTIME_DEPENDENCIES` for standalone use)
- `configure_runtime(Runtime)` -- OpenCode Zen
- `configure_xai_request_prep_runtime(XAIRequestPrepRuntime)` -- xAI/Grok
- Antigravity passes its `Runtime` per-call through function arguments rather
  than a module-global seam.

OpenCode Zen additionally provides `install(host_globals)`, which publishes
same-object facades for every name in `_HOST_FUNCTION_NAMES` into the god
module's globals dictionary. Installed functions resolve host dependencies
through live `host_globals[name]` lookups at call time, so existing
monkeypatches on the god module remain reachable without rebinding.

`providers/xai/request_prep.py` documents its seam contract in
`XAI_REQUEST_PREP_SEAM_DISPOSITION`, mapping each callback name to its
`runtime.<field>` resolution path.

**Candidate-unavailable vocabulary.** `providers/common.py` owns the shared
`_raise_candidate_unavailable` primitive and per-provider wrappers
(`_raise_opencode_zen_auto_agent_candidate_unavailable`,
`_raise_antigravity_auto_agent_candidate_unavailable`,
`_raise_codex_native_openai_auto_agent_candidate_unavailable`,
`_raise_xai_oauth_auto_agent_candidate_unavailable`,
`_raise_grok_native_auto_agent_candidate_unavailable`). Each wrapper extracts
provider-specific detail through an injected `Runtime` (status-code and
detail extraction callbacks) and raises a structured `ProxyException` with
code `aawm_codex_auto_agent_candidate_unavailable`, error type
`rate_limit_error`, and status 429. Provider modules import these raisers
directly; the god module re-exports them for backward compatibility.

#### Wave 6C Google retry and Code Assist extraction

Wave 6C extracts Google adapter retry/cooldown runtime and Google Code Assist
request/stream functions from `llm_passthrough_endpoints.py` into two focused
modules under `litellm/proxy/pass_through_endpoints/providers/google/`. The
package `__init__.py` exports both modules.

| Concern | Module | Symbols |
|---------|--------|---------|
| Google adapter retry sequencing, cooldown waits, rate-limit/transient failure handling, hidden retry metadata, terminal failure logging, and pass-through request execution | `providers/google/retry_runtime.py` | 11 functions + `_GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES` |
| Google Code Assist request building, tool-call replay/repair, tool-call name/argument caches, schema sanitization, response translation, stream collection, and streaming response assembly | `providers/google/codex_code_assist.py` | 45 functions + 3 constants (`_GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH`, `_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_TTL_SECONDS`, `_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE`) |

**Process-cache ownership.** `retry_runtime.py` delegates semaphore acquisition
to `process_cache._get_google_adapter_semaphore` and does not own semaphore or
cache state. `codex_code_assist.py` receives canonical tool-call cache mappings
from `process_cache` through its `Runtime` dataclass; it does not create a
second cache owner. The god module retains canonical process-cache aliases
(project cache/lock, prime quota/lock, token cache, semaphores, user-prompt turn
state) and a direct `_get_google_adapter_semaphore` delegate that calls
`process_cache` with the assembled runtime.

**Configuration order.** The god module calls
`configure_google_retry_runtime(Runtime(...))` before
`codex_code_assist.install(globals())`. This ordering is contractual: the retry
runtime must be fully wired before Code Assist facades resolve host-global
collaborators at call time.

**Facade and monkeypatch contract.** All 11 retry functions are exposed as
same-object aliases on the god module (direct assignment from
`_google_retry_runtime.<name>`). All 45 Code Assist functions are published into
the god module globals dictionary by `install(globals())`. Installed functions
resolve host dependencies through live `host_globals[name]` lookups at call
time, so existing monkeypatches on the god module remain reachable without
rebinding. Neither module imports `llm_passthrough_endpoints` at module scope.

**God-module retention.** `llm_passthrough_endpoints.py` retains FastAPI route
bodies, six route/delegate boundaries for Google traffic, canonical
process-cache aliases, the direct semaphore delegate, compatibility re-export
surfaces, and runtime assembly (constructing `Runtime` dataclasses and calling
`configure_google_retry_runtime` / `codex_code_assist.install`).

**Scope boundary.** Live Google route bodies remain unchanged by Wave 6C. This
wave does not claim ownership of Wave 6D-6F concerns (additional provider
extractions, further route-body decomposition, or cross-provider runtime
consolidation).

#### Wave 6D request-policy ownership

Wave 6D extracts request-policy concerns from `llm_passthrough_endpoints.py`
into three focused modules under
`litellm/proxy/pass_through_endpoints/aawm_request_policy/`:

| Concern | Module | Functions |
|---------|--------|-----------|
| Claude persisted-output expansion, Google adapter compaction delegates, content-text estimation | `persisted_output.py` | 14 |
| Shared metadata primitives, session/repository extraction, tool-definition snapshots, Claude/Gemini/Codex breakouts, Anthropic billing headers, route-family logging | `observability_metadata.py` | 43 |
| Alias-specific system instruction shaping: prevention guidance, read-agent guidance | `alias_guidance.py` | 6 |

Total: 63 functions. No symbol is owned by more than one Wave 6D module, and
no Wave 6D symbol remains a `FunctionDef` in the god module.

**Estimator ownership.** `_estimate_google_content_text_chars` is owned by
`persisted_output.py`, not by `providers/google/env_policy.py`. The
persisted-output module provides a single behavior-compatible implementation
used by both direct module callers and installed host facades, eliminating the
prior dual-path divergence.

**Facade and monkeypatch contract.** `persisted_output.py` publishes
same-object facades via `install(host_globals)`, rebinding each function's
`__globals__` to the live god-module namespace so existing monkeypatches
remain reachable at call time. `observability_metadata.py` and
`alias_guidance.py` facades are direct same-object assignments on the god
module (no rebinding needed; they resolve dependencies through explicit
runtime configuration seams).

**Callback seams and configuration order.** The god module configures Wave 6D
in contractual order:

1. `observability_metadata.configure_observability_metadata_runtime(...)` --
   binds tenant, headers, and env callbacks.
2. `persisted_output.bind_runtime(globals())` +
   `persisted_output.configure_persisted_output_logging_callback(...)` +
   `persisted_output.install(globals())` -- binds runtime deps, sets the
   observability-owned logging callback, then installs rebound facades.
3. `alias_guidance.configure_alias_guidance_runtime(...)` -- binds canonical
   `_merge_litellm_metadata` and `_build_langfuse_span_descriptor` callbacks
   from the observability module.

This ordering is contractual: observability metadata must be configured before
persisted-output installation (which references the logging callback), and
alias guidance must be configured after observability (which supplies its
merge/span callbacks).

**Control-plane canonicalization.** `aawm_claude_control_plane.py` imports
three canonical helpers from `observability_metadata`:

- `_iter_anthropic_text_fragments`
- `_extract_claude_agent_and_tenant_from_request_body`
- `_detect_claude_post_rewrite_context_files`

These are same-object references to the observability module's definitions.
The control plane retains its own distinct local `_get_nested_str_value`
implementation, which is intentionally separate from the
observability-metadata facade of the same name.

**Import boundary.** No Wave 6D module imports `llm_passthrough_endpoints` at
module scope. `persisted_output.py` imports from `providers/google/env_policy`,
`providers/google/persisted_output` (shaping), and `aawm_alias_routing/lane_keys`
for patterns and caps. `alias_guidance.py` imports from `lane_keys` (policy
constants) and `aawm_alias_routing/policy` (alias names).

#### Wave 6E request-policy ownership

Wave 6E extracts the remaining request-policy concerns from
`llm_passthrough_endpoints.py` into three modules under
`aawm_request_policy/`:

- `codex_tool_policy.py` -- Codex spawn-agent / core-tool description patches,
  model-capability policy lookups, custom-tool-to-function and namespace-tool
  adaptation, unsupported hosted-tool / request-param / input-item drops,
  tool-choice cleanup, and Grok-native input-item policy. 51 owned symbols.
- `claude_prompt_replacement.py` -- Claude auto-memory section replacement and
  prompt-patch manifest application, plus their logging-metadata helpers.
  14 owned symbols.
- `anthropic_body_prep.py` -- OpenAI-adapter Claude-context compaction,
  Anthropic tool-block validation / tool-use-id repair, and the final
  `_prepare_anthropic_request_body_for_passthrough` orchestration.
  11 owned symbols.

Total: 76 functions. No symbol is owned by more than one Wave 6E module, and
no Wave 6E symbol overlaps a Wave 4, 6A, or 6D inventory symbol.

**Facade model.** Two facade styles coexist on the god module:

- *Same-object facades* (34): the 14 `claude_prompt_replacement` symbols, the
  11 `anthropic_body_prep` symbols, and 9 pure `codex_tool_policy` helpers with
  no external dependency are bound by direct assignment
  (`_name = _aawm_<module>._name`). These are `is`-identical to the module
  definitions and are NOT `FunctionDef`s in the god module.
- *Thin callback wrappers* (42): the dependency-injecting `codex_tool_policy`
  functions are exposed as small god-module `FunctionDef`s that bind the shared
  `_CODEX_TOOL_POLICY_CALLBACKS` (or `_normalize_low_cardinality_tag_value`)
  and delegate. These remain `FunctionDef`s by design; they are compatibility
  wrappers, not owned logic.

`_extract_openai_passthrough_tool_choice` stays Wave 6D
`observability_metadata`-owned; Wave 6E does not shadow it.

**Callback seams and configuration order.** Wave 6E is configured after all of
Wave 6D. `codex_tool_policy` builds `_CODEX_TOOL_POLICY_CALLBACKS` from the
already-configured observability callbacks (`_normalize_low_cardinality_tag_value`,
`_dedupe_sorted_str_list`, `_merge_litellm_metadata`,
`_build_langfuse_span_descriptor`) plus live host-global model/xai/grok
resolvers. `anthropic_body_prep.configure_anthropic_body_prep_runtime(...)` is
then called with the live host callbacks for the non-owned orchestration steps
(persisted-output expansion, billing headers, control-plane rewrites, dynamic
directives, post-rewrite context metadata, web-search sanitization,
observability, tenant header). The owned preparation order inside
`_prepare_anthropic_request_body_for_passthrough` is: persisted-output
expansion; billing headers; control-plane rewrites; dynamic directives;
post-rewrite context metadata; web-search sanitization; child-agent
observability; billing/route/request-breakout/passthrough metadata; tool-use-id
repair; tool-block validation.

**Import boundary.** No Wave 6E module imports `llm_passthrough_endpoints` at
module scope. `claude_prompt_replacement.py` and `anthropic_body_prep.py` import
shared metadata primitives directly from `observability_metadata`.
`anthropic_body_prep.py` additionally imports auth/shaping helpers from
`aawm_alias_routing`. `codex_tool_policy.py` takes every external dependency
through `CodexToolPolicyCallbacks` and imports no provider or god module.

Wave 6E does not claim Wave 6F ownership: provider calls, candidate calls,
dispatch gates, route-pair delegates, the OpenCode wrapper, native Anthropic
routes, and the aawm.2/aawm.5 patches remain in the god module / Wave 6F scope.

#### Wave 6F adapter-call and dispatch ownership

Wave 6F extracts adapter execution and dispatch orchestration into four modules
under `aawm_adapter_runtime/`:

| Concern | Module | Callable surface |
|---------|--------|------------------|
| Shared Anthropic adapter request policies, execution, streaming, response finalization, and route logging support | `anthropic_adapter_calls.py` | 46 |
| Codex auto-agent candidate execution plus Kimi, Alibaba, OpenCode, and OpenRouter candidate calls | `codex_candidate_calls.py` | 14 |
| Codex Responses adapter recognition and optional dispatch | `codex_dispatch.py` | 1 |
| Anthropic-shaped adapter recognition and optional dispatch through an explicit runtime | `anthropic_dispatch.py` | 1 |

The authored surface is 62 callables. Exactly 55 former god-module
`FunctionDef`s move to the first two modules: 41 Anthropic adapter-call
definitions and 14 Codex candidate-call definitions. Five Anthropic-call names
were already compatibility assignments rather than definitions.
`_add_route_family_logging_metadata` remains canonically owned by Wave 6D
`observability_metadata.py`; Wave 6F installation restores that same object on
both the god module and `anthropic_adapter_calls.py`.

`aawm_adapter_runtime.install_wave6f()` runs only after the Wave 6D and Wave 6E
runtime callbacks are configured. It installs Anthropic adapter calls first,
then Codex candidate calls, then Codex dispatch. Candidate calls therefore
exist on the live host namespace before the Codex dispatch gate is published.
`anthropic_dispatch.py` is not installed through host-global rebinding; the god
module constructs `AnthropicDispatchRuntime` with late-binding resolver and
handler callbacks.

The Codex gate replaces only the inline recognition cascade after existing
request preparation. A returned `Response` ends dispatch; `None` preserves the
prepared body and falls through to the normal OpenAI pass-through path. The
Anthropic gate runs after body preparation and the auto-agent alias route but
before native Anthropic handling. Its resolver priority is xAI OAuth, Grok
native OAuth, OpenAI Responses, Antigravity, OpenCode, Kimi, Alibaba, Google,
NVIDIA, OpenRouter completion, then OpenRouter Responses. `None` falls through
to native Anthropic normalization, context-1m handling, and passthrough.

The 12 concrete Anthropic prepare/handle route-family pairs remain visible
god-module delegates, including the asymmetric Google pair
`_prepare_anthropic_google_completion_adapter_request` /
`_handle_anthropic_google_completion_adapter_route`. The combined OpenCode
wrapper, native Anthropic routes, route decorators and registrations, candidate
loop/cooldown ownership, and the aawm.2 OAuth and aawm.5 audit patches remain in
their prior owners.


#### Wave 7 consolidated system state

Wave 7 is the final documentation consolidation. It records the settled
ownership map, residual god-module responsibilities, facade deletion policy,
install-order contract, and the native-Anthropic egress boundary as the
authoritative reference for the pass-through subsystem.

##### Extracted module ownership (Waves 6D-6F)

| Module | Package | Owned concern |
|--------|---------|---------------|
| `alias_guidance.py` | `aawm_request_policy/` | Alias-specific system instruction shaping: prevention guidance, read-agent guidance |
| `observability_metadata.py` | `aawm_request_policy/` | Shared metadata primitives, session/repository extraction, tool-definition snapshots, Claude/Gemini/Codex breakouts, Anthropic billing headers, route-family logging |
| `persisted_output.py` | `aawm_request_policy/` | Claude persisted-output expansion, Google adapter compaction delegates, content-text estimation |
| `anthropic_body_prep.py` | `aawm_request_policy/` | OpenAI-adapter Claude-context compaction, Anthropic tool-block validation / tool-use-id repair, final `_prepare_anthropic_request_body_for_passthrough` orchestration |
| `claude_prompt_replacement.py` | `aawm_request_policy/` | Claude auto-memory section replacement, prompt-patch manifest application, logging-metadata helpers |
| `codex_tool_policy.py` | `aawm_request_policy/` | Codex spawn-agent / core-tool description patches, model-capability policy, custom-tool-to-function and namespace-tool adaptation, unsupported-field drops, tool-choice cleanup, Grok-native input-item policy |
| `anthropic_adapter_calls.py` | `aawm_adapter_runtime/` | Shared Anthropic adapter request policies, execution, streaming, response finalization, route logging support |
| `anthropic_dispatch.py` | `aawm_adapter_runtime/` | Anthropic-shaped adapter recognition and optional dispatch through explicit `AnthropicDispatchRuntime` |
| `codex_candidate_calls.py` | `aawm_adapter_runtime/` | Codex auto-agent candidate execution plus Kimi, Alibaba, OpenCode, and OpenRouter candidate calls |
| `codex_dispatch.py` | `aawm_adapter_runtime/` | Codex Responses adapter recognition and optional dispatch |

##### Residual god-module responsibilities

`llm_passthrough_endpoints.py` retains only integration-layer concerns:

- **FastAPI route registration and decorators** -- all `@router.post`,
  `@router.get`, `@router.websocket` decorators and their route paths.
- **Route orchestration** -- handler bodies that sequence preparation,
  dispatch, and response delivery for each provider route.
- **`try_dispatch` ingress and fall-through** -- the top-level dispatch gate
  that invokes the Codex and Anthropic dispatch modules and falls through to
  native pass-through when dispatch returns `None`.
- **Explicit install order and runtime assembly** -- constructing `Runtime`
  dataclasses, calling `configure_*` functions, and invoking `install()`
  entrypoints in the contractual order (see install-order summary below).
- **Proven late-host callbacks** -- callbacks that must resolve against live
  god-module globals at call time (monkeypatch compatibility).
- **12 prepare/handle route delegates** -- the concrete Anthropic
  prepare/handle route-family pairs, including the asymmetric Google pair
  `_prepare_anthropic_google_completion_adapter_request` /
  `_handle_anthropic_google_completion_adapter_route`.
- **Combined OpenCode wrapper** -- the unified OpenCode Zen route handler that
  composes normalization, dispatch, and streaming.
- **Compatibility re-exports** -- same-object facades and thin callback
  wrappers exist solely for current consumers (monkeypatch surfaces, test
  fixtures, sibling module imports). No new logic is added to compatibility
  surfaces; they are deletion candidates once consumers migrate.

##### Facade deletion policy

Pure owner bindings (same-object assignments where the god-module name is
never monkeypatched and no external consumer imports the name from the god
module) are removed after an exact `rg` grep confirms zero call-site
references outside the god module itself.

Route-layer seams and host-global seams are retained indefinitely:

- Names that appear in `@router` handler bodies.
- Names resolved through `host_globals[name]` lookups by installed modules.
- Names referenced by `configure_*` / `install()` runtime assembly.
- Names imported by sibling modules (`aawm_claude_control_plane.py`,
  `aawm_alias_routing/`, provider packages) for compatibility.

Deletion is per-name, not per-module. A module may have most facades deleted
while retaining a handful of route or host-global seams.

##### Module-scope import boundary

No extracted module (Waves 4 through 6F) imports
`llm_passthrough_endpoints` at module scope. All host dependencies flow
through:

1. Explicit `configure_*_runtime(...)` calls with frozen dataclass contracts.
2. `install(host_globals)` rebinding for monkeypatch-compatible facades.
3. Per-call `Runtime` arguments (Antigravity pattern).

This boundary is enforced by lint and is a prerequisite for future god-module
decomposition or replacement.

##### Install-order summary

The god module executes configuration and installation in this contractual
order during module initialization:

1. Wave 6D: `observability_metadata.configure_observability_metadata_runtime`
2. Wave 6D: `persisted_output.bind_runtime` +
   `configure_persisted_output_logging_callback` + `persisted_output.install`
3. Wave 6D: `alias_guidance.configure_alias_guidance_runtime`
4. Wave 6E: `codex_tool_policy` callback assembly from configured
   observability callbacks + host-global resolvers
5. Wave 6E: `anthropic_body_prep.configure_anthropic_body_prep_runtime`
6. Wave 6C: `configure_google_retry_runtime` then `codex_code_assist.install`
7. Wave 6B: `configure_openrouter_runtime`, `configure_nvidia_runtime`,
   `configure_runtime` (OpenCode Zen), `configure_xai_request_prep_runtime`
8. Wave 6F: `aawm_adapter_runtime.install_wave6f()` (Anthropic adapter calls,
   then Codex candidate calls, then Codex dispatch)
9. Wave 6F: `AnthropicDispatchRuntime` construction with late-binding
   resolver and handler callbacks

Each step may depend on all prior steps. Reordering breaks runtime wiring.

##### Native-Anthropic egress fail-closed boundary

When the Anthropic dispatch gate (`anthropic_dispatch.py`) returns `None`
(no adapter candidate matched), the request falls through to native Anthropic
normalization, context-1m handling, and passthrough. This native path sends
traffic exclusively through the Anthropic-native provider route using
credentials accepted for that route.

If the native Anthropic route or credential is unavailable, the system fails
closed with an explicit routing/authentication error and audit evidence. It
does not reroute Anthropic-model traffic through Codex, ChatGPT OAuth, or any
cross-provider egress path. This boundary applies to normal routing, alias
candidates, cross-provider fallbacks, retries, cooldown recovery, probes,
smoke tests, and acceptance harnesses.
### Runtime invariants

- Durable Redis keys use
  `aawm:alias-routing:{namespace}:{family}:{kind}:{sha256_hex(state_key)}`.
  Writes preserve the greatest existing `expires_at_epoch`, floor Redis TTL at
  one second, write Redis with propagated errors first, and then perform
  best-effort DualCache write-through. Affinity writes also pass a bounded
  process-local cardinality gate before creating another durable key.
- DualCache selection prefers the dedicated alias-routing manager. When that
  Redis target is configured but unavailable, routing falls back to local state
  rather than writing alias-routing keys into the shared usage cache.
- Google tool-call reconstruction always threads a stable request/session scope
  through remember and lookup chains.
- Candidate provider probes are single-flight per alias family and cooldown key;
  waiters re-check cooldown state before opening another provider call.
- Candidate stream validation buffers only within explicit chunk/byte limits.
  Overflow returns the buffered prefix, overflow chunk, and lazy tail in order;
  it never drains or truncates the remaining upstream stream.
- System-reminder parsing uses linear delimiter spans or a cheap closer guard,
  including unmatched-opener paths.
- Terminal audit context (host, repository, client, dispatch, trace, session,
  prior tool activity) is memoized on `request.state`.
