# Test Harness

This repository has two distinct local validation paths:

- the baseline local acceptance harness for the normal LiteLLM runtime
- the Anthropic-route adapter harness for real Claude CLI validation on `:4001`

The adapter suite is the one that matters for the Anthropic -> OpenAI/Codex,
Anthropic -> Google Code Assist, and Anthropic -> OpenRouter work.

For the production promotion process, see `PROD_RELEASE.md`.

## Runtimes

- `:4000`
  - production-style LiteLLM runtime
  - used for prod-target adapter promotion checks and the main local acceptance suite
- `:4001`
  - `litellm-dev`
  - primary development runtime for Anthropic-route adapter work
  - native Anthropic egress is enabled again, but the adapter suite still targets explicit adapted models and mixed fanout rather than plain top-level Claude traffic

Because the adapter suite is validating the Anthropic-route adapter specifically,
top-level Claude runs without an adapted `--model` are not meaningful
acceptance targets there.

## Baseline local acceptance suite

Files:

- `scripts/local-ci/run_acceptance.sh`
- `scripts/local-ci/run_acceptance.py`
- `scripts/local-ci/config.json`

Typical usage:

```bash
bash scripts/local-ci/run_acceptance.sh /tmp/local-acceptance.json
```

This suite validates the normal LiteLLM routing path, Langfuse trace shaping,
and the current production-style expectations for Codex, Gemini, and Claude.

## Anthropic adapter suite

Files:

- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `scripts/local-ci/run_anthropic_adapter_acceptance.sh`
- `scripts/local-ci/run_anthropic_adapter_smoke.py`
- `scripts/local-ci/anthropic_adapter_config.json`

Typical usage:

```bash
set -a
source .env >/dev/null 2>&1
set +a
python scripts/local-ci/run_anthropic_adapter_acceptance.py \
  --config scripts/local-ci/anthropic_adapter_config.json \
  --target dev \
  --write-artifact /tmp/anthropic-adapter-acceptance.json
```

Use `--target prod` for the production-style `:4000` instance. The target
profile rewrites the Claude CLI `ANTHROPIC_BASE_URL`, runtime health URL,
runtime Docker log container, and expected Langfuse trace environment:

- `dev`: `http://127.0.0.1:4001/anthropic`, container `litellm-dev`, trace environment `dev`
- `prod`: `http://127.0.0.1:4000/anthropic`, container `aawm-litellm`, trace environment `prod`

The harness should fail if the selected target emits traces under the wrong
Langfuse environment.

Claude trace-user checks are intentionally controlled by the harness rather than
by ambient operator settings. Claude cases inject `ANTHROPIC_CUSTOM_HEADERS` with
a generated `x-litellm-end-user-id` / `langfuse_trace_user_id` value and validate
that exact value in Langfuse. The command runner also writes a temporary
per-run Claude `--settings` overlay for `ANTHROPIC_BASE_URL` and
`ANTHROPIC_CUSTOM_HEADERS`; this is required because local Claude user/project
settings can otherwise override process-level header env vars. The overlay must
not contain secrets. Set `AAWM_CLAUDE_HARNESS_USER_ID` only when a run needs a
stable known identity. New harness work should prefer the generic
`AAWM_HARNESS_USER_ID`; when the harness is launched under pytest-classifier
observability, it derives `pytest-classifier` as the caller user id.

This suite shells out to the real Claude CLI and then validates:

- adapted route tags / metadata in Langfuse
- Anthropic-compatible response shape seen by Claude Code
- backend `session_history` attribution and cost
- provider-family egress separation
- adapted access-log labeling

Basic OpenAI smoke cases (`gpt-5.4`, `gpt-5.5`, and `gpt-5.4-mini`) intentionally
do not hard-gate the exact natural-language result string. They hard-gate command
success, usage/cost, routing, request payload logging, Langfuse
trace/user/session context, runtime logs, and `session_history` invariants.

Native Codex and Gemini CLI passthrough cases inject the current git repository
identity through `x-aawm-repository` and hard-gate that
`public.session_history.repository` is populated for the emitted provider
session. Codex CLI cases also inject harness-owned `x-litellm-end-user-id`,
`langfuse_trace_user_id`, `langfuse_trace_name`, and `session_id` config
overrides so stale ambient Codex profile headers cannot own attribution.

The native Codex Responses case also hard-gates the AAWM `spawn_agent`
tool-description policy rewrite. It validates the real Codex request logged in
Langfuse, requiring the `codex-tool-description-patch` /
`codex-tool-description-patch:spawn-agent-fanout-policy` trace tags, the generic
fanout policy text in the request, and absence of the restrictive
`Only use spawn_agent if and only if...` wording. This check is intentionally
request-shape based rather than requiring Codex to actually spawn subagents on
every smoke run.

OpenRouter Responses hard gates must catch both incomplete-but-useful streams
and successful-empty streams. If OpenRouter omits `response.completed` but emits
usable text, the adapter may synthesize a valid Anthropic `message_delta` /
`message_stop` and persist estimated usage/cost. If OpenRouter returns a
successful stream/body with no usable assistant text, tool call, or thinking
content, the OpenRouter adapter rejects it and logs bounded raw event/body
diagnostics. Harness cases with `fail_empty_success` hard-fail empty successful
Claude command output even when the case is otherwise warning-only.

Ling is retired from active OpenRouter harness targets. The aawm.37 prod
cutover exposed successful-empty Ling/free behavior, and the focused dev rerun
on 2026-04-29 shows OpenRouter returning `404` because
`inclusionai/ling-2.6-flash:free` is no longer available as a free model. Use a
currently available replacement OpenRouter model for future parallel read-tool
proofs.

OpenRouter GPT-OSS edge cases remain explicit opt-in checks, not default hard
gates. When selected, `claude_adapter_gpt_oss_120b` has one narrow allowed
soft-fail: the overlapping runtime logs must contain the OpenRouter adapter
attempt plus `503`, `provider=OpenInference`, and `raw=no healthy upstream`.
Other timeouts still fail hard so local adapter or logging regressions are not
hidden.

## Current model policy

### Hard gates

These are expected to pass on the real Claude CLI path:

- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.3-codex-spark`
- `claude_adapter_gemini_fanout`

The full adapter suite intentionally runs `claude_adapter_gemini_fanout`
before `claude_adapter_peeromega_fanout` so the dedicated Gemini gate is not
contaminated by the mixed fanout's short-window upstream pressure.

### Warning-only canaries

These are real validations, but they are allowed to warn instead of failing the
suite when the upstream provider is noisy:

- `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`
  - route to Google Code Assist directly
  - single-model `429` / `RESOURCE_EXHAUSTED` responses can still be upstream
    noise, which is why the individual model cases remain warning-only
- `openrouter/free`
- `openai/gpt-oss-20b:free`
  - excluded from the default full suite; run explicitly with
    `--cases claude_adapter_gpt_oss_20b` when this edge OpenRouter target needs
    validation
- `openai/gpt-oss-120b:free`
  - excluded from the default full suite; run explicitly with
    `--cases claude_adapter_gpt_oss_120b` when this edge OpenRouter target needs
    validation
- `google/gemma-4-31b-it:free`
- `google/gemma-4-26b-a4b-it:free`

The focused OpenRouter replacement parallel proof is not a warning-only
canary:

- `claude_adapter_openrouter_nemotron_child_parallel_read_tools`
  - uses `nvidia/nemotron-3-super-120b-a12b:free`
  - hard-gates the OpenRouter Responses route, session-history persistence,
    tool-activity persistence, and one-message parallel `Read` / `Glob` /
    `Grep` transcript shape

### Manual-only spot checks

Keep these out of the standard adapter harness run for now:

- `poolside/laguna-m.1:free`
  - OpenRouter lists it as free and tool-capable, but Claude Code rejected it
    as unavailable/inaccessible as a child-agent model during the replacement
    parallel-proof attempt
- `openai/gpt-oss-20b:free`
- `openai/gpt-oss-120b:free`
- `google/gemma-4-31b-it:free`
- `google/gemma-4-26b-a4b-it:free`
- `meta-llama/llama-3.3-70b-instruct:free`
- `minimax/minimax-m2.5:free`

### Preferred Anthropic-adapter model spellings

- direct OpenAI targets: `openai/gpt-5.4`, `openai/gpt-5.4-mini`, `openai/gpt-5.3-codex-spark`
- direct Google Code Assist targets: `google/gemini-3.1-pro-preview`, `google/gemini-3-flash-preview`, `google/gemini-3.1-flash-lite-preview`
- direct OpenRouter targets: `openrouter/openai/gpt-oss-120b:free`, `openrouter/google/gemma-4-31b-it:free`
- legacy unprefixed or vendor-only spellings still resolve for compatibility, but explicit provider prefixes are preferred because adapter routing is provider-first

## How to interpret results

### Source of truth for adapted cost

For adapted-model runs, trust:

- LiteLLM backend accounting
- `public.session_history`
- Langfuse trace metadata

Do not treat Claude CLI display cost as authoritative.

For OpenRouter `:free` aliases, the model config currently mirrors the paid
OpenRouter twin whenever OpenRouter publishes one. Current true zero-cost
exceptions remain:

- `openrouter/free`

No retired Ling target should be added back to the active canary list without a
new available model decision.

### Source of truth for OpenRouter observability

For OpenRouter-adapted traffic, treat these as reliable:

- trace tags
- adapter metadata
- backend `session_history` rows

Do not hard-gate on Langfuse generation usage fields yet; they are still too
noisy on some OpenRouter-adapted runs.

Important lane note:
- `inclusionai/ling-2.6-flash:free` / `ling-2-6-flash` is historical only and
  no longer has an active harness case or adapter alias.
- `openrouter/elephant-alpha` remains the special Anthropic -> OpenRouter
  `chat/completions` detour for the legacy agent/model mapping

Current dev pacing on `:4001` for OpenRouter free-model retries:
- short hidden retry budget: `12s`
- longer per-model post-failure cooldown: `300s`

This means a brief transient `429` can still be hidden locally, but repeated
manual retests against the same persistently throttled free model should fail
fast after the circuit opens instead of re-spending the full old retry window.

Warning-only free-model canaries are still allowed to soft-fail on upstream
timeouts or provider throttling. Those outcomes should remain warnings /
`soft_failures` in the harness artifact rather than hard suite failures.

`google/gemma-4-31b-it:free` and `google/gemma-4-26b-a4b-it:free` stay in the
config as opt-in canaries, but they are excluded from the default full suite
and should only run when explicitly requested.

Operational expectation:
- adapter-managed upstream `429` / `500` / `502` / `503` / `504` responses may
  still appear as adapter warning/backoff lines in `litellm-dev` logs
- they should not emit the generic pass-through exception traceback for the
  current request path

### Source of truth for Gemini routing

The `gemini-3.1*` adapter lane does **not** go through OpenRouter. It routes
directly to Google Code Assist using the local Google OAuth credentials mounted
into `litellm-dev`.

Current request-shape conclusion:
- the Gemini CLI bundle and the Anthropic adapter both send the same Code
  Assist envelope: `model`, `project`, `user_prompt_id`, and `request` with
  `session_id` / `contents` / tools / generation config
- if standalone Gemini CLI use is healthy but `claude_adapter_gemini_fanout`
  fails, treat that first as a local pacing/serialization regression rather
  than as authoritative provider-capacity evidence

Do not treat Google `429` / `RESOURCE_EXHAUSTED` / `MODEL_CAPACITY_EXHAUSTED`
responses as authoritative provider truth by themselves. Only close them as
provider issues after interactive Gemini CLI `/model` corroboration on the same
account context.

Telemetry expectation:
- when explicit Gemini reasoning token counts are missing but thought
  signatures are present, Langfuse generation metadata and `session_history`
  should still carry a non-null reasoning signal using the
  `provider_signature_present` source
- for Anthropic rows, only use `reasoning_tokens_source=provider_reported` when
  the provider reported a positive count; zero-value placeholders must fall
  through to estimation or remain unset
- `reasoning_tokens_source` should never be left null in `public.session_history`
  after backfill / repair passes; use `not_applicable` or `not_available` when
  no positive provider or estimated reasoning count exists
- for Anthropic, OpenAI, OpenRouter, and Gemini rows, `public.session_history`
  should also carry normalized provider-cache telemetry:
  - `provider_cache_status` should be one of `hit`, `write`, `miss`,
    `unsupported`, or `not_attempted`
  - `provider_cache_miss_reason` should be populated when a cache attempt or
    write is recognized as a miss-shaped outcome
  - `provider_cache_miss_token_count` / `provider_cache_miss_cost_usd` are
    best-effort fields and should only be populated when the missed cache token
    count is explicit enough to price defensibly
  - current detection uses provider-native cache hints:
    `cache_control` for Anthropic/OpenRouter, `cachedContent` for Gemini, and
    `input_tokens_details.cached_tokens` for OpenAI-style usage
- new `public.session_history` rows should identify the LiteLLM runtime and
  initiating client:
  `litellm_environment`, `litellm_version`, `litellm_fork_version`,
  `litellm_wheel_versions`, `client_name`, `client_version`, and
  `client_user_agent`; dev/prod harness target profiles inject the expected
  environment into session-history validation
- the default Anthropic adapter suite includes a provider-cache canary in
  `claude_adapter_peeromega_fanout`: at least one Anthropic child row must have
  `provider_cache_attempted=true` and `provider_cache_status` equal to `hit` or
  `write`
- session-history writer changes must stay synced between the in-repo dev
  callback (`litellm.integrations.aawm_agent_identity`) and the callback overlay
  wheel source (`.wheel-build/aawm_litellm_callbacks/agent_identity.py`), because
  the port-4000 `aawm-litellm` image imports the wheel module rather than the
  in-repo dev module
- `scripts/repair_session_history_provider_cache.py` is the local repair path
  for existing rows when source spend logs are not available; despite its
  historical name it now repairs inferred provider, reasoning source/count
  normalization, provider-cache telemetry, cache miss token/cost fields, and git
  commit/push rollups
- for Codex/OpenAI streaming tool runs, `response.output_item.*` and
  `response.function_call_arguments.*` events must roll up into
  `usage_tool_call_count`, `codex_tool_call_count`, and at least one
  `public.session_history_tool_activity` row when a real tool was invoked
- the dedicated `claude_adapter_codex_tool_activity` case on `:4001` is the
  hard gate for that behavior; it must persist a `Bash` / `pwd` tool row into
  `public.session_history_tool_activity`
- `claude_adapter_ctx_marker` is the hard gate for routed context markers and
  must keep validating the literal `:#port-allocation.ctx#:` rewrite path
- `claude_adapter_ctx_marker_escaped` is the hard gate for literal marker
  escaping; `\\:#name.ctx#\\:` should be rewritten to visible
  `:#name.ctx#:` without stored-procedure lookup or appendix injection
- dispatched child-agent prompts may also auto-resolve single-backticked topics
  and bare uppercase acronyms through the same `tristore_search_exact` path;
  preserve the inline text exactly and keep no-result lookups silent
- the general Claude CommonMark-formatting sentence should also carry the
  tenant/agent-scoped technical-identifier list when that prompt fragment is
  present; the current implementation uses a direct query until the stored
  procedure lands
- NVIDIA Anthropic-adapter targets are currently opt-in spot checks, not
  default-suite coverage. Preferred spellings are:
  - `nvidia/deepseek-ai/deepseek-v3.2`
  - `nvidia/deepseek-ai/deepseek-v3.1-terminus`
  - `nvidia/mistralai/devstral-2-123b-instruct-2512`
  - `nvidia/z-ai/glm4.7`
  - `nvidia/minimaxai/minimax-m2.7`
  - compatibility alias: `nvidia/minimax/minimax-m2.7`
- current NVIDIA harness cases are `claude_adapter_nvidia_deepseek_v32`,
  `claude_adapter_nvidia_glm47`, and `claude_adapter_nvidia_minimax_m27`; all
  are excluded from the default suite and should be run explicitly with
  `--cases`
- `claude_adapter_nvidia_minimax_m27` is now an explicit opt-in harness case;
  keep the exact `nvidia/minimaxai/minimax-m2.7` spelling and expect this case
  to use upstream non-stream plus Anthropic-compatible fake streaming because
  MiniMax is materially slower than the other NVIDIA targets
- these cases validate the Anthropic -> NVIDIA completion adapter on
  `nvidia:/v1/chat/completions` via `provider=nvidia_nim`
- for NVIDIA-adapted runs, expect the same parity as the other adapted
  providers:
  - `public.session_history` should persist `provider=nvidia_nim` rows with the
    normalized upstream model and non-zero cost when the model has mapped
    pricing
  - `public.session_history_tool_activity` should populate when tool or
    delegated-agent activity occurs
  - Langfuse should carry `route:anthropic_nvidia_completion_adapter`,
    `anthropic-nvidia-completion-adapter`,
    `anthropic-adapter-target:nvidia:/v1/chat/completions`, and the
    `anthropic.nvidia_completion_adapter` span name
  - cost tracking should not remain permanently zero or unmapped; if NVIDIA
    lacks usable non-free pricing, use the closest equivalent OpenRouter
    pricing as the fallback basis
- for Gemini fanout acceptance, do not assume every Gemini child model emits
  its own command row; the stable invariant is:
  - session-wide delegated `Agent` rows are present on the parent session
  - `claude_adapter_gemini_fanout` should persist at least three parent-session
    `Agent` rows and `claude_adapter_peeromega_fanout` should persist at least
    eight
  - `public.session_history` still contains provider/model/cost rows for each
    expected Gemini child model

## Latency and lifecycle debugging

When the goal is to understand where local time is being spent on our side of
the round trip, use these surfaces together:

- access logs
  - confirm the ingress route and the actual adapted target backend
- Langfuse trace metadata / spans
  - control-plane spans such as prompt rewrites and AAWM dynamic injection
  - adapted-route metadata and target-provider attribution
- `public.session_history`
  - authoritative provider/model/cost row for the completed call
- existing request metadata
  - `queue_time_seconds` for proxy-side queueing
  - `completion_start_time` / TTFT where the provider path exposes it
- DEBUG-only performance signals
  - `AawmAgentIdentity: flushed N session_history records in Xms`
  - `aawm_dynamic_injection_cache_hits` / `..._misses` in Claude trace metadata

Important: payload capture is no longer a normal always-on debugging surface. It
only runs when both `AAWM_CAPTURE=1` and `LITELLM_LOG=DEBUG` are set.

## Naming notes

Anthropic fanout prompts should keep using the Claude agent names from
`~/.claude/agents`, for example `gemini-3-flash-preview` or `gpt-5-4`.
Those agent files now point at explicit provider-prefixed model values such as
`google/gemini-3-flash-preview` and `openai/gpt-5.4`, so the prompt-side agent
name and the underlying routed model string are intentionally different.

## Promotion rule for new adapted models

Before promoting a new model from canary/manual-only to a hard gate, require:

- real Claude CLI smoke on `:4001`
- adapted trace tags / metadata present
- backend `session_history` provider / model / cost row correct
- no provider-family egress violations

For OpenRouter candidates, do not require Langfuse generation usage fields to be
clean yet. For Google Code Assist candidates, quota-window `429`s are not on
their own evidence of a bad adapter route.

## Release artifact

The harness is also published separately as a compressed artifact under `h-v*`
releases. See `WHEEL.md` for the artifact layout and `scripts/local-ci/README.md`
for the bundle-local usage notes.

Current minimum harness bundle version is `h-v0.0.21` for the aawm.37 /
`cb-v0.0.12` prod validation line. It includes the controlled Claude settings
overlay, tenant-only trace-user validation, the longer peeromega fanout timeout
for prod `:4000` validation, the narrow OpenRouter provider-unavailable timeout
/ command-failure classifier, and the default-suite GPT-OSS exclusions used by
the prod promotion suite. Older `h-v0.0.13` / `h-v0.0.14` notes are historical
only.
