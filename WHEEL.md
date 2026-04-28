# Overlay Wheels

This fork now ships AAWM-specific non-core behavior as independent wheel lines
so infrastructure can pin a specific LiteLLM fork release and still pull newer
AAWM overlays during image rebuilds.

The repo also ships the local acceptance harness as a separate compressed
archive so validation tooling can move independently from the main codebase.
The standalone model pricing/capability config is also published as its own
archive so infrastructure can consume that data without pulling the full repo.
See `TEST_HARNESS.md` for the actual validation process and interpretation.

## Distinction

- **LiteLLM fork release (`v*-aawm.*`)**
  - carries the base forked LiteLLM release
  - should include only the compatibility / correctness patches required for our
    supported provider traffic and runtime shape
- **AAWM overlay wheels**
  - carry AAWM-specific behavior that is not part of LiteLLM core
  - can move independently of the pinned base LiteLLM release

This distinction is important: base compatibility fixes must stay tied to the
validated LiteLLM release, while prompt/tagging/enrichment overlays can ship on
their own cadence.

## Wheel Lines

### Callback wheel

Published source:

- `.wheel-build/pyproject.toml`
- `.wheel-build/aawm_litellm_callbacks/__init__.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`

Release workflow:

- `.github/workflows/aawm-callback.yml`

Tag line:

- `cb-v*`
- moving pointer: `cb-latest`

Current responsibilities:

- Claude Code / agent identity extraction
- Langfuse trace naming, tenant-only trace user ids, and request-tag normalization
- stale Claude Code orchestrator trace-header rewrite for dispatched child
  traces while preserving unrelated explicit caller trace names
- Claude/Gemini reasoning and signature enrichment
- `public.session_history` persistence into the AAWM tristore
- `public.session_history_tool_activity` classification for delegated agents and
  provider-native tool names
- background batching for `session_history` writes with configurable batch/flush tuning
- Gemini/Codex usage breakout normalization for cache, reasoning, and tool-call fields

### Claude control-plane wheel

Published source:

- `.control-plane-wheel-build/pyproject.toml`
- `litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py`
- `context-replacement/claude-code/...`

Release workflow:

- `.github/workflows/aawm-control-plane.yml`

Tag line:

- `cp-v*`
- moving pointer: `cp-latest`

Current responsibilities:

- Claude Code `# auto memory` system-prompt replacement
- exact-match prompt patch manifest application
- AAWM dynamic directive expansion like `AAWM p=get_agent_memories ...`
- short-TTL caching for dynamic directive expansion results on repeated session/agent/tenant contexts
- post-rewrite `MEMORY.md` / `CLAUDE.md` trace tagging
- related Langfuse metadata/span emission for the above control-plane actions

## Acceptance Harness Archive

Published source:

- `scripts/local-ci/README.md`
- `scripts/local-ci/run_acceptance.sh`
- `scripts/local-ci/run_acceptance.py`
- `scripts/local-ci/config.json`
- `scripts/local-ci/build_harness_bundle.py`

Release workflow:

- `.github/workflows/aawm-harness.yml`

Tag line:

- `h-v*`
- moving pointer: `h-latest`

Artifact:

- `litellm-local-ci-harness-<version>.tar.gz`

Current responsibilities:

- CLI routing and Langfuse trace verification across Codex, Gemini, and Claude
- Claude request rewrite verification, including prompt-shape watchpoints like
  future `verbosity` payload adoption
- repeatable local regression validation outside the main repo checkout

## Model Config Archive

Published source:

- `model_prices_and_context_window.json`
- `litellm/bundled_model_prices_and_context_window_fallback.json`
- `scripts/build_model_config_bundle.py`

Release workflow:

- `.github/workflows/aawm-config.yml`

Tag line:

- `cfg-v*`
- moving pointer: `cfg-latest`

Artifact:

- `litellm-model-config-<version>.tar.gz`

Current responsibilities:

- standalone pricing/capability/config distribution for infrastructure and tooling
- versioned packaging of the model registry independent of the base fork image
- fast drift checks when model metadata changes without a full base release
- current adapter-cost policy support, including paid-equivalent pricing for
  selected OpenRouter `:free` aliases when OpenRouter publishes a non-free twin

Source-of-truth rule:

- `model_prices_and_context_window.json` is the canonical editable file
- `litellm/bundled_model_prices_and_context_window_fallback.json` is the packaged
  offline mirror used when `LITELLM_LOCAL_MODEL_COST_MAP=True`
- after editing the canonical file, run `make sync-model-cost-map`

## Runtime model

The base LiteLLM fork release provides the stable hook points. Overlay wheels
are then installed on top of that pinned release:

1. install the pinned LiteLLM fork release or image
2. install the latest callback wheel
3. install the latest control-plane wheel
4. optionally install the latest model-config archive payload into the runtime
   image when infrastructure wants model/pricing changes without a base-fork
   release cut

This keeps AAWM-specific overlays moving independently without forcing a full
LiteLLM release cut for every prompt/tagging enhancement.

The base LiteLLM wheel now explicitly excludes the callback/control-plane
overlay sources and the local harness sources. Those artifacts are published on
their own release lines instead of riding inside the main LiteLLM package.

## Activation points

Installing the wheels is not sufficient by itself. Infra must also wire the
runtime so the overlay code is actually exercised.

### Callback overlay activation

The callback wheel is not auto-discovered. It must be registered in LiteLLM
config.

Example proxy config:

```yaml
litellm_settings:
  callbacks:
    - "aawm_litellm_callbacks.agent_identity.AawmAgentIdentity"
  success_callback:
    - "langfuse"
```

Notes:

- use the wheel module path above, not the in-repo development import path
- keep `langfuse` enabled if you want the overlay trace tags, spans, and
  metadata to actually be emitted upstream
- `session_history` persistence requires AAWM DB connectivity through either:
  - `AAWM_DB_URL` / `AAWM_DATABASE_URL`
  - or the component vars:
    - `AAWM_DB_HOST`
    - `AAWM_DB_PORT`
    - `AAWM_DB_NAME`
    - `AAWM_DB_USER`
    - `AAWM_DB_PASSWORD`
    - optional SSL vars

If the callback wheel is installed but not registered in `callbacks:`, none of
the identity/session-history enrichment will run.

### Control-plane overlay activation

The Claude control-plane wheel is activated by installation order, not by a
separate LiteLLM callback registration.

The base fork imports:

- `litellm.proxy.pass_through_endpoints.aawm_claude_control_plane`

So infra must:

1. install the pinned base LiteLLM fork release
2. install the control-plane wheel after that

The second install must happen last so the overlay module and packaged prompt
assets are what end up on disk in site-packages.

Notes:

- there is no extra `callbacks:` or `success_callback:` entry for the
  control-plane wheel itself
- the control-plane overlay only applies on the Claude / Anthropic passthrough
  request path
- Langfuse must still be enabled if you want the control-plane tags/spans to be
  visible in traces
- AAWM dynamic memory injection requires the same `AAWM_DB_*` / `AAWM_DB_URL`
  database settings; otherwise the directive-expansion path will fall back to
  its failure behavior

## Performance-related runtime defaults

Recent AAWM runtime work also changed the default performance posture:

- payload capture is now debug-only
  - `litellm.integrations.aawm_payload_capture` only writes captures when both:
    - `AAWM_CAPTURE=1`
    - `LITELLM_LOG=DEBUG`
- callback-side `session_history` persistence is now batched in the background
  - tune with:
    - `AAWM_SESSION_HISTORY_BATCH_SIZE`
    - `AAWM_SESSION_HISTORY_FLUSH_INTERVAL_MS`
    - `AAWM_SESSION_HISTORY_POOL_MAX_SIZE`
  - the writer reuses a bounded asyncpg pool and bounds overflow flushing so
    provider bursts do not create unbounded PostgreSQL client spikes
- Claude dynamic injection now has a small TTL cache
  - tune with:
    - `AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS`

Useful operator-visible instrumentation from those changes:

- DEBUG log: `AawmAgentIdentity: flushed N session_history records in Xms`
- Claude metadata / Langfuse span fields:
  - `aawm_dynamic_injection_cache_hits`
  - `aawm_dynamic_injection_cache_misses`
  - `aawm_dynamic_injection_cache_statuses`

These are runtime-behavior changes, not separate artifact lines, but they matter
for how infra should evaluate tail latency on rebuilt images.

## Artifact resolution

The current release model has two kinds of refs:

- versioned wheel releases with assets:
  - `cb-v*`
  - `cp-v*`
- moving git tags:
  - `cb-latest`
  - `cp-latest`

Important: the installable wheel assets are currently attached to the versioned
releases, not to separate `cb-latest` / `cp-latest` release objects.

That means infra must currently do one of these:

1. resolve the newest versioned callback/control-plane release and download that
   wheel asset
2. pin an exact wheel version tag

Do not assume `cb-latest` or `cp-latest` is itself a directly installable wheel
URL unless the workflows are later extended to publish release assets on those
moving tags too.

The same rule applies to:

- `h-latest`
- `cfg-latest`

## Infrastructure consumption

For the end-to-end production promotion checklist, see `PROD_RELEASE.md`.

The intended infrastructure pattern is:

1. pin a specific LiteLLM fork release or image for the runtime base
2. during image rebuilds, fetch the newest callback wheel
3. during image rebuilds, fetch the newest control-plane wheel

In other words:

- the base image answers: "which LiteLLM fork release are we running?"
- the overlay wheels answer: "which AAWM non-core behavior set are we layering on top?"

That keeps base compatibility changes and AAWM enhancement changes on separate
release cadences.

## Current adapter boundary

The current Anthropic-route multi-provider adapter work is split intentionally:

- base fork release:
  - Anthropic-route adapter routing and translation logic
  - provider-family egress guard
  - adapted access-log labeling
  - backend `session_history` / Langfuse plumbing needed for adapted traffic
- callback wheel:
  - callback-side enrichment and `session_history` persistence
- control-plane wheel:
  - Claude-specific prompt rewrites and dynamic context injection
- harness/config artifacts:
  - validation policy and model/pricing distribution

In other words: OpenAI/Codex, Google Code Assist, and OpenRouter adaptation on
`/anthropic/v1/messages` is currently base-fork behavior, not an overlay wheel.

### What infra should pin

- base fork image / release tag:
  - example: `ghcr.io/zepfu/litellm:1.82.3-aawm.27`

### What infra should float

- callback overlay wheel line:
  - resolve latest versioned `cb-v*` wheel release
- control-plane overlay wheel line:
  - resolve latest versioned `cp-v*` wheel release

If a deployment needs fully frozen behavior for incident response or rollback
testing, pin the wheel tags too. But the normal operating model is:

- pinned base LiteLLM fork release
- latest callback overlay wheel
- latest control-plane overlay wheel

### Operational rule

Do not treat callback/control-plane wheel updates as reasons to cut a new base
LiteLLM fork release unless the change also requires a new core compatibility
patch in the fork itself.

## Main Push Automation

Changes merged to `main` now auto-bump and tag the independently shipped AAWM
artifacts:

- callback wheel -> `cb-v*`
- control-plane wheel -> `cp-v*`
- harness archive -> `h-v*`
- model config archive -> `cfg-v*`

The workflow responsible for that is:

- `.github/workflows/aawm-artifact-autobump.yml`

This keeps the artifact release cadence tied to actual overlay/config changes
without forcing a human to hand-bump versions on every merge.

## Rebase rule

When rebasing this fork:

- keep both wheel build directories and both wheel workflows
- keep the overlay module boundaries stable unless there is a clear reason to move them
- treat overlay wheel parity as part of the production acceptance bar
- bump the relevant wheel version whenever wheel-visible behavior changes
