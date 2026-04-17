# Overlay Wheels

This fork now ships AAWM-specific non-core behavior as independent wheel lines
so infrastructure can pin a specific LiteLLM fork release and still pull newer
AAWM overlays during image rebuilds.

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
- Langfuse trace naming and request-tag normalization
- Claude/Gemini reasoning and signature enrichment
- `public.session_history` persistence into the AAWM tristore
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
- post-rewrite `MEMORY.md` / `CLAUDE.md` trace tagging
- related Langfuse metadata/span emission for the above control-plane actions

## Runtime model

The base LiteLLM fork release provides the stable hook points. Overlay wheels
are then installed on top of that pinned release:

1. install the pinned LiteLLM fork release or image
2. install the latest callback wheel
3. install the latest control-plane wheel

This keeps AAWM-specific overlays moving independently without forcing a full
LiteLLM release cut for every prompt/tagging enhancement.

## Infrastructure consumption

The intended infrastructure pattern is:

1. pin a specific LiteLLM fork release or image for the runtime base
2. during image rebuilds, fetch the newest callback wheel
3. during image rebuilds, fetch the newest control-plane wheel

In other words:

- the base image answers: "which LiteLLM fork release are we running?"
- the overlay wheels answer: "which AAWM non-core behavior set are we layering on top?"

That keeps base compatibility changes and AAWM enhancement changes on separate
release cadences.

### What infra should pin

- base fork image / release tag:
  - example: `ghcr.io/zepfu/litellm:1.82.3-aawm.17`

### What infra should float

- callback overlay wheel line:
  - moving pointer tag: `cb-latest`
- control-plane overlay wheel line:
  - moving pointer tag: `cp-latest`

If a deployment needs fully frozen behavior for incident response or rollback
testing, pin the wheel tags too. But the normal operating model is:

- pinned base LiteLLM fork release
- latest callback overlay wheel
- latest control-plane overlay wheel

### Operational rule

Do not treat callback/control-plane wheel updates as reasons to cut a new base
LiteLLM fork release unless the change also requires a new core compatibility
patch in the fork itself.

## Rebase rule

When rebasing this fork:

- keep both wheel build directories and both wheel workflows
- keep the overlay module boundaries stable unless there is a clear reason to move them
- treat overlay wheel parity as part of the production acceptance bar
- bump the relevant wheel version whenever wheel-visible behavior changes
