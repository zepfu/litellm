# Test Harness

This repository has two distinct local validation paths:

- the baseline local acceptance harness for the normal LiteLLM runtime
- the Anthropic-route adapter harness for real Claude CLI validation on `:4001`

The adapter suite is the one that matters for the Anthropic -> OpenAI/Codex,
Anthropic -> Google Code Assist, and Anthropic -> OpenRouter work.

## Runtimes

- `:4000`
  - production-style LiteLLM runtime
  - used for the main local acceptance suite
- `:4001`
  - `litellm-dev`
  - the only supported runtime for Anthropic-route adapter work
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
  --write-artifact /tmp/anthropic-adapter-acceptance.json
```

This suite shells out to the real Claude CLI and then validates:

- adapted route tags / metadata in Langfuse
- Anthropic-compatible response shape seen by Claude Code
- backend `session_history` attribution and cost
- provider-family egress separation
- adapted access-log labeling

## Current model policy

### Hard gates

These are expected to pass on the real Claude CLI path:

- `gpt-5.4`
- `gpt-5.3-codex-spark`
- `openai/gpt-oss-120b:free`

### Warning-only canaries

These are real validations, but they are allowed to warn instead of failing the
suite when the upstream provider is noisy:

- `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`
  - route to Google Code Assist directly
  - repeated `429` / `RESOURCE_EXHAUSTED` responses are quota pressure, not a
    local routing failure
- `openrouter/free`
- `openrouter/elephant-alpha`
- `openai/gpt-oss-20b:free`
- `google/gemma-4-31b-it:free`
- `google/gemma-4-26b-a4b-it:free`
- `nvidia/nemotron-3-super-120b-a12b:free`

### Manual-only spot checks

Keep these out of the standard adapter harness run for now:

- `meta-llama/llama-3.3-70b-instruct:free`
- `minimax/minimax-m2.5:free`

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
- `openrouter/elephant-alpha`

### Source of truth for OpenRouter observability

For OpenRouter-adapted traffic, treat these as reliable:

- trace tags
- adapter metadata
- backend `session_history` rows

Do not hard-gate on Langfuse generation usage fields yet; they are still too
noisy on some OpenRouter-adapted runs.

Important lane note:
- `openrouter/elephant-alpha` is intentionally routed through the Anthropic ->
  OpenRouter `chat/completions` completion lane
- the other OpenRouter free models stay on the generic OpenRouter `Responses`
  lane unless real Claude behavior proves they need the same detour

Current dev pacing on `:4001` for OpenRouter free-model retries:
- short hidden retry budget: `12s`
- longer per-model post-failure cooldown: `300s`

This means a brief transient `429` can still be hidden locally, but repeated
manual retests against the same persistently throttled free model should fail
fast after the circuit opens instead of re-spending the full old retry window.

### Source of truth for Gemini routing

The `gemini-3.1*` adapter lane does **not** go through OpenRouter. It routes
directly to Google Code Assist using the local Google OAuth credentials mounted
into `litellm-dev`.

Do not treat Google `429` / `RESOURCE_EXHAUSTED` / `MODEL_CAPACITY_EXHAUSTED`
responses as authoritative provider truth by themselves. Only close them as
provider issues after interactive Gemini CLI `/model` corroboration on the same
account context.

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
