# LiteLLM Local Acceptance Harness

This bundle contains the standalone local acceptance harness used to validate
LiteLLM routing, request rewrites, Langfuse traces, and provider-specific
metadata across Codex, Gemini, and Claude. See `TEST_HARNESS.md` in the main
repo for the full validation policy and runtime split.

## Included Files

- `run_acceptance.sh`
- `run_acceptance.py`
- `compare_artifacts.py`
- `config.json`
- `claude_acceptance_prompt.txt`
- `claude_acceptance_prompt_full_fanout.txt`
- `run_anthropic_adapter_acceptance.py`
- `run_anthropic_adapter_smoke.py`
- `anthropic_adapter_config.json`

## Required Environment

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- optional: `LANGFUSE_QUERY_URL`
- optional: `LITELLM_BASE_URL`
- optional: `ACCEPTANCE_CONFIG_PATH`

Provider CLIs must also be installed and configured separately.

## Usage

```bash
bash run_acceptance.sh /tmp/local-acceptance.json
```

To compare two acceptance artifacts:

```bash
python3 compare_artifacts.py old.json new.json
```

## Release Artifact

This harness is released separately as a compressed artifact under `h-v*` tags.

Example download pattern:

```bash
curl -L -o litellm-local-ci-harness.tar.gz \
  https://github.com/zepfu/litellm/releases/download/h-v0.0.1/litellm-local-ci-harness-0.0.1.tar.gz
```

## Anthropic Adapter Suite

Use the adapter suite for real-Claude validation on `litellm-dev` (`:4001`).
It shells out to the actual Claude CLI and verifies Langfuse plus
`session_history` for adapted Anthropic-route models.

Current first-wave adapted coverage:
- OpenAI/Codex hard gates: `gpt-5.4`, `gpt-5.5`, `gpt-5.4-mini`, `gpt-5.3-codex-spark`
- Gemini fanout hard gate: `claude_adapter_gemini_fanout`
  - isolates the exact multi-Gemini dispatch path on `:4001`
  - use this before spending time on the full adapter suite when a Gemini
    fanout regression is suspected
  - the full suite runs this before `claude_adapter_peeromega_fanout` so the
    dedicated Gemini gate is not polluted by the mixed fanout's short-window
    upstream pressure
- Mixed fanout dispatch target set: `analyst`, `data`, `gpt-5-3-codex-spark`, `gpt-5-4`, `gemini-3-flash-preview`, `gemini-3-1-pro-preview`, `gemini-3-1-flash-lite-preview`, `ling-2-6-flash`
- Google Code Assist canaries: `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`
  - the adapter routes Gemini Anthropic-adapter models directly to Google Code Assist on `:4001`
  - `gemini-3.1-pro-preview` and `gemini-3-flash-preview` are the main real-Claude validation targets; `gemini-3.1-flash-lite-preview` remains quota-sensitive
  - keep them warning-only in the harness, but do not treat `429` / `RESOURCE_EXHAUSTED` as authoritative upstream truth without interactive Gemini CLI `/model` corroboration on the same account context
- OpenRouter hard gate: `openai/gpt-oss-120b:free`
  - for OpenRouter-adapted cases, rely on trace tags/metadata plus `session_history`; do not hard-gate on Langfuse generation usage fields yet
- OpenRouter preferred free targets under active validation: `inclusionai/ling-2.6-flash:free`, `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-super-120b-a12b:free`
- OpenRouter warning-only canaries: `openrouter/free`, `inclusionai/ling-2.6-flash:free`, `openai/gpt-oss-20b:free`, `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-super-120b-a12b:free`
  - `google/gemma-4-31b-it:free` and `google/gemma-4-26b-a4b-it:free` remain available in config but are excluded from the default full suite
  - run them only by explicit selection, for example:
    `--cases claude_adapter_gemma_31b,claude_adapter_gemma_26b_a4b`
- OpenRouter manual-only spot checks for now: `meta-llama/llama-3.3-70b-instruct:free`, `minimax/minimax-m2.5:free`
- `inclusionai/ling-2.6-flash:free` stays on the generic Anthropic -> OpenRouter `Responses` lane with the other `vendor/model:free` targets
- current dev OpenRouter pacing on `:4001`:
  - `AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS=12`
  - `AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS=300`
  This is meant to preserve short hidden retries for sporadic free-model recovery while preventing repeated manual retests from re-burning the same ~40s retry window on persistently throttled models.
  Adapter-managed upstream `429` / `500` / `502` / `503` / `504` responses may still show up as adapter warning/backoff lines, but they should not produce the generic pass-through exception traceback for the current request path.

Preferred Anthropic-adapter model spellings:
- direct OpenAI targets: `openai/gpt-5.4`, `openai/gpt-5.5`, `openai/gpt-5.4-mini`, `openai/gpt-5.3-codex-spark`
- direct Google Code Assist targets: `google/gemini-3.1-pro-preview`, `google/gemini-3-flash-preview`, `google/gemini-3.1-flash-lite-preview`
- direct OpenRouter targets: `openrouter/openai/gpt-oss-120b:free`, `openrouter/inclusionai/ling-2.6-flash:free`, `openrouter/google/gemma-4-31b-it:free`
- legacy unprefixed or vendor-only spellings still resolve for compatibility, but explicit provider prefixes are preferred because routing is now provider-first

Run it with repo env loaded so Langfuse and DB credentials are available:

```bash
set -a
source .env >/dev/null 2>&1
set +a
python3 run_anthropic_adapter_acceptance.py \
  --target dev \
  --write-artifact /tmp/anthropic-adapter-dev.json
```

Target profiles:
- `--target dev` validates `http://127.0.0.1:4001/anthropic`, `litellm-dev`, and Langfuse trace environment `dev`.
- `--target prod` validates `http://127.0.0.1:4000/anthropic`, `aawm-litellm`, and Langfuse trace environment `prod`.
- Use `--litellm-base-url`, `--anthropic-base-url`, `--docker-container-name`, or `--expected-trace-environment` only for explicit one-off overrides.

Important notes:
- top-level Claude runs without an adapted model are not the acceptance target for this suite.
- for adapted free models, LiteLLM / `session_history` are the source of truth for cost, not Claude CLI display cost. For OpenRouter free models, mirror the paid counterpart cost when a non-free twin exists; keep `openrouter/free` and `inclusionai/ling-2.6-flash:free` at zero because OpenRouter does not publish a paid twin for them.
- The Google Code Assist lane is warning-only in the harness; the route works, but `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, and `gemini-3.1-flash-lite-preview` can all hit real `429` / `RESOURCE_EXHAUSTED` responses from `cloudcode-pa.googleapis.com`.
- The current Gemini CLI bundle and the Anthropic adapter use the same Code Assist request envelope: `model`, `project`, `user_prompt_id`, and `request` with `session_id` / `contents` / tools / generation config. When standalone Gemini CLI use is healthy, treat `claude_adapter_gemini_fanout` failures first as local pacing/serialization bugs, not as authoritative provider-capacity proof.
- When explicit Gemini reasoning token counts are absent but thought signatures are present, treat `provider_signature_present` as the expected fallback source in Langfuse generation metadata and `session_history`.
- For Anthropic rows, `reasoning_tokens_source=provider_reported` is only valid when the provider supplied a positive count; zero placeholders should fall through to estimation or remain unset.
- `reasoning_tokens_source` should not remain null in repaired or newly written `session_history` rows; use `not_applicable` when no reasoning is present and `not_available` when reasoning is present but no positive provider/estimated count exists.
- Anthropic/OpenAI/Gemini/OpenRouter `session_history` rows should also carry normalized provider-cache telemetry. Expect `provider_cache_status` to land as `hit`, `write`, `miss`, `unsupported`, or `not_attempted`, with `provider_cache_miss_reason` populated for miss-shaped outcomes. `provider_cache_miss_token_count` / `provider_cache_miss_cost_usd` are best-effort and should only appear when the miss token count is explicit enough to price honestly.
- The default adapter suite now includes an explicit provider-cache canary through `claude_adapter_peeromega_fanout`: at least one Anthropic child row must show `provider_cache_attempted=true` and `provider_cache_status` of `hit` or `write`.
- When the original proxy spend-log source is unavailable locally, use [scripts/repair_session_history_provider_cache.py](/home/zepfu/projects/litellm/scripts/repair_session_history_provider_cache.py) to repair session-history observability directly from existing rows before relying on historical aggregates. The repair now covers inferred `provider`, invalid zero-count `provider_reported` reasoning rows, provider-cache state, cache miss token/cost fields, and git commit/push rollups from `session_history_tool_activity`.
- Keep callback overlay parity in mind: `litellm-dev` imports `litellm.integrations.aawm_agent_identity`, while the port-4000 `aawm-litellm` image imports the installed callback wheel module at `aawm_litellm_callbacks.agent_identity`. Any session-history writer change must be applied to both sources and released through a new callback wheel before rebuilding the production-style image.
- For Codex/OpenAI streaming tool runs, the local source of truth is the reconstructed `response.output_item.*` / `response.function_call_arguments.*` stream state. Expect `usage_tool_call_count`, `codex_tool_call_count`, and `session_history_tool_activity` rows to reflect real tool invocations on `:4001`.
- `claude_adapter_codex_tool_activity` is the hard gate for that path. It uses a single `Bash` / `pwd` tool call and must persist a matching `session_history_tool_activity` row.
- `claude_adapter_ctx_marker` is the hard gate for dynamic-context marker rewriting and uses `:#port-allocation.ctx#:` as the canonical stored-procedure validation fixture.
- `claude_adapter_ctx_marker_escaped` validates literal marker escaping. Prompts can use `\\:#name.ctx#\\:` to show `:#name.ctx#:` to the model without calling `tristore_search_exact` or appending context.
- dispatched child-agent prompts also resolve single-backticked topics like `` `port-allocation` `` and bare uppercase acronyms like `API` through the same tristore lookup, while preserving the inline text and staying silent on misses.
- the general Claude CommonMark-formatting sentence is also rewritten with a tenant/agent-scoped technical-identifier list sourced from `ag_catalog.raw_content`; this currently uses a direct query and is expected to move to a stored procedure.
- NVIDIA Anthropic-adapter targets are currently opt-in spot checks. Preferred spellings are `nvidia/deepseek-ai/deepseek-v3.2`, `nvidia/deepseek-ai/deepseek-v3.1-terminus`, `nvidia/mistralai/devstral-2-123b-instruct-2512`, `nvidia/z-ai/glm4.7`, and `nvidia/minimaxai/minimax-m2.7`; keep the compatibility alias `nvidia/minimax/minimax-m2.7` resolving to `minimaxai/minimax-m2.7`.
- Current NVIDIA harness cases are `claude_adapter_nvidia_deepseek_v32`, `claude_adapter_nvidia_glm47`, and `claude_adapter_nvidia_minimax_m27`. They are excluded from the default suite and should be selected explicitly with `--cases` while the lane is still under active validation.
- `claude_adapter_nvidia_minimax_m27` now uses upstream non-stream plus Anthropic-compatible fake streaming. Keep the exact `nvidia/minimaxai/minimax-m2.7` spelling, and treat it as an explicit opt-in spot check rather than a default-suite canary because MiniMax is materially slower than the other NVIDIA targets.
- These NVIDIA spot checks validate the Anthropic -> NVIDIA completion adapter on `nvidia:/v1/chat/completions` via `provider=nvidia_nim`.
- For NVIDIA-adapted runs, expect the same observability parity as the other adapted providers: `session_history.provider=nvidia_nim` with the normalized upstream model and non-zero cost when pricing is mapped, `session_history_tool_activity` rows when tool or agent-dispatch work occurs, Langfuse trace environment matching the selected `--target`, `route:anthropic_nvidia_completion_adapter` / `anthropic-nvidia-completion-adapter` / `anthropic-adapter-target:nvidia:/v1/chat/completions` tags plus the `anthropic.nvidia_completion_adapter` span, and usable cost tracking. If NVIDIA pricing is not available for a target model, fall back to the closest equivalent OpenRouter pricing rather than leaving long-term cost tracking unmapped.
- For Gemini fanout, the stable tool-activity invariant is the parent session’s delegated `Agent` rows, not child-model command rows. `claude_adapter_gemini_fanout` should persist at least three `Agent` rows, `claude_adapter_peeromega_fanout` should persist at least eight, and `session_history` still hard-gates the expected Gemini provider/model/cost rows for each child model.
- Harness `0.0.13` keeps `openai/gpt-oss-120b:free` as the OpenRouter hard gate and catches OpenRouter Responses streams that omit `response.completed`; the adapter should still persist non-zero estimated usage/cost from streamed output plus checked-in/bundled model-price JSON. If that hard gate times out or command-fails and the overlapping runtime logs show the exact OpenRouter provider-unavailable signature (`503`, `provider=OpenInference`, `raw=no healthy upstream`), the harness soft-fails it as upstream availability without masking local adapter/logging failures.
- Fanout prompts should continue using the Claude agent names from `~/.claude/agents` such as `gemini-3-flash-preview`; those agent files now carry explicit provider-prefixed `model:` values like `google/gemini-3-flash-preview`.
- `openrouter/free` and `inclusionai/ling-2.6-flash:free` are canaries, not hard gates; upstream routing / rate limits can make them noisy even when the local adapter path is correct.
- `warning_only` canaries stay non-gating even when the subprocess itself times out; those conditions should surface as `soft_failures` / warnings in the artifact, not as suite-stopping failures.
- `ling-2-6-flash` now validates on the generic OpenRouter `Responses` lane, so trace tags / metadata / `session_history` should match the other free-model response adapters.
