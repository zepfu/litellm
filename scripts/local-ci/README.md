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
- OpenAI/Codex hard gates: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex-spark`
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
- OpenRouter manual-only spot checks for now: `meta-llama/llama-3.3-70b-instruct:free`, `minimax/minimax-m2.5:free`
- `inclusionai/ling-2.6-flash:free` stays on the generic Anthropic -> OpenRouter `Responses` lane with the other `vendor/model:free` targets
- current dev OpenRouter pacing on `:4001`:
  - `AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS=12`
  - `AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS=300`
  This is meant to preserve short hidden retries for sporadic free-model recovery while preventing repeated manual retests from re-burning the same ~40s retry window on persistently throttled models.

Preferred Anthropic-adapter model spellings:
- direct OpenAI targets: `openai/gpt-5.4`, `openai/gpt-5.4-mini`, `openai/gpt-5.3-codex-spark`
- direct Google Code Assist targets: `google/gemini-3.1-pro-preview`, `google/gemini-3-flash-preview`, `google/gemini-3.1-flash-lite-preview`
- direct OpenRouter targets: `openrouter/openai/gpt-oss-120b:free`, `openrouter/inclusionai/ling-2.6-flash:free`, `openrouter/google/gemma-4-31b-it:free`
- legacy unprefixed or vendor-only spellings still resolve for compatibility, but explicit provider prefixes are preferred because routing is now provider-first

Run it with repo env loaded so Langfuse and DB credentials are available:

```bash
set -a
source .env >/dev/null 2>&1
python3 run_anthropic_adapter_acceptance.py   --write-artifact /tmp/anthropic-adapter-acceptance.json
```

Important notes:
- top-level Claude runs without an adapted model are not the acceptance target for this suite.
- for adapted free models, LiteLLM / `session_history` are the source of truth for cost, not Claude CLI display cost. For OpenRouter free models, mirror the paid counterpart cost when a non-free twin exists; keep `openrouter/free` and `inclusionai/ling-2.6-flash:free` at zero because OpenRouter does not publish a paid twin for them.
- The Google Code Assist lane is warning-only in the harness; the route works, but `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, and `gemini-3.1-flash-lite-preview` can all hit real `429` / `RESOURCE_EXHAUSTED` responses from `cloudcode-pa.googleapis.com`.
- The current Gemini CLI bundle and the Anthropic adapter use the same Code Assist request envelope: `model`, `project`, `user_prompt_id`, and `request` with `session_id` / `contents` / tools / generation config. When standalone Gemini CLI use is healthy, treat `claude_adapter_gemini_fanout` failures first as local pacing/serialization bugs, not as authoritative provider-capacity proof.
- `openrouter/free` and `inclusionai/ling-2.6-flash:free` are canaries, not hard gates; upstream routing / rate limits can make them noisy even when the local adapter path is correct.
- `ling-2-6-flash` now validates on the generic OpenRouter `Responses` lane, so trace tags / metadata / `session_history` should match the other free-model response adapters.
