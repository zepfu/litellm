# LiteLLM Local Acceptance Harness

This bundle contains the standalone local acceptance harness used to validate
LiteLLM routing, request rewrites, Langfuse traces, and provider-specific
metadata across Claude, Codex, OpenRouter, NVIDIA, and Grok paths. See
`TEST_HARNESS.md` in the main repo for the full validation policy and runtime
split, and `PROD_RELEASE.md` for the production promotion runbook.

## Included Files

The standalone `h-v*` harness archive currently includes only the baseline
local acceptance harness:

- `README.md`
- `run_acceptance.sh`
- `run_acceptance.py`
- `compare_artifacts.py`
- `config.json`
- `claude_acceptance_prompt.txt`
- `claude_acceptance_prompt_full_fanout.txt`

The Anthropic adapter harness is repo-local and is not currently packaged by
`build_harness_bundle.py`:

- `run_anthropic_adapter_acceptance.py`
- `run_anthropic_adapter_smoke.py`
- `anthropic_adapter_config.json`

Adapter case prompts use paths relative to the repository root. Run the adapter
harness from the checkout it belongs to so fixture paths such as
`scripts/local-ci/sequential_core_tools_fixture.txt` resolve consistently.

## Required Environment

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- optional: `LANGFUSE_QUERY_URL`
- optional: `LITELLM_BASE_URL`
- optional: `ACCEPTANCE_CONFIG_PATH`

Provider CLIs must also be installed and configured separately.

The Python harness launches provider CLIs with a scrubbed environment. Provider
authentication variables such as `ANTHROPIC_*` or `OPENAI_*` may be passed
through, but Langfuse, database, PostgreSQL, and LiteLLM admin secrets such as
`LITELLM_MASTER_KEY` are excluded. Only explicit non-secret LiteLLM routing
keys, including `LITELLM_BASE_URL`, are inherited.

The shell wrapper does not source `.env`. It parses the file and exports only
Langfuse settings plus named harness overrides to the parent Python process;
database and unrelated provider secrets in `.env` are ignored.

CLI stdout and stderr stored in the JSON artifact are capped at 200,000
characters per stream by default. Set `ACCEPTANCE_CLI_OUTPUT_MAX_CHARS` to tune
the cap. Artifact fields record whether truncation occurred and the original
character counts.

Config command tokens may use `@{config_dir}/path`; the harness expands them
relative to the loaded config file. Family configs may also set
`minimum_trace_count`, `skip_generation_quality_checks` (or
`skip_quality_checks`), and `allow_zero_cost`.

## Usage

```bash
bash run_acceptance.sh /tmp/local-acceptance.json
```

To compare two acceptance artifacts:

```bash
python3 compare_artifacts.py old.json new.json
```

Claude comparisons require at least one persona-enriched trace name such as
`claude-code.explorer`. The bare `claude-code` default and
`claude-code.orchestrator` do not count as subagent/persona evidence.

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
- Native Anthropic telemetry focused gate: `native_anthropic_passthrough_claude`
  - validates a single non-fanout Claude CLI passthrough request
  - hard-gates `session_history` plus Anthropic
    `rate_limit_observations` rows from provider response headers
  - excluded from the default suite until the live native auth path has a
    clean dev/prod proof; run explicitly with
    `--cases native_anthropic_passthrough_claude`
- Native Codex telemetry focused gates:
  `native_openai_passthrough_responses_codex`,
  `native_openai_passthrough_responses_codex_tool_activity`
  - validate `session_history` rows for the native client/provider shape
  - hard-gate Codex `codex_response_headers` quota rows in
    `rate_limit_observations`
  - excluded from the default suite; run explicitly with `--cases`
- OpenAI/Codex hard gates: `gpt-5.4`, `gpt-5.5`, `gpt-5.4-mini`, `gpt-5.3-codex-spark`
  through `claude_adapter_gpt54`, `claude_adapter_gpt55`,
  `claude_adapter_gpt54_mini`, `claude_adapter_spark`,
  `claude_adapter_gpt55_read_pages_sanitizer`, and
  `claude_adapter_codex_tool_activity`
- Mixed fanout opt-in target set: `analyst`, `data`, `gpt-5-3-codex-spark`, `gpt-5-4`
  through `claude_adapter_peeromega_fanout`
- OpenRouter opt-in edge-lane check: `openai/gpt-oss-120b:free`
  - for OpenRouter-adapted cases, rely on trace tags/metadata plus `session_history`; do not hard-gate on Langfuse generation usage fields yet
  - this case remains available by explicit `--cases claude_adapter_gpt_oss_120b`, but is excluded from the default suite because OpenRouter frequently returns provider-unavailable `503 provider=OpenInference raw=no healthy upstream`; it is an opt-in hard gate with only that narrow provider-unavailable soft-fail, not a `warning_only` canary
- OpenRouter free daily meter gate: `native_openrouter_free_daily_meter_chat`
  - sends `openrouter/openai/gpt-oss-20b:free` through dev LiteLLM `/chat/completions` on `:4001`
  - hard-gates `session_history` plus the local OpenRouter free daily request snapshot in `public.rate_limit_observations`
  - expects `provider=openrouter`, `client=openrouter`, `source=openrouter_free_daily_local_meter`, `quota_key=openrouter_free_daily_requests:requests`, `quota_type=requests`, and `quota_period=daily`
  - allows latest-snapshot fallback because duplicate unchanged snapshots are intentionally suppressed by the observation insert
- OpenRouter preferred free target under active validation: `nvidia/nemotron-3-super-120b-a12b:free`
- OpenRouter paid child-agent proof target under active validation: `deepseek/deepseek-v4-flash`
- xAI auth split (hard-gated in D1-251):
  - `claude_adapter_xai_grok_composer_child_parallel_read_tools` validates
    `xai/grok-composer-2.5-fast` through the native OIDC adapter route
    `route:anthropic_grok_native_responses_adapter`
  - `claude_adapter_xai_oa_xai_grok_build_child_parallel_read_tools` remains the OAuth
    case for `oa_xai/grok-build` and continues to validate
    `route:anthropic_xai_oauth_responses_adapter`; it is not the Composer unblock path
- OpenRouter focused replacement parallel proof: `claude_adapter_openrouter_nemotron_child_parallel_read_tools`
  - uses `nvidia/nemotron-3-super-120b-a12b:free`
  - hard-gates OpenRouter Responses routing, `session_history`, persisted tool activity, and the one-message parallel `Read` / `Glob` / `Grep` transcript shape
- OpenRouter warning-only canaries: `openai/gpt-oss-20b:free`, `nvidia/nemotron-3-super-120b-a12b:free`
  - `openrouter/free` remains a legacy manual-only diagnostic and is excluded from the default suite; do not use it for release or deployed-container validation because provider selection is intentionally ambiguous.
  - `openai/gpt-oss-20b:free` remains available in config but is excluded from the default full suite because it is an edge OpenRouter target with noisy upstream availability
  - `claude_adapter_gpt_oss_20b` also validates the OpenRouter free daily request row, but failures remain warning-only with the rest of that canary
  - run them only by explicit selection, for example:
    `--cases claude_adapter_gpt_oss_20b,claude_adapter_nemotron_super`
- OpenRouter manual-only spot checks for now: `meta-llama/llama-3.3-70b-instruct:free`, `minimax/minimax-m2.5:free`
- `inclusionai/ling-2.6-flash:free` / `ling-2-6-flash` is retired from active harness targets after OpenRouter started returning `404` for the old free alias. Keep the historical artifacts as breadcrumbs only; the replacement parallel proof is `claude_adapter_openrouter_nemotron_child_parallel_read_tools`.
- `poolside/laguna-m.1:free` is currently listed by OpenRouter as a free tool-capable model, but Claude Code rejected it as unavailable/inaccessible when used as a child-agent model, so do not use it for the parallel proof without a separate model-resolution fix.
- current dev OpenRouter pacing on `:4001`:
  - `AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS=12`
  - `AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS=300`
  - standard `/v1/chat/completions` router pacing in `litellm-dev-config.yaml`:
    `num_retries=2`, `retry_after=10`, `allowed_fails=3`, and
    `cooldown_time=60`
  - no automatic fallback is configured for
    `openrouter/inclusionai/ling-2.6-flash`; switching that lane to another
    model requires an explicit model-quality/cost decision
  This is meant to preserve short hidden retries for sporadic free-model recovery while preventing repeated manual retests from re-burning the same ~40s retry window on persistently throttled models.
  Adapter-managed upstream `429` / `500` / `502` / `503` / `504` responses may still show up as adapter warning/backoff lines, but they should not produce the generic pass-through exception traceback for the current request path.

Preferred Anthropic-adapter model spellings:
- direct OpenAI targets: `openai/gpt-5.4`, `openai/gpt-5.5`, `openai/gpt-5.4-mini`, `openai/gpt-5.3-codex-spark`
- direct OpenRouter targets: `openrouter/openai/gpt-oss-120b:free`, `openrouter/deepseek/deepseek-v4-flash`, `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
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

Moonshot Anthropic Messages acceptance is an opt-in, full agentic case. It
dispatches the `sota-moonshot` child profile through the single canonical
`aawm-sota-moonshot` alias, requires Read then Grep with a tool-result
continuation, and accepts the final marker only after that sequence. It is not
a raw `/v1/messages`, chat-completions, or one-turn completion smoke test.
Run it only against the restarted development service during the bounded
Moonshot live-proof step:

```bash
python3 run_anthropic_adapter_acceptance.py \
  --target dev \
  --cases claude_adapter_aawm_sota_moonshot_agentic_tool_continuation \
  --write-artifact /tmp/moonshot-anthropic-agentic-dev.json
```

The artifact records the safe adapter path
`anthropic_kimi_chat_completions_adapter`, canonical alias, selected K3 model,
and tool-result replay evidence. It does not require or print credential
material.

Native passthrough checks live in the same harness and are opt-in. HTTP cases
send direct requests through the provider passthrough routes. CLI cases launch
the real local provider CLIs (`claude`, `codex`) so they use the same
authentication state as normal operator traffic. All cases validate Langfuse
plus `public.session_history` for the actual provider session id emitted or
derived by that client.
Codex CLI cases also send the current git repository identity via
`x-aawm-repository` and require `public.session_history.repository` to be
populated. Codex CLI cases inject harness-owned tracing headers via `-c
model_providers.<profile>.http_headers.*` overrides. The harness validates
Codex telemetry against the injected `session_id` header rather than the Codex
stdout `thread_id`, since LiteLLM persists the explicit header as the provider
session. Set `AAWM_HARNESS_USER_ID` for a stable caller identity;
pytest-classifier observability maps to `pytest-classifier` automatically.

The native Codex Responses case additionally checks the Codex `spawn_agent`
tool-description rewrite. It requires the
`codex-tool-description-patch` /
`codex-tool-description-patch:spawn-agent-fanout-policy` trace tags, verifies
the generic fanout policy text appears in the logged request, and fails if the
restrictive `Only use spawn_agent if and only if...` language is still present.
This is the deterministic default proof; behavioral subagent fanout can remain
a separate canary.

```bash
python3 run_anthropic_adapter_acceptance.py \
  --target dev \
  --cases native_openai_passthrough_chat,native_openai_passthrough_responses \
  --write-artifact /tmp/native-openai-dev.json
```

Current native cases:
- `native_openai_passthrough_chat`
- `native_openai_passthrough_responses`
- `native_openai_passthrough_responses_codex`
- `native_openai_passthrough_responses_codex_tool_activity`
- `native_anthropic_passthrough_claude`

Provider credentials:
- OpenAI chat/responses use the proxy runtime OpenAI key.
- Codex-native Responses launches `codex exec` against the configured
  `litellm` / `litellm-dev` profile and validates the Codex `thread_id` that
  the CLI emits.
- Native Anthropic launches `claude` with the harness settings overlay and
  validates the Claude CLI `session_id`.
- Use `--litellm-base-url`, `--anthropic-base-url`, `--docker-container-name`, or `--expected-trace-environment` only for explicit one-off overrides.

Important notes:
- top-level Claude runs without an adapted model are not the acceptance target for this suite.
- for adapted free models, LiteLLM / `session_history` are the source of truth for cost, not Claude CLI display cost. For OpenRouter free models, mirror the paid counterpart cost when a non-free twin exists; keep legacy manual-only `openrouter/free` at zero because OpenRouter does not publish a paid twin for it.
- Harness artifacts include `summary.prompt_overhead_cost_share`, an aggregate of
  the `public.session_history` prompt-overhead fields for each validated case.
  The report groups by case, client, route family, counted request shape,
  environment, provider, and model, and breaks input tokens into system,
  tool-advertisement, conversation, residual/other, and system classifier
  buckets. Cost-share fields are estimates derived by weighting
  `response_cost_usd` by each row's prompt-overhead input-token share; treat
  token shares as the primary signal until `session_history` stores exact input
  cost fields.
- For Anthropic rows, `reasoning_tokens_source=provider_reported` is only valid when the provider supplied a positive count; zero placeholders should fall through to estimation or remain unset.
- `reasoning_tokens_source` should not remain null in repaired or newly written `session_history` rows; use `not_applicable` when no reasoning is present and `not_available` when reasoning is present but no positive provider/estimated count exists.
- Anthropic Responses adapter streaming may synthesize client-visible
  `message_delta.usage` only when upstream Responses events omit usage after
  active output. Treat
  `metadata.anthropic_adapter_client_visible_usage_source=estimated` as a
  Claude Code UI compatibility signal, not provider truth; canonical
  `session_history` token, cache-token, and cost fields must continue to use
  provider-reported usage only. Explicit zero provider usage should remain zero
  and should not be replaced by estimates.
- Anthropic/OpenAI/OpenRouter `session_history` rows should also carry normalized provider-cache telemetry. Expect `provider_cache_status` to land as `hit`, `write`, `miss`, `unsupported`, or `not_attempted`, with `provider_cache_miss_reason` populated for miss-shaped outcomes. `provider_cache_miss_token_count` / `provider_cache_miss_cost_usd` are best-effort and should only appear when the miss token count is explicit enough to price honestly.
- AAWM alias-routed rows keep concrete provider attribution in
  `session_history.provider` and `session_history.model`. Use
  `session_history.inbound_model_alias` as the canonical field for the requested
  alias (for example `aawm-read`, `aawm-low`, `aawm-code-anthropic`), with
  `metadata.model_alias_label` and `requested_model_alias` retained for
  compatibility. The alias is also surfaced as generic trace tag
  `model-alias:<alias>`.
- The opt-in fanout suite includes an explicit provider-cache canary through `claude_adapter_peeromega_fanout`: at least one Anthropic child row must show `provider_cache_attempted=true` and `provider_cache_status` of `hit` or `write`.
- When the original proxy spend-log source is unavailable locally, use [scripts/repair_session_history_provider_cache.py](/home/zepfu/projects/litellm/scripts/repair_session_history_provider_cache.py) to repair session-history observability directly from existing rows before relying on historical aggregates. The repair covers inferred `provider`, invalid zero-count `provider_reported` reasoning rows, provider-cache state, cache miss token/cost fields, and git commit/push rollups from `session_history_tool_activity`. Default is dry-run; `--apply` and opt-in `--ensure-schema` both require `current_database()` to match `--target-db-name` (default `aawm_tristore`). Schema DDL is migration-owned and off by default. Git rollups trust complete tool-activity joins only (can lower over-counts; incomplete joins leave stored values).
- Callback agent identity is single-source (RR-003): edit only `litellm/integrations/aawm_agent_identity.py`. `.wheel-build/aawm_litellm_callbacks/agent_identity.py` is a thin checkout re-export loader; hatch force-includes the canonical module into the published `aawm-litellm-callbacks` wheel. `litellm-dev` imports `litellm.integrations.aawm_agent_identity`, while the port-4000 `aawm-litellm` image imports the installed wheel module at `aawm_litellm_callbacks.agent_identity` (same canonical body). After session-history writer changes, rebuild/publish the callback wheel and the production-style image—do not reintroduce a second full source copy under `.wheel-build/`. Guard with `./.venv/bin/python scripts/sync_aawm_agent_identity_to_wheel.py --check`.
- For Codex/OpenAI streaming tool runs, the local source of truth is the reconstructed `response.output_item.*` / `response.function_call_arguments.*` stream state. Expect `usage_tool_call_count`, `codex_tool_call_count`, and `session_history_tool_activity` rows to reflect real tool invocations on `:4001`.
- `claude_adapter_codex_tool_activity` is the hard gate for that path. It uses a single `Bash` / `pwd` tool call and must persist a matching `session_history_tool_activity` row.
- D1-322 low-alias focused acceptance cases are opt-in only. `aawm-low` and `aawm-low-anthropic` now use the default low-alias candidate order without a staging env gate: Antigravity `gemini-3.5-flash-low`, OpenRouter North Mini, OpenRouter Owl Alpha, then the existing candidates. Run explicitly with `--cases claude_adapter_aawm_low_anthropic_alias_child_parallel_read_tools` or `--cases native_openai_passthrough_responses_codex_aawm_low_tool_activity`. Both cases stay excluded from the default suite and validate alias metadata, declared default low-alias candidates, and tool replay for `aawm-low-anthropic` and `aawm-low` respectively. To prove the two OpenRouter chat-completions candidates directly without forcing alias failover, run `--cases claude_adapter_openrouter_north_mini_completion_adapter_smoke,claude_adapter_openrouter_owl_alpha_completion_adapter_smoke`; these cases also stay default-excluded and assert the `anthropic_openrouter_completion_adapter` route family.
- `claude_adapter_ctx_marker` is the hard gate for dynamic-context marker rewriting and uses `:#port-allocation.ctx#:` as the canonical stored-procedure validation fixture.
- `claude_adapter_ctx_marker_escaped` validates literal marker escaping. Prompts can use `\\:#name.ctx#\\:` to show `:#name.ctx#:` to the model without calling `tristore_search_exact` or appending context.
- dispatched child-agent prompts also resolve single-backticked topics like `` `port-allocation` `` and bare uppercase acronyms like `API` through the same tristore lookup, while preserving the inline text and staying silent on misses.
- the general Claude CommonMark-formatting sentence is also rewritten with a tenant/agent-scoped technical-identifier list sourced from `ag_catalog.raw_content`; this currently uses a direct query and is expected to move to a stored procedure.
- NVIDIA Anthropic-adapter targets are currently opt-in spot checks. Preferred spellings are `nvidia/deepseek-ai/deepseek-v3.2`, `nvidia/deepseek-ai/deepseek-v3.1-terminus`, `nvidia/mistralai/devstral-2-123b-instruct-2512`, `nvidia/z-ai/glm4.7`, and `nvidia/minimaxai/minimax-m2.7`; keep the compatibility alias `nvidia/minimax/minimax-m2.7` resolving to `minimaxai/minimax-m2.7`.
- Current NVIDIA harness cases are `claude_adapter_nvidia_deepseek_v32`, `claude_adapter_nvidia_glm47`, and `claude_adapter_nvidia_minimax_m27`. They are excluded from the default suite and should be selected explicitly with `--cases` while the lane is still under active validation.
- Explicit `nvidia/*` model names may also route through the NVIDIA completion adapter by wildcard for early NVIDIA NIM testing. Prefer mapped model names for release gates when cost assertions matter; wildcard trials should still prove `provider=nvidia_nim`, normalized upstream model attribution, route tags, and tool activity when tools are involved.
- `claude_adapter_nvidia_minimax_m27` now uses upstream non-stream plus Anthropic-compatible fake streaming. Keep the exact `nvidia/minimaxai/minimax-m2.7` spelling, and treat it as an explicit opt-in spot check rather than a default-suite canary because MiniMax is materially slower than the other NVIDIA targets.
- These NVIDIA spot checks validate the Anthropic -> NVIDIA completion adapter on `nvidia:/v1/chat/completions` via `provider=nvidia_nim`.
- For NVIDIA-adapted runs, expect the same observability parity as the other adapted providers: `session_history.provider=nvidia_nim` with the normalized upstream model and non-zero cost when pricing is mapped, `session_history_tool_activity` rows when tool or agent-dispatch work occurs, Langfuse trace environment matching the selected `--target`, `route:anthropic_nvidia_completion_adapter` / `anthropic-nvidia-completion-adapter` / `anthropic-adapter-target:nvidia:/v1/chat/completions` tags plus the `anthropic.nvidia_completion_adapter` span, and usable cost tracking. If NVIDIA pricing is not available for a target model, fall back to the closest equivalent OpenRouter pricing rather than leaving long-term cost tracking unmapped.
- For opt-in mixed fanout, the stable tool-activity invariant is the parent session’s delegated `Agent` rows, not child-model command rows. `claude_adapter_peeromega_fanout` should persist at least four `Agent` rows, and `session_history` still hard-gates the expected provider/model/cost rows for each child model.
- The harness keeps the OpenRouter GPT-OSS edge cases available as explicit opt-in checks while excluding them from the default suite; the adapter should still persist non-zero estimated usage/cost from streamed output plus checked-in/bundled model-price JSON when those cases are selected. If `gpt-oss-120b` times out or command-fails solely because the overlapping runtime logs show the exact OpenRouter provider-unavailable signature (`503`, `provider=OpenInference`, `raw=no healthy upstream`), the harness soft-fails it as upstream availability without masking local adapter/logging failures.
- Provider-unavailable soft-fail only converts connectivity/timeout-class failures (for example command failure, healthcheck/container runtime failures, connection/timeout errors). Unrelated hard failures such as session_history, tool_activity, Langfuse trace, or forbidden runtime-log matches stay hard even when the OpenRouter 503 signature is present.
- Docker CLI calls used by this harness (`docker ps` / `docker logs`) use a 30s subprocess timeout so an unresponsive daemon cannot hang the suite indefinitely.
- Runtime-log ignore for concurrent container noise requires positive foreign-traffic evidence near the match. Missing case attribution alone is never enough; upstream `429`/`5xx` signatures also need a foreign model field near the match plus concurrent route evidence.
- Claude projects root for transcript checks resolves from case `claude_projects_root`/`projects_root`, then `CLAUDE_PROJECTS_ROOT`/`CLAUDE_PROJECTS_DIR`, then `Path.home() / '.claude' / 'projects'` (no hardcoded operator home path).
- Cases with missing `required_env` are recorded as skipped with a warning by default. Summary JSON includes `skipped_count` / `skipped_cases`, and stderr prints `[summary] skipped_cases=...`. Set case-level or suite-level `fail_on_skip: true` to treat those skips as hard failures (exit 1).
- Fanout prompts should continue using the Claude agent names from
  `~/.claude/agents`; provider-prefixed routing lives in each agent file's
  `model:` value.
- `openrouter/free` is a legacy manual-only diagnostic, not a default-suite canary or hard gate; upstream routing, rate limits, or model availability can make it noisy even when the local adapter path is correct.
- `warning_only` canaries stay non-gating even when the subprocess itself times out; those conditions should surface as `soft_failures` / warnings in the artifact, not as suite-stopping failures.
- Do not re-add `ling-2-6-flash` as a harness target unless the model is intentionally moved to a currently available paid target and the agent file is restored. Use a replacement OpenRouter model for future release-gating parallel proofs.
