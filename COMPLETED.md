# Completed

## 2026-04-24

- Promoted the local prod `:4000` LiteLLM container to the `aawm.31` image and isolated the remaining prod validation blockers.
  Prod readiness reported LiteLLM `1.82.3+aawm.31`; `claude_adapter_peeromega_fanout` and `claude_adapter_openrouter_ling_26_flash` passed. `claude_adapter_gpt_oss_120b` was blocked by OpenRouter `503 provider=OpenInference raw=no healthy upstream`, NVIDIA DeepSeek/GLM succeeded but exposed a Langfuse trace-environment propagation gap, and NVIDIA MiniMax remained too slow for the current 300s opt-in spot-check timeout.

- Hardened the next prod validation release for the remaining adapter issues.
  NVIDIA/OpenRouter completion adapters now mirror trace/session context from `litellm_metadata` into normal completion `metadata` so Langfuse trace environment matches session-history environment. Harness `0.0.12` adds a narrow soft-fail classifier for the exact OpenRouter provider-unavailable timeout signature while keeping other hard-gate timeouts hard.

- Promoted the local prod `:4000` LiteLLM container to the durable `aawm.28` image.
  The running `aawm-litellm` container now reports LiteLLM `1.82.3+aawm.28` with callback wheel `0.0.8` and control-plane wheel `0.0.5`.

- Completed prod Anthropic-adapter harness validation against `:4000`.
  Isolated `claude_adapter_gpt54`, `claude_adapter_gpt55`, and `claude_adapter_gpt54_mini` runs passed, followed by a clean full default prod harness pass at `/tmp/litellm-prod-harness-aawm28.json` with zero failures and zero warnings.

- Hardened Claude harness header injection against local Claude settings precedence.
  Harness version `0.0.8` carries the temporary per-run `--settings` overlay for the harness-controlled `ANTHROPIC_BASE_URL` and `ANTHROPIC_CUSTOM_HEADERS`, so `x-litellm-end-user-id` / `langfuse_trace_user_id` validation no longer depends on ambient user or project settings.

- Removed brittle exact-response checks from basic OpenAI smoke cases.
  `gpt-5.4`, `gpt-5.5`, and `gpt-5.4-mini` smoke cases now hard-gate successful command completion, token/cost reporting, adapted-route tags, request payload logging, Langfuse trace/user/session context, runtime logs, and `session_history` rows without failing on harmless natural-language variation from Claude Code context.

## 2026-04-23

- Added LiteLLM runtime and initiating-client identity to `public.session_history`.
  New rows now persist `litellm_environment`, `litellm_version`, `litellm_fork_version`, `litellm_wheel_versions`, `client_name`, `client_version`, and `client_user_agent`; the same values are mirrored into row metadata for Langfuse/backfill context. Live writes derive these from trace environment metadata, installed package versions, associated overlay/config version env vars, request User-Agent, and Claude Code billing headers. Backfill paths only use values already present in spend-log/Langfuse metadata.

- Hardened the Anthropic adapter harness around the new runtime/client identity fields.
  Dev/prod target profile selection now injects `expected_litellm_environment` into session-history checks, and the harness fails rows missing runtime version, fork version, wheel-version JSON, client name, or client version. The callback overlay source remains byte-for-byte synced with the in-repo dev callback and its wheel version is now `0.0.8`.

- Advanced the callback, harness, and model-config artifact versions after branch promotion.
  Remote `cb-v0.0.7`, `h-v0.0.4`, and `cfg-v0.0.4` already existed on a release-only commit, so the durable release artifacts now use fresh non-rewritten versions: callback `0.0.8`, harness `0.0.5`, and model config `0.0.5`.

- Hardened the harness Claude trace-user setup for prod/dev parity.
  Harness version `0.0.6` injects a generated Claude trace-user header value and validates that exact value in Langfuse, without hard-coding an ambient operator identity.

- Advanced the fork image version to `1.82.3+aawm.27` for main-head promotion.
  The initial `v1.82.3-aawm.26` tag was cut before `main` converged, and the guarded image publisher rejects non-main-head tags; `aawm.27` avoids force-moving the published `aawm.26` tag.

- Advanced the fork image version to `1.82.3+aawm.28` after prod-cutover validation found a pass-through trace-user gap.
  Generic and Anthropic pass-through logging now resolve `user_api_key_end_user_id` from standard customer headers such as `x-litellm-end-user-id` when those headers are explicitly supplied, which lets the harness prove custom-header user identity flow end-to-end.

- Repaired `public.session_history` observability gaps in the local `aawm_tristore` database.
  The repair normalized null providers, removed invalid `reasoning_tokens_source=provider_reported` rows with zero reported reasoning tokens, populated target-provider cache statuses, and recalculated cache miss token/cost fields where usage exposed cache-write tokens.

- Hardened the session-history writer against the same regressions.
  New writes now infer missing providers from model/route metadata, sanitize non-positive reasoning token placeholders before persistence, default target-provider cache telemetry from stored cache counters, and preserve valid Gemini `provider_signature_present` reasoning signals.

- Synced the callback overlay source with the in-repo AAWM callback and bumped the callback wheel version to `0.0.6`.
  This prevents future `aawm-litellm` image rebuilds from reinstalling the stale callback wheel that omitted provider-cache and tool-activity fields.

- Broadened git tool-activity extraction.
  Command parsing now handles nested command payloads and git global options such as `git -C /repo commit` / `git --git-dir=/repo/.git push`, with regression coverage for parent `git_commit_count` / `git_push_count` rollups.

- Tightened adapter harness validation for session-history rows.
  Harness validation now selects file/git rollup fields and fails rows with null providers, null reasoning sources, invalid `provider_reported` zero counts, or missing provider-cache status on target provider families.

- Added literal ctx-marker escaping.
  Prompts can now use `\\:#name.ctx#\\:` to preserve visible `:#name.ctx#:` text without triggering `tristore_search_exact`, context appendix injection, or AAWM dynamic-injection metadata.

- Added default-suite harness coverage for escaped ctx markers and provider-cache activity.
  `claude_adapter_ctx_marker_escaped` validates the literal escape path, and `claude_adapter_peeromega_fanout` now requires at least one Anthropic child `session_history` row with `provider_cache_attempted=true` and `provider_cache_status` of `hit` or `write`.

- Added dev/prod target profiles to the Anthropic adapter harness.
  `--target dev` validates `:4001`, `litellm-dev`, and Langfuse trace environment `dev`; `--target prod` validates `:4000`, `aawm-litellm`, and Langfuse trace environment `prod`. The pass-through trace context helper now correctly preserves `session_id` and `trace_environment` metadata on rewritten requests.

- Fixed OpenRouter Anthropic-adapter Responses handling found during prod cutover.
  Native OpenRouter `/v1/responses` payloads now build a Responses-aware logging object instead of falling through OpenAI chat-completions parsing, preventing the prior async `KeyError: choices` logging exception while preserving Codex Responses `local_shell_call` output metadata. Translated Anthropic responses now strip stale upstream `content-length`, `content-encoding`, and `transfer-encoding` headers so synthesized bodies do not trigger `IncompleteRead` / `Too little data for declared Content-Length`. Validated on dev `:4001` with a direct `inclusionai/ling-2.6-flash:free` request returning a complete `200` response.

- Hardened the Anthropic adapter harness against the prod cutover failure modes.
  Runtime-log validation now defaults to failing on async task exceptions, ASGI exceptions, `KeyError: choices`, `h11` protocol errors, stale content-length failures, and upstream 429/5xx passthrough exceptions. Warning-only optional cases still allow configured quality mismatches, but no longer hide command timeouts or runtime-log hard failures. Added local harness unit coverage for timeout hard-fail behavior and default runtime-log detection.

- Fixed the shared OpenAI Responses function-schema failure that was blocking the dev harness.
  Anthropic-to-Responses tool translation now recursively normalizes nested object schemas, including `$defs` nodes that previously declared `type=object` without `properties`. That closes the upstream `400 Invalid schema for function ... object schema missing properties` failure seen on ChatGPT/Codex Responses traffic. Validated with focused passthrough regression tests, isolated live `claude_adapter_gpt54` / `claude_adapter_gpt54_mini` / `claude_adapter_ctx_marker` runs, and a clean full dev harness pass on `:4001`.

- Fixed OpenAI passthrough reconstruction for reasoning-only Responses output items.
  The passthrough logging handler now rebuilds valid `ModelResponse` objects when `response.completed` contains only `ResponseReasoningItem` output, including provider shapes that place reasoning text under `summary` or under `content[type=reasoning_text]`. That removes the prior `Error rebuilding complete responses API stream: Unknown items in responses API response: [ResponseReasoningItem(...)]` warning on `gpt-oss-120b` while preserving usage, hidden `responses_output`, and reasoning text on the reconstructed message. Validated with focused unit coverage plus an isolated live `claude_adapter_gpt_oss_120b` run on `:4001` and a clean overlapping `litellm-dev` log window.

- Added GPT-5.5 support to the Anthropic -> OpenAI Responses adapter.
  The adapter now allowlists `gpt-5.5` / `openai/gpt-5.5`, the local model cost maps include `gpt-5.5` and `chatgpt/gpt-5.5`, and the default Anthropic adapter harness includes `claude_adapter_gpt55`. The cost-map rates match the live Claude CLI `modelUsage` surfaced during validation. Validated with focused unit coverage, JSON/compile checks, and an isolated live `claude_adapter_gpt55` run on dev `:4001` with a clean overlapping `litellm-dev` log window.

## 2026-04-22

- Started direct NVIDIA adapter support for Anthropic-routed agent models.
  Added a dedicated `anthropic_nvidia_completion_adapter` lane for `nvidia/...` models targeting `nvidia:/v1/chat/completions`, normalized provider-prefix handling for `nvidia` / `nvidia_nim`, and compatibility aliasing for `nvidia/minimax/minimax-m2.7` -> `minimaxai/minimax-m2.7`.

- Added NVIDIA-specific egress-family and logging support.
  `integrate.api.nvidia.com` / `ai.api.nvidia.com` now classify as `nvidia` targets for outbound credential-guard checks and as OpenAI-compatible endpoints for passthrough logging / synthetic adapted-route reconstruction.

- Added NVIDIA env passthrough and model metadata needed for local validation.
  `docker-compose.dev.yml` now passes the NVIDIA API/base env vars through to `litellm-dev`, `model_prices_and_context_window.json` now has `nvidia_nim/...` entries for the new agent targets, and the local Anthropic adapter harness config now includes optional NVIDIA spot-check cases excluded from the default suite.

- Added focused unit coverage for the NVIDIA lane.
  Current tests cover NVIDIA egress validation, Anthropic adapter model resolution and route rewriting, and OpenAI-compatible logging host detection for NVIDIA completion endpoints.

- Hardened the NVIDIA completion adapter against provider-specific request and timeout issues.
  The route no longer forwards the empty `standard_callback_dynamic_params` payload into NVIDIA chat-completions requests, and it now applies a short per-attempt timeout plus hidden retry for transient `408` / `429` / `5xx` failures before surfacing an HTTP error.

- Stabilized the high-latency MiniMax path inside the NVIDIA Anthropic adapter.
  The NVIDIA lane now avoids nested OpenAI-provider retries by forcing the inner provider `max_retries` to `0`, uses a higher default adapter timeout, and fake-streams `minimaxai/minimax-m2.7` through Anthropic SSE while keeping the upstream NVIDIA call non-streaming.

- Established stable live NVIDIA spot checks on `:4001`.
  `claude_adapter_nvidia_deepseek_v32`, `claude_adapter_nvidia_glm47`, and `claude_adapter_nvidia_minimax_m27` now pass explicit end-to-end validation against `litellm-dev`, including `session_history` provider/model/cost validation, request-body or adapted-request checks, trace tag checks, and runtime-log traceback guards. They remain excluded from the default suite because the NVIDIA lane is still opt-in and MiniMax is materially slower than the other NVIDIA targets.

- Added normalized provider-cache telemetry to `public.session_history` for Anthropic, OpenAI, Gemini, and OpenRouter.
  Stored fields now include `provider_cache_attempted`, `provider_cache_status`, `provider_cache_miss`, and `provider_cache_miss_reason`.

- Added best-effort provider-cache miss quantification for explicit cache-write misses.
  When the miss token count is knowable from provider usage, `session_history` now also stores `provider_cache_miss_token_count` and `provider_cache_miss_cost_usd` using the write-vs-read delta for the affected tokens.

- Added [scripts/repair_session_history_provider_cache.py](/home/zepfu/projects/litellm/scripts/repair_session_history_provider_cache.py) for historical provider-cache repair when the original proxy spend-log source is not available locally.
  The script repairs `session_history` in place from the stored provider/model/cache counters and persisted metadata, and only prices miss cost when the missed cache token count is explicit enough to do so honestly.

- Added provider-cache metadata enrichment for Langfuse / request metadata.
  Current normalized states are `hit`, `write`, `miss`, `unsupported`, and `not_attempted`.

- Added targeted unit coverage for provider-cache detection from both direct LiteLLM result shaping and Langfuse-trace backfill shaping.

- Tightened the adapter harness and docs around current `session_history` invariants so reasoning, tool activity, and provider-cache signals stay visible during validation work on `:4001`.
