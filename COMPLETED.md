# Completed

## 2026-04-23

- Repaired `public.session_history` observability gaps in the local `aawm_tristore` database.
  The repair normalized null providers, removed invalid `reasoning_tokens_source=provider_reported` rows with zero reported reasoning tokens, populated target-provider cache statuses, and recalculated cache miss token/cost fields where usage exposed cache-write tokens.

- Hardened the session-history writer against the same regressions.
  New writes now infer missing providers from model/route metadata, sanitize non-positive reasoning token placeholders before persistence, default target-provider cache telemetry from stored cache counters, and preserve valid Gemini `provider_signature_present` reasoning signals.

- Synced the callback overlay source with the in-repo AAWM callback and bumped the callback wheel version to `0.0.5`.
  This prevents future `aawm-litellm` image rebuilds from reinstalling the stale callback wheel that omitted provider-cache and tool-activity fields.

- Broadened git tool-activity extraction.
  Command parsing now handles nested command payloads and git global options such as `git -C /repo commit` / `git --git-dir=/repo/.git push`, with regression coverage for parent `git_commit_count` / `git_push_count` rollups.

- Tightened adapter harness validation for session-history rows.
  Harness validation now selects file/git rollup fields and fails rows with null providers, null reasoning sources, invalid `provider_reported` zero counts, or missing provider-cache status on target provider families.

- Added literal ctx-marker escaping.
  Prompts can now use `\\:#name.ctx#\\:` to preserve visible `:#name.ctx#:` text without triggering `tristore_search_exact`, context appendix injection, or AAWM dynamic-injection metadata.

- Added default-suite harness coverage for escaped ctx markers and provider-cache activity.
  `claude_adapter_ctx_marker_escaped` validates the literal escape path, and `claude_adapter_peeromega_fanout` now requires at least one Anthropic child `session_history` row with `provider_cache_attempted=true` and `provider_cache_status` of `hit` or `write`.

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
