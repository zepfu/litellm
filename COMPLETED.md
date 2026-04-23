# Completed

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

- Established stable live NVIDIA canaries on `:4001`.
  `claude_adapter_nvidia_deepseek_v32` and `claude_adapter_nvidia_glm47` now pass end to end against `litellm-dev`, including `session_history` provider/model/cost validation, request-body/trace tag checks, and runtime-log traceback guards. `claude_adapter_nvidia_minimax_m27` remains supported via routing/unit coverage but is manual-only for now because the live Claude/NVIDIA path still stalls intermittently.

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
