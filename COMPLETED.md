# Completed

## 2026-04-22

- Added normalized provider-cache telemetry to `public.session_history` for Anthropic, OpenAI, Gemini, and OpenRouter.
  Stored fields now include `provider_cache_attempted`, `provider_cache_status`, `provider_cache_miss`, and `provider_cache_miss_reason`.

- Added provider-cache metadata enrichment for Langfuse / request metadata.
  Current normalized states are `hit`, `write`, `miss`, `unsupported`, and `not_attempted`.

- Added targeted unit coverage for provider-cache detection from both direct LiteLLM result shaping and Langfuse-trace backfill shaping.

- Tightened the adapter harness and docs around current `session_history` invariants so reasoning, tool activity, and provider-cache signals stay visible during validation work on `:4001`.
