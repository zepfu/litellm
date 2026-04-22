# TODO

## In Progress

- Keep `session_history` invariants enforced after the recent repair work.
  Current target: `reasoning_tokens_source` should remain non-null for new rows, Gemini thought-signature fallback should persist as `provider_signature_present`, Anthropic zero placeholders must not be labeled `provider_reported`, and Anthropic/OpenAI/Gemini/OpenRouter rows should keep a normalized `provider_cache_status` / `provider_cache_miss_reason` shape in `session_history`. When the missed cache token count is explicit, keep `provider_cache_miss_token_count` / `provider_cache_miss_cost_usd` backfilled as well.

- Keep Codex/OpenAI streaming tool activity aligned across Langfuse and `session_history`.
  Current target: `response.output_item.*` / `response.function_call_arguments.*` reconstruction should continue to yield `usage_tool_call_count`, `codex_tool_call_count`, and `session_history_tool_activity` rows for Claude-to-Codex tool runs on `:4001`.

- Keep the adapter harness aligned with the real stored session shapes on `:4001`.
  Current target: `claude_adapter_codex_tool_activity` must hard-gate the `Bash` / `pwd` persistence path, `claude_adapter_ctx_marker` must keep validating `:#port-allocation.ctx#:` via the rewritten request body instead of a brittle exact model reply, dispatched child-agent backtick and bare-acronym lookups should stay aligned with the same `tristore_search_exact` semantics, the CommonMark system-prompt identifier list rewrite should stay aligned with the tenant/agent-scoped raw-content query until the stored procedure lands, and Gemini fanout should keep validating the parent session’s delegated `Agent` rows without assuming every Gemini child emits its own command activity.

## Next

- Add literal ctx-marker escaping for prompts that need to mention `:#name.ctx#:` without triggering context injection.
  Proposed syntax: `\\:#name.ctx#\\:`
  Intended behavior: treat the wrapped marker text as literal prompt content, preserve the visible `:#name.ctx#:` form for the model, and skip stored-procedure lookup and appendix injection.

- Extend the adapter harness to cover the escaped ctx-marker path after the current reasoning/429 fixes are fully validated.

- Add a low-cost provider-cache canary to the live adapter harness once we have a stable fixture that can exercise prompt caching without incurring broad provider spend.
  Intended behavior: a dedicated case should assert `session_history.provider_cache_status` on at least one real adapted path and keep cache-state regressions from hiding behind otherwise successful model replies.
