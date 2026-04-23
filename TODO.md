# TODO

## In Progress

- Finish the NVIDIA Anthropic-adapter lane and validation work.
  Current target: direct `nvidia/...` agent models from `~/.claude/agents` should route through a dedicated `anthropic_nvidia_completion_adapter` lane with `nvidia` egress-family enforcement, OpenAI-compatible NVIDIA logging support, and `nvidia_nim/...` cost-map coverage. The end state should match the existing adapted providers: `session_history` rows with the normalized upstream model and non-zero cost when pricing is mapped, `session_history_tool_activity` rows when tool/agent dispatch occurs, and Langfuse tags / metadata / spans including `route:anthropic_nvidia_completion_adapter`, `anthropic-nvidia-completion-adapter`, `anthropic-adapter-target:nvidia:/v1/chat/completions`, and the `anthropic.nvidia_completion_adapter` span name. If NVIDIA does not expose usable non-free pricing for a target model, use the closest equivalent OpenRouter pricing as the fallback basis rather than leaving long-term cost tracking unmapped. The remaining work is to keep the focused `:4001` harness cases for `claude_adapter_nvidia_deepseek_v32`, `claude_adapter_nvidia_glm47`, and `claude_adapter_nvidia_minimax_m27` green while documenting that MiniMax currently uses upstream non-stream plus Anthropic-compatible fake streaming and remains excluded from the default suite because of its higher latency profile.

- Keep Codex/OpenAI streaming tool activity aligned across Langfuse and `session_history`.
  Current target: `response.output_item.*` / `response.function_call_arguments.*` reconstruction should continue to yield `usage_tool_call_count`, `codex_tool_call_count`, and `session_history_tool_activity` rows for Claude-to-Codex tool runs on `:4001`.

- Keep the adapter harness aligned with the real stored session shapes on `:4001`.
  Current target: `claude_adapter_codex_tool_activity` must hard-gate the `Bash` / `pwd` persistence path, `claude_adapter_ctx_marker` must keep validating `:#port-allocation.ctx#:` via the rewritten request body instead of a brittle exact model reply, dispatched child-agent backtick and bare-acronym lookups should stay aligned with the same `tristore_search_exact` semantics, the CommonMark system-prompt identifier list rewrite should stay aligned with the tenant/agent-scoped raw-content query until the stored procedure lands, and Gemini fanout should keep validating the parent session’s delegated `Agent` rows without assuming every Gemini child emits its own command activity.

## Next

- Publish / rebuild the callback overlay wheel after the `session_history` repair hardening lands.
  Current local runtime mitigation: `aawm-litellm` was patched in-place by copying the updated callback into both `/usr/lib/python3.13/site-packages/litellm/integrations/aawm_agent_identity.py` and `/usr/lib/python3.13/site-packages/aawm_litellm_callbacks/agent_identity.py`, then restarting the compose service. The durable fix is the checked-in `.wheel-build` source plus version `0.0.5`; the next image rebuild should consume a new `cb-v*` callback wheel instead of the old released wheel.

- Add literal ctx-marker escaping for prompts that need to mention `:#name.ctx#:` without triggering context injection.
  Proposed syntax: `\\:#name.ctx#\\:`
  Intended behavior: treat the wrapped marker text as literal prompt content, preserve the visible `:#name.ctx#:` form for the model, and skip stored-procedure lookup and appendix injection.

- Extend the adapter harness to cover the escaped ctx-marker path after the current reasoning/429 fixes are fully validated.

- Add a low-cost provider-cache canary to the live adapter harness once we have a stable fixture that can exercise prompt caching without incurring broad provider spend.
  Intended behavior: a dedicated case should assert `session_history.provider_cache_status` on at least one real adapted path and keep cache-state regressions from hiding behind otherwise successful model replies.
