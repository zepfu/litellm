## Deferred Adapter Fidelity Backlog

- Anthropic-route adapter validation and promotion policy
  - Keep the focused Claude/adapter harness as the promotion gate for native
    OpenAI/Codex, Google Code Assist, and OpenRouter paths.
  - Keep warning-only OpenRouter canaries separate from hard gates unless the
    provider/model has stable quota and output behavior.

- Gemini adapter follow-ups
  - Keep validating Gemini fanout against live traffic rather than treating
    isolated `429` / `MODEL_CAPACITY_EXHAUSTED` as final provider truth without
    corroborating native Gemini CLI `/model` status on the same account.
  - Remaining broad work is harness automation plus wall-clock reduction for
    paced follow-up turns.

- OpenRouter elephant-alpha / wildcard reliability
  - Remaining issue is child-agent convergence/tool-use behavior plus
    intermittent free-tier pacing under repeated provider throttling windows.
  - Continue to distinguish provider instability from adapter regressions with
    narrow direct-model smokes and captured request/response shape.

- Raw Anthropic MCP server/toolset requests are intentionally deferred.

- Prompt caching semantics
  - Keep provider-cache attempt/hit/miss semantics explicit per provider.
  - Avoid global changes that would reclassify partial hits or request-side
    cache markers without tests and reporting-contract updates.

- Documents / files / citations / PDFs
  - Deferred advanced Anthropic-native fidelity work. Do not mix into current
    auto-agent or session-history fixes without a fresh scoped task.

- Thinking / reasoning fidelity
  - Deferred advanced Anthropic-native fidelity work.

- Context management fidelity
  - Deferred advanced Anthropic-native fidelity work.

- Anthropic-native advanced tools
  - Deferred advanced Anthropic-native fidelity work.

- Response-shape fidelity beyond the basic path
  - Deferred advanced adapter fidelity work.

- Full response lifecycle timing instrumentation
  - Session-history latency fields are live. Any additional lifecycle timing
    should start from a new concrete gap.

- Remote adapter/model config reload
  - Deferred operational enhancement.
