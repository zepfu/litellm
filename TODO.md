# TODO

## Cold Start Orientation

Use this file as the primary restart surface. For a no-context session, first read
the current `In Progress` and `Next` sections below, then branch into these docs
only as needed:

- [COMPLETED.md](COMPLETED.md) - recent completed work, validation results,
  repair stats, and release/session-history context from prior sessions.
- [TEST_HARNESS.md](TEST_HARNESS.md) and
  [scripts/local-ci/README.md](scripts/local-ci/README.md) - existing adapter
  harness structure, target profiles, real CLI coverage, case selection,
  artifacts, and session-history/Langfuse assertions.
- [PROD_RELEASE.md](PROD_RELEASE.md) - dev-to-prod promotion process, exact
  image/wheel/config verification, prod `:4000` harness validation, release
  asset expectations, and rollback notes.
- [PATCHES.md](PATCHES.md) - local fork patch workflow and patch-management
  expectations.
- [WHEEL.md](WHEEL.md) - callback/control-plane wheel packaging, artifact
  versioning, install paths, and runtime/session-history behavior carried by
  the overlay wheels.
- [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) - repo conventions,
  development commands, and broader LiteLLM testing/lint guidance.

## In Progress

- None at the moment.

## Next

- Rerun the full default dev/prod harnesses after the upstream Codex/OpenAI account usage limit resets.
  The PostgreSQL connection-pressure hardening is complete and targeted recent-window dev/prod log scans no longer show DB pressure signatures. The latest full default dev and prod runs were blocked by upstream OpenAI/Codex `usage_limit_reached` 429s in the Codex/Spark cases; the dev failure reported reset time `2026-04-25 20:27:55 UTC`, and the prod OpenRouter canary shard passed with warning-only optional Gemma notes. Do not skip or soften those cases; rerun the normal default harnesses once the upstream quota reset clears.

## Ongoing

- Keep Codex/OpenAI streaming tool activity aligned across Langfuse and `session_history`.
  Current target: `response.output_item.*` / `response.function_call_arguments.*` reconstruction should continue to yield `usage_tool_call_count`, `codex_tool_call_count`, and `session_history_tool_activity` rows for Claude-to-Codex tool runs on `:4001`.

- Keep the adapter harness aligned with the real stored session shapes on both `:4001` dev and `:4000` prod targets.
  Current target: `claude_adapter_codex_tool_activity` must hard-gate the `Bash` / `pwd` persistence path, `claude_adapter_ctx_marker` must keep validating `:#port-allocation.ctx#:` via the rewritten request body instead of a brittle exact model reply, `claude_adapter_ctx_marker_escaped` must keep validating `\\:#name.ctx#\\:` literal escaping, dispatched child-agent backtick and bare-acronym lookups should stay aligned with the same `tristore_search_exact` semantics, the CommonMark system-prompt identifier list rewrite should stay aligned with the tenant/agent-scoped raw-content query until the stored procedure lands, the provider-cache canary should keep finding at least one Anthropic `hit` / `write` row in the default suite, Gemini fanout should keep validating the parent session’s delegated `Agent` rows without assuming every Gemini child emits its own command activity, and `--target dev` / `--target prod` should continue enforcing the correct port, Docker container, and Langfuse trace environment. Claude trace-user validation must inject a harness-controlled `x-litellm-end-user-id` / `langfuse_trace_user_id` value via the explicit per-run Claude `--settings` overlay and validate that exact value without hard-coding an ambient operator identity. Basic OpenAI smoke cases should validate success, usage, cost, routing, Langfuse, and session-history invariants rather than brittle exact natural-language output. Keep the prod-cutover failure guards active by default: async task exceptions, ASGI exceptions, `KeyError: choices`, stale `Content-Length` / `h11` protocol failures, upstream passthrough 429/5xx traces, and the OpenAI Responses nested-object-schema regression must fail the run instead of surfacing only as downstream session-history gaps; warning-only optional cases must not mask command timeouts or runtime-log hard failures. Before future prod promotions, add a production-style preflight that validates the exact image / installed wheel path on `:4001` plus a small explicit promotion-gate set for opt-in provider lanes, so packaging and adapter metadata gaps are caught before touching `:4000`.

- Keep future harness bundle publishes on version `0.0.14` or newer.
  The `0.0.14` harness bundle includes the controlled Claude trace `userId` validation, explicit per-run Claude settings overlay, longer peeromega fanout timeout, the narrow OpenRouter provider-unavailable timeout / command-failure classifier, and the default-suite exclusion for GPT-OSS edge cases needed for real prod validation.
