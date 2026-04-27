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

- Implement a Google-only Gemini system prompt policy for Claude-initiated
  `/anthropic` requests that route through `anthropic_google_completion_adapter`.
  Saved design note and suggested prompt:
  [.analysis/google-anthropic-system-prompt-policy-2026-04-27.md](.analysis/google-anthropic-system-prompt-policy-2026-04-27.md).
  Restartable plan:
  1. Add the rewrite in
     `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` inside
     `_build_google_code_assist_request_from_completion_kwargs()`, after
     Anthropic input is canonicalized and before `_transform_request_body()`
     creates Gemini `systemInstruction`.
  2. Keep the existing Claude-to-Gemini native tool aliasing intact; the prompt
     policy should change operating instructions, not the tool schema mapping.
  3. Feature-flag the policy with values such as `off`, `append`, and
     `replace_compact`; use `replace_compact` as the target behavior after
     validation.
  4. Strip Claude/Anthropic provider overhead from the system prompt while
     preserving project, workspace, safety, memory, and task constraints.
  5. Inject the compact Gemini CLI-style agent prompt from the analysis note and
     append preserved instructions under a clear heading.
     Suggested compact prompt:
     ```text
     You are a non-interactive CLI software engineering agent.

     Work in this cycle: understand, plan briefly, implement, verify, finalize.
     Use the provided tools to inspect and modify the workspace when the task
     requires it.

     Tool usage:
     - Prefer search tools before broad file reads.
     - Batch independent search/read calls in parallel when possible.
     - Use write/edit tools to complete requested artifacts or code changes.
     - If a tool is unavailable or blocked, recover with another available tool.
     - Do not remain in read-only exploration when the user requested an
       implementation or artifact.

     Follow the preserved project, workspace, safety, and operator instructions
     below.
     ```
  6. Record prompt-policy metadata without logging the full prompt by default:
     policy name/version, original and rewritten system text sizes, removed
     Claude-overhead size, and preserved-instruction size.
  7. Add focused unit coverage for disabled/append/replace modes, preservation
     of safety/project instructions, removal of known Claude-only markers,
     generated `request.systemInstruction`, metadata summary fields, and no
     regression in Gemini-native tool aliases.
  8. Extend the existing Anthropic adapter harness with a focused
     `claude_adapter_gemini_*` case rather than creating a new harness. The case
     should launch the real Claude CLI through `:4001/anthropic`, select a
     Gemini child model through the normal Claude agent/fanout path, require a
     bounded artifact-write/edit outcome, and validate the logged Google Code
     Assist request body contains the compact Gemini prompt marker while
     excluding known Claude-only overhead markers.
  9. Harness validation should also assert the prompt-policy metadata in
     Langfuse/session-history, preserve the existing Gemini provider/model/cost
     and trace-environment checks, scan runtime logs for blocker patterns, and
     treat `write_file`/write-edit tool activity as a hard gate only where the
     current harness can observe child tool activity reliably; otherwise hard
     gate final artifact creation plus logged request-body prompt-policy checks.
  10. Validate with the focused dev harness case, the full default dev harness,
      then the same focused/full checks on prod during the next promotion.

- Normalize invalid empty optional Claude tool arguments returned by
  non-Anthropic models on the `/anthropic` adapter path. Saved issue note:
  [.analysis/gpt55-claude-read-pages-empty-arg-2026-04-27.md](.analysis/gpt55-claude-read-pages-empty-arg-2026-04-27.md).
  Restartable plan:
  1. Add a narrow adapter response sanitizer for Claude tool calls before they
     are returned to Claude Code. Immediate target: `Read` tool calls with
     `pages: ""` should omit `pages` entirely because Claude Code only accepts
     meaningful PDF page ranges.
  2. Prefer a shared helper in the Anthropic experimental pass-through adapter
     translation layer so non-streaming and streaming tool-call output paths get
     the same normalization. Inspect
     `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`
     and `streaming_iterator.py` before choosing the exact hook point.
  3. Keep the sanitizer narrow: do not globally delete empty strings from all
     tool arguments, since empty strings may be valid for other tools.
  4. Preserve existing tool-name mapping behavior so the sanitizer can act on
     the original Claude tool name even when provider-facing tool names were
     truncated or aliased.
  5. Add unit tests for non-streaming output where GPT/OpenAI returns
     `Read({"file_path": "...", "pages": ""})` and the Anthropic `tool_use`
     response omits `pages`; add streaming coverage if the existing streaming
     test helpers can exercise assembled tool-call arguments.
  6. Add a negative unit test proving unrelated tools or valid empty-string
     arguments are not stripped.
  7. Extend the existing Anthropic adapter harness with a focused GPT-5.5/OpenAI
     Claude-dispatch case that reads a small source/text file, validates the
     run does not loop on `Invalid pages parameter`, validates the logged
     translated tool activity/request output does not contain `Read.pages=""`,
     and keeps normal provider/model/cost, Langfuse, runtime-log, and
     `session_history` invariants.
  8. Optionally add a prompt-side belt-and-suspenders instruction to relevant
     GPT/OpenAI dispatch prompts: when calling `Read`, omit `pages` unless
     reading a PDF with a real page range. This should supplement, not replace,
     adapter-side sanitization.

## Ongoing

- Keep Codex/OpenAI streaming tool activity aligned across Langfuse and `session_history`.
  Current target: `response.output_item.*` / `response.function_call_arguments.*` reconstruction should continue to yield `usage_tool_call_count`, `codex_tool_call_count`, and `session_history_tool_activity` rows for Claude-to-Codex tool runs on `:4001`.

- Keep the adapter harness aligned with the real stored session shapes on both `:4001` dev and `:4000` prod targets.
  Current target: `claude_adapter_codex_tool_activity` must hard-gate the `Bash` / `pwd` persistence path, `claude_adapter_ctx_marker` must keep validating `:#port-allocation.ctx#:` via the rewritten request body instead of a brittle exact model reply, `claude_adapter_ctx_marker_escaped` must keep validating `\\:#name.ctx#\\:` literal escaping, dispatched child-agent backtick and bare-acronym lookups should stay aligned with the same `tristore_search_exact` semantics, the CommonMark system-prompt identifier list rewrite should stay aligned with the tenant/agent-scoped raw-content query until the stored procedure lands, the provider-cache canary should keep finding at least one Anthropic `hit` / `write` row in the default suite, Gemini fanout should keep validating the parent session’s delegated `Agent` rows without assuming every Gemini child emits its own command activity, and `--target dev` / `--target prod` should continue enforcing the correct port, Docker container, and Langfuse trace environment. Claude trace-user validation must inject a harness-controlled `x-litellm-end-user-id` / `langfuse_trace_user_id` value via the explicit per-run Claude `--settings` overlay and validate that exact value without hard-coding an ambient operator identity. Basic OpenAI smoke cases should validate success, usage, cost, routing, Langfuse, and session-history invariants rather than brittle exact natural-language output. Keep the prod-cutover failure guards active by default: async task exceptions, ASGI exceptions, `KeyError: choices`, stale `Content-Length` / `h11` protocol failures, upstream passthrough 429/5xx traces, and the OpenAI Responses nested-object-schema regression must fail the run instead of surfacing only as downstream session-history gaps; warning-only optional cases must not mask command timeouts or runtime-log hard failures. Before future prod promotions, add a production-style preflight that validates the exact image / installed wheel path on `:4001` plus a small explicit promotion-gate set for opt-in provider lanes, so packaging and adapter metadata gaps are caught before touching `:4000`.

- Keep future harness bundle publishes on version `0.0.14` or newer.
  The `0.0.14` harness bundle includes the controlled Claude trace `userId` validation, explicit per-run Claude settings overlay, longer peeromega fanout timeout, the narrow OpenRouter provider-unavailable timeout / command-failure classifier, and the default-suite exclusion for GPT-OSS edge cases needed for real prod validation.
