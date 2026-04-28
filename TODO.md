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

## Operating Loop

- Preserve context aggressively. For substantial investigations, feature work,
  live harness analysis, or multi-lane debugging, fan out to subagents whenever
  the subtask is concrete and can run independently. Use the main thread for
  integration, decisions, and the current critical path.
- Keep [TODO.md](TODO.md) and [COMPLETED.md](COMPLETED.md) current while work is
  underway, not just at the end. `TODO.md` should say what is currently
  unresolved, what failed, and the next plan of attack. `COMPLETED.md` should
  record what was fixed, what was validated, exact artifacts/traces when useful,
  and any dead ends that should not be repeated.
- When a fix is believed to work, proactively add the evidence to
  [COMPLETED.md](COMPLETED.md). If later validation fails, update
  [COMPLETED.md](COMPLETED.md) with the failed result and update [TODO.md](TODO.md)
  with the next iteration before continuing.
- After each major feature or repair is delivered and checked in locally, commit
  it on the `develop` branch. Keep the commit boundary aligned with the proven
  behavior so later sessions can tell what code, tests, and notes belonged
  together.
- Avoid repeating known dead ends. Before rerunning an approach, check
  [COMPLETED.md](COMPLETED.md) for prior failures and [TODO.md](TODO.md) for the
  current planned route.

## In Progress

- Sequential Claude-dispatch base-tool proof is complete for OpenAI/GPT and
  Gemini through dev `:4001`. GPT-5.5 passed at
  `/tmp/claude_adapter_gpt55_child_sequential_core_tools_unique.json`; Gemini
  3.1 Pro remains classified as a Google Code Assist quota/capacity block at
  `/tmp/claude_adapter_gemini31_pro_child_sequential_core_tools_unique.json`;
  Gemini 3 Flash passed at
  `/tmp/claude_adapter_gemini3_flash_child_sequential_core_tools_after_tool_pair_boundary.json`
  with exactly one each of `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`,
  `WebSearch`, and `WebFetch`, `max_tool_uses_in_single_assistant_message=1`,
  no transcript tool-result errors, clean runtime logs, valid tenant-only
  Langfuse user ids, and durable tool activity. Keep the dead-end breadcrumbs in
  [COMPLETED.md](COMPLETED.md): do not rerun the neutral-fixture prompt
  unchanged, do not use parent `--tools Agent`, do not chase the old synthetic
  `Carvercall` text unless it recurs as live assistant output, and do not treat
  prompt hardening as the fix for the post-Bash Gemini stop.

- Keep the Claude-dispatched Gemini parallel tool-call stream fix classified as
  validated for the available Gemini lane. Smoking gun: Langfuse trace
  `d0275abf-2e5f-4964-b875-74159927acb1` recorded one upstream Gemini
  generation with `usage_tool_call_count=5` (`read_file` x4 plus
  `run_shell_command`), while Claude Code child transcript
  `agent-ab792fb928eb0da98.jsonl` only received one `Read` tool_use for the
  same turn and immediately logged `API Error: Content block not found`. The
  adapter patch now buffers Gemini streaming tool-call deltas and emits one
  Anthropic `tool_use` block per Gemini tool call. Focused unit validation
  passed in
  `tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_parallel_tool_calls.py`
  including pre-terminal and terminal parallel-call chunks; the latest combined
  stream/SSE run passed (`19 passed`) and adapter transformation tests passed
  (`63 passed`). `litellm-dev` has been restarted with `:4001` healthy. Gemini
  3 Flash also passed the live parallel read-tool proof on dev `:4001` at
  `/tmp/claude_adapter_gemini3_flash_child_parallel_read_tools_rerun.json`.
  Gemini 3.1 Pro can be rerun when the Google Code Assist quota/capacity block
  clears, but do not treat that quota block as an unvalidated stream-fix gap.

## Next

- Finish the separate parallel-call proof now that sequential base-tool
  coverage has passed. Gemini 3 Flash passed the live Claude-dispatch parallel
  read-tool proof on dev `:4001` at
  `/tmp/claude_adapter_gemini3_flash_child_parallel_read_tools_rerun.json`,
  with `Read`, `Glob`, and `Grep` emitted in one assistant message and durable
  tool activity recorded. GPT-5.5 still serializes the same request: artifact
  `/tmp/claude_adapter_gpt55_child_parallel_read_tools.json` failed because the
  child transcript never had `>= 3` `tool_use` blocks in a single assistant
  message (`max=1`). The OpenAI Responses adapter now emits
  `parallel_tool_calls=true` for function-tool requests when Anthropic has not
  disabled parallel tool use, and the GPT harness case now hard-gates that
  logged payload field. Live rerun
  `/tmp/claude_adapter_gpt55_child_parallel_read_tools_rerun.json` confirmed the
  flag was present but GPT-5.5 still serialized `Read`, `Glob`, and `Grep` into
  separate assistant messages. Direct `/openai_passthrough/v1/responses` probes
  showed GPT-5.5 can parallelize the same tools and same child task when the
  large Claude Code injected context is absent, but emits only `Read` when the
  full logged three-block Claude Code input is reused. Next work is context
  shaping for OpenAI-dispatched Claude Code subagents: minimize the logged
  Claude Code injected context, identify the exact semantic interference,
  decide which real context should be summarized, omitted, or moved for adapted
  non-Anthropic models without breaking normal agent grounding, then rerun
  `claude_adapter_gpt55_child_parallel_read_tools` on dev `:4001`.

## Ongoing

- Keep Codex/OpenAI streaming tool activity aligned across Langfuse and `session_history`.
  Current target: `response.output_item.*` / `response.function_call_arguments.*` reconstruction should continue to yield `usage_tool_call_count`, `codex_tool_call_count`, and `session_history_tool_activity` rows for Claude-to-Codex tool runs on `:4001`.

- Keep GPT/OpenAI Claude-dispatch file-write behavior classified correctly.
  Smoking gun for the 2026-04-28 aawm-tap failure: the real GPT-5.5 request put
  the Claude Code no-write instruction in a plain string `instructions` field,
  while the runtime prompt patcher only rewrote `{"type":"text","text":...}`
  blocks. Trace `0f1f34a2-c892-4a2f-87ef-e8dd4687bbcf` therefore still carried
  the old absolute no-write sentence and lacked
  `subagent-report-file-explicit-request`. Fixed by patching plain strings and
  the `${...}` template variant in
  `aawm_claude_control_plane.py`; live dev validation passed at
  `/tmp/claude_adapter_gpt55_child_analysis_write_probe_rerun.json`, where the
  request text carried the patched sentence, the old sentence was absent,
  `session_history` recorded `file_modified_count=1`, and GPT-5.5 persisted
  both `Bash` and `Write` tool activity for
  `/tmp/gpt55-analysis-write-probe.md`.

- Keep Claude-dispatched model latency classified separately from native CLI
  latency. For GPT-5.5, current evidence still points to large model turns plus
  Claude Code/orchestrator overhead: the focused GPT-5.5 write probe took
  `50.192s` wall clock through Claude Code while the GPT adapter session logged
  about `2.156s` proxy/upstream duration; later aawm-tap GPT spans still ran
  `64-172s`. For Gemini, do not classify all delay as context size: the v11
  aawm-tap trace above proved a concrete adapter stream bug where five upstream
  Gemini tool calls collapsed to one Claude-visible `Read` and a Claude Code
  protocol error. Native Codex CLI / Gemini CLI speed remains separate because
  this work targets Claude Code dispatch through LiteLLM's Anthropic-compatible
  adapter.

- Keep the adapter harness aligned with the real stored session shapes on both `:4001` dev and `:4000` prod targets.
  Current target: `claude_adapter_codex_tool_activity` must hard-gate the `Bash` / `pwd` persistence path, `claude_adapter_ctx_marker` must keep validating `:#port-allocation.ctx#:` via the rewritten request body instead of a brittle exact model reply, `claude_adapter_ctx_marker_escaped` must keep validating `\\:#name.ctx#\\:` literal escaping, dispatched child-agent backtick and bare-acronym lookups should stay aligned with the same `tristore_search_exact` semantics, the CommonMark system-prompt identifier list rewrite should stay aligned with the tenant/agent-scoped raw-content query until the stored procedure lands, the provider-cache canary should keep finding at least one Anthropic `hit` / `write` row in the default suite, Gemini fanout should now deliberately hard-gate child native `run_shell_command` rows rather than relying on plausible final text, and the direct Gemini Read gate is `claude_adapter_gemini31_pro_read_tool_id_sanitizer` rather than the fanout case. The post-tool-result Gemini gate is `claude_adapter_gemini31_pro_bash_then_read_stream_state`; keep it default-excluded but run it explicitly when Claude-dispatched Gemini fails after an initial tool call. Persisted Gemini tool activity uses native `read_file` / `run_shell_command` while Claude Code sees restored `Read` / `Bash`, so validators should match the stored native tool names and require the matching tool-call rows instead of the latest no-tool final row. `--target dev` / `--target prod` should continue enforcing the correct port, Docker container, and Langfuse trace environment. Claude trace-user validation should validate tenant-only Langfuse user ids (`userId=<tenant_id>`) while trace names carry the agent (`claude-code.<agent>`); do not use `project.agent` as the user id. Basic OpenAI smoke cases should validate success, usage, cost, routing, Langfuse, and session-history invariants rather than brittle exact natural-language output. Keep the prod-cutover failure guards active by default: async task exceptions, ASGI exceptions, `KeyError: choices`, stale `Content-Length` / `h11` protocol failures, upstream passthrough 429/5xx traces, and the OpenAI Responses nested-object-schema regression must fail the run instead of surfacing only as downstream session-history gaps; warning-only optional cases must not mask command timeouts or runtime-log hard failures. Before future prod promotions, add a production-style preflight that validates the exact image / installed wheel path on `:4001` plus a small explicit promotion-gate set for opt-in provider lanes, so packaging and adapter metadata gaps are caught before touching `:4000`.

- Keep future harness bundle publishes on version `0.0.14` or newer.
  The `0.0.14` harness bundle includes the controlled Claude trace `userId` validation, explicit per-run Claude settings overlay, longer peeromega fanout timeout, the narrow OpenRouter provider-unavailable timeout / command-failure classifier, and the default-suite exclusion for GPT-OSS edge cases needed for real prod validation.
