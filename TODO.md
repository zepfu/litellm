# TODO

## Cold Start Orientation

Use this file as the primary restart surface. For a no-context session, first read
the current `Validated Context` and `Next` sections below, then branch into
these docs only as needed:

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

## Validated Context

- Sequential Claude-dispatch base-tool proof is complete for OpenAI/GPT and
  Gemini through dev `:4001`. GPT-5.5 passed at
  `/tmp/claude_adapter_gpt55_child_sequential_core_tools_unique.json`; Gemini
  3.1 Pro passed after the Google Code Assist quota reset at
  `/tmp/claude_adapter_gemini31_pro_quota_reset_seq_parallel.json`;
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
  3 Flash passed the live parallel read-tool proof on dev `:4001` at
  `/tmp/claude_adapter_gemini3_flash_child_parallel_read_tools_rerun.json`, and
  Gemini 3.1 Pro passed the matching live parallel proof after quota reset at
  `/tmp/claude_adapter_gemini31_pro_quota_reset_seq_parallel.json`.

- GPT-5.5/OpenAI Claude-dispatched parallel tool calls are now validated on dev
  `:4001`. Keep the dead-end breadcrumb in [COMPLETED.md](COMPLETED.md): context
  compaction alone was insufficient; the passing repair also rewrites the
  adapted OpenAI Responses `instructions` to a compact function-calling policy
  when `parallel_tool_calls=true` and multiple function tools are present. Live
  artifact:
  `/tmp/claude_adapter_gpt55_child_parallel_read_tools_parallel_instruction_policy.json`.

- OpenRouter and NVIDIA `/anthropic` parallel read-tool proofs were validated
  historically on dev `:4001`. The old OpenRouter proof used
  `openrouter/inclusionai/ling-2.6-flash:free`, but that target is now retired
  from active harness config because OpenRouter reports Ling 2.6 Flash is no
  longer available as a free model. The earlier
  `openrouter/qwen/qwen3-coder:free` attempt hit provider 429 from Venice
  before adapter validation and should remain a dead-end breadcrumb, not a
  regression. NVIDIA proof uses
  `nvidia/deepseek-ai/deepseek-v3.2`; no separate completion-side parallel
  policy was needed because the model emitted the three tools together once the
  child trace-name metadata merge was fixed. Artifacts:
  `/tmp/claude_adapter_openrouter_ling_nvidia_parallel_read_tools.json` and
  `/tmp/claude_adapter_nvidia_parallel_read_tools_trace_fix.json`.

## Next

- Prod `aawm-litellm` on `:4000` is running the rebuilt aawm.37 image with
  `aawm-litellm-callbacks==0.0.12`, `aawm-litellm-control-plane==0.0.6`, and
  the `h-v0.0.21` harness. The stale `langfuse_trace_name:
  claude-code.orchestrator` child trace-name overwrite is resolved and proven in
  `/tmp/litellm-prod-aawm37-cb12-focused-no-openrouter.json`; do not redo that
  fix unless the same trace-header overwrite signature recurs.

- OpenRouter `inclusionai/ling-2.6-flash:free` is no longer a viable release
  gate or active target path. The focused dev rerun at
  `/tmp/litellm-dev-ling-26-flash-empty-success-rerun.json` now receives a live
  OpenRouter `404` saying `Ling-2.6-flash is no longer available as a free
  model`; the Ling smoke, OpenRouter-prefixed smoke, parallel child case, and
  mixed-fanout child were removed from active config. Do not spend fanout
  validation time on Ling unless the model is intentionally moved to a paid
  target and the agent file is restored. The old prod failure is still useful
  history: prod artifact
  `/tmp/litellm-prod-harness-aawm37-cb12.json` timed out because the
  `ling-2-6-flash` child never produced an assistant completion, and
  `session_history` recorded a zero-token OpenRouter Ling row for parent session
  `9db3bb66-6898-4257-a597-95090851414d`.

- The no-empty-response classifier is now in place for OpenRouter Responses:
  the adapter rejects empty successful OpenRouter Responses streams/non-stream
  bodies and logs a bounded raw event/body diagnostic, while the harness
  hard-fails successful empty command output when a selected case opts in.
  Next OpenRouter work should pick a currently available replacement free model
  for the parallel read-tool proof, then run only that focused case before
  reconsidering any peeromega fanout rerun.

- Keep the native Codex/Gemini repository-attribution gap as an explicit
  regression gate. The original issue was Codex CLI and Gemini CLI runs through
  LiteLLM not populating `public.session_history.repository`; the 2026-04-28
  fix added the top-level column, header propagation, metadata mirroring, and
  focused dev proof. Until the next prod/default validation proves it again,
  run only the focused native cases
  `native_openai_passthrough_responses_codex`,
  `native_gemini_passthrough_generate_content`, and
  `native_gemini_passthrough_stream_generate_content` when touching this path.
  They must keep requiring top-level `repository` containing `litellm` plus
  `metadata.repository`; relevant files are
  `litellm/integrations/aawm_agent_identity.py`,
  `.wheel-build/aawm_litellm_callbacks/agent_identity.py`,
  `scripts/local-ci/run_anthropic_adapter_acceptance.py`, and
  `scripts/local-ci/anthropic_adapter_config.json`.

- `/anthropic` streaming/tool-call parity audit follow-up: do not run the full
  multi-path harness for this work until the focused unit/path tests below pass.
  The implementation target is only traffic that enters LiteLLM through
  `/anthropic` and then fans out to OpenAI/Codex, Google/Gemini, NVIDIA, or
  OpenRouter. Native Codex CLI and Gemini CLI traffic is comparison evidence
  only: use it to capture provider-native envelopes, stream event ordering, and
  durable logging expectations, not as extra non-`/anthropic` implementation
  scope.

  OpenAI/Codex Responses adapter:
  - Preserve intra-assistant content ordering in
    `translate_messages_to_responses_input()` when a prior Anthropic assistant
    turn mixes text and `tool_use` blocks; add a regression with text before
    and after a tool call. Also decide whether prior Anthropic `thinking`
    history should be dropped, mapped to Responses reasoning items, or kept as
    encrypted reasoning metadata instead of visible assistant `output_text`.
  - Track raw MCP parity separately: the transformation layer can map
    `mcp_servers` / `mcp_toolset` into Responses `mcp` tools, but the proxy
    route rejects raw MCP for adapted `/anthropic` requests. Either de-scope the
    dead translation path or implement end-to-end parity with native OpenAI
    Responses passthrough.
  - Add one hard-gated parity fixture that uses a native Codex streaming tool
    call as the baseline and a `/anthropic`-adapted Claude Code tool call as
    the system under test, then compares reconstructed `response.output_item.*` /
    `response.function_call_arguments.*` state plus
    `session_history_tool_activity`.

  Google/Gemini Code Assist adapter:
  - Add Anthropic-to-Google streaming behavior checks for partial and parallel
    `functionCall` chunks. The adapter always sends upstream
    `v1internal:streamGenerateContent`; the Anthropic wrapper currently buffers
    Gemini tool-call deltas until a terminal chunk, while native Gemini/Google
    adapter code can emit a function call once accumulated JSON parses and flush
    leftovers at stream end.
  - Document and test the session/id contract against native Gemini CLI:
    native Gemini passthrough preserves CLI `request.session_id` as baseline
    behavior, while `/anthropic` to Google derives a model-scoped Code Assist
    session and hand-builds `user_prompt_id`. Add a golden-envelope comparison
    for the same prompt/session, especially across follow-up tool turns.
  - Add explicit alias-contract tests for every Claude core tool (`Read`,
    `Grep`, `Bash`, etc.) across tools, `tool_choice`, assistant tool calls,
    streaming restored names, and native Gemini `functionDeclaration` names.

  NVIDIA/OpenRouter completion adapters:
  - Preserve parallel streaming tool calls by tracking each upstream
    `delta.tool_calls[index]` as a separate Anthropic content block. The
    non-stream adapter emits one `tool_use` per OpenAI tool call, but the
    streaming translator currently starts from `tool_calls[0]` and concatenates
    all streamed argument deltas into one `input_json_delta`.
  - Delay the initial streaming `content_block_start` until the first upstream
    chunk identifies text, thinking, or tool output. The completion stream
    wrapper currently opens an empty text block before reading the first chunk,
    so tool-only responses can differ from native Anthropic and from the
    non-stream adapter.
  - Add an explicit hosted-tool compatibility policy for completion adapters:
    translate supported Anthropic hosted/beta tools to the target provider
    shape, or reject/drop unsupported tools with metadata explaining the
    downgrade.
  - Focused coverage to add before live harness work: one chunk containing two
    `delta.tool_calls` entries, a tool-only stream with no leading empty text
    block, and provider-specific hosted-tool translation/rejection assertions.

  OpenRouter Responses adapter:
  - Treat it as the OpenAI Responses parity path plus OpenRouter-specific
    routing/empty-success behavior for `/anthropic` traffic. There is no
    first-party OpenRouter CLI baseline, so compare request/stream
    reconstruction to the native OpenAI/Codex baseline and then run a single
    currently available free-model OpenRouter `/anthropic` case before any
    peeromega fanout rerun.

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

- Keep future harness bundle publishes ahead of `h-v0.0.21` for the current
  aawm.37 prod-validation line. `h-v0.0.21` is the minimum known-good bundle for
  the rebuilt `cb-v0.0.12` prod image: it includes controlled Claude trace
  `userId` validation, explicit per-run Claude settings overlay, longer
  peeromega fanout timeout, the narrow OpenRouter provider-unavailable timeout /
  command-failure classifier, the default-suite exclusion for GPT-OSS edge
  cases, and the focused prod child trace-name coverage used during the latest
  `:4000` validation.
