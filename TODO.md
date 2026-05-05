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

- Full monolithic `make test` is still not a clean local release gate. The
  2026-05-01 rerun collected `18172 items / 83 errors` after fixing the three
  documentation collection regressions from the earlier `86`-error run. Keep
  using focused unit/doc/harness gates for this release lane unless the broader
  suite is split or the remaining collection/setup blockers are intentionally
  addressed: duplicate test module basenames in one pytest invocation, missing
  optional deps (`PIL`, `google.genai`), missing live credentials, old proxy/live
  tests executing at import time, and the Vertex vector-store transformation
  import gap.

- D1-060 prompt-overhead tracking now has live `session_history` fields,
  translated-shape unit coverage, native Codex assertions, and
  `summary.prompt_overhead_cost_share` in the local harness artifact. Native
  live coverage on dev `:4001` has populated Claude/OpenAI/Codex/Gemini rows;
  keep the item open for adapted-route cost-share coverage across Anthropic ->
  OpenAI/Gemini/NVIDIA/OpenRouter and for a later exact input-cost field if the
  proportional `response_cost_usd` estimate is not enough.

- The `aawm.38`, `aawm.39`, and stale pre-publication `aawm.40` / `aawm.41` release
  candidates are superseded for current `develop`. The `aawm.38` image predates
  the local embed/rerank/Nomic routes and explicit `openrouter/*` Claude
  adapter routing; the `aawm.39` image predates explicit `nvidia/*` Claude
  adapter wildcard routing; and the `aawm.40` / `aawm.41` tags predate harness
  autobumps on current `main`. Do not promote those lines as the final cutover
  candidate for current code. The next prod candidate is
  `v1.82.3-aawm.42`, which should be rebuilt through infrastructure and
  validated on prod `:4000` only after deployment approval.

- Post-`aawm.38` config/code is now present on `develop`: current OpenRouter
  rerank/embedding catalog entries, NVIDIA NIM free endpoint rerank/embedding
  entries, local `local_embed/*` routes for MedCPT article/query, SPECTER2,
  Indus, SapBERT, Nomic code embeddings, and
  `local_rerank/BAAI/bge-reranker-v2-m3`. `cfg-v0.0.9` is published with the
  local-route cost-map entries, `h-v0.0.27` is the current harness overlay, and
  the next base image line is
  `ghcr.io/zepfu/litellm:1.82.3-aawm.42`. Before declaring prod cutover
  complete, rebuild infrastructure, recreate/restart `aawm-litellm` only after
  explicit deployment approval, then verify `NVIDIA_NIM_API_KEY` where direct
  NVIDIA NIM routes should be callable and verify local TEI/Nomic/rerank
  services are reachable from the running container via `host.docker.internal`.

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
  The replacement free-model parallel proof
  `claude_adapter_openrouter_nemotron_child_parallel_read_tools` using
  `nvidia/nemotron-3-super-120b-a12b:free` passed once on dev at
  `/tmp/claude_adapter_openrouter_nemotron_parallel_read_tools.json`; keep it
  as the focused OpenRouter parallel proof and use it before reconsidering any
  peeromega fanout rerun. Do not retry the attempted Poolside Laguna
  replacement without a separate Claude Code model-resolution fix; Claude Code
  rejected
  `openrouter/poolside/laguna-m.1:free` as unavailable/inaccessible before
  traffic reached the OpenRouter adapter.

- Keep native Codex/Gemini repository attribution as an explicit regression
  gate, not as currently open. The original issue was Codex CLI and Gemini CLI
  runs through LiteLLM not populating `public.session_history.repository`; the
  2026-04-28 fix added the top-level column, header propagation, metadata
  mirroring, and focused dev proof. A 2026-04-29 reporting-query follow-up
  clarified that `tenant_id` is the durable repository grouping field, so
  repository-only native rows must now also set `tenant_id` and
  `metadata.tenant_id` from the resolved repository when no explicit tenant is
  present. A 2026-04-29 real Codex session from
  `/home/zepfu/projects/aawm` showed the missing-harness-header gap still
  existed for normal Codex traffic; the follow-up fix infers repository from
  prepared `litellm_metadata` and workspace context text such as
  `AGENTS.md instructions for /path` / `<cwd>/path</cwd>`. A second 2026-04-29
  report from `mcp-pg` exposed the Gemini CLI shape
  `- **Workspace Directories:**\n  - /path`, plus the need to recursively honor
  structured workspace-root keys. Dev proofs now include Codex session
  `019dd8b8-c4f5-7c21-81bd-f4cab0715d6c` with `repository=aawm`, Codex session
  `019dd8d1-931c-7b81-8e81-a720f5df048c` with `repository=mcp-pg`, Gemini
  session `f52dd42a-ef02-4592-beef-ee9d81267778` with `repository=mcp-pg`, and
  focused Gemini native proof at
  `/tmp/native_gemini_repository_regression_after_inference.json`.
  Until the next prod/default validation proves it again, run only the focused
  native cases
  `native_openai_passthrough_responses_codex`,
  `native_gemini_passthrough_generate_content`, and
  `native_gemini_passthrough_stream_generate_content` when touching this path.
  They must keep requiring top-level `repository`, `metadata.repository`,
  fallback `tenant_id`, and `metadata.tenant_id` for repository-only rows, and
  should include non-harness-header Codex and Gemini workspace-text proofs;
  relevant files are
  `litellm/integrations/aawm_agent_identity.py`,
  `.wheel-build/aawm_litellm_callbacks/agent_identity.py`,
  `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`,
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
  After the focused NVIDIA and Gemini refinements, the broad dev default
  `/anthropic` harness passed at
  `/tmp/anthropic_full_dev_after_streaming_refinements.json`.

  OpenAI/Codex Responses adapter:
  - Codex stream/tool parity is now validated for the narrow focused pair.
    Keep
    `native_openai_passthrough_responses_codex_tool_activity,claude_adapter_codex_tool_activity`
    as the OpenAI/Codex gate before broad harness work. The latest passing
    artifact is `/tmp/anthropic_codex_tool_alias_after_fix.json`: upstream
    OpenAI/Codex sees native `exec_command` / `cmd` tool state while Claude
    Code still sees `Bash` / `command` on the Anthropic side, scoped only to the
    Codex-backed `/anthropic` route.
  - Completed locally, external full-harness gate blocked: strip
    Anthropic-only tool metadata such as
    `defer_loading`, `eager_input_streaming`, `allowed_callers`, and
    `input_examples` from provider tool schemas on `/anthropic` to
    non-Anthropic paths, while preserving enough adapter-side metadata for
    future work. The OpenAI-compatible chat path now strips these fields before
    forwarding provider tool schemas, and the OpenAI Responses path has an
    explicit regression test proving the fields are not forwarded. Focused
    local validation and a focused live failed-case rerun have passed. Broad
    dev `/anthropic` harness reruns at
    `/tmp/anthropic_tool_metadata_strip_full_dev.json` and
    `/tmp/anthropic_tool_metadata_strip_full_dev_rerun.json` are not clean
    because live Gemini 3.1 Pro child-agent requests in fanout hit upstream
    Google Code Assist `MODEL_CAPACITY_EXHAUSTED`; the second run otherwise
    cleared the OpenAI/Codex path and every non-fanout default case, and the
    standalone `claude_adapter_gemini31_pro` lane passed in the same broad run.
    Do not classify this as ordinary quota/capacity exhaustion without checking:
    the user's Google usage UI showed available Flash, Flash Lite, and Pro
    capacity. Next action is a focused fanout-only investigation comparing the
    failing child-agent Pro request against the standalone Pro request,
    including model, project, access-token lane, session/user-prompt ids,
    request shape, adapter semaphore/cooldown behavior, and any child-agent
    scheduling differences. Do not chase this as a schema-strip regression, do
    not run the full multi-path harness again until the focused Gemini fanout
    question is resolved, and do not start the broader deferred-search control
    loop as part of this cleanup.
    The focused repair now caches refreshed Gemini OAuth access tokens
    in-process and serializes Code Assist project/prime bootstrap so fanout
    children stay on the intended account/project lane. Focused local
    concurrency/retry tests passed, and the two previously failing live paths
    now pass on dev:
    `/tmp/gemini_fanout_after_oauth_lane_cache.json` and
    `/tmp/peeromega_fanout_after_oauth_lane_cache.json`. The first broad rerun
    at `/tmp/anthropic_full_dev_after_oauth_lane_cache.json` narrowed the
    remaining failure to repeated Pro model-capacity pacing in
    `claude_adapter_peeromega_fanout`; the preceding Gemini fanout and later
    standalone Pro case both passed. Dev now uses a longer explicit Code Assist
    capacity retry envelope in `docker-compose.dev.yml`. After recreating
    `litellm-dev` with the new environment, the focused peeromega rerun passed
    at `/tmp/peeromega_fanout_after_capacity_retry_budget.json` with the
    Gemini 3.1 Pro session-history and tool-activity rows present. Per the
    2026-04-29 user pause, do not make additional changes or run additional
    harness coverage in this session unless explicitly resumed. Remaining
    validation gap: the broad default dev `/anthropic` harness has not been
    rerun after the longer retry envelope; the latest broad artifact is still
    `/tmp/anthropic_full_dev_after_oauth_lane_cache.json`, which failed only
    `claude_adapter_peeromega_fanout` before the retry-envelope update.
  - Deferred/parked: adapter-owned support for Anthropic deferred tool loading
    on `/anthropic` to non-Anthropic paths. The preferred design is still a
    bounded internal tool-search loop: hold `defer_loading=true` tool
    definitions in an adapter registry, send upstream only normal tools plus a
    compact synthetic search tool, intercept synthetic search tool calls
    without streaming them to Claude Code, rank/select a small set of matching
    deferred tools, and issue the follow-up provider call with those full
    definitions expanded. Start with OpenAI Responses/Codex when this is
    resumed, then extend to Gemini, OpenRouter, and NVIDIA after the control
    loop, usage aggregation, and stream hiding behavior are stable. Focused
    tests must prove deferred tool schemas are absent from the first upstream
    request, selected deferred schemas appear only after internal search,
    internal search calls never surface as Anthropic `tool_use` blocks,
    explicit `tool_choice` for a deferred tool pre-expands that tool, and
    usage/session-history metadata can represent the multi-call provider turn
    without double-counting. Do not run the full multi-path harness until the
    unit and one-lane focused live proof pass.
  - Deferred/parked: revisit `eager_input_streaming`.
    For OpenAI/OpenRouter/NVIDIA chat-style streams we already forward upstream
    function-argument deltas as Anthropic `input_json_delta` events when the
    provider streams them. Gemini intentionally buffers parallel tool calls
    until terminal chunks, and the OpenAI Responses/Codex path currently buffers
    Codex `exec_command` and `Read` until valid JSON for alias/sanitizer
    correctness. Decide per tool/provider whether `eager_input_streaming=true`
    should bypass that buffering, and gate it with focused fixtures using
    large write/edit-style arguments before enabling it broadly.

  Google/Gemini Code Assist adapter:
  - Native Gemini CLI Code Assist envelope gates are now live-proven for both
    normal and stream-json CLI modes at
    `/tmp/native_gemini_payload_gates_after_config.json`: the captured provider
    request includes top-level `model`, `project`, `user_prompt_id`,
    `request.session_id`, `request.contents`, `request.systemInstruction`,
    `request.generationConfig.thinkingConfig`, and `request.tools`, with
    `gemini-2.5-flash`, `includeThoughts=true`, and `thinkingBudget=8192`.
    The focused `/anthropic` comparison is now live-proven by
    `claude_adapter_gemini3_flash_child_parallel_read_tools`, including the
    Code Assist envelope fields above plus native Gemini function declaration
    names. Unit coverage now pins the model-scoped Code Assist `session_id`,
    hand-built `user_prompt_id`, native function declaration aliases, full
    Claude-core native alias mapping, `tool_choice`, assistant tool-call
    aliases, restored streaming tool names, parallel tool-call buffering, and
    terminal usage preservation.
  - Track the live policy drift explicitly: native Gemini CLI defaults its
    thinking config to `thinkingBudget=8192`, while the current `/anthropic`
    Gemini adapter effort path has been observed using `thinkingLevel` for
    selected Gemini 3 cases. This is currently treated as intentional
    model/API behavior: the native capture is `gemini-2.5-flash`, while the
    `/anthropic` effort cases target `gemini-3-flash-preview`, and the shared
    Gemini mapper uses `thinkingLevel` rather than `thinkingBudget` for Gemini
    3. The harness now hard-gates that the Gemini 3 effort cases include
    `request.session_id`, `request.systemInstruction`, and `thinkingLevel`, and
    that they do not emit a Gemini 2-style `thinkingBudget`. Native Gemini
    tool-use capture is still only envelope evidence, not a first-party
    tool-call parity baseline.
  - If live Gemini captures show true partial `functionCall.args` fragments
    across valid Code Assist SSE events, add a focused fixture for that exact
    shape. Current focused coverage proves multiple function calls spread across
    Code Assist chunks; add a route-level raw-transport split fixture only if
    live captures make that shape relevant.
  - `claude_adapter_gemini3_flash_child_parallel_read_tools` now carries and
    passes the focused `/anthropic` Code Assist envelope comparison gate for
    model/project/user_prompt/session/systemInstruction/tools/thinkingConfig.
    Keep this as the Gemini tool-envelope proof before broad harness work.

  NVIDIA/OpenRouter completion adapters:
  - Run live focused `/anthropic` cases before broad harness work to confirm the
    new hosted-tool policy behaves correctly against NVIDIA/OpenRouter-style
    completion targets. Unit coverage now keeps Anthropic hosted/beta tools out
    of OpenAI-compatible `tools`, preserves web search as `web_search_options`,
    removes forced `tool_choice` values that target dropped hosted tools, and
    records unsupported hosted tool downgrades in metadata.
    The NVIDIA live gate now passes via
    `claude_adapter_nvidia_hosted_tool_policy`; keep the remaining unresolved
    work scoped to any future OpenRouter-style completion target that can
    exercise the same policy. Do not restore `request_payload_checks` for the
    NVIDIA case unless the harness gains transformed upstream-payload capture;
    the current Langfuse generation input does not expose NVIDIA completion
    `tools`, `web_search_options`, or adapter metadata reliably for that path.

  OpenRouter Responses adapter:
  - Treat it as the OpenAI Responses parity path plus OpenRouter-specific
    routing/empty-success behavior for `/anthropic` traffic. There is no
    first-party OpenRouter CLI baseline, so compare request/stream
    reconstruction to the native OpenAI/Codex baseline and then run
    `claude_adapter_openrouter_nemotron_child_parallel_read_tools` as the
    focused replacement proof before any peeromega fanout rerun.

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

- Keep the aawm.37 harness note historical and use `h-v0.0.27` for the prepared
  aawm.42 release candidate. `h-v0.0.21` is the minimum known-good released
  bundle for the rebuilt `cb-v0.0.12` prod image, while the repo-local harness
  source and current released candidate bundle are now `0.0.27`. The aawm.37
  bundle includes controlled Claude trace
  `userId` validation, explicit per-run Claude settings overlay, longer
  peeromega fanout timeout, the narrow OpenRouter provider-unavailable timeout /
  command-failure classifier, the default-suite exclusion for GPT-OSS edge
  cases, and the focused prod child trace-name coverage used during the latest
  `:4000` validation. The repo-local Anthropic adapter/native harness is broader
  than the standalone `h-v*` archive and now reports
  `summary.prompt_overhead_cost_share` for prompt-overhead analysis.
