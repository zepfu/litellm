# Completed

## 2026-04-28

- Promoted prod `aawm-litellm` on `:4000` to the rebuilt aawm.37 image with
  overlay artifacts `cb-v0.0.12`, `cp-v0.0.6`, and `h-v0.0.21`.
  Readiness passed with `litellm_version=1.82.3+aawm.37` and callbacks including
  `AawmAgentIdentity`; container package inspection reported
  `litellm=1.82.3+aawm.37`, `aawm-litellm-callbacks=0.0.12`, and
  `aawm-litellm-control-plane=0.0.6`. The focused prod harness artifact
  `/tmp/litellm-prod-aawm37-cb12-focused-no-openrouter.json` passed all selected
  GPT-5.5, Gemini 3.1 Pro, Gemini 3 Flash, and NVIDIA DeepSeek cases with zero
  failures and zero warnings. It also proved the child trace-name fix in prod:
  each child trace appeared as its `claude-code.harness-*` name and each
  expected trace carried tenant-only user id `adapter-harness-tenant`.

- Ran the default prod adapter harness after the focused prod pass. Artifact
  `/tmp/litellm-prod-harness-aawm37-cb12.json` is not a clean release pass:
  overall `passed=false` because `claude_adapter_peeromega_fanout` timed out
  after `420s` with no stdout/stderr, no response excerpt, and zero Langfuse
  traces. The same artifact passed the other default OpenAI, Gemini, Codex,
  GPT-5.5 read-pages, OpenRouter/NVIDIA Nemotron, and context-marker gates.
  OpenRouter free/Ling cases passed only as warning-only: they returned empty
  successful Claude CLI results with zero usage, missing generation token/cost
  telemetry, and missing `Bash` tool activity for Ling. The isolated OpenRouter
  parallel proof
  `/tmp/litellm-prod-aawm37-cb12-openrouter-parallel.json` also timed out after
  `300s` with no stdout/stderr and zero Langfuse traces, so classify that lane
  as provider/model no-response or latency until a deeper per-child transcript
  capture proves otherwise. Post-run prod log scan did not show ASGI/task,
  `KeyError: choices`, stale `Content-Length`/`h11`, or quota-pressure
  blockers; expected `LITELLM_MASTER_KEY` startup warning and provider attempt
  warnings were present.

- Classified the `claude_adapter_peeromega_fanout` timeout down to a specific
  child lane. Parent Claude session
  `9db3bb66-6898-4257-a597-95090851414d` launched all eight requested agents;
  the parent transcript's last assistant message at `2026-04-28T22:10:06Z`
  says it was still waiting on `ling-2-6-flash`. The Ling child transcript
  `agent-a71b22a70294d7082.jsonl` has only the user prompt and injected context,
  with no assistant message or completion notification. Prod `session_history`
  still recorded the OpenRouter Ling request for that session from
  `2026-04-28T22:09:15.779Z` to `22:09:16.416Z`, but it had
  `input_tokens=0`, `output_tokens=0`, `tool_call_count=0`, and cost `0`.
  Conclusion: the mega fanout timed out because OpenRouter Ling returned an
  empty successful response that never became a completed Claude Code child, not
  because all fanout lanes were slow.

- Fixed the prod aawm.37 focused-harness child trace-name blocker found after
  restarting `aawm-litellm` on `:4000`. Root cause: `/anthropic` child-agent
  metadata correctly set `trace_name=claude-code.<agent>` and
  `trace_user_id=adapter-harness-tenant`, but Langfuse later re-read the stale
  inbound `langfuse_trace_name: claude-code.orchestrator` request header and
  overwrote the child trace name. `AawmAgentIdentity` now rewrites stale
  `claude-code*` Langfuse trace-name headers to the child trace name, mirroring
  the existing trace-user-id header correction. The callback overlay source in
  `.wheel-build/aawm_litellm_callbacks/agent_identity.py` was synced with the
  in-repo callback source and the callback artifact version was bumped to
  `0.0.12`. The Gemini 3 Flash sequential harness case now checks the returned
  final marker as a substring because the prod artifact proved all eight tools
  completed while Claude wrapped the child final line. Local validation passed:
  `test_aawm_agent_identity.py -k "child_dispatch_trace_metadata or stale_orchestrator"`
  (`2 passed`),
  `test_llm_pass_through_endpoints.py -k "child_identity or child_trace_context or overrides_orchestrator"`
  (`3 passed`),
  `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py` (`41 passed`),
  JSON validation for `scripts/local-ci/anthropic_adapter_config.json`, and
  `py_compile` for both callback source copies. Prod follow-up validation with
  the rebuilt `cb-v0.0.12` / `h-v0.0.21` image proved this trace-name fix in the
  focused prod harness; remaining prod failures are now peeromega/OpenRouter
  provider-lane behavior, not this stale-header overwrite.

- Re-verified the aawm.37 prod-prep state after the terminal crash without
  restarting `aawm-litellm`. Re-read [PROD_RELEASE.md](PROD_RELEASE.md) and
  confirmed the remaining cutover step is intentionally deferred until explicit
  approval. GitHub release checks confirmed `v1.82.3-aawm.37`, `cb-v0.0.11`,
  `cp-v0.0.6`, `h-v0.0.20`, and the floated `cfg-v0.0.7` model-config asset
  are published; the `aawm-publish.yml` workflow for `v1.82.3-aawm.37` is
  `completed/success`. Infra `develop` is still clean and pinned to
  `ghcr.io/zepfu/litellm:1.82.3-aawm.37`; local `aawm-litellm:latest` image
  inspection still reports `litellm=1.82.3+aawm.37`,
  `aawm-litellm-callbacks=0.0.11`, and
  `aawm-litellm-control-plane=0.0.6`. The running prod container remained
  untouched: `aawm-litellm:155693327495:Up 3 days (healthy)`.

- Prepared the validated adapter/harness state for the next prod cutover without
  restarting `aawm-litellm`. LiteLLM `develop` and `main` were fast-forwarded to
  the validated aawm.37 release head, the artifact autobump advanced callback
  to `cb-v0.0.11`, control-plane to `cp-v0.0.6`, and harness to `h-v0.0.20`,
  and the missing GitHub Release assets were published for all three overlay
  tags. The fork image workflow passed for `v1.82.3-aawm.37`, publishing
  `ghcr.io/zepfu/litellm:1.82.3-aawm.37`. In
  `/home/zepfu/projects/aawm-infrastructure`, `develop` was updated and pushed
  at `d727d34` to pin `Dockerfile.litellm` and
  `docker-compose.litellm.yml` to the aawm.37 image. A local infrastructure
  build completed without running `up -d`; built image inspection reported
  `litellm=1.82.3+aawm.37`, `aawm-litellm-callbacks=0.0.11`, and
  `aawm-litellm-control-plane=0.0.6`. The running prod container remained
  untouched and was still `aawm-litellm:latest` up for 3 days/healthy during
  prep. Actual `:4000` restart and prod harness validation are deferred until
  explicit approval.

- Closed the Gemini 3.1 Pro quota-reset validation loop on dev `:4001`.
  Added the default-excluded
  `claude_adapter_gemini31_pro_child_parallel_read_tools` harness case and
  reran it with the existing
  `claude_adapter_gemini31_pro_child_sequential_core_tools` case after Google
  Code Assist capacity reset. Live artifact:
  `/tmp/claude_adapter_gemini31_pro_quota_reset_seq_parallel.json`. Both cases
  passed with zero failures and zero warnings. Sequential proof recorded final
  `GEMINI31 PRO SEQUENTIAL TOOLS PASSED`, trace
  `claude-code.harness-gemini31-pro-sequential-core-tools`, tenant-only user id
  `adapter-harness-tenant`, exact ordered `Read`, `Write`, `Edit`, `Glob`,
  `Grep`, `Bash`, `WebSearch`, and `WebFetch` transcript tools with
  `max_tool_uses_in_single_assistant_message=1`, durable native Gemini
  `read_file`, `write_file`, `replace`, `glob`, `grep_search`,
  `run_shell_command`, `google_web_search`, and `web_fetch` activity, and no
  transcript tool-result errors. Parallel proof recorded final
  `GEMINI31 PRO PARALLEL TOOLS PASSED`, trace
  `claude-code.harness-gemini31-pro-parallel-read-tools`, native Gemini
  `read_file`/`glob`/`grep_search` rows, and
  `max_tool_uses_in_single_assistant_message=3`. The harness runtime log guard
  found no live `Content block not found` or `Invalid pages parameter` blockers
  in the checked run window.

- Added the first half of the expanded Claude-dispatch base-tool harness:
  default-excluded sequential child-agent cases for GPT-5.5 and Gemini 3.1 Pro,
  plus transcript-level validation of Claude Code child `.jsonl` files. The new
  `transcript_tool_use_validation` block locates subagent transcripts by the
  command `session_id`, filters by child agent type, counts
  `message.content[].type == "tool_use"` by tool name and assistant
  `message.id`, and can require sequential proof with
  `maximum_tool_uses_per_assistant_message=1`. The live cases are intentionally
  not parallel tests yet; they require sequential `Read`, `Write`, `Edit`,
  `Glob`, `Grep`, `Bash`, `WebSearch`, and `WebFetch` first.

- Unblocked the planned Gemini sequential web-tool proof by broadening the
  Google Code Assist follow-up tool allowlist from the six file/shell tools to
  the full base set, adding `WebSearch` and `WebFetch` plus their native aliases
  (`google_web_search`, `web_fetch`). `WebFetch` and Gemini `replace` are now
  classified correctly in `session_history_tool_activity` as read and modify
  activity respectively, with the callback overlay source kept in parity.
  Focused checks passed:
  `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
  (`31 passed`),
  `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k "trims_followup_google_tools_to_core_set" -q`
  (`2 passed`), and
  `./.venv/bin/python -m json.tool scripts/local-ci/anthropic_adapter_config.json`.

- Ran the first live GPT-5.5 sequential base-tool proof on `:4001` and captured
  the exact remaining blockers in
  `/tmp/claude_adapter_gpt55_child_sequential_core_tools.json`. The child did
  complete with `GPT55 SEQUENTIAL TOOLS PASSED` and durable tool activity
  included every required base tool, but the harness failed because
  `session_history_validation` incorrectly required file read/write counters on
  a single multi-turn response row and because `WebSearch` retried after
  Claude Code emitted `allowed_domains: []` / `blocked_domains: []`, which
  Anthropic rejects as ambiguous. The sequential cases now rely on
  `tool_activity_validation` for per-tool proof, and the Anthropic request
  preparation path sanitizes empty web-search domain lists to `null` while
  tagging the request with
  `claude-web-search-domain-filter-sanitized`.

- Reran the GPT-5.5 sequential proof after deploying the web-search sanitizer.
  Artifact
  `/tmp/claude_adapter_gpt55_child_sequential_core_tools_rerun.json` confirmed
  `WebSearch` ran once and the request carried
  `claude-web-search-domain-filter-sanitized`; there were no new web-search
  empty-domain 400s. The run still failed correctly because the reused
  `/tmp/gpt55-sequential-tool-probe.txt` already existed, so Claude Code's
  `Write` tool returned `File has not been read yet` and GPT had to issue an
  extra `Read` plus retry `Write`. The transcript also showed `WebFetch` hit a
  403 on the OpenAI docs URL. The harness/config now uses per-run unique temp
  probe paths and can reject transcript `tool_result` errors, with the
  sequential probes switched to `https://example.com/` for a stable fetch target.

- Passed the GPT-5.5 sequential base-tool proof on dev `:4001` with the unique
  temp path and stricter transcript checks. Artifact:
  `/tmp/claude_adapter_gpt55_child_sequential_core_tools_unique.json`. The
  transcript shows exactly one each of `Read`, `Write`, `Edit`, `Glob`, `Grep`,
  `Bash`, `WebSearch`, and `WebFetch`, with
  `max_tool_uses_in_single_assistant_message=1` and no `tool_result` errors.
  Langfuse/session validation passed, durable tool activity included all base
  tools, and the sanitizer tag
  `claude-web-search-domain-filter-sanitized` was present.

- Ran the Gemini 3.1 Pro sequential base-tool proof after GPT passed. Artifact:
  `/tmp/claude_adapter_gemini31_pro_child_sequential_core_tools_unique.json`.
  The child never reached tool execution: Claude Code returned a 429
  `RESOURCE_EXHAUSTED` / `QUOTA_EXHAUSTED` from Google Code Assist for
  `gemini-3.1-pro-preview`, with reset text around
  `2026-04-28T13:08:24Z`; the transcript for
  `gemini-3-1-pro-preview` had zero `tool_use` blocks. Added a separate
  default-excluded `claude_adapter_gemini3_flash_child_sequential_core_tools`
  case so the Gemini adapter/base-tool path can still be proven on an available
  Gemini model without marking the Pro quota failure as a tool-path result.

- Ran the Gemini 3 Flash sequential base-tool proof twice and classified the
  remaining failure as post-tool-result malformed tool syntax, not a missing
  harness gate. The first Flash run exercised all eight base tools and had no
  transcript tool-result errors, but failed strict prompt/final-output checks:
  it searched LiteLLM/Gemini docs instead of the required `IANA example domain`
  query and returned a verbose final answer. The stricter rerun failed after one
  valid Claude-visible `Read`; the child then emitted plain text
  `Calling tool Write. Carvercall:default_api:write_file{...` with
  `stop_reason=end_turn` instead of a second `tool_use`. Artifact:
  `/tmp/claude_adapter_gemini3_flash_child_sequential_core_tools_strict.json`;
  transcript:
  `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/bafed1b3-89cf-42e1-a46c-879929986ab4/subagents/agent-ad7916f2f4f6fbfd6.jsonl`.
  Do not rerun the same Flash proof unchanged; next work is to isolate whether
  the bad `Carvercall` text comes from model behavior, the Gemini adapter
  response conversion, or Claude Code's streamed tool-call parsing after a tool
  result.

- Identified and patched the Claude-dispatched Gemini parallel tool-call stream
  collapse. Live evidence: aawm-tap Langfuse trace
  `d0275abf-2e5f-4964-b875-74159927acb1` recorded a single Gemini upstream
  generation with five native tool calls (`read_file` x4 plus
  `run_shell_command`), but Claude Code transcript
  `/home/zepfu/.claude/projects/-home-zepfu-projects-aawm-tap/b0428c41-09d8-40ec-b2b5-4d87ffdb40dd/subagents/agent-ab792fb928eb0da98.jsonl`
  only received one `Read` block for that same assistant turn and then logged
  `API Error: Content block not found`. The Anthropic streaming wrapper now
  buffers Gemini streaming tool-call deltas until the terminal tool-call chunk
  and emits one Claude-visible `tool_use` block per Gemini tool call, preserving
  ids, mapped tool names, and JSON arguments. Focused validation passed with
  `./.venv/bin/python -m pytest tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_parallel_tool_calls.py -q`
  (`5 passed` after adding both pre-terminal and terminal parallel-call
  regression shapes).

- Broadened validation for the Gemini stream patch and deployed it to dev.
  Focused test runs passed:
  `tests/test_litellm/llms/anthropic/experimental_pass_through/adapters/test_anthropic_experimental_pass_through_adapters_transformation.py`
  (`63 passed`), and
  `tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_parallel_tool_calls.py`
  plus
  `tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_sse_wrapper.py`
  plus
  `tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_content_after_stop_reason.py`
  (`19 passed`). Restarted `litellm-dev`; `curl -sS http://127.0.0.1:4001/health`
  returned `healthy_count=3` and `unhealthy_count=0`.

- Classified the later Gemini v11 continuation failure as a separate upstream
  Google Code Assist capacity/quota error, not the same adapter stream bug. The
  same child agent `ab792fb928eb0da98` later returned
  `RESOURCE_EXHAUSTED` / `QUOTA_EXHAUSTED` from
  `cloudcode-pa.googleapis.com` with reset timestamp
  `2026-04-28T13:08:24Z`. The user may still see remaining Gemini UI usage
  because this error is from the Cloud Code / Code Assist quota bucket used by
  the Claude-dispatch adapter path.

- Fixed the live GPT-5.5 Claude-dispatch report-file refusal. Smoking gun:
  aawm-tap trace `0f1f34a2-c892-4a2f-87ef-e8dd4687bbcf` put the old Claude
  Code no-write instruction in a plain string `instructions` field, but the
  runtime control-plane patcher only rewrote `{"type":"text","text":...}`
  blocks. `aawm_claude_control_plane.py` now applies manifest patches to plain
  strings too and adds a pattern fallback for the rendered `Write` form and the
  `${...}` template form of the report-file instruction. Focused tests passed:
  `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k "claude_prompt_patches or report_file_instruction" -q`
  (`3 passed`) and
  `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
  (`28 passed`).

- Added and passed a focused live dev harness gate for the exact failure mode:
  `claude_adapter_gpt55_child_analysis_write_probe`, default-excluded, validates
  real Claude Code dispatch through `:4001` to GPT-5.5, the patched request text,
  absence of the old no-write sentence, tenant trace ids, session-history, and
  child `Bash` plus `Write` tool activity. Live artifact:
  `/tmp/claude_adapter_gpt55_child_analysis_write_probe_rerun.json`. The run
  passed with no failures/warnings; `session_id=9483c5ae-c37c-412f-a076-ebfbbdbf6ec5`,
  `file_modified_count=1`, GPT-5.5 `tool_name=Write`, and
  `/tmp/gpt55-analysis-write-probe.md` contained
  `write probe passed: 2026-04-28T03:55:21.606747634Z`.

- Fixed a harness lookup bug exposed by the write probe. Name-based trace
  lookup was not constrained by the harness user when
  `expected_trace_user_ids_by_name` was configured, so overlapping historical
  `claude-code.*` traces from another session could produce false
  `sessionId mismatch` failures. The harness now resolves a lookup user from
  `expected_user_ids` or a single common per-trace expected user id before
  polling by trace name, with regression tests covering both paths and the
  mixed-user no-guess path.

- Removed a false failure source from the sequential tool harness by filtering
  Anthropic `/anthropic/v1/messages/count_tokens` observations out of
  completion-generation and session-history route checks. This prevented
  count-token traces with null model/cost from being mistaken for the child
  model completion in live sequential runs.

- Classified the latest Gemini 3 Flash sequential rerun after the synthetic
  `Calling tool ...` context fix and neutral fixture change. The old live
  `Carvercall` malformed text did not recur, and the child reached the first
  four tools correctly, but the normal global `gemini-3-flash-preview` agent
  still drifted: invalid `Grep(file_path=...)`, retry `Grep`, extra
  `Read`/`Bash`/`Edit`, and no `WebSearch`/`WebFetch`. Artifact:
  `/tmp/claude_adapter_gemini3_flash_child_sequential_core_tools_neutral_fixture.json`.
  Next validation should use a case-local harness agent with only the eight
  base tools rather than the normal coding/investigation agent.

- Converted the sequential base-tool cases to case-local harness agents and
  caught a Claude Code flag trap before continuing. A live Gemini 3 Flash run
  with `--tools Agent` dispatched the custom child but gave it no base tool
  declarations; the child returned a Python snippet and the request metadata
  showed `usage_tool_call_count=0`. Trace
  `7c0f2ff0-8c6b-420d-91aa-019416f9f522`, artifact
  `/tmp/claude_adapter_gemini3_flash_child_sequential_core_tools_harness_agent.json`,
  transcript
  `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/3abda483-a70c-4b46-a8bd-3d49ac83710c/subagents/agent-a4e1afdbf072999e5.jsonl`.
  The harness now uses `--allowedTools Agent` for the parent, preserving the
  built-in tool registry for the child while still forcing parent dispatch.
  Focused validation passed:
  `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
  (`39 passed`).

- Reran the Gemini 3 Flash sequential base-tool proof with the corrected
  parent flag, `--allowedTools Agent`. This restored child tool availability:
  the custom harness agent executed the first six tools in exact order
  (`Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`) with no zero-tool registry
  failure. The run still failed, now at the real remaining boundary: after the
  `Bash` date command, Gemini produced final text instead of continuing to
  `WebSearch` and `WebFetch`. Artifact:
  `/tmp/claude_adapter_gemini3_flash_child_sequential_core_tools_allowed_tools_agent.json`;
  transcript:
  `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/4f7ee447-3c96-4b44-b71f-b3cfe212f5e5/subagents/agent-aca1b5abcf7e0f51c.jsonl`.

- Tried parent and child prompt hardening for the Gemini 3 Flash post-Bash
  stop and classified it as a dead end. Focused config validation still passed
  (`39 passed`), but the live rerun again stopped after `Read`, `Write`,
  `Edit`, `Glob`, `Grep`, and `Bash`; the child returned only the Bash
  timestamp. Artifact:
  `/tmp/claude_adapter_gemini3_flash_child_sequential_core_tools_after_web_guardrail.json`;
  transcript:
  `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/8e120b2a-6910-4b6a-9fd9-b5199027369e/subagents/agent-a25c33e8f798ab2ee.jsonl`.
  Trace inspection showed `google_web_search` and `web_fetch` were declared in
  the post-Bash request, so this was not tool availability or schema loss. The
  request text had lost the original numbered task because the Google
  completion-message window kept only the recent tail; fallback text context
  carried only the latest `Grep` and `Bash` tool results.

- Classified the first Gemini 3 Flash live rerun after adding preserved active
  task state. The child again executed `Read`, `Write`, `Edit`, `Glob`, `Grep`,
  and `Bash` in order, but the post-Bash request failed with a proxy 500 before
  `WebSearch`. Artifact:
  `/tmp/claude_adapter_gemini3_flash_child_sequential_core_tools_after_task_state_retention.json`;
  transcript:
  `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/cb4e5338-f3f5-4af4-bc58-c569c7e0ab90/subagents/agent-a394ea6c1467d0f19.jsonl`.
  The LiteLLM stack trace was
  `Missing corresponding tool call for tool response message`: inserting the
  preserved task reminder reduced the retained tail to `max_window - 1`, and
  that tail began with a `tool` result whose matching assistant `tool_calls`
  message had been trimmed away.

- Patched the Google completion-message window retention logic to preserve a
  valid assistant/tool-result pair boundary while still keeping the active task
  reminder. Added regression coverage for the orphan-tool-result shape and
  reran the focused pass-through tests:
  `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k "completion_message_window or preserves_task_state or orphan_tool_result or fallback_text_context or fallback_excludes_synthetic" -q`
  (`4 passed`).

- Passed the Gemini 3 Flash sequential base-tool proof on dev `:4001` after
  deploying the valid tool-pair boundary fix. Artifact:
  `/tmp/claude_adapter_gemini3_flash_child_sequential_core_tools_after_tool_pair_boundary.json`;
  transcript:
  `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/6a386d08-8a2a-4dd6-abe9-27b84ee95d3e/subagents/agent-a79dba772e8299f95.jsonl`.
  The child emitted exactly eight Claude-visible tool calls, one per assistant
  message: `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, `WebSearch`, and
  `WebFetch`. The post-Bash turn correctly called `WebSearch` with
  `IANA example domain`, `WebFetch` fetched `https://example.com/`, the final
  result was `GEMINI3 FLASH SEQUENTIAL TOOLS PASSED`, transcript validation
  showed `max_tool_uses_in_single_assistant_message=1` and no tool-result
  errors, Langfuse user ids were tenant-only (`adapter-harness-tenant`), durable
  tool activity included the required tools, and runtime logs had no forbidden
  substrings.

- Added the first parallel read-tool harness gates for Claude-dispatched
  OpenAI/GPT and Gemini children. The default-excluded cases require `Read`,
  `Glob`, and `Grep` to appear in one Claude-visible assistant message, enforce
  exactly three child tool uses, and require matching durable
  `session_history_tool_activity` rows. Harness config validation passed in
  `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`, including the
  new `minimum_tools_in_single_assistant_message` transcript validator shape.

- Ran the initial live parallel read-tool proofs on dev `:4001`. Gemini 3 Flash
  passed after loosening the parent command JSON check to `required_contains`
  for the final line. Artifact:
  `/tmp/claude_adapter_gemini3_flash_child_parallel_read_tools_rerun.json`;
  transcript:
  `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/951409fd-1786-476b-94e4-ab0db4af5e61/subagents/agent-a853e8ba3c302886b.jsonl`.
  The child emitted `Read`, `Glob`, and `Grep` under one assistant
  `message.id` and the run passed with zero failures/warnings. GPT-5.5 failed
  the same proof at
  `/tmp/claude_adapter_gpt55_child_parallel_read_tools.json`: it emitted
  `Read`, waited for the result, then `Glob`, waited, then `Grep`, so transcript
  validation failed with `max=1` tool_use block in a single assistant message.
  Treat this as an active OpenAI/Responses parallel-tool request translation gap,
  not a missing Gemini stream fix.

- Patched the Anthropic-to-OpenAI Responses adapter to send
  `parallel_tool_calls=true` when translated requests contain only function
  tools and Anthropic did not set `disable_parallel_tool_use`. It still sends
  `false` when Anthropic explicitly disables parallel use and leaves the field
  unset for built-in or mixed Responses tools. Focused tests passed:
  `./.venv/bin/python -m pytest tests/test_litellm/llms/anthropic/experimental_pass_through/responses_adapters/test_responses_adapters_transformation.py::TestTranslateParallelToolCallsToResponsesAPI tests/test_litellm/llms/anthropic/experimental_pass_through/responses_adapters/test_responses_adapters_transformation.py::TestTranslateRequestBroaderCoverage::test_disable_parallel_tool_use_sets_parallel_tool_calls_false tests/test_litellm/llms/anthropic/experimental_pass_through/responses_adapters/test_responses_adapters_transformation.py::TestTranslateRequestBroaderCoverage::test_function_tools_set_parallel_tool_calls_true_without_tool_choice tests/test_litellm/llms/anthropic/experimental_pass_through/responses_adapters/test_responses_adapters_transformation.py::TestTranslateRequestBroaderCoverage::test_function_tools_set_parallel_tool_calls_true_with_auto_tool_choice tests/test_litellm/llms/anthropic/experimental_pass_through/responses_adapters/test_responses_adapters_transformation.py::TestTranslateRequestBroaderCoverage::test_disable_parallel_tool_use_not_set_for_built_in_tools tests/test_litellm/llms/anthropic/experimental_pass_through/responses_adapters/test_responses_adapters_transformation.py::TestTranslateRequestBroaderCoverage::test_no_optional_fields_does_not_add_spurious_keys -q`
  (`11 passed`). The broader transformation-file run was stopped after hanging
  before the relevant assertions.

- Reran the GPT-5.5 parallel read-tool proof after restarting `litellm-dev`.
  Artifact:
  `/tmp/claude_adapter_gpt55_child_parallel_read_tools_rerun.json`; transcript:
  `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/7a30aa90-9f0a-4aa0-bc7d-e0a9256e1b0b/subagents/agent-a6504af3f9f1ef52f.jsonl`.
  Langfuse confirmed the OpenAI Responses payload now included
  `parallel_tool_calls=true` with `Read`, `Glob`, and `Grep` function tools, and
  the harness now hard-gates that request field. GPT-5.5 still serialized the
  tool calls into three assistant messages (`max_tool_uses_in_single_assistant_message=1`),
  so the remaining GPT gap is no longer the missing request flag.

- Narrowed the remaining GPT-5.5 parallel gap with direct Responses probes
  through `:4001`. A simple direct GPT-5.5 Responses call with
  `parallel_tool_calls=true` emitted three function calls in one response. A
  direct call using the exact logged `Read`/`Glob`/`Grep` schemas plus the exact
  child task also emitted all three function calls, even with the Claude Code
  `instructions` string included. Reusing the full logged Claude Code input
  shape, which had three user content blocks (about 8 KB subagent context,
  36 KB project/Claude context, then the 858 byte task), made GPT-5.5 emit only
  `Read`. Replacing that real context with dummy context of similar size still
  parallelized. Current classification: GPT-5.5 can parallelize via the
  Responses backend; the remaining issue is semantic interference from the real
  injected Claude Code context for adapted OpenAI subagents.

- Tried the first OpenAI Responses adapter context-shaping repair for GPT-5.5
  parallel tool calls and classified it as insufficient by itself. The adapter
  now compacts large Claude Code `SubagentStart` / `# claudeMd` reminder blocks
  before Anthropic-to-Responses translation, preserves the actual child task,
  and records `openai-adapter-claude-context-compacted` metadata/tags plus an
  `openai_adapter.claude_context_compaction` span. Focused checks passed:
  `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k "OpenAIAdapterClaudeContextCompaction or trims_followup_google_tools_to_core_set or openai_responses_adapter_preserves_agent_project_litellm_metadata" -q`
  (`6 passed`), `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
  (`41 passed`), JSON config validation, and
  `./.venv/bin/ruff check --select E9,F821,F823 ...`. Full Ruff on the touched
  files still reports pre-existing unrelated lint debt. Live dev artifact
  `/tmp/claude_adapter_gpt55_child_parallel_read_tools_context_shaped.json`
  proved compaction ran (`43112` original context chars to `1418`, first GPT
  request `2237` input tokens, request-payload checks passed), but GPT-5.5
  still serialized `Read`, `Glob`, and `Grep` into three assistant messages
  (`max_tool_uses_in_single_assistant_message=1`). Do not repeat "compact
  Claude Code context only" as the next fix.

- Closed the GPT-5.5/OpenAI Claude-dispatched parallel tool-call gap on dev
  `:4001`. Exact compacted-payload probes showed the remaining blocker was the
  full Claude Code `instructions` string: the exact compacted payload still
  serialized, while a compact OpenAI Responses function-calling instruction
  emitted `Read`, `Glob`, and `Grep` together in repeated direct probes. The
  adapter now applies that instruction policy after Anthropic-to-Responses
  translation only when `parallel_tool_calls=true` and multiple function tools
  are present, and records
  `openai-adapter-parallel-instruction-policy`,
  `openai-adapter-parallel-tool:<tool>`, and
  `openai_adapter.parallel_instruction_policy` metadata/spans. Focused checks
  passed: `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k "OpenAIAdapterClaudeContextCompaction or applies_openai_parallel_instruction_policy or skips_openai_parallel_instruction_policy or trims_followup_google_tools_to_core_set or openai_responses_adapter_preserves_agent_project_litellm_metadata" -q`
  (`10 passed`), `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
  (`41 passed`), JSON config validation, and
  `./.venv/bin/ruff check --select E9,F821,F823 ...`. Live dev harness passed:
  `/tmp/claude_adapter_gpt55_child_parallel_read_tools_parallel_instruction_policy.json`.
  The request payload checks found
  `litellm_metadata.openai_adapter_parallel_instruction_policy_applied=true`,
  Langfuse trace names were `claude-code.orchestrator` and
  `claude-code.harness-gpt55-parallel-read-tools` with user id
  `adapter-harness-tenant`, runtime logs had no forbidden errors, transcript
  validation saw one assistant message containing `Read`, `Glob`, and `Grep`
  (`max_tool_uses_in_single_assistant_message=3`), and durable tool activity
  recorded the three OpenAI `responses.output` rows at the same timestamp.

- Started the OpenRouter/NVIDIA `/anthropic` parallel read-tool extension.
  Focused code/config checks passed before the first live run:
  `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k "parallel_instruction_policy or completion_parallel_tool_policy or OpenAIAdapterClaudeContextCompaction" -q`
  (`10 passed` at that point), `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
  (`41 passed`), JSON validation, and the Ruff E9/F821/F823 gate. Live artifact
  `/tmp/claude_adapter_openrouter_nvidia_parallel_read_tools.json` split the
  failure modes: OpenRouter/Qwen never reached adapter proof because
  `qwen/qwen3-coder:free` returned provider 429s from Venice on all four
  upstream attempts; NVIDIA/DeepSeek did emit `Read`, `Glob`, and `Grep` in one
  child assistant message (`max_tool_uses_in_single_assistant_message=3`) and
  durable `nvidia_nim` `Read`/`Glob`/`Grep` rows, but the harness correctly
  failed because Langfuse still named the child trace as
  `claude-code.orchestrator`. Do not repeat the OpenRouter/Qwen free-provider
  attempt as adapter validation; use a different available OpenRouter lane.

- Closed the OpenRouter and NVIDIA `/anthropic` parallel read-tool proof on dev
  `:4001`. The OpenRouter Responses route now applies the same compact
  function-calling instruction policy with OpenRouter-specific
  `openrouter-adapter-*` metadata and compacts Claude context before
  Anthropic-to-Responses translation. The passing OpenRouter artifact is
  `/tmp/claude_adapter_openrouter_ling_nvidia_parallel_read_tools.json` using
  `openrouter/inclusionai/ling-2.6-flash:free`; it recorded trace names
  `claude-code.orchestrator` and
  `claude-code.harness-openrouter-ling-parallel-read-tools`, user id
  `adapter-harness-tenant`, request payload gates for
  `parallel_tool_calls=true`,
  `openrouter_adapter_claude_context_compacted=true`, and
  `openrouter_adapter_parallel_instruction_policy_applied=true`, transcript
  `Read`/`Glob`/`Grep` in one assistant message
  (`max_tool_uses_in_single_assistant_message=3`), and durable OpenRouter
  `Read`/`Glob`/`Grep` rows at the same timestamp. The passing NVIDIA artifact
  is `/tmp/claude_adapter_nvidia_parallel_read_tools_trace_fix.json` using
  `nvidia/deepseek-ai/deepseek-v3.2`; it recorded trace names
  `claude-code.orchestrator` and
  `claude-code.harness-nvidia-deepseek-parallel-read-tools`, user id
  `adapter-harness-tenant`, transcript `Read`/`Glob`/`Grep` in one assistant
  message (`max_tool_uses_in_single_assistant_message=3`), and durable
  `nvidia_nim` `Read`/`Glob`/`Grep` rows at the same timestamp. The NVIDIA
  fix was not a model prompting change; completion-adapter metadata now lets
  child `litellm_metadata.trace_name` override stale orchestrator
  `metadata.trace_name` when building the actual completion call.

## 2026-04-27

- Closed the latest `aawm-tap` Claude-dispatched Gemini/GPT-5.5 investigation
  with concrete trace evidence. Gemini 3.1 Pro did not fail in the latest v10
  review; it wrote
  `/home/zepfu/projects/aawm-tap/.analysis/plan-phase01-remediation-v10-cc-gemini3.1pro.md`
  at `2026-04-28T01:58:56Z` after a long model turn before the `Write` call.
  GPT-5.5 did batch multiple `Read` calls, so its slow review was not caused by
  Claude Code forcing all tools to run serially; the long spans were large model
  synthesis turns through the Claude Code Anthropic-compatible child-agent path.
  Native Codex CLI and Gemini CLI speed remains separate evidence: we have not
  built Codex-CLI/Gemini-CLI translation between those CLIs, and the problematic
  path was specifically Claude Code dispatch through LiteLLM's Anthropic
  adapter.
  A transcript sweep for session `b0428c41-09d8-40ec-b2b5-4d87ffdb40dd`
  found no latest-subagent interrupt marker. Gemini had a recoverable
  `Content block not found` API marker but still completed and wrote the file;
  do not treat that marker as the failure root cause unless it recurs with an
  actual missing final/write.

- Fixed the dev-runtime cause of GPT-5.5 still seeing the old Claude Code
  report-file block. The running `litellm-dev` container's
  `/app/context-replacement/claude-code/prompt-patches/roman01la-2026-04-02.json`
  was stale and lacked `subagent-report-file-explicit-request`, even though the
  repo file had it. `docker-compose.dev.yml` now bind-mounts
  `./context-replacement` into `/app/context-replacement`, and the container was
  recreated so the prompt-patch cache was cleared. Direct container validation
  confirmed the exact old no-report-files sentence rewrites to the explicit
  request form and records `subagent-report-file-explicit-request`.

- Strengthened the live GPT-5.5 child Bash harness gate to validate the prompt
  patch as part of the same real Claude dispatch. The initial live run proved
  the product path was fixed but exposed a harness filter gap: the OpenAI child
  trace had route tags and `passthrough_route_family` but no
  `user_api_key_request_route`, so generation filtering dropped the child before
  trace-tag/request-text checks. The harness now accepts route-family tags for
  allowed-route matching, with a regression test covering this shape. Focused
  validation passed with
  `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
  (`25 passed`) and
  `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k "claude_prompt_patches" -q`
  (`2 passed`). Live dev validation passed at
  `/tmp/claude_adapter_gpt55_child_bash_prompt_patch_rerun.json`: `claude-code.gpt-5-5`
  has `userId=adapter-harness-tenant`, OpenAI/GPT-5.5 route tags, the
  `claude-prompt-patch:subagent-report-file-explicit-request` tag, and exactly
  one persisted `Bash` row for `date -u +%Y-%m-%dT%H:%M:%S.%NZ`.

- Corrected the temporary `project.agent` trace-user interpretation. The
  actual convention is `trace.name=claude-code.<agent>` and Langfuse
  `userId=<tenant_id>` only; `aawm_claude_project` remains separate project
  context and must not be folded into the Langfuse user id. The metadata helper
  now sets `trace_user_id` from the explicit tenant identity, and the AAWM
  logging hook rewrites `langfuse_trace_user_id` to that tenant value for child
  traces so Langfuse headers cannot override it back to a synthetic harness
  user. Live dev `claude_adapter_gemini_fanout` passed at
  `/tmp/claude_adapter_gemini_fanout_after_tenant_userid_fix.json` for session
  `78f60054-fef4-4764-8614-000966d91cc6`; direct Langfuse verification showed
  `claude-code.orchestrator` and all three `claude-code.gemini-*` traces with
  `userId=adapter-harness-tenant`.

- Verified the same tenant-only trace-user behavior for OpenAI models. The
  direct GPT-5.5 harness case now expects tenant-only user ids and passed at
  `/tmp/claude_adapter_gpt55_after_tenant_userid_harness_fix.json`; direct
  Langfuse verification for session `bf0c814c-8521-487c-b0cc-fe676416a9b2`
  showed `claude-code.orchestrator` with `userId=adapter-harness-tenant`. Added
  the focused default-excluded `claude_adapter_gpt55_child_bash_identity` case
  so OpenAI child dispatch can be hard-gated with a real Bash tool call. Live
  dev validation passed at `/tmp/claude_adapter_gpt55_child_bash_identity.json`
  for session `2e83ca42-df97-4a19-823b-4075482dca29`; Langfuse showed
  `claude-code.gpt-5-5` with `userId=adapter-harness-tenant`, and
  `session_history_tool_activity` recorded `provider=openai`,
  `model=gpt-5.5`, `tool_name=Bash`, command
  `date -u +%Y-%m-%dT%H:%M:%S.%NZ`.

- Implemented source-side Claude Code child-model observability preservation
  for Anthropic-adapted routes. `_prepare_anthropic_request_body_for_passthrough`
  now derives Claude child agent/project context before provider translation and
  stores `agent_name`, `aawm_claude_agent_name`, `tenant_id`, `aawm_tenant_id`,
  `aawm_claude_project`, `trace_name=claude-code.<agent>`, and
  `trace_user_id=<tenant_id>` in `litellm_metadata`; completion-adapter
  metadata mirrors `trace_user_id`; and `AawmAgentIdentity` now prefers those
  explicit metadata keys before scanning prompt text. Focused validation passed
  for OpenAI Responses, OpenRouter Responses, Google/Gemini completion, NVIDIA
  completion, and callback/session-history identity preservation.

- Strengthened the Anthropic adapter harness so smoke cases can prove tool use,
  not just plausible final text. `command_json_checks` now supports
  `required_regex`, `tool_activity_validation` supports `maximum_count` and
  forbidden command substrings, Ling/OpenRouter smoke prompts now require a
  high-resolution UTC `Bash` timestamp command, and
  `claude_adapter_gemini_fanout` now requires native Gemini
  `run_shell_command` rows for all three Gemini children. The later live dev
  artifact `/tmp/claude_adapter_gemini_fanout_final.json` confirmed
  `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, and
  `gemini-3.1-flash-lite-preview` all persisted the exact
  `date -u +%Y-%m-%dT%H:%M:%S.%NZ` command. The later tenant and Langfuse
  user-attribution issues were closed separately; do not reclassify this as
  missing Gemini Bash/tool activity.

- Fixed and validated the Gemini Code Assist wrapped-stream logging gap that
  made earlier fanout runs look like Flash/Flash-lite had not used Bash. The
  Gemini passthrough logger now assembles multi-chunk Code Assist streams
  instead of transforming only the final wrapped `response` chunk, preserving
  non-final native `functionCall` chunks for Langfuse and
  `session_history_tool_activity`. Focused tests passed with
  `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
  (`38 passed`), and direct dev DB checks for session
  `fe80c0ed-e1a7-4f59-81c3-8742327932fd` show all three Gemini
  `run_shell_command` rows.

- Tightened the harness session-history diagnostics for expected-row failures.
  When provider/model rows exist but fail a secondary predicate, such as the
  current Gemini fanout `tenant_id` mismatch, artifacts now include
  `all_records` and failure messages include candidate rows with the mismatched
  fields. Focused validation now passes with `39 passed` for the Gemini
  passthrough logging and local-ci harness tests.

- Fixed the Claude Code child `session_history` tenant mismatch exactly. The
  dev container was already importing the bind-mounted local source; the
  mismatch was caused by `_add_claude_child_agent_observability_metadata()`
  writing `litellm_metadata.tenant_id` from the Claude project text
  (`litellm`) before `AawmAgentIdentity` considered the harness
  `x-aawm-tenant-id` header. The preparation path now feeds the explicit
  tenant header into child observability metadata, while preserving the Claude
  project separately as `aawm_claude_project`. After recreating `litellm-dev`,
  live `claude_adapter_gemini_fanout` passed at
  `/tmp/claude_adapter_gemini_fanout_after_tenant_fix.json` with zero failures:
  all three Gemini child `session_history` rows have
  `tenant_id=adapter-harness-tenant`, Langfuse child traces retain names such
  as `claude-code.gemini-3-flash-preview`, and all three native
  `run_shell_command` rows persist.

- Closed the remaining Claude Code Gemini 3.1 Pro post-tool-result dispatch
  failure on dev `:4001`. The v10 aawm-tap transcript at
  `/home/zepfu/.claude/projects/-home-zepfu-projects-aawm-tap/b0428c41-09d8-40ec-b2b5-4d87ffdb40dd/subagents/agent-ae56c557397dcc651.jsonl`
  showed Gemini ran one `Bash` call, then Claude Code recorded only a
  thinking block with `output_tokens=0` and `stop_reason=end_turn`. Langfuse
  for the same window showed prompt policy `2026-04-27.v2` was present and
  Gemini had emitted follow-up `read_file` tool calls, so the remaining issue
  was not the direct `Read.pages` / malformed-id case. The Anthropic stream
  wrapper now keeps per-instance stream state instead of sharing the class
  `chunk_queue`, preserves terminal chunks that carry both `finish_reason` and
  tool-call deltas, and suppresses Gemini provider thinking/signature deltas so
  Claude Code receives the actionable `tool_use` blocks.

- Added and validated the post-tool-result Gemini harness gate.
  `claude_adapter_gemini31_pro_bash_then_read_stream_state` is default-excluded
  and forces real Claude Code on `:4001` through `Bash` then `Read`, validating
  Langfuse, `session_history`, and native Gemini tool activity
  (`run_shell_command` and `read_file`). Live dev validation passed at
  `/tmp/gemini_bash_then_read_stream_state.json` with zero failures/warnings.
  The older direct `Read` gate still passed after the streaming fix at
  `/tmp/gemini_read_tool_id_sanitizer_after_stream_fix.json`.

- Corrected the GPT-5.5 Claude-dispatched write-file refusal root cause. The
  v10 GPT subagent transcript
  `/home/zepfu/.claude/projects/-home-zepfu-projects-aawm-tap/b0428c41-09d8-40ec-b2b5-4d87ffdb40dd/subagents/agent-a05c7877ad61b70ca.jsonl`
  line 1 contains the explicit request to write
  `.analysis/plan-phase01-remediation-v10-cc-gpt5.5.md`, line 4 says it will
  return validation inline instead, and line 112's Claude Code continuation
  summary repeats the higher-priority instruction. The decisive source check is
  Langfuse trace `4adee2dc-ac8f-43c8-ac41-484dec7a324b`: the GPT-5.5 upstream
  OpenAI Responses request already contained the no-write sentence in
  `instructions`. `strings` on Claude Code binary
  `/home/zepfu/.local/share/claude/versions/2.1.119` shows the sentence in the
  built-in background/subagent system prompt. LiteLLM did not author the
  sentence, but the OpenAI translation path did preserve it when converting the
  incoming Claude `system` prompt into OpenAI `instructions`; the prompt-patch
  metadata was adjacent and only changed the code-snippet guidance.

- Mitigated the Claude Code subagent report-file block through the existing
  prompt-patch path instead of editing the Claude binary. The new
  `subagent-report-file-explicit-request` patch rewrites the rendered sentence
  to allow report/summary/findings/analysis `.md` writes when the user
  explicitly asks for them, while still requiring the model to return findings
  in its final assistant message. Focused validation passed with
  `./.venv/bin/python -m json.tool context-replacement/claude-code/prompt-patches/roman01la-2026-04-02.json`
  and
  `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k test_prepare_anthropic_request_body_applies_claude_prompt_patches -q`
  (`1 passed, 220 deselected`).

- Closed the Gemini Claude Code direct-`Read` failure on the dev adapter.
  The failing transcript showed upstream Gemini produced a real native
  `read_file` call, but the Gemini thought signature had been embedded into the
  Anthropic `tool_use.id`; Claude Code then surfaced only `Calling tool Read.`
  and did not execute the read. The Anthropic adapter now strips `__thought__`
  suffixes from tool-use ids while preserving the signature in
  `provider_specific_fields`, drops synthetic `Calling tool ...` text when a
  real tool call is present, and keeps the existing empty `Read.pages`
  sanitizer. The Google adapter system prompt policy is now
  `2026-04-27.v2` and explicitly requires visible final assistant text after
  tool results, which fixed the follow-up Gemini thinking-only final turn seen
  in the first live validation.

- Added and validated the targeted Gemini Read harness gate.
  `claude_adapter_gemini31_pro_read_tool_id_sanitizer` is default-excluded but
  hard-fails when selected. It validates the real Claude Code path through
  `:4001`, requires final result `# TODO`, checks prompt-policy v2 metadata,
  validates the matching Gemini session row with `tool_call_count=1` and
  `file_read_count=1`, and matches persisted tool activity as native
  `read_file`. Live dev validation passed at
  `/tmp/gemini_read_tool_id_sanitizer_v2.json` for session
  `4a08eeda-4aa6-40ca-9394-fec5dfa501f0` with result `# TODO`, one persisted
  `read_file` row, and zero failures/warnings. The earlier
  `claude_adapter_gemini_fanout` case remains useful only for child-agent
  fanout/session-history coverage and should not be treated as proof that
  Gemini direct Read/tool-use completion works.

- Confirmed the GPT-5.5 review write-file behavior is expected for Claude Code
  background/subagent sessions unless the OpenAI adapter strips or overrides
  that Claude-native no-report-files instruction for non-Claude models.

- Implemented the Google-only Gemini system prompt policy for Claude-initiated
  `/anthropic` requests routed through `anthropic_google_completion_adapter`.
  The Google request builder now rewrites canonicalized system messages before
  Gemini `systemInstruction` generation, supports `off`, `append`, and
  `replace_compact` policy modes through
  `AAWM_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY`, preserves project/safety/operator
  instructions under a clear heading, keeps Gemini native tool aliasing intact,
  and records prompt-policy summary metadata without logging full prompt text.

- Normalized invalid empty Claude `Read.pages` tool arguments on the
  non-Anthropic `/anthropic` adapter response path. The shared sanitizer removes
  empty `pages` only for restored Claude `Read` tool calls, preserves unrelated
  empty-string arguments, covers non-streaming responses plus complete streaming
  JSON deltas, and respects tool-name mappings such as Gemini `read_file` back
  to Claude `Read`.

- Extended focused harness coverage for the adapter hardening. The Anthropic
  adapter harness now has a GPT-5.5 `Read` sanitizer case, supports forbidden
  substrings in persisted tool-activity arguments, and validates Gemini prompt
  policy markers/metadata on the existing Gemini adapter lane. Focused unit
  validation passed with `303 passed` across the touched adapter, proxy, SSE,
  and local-ci test files; the harness JSON config parses successfully.

## 2026-04-25

- Exposed OpenRouter-hosted embedding and rerank models through the LiteLLM proxy surface.
  `openrouter/qwen/qwen3-embedding-8b` is now available through `/v1/embeddings`
  with OpenRouter provider routing support for DeepInfra pinning, and
  `openrouter/cohere/rerank-4-pro` is available through `/rerank` / `/v1/rerank`
  using OpenRouter's `/api/v1/rerank` endpoint while preserving Cohere-style
  request/response shapes. The dev config exposes both models, the dev compose
  mounts the required OpenRouter and callback files, and consumers authenticate
  only to LiteLLM while the proxy uses the configured OpenRouter key upstream.

- Completed OpenRouter embedding/rerank cost and `session_history` attribution coverage.
  The primary and bundled model cost maps now include Qwen3 Embedding 8B with
  4096-dimensional vectors, 32k context, and `$0.01 / 1M` input-token pricing,
  plus Cohere Rerank 4 Pro with `$0.0025` per search unit. `AawmAgentIdentity`
  now preserves OpenRouter usage/cost/provider metadata, uses hidden upstream
  OpenRouter response cost when present, falls back to the bundled local model
  map when the runtime remote map is stale, prefers caller-facing
  `openrouter/...` proxy model names for OpenRouter rows, and persists
  estimated prompt/total tokens for rerank calls even when OpenRouter reports
  only `search_units`.

- Added OpenRouter embedding/rerank consumer documentation and validation.
  [OPENROUTER_EMBED_RERANK_CONSUMER.md](OPENROUTER_EMBED_RERANK_CONSUMER.md)
  documents the LiteLLM endpoints, required LiteLLM auth, attribution headers,
  DeepInfra provider routing, and session-history expectations. Focused tests
  passed for OpenRouter rerank transformation, AAWM session-history behavior,
  and exact Qwen/rerank cost-map coverage. Live dev proxy validation on `:4001`
  passed for session `or-embed-rerank-20260425-3`: the embedding row stored
  `provider=openrouter`, `model=openrouter/qwen/qwen3-embedding-8b`,
  `tenant_id=tenant-openrouter-validation`, `input_tokens=18`,
  `response_cost_usd=1.8e-07`, and `openrouter_provider=DeepInfra`; the rerank
  row stored `model=openrouter/cohere/rerank-4-pro`, `input_tokens=40`,
  `total_tokens=40`, `response_cost_usd=0.0025`, `usage_search_units=1`,
  `usage_openrouter_cost=0.0025`, `openrouter_provider=Cohere`, and
  `openrouter_response_model=rerank-v4.0-pro`.

- Finished the NVIDIA Anthropic-adapter lane and validation closure.
  The completion-adapter metadata bridge now mirrors NVIDIA route family, adapter model/original model, target endpoint, Langfuse spans, and route tags from `litellm_metadata` into normal completion `metadata`, so downstream Langfuse and `session_history` paths see the same adapter context as native passthrough lanes; adapter-owned metadata now also overwrites stale/conflicting caller metadata while preserving ordinary caller trace fields. The NVIDIA MiniMax model cost map now uses the closest OpenRouter MiniMax M2.5 pricing as its fallback basis instead of leaving durable cost tracking unmapped, and both primary/bundled model cost maps have non-zero NVIDIA adapter pricing coverage. The NVIDIA harness cases now require the `anthropic.nvidia_completion_adapter` span, adapter metadata, tags, and non-zero `response_cost_usd`; the focused dev shard passed at `/tmp/litellm-dev-nvidia-final-escalated.json` for cache-control strip, DeepSeek V3.2, and GLM 4.7.

- Hardened PostgreSQL connection usage for heavy adapter harness runs.
  The existing harness now reuses a per-target psycopg validation connection for `session_history` and `session_history_tool_activity` checks instead of opening short-lived connections for every validator. `AawmAgentIdentity` now writes through a bounded asyncpg pool per worker event loop/DSN, keeps the worker loop reusable, closes pools on worker shutdown, and bounds overflow flushing with a retry-before-drop path when the queue is saturated. Callback source parity was preserved between `litellm/integrations/aawm_agent_identity.py` and `.wheel-build/aawm_litellm_callbacks/agent_identity.py`, and `WHEEL.md` documents `AAWM_SESSION_HISTORY_POOL_MAX_SIZE`.

- Validated the NVIDIA and PostgreSQL pressure hardening work on dev/prod targets.
  Focused unit suites passed for AAWM session-history pooling, local-ci harness DB reuse, NVIDIA cost-map coverage, and NVIDIA pass-through metadata assertions. The dev NVIDIA shard passed with zero failures/warnings, and dev/prod OpenRouter canary shards passed with only warning-only optional Gemma notes. The full default dev harness at `/tmp/litellm-dev-default-final.json` and the latest full default prod harness both hard-failed only on upstream Codex/OpenAI `usage_limit_reached` 429s in the Codex/Spark path; the dev run reported reset time `2026-04-25 20:27:55 UTC`. Recent-window dev/prod log scans found no PostgreSQL pressure signatures (`too many clients`, failed AAWM flush, queue full, or FATAL); only the expected OpenAI/Codex 429 runtime blocker appeared in the dev window.

- Repaired `public.session_history` costs after the GPT-5.5/model-cost updates.
  The host-side backfill initially imported LiteLLM's remote `main` model cost map, which diverged from the deployed local/promoted map for GPT-5.5. `scripts/backfill_session_history.py` now defaults historical repairs to `LITELLM_LOCAL_MODEL_COST_MAP=True` unless explicitly overridden, avoids the ChatGPT auth-backed cost path for stored `provider=chatgpt` rows by pricing them through the static OpenAI model key, reconstructs Anthropic 1-hour cache-write details, and normalizes older rows where cache counters exceed stored `input_tokens`. The corrected repair passes updated 6,445 rows for local-map pricing, 22,045 rows for Anthropic 1-hour cache writes, and 6,355 rows for historical cache-token total normalization. The final full-table dry-run scanned 48,047 rows with `cost_updates=0`. Post-repair checks show 43,273 rows marked with `metadata.response_cost_source=session_history_repair`, zero token-bearing rows with null `response_cost_usd`, zero negative costs, and 4,250 GPT-5.5 rows with populated local-map cost data.

- Promoted the production `:4000` LiteLLM container to the `aawm.35` base image and completed release validation.
  The missing overlay GitHub Releases were published from existing tags for callback `cb-v0.0.9`, model config `cfg-v0.0.6`, and harness `h-v0.0.17`; the prod image was rebuilt with cache busting and verified to install `aawm-litellm-callbacks 0.0.9`, `aawm-litellm-control-plane 0.0.5`, and a model config containing GPT-5.5 pricing. Prod readiness is healthy on `:4000`. The final default prod harness passed at `/tmp/litellm-prod-harness-aawm35-default-final.json` with zero failures and warning-only Gemini/OpenRouter canary notes; explicit prod shards passed for native passthrough at `/tmp/litellm-prod-harness-aawm35-native-openai-rerun.json`, OpenAI/Gemini effort/cache at `/tmp/litellm-prod-harness-aawm35-effort-cache-openai-gemini-rerun.json`, and OpenRouter effort/cache at `/tmp/litellm-prod-harness-aawm35-effort-cache-openrouter.json`.

- Hardened the release process after the `aawm.35` prod cutover.
  `PROD_RELEASE.md` now calls out that overlay git tags are insufficient unless matching GitHub Release assets exist, that `--no-cache` rebuilds are required after publishing missing overlay releases, that built images should be inspected before restarting prod, and that direct `/openai_passthrough/*` prod validation requires `OPENAI_API_KEY` mapped into the container. The infrastructure compose file now maps `OPENAI_API_KEY` from `AAWM_OPENAI_API_KEY`, matching the dev runtime behavior.

- Removed static session ids from opt-in effort/cache harness cases.
  The OpenAI effort, Gemini effort, and OpenRouter effort/cache cases now use the existing generated per-run session id path, avoiding false prod failures caused by older dev `public.session_history` rows sharing the same static session id. The affected prod shard passed cleanly after the change.

- Published the follow-up harness artifact from the post-release autobump.
  Pushing the release-doc and harness-session cleanup to `main` auto-bumped the harness archive to `h-v0.0.18`; the missing GitHub Release asset was published and verified at `h-v0.0.18` with `litellm-local-ci-harness-0.0.18.tar.gz`.

- Broadened live harness coverage for the completed `/anthropic` effort/cache translation work.
  Added default-excluded existing-harness cases for Gemini minimal/max effort with cache-control variants, OpenRouter max/none/no-effort with cache-control variants, and an ordered OpenAI two-pass prompt-cache case. The harness now supports repeated HTTP passes, compact repeated-text fixtures, env-expanded HTTP headers, multi-row session-history minimums, and provider cache hit validation without introducing a second harness. Live dev shard artifacts passed with zero failures/warnings at `/tmp/anthropic-effort-cache-gemini-dev.json`, `/tmp/anthropic-effort-cache-openrouter-dev.json`, and `/tmp/anthropic-effort-cache-openai-dev.json`.

- Fixed OpenAI Responses adapter cache-intent persistence for Anthropic `cache_control`.
  The OpenAI Responses adapter now derives bounded `prompt_cache_key` values from Anthropic cache-control material, preserves route-level `litellm_metadata` while adding adapter cache metadata, and `public.session_history.metadata` now persists `openai_prompt_cache_key_present` plus `anthropic_adapter_cache_control_present`. The OpenAI two-pass live shard validates two rows for the generated session and requires a provider cache hit with cached input tokens on the warmed request.

- Validated the broadened effort/cache harness work on dev.
  Focused tests passed for local-ci harness hardening (`14 passed`), AAWM cache/session-history coverage (`9 passed / 55 deselected`), adapter/proxy cache transformation coverage (`28 passed / 350 deselected`), JSON/compile checks, and callback source parity. The full default dev Anthropic adapter harness passed at `/tmp/anthropic-adapter-dev-full-after-effort-cache.json` with no failures; the only warnings were the configured warning-only Gemini canary output/usage checks for `claude_adapter_gemini31_pro` and `claude_adapter_gemini31_flash`.

## 2026-04-24

- Purged the 24 operator-approved ambiguous historical Gemini `unknown` zero-token rows from local `aawm_tristore`.
  The cleanup was deliberately scoped to old dev harness sessions for `native_gemini_passthrough_stream_generate_content` between `2026-04-24 16:21:36+00` and `2026-04-24 16:25:16+00`, with `provider=gemini`, `model=unknown`, zero input/output tokens, and `litellm_environment=dev`. No matching `session_history_tool_activity` rows existed, the delete returned 24 rows, and the post-delete verification count for `gemini/unknown` zero-token rows is zero.

- Extended the existing Anthropic adapter harness tenant assertions across native and adapted lanes.
  The harness now carries a shared `default_tenant_id`, injects `x-aawm-tenant-id` through HTTP, Claude CLI, Codex CLI, and Gemini CLI context setup, and defaults `public.session_history.tenant_id`, `metadata.tenant_id`, and `metadata.tenant_id_source` assertions for every session-history case. Multi-row fanout validations now merge `required_equals.tenant_id` into each expected row. Live dev validation passed for the native Anthropic, OpenAI chat, OpenAI Responses, Codex Responses, Gemini generateContent, and Gemini streamGenerateContent cases with zero failures and zero warnings.

- Broadened opt-in `/anthropic` effort/cache harness coverage.
  Added default-excluded HTTP cases for Gemini `output_config.effort` translation and OpenRouter effort plus cache-control intent. The final opt-in dev bundle passed for `claude_adapter_openai_output_config_effort`, `claude_adapter_gemini_output_config_effort`, `claude_adapter_openrouter_output_config_effort_cache`, and `claude_adapter_nvidia_cache_control_strip` with zero failures and zero warnings. The OpenRouter case validates durable session-history reasoning/cache metadata and row-level provider-cache status; deeper outbound request-payload assertions remain pending until that lane exposes a stable parsed request shape.

- Added a safe historical Gemini control-plane repair mode.
  `scripts/backfill_session_history.py --repair-session-history --repair-gemini-control-plane {delete,mark}` now dry-runs by default, honors the existing request/session/trace/provider/model/time/limit filters, uses id-keyset pagination, deletes tool-activity rows before session rows when applying deletes, and only matches rows with explicit `loadCodeAssist`, `listExperiments`, `retrieveUserQuota`, or `fetchAdminControls` evidence. Local dry-run against `aawm_tristore` matched zero explicit-method rows; the 24 ambiguous `gemini/unknown` zero-token rows without method evidence were intentionally left untouched.

- Validated the updated harness/backfill package on dev.
  Focused tests passed for backfill repair helpers (`6 passed`), adapter/provider/session-history unit coverage (`79 passed` and `24 passed / 138 deselected` in the focused suites), JSON/compile checks passed, `git diff --check` passed, the full default Anthropic adapter harness on `:4001` passed with zero failures, native passthrough tenant validation passed with zero failures/warnings, and the new opt-in effort/cache bundle passed with zero failures/warnings.

- Completed the Claude `/anthropic` reasoning-effort and cache-translation implementation across OpenAI/Codex, Gemini, OpenRouter, and NVIDIA lanes.
  Added a shared Anthropic adapter normalization helper for `output_config.effort`, direct `reasoning_effort`, and `thinking` budgets; OpenAI Responses now preserves `xhigh` when model metadata supports it, Gemini receives normalized `reasoning_effort`, OpenRouter and NVIDIA use config-driven capability checks, and unsupported raw Anthropic fields are stripped before provider egress. Cache intent from Anthropic `cache_control` is preserved as normalized metadata, OpenAI derives bounded `prompt_cache_key` values only when cache intent is present, OpenRouter native cache-control support is metadata-driven, and NVIDIA records an explicit miss state instead of forwarding unsupported cache controls.

- Extended session-history and harness validation for the new adapter effort/cache paths.
  `public.session_history.metadata` now persists normalized reasoning-effort fields, NVIDIA is included in provider-cache family handling, and the existing Anthropic adapter harness now supports `required_equals`, `required_one_of`, `forbidden_paths`, and tenant-id assertions. Focused live dev validation on `:4001` passed for `claude_adapter_openai_output_config_effort` and `claude_adapter_nvidia_cache_control_strip` with zero failures and zero warnings after installing the changed helper files into `litellm-dev`.

- Updated GPT-5.5 pricing coverage in the model cost maps.
  `gpt-5.5` and `chatgpt/gpt-5.5` are present in both the primary and bundled fallback model cost maps with `$5.00 / 1M` input tokens, `$0.50 / 1M` cached input tokens, and `$25.00 / 1M` output tokens. Focused session-history coverage confirms GPT-5.5 cost calculation now produces the corrected output-token cost.

- Preserved explicit tenant identity into `public.session_history`.
  `AawmAgentIdentity` now resolves tenant ids from canonical metadata and request headers such as `tenant_id`, `aawm_tenant_id`, `user_api_key_org_id`, org/team aliases, and AAWM/LiteLLM tenant headers before falling back to prompt-text project extraction. Live and Langfuse/backfill records persist both `tenant_id` and `metadata.tenant_id_source`, and spend-log reconstruction now restores stored `proxy_server_request.headers` so historical header-derived tenants can be repaired.

- Enhanced and ran the existing session-history backfill script for tenant and cost repair.
  `scripts/backfill_session_history.py --repair-session-history` can now repair existing rows in place, with `--repair-costs`, `--repair-tenant-ids`, and existing request/session/trace/provider/model/time filters. Local repair updated GPT-5.5 costs for 2,579 rows across scoped apply passes, and filled 1,306 missing `tenant_id` values while synchronizing tenant metadata on 13,323 rows.

- Completed native passthrough logging parity for Anthropic, OpenAI/Codex, and Gemini.
  Added a shared passthrough callback contract that preserves model/provider/cost fields, request headers, existing metadata, passthrough payloads, provider-native usage objects, and downstream `AawmAgentIdentity` derivability while keeping provider-specific parsing in the Anthropic, OpenAI, and Gemini handlers. OpenAI/Codex Responses and chat, Anthropic Messages, and Gemini `generateContent` / `streamGenerateContent` now all feed callbacks/session-history with consistent kwargs and metadata.

- Added native passthrough validation to the existing Anthropic adapter harness.
  `scripts/local-ci/run_anthropic_adapter_acceptance.py` and `anthropic_adapter_config.json` now include selectable native HTTP and real-CLI cases for Anthropic, OpenAI/Codex, and Gemini without introducing a separate harness entrypoint. The CLI cases launch the actual `claude`, `codex`, and `gemini` binaries so local auth/config remains the source of truth, and the harness validates Langfuse plus `public.session_history` for the actual provider session id emitted by each client.

- Fixed Gemini native CLI control-plane calls creating `gemini/unknown` session-history rows.
  Gemini Code Assist startup/admin calls such as `loadCodeAssist`, `listExperiments`, `retrieveUserQuota`, and `fetchAdminControls` now skip model-call logging in both non-streaming and streaming passthrough paths. Only real Gemini model operations (`generateContent`, `streamGenerateContent`, and `predictLongRunning`) produce model `session_history` rows. Live dev validation after the fix produced one `gemini/gemini-2.5-flash` row for the Gemini CLI session and zero new `gemini/unknown` rows.

- Validated the native passthrough logging work and pushed it to `develop`.
  The committed change `1168a12e74 feat: align native passthrough logging` passed the shared passthrough/provider/session-history test set (`125 passed`), the broader passthrough endpoint suite (`212 passed`), compile checks, `git diff --check`, callback source parity, and the combined real-CLI harness on `:4001` for `native_anthropic_passthrough_claude`, `native_openai_passthrough_responses_codex`, and `native_gemini_passthrough_stream_generate_content`.

- Finalized the production release process documentation.
  Added `PROD_RELEASE.md` with the dev/prod runtime split, pre-promotion validation, fork image tagging, overlay artifact handling, infrastructure promotion, prod harness validation, optional provider lanes, finalization, and rollback process. Updated `TEST_HARNESS.md`, `WHEEL.md`, and the bundled local-ci README so the prod process is discoverable and GPT-OSS 20B/120B are documented as explicit opt-in edge checks rather than default hard gates.

- Promoted the local prod `:4000` LiteLLM container to the `aawm.34` image and completed validation.
  Prod readiness reported LiteLLM `1.82.3+aawm.34`; focused `claude_adapter_openrouter_ling_26_flash`, `claude_adapter_peeromega_fanout`, NVIDIA GLM, and NVIDIA DeepSeek passed. `claude_adapter_gpt_oss_120b` hit the exact OpenRouter provider-unavailable signature (`503 provider=OpenInference raw=no healthy upstream`) and was classified as a soft warning during isolated validation.

- Cleaned up the default prod adapter suite after OpenRouter edge-model noise.
  `claude_adapter_gpt_oss_20b` and `claude_adapter_gpt_oss_120b` remain available as explicit `--cases` selections, but are excluded from the default suite because repeated OpenRouter provider-unavailable responses made default promotion runs noisy and slow. Harness `0.0.14` carries the updated defaults, and the clean default prod harness pass at `/tmp/litellm-prod-harness-aawm34-no-gptoss.json` completed with zero failures and zero warnings.

- Captured the next promotion hardening requirement.
  Future prod cutovers should include a production-style preflight against the exact built image / installed wheel path on `:4001`, plus a small explicit promotion-gate set for opt-in provider lanes, before touching the `:4000` container.

- Advanced the harness artifact to `h-v0.0.13` and prepared the follow-up `aawm.34` image tag.
  The `aawm.33` main push correctly triggered the harness artifact autobump because the prod hardening changed harness code. Since the image publisher requires fork image tags to point at current `main`, `aawm.34` carries the same NVIDIA metadata-preservation fix plus the `h-v0.0.13` artifact bump.

- Promoted the local prod `:4000` LiteLLM container to the `aawm.32` image and isolated the remaining NVIDIA blocker.
  Prod readiness reported LiteLLM `1.82.3+aawm.32`; `claude_adapter_peeromega_fanout`, `claude_adapter_openrouter_ling_26_flash`, and `claude_adapter_gpt_oss_120b` passed. Focused NVIDIA validation showed DeepSeek and GLM still landing Langfuse trace `environment=default` despite correct `session_history.litellm_environment=prod`, and MiniMax timed out at 300s.

- Fixed the NVIDIA trace-environment root cause for the next image release.
  The Anthropic-to-completion transformer now preserves completion-adapter metadata through to the inner `litellm.acompletion()` call, so `trace_environment`, `session_id`, and related LiteLLM metadata are available to normal completion callbacks instead of being dropped after `metadata.user_id` conversion.

- Promoted the local prod `:4000` LiteLLM container to the `aawm.31` image and isolated the remaining prod validation blockers.
  Prod readiness reported LiteLLM `1.82.3+aawm.31`; `claude_adapter_peeromega_fanout` and `claude_adapter_openrouter_ling_26_flash` passed. `claude_adapter_gpt_oss_120b` was blocked by OpenRouter `503 provider=OpenInference raw=no healthy upstream`, NVIDIA DeepSeek/GLM succeeded but exposed a Langfuse trace-environment propagation gap, and NVIDIA MiniMax remained too slow for the current 300s opt-in spot-check timeout.

- Hardened the next prod validation release for the remaining adapter issues.
  NVIDIA/OpenRouter completion adapters now mirror trace/session context from `litellm_metadata` into normal completion `metadata` so Langfuse trace environment matches session-history environment. Harness `0.0.12` adds a narrow soft-fail classifier for the exact OpenRouter provider-unavailable timeout signature while keeping other hard-gate timeouts hard.

- Promoted the local prod `:4000` LiteLLM container to the durable `aawm.28` image.
  The running `aawm-litellm` container now reports LiteLLM `1.82.3+aawm.28` with callback wheel `0.0.8` and control-plane wheel `0.0.5`.

- Completed prod Anthropic-adapter harness validation against `:4000`.
  Isolated `claude_adapter_gpt54`, `claude_adapter_gpt55`, and `claude_adapter_gpt54_mini` runs passed, followed by a clean full default prod harness pass at `/tmp/litellm-prod-harness-aawm28.json` with zero failures and zero warnings.

- Hardened Claude harness header injection against local Claude settings precedence.
  Harness version `0.0.8` carries the temporary per-run `--settings` overlay for the harness-controlled `ANTHROPIC_BASE_URL` and `ANTHROPIC_CUSTOM_HEADERS`, so `x-litellm-end-user-id` / `langfuse_trace_user_id` validation no longer depends on ambient user or project settings.

- Removed brittle exact-response checks from basic OpenAI smoke cases.
  `gpt-5.4`, `gpt-5.5`, and `gpt-5.4-mini` smoke cases now hard-gate successful command completion, token/cost reporting, adapted-route tags, request payload logging, Langfuse trace/user/session context, runtime logs, and `session_history` rows without failing on harmless natural-language variation from Claude Code context.

## 2026-04-23

- Added LiteLLM runtime and initiating-client identity to `public.session_history`.
  New rows now persist `litellm_environment`, `litellm_version`, `litellm_fork_version`, `litellm_wheel_versions`, `client_name`, `client_version`, and `client_user_agent`; the same values are mirrored into row metadata for Langfuse/backfill context. Live writes derive these from trace environment metadata, installed package versions, associated overlay/config version env vars, request User-Agent, and Claude Code billing headers. Backfill paths only use values already present in spend-log/Langfuse metadata.

- Hardened the Anthropic adapter harness around the new runtime/client identity fields.
  Dev/prod target profile selection now injects `expected_litellm_environment` into session-history checks, and the harness fails rows missing runtime version, fork version, wheel-version JSON, client name, or client version. The callback overlay source remains byte-for-byte synced with the in-repo dev callback and its wheel version is now `0.0.8`.

- Advanced the callback, harness, and model-config artifact versions after branch promotion.
  Remote `cb-v0.0.7`, `h-v0.0.4`, and `cfg-v0.0.4` already existed on a release-only commit, so the durable release artifacts now use fresh non-rewritten versions: callback `0.0.8`, harness `0.0.5`, and model config `0.0.5`.

- Hardened the harness Claude trace-user setup for prod/dev parity.
  Harness version `0.0.6` injects a generated Claude trace-user header value and validates that exact value in Langfuse, without hard-coding an ambient operator identity.

- Advanced the fork image version to `1.82.3+aawm.27` for main-head promotion.
  The initial `v1.82.3-aawm.26` tag was cut before `main` converged, and the guarded image publisher rejects non-main-head tags; `aawm.27` avoids force-moving the published `aawm.26` tag.

- Advanced the fork image version to `1.82.3+aawm.28` after prod-cutover validation found a pass-through trace-user gap.
  Generic and Anthropic pass-through logging now resolve `user_api_key_end_user_id` from standard customer headers such as `x-litellm-end-user-id` when those headers are explicitly supplied, which lets the harness prove custom-header user identity flow end-to-end.

- Repaired `public.session_history` observability gaps in the local `aawm_tristore` database.
  The repair normalized null providers, removed invalid `reasoning_tokens_source=provider_reported` rows with zero reported reasoning tokens, populated target-provider cache statuses, and recalculated cache miss token/cost fields where usage exposed cache-write tokens.

- Hardened the session-history writer against the same regressions.
  New writes now infer missing providers from model/route metadata, sanitize non-positive reasoning token placeholders before persistence, default target-provider cache telemetry from stored cache counters, and preserve valid Gemini `provider_signature_present` reasoning signals.

- Synced the callback overlay source with the in-repo AAWM callback and bumped the callback wheel version to `0.0.6`.
  This prevents future `aawm-litellm` image rebuilds from reinstalling the stale callback wheel that omitted provider-cache and tool-activity fields.

- Broadened git tool-activity extraction.
  Command parsing now handles nested command payloads and git global options such as `git -C /repo commit` / `git --git-dir=/repo/.git push`, with regression coverage for parent `git_commit_count` / `git_push_count` rollups.

- Tightened adapter harness validation for session-history rows.
  Harness validation now selects file/git rollup fields and fails rows with null providers, null reasoning sources, invalid `provider_reported` zero counts, or missing provider-cache status on target provider families.

- Added literal ctx-marker escaping.
  Prompts can now use `\\:#name.ctx#\\:` to preserve visible `:#name.ctx#:` text without triggering `tristore_search_exact`, context appendix injection, or AAWM dynamic-injection metadata.

- Added default-suite harness coverage for escaped ctx markers and provider-cache activity.
  `claude_adapter_ctx_marker_escaped` validates the literal escape path, and `claude_adapter_peeromega_fanout` now requires at least one Anthropic child `session_history` row with `provider_cache_attempted=true` and `provider_cache_status` of `hit` or `write`.

- Added dev/prod target profiles to the Anthropic adapter harness.
  `--target dev` validates `:4001`, `litellm-dev`, and Langfuse trace environment `dev`; `--target prod` validates `:4000`, `aawm-litellm`, and Langfuse trace environment `prod`. The pass-through trace context helper now correctly preserves `session_id` and `trace_environment` metadata on rewritten requests.

- Fixed OpenRouter Anthropic-adapter Responses handling found during prod cutover.
  Native OpenRouter `/v1/responses` payloads now build a Responses-aware logging object instead of falling through OpenAI chat-completions parsing, preventing the prior async `KeyError: choices` logging exception while preserving Codex Responses `local_shell_call` output metadata. Translated Anthropic responses now strip stale upstream `content-length`, `content-encoding`, and `transfer-encoding` headers so synthesized bodies do not trigger `IncompleteRead` / `Too little data for declared Content-Length`. Validated on dev `:4001` with a direct `inclusionai/ling-2.6-flash:free` request returning a complete `200` response.

- Hardened the Anthropic adapter harness against the prod cutover failure modes.
  Runtime-log validation now defaults to failing on async task exceptions, ASGI exceptions, `KeyError: choices`, `h11` protocol errors, stale content-length failures, and upstream 429/5xx passthrough exceptions. Warning-only optional cases still allow configured quality mismatches, but no longer hide command timeouts or runtime-log hard failures. Added local harness unit coverage for timeout hard-fail behavior and default runtime-log detection.

- Fixed the shared OpenAI Responses function-schema failure that was blocking the dev harness.
  Anthropic-to-Responses tool translation now recursively normalizes nested object schemas, including `$defs` nodes that previously declared `type=object` without `properties`. That closes the upstream `400 Invalid schema for function ... object schema missing properties` failure seen on ChatGPT/Codex Responses traffic. Validated with focused passthrough regression tests, isolated live `claude_adapter_gpt54` / `claude_adapter_gpt54_mini` / `claude_adapter_ctx_marker` runs, and a clean full dev harness pass on `:4001`.

- Fixed OpenAI passthrough reconstruction for reasoning-only Responses output items.
  The passthrough logging handler now rebuilds valid `ModelResponse` objects when `response.completed` contains only `ResponseReasoningItem` output, including provider shapes that place reasoning text under `summary` or under `content[type=reasoning_text]`. That removes the prior `Error rebuilding complete responses API stream: Unknown items in responses API response: [ResponseReasoningItem(...)]` warning on `gpt-oss-120b` while preserving usage, hidden `responses_output`, and reasoning text on the reconstructed message. Validated with focused unit coverage plus an isolated live `claude_adapter_gpt_oss_120b` run on `:4001` and a clean overlapping `litellm-dev` log window.

- Added GPT-5.5 support to the Anthropic -> OpenAI Responses adapter.
  The adapter now allowlists `gpt-5.5` / `openai/gpt-5.5`, the local model cost maps include `gpt-5.5` and `chatgpt/gpt-5.5`, and the default Anthropic adapter harness includes `claude_adapter_gpt55`. The cost-map rates match the live Claude CLI `modelUsage` surfaced during validation. Validated with focused unit coverage, JSON/compile checks, and an isolated live `claude_adapter_gpt55` run on dev `:4001` with a clean overlapping `litellm-dev` log window.

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

- Stabilized the high-latency MiniMax path inside the NVIDIA Anthropic adapter.
  The NVIDIA lane now avoids nested OpenAI-provider retries by forcing the inner provider `max_retries` to `0`, uses a higher default adapter timeout, and fake-streams `minimaxai/minimax-m2.7` through Anthropic SSE while keeping the upstream NVIDIA call non-streaming.

- Established stable live NVIDIA spot checks on `:4001`.
  `claude_adapter_nvidia_deepseek_v32`, `claude_adapter_nvidia_glm47`, and `claude_adapter_nvidia_minimax_m27` now pass explicit end-to-end validation against `litellm-dev`, including `session_history` provider/model/cost validation, request-body or adapted-request checks, trace tag checks, and runtime-log traceback guards. They remain excluded from the default suite because the NVIDIA lane is still opt-in and MiniMax is materially slower than the other NVIDIA targets.

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
