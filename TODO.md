# TODO

## Cold Start Orientation

Use this file as the primary restart surface. For a no-context session, first read
the current `Validated Context` and `Next` sections below, then branch into
these docs only as needed:

- Local `.analysis/completed*.md` ledgers - recent completed work, validation
  results, repair stats, and release/session-history context from prior
  sessions. These ledgers are intentionally not committed.
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
- Keep [TODO.md](TODO.md) and the local `.analysis/completed*.md` ledgers
  current while work is underway, not just at the end. `TODO.md` should say
  what is currently unresolved, what failed, and the next plan of attack. The
  local completed ledgers should record what was fixed, what was validated,
  exact artifacts/traces when useful, and any dead ends that should not be
  repeated.
- When a fix is believed to work, proactively add the evidence to
  the local completed ledger. If later validation fails, update that ledger
  with the failed result and update [TODO.md](TODO.md) with the next iteration
  before continuing.
- After each major feature or repair is delivered and checked in locally, commit
  it on the `develop` branch. Keep the commit boundary aligned with the proven
  behavior so later sessions can tell what code, tests, and notes belonged
  together.
- Avoid repeating known dead ends. Before rerunning an approach, check the
  local completed ledgers for prior failures and [TODO.md](TODO.md) for the
  current planned route.

## Validated Context

- Anthropic rate-limit observation capture/backfill is implemented and live on
  prod `:4000`. Root cause was missing non-stream Anthropic response header
  capture plus extractor blind spots for hidden
  `llm_provider-anthropic-ratelimit-*` headers. The release is published as
  `v1.82.3-aawm.50` with callback overlay `cb-v0.0.26`; prod container
  `dc96d643dff4` is healthy from local image id `5d05f2dd5e0e`, with
  `litellm=1.82.3+aawm.50`, `aawm-litellm-callbacks=0.0.26`, and
  `aawm-litellm-control-plane=0.0.7`. Exact database `aawm_tristore` has live
  post-cutover Anthropic observations from `anthropic_response_headers`, and
  `public.rate_limit_intervals` was refreshed/analyzed after cutover.

- `public.session_history` malformed repository identity rows were repaired in
  exact database `aawm_tristore`: historical free-text rollout descriptors,
  JSON-schema fragments, and CLI option strings were cleared or normalized, and
  verification showed `0` malformed `repository`, `tenant_id`, and
  `metadata.repository` rows. The source and callback-wheel copies of
  `aawm_agent_identity.py` now reject non-string and non-repo-shaped repository
  values, with focused tests passing. Historical deployment evidence recorded
  that the callback hotfix was promoted to prod as `cb-v0.0.24`; prod `:4000`
  was rebuilt on the existing `aawm.49` base and was healthy with callback
  `0.0.24`. Post-deploy DB checks stayed at `0` malformed identity rows while
  new prod `/openai_passthrough/responses` traffic was arriving at that time.

- Historical `aawm.49` production evidence recorded a `:4000` deployment using
  the `aawm.49` base image with callback hotfix overlay `0.0.24`. At that time,
  LiteLLM `main` was `7bd79f127f` after repository identity hotfix commit
  `3878c1a783` and artifact autobump run `25839627206`, with releases
  `v1.82.3-aawm.49`, `cb-v0.0.24`, `cp-v0.0.7`, `cfg-v0.0.10`, and
  `h-v0.0.29`. The recorded container `d1a51fdc2b4a` was healthy on
  `127.0.0.1:4000` from local image `aawm-litellm:latest` image id
  `a07aceecfb05`; package inspection reported `litellm=1.82.3+aawm.49`,
  `aawm-litellm-callbacks=0.0.24`, and `aawm-litellm-control-plane=0.0.7`.
  The native prod passthrough shard passed at
  `/tmp/litellm-prod-native-aawm49-cb23.json`, recording Codex repository
  attribution (`repository=zepfu/litellm`) and Gemini CLI normalization
  (`client_name=gemini-cli` for `GeminiCLI-tui/0.42.0/...`) as historical
  evidence, not as a current harness lane. The default prod harness artifact
  `/tmp/litellm-prod-harness-aawm49-cb23.json` was not fully green because
  Spark/Codex cases hit `usage_limit_reached` with reset
  `2026-05-18 15:08:41 UTC`; the only non-Codex failure was rerun cleanly at
  `/tmp/litellm-prod-gpt54-mini-aawm49-cb23-rerun.json`. The final filtered
  prod log scan found no release-blocking patterns at that time.

- Historical aawm.48 release evidence: the production image, callback overlay,
  control-plane package, configuration, and harness releases were published and
  inspected successfully. Production-profile Gemini text and tool-use smokes
  completed at that time, recorded provider and policy metadata, and produced
  matching Google Code Assist quota observations. This was retained only as
  past release evidence.

- Historical aawm.47 release evidence: the production image and callback
  overlay were published and inspected successfully. A production-profile
  Gemini smoke completed at that time, recorded provider and policy metadata,
  and produced matching Google Code Assist quota observations. This is retained
  only as a past release outcome, not as a current production instruction.

- Historical sequential tool evidence: GPT-5.5 and the then-available Gemini
  lanes completed Claude-dispatched base-tool proofs before May 2026. The
  Gemini results recorded single-tool turns, clean tool-result handling,
  tenant-only Langfuse user IDs, and durable tool activity.

- Historical parallel tool evidence: before May 2026, a trace showed five
  upstream Gemini tool calls collapsing to one Claude-visible tool call. The
  adapter was changed to buffer streaming deltas and emit one Anthropic
  `tool_use` block per upstream call; focused stream and transformation tests,
  followed by live proofs on the then-available Gemini lanes, passed at that
  time.

- GPT-5.5/OpenAI Claude-dispatched parallel tool calls are now validated on dev
  `:4001`. Keep the dead-end breadcrumb in the local completion ledger: context
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

- Publish and promote `v1.82.3-aawm.139` for complete portable provider-health
  parity on Thoth. The candidate adds read-only Grok/Codex/xAI/Kimi auth-health
  observations and a bounded anomaly-scan timeout; the infrastructure candidate
  adds exact Codex credential sync/read-only mounting, ICMP `NET_RAW`, Codex
  reset-credit, and anomaly polling while retaining workstation-only OAuth
  refresh ownership. WSL dev proof passed all 163 provider-status tests, Ruff,
  Mypy, the full Thoth infrastructure smoke, fresh endpoint/auth/quota/reset
  rows, clean structured logs, and zero sidecar restarts. Follow
  `PROD_RELEASE.md`, publish the immutable `.139` release, deploy only the named
  Thoth services, and prove fresh rows in all eight Thoth health surfaces with
  workstation marker isolation.

- Rerun the Spark/Codex-dependent prod harness cases after the upstream Codex
  quota reset at `2026-05-18 15:08:41 UTC`: at minimum
  `claude_adapter_codex_tool_activity`, `claude_adapter_peeromega_fanout`, and
  `claude_adapter_spark`. The 2026-05-13 prod artifact
  `/tmp/litellm-prod-harness-aawm49-cb23.json` shows the exact
  `usage_limit_reached` reset payloads, while non-Codex lanes passed or were
  rerun cleanly.

- Full monolithic `make test` is still not a clean local release gate. The
  2026-05-01 rerun collected `18172 items / 83 errors` after fixing the three
  documentation collection regressions from the earlier `86`-error run. Keep
  using focused unit/doc/harness gates for this release lane unless the broader
  suite is split or the remaining collection/setup blockers are intentionally
  addressed: duplicate test module basenames in one pytest invocation, missing
  optional deps (`PIL`, `google.genai`), missing live credentials, old proxy/live
  tests executing at import time, and the Vertex vector-store transformation
  import gap.

- D1-060 prompt-overhead tracking has live `session_history` fields,
  translated-shape unit coverage, native Codex assertions, and
  `summary.prompt_overhead_cost_share` in the local harness artifact. Native
  live coverage had populated Claude, OpenAI, Codex, and Gemini rows by May
  2026. The remaining provider-independent question is whether the proportional
  `response_cost_usd` estimate needs a later exact input-cost field.

- Prod `:4000` is on `v1.82.3-aawm.43` with callback overlay `0.0.18`;
  detailed cutover, smoke, and backfill evidence lives in
  the local completion ledger. No prod restart/backfill is currently pending
  for D1-075/D1-076/D1-077/D1-078. Next useful release follow-up is optional
  focused/default prod harness validation and normal monitoring of new prod
  Codex Responses prompt-overhead rows after the callback cutover.

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

- Historical repository-attribution evidence: on 2026-04-28, Codex CLI and
  Gemini CLI traffic through LiteLLM had failed to populate
  `public.session_history.repository`. The fix added the top-level column,
  header propagation, metadata mirroring, and focused development proof. On
  2026-04-29, follow-up sessions established `tenant_id` as the durable
  repository grouping field and expanded repository inference to prepared
  metadata, Codex workspace context, Gemini CLI workspace-directory text, and
  structured workspace-root keys. The recorded Codex and Gemini proofs included
  top-level and metadata repository values plus tenant fallbacks for
  repository-only rows.

- Historical `/anthropic` streaming evidence: by 2026-04-29, focused fixes for
  provider stream reconstruction and tool-call parity had passed, followed by a
  broad development harness pass. Native CLI captures were used only as
  comparison evidence for provider envelopes, event ordering, and durable
  logging.

### OpenAI/Codex Responses Adapter

- Codex stream/tool parity is now validated for the narrow focused pair. Keep
  `native_openai_passthrough_responses_codex_tool_activity,claude_adapter_codex_tool_activity`
  as the OpenAI/Codex gate before broad harness work. The latest passing
  artifact is `/tmp/anthropic_codex_tool_alias_after_fix.json`: upstream
  OpenAI/Codex sees native `exec_command` / `cmd` tool state while Claude Code
  still sees `Bash` / `command` on the Anthropic side, scoped only to the
  Codex-backed `/anthropic` route.

- Anthropic-only tool metadata stripping for non-Anthropic provider schemas was
  completed locally. The OpenAI-compatible chat path strips `defer_loading`,
  `eager_input_streaming`, `allowed_callers`, and `input_examples` before
  forwarding provider tools, and the OpenAI Responses path has focused
  regression coverage. Focused local validation and the failed-case live rerun
  passed. Broad runs on 2026-04-29 also cleared the OpenAI/Codex path and all
  non-fanout default cases.

### Deferred Tool Work

- Deferred/parked: adapter-owned support for Anthropic deferred tool loading on
  `/anthropic` to non-Anthropic paths. The preferred design is a bounded
  internal tool-search loop: hold `defer_loading=true` tool definitions in an
  adapter registry, send upstream only normal tools plus a compact synthetic
  search tool, intercept synthetic search calls without streaming them to
  Claude Code, rank and select matching deferred tools, and issue the follow-up
  provider call with those definitions expanded. Start with OpenAI
  Responses/Codex when this is resumed, then evaluate OpenRouter and NVIDIA
  after the control loop, usage aggregation, and stream hiding behavior are
  stable. Focused tests must prove the deferred-schema, internal-search,
  explicit-tool-choice, and usage-accounting behavior before broad coverage.

- Deferred/parked: revisit `eager_input_streaming`.
  OpenAI/OpenRouter/NVIDIA chat-style streams already forward upstream
  function-argument deltas as Anthropic `input_json_delta` events when the
  provider streams them. The OpenAI Responses/Codex path currently buffers
  Codex `exec_command` and `Read` until valid JSON for alias/sanitizer
  correctness. Decide per tool/provider whether `eager_input_streaming=true`
  should bypass that buffering, using focused fixtures with large
  write/edit-style arguments before enabling it broadly.

### Historical Google/Gemini Code Assist Evidence

- Historical Google/Gemini Code Assist evidence: in April 2026, native and
  `/anthropic` captures established the Code Assist request envelope,
  model-scoped session metadata, native function aliases, streaming tool-name
  restoration, parallel tool-call buffering, and terminal usage preservation.
  The captures also documented the then-observed `thinkingBudget` and
  `thinkingLevel` differences between Gemini model generations. These results
  are retained as implementation history, not as an active harness lane.

### NVIDIA/OpenRouter Completion Adapters

- Current unit coverage keeps Anthropic hosted and beta tools out of
  OpenAI-compatible `tools`, preserves web search as `web_search_options`,
  removes forced `tool_choice` values that target dropped hosted tools, and
  records unsupported hosted-tool downgrades in metadata. The NVIDIA focused
  live proof passed. Remaining work is limited to an available OpenRouter-style
  completion target that can exercise the same policy; transformed upstream
  payload capture is required before payload assertions can cover that path.

### OpenRouter Responses Adapter

- Treat it as the OpenAI Responses parity path plus OpenRouter-specific
  routing/empty-success behavior for `/anthropic` traffic. There is no
  first-party OpenRouter CLI baseline, so compare request/stream reconstruction
  to the native OpenAI/Codex baseline and then run
  `claude_adapter_openrouter_nemotron_child_parallel_read_tools` as the focused
  replacement proof before any peeromega fanout rerun.

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

- Claude-dispatched model latency remains classified separately from native CLI
  latency. For GPT-5.5, current evidence points to large model turns plus
  Claude Code/orchestrator overhead: the focused GPT-5.5 write probe took
  `50.192s` wall clock through Claude Code while the GPT adapter session logged
  about `2.156s` proxy/upstream duration; later aawm-tap GPT spans still ran
  `64-172s`. Historical Gemini evidence from the v11 aawm-tap trace showed a
  concrete adapter stream bug in which five upstream tool calls collapsed to
  one Claude-visible `Read` and a Claude Code protocol error.

- Current generic adapter-harness invariants cover target port, container, and
  trace-environment alignment; tenant-only Langfuse user IDs; durable tool
  activity; usage, cost, routing, and session-history evidence; and hard
  runtime-failure guards. Warning-only optional cases remain limited to
  documented provider-availability failures; command timeouts and runtime-log
  failures remain hard failures.

- Historical Gemini harness evidence: retired fanout, direct-read, and
  post-tool-result cases once matched persisted native tool names to
  Claude-visible aliases. Those named gates are no longer current operator
  guidance.

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
