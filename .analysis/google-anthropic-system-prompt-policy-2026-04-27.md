# Google Anthropic Adapter System Prompt Policy

## Scope

This applies only to the `/anthropic/v1/messages` route when the adapter routes
the request to Google Code Assist / Gemini through
`anthropic_google_completion_adapter`.

The goal is not to change native Gemini CLI traffic. The goal is to make
Claude-initiated Gemini model invocations receive a Gemini-compatible CLI agent
operating contract instead of carrying Claude-specific provider overhead through
the translated request.

## Recommendation

Add a Google-only system prompt rewrite policy inside
`_build_google_code_assist_request_from_completion_kwargs()` after Anthropic
input has been converted to canonical completion kwargs and before
`_transform_request_body()` builds the Gemini `systemInstruction`.

The policy should:

- Remove Claude/Anthropic operational overhead from the translated system text,
  including Claude identity text, Claude tool-use scaffolding, and Claude-only
  workflow conventions that are not useful to Gemini.
- Preserve task, project, workspace, safety, memory, and operator constraints
  that still apply to the request.
- Replace the removed overhead with a compact AAWM-owned Gemini CLI-style agent
  prompt.
- Keep the existing Claude-to-Gemini tool aliasing intact:
  `Bash -> run_shell_command`, `Read -> read_file`, `Write -> write_file`,
  `Edit -> replace`, `Glob -> glob`, `Grep -> grep_search`,
  `WebFetch -> web_fetch`, and `WebSearch -> google_web_search`.
- Make the policy feature-flagged and observable. Suggested policy values:
  `off`, `append`, and `replace_compact`, with `replace_compact` as the intended
  end state after harness validation.
- Persist summary metadata, not full prompt text, such as policy name, policy
  version, original system character count, rewritten system character count,
  removed Claude-overhead character count, and preserved instruction character
  count.

Do not copy Gemini CLI's bundled system prompt verbatim. Use a small local
prompt that captures the behavior we need and can own/revise.

## Suggested Compact Prompt

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

The final Gemini system instruction should place preserved instructions after
that compact prompt under a clear heading, for example:

```text
# Preserved Project And Safety Instructions
...
```

## Existing Harness Validation Plan

Use the existing Anthropic adapter harness instead of adding a new harness.

Unit tests should cover:

- The rewrite helper removes Claude-specific identity/tool/workflow overhead
  while preserving project, workspace, safety, memory, and task constraints.
- The policy can be disabled with `off` and leaves the translated system prompt
  unchanged.
- `append` keeps the original system text but prepends/appends the compact
  Gemini agent contract for comparison during rollout.
- `replace_compact` emits the compact prompt plus preserved instructions and
  excludes known Claude-only markers such as `You are Claude Code`.
- The Google adapter request body contains the rewritten
  `request.systemInstruction` before egress.
- Existing native tool aliasing still yields Gemini-native function names.
- Request/logging metadata records the prompt policy summary without storing the
  full prompt in normal logs.

Live harness coverage should add or extend an existing `claude_adapter_gemini_*`
case:

- Launch the real Claude CLI through `:4001/anthropic` with a Gemini child model
  selected through the normal Claude agent/fanout path.
- Use a bounded artifact-writing prompt so success requires the Gemini child to
  proceed beyond file reads into write/edit behavior.
- Validate the logged Google Code Assist request body contains the compact
  Gemini prompt marker in `request.systemInstruction`.
- Validate the logged request body does not contain known Claude-only overhead
  markers in `request.systemInstruction`.
- Validate route metadata/session-history metadata contains the selected prompt
  policy and prompt-policy version.
- Validate normal Gemini adapter invariants still hold: provider/model/cost
  session-history rows, Langfuse trace environment for the selected target,
  adapted route tags, and no runtime-log blocker patterns.
- Where the existing harness can observe it reliably, validate `write_file` or
  equivalent write/edit tool activity. If child tool activity remains a parent
  Claude `Agent` invariant only, keep the hard gate on final artifact creation
  plus the logged Gemini request prompt-policy checks.

Run the focused dev case first, then the full default dev harness. After dev is
clean, run the same focused case and full default harness with `--target prod`
as part of the normal production promotion process.
