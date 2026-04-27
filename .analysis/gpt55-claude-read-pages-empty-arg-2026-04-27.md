# GPT-5.5 Claude Read Tool Empty `pages` Argument

## Observation

In a Claude-orchestrated GPT-5.5 dispatch, the orchestrator reported that the
child agent repeatedly called Claude Code's `Read` tool with an empty optional
argument:

```json
{"pages": ""}
```

Claude Code rejected each call with `Invalid pages parameter`. GPT-5.5 retried
the same invalid call pattern until it exhausted its tool-call budget and
stalled.

This is different from a normal inefficient re-read loop. The model was trying
to use the tool, but the translated tool call contained an invalid optional
argument shape for Claude's tool executor.

## Likely Adapter Class

This belongs to non-Anthropic model output normalization for the `/anthropic`
adapter path. The model sees Claude tools through a translated schema and emits
OpenAI/Gemini/OpenRouter-style tool calls. Before those are translated back into
Anthropic `tool_use` blocks for Claude Code, known invalid optional argument
values should be normalized.

The immediate target is:

- Tool: `Read`
- Argument: `pages`
- Invalid value: empty string
- Safe normalization: omit `pages` entirely

The normalization should be narrow. Do not strip all empty strings globally,
because an empty string may be meaningful for some tools or user workflows.

## Suggested Fix

Add a small Claude-tool argument sanitizer on the adapter response path used by
non-Anthropic model invocations:

- Detect returned tool calls whose original Anthropic tool name is `Read`.
- Parse the tool-call arguments as JSON if they are serialized.
- If `pages` is present and is `""`, `[]`, or otherwise empty/non-informative,
  remove only `pages`.
- Re-serialize the arguments for the existing output translator, or sanitize the
  parsed Anthropic `tool_use.input` block after translation.
- Add metadata or debug-level counters such as
  `anthropic_adapter_sanitized_tool_args_count` and
  `anthropic_adapter_sanitized_tool_args=["Read.pages"]`.

Prefer a shared helper that can be reused by both non-streaming and streaming
translation paths. Candidate locations to inspect:

- `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`
- `litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py`
- provider-specific wrapper calls in
  `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`

## Prompt-Side Belt And Suspenders

The Claude agent prompt for GPT/OpenAI dispatch can also include:

```text
When calling the Read tool, do not pass `pages` unless reading a PDF and you
have a real page range. For normal source files, omit `pages` entirely.
```

That is useful, but should not be the only fix. The adapter should sanitize the
known invalid shape because this is a provider/tool-use quirk that can recur.

## Existing Harness Validation Plan

Use the existing Anthropic adapter harness.

Unit tests should cover:

- Non-streaming GPT/OpenAI adapter output where `Read` tool arguments include
  `{"file_path": "...", "pages": ""}` and the returned Anthropic `tool_use`
  block omits `pages`.
- Streaming output reconstruction for the same shape, if the current streaming
  test utilities can exercise tool-call argument assembly.
- Sanitization only applies to `Read.pages`, not arbitrary empty-string
  arguments on unrelated tools.
- Sanitization works after tool-name mapping/truncation restores the original
  Claude tool name.

Live harness coverage should add or extend a focused GPT-5.5/Codex/OpenAI
Claude-dispatch case:

- Launch the real Claude CLI through `:4001/anthropic` with a GPT-5.5/OpenAI
  child model selected through the normal Claude dispatch path.
- Use a prompt that causes the child model to read a small source/text file.
- Validate the run does not get stuck in repeated `Read` failures.
- Validate logged translated response/tool activity does not include
  `Read.pages=""`.
- Validate `session_history_tool_activity` records successful `Read` activity
  where observable, along with the normal provider/model/cost and trace
  environment invariants.
