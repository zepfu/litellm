# AAWM Runtime Error Logs

Managed AAWM LiteLLM environments can mirror `ERROR`-level LiteLLM log records
into the local repo analysis intake directory. This is an operator workflow aid:
the durable queue remains `.analysis/todo.md`, while `*-error.jsonl` files are
short-lived structured intake artifacts that should be converted into TODO items
and then cleaned up after resolution.

During the migration window, agents should also inspect legacy
`.analysis/*-error.log` files. Those plain-text files are retained for discovery
only until all writers and intake instructions have moved to JSONL.

## Enablement

Set `LITELLM_AAWM_ERROR_LOG_ENABLED=1` to enable the sink. The target directory
defaults to `.analysis` under the current working directory and can be overridden
with `LITELLM_AAWM_ERROR_LOG_DIR`. The environment name in the filename comes
from `LITELLM_AAWM_ERROR_LOG_ENV`, then `LITELLM_LANGFUSE_TRACE_ENVIRONMENT`,
then `LITELLM_ENV` or `ENVIRONMENT`; unsafe characters are normalized.

For example, the dev proxy writes `/app/.analysis/dev-error.jsonl` inside the
container, mounted to this repo's `.analysis/dev-error.jsonl`. The matching
legacy path is `/app/.analysis/dev-error.log`, mounted to
`.analysis/dev-error.log`.

## JSONL intake shape

The sink writes append-safe newline-delimited JSON. Each line is one complete
JSON object, so repeated runtime failures can be appended without rewriting the
file.

Each event should include at least:

- `schema_version`
- `environment`
- `observed_at`
- `logger`
- `level`
- `message`
- `traceback`, `traceback_text`, and `traceback_lines`
- `raw_text` with the original redacted error text
- `fingerprint` for grouping repeated failures
- `context` with route/provider metadata when available

Practical `context` fields to inspect are:

- `source`
- `container`
- `endpoint`
- `upstream_url`
- `provider`
- `model`
- `model_alias`
- `route_family`
- `status_code`
- `trace_id`
- `litellm_call_id`

Use the structured fields to group and triage failures, but keep
`.analysis/todo.md` as the source of truth for active work.

## Agent intake workflow

At TODO-driven work startup, inspect:

- `.analysis/*-error.jsonl`
- legacy `.analysis/*-error.log`

For every active intake file, add or update a `.analysis/todo.md` entry for the
underlying resolution. Capture the environment, error detail, traceback context,
intended fix, acceptance evidence, and a cleanup requirement for the source
`*-error.jsonl` or legacy `*-error.log` file.

Do not mark an error-intake-driven item complete until the underlying error is
fixed, verification evidence is recorded, and the corresponding active JSONL or
legacy text intake file has been deleted or archived out of active intake.

## Redaction and local-only handling

The sink reuses LiteLLM's secret-redaction filter and records traceback context
without adding request bodies, prompt payloads, or tool arguments. Error-intake
files are local sensitive artifacts and must not be committed or pushed.
