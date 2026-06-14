# AAWM Runtime Error Logs

Managed AAWM LiteLLM environments can mirror `ERROR`-level LiteLLM log records
into the local repo analysis intake directory. This is an operator workflow aid:
the durable queue remains `.analysis/todo.md`, while `*-error.log` files are
short-lived intake artifacts that should be converted into TODO items and then
cleaned up after resolution.

Set `LITELLM_AAWM_ERROR_LOG_ENABLED=1` to enable the sink. The target directory
defaults to `.analysis` under the current working directory and can be overridden
with `LITELLM_AAWM_ERROR_LOG_DIR`. The environment name in the filename comes
from `LITELLM_AAWM_ERROR_LOG_ENV`, then `LITELLM_LANGFUSE_TRACE_ENVIRONMENT`,
then `LITELLM_ENV` or `ENVIRONMENT`; unsafe characters are normalized. For
example, the dev proxy writes `/app/.analysis/dev-error.log` inside the
container, mounted to this repo's `.analysis/dev-error.log`.

The sink reuses LiteLLM's secret-redaction filter and records traceback context
without adding request bodies, prompt payloads, or tool arguments. Error-log
files are local sensitive artifacts and must not be committed or pushed.
