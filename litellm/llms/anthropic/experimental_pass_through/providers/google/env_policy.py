"""Wave 4 extraction: google_env_policy pure-leaf functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Optional, cast

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global modules
    _aawm_alias_retry: Any
    _anthropic_google_shaping: Any

    # Host-global constants
    PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS: tuple

    # Host-global functions
    def _clean_codex_auth_value(value: Any) -> Optional[str]: ...
    def _get_passthrough_hidden_retry_wait_seconds(index: int) -> float: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_get_google_code_assist_prime_ttl_seconds",
    "_get_google_code_assist_prime_cache_key",
    "_get_google_adapter_max_concurrent",
    "_get_google_adapter_shared_lane_key",
    "_get_google_adapter_rate_limit_key",
    "_get_google_adapter_rate_limit_key_from_kwargs",
    "_get_google_adapter_max_retries",
    "_coerce_non_negative_int",
    "_coerce_non_negative_float",
    "_get_google_adapter_post_tool_cooldown_seconds",
    "_google_code_assist_unwrapped_chunk_contains_tool_call",
    "_get_google_adapter_max_output_tokens_cap",
    "_get_google_adapter_default_thinking_level",
    "_get_google_adapter_max_contents_window",
    "_get_google_adapter_max_contents_text_chars",
    "_estimate_google_content_text_chars",
    "_google_content_has_text",
    "_get_google_adapter_oversized_text_part_char_cap",
    "_get_google_adapter_pure_context_text_part_char_cap",
    "_get_google_adapter_subagent_context_text_part_char_cap",
    "_get_google_adapter_followup_subagent_context_text_part_char_cap",
    "_get_google_adapter_followup_allowed_tool_names",
    "_get_google_adapter_model_capacity_max_retries",
    "_get_google_adapter_capacity_backoff_seconds",
    "_get_google_adapter_hidden_retry_budget_seconds",
    "_get_google_adapter_transient_retry_max_attempts",
    "_get_google_adapter_transient_backoff_seconds",
    "_get_google_adapter_fallback_context_char_cap",
    "_get_google_adapter_system_prompt_policy",
    "_get_google_code_assist_native_tool_aliases",
    "_get_google_adapter_max_completion_messages_window",
    "_get_google_adapter_preserved_task_state_char_cap",
    "_get_google_adapter_native_user_agent",
    "_get_google_adapter_native_api_client_header",
    "_get_google_adapter_persisted_output_char_cap",
    "_get_google_adapter_auxiliary_context_char_cap",
    "_get_google_adapter_followup_persisted_output_char_cap",
    "_get_google_adapter_followup_auxiliary_context_char_cap",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = _rebound
        host_globals[_name] = _rebound


# ── Restored constants ──────────────────────────────────────────────

_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME = "google_anthropic_system_prompt_policy"

_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION = "2026-04-27.v2"

_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_ENV = "AAWM_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY"

_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT = "replace_compact"

_GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT = """You are a non-interactive CLI software engineering agent.

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
- Final responses must include visible assistant text. Never end a completed
  task with only thoughts or reasoning. After tool results, write the requested
  final answer in normal text.

Follow the preserved project, workspace, safety, and operator instructions
below."""

_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME = "codex_google_code_assist_tool_contract_policy"

_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION = "2026-05-12.v1"

_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_ENV = "AAWM_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY"

_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT = "append"

_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT = """Codex tool contract:
- Tool results are observations only. Never copy a previous tool result, terminal transcript, "Chunk ID", "Wall time", "Process exited", or "Output:" text into the arguments for a later tool call.
- For every function call, construct arguments from the declared tool schema. If the tool requires `cmd`, the arguments must contain a non-empty `cmd` string. Do not use `output`, `content`, or raw terminal transcript text as a substitute.
- After a tool result, continue the assigned task. Use the latest user task and requested output shape as authoritative.
- If a previous tool call failed because required arguments were missing, either retry once with schema-valid arguments or stop and explain the blocker in the final answer.
- Final answers must address the assigned task directly. Do not return generic descriptions of files unless the user asked for a file overview."""

# ── Extracted functions ─────────────────────────────────────────────

def _get_google_code_assist_prime_ttl_seconds() -> float:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_CODE_ASSIST_PRIME_TTL_SECONDS"))
    if raw_value is None:
        # Current Gemini CLI caches Code Assist user/project setup for 30s.
        # Match that default instead of re-priming on every adapted request.
        return 30.0
    try:
        parsed = float(raw_value)
    except Exception:
        return 30.0
    return max(0.0, parsed)

def _get_google_code_assist_prime_cache_key(
    access_token: str,
    companion_project: str,
) -> str:
    token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:12]
    return f"{token_hash}:{companion_project}"

def _get_google_adapter_max_concurrent() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_CONCURRENT"))
    if raw_value is None:
        return 1
    try:
        parsed = int(raw_value)
    except Exception:
        return 1
    return max(1, parsed)

def _get_google_adapter_shared_lane_key(
    *,
    access_token: Optional[str],
    companion_project: Optional[str],
) -> Optional[str]:
    # Gemini CLI's Code Assist envelope matches our request shape, but its
    # actual traffic is serialized on the shared account/project lane instead of
    # being split by model id. Mirror that here to avoid fanout-only 429s.
    cleaned_access_token = _clean_codex_auth_value(access_token)
    cleaned_companion_project = _clean_codex_auth_value(companion_project)
    if cleaned_access_token is None or cleaned_companion_project is None:
        return None
    return _get_google_code_assist_prime_cache_key(
        cleaned_access_token,
        cleaned_companion_project,
    )

def _get_google_adapter_rate_limit_key(
    model: Optional[str],
    *,
    access_token: Optional[str] = None,
    companion_project: Optional[str] = None,
) -> str:
    shared_lane_key = _get_google_adapter_shared_lane_key(
        access_token=access_token,
        companion_project=companion_project,
    )
    if shared_lane_key is not None:
        return shared_lane_key
    normalized = _clean_codex_auth_value(model)
    if normalized is None:
        return "__default__"
    return normalized

def _get_google_adapter_rate_limit_key_from_kwargs(kwargs: dict[str, Any]) -> str:
    explicit_rate_limit_key = _clean_codex_auth_value(cast(Optional[str], kwargs.get("google_adapter_rate_limit_key")))
    if explicit_rate_limit_key is not None:
        return explicit_rate_limit_key
    custom_body = kwargs.get("custom_body")
    model = custom_body.get("model") if isinstance(custom_body, dict) else None
    project = custom_body.get("project") if isinstance(custom_body, dict) else None
    access_token = cast(Optional[str], kwargs.get("google_access_token"))
    return _get_google_adapter_rate_limit_key(
        cast(Optional[str], model),
        access_token=access_token,
        companion_project=cast(Optional[str], project),
    )

def _get_google_adapter_max_retries() -> int:
    return _aawm_alias_retry.parse_non_negative_int_env(  # noqa: F821
        "AAWM_GOOGLE_ADAPTER_MAX_RETRIES",
        default=1,
    )

def _coerce_non_negative_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(0, parsed)

def _coerce_non_negative_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(0.0, parsed)

def _get_google_adapter_post_tool_cooldown_seconds() -> float:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_POST_TOOL_COOLDOWN_SECONDS"))
    if raw_value is None:
        return 0.0
    try:
        parsed = float(raw_value)
    except Exception:
        return 0.0
    return max(0.0, parsed)

def _google_code_assist_unwrapped_chunk_contains_tool_call(
    unwrapped: dict[str, Any],
) -> bool:
    candidates = unwrapped.get("candidates")
    if not isinstance(candidates, list):
        return False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("functionCall"), dict):
                return True
    return False

def _get_google_adapter_max_output_tokens_cap() -> Optional[int]:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_OUTPUT_TOKENS_CAP"))
    if raw_value is None:
        return 8192
    try:
        parsed = int(raw_value)
    except Exception:
        return 8192
    if parsed <= 0:
        return None
    return parsed

def _get_google_adapter_default_thinking_level(model: Optional[str]) -> Optional[str]:
    disabled = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG"))
    if isinstance(disabled, str) and disabled.lower() in {"1", "true", "yes", "on"}:
        return None

    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_DEFAULT_THINKING_LEVEL"))
    if raw_value:
        return raw_value

    normalized_model = (model or "").lower()
    if "flash-lite" in normalized_model:
        return "minimal"
    return "low"

def _get_google_adapter_max_contents_window() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_CONTENTS_WINDOW"))
    if raw_value is None:
        return 24
    try:
        parsed = int(raw_value)
    except Exception:
        return 24
    return max(2, parsed)

def _get_google_adapter_max_contents_text_chars() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_CONTENTS_TEXT_CHARS"))
    if raw_value is None:
        return 12000
    try:
        parsed = int(raw_value)
    except Exception:
        return 12000
    return max(1000, parsed)

def _estimate_google_content_text_chars(content_block: Any) -> int:
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._estimate_google_content_text_chars(content_block)  # noqa: F821

def _google_content_has_text(content_block: Any) -> bool:
    return _estimate_google_content_text_chars(content_block) > 0

def _get_google_adapter_oversized_text_part_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_OVERSIZED_TEXT_PART_CHAR_CAP"))
    if raw_value is None:
        return 6000
    try:
        parsed = int(raw_value)
    except Exception:
        return 6000
    return max(1500, parsed)

def _get_google_adapter_pure_context_text_part_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_PURE_CONTEXT_TEXT_PART_CHAR_CAP"))
    if raw_value is None:
        return 6000
    try:
        parsed = int(raw_value)
    except Exception:
        return 6000
    return max(512, parsed)

def _get_google_adapter_subagent_context_text_part_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_SUBAGENT_CONTEXT_TEXT_PART_CHAR_CAP"))
    if raw_value is None:
        return 2000
    try:
        parsed = int(raw_value)
    except Exception:
        return 2000
    return max(512, parsed)

def _get_google_adapter_followup_subagent_context_text_part_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_SUBAGENT_CONTEXT_TEXT_PART_CHAR_CAP"))
    if raw_value is None:
        return 1200
    try:
        parsed = int(raw_value)
    except Exception:
        return 1200
    return max(256, parsed)

def _get_google_adapter_followup_allowed_tool_names() -> set[str]:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_ALLOWED_TOOL_NAMES"))
    if raw_value:
        allowed_tool_names = {item.strip() for item in raw_value.split(",") if isinstance(item, str) and item.strip()}
    else:
        allowed_tool_names = {
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "Bash",
            "WebSearch",
            "WebFetch",
        }

    aliases = _get_google_code_assist_native_tool_aliases()
    expanded_tool_names = set(allowed_tool_names)
    for tool_name in list(allowed_tool_names):
        alias_name = aliases.get(tool_name)
        if isinstance(alias_name, str) and alias_name:
            expanded_tool_names.add(alias_name)

    return expanded_tool_names

def _get_google_adapter_model_capacity_max_retries() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_MAX_RETRIES"))
    if raw_value is None:
        return 3
    try:
        parsed = int(raw_value)
    except Exception:
        return 3
    return max(0, parsed)

def _get_google_adapter_capacity_backoff_seconds(attempt: int) -> float:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_BACKOFF_SECONDS"))
    if raw_value:
        try:
            values = [max(1.0, float(item.strip())) for item in raw_value.split(",") if item.strip()]
        except Exception:
            values = []
        if values:
            index = min(max(1, attempt) - 1, len(values) - 1)
            return values[index]
    schedule = (5.0, 15.0, 30.0, 60.0)
    index = min(max(1, attempt) - 1, len(schedule) - 1)
    return schedule[index]

def _get_google_adapter_hidden_retry_budget_seconds() -> float:
    return _aawm_alias_retry.parse_non_negative_float_env(  # noqa: F821
        "AAWM_GOOGLE_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS",
        default=0.0,
    )

def _get_google_adapter_transient_retry_max_attempts() -> int:
    return len(PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS) + 1  # noqa: F821

def _get_google_adapter_transient_backoff_seconds(attempt: int) -> float:
    return _get_passthrough_hidden_retry_wait_seconds(max(0, attempt - 1))

def _get_google_adapter_fallback_context_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_FALLBACK_CONTEXT_CHAR_CAP"))
    if raw_value is None:
        return 2000
    try:
        parsed = int(raw_value)
    except Exception:
        return 2000
    return max(256, parsed)

def _get_google_adapter_system_prompt_policy() -> str:
    raw_value = _clean_codex_auth_value(os.getenv(_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_ENV))
    if raw_value is None:
        return _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"0", "false", "disabled", "none", "off"}:
        return "off"
    if normalized_value in {"append", "replace_compact"}:
        return normalized_value
    return _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT

def _get_google_code_assist_native_tool_aliases() -> dict[str, str]:
    return {
        "Bash": "run_shell_command",
        "Read": "read_file",
        "Write": "write_file",
        "Edit": "replace",
        "Glob": "glob",
        "Grep": "grep_search",
        "WebFetch": "web_fetch",
        "WebSearch": "google_web_search",
    }

def _get_google_adapter_max_completion_messages_window() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_COMPLETION_MESSAGES_WINDOW"))
    if raw_value is None:
        return 12
    try:
        parsed = int(raw_value)
    except Exception:
        return 12
    return max(2, parsed)

def _get_google_adapter_preserved_task_state_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_PRESERVED_TASK_STATE_CHAR_CAP"))
    if raw_value is None:
        return 6000
    try:
        parsed = int(raw_value)
    except Exception:
        return 6000
    return max(512, parsed)

def _get_google_adapter_native_user_agent(model: Optional[str]) -> str:
    configured = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_NATIVE_USER_AGENT"))
    if configured:
        return configured
    model_name = model or "gemini-3-flash-preview"
    return f"GeminiCLI/0.38.2/{model_name} (linux; x64; terminal) google-api-nodejs-client/9.15.1"

def _get_google_adapter_native_api_client_header() -> str:
    configured = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_NATIVE_X_GOOG_API_CLIENT"))
    if configured:
        return configured
    return "gl-node/24.13.1"

def _get_google_adapter_persisted_output_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP"))
    if raw_value is None:
        return 2000
    try:
        parsed = int(raw_value)
    except Exception:
        return 2000
    return max(256, parsed)

def _get_google_adapter_auxiliary_context_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_AUXILIARY_CONTEXT_CHAR_CAP"))
    if raw_value is None:
        return 4000
    try:
        parsed = int(raw_value)
    except Exception:
        return 4000
    return max(512, parsed)

def _get_google_adapter_followup_persisted_output_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_PERSISTED_OUTPUT_CHAR_CAP"))
    if raw_value is None:
        return 512
    try:
        parsed = int(raw_value)
    except Exception:
        return 512
    return max(128, parsed)

def _get_google_adapter_followup_auxiliary_context_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_AUXILIARY_CONTEXT_CHAR_CAP"))
    if raw_value is None:
        return 1024
    try:
        parsed = int(raw_value)
    except Exception:
        return 1024
    return max(256, parsed)
