"""AAWM observability callback for Langfuse attribution.

Extracts agent identity from the SubagentStart hook context injected into
request prompts, then enriches the langfuse_trace_name request header so
each agent's API calls can be distinguished in Langfuse.

The hook injects: "You are '<agent-name>' and you are working..."
When no agent designation is found, defaults to "orchestrator".

Enriches langfuse_trace_name from "claude-code" to "claude-code.<agent>"
(e.g. "claude-code.ops").

Uses BOTH logging_hook() (sync) and async_logging_hook() (async) to modify
headers BEFORE Langfuse's add_metadata_from_header() reads them. The sync
hook is critical for pass-through endpoints because Langfuse runs as a string
callback ("langfuse") in the sync success_handler - the async hook alone
would race with the thread-pool-submitted sync handler.

Registration in litellm-config.yaml:
    litellm_settings:
      callbacks: ["aawm_litellm_callbacks.agent_identity.AawmAgentIdentity"]
      success_callback: ["langfuse"]
"""

import asyncio
import base64
import hashlib
import importlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.secret_managers.main import get_secret_str

_AGENT_RE = re.compile(r"You are '([^']+)' and you are working")
_AGENT_TENANT_RE = re.compile(
    r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project"
)
_DEFAULT_AGENT = "orchestrator"
_CLAUDE_EXPERIMENT_ID_RE = re.compile(
    rb"(?<![A-Za-z0-9._-])([A-Za-z][A-Za-z0-9._-]{11,})(?![A-Za-z0-9._-])"
)
_GEMINI_MARKER = bytes.fromhex("8f3d6b5f")
_AAWM_DB_HOST_ENV_VARS = (
    "AAWM_DB_HOST",
    "AAWM_POSTGRES_SERVER",
    "POSTGRES_SERVER",
    "PGHOST",
)
_AAWM_DB_PORT_ENV_VARS = (
    "AAWM_DB_PORT",
    "AAWM_POSTGRES_PORT",
    "POSTGRES_PORT",
    "PGPORT",
)
_AAWM_DB_USER_ENV_VARS = (
    "AAWM_DB_USER",
    "AAWM_POSTGRES_USER",
    "POSTGRES_USER",
    "PGUSER",
)
_AAWM_DB_PASSWORD_ENV_VARS = (
    "AAWM_DB_PASSWORD",
    "AAWM_DB_PWD",
    "AAWM_POSTGRES_PASSWORD",
    "AAWM_POSTGRES_PWD",
    "POSTGRES_PASSWORD",
    "POSTGRES_PWD",
    "PGPASSWORD",
)
_AAWM_DB_NAME_ENV_VARS = (
    "AAWM_DB_NAME",
    "AAWM_POSTGRES_DATABASE",
    "POSTGRES_DATABASE",
    "PGDATABASE",
)
_AAWM_DB_SSLMODE_ENV_VARS = (
    "AAWM_DB_SSLMODE",
    "AAWM_POSTGRES_SSLMODE",
    "POSTGRES_SSLMODE",
    "PGSSLMODE",
)
_AAWM_DB_SSL_BOOL_ENV_VARS = (
    "AAWM_DB_SSL",
    "AAWM_POSTGRES_SSL",
    "POSTGRES_SSL",
)
_AAWM_DB_URL_ENV_VARS = (
    "AAWM_DB_URL",
    "AAWM_DATABASE_URL",
    "AAWM_POSTGRES_URL",
)
_AAWM_SESSION_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.session_history (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    litellm_call_id TEXT UNIQUE,
    session_id TEXT NOT NULL,
    trace_id TEXT,
    provider_response_id TEXT,
    provider TEXT,
    model TEXT NOT NULL,
    model_group TEXT,
    agent_name TEXT,
    tenant_id TEXT,
    call_type TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens_reported INTEGER,
    reasoning_tokens_estimated INTEGER,
    reasoning_tokens_source TEXT,
    reasoning_present BOOLEAN NOT NULL DEFAULT FALSE,
    thinking_signature_present BOOLEAN NOT NULL DEFAULT FALSE,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    tool_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    response_cost_usd DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""
_AAWM_SESSION_HISTORY_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS session_history_session_created_idx ON public.session_history (session_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_session_model_created_idx ON public.session_history (session_id, model, created_at DESC)",
)
_AAWM_SESSION_HISTORY_INSERT_SQL = """
INSERT INTO public.session_history (
    litellm_call_id,
    session_id,
    trace_id,
    provider_response_id,
    provider,
    model,
    model_group,
    agent_name,
    tenant_id,
    call_type,
    start_time,
    end_time,
    input_tokens,
    output_tokens,
    total_tokens,
    cache_read_input_tokens,
    cache_creation_input_tokens,
    reasoning_tokens_reported,
    reasoning_tokens_estimated,
    reasoning_tokens_source,
    reasoning_present,
    thinking_signature_present,
    tool_call_count,
    tool_names,
    response_cost_usd,
    metadata
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
    $21, $22, $23, $24::jsonb, $25, $26::jsonb
)
ON CONFLICT (litellm_call_id) DO UPDATE SET
    session_id = COALESCE(NULLIF(EXCLUDED.session_id, ''), session_history.session_id),
    trace_id = COALESCE(NULLIF(EXCLUDED.trace_id, ''), session_history.trace_id),
    provider_response_id = COALESCE(
        NULLIF(EXCLUDED.provider_response_id, ''),
        session_history.provider_response_id
    ),
    provider = COALESCE(NULLIF(EXCLUDED.provider, ''), session_history.provider),
    model = COALESCE(NULLIF(EXCLUDED.model, ''), session_history.model),
    model_group = COALESCE(NULLIF(EXCLUDED.model_group, ''), session_history.model_group),
    agent_name = COALESCE(NULLIF(EXCLUDED.agent_name, ''), session_history.agent_name),
    tenant_id = COALESCE(NULLIF(EXCLUDED.tenant_id, ''), session_history.tenant_id),
    call_type = COALESCE(NULLIF(EXCLUDED.call_type, ''), session_history.call_type),
    start_time = COALESCE(session_history.start_time, EXCLUDED.start_time),
    end_time = COALESCE(EXCLUDED.end_time, session_history.end_time),
    input_tokens = GREATEST(session_history.input_tokens, EXCLUDED.input_tokens),
    output_tokens = GREATEST(session_history.output_tokens, EXCLUDED.output_tokens),
    total_tokens = GREATEST(session_history.total_tokens, EXCLUDED.total_tokens),
    cache_read_input_tokens = GREATEST(
        session_history.cache_read_input_tokens,
        EXCLUDED.cache_read_input_tokens
    ),
    cache_creation_input_tokens = GREATEST(
        session_history.cache_creation_input_tokens,
        EXCLUDED.cache_creation_input_tokens
    ),
    reasoning_tokens_reported = COALESCE(
        GREATEST(session_history.reasoning_tokens_reported, EXCLUDED.reasoning_tokens_reported),
        session_history.reasoning_tokens_reported,
        EXCLUDED.reasoning_tokens_reported
    ),
    reasoning_tokens_estimated = COALESCE(
        GREATEST(session_history.reasoning_tokens_estimated, EXCLUDED.reasoning_tokens_estimated),
        session_history.reasoning_tokens_estimated,
        EXCLUDED.reasoning_tokens_estimated
    ),
    reasoning_tokens_source = COALESCE(
        NULLIF(EXCLUDED.reasoning_tokens_source, ''),
        session_history.reasoning_tokens_source
    ),
    reasoning_present = session_history.reasoning_present OR EXCLUDED.reasoning_present,
    thinking_signature_present = session_history.thinking_signature_present OR EXCLUDED.thinking_signature_present,
    tool_call_count = GREATEST(session_history.tool_call_count, EXCLUDED.tool_call_count),
    tool_names = CASE
        WHEN jsonb_array_length(EXCLUDED.tool_names) > jsonb_array_length(session_history.tool_names)
            THEN EXCLUDED.tool_names
        ELSE session_history.tool_names
    END,
    response_cost_usd = COALESCE(
        GREATEST(session_history.response_cost_usd, EXCLUDED.response_cost_usd),
        session_history.response_cost_usd,
        EXCLUDED.response_cost_usd
    ),
    metadata = COALESCE(session_history.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
"""
_AAWM_SESSION_HISTORY_METADATA_KEYS = (
    "trace_name",
    "cc_version",
    "cc_entrypoint",
    "route_tag",
    "reasoning_content_present",
    "thinking_signature_present",
    "usage_reasoning_tokens_reported",
    "usage_reasoning_tokens_source",
    "usage_cache_read_input_tokens",
    "usage_cache_creation_input_tokens",
    "usage_tool_call_count",
    "usage_tool_names",
)
_aawm_session_history_pool: Optional[Any] = None
_aawm_session_history_pool_lock = asyncio.Lock()
_aawm_session_history_schema_ready = False
_aawm_session_history_schema_lock = asyncio.Lock()


def _content_to_text(content: Any) -> str:
    """Convert message content (string or Anthropic content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content) if content else ""


def _clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _get_first_secret_value(secret_names: tuple[str, ...]) -> Optional[str]:
    for secret_name in secret_names:
        value = _clean_secret_string(get_secret_str(secret_name))
        if value:
            return value
    return None


def _normalize_aawm_sslmode(value: Optional[str]) -> Optional[str]:
    cleaned = _clean_secret_string(value)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return "require"
    if lowered in {"0", "false", "no", "off"}:
        return "disable"
    return cleaned


def _build_aawm_dsn() -> Optional[str]:
    host = _get_first_secret_value(_AAWM_DB_HOST_ENV_VARS)
    port = _get_first_secret_value(_AAWM_DB_PORT_ENV_VARS)
    user = _get_first_secret_value(_AAWM_DB_USER_ENV_VARS)
    password = _get_first_secret_value(_AAWM_DB_PASSWORD_ENV_VARS)
    database = _get_first_secret_value(_AAWM_DB_NAME_ENV_VARS)
    sslmode = _normalize_aawm_sslmode(
        _get_first_secret_value(_AAWM_DB_SSLMODE_ENV_VARS)
        or _get_first_secret_value(_AAWM_DB_SSL_BOOL_ENV_VARS)
    )

    has_component_config = any((host, port, user, password, database, sslmode))
    if has_component_config:
        if not host or not user or not database:
            return None

        credentials = quote(user, safe="")
        if password:
            credentials += f":{quote(password, safe='')}"
        dsn = (
            f"postgresql://{credentials}@{host}:{port or '5432'}/"
            f"{quote(database, safe='')}"
        )
        if sslmode:
            dsn += f"?{urlencode({'sslmode': sslmode})}"
        return dsn

    return _get_first_secret_value(_AAWM_DB_URL_ENV_VARS)


def _extract_agent_context_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    tenant_match = _AGENT_TENANT_RE.search(text)
    if tenant_match:
        return tenant_match.group("agent"), tenant_match.group("tenant")

    agent_match = _AGENT_RE.search(text)
    if agent_match:
        return agent_match.group(1), None

    return None, None


def _extract_agent_context(kwargs: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract agent/tenant from request content when present."""
    messages = kwargs.get("messages")
    if messages and isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "system":
                continue
            text = _content_to_text(message.get("content", ""))
            agent_name, tenant_id = _extract_agent_context_from_text(text)
            if agent_name:
                return agent_name, tenant_id

    system_direct = kwargs.get("system")
    if system_direct:
        text = _content_to_text(system_direct)
        agent_name, tenant_id = _extract_agent_context_from_text(text)
        if agent_name:
            return agent_name, tenant_id

    payload = kwargs.get("passthrough_logging_payload")
    if isinstance(payload, dict):
        request_body = payload.get("request_body")
        if isinstance(request_body, dict):
            system = request_body.get("system")
            if system:
                text = _content_to_text(system)
                agent_name, tenant_id = _extract_agent_context_from_text(text)
                if agent_name:
                    return agent_name, tenant_id

            pt_messages = request_body.get("messages")
            if pt_messages and isinstance(pt_messages, list):
                for msg in pt_messages[:3]:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "user":
                        continue
                    text = _content_to_text(msg.get("content", ""))
                    agent_name, tenant_id = _extract_agent_context_from_text(text)
                    if agent_name:
                        return agent_name, tenant_id
                    break

    return None, None


def _extract_agent_name(kwargs: Dict[str, Any]) -> str:
    agent_name, _tenant_id = _extract_agent_context(kwargs)
    return agent_name or _DEFAULT_AGENT


def _ensure_mutable_headers(kwargs: Dict[str, Any]) -> dict:
    """Ensure proxy_server_request.headers is a mutable dict."""
    litellm_params = kwargs.get("litellm_params") or {}
    psr = litellm_params.get("proxy_server_request") or {}
    headers = psr.get("headers")

    if headers is None:
        return {}

    if not isinstance(headers, dict):
        headers = dict(headers)
        psr["headers"] = headers

    return headers


def _ensure_mutable_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    litellm_params["metadata"] = metadata
    kwargs["litellm_params"] = litellm_params
    return metadata


def _merge_tags(metadata: Dict[str, Any], tags_to_add: List[str]) -> None:
    existing_tags = metadata.get("tags") or []
    if not isinstance(existing_tags, list):
        existing_tags = []

    merged_tags = list(existing_tags)
    for tag in tags_to_add:
        if tag and tag not in merged_tags:
            merged_tags.append(tag)
    metadata["tags"] = merged_tags


def _sync_standard_logging_object(kwargs: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    standard_logging_object = kwargs.get("standard_logging_object")
    if not isinstance(standard_logging_object, dict):
        return

    standard_logging_metadata = standard_logging_object.get("metadata")
    if not isinstance(standard_logging_metadata, dict):
        standard_logging_metadata = {}
    standard_logging_metadata.update(metadata)
    standard_logging_object["metadata"] = standard_logging_metadata

    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    existing_request_tags = standard_logging_object.get("request_tags") or []
    if not isinstance(existing_request_tags, list):
        existing_request_tags = []

    merged_request_tags = list(existing_request_tags)
    for tag in tags:
        if isinstance(tag, str) and tag and tag not in merged_request_tags:
            merged_request_tags.append(tag)
    standard_logging_object["request_tags"] = merged_request_tags
    kwargs["standard_logging_object"] = standard_logging_object


def _maybe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_first_response_message(result: Any) -> Any:
    choices = _maybe_get(result, "choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return None

    first_choice = choices[0]
    message = _maybe_get(first_choice, "message")
    if message is not None:
        return message
    return _maybe_get(first_choice, "delta")


def _extract_provider_specific_fields(message: Any) -> Dict[str, Any]:
    provider_specific_fields = _maybe_get(message, "provider_specific_fields")
    if isinstance(provider_specific_fields, dict):
        return provider_specific_fields
    return {}


def _extract_reasoning_content(message: Any, thinking_blocks: List[dict]) -> str:
    reasoning_content = _maybe_get(message, "reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content

    thinking_parts: List[str] = []
    for block in thinking_blocks:
        thinking_text = _maybe_get(block, "thinking")
        if isinstance(thinking_text, str) and thinking_text:
            thinking_parts.append(thinking_text)
    return "\n".join(thinking_parts)


def _extract_thinking_blocks(message: Any) -> List[dict]:
    thinking_blocks = _maybe_get(message, "thinking_blocks")
    if not isinstance(thinking_blocks, list):
        provider_specific_fields = _extract_provider_specific_fields(message)
        thinking_blocks = provider_specific_fields.get("thinking_blocks")
    if not isinstance(thinking_blocks, list):
        return []
    return [block for block in thinking_blocks if isinstance(block, dict)]


def _normalize_base64_text(value: str) -> str:
    return "".join(value.split())


def _decode_base64_bytes(value: str) -> bytes:
    normalized_value = _normalize_base64_text(value)
    padding = (-len(normalized_value)) % 4
    if padding:
        normalized_value += "=" * padding
    return base64.b64decode(normalized_value)


def _short_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:12]


def _format_langfuse_span_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_langfuse_span(
    metadata: Dict[str, Any],
    *,
    name: str,
    span_metadata: Optional[Dict[str, Any]] = None,
    input_data: Any = None,
    output_data: Any = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
    existing_spans = metadata.get("langfuse_spans") or []
    if not isinstance(existing_spans, list):
        existing_spans = []

    span_descriptor: Dict[str, Any] = {"name": name}
    if input_data is not None:
        span_descriptor["input"] = input_data
    if output_data is not None:
        span_descriptor["output"] = output_data
    if span_metadata:
        span_descriptor["metadata"] = span_metadata
    if start_time is not None:
        span_descriptor["start_time"] = _format_langfuse_span_timestamp(start_time)
    if end_time is not None:
        span_descriptor["end_time"] = _format_langfuse_span_timestamp(end_time)

    existing_spans.append(span_descriptor)
    metadata["langfuse_spans"] = existing_spans


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _safe_json_load(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _normalize_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime_value(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            normalized = stripped.replace("Z", "+00:00")
            return _normalize_datetime(datetime.fromisoformat(normalized))
        except ValueError:
            return None
    return None


def _extract_usage_object(kwargs: Dict[str, Any], result: Any) -> Any:
    usage_obj = _maybe_get(result, "usage")
    if usage_obj is not None:
        return usage_obj

    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        response = standard_logging_object.get("response")
        if isinstance(response, dict) and response.get("usage") is not None:
            return response["usage"]

        metadata = standard_logging_object.get("metadata")
        if isinstance(metadata, dict) and metadata.get("usage_object") is not None:
            return metadata["usage_object"]

    return None


def _extract_prompt_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "prompt_tokens"))
        or _safe_int(_maybe_get(usage_obj, "input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "input"))
        or 0
    )


def _extract_completion_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "completion_tokens"))
        or _safe_int(_maybe_get(usage_obj, "output_tokens"))
        or _safe_int(_maybe_get(usage_obj, "candidatesTokenCount"))
        or 0
    )


def _extract_total_tokens(usage_obj: Any, prompt_tokens: int, completion_tokens: int) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "total_tokens"))
        or _safe_int(_maybe_get(usage_obj, "totalTokenCount"))
        or (prompt_tokens + completion_tokens)
    )


def _extract_prompt_tokens_details(usage_obj: Any) -> Any:
    return _first_non_none(
        _maybe_get(usage_obj, "prompt_tokens_details"),
        _maybe_get(usage_obj, "input_tokens_details"),
        _maybe_get(usage_obj, "promptTokensDetails"),
        _maybe_get(usage_obj, "inputTokensDetails"),
    )


def _extract_completion_tokens_details(usage_obj: Any) -> Any:
    return _first_non_none(
        _maybe_get(usage_obj, "completion_tokens_details"),
        _maybe_get(usage_obj, "output_tokens_details"),
        _maybe_get(usage_obj, "completionTokensDetails"),
        _maybe_get(usage_obj, "outputTokensDetails"),
        _maybe_get(usage_obj, "responseTokensDetails"),
        _maybe_get(usage_obj, "candidatesTokensDetails"),
    )


def _extract_cache_read_input_tokens(usage_obj: Any) -> int:
    prompt_tokens_details = _extract_prompt_tokens_details(usage_obj)
    return (
        _safe_int(_maybe_get(usage_obj, "cache_read_input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheReadInputTokens"))
        or _safe_int(_maybe_get(usage_obj, "cachedContentTokenCount"))
        or _safe_int(_maybe_get(prompt_tokens_details, "cached_tokens"))
        or _safe_int(_maybe_get(prompt_tokens_details, "cachedTokens"))
        or 0
    )


def _extract_cache_creation_input_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "cache_creation_input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheWriteInputTokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheWriteInputTokenCount"))
        or _safe_int(_maybe_get(usage_obj, "cacheCreationInputTokens"))
        or 0
    )


def _extract_reported_reasoning_tokens(usage_obj: Any) -> Optional[int]:
    completion_tokens_details = _extract_completion_tokens_details(usage_obj)
    return _first_non_none(
        _safe_int(_maybe_get(usage_obj, "reasoning_tokens")),
        _safe_int(_maybe_get(usage_obj, "reasoningTokens")),
        _safe_int(_maybe_get(usage_obj, "reasoning_token_count")),
        _safe_int(_maybe_get(usage_obj, "thoughtsTokenCount")),
        _safe_int(_maybe_get(completion_tokens_details, "reasoning_tokens")),
        _safe_int(_maybe_get(completion_tokens_details, "reasoningTokens")),
    )


def _estimate_reasoning_tokens(model: str, reasoning_text: str) -> Optional[int]:
    stripped_reasoning = reasoning_text.strip()
    if not stripped_reasoning:
        return None

    try:
        import litellm

        return litellm.token_counter(
            model=model or "",
            text=stripped_reasoning,
            count_response_tokens=True,
        )
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to estimate reasoning tokens for model=%s: %s",
            model,
            exc,
        )
        return None


def _extract_tool_call_info(message: Any) -> Tuple[int, List[str]]:
    raw_tool_calls = _maybe_get(message, "tool_calls")
    if isinstance(raw_tool_calls, list):
        tool_names: List[str] = []
        for tool_call in raw_tool_calls:
            function_obj = _maybe_get(tool_call, "function")
            tool_name = _maybe_get(function_obj, "name") or _maybe_get(
                tool_call, "name"
            )
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        return len(raw_tool_calls), tool_names

    content = _maybe_get(message, "content")
    if isinstance(content, list):
        tool_names = []
        tool_call_count = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {"tool_use", "function_call"}:
                continue
            tool_call_count += 1
            tool_name = block.get("name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        if tool_call_count:
            return tool_call_count, tool_names

    provider_specific_fields = _extract_provider_specific_fields(message)
    provider_tool_calls = provider_specific_fields.get("tool_calls")
    if isinstance(provider_tool_calls, list):
        tool_names = []
        for tool_call in provider_tool_calls:
            tool_name = _maybe_get(_maybe_get(tool_call, "function"), "name") or _maybe_get(
                tool_call, "name"
            )
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        return len(provider_tool_calls), tool_names

    return 0, []


def _extract_response_output_tool_call_info(result: Any) -> Tuple[int, List[str]]:
    output_items = result if isinstance(result, list) else _maybe_get(result, "output")
    if not isinstance(output_items, list):
        return 0, []

    tool_call_count = 0
    tool_names: List[str] = []
    for item in output_items:
        item_type = _maybe_get(item, "type")
        if item_type not in {"function_call", "apply_patch_call"}:
            continue
        tool_call_count += 1
        tool_name = _maybe_get(item, "name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            if item_type == "apply_patch_call":
                tool_name = "apply_patch"
        if isinstance(tool_name, str) and tool_name.strip():
            tool_names.append(tool_name)

    return tool_call_count, tool_names


def _extract_session_id(kwargs: Dict[str, Any]) -> Optional[str]:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    standard_metadata = standard_logging_object.get("metadata") or {}

    for candidate in (
        litellm_params.get("litellm_session_id"),
        kwargs.get("litellm_session_id"),
        metadata.get("session_id"),
        standard_metadata.get("session_id"),
        standard_logging_object.get("session_id"),
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate)
    return None


def _extract_trace_id(kwargs: Dict[str, Any]) -> Optional[str]:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    standard_logging_object = kwargs.get("standard_logging_object") or {}

    for candidate in (
        litellm_params.get("litellm_trace_id"),
        kwargs.get("litellm_trace_id"),
        metadata.get("trace_id"),
        standard_logging_object.get("trace_id"),
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate)
    return None


def _infer_usage_breakout_provider_prefix(
    kwargs: Dict[str, Any], metadata: Dict[str, Any]
) -> Optional[str]:
    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_family_lower = route_family.lower()
        if route_family_lower == "codex_responses":
            return "codex"
        if "gemini" in route_family_lower:
            return "gemini"

    provider = kwargs.get("custom_llm_provider")
    if isinstance(provider, str) and provider.strip():
        provider_lower = provider.lower()
        if provider_lower == "gemini":
            return "gemini"

    model = kwargs.get("model")
    if isinstance(model, str) and model.strip():
        model_lower = model.lower()
        if "gemini" in model_lower:
            return "gemini"
        if "codex" in model_lower:
            return "codex"

    return None


def _enrich_usage_breakout_metadata(kwargs: Dict[str, Any], result: Any) -> None:
    metadata = _ensure_mutable_metadata(kwargs)
    provider_prefix = _infer_usage_breakout_provider_prefix(kwargs, metadata)
    if provider_prefix is None:
        return

    usage_obj = _extract_usage_object(kwargs, result)
    if usage_obj is None:
        return

    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_obj)
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)

    message = _extract_first_response_message(result)
    tool_call_count, tool_names = _extract_tool_call_info(message)
    if tool_call_count == 0:
        tool_call_count, tool_names = _extract_response_output_tool_call_info(result)

    metadata["usage_cache_read_input_tokens"] = cache_read_input_tokens
    metadata["usage_cache_creation_input_tokens"] = cache_creation_input_tokens
    metadata["usage_tool_call_count"] = tool_call_count
    metadata["usage_tool_names"] = tool_names
    metadata[f"{provider_prefix}_cache_read_input_tokens"] = cache_read_input_tokens
    metadata[f"{provider_prefix}_cache_creation_input_tokens"] = (
        cache_creation_input_tokens
    )
    metadata[f"{provider_prefix}_tool_call_count"] = tool_call_count
    metadata[f"{provider_prefix}_tool_names"] = tool_names

    if reported_reasoning_tokens is not None:
        metadata["usage_reasoning_tokens_reported"] = reported_reasoning_tokens
        metadata["usage_reasoning_tokens_source"] = "provider_reported"
        metadata[f"{provider_prefix}_reasoning_tokens_reported"] = (
            reported_reasoning_tokens
        )

    tags_to_add = [f"{provider_prefix}-usage-breakout"]
    if reported_reasoning_tokens is not None:
        tags_to_add.extend(
            ["reasoning-tokens-reported", f"{provider_prefix}-reasoning-tokens-reported"]
        )
    if cache_read_input_tokens > 0:
        tags_to_add.extend(
            ["cache-read-input-tokens", f"{provider_prefix}-cache-read-input-tokens"]
        )
    if cache_creation_input_tokens > 0:
        tags_to_add.extend(
            [
                "cache-creation-input-tokens",
                f"{provider_prefix}-cache-creation-input-tokens",
            ]
        )
    if tool_call_count > 0:
        tags_to_add.extend(["tool-calls-present", f"{provider_prefix}-tool-calls-present"])
    _merge_tags(metadata, tags_to_add)

    _append_langfuse_span(
        metadata,
        name=f"{provider_prefix}.usage_breakout",
        span_metadata={
            "reported_reasoning_tokens": reported_reasoning_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "tool_call_count": tool_call_count,
            "tool_names": tool_names,
        },
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
    )


def _extract_trace_id_from_spend_log_row(spend_log_row: Dict[str, Any]) -> Tuple[Optional[str], str]:
    metadata = _safe_json_load(spend_log_row.get("metadata"), {})
    request_body = _safe_json_load(spend_log_row.get("proxy_server_request"), {})

    for candidate in (
        metadata.get("trace_id") if isinstance(metadata, dict) else None,
        request_body.get("trace_id") if isinstance(request_body, dict) else None,
        spend_log_row.get("session_id"),
        spend_log_row.get("request_id"),
    ):
        if candidate is not None and str(candidate).strip():
            candidate_str = str(candidate).strip()
            if candidate is spend_log_row.get("session_id"):
                return candidate_str, "legacy_spend_log_session_field"
            if candidate is spend_log_row.get("request_id"):
                return candidate_str, "request_id_fallback"
            return candidate_str, "metadata_or_request_body"

    return None, "missing"


def _coerce_nested_session_id(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        session_candidate = value.get("session_id") or value.get("sessionId")
        if session_candidate is not None and str(session_candidate).strip():
            return str(session_candidate).strip()
        return None

    if isinstance(value, str):
        parsed = _safe_json_load(value, None)
        if parsed is not None:
            return _coerce_nested_session_id(parsed)
        if value.strip():
            return value.strip()

    return None


def _extract_session_id_from_spend_log_row(
    spend_log_row: Dict[str, Any],
) -> Tuple[Optional[str], str]:
    metadata = _safe_json_load(spend_log_row.get("metadata"), {})
    request_body = _safe_json_load(spend_log_row.get("proxy_server_request"), {})
    response_body = _safe_json_load(spend_log_row.get("response"), {})

    if isinstance(request_body, dict):
        metadata_payload = request_body.get("metadata")
        if isinstance(metadata_payload, dict):
            session_candidate = metadata_payload.get("session_id")
            if session_candidate is not None and str(session_candidate).strip():
                return str(session_candidate).strip(), "request_body.metadata.session_id"

            user_id_payload = metadata_payload.get("user_id")
            nested_session_id = _coerce_nested_session_id(user_id_payload)
            if nested_session_id:
                return nested_session_id, "request_body.metadata.user_id.session_id"

        top_level_session_id = request_body.get("session_id")
        if top_level_session_id is not None and str(top_level_session_id).strip():
            return str(top_level_session_id).strip(), "request_body.session_id"

        request_payload = request_body.get("request")
        if isinstance(request_payload, dict):
            request_session_id = request_payload.get("session_id")
            if request_session_id is not None and str(request_session_id).strip():
                return str(request_session_id).strip(), "request_body.request.session_id"

    if isinstance(metadata, dict):
        for key in ("session_id", "sessionId"):
            session_candidate = metadata.get(key)
            if session_candidate is not None and str(session_candidate).strip():
                return str(session_candidate).strip(), f"metadata.{key}"

    if isinstance(response_body, dict):
        for key in ("session_id", "sessionId"):
            session_candidate = response_body.get(key)
            if session_candidate is not None and str(session_candidate).strip():
                return str(session_candidate).strip(), f"response.{key}"

    legacy_session_field = spend_log_row.get("session_id")
    if legacy_session_field is not None and str(legacy_session_field).strip():
        return str(legacy_session_field).strip(), "legacy_spend_log_session_field"

    return None, "missing"


def _coerce_spend_log_request_tags(value: Any) -> List[str]:
    parsed = _safe_json_load(value, value)
    if not isinstance(parsed, list):
        return []
    return [str(tag) for tag in parsed if isinstance(tag, str) and tag.strip()]


def _synthesize_result_from_spend_log_row(
    spend_log_row: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    result = _safe_json_load(spend_log_row.get("response"), {})
    if not isinstance(result, dict):
        result = {"response": result}

    usage_object = metadata.get("usage_object")
    if not isinstance(usage_object, dict):
        usage_object = {}

    if not isinstance(result.get("usage"), dict):
        reconstructed_usage = dict(usage_object)
        reconstructed_usage.setdefault(
            "prompt_tokens", _safe_int(spend_log_row.get("prompt_tokens")) or 0
        )
        reconstructed_usage.setdefault(
            "completion_tokens", _safe_int(spend_log_row.get("completion_tokens")) or 0
        )
        reconstructed_usage.setdefault(
            "total_tokens", _safe_int(spend_log_row.get("total_tokens")) or 0
        )
        result["usage"] = reconstructed_usage

    return result


def _build_backfill_kwargs_from_spend_log_row(
    spend_log_row: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    request_id = spend_log_row.get("request_id")
    model = spend_log_row.get("model")
    if request_id is None or not str(request_id).strip():
        return None
    if model is None or not str(model).strip():
        return None

    metadata = _safe_json_load(spend_log_row.get("metadata"), {})
    if not isinstance(metadata, dict):
        metadata = {}
    request_body = _safe_json_load(spend_log_row.get("proxy_server_request"), {})
    if not isinstance(request_body, dict):
        request_body = {}
    request_tags = _coerce_spend_log_request_tags(spend_log_row.get("request_tags"))

    session_id, session_id_source = _extract_session_id_from_spend_log_row(spend_log_row)
    trace_id, trace_id_source = _extract_trace_id_from_spend_log_row(spend_log_row)

    litellm_metadata: Dict[str, Any] = dict(metadata)
    if session_id:
        litellm_metadata["session_id"] = session_id
    if trace_id:
        litellm_metadata["trace_id"] = trace_id
    if spend_log_row.get("model_group"):
        litellm_metadata["model_group"] = spend_log_row.get("model_group")

    standard_logging_metadata = dict(litellm_metadata)
    if isinstance(metadata.get("usage_object"), dict):
        standard_logging_metadata["usage_object"] = metadata.get("usage_object")

    standard_logging_object: Dict[str, Any] = {
        "metadata": standard_logging_metadata,
        "request_tags": list(request_tags),
        "trace_id": trace_id,
        "model": str(model),
        "model_group": spend_log_row.get("model_group"),
        "response_cost": _safe_float(spend_log_row.get("spend")),
        "prompt_tokens": _safe_int(spend_log_row.get("prompt_tokens")) or 0,
        "completion_tokens": _safe_int(spend_log_row.get("completion_tokens")) or 0,
        "total_tokens": _safe_int(spend_log_row.get("total_tokens")) or 0,
    }

    kwargs: Dict[str, Any] = {
        "model": str(model),
        "custom_llm_provider": spend_log_row.get("custom_llm_provider"),
        "call_type": spend_log_row.get("call_type"),
        "litellm_call_id": str(request_id),
        "litellm_trace_id": trace_id,
        "litellm_session_id": session_id,
        "litellm_params": {
            "metadata": litellm_metadata,
            "litellm_trace_id": trace_id,
            "litellm_session_id": session_id,
            "proxy_server_request": {"body": request_body},
        },
        "standard_logging_object": standard_logging_object,
        "passthrough_logging_payload": {"request_body": request_body},
        "response_cost": _safe_float(spend_log_row.get("spend")),
    }

    messages = _safe_json_load(spend_log_row.get("messages"), None)
    if isinstance(messages, list):
        kwargs["messages"] = messages

    system = request_body.get("system")
    if system is not None:
        kwargs["system"] = system

    result = _synthesize_result_from_spend_log_row(spend_log_row, metadata)

    provenance = {
        "session_id_source": session_id_source,
        "trace_id_source": trace_id_source,
        "source_request_id": str(request_id),
        "source_spend_log_session_field": (
            str(spend_log_row.get("session_id")).strip()
            if spend_log_row.get("session_id") is not None
            and str(spend_log_row.get("session_id")).strip()
            else None
        ),
    }

    return kwargs, result, provenance


def _build_session_history_record_from_spend_log_row(
    spend_log_row: Dict[str, Any],
    *,
    backfill_run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    prepared = _build_backfill_kwargs_from_spend_log_row(spend_log_row)
    if prepared is None:
        return None

    kwargs, result, provenance = prepared
    kwargs, result = _enrich_trace_name_and_provider_metadata(kwargs, result)

    record = _build_session_history_record(
        kwargs=kwargs,
        result=result,
        start_time=_parse_datetime_value(spend_log_row.get("startTime")),
        end_time=_parse_datetime_value(spend_log_row.get("endTime")),
    )
    if record is None:
        return None

    metadata = record.get("metadata") or {}
    metadata.update(
        {
            "backfilled": True,
            "backfill_source": "LiteLLM_SpendLogs",
            "backfill_run_id": backfill_run_id,
            "source_request_id": provenance["source_request_id"],
            "source_spend_log_session_field": provenance["source_spend_log_session_field"],
            "session_id_source": provenance["session_id_source"],
            "trace_id_source": provenance["trace_id_source"],
            "source_status": spend_log_row.get("status"),
        }
    )
    if spend_log_row.get("agent_id") is not None:
        metadata["source_agent_id"] = spend_log_row.get("agent_id")
    record["metadata"] = metadata
    record["trace_id"] = kwargs.get("litellm_trace_id") or record.get("trace_id")
    return record


def _derive_langfuse_trace_tags_from_spend_log_row(
    spend_log_row: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    prepared = _build_backfill_kwargs_from_spend_log_row(spend_log_row)
    if prepared is None:
        return None, []

    kwargs, result, _provenance = prepared
    kwargs, result = _enrich_trace_name_and_provider_metadata(kwargs, result)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    request_tags = standard_logging_object.get("request_tags") or []
    if not isinstance(request_tags, list):
        request_tags = []
    trace_id = kwargs.get("litellm_trace_id")
    if trace_id is not None and str(trace_id).strip():
        trace_id = str(trace_id).strip()
    else:
        trace_id = None
    return trace_id, [tag for tag in request_tags if isinstance(tag, str) and tag.strip()]


def _serialize_searchable_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _extract_agent_context_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    for candidate in (
        observation.get("input"),
        trace.get("input"),
        observation.get("output"),
        trace.get("output"),
    ):
        agent_name, tenant_id = _extract_agent_context_from_text(
            _serialize_searchable_text(candidate)
        )
        if agent_name:
            return agent_name, tenant_id

    trace_name = trace.get("name")
    if isinstance(trace_name, str) and trace_name.startswith("claude-code."):
        return trace_name.split(".", 1)[1], None

    return None, None


def _extract_langfuse_session_id(
    trace: Dict[str, Any],
    observation_metadata: Dict[str, Any],
) -> Tuple[Optional[str], str]:
    for candidate in (
        trace.get("sessionId"),
        trace.get("session_id"),
        observation_metadata.get("session_id"),
        _coerce_nested_session_id(observation_metadata.get("user_id")),
        _coerce_nested_session_id(observation_metadata.get("user_api_key_end_user_id")),
    ):
        if candidate is not None and str(candidate).strip():
            if candidate == trace.get("sessionId"):
                return str(candidate).strip(), "trace.sessionId"
            if candidate == trace.get("session_id"):
                return str(candidate).strip(), "trace.session_id"
            if candidate == observation_metadata.get("session_id"):
                return str(candidate).strip(), "observation.metadata.session_id"
            if candidate == _coerce_nested_session_id(observation_metadata.get("user_id")):
                return str(candidate).strip(), "observation.metadata.user_id.session_id"
            return (
                str(candidate).strip(),
                "observation.metadata.user_api_key_end_user_id.session_id",
            )

    return None, "missing"


def _build_usage_object_from_langfuse_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    metadata = observation.get("metadata")
    usage = observation.get("usage")
    usage_details = observation.get("usageDetails")

    usage_object: Dict[str, Any] = {}
    if isinstance(metadata, dict) and isinstance(metadata.get("usage_object"), dict):
        usage_object.update(metadata["usage_object"])
    if isinstance(usage, dict):
        usage_object.update(usage)
    if isinstance(usage_details, dict):
        usage_object.update(usage_details)

    prompt_tokens = _safe_int(
        _first_non_none(
            observation.get("promptTokens"),
            observation.get("inputTokens"),
            usage_object.get("prompt_tokens"),
            usage_object.get("input_tokens"),
            usage_object.get("input"),
        )
    )
    completion_tokens = _safe_int(
        _first_non_none(
            observation.get("completionTokens"),
            observation.get("outputTokens"),
            usage_object.get("completion_tokens"),
            usage_object.get("output_tokens"),
            usage_object.get("output"),
        )
    )
    total_tokens = _safe_int(
        _first_non_none(
            observation.get("totalTokens"),
            usage_object.get("total_tokens"),
            usage_object.get("total"),
        )
    )

    if prompt_tokens is not None:
        usage_object["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage_object["completion_tokens"] = completion_tokens
        usage_object.setdefault("output_tokens", completion_tokens)
    if total_tokens is not None:
        usage_object["total_tokens"] = total_tokens

    prompt_tokens_details = _extract_prompt_tokens_details(usage_object)
    if isinstance(prompt_tokens_details, dict):
        usage_object.setdefault("prompt_tokens_details", prompt_tokens_details)

    completion_tokens_details = _extract_completion_tokens_details(usage_object)
    if isinstance(completion_tokens_details, dict):
        usage_object.setdefault(
            "completion_tokens_details", completion_tokens_details
        )

    cache_read_tokens = _safe_int(usage_object.get("cache_read_input_tokens"))
    if cache_read_tokens is None:
        cache_read_tokens = _safe_int(usage_object.get("cachedContentTokenCount"))
    cache_creation_tokens = _safe_int(usage_object.get("cache_creation_input_tokens"))
    if cache_read_tokens is not None:
        usage_object["cache_read_input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None:
        usage_object["cache_creation_input_tokens"] = cache_creation_tokens
    if usage_object.get("reasoning_tokens") is None:
        thoughts_token_count = _safe_int(usage_object.get("thoughtsTokenCount"))
        if thoughts_token_count is not None:
            usage_object["reasoning_tokens"] = thoughts_token_count

    return usage_object


def _extract_first_langfuse_response_message(output_payload: Any) -> Any:
    if isinstance(output_payload, dict):
        if isinstance(output_payload.get("choices"), list):
            return _extract_first_response_message(output_payload)
        if isinstance(output_payload.get("message"), dict):
            return output_payload["message"]
        if any(
            key in output_payload
            for key in ("content", "tool_calls", "reasoning_content", "thinking_blocks")
        ):
            return output_payload
    return None


def _infer_provider_from_langfuse_observation(
    observation: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_lower = route_family.lower()
        if "anthropic" in route_lower:
            return "anthropic"
        if "gemini" in route_lower:
            return "gemini"
        if "codex" in route_lower:
            return "openai"
        if "openai" in route_lower:
            return "openai"

    request_route = metadata.get("user_api_key_request_route")
    if isinstance(request_route, str) and request_route.strip():
        route_lower = request_route.lower()
        if route_lower.startswith("/anthropic/"):
            return "anthropic"
        if "gemini" in route_lower:
            return "gemini"
        if route_lower.startswith("/v1/"):
            return "openai"

    api_base = (
        metadata.get("api_base")
        or _maybe_get(metadata.get("hidden_params"), "api_base")
        or observation.get("apiBase")
    )
    if isinstance(api_base, str) and api_base.strip():
        api_base_lower = api_base.lower()
        if "anthropic.com" in api_base_lower:
            return "anthropic"
        if "googleapis.com" in api_base_lower or "generativelanguage" in api_base_lower:
            return "gemini"
        if "openai.com" in api_base_lower:
            return "openai"

    model = observation.get("model")
    if isinstance(model, str) and model.strip():
        model_lower = model.lower()
        if "claude" in model_lower:
            return "anthropic"
        if "gemini" in model_lower:
            return "gemini"
        if (
            model_lower.startswith("gpt")
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
            or model_lower.startswith("o4")
            or "codex" in model_lower
            or "text-embedding" in model_lower
        ):
            return "openai"

    return None


def _derive_request_tags_from_langfuse_metadata(metadata: Dict[str, Any]) -> List[str]:
    request_tags = metadata.get("tags")
    normalized_tags = [
        str(tag) for tag in request_tags if isinstance(tag, str) and tag.strip()
    ] if isinstance(request_tags, list) else []

    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        normalized_tags.append(f"route:{route_family.strip()}")

    billing_header_fields = metadata.get("anthropic_billing_header_fields")
    if isinstance(billing_header_fields, dict) and billing_header_fields:
        normalized_tags.append("anthropic-billing-header")
        for key, value in billing_header_fields.items():
            if isinstance(key, str) and key.strip():
                normalized_tags.append(f"anthropic-billing-header-key:{key}")
                if value is not None and str(value).strip():
                    normalized_tags.append(
                        f"anthropic-billing-header:{key}={str(value).strip()}"
                    )

    thinking_type = metadata.get("claude_thinking_type")
    if isinstance(thinking_type, str) and thinking_type.strip():
        normalized_tags.append(f"claude-thinking-type:{thinking_type}")
        normalized_tags.append(f"thinking-type:{thinking_type}")

    effort = metadata.get("claude_effort")
    if isinstance(effort, str) and effort.strip():
        normalized_tags.append(f"claude-effort:{effort}")
        normalized_tags.append(f"effort:{effort}")

    if metadata.get("thinking_signature_present") is True:
        normalized_tags.append("thinking-signature-present")
    if metadata.get("claude_thinking_signature_present") is True:
        normalized_tags.append("claude-thinking-signature")
    if metadata.get("gemini_thought_signature_present") is True:
        normalized_tags.append("gemini-thought-signature")
    if metadata.get("thinking_signature_decoded") is True:
        normalized_tags.append("thinking-signature-decoded")
    if metadata.get("claude_thinking_signature_decoded") is True:
        normalized_tags.append("claude-thinking-decoded")
    if metadata.get("reasoning_content_present") is True:
        normalized_tags.append("reasoning-present")
    elif metadata.get("reasoning_content_present") is False:
        normalized_tags.append("reasoning-empty")
    if metadata.get("thinking_blocks_present") is True:
        normalized_tags.append("thinking-blocks-present")
    elif metadata.get("thinking_blocks_present") is False:
        normalized_tags.append("thinking-blocks-empty")

    return sorted({tag for tag in normalized_tags if isinstance(tag, str) and tag.strip()})


def _build_session_history_record_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    *,
    backfill_run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if observation.get("type") != "GENERATION":
        return None

    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    session_id, session_id_source = _extract_langfuse_session_id(trace, metadata)
    if not session_id:
        return None

    trace_id = trace.get("id") or observation.get("traceId")
    if trace_id is not None and str(trace_id).strip():
        trace_id = str(trace_id).strip()
    else:
        trace_id = None

    usage_object = _build_usage_object_from_langfuse_observation(observation)
    prompt_tokens = _extract_prompt_tokens(usage_object)
    completion_tokens = _extract_completion_tokens(usage_object)
    total_tokens = _extract_total_tokens(usage_object, prompt_tokens, completion_tokens)
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_object)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_object)
    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_object)

    output_payload = observation.get("output")
    message = _extract_first_langfuse_response_message(output_payload)
    thinking_blocks = _extract_thinking_blocks(message) if message is not None else []
    reasoning_text = (
        _extract_reasoning_content(message, thinking_blocks)
        if message is not None
        else ""
    )

    reasoning_present = bool(
        (isinstance(reasoning_text, str) and reasoning_text.strip())
        or thinking_blocks
        or metadata.get("reasoning_content_present")
        or (reported_reasoning_tokens and reported_reasoning_tokens > 0)
    )
    estimated_reasoning_tokens = None
    reasoning_tokens_source: Optional[str] = None
    if reported_reasoning_tokens is not None:
        reasoning_tokens_source = "provider_reported"
    elif reasoning_present:
        estimated_reasoning_tokens = _estimate_reasoning_tokens(
            model=str(observation.get("model") or ""),
            reasoning_text=reasoning_text,
        )
        reasoning_tokens_source = (
            "estimated_from_reasoning_text"
            if estimated_reasoning_tokens is not None
            else "not_available"
        )

    tool_call_count, tool_names = _extract_tool_call_info(message)
    if tool_call_count == 0:
        output_tool_call_count, output_tool_names = _extract_response_output_tool_call_info(
            output_payload
        )
        if output_tool_call_count > 0:
            tool_call_count, tool_names = output_tool_call_count, output_tool_names
    if tool_call_count == 0:
        fallback_tool_names = observation.get("toolCallNames") or observation.get(
            "tool_call_names"
        )
        if isinstance(fallback_tool_names, list):
            normalized_tool_names = [
                str(tool_name)
                for tool_name in fallback_tool_names
                if isinstance(tool_name, str) and tool_name.strip()
            ]
            if normalized_tool_names:
                tool_call_count = len(normalized_tool_names)
                tool_names = normalized_tool_names
    if tool_call_count == 0:
        metadata_tool_names = metadata.get("usage_tool_names")
        if isinstance(metadata_tool_names, list):
            normalized_tool_names = [
                str(tool_name)
                for tool_name in metadata_tool_names
                if isinstance(tool_name, str) and tool_name.strip()
            ]
            if normalized_tool_names:
                tool_call_count = len(normalized_tool_names)
                tool_names = normalized_tool_names
    agent_name, tenant_id = _extract_agent_context_from_langfuse_trace_observation(
        trace,
        observation,
    )
    request_tags = _derive_request_tags_from_langfuse_metadata(metadata)
    provider = _infer_provider_from_langfuse_observation(observation, metadata)

    history_metadata = _build_session_history_metadata(
        metadata=metadata,
        request_tags=request_tags,
        tenant_id=tenant_id,
    )
    history_metadata.update(
        {
            "backfilled": True,
            "backfill_source": "LangfuseTraces",
            "backfill_run_id": backfill_run_id,
            "source_trace_id": trace_id,
            "source_observation_id": observation.get("id"),
            "session_id_source": session_id_source,
            "trace_id_source": "trace.id" if trace_id else "missing",
            "source_trace_environment": trace.get("environment"),
            "source_status": "failure"
            if observation.get("statusMessage")
            else "success",
        }
    )

    return {
        "litellm_call_id": observation.get("id"),
        "session_id": session_id,
        "trace_id": trace_id,
        "provider_response_id": _maybe_get(output_payload, "id"),
        "provider": provider,
        "model": str(observation.get("model") or ""),
        "model_group": metadata.get("model_group"),
        "agent_name": agent_name,
        "tenant_id": tenant_id,
        "call_type": metadata.get("user_api_key_request_route") or observation.get("name"),
        "start_time": _parse_datetime_value(observation.get("startTime")),
        "end_time": _parse_datetime_value(observation.get("endTime")),
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "reasoning_tokens_reported": reported_reasoning_tokens,
        "reasoning_tokens_estimated": estimated_reasoning_tokens,
        "reasoning_tokens_source": reasoning_tokens_source,
        "reasoning_present": reasoning_present,
        "thinking_signature_present": bool(metadata.get("thinking_signature_present")),
        "tool_call_count": tool_call_count,
        "tool_names": tool_names,
        "response_cost_usd": _safe_float(
            _first_non_none(
                _maybe_get(observation.get("costDetails"), "total"),
                observation.get("calculatedTotalCost"),
                metadata.get("litellm_response_cost"),
                trace.get("totalCost"),
            )
        ),
        "metadata": history_metadata,
    }


def _derive_langfuse_trace_tags_from_langfuse_trace(
    trace: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    trace_id = trace.get("id")
    normalized_trace_id = (
        str(trace_id).strip() if trace_id is not None and str(trace_id).strip() else None
    )

    derived_tags: List[str] = []
    existing_trace_tags = trace.get("tags")
    if isinstance(existing_trace_tags, list):
        derived_tags.extend(
            str(tag) for tag in existing_trace_tags if isinstance(tag, str) and tag.strip()
        )

    observations = trace.get("observations")
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, dict) or observation.get("type") != "GENERATION":
                continue
            metadata = observation.get("metadata")
            if not isinstance(metadata, dict):
                continue
            derived_tags.extend(_derive_request_tags_from_langfuse_metadata(metadata))

    return normalized_trace_id, sorted(
        {tag for tag in derived_tags if isinstance(tag, str) and tag.strip()}
    )


def _build_session_history_metadata(
    *,
    metadata: Dict[str, Any],
    request_tags: List[str],
    tenant_id: Optional[str],
) -> Dict[str, Any]:
    history_metadata: Dict[str, Any] = {"request_tags": request_tags}
    if tenant_id:
        history_metadata["tenant_id"] = tenant_id

    for key in _AAWM_SESSION_HISTORY_METADATA_KEYS:
        value = metadata.get(key)
        if value is not None:
            history_metadata[key] = value

    return history_metadata


def _build_session_history_record(
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
) -> Optional[Dict[str, Any]]:
    session_id = _extract_session_id(kwargs)
    if not session_id:
        return None

    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    request_tags = standard_logging_object.get("request_tags") or metadata.get("tags") or []
    if not isinstance(request_tags, list):
        request_tags = []

    usage_obj = _extract_usage_object(kwargs, result)
    prompt_tokens = _extract_prompt_tokens(usage_obj)
    completion_tokens = _extract_completion_tokens(usage_obj)
    total_tokens = _extract_total_tokens(usage_obj, prompt_tokens, completion_tokens)
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)
    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_obj)

    message = _extract_first_response_message(result)
    thinking_blocks = _extract_thinking_blocks(message) if message is not None else []
    reasoning_text = (
        _extract_reasoning_content(message, thinking_blocks)
        if message is not None
        else ""
    )

    reasoning_present = bool(
        (isinstance(reasoning_text, str) and reasoning_text.strip())
        or thinking_blocks
        or metadata.get("reasoning_content_present")
        or (reported_reasoning_tokens and reported_reasoning_tokens > 0)
    )
    estimated_reasoning_tokens = None
    reasoning_tokens_source: Optional[str] = None
    if reported_reasoning_tokens is not None:
        reasoning_tokens_source = "provider_reported"
    elif reasoning_present:
        estimated_reasoning_tokens = _estimate_reasoning_tokens(
            model=str(kwargs.get("model") or ""),
            reasoning_text=reasoning_text,
        )
        reasoning_tokens_source = (
            "estimated_from_reasoning_text"
            if estimated_reasoning_tokens is not None
            else "not_available"
        )

    tool_call_count, tool_names = _extract_tool_call_info(message)
    if tool_call_count == 0:
        output_tool_call_count, output_tool_names = _extract_response_output_tool_call_info(
            result
        )
        if output_tool_call_count > 0:
            tool_call_count, tool_names = output_tool_call_count, output_tool_names
    agent_name, tenant_id = _extract_agent_context(kwargs)

    return {
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "session_id": session_id,
        "trace_id": _extract_trace_id(kwargs),
        "provider_response_id": _maybe_get(result, "id"),
        "provider": kwargs.get("custom_llm_provider"),
        "model": str(kwargs.get("model") or standard_logging_object.get("model") or ""),
        "model_group": metadata.get("model_group") or standard_logging_object.get("model_group"),
        "agent_name": agent_name,
        "tenant_id": tenant_id,
        "call_type": kwargs.get("call_type") or standard_logging_object.get("call_type"),
        "start_time": _normalize_datetime(start_time),
        "end_time": _normalize_datetime(end_time),
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "reasoning_tokens_reported": reported_reasoning_tokens,
        "reasoning_tokens_estimated": estimated_reasoning_tokens,
        "reasoning_tokens_source": reasoning_tokens_source,
        "reasoning_present": reasoning_present,
        "thinking_signature_present": bool(metadata.get("thinking_signature_present")),
        "tool_call_count": tool_call_count,
        "tool_names": tool_names,
        "response_cost_usd": _safe_float(
            _first_non_none(
                kwargs.get("response_cost"),
                standard_logging_object.get("response_cost"),
            )
        ),
        "metadata": _build_session_history_metadata(
            metadata=metadata,
            request_tags=[tag for tag in request_tags if isinstance(tag, str)],
            tenant_id=tenant_id,
        ),
    }


async def _get_aawm_session_history_pool() -> Any:
    global _aawm_session_history_pool

    if _aawm_session_history_pool is not None:
        return _aawm_session_history_pool

    async with _aawm_session_history_pool_lock:
        if _aawm_session_history_pool is not None:
            return _aawm_session_history_pool

        dsn = _build_aawm_dsn()
        if not dsn:
            raise RuntimeError("AAWM session history database configuration is missing")

        try:
            asyncpg = importlib.import_module("asyncpg")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AAWM session history requires asyncpg to be installed"
            ) from exc

        _aawm_session_history_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=4,
            command_timeout=10,
        )
        return _aawm_session_history_pool


async def _ensure_session_history_schema(pool: Any) -> None:
    global _aawm_session_history_schema_ready

    if _aawm_session_history_schema_ready:
        return

    async with _aawm_session_history_schema_lock:
        if _aawm_session_history_schema_ready:
            return

        async with pool.acquire() as conn:
            await conn.execute(_AAWM_SESSION_HISTORY_TABLE_SQL)
            for statement in _AAWM_SESSION_HISTORY_INDEX_STATEMENTS:
                await conn.execute(statement)

        _aawm_session_history_schema_ready = True


async def _persist_session_history_record(record: Dict[str, Any]) -> None:
    pool = await _get_aawm_session_history_pool()
    await _ensure_session_history_schema(pool)

    await pool.execute(
        _AAWM_SESSION_HISTORY_INSERT_SQL,
        record["litellm_call_id"],
        record["session_id"],
        record["trace_id"],
        record["provider_response_id"],
        record["provider"],
        record["model"],
        record["model_group"],
        record["agent_name"],
        record["tenant_id"],
        record["call_type"],
        record["start_time"],
        record["end_time"],
        record["input_tokens"],
        record["output_tokens"],
        record["total_tokens"],
        record["cache_read_input_tokens"],
        record["cache_creation_input_tokens"],
        record["reasoning_tokens_reported"],
        record["reasoning_tokens_estimated"],
        record["reasoning_tokens_source"],
        record["reasoning_present"],
        record["thinking_signature_present"],
        record["tool_call_count"],
        json.dumps(record["tool_names"]),
        record["response_cost_usd"],
        json.dumps(record["metadata"]),
    )


async def _persist_session_history_records(records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    pool = await _get_aawm_session_history_pool()
    await _ensure_session_history_schema(pool)

    payloads = [
        (
            record["litellm_call_id"],
            record["session_id"],
            record["trace_id"],
            record["provider_response_id"],
            record["provider"],
            record["model"],
            record["model_group"],
            record["agent_name"],
            record["tenant_id"],
            record["call_type"],
            record["start_time"],
            record["end_time"],
            record["input_tokens"],
            record["output_tokens"],
            record["total_tokens"],
            record["cache_read_input_tokens"],
            record["cache_creation_input_tokens"],
            record["reasoning_tokens_reported"],
            record["reasoning_tokens_estimated"],
            record["reasoning_tokens_source"],
            record["reasoning_present"],
            record["thinking_signature_present"],
            record["tool_call_count"],
            json.dumps(record["tool_names"]),
            record["response_cost_usd"],
            json.dumps(record["metadata"]),
        )
        for record in records
    ]

    async with pool.acquire() as conn:
        await conn.executemany(_AAWM_SESSION_HISTORY_INSERT_SQL, payloads)


def _get_reasoning_state_tags(
    provider_prefix: str,
    reasoning_content: str,
    thinking_blocks: List[dict],
) -> List[str]:
    stripped_reasoning = reasoning_content.strip()
    tags: List[str] = []
    if stripped_reasoning:
        tags.append("reasoning-present")
        tags.append(f"{provider_prefix}-reasoning-present")
    else:
        tags.append("reasoning-empty")
        tags.append(f"{provider_prefix}-reasoning-empty")

    if thinking_blocks:
        tags.append("thinking-blocks-present")
        tags.append(f"{provider_prefix}-thinking-blocks-present")
    else:
        tags.append("thinking-blocks-empty")
        tags.append(f"{provider_prefix}-thinking-blocks-empty")
    return tags


def _extract_claude_experiment_ids(decoded_bytes: bytes) -> List[str]:
    experiment_ids: List[str] = []
    for offset, current_byte in enumerate(decoded_bytes[:-2]):
        if current_byte != 0x32:
            continue
        candidate_length = decoded_bytes[offset + 1]
        candidate_start = offset + 2
        candidate_end = candidate_start + candidate_length
        if candidate_end > len(decoded_bytes):
            continue
        candidate_bytes = decoded_bytes[candidate_start:candidate_end]
        if not all(32 <= byte <= 126 for byte in candidate_bytes):
            continue
        decoded_match = candidate_bytes.decode("ascii", errors="ignore")
        if decoded_match.count("-") < 2:
            continue
        if decoded_match not in experiment_ids:
            experiment_ids.append(decoded_match)

    if experiment_ids:
        return experiment_ids

    for match in _CLAUDE_EXPERIMENT_ID_RE.findall(decoded_bytes):
        decoded_match = match.decode("ascii", errors="ignore")
        if decoded_match.count("-") < 2:
            continue
        if decoded_match not in experiment_ids:
            experiment_ids.append(decoded_match)
    return experiment_ids


def _enrich_claude_thinking_metadata(metadata: Dict[str, Any], message: Any) -> None:
    span_started_at = datetime.now(timezone.utc)
    thinking_blocks = _extract_thinking_blocks(message)
    if not thinking_blocks:
        return
    reasoning_content = _extract_reasoning_content(message, thinking_blocks)

    signatures: List[str] = []
    for block in thinking_blocks:
        if _maybe_get(block, "type") != "thinking":
            continue
        signature = _maybe_get(block, "signature")
        if isinstance(signature, str) and signature.strip():
            signatures.append(signature)

    if not signatures:
        return

    decoded_hashes: List[str] = []
    experiment_ids: List[str] = []
    decode_errors: List[str] = []
    decoded_any = False

    for signature in signatures:
        try:
            decoded_bytes = _decode_base64_bytes(signature)
            decoded_hashes.append(_short_hash(decoded_bytes))
            decoded_any = True
            for experiment_id in _extract_claude_experiment_ids(decoded_bytes):
                if experiment_id not in experiment_ids:
                    experiment_ids.append(experiment_id)
        except Exception as exc:
            decode_errors.append(str(exc))

    metadata["claude_thinking_signature_present"] = len(signatures) > 0
    metadata["claude_thinking_signature_count"] = len(signatures)
    metadata["claude_thinking_signature_hashes"] = decoded_hashes
    metadata["claude_thinking_signature_decoded"] = decoded_any
    metadata["claude_thinking_decode_version"] = "v1"
    metadata["claude_reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["claude_reasoning_content_empty_or_short"] = (
        len(reasoning_content.strip()) < 16
    )
    if experiment_ids:
        metadata["claude_thinking_experiment_ids"] = experiment_ids
        if len(experiment_ids) == 1:
            metadata["claude_thinking_experiment_id"] = experiment_ids[0]
    if decode_errors:
        metadata["claude_thinking_decode_errors"] = decode_errors

    metadata["thinking_signature_present"] = True
    metadata["thinking_signature_decoded"] = decoded_any
    metadata["reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    metadata["thinking_blocks_present"] = len(thinking_blocks) > 0

    tags_to_add = ["claude-thinking-signature", "thinking-signature-present"]
    if decoded_any:
        tags_to_add.extend(["claude-thinking-decoded", "thinking-signature-decoded"])
    tags_to_add.extend(
        _get_reasoning_state_tags(
            provider_prefix="claude",
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    tags_to_add.extend(f"claude-exp:{experiment_id}" for experiment_id in experiment_ids)
    _merge_tags(metadata, tags_to_add)
    _append_langfuse_span(
        metadata,
        name="claude.thinking_signature_decode",
        span_metadata={
            "signature_count": len(signatures),
            "decoded_signature_count": len(decoded_hashes),
            "thinking_block_count": len(thinking_blocks),
            "reasoning_content_present": bool(reasoning_content.strip()),
            "experiment_ids": experiment_ids,
        },
        start_time=span_started_at,
        end_time=datetime.now(timezone.utc),
    )


def _read_varint(data: bytes, offset: int) -> Tuple[Optional[int], int]:
    value = 0
    shift = 0
    current_offset = offset
    while current_offset < len(data):
        current_byte = data[current_offset]
        value |= (current_byte & 0x7F) << shift
        current_offset += 1
        if current_byte < 0x80:
            return value, current_offset
        shift += 7
        if shift > 63:
            break
    return None, offset


def _extract_gemini_signature_summary(signature: str) -> Dict[str, Any]:
    decoded_bytes = _decode_base64_bytes(signature)
    signature_hash = _short_hash(decoded_bytes)

    record_sizes: List[int] = []
    prefixes: List[str] = []
    marker_offsets: List[int] = []
    indexed_fields: Dict[str, Any] = {}

    offset = 0
    record_index = 0
    while offset < len(decoded_bytes):
        if decoded_bytes[offset] != 0x0A:
            break
        record_size, payload_offset = _read_varint(decoded_bytes, offset + 1)
        if record_size is None:
            break
        payload_end = payload_offset + record_size
        if payload_end > len(decoded_bytes):
            break

        payload = decoded_bytes[payload_offset:payload_end]
        marker_index = payload.find(_GEMINI_MARKER)
        prefix_hex = ""
        absolute_marker_offset = None
        if marker_index >= 0:
            prefix_hex = payload[:marker_index].hex()
            absolute_marker_offset = payload_offset + marker_index
            marker_offsets.append(absolute_marker_offset)

        record_sizes.append(record_size)
        prefixes.append(prefix_hex)
        indexed_fields[f"gemini_tsig_0_record_{record_index}_size"] = record_size
        indexed_fields[f"gemini_tsig_0_record_{record_index}_prefix"] = prefix_hex
        if absolute_marker_offset is not None:
            indexed_fields[
                f"gemini_tsig_0_record_{record_index}_marker_offset"
            ] = absolute_marker_offset

        record_index += 1
        offset = payload_end

    shape_components = {
        "decoded_bytes": len(decoded_bytes),
        "record_sizes": record_sizes,
        "prefixes": prefixes,
        "marker_offsets": marker_offsets,
    }
    shape_hash = _short_hash(str(shape_components).encode("utf-8"))

    summary: Dict[str, Any] = {
        "decoded_bytes": len(decoded_bytes),
        "record_count": len(record_sizes),
        "record_sizes": record_sizes,
        "prefixes": prefixes,
        "marker_offsets": marker_offsets,
        "marker_hex": _GEMINI_MARKER.hex(),
        "shape_hash": shape_hash,
        "signature_hash": signature_hash,
        "indexed_fields": indexed_fields,
    }
    return summary


def _enrich_gemini_thought_signature_metadata(
    metadata: Dict[str, Any], message: Any
) -> None:
    span_started_at = datetime.now(timezone.utc)
    provider_specific_fields = _extract_provider_specific_fields(message)
    thought_signatures = provider_specific_fields.get("thought_signatures")
    thinking_blocks = _extract_thinking_blocks(message)
    reasoning_content = _extract_reasoning_content(message, thinking_blocks)

    if not isinstance(thought_signatures, list):
        thought_signatures = []
    thought_signatures = [
        signature
        for signature in thought_signatures
        if isinstance(signature, str) and signature.strip()
    ]

    if not thought_signatures:
        return

    summaries: List[Dict[str, Any]] = []
    decode_errors: List[str] = []
    signature_hashes: List[str] = []
    shape_hashes: List[str] = []

    for index, signature in enumerate(thought_signatures):
        try:
            summary = _extract_gemini_signature_summary(signature)
            summaries.append(summary)
            signature_hashes.append(summary["signature_hash"])
            shape_hashes.append(summary["shape_hash"])
            metadata[f"gemini_tsig_{index}_decoded_bytes"] = summary["decoded_bytes"]
            metadata[f"gemini_tsig_{index}_record_count"] = summary["record_count"]
            metadata[f"gemini_tsig_{index}_record_sizes"] = summary["record_sizes"]
            metadata[f"gemini_tsig_{index}_prefixes"] = summary["prefixes"]
            metadata[f"gemini_tsig_{index}_marker_offsets"] = summary["marker_offsets"]
            metadata[f"gemini_tsig_{index}_marker_hex"] = summary["marker_hex"]
            metadata[f"gemini_tsig_{index}_shape_hash"] = summary["shape_hash"]

            indexed_fields = summary["indexed_fields"]
            for key, value in indexed_fields.items():
                if key.startswith("gemini_tsig_0_"):
                    metadata[key.replace("gemini_tsig_0_", f"gemini_tsig_{index}_")] = value
        except Exception as exc:
            decode_errors.append(str(exc))

    metadata["gemini_thought_signature_present"] = len(thought_signatures) > 0
    metadata["gemini_thought_signature_count"] = len(thought_signatures)
    metadata["gemini_tsig_signature_hashes"] = signature_hashes
    metadata["gemini_tsig_shape_hashes"] = sorted(set(shape_hashes))
    metadata["gemini_reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["gemini_reasoning_content_empty_or_short"] = (
        len(reasoning_content.strip()) < 16
    )
    metadata["gemini_thinking_blocks_present"] = len(thinking_blocks) > 0
    if summaries:
        first_summary = summaries[0]
        metadata["gemini_tsig_decoded_bytes"] = first_summary["decoded_bytes"]
        metadata["gemini_tsig_record_count"] = first_summary["record_count"]
        metadata["gemini_tsig_record_sizes"] = first_summary["record_sizes"]
        metadata["gemini_tsig_prefixes"] = first_summary["prefixes"]
        metadata["gemini_tsig_marker_offsets"] = first_summary["marker_offsets"]
        metadata["gemini_tsig_marker_hex"] = first_summary["marker_hex"]
        metadata["gemini_tsig_shape_hash"] = first_summary["shape_hash"]
    if decode_errors:
        metadata["gemini_tsig_decode_errors"] = decode_errors

    metadata["thinking_signature_present"] = True
    metadata["thinking_signature_decoded"] = len(summaries) > 0
    metadata["reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    metadata["thinking_blocks_present"] = len(thinking_blocks) > 0

    tags_to_add = ["gemini-thought-signature", "thinking-signature-present"]
    if summaries:
        tags_to_add.extend(
            ["gemini-thought-signature-decoded", "thinking-signature-decoded"]
        )
        for shape_hash in sorted(set(shape_hashes)):
            tags_to_add.append(f"gemini-tsig-shape:{shape_hash}")
        for record_count in sorted({summary["record_count"] for summary in summaries}):
            tags_to_add.append(f"gemini-tsig-records:{record_count}")

    tags_to_add.extend(
        _get_reasoning_state_tags(
            provider_prefix="gemini",
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    _merge_tags(metadata, tags_to_add)
    _append_langfuse_span(
        metadata,
        name="gemini.thought_signature_decode",
        span_metadata={
            "signature_count": len(thought_signatures),
            "decoded_signature_count": len(summaries),
            "shape_hashes": sorted(set(shape_hashes)),
            "record_counts": sorted(
                {summary["record_count"] for summary in summaries} if summaries else []
            ),
            "reasoning_content_present": bool(reasoning_content.strip()),
        },
        start_time=span_started_at,
        end_time=datetime.now(timezone.utc),
    )


def _enrich_trace_name_and_provider_metadata(
    kwargs: Dict[str, Any], result: Any
) -> Tuple[dict, Any]:
    agent_name = _extract_agent_name(kwargs)
    headers = _ensure_mutable_headers(kwargs)
    metadata = _ensure_mutable_metadata(kwargs)

    if headers:
        current = headers.get("langfuse_trace_name")
        if current == "claude-code":
            headers["langfuse_trace_name"] = f"claude-code.{agent_name}"
            verbose_logger.debug(
                "AawmAgentIdentity: enriched header trace_name to claude-code.%s",
                agent_name,
            )

    current_trace_name = metadata.get("trace_name")
    if current_trace_name == "claude-code":
        metadata["trace_name"] = f"claude-code.{agent_name}"
    elif not current_trace_name:
        metadata["trace_name"] = agent_name

    message = _extract_first_response_message(result)
    if message is not None:
        _enrich_claude_thinking_metadata(metadata, message)
        _enrich_gemini_thought_signature_metadata(metadata, message)
    _enrich_usage_breakout_metadata(kwargs, result)

    _sync_standard_logging_object(kwargs, metadata)

    verbose_logger.debug(
        "AawmAgentIdentity: agent=%s, trace_name=%s, tags=%s",
        agent_name,
        metadata.get("trace_name"),
        metadata.get("tags"),
    )
    return kwargs, result


class AawmAgentIdentity(CustomLogger):
    """CustomLogger that enriches Langfuse trace_name with agent identity.

    Implements both sync logging_hook() and async async_logging_hook() to
    cover all code paths:
    - Sync: pass-through endpoints run Langfuse in sync success_handler (thread pool)
    - Async: standard LLM calls run Langfuse in async_success_handler
    """

    def logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Sync hook - runs before Langfuse in sync success handler."""
        try:
            return _enrich_trace_name_and_provider_metadata(kwargs, result)
        except Exception as exc:
            verbose_logger.warning("AawmAgentIdentity.logging_hook failed: %s", exc)
            return kwargs, result

    async def async_logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Async hook - runs before Langfuse in async success handler."""
        try:
            return _enrich_trace_name_and_provider_metadata(kwargs, result)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_logging_hook failed: %s", exc
            )
            return kwargs, result

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Persist one finalized session-history row per completed LiteLLM call."""
        try:
            record = _build_session_history_record(
                kwargs=kwargs,
                result=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
            if record is None:
                return

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(_persist_session_history_record(record))
            else:
                loop.create_task(_persist_session_history_record(record))
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.log_success_event failed: %s", exc
            )

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        try:
            record = _build_session_history_record(
                kwargs=kwargs,
                result=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
            if record is None:
                return

            await _persist_session_history_record(record)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_log_success_event failed: %s", exc
            )


# Module-level instance for config registration via get_instance_fn().
# Config must reference this instance name, not the class name:
#   callbacks: ["litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance"]
aawm_agent_identity_instance = AawmAgentIdentity()
