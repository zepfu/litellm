"""
What is this?

Provider-specific Pass-Through Endpoints

Use litellm with Anthropic SDK, Vertex AI SDK, Cohere SDK, etc.
"""

import asyncio
import importlib
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
from urllib.parse import quote, urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState

import litellm
from litellm import get_llm_provider
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS,
    BEDROCK_AGENT_RUNTIME_PASS_THROUGH_ROUTES,
)
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.proxy._types import *
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
    _safe_set_request_parsed_body,
    get_form_data,
    get_request_body,
)
from litellm.proxy.pass_through_endpoints.common_utils import get_litellm_virtual_key
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
    create_pass_through_route,
    create_websocket_passthrough_route,
    websocket_passthrough_request,
)
from litellm.proxy.pass_through_endpoints.aawm_claude_control_plane import (
    add_claude_post_rewrite_context_file_logging_metadata as _aawm_add_claude_post_rewrite_context_file_logging_metadata,
)
from litellm.proxy.pass_through_endpoints.aawm_claude_control_plane import (
    apply_claude_prompt_patches_to_anthropic_request_body as _aawm_apply_claude_prompt_patches_to_anthropic_request_body,
)
from litellm.proxy.pass_through_endpoints.aawm_claude_control_plane import (
    expand_aawm_dynamic_directives_in_anthropic_request_body as _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body,
)
from litellm.proxy.pass_through_endpoints.aawm_claude_control_plane import (
    replace_claude_system_prompt_in_anthropic_request_body as _aawm_replace_claude_system_prompt_in_anthropic_request_body,
)
from litellm.proxy.utils import is_known_model
from litellm.proxy.vector_store_endpoints.utils import (
    is_allowed_to_call_vector_store_endpoint,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from .passthrough_endpoint_router import PassthroughEndpointRouter

if TYPE_CHECKING:
    import asyncpg

vertex_llm_base = VertexBase()
router = APIRouter()
default_vertex_config = None

passthrough_endpoint_router = PassthroughEndpointRouter()

_CLAUDE_PERSISTED_OUTPUT_PATTERN = re.compile(
    r"\A<system-reminder>\n"
    r"(?P<hook>SubagentStart|SubAgentStart|SessionStart) hook additional context: <persisted-output>\n"
    r"Output too large \([^)]+\)\. Full output saved to: (?P<path>/[^\n]+)\n\n"
    r"Preview \(first 2KB\):\n"
    r"(?P<preview>.*)"
    r"\n</persisted-output>\n</system-reminder>\n?\Z",
    re.DOTALL,
)
_ANTHROPIC_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"
_PASSTHROUGH_SESSION_ID_HEADER_NAMES = (
    "session_id",
    "Session_Id",
    "x-session-id",
    "X-Session-Id",
)
_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR = (
    Path(__file__).resolve().parents[3] / "context-replacement" / "claude-code"
)
_CLAUDE_AUTO_MEMORY_TEMPLATE_PATH = (
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR / "2.1.110" / "auto-memory-replacement.md"
)
_CLAUDE_PROMPT_PATCH_MANIFEST_PATH = (
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR
    / "prompt-patches"
    / "roman01la-2026-04-02.json"
)
_CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION = (2, 1, 110)
_CLAUDE_AUTO_MEMORY_SECTION_PATTERN = re.compile(
    r"(?ms)^# auto memory\n.*?(?=^# Environment\b|\Z)"
)
_CLAUDE_TYPES_XML_BLOCK_PATTERN = re.compile(r"<types>\n.*?\n</types>", re.DOTALL)
_CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")
_CLAUDE_CC_VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")
_AAWM_DYNAMIC_DIRECTIVE_PATTERN = re.compile(
    r"<!--\s*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<html_attrs>.*?)\s*-->"
    r"|@@@\s*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<at_attrs>.*?)\s*@@@"
    r"|^[ \t]*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<line_attrs>[^\r\n]+?)\s*$",
    re.DOTALL | re.MULTILINE,
)
_AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_-]*)='
    r'(?:"(?P<double>[^"]*)"|\'(?P<single>[^\']*)\'|(?P<bare>[^\s]+))'
)
_CLAUDE_AGENT_TENANT_PATTERN = re.compile(
    r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project\b"
)
_CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS: tuple[tuple[str, str], ...] = (
    ("MEMORY.md", "memory-md"),
    ("CLAUDE.md", "claude-md"),
)
_AAWM_AGENT_MEMORY_PROC_NAME = "get_agent_memories"
_AAWM_DYNAMIC_PROC_ALIASES = {"get_agent_memory": _AAWM_AGENT_MEMORY_PROC_NAME}
_AAWM_DYNAMIC_PROC_DEFAULT_CTX_FIELDS: dict[str, tuple[str, ...]] = {
    _AAWM_AGENT_MEMORY_PROC_NAME: ("agent", "tenant"),
}
_AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE = (
    "## AAWM Injection Status\n\n"
    'AAWM "{proc_name}" failed for this session.\n'
    "Alert the user or session orchestrator.\n"
)
_AAWM_NO_MEMORIES_TEMPLATE = (
    "# Memory Injection\n"
    "You have saved no memories as of yet.\n"
)
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
_aawm_dynamic_injection_pool: Optional[Any] = None
_aawm_dynamic_injection_pool_lock = asyncio.Lock()
_claude_context_replacement_template_cache: dict[Path, str] = {}
_claude_prompt_patch_manifest_cache: dict[Path, dict[str, Any]] = {}


def _is_openai_responses_endpoint(endpoint: str) -> bool:
    normalized_path = httpx.URL(endpoint).path.rstrip("/")
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    return (
        normalized_path == "/responses"
        or normalized_path == "/v1/responses"
        or normalized_path.startswith("/responses/")
        or normalized_path.startswith("/v1/responses/")
    )


def _request_has_openai_client_auth(request: Request) -> bool:
    headers = _safe_get_request_headers(request)
    return bool(
        headers.get("authorization")
        or headers.get("Authorization")
        or headers.get("api-key")
        or headers.get("Api-Key")
    )


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


def _is_claude_persisted_output_expansion_enabled() -> bool:
    value = os.getenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _get_claude_persisted_output_root() -> Path:
    return Path(
        os.getenv(
            "LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", "/home/zepfu/.claude/projects"
        )
    ).expanduser()


def _resolve_claude_persisted_output_path(path_str: str) -> Optional[Path]:
    try:
        root = _get_claude_persisted_output_root().resolve(strict=True)
        candidate = Path(path_str).expanduser().resolve(strict=True)
    except Exception:
        return None

    if not candidate.is_file():
        return None
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if "tool-results" not in candidate.parts:
        return None
    if not candidate.name.endswith("-additionalContext.txt"):
        return None
    return candidate


def _build_claude_persisted_output_source_metadata(
    *, resolved_path: Path, file_text: str
) -> dict[str, Any]:
    file_bytes = file_text.encode("utf-8")
    return {
        "path": str(resolved_path),
        "basename": resolved_path.name,
        "content_hash": hashlib.sha256(file_bytes).hexdigest(),
        "bytes": len(file_bytes),
    }


def _expand_claude_persisted_output_text(
    text: str,
) -> Tuple[str, bool, Optional[str], Optional[dict[str, Any]]]:
    if not _is_claude_persisted_output_expansion_enabled():
        return text, False, None, None

    match = _CLAUDE_PERSISTED_OUTPUT_PATTERN.match(text)
    if match is None:
        return text, False, None, None

    resolved_path = _resolve_claude_persisted_output_path(match.group("path"))
    if resolved_path is None:
        return text, False, None, None

    try:
        file_text = resolved_path.read_text(encoding="utf-8", errors="replace").rstrip(
            "\n"
        )
    except Exception:
        return text, False, None, None

    hook = match.group("hook")
    expanded = (
        "<system-reminder>\n"
        f"{hook} hook additional context: <persisted-output>\n"
        f"{file_text}\n"
        "</persisted-output>\n"
        "</system-reminder>\n"
    )
    return (
        expanded,
        True,
        hook.lower(),
        _build_claude_persisted_output_source_metadata(
            resolved_path=resolved_path,
            file_text=file_text,
        ),
    )


def _expand_claude_persisted_output_value(
    value: Any,
) -> Tuple[Any, int, set[str], list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            (
                expanded_text,
                was_expanded,
                hook_name,
                source_metadata,
            ) = _expand_claude_persisted_output_text(value["text"])
            if was_expanded:
                updated_value = dict(value)
                updated_value["text"] = expanded_text
                return (
                    updated_value,
                    1,
                    {hook_name} if hook_name else set(),
                    [source_metadata] if source_metadata else [],
                )
            return value, 0, set(), []

        updated_dict: dict[str, Any] = {}
        expanded_count = 0
        hooks: set[str] = set()
        source_metadata_items: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            (
                updated_child,
                child_expanded_count,
                child_hooks,
                child_source_metadata_items,
            ) = _expand_claude_persisted_output_value(child)
            updated_dict[key] = updated_child
            expanded_count += child_expanded_count
            hooks.update(child_hooks)
            source_metadata_items.extend(child_source_metadata_items)
            if updated_child is not child:
                changed = True
        return (
            updated_dict if changed else value,
            expanded_count,
            hooks,
            source_metadata_items,
        )

    if isinstance(value, list):
        updated_list = []
        expanded_count = 0
        hooks: set[str] = set()
        source_metadata_items: list[dict[str, Any]] = []
        changed = False
        for child in value:
            (
                updated_child,
                child_expanded_count,
                child_hooks,
                child_source_metadata_items,
            ) = _expand_claude_persisted_output_value(child)
            updated_list.append(updated_child)
            expanded_count += child_expanded_count
            hooks.update(child_hooks)
            source_metadata_items.extend(child_source_metadata_items)
            if updated_child is not child:
                changed = True
        return (
            updated_list if changed else value,
            expanded_count,
            hooks,
            source_metadata_items,
        )

    return value, 0, set(), []


def _merge_litellm_metadata(
    request_body: dict[str, Any],
    *,
    tags_to_add: Optional[list[str]] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    updated_body = dict(request_body)
    litellm_metadata = dict(updated_body.get("litellm_metadata") or {})
    existing_tags = litellm_metadata.get("tags") or []
    if not isinstance(existing_tags, list):
        existing_tags = []

    merged_tags = list(existing_tags)
    for tag in tags_to_add or []:
        if tag not in merged_tags:
            merged_tags.append(tag)

    litellm_metadata["tags"] = merged_tags
    if extra_fields:
        existing_spans = litellm_metadata.get("langfuse_spans")
        incoming_spans = extra_fields.get("langfuse_spans")
        if isinstance(existing_spans, list) and isinstance(incoming_spans, list):
            merged_extra_fields = dict(extra_fields)
            merged_extra_fields["langfuse_spans"] = list(existing_spans) + list(
                incoming_spans
            )
            litellm_metadata.update(merged_extra_fields)
        else:
            litellm_metadata.update(extra_fields)

    updated_body["litellm_metadata"] = litellm_metadata
    return updated_body


def _format_langfuse_span_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_langfuse_span_descriptor(
    *,
    name: str,
    metadata: Optional[dict[str, Any]] = None,
    input_data: Any = None,
    output_data: Any = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"name": name}
    if input_data is not None:
        descriptor["input"] = input_data
    if output_data is not None:
        descriptor["output"] = output_data
    if metadata:
        descriptor["metadata"] = metadata
    if start_time is not None:
        descriptor["start_time"] = _format_langfuse_span_timestamp(start_time)
    if end_time is not None:
        descriptor["end_time"] = _format_langfuse_span_timestamp(end_time)
    return descriptor


def _get_nested_str_value(source: Any, path: tuple[str, ...]) -> Optional[str]:
    current = source
    for key in path:
        if isinstance(current, str):
            stripped_current = current.strip()
            if not stripped_current:
                return None
            try:
                current = json.loads(stripped_current)
            except json.JSONDecodeError:
                return None
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, str) and current.strip():
        return current.strip()
    return None


def _extract_passthrough_session_id(
    request: Request, request_body: Optional[dict[str, Any]] = None
) -> Optional[str]:
    if isinstance(request_body, dict):
        for path in (
            ("session_id",),
            ("request", "session_id"),
            ("metadata", "session_id"),
            ("metadata", "user_id", "session_id"),
        ):
            value = _get_nested_str_value(request_body, path)
            if value:
                return value

    headers = _safe_get_request_headers(request)
    for header_name in _PASSTHROUGH_SESSION_ID_HEADER_NAMES:
        value = headers.get(header_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _get_passthrough_trace_environment() -> Optional[str]:
    for env_var in (
        "LITELLM_LANGFUSE_TRACE_ENVIRONMENT",
        "LANGFUSE_TRACING_ENVIRONMENT",
    ):
        value = os.getenv(env_var)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _add_passthrough_trace_context_metadata(
    request_body: dict[str, Any],
    *,
    session_id: Optional[str],
    trace_environment: Optional[str],
) -> dict[str, Any]:
    updated_body = dict(request_body)
    litellm_metadata = dict(updated_body.get("litellm_metadata") or {})
    changed = False

    if session_id and not litellm_metadata.get("session_id"):
        litellm_metadata["session_id"] = session_id
        changed = True

    if trace_environment and not litellm_metadata.get("trace_environment"):
        litellm_metadata["trace_environment"] = trace_environment
        changed = True

    if not changed:
        return request_body

    updated_body["litellm_metadata"] = litellm_metadata
    return updated_body


def _prepare_request_body_for_passthrough_observability(
    request: Request, request_body: dict[str, Any]
) -> dict[str, Any]:
    session_id = _extract_passthrough_session_id(
        request=request, request_body=request_body
    )
    trace_environment = _get_passthrough_trace_environment()
    return _add_passthrough_trace_context_metadata(
        request_body,
        session_id=session_id,
        trace_environment=trace_environment,
    )


def _add_route_family_logging_metadata(
    request_body: dict[str, Any], route_family: str
) -> dict[str, Any]:
    normalized_route_family = _normalize_low_cardinality_tag_value(route_family)
    if not normalized_route_family:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=[f"route:{normalized_route_family}"],
        extra_fields={"passthrough_route_family": normalized_route_family},
    )


def _normalize_low_cardinality_tag_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned or None
    return None


def _dedupe_sorted_str_list(values: list[str]) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


def _extract_claude_request_breakout_fields(
    request_body: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    tags_to_add: list[str] = []
    extra_fields: dict[str, Any] = {}

    thinking = request_body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = _normalize_low_cardinality_tag_value(thinking.get("type"))
        if thinking_type:
            tags_to_add.extend(
                [f"claude-thinking-type:{thinking_type}", f"thinking-type:{thinking_type}"]
            )
            extra_fields["claude_thinking_type"] = thinking_type

    output_config = request_body.get("output_config")
    if isinstance(output_config, dict):
        effort = _normalize_low_cardinality_tag_value(output_config.get("effort"))
        if effort:
            tags_to_add.extend([f"claude-effort:{effort}", f"effort:{effort}"])
            extra_fields["claude_effort"] = effort

    context_management = request_body.get("context_management")
    context_edits = []
    if isinstance(context_management, dict):
        edits = context_management.get("edits")
        if isinstance(edits, list):
            context_edits = [edit for edit in edits if isinstance(edit, dict)]

    edit_types: list[str] = []
    keep_values: list[str] = []
    for edit in context_edits:
        edit_type = _normalize_low_cardinality_tag_value(edit.get("type"))
        if edit_type:
            edit_types.append(edit_type)
            tags_to_add.append(f"claude-context-edit:{edit_type}")
        keep_value = _normalize_low_cardinality_tag_value(edit.get("keep"))
        if keep_value:
            keep_values.append(keep_value)
            tags_to_add.append(f"claude-context-keep:{keep_value}")

    if context_edits:
        extra_fields["claude_context_edit_count"] = len(context_edits)
    if edit_types:
        extra_fields["claude_context_edit_types"] = _dedupe_sorted_str_list(edit_types)
    if keep_values:
        extra_fields["claude_context_keep_values"] = _dedupe_sorted_str_list(
            keep_values
        )

    account_uuid = _get_nested_str_value(request_body, ("metadata", "user_id", "account_uuid"))
    if account_uuid:
        extra_fields["claude_account_uuid"] = account_uuid
    device_id = _get_nested_str_value(request_body, ("metadata", "user_id", "device_id"))
    if device_id:
        extra_fields["claude_device_id"] = device_id

    return tags_to_add, extra_fields


def _add_claude_request_breakout_logging_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tags_to_add, extra_fields = _extract_claude_request_breakout_fields(request_body)
    if not tags_to_add and not extra_fields:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _extract_gemini_request_breakout_fields(
    request_body: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    tags_to_add: list[str] = []
    extra_fields: dict[str, Any] = {}

    generation_config = request_body.get("generationConfig")
    if not isinstance(generation_config, dict):
        request_block = request_body.get("request")
        if isinstance(request_block, dict):
            nested_generation_config = request_block.get("generationConfig")
            if isinstance(nested_generation_config, dict):
                generation_config = nested_generation_config

    if isinstance(generation_config, dict):
        thinking_config = generation_config.get("thinkingConfig")
        if isinstance(thinking_config, dict):
            tags_to_add.append("gemini-thinking-config-present")
            extra_fields["gemini_thinking_config_present"] = True

            include_thoughts = thinking_config.get("includeThoughts")
            if isinstance(include_thoughts, bool):
                include_thoughts_tag = "true" if include_thoughts else "false"
                tags_to_add.extend(
                    [
                        f"gemini-include-thoughts:{include_thoughts_tag}",
                        f"include-thoughts:{include_thoughts_tag}",
                    ]
                )
                extra_fields["gemini_include_thoughts"] = include_thoughts

            thinking_level = thinking_config.get("thinkingLevel")
            normalized_thinking_level = _normalize_low_cardinality_tag_value(
                thinking_level
            )
            if normalized_thinking_level:
                tags_to_add.extend(
                    [
                        f"gemini-thinking-level:{normalized_thinking_level}",
                        f"thinking-level:{normalized_thinking_level}",
                    ]
                )
                extra_fields["gemini_thinking_level"] = normalized_thinking_level

            thinking_budget = thinking_config.get("thinkingBudget")
            if isinstance(thinking_budget, (int, float)) and thinking_budget > 0:
                tags_to_add.append("gemini-thinking-budget-configured")
                extra_fields["gemini_thinking_budget"] = thinking_budget

    tools = request_body.get("tools")
    if not isinstance(tools, list):
        request_block = request_body.get("request")
        if isinstance(request_block, dict):
            nested_tools = request_block.get("tools")
            if isinstance(nested_tools, list):
                tools = nested_tools

    if isinstance(tools, list) and tools:
        tags_to_add.append("gemini-tools-present")
        extra_fields["gemini_tools_present"] = True
        extra_fields["gemini_tool_count"] = len(tools)

    for key in ("user_prompt_id", "project"):
        value = request_body.get(key)
        if not value and isinstance(request_body.get("request"), dict):
            value = request_body["request"].get(key)
        if isinstance(value, str) and value.strip():
            extra_fields[f"gemini_{key}"] = value.strip()

    return tags_to_add, extra_fields


def _add_gemini_request_breakout_logging_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tags_to_add, extra_fields = _extract_gemini_request_breakout_fields(request_body)
    if not tags_to_add and not extra_fields:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _extract_openai_passthrough_tool_choice(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return _normalize_low_cardinality_tag_value(value)
    if isinstance(value, dict):
        for key in ("type", "name"):
            normalized = _normalize_low_cardinality_tag_value(value.get(key))
            if normalized:
                return normalized
    return None


def _extract_codex_request_breakout_fields(
    request_body: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    tags_to_add: list[str] = []
    extra_fields: dict[str, Any] = {}

    reasoning = request_body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = _normalize_low_cardinality_tag_value(reasoning.get("effort"))
        if effort:
            tags_to_add.extend([f"codex-effort:{effort}", f"effort:{effort}"])
            extra_fields["codex_reasoning_effort"] = effort

    tool_choice = _extract_openai_passthrough_tool_choice(request_body.get("tool_choice"))
    if tool_choice:
        tags_to_add.append(f"codex-tool-choice:{tool_choice}")
        extra_fields["codex_tool_choice"] = tool_choice

    parallel_tool_calls = request_body.get("parallel_tool_calls")
    if isinstance(parallel_tool_calls, bool):
        tags_to_add.append(
            f"codex-parallel-tools:{'true' if parallel_tool_calls else 'false'}"
        )
        extra_fields["codex_parallel_tool_calls"] = parallel_tool_calls

    include = request_body.get("include")
    normalized_includes: list[str] = []
    if isinstance(include, list):
        for value in include:
            normalized = _normalize_low_cardinality_tag_value(value)
            if normalized:
                normalized_includes.append(normalized)
                tags_to_add.append(f"codex-include:{normalized}")
    if normalized_includes:
        extra_fields["codex_include"] = _dedupe_sorted_str_list(normalized_includes)

    prompt_cache_key = request_body.get("prompt_cache_key")
    if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
        extra_fields["codex_prompt_cache_key_present"] = True

    return tags_to_add, extra_fields


def _add_codex_request_breakout_logging_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tags_to_add, extra_fields = _extract_codex_request_breakout_fields(request_body)
    if not tags_to_add and not extra_fields:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _add_claude_persisted_output_logging_metadata(
    request_body: dict[str, Any],
    expanded_count: int,
    hooks: set[str],
    source_metadata_items: list[dict[str, Any]],
) -> dict[str, Any]:
    span_metadata: dict[str, Any] = {
        "expanded_count": expanded_count,
        "hook_count": len(hooks),
    }
    if hooks:
        span_metadata["hooks"] = sorted(hooks)
    if source_metadata_items:
        span_metadata["source_count"] = len(source_metadata_items)
        span_metadata["source_paths"] = [
            item["path"]
            for item in source_metadata_items
            if isinstance(item.get("path"), str)
        ]
        span_metadata["source_content_hashes"] = [
            item["content_hash"]
            for item in source_metadata_items
            if isinstance(item.get("content_hash"), str)
        ]
        span_metadata["source_bytes"] = [
            item["bytes"]
            for item in source_metadata_items
            if isinstance(item.get("bytes"), int)
        ]
    tags_to_add = ["claude-persisted-output-expanded"]
    tags_to_add.extend(
        f"claude-persisted-output-hook:{hook}" for hook in sorted(hooks) if hook
    )
    extra_fields: dict[str, Any] = {
        "claude_persisted_output_expanded": True,
        "claude_persisted_output_expanded_count": expanded_count,
        "langfuse_spans": [
            _build_langfuse_span_descriptor(
                name="claude.persisted_output_expand",
                metadata=span_metadata,
            )
        ],
    }
    if hooks:
        extra_fields["claude_persisted_output_hooks"] = sorted(hooks)
    if source_metadata_items:
        extra_fields["claude_persisted_output_source_paths"] = [
            item["path"]
            for item in source_metadata_items
            if isinstance(item.get("path"), str)
        ]
        extra_fields["claude_persisted_output_source_basenames"] = [
            item["basename"]
            for item in source_metadata_items
            if isinstance(item.get("basename"), str)
        ]
        extra_fields["claude_persisted_output_source_content_hashes"] = [
            item["content_hash"]
            for item in source_metadata_items
            if isinstance(item.get("content_hash"), str)
        ]
        extra_fields["claude_persisted_output_source_bytes"] = [
            item["bytes"]
            for item in source_metadata_items
            if isinstance(item.get("bytes"), int)
        ]
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _parse_anthropic_billing_header_text(text: str) -> dict[str, str]:
    parsed_fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line.lower().startswith(_ANTHROPIC_BILLING_HEADER_PREFIX):
            continue
        raw_header_value = stripped_line.split(":", 1)[1].strip()
        for segment in raw_header_value.split(";"):
            cleaned_segment = segment.strip()
            if not cleaned_segment or "=" not in cleaned_segment:
                continue
            key, value = cleaned_segment.split("=", 1)
            cleaned_key = key.strip()
            cleaned_value = value.strip()
            if cleaned_key and cleaned_value:
                parsed_fields[cleaned_key] = cleaned_value
    return parsed_fields


def _extract_anthropic_billing_header_fields(value: Any) -> dict[str, str]:
    parsed_fields: dict[str, str] = {}

    if isinstance(value, str):
        return _parse_anthropic_billing_header_text(value)

    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            parsed_fields.update(_parse_anthropic_billing_header_text(value["text"]))
        for child in value.values():
            parsed_fields.update(_extract_anthropic_billing_header_fields(child))
        return parsed_fields

    if isinstance(value, list):
        for child in value:
            parsed_fields.update(_extract_anthropic_billing_header_fields(child))

    return parsed_fields


def _extract_anthropic_billing_header_fields_from_request_body(
    request_body: dict[str, Any]
) -> dict[str, str]:
    return _extract_anthropic_billing_header_fields(request_body.get("system"))


def _parse_claude_code_version(cc_version: Optional[str]) -> Optional[tuple[int, int, int]]:
    if not cc_version:
        return None

    match = _CLAUDE_CC_VERSION_PATTERN.match(cc_version.strip())
    if match is None:
        return None

    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _resolve_claude_auto_memory_template_path(cc_version: Optional[str]) -> Optional[Path]:
    parsed_version = _parse_claude_code_version(cc_version)
    if parsed_version is None:
        return None

    major, minor, patch = parsed_version
    min_major, min_minor, min_patch = _CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION

    if (major, minor) != (min_major, min_minor):
        return None
    if patch < min_patch:
        return None

    return _CLAUDE_AUTO_MEMORY_TEMPLATE_PATH


def _load_claude_context_replacement_template(template_path: Path) -> str:
    cached_template = _claude_context_replacement_template_cache.get(template_path)
    if cached_template is not None:
        return cached_template

    template_text = template_path.read_text(encoding="utf-8").strip()
    if not template_text:
        raise ValueError(f"Claude context replacement template is empty: {template_path}")

    cached_template = template_text + "\n"
    _claude_context_replacement_template_cache[template_path] = cached_template
    return cached_template


def _load_claude_prompt_patch_manifest(template_path: Path) -> dict[str, Any]:
    cached_manifest = _claude_prompt_patch_manifest_cache.get(template_path)
    if cached_manifest is not None:
        return cached_manifest

    manifest = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid Claude prompt patch manifest: {template_path}")

    patches = manifest.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError(f"Claude prompt patch manifest has no patches: {template_path}")

    normalized_patches: list[dict[str, str]] = []
    for patch_descriptor in patches:
        if not isinstance(patch_descriptor, dict):
            raise ValueError(
                f"Invalid Claude prompt patch descriptor in {template_path}"
            )
        patch_id = patch_descriptor.get("id")
        before_text = patch_descriptor.get("before")
        after_text = patch_descriptor.get("after")
        if not isinstance(patch_id, str) or not patch_id:
            raise ValueError(
                f"Claude prompt patch manifest is missing patch id in {template_path}"
            )
        if not isinstance(before_text, str) or not before_text:
            raise ValueError(
                f"Claude prompt patch manifest is missing before text for {patch_id}"
            )
        if not isinstance(after_text, str) or not after_text:
            raise ValueError(
                f"Claude prompt patch manifest is missing after text for {patch_id}"
            )
        normalized_patches.append(
            {
                "id": patch_id,
                "before": before_text,
                "after": after_text,
            }
        )

    normalized_manifest = {
        "source": manifest.get("source"),
        "patches": normalized_patches,
    }
    _claude_prompt_patch_manifest_cache[template_path] = normalized_manifest
    return normalized_manifest


def _extract_markdown_section(markdown_text: str, heading: str) -> str:
    section_pattern = re.compile(
        rf"(?ms)^## {re.escape(heading)}\n.*?(?=^## |\Z)"
    )
    match = section_pattern.search(markdown_text)
    if match is None:
        raise ValueError(f"Missing Claude auto-memory section: {heading}")
    return match.group(0).rstrip()


def _render_claude_auto_memory_replacement(
    auto_memory_section: str, cc_version: str
) -> tuple[str, Path]:
    template_path = _resolve_claude_auto_memory_template_path(cc_version)
    if template_path is None:
        raise ValueError(f"Unsupported Claude Code version for auto-memory override: {cc_version}")

    template_text = _load_claude_context_replacement_template(template_path)
    types_match = _CLAUDE_TYPES_XML_BLOCK_PATTERN.search(auto_memory_section)
    if types_match is None:
        raise ValueError("Missing Claude auto-memory <types> block")

    rendered_text = template_text
    rendered_text = rendered_text.replace(
        "{{TYPES_XML_BLOCK}}", types_match.group(0).rstrip()
    )
    rendered_text = rendered_text.replace(
        "{{WHAT_NOT_TO_SAVE_SECTION}}",
        _extract_markdown_section(auto_memory_section, "What NOT to save in memory"),
    )
    rendered_text = rendered_text.replace(
        "{{BEFORE_RECOMMENDING_SECTION}}",
        _extract_markdown_section(auto_memory_section, "Before recommending from memory"),
    )
    rendered_text = rendered_text.replace(
        "{{MEMORY_AND_PERSISTENCE_SECTION}}",
        _extract_markdown_section(
            auto_memory_section, "Memory and other forms of persistence"
        ),
    )

    unresolved_placeholders = _CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN.findall(
        rendered_text
    )
    if unresolved_placeholders:
        raise ValueError(
            "Unresolved Claude context replacement placeholders: "
            + ", ".join(sorted(unresolved_placeholders))
        )

    return rendered_text.rstrip() + "\n", template_path


def _replace_claude_auto_memory_section_in_text(
    text: str, cc_version: str
) -> tuple[str, Optional[dict[str, Any]]]:
    if "# auto memory" not in text:
        return text, None

    section_match = _CLAUDE_AUTO_MEMORY_SECTION_PATTERN.search(text)
    if section_match is None:
        return text, None

    replacement_text, template_path = _render_claude_auto_memory_replacement(
        section_match.group(0),
        cc_version,
    )
    replacement_event: dict[str, Any] = {
        "id": "auto-memory",
        "status": "resolved",
        "cc_version": cc_version,
        "template_path": str(template_path.relative_to(Path(__file__).resolve().parents[3])),
        "output_chars": len(replacement_text),
    }
    return (
        text[: section_match.start()] + replacement_text + text[section_match.end() :],
        replacement_event,
    )


def _replace_claude_system_prompt_override_in_value(
    value: Any, cc_version: str
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            if "# auto memory" not in value["text"]:
                return value, []
            try:
                updated_text, event = _replace_claude_auto_memory_section_in_text(
                    value["text"], cc_version
                )
            except Exception as exc:
                return value, [
                    {
                        "id": "auto-memory",
                        "status": "failed",
                        "cc_version": cc_version,
                        "error": exc.__class__.__name__,
                    }
                ]

            if event is None:
                return value, []
            updated_value = dict(value)
            updated_value["text"] = updated_text
            return updated_value, [event]

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = _replace_claude_system_prompt_override_in_value(
                child,
                cc_version,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_system_prompt_override_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), combined_events

    return value, []


def _add_claude_system_prompt_override_logging_metadata(
    request_body: dict[str, Any], override_events: list[dict[str, Any]]
) -> dict[str, Any]:
    override_ids = sorted(
        {
            event["id"]
            for event in override_events
            if isinstance(event.get("id"), str) and event["id"]
        }
    )
    failure_ids = sorted(
        {
            event["id"]
            for event in override_events
            if event.get("status") == "failed"
            and isinstance(event.get("id"), str)
            and event["id"]
        }
    )
    statuses = [
        event["status"]
        for event in override_events
        if isinstance(event.get("status"), str) and event["status"]
    ]
    cc_versions = sorted(
        {
            event["cc_version"]
            for event in override_events
            if isinstance(event.get("cc_version"), str) and event["cc_version"]
        }
    )
    template_paths = sorted(
        {
            event["template_path"]
            for event in override_events
            if isinstance(event.get("template_path"), str) and event["template_path"]
        }
    )

    tags_to_add = ["claude-system-prompt-override"]
    tags_to_add.extend(
        f"claude-system-prompt-override:{override_id}" for override_id in override_ids
    )
    if failure_ids:
        tags_to_add.append("claude-system-prompt-override-failed")

    span_metadata: dict[str, Any] = {
        "override_count": len(override_events),
        "failure_count": len(failure_ids),
    }
    if override_ids:
        span_metadata["override_ids"] = override_ids
    if cc_versions:
        span_metadata["cc_versions"] = cc_versions

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_system_prompt_override_count": len(override_events),
            "claude_system_prompt_override_ids": override_ids,
            "claude_system_prompt_override_failure_ids": failure_ids,
            "claude_system_prompt_override_statuses": statuses,
            "claude_system_prompt_override_cc_versions": cc_versions,
            "claude_system_prompt_override_template_paths": template_paths,
            "claude_system_prompt_override_events": override_events,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="claude.system_prompt_override",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _replace_claude_system_prompt_in_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cc_version = billing_header_fields.get("cc_version")
    template_path = _resolve_claude_auto_memory_template_path(cc_version)
    if template_path is None or "system" not in request_body:
        return request_body, []

    span_started_at = datetime.now(timezone.utc)
    updated_body = dict(request_body)
    updated_system, override_events = _replace_claude_system_prompt_override_in_value(
        request_body["system"],
        cc_version,
    )
    if not override_events:
        return request_body, []

    updated_body["system"] = updated_system
    updated_body = _add_claude_system_prompt_override_logging_metadata(
        updated_body,
        override_events,
    )

    litellm_metadata = updated_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        langfuse_spans = litellm_metadata.get("langfuse_spans")
        if isinstance(langfuse_spans, list):
            for span_descriptor in langfuse_spans:
                if (
                    isinstance(span_descriptor, dict)
                    and span_descriptor.get("name") == "claude.system_prompt_override"
                ):
                    span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )
    return updated_body, override_events


def _apply_claude_prompt_patches_in_text(
    text: str, cc_version: str
) -> tuple[str, list[dict[str, Any]]]:
    manifest_path = _CLAUDE_PROMPT_PATCH_MANIFEST_PATH
    manifest = _load_claude_prompt_patch_manifest(manifest_path)
    updated_text = text
    patch_events: list[dict[str, Any]] = []
    relative_manifest_path = str(
        manifest_path.relative_to(Path(__file__).resolve().parents[3])
    )

    for patch_descriptor in manifest["patches"]:
        before_text = patch_descriptor["before"]
        if before_text not in updated_text:
            continue

        after_text = patch_descriptor["after"]
        occurrences = updated_text.count(before_text)
        updated_text = updated_text.replace(before_text, after_text)
        patch_events.append(
            {
                "id": patch_descriptor["id"],
                "status": "resolved",
                "cc_version": cc_version,
                "manifest_path": relative_manifest_path,
                "occurrences": occurrences,
            }
        )

    return updated_text, patch_events


def _replace_claude_prompt_patches_in_value(
    value: Any, cc_version: str
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            try:
                updated_text, patch_events = _apply_claude_prompt_patches_in_text(
                    value["text"], cc_version
                )
            except Exception as exc:
                return value, [
                    {
                        "id": "manifest-load",
                        "status": "failed",
                        "cc_version": cc_version,
                        "error": exc.__class__.__name__,
                    }
                ]
            if not patch_events:
                return value, []
            updated_value = dict(value)
            updated_value["text"] = updated_text
            return updated_value, patch_events

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = _replace_claude_prompt_patches_in_value(
                child,
                cc_version,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_prompt_patches_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), combined_events

    return value, []


def _add_claude_prompt_patch_logging_metadata(
    request_body: dict[str, Any], patch_events: list[dict[str, Any]]
) -> dict[str, Any]:
    patch_ids = sorted(
        {
            event["id"]
            for event in patch_events
            if isinstance(event.get("id"), str) and event["id"]
        }
    )
    failure_ids = sorted(
        {
            event["id"]
            for event in patch_events
            if event.get("status") == "failed"
            and isinstance(event.get("id"), str)
            and event["id"]
        }
    )
    statuses = [
        event["status"]
        for event in patch_events
        if isinstance(event.get("status"), str) and event["status"]
    ]
    cc_versions = sorted(
        {
            event["cc_version"]
            for event in patch_events
            if isinstance(event.get("cc_version"), str) and event["cc_version"]
        }
    )
    manifest_paths = sorted(
        {
            event["manifest_path"]
            for event in patch_events
            if isinstance(event.get("manifest_path"), str) and event["manifest_path"]
        }
    )
    total_occurrences = sum(
        event["occurrences"]
        for event in patch_events
        if isinstance(event.get("occurrences"), int)
    )

    tags_to_add = ["claude-prompt-patch"]
    tags_to_add.extend(f"claude-prompt-patch:{patch_id}" for patch_id in patch_ids)
    if failure_ids:
        tags_to_add.append("claude-prompt-patch-failed")

    span_metadata: dict[str, Any] = {
        "patch_count": len(patch_events),
        "replacement_count": total_occurrences,
        "failure_count": len(failure_ids),
    }
    if patch_ids:
        span_metadata["patch_ids"] = patch_ids
    if cc_versions:
        span_metadata["cc_versions"] = cc_versions

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_prompt_patch_count": len(patch_events),
            "claude_prompt_patch_replacement_count": total_occurrences,
            "claude_prompt_patch_ids": patch_ids,
            "claude_prompt_patch_failure_ids": failure_ids,
            "claude_prompt_patch_statuses": statuses,
            "claude_prompt_patch_cc_versions": cc_versions,
            "claude_prompt_patch_manifest_paths": manifest_paths,
            "claude_prompt_patch_events": patch_events,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="claude.prompt_patch",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _apply_claude_prompt_patches_to_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cc_version = billing_header_fields.get("cc_version")
    if not cc_version:
        return request_body, []

    span_started_at = datetime.now(timezone.utc)
    updated_body, patch_events = _replace_claude_prompt_patches_in_value(
        request_body,
        cc_version,
    )
    if not patch_events:
        return request_body, []

    if not isinstance(updated_body, dict):
        return request_body, []

    updated_body = _add_claude_prompt_patch_logging_metadata(
        updated_body,
        patch_events,
    )

    litellm_metadata = updated_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        langfuse_spans = litellm_metadata.get("langfuse_spans")
        if isinstance(langfuse_spans, list):
            for span_descriptor in langfuse_spans:
                if (
                    isinstance(span_descriptor, dict)
                    and span_descriptor.get("name") == "claude.prompt_patch"
                ):
                    span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )
    return updated_body, patch_events


def _add_anthropic_billing_header_logging_metadata(
    request_body: dict[str, Any],
    billing_header_fields: dict[str, str],
) -> dict[str, Any]:
    tags_to_add = ["anthropic-billing-header"]
    for key in sorted(billing_header_fields):
        value = billing_header_fields[key]
        tags_to_add.append(f"anthropic-billing-header-key:{key}")
        tags_to_add.append(f"anthropic-billing-header:{key}={value}")

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "anthropic_billing_header_present": True,
            "anthropic_billing_header_keys": sorted(billing_header_fields),
            "anthropic_billing_header_fields": dict(billing_header_fields),
        },
    )


def _expand_claude_persisted_output_in_anthropic_request_body(
    request_body: dict[str, Any]
) -> Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]]:
    span_started_at = datetime.now(timezone.utc)
    (
        updated_body,
        expanded_count,
        hooks,
        source_metadata_items,
    ) = _expand_claude_persisted_output_value(
        request_body
    )
    if isinstance(updated_body, dict):
        if expanded_count > 0:
            updated_body = _add_claude_persisted_output_logging_metadata(
                updated_body,
                expanded_count,
                hooks,
                source_metadata_items,
            )
            litellm_metadata = updated_body.get("litellm_metadata")
            if isinstance(litellm_metadata, dict):
                langfuse_spans = litellm_metadata.get("langfuse_spans")
                if isinstance(langfuse_spans, list):
                    for span_descriptor in langfuse_spans:
                        if (
                            isinstance(span_descriptor, dict)
                            and span_descriptor.get("name")
                            == "claude.persisted_output_expand"
                        ):
                            span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                                span_started_at
                            )
                            span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                                datetime.now(timezone.utc)
                            )
        return updated_body, expanded_count, hooks, source_metadata_items
    return request_body, 0, set(), []


def _parse_aawm_directive_attributes(attrs_text: str) -> dict[str, str]:
    parsed_attrs: dict[str, str] = {}
    for match in _AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN.finditer(attrs_text):
        value = (
            match.group("double")
            or match.group("single")
            or match.group("bare")
            or ""
        ).strip()
        if value:
            parsed_attrs[match.group("key")] = value
    return parsed_attrs


def _get_aawm_directive_attrs_text(match: re.Match[str]) -> str:
    return (
        (match.group("html_attrs") or match.group("at_attrs") or match.group("line_attrs") or "")
    ).strip()


def _iter_anthropic_text_fragments(value: Any):
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            yield value["text"]
            return
        for child in value.values():
            yield from _iter_anthropic_text_fragments(child)
        return

    if isinstance(value, list):
        for child in value:
            yield from _iter_anthropic_text_fragments(child)


def _extract_claude_agent_and_tenant_from_request_body(
    request_body: dict[str, Any]
) -> tuple[Optional[str], Optional[str]]:
    for top_level_key in ("messages", "system"):
        for fragment in _iter_anthropic_text_fragments(request_body.get(top_level_key)):
            match = _CLAUDE_AGENT_TENANT_PATTERN.search(fragment)
            if match is None:
                continue
            agent = match.group("agent").strip()
            tenant = match.group("tenant").strip()
            if agent and tenant:
                return agent, tenant
    return None, None


def _build_aawm_context_for_anthropic_request(
    request_body: dict[str, Any]
) -> dict[str, str]:
    context: dict[str, str] = {}
    agent, tenant = _extract_claude_agent_and_tenant_from_request_body(request_body)
    if agent:
        context["agent"] = agent
    if tenant:
        context["tenant"] = tenant
    return context


def _detect_claude_post_rewrite_context_files(
    request_body: dict[str, Any]
) -> list[str]:
    present_files: list[str] = []
    seen_files: set[str] = set()

    for top_level_key in ("system", "messages"):
        for fragment in _iter_anthropic_text_fragments(request_body.get(top_level_key)):
            for marker, _tag_suffix in _CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS:
                if marker in seen_files:
                    continue
                if marker in fragment:
                    present_files.append(marker)
                    seen_files.add(marker)

    return present_files


def _add_claude_post_rewrite_context_file_logging_metadata(
    request_body: dict[str, Any]
) -> dict[str, Any]:
    present_files = _detect_claude_post_rewrite_context_files(request_body)
    if not present_files:
        return request_body

    tags_to_add = ["claude-post-rewrite-context-file-present"]
    for marker, tag_suffix in _CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS:
        if marker in present_files:
            tags_to_add.append(f"claude-post-rewrite-context-file:{tag_suffix}")

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_post_rewrite_context_files_present": present_files,
            "claude_post_rewrite_context_file_count": len(present_files),
        },
    )


def _build_aawm_dynamic_injection_failure_text(proc_name: str) -> str:
    return _AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE.format(
        proc_name=proc_name or "unknown"
    )


def _build_aawm_dynamic_injection_dsn() -> Optional[str]:
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


async def _get_aawm_dynamic_injection_pool() -> Any:
    global _aawm_dynamic_injection_pool

    if _aawm_dynamic_injection_pool is not None:
        return _aawm_dynamic_injection_pool

    async with _aawm_dynamic_injection_pool_lock:
        if _aawm_dynamic_injection_pool is not None:
            return _aawm_dynamic_injection_pool

        dsn = _build_aawm_dynamic_injection_dsn()
        if not dsn:
            raise RuntimeError("AAWM dynamic injection database configuration is missing")

        try:
            asyncpg = importlib.import_module("asyncpg")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AAWM dynamic injection requires asyncpg to be installed"
            ) from exc

        _aawm_dynamic_injection_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=4,
            command_timeout=10,
        )
        return _aawm_dynamic_injection_pool


async def _call_aawm_get_agent_memories(
    *, agent_name: str, tenant_id: str
) -> Optional[str]:
    pool = await _get_aawm_dynamic_injection_pool()
    result = await pool.fetchval(
        "SELECT get_agent_memories($1, $2)",
        agent_name,
        tenant_id,
    )
    if isinstance(result, str):
        stripped_result = result.strip()
        if stripped_result:
            return stripped_result
    return None


def _resolve_aawm_dynamic_context_fields(
    proc_name: str, directive_attrs: dict[str, str]
) -> tuple[str, ...]:
    raw_ctx = directive_attrs.get("ctx")
    if raw_ctx:
        ctx_fields = tuple(
            field.strip() for field in raw_ctx.split(",") if field.strip()
        )
    else:
        ctx_fields = _AAWM_DYNAMIC_PROC_DEFAULT_CTX_FIELDS.get(proc_name, ())

    if not ctx_fields:
        raise ValueError("No AAWM context fields were provided")

    return tuple(dict.fromkeys(ctx_fields))


def _select_aawm_dynamic_context(
    *, context_fields: tuple[str, ...], available_context: dict[str, str]
) -> dict[str, str]:
    selected_context: dict[str, str] = {}
    for field_name in context_fields:
        if field_name not in {"agent", "tenant"}:
            raise ValueError(f"Unsupported AAWM context field: {field_name}")

        field_value = available_context.get(field_name)
        if not field_value:
            raise ValueError(f"Missing AAWM context field: {field_name}")
        selected_context[field_name] = field_value
    return selected_context


async def _resolve_aawm_dynamic_directive(
    directive_attrs: dict[str, str],
    available_context: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    raw_proc_name = directive_attrs.get("p") or directive_attrs.get("proc") or "unknown"
    proc_name = _AAWM_DYNAMIC_PROC_ALIASES.get(raw_proc_name, raw_proc_name)
    context_fields = _resolve_aawm_dynamic_context_fields(proc_name, directive_attrs)
    selected_context = _select_aawm_dynamic_context(
        context_fields=context_fields,
        available_context=available_context,
    )

    event: dict[str, Any] = {
        "proc": proc_name,
        "context_keys": list(context_fields),
    }
    version = directive_attrs.get("v") or directive_attrs.get("version")
    if version:
        event["version"] = version
    scope = directive_attrs.get("s") or directive_attrs.get("scope")
    if scope:
        event["scope"] = scope

    if proc_name == _AAWM_AGENT_MEMORY_PROC_NAME:
        injected_text = await _call_aawm_get_agent_memories(
            agent_name=selected_context["agent"],
            tenant_id=selected_context["tenant"],
        )
        if injected_text is None:
            event["status"] = "empty"
            return _AAWM_NO_MEMORIES_TEMPLATE, event

        event["status"] = "resolved"
        event["output_chars"] = len(injected_text)
        return injected_text, event

    raise ValueError(f"Unsupported AAWM proc: {proc_name}")


async def _expand_aawm_dynamic_directives_in_text(
    text: str, available_context: dict[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    matches = list(_AAWM_DYNAMIC_DIRECTIVE_PATTERN.finditer(text))
    if not matches:
        return text, []

    rebuilt_parts: list[str] = []
    injection_events: list[dict[str, Any]] = []
    cursor = 0

    for match in matches:
        rebuilt_parts.append(text[cursor:match.start()])
        directive_attrs = _parse_aawm_directive_attributes(
            _get_aawm_directive_attrs_text(match)
        )
        proc_name = (
            directive_attrs.get("p")
            or directive_attrs.get("proc")
            or "unknown"
        )
        try:
            replacement_text, event = await _resolve_aawm_dynamic_directive(
                directive_attrs,
                available_context,
            )
        except Exception as exc:
            normalized_proc_name = _AAWM_DYNAMIC_PROC_ALIASES.get(proc_name, proc_name)
            replacement_text = _build_aawm_dynamic_injection_failure_text(
                normalized_proc_name
            )
            requested_context_fields = []
            raw_ctx = directive_attrs.get("ctx")
            if raw_ctx:
                requested_context_fields = [
                    field.strip() for field in raw_ctx.split(",") if field.strip()
                ]
            event = {
                "proc": normalized_proc_name,
                "status": "failed",
                "error": exc.__class__.__name__,
                "context_keys": requested_context_fields or list(available_context.keys()),
            }
        rebuilt_parts.append(replacement_text)
        injection_events.append(event)
        cursor = match.end()

    rebuilt_parts.append(text[cursor:])
    return "".join(rebuilt_parts), injection_events


async def _expand_aawm_dynamic_directives_in_value(
    value: Any,
    available_context: dict[str, str],
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            updated_text, injection_events = await _expand_aawm_dynamic_directives_in_text(
                value["text"],
                available_context,
            )
            if injection_events:
                updated_value = dict(value)
                updated_value["text"] = updated_text
                return updated_value, injection_events
            return value, []

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), combined_events

    return value, []


def _add_aawm_dynamic_injection_logging_metadata(
    request_body: dict[str, Any], injection_events: list[dict[str, Any]]
) -> dict[str, Any]:
    proc_names = sorted(
        {
            event["proc"]
            for event in injection_events
            if isinstance(event.get("proc"), str) and event["proc"]
        }
    )
    failure_procs = sorted(
        {
            event["proc"]
            for event in injection_events
            if event.get("status") == "failed"
            and isinstance(event.get("proc"), str)
            and event["proc"]
        }
    )
    context_keys = sorted(
        {
            context_key
            for event in injection_events
            for context_key in event.get("context_keys", [])
            if isinstance(context_key, str) and context_key
        }
    )
    status_values = [
        event["status"]
        for event in injection_events
        if isinstance(event.get("status"), str) and event["status"]
    ]

    tags_to_add = ["aawm-dynamic-injection"]
    tags_to_add.extend(f"aawm-proc:{proc_name}" for proc_name in proc_names)
    if failure_procs:
        tags_to_add.append("aawm-dynamic-injection-failed")

    span_metadata: dict[str, Any] = {
        "injection_count": len(injection_events),
        "failure_count": len(failure_procs),
    }
    if proc_names:
        span_metadata["procs"] = proc_names
    if context_keys:
        span_metadata["context_keys"] = context_keys

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "aawm_dynamic_injection_count": len(injection_events),
            "aawm_dynamic_injection_procs": proc_names,
            "aawm_dynamic_injection_failure_procs": failure_procs,
            "aawm_dynamic_injection_context_keys": context_keys,
            "aawm_dynamic_injection_statuses": status_values,
            "aawm_dynamic_injection_events": injection_events,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="aawm.dynamic_injection",
                    metadata=span_metadata,
                )
            ],
        },
    )


async def _expand_aawm_dynamic_directives_in_anthropic_request_body(
    request_body: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available_context = _build_aawm_context_for_anthropic_request(request_body)
    span_started_at = datetime.now(timezone.utc)
    updated_body = dict(request_body)
    injection_events: list[dict[str, Any]] = []
    changed = False

    for top_level_key in ("system", "messages"):
        if top_level_key not in request_body:
            continue
        updated_value, value_events = await _expand_aawm_dynamic_directives_in_value(
            request_body[top_level_key],
            available_context,
        )
        if value_events:
            updated_body[top_level_key] = updated_value
            injection_events.extend(value_events)
            changed = True

    if not injection_events:
        return request_body, []

    updated_body = _add_aawm_dynamic_injection_logging_metadata(
        updated_body,
        injection_events,
    )
    if changed:
        litellm_metadata = updated_body.get("litellm_metadata")
        if isinstance(litellm_metadata, dict):
            langfuse_spans = litellm_metadata.get("langfuse_spans")
            if isinstance(langfuse_spans, list):
                for span_descriptor in langfuse_spans:
                    if (
                        isinstance(span_descriptor, dict)
                        and span_descriptor.get("name") == "aawm.dynamic_injection"
                    ):
                        span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                            span_started_at
                        )
                        span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                            datetime.now(timezone.utc)
                        )
    return updated_body, injection_events


async def _prepare_anthropic_request_body_for_passthrough(
    request: Request, request_body: dict[str, Any]
) -> Tuple[dict[str, Any], int, set[str], dict[str, str]]:
    updated_body, expanded_count, hooks, _source_metadata_items = (
        _expand_claude_persisted_output_in_anthropic_request_body(request_body)
    )
    billing_header_fields = _extract_anthropic_billing_header_fields_from_request_body(
        updated_body
    )
    updated_body, _claude_system_prompt_override_events = (
        _aawm_replace_claude_system_prompt_in_anthropic_request_body(
            updated_body,
            billing_header_fields,
        )
    )
    updated_body, _claude_prompt_patch_events = (
        _aawm_apply_claude_prompt_patches_to_anthropic_request_body(
            updated_body,
            billing_header_fields,
        )
    )
    updated_body, _aawm_injection_events = (
        await _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body(
            updated_body
        )
    )
    updated_body = _aawm_add_claude_post_rewrite_context_file_logging_metadata(
        updated_body
    )
    if billing_header_fields:
        updated_body = _add_anthropic_billing_header_logging_metadata(
            updated_body,
            billing_header_fields,
        )
    updated_body = _add_route_family_logging_metadata(
        updated_body, "anthropic_messages"
    )
    updated_body = _add_claude_request_breakout_logging_metadata(updated_body)
    updated_body = _prepare_request_body_for_passthrough_observability(
        request=request,
        request_body=updated_body,
    )
    return updated_body, expanded_count, hooks, billing_header_fields


def _request_uses_codex_native_auth(request: Request) -> bool:
    headers = _safe_get_request_headers(request)
    chatgpt_account_id = headers.get("chatgpt-account-id") or headers.get(
        "ChatGPT-Account-Id"
    )
    originator = headers.get("originator") or headers.get("Originator")
    user_agent = headers.get("user-agent") or headers.get("User-Agent")
    session_id = headers.get("session_id") or headers.get("Session_Id")

    if isinstance(chatgpt_account_id, str) and len(chatgpt_account_id) > 0:
        return True
    if isinstance(originator, str) and "codex" in originator.lower():
        return True
    return bool(
        isinstance(user_agent, str)
        and "codex" in user_agent.lower()
        and isinstance(session_id, str)
        and len(session_id) > 0
    )


def _should_preserve_openai_client_auth(request: Request, endpoint: str) -> bool:
    """
    Preserve inbound client auth only for OpenAI Responses passthrough traffic.

    This keeps Codex-style already-authenticated requests as close to native
    behavior as possible while leaving the existing server-authenticated
    passthrough behavior intact for other OpenAI endpoints.
    """
    return _is_openai_responses_endpoint(endpoint) and _request_has_openai_client_auth(
        request
    )


def _get_openai_passthrough_target_base(request: Request, endpoint: str) -> str:
    if _should_preserve_openai_client_auth(request=request, endpoint=endpoint):
        if _request_uses_codex_native_auth(request):
            return os.getenv("CHATGPT_API_BASE") or CHATGPT_API_BASE
    return os.getenv("OPENAI_API_BASE") or "https://api.openai.com/"


def _is_gemini_code_assist_endpoint(endpoint: str) -> bool:
    normalized_endpoint = endpoint.lstrip("/")
    return normalized_endpoint.startswith("v1internal:")


def _get_gemini_passthrough_target_base(
    endpoint: str,
    has_google_oauth_bearer: bool,
) -> str:
    if has_google_oauth_bearer and _is_gemini_code_assist_endpoint(endpoint):
        return os.getenv("CODE_ASSIST_ENDPOINT") or "https://cloudcode-pa.googleapis.com"

    return os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com"


def create_request_copy(request: Request):
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": _safe_get_request_headers(request).copy(),
        "cookies": request.cookies,
        "query_params": dict(request.query_params),
    }


def is_passthrough_request_using_router_model(
    request_body: dict, llm_router: Optional[litellm.Router]
) -> bool:
    """
    Returns True if the model is in the llm_router model names
    """
    try:
        model = request_body.get("model")
        return is_known_model(model, llm_router)
    except Exception:
        return False


def is_passthrough_request_streaming(request_body: dict) -> bool:
    """
    Returns True if the request is streaming
    """
    return request_body.get("stream", False)


async def llm_passthrough_factory_proxy_route(
    custom_llm_provider: str,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Factory function for creating pass-through endpoints for LLM providers.
    """
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    provider_config = ProviderConfigManager.get_provider_model_info(
        provider=LlmProviders(custom_llm_provider),
        model=None,
    )
    if provider_config is None:
        raise HTTPException(
            status_code=404, detail=f"Provider {custom_llm_provider} not found"
        )

    base_target_url = provider_config.get_api_base()

    if base_target_url is None:
        raise HTTPException(
            status_code=404, detail=f"Provider {custom_llm_provider} api base not found"
        )

    if _is_gemini_code_assist_endpoint(endpoint):
        encoded_endpoint = endpoint.split("?", 1)[0]
    else:
        encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    # Join paths correctly by removing trailing/leading slashes as needed
    if not base_url.path or base_url.path == "/":
        # If base URL has no path, just use the new path
        updated_url = base_url.copy_with(path=encoded_endpoint)
    else:
        # Otherwise, combine the paths
        base_path = base_url.path.rstrip("/")
        clean_path = encoded_endpoint.lstrip("/")
        full_path = f"{base_path}/{clean_path}"
        updated_url = base_url.copy_with(path=full_path)

    # Add or update query parameters
    provider_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider=custom_llm_provider,
        region_name=None,
    )

    auth_headers = provider_config.validate_environment(
        headers={},
        model="",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=provider_api_key,
        api_base=base_target_url,
    )

    ## check for streaming
    is_streaming_request = False
    # anthropic is streaming when 'stream' = True is in the body
    if request.method == "POST":
        if "multipart/form-data" not in request.headers.get("content-type", ""):
            _request_body = await request.json()
        else:
            _request_body = await get_form_data(request)

        if _request_body.get("stream"):
            is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers=auth_headers,
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/gemini/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Google AI Studio Pass-through", "pass-through"],
)
async def gemini_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/google_ai_studio)
    """
    ## CHECK FOR LITELLM API KEY IN THE QUERY PARAMS - ?..key=LITELLM_API_KEY
    google_ai_studio_api_key = request.query_params.get("key") or request.headers.get(
        "x-goog-api-key"
    )

    user_api_key_dict = await user_api_key_auth(
        request=request, api_key=f"Bearer {google_ai_studio_api_key}"
    )

    _auth_header = request.headers.get("authorization", "")
    _is_google_oauth = _auth_header.startswith("Bearer ya29.")

    base_target_url = _get_gemini_passthrough_target_base(
        endpoint=endpoint,
        has_google_oauth_bearer=_is_google_oauth,
    )
    if _is_gemini_code_assist_endpoint(endpoint):
        encoded_endpoint = endpoint.split("?", 1)[0]
    else:
        encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    merged_params = dict(request.query_params)
    if _is_google_oauth:
        # Remove the 'key' param if the client sent one; Google OAuth auth
        # does not use API key query params.
        merged_params.pop("key", None)
    else:
        gemini_api_key: Optional[str] = passthrough_endpoint_router.get_credentials(
            custom_llm_provider="gemini",
            region_name=None,
        )
        if gemini_api_key is None:
            raise Exception(
                "Required 'GEMINI_API_KEY'/'GOOGLE_API_KEY' in environment to make pass-through calls to Google AI Studio."
            )
        # Merge query parameters, giving precedence to those in updated_url
        merged_params.update({"key": gemini_api_key})

    ## check for streaming
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True

    if request.method == "POST":
        request_body = await get_request_body(request)
        prepared_request_body = _add_gemini_request_breakout_logging_metadata(
            request_body
        )
        gemini_route_family = (
            "gemini_stream_generate_content"
            if "streamgeneratecontent" in endpoint.lower()
            else "gemini_generate_content"
        )
        prepared_request_body = _add_route_family_logging_metadata(
            prepared_request_body,
            gemini_route_family,
        )
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_llm_provider="gemini",
        _forward_headers=_is_google_oauth,
        is_streaming_request=is_streaming_request,
        query_params=merged_params,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/cohere/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Cohere Pass-through", "pass-through"],
)
async def cohere_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/cohere)
    """
    base_target_url = os.getenv("COHERE_API_BASE") or "https://api.cohere.com"
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    cohere_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="cohere",
        region_name=None,
    )

    ## check for streaming
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": "Bearer {}".format(cohere_api_key)},
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/vllm/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["VLLM Pass-through", "pass-through"],
)
async def vllm_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/vllm)
    """
    from litellm.proxy.proxy_server import llm_router

    request_body = await get_request_body(request)
    is_router_model = is_passthrough_request_using_router_model(
        request_body, llm_router
    )
    is_streaming_request = is_passthrough_request_streaming(request_body)
    if is_router_model and llm_router:
        result = cast(
            httpx.Response,
            await llm_router.allm_passthrough_route(
                model=request_body.get("model"),
                method=request.method,
                endpoint=endpoint,
                request_query_params=request.query_params,
                request_headers=_safe_get_request_headers(request),
                stream=request_body.get("stream", False),
                content=None,
                data=None,
                files=None,
                json=(
                    request_body
                    if request.headers.get("content-type") == "application/json"
                    else None
                ),
                params=None,
                headers=None,
                cookies=None,
            ),
        )

        if is_streaming_request:
            return StreamingResponse(
                content=result.aiter_bytes(),
                status_code=result.status_code,
                headers=HttpPassThroughEndpointHelpers.get_response_headers(
                    headers=result.headers,
                    custom_headers=None,
                ),
            )

        content = await result.aread()
        return Response(
            content=content,
            status_code=result.status_code,
            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                headers=result.headers,
                custom_headers=None,
            ),
        )

    return await llm_passthrough_factory_proxy_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        custom_llm_provider="vllm",
    )


@router.api_route(
    "/mistral/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Mistral Pass-through", "pass-through"],
)
async def mistral_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/mistral)
    """
    base_target_url = os.getenv("MISTRAL_API_BASE") or "https://api.mistral.ai"
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    mistral_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="mistral",
        region_name=None,
    )

    ## check for streaming
    is_streaming_request = False
    # anthropic is streaming when 'stream' = True is in the body
    if request.method == "POST":
        _request_body = await request.json()
        if _request_body.get("stream"):
            is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": "Bearer {}".format(mistral_api_key)},
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/milvus/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Milvus Pass-through", "pass-through"],
)
async def milvus_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Enable using Milvus `/vectors` endpoint as a pass-through endpoint.
    """

    provider_config = ProviderConfigManager.get_provider_vector_stores_config(
        provider=LlmProviders.MILVUS
    )
    if not provider_config:
        raise HTTPException(
            status_code=500,
            detail="Unable to find Milvus vector store config.",
        )

    # check if managed vector store index is used
    request_body = await get_request_body(request)

    # check collectionName
    collection_name = cast(Optional[str], request_body.get("collectionName"))
    extra_headers = {}
    base_target_url: Optional[str] = None
    if not collection_name:
        raise HTTPException(
            status_code=400,
            detail=f"Collection name is required. Got {request_body}",
        )

    if not litellm.vector_store_index_registry or not litellm.vector_store_registry:
        raise HTTPException(
            status_code=500,
            detail="Unable to find Milvus vector store index registry or vector store registry.",
        )

    # check if vector store index
    is_vector_store_index = litellm.vector_store_index_registry.is_vector_store_index(
        vector_store_index_name=collection_name
    )

    if not is_vector_store_index:
        raise HTTPException(
            status_code=400,
            detail=f"Collection {collection_name} is not a litellm managed vector store index. Only litellm managed vector store indexes are supported.",
        )

    is_allowed_to_call_vector_store_endpoint(
        index_name=collection_name,
        provider=LlmProviders.MILVUS,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    # get the vector store name from index registry

    index_object = (
        (
            litellm.vector_store_index_registry.get_vector_store_index_by_name(
                vector_store_index_name=collection_name
            )
        )
        if litellm.vector_store_index_registry is not None
        else None
    )
    if index_object is None:
        raise Exception(f"Vector store index not found for {collection_name}")

    vector_store_name = index_object.litellm_params.vector_store_name
    vector_store_index = index_object.litellm_params.vector_store_index

    request_body["collectionName"] = vector_store_index

    # Update the request object with the modified collection name
    _safe_set_request_parsed_body(request, request_body)

    vector_store = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry_by_name(
        vector_store_name=vector_store_name
    )
    if vector_store is None:
        raise Exception(f"Vector store not found for {vector_store_name}")
    litellm_params = vector_store.get("litellm_params") or {}
    auth_credentials = provider_config.get_auth_credentials(
        litellm_params=litellm_params
    )

    extra_headers = auth_credentials.get("headers") or {}

    litellm_params = vector_store.get("litellm_params") or {}

    base_target_url = provider_config.get_complete_url(
        api_base=litellm_params.get("api_base"), litellm_params=litellm_params
    )

    if base_target_url is None:
        raise Exception(
            f"api_base not found in vector store configuration for {vector_store_name}"
        )

    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)
    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers=extra_headers,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


async def is_streaming_request_fn(request: Request) -> bool:
    if request.method == "POST":
        content_type = request.headers.get("content-type", None)
        if content_type and "multipart/form-data" in content_type:
            _request_body = await get_form_data(request)
        else:
            _request_body = await _read_request_body(request)
        if _request_body.get("stream"):
            return True
    return False


@router.api_route(
    "/anthropic/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Anthropic Pass-through", "pass-through"],
)
async def anthropic_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/anthropic_completion)
    """
    base_target_url = os.getenv("ANTHROPIC_API_BASE") or "https://api.anthropic.com"
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    anthropic_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="anthropic",
        region_name=None,
    )

    custom_headers = {}
    if (
        "authorization" not in request.headers
        and "x-api-key" not in request.headers
        and anthropic_api_key is not None
    ):
        custom_headers["x-api-key"] = "{}".format(anthropic_api_key)

    if request.method == "POST":
        request_body = await get_request_body(request)
        (
            prepared_request_body,
            expanded_count,
            hooks,
            billing_header_fields,
        ) = await _prepare_anthropic_request_body_for_passthrough(request, request_body)
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)
            verbose_proxy_logger.debug(
                "Prepared Anthropic passthrough request body; expanded_persisted_output=%s hooks=%s billing_header_keys=%s",
                expanded_count,
                sorted(hooks),
                sorted(billing_header_fields),
            )

    ## check for streaming
    is_streaming_request = await is_streaming_request_fn(request)

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers=custom_headers,
        _forward_headers=True,
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


# Bedrock endpoint actions - consolidated list used for model extraction and streaming detection
BEDROCK_ENDPOINT_ACTIONS = {
    "invoke",
    "invoke-with-response-stream",
    "converse",
    "converse-stream",
    "count_tokens",
    "count-tokens",
}

BEDROCK_STREAMING_ACTIONS = {"invoke-with-response-stream", "converse-stream"}


def _extract_model_from_bedrock_endpoint(endpoint: str) -> str:
    """
    Extract model name from Bedrock endpoint path.

    Handles model names with slashes (e.g., aws/anthropic/bedrock-claude-3-5-sonnet-v1)
    by finding the action in the endpoint and extracting everything between "model" and the action.

    Args:
        endpoint: The endpoint path (e.g., "/model/aws/anthropic/model-name/invoke" or "v2/model/model-name/invoke")

    Returns:
        The extracted model name (e.g., "aws/anthropic/model-name" or "model-name")

    Raises:
        ValueError: If model cannot be extracted from endpoint
    """
    try:
        endpoint_parts = endpoint.split("/")

        if "application-inference-profile" in endpoint:
            # Format: model/application-inference-profile/{profile-id}/{action}
            return "/".join(endpoint_parts[1:3])

        # Format: model/{modelId}/{action} or v2/model/{modelId}/{action}
        # Find the index of "model" in the endpoint parts
        model_index = None
        for idx, part in enumerate(endpoint_parts):
            if part == "model":
                model_index = idx
                break

        # If "model" keyword not found, try to extract model from the endpoint
        # by finding the action and taking everything before it
        if model_index is None:
            # Find the index of the action in the endpoint parts
            action_index = None
            for idx, part in enumerate(endpoint_parts):
                if part in BEDROCK_ENDPOINT_ACTIONS:
                    action_index = idx
                    break

            if action_index is not None and action_index > 1:
                # Join all parts before the action (excluding empty strings)
                model_parts = [p for p in endpoint_parts[1:action_index] if p]
                if model_parts:
                    return "/".join(model_parts)

            raise ValueError(
                f"'model' keyword not found and unable to extract model from endpoint. Expected format: /model/{{modelId}}/{{action}}. Got: {endpoint}"
            )

        # Find the index of the action in the endpoint parts
        action_index = None
        for idx, part in enumerate(endpoint_parts):
            if part in BEDROCK_ENDPOINT_ACTIONS:
                action_index = idx
                break

        if action_index is not None and action_index > model_index + 1:
            # Join all parts between "model" and the action (excluding "model" itself)
            return "/".join(endpoint_parts[model_index + 1 : action_index])

        # Fallback to taking everything after "model" if no action found
        model_parts = [p for p in endpoint_parts[model_index + 1 :] if p]
        if model_parts:
            return "/".join(model_parts)

        raise ValueError(
            f"No model ID found after 'model' keyword. Expected format: /model/{{modelId}}/{{action}}. Got: {endpoint}"
        )

    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        raise ValueError(
            f"Model missing from endpoint. Expected format: /model/{{modelId}}/{{action}}. Got: {endpoint}"
        ) from e


async def handle_bedrock_passthrough_router_model(
    model: str,
    endpoint: str,
    request: Request,
    request_body: dict,
    llm_router: litellm.Router,
    user_api_key_dict: UserAPIKeyAuth,
    proxy_logging_obj,
    general_settings: dict,
    proxy_config,
    select_data_generator,
    user_model: Optional[str],
    user_temperature: Optional[float],
    user_request_timeout: Optional[float],
    user_max_tokens: Optional[int],
    user_api_base: Optional[str],
    version: Optional[str],
) -> Union[Response, StreamingResponse]:
    """
    Handle Bedrock passthrough for router models (models defined in config.yaml).

    Uses the same common processing path as non-router models to ensure
    metadata and hooks are properly initialized.

    Args:
        model: The router model name (e.g., "aws/anthropic/bedrock-claude-3-5-sonnet-v1")
        endpoint: The Bedrock endpoint path (e.g., "/model/{modelId}/invoke")
        request: The FastAPI request object
        request_body: The parsed request body
        llm_router: The LiteLLM router instance
        user_api_key_dict: The user API key authentication dictionary
        (additional args for common processing)

    Returns:
        Response or StreamingResponse depending on endpoint type
    """
    from fastapi import Response as FastAPIResponse

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    # Detect streaming based on endpoint
    is_streaming = any(action in endpoint for action in BEDROCK_STREAMING_ACTIONS)

    verbose_proxy_logger.debug(
        f"Bedrock router passthrough: model='{model}', endpoint='{endpoint}', streaming={is_streaming}"
    )

    # Use the common processing path (same as non-router models)
    # This ensures all metadata, hooks, and logging are properly initialized
    data: Dict[str, Any] = {}
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)

    data["model"] = model
    data["method"] = request.method
    data["endpoint"] = endpoint
    data["data"] = request_body
    data["custom_llm_provider"] = "bedrock"

    # Use the common passthrough processing to handle metadata and hooks
    # This also handles all response formatting (streaming/non-streaming) and exceptions
    try:
        result = await base_llm_response_processor.base_passthrough_process_llm_request(
            request=request,
            fastapi_response=FastAPIResponse(),
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        return result
    except Exception as e:
        # Use common exception handling
        raise await base_llm_response_processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )


async def handle_bedrock_count_tokens(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Handle AWS Bedrock CountTokens API requests.

    This function processes count_tokens endpoints like:
    - /v1/messages/count_tokens
    - /v1/messages/count-tokens
    """
    from litellm.llms.bedrock.common_utils import BedrockError
    from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
    from litellm.proxy.proxy_server import llm_router

    try:
        # Initialize the handler
        handler = BedrockCountTokensHandler()

        # Extract model from request body
        model = request_body.get("model")
        if not model:
            raise HTTPException(
                status_code=400, detail={"error": "Model is required in request body"}
            )

        # Get model parameters from router
        litellm_params = {"user_api_key_dict": user_api_key_dict}
        resolved_model = model  # Default fallback

        if llm_router:
            deployments = llm_router.get_model_list(model_name=model)
            if deployments and len(deployments) > 0:
                # Get the first matching deployment
                deployment = deployments[0]
                model_litellm_params = deployment.get("litellm_params", {})

                # Get the resolved model ID from the configuration
                if "model" in model_litellm_params:
                    resolved_model = model_litellm_params["model"]

                # Copy all litellm_params - BaseAWSLLM will handle AWS credential discovery
                for key, value in model_litellm_params.items():
                    if key != "user_api_key_dict":  # Don't overwrite user_api_key_dict
                        litellm_params[key] = value  # type: ignore

        verbose_proxy_logger.debug(f"Count tokens litellm_params: {litellm_params}")
        verbose_proxy_logger.debug(f"Resolved model: {resolved_model}")

        # Handle the count tokens request
        result = await handler.handle_count_tokens_request(
            request_data=request_body,
            litellm_params=litellm_params,
            resolved_model=resolved_model,
        )

        return result

    except BedrockError as e:
        # Convert BedrockError to HTTPException for FastAPI
        verbose_proxy_logger.error(
            f"BedrockError in handle_bedrock_count_tokens: {str(e)}"
        )
        raise HTTPException(status_code=e.status_code, detail={"error": e.message})
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error in handle_bedrock_count_tokens: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": f"CountTokens processing error: {str(e)}"}
        )


async def bedrock_llm_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Handles Bedrock LLM API calls.

    Supports both direct Bedrock models and router models from config.yaml.

    Endpoints:
    - /model/{modelId}/invoke
    - /model/{modelId}/invoke-with-response-stream
    - /model/{modelId}/converse
    - /model/{modelId}/converse-stream
    - /model/application-inference-profile/{profileId}/{action}
    """
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    request_body = await _read_request_body(request=request)

    # Special handling for count_tokens endpoints
    if "count_tokens" in endpoint or "count-tokens" in endpoint:
        return await handle_bedrock_count_tokens(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            request_body=request_body,
        )

    # Extract model from endpoint path using helper
    try:
        model = _extract_model_from_bedrock_endpoint(endpoint=endpoint)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)},
        )

    # Check if this is a router model (from config.yaml)
    is_router_model = is_passthrough_request_using_router_model(
        request_body={"model": model}, llm_router=llm_router
    )

    # If router model, use dedicated router passthrough handler
    # This uses the same common processing path as non-router models
    if is_router_model and llm_router:
        return await handle_bedrock_passthrough_router_model(
            model=model,
            endpoint=endpoint,
            request=request,
            request_body=request_body,
            llm_router=llm_router,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

    # Fall back to existing implementation for direct Bedrock models
    verbose_proxy_logger.debug(
        f"Bedrock passthrough: Using direct Bedrock model '{model}' for endpoint '{endpoint}'"
    )

    data: Dict[str, Any] = {}
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)

    data["method"] = request.method
    data["endpoint"] = endpoint
    data["data"] = request_body
    data["custom_llm_provider"] = "bedrock"

    try:
        result = await base_llm_response_processor.base_passthrough_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        return result
    except Exception as e:
        raise await base_llm_response_processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )


@router.api_route(
    "/bedrock/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Bedrock Pass-through", "pass-through"],
)
async def bedrock_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    This is the v1 passthrough for Bedrock.
    V2 is handled by the `/bedrock/v2` endpoint.
    [Docs](https://docs.litellm.ai/docs/pass_through/bedrock)
    """
    create_request_copy(request)

    try:
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials
    except ImportError:
        raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

    aws_region_name = litellm.utils.get_secret(secret_name="AWS_REGION_NAME")
    if _is_bedrock_agent_runtime_route(endpoint=endpoint):  # handle bedrock agents
        base_target_url = (
            f"https://bedrock-agent-runtime.{aws_region_name}.amazonaws.com"
        )
    else:
        return await bedrock_llm_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
        )
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    from litellm.llms.bedrock.chat import BedrockConverseLLM

    bedrock_llm = BedrockConverseLLM()
    credentials: Credentials = bedrock_llm.get_credentials()  # type: ignore
    sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
    headers = {"Content-Type": "application/json"}
    # Assuming the body contains JSON data, parse it
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": e})
    _request = AWSRequest(
        method="POST", url=str(updated_url), data=json.dumps(data), headers=headers
    )
    sigv4.add_auth(_request)
    prepped = _request.prepare()

    ## check for streaming
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(prepped.url),
        custom_headers=prepped.headers,  # type: ignore
        is_streaming_request=is_streaming_request,
        _forward_headers=True,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
        custom_body=data,  # type: ignore
    )

    return received_value


def _resolve_vertex_model_from_router(
    model_id: str,
    llm_router: Optional[litellm.Router],
    encoded_endpoint: str,
    endpoint: str,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Resolve Vertex AI model configuration from router.

    Args:
        model_id: The model ID extracted from the URL (e.g., "gcp/google/gemini-2.5-flash")
        llm_router: The LiteLLM router instance
        encoded_endpoint: The encoded endpoint path
        endpoint: The original endpoint path
        vertex_project: Current vertex project (may be from URL)
        vertex_location: Current vertex location (may be from URL)

    Returns:
        Tuple of (encoded_endpoint, endpoint, vertex_project, vertex_location)
        with resolved values from router config
    """
    if not llm_router:
        return encoded_endpoint, endpoint, vertex_project, vertex_location

    try:
        deployment = llm_router.get_available_deployment_for_pass_through(
            model=model_id
        )
        if not deployment:
            return encoded_endpoint, endpoint, vertex_project, vertex_location

        litellm_params = deployment.get("litellm_params", {})

        # Always override with router config values (they take precedence over URL values)
        config_vertex_project = litellm_params.get("vertex_project")
        config_vertex_location = litellm_params.get("vertex_location")
        if config_vertex_project:
            vertex_project = config_vertex_project
        if config_vertex_location:
            vertex_location = config_vertex_location

        # Get the actual Vertex AI model name by stripping the provider prefix
        # e.g., "vertex_ai/gemini-2.0-flash-exp" -> "gemini-2.0-flash-exp"
        model_from_config = litellm_params.get("model", "")
        if model_from_config:
            # get_llm_provider returns (model, custom_llm_provider, dynamic_api_key, api_base)
            # For "vertex_ai/gemini-2.0-flash-exp" it returns:
            # model="gemini-2.0-flash-exp", custom_llm_provider="vertex_ai"
            actual_model, custom_llm_provider, _, _ = get_llm_provider(
                model=model_from_config
            )

            # Log only non-sensitive information (model names and provider), never API keys or secrets.
            safe_actual_model = actual_model
            safe_custom_llm_provider = custom_llm_provider
            verbose_proxy_logger.debug(
                "get_llm_provider returned: actual_model=%s, custom_llm_provider=%s, model_id=%s",
                safe_actual_model,
                safe_custom_llm_provider,
                model_id,
            )

            if actual_model and model_id != actual_model:
                verbose_proxy_logger.debug(
                    "Resolved router model '%s' to '%s' (provider=%s) with project=%s, location=%s",
                    model_id,
                    actual_model,
                    custom_llm_provider,
                    vertex_project,
                    vertex_location,
                )
                encoded_endpoint = encoded_endpoint.replace(model_id, actual_model)
                endpoint = endpoint.replace(model_id, actual_model)

    except Exception as e:
        verbose_proxy_logger.debug(
            f"Error resolving vertex model from router for model {model_id}: {e}"
        )

    return encoded_endpoint, endpoint, vertex_project, vertex_location


def _is_bedrock_agent_runtime_route(endpoint: str) -> bool:
    """
    Return True, if the endpoint should be routed to the `bedrock-agent-runtime` endpoint.
    """
    for _route in BEDROCK_AGENT_RUNTIME_PASS_THROUGH_ROUTES:
        if _route in endpoint:
            return True
    return False


@router.api_route(
    "/assemblyai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["AssemblyAI Pass-through", "pass-through"],
)
@router.api_route(
    "/eu.assemblyai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["AssemblyAI EU Pass-through", "pass-through"],
)
async def assemblyai_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.assembly_passthrough_logging_handler import (
        AssemblyAIPassthroughLoggingHandler,
    )

    """
    [Docs](https://api.assemblyai.com)
    """
    # Set base URL based on the route
    assembly_region = AssemblyAIPassthroughLoggingHandler._get_assembly_region_from_url(
        url=str(request.url)
    )
    base_target_url = (
        AssemblyAIPassthroughLoggingHandler._get_assembly_base_url_from_region(
            region=assembly_region
        )
    )
    encoded_endpoint = httpx.URL(endpoint).path
    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    assemblyai_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="assemblyai",
        region_name=assembly_region,
    )

    ## check for streaming
    is_streaming_request = False
    # assemblyai is streaming when 'stream' = True is in the body
    if request.method == "POST":
        _request_body = await request.json()
        if _request_body.get("stream"):
            is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": "{}".format(assemblyai_api_key)},
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/azure_ai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Azure AI Pass-through", "pass-through"],
)
@router.api_route(
    "/azure/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Azure Pass-through", "pass-through"],
)
async def azure_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Call any azure endpoint using the proxy.

    Just use `{PROXY_BASE_URL}/azure/{endpoint:path}`

    Checks if the deployment id in the url is a litellm model name. If so, it will route using the llm_router.allm_passthrough_route.
    """
    from litellm.proxy.proxy_server import llm_router

    parts = endpoint.split(
        "/"
    )  # azure model is in the url - e.g. https://{endpoint}/openai/deployments/{deployment-id}/completions?api-version=2024-10-21

    if len(parts) > 1 and llm_router:
        for part in parts:
            # check if LLM MODEL
            is_router_model = is_passthrough_request_using_router_model(
                request_body={"model": part}, llm_router=llm_router
            )
            # check if vector store index
            is_vector_store_index = (
                (
                    litellm.vector_store_index_registry.is_vector_store_index(
                        vector_store_index_name=part
                    )
                )
                if litellm.vector_store_index_registry is not None
                else False
            )

            if is_router_model:
                request_body = await get_request_body(request)
                is_streaming_request = is_passthrough_request_streaming(request_body)
                result = await llm_router.allm_passthrough_route(
                    model=part,
                    method=request.method,
                    endpoint=endpoint,
                    request_query_params=request.query_params,
                    request_headers=_safe_get_request_headers(request),
                    stream=request_body.get("stream", False),
                    content=None,
                    data=None,
                    files=None,
                    json=(
                        request_body
                        if request.headers.get("content-type") == "application/json"
                        else None
                    ),
                    params=None,
                    headers=None,
                    cookies=None,
                )

                if is_streaming_request:
                    # Check if result is an async generator (from _async_streaming)
                    import inspect

                    if inspect.isasyncgen(result):
                        # Result is already an async generator, use it directly
                        return StreamingResponse(
                            content=result,
                            status_code=200,
                            headers={"content-type": "text/event-stream"},
                        )
                    else:
                        # Result is an httpx.Response, use aiter_bytes()
                        result = cast(httpx.Response, result)
                        return StreamingResponse(
                            content=result.aiter_bytes(),
                            status_code=result.status_code,
                            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                                headers=result.headers,
                                custom_headers=None,
                            ),
                        )

                # Non-streaming response
                result = cast(httpx.Response, result)
                content = await result.aread()
                return Response(
                    content=content,
                    status_code=result.status_code,
                    headers=HttpPassThroughEndpointHelpers.get_response_headers(
                        headers=result.headers,
                        custom_headers=None,
                    ),
                )
            elif is_vector_store_index:
                # get the api key from the provider config
                provider_config = (
                    ProviderConfigManager.get_provider_vector_stores_config(
                        provider=litellm.LlmProviders.AZURE_AI
                    )
                )
                if provider_config is None:
                    raise Exception("Provider config not found for Azure AI")
                # get the index from registry
                if litellm.vector_store_registry is None:
                    raise Exception("Vector store registry not found")

                is_allowed_to_call_vector_store_endpoint(
                    index_name=part,
                    provider=litellm.LlmProviders.AZURE_AI,
                    request=request,
                    user_api_key_dict=user_api_key_dict,
                )
                # get the vector store name from index registry
                index_object = (
                    (
                        litellm.vector_store_index_registry.get_vector_store_index_by_name(
                            vector_store_index_name=part
                        )
                    )
                    if litellm.vector_store_index_registry is not None
                    else None
                )
                if index_object is None:
                    raise Exception(f"Vector store index not found for {part}")

                vector_store_name = index_object.litellm_params.vector_store_name

                vector_store = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry_by_name(
                    vector_store_name=vector_store_name
                )
                if vector_store is None:
                    raise Exception(f"Vector store not found for {vector_store_name}")
                litellm_params = vector_store.get("litellm_params") or {}
                auth_credentials = provider_config.get_auth_credentials(
                    litellm_params=litellm_params
                )

                extra_headers = auth_credentials.get("headers") or {}

                base_target_url = litellm_params.get("api_base")
                if base_target_url is None:
                    raise Exception(f"API base not found for {part}")
                return await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                    endpoint=endpoint,
                    request=request,
                    fastapi_response=fastapi_response,
                    user_api_key_dict=user_api_key_dict,
                    base_target_url=base_target_url,
                    api_key=None,
                    custom_llm_provider=litellm.LlmProviders.AZURE_AI,
                    extra_headers=cast(dict, extra_headers),
                )

    base_target_url = get_secret_str(secret_name="AZURE_API_BASE")
    if base_target_url is None:
        raise Exception(
            "Required 'AZURE_API_BASE' in environment to make pass-through calls to Azure."
        )
    # Add or update query parameters
    azure_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider=litellm.LlmProviders.AZURE.value,
        region_name=None,
    )
    if azure_api_key is None:
        raise Exception(
            "Required 'AZURE_API_KEY' in environment to make pass-through calls to Azure."
        )

    return await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        base_target_url=base_target_url,
        api_key=azure_api_key,
        custom_llm_provider=litellm.LlmProviders.AZURE,
    )


from abc import ABC, abstractmethod


class BaseVertexAIPassThroughHandler(ABC):
    @staticmethod
    @abstractmethod
    def get_default_base_target_url(vertex_location: Optional[str]) -> str:
        pass

    @staticmethod
    @abstractmethod
    def update_base_target_url_with_credential_location(
        base_target_url: str, vertex_location: Optional[str]
    ) -> str:
        pass


class VertexAIDiscoveryPassThroughHandler(BaseVertexAIPassThroughHandler):
    @staticmethod
    def get_default_base_target_url(vertex_location: Optional[str]) -> str:
        return "https://discoveryengine.googleapis.com/"

    @staticmethod
    def update_base_target_url_with_credential_location(
        base_target_url: str, vertex_location: Optional[str]
    ) -> str:
        return base_target_url


class VertexAIPassThroughHandler(BaseVertexAIPassThroughHandler):
    @staticmethod
    def get_default_base_target_url(vertex_location: Optional[str]) -> str:
        return get_vertex_base_url(vertex_location)

    @staticmethod
    def update_base_target_url_with_credential_location(
        base_target_url: str, vertex_location: Optional[str]
    ) -> str:
        return get_vertex_base_url(vertex_location)


def get_vertex_base_url(vertex_location: Optional[str]) -> str:
    """
    Returns the base URL for Vertex AI based on the provided location.
    """
    if vertex_location == "global":
        return "https://aiplatform.googleapis.com/"
    return f"https://{vertex_location}-aiplatform.googleapis.com/"


def get_vertex_ai_allowed_incoming_headers(request: Request) -> dict:
    """
    Extract only the allowed headers from incoming request for Vertex AI pass-through.

    Uses an allowlist approach for security - only forwards headers we explicitly trust.
    This prevents accidentally forwarding sensitive headers like the LiteLLM auth token.

    Args:
        request: The FastAPI request object

    Returns:
        dict: Headers dictionary with only allowed headers
    """
    incoming_headers = _safe_get_request_headers(request)
    headers = {}
    for header_name in ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS:
        if header_name in incoming_headers:
            headers[header_name] = incoming_headers[header_name]
    return headers


def get_vertex_pass_through_handler(
    call_type: Literal["discovery", "aiplatform"],
) -> BaseVertexAIPassThroughHandler:
    if call_type == "discovery":
        return VertexAIDiscoveryPassThroughHandler()
    elif call_type == "aiplatform":
        return VertexAIPassThroughHandler()
    else:
        raise ValueError(f"Invalid call type: {call_type}")


def _override_vertex_params_from_router_credentials(
    router_credentials: Optional[Any],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Override vertex_project and vertex_location with values from router_credentials if available.

    Args:
        router_credentials: Optional vector store credentials from registry (LiteLLM_ManagedVectorStore)
        vertex_project: Current vertex project ID (from URL)
        vertex_location: Current vertex location (from URL)

    Returns:
        Tuple of (vertex_project, vertex_location) with overridden values if applicable
    """
    if router_credentials is None:
        return vertex_project, vertex_location

    verbose_proxy_logger.debug(
        "Using vector store credentials to override vertex project and location"
    )

    litellm_params = router_credentials.get("litellm_params", {})
    if not litellm_params:
        verbose_proxy_logger.warning(
            "Vector store credentials found but litellm_params is empty"
        )
        return vertex_project, vertex_location

    # Extract vertex_project and vertex_location from litellm_params
    vector_store_project = litellm_params.get("vertex_project")
    vector_store_location = litellm_params.get("vertex_location")

    if vector_store_project:
        verbose_proxy_logger.debug(
            "Overriding vertex_project from URL (%s) with vector store value: %s",
            vertex_project,
            vector_store_project,
        )
        vertex_project = vector_store_project
    else:
        verbose_proxy_logger.warning(
            "Vector store credentials found but missing vertex_project in litellm_params"
        )

    if vector_store_location:
        verbose_proxy_logger.debug(
            "Overriding vertex_location from URL (%s) with vector store value: %s",
            vertex_location,
            vector_store_location,
        )
        vertex_location = vector_store_location
    else:
        verbose_proxy_logger.warning(
            "Vector store credentials found but missing vertex_location in litellm_params"
        )

    return vertex_project, vertex_location


async def _prepare_vertex_auth_headers(
    request: Request,
    vertex_credentials: Optional[Any],
    router_credentials: Optional[Any],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    base_target_url: Optional[str],
    get_vertex_pass_through_handler: BaseVertexAIPassThroughHandler,
) -> Tuple[dict, Optional[str], bool, Optional[str], Optional[str]]:
    """
    Prepare authentication headers for Vertex AI pass-through requests.

    Args:
        request: FastAPI request object
        vertex_credentials: Vertex AI credentials from config
        router_credentials: Optional vector store credentials from registry
        vertex_project: Vertex project ID
        vertex_location: Vertex location
        base_target_url: Base URL for the Vertex AI service
        get_vertex_pass_through_handler: Handler for the specific Vertex AI service

    Returns:
        Tuple containing:
            - headers: dict - Authentication headers to use
            - base_target_url: Optional[str] - Updated base target URL
            - headers_passed_through: bool - Whether headers were passed through from request
            - vertex_project: Optional[str] - Updated vertex project ID
            - vertex_location: Optional[str] - Updated vertex location
    """
    vertex_llm_base = VertexBase()
    headers_passed_through = False

    # Use headers from the incoming request if no vertex credentials are found
    if (
        vertex_credentials is None or vertex_credentials.vertex_project is None
    ) and router_credentials is None:
        headers = _safe_get_request_headers(request).copy()
        headers_passed_through = True
        verbose_proxy_logger.debug(
            "default_vertex_config  not set, incoming request headers %s", headers
        )
        headers.pop("content-length", None)
        headers.pop("host", None)
    else:
        if router_credentials is not None:
            vertex_credentials_str = None
        elif vertex_credentials is not None:
            # Use credentials from vertex_credentials
            # When vertex_credentials are provided (including default credentials),
            # use their project/location values if available
            if vertex_credentials.vertex_project is not None:
                vertex_project = vertex_credentials.vertex_project
            if vertex_credentials.vertex_location is not None:
                vertex_location = vertex_credentials.vertex_location
            vertex_credentials_str = vertex_credentials.vertex_credentials
        else:
            raise ValueError("No vertex credentials found")

        _auth_header, vertex_project = await vertex_llm_base._ensure_access_token_async(
            credentials=vertex_credentials_str,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )

        auth_header, _ = vertex_llm_base._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials_str,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base="",
        )

        # Use allowlist approach - only forward specific safe headers
        headers = get_vertex_ai_allowed_incoming_headers(request)
        # Add the Authorization header with vendor credentials
        headers["Authorization"] = f"Bearer {auth_header}"

        if base_target_url is not None:
            base_target_url = get_vertex_pass_through_handler.update_base_target_url_with_credential_location(
                base_target_url, vertex_location
            )

    return (
        headers,
        base_target_url,
        headers_passed_through,
        vertex_project,
        vertex_location,
    )


async def _base_vertex_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    get_vertex_pass_through_handler: BaseVertexAIPassThroughHandler,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    router_credentials: Optional[Any] = None,
):
    """
    Base function for Vertex AI passthrough routes.
    Handles common logic for all Vertex AI services.

    Default base_target_url is `https://{vertex_location}-aiplatform.googleapis.com/`

    Args:
        endpoint: The endpoint path
        request: FastAPI request object
        fastapi_response: FastAPI response object
        get_vertex_pass_through_handler: Handler for the specific Vertex AI service
        user_api_key_dict: User API key authentication dict
        router_credentials: Optional vector store credentials from registry (LiteLLM_ManagedVectorStore)
    """
    from litellm.llms.vertex_ai.common_utils import (
        construct_target_url,
        get_vertex_location_from_url,
        get_vertex_model_id_from_url,
        get_vertex_project_id_from_url,
    )
    from litellm.proxy.proxy_server import llm_router

    encoded_endpoint = httpx.URL(endpoint).path
    verbose_proxy_logger.debug("requested endpoint %s", endpoint)
    headers: dict = {}
    api_key_to_use = get_litellm_virtual_key(request=request)
    user_api_key_dict = await user_api_key_auth(
        request=request,
        api_key=api_key_to_use,
    )

    if user_api_key_dict is None:
        api_key_to_use = get_litellm_virtual_key(request=request)
        user_api_key_dict = await user_api_key_auth(
            request=request,
            api_key=api_key_to_use,
        )

    vertex_project: Optional[str] = get_vertex_project_id_from_url(endpoint)
    vertex_location: Optional[str] = get_vertex_location_from_url(endpoint)

    # Override with vector store credentials if available
    vertex_project, vertex_location = _override_vertex_params_from_router_credentials(
        router_credentials=router_credentials,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )

    # Check if model is in router config - always do this to resolve custom model names
    model_id = get_vertex_model_id_from_url(endpoint)
    if model_id:
        if llm_router:
            # Resolve model configuration from router
            (
                encoded_endpoint,
                endpoint,
                vertex_project,
                vertex_location,
            ) = _resolve_vertex_model_from_router(
                model_id=model_id,
                llm_router=llm_router,
                encoded_endpoint=encoded_endpoint,
                endpoint=endpoint,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
            )

    vertex_credentials = passthrough_endpoint_router.get_vertex_credentials(
        project_id=vertex_project,
        location=vertex_location,
    )

    base_target_url = get_vertex_pass_through_handler.get_default_base_target_url(
        vertex_location
    )

    # Prepare authentication headers
    (
        headers,
        base_target_url,
        headers_passed_through,
        vertex_project,
        vertex_location,
    ) = await _prepare_vertex_auth_headers(  # type: ignore
        request=request,
        vertex_credentials=vertex_credentials,
        router_credentials=router_credentials,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        base_target_url=base_target_url,
        get_vertex_pass_through_handler=get_vertex_pass_through_handler,
    )

    if base_target_url is None:
        base_target_url = get_vertex_base_url(vertex_location)

    request_route = encoded_endpoint
    verbose_proxy_logger.debug("request_route %s", request_route)

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    updated_url = construct_target_url(
        base_url=base_target_url,
        requested_route=encoded_endpoint,
        vertex_location=vertex_location,
        vertex_project=vertex_project,
    )

    verbose_proxy_logger.debug("updated url %s", updated_url)

    ## check for streaming
    target = str(updated_url)
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True
        target += "?alt=sse"

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=target,
        custom_headers=headers,
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path

    try:
        received_value = await endpoint_func(
            request,
            fastapi_response,
            user_api_key_dict,
        )
    except ProxyException as e:
        if headers_passed_through:
            e.message = f"No credentials found on proxy for project_name={vertex_project} + location={vertex_location}, check `/model/info` for allowed project + region combinations with `use_in_pass_through: true`. Headers were passed through directly but request failed with error: {e.message}"
        raise e

    return received_value


@router.api_route(
    "/vertex_ai/discovery/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Vertex AI Pass-through", "pass-through"],
)
async def vertex_discovery_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    Call any vertex discovery endpoint using the proxy.

    Just use `{PROXY_BASE_URL}/vertex_ai/discovery/{endpoint:path}`

    Target url: `https://discoveryengine.googleapis.com`
    """
    import re

    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    # Extract vector store ID from endpoint if present (e.g., dataStores/test-litellm-app_1761094730750)
    vector_store_credentials: Optional[LiteLLM_ManagedVectorStore] = None
    vector_store_id_match = re.search(r"dataStores/([^/]+)", endpoint)

    if vector_store_id_match:
        vector_store_id = vector_store_id_match.group(1)
        verbose_proxy_logger.debug(
            "Extracted vector store ID from endpoint: %s", vector_store_id
        )

        # Retrieve vector store credentials from the registry
        vector_store_credentials = (
            passthrough_endpoint_router.get_vector_store_credentials(
                vector_store_id=vector_store_id
            )
        )

        if vector_store_credentials:
            verbose_proxy_logger.debug(
                "Found vector store credentials for ID: %s", vector_store_id
            )
        else:
            verbose_proxy_logger.warning(
                "Vector store ID %s found in endpoint but no credentials found in registry",
                vector_store_id,
            )

    discovery_handler = get_vertex_pass_through_handler(call_type="discovery")
    return await _base_vertex_proxy_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        get_vertex_pass_through_handler=discovery_handler,
        router_credentials=vector_store_credentials,
    )


@router.api_route(
    "/vertex-ai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Vertex AI Pass-through", "pass-through"],
    include_in_schema=False,
)
@router.api_route(
    "/vertex_ai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Vertex AI Pass-through", "pass-through"],
)
async def vertex_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Call LiteLLM proxy via Vertex AI SDK.

    [Docs](https://docs.litellm.ai/docs/pass_through/vertex_ai)
    """
    ai_platform_handler = get_vertex_pass_through_handler(call_type="aiplatform")

    return await _base_vertex_proxy_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        get_vertex_pass_through_handler=ai_platform_handler,
        user_api_key_dict=user_api_key_dict,
    )


@router.api_route(
    "/openai_passthrough/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["OpenAI Pass-through", "pass-through"],
)
@router.api_route(
    "/openai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["OpenAI Pass-through", "pass-through"],
)
async def openai_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Pass-through endpoint for OpenAI API calls.

    Available on both routes:
    - /openai/{endpoint:path} - Standard OpenAI passthrough route
    - /openai_passthrough/{endpoint:path} - Dedicated passthrough route (recommended for Responses API)

    Use /openai_passthrough/* when you need guaranteed passthrough to OpenAI without conflicts
    with LiteLLM's native implementations (e.g., for the Responses API at /v1/responses).

    Examples:
        Standard route:
        - /openai/v1/chat/completions
        - /openai/v1/assistants
        - /openai/v1/threads

        Dedicated passthrough (for Responses API):
        - /openai_passthrough/v1/responses
        - /openai_passthrough/v1/responses/{response_id}
        - /openai_passthrough/v1/responses/{response_id}/input_items

    [Docs](https://docs.litellm.ai/docs/pass_through/openai_passthrough)
    """
    base_target_url = _get_openai_passthrough_target_base(
        request=request,
        endpoint=endpoint,
    )
    preserve_client_auth = _should_preserve_openai_client_auth(
        request=request,
        endpoint=endpoint,
    )
    openai_api_key: Optional[str] = None
    forward_headers = False
    if preserve_client_auth:
        forward_headers = True
    else:
        openai_api_key = passthrough_endpoint_router.get_credentials(
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
            region_name=None,
        )
        if openai_api_key is None:
            raise Exception(
                "Required 'OPENAI_API_KEY' in environment to make pass-through calls to OpenAI."
            )

    return await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        base_target_url=base_target_url,
        api_key=openai_api_key,
        custom_llm_provider=litellm.LlmProviders.OPENAI,
        forward_headers=forward_headers,
    )


class BaseOpenAIPassThroughHandler:
    @staticmethod
    async def _base_openai_pass_through_handler(
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        base_target_url: str,
        api_key: Optional[str],
        custom_llm_provider: litellm.LlmProviders,
        extra_headers: Optional[dict] = None,
        forward_headers: bool = False,
    ):
        encoded_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
            endpoint=endpoint,
            base_target_url=base_target_url,
        )

        # Construct the full target URL by properly joining the base URL and endpoint path
        base_url = httpx.URL(base_target_url)
        updated_url = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url=base_url,
            path=encoded_endpoint,
            custom_llm_provider=custom_llm_provider,
        )

        if request.method == "POST":
            request_body = await get_request_body(request)
            prepared_request_body = request_body
            if _request_uses_codex_native_auth(request) and _is_openai_responses_endpoint(
                endpoint
            ):
                prepared_request_body = _add_route_family_logging_metadata(
                    prepared_request_body,
                    "codex_responses",
                )
                prepared_request_body = _add_codex_request_breakout_logging_metadata(
                    prepared_request_body
                )
            prepared_request_body = _prepare_request_body_for_passthrough_observability(
                request=request,
                request_body=prepared_request_body,
            )
            if prepared_request_body is not request_body:
                _safe_set_request_parsed_body(request, prepared_request_body)

        ## check for streaming
        is_streaming_request = False
        if "stream" in str(updated_url):
            is_streaming_request = True

        ## CREATE PASS-THROUGH
        endpoint_func = create_pass_through_route(
            endpoint=endpoint,
            target=str(updated_url),
            custom_headers=BaseOpenAIPassThroughHandler._assemble_headers(
                api_key=api_key, request=request, extra_headers=extra_headers
            ),
            _forward_headers=forward_headers,
            is_streaming_request=is_streaming_request,  # type: ignore
        )  # dynamically construct pass-through endpoint based on incoming path
        received_value = await endpoint_func(
            request,
            fastapi_response,
            user_api_key_dict,
        )

        return received_value

    @staticmethod
    def _append_openai_beta_header(headers: dict, request: Request) -> dict:
        """
        Appends the OpenAI-Beta header to the headers if the request is an OpenAI Assistants API request
        """
        if (
            RouteChecks._is_assistants_api_request(request) is True
            and "OpenAI-Beta" not in headers
        ):
            headers["OpenAI-Beta"] = "assistants=v2"
        return headers

    @staticmethod
    def _assemble_headers(
        api_key: Optional[str], request: Request, extra_headers: Optional[dict] = None
    ) -> dict:
        base_headers = {}
        if api_key is not None:
            base_headers = {
                "authorization": "Bearer {}".format(api_key),
                "api-key": "{}".format(api_key),
            }
        if extra_headers is not None:
            base_headers.update(extra_headers)
        return BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers=base_headers,
            request=request,
        )

    @staticmethod
    def _join_url_paths(
        base_url: httpx.URL, path: str, custom_llm_provider: litellm.LlmProviders
    ) -> str:
        """
        Properly joins a base URL with a path, preserving any existing path in the base URL.
        """
        # Join paths correctly by removing trailing/leading slashes as needed
        if not base_url.path or base_url.path == "/":
            # If base URL has no path, just use the new path
            joined_path_str = str(base_url.copy_with(path=path))
        else:
            # Otherwise, combine the paths
            base_path = base_url.path.rstrip("/")
            clean_path = path.lstrip("/")
            full_path = f"{base_path}/{clean_path}"
            joined_path_str = str(base_url.copy_with(path=full_path))

        # Apply OpenAI-specific path handling for both branches
        if (
            custom_llm_provider == litellm.LlmProviders.OPENAI
            and "/v1/" not in joined_path_str
        ):
            # Insert v1 after api.openai.com for OpenAI requests
            joined_path_str = joined_path_str.replace(
                "api.openai.com/", "api.openai.com/v1/"
            )

        return joined_path_str

    @staticmethod
    def _normalize_endpoint_for_target(
        endpoint: str, base_target_url: str
    ) -> str:
        normalized_endpoint = httpx.URL(endpoint).path
        if not normalized_endpoint.startswith("/"):
            normalized_endpoint = "/" + normalized_endpoint

        base_url = httpx.URL(base_target_url)
        if (
            base_url.host
            and "chatgpt.com" in base_url.host
            and base_url.path.rstrip("/") == "/backend-api/codex"
            and normalized_endpoint.startswith("/v1/")
        ):
            return normalized_endpoint[len("/v1") :]
        return normalized_endpoint


@router.api_route(
    "/cursor/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Cursor Pass-through", "pass-through"],
)
async def cursor_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Pass-through endpoint for the Cursor Cloud Agents API.

    Supports all Cursor Cloud Agents endpoints:
    - GET    /v0/agents         — List agents
    - POST   /v0/agents         — Launch an agent
    - GET    /v0/agents/{id}    — Agent status
    - GET    /v0/agents/{id}/conversation — Agent conversation
    - POST   /v0/agents/{id}/followup    — Add follow-up
    - POST   /v0/agents/{id}/stop        — Stop an agent
    - DELETE /v0/agents/{id}    — Delete an agent
    - GET    /v0/me             — API key info
    - GET    /v0/models         — List models
    - GET    /v0/repositories   — List GitHub repositories

    Uses Basic Authentication (base64-encoded `API_KEY:`).

    Credential lookup order:
    1. passthrough_endpoint_router (config.yaml deployments with use_in_pass_through)
    2. litellm.credential_list (credentials added via UI)
    3. CURSOR_API_KEY environment variable
    """
    import base64

    base_target_url = os.getenv("CURSOR_API_BASE") or "https://api.cursor.com"

    cursor_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="cursor",
        region_name=None,
    )

    if cursor_api_key is None:
        for credential in litellm.credential_list:
            if (
                credential.credential_info
                and credential.credential_info.get("custom_llm_provider") == "cursor"
            ):
                cursor_api_key = credential.credential_values.get("api_key")
                credential_api_base = credential.credential_values.get("api_base")
                if credential_api_base:
                    base_target_url = credential_api_base
                break

    if cursor_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Cursor API key not found. Add Cursor credentials via the UI (Models + Endpoints → LLM Credentials) or set CURSOR_API_KEY environment variable.",
        )

    encoded_endpoint = httpx.URL(endpoint).path

    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    auth_value = base64.b64encode(f"{cursor_api_key}:".encode("utf-8")).decode("ascii")

    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": f"Basic {auth_value}"},
        custom_llm_provider="cursor",
    )
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


async def vertex_ai_live_websocket_passthrough(
    websocket: WebSocket,
    model: Optional[str] = None,
    vertex_project: Optional[str] = None,
    vertex_location: Optional[str] = None,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
):
    """
    Vertex AI Live API WebSocket Pass-through Function

    This function provides WebSocket passthrough functionality for Vertex AI Live API,
    allowing real-time communication with Google's Live API service.

    Note: This function should be registered in proxy_server.py using:
    app.websocket("/vertex_ai/live")(vertex_ai_live_websocket_passthrough)
    """
    from litellm.proxy.proxy_server import proxy_logging_obj

    _ = user_api_key_dict  # passthrough route already authenticated; avoid lint warnings

    await websocket.accept()

    incoming_headers = dict(websocket.headers)
    vertex_credentials_config = passthrough_endpoint_router.get_vertex_credentials(
        project_id=vertex_project,
        location=vertex_location,
    )

    if vertex_credentials_config is None:
        # Attempt to load defaults from environment/config if not already initialised
        passthrough_endpoint_router.set_default_vertex_config()
        vertex_credentials_config = passthrough_endpoint_router.get_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
        )

    resolved_project = vertex_project
    resolved_location: Optional[str] = vertex_location
    credentials_value: Optional[str] = None

    if vertex_credentials_config is not None:
        resolved_project = resolved_project or vertex_credentials_config.vertex_project
        temp_location = resolved_location or vertex_credentials_config.vertex_location
        # Ensure resolved_location is a string
        if isinstance(temp_location, dict):
            resolved_location = str(temp_location)
        elif temp_location is not None:
            resolved_location = str(temp_location)
        else:
            resolved_location = None
        credentials_value = (
            str(vertex_credentials_config.vertex_credentials)
            if vertex_credentials_config.vertex_credentials is not None
            else None
        )

    try:
        resolved_location = resolved_location or (
            vertex_llm_base.get_default_vertex_location()
        )
        if model:
            resolved_location = vertex_llm_base.get_vertex_region(
                vertex_region=resolved_location,
                model=model,
            )

        (
            access_token,
            resolved_project,
        ) = await vertex_llm_base._ensure_access_token_async(
            credentials=credentials_value,
            project_id=resolved_project,
            custom_llm_provider="vertex_ai_beta",
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            "Failed to prepare Vertex AI credentials for live passthrough"
        )
        # Log the authentication failure using proxy_logging_obj
        if proxy_logging_obj and user_api_key_dict:
            await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict,
                original_exception=e,
                request_data={},
            )
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011, reason="Vertex AI authentication failed")
        return

    host_location = resolved_location or vertex_llm_base.get_default_vertex_location()
    host = (
        "aiplatform.googleapis.com"
        if host_location == "global"
        else f"{host_location}-aiplatform.googleapis.com"
    )
    service_url = (
        f"wss://{host}/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    )

    upstream_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    if resolved_project:
        upstream_headers["x-goog-user-project"] = resolved_project

    # Forward any custom x-goog-* headers provided by the caller if we haven't overridden them
    for header_name, header_value in incoming_headers.items():
        lower_header = header_name.lower()
        if lower_header.startswith("x-goog-") and header_name not in upstream_headers:
            upstream_headers[header_name] = header_value

    # Use the new WebSocket passthrough pattern
    if user_api_key_dict is None:
        raise ValueError("user_api_key_dict is required for WebSocket passthrough")

    return await websocket_passthrough_request(
        websocket=websocket,
        target=service_url,
        custom_headers=upstream_headers,
        user_api_key_dict=user_api_key_dict,
        forward_headers=False,
        endpoint="/vertex_ai/live",
        accept_websocket=False,
    )


def create_vertex_ai_live_websocket_endpoint():
    """
    Create a Vertex AI Live WebSocket endpoint using the new passthrough pattern.

    This demonstrates how to use the create_websocket_passthrough_route function
    for a provider-specific WebSocket endpoint.
    """
    # This would be used like:
    # endpoint_func = create_vertex_ai_live_websocket_endpoint()
    # app.websocket("/vertex_ai/live")(endpoint_func)

    # For now, we'll keep the existing implementation since it has
    # provider-specific logic for Vertex AI credentials and headers
    return vertex_ai_live_websocket_passthrough


def create_generic_websocket_passthrough_endpoint(
    provider: str,
    target_url: str,
    custom_headers: Optional[dict] = None,
    forward_headers: bool = False,
    cost_per_request: Optional[float] = None,
):
    """
    Create a generic WebSocket passthrough endpoint for any provider.

    This demonstrates the new WebSocket passthrough pattern that's similar to
    the HTTP create_pass_through_route function.

    Args:
        provider: The provider name (e.g., "anthropic", "cohere")
        target_url: The target WebSocket URL
        custom_headers: Custom headers to include
        forward_headers: Whether to forward incoming headers

    Returns:
        A WebSocket endpoint function that can be registered with app.websocket()

    Example usage:
        # Create a WebSocket endpoint for Anthropic
        anthropic_ws_func = create_generic_websocket_passthrough_endpoint(
            provider="anthropic",
            target_url="wss://api.anthropic.com/v1/ws",
            custom_headers={"x-api-key": "your-api-key"},
            forward_headers=True
        )

        # Register it in proxy_server.py
        app.websocket("/anthropic/ws")(anthropic_ws_func)
    """
    return create_websocket_passthrough_route(
        endpoint=f"/{provider}/ws",
        target=target_url,
        custom_headers=custom_headers,
        _forward_headers=forward_headers,
        cost_per_request=cost_per_request,
    )
