"""Wave 6D observability metadata extraction.

This module owns shared metadata primitives plus Claude request text,
child-agent, post-rewrite context-file, session/repository, tool-definition
snapshot, Claude/Gemini/Codex breakout, Anthropic billing header, and
persisted-output metadata handling. It intentionally does not import
``llm_passthrough_endpoints`` at module scope.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional, TypeVar
from urllib.parse import urlparse

MetadataMergeCallback = Callable[..., dict[str, Any]]
NormalizeTagValueCallback = Callable[[Any], Optional[str]]
RequestTenantIdGetter = Callable[[Any], Optional[str]]
ContextFilesLogger = Callable[[dict[str, Any], list[str]], None]
RequestHeadersGetter = Callable[[Any], dict[str, str]]
EnvGetter = Callable[[str], Optional[str]]

_WalkResultT = TypeVar("_WalkResultT")

_CLAUDE_AGENT_TENANT_PATTERN = re.compile(
    r"You are '(?P<agent>[^']+)' and you are working on the "
    r"'(?P<tenant>[^']+)' project\b"
)
_CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS: tuple[tuple[str, str], ...] = (
    ("MEMORY.md", "memory-md"),
    ("CLAUDE.md", "claude-md"),
)

_request_tenant_id_getter: Optional[RequestTenantIdGetter] = None
_context_files_logger: Optional[ContextFilesLogger] = None
_request_headers_getter: Optional[RequestHeadersGetter] = None
_env_getter: Optional[EnvGetter] = None


def configure_observability_metadata_runtime(
    *,
    get_explicit_tenant_id: Optional[RequestTenantIdGetter] = None,
    log_context_files: Optional[ContextFilesLogger] = None,
    get_request_headers: Optional[RequestHeadersGetter] = None,
    get_env: Optional[EnvGetter] = None,
) -> None:
    """Bind optional host request and logging callbacks.

    Passing no callbacks restores the behavior-preserving, side-effect-free
    defaults.
    """
    global _context_files_logger, _request_tenant_id_getter, _request_headers_getter, _env_getter
    _request_tenant_id_getter = get_explicit_tenant_id
    _context_files_logger = log_context_files
    _request_headers_getter = get_request_headers
    _env_getter = get_env


def _get_env(name: str) -> Optional[str]:
    getter = _env_getter
    if getter is not None:
        return getter(name)
    return os.getenv(name)


def _safe_get_headers(request: Any) -> dict[str, str]:
    getter = _request_headers_getter
    if getter is not None:
        return getter(request)
    headers = getattr(request, "headers", None)
    if isinstance(headers, dict):
        return headers
    return {}


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


def _normalize_low_cardinality_tag_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned or None
    return None


def _dedupe_sorted_str_list(values: list[str]) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


def _iter_anthropic_text_fragments(value: Any) -> Iterator[str]:
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
    request_body: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    for top_level_key in ("messages", "system"):
        for fragment in _iter_anthropic_text_fragments(
            request_body.get(top_level_key)
        ):
            match = _CLAUDE_AGENT_TENANT_PATTERN.search(fragment)
            if match is None:
                continue
            agent = match.group("agent").strip()
            tenant = match.group("tenant").strip()
            if agent and tenant:
                return agent, tenant
    return None, None


def _add_claude_child_agent_observability_metadata(
    request_body: dict[str, Any],
    *,
    explicit_tenant_id: Optional[str] = None,
    request: Any = None,
    get_explicit_tenant_id: Optional[RequestTenantIdGetter] = None,
    merge_metadata: Optional[MetadataMergeCallback] = None,
    normalize_tag_value: Optional[NormalizeTagValueCallback] = None,
) -> dict[str, Any]:
    agent, tenant = _extract_claude_agent_and_tenant_from_request_body(request_body)
    if not agent and not tenant:
        return request_body

    tenant_id_getter = get_explicit_tenant_id or _request_tenant_id_getter
    if explicit_tenant_id is None and request is not None and tenant_id_getter:
        explicit_tenant_id = tenant_id_getter(request)

    merge_metadata_callback = merge_metadata or _merge_litellm_metadata
    normalize_tag_value_callback = (
        normalize_tag_value or _normalize_low_cardinality_tag_value
    )

    extra_fields: dict[str, Any] = {}
    tags_to_add: list[str] = []
    litellm_metadata = request_body.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        litellm_metadata = {}

    if agent:
        extra_fields["agent_name"] = agent
        extra_fields["aawm_claude_agent_name"] = agent
        normalized_agent = normalize_tag_value_callback(agent) or "unknown"
        tags_to_add.append(f"claude-agent:{normalized_agent}")

        existing_trace_name = litellm_metadata.get("trace_name")
        child_trace_name = f"claude-code.{agent}"
        if existing_trace_name != child_trace_name:
            if existing_trace_name and not litellm_metadata.get("source_trace_name"):
                extra_fields["source_trace_name"] = existing_trace_name
            extra_fields["trace_name"] = child_trace_name

    if tenant:
        tenant_for_identity = explicit_tenant_id or tenant
        extra_fields["tenant_id"] = tenant_for_identity
        extra_fields["aawm_tenant_id"] = tenant_for_identity
        extra_fields["aawm_claude_project"] = tenant
        existing_trace_user_id = litellm_metadata.get("trace_user_id")
        if existing_trace_user_id != tenant_for_identity:
            if existing_trace_user_id and not litellm_metadata.get(
                "source_trace_user_id"
            ):
                extra_fields["source_trace_user_id"] = existing_trace_user_id
            extra_fields["trace_user_id"] = tenant_for_identity
        normalized_tenant = normalize_tag_value_callback(tenant) or "unknown"
        tags_to_add.append(f"claude-project:{normalized_tenant}")

    return merge_metadata_callback(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _detect_claude_post_rewrite_context_files(
    request_body: dict[str, Any],
) -> list[str]:
    present_files: list[str] = []
    seen_files: set[str] = set()

    for top_level_key in ("system", "messages"):
        for fragment in _iter_anthropic_text_fragments(
            request_body.get(top_level_key)
        ):
            for marker, _tag_suffix in _CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS:
                if marker in seen_files:
                    continue
                if marker in fragment:
                    present_files.append(marker)
                    seen_files.add(marker)

    return present_files


def _add_claude_post_rewrite_context_file_logging_metadata(
    request_body: dict[str, Any],
    *,
    detect_context_files: Optional[
        Callable[[dict[str, Any]], list[str]]
    ] = None,
    merge_metadata: Optional[MetadataMergeCallback] = None,
    log_context_files: Optional[ContextFilesLogger] = None,
) -> dict[str, Any]:
    detect_context_files_callback = (
        detect_context_files or _detect_claude_post_rewrite_context_files
    )
    present_files = detect_context_files_callback(request_body)
    if not present_files:
        return request_body

    context_files_logger = log_context_files or _context_files_logger
    if context_files_logger is not None:
        context_files_logger(request_body, present_files)

    tags_to_add = ["claude-post-rewrite-context-file-present"]
    for marker, tag_suffix in _CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS:
        if marker in present_files:
            tags_to_add.append(f"claude-post-rewrite-context-file:{tag_suffix}")

    merge_metadata_callback = merge_metadata or _merge_litellm_metadata
    return merge_metadata_callback(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_post_rewrite_context_files_present": present_files,
            "claude_post_rewrite_context_file_count": len(present_files),
        },
    )


# ---------------------------------------------------------------------------
# Session / repository extraction constants and helpers
# ---------------------------------------------------------------------------

_PASSTHROUGH_SESSION_ID_HEADER_NAMES: tuple[str, ...] = (
    "session_id",
    "Session_Id",
    "x-session-id",
    "X-Session-Id",
)
_PASSTHROUGH_REPOSITORY_HEADER_NAMES: tuple[str, ...] = (
    "x-aawm-repository",
    "x-litellm-repository",
    "x-repository",
    "x-git-repository",
)
_PASSTHROUGH_REPOSITORY_BODY_KEYS: frozenset[str] = frozenset(
    {
        "repository",
        "repo",
        "workspace_root",
        "workspaceRoot",
        "project_root",
        "projectRoot",
        "root_path",
        "rootPath",
        "working_directory",
        "workingDirectory",
        "cwd_path",
        "cwdPath",
        "cwd_uri",
        "cwdUri",
    }
)
_PASSTHROUGH_REPOSITORY_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<environment_context>[\s\S]{0,2000}<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>",
        re.IGNORECASE,
    ),
    re.compile(r"<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>"),
    re.compile(r"AGENTS\.md instructions for\s+[`'\"]?(?P<path>/[^\n<`'\"]+)"),
    re.compile(r"\bcwd\b\s*[:=]\s*[`'\"]?(?P<path>/[^`'\"\n<]+)"),
    re.compile(
        r"\*{0,2}Workspace Directories:\*{0,2}\s*\n\s*[-*]\s*[`'\"]?(?P<path>/[^\n`'\"]+)",
        re.IGNORECASE,
    ),
)
_PASSTHROUGH_REPOSITORY_PLACEHOLDER_VALUES: set[str] = {
    "...",
    "memories",
    "new",
    "path",
    "project",
    "remote",
    "repo",
    "repository",
    "unknown",
}
_PASSTHROUGH_REPOSITORY_AGENT_ROLE_VALUES: set[str] = {
    "agent",
    "analyst",
    "architect",
    "engineer",
    "infra",
    "ops",
    "orchestrator",
    "principal",
    "qa",
    "researcher",
    "reviewer",
    "salvage",
    "tester",
}
_PASSTHROUGH_REPOSITORY_AGENT_ID_RE: re.Pattern[str] = re.compile(
    r"^agent-[a-f0-9]{3,}$",
    re.IGNORECASE,
)
_PASSTHROUGH_REPOSITORY_WAVE_AGENT_RE: re.Pattern[str] = re.compile(
    r"^wave\d+-(?:analyst|engineer|infra|ops|principal|qa|researcher|reviewer|salvage|tester)$",
    re.IGNORECASE,
)
_PASSTHROUGH_REPOSITORY_TRANSCRIPT_ARTIFACT_RE: re.Pattern[str] = re.compile(
    r"^(?:rollout-\d{4}(?:-[A-Za-z0-9_.-]*)?|.*\.jsonl?)$",
    re.IGNORECASE,
)
_AAWM_REQUEST_BODY_WALK_MAX_DEPTH = 64
_AAWM_REQUEST_BODY_WALK_MAX_NODES = 4000


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
    request: Any,
    request_body: Optional[dict[str, Any]] = None,
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

    headers = _safe_get_headers(request)
    for header_name in _PASSTHROUGH_SESSION_ID_HEADER_NAMES:
        value = headers.get(header_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_passthrough_repository(value: str) -> Optional[str]:
    cleaned = value.strip().strip("`'\"")
    if not cleaned:
        return None
    if cleaned.startswith("git@") and ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    elif "://" in cleaned:
        parsed = urlparse(cleaned)
        path = parsed.path.strip("/")
        netloc = parsed.netloc.split("@", 1)[-1]
        if parsed.scheme == "file" and path:
            cleaned = path.rstrip("/").rsplit("/", 1)[-1]
        elif netloc.lower().endswith("github.com") and path:
            cleaned = path
        else:
            cleaned = f"{netloc}/{path}".strip("/")
    elif cleaned.startswith("/"):
        cleaned = cleaned.rstrip("/").rsplit("/", 1)[-1]
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.strip("/")
    if not cleaned:
        return None

    normalized = cleaned.lower()
    if normalized.endswith(" (memory)"):
        normalized = normalized[: -len(" (memory)")]
    if (
        normalized in _PASSTHROUGH_REPOSITORY_PLACEHOLDER_VALUES
        or _PASSTHROUGH_REPOSITORY_TRANSCRIPT_ARTIFACT_RE.fullmatch(normalized)
        or normalized in _PASSTHROUGH_REPOSITORY_AGENT_ROLE_VALUES
        or _PASSTHROUGH_REPOSITORY_AGENT_ID_RE.fullmatch(normalized)
        or _PASSTHROUGH_REPOSITORY_WAVE_AGENT_RE.fullmatch(normalized)
    ):
        return None

    return cleaned


def _extract_passthrough_repository_from_text(value: str) -> Optional[str]:
    for pattern in _PASSTHROUGH_REPOSITORY_TEXT_PATTERNS:
        matches = list(pattern.finditer(value))
        for match in reversed(matches):
            repository = _normalize_passthrough_repository(match.group("path"))
            if repository:
                return repository
    return None


def _walk_request_value_with_budget(
    value: object,
    *,
    visitor: Callable[[object, int], Optional[_WalkResultT]],
    max_depth: int = _AAWM_REQUEST_BODY_WALK_MAX_DEPTH,
    max_nodes: int = _AAWM_REQUEST_BODY_WALK_MAX_NODES,
    _depth: int = 0,
    _state: Optional[dict[str, int]] = None,
) -> Optional[_WalkResultT]:
    """Bounded recursive walk used by request-body sanitizers/extractors."""
    if _state is None:
        _state = {"nodes": 0}
    if _depth > max_depth:
        return None
    _state["nodes"] += 1
    if _state["nodes"] > max_nodes:
        return None
    result = visitor(value, _depth)
    if result is not None:
        return result
    if isinstance(value, dict):
        for child in value.values():
            found = _walk_request_value_with_budget(
                child,
                visitor=visitor,
                max_depth=max_depth,
                max_nodes=max_nodes,
                _depth=_depth + 1,
                _state=_state,
            )
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in reversed(value):
            found = _walk_request_value_with_budget(
                child,
                visitor=visitor,
                max_depth=max_depth,
                max_nodes=max_nodes,
                _depth=_depth + 1,
                _state=_state,
            )
            if found is not None:
                return found
    return None


def _extract_passthrough_repository_from_body_text(value: Any) -> Optional[str]:
    def _visitor(node: object, _depth: int) -> Optional[str]:
        if isinstance(node, str):
            return _extract_passthrough_repository_from_text(node)
        if isinstance(node, dict):
            for key, child in node.items():
                if key in _PASSTHROUGH_REPOSITORY_BODY_KEYS and isinstance(child, str):
                    repository = _normalize_passthrough_repository(child)
                    if repository:
                        return repository
        return None

    if isinstance(value, dict):
        for key, child in value.items():
            if key in _PASSTHROUGH_REPOSITORY_BODY_KEYS and isinstance(child, str):
                repository = _normalize_passthrough_repository(child)
                if repository:
                    return repository
    return _walk_request_value_with_budget(value, visitor=_visitor)


def _extract_passthrough_repository(
    request: Any,
    request_body: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    if isinstance(request_body, dict):
        for path in (
            ("repository",),
            ("repo",),
            ("workspace_root",),
            ("workspaceRoot",),
            ("project_root",),
            ("projectRoot",),
            ("root_path",),
            ("rootPath",),
            ("working_directory",),
            ("workingDirectory",),
            ("cwd_path",),
            ("cwdPath",),
            ("cwd_uri",),
            ("cwdUri",),
            ("metadata", "repository"),
            ("metadata", "repo"),
            ("metadata", "workspace_root"),
            ("metadata", "workspaceRoot"),
            ("litellm_metadata", "repository"),
            ("request", "repository"),
            ("request", "workspace_root"),
            ("request", "workspaceRoot"),
            ("request", "project_root"),
            ("request", "projectRoot"),
            ("request", "root_path"),
            ("request", "rootPath"),
            ("request", "working_directory"),
            ("request", "workingDirectory"),
            ("request", "cwd_path"),
            ("request", "cwdPath"),
            ("request", "cwd_uri"),
            ("request", "cwdUri"),
            ("request", "metadata", "repository"),
            ("request", "metadata", "workspace_root"),
            ("request", "metadata", "workspaceRoot"),
        ):
            value = _get_nested_str_value(request_body, path)
            if value:
                return _normalize_passthrough_repository(value)
        repository = _extract_passthrough_repository_from_body_text(request_body)
        if repository:
            return repository

    headers = _safe_get_headers(request)
    for header_name in _PASSTHROUGH_REPOSITORY_HEADER_NAMES:
        value = headers.get(header_name)
        if isinstance(value, str) and value.strip():
            return _normalize_passthrough_repository(value)
    return None


def _get_passthrough_trace_environment() -> Optional[str]:
    for env_var in (
        "LITELLM_LANGFUSE_TRACE_ENVIRONMENT",
        "LANGFUSE_TRACING_ENVIRONMENT",
    ):
        value = _get_env(env_var)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _add_passthrough_trace_context_metadata(
    request_body: dict[str, Any],
    *,
    session_id: Optional[str],
    trace_environment: Optional[str],
    repository: Optional[str] = None,
) -> dict[str, Any]:
    updated_body = dict(request_body)
    litellm_metadata = dict(updated_body.get("litellm_metadata") or {})
    changed = False

    if session_id and not litellm_metadata.get("session_id"):
        litellm_metadata["session_id"] = session_id
        changed = True

    if trace_environment:
        existing_trace_environment = litellm_metadata.get("trace_environment")
        if existing_trace_environment != trace_environment:
            if existing_trace_environment and not litellm_metadata.get(
                "source_trace_environment"
            ):
                litellm_metadata["source_trace_environment"] = (
                    existing_trace_environment
                )
            litellm_metadata["trace_environment"] = trace_environment
            changed = True

    if repository and not litellm_metadata.get("repository"):
        litellm_metadata["repository"] = repository
        changed = True

    if not changed:
        return request_body

    updated_body["litellm_metadata"] = litellm_metadata
    return updated_body


# ---------------------------------------------------------------------------
# Tool-definition snapshot metadata
# ---------------------------------------------------------------------------

_AAWM_TOOL_DEFINITION_CAPTURE_VERSION = "v1"
_AAWM_TOOL_DEFINITION_MAX_TOOLS = 64
_AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS = 128
_AAWM_TOOL_DEFINITION_MAX_STRING_CHARS = 4096
_AAWM_TOOL_DEFINITION_MAX_DEPTH = 20
_AAWM_TOOL_DEFINITION_REDACTED = "redacted-by-litellm"
_AAWM_TOOL_DEFINITION_SECRET_KEY_RE: re.Pattern[str] = re.compile(
    r"("
    r"authorization|api[_-]?key|bearer|credential|password|secret|^token$"
    r"|access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token"
    r")",
    re.IGNORECASE,
)
_AAWM_TOOL_DEFINITION_SECRET_VALUE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk|rk|ak)-[A-Za-z0-9._-]{8,}\b", re.IGNORECASE),
    re.compile(r"\b(?:xox[baprs]-)[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bya29\.[0-9A-Za-z\-_.]{20,}\b"),
    re.compile(
        r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"
    ),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([^\s'\"]{8,})"
    ),
)


def _truncate_tool_definition_string(value: str) -> tuple[str, bool]:
    if len(value) <= _AAWM_TOOL_DEFINITION_MAX_STRING_CHARS:
        return value, False
    return value[:_AAWM_TOOL_DEFINITION_MAX_STRING_CHARS], True


def _redact_tool_definition_string(value: str) -> str:
    redacted = value
    for pattern in _AAWM_TOOL_DEFINITION_SECRET_VALUE_RES:
        redacted = pattern.sub(_AAWM_TOOL_DEFINITION_REDACTED, redacted)
    return redacted


def _sanitize_tool_definition_value(
    value: Any,
    *,
    depth: int = 0,
    key_hint: Optional[str] = None,
) -> tuple[Any, bool]:
    if depth > _AAWM_TOOL_DEFINITION_MAX_DEPTH:
        return {"__truncated__": "max_depth"}, True
    if key_hint and _AAWM_TOOL_DEFINITION_SECRET_KEY_RE.search(key_hint):
        return _AAWM_TOOL_DEFINITION_REDACTED, False
    if isinstance(value, str):
        return _truncate_tool_definition_string(_redact_tool_definition_string(value))
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    if isinstance(value, list):
        truncated = len(value) > _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS
        sanitized_items: list[Any] = []
        for item in value[:_AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS]:
            sanitized_item, item_truncated = _sanitize_tool_definition_value(
                item,
                depth=depth + 1,
                key_hint=key_hint,
            )
            truncated = truncated or item_truncated
            sanitized_items.append(sanitized_item)
        if len(value) > _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS:
            sanitized_items.append(
                {"__truncated_items__": len(value) - _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS}
            )
        return sanitized_items, truncated
    if isinstance(value, dict):
        truncated = len(value) > _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS
        sanitized_dict: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS:
                break
            sanitized_item, item_truncated = _sanitize_tool_definition_value(
                item,
                depth=depth + 1,
                key_hint=str(key),
            )
            truncated = truncated or item_truncated
            sanitized_dict[str(key)] = sanitized_item
        if len(value) > _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS:
            sanitized_dict["__truncated_keys__"] = (
                len(value) - _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS
            )
        return sanitized_dict, truncated
    return str(value), False


def _tool_definition_name(tool: dict[str, Any]) -> Optional[str]:
    function_definition = tool.get("function")
    for candidate in (
        tool.get("name"),
        function_definition.get("name")
        if isinstance(function_definition, dict)
        else None,
        tool.get("tool_name"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _tool_definition_description(tool: dict[str, Any]) -> Optional[str]:
    function_definition = tool.get("function")
    for candidate in (
        tool.get("description"),
        function_definition.get("description")
        if isinstance(function_definition, dict)
        else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _tool_definition_parameters(tool: dict[str, Any]) -> Any:
    function_definition = tool.get("function")
    if isinstance(function_definition, dict) and "parameters" in function_definition:
        return function_definition.get("parameters")
    for key in ("parameters", "input_schema", "schema", "json_schema"):
        if key in tool:
            return tool.get(key)
    return None


def _build_tool_definition_snapshot_entry(
    *,
    source: str,
    index: int,
    tool: Any,
) -> tuple[Optional[dict[str, Any]], bool]:
    if not isinstance(tool, dict):
        return None, False

    sanitized_definition, definition_truncated = _sanitize_tool_definition_value(tool)
    sanitized_parameters, parameters_truncated = _sanitize_tool_definition_value(
        _tool_definition_parameters(tool)
    )
    description, description_truncated = _truncate_tool_definition_string(
        _redact_tool_definition_string(_tool_definition_description(tool) or "")
    )
    entry = {
        "source": source,
        "index": index,
        "type": tool.get("type"),
        "name": _tool_definition_name(tool),
        "description": description,
        "parameters": sanitized_parameters,
        "definition": sanitized_definition,
    }
    return entry, bool(
        definition_truncated or parameters_truncated or description_truncated
    )


def _tool_definition_snapshot_hash(snapshot: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_passthrough_tool_definition_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tool_sources: tuple[tuple[str, Any], ...] = (
        ("tools", request_body.get("tools")),
        ("functions", request_body.get("functions")),
    )
    snapshot: list[dict[str, Any]] = []
    available_count = 0
    truncated = False
    source_names: list[str] = []

    for source, tools in tool_sources:
        if not isinstance(tools, list):
            continue
        source_names.append(source)
        available_count += len(tools)
        for index, tool in enumerate(tools):
            if len(snapshot) >= _AAWM_TOOL_DEFINITION_MAX_TOOLS:
                truncated = True
                break
            entry, entry_truncated = _build_tool_definition_snapshot_entry(
                source=source,
                index=index,
                tool=tool,
            )
            if entry is None:
                continue
            snapshot.append(entry)
            truncated = truncated or entry_truncated

    if not snapshot:
        return {}

    names = [
        entry["name"]
        for entry in snapshot
        if isinstance(entry.get("name"), str) and entry.get("name")
    ]
    tool_types = [
        entry["type"]
        for entry in snapshot
        if isinstance(entry.get("type"), str) and entry.get("type")
    ]
    include_full_snapshot = (
        (_get_env("AAWM_TOOL_DEFINITION_INCLUDE_FULL_SNAPSHOT") or "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    metadata: dict[str, Any] = {
        "aawm_tool_definition_capture_version": _AAWM_TOOL_DEFINITION_CAPTURE_VERSION,
        "aawm_tool_definition_capture_source": "passthrough_request_body",
        "aawm_tool_definition_count": available_count,
        "aawm_tool_definition_captured_count": len(snapshot),
        "aawm_tool_definition_sources": source_names,
        "aawm_tool_definition_names": names,
        "aawm_tool_definition_types": tool_types,
        "aawm_tool_definition_snapshot_hash": _tool_definition_snapshot_hash(snapshot),
        "aawm_tool_definition_snapshot_truncated": truncated
        or available_count > len(snapshot),
        "aawm_tool_definition_snapshot_storage": "session_history_tool_definition_snapshots",
        "aawm_tool_definition_snapshot_storage_key": "session_id,aawm_tool_definition_snapshot_hash",
    }
    if include_full_snapshot:
        metadata["aawm_tool_definition_snapshot"] = snapshot
    return metadata


def _add_passthrough_tool_definition_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tool_definition_metadata = _build_passthrough_tool_definition_metadata(request_body)
    if not tool_definition_metadata:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        extra_fields=tool_definition_metadata,
    )


# ---------------------------------------------------------------------------
# _prepare_request_body_for_passthrough_observability
# ---------------------------------------------------------------------------


def _prepare_request_body_for_passthrough_observability(
    request: Any,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    session_id = _extract_passthrough_session_id(
        request=request, request_body=request_body
    )
    repository = _extract_passthrough_repository(
        request=request, request_body=request_body
    )
    trace_environment = _get_passthrough_trace_environment()
    prepared_body = _add_passthrough_trace_context_metadata(
        request_body,
        session_id=session_id,
        trace_environment=trace_environment,
        repository=repository,
    )
    return _add_passthrough_tool_definition_metadata(prepared_body)


# ---------------------------------------------------------------------------
# Claude / Gemini / Codex breakout extraction and logging
# ---------------------------------------------------------------------------


def _extract_openai_passthrough_tool_choice(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return _normalize_low_cardinality_tag_value(value)
    if isinstance(value, dict):
        for key in ("type", "name"):
            normalized = _normalize_low_cardinality_tag_value(value.get(key))
            if normalized:
                return normalized
    return None


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
                [
                    f"claude-thinking-type:{thinking_type}",
                    f"thinking-type:{thinking_type}",
                ]
            )
            extra_fields["claude_thinking_type"] = thinking_type

    output_config = request_body.get("output_config")
    if isinstance(output_config, dict):
        effort = _normalize_low_cardinality_tag_value(output_config.get("effort"))
        if effort:
            tags_to_add.extend([f"claude-effort:{effort}", f"effort:{effort}"])
            extra_fields["claude_effort"] = effort

    context_management = request_body.get("context_management")
    context_edits: list[dict[str, Any]] = []
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

    account_uuid = _get_nested_str_value(
        request_body, ("metadata", "user_id", "account_uuid")
    )
    if account_uuid:
        extra_fields["claude_account_uuid"] = account_uuid
    device_id = _get_nested_str_value(
        request_body, ("metadata", "user_id", "device_id")
    )
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

    tool_choice = _extract_openai_passthrough_tool_choice(
        request_body.get("tool_choice")
    )
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


# ---------------------------------------------------------------------------
# Anthropic billing header parse / extract / logging
# ---------------------------------------------------------------------------

_ANTHROPIC_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


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
    request_body: dict[str, Any],
) -> dict[str, str]:
    return _extract_anthropic_billing_header_fields(request_body.get("system"))


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


# ---------------------------------------------------------------------------
# Claude persisted-output logging metadata
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Route-family logging metadata
# ---------------------------------------------------------------------------


def _add_route_family_logging_metadata(
    request_body: dict[str, Any],
    route_family: str,
) -> dict[str, Any]:
    normalized_route_family = _normalize_low_cardinality_tag_value(route_family)
    if not normalized_route_family:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=[f"route:{normalized_route_family}"],
        extra_fields={"passthrough_route_family": normalized_route_family},
    )


# ---------------------------------------------------------------------------
# Owned-symbol inventory
# ---------------------------------------------------------------------------

OWNED_SYMBOLS: tuple[str, ...] = (
    # Callback types
    "MetadataMergeCallback",
    "NormalizeTagValueCallback",
    "RequestTenantIdGetter",
    "ContextFilesLogger",
    "RequestHeadersGetter",
    "EnvGetter",
    # Configuration
    "configure_observability_metadata_runtime",
    # Shared primitives
    "_merge_litellm_metadata",
    "_format_langfuse_span_timestamp",
    "_build_langfuse_span_descriptor",
    "_normalize_low_cardinality_tag_value",
    "_dedupe_sorted_str_list",
    "_iter_anthropic_text_fragments",
    # Claude child-agent
    "_extract_claude_agent_and_tenant_from_request_body",
    "_add_claude_child_agent_observability_metadata",
    # Post-rewrite context files
    "_detect_claude_post_rewrite_context_files",
    "_add_claude_post_rewrite_context_file_logging_metadata",
    # Session / repository
    "_get_nested_str_value",
    "_extract_passthrough_session_id",
    "_normalize_passthrough_repository",
    "_extract_passthrough_repository_from_text",
    "_walk_request_value_with_budget",
    "_extract_passthrough_repository_from_body_text",
    "_extract_passthrough_repository",
    "_get_passthrough_trace_environment",
    "_add_passthrough_trace_context_metadata",
    # Tool-definition snapshot
    "_truncate_tool_definition_string",
    "_redact_tool_definition_string",
    "_sanitize_tool_definition_value",
    "_tool_definition_name",
    "_tool_definition_description",
    "_tool_definition_parameters",
    "_build_tool_definition_snapshot_entry",
    "_tool_definition_snapshot_hash",
    "_build_passthrough_tool_definition_metadata",
    "_add_passthrough_tool_definition_metadata",
    # Prepare request body
    "_prepare_request_body_for_passthrough_observability",
    # Breakout extraction / logging
    "_extract_openai_passthrough_tool_choice",
    "_extract_claude_request_breakout_fields",
    "_add_claude_request_breakout_logging_metadata",
    "_extract_gemini_request_breakout_fields",
    "_add_gemini_request_breakout_logging_metadata",
    "_extract_codex_request_breakout_fields",
    "_add_codex_request_breakout_logging_metadata",
    # Anthropic billing header
    "_parse_anthropic_billing_header_text",
    "_extract_anthropic_billing_header_fields",
    "_extract_anthropic_billing_header_fields_from_request_body",
    "_add_anthropic_billing_header_logging_metadata",
    # Claude persisted-output
    "_add_claude_persisted_output_logging_metadata",
    # Route family
    "_add_route_family_logging_metadata",
    # Constants
    "_ANTHROPIC_BILLING_HEADER_PREFIX",
    "_AAWM_TOOL_DEFINITION_CAPTURE_VERSION",
    "_AAWM_TOOL_DEFINITION_MAX_TOOLS",
    "_PASSTHROUGH_SESSION_ID_HEADER_NAMES",
    "_PASSTHROUGH_REPOSITORY_HEADER_NAMES",
    "_PASSTHROUGH_REPOSITORY_BODY_KEYS",
    "_PASSTHROUGH_REPOSITORY_TEXT_PATTERNS",
    "_PASSTHROUGH_REPOSITORY_PLACEHOLDER_VALUES",
    "_PASSTHROUGH_REPOSITORY_AGENT_ROLE_VALUES",
    "OWNED_SYMBOLS",
)
