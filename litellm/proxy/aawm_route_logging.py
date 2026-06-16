import logging
import re
from datetime import datetime
from typing import Any, Callable, Optional, Union
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import httpx
from fastapi import Request

from litellm._logging import (
    register_aawm_route_access_log_replacement,
    verbose_aawm_route_logger,
)

_AAWM_ROUTE_ACCESS_LOG_SCOPE_KEY = "aawm_route_access_log_emitted"
_AAWM_ROUTE_ACCESS_LOGGER_NAME = verbose_aawm_route_logger.name
_AAWM_ROUTE_ACCESS_LOG_TYPE = "ROUTE"
_AAWM_ROUTE_LOG_EMBED_TYPE = "EMBED"
_AAWM_ROUTE_LOG_RERANK_TYPE = "RERANK"
_AAWM_ROUTE_LOG_DEFAULT_AGENT = "orchestrator"
_AAWM_ROUTE_LOG_MAX_FIELD_CHARS = 180
_AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS = 96
_AAWM_ROUTE_LOG_SAFE_QUERY_KEYS = frozenset(
    {
        "alt",
        "api-version",
        "beta",
        "stream",
    }
)
_AAWM_ROUTE_LOG_AGENT_METADATA_KEYS = (
    "agent_name",
    "aawm_agent_name",
    "aawm_claude_agent_name",
)
_AAWM_ROUTE_LOG_AGENT_HEADER_KEYS = (
    "x-aawm-agent-name",
    "x-litellm-agent-name",
    "x-agent-name",
)
_AAWM_ROUTE_LOG_AGENT_ID_METADATA_KEYS = (
    "agent_id",
    "aawm_agent_id",
    "source_agent_id",
    "subagent_id",
    "task_id",
)
_AAWM_ROUTE_LOG_AGENT_ID_HEADER_KEYS = (
    "x-aawm-agent-id",
    "x-grok-agent-id",
    "x-litellm-agent-id",
    "x-agent-id",
)
_AAWM_ROUTE_LOG_REPOSITORY_METADATA_KEYS = (
    "repository",
    "aawm_repository",
    "source_repository",
    "repo",
    "repo_name",
    "repository_name",
    "git_repository",
    "vcs_repository",
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
    "aawm_claude_project",
)
_AAWM_ROUTE_LOG_REPOSITORY_HEADER_KEYS = (
    "x-aawm-repository",
    "x-litellm-repository",
    "x-repository",
    "x-git-repository",
)
_AAWM_ROUTE_LOG_CLIENT_LABEL_METADATA_KEYS = (
    "client_name_version",
    "client_label",
    "aawm_client_label",
    "client_user_agent",
    "user_agent",
)
_AAWM_ROUTE_LOG_CLIENT_NAME_METADATA_KEYS = (
    "client_name",
    "aawm_client_name",
    "application_name",
    "app_name",
)
_AAWM_ROUTE_LOG_CLIENT_VERSION_METADATA_KEYS = (
    "client_version",
    "aawm_client_version",
    "application_version",
    "app_version",
)
_AAWM_ROUTE_LOG_CLIENT_LABEL_HEADER_KEYS = (
    "x-aawm-client",
    "x-litellm-client",
    "x-client",
    "x-client-name-version",
    "user-agent",
)
_AAWM_ROUTE_LOG_CLIENT_NAME_HEADER_KEYS = (
    "x-aawm-client-name",
    "x-litellm-client-name",
    "x-client-name",
)
_AAWM_ROUTE_LOG_CLIENT_VERSION_HEADER_KEYS = (
    "x-aawm-client-version",
    "x-litellm-client-version",
    "x-client-version",
)
_AAWM_ROUTE_LOG_MODEL_ALIAS_METADATA_KEYS = (
    "inbound_model_alias",
    "requested_model_alias",
    "model_alias_label",
    "anthropic_auto_agent_alias",
    "codex_auto_agent_alias",
    "aawm_auto_agent_alias",
)
_AAWM_ROUTE_LOG_SELECTED_MODEL_METADATA_KEYS = (
    "codex_auto_agent_selected_model",
    "anthropic_auto_agent_selected_model",
    "anthropic_adapter_model",
    "xai_oauth_public_model",
    "xai_oauth_upstream_model",
    "grok_model_override",
)
_AAWM_ROUTE_LOG_METADATA_NESTED_KEYS = (
    "metadata",
    "litellm_metadata",
    "request_metadata",
    "user_api_key_metadata",
)
_AAWM_ROUTE_LOG_REPOSITORY_BODY_PATHS = (
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
    ("litellm_metadata", "repo"),
    ("litellm_metadata", "workspace_root"),
    ("litellm_metadata", "workspaceRoot"),
    ("request", "repository"),
    ("request", "repo"),
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
    ("request", "metadata", "repo"),
    ("request", "metadata", "workspace_root"),
    ("request", "metadata", "workspaceRoot"),
    ("request", "litellm_metadata", "repository"),
    ("request", "litellm_metadata", "repo"),
)
_AAWM_ROUTE_LOG_REPOSITORY_BODY_KEYS = frozenset(
    path[-1] for path in _AAWM_ROUTE_LOG_REPOSITORY_BODY_PATHS
)
_AAWM_ROUTE_LOG_REPOSITORY_TEXT_PATTERNS = (
    re.compile(
        r"<environment_context>[\s\S]{0,2000}<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>",
        re.IGNORECASE,
    ),
    re.compile(r"<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>", re.IGNORECASE),
    re.compile(
        r"AGENTS\.md instructions for\s+[`'\"]?(?P<path>/[^\n<`'\"]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\bcwd\b\s*[:=]\s*[`'\"]?(?P<path>/[^`'\"\n<]+)", re.IGNORECASE),
    re.compile(
        r"\*{0,2}Workspace Directories:\*{0,2}\s*\n\s*[-*]\s*[`'\"]?(?P<path>/[^\n`'\"]+)",
        re.IGNORECASE,
    ),
)
_AAWM_ROUTE_LOG_REPOSITORY_PLACEHOLDER_VALUES = {
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
_AAWM_ROUTE_LOG_REPOSITORY_AGENT_ROLE_VALUES = {
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
_AAWM_ROUTE_LOG_REPOSITORY_AGENT_ID_RE = re.compile(
    r"^agent-[a-f0-9]{3,}$",
    re.IGNORECASE,
)
_AAWM_ROUTE_LOG_REPOSITORY_WAVE_AGENT_RE = re.compile(
    r"^w\d+(?:[-_ ].*)?$",
    re.IGNORECASE,
)
_AAWM_ROUTE_LOG_TENANT_REPOSITORY_FRAGMENTS = (
    "litellm",
    "pytest-testable",
    "dashboard-shell",
    "aawm-tap",
    "aawm-infrastructure",
    "aawm-devtools",
    "aawm-observe",
    "mcp-pg",
)
_AAWM_ROUTE_LOG_NORMALIZED_TUI_CLIENTS = frozenset(
    {"Claude", "Codex", "Grok", "OpenCode"}
)


def _clean_aawm_route_log_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)) or not isinstance(
        value, (str, int, float)
    ):
        return None

    cleaned = "".join(
        char if char.isprintable() and char not in "\r\n\t" else " "
        for char in str(value).strip()
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None

    lower_cleaned = cleaned.lower()
    if lower_cleaned.startswith(("bearer ", "sk-", "pk-", "xai-", "ya29.")):
        return None

    if len(cleaned) > _AAWM_ROUTE_LOG_MAX_FIELD_CHARS:
        cleaned = cleaned[: _AAWM_ROUTE_LOG_MAX_FIELD_CHARS - 3] + "..."
    return cleaned


def _is_aawm_route_log_slug(value: str) -> bool:
    return bool(value) and any(char.isalnum() for char in value) and all(
        char.isalnum() or char in "._-" for char in value
    )


def _normalize_aawm_route_log_agent_label(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None

    if "://" in cleaned or any(char in cleaned for char in "@;,=:{}`[]<>|\\"):
        return None
    if cleaned.endswith(".") or len(cleaned.split()) > 6:
        return None
    if not any(char.isalnum() for char in cleaned):
        return None
    if not all(char.isalnum() or char in " ._/-" for char in cleaned):
        return None
    return cleaned


def _normalize_aawm_route_log_agent_id_label(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None
    if "://" in cleaned or any(char in cleaned for char in " @;,=:{}`[]<>|\\"):
        return None
    if not any(char.isalnum() for char in cleaned):
        return None
    if not all(char.isalnum() or char in "._-" for char in cleaned):
        return None
    return cleaned


def _normalize_aawm_route_log_repository_label(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None
    if any(char in cleaned for char in " @;,=:{}`[]<>|\\"):
        return None

    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.path:
        cleaned = parsed.path

    if "/" in cleaned:
        path_parts = [part for part in cleaned.rstrip("/").split("/") if part]
        if not path_parts:
            return None
        if not cleaned.startswith("/") and len(path_parts) == 2:
            owner, repo = path_parts
            if _is_aawm_route_log_slug(owner) and _is_aawm_route_log_slug(repo):
                return f"{owner}/{repo}"
        path_parts = _trim_aawm_route_log_worktree_path_parts(path_parts)
        cleaned = path_parts[-1]

    normalized = cleaned.lower()
    if (
        normalized in _AAWM_ROUTE_LOG_REPOSITORY_PLACEHOLDER_VALUES
        or normalized in _AAWM_ROUTE_LOG_REPOSITORY_AGENT_ROLE_VALUES
        or _AAWM_ROUTE_LOG_REPOSITORY_AGENT_ID_RE.fullmatch(normalized)
        or _AAWM_ROUTE_LOG_REPOSITORY_WAVE_AGENT_RE.fullmatch(normalized)
    ):
        return None

    if _is_aawm_route_log_slug(cleaned):
        return cleaned
    return None


def _normalize_aawm_route_log_tenant_repository_label(value: Any) -> Optional[str]:
    repository = _normalize_aawm_route_log_repository_label(value)
    if not repository:
        return None
    normalized = repository.lower()
    if "litellm" in normalized:
        return "litellm"
    if normalized.endswith("-dev") or "tenant" in normalized:
        return None
    if any(
        fragment in normalized
        for fragment in _AAWM_ROUTE_LOG_TENANT_REPOSITORY_FRAGMENTS
    ):
        return repository
    return repository


def _trim_aawm_route_log_worktree_path_parts(path_parts: list[str]) -> list[str]:
    for marker in (".claude", ".codex", ".agents"):
        if marker in path_parts:
            marker_index = path_parts.index(marker)
            if marker_index > 0:
                return path_parts[:marker_index]
    for marker in ("worktrees", "worktree"):
        if marker in path_parts:
            marker_index = path_parts.index(marker)
            if marker_index > 0:
                return path_parts[:marker_index]
    return path_parts


def _normalize_aawm_route_log_client_product(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None

    product = cleaned.split()[0].strip("()")
    if not product or any(char in product for char in " @;,=:{}`[]<>|\\"):
        return None
    if "/" in product:
        name, version = product.split("/", 1)
        if _is_aawm_route_log_slug(name) and _is_aawm_route_log_slug(version):
            return f"{name}/{version}"
        return None
    if _is_aawm_route_log_slug(product):
        return product
    return None


def _normalize_aawm_route_log_client_label(value: Any) -> Optional[str]:
    product = _normalize_aawm_route_log_client_product(value)
    if not product:
        return None

    name, separator, version = product.partition("/")
    normalized_name = name.lower()
    if normalized_name in {"claude", "claude-code", "claude-cli"}:
        name = "Claude"
    elif normalized_name in {"codex", "codex-cli", "codex-tui"}:
        name = "Codex"
    elif normalized_name in {"grok", "grok-cli", "grok-build", "grok-tui"}:
        name = "Grok"
    elif normalized_name in {"opencode", "opencode-cli", "opencode-tui"}:
        name = "OpenCode"
    else:
        name = name

    if separator and version:
        return f"{name}/{version}"
    return name


def _is_aawm_route_log_tui_client_label(value: Optional[str]) -> bool:
    if not value:
        return False
    name = value.split("/", 1)[0]
    return name in _AAWM_ROUTE_LOG_NORMALIZED_TUI_CLIENTS


def _first_aawm_route_log_value(
    *sources: Optional[dict[str, Any]],
    keys: tuple[str, ...],
    normalizer: Callable[[Any], Optional[str]] = _clean_aawm_route_log_field,
) -> Optional[str]:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = normalizer(source.get(key))
            if value:
                return value
    return None


def _iter_aawm_route_log_metadata_sources(
    request_body: Optional[dict[str, Any]],
    kwargs: Optional[dict],
    metadata: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if isinstance(metadata, dict):
        sources.append(metadata)
    if isinstance(request_body, dict):
        sources.append(request_body)
        for metadata_key in _AAWM_ROUTE_LOG_METADATA_NESTED_KEYS:
            nested = request_body.get(metadata_key)
            if isinstance(nested, dict):
                sources.append(nested)
    if isinstance(kwargs, dict):
        for candidate in (
            kwargs.get("metadata"),
            kwargs.get("standard_logging_object"),
            kwargs.get("passthrough_logging_payload"),
        ):
            if isinstance(candidate, dict):
                sources.append(candidate)
                nested_metadata = candidate.get("metadata")
                if isinstance(nested_metadata, dict):
                    sources.append(nested_metadata)

        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            sources.append(litellm_params)
            for metadata_key in _AAWM_ROUTE_LOG_METADATA_NESTED_KEYS:
                nested = litellm_params.get(metadata_key)
                if isinstance(nested, dict):
                    sources.append(nested)
            proxy_request = litellm_params.get("proxy_server_request")
            if isinstance(proxy_request, dict):
                proxy_body = proxy_request.get("body")
                if isinstance(proxy_body, dict):
                    sources.append(proxy_body)
                    for metadata_key in _AAWM_ROUTE_LOG_METADATA_NESTED_KEYS:
                        nested = proxy_body.get(metadata_key)
                        if isinstance(nested, dict):
                            sources.append(nested)

        passthrough_logging_payload = kwargs.get("passthrough_logging_payload")
        payload_body = getattr(passthrough_logging_payload, "request_body", None)
        if isinstance(payload_body, dict):
            sources.append(payload_body)
        elif isinstance(passthrough_logging_payload, dict):
            payload_body = passthrough_logging_payload.get("request_body")
            if isinstance(payload_body, dict):
                sources.append(payload_body)
        if isinstance(payload_body, dict):
            for metadata_key in _AAWM_ROUTE_LOG_METADATA_NESTED_KEYS:
                nested = payload_body.get(metadata_key)
                if isinstance(nested, dict):
                    sources.append(nested)

    deduped_sources: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for source in sources:
        source_id = id(source)
        if source_id in seen_ids:
            continue
        seen_ids.add(source_id)
        deduped_sources.append(source)
    return deduped_sources


def _get_nested_aawm_route_log_value(
    source: dict[str, Any],
    path: tuple[str, ...],
) -> Optional[Any]:
    current: Any = source
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_aawm_route_log_repository_from_text(value: str) -> Optional[str]:
    for pattern in _AAWM_ROUTE_LOG_REPOSITORY_TEXT_PATTERNS:
        matches = list(pattern.finditer(value))
        for match in reversed(matches):
            repository = _normalize_aawm_route_log_repository_label(
                match.group("path")
            )
            if repository:
                return repository
    return None


def _extract_aawm_route_log_repository_from_body_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return _extract_aawm_route_log_repository_from_text(value)
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _AAWM_ROUTE_LOG_REPOSITORY_BODY_KEYS and isinstance(child, str):
                repository = _normalize_aawm_route_log_repository_label(child)
                if repository:
                    return repository
            repository = _extract_aawm_route_log_repository_from_body_text(child)
            if repository:
                return repository
    if isinstance(value, list):
        for child in reversed(value):
            repository = _extract_aawm_route_log_repository_from_body_text(child)
            if repository:
                return repository
    return None


def _extract_aawm_route_log_repository_from_body(
    request_body: Optional[dict[str, Any]],
    kwargs: Optional[dict],
) -> Optional[str]:
    body_candidates: list[dict[str, Any]] = []
    if isinstance(request_body, dict):
        body_candidates.append(request_body)
    if isinstance(kwargs, dict):
        passthrough_logging_payload = kwargs.get("passthrough_logging_payload")
        payload_body = getattr(passthrough_logging_payload, "request_body", None)
        if isinstance(payload_body, dict):
            body_candidates.append(payload_body)
        elif isinstance(passthrough_logging_payload, dict):
            payload_body = passthrough_logging_payload.get("request_body")
            if isinstance(payload_body, dict):
                body_candidates.append(payload_body)

    for body in body_candidates:
        for path in _AAWM_ROUTE_LOG_REPOSITORY_BODY_PATHS:
            value = _get_nested_aawm_route_log_value(body, path)
            if isinstance(value, str):
                repository = _normalize_aawm_route_log_repository_label(value)
                if repository:
                    return repository
        repository = _extract_aawm_route_log_repository_from_body_text(body)
        if repository:
            return repository
    return None


def _extract_aawm_route_log_metadata(
    request_body: Optional[dict[str, Any]],
    kwargs: Optional[dict],
) -> dict[str, Any]:
    body_metadata: dict[str, Any] = {}
    if isinstance(request_body, dict):
        for metadata_key in ("litellm_metadata", "metadata"):
            metadata_value = request_body.get(metadata_key)
            if isinstance(metadata_value, dict):
                body_metadata.update(metadata_value)

    kwargs_metadata: dict[str, Any] = {}
    if isinstance(kwargs, dict):
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            metadata = litellm_params.get("metadata")
            if isinstance(metadata, dict):
                kwargs_metadata = metadata
    return {**body_metadata, **kwargs_metadata}


def _extract_aawm_route_log_headers(
    request: Request,
    kwargs: Optional[dict],
) -> dict[str, Any]:
    headers: dict[str, Any] = dict(getattr(request, "headers", {}) or {})
    if not isinstance(kwargs, dict):
        return headers

    header_sources = []
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        proxy_request = litellm_params.get("proxy_server_request")
        if isinstance(proxy_request, dict):
            header_sources.append(proxy_request.get("headers"))

    passthrough_logging_payload = kwargs.get("passthrough_logging_payload")
    payload_headers = getattr(passthrough_logging_payload, "request_headers", None)
    header_sources.append(payload_headers)
    if isinstance(passthrough_logging_payload, dict):
        header_sources.append(passthrough_logging_payload.get("request_headers"))

    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        header_sources.append(standard_logging_object.get("request_headers"))
        header_sources.append(standard_logging_object.get("headers"))

    for source in header_sources:
        if not source:
            continue
        if not isinstance(source, dict):
            try:
                source = dict(source)
            except (TypeError, ValueError):
                continue
        headers.update(source)
    return headers


def _get_case_insensitive_header_value(
    headers: Optional[dict[str, Any]],
    keys: tuple[str, ...],
    normalizer: Callable[[Any], Optional[str]] = _clean_aawm_route_log_field,
) -> Optional[str]:
    if not isinstance(headers, dict):
        return None
    normalized_headers = {str(key).lower(): value for key, value in headers.items()}
    for key in keys:
        value = normalizer(normalized_headers.get(key.lower()))
        if value:
            return value
    return None


def _get_aawm_route_log_client_product_label(
    metadata: dict[str, Any],
    headers: dict[str, Any],
) -> Optional[str]:
    direct_label = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_CLIENT_LABEL_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_CLIENT_LABEL_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    )
    if direct_label:
        return direct_label

    client_name = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_CLIENT_NAME_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_CLIENT_NAME_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    )
    if not client_name:
        return None

    client_version = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_CLIENT_VERSION_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_CLIENT_VERSION_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_client_product,
    )
    if client_version:
        return _normalize_aawm_route_log_client_label(f"{client_name}/{client_version}")
    return client_name


def _get_aawm_route_log_agent_id(
    *,
    metadata_sources: list[dict[str, Any]],
    headers: dict[str, Any],
) -> Optional[str]:
    return _first_aawm_route_log_value(
        *metadata_sources,
        keys=_AAWM_ROUTE_LOG_AGENT_ID_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_agent_id_label,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_AGENT_ID_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_agent_id_label,
    )


def _get_aawm_route_log_repository(
    *,
    metadata_sources: list[dict[str, Any]],
    headers: dict[str, Any],
    request_body: Optional[dict[str, Any]],
    kwargs: Optional[dict],
) -> Optional[str]:
    repository = _first_aawm_route_log_value(
        *metadata_sources,
        keys=_AAWM_ROUTE_LOG_REPOSITORY_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_repository_label,
    )
    if repository:
        return repository

    repository = _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_REPOSITORY_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_repository_label,
    )
    if repository:
        return repository

    repository = _extract_aawm_route_log_repository_from_body(request_body, kwargs)
    if repository:
        return repository

    tenant_source = _first_aawm_route_log_value(
        *metadata_sources,
        keys=("tenant_id_source",),
    )
    if tenant_source == "request_headers":
        for key in (
            "tenant_id",
            "aawm_tenant_id",
            "user_api_key_org_id",
            "organization_id",
            "org_id",
            "litellm_organization_id",
            "litellm_org_id",
            "user_api_key_team_id",
            "team_id",
            "litellm_team_id",
        ):
            repository = _first_aawm_route_log_value(
                *metadata_sources,
                keys=(key,),
                normalizer=_normalize_aawm_route_log_tenant_repository_label,
            )
            if repository:
                return repository

    return None


def _get_aawm_route_log_type(
    *,
    request: Request,
    request_body: Optional[dict[str, Any]],
    kwargs: Optional[dict],
) -> str:
    route_type = None
    if isinstance(kwargs, dict):
        route_type = _clean_aawm_route_log_field(kwargs.get("aawm_route_type"))
    route_type = (route_type or "").lower()

    incoming_endpoint = _safe_aawm_route_endpoint_label(request).lower()
    model = ""
    if isinstance(request_body, dict):
        model = str(request_body.get("model") or "").lower()

    if route_type in {"aembedding", "embedding", "embeddings"} or "embedding" in incoming_endpoint:
        return _AAWM_ROUTE_LOG_EMBED_TYPE
    if (
        route_type in {"arerank", "rerank", "ranking"}
        or "rerank" in incoming_endpoint
        or "ranking" in incoming_endpoint
        or "rerank" in model
    ):
        return _AAWM_ROUTE_LOG_RERANK_TYPE
    return _AAWM_ROUTE_ACCESS_LOG_TYPE


def _build_aawm_route_log_identity_label(
    *,
    log_type: str,
    agent_name: Optional[str],
    agent_id: Optional[str],
    repository: Optional[str],
    model_label: Optional[str],
) -> Optional[str]:
    if log_type in {_AAWM_ROUTE_LOG_EMBED_TYPE, _AAWM_ROUTE_LOG_RERANK_TYPE}:
        return model_label

    owner_label = None
    if agent_name:
        owner_label = agent_name
        if agent_id:
            owner_label = f"{owner_label}#{agent_id}"
    elif agent_id:
        owner_label = f"#{agent_id}"

    if owner_label and repository:
        owner_label = f"{owner_label}@{repository}"
    elif repository:
        owner_label = repository

    if owner_label and model_label:
        return f"{owner_label}.{model_label}"
    return owner_label or model_label


def _default_aawm_route_log_agent_name(
    *,
    log_type: str,
    agent_name: Optional[str],
    repository: Optional[str],
    client_product_label: Optional[str],
) -> Optional[str]:
    if agent_name:
        return agent_name
    if log_type != _AAWM_ROUTE_ACCESS_LOG_TYPE or not repository:
        return None
    if _is_aawm_route_log_tui_client_label(client_product_label):
        return _AAWM_ROUTE_LOG_DEFAULT_AGENT
    return None


def _safe_aawm_route_endpoint_label(request: Request) -> str:
    request_url = getattr(request, "url", None)
    parsed_url = urlparse(str(request_url or ""))
    path = parsed_url.path or getattr(request, "path", None) or "/"
    query_pairs = []
    for key, value in parse_qsl(parsed_url.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key not in _AAWM_ROUTE_LOG_SAFE_QUERY_KEYS:
            continue
        safe_key = _clean_aawm_route_log_field(key)
        safe_value = _clean_aawm_route_log_field(value)
        if not safe_key or safe_value is None or "->" in safe_value:
            continue
        query_pairs.append((safe_key, safe_value))

    if not query_pairs:
        return path
    return f"{path}?{urlencode(query_pairs)}"


def _safe_aawm_route_target_label(target: Union[str, httpx.URL]) -> str:
    parsed_url = urlparse(str(target))
    hostname = parsed_url.hostname or "unknown-target"
    path = parsed_url.path or "/"
    return f"{hostname}{path}"


def _get_aawm_route_client_label(request: Request) -> Optional[str]:
    scope = getattr(request, "scope", None)
    client = scope.get("client") if isinstance(scope, dict) else None
    if isinstance(client, tuple) and len(client) >= 2:
        host = _clean_aawm_route_log_field(client[0])
        port = _clean_aawm_route_log_field(client[1])
        if host and port:
            return f"{host}:{port}"

    request_client = getattr(request, "client", None)
    host = _clean_aawm_route_log_field(getattr(request_client, "host", None))
    port = _clean_aawm_route_log_field(getattr(request_client, "port", None))
    if host and port:
        return f"{host}:{port}"
    return None


def _get_aawm_route_protocol_label(request: Request) -> str:
    scope = getattr(request, "scope", None)
    http_version = scope.get("http_version") if isinstance(scope, dict) else None
    version = _clean_aawm_route_log_field(http_version)
    if not version:
        return "HTTP"
    return f"HTTP/{version}"


def _get_aawm_route_native_access_log_path(request: Request) -> Optional[str]:
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return None

    path = scope.get("path")
    if not isinstance(path, str):
        return None

    full_path = quote(path)
    query_string = scope.get("query_string")
    if not query_string:
        return full_path

    try:
        if isinstance(query_string, bytes):
            query_label = query_string.decode("ascii")
        else:
            query_label = str(query_string)
    except UnicodeDecodeError:
        return full_path

    return f"{full_path}?{query_label}"


def _register_aawm_route_access_log_replacement(request: Request) -> None:
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return

    client = scope.get("client")
    client_addr = None
    if isinstance(client, (list, tuple)) and len(client) >= 2:
        client_addr = f"{client[0]}:{client[1]}"

    register_aawm_route_access_log_replacement(
        client_addr=client_addr,
        method=str(scope.get("method") or getattr(request, "method", "") or ""),
        full_path=_get_aawm_route_native_access_log_path(request),
        http_version=str(scope.get("http_version") or ""),
    )


def _get_aawm_route_log_model_label(
    request_body: Optional[dict[str, Any]],
    metadata: dict[str, Any],
) -> Optional[str]:
    model = None
    if isinstance(request_body, dict):
        model = _clean_aawm_route_log_field(request_body.get("model"))
    if model is None:
        model = _first_aawm_route_log_value(
            metadata,
            keys=_AAWM_ROUTE_LOG_SELECTED_MODEL_METADATA_KEYS,
        )

    alias = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_MODEL_ALIAS_METADATA_KEYS,
    )
    if model and alias and alias != model:
        return f"{model}({alias})"
    return model or alias


def build_aawm_route_access_log_line(
    *,
    request: Request,
    target: Union[str, httpx.URL],
    request_body: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> str:
    metadata = _extract_aawm_route_log_metadata(request_body, kwargs)
    metadata_sources = _iter_aawm_route_log_metadata_sources(
        request_body,
        kwargs,
        metadata,
    )
    headers = _extract_aawm_route_log_headers(request, kwargs)
    client_product_label = _get_aawm_route_log_client_product_label(
        metadata,
        headers,
    )
    if client_product_label:
        client_product_label = _normalize_aawm_route_log_client_label(
            client_product_label
        )
    log_type = _get_aawm_route_log_type(
        request=request,
        request_body=request_body,
        kwargs=kwargs,
    )
    agent_name = _first_aawm_route_log_value(
        *metadata_sources,
        keys=_AAWM_ROUTE_LOG_AGENT_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_agent_label,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_AGENT_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_agent_label,
    )
    agent_id = _get_aawm_route_log_agent_id(
        metadata_sources=metadata_sources,
        headers=headers,
    )
    repository = _get_aawm_route_log_repository(
        metadata_sources=metadata_sources,
        headers=headers,
        request_body=request_body,
        kwargs=kwargs,
    )
    agent_name = _default_aawm_route_log_agent_name(
        log_type=log_type,
        agent_name=agent_name,
        repository=repository,
        client_product_label=client_product_label,
    )
    model_label = _get_aawm_route_log_model_label(request_body, metadata)
    client_label = _get_aawm_route_client_label(request)
    method = _clean_aawm_route_log_field(getattr(request, "method", None)) or "REQUEST"
    incoming_endpoint = _safe_aawm_route_endpoint_label(request)
    outgoing_target = _safe_aawm_route_target_label(target)
    timestamp = (now or datetime.now()).strftime("%Y%m%d %H:%M:%S")
    identity_label = _build_aawm_route_log_identity_label(
        log_type=log_type,
        agent_name=agent_name,
        agent_id=agent_id,
        repository=repository,
        model_label=model_label,
    )

    segments: list[str] = [timestamp, log_type]
    if client_product_label:
        segments.append(f"{client_product_label} -")
    if identity_label:
        segments.append(identity_label)
    segments.append(method)
    if client_label:
        segments.append(client_label)
    segments.append(f"{incoming_endpoint} -> {outgoing_target}")
    return " ".join(segments)


def emit_aawm_route_access_log(
    *,
    request: Request,
    target: Union[str, httpx.URL],
    request_body: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict] = None,
) -> None:
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        if scope.get(_AAWM_ROUTE_ACCESS_LOG_SCOPE_KEY):
            return
        scope[_AAWM_ROUTE_ACCESS_LOG_SCOPE_KEY] = True

    line = build_aawm_route_access_log_line(
        request=request,
        target=target,
        request_body=request_body,
        kwargs=kwargs,
    )
    _register_aawm_route_access_log_replacement(request)
    logging.getLogger(_AAWM_ROUTE_ACCESS_LOGGER_NAME).info("%s", line)
