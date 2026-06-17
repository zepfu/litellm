import logging
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
_AAWM_ROUTE_LOG_AGENT_ID_METADATA_KEYS = (
    "agent_id",
    "aawm_agent_id",
    "aawm_claude_agent_id",
    "claude_agent_id",
    "codex_agent_id",
)
_AAWM_ROUTE_LOG_AGENT_HEADER_KEYS = (
    "x-aawm-agent-name",
    "x-litellm-agent-name",
    "x-agent-name",
)
_AAWM_ROUTE_LOG_AGENT_ID_HEADER_KEYS = (
    "x-aawm-agent-id",
    "x-litellm-agent-id",
    "x-agent-id",
    "x-claude-agent-id",
    "x-codex-agent-id",
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
_AAWM_ROUTE_LOG_REPOSITORY_TENANT_HEADER_KEYS = (
    "x-aawm-tenant-id",
    "x-litellm-tenant-id",
    "x-litellm-organization-id",
    "x-litellm-org-id",
    "x-organization-id",
    "x-org-id",
    "x-litellm-team-id",
    "x-team-id",
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
)
_AAWM_ROUTE_LOG_SELECTED_MODEL_METADATA_KEYS = (
    "codex_auto_agent_selected_model",
    "anthropic_auto_agent_selected_model",
    "anthropic_adapter_model",
    "xai_oauth_public_model",
    "xai_oauth_upstream_model",
    "grok_model_override",
)
_AAWM_ROUTE_LOG_TOP_LEVEL_METADATA_KEYS = (
    _AAWM_ROUTE_LOG_AGENT_METADATA_KEYS
    + _AAWM_ROUTE_LOG_AGENT_ID_METADATA_KEYS
    + _AAWM_ROUTE_LOG_REPOSITORY_METADATA_KEYS
    + _AAWM_ROUTE_LOG_CLIENT_LABEL_METADATA_KEYS
    + _AAWM_ROUTE_LOG_CLIENT_NAME_METADATA_KEYS
    + _AAWM_ROUTE_LOG_CLIENT_VERSION_METADATA_KEYS
    + _AAWM_ROUTE_LOG_MODEL_ALIAS_METADATA_KEYS
    + _AAWM_ROUTE_LOG_SELECTED_MODEL_METADATA_KEYS
    + ("trace_name", "trace_user_id")
)
_AAWM_ROUTE_LOG_TENANT_REPOSITORY_FRAGMENTS = (
    "harness",
    "validation",
)


def _normalize_aawm_route_log_type(
    route_type: Optional[str],
    incoming_endpoint: Optional[str] = None,
) -> str:
    route_type_label = (route_type or "").lower().strip()
    endpoint_label = (incoming_endpoint or "").lower()
    if route_type_label in {"aembedding", "embedding", "embeddings"}:
        return "EMBED"
    if route_type_label in {"arerank", "rerank"}:
        return "RERANK"
    if "/embeddings" in endpoint_label:
        return "EMBED"
    if "/rerank" in endpoint_label:
        return "RERANK"
    return _AAWM_ROUTE_ACCESS_LOG_TYPE


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


def _normalize_aawm_route_log_agent_id(value: Any) -> Optional[str]:
    cleaned = _clean_aawm_route_log_field(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_LOG_MAX_IDENTITY_CHARS:
        return None
    if _is_aawm_route_log_slug(cleaned):
        return cleaned
    return None


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
        cleaned = path_parts[-1]

    if _is_aawm_route_log_slug(cleaned):
        return cleaned
    return None


def _normalize_aawm_route_log_tenant_repository_label(value: Any) -> Optional[str]:
    repository = _normalize_aawm_route_log_repository_label(value)
    if repository is None:
        return None

    normalized = repository.lower()
    if any(
        fragment in normalized
        for fragment in _AAWM_ROUTE_LOG_TENANT_REPOSITORY_FRAGMENTS
    ):
        return "litellm"
    if normalized.endswith("-dev") or "tenant" in normalized:
        return None
    return repository


def _normalize_aawm_route_log_known_client_name(name: str) -> str:
    normalized_name = name.lower().replace("_", "-")
    if normalized_name in {"claude", "claude-cli", "claude-code"}:
        return "Claude"
    if normalized_name in {"codex", "codex-cli", "codex-tui", "codex-cli-rs"}:
        return "Codex"
    if normalized_name in {"grok", "grok-build", "grok-pager"}:
        return "Grok"
    if normalized_name in {"gemini", "gemini-cli"}:
        return "Gemini"
    if normalized_name in {"opencode", "opencode-tui"}:
        return "OpenCode"
    if normalized_name in {"cursor", "cursor-cli"}:
        return "Cursor"
    return name


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
            return f"{_normalize_aawm_route_log_known_client_name(name)}/{version}"
        return None
    if _is_aawm_route_log_slug(product):
        return _normalize_aawm_route_log_known_client_name(product)
    return None


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


def _extract_aawm_route_log_metadata(
    request_body: Optional[dict[str, Any]],
    kwargs: Optional[dict],
) -> dict[str, Any]:
    body_metadata: dict[str, Any] = {}
    if isinstance(request_body, dict):
        for key in _AAWM_ROUTE_LOG_TOP_LEVEL_METADATA_KEYS:
            if key in request_body:
                body_metadata[key] = request_body[key]
        for metadata_key in ("litellm_metadata", "metadata"):
            metadata_value = request_body.get(metadata_key)
            if isinstance(metadata_value, dict):
                body_metadata.update(
                    {
                        key: value
                        for key, value in metadata_value.items()
                        if key in _AAWM_ROUTE_LOG_TOP_LEVEL_METADATA_KEYS
                    }
                )

    kwargs_metadata: dict[str, Any] = {}
    if isinstance(kwargs, dict):
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            metadata = litellm_params.get("metadata")
            if isinstance(metadata, dict):
                kwargs_metadata = metadata
    return {**body_metadata, **kwargs_metadata}


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
        return f"{client_name}/{client_version}"
    return client_name


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


def _get_aawm_route_log_context_label(
    *,
    agent_name: Optional[str],
    agent_id: Optional[str],
    repository: Optional[str],
    model_label: Optional[str],
) -> Optional[str]:
    agent_label = agent_name
    if agent_label and agent_id:
        agent_label = f"{agent_label}#{agent_id}"
    elif agent_id:
        agent_label = f"#{agent_id}"

    if agent_label and repository:
        owner_label = f"{agent_label}@{repository}"
    else:
        owner_label = agent_label or repository

    if owner_label and model_label:
        return f"{owner_label}.{model_label}"
    return owner_label or model_label


def _get_aawm_route_log_trace_user_repository(
    metadata: dict[str, Any],
    headers: dict[str, Any],
) -> Optional[str]:
    trace_name = _first_aawm_route_log_value(
        metadata,
        keys=("trace_name",),
    ) or _get_case_insensitive_header_value(
        headers,
        ("langfuse_trace_name",),
    )
    if not trace_name:
        return None

    normalized_trace_name = trace_name.lower()
    if not normalized_trace_name.startswith(
        ("claude-code", "codex", "grok-build", "grok")
    ):
        return None

    return _first_aawm_route_log_value(
        metadata,
        keys=("trace_user_id",),
        normalizer=_normalize_aawm_route_log_repository_label,
    ) or _get_case_insensitive_header_value(
        headers,
        ("langfuse_trace_user_id",),
        normalizer=_normalize_aawm_route_log_repository_label,
    )


def build_aawm_route_access_log_line(
    *,
    request: Request,
    target: Union[str, httpx.URL],
    request_body: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict] = None,
    route_type: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    metadata = _extract_aawm_route_log_metadata(request_body, kwargs)
    headers = dict(getattr(request, "headers", {}) or {})
    client_product_label = _get_aawm_route_log_client_product_label(
        metadata,
        headers,
    )
    agent_name = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_AGENT_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_agent_label,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_AGENT_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_agent_label,
    )
    agent_id = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_AGENT_ID_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_agent_id,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_AGENT_ID_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_agent_id,
    )
    repository = _first_aawm_route_log_value(
        metadata,
        keys=_AAWM_ROUTE_LOG_REPOSITORY_METADATA_KEYS,
        normalizer=_normalize_aawm_route_log_repository_label,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_REPOSITORY_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_repository_label,
    ) or _get_aawm_route_log_trace_user_repository(
        metadata,
        headers,
    ) or _get_case_insensitive_header_value(
        headers,
        _AAWM_ROUTE_LOG_REPOSITORY_TENANT_HEADER_KEYS,
        normalizer=_normalize_aawm_route_log_tenant_repository_label,
    )
    model_label = _get_aawm_route_log_model_label(request_body, metadata)
    context_label = _get_aawm_route_log_context_label(
        agent_name=agent_name,
        agent_id=agent_id,
        repository=repository,
        model_label=model_label,
    )
    client_label = _get_aawm_route_client_label(request)
    method = _clean_aawm_route_log_field(getattr(request, "method", None)) or "REQUEST"
    incoming_endpoint = _safe_aawm_route_endpoint_label(request)
    log_type = _normalize_aawm_route_log_type(route_type, incoming_endpoint)
    outgoing_target = _safe_aawm_route_target_label(target)
    timestamp = (now or datetime.now()).strftime("%Y%m%d %H:%M:%S")

    segments: list[str] = [timestamp, f"[{log_type}]"]
    if client_product_label:
        segments.append(client_product_label)
    if context_label:
        segments.append(f"- {context_label}")
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
    route_type: Optional[str] = None,
) -> None:
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        if scope.get(_AAWM_ROUTE_ACCESS_LOG_SCOPE_KEY):
            _register_aawm_route_access_log_replacement(request)
            return
        scope[_AAWM_ROUTE_ACCESS_LOG_SCOPE_KEY] = True

    line = build_aawm_route_access_log_line(
        request=request,
        target=target,
        request_body=request_body,
        kwargs=kwargs,
        route_type=route_type,
    )
    _register_aawm_route_access_log_replacement(request)
    logging.getLogger(_AAWM_ROUTE_ACCESS_LOGGER_NAME).info("%s", line)
