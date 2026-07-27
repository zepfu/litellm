"""Antigravity pass-through runtime helpers.

Host-owned behavior is supplied through ``Runtime`` so this module remains
independent of ``llm_passthrough_endpoints``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx
from fastapi import HTTPException, Request

from litellm.llms.anthropic.experimental_pass_through.providers.antigravity import (
    adapter as _antigravity_adapter,
)
from litellm.llms.anthropic.experimental_pass_through.providers.antigravity import (
    constants as _antigravity_constants,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    antigravity_oauth as _antigravity_oauth,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload

_ANTIGRAVITY_CODE_ASSIST_DEFAULT_BASE_URL = (
    "https://daily-cloudcode-pa.googleapis.com"
)
_ANTIGRAVITY_CLIENT_HEADER_DEFAULT = "antigravity-cli/1.0.4"
_ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST = (
    _antigravity_constants._ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST
)


@dataclass(frozen=True)
class Runtime:
    """Injected host configuration for Antigravity request shaping."""

    clean_value: Callable[[object], Optional[str]]
    merge_metadata: Callable[..., Payload]
    prepare_observability: Callable[..., Payload]
    split_provider_prefix: Callable[
        [object], tuple[Optional[str], Optional[str]]
    ]
    format_api_key: Callable[[Optional[str]], str]
    oauth_error_code: Callable[[httpx.Response], Optional[str]]
    allowed_models: frozenset[str] = frozenset()
    cli_binary_path_env_vars: tuple[str, ...] = (
        _antigravity_oauth._ANTIGRAVITY_CLI_BINARY_PATH_ENV_VARS
    )
    default_cli_binary_paths: tuple[str, ...] = (
        _antigravity_oauth._ANTIGRAVITY_DEFAULT_CLI_BINARY_PATHS
    )
    default_base_url: str = _ANTIGRAVITY_CODE_ASSIST_DEFAULT_BASE_URL
    client_header_default: str = _ANTIGRAVITY_CLIENT_HEADER_DEFAULT
    getenv: Callable[[str], Optional[str]] = os.getenv
    http_exception_type: type[Exception] = HTTPException


def _iter_antigravity_cli_binary_candidates(
    *,
    runtime: Runtime,
) -> list[Path]:
    candidate_files: list[Path] = []
    seen_paths: set[str] = set()
    for env_name in runtime.cli_binary_path_env_vars:
        raw_value = runtime.clean_value(runtime.getenv(env_name))
        if not raw_value:
            continue
        candidate = Path(raw_value).expanduser()
        if candidate.is_file():
            resolved = str(candidate.resolve())
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                candidate_files.append(candidate)

    for candidate_str in runtime.default_cli_binary_paths:
        candidate = Path(candidate_str).expanduser()
        if not candidate.is_file():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidate_files.append(candidate)
    return candidate_files


def _format_antigravity_oauth_refresh_failure_detail(
    *,
    response: httpx.Response,
    runtime: Runtime,
) -> str:
    error_code = runtime.oauth_error_code(response)
    suffix = (
        f"status={response.status_code}, error={error_code}"
        if error_code
        else f"status={response.status_code}"
    )
    return (
        f"Failed to refresh Antigravity OAuth access token ({suffix}). "
        "Re-authenticate Antigravity CLI or configure valid OAuth client "
        "environment overrides."
    )


def _get_antigravity_passthrough_target_base(
    *,
    runtime: Runtime,
) -> str:
    return (
        runtime.getenv("ANTIGRAVITY_CODE_ASSIST_ENDPOINT")
        or runtime.getenv("ANTIGRAVITY_CLI_CODE_ASSIST_ENDPOINT")
        or runtime.default_base_url
    )


def _get_antigravity_client_header(
    *,
    runtime: Runtime,
) -> str:
    return (
        runtime.clean_value(runtime.getenv("AAWM_ANTIGRAVITY_CLIENT_HEADER"))
        or runtime.client_header_default
    )


def _get_anthropic_antigravity_runtime(
    *,
    runtime: Runtime,
) -> _antigravity_adapter.Runtime:
    return _antigravity_adapter.Runtime(
        get_client_header=lambda: _get_antigravity_client_header(
            runtime=runtime
        ),
        merge_metadata=runtime.merge_metadata,
        prepare_observability=runtime.prepare_observability,
        split_provider_prefix=runtime.split_provider_prefix,
        allowed_models=runtime.allowed_models,
        http_exception_type=runtime.http_exception_type,
    )


def _build_antigravity_native_headers(
    access_token: str,
    *,
    runtime: Runtime,
) -> dict[str, str]:
    return _antigravity_adapter._build_antigravity_native_headers(
        access_token,
        runtime=_get_anthropic_antigravity_runtime(runtime=runtime),
    )


def _request_has_google_oauth_bearer(request: Request) -> bool:
    authorization = request.headers.get("authorization", "").strip()
    return authorization.lower().startswith("bearer ya29.")


def _get_antigravity_litellm_auth_header(
    request: Request,
    *,
    runtime: Runtime,
) -> str:
    header_key = request.headers.get("x-litellm-api-key")
    if header_key:
        return runtime.format_api_key(header_key)

    query_key = request.query_params.get("key")
    if query_key:
        return runtime.format_api_key(query_key)

    return request.headers.get("authorization", "")


def _prepare_antigravity_request_body_for_passthrough(
    *,
    request: Request,
    request_body: Payload,
    runtime: Runtime,
) -> Payload:
    return _antigravity_adapter._prepare_antigravity_request_body_for_passthrough(
        runtime=_get_anthropic_antigravity_runtime(runtime=runtime),
        request=request,
        request_body=request_body,
    )


def _get_antigravity_request_project(
    request_body: dict[str, object],
    *,
    runtime: Runtime,
) -> Optional[str]:
    return runtime.clean_value(request_body.get("project"))


def _get_antigravity_passthrough_logging_metadata(
    request: Request,
    *,
    runtime: Runtime,
) -> dict[str, object]:
    logging_body = _prepare_antigravity_request_body_for_passthrough(
        request=request,
        request_body={},
        runtime=runtime,
    )
    litellm_metadata = logging_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        return dict(litellm_metadata)
    return {}


def _normalize_antigravity_endpoint_for_target(endpoint: str) -> str:
    return _antigravity_adapter._normalize_antigravity_endpoint_for_target(
        endpoint
    )


def _join_antigravity_passthrough_url(
    base_target_url: str,
    endpoint: str,
) -> str:
    endpoint_path = _normalize_antigravity_endpoint_for_target(endpoint)
    base_url = httpx.URL(base_target_url)
    base_path = base_url.path.rstrip("/")
    if base_path:
        endpoint_path = f"{base_path}/{endpoint_path.lstrip('/')}"
    return str(base_url.copy_with(path=endpoint_path))


def _is_antigravity_streaming_endpoint(
    endpoint: str,
    request: Request,
) -> bool:
    return _antigravity_adapter._is_antigravity_streaming_endpoint(
        endpoint,
        request,
    )


__all__ = [
    "Runtime",
    "_ANTIGRAVITY_CODE_ASSIST_DEFAULT_BASE_URL",
    "_ANTIGRAVITY_CLIENT_HEADER_DEFAULT",
    "_ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST",
    "_build_antigravity_native_headers",
    "_format_antigravity_oauth_refresh_failure_detail",
    "_get_anthropic_antigravity_runtime",
    "_get_antigravity_client_header",
    "_get_antigravity_litellm_auth_header",
    "_get_antigravity_passthrough_logging_metadata",
    "_get_antigravity_passthrough_target_base",
    "_get_antigravity_request_project",
    "_is_antigravity_streaming_endpoint",
    "_iter_antigravity_cli_binary_candidates",
    "_join_antigravity_passthrough_url",
    "_normalize_antigravity_endpoint_for_target",
    "_prepare_antigravity_request_body_for_passthrough",
    "_request_has_google_oauth_bearer",
]
