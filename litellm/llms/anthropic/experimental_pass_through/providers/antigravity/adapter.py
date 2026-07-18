"""Antigravity request and header shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload


def _default_split_provider_prefix(
    _value: object,
) -> tuple[Optional[str], Optional[str]]:
    return None, None


@dataclass(frozen=True)
class Runtime:
    """Injected configuration and observability transforms."""

    get_client_header: Callable[[], str]
    merge_metadata: Callable[..., Payload]
    prepare_observability: Callable[..., Payload]
    split_provider_prefix: Callable[
        [object], tuple[Optional[str], Optional[str]]
    ] = _default_split_provider_prefix
    allowed_models: frozenset[str] = frozenset()
    http_exception_type: type[Exception] = Exception


def _normalize_antigravity_code_assist_adapter_model_name(
    model: object,
    *,
    runtime: Runtime,
) -> Optional[str]:
    explicit_provider, candidate = runtime.split_provider_prefix(model)
    if (
        explicit_provider not in {"antigravity", "agy", "google-antigravity"}
        or candidate is None
    ):
        return None
    normalized_candidate = candidate.strip()
    if normalized_candidate in runtime.allowed_models:
        return normalized_candidate
    return None


def _is_codex_auto_agent_antigravity_auth_degraded_exception(
    exc: object,
    *,
    runtime: Runtime,
) -> bool:
    if not isinstance(exc, runtime.http_exception_type):
        return False
    detail_text = str(getattr(exc, "detail", ""))
    return "Antigravity OAuth" in detail_text and (
        "expired or invalid" in detail_text
        or "does not contain" in detail_text
        or "sidecar owns Antigravity auth refresh" in detail_text
    )


def _build_antigravity_native_headers(
    access_token: str,
    *,
    runtime: Runtime,
) -> dict[str, str]:
    """Build the Antigravity Code Assist native header contract."""
    client_header = runtime.get_client_header()
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": client_header,
        "x-goog-api-client": client_header,
        "Accept": "application/json",
    }


def _prepare_antigravity_request_body_for_passthrough(
    *,
    runtime: Runtime,
    request: object,
    request_body: Payload,
) -> Payload:
    """Attach Antigravity route metadata before shared observability shaping."""
    updated_body = runtime.merge_metadata(
        request_body,
        tags_to_add=[
            "antigravity-code-assist",
            "route:antigravity_code_assist",
        ],
        extra_fields={
            "client_name": "antigravity-cli",
            "antigravity_code_assist": True,
            "passthrough_route_family": "antigravity_code_assist",
        },
    )
    return runtime.prepare_observability(
        request=request,
        request_body=updated_body,
    )


def _normalize_antigravity_endpoint_for_target(endpoint: str) -> str:
    """Normalize a client endpoint for joining to the Antigravity target."""
    endpoint_without_query = endpoint.split("?", 1)[0]
    normalized_endpoint = endpoint_without_query.lstrip("/")
    if not normalized_endpoint:
        return "/"
    return f"/{normalized_endpoint}"


def _is_antigravity_streaming_endpoint(endpoint: str, request: object) -> bool:
    normalized_endpoint = endpoint.lstrip("/")
    query_params = getattr(request, "query_params", {})
    return "streamGenerateContent" in normalized_endpoint or (
        str(query_params.get("alt", "")).lower() == "sse"
    )


__all__ = [
    "Runtime",
    "_build_antigravity_native_headers",
    "_is_antigravity_streaming_endpoint",
    "_is_codex_auto_agent_antigravity_auth_degraded_exception",
    "_normalize_antigravity_code_assist_adapter_model_name",
    "_normalize_antigravity_endpoint_for_target",
    "_prepare_antigravity_request_body_for_passthrough",
]
