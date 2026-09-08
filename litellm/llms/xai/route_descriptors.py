"""Authoritative route descriptors for managed and native xAI models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional
from urllib.parse import urlsplit

XAIRouteFamily = Literal["xai_oauth_api", "grok_cli_chat_proxy"]
XAICredentialFamily = Literal["xai_oauth", "xai_grok_oidc"]
XAIAuthMode = Literal["oauth", "grok_oidc"]

OA_XAI_PROVIDER_PREFIX = "oa_xai/"
XAI_OAUTH_API_HOST = "api.x.ai"
XAI_OAUTH_API_BASE_PATHS = frozenset({"", "/", "/v1"})
XAI_OAUTH_API_ALLOWED_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/responses",
    }
)
XAI_OAUTH_ROUTE_FAMILY: XAIRouteFamily = "xai_oauth_api"
XAI_OAUTH_CREDENTIAL_FAMILY: XAICredentialFamily = "xai_oauth"
GROK_NATIVE_OAUTH_ROUTE_FAMILY: XAIRouteFamily = "grok_cli_chat_proxy"
GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY: XAICredentialFamily = "xai_grok_oidc"


@dataclass(frozen=True)
class XAIRouteDescriptor:
    """Canonical model, route, and credential-family selection."""

    public_model: str
    upstream_model: str
    route_family: XAIRouteFamily
    credential_family: XAICredentialFamily
    auth_mode: XAIAuthMode


def _parse_xai_oauth_url(url: Any, *, label: str) -> Any:
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{label} must be a non-empty URL.")

    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError(f"{label} must use HTTPS.")
    if (parsed.hostname or "").lower() != XAI_OAUTH_API_HOST:
        raise ValueError(f"{label} must target {XAI_OAUTH_API_HOST}.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not include URL credentials.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} has an invalid port.") from exc
    if port not in (None, 443):
        raise ValueError(f"{label} must use the default HTTPS port.")
    if parsed.fragment:
        raise ValueError(f"{label} must not include a URL fragment.")
    return parsed


def validate_xai_oauth_api_base(url: Any) -> None:
    """Fail closed unless a managed xAI API base is the approved host/path."""

    parsed = _parse_xai_oauth_url(url, label="xAI OAuth API base")
    normalized_path = parsed.path.rstrip("/") or "/"
    normalized_base_paths = {
        path.rstrip("/") or "/" for path in XAI_OAUTH_API_BASE_PATHS
    }
    if normalized_path not in normalized_base_paths:
        raise ValueError(
            "xAI OAuth API base must use the root or /v1 API path."
        )
    if parsed.query:
        raise ValueError("xAI OAuth API base must not include query parameters.")


def validate_xai_oauth_api_target(url: Any) -> None:
    """Fail closed unless a managed xAI request uses an approved API path."""

    parsed = _parse_xai_oauth_url(url, label="xAI OAuth API target")
    normalized_path = parsed.path.rstrip("/") or "/"
    normalized_target_paths = {
        path.rstrip("/") or "/" for path in XAI_OAUTH_API_ALLOWED_PATHS
    }
    if normalized_path not in normalized_target_paths:
        raise ValueError(
            "xAI OAuth API target must use /v1/responses or "
            "/v1/chat/completions."
        )


def _managed_descriptor(
    public_model: str,
    upstream_model: str,
) -> XAIRouteDescriptor:
    return XAIRouteDescriptor(
        public_model=public_model,
        upstream_model=upstream_model,
        route_family=XAI_OAUTH_ROUTE_FAMILY,
        credential_family=XAI_OAUTH_CREDENTIAL_FAMILY,
        auth_mode="oauth",
    )


def _native_descriptor(model: str) -> XAIRouteDescriptor:
    return XAIRouteDescriptor(
        public_model=model,
        upstream_model=model,
        route_family=GROK_NATIVE_OAUTH_ROUTE_FAMILY,
        credential_family=GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY,
        auth_mode="grok_oidc",
    )


OA_XAI_ROUTE_DESCRIPTORS: Mapping[str, XAIRouteDescriptor] = MappingProxyType(
    {
        public_model: _managed_descriptor(public_model, upstream_model)
        for public_model, upstream_model in (
            ("oa_xai/grok-4.3", "xai/grok-4.3"),
            ("oa_xai/grok-4.5", "xai/grok-4.5"),
            ("oa_xai/grok-4.6", "xai/grok-4.6"),
            (
                "oa_xai/grok-4.20-0309-reasoning",
                "xai/grok-4.20-0309-reasoning",
            ),
            (
                "oa_xai/grok-4.20-0309-non-reasoning",
                "xai/grok-4.20-0309-non-reasoning",
            ),
            (
                "oa_xai/grok-4.20-multi-agent-0309",
                "xai/grok-4.20-multi-agent-0309",
            ),
        )
    }
)

GROK_NATIVE_ROUTE_DESCRIPTORS: Mapping[str, XAIRouteDescriptor] = MappingProxyType(
    {
        model: _native_descriptor(model)
        for model in (
            "grok-build",
            "grok-build-0.1",
            "grok-composer-2.5-fast",
            "grok-4.5",
            "grok-4.6",
        )
    }
)


def get_oa_xai_route_descriptor(model: Any) -> Optional[XAIRouteDescriptor]:
    """Return managed OAuth descriptor, including open-ended prefix fallback."""

    if not isinstance(model, str) or not model.startswith(OA_XAI_PROVIDER_PREFIX):
        return None
    descriptor = OA_XAI_ROUTE_DESCRIPTORS.get(model)
    if descriptor is not None:
        return descriptor
    return _managed_descriptor(
        public_model=model,
        upstream_model="xai/" + model[len(OA_XAI_PROVIDER_PREFIX) :],
    )


def resolve_oa_xai_route_descriptor(model: str) -> XAIRouteDescriptor:
    descriptor = get_oa_xai_route_descriptor(model)
    if descriptor is None:
        raise ValueError(f"Unsupported xAI OAuth-managed model: {model}")
    return descriptor


def get_grok_native_route_descriptor(model: Any) -> Optional[XAIRouteDescriptor]:
    """Return a native OIDC descriptor for known and explicitly prefixed models."""

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if candidate.startswith("xai/"):
        native_model = candidate[len("xai/") :].strip()
        if not native_model:
            return None
        return GROK_NATIVE_ROUTE_DESCRIPTORS.get(native_model) or _native_descriptor(
            native_model
        )
    return GROK_NATIVE_ROUTE_DESCRIPTORS.get(candidate)


def get_xai_route_descriptor(model: Any) -> Optional[XAIRouteDescriptor]:
    """Resolve one xAI model without crossing credential families."""

    return get_oa_xai_route_descriptor(model) or get_grok_native_route_descriptor(model)
