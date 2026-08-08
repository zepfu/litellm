"""Authoritative route descriptors for managed and native xAI models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional

XAIRouteFamily = Literal["xai_oauth_api", "grok_cli_chat_proxy"]
XAICredentialFamily = Literal["xai_oauth", "xai_grok_oidc"]
XAIAuthMode = Literal["oauth", "grok_oidc"]

OA_XAI_PROVIDER_PREFIX = "oa_xai/"
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
    """Return a native OIDC descriptor only for explicitly allowed models."""

    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if candidate.startswith("xai/"):
        candidate = candidate[len("xai/") :]
    return GROK_NATIVE_ROUTE_DESCRIPTORS.get(candidate)


def get_xai_route_descriptor(model: Any) -> Optional[XAIRouteDescriptor]:
    """Resolve one xAI model without crossing credential families."""

    return get_oa_xai_route_descriptor(model) or get_grok_native_route_descriptor(model)
