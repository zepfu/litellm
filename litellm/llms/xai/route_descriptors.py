"""Authoritative route descriptors for managed and native xAI models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
XAI_NATIVE_GROK_CONTINUATION_RETRY_CAPABILITY = "native_grok_continuation_retry"
_XAI_MODEL_CAPABILITIES_KEY = "capabilities"


@dataclass(frozen=True)
class XAIRouteDescriptor:
    """Canonical model, route, and credential-family selection."""

    public_model: str
    upstream_model: str
    route_family: XAIRouteFamily
    credential_family: XAICredentialFamily
    auth_mode: XAIAuthMode
    capabilities: frozenset[str] = frozenset()


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


def _native_descriptor(
    model: str,
    *,
    capabilities: frozenset[str] = frozenset(),
) -> XAIRouteDescriptor:
    return XAIRouteDescriptor(
        public_model=model,
        upstream_model=model,
        route_family=GROK_NATIVE_OAUTH_ROUTE_FAMILY,
        credential_family=GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY,
        auth_mode="grok_oidc",
        capabilities=capabilities,
    )


@lru_cache(maxsize=128)
def _get_xai_model_capabilities(model: str) -> frozenset[str]:
    """Read xAI model capabilities from the canonical model metadata."""

    try:
        from litellm.utils import get_model_info

        model_info = get_model_info(
            model=model,
            custom_llm_provider="xai",
        )
    except Exception:
        return frozenset()

    provider_specific_entry = model_info.get("provider_specific_entry")
    if not isinstance(provider_specific_entry, Mapping):
        return frozenset()
    provider_entry = provider_specific_entry.get("xai")
    if not isinstance(provider_entry, Mapping):
        return frozenset()
    raw_capabilities = provider_entry.get(_XAI_MODEL_CAPABILITIES_KEY)
    if not isinstance(raw_capabilities, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(
        str(capability).strip()
        for capability in raw_capabilities
        if str(capability).strip()
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
        return _native_descriptor(
            native_model,
            capabilities=_get_xai_model_capabilities(native_model),
        )
    descriptor = GROK_NATIVE_ROUTE_DESCRIPTORS.get(candidate)
    if descriptor is None:
        return None
    return _native_descriptor(
        candidate,
        capabilities=_get_xai_model_capabilities(candidate),
    )


def has_grok_native_route_capability(model: Any, capability: str) -> bool:
    """Return whether a native Grok model explicitly declares ``capability``."""

    descriptor = get_grok_native_route_descriptor(model)
    return descriptor is not None and capability in descriptor.capabilities


def get_xai_route_descriptor(model: Any) -> Optional[XAIRouteDescriptor]:
    """Resolve one xAI model without crossing credential families."""

    return get_oa_xai_route_descriptor(model) or get_grok_native_route_descriptor(model)
