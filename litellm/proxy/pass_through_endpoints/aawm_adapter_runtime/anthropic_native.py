"""Wave 7 extraction: Anthropic native passthrough owner functions.

Behavior-preserving extraction from ``llm_passthrough_endpoints.py``.
Do not import llm_passthrough_endpoints at module scope.

Owned symbols:
- ``_get_header_value_case_insensitive``
- ``_append_anthropic_beta_header_value``
- ``_prepare_anthropic_oauth_native_passthrough_headers``
- ``_normalize_anthropic_native_passthrough_model_alias``
- ``_prepare_anthropic_context_1m_native_passthrough``
- ``_perform_anthropic_native_passthrough_request``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
)

from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key
from litellm.types.llms.anthropic import ANTHROPIC_OAUTH_BETA_HEADER

if TYPE_CHECKING:
    from fastapi import Response
    from starlette.requests import Request

    from litellm.proxy._types import UserAPIKeyAuth

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX = "[1m]"
_ANTHROPIC_CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"
_ANTHROPIC_BETA_HEADER_NAME = "anthropic-beta"
_ANTHROPIC_BETA_XPASS_HEADER_NAME = f"x-pass-{_ANTHROPIC_BETA_HEADER_NAME}"
_ANTHROPIC_DANGEROUS_DIRECT_BROWSER_ACCESS_HEADER_NAME = (
    "anthropic-dangerous-direct-browser-access"
)
_ANTHROPIC_NATIVE_PASSTHROUGH_MODEL_ALIASES = {
    "opus": "claude-opus-4-6",
    "opus-4-6": "claude-opus-4-6",
    "opus-4-8": "claude-opus-4-8",
    "fable-5": "claude-fable-5",
    "claude-fable-5": "claude-fable-5",
    "sonnet": "claude-sonnet-4-20250514",
    "sonnet-4-6": "claude-sonnet-4-6",
    "sonnet-4-20250514": "claude-sonnet-4-20250514",
    "sonnet-5": "claude-sonnet-5",
    "claude-sonnet-5": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
    "haiku-4-5": "claude-haiku-4-5",
    "haiku-4-5-20251001": "claude-haiku-4-5-20251001",
}

# ---------------------------------------------------------------------------
# Owner functions
# ---------------------------------------------------------------------------


def _get_header_value_case_insensitive(
    headers: Any,
    header_name: str,
) -> Optional[str]:
    header_value = headers.get(header_name)
    if header_value is not None:
        return str(header_value)

    lowered_header_name = header_name.lower()
    for candidate_name, candidate_value in headers.items():
        if str(candidate_name).lower() == lowered_header_name:
            return str(candidate_value)
    return None


def _append_anthropic_beta_header_value(
    headers: dict[str, Any],
    beta_value: str,
) -> dict[str, Any]:
    existing_header_name = next(
        (
            header_name
            for header_name in headers
            if str(header_name).lower() == _ANTHROPIC_BETA_HEADER_NAME
        ),
        None,
    )
    existing_beta = (
        headers.pop(existing_header_name)
        if existing_header_name is not None
        else None
    )
    if existing_beta is None:
        headers[_ANTHROPIC_BETA_HEADER_NAME] = beta_value
        return headers

    existing_values = [
        value.strip() for value in str(existing_beta).split(",") if value.strip()
    ]
    if beta_value not in existing_values:
        existing_values.append(beta_value)
    headers[_ANTHROPIC_BETA_HEADER_NAME] = ", ".join(existing_values)
    return headers


def _prepare_anthropic_oauth_native_passthrough_headers(
    *,
    request: "Request",
    custom_headers: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    auth_header = _get_header_value_case_insensitive(
        request.headers, "authorization"
    )
    if not is_anthropic_oauth_key(auth_header):
        return custom_headers, False

    updated_headers = dict(custom_headers)
    request_beta = _get_header_value_case_insensitive(
        request.headers,
        _ANTHROPIC_BETA_HEADER_NAME,
    )
    if request_beta:
        for beta_value in str(request_beta).split(","):
            stripped_beta_value = beta_value.strip()
            if stripped_beta_value:
                _append_anthropic_beta_header_value(
                    updated_headers,
                    stripped_beta_value,
                )
    _append_anthropic_beta_header_value(
        updated_headers,
        ANTHROPIC_OAUTH_BETA_HEADER,
    )
    updated_headers[
        _ANTHROPIC_DANGEROUS_DIRECT_BROWSER_ACCESS_HEADER_NAME
    ] = "true"
    return updated_headers, True


def _normalize_anthropic_native_passthrough_model_alias(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    model = request_body.get("model")
    if not isinstance(model, str):
        return request_body, False

    stripped_model = model.strip()
    if not stripped_model:
        return request_body, False

    suffix = ""
    alias_model = stripped_model
    if stripped_model.lower().endswith(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX):
        suffix = stripped_model[-len(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX) :]
        alias_model = stripped_model[
            : -len(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX)
        ].strip()

    normalized_model = _ANTHROPIC_NATIVE_PASSTHROUGH_MODEL_ALIASES.get(
        alias_model.lower()
    )
    if normalized_model is None:
        return request_body, False

    provider_model = f"{normalized_model}{suffix}"
    if provider_model == stripped_model:
        return request_body, False

    updated_body = dict(request_body)
    updated_body["model"] = provider_model
    metadata = updated_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata.setdefault("inbound_model_alias", stripped_model)
    metadata.setdefault("requested_model_alias", stripped_model)
    metadata.setdefault("model_alias_label", stripped_model)
    metadata["anthropic_native_passthrough_model_alias"] = stripped_model
    metadata["anthropic_native_passthrough_normalized_model"] = provider_model
    updated_body["litellm_metadata"] = metadata
    return updated_body, True


def _prepare_anthropic_context_1m_native_passthrough(
    *,
    request: "Request",
    request_body: dict[str, Any],
    custom_headers: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    model = request_body.get("model")
    if not isinstance(model, str):
        return request_body, custom_headers, False

    stripped_model = model.strip()
    if not stripped_model.lower().endswith(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX):
        return request_body, custom_headers, False

    base_model = stripped_model[
        : -len(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX)
    ].strip()
    if not base_model:
        return request_body, custom_headers, False

    updated_body = dict(request_body)
    updated_body["model"] = base_model

    metadata = updated_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata.setdefault("inbound_model_alias", stripped_model)
    metadata.setdefault("requested_model_alias", stripped_model)
    metadata.setdefault("model_alias_label", stripped_model)
    metadata.setdefault(
        "anthropic_native_passthrough_model_alias", stripped_model
    )
    metadata["anthropic_native_passthrough_normalized_model"] = base_model
    updated_body["litellm_metadata"] = metadata

    updated_headers = dict(custom_headers)
    for beta_header_name in (
        _ANTHROPIC_BETA_HEADER_NAME,
        _ANTHROPIC_BETA_XPASS_HEADER_NAME,
    ):
        request_beta = _get_header_value_case_insensitive(
            request.headers,
            beta_header_name,
        )
        if isinstance(request_beta, str) and request_beta.strip():
            _append_anthropic_beta_header_value(updated_headers, request_beta)
    _append_anthropic_beta_header_value(
        updated_headers,
        _ANTHROPIC_CONTEXT_1M_BETA_HEADER,
    )
    return updated_body, updated_headers, True


# ---------------------------------------------------------------------------
# Runtime seam for _perform_anthropic_native_passthrough_request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnthropicNativeRuntime:
    """Host-owned callbacks required by the native passthrough executor.

    Every field is a callable injected by the integrator so that this
    module never imports the god module at module scope.
    """

    is_streaming_request_fn: Callable[["Request"], Awaitable[bool]]
    """Async predicate: (request) -> whether the body requests streaming."""

    create_pass_through_route: Callable[..., Callable[..., Awaitable["Response"]]]
    """Factory: (endpoint, target, custom_headers, ...) -> endpoint_func."""


_runtime: Optional[AnthropicNativeRuntime] = None


async def _perform_anthropic_native_passthrough_request(
    *,
    endpoint: str,
    request: "Request",
    fastapi_response: "Response",
    user_api_key_dict: "UserAPIKeyAuth",
    target_url: str,
    custom_headers: dict[str, Any],
    blocked_pass_through_prefixed_headers: Optional[list[str]] = None,
) -> "Response":
    """Execute a native Anthropic passthrough request.

    Fails closed if the runtime has not been installed.  All egress
    goes exclusively through the Anthropic-native target_url provided
    by the caller; this function never selects or mutates the provider
    route.
    """
    if _runtime is None:
        raise RuntimeError(
            "anthropic_native runtime not installed; "
            "call install() with AnthropicNativeRuntime before use"
        )
    is_streaming_request = await _runtime.is_streaming_request_fn(request)
    endpoint_func = _runtime.create_pass_through_route(
        endpoint=endpoint,
        target=target_url,
        custom_headers=custom_headers,
        _forward_headers=True,
        is_streaming_request=is_streaming_request,
        blocked_pass_through_prefixed_headers=blocked_pass_through_prefixed_headers,
    )
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )
    return received_value


# ---------------------------------------------------------------------------
# Integration seam
# ---------------------------------------------------------------------------

_OWNED_SYMBOLS = (
    "_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX",
    "_ANTHROPIC_CONTEXT_1M_BETA_HEADER",
    "_ANTHROPIC_BETA_HEADER_NAME",
    "_ANTHROPIC_BETA_XPASS_HEADER_NAME",
    "_ANTHROPIC_DANGEROUS_DIRECT_BROWSER_ACCESS_HEADER_NAME",
    "_ANTHROPIC_NATIVE_PASSTHROUGH_MODEL_ALIASES",
    "_get_header_value_case_insensitive",
    "_append_anthropic_beta_header_value",
    "_prepare_anthropic_oauth_native_passthrough_headers",
    "_normalize_anthropic_native_passthrough_model_alias",
    "_prepare_anthropic_context_1m_native_passthrough",
    "_perform_anthropic_native_passthrough_request",
    "AnthropicNativeRuntime",
)


def install(
    host_globals: dict[str, Any],
    *,
    runtime: Optional[AnthropicNativeRuntime] = None,
) -> None:
    """Publish owned symbols to the host module namespace.

    The five pure functions and constants need no rebinding.
    ``_perform_anthropic_native_passthrough_request`` requires an
    ``AnthropicNativeRuntime`` instance; pass it via *runtime* to
    activate the executor.  If *runtime* is ``None`` the executor
    is still published but will fail closed on call.
    """
    global _runtime  # noqa: PLW0603
    if runtime is not None:
        _runtime = runtime
    _mod = globals()
    for _name in _OWNED_SYMBOLS:
        host_globals[_name] = _mod[_name]
