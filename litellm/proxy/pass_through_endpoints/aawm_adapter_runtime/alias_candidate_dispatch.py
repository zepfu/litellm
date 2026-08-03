"""Wave 7 extraction: alias candidate dispatch owner.

Behavior-preserving extraction of:
- ``_dispatch_auto_agent_alias_candidate_request``
- ``_perform_anthropic_auto_agent_alias_candidate_request``

from ``llm_passthrough_endpoints.py`` (lines ~6818-7024).

Omits removed Google Code Assist and Antigravity candidates.
Do not import llm_passthrough_endpoints at module scope.

Owned symbols:
- ``_dispatch_auto_agent_alias_candidate_request``
- ``_perform_anthropic_auto_agent_alias_candidate_request``
- ``AliasCandidateDispatchRuntime``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Mapping,
    Optional,
)

if TYPE_CHECKING:
    from fastapi import Response
    from starlette.requests import Request

    from litellm.proxy._types import UserAPIKeyAuth

Payload = dict[str, Any]

# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------

AdapterHandlerFn = Callable[..., Awaitable["Response"]]
"""Keyword-only async adapter handler returning a FastAPI Response."""

NormalizeModelAliasFn = Callable[[Payload], tuple[Payload, Any]]
"""(candidate_body) -> (normalized_body, normalized_alias)"""

PrepareContext1mFn = Callable[..., tuple[Payload, dict[str, Any], Any]]
"""(request, request_body, custom_headers) -> (body, headers, normalized_model)"""

SafeSetBodyFn = Callable[["Request", Payload], None]
"""(request, body) -> None; mutates request parsed body."""

NativePassthroughFn = Callable[..., Awaitable["Response"]]
"""Keyword-only async native passthrough executor."""


# ---------------------------------------------------------------------------
# Runtime seam container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasCandidateDispatchRuntime:
    """Host-owned callbacks required by the alias candidate dispatch.

    Every field is a callable injected by the integrator so that this
    module never imports the god module at module scope.
    """

    # -- adapter handler callbacks (uniform keyword signature) --
    handle_openai_responses: AdapterHandlerFn
    """Codex/OpenAI responses adapter route handler."""

    handle_openrouter_completion: AdapterHandlerFn
    """OpenRouter completion adapter route handler."""

    handle_openrouter_responses: AdapterHandlerFn
    """OpenRouter responses adapter route handler."""

    handle_xai_oauth_responses: AdapterHandlerFn
    """xAI OAuth responses adapter route handler."""

    handle_grok_native_oauth_responses: AdapterHandlerFn
    """Grok native OAuth responses adapter route handler."""

    handle_opencode_zen: AdapterHandlerFn
    """OpenCode Zen adapter route handler."""

    handle_kimi_chat_completions: AdapterHandlerFn
    """Kimi chat completions adapter route handler."""

    handle_alibaba_token_plan: AdapterHandlerFn
    """Alibaba token plan adapter route handler."""

    # -- native Anthropic passthrough helpers --
    normalize_native_model_alias: NormalizeModelAliasFn
    """_normalize_anthropic_native_passthrough_model_alias"""

    prepare_context_1m_native: PrepareContext1mFn
    """_prepare_anthropic_context_1m_native_passthrough"""

    safe_set_request_parsed_body: SafeSetBodyFn
    """_safe_set_request_parsed_body"""

    perform_native_passthrough: NativePassthroughFn
    """_perform_anthropic_native_passthrough_request"""

    # -- constants --
    provider_native: str
    """_CODEX_AUTO_AGENT_NATIVE_PROVIDER"""

    provider_openrouter: str
    """_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER"""

    provider_xai: str
    """_CODEX_AUTO_AGENT_XAI_PROVIDER"""

    provider_opencode: str
    """_CODEX_AUTO_AGENT_OPENCODE_PROVIDER"""

    provider_kimi: str
    """_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER"""

    provider_alibaba: str
    """_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER"""

    anthropic_beta_header_name: str
    """_ANTHROPIC_BETA_HEADER_NAME"""


# ---------------------------------------------------------------------------
# Seam disposition (for test parity and documentation)
# ---------------------------------------------------------------------------

ALIAS_CANDIDATE_DISPATCH_SEAM_DISPOSITION: dict[str, str] = {
    "handle_openai_responses": "runtime.handle_openai_responses",
    "handle_openrouter_completion": "runtime.handle_openrouter_completion",
    "handle_openrouter_responses": "runtime.handle_openrouter_responses",
    "handle_xai_oauth_responses": "runtime.handle_xai_oauth_responses",
    "handle_grok_native_oauth_responses": "runtime.handle_grok_native_oauth_responses",
    "handle_opencode_zen": "runtime.handle_opencode_zen",
    "handle_kimi_chat_completions": "runtime.handle_kimi_chat_completions",
    "handle_alibaba_token_plan": "runtime.handle_alibaba_token_plan",
    "normalize_native_model_alias": "runtime.normalize_native_model_alias",
    "prepare_context_1m_native": "runtime.prepare_context_1m_native",
    "safe_set_request_parsed_body": "runtime.safe_set_request_parsed_body",
    "perform_native_passthrough": "runtime.perform_native_passthrough",
    "provider_native": "runtime.provider_native",
    "provider_openrouter": "runtime.provider_openrouter",
    "provider_xai": "runtime.provider_xai",
    "provider_opencode": "runtime.provider_opencode",
    "provider_kimi": "runtime.provider_kimi",
    "provider_alibaba": "runtime.provider_alibaba",
    "anthropic_beta_header_name": "runtime.anthropic_beta_header_name",
}


# ---------------------------------------------------------------------------
# Module-level runtime slot
# ---------------------------------------------------------------------------

_runtime: Optional[AliasCandidateDispatchRuntime] = None


# ---------------------------------------------------------------------------
# Owner functions
# ---------------------------------------------------------------------------


async def _dispatch_auto_agent_alias_candidate_request(
    *,
    candidate: Payload,
    provider_handlers: Mapping[str, Callable[[], Awaitable["Response"]]],
    default_handler: Callable[[], Awaitable["Response"]],
    route_family_handlers: Optional[
        Mapping[str, Mapping[str, Callable[[], Awaitable["Response"]]]]
    ] = None,
) -> "Response":
    """Table-driven provider/route_family candidate dispatch (RR-054 #10).

    Anthropic and Codex families keep different handler callables, but share one
    dispatch shape so provider branching does not re-grow divergent control flow.
    """
    provider = str(candidate.get("provider") or "")
    route_family = str(candidate.get("route_family") or "")
    if route_family_handlers and provider in route_family_handlers:
        family_map = route_family_handlers[provider]
        handler = family_map.get(route_family) or family_map.get("*")
        if handler is not None:
            return await handler()
    handler = provider_handlers.get(provider)
    if handler is not None:
        return await handler()
    return await default_handler()


async def _perform_anthropic_auto_agent_alias_candidate_request(
    *,
    endpoint: str,
    request: "Request",
    fastapi_response: "Response",
    user_api_key_dict: "UserAPIKeyAuth",
    candidate: dict[str, Any],
    candidate_body: dict[str, Any],
    target_url: str,
    custom_headers: dict[str, Any],
) -> "Response":
    """Build per-provider closures and dispatch via the table-driven selector.

    Fails closed if the runtime has not been installed.  Retained providers:
    Codex/OpenAI, OpenRouter (completion + responses), xAI/Grok (OAuth + native),
    OpenCode, Kimi, Alibaba, and native Anthropic passthrough.

    Removed: Google Code Assist and Antigravity candidates.
    """
    if _runtime is None:
        raise RuntimeError(
            "alias_candidate_dispatch runtime not installed; "
            "call install() with AliasCandidateDispatchRuntime before use"
        )
    rt = _runtime
    adapter_model = candidate["model"]

    async def _openai() -> "Response":
        return await rt.handle_openai_responses(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _openrouter_completion() -> "Response":
        return await rt.handle_openrouter_completion(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _openrouter_responses() -> "Response":
        return await rt.handle_openrouter_responses(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _xai_oauth() -> "Response":
        return await rt.handle_xai_oauth_responses(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _grok_native() -> "Response":
        return await rt.handle_grok_native_oauth_responses(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _opencode() -> "Response":
        return await rt.handle_opencode_zen(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _kimi_code() -> "Response":
        return await rt.handle_kimi_chat_completions(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _alibaba_token_plan() -> "Response":
        return await rt.handle_alibaba_token_plan(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _native() -> "Response":
        native_candidate_body = dict(candidate_body)
        native_custom_headers = custom_headers
        blocked_pass_through_prefixed_headers: Optional[list[str]] = None
        (
            native_candidate_body,
            _normalized_native_model_alias,
        ) = rt.normalize_native_model_alias(native_candidate_body)
        raw_native_effort = native_candidate_body.pop("reasoning_effort", None)
        if isinstance(raw_native_effort, str) and raw_native_effort:
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig
            from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
                normalize_reasoning_effort_for_provider,
            )

            native_model = str(native_candidate_body.get("model") or adapter_model)
            normalized_native_effort = normalize_reasoning_effort_for_provider(
                reasoning_effort=raw_native_effort,
                model=native_model,
                custom_llm_provider="anthropic",
                native_provider="anthropic",
            )
            if normalized_native_effort and normalized_native_effort.native_value:
                mapped_native_thinking = AnthropicConfig._map_reasoning_effort(
                    normalized_native_effort.native_value,
                    native_model,
                )
                if mapped_native_thinking:
                    native_candidate_body["thinking"] = mapped_native_thinking
        (
            native_candidate_body,
            native_custom_headers,
            normalized_context_1m_model,
        ) = rt.prepare_context_1m_native(
            request=request,
            request_body=native_candidate_body,
            custom_headers=native_custom_headers,
        )
        if normalized_context_1m_model:
            blocked_pass_through_prefixed_headers = [rt.anthropic_beta_header_name]
        rt.safe_set_request_parsed_body(request, native_candidate_body)
        return await rt.perform_native_passthrough(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            target_url=target_url,
            custom_headers=native_custom_headers,
            blocked_pass_through_prefixed_headers=blocked_pass_through_prefixed_headers,
        )

    return await _dispatch_auto_agent_alias_candidate_request(
        candidate=candidate,
        provider_handlers={
            rt.provider_native: _openai,
            rt.provider_opencode: _opencode,
            rt.provider_kimi: _kimi_code,
            rt.provider_alibaba: _alibaba_token_plan,
        },
        route_family_handlers={
            rt.provider_openrouter: {
                "anthropic_openrouter_completion_adapter": _openrouter_completion,
                "*": _openrouter_responses,
            },
            rt.provider_xai: {
                "anthropic_xai_oauth_responses_adapter": _xai_oauth,
                "*": _grok_native,
            },
        },
        default_handler=_native,
    )


# ---------------------------------------------------------------------------
# Integration seam
# ---------------------------------------------------------------------------

_OWNED_SYMBOLS = (
    "_dispatch_auto_agent_alias_candidate_request",
    "_perform_anthropic_auto_agent_alias_candidate_request",
    "AliasCandidateDispatchRuntime",
)


def install(
    host_globals: dict[str, Any],
    *,
    runtime: Optional[AliasCandidateDispatchRuntime] = None,
) -> None:
    """Publish owned symbols to the host module namespace.

    ``_perform_anthropic_auto_agent_alias_candidate_request`` requires an
    ``AliasCandidateDispatchRuntime`` instance; pass it via *runtime* to
    activate the executor.  If *runtime* is ``None`` the function is still
    published but will fail closed on call.
    """
    global _runtime  # noqa: PLW0603
    if runtime is not None:
        _runtime = runtime
    _mod = globals()
    for _name in _OWNED_SYMBOLS:
        host_globals[_name] = _mod[_name]
