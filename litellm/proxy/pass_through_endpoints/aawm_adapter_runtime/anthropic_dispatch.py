"""Wave 6F extraction: Anthropic adapter dispatch gate.

Behavior-preserving extraction of the adapter-recognition chain from
``anthropic_proxy_route`` in ``llm_passthrough_endpoints.py``.
Do not import llm_passthrough_endpoints at module scope.

The auto-agent alias route and native Anthropic passthrough (including
aawm.2/aawm.5 OAuth/audit-sensitive patches) remain in the god module
for the later integrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from fastapi import Response
    from starlette.requests import Request

    from litellm.proxy._types import UserAPIKeyAuth

Payload = dict[str, Any]

# ---------------------------------------------------------------------------
# Callable type aliases for the seam contract
# ---------------------------------------------------------------------------

ResolverFn = Callable[[Payload, str], Optional[str]]
"""(request_body, endpoint) -> Optional[adapter_model]"""

HandlerFn = Callable[..., Awaitable["Response"]]
"""Keyword-only async handler returning a FastAPI Response."""


# ---------------------------------------------------------------------------
# Runtime seam container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnthropicDispatchRuntime:
    """Host-owned callbacks required by the adapter dispatch gate.

    Every field is a callable injected by the integrator so that this
    module never imports the god module at module scope.
    """

    # -- resolvers (order matters) --
    resolve_xai_oauth: ResolverFn
    resolve_grok_native_oauth: ResolverFn
    resolve_openai_responses: ResolverFn
    resolve_opencode_zen: ResolverFn
    resolve_kimi: ResolverFn
    resolve_alibaba: ResolverFn
    resolve_nvidia: ResolverFn
    resolve_openrouter_completion: ResolverFn
    resolve_openrouter_responses: ResolverFn

    # -- handlers --
    handle_xai_oauth_responses: HandlerFn
    handle_xai_oauth_completion: HandlerFn
    handle_grok_native_oauth_responses: HandlerFn
    handle_openai_responses: HandlerFn
    handle_opencode_zen: HandlerFn
    handle_kimi: HandlerFn
    handle_alibaba: HandlerFn
    handle_nvidia: HandlerFn
    handle_openrouter_completion: HandlerFn
    handle_openrouter_responses: HandlerFn

    # -- classification helper --
    is_oa_xai_responses_model: Callable[[Any], bool]



# ---------------------------------------------------------------------------
# Seam disposition (for test parity and documentation)
# ---------------------------------------------------------------------------

ANTHROPIC_DISPATCH_SEAM_DISPOSITION: dict[str, str] = {
    "resolve_xai_oauth": "runtime.resolve_xai_oauth",
    "resolve_grok_native_oauth": "runtime.resolve_grok_native_oauth",
    "resolve_openai_responses": "runtime.resolve_openai_responses",
    "resolve_opencode_zen": "runtime.resolve_opencode_zen",
    "resolve_kimi": "runtime.resolve_kimi",
    "resolve_alibaba": "runtime.resolve_alibaba",
    "resolve_nvidia": "runtime.resolve_nvidia",
    "resolve_openrouter_completion": "runtime.resolve_openrouter_completion",
    "resolve_openrouter_responses": "runtime.resolve_openrouter_responses",
    "handle_xai_oauth_responses": "runtime.handle_xai_oauth_responses",
    "handle_xai_oauth_completion": "runtime.handle_xai_oauth_completion",
    "handle_grok_native_oauth_responses": "runtime.handle_grok_native_oauth_responses",
    "handle_openai_responses": "runtime.handle_openai_responses",
    "handle_opencode_zen": "runtime.handle_opencode_zen",
    "handle_kimi": "runtime.handle_kimi",
    "handle_alibaba": "runtime.handle_alibaba",
    "handle_nvidia": "runtime.handle_nvidia",
    "handle_openrouter_completion": "runtime.handle_openrouter_completion",
    "handle_openrouter_responses": "runtime.handle_openrouter_responses",
    "is_oa_xai_responses_model": "runtime.is_oa_xai_responses_model",
}


# ---------------------------------------------------------------------------
# Dispatch gate
# ---------------------------------------------------------------------------


async def _dispatch_anthropic_adapter_with_lease(request, handler_coro):
    """Await one nested adapter handler and finalize the shared lease."""
    try:
        response = await handler_coro
    except Exception as exc:
        await _finalize_nested_session_owner_lease(request, exc=exc)
        raise
    await _finalize_nested_session_owner_lease(request, response)
    return response


async def _finalize_nested_session_owner_lease(request, response=None, *, exc=None):
    """Promote/release via the shared session_affinity lifecycle API."""
    import sys as _sys

    _sa = _sys.modules.get(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity"
    )
    if _sa is None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as _sa,
        )
    return await _sa.finalize_request_session_owner_lease(
        request,
        response,
        exc=exc,
        failure_phase="session_owner_anthropic_nested_promote",
    )


async def try_dispatch_anthropic_adapter(
    runtime: AnthropicDispatchRuntime,
    *,
    endpoint: str,
    request: "Request",
    fastapi_response: "Response",
    user_api_key_dict: "UserAPIKeyAuth",
    prepared_request_body: Payload,
) -> Optional["Response"]:
    """Walk the adapter recognition chain in priority order.

    Returns the adapter ``Response`` when a route is selected, or ``None``
    when no adapter applies and the caller should fall through to native
    Anthropic passthrough.

    Ordering and failure behavior exactly mirror the inline chain in
    ``anthropic_proxy_route``: each resolver is tried in sequence; the
    first non-None result dispatches to its handler.  Exceptions from
    resolvers or handlers propagate unmodified.

    D1-612: early nested ownership is consult-only. Concrete
    provider/model/route reservation happens at the last common pre-egress
    points inside the adapter perform helpers once the selected adapter
    identity is known, so promotion cannot pin the generic
    ``anthropic/<inbound>/anthropic_nested`` placeholder.
    """
    import sys as _sys

    _sa = _sys.modules.get(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity"
    )
    if _sa is None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as _sa,
        )
    # Consult-only: fail closed on Redis/store errors before adapter selection.
    # Do not reserve or promote with generic inbound/nested placeholder attrs.
    if not _sa.request_session_owner_already_guarded(request):
        _sid = _sa.resolve_canonical_session_identity(request, prepared_request_body)
        if _sid is not None:
            _, _ck, _err = await _sa.get_session_owner_record(
                session_identity=_sid
            )
            if _err is not None:
                _sa.raise_session_owner_redispatch_required(
                    session_identity=_sid,
                    failure_phase="session_owner_anthropic_nested_redis",
                    guard=_sa.SessionOwnerGuardResult(
                        decision=_sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                        session_identity=_sid,
                        cache_key=_ck,
                        mismatch_reason=_err,
                        provenance=_sa.build_session_owner_provenance(
                            session_identity=_sid,
                            decision="redispatch_required",
                            mismatch_reason=_err,
                            cache_key=_ck,
                        ),
                    ),
                    request=request,
                )

    # Common handler kwargs shared across all adapter routes.
    common: dict[str, Any] = {
        "endpoint": endpoint,
        "request": request,
        "fastapi_response": fastapi_response,
        "user_api_key_dict": user_api_key_dict,
        "prepared_request_body": prepared_request_body,
    }

    # 1. xAI OAuth (responses vs completion split)
    xai_oauth_adapter_model = runtime.resolve_xai_oauth(
        prepared_request_body, endpoint,
    )
    if xai_oauth_adapter_model is not None:
        if runtime.is_oa_xai_responses_model(xai_oauth_adapter_model):
            return await _dispatch_anthropic_adapter_with_lease(
                request,
                runtime.handle_xai_oauth_responses(
                    **common, adapter_model=xai_oauth_adapter_model,
                ),
            )
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_xai_oauth_completion(
                **common, adapter_model=xai_oauth_adapter_model,
            ),
        )

    # 2. Grok native OAuth
    grok_native_oauth_adapter_model = runtime.resolve_grok_native_oauth(
        prepared_request_body, endpoint,
    )
    if grok_native_oauth_adapter_model is not None:
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_grok_native_oauth_responses(
                **common, adapter_model=grok_native_oauth_adapter_model,
            ),
        )

    # 3. OpenAI responses adapter
    adapter_model = runtime.resolve_openai_responses(
        prepared_request_body, endpoint,
    )
    if adapter_model is not None:
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_openai_responses(
                **common, adapter_model=adapter_model,
            ),
        )

    # 4. OpenCode Zen
    opencode_zen_adapter_model = runtime.resolve_opencode_zen(
        prepared_request_body, endpoint,
    )
    if opencode_zen_adapter_model is not None:
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_opencode_zen(
                **common, adapter_model=opencode_zen_adapter_model,
            ),
        )

    # 5. Kimi code chat completions
    kimi_code_adapter_model = runtime.resolve_kimi(
        prepared_request_body, endpoint,
    )
    if kimi_code_adapter_model is not None:
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_kimi(
                **common, adapter_model=kimi_code_adapter_model,
            ),
        )

    # 6. Alibaba token plan
    alibaba_token_plan_adapter_model = runtime.resolve_alibaba(
        prepared_request_body, endpoint,
    )
    if alibaba_token_plan_adapter_model is not None:
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_alibaba(
                **common, adapter_model=alibaba_token_plan_adapter_model,
            ),
        )

    # 7. NVIDIA completion adapter
    nvidia_adapter_model = runtime.resolve_nvidia(
        prepared_request_body, endpoint,
    )
    if nvidia_adapter_model is not None:
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_nvidia(
                **common, adapter_model=nvidia_adapter_model,
            ),
        )

    # 8. OpenRouter completion adapter
    openrouter_completion_adapter_model = runtime.resolve_openrouter_completion(
        prepared_request_body, endpoint,
    )
    if openrouter_completion_adapter_model is not None:
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_openrouter_completion(
                **common, adapter_model=openrouter_completion_adapter_model,
            ),
        )

    # 9. OpenRouter responses adapter
    openrouter_adapter_model = runtime.resolve_openrouter_responses(
        prepared_request_body, endpoint,
    )
    if openrouter_adapter_model is not None:
        return await _dispatch_anthropic_adapter_with_lease(
            request,
            runtime.handle_openrouter_responses(
                **common, adapter_model=openrouter_adapter_model,
            ),
        )

    return None
