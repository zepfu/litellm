"""Wave 6F extraction: Codex dispatch gate for the OpenAI pass-through handler.

Behavior-preserving extraction of the ``is_codex_responses_request`` dispatch
cascade from ``BaseOpenAIPassThroughHandler._base_openai_pass_through_handler``
in ``llm_passthrough_endpoints.py``.

Do not import ``llm_passthrough_endpoints`` at module scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from fastapi import Request, Response

from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:

    # Host-global constants
    _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER: str

    # Host-global functions (bound via install())
    def _resolve_codex_auto_agent_alias_model(
        request_body: dict[str, Any], *, endpoint: str
    ) -> Optional[str]: ...
    def _apply_codex_auto_agent_prevention_guidance_to_request_body(
        request_body: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]: ...
    def _apply_aawm_read_agent_guidance_to_request_body(
        request_body: dict[str, Any],
        *,
        alias_model: str,
        target_field: str,
    ) -> tuple[dict[str, Any], Any]: ...
    def _prepare_request_body_for_passthrough_observability(
        *, request: Request, request_body: dict[str, Any]
    ) -> dict[str, Any]: ...
    def _safe_set_request_parsed_body(
        request: Request, body: dict[str, Any]
    ) -> None: ...
    async def _handle_codex_auto_agent_alias_route(
        *,
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        prepared_request_body: dict[str, Any],
        target_url: str,
        api_key: Optional[str],
        forward_headers: bool,
    ) -> Response: ...
    def _resolve_codex_opencode_zen_adapter_model(
        request_body: dict[str, Any], *, endpoint: str
    ) -> Optional[str]: ...
    async def _handle_codex_opencode_zen_adapter_route(
        *,
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        prepared_request_body: dict[str, Any],
        adapter_model: str,
    ) -> Response: ...
    def _resolve_codex_kimi_chat_completions_adapter_model(
        request_body: dict[str, Any], *, endpoint: str
    ) -> Optional[str]: ...
    async def _handle_codex_kimi_chat_completions_adapter_route(
        *,
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        prepared_request_body: dict[str, Any],
        adapter_model: str,
    ) -> Response: ...
    def _resolve_codex_alibaba_token_plan_adapter_model(
        request_body: dict[str, Any], *, endpoint: str
    ) -> Optional[str]: ...
    async def _handle_codex_alibaba_token_plan_adapter_route(
        *,
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        prepared_request_body: dict[str, Any],
        adapter_model: str,
    ) -> Response: ...
    def _resolve_codex_antigravity_code_assist_adapter_model(
        request_body: dict[str, Any], *, endpoint: str
    ) -> Optional[str]: ...
    def _resolve_codex_google_code_assist_adapter_model(
        request_body: dict[str, Any], *, endpoint: str
    ) -> Optional[str]: ...
    async def _handle_codex_google_code_assist_adapter_route(
        *,
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        prepared_request_body: dict[str, Any],
        adapter_model: str,
        adapter_provider: str = ...,
    ) -> Response: ...
    def _normalize_codex_reasoning_effort_for_resolved_route(
        request_body: dict[str, Any], *, resolved_route: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "try_dispatch_codex_request",
)


def install(
    host_globals: dict[str, Any],
    *,
    publish_to_module: bool = False,
) -> None:
    """Rebind moved functions to *host_globals* for live lookup.

    Each named function's ``__globals__`` is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility. Production
    installation may also publish the rebound object to this module; secondary
    hosts receive isolated rebound copies without replacing the canonical
    production facade.
    """
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        if not isinstance(_obj, FunctionType):
            host_globals[_name] = _obj
            continue
        _rebound = FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        if publish_to_module:
            _mod[_name] = _rebound
        host_globals[_name] = _rebound


# ── Extracted function ──────────────────────────────────────────────


async def try_dispatch_codex_request(
    *,
    endpoint: str,
    request: Request,
    request_body: dict[str, Any],
    prepared_request_body: dict[str, Any],
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    target_url: str,
    api_key: Optional[str],
    forward_headers: bool,
) -> Optional[Response]:
    """Attempt to dispatch a Codex responses request to an adapter route.

    Returns the dispatched ``Response`` when a supported Codex/AAWM adapter
    path matches, or ``None`` when no adapter matched so the caller can fall
    through to the default pass-through path.

    This is a behavior-preserving extraction of the ``elif
    is_codex_responses_request:`` block inside
    ``BaseOpenAIPassThroughHandler._base_openai_pass_through_handler``.
    """
    import litellm

    codex_auto_agent_alias = _resolve_codex_auto_agent_alias_model(
        prepared_request_body,
        endpoint=endpoint,
    )
    if codex_auto_agent_alias is not None:
        (
            prepared_request_body,
            _codex_auto_agent_guidance_changes,
        ) = _apply_codex_auto_agent_prevention_guidance_to_request_body(prepared_request_body)
        (
            prepared_request_body,
            _codex_read_guidance_changes,
        ) = _apply_aawm_read_agent_guidance_to_request_body(
            prepared_request_body,
            alias_model=codex_auto_agent_alias,
            target_field="instructions",
        )
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)
        return await _handle_codex_auto_agent_alias_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=prepared_request_body,
            target_url=target_url,
            api_key=api_key,
            forward_headers=forward_headers,
        )

    opencode_zen_adapter_model = _resolve_codex_opencode_zen_adapter_model(
        prepared_request_body,
        endpoint=endpoint,
    )
    if opencode_zen_adapter_model is not None:
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)
        return await _handle_codex_opencode_zen_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=prepared_request_body,
            adapter_model=opencode_zen_adapter_model,
        )

    kimi_code_adapter_model = _resolve_codex_kimi_chat_completions_adapter_model(
        prepared_request_body,
        endpoint=endpoint,
    )
    if kimi_code_adapter_model is not None:
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)
        return await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=prepared_request_body,
            adapter_model=kimi_code_adapter_model,
        )

    alibaba_token_plan_adapter_model = _resolve_codex_alibaba_token_plan_adapter_model(
        prepared_request_body,
        endpoint=endpoint,
    )
    if alibaba_token_plan_adapter_model is not None:
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)
        return await _handle_codex_alibaba_token_plan_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=prepared_request_body,
            adapter_model=alibaba_token_plan_adapter_model,
        )

    antigravity_adapter_model = _resolve_codex_antigravity_code_assist_adapter_model(
        prepared_request_body,
        endpoint=endpoint,
    )
    if antigravity_adapter_model is not None:
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)
        return await _handle_codex_google_code_assist_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=prepared_request_body,
            adapter_model=antigravity_adapter_model,
            adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,  # noqa: F821
        )

    google_adapter_model = _resolve_codex_google_code_assist_adapter_model(
        prepared_request_body,
        endpoint=endpoint,
    )
    if google_adapter_model is not None:
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)
        return await _handle_codex_google_code_assist_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=prepared_request_body,
            adapter_model=google_adapter_model,
        )

    # No adapter matched -- apply direct-model reasoning effort normalization
    # (side-effect on prepared_request_body) and fall through.
    direct_model = prepared_request_body.get("model")
    if isinstance(direct_model, str) and direct_model:
        (
            normalized_request_body,
            _direct_reasoning_effort_metadata,
        ) = _normalize_codex_reasoning_effort_for_resolved_route(
            prepared_request_body,
            resolved_route={
                "provider": litellm.LlmProviders.OPENAI.value,
                "model": direct_model,
                "route_family": "codex_responses",
            },
        )
        if normalized_request_body is not prepared_request_body:
            prepared_request_body.clear()
            prepared_request_body.update(normalized_request_body)

    return None
