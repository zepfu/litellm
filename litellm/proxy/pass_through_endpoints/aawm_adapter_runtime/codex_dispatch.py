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

    # Host-global functions (bound via install())
    def _resolve_codex_auto_agent_alias_model(
        request_body: dict[str, Any],
        *,
        endpoint: str,
        request: Request,
    ) -> Optional[str]: ...
    def _apply_codex_auto_agent_prevention_guidance_to_request_body(
        request_body: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]: ...
    def _apply_aawm_read_agent_guidance_to_request_body(
        request_body: dict[str, Any],
        *,
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
        canonical_alias: str,
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
    def _normalize_codex_reasoning_effort_for_resolved_route(
        request_body: dict[str, Any], *, resolved_route: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "try_dispatch_codex_request",
    "_prepare_opencode_zen_direct_tools_mode",
    "_session_affinity_mod",
    "_stash_codex_nested_owner_consult",
    "_consult_codex_nested_session_owner",
    "_ensure_codex_nested_session_owner_pre_egress",
    "_finalize_nested_session_owner_lease",
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


def _prepare_opencode_zen_direct_tools_mode(
    request: Request,
    prepared_request_body: dict[str, Any],
    *,
    direct_adapter_model: Optional[str],
) -> dict[str, Any]:
    """D1-574: pre-alias tools-mode defaulting for direct OpenCode Zen models.

    Ensures litellm_metadata.opencode_zen_unsupported_tools_mode='drop' is
    set for supported direct OpenCode Zen models before the auto-agent alias
    route.  Only applies when the request body directly selects a supported
    OpenCode Zen model (not arbitrary aliases/candidates).  Explicit body
    metadata wins; absent header and no explicit mode injects 'drop' by
    default so supported direct dispatch no longer depends on the run-scoped
    x-aawm-opencode-zen-unsupported-tools-mode header; an explicitly supplied
    invalid/empty header value raises bounded 400.
    """
    if direct_adapter_model is None:
        return prepared_request_body

    header_value = request.headers.get(
        "x-aawm-opencode-zen-unsupported-tools-mode"
    )
    mode = header_value.strip() if isinstance(header_value, str) else "drop"

    existing_metadata = prepared_request_body.get("litellm_metadata")
    existing_mode = (
        existing_metadata.get("opencode_zen_unsupported_tools_mode")
        if isinstance(existing_metadata, dict)
        else None
    )
    if existing_mode is not None:
        return prepared_request_body

    if mode != "drop":
        from litellm.proxy._types import ProxyException

        raise ProxyException(
            message=(
                "x-aawm-opencode-zen-unsupported-tools-mode must be "
                "'drop' when set."
            ),
            type="invalid_request_error",
            param="x-aawm-opencode-zen-unsupported-tools-mode",
            code=400,
        )

    prepared_request_body = dict(prepared_request_body)
    meta = dict(prepared_request_body.get("litellm_metadata") or {})
    meta["opencode_zen_unsupported_tools_mode"] = "drop"
    prepared_request_body["litellm_metadata"] = meta
    return prepared_request_body


def _session_affinity_mod():
    """Lazy session_affinity import (safe under module rebinding / tests)."""
    import sys as _sys

    _sa = _sys.modules.get(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity"
    )
    if _sa is None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as _sa,
        )
    return _sa


async def _finalize_nested_session_owner_lease(request, response=None, *, exc=None):
    """Promote on success / release on failure for nested adapter egress."""
    _sa = _session_affinity_mod()
    return await _sa.finalize_request_session_owner_lease(
        request,
        response,
        exc=exc,
        failure_phase="session_owner_codex_nested_promote",
    )


def _stash_codex_nested_owner_consult(
    request,
    *,
    session_identity,
    owner_record,
    cache_key,
) -> None:
    """Expose consult-only owner state for compatible nested route choice."""
    state = getattr(request, "state", None)
    if state is None:
        return
    _sa = _session_affinity_mod()
    setattr(state, "_aawm_session_owner_consult_identity", session_identity)
    setattr(state, "_aawm_session_owner_consult_cache_key", cache_key)
    if isinstance(owner_record, dict):
        setattr(state, "_aawm_session_owner_consult_record", owner_record)
        affinity = _sa.owner_record_as_affinity_hint(owner_record)
        if affinity:
            setattr(state, "_aawm_session_owner_consult_affinity", affinity)


async def _consult_codex_nested_session_owner(
    *,
    request,
    prepared_request_body,
) -> None:
    """Early nested ownership consult only (no generic exact compare/reserve).

    Fail closed on durable-store errors. When an owned record exists, stash
    concrete owner affinity for route choice. Exact match/reserve happens only
    after each nested route resolves concrete provider/model/route attributes.
    """

    _sa = _session_affinity_mod()
    if _sa.request_session_owner_already_guarded(request):
        return
    session_identity = _sa.resolve_canonical_session_identity(
        request, prepared_request_body
    )
    if session_identity is None:
        return
    owner_record, cache_key, error = await _sa.get_session_owner_record(
        session_identity=session_identity
    )
    if error is not None:
        _sa.raise_session_owner_redispatch_required(
            session_identity=session_identity,
            failure_phase="session_owner_codex_nested_redis",
            guard=_sa.SessionOwnerGuardResult(
                decision=_sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=session_identity,
                cache_key=cache_key,
                mismatch_reason=error,
                provenance=_sa.build_session_owner_provenance(
                    session_identity=session_identity,
                    decision="redispatch_required",
                    mismatch_reason=error,
                    cache_key=cache_key,
                ),
            ),
        )
    _stash_codex_nested_owner_consult(
        request,
        session_identity=session_identity,
        owner_record=owner_record,
        cache_key=cache_key,
    )


async def _ensure_codex_nested_session_owner_pre_egress(
    *,
    request,
    request_body,
    session_identity=None,
    provider,
    model,
    route_family,
    endpoint_contract: str = "openai_responses",
    state_format: str = "openai_responses",
    failure_phase: str = "session_owner_codex_nested_pre_egress",
) -> None:
    """Exact guard/reserve with concrete nested Codex attrs before provider send.

    Idempotent:
    - alias candidate_loop that already reserved concrete same-owner attrs is
      renewed in place (valid continuation)
    - generic ``codex_nested`` placeholders are upgraded to the concrete
      resolved identity before send
    - concrete mismatch against an existing owner raises redispatch_required
      before provider I/O
    """

    _sa = _session_affinity_mod()
    body = request_body if isinstance(request_body, dict) else {}
    existing = _sa.get_request_session_owner_lease(request)
    prior = (
        existing.attributes
        if existing is not None and isinstance(existing.attributes, dict)
        else None
    )
    prior_route = str((prior or {}).get("route_family") or "").strip().lower()
    prior_is_generic_placeholder = (not prior) or prior_route in {
        "",
        "codex_nested",
        "anthropic_nested",
    }

    account_identity = _sa.extract_account_identity_from_context(
        request=request,
        request_body=body,
    )
    concrete_attrs = _sa.build_session_owner_attributes(
        provider=provider,
        model=model,
        route_family=route_family,
        endpoint_contract=endpoint_contract,
        state_format=state_format,
        ingress="codex_nested_pre_egress",
        requested_model=body.get("model") if isinstance(body, dict) else None,
        extra=account_identity,
    )

    resolved_sid = (
        session_identity
        or (existing.session_identity if existing is not None else None)
        or _sa.resolve_canonical_session_identity(request, body)
    )

    if (
        existing is not None
        and existing.held_reservation
        and prior
        and not prior_is_generic_placeholder
    ):
        # Same-owner continuation (e.g. candidate_loop already reserved).
        await _sa.ensure_session_owner_guard_for_request(
            request=request,
            request_body=body,
            session_identity=resolved_sid,
            requested_attributes=prior,
            alias_model=str(model) if model is not None else None,
            require_exact_attributes=True,
            failure_phase=failure_phase,
        )
        return

    if existing is not None and existing.held_reservation and prior_is_generic_placeholder:
        _sa.refresh_request_session_owner_lease_attributes(request, concrete_attrs)

    await _sa.ensure_session_owner_guard_for_request(
        request=request,
        request_body=body,
        session_identity=resolved_sid,
        requested_attributes=concrete_attrs,
        alias_model=str(model) if model is not None else None,
        require_exact_attributes=True,
        failure_phase=failure_phase,
    )


async def try_dispatch_codex_request(  # noqa: PLR0915
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

    # D1-612: early nested ownership is consult-only. Do not exact-compare a
    # concrete owner against generic openai/<inbound>/codex_nested. Concrete
    # guard/reserve runs after each nested route resolves and before send.
    _sa = _session_affinity_mod()
    await _consult_codex_nested_session_owner(
        request=request,
        prepared_request_body=prepared_request_body,
    )
    _sid = _sa.resolve_canonical_session_identity(request, prepared_request_body)

    opencode_zen_adapter_model = _resolve_codex_opencode_zen_adapter_model(
        prepared_request_body,
        endpoint=endpoint,
    )
    prepared_request_body = _prepare_opencode_zen_direct_tools_mode(
        request,
        prepared_request_body,
        direct_adapter_model=opencode_zen_adapter_model,
    )

    try:
        codex_auto_agent_alias = _resolve_codex_auto_agent_alias_model(
            prepared_request_body,
            endpoint=endpoint,
            request=request,
        )
    except TypeError:
        # Host/test stubs may omit the request kwarg.
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
            canonical_alias=codex_auto_agent_alias,
        )

    if opencode_zen_adapter_model is not None:
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)

        # Concrete pre-egress reserve after opencode_zen route resolution.
        await _ensure_codex_nested_session_owner_pre_egress(
            request=request,
            request_body=prepared_request_body,
            session_identity=_sid,
            provider="opencode",
            model=opencode_zen_adapter_model,
            route_family="codex_opencode_zen_adapter",
        )
        try:
            _resp = await _handle_codex_opencode_zen_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=opencode_zen_adapter_model,
            )
        except Exception as _exc:
            await _finalize_nested_session_owner_lease(request, exc=_exc)
            raise
        await _finalize_nested_session_owner_lease(request, _resp)
        return _resp

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

        # Concrete pre-egress reserve after kimi route resolution.
        await _ensure_codex_nested_session_owner_pre_egress(
            request=request,
            request_body=prepared_request_body,
            session_identity=_sid,
            provider="kimi_code",
            model=kimi_code_adapter_model,
            route_family="codex_kimi_chat_completions_adapter",
        )
        try:
            _resp = await _handle_codex_kimi_chat_completions_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=kimi_code_adapter_model,
            )
        except Exception as _exc:
            await _finalize_nested_session_owner_lease(request, exc=_exc)
            raise
        await _finalize_nested_session_owner_lease(request, _resp)
        return _resp

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

        # Concrete pre-egress reserve after alibaba route resolution.
        await _ensure_codex_nested_session_owner_pre_egress(
            request=request,
            request_body=prepared_request_body,
            session_identity=_sid,
            provider="alibaba_token_plan",
            model=alibaba_token_plan_adapter_model,
            route_family="codex_alibaba_token_plan_chat_completions_adapter",
        )
        try:
            _resp = await _handle_codex_alibaba_token_plan_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=alibaba_token_plan_adapter_model,
            )
        except Exception as _exc:
            await _finalize_nested_session_owner_lease(request, exc=_exc)
            raise
        await _finalize_nested_session_owner_lease(request, _resp)
        return _resp

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
