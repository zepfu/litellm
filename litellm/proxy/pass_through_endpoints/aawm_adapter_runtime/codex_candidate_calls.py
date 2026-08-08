"""Wave 6F extraction: Codex auto-agent provider candidate request functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations
# ruff: noqa: F821 - free names resolve via host globals after install() rebind

import json
from types import FunctionType
from typing import TYPE_CHECKING, Any, Optional, Union, cast

if TYPE_CHECKING:
    import httpx
    import litellm as litellm
    from fastapi import HTTPException
    from fastapi.responses import Response, StreamingResponse
    from starlette.requests import Request

    from litellm.llms.alibaba_token_plan.adapters import (
        adapter as _alibaba_token_plan_adapters,
    )
    from litellm.llms.kimi_code.adapters import adapter as _kimi_code_adapters
    from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

    from ..aawm_alias_routing import adapter_config as _aawm_adapter_config
    from ..aawm_alias_routing import adapter_driver as _aawm_adapter_driver
    from ..aawm_alias_routing import streaming as _aawm_alias_streaming
    from ..aawm_alias_routing.types import Payload

    _anthropic_opencode_zen_normalization: Any

    # Host-global classes / helpers (bound via install())
    class BaseOpenAIPassThroughHandler:
        @staticmethod
        def _assemble_headers(**kwargs: Any) -> dict[str, Any]: ...
        @staticmethod
        def _normalize_endpoint_for_target(**kwargs: Any) -> str: ...
        @staticmethod
        def _join_url_paths(*args: Any) -> Any: ...
        @staticmethod
        async def _prepare_openai_grok_native_oauth_context(**kwargs: Any) -> Any: ...
        @staticmethod
        async def _prepare_openai_oa_xai_context(**kwargs: Any) -> Any: ...

    class HttpPassThroughEndpointHelpers:
        @staticmethod
        def validate_outgoing_egress(**kwargs: Any) -> None: ...

    class ProxyException(Exception):
        message: str
        def __init__(self, *, message: str, type: str, param: str, code: int) -> None: ...

    async def pass_through_request(**kwargs: Any) -> Response: ...

    # Host-global constants
    _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES: list[int]
    _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT: list[int]
    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES: int
    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS: int
    _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER: str
    _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER: str
    _CODEX_AUTO_AGENT_OPENCODE_PROVIDER: str
    _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER: str
    _CODEX_AUTO_AGENT_XAI_PROVIDER: str

    # Host-global functions
    def _adapt_codex_custom_tools_to_functions_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _adapt_codex_namespace_tools_to_functions_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _add_route_family_logging_metadata(body: Any, family: str) -> Any: ...
    def _annotate_request_scope_for_adapted_access_log(request: Request, url: Any) -> None: ...
    def _apply_codex_tool_description_patches_to_request_body(body: Any) -> tuple[Any, Any]: ...
    def _apply_openrouter_completion_message_sanitization(**kwargs: Any) -> tuple[Any, Any, Any]: ...
    def _build_adapted_route_rollup_kwargs(metadata: Any) -> dict[str, Any]: ...
    def _build_langfuse_span_descriptor(**kwargs: Any) -> Any: ...
    def _build_malformed_tool_call_intake_context(*args: Any, **kwargs: Any) -> Any: ...
    def _build_openrouter_default_headers() -> dict[str, str]: ...
    def _build_responses_response_from_adapter_response(response_obj: Any) -> Response: ...
    def _codex_native_openai_candidate_unavailable_detail(exc: Any) -> Optional[str]: ...
    async def _collect_responses_response_from_stream(response: Any, **kwargs: Any) -> dict[str, Any]: ...
    def _decode_http_response_body(body: Any) -> str: ...
    async def _dispatch_auto_agent_alias_candidate_request(**kwargs: Any) -> Response: ...
    def _drop_tool_choice_without_tools_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _drop_unsupported_codex_hosted_tools_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _drop_unsupported_codex_input_items_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _emit_adapted_route_access_log(**kwargs: Any) -> None: ...
    def _get_anthropic_opencode_zen_normalization_runtime() -> Any: ...
    def _get_opencode_zen_target_base() -> str: ...
    def _get_openrouter_api_key() -> Optional[str]: ...
    def _get_openrouter_completion_adapter_upstream_model(model: str) -> Optional[str]: ...
    def _get_openrouter_target_base() -> str: ...
    def _get_proxy_shared_aiohttp_session() -> Optional[Any]: ...
    def _grok_native_candidate_unavailable_detail(exc: Exception) -> Optional[str]: ...
    def _is_codex_auto_agent_empty_success_responses_body(body: Any) -> bool: ...
    def _is_codex_auto_agent_malformed_tool_call_text_output(body: Any) -> bool: ...
    def _is_failed_responses_body(body: Any) -> bool: ...
    def _join_opencode_zen_passthrough_url(base_target_url: str, endpoint: str) -> str: ...
    async def _load_opencode_zen_api_key_for_candidate(**kwargs: Any) -> str: ...
    def _merge_litellm_metadata(body: Any, **kwargs: Any) -> Any: ...
    def _opencode_zen_candidate_unavailable_detail(exc: Exception) -> Optional[str]: ...
    async def _perform_openrouter_adapter_pass_through_request(**kwargs: Any) -> Any: ...
    async def _perform_openrouter_completion_adapter_operation(**kwargs: Any) -> Any: ...
    def _classify_codex_auto_agent_retryable_exhaustion(exc: Any) -> Optional[str]: ...
    def _extract_adapter_upstream_headers(exc: Any) -> dict[str, Any]: ...
    def _parse_retry_after_seconds_from_headers(headers: dict[str, Any]) -> Optional[float]: ...
    def _raise_codex_auto_agent_empty_success_response(**kwargs: Any) -> Any: ...
    def _raise_codex_auto_agent_failed_responses_payload(**kwargs: Any) -> Any: ...
    def _raise_codex_auto_agent_malformed_tool_call_text_payload(**kwargs: Any) -> Any: ...
    def _raise_codex_native_openai_auto_agent_candidate_unavailable(exc: Exception) -> Any: ...
    def _raise_grok_native_auto_agent_candidate_unavailable(exc: Exception) -> Any: ...
    def _raise_opencode_zen_auto_agent_candidate_unavailable(exc: Exception) -> Any: ...
    def _raise_xai_oauth_auto_agent_candidate_unavailable(exc: Exception) -> Any: ...
    def _record_adapted_completed_route_rollup_after_stream(response: Any, rollup: Any, **kwargs: Any) -> Any: ...
    def _record_adapted_completed_route_rollup_turn(rollup: Any, **kwargs: Any) -> None: ...
    def _responses_sse_from_iterator(iterator: Any, **kwargs: Any) -> Any: ...
    def _responses_sse_from_repaired_response_body(response_body: dict[str, Any]) -> Any: ...
    def _serialize_responses_adapter_response(response_obj: Any) -> str: ...
    async def _validate_codex_auto_agent_responses_payload(response: Any, **kwargs: Any) -> Any: ...
    def _xai_oauth_candidate_unavailable_detail(exc: Exception) -> Optional[str]: ...


# ── Host-global function names (bound via install()) ────────────────

_HOST_FUNCTION_NAMES = (
    # Top-level candidate dispatcher
    "_perform_codex_auto_agent_alias_candidate_request",
    # Provider-specific candidate requests
    "_perform_codex_auto_agent_native_openai_request",
    "_perform_codex_auto_agent_grok_native_responses_request",
    "_perform_codex_auto_agent_oa_xai_responses_request",
    "_validate_codex_auto_agent_openrouter_responses_stream",
    "_perform_codex_auto_agent_openrouter_responses_request",
    "_perform_codex_auto_agent_openrouter_completion_request",
    # Kimi
    "_prepare_codex_kimi_chat_completions_adapter_route",
    "_perform_codex_kimi_chat_completions_adapter_call",
    "_handle_codex_kimi_chat_completions_adapter_route",
    # Alibaba
    "_prepare_codex_alibaba_token_plan_adapter_route",
    "_perform_codex_alibaba_token_plan_adapter_call",
    "_handle_codex_alibaba_token_plan_adapter_route",
    # OpenCode
    "_handle_codex_opencode_zen_adapter_route",
    "_consume_opencode_zen_tools_mode_header",
    "_build_opencode_zen_completion_call_kwargs",
    "_perform_opencode_zen_completion_call",
    "_prepare_opencode_zen_direct_observability_metadata",
    "_prepare_opencode_zen_known_free_logging",
    "_opencode_zen_callback_headers",
    # D1-574 OpenCode direct 429
    "_opencode_zen_direct_safe_retry_after",
    "_maybe_raise_opencode_zen_direct_rate_limit",
    "_opencode_zen_direct_stream_terminal_error",
    "_OPENCODE_ZEN_DIRECT_429_ERROR_CLASSES",
    "_OPENCODE_ZEN_DIRECT_RETRY_AFTER_CEILING_SECONDS",
    "_OPENCODE_ZEN_DIRECT_PEEK_MAX_BYTES",
    # CFG-004 encrypted reasoning detection
    "_is_fernet_encrypted_token",
    "_responses_output_contains_encrypted_reasoning_arguments",
    "_FERNET_TOKEN_PREFIX",
    "_FERNET_MIN_TOKEN_LENGTH",
    "_ALIBABA_ENCRYPTED_REASONING_MAX_RETRIES",
)


def install(
    host_globals: dict[str, Any],
    *,
    publish_to_module: bool = False,
) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
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


# ── Extracted functions ─────────────────────────────────────────────


# ── CFG-004: encrypted reasoning detection ─────────────────────────

_FERNET_TOKEN_PREFIX = "gAAAA"
_FERNET_MIN_TOKEN_LENGTH = 64
_ALIBABA_ENCRYPTED_REASONING_MAX_RETRIES = 1


def _is_fernet_encrypted_token(value: str) -> bool:
    """Detect Fernet-encrypted reasoning tokens by their version prefix.

    Fernet tokens are base64url-encoded and begin with a fixed version
    byte (0x80) that encodes to ``gAAAA...`` in base64.  Legitimate tool
    call argument values do not start with this prefix at this length.
    """
    stripped = value.strip()
    return (
        len(stripped) >= _FERNET_MIN_TOKEN_LENGTH
        and stripped.startswith(_FERNET_TOKEN_PREFIX)
    )


def _responses_output_contains_encrypted_reasoning_arguments(
    responses_api_response: Any,
) -> list[dict[str, Any]]:
    """Detect Fernet-encrypted reasoning tokens in function_call arguments.

    Upstream chat-completion models may leak encrypted reasoning content
    into tool call argument values.  Returns a list of diagnostic dicts
    naming each affected tool call (by name and argument key) so the
    caller can fail closed via the bounded malformed-tool-call path
    instead of dispatching an encrypted/empty child assignment.

    Returns an empty list when no encrypted tokens are found.
    """
    output = getattr(responses_api_response, "output", None)
    if not isinstance(output, list):
        return []

    findings: list[dict[str, Any]] = []
    for item in output:
        if getattr(item, "type", None) != "function_call":
            continue
        arguments = getattr(item, "arguments", None)
        if not isinstance(arguments, str) or not arguments:
            continue
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue

        for key in list(parsed):
            value = parsed[key]
            if isinstance(value, str) and _is_fernet_encrypted_token(value):
                findings.append(
                    {
                        "name": getattr(item, "name", None) or "",
                        "argument_key": key,
                        "call_id": getattr(item, "call_id", None) or "",
                    }
                )

    return findings


async def _perform_codex_auto_agent_alias_candidate_request(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Any,
    candidate: dict[str, Any],
    candidate_body: dict[str, Any],
    target_url: str,
    api_key: Optional[str],
    forward_headers: bool,
) -> Response:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.codex_oauth import (
        _bind_codex_oauth_candidate_to_request,
    )

    _bind_codex_oauth_candidate_to_request(request, candidate)
    adapter_model = candidate["model"]

    async def _openrouter_completion() -> Response:
        return await _perform_codex_auto_agent_openrouter_completion_request(
            request=request,
            adapter_model=adapter_model,
            request_body=candidate_body,
            use_alias_candidate_probe=True,
        )

    async def _openrouter_responses() -> Response:
        return await _perform_codex_auto_agent_openrouter_responses_request(
            endpoint=endpoint,
            request=request,
            user_api_key_dict=user_api_key_dict,
            adapter_model=adapter_model,
            request_body=candidate_body,
            use_alias_candidate_probe=True,
        )

    async def _xai_oauth() -> Response:
        return await _perform_codex_auto_agent_oa_xai_responses_request(
            endpoint=endpoint,
            request=request,
            user_api_key_dict=user_api_key_dict,
            request_body=candidate_body,
        )

    async def _grok_native() -> Response:
        return await _perform_codex_auto_agent_grok_native_responses_request(
            endpoint=endpoint,
            request=request,
            user_api_key_dict=user_api_key_dict,
            request_body=candidate_body,
        )

    async def _opencode() -> Response:
        return await _handle_codex_opencode_zen_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _kimi_code() -> Response:
        return await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _alibaba_token_plan() -> Response:
        return await _handle_codex_alibaba_token_plan_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _native() -> Response:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.codex_oauth import (
            _codex_oauth_responses_target_url,
            _load_bound_codex_oauth_auth,
        )

        selected_auth = await _load_bound_codex_oauth_auth(request)
        return await _perform_codex_auto_agent_native_openai_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            target_url=_codex_oauth_responses_target_url(),
            api_key=None,
            forward_headers=False,
            request_body=candidate_body,
            custom_headers=selected_auth.headers,
        )

    return await _dispatch_auto_agent_alias_candidate_request(
        candidate=candidate,
        provider_handlers={
            _CODEX_AUTO_AGENT_OPENCODE_PROVIDER: _opencode,
            _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER: _kimi_code,
            _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER: _alibaba_token_plan,
        },
        route_family_handlers={
            _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER: {
                "codex_openrouter_completion_adapter": _openrouter_completion,
                "*": _openrouter_responses,
            },
            _CODEX_AUTO_AGENT_XAI_PROVIDER: {
                "codex_xai_oauth_responses_adapter": _xai_oauth,
                "*": _grok_native,
            },
        },
        default_handler=_native,
    )


async def _perform_codex_auto_agent_native_openai_request(
    *,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Any,
    target_url: str,
    api_key: Optional[str],
    forward_headers: bool,
    request_body: dict[str, Any],
    custom_headers: Optional[dict[str, str]] = None,
) -> Response:
    is_streaming_request = "stream" in str(target_url)
    resolved_headers = (
        dict(custom_headers)
        if custom_headers is not None
        else BaseOpenAIPassThroughHandler._assemble_headers(
            api_key=api_key,
            request=request,
        )
    )
    try:
        return await pass_through_request(
            request=request,
            target=target_url,
            custom_headers=resolved_headers,
            user_api_key_dict=user_api_key_dict,
            forward_headers=forward_headers,
            stream=is_streaming_request,
            custom_body=request_body,
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
            egress_credential_family=(
                "openai"
                if custom_headers is not None or forward_headers
                else None
            ),
            expected_target_family="openai",
            # RR-054 #24
            retryable_upstream_status_codes=list(_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT),
            caller_managed_hidden_retry=False,
        )
    except Exception as exc:
        if _codex_native_openai_candidate_unavailable_detail(exc) is not None:
            _raise_codex_native_openai_auto_agent_candidate_unavailable(exc)
        raise


async def _perform_codex_auto_agent_grok_native_responses_request(
    *,
    endpoint: str,
    request: Request,
    user_api_key_dict: Any,
    request_body: dict[str, Any],
) -> Response:
    (
        adapted_request_body,
        _adapted_custom_tools,
    ) = _adapt_codex_custom_tools_to_functions_from_request_body(request_body)
    try:
        grok_context = await BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
            endpoint=endpoint,
            request=request,
            request_body=adapted_request_body,
            extra_headers={},
        )
    except Exception as exc:
        if _grok_native_candidate_unavailable_detail(exc) is not None:
            _raise_grok_native_auto_agent_candidate_unavailable(exc)
        raise
    if grok_context is None:
        _raise_grok_native_auto_agent_candidate_unavailable(
            Exception("Grok native Codex auto-agent candidate requires a managed " "Grok OIDC credential.")
        )
    assert grok_context is not None
    _, grok_headers, grok_prepared_body, updated_url = grok_context
    try:
        response = await pass_through_request(
            request=request,
            target=updated_url,
            custom_headers=grok_headers,
            user_api_key_dict=user_api_key_dict,
            forward_headers=False,
            stream=bool(grok_prepared_body.get("stream")),
            custom_body=grok_prepared_body,
            custom_llm_provider=litellm.LlmProviders.XAI.value,
            egress_credential_family="xai",
            expected_target_family="xai",
            retryable_upstream_status_codes=[
                429,
                *_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES,
            ],
            caller_managed_hidden_retry=True,
        )
    except Exception as exc:
        if _grok_native_candidate_unavailable_detail(exc) is not None:
            _raise_grok_native_auto_agent_candidate_unavailable(exc)
        raise
    return await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=str(grok_prepared_body.get("model") or request_body.get("model") or "unknown-model"),
        adapter="codex_auto_agent_grok_native_responses",
        adapter_label="Grok native",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            request_body,
            adapter="codex_auto_agent_grok_native_responses",
            upstream_url=str(updated_url),
            provider="grok",
        ),
        request_body=request_body,
    )


async def _perform_codex_auto_agent_oa_xai_responses_request(
    *,
    endpoint: str,
    request: Request,
    user_api_key_dict: Any,
    request_body: dict[str, Any],
) -> Response:
    (
        adapted_request_body,
        _adapted_custom_tools,
    ) = _adapt_codex_custom_tools_to_functions_from_request_body(request_body)
    try:
        oa_xai_context = await BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
            endpoint=endpoint,
            request_body=adapted_request_body,
        )
    except Exception as exc:
        if _xai_oauth_candidate_unavailable_detail(exc) is not None:
            _raise_xai_oauth_auto_agent_candidate_unavailable(exc)
        raise
    if oa_xai_context is None:
        _raise_xai_oauth_auto_agent_candidate_unavailable(
            Exception("Codex auto-agent xAI OAuth candidate requires a managed xAI " "OAuth credential.")
        )
    assert oa_xai_context is not None
    _, oa_xai_api_key, oa_xai_prepared_body, updated_url = oa_xai_context
    try:
        response = await pass_through_request(
            request=request,
            target=updated_url,
            custom_headers=BaseOpenAIPassThroughHandler._assemble_headers(
                api_key=oa_xai_api_key,
                request=request,
            ),
            user_api_key_dict=user_api_key_dict,
            forward_headers=False,
            stream=bool(oa_xai_prepared_body.get("stream")),
            custom_body=oa_xai_prepared_body,
            custom_llm_provider=litellm.LlmProviders.XAI.value,
            egress_credential_family="xai",
            expected_target_family="xai",
            retryable_upstream_status_codes=[
                429,
                *_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES,
            ],
            caller_managed_hidden_retry=True,
        )
    except Exception as exc:
        if _xai_oauth_candidate_unavailable_detail(exc) is not None:
            _raise_xai_oauth_auto_agent_candidate_unavailable(exc)
        raise
    return await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=str(oa_xai_prepared_body.get("model") or request_body.get("model") or "unknown-model"),
        adapter="codex_auto_agent_xai_oauth_responses",
        adapter_label="xAI OAuth",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            request_body,
            adapter="codex_auto_agent_xai_oauth_responses",
            upstream_url=str(updated_url),
            provider="xai",
        ),
        request_body=request_body,
    )


async def _validate_codex_auto_agent_openrouter_responses_stream(
    response: StreamingResponse,
    *,
    adapter_model: str,
    intake_context: Optional[dict[str, Any]] = None,
) -> StreamingResponse:
    event_summaries: list[dict[str, Any]] = []
    peek = await _aawm_alias_streaming.peek_streaming_response(
        response,
        max_chunks=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS,
        max_bytes=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES,
    )
    if not peek.exhausted:
        return peek.response
    try:
        response_body = await _collect_responses_response_from_stream(
            peek.response,
            event_summaries=event_summaries,
        )
    except HTTPException as exc:
        if (
            exc.status_code == 502
            and str(exc.detail) == "OpenAI Responses stream completed without a response payload."
        ):
            _raise_codex_auto_agent_empty_success_response(
                response_body={
                    "model": adapter_model,
                    "status": "completed",
                    "output": [],
                },
                adapter_model=adapter_model,
                stream_event_summaries=event_summaries,
            )
        raise
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            stream_event_summaries=event_summaries,
        )
    if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
        _raise_codex_auto_agent_malformed_tool_call_text_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_responses",
            adapter_label="OpenRouter",
            intake_context=intake_context,
            stream_event_summaries=event_summaries,
        )
    if _is_failed_responses_body(response_body):
        _raise_codex_auto_agent_failed_responses_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_responses",
            adapter_label="OpenRouter",
            stream_event_summaries=event_summaries,
        )

    async def _replay_iterator() -> Any:
        for raw_chunk in peek.buffered_chunks:
            yield raw_chunk

    return StreamingResponse(
        _replay_iterator(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )


async def _perform_codex_auto_agent_openrouter_responses_request(
    *,
    request: Request,
    user_api_key_dict: Any,
    endpoint: str,
    adapter_model: str,
    request_body: dict[str, Any],
    use_alias_candidate_probe: bool = False,
) -> Response:
    openrouter_api_key = _get_openrouter_api_key()
    if openrouter_api_key is None:
        exc = ProxyException(
            message=(
                "OpenRouter Codex auto-agent candidate requires " "AAWM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
            ),
            type="rate_limit_error",
            param="model",
            code=429,
        )
        setattr(
            exc,
            "detail",
            {
                "error": {
                    "message": exc.message,
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                }
            },
        )
        raise exc

    target_base_url = _get_openrouter_target_base()
    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint=endpoint,
        base_target_url=target_base_url,
    )
    target_url = BaseOpenAIPassThroughHandler._join_url_paths(
        httpx.URL(target_base_url),
        normalized_endpoint,
        litellm.LlmProviders.OPENROUTER.value,
    )
    custom_headers: dict[str, Any] = BaseOpenAIPassThroughHandler._assemble_headers(
        api_key=openrouter_api_key,
        request=request,
    )
    custom_headers.update(_build_openrouter_default_headers())
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    response = await _perform_openrouter_adapter_pass_through_request(
        adapter_model=adapter_model,
        log_warnings=not use_alias_candidate_probe,
        use_alias_candidate_probe=use_alias_candidate_probe,
        request=request,
        target=str(target_url),
        custom_headers=custom_headers,
        user_api_key_dict=user_api_key_dict,
        custom_body=request_body,
        forward_headers=False,
        allowed_forward_headers=[],
        allowed_pass_through_prefixed_headers=[],
        stream=bool(request_body.get("stream")),
        custom_llm_provider=litellm.LlmProviders.OPENROUTER.value,
        egress_credential_family="openrouter",
        expected_target_family="openrouter",
    )
    if isinstance(response, StreamingResponse):
        return await _validate_codex_auto_agent_openrouter_responses_stream(
            response,
            adapter_model=adapter_model,
            intake_context=_build_malformed_tool_call_intake_context(
                request,
                request_body,
                adapter="codex_auto_agent_openrouter_responses",
                upstream_url=str(target_url),
                provider="openrouter",
            ),
        )
    if isinstance(response, Response) and not isinstance(response, StreamingResponse):
        try:
            response_body = json.loads(_decode_http_response_body(response.body))
        except Exception:
            return response
        if isinstance(response_body, dict) and _is_codex_auto_agent_empty_success_responses_body(response_body):
            _raise_codex_auto_agent_empty_success_response(
                response_body=response_body,
                adapter_model=adapter_model,
            )
        if isinstance(response_body, dict) and _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
            _raise_codex_auto_agent_malformed_tool_call_text_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter="codex_auto_agent_openrouter_responses",
                adapter_label="OpenRouter",
                intake_context=_build_malformed_tool_call_intake_context(
                    request,
                    request_body,
                    adapter="codex_auto_agent_openrouter_responses",
                    upstream_url=str(target_url),
                    provider="openrouter",
                ),
            )
        if isinstance(response_body, dict) and _is_failed_responses_body(response_body):
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter="codex_auto_agent_openrouter_responses",
                adapter_label="OpenRouter",
            )
    return response


async def _prepare_codex_kimi_chat_completions_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    prepared_request_body = _kimi_code_adapters.normalize_kimi_code_custom_tool_outputs(prepared_request_body)
    adapted_request_body, _adapted_custom_tools = _adapt_codex_custom_tools_to_functions_from_request_body(
        prepared_request_body
    )
    adapted_request_body, _adapted_namespace_tools = _adapt_codex_namespace_tools_to_functions_from_request_body(
        adapted_request_body
    )
    (
        adapted_request_body,
        _codex_tool_description_patch_events,
    ) = _apply_codex_tool_description_patches_to_request_body(adapted_request_body)
    adapted_request_body, _unsupported_hosted_tools = _drop_unsupported_codex_hosted_tools_from_request_body(
        adapted_request_body
    )
    adapted_request_body, _unsupported_input_items = _drop_unsupported_codex_input_items_from_request_body(
        adapted_request_body
    )
    adapted_request_body, _removed_tool_choice = _drop_tool_choice_without_tools_from_request_body(adapted_request_body)
    return await _kimi_code_adapters.prepare_codex_kimi_chat_completions_adapter_route(
        request=request,
        prepared_request_body=adapted_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _perform_codex_kimi_chat_completions_adapter_call(
    *,
    config: "_aawm_adapter_config.AnthropicCompletionAdapterConfig",
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    target_url: Union[str, httpx.URL],
    api_key: str,
    api_base: str,
    client_requested_stream: bool,
    completion_kwargs: Payload,
    request_input: Any,
    responses_api_request: ResponsesAPIOptionalRequestParams,
    litellm_metadata: Payload,
    upstream_model: str,
) -> Response:
    """Execute Kimi chat completions and reuse the standard Responses wrapper."""
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    _ = config
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(str(target_url)))
    completion_response = await litellm.acompletion(
        **completion_kwargs,
        api_key=api_key,
        api_base=api_base,
        litellm_metadata=litellm_metadata,
        proxy_server_request={
            "headers": dict(request.headers),
            "body": prepared_request_body,
        },
        shared_session=_get_proxy_shared_aiohttp_session(),
    )
    if client_requested_stream:
        return StreamingResponse(
            _responses_sse_from_iterator(
                LiteLLMCompletionStreamingIterator(
                    model=upstream_model,
                    litellm_custom_stream_wrapper=completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                    custom_llm_provider=litellm.LlmProviders.KIMI_CODE.value,
                    litellm_metadata=litellm_metadata,
                )
            ),
            media_type="text/event-stream",
        )
    responses_api_response = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
    )
    return _build_responses_response_from_adapter_response(responses_api_response)


async def _handle_codex_kimi_chat_completions_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Any,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    rollup_kwargs: dict[str, Any] = {}

    async def _prepare_and_emit_route_log(
        **kwargs: Any,
    ) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
        plan = await _prepare_codex_kimi_chat_completions_adapter_route(**kwargs)
        metadata = plan.perform_kwargs.get("litellm_metadata")
        if not isinstance(metadata, dict):
            metadata = plan.prepared_request_body.get("litellm_metadata")
        rollup_kwargs.update(_build_adapted_route_rollup_kwargs(metadata if isinstance(metadata, dict) else {}))
        _annotate_request_scope_for_adapted_access_log(request, plan.target_url)
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="Kimi Code",
        )
        return plan

    response = await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_and_emit_route_log,
        perform=_perform_codex_kimi_chat_completions_adapter_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    validated_response = await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter="codex_kimi_chat_completions_adapter",
        adapter_label="Kimi Code",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            prepared_request_body,
            adapter="codex_kimi_chat_completions_adapter",
            provider="kimi_code",
        ),
        request_body=prepared_request_body,
    )
    if isinstance(validated_response, StreamingResponse):
        return _record_adapted_completed_route_rollup_after_stream(
            validated_response,
            rollup_kwargs,
            adapter_label="Kimi Code",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="Kimi Code",
    )
    return validated_response


async def _prepare_codex_alibaba_token_plan_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    prepared_request_body = _alibaba_token_plan_adapters.normalize_alibaba_token_plan_custom_tool_outputs(
        prepared_request_body
    )
    adapted_request_body, _adapted_custom_tools = _adapt_codex_custom_tools_to_functions_from_request_body(
        prepared_request_body
    )
    adapted_request_body, _adapted_namespace_tools = _adapt_codex_namespace_tools_to_functions_from_request_body(
        adapted_request_body
    )
    (
        adapted_request_body,
        _codex_tool_description_patch_events,
    ) = _apply_codex_tool_description_patches_to_request_body(adapted_request_body)
    adapted_request_body, _unsupported_hosted_tools = _drop_unsupported_codex_hosted_tools_from_request_body(
        adapted_request_body
    )
    adapted_request_body, _unsupported_input_items = _drop_unsupported_codex_input_items_from_request_body(
        adapted_request_body
    )
    adapted_request_body, _removed_tool_choice = _drop_tool_choice_without_tools_from_request_body(adapted_request_body)
    return await _alibaba_token_plan_adapters.prepare_codex_alibaba_token_plan_adapter_route(
        request=request,
        prepared_request_body=adapted_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _perform_codex_alibaba_token_plan_adapter_call(
    *,
    config: "_aawm_adapter_config.AnthropicCompletionAdapterConfig",
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    target_url: Union[str, httpx.URL],
    api_key: str,
    api_base: str,
    client_requested_stream: bool,
    completion_kwargs: Payload,
    request_input: Any,
    responses_api_request: ResponsesAPIOptionalRequestParams,
    litellm_metadata: Payload,
    upstream_model: str,
) -> Response:
    """Execute Token Plan chat completions through the standard Responses wrapper."""
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    _ = config, adapter_model
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(str(target_url)))
    _acompletion_kwargs = dict(
        completion_kwargs,
        api_key=api_key,
        api_base=api_base,
        litellm_metadata=litellm_metadata,
        proxy_server_request={
            "headers": dict(request.headers),
            "body": prepared_request_body,
        },
        shared_session=_get_proxy_shared_aiohttp_session(),
    )
    # CFG-004 streaming path: the client requested SSE, but we must inspect
    # the full upstream response for encrypted reasoning tokens *before* any
    # bytes reach the client.  Buffer the response as non-streaming, check
    # for Fernet tokens in tool call arguments, retry once on the same
    # Alibaba provider/model/route if found, and only then emit a valid
    # Responses SSE stream from the confirmed-plaintext body.  If encrypted
    # content persists after the bounded retry, fail closed without
    # dispatching ciphertext.
    if client_requested_stream:
        _stream_acompletion_kwargs = dict(_acompletion_kwargs, stream=False)
        _stream_completion_response = await litellm.acompletion(
            **_stream_acompletion_kwargs
        )
        for _stream_attempt in range(_ALIBABA_ENCRYPTED_REASONING_MAX_RETRIES + 1):
            _stream_responses_api_response = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=_stream_completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                )
            )
            _stream_encrypted_findings = (
                _responses_output_contains_encrypted_reasoning_arguments(
                    _stream_responses_api_response
                )
            )
            if not _stream_encrypted_findings:
                _stream_response_body = json.loads(
                    _serialize_responses_adapter_response(
                        _stream_responses_api_response
                    )
                )
                return StreamingResponse(
                    _responses_sse_from_repaired_response_body(
                        _stream_response_body
                    ),
                    media_type="text/event-stream",
                )
            if _stream_attempt < _ALIBABA_ENCRYPTED_REASONING_MAX_RETRIES:
                _stream_completion_response = await litellm.acompletion(
                    **_stream_acompletion_kwargs
                )
        _stream_response_body = json.loads(
            _serialize_responses_adapter_response(_stream_responses_api_response)
        )
        _raise_codex_auto_agent_malformed_tool_call_text_payload(
            response_body=_stream_response_body,
            adapter_model=adapter_model,
            adapter="codex_alibaba_token_plan_chat_completions_adapter",
            adapter_label="Alibaba Token Plan",
            intake_context=_build_malformed_tool_call_intake_context(
                request,
                prepared_request_body,
                adapter="codex_alibaba_token_plan_chat_completions_adapter",
                provider="alibaba_token_plan",
            ),
        )
        # Unreachable: the raise helper always raises.
        return StreamingResponse(
            _responses_sse_from_repaired_response_body(_stream_response_body),
            media_type="text/event-stream",
        )
    completion_response = await litellm.acompletion(**_acompletion_kwargs)
    # CFG-004: bounded retry when encrypted reasoning occupies tool arguments.
    # The upstream model may non-deterministically leak a Fernet token into
    # a tool call argument (e.g. spawn_agent.message) instead of plaintext.
    # No plaintext exists to restore.  Retry the upstream call a bounded
    # number of times on the same Alibaba provider/model/route; if the leak
    # persists, fail closed via the malformed-tool-call path so the caller
    # observes the Alibaba provider and can route accordingly.
    _last_encrypted_findings: list[dict[str, Any]] = []
    for _attempt in range(_ALIBABA_ENCRYPTED_REASONING_MAX_RETRIES + 1):
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                chat_completion_response=completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
            )
        )
        _encrypted_findings = (
            _responses_output_contains_encrypted_reasoning_arguments(
                responses_api_response
            )
        )
        if not _encrypted_findings:
            return _build_responses_response_from_adapter_response(
                responses_api_response
            )
        _last_encrypted_findings = _encrypted_findings
        if _attempt < _ALIBABA_ENCRYPTED_REASONING_MAX_RETRIES:
            completion_response = await litellm.acompletion(**_acompletion_kwargs)

    _response_body = json.loads(
        _serialize_responses_adapter_response(responses_api_response)
    )
    _raise_codex_auto_agent_malformed_tool_call_text_payload(
        response_body=_response_body,
        adapter_model=adapter_model,
        adapter="codex_alibaba_token_plan_chat_completions_adapter",
        adapter_label="Alibaba Token Plan",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            prepared_request_body,
            adapter="codex_alibaba_token_plan_chat_completions_adapter",
            provider="alibaba_token_plan",
        ),
    )
    return _build_responses_response_from_adapter_response(responses_api_response)


async def _handle_codex_alibaba_token_plan_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Any,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    rollup_kwargs: dict[str, Any] = {}

    async def _prepare_and_emit_route_log(
        **kwargs: Any,
    ) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
        plan = await _prepare_codex_alibaba_token_plan_adapter_route(**kwargs)
        metadata = plan.perform_kwargs.get("litellm_metadata")
        if not isinstance(metadata, dict):
            metadata = plan.prepared_request_body.get("litellm_metadata")
        rollup_kwargs.update(_build_adapted_route_rollup_kwargs(metadata if isinstance(metadata, dict) else {}))
        _annotate_request_scope_for_adapted_access_log(request, plan.target_url)
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="Alibaba Token Plan",
        )
        return plan

    response = await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_and_emit_route_log,
        perform=_perform_codex_alibaba_token_plan_adapter_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    validated_response = await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter="codex_alibaba_token_plan_chat_completions_adapter",
        adapter_label="Alibaba Token Plan",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            prepared_request_body,
            adapter="codex_alibaba_token_plan_chat_completions_adapter",
            provider="alibaba_token_plan",
        ),
        request_body=prepared_request_body,
    )
    if isinstance(validated_response, StreamingResponse):
        return _record_adapted_completed_route_rollup_after_stream(
            validated_response,
            rollup_kwargs,
            adapter_label="Alibaba Token Plan",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="Alibaba Token Plan",
    )
    return validated_response



# ── D1-574: OpenCode Zen direct-route 429 preservation ─────────────

_OPENCODE_ZEN_DIRECT_429_ERROR_CLASSES = frozenset(
    {"capacity_exhausted", "rate_limited", "usage_limit_reached"}
)
_OPENCODE_ZEN_DIRECT_RETRY_AFTER_CEILING_SECONDS = 86400.0
_OPENCODE_ZEN_DIRECT_PEEK_MAX_BYTES = 65536


def _opencode_zen_direct_safe_retry_after(exc: Exception) -> Optional[str]:
    """Extract a safe, bounded Retry-After value from upstream headers."""
    headers = _extract_adapter_upstream_headers(exc)
    raw_retry_after: Any = None
    for header_name, header_value in headers.items():
        if str(header_name).lower() == "retry-after":
            raw_retry_after = header_value
            break
    if raw_retry_after is None:
        return None
    try:
        raw_retry_after_seconds = float(str(raw_retry_after).strip())
    except (TypeError, ValueError):
        return None
    if not (
        0
        <= raw_retry_after_seconds
        <= _OPENCODE_ZEN_DIRECT_RETRY_AFTER_CEILING_SECONDS
    ):
        return None
    retry_after = _parse_retry_after_seconds_from_headers(headers)
    if retry_after is None:
        return None
    if not (
        0 <= retry_after <= _OPENCODE_ZEN_DIRECT_RETRY_AFTER_CEILING_SECONDS
    ):
        return None
    if retry_after == int(retry_after):
        return str(int(retry_after))
    return str(round(retry_after, 1))


def _maybe_raise_opencode_zen_direct_rate_limit(exc: Exception) -> None:
    """Raise a bounded 429 ProxyException for qualifying direct-mode failures."""
    error_class = _classify_codex_auto_agent_retryable_exhaustion(exc)
    if error_class not in _OPENCODE_ZEN_DIRECT_429_ERROR_CLASSES:
        return
    retry_after = _opencode_zen_direct_safe_retry_after(exc)
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    raise ProxyException(
        message=(
            "OpenCode Zen upstream capacity is temporarily exhausted. "
            "Retry later."
        ),
        type="rate_limit_error",
        param="model",
        code=429,
        headers=headers,
    ) from exc


def _opencode_zen_direct_stream_terminal_error(exc: Exception) -> Optional[str]:
    """Return a bounded response.failed SSE event for post-first-event failures."""
    error_class = _classify_codex_auto_agent_retryable_exhaustion(exc)
    if error_class not in _OPENCODE_ZEN_DIRECT_429_ERROR_CLASSES:
        return None
    payload = {
        "type": "response.failed",
        "response": {
            "object": "response",
            "status": "failed",
            "error": {
                "type": "rate_limit_error",
                "code": "opencode_zen_capacity_exhausted",
                "message": (
                    "OpenCode Zen upstream capacity is temporarily "
                    "exhausted."
                ),
            },
        },
    }
    return (
        "event: response.failed\ndata: "
        + json.dumps(payload, separators=(",", ":"))
        + "\n\n"
    )


def _consume_opencode_zen_tools_mode_header(
    request: Request,
    prepared_request_body: dict[str, Any],
    use_alias_candidate_probe: bool,
) -> dict[str, Any]:
    """D1-574/MS-033: resolve direct-route unsupported-tools mode.

    Direct mode defaults to ``drop`` immediately before normalization. Body
    litellm_metadata wins if already present. Alias probes remain strict.
    """
    if use_alias_candidate_probe:
        return prepared_request_body

    _existing_metadata = prepared_request_body.get("litellm_metadata")
    _existing_mode = (
        _existing_metadata.get("opencode_zen_unsupported_tools_mode")
        if isinstance(_existing_metadata, dict)
        else None
    )
    if _existing_mode is not None:
        return prepared_request_body

    _header_mode_raw = request.headers.get(
        "x-aawm-opencode-zen-unsupported-tools-mode"
    )
    if _header_mode_raw is not None and _header_mode_raw.strip() != "drop":
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
    _meta = dict(prepared_request_body.get("litellm_metadata") or {})
    _meta["opencode_zen_unsupported_tools_mode"] = "drop"
    prepared_request_body["litellm_metadata"] = _meta
    return prepared_request_body


def _opencode_zen_callback_headers(request: Request) -> dict[str, Any]:
    """Copy headers without raw Langfuse trace identity overrides."""
    return {
        raw_name: raw_value
        for raw_name, raw_value in request.headers.items()
        if raw_name.strip().lower().replace("-", "_")
        not in {"langfuse_trace_name", "langfuse_trace_user_id"}
    }


def _prepare_opencode_zen_direct_observability_metadata(
    request: Request,
    prepared_request_body: dict[str, Any],
    use_alias_candidate_probe: bool,
    user_api_key_dict: Any = None,
) -> tuple[dict[str, Any], Optional[str]]:
    """Import bounded trusted identity only for direct Codex/OpenCode."""
    if use_alias_candidate_probe:
        return prepared_request_body, None

    trace_identity: dict[str, str] = {}
    bounded_end_user_header: Optional[str] = None
    for raw_name, raw_value in request.headers.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            continue
        normalized_name = raw_name.strip().lower()
        cleaned_value = raw_value.strip()
        if not cleaned_value or len(cleaned_value) > 512:
            continue
        if normalized_name.replace("-", "_") == "langfuse_trace_name":
            trace_identity["trace_name"] = cleaned_value
        elif normalized_name == "x-litellm-end-user-id":
            bounded_end_user_header = cleaned_value

    raw_end_user_id = getattr(user_api_key_dict, "end_user_id", None)
    bounded_authenticated_end_user_id = (
        raw_end_user_id.strip()
        if isinstance(raw_end_user_id, str)
        and raw_end_user_id.strip()
        and len(raw_end_user_id.strip()) <= 512
        else None
    )
    accepted_trace_user_id = (
        bounded_end_user_header
        if bounded_end_user_header is not None
        and bounded_authenticated_end_user_id == bounded_end_user_header
        else None
    )
    if accepted_trace_user_id is not None:
        trace_identity["trace_user_id"] = accepted_trace_user_id
    if not trace_identity:
        return prepared_request_body, None

    existing_metadata = prepared_request_body.get("litellm_metadata")
    litellm_metadata = (
        dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
    )
    changed = False
    for metadata_name, explicit_value in trace_identity.items():
        existing_value = litellm_metadata.get(metadata_name)
        if existing_value == explicit_value:
            continue
        source_name = f"source_{metadata_name}"
        if existing_value and not litellm_metadata.get(source_name):
            litellm_metadata[source_name] = existing_value
        litellm_metadata[metadata_name] = explicit_value
        changed = True

    if not changed:
        return prepared_request_body, accepted_trace_user_id

    updated_body = dict(prepared_request_body)
    updated_body["litellm_metadata"] = litellm_metadata
    return updated_body, accepted_trace_user_id


def _build_opencode_zen_completion_call_kwargs(
    *,
    completion_kwargs: dict[str, Any],
    api_key: str,
    target_base_url: str,
    litellm_metadata: dict[str, Any],
    request: Request,
    use_alias_candidate_probe: bool,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    return {
        **completion_kwargs,
        "api_key": api_key,
        "api_base": f"{target_base_url.rstrip('/')}/v1",
        "litellm_metadata": litellm_metadata,
        "proxy_server_request": {
            "headers": (
                dict(request.headers)
                if use_alias_candidate_probe
                else _opencode_zen_callback_headers(request)
            ),
            "body": request_body,
        },
        "shared_session": _get_proxy_shared_aiohttp_session(),
    }


def _prepare_opencode_zen_known_free_logging(
    *,
    completion_call_kwargs: dict[str, Any],
    is_known_free_direct: bool,
) -> dict[str, Any]:
    if not is_known_free_direct:
        return completion_call_kwargs

    import datetime
    import uuid

    completion_call_kwargs.setdefault(
        "litellm_call_id",
        str(uuid.uuid4()),
    )
    logging_obj, completion_call_kwargs = litellm.utils.function_setup(
        original_function="acompletion",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.datetime.now(),
        **completion_call_kwargs,
    )
    logging_obj.model_call_details["response_cost"] = 0.0
    completion_call_kwargs["litellm_logging_obj"] = logging_obj
    return completion_call_kwargs


async def _perform_opencode_zen_completion_call(
    *,
    completion_call_kwargs: dict[str, Any],
    litellm_metadata: dict[str, Any],
    accepted_trace_user_id: Optional[str],
    is_known_free_direct: bool,
) -> Any:
    if accepted_trace_user_id is not None:
        # Promote only the bounded identity accepted from the direct route
        # header, without changing the normalized top-level client user.
        litellm_metadata["user_api_key_end_user_id"] = accepted_trace_user_id
        completion_call_kwargs["metadata"] = litellm_metadata

    completion_call_kwargs = _prepare_opencode_zen_known_free_logging(
        completion_call_kwargs=completion_call_kwargs,
        is_known_free_direct=is_known_free_direct,
    )
    return await litellm.acompletion(**completion_call_kwargs)


async def _handle_codex_opencode_zen_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Any,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )
    from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen.constants import (
        _OPENCODE_ZEN_FREE_MODELS,
    )

    _ = fastapi_response
    is_known_free_direct = (
        not use_alias_candidate_probe
        and adapter_model in _OPENCODE_ZEN_FREE_MODELS
    )
    prepared_request_body = _consume_opencode_zen_tools_mode_header(
        request, prepared_request_body, use_alias_candidate_probe
    )
    (
        prepared_request_body,
        accepted_trace_user_id,
    ) = _prepare_opencode_zen_direct_observability_metadata(
        request,
        prepared_request_body,
        use_alias_candidate_probe,
        user_api_key_dict,
    )
    normalized_request = await _anthropic_opencode_zen_normalization.normalize_codex_request(
        _get_anthropic_opencode_zen_normalization_runtime(),
        prepared_request_body,
        adapter_model=adapter_model,
    )
    request_body = normalized_request.request_body
    request_input = normalized_request.request_input
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        normalized_request.responses_api_request,
    )
    litellm_metadata = normalized_request.litellm_metadata
    completion_kwargs = normalized_request.completion_kwargs

    target_base_url = _get_opencode_zen_target_base()
    target_url = _join_opencode_zen_passthrough_url(
        base_target_url=target_base_url,
        endpoint="/v1/chat/completions",
    )
    api_key = await _load_opencode_zen_api_key_for_candidate(
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    custom_headers = BaseOpenAIPassThroughHandler._assemble_headers(
        api_key=api_key,
        request=request,
    )
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=target_url,
        headers=custom_headers,
        credential_family="opencode",
        expected_target_family="opencode",
    )
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(target_url))
    rollup_kwargs = _build_adapted_route_rollup_kwargs(litellm_metadata)
    _emit_adapted_route_access_log(
        request=request,
        target_url=target_url,
        request_body=request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label="OpenCode Zen",
    )
    completion_call_kwargs = _build_opencode_zen_completion_call_kwargs(
        completion_kwargs=completion_kwargs,
        api_key=api_key,
        target_base_url=target_base_url,
        litellm_metadata=litellm_metadata,
        request=request,
        use_alias_candidate_probe=use_alias_candidate_probe,
        request_body=request_body,
    )
    try:
        completion_response = await _perform_opencode_zen_completion_call(
            completion_call_kwargs=completion_call_kwargs,
            litellm_metadata=litellm_metadata,
            accepted_trace_user_id=accepted_trace_user_id,
            is_known_free_direct=is_known_free_direct,
        )
    except Exception as exc:
        if use_alias_candidate_probe and _opencode_zen_candidate_unavailable_detail(exc) is not None:
            _raise_opencode_zen_auto_agent_candidate_unavailable(exc)
        # D1-574: direct-mode capacity/rate-limit/usage-limit -> bounded 429
        if not use_alias_candidate_probe:
            _maybe_raise_opencode_zen_direct_rate_limit(exc)
        raise
    # D1-574: known-free OpenCode models have zero cost; supply explicit
    # response_cost so the Logging -> Langfuse path records 0.0 instead of
    # null (the generic cost lookup cannot resolve openai/<model> to the
    # opencode/<model> zero-price entry).
    if is_known_free_direct:
        _hidden = getattr(completion_response, "_hidden_params", None)
        if isinstance(_hidden, dict):
            _hidden["response_cost"] = 0.0
    if bool(request_body.get("stream")):
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        stream_response = StreamingResponse(
            _responses_sse_from_iterator(
                LiteLLMCompletionStreamingIterator(
                    model=adapter_model,
                    litellm_custom_stream_wrapper=completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                    custom_llm_provider=litellm.LlmProviders.OPENAI.value,
                    litellm_metadata=litellm_metadata,
                ),
                on_complete=lambda: _record_adapted_completed_route_rollup_turn(
                    rollup_kwargs,
                    adapter_label="OpenCode Zen",
                ),
                on_stream_error=(
                    None
                    if use_alias_candidate_probe
                    else _opencode_zen_direct_stream_terminal_error
                ),
            ),
            media_type="text/event-stream",
        )
        if use_alias_candidate_probe:
            return stream_response
        # D1-574: peek for pre-first-byte streaming failures
        try:
            peek = await _aawm_alias_streaming.peek_streaming_response(
                stream_response,
                max_chunks=1,
                max_bytes=_OPENCODE_ZEN_DIRECT_PEEK_MAX_BYTES,
            )
        except Exception as peek_exc:
            _maybe_raise_opencode_zen_direct_rate_limit(peek_exc)
            raise
        return peek.response

    responses_api_response = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
    )
    response_body = json.loads(_serialize_responses_adapter_response(responses_api_response))
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_opencode_zen_completion_adapter",
            adapter_label="OpenCode Zen chat-completions",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="OpenCode Zen",
    )
    return _build_responses_response_from_adapter_response(responses_api_response)


async def _perform_codex_auto_agent_openrouter_completion_request(
    *,
    request: Request,
    adapter_model: str,
    request_body: dict[str, Any],
    use_alias_candidate_probe: bool = False,
) -> Response:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    openrouter_api_key = _get_openrouter_api_key()
    if openrouter_api_key is None:
        exc = ProxyException(
            message=(
                "OpenRouter Codex auto-agent candidate requires " "AAWM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
            ),
            type="rate_limit_error",
            param="model",
            code=429,
        )
        setattr(
            exc,
            "detail",
            {
                "error": {
                    "message": exc.message,
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                }
            },
        )
        raise exc

    requested_model = request_body.get("model")
    upstream_adapter_model = _get_openrouter_completion_adapter_upstream_model(adapter_model) or adapter_model
    route_family = "codex_openrouter_completion_adapter"
    request_body = _merge_litellm_metadata(
        _add_route_family_logging_metadata(request_body, route_family),
        tags_to_add=[
            "codex-openrouter-completion-adapter",
            f"codex-adapter-model:{adapter_model}",
            "codex-adapter-target:openrouter:/v1/chat/completions",
        ],
        extra_fields={
            "codex_adapter_model": adapter_model,
            "codex_adapter_original_model": requested_model,
            "codex_adapter_target_endpoint": "openrouter:/v1/chat/completions",
            "codex_adapter_input_shape": "openai_responses",
            "codex_adapter_output_shape": "openai_responses",
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="codex.openrouter_completion_adapter",
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": adapter_model,
                        "stream": bool(request_body.get("stream")),
                    },
                )
            ],
        },
    )
    # Restore dispatchable tool identities: adapt namespace tools to flat
    # function tools before the chat-completion transformation so upstream
    # sees spawn_agent / exec_command, not functions.collaboration.spawn_agent
    # / functions.exec.  Tool call/result IDs are preserved by the adapter.
    # Retain the canonical (namespaced) body for response validation so
    # tool_call_restore can reconstruct the original namespace map.
    canonical_request_body = request_body
    (
        request_body,
        _adapted_namespace_tools,
    ) = _adapt_codex_namespace_tools_to_functions_from_request_body(request_body)
    request_input = request_body.get("input") or ""
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        {key: value for key, value in request_body.items() if key not in {"input", "model", "litellm_metadata"}},
    )
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=upstream_adapter_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider=litellm.LlmProviders.OPENROUTER.value,
        stream=bool(request_body.get("stream")),
        metadata=litellm_metadata,
    )
    completion_kwargs["metadata"] = litellm_metadata
    (
        request_body,
        completion_kwargs,
        litellm_metadata,
    ) = _apply_openrouter_completion_message_sanitization(
        request_body=request_body,
        completion_kwargs=completion_kwargs,
        litellm_metadata=litellm_metadata,
        span_name="codex_openrouter.chat_message_shape_sanitized",
        tag="openrouter-chat-message-shape-sanitized",
    )

    target_base_url = _get_openrouter_target_base()
    target_url = f"{target_base_url.rstrip('/')}/v1/chat/completions"
    validation_headers = {
        **_build_openrouter_default_headers(),
        "Authorization": f"Bearer {openrouter_api_key}",
    }
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=target_url,
        headers=validation_headers,
        credential_family="openrouter",
        expected_target_family="openrouter",
    )
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(target_url))
    rollup_kwargs = _build_adapted_route_rollup_kwargs(litellm_metadata)
    _emit_adapted_route_access_log(
        request=request,
        target_url=target_url,
        request_body=request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label="OpenRouter chat-completions",
    )

    completion_response = await _perform_openrouter_completion_adapter_operation(
        adapter_model=upstream_adapter_model,
        operation=lambda: litellm.acompletion(
            **completion_kwargs,
            api_key=openrouter_api_key,
            api_base=f"{target_base_url.rstrip('/')}/v1",
            headers=_build_openrouter_default_headers(),
            litellm_metadata=litellm_metadata,
            proxy_server_request={
                "headers": dict(request.headers),
                "body": request_body,
            },
            shared_session=_get_proxy_shared_aiohttp_session(),
        ),
        log_warnings=not use_alias_candidate_probe,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    intake_context = _build_malformed_tool_call_intake_context(
        request,
        request_body,
        adapter="codex_auto_agent_openrouter_completion_adapter",
        upstream_url=target_url,
        provider="openrouter",
    )
    if bool(request_body.get("stream")):
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        stream_response = StreamingResponse(
            _responses_sse_from_iterator(
                LiteLLMCompletionStreamingIterator(
                    model=upstream_adapter_model,
                    litellm_custom_stream_wrapper=completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                    custom_llm_provider=litellm.LlmProviders.OPENROUTER.value,
                    litellm_metadata=litellm_metadata,
                ),
            ),
            media_type="text/event-stream",
        )
        validated_response = await _validate_codex_auto_agent_responses_payload(
            stream_response,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_completion_adapter",
            adapter_label="OpenRouter chat-completions",
            intake_context=intake_context,
            request_body=canonical_request_body,
        )
        if isinstance(validated_response, StreamingResponse):
            return _record_adapted_completed_route_rollup_after_stream(
                validated_response,
                rollup_kwargs,
                adapter_label="OpenRouter chat-completions",
            )
        _record_adapted_completed_route_rollup_turn(
            rollup_kwargs,
            adapter_label="OpenRouter chat-completions",
        )
        return validated_response

    responses_api_response = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
    )
    response_body = json.loads(_serialize_responses_adapter_response(responses_api_response))
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_completion_adapter",
            adapter_label="OpenRouter chat-completions",
        )
    built_response = _build_responses_response_from_adapter_response(responses_api_response)
    validated_response = await _validate_codex_auto_agent_responses_payload(
        built_response,
        adapter_model=adapter_model,
        adapter="codex_auto_agent_openrouter_completion_adapter",
        adapter_label="OpenRouter chat-completions",
        intake_context=intake_context,
        request_body=canonical_request_body,
    )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="OpenRouter chat-completions",
    )
    return validated_response
