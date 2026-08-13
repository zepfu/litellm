"""Wave 6A Author D extraction: payload_validation functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.

Provider-neutral SSE/build/decode helpers and ``_mapping_or_attr_get`` are
host-global integration dependencies, not locally owned functions.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional, cast

from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.integrations.aawm_agent_quality_rules import (
    is_malformed_composer_call_literal_text,
    is_malformed_grok_literal_tool_label_transcript_text,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.aawm_runtime_error_logging import (
    schedule_persist_malformed_tool_call_detection,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global modules (bound via install())
    _aawm_alias_streaming: Any

    # Host-global constants
    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS: int
    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES: int

    # Host-global functions
    def _build_malformed_tool_call_intake_context(request: Any, request_body: Any, *, adapter: str, upstream_url: Any = None, provider: Any = None, model_alias: Any = None) -> dict: ...
    def _is_empty_success_responses_body(response_body: dict) -> bool: ...
    def _is_failed_responses_body(response_body: dict) -> bool: ...
    def _should_log_aawm_alias_routing_event(log_key: str) -> bool: ...
    def _mapping_or_attr_get(obj: Any, key: str, default: Any = None) -> Any: ...
    def _decode_http_response_body(body: Any) -> str: ...
    async def _collect_responses_response_from_stream(response: Any, event_summaries: Any = None) -> dict: ...
    def _restore_adapted_custom_tool_calls_in_streaming_response(response: Any, *, request_body: Any = None, adapter_model: str = "") -> Any: ...
    def _restore_adapted_namespace_tool_calls_in_streaming_response(response: Any, *, request_body: Any = None, adapter_model: str = "") -> Any: ...
    def _restore_adapted_custom_tool_calls_in_response_body(response_body: dict, *, request_body: Any = None, adapter_model: str = "") -> tuple: ...
    def _restore_adapted_namespace_tool_calls_in_response_body(response_body: dict, *, request_body: Any = None, adapter_model: str = "") -> tuple: ...
    def _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(response_body: dict, *, request_body: Any = None) -> Any: ...
    def _raise_codex_auto_agent_malformed_adapted_custom_tool_call(*, response_body: dict, adapter_model: str, adapter: str, adapter_label: str, adapter_error: Any, stream_event_summaries: Any = None) -> None: ...
    def _responses_sse_from_repaired_response_body(
        response_body: dict,
    ) -> AsyncIterator[str]: ...
    def _build_empty_success_responses_diagnostic(*, response_body: dict, diagnostic_context: Any) -> dict: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_is_codex_auto_agent_malformed_tool_call_text_output",
    "_validate_alias_candidate_responses_stream_if_needed",
    "_build_malformed_intake_context_for_anthropic_responses_adapter",
    "_is_codex_auto_agent_empty_success_responses_body",
    "_coerce_optional_int",
    "_usage_has_no_more_than_one_output_token",
    "_model_response_usage_dict",
    "_raise_codex_auto_agent_empty_success_response",
    "_build_failed_responses_diagnostic",
    "_raise_codex_auto_agent_malformed_tool_call_text_payload",
    "_raise_codex_auto_agent_failed_responses_payload",
    "_raise_responses_adapter_failed_response",
    "_preserve_distinct_function_call_identity_fields",
    "_validate_codex_auto_agent_responses_payload",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
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
        _mod[_name] = _rebound
        host_globals[_name] = _rebound


# ── Extracted functions ─────────────────────────────────────────────


def _is_codex_auto_agent_malformed_tool_call_text_output(
    response_body: dict[str, Any],
) -> bool:
    output = response_body.get("output")
    if not isinstance(output, list):
        return False

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, str):
                if is_malformed_composer_call_literal_text(content):
                    return True
                if is_malformed_grok_literal_tool_label_transcript_text(content):
                    return True
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") not in {"text", "output_text"}:
                    continue
                part_text = part.get("text") or ""
                if is_malformed_composer_call_literal_text(part_text):
                    return True
                if is_malformed_grok_literal_tool_label_transcript_text(part_text):
                    return True
            continue

        if item.get("type") in {"function_call", "mcp_call"}:
            name = item.get("name")
            if isinstance(name, str) and name.strip().lower() == "composer_call":
                return True
            continue
    return False


async def _validate_alias_candidate_responses_stream_if_needed(
    response: StreamingResponse,
    *,
    enabled: bool,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    request: Optional[Request] = None,
    request_body: Optional[dict[str, Any]] = None,
    upstream_url: Optional[str] = None,
    provider: Optional[str] = None,
    model_alias: Optional[str] = None,
) -> StreamingResponse:
    if not enabled:
        return response
    intake_context = _build_malformed_tool_call_intake_context(  # noqa: F821
        request,
        request_body,
        adapter=adapter,
        upstream_url=upstream_url,
        provider=provider,
        model_alias=model_alias,
    )
    return cast(
        StreamingResponse,
        await _validate_codex_auto_agent_responses_payload(
            response,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            intake_context=intake_context,
        ),
    )


def _build_malformed_intake_context_for_anthropic_responses_adapter(
    *,
    request: Optional[Request],
    request_body: Optional[dict[str, Any]],
    adapter: str,
    adapter_model: str,
    upstream_url: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    return _build_malformed_tool_call_intake_context(  # noqa: F821
        request,
        request_body,
        adapter=adapter,
        upstream_url=upstream_url,
        provider=provider,
        model_alias=(
            request_body.get("model")
            if isinstance(request_body, dict) and isinstance(request_body.get("model"), str)
            else None
        ),
    )


def _is_codex_auto_agent_empty_success_responses_body(
    response_body: dict[str, Any],
) -> bool:
    if not _is_empty_success_responses_body(response_body):  # noqa: F821
        return False
    usage = response_body.get("usage") or {}
    if not isinstance(usage, dict):
        return False
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        return False
    try:
        return int(output_tokens) <= 1
    except Exception:
        return False

def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _usage_has_no_more_than_one_output_token(usage: Any) -> bool:
    if usage is None:
        return True
    saw_output_field = False
    for field in ("completion_tokens", "output_tokens", "output"):
        token_count = _coerce_optional_int(_mapping_or_attr_get(usage, field))
        if token_count is None:
            continue
        saw_output_field = True
        if token_count > 1:
            return False
    if saw_output_field:
        return True
    total_tokens = _coerce_optional_int(_mapping_or_attr_get(usage, "total_tokens"))
    if total_tokens == 0:
        return True
    return False


def _model_response_usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    result: dict[str, Any] = {}
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "output_tokens",
    ):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return result


def _raise_codex_auto_agent_empty_success_response(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str = "codex_auto_agent_openrouter_responses",
    adapter_label: str = "OpenRouter",
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    diagnostic = _build_empty_success_responses_diagnostic(  # noqa: F821
        response_body=response_body,
        diagnostic_context={
            "adapter": adapter,
            "adapter_model": adapter_model,
            **({"stream_events": stream_event_summaries} if stream_event_summaries is not None else {}),
        },
    )
    # RR-054 #23: empty successful payload is retryable upstream emptiness, not rate limit.
    exc = ProxyException(
        message=(f"Codex auto-agent {adapter_label} candidate returned an empty successful " "Responses payload."),
        type="upstream_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": exc.message,
                "code": "aawm_codex_auto_agent_empty_success",
                "status": "EMPTY_SUCCESS_RESPONSE",
                "type": "upstream_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _build_failed_responses_diagnostic(
    *,
    response_body: dict[str, Any],
    adapter: str,
    adapter_model: str,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    output = response_body.get("output") or []
    diagnostic: dict[str, Any] = {
        "adapter": adapter,
        "adapter_model": adapter_model,
        "response_id": response_body.get("id"),
        "status": response_body.get("status"),
        "model": response_body.get("model"),
        "error": response_body.get("error"),
        "incomplete_details": response_body.get("incomplete_details"),
        "output_count": len(output) if isinstance(output, list) else 0,
        "output_types": [item.get("type") for item in output[:20] if isinstance(item, dict)]
        if isinstance(output, list)
        else [],
    }
    if stream_event_summaries is not None:
        diagnostic["stream_events"] = stream_event_summaries
    return diagnostic


def _raise_codex_auto_agent_malformed_tool_call_text_payload(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    intake_context: Optional[dict[str, Any]] = None,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    try:
        # Offload synchronous JSONL intake when called under a running loop so
        # async request handlers do not block on disk I/O while rejecting
        # malformed tool-call text. Sync callers still persist inline.
        schedule_persist_malformed_tool_call_detection(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            intake_context=intake_context,
            stream_event_summaries=stream_event_summaries,
        )
    except Exception:
        # RR-054 #38: intake must stay best-effort, but never become silent.
        verbose_proxy_logger.exception("Failed to schedule malformed tool-call detection intake")
    diagnostic = _build_failed_responses_diagnostic(
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    # RR-054 #23: malformed tool-call text is not a rate limit.
    exc = ProxyException(
        message=(f"Codex auto-agent {adapter_label} candidate returned a malformed " "Responses marker payload."),
        type="invalid_request_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": exc.message,
                "code": "aawm_auto_agent_malformed_tool_call_text",
                "status": "RESPONSES_MALFORMED_TOOL_CALL",
                "type": "invalid_request_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _raise_codex_auto_agent_failed_responses_payload(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    diagnostic = _build_failed_responses_diagnostic(
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    # RR-054 #23: failed upstream Responses status is a bad gateway / upstream error.
    exc = ProxyException(
        message=(f"Auto-agent {adapter_label} candidate returned a failed Responses " "payload."),
        type="upstream_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": exc.message,
                "code": "aawm_auto_agent_failed_responses_payload",
                "status": "RESPONSES_STATUS_FAILED",
                "type": "upstream_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _raise_responses_adapter_failed_response(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    retryable_alias_candidate: bool = False,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    if retryable_alias_candidate:
        _raise_codex_auto_agent_failed_responses_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            stream_event_summaries=stream_event_summaries,
        )

    diagnostic = _build_failed_responses_diagnostic(
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    raise HTTPException(
        status_code=502,
        detail={
            "error": f"{adapter_label} Responses adapter returned a failed response.",
            "diagnostic": diagnostic,
        },
    )



def _preserve_distinct_function_call_identity_fields(
    response_body: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve distinct Responses function_call ``id`` / ``call_id`` fields.

    OPENAI-007: ``call_id`` is exclusively the upstream provider tool id and
    ``id`` is the Responses item id (``fc_*``). For typed ``function_call``
    output only, repair malformed/non-native item ids deterministically through
    the shared identity helper using ``call_id``, while preserving ``call_id``
    byte-for-byte and leaving valid native ``fc_*`` ids untouched.
    """
    if not isinstance(response_body, dict):
        return response_body
    output = response_body.get("output")
    if not isinstance(output, list):
        return response_body

    preserved_output: list[Any] = []
    changed = False
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            preserved_output.append(item)
            continue

        clean_item = dict(item)
        item_id = clean_item.get("id")
        call_id = clean_item.get("call_id")

        # Drop blank placeholders only. Never invent call_id from id.
        if "id" in clean_item and not (isinstance(item_id, str) and item_id.strip()):
            clean_item.pop("id", None)
            item_id = None
            changed = True
        if "call_id" in clean_item and not (
            isinstance(call_id, str) and call_id.strip()
        ):
            clean_item.pop("call_id", None)
            call_id = None
            changed = True

        # Repair malformed/non-native item ids from provider call_id only.
        # Preserve call_id byte-for-byte and leave valid native fc_* item ids
        # untouched. Never mutate function_call_output or nested/untyped fields.
        # Import inside the function so install()-rebound host globals still work.
        if isinstance(call_id, str) and call_id.strip():
            from litellm.responses.litellm_completion_transformation.function_call_identity import (
                is_native_responses_function_call_item_id,
                resolve_responses_function_call_identity,
            )

            resolved_item_id, _resolved_call_id = (
                resolve_responses_function_call_identity(call_id)
            )
            item_id_is_native = (
                isinstance(item_id, str)
                and bool(item_id.strip())
                and is_native_responses_function_call_item_id(item_id)
            )
            if not item_id_is_native and resolved_item_id:
                if clean_item.get("id") != resolved_item_id:
                    clean_item["id"] = resolved_item_id
                    changed = True

        preserved_output.append(clean_item)

    if not changed:
        return response_body
    updated = dict(response_body)
    updated["output"] = preserved_output
    return updated


async def _validate_codex_auto_agent_responses_payload(  # noqa: PLR0915
    response: Response,
    *,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    intake_context: Optional[dict[str, Any]] = None,
    request_body: Optional[dict[str, Any]] = None,
) -> Response:
    if isinstance(response, StreamingResponse):
        event_summaries: list[dict[str, Any]] = []
        peek = await _aawm_alias_streaming.peek_streaming_response(  # noqa: F821
            response,
            max_chunks=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS,  # noqa: F821
            max_bytes=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES,  # noqa: F821
            terminalizer=_aawm_alias_streaming._get_stream_timeout_terminalizer(  # noqa: F821
                response
            ),
        )
        if not peek.exhausted:
            correlation = intake_context or {}
            model_alias = correlation.get("model_alias")
            if model_alias is None and isinstance(request_body, dict):
                model_alias = request_body.get("model")
            session_id = correlation.get("session_id")
            litellm_call_id = correlation.get("litellm_call_id")
            trace_id = correlation.get("trace_id")
            if peek.stop_reason == "pending_stream":
                verbose_proxy_logger.debug(
                    "Codex auto-agent responses validation continued lazily "
                    "(reason=%s chunks=%s bytes=%s adapter=%s "
                    "adapter_model=%s model_alias=%s session_id=%s "
                    "litellm_call_id=%s trace_id=%s); preserving the complete "
                    "upstream stream",
                    peek.stop_reason,
                    len(peek.buffered_chunks),
                    peek.buffered_bytes,
                    adapter,
                    adapter_model,
                    model_alias or "<missing>",
                    session_id or "<missing>",
                    litellm_call_id or "<missing>",
                    trace_id or "<missing>",
                )
            elif _should_log_aawm_alias_routing_event(f"validate-stream-limit:{adapter}:{peek.stop_reason}"):  # noqa: F821
                verbose_proxy_logger.warning(
                    "Codex auto-agent responses validation bypassed after bounded "
                    "peek limit (reason=%s chunks=%s bytes=%s max_chunks=%s "
                    "max_bytes=%s adapter=%s adapter_model=%s model_alias=%s "
                    "session_id=%s litellm_call_id=%s trace_id=%s); preserving "
                    "the complete upstream stream",
                    peek.stop_reason,
                    len(peek.buffered_chunks),
                    peek.buffered_bytes,
                    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS,  # noqa: F821
                    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES,  # noqa: F821
                    adapter,
                    adapter_model,
                    model_alias or "<missing>",
                    session_id or "<missing>",
                    litellm_call_id or "<missing>",
                    trace_id or "<missing>",
                )
            restored_response = _restore_adapted_custom_tool_calls_in_streaming_response(  # noqa: F821
                peek.response,
                request_body=request_body,
                adapter_model=adapter_model,
            )
            return _restore_adapted_namespace_tool_calls_in_streaming_response(  # noqa: F821
                restored_response,
                request_body=request_body,
                adapter_model=adapter_model,
            )
        response_body = await _collect_responses_response_from_stream(  # noqa: F821
            peek.response,
            event_summaries=event_summaries,
        )
        identity_changed = False
        if isinstance(response_body, dict):
            preserved_body = _preserve_distinct_function_call_identity_fields(
                response_body
            )
            identity_changed = preserved_body is not response_body
            response_body = preserved_body
        if _is_failed_responses_body(response_body):  # noqa: F821
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                stream_event_summaries=event_summaries,
            )
        response_changed = identity_changed
        repaired_body = (
            _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(  # noqa: F821
                response_body,
                request_body=request_body,
            )
            if adapter == "codex_auto_agent_grok_native_responses"
            else None
        )
        if isinstance(repaired_body, dict):
            response_body = repaired_body
            response_changed = True
        (
            restored_body,
            restored_custom_tool_count,
            custom_tool_adapter_error,
        ) = _restore_adapted_custom_tool_calls_in_response_body(  # noqa: F821
            response_body,
            request_body=request_body,
            adapter_model=adapter_model,
        )
        if custom_tool_adapter_error is not None:
            _raise_codex_auto_agent_malformed_adapted_custom_tool_call(  # noqa: F821
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                adapter_error=custom_tool_adapter_error,
                stream_event_summaries=event_summaries,
            )
        if restored_custom_tool_count:
            response_body = restored_body
            response_changed = True
        (
            restored_body,
            restored_namespace_tool_count,
        ) = _restore_adapted_namespace_tool_calls_in_response_body(  # noqa: F821
            response_body,
            request_body=request_body,
            adapter_model=adapter_model,
        )
        if restored_namespace_tool_count:
            response_body = restored_body
            response_changed = True
        if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
            _raise_codex_auto_agent_malformed_tool_call_text_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                intake_context=intake_context,
                stream_event_summaries=event_summaries,
            )
        if response_changed:
            return StreamingResponse(
                _responses_sse_from_repaired_response_body(response_body),  # noqa: F821
                headers=dict(response.headers),
                status_code=response.status_code,
                media_type=response.media_type or "text/event-stream",
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

    if isinstance(response, Response) and not isinstance(response, StreamingResponse):
        try:
            response_body = json.loads(_decode_http_response_body(response.body))  # noqa: F821
        except Exception:
            return response
        identity_changed = False
        if isinstance(response_body, dict):
            preserved_body = _preserve_distinct_function_call_identity_fields(
                response_body
            )
            identity_changed = preserved_body is not response_body
            response_body = preserved_body
        if isinstance(response_body, dict) and _is_failed_responses_body(response_body):  # noqa: F821
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
            )
        if isinstance(response_body, dict):
            repaired_body = (
                _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(  # noqa: F821
                    response_body,
                    request_body=request_body,
                )
                if adapter == "codex_auto_agent_grok_native_responses"
                else None
            )
            if isinstance(repaired_body, dict):
                response_body = repaired_body
            (
                restored_body,
                restored_custom_tool_count,
                custom_tool_adapter_error,
            ) = _restore_adapted_custom_tool_calls_in_response_body(  # noqa: F821
                response_body,
                request_body=request_body,
                adapter_model=adapter_model,
            )
            if custom_tool_adapter_error is not None:
                _raise_codex_auto_agent_malformed_adapted_custom_tool_call(  # noqa: F821
                    response_body=response_body,
                    adapter_model=adapter_model,
                    adapter=adapter,
                    adapter_label=adapter_label,
                    adapter_error=custom_tool_adapter_error,
                )
            if restored_custom_tool_count:
                response_body = restored_body
            (
                restored_body,
                restored_namespace_tool_count,
            ) = _restore_adapted_namespace_tool_calls_in_response_body(  # noqa: F821
                response_body,
                request_body=request_body,
                adapter_model=adapter_model,
            )
            if restored_namespace_tool_count:
                response_body = restored_body
            if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
                _raise_codex_auto_agent_malformed_tool_call_text_payload(
                    response_body=response_body,
                    adapter_model=adapter_model,
                    adapter=adapter,
                    adapter_label=adapter_label,
                    intake_context=intake_context,
                )
            # Serialize identity repairs even when no unrelated repair flag is set.
            if (
                identity_changed
                or isinstance(repaired_body, dict)
                or restored_custom_tool_count
                or restored_namespace_tool_count
            ):
                return Response(
                    content=json.dumps(response_body),
                    media_type=response.media_type or "application/json",
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
    return response
