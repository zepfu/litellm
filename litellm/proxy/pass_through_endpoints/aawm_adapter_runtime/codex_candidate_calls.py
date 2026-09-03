"""Wave 6F extraction: Codex auto-agent provider candidate request functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations
# ruff: noqa: F821 - free names resolve via host globals after install() rebind

import asyncio
import copy
import json
import math
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from types import FunctionType
from typing import TYPE_CHECKING, Any, Optional, Union, cast

from litellm.proxy.pass_through_endpoints.aawm_text_watermark.config import (
    load_text_watermark_config,
)
from litellm.proxy.pass_through_endpoints.aawm_text_watermark.policy import (
    apply_request_watermark_egress,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.audit_persist import (
    _emit_aawm_terminal_error,
)
from litellm.secret_managers.credential_error_sanitizer import (
    sanitize_credential_error_message,
)

_OPENCODE_GO_ALIAS_CANDIDATE_TIMEOUT_SECONDS = 30.0
_CURSOR_REPLAY_TTL_SECONDS = 600.0
_CURSOR_REPLAY_MAX_SIZE = 256
_CURSOR_REPLAY_REGISTRY: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CURSOR_TOOL_CONTINUATION_CUE = (
    "Finish the original user request using the completed tool result above. "
    "Do not repeat completed tool calls."
)
_CURSOR_TOOL_CONTINUATION_CUE_MARKER = "_cursor_tool_continuation_cue"
_CURSOR_SESSION_CONTINUATION_FAILURE_MARKER = (
    "_cursor_session_continuation_failure"
)
_CURSOR_REPLAY_STATE_FIELD = "_cursor_replay_state"
_CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD = "cursor_sanitized_proto_structure"
_CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD = (
    "cursor_replay_fresh_dispatch_reject"
)
_CURSOR_PROTO_STRUCTURE_MAX_DEPTH = 3
_CURSOR_PROTO_STRUCTURE_MAX_ITEMS = 64
_CURSOR_REPLAY_DIAGNOSTIC_MAX_INDEX = 4096
_CURSOR_REPLAY_DIAGNOSTIC_MAX_KEY_COUNT = 32
_CURSOR_REPLAY_DIAGNOSTIC_MAX_TOKEN_CHARS = 64
_CURSOR_REPLAY_FRESH_DISPATCH_REJECTION_STAGES = frozenset(
    {
        "stock_full_history",
        "provider_neutral_tools",
        "fresh_body_copy",
        "rebuilt_body_replay_unsafe",
    }
)
_CURSOR_REPLAY_FRESH_DISPATCH_REJECTION_REASONS = frozenset(
    {
        "request_body_shape",
        "missing_replay_state",
        "replay_state_lookup",
        "replay_state_copy",
        "invalid_replay_state",
        "retained_session_present",
        "messages_container",
        "messages_empty",
        "message_role",
        "message_conversion",
        "replayed_input_container",
        "input_container",
        "input_item_count",
        "item_not_object",
        "item_key_set",
        "item_type",
        "id_shape",
        "metadata_shape",
        "metadata_key_set",
        "metadata_value_type",
        "content_container",
        "content_part_container",
        "content_part_keys",
        "content_part_type",
        "content_part_text_type",
        "empty_user_text",
        "function_call_position",
        "function_call_count",
        "function_call_output_position",
        "function_call_output_count",
        "function_call_fields",
        "call_id_shape",
        "call_id_alias_mismatch",
        "function_name",
        "arguments_not_object",
        "output_container",
        "output_not_string",
        "call_graph",
        "unresolved_call_id",
        "tool_container",
        "tool_item",
        "tool_type_validation",
        "tool_key_set",
        "tool_name",
        "tool_transform",
        "tool_validation",
        "tool_canonical_mismatch",
        "tool_copy_failure",
        "copy_failure",
        "replay_safety_rejected",
        "previous_response_id",
        "id_only_reasoning_reference",
        "explicit_item_reference",
        "invalid_body_shape",
    }
)
_CURSOR_CONTINUATION_FIELDS = frozenset(
    {
        "previous_response_id",
        "message_id",
        "messageId",
        "conversation_id",
        "conversationId",
        "conversation_group_id",
        "conversationGroupId",
        "run_id",
        "runId",
        "agent_session_id",
        "agentSessionId",
    }
)
_CURSOR_REPLAY_PRESERVED_STATUS_CODES = frozenset(
    {408, 500, 502, 503, 504, 529}
)


class _CursorPostEgressOutputError(ValueError):
    """A returned Cursor payload could not be normalized after provider Run."""


@dataclass(frozen=True)
class _CursorReplayFreshDispatchReject:
    stage: str
    reason: str
    item_index: Optional[int] = None
    tool_index: Optional[int] = None
    item_type: Optional[str] = None
    tool_type: Optional[str] = None
    item_keys: Optional[tuple[str, ...]] = None
    tool_keys: Optional[tuple[str, ...]] = None

    def to_dict(self) -> dict[str, Any]:
        diagnostic: dict[str, Any] = {
            "stage": self.stage,
            "reason": self.reason,
        }
        for field in ("item_index", "tool_index", "item_type", "tool_type"):
            value = getattr(self, field)
            if value is not None:
                diagnostic[field] = value
        for field in ("item_keys", "tool_keys"):
            value = getattr(self, field)
            if value is not None:
                diagnostic[field] = list(value)
        return diagnostic


@dataclass(frozen=True)
class _CursorReplayValidationResult:
    value: Any
    rejection: Optional[_CursorReplayFreshDispatchReject] = None


@dataclass(frozen=True)
class _CursorReplayFreshDispatchBuildResult:
    body: Optional[dict[str, Any]]
    rejection: Optional[_CursorReplayFreshDispatchReject] = None


def _cursor_replay_safe_diagnostic_token(value: Any) -> Optional[str]:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _CURSOR_REPLAY_DIAGNOSTIC_MAX_TOKEN_CHARS
        or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None
    ):
        return None
    return value


def _cursor_replay_safe_diagnostic_keys(
    value: Any,
) -> Optional[tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return None
    try:
        raw_keys = list(value.keys())
    except Exception:  # noqa: BLE001
        return None
    safe_keys = {
        key
        for key in raw_keys
        if _cursor_replay_safe_diagnostic_token(key) is not None
    }
    return tuple(sorted(safe_keys)[:_CURSOR_REPLAY_DIAGNOSTIC_MAX_KEY_COUNT])


def _cursor_replay_bounded_diagnostic_index(value: Any) -> Optional[int]:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _CURSOR_REPLAY_DIAGNOSTIC_MAX_INDEX
    ):
        return None
    return value


def _cursor_replay_rejection(
    stage: str,
    reason: str,
    *,
    item_index: Any = None,
    tool_index: Any = None,
    item: Any = None,
    tool: Any = None,
) -> _CursorReplayFreshDispatchReject:
    item_type = (
        _cursor_replay_safe_diagnostic_token(item.get("type"))
        if isinstance(item, Mapping)
        else None
    )
    tool_type = (
        _cursor_replay_safe_diagnostic_token(tool.get("type"))
        if isinstance(tool, Mapping)
        else None
    )
    return _CursorReplayFreshDispatchReject(
        stage=stage,
        reason=reason,
        item_index=_cursor_replay_bounded_diagnostic_index(item_index),
        tool_index=_cursor_replay_bounded_diagnostic_index(tool_index),
        item_type=item_type,
        tool_type=tool_type,
        item_keys=_cursor_replay_safe_diagnostic_keys(item),
        tool_keys=_cursor_replay_safe_diagnostic_keys(tool),
    )


def _cursor_replay_rejected(
    stage: str,
    reason: str,
    *,
    item_index: Any = None,
    tool_index: Any = None,
    item: Any = None,
    tool: Any = None,
) -> _CursorReplayValidationResult:
    return _CursorReplayValidationResult(
        value=None,
        rejection=_cursor_replay_rejection(
            stage,
            reason,
            item_index=item_index,
            tool_index=tool_index,
            item=item,
            tool=tool,
        ),
    )


def _cursor_replay_build_rejected(
    stage: str,
    reason: str,
    *,
    item_index: Any = None,
    tool_index: Any = None,
    item: Any = None,
    tool: Any = None,
) -> _CursorReplayFreshDispatchBuildResult:
    return _CursorReplayFreshDispatchBuildResult(
        body=None,
        rejection=_cursor_replay_rejection(
            stage,
            reason,
            item_index=item_index,
            tool_index=tool_index,
            item=item,
            tool=tool,
        ),
    )


def _cursor_replay_fresh_dispatch_reject_for_replay_safety(
    replay_safety: Any,
) -> Optional[dict[str, Any]]:
    if getattr(replay_safety, "safe", False):
        return None
    classification = getattr(replay_safety, "classification", None)
    if classification not in {
        "previous_response_id",
        "id_only_reasoning_reference",
        "explicit_item_reference",
        "invalid_body_shape",
    }:
        classification = "replay_safety_rejected"
    return _CursorReplayFreshDispatchReject(
        stage="rebuilt_body_replay_unsafe",
        reason=classification,
    ).to_dict()


def _sanitize_cursor_proto_structure_for_telemetry(
    body: Any,
) -> Optional[dict[str, Any]]:
    """Copy only the bounded, value-free Cursor protobuf wire structure."""
    if not isinstance(body, dict):
        return None
    raw_fields = body.get("fields")
    if not isinstance(raw_fields, list):
        return None

    item_count = 0

    def _sanitize_fields(
        fields: Any,
        *,
        depth: int,
    ) -> Optional[list[dict[str, Any]]]:
        nonlocal item_count
        if not isinstance(fields, list) or depth > _CURSOR_PROTO_STRUCTURE_MAX_DEPTH:
            return None

        sanitized_fields: list[dict[str, Any]] = []
        for raw_field in fields:
            if (
                item_count >= _CURSOR_PROTO_STRUCTURE_MAX_ITEMS
                or not isinstance(raw_field, dict)
            ):
                return None
            field_number = raw_field.get("field_number")
            wire_type = raw_field.get("wire_type")
            payload_length = raw_field.get("payload_length")
            if (
                isinstance(field_number, bool)
                or not isinstance(field_number, int)
                or field_number <= 0
                or isinstance(wire_type, bool)
                or not isinstance(wire_type, int)
                or not 0 <= wire_type <= 7
                or isinstance(payload_length, bool)
                or not isinstance(payload_length, int)
                or payload_length < 0
            ):
                return None

            item_count += 1
            sanitized_field: dict[str, Any] = {
                "field_number": field_number,
                "wire_type": wire_type,
                "payload_length": payload_length,
            }
            nested_fields = raw_field.get("nested_fields")
            if nested_fields is not None:
                if depth >= _CURSOR_PROTO_STRUCTURE_MAX_DEPTH:
                    return None
                sanitized_nested_fields = _sanitize_fields(
                    nested_fields,
                    depth=depth + 1,
                )
                if sanitized_nested_fields is None:
                    return None
                sanitized_field["nested_fields"] = sanitized_nested_fields
            sanitized_fields.append(sanitized_field)
        return sanitized_fields

    sanitized_fields = _sanitize_fields(raw_fields, depth=0)
    if sanitized_fields is None:
        return None
    return {"fields": sanitized_fields}


def _close_cursor_retained_session(state: Optional[dict[str, Any]]) -> None:
    if not isinstance(state, dict):
        return
    session = state.get("retained_session")
    close = getattr(session, "close", None)
    if callable(close):
        close()


def _cancel_cursor_replay_expiry(state: Optional[dict[str, Any]]) -> None:
    if not isinstance(state, dict):
        return
    handle = state.get("expiry_handle")
    state["expiry_handle"] = None
    cancel = getattr(handle, "cancel", None)
    if callable(cancel):
        cancel()


def _dispose_cursor_replay_state(
    state: Optional[dict[str, Any]],
    *,
    close_retained_session: bool = True,
) -> None:
    _cancel_cursor_replay_expiry(state)
    if close_retained_session:
        _close_cursor_retained_session(state)


def _expire_cursor_replay_state(
    response_id: str,
    expected_state: dict[str, Any],
) -> None:
    current = _CURSOR_REPLAY_REGISTRY.get(response_id)
    if current is not expected_state:
        return
    _CURSOR_REPLAY_REGISTRY.pop(response_id, None)
    expected_state["expiry_handle"] = None
    _close_cursor_retained_session(expected_state)


def _schedule_cursor_replay_expiry(
    response_id: str,
    state: dict[str, Any],
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    delay = max(0.0, float(state["expires_at"]) - time.monotonic())
    state["expiry_handle"] = loop.call_later(
        delay,
        _expire_cursor_replay_state,
        response_id,
        state,
    )


def _prune_cursor_replay_registry(now: Optional[float] = None) -> None:
    current = time.monotonic() if now is None else now
    expired = [
        response_id
        for response_id, state in _CURSOR_REPLAY_REGISTRY.items()
        if float(state["expires_at"]) <= current
    ]
    for response_id in expired:
        state = _CURSOR_REPLAY_REGISTRY.pop(response_id, None)
        _dispose_cursor_replay_state(state)


def _store_cursor_replay_state(
    response_id: str,
    *,
    messages: list[dict[str, Any]],
    tools: list[Any],
    retained_session: Any = None,
) -> None:
    now = time.monotonic()
    _prune_cursor_replay_registry(now)
    previous = _CURSOR_REPLAY_REGISTRY.pop(response_id, None)
    _dispose_cursor_replay_state(previous)
    state = {
        "expires_at": now + _CURSOR_REPLAY_TTL_SECONDS,
        "expiry_handle": None,
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(tools),
        "retained_session": retained_session,
    }
    _CURSOR_REPLAY_REGISTRY[response_id] = state
    _schedule_cursor_replay_expiry(response_id, state)
    _CURSOR_REPLAY_REGISTRY.move_to_end(response_id)
    while len(_CURSOR_REPLAY_REGISTRY) > _CURSOR_REPLAY_MAX_SIZE:
        _evicted_id, evicted_state = _CURSOR_REPLAY_REGISTRY.popitem(last=False)
        _dispose_cursor_replay_state(evicted_state)


def _peek_cursor_replay_state(response_id: str) -> dict[str, Any]:
    from litellm.llms.cursor_agent.connect import CursorConnectError

    now = time.monotonic()
    state = _CURSOR_REPLAY_REGISTRY.get(response_id)
    if state is None:
        _prune_cursor_replay_registry(now)
        raise CursorConnectError(
            "Cursor Agent continuation state is missing for previous_response_id.",
            status_code=409,
        )
    if float(state["expires_at"]) <= now:
        _CURSOR_REPLAY_REGISTRY.pop(response_id, None)
        _dispose_cursor_replay_state(state)
        _prune_cursor_replay_registry(now)
        raise CursorConnectError(
            "Cursor Agent continuation state expired for previous_response_id.",
            status_code=409,
        )
    _prune_cursor_replay_registry(now)
    _CURSOR_REPLAY_REGISTRY.move_to_end(response_id)
    return state


def _consume_cursor_replay_state(
    response_id: str,
    *,
    expected_state: Optional[dict[str, Any]] = None,
    close_retained_session: bool = True,
) -> None:
    current = _CURSOR_REPLAY_REGISTRY.get(response_id)
    if expected_state is not None and current is not expected_state:
        return
    state = _CURSOR_REPLAY_REGISTRY.pop(response_id, None)
    _dispose_cursor_replay_state(
        state,
        close_retained_session=close_retained_session,
    )


def _take_cursor_replay_state(response_id: str) -> dict[str, Any]:
    state = _peek_cursor_replay_state(response_id)
    _consume_cursor_replay_state(response_id, expected_state=state)
    return state


def _cursor_replay_state_snapshot(
    state: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not isinstance(state, dict):
        return None
    try:
        return {
            "messages": copy.deepcopy(state.get("messages")),
            "tools": copy.deepcopy(state.get("tools")),
            "retained_session": state.get("retained_session"),
        }
    except Exception:  # noqa: BLE001
        return None


def _raise_cursor_session_continuation_unavailable(
    *,
    previous_response_id: Optional[str] = None,
    replay_state: Optional[dict[str, Any]] = None,
) -> None:
    from litellm.llms.cursor_agent.connect import CursorConnectError

    exc = CursorConnectError(
        "Cursor Agent tool-output continuation cannot resume because its "
        "live retained session is unavailable.",
        status_code=409,
    )
    replay_snapshot = _cursor_replay_state_snapshot(replay_state)
    if replay_snapshot is not None:
        setattr(exc, _CURSOR_REPLAY_STATE_FIELD, replay_snapshot)
    if previous_response_id and replay_state is not None:
        _consume_cursor_replay_state(
            previous_response_id,
            expected_state=replay_state,
        )
    setattr(exc, _CURSOR_SESSION_CONTINUATION_FAILURE_MARKER, True)
    raise exc


def _cursor_replay_failure_is_transient(
    exc: BaseException,
    *,
    transport_failure: bool = False,
) -> bool:
    raw_status = getattr(exc, "status_code", None)
    try:
        status_code = int(raw_status) if raw_status is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code in _CURSOR_REPLAY_PRESERVED_STATUS_CODES:
        return True
    return transport_failure and status_code is None


def _clear_cursor_replay_registry() -> None:
    states = list(_CURSOR_REPLAY_REGISTRY.values())
    _CURSOR_REPLAY_REGISTRY.clear()
    for state in states:
        _dispose_cursor_replay_state(state)


def _watermark_endpoint_from_path(*parts: Any) -> str:
    combined = " ".join(
        str(part or "") for part in parts if part is not None
    ).lower()
    if "chat/completions" in combined or "chat_completions" in combined:
        return "chat_completions"
    return "responses"


def _get_runtime_text_watermark_config() -> Any:
    payload = None
    try:
        from litellm.proxy.proxy_server import general_settings as _gs

        if isinstance(_gs, dict):
            payload = _gs.get("openai_passthrough_text_watermark")
        else:
            payload = getattr(_gs, "openai_passthrough_text_watermark", None)
    except Exception:
        payload = None
    return load_text_watermark_config(payload)


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
    from litellm.llms.zai_coding_plan.adapters import (
        adapter as _zai_coding_plan_adapters,
    )
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
    _CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER: str
    _CODEX_AUTO_AGENT_COHERE_PROVIDER: str
    _CODEX_AUTO_AGENT_NOUS_PROVIDER: str
    _CODEX_AUTO_AGENT_NVIDIA_PROVIDER: str
    _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER: str
    _CODEX_AUTO_AGENT_OPENCODE_PROVIDER: str
    _CODEX_AUTO_AGENT_OPENCODE_GO_PROVIDER: str
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
    def _build_responses_response_from_adapter_response(
        response_obj: Any,
        *,
        request_body: Any = None,
    ) -> Response: ...
    def _codex_native_openai_candidate_unavailable_detail(
        exc: Any,
        **kwargs: Any,
    ) -> Optional[str]: ...
    async def _collect_responses_response_from_stream(response: Any, **kwargs: Any) -> dict[str, Any]: ...
    def _decode_http_response_body(body: Any) -> str: ...
    async def _dispatch_auto_agent_alias_candidate_request(**kwargs: Any) -> Response: ...
    def _drop_tool_choice_without_tools_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _drop_unsupported_codex_hosted_tools_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _drop_unsupported_codex_input_items_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _drop_unsupported_codex_request_params_from_request_body(body: Any) -> tuple[Any, Any]: ...
    def _emit_adapted_route_access_log(**kwargs: Any) -> None: ...
    def _get_anthropic_opencode_zen_normalization_runtime() -> Any: ...
    def _get_opencode_zen_target_base() -> str: ...
    def _get_opencode_go_target_base() -> str: ...
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
    def _raise_codex_native_openai_auto_agent_candidate_unavailable(
        exc: Exception,
        **kwargs: Any,
    ) -> Any: ...
    def _raise_grok_native_auto_agent_candidate_unavailable(exc: Exception) -> Any: ...
    def _raise_opencode_zen_auto_agent_candidate_unavailable(exc: Exception) -> Any: ...
    def _raise_xai_oauth_auto_agent_candidate_unavailable(exc: Exception) -> Any: ...
    def _record_adapted_completed_route_rollup_after_stream(response: Any, rollup: Any, **kwargs: Any) -> Any: ...
    def _record_adapted_completed_route_rollup_turn(rollup: Any, **kwargs: Any) -> None: ...
    def _record_opencode_go_provider_rejection_evidence(
        request: Any, evidence: dict[str, Any]
    ) -> dict[str, Any]: ...
    def _restore_adapted_custom_tool_calls_in_response_body(
        response_body: dict[str, Any],
        *,
        request_body: Any = None,
        adapter_model: str = "",
    ) -> tuple: ...
    def _restore_adapted_namespace_tool_calls_in_response_body(
        response_body: dict[str, Any],
        *,
        request_body: Any = None,
        adapter_model: str = "",
    ) -> tuple: ...
    def _responses_sse_from_iterator(iterator: Any, **kwargs: Any) -> Any: ...
    def _responses_sse_from_repaired_response_body(
        response_body: dict[str, Any],
        *,
        request_body: Any = None,
    ) -> Any: ...
    def _serialize_responses_adapter_response(response_obj: Any) -> str: ...
    async def _validate_codex_auto_agent_responses_payload(response: Any, **kwargs: Any) -> Any: ...
    def _xai_oauth_candidate_unavailable_detail(exc: Exception) -> Optional[str]: ...
    def _build_opencode_go_provider_rejection_evidence(**kwargs: Any) -> dict[str, Any]: ...


# ── Host-global function names (bound via install()) ────────────────

_HOST_FUNCTION_NAMES = (
    # Top-level candidate dispatcher
    "_perform_codex_auto_agent_alias_candidate_request",
    # Provider-specific candidate requests
    "_perform_codex_auto_agent_native_openai_request",
    "_perform_codex_auto_agent_grok_native_responses_request",
    "_perform_codex_auto_agent_oa_xai_responses_request",
    "_maybe_wrap_xai_passthrough_responses_stream",
    "_bind_responses_stream_timeout_terminalizer",
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
    # Z.AI Coding Plan
    "_prepare_codex_zai_coding_plan_adapter_route",
    "_perform_codex_zai_coding_plan_adapter_call",
    "_handle_codex_zai_coding_plan_adapter_route",
    # OpenCode
    "_handle_codex_opencode_zen_adapter_route",
    "_handle_codex_opencode_go_adapter_route",
    "_build_opencode_go_provider_rejection_evidence",
    "_record_opencode_go_provider_rejection_evidence",
    "_raise_opencode_go_alias_candidate_upstream_timeout",
    "_handle_codex_nous_chat_completions_adapter_route",
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

_COHERE_FUNCTION_NAMES = (
    "_build_codex_cohere_adapter_request_body",
    "_strip_strict_from_cohere_completion_tools",
    "_prepare_codex_cohere_chat_completions_adapter_route",
    "_perform_codex_cohere_chat_completions_adapter_call",
    "_handle_codex_cohere_chat_completions_adapter_route",
)

_NVIDIA_FUNCTION_NAMES = (
    "_build_codex_nvidia_adapter_request_body",
    "_prepare_codex_nvidia_completion_adapter_route",
    "_perform_codex_nvidia_completion_adapter_call",
    "_handle_codex_nvidia_completion_adapter_route",
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
    host_globals.setdefault("_emit_aawm_terminal_error", _emit_aawm_terminal_error)
    for _name in (
        "_perform_codex_auto_agent_cursor_agent_request",
        "_raise_cursor_agent_alias_error",
        "_raise_codex_auto_agent_missing_credential_preflight",
        "_load_codex_auto_agent_opencode_zen_api_key",
    ):
        host_globals.setdefault(_name, _mod[_name])

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

    for _name in (*_COHERE_FUNCTION_NAMES, *_NVIDIA_FUNCTION_NAMES):
        _obj = _mod.get(_name)
        if not isinstance(_obj, FunctionType):
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
    for _name, _value in (
        ("apply_request_watermark_egress", apply_request_watermark_egress),
        ("load_text_watermark_config", load_text_watermark_config),
        ("_get_runtime_text_watermark_config", _get_runtime_text_watermark_config),
        ("_watermark_endpoint_from_path", _watermark_endpoint_from_path),
        ("_opencode_go_tool_type", _opencode_go_tool_type),
        ("_opencode_go_tool_types", _opencode_go_tool_types),
        ("_extract_opencode_go_offending_tool_index", _extract_opencode_go_offending_tool_index),
        ("_sanitize_opencode_go_error_text", _sanitize_opencode_go_error_text),
        ("_OPENCODE_GO_CHAT_COMPLETIONS_ROUTE", _OPENCODE_GO_CHAT_COMPLETIONS_ROUTE),
        ("_OPENCODE_GO_TOOLS_INDEX_RE", _OPENCODE_GO_TOOLS_INDEX_RE),
    ):
        host_globals.setdefault(_name, _value)


# ── Extracted functions ─────────────────────────────────────────────


def _strip_strict_from_cohere_completion_tools(
    completion_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Remove function-tool strict flags without changing caller-owned data."""
    tools = completion_kwargs.get("tools")
    if not isinstance(tools, list):
        return completion_kwargs

    from copy import deepcopy

    sanitized_kwargs = dict(completion_kwargs)
    sanitized_tools = deepcopy(tools)
    for tool in sanitized_tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        tool.pop("strict", None)
        function = tool.get("function")
        if isinstance(function, dict):
            function.pop("strict", None)
    sanitized_kwargs["tools"] = sanitized_tools
    return sanitized_kwargs


def _maybe_wrap_xai_passthrough_responses_stream(
    response: Response,
    *,
    request: Request,
    request_body: dict[str, Any],
    route_family: str,
    resolved_model: Any = None,
) -> Response:
    """Live-forward CFG-025 wrap for xAI alias Responses SSE."""
    from fastapi.responses import StreamingResponse
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.repetitive_output import (
        inherit_or_wrap_passthrough_streaming_response,
        maybe_wrap_passthrough_responses_stream,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.output_guard_config import (
        output_guard_context_from_passthrough,
    )

    if not isinstance(response, StreamingResponse):
        return response
    request_context = output_guard_context_from_passthrough(
        ingress_path=str(getattr(getattr(request, "url", None), "path", "") or ""),
        method=str(getattr(request, "method", None) or "POST"),
        custom_llm_provider=litellm.LlmProviders.XAI.value,
        egress_credential_family="xai",
        route_family=route_family,
        resolved_model=resolved_model,
        request_body=request_body,
    )
    wrapped_iter = maybe_wrap_passthrough_responses_stream(
        response.body_iterator,
        request_context=request_context,
    )
    if wrapped_iter is response.body_iterator:
        return inherit_or_wrap_passthrough_streaming_response(
            response,
            request_context=request_context,
        )
    reconstructed = StreamingResponse(
        wrapped_iter,
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )
    return inherit_or_wrap_passthrough_streaming_response(
        reconstructed,
        request_context=request_context,
    )


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
    if isinstance(candidate_body, dict):
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            strip_route_identity_from_request_body,
        )

        stripped_candidate_body = strip_route_identity_from_request_body(
            candidate_body
        )
        if (
            stripped_candidate_body is not candidate_body
            and isinstance(stripped_candidate_body, dict)
        ):
            candidate_body.clear()
            candidate_body.update(stripped_candidate_body)
    adapter_model = candidate["model"]
    cohere_provider = globals().get("_CODEX_AUTO_AGENT_COHERE_PROVIDER", "cohere")
    zai_coding_plan_provider = globals().get(
        "_CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER", "zai_coding_plan"
    )
    cursor_agent_provider = globals().get(
        "_CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER", "cursor_agent"
    )
    nvidia_provider = globals().get("_CODEX_AUTO_AGENT_NVIDIA_PROVIDER", "nvidia")

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

    opencode_go_provider = globals().get(
        "_CODEX_AUTO_AGENT_OPENCODE_GO_PROVIDER", "opencode_go"
    )

    async def _opencode_go() -> Response:
        if candidate.get("route_family") != "codex_opencode_go_adapter":
            raise ValueError(
                "OpenCode Go alias candidates require "
                "codex_opencode_go_adapter."
            )
        return await _handle_codex_opencode_go_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    nous_provider = globals().get("_CODEX_AUTO_AGENT_NOUS_PROVIDER", "nous")

    async def _nous() -> Response:
        if candidate.get("route_family") != "codex_nous_chat_completions_adapter":
            raise ValueError(
                "Nous alias candidates require "
                "codex_nous_chat_completions_adapter."
            )
        return await _handle_codex_nous_chat_completions_adapter_route(
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

    async def _zai_coding_plan() -> Response:
        return await _handle_codex_zai_coding_plan_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _cohere() -> Response:
        if (
            candidate.get("route_family")
            != "codex_cohere_chat_completions_adapter"
        ):
            raise ValueError(
                "Cohere alias candidates require "
                "codex_cohere_chat_completions_adapter."
            )
        return await _handle_codex_cohere_chat_completions_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _cursor_agent() -> Response:
        try:
            return await _perform_codex_auto_agent_cursor_agent_request(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                candidate=candidate,
                candidate_body=candidate_body,
                target_url=target_url,
                api_key=api_key,
                forward_headers=forward_headers,
            )
        except Exception as exc:
            from litellm.proxy._types import ProxyException

            if isinstance(exc, ProxyException):
                raise
            _raise_cursor_agent_alias_error(
                exc=exc,
                candidate=candidate,
            )

    async def _nvidia() -> Response:
        if candidate.get("route_family") != "codex_nvidia_completion_adapter":
            raise ValueError(
                "NVIDIA alias candidates require "
                "codex_nvidia_completion_adapter."
            )
        return await _handle_codex_nvidia_completion_adapter_route(
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
            opencode_go_provider: _opencode_go,
            nous_provider: _nous,
            _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER: _kimi_code,
            _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER: _alibaba_token_plan,
            zai_coding_plan_provider: _zai_coding_plan,
            cohere_provider: _cohere,
            cursor_agent_provider: _cursor_agent,
            nvidia_provider: _nvidia,
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


def _raise_cursor_agent_alias_not_implemented(
    *,
    ingress: str,
    candidate: dict[str, Any],
) -> None:
    """Fail closed for Cursor Agent alias dispatch in this catalog wave.

    Catalog/alias selection may compile ``cursor_agent`` candidates, but the
    Codex/Anthropic aiserver adapter is not implemented yet. Do not fall
    through to Cloud Agents ``cursor`` or native Codex/OpenAI credentials.
    """
    from litellm.proxy._types import ProxyException

    model = str(candidate.get("model") or "")
    route_family = str(candidate.get("route_family") or "")
    message = (
        "aawm_codex_auto_agent_candidate_unavailable: "
        "cursor_agent alias dispatch is not implemented for this wave; "
        f"ingress={ingress} model={model} route_family={route_family}. "
        "Do not route through Cloud Agents cursor."
    )
    exc = ProxyException(
        message=message,
        type="rate_limit_error",
        param="model",
        code=429,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": message,
                "code": "aawm_codex_auto_agent_candidate_unavailable",
            }
        },
    )
    raise exc


def _cursor_as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def _cursor_response_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "input_text", "output_text"):
            nested = value.get(key)
            if isinstance(nested, str):
                return nested
            if isinstance(nested, dict):
                text = _cursor_response_content_text(nested)
                if text:
                    return text
        return ""
    if isinstance(value, list):
        return "".join(_cursor_response_content_text(item) for item in value)
    return ""


def _cursor_function_call_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, int, float, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "{}"


def _cursor_call_id(item: dict[str, Any]) -> str:
    return str(
        item.get("call_id")
        or item.get("callId")
        or item.get("id")
        or ""
    )


def _cursor_function_name(item: dict[str, Any]) -> str:
    function = _cursor_as_mapping(item.get("function"))
    return str(item.get("name") or function.get("name") or "")


def _cursor_function_call_message(
    item: dict[str, Any],
    function_calls: dict[str, str],
) -> dict[str, Any]:
    call_id = _cursor_call_id(item)
    name = _cursor_function_name(item)
    function = _cursor_as_mapping(item.get("function"))
    arguments = item.get("arguments")
    if arguments is None:
        arguments = function.get("arguments")
    if not call_id or not name:
        raise ValueError(
            "Cursor Agent continuation requires function_call call_id and name."
        )
    function_calls[call_id] = name
    return {
        "role": "assistant",
        "content": _cursor_response_content_text(item.get("content")),
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _cursor_function_call_arguments(arguments),
                },
            }
        ],
    }


def _validate_cursor_returned_tool_calls(tool_calls: list[Any]) -> None:
    for tool_call in tool_calls:
        item = _cursor_as_mapping(tool_call)
        if not item.get("call_id") or not item.get("name"):
            raise _CursorPostEgressOutputError(
                "Cursor Agent returned a malformed function call; "
                "call_id and name are required."
            )


def _cursor_tool_result_message(
    item: dict[str, Any],
    function_calls: dict[str, str],
) -> dict[str, Any]:
    call_id = _cursor_call_id(item)
    if not call_id or call_id not in function_calls:
        raise ValueError(
            "Cursor Agent continuation requires a matching "
            "function_call for every function_call_output."
        )
    output = item.get("output")
    if output is None:
        output = item.get("content")
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": _cursor_response_content_text(output),
    }


def _remember_cursor_message_tool_calls(
    tool_calls: Any,
    function_calls: dict[str, str],
) -> None:
    if not isinstance(tool_calls, list):
        return
    for tool_call in tool_calls:
        call_mapping = _cursor_as_mapping(tool_call)
        call_id = _cursor_call_id(call_mapping)
        name = _cursor_function_name(call_mapping)
        if call_id and name:
            function_calls[call_id] = name


def _cursor_message_input_item(
    item: dict[str, Any],
    function_calls: dict[str, str],
) -> Optional[dict[str, Any]]:
    item_type = str(item.get("type") or "")
    if item_type == "input_text":
        return {
            "role": "user",
            "content": _cursor_response_content_text(item.get("text")),
        }
    role = str(item.get("role") or "")
    if role not in {"user", "assistant", "system", "developer"} and item_type != "message":
        return None
    message_role = role or "user"
    if message_role == "developer":
        message_role = "system"
    message: dict[str, Any] = {
        "role": message_role,
        "content": _cursor_response_content_text(item.get("content")),
    }
    if message_role == "assistant":
        tool_calls = item.get("tool_calls") or item.get("toolCalls")
        if isinstance(tool_calls, list):
            message["tool_calls"] = tool_calls
            _remember_cursor_message_tool_calls(tool_calls, function_calls)
    return message


def _cursor_response_input_items(request_body: dict[str, Any]) -> list[Any]:
    raw_input = request_body.get("input", "")
    if isinstance(raw_input, list):
        return raw_input
    if raw_input is None:
        return []
    return [raw_input]


def _cursor_function_call_outputs(
    request_body: dict[str, Any],
) -> list[tuple[str, str]]:
    outputs: list[tuple[str, str]] = []
    for raw_item in _cursor_response_input_items(request_body):
        item = _cursor_as_mapping(raw_item)
        if not item:
            continue
        item_type = str(item.get("type") or "")
        if item_type not in {"function_call_output", "mcp_call_output"} and (
            item.get("role") != "tool"
        ):
            continue
        call_id = _cursor_call_id(item)
        output = item.get("output")
        if output is None:
            output = item.get("content")
        outputs.append((call_id, _cursor_response_content_text(output)))
    return outputs


def _cursor_replay_function_call_output_items(
    request_body: dict[str, Any],
    *,
    allow_missing_metadata: bool = False,
) -> _CursorReplayValidationResult:
    input_items = _cursor_response_input_items(request_body)
    if not input_items:
        return _cursor_replay_rejected(
            "fresh_body_copy",
            "output_container",
        )

    outputs: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    for item_index, raw_item in enumerate(input_items):
        if not isinstance(raw_item, Mapping):
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "item_not_object",
                item_index=item_index,
                item=raw_item,
            )
        item = dict(raw_item)
        if item.get("type") != "function_call_output":
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "item_type",
                item_index=item_index,
                item=item,
            )
        if set(item) - {
            "type",
            "id",
            "call_id",
            "callId",
            "output",
            "internal_chat_message_metadata_passthrough",
        }:
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "item_key_set",
                item_index=item_index,
                item=item,
            )
        item_id = item.get("id")
        if item_id is not None:
            if not isinstance(item_id, str) or not item_id.startswith("fco_"):
                return _cursor_replay_rejected(
                    "fresh_body_copy",
                    "id_shape",
                    item_index=item_index,
                    item=item,
                )
            try:
                canonical_item_uuid = str(uuid.UUID(item_id.removeprefix("fco_")))
            except (AttributeError, ValueError):
                return _cursor_replay_rejected(
                    "fresh_body_copy",
                    "id_shape",
                    item_index=item_index,
                    item=item,
                )
            if item_id != f"fco_{canonical_item_uuid}" or item_id in seen_item_ids:
                return _cursor_replay_rejected(
                    "fresh_body_copy",
                    "id_shape",
                    item_index=item_index,
                    item=item,
                )
            seen_item_ids.add(item_id)
        metadata_present = "internal_chat_message_metadata_passthrough" in item
        metadata = item.get("internal_chat_message_metadata_passthrough")
        if metadata is not None:
            if (
                not isinstance(metadata, Mapping)
                or set(metadata) != {"turn_id", "create_time"}
            ):
                return _cursor_replay_rejected(
                    "fresh_body_copy",
                    "metadata_key_set"
                    if isinstance(metadata, Mapping)
                    else "metadata_shape",
                    item_index=item_index,
                    item=item,
                )
            turn_id = metadata.get("turn_id")
            create_time = metadata.get("create_time")
            try:
                canonical_turn_id = (
                    str(uuid.UUID(turn_id)) if isinstance(turn_id, str) else None
                )
            except (AttributeError, ValueError):
                return _cursor_replay_rejected(
                    "fresh_body_copy",
                    "id_shape",
                    item_index=item_index,
                    item=item,
                )
            if (
                not isinstance(turn_id, str)
                or turn_id != canonical_turn_id
                or isinstance(create_time, bool)
                or not isinstance(create_time, float)
                or not math.isfinite(create_time)
            ):
                return _cursor_replay_rejected(
                    "fresh_body_copy",
                    "metadata_value_type",
                    item_index=item_index,
                    item=item,
                )
        if allow_missing_metadata:
            if item_id is None or (metadata_present and metadata is None):
                return _cursor_replay_rejected(
                    "fresh_body_copy",
                    "metadata_shape",
                    item_index=item_index,
                    item=item,
                )
        elif (item_id is None) != (metadata is None):
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "metadata_shape",
                item_index=item_index,
                item=item,
            )
        snake_call_id = item.get("call_id")
        camel_call_id = item.get("callId")
        if snake_call_id is not None and (
            not isinstance(snake_call_id, str) or not snake_call_id.strip()
        ):
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "call_id_shape",
                item_index=item_index,
                item=item,
            )
        if camel_call_id is not None and (
            not isinstance(camel_call_id, str) or not camel_call_id.strip()
        ):
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "call_id_shape",
                item_index=item_index,
                item=item,
            )
        normalized_snake_call_id = (
            snake_call_id.strip() if isinstance(snake_call_id, str) else None
        )
        normalized_camel_call_id = (
            camel_call_id.strip() if isinstance(camel_call_id, str) else None
        )
        if (
            normalized_snake_call_id is not None
            and normalized_camel_call_id is not None
            and normalized_snake_call_id != normalized_camel_call_id
        ):
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "call_id_alias_mismatch",
                item_index=item_index,
                item=item,
            )
        call_id = normalized_snake_call_id or normalized_camel_call_id
        if call_id is None:
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "call_id_shape",
                item_index=item_index,
                item=item,
            )
        if "output" not in item:
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "output_not_string",
                item_index=item_index,
                item=item,
            )
        output = item["output"]
        if not isinstance(output, str):
            return _cursor_replay_rejected(
                "fresh_body_copy",
                "output_not_string",
                item_index=item_index,
                item=item,
            )
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        )
    return _CursorReplayValidationResult(value=outputs)


def _cursor_replay_is_canonical_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return value == str(uuid.UUID(value))
    except (AttributeError, ValueError):
        return False


def _cursor_replay_stock_codex_message_item(
    raw_item: Mapping[str, Any],
    *,
    allow_missing_metadata: bool = False,
) -> _CursorReplayValidationResult:
    item = dict(raw_item)
    metadata_key = "internal_chat_message_metadata_passthrough"
    expected_item_keys = {
        "type",
        "id",
        "role",
        "content",
        metadata_key,
    }
    if set(item) != expected_item_keys and not (
        allow_missing_metadata and set(item) == expected_item_keys - {metadata_key}
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "item_key_set",
            item=item,
        )
    role = item.get("role")
    if role not in {"developer", "user", "assistant"}:
        return _cursor_replay_rejected(
            "stock_full_history",
            "message_role",
            item=item,
        )

    item_id = item.get("id")
    if not isinstance(item_id, str):
        return _cursor_replay_rejected(
            "stock_full_history",
            "id_shape",
            item=item,
        )
    if role == "assistant":
        if re.fullmatch(r"msg_resp_[0-9a-f]{32}", item_id) is None:
            return _cursor_replay_rejected(
                "stock_full_history",
                "id_shape",
                item=item,
            )
    elif not item_id.startswith("msg_") or not _cursor_replay_is_canonical_uuid(
        item_id.removeprefix("msg_")
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "id_shape",
            item=item,
        )

    metadata_present = "internal_chat_message_metadata_passthrough" in item
    metadata = item.get("internal_chat_message_metadata_passthrough")
    expected_metadata_keys = (
        {"turn_id", "content_item_kinds"}
        if role == "assistant"
        else {"turn_id", "create_time", "content_item_kinds"}
    )
    if metadata is None and (not allow_missing_metadata or metadata_present):
        return _cursor_replay_rejected(
            "stock_full_history",
            "metadata_shape",
            item=item,
        )
    if metadata is not None and (
        not isinstance(metadata, Mapping) or set(metadata) != expected_metadata_keys
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "metadata_key_set"
            if isinstance(metadata, Mapping)
            else "metadata_shape",
            item=item,
        )
    if metadata is not None and not _cursor_replay_is_canonical_uuid(
        metadata.get("turn_id")
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "id_shape",
            item=item,
        )
    create_time = metadata.get("create_time") if metadata is not None else None
    if metadata is not None and role != "assistant" and (
        isinstance(create_time, bool)
        or not isinstance(create_time, float)
        or not math.isfinite(create_time)
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "metadata_value_type",
            item=item,
        )
    content_item_kinds = (
        metadata.get("content_item_kinds") if metadata is not None else None
    )
    if metadata is not None and (
        not isinstance(content_item_kinds, list)
        or not content_item_kinds
        or any(
            not isinstance(kind, str)
            or not kind
            or kind != kind.strip()
            or re.fullmatch(r"[a-z][a-z0-9_.-]*", kind) is None
            for kind in content_item_kinds
        )
        or len(content_item_kinds) != len(set(content_item_kinds))
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "metadata_value_type",
            item=item,
        )

    content = item.get("content")
    expected_content_type = "output_text" if role == "assistant" else "input_text"
    if not isinstance(content, list) or not content:
        return _cursor_replay_rejected(
            "stock_full_history",
            "content_container",
            item=item,
        )
    canonical_content_parts: list[str] = []
    for raw_part in content:
        if not isinstance(raw_part, Mapping):
            return _cursor_replay_rejected(
                "stock_full_history",
                "content_part_container",
                item=item,
            )
        part = dict(raw_part)
        if (
            set(part) != {"type", "text"}
            or part.get("type") != expected_content_type
            or not isinstance(part.get("text"), str)
        ):
            if set(part) != {"type", "text"}:
                reason = "content_part_keys"
            elif part.get("type") != expected_content_type:
                reason = "content_part_type"
            else:
                reason = "content_part_text_type"
            return _cursor_replay_rejected(
                "stock_full_history",
                reason,
                item=item,
            )
        canonical_content_parts.append(part["text"])
    return _CursorReplayValidationResult(
        value={
            "role": role,
            "content": "".join(canonical_content_parts),
        }
    )


def _cursor_replay_stock_codex_function_call_item(
    raw_item: Mapping[str, Any],
    *,
    allow_missing_metadata: bool = False,
) -> _CursorReplayValidationResult:
    item = dict(raw_item)
    metadata_key = "internal_chat_message_metadata_passthrough"
    expected_item_keys = {
        "type",
        "id",
        "name",
        "arguments",
        "call_id",
        metadata_key,
    }
    if set(item) != expected_item_keys and not (
        allow_missing_metadata and set(item) == expected_item_keys - {metadata_key}
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "item_key_set",
            item=item,
        )

    item_id = item.get("id")
    item_id_match = (
        re.fullmatch(
            r"fc_([0-9a-f-]{36})(?:_(?:0|[1-9][0-9]*))?",
            item_id,
        )
        if isinstance(item_id, str)
        else None
    )
    if item_id_match is None or not _cursor_replay_is_canonical_uuid(
        item_id_match.group(1)
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "id_shape",
            item=item,
        )

    metadata_present = "internal_chat_message_metadata_passthrough" in item
    metadata = item.get("internal_chat_message_metadata_passthrough")
    if metadata is None and (not allow_missing_metadata or metadata_present):
        return _cursor_replay_rejected(
            "stock_full_history",
            "metadata_shape",
            item=item,
        )
    if metadata is not None and (
        not isinstance(metadata, Mapping)
        or set(metadata) != {"turn_id"}
        or not _cursor_replay_is_canonical_uuid(metadata.get("turn_id"))
    ):
        return _cursor_replay_rejected(
            "stock_full_history",
            "metadata_key_set"
            if isinstance(metadata, Mapping)
            else "metadata_shape",
            item=item,
        )

    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if (
        not isinstance(call_id, str)
        or not call_id
        or call_id != call_id.strip()
        or not isinstance(name, str)
        or not name
        or name != name.strip()
        or not isinstance(arguments, str)
    ):
        if not isinstance(call_id, str) or not call_id or call_id != call_id.strip():
            reason = "call_id_shape"
        elif not isinstance(name, str) or not name or name != name.strip():
            reason = "function_name"
        else:
            reason = "arguments_not_object"
        return _cursor_replay_rejected(
            "stock_full_history",
            reason,
            item=item,
        )
    try:
        parsed_arguments = json.loads(arguments)
    except (TypeError, ValueError):
        return _cursor_replay_rejected(
            "stock_full_history",
            "arguments_not_object",
            item=item,
        )
    if not isinstance(parsed_arguments, dict):
        return _cursor_replay_rejected(
            "stock_full_history",
            "arguments_not_object",
            item=item,
        )
    return _CursorReplayValidationResult(
        value={
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }
    )


def _cursor_replay_stock_codex_full_history_input(
    request_body: dict[str, Any],
) -> _CursorReplayValidationResult:
    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return _cursor_replay_rejected(
            "stock_full_history",
            "input_container",
        )
    if len(input_items) < 3:
        return _cursor_replay_rejected(
            "stock_full_history",
            "input_item_count",
        )

    replayed_input: list[dict[str, Any]] = []
    saw_nonempty_user_message = False
    function_call_count = 0
    output_count = 0
    for item_index, raw_item in enumerate(input_items):
        if not isinstance(raw_item, Mapping):
            return _cursor_replay_rejected(
                "stock_full_history",
                "item_not_object",
                item_index=item_index,
                item=raw_item,
            )
        item_type = raw_item.get("type")
        if item_type == "message":
            message_result = _cursor_replay_stock_codex_message_item(
                raw_item,
                allow_missing_metadata=True,
            )
            if message_result.rejection is not None:
                rejection = message_result.rejection
                return _CursorReplayValidationResult(
                    value=None,
                    rejection=_cursor_replay_rejection(
                        "stock_full_history",
                        rejection.reason,
                        item_index=item_index,
                        item=raw_item,
                    ),
                )
            message_item = message_result.value
            if message_item["role"] == "user" and bool(
                _cursor_response_content_text(message_item["content"]).strip()
            ):
                saw_nonempty_user_message = True
            replayed_input.append(message_item)
            continue
        if item_type == "function_call":
            if item_index != len(input_items) - 2:
                return _cursor_replay_rejected(
                    "stock_full_history",
                    "function_call_position",
                    item_index=item_index,
                    item=raw_item,
                )
            if function_call_count:
                return _cursor_replay_rejected(
                    "stock_full_history",
                    "function_call_count",
                    item_index=item_index,
                    item=raw_item,
                )
            function_call_result = _cursor_replay_stock_codex_function_call_item(
                raw_item,
                allow_missing_metadata=True,
            )
            if function_call_result.rejection is not None:
                rejection = function_call_result.rejection
                return _CursorReplayValidationResult(
                    value=None,
                    rejection=_cursor_replay_rejection(
                        "stock_full_history",
                        rejection.reason,
                        item_index=item_index,
                        item=raw_item,
                    ),
                )
            function_call_item = function_call_result.value
            function_call_count += 1
            replayed_input.append(function_call_item)
            continue
        if item_type == "function_call_output":
            if item_index != len(input_items) - 1:
                return _cursor_replay_rejected(
                    "stock_full_history",
                    "function_call_output_position",
                    item_index=item_index,
                    item=raw_item,
                )
            if function_call_count != 1:
                return _cursor_replay_rejected(
                    "stock_full_history",
                    "function_call_count",
                    item_index=item_index,
                    item=raw_item,
                )
            if output_count:
                return _cursor_replay_rejected(
                    "stock_full_history",
                    "function_call_output_count",
                    item_index=item_index,
                    item=raw_item,
                )
            if set(raw_item) not in (
                {
                    "type",
                    "id",
                    "call_id",
                    "output",
                },
                {
                    "type",
                    "id",
                    "call_id",
                    "output",
                    "internal_chat_message_metadata_passthrough",
                },
            ):
                return _cursor_replay_rejected(
                    "stock_full_history",
                    "item_key_set",
                    item_index=item_index,
                    item=raw_item,
                )
            output_result = _cursor_replay_function_call_output_items(
                {"input": [raw_item]},
                allow_missing_metadata=True,
            )
            if output_result.rejection is not None:
                rejection = output_result.rejection
                return _CursorReplayValidationResult(
                    value=None,
                    rejection=_cursor_replay_rejection(
                        "stock_full_history",
                        rejection.reason,
                        item_index=item_index,
                        item=raw_item,
                    ),
                )
            output_items = output_result.value
            if len(output_items) != 1:
                return _cursor_replay_rejected(
                    "stock_full_history",
                    "function_call_output_count",
                    item_index=item_index,
                    item=raw_item,
                )
            output_count += 1
            replayed_input.extend(output_items)
            continue
        return _cursor_replay_rejected(
            "stock_full_history",
            "item_type",
            item_index=item_index,
            item=raw_item,
        )

    if not saw_nonempty_user_message:
        return _cursor_replay_rejected(
            "stock_full_history",
            "empty_user_text",
        )
    if function_call_count != 1:
        return _cursor_replay_rejected(
            "stock_full_history",
            "function_call_count",
        )
    if output_count != 1:
        return _cursor_replay_rejected(
            "stock_full_history",
            "function_call_output_count",
        )
    unresolved_result = _cursor_replay_unresolved_function_call_ids(
        replayed_input,
        stage="stock_full_history",
    )
    if unresolved_result.rejection is not None:
        return unresolved_result
    if unresolved_result.value != set():
        return _cursor_replay_rejected(
            "stock_full_history",
            "unresolved_call_id",
        )
    return _CursorReplayValidationResult(value=replayed_input)


def _cursor_replay_unresolved_function_call_ids(
    replayed_input: list[Any],
    *,
    stage: str = "stock_full_history",
) -> _CursorReplayValidationResult:
    seen_call_ids: set[str] = set()
    seen_output_ids: set[str] = set()
    unresolved_call_ids: set[str] = set()

    for item_index, item in enumerate(replayed_input):
        if not isinstance(item, dict):
            return _cursor_replay_rejected(
                stage,
                "item_not_object",
                item_index=item_index,
                item=item,
            )
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or not isinstance(name, str)
                or not name.strip()
            ):
                return _cursor_replay_rejected(
                    stage,
                    "function_call_fields",
                    item_index=item_index,
                    item=item,
                )
            call_id = call_id.strip()
            if call_id in seen_call_ids:
                return _cursor_replay_rejected(
                    stage,
                    "call_graph",
                    item_index=item_index,
                    item=item,
                )
            item["call_id"] = call_id
            item["name"] = name.strip()
            seen_call_ids.add(call_id)
            unresolved_call_ids.add(call_id)
        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                return _cursor_replay_rejected(
                    stage,
                    "unresolved_call_id",
                    item_index=item_index,
                    item=item,
                )
            call_id = call_id.strip()
            if call_id in seen_output_ids or call_id not in unresolved_call_ids:
                return _cursor_replay_rejected(
                    stage,
                    "unresolved_call_id",
                    item_index=item_index,
                    item=item,
                )
            item["call_id"] = call_id
            seen_output_ids.add(call_id)
            unresolved_call_ids.remove(call_id)

    if not seen_call_ids:
        return _cursor_replay_rejected(stage, "function_call_count")
    return _CursorReplayValidationResult(value=unresolved_call_ids)


def _cursor_replay_canonicalize_stock_tool_search(
    tool: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if set(tool) != {"description", "execution", "parameters", "type"}:
        return None
    if (
        tool.get("type") != "tool_search"
        or tool.get("execution") != "client"
        or not isinstance(tool.get("description"), str)
        or not tool["description"]
    ):
        return None

    parameters = tool.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    if set(parameters) != {
        "additionalProperties",
        "properties",
        "required",
        "type",
    }:
        return None
    if (
        parameters.get("type") != "object"
        or parameters.get("required") != ["query"]
        or parameters.get("additionalProperties") is not False
    ):
        return None

    properties = parameters.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != {
        "limit",
        "query",
    }:
        return None
    for property_name, property_type in (
        ("query", "string"),
        ("limit", "number"),
    ):
        property_schema = properties.get(property_name)
        if (
            not isinstance(property_schema, Mapping)
            or set(property_schema) != {"description", "type"}
            or property_schema.get("type") != property_type
            or not isinstance(property_schema.get("description"), str)
            or not property_schema["description"]
        ):
            return None

    try:
        canonical_tool = json.loads(json.dumps(dict(tool)))
    except Exception:  # noqa: BLE001
        return None
    return (
        canonical_tool
        if isinstance(canonical_tool, dict) and canonical_tool == dict(tool)
        else None
    )


def _cursor_replay_canonicalize_stock_web_search(
    tool: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if set(tool) != {"type", "external_web_access"}:
        return None
    if (
        tool.get("type") != "web_search"
        or not isinstance(tool.get("external_web_access"), bool)
    ):
        return None
    return dict(tool)


def _cursor_replay_provider_neutral_tools(
    replay_tools: Any,
) -> _CursorReplayValidationResult:
    if not isinstance(replay_tools, list):
        return _cursor_replay_rejected(
            "provider_neutral_tools",
            "tool_container",
        )

    try:
        from openai.types.responses.response_create_params import ToolParam
        from pydantic import TypeAdapter

        from litellm.llms.openai.responses.count_tokens.transformation import (
            OpenAICountTokensConfig,
        )

        tools = [
            dict(tool) if isinstance(tool, Mapping) else None
            for tool in replay_tools
        ]
        for tool_index, tool in enumerate(tools):
            if tool is None:
                return _cursor_replay_rejected(
                    "provider_neutral_tools",
                    "tool_item",
                    tool_index=tool_index,
                    tool=(
                        replay_tools[tool_index]
                        if tool_index < len(replay_tools)
                        else None
                    ),
                )
        transformed_tools = OpenAICountTokensConfig._transform_tools_for_responses_api(
            cast(list[dict[str, Any]], tools)
        )
        tool_adapter = TypeAdapter(ToolParam)
    except Exception:  # noqa: BLE001
        return _cursor_replay_rejected(
            "provider_neutral_tools",
            "tool_transform",
        )

    provider_neutral_tools: list[dict[str, Any]] = []
    for tool_index, (original_tool, transformed_tool) in enumerate(
        zip(tools, transformed_tools)
    ):
        original_tool_value = (
            replay_tools[tool_index]
            if tool_index < len(replay_tools)
            else original_tool
        )
        if not isinstance(original_tool, dict) or not isinstance(
            transformed_tool, dict
        ):
            return _cursor_replay_rejected(
                "provider_neutral_tools",
                "tool_item",
                tool_index=tool_index,
                tool=original_tool_value,
            )
        validation_tool = dict(transformed_tool)
        if validation_tool.get("type") == "function":
            function = original_tool.get("function")
            if function is not None:
                if (
                    set(original_tool) != {"type", "function"}
                    or not isinstance(function, Mapping)
                    or set(function)
                    - {"name", "description", "parameters", "strict"}
                ):
                    return _cursor_replay_rejected(
                        "provider_neutral_tools",
                        "tool_key_set",
                        tool_index=tool_index,
                        tool=original_tool,
                    )
            name = validation_tool.get("name")
            if not isinstance(name, str) or not name.strip():
                return _cursor_replay_rejected(
                    "provider_neutral_tools",
                    "tool_name",
                    tool_index=tool_index,
                    tool=original_tool,
                )
            validation_tool["name"] = name.strip()
            validation_tool.setdefault("parameters", {})
            validation_tool.setdefault("strict", None)
        if validation_tool.get("type") == "tool_search":
            canonical_tool = _cursor_replay_canonicalize_stock_tool_search(
                validation_tool
            )
            if canonical_tool is None:
                return _cursor_replay_rejected(
                    "provider_neutral_tools",
                    "tool_validation",
                    tool_index=tool_index,
                    tool=original_tool,
                )
        elif "external_web_access" in validation_tool:
            canonical_tool = _cursor_replay_canonicalize_stock_web_search(
                validation_tool
            )
            if canonical_tool is None:
                return _cursor_replay_rejected(
                    "provider_neutral_tools",
                    "tool_validation",
                    tool_index=tool_index,
                    tool=original_tool,
                )
        else:
            try:
                validated_tool = tool_adapter.validate_python(
                    validation_tool,
                    strict=True,
                )
                canonical_tool = json.loads(tool_adapter.dump_json(validated_tool))
            except Exception:  # noqa: BLE001
                reason = (
                    "tool_type_validation"
                    if not isinstance(validation_tool.get("type"), str)
                    or not validation_tool.get("type", "").strip()
                    else "tool_validation"
                )
                return _cursor_replay_rejected(
                    "provider_neutral_tools",
                    reason,
                    tool_index=tool_index,
                    tool=original_tool,
                )
        if canonical_tool != validation_tool:
            return _cursor_replay_rejected(
                "provider_neutral_tools",
                "tool_canonical_mismatch",
                tool_index=tool_index,
                tool=original_tool,
            )
        try:
            provider_neutral_tools.append(copy.deepcopy(original_tool))
        except Exception:  # noqa: BLE001
            return _cursor_replay_rejected(
                "provider_neutral_tools",
                "tool_copy_failure",
                tool_index=tool_index,
                tool=original_tool,
            )
    return _CursorReplayValidationResult(value=provider_neutral_tools)


def _build_cursor_replay_safe_fresh_dispatch_body_result(  # noqa: PLR0915
    request_body: Any,
    *,
    continuation_exc: Optional[BaseException] = None,
) -> _CursorReplayFreshDispatchBuildResult:
    if not isinstance(request_body, dict):
        return _cursor_replay_build_rejected(
            "fresh_body_copy",
            "request_body_shape",
        )

    replay_state = getattr(
        continuation_exc,
        _CURSOR_REPLAY_STATE_FIELD,
        None,
    )
    registry_state: Optional[dict[str, Any]] = None
    replayed_input: Optional[list[dict[str, Any]]] = None
    output_items: Optional[list[dict[str, Any]]] = None
    instructions: Optional[str] = None
    replay_tools_source: Any = None
    if not isinstance(replay_state, dict):
        previous_response_id = request_body.get("previous_response_id")
        if isinstance(previous_response_id, str) and previous_response_id:
            try:
                registry_state = _peek_cursor_replay_state(previous_response_id)
            except Exception:  # noqa: BLE001
                return _cursor_replay_build_rejected(
                    "fresh_body_copy",
                    "replay_state_lookup",
                )
            replay_state = _cursor_replay_state_snapshot(registry_state)
            if replay_state is None:
                return _cursor_replay_build_rejected(
                    "fresh_body_copy",
                    "replay_state_copy",
                )
        elif (
            previous_response_id is None
            and getattr(
                continuation_exc,
                _CURSOR_SESSION_CONTINUATION_FAILURE_MARKER,
                False,
            )
        ):
            full_history_result = _cursor_replay_stock_codex_full_history_input(
                request_body
            )
            if full_history_result.rejection is not None:
                return _CursorReplayFreshDispatchBuildResult(
                    body=None,
                    rejection=full_history_result.rejection,
                )
            replayed_input = full_history_result.value
            replay_tools_source = request_body.get("tools")
        else:
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "missing_replay_state",
            )
    if replayed_input is None:
        if not isinstance(replay_state, dict):
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "invalid_replay_state",
            )
        if replay_state.get("retained_session") is not None:
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "retained_session_present",
            )

        messages = replay_state.get("messages")
        if not isinstance(messages, list):
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "messages_container",
            )
        if not messages:
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "messages_empty",
            )
        if any(
            str(_cursor_as_mapping(message).get("role") or "").strip().lower()
            not in {"system", "developer", "user", "assistant", "tool"}
            for message in messages
        ):
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "message_role",
            )
        if not any(
            str(_cursor_as_mapping(message).get("role") or "").strip().lower()
            == "user"
            and bool(
                _cursor_response_content_text(
                    _cursor_as_mapping(message).get("content")
                ).strip()
            )
            for message in messages
        ):
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "empty_user_text",
            )
        try:
            from litellm.llms.openai.responses.count_tokens.transformation import (
                OpenAICountTokensConfig,
            )

            replayed_input, instructions = (
                OpenAICountTokensConfig.messages_to_responses_input(messages)
            )
        except Exception:  # noqa: BLE001
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "message_conversion",
            )
        if not isinstance(replayed_input, list):
            return _cursor_replay_build_rejected(
                "fresh_body_copy",
                "replayed_input_container",
            )
        unresolved_result = _cursor_replay_unresolved_function_call_ids(
            replayed_input,
            stage="fresh_body_copy",
        )
        if unresolved_result.rejection is not None:
            return _CursorReplayFreshDispatchBuildResult(
                body=None,
                rejection=unresolved_result.rejection,
            )
        unresolved_call_ids = unresolved_result.value
        output_result = _cursor_replay_function_call_output_items(request_body)
        if output_result.rejection is not None:
            return _CursorReplayFreshDispatchBuildResult(
                body=None,
                rejection=output_result.rejection,
            )
        output_items = output_result.value

        for output_index, output_item in enumerate(output_items):
            call_id = output_item["call_id"]
            if call_id not in unresolved_call_ids:
                return _cursor_replay_build_rejected(
                    "fresh_body_copy",
                    "unresolved_call_id",
                    item_index=output_index,
                    item=output_item,
                )
            unresolved_call_ids.remove(call_id)
        replay_tools_source = replay_state.get("tools")

    replay_tools_result = _cursor_replay_provider_neutral_tools(replay_tools_source)
    if replay_tools_result.rejection is not None:
        return _CursorReplayFreshDispatchBuildResult(
            body=None,
            rejection=replay_tools_result.rejection,
        )
    replay_tools = replay_tools_result.value

    try:
        fresh_body = copy.deepcopy(request_body)
        fresh_body["input"] = [
            *replayed_input,
            *(output_items or []),
        ]
        fresh_body["tools"] = replay_tools
    except Exception:  # noqa: BLE001
        return _cursor_replay_build_rejected(
            "fresh_body_copy",
            "copy_failure",
        )
    for field in _CURSOR_CONTINUATION_FIELDS:
        fresh_body.pop(field, None)
    if instructions is not None:
        fresh_body["instructions"] = instructions

    if registry_state is not None:
        previous_response_id = request_body.get("previous_response_id")
        if isinstance(previous_response_id, str) and previous_response_id:
            _consume_cursor_replay_state(
                previous_response_id,
                expected_state=registry_state,
                close_retained_session=False,
            )
    return _CursorReplayFreshDispatchBuildResult(body=fresh_body)


def _build_cursor_replay_safe_fresh_dispatch_body(
    request_body: Any,
    *,
    continuation_exc: Optional[BaseException] = None,
    rejection_diagnostic_out: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    if rejection_diagnostic_out is not None:
        rejection_diagnostic_out.clear()
    result = _build_cursor_replay_safe_fresh_dispatch_body_result(
        request_body,
        continuation_exc=continuation_exc,
    )
    if (
        rejection_diagnostic_out is not None
        and result.rejection is not None
    ):
        rejection_diagnostic_out[
            _CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
        ] = result.rejection.to_dict()
    return result.body


def _responses_input_to_cursor_messages(  # noqa: PLR0915
    request_body: dict[str, Any],
    *,
    prior_messages: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Translate one Responses request, including a tool continuation."""
    messages = copy.deepcopy(prior_messages or [])
    instructions = request_body.get("instructions")
    if instructions and not prior_messages:
        messages.append(
            {
                "role": "system",
                "content": _cursor_response_content_text(instructions),
            }
        )

    function_calls: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "assistant":
            _remember_cursor_message_tool_calls(
                message.get("tool_calls") or message.get("toolCalls"),
                function_calls,
            )
    saw_function_call_output = False
    last_item_was_user = False
    function_call_output_ends_input = False
    input_items = _cursor_response_input_items(request_body)

    for item_index, raw_item in enumerate(input_items):
        if isinstance(raw_item, str):
            messages.append({"role": "user", "content": raw_item})
            last_item_was_user = bool(raw_item)
            continue
        item = _cursor_as_mapping(raw_item)
        if not item:
            raise ValueError("Cursor Agent received an unsupported empty input item.")
        item_type = str(item.get("type") or "")
        role = str(item.get("role") or "")

        if item_type in {"function_call", "mcp_call"}:
            messages.append(_cursor_function_call_message(item, function_calls))
            last_item_was_user = False
            continue

        if item_type in {"function_call_output", "mcp_call_output"}:
            messages.append(_cursor_tool_result_message(item, function_calls))
            saw_function_call_output = True
            function_call_output_ends_input = (
                item_type == "function_call_output"
                and item_index == len(input_items) - 1
            )
            last_item_was_user = False
            continue

        if role == "tool":
            messages.append(_cursor_tool_result_message(item, function_calls))
            saw_function_call_output = True
            last_item_was_user = False
            continue

        if item_type in {"reasoning", "computer_call", "computer_call_output"}:
            last_item_was_user = False
            continue

        message = _cursor_message_input_item(item, function_calls)
        if message is not None:
            messages.append(message)
            last_item_was_user = message["role"] == "user" and bool(
                message["content"]
            )
            continue

        raise ValueError(
            f"Cursor Agent received unsupported Responses input type: {item_type or role or 'unknown'}."
        )

    # A tool result is a continuation of the interrupted Cursor turn.
    if saw_function_call_output and not last_item_was_user:
        continuation_message: dict[str, Any] = {"role": "user", "content": ""}
        if function_call_output_ends_input:
            continuation_message = {
                "role": "user",
                "content": _CURSOR_TOOL_CONTINUATION_CUE,
                _CURSOR_TOOL_CONTINUATION_CUE_MARKER: True,
            }
        messages.append(continuation_message)
    if not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def _cursor_messages_with_result_tool_calls(
    messages: list[dict[str, Any]],
    tool_calls: list[Any],
) -> list[dict[str, Any]]:
    replay_messages = copy.deepcopy(messages)
    _validate_cursor_returned_tool_calls(tool_calls)
    if replay_messages and replay_messages[-1].get(
        _CURSOR_TOOL_CONTINUATION_CUE_MARKER
    ):
        replay_messages.pop()
    function_calls: dict[str, str] = {}
    for message in replay_messages:
        if message.get("role") == "assistant":
            _remember_cursor_message_tool_calls(
                message.get("tool_calls") or message.get("toolCalls"),
                function_calls,
            )
    for tool_call in tool_calls:
        replay_messages.append(
            _cursor_function_call_message(tool_call, function_calls)
        )
    return replay_messages


def _cursor_responses_response_body(
    *,
    model: str,
    result: Any,
) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    output: list[dict[str, Any]] = []
    if result.text:
        output.append(
            {
                "id": f"msg_{response_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.text,
                        "annotations": [],
                    }
                ],
            }
        )
    for tool_call in result.tool_calls:
        output.append(
            {
                "id": str(tool_call.get("id") or f"fc_{tool_call['call_id']}"),
                "type": "function_call",
                "status": "completed",
                "call_id": str(tool_call["call_id"]),
                "name": str(tool_call["name"]),
                "arguments": str(tool_call.get("arguments") or "{}"),
            }
        )
    body: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": result.text,
    }
    if result.usage:
        body["usage"] = dict(result.usage)
    if result.provider_metadata:
        body["provider_specific_fields"] = dict(result.provider_metadata)
    return body


def _validate_cursor_result_and_consume_replay_state(
    *,
    result: Any,
    previous_response_id: Any,
    replay_state: Optional[dict[str, Any]],
) -> None:
    from litellm.llms.cursor_agent.connect import CursorConnectError

    try:
        result.validate_terminal()
        if result.exec_server_messages and not (
            result.tool_calls or result.text or result.turn_ended
        ):
            raise CursorConnectError(
                "Cursor Agent returned execServerMessage without a replayable "
                "tool-call event; the bounded fresh-Run continuation bridge "
                "cannot represent that server message.",
                status_code=502,
            )
        if not result.tool_calls and not result.text and not result.turn_ended:
            raise CursorConnectError(
                "Cursor Agent Connect completed without text, a function call, or turnEnded.",
                status_code=502,
            )
    except Exception as exc:
        if (
            isinstance(previous_response_id, str)
            and replay_state is not None
            and not _cursor_replay_failure_is_transient(exc)
        ):
            _consume_cursor_replay_state(
                previous_response_id,
                expected_state=replay_state,
            )
        raise

    if isinstance(previous_response_id, str) and replay_state is not None:
        _consume_cursor_replay_state(
            previous_response_id,
            expected_state=replay_state,
        )


async def _perform_codex_auto_agent_cursor_agent_request(  # noqa: PLR0915
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
    """Execute a Codex Responses request through Cursor Agent Connect."""
    _ = (
        endpoint,
        fastapi_response,
        user_api_key_dict,
        target_url,
        api_key,
        forward_headers,
    )
    from fastapi.responses import Response, StreamingResponse

    from litellm.llms.cursor_agent.common_utils import (
        build_run_request,
        run_url,
    )
    from litellm.llms.cursor_agent.connect import (
        CursorAgentConnectClient,
        CursorConnectError,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
        _responses_sse_from_repaired_response_body,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.anthropic_adapter_calls import (
        _build_adapted_route_rollup_kwargs,
        _emit_adapted_route_access_log,
        _record_adapted_completed_route_rollup_turn,
    )

    if candidate.get("route_family") != "codex_cursor_agent_aiserver_adapter":
        raise ValueError(
            "Cursor Agent Codex candidates require "
            "codex_cursor_agent_aiserver_adapter."
        )

    request_body = dict(candidate_body)
    litellm_metadata = request_body.get("litellm_metadata")
    rollup_kwargs = _build_adapted_route_rollup_kwargs(
        litellm_metadata if isinstance(litellm_metadata, dict) else {}
    )
    candidate_api_base = candidate.get("api_base")
    cursor_url = run_url(
        str(candidate_api_base)
        if isinstance(candidate_api_base, str) and candidate_api_base.strip()
        else None
    )
    _emit_adapted_route_access_log(
        request=request,
        target_url=cursor_url,
        request_body=request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label="Cursor Agent",
    )
    replay_state: Optional[dict[str, Any]] = None
    previous_response_id = request_body.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        replay_state = _peek_cursor_replay_state(previous_response_id)
    request_tools = request_body.get("tools")
    if not isinstance(request_tools, list) and isinstance(replay_state, dict):
        request_tools = replay_state.get("tools")

    retained_session = (
        replay_state.get("retained_session")
        if isinstance(replay_state, dict)
        else None
    )
    cursor_tool_outputs = _cursor_function_call_outputs(request_body)
    if cursor_tool_outputs and retained_session is None:
        _raise_cursor_session_continuation_unavailable(
            previous_response_id=(
                previous_response_id
                if isinstance(previous_response_id, str)
                else None
            ),
            replay_state=replay_state if isinstance(replay_state, dict) else None,
        )

    messages = _responses_input_to_cursor_messages(
        request_body,
        prior_messages=(
            replay_state.get("messages")
            if isinstance(replay_state, dict)
            and isinstance(replay_state.get("messages"), list)
            else None
        ),
    )
    optional_params: dict[str, Any] = {}
    if retained_session is not None:
        _consume_cursor_replay_state(
            previous_response_id,
            expected_state=replay_state,
            close_retained_session=False,
        )
        try:
            result = await retained_session.continue_with_tool_outputs(
                cursor_tool_outputs
            )
            _validate_cursor_result_and_consume_replay_state(
                result=result,
                previous_response_id=None,
                replay_state=None,
            )
        except Exception:
            await retained_session.aclose()
            raise

        try:
            replay_messages = _cursor_messages_with_result_tool_calls(
                messages,
                result.tool_calls,
            )
        except _CursorPostEgressOutputError as exc:
            _raise_cursor_agent_alias_error(exc=exc, candidate=candidate)
        model = str(candidate.get("model") or request_body.get("model") or "")
        response_body = _cursor_responses_response_body(
            model=model,
            result=result,
        )
        if result.tool_calls:
            next_session = result.retained_session
            _store_cursor_replay_state(
                response_body["id"],
                messages=replay_messages,
                tools=request_tools if isinstance(request_tools, list) else [],
                retained_session=next_session,
            )
            if next_session is None:
                await retained_session.aclose()
        else:
            await retained_session.aclose()
        _record_adapted_completed_route_rollup_turn(
            rollup_kwargs,
            adapter_label="Cursor Agent",
        )
        if bool(request_body.get("stream")):
            return StreamingResponse(
                _responses_sse_from_repaired_response_body(
                    response_body,
                    request_body=request_body,
                ),
                media_type="text/event-stream",
            )
        return Response(
            content=json.dumps(response_body, ensure_ascii=False),
            media_type="application/json",
        )

    if isinstance(request_tools, list):
        optional_params["tools"] = request_tools
    for source_names, cursor_name in (
        (("message_id", "messageId"), "message_id"),
        (("conversation_id", "conversationId"), "conversation_id"),
        (("conversation_group_id", "conversationGroupId"), "conversation_group_id"),
        (("run_id", "runId"), "run_id"),
        (("agent_session_id", "agentSessionId"), "agent_session_id"),
    ):
        value = next(
            (
                request_body.get(source_name)
                for source_name in source_names
                if request_body.get(source_name)
            ),
            None,
        )
        if value:
            optional_params[cursor_name] = value
    cursor_request = build_run_request(
        model=str(candidate.get("model") or request_body.get("model") or ""),
        messages=messages,
        optional_params=optional_params,
    )

    extra_headers: dict[str, str] = {}
    request_headers = getattr(request, "headers", None)
    if request_headers is not None:
        request_id = request_headers.get("x-request-id")
        if request_id:
            extra_headers["x-request-id"] = str(request_id)

    client = CursorAgentConnectClient()
    try:
        result = await client.run(
            cursor_request,
            url=cursor_url,
            extra_headers=extra_headers,
            stop_on_tool_call=True,
            retain_on_tool_call=True,
        )
    except _CursorPostEgressOutputError:
        raise
    except Exception as exc:
        if (
            isinstance(previous_response_id, str)
            and replay_state is not None
            and not _cursor_replay_failure_is_transient(
                exc,
                transport_failure=not isinstance(exc, CursorConnectError),
            )
        ):
            _consume_cursor_replay_state(
                previous_response_id,
                expected_state=replay_state,
            )
        raise

    _validate_cursor_result_and_consume_replay_state(
        result=result,
        previous_response_id=previous_response_id,
        replay_state=replay_state,
    )
    try:
        _validate_cursor_returned_tool_calls(result.tool_calls)
        model = str(candidate.get("model") or request_body.get("model") or "")
        response_body = _cursor_responses_response_body(model=model, result=result)
    except _CursorPostEgressOutputError as exc:
        _raise_cursor_agent_alias_error(exc=exc, candidate=candidate)
    if result.tool_calls:
        try:
            replay_messages = _cursor_messages_with_result_tool_calls(
                messages,
                result.tool_calls,
            )
        except _CursorPostEgressOutputError as exc:
            _raise_cursor_agent_alias_error(exc=exc, candidate=candidate)
        _store_cursor_replay_state(
            response_body["id"],
            messages=replay_messages,
            tools=request_tools if isinstance(request_tools, list) else [],
            retained_session=result.retained_session,
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="Cursor Agent",
    )
    if bool(request_body.get("stream")):
        return StreamingResponse(
            _responses_sse_from_repaired_response_body(
                response_body,
                request_body=request_body,
            ),
            media_type="text/event-stream",
        )
    return Response(
        content=json.dumps(response_body, ensure_ascii=False),
        media_type="application/json",
    )


_CURSOR_PROVIDER_ERROR_FIELDS = frozenset(
    {
        "code",
        "message",
        "model",
        "model_id",
        "model_name",
        "requested_model",
        "status",
        "type",
        "upstream_model",
    }
)
_CURSOR_PROVIDER_ERROR_NESTED_FIELDS = frozenset(
    {"context", "data", "details", "metadata"}
)


def _sanitize_cursor_provider_error_body(body: Any) -> Optional[dict[str, Any]]:
    """Keep only bounded structured fields from a real Cursor error body."""
    parsed_body: Any = body
    if isinstance(body, (bytes, bytearray)):
        try:
            parsed_body = json.loads(bytes(body).decode("utf-8", errors="ignore"))
        except (TypeError, ValueError):
            return None
    elif isinstance(body, str):
        try:
            parsed_body = json.loads(body)
        except (TypeError, ValueError):
            return None
    if not isinstance(parsed_body, Mapping):
        return None

    provider_error = parsed_body.get("error")
    if not isinstance(provider_error, Mapping):
        provider_error = parsed_body
    if not any(field in provider_error for field in _CURSOR_PROVIDER_ERROR_FIELDS):
        return None

    def _sanitize_value(value: Any, *, limit: int) -> Any:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            return sanitize_credential_error_message(value, limit=limit)
        return None

    sanitized_error: dict[str, Any] = {}
    for field in _CURSOR_PROVIDER_ERROR_FIELDS:
        value = provider_error.get(field)
        if field == "message":
            sanitized = _sanitize_value(value, limit=512)
        elif field == "code":
            sanitized = _sanitize_value(value, limit=128)
        elif field in {
            "model",
            "model_id",
            "model_name",
            "requested_model",
            "upstream_model",
        }:
            sanitized = _sanitize_value(value, limit=256)
        else:
            sanitized = _sanitize_value(value, limit=128)
        if sanitized is not None:
            sanitized_error[field] = sanitized

    for field in _CURSOR_PROVIDER_ERROR_NESTED_FIELDS:
        nested = provider_error.get(field)
        if not isinstance(nested, Mapping):
            continue
        sanitized_nested: dict[str, Any] = {}
        for nested_field in _CURSOR_PROVIDER_ERROR_FIELDS:
            value = nested.get(nested_field)
            sanitized = _sanitize_value(
                value,
                limit=512 if nested_field == "message" else 256,
            )
            if sanitized is not None:
                sanitized_nested[nested_field] = sanitized
        if sanitized_nested:
            sanitized_error[field] = sanitized_nested

    if not sanitized_error:
        return None
    return {"error": sanitized_error}


def _raise_cursor_agent_alias_error(  # noqa: PLR0915
    *,
    exc: Exception,
    candidate: dict[str, Any],
) -> None:
    """Translate a Cursor Agent failure while preserving upstream semantics.

    Pre-egress request-conversion ``ValueError`` failures and Cursor Connect
    protocol rejections of unsupported exec/interactive operations are
    deterministic candidate ineligibility, not upstream 502s: they map to
    the ``aawm_codex_auto_agent_candidate_ineligible`` contract so the
    candidate loop records a no-cooldown ineligibility instead of a
    transient upstream retry.
    Transport/upstream 500/502/503/529 keep their status and map to the
    existing transient/timeout classification so a Cursor blip advances to
    the next candidate instead of publishing a durable candidate cooldown.
    HTTP 408/504 map to the existing timeout classification.
    Provider-returned auth and other non-transient 4xx failures use the
    terminal, non-coolable provider-error contract. Local failures without
    provider attribution preserve their sanitized error shape but remain
    terminal without advancing or cooling the candidate.
    """
    from litellm.llms.cursor_agent.connect import CursorConnectProtocolError
    from litellm.proxy._types import ProxyException

    message = str(getattr(exc, "message", None) or exc)
    model = str(candidate.get("model") or "")
    route_family = str(candidate.get("route_family") or "")
    provider_returned = (
        getattr(exc, "_aawm_provider_returned", False) is True
        and not isinstance(exc, CursorConnectProtocolError)
        and not isinstance(exc, _CursorPostEgressOutputError)
        and not getattr(exc, _CURSOR_SESSION_CONTINUATION_FAILURE_MARKER, False)
    )
    status_code = int(getattr(exc, "status_code", 502) or 502)
    if status_code < 400 or status_code > 599:
        status_code = 502
    cursor_sanitized_provider_error = (
        _sanitize_cursor_provider_error_body(getattr(exc, "body", None))
        if provider_returned or 400 <= status_code < 500
        else None
    )
    provider_error_fields = (
        cursor_sanitized_provider_error.get("error", {})
        if isinstance(cursor_sanitized_provider_error, dict)
        else {}
    )
    provider_message = provider_error_fields.get("message")
    mapped_provider_message = (
        provider_message
        if isinstance(provider_message, str) and provider_message
        else None
    )
    cursor_sanitized_proto_structure = (
        _sanitize_cursor_proto_structure_for_telemetry(getattr(exc, "body", None))
        if isinstance(exc, CursorConnectProtocolError)
        else None
    )

    def _set_mapped_detail(
        proxy_exc: ProxyException,
        error: dict[str, Any],
    ) -> None:
        mapped_error = dict(error)
        if provider_error_fields:
            mapped_error.update(copy.deepcopy(provider_error_fields))
        mapped_detail: dict[str, Any] = {"error": mapped_error}
        if cursor_sanitized_provider_error is not None:
            provider_body = copy.deepcopy(cursor_sanitized_provider_error)
            mapped_detail["cursor_sanitized_provider_error"] = provider_body
            setattr(proxy_exc, "body", copy.deepcopy(provider_body))
        if cursor_sanitized_proto_structure is not None:
            structure = copy.deepcopy(cursor_sanitized_proto_structure)
            mapped_detail[_CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD] = structure
            setattr(
                proxy_exc,
                _CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD,
                copy.deepcopy(structure),
            )
        setattr(proxy_exc, "detail", mapped_detail)

    attempted_provider_call: Optional[bool] = None
    ineligibility_summary = ""
    if getattr(exc, _CURSOR_SESSION_CONTINUATION_FAILURE_MARKER, False):
        detail_message = (
            "Cursor Agent candidate is ineligible: the requested tool-output "
            "continuation has no live retained session; "
            f"model={model} route_family={route_family}; {message}"
        )
        proxy_exc = ProxyException(
            message=detail_message,
            type="invalid_request_error",
            param="model",
            code=409,
        )
        setattr(proxy_exc, "status_code", 409)
        setattr(proxy_exc, "candidate_status", "ineligible")
        setattr(proxy_exc, "ineligibility_reason", "preflight_skipped")
        setattr(proxy_exc, "failure_phase", "cursor_session_continuation")
        setattr(proxy_exc, "attempted_provider_call", False)
        setattr(proxy_exc, _CURSOR_SESSION_CONTINUATION_FAILURE_MARKER, True)
        replay_state = getattr(exc, _CURSOR_REPLAY_STATE_FIELD, None)
        if isinstance(replay_state, dict):
            setattr(proxy_exc, _CURSOR_REPLAY_STATE_FIELD, replay_state)
        _set_mapped_detail(
            proxy_exc,
            {
                "message": detail_message,
                "code": "aawm_codex_auto_agent_candidate_ineligible",
            },
        )
        raise proxy_exc from exc
    if isinstance(exc, CursorConnectProtocolError) and message.startswith(
        (
            "Cursor Agent requested unsupported external exec field ",
            "Cursor Agent requested unsupported local exec operation ",
            "Cursor Agent requested an unsupported interactive client response.",
        )
    ):
        attempted_provider_call = True
        ineligibility_summary = (
            "the Cursor Agent session requested an unsupported operation"
        )
    if isinstance(exc, _CursorPostEgressOutputError):
        error_message = (
            "Cursor Agent request failed after provider Run; "
            f"model={model} route_family={route_family}; {message}"
        )
        proxy_exc = ProxyException(
            message=error_message,
            type="upstream_error",
            param="model",
            code=502,
        )
        setattr(proxy_exc, "status_code", 502)
        setattr(proxy_exc, "candidate_status", "retryable")
        setattr(proxy_exc, "failure_phase", "candidate_post_egress_normalization")
        setattr(proxy_exc, "attempted_provider_call", True)
        _set_mapped_detail(
            proxy_exc,
            {
                "message": error_message,
                "code": "upstream_transient_internal",
                "type": "upstream_error",
            },
        )
        raise proxy_exc from exc
    elif isinstance(exc, ValueError):
        attempted_provider_call = False
        ineligibility_summary = "pre-egress Cursor request conversion failed"
    if attempted_provider_call is not None:
        detail_message = (
            "Cursor Agent candidate is ineligible: "
            f"{ineligibility_summary}; model={model} "
            f"route_family={route_family}; {message}"
        )
        proxy_exc = ProxyException(
            message=detail_message,
            type="invalid_request_error",
            param="model",
            code=400,
        )
        setattr(proxy_exc, "status_code", 400)
        setattr(proxy_exc, "candidate_status", "ineligible")
        setattr(proxy_exc, "ineligibility_reason", "unsupported")
        setattr(proxy_exc, "failure_phase", "candidate_preflight")
        setattr(proxy_exc, "attempted_provider_call", attempted_provider_call)
        _set_mapped_detail(
            proxy_exc,
            {
                "message": detail_message,
                "code": "aawm_codex_auto_agent_candidate_ineligible",
            },
        )
        raise proxy_exc from exc

    if status_code in (408, 504):
        error_code = "upstream_timeout"
        error_type = "upstream_timeout"
    elif status_code in (500, 502, 503, 529):
        error_code = "upstream_transient_internal"
        error_type = "upstream_error"
    elif provider_returned and status_code == 429:
        error_code = "rate_limited"
        error_type = "upstream_error"
    elif provider_returned or 400 <= status_code < 500:
        error_code = "provider_terminal_error"
        error_type = (
            "authentication_error"
            if status_code in {401, 403}
            else "upstream_error"
        )
    else:
        error_code = "aawm_codex_auto_agent_candidate_unavailable"
        error_type = "upstream_error"
    detail = (
        f"cursor_agent request failed; model={model} "
        f"route_family={route_family}; status={status_code}; {message}"
    )
    mapped_message = mapped_provider_message or detail
    proxy_exc = ProxyException(
        message=mapped_message,
        type=error_type,
        param="model",
        code=status_code,
    )
    setattr(proxy_exc, "status_code", status_code)
    if provider_returned:
        setattr(proxy_exc, "attempted_provider_call", True)
        setattr(proxy_exc, "_aawm_provider_returned", True)
    elif 400 <= status_code < 500:
        setattr(proxy_exc, "attempted_provider_call", False)
    _set_mapped_detail(
        proxy_exc,
        {
            "message": mapped_message,
            "type": error_type,
            "code": error_code,
        },
    )
    raise proxy_exc from exc


def _build_codex_cohere_adapter_request_body(
    *,
    prepared_request_body: Payload,
    adapter_model: str,
    upstream_model: str,
    config: "_aawm_adapter_config.AnthropicCompletionAdapterConfig",
) -> Payload:
    request_body = dict(prepared_request_body)
    metadata = dict(request_body.get("litellm_metadata") or {})
    tags = list(metadata.get("tags") or [])
    for tag in (
        f"route:{config.route_family}",
        config.tag_prefix,
        f"{config.tag_prefix}-model:{adapter_model}",
        f"{config.tag_prefix}-target:{config.target_endpoint_label}",
    ):
        if tag not in tags:
            tags.append(tag)

    spans = list(metadata.get("langfuse_spans") or [])
    spans.append(
        {
            "name": config.span_name,
            "metadata": {
                "requested_model": prepared_request_body.get("model"),
                "adapter_model": adapter_model,
                "upstream_model": upstream_model,
                "stream": bool(prepared_request_body.get("stream")),
            },
        }
    )
    metadata.update(
        {
            "tags": tags,
            "langfuse_spans": spans,
            "passthrough_route_family": config.route_family,
            "route_family": config.route_family,
            "codex_cohere_adapter_model": adapter_model,
            "codex_cohere_upstream_model": upstream_model,
            "codex_adapter_model": adapter_model,
            "codex_adapter_original_model": prepared_request_body.get("model"),
            "codex_adapter_target_endpoint": config.target_endpoint_label,
        }
    )
    request_body["litellm_metadata"] = metadata
    return request_body


async def _prepare_codex_cohere_chat_completions_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    _ = request, use_alias_candidate_probe
    normalized_model = adapter_model.strip() if isinstance(adapter_model, str) else ""
    provider, separator, upstream_model = normalized_model.partition("/")
    if (
        provider != _CODEX_AUTO_AGENT_COHERE_PROVIDER
        or not separator
        or not upstream_model
        or "/" in upstream_model
    ):
        raise ValueError(
            "Codex Cohere adapter requires a cohere/<model> candidate."
        )

    from litellm.proxy.pass_through_endpoints.providers.cohere import (
        runtime as _cohere_runtime,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    adapted_request_body, _adapted_custom_tools = (
        _adapt_codex_custom_tools_to_functions_from_request_body(
            prepared_request_body
        )
    )
    adapted_request_body, _adapted_namespace_tools = (
        _adapt_codex_namespace_tools_to_functions_from_request_body(
            adapted_request_body
        )
    )
    adapted_request_body, _tool_description_patch_events = (
        _apply_codex_tool_description_patches_to_request_body(adapted_request_body)
    )
    adapted_request_body, _unsupported_hosted_tools = (
        _drop_unsupported_codex_hosted_tools_from_request_body(adapted_request_body)
    )
    adapted_request_body, _unsupported_input_items = (
        _drop_unsupported_codex_input_items_from_request_body(adapted_request_body)
    )
    adapted_request_body, _removed_tool_choice = (
        _drop_tool_choice_without_tools_from_request_body(adapted_request_body)
    )
    config = _aawm_adapter_config.CODEX_COHERE_CHAT_COMPLETIONS
    request_body = _build_codex_cohere_adapter_request_body(
        prepared_request_body=adapted_request_body,
        adapter_model=normalized_model,
        upstream_model=upstream_model,
        config=config,
    )
    request_input = request_body.get("input", "")
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        {
            key: value
            for key, value in request_body.items()
            if key not in {"input", "model", "litellm_metadata"}
        },
    )
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=upstream_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider=_CODEX_AUTO_AGENT_COHERE_PROVIDER,
        stream=bool(request_body.get("stream")),
        metadata=litellm_metadata,
    )
    completion_kwargs = _strip_strict_from_cohere_completion_tools(completion_kwargs)
    completion_kwargs.update(
        {
            "metadata": litellm_metadata,
            "custom_llm_provider": _CODEX_AUTO_AGENT_COHERE_PROVIDER,
            "num_retries": 0,
        }
    )
    previous_response_id = responses_api_request.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        completion_kwargs = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id=previous_response_id,
            litellm_completion_request=completion_kwargs,
        )

    target_url = _cohere_runtime._get_cohere_target_base()
    api_key = _cohere_runtime._require_cohere_api_key()
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=target_url,
        headers={"Authorization": f"Bearer {api_key}"},
        credential_family=config.credential_family,
        expected_target_family=config.expected_target_family,
    )
    return _aawm_adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=request_body,
        target_url=target_url,
        api_key=api_key,
        api_base=target_url,
        client_requested_stream=bool(request_body.get("stream")),
        perform_kwargs={
            "completion_kwargs": completion_kwargs,
            "request_input": request_input,
            "responses_api_request": responses_api_request,
            "litellm_metadata": litellm_metadata,
            "upstream_model": upstream_model,
        },
    )


async def _perform_codex_cohere_chat_completions_adapter_call(
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
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    _ = config, adapter_model
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(str(target_url)))
    _watermark_intake = None
    try:
        _watermark_intake = getattr(getattr(request, "state", None), "watermark_intake", None)
    except Exception:
        _watermark_intake = None
    _watermark_metadata = litellm_metadata if isinstance(litellm_metadata, dict) else {}
    _watermark_egress = apply_request_watermark_egress(
        body=completion_kwargs,
        intake=_watermark_intake,
        config=_get_runtime_text_watermark_config(),
        endpoint=_watermark_endpoint_from_path("chat/completions", target_url),
        direction="request",
        metadata=_watermark_metadata,
        litellm_metadata=_watermark_metadata,
    )
    if isinstance(getattr(_watermark_egress, "body", None), dict):
        completion_kwargs = _watermark_egress.body
    completion_response = await litellm.acompletion(
        **completion_kwargs,
        api_key=api_key,
        api_base=api_base,
        litellm_metadata=litellm_metadata,
        proxy_server_request={
            "headers": {},
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
                    custom_llm_provider=_CODEX_AUTO_AGENT_COHERE_PROVIDER,
                    litellm_metadata=litellm_metadata,
                )
            ),
            media_type="text/event-stream",
        )

    responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        chat_completion_response=completion_response,
        request_input=request_input,
        responses_api_request=responses_api_request,
    )
    return _build_responses_response_from_adapter_response(
        responses_api_response,
        request_body=(
            prepared_request_body
            if isinstance(prepared_request_body, dict)
            else (
                {"litellm_metadata": litellm_metadata}
                if isinstance(litellm_metadata, dict)
                else None
            )
        ),
    )


async def _handle_codex_cohere_chat_completions_adapter_route(
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
        plan = await _prepare_codex_cohere_chat_completions_adapter_route(**kwargs)
        metadata = plan.perform_kwargs.get("litellm_metadata")
        if not isinstance(metadata, dict):
            metadata = plan.prepared_request_body.get("litellm_metadata")
        rollup_kwargs.update(
            _build_adapted_route_rollup_kwargs(
                metadata if isinstance(metadata, dict) else {}
            )
        )
        _annotate_request_scope_for_adapted_access_log(request, plan.target_url)
        provider_bound_body = plan.perform_kwargs.get("completion_kwargs")
        if not isinstance(provider_bound_body, dict):
            provider_bound_body = None
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="Cohere",
            provider_bound_body=provider_bound_body,
        )
        return plan

    response = await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_and_emit_route_log,
        perform=_perform_codex_cohere_chat_completions_adapter_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    intake_context = _build_malformed_tool_call_intake_context(
        request,
        prepared_request_body,
        adapter="codex_cohere_chat_completions_adapter",
        provider=_CODEX_AUTO_AGENT_COHERE_PROVIDER,
    )
    if isinstance(response, StreamingResponse):
        response = _bind_responses_stream_timeout_terminalizer(
            response,
            adapter_model=adapter_model,
            adapter_label="Cohere",
            provider=_CODEX_AUTO_AGENT_COHERE_PROVIDER,
            intake_context=intake_context,
            rollup_kwargs=rollup_kwargs,
        )
    validated_response = await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter="codex_cohere_chat_completions_adapter",
        adapter_label="Cohere",
        intake_context=intake_context,
        request_body=prepared_request_body,
    )
    if isinstance(validated_response, StreamingResponse):
        return _record_adapted_completed_route_rollup_after_stream(
            validated_response,
            rollup_kwargs,
            adapter_label="Cohere",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="Cohere",
    )
    return validated_response



def _build_codex_nvidia_adapter_request_body(
    *,
    prepared_request_body: Payload,
    adapter_model: str,
    upstream_model: str,
    config: "_aawm_adapter_config.AnthropicCompletionAdapterConfig",
) -> Payload:
    request_body = dict(prepared_request_body)
    metadata = dict(request_body.get("litellm_metadata") or {})
    tags = list(metadata.get("tags") or [])
    for tag in (
        f"route:{config.route_family}",
        config.tag_prefix,
        f"{config.tag_prefix}-model:{adapter_model}",
        f"{config.tag_prefix}-target:{config.target_endpoint_label}",
    ):
        if tag not in tags:
            tags.append(tag)

    spans = list(metadata.get("langfuse_spans") or [])
    spans.append(
        {
            "name": config.span_name,
            "metadata": {
                "requested_model": prepared_request_body.get("model"),
                "adapter_model": adapter_model,
                "upstream_model": upstream_model,
                "stream": bool(prepared_request_body.get("stream")),
            },
        }
    )
    metadata.update(
        {
            "tags": tags,
            "langfuse_spans": spans,
            "passthrough_route_family": config.route_family,
            "route_family": config.route_family,
            "codex_nvidia_adapter_model": adapter_model,
            "codex_nvidia_upstream_model": upstream_model,
            "codex_adapter_model": adapter_model,
            "codex_adapter_original_model": prepared_request_body.get("model"),
            "codex_adapter_target_endpoint": config.target_endpoint_label,
        }
    )
    request_body["litellm_metadata"] = metadata
    return request_body


async def _prepare_codex_nvidia_completion_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    _ = request, use_alias_candidate_probe
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
        normalize_nvidia_completion_adapter_model_name,
        nvidia_completion_adapter_upstream_model,
    )
    from litellm.proxy.pass_through_endpoints.providers.nvidia import (
        runtime as _nvidia_runtime,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    canonical_model = normalize_nvidia_completion_adapter_model_name(adapter_model)
    upstream_model = nvidia_completion_adapter_upstream_model(adapter_model)
    if canonical_model is None or not upstream_model:
        raise ValueError(
            "Codex NVIDIA adapter requires a nvidia/<model> candidate."
        )

    adapted_request_body, _adapted_custom_tools = (
        _adapt_codex_custom_tools_to_functions_from_request_body(
            prepared_request_body
        )
    )
    adapted_request_body, _adapted_namespace_tools = (
        _adapt_codex_namespace_tools_to_functions_from_request_body(
            adapted_request_body
        )
    )
    adapted_request_body, _tool_description_patch_events = (
        _apply_codex_tool_description_patches_to_request_body(adapted_request_body)
    )
    adapted_request_body, _unsupported_hosted_tools = (
        _drop_unsupported_codex_hosted_tools_from_request_body(adapted_request_body)
    )
    adapted_request_body, _unsupported_input_items = (
        _drop_unsupported_codex_input_items_from_request_body(adapted_request_body)
    )
    adapted_request_body, _removed_tool_choice = (
        _drop_tool_choice_without_tools_from_request_body(adapted_request_body)
    )
    config = _aawm_adapter_config.CODEX_NVIDIA_COMPLETION
    request_body = _build_codex_nvidia_adapter_request_body(
        prepared_request_body=adapted_request_body,
        adapter_model=canonical_model,
        upstream_model=upstream_model,
        config=config,
    )
    request_input = request_body.get("input", "")
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        {
            key: value
            for key, value in request_body.items()
            if key not in {"input", "model", "litellm_metadata"}
        },
    )
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=upstream_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider=config.custom_llm_provider,
        stream=bool(request_body.get("stream")),
        metadata=litellm_metadata,
    )
    completion_kwargs.update(
        {
            "metadata": litellm_metadata,
            "custom_llm_provider": config.custom_llm_provider,
            "num_retries": 0,
        }
    )
    previous_response_id = responses_api_request.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        completion_kwargs = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id=previous_response_id,
            litellm_completion_request=completion_kwargs,
        )

    api_key = _nvidia_runtime._get_anthropic_adapter_nvidia_api_key()
    if not api_key:
        exc = ProxyException(
            message=(
                "Codex NVIDIA adapter requests require "
                "'AAWM_NVIDIA_API_KEY', 'NVIDIA_NIM_API_KEY', or "
                "'NVIDIA_API_KEY' in environment."
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

    target_base_url = _nvidia_runtime._get_anthropic_adapter_nvidia_target_base()
    api_base = f"{str(target_base_url).rstrip('/')}/v1"
    target_url = f"{api_base}/chat/completions"
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=target_url,
        headers={"Authorization": f"Bearer {api_key}"},
        credential_family=config.credential_family,
        expected_target_family=config.expected_target_family,
    )
    return _aawm_adapter_driver.CompletionAdapterRoutePlan(
        config=config,
        prepared_request_body=request_body,
        target_url=target_url,
        api_key=api_key,
        api_base=api_base,
        client_requested_stream=bool(request_body.get("stream")),
        perform_kwargs={
            "completion_kwargs": completion_kwargs,
            "request_input": request_input,
            "responses_api_request": responses_api_request,
            "litellm_metadata": litellm_metadata,
            "upstream_model": upstream_model,
        },
    )


async def _perform_codex_nvidia_completion_adapter_call(
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
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    _ = adapter_model
    custom_llm_provider = config.custom_llm_provider
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(str(target_url)))
    _watermark_intake = None
    try:
        _watermark_intake = getattr(getattr(request, "state", None), "watermark_intake", None)
    except Exception:
        _watermark_intake = None
    _watermark_metadata = litellm_metadata if isinstance(litellm_metadata, dict) else {}
    _watermark_egress = apply_request_watermark_egress(
        body=completion_kwargs,
        intake=_watermark_intake,
        config=_get_runtime_text_watermark_config(),
        endpoint=_watermark_endpoint_from_path("chat/completions", target_url),
        direction="request",
        metadata=_watermark_metadata,
        litellm_metadata=_watermark_metadata,
    )
    if isinstance(getattr(_watermark_egress, "body", None), dict):
        completion_kwargs = _watermark_egress.body
    completion_response = await litellm.acompletion(
        **completion_kwargs,
        api_key=api_key,
        api_base=api_base,
        litellm_metadata=litellm_metadata,
        proxy_server_request={
            "headers": {},
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
                    custom_llm_provider=custom_llm_provider,
                    litellm_metadata=litellm_metadata,
                )
            ),
            media_type="text/event-stream",
        )

    responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        chat_completion_response=completion_response,
        request_input=request_input,
        responses_api_request=responses_api_request,
    )
    return _build_responses_response_from_adapter_response(
        responses_api_response,
        request_body=(
            prepared_request_body
            if isinstance(prepared_request_body, dict)
            else (
                {"litellm_metadata": litellm_metadata}
                if isinstance(litellm_metadata, dict)
                else None
            )
        ),
    )


async def _handle_codex_nvidia_completion_adapter_route(
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
        plan = await _prepare_codex_nvidia_completion_adapter_route(**kwargs)
        metadata = plan.perform_kwargs.get("litellm_metadata")
        if not isinstance(metadata, dict):
            metadata = plan.prepared_request_body.get("litellm_metadata")
        rollup_kwargs.update(
            _build_adapted_route_rollup_kwargs(
                metadata if isinstance(metadata, dict) else {}
            )
        )
        _annotate_request_scope_for_adapted_access_log(request, plan.target_url)
        provider_bound_body = plan.perform_kwargs.get("completion_kwargs")
        if not isinstance(provider_bound_body, dict):
            provider_bound_body = None
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="NVIDIA",
            provider_bound_body=provider_bound_body,
        )
        return plan

    response = await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_and_emit_route_log,
        perform=_perform_codex_nvidia_completion_adapter_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    nvidia_provider = globals().get("_CODEX_AUTO_AGENT_NVIDIA_PROVIDER", "nvidia")
    intake_context = _build_malformed_tool_call_intake_context(
        request,
        prepared_request_body,
        adapter="codex_nvidia_completion_adapter",
        provider=nvidia_provider,
    )
    if isinstance(response, StreamingResponse):
        response = _bind_responses_stream_timeout_terminalizer(
            response,
            adapter_model=adapter_model,
            adapter_label="NVIDIA",
            provider=nvidia_provider,
            intake_context=intake_context,
            rollup_kwargs=rollup_kwargs,
        )
    validated_response = await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter="codex_nvidia_completion_adapter",
        adapter_label="NVIDIA",
        intake_context=intake_context,
        request_body=prepared_request_body,
    )
    if isinstance(validated_response, StreamingResponse):
        return _record_adapted_completed_route_rollup_after_stream(
            validated_response,
            rollup_kwargs,
            adapter_label="NVIDIA",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="NVIDIA",
    )
    return validated_response



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
    # OPENAI-007: legacy history may collapse provider tool ids into item id.
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.direct_openai_function_call_history import (
        normalize_direct_openai_legacy_function_call_history_ids,
    )

    request_body = normalize_direct_openai_legacy_function_call_history_ids(
        request_body
    )
    # Ingress drop keys off the caller/alias model id. Alias names such as
    # ``work`` / ``expert`` / ``sota`` are not cost-map keys, so Ohmypi
    # ``max_output_tokens`` survives until this resolved Codex candidate.
    (
        request_body,
        _codex_unsupported_request_params,
    ) = _drop_unsupported_codex_request_params_from_request_body(request_body)
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
        guard_openai_encrypted_reasoning_egress,
    )

    request_body, _encrypted_reasoning_disposition = (
        guard_openai_encrypted_reasoning_egress(
            request_body,
            url=target_url,
        )
    )
    request_body = {
        **request_body,
        "store": False,
        "stream": True,
    }
    is_streaming_request = bool(request_body.get("stream"))
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
        provider_returned = bool(getattr(exc, "_aawm_provider_returned", False))
        native_openai_error_context = {
            "target_url": target_url,
            "custom_llm_provider": litellm.LlmProviders.OPENAI.value,
            "provider_returned": provider_returned,
            "expected_model": request_body.get("model"),
        }
        if _codex_native_openai_candidate_unavailable_detail(
            exc,
            **native_openai_error_context,
        ) is not None:
            _raise_codex_native_openai_auto_agent_candidate_unavailable(
                exc,
                **native_openai_error_context,
            )
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
    response = _maybe_wrap_xai_passthrough_responses_stream(
        response,
        request=request,
        request_body=request_body,
        route_family="codex_auto_agent_grok_native_responses",
        resolved_model=grok_prepared_body.get("model") or request_body.get("model"),
    )
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
    canonical_request_body = copy.deepcopy(request_body)
    (
        adapted_request_body,
        _adapted_custom_tools,
    ) = _adapt_codex_custom_tools_to_functions_from_request_body(request_body)
    (
        adapted_request_body,
        _adapted_namespace_tools,
    ) = _adapt_codex_namespace_tools_to_functions_from_request_body(
        adapted_request_body
    )
    (
        adapted_request_body,
        _tool_description_patch_events,
    ) = _apply_codex_tool_description_patches_to_request_body(
        adapted_request_body
    )
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
    response = _maybe_wrap_xai_passthrough_responses_stream(
        response,
        request=request,
        request_body=canonical_request_body,
        route_family="codex_auto_agent_xai_oauth_responses",
        resolved_model=oa_xai_prepared_body.get("model")
        or canonical_request_body.get("model"),
    )
    return await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=str(oa_xai_prepared_body.get("model") or canonical_request_body.get("model") or "unknown-model"),
        adapter="codex_auto_agent_xai_oauth_responses",
        adapter_label="xAI OAuth",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            canonical_request_body,
            adapter="codex_auto_agent_xai_oauth_responses",
            upstream_url=str(updated_url),
            provider="xai",
        ),
        request_body=canonical_request_body,
    )


def _bind_responses_stream_timeout_terminalizer(
    response: StreamingResponse,
    *,
    adapter_model: str,
    adapter_label: str,
    provider: str,
    intake_context: Optional[dict[str, Any]],
    rollup_kwargs: dict[str, Any],
    stream_error_callback: Optional[Any] = None,
) -> StreamingResponse:
    finalized = False

    async def _terminalize(
        exc: BaseException,
        progress: Any,
    ) -> list[Any]:
        nonlocal finalized
        if finalized:
            return []
        finalized = True
        if not isinstance(exc, Exception):
            raise exc

        import time
        from datetime import datetime, timedelta, timezone

        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )
        from litellm.types.passthrough_endpoints.pass_through_endpoints import (
            EndpointType,
        )

        identity = intake_context if isinstance(intake_context, dict) else {}
        failure_context = {
            key: identity.get(key)
            for key in (
                "request_id",
                "litellm_call_id",
                "trace_id",
                "session_id",
                "endpoint",
                "upstream_url",
                "model_alias",
                "route_family",
            )
            if identity.get(key) is not None
        }
        failure_context.update(
            {
                "provider": provider,
                "model": adapter_model,
                "adapter_label": adapter_label,
                "failure_kind": "streaming_upstream_read_timeout",
                "stream_failure_stage": "stream_interrupted_after_first_byte",
                "stream_chunks_seen": progress.chunk_count,
                "stream_bytes_seen": progress.total_emitted_bytes,
                "stream_hidden_retry_safe": False,
                "status_code": 504,
            }
        )
        if progress.last_emission_timestamp is not None:
            idle_seconds = max(
                0.0,
                time.monotonic() - progress.last_emission_timestamp,
            )
            failure_context["stream_last_emission_at"] = (
                datetime.now(timezone.utc) - timedelta(seconds=idle_seconds)
            ).isoformat()
            failure_context["stream_idle_ms"] = round(idle_seconds * 1000.0, 3)

        litellm_params = rollup_kwargs.get("litellm_params")
        metadata = (
            litellm_params.get("metadata")
            if isinstance(litellm_params, dict)
            and isinstance(litellm_params.get("metadata"), dict)
            else None
        )
        if isinstance(metadata, dict):
            if metadata.get("aawm_stream_terminal_emitted"):
                return []
            metadata.update(failure_context)
            metadata["aawm_stream_interrupted"] = True
            metadata["aawm_stream_terminal_emitted"] = True
            metadata["aawm_route_rollup_turn_suppressed"] = True
            metadata["aawm_route_rollup_turn_recorded"] = True

        _emit_aawm_terminal_error(
            {
                "event_type": "responses_stream_terminal",
                "endpoint": identity.get("endpoint"),
                "alias_family": identity.get("route_family") or adapter_label,
                "alias_model": (
                    identity.get("model_alias")
                    or identity.get("requested_model_alias")
                    or adapter_model
                ),
                "selected_provider": identity.get("provider") or provider,
                "selected_model": adapter_model,
                "selected_route": (
                    identity.get("route_family") or identity.get("upstream_url")
                ),
                "status_code": 504,
                "error_code": "streaming_upstream_read_timeout",
                "failure_class": "streaming_upstream_read_timeout",
                "failure_phase": failure_context["stream_failure_stage"],
                "attempted_provider_call": True,
                "redispatch_required": False,
                "terminal_outcome": "failed",
                "fallback_result": "none",
                "attempt_count": (
                    metadata.get("attempt_count")
                    if isinstance(metadata, dict)
                    else None
                ),
                "correlation_id": (
                    identity.get("litellm_call_id")
                    or identity.get("request_id")
                    or identity.get("trace_id")
                ),
            },
            marker=metadata if isinstance(metadata, dict) else identity,
        )

        PassThroughStreamingHandler._record_post_first_byte_stream_terminal_rollup(
            success_handler_kwargs=rollup_kwargs,
            failure_context=failure_context,
            exc=exc,
        )

        terminal_event = (
            stream_error_callback(exc)
            if stream_error_callback is not None
            else None
        )
        if terminal_event is not None:
            rendered = (
                terminal_event.decode("utf-8", errors="replace")
                if isinstance(terminal_event, bytes)
                else str(terminal_event)
            )
            chunks = [terminal_event]
            if "data: [DONE]" not in rendered:
                chunks.append("data: [DONE]\n\n")
            return chunks

        return PassThroughStreamingHandler._build_post_first_byte_terminal_stream_chunks(
            endpoint_type=EndpointType.OPENAI,
            url_route="https://api.openai.com/v1/responses",
            custom_llm_provider=provider,
            failure_context=failure_context,
            exc=exc,
        )

    return _aawm_alias_streaming._bind_stream_timeout_terminalizer(
        response,
        _terminalize,
    )


async def _validate_codex_auto_agent_openrouter_responses_stream(
    response: StreamingResponse,
    *,
    adapter_model: str,
    intake_context: Optional[dict[str, Any]] = None,
    request_body: Optional[dict[str, Any]] = None,
) -> StreamingResponse:
    event_summaries: list[dict[str, Any]] = []
    peek = await _aawm_alias_streaming.peek_streaming_response(
        response,
        max_chunks=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS,
        max_bytes=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES,
        terminalizer=_aawm_alias_streaming._get_stream_timeout_terminalizer(
            response
        ),
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
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            stamp_route_identity_in_sse_chunk,
        )

        identity_request_body = request_body if isinstance(request_body, dict) else None
        for raw_chunk in peek.buffered_chunks:
            yield stamp_route_identity_in_sse_chunk(
                raw_chunk,
                request_body=identity_request_body,
            )

    return StreamingResponse(
        _replay_iterator(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )


def _raise_codex_auto_agent_missing_credential_preflight(
    *,
    message: str,
    exc: Optional[Exception] = None,
) -> Any:
    from litellm.proxy._types import ProxyException

    proxy_exc = ProxyException(
        message=message,
        type="invalid_request_error",
        param="model",
        code=400,
    )
    setattr(proxy_exc, "status_code", 400)
    setattr(proxy_exc, "candidate_status", "ineligible")
    setattr(proxy_exc, "ineligibility_reason", "preflight_skipped")
    setattr(proxy_exc, "failure_phase", "candidate_preflight")
    setattr(proxy_exc, "attempted_provider_call", False)
    setattr(
        proxy_exc,
        "detail",
        {
            "error": {
                "message": message,
                "code": "aawm_codex_auto_agent_candidate_ineligible",
            }
        },
    )
    raise proxy_exc from exc


async def _load_codex_auto_agent_opencode_zen_api_key(
    *,
    use_alias_candidate_probe: bool,
    load_api_key: Any,
) -> str:
    try:
        return await load_api_key(
            use_alias_candidate_probe=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        if use_alias_candidate_probe:
            _raise_codex_auto_agent_missing_credential_preflight(
                message=(
                    "OpenCode Zen auto-agent candidate requires a valid "
                    f"OpenCode API-key credential: {exc}"
                ),
                exc=exc,
            )
        raise


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
            request_body=request_body if isinstance(request_body, dict) else None,
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
    adapted_request_body, _unsupported_request_params = (
        _drop_unsupported_codex_request_params_from_request_body(
            adapted_request_body
        )
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
    _watermark_intake = None
    try:
        _watermark_intake = getattr(getattr(request, "state", None), "watermark_intake", None)
    except Exception:
        _watermark_intake = None
    _watermark_metadata = litellm_metadata if isinstance(litellm_metadata, dict) else {}
    _watermark_egress = apply_request_watermark_egress(
        body=completion_kwargs,
        intake=_watermark_intake,
        config=_get_runtime_text_watermark_config(),
        endpoint=_watermark_endpoint_from_path("chat/completions", target_url),
        direction="request",
        metadata=_watermark_metadata,
        litellm_metadata=_watermark_metadata,
    )
    if isinstance(getattr(_watermark_egress, "body", None), dict):
        completion_kwargs = _watermark_egress.body
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
    _identity_request_body = (
        prepared_request_body
        if isinstance(prepared_request_body, dict)
        else (
            {"litellm_metadata": litellm_metadata}
            if isinstance(litellm_metadata, dict)
            else None
        )
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
                ),
                request_body=_identity_request_body,
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
    return _build_responses_response_from_adapter_response(
        responses_api_response,
        request_body=_identity_request_body,
    )


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
        provider_bound_body = plan.perform_kwargs.get("completion_kwargs")
        if not isinstance(provider_bound_body, dict):
            provider_bound_body = None
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="Kimi Code",
            provider_bound_body=provider_bound_body,
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
    intake_context = _build_malformed_tool_call_intake_context(
        request,
        prepared_request_body,
        adapter="codex_kimi_chat_completions_adapter",
        provider="kimi_code",
    )
    if isinstance(response, StreamingResponse):
        response = _bind_responses_stream_timeout_terminalizer(
            response,
            adapter_model=adapter_model,
            adapter_label="Kimi Code",
            provider="kimi_code",
            intake_context=intake_context,
            rollup_kwargs=rollup_kwargs,
        )
    validated_response = await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter="codex_kimi_chat_completions_adapter",
        adapter_label="Kimi Code",
        intake_context=intake_context,
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
    _watermark_intake = None
    try:
        _watermark_intake = getattr(getattr(request, "state", None), "watermark_intake", None)
    except Exception:
        _watermark_intake = None
    _watermark_metadata = litellm_metadata if isinstance(litellm_metadata, dict) else {}
    _watermark_egress = apply_request_watermark_egress(
        body=completion_kwargs,
        intake=_watermark_intake,
        config=_get_runtime_text_watermark_config(),
        endpoint=_watermark_endpoint_from_path("chat/completions", target_url),
        direction="request",
        metadata=_watermark_metadata,
        litellm_metadata=_watermark_metadata,
    )
    if isinstance(getattr(_watermark_egress, "body", None), dict):
        completion_kwargs = _watermark_egress.body
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
                        _stream_response_body,
                        request_body=(
                            prepared_request_body
                            if isinstance(prepared_request_body, dict)
                            else (
                                {"litellm_metadata": litellm_metadata}
                                if isinstance(litellm_metadata, dict)
                                else None
                            )
                        ),
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
            _responses_sse_from_repaired_response_body(
                _stream_response_body,
                request_body=(
                    prepared_request_body
                    if isinstance(prepared_request_body, dict)
                    else (
                        {"litellm_metadata": litellm_metadata}
                        if isinstance(litellm_metadata, dict)
                        else None
                    )
                ),
            ),
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
                responses_api_response,
                request_body=(
                    prepared_request_body
                    if isinstance(prepared_request_body, dict)
                    else (
                        {"litellm_metadata": litellm_metadata}
                        if isinstance(litellm_metadata, dict)
                        else None
                    )
                ),
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
    return _build_responses_response_from_adapter_response(
        responses_api_response,
        request_body=(
            prepared_request_body
            if isinstance(prepared_request_body, dict)
            else (
                {"litellm_metadata": litellm_metadata}
                if isinstance(litellm_metadata, dict)
                else None
            )
        ),
    )


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
        provider_bound_body = plan.perform_kwargs.get("completion_kwargs")
        if not isinstance(provider_bound_body, dict):
            provider_bound_body = None
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="Alibaba Token Plan",
            provider_bound_body=provider_bound_body,
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


async def _prepare_codex_zai_coding_plan_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    prepared_request_body = _zai_coding_plan_adapters.normalize_zai_coding_plan_custom_tool_outputs(
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
    return await _zai_coding_plan_adapters.prepare_codex_zai_coding_plan_adapter_route(
        request=request,
        prepared_request_body=adapted_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _perform_codex_zai_coding_plan_adapter_call(
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
    """Execute Coding Plan chat completions and reuse the standard Responses wrapper."""
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
                    custom_llm_provider=litellm.LlmProviders.ZAI_CODING_PLAN.value,
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
    # Codex /v1/responses contract: never forward the chat.completion object tag.
    if getattr(responses_api_response, "object", None) != "response":
        responses_api_response.object = "response"
    return _build_responses_response_from_adapter_response(
        responses_api_response,
        request_body=(
            prepared_request_body
            if isinstance(prepared_request_body, dict)
            else (
                {"litellm_metadata": litellm_metadata}
                if isinstance(litellm_metadata, dict)
                else None
            )
        ),
    )


async def _handle_codex_zai_coding_plan_adapter_route(
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
        plan = await _prepare_codex_zai_coding_plan_adapter_route(**kwargs)
        metadata = plan.perform_kwargs.get("litellm_metadata")
        if not isinstance(metadata, dict):
            metadata = plan.prepared_request_body.get("litellm_metadata")
        rollup_kwargs.update(_build_adapted_route_rollup_kwargs(metadata if isinstance(metadata, dict) else {}))
        _annotate_request_scope_for_adapted_access_log(request, plan.target_url)
        provider_bound_body = plan.perform_kwargs.get("completion_kwargs")
        if not isinstance(provider_bound_body, dict):
            provider_bound_body = None
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="Z.AI Coding Plan",
            provider_bound_body=provider_bound_body,
        )
        return plan

    response = await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_and_emit_route_log,
        perform=_perform_codex_zai_coding_plan_adapter_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    intake_context = _build_malformed_tool_call_intake_context(
        request,
        prepared_request_body,
        adapter="codex_zai_coding_plan_chat_completions_adapter",
        provider="zai_coding_plan",
    )
    if isinstance(response, StreamingResponse):
        response = _bind_responses_stream_timeout_terminalizer(
            response,
            adapter_model=adapter_model,
            adapter_label="Z.AI Coding Plan",
            provider="zai_coding_plan",
            intake_context=intake_context,
            rollup_kwargs=rollup_kwargs,
        )
    validated_response = await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter="codex_zai_coding_plan_chat_completions_adapter",
        adapter_label="Z.AI Coding Plan",
        intake_context=intake_context,
        request_body=prepared_request_body,
    )
    if isinstance(validated_response, StreamingResponse):
        return _record_adapted_completed_route_rollup_after_stream(
            validated_response,
            rollup_kwargs,
            adapter_label="Z.AI Coding Plan",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="Z.AI Coding Plan",
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
    litellm_metadata wins if already present. Alias probes also default to
    ``drop`` so unsupported Codex-native tools do not terminate the whole
    alias before the provider can consume retained function tools.
    """
    _existing_metadata = prepared_request_body.get("litellm_metadata")
    _existing_mode = (
        _existing_metadata.get("opencode_zen_unsupported_tools_mode")
        if isinstance(_existing_metadata, dict)
        else None
    )
    if _existing_mode is not None:
        return prepared_request_body

    if use_alias_candidate_probe:
        prepared_request_body = dict(prepared_request_body)
        _meta = dict(prepared_request_body.get("litellm_metadata") or {})
        _meta["opencode_zen_unsupported_tools_mode"] = "drop"
        prepared_request_body["litellm_metadata"] = _meta
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
    _watermark_metadata = litellm_metadata if isinstance(litellm_metadata, dict) else {}
    _watermark_egress = apply_request_watermark_egress(
        body=completion_call_kwargs,
        config=_get_runtime_text_watermark_config(),
        endpoint=_watermark_endpoint_from_path("chat/completions"),
        direction="request",
        metadata=_watermark_metadata,
        litellm_metadata=_watermark_metadata,
    )
    if isinstance(getattr(_watermark_egress, "body", None), dict):
        completion_call_kwargs = _watermark_egress.body
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
    api_key = await _load_codex_auto_agent_opencode_zen_api_key(
        use_alias_candidate_probe=use_alias_candidate_probe,
        load_api_key=_load_opencode_zen_api_key_for_candidate,
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
    completion_call_kwargs = _build_opencode_zen_completion_call_kwargs(
        completion_kwargs=completion_kwargs,
        api_key=api_key,
        target_base_url=target_base_url,
        litellm_metadata=litellm_metadata,
        request=request,
        use_alias_candidate_probe=use_alias_candidate_probe,
        request_body=request_body,
    )
    # D1-521: log the exact final translated completion kwargs while retaining
    # the original request body/model label for access-log model labeling.
    _emit_adapted_route_access_log(
        request=request,
        target_url=target_url,
        request_body=request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label="OpenCode Zen",
        provider_bound_body=completion_kwargs,
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
        stream_response = _bind_responses_stream_timeout_terminalizer(
            stream_response,
            adapter_model=adapter_model,
            adapter_label="OpenCode Zen",
            provider="opencode",
            intake_context=_build_malformed_tool_call_intake_context(
                request,
                request_body,
                adapter="codex_opencode_zen_completion_adapter",
                upstream_url=target_url,
                provider="opencode",
            ),
            rollup_kwargs=rollup_kwargs,
            stream_error_callback=_opencode_zen_direct_stream_terminal_error,
        )
        # D1-574: peek for pre-first-byte streaming failures
        try:
            peek = await _aawm_alias_streaming.peek_streaming_response(
                stream_response,
                max_chunks=1,
                max_bytes=_OPENCODE_ZEN_DIRECT_PEEK_MAX_BYTES,
                terminalizer=_aawm_alias_streaming._get_stream_timeout_terminalizer(
                    stream_response
                ),
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
    return _build_responses_response_from_adapter_response(
        responses_api_response,
        request_body=(
            request_body
            if isinstance(request_body, dict)
            else (
                {"litellm_metadata": litellm_metadata}
                if isinstance(litellm_metadata, dict)
                else None
            )
        ),
    )


_OPENCODE_GO_CHAT_COMPLETIONS_ROUTE = "/zen/go/v1/chat/completions"
_OPENCODE_GO_TOOLS_INDEX_RE = re.compile(r"tools\[(\d+)\]")


def _opencode_go_tool_type(tool: Any) -> Optional[str]:
    if not isinstance(tool, dict):
        return None
    tool_type = tool.get("type")
    if isinstance(tool_type, str) and tool_type.strip():
        return tool_type.strip()
    function = tool.get("function")
    if isinstance(function, dict):
        return "function"
    return None


def _opencode_go_tool_types(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    summarized: list[str] = []
    for tool in tools:
        tool_type = _opencode_go_tool_type(tool)
        summarized.append(tool_type or "unknown")
    return summarized


def _extract_opencode_go_offending_tool_index(message: str) -> Optional[int]:
    match = _OPENCODE_GO_TOOLS_INDEX_RE.search(message)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _sanitize_opencode_go_error_text(message: Any, *, api_key: Any = None) -> str:
    text = str(message or "")
    if isinstance(api_key, str) and api_key:
        text = text.replace(api_key, "[REDACTED]")
    return sanitize_credential_error_message(text, limit=512)


def _build_opencode_go_provider_rejection_evidence(
    *,
    target_url: Any,
    exc: BaseException,
    advertised_tools: Any = None,
    completion_tools: Any = None,
    api_key: Any = None,
) -> dict[str, Any]:
    advertised_types = _opencode_go_tool_types(advertised_tools)
    completion_types = _opencode_go_tool_types(completion_tools)
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int) or status_code <= 0:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    raw_message = getattr(exc, "message", None)
    if not raw_message:
        raw_message = getattr(exc, "detail", None) or str(exc)
    sanitized_message = _sanitize_opencode_go_error_text(
        raw_message,
        api_key=api_key,
    )
    offending_index = _extract_opencode_go_offending_tool_index(sanitized_message)
    offending_type = None
    if offending_index is not None:
        if 0 <= offending_index < len(advertised_types):
            offending_type = advertised_types[offending_index]
        elif 0 <= offending_index < len(completion_types):
            offending_type = completion_types[offending_index]
    target = str(target_url or "")
    return {
        "route": "codex_opencode_go_adapter",
        "target_url_family": _OPENCODE_GO_CHAT_COMPLETIONS_ROUTE,
        "target_url": (
            target
            if _OPENCODE_GO_CHAT_COMPLETIONS_ROUTE in target
            else _OPENCODE_GO_CHAT_COMPLETIONS_ROUTE
        ),
        "error": {
            "status": status_code if isinstance(status_code, int) else None,
            "type": type(exc).__name__,
            "message": sanitized_message,
        },
        "tool_count": len(advertised_types),
        "tool_types": advertised_types,
        "completion_tool_count": len(completion_types),
        "completion_tool_types": completion_types,
        "offending_index": offending_index,
        "offending_type": offending_type,
    }


def _record_opencode_go_provider_rejection_evidence(
    request: Any,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    recorded = dict(evidence)
    state = getattr(request, "state", None)
    if state is not None:
        try:
            setattr(state, "opencode_go_provider_rejection_evidence", recorded)
        except Exception:
            pass
        extra = getattr(state, "opencode_go_logger_extra", None)
        if not isinstance(extra, dict):
            extra = {}
            try:
                setattr(state, "opencode_go_logger_extra", extra)
            except Exception:
                extra = {}
        extra["opencode_go_provider_rejection"] = recorded
    return recorded


def _raise_opencode_go_alias_candidate_upstream_timeout(
    exc: Exception,
) -> None:
    from litellm.proxy._types import ProxyException

    proxy_exc = ProxyException(
        message="OpenCode Go alias candidate timed out upstream.",
        type="upstream_timeout",
        param="model",
        code=504,
    )
    setattr(proxy_exc, "status_code", 504)
    setattr(
        proxy_exc,
        "detail",
        {
            "error": {
                "message": proxy_exc.message,
                "type": "upstream_timeout",
                "code": "upstream_timeout",
            }
        },
    )
    raise proxy_exc from exc


async def _handle_codex_opencode_go_adapter_route(  # noqa: PLR0915
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Any,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    import asyncio

    import litellm
    from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen.constants import (
        _OPENCODE_GO_FREE_MODELS,
    )
    from fastapi.responses import StreamingResponse
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.request_build import (
        _is_empty_success_responses_body as _go_is_empty_success_responses_body,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
        _responses_sse_from_repaired_response_body,
        _serialize_responses_adapter_response as _serialize_go_response,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.stream_collect import (
        _build_empty_success_responses_diagnostic as _go_build_empty_success_diagnostic,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_candidate_calls import (
        _OPENCODE_GO_ALIAS_CANDIDATE_TIMEOUT_SECONDS as _go_probe_timeout_seconds,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    _ = endpoint, fastapi_response, user_api_key_dict
    _ = (
        not use_alias_candidate_probe
        and adapter_model in _OPENCODE_GO_FREE_MODELS
    )
    request_body = dict(prepared_request_body)
    request_body["model"] = adapter_model
    advertised_tools = (
        list(request_body.get("tools") or [])
        if isinstance(request_body.get("tools"), list)
        else []
    )
    # Restore dispatchable tool identities before the chat-completion
    # transformation. Match the Cohere/OpenRouter Responses prep order:
    # adapt custom tools, flatten namespace tools, apply description
    # patches, drop unsupported hosted tools and input items, then clean
    # incompatible tool_choice. Console Go chat-completions accept only
    # function tools. Retain the canonical (namespaced/custom) body so
    # tool_call_restore can reconstruct Codex custom_tool_call items.
    canonical_request_body = request_body
    (
        request_body,
        _adapted_custom_tools,
    ) = _adapt_codex_custom_tools_to_functions_from_request_body(request_body)
    (
        request_body,
        _adapted_namespace_tools,
    ) = _adapt_codex_namespace_tools_to_functions_from_request_body(request_body)
    (
        request_body,
        _tool_description_patch_events,
    ) = _apply_codex_tool_description_patches_to_request_body(request_body)
    (
        request_body,
        _unsupported_hosted_tools,
    ) = _drop_unsupported_codex_hosted_tools_from_request_body(request_body)
    (
        request_body,
        _unsupported_input_items,
    ) = _drop_unsupported_codex_input_items_from_request_body(request_body)
    (
        request_body,
        _removed_tool_choice,
    ) = _drop_tool_choice_without_tools_from_request_body(request_body)
    request_input = request_body.get("input", "")
    responses_api_request = {
        key: value
        for key, value in request_body.items()
        if key not in {"input", "model", "litellm_metadata"}
    }
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    # Console Go chat-completions must be complete-upstream. Forwarding
    # client stream=True into acompletion returns a stream wrapper; the
    # Responses transform then emits output:[] / output_tokens=0 and the
    # adapter cools ox-alpha-free as empty success. Client stream is
    # reconstructed from the completed body below.
    client_requested_stream = bool(request_body.get("stream"))
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=adapter_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider="openai",
        stream=False,
        metadata=litellm_metadata,
    )
    completion_kwargs["model"] = adapter_model
    completion_kwargs["stream"] = False
    target_base_url = _get_opencode_go_target_base()
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
        adapter_label="OpenCode Go",
        provider_bound_body=completion_kwargs,
    )
    completion_call_kwargs = {
        **completion_kwargs,
        "api_key": api_key,
        "api_base": f"{target_base_url.rstrip('/')}/v1",
        "litellm_metadata": litellm_metadata,
    }
    perform = globals().get("_perform_opencode_zen_completion_call")
    try:
        if callable(perform):
            completion_awaitable = perform(
                completion_call_kwargs=completion_call_kwargs,
                litellm_metadata=litellm_metadata,
                accepted_trace_user_id=None,
                is_known_free_direct=False,
            )
        else:
            completion_awaitable = litellm.acompletion(**completion_call_kwargs)
        if use_alias_candidate_probe:
            completion_response = await asyncio.wait_for(
                completion_awaitable,
                timeout=_go_probe_timeout_seconds,
            )
        else:
            completion_response = await completion_awaitable
    except Exception as exc:
        evidence = _build_opencode_go_provider_rejection_evidence(
            target_url=target_url,
            exc=exc,
            advertised_tools=advertised_tools,
            completion_tools=completion_kwargs.get("tools"),
            api_key=api_key,
        )
        _record_opencode_go_provider_rejection_evidence(request, evidence)
        if use_alias_candidate_probe:
            if (
                isinstance(exc, asyncio.TimeoutError)
                or evidence["error"]["status"] == 408
            ):
                _raise_opencode_go_alias_candidate_upstream_timeout(exc)
            from litellm.proxy.pass_through_endpoints.providers.common import (
                _opencode_go_candidate_unavailable_detail,
                _raise_opencode_go_auto_agent_candidate_unavailable,
            )

            if _opencode_go_candidate_unavailable_detail(exc) is not None:
                _raise_opencode_go_auto_agent_candidate_unavailable(exc)
        raise
    if isinstance(completion_response, dict):
        from litellm.types.utils import ModelResponse

        completion_response = ModelResponse(**completion_response)
    responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        chat_completion_response=completion_response,
        request_input=request_input,
        responses_api_request=responses_api_request,
    )
    try:
        responses_api_response.object = "response"
    except Exception:
        pass
    serialized = _serialize_go_response(responses_api_response)
    try:
        response_body = json.loads(serialized)
    except (TypeError, ValueError, NameError):
        response_body = None
    if not isinstance(response_body, dict):
        import json as _json

        try:
            response_body = _json.loads(serialized)
        except (TypeError, ValueError):
            response_body = {"id": getattr(completion_response, "id", "resp_opencode_go")}
    response_body["object"] = "response"
    restore_custom = globals().get("_restore_adapted_custom_tool_calls_in_response_body")
    if callable(restore_custom):
        restored_body, restored_custom_count, _custom_tool_adapter_error = restore_custom(
            response_body,
            request_body=canonical_request_body,
            adapter_model=adapter_model,
        )
        if restored_custom_count:
            response_body = restored_body
    restore_namespace = globals().get(
        "_restore_adapted_namespace_tool_calls_in_response_body"
    )
    if callable(restore_namespace):
        restored_body, restored_namespace_count = restore_namespace(
            response_body,
            request_body=canonical_request_body,
            adapter_model=adapter_model,
        )
        if restored_namespace_count:
            response_body = restored_body
    _is_codex_auto_agent_empty_success_responses_body.__globals__.setdefault(
        "_is_empty_success_responses_body",
        _go_is_empty_success_responses_body,
    )
    _raise_codex_auto_agent_empty_success_response.__globals__.setdefault(
        "_build_empty_success_responses_diagnostic",
        _go_build_empty_success_diagnostic,
    )
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_opencode_go_adapter",
            adapter_label="OpenCode Go",
        )
    # ACCESS replacement is registered at emit time. A completed Go
    # turn still has to record the rollup so the 60s flush emits
    # litellm#Ohmypi / litellm#Codex headers. Without this, a
    # successful PONG / child-spawn leaves a 0-byte docker-logs window.
    if client_requested_stream:
        identity_request_body = (
            canonical_request_body
            if isinstance(canonical_request_body, dict)
            else request_body
            if isinstance(request_body, dict)
            else None
        )
        return _record_adapted_completed_route_rollup_after_stream(
            StreamingResponse(
                _responses_sse_from_repaired_response_body(
                    response_body,
                    request_body=identity_request_body,
                ),
                media_type="text/event-stream",
            ),
            rollup_kwargs,
            adapter_label="OpenCode Go",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="OpenCode Go",
    )
    return _build_responses_response_from_adapter_response(
        response_body,
        request_body=(
            canonical_request_body
            if isinstance(canonical_request_body, dict)
            else request_body
            if isinstance(request_body, dict)
            else None
        ),
    )


async def _handle_codex_nous_chat_completions_adapter_route(  # noqa: PLR0915
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Any,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    import json as _json

    import litellm
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
        _serialize_responses_adapter_response as _serialize_nous_response,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )
    from litellm.secret_managers.credential_error_sanitizer import (
        sanitize_credential_error_message,
    )
    from litellm.secret_managers.hermes_nous_auth import load_nous_invoke_jwt

    _ = endpoint, fastapi_response, user_api_key_dict
    if use_alias_candidate_probe and (
        bool(prepared_request_body.get("stream"))
        or bool(prepared_request_body.get("tools"))
        or bool(prepared_request_body.get("tool_choice"))
    ):
        from litellm.proxy._types import ProxyException

        incompatibility = ValueError(
            "Nous stealth/ox-alpha cannot accept the stock Codex "
            "streaming and tool request contract."
        )
        message = (
            "Nous auto-agent candidate is incompatible with the "
            "requested Codex streaming or tool contract."
        )
        exc = ProxyException(
            message=message,
            type="invalid_request_error",
            param="model",
            code=400,
        )
        setattr(exc, "candidate_status", "ineligible")
        setattr(exc, "ineligibility_reason", "contract_incompatible")
        setattr(exc, "failure_phase", "candidate_preflight")
        setattr(exc, "attempted_provider_call", False)
        setattr(
            exc,
            "detail",
            {
                "error": {
                    "message": message,
                    "code": "aawm_codex_auto_agent_candidate_ineligible",
                }
            },
        )
        raise exc from incompatibility
    request_body = dict(prepared_request_body)
    request_body["model"] = adapter_model
    request_input = request_body.get("input", "")
    responses_api_request = {
        key: value
        for key, value in request_body.items()
        if key not in {"input", "model", "litellm_metadata"}
    }
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=adapter_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider="nous",
        stream=False,
        metadata=litellm_metadata,
    )
    completion_kwargs["model"] = adapter_model
    target_url = "https://inference-api.nousresearch.com/v1/chat/completions"
    try:
        api_key = load_nous_invoke_jwt()
    except Exception:
        from litellm.proxy._types import ProxyException

        message = (
            "Nous Codex auto-agent candidate is unavailable: "
            "Hermes Nous Portal invoke JWT could not be loaded."
        )
        exc = ProxyException(
            message=message,
            type="rate_limit_error",
            param="model",
            code=429,
        )
        setattr(
            exc,
            "detail",
            {
                "error": {
                    "message": message,
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                }
            },
        )
        raise exc from None
    custom_headers = BaseOpenAIPassThroughHandler._assemble_headers(
        api_key=api_key,
        request=request,
    )
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=target_url,
        headers=custom_headers,
        credential_family="nous",
        expected_target_family="nous",
    )
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(target_url))
    rollup_kwargs = _build_adapted_route_rollup_kwargs(litellm_metadata)
    _emit_adapted_route_access_log(
        request=request,
        target_url=target_url,
        request_body=request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label="Nous",
        provider_bound_body=completion_kwargs,
    )
    completion_call_kwargs = {
        **completion_kwargs,
        "api_key": api_key,
        "api_base": "https://inference-api.nousresearch.com/v1",
        "litellm_metadata": litellm_metadata,
    }
    try:
        completion_response = await litellm.acompletion(**completion_call_kwargs)
    except Exception as exc:
        status_code = int(getattr(exc, "status_code", 0) or 0)
        if status_code == 0:
            response = getattr(exc, "response", None)
            status_code = int(getattr(response, "status_code", 0) or 0)
        raw = str(getattr(exc, "detail", "") or exc)
        if api_key:
            raw = raw.replace(api_key, "[REDACTED]")
        detail = sanitize_credential_error_message(raw)
        if status_code == 0:
            raise type(exc)(detail) from None
        setattr(exc, "status_code", status_code)
        setattr(exc, "detail", detail)
        if hasattr(exc, "args"):
            try:
                exc.args = (detail, *exc.args[1:])
            except Exception:
                pass
        raise
    if isinstance(completion_response, dict):
        from litellm.types.utils import ModelResponse

        completion_response = ModelResponse(**completion_response)
    responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        chat_completion_response=completion_response,
        request_input=request_input,
        responses_api_request=responses_api_request,
    )
    try:
        responses_api_response.object = "response"
    except Exception:
        pass
    serialized = _serialize_nous_response(responses_api_response)
    try:
        response_body = json.loads(serialized)
    except (TypeError, ValueError, NameError):
        response_body = None
    if not isinstance(response_body, dict):
        try:
            response_body = _json.loads(serialized)
        except (TypeError, ValueError):
            response_body = {"id": getattr(completion_response, "id", "resp_nous")}
    response_body["object"] = "response"
    return _build_responses_response_from_adapter_response(
        response_body,
        request_body=(
            request_body
            if isinstance(request_body, dict)
            else (
                {"litellm_metadata": litellm_metadata}
                if isinstance(litellm_metadata, dict)
                else None
            )
        ),
    )


async def _perform_codex_auto_agent_openrouter_completion_request(  # noqa: PLR0915
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
        _raise_codex_auto_agent_missing_credential_preflight(
            message=(
                "OpenRouter Codex auto-agent candidate requires " "AAWM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
            ),
        )

    if isinstance(request_body, dict):
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
            strip_route_identity_from_request_body,
        )

        stripped_request_body = strip_route_identity_from_request_body(request_body)
        if stripped_request_body is not request_body and isinstance(
            stripped_request_body, dict
        ):
            request_body.clear()
            request_body.update(stripped_request_body)
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
    # Restore dispatchable tool identities before the chat-completion
    # transformation. Match the Cohere/Kimi/Alibaba Responses prep order:
    # adapt custom tools, flatten namespace tools, apply description
    # patches, drop unsupported hosted tools and input items, then clean
    # incompatible tool_choice. Upstream chat-completions (including
    # OpenRouter -> Cohere) accept only function tools.
    # Retain the canonical (namespaced) body for response validation so
    # tool_call_restore can reconstruct the original namespace map.
    canonical_request_body = request_body
    (
        request_body,
        _adapted_custom_tools,
    ) = _adapt_codex_custom_tools_to_functions_from_request_body(request_body)
    (
        request_body,
        _adapted_namespace_tools,
    ) = _adapt_codex_namespace_tools_to_functions_from_request_body(request_body)
    (
        request_body,
        _tool_description_patch_events,
    ) = _apply_codex_tool_description_patches_to_request_body(request_body)
    (
        request_body,
        _unsupported_hosted_tools,
    ) = _drop_unsupported_codex_hosted_tools_from_request_body(request_body)
    (
        request_body,
        _unsupported_input_items,
    ) = _drop_unsupported_codex_input_items_from_request_body(request_body)
    (
        request_body,
        _removed_tool_choice,
    ) = _drop_tool_choice_without_tools_from_request_body(request_body)
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
    # D1-521: pass the exact final translated/clamped completion kwargs as
    # provider_bound_body while retaining request_body for model label.
    _emit_adapted_route_access_log(
        request=request,
        target_url=target_url,
        request_body=request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label="OpenRouter chat-completions",
        provider_bound_body=completion_kwargs,
    )
    _watermark_intake = None
    try:
        _watermark_intake = getattr(getattr(request, "state", None), "watermark_intake", None)
    except Exception:
        _watermark_intake = None
    _watermark_metadata = litellm_metadata if isinstance(litellm_metadata, dict) else {}
    _watermark_egress = apply_request_watermark_egress(
        body=completion_kwargs,
        intake=_watermark_intake,
        config=_get_runtime_text_watermark_config(),
        endpoint=_watermark_endpoint_from_path("chat/completions", target_url),
        direction="request",
        metadata=_watermark_metadata,
        litellm_metadata=_watermark_metadata,
    )
    if isinstance(getattr(_watermark_egress, "body", None), dict):
        completion_kwargs = _watermark_egress.body

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
        stream_response = _bind_responses_stream_timeout_terminalizer(
            stream_response,
            adapter_model=adapter_model,
            adapter_label="OpenRouter chat-completions",
            provider="openrouter",
            intake_context=intake_context,
            rollup_kwargs=rollup_kwargs,
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
    built_response = _build_responses_response_from_adapter_response(
        responses_api_response,
        request_body=(
            canonical_request_body
            if isinstance(canonical_request_body, dict)
            else request_body if isinstance(request_body, dict) else None
        ),
    )
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
