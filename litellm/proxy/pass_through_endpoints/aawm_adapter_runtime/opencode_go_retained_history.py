"""OpenCode Go ``previous_response_id`` retained-history contract.

Chat Completions egress cannot honor a Responses continuation id, and native
Go Responses cannot look up LiteLLM-encoded ids. Materialize retained history
through the standard session handler, or reject before any provider call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Never, Optional, Sequence

from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    OPENCODE_GO_PROVIDER,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity import (
    extract_account_identity_from_context,
    get_request_session_owner_lease,
)
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)

_GO_ROUTE_FAMILY = "codex_opencode_go_adapter"
_GO_API_BASE_MARKERS = ("/zen/go/",)
_FAILURE_PHASE = "opencode_go_retained_history"

_UNKNOWN_MESSAGE = "OpenCode Go previous_response_id was not found."
_UNAVAILABLE_MESSAGE = "OpenCode Go previous_response_id is unavailable."
_CROSS_OWNER_MESSAGE = (
    "OpenCode Go cannot mix retained history across providers or accounts."
)
_INVALID_TOOL_MESSAGE = (
    "OpenCode Go retained history is not a valid tool-call/tool-result sequence."
)
_INCONSISTENT_MESSAGE = (
    "OpenCode Go retained history does not match the current request."
)


@dataclass(frozen=True)
class OpenCodeGoRetainedHistoryDecision:
    previous_response_id: Optional[str]
    prepend: bool


def extract_opencode_go_previous_response_id(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("previous_response_id")
    if value is None:
        return None
    if not isinstance(value, str):
        _raise_retained_history_rejection(
            message=_UNKNOWN_MESSAGE,
            status_code=400,
            reason="previous_response_id_type",
        )
    cleaned = value.strip()
    return cleaned or None


def _raise_retained_history_rejection(
    *,
    message: str,
    status_code: int,
    reason: str,
) -> Never:
    proxy_exc = ProxyException(
        message=message,
        type="invalid_request_error",
        param="previous_response_id",
        code=status_code,
    )
    setattr(proxy_exc, "status_code", status_code)
    setattr(proxy_exc, "attempted_provider_call", False)
    setattr(proxy_exc, "failure_phase", _FAILURE_PHASE)
    setattr(proxy_exc, "ineligibility_reason", reason)
    setattr(
        proxy_exc,
        "detail",
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": "previous_response_id",
                "code": reason,
            },
            "attempted_provider_call": False,
            "failure_phase": _FAILURE_PHASE,
        },
    )
    raise proxy_exc


def _as_message_mapping(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    for attr in ("model_dump", "dict"):
        dumper = getattr(message, attr, None)
        if callable(dumper):
            try:
                dumped = dumper()
            except Exception:
                dumped = None
            if isinstance(dumped, dict):
                return dumped
    mapping: dict[str, Any] = {}
    for key in ("role", "content", "tool_calls", "tool_call_id", "name"):
        if hasattr(message, key):
            mapping[key] = getattr(message, key)
    return mapping


def _message_role(message: Any) -> str:
    mapping = _as_message_mapping(message)
    role = mapping.get("role")
    if isinstance(role, str) and role.strip():
        return role.strip()
    return ""


def _tool_call_ids(message: Any) -> list[str]:
    mapping = _as_message_mapping(message)
    raw_calls = mapping.get("tool_calls")
    if raw_calls is None:
        raw_calls = getattr(message, "tool_calls", None)
    if not isinstance(raw_calls, list):
        return []
    ids: list[str] = []
    for tool_call in raw_calls:
        call_id: Any = None
        if isinstance(tool_call, Mapping):
            call_id = tool_call.get("id")
        else:
            call_id = getattr(tool_call, "id", None)
        if isinstance(call_id, str) and call_id.strip():
            ids.append(call_id.strip())
        else:
            ids.append("")
    return ids


def _tool_result_id(message: Any) -> str:
    mapping = _as_message_mapping(message)
    raw = mapping.get("tool_call_id")
    if raw is None:
        raw = getattr(message, "tool_call_id", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return ""


def _content_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except (TypeError, ValueError):
        return ""


def _message_alignment_key(message: Any) -> tuple[Any, ...]:
    mapping = _as_message_mapping(message)
    role = _message_role(mapping)
    if role == "tool":
        return ("tool", _tool_result_id(mapping))
    if role == "assistant":
        return (
            "assistant",
            tuple(_tool_call_ids(mapping)),
            _content_key(mapping.get("content")),
        )
    return (role, _content_key(mapping.get("content")))


def _classify_history_alignment(
    session_messages: Sequence[Any],
    current_messages: Sequence[Any],
) -> str:
    if not session_messages:
        return "missing"
    session_keys = [_message_alignment_key(item) for item in session_messages]
    current_keys = [_message_alignment_key(item) for item in current_messages]
    prefix_len = len(session_keys)
    if current_keys[:prefix_len] == session_keys:
        return "prefix"
    if set(session_keys).intersection(current_keys):
        return "inconsistent"
    return "prepend"


def _validate_ordered_tool_history(messages: Sequence[Any]) -> Optional[str]:
    seen_call_ids: set[str] = set()
    for message in messages:
        role = _message_role(message)
        if role == "assistant":
            call_ids = _tool_call_ids(message)
            if any(not call_id for call_id in call_ids):
                return "empty_tool_call_id"
            if len(call_ids) != len(set(call_ids)):
                return "duplicate_tool_call_id"
            seen_call_ids.update(call_ids)
            continue
        if role != "tool":
            continue
        result_id = _tool_result_id(message)
        if not result_id:
            return "empty_tool_result_id"
        if result_id not in seen_call_ids:
            return "tool_result_without_matching_call"
    return None


def _metadata_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _spend_log_is_opencode_go(spend_log: Mapping[str, Any]) -> bool:
    provider = str(spend_log.get("custom_llm_provider") or "").strip().lower()
    if provider == OPENCODE_GO_PROVIDER:
        return True
    api_base = str(spend_log.get("api_base") or "")
    if any(marker in api_base for marker in _GO_API_BASE_MARKERS):
        return True
    metadata = _metadata_mapping(spend_log.get("metadata"))
    route_family = str(
        metadata.get("route_family") or metadata.get("aawm_route_family") or ""
    ).strip().lower()
    if route_family == _GO_ROUTE_FAMILY:
        return True
    tags = metadata.get("tags") or spend_log.get("request_tags")
    tag_blob = tags if isinstance(tags, str) else _content_key(tags)
    if "codex_opencode_go_adapter" in tag_blob or "opencode_go" in tag_blob:
        return True
    return False


def _account_token(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _reject_cross_owner_history(
    *,
    request: Any,
    request_body: Optional[Mapping[str, Any]],
    spend_logs: Sequence[Mapping[str, Any]],
) -> None:
    if not spend_logs or not all(
        _spend_log_is_opencode_go(spend_log) for spend_log in spend_logs
    ):
        _raise_retained_history_rejection(
            message=_CROSS_OWNER_MESSAGE,
            status_code=409,
            reason="cross_provider_history",
        )

    lease = get_request_session_owner_lease(request)
    lease_attrs = (
        lease.attributes
        if lease is not None and isinstance(lease.attributes, Mapping)
        else {}
    )
    lease_provider = str(lease_attrs.get("provider") or "").strip().lower()
    lease_route = str(lease_attrs.get("route_family") or "").strip().lower()
    if lease_provider and lease_provider != OPENCODE_GO_PROVIDER:
        _raise_retained_history_rejection(
            message=_CROSS_OWNER_MESSAGE,
            status_code=409,
            reason="cross_provider_history",
        )
    if lease_route and lease_route != _GO_ROUTE_FAMILY:
        _raise_retained_history_rejection(
            message=_CROSS_OWNER_MESSAGE,
            status_code=409,
            reason="cross_provider_history",
        )

    current_account = extract_account_identity_from_context(
        request=request,
        request_body=request_body,
    )
    current_hash = _account_token(current_account.get("account_hash"))
    current_lane = _account_token(current_account.get("account_lane"))
    owner_hash = _account_token(lease_attrs.get("account_hash"))
    owner_lane = _account_token(lease_attrs.get("account_lane"))
    if current_hash and owner_hash and current_hash != owner_hash:
        _raise_retained_history_rejection(
            message=_CROSS_OWNER_MESSAGE,
            status_code=409,
            reason="cross_account_history",
        )
    if current_lane and owner_lane and current_lane != owner_lane:
        _raise_retained_history_rejection(
            message=_CROSS_OWNER_MESSAGE,
            status_code=409,
            reason="cross_account_history",
        )

    spend_keys = {
        _account_token(spend_log.get("api_key"))
        for spend_log in spend_logs
        if _account_token(spend_log.get("api_key")) is not None
    }
    if len(spend_keys) > 1:
        _raise_retained_history_rejection(
            message=_CROSS_OWNER_MESSAGE,
            status_code=409,
            reason="cross_account_history",
        )


def _combined_messages(
    *,
    session_messages: Sequence[Any],
    current_messages: Sequence[Any],
    alignment: str,
) -> list[Any]:
    if alignment == "prefix":
        return list(current_messages)
    return list(session_messages) + list(current_messages)


def drop_opencode_go_previous_response_id(payload: dict[str, Any]) -> dict[str, Any]:
    if "previous_response_id" in payload:
        payload.pop("previous_response_id", None)
    return payload


async def resolve_opencode_go_retained_history(
    *,
    previous_response_id: Any,
    current_messages: Sequence[Any],
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
) -> OpenCodeGoRetainedHistoryDecision:
    if previous_response_id is None:
        return OpenCodeGoRetainedHistoryDecision(
            previous_response_id=None,
            prepend=False,
        )
    if not isinstance(previous_response_id, str):
        _raise_retained_history_rejection(
            message=_UNKNOWN_MESSAGE,
            status_code=400,
            reason="previous_response_id_type",
        )
    cleaned = previous_response_id.strip()
    if not cleaned:
        return OpenCodeGoRetainedHistoryDecision(
            previous_response_id=None,
            prepend=False,
        )

    try:
        spend_logs = (
            await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
                cleaned
            )
        )
    except ProxyException:
        raise
    except Exception:
        _raise_retained_history_rejection(
            message=_UNAVAILABLE_MESSAGE,
            status_code=400,
            reason="previous_response_id_unavailable",
        )

    if not spend_logs:
        _raise_retained_history_rejection(
            message=_UNKNOWN_MESSAGE,
            status_code=400,
            reason="previous_response_id_unknown",
        )

    _reject_cross_owner_history(
        request=request,
        request_body=request_body,
        spend_logs=spend_logs,
    )

    try:
        session = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            cleaned
        )
    except ProxyException:
        raise
    except Exception:
        _raise_retained_history_rejection(
            message=_UNAVAILABLE_MESSAGE,
            status_code=400,
            reason="previous_response_id_unavailable",
        )

    session_messages = list(session.get("messages") or []) if session else []
    if not session_messages:
        _raise_retained_history_rejection(
            message=_UNAVAILABLE_MESSAGE,
            status_code=400,
            reason="previous_response_id_unavailable",
        )

    alignment = _classify_history_alignment(session_messages, current_messages)
    if alignment == "missing":
        _raise_retained_history_rejection(
            message=_UNAVAILABLE_MESSAGE,
            status_code=400,
            reason="previous_response_id_unavailable",
        )
    if alignment == "inconsistent":
        _raise_retained_history_rejection(
            message=_INCONSISTENT_MESSAGE,
            status_code=400,
            reason="inconsistent_retained_history",
        )

    combined = _combined_messages(
        session_messages=session_messages,
        current_messages=current_messages,
        alignment=alignment,
    )
    tool_failure = _validate_ordered_tool_history(combined)
    if tool_failure is not None:
        _raise_retained_history_rejection(
            message=_INVALID_TOOL_MESSAGE,
            status_code=400,
            reason="invalid_tool_history",
        )

    return OpenCodeGoRetainedHistoryDecision(
        previous_response_id=cleaned,
        prepend=alignment == "prepend",
    )


async def apply_opencode_go_retained_chat_history(
    *,
    completion_kwargs: dict[str, Any],
    decision: OpenCodeGoRetainedHistoryDecision,
) -> dict[str, Any]:
    drop_opencode_go_previous_response_id(completion_kwargs)
    if decision.previous_response_id is None or not decision.prepend:
        return completion_kwargs
    # Shallow-copy so access logs that captured the current-turn body cannot
    # observe the prepended retained history.
    completion_kwargs = dict(completion_kwargs)
    completion_kwargs = (
        await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id=decision.previous_response_id,
            litellm_completion_request=completion_kwargs,
        )
    )
    drop_opencode_go_previous_response_id(completion_kwargs)
    return completion_kwargs


async def apply_opencode_go_retained_responses_history(
    *,
    adapted_request_body: dict[str, Any],
    decision: OpenCodeGoRetainedHistoryDecision,
    adapter_model: str,
    litellm_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    drop_opencode_go_previous_response_id(adapted_request_body)
    if decision.previous_response_id is None or not decision.prepend:
        return adapted_request_body

    adapted_request_body = dict(adapted_request_body)
    request_input = adapted_request_body.get("input", "")
    responses_api_request = {
        key: value
        for key, value in adapted_request_body.items()
        if key not in {"input", "model", "litellm_metadata"}
    }
    completion_kwargs = (
        LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=adapter_model,
            input=request_input,
            responses_api_request=responses_api_request,
            custom_llm_provider=OPENCODE_GO_PROVIDER,
            stream=False,
            metadata=dict(litellm_metadata or {}),
        )
    )
    completion_kwargs = await apply_opencode_go_retained_chat_history(
        completion_kwargs=completion_kwargs,
        decision=decision,
    )
    raw_messages = completion_kwargs.get("messages") or []
    message_dicts = [_as_message_mapping(message) for message in raw_messages]
    input_items, _instructions = OpenAICountTokensConfig.messages_to_responses_input(
        message_dicts
    )
    adapted_request_body["input"] = input_items
    drop_opencode_go_previous_response_id(adapted_request_body)
    return adapted_request_body
