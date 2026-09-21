"""OpenCode Go ``previous_response_id`` retained-history contract.

Chat Completions egress cannot honor a Responses continuation id, and native
Go Responses cannot look up LiteLLM-encoded ids. Materialize retained history
before credential load, or reject before any provider call.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Never, Optional, Sequence

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

_GO_ROUTE_FAMILY = "codex_opencode_go_adapter"
_GO_API_BASE_MARKERS = ("/zen/go/",)
_OPENAI_API_BASE_MARKERS = ("api.openai.com",)
_ANTHROPIC_API_BASE_MARKERS = ("api.anthropic.com",)
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
_UNSUPPORTED_MESSAGE = (
    "OpenCode Go retained history contains an unsupported Responses form."
)

_SUPPORTED_CONTENT_TYPES = frozenset(
    {
        "text",
        "input_text",
        "output_text",
        "image_url",
        "input_image",
        "file",
        "input_file",
    }
)
_INSTRUCTION_CONTENT_TYPES = frozenset(
    {
        "text",
        "input_text",
        "output_text",
    }
)
_UNSUPPORTED_NATIVE_TYPES = frozenset(
    {
        "computer_call",
        "computer_call_output",
        "web_search_call",
        "file_search_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
        "item_reference",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_search",
        "reasoning",
    }
)
_ACCOUNT_HASH_KEYS = (
    "account_hash",
    "codex_oauth_account_hash",
    "xai_oauth_account_hash",
    "codex_auto_agent_selected_account_hash",
    "anthropic_auto_agent_selected_account_hash",
)
_ACCOUNT_LANE_KEYS = (
    "account_lane",
    "codex_oauth_lane_key",
    "xai_oauth_lane_key",
    "lane_key",
    "codex_auto_agent_selected_account_lane",
    "anthropic_auto_agent_selected_account_lane",
)
_USER_KEY_KEYS = ("user_api_key", "user_api_key_hash")


@dataclass(frozen=True)
class OpenCodeGoRetainedHistoryDecision:
    previous_response_id: Optional[str]
    prepend: bool
    chat_messages: tuple[Any, ...] = field(default_factory=tuple)
    native_input: tuple[Any, ...] = field(default_factory=tuple)
    instructions: Optional[str] = None


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


def _raise_unsupported_retained_history() -> Never:
    _raise_retained_history_rejection(
        message=_UNSUPPORTED_MESSAGE,
        status_code=400,
        reason="unsupported_retained_history",
    )


def _raise_previous_response_unavailable() -> Never:
    _raise_retained_history_rejection(
        message=_UNAVAILABLE_MESSAGE,
        status_code=400,
        reason="previous_response_id_unavailable",
    )


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
    for key in (
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "call_id",
        "name",
        "type",
        "arguments",
        "output",
        "instructions",
    ):
        if hasattr(message, key):
            mapping[key] = getattr(message, key)
    return mapping


def _optional_string_item_type(item_type: Any) -> Optional[str]:
    if item_type is None:
        return None
    if not isinstance(item_type, str):
        _raise_previous_response_unavailable()
    return item_type


def _function_name(mapping: Mapping[str, Any]) -> Optional[str]:
    function = mapping.get("function")
    function_map = function if isinstance(function, Mapping) else {}
    name = function_map.get("name")
    if isinstance(name, str):
        return name
    top = mapping.get("name")
    return top if isinstance(top, str) else None


def _function_arguments(mapping: Mapping[str, Any]) -> tuple[bool, Any]:
    function = mapping.get("function")
    function_map = function if isinstance(function, Mapping) else {}
    arguments = function_map.get("arguments")
    if arguments is None:
        arguments = mapping.get("arguments")
    return arguments is not None, arguments


def _has_function_call_payload(mapping: Mapping[str, Any]) -> bool:
    name = _function_name(mapping)
    has_arguments, _ = _function_arguments(mapping)
    return name is not None and has_arguments


def _has_tool_result_payload(mapping: Mapping[str, Any]) -> bool:
    return mapping.get("output") is not None or mapping.get("content") is not None


def _assistant_tool_calls_have_required_payload(raw_calls: Any) -> bool:
    if not isinstance(raw_calls, list) or not raw_calls:
        return False
    for tool_call in raw_calls:
        if not _has_function_call_payload(_as_message_mapping(tool_call)):
            return False
    return True


def _text_block_has_representable_text(block: Mapping[str, Any]) -> bool:
    return isinstance(block.get("text"), str)


def _item_has_required_fields(mapping: Mapping[str, Any]) -> bool:
    item_type = mapping.get("type")
    if item_type is not None and not isinstance(item_type, str):
        return False
    type_name = item_type if isinstance(item_type, str) and item_type else ""
    role = mapping.get("role")
    role_name = role.strip() if isinstance(role, str) else ""
    has_content = "content" in mapping and mapping.get("content") is not None
    raw_calls = mapping.get("tool_calls")
    has_tool_calls = isinstance(raw_calls, list) and bool(raw_calls)
    if type_name in _UNSUPPORTED_NATIVE_TYPES:
        return True
    if type_name == "function_call":
        return _has_function_call_payload(mapping)
    if type_name == "function_call_output":
        return _has_tool_result_payload(mapping)
    if has_tool_calls:
        return _assistant_tool_calls_have_required_payload(raw_calls)
    if role_name == "tool" or mapping.get("tool_call_id") is not None:
        return _has_tool_result_payload(mapping)
    if role_name == "assistant":
        return has_content or has_tool_calls
    if type_name == "message" or role_name:
        return has_content
    return has_content


def _item_has_structural_form(mapping: Mapping[str, Any]) -> bool:
    return _item_has_required_fields(mapping)


def _require_interpretable_item_mapping(raw_item: Any) -> dict[str, Any]:
    if raw_item is None:
        _raise_previous_response_unavailable()
    if isinstance(raw_item, Mapping) and not raw_item:
        _raise_previous_response_unavailable()
    mapping = _as_message_mapping(raw_item)
    if not mapping or not _item_has_structural_form(mapping):
        _raise_previous_response_unavailable()
    return mapping


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return value


def _metadata_mapping(value: Any) -> dict[str, Any]:
    parsed = _parse_json_value(value)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _account_token(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _exact_tool_id(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


def _tag_list(value: Any) -> list[str]:
    parsed = _parse_json_value(value)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _tokens_from_mapping(mapping: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for key in keys:
        token = _account_token(mapping.get(key))
        if token is not None:
            tokens.append(token)
    return tokens


def _tokens_from_mappings(
    *mappings: Mapping[str, Any],
    keys: Sequence[str],
) -> list[str]:
    tokens: list[str] = []
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        tokens.extend(_tokens_from_mapping(mapping, keys))
    return tokens


def _collapse_identity_kind(tokens: Sequence[str]) -> Optional[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    if len(unique) > 1:
        _raise_retained_history_rejection(
            message=_CROSS_OWNER_MESSAGE,
            status_code=409,
            reason="cross_account_history",
        )
    if not unique:
        return None
    return unique[0]


def _content_key(value: Any) -> tuple[str, str]:
    if value is None:
        return ("none", "")
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return ("bytes", value.decode("utf-8"))
        except Exception:
            return ("bytes", "")
    try:
        dumped = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except (TypeError, ValueError):
        return ("unserializable", type(value).__name__)
    if isinstance(value, Mapping):
        return ("map", dumped)
    if isinstance(value, list):
        return ("list", dumped)
    if isinstance(value, bool):
        return ("bool", dumped)
    if isinstance(value, (int, float)):
        return ("number", dumped)
    return ("other", dumped)


def _clone_item(value: Any) -> Any:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if isinstance(value, list):
        return deepcopy(value)
    return deepcopy(value) if value is not None else value


def drop_opencode_go_previous_response_id(payload: dict[str, Any]) -> dict[str, Any]:
    if "previous_response_id" in payload:
        payload.pop("previous_response_id", None)
    return payload


def _request_dict_from_proxy_server_request(value: Any) -> dict[str, Any]:
    parsed = _parse_json_value(value)
    if not isinstance(parsed, Mapping):
        return {}
    if "input" in parsed or "messages" in parsed or "instructions" in parsed:
        return dict(parsed)
    body = parsed.get("body")
    body_parsed = _parse_json_value(body)
    if isinstance(body_parsed, Mapping):
        return dict(body_parsed)
    return dict(parsed)


def _spend_log_is_opencode_go(spend_log: Mapping[str, Any]) -> bool:
    api_base = str(spend_log.get("api_base") or "")
    has_go_destination = any(marker in api_base for marker in _GO_API_BASE_MARKERS)
    has_openai_origin = any(
        marker in api_base for marker in _OPENAI_API_BASE_MARKERS
    )
    has_anthropic_origin = any(
        marker in api_base for marker in _ANTHROPIC_API_BASE_MARKERS
    )
    if has_go_destination:
        return True
    if has_openai_origin or has_anthropic_origin:
        return False
    provider = str(spend_log.get("custom_llm_provider") or "").strip().lower()
    if provider == OPENCODE_GO_PROVIDER:
        return True
    if provider:
        return False
    metadata = _metadata_mapping(spend_log.get("metadata"))
    route_family = str(
        metadata.get("route_family") or metadata.get("aawm_route_family") or ""
    ).strip().lower()
    if route_family == _GO_ROUTE_FAMILY:
        return True
    nested = _metadata_mapping(metadata.get("spend_logs_metadata"))
    nested_route = str(
        nested.get("route_family") or nested.get("aawm_route_family") or ""
    ).strip().lower()
    if nested_route == _GO_ROUTE_FAMILY:
        return True
    tags = _tag_list(metadata.get("tags")) + _tag_list(spend_log.get("request_tags"))
    exact_go_tags = {_GO_ROUTE_FAMILY, OPENCODE_GO_PROVIDER}
    return any(tag in exact_go_tags for tag in tags)


def _identity_from_mappings(
    *mappings: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    account_hash = _collapse_identity_kind(
        _tokens_from_mappings(*mappings, keys=_ACCOUNT_HASH_KEYS)
    )
    account_lane = _collapse_identity_kind(
        _tokens_from_mappings(*mappings, keys=_ACCOUNT_LANE_KEYS)
    )
    user_key = _collapse_identity_kind(
        _tokens_from_mappings(*mappings, keys=_USER_KEY_KEYS)
    )
    return account_hash, account_lane, user_key


def _spend_log_owner_identity(
    spend_log: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    metadata = _metadata_mapping(spend_log.get("metadata"))
    nested = _metadata_mapping(metadata.get("spend_logs_metadata"))
    proxy = _request_dict_from_proxy_server_request(
        spend_log.get("proxy_server_request")
    )
    proxy_meta = _metadata_mapping(proxy.get("litellm_metadata"))
    persisted_key: dict[str, Any] = {}
    api_key = _account_token(spend_log.get("api_key"))
    if api_key is not None:
        persisted_key["user_api_key"] = api_key
    return _identity_from_mappings(
        proxy_meta,
        nested,
        metadata,
        spend_log,
        persisted_key,
    )


def _current_owner_identity(
    *,
    request: Any,
    request_body: Optional[Mapping[str, Any]],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    current_account = extract_account_identity_from_context(
        request=request,
        request_body=request_body,
    )
    lease = get_request_session_owner_lease(request)
    lease_attrs = (
        lease.attributes
        if lease is not None and isinstance(lease.attributes, Mapping)
        else {}
    )
    body = request_body if isinstance(request_body, Mapping) else {}
    body_meta = _metadata_mapping(body.get("litellm_metadata"))
    state = getattr(request, "state", None)
    user_api_key_dict = getattr(state, "user_api_key_dict", None)
    dict_identity: dict[str, Any] = {}
    if user_api_key_dict is not None:
        for attr, dest in (
            ("api_key", "user_api_key"),
            ("token", "user_api_key"),
            ("hashed_token", "user_api_key_hash"),
        ):
            token = _account_token(getattr(user_api_key_dict, attr, None))
            if token is not None and dest not in dict_identity:
                dict_identity[dest] = token
    # Preserve current-account-versus-lease before request metadata can
    # collapse one side of a same-kind conflict.
    _identity_from_mappings(current_account, lease_attrs)
    account_hash, account_lane, user_key = _identity_from_mappings(
        current_account,
        lease_attrs,
        body_meta,
        body,
        dict_identity,
    )
    return account_hash, account_lane, user_key


def _identities_conflict(
    left: Optional[str],
    right: Optional[str],
) -> bool:
    return bool(left and right and left != right)


def _identities_bind(
    spend_identity: tuple[Optional[str], Optional[str], Optional[str]],
    current_identity: tuple[Optional[str], Optional[str], Optional[str]],
) -> bool:
    spend_hash, spend_lane, spend_key = spend_identity
    current_hash, current_lane, current_key = current_identity
    if _identities_conflict(spend_hash, current_hash):
        return False
    if _identities_conflict(spend_lane, current_lane):
        return False
    if _identities_conflict(spend_key, current_key):
        return False
    return any(
        (
            spend_hash and current_hash and spend_hash == current_hash,
            spend_lane and current_lane and spend_lane == current_lane,
            spend_key and current_key and spend_key == current_key,
        )
    )


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

    current_identity = _current_owner_identity(
        request=request,
        request_body=request_body,
    )
    spend_hashes: list[str] = []
    spend_lanes: list[str] = []
    spend_keys: list[str] = []
    for spend_log in spend_logs:
        spend_identity = _spend_log_owner_identity(spend_log)
        if not _identities_bind(spend_identity, current_identity):
            _raise_retained_history_rejection(
                message=_CROSS_OWNER_MESSAGE,
                status_code=409,
                reason="cross_account_history",
            )
        spend_hash, spend_lane, spend_key = spend_identity
        if spend_hash is not None:
            spend_hashes.append(spend_hash)
        if spend_lane is not None:
            spend_lanes.append(spend_lane)
        if spend_key is not None:
            spend_keys.append(spend_key)
    _collapse_identity_kind(spend_hashes)
    _collapse_identity_kind(spend_lanes)
    _collapse_identity_kind(spend_keys)


def _validate_lossless_content(content: Any) -> None:
    if content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        _raise_unsupported_retained_history()
    for block in content:
        if isinstance(block, str):
            continue
        if not isinstance(block, Mapping):
            _raise_unsupported_retained_history()
        block_type = block.get("type")
        if block_type is None:
            if "text" in block or "image_url" in block or "file" in block:
                if "text" in block and not _text_block_has_representable_text(block):
                    _raise_previous_response_unavailable()
                continue
            _raise_unsupported_retained_history()
        if not isinstance(block_type, str) or block_type not in _SUPPORTED_CONTENT_TYPES:
            _raise_unsupported_retained_history()
        if block_type in {"text", "input_text", "output_text"}:
            if not _text_block_has_representable_text(block):
                _raise_previous_response_unavailable()


def _validate_instruction_content(content: Any) -> None:
    if content is None or isinstance(content, str):
        return
    if not isinstance(content, list):
        _raise_unsupported_retained_history()
    for block in content:
        if isinstance(block, str):
            continue
        if not isinstance(block, Mapping):
            _raise_unsupported_retained_history()
        block_type = block.get("type")
        if block_type is None:
            if "image_url" in block or "file" in block:
                _raise_unsupported_retained_history()
            if "text" in block:
                if not isinstance(block.get("text"), str):
                    _raise_unsupported_retained_history()
                continue
            _raise_unsupported_retained_history()
        if (
            not isinstance(block_type, str)
            or block_type not in _INSTRUCTION_CONTENT_TYPES
        ):
            _raise_unsupported_retained_history()
        if not isinstance(block.get("text"), str):
            _raise_unsupported_retained_history()


def _instruction_content_for_validation(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        return [value]
    return value


def _instructions_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    _validate_instruction_content(_instruction_content_for_validation(value))
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str) and block:
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        combined = "\n".join(parts)
        return combined or None
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str) and text:
            return text
        return None
    _raise_unsupported_retained_history()


def _merge_instructions(*values: Optional[str]) -> Optional[str]:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        parts.append(value)
    if not parts:
        return None
    return "\n".join(parts)


def _native_content_from_chat_content(content: Any) -> Any:
    _validate_lossless_content(content)
    if content is None or isinstance(content, str):
        return content
    native_blocks: list[Any] = []
    for block in content:
        if isinstance(block, str):
            native_blocks.append({"type": "input_text", "text": block})
            continue
        block_map = dict(block)
        block_type = block_map.get("type")
        if isinstance(block_type, str) and block_type in {"text", "output_text"}:
            if not _text_block_has_representable_text(block_map):
                _raise_previous_response_unavailable()
            native_blocks.append({"type": "input_text", "text": block_map.get("text")})
            continue
        if block_type == "input_text":
            if not _text_block_has_representable_text(block_map):
                _raise_previous_response_unavailable()
            native_blocks.append(block_map)
            continue
        if block_type == "image_url":
            image_url = block_map.get("image_url")
            url = image_url
            detail = None
            if isinstance(image_url, Mapping):
                url = image_url.get("url")
                detail = image_url.get("detail")
            native_item: dict[str, Any] = {"type": "input_image", "image_url": url}
            if detail is not None:
                native_item["detail"] = detail
            native_blocks.append(native_item)
            continue
        if block_type == "input_image":
            native_blocks.append(block_map)
            continue
        if block_type == "file":
            file_obj = block_map.get("file")
            native_file = {"type": "input_file"}
            if isinstance(file_obj, Mapping):
                native_file.update(file_obj)
            else:
                native_file.update(
                    {
                        key: block_map[key]
                        for key in ("file_id", "file_data", "filename")
                        if key in block_map
                    }
                )
            native_blocks.append(native_file)
            continue
        if block_type == "input_file":
            native_blocks.append(block_map)
            continue
        _raise_unsupported_retained_history()
    return native_blocks


def _chat_content_from_native_content(content: Any) -> Any:
    _validate_lossless_content(content)
    if content is None or isinstance(content, str):
        return content
    chat_blocks: list[Any] = []
    for block in content:
        if isinstance(block, str):
            chat_blocks.append({"type": "text", "text": block})
            continue
        block_map = dict(block)
        block_type = block_map.get("type")
        if isinstance(block_type, str) and block_type in {
            "input_text",
            "output_text",
            "text",
        }:
            if not _text_block_has_representable_text(block_map):
                _raise_previous_response_unavailable()
            chat_blocks.append({"type": "text", "text": block_map.get("text")})
            continue
        if block_type == "input_image":
            image_url = block_map.get("image_url") or block_map.get("url")
            detail = block_map.get("detail")
            image_obj: dict[str, Any] = {"url": image_url}
            if detail is not None:
                image_obj["detail"] = detail
            chat_blocks.append({"type": "image_url", "image_url": image_obj})
            continue
        if block_type == "image_url":
            chat_blocks.append(block_map)
            continue
        if block_type == "input_file":
            file_obj = {
                key: block_map[key]
                for key in ("file_id", "file_data", "filename")
                if key in block_map
            }
            chat_blocks.append({"type": "file", "file": file_obj})
            continue
        if block_type == "file":
            chat_blocks.append(block_map)
            continue
        _raise_unsupported_retained_history()
    return chat_blocks


def _function_call_canonical_id(mapping: Mapping[str, Any]) -> Optional[str]:
    call_id = _exact_tool_id(mapping.get("call_id"))
    if call_id is not None:
        return call_id
    return _exact_tool_id(mapping.get("id"))


def _function_call_output_canonical_id(mapping: Mapping[str, Any]) -> Optional[str]:
    call_id = _exact_tool_id(mapping.get("call_id"))
    if call_id is not None:
        return call_id
    return _exact_tool_id(mapping.get("tool_call_id"))


def _canonicalize_function_call_item(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not _has_function_call_payload(mapping):
        _raise_previous_response_unavailable()
    item = _clone_item(mapping)
    if not isinstance(item, dict):
        item = dict(mapping)
    item["type"] = "function_call"
    call_id = _function_call_canonical_id(item)
    item["call_id"] = call_id if call_id is not None else ""
    return item


def _canonicalize_function_call_output_item(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if not _has_tool_result_payload(mapping):
        _raise_previous_response_unavailable()
    item = _clone_item(mapping)
    if not isinstance(item, dict):
        item = dict(mapping)
    item["type"] = "function_call_output"
    call_id = _function_call_output_canonical_id(item)
    item["call_id"] = call_id if call_id is not None else ""
    if item.get("output") is None:
        item["output"] = item.get("content")
    return item


def _tool_call_payload(tool_call: Any) -> tuple[str, str, tuple[str, str]]:
    mapping = _as_message_mapping(tool_call)
    call_id = _function_call_canonical_id(mapping) or ""
    function = mapping.get("function")
    function_map = function if isinstance(function, Mapping) else {}
    name = function_map.get("name")
    if not isinstance(name, str):
        name = mapping.get("name") if isinstance(mapping.get("name"), str) else ""
    arguments = function_map.get("arguments")
    if arguments is None:
        arguments = mapping.get("arguments")
    return call_id, name, _content_key(arguments)


def _native_items_from_assistant_chat(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    content = message.get("content")
    if content:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": _native_content_from_chat_content(content),
            }
        )
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        if not items:
            if content is None:
                _raise_previous_response_unavailable()
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": _native_content_from_chat_content(content),
                }
            )
        return items
    for tool_call in raw_calls:
        mapping = _as_message_mapping(tool_call)
        name = _function_name(mapping)
        has_arguments, arguments = _function_arguments(mapping)
        if name is None or not has_arguments:
            _raise_previous_response_unavailable()
        items.append(
            _canonicalize_function_call_item(
                {
                    "type": "function_call",
                    "call_id": _function_call_canonical_id(mapping),
                    "id": mapping.get("id"),
                    "name": name,
                    "arguments": arguments,
                }
            )
        )
    return items


def _native_items_from_chat_messages(messages: Sequence[Any]) -> tuple[list[dict[str, Any]], Optional[str]]:
    items: list[dict[str, Any]] = []
    instructions: Optional[str] = None
    for message in messages:
        mapping = _require_interpretable_item_mapping(message)
        role = mapping.get("role")
        role_name = role.strip() if isinstance(role, str) else ""
        item_type = _optional_string_item_type(mapping.get("type"))
        if item_type in _UNSUPPORTED_NATIVE_TYPES:
            _raise_unsupported_retained_history()
        if role_name in {"system", "developer"}:
            instructions = _merge_instructions(
                instructions, _instructions_text(mapping.get("content"))
            )
            continue
        if role_name == "tool" or item_type == "function_call_output":
            output = mapping.get("content")
            if output is None:
                output = mapping.get("output")
            if output is None:
                _raise_previous_response_unavailable()
            items.append(
                _canonicalize_function_call_output_item(
                    {
                        "type": "function_call_output",
                        "call_id": _function_call_output_canonical_id(mapping),
                        "tool_call_id": mapping.get("tool_call_id"),
                        "output": output,
                    }
                )
            )
            continue
        if item_type == "function_call":
            items.append(_canonicalize_function_call_item(mapping))
            continue
        if role_name == "assistant":
            items.extend(_native_items_from_assistant_chat(mapping))
            continue
        content = mapping.get("content")
        if content is None:
            _raise_previous_response_unavailable()
        items.append(
            {
                "type": "message",
                "role": role_name or "user",
                "content": _native_content_from_chat_content(content),
            }
        )
    return items, instructions


def _native_items_from_input(
    value: Any,
    *,
    instructions: Any = None,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    collected_instructions = _instructions_text(instructions)
    if value is None or value == "":
        return [], collected_instructions
    if isinstance(value, str):
        return (
            [{"type": "message", "role": "user", "content": value}],
            collected_instructions,
        )
    parsed = _parse_json_value(value)
    if isinstance(parsed, str):
        return (
            [{"type": "message", "role": "user", "content": parsed}],
            collected_instructions,
        )
    if isinstance(parsed, Mapping):
        parsed_instructions = _merge_instructions(
            collected_instructions, _instructions_text(parsed.get("instructions"))
        )
        if "input" in parsed:
            items, nested_instructions = _native_items_from_input(
                parsed.get("input"),
                instructions=None,
            )
            return items, _merge_instructions(parsed_instructions, nested_instructions)
        if "messages" in parsed and isinstance(parsed.get("messages"), list):
            items, nested_instructions = _native_items_from_chat_messages(
                parsed.get("messages") or []
            )
            return items, _merge_instructions(parsed_instructions, nested_instructions)
        parsed = [parsed]
    if not isinstance(parsed, list):
        _raise_unsupported_retained_history()
    items: list[dict[str, Any]] = []
    for raw_item in parsed:
        mapping = _require_interpretable_item_mapping(raw_item)
        item_type = _optional_string_item_type(mapping.get("type"))
        if item_type in _UNSUPPORTED_NATIVE_TYPES:
            _raise_unsupported_retained_history()
        role = mapping.get("role")
        role_name = role.strip() if isinstance(role, str) else ""
        if item_type == "function_call":
            items.append(_canonicalize_function_call_item(mapping))
            continue
        if item_type == "function_call_output":
            items.append(_canonicalize_function_call_output_item(mapping))
            continue
        if role_name in {"system", "developer"}:
            collected_instructions = _merge_instructions(
                collected_instructions, _instructions_text(mapping.get("content"))
            )
            continue
        if role_name == "tool" or mapping.get("tool_call_id") is not None:
            chat_items, chat_instructions = _native_items_from_chat_messages([mapping])
            items.extend(chat_items)
            collected_instructions = _merge_instructions(
                collected_instructions, chat_instructions
            )
            continue
        if role_name == "assistant" and mapping.get("tool_calls"):
            items.extend(_native_items_from_assistant_chat(mapping))
            continue
        if item_type in {None, "message"}:
            content = mapping.get("content")
            if content is None:
                _raise_previous_response_unavailable()
            _validate_lossless_content(content)
            items.append(
                {
                    "type": "message",
                    "role": role_name or "user",
                    "content": _clone_item(content),
                }
            )
            continue
        _raise_unsupported_retained_history()
    return items, collected_instructions


def _parsed_response_mapping(response: Any) -> Mapping[str, Any]:
    if response is None:
        _raise_previous_response_unavailable()
    parsed = _parse_json_value(response)
    if isinstance(response, (bytes, bytearray, str)) and not isinstance(
        parsed, Mapping
    ):
        _raise_previous_response_unavailable()
    if not isinstance(parsed, Mapping) or not parsed:
        _raise_previous_response_unavailable()
    return parsed


def _request_native_items_from_spend_log(
    spend_log: Mapping[str, Any],
    proxy: Any,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    request_dict = _request_dict_from_proxy_server_request(proxy)
    if not request_dict:
        request_dict = _request_dict_from_proxy_server_request(
            spend_log.get("proxy_server_request")
        )
    if request_dict:
        request_items, request_instructions = _native_items_from_input(
            request_dict.get("input", request_dict.get("messages")),
            instructions=request_dict.get("instructions"),
        )
        if request_items:
            return request_items, request_instructions
    messages = spend_log.get("messages")
    if isinstance(messages, list) and messages:
        request_items, request_instructions = _native_items_from_chat_messages(
            messages
        )
        if request_items:
            return request_items, request_instructions
    _raise_previous_response_unavailable()


def _output_native_items_from_spend_log(
    spend_log: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    _parsed_response_mapping(spend_log.get("response"))
    output_items, output_instructions = _native_output_items_from_response(
        spend_log.get("response")
    )
    if not output_items:
        _raise_previous_response_unavailable()
    return output_items, output_instructions


def _native_output_items_from_response(response: Any) -> tuple[list[dict[str, Any]], Optional[str]]:
    parsed = _parse_json_value(response)
    if not isinstance(parsed, Mapping) or not parsed:
        return [], None
    output = parsed.get("output")
    if isinstance(output, list):
        return _native_items_from_input(output)
    choices = parsed.get("choices")
    if isinstance(choices, list):
        messages: list[Any] = []
        for choice in choices:
            choice_map = _as_message_mapping(choice)
            message = choice_map.get("message")
            if message is None:
                continue
            messages.append(message)
        if messages:
            return _native_items_from_chat_messages(messages)
    return [], None


async def _native_history_from_spend_logs(
    spend_logs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    items: list[dict[str, Any]] = []
    instructions: Optional[str] = None
    for spend_log in spend_logs:
        try:
            proxy = await ResponsesSessionHandler.get_proxy_server_request_from_spend_log(
                spend_log  # type: ignore[arg-type]
            )
        except ProxyException:
            raise
        except Exception:
            _raise_previous_response_unavailable()
        request_items, request_instructions = _request_native_items_from_spend_log(
            spend_log,
            proxy,
        )
        output_items, output_instructions = _output_native_items_from_spend_log(
            spend_log
        )
        items = _compose_retained_items(items, request_items, output_items)
        instructions = _merge_instructions(
            instructions, request_instructions, output_instructions
        )
    return items, instructions


def _function_name_and_arguments(item: Mapping[str, Any]) -> tuple[str, tuple[str, str]]:
    function = item.get("function")
    function_map = function if isinstance(function, Mapping) else {}
    name = function_map.get("name")
    if not isinstance(name, str):
        name = item.get("name") if isinstance(item.get("name"), str) else ""
    arguments = function_map.get("arguments")
    if arguments is None:
        arguments = item.get("arguments")
    return name or "", _content_key(arguments)


def _native_id_key(item: Any) -> tuple[Any, ...]:
    mapping = _as_message_mapping(item)
    item_type = mapping.get("type")
    if item_type == "function_call":
        call_id = _function_call_canonical_id(mapping) or ""
        return ("function_call", call_id)
    if item_type == "function_call_output":
        call_id = _function_call_output_canonical_id(mapping) or ""
        return ("function_call_output", call_id)
    if mapping.get("tool_calls"):
        call_ids = tuple(
            _tool_call_payload(tool_call)[0] for tool_call in mapping.get("tool_calls")
        )
        return ("assistant_tool_calls", call_ids)
    role = mapping.get("role")
    role_name = role.strip() if isinstance(role, str) else ""
    if role_name == "tool":
        call_id = _function_call_output_canonical_id(mapping) or ""
        return ("function_call_output", call_id)
    return (
        item_type or "message",
        role_name,
        _content_key(mapping.get("content")),
    )


def _native_semantic_key(item: Any) -> tuple[Any, ...]:
    mapping = _as_message_mapping(item)
    item_type = mapping.get("type")
    if item_type == "function_call":
        call_id = _exact_tool_id(mapping.get("call_id"))
        if call_id is None:
            call_id = _exact_tool_id(mapping.get("id")) or ""
        name, arguments = _function_name_and_arguments(mapping)
        return ("function_call", call_id, name, arguments)
    if item_type == "function_call_output":
        call_id = _exact_tool_id(mapping.get("call_id"))
        if call_id is None:
            call_id = _exact_tool_id(mapping.get("tool_call_id")) or ""
        output = mapping.get("output")
        if output is None:
            output = mapping.get("content")
        return ("function_call_output", call_id, _content_key(output))
    if mapping.get("tool_calls"):
        payloads = tuple(
            _tool_call_payload(tool_call) for tool_call in mapping.get("tool_calls")
        )
        return (
            "assistant_tool_calls",
            payloads,
            _content_key(mapping.get("content")),
        )
    role = mapping.get("role")
    role_name = role.strip() if isinstance(role, str) else ""
    if role_name == "tool":
        call_id = _exact_tool_id(mapping.get("tool_call_id")) or ""
        return ("function_call_output", call_id, _content_key(mapping.get("content")))
    return (
        item_type or "message",
        role_name,
        _content_key(mapping.get("content")),
    )


def _request_restates_retained_history(
    existing_keys: Sequence[Any],
    request_keys: Sequence[Any],
) -> bool:
    if not existing_keys or not request_keys:
        return False
    existing_len = len(existing_keys)
    return (
        len(request_keys) >= existing_len
        and request_keys[:existing_len] == list(existing_keys)
    )


def _compose_retained_items(
    existing: Sequence[Any],
    request_items: Sequence[Any],
    output_items: Sequence[Any],
) -> list[Any]:
    # Decide whether the request restates history before appending output.
    # New output must not complete an apparent cumulative prefix.
    cloned_output = [_clone_item(item) for item in output_items]
    if not request_items:
        return [_clone_item(item) for item in existing] + cloned_output
    if not existing:
        return [_clone_item(item) for item in request_items] + cloned_output
    existing_keys = [_native_semantic_key(item) for item in existing]
    request_keys = [_native_semantic_key(item) for item in request_items]
    if _request_restates_retained_history(existing_keys, request_keys):
        return [_clone_item(item) for item in request_items] + cloned_output
    incoming = list(request_items) + list(output_items)
    incoming_keys = [_native_semantic_key(item) for item in incoming]
    existing_len = len(existing_keys)
    incoming_len = len(incoming_keys)
    if incoming_len < existing_len and existing_keys[:incoming_len] == incoming_keys:
        _raise_previous_response_unavailable()
    overlap_limit = min(existing_len, len(request_keys))
    for k in range(1, overlap_limit):
        if existing_keys[-k:] == request_keys[:k]:
            _raise_previous_response_unavailable()
    return [_clone_item(item) for item in existing] + [
        _clone_item(item) for item in incoming
    ]


def _classify_history_alignment(
    session_items: Sequence[Any],
    current_items: Sequence[Any],
) -> str:
    if not session_items:
        return "missing"
    session_semantic = [_native_semantic_key(item) for item in session_items]
    current_semantic = [_native_semantic_key(item) for item in current_items]
    prefix_len = len(session_semantic)
    if current_semantic[:prefix_len] == session_semantic:
        return "prefix"
    session_ids = [_native_id_key(item) for item in session_items]
    current_ids = [_native_id_key(item) for item in current_items]
    if current_ids[:prefix_len] == session_ids:
        return "replace"
    if set(session_semantic).intersection(current_semantic):
        return "inconsistent"
    return "prepend"


def _validate_native_tool_history(items: Sequence[Any]) -> Optional[str]:
    seen_call_ids: set[str] = set()
    for item in items:
        mapping = _as_message_mapping(item)
        item_type = mapping.get("type")
        if item_type == "function_call" or mapping.get("tool_calls"):
            if item_type == "function_call":
                outgoing_id = _exact_tool_id(mapping.get("call_id"))
                if not outgoing_id:
                    return "empty_tool_call_id"
                call_ids = [outgoing_id]
            else:
                raw_calls = mapping.get("tool_calls")
                if not isinstance(raw_calls, list):
                    continue
                call_ids = [_tool_call_payload(tool_call)[0] for tool_call in raw_calls]
            if any(not call_id for call_id in call_ids):
                return "empty_tool_call_id"
            if len(call_ids) != len(set(call_ids)):
                return "duplicate_tool_call_id"
            for call_id in call_ids:
                if call_id in seen_call_ids:
                    return "duplicate_tool_call_id"
                seen_call_ids.add(call_id)
            continue
        if item_type != "function_call_output" and mapping.get("role") != "tool":
            continue
        result_id = _exact_tool_id(mapping.get("call_id"))
        if result_id is None:
            result_id = _exact_tool_id(mapping.get("tool_call_id"))
        if not result_id:
            return "empty_tool_result_id"
        if result_id not in seen_call_ids:
            return "tool_result_without_matching_call"
    return None


def _combined_native_items(
    *,
    session_items: Sequence[Any],
    current_items: Sequence[Any],
    alignment: str,
) -> list[Any]:
    if alignment == "prefix":
        return [_clone_item(item) for item in current_items]
    if alignment == "replace":
        suffix = list(current_items[len(session_items) :])
        return [_clone_item(item) for item in list(session_items) + suffix]
    return [_clone_item(item) for item in list(session_items) + list(current_items)]


def _chat_messages_from_native_items(
    items: Sequence[Any],
    *,
    instructions: Optional[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    pending_assistant: Optional[dict[str, Any]] = None

    def flush_assistant() -> None:
        nonlocal pending_assistant
        if pending_assistant is not None:
            messages.append(pending_assistant)
            pending_assistant = None

    for item in items:
        mapping = _as_message_mapping(item)
        item_type = mapping.get("type")
        if item_type == "function_call":
            call_id = _exact_tool_id(mapping.get("call_id"))
            if call_id is None:
                call_id = _exact_tool_id(mapping.get("id"))
            name = mapping.get("name") if isinstance(mapping.get("name"), str) else ""
            arguments = mapping.get("arguments")
            function = mapping.get("function")
            if isinstance(function, Mapping):
                function_name = function.get("name")
                if isinstance(function_name, str):
                    name = function_name
                if arguments is None:
                    arguments = function.get("arguments")
            tool_call = {
                "id": call_id if call_id is not None else "",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments if arguments is not None else "",
                },
            }
            if pending_assistant is None:
                pending_assistant = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                }
            else:
                existing = pending_assistant.get("tool_calls")
                if isinstance(existing, list):
                    existing.append(tool_call)
                else:
                    pending_assistant["tool_calls"] = [tool_call]
            continue
        flush_assistant()
        if item_type == "function_call_output":
            call_id = _exact_tool_id(mapping.get("call_id"))
            if call_id is None:
                call_id = _exact_tool_id(mapping.get("tool_call_id"))
            output = mapping.get("output")
            if output is None:
                output = mapping.get("content")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id if call_id is not None else "",
                    "content": output if output is not None else "",
                }
            )
            continue
        role = mapping.get("role")
        role_name = role.strip() if isinstance(role, str) else "user"
        messages.append(
            {
                "role": role_name or "user",
                "content": _chat_content_from_native_content(mapping.get("content")),
            }
        )
    flush_assistant()
    return messages


def _empty_decision() -> OpenCodeGoRetainedHistoryDecision:
    return OpenCodeGoRetainedHistoryDecision(
        previous_response_id=None,
        prepend=False,
    )


async def resolve_opencode_go_retained_history(
    *,
    previous_response_id: Any,
    current_input: Any = None,
    current_instructions: Any = None,
    current_messages: Sequence[Any] | None = None,
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
) -> OpenCodeGoRetainedHistoryDecision:
    if previous_response_id is None:
        return _empty_decision()
    if not isinstance(previous_response_id, str):
        _raise_retained_history_rejection(
            message=_UNKNOWN_MESSAGE,
            status_code=400,
            reason="previous_response_id_type",
        )
    cleaned = previous_response_id.strip()
    if not cleaned:
        return _empty_decision()

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

    session_items, session_instructions = await _native_history_from_spend_logs(
        spend_logs
    )
    if not session_items:
        _raise_retained_history_rejection(
            message=_UNAVAILABLE_MESSAGE,
            status_code=400,
            reason="previous_response_id_unavailable",
        )

    if current_input is None and current_messages is not None:
        current_items, extracted_instructions = _native_items_from_chat_messages(
            current_messages
        )
        current_instruction_text = _merge_instructions(
            extracted_instructions, _instructions_text(current_instructions)
        )
    else:
        current_items, current_instruction_text = _native_items_from_input(
            current_input,
            instructions=current_instructions,
        )

    alignment = _classify_history_alignment(session_items, current_items)
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

    combined = _combined_native_items(
        session_items=session_items,
        current_items=current_items,
        alignment=alignment,
    )
    tool_failure = _validate_native_tool_history(combined)
    if tool_failure is not None:
        _raise_retained_history_rejection(
            message=_INVALID_TOOL_MESSAGE,
            status_code=400,
            reason="invalid_tool_history",
        )

    outgoing_instructions = _merge_instructions(
        session_instructions, current_instruction_text
    )
    chat_messages = _chat_messages_from_native_items(
        combined,
        instructions=outgoing_instructions,
    )
    return OpenCodeGoRetainedHistoryDecision(
        previous_response_id=cleaned,
        prepend=alignment in {"prepend", "replace"},
        chat_messages=tuple(chat_messages),
        native_input=tuple(combined),
        instructions=outgoing_instructions,
    )


def _require_materialized_result(
    decision: OpenCodeGoRetainedHistoryDecision,
    *,
    native: bool,
) -> None:
    materialized = decision.native_input if native else decision.chat_messages
    if decision.previous_response_id is not None and not materialized:
        _raise_retained_history_rejection(
            message=_UNAVAILABLE_MESSAGE,
            status_code=400,
            reason="previous_response_id_unavailable",
        )


async def apply_opencode_go_retained_chat_history(
    *,
    completion_kwargs: dict[str, Any],
    decision: OpenCodeGoRetainedHistoryDecision,
) -> dict[str, Any]:
    drop_opencode_go_previous_response_id(completion_kwargs)
    if decision.previous_response_id is None:
        return completion_kwargs
    _require_materialized_result(decision, native=False)
    # Shallow-copy so access logs that captured the current-turn body cannot
    # observe the prepended retained history.
    completion_kwargs = dict(completion_kwargs)
    completion_kwargs["messages"] = [
        _clone_item(message) for message in decision.chat_messages
    ]
    drop_opencode_go_previous_response_id(completion_kwargs)
    return completion_kwargs


async def apply_opencode_go_retained_responses_history(
    *,
    adapted_request_body: dict[str, Any],
    decision: OpenCodeGoRetainedHistoryDecision,
    adapter_model: str = "",
    litellm_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    _ = adapter_model, litellm_metadata
    drop_opencode_go_previous_response_id(adapted_request_body)
    if decision.previous_response_id is None:
        return adapted_request_body
    _require_materialized_result(decision, native=True)
    adapted_request_body = dict(adapted_request_body)
    adapted_request_body["input"] = [
        _clone_item(item) for item in decision.native_input
    ]
    if decision.instructions is not None:
        adapted_request_body["instructions"] = decision.instructions
    drop_opencode_go_previous_response_id(adapted_request_body)
    return adapted_request_body
