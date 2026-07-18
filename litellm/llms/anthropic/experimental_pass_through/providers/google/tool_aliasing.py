"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Mapping
from typing import Any, Optional


from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    task_state as _aawm_task_state,
)


@dataclass(frozen=True)
class Runtime:
    _aawm_provider_shaping: Any
    _completion_message_has_tool_result: Any
    _completion_message_has_visible_text: Any
    _estimate_completion_message_text_chars: Any
    _extract_completion_message_text: Any
    _get_google_adapter_max_completion_messages_window: Any
    _get_google_adapter_preserved_task_state_char_cap: Any
    _get_google_code_assist_native_tool_aliases: Any
    _google_code_assist_duplicate_tool_results_from_completion_messages: Any
    _google_code_assist_function_call_args_for_id: Any
    _google_code_assist_function_response_id: Any
    _google_code_assist_tool_results_from_completion_messages: Any
    _is_google_adapter_synthetic_tool_context_message: Any
    _trim_completion_message_tail_preserving_tool_pairs: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _aawm_provider_shaping=namespace["_aawm_provider_shaping"],
        _completion_message_has_tool_result=namespace["_completion_message_has_tool_result"],
        _completion_message_has_visible_text=namespace["_completion_message_has_visible_text"],
        _estimate_completion_message_text_chars=namespace["_estimate_completion_message_text_chars"],
        _extract_completion_message_text=namespace["_extract_completion_message_text"],
        _get_google_adapter_max_completion_messages_window=namespace[
            "_get_google_adapter_max_completion_messages_window"
        ],
        _get_google_adapter_preserved_task_state_char_cap=namespace[
            "_get_google_adapter_preserved_task_state_char_cap"
        ],
        _get_google_code_assist_native_tool_aliases=namespace["_get_google_code_assist_native_tool_aliases"],
        _google_code_assist_duplicate_tool_results_from_completion_messages=namespace[
            "_google_code_assist_duplicate_tool_results_from_completion_messages"
        ],
        _google_code_assist_function_call_args_for_id=namespace["_google_code_assist_function_call_args_for_id"],
        _google_code_assist_function_response_id=namespace["_google_code_assist_function_response_id"],
        _google_code_assist_tool_results_from_completion_messages=namespace[
            "_google_code_assist_tool_results_from_completion_messages"
        ],
        _is_google_adapter_synthetic_tool_context_message=namespace[
            "_is_google_adapter_synthetic_tool_context_message"
        ],
        _trim_completion_message_tail_preserving_tool_pairs=namespace[
            "_trim_completion_message_tail_preserving_tool_pairs"
        ],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _apply_google_code_assist_alias_to_function_block(
    function_block: dict[str, Any],
    *,
    aliases: dict[str, str],
    tool_name_mapping: dict[str, str],
) -> tuple[dict[str, Any], Optional[str]]:
    original_name = function_block.get("name")
    if not isinstance(original_name, str) or not original_name:
        return function_block, None

    alias_name = aliases.get(original_name)
    if not isinstance(alias_name, str) or not alias_name:
        return function_block, None

    updated_function = dict(function_block)
    updated_function["name"] = alias_name
    tool_name_mapping[alias_name] = tool_name_mapping.get(original_name, original_name)
    return updated_function, alias_name


def _apply_google_code_assist_alias_to_tool(
    tool: Any,
    *,
    aliases: dict[str, str],
    tool_name_mapping: dict[str, str],
) -> tuple[Any, Optional[str]]:
    if not isinstance(tool, dict):
        return tool, None
    if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
        return tool, None

    updated_function, alias_name = _apply_google_code_assist_alias_to_function_block(
        dict(tool["function"]),
        aliases=aliases,
        tool_name_mapping=tool_name_mapping,
    )
    if alias_name is None:
        return tool, None

    updated_tool = dict(tool)
    updated_tool["function"] = updated_function
    return updated_tool, alias_name


def _apply_google_code_assist_aliases_to_tool_calls(
    tool_calls: Any,
    *,
    aliases: dict[str, str],
    tool_name_mapping: dict[str, str],
) -> tuple[Any, set[str]]:
    if not isinstance(tool_calls, list):
        return tool_calls, set()

    updated_tool_calls: list[Any] = []
    aliased_names: set[str] = set()
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            updated_tool_calls.append(tool_call)
            continue
        function_block = tool_call.get("function")
        if not isinstance(function_block, dict):
            updated_tool_calls.append(tool_call)
            continue

        (
            updated_function,
            alias_name,
        ) = _apply_google_code_assist_alias_to_function_block(
            function_block,
            aliases=aliases,
            tool_name_mapping=tool_name_mapping,
        )
        if alias_name is None:
            updated_tool_calls.append(tool_call)
            continue

        updated_tool_call = dict(tool_call)
        updated_tool_call["function"] = updated_function
        updated_tool_calls.append(updated_tool_call)
        aliased_names.add(alias_name)

    return updated_tool_calls, aliased_names


def _apply_google_code_assist_aliases_to_message(
    message: Any,
    *,
    aliases: dict[str, str],
    tool_name_mapping: dict[str, str],
) -> tuple[Any, set[str]]:
    if not isinstance(message, dict):
        return message, set()
    updated_tool_calls, aliased_names = _apply_google_code_assist_aliases_to_tool_calls(
        message.get("tool_calls"),
        aliases=aliases,
        tool_name_mapping=tool_name_mapping,
    )
    if not aliased_names:
        return message, set()

    updated_message = dict(message)
    updated_message["tool_calls"] = updated_tool_calls
    return updated_message, aliased_names


def _apply_google_code_assist_native_tool_aliases(
    completion_kwargs: dict[str, Any],
    tool_name_mapping: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aliases = _runtime()._get_google_code_assist_native_tool_aliases()
    alias_count = 0
    aliased_names: set[str] = set()

    tools = completion_kwargs.get("tools")
    if isinstance(tools, list):
        updated_tools = []
        for tool in tools:
            updated_tool, alias_name = _apply_google_code_assist_alias_to_tool(
                tool,
                aliases=aliases,
                tool_name_mapping=tool_name_mapping,
            )
            updated_tools.append(updated_tool)
            if alias_name is not None:
                alias_count += 1
                aliased_names.add(alias_name)
        completion_kwargs["tools"] = updated_tools

    messages = completion_kwargs.get("messages")
    if isinstance(messages, list):
        updated_messages = []
        for message in messages:
            (
                updated_message,
                message_aliases,
            ) = _apply_google_code_assist_aliases_to_message(
                message,
                aliases=aliases,
                tool_name_mapping=tool_name_mapping,
            )
            updated_messages.append(updated_message)
            aliased_names.update(message_aliases)
        completion_kwargs["messages"] = updated_messages

    tool_choice = completion_kwargs.get("tool_choice")
    if isinstance(tool_choice, dict):
        updated_tool_choice = dict(tool_choice)
        function_block = updated_tool_choice.get("function")
        if isinstance(function_block, dict):
            (
                updated_function,
                alias_name,
            ) = _apply_google_code_assist_alias_to_function_block(
                function_block,
                aliases=aliases,
                tool_name_mapping=tool_name_mapping,
            )
            if alias_name is not None:
                updated_tool_choice["function"] = updated_function
                completion_kwargs["tool_choice"] = updated_tool_choice
                aliased_names.add(alias_name)
                alias_count += 1

    changes: dict[str, Any] = {}
    if alias_count > 0 or aliased_names:
        changes["google_native_tool_alias_count"] = alias_count
        changes["google_native_tool_aliases"] = sorted(aliased_names)
    return completion_kwargs, changes


def _inject_google_adapter_tool_call_context_text(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated_messages: list[dict[str, Any]] = []
    suppressed_count = 0

    for message in messages:
        if not isinstance(message, dict):
            updated_messages.append(message)
            continue
        if message.get("role") != "assistant":
            updated_messages.append(message)
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) == 0:
            updated_messages.append(message)
            continue
        if _runtime()._completion_message_has_visible_text(message):
            if _runtime()._is_google_adapter_synthetic_tool_context_message(message):
                updated_message = dict(message)
                updated_message["content"] = ""
                updated_messages.append(updated_message)
                suppressed_count += 1
                continue
            updated_messages.append(message)
            continue

        updated_messages.append(message)

    if suppressed_count == 0:
        return messages, {}
    return updated_messages, {
        "google_adapter_suppressed_tool_call_context_text_count": suppressed_count,
    }


def _extract_google_adapter_preserved_task_excerpt(text: str) -> str:
    text_value = text.strip()
    reminder_matches = _runtime()._aawm_provider_shaping.iter_delimited_spans(
        text_value,
        "<system-reminder>",
        "</system-reminder>",
    )
    if reminder_matches:
        trailing_text = text_value[reminder_matches[-1].end :].strip()
        if trailing_text:
            text_value = trailing_text

    cap = _runtime()._get_google_adapter_preserved_task_state_char_cap()
    if len(text_value) <= cap:
        return text_value
    return text_value[-cap:].lstrip()


def _build_google_adapter_preserved_task_state_message(
    messages: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Preserve active task state using structured/configurable contract (RR-054 #35)."""

    def _is_skippable(message: dict[str, Any]) -> bool:
        if message.get("role") == "tool":
            return True
        if _runtime()._completion_message_has_tool_result(message):
            return True
        if _runtime()._is_google_adapter_synthetic_tool_context_message(message):
            return True
        return False

    selected = _aawm_task_state.select_task_state_source(
        messages,
        extract_text=_runtime()._extract_completion_message_text,
        is_skippable=_is_skippable,
    )
    if selected is None:
        return None, {}

    source_index, source_text, source_kind = selected
    excerpt = _extract_google_adapter_preserved_task_excerpt(source_text)
    if not excerpt:
        return None, {}

    preserved_text = (
        "<system-reminder>\n"
        "Gemini adapter preserved active child-agent task state from trimmed conversation. "
        "Continue to follow this original task and its next-tool obligations.\n\n"
        "Original task excerpt:\n"
        f"{excerpt}\n"
        "</system-reminder>"
    )
    return {
        "role": "user",
        "content": preserved_text,
    }, {
        "preserved_active_task_state": True,
        "preserved_active_task_state_chars": len(excerpt),
        "preserved_active_task_state_source_index": source_index,
        "preserved_active_task_state_source_kind": source_kind,
    }


def _apply_google_adapter_completion_message_window(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(messages) <= 2:
        return messages, {}
    max_window = _runtime()._get_google_adapter_max_completion_messages_window()
    original_count = len(messages)
    original_text_chars = sum(_runtime()._estimate_completion_message_text_chars(message) for message in messages)
    trimmed_messages = list(messages[-max_window:])
    preserved_task_message: Optional[dict[str, Any]] = None
    preserved_task_changes: dict[str, Any] = {}
    if max_window >= 3 and any(_runtime()._completion_message_has_tool_result(message) for message in messages):
        first_retained_index = max(0, original_count - max_window)
        initial_text_index = next(
            (
                index
                for index, message in enumerate(messages)
                if _runtime()._extract_completion_message_text(message).strip()
            ),
            None,
        )
        if initial_text_index is not None and initial_text_index < first_retained_index:
            (
                preserved_task_message,
                preserved_task_changes,
            ) = _build_google_adapter_preserved_task_state_message(messages)
            if preserved_task_message is not None:
                (
                    retained_tail,
                    tail_boundary_changes,
                ) = _runtime()._trim_completion_message_tail_preserving_tool_pairs(messages, max_window - 1)
                trimmed_messages = [
                    preserved_task_message,
                    *retained_tail,
                ]
                preserved_task_changes = {
                    **preserved_task_changes,
                    **tail_boundary_changes,
                }
    trimmed_text_chars = sum(
        _runtime()._estimate_completion_message_text_chars(message) for message in trimmed_messages
    )
    if len(trimmed_messages) == original_count:
        return messages, {}
    return trimmed_messages, {
        "trimmed_completion_messages_from_count": original_count,
        "trimmed_completion_messages_to_count": len(trimmed_messages),
        "trimmed_completion_messages_from_text_chars": original_text_chars,
        "trimmed_completion_messages_to_text_chars": trimmed_text_chars,
        "trimmed_completion_messages_max_window": max_window,
        **preserved_task_changes,
    }


def _normalize_google_code_assist_httpx_payload(value: Any) -> Any:
    key_mapping = {
        "function_call": "functionCall",
        "function_response": "functionResponse",
        "inline_data": "inlineData",
        "file_data": "fileData",
        "mime_type": "mimeType",
        "file_uri": "fileUri",
        "media_resolution": "mediaResolution",
        "function_declarations": "functionDeclarations",
        "allowed_function_names": "allowedFunctionNames",
    }
    if isinstance(value, list):
        return [_normalize_google_code_assist_httpx_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = key_mapping.get(key, key) if isinstance(key, str) else str(key)
        normalized[normalized_key] = _normalize_google_code_assist_httpx_payload(item)
    return normalized


def _annotate_google_code_assist_duplicate_tool_response_parts(
    contents: list[Any],
    duplicate_tool_results: list[tuple[str, str]],
    *,
    annotate_function_response_id: bool = False,
) -> int:
    annotated_count = 0
    pending_index = 0
    for content in contents:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if pending_index >= len(duplicate_tool_results):
                break
            if not isinstance(part, dict):
                continue
            function_response = part.get("functionResponse")
            if not isinstance(function_response, dict):
                continue
            function_name, tool_call_id = duplicate_tool_results[pending_index]
            if function_response.get("name") != function_name:
                continue
            response_payload = function_response.get("response")
            if not isinstance(response_payload, dict):
                response_payload = {}
                function_response["response"] = response_payload
            if annotate_function_response_id:
                function_response.setdefault("id", tool_call_id)
            response_payload.setdefault("tool_use_id", tool_call_id)
            annotated_count += 1
            pending_index += 1
    return annotated_count


def _annotate_google_code_assist_duplicate_tool_responses(
    google_request_dict: dict[str, Any],
    completion_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve Claude tool_use ids when Gemini has same-name parallel tool results."""
    duplicate_tool_results = _runtime()._google_code_assist_duplicate_tool_results_from_completion_messages(
        completion_messages
    )
    if not duplicate_tool_results:
        return {}

    contents = google_request_dict.get("contents")
    if not isinstance(contents, list):
        return {}

    annotated_count = _annotate_google_code_assist_duplicate_tool_response_parts(
        contents,
        duplicate_tool_results,
    )
    if annotated_count == 0:
        return {}
    return {
        "google_adapter_annotated_duplicate_tool_response_count": annotated_count,
    }


def _annotate_google_code_assist_claude_tool_response_ids(
    google_request_dict: dict[str, Any],
    completion_messages: list[dict[str, Any]],
    *,
    google_model: str,
) -> dict[str, Any]:
    if "claude" not in google_model.lower():
        return {}

    tool_results = _runtime()._google_code_assist_tool_results_from_completion_messages(completion_messages)
    if not tool_results:
        return {}

    contents = google_request_dict.get("contents")
    if not isinstance(contents, list):
        return {}

    annotated_count = _annotate_google_code_assist_duplicate_tool_response_parts(
        contents,
        tool_results,
        annotate_function_response_id=True,
    )
    if annotated_count == 0:
        return {}
    return {
        "google_adapter_annotated_claude_tool_response_id_count": annotated_count,
    }


def _insert_google_code_assist_missing_claude_function_call_pairs(
    google_request_dict: dict[str, Any],
    *,
    google_model: str,
    scope_key: Optional[str] = None,
) -> dict[str, Any]:
    if "claude" not in google_model.lower():
        return {}

    contents = google_request_dict.get("contents")
    if not isinstance(contents, list):
        return {}

    updated_contents: list[Any] = []
    seen_function_call_ids: set[str] = set()
    inserted_count = 0

    for content in contents:
        if not isinstance(content, dict):
            updated_contents.append(content)
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            updated_contents.append(content)
            continue

        missing_function_call_parts: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                function_call_id = function_call.get("id")
                if isinstance(function_call_id, str) and function_call_id.strip():
                    seen_function_call_ids.add(function_call_id.strip())
                continue

            function_response = part.get("functionResponse")
            if not isinstance(function_response, dict):
                continue
            tool_call_id = _runtime()._google_code_assist_function_response_id(function_response)
            function_name = function_response.get("name")
            if (
                not isinstance(tool_call_id, str)
                or not isinstance(function_name, str)
                or not function_name
                or tool_call_id in seen_function_call_ids
            ):
                continue
            missing_function_call_parts.append(
                {
                    "functionCall": {
                        "name": function_name,
                        "args": _runtime()._google_code_assist_function_call_args_for_id(
                            tool_call_id,
                            scope_key=scope_key,
                        ),
                        "id": tool_call_id,
                    }
                }
            )
            seen_function_call_ids.add(tool_call_id)
            inserted_count += 1

        if missing_function_call_parts:
            updated_contents.append({"role": "model", "parts": missing_function_call_parts})
        updated_contents.append(content)

    if inserted_count == 0:
        return {}

    google_request_dict["contents"] = updated_contents
    return {"google_adapter_inserted_claude_function_call_pair_count": inserted_count}
