"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import json
from collections.abc import Mapping
from typing import Any, Optional


@dataclass(frozen=True)
class Runtime:
    _codex_google_code_assist_display_tool_call_id: Any
    _codex_google_code_assist_orphan_tool_result_context_text: Any
    _codex_google_code_assist_tool_call_function_arguments: Any
    _codex_google_code_assist_tool_call_function_name: Any
    _codex_google_code_assist_tool_result_message_content: Any
    _completion_message_has_tool_result: Any
    _completion_message_tool_call_ids: Any
    _lookup_codex_google_code_assist_tool_call_arguments: Any
    _lookup_codex_google_code_assist_tool_call_name: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _codex_google_code_assist_display_tool_call_id=namespace["_codex_google_code_assist_display_tool_call_id"],
        _codex_google_code_assist_orphan_tool_result_context_text=namespace[
            "_codex_google_code_assist_orphan_tool_result_context_text"
        ],
        _codex_google_code_assist_tool_call_function_arguments=namespace[
            "_codex_google_code_assist_tool_call_function_arguments"
        ],
        _codex_google_code_assist_tool_call_function_name=namespace[
            "_codex_google_code_assist_tool_call_function_name"
        ],
        _codex_google_code_assist_tool_result_message_content=namespace[
            "_codex_google_code_assist_tool_result_message_content"
        ],
        _completion_message_has_tool_result=namespace["_completion_message_has_tool_result"],
        _completion_message_tool_call_ids=namespace["_completion_message_tool_call_ids"],
        _lookup_codex_google_code_assist_tool_call_arguments=namespace[
            "_lookup_codex_google_code_assist_tool_call_arguments"
        ],
        _lookup_codex_google_code_assist_tool_call_name=namespace["_lookup_codex_google_code_assist_tool_call_name"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _normalize_codex_google_code_assist_reasoning_effort(
    mappable_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reasoning_effort = mappable_params.get("reasoning_effort")
    if reasoning_effort != "xhigh":
        return mappable_params, {}
    updated_params = dict(mappable_params)
    updated_params["reasoning_effort"] = "high"
    return updated_params, {
        "google_adapter_codex_reasoning_effort_normalized_from": "xhigh",
        "google_adapter_codex_reasoning_effort_normalized_to": "high",
    }


def _normalize_google_code_assist_thinking_max_tokens(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thinking = completion_kwargs.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return completion_kwargs, {}

    budget_tokens = thinking.get("budget_tokens")
    max_tokens = completion_kwargs.get("max_tokens")
    if (
        not isinstance(budget_tokens, int)
        or isinstance(budget_tokens, bool)
        or budget_tokens <= 0
        or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens > budget_tokens
    ):
        return completion_kwargs, {}

    normalized_max_tokens = budget_tokens + 1024
    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["max_tokens"] = normalized_max_tokens
    return updated_kwargs, {
        "google_adapter_thinking_max_tokens_normalized": True,
        "google_adapter_thinking_budget_tokens": budget_tokens,
        "google_adapter_thinking_original_max_tokens": max_tokens,
        "google_adapter_thinking_normalized_max_tokens": normalized_max_tokens,
    }


def _normalize_codex_google_code_assist_tool_call_arguments(
    function_arguments: Any,
) -> Optional[str]:
    if function_arguments is None:
        return None
    if isinstance(function_arguments, str):
        return function_arguments
    if isinstance(function_arguments, (dict, list)):
        try:
            return json.dumps(function_arguments, separators=(",", ":"))
        except Exception:
            return None
    return None


def _infer_single_codex_google_code_assist_function_tool_name(
    tools: Any,
) -> Optional[str]:
    if not isinstance(tools, list):
        return None
    function_names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
        else:
            name = tool.get("name")
        if isinstance(name, str) and name:
            function_names.append(name)
    if len(function_names) == 1:
        return function_names[0]
    return None


def _is_codex_google_code_assist_empty_text_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if not isinstance(content, list):
        return False
    if not content:
        return True
    for part in content:
        if isinstance(part, str):
            if part.strip():
                return False
            continue
        if not isinstance(part, dict):
            return False
        part_type = part.get("type")
        if part_type not in (None, "text", "output_text"):
            return False
        text = part.get("text")
        if not isinstance(text, str) or text.strip():
            return False
    return True


def _previous_codex_google_code_assist_assistant_index(
    messages: list[Any],
    *,
    before_index: int,
) -> Optional[int]:
    for candidate_index in range(before_index - 1, -1, -1):
        candidate = messages[candidate_index]
        if isinstance(candidate, dict) and candidate.get("role") == "assistant":
            return candidate_index
    return None


def _previous_codex_google_code_assist_contiguous_assistant_index(
    messages: list[Any],
    *,
    before_index: int,
) -> Optional[int]:
    previous_assistant_index = _previous_codex_google_code_assist_assistant_index(
        messages,
        before_index=before_index,
    )
    if previous_assistant_index is None:
        return None
    for candidate_index in range(before_index - 1, previous_assistant_index, -1):
        candidate = messages[candidate_index]
        if not isinstance(candidate, dict) or candidate.get("role") != "tool":
            return None
    return previous_assistant_index


def _previous_codex_google_code_assist_tool_call(
    messages: list[Any],
    *,
    before_index: int,
    tool_call_id: str,
) -> Optional[dict[str, Any]]:
    previous_assistant_index = _previous_codex_google_code_assist_assistant_index(
        messages,
        before_index=before_index,
    )
    if previous_assistant_index is None:
        return None
    previous_assistant = messages[previous_assistant_index]
    if not isinstance(previous_assistant, dict):
        return None
    tool_calls = previous_assistant.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        if tool_call.get("id") == tool_call_id:
            return tool_call
    return None


def _build_codex_google_code_assist_synthetic_tool_call(
    *,
    tool_call_id: str,
    function_name: str,
    function_arguments: str,
) -> dict[str, Any]:
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": function_arguments,
        },
    }


def _append_codex_google_code_assist_tool_call_to_assistant(
    *,
    assistant_message: dict[str, Any],
    synthetic_tool_call: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    updated_assistant = dict(assistant_message)
    tool_calls = updated_assistant.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    blank_text_suppressed = False
    if _is_codex_google_code_assist_empty_text_content(updated_assistant.get("content")):
        updated_assistant.pop("content", None)
        blank_text_suppressed = True
    updated_assistant["tool_calls"] = [
        *tool_calls,
        synthetic_tool_call,
    ]
    return updated_assistant, blank_text_suppressed


def _build_codex_google_code_assist_tool_pair_repair_changes(
    *,
    repaired_count: int,
    inserted_count: int,
    blank_text_suppressed_count: int,
    repaired_names: set[str],
) -> dict[str, Any]:
    changes: dict[str, Any] = {
        "google_adapter_codex_repaired_missing_tool_call_names": sorted(repaired_names),
    }
    if repaired_count:
        changes["google_adapter_codex_repaired_missing_tool_call_count"] = repaired_count
    if inserted_count:
        changes["google_adapter_codex_inserted_missing_tool_call_count"] = inserted_count
    if blank_text_suppressed_count:
        changes["google_adapter_codex_repaired_blank_tool_call_text_suppressed_count"] = blank_text_suppressed_count
    return changes


def _append_codex_google_code_assist_orphan_tool_result_context(
    *,
    messages: list[Any],
    index: int,
    context_text: str,
) -> None:
    if index > 0:
        previous_message = messages[index - 1]
        if (
            isinstance(previous_message, dict)
            and previous_message.get("role") == "user"
            and not _runtime()._completion_message_has_tool_result(previous_message)
        ):
            updated_previous = dict(previous_message)
            previous_content = updated_previous.get("content")
            if isinstance(previous_content, str):
                if previous_content.strip():
                    updated_previous["content"] = f"{previous_content.rstrip()}\n\n{context_text}"
                else:
                    updated_previous["content"] = context_text
            elif previous_content is None:
                updated_previous["content"] = context_text
            else:
                updated_previous["content"] = context_text
            messages[index - 1] = updated_previous
            return

    messages.insert(
        index,
        {
            "role": "user",
            "content": context_text,
        },
    )


def _sanitize_codex_google_code_assist_orphan_tool_results(  # noqa: PLR0915
    completion_kwargs: dict[str, Any],
    *,
    scope_key: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    updated_messages = list(messages)
    converted_count = 0
    removed_blank_assistant_count = 0
    converted_tool_call_ids: list[str] = []
    processed_tool_call_ids: set[str] = set()
    fallback_tool_name = _infer_single_codex_google_code_assist_function_tool_name(completion_kwargs.get("tools"))

    index = 0
    while index < len(updated_messages):
        message = updated_messages[index]
        if not isinstance(message, dict) or message.get("role") != "tool":
            index += 1
            continue

        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            index += 1
            continue

        previous_assistant_index = _previous_codex_google_code_assist_assistant_index(
            updated_messages,
            before_index=index,
        )
        previous_assistant = (
            updated_messages[previous_assistant_index] if previous_assistant_index is not None else None
        )
        existing_tool_call_ids = (
            _runtime()._completion_message_tool_call_ids(previous_assistant)
            if isinstance(previous_assistant, dict)
            else set()
        )
        if tool_call_id in existing_tool_call_ids:
            index += 1
            continue

        function_name = (
            _runtime()._lookup_codex_google_code_assist_tool_call_name(tool_call_id, scope_key=scope_key)
            or fallback_tool_name
        )
        if function_name:
            index += 1
            continue

        normalized_tool_call_id = _runtime()._codex_google_code_assist_display_tool_call_id(tool_call_id)
        if normalized_tool_call_id in processed_tool_call_ids:
            updated_messages.pop(index)
            converted_count += 1
            converted_tool_call_ids.append(normalized_tool_call_id)
            continue

        context_text = _runtime()._codex_google_code_assist_orphan_tool_result_context_text(
            tool_call_id=normalized_tool_call_id,
            content=_runtime()._codex_google_code_assist_tool_result_message_content(message),
        )
        updated_messages.pop(index)
        converted_count += 1
        converted_tool_call_ids.append(normalized_tool_call_id)
        processed_tool_call_ids.add(normalized_tool_call_id)

        if (
            previous_assistant_index is not None
            and isinstance(previous_assistant, dict)
            and previous_assistant.get("role") == "assistant"
            and not _runtime()._completion_message_tool_call_ids(previous_assistant)
            and _is_codex_google_code_assist_empty_text_content(previous_assistant.get("content"))
        ):
            if previous_assistant_index < index:
                index -= 1
            updated_messages.pop(previous_assistant_index)
            removed_blank_assistant_count += 1

        _append_codex_google_code_assist_orphan_tool_result_context(
            messages=updated_messages,
            index=index,
            context_text=context_text,
        )
        index += 1

    if converted_count == 0:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    changes: dict[str, Any] = {
        "google_adapter_codex_converted_orphan_tool_result_count": converted_count,
        "google_adapter_codex_converted_orphan_tool_result_ids": sorted(converted_tool_call_ids),
    }
    if removed_blank_assistant_count:
        changes[
            "google_adapter_codex_removed_blank_assistant_before_orphan_tool_result_count"
        ] = removed_blank_assistant_count
    return updated_kwargs, changes


def _ensure_codex_google_code_assist_tool_results_have_calls(
    completion_kwargs: dict[str, Any],
    *,
    scope_key: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    updated_messages = list(messages)
    repaired_count = 0
    inserted_count = 0
    blank_text_suppressed_count = 0
    repaired_names: set[str] = set()
    fallback_tool_name = _infer_single_codex_google_code_assist_function_tool_name(completion_kwargs.get("tools"))

    index = 0
    while index < len(updated_messages):
        message = updated_messages[index]
        if not isinstance(message, dict) or message.get("role") != "tool":
            index += 1
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            index += 1
            continue
        previous_assistant_index = _previous_codex_google_code_assist_contiguous_assistant_index(
            updated_messages,
            before_index=index,
        )
        previous_tool_call = _previous_codex_google_code_assist_tool_call(
            updated_messages,
            before_index=index,
            tool_call_id=tool_call_id,
        )

        function_name = (
            _runtime()._lookup_codex_google_code_assist_tool_call_name(tool_call_id, scope_key=scope_key)
            or fallback_tool_name
            or _runtime()._codex_google_code_assist_tool_call_function_name(previous_tool_call)
        )
        if not function_name:
            index += 1
            continue
        function_arguments = (
            _runtime()._lookup_codex_google_code_assist_tool_call_arguments(tool_call_id, scope_key=scope_key)
            or _runtime()._codex_google_code_assist_tool_call_function_arguments(previous_tool_call)
            or "{}"
        )
        synthetic_tool_call = _build_codex_google_code_assist_synthetic_tool_call(
            tool_call_id=tool_call_id,
            function_name=function_name,
            function_arguments=function_arguments,
        )

        if previous_assistant_index is None:
            updated_messages.insert(
                index,
                {
                    "role": "assistant",
                    "tool_calls": [synthetic_tool_call],
                },
            )
            inserted_count += 1
            repaired_names.add(function_name)
            index += 2
            continue

        previous_assistant = updated_messages[previous_assistant_index]
        if not isinstance(previous_assistant, dict):
            index += 1
            continue
        existing_tool_call_ids = _runtime()._completion_message_tool_call_ids(previous_assistant)
        if tool_call_id in existing_tool_call_ids:
            index += 1
            continue

        (
            updated_assistant,
            blank_text_suppressed,
        ) = _append_codex_google_code_assist_tool_call_to_assistant(
            assistant_message=previous_assistant,
            synthetic_tool_call=synthetic_tool_call,
        )
        if blank_text_suppressed:
            blank_text_suppressed_count += 1
        updated_messages[previous_assistant_index] = updated_assistant
        repaired_count += 1
        repaired_names.add(function_name)
        index += 1

    if repaired_count == 0 and inserted_count == 0:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    return updated_kwargs, _build_codex_google_code_assist_tool_pair_repair_changes(
        repaired_count=repaired_count,
        inserted_count=inserted_count,
        blank_text_suppressed_count=blank_text_suppressed_count,
        repaired_names=repaired_names,
    )
