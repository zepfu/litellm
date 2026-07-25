"""Wave 4 extraction: google_context_window pure-leaf functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

from typing import Any, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global modules
    _anthropic_google_shaping: Any

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_google_content_has_function_exchange",
    "_google_content_has_function_call",
    "_apply_google_adapter_contents_window_policy",
    "_extract_completion_message_text",
    "_completion_message_has_visible_text",
    "_estimate_completion_message_text_chars",
    "_completion_message_has_tool_result",
    "_completion_message_tool_call_ids",
    "_completion_message_tool_result_ids",
    "_trim_completion_message_tail_preserving_tool_pairs",
    "_apply_google_adapter_completion_message_window",
    "_google_code_assist_duplicate_tool_results_from_completion_messages",
    "_google_code_assist_tool_results_from_completion_messages",
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

def _google_content_has_function_exchange(content_block: Any) -> bool:
    if not isinstance(content_block, dict):
        return False
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return False
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("functionCall"), dict) or isinstance(part.get("function_call"), dict):
            return True
        if isinstance(part.get("functionResponse"), dict) or isinstance(part.get("function_response"), dict):
            return True
    return False

def _google_content_has_function_call(content_block: Any) -> bool:
    if not isinstance(content_block, dict):
        return False
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return False
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("functionCall"), dict) or isinstance(part.get("function_call"), dict):
            return True
    return False

def _apply_google_adapter_contents_window_policy(request_block: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._apply_google_adapter_contents_window_policy(request_block)  # noqa: F821

def _extract_completion_message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts)

def _completion_message_has_visible_text(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return True
    return False

def _estimate_completion_message_text_chars(message: Any) -> int:
    if not isinstance(message, dict):
        return 0
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                total += len(text)
        return total
    return 0

def _completion_message_has_tool_result(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") == "tool":
        return True
    if isinstance(message.get("tool_call_id"), str):
        return True
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_result":
                return True
            if isinstance(part.get("tool_result"), dict):
                return True
    return False

def _completion_message_tool_call_ids(message: Any) -> set[str]:
    if not isinstance(message, dict):
        return set()
    tool_call_ids: set[str] = set()
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                tool_call_ids.add(tool_call_id)
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use":
                tool_call_id = part.get("id")
                if isinstance(tool_call_id, str) and tool_call_id:
                    tool_call_ids.add(tool_call_id)
    return tool_call_ids

def _completion_message_tool_result_ids(message: Any) -> set[str]:
    if not isinstance(message, dict):
        return set()
    tool_result_ids: set[str] = set()
    tool_call_id = message.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        tool_result_ids.add(tool_call_id)
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_tool_use_id = part.get("tool_use_id")
            if isinstance(part_tool_use_id, str) and part_tool_use_id:
                tool_result_ids.add(part_tool_use_id)
            tool_result = part.get("tool_result")
            if isinstance(tool_result, dict):
                nested_tool_use_id = tool_result.get("tool_use_id")
                if isinstance(nested_tool_use_id, str) and nested_tool_use_id:
                    tool_result_ids.add(nested_tool_use_id)
    return tool_result_ids

def _trim_completion_message_tail_preserving_tool_pairs(
    messages: list[dict[str, Any]],
    tail_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tail_budget <= 0:
        return [], {}

    tail_start = max(0, len(messages) - tail_budget)
    boundary_adjustments = 0
    while tail_start < len(messages) and _completion_message_has_tool_result(messages[tail_start]):
        tail_start += 1
        boundary_adjustments += 1

    while tail_start < len(messages):
        seen_tool_call_ids: set[str] = set()
        orphan_index: Optional[int] = None
        for index, message in enumerate(messages[tail_start:]):
            seen_tool_call_ids.update(_completion_message_tool_call_ids(message))
            tool_result_ids = _completion_message_tool_result_ids(message)
            if tool_result_ids and not tool_result_ids.issubset(seen_tool_call_ids):
                orphan_index = tail_start + index
                break
        if orphan_index is None:
            break
        tail_start = orphan_index + 1
        boundary_adjustments += 1
        while tail_start < len(messages) and _completion_message_has_tool_result(messages[tail_start]):
            tail_start += 1
            boundary_adjustments += 1

    changes: dict[str, Any] = {}
    if boundary_adjustments:
        changes["trimmed_completion_messages_tool_pair_boundary_adjustments"] = boundary_adjustments
    return messages[tail_start:], changes

def _apply_google_adapter_completion_message_window(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._apply_google_adapter_completion_message_window(messages)  # noqa: F821

def _google_code_assist_duplicate_tool_results_from_completion_messages(
    completion_messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    duplicate_tool_results: list[tuple[str, str]] = []
    pending_tool_calls_by_id: dict[str, str] = {}
    duplicate_tool_call_names: set[str] = set()

    for message in completion_messages:
        role = message.get("role")
        if role == "assistant":
            pending_tool_calls_by_id.clear()
            duplicate_tool_call_names.clear()
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            tool_call_name_counts: dict[str, int] = {}
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(tool_call_id, str) or not isinstance(function, dict):
                    continue
                function_name = function.get("name")
                if not isinstance(function_name, str) or not function_name:
                    continue
                pending_tool_calls_by_id[tool_call_id] = function_name
                tool_call_name_counts[function_name] = tool_call_name_counts.get(function_name, 0) + 1
            duplicate_tool_call_names = {name for name, count in tool_call_name_counts.items() if count > 1}
            continue

        if role != "tool":
            pending_tool_calls_by_id.clear()
            duplicate_tool_call_names.clear()
            continue

        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            continue
        function_name = pending_tool_calls_by_id.get(tool_call_id)
        if function_name in duplicate_tool_call_names:
            duplicate_tool_results.append((function_name, tool_call_id))

    return duplicate_tool_results

def _google_code_assist_tool_results_from_completion_messages(
    completion_messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    tool_results: list[tuple[str, str]] = []
    pending_tool_calls_by_id: dict[str, str] = {}

    for message in completion_messages:
        role = message.get("role")
        if role == "assistant":
            pending_tool_calls_by_id.clear()
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(tool_call_id, str) or not isinstance(function, dict):
                    continue
                function_name = function.get("name")
                if isinstance(function_name, str) and function_name:
                    pending_tool_calls_by_id[tool_call_id] = function_name
            continue

        if role != "tool":
            pending_tool_calls_by_id.clear()
            continue

        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            continue
        function_name = pending_tool_calls_by_id.get(tool_call_id)
        if isinstance(function_name, str) and function_name:
            tool_results.append((function_name, tool_call_id))

    return tool_results
