"""Path-aware visible-text extraction for OpenAI Responses and Chat Completions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

_MESSAGE_ITEM_TYPES = frozenset({"message", "input_message", "output_message"})
_VISIBLE_PART_TYPES = frozenset(
    {"input_text", "output_text", "text", "refusal"}
)
_SKIP_ITEM_TYPES = frozenset(
    {
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "tool_call",
        "tool_result",
        "reasoning",
        "item_reference",
        "computer_call",
        "computer_call_output",
        "file_search_call",
        "web_search_call",
        "image_generation_call",
        "code_interpreter_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
    }
)
_PROTECTED_KEYS = frozenset(
    {
        "arguments",
        "encrypted_content",
        "encrypted_reasoning",
        "signature",
        "id",
        "call_id",
        "tool_call_id",
        "name",
        "model",
        "tools",
        "functions",
        "tool_calls",
        "parameters",
        "schema",
        "metadata",
        "url",
        "file_id",
        "image_url",
        "input_image",
        "previous_response_id",
    }
)


@dataclass(frozen=True)
class VisibleTextNode:
    path: str
    text: str
    role: Optional[str] = None
    path_parts: tuple[Any, ...] = ()


def _format_path(parts: Sequence[Any]) -> str:
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        elif not out:
            out = str(part)
        else:
            out += f".{part}"
    return out or ""


def _normalize_endpoint(endpoint: str) -> str:
    text = str(endpoint or "").strip().lower().replace("-", "_")
    if "chat/completions" in text or text.endswith("chat_completions"):
        return "chat_completions"
    if text.endswith("responses") or "/responses" in text or text == "responses":
        return "responses"
    return text


def _yield_text(
    path_parts: tuple[Any, ...],
    text: Any,
    role: Optional[str],
) -> Iterator[VisibleTextNode]:
    if isinstance(text, str) and text != "":
        yield VisibleTextNode(
            path=_format_path(path_parts),
            text=text,
            role=role,
            path_parts=path_parts,
        )


def _iter_content_parts(
    content: Any,
    path_parts: tuple[Any, ...],
    role: Optional[str],
) -> Iterator[VisibleTextNode]:
    if isinstance(content, str):
        yield from _yield_text(path_parts, content, role)
        return
    if not isinstance(content, list):
        return
    for index, part in enumerate(content):
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type not in _VISIBLE_PART_TYPES and part_type is not None:
            continue
        text = part.get("text")
        if not isinstance(text, str):
            continue
        yield from _yield_text(path_parts + (index, "text"), text, role)


def _iter_responses_input_items(
    items: list[Any],
    path_prefix: tuple[Any, ...],
) -> Iterator[VisibleTextNode]:
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        item_path = path_prefix + (index,)
        if item_type in _SKIP_ITEM_TYPES:
            continue
        if item_type in _MESSAGE_ITEM_TYPES or "content" in item:
            role = item.get("role") if isinstance(item.get("role"), str) else None
            yield from _iter_content_parts(
                item.get("content"), item_path + ("content",), role
            )
            continue
        if item_type in _VISIBLE_PART_TYPES:
            yield from _yield_text(
                item_path + ("text",),
                item.get("text"),
                "user",
            )


def _iter_responses_request(body: Mapping[str, Any]) -> Iterator[VisibleTextNode]:
    instructions = body.get("instructions")
    if isinstance(instructions, str):
        yield from _yield_text(("instructions",), instructions, "system")
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        yield from _yield_text(("input",), raw_input, "user")
    elif isinstance(raw_input, list):
        yield from _iter_responses_input_items(raw_input, ("input",))


def _iter_responses_response(body: Mapping[str, Any]) -> Iterator[VisibleTextNode]:
    output_text = body.get("output_text")
    if isinstance(output_text, str):
        yield from _yield_text(("output_text",), output_text, "assistant")
    output = body.get("output")
    if not isinstance(output, list):
        return
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type in _SKIP_ITEM_TYPES:
            continue
        if item_type in _MESSAGE_ITEM_TYPES or item_type is None:
            role = item.get("role") if isinstance(item.get("role"), str) else "assistant"
            yield from _iter_content_parts(
                item.get("content"),
                ("output", index, "content"),
                role,
            )
        refusal = item.get("refusal")
        if isinstance(refusal, str):
            yield from _yield_text(("output", index, "refusal"), refusal, "assistant")


def _iter_chat_messages(
    messages: Any,
    path_key: str,
) -> Iterator[VisibleTextNode]:
    if not isinstance(messages, list):
        return
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role") if isinstance(message.get("role"), str) else None
        yield from _iter_content_parts(
            message.get("content"),
            (path_key, index, "content"),
            role,
        )


def _iter_chat_request(body: Mapping[str, Any]) -> Iterator[VisibleTextNode]:
    yield from _iter_chat_messages(body.get("messages"), "messages")


def _iter_chat_response(body: Mapping[str, Any]) -> Iterator[VisibleTextNode]:
    choices = body.get("choices")
    if not isinstance(choices, list):
        return
    for index, choice in enumerate(choices):
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if isinstance(message, Mapping):
            role = message.get("role") if isinstance(message.get("role"), str) else "assistant"
            yield from _iter_content_parts(
                message.get("content"),
                ("choices", index, "message", "content"),
                role,
            )
            refusal = message.get("refusal")
            if isinstance(refusal, str):
                yield from _yield_text(
                    ("choices", index, "message", "refusal"),
                    refusal,
                    role,
                )
        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            yield from _iter_content_parts(
                delta.get("content"),
                ("choices", index, "delta", "content"),
                "assistant",
            )


def extract_visible_text_nodes(
    body: Any,
    *,
    endpoint: str,
    direction: str = "request",
) -> Iterator[VisibleTextNode]:
    """Yield user-visible text nodes. Protected surfaces are not visited."""

    if not isinstance(body, Mapping):
        return
    family = _normalize_endpoint(endpoint)
    direction_name = str(direction or "request").strip().lower()
    if family == "chat_completions":
        iterator = (
            _iter_chat_request(body)
            if direction_name != "response"
            else _iter_chat_response(body)
        )
    else:
        iterator = (
            _iter_responses_request(body)
            if direction_name != "response"
            else _iter_responses_response(body)
        )
    for node in iterator:
        path_l = node.path.lower()
        if any(key in path_l.split(".") for key in _PROTECTED_KEYS):
            continue
        if "arguments" in path_l or "encrypted" in path_l:
            continue
        yield node


def assign_text_node(body: MutableMapping[str, Any], node: VisibleTextNode, text: str) -> None:
    """Write sanitized text back onto a previously extracted node path."""

    if not node.path_parts:
        return
    current: Any = body
    for part in node.path_parts[:-1]:
        current = current[part]
    current[node.path_parts[-1]] = text
