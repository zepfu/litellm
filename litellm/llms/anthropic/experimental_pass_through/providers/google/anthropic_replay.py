"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Callable, Optional, cast


@dataclass(frozen=True)
class Runtime:
    _codex_google_code_assist_anthropic_tool_result_to_openai_tool_message: Any
    _codex_google_code_assist_anthropic_tool_use_to_openai_tool_call: Any
    _is_anthropic_tool_result_content_block: Any
    _is_anthropic_tool_use_content_block: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _codex_google_code_assist_anthropic_tool_result_to_openai_tool_message=namespace[
            "_codex_google_code_assist_anthropic_tool_result_to_openai_tool_message"
        ],
        _codex_google_code_assist_anthropic_tool_use_to_openai_tool_call=namespace[
            "_codex_google_code_assist_anthropic_tool_use_to_openai_tool_call"
        ],
        _is_anthropic_tool_result_content_block=namespace["_is_anthropic_tool_result_content_block"],
        _is_anthropic_tool_use_content_block=namespace["_is_anthropic_tool_use_content_block"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _normalize_codex_openai_chat_kwargs_for_google_code_assist(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    developer_message_count = 0
    normalized_messages: list[Any] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "developer":
            updated_message = dict(message)
            updated_message["role"] = "system"
            normalized_messages.append(updated_message)
            developer_message_count += 1
        else:
            normalized_messages.append(message)

    if developer_message_count == 0:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = normalized_messages
    return updated_kwargs, {"google_adapter_codex_developer_messages_as_system_count": (developer_message_count)}


def _has_codex_google_code_assist_anthropic_tool_replay_blocks(
    messages: list[Any],
) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            _runtime()._is_anthropic_tool_use_content_block(block)
            or _runtime()._is_anthropic_tool_result_content_block(block)
            for block in content
        ):
            return True
    return False


def _normalize_codex_google_code_assist_anthropic_assistant_message(
    *,
    message: dict[str, Any],
    message_index: int,
) -> tuple[dict[str, Any], int]:
    content = message.get("content")
    if not isinstance(content, list):
        return message, 0

    updated_message = dict(message)
    existing_tool_calls = updated_message.get("tool_calls")
    tool_calls = list(existing_tool_calls) if isinstance(existing_tool_calls, list) else []
    text_parts: list[dict[str, Any]] = []
    converted_tool_use_count = 0
    for content_index, block in enumerate(content):
        if _runtime()._is_anthropic_tool_use_content_block(block):
            tool_calls.append(
                _runtime()._codex_google_code_assist_anthropic_tool_use_to_openai_tool_call(
                    block=cast(dict[str, Any], block),
                    message_index=message_index,
                    content_index=content_index,
                )
            )
            converted_tool_use_count += 1
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(
                {
                    "type": "text",
                    "text": str(block.get("text") or ""),
                }
            )

    updated_message["tool_calls"] = tool_calls
    if text_parts:
        updated_message["content"] = text_parts[0]["text"] if len(text_parts) == 1 else text_parts
    else:
        updated_message["content"] = None
    return updated_message, converted_tool_use_count


def _normalize_codex_google_code_assist_anthropic_user_message(
    *,
    message: dict[str, Any],
    message_index: int,
) -> tuple[list[dict[str, Any]], int]:
    content = message.get("content")
    if not isinstance(content, list):
        return [message], 0

    remaining_user_content: list[Any] = []
    normalized_messages: list[dict[str, Any]] = []
    converted_tool_result_count = 0
    for content_index, block in enumerate(content):
        if not _runtime()._is_anthropic_tool_result_content_block(block):
            remaining_user_content.append(block)
            continue
        normalized_messages.append(
            _runtime()._codex_google_code_assist_anthropic_tool_result_to_openai_tool_message(
                block=cast(dict[str, Any], block),
                message_index=message_index,
                content_index=content_index,
            )
        )
        converted_tool_result_count += 1

    if remaining_user_content:
        updated_message = dict(message)
        updated_message["content"] = remaining_user_content
        normalized_messages.append(updated_message)
    return normalized_messages, converted_tool_result_count


def _build_codex_google_code_assist_anthropic_replay_changes(
    *,
    repaired_count: int,
    converted_tool_use_count: int,
    converted_tool_result_count: int,
) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    if repaired_count:
        changes["google_adapter_codex_repaired_anthropic_tool_replay_id_count"] = repaired_count
    if converted_tool_use_count:
        changes["google_adapter_codex_converted_anthropic_tool_use_count"] = converted_tool_use_count
    if converted_tool_result_count:
        changes["google_adapter_codex_converted_anthropic_tool_result_count"] = converted_tool_result_count
    return changes


def _normalize_codex_google_code_assist_anthropic_tool_replay(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list) or not _has_codex_google_code_assist_anthropic_tool_replay_blocks(messages):
        return completion_kwargs, {}

    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    (
        repaired_messages,
        repaired_count,
    ) = LiteLLMAnthropicMessagesAdapter.repair_missing_anthropic_tool_use_ids(messages)

    normalized_messages: list[Any] = []
    converted_tool_use_count = 0
    converted_tool_result_count = 0

    for message_index, message in enumerate(repaired_messages):
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue

        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            normalized_messages.append(message)
            continue

        if role == "assistant" and any(_runtime()._is_anthropic_tool_use_content_block(block) for block in content):
            (
                updated_message,
                message_tool_use_count,
            ) = _normalize_codex_google_code_assist_anthropic_assistant_message(
                message=message,
                message_index=message_index,
            )
            converted_tool_use_count += message_tool_use_count
            normalized_messages.append(updated_message)
            continue

        if role == "user" and any(_runtime()._is_anthropic_tool_result_content_block(block) for block in content):
            (
                new_messages,
                message_tool_result_count,
            ) = _normalize_codex_google_code_assist_anthropic_user_message(
                message=message,
                message_index=message_index,
            )
            normalized_messages.extend(new_messages)
            converted_tool_result_count += message_tool_result_count
            continue

        normalized_messages.append(message)

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = normalized_messages
    return updated_kwargs, _build_codex_google_code_assist_anthropic_replay_changes(
        repaired_count=repaired_count,
        converted_tool_use_count=converted_tool_use_count,
        converted_tool_result_count=converted_tool_result_count,
    )


def _deterministic_codex_google_code_assist_tool_call_id(
    *,
    message_index: int,
    tool_call_index: int,
    tool_call: dict[str, Any],
) -> str:
    try:
        seed_payload = json.dumps(tool_call, sort_keys=True, default=str)
    except Exception:
        seed_payload = str(tool_call)
    seed = "|".join(
        (
            "codex-google-code-assist-tool-call-id",
            str(message_index),
            str(tool_call_index),
            seed_payload,
        )
    )
    return f"call_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:28]}"


def _next_codex_google_code_assist_tool_messages(
    messages: list[Any],
    *,
    message_index: int,
) -> list[tuple[int, dict[str, Any]]]:
    next_tool_messages: list[tuple[int, dict[str, Any]]] = []
    for next_index in range(message_index + 1, len(messages)):
        next_message = messages[next_index]
        if not isinstance(next_message, dict):
            continue
        if next_message.get("role") == "assistant":
            break
        if next_message.get("role") == "tool":
            next_tool_messages.append((next_index, next_message))
    return next_tool_messages


def _paired_codex_google_code_assist_tool_message(
    next_tool_messages: list[tuple[int, dict[str, Any]]],
    *,
    tool_call_index: int,
) -> tuple[int, dict[str, Any]] | None:
    if tool_call_index < len(next_tool_messages):
        return next_tool_messages[tool_call_index]
    return None


def _repair_codex_google_code_assist_tool_call_id(
    *,
    message_index: int,
    tool_call_index: int,
    tool_call: dict[str, Any],
    paired_tool_message: tuple[int, dict[str, Any]] | None,
    copy_message_at: Callable[[int], Optional[dict[str, Any]]],
) -> bool:
    existing_id = tool_call.get("id")
    if isinstance(existing_id, str) and existing_id.strip():
        return False

    paired_tool_call_id = paired_tool_message[1].get("tool_call_id") if paired_tool_message is not None else None
    repaired_id = (
        paired_tool_call_id.strip()
        if isinstance(paired_tool_call_id, str) and paired_tool_call_id.strip()
        else _deterministic_codex_google_code_assist_tool_call_id(
            message_index=message_index,
            tool_call_index=tool_call_index,
            tool_call=tool_call,
        )
    )

    assistant_copy = copy_message_at(message_index)
    if assistant_copy is None:
        return False
    copied_tool_calls = assistant_copy.get("tool_calls")
    if not isinstance(copied_tool_calls, list):
        return False
    copied_tool_call = copied_tool_calls[tool_call_index]
    if not isinstance(copied_tool_call, dict):
        return False
    copied_tool_call["id"] = repaired_id

    if paired_tool_message is not None and not (isinstance(paired_tool_call_id, str) and paired_tool_call_id.strip()):
        tool_message_copy = copy_message_at(paired_tool_message[0])
        if tool_message_copy is not None:
            tool_message_copy["tool_call_id"] = repaired_id

    return True


def _repair_codex_google_code_assist_openai_tool_call_ids(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    updated_messages = list(messages)
    copied_messages: set[int] = set()
    repaired_count = 0

    def copy_message_at(index: int) -> Optional[dict[str, Any]]:
        message = updated_messages[index]
        if not isinstance(message, dict):
            return None
        if index not in copied_messages:
            message = dict(message)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                message["tool_calls"] = [
                    dict(tool_call) if isinstance(tool_call, dict) else tool_call for tool_call in tool_calls
                ]
            updated_messages[index] = message
            copied_messages.add(index)
        return cast(dict[str, Any], message)

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        next_tool_messages = _next_codex_google_code_assist_tool_messages(
            messages,
            message_index=message_index,
        )

        for tool_call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            repaired = _repair_codex_google_code_assist_tool_call_id(
                message_index=message_index,
                tool_call_index=tool_call_index,
                tool_call=tool_call,
                paired_tool_message=_paired_codex_google_code_assist_tool_message(
                    next_tool_messages,
                    tool_call_index=tool_call_index,
                ),
                copy_message_at=copy_message_at,
            )
            if repaired:
                repaired_count += 1

    if repaired_count == 0:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    return updated_kwargs, {"google_adapter_codex_repaired_openai_tool_call_id_count": repaired_count}
