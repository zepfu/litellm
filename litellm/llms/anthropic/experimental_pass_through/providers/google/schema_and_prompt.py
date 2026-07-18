"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import re
from collections.abc import Mapping
from typing import Any, Optional, cast


@dataclass(frozen=True)
class Runtime:
    _ANTHROPIC_BILLING_HEADER_PREFIX: Any
    _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME: Any
    _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION: Any
    _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT: Any
    _GOOGLE_ADAPTER_CLAUDE_OVERHEAD_MARKERS: Any
    _GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT: Any
    _GOOGLE_ADAPTER_ORIGINAL_SYSTEM_PROMPT_HEADING: Any
    _GOOGLE_ADAPTER_PRESERVED_SYSTEM_PROMPT_HEADING: Any
    _GOOGLE_ADAPTER_SYNTHETIC_TOOL_CONTEXT_PATTERN: Any
    _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME: Any
    _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION: Any
    _extract_completion_message_text: Any
    _get_codex_google_code_assist_tool_contract_policy: Any
    _get_google_adapter_fallback_context_char_cap: Any
    _get_google_adapter_system_prompt_policy: Any
    _google_content_has_text: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _ANTHROPIC_BILLING_HEADER_PREFIX=namespace["_ANTHROPIC_BILLING_HEADER_PREFIX"],
        _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME=namespace[
            "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME"
        ],
        _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION=namespace[
            "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION"
        ],
        _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT=namespace["_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT"],
        _GOOGLE_ADAPTER_CLAUDE_OVERHEAD_MARKERS=namespace["_GOOGLE_ADAPTER_CLAUDE_OVERHEAD_MARKERS"],
        _GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT=namespace["_GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT"],
        _GOOGLE_ADAPTER_ORIGINAL_SYSTEM_PROMPT_HEADING=namespace["_GOOGLE_ADAPTER_ORIGINAL_SYSTEM_PROMPT_HEADING"],
        _GOOGLE_ADAPTER_PRESERVED_SYSTEM_PROMPT_HEADING=namespace["_GOOGLE_ADAPTER_PRESERVED_SYSTEM_PROMPT_HEADING"],
        _GOOGLE_ADAPTER_SYNTHETIC_TOOL_CONTEXT_PATTERN=namespace["_GOOGLE_ADAPTER_SYNTHETIC_TOOL_CONTEXT_PATTERN"],
        _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME=namespace["_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME"],
        _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION=namespace["_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION"],
        _extract_completion_message_text=namespace["_extract_completion_message_text"],
        _get_codex_google_code_assist_tool_contract_policy=namespace[
            "_get_codex_google_code_assist_tool_contract_policy"
        ],
        _get_google_adapter_fallback_context_char_cap=namespace["_get_google_adapter_fallback_context_char_cap"],
        _get_google_adapter_system_prompt_policy=namespace["_get_google_adapter_system_prompt_policy"],
        _google_content_has_text=namespace["_google_content_has_text"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _is_google_adapter_synthetic_tool_context_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return bool(_runtime()._GOOGLE_ADAPTER_SYNTHETIC_TOOL_CONTEXT_PATTERN.fullmatch(text.strip()))


def _is_google_adapter_synthetic_tool_context_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) == 0:
        return False
    return _is_google_adapter_synthetic_tool_context_text(_runtime()._extract_completion_message_text(message))


def _inject_google_adapter_fallback_text_context(
    google_request_dict: dict[str, Any], completion_messages: list[dict[str, Any]]
) -> dict[str, Any]:
    contents = google_request_dict.get("contents")
    if not isinstance(contents, list):
        return {}
    if any(_runtime()._google_content_has_text(content) for content in contents):
        return {}

    text_snippets: list[str] = []
    for message in reversed(completion_messages):
        if _is_google_adapter_synthetic_tool_context_message(message):
            continue
        text = _runtime()._extract_completion_message_text(message).strip()
        if text:
            text_snippets.append(text)
        if len(text_snippets) >= 2:
            break
    if not text_snippets:
        return {}

    text_snippets.reverse()
    fallback_text = "\n\n".join(text_snippets)
    cap = _runtime()._get_google_adapter_fallback_context_char_cap()
    if len(fallback_text) > cap:
        fallback_text = fallback_text[-cap:].lstrip()

    google_request_dict["contents"] = [
        {
            "role": "user",
            "parts": [{"text": fallback_text}],
        },
        *contents,
    ]
    return {
        "inserted_fallback_text_context": True,
        "inserted_fallback_text_context_chars": len(fallback_text),
        "inserted_fallback_text_context_sources": len(text_snippets),
    }


def _extract_google_adapter_system_text_from_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in {None, "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    if not text_parts:
        return None
    return "\n\n".join(text_parts)


def _replace_google_adapter_system_message_text(
    message: dict[str, Any],
    rewritten_text: str,
) -> dict[str, Any]:
    updated_message = dict(message)
    content = updated_message.get("content")
    if isinstance(content, list):
        first_text_index: Optional[int] = None
        updated_content: list[Any] = []
        for index, part in enumerate(content):
            if (
                first_text_index is None
                and isinstance(part, dict)
                and part.get("type") in {None, "text"}
                and isinstance(part.get("text"), str)
            ):
                first_text_index = index
                updated_part = dict(part)
                updated_part["text"] = rewritten_text
                updated_content.append(updated_part)
                continue
            if (
                first_text_index is not None
                and isinstance(part, dict)
                and part.get("type") in {None, "text"}
                and isinstance(part.get("text"), str)
            ):
                continue
            updated_content.append(part)
        if first_text_index is not None:
            updated_message["content"] = updated_content
            return updated_message
    updated_message["content"] = rewritten_text
    return updated_message


def _append_codex_google_code_assist_tool_contract_to_system_text(
    system_text: str,
) -> str:
    stripped_system_text = system_text.strip()
    if _runtime()._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT in stripped_system_text:
        return stripped_system_text
    if not stripped_system_text:
        return _runtime()._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT
    return f"{stripped_system_text}\n\n" f"{_runtime()._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT}"


def _apply_codex_google_code_assist_tool_contract_policy(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_mode = _runtime()._get_codex_google_code_assist_tool_contract_policy()
    metadata = dict(completion_kwargs.get("metadata") or {})
    policy_metadata: dict[str, Any] = {
        "codex_google_code_assist_tool_contract_policy_name": (
            _runtime()._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME
        ),
        "codex_google_code_assist_tool_contract_policy": policy_mode,
        "codex_google_code_assist_tool_contract_policy_version": (
            _runtime()._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION
        ),
        "codex_google_code_assist_tool_contract_policy_applied": (policy_mode != "off"),
    }
    if policy_mode != "off":
        policy_metadata["codex_google_code_assist_tool_contract_prompt_chars"] = len(
            _runtime()._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT
        )

    metadata.update(policy_metadata)
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        tags = []
    metadata["tags"] = list(
        dict.fromkeys(
            [
                *tags,
                "codex-google-code-assist-tool-contract-policy",
                f"codex-google-code-assist-tool-contract-policy:{policy_mode}",
            ]
        )
    )

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["metadata"] = metadata
    if policy_mode == "off":
        return updated_kwargs, policy_metadata

    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return updated_kwargs, policy_metadata

    updated_messages = list(messages)
    system_message_index: Optional[int] = None
    system_text: Optional[str] = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        candidate_text = _extract_google_adapter_system_text_from_content(message.get("content"))
        if isinstance(candidate_text, str):
            system_message_index = index
            system_text = candidate_text
            break

    if system_message_index is None or system_text is None:
        updated_messages.insert(
            0,
            {
                "role": "system",
                "content": _runtime()._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT,
            },
        )
    else:
        updated_messages[system_message_index] = _replace_google_adapter_system_message_text(
            cast(dict[str, Any], updated_messages[system_message_index]),
            _append_codex_google_code_assist_tool_contract_to_system_text(system_text),
        )
    updated_kwargs["messages"] = updated_messages
    return updated_kwargs, policy_metadata


def _is_google_adapter_claude_overhead_block(block: str) -> bool:
    stripped_block = block.strip()
    if not stripped_block:
        return False

    lowered_block = stripped_block.lower()
    if lowered_block.startswith(_runtime()._ANTHROPIC_BILLING_HEADER_PREFIX):
        return True
    if any(marker in lowered_block for marker in _runtime()._GOOGLE_ADAPTER_CLAUDE_OVERHEAD_MARKERS):
        return True
    if "claude code" in lowered_block and any(
        marker in lowered_block
        for marker in (
            "slash command",
            "task management",
            "todowrite",
            "tool use policy",
        )
    ):
        return True
    return False


def _strip_google_adapter_claude_system_overhead(
    system_text: str,
) -> tuple[str, int]:
    preserved_blocks: list[str] = []
    removed_chars = 0
    for block in re.split(r"\n{2,}", system_text):
        stripped_block = block.strip()
        if not stripped_block:
            continue
        if _is_google_adapter_claude_overhead_block(stripped_block):
            removed_chars += len(block)
            continue
        preserved_blocks.append(stripped_block)
    return "\n\n".join(preserved_blocks).strip(), removed_chars


def _build_google_adapter_system_prompt_policy_text(
    *,
    original_text: str,
    policy_mode: str,
) -> tuple[str, dict[str, Any]]:
    normalized_original_text = original_text.strip()
    if policy_mode == "off":
        rewritten_text = original_text
        preserved_text = original_text
        removed_chars = 0
    elif policy_mode == "append":
        preserved_text = normalized_original_text
        removed_chars = 0
        rewritten_text = (
            f"{_runtime()._GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT}\n\n"
            f"{_runtime()._GOOGLE_ADAPTER_ORIGINAL_SYSTEM_PROMPT_HEADING}\n\n"
            f"{preserved_text}"
        ).strip()
    else:
        preserved_text, removed_chars = _strip_google_adapter_claude_system_overhead(normalized_original_text)
        if preserved_text:
            rewritten_text = (
                f"{_runtime()._GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT}\n\n"
                f"{_runtime()._GOOGLE_ADAPTER_PRESERVED_SYSTEM_PROMPT_HEADING}\n\n"
                f"{preserved_text}"
            )
        else:
            rewritten_text = _runtime()._GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT

    metadata = {
        "google_adapter_system_prompt_policy_name": _runtime()._GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME,
        "google_adapter_system_prompt_policy": policy_mode,
        "google_adapter_system_prompt_policy_version": _runtime()._GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION,
        "google_adapter_system_prompt_original_chars": len(original_text),
        "google_adapter_system_prompt_rewritten_chars": len(rewritten_text),
        "google_adapter_system_prompt_removed_claude_overhead_chars": removed_chars,
        "google_adapter_system_prompt_preserved_instruction_chars": len(preserved_text),
        "google_adapter_system_prompt_policy_applied": policy_mode != "off",
    }
    return rewritten_text, metadata


def _apply_google_adapter_system_prompt_policy(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    system_message_index: Optional[int] = None
    system_text: Optional[str] = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        candidate_text = _extract_google_adapter_system_text_from_content(message.get("content"))
        if isinstance(candidate_text, str):
            system_message_index = index
            system_text = candidate_text
            break
    if system_message_index is None or system_text is None:
        return completion_kwargs, {}

    policy_mode = _runtime()._get_google_adapter_system_prompt_policy()
    rewritten_text, policy_metadata = _build_google_adapter_system_prompt_policy_text(
        original_text=system_text,
        policy_mode=policy_mode,
    )

    updated_kwargs = dict(completion_kwargs)
    updated_messages = list(messages)
    if policy_mode != "off":
        updated_messages[system_message_index] = _replace_google_adapter_system_message_text(
            cast(dict[str, Any], updated_messages[system_message_index]),
            rewritten_text,
        )
        updated_kwargs["messages"] = updated_messages

    metadata = updated_kwargs.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata.update(policy_metadata)
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        tags = []
    metadata["tags"] = list(
        dict.fromkeys(
            [
                *tags,
                "google-adapter-system-prompt-policy",
                f"google-adapter-system-prompt-policy:{policy_mode}",
            ]
        )
    )
    updated_kwargs["metadata"] = metadata
    return updated_kwargs, policy_metadata
