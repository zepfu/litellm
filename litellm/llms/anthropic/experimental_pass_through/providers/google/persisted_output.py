"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import re
from collections.abc import Mapping
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class Runtime:
    _CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN: Any
    _CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN: Any
    _CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN: Any
    _estimate_google_content_text_chars: Any
    _get_google_adapter_auxiliary_context_char_cap: Any
    _get_google_adapter_followup_auxiliary_context_char_cap: Any
    _get_google_adapter_followup_persisted_output_char_cap: Any
    _get_google_adapter_persisted_output_char_cap: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN=namespace[
            "_CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN"
        ],
        _CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN=namespace["_CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN"],
        _CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN=namespace["_CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN"],
        _estimate_google_content_text_chars=namespace["_estimate_google_content_text_chars"],
        _get_google_adapter_auxiliary_context_char_cap=namespace["_get_google_adapter_auxiliary_context_char_cap"],
        _get_google_adapter_followup_auxiliary_context_char_cap=namespace[
            "_get_google_adapter_followup_auxiliary_context_char_cap"
        ],
        _get_google_adapter_followup_persisted_output_char_cap=namespace[
            "_get_google_adapter_followup_persisted_output_char_cap"
        ],
        _get_google_adapter_persisted_output_char_cap=namespace["_get_google_adapter_persisted_output_char_cap"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _compact_google_adapter_persisted_output_preview_and_expanded_text(
    text: str,
    *,
    cap: int,
) -> tuple[str, int, set[str], list[dict[str, Any]]]:
    updated_text = text
    compacted_count = 0
    hooks: set[str] = set()
    metadata_items: list[dict[str, Any]] = []

    preview_matches = list(_runtime()._CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN.finditer(text))
    for match in reversed(preview_matches):
        hook = match.group("hook")
        resolved_path = match.group("path")
        compacted_block = (
            "<system-reminder>\n"
            f"{hook} hook additional context: <persisted-output>\n"
            f"[Gemini adapter compacted persisted-output preview. Full output saved to: {resolved_path}]\n"
            "</persisted-output>\n"
            "</system-reminder>\n"
        )
        updated_text = updated_text[: match.start()] + compacted_block + updated_text[match.end() :]
        compacted_count += 1
        hooks.add(hook.lower())
        metadata_items.append(
            {
                "original_chars": len(match.group(0)),
                "kept_chars": len(compacted_block),
                "mode": "preview_block_cap",
            }
        )

    matches = list(_runtime()._CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN.finditer(updated_text))
    for match in reversed(matches):
        content = match.group("content")
        if len(content) <= cap:
            continue
        hook = match.group("hook")
        summary_lines = [line.strip() for line in content.splitlines() if line.strip()][:3]
        summary_text = "\n".join(summary_lines)
        truncated = summary_text[:cap].rstrip()
        compacted_block = (
            "<system-reminder>\n"
            f"{hook} hook additional context: <persisted-output>\n"
            f"{truncated}\n\n"
            f"[Gemini adapter compacted persisted-output from {len(content)} chars to {len(truncated)} chars. Refer to current prompt and tools for full context.]\n"
            "</persisted-output>\n"
            "</system-reminder>\n"
        )
        updated_text = updated_text[: match.start()] + compacted_block + updated_text[match.end() :]
        compacted_count += 1
        hooks.add(hook.lower())
        metadata_items.append(
            {
                "original_chars": len(content),
                "kept_chars": len(truncated),
            }
        )

    return updated_text, compacted_count, hooks, metadata_items


def _compact_expanded_claude_persisted_output_text_for_google_adapter(
    text: str,
    *,
    persisted_output_char_cap: Optional[int] = None,
    auxiliary_context_char_cap: Optional[int] = None,
) -> Tuple[str, int, set[str], list[dict[str, Any]]]:
    if "<system-reminder>" in text and "</system-reminder>" not in text:
        return text, 0, set(), []
    cap = persisted_output_char_cap or _runtime()._get_google_adapter_persisted_output_char_cap()
    updated_text = text
    compacted_count = 0
    hooks: set[str] = set()
    metadata_items: list[dict[str, Any]] = []
    (
        updated_text,
        compacted_count,
        hooks,
        metadata_items,
    ) = _compact_google_adapter_persisted_output_preview_and_expanded_text(
        updated_text,
        cap=cap,
    )

    auxiliary_cap = auxiliary_context_char_cap or _runtime()._get_google_adapter_auxiliary_context_char_cap()
    auxiliary_matches = list(_runtime()._CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN.finditer(updated_text))
    for match in reversed(auxiliary_matches):
        auxiliary_block = match.group(0)
        if len(auxiliary_block) <= auxiliary_cap:
            continue
        hook = match.group("hook")
        body = match.group("body").lstrip("\n")
        summary_lines = [line.strip() for line in body.splitlines() if line.strip()][:4]
        summary_text = "\n".join(summary_lines).strip()
        if not summary_text:
            summary_text = "[Additional context omitted for Gemini adapter compaction.]"
        truncated = summary_text[:auxiliary_cap].rstrip()
        compacted_block = (
            "<system-reminder>\n"
            f"{hook} hook additional context:\n"
            f"{truncated}\n\n"
            f"[Gemini adapter compacted auxiliary context block from {len(auxiliary_block)} chars to {len(truncated)} chars. Refer to the current prompt, tools, and recent messages for full context.]\n"
            "</system-reminder>\n"
        )
        updated_text = updated_text[: match.start()] + compacted_block + updated_text[match.end() :]
        compacted_count += 1
        hooks.add(hook.lower())
        metadata_items.append(
            {
                "original_chars": len(auxiliary_block),
                "kept_chars": len(truncated),
                "mode": "auxiliary_context_block_cap",
            }
        )

    marker = "hook additional context:"
    stripped_updated_text = updated_text.strip()
    if (
        marker in updated_text
        and len(updated_text) > auxiliary_cap
        and stripped_updated_text.startswith("<system-reminder>")
        and stripped_updated_text.endswith("</system-reminder>")
        and stripped_updated_text.count("<system-reminder>") == 1
    ):
        hook_match = re.search(
            r"(SubagentStart|SubAgentStart|SessionStart) hook additional context:",
            updated_text,
        )
        fallback_hook = hook_match.group(1).lower() if hook_match else None
        truncated_text = updated_text[:auxiliary_cap].rstrip()
        updated_text = (
            f"{truncated_text}\n\n"
            f"[Gemini adapter compacted oversized additional context from {len(text)} chars to {len(truncated_text)} chars.]"
        )
        compacted_count += 1
        if fallback_hook:
            hooks.add(fallback_hook)
        metadata_items.append(
            {
                "original_chars": len(text),
                "kept_chars": len(truncated_text),
                "mode": "fallback_text_cap",
            }
        )

    metadata_items.reverse()
    return updated_text, compacted_count, hooks, metadata_items


def _compact_google_adapter_text_part_sequence(
    parts: list[Any],
) -> Tuple[list[Any], int, set[str], list[dict[str, Any]], bool]:
    updated_parts: list[Any] = []
    compacted_count = 0
    hooks: set[str] = set()
    metadata_items: list[dict[str, Any]] = []
    changed = False
    index = 0

    while index < len(parts):
        item = parts[index]
        if not (isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)):
            updated_parts.append(item)
            index += 1
            continue

        group: list[dict[str, Any]] = []
        while index < len(parts):
            candidate = parts[index]
            if not (
                isinstance(candidate, dict)
                and candidate.get("type") == "text"
                and isinstance(candidate.get("text"), str)
            ):
                break
            group.append(candidate)
            index += 1

        combined_text = "".join(str(part.get("text") or "") for part in group)
        (
            compacted_text,
            child_count,
            child_hooks,
            child_metadata,
        ) = _compact_expanded_claude_persisted_output_text_for_google_adapter(combined_text)
        if child_count > 0 or len(group) > 1:
            replacement = dict(group[0])
            replacement["text"] = compacted_text
            updated_parts.append(replacement)
            compacted_count += child_count
            hooks.update(child_hooks)
            metadata_items.extend(child_metadata)
            changed = True
        else:
            updated_parts.extend(group)

    return updated_parts, compacted_count, hooks, metadata_items, changed


def _compact_google_adapter_followup_request_contents(
    request_block: dict[str, Any],
) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list) or len(contents) <= 1:
        return {}

    followup_persisted_cap = _runtime()._get_google_adapter_followup_persisted_output_char_cap()
    followup_auxiliary_cap = _runtime()._get_google_adapter_followup_auxiliary_context_char_cap()
    original_text_chars = sum(_runtime()._estimate_google_content_text_chars(item) for item in contents)
    updated_contents: list[Any] = []
    compacted_count = 0
    hooks: set[str] = set()
    metadata_items: list[dict[str, Any]] = []
    changed = False

    for content in contents:
        if not isinstance(content, dict):
            updated_contents.append(content)
            continue
        parts = content.get("parts")
        if content.get("role") != "user" or not isinstance(parts, list):
            updated_contents.append(content)
            continue

        updated_parts: list[Any] = []
        part_changed = False
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                (
                    compacted_text,
                    child_count,
                    child_hooks,
                    child_metadata,
                ) = _compact_expanded_claude_persisted_output_text_for_google_adapter(
                    part["text"],
                    persisted_output_char_cap=followup_persisted_cap,
                    auxiliary_context_char_cap=followup_auxiliary_cap,
                )
                compacted_count += child_count
                hooks.update(child_hooks)
                metadata_items.extend(child_metadata)
                if compacted_text != part["text"]:
                    updated_part = dict(part)
                    updated_part["text"] = compacted_text
                    updated_parts.append(updated_part)
                    part_changed = True
                    changed = True
                else:
                    updated_parts.append(part)
            else:
                updated_parts.append(part)

        if part_changed:
            updated_content = dict(content)
            updated_content["parts"] = updated_parts
            updated_contents.append(updated_content)
        else:
            updated_contents.append(content)

    if not changed:
        return {}

    request_block["contents"] = updated_contents
    compacted_text_chars = sum(_runtime()._estimate_google_content_text_chars(item) for item in updated_contents)
    changes: dict[str, Any] = {
        "followup_persisted_output_compacted_count": compacted_count,
        "followup_persisted_output_text_chars_before": original_text_chars,
        "followup_persisted_output_text_chars_after": compacted_text_chars,
        "followup_persisted_output_char_cap": followup_persisted_cap,
        "followup_auxiliary_context_char_cap": followup_auxiliary_cap,
    }
    if hooks:
        changes["followup_persisted_output_hooks"] = sorted(hooks)
    if metadata_items:
        changes["followup_persisted_output_compaction"] = metadata_items
    return changes


def _compact_google_adapter_persisted_output_value(
    value: Any,
) -> Tuple[Any, int, set[str], list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            (
                compacted_text,
                compacted_count,
                compacted_hooks,
                compact_metadata,
            ) = _compact_expanded_claude_persisted_output_text_for_google_adapter(value["text"])
            if compacted_count > 0:
                updated_value = dict(value)
                updated_value["text"] = compacted_text
                return (
                    updated_value,
                    compacted_count,
                    compacted_hooks,
                    compact_metadata,
                )
            return value, 0, set(), []

        updated_dict: dict[str, Any] = {}
        compacted_count = 0
        hooks: set[str] = set()
        metadata_items: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            (
                updated_child,
                child_count,
                child_hooks,
                child_metadata,
            ) = _compact_google_adapter_persisted_output_value(child)
            updated_dict[key] = updated_child
            compacted_count += child_count
            hooks.update(child_hooks)
            metadata_items.extend(child_metadata)
            changed = changed or updated_child is not child
        if changed:
            return updated_dict, compacted_count, hooks, metadata_items
        return value, compacted_count, hooks, metadata_items

    if isinstance(value, list):
        updated_list: list[Any] = value
        compacted_count = 0
        list_hooks: set[str] = set()
        list_metadata_items: list[dict[str, Any]] = []
        changed = False

        if any(
            isinstance(child, dict) and child.get("type") == "text" and isinstance(child.get("text"), str)
            for child in value
        ):
            (
                updated_list,
                sequence_count,
                sequence_hooks,
                sequence_metadata,
                sequence_changed,
            ) = _compact_google_adapter_text_part_sequence(value)
            compacted_count += sequence_count
            list_hooks.update(sequence_hooks)
            list_metadata_items.extend(sequence_metadata)
            changed = changed or sequence_changed

        recursively_updated_list = []
        for child in updated_list:
            (
                updated_child,
                child_count,
                child_hooks,
                child_metadata,
            ) = _compact_google_adapter_persisted_output_value(child)
            recursively_updated_list.append(updated_child)
            compacted_count += child_count
            list_hooks.update(child_hooks)
            list_metadata_items.extend(child_metadata)
            changed = changed or updated_child is not child
        if changed:
            return (
                recursively_updated_list,
                compacted_count,
                list_hooks,
                list_metadata_items,
            )
        return value, compacted_count, list_hooks, list_metadata_items

    return value, 0, set(), []


def _compact_google_adapter_persisted_output_in_anthropic_request_body(
    request_body: dict[str, Any],
) -> Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]]:
    (
        updated_body,
        compacted_count,
        hooks,
        metadata_items,
    ) = _compact_google_adapter_persisted_output_value(request_body)
    if not isinstance(updated_body, dict):
        return request_body, 0, set(), []
    return updated_body, compacted_count, hooks, metadata_items
