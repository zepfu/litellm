"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import hashlib
from collections.abc import Mapping
from typing import Any, Optional


@dataclass(frozen=True)
class Runtime:
    _ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES: Any
    _CLAUDE_AGENT_TENANT_PATTERN: Any
    _CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES: Any
    _aawm_provider_shaping: Any
    _extract_completion_message_text: Any
    _find_prior_google_function_call_content_index: Any
    _get_google_adapter_followup_allowed_tool_names: Any
    _google_adapter_function_call_anchor_content: Any
    _google_content_function_call_ids: Any
    _google_content_function_response_ids: Any
    _google_content_has_function_call: Any
    _normalize_google_completion_adapter_model_name: Any
    _request_block_has_google_function_response: Any
    _split_anthropic_adapter_provider_prefix: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES=namespace[
            "_ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES"
        ],
        _CLAUDE_AGENT_TENANT_PATTERN=namespace["_CLAUDE_AGENT_TENANT_PATTERN"],
        _CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES=namespace[
            "_CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES"
        ],
        _aawm_provider_shaping=namespace["_aawm_provider_shaping"],
        _extract_completion_message_text=namespace["_extract_completion_message_text"],
        _find_prior_google_function_call_content_index=namespace["_find_prior_google_function_call_content_index"],
        _get_google_adapter_followup_allowed_tool_names=namespace["_get_google_adapter_followup_allowed_tool_names"],
        _google_adapter_function_call_anchor_content=namespace["_google_adapter_function_call_anchor_content"],
        _google_content_function_call_ids=namespace["_google_content_function_call_ids"],
        _google_content_function_response_ids=namespace["_google_content_function_response_ids"],
        _google_content_has_function_call=namespace["_google_content_has_function_call"],
        _normalize_google_completion_adapter_model_name=namespace["_normalize_google_completion_adapter_model_name"],
        _request_block_has_google_function_response=namespace["_request_block_has_google_function_response"],
        _split_anthropic_adapter_provider_prefix=namespace["_split_anthropic_adapter_provider_prefix"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _normalize_anthropic_google_completion_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _runtime()._split_anthropic_adapter_provider_prefix(model)
    if explicit_provider not in (None, "google") or candidate is None:
        return None
    normalized_candidate = _runtime()._normalize_google_completion_adapter_model_name(candidate)
    if normalized_candidate.startswith(_runtime()._ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES):
        return normalized_candidate
    return None


def _normalize_codex_google_code_assist_adapter_model_name(
    model: Any,
) -> Optional[str]:
    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    if lowered.startswith("openrouter/"):
        return None
    for prefix in ("google-code-assist/", "code-assist/"):
        if lowered.startswith(prefix):
            candidate = candidate.split("/", 1)[1]
            lowered = candidate.lower()
            break
    if lowered.startswith("codex-gemini-"):
        candidate = candidate[len("codex-") :]

    explicit_provider, split_candidate = _runtime()._split_anthropic_adapter_provider_prefix(candidate)
    if explicit_provider not in (None, "google") or split_candidate is None:
        return None
    normalized_candidate = _runtime()._normalize_google_completion_adapter_model_name(split_candidate)
    if normalized_candidate.startswith(_runtime()._CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES):
        return normalized_candidate
    return None


def _extract_google_adapter_agent_name_from_completion_messages(
    completion_messages: list[dict[str, Any]],
) -> Optional[str]:
    for message in completion_messages:
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        match = _runtime()._CLAUDE_AGENT_TENANT_PATTERN.search(content)
        if match:
            agent_name = match.group("agent").strip()
            if agent_name:
                return agent_name
    return None


def _extract_google_adapter_latest_user_prompt_text(
    completion_messages: list[dict[str, Any]],
) -> Optional[str]:
    for message in reversed(completion_messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _runtime()._extract_completion_message_text(message).strip()
        if not text:
            continue
        reminder_matches = _runtime()._aawm_provider_shaping.iter_delimited_spans(
            text,
            "<system-reminder>",
            "</system-reminder>",
        )
        if reminder_matches:
            trailing_text = text[reminder_matches[-1].end :].strip()
            if trailing_text:
                return trailing_text
            continue
        return text
    return None


def _extract_google_adapter_latest_tool_result_fingerprint(
    completion_messages: list[dict[str, Any]],
) -> Optional[str]:
    for message in reversed(completion_messages):
        if not isinstance(message, dict) or message.get("role") not in {
            "tool",
            "function",
        }:
            continue
        tool_call_id = message.get("tool_call_id") or message.get("name") or ""
        text = _runtime()._extract_completion_message_text(message).strip()
        if not tool_call_id and not text:
            continue
        result_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        return f"{tool_call_id}:{result_hash}"
    return None


def _estimate_google_content_text_chars(content_block: Any) -> int:
    if not isinstance(content_block, dict):
        return 0
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return 0
    total = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            total += len(text)
    return total


def _selected_google_contents_have_paired_function_responses(
    contents: list[Any],
    selected_indices: list[int],
) -> bool:
    seen_function_call_ids: set[str] = set()
    for index in selected_indices:
        content = contents[index]
        response_ids = _runtime()._google_content_function_response_ids(content)
        if response_ids and not response_ids.issubset(seen_function_call_ids):
            return False
        seen_function_call_ids.update(_runtime()._google_content_function_call_ids(content))
    return True


def _selected_google_contents_have_complete_function_exchanges(
    contents: list[Any],
    selected_indices: list[int],
) -> bool:
    seen_function_call_ids: set[str] = set()
    pending_function_call_ids: set[str] = set()
    for index in selected_indices:
        content = contents[index]
        response_ids = _runtime()._google_content_function_response_ids(content)
        if response_ids and not response_ids.issubset(seen_function_call_ids):
            return False
        pending_function_call_ids.difference_update(response_ids)

        function_call_ids = _runtime()._google_content_function_call_ids(content)
        seen_function_call_ids.update(function_call_ids)
        pending_function_call_ids.update(function_call_ids)
    return not pending_function_call_ids


def _add_required_google_function_call_pair_indices(
    contents: list[Any],
    selected_indices: list[int],
) -> list[int]:
    selected_index_set = set(selected_indices)
    for index in list(selected_indices):
        for function_response_id in _runtime()._google_content_function_response_ids(contents[index]):
            if any(
                function_response_id in _runtime()._google_content_function_call_ids(contents[prior_index])
                for prior_index in selected_indices
                if prior_index < index
            ):
                continue
            paired_index = _runtime()._find_prior_google_function_call_content_index(
                contents,
                before_index=index,
                function_response_id=function_response_id,
            )
            if paired_index is not None:
                selected_index_set.add(paired_index)
    return sorted(selected_index_set)


def _trim_google_content_indices_to_window(
    contents: list[Any],
    selected_indices: list[int],
    *,
    protected_text_indices: set[int],
    max_window: int,
) -> list[int]:
    selected_indices = list(selected_indices)
    while len(selected_indices) > max_window:
        removed = False
        for position, index in enumerate(selected_indices):
            if index in protected_text_indices:
                continue
            trial_indices = selected_indices[:position] + selected_indices[position + 1 :]
            if _selected_google_contents_have_complete_function_exchanges(
                contents,
                trial_indices,
            ):
                selected_indices = trial_indices
                removed = True
                break
        if removed:
            continue

        # RR-054 #53: if only protected indices remain, stop rather than pop a protected turn.
        removable_position = next(
            (position for position, index in enumerate(selected_indices) if index not in protected_text_indices),
            None,
        )
        if removable_position is None:
            break
        selected_indices.pop(removable_position)

    while not _selected_google_contents_have_complete_function_exchanges(
        contents,
        selected_indices,
    ):
        for position, index in enumerate(selected_indices):
            response_ids = _runtime()._google_content_function_response_ids(contents[index])
            prior_call_ids: set[str] = set()
            for prior_index in selected_indices[:position]:
                prior_call_ids.update(_runtime()._google_content_function_call_ids(contents[prior_index]))
            if not response_ids.issubset(prior_call_ids):
                selected_indices.pop(position)
                break

            function_call_ids = _runtime()._google_content_function_call_ids(contents[index])
            later_response_ids: set[str] = set()
            for later_index in selected_indices[position + 1 :]:
                later_response_ids.update(_runtime()._google_content_function_response_ids(contents[later_index]))
            if function_call_ids and not function_call_ids.issubset(later_response_ids):
                selected_indices.pop(position)
                break
        else:
            break
    return selected_indices


def _trim_google_adapter_followup_tools(request_block: dict[str, Any]) -> dict[str, Any]:
    if not _runtime()._request_block_has_google_function_response(request_block):
        return {}

    allowed_tool_names = _runtime()._get_google_adapter_followup_allowed_tool_names()
    tools = request_block.get("tools")
    if not isinstance(tools, list) or not allowed_tool_names:
        return {}

    original_decl_count = 0
    trimmed_decl_count = 0
    any_trimmed = False
    updated_tools: list[Any] = []

    for tool_entry in tools:
        if not isinstance(tool_entry, dict):
            updated_tools.append(tool_entry)
            continue
        key = None
        decls = tool_entry.get("functionDeclarations")
        if isinstance(decls, list):
            key = "functionDeclarations"
        else:
            decls = tool_entry.get("function_declarations")
            if isinstance(decls, list):
                key = "function_declarations"
        if key is None or not isinstance(decls, list):
            updated_tools.append(tool_entry)
            continue
        original_decl_count += len(decls)
        filtered_decls = []
        for decl in decls:
            if not isinstance(decl, dict):
                continue
            name = decl.get("name")
            if isinstance(name, str) and name in allowed_tool_names:
                filtered_decls.append(decl)
        trimmed_decl_count += len(filtered_decls)
        if len(filtered_decls) != len(decls):
            any_trimmed = True
        if filtered_decls:
            copied_entry = dict(tool_entry)
            copied_entry[key] = filtered_decls
            updated_tools.append(copied_entry)

    if not any_trimmed:
        return {}

    request_block["tools"] = updated_tools
    return {
        "trimmed_followup_function_declarations_from": original_decl_count,
        "trimmed_followup_function_declarations_to": trimmed_decl_count,
    }


def _is_google_function_call_allowed_predecessor(content_block: Any) -> bool:
    if not isinstance(content_block, dict):
        return False
    if content_block.get("role") == "user":
        return True
    return bool(_runtime()._google_content_function_response_ids(content_block))


def _merge_google_model_content_parts(
    first_content: dict[str, Any],
    second_content: dict[str, Any],
) -> dict[str, Any]:
    first_parts = first_content.get("parts")
    second_parts = second_content.get("parts")
    merged = dict(first_content)
    merged["parts"] = [
        *(first_parts if isinstance(first_parts, list) else []),
        *(second_parts if isinstance(second_parts, list) else []),
    ]
    return merged


def _repair_google_adapter_function_call_turn_adjacency(
    request_block: dict[str, Any],
) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return {}

    updated_contents: list[Any] = []
    merged_model_turn_count = 0
    inserted_anchor_count = 0
    changed = False

    for content in contents:
        updated_contents.append(content)
        if (
            not isinstance(content, dict)
            or content.get("role") != "model"
            or not _runtime()._google_content_has_function_call(content)
        ):
            continue

        while (
            len(updated_contents) >= 2
            and isinstance(updated_contents[-1], dict)
            and isinstance(updated_contents[-2], dict)
            and updated_contents[-1].get("role") == "model"
            and updated_contents[-2].get("role") == "model"
        ):
            updated_contents[-2] = _merge_google_model_content_parts(
                updated_contents[-2],
                updated_contents[-1],
            )
            updated_contents.pop()
            merged_model_turn_count += 1
            changed = True

        current_index = len(updated_contents) - 1
        predecessor = updated_contents[current_index - 1] if current_index > 0 else None
        if not _is_google_function_call_allowed_predecessor(predecessor):
            updated_contents.insert(
                current_index,
                _runtime()._google_adapter_function_call_anchor_content(),
            )
            inserted_anchor_count += 1
            changed = True

    if not changed:
        return {}

    request_block["contents"] = updated_contents
    changes: dict[str, Any] = {}
    if merged_model_turn_count:
        changes["repaired_function_call_adjacency_merged_model_turn_count"] = merged_model_turn_count
    if inserted_anchor_count:
        changes["repaired_function_call_adjacency_inserted_user_anchor_count"] = inserted_anchor_count
    return changes
