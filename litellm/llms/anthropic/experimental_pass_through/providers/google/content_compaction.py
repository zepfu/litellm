"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Mapping
from typing import Any, Optional


@dataclass(frozen=True)
class Runtime:
    _aawm_provider_shaping: Any
    _add_required_google_function_call_pair_indices: Any
    _compact_google_adapter_followup_request_contents: Any
    _estimate_google_content_text_chars: Any
    _get_google_adapter_default_thinking_level: Any
    _get_google_adapter_followup_subagent_context_text_part_char_cap: Any
    _get_google_adapter_max_contents_text_chars: Any
    _get_google_adapter_max_contents_window: Any
    _get_google_adapter_max_output_tokens_cap: Any
    _get_google_adapter_oversized_text_part_char_cap: Any
    _get_google_adapter_pure_context_text_part_char_cap: Any
    _get_google_adapter_subagent_context_text_part_char_cap: Any
    _google_content_has_function_exchange: Any
    _google_content_has_text: Any
    _repair_google_adapter_function_call_turn_adjacency: Any
    _trim_google_adapter_followup_tools: Any
    _trim_google_content_indices_to_window: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _aawm_provider_shaping=namespace["_aawm_provider_shaping"],
        _add_required_google_function_call_pair_indices=namespace["_add_required_google_function_call_pair_indices"],
        _compact_google_adapter_followup_request_contents=namespace[
            "_compact_google_adapter_followup_request_contents"
        ],
        _estimate_google_content_text_chars=namespace["_estimate_google_content_text_chars"],
        _get_google_adapter_default_thinking_level=namespace["_get_google_adapter_default_thinking_level"],
        _get_google_adapter_followup_subagent_context_text_part_char_cap=namespace[
            "_get_google_adapter_followup_subagent_context_text_part_char_cap"
        ],
        _get_google_adapter_max_contents_text_chars=namespace["_get_google_adapter_max_contents_text_chars"],
        _get_google_adapter_max_contents_window=namespace["_get_google_adapter_max_contents_window"],
        _get_google_adapter_max_output_tokens_cap=namespace["_get_google_adapter_max_output_tokens_cap"],
        _get_google_adapter_oversized_text_part_char_cap=namespace["_get_google_adapter_oversized_text_part_char_cap"],
        _get_google_adapter_pure_context_text_part_char_cap=namespace[
            "_get_google_adapter_pure_context_text_part_char_cap"
        ],
        _get_google_adapter_subagent_context_text_part_char_cap=namespace[
            "_get_google_adapter_subagent_context_text_part_char_cap"
        ],
        _google_content_has_function_exchange=namespace["_google_content_has_function_exchange"],
        _google_content_has_text=namespace["_google_content_has_text"],
        _repair_google_adapter_function_call_turn_adjacency=namespace[
            "_repair_google_adapter_function_call_turn_adjacency"
        ],
        _trim_google_adapter_followup_tools=namespace["_trim_google_adapter_followup_tools"],
        _trim_google_content_indices_to_window=namespace["_trim_google_content_indices_to_window"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _split_google_adapter_inline_context_and_prompt(request_block: dict[str, Any]) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return {}

    updated_contents: list[Any] = []
    split_count = 0
    split_prompt_chars = 0

    for content in contents:
        if not isinstance(content, dict):
            updated_contents.append(content)
            continue
        if content.get("role") != "user":
            updated_contents.append(content)
            continue
        parts = content.get("parts")
        if not isinstance(parts, list) or len(parts) != 1:
            updated_contents.append(content)
            continue
        part = parts[0]
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            updated_contents.append(content)
            continue
        text_value = part["text"]
        stripped_text = text_value.lstrip()
        if not stripped_text.startswith("<system-reminder>"):
            updated_contents.append(content)
            continue

        reminder_matches = _runtime()._aawm_provider_shaping.iter_delimited_spans(
            text_value,
            "<system-reminder>",
            "</system-reminder>",
        )
        if not reminder_matches:
            updated_contents.append(content)
            continue
        trailing_text = text_value[reminder_matches[-1].end :].strip()
        if not trailing_text:
            updated_contents.append(content)
            continue

        split_count += 1
        split_prompt_chars += len(trailing_text)
        context_text = text_value[: reminder_matches[-1].end].rstrip() + "\n"
        updated_contents.append({"role": "user", "parts": [{"text": context_text}]})
        updated_contents.append({"role": "user", "parts": [{"text": trailing_text}]})

    if split_count == 0:
        return {}

    request_block["contents"] = updated_contents
    return {
        "split_inline_context_prompt_count": split_count,
        "split_inline_context_prompt_chars": split_prompt_chars,
    }


def _compact_google_adapter_oversized_text_part(
    part: Any,
    *,
    cap: int,
    pure_context_cap: int,
    head_keep: int,
    tail_keep: int,
    is_followup_request: bool,
) -> tuple[Any, bool, dict[str, int]]:
    stats = {
        "original_text_chars": 0,
        "compacted_text_chars": 0,
        "compacted_count": 0,
        "pure_context_compacted_count": 0,
        "subagent_context_compacted_count": 0,
    }
    if not isinstance(part, dict) or not isinstance(part.get("text"), str):
        return part, False, stats

    text_value = part["text"]
    stats["original_text_chars"] = len(text_value)
    stripped_text = text_value.strip()
    reminder_matches = _runtime()._aawm_provider_shaping.iter_delimited_spans(
        text_value,
        "<system-reminder>",
        "</system-reminder>",
    )
    trailing_text = text_value[reminder_matches[-1].end :].strip() if reminder_matches else None
    is_reminder_only_context = (
        bool(reminder_matches) and stripped_text.startswith("<system-reminder>") and not trailing_text
    )
    is_subagent_context = (
        "SubagentStart hook additional context:" in text_value or "SubAgentStart hook additional context:" in text_value
    )
    reminder_only_context_cap = pure_context_cap if is_followup_request else cap
    if is_subagent_context:
        reminder_only_context_cap = (
            _runtime()._get_google_adapter_followup_subagent_context_text_part_char_cap()
            if is_followup_request
            else _runtime()._get_google_adapter_subagent_context_text_part_char_cap()
        )

    if is_reminder_only_context and len(text_value) > reminder_only_context_cap:
        updated_part = dict(part)
        updated_part["text"] = text_value[:reminder_only_context_cap].rstrip()
        stats["compacted_text_chars"] = len(updated_part["text"])
        stats["compacted_count"] = 1
        stats["pure_context_compacted_count"] = 1
        stats["subagent_context_compacted_count"] = int(is_subagent_context)
        return updated_part, True, stats

    if len(text_value) <= cap:
        stats["compacted_text_chars"] = len(text_value)
        return part, False, stats

    prefix = text_value[:head_keep].rstrip()
    suffix = text_value[-tail_keep:].lstrip()
    compacted_text = (
        f"{prefix}\n\n"
        f"[Gemini adapter compacted oversized user text from {len(text_value)} chars to preserve head/tail context.]\n\n"
        f"{suffix}"
    )
    updated_part = dict(part)
    updated_part["text"] = compacted_text
    stats["compacted_text_chars"] = len(compacted_text)
    stats["compacted_count"] = 1
    return updated_part, True, stats


def _compact_google_adapter_oversized_text_parts(request_block: dict[str, Any]) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return {}

    cap = _runtime()._get_google_adapter_oversized_text_part_char_cap()
    pure_context_cap = _runtime()._get_google_adapter_pure_context_text_part_char_cap()
    head_keep = max(512, cap // 3)
    tail_keep = max(1024, cap - head_keep - 64)
    updated_contents: list[Any] = []
    compacted_count = 0
    original_text_chars = 0
    compacted_text_chars = 0
    pure_context_compacted_count = 0
    subagent_context_compacted_count = 0
    is_followup_request = len(contents) > 2

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
            updated_part, changed, stats = _compact_google_adapter_oversized_text_part(
                part,
                cap=cap,
                pure_context_cap=pure_context_cap,
                head_keep=head_keep,
                tail_keep=tail_keep,
                is_followup_request=is_followup_request,
            )
            original_text_chars += stats["original_text_chars"]
            compacted_text_chars += stats["compacted_text_chars"]
            compacted_count += stats["compacted_count"]
            pure_context_compacted_count += stats["pure_context_compacted_count"]
            subagent_context_compacted_count += stats["subagent_context_compacted_count"]
            part_changed = part_changed or changed
            updated_parts.append(updated_part)

        if part_changed:
            if updated_parts:
                updated_content = dict(content)
                updated_content["parts"] = updated_parts
                updated_contents.append(updated_content)
            else:
                continue
        else:
            updated_contents.append(content)

    if compacted_count == 0:
        return {}

    request_block["contents"] = updated_contents
    changes = {
        "compacted_oversized_text_parts_count": compacted_count,
        "compacted_oversized_text_parts_cap": cap,
        "compacted_oversized_text_parts_before_chars": original_text_chars,
        "compacted_oversized_text_parts_after_chars": compacted_text_chars,
    }
    if pure_context_compacted_count > 0:
        changes["retained_followup_reminder_only_context_count"] = pure_context_compacted_count
        changes["compacted_pure_context_text_parts_count"] = pure_context_compacted_count
        changes["compacted_pure_context_text_parts_cap"] = (
            _runtime()._get_google_adapter_followup_subagent_context_text_part_char_cap()
            if is_followup_request and subagent_context_compacted_count == pure_context_compacted_count
            else pure_context_cap
        )
    if subagent_context_compacted_count > 0:
        changes["subagent_context_text_parts_compacted_count"] = subagent_context_compacted_count
        changes["subagent_context_text_parts_cap"] = (
            _runtime()._get_google_adapter_followup_subagent_context_text_part_char_cap()
            if is_followup_request
            else _runtime()._get_google_adapter_subagent_context_text_part_char_cap()
        )
    return changes


def _apply_google_adapter_contents_window_policy(request_block: dict[str, Any]) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list) or len(contents) <= 2:
        return {}
    session_id = request_block.get("session_id")
    if not isinstance(session_id, str) or len(session_id) == 0:
        return {}

    original_count = len(contents)
    original_text_chars = sum(_runtime()._estimate_google_content_text_chars(item) for item in contents)
    max_window = _runtime()._get_google_adapter_max_contents_window()
    max_text_chars = _runtime()._get_google_adapter_max_contents_text_chars()

    selected_indices = list(range(max(0, original_count - max_window), original_count))
    text_indices = [idx for idx, item in enumerate(contents) if _runtime()._google_content_has_text(item)]
    protected_text_indices = text_indices[-2:]
    protected_text_index_set = set(protected_text_indices)
    selected_indices = _runtime()._add_required_google_function_call_pair_indices(
        contents,
        sorted(set(protected_text_indices + selected_indices)),
    )
    protected_indices = sorted(
        index
        for index in selected_indices
        if _runtime()._google_content_has_function_exchange(contents[index]) or index in protected_text_index_set
    )
    if protected_indices:
        selected_indices = _runtime()._trim_google_content_indices_to_window(
            contents,
            selected_indices,
            protected_text_indices=protected_text_index_set,
            max_window=max_window,
        )

    trimmed_contents = [contents[idx] for idx in selected_indices]
    protected_positions = {pos for pos, idx in enumerate(selected_indices) if idx in set(protected_indices)}
    trimmed_text_chars = sum(_runtime()._estimate_google_content_text_chars(item) for item in trimmed_contents)
    while len(trimmed_contents) > 2 and trimmed_text_chars > max_text_chars:
        removable_pos = next(
            (pos for pos in range(len(trimmed_contents)) if pos not in protected_positions),
            None,
        )
        if removable_pos is None:
            break
        removed = trimmed_contents.pop(removable_pos)
        trimmed_text_chars -= _runtime()._estimate_google_content_text_chars(removed)
        protected_positions = {
            pos - 1 if pos > removable_pos else pos for pos in protected_positions if pos != removable_pos
        }

    if len(trimmed_contents) == original_count and trimmed_text_chars == original_text_chars:
        return {}

    request_block["contents"] = trimmed_contents
    return {
        "trimmed_contents_from_count": original_count,
        "trimmed_contents_to_count": len(trimmed_contents),
        "trimmed_contents_from_text_chars": original_text_chars,
        "trimmed_contents_to_text_chars": trimmed_text_chars,
        "trimmed_contents_max_window": max_window,
        "trimmed_contents_max_text_chars": max_text_chars,
        "trimmed_contents_preserved_text_entries": len(protected_text_indices),
        "trimmed_contents_preserved_function_exchange_entries": len(
            [idx for idx in selected_indices if _runtime()._google_content_has_function_exchange(contents[idx])]
        ),
    }


def _apply_google_adapter_generation_config_policy(
    request_block: dict[str, Any],
    *,
    model: Optional[str],
) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    generation_config = request_block.get("generationConfig")
    if not isinstance(generation_config, dict):
        generation_config = {}
        request_block["generationConfig"] = generation_config

    if not isinstance(generation_config.get("thinkingConfig"), dict):
        default_thinking_level = _runtime()._get_google_adapter_default_thinking_level(model)
        if default_thinking_level:
            generation_config["thinkingConfig"] = {
                "includeThoughts": False,
                "thinkingLevel": default_thinking_level,
            }
            changes["injected_default_thinking_config"] = True
            changes["injected_default_thinking_level"] = default_thinking_level

    max_output_tokens = generation_config.get("max_output_tokens")
    cap = _runtime()._get_google_adapter_max_output_tokens_cap()
    thinking_config = generation_config.get("thinkingConfig")
    thinking_budget = thinking_config.get("thinkingBudget") if isinstance(thinking_config, dict) else None
    should_preserve_max_output_for_thinking = (
        isinstance(max_output_tokens, int)
        and not isinstance(max_output_tokens, bool)
        and isinstance(thinking_budget, int)
        and not isinstance(thinking_budget, bool)
        and thinking_budget > 0
        and max_output_tokens > thinking_budget
    )
    if (
        isinstance(max_output_tokens, int)
        and cap is not None
        and max_output_tokens > cap
        and should_preserve_max_output_for_thinking
    ):
        changes["preserved_oversized_max_output_tokens_for_thinking_budget"] = max_output_tokens
        changes["preserved_oversized_thinking_budget"] = thinking_budget
        changes["preserved_oversized_max_output_tokens_cap"] = cap
    elif isinstance(max_output_tokens, int) and cap is not None and max_output_tokens > cap:
        generation_config.pop("max_output_tokens", None)
        changes["removed_oversized_max_output_tokens_from"] = max_output_tokens
        changes["removed_oversized_max_output_tokens_cap"] = cap

    temperature = generation_config.get("temperature")
    if isinstance(temperature, (int, float)) and float(temperature) == 1.0:
        generation_config.pop("temperature", None)
        changes["removed_default_temperature"] = True

    if not generation_config:
        request_block.pop("generationConfig", None)
        changes["removed_empty_generation_config"] = True

    return changes


def _apply_google_adapter_request_shape_policy(payload: dict[str, Any]) -> dict[str, Any]:
    request_block = payload.get("request") if isinstance(payload.get("request"), dict) else None
    if not isinstance(request_block, dict):
        return {}

    changes: dict[str, Any] = {}
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    split_changes = _split_google_adapter_inline_context_and_prompt(request_block)
    if split_changes:
        changes.update(split_changes)
    followup_content_changes = _runtime()._compact_google_adapter_followup_request_contents(request_block)
    if followup_content_changes:
        changes.update(followup_content_changes)
    followup_tool_changes = _runtime()._trim_google_adapter_followup_tools(request_block)
    if followup_tool_changes:
        changes.update(followup_tool_changes)
    oversized_text_changes = _compact_google_adapter_oversized_text_parts(request_block)
    if oversized_text_changes:
        changes.update(oversized_text_changes)
    content_window_changes = _apply_google_adapter_contents_window_policy(request_block)
    if content_window_changes:
        changes.update(content_window_changes)
    function_call_adjacency_changes = _runtime()._repair_google_adapter_function_call_turn_adjacency(request_block)
    if function_call_adjacency_changes:
        changes.update(function_call_adjacency_changes)
    generation_config_changes = _apply_google_adapter_generation_config_policy(
        request_block,
        model=model,
    )
    if generation_config_changes:
        changes.update(generation_config_changes)

    return changes
