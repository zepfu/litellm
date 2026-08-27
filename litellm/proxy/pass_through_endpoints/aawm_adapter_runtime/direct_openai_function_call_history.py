"""OPENAI-021 direct OpenAI legacy-history custom-tool identity normalization."""

from __future__ import annotations

from typing import Any

from litellm.responses.litellm_completion_transformation.function_call_identity import (
    is_native_responses_function_call_item_id,
    resolve_responses_custom_tool_call_item_id,
    resolve_responses_function_call_identity,
)


def normalize_direct_openai_legacy_function_call_history_ids(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Normalize typed Responses input function and custom-tool item ids.

    Immediately before direct OpenAI native pass-through, rewrite only
    malformed/non-native ``id`` values on top-level ``input`` items whose
    ``type`` is exactly ``function_call`` or ``custom_tool_call``. Preserve
    ``call_id`` byte-for-byte, leave valid native ``fc_*`` and ``ctc_*`` item
    ids alone, and never rewrite output items, other types, or nested ``id``
    fields outside those typed items.
    """
    if not isinstance(request_body, dict):
        return request_body
    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body

    normalized_input: list[Any] = []
    changed = False
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            normalized_input.append(item)
            continue

        item_type = item.get("type")
        if item_type == "custom_tool_call":
            resolved_item_id = resolve_responses_custom_tool_call_item_id(
                item.get("id"),
                item.get("call_id"),
                fallback=f"input:{index}",
            )
            if not resolved_item_id or item.get("id") == resolved_item_id:
                normalized_input.append(item)
                continue

            clean_item = dict(item)
            clean_item["id"] = resolved_item_id
            # call_id remains the original provider tool id byte-for-byte.
            normalized_input.append(clean_item)
            changed = True
            continue

        if item_type != "function_call":
            normalized_input.append(item)
            continue

        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            # Identity is anchored on call_id; do not invent one.
            normalized_input.append(item)
            continue

        raw_item_id = item.get("id")
        item_id = raw_item_id if isinstance(raw_item_id, str) else None
        if is_native_responses_function_call_item_id(item_id):
            normalized_input.append(item)
            continue

        resolved_item_id, _resolved_call_id = resolve_responses_function_call_identity(
            call_id
        )
        if not resolved_item_id or item_id == resolved_item_id:
            normalized_input.append(item)
            continue

        clean_item = dict(item)
        clean_item["id"] = resolved_item_id
        # call_id remains the original provider tool id byte-for-byte.
        normalized_input.append(clean_item)
        changed = True

    if not changed:
        return request_body
    updated = dict(request_body)
    updated["input"] = normalized_input
    return updated
