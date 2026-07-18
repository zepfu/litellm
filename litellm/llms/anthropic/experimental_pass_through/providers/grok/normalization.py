"""Grok-owned request normalization for OpenAI Responses-compatible payloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Optional

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload


@dataclass(frozen=True)
class Runtime:
    """Route-layer callbacks used by Grok normalization."""

    normalize_tag: Callable[[object], Optional[str]]
    dedupe_sorted: Callable[[list[str]], list[str]]
    merge_metadata: Callable[..., Payload]
    build_span: Callable[..., Payload]
    get_rewrite_input_item_types: Callable[[object], set[str]]


def coerce_function_call_arguments_value(
    arguments_value: object,
) -> tuple[Payload, Optional[str]]:
    if isinstance(arguments_value, dict):
        return dict(arguments_value), None
    if arguments_value is None:
        return {}, "missing"
    if isinstance(arguments_value, str):
        stripped_arguments = arguments_value.strip()
        if not stripped_arguments:
            return {}, "empty"
        try:
            parsed_arguments = json.loads(stripped_arguments)
        except json.JSONDecodeError:
            return {}, "invalid_json"
        if isinstance(parsed_arguments, dict):
            return parsed_arguments, "parsed_json_string"
        return {}, "non_object_json"
    return {}, "unsupported_type"


def sanitize_function_call_arguments_request_body(
    request_body: Payload,
) -> tuple[Payload, list[Payload]]:
    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, []

    sanitized_items: list[object] = []
    changes: list[Payload] = []
    for index, item in enumerate(input_items):
        if not isinstance(item, dict) or item.get("type") != "function_call":
            sanitized_items.append(item)
            continue

        updated_item: Payload = dict(item)
        coerced_arguments, reason = coerce_function_call_arguments_value(
            updated_item.get("arguments")
        )
        if updated_item.get("arguments") != coerced_arguments:
            updated_item["arguments"] = coerced_arguments
            change: Payload = {"type": "function_call", "index": index}
            call_id = updated_item.get("call_id")
            if isinstance(call_id, str) and call_id.strip():
                change["call_id"] = call_id.strip()
            name = updated_item.get("name")
            if isinstance(name, str) and name.strip():
                change["name"] = name.strip()
            if reason:
                change["reason"] = reason
            changes.append(change)
        sanitized_items.append(updated_item)

    if not changes:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["input"] = sanitized_items
    return updated_body, changes


def sanitize_function_call_arguments_in_place(
    runtime: Runtime,
    request_body: Payload,
) -> list[Payload]:
    updated_body, changes = sanitize_function_call_arguments_request_body(request_body)
    if updated_body is not request_body:
        request_body.clear()
        request_body.update(updated_body)
    if not changes:
        return []

    reasons = runtime.dedupe_sorted(
        [
            normalized
            for change in changes
            if (normalized := runtime.normalize_tag(change.get("reason")))
        ]
    )
    metadata_changes = [
        {
            key: value
            for key, value in change.items()
            if key in {"type", "index", "call_id", "name", "reason"}
        }
        for change in changes
    ]
    merged_body = runtime.merge_metadata(
        request_body,
        tags_to_add=[
            "grok-native-function-call-arguments-sanitized",
            *(
                f"grok-native-function-call-arguments-reason:{reason}"
                for reason in reasons
            ),
        ],
        extra_fields={
            "grok_native_function_call_arguments_sanitized": True,
            "grok_native_function_call_arguments_sanitized_count": len(changes),
            "grok_native_function_call_arguments_sanitized_reasons": reasons,
            "grok_native_function_call_arguments_sanitized_items": metadata_changes,
            "langfuse_spans": [
                runtime.build_span(
                    name="grok.native_function_call_arguments_sanitized",
                    metadata={
                        "sanitized_count": len(changes),
                        "reasons": reasons,
                    },
                )
            ],
        },
    )
    request_body.clear()
    request_body.update(merged_body)
    return changes


def stringify_input_item_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def format_function_call_input_message(
    item: Payload,
    *,
    include_correlation_ref: bool = True,
) -> str:
    lines = [
        "[Context note - prior assistant step; not an executable tool invocation]",
    ]
    name = item.get("name")
    if isinstance(name, str) and name.strip():
        lines.append(f"Tool label: {name.strip()}")
    call_id = item.get("call_id")
    if include_correlation_ref and isinstance(call_id, str) and call_id.strip():
        lines.append(f"Correlation ref: {call_id.strip()}")
    arguments = stringify_input_item_value(item.get("arguments")).strip()
    if arguments:
        lines.append(f"Input payload: {arguments}")
    return "\n".join(lines)


def format_function_call_output_input_message(
    item: Payload,
    *,
    include_correlation_ref: bool = True,
) -> str:
    lines = [
        "[Context note - prior tool outcome; not an executable tool invocation]",
    ]
    call_id = item.get("call_id")
    if include_correlation_ref and isinstance(call_id, str) and call_id.strip():
        lines.append(f"Correlation ref: {call_id.strip()}")
    output = stringify_input_item_value(item.get("output")).strip()
    if output:
        lines.append(f"Outcome text: {output}")
    return "\n".join(lines)


def rewrite_input_item_for_model_input(
    item: Payload,
    *,
    item_type: str,
    include_correlation_ref: bool = True,
) -> Optional[Payload]:
    if item_type == "function_call":
        return {
            "type": "message",
            "role": "assistant",
            "content": format_function_call_input_message(
                item,
                include_correlation_ref=include_correlation_ref,
            ),
        }
    if item_type == "function_call_output":
        return {
            "type": "message",
            "role": "user",
            "content": format_function_call_output_input_message(
                item,
                include_correlation_ref=include_correlation_ref,
            ),
        }
    return None


def is_anthropic_responses_adapter_body(request_body: Payload) -> bool:
    metadata = request_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get("route_family") == "anthropic_grok_native_responses_adapter"
        or metadata.get("passthrough_route_family")
        == "anthropic_grok_native_responses_adapter"
    )


def add_input_item_rewrite_logging_metadata(
    runtime: Runtime,
    request_body: Payload,
    *,
    rewritten_items: list[Payload],
) -> Payload:
    rewritten_item_types = runtime.dedupe_sorted(
        [
            item_type
            for item in rewritten_items
            if isinstance((item_type := item.get("type")), str) and item_type
        ]
    )
    return runtime.merge_metadata(
        request_body,
        tags_to_add=[
            "grok-native-input-item-rewritten",
            *(
                f"grok-native-input-item:{item_type}"
                for item_type in rewritten_item_types
            ),
        ],
        extra_fields={
            "grok_native_input_item_rewrite_count": len(rewritten_items),
            "grok_native_input_item_rewrite_types": rewritten_item_types,
            "grok_native_input_item_rewrites": rewritten_items,
            "langfuse_spans": [
                runtime.build_span(
                    name="grok.native_input_item_rewritten",
                    metadata={
                        "rewritten_count": len(rewritten_items),
                        "rewritten_item_types": rewritten_item_types,
                    },
                )
            ],
        },
    )


def rewrite_unsupported_input_items_from_request_body(
    runtime: Runtime,
    request_body: Payload,
) -> tuple[Payload, list[Payload]]:
    rewrite_input_item_types = runtime.get_rewrite_input_item_types(
        request_body.get("model")
    )
    if not rewrite_input_item_types:
        return request_body, []
    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, []

    updated_input_items: list[object] = []
    rewritten_items: list[Payload] = []
    include_correlation_ref = not is_anthropic_responses_adapter_body(request_body)
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            updated_input_items.append(item)
            continue
        item_type = runtime.normalize_tag(item.get("type"))
        if item_type not in rewrite_input_item_types:
            updated_input_items.append(item)
            continue
        rewritten_item = rewrite_input_item_for_model_input(
            item,
            item_type=item_type,
            include_correlation_ref=include_correlation_ref,
        )
        if rewritten_item is None:
            updated_input_items.append(item)
            continue

        metadata_item: Payload = {
            "type": item_type,
            "index": index,
            "rewritten_type": rewritten_item.get("type"),
            "rewritten_role": rewritten_item.get("role"),
        }
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id.strip():
            normalized_call_id = call_id.strip()
            if include_correlation_ref:
                metadata_item["call_id"] = normalized_call_id
            else:
                metadata_item["call_id_hash"] = hashlib.sha256(
                    normalized_call_id.encode("utf-8", errors="replace")
                ).hexdigest()[:12]
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            metadata_item["name"] = name.strip()
        if item_type == "function_call_output":
            metadata_item["output_chars"] = len(
                stringify_input_item_value(item.get("output"))
            )
        rewritten_items.append(metadata_item)
        updated_input_items.append(rewritten_item)

    if not rewritten_items:
        return request_body, []
    updated_body = dict(request_body)
    updated_body["input"] = updated_input_items
    return (
        add_input_item_rewrite_logging_metadata(
            runtime,
            updated_body,
            rewritten_items=rewritten_items,
        ),
        rewritten_items,
    )


def rewrite_unsupported_input_items_in_place(
    runtime: Runtime,
    request_body: Payload,
) -> list[Payload]:
    updated_body, rewritten_items = rewrite_unsupported_input_items_from_request_body(
        runtime,
        request_body,
    )
    if updated_body is not request_body:
        request_body.clear()
        request_body.update(updated_body)
    return rewritten_items
