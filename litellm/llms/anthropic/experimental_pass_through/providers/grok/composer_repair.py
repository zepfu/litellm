"""Grok Composer literal-tool parsing and response repair."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional
from uuid import uuid4

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload


AdvertisedTools = dict[str, Payload]


@dataclass(frozen=True)
class Runtime:
    """Generic route-layer services used by Grok Composer repair."""

    decode_json_prefix: Callable[..., tuple[str, int]]
    strip_text_spans: Callable[[str, list[tuple[int, int]]], str]
    build_advertised_function_tools_index: Callable[
        [Optional[Payload]], AdvertisedTools
    ]
    validate_tool_arguments: Callable[..., Optional[str]]
    is_malformed_composer_literal_text: Callable[[str], bool]
    is_malformed_tool_call_text_output: Callable[[Payload], bool]


_LITERAL_TOOL_LABEL_LINE_RE = re.compile(
    r"(?im)^Tool label:\s*(?P<name>[^\n]+)\s*$"
)
_LITERAL_CORRELATION_REF_LINE_RE = re.compile(
    r"(?im)^Correlation ref:\s*(?P<call_id>[^\n]+)\s*$"
)
_LITERAL_INPUT_PAYLOAD_LINE_RE = re.compile(
    r"(?im)^Input payload:\s*(?P<payload>.+?)\s*$"
)
_LITERAL_CONTEXT_NOTE_LINE_RE = re.compile(
    r"(?im)^\s*\[Context note - prior assistant step; not an executable tool invocation\]\s*$"
)
_LITERAL_TOOL_ARGUMENT_METADATA_KEYS = frozenset({"description"})


def literal_tool_block_strip_start(text: str, label_start: int) -> int:
    if label_start <= 0:
        return label_start

    prefix = text[:label_start]
    context_note_matches = list(_LITERAL_CONTEXT_NOTE_LINE_RE.finditer(prefix))
    if not context_note_matches:
        return label_start

    context_note_match = context_note_matches[-1]
    between = prefix[context_note_match.end() : label_start]
    if between.strip():
        return label_start
    return context_note_match.start()


def parse_literal_tool_label_blocks(runtime: Runtime, text: str) -> list[Payload]:
    if not text.strip():
        return []

    blocks: list[Payload] = []
    label_matches = list(_LITERAL_TOOL_LABEL_LINE_RE.finditer(text))
    if not label_matches:
        return []

    for index, label_match in enumerate(label_matches):
        block_start = literal_tool_block_strip_start(text, label_match.start())
        block_end = (
            label_matches[index + 1].start()
            if index + 1 < len(label_matches)
            else len(text)
        )
        block_text = text[block_start:block_end]

        current_name = label_match.group("name").strip()
        if not current_name:
            continue
        correlation_match = _LITERAL_CORRELATION_REF_LINE_RE.search(block_text)
        payload_match = _LITERAL_INPUT_PAYLOAD_LINE_RE.search(block_text)
        if not payload_match:
            continue

        payload_source = block_text[payload_match.start("payload") :]
        raw_payload = payload_source.strip()
        if not raw_payload:
            continue
        payload_end = block_end
        try:
            raw_payload, payload_decode_end = runtime.decode_json_prefix(
                payload_source,
                fallback_transform=escape_unescaped_newlines_in_json_payload,
            )
            payload_end = (
                block_start + payload_match.start("payload") + payload_decode_end
            )
        except json.JSONDecodeError:
            pass

        blocks.append(
            {
                "name": current_name,
                "call_id": (
                    correlation_match.group("call_id").strip()
                    if correlation_match
                    else None
                ),
                "payload": raw_payload,
                "start": block_start,
                "end": payload_end,
            }
        )

    return blocks


def parse_literal_tool_payload_json(payload: str) -> object:
    if not payload.strip():
        raise ValueError("empty_tool_payload")

    def _parse_single_json_value(payload_text: str) -> object:
        parsed_value, decode_index = json.JSONDecoder().raw_decode(payload_text)
        if payload_text[decode_index:].strip():
            raise ValueError("invalid_tool_payload_json")
        return parsed_value

    try:
        return _parse_single_json_value(payload.strip())
    except (json.JSONDecodeError, ValueError) as first_error:
        escaped_payload = escape_unescaped_newlines_in_json_payload(payload)
        try:
            return _parse_single_json_value(escaped_payload.strip())
        except (json.JSONDecodeError, ValueError):
            raise ValueError("invalid_tool_payload_json") from first_error


def sanitize_literal_tool_arguments(
    arguments: object,
    parameters: Payload,
) -> object:
    if not isinstance(arguments, dict):
        return arguments
    if parameters.get("additionalProperties") is not False:
        return arguments
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    sanitized = dict(arguments)
    for key in _LITERAL_TOOL_ARGUMENT_METADATA_KEYS:
        if key not in properties:
            sanitized.pop(key, None)
    return sanitized


def escape_unescaped_newlines_in_json_payload(payload: str) -> str:
    if not payload:
        return payload

    output_chars: list[str] = []
    in_json_string = False
    escape_next = False
    for char in payload:
        if escape_next:
            output_chars.append(char)
            escape_next = False
            continue
        if char == "\\":
            output_chars.append(char)
            escape_next = True
            continue
        if char == '"':
            in_json_string = not in_json_string
            output_chars.append(char)
            continue
        if char == "\n" and in_json_string:
            output_chars.append("\\n")
            continue
        output_chars.append(char)
    return "".join(output_chars)


def build_repaired_function_call_output_item(
    *,
    tool_name: str,
    call_id: Optional[str],
    arguments: Payload,
    block_index: int,
) -> Payload:
    resolved_call_id = (
        call_id.strip()
        if isinstance(call_id, str) and call_id.strip()
        else f"call_repaired_{block_index}_{uuid4().hex[:12]}"
    )
    return {
        "type": "function_call",
        "name": tool_name,
        "call_id": resolved_call_id,
        "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
    }


def dedupe_repaired_call_id(
    call_id: Optional[str],
    *,
    block_index: int,
    used_call_ids: set[str],
) -> Optional[str]:
    if not isinstance(call_id, str) or not call_id.strip():
        return call_id
    normalized_call_id = call_id.strip()
    if normalized_call_id not in used_call_ids:
        used_call_ids.add(normalized_call_id)
        return normalized_call_id
    deduped_call_id = f"{normalized_call_id}_repaired_{block_index}"
    suffix = 2
    while deduped_call_id in used_call_ids:
        deduped_call_id = f"{normalized_call_id}_repaired_{block_index}_{suffix}"
        suffix += 1
    used_call_ids.add(deduped_call_id)
    return deduped_call_id


def repair_literal_tool_calls_in_text(
    runtime: Runtime,
    text: str,
    *,
    advertised_tools: AdvertisedTools,
) -> tuple[Optional[str], list[Payload]]:
    blocks = parse_literal_tool_label_blocks(runtime, text)
    if not blocks:
        return None, []

    repaired_items: list[Payload] = []
    strip_spans: list[tuple[int, int]] = []
    used_call_ids: set[str] = set()
    for block_index, block in enumerate(blocks):
        tool_name = str(block.get("name") or "").strip()
        try:
            parsed_arguments = parse_literal_tool_payload_json(
                str(block.get("payload") or "")
            )
        except ValueError:
            return None, []
        if tool_name not in advertised_tools:
            continue
        block_start = block.get("start")
        block_end = block.get("end")
        if not isinstance(block_start, int) or not isinstance(block_end, int):
            return None, []
        strip_spans.append(
            (
                block_start,
                block_end,
            )
        )
        parsed_arguments = sanitize_literal_tool_arguments(
            parsed_arguments,
            advertised_tools[tool_name],
        )
        validation_error = runtime.validate_tool_arguments(
            tool_name=tool_name,
            arguments=parsed_arguments,
            parameters=advertised_tools[tool_name],
        )
        if validation_error is not None:
            return None, []
        if not isinstance(parsed_arguments, dict):
            return None, []
        call_id = block.get("call_id")
        repaired_call_id = dedupe_repaired_call_id(
            call_id if isinstance(call_id, str) else None,
            block_index=block_index,
            used_call_ids=used_call_ids,
        )
        repaired_items.append(
            build_repaired_function_call_output_item(
                tool_name=tool_name,
                call_id=repaired_call_id,
                arguments=dict(parsed_arguments),
                block_index=block_index,
            )
        )
    if not repaired_items or not strip_spans:
        return None, []

    stripped_text = runtime.strip_text_spans(text, strip_spans)
    stripped_text = re.sub(r"\n{3,}", "\n\n", stripped_text).strip()
    if stripped_text and runtime.is_malformed_composer_literal_text(stripped_text):
        return None, []
    return stripped_text or None, repaired_items


def response_body_has_literal_tool_label_blocks(
    runtime: Runtime,
    response_body: Payload,
) -> bool:
    output = response_body.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            if parse_literal_tool_label_blocks(runtime, content):
                return True
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"text", "output_text"}:
                continue
            text = part.get("text")
            if isinstance(text, str) and parse_literal_tool_label_blocks(runtime, text):
                return True
    return False


def repair_literal_tool_calls_in_message_item(
    runtime: Runtime,
    item: Payload,
    *,
    advertised_tools: AdvertisedTools,
) -> Optional[tuple[list[Payload], bool]]:
    content = item.get("content")
    if isinstance(content, str):
        content_parts: list[object] = [{"type": "output_text", "text": content}]
    elif isinstance(content, list):
        content_parts = content
    else:
        return [item], False

    message_items: list[Payload] = []
    leftover_text_parts: list[str] = []
    message_has_literal_blocks = False
    for part in content_parts:
        if not isinstance(part, dict) or part.get("type") not in {
            "text",
            "output_text",
        }:
            return None if message_has_literal_blocks else ([item], False)

        text = part.get("text")
        if not isinstance(text, str):
            text = ""
        if not parse_literal_tool_label_blocks(runtime, text):
            if text.strip():
                leftover_text_parts.append(text.strip())
            continue

        message_has_literal_blocks = True
        leftover_text, repaired_calls = repair_literal_tool_calls_in_text(
            runtime,
            text,
            advertised_tools=advertised_tools,
        )
        if not repaired_calls:
            return None
        if isinstance(leftover_text, str) and leftover_text.strip():
            leftover_text_parts.append(leftover_text.strip())
        message_items.extend(repaired_calls)

    if not message_items:
        return [item], False

    repaired_items: list[Payload] = []
    if leftover_text_parts:
        repaired_item = dict(item)
        repaired_item["content"] = [
            {
                "type": "output_text",
                "text": "\n\n".join(leftover_text_parts),
            }
        ]
        repaired_items.append(repaired_item)
    repaired_items.extend(message_items)
    return repaired_items, True


def try_repair_literal_tool_call_response_body(
    runtime: Runtime,
    response_body: Payload,
    *,
    request_body: Optional[Payload],
) -> Optional[Payload]:
    if not (
        runtime.is_malformed_tool_call_text_output(response_body)
        or response_body_has_literal_tool_label_blocks(runtime, response_body)
    ):
        return None

    advertised_tools = runtime.build_advertised_function_tools_index(request_body)
    if not advertised_tools:
        return None

    output = response_body.get("output")
    if not isinstance(output, list):
        return None

    repaired_output: list[object] = []
    repaired_any = False
    for item in output:
        if not isinstance(item, dict):
            repaired_output.append(item)
            continue
        if item.get("type") != "message":
            repaired_output.append(item)
            continue

        repaired_message = repair_literal_tool_calls_in_message_item(
            runtime,
            dict(item),
            advertised_tools=advertised_tools,
        )
        if repaired_message is None:
            return None
        repaired_items, repaired_item = repaired_message
        repaired_output.extend(repaired_items)
        repaired_any = repaired_any or repaired_item

    if not repaired_any:
        return None

    repaired_body = dict(response_body)
    repaired_body["output"] = repaired_output
    if runtime.is_malformed_tool_call_text_output(repaired_body):
        return None
    return repaired_body


__all__ = [
    "AdvertisedTools",
    "Runtime",
    "build_repaired_function_call_output_item",
    "dedupe_repaired_call_id",
    "escape_unescaped_newlines_in_json_payload",
    "literal_tool_block_strip_start",
    "parse_literal_tool_label_blocks",
    "parse_literal_tool_payload_json",
    "repair_literal_tool_calls_in_message_item",
    "repair_literal_tool_calls_in_text",
    "response_body_has_literal_tool_label_blocks",
    "sanitize_literal_tool_arguments",
    "try_repair_literal_tool_call_response_body",
]
