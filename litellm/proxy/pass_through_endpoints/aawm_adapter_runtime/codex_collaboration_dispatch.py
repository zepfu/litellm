"""Codex V2 collaboration dispatch normalization for LiteLLM egress.

Unmodified Codex clients mark collaboration ``message`` arguments as encrypted
in the advertised tool schema. LiteLLM removes that annotation only for
recognized collaboration message schemas and asks the client to send an
explicit versioned plaintext frame. Assignments which cannot be represented
that way fail closed before any provider call.

This module intentionally owns neither encrypted reasoning nor encrypted
function-output state.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from fastapi import HTTPException


COLLABORATION_TOOL_NAMES = frozenset(
    {
        "spawn_agent",
        "followup_task",
        "send_message",
    }
)
COLLABORATION_MESSAGE_PROPERTY = "message"
COLLABORATION_FRAME_VERSION = 1
COLLABORATION_TEXT_ENCODING = "text"
ASSIGNMENT_UNREADABLE_ERROR_CODE = "aawm_codex_assignment_unreadable"

_MAX_FRAME_CHARS = 4 * 1024 * 1024
_ENCRYPTED_SCHEMA_KEYS = frozenset({"$ref", "allOf", "anyOf", "oneOf", "not"})
_MESSAGE_FRAME_INSTRUCTION = (
    'The message argument must be a plaintext JSON frame of the form '
    '{"cfg047":1,"encoding":"text","text":"<the exact assignment text>"}. '
    "Do not encrypt this argument."
)
_CODEX_EMPTY_ENVELOPE_PATTERN = re.compile(
    r"\AMessage Type: (NEW_TASK|MESSAGE)\n"
    r"Task name: ([^\n]+)\n"
    r"Sender: ([^\n]+)\n"
    r"Payload:(?:\n?)\Z"
)
_CODEX_ENVELOPE_PREFIX_PATTERN = re.compile(
    r"\AMessage Type: (?:NEW_TASK|MESSAGE)\n"
    r"Task name: ([^\n]+)\n"
    r"Sender: ([^\n]+)\n"
    r"Payload:\n?"
)
_FERNET_TOKEN_PREFIX = "gAAAA"


class CodexCollaborationDispatchError(ValueError):
    """A collaboration assignment is present but not safely readable."""

    def __init__(self, reason: str):
        super().__init__(f"unsupported Codex collaboration assignment: {reason}")
        self.reason = reason


def raise_codex_assignment_unreadable(
    *,
    reason: str,
    failure_phase: str = "codex_collaboration_dispatch_preflight",
) -> None:
    """Fail before provider egress with a regenerate-required contract."""
    raise HTTPException(
        status_code=409,
        detail={
            "error": {
                "message": (
                    "The Codex collaboration assignment cannot be read by the "
                    "selected route. Regenerate the agent assignment."
                ),
                "type": "invalid_request_error",
                "code": ASSIGNMENT_UNREADABLE_ERROR_CODE,
                "reason": reason,
            },
            "reason": reason,
            "attempted_provider_call": False,
            "failure_phase": failure_phase,
            "non_resumable": True,
            "regenerate_assignment_required": True,
        },
    )


def _fail_unreadable(reason: str):
    raise_codex_assignment_unreadable(reason=reason)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CodexCollaborationDispatchError("invalid_envelope")
        result[key] = value
    return result


def parse_codex_collaboration_text_frame(value: Any) -> str:
    """Return exact assignment text from a CFG-047 frame or fail closed."""
    if not isinstance(value, str):
        raise CodexCollaborationDispatchError("invalid_envelope")
    if not value:
        raise CodexCollaborationDispatchError("invalid_envelope")
    if len(value) > _MAX_FRAME_CHARS:
        raise CodexCollaborationDispatchError("invalid_envelope")

    try:
        decoded, remainder = json.JSONDecoder(
            object_pairs_hook=_duplicate_rejecting_object,
        ).raw_decode(value)
    except CodexCollaborationDispatchError:
        raise
    except (ValueError, TypeError):
        raise CodexCollaborationDispatchError("invalid_envelope") from None
    if remainder != len(value):
        raise CodexCollaborationDispatchError("invalid_envelope")
    if not isinstance(decoded, dict) or set(decoded) != {
        "cfg047",
        "encoding",
        "text",
    }:
        raise CodexCollaborationDispatchError("unknown_representation")
    if decoded.get("cfg047") is not COLLABORATION_FRAME_VERSION:
        raise CodexCollaborationDispatchError("unknown_representation")
    if decoded.get("encoding") != COLLABORATION_TEXT_ENCODING:
        raise CodexCollaborationDispatchError("unknown_representation")
    text = decoded.get("text")
    if not isinstance(text, str) or not text:
        raise CodexCollaborationDispatchError("invalid_envelope")
    return text


def _is_collaboration_tool(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    if tool.get("type") == "function":
        return tool.get("name") in COLLABORATION_TOOL_NAMES
    if tool.get("type") == "namespace":
        return any(_is_collaboration_tool(child) for child in tool.get("tools") or [])
    return False


def _remove_message_encryption(schema: Any) -> tuple[Any, bool]:
    """Remove encryption only from an exact collaboration string schema."""
    if not isinstance(schema, dict):
        raise CodexCollaborationDispatchError("invalid_tool_schema")
    if any(key in schema for key in _ENCRYPTED_SCHEMA_KEYS):
        raise CodexCollaborationDispatchError("invalid_tool_schema")
    if schema.get("type") != "string":
        raise CodexCollaborationDispatchError("invalid_tool_schema")
    if schema.get("encrypted") is not True:
        return schema, False

    normalized = dict(schema)
    normalized.pop("encrypted")
    description = normalized.get("description")
    if isinstance(description, str) and description and not description.endswith("."):
        description = f"{description}."
    normalized["description"] = (
        f"{description} {_MESSAGE_FRAME_INSTRUCTION}"
        if description
        else _MESSAGE_FRAME_INSTRUCTION
    )
    return normalized, True


def _normalize_encrypted_markers(node: Any) -> tuple[Any, int]:
    """Normalize any encrypted marker or fail closed if it is not supported."""
    if isinstance(node, dict):
        if node.get("encrypted") is True:
            normalized, changed = _remove_message_encryption(node)
            return normalized, int(changed)
        changed_count = 0
        normalized_node = dict(node)
        for key, child in node.items():
            normalized_child, child_changes = _normalize_encrypted_markers(child)
            if normalized_child is not child:
                normalized_node[key] = normalized_child
            changed_count += child_changes
        return normalized_node, changed_count
    if isinstance(node, list):
        changed_count = 0
        normalized_node = list(node)
        for index, child in enumerate(node):
            normalized_child, child_changes = _normalize_encrypted_markers(child)
            if normalized_child is not child:
                normalized_node[index] = normalized_child
            changed_count += child_changes
        return normalized_node, changed_count
    return node, 0


def normalize_codex_collaboration_tool_schemas(body: dict[str, Any]) -> dict[str, Any]:
    """De-encrypt recognized collaboration message schemas in-place.

    Namespaced collaboration tools are handled by recursing into their child
    function definitions. Schemas outside the recognized tools are untouched.
    """
    tools = body.get("tools")
    if not isinstance(tools, list):
        return body

    changed = False
    for index, tool in enumerate(tools):
        if not _is_collaboration_tool(tool):
            continue
        normalized_tool, changes = _normalize_encrypted_markers(tool)
        if changes and isinstance(normalized_tool, dict):
            tools[index] = normalized_tool
            changed = True
        elif changes:
            _fail_unreadable("invalid_tool_schema")
    if not changed:
        return body
    return dict(body)


def _split_envelope_and_frame(text: str) -> tuple[str, str, str]:
    """Return (envelope prefix, remainder, parsed frame source)."""
    if not text.startswith("Message Type: "):
        return "", "", text
    prefix_match = _CODEX_ENVELOPE_PREFIX_PATTERN.match(text)
    if prefix_match is None:
        return "", "", text
    remainder = text[prefix_match.end() :]
    return prefix_match.group(0), remainder, remainder


def normalize_codex_collaboration_input(  # noqa: PLR0915
    body: dict[str, Any],
) -> dict[str, Any]:
    """Restore readable envelopes and validate all collaboration payloads."""
    input_items = body.get("input")
    if not isinstance(input_items, list):
        return body

    changed = False
    updated_items: list[Any] = []
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            updated_items.append(item)
            continue
        content = item.get("content")
        if not isinstance(content, list) or len(content) != 2:
            updated_items.append(item)
            continue
        visible_part, payload_part = content
        if not isinstance(visible_part, dict) or not isinstance(payload_part, dict):
            updated_items.append(item)
            continue
        if visible_part.get("type") != "input_text" or not isinstance(
            visible_part.get("text"),
            str,
        ):
            updated_items.append(item)
            continue

        visible_text = visible_part["text"]
        envelope_match = _CODEX_EMPTY_ENVELOPE_PATTERN.fullmatch(visible_text)
        if envelope_match is not None:
            payload = payload_part.get("encrypted_content")
            if payload_part.get("type") != "encrypted_content" or not isinstance(
                payload,
                str,
            ):
                updated_items.append(item)
                continue
            if payload.startswith(_FERNET_TOKEN_PREFIX):
                _fail_unreadable("opaque")
            try:
                assignment = parse_codex_collaboration_text_frame(payload)
            except CodexCollaborationDispatchError as exc:
                _fail_unreadable(exc.reason)
            updated_item = dict(item)
            updated_item["content"] = [
                {
                    "type": "input_text",
                    "text": f"{visible_text}{assignment}",
                }
            ]
            updated_items.append(updated_item)
            changed = True
            continue

        # Validate material restored from current or historical input. Empty
        # envelopes remain valid; legacy readable text is preserved exactly;
        # ciphertext and malformed/unknown frames fail closed.
        if visible_text.startswith("Message Type: "):
            _prefix, remainder, _frame_source = _split_envelope_and_frame(
                visible_text,
            )
            if remainder == "":
                updated_items.append(item)
                continue
            if remainder.startswith("{"):
                try:
                    parse_codex_collaboration_text_frame(remainder)
                except CodexCollaborationDispatchError as exc:
                    _fail_unreadable(exc.reason)
            elif remainder.startswith(_FERNET_TOKEN_PREFIX):
                _fail_unreadable("opaque")
        updated_items.append(item)

    if not changed:
        return body
    updated = dict(body)
    updated["input"] = updated_items
    return updated


def normalize_codex_collaboration_dispatch_body(
    request_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize schemas, restore readable assignments, and fail closed."""
    if not isinstance(request_body, dict):
        return dict(request_body) if isinstance(request_body, Mapping) else {}

    body = dict(request_body)
    body = normalize_codex_collaboration_tool_schemas(body)
    return normalize_codex_collaboration_input(body)


def restore_codex_agent_message_payloads_for_openai_egress(
    request_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the previous restore-only entry point."""
    return normalize_codex_collaboration_dispatch_body(request_body)
