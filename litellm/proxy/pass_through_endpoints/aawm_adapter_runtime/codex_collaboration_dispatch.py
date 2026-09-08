"""Codex V2 collaboration dispatch normalization for LiteLLM egress.

Codex V2 advertises collaboration ``message`` arguments as encrypted even
though the value is ordinary task text. LiteLLM removes that annotation only
from recognized V2 collaboration schemas and requires an explicit, versioned
plaintext frame in the string argument. Assignments which cannot be proven
readable fail closed before provider egress.

This module owns neither encrypted reasoning nor encrypted function-output
state. Tool-schema handling is deliberately limited to the advertised
``tools``/``functions`` definitions; arbitrary nested request data is never
rewritten.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from fastapi import HTTPException


COLLABORATION_MESSAGE_PROPERTY = "message"
COLLABORATION_FRAME_VERSION = 1
COLLABORATION_TEXT_ENCODING = "text"
ASSIGNMENT_UNREADABLE_ERROR_CODE = "aawm_codex_assignment_unreadable"

_MAX_FRAME_CHARS = 4 * 1024 * 1024
_COLLABORATION_NAMESPACES = frozenset(
    {
        "collaboration",
        "functions.collaboration",
    }
)
_V1_NAMESPACES = frozenset(
    {
        "multi_agent_v1",
        "functions.multi_agent_v1",
    }
)
COLLABORATION_TOOL_NAMES = frozenset(
    {
        "spawn_agent",
        "followup_task",
        "send_message",
    }
)
_QUALIFIED_COLLABORATION_TOOL_NAMES = frozenset(
    f"{namespace}.{name}"
    for namespace in _COLLABORATION_NAMESPACES
    for name in COLLABORATION_TOOL_NAMES
)
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "$ref",
        "$dynamicRef",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentSchemas",
    }
)
_MESSAGE_FRAME_INSTRUCTION = (
    'The message argument must be a plaintext JSON frame of the form '
    '{"cfg047":1,"encoding":"text","text":"<the exact assignment text>"}. '
    "Do not encrypt this argument."
)
_CODEX_ENVELOPE_PATTERN = re.compile(
    r"\AMessage Type: (NEW_TASK|MESSAGE|FINAL_ANSWER)\n"
    r"Task name: ([^\n]+)\n"
    r"Sender: ([^\n]+)\n"
    r"Payload:\n"
)
_FRAME_PREFIX_PATTERN = re.compile(r'\A\{\s*"cfg047"\s*:')
_OPAQUE_PREFIXES = (
    "gAAAA",
    "aawm_erp:",
    "litellm_enc:",
)


class CodexCollaborationDispatchError(ValueError):
    """A collaboration assignment or schema is not safely readable."""

    def __init__(self, reason: str):
        super().__init__(f"unsupported Codex collaboration assignment: {reason}")
        self.reason = reason


class _NormalizedCodexAgentMessage(dict[str, Any]):
    """Python-only marker for an already materialized collaboration payload."""


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


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CodexCollaborationDispatchError("invalid_envelope")
        result[key] = value
    return result


def parse_codex_collaboration_text_frame(value: Any) -> str:
    """Return exact assignment text from a CFG-047 frame or fail closed."""
    if not isinstance(value, str) or not value or len(value) > _MAX_FRAME_CHARS:
        raise CodexCollaborationDispatchError("invalid_envelope")

    try:
        decoded, remainder = json.JSONDecoder(
            object_pairs_hook=_duplicate_rejecting_object,
        ).raw_decode(value)
    except CodexCollaborationDispatchError:
        raise
    except (RecursionError, TypeError, ValueError):
        raise CodexCollaborationDispatchError("unknown_representation") from None

    if remainder != len(value):
        raise CodexCollaborationDispatchError("invalid_envelope")
    if not isinstance(decoded, dict) or set(decoded) != {
        "cfg047",
        "encoding",
        "text",
    }:
        raise CodexCollaborationDispatchError("unknown_representation")
    frame_version = decoded.get("cfg047")
    if (
        not isinstance(frame_version, int)
        or isinstance(frame_version, bool)
        or frame_version != COLLABORATION_FRAME_VERSION
    ):
        raise CodexCollaborationDispatchError("unknown_representation")
    if decoded.get("encoding") != COLLABORATION_TEXT_ENCODING:
        raise CodexCollaborationDispatchError("unknown_representation")
    text = decoded.get("text")
    if not isinstance(text, str) or not text:
        raise CodexCollaborationDispatchError("invalid_envelope")
    return text


def _is_opaque_representation(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in _OPAQUE_PREFIXES)


def _schema_has_unsupported_composition(schema: Mapping[str, Any]) -> bool:
    return any(key in schema for key in _UNSUPPORTED_SCHEMA_KEYS)


def _schema_properties(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        raise CodexCollaborationDispatchError("unsupported_collaboration_message_schema")
    if _schema_has_unsupported_composition(parameters):
        raise CodexCollaborationDispatchError("unsupported_collaboration_message_schema")
    if parameters.get("type") != "object":
        raise CodexCollaborationDispatchError("unsupported_collaboration_message_schema")
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise CodexCollaborationDispatchError("unsupported_collaboration_message_schema")
    return properties


def _is_required(parameters: Mapping[str, Any], name: str) -> bool:
    required = parameters.get("required")
    return isinstance(required, list) and name in required


def _is_string_schema(schema: Any) -> bool:
    return (
        isinstance(schema, dict)
        and not _schema_has_unsupported_composition(schema)
        and schema.get("type") == "string"
    )


def _message_schema(
    parameters: Any,
    *,
    tool_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    properties = _schema_properties(parameters)
    message_schema = properties.get(COLLABORATION_MESSAGE_PROPERTY)
    if not _is_string_schema(message_schema):
        raise CodexCollaborationDispatchError(
            "unsupported_collaboration_message_schema"
        )
    if (
        "encrypted" in message_schema
        and not isinstance(message_schema["encrypted"], bool)
    ):
        raise CodexCollaborationDispatchError(
            "unsupported_collaboration_message_schema"
        )

    if tool_name == "spawn_agent":
        # V1 has ``items``/``fork_context`` and no required ``task_name``.
        # V2 requires both ``task_name`` and ``message`` and uses
        # ``fork_turns`` instead of the V1 context switch.
        task_schema = properties.get("task_name")
        if (
            not _is_string_schema(task_schema)
            or "items" in properties
            or "fork_context" in properties
            or not _is_required(parameters, "task_name")
            or not _is_required(parameters, COLLABORATION_MESSAGE_PROPERTY)
        ):
            raise CodexCollaborationDispatchError(
                "unsupported_collaboration_message_schema"
            )
    else:
        target_schema = properties.get("target")
        if (
            not _is_string_schema(target_schema)
            or not _is_required(parameters, "target")
            or not _is_required(parameters, COLLABORATION_MESSAGE_PROPERTY)
        ):
            raise CodexCollaborationDispatchError(
                "unsupported_collaboration_message_schema"
            )
    return properties, message_schema


def _append_frame_instruction(description: Any) -> str:
    if not isinstance(description, str) or not description:
        return _MESSAGE_FRAME_INSTRUCTION
    if _MESSAGE_FRAME_INSTRUCTION in description:
        return description
    separator = "" if description.endswith((" ", "\n")) else " "
    return f"{description}{separator}{_MESSAGE_FRAME_INSTRUCTION}"


def _normalize_targeted_parameters(
    parameters: Any,
    *,
    tool_name: str,
) -> tuple[dict[str, Any], bool]:
    properties, message_schema = _message_schema(
        parameters,
        tool_name=tool_name,
    )
    if message_schema.get("encrypted") is not True:
        return parameters, False

    normalized_message = dict(message_schema)
    normalized_message.pop("encrypted")
    normalized_message["description"] = _append_frame_instruction(
        normalized_message.get("description")
    )
    normalized_properties = dict(properties)
    normalized_properties[COLLABORATION_MESSAGE_PROPERTY] = normalized_message
    normalized_parameters = dict(parameters)
    normalized_parameters["properties"] = normalized_properties
    return normalized_parameters, True


def _function_tool_parts(
    tool: dict[str, Any],
) -> Optional[tuple[dict[str, Any], dict[str, Any], str, Any]]:
    if tool.get("type") != "function":
        return None
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str):
            return tool, function, name, function.get("parameters")
        return None
    name = tool.get("name")
    if isinstance(name, str):
        return tool, tool, name, tool.get("parameters")
    return None


def _tool_namespace(tool: Mapping[str, Any], function: Mapping[str, Any]) -> Any:
    namespace = tool.get("namespace")
    if namespace is None:
        namespace = function.get("namespace")
    return namespace


def _is_v1_spawn_schema(parameters: Any) -> bool:
    if not isinstance(parameters, dict):
        return False
    properties = parameters.get("properties")
    return (
        isinstance(properties, dict)
        and "task_name" not in properties
        and ("items" in properties or "fork_context" in properties)
    )


def _has_encrypted_message_marker(parameters: Any) -> bool:
    if not isinstance(parameters, dict):
        return False
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return False
    message_schema = properties.get(COLLABORATION_MESSAGE_PROPERTY)
    return (
        isinstance(message_schema, dict)
        and message_schema.get("encrypted") is True
    )


def _is_canonical_v2_tool_schema(
    parameters: Any,
    *,
    tool_name: str,
) -> bool:
    """Require the encrypted message contract before bare-name matching."""
    if not _has_encrypted_message_marker(parameters):
        return False
    _message_schema(parameters, tool_name=tool_name)
    return True


def _targeted_tool_name(
    *,
    name: str,
    namespace: Any,
    parameters: Any,
    namespace_context: Optional[str] = None,
) -> Optional[str]:
    if namespace is not None and (
        not isinstance(namespace, str)
        or namespace not in _COLLABORATION_NAMESPACES
    ):
        # An explicit non-Codex namespace owns its bare tool names.
        return None
    if namespace_context in _V1_NAMESPACES:
        return None

    if namespace_context in _COLLABORATION_NAMESPACES:
        if name in COLLABORATION_TOOL_NAMES:
            return name
        return None

    if name in _QUALIFIED_COLLABORATION_TOOL_NAMES:
        return name.rsplit(".", 1)[-1]
    if namespace in _COLLABORATION_NAMESPACES and name in COLLABORATION_TOOL_NAMES:
        return name
    if namespace in _V1_NAMESPACES:
        return None
    if name in {"followup_task", "send_message"}:
        return (
            name
            if _is_canonical_v2_tool_schema(parameters, tool_name=name)
            else None
        )
    if name == "spawn_agent":
        if _is_v1_spawn_schema(parameters):
            return None
        return (
            name
            if _is_canonical_v2_tool_schema(parameters, tool_name=name)
            else None
        )
    return None


def _normalize_function_tool(
    tool: dict[str, Any],
    *,
    namespace_context: Optional[str] = None,
) -> tuple[dict[str, Any], bool]:
    parts = _function_tool_parts(tool)
    if parts is None:
        return tool, False
    _, function, name, parameters = parts
    explicit_namespaces = [
        namespace
        for namespace in (tool.get("namespace"), function.get("namespace"))
        if namespace is not None
    ]
    if any(
        not isinstance(namespace, str)
        or namespace not in _COLLABORATION_NAMESPACES
        for namespace in explicit_namespaces
    ) or len(set(explicit_namespaces)) > 1:
        # Foreign, V1, malformed, or conflicting namespace evidence is
        # outside this normalizer's ownership.
        return tool, False
    target_name = _targeted_tool_name(
        name=name,
        namespace=_tool_namespace(tool, function),
        parameters=parameters,
        namespace_context=namespace_context,
    )
    if target_name is None:
        return tool, False

    normalized_parameters, changed = _normalize_targeted_parameters(
        parameters,
        tool_name=target_name,
    )
    if not changed:
        return tool, False

    normalized_tool = dict(tool)
    if function is tool:
        normalized_tool["parameters"] = normalized_parameters
    else:
        normalized_function = dict(function)
        normalized_function["parameters"] = normalized_parameters
        normalized_tool["function"] = normalized_function
    return normalized_tool, True


def _normalize_namespace_tool(tool: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    namespace = tool.get("name")
    if namespace not in _COLLABORATION_NAMESPACES:
        return tool, False
    children = tool.get("tools")
    if not isinstance(children, list):
        raise CodexCollaborationDispatchError(
            "unsupported_collaboration_message_schema"
        )

    normalized_children = list(children)
    changed = False
    for index, child in enumerate(children):
        if not isinstance(child, dict) or child.get("type") != "function":
            continue
        normalized_child, child_changed = _normalize_function_tool(
            child,
            namespace_context=namespace,
        )
        if child_changed:
            normalized_children[index] = normalized_child
            changed = True
    if not changed:
        return tool, False
    normalized_tool = dict(tool)
    normalized_tool["tools"] = normalized_children
    return normalized_tool, True


def _normalize_tool_definition(tool: Any) -> tuple[Any, bool]:
    if not isinstance(tool, dict):
        return tool, False
    if tool.get("type") == "namespace":
        return _normalize_namespace_tool(tool)
    return _normalize_function_tool(tool)


def _normalize_tool_list(tools: Any) -> tuple[Any, bool]:
    if not isinstance(tools, list):
        return tools, False
    normalized_tools = list(tools)
    changed = False
    for index, tool in enumerate(tools):
        normalized_tool, tool_changed = _normalize_tool_definition(tool)
        if tool_changed:
            normalized_tools[index] = normalized_tool
            changed = True
    return (normalized_tools if changed else tools), changed


def _normalize_tool_schemas_without_error_mapping(
    body: dict[str, Any],
) -> dict[str, Any]:
    normalized_body = body
    changed = False
    for key in ("tools", "functions"):
        normalized_tools, tools_changed = _normalize_tool_list(
            normalized_body.get(key)
        )
        if not tools_changed:
            continue
        if not changed:
            normalized_body = dict(body)
            changed = True
        normalized_body[key] = normalized_tools
    return normalized_body


def normalize_codex_collaboration_tool_schemas(
    body: dict[str, Any],
) -> dict[str, Any]:
    """Normalize only recognized V2 collaboration message schemas."""
    try:
        return _normalize_tool_schemas_without_error_mapping(body)
    except CodexCollaborationDispatchError as exc:
        raise_codex_assignment_unreadable(reason=exc.reason)
    raise AssertionError("unreachable")


def _parse_collaboration_envelope(text: str) -> Optional[tuple[str, str, str, int]]:
    if not text.startswith("Message Type: "):
        return None
    match = _CODEX_ENVELOPE_PATTERN.match(text)
    if match is None:
        raise CodexCollaborationDispatchError("invalid_envelope")
    return (
        match.group(1),
        match.group(2),
        match.group(3),
        match.end(),
    )


def _validate_envelope_identity(
    item: Mapping[str, Any],
    *,
    task_name: str,
    sender: str,
) -> None:
    author = item.get("author")
    recipient = item.get("recipient")
    if (
        not isinstance(author, str)
        or not author
        or not isinstance(recipient, str)
        or not recipient
        or task_name != recipient
        or sender != author
    ):
        raise CodexCollaborationDispatchError("invalid_envelope")


def _validate_visible_agent_message(
    item: dict[str, Any],
    visible_part: dict[str, Any],
) -> Optional[dict[str, Any]]:
    visible_text = visible_part.get("text")
    if not isinstance(visible_text, str):
        raise CodexCollaborationDispatchError("invalid_envelope")
    envelope = _parse_collaboration_envelope(visible_text)
    if envelope is None:
        return None

    message_type, task_name, sender, payload_offset = envelope
    _validate_envelope_identity(item, task_name=task_name, sender=sender)
    remainder = visible_text[payload_offset:]
    if message_type == "FINAL_ANSWER":
        # A child result is content, even when its text happens to begin with
        # a representation-looking prefix. Never reinterpret it as a task.
        return None
    if not remainder:
        raise CodexCollaborationDispatchError("invalid_envelope")
    if _is_opaque_representation(remainder):
        raise CodexCollaborationDispatchError("opaque")
    if _FRAME_PREFIX_PATTERN.match(remainder):
        assignment = parse_codex_collaboration_text_frame(remainder)
        normalized_item = _NormalizedCodexAgentMessage(item)
        normalized_item["content"] = [
            {
                "type": visible_part["type"],
                "text": f"{visible_text[:payload_offset]}{assignment}",
            }
        ]
        return normalized_item
    return None


def _normalize_agent_message_item(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if isinstance(item, _NormalizedCodexAgentMessage):
        return item, False

    content = item.get("content")
    if not isinstance(content, list):
        return item, False

    encrypted_parts = [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "encrypted_content"
    ]
    if encrypted_parts:
        if len(content) != 2 or len(encrypted_parts) != 1:
            raise CodexCollaborationDispatchError("invalid_envelope")
        visible_part, payload_part = content
        if (
            not isinstance(visible_part, dict)
            or set(visible_part) != {"type", "text"}
            or visible_part.get("type") not in {"input_text", "text"}
            or not isinstance(payload_part, dict)
            or set(payload_part) != {"type", "encrypted_content"}
            or payload_part.get("type") != "encrypted_content"
        ):
            raise CodexCollaborationDispatchError("invalid_envelope")
        visible_text = visible_part.get("text")
        if not isinstance(visible_text, str):
            raise CodexCollaborationDispatchError("invalid_envelope")
        envelope = _parse_collaboration_envelope(visible_text)
        if envelope is None:
            raise CodexCollaborationDispatchError("invalid_envelope")
        message_type, task_name, sender, payload_offset = envelope
        if payload_offset != len(visible_text):
            raise CodexCollaborationDispatchError("invalid_envelope")
        _validate_envelope_identity(item, task_name=task_name, sender=sender)
        if message_type == "FINAL_ANSWER":
            raise CodexCollaborationDispatchError("invalid_envelope")
        payload = payload_part.get("encrypted_content")
        if not isinstance(payload, str) or not payload:
            raise CodexCollaborationDispatchError("invalid_envelope")
        if _is_opaque_representation(payload):
            raise CodexCollaborationDispatchError("opaque")
        assignment = parse_codex_collaboration_text_frame(payload)
        normalized_item = _NormalizedCodexAgentMessage(item)
        normalized_item["content"] = [
            {
                "type": visible_part["type"],
                "text": f"{visible_text}{assignment}",
            }
        ]
        return normalized_item, True

    if len(content) != 1:
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"input_text", "text"}:
                continue
            text = part.get("text")
            if isinstance(text, str) and text.startswith("Message Type: "):
                raise CodexCollaborationDispatchError("invalid_envelope")

    if len(content) == 1 and isinstance(content[0], dict):
        visible_part = content[0]
        if visible_part.get("type") in {"input_text", "text"}:
            if set(visible_part) != {"type", "text"}:
                raise CodexCollaborationDispatchError("invalid_envelope")
            visible_text = visible_part.get("text")
            if isinstance(visible_text, str) and _is_opaque_representation(
                visible_text
            ):
                raise CodexCollaborationDispatchError("opaque")
            normalized_item = _validate_visible_agent_message(item, visible_part)
            if normalized_item is not None:
                return normalized_item, True
    return item, False


def _normalize_input_without_error_mapping(body: dict[str, Any]) -> dict[str, Any]:
    input_items = body.get("input")
    if not isinstance(input_items, list):
        return body

    normalized_items = list(input_items)
    changed = False
    for index, item in enumerate(input_items):
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        normalized_item, item_changed = _normalize_agent_message_item(item)
        if item_changed:
            normalized_items[index] = normalized_item
            changed = True
    if not changed:
        return body
    normalized_body = dict(body)
    normalized_body["input"] = normalized_items
    return normalized_body


def normalize_codex_collaboration_input(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize validated collaboration envelopes and reject opaque payloads."""
    try:
        return _normalize_input_without_error_mapping(body)
    except CodexCollaborationDispatchError as exc:
        raise_codex_assignment_unreadable(reason=exc.reason)
    raise AssertionError("unreachable")


def normalize_codex_collaboration_dispatch_body(
    request_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize schemas and assignments, failing closed before provider send."""
    if not isinstance(request_body, dict):
        if isinstance(request_body, Mapping):
            return dict(request_body)
        return {}

    try:
        body = _normalize_tool_schemas_without_error_mapping(request_body)
        return _normalize_input_without_error_mapping(body)
    except CodexCollaborationDispatchError as exc:
        raise_codex_assignment_unreadable(reason=exc.reason)
    raise AssertionError("unreachable")


def restore_codex_agent_message_payloads_for_openai_egress(
    request_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the former restore-only entry point."""
    return normalize_codex_collaboration_dispatch_body(request_body)
