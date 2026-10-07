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

import hashlib
import json
import re
import contextvars
from dataclasses import dataclass
from typing import Any, Mapping, MutableSequence, Optional

from fastapi import HTTPException

from litellm.responses.function_name_sanitization import (
    ResponsesFunctionIdentity,
)


COLLABORATION_MESSAGE_PROPERTY = "message"
COLLABORATION_FRAME_VERSION = 1
COLLABORATION_TEXT_ENCODING = "text"
ASSIGNMENT_UNREADABLE_ERROR_CODE = "aawm_codex_assignment_unreadable"
SEND_MESSAGE_NORMALIZATION_FAILURE_PHASE = (
    "codex_collaboration_send_message_normalization"
)
CODEX_COLLABORATION_TOOL_IDENTITIES_STATE_FIELD = (
    "_aawm_codex_collaboration_tool_identities"
)
CODEX_CAPTURE_EVIDENCE_STATE_FIELD = (
    "_aawm_cfg072_verified_capture"
)
CODEX_CAPTURE_EVIDENCE_CONTEXT_FIELD = (
    "_aawm_cfg072_verified_capture_context"
)
CFG072_REJECT_MARKER = "__cfg072_reject_requires_exact_cfg047_text_frame"
CFG072_SEND_MESSAGE_OUTPUT_REJECTED_CODE = (
    "aawm_cfg072_send_message_output_rejected"
)

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


@dataclass(frozen=True)
class CodexCollaborationToolAlias:
    """A validated client identity and its native OpenAI wire alias."""

    original: ResponsesFunctionIdentity
    upstream_name: str


def bind_codex_collaboration_tool_identities(
    request: Any,
    identities: MutableSequence[ResponsesFunctionIdentity]
    | tuple[ResponsesFunctionIdentity, ...],
) -> None:
    """Keep validated collaboration identities beside, never inside, the body."""
    if not identities:
        return
    state = getattr(request, "state", None)
    if state is None:
        return
    existing = getattr(
        state,
        CODEX_COLLABORATION_TOOL_IDENTITIES_STATE_FIELD,
        (),
    )
    if not isinstance(existing, tuple):
        existing = ()
    merged = tuple(dict.fromkeys((*existing, *identities)))
    try:
        setattr(
            state,
            CODEX_COLLABORATION_TOOL_IDENTITIES_STATE_FIELD,
            merged,
        )
    except Exception:
        return


def get_bound_codex_collaboration_tool_identities(
    request: Any,
) -> tuple[ResponsesFunctionIdentity, ...]:
    """Return request-local validated identities without exposing body state."""
    state = getattr(request, "state", None)
    identities = getattr(
        state,
        CODEX_COLLABORATION_TOOL_IDENTITIES_STATE_FIELD,
        (),
    )
    if not isinstance(identities, tuple):
        return ()
    return tuple(
        identity
        for identity in identities
        if isinstance(identity, ResponsesFunctionIdentity)
    )


def _codex_alias_token(value: Optional[str], *, fallback: str) -> str:
    if value is None:
        return fallback
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return token or fallback


def build_codex_collaboration_wire_aliases(
    identities: tuple[ResponsesFunctionIdentity, ...]
    | list[ResponsesFunctionIdentity],
    *,
    reserved_names: Optional[set[str] | frozenset[str]] = None,
) -> tuple[CodexCollaborationToolAlias, ...]:
    """Build stable, nonreserved aliases for validated V2 tool identities."""
    aliases: list[CodexCollaborationToolAlias] = []
    used_upstream_names: set[str] = set()
    reserved_names = reserved_names or frozenset()
    for identity in sorted(
        set(identities),
        key=lambda value: (value.namespace or "", value.name),
    ):
        namespace_token = _codex_alias_token(
            identity.namespace,
            fallback="default",
        )
        name_token = _codex_alias_token(identity.name, fallback="tool")
        candidate = f"aawm_cfg047_v1_{namespace_token}_{name_token}"
        if len(candidate) > 64:
            digest = hashlib.sha256(
                f"{identity.namespace or ''}\0{identity.name}".encode("utf-8")
            ).hexdigest()[:16]
            candidate = f"aawm_cfg047_v1_{digest}"
        if candidate in used_upstream_names or candidate in reserved_names:
            raise_codex_assignment_unreadable(
                reason="collaboration_tool_alias_collision"
            )
        used_upstream_names.add(candidate)
        aliases.append(
            CodexCollaborationToolAlias(
                original=identity,
                upstream_name=candidate,
            )
        )
    return tuple(aliases)


def collect_codex_collaboration_advertised_tool_names(
    body: Mapping[str, Any],
) -> frozenset[str]:
    """Return function names advertised by the request's tool definitions."""
    names: set[str] = set()

    def visit(tool: Any, *, allow_legacy_function: bool) -> None:
        if not isinstance(tool, dict):
            return
        if tool.get("type") == "namespace":
            children = tool.get("tools")
            if isinstance(children, list):
                for child in children:
                    visit(child, allow_legacy_function=False)
            return
        tool_type = tool.get("type")
        if tool_type is not None and tool_type != "function":
            return
        function = tool.get("function")
        name = (
            function.get("name")
            if isinstance(function, dict)
            else tool.get("name")
        )
        if isinstance(name, str):
            names.add(name)
        elif allow_legacy_function:
            legacy_name = tool.get("name")
            if isinstance(legacy_name, str):
                names.add(legacy_name)

    for key in ("tools", "functions"):
        definitions = body.get(key)
        if not isinstance(definitions, list):
            continue
        allow_legacy_function = key == "functions"
        for tool in definitions:
            visit(tool, allow_legacy_function=allow_legacy_function)
    return frozenset(names)


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


def is_codex_collaboration_send_message_identity(
    identity: Any,
) -> bool:
    """Return whether an identity is the validated V2 send-message tool."""
    return (
        isinstance(identity, ResponsesFunctionIdentity)
        and identity.name == "send_message"
        and identity.namespace in _COLLABORATION_NAMESPACES
    )


def canonicalize_codex_send_message_argument(value: Any) -> str:
    """Wrap readable generated text in the strict CFG-047 message frame.

    A strict frame is returned unchanged. Plain readable text is preserved
    exactly inside a newly allocated frame. Frame-looking values other than a
    strict frame are rejected rather than guessed at; this deliberately covers
    four-key near-frames until their recipient semantics are verified.
    """
    if not isinstance(value, str) or not value:
        raise CodexCollaborationDispatchError("invalid_envelope")
    if _is_opaque_representation(value):
        raise CodexCollaborationDispatchError("opaque")
    if _FRAME_PREFIX_PATTERN.match(value):
        # A strict frame is already canonical; validate and preserve its exact
        # serialized bytes instead of re-encoding it.
        parse_codex_collaboration_text_frame(value)
        return value
    canonical = json.dumps(
        {
            "cfg047": COLLABORATION_FRAME_VERSION,
            "encoding": COLLABORATION_TEXT_ENCODING,
            "text": value,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(canonical) > _MAX_FRAME_CHARS:
        raise CodexCollaborationDispatchError("invalid_envelope")
    if parse_codex_collaboration_text_frame(canonical) != value:
        raise CodexCollaborationDispatchError("invalid_envelope")
    return canonical


def _decode_complete_json(value: str) -> tuple[Any, int]:
    try:
        decoded, remainder = json.JSONDecoder(
            object_pairs_hook=_duplicate_rejecting_object,
        ).raw_decode(value)
    except CodexCollaborationDispatchError:
        raise
    except (RecursionError, TypeError, ValueError):
        raise CodexCollaborationDispatchError("unknown_representation") from None
    return decoded, remainder


def gate_generated_codex_send_message_call_arguments(
    value: Any,
    upstream_names: Any,
) -> Any:
    """Gate generated response send-message calls without changing history."""
    if not isinstance(upstream_names, (set, frozenset)) or not upstream_names:
        return value
    if not isinstance(value, list):
        return value
    result: list[Any] = []
    changed = False
    for item in value:
        if (
            not isinstance(item, dict)
            or item.get("type") != "function_call"
            or item.get("name") not in upstream_names
        ):
            result.append(item)
            continue
        try:
            arguments = canonicalize_generated_codex_send_message_call_arguments(
                item.get("arguments")
            )
        except CodexCollaborationDispatchError as exc:
            raise_codex_send_message_output_rejected(exc.reason)
        if arguments == item.get("arguments"):
            result.append(item)
            continue
        updated_item = dict(item)
        updated_item["arguments"] = arguments
        result.append(updated_item)
        changed = True
    return result if changed else value


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
    if _is_opaque_representation(text):
        raise CodexCollaborationDispatchError("opaque")
    return text


def _parse_codex_collaboration_payload(value: Any) -> str:
    """Validate visible task text; encrypted slots require a complete frame."""
    if not isinstance(value, str) or not value or len(value) > _MAX_FRAME_CHARS:
        raise CodexCollaborationDispatchError("invalid_envelope")
    if _is_opaque_representation(value):
        raise CodexCollaborationDispatchError("opaque")
    if _FRAME_PREFIX_PATTERN.match(value):
        return parse_codex_collaboration_text_frame(value)
    return value


def _is_opaque_representation(value: str) -> bool:
    stripped = value.strip()
    return any(stripped.startswith(prefix) for prefix in _OPAQUE_PREFIXES) or (
        len(stripped) >= 64
        and stripped.startswith("gAAAA")
        and re.fullmatch(r"[A-Za-z0-9_=-]+", stripped) is not None
    )


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
    description = message_schema.get("description")
    if (
        "encrypted" not in message_schema
        and isinstance(description, str)
        and _MESSAGE_FRAME_INSTRUCTION in description
    ):
        return parameters, False

    normalized_message = dict(message_schema)
    normalized_message.pop("encrypted", None)
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
    *,
    allow_legacy_function: bool = False,
) -> Optional[tuple[dict[str, Any], dict[str, Any], str, Any]]:
    if tool.get("type") == "namespace":
        return None
    if tool.get("type") != "function" and not (
        allow_legacy_function and tool.get("type") is None
    ):
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


def _collaboration_identity_parts(
    name: str,
    *,
    explicit_namespace: Any,
    namespace_context: Optional[str],
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    namespace = (
        explicit_namespace
        if explicit_namespace is not None
        else namespace_context
    )
    if namespace is not None:
        return name, namespace if isinstance(namespace, str) else None, None, None
    if "." in name:
        qualified_namespace, qualified_name = name.rsplit(".", 1)
        if qualified_namespace in _COLLABORATION_NAMESPACES:
            return qualified_name, qualified_namespace, name, None
    return name, None, None, None


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
    """Recognize the stock schema and our already prepared plaintext schema."""
    if not _has_encrypted_message_marker(parameters):
        if not isinstance(parameters, dict):
            return False
        properties = parameters.get("properties")
        message = properties.get("message") if isinstance(properties, dict) else None
        if (
            not isinstance(message, dict)
            or "encrypted" in message
            or not isinstance(message.get("description"), str)
            or _MESSAGE_FRAME_INSTRUCTION not in message["description"]
        ):
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
    allow_legacy_function: bool = False,
    identity_collector: Optional[
        MutableSequence[ResponsesFunctionIdentity]
    ] = None,
) -> tuple[dict[str, Any], bool]:
    parts = _function_tool_parts(
        tool,
        allow_legacy_function=allow_legacy_function,
    )
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
    if identity_collector is not None:
        explicit_namespace = _tool_namespace(tool, function)
        (
            identity_name,
            identity_namespace,
            original_name,
            original_namespace,
        ) = _collaboration_identity_parts(
            name,
            explicit_namespace=explicit_namespace,
            namespace_context=namespace_context,
        )
        identity_collector.append(
            ResponsesFunctionIdentity(
                name=identity_name,
                namespace=identity_namespace,
                original_name=original_name,
                original_namespace=original_namespace,
            )
        )

    # Alias discovery must survive repeated preparation and copied requests;
    # it is not a side effect of removing the encryption annotation.
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


def _normalize_namespace_tool(
    tool: dict[str, Any],
    *,
    identity_collector: Optional[
        MutableSequence[ResponsesFunctionIdentity]
    ] = None,
) -> tuple[dict[str, Any], bool]:
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
            identity_collector=identity_collector,
        )
        if child_changed:
            normalized_children[index] = normalized_child
            changed = True
    if not changed:
        return tool, False
    normalized_tool = dict(tool)
    normalized_tool["tools"] = normalized_children
    return normalized_tool, True


def _normalize_tool_definition(
    tool: Any,
    *,
    allow_legacy_function: bool = False,
    identity_collector: Optional[
        MutableSequence[ResponsesFunctionIdentity]
    ] = None,
) -> tuple[Any, bool]:
    if not isinstance(tool, dict):
        return tool, False
    if tool.get("type") == "namespace":
        return _normalize_namespace_tool(
            tool,
            identity_collector=identity_collector,
        )
    return _normalize_function_tool(
        tool,
        allow_legacy_function=allow_legacy_function,
        identity_collector=identity_collector,
    )


def _normalize_tool_list(
    tools: Any,
    *,
    allow_legacy_function: bool = False,
    identity_collector: Optional[
        MutableSequence[ResponsesFunctionIdentity]
    ] = None,
) -> tuple[Any, bool]:
    if not isinstance(tools, list):
        return tools, False
    normalized_tools = list(tools)
    changed = False
    for index, tool in enumerate(tools):
        normalized_tool, tool_changed = _normalize_tool_definition(
            tool,
            allow_legacy_function=allow_legacy_function,
            identity_collector=identity_collector,
        )
        if tool_changed:
            normalized_tools[index] = normalized_tool
            changed = True
    return (normalized_tools if changed else tools), changed


def _normalize_tool_schemas_without_error_mapping(
    body: dict[str, Any],
    *,
    identity_collector: Optional[
        MutableSequence[ResponsesFunctionIdentity]
    ] = None,
) -> dict[str, Any]:
    normalized_body = body
    changed = False
    # Stock Codex Responses Lite puts the current toolset in a leading
    # developer item rather than tools[]. Materialize that declaration before
    # the existing schema, alias, adapter and restoration paths inspect it.
    input_items = body.get("input")
    if isinstance(input_items, list) and input_items:
        declaration = input_items[0]
        if (
            isinstance(declaration, dict)
            and declaration.get("type") == "additional_tools"
            and declaration.get("role") == "developer"
        ):
            tools = declaration.get("tools")
            if not isinstance(tools, list) or body.get("tools") is not None:
                raise CodexCollaborationDispatchError(
                    "unsupported_collaboration_message_schema"
                )
            normalized_body = dict(body)
            normalized_body["tools"] = tools
            normalized_body["input"] = input_items[1:]
            changed = True
    for key in ("tools", "functions"):
        normalized_tools, tools_changed = _normalize_tool_list(
            normalized_body.get(key),
            allow_legacy_function=key == "functions",
            identity_collector=identity_collector,
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
    *,
    identity_collector: Optional[
        MutableSequence[ResponsesFunctionIdentity]
    ] = None,
) -> dict[str, Any]:
    """Normalize only recognized V2 collaboration message schemas."""
    try:
        return _normalize_tool_schemas_without_error_mapping(
            body,
            identity_collector=identity_collector,
        )
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
    if author is None and recipient is None:
        return
    if (
        not isinstance(author, str)
        or not author
        or not isinstance(recipient, str)
        or not recipient
        or task_name != recipient
        or sender != author
    ):
        raise CodexCollaborationDispatchError("invalid_envelope")


def _replay_canonical_codex_message_payload(
    item: Mapping[str, Any],
    visible_text: str,
    payload: Any,
) -> Optional[str]:
    """Return normalized exact MESSAGE text only from a captured history item.

    Recovery requires actual matching author/recipient identity. Bare visible
    envelopes therefore remain rejected, while malformed captured payload from
    a real inter-agent MESSAGE can be repaired on replay without inventing
    identity or accepting an opaque representation.
    """
    author = item.get("author")
    recipient = item.get("recipient")
    if (
        not isinstance(author, str)
        or not author
        or not isinstance(recipient, str)
        or not recipient
    ):
        return None
    try:
        return canonicalize_codex_send_message_argument(payload)
    except CodexCollaborationDispatchError:
        return None


def _normalize_codex_message_payload(
    item: dict[str, Any],
    visible_part: dict[str, Any],
    visible_text: str,
    payload: Any,
) -> tuple[dict[str, Any], bool]:
    normalized_payload = _replay_canonical_codex_message_payload(
        item,
        visible_text,
        payload,
    )
    assignment = (
        parse_codex_collaboration_text_frame(normalized_payload)
        if normalized_payload is not None
        else parse_codex_collaboration_text_frame(payload)
    )
    normalized_item = _NormalizedCodexAgentMessage(item)
    normalized_item["content"] = [
        {
            "type": visible_part["type"],
            "text": f"{visible_text}{assignment}",
        }
    ]
    return normalized_item, True


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
    if message_type not in {"NEW_TASK", "MESSAGE"}:
        # A child result is content, even when its text happens to begin with
        # a representation-looking prefix. Never reinterpret it as a task.
        return None
    if not remainder:
        raise CodexCollaborationDispatchError("invalid_envelope")
    if remainder:
        assignment = _parse_codex_collaboration_payload(remainder)
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
        if message_type == "MESSAGE":
            return _normalize_codex_message_payload(
                item,
                visible_part,
                visible_text,
                payload_part.get("encrypted_content"),
            )
        if message_type != "NEW_TASK":
            return item, False
        payload = payload_part.get("encrypted_content")
        if not isinstance(payload, str) or not payload:
            raise CodexCollaborationDispatchError("invalid_envelope")
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
            text = part.get("text")
            if not isinstance(text, str) or not text.startswith("Message Type: "):
                continue
            if part.get("type") not in {"input_text", "text"}:
                raise CodexCollaborationDispatchError("invalid_envelope")
            raise CodexCollaborationDispatchError("invalid_envelope")

    if len(content) == 1 and isinstance(content[0], dict):
        visible_part = content[0]
        visible_text = visible_part.get("text")
        if isinstance(visible_text, str) and visible_text.startswith("Message Type: "):
            if visible_part.get("type") not in {"input_text", "text"}:
                raise CodexCollaborationDispatchError("invalid_envelope")
        if visible_part.get("type") in {"input_text", "text"}:
            if set(visible_part) != {"type", "text"}:
                raise CodexCollaborationDispatchError("invalid_envelope")
            if isinstance(visible_text, str) and _is_opaque_representation(
                visible_text
            ):
                raise CodexCollaborationDispatchError("opaque")
            normalized_item = _validate_visible_agent_message(item, visible_part)
            if normalized_item is not None:
                return normalized_item, True
    return item, False


def canonicalize_generated_codex_send_message_call_arguments(
    value: Any,
) -> str:
    """Transform one safely representable generated targeted call."""
    if not isinstance(value, str) or not value:
        raise CodexCollaborationDispatchError("invalid_envelope")
    try:
        decoded, remainder = _decode_complete_json(value)
    except CodexCollaborationDispatchError:
        raise
    if remainder != len(value) or not isinstance(decoded, dict):
        raise CodexCollaborationDispatchError("invalid_envelope")
    if set(decoded) != {"message", "target"}:
        raise CodexCollaborationDispatchError("unknown_representation")
    if CFG072_REJECT_MARKER in decoded:
        raise CodexCollaborationDispatchError("marker_collision")
    target = decoded.get("target")
    message = decoded.get("message")
    if not isinstance(target, str) or not target:
        raise CodexCollaborationDispatchError("invalid_envelope")
    if (
        isinstance(message, str)
        and message
        and not _is_opaque_representation(message)
        and _FRAME_PREFIX_PATTERN.match(message)
    ):
        projected = _project_generated_four_key_frame(message, target=target)
        if projected is not None:
            return json.dumps(
                {"message": projected, "target": target},
                ensure_ascii=False,
                separators=(",", ":"),
            )
    try:
        canonical_message = canonicalize_codex_send_message_argument(message)
    except CodexCollaborationDispatchError:
        rejected: dict[str, Any] = {CFG072_REJECT_MARKER: True}
        if isinstance(message, str):
            rejected["message"] = message
        if isinstance(target, str):
            rejected["target"] = target
        return json.dumps(
            rejected,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return json.dumps(
        {"message": canonical_message, "target": target},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _project_generated_four_key_frame(
    value: str,
    *,
    target: str,
) -> Optional[str]:
    """Project only a valid four-key frame whose task target matches."""
    try:
        decoded, remainder = _decode_complete_json(value)
    except CodexCollaborationDispatchError:
        return None
    if remainder != len(value) or not isinstance(decoded, dict):
        return None
    if set(decoded) != {"cfg047", "encoding", "text", "task_name"}:
        return None
    frame_version = decoded.get("cfg047")
    if (
        not isinstance(frame_version, int)
        or isinstance(frame_version, bool)
        or frame_version != COLLABORATION_FRAME_VERSION
        or decoded.get("encoding") != COLLABORATION_TEXT_ENCODING
    ):
        return None
    text = decoded.get("text")
    task_name = decoded.get("task_name")
    if (
        not isinstance(text, str)
        or not text
        or _is_opaque_representation(text)
        or not isinstance(task_name, str)
        or not task_name
        or task_name != target
    ):
        return None
    canonical = json.dumps(
        {
            "cfg047": COLLABORATION_FRAME_VERSION,
            "encoding": COLLABORATION_TEXT_ENCODING,
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(canonical) > _MAX_FRAME_CHARS:
        return None
    if parse_codex_collaboration_text_frame(canonical) != text:
        return None
    return canonical


def raise_codex_send_message_output_rejected(reason: str) -> None:
    """Reject unsafe generated output through the local wire policy path."""
    exc = CodexCollaborationDispatchError(reason)
    setattr(
        exc,
        "_aawm_policy_failure",
        {
            "failure_phase": SEND_MESSAGE_NORMALIZATION_FAILURE_PHASE,
            "policy_failure_code": CFG072_SEND_MESSAGE_OUTPUT_REJECTED_CODE,
            "policy_failure_kind": "generated_send_message_output_rejected",
        },
    )
    raise exc from None


def validate_codex_message_capture_evidence(
    evidence: Any,
    *,
    item: Mapping[str, Any],
    payload: str,
    task_name: str,
    sender: str,
) -> Optional[str]:
    """Validate trusted request-local capture evidence and return target.

    The artifact must contain the original decoded sender message string,
    the complete native targeted invocation target, and a nonempty path of
    native delivery/resolution records. Literal target equality is accepted
    only for absolute targets; every other selector requires a record ending
    at the exact recipient. Recovered messages project to the strict 3-key
    frame; strict frames are returned unchanged.
    """
    if (
        not isinstance(evidence, dict)
        or not isinstance(item, Mapping)
        or not isinstance(payload, str)
        or not payload
        or not isinstance(sender, str)
        or not sender
    ):
        return None
    if evidence.get("sender_message") != payload:
        return None
    author = item.get("author")
    recipient = item.get("recipient")
    header_task = evidence.get("header_task_name")
    if (
        not isinstance(author, str)
        or not author
        or not isinstance(recipient, str)
        or not recipient
        or evidence.get("author") != author
        or evidence.get("recipient") != recipient
        or header_task != recipient
        or task_name != recipient
    ):
        return None
    target = evidence.get("target")
    resolution = evidence.get("resolution")
    if not isinstance(target, str) or not target or not isinstance(resolution, list) or not resolution:
        return None
    for record in resolution:
        if not isinstance(record, dict):
            return None
    if not target.startswith("/"):
        if resolution[-1].get("resolved_target") != recipient:
            return None
    elif target != recipient:
        return None
    frame = {
        "cfg047": COLLABORATION_FRAME_VERSION,
        "encoding": COLLABORATION_TEXT_ENCODING,
        "text": payload,
    }
    canonical = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
    if len(canonical) > _MAX_FRAME_CHARS:
        return None
    if parse_codex_collaboration_text_frame(canonical) != payload:
        return None
    return canonical


def bind_codex_verified_capture(request: Any, evidence: Any) -> None:
    """Bind one trusted capture-evidence object to the current request."""
    state = getattr(request, "state", None)
    if state is None:
        return
    try:
        setattr(state, CODEX_CAPTURE_EVIDENCE_STATE_FIELD, evidence)
    except Exception:
        return


def get_codex_verified_capture(request: Any) -> Any:
    """Return the current request's trusted capture evidence, if bound."""
    state = getattr(request, "state", None)
    evidence = getattr(state, CODEX_CAPTURE_EVIDENCE_STATE_FIELD, None)
    return evidence if isinstance(evidence, dict) else None


_CODEX_CAPTURE_CONTEXT = contextvars.ContextVar(
    CODEX_CAPTURE_EVIDENCE_CONTEXT_FIELD,
    default=None,
)


def set_codex_verified_capture_context(evidence: Any) -> Any:
    """Bind trusted capture evidence to the current execution context."""
    return _CODEX_CAPTURE_CONTEXT.set(evidence)


def get_codex_verified_capture_from_context() -> Any:
    """Return trusted capture evidence visible to the current request."""
    return _CODEX_CAPTURE_CONTEXT.get()


def _validate_envelope_identity(
    item: Mapping[str, Any],
    *,
    task_name: str,
    sender: str,
) -> None:
    author = item.get("author")
    recipient = item.get("recipient")
    if author is None and recipient is None:
        return
    if (
        not isinstance(author, str)
        or not author
        or not isinstance(recipient, str)
        or not recipient
        or task_name != recipient
        or sender != author
    ):
        raise CodexCollaborationDispatchError("invalid_envelope")


def _replay_canonical_codex_message_payload(
    item: Mapping[str, Any],
    visible_text: str,
    payload: Any,
    *,
    capture_evidence: Any = None,
) -> Optional[str]:
    """Return a strict frame only for evidence-bound captured history."""
    author = item.get("author")
    recipient = item.get("recipient")
    if (
        not isinstance(author, str)
        or not author
        or not isinstance(recipient, str)
        or not recipient
    ):
        return None
    if not isinstance(payload, str):
        return None
    try:
        if canonicalize_codex_send_message_argument(payload) == payload:
            return payload
    except CodexCollaborationDispatchError:
        pass
    if _is_opaque_representation(payload):
        return None
    try:
        decoded, remainder = _decode_complete_json(payload)
    except CodexCollaborationDispatchError:
        return None
    if remainder != len(payload) or not isinstance(decoded, dict):
        return None
    task_name = decoded.get("task_name", recipient)
    if not isinstance(task_name, str) or not task_name:
        return None
    assignment_text = decoded.get("text")
    if not isinstance(assignment_text, str) or not assignment_text:
        return None
    return validate_codex_message_capture_evidence(
        capture_evidence,
        item=item,
        payload=assignment_text,
        task_name=task_name,
        sender=author,
    )


def _normalize_codex_message_payload(
    item: dict[str, Any],
    visible_part: dict[str, Any],
    visible_text: str,
    payload: Any,
) -> tuple[dict[str, Any], bool]:
    capture_evidence = get_codex_verified_capture_from_context()
    normalized_payload = _replay_canonical_codex_message_payload(
        item,
        visible_text,
        payload,
        capture_evidence=capture_evidence,
    )
    assignment = (
        parse_codex_collaboration_text_frame(normalized_payload)
        if normalized_payload is not None
        else parse_codex_collaboration_text_frame(payload)
    )
    normalized_item = _NormalizedCodexAgentMessage(item)
    normalized_item["content"] = [
        {
            "type": visible_part["type"],
            "text": f"{visible_text}{assignment}",
        }
    ]
    return normalized_item, True


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
    if message_type not in {"NEW_TASK", "MESSAGE"}:
        # A child result is content, even when its text happens to begin with
        # a representation-looking prefix. Never reinterpret it as a task.
        return None
    if not remainder:
        raise CodexCollaborationDispatchError("invalid_envelope")
    if remainder:
        assignment = _parse_codex_collaboration_payload(remainder)
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
        if message_type == "MESSAGE":
            return _normalize_codex_message_payload(
                item,
                visible_part,
                visible_text,
                payload_part.get("encrypted_content"),
            )
        if message_type != "NEW_TASK":
            return item, False
        payload = payload_part.get("encrypted_content")
        if not isinstance(payload, str) or not payload:
            raise CodexCollaborationDispatchError("invalid_envelope")
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
            text = part.get("text")
            if not isinstance(text, str) or not text.startswith("Message Type: "):
                continue
            if part.get("type") not in {"input_text", "text"}:
                raise CodexCollaborationDispatchError("invalid_envelope")
            raise CodexCollaborationDispatchError("invalid_envelope")

    if len(content) == 1 and isinstance(content[0], dict):
        visible_part = content[0]
        visible_text = visible_part.get("text")
        if isinstance(visible_text, str) and visible_text.startswith("Message Type: "):
            if visible_part.get("type") not in {"input_text", "text"}:
                raise CodexCollaborationDispatchError("invalid_envelope")
        if visible_part.get("type") in {"input_text", "text"}:
            if set(visible_part) != {"type", "text"}:
                raise CodexCollaborationDispatchError("invalid_envelope")
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
    *,
    identity_collector: Optional[
        MutableSequence[ResponsesFunctionIdentity]
    ] = None,
    request: Any = None,
) -> dict[str, Any]:
    """Normalize schemas and assignments, failing closed before provider send."""
    capture_evidence = get_codex_verified_capture(request) if request is not None else None
    token = set_codex_verified_capture_context(capture_evidence)
    try:
        if not isinstance(request_body, dict):
            if isinstance(request_body, Mapping):
                return dict(request_body)
            return {}
        body = _normalize_tool_schemas_without_error_mapping(
            request_body,
            identity_collector=identity_collector,
        )
        return _normalize_input_without_error_mapping(body)
    except CodexCollaborationDispatchError as exc:
        raise_codex_assignment_unreadable(reason=exc.reason)
        raise AssertionError("unreachable") from exc
    finally:
        _CODEX_CAPTURE_CONTEXT.reset(token)


def restore_codex_agent_message_payloads_for_openai_egress(
    request_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the former restore-only entry point."""
    return normalize_codex_collaboration_dispatch_body(request_body)
