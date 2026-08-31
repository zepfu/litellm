"""
Identity, hosts, headers, and request helpers for Cursor Agent CLI.

This package is `cursor_agent`. It must not reuse Cloud Agents `cursor`.
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from .constants import (
    CLOUD_AGENTS_PROVIDER,
    CURSOR_AGENT_AUTH_EXCHANGE_PATH,
    CURSOR_AGENT_CLIENT_VERSION,
    CURSOR_AGENT_DASHBOARD_HOST,
    CURSOR_AGENT_PROVIDER,
    CURSOR_AGENT_REQUESTED_MODEL_OVERRIDES,
    CURSOR_AGENT_RUN_PATH,
    CURSOR_AGENT_TURN_HOST,
    CURSOR_AGENT_USAGE_PATH,
    CURSOR_API_KEY_ENV,
    CURSOR_AUTH_TOKEN_ENV,
    CURSOR_CLI_KEY_ENV,
)
from .dashboard import (
    build_dashboard_headers,
    build_turn_headers,
    cursor_agent_user_agent,
    current_period_usage_url,
    resolve_dashboard_api_base,
)
from .connect import (
    CursorConnectError,
    resolve_cursor_access_token_sync,
)

__all__ = [
    "CLOUD_AGENTS_PROVIDER",
    "CURSOR_AGENT_AUTH_EXCHANGE_PATH",
    "CURSOR_AGENT_CLIENT_VERSION",
    "CURSOR_AGENT_DASHBOARD_HOST",
    "CURSOR_AGENT_PROVIDER",
    "CURSOR_AGENT_RUN_PATH",
    "CURSOR_AGENT_TURN_HOST",
    "CURSOR_AGENT_USAGE_PATH",
    "CURSOR_API_KEY_ENV",
    "CURSOR_AUTH_TOKEN_ENV",
    "CURSOR_CLI_KEY_ENV",
    "CursorAgentError",
    "auth_exchange_url",
    "build_dashboard_headers",
    "build_run_request",
    "build_turn_headers",
    "cursor_agent_user_agent",
    "current_period_usage_url",
    "exchange_api_key_for_access_token",
    "extract_text_from_agent_payload",
    "extract_user_text",
    "message_role",
    "message_text",
    "resolve_access_token",
    "resolve_dashboard_api_base",
    "resolve_provider_info",
    "resolve_turn_api_base",
    "run_url",
    "strip_provider_prefix",
]


class CursorAgentError(BaseLLMException):
    """Provider error for the Cursor Agent CLI Connect route."""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers or {},
        )


def strip_provider_prefix(model: str) -> str:
    if model.startswith(f"{CURSOR_AGENT_PROVIDER}/"):
        return model.split("/", 1)[1]
    return model


def resolve_turn_api_base(api_base: Optional[str]) -> str:
    return (api_base or CURSOR_AGENT_TURN_HOST).rstrip("/")


def run_url(api_base: Optional[str] = None) -> str:
    base = resolve_turn_api_base(api_base)
    if base.endswith(CURSOR_AGENT_RUN_PATH):
        return base
    return f"{base}{CURSOR_AGENT_RUN_PATH}"


def auth_exchange_url(dashboard_base: Optional[str] = None) -> str:
    return (
        f"{resolve_dashboard_api_base(dashboard_base)}{CURSOR_AGENT_AUTH_EXCHANGE_PATH}"
    )


def resolve_provider_info(
    api_base: Optional[str], api_key: Optional[str]
) -> Tuple[str, Optional[str]]:
    """
    Resolve the HTTP/2 turn host and an optional access token.

    Missing credentials stay None so `get_llm_provider` can still resolve the
    `cursor_agent/<slug>` prefix. Request time still requires a token.
    """
    resolved_key = (api_key or "").strip() or None
    if resolved_key is None:
        resolved_key = (get_secret_str(CURSOR_AUTH_TOKEN_ENV) or "").strip() or None
    if resolved_key is None:
        resolved_key = (get_secret_str(CURSOR_API_KEY_ENV) or "").strip() or None
    return resolve_turn_api_base(api_base), resolved_key


def resolve_access_token(
    api_key: Optional[str] = None,
    *,
    allow_exchange: bool = False,
) -> str:
    """
    Return the Bearer access token for Agent CLI turns.

    Preference: explicit api_key argument, then CURSOR_AUTH_TOKEN, then
    CURSOR_API_KEY. CURSOR_CLI_KEY is ignored. Cloud Agents Basic-auth
    lookup is not used.

    CURSOR_API_KEY is a raw key. Exchange is opt-in because it is a
    network call; callers that already have an access token must pass
    it as api_key or CURSOR_AUTH_TOKEN.
    """
    try:
        return resolve_cursor_access_token_sync(
            api_key,
            allow_exchange=allow_exchange,
        )
    except CursorConnectError as exc:
        raise CursorAgentError(
            status_code=exc.status_code,
            message=exc.message,
            headers=exc.headers,
        ) from exc


def exchange_api_key_for_access_token(
    raw_api_key: str,
    dashboard_base: Optional[str] = None,
) -> str:
    from .connect import exchange_cursor_api_key_for_access_token_sync

    try:
        return exchange_cursor_api_key_for_access_token_sync(
            raw_api_key,
            dashboard_base=dashboard_base,
        )
    except CursorConnectError as exc:
        raise CursorAgentError(
            status_code=exc.status_code,
            message=exc.message,
            headers=exc.headers,
        ) from exc


_DEFAULT_MCP_PROVIDER_IDENTIFIER = "litellm"


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def message_text(message: Any) -> str:
    if isinstance(message, dict):
        return convert_content_list_to_str(message=message)
    mapping = _as_mapping(message)
    if mapping:
        return convert_content_list_to_str(message=mapping)  # type: ignore[arg-type]
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return ""


def extract_user_text(messages: List[AllMessageValues]) -> str:
    for message in reversed(messages):
        if message_role(message) != "user":
            continue
        return message_text(message)
    if messages:
        return message_text(messages[-1])
    return ""


def _extract_system_prompt(messages: List[AllMessageValues]) -> str:
    parts: List[str] = []
    for message in messages:
        if message_role(message) not in {"system", "developer"}:
            continue
        text = message_text(message)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _history_text_content(text: str) -> Dict[str, Any]:
    return {"text": {"text": text}}


def _history_user_message(text: str) -> Dict[str, Any]:
    return {"user": {"content": [_history_text_content(text)]}}


def _history_assistant_message(
    text: str, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    if text:
        content.append(_history_text_content(text))
    for tool_call in tool_calls or []:
        content.append({"toolCall": tool_call})
    return {"assistant": {"content": content}}


def _history_tool_message(
    *,
    tool_call_id: str,
    tool_name: str,
    text: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "toolCallId": tool_call_id,
        "content": [_history_text_content(text)],
    }
    if tool_name:
        payload["toolName"] = tool_name
    return {"tool": payload}


def _openai_tool_call_to_history(tool_call: Any) -> Optional[Dict[str, Any]]:
    mapping = _as_mapping(tool_call)
    function = _as_mapping(mapping.get("function"))
    name = str(function.get("name") or mapping.get("name") or "")
    tool_call_id = str(mapping.get("id") or mapping.get("tool_call_id") or "")
    arguments = function.get("arguments")
    if arguments is None:
        arguments = mapping.get("arguments")
    if isinstance(arguments, (dict, list)):
        args_json = json.dumps(arguments)
    else:
        args_json = str(arguments or "")
    if not name and not tool_call_id and not args_json:
        return None
    history_call: Dict[str, Any] = {
        "toolCallId": tool_call_id,
        "toolName": name,
        "argsJson": args_json,
    }
    return history_call


def _tool_calls_from_message(message: Any) -> List[Any]:
    mapping = message if isinstance(message, dict) else _as_mapping(message)
    tool_calls = mapping.get("tool_calls")
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _tool_name_for_call_id(
    messages: List[AllMessageValues], tool_call_id: str
) -> str:
    if not tool_call_id:
        return ""
    for message in messages:
        for tool_call in _tool_calls_from_message(message):
            mapping = _as_mapping(tool_call)
            function = _as_mapping(mapping.get("function"))
            call_id = str(mapping.get("id") or mapping.get("tool_call_id") or "")
            if call_id == tool_call_id:
                return str(function.get("name") or mapping.get("name") or "")
    return ""


def _build_conversation_history(
    messages: List[AllMessageValues],
    *,
    last_user_index: Optional[int],
    instruction_text: str = "",
) -> Optional[Dict[str, Any]]:
    history_messages: List[Dict[str, Any]] = []
    if instruction_text:
        # Cursor's public AgentService path rejects customSystemPrompt through
        # its internal --system-prompt option. ConversationHistory has no
        # system role, so preserve the guidance as a leading user-level entry.
        history_messages.append(
            _history_user_message(
                "System and developer instructions for this run:\n\n"
                f"{instruction_text}"
            )
        )
    for index, message in enumerate(messages):
        role = message_role(message)
        if role in {"system", "developer"}:
            continue
        if index == last_user_index and role == "user":
            continue
        text = message_text(message)
        mapping = message if isinstance(message, dict) else _as_mapping(message)
        if role == "user":
            if text:
                history_messages.append(_history_user_message(text))
            continue
        if role == "assistant":
            tool_calls = []
            for tool_call in _tool_calls_from_message(message):
                converted = _openai_tool_call_to_history(tool_call)
                if converted is not None:
                    tool_calls.append(converted)
            if text or tool_calls:
                history_messages.append(
                    _history_assistant_message(text, tool_calls or None)
                )
            continue
        if role == "tool":
            tool_call_id = str(
                mapping.get("tool_call_id") or mapping.get("id") or ""
            )
            tool_name = str(
                mapping.get("name")
                or mapping.get("tool_name")
                or _tool_name_for_call_id(messages, tool_call_id)
            )
            history_messages.append(
                _history_tool_message(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    text=text,
                )
            )
    if not history_messages:
        return None
    return {"messages": history_messages}


def _last_user_index(messages: List[AllMessageValues]) -> Optional[int]:
    last_index: Optional[int] = None
    for index, message in enumerate(messages):
        if message_role(message) == "user":
            last_index = index
    return last_index


def _openai_tool_name(tool: Any) -> str:
    mapping = _as_mapping(tool)
    function = _as_mapping(mapping.get("function"))
    return str(function.get("name") or mapping.get("name") or "")


def _openai_tool_description(tool: Any) -> str:
    mapping = _as_mapping(tool)
    function = _as_mapping(mapping.get("function"))
    return str(function.get("description") or mapping.get("description") or "")


def _openai_tool_parameters(tool: Any) -> Any:
    mapping = _as_mapping(tool)
    function = _as_mapping(mapping.get("function"))
    if "parameters" in function:
        return function.get("parameters")
    return mapping.get("parameters") or mapping.get("input_schema")


def _tool_provider_identifier(tool: Any) -> str:
    mapping = _as_mapping(tool)
    return str(
        mapping.get("provider_identifier")
        or mapping.get("server_identifier")
        or mapping.get("mcp_server")
        or _DEFAULT_MCP_PROVIDER_IDENTIFIER
    )


def _build_mcp_tool_definition(tool: Any) -> Optional[Dict[str, Any]]:
    name = _openai_tool_name(tool)
    if not name:
        return None
    provider_identifier = _tool_provider_identifier(tool)
    definition: Dict[str, Any] = {
        "name": name,
        "providerIdentifier": provider_identifier,
        "toolName": name,
    }
    description = _openai_tool_description(tool)
    if description:
        definition["description"] = description
    parameters = _openai_tool_parameters(tool)
    if parameters is not None:
        if isinstance(parameters, str):
            definition["inputSchemaJson"] = parameters
        else:
            definition["inputSchemaJson"] = json.dumps(parameters)
    return definition


def _build_mcp_tools(optional_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    existing = optional_params.get("mcp_tools")
    if isinstance(existing, dict) and existing:
        return _camelize_proto_mapping(existing)
    tools = optional_params.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    definitions: List[Dict[str, Any]] = []
    for tool in tools:
        definition = _build_mcp_tool_definition(tool)
        if definition is not None:
            definitions.append(definition)
    if not definitions:
        return None
    return {"mcpTools": definitions}


def _merge_conversation_state(
    optional_params: Dict[str, Any],
) -> Dict[str, Any]:
    existing = optional_params.get("conversation_state")
    if isinstance(existing, dict) and existing:
        return _camelize_proto_mapping(existing)
    return {}


def _build_requested_model(model_id: str) -> Dict[str, Any]:
    configured = CURSOR_AGENT_REQUESTED_MODEL_OVERRIDES.get(model_id)
    if configured is not None:
        return copy.deepcopy(configured)
    return {"modelId": model_id}


_PROTO_FIELD_NAMES = {
    "conversation_state": "conversationState",
    "root_prompt_messages_json": "rootPromptMessagesJson",
    "user_message_action": "userMessageAction",
    "user_message": "userMessage",
    "conversation_history": "conversationHistory",
    "model_details": "modelDetails",
    "model_id": "modelId",
    "requested_model": "requestedModel",
    "mcp_tools": "mcpTools",
    "provider_identifier": "providerIdentifier",
    "tool_name": "toolName",
    "input_schema_json": "inputSchemaJson",
    "tool_call_id": "toolCallId",
    "args_json": "argsJson",
    "message_id": "messageId",
    "conversation_id": "conversationId",
    "conversation_group_id": "conversationGroupId",
    "run_id": "runId",
    "agent_session_id": "agentSessionId",
}


def _camelize_proto_key(key: Any) -> str:
    text = str(key)
    mapped = _PROTO_FIELD_NAMES.get(text)
    if mapped is not None:
        return mapped
    parts = text.split("_")
    if len(parts) == 1:
        return text
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _camelize_proto_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _camelize_proto_key(key): _camelize_proto_mapping(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_camelize_proto_mapping(item) for item in value]
    return value


def build_run_request(
    model: str,
    messages: List[AllMessageValues],
    optional_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    optional_params = optional_params or {}
    unsupported = [
        name
        for name in ("temperature", "max_tokens", "tool_choice")
        if name in optional_params
    ]
    if unsupported:
        raise CursorAgentError(
            status_code=400,
            message=(
                "cursor_agent does not support parameters: "
                f"{unsupported}. AgentRunRequest has mcp_tools / "
                "conversation_history / conversation_state, not OpenAI "
                "temperature / max_tokens / tool_choice."
            ),
        )
    model_id = strip_provider_prefix(model)
    last_user_index = _last_user_index(messages)
    prompt = extract_user_text(messages)
    custom_system_prompt = (
        optional_params.get("custom_system_prompt")
        or optional_params.get("customSystemPrompt")
        or _extract_system_prompt(messages)
    )
    history = _build_conversation_history(
        messages,
        last_user_index=last_user_index,
        instruction_text=str(custom_system_prompt or ""),
    )
    message_id = (
        optional_params.get("message_id")
        or optional_params.get("messageId")
        or str(uuid.uuid4())
    )
    conversation_id = (
        optional_params.get("conversation_id")
        or optional_params.get("conversationId")
        or str(uuid.uuid4())
    )
    conversation_group_id = (
        optional_params.get("conversation_group_id")
        or optional_params.get("conversationGroupId")
        or conversation_id
    )
    run_id = (
        optional_params.get("run_id")
        or optional_params.get("runId")
        or str(uuid.uuid4())
    )
    agent_session_id = optional_params.get("agent_session_id")

    request: Dict[str, Any] = {
        "conversationState": _merge_conversation_state(optional_params),
        "action": {
            "userMessageAction": {
                "userMessage": {
                    "text": prompt,
                    "messageId": message_id,
                    "selectedContext": {},
                    "mode": "AGENT_MODE_AGENT",
                },
                "requestContext": {},
            }
        },
        "requestedModel": _build_requested_model(model_id),
        "mcpTools": _build_mcp_tools(optional_params) or {},
        "conversationId": conversation_id,
        "conversationGroupId": conversation_group_id,
        "runId": run_id,
    }
    if history is not None:
        request["action"]["userMessageAction"]["conversationHistory"] = history
    if agent_session_id:
        request["agentSessionId"] = agent_session_id
    return {
        "runRequest": request,
    }


def _interaction_text(update: Dict[str, Any]) -> Tuple[str, bool]:
    if not isinstance(update, dict):
        return "", False
    if "textDelta" in update and isinstance(update["textDelta"], dict):
        return str(update["textDelta"].get("text") or ""), False
    if "text_delta" in update and isinstance(update["text_delta"], dict):
        return str(update["text_delta"].get("text") or ""), False
    if "turnEnded" in update or "turn_ended" in update:
        return "", True
    message = update.get("message")
    if isinstance(message, dict):
        case = message.get("case")
        value = message.get("value") if isinstance(message.get("value"), dict) else {}
        if case in {"textDelta", "text_delta"}:
            return str(value.get("text") or ""), False
        if case in {"turnEnded", "turn_ended"}:
            return "", True
    return "", False


def extract_text_from_agent_payload(payload: Any) -> Tuple[str, bool]:
    """
    Return (text, turn_ended) from an AgentServerMessage-shaped payload.
    Accepts proto-JSON field names from the CLI bundle.
    """
    if payload is None:
        return "", False
    if isinstance(payload, str):
        return payload, False
    if not isinstance(payload, dict):
        return "", False

    if "interactionUpdate" in payload:
        return _interaction_text(payload["interactionUpdate"])
    if "interaction_update" in payload:
        return _interaction_text(payload["interaction_update"])

    message = payload.get("message")
    if isinstance(message, dict):
        case = message.get("case")
        value = message.get("value") if isinstance(message.get("value"), dict) else {}
        if case in {"interactionUpdate", "interaction_update"}:
            return _interaction_text(value)
        text, ended = _interaction_text(message)
        if text or ended:
            return text, ended

    text, ended = _interaction_text(payload)
    if text or ended:
        return text, ended

    if "text" in payload and isinstance(payload["text"], str):
        return payload["text"], bool(payload.get("turn_ended"))
    return "", False
