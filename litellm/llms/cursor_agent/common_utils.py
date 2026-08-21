"""
Identity, hosts, headers, and request helpers for Cursor Agent CLI.

This package is `cursor_agent`. It must not reuse Cloud Agents `cursor`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

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
    explicit = (api_key or "").strip()
    auth_token = (get_secret_str(CURSOR_AUTH_TOKEN_ENV) or "").strip()
    raw_key = (get_secret_str(CURSOR_API_KEY_ENV) or "").strip()

    if explicit:
        return explicit
    if auth_token:
        return auth_token
    if raw_key and allow_exchange:
        return exchange_api_key_for_access_token(raw_key)
    if raw_key:
        return raw_key

    raise CursorAgentError(
        status_code=401,
        message=(
            "cursor_agent requires CURSOR_AUTH_TOKEN or CURSOR_API_KEY. "
            "CURSOR_CLI_KEY is not used. Cloud Agents cursor credentials "
            "are not a substitute."
        ),
    )


def exchange_api_key_for_access_token(
    raw_api_key: str,
    dashboard_base: Optional[str] = None,
) -> str:
    response = httpx.post(
        auth_exchange_url(dashboard_base),
        headers={
            "Authorization": f"Bearer {raw_api_key}",
            "Content-Type": "application/json",
        },
        json={},
        timeout=30.0,
    )
    if response.status_code >= 400:
        raise CursorAgentError(
            status_code=response.status_code,
            message="cursor_agent API key exchange failed",
            headers=dict(response.headers),
        )
    payload = response.json()
    access_token = payload.get("accessToken") if isinstance(payload, dict) else None
    if not access_token:
        raise CursorAgentError(
            status_code=401,
            message="cursor_agent API key exchange did not return accessToken",
        )
    return str(access_token)


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
        text = message_text(message)
        if text:
            return text
    if messages:
        return message_text(messages[-1])
    return ""


def _extract_system_prompt(messages: List[AllMessageValues]) -> str:
    parts: List[str] = []
    for message in messages:
        if message_role(message) != "system":
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
        content.append({"tool_call": tool_call})
    return {"assistant": {"content": content}}


def _history_tool_message(
    *,
    tool_call_id: str,
    tool_name: str,
    text: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "content": [_history_text_content(text)],
    }
    if tool_name:
        payload["tool_name"] = tool_name
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
        "tool_call_id": tool_call_id,
        "tool_name": name,
        "args_json": args_json,
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
) -> Optional[Dict[str, Any]]:
    history_messages: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message_role(message)
        if role == "system":
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
        if message_role(message) == "user" and message_text(message):
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
        "provider_identifier": provider_identifier,
        "tool_name": name,
    }
    description = _openai_tool_description(tool)
    if description:
        definition["description"] = description
    parameters = _openai_tool_parameters(tool)
    if parameters is not None:
        if isinstance(parameters, str):
            definition["input_schema_json"] = parameters
        else:
            definition["input_schema_json"] = json.dumps(parameters)
    return definition


def _build_mcp_tools(optional_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    existing = optional_params.get("mcp_tools")
    if isinstance(existing, dict) and existing:
        return existing
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
    return {"mcp_tools": definitions}


def _merge_conversation_state(
    optional_params: Dict[str, Any],
    history: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    existing = optional_params.get("conversation_state")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    if history is None:
        return {}
    # AgentRunRequest.conversation_state is ConversationState. History JSON is
    # a verified ConversationState.root_prompt_messages_json scalar so earlier
    # OpenAI turns are not reduced to the last user message.
    return {"root_prompt_messages_json": [json.dumps(history)]}


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
                "custom_system_prompt / conversation_state, not OpenAI "
                "temperature / max_tokens / tool_choice."
            ),
        )
    model_id = strip_provider_prefix(model)
    last_user_index = _last_user_index(messages)
    prompt = extract_user_text(messages)
    history = _build_conversation_history(
        messages, last_user_index=last_user_index
    )
    conversation_id = optional_params.get("conversation_id")
    run_id = optional_params.get("run_id") or str(uuid.uuid4())
    agent_session_id = optional_params.get("agent_session_id")
    custom_system_prompt = optional_params.get("custom_system_prompt")
    if not custom_system_prompt:
        custom_system_prompt = _extract_system_prompt(messages)

    user_message_action: Dict[str, Any] = {
        "user_message": {
            "text": prompt,
        }
    }
    if history is not None:
        # UserMessageAction.conversation_history is the verified field for
        # prior OpenAI turns / tool results on the same Run.
        user_message_action["conversation_history"] = history

    request: Dict[str, Any] = {
        "conversation_state": _merge_conversation_state(optional_params, history),
        "action": {
            "user_message_action": user_message_action
        },
        "model_details": {
            "model_id": model_id,
        },
        "requested_model": {
            "model_id": model_id,
        },
        "run_id": run_id,
    }
    if custom_system_prompt:
        request["custom_system_prompt"] = custom_system_prompt
    mcp_tools = _build_mcp_tools(optional_params)
    if mcp_tools is not None:
        request["mcp_tools"] = mcp_tools
    if conversation_id:
        request["conversation_id"] = conversation_id
    if agent_session_id:
        request["agent_session_id"] = agent_session_id
    return {
        "run_request": request,
    }


def _interaction_text(update: Dict[str, Any]) -> Tuple[str, bool]:
    if not isinstance(update, dict):
        return "", False
    if "text_delta" in update and isinstance(update["text_delta"], dict):
        return str(update["text_delta"].get("text") or ""), False
    if "turn_ended" in update:
        return "", True
    message = update.get("message")
    if isinstance(message, dict):
        case = message.get("case")
        value = message.get("value") if isinstance(message.get("value"), dict) else {}
        if case == "text_delta":
            return str(value.get("text") or ""), False
        if case == "turn_ended":
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

    if "interaction_update" in payload:
        return _interaction_text(payload["interaction_update"])

    message = payload.get("message")
    if isinstance(message, dict):
        case = message.get("case")
        value = message.get("value") if isinstance(message.get("value"), dict) else {}
        if case == "interaction_update":
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
