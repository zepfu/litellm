"""
Identity, hosts, headers, and request helpers for Cursor Agent CLI.

This package is `cursor_agent`. It must not reuse Cloud Agents `cursor`.
"""

from __future__ import annotations

import platform
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

CURSOR_AGENT_PROVIDER = "cursor_agent"
CLOUD_AGENTS_PROVIDER = "cursor"
CURSOR_AGENT_TURN_HOST = "https://agentn.global.api5.cursor.sh"
CURSOR_AGENT_DASHBOARD_HOST = "https://api2.cursor.sh"
CURSOR_AGENT_RUN_PATH = "/agent.v1.AgentService/Run"
CURSOR_AGENT_AUTH_EXCHANGE_PATH = "/auth/exchange_user_api_key"
CURSOR_AGENT_CLIENT_VERSION = "2026.08.11-e8db854"
CURSOR_API_KEY_ENV = "CURSOR_API_KEY"
CURSOR_AUTH_TOKEN_ENV = "CURSOR_AUTH_TOKEN"
CURSOR_CLI_KEY_ENV = "CURSOR_CLI_KEY"


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


def cursor_agent_user_agent() -> str:
    system = platform.system().lower() or "linux"
    machine = platform.machine().lower() or "x64"
    if machine in {"x86_64", "amd64"}:
        machine = "x64"
    elif machine in {"aarch64", "arm64"}:
        machine = "arm64"
    return f"Cursor-CLI/{CURSOR_AGENT_CLIENT_VERSION} ({system} {machine})"


def strip_provider_prefix(model: str) -> str:
    if model.startswith(f"{CURSOR_AGENT_PROVIDER}/"):
        return model.split("/", 1)[1]
    return model


def resolve_turn_api_base(api_base: Optional[str]) -> str:
    return (api_base or CURSOR_AGENT_TURN_HOST).rstrip("/")


def resolve_dashboard_api_base(api_base: Optional[str] = None) -> str:
    return (api_base or CURSOR_AGENT_DASHBOARD_HOST).rstrip("/")


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


def build_turn_headers(
    access_token: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    *,
    request_id: Optional[str] = None,
    http2: bool = True,
) -> Dict[str, str]:
    """
    Build HTTP/2 AgentService/Run headers.

    `x-cursor-streaming` is an HTTP/1.1 RunSSE interceptor header and is never
    set here. Checksum is not part of the Agent CLI turn.
    """
    _ = http2
    headers: Dict[str, str] = {
        "authorization": f"Bearer {access_token}",
        "user-agent": cursor_agent_user_agent(),
        "x-cursor-client-version": f"cli-{CURSOR_AGENT_CLIENT_VERSION}",
        "x-cursor-client-type": "cli",
        "x-ghost-mode": "true",
        "x-request-id": request_id or str(uuid.uuid4()),
        "connect-protocol-version": "1",
        "content-type": "application/json",
    }
    if extra_headers:
        for key, value in extra_headers.items():
            if value is None:
                continue
            lowered = key.lower()
            if lowered in {
                "authorization",
                "x-cursor-streaming",
                "x-cursor-checksum",
            }:
                continue
            headers[lowered] = str(value)
    return headers


def extract_user_text(messages: List[AllMessageValues]) -> str:
    for message in reversed(messages):
        role = ""
        if isinstance(message, dict):
            role = str(message.get("role") or "")
        else:
            role = str(getattr(message, "role", "") or "")
        if role != "user":
            continue
        text = convert_content_list_to_str(message=message)
        if text:
            return text
    if messages:
        return convert_content_list_to_str(message=messages[-1])
    return ""


def build_run_request(
    model: str,
    messages: List[AllMessageValues],
    optional_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    optional_params = optional_params or {}
    model_id = strip_provider_prefix(model)
    prompt = extract_user_text(messages)
    conversation_id = optional_params.get("conversation_id")
    run_id = optional_params.get("run_id") or str(uuid.uuid4())
    agent_session_id = optional_params.get("agent_session_id")
    request: Dict[str, Any] = {
        "conversation_state": optional_params.get("conversation_state") or {},
        "action": {
            "user_message_action": {
                "user_message": {
                    "text": prompt,
                }
            }
        },
        "model_details": {
            "model_id": model_id,
        },
        "requested_model": {
            "model_id": model_id,
        },
        "run_id": run_id,
    }
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
