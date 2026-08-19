"""Stdlib Cursor Agent usage-poll helpers for the provider-status sidecar.

This module must stay importable without httpx, secret_managers.main, or the
rest of the LiteLLM package. The slim sidecar image copies this file and
``usage.py`` plus empty ``llms`` / ``cursor_agent`` inits; it does not copy
``common_utils.py``.
"""

from __future__ import annotations

import os
import platform
import uuid
from typing import Any, Dict, Optional

CURSOR_AGENT_PROVIDER = "cursor_agent"
CURSOR_AGENT_DASHBOARD_HOST = "https://api2.cursor.sh"
CURSOR_AGENT_USAGE_PATH = (
    "/aiserver.v1.DashboardService/GetCurrentPeriodUsage"
)
CURSOR_AGENT_CLIENT_VERSION = "2026.08.11-e8db854"
CURSOR_API_KEY_ENV = "CURSOR_API_KEY"
CURSOR_AUTH_TOKEN_ENV = "CURSOR_AUTH_TOKEN"
CURSOR_CLI_KEY_ENV = "CURSOR_CLI_KEY"


class CursorAgentUsageAuthError(Exception):
    """Raised when sidecar usage polling has no Cursor access token."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def cursor_agent_user_agent() -> str:
    system = platform.system().lower() or "linux"
    machine = platform.machine().lower() or "x64"
    if machine in {"x86_64", "amd64"}:
        machine = "x64"
    elif machine in {"aarch64", "arm64"}:
        machine = "arm64"
    return f"Cursor-CLI/{CURSOR_AGENT_CLIENT_VERSION} ({system} {machine})"


def resolve_dashboard_api_base(api_base: Optional[str] = None) -> str:
    return (api_base or CURSOR_AGENT_DASHBOARD_HOST).rstrip("/")


def current_period_usage_url(dashboard_base: Optional[str] = None) -> str:
    """Dashboard unary Connect path. Not Cloud Agents GET /v0/me."""
    return f"{resolve_dashboard_api_base(dashboard_base)}{CURSOR_AGENT_USAGE_PATH}"


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


def build_dashboard_headers(
    access_token: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    *,
    request_id: Optional[str] = None,
) -> Dict[str, str]:
    """Headers for DashboardService unary Connect JSON RPCs."""
    return build_turn_headers(
        access_token,
        extra_headers=extra_headers,
        request_id=request_id,
        http2=True,
    )


def resolve_access_token(
    api_key: Optional[str] = None,
    *,
    allow_exchange: bool = False,
) -> str:
    """
    Return the Bearer access token for Dashboard usage polling.

    Preference: explicit api_key argument, then CURSOR_AUTH_TOKEN, then
    CURSOR_API_KEY as a raw Bearer. CURSOR_CLI_KEY is ignored. This helper
    never network-exchanges, even when allow_exchange=True.
    """
    _ = allow_exchange
    explicit = (api_key or "").strip()
    auth_token = (os.getenv(CURSOR_AUTH_TOKEN_ENV) or "").strip()
    raw_key = (os.getenv(CURSOR_API_KEY_ENV) or "").strip()

    if explicit:
        return explicit
    if auth_token:
        return auth_token
    if raw_key:
        return raw_key

    raise CursorAgentUsageAuthError(
        "cursor_agent requires CURSOR_AUTH_TOKEN or CURSOR_API_KEY. "
        "CURSOR_CLI_KEY is not used. Cloud Agents cursor credentials "
        "are not a substitute."
    )
