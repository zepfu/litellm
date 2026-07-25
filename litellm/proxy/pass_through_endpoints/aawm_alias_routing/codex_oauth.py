"""Codex auth-file discovery, JWT decode, token validation, and request detection.

Wave 5A extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Runtime dependency ``_get_request_header_or_passthrough_alias`` is injected via
:func:`configure_codex_oauth_runtime` (the function lives in the god module and
depends on the pass-through header prefix constant).
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from fastapi import HTTPException, Request

from litellm.llms.chatgpt.common_utils import get_chatgpt_default_headers
from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS = (
    "LITELLM_CODEX_AUTH_FILE",
    "CHATGPT_AUTH_FILE",
)
_ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS = (
    "LITELLM_CODEX_TOKEN_DIR",
    "CHATGPT_TOKEN_DIR",
)
_ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS = (
    "~/.codex/auth.json",
    "~/.codex/auth.json",
    "~/.config/litellm/chatgpt/auth.json",
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
CodexAuthData = dict[str, object]
CodexTokenData = dict[str, object]
OAuthJsonData = dict[str, object]

# ---------------------------------------------------------------------------
# Injected runtime dependencies
# ---------------------------------------------------------------------------
_get_request_header_or_passthrough_alias: Optional[
    Callable[[Request, str], Optional[str]]
] = None


def configure_codex_oauth_runtime(
    *,
    get_request_header_or_passthrough_alias: Callable[[Request, str], Optional[str]],
) -> None:
    """Bind god-module request-header helpers after module load."""
    global _get_request_header_or_passthrough_alias
    _get_request_header_or_passthrough_alias = get_request_header_or_passthrough_alias


# ---------------------------------------------------------------------------
# Auth-value cleaning
# ---------------------------------------------------------------------------


def _clean_codex_auth_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


# ---------------------------------------------------------------------------
# Auth-file discovery
# ---------------------------------------------------------------------------


def _get_anthropic_adapter_codex_auth_file_path() -> Optional[Path]:
    for env_name in _ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS:
        raw_value = _clean_codex_auth_value(os.getenv(env_name))
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        if path.exists():
            return path

    token_dir: Optional[Path] = None
    for env_name in _ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS:
        raw_value = _clean_codex_auth_value(os.getenv(env_name))
        if not raw_value:
            continue
        candidate = Path(raw_value).expanduser()
        if candidate.exists():
            token_dir = candidate
            break
    if token_dir is not None:
        candidate = token_dir / "auth.json"
        if candidate.exists():
            return candidate

    for candidate_str in _ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if candidate.exists():
            return candidate

    return None


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _decode_jwt_claims_without_validation(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
    except Exception:
        return {}


def _extract_codex_account_id_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    claims = _decode_jwt_claims_without_validation(token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


# ---------------------------------------------------------------------------
# Token data / validation
# ---------------------------------------------------------------------------


def _get_codex_auth_token_data(auth_data: CodexAuthData) -> CodexTokenData:
    token_data = auth_data.get("tokens")
    if isinstance(token_data, dict):
        return dict(token_data)
    return auth_data


def _get_codex_auth_token_expiry(access_token: str) -> Optional[int]:
    claims = _decode_jwt_claims_without_validation(access_token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        return int(exp)
    return None


def _codex_auth_access_token_is_valid(token_data: CodexTokenData) -> bool:
    access_token = _clean_codex_auth_value(token_data.get("access_token"))
    if access_token is None:
        return False
    expires_at = token_data.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        expires_at = _get_codex_auth_token_expiry(access_token)
    if not isinstance(expires_at, (int, float)):
        return True
    return time.time() < float(expires_at) - 60


# ---------------------------------------------------------------------------
# Auth data loading
# ---------------------------------------------------------------------------


async def _load_codex_auth_data_from_path(auth_path: Path) -> Optional[CodexAuthData]:
    try:
        auth_data = json.loads(auth_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(auth_data, dict):
        return None
    return auth_data


async def _load_local_codex_auth_headers(request: Request) -> Optional[dict[str, str]]:
    auth_path = _get_anthropic_adapter_codex_auth_file_path()
    if auth_path is None:
        return None

    auth_data = await _load_codex_auth_data_from_path(auth_path)
    if auth_data is None:
        return None

    token_data = _get_codex_auth_token_data(auth_data)
    access_token = _clean_codex_auth_value(token_data.get("access_token"))
    if access_token is None:
        return None
    if not _codex_auth_access_token_is_valid(token_data):
        raise HTTPException(
            status_code=500,
            detail=(
                "Codex OAuth access token is expired or invalid. The "
                "provider-status sidecar owns Codex auth refresh; confirm the "
                "sidecar can write the configured auth file and refresh "
                f"{auth_path}."
            ),
        )

    account_id = _clean_codex_auth_value(token_data.get("account_id")) or _extract_codex_account_id_from_token(
        _clean_codex_auth_value(token_data.get("id_token")) or access_token
    )

    headers = _safe_get_request_headers(request)
    assert _get_request_header_or_passthrough_alias is not None
    session_id = (
        _get_request_header_or_passthrough_alias(request, "session_id")
        or headers.get("x-claude-code-session-id")
        or headers.get("X-Claude-Code-Session-Id")
    )

    return get_chatgpt_default_headers(
        access_token=access_token,
        account_id=account_id,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Codex-native-auth request detection
# ---------------------------------------------------------------------------


def _anthropic_adapter_request_uses_codex_native_auth(request: Request) -> bool:
    assert _get_request_header_or_passthrough_alias is not None
    chatgpt_account_id = _get_request_header_or_passthrough_alias(request, "ChatGPT-Account-Id")
    originator = _get_request_header_or_passthrough_alias(request, "originator")
    user_agent = _get_request_header_or_passthrough_alias(request, "user-agent")
    session_id = _get_request_header_or_passthrough_alias(request, "session_id")

    if isinstance(chatgpt_account_id, str) and len(chatgpt_account_id) > 0:
        return True
    if isinstance(originator, str) and "codex" in originator.lower():
        return True
    return bool(
        isinstance(user_agent, str)
        and "codex" in user_agent.lower()
        and isinstance(session_id, str)
        and len(session_id) > 0
    )


def _anthropic_adapter_request_has_openai_client_auth(request: Request) -> bool:
    # On the Anthropic route, direct Authorization headers are typically Anthropic auth
    # from Claude clients, not OpenAI/Codex credentials. Treat direct auth as OpenAI
    # client auth only when the request also carries Codex-native request markers.
    assert _get_request_header_or_passthrough_alias is not None
    if _get_request_header_or_passthrough_alias(
        request, "x-pass-authorization"
    ) or _get_request_header_or_passthrough_alias(request, "x-pass-api-key"):
        return True

    if _anthropic_adapter_request_uses_codex_native_auth(request):
        return bool(
            _get_request_header_or_passthrough_alias(request, "authorization")
            or _get_request_header_or_passthrough_alias(request, "api-key")
        )

    return False


def _anthropic_adapter_should_forward_direct_auth_headers(request: Request) -> bool:
    return _anthropic_adapter_request_has_openai_client_auth(request)


def _request_uses_codex_native_auth(request: Request) -> bool:
    headers = _safe_get_request_headers(request)
    chatgpt_account_id = headers.get("chatgpt-account-id") or headers.get("ChatGPT-Account-Id")
    originator = headers.get("originator") or headers.get("Originator")
    user_agent = headers.get("user-agent") or headers.get("User-Agent")
    session_id = headers.get("session_id") or headers.get("Session_Id")

    if isinstance(chatgpt_account_id, str) and len(chatgpt_account_id) > 0:
        return True
    if isinstance(originator, str) and "codex" in originator.lower():
        return True
    return bool(
        isinstance(user_agent, str)
        and "codex" in user_agent.lower()
        and isinstance(session_id, str)
        and len(session_id) > 0
    )


# ---------------------------------------------------------------------------
# OAuth error helpers
# ---------------------------------------------------------------------------


def _get_oauth_token_error_code(response: httpx.Response) -> Optional[str]:
    try:
        response_body = response.json()
    except ValueError:
        return None
    if not isinstance(response_body, dict):
        return None
    return _clean_codex_auth_value(response_body.get("error"))


def _format_oauth_refresh_failure_detail(
    *,
    provider_label: str,
    response: httpx.Response,
) -> str:
    error_code = _get_oauth_token_error_code(response)
    suffix = f"status={response.status_code}, error={error_code}" if error_code else f"status={response.status_code}"
    return (
        f"Failed to refresh {provider_label} OAuth access token ({suffix}). "
        f"Re-authenticate {provider_label} CLI or configure valid OAuth client "
        "environment overrides."
    )
