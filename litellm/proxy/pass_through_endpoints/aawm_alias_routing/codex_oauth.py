"""Codex OAuth inventory loading, token validation, and request detection.

Runtime dependency ``_get_request_header_or_passthrough_alias`` is injected via
:func:`configure_codex_oauth_runtime` (the function lives in the god module and
depends on the pass-through header prefix constant).
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from fastapi import HTTPException, Request

from litellm.llms.chatgpt.common_utils import (
    CHATGPT_API_BASE,
    get_chatgpt_default_headers,
)
from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers
from litellm.secret_managers.codex_oauth_inventory import (
    CodexOAuthCredentialRecord,
    CodexOAuthCredentialSnapshot,
    CodexOAuthInventoryError,
    load_codex_oauth_credential,
    load_codex_oauth_inventory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Legacy facade symbols retained for decomposition compatibility. The active
# loader below does not consult these paths or enroll credentials from them.
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
# Explicit inventory compatibility accessor
# ---------------------------------------------------------------------------


def _get_anthropic_adapter_codex_auth_file_path() -> Optional[Path]:
    """Compatibility accessor for the first explicit enabled inventory record."""
    inventory = load_codex_oauth_inventory()
    return inventory.select_record().auth_path


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


@dataclass(frozen=True)
class CodexOAuthRequestAuth:
    """Selected account metadata plus headers, with secrets hidden from repr."""

    account_label: str
    account_hash: str
    lane_key: str
    headers: dict[str, str] = field(repr=False)


def _codex_oauth_account_lane_key(
    *,
    account_label: str,
    account_hash: str,
) -> str:
    """Return a server-owned, secret-safe lane for one configured account."""
    return f"codex-oauth:{account_label}:{account_hash}"


def _codex_oauth_candidate_identity(
    candidate: dict[str, Any],
) -> Optional[dict[str, str]]:
    """Return the safe selected-account identity carried by a candidate."""
    account_label = _clean_codex_auth_value(
        candidate.get("codex_oauth_account_label")
    )
    account_hash = _clean_codex_auth_value(
        candidate.get("codex_oauth_account_hash")
    )
    lane_key = _clean_codex_auth_value(candidate.get("codex_oauth_lane_key"))
    present_count = sum(
        value is not None for value in (account_label, account_hash, lane_key)
    )
    if present_count == 0:
        return None
    if present_count != 3:
        raise HTTPException(
            status_code=500,
            detail="Selected Codex OAuth account context is incomplete.",
        )
    assert account_label is not None
    assert account_hash is not None
    assert lane_key is not None
    expected_lane = _codex_oauth_account_lane_key(
        account_label=account_label,
        account_hash=account_hash,
    )
    if lane_key != expected_lane:
        raise HTTPException(
            status_code=500,
            detail="Selected Codex OAuth account lane is invalid.",
        )
    return {
        "account_label": account_label,
        "account_hash": account_hash,
        "lane_key": lane_key,
    }


def _bind_codex_oauth_candidate_to_request(
    request: Request,
    candidate: dict[str, Any],
) -> Optional[dict[str, str]]:
    """Bind only the safe selected-account identity to this request."""
    identity = _codex_oauth_candidate_identity(candidate)
    if identity is None:
        setattr(request.state, "aawm_codex_oauth_selected_account", None)
        return None
    bound = {
        **identity,
        "model": str(candidate.get("model") or ""),
    }
    setattr(request.state, "aawm_codex_oauth_selected_account", bound)
    return dict(bound)


def _get_bound_codex_oauth_candidate_identity(
    request: Request,
) -> Optional[dict[str, str]]:
    bound = getattr(request.state, "aawm_codex_oauth_selected_account", None)
    if not isinstance(bound, dict):
        return None
    candidate = {
        "codex_oauth_account_label": bound.get("account_label"),
        "codex_oauth_account_hash": bound.get("account_hash"),
        "codex_oauth_lane_key": bound.get("lane_key"),
    }
    identity = _codex_oauth_candidate_identity(candidate)
    if identity is None:
        return None
    identity["model"] = str(bound.get("model") or "")
    return identity


def _codex_oauth_responses_target_url() -> str:
    """Return the OAuth-only ChatGPT Codex Responses target."""
    return f"{(os.getenv('CHATGPT_API_BASE') or CHATGPT_API_BASE).rstrip('/')}/responses"


def _codex_oauth_credential_snapshot_is_valid(
    credential: CodexOAuthCredentialSnapshot,
) -> bool:
    if credential.expires_at is None:
        return True
    return time.time() < credential.expires_at - 60


async def _load_codex_oauth_headers_for_record(
    request: Request,
    record: CodexOAuthCredentialRecord,
) -> CodexOAuthRequestAuth:
    """Build headers only from one already-selected immutable record."""
    try:
        credential = load_codex_oauth_credential(record)
    except CodexOAuthInventoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None

    if not _codex_oauth_credential_snapshot_is_valid(credential):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Codex OAuth credential '{record.label}' "
                f"(account_hash={credential.account_hash}) is expired or "
                "invalid. The "
                "provider-status sidecar owns Codex auth refresh; confirm the "
                "configured account can be refreshed."
            ),
        )

    headers = _safe_get_request_headers(request)
    assert _get_request_header_or_passthrough_alias is not None
    session_id = (
        _get_request_header_or_passthrough_alias(request, "session_id")
        or headers.get("x-claude-code-session-id")
        or headers.get("X-Claude-Code-Session-Id")
    )

    return CodexOAuthRequestAuth(
        account_label=record.label,
        account_hash=credential.account_hash,
        lane_key=_codex_oauth_account_lane_key(
            account_label=record.label,
            account_hash=credential.account_hash,
        ),
        headers=get_chatgpt_default_headers(
            access_token=credential.access_token,
            account_id=credential.account_id,
            session_id=session_id,
        ),
    )


async def _load_local_codex_auth_selection(
    request: Request,
    *,
    account_label: Optional[str] = None,
    model: Optional[str] = None,
) -> CodexOAuthRequestAuth:
    """Select from the explicit inventory and load exactly that record."""
    try:
        inventory = load_codex_oauth_inventory()
        record = inventory.select_record(label=account_label, model=model)
    except CodexOAuthInventoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return await _load_codex_oauth_headers_for_record(request, record)


async def _load_bound_codex_oauth_auth(
    request: Request,
) -> CodexOAuthRequestAuth:
    """Load exactly the server-selected account or fail closed without secrets."""
    identity = _get_bound_codex_oauth_candidate_identity(request)
    if identity is None:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": (
                        "Codex OAuth dispatch requires a server-selected "
                        "configured account."
                    ),
                    "type": "rate_limit_error",
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                },
                "failure_phase": "pre_dispatch_auth",
                "attempted_provider_call": False,
            },
        )
    try:
        selection = await _load_local_codex_auth_selection(
            request,
            account_label=identity["account_label"],
            model=identity["model"] or None,
        )
    except HTTPException:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": (
                        "Selected Codex OAuth account is not currently "
                        "authentication-ready."
                    ),
                    "type": "rate_limit_error",
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                },
                "account": identity,
                "failure_phase": "pre_dispatch_auth",
                "attempted_provider_call": False,
            },
        ) from None
    if (
        selection.account_hash != identity["account_hash"]
        or selection.lane_key != identity["lane_key"]
    ):
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": (
                        "Selected Codex OAuth account identity changed before "
                        "dispatch."
                    ),
                    "type": "rate_limit_error",
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                },
                "account": identity,
                "failure_phase": "pre_dispatch_auth",
                "attempted_provider_call": False,
            },
        )
    return selection


async def _load_local_codex_auth_headers(request: Request) -> dict[str, str]:
    """Compatibility wrapper for the current first-eligible account consumer."""
    selection = await _load_local_codex_auth_selection(request)
    return dict(selection.headers)


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
