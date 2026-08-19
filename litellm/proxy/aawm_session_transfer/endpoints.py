"""Authenticated read-only session-transfer status endpoint."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.aawm_session_transfer.registry import get_session_transfer_registry
from litellm.proxy.aawm_session_transfer.schema import (
    DEFAULT_QUERY_LIMIT,
    MAX_QUERY_RESULTS,
    TRANSFER_PERMISSION,
    TRANSFER_ROUTE,
    assert_content_free,
    sanitize_identity,
)

router = APIRouter()


def caller_may_read_session_transfer(user_api_key_dict: UserAPIKeyAuth) -> bool:
    role = str(getattr(user_api_key_dict, "user_role", "") or "").lower()
    if role in {"proxy_admin", "proxy_admin_viewer"}:
        return True
    permissions = getattr(user_api_key_dict, "permissions", None) or []
    if isinstance(permissions, dict):
        return bool(permissions.get(TRANSFER_PERMISSION))
    if isinstance(permissions, (list, tuple, set)):
        return TRANSFER_PERMISSION in permissions
    return False


@router.get(TRANSFER_ROUTE, tags=["aawm"])
async def get_session_transfer_status(
    session_id: Optional[str] = Query(default=None),
    codex_session_id: Optional[str] = Query(default=None),
    canonical_session_id: Optional[str] = Query(default=None),
    agent_id: Optional[str] = Query(default=None),
    litellm_call_id: Optional[str] = Query(default=None),
    active_only: bool = Query(default=False),
    limit: int = Query(default=DEFAULT_QUERY_LIMIT, ge=1, le=MAX_QUERY_RESULTS),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    if not caller_may_read_session_transfer(user_api_key_dict):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Virtual key is not allowed to call this route.",
        )
    has_filter = any(
        sanitize_identity(value)
        for value in (
            session_id,
            codex_session_id,
            canonical_session_id,
            agent_id,
            litellm_call_id,
        )
    )
    if not has_filter:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Provide session_id, codex_session_id, canonical_session_id, "
                "agent_id, or litellm_call_id."
            ),
        )
    payload = await get_session_transfer_registry().query(
        litellm_call_id=litellm_call_id,
        session_id=session_id,
        codex_session_id=codex_session_id,
        canonical_session_id=canonical_session_id,
        agent_id=agent_id,
        active_only=active_only,
        limit=limit,
    )
    assert_content_free(payload)
    return payload
