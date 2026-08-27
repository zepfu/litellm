"""Focused tests for the real token_invalidated credential reload helper.

Exercises ``reload_codex_oauth_credential_after_token_invalidated`` end to end
against a real inventory environment, real on-disk credential files (mode
0600), and the real credential file lock. Assertions compare only secret-safe
SHA-256 fingerprints and account identity; token values are never asserted or
printed.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import pytest
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import codex_oauth
from litellm.secret_managers.codex_oauth_inventory import (
    CODEX_OAUTH_INVENTORY_ENV,
    codex_oauth_account_identity_hash,
)

_ACCOUNT_LABEL = "account1"
_ACCOUNT_ID = "acct-one"


def _jwt(account_id: str, *, expires_at: int) -> str:
    payload = {
        "exp": expires_at,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return f"header.{encoded.rstrip('=')}.signature"


def _write_auth(path: Path, *, expires_at: int) -> None:
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _jwt(_ACCOUNT_ID, expires_at=expires_at),
                    "refresh_token": "refresh-material",
                    "id_token": _jwt(_ACCOUNT_ID, expires_at=expires_at),
                    "account_id": _ACCOUNT_ID,
                    "expires_at": expires_at,
                }
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
    )


@pytest.fixture(autouse=True)
def _configure_codex_oauth_runtime() -> None:
    codex_oauth.configure_codex_oauth_runtime(
        get_request_header_or_passthrough_alias=lambda request, name: (
            request.headers.get(name) or request.headers.get(f"x-pass-{name}")
        )
    )


@pytest.fixture
def deployed_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    auth_path = tmp_path / f"{_ACCOUNT_LABEL}.json"
    lock_path = tmp_path / f"{_ACCOUNT_LABEL}.lock"
    _write_auth(auth_path, expires_at=int(time.time()) + 3600)
    account_hash = codex_oauth_account_identity_hash(_ACCOUNT_ID)
    monkeypatch.setenv(
        CODEX_OAUTH_INVENTORY_ENV,
        json.dumps(
            {
                "schema_version": 1,
                "accounts": [
                    {
                        "label": _ACCOUNT_LABEL,
                        "auth_path": str(auth_path),
                        "lock_path": str(lock_path),
                        "priority": 10,
                        "weight": 1.0,
                        "enabled": True,
                        "models": ["*"],
                        "expected_account_hash": account_hash,
                    }
                ],
            }
        ),
    )
    return {
        "auth_path": auth_path,
        "account_hash": account_hash,
        "lane_key": codex_oauth._codex_oauth_account_lane_key(
            account_label=_ACCOUNT_LABEL, account_hash=account_hash
        ),
    }


async def _dispatch(request: Request) -> codex_oauth.CodexOAuthRequestAuth:
    """Real dispatch-time auth load plus real dispatched-fingerprint record."""
    selection = await codex_oauth._load_local_codex_auth_selection(
        request, account_label=_ACCOUNT_LABEL
    )
    codex_oauth._record_codex_oauth_dispatched_auth(request, selection)
    return selection


def _recovery_state(request: Request) -> dict[str, Any]:
    state = getattr(
        request.state,
        codex_oauth._CODEX_OAUTH_INVALIDATION_RECOVERY_STATE_ATTR,
        None,
    )
    assert isinstance(state, dict)
    return state


async def test_reload_returns_changed_credential(
    deployed_credential: dict[str, Any],
) -> None:
    request = _request()
    dispatched = await _dispatch(request)
    dispatched_fingerprint = codex_oauth._codex_oauth_auth_fingerprint(
        dispatched
    )

    _write_auth(
        deployed_credential["auth_path"],
        expires_at=int(time.time()) + 7200,
    )

    refreshed = (
        await codex_oauth.reload_codex_oauth_credential_after_token_invalidated(
            request,
            account_label=_ACCOUNT_LABEL,
            expected_account_hash=deployed_credential["account_hash"],
            expected_lane_key=deployed_credential["lane_key"],
        )
    )

    assert refreshed is not None
    assert refreshed.account_label == _ACCOUNT_LABEL
    assert refreshed.account_hash == deployed_credential["account_hash"]
    assert refreshed.lane_key == deployed_credential["lane_key"]
    refreshed_fingerprint = codex_oauth._codex_oauth_auth_fingerprint(refreshed)
    assert refreshed_fingerprint != dispatched_fingerprint
    state = _recovery_state(request)
    assert state["fingerprints"][_ACCOUNT_LABEL] == refreshed_fingerprint
    assert state["reloaded"] == {_ACCOUNT_LABEL}


async def test_reload_returns_none_when_credential_unchanged(
    deployed_credential: dict[str, Any],
) -> None:
    request = _request()
    dispatched = await _dispatch(request)
    dispatched_fingerprint = codex_oauth._codex_oauth_auth_fingerprint(
        dispatched
    )

    refreshed = (
        await codex_oauth.reload_codex_oauth_credential_after_token_invalidated(
            request,
            account_label=_ACCOUNT_LABEL,
            expected_account_hash=deployed_credential["account_hash"],
            expected_lane_key=deployed_credential["lane_key"],
        )
    )

    assert refreshed is None
    state = _recovery_state(request)
    assert state["fingerprints"][_ACCOUNT_LABEL] == dispatched_fingerprint
    assert state["reloaded"] == {_ACCOUNT_LABEL}


async def test_reload_is_one_shot_per_request(
    deployed_credential: dict[str, Any],
) -> None:
    request = _request()
    await _dispatch(request)

    _write_auth(
        deployed_credential["auth_path"],
        expires_at=int(time.time()) + 7200,
    )
    first_refresh = (
        await codex_oauth.reload_codex_oauth_credential_after_token_invalidated(
            request, account_label=_ACCOUNT_LABEL
        )
    )
    assert first_refresh is not None
    first_fingerprint = codex_oauth._codex_oauth_auth_fingerprint(
        first_refresh
    )

    # A second invalidation on the same request must not re-read again.
    _write_auth(
        deployed_credential["auth_path"],
        expires_at=int(time.time()) + 10800,
    )
    second_refresh = (
        await codex_oauth.reload_codex_oauth_credential_after_token_invalidated(
            request, account_label=_ACCOUNT_LABEL
        )
    )

    assert second_refresh is None
    state = _recovery_state(request)
    assert state["fingerprints"][_ACCOUNT_LABEL] == first_fingerprint
    assert state["reloaded"] == {_ACCOUNT_LABEL}


@pytest.mark.parametrize("mismatch", ["account_hash", "lane_key"])
async def test_reload_returns_none_on_identity_mismatch(
    deployed_credential: dict[str, Any], mismatch: str
) -> None:
    request = _request()
    dispatched = await _dispatch(request)
    dispatched_fingerprint = codex_oauth._codex_oauth_auth_fingerprint(
        dispatched
    )

    _write_auth(
        deployed_credential["auth_path"],
        expires_at=int(time.time()) + 7200,
    )

    expected_account_hash = deployed_credential["account_hash"]
    expected_lane_key = deployed_credential["lane_key"]
    if mismatch == "account_hash":
        expected_account_hash = "0" * 12
    else:
        expected_lane_key = f"codex-oauth:{_ACCOUNT_LABEL}:{'0' * 12}"

    refreshed = (
        await codex_oauth.reload_codex_oauth_credential_after_token_invalidated(
            request,
            account_label=_ACCOUNT_LABEL,
            expected_account_hash=expected_account_hash,
            expected_lane_key=expected_lane_key,
        )
    )

    assert refreshed is None
    state = _recovery_state(request)
    # Mismatched material is never recorded and the one-shot slot is consumed.
    assert state["fingerprints"][_ACCOUNT_LABEL] == dispatched_fingerprint
    assert state["reloaded"] == {_ACCOUNT_LABEL}
