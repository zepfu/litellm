from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import codex_oauth
from litellm.secret_managers.codex_oauth_inventory import (
    CODEX_OAUTH_INVENTORY_ENV,
    CodexOAuthCredentialError,
    CodexOAuthInventoryError,
    codex_oauth_account_identity_hash,
    load_codex_oauth_credential,
    load_codex_oauth_inventory,
)
from scripts import codex_oauth_refresh


def _jwt(account_id: str, *, expires_at: int) -> str:
    payload = {
        "exp": expires_at,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return f"header.{encoded.rstrip('=')}.signature"


def _write_auth(
    path: Path,
    *,
    account_id: str,
    token_name: str,
    expires_at: int,
) -> str:
    access_token = _jwt(account_id, expires_at=expires_at)
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": f"refresh-{token_name}",
                    "id_token": _jwt(account_id, expires_at=expires_at),
                    "account_id": account_id,
                    "expires_at": expires_at,
                }
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return access_token


def _account(
    tmp_path: Path,
    *,
    label: str,
    account_id: str,
    priority: int,
    models: list[str],
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "label": label,
        "auth_path": str(tmp_path / f"{label}.json"),
        "lock_path": str(tmp_path / f"{label}.lock"),
        "priority": priority,
        "weight": 1.0,
        "enabled": enabled,
        "models": models,
        "expected_account_hash": codex_oauth_account_identity_hash(account_id),
    }


def _inventory_json(accounts: list[dict[str, Any]]) -> str:
    return json.dumps({"schema_version": 1, "accounts": accounts})


def _request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [
        (name.lower().encode(), value.encode())
        for name, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": raw_headers,
            "client": ("test", 123),
            "server": ("test", 80),
        }
    )


@pytest.fixture(autouse=True)
def _configure_codex_oauth_runtime() -> None:
    codex_oauth.configure_codex_oauth_runtime(
        get_request_header_or_passthrough_alias=lambda request, name: (
            request.headers.get(name)
            or request.headers.get(f"x-pass-{name}")
        )
    )


def test_inventory_is_explicit_ordered_and_model_eligible(tmp_path: Path) -> None:
    account1 = _account(
        tmp_path,
        label="account1",
        account_id="acct-one",
        priority=20,
        models=["*"],
    )
    account2 = _account(
        tmp_path,
        label="account2",
        account_id="acct-two",
        priority=10,
        models=["gpt-selected"],
    )
    disabled = _account(
        tmp_path,
        label="disabled",
        account_id="acct-disabled",
        priority=0,
        models=["*"],
        enabled=False,
    )

    inventory = load_codex_oauth_inventory(
        _inventory_json([account1, account2, disabled])
    )

    assert [record.label for record in inventory.records] == [
        "account1",
        "account2",
        "disabled",
    ]
    assert [record.label for record in inventory.ordered_records()] == [
        "disabled",
        "account2",
        "account1",
    ]
    assert [
        record.label
        for record in inventory.ordered_records(enabled_only=True)
    ] == ["account2", "account1"]
    assert inventory.select_record(model="gpt-selected").label == "account2"
    assert inventory.select_record(model="other-model").label == "account1"
    assert inventory.records[0].weight == 1.0
    assert str(inventory.records[0].auth_path) not in repr(inventory.records[0])


@pytest.mark.parametrize(
    "duplicate_field",
    ["label", "auth_path", "lock_path", "expected_account_hash"],
)
def test_inventory_rejects_duplicate_identity_fields(
    tmp_path: Path,
    duplicate_field: str,
) -> None:
    account1 = _account(
        tmp_path,
        label="account1",
        account_id="acct-one",
        priority=10,
        models=["*"],
    )
    account2 = _account(
        tmp_path,
        label="account2",
        account_id="acct-two",
        priority=20,
        models=["*"],
    )
    account2[duplicate_field] = account1[duplicate_field]

    with pytest.raises(CodexOAuthInventoryError):
        load_codex_oauth_inventory(_inventory_json([account1, account2]))


def test_inventory_rejects_implicit_path_globs(tmp_path: Path) -> None:
    account = _account(
        tmp_path,
        label="account1",
        account_id="acct-one",
        priority=10,
        models=["*"],
    )
    account["auth_path"] = str(tmp_path / "oauth.*.json")

    with pytest.raises(CodexOAuthInventoryError, match="explicit path"):
        load_codex_oauth_inventory(_inventory_json([account]))


@pytest.mark.asyncio
async def test_selected_record_alone_builds_headers_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = int(time.time())
    account1 = _account(
        tmp_path,
        label="account1",
        account_id="acct-one",
        priority=10,
        models=["gpt-selected"],
    )
    account2 = _account(
        tmp_path,
        label="account2",
        account_id="acct-two",
        priority=20,
        models=["gpt-selected"],
    )
    account1_token = _write_auth(
        Path(account1["auth_path"]),
        account_id="acct-one",
        token_name="one",
        expires_at=now + 3600,
    )
    account2_token = _write_auth(
        Path(account2["auth_path"]),
        account_id="acct-two",
        token_name="two",
        expires_at=now + 3600,
    )
    before = Path(account1["auth_path"]).read_bytes()
    monkeypatch.setenv(
        CODEX_OAUTH_INVENTORY_ENV,
        _inventory_json([account1, account2]),
    )

    selection = await codex_oauth._load_local_codex_auth_selection(
        _request(
            {
                "session_id": "session-1",
                "ChatGPT-Account-Id": "acct-two",
            }
        ),
        account_label="account1",
        model="gpt-selected",
    )

    assert selection.account_label == "account1"
    assert selection.account_hash == account1["expected_account_hash"]
    assert selection.headers["Authorization"] == f"Bearer {account1_token}"
    assert selection.headers["ChatGPT-Account-Id"] == "acct-one"
    assert account2_token not in json.dumps(selection.headers)
    assert Path(account1["auth_path"]).read_bytes() == before
    assert not Path(account1["lock_path"]).exists()
    assert not Path(account2["lock_path"]).exists()


@pytest.mark.asyncio
async def test_identity_swap_fails_with_only_safe_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = _account(
        tmp_path,
        label="account1",
        account_id="acct-one",
        priority=10,
        models=["*"],
    )
    leaked_token = _write_auth(
        Path(account["auth_path"]),
        account_id="acct-two",
        token_name="leaked",
        expires_at=int(time.time()) + 3600,
    )
    monkeypatch.setenv(
        CODEX_OAUTH_INVENTORY_ENV,
        _inventory_json([account]),
    )

    with pytest.raises(HTTPException) as exc_info:
        await codex_oauth._load_local_codex_auth_headers(_request())

    detail = str(exc_info.value.detail)
    assert "identity mismatch" in detail
    assert account["expected_account_hash"] in detail
    assert codex_oauth_account_identity_hash("acct-two") in detail
    assert "acct-one" not in detail
    assert "acct-two" not in detail
    assert leaked_token not in detail
    assert str(account["auth_path"]) not in detail


def test_loader_fails_closed_for_missing_malformed_symlink_and_mode(
    tmp_path: Path,
) -> None:
    account = _account(
        tmp_path,
        label="account1",
        account_id="acct-one",
        priority=10,
        models=["*"],
    )
    record = load_codex_oauth_inventory(
        _inventory_json([account])
    ).records[0]
    auth_path = Path(account["auth_path"])

    with pytest.raises(CodexOAuthCredentialError, match="missing"):
        load_codex_oauth_credential(record)

    auth_path.write_text("{", encoding="utf-8")
    auth_path.chmod(0o600)
    with pytest.raises(CodexOAuthCredentialError, match="malformed"):
        load_codex_oauth_credential(record)

    _write_auth(
        auth_path,
        account_id="acct-one",
        token_name="one",
        expires_at=int(time.time()) + 3600,
    )
    auth_path.chmod(0o644)
    with pytest.raises(CodexOAuthCredentialError, match="permissions"):
        load_codex_oauth_credential(record)

    auth_path.unlink()
    target = tmp_path / "target.json"
    _write_auth(
        target,
        account_id="acct-one",
        token_name="one",
        expires_at=int(time.time()) + 3600,
    )
    auth_path.symlink_to(target)
    with pytest.raises(CodexOAuthCredentialError, match="symlink"):
        load_codex_oauth_credential(record)


@pytest.mark.asyncio
async def test_missing_inventory_fails_before_api_key_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_path = tmp_path / ".codex" / "auth.json"
    legacy_path.parent.mkdir()
    _write_auth(
        legacy_path,
        account_id="acct-legacy",
        token_name="legacy",
        expires_at=int(time.time()) + 3600,
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(CODEX_OAUTH_INVENTORY_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")

    with pytest.raises(HTTPException) as exc_info:
        await codex_oauth._load_local_codex_auth_headers(_request())

    detail = str(exc_info.value.detail)
    assert CODEX_OAUTH_INVENTORY_ENV in detail
    assert "must-not-be-used" not in detail
    assert "api.openai.com" not in detail


def test_record_refresh_rejects_identity_change_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = _account(
        tmp_path,
        label="account1",
        account_id="acct-one",
        priority=10,
        models=["*"],
    )
    auth_path = Path(account["auth_path"])
    _write_auth(
        auth_path,
        account_id="acct-one",
        token_name="one",
        expires_at=1,
    )
    before = auth_path.read_bytes()
    record = load_codex_oauth_inventory(
        _inventory_json([account])
    ).records[0]
    replacement_token = _jwt(
        "acct-two",
        expires_at=int(time.time()) + 3600,
    )
    monkeypatch.setattr(
        codex_oauth_refresh,
        "_refresh_token_data",
        lambda *args, **kwargs: {
            "access_token": replacement_token,
            "id_token": replacement_token,
            "refresh_token": "replacement-refresh",
        },
    )

    result = codex_oauth_refresh.refresh_codex_oauth_inventory_record(
        record,
        force=True,
    )

    assert result["refreshed"] is False
    assert result["error_class"] == "CodexOAuthIdentityMismatchError"
    assert result["account_label"] == "account1"
    assert result["account_hash"] == account["expected_account_hash"]
    assert "auth_file" not in result
    assert "account_id" not in result
    assert "acct-one" not in json.dumps(result)
    assert "acct-two" not in json.dumps(result)
    assert str(auth_path) not in json.dumps(result)
    assert auth_path.read_bytes() == before
    assert Path(account["lock_path"]).exists()
