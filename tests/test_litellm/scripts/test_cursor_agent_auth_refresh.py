"""Focused CURSOR-012 coverage for the Cursor Agent auth refresh core."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "cursor_agent_auth_refresh.py"


def _load_module():
    name = "cursor_agent_auth_refresh_cursor_012"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cursor():
    return _load_module()


def _jwt(exp: int) -> str:
    def encode(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'exp': exp})}.signature"


class _Response:
    def __init__(self, payload: Any, *, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body


def _write_auth(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, mode)


def test_defaults_match_cursor_sidecar_contract(cursor) -> None:
    assert cursor.DEFAULT_CURSOR_AGENT_AUTH_FILE == (
        "/home/zepfu/.config/cursor/auth.json"
    )
    assert cursor.DEFAULT_CURSOR_AGENT_AUTH_LOCK_FILE == (
        "/home/zepfu/.config/cursor/auth.json.lock"
    )
    assert cursor.DEFAULT_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS == 300.0
    assert cursor.DEFAULT_CURSOR_AGENT_AUTH_REFRESH_BUFFER_SECONDS == 300
    assert cursor.DEFAULT_CURSOR_AGENT_AUTH_FORCE_REFRESH is False
    assert cursor.DEFAULT_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS == 30.0


def test_health_and_eligibility_are_sanitized_and_read_only(cursor, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    secret = "access-secret-value"
    _write_auth(
        auth_path,
        {
            "accessToken": _jwt(2_000),
            "refreshToken": "refresh-secret-value",
            "apiKey": "api-secret-value",
        },
    )

    health = cursor.inspect_cursor_agent_auth_credential_health(
        auth_path,
        now=1_200,
    )
    eligibility = cursor.inspect_cursor_agent_auth_refresh_eligibility(
        auth_path,
        now=1_800,
        buffer_seconds=300,
        poll_interval_seconds=300,
    )

    assert health["provider"] == "cursor_agent"
    assert health["credential_shape"] == "accessToken+refreshToken+apiKey"
    assert health["credential_fingerprint"]
    assert health["expires_at"].startswith("1970-01-01T00:33:20")
    assert eligibility["eligible"] is True
    assert eligibility["access_token_state"] == "due"
    assert eligibility["refresh_due_at"].startswith("1970-01-01T00:28:20")
    serialized = json.dumps([health, eligibility])
    for value in (secret, "refresh-secret-value", "api-secret-value"):
        assert value not in serialized
    assert not (tmp_path / "auth.json.lock").exists()


def test_fresh_access_token_skips_without_http(cursor, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    lock_path = tmp_path / "auth.lock"
    _write_auth(
        auth_path,
        {
            "accessToken": _jwt(10_000),
            "refreshToken": "refresh-value",
            "apiKey": "api-value",
            "other": "preserve-me",
        },
    )

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        side_effect=AssertionError("fresh access token must not exchange"),
    ):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            lock_file=lock_path,
            now=1_000,
        )

    assert result["attempted"] is False
    assert result["refreshed"] is False
    assert result["skipped"] is True
    assert result["error_message"] is None
    assert json.loads(auth_path.read_text(encoding="utf-8"))["other"] == "preserve-me"


def test_due_access_token_exchanges_api_key_and_persists_complete_shape(
    cursor,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "nested" / "auth.json"
    lock_path = tmp_path / "nested" / "auth.lock"
    _write_auth(
        auth_path,
        {
            "accessToken": _jwt(1_200),
            "refreshToken": "old-refresh-secret",
            "apiKey": "api-key-secret",
            "metadata": {"keep": True},
        },
    )
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        captured["body"] = request.data
        captured["timeout"] = timeout
        return _Response(
            {
                "accessToken": _jwt(20_000),
                "refreshToken": "rotated-refresh-secret",
            }
        )

    with patch.object(cursor.urllib_request, "urlopen", side_effect=fake_urlopen):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            buffer_seconds=300,
            lock_file=lock_path,
            now=1_000,
            dashboard_base="https://cursor.test",
            http_timeout_seconds=12.5,
        )

    assert result["attempted"] is True
    assert result["refreshed"] is True
    assert result["refresh_method"] == "apiKey_exchange"
    assert result["credential_shape"] == "accessToken+refreshToken+apiKey"
    assert result["previous_credential_fingerprint"]
    assert result["credential_fingerprint"] != result["previous_credential_fingerprint"]
    assert captured["url"] == (
        "https://cursor.test/auth/exchange_user_api_key"
    )
    assert captured["method"] == "POST"
    assert captured["headers"]["authorization"] == "Bearer api-key-secret"
    assert captured["headers"]["content-type"] == "application/json"
    assert captured["headers"]["user-agent"].startswith("Cursor-CLI/")
    assert captured["body"] == b"{}"
    assert captured["timeout"] == 12.5

    saved = json.loads(auth_path.read_text(encoding="utf-8"))
    assert saved["accessToken"] == _jwt(20_000)
    assert saved["refreshToken"] == "rotated-refresh-secret"
    assert saved["apiKey"] == "api-key-secret"
    assert saved["obtained_at"] == 1_000
    assert saved["metadata"] == {"keep": True}
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert list(auth_path.parent.glob(".auth.json.*.tmp")) == []


def test_exchange_without_expiry_clears_stale_fields_and_next_call_skips(
    cursor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth(
        auth_path,
        {
            "accessToken": _jwt(900),
            "refreshToken": "old-refresh",
            "apiKey": "api-key",
            "expiresAt": 900,
            "expires_at": 900,
            "expiresAtMs": 900_000,
            "expires_at_ms": 900_000,
            "expiresIn": 1,
            "expires_in": 1,
        },
    )
    calls = {"exchange": 0, "write": 0}
    real_write = cursor._write_auth_data

    def fake_urlopen(*_args: Any, **_kwargs: Any):
        calls["exchange"] += 1
        return _Response(
            {
                "accessToken": _jwt(20_000),
                "refreshToken": "new-refresh",
            }
        )

    def counted_write(path, payload):
        calls["write"] += 1
        return real_write(path, payload)

    monkeypatch.setattr(cursor, "_write_auth_data", counted_write)
    with patch.object(cursor.urllib_request, "urlopen", side_effect=fake_urlopen):
        refreshed = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            now=1_000,
        )
        skipped = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            now=1_000,
        )

    assert refreshed["refreshed"] is True
    assert refreshed["access_token_state"] == "fresh"
    assert skipped["attempted"] is False
    assert skipped["skipped"] is True
    assert calls == {"exchange": 1, "write": 1}
    saved = json.loads(auth_path.read_text(encoding="utf-8"))
    assert saved["accessToken"] == _jwt(20_000)
    assert saved["refreshToken"] == "new-refresh"
    assert not {
        "expiresAt",
        "expires_at",
        "expiresAtMs",
        "expires_at_ms",
        "expiresIn",
        "expires_in",
    }.intersection(saved)


def test_expired_access_token_exchanges_api_key(cursor, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth(
        auth_path,
        {
            "accessToken": _jwt(900),
            "apiKey": "api-key",
        },
    )

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        return_value=_Response(
            {
                "accessToken": _jwt(20_000),
                "refreshToken": "new-refresh",
            }
        ),
    ) as exchange:
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            now=1_000,
        )

    assert result["refreshed"] is True
    assert result["access_token_state"] == "fresh"
    assert exchange.call_count == 1


def test_api_key_only_exchange_adds_access_and_refresh_tokens(
    cursor,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path, {"apiKey": "api-only-secret", "kind": "cursor"})

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        return_value=_Response(
            {
                "accessToken": _jwt(20_000),
                "refreshToken": "new-refresh-secret",
            }
        ),
    ):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            now=1_000,
        )

    assert result["refreshed"] is True
    saved = json.loads(auth_path.read_text(encoding="utf-8"))
    assert saved == {
        "apiKey": "api-only-secret",
        "kind": "cursor",
        "accessToken": _jwt(20_000),
        "refreshToken": "new-refresh-secret",
        "obtained_at": 1_000,
    }


def test_due_access_token_with_refresh_token_only_fails_closed(
    cursor,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth(
        auth_path,
        {
            "accessToken": _jwt(1_200),
            "refreshToken": "refresh-only-secret",
        },
    )

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        side_effect=AssertionError("refreshToken-only grant must not be invented"),
    ):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            buffer_seconds=300,
            now=1_000,
        )

    assert result["attempted"] is True
    assert result["refreshed"] is False
    assert result["error_class"] == "CursorAgentRefreshTokenOnlyError"
    assert "refreshToken-only grant" in result["error_message"]
    assert "refresh-only-secret" not in json.dumps(result)
    assert json.loads(auth_path.read_text(encoding="utf-8"))["accessToken"] == _jwt(
        1_200
    )


def test_http_failure_never_returns_raw_payload_or_secrets(cursor, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth(
        auth_path,
        {
            "accessToken": _jwt(1_000),
            "apiKey": "api-secret-value",
        },
    )

    class _FailingResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def getcode(self) -> int:
            return 401

        def read(self) -> bytes:
            return json.dumps(
                {
                    "apiKey": "api-secret-value",
                    "accessToken": "access-secret-value",
                    "refreshToken": "refresh-secret-value",
                    "message": "raw provider payload",
                }
            ).encode("utf-8")

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        return_value=_FailingResponse(),
    ):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            force=True,
            now=1_000,
        )

    assert result["refreshed"] is False
    assert result["error_class"] == "CursorAgentAuthExchangeError"
    serialized = json.dumps(result)
    for value in (
        "api-secret-value",
        "access-secret-value",
        "refresh-secret-value",
        "raw provider payload",
    ):
        assert value not in serialized


def test_metadata_overrides_apply_to_atomic_replacement(
    cursor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth(
        auth_path,
        {
            "accessToken": _jwt(1_200),
            "apiKey": "api-value",
        },
        mode=0o644,
    )
    monkeypatch.setenv("AAWM_CURSOR_AGENT_AUTH_FILE_MODE", "0o600")
    monkeypatch.setenv("AAWM_CURSOR_AGENT_AUTH_FILE_UID", str(os.getuid()))
    monkeypatch.setenv("AAWM_CURSOR_AGENT_AUTH_FILE_GID", str(os.getgid()))

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        return_value=_Response(
            {
                "accessToken": _jwt(20_000),
                "refreshToken": "new-refresh",
            }
        ),
    ):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            force=True,
            now=1_000,
        )

    assert result["refreshed"] is True
    metadata = auth_path.stat()
    assert metadata.st_mode & 0o777 == 0o600
    assert metadata.st_uid == os.getuid()
    assert metadata.st_gid == os.getgid()


def test_lock_failure_is_sanitized_and_fail_closed(cursor, tmp_path: Path, monkeypatch):
    from litellm.secret_managers.credential_file_lock import (
        CredentialFileLockError,
    )

    class _FailingLock:
        def __enter__(self):
            raise CredentialFileLockError(
                "lock held for apiKey=api-secret-value"
            )

        def __exit__(self, *_args: Any) -> None:
            return None

    auth_path = tmp_path / "auth.json"
    _write_auth(
        auth_path,
        {"accessToken": _jwt(1_000), "apiKey": "api-secret-value"},
    )
    monkeypatch.setattr(cursor, "_credential_file_lock", lambda _path: _FailingLock())

    result = cursor.refresh_cursor_agent_auth_file(
        auth_path,
        force=True,
        now=1_000,
    )

    assert result["error_class"] == "CredentialFileLockError"
    assert "api-secret-value" not in json.dumps(result)
    assert result["refreshed"] is False


def test_filesystem_lock_rereads_and_preserves_newest_credential(
    cursor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    lock_path = tmp_path / "auth.lock"
    _write_auth(
        auth_path,
        {"accessToken": _jwt(900), "apiKey": "old-api-key"},
    )
    newest = {
        "accessToken": _jwt(20_000),
        "refreshToken": "newest-refresh",
        "apiKey": "newest-api-key",
        "generation": 2,
    }
    acquired: list[Path] = []

    class _ReplacingLock:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self):
            acquired.append(self.path)
            _write_auth(auth_path, newest)
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(
        cursor,
        "_credential_file_lock",
        lambda path: _ReplacingLock(path),
    )
    monkeypatch.setattr(
        cursor,
        "_write_auth_data",
        lambda *_args, **_kwargs: pytest.fail(
            "newest credential must not be overwritten"
        ),
    )

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        side_effect=AssertionError("newest credential must not exchange"),
    ):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            lock_file=lock_path,
            now=1_000,
        )

    assert acquired == [lock_path]
    assert result["attempted"] is False
    assert result["skipped"] is True
    assert json.loads(auth_path.read_text(encoding="utf-8")) == newest


def test_process_local_singleflight_performs_one_exchange_and_write(
    cursor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    lock_path = tmp_path / "auth.lock"
    _write_auth(
        auth_path,
        {"accessToken": _jwt(1_000), "apiKey": "api-value"},
    )
    calls = {"exchange": 0, "write": 0}
    entered = threading.Event()
    release = threading.Event()
    real_write = cursor._write_auth_data

    def fake_urlopen(*_args: Any, **_kwargs: Any):
        calls["exchange"] += 1
        entered.set()
        assert release.wait(timeout=5)
        return _Response(
            {
                "accessToken": _jwt(20_000),
                "refreshToken": "rotated-refresh",
            }
        )

    def counted_write(path, payload):
        calls["write"] += 1
        return real_write(path, payload)

    monkeypatch.setattr(cursor, "_write_auth_data", counted_write)
    results: list[dict[str, Any]] = []

    def run() -> None:
        results.append(
            cursor.refresh_cursor_agent_auth_file(
                auth_path,
                lock_file=lock_path,
                force=True,
                now=1_000,
            )
        )

    with patch.object(cursor.urllib_request, "urlopen", side_effect=fake_urlopen):
        first = threading.Thread(target=run)
        second = threading.Thread(target=run)
        first.start()
        assert entered.wait(timeout=5)
        second.start()
        time.sleep(0.05)
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert len(results) == 2
    assert calls == {"exchange": 1, "write": 1}
    assert all(result["refreshed"] for result in results)
    assert json.loads(auth_path.read_text(encoding="utf-8"))["refreshToken"] == (
        "rotated-refresh"
    )


def test_auth_path_symlink_fails_closed_without_http_or_write(
    cursor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_path = tmp_path / "real-auth.json"
    real_bytes = json.dumps(
        {
            "accessToken": "target-access-secret",
            "apiKey": "target-api-secret",
        }
    ).encode("utf-8")
    real_path.write_bytes(real_bytes)
    auth_path = tmp_path / "auth.json"
    auth_path.symlink_to(real_path)
    monkeypatch.setattr(
        cursor,
        "_write_auth_data",
        lambda *_args, **_kwargs: pytest.fail("symlink must not be written"),
    )

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        side_effect=AssertionError("symlink must not exchange"),
    ):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            lock_file=tmp_path / "auth.lock",
            now=1_000,
        )

    assert result["error_class"] == "CredentialPathIsSymlinkError"
    assert result["refreshed"] is False
    assert real_path.read_bytes() == real_bytes
    assert auth_path.is_symlink()
    serialized = json.dumps(result)
    assert "target-access-secret" not in serialized
    assert "target-api-secret" not in serialized


def test_truncated_auth_file_fails_closed_without_http_or_write(
    cursor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    original = b'{"accessToken":"truncated-access-secret",'
    auth_path.write_bytes(original)
    os.chmod(auth_path, 0o600)
    monkeypatch.setattr(
        cursor,
        "_write_auth_data",
        lambda *_args, **_kwargs: pytest.fail(
            "truncated credential must not be written"
        ),
    )

    with patch.object(
        cursor.urllib_request,
        "urlopen",
        side_effect=AssertionError("truncated credential must not exchange"),
    ):
        result = cursor.refresh_cursor_agent_auth_file(
            auth_path,
            lock_file=tmp_path / "auth.lock",
            now=1_000,
        )

    assert result["error_class"] == "CursorAgentCredentialError"
    assert result["refreshed"] is False
    assert auth_path.read_bytes() == original
    assert "truncated-access-secret" not in json.dumps(result)


def test_shared_lock_is_used_and_atomic_writer_is_shared(cursor) -> None:
    source = Path(cursor.__file__).read_text(encoding="utf-8")
    assert "from litellm.secret_managers.credential_file_lock import" in source
    assert "from litellm.secret_managers.credential_file_write import" in source
    assert "write_and_publish_private_text" in source
    assert "with credential_file_lock(lock_path)" in source
    assert "grant_type" not in source
    assert "os.replace(" not in source
    assert "raw provider payload" not in source


def test_refresh_interval_environment_default_is_used(cursor, tmp_path: Path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path, {"accessToken": "opaque-access-token"})
    monkeypatch.setenv("AAWM_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS", "17")

    result = cursor.inspect_cursor_agent_auth_refresh_eligibility(
        auth_path,
        now=1_000,
    )

    assert result["eligible"] is False
    assert result["next_refresh_check_at"].startswith("1970-01-01T00:16:57")


def test_eligibility_uses_persisted_timestamp_lifetime(
    cursor, tmp_path: Path
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth(
        auth_path,
        {
            "accessToken": "opaque-access",
            "refreshToken": "old-refresh",
            "obtained_at": "1970-01-01T00:16:40Z",
            "expiresAt": 1_900,
        },
    )

    result = cursor.inspect_cursor_agent_auth_refresh_eligibility(
        auth_path,
        now=1_000,
        poll_interval_seconds=300,
    )

    assert result["issued_lifetime_seconds"] == 900.0
    assert result["refresh_threshold_seconds"] == 450.0
    assert result["refresh_threshold_source"] == "persisted_timestamp"
    assert result["refresh_threshold_degraded"] is False
    assert result["refresh_due_at"] == "1970-01-01T00:24:10Z"
    assert result["eligible"] is False


def test_missing_lifetime_is_explicitly_degraded_in_eligibility_and_summary(
    cursor, tmp_path: Path
) -> None:
    observed_at = datetime.now(timezone.utc)
    auth_path = tmp_path / "auth.json"
    _write_auth(
        auth_path,
        {
            "accessToken": "opaque-access",
            "refreshToken": "old-refresh",
            "expiresAt": (
                observed_at + timedelta(hours=2)
            ).isoformat().replace("+00:00", "Z"),
        },
    )

    eligibility = cursor.inspect_cursor_agent_auth_refresh_eligibility(
        auth_path,
        now=observed_at,
        poll_interval_seconds=300,
    )
    summary = cursor.refresh_cursor_agent_auth_file(
        auth_path,
        buffer_seconds=300,
        force=False,
        lock_file=tmp_path / "auth.json.lock",
    )

    assert eligibility["credential_health"] == "fresh"
    assert eligibility["issued_lifetime_seconds"] is None
    assert eligibility["refresh_threshold_seconds"] == 300.0
    assert eligibility["refresh_threshold_source"] == "fallback"
    assert eligibility["refresh_threshold_degraded"] is True
    assert summary["skipped"] is True
    assert summary["auth_degraded"] is True
    assert summary["issued_lifetime_seconds"] is None
    assert summary["refresh_threshold_seconds"] == 300.0
    assert summary["refresh_threshold_source"] == "fallback"
    assert summary["refresh_threshold_degraded"] is True
