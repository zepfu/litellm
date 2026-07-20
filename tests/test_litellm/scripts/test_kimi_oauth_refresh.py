"""Focused coverage for the shared Kimi Code OAuth credential refresher."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time
from contextlib import contextmanager
from email.message import Message
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch
from urllib import error as urllib_error

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "kimi_oauth_refresh.py"


def _load_module():
    name = "kimi_oauth_refresh_test"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kimi():
    return _load_module()


class _Response:
    def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _credential(
    *,
    access_token: str = "old-access",
    refresh_token: str = "old-refresh",
    expires_at: float = 1,
    expires_in: int = 900,
) -> dict[str, Any]:
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "expires_in": expires_in,
        "scope": "kimi-code",
        "token_type": "Bearer",
    }


def _write_credential(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)


def _lock_sentinel(auth_path: Path) -> Path:
    return auth_path.parent / "oauth" / "kimi-code"


def _refresh(kimi, auth_path: Path, **kwargs: Any) -> dict[str, Any]:
    return kimi.refresh_kimi_oauth_auth_file(
        auth_path,
        lock_file=_lock_sentinel(auth_path),
        **kwargs,
    )


def _http_error(status: int, payload: dict[str, Any]) -> urllib_error.HTTPError:
    return urllib_error.HTTPError(
        url="https://auth.kimi.com/api/oauth/token",
        code=status,
        msg="OAuth failure",
        hdrs=Message(),
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


def test_defaults_use_shared_kimi_code_credential_and_native_lock(kimi) -> None:
    assert kimi.DEFAULT_KIMI_OAUTH_AUTH_FILE == "~/.kimi-code/credentials/kimi-code.json"
    assert kimi.DEFAULT_KIMI_OAUTH_LOCK_FILE == "~/.kimi-code/oauth/kimi-code"
    assert kimi.DEFAULT_KIMI_OAUTH_CLIENT_ID == "17e5f671-d194-4dfb-9706-5516cb48c098"
    assert kimi.DEFAULT_KIMI_OAUTH_SCOPE == "kimi-code"
    assert kimi.DEFAULT_KIMI_OAUTH_TOKEN_ENDPOINT.endswith("/api/oauth/token")
    assert not hasattr(kimi, "login_kimi_oauth_service_credential")
    assert not hasattr(kimi, "DEFAULT_KIMI_OAUTH_DEVICE_AUTHORIZATION_ENDPOINT")
    assert "--login" not in kimi._build_argument_parser().format_help()


def test_refresh_threshold_is_450_seconds_for_900_second_lease(kimi) -> None:
    assert kimi._refresh_threshold_seconds(900) == 450
    assert kimi._refresh_threshold_seconds(10) == 300


def test_refresh_skips_when_lifetime_exceeds_dynamic_threshold(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(
        auth_path,
        _credential(expires_at=time.time() + 451, expires_in=900),
    )
    with patch.object(kimi.urllib_request, "urlopen") as urlopen:
        result = _refresh(kimi, auth_path)

    assert result["skipped"] is True
    assert result["attempted"] is False
    assert result["refreshed"] is False
    assert "logged_in" not in result
    assert result["auth_degraded"] is False
    urlopen.assert_not_called()


def test_force_refresh_rotates_refresh_token_and_never_returns_tokens(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(auth_path, _credential(expires_at=time.time() + 3_600))
    with patch.object(
        kimi.urllib_request,
        "urlopen",
        return_value=_Response(
            {
                "access_token": "new-access",
                "refresh_token": "rotated-refresh",
                "expires_in": 900,
                "scope": "kimi-code",
                "token_type": "Bearer",
            }
        ),
    ):
        result = _refresh(kimi, auth_path, force=True)

    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert result["refreshed"] is True
    assert "access_token" not in result
    assert "refresh_token" not in result
    assert persisted["access_token"] == "new-access"
    assert persisted["refresh_token"] == "rotated-refresh"
    assert set(persisted) == {
        "access_token",
        "refresh_token",
        "expires_at",
        "expires_in",
        "scope",
        "token_type",
    }


def test_native_lock_creates_sentinel_directory_and_cleans_up(kimi, tmp_path: Path) -> None:
    sentinel = tmp_path / "oauth" / "kimi-code"
    lock_path = Path(f"{sentinel}.lock")

    with kimi._kimi_code_lock(sentinel, retry_sleep=lambda _seconds: None) as lock:
        assert sentinel.is_file()
        assert lock.lock_path == lock_path
        assert lock_path.is_dir()

    assert sentinel.is_file()
    assert not lock_path.exists()


def test_active_native_lock_retries_then_fails_closed_with_injectable_timing(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(auth_path, _credential())
    sentinel = _lock_sentinel(auth_path)
    retry_delays: list[float] = []

    with kimi._kimi_code_lock(sentinel, retry_sleep=lambda _seconds: None):
        with patch.object(kimi.urllib_request, "urlopen") as urlopen:
            result = _refresh(
                kimi,
                auth_path,
                force=True,
                lock_retries=2,
                lock_retry_sleep=retry_delays.append,
            )

    assert retry_delays == [0.5, 0.5]
    assert result["attempted"] is False
    assert result["refreshed"] is False
    assert result["error_class"] == "KimiOAuthLockError"
    urlopen.assert_not_called()


def test_stale_native_lock_is_taken_over(kimi, tmp_path: Path) -> None:
    sentinel = tmp_path / "oauth" / "kimi-code"
    sentinel.parent.mkdir(parents=True)
    sentinel.touch()
    lock_path = Path(f"{sentinel}.lock")
    lock_path.mkdir()
    old_time = time.time() - 10
    os.utime(lock_path, (old_time, old_time))

    with kimi._kimi_code_lock(sentinel, retry_sleep=lambda _seconds: None):
        assert lock_path.is_dir()

    assert not lock_path.exists()


def test_heartbeat_updates_native_lock_mtime_during_refresh(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(auth_path, _credential())
    lock_path = Path(f"{_lock_sentinel(auth_path)}.lock")
    observed: dict[str, int] = {}

    def slow_response(*_args: Any, **_kwargs: Any) -> _Response:
        observed["before"] = lock_path.stat().st_mtime_ns
        time.sleep(0.05)
        observed["after"] = lock_path.stat().st_mtime_ns
        return _Response(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 900,
            }
        )

    with patch.object(kimi.urllib_request, "urlopen", side_effect=slow_response):
        result = _refresh(
            kimi,
            auth_path,
            force=True,
            lock_heartbeat_interval_seconds=0.01,
        )

    assert result["refreshed"] is True
    assert observed["after"] > observed["before"]
    assert not lock_path.exists()


def test_lock_ownership_change_fails_closed_before_credential_write(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    original = _credential()
    _write_credential(auth_path, original)
    lock_path = Path(f"{_lock_sentinel(auth_path)}.lock")

    def replace_lock(*_args: Any, **_kwargs: Any) -> _Response:
        lock_path.rmdir()
        lock_path.mkdir()
        return _Response(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 900,
            }
        )

    with patch.object(kimi.urllib_request, "urlopen", side_effect=replace_lock):
        result = _refresh(kimi, auth_path, force=True)

    assert result["refreshed"] is False
    assert result["error_class"] == "KimiOAuthLockOwnershipError"
    assert json.loads(auth_path.read_text(encoding="utf-8")) == original
    assert lock_path.is_dir()
    lock_path.rmdir()


def test_force_refresh_coalesces_peer_rotation_while_waiting(
    kimi, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(auth_path, _credential(expires_at=1))

    @contextmanager
    def peer_refresh_completed(*_args: Any, **_kwargs: Any) -> Iterator[Any]:
        _write_credential(
            auth_path,
            _credential(
                access_token="peer-access",
                refresh_token="peer-refresh",
                expires_at=time.time() + 600,
            ),
        )

        class _PeerLock:
            @staticmethod
            def assert_owned() -> None:
                return None

        yield _PeerLock()

    monkeypatch.setattr(kimi, "_kimi_code_lock", peer_refresh_completed)
    with patch.object(kimi.urllib_request, "urlopen") as urlopen:
        result = _refresh(kimi, auth_path, force=True)

    assert result["attempted"] is False
    assert result["skipped"] is True
    assert json.loads(auth_path.read_text(encoding="utf-8"))["access_token"] == "peer-access"
    urlopen.assert_not_called()


def test_refresh_retries_transport_equivalent_http_failures_at_most_three_times(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(auth_path, _credential())
    with patch.object(
        kimi.urllib_request,
        "urlopen",
        side_effect=[
            _http_error(503, {"error": "server_error"}),
            _http_error(429, {"error": "rate_limited"}),
            _Response(
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 900,
                }
            ),
        ],
    ) as urlopen:
        result = _refresh(
            kimi,
            auth_path,
            force=True,
            sleep=lambda _seconds: None,
        )

    assert result["refreshed"] is True
    assert urlopen.call_count == 3


def test_refresh_rejects_missing_rotated_refresh_token_without_corrupting_prior(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    original = _credential()
    _write_credential(auth_path, original)
    with patch.object(
        kimi.urllib_request,
        "urlopen",
        return_value=_Response(
            {
                "access_token": "new-access",
                "expires_in": 900,
            }
        ),
    ):
        result = _refresh(kimi, auth_path, force=True)

    assert result["refreshed"] is False
    assert result["auth_degraded"] is False
    assert "missing refresh_token" in (result["error_message"] or "")
    assert json.loads(auth_path.read_text(encoding="utf-8")) == original


def test_unauthorized_refresh_tombstones_when_refresh_token_did_not_rotate(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(auth_path, _credential())
    with patch.object(
        kimi.urllib_request,
        "urlopen",
        side_effect=_http_error(
            401,
            {
                "error": "invalid_grant",
                "error_description": "refresh_token=old-refresh rejected",
            },
        ),
    ) as urlopen:
        result = _refresh(kimi, auth_path, force=True, sleep=lambda _seconds: None)

    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert urlopen.call_count == 1
    assert result["refreshed"] is False
    assert result["auth_degraded"] is True
    assert "old-refresh" not in (result["error_message"] or "")
    assert "refresh_token=[REDACTED]" in (result["error_message"] or "")
    assert persisted == {
        "access_token": "",
        "refresh_token": "",
        "expires_at": 0,
        "expires_in": 0,
        "scope": "kimi-code",
        "token_type": "Bearer",
    }


def test_unauthorized_refresh_accepts_peer_rotation_instead_of_tombstoning(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(auth_path, _credential())
    sleep_delays: list[float] = []

    def peer_rotation_then_rejection(*_args: Any, **_kwargs: Any) -> _Response:
        _write_credential(
            auth_path,
            _credential(
                access_token="peer-access",
                refresh_token="peer-refresh",
                expires_at=time.time() + 900,
            ),
        )
        raise _http_error(401, {"error": "invalid_grant"})

    with patch.object(
        kimi.urllib_request,
        "urlopen",
        side_effect=peer_rotation_then_rejection,
    ):
        result = _refresh(
            kimi,
            auth_path,
            force=True,
            sleep=sleep_delays.append,
        )

    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert sleep_delays == [0.1]
    assert result["attempted"] is True
    assert result["refreshed"] is False
    assert result["skipped"] is True
    assert result["auth_degraded"] is False
    assert persisted["access_token"] == "peer-access"
    assert persisted["refresh_token"] == "peer-refresh"


def test_non_auth_failure_is_redacted_without_corrupting_prior_credential(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    original = _credential()
    _write_credential(auth_path, original)
    with patch.object(
        kimi.urllib_request,
        "urlopen",
        side_effect=_http_error(
            400,
            {
                "error": "invalid_request",
                "error_description": "access_token=leaked-access was malformed",
            },
        ),
    ):
        result = _refresh(kimi, auth_path, force=True)

    assert result["auth_degraded"] is False
    assert "leaked-access" not in (result["error_message"] or "")
    assert "access_token=[REDACTED]" in (result["error_message"] or "")
    assert json.loads(auth_path.read_text(encoding="utf-8")) == original


def test_atomic_writer_preserves_private_mode_and_metadata(kimi, tmp_path: Path) -> None:
    auth_path = tmp_path / "credentials" / "kimi-code.json"
    _write_credential(auth_path, _credential())
    os.chmod(auth_path, 0o600)

    kimi._write_credential_payload(auth_path, _credential(access_token="new-access"))

    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(auth_path.read_text(encoding="utf-8"))["access_token"] == "new-access"
    assert list(auth_path.parent.glob(f".{auth_path.name}.*.tmp")) == []
