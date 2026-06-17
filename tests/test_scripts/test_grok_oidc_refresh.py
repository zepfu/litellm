from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import error as urllib_error

import pytest

from scripts import grok_oidc_refresh as refresh


def _scoped_payload(
    *,
    token: str = "old-access-token",
    refresh_token: str = "old-refresh-token",
    expires_at: datetime,
    scope: str = refresh.DEFAULT_GROK_OIDC_SCOPE,
) -> dict:
    return {
        scope: {
            "key": token,
            "access_token": token,
            "refresh_token": refresh_token,
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
            "oidc_client_id": "client-id",
            "token_endpoint": "https://auth.test/token",
        }
    }


def test_refresh_grok_oidc_auth_file_skips_when_credential_is_still_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            _scoped_payload(
                expires_at=datetime.now(timezone.utc) + timedelta(hours=2)
            )
        ),
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("token endpoint should not be called")

    monkeypatch.setattr(refresh.urllib_request, "urlopen", fail_if_called)

    summary = refresh.refresh_grok_oidc_auth_file(auth_path, buffer_seconds=300)

    assert summary == {
        "attempted": False,
        "refreshed": False,
        "skipped": True,
        "auth_file": str(auth_path),
        "scope": refresh.DEFAULT_GROK_OIDC_SCOPE,
        "expires_at": json.loads(auth_path.read_text(encoding="utf-8"))[
            refresh.DEFAULT_GROK_OIDC_SCOPE
        ]["expires_at"],
        "error_class": None,
        "error_message": None,
    }


def test_refresh_grok_oidc_auth_file_refreshes_near_expiry_and_sanitizes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            _scoped_payload(
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=30)
            )
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, body: str) -> None:
            self._body = body.encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=30.0):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["data"] = request.data.decode("utf-8")
        return FakeResponse(
            json.dumps(
                {
                    "access_token": "new-access-token",
                    "refresh_token": "new-refresh-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                }
            )
        )

    monkeypatch.setattr(refresh.urllib_request, "urlopen", fake_urlopen)

    summary = refresh.refresh_grok_oidc_auth_file(auth_path, buffer_seconds=300)

    refreshed = json.loads(auth_path.read_text(encoding="utf-8"))[
        refresh.DEFAULT_GROK_OIDC_SCOPE
    ]
    assert refreshed["key"] == "new-access-token"
    assert refreshed["access_token"] == "new-access-token"
    assert refreshed["refresh_token"] == "new-refresh-token"
    assert refreshed["token_type"] == "Bearer"
    assert summary["attempted"] is True
    assert summary["refreshed"] is True
    assert summary["skipped"] is False
    assert summary["expires_at"] == refreshed["expires_at"]
    assert "new-access-token" not in json.dumps(summary)
    assert "new-refresh-token" not in json.dumps(summary)
    assert captured["url"] == "https://auth.test/token"
    assert "grant_type=refresh_token" in str(captured["data"])
    assert "client_id=client-id" in str(captured["data"])
    if os.name != "nt":
        assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600


def test_refresh_grok_oidc_auth_file_force_true_refreshes_even_when_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            _scoped_payload(
                expires_at=datetime.now(timezone.utc) + timedelta(hours=2)
            )
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "access_token": "forced-access-token",
                    "expires_in": 1800,
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(refresh.urllib_request, "urlopen", lambda *_a, **_k: FakeResponse())

    summary = refresh.refresh_grok_oidc_auth_file(auth_path, force=True)

    refreshed = json.loads(auth_path.read_text(encoding="utf-8"))[
        refresh.DEFAULT_GROK_OIDC_SCOPE
    ]
    assert summary["refreshed"] is True
    assert refreshed["access_token"] == "forced-access-token"


def test_refresh_grok_oidc_auth_file_missing_refresh_token_does_not_leak_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                refresh.DEFAULT_GROK_OIDC_SCOPE: {
                    "key": "old-access-token",
                    "access_token": "old-access-token",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(seconds=30)
                    )
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "oidc_client_id": "client-id",
                }
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("token endpoint should not be called")

    monkeypatch.setattr(refresh.urllib_request, "urlopen", fail_if_called)

    summary = refresh.refresh_grok_oidc_auth_file(auth_path)

    assert summary["refreshed"] is False
    assert summary["error_class"] == "ValueError"
    assert "refresh_token" in (summary["error_message"] or "")
    assert "old-access-token" not in json.dumps(summary)


def test_refresh_grok_oidc_auth_file_missing_client_id_does_not_leak_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                refresh.DEFAULT_GROK_OIDC_SCOPE: {
                    "key": "old-access-token",
                    "access_token": "old-access-token",
                    "refresh_token": "old-refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(seconds=30)
                    )
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("token endpoint should not be called")

    monkeypatch.setattr(refresh.urllib_request, "urlopen", fail_if_called)

    summary = refresh.refresh_grok_oidc_auth_file(auth_path)

    assert summary["refreshed"] is False
    assert summary["error_class"] == "ValueError"
    assert "oidc_client_id" in (summary["error_message"] or "")
    assert "old-refresh-token" not in json.dumps(summary)


def test_refresh_grok_oidc_auth_file_sanitizes_http_error_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            _scoped_payload(
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=30)
            )
        ),
        encoding="utf-8",
    )

    class FakeHTTPError(urllib_error.HTTPError):
        def read(self) -> bytes:
            return json.dumps(
                {
                    "error": "invalid_grant",
                    "error_description": "refresh_token=super-secret-token",
                }
            ).encode("utf-8")

    def fake_urlopen(*_args, **_kwargs):
        raise FakeHTTPError(
            url="https://auth.test/token",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(refresh.urllib_request, "urlopen", fake_urlopen)

    summary = refresh.refresh_grok_oidc_auth_file(auth_path)

    assert summary["refreshed"] is False
    assert summary["error_class"] == "ValueError"
    assert "invalid_grant" in (summary["error_message"] or "")
    assert "super-secret-token" not in json.dumps(summary)
