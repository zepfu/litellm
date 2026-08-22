"""Focused coverage for the Hermes Nous Portal OAuth credential refresher.

Lock-path decision: public Hermes uses ``Path.with_suffix(".lock")`` on the
auth file, so ``auth.json`` locks as ``auth.lock`` (not ``auth.json.lock``).
Tests lock whatever path the helper actually uses and require the Hermes-native
``auth.lock`` default. Do not require both lock files.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib import error as urllib_error
from urllib.parse import parse_qs

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "nous_oauth_refresh.py"
_LOOP_SCRIPT = _REPO_ROOT / "scripts" / "run_provider_status_observations_loop.py"

_TOKEN_ENDPOINT = "https://portal.nousresearch.com/api/oauth/token"
_CLIENT_ID = "hermes-cli"
_SCOPE = "inference:invoke"
_INFERENCE_BASE = "https://inference-api.nousresearch.com/v1"
_PORTAL_BASE = "https://portal.nousresearch.com"
_REFRESH_HEADER = "x-nous-refresh-token"


def _load_module():
    if not _SCRIPT.is_file() or _SCRIPT.stat().st_size == 0:
        pytest.fail(
            "scripts/nous_oauth_refresh.py is missing or empty; Wave 1 helper is required"
        )
    name = "nous_oauth_refresh_test"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def nous():
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


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _nous_record(
    *,
    access_token: str = "old-access",
    refresh_token: str = "old-refresh",
    agent_key: str = "old-agent-key",
    expires_at: str | None = "2099-01-01T00:00:00Z",
    agent_key_expires_at: str | None = "2099-01-01T00:00:00Z",
    **extra: Any,
) -> dict[str, Any]:
    record = {
        "access_token": access_token,
        "agent_key": agent_key,
        "agent_key_expires_at": agent_key_expires_at,
        "agent_key_expires_in": 3600,
        "agent_key_id": "akid-keep-me",
        "agent_key_obtained_at": "2026-01-01T00:00:00Z",
        "agent_key_reused": True,
        "client_id": _CLIENT_ID,
        "expires_at": expires_at,
        "expires_in": 3600,
        "inference_base_url": _INFERENCE_BASE,
        "obtained_at": "2026-01-01T00:00:00Z",
        "portal_base_url": _PORTAL_BASE,
        "refresh_token": refresh_token,
        "scope": _SCOPE,
        "tls": {"ca_bundle": None, "insecure": False},
        "token_type": "Bearer",
    }
    record.update(extra)
    return record


def _nous_pool_entry(record: dict[str, Any], **extra: Any) -> dict[str, Any]:
    entry = {
        "auth_type": "oauth",
        "id": "nous-device-1",
        "label": "device_code",
        "last_error_code": "prior-code",
        "last_error_message": "prior-message",
        "last_error_reason": "prior-reason",
        "last_error_reset_at": None,
        "last_status": "ok",
        "last_status_at": "2026-01-01T00:00:00Z",
        "priority": 10,
        "request_count": 7,
        "source": "device_code",
        **record,
    }
    entry.update(extra)
    return entry


def _hermes_document(
    *,
    nous_record: dict[str, Any] | None = None,
    nous_pool: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    record = nous_record if nous_record is not None else _nous_record()
    pool = nous_pool if nous_pool is not None else [_nous_pool_entry(record)]
    return {
        "version": 1,
        "active_provider": "xai-oauth",
        "providers": {
            "xai-oauth": {
                "access_token": "xai-unrelated-access",
                "refresh_token": "xai-unrelated-refresh",
                "expires_at": "2099-01-01T00:00:00Z",
            },
            "nous": record,
        },
        "credential_pool": {
            "xai-oauth": [
                {
                    "id": "xai-pool-1",
                    "label": "xai-oauth",
                    "access_token": "xai-pool-unrelated-access",
                    "refresh_token": "xai-pool-unrelated-refresh",
                }
            ],
            "copilot": [
                {
                    "id": "copilot-1",
                    "label": "copilot",
                    "access_token": "copilot-unrelated-access",
                }
            ],
            "nous": pool,
        },
        "updated_at": "2026-01-01T00:00:00Z",
    }


def _write_hermes(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _lock_path_for(nous, auth_path: Path) -> Path:
    """Lock the helper's real default name; prefer Hermes-native auth.lock."""
    default_name = Path(nous.DEFAULT_NOUS_OAUTH_LOCK_FILE).name
    return auth_path.parent / default_name


def _refresh(nous, auth_path: Path, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("lock_file", _lock_path_for(nous, auth_path))
    return nous.refresh_nous_oauth_auth_file(auth_path, **kwargs)


def _header_map(request: Any) -> dict[str, str]:
    headers: dict[str, str] = {}
    header_items = getattr(request, "header_items", None)
    if callable(header_items):
        for key, value in header_items():
            headers[str(key).lower()] = str(value)
    raw_headers = getattr(request, "headers", None)
    if isinstance(raw_headers, dict):
        for key, value in raw_headers.items():
            headers[str(key).lower()] = str(value)
    return headers


def _http_error(status: int, payload: dict[str, Any]) -> urllib_error.HTTPError:
    return urllib_error.HTTPError(
        url=_TOKEN_ENDPOINT,
        code=status,
        msg="OAuth failure",
        hdrs=Message(),
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


def _token_response(**extra: Any) -> _Response:
    payload = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": _SCOPE,
    }
    payload.update(extra)
    return _Response(payload)


def test_defaults_are_portable_tilde_paths(nous) -> None:
    assert nous.DEFAULT_NOUS_OAUTH_AUTH_FILE == "~/.hermes/auth.json"
    assert nous.DEFAULT_NOUS_OAUTH_AUTH_FILE.startswith("~/")
    assert nous.DEFAULT_NOUS_OAUTH_LOCK_FILE.startswith("~/")
    assert "/home/zepfu" not in nous.DEFAULT_NOUS_OAUTH_AUTH_FILE
    assert "/home/zepfu" not in nous.DEFAULT_NOUS_OAUTH_LOCK_FILE
    assert nous.DEFAULT_NOUS_OAUTH_TOKEN_ENDPOINT == _TOKEN_ENDPOINT
    assert nous.DEFAULT_NOUS_OAUTH_CLIENT_ID == _CLIENT_ID
    assert nous.DEFAULT_NOUS_OAUTH_REFRESH_BUFFER_SECONDS == 900
    assert nous.DEFAULT_NOUS_OAUTH_HTTP_TIMEOUT_SECONDS == 30.0
    assert nous.DEFAULT_NOUS_OAUTH_AUTH_FILE_MODE == 0o600
    assert getattr(nous, "DEFAULT_NOUS_OAUTH_FORCE_REFRESH", False) is False
    assert nous.NOUS_REFRESH_TOKEN_HEADER == _REFRESH_HEADER
    interval = getattr(nous, "DEFAULT_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS", 300)
    assert int(interval) == 300


def test_default_lock_path_is_hermes_native_auth_lock(nous) -> None:
    hermes_native = Path("auth.json").with_suffix(".lock")
    assert hermes_native.name == "auth.lock"
    assert Path(nous.DEFAULT_NOUS_OAUTH_LOCK_FILE).name == "auth.lock"
    assert not nous.DEFAULT_NOUS_OAUTH_LOCK_FILE.endswith("auth.json.lock")
    src = Path(nous.__file__).read_text(encoding="utf-8")
    assert "with_suffix" in src or Path(nous.DEFAULT_NOUS_OAUTH_LOCK_FILE).name == "auth.lock"


def test_default_paths_expanduser(nous) -> None:
    auth = Path(nous.DEFAULT_NOUS_OAUTH_AUTH_FILE).expanduser()
    lock = Path(nous.DEFAULT_NOUS_OAUTH_LOCK_FILE).expanduser()
    assert str(auth).startswith(str(Path.home()))
    assert str(lock).startswith(str(Path.home()))
    assert "~" not in str(auth)
    assert auth.name == "auth.json"
    assert lock.name == "auth.lock"


def test_lock_wrapper_delegates_only_to_shared_helper(nous) -> None:
    src = Path(nous.__file__).read_text(encoding="utf-8")
    assert "from litellm.secret_managers.credential_file_lock import" in src
    assert "from litellm.secret_managers.credential_file_metadata import" in src
    assert "from litellm.secret_managers.credential_file_write import" in src
    assert "write_and_publish_private_text" in src
    assert "import fcntl" not in src
    assert "fcntl.flock" not in src
    assert "LOCK_EX" not in src
    assert "LOCK_UN" not in src
    assert "time.monotonic_ns()" not in src
    assert 'f".{auth_path.name}.{os.getpid()}' not in src
    assert "os.replace(tmp_path" not in src
    assert "sort_keys=True" not in src
    assert "refuse_symlink=True" in src
    assert src.count("with credential_file_lock(") >= 1


def test_refresh_uses_helper_lock_path_and_does_not_require_both(
    nous, tmp_path: Path
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_hermes(auth_path, _hermes_document(nous_record=_nous_record(expires_at=None)))
    lock_path = _lock_path_for(nous, auth_path)
    seen: list[Path] = []
    real_lock = nous.credential_file_lock

    def spy(path):  # noqa: ANN001
        seen.append(Path(path))
        return real_lock(path)

    with patch.object(nous, "credential_file_lock", side_effect=spy):
        with patch.object(nous.urllib_request, "urlopen", return_value=_token_response()):
            result = _refresh(nous, auth_path, force=True)

    assert result["refreshed"] is True
    assert seen
    assert seen[0] == lock_path
    assert lock_path.name == Path(nous.DEFAULT_NOUS_OAUTH_LOCK_FILE).name
    sibling_names = {path.name for path in auth_path.parent.glob("auth*.lock")}
    assert lock_path.name in sibling_names
    assert not {"auth.lock", "auth.json.lock"} <= sibling_names


def test_eligibility_not_due_does_not_call_token_endpoint_or_throttle(
    nous, tmp_path: Path
) -> None:
    auth_path = tmp_path / "auth.json"
    future = _iso_z(datetime.now(timezone.utc) + timedelta(hours=2))
    original = _hermes_document(
        nous_record=_nous_record(expires_at=future, agent_key_expires_at=future)
    )
    _write_hermes(auth_path, original)
    attempts: list[int] = []

    with patch.object(nous.urllib_request, "urlopen") as urlopen:
        result = _refresh(
            nous,
            auth_path,
            force=False,
            buffer_seconds=900,
            on_token_endpoint_attempt=lambda: attempts.append(1),
        )

    assert result["skipped"] is True
    assert result["attempted"] is False
    assert result["refreshed"] is False
    assert attempts == []
    urlopen.assert_not_called()
    assert json.loads(auth_path.read_text(encoding="utf-8")) == original


def test_inspect_eligibility_not_due_is_read_only(nous, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    future = _iso_z(datetime.now(timezone.utc) + timedelta(hours=2))
    original = _hermes_document(
        nous_record=_nous_record(expires_at=future, agent_key_expires_at=future)
    )
    _write_hermes(auth_path, original)
    before = auth_path.read_bytes()
    before_stat = auth_path.stat()

    with patch.object(nous, "credential_file_lock") as lock:
        with patch.object(nous.urllib_request, "urlopen") as urlopen:
            summary = nous.inspect_nous_oauth_refresh_eligibility(
                auth_path,
                buffer_seconds=900,
            )

    assert summary["eligible"] is False
    assert summary["credential_health"] == "fresh"
    lock.assert_not_called()
    urlopen.assert_not_called()
    assert auth_path.read_bytes() == before
    assert auth_path.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_missing_expires_at_is_eligible_and_degraded(nous, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_hermes(
        auth_path,
        _hermes_document(
            nous_record=_nous_record(expires_at=None, agent_key_expires_at=None)
        ),
    )

    summary = nous.inspect_nous_oauth_refresh_eligibility(
        auth_path,
        buffer_seconds=900,
    )

    assert summary["eligible"] is True
    assert summary["credential_health"] == "degraded"
    assert summary["usable"] is True


def test_unparseable_expires_at_is_eligible_and_degraded(nous, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_hermes(
        auth_path,
        _hermes_document(
            nous_record=_nous_record(
                expires_at="not-a-date",
                agent_key_expires_at="also-bad",
            )
        ),
    )

    summary = nous.inspect_nous_oauth_refresh_eligibility(
        auth_path,
        buffer_seconds=900,
    )

    assert summary["eligible"] is True
    assert summary["credential_health"] == "degraded"


def test_eligibility_uses_earlier_of_access_and_agent_key_expiry(
    nous, tmp_path: Path
) -> None:
    auth_path = tmp_path / "auth.json"
    now = datetime.now(timezone.utc)
    access_expiry = _iso_z(now + timedelta(hours=2))
    agent_expiry = _iso_z(now + timedelta(seconds=60))
    _write_hermes(
        auth_path,
        _hermes_document(
            nous_record=_nous_record(
                expires_at=access_expiry,
                agent_key_expires_at=agent_expiry,
            )
        ),
    )

    summary = nous.inspect_nous_oauth_refresh_eligibility(
        auth_path,
        buffer_seconds=900,
        now=lambda: now,
    )

    assert summary["eligible"] is True
    assert summary["expires_at"] == agent_expiry


def test_due_refresh_posts_portal_form_and_refresh_header(
    nous, tmp_path: Path
) -> None:
    auth_path = tmp_path / "auth.json"
    original = _hermes_document(
        nous_record=_nous_record(
            expires_at="2020-01-01T00:00:00Z",
            agent_key_expires_at="2020-01-01T00:00:00Z",
        )
    )
    _write_hermes(auth_path, original)
    attempts: list[int] = []

    with patch.object(
        nous.urllib_request, "urlopen", return_value=_token_response()
    ) as urlopen:
        result = _refresh(
            nous,
            auth_path,
            force=False,
            buffer_seconds=900,
            on_token_endpoint_attempt=lambda: attempts.append(1),
        )

    assert result["refreshed"] is True
    assert result["attempted"] is True
    assert attempts == [1]
    assert "access_token" not in result
    assert "refresh_token" not in result
    assert "agent_key" not in result
    assert urlopen.call_count == 1
    request = urlopen.call_args[0][0]
    assert request.full_url == _TOKEN_ENDPOINT
    assert request.get_method() == "POST"
    timeout = urlopen.call_args.kwargs.get("timeout")
    if timeout is None and len(urlopen.call_args.args) > 1:
        timeout = urlopen.call_args.args[1]
    assert timeout == 30.0 or timeout == 30
    headers = _header_map(request)
    assert _REFRESH_HEADER in headers
    assert headers[_REFRESH_HEADER] == "old-refresh"
    form = parse_qs(request.data.decode("utf-8"))
    assert form["grant_type"] == ["refresh_token"]
    assert form["client_id"] == [_CLIENT_ID]
    assert "refresh_token" not in form


def test_refresh_updates_nous_provider_and_matching_pool_entry(
    nous, tmp_path: Path
) -> None:
    auth_path = tmp_path / "auth.json"
    original = _hermes_document(
        nous_record=_nous_record(
            expires_at="2020-01-01T00:00:00Z",
            agent_key_expires_at="2020-01-01T00:00:00Z",
        )
    )
    _write_hermes(auth_path, original)

    with patch.object(nous.urllib_request, "urlopen", return_value=_token_response()):
        result = _refresh(nous, auth_path, force=True)

    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    nous_provider = persisted["providers"]["nous"]
    nous_pool = persisted["credential_pool"]["nous"][0]
    assert result["refreshed"] is True
    assert nous_provider["access_token"] == "new-access"
    assert nous_provider["refresh_token"] == "new-refresh"
    assert nous_provider["agent_key"] == "new-access"
    assert nous_provider["agent_key_id"] == "akid-keep-me"
    assert nous_provider["agent_key_reused"] is False
    assert nous_provider["inference_base_url"] == _INFERENCE_BASE
    assert nous_provider["expires_at"] == nous_provider["agent_key_expires_at"]
    assert nous_provider["expires_in"] == 3600
    assert nous_provider["agent_key_expires_in"] == 3600
    for field in (
        "access_token",
        "refresh_token",
        "agent_key",
        "expires_at",
        "expires_in",
        "agent_key_expires_at",
        "agent_key_expires_in",
    ):
        assert nous_pool[field] == nous_provider[field]
    assert nous_pool["id"] == "nous-device-1"
    assert nous_pool["label"] == "device_code"
    assert nous_pool["priority"] == 10
    assert nous_pool["request_count"] == 7
    assert nous_pool["auth_type"] == "oauth"
    assert nous_pool["source"] == "device_code"
    assert nous_pool["last_error_code"] == "prior-code"
    assert nous_pool["last_error_message"] == "prior-message"
    assert nous_pool["last_status"] == "ok"


def test_refresh_preserves_unrelated_hermes_fields(nous, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    original = _hermes_document(
        nous_record=_nous_record(
            expires_at="2020-01-01T00:00:00Z",
            agent_key_expires_at="2020-01-01T00:00:00Z",
        )
    )
    _write_hermes(auth_path, original)

    with patch.object(nous.urllib_request, "urlopen", return_value=_token_response()):
        _refresh(nous, auth_path, force=True)

    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert persisted["active_provider"] == "xai-oauth"
    assert persisted["active_provider"] != "nous"
    assert persisted["providers"]["xai-oauth"] == original["providers"]["xai-oauth"]
    assert persisted["credential_pool"]["copilot"] == original["credential_pool"]["copilot"]
    assert (
        persisted["credential_pool"]["xai-oauth"]
        == original["credential_pool"]["xai-oauth"]
    )
    assert persisted["version"] == original["version"]
    assert list(persisted.keys()) == list(original.keys())
    assert list(persisted["providers"].keys()) == list(original["providers"].keys())
    assert list(persisted["credential_pool"].keys()) == list(
        original["credential_pool"].keys()
    )
    assert json.dumps(persisted["providers"]["xai-oauth"], sort_keys=True) == json.dumps(
        original["providers"]["xai-oauth"], sort_keys=True
    )
    assert json.dumps(persisted["credential_pool"]["copilot"], sort_keys=True) == json.dumps(
        original["credential_pool"]["copilot"], sort_keys=True
    )
    assert json.dumps(
        persisted["credential_pool"]["xai-oauth"], sort_keys=True
    ) == json.dumps(original["credential_pool"]["xai-oauth"], sort_keys=True)


def test_refresh_keeps_unrotated_refresh_token(nous, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_hermes(
        auth_path,
        _hermes_document(
            nous_record=_nous_record(
                expires_at="2020-01-01T00:00:00Z",
                agent_key_expires_at="2020-01-01T00:00:00Z",
            )
        ),
    )

    with patch.object(
        nous.urllib_request,
        "urlopen",
        return_value=_Response(
            {
                "access_token": "new-access",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": _SCOPE,
            }
        ),
    ):
        result = _refresh(nous, auth_path, force=True)

    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert result["refreshed"] is True
    assert persisted["providers"]["nous"]["refresh_token"] == "old-refresh"
    assert persisted["credential_pool"]["nous"][0]["refresh_token"] == "old-refresh"


def test_published_file_uses_env_uid_gid_and_private_mode(
    nous, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_hermes(
        auth_path,
        _hermes_document(
            nous_record=_nous_record(
                expires_at="2020-01-01T00:00:00Z",
                agent_key_expires_at="2020-01-01T00:00:00Z",
            )
        ),
    )
    chown_calls: list[tuple[int, int]] = []

    def fake_chown(target, uid, gid, *args, **kwargs):  # noqa: ANN001
        chown_calls.append((uid, gid))

    monkeypatch.setenv("AAWM_NOUS_OAUTH_AUTH_FILE_UID", "1000")
    monkeypatch.setenv("AAWM_NOUS_OAUTH_AUTH_FILE_GID", "1000")
    monkeypatch.setenv("AAWM_NOUS_OAUTH_AUTH_FILE_MODE", "0o600")
    monkeypatch.setattr(
        "litellm.secret_managers.credential_file_metadata.os.chown",
        fake_chown,
    )

    with patch.object(nous.urllib_request, "urlopen", return_value=_token_response()):
        result = _refresh(nous, auth_path, force=True)

    assert result["refreshed"] is True
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert chown_calls
    assert chown_calls[-1] == (1000, 1000)
    leftovers = list(auth_path.parent.glob(f".{auth_path.name}.*.tmp"))
    assert leftovers == []


def test_write_payload_uses_shared_publish_not_pid_only_temp(
    nous, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm.secret_managers.credential_file_write as write_mod

    auth_path = tmp_path / "auth.json"
    payload = _hermes_document()
    seen_temps: list[Path] = []
    real_write_temp = write_mod.write_private_temp_file_text

    def spy_temp(final_path, content, **kwargs):  # noqa: ANN001
        tmp = real_write_temp(final_path, content, **kwargs)
        seen_temps.append(tmp)
        return tmp

    monkeypatch.setattr(write_mod, "write_private_temp_file_text", spy_temp)
    nous._write_credential_payload(auth_path, payload)
    assert seen_temps
    temp_name = seen_temps[0].name
    assert temp_name.startswith(f".{auth_path.name}.")
    assert temp_name.endswith(".tmp")
    assert str(os.getpid()) in temp_name
    parts = temp_name.split(".")
    assert any(len(part) >= 16 and all(ch in "0123456789abcdef" for ch in part) for part in parts)
    leftovers = list(tmp_path.glob(f".{auth_path.name}.*.tmp"))
    assert leftovers == []
    assert auth_path.stat().st_mode & 0o777 == 0o600


def test_http_200_then_write_failure_does_not_retry_token_endpoint(
    nous, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_path = tmp_path / "auth.json"
    original = _hermes_document(
        nous_record=_nous_record(
            expires_at="2020-01-01T00:00:00Z",
            agent_key_expires_at="2020-01-01T00:00:00Z",
        )
    )
    _write_hermes(auth_path, original)
    attempts: list[int] = []

    def fail_write(*_args: Any, **_kwargs: Any):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(nous, "write_and_publish_private_text", fail_write)

    with patch.object(
        nous.urllib_request, "urlopen", return_value=_token_response()
    ) as urlopen:
        result = _refresh(
            nous,
            auth_path,
            force=True,
            on_token_endpoint_attempt=lambda: attempts.append(1),
        )

    assert urlopen.call_count == 1
    assert attempts == [1]
    assert result["refreshed"] is False
    assert result["attempted"] is True
    assert result.get("skipped") is False
    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert persisted["providers"]["nous"]["refresh_token"] == "old-refresh"


@pytest.mark.parametrize(
    "error_code",
    ["invalid_grant", "refresh_token_reused"],
)
def test_terminal_oauth_errors_are_sanitized_without_secret_leak(
    nous, tmp_path: Path, error_code: str
) -> None:
    auth_path = tmp_path / "auth.json"
    original = _hermes_document(
        nous_record=_nous_record(
            expires_at="2020-01-01T00:00:00Z",
            agent_key_expires_at="2020-01-01T00:00:00Z",
        )
    )
    _write_hermes(auth_path, original)

    with patch.object(
        nous.urllib_request,
        "urlopen",
        side_effect=_http_error(
            400,
            {
                "error": error_code,
                "error_description": (
                    "refresh_token=old-refresh agent_key=old-agent-key "
                    "access_token=old-access rejected"
                ),
            },
        ),
    ) as urlopen:
        result = _refresh(nous, auth_path, force=True)

    assert urlopen.call_count == 1
    assert result["refreshed"] is False
    assert result["attempted"] is True
    message = result["error_message"] or ""
    rendered = json.dumps(result)
    assert "old-refresh" not in message
    assert "old-agent-key" not in message
    assert "old-access" not in message
    assert "old-refresh" not in rendered
    assert "old-agent-key" not in rendered
    assert error_code in message or error_code in (result.get("error_class") or "")
    assert json.loads(auth_path.read_text(encoding="utf-8"))["providers"]["nous"][
        "refresh_token"
    ] == "old-refresh"


def test_on_token_endpoint_attempt_fires_only_on_actual_post(
    nous, tmp_path: Path
) -> None:
    auth_path = tmp_path / "auth.json"
    future = _iso_z(datetime.now(timezone.utc) + timedelta(hours=2))
    _write_hermes(
        auth_path,
        _hermes_document(
            nous_record=_nous_record(expires_at=future, agent_key_expires_at=future)
        ),
    )
    attempts: list[str] = []

    with patch.object(nous.urllib_request, "urlopen") as urlopen:
        skipped = _refresh(
            nous,
            auth_path,
            force=False,
            on_token_endpoint_attempt=lambda: attempts.append("skip"),
        )
    assert skipped["skipped"] is True
    urlopen.assert_not_called()
    assert attempts == []

    with patch.object(
        nous.urllib_request, "urlopen", return_value=_token_response()
    ) as urlopen:
        refreshed = _refresh(
            nous,
            auth_path,
            force=True,
            on_token_endpoint_attempt=lambda: attempts.append("post"),
        )

    assert refreshed["refreshed"] is True
    assert attempts == ["post"]
    urlopen.assert_called_once()


def test_loop_nous_oauth_help_defaults_are_portable() -> None:
    if not _LOOP_SCRIPT.is_file() or _LOOP_SCRIPT.stat().st_size == 0:
        pytest.fail(
            "scripts/run_provider_status_observations_loop.py is missing or empty"
        )
    name = "run_provider_status_observations_loop_nous_help"
    spec = importlib.util.spec_from_file_location(name, _LOOP_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    parser = mod._build_parser()
    help_by_dest = {
        action.dest: (action.help or "")
        for action in parser._actions
        if getattr(action, "dest", None)
    }
    auth_help = help_by_dest["nous_oauth_auth_file"]
    lock_help = help_by_dest["nous_oauth_lock_file"]
    assert "~/.hermes/auth.json" in auth_help
    assert "~/.hermes/auth.lock" in lock_help
    assert "/home/zepfu" not in auth_help
    assert "/home/zepfu" not in lock_help
    assert mod.DEFAULT_NOUS_OAUTH_AUTH_FILE.startswith("~/")
    assert mod.DEFAULT_NOUS_OAUTH_LOCK_FILE.startswith("~/")
    assert Path(mod.DEFAULT_NOUS_OAUTH_LOCK_FILE).name == "auth.lock"
    assert "/home/zepfu" not in mod.DEFAULT_NOUS_OAUTH_AUTH_FILE
    assert "/home/zepfu" not in mod.DEFAULT_NOUS_OAUTH_LOCK_FILE
    interval = getattr(mod, "DEFAULT_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS", 300.0)
    assert float(interval) in {300.0, 3600.0}
    compose = (_REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
    assert (
        "AAWM_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS="
        "${AAWM_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS:-300}"
    ) in compose
    if float(interval) != 300.0:
        pytest.fail(
            "loop DEFAULT_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS is still "
            f"{interval}; compose override is 300. Engineer freeze target is 300."
        )


def test_sidecar_buffer_and_attempt_interval_defaults(nous) -> None:
    assert nous.DEFAULT_NOUS_OAUTH_REFRESH_BUFFER_SECONDS == 900
    interval = getattr(nous, "DEFAULT_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS", 300)
    assert int(interval) == 300
    if not _LOOP_SCRIPT.is_file() or _LOOP_SCRIPT.stat().st_size == 0:
        pytest.fail(
            "scripts/run_provider_status_observations_loop.py is missing or empty"
        )
    src = _LOOP_SCRIPT.read_text(encoding="utf-8")
    assert "DEFAULT_NOUS_OAUTH_REFRESH_BUFFER_SECONDS" in src or (
        "nous_oauth_refresh.DEFAULT_NOUS_OAUTH_REFRESH_BUFFER_SECONDS" in src
    )
    assert "AAWM_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS" in src
    assert "AAWM_NOUS_OAUTH_REFRESH_BUFFER_SECONDS" in src
    compose = (_REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
    assert (
        "AAWM_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS="
        "${AAWM_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS:-300}"
    ) in compose
    assert (
        "AAWM_NOUS_OAUTH_REFRESH_BUFFER_SECONDS="
        "${AAWM_NOUS_OAUTH_REFRESH_BUFFER_SECONDS:-900}"
    ) in compose
    assert (
        "AAWM_NOUS_OAUTH_FORCE_REFRESH="
        "${AAWM_NOUS_OAUTH_FORCE_REFRESH:-0}"
    ) in compose
