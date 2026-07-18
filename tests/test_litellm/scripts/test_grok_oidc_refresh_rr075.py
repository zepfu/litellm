"""RR-075 residuals for scripts/grok_oidc_refresh.py.

Covers consumer migration onto shared credential write/metadata APIs:
1/4/5. Lock delegates solely to shared credential_file_lock (no dead fcntl copy);
       flock failures surface as warnings via the shared helper.
2.     Credential publish uses write_and_publish_private_text (private exclusive
       temp, non-pid-only name, symlink refusal, shared metadata apply).
3.     Metadata helpers delegate to shared credential_file_metadata.
6.     Missing expires_at fails safe toward refresh.
7.     Error sanitizer redacts secret values and bounds output length.
       Also covers consumer-level quoted Authorization Bearer forms and
       even-backslash quoted secret suffix cases via the shared sanitizer path.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "grok_oidc_refresh.py"


def _load_module():
    name = "grok_oidc_refresh_rr075"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def grok():
    return _load_module()


def _scoped_payload(
    grok,
    *,
    token: str = "old-access-token",
    refresh_token: str = "old-refresh-token",
    expires_at: datetime | None = None,
    include_expires_at: bool = True,
) -> dict:
    record: dict[str, Any] = {
        "key": token,
        "access_token": token,
        "refresh_token": refresh_token,
        "oidc_client_id": "client-id",
        "token_endpoint": "https://auth.test/token",
    }
    if include_expires_at:
        if expires_at is None:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=2)
        record["expires_at"] = expires_at.isoformat().replace("+00:00", "Z")
    return {grok.DEFAULT_GROK_OIDC_SCOPE: record}


class _FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Findings #1/#4/#5: shared lock only; failures surface via shared helper
# ---------------------------------------------------------------------------


def test_lock_wrapper_delegates_only_to_shared_helper(grok) -> None:
    src = Path(grok.__file__).read_text(encoding="utf-8")
    assert "from litellm.secret_managers.credential_file_lock import" in src
    assert "from litellm.secret_managers.credential_file_metadata import" in src
    assert "from litellm.secret_managers.credential_file_write import" in src
    assert "write_and_publish_private_text" in src
    # No local fcntl usage (shared module owns module-scoped import + warnings).
    assert "import fcntl" not in src
    assert "fcntl.flock" not in src
    assert src.count("with credential_file_lock(lock_path)") == 1
    # Dead post-yield flock path from partial factoring must not remain.
    assert "LOCK_EX" not in src
    assert "LOCK_UN" not in src
    # Pid-only temp publication must not remain in the Grok consumer.
    assert 'f".{credential_path.name}.{os.getpid()}.tmp"' not in src
    assert "os.replace(tmp_path, credential_path)" not in src
    # Shared metadata wrappers refuse symlinks explicitly.
    assert "refuse_symlink=True" in src
    assert src.count("refuse_symlink=True") >= 4


def test_lock_failures_surface_through_shared_helper(
    grok,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import litellm.secret_managers.credential_file_lock as lock_mod

    class _FakeFcntl:
        LOCK_EX = 1
        LOCK_UN = 2

        @staticmethod
        def flock(fd, op):  # noqa: ARG004
            raise OSError("simulated flock failure")

    monkeypatch.setattr(lock_mod, "_fcntl", _FakeFcntl)
    lock = tmp_path / "auth.json.lock"
    with caplog.at_level(logging.WARNING, logger=lock_mod.__name__):
        with grok._credential_file_lock(lock):
            pass
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("LOCK_EX failed" in msg for msg in messages)
    joined = " ".join(messages)
    assert "access_token" not in joined
    assert "refresh_token" not in joined


# ---------------------------------------------------------------------------
# Finding #3: shared metadata helpers
# ---------------------------------------------------------------------------


def test_metadata_helpers_use_shared_owner(grok, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialFileMetadata,
    )

    target = tmp_path / "auth.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o600)
    meta = grok._snapshot_credential_file_metadata(target)
    assert isinstance(meta, CredentialFileMetadata)
    assert meta.mode == 0o600

    resolved = grok._resolve_credential_file_metadata(target)
    assert isinstance(resolved, CredentialFileMetadata)
    assert resolved.mode == 0o600

    # Resolve path must self-heal unsafe existing modes via shared clamp.
    os.chmod(target, 0o644)
    resolved_wide = grok._resolve_credential_file_metadata(target)
    assert resolved_wide.mode == 0o600
    assert not (resolved_wide.mode & 0o077)

    src = Path(grok.__file__).read_text(encoding="utf-8")
    assert "apply_credential_file_metadata" in src
    assert "resolve_credential_file_metadata(" in src
    assert "base_metadata=_snapshot_credential_file_metadata" in src
    # Explicit refuse_symlink on snapshot/resolve/apply wrappers.
    assert "refuse_symlink=True" in src
    assert src.count("refuse_symlink=True") >= 4


def test_snapshot_and_resolve_refuse_symlink_auth_path(grok, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real-auth.json"
    real.write_text("{}", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "auth.json"
    link.symlink_to(real)

    with pytest.raises(CredentialPathIsSymlinkError):
        grok._snapshot_credential_file_metadata(link)
    with pytest.raises(CredentialPathIsSymlinkError):
        grok._resolve_credential_file_metadata(link)


def test_apply_metadata_refuses_symlink(grok, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialFileMetadata,
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real-auth.json"
    real.write_text("{}", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "auth.json"
    link.symlink_to(real)
    meta = CredentialFileMetadata(uid=None, gid=None, mode=0o600)
    with pytest.raises(CredentialPathIsSymlinkError):
        grok._apply_credential_file_metadata(link, meta)
    assert real.stat().st_mode & 0o777 == 0o600


def test_write_credential_payload_preserves_private_mode(grok, tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    payload = {
        grok.DEFAULT_GROK_OIDC_SCOPE: {
            "access_token": "at-new",
            "refresh_token": "rt-new",
            "expires_at": "2099-01-01T00:00:00Z",
        }
    }
    grok._write_credential_payload(target, payload)
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data[grok.DEFAULT_GROK_OIDC_SCOPE]["access_token"] == "at-new"
    leftovers = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert leftovers == []


def test_write_credential_payload_uses_shared_publish(
    grok, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "auth.json"
    seen: list[dict[str, object]] = []

    def fake_publish(final_path, content, *, metadata=None, mode=None, default_mode=0o600, mkdir_parents=True):  # noqa: ANN001
        seen.append(
            {
                "final_path": Path(final_path),
                "content": content,
                "metadata": metadata,
                "default_mode": default_mode,
                "mkdir_parents": mkdir_parents,
            }
        )
        Path(final_path).parent.mkdir(parents=True, exist_ok=True)
        Path(final_path).write_text(content, encoding="utf-8")
        os.chmod(final_path, 0o600)
        return Path(final_path)

    monkeypatch.setattr(grok, "write_and_publish_private_text", fake_publish)
    payload = {"scope": {"access_token": "a", "refresh_token": "b"}}
    grok._write_credential_payload(target, payload)
    assert len(seen) == 1
    assert seen[0]["final_path"] == target
    assert '"access_token": "a"' in str(seen[0]["content"])
    assert seen[0]["default_mode"] == grok.DEFAULT_GROK_OIDC_AUTH_FILE_MODE
    assert seen[0]["mkdir_parents"] is True
    assert seen[0]["metadata"] is not None


def test_write_credential_payload_refuses_symlink_target(
    grok, tmp_path: Path
) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real-auth.json"
    real.write_text('{"keep":true}\n', encoding="utf-8")
    os.chmod(real, 0o600)
    target = tmp_path / "auth.json"
    target.symlink_to(real)

    with pytest.raises(CredentialPathIsSymlinkError):
        grok._write_credential_payload(
            target,
            {
                grok.DEFAULT_GROK_OIDC_SCOPE: {
                    "access_token": "attacker",
                    "refresh_token": "attacker-rt",
                }
            },
        )

    assert real.read_text(encoding="utf-8") == '{"keep":true}\n'
    leftovers = list(tmp_path.glob(".auth.json.*.tmp"))
    assert leftovers == []


def test_write_credential_payload_temp_name_is_not_pid_only(
    grok, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm.secret_managers.credential_file_write as write_mod

    target = tmp_path / "auth.json"
    seen_temps: list[Path] = []
    real_write_temp = write_mod.write_private_temp_file_text

    def spy_temp(final_path, content, **kwargs):  # noqa: ANN001
        tmp = real_write_temp(final_path, content, **kwargs)
        seen_temps.append(tmp)
        return tmp

    monkeypatch.setattr(write_mod, "write_private_temp_file_text", spy_temp)
    grok._write_credential_payload(target, {"scope": {"key": "new"}})
    assert seen_temps
    temp_name = seen_temps[0].name
    # Shared maker uses .<final>.<pid>.<token>.tmp — not pid-only.
    assert temp_name.startswith(f".{target.name}.")
    assert temp_name.endswith(".tmp")
    assert str(os.getpid()) in temp_name
    parts = temp_name.split(".")
    assert any(len(p) >= 16 and all(c in "0123456789abcdef" for c in p) for p in parts)
    assert target.is_file()
    leftovers = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert leftovers == []


def test_write_private_file_text_creates_at_0600(grok, tmp_path: Path) -> None:
    path = tmp_path / "secret.tmp"
    grok._write_private_file_text(path, "token-value\n", mode=0o600)
    assert path.stat().st_mode & 0o777 == 0o600


def test_write_private_file_text_clamps_wide_mode_at_create(
    grok, tmp_path: Path
) -> None:
    path = tmp_path / "secret-wide.tmp"
    # Even if a caller passes a group/other-readable mode, open must still be private.
    grok._write_private_file_text(path, "token-value\n", mode=0o644)
    assert path.stat().st_mode & 0o777 == 0o600


def test_write_private_file_text_refuses_symlink_target(grok, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "elsewhere.secret"
    real.write_text("victim", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "secret.tmp"
    link.symlink_to(real)
    with pytest.raises((CredentialPathIsSymlinkError, OSError)):
        grok._write_private_file_text(link, "attacker\n", mode=0o600)
    assert real.read_text(encoding="utf-8") == "victim"


def test_write_credential_payload_clamps_wide_mode(
    grok, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{}", encoding="utf-8")
    auth_path.chmod(0o640)
    monkeypatch.delenv("AAWM_GROK_OIDC_AUTH_FILE_MODE", raising=False)
    grok._write_credential_payload(auth_path, {"scope": {"key": "new"}})
    assert auth_path.stat().st_mode & 0o777 == 0o600


def test_write_credential_payload_honors_env_uid_gid(
    grok, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_path = tmp_path / "auth.json"
    chown_calls: list[tuple[str, int, int]] = []

    def fake_chown(
        target: str | os.PathLike[str],
        uid: int,
        gid: int,
        *args,
        **kwargs,
    ) -> None:
        chown_calls.append((os.fspath(target), uid, gid))

    monkeypatch.setenv("AAWM_GROK_OIDC_AUTH_FILE_UID", "1001")
    monkeypatch.setenv("AAWM_GROK_OIDC_AUTH_FILE_GID", "1002")
    # shared apply_credential_file_metadata uses os.chown from its module
    monkeypatch.setattr(
        "litellm.secret_managers.credential_file_metadata.os.chown",
        fake_chown,
    )

    grok._write_credential_payload(auth_path, {"scope": {"key": "new"}})
    assert len(chown_calls) == 1
    assert chown_calls[0][1:] == (1001, 1002)


def test_apply_metadata_delegates_to_shared_helper(
    grok, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "auth.json"
    target.write_text("{}", encoding="utf-8")
    seen: list[tuple[object, object]] = []

    def fake_apply(path, metadata, *, default_mode=0o600, refuse_symlink=True):  # noqa: ANN001
        seen.append((path, metadata))
        # Keep behavior realistic enough for mode assertions.
        os.chmod(path, metadata.mode & 0o777)

    monkeypatch.setattr(
        "litellm.secret_managers.credential_file_metadata.apply_credential_file_metadata",
        fake_apply,
    )
    # Also patch the name imported into the grok script module.
    monkeypatch.setattr(grok, "apply_credential_file_metadata", fake_apply)

    meta = grok.CredentialFileMetadata(uid=7, gid=8, mode=0o600)
    grok._apply_credential_file_metadata(target, meta)
    assert len(seen) == 1
    assert Path(seen[0][0]) == target
    assert seen[0][1] == meta


# ---------------------------------------------------------------------------
# Finding #6: missing expires_at fails safe toward refresh
# ---------------------------------------------------------------------------


def test_credential_needs_refresh_missing_expires_at_fail_safe(grok) -> None:
    assert (
        grok._credential_needs_refresh({"access_token": "t"}, buffer_seconds=300)
        is True
    )
    assert (
        grok._credential_needs_refresh(
            {"access_token": "t", "expires_at": None}, buffer_seconds=300
        )
        is True
    )
    assert (
        grok._credential_needs_refresh(
            {"access_token": "t", "expires_at": "not-a-date"}, buffer_seconds=300
        )
        is True
    )


def test_credential_needs_refresh_respects_buffer(grok) -> None:
    fresh = {
        "access_token": "t",
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=2))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    near = {
        "access_token": "t",
        "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    assert grok._credential_needs_refresh(fresh, buffer_seconds=300) is False
    assert grok._credential_needs_refresh(near, buffer_seconds=300) is True


def test_refresh_attempts_when_expires_at_missing(grok, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(_scoped_payload(grok, include_expires_at=False)),
        encoding="utf-8",
    )
    os.chmod(auth_path, 0o600)

    with patch.object(
        grok.urllib_request,
        "urlopen",
        return_value=_FakeResponse(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }
        ),
    ):
        result = grok.refresh_grok_oidc_auth_file(auth_path, force=False)

    assert result["attempted"] is True
    assert result["refreshed"] is True
    assert result["skipped"] is False
    data = json.loads(auth_path.read_text(encoding="utf-8"))
    assert data[grok.DEFAULT_GROK_OIDC_SCOPE]["access_token"] == "new-access"
    assert auth_path.stat().st_mode & 0o777 == 0o600


def test_refresh_skips_when_still_fresh(grok, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            _scoped_payload(
                grok,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
            )
        ),
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("token endpoint should not be called")

    with patch.object(grok.urllib_request, "urlopen", side_effect=fail_if_called):
        result = grok.refresh_grok_oidc_auth_file(auth_path, buffer_seconds=300)

    assert result["skipped"] is True
    assert result["attempted"] is False
    assert result["refreshed"] is False


# ---------------------------------------------------------------------------
# Shared error sanitizer residual (RR-075)
# ---------------------------------------------------------------------------


def test_sanitize_redacts_secret_values_not_only_labels(grok) -> None:
    raw = (
        "invalid_grant: access_token=eyJhbGciOi.live.token.value "
        "refresh_token: rt-super-secret "
        "client_secret=cs-xyz id_token= id.tok"
    )
    sanitized = grok._sanitize_error_message(raw)
    assert "eyJhbGciOi.live.token.value" not in sanitized
    assert "rt-super-secret" not in sanitized
    assert "cs-xyz" not in sanitized
    assert "id.tok" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized
    assert "super-secret" not in sanitized


def test_sanitize_still_truncates_long_messages(grok) -> None:
    long = "x" * 2000
    out = grok._sanitize_error_message(long, limit=100)
    assert len(out) <= 100
    assert out.endswith("...")


def test_sanitize_default_limit_bounds_output(grok) -> None:
    long = "y" * 2000
    out = grok._sanitize_error_message(long)
    assert len(out) <= grok.DEFAULT_GROK_OIDC_ERROR_MESSAGE_LIMIT
    assert out.endswith("...")


def test_sanitize_delegates_to_shared_helper(grok) -> None:
    from litellm.secret_managers.credential_error_sanitizer import (
        DEFAULT_SECRET_FIELD_NAMES,
        sanitize_credential_error_message,
    )

    assert grok._SECRET_FIELD_NAMES is DEFAULT_SECRET_FIELD_NAMES
    raw = "refresh_token=secret-value"
    assert grok._sanitize_error_message(raw) == sanitize_credential_error_message(
        raw, limit=500
    )
    long = "z" * 2000
    assert grok._sanitize_error_message(long, limit=100) == (
        sanitize_credential_error_message(long, limit=100)
    )


def test_extract_oauth_error_hint_sanitizes_embedded_tokens(grok) -> None:
    body = json.dumps(
        {
            "error": "invalid_grant",
            "error_description": "access_token=eyJ.leaked.value was rejected",
        }
    )
    hint = grok._extract_oauth_error_hint(body)
    assert hint is not None
    assert "eyJ.leaked.value" not in hint
    assert "access_token=[REDACTED]" in hint or "invalid_grant" in hint


def test_sanitize_redacts_even_backslash_quoted_suffix_leaks(grok) -> None:
    """Consumer path: even-backslash closes early; quoted suffix must not leak."""
    # Two backslashes: naive consumers close after \\" then leave suffix".
    raw_even = r'access_token="secret\\"suffix"'
    sanitized = grok._sanitize_error_message(raw_even)
    assert "secret" not in sanitized
    assert "suffix" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert sanitized.count("[REDACTED]") == 1

    raw_sq = r"access_token='secret\\'suffix'"
    sanitized_sq = grok._sanitize_error_message(raw_sq)
    assert "secret" not in sanitized_sq
    assert "suffix" not in sanitized_sq
    assert "access_token=[REDACTED]" in sanitized_sq

    # Four-backslash variant also must not leave a suffix.
    raw_four = r'access_token="secret\\\\"suffix"'
    out_four = grok._sanitize_error_message(raw_four)
    assert "suffix" not in out_four
    assert "access_token=[REDACTED]" in out_four


def test_sanitize_redacts_escaped_quotes_inside_quoted_values(grok) -> None:
    # Escaped double quote inside the value: access_token="sec\"ret-value"
    raw = r'access_token="sec\"ret-value"'
    sanitized = grok._sanitize_error_message(raw)
    assert "ret-value" not in sanitized
    assert "access_token=[REDACTED]" in sanitized


def test_sanitize_redacts_authorization_bearer_json_dict_and_quoted_assign(
    grok,
) -> None:
    cases = [
        (
            '{"Authorization":"Bearer tok-json-secret"}',
            "tok-json-secret",
            '"Bearer [REDACTED]"',
        ),
        (
            "{'Authorization': 'Bearer tok-dict-secret'}",
            "tok-dict-secret",
            "'Bearer [REDACTED]'",
        ),
        (
            'Authorization="Bearer tok-quoted-secret"',
            "tok-quoted-secret",
            '"Bearer [REDACTED]"',
        ),
        (
            "Authorization='Bearer tok-quoted-sq'",
            "tok-quoted-sq",
            "'Bearer [REDACTED]'",
        ),
        (
            "Authorization=Bearer tok-eq-secret",
            "tok-eq-secret",
            "Bearer [REDACTED]",
        ),
        (
            '{ "Authorization" : "Bearer spaced-token" }',
            "spaced-token",
            "Bearer [REDACTED]",
        ),
        (
            "upstream 401 Authorization: Bearer eyJhbGciOi.bearer.secret",
            "eyJhbGciOi.bearer.secret",
            "Authorization: Bearer [REDACTED]",
        ),
    ]
    for raw, secret, marker in cases:
        sanitized = grok._sanitize_error_message(raw)
        assert secret not in sanitized, raw
        assert marker in sanitized, (raw, sanitized)
        # Consumer default limit still applies (parity with shared helper).
        assert sanitized == grok._sanitize_error_message(
            raw, limit=grok.DEFAULT_GROK_OIDC_ERROR_MESSAGE_LIMIT
        )
        from litellm.secret_managers.credential_error_sanitizer import (
            sanitize_credential_error_message,
        )

        assert sanitized == sanitize_credential_error_message(raw, limit=500)
