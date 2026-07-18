"""RR-092 residuals for scripts/xai_oauth_refresh.py.

Covers remaining findings after shared lock + private write landings, plus the
consumer migration onto shared credential_file_write:
1/4/6. Lock delegates solely to shared credential_file_lock (no dead fcntl copy).
2.     Credential publish uses write_and_publish_private_text (exclusive private
       temp, no local predictable temp-publication path, mode 0600).
3.     Error sanitizer redacts secret values and bounds output to 500 chars,
       including consumer-level even-backslash and Authorization Bearer cases.
5.     Metadata helpers self-heal unsafe modes via shared credential_file_metadata
       and pass refuse_symlink=True.
7.     Missing expires_at fails safe toward refresh.
8.     Defaults are ~-relative portable paths (no hardcoded operator home).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "xai_oauth_refresh.py"


def _load_module():
    name = "xai_oauth_refresh_rr092"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def xai():
    return _load_module()


# ---------------------------------------------------------------------------
# Finding #8: portable defaults
# ---------------------------------------------------------------------------


def test_defaults_are_portable_tilde_paths(xai) -> None:
    assert xai.DEFAULT_XAI_OAUTH_AUTH_FILE.startswith("~/")
    assert xai.DEFAULT_XAI_OAUTH_LOCK_FILE.startswith("~/")
    assert "/home/zepfu" not in xai.DEFAULT_XAI_OAUTH_AUTH_FILE
    assert "/home/zepfu" not in xai.DEFAULT_XAI_OAUTH_LOCK_FILE


def test_default_paths_expanduser(xai) -> None:
    auth = Path(xai.DEFAULT_XAI_OAUTH_AUTH_FILE).expanduser()
    lock = Path(xai.DEFAULT_XAI_OAUTH_LOCK_FILE).expanduser()
    assert str(auth).startswith(str(Path.home()))
    assert str(lock).startswith(str(Path.home()))
    assert "~" not in str(auth)
    assert auth.name == "oauth-auth.json"
    assert lock.name == "oauth-auth.json.lock"


def test_loop_xai_oauth_help_defaults_are_portable() -> None:
    """Sidecar CLI help must document ~ defaults, not a hardcoded operator home."""
    loop_path = _REPO_ROOT / "scripts" / "run_provider_status_observations_loop.py"
    name = "run_provider_status_observations_loop_rr092_help"
    spec = importlib.util.spec_from_file_location(name, loop_path)
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
    auth_help = help_by_dest["xai_oauth_auth_file"]
    lock_help = help_by_dest["xai_oauth_lock_file"]
    assert "~/.litellm/xai/oauth-auth.json" in auth_help
    assert "~/.litellm/xai/oauth-auth.json.lock" in lock_help
    assert "/home/zepfu" not in auth_help
    assert "/home/zepfu" not in lock_help
    assert mod.DEFAULT_XAI_OAUTH_AUTH_FILE.startswith("~/")
    assert mod.DEFAULT_XAI_OAUTH_LOCK_FILE.startswith("~/")
    assert "/home/zepfu" not in mod.DEFAULT_XAI_OAUTH_AUTH_FILE
    assert "/home/zepfu" not in mod.DEFAULT_XAI_OAUTH_LOCK_FILE


# ---------------------------------------------------------------------------
# Findings #1/#4/#6: shared lock only + publish migration source shape
# ---------------------------------------------------------------------------


def test_lock_wrapper_delegates_only_to_shared_helper(xai) -> None:
    src = Path(xai.__file__).read_text(encoding="utf-8")
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
    # Predictable/local temp publication path must not remain.
    assert "time.monotonic_ns()" not in src
    assert "import time" not in src
    assert "os.replace(tmp_path" not in src
    assert 'f".{auth_path.name}.{os.getpid()}' not in src
    # Shared metadata wrappers refuse symlinks explicitly.
    assert "refuse_symlink=True" in src
    assert src.count("refuse_symlink=True") >= 4


# ---------------------------------------------------------------------------
# Finding #5: shared metadata helpers + mode self-heal + symlink safety
# ---------------------------------------------------------------------------


def test_metadata_helpers_use_shared_owner(xai, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialFileMetadata,
    )

    target = tmp_path / "oauth-auth.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o600)
    meta = xai._snapshot_credential_file_metadata(target)
    assert isinstance(meta, CredentialFileMetadata)
    assert meta.mode == 0o600

    resolved = xai._resolve_credential_file_metadata(target)
    assert isinstance(resolved, CredentialFileMetadata)
    assert resolved.mode == 0o600

    src = Path(xai.__file__).read_text(encoding="utf-8")
    assert "apply_credential_file_metadata" in src
    assert "resolve_credential_file_metadata(" in src
    assert "base_metadata=_snapshot_credential_file_metadata" in src


def test_resolve_metadata_clamps_unsafe_existing_mode(xai, tmp_path: Path) -> None:
    target = tmp_path / "oauth-auth.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o644)
    resolved = xai._resolve_credential_file_metadata(target)
    assert resolved.mode == 0o600
    assert not (resolved.mode & 0o077)


def test_snapshot_and_resolve_refuse_symlink_auth_path(xai, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real-oauth-auth.json"
    real.write_text("{}", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "oauth-auth.json"
    link.symlink_to(real)

    with pytest.raises(CredentialPathIsSymlinkError):
        xai._snapshot_credential_file_metadata(link)
    with pytest.raises(CredentialPathIsSymlinkError):
        xai._resolve_credential_file_metadata(link)


def test_write_credential_payload_self_heals_unsafe_mode(xai, tmp_path: Path) -> None:
    target = tmp_path / "oauth-auth.json"
    payload = {
        "access_token": "at-new",
        "refresh_token": "rt-new",
        "expires_at": "2099-01-01T00:00:00Z",
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(target, 0o644)

    xai._write_credential_payload(target, payload)
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["access_token"] == "at-new"
    leftovers = list(target.parent.glob(f".{target.name}.*.tmp"))
    assert leftovers == []


def test_write_credential_payload_uses_shared_publish(
    xai, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "oauth-auth.json"
    seen: list[dict[str, object]] = []

    def fake_publish(
        final_path,
        content,
        *,
        metadata=None,
        mode=None,
        default_mode=0o600,
        mkdir_parents=True,
    ):  # noqa: ANN001
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

    monkeypatch.setattr(xai, "write_and_publish_private_text", fake_publish)
    payload = {"access_token": "a", "refresh_token": "b"}
    xai._write_credential_payload(target, payload)
    assert len(seen) == 1
    assert seen[0]["final_path"] == target
    assert '"access_token": "a"' in str(seen[0]["content"])
    assert seen[0]["default_mode"] == xai.DEFAULT_XAI_OAUTH_AUTH_FILE_MODE
    assert seen[0]["mkdir_parents"] is True
    assert seen[0]["metadata"] is not None


def test_write_credential_payload_refuses_symlink_target(xai, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real-oauth-auth.json"
    real.write_text(json.dumps({"access_token": "keep"}), encoding="utf-8")
    os.chmod(real, 0o600)
    target = tmp_path / "oauth-auth.json"
    target.symlink_to(real)

    with pytest.raises(ValueError) as excinfo:
        xai._write_credential_payload(
            target,
            {
                "access_token": "attacker",
                "refresh_token": "attacker-rt",
            },
        )
    assert isinstance(excinfo.value.__cause__, CredentialPathIsSymlinkError) or (
        "symlink" in str(excinfo.value).lower()
    )
    data = json.loads(real.read_text(encoding="utf-8"))
    assert data["access_token"] == "keep"
    leftovers = list(tmp_path.glob(".oauth-auth.json.*.tmp"))
    assert leftovers == []


def test_write_credential_payload_temp_name_is_not_pid_only(
    xai, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm.secret_managers.credential_file_write as write_mod

    target = tmp_path / "oauth-auth.json"
    seen_temps: list[Path] = []
    real_write_temp = write_mod.write_private_temp_file_text

    def spy_temp(final_path, content, **kwargs):  # noqa: ANN001
        tmp = real_write_temp(final_path, content, **kwargs)
        seen_temps.append(tmp)
        return tmp

    monkeypatch.setattr(write_mod, "write_private_temp_file_text", spy_temp)
    xai._write_credential_payload(target, {"access_token": "new", "refresh_token": "rt"})
    assert seen_temps
    temp_name = seen_temps[0].name
    # Shared maker uses .<final>.<pid>.<token>.tmp — not pid-only / monotonic_ns.
    assert temp_name.startswith(f".{target.name}.")
    assert temp_name.endswith(".tmp")
    assert str(os.getpid()) in temp_name
    parts = temp_name.split(".")
    assert any(len(p) >= 16 and all(c in "0123456789abcdef" for c in p) for p in parts)
    assert target.is_file()
    leftovers = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert leftovers == []


def test_write_private_file_text_creates_at_0600(xai, tmp_path: Path) -> None:
    path = tmp_path / "secret.tmp"
    xai._write_private_file_text(path, "token-value\n", mode=0o600)
    assert path.stat().st_mode & 0o777 == 0o600


def test_write_private_file_text_clamps_unsafe_mode(xai, tmp_path: Path) -> None:
    path = tmp_path / "loose.tmp"
    xai._write_private_file_text(path, "token-value\n", mode=0o644)
    assert path.stat().st_mode & 0o777 == 0o600


def test_write_private_file_text_refuses_symlink_target(xai, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "elsewhere.secret"
    real.write_text("victim", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "secret.tmp"
    link.symlink_to(real)
    with pytest.raises((CredentialPathIsSymlinkError, OSError)):
        xai._write_private_file_text(link, "attacker\n", mode=0o600)
    assert real.read_text(encoding="utf-8") == "victim"


# ---------------------------------------------------------------------------
# Finding #3: value-redacting sanitizer bounded to 500 chars by default
# ---------------------------------------------------------------------------


def test_sanitize_redacts_secret_values_not_only_labels(xai) -> None:
    raw = (
        "invalid_grant: access_token=eyJhbGciOi.live.token.value "
        "refresh_token: rt-super-secret "
        "client_secret=cs-xyz id_token= id.tok"
    )
    sanitized = xai._sanitize_error_message(raw)
    assert "eyJhbGciOi.live.token.value" not in sanitized
    assert "rt-super-secret" not in sanitized
    assert "cs-xyz" not in sanitized
    assert "id.tok" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized
    # Field labels alone (no value) are fine to keep; values must not leak.
    assert "super-secret" not in sanitized


def test_sanitize_still_truncates_long_messages(xai) -> None:
    long = "x" * 2000
    out = xai._sanitize_error_message(long, limit=100)
    assert len(out) <= 100
    assert out.endswith("...")


def test_sanitize_default_limit_is_500(xai) -> None:
    assert xai.DEFAULT_XAI_OAUTH_ERROR_MESSAGE_LIMIT == 500
    long = "y" * 2000
    out = xai._sanitize_error_message(long)
    assert len(out) <= 500
    assert out.endswith("...")


def test_refresh_http_error_summary_redacts_token_values(xai, tmp_path: Path) -> None:
    auth_path = tmp_path / "oauth-auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "client_id": "cid",
                "expires_at": 1,
            }
        ),
        encoding="utf-8",
    )
    os.chmod(auth_path, 0o600)

    def _raise(*_a: Any, **_k: Any):
        raise xai.urllib_error.HTTPError(
            url="https://auth.x.ai/oauth2/token",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=type(
                "FP",
                (),
                {
                    "read": lambda self: (
                        b'{"error_description":'
                        b'"access_token=eyJ.leaked.value was rejected"}'
                    ),
                    "close": lambda self: None,
                },
            )(),
        )

    with patch.object(xai.urllib_request, "urlopen", side_effect=_raise):
        result = xai.refresh_xai_oauth_auth_file(auth_path, force=True)

    assert result["attempted"] is True
    assert result["refreshed"] is False
    assert "eyJ.leaked.value" not in (result["error_message"] or "")
    assert "access_token=[REDACTED]" in (result["error_message"] or "")
    assert len(result["error_message"] or "") <= 500


# ---------------------------------------------------------------------------
# Finding #7: missing expires_at fails safe
# ---------------------------------------------------------------------------


def test_credential_needs_refresh_when_expires_at_missing(xai) -> None:
    assert (
        xai._credential_needs_refresh(
            {"access_token": "a", "refresh_token": "r"},
            buffer_seconds=300,
        )
        is True
    )


def test_credential_needs_refresh_when_expires_at_unparseable(xai) -> None:
    assert (
        xai._credential_needs_refresh(
            {"access_token": "a", "expires_at": "not-a-date"},
            buffer_seconds=300,
        )
        is True
    )


def test_credential_needs_refresh_false_when_fresh(xai) -> None:
    future = (
        (datetime.now(timezone.utc) + timedelta(hours=2))
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert (
        xai._credential_needs_refresh(
            {"access_token": "a", "expires_at": future},
            buffer_seconds=300,
        )
        is False
    )


def test_credential_needs_refresh_true_when_near_expiry(xai) -> None:
    near = (
        (datetime.now(timezone.utc) + timedelta(seconds=60))
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert (
        xai._credential_needs_refresh(
            {"access_token": "a", "expires_at": near},
            buffer_seconds=300,
        )
        is True
    )


# ---------------------------------------------------------------------------
# End-to-end-ish offline refresh with mocked HTTP
# ---------------------------------------------------------------------------


def test_refresh_writes_new_tokens_under_lock(xai, tmp_path: Path) -> None:
    auth_path = tmp_path / "oauth-auth.json"
    lock_path = tmp_path / "oauth-auth.json.lock"
    auth_path.write_text(
        json.dumps(
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "client_id": "cid-1",
                "expires_at": 1,  # expired
            }
        ),
        encoding="utf-8",
    )
    os.chmod(auth_path, 0o600)

    class _Resp:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    with patch.object(xai.urllib_request, "urlopen", return_value=_Resp()):
        result = xai.refresh_xai_oauth_auth_file(
            auth_path,
            lock_file=lock_path,
            force=True,
        )

    assert result["refreshed"] is True
    assert result["error_message"] is None
    data = json.loads(auth_path.read_text(encoding="utf-8"))
    assert data["access_token"] == "new-access"
    assert data["refresh_token"] == "new-refresh"
    assert data["key"] == "new-access"
    assert auth_path.stat().st_mode & 0o777 == 0o600
    leftovers = list(auth_path.parent.glob(f".{auth_path.name}.*.tmp"))
    assert leftovers == []


def test_refresh_skips_when_fresh_and_not_forced(xai, tmp_path: Path) -> None:
    auth_path = tmp_path / "oauth-auth.json"
    future = (
        (datetime.now(timezone.utc) + timedelta(hours=2))
        .isoformat()
        .replace("+00:00", "Z")
    )
    auth_path.write_text(
        json.dumps(
            {
                "access_token": "fresh-access",
                "refresh_token": "fresh-refresh",
                "client_id": "cid-1",
                "expires_at": future,
            }
        ),
        encoding="utf-8",
    )
    result = xai.refresh_xai_oauth_auth_file(auth_path, force=False)
    assert result["skipped"] is True
    assert result["attempted"] is False
    assert result["refreshed"] is False


def test_sanitize_delegates_to_shared_helper(xai) -> None:
    from litellm.secret_managers.credential_error_sanitizer import (
        DEFAULT_SECRET_FIELD_NAMES,
        sanitize_credential_error_message,
    )

    assert xai._SECRET_FIELD_NAMES is DEFAULT_SECRET_FIELD_NAMES
    raw = "access_token=abc refresh_token=def"
    assert xai._sanitize_error_message(raw) == sanitize_credential_error_message(
        raw, limit=500
    )
    long = "x" * 2000
    assert xai._sanitize_error_message(long, limit=100) == (
        sanitize_credential_error_message(long, limit=100)
    )


def test_sanitize_redacts_even_backslash_quoted_suffix_leaks_via_consumer(xai) -> None:
    """Consumer wrapper must keep even-backslash quoted-suffix secrets redacted.

    Regression for shared credential_error_sanitizer even-backslash handling
    when invoked only through scripts/xai_oauth_refresh._sanitize_error_message.
    """
    # Two backslashes: naive consumers close after \\" then leave suffix".
    raw_even = r'access_token="secret\\"suffix"'
    sanitized = xai._sanitize_error_message(raw_even)
    assert "secret" not in sanitized
    assert "suffix" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert sanitized.count("[REDACTED]") == 1

    raw_sq = r"access_token='secret\\'suffix'"
    sanitized_sq = xai._sanitize_error_message(raw_sq)
    assert "secret" not in sanitized_sq
    assert "suffix" not in sanitized_sq
    assert "access_token=[REDACTED]" in sanitized_sq

    # Four-backslash variant also must not leave a suffix.
    raw_four = r'access_token="secret\\\\"suffix"'
    out_four = xai._sanitize_error_message(raw_four)
    assert "suffix" not in out_four
    assert "access_token=[REDACTED]" in out_four


def test_sanitize_redacts_authorization_bearer_quoted_and_json_via_consumer(
    xai,
) -> None:
    """Consumer wrapper must redact quoted/JSON Authorization Bearer forms."""
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
            "upstream 401 Authorization: Bearer eyJhbGciOi.bearer.secret "
            "note bearer of bad news without header should stay",
            "eyJhbGciOi.bearer.secret",
            "Authorization: Bearer [REDACTED]",
        ),
    ]
    for raw, secret, marker in cases:
        sanitized = xai._sanitize_error_message(raw)
        assert secret not in sanitized, raw
        assert marker in sanitized, (raw, sanitized)
        if "bearer of bad news" in raw:
            assert "bearer of bad news" in sanitized
