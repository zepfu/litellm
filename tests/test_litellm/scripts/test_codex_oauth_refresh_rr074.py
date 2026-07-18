"""RR-074 residuals for scripts/codex_oauth_refresh.py.

Covers remaining findings after shared lock + private write landings, plus the
consumer migration onto shared credential_file_write:
1/5/6. Lock delegates solely to shared credential_file_lock (no dead fcntl copy).
2.     Credential publish uses write_and_publish_private_text (exclusive private
       temp, no local temp-publication path, mode 0600).
3.     Error sanitizer redacts secret values, not only field-name labels.
4.     Metadata helpers delegate to shared credential_file_metadata and clamp modes.
7.     Non-dict `tokens` key raises ValueError.
8.     Defaults are ~-relative portable paths (no hardcoded operator home).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "codex_oauth_refresh.py"


def _load_module():
    name = "codex_oauth_refresh_rr074"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def codex():
    return _load_module()


# ---------------------------------------------------------------------------
# Finding #8: portable defaults
# ---------------------------------------------------------------------------


def test_defaults_are_portable_tilde_paths(codex) -> None:
    assert codex.DEFAULT_CODEX_AUTH_FILE.startswith("~/")
    assert codex.DEFAULT_CODEX_LOCK_FILE.startswith("~/")
    assert "/home/zepfu" not in codex.DEFAULT_CODEX_AUTH_FILE
    assert "/home/zepfu" not in codex.DEFAULT_CODEX_LOCK_FILE


def test_default_paths_expanduser(codex) -> None:
    auth = Path(codex.DEFAULT_CODEX_AUTH_FILE).expanduser()
    lock = Path(codex.DEFAULT_CODEX_LOCK_FILE).expanduser()
    assert str(auth).startswith(str(Path.home()))
    assert str(lock).startswith(str(Path.home()))
    assert "~" not in str(auth)
    assert auth.name == "auth.json"
    assert lock.name == "auth.json.lock"


# ---------------------------------------------------------------------------
# Findings #1/#5/#6: shared lock only
# ---------------------------------------------------------------------------


def test_lock_wrapper_delegates_only_to_shared_helper(codex) -> None:
    src = Path(codex.__file__).read_text(encoding="utf-8")
    assert "from litellm.secret_managers.credential_file_lock import" in src
    assert "from litellm.secret_managers.credential_file_metadata import" in src
    # No local fcntl usage (shared module owns module-scoped import + warnings).
    assert "import fcntl" not in src
    assert "fcntl.flock" not in src
    assert src.count("with credential_file_lock(lock_path)") == 1
    # Dead post-yield flock path from partial factoring must not remain.
    assert "LOCK_EX" not in src
    assert "LOCK_UN" not in src


# ---------------------------------------------------------------------------
# Finding #4: shared metadata helpers
# ---------------------------------------------------------------------------


def test_metadata_helpers_use_shared_owner(codex, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialFileMetadata,
    )

    target = tmp_path / "auth.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o600)
    meta = codex._snapshot_credential_file_metadata(target)
    assert isinstance(meta, CredentialFileMetadata)
    assert meta.mode == 0o600

    resolved = codex._resolve_credential_file_metadata(target)
    assert isinstance(resolved, CredentialFileMetadata)
    assert resolved.mode == 0o600

    # Resolve path must self-heal unsafe existing modes via shared clamp.
    os.chmod(target, 0o644)
    resolved_wide = codex._resolve_credential_file_metadata(target)
    assert resolved_wide.mode == 0o600
    assert not (resolved_wide.mode & 0o077)

    src = Path(codex.__file__).read_text(encoding="utf-8")
    assert "apply_credential_file_metadata" in src
    assert "resolve_credential_file_metadata(" in src
    assert "base_metadata=_snapshot_credential_file_metadata" in src


def test_write_auth_data_uses_shared_publish_api(codex) -> None:
    src = Path(codex.__file__).read_text(encoding="utf-8")
    assert "from litellm.secret_managers.credential_file_write import" in src
    assert "write_and_publish_private_text" in src
    # Local unsafe temp publication path must be gone.
    assert "def _write_private_file_text" not in src
    assert "os.replace(tmp_path" not in src
    assert "time.monotonic_ns()" not in src
    assert "with_name(" not in src or "auth_path.with_name" not in src
    # Shared metadata wrappers refuse symlinks explicitly.
    assert "refuse_symlink=True" in src


def test_write_auth_data_preserves_private_mode(codex, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "auth.json"
    payload = {
        "tokens": {
            "access_token": "at-new",
            "refresh_token": "rt-new",
            "expires_at": 4_000_000_000,
        }
    }
    codex._write_auth_data(target, payload)
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["tokens"]["access_token"] == "at-new"
    leftovers = list(target.parent.glob(f".{target.name}.*.tmp"))
    assert leftovers == []


def test_write_auth_data_self_heals_unsafe_mode(codex, tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    payload = {
        "tokens": {
            "access_token": "at-new",
            "refresh_token": "rt-new",
            "expires_at": 4_000_000_000,
        }
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(target, 0o644)

    codex._write_auth_data(target, payload)
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["tokens"]["access_token"] == "at-new"


def test_write_auth_data_refuses_symlink_target(codex, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real-auth.json"
    real.write_text(json.dumps({"tokens": {"access_token": "keep"}}), encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "auth.json"
    link.symlink_to(real)
    payload = {
        "tokens": {
            "access_token": "attacker",
            "refresh_token": "rt",
            "expires_at": 4_000_000_000,
        }
    }
    with pytest.raises(ValueError) as excinfo:
        codex._write_auth_data(link, payload)
    assert isinstance(excinfo.value.__cause__, CredentialPathIsSymlinkError) or (
        "symlink" in str(excinfo.value).lower()
    )
    assert real.read_text(encoding="utf-8")
    data = json.loads(real.read_text(encoding="utf-8"))
    assert data["tokens"]["access_token"] == "keep"
    leftovers = list(tmp_path.glob(".auth.json.*.tmp"))
    assert leftovers == []


def test_write_auth_data_delegates_to_write_and_publish(
    codex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "auth.json"
    payload = {
        "tokens": {
            "access_token": "delegated",
            "refresh_token": "rt",
            "expires_at": 4_000_000_000,
        }
    }
    calls: list[dict[str, object]] = []

    def fake_publish(final_path, content, **kwargs):  # noqa: ANN001
        calls.append({"final_path": final_path, "content": content, **kwargs})
        path = Path(final_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    monkeypatch.setattr(codex, "write_and_publish_private_text", fake_publish)
    codex._write_auth_data(target, payload)
    assert len(calls) == 1
    assert Path(calls[0]["final_path"]) == target
    assert '"access_token": "delegated"' in str(calls[0]["content"])
    assert calls[0]["default_mode"] == codex.DEFAULT_CODEX_AUTH_FILE_MODE
    assert calls[0]["mkdir_parents"] is True
    assert calls[0]["metadata"] is not None
    assert target.is_file()


# ---------------------------------------------------------------------------
# Finding #3: value-redacting sanitizer
# ---------------------------------------------------------------------------


def test_sanitize_redacts_secret_values_not_only_labels(codex) -> None:
    raw = (
        "invalid_grant: access_token=eyJhbGciOi.live.token.value "
        "refresh_token: rt-super-secret "
        "client_secret=cs-xyz id_token= id.tok"
    )
    sanitized = codex._sanitize_error_message(raw)
    assert "eyJhbGciOi.live.token.value" not in sanitized
    assert "rt-super-secret" not in sanitized
    assert "cs-xyz" not in sanitized
    assert "id.tok" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized
    # Field labels alone (no value) are fine to keep; values must not leak.
    assert "super-secret" not in sanitized


def test_sanitize_still_truncates_long_messages(codex) -> None:
    long = "x" * 2000
    out = codex._sanitize_error_message(long, limit=100)
    assert len(out) <= 100
    assert out.endswith("...")


def test_extract_oauth_error_hint_sanitizes_embedded_tokens(codex) -> None:
    body = json.dumps(
        {
            "error": "invalid_grant",
            "error_description": "access_token=eyJ.leaked.value was rejected",
        }
    )
    hint = codex._extract_oauth_error_hint(body)
    assert hint is not None
    assert "eyJ.leaked.value" not in hint
    assert "access_token=[REDACTED]" in hint or "invalid_grant" in hint


# ---------------------------------------------------------------------------
# Finding #7: strict tokens container
# ---------------------------------------------------------------------------


def test_get_token_data_accepts_tokens_dict(codex) -> None:
    auth = {"tokens": {"access_token": "a", "refresh_token": "r"}}
    token_data = codex._get_token_data(auth)
    assert token_data is auth["tokens"]
    assert token_data["access_token"] == "a"


def test_get_token_data_falls_back_when_tokens_key_absent(codex) -> None:
    auth = {"access_token": "a", "refresh_token": "r"}
    token_data = codex._get_token_data(auth)
    assert token_data is auth


def test_get_token_data_rejects_non_dict_tokens(codex) -> None:
    for bad in ("not-a-dict", ["list"], 123, None):
        with pytest.raises(ValueError, match="tokens"):
            codex._get_token_data({"tokens": bad})


def test_refresh_returns_sanitized_error_for_malformed_tokens(
    codex, tmp_path: Path
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"tokens": "corrupt"}), encoding="utf-8")
    result = codex.refresh_codex_oauth_auth_file(auth_path, force=True)
    assert result["attempted"] is True
    assert result["refreshed"] is False
    assert result["error_class"] == "ValueError"
    assert "tokens" in (result["error_message"] or "")


# ---------------------------------------------------------------------------
# End-to-end-ish offline refresh with mocked HTTP
# ---------------------------------------------------------------------------


def test_refresh_writes_new_tokens_under_lock(codex, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    lock_path = tmp_path / "auth.json.lock"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                    "expires_at": 1,  # expired
                }
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

    with patch.object(codex.urllib_request, "urlopen", return_value=_Resp()):
        result = codex.refresh_codex_oauth_auth_file(
            auth_path,
            lock_file=lock_path,
            force=True,
        )

    assert result["refreshed"] is True
    assert result["error_message"] is None
    data = json.loads(auth_path.read_text(encoding="utf-8"))
    assert data["tokens"]["access_token"] == "new-access"
    assert data["tokens"]["refresh_token"] == "new-refresh"
    assert auth_path.stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------------------
# Finding #4 continued: env overrides + repair path
# ---------------------------------------------------------------------------


def test_resolve_metadata_honors_mode_env(
    codex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "auth.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o600)
    monkeypatch.setenv("AAWM_CODEX_AUTH_FILE_MODE", "0o600")
    resolved = codex._resolve_credential_file_metadata(target)
    assert resolved.mode == 0o600

    # Group/other bits in override are rejected / clamped.
    monkeypatch.setenv("AAWM_CODEX_AUTH_FILE_MODE", "0o644")
    resolved_wide = codex._resolve_credential_file_metadata(target)
    assert resolved_wide.mode == 0o600


def test_repair_credential_file_metadata_clamps_existing(codex, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    lock_path = tmp_path / "auth.json.lock"
    auth_path.write_text("{}", encoding="utf-8")
    lock_path.write_text("", encoding="utf-8")
    os.chmod(auth_path, 0o644)
    os.chmod(lock_path, 0o644)
    codex._repair_credential_file_metadata(auth_path, lock_path)
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert lock_path.stat().st_mode & 0o777 == 0o600


def test_snapshot_and_resolve_refuse_symlink_auth_path(codex, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "auth.json"
    link.symlink_to(real)
    with pytest.raises(CredentialPathIsSymlinkError):
        codex._snapshot_credential_file_metadata(link)
    with pytest.raises(CredentialPathIsSymlinkError):
        codex._resolve_credential_file_metadata(link)


def test_extract_oauth_error_hint_from_exception_payload(codex) -> None:
    body = {
        "error": "invalid_grant",
        "error_description": "refresh_token=rt-leaked-value was rejected",
    }
    exc = ValueError(json.dumps(body))
    hint = codex._extract_oauth_error_hint(exc)
    assert hint is not None
    assert "rt-leaked-value" not in hint
    assert "refresh_token=[REDACTED]" in hint or "invalid_grant" in hint


def test_default_lock_path_derives_from_auth_when_unset(codex, tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "fresh",
                    "refresh_token": "rt",
                    "expires_at": 4_000_000_000,
                }
            }
        ),
        encoding="utf-8",
    )
    os.chmod(auth_path, 0o600)
    result = codex.refresh_codex_oauth_auth_file(auth_path, force=False)
    assert result["skipped"] is True
    # Default lock should have been created next to the auth file.
    assert (tmp_path / "auth.json.lock").exists()


def test_sanitize_delegates_to_shared_helper(codex) -> None:
    from litellm.secret_managers.credential_error_sanitizer import (
        DEFAULT_SECRET_FIELD_NAMES,
        sanitize_credential_error_message,
    )

    assert codex._SECRET_FIELD_NAMES is DEFAULT_SECRET_FIELD_NAMES
    raw = "access_token=abc refresh_token=def"
    assert codex._sanitize_error_message(raw) == sanitize_credential_error_message(
        raw, limit=500
    )
    long = "x" * 2000
    assert codex._sanitize_error_message(long, limit=100) == (
        sanitize_credential_error_message(long, limit=100)
    )


def test_sanitize_redacts_quoted_authorization_bearer_forms(codex) -> None:
    """Consumer path must redact quoted Authorization Bearer shapes (shared helper)."""
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
            "upstream 401 Authorization: Bearer eyJhbGciOi.bearer.secret "
            "note bearer of bad news without header should stay",
            "eyJhbGciOi.bearer.secret",
            "Authorization: Bearer [REDACTED]",
        ),
    ]
    for raw, secret, marker in cases:
        sanitized = codex._sanitize_error_message(raw)
        assert secret not in sanitized, raw
        assert marker in sanitized, (raw, sanitized)
        # English "bearer" prose must not be treated as a credential.
        if "bearer of bad news" in raw:
            assert "bearer of bad news" in sanitized


def test_sanitize_redacts_even_backslash_quoted_suffix_leaks(codex) -> None:
    """Even-backslash closes early; trailing quoted suffix must not leak via consumer."""
    # Two backslashes: naive consumers close after \\" then leave suffix".
    raw_even = r'access_token="secret\\"suffix"'
    sanitized = codex._sanitize_error_message(raw_even)
    assert "secret" not in sanitized
    assert "suffix" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert sanitized.count("[REDACTED]") == 1

    raw_sq = r"access_token='secret\\'suffix'"
    sanitized_sq = codex._sanitize_error_message(raw_sq)
    assert "secret" not in sanitized_sq
    assert "suffix" not in sanitized_sq
    assert "access_token=[REDACTED]" in sanitized_sq

    # Four-backslash variant also must not leave a suffix.
    raw_four = r'access_token="secret\\\\"suffix"'
    out_four = codex._sanitize_error_message(raw_four)
    assert "suffix" not in out_four
    assert "access_token=[REDACTED]" in out_four

    # Escaped double quote inside the value must not partial-leak.
    raw_escaped = r'access_token="sec\"ret-value"'
    sanitized_escaped = codex._sanitize_error_message(raw_escaped)
    assert "ret-value" not in sanitized_escaped
    assert "access_token=[REDACTED]" in sanitized_escaped
