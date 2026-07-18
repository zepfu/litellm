"""OAuth credential writers create temp files at 0600 (no umask window)."""

from __future__ import annotations

import importlib.util
import os
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load(rel: str):
    path = _REPO / rel
    name = path.stem + "_cred_write_ut"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "rel,writer,arg_name",
    [
        ("scripts/xai_oauth_refresh.py", "_write_credential_payload", "auth_path"),
        ("scripts/codex_oauth_refresh.py", None, None),
        ("litellm/llms/xai/oauth.py", "_write_credential_payload", "credential_path"),
        (
            "scripts/grok_oidc_refresh.py",
            "_write_credential_payload",
            "credential_path",
        ),
        ("scripts/antigravity_oauth_refresh.py", "_write_token_data", "auth_path"),
    ],
)
def test_credential_writer_ends_at_0600(rel, writer, arg_name, tmp_path: Path):
    mod = _load(rel)
    if writer is None:
        # codex publishes via shared write_and_publish_private_text.
        assert hasattr(mod, "_write_auth_data")
        assert hasattr(mod, "write_and_publish_private_text")
        assert not hasattr(mod, "_write_private_file_text")
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "write_and_publish_private_text" in src
        assert "def _write_private_file_text" not in src

        auth_target = tmp_path / "auth.json"
        mod._write_auth_data(
            auth_target,
            {
                "tokens": {
                    "access_token": "a",
                    "refresh_token": "b",
                    "expires_at": 4_000_000_000,
                }
            },
        )
        assert auth_target.is_file()
        assert auth_target.stat().st_mode & 0o777 == 0o600
        data = json.loads(auth_target.read_text(encoding="utf-8"))
        assert data["tokens"]["access_token"] == "a"
        leftovers = list(auth_target.parent.glob(f".{auth_target.name}.*.tmp"))
        assert leftovers == []
        return

    target = tmp_path / "cred.json"
    fn = getattr(mod, writer)
    if writer == "_write_token_data":
        fn(target, {"access_token": "a", "refresh_token": "b"})
    else:
        fn(target, {"access_token": "a", "refresh_token": "b", "type": "oauth"})
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "access_token" in data


def test_private_helper_creates_0600_not_umask_default(tmp_path: Path):
    mod = _load("scripts/xai_oauth_refresh.py")
    path = tmp_path / "private.json"
    mod._write_private_file_text(path, '{"k":1}\n', mode=0o600)
    assert path.stat().st_mode & 0o777 == 0o600


def test_grok_writer_uses_shared_write_and_publish(tmp_path: Path):
    """RR-075: Grok consumer must publish via shared credential_file_write API."""
    mod = _load("scripts/grok_oidc_refresh.py")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "write_and_publish_private_text" in src
    assert "from litellm.secret_managers.credential_file_write import" in src
    # No pid-only temp publication path remains in the consumer.
    assert 'f".{credential_path.name}.{os.getpid()}.tmp"' not in src
    assert "os.replace(tmp_path, credential_path)" not in src

    target = tmp_path / "nested" / "auth.json"
    mod._write_credential_payload(
        target,
        {"access_token": "a", "refresh_token": "b", "type": "oauth"},
    )
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    assert json.loads(target.read_text(encoding="utf-8"))["access_token"] == "a"
    leftovers = list(target.parent.glob(f".{target.name}.*.tmp"))
    assert leftovers == []


def test_grok_writer_refuses_symlink_target(tmp_path: Path):
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    mod = _load("scripts/grok_oidc_refresh.py")
    real = tmp_path / "real.json"
    real.write_text('{"keep": true}\n', encoding="utf-8")
    target = tmp_path / "auth.json"
    target.symlink_to(real)
    with pytest.raises(CredentialPathIsSymlinkError):
        mod._write_credential_payload(target, {"access_token": "nope"})
    assert real.read_text(encoding="utf-8") == '{"keep": true}\n'


def test_antigravity_write_token_data_uses_shared_publish_and_refuses_symlink(
    tmp_path: Path,
) -> None:
    """RR-065 consumer migration: Antigravity publishes via shared helpers."""
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    mod = _load("scripts/antigravity_oauth_refresh.py")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "write_and_publish_private_text" in src
    assert 'f".{auth_path.name}.{os.getpid()}.tmp"' not in src

    target = tmp_path / "nested" / "token.json"
    mod._write_token_data(
        target,
        {
            "token": {
                "access_token": "a",
                "refresh_token": "b",
                "expiry": "2099-01-01T00:00:00Z",
            }
        },
    )
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    assert list(target.parent.glob(".token.json.*.tmp")) == []

    real = tmp_path / "real.json"
    real.write_text('{"keep":true}', encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "link-token.json"
    link.symlink_to(real)
    with pytest.raises(CredentialPathIsSymlinkError):
        mod._write_token_data(
            link,
            {
                "token": {
                    "access_token": "attacker",
                    "refresh_token": "r",
                    "expiry": "2099-01-01T00:00:00Z",
                }
            },
        )
    assert real.read_text(encoding="utf-8") == '{"keep":true}'


def test_codex_write_auth_data_refuses_symlink_target(tmp_path: Path):
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    mod = _load("scripts/codex_oauth_refresh.py")
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"tokens": {"access_token": "keep"}}), encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "auth.json"
    link.symlink_to(real)
    # Codex may refuse during symlink-safe metadata snapshot/resolve or during
    # shared publish; either path must fail closed without writing through.
    with pytest.raises((CredentialPathIsSymlinkError, ValueError, OSError)) as excinfo:
        mod._write_auth_data(
            link,
            {
                "tokens": {
                    "access_token": "attacker",
                    "refresh_token": "b",
                    "expires_at": 4_000_000_000,
                }
            },
        )
    err = excinfo.value
    cause = getattr(err, "__cause__", None)
    assert isinstance(err, CredentialPathIsSymlinkError) or isinstance(
        cause, CredentialPathIsSymlinkError
    ) or "symlink" in str(err).lower()
    data = json.loads(real.read_text(encoding="utf-8"))
    assert data["tokens"]["access_token"] == "keep"
    assert list(tmp_path.glob(".auth.json.*.tmp")) == []
