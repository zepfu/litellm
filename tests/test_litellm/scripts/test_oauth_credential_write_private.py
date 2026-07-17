"""OAuth credential writers create temp files at 0600 (no umask window)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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
        ("scripts/grok_oidc_refresh.py", "_write_credential_payload", "credential_path"),
        ("scripts/antigravity_oauth_refresh.py", "_write_token_data", "auth_path"),
    ],
)
def test_credential_writer_ends_at_0600(rel, writer, arg_name, tmp_path: Path):
    mod = _load(rel)
    if writer is None:
        # codex uses different payload shape — exercise private helper only
        assert hasattr(mod, "_write_private_file_text")
        target = tmp_path / "c.tmp"
        mod._write_private_file_text(target, "secret\n", mode=0o600)
        assert target.stat().st_mode & 0o777 == 0o600
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
