"""RR-040 residuals for litellm/llms/xai/oauth.py (#4–#6 + closed #1–#3 evidence)."""

from __future__ import annotations

import ast
import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from litellm.llms.xai import oauth as oauth_mod


def _future_expires() -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(hours=1))
        .isoformat()
        .replace("+00:00", "Z")
    )


def _past_expires() -> str:
    return (
        (datetime.now(timezone.utc) - timedelta(hours=1))
        .isoformat()
        .replace("+00:00", "Z")
    )


def test_rr040_module_has_no_inprocess_refresh_entrypoints() -> None:
    """Issue #4: dead locked-refresh path must not remain as reachable code."""
    source = Path(oauth_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_get_xai_oauth_access_token_locked" not in defined
    assert "_refresh_xai_oauth_credential" not in defined
    assert "_credential_file_lock" not in defined
    assert "_oauth_credential_subject" not in defined
    assert "_oauth_refresh_action" not in defined
    assert "_extract_oauth_error_hint" not in defined
    assert "_update_credential_record" not in defined
    # Production entrypoints remain read-only.
    assert "get_xai_oauth_access_token" in defined
    assert "_get_xai_oauth_access_token_read_only" in defined
    assert "_get_grok_native_oauth_access_token_read_only" in defined


def test_rr040_no_inline_fcntl_or_httpx_refresh_client() -> None:
    """Issues #1/#5: this file no longer owns flock; no inline fcntl/httpx refresh."""
    source = Path(oauth_mod.__file__).read_text(encoding="utf-8")
    # Strip module docstring for code checks.
    tree = ast.parse(source)
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "fcntl" not in imports
    assert "contextlib" not in imports
    # Shared lock is not re-imported for a local wrapper anymore.
    assert "litellm.secret_managers.credential_file_lock" not in imports
    # httpx may remain as an unused harness patch surface; it must not be used
    # by any defined refresh helper (those helpers are deleted).


def test_rr040_credential_needs_refresh_missing_expires_at_fail_safe() -> None:
    """Issue #6: missing expires_at is not permanently fresh."""
    assert oauth_mod._credential_needs_refresh({"access_token": "t"}) is True
    assert (
        oauth_mod._credential_needs_refresh({"access_token": "t", "expires_at": None})
        is True
    )
    assert (
        oauth_mod._credential_needs_refresh(
            {"access_token": "t", "expires_at": "not-a-date"}
        )
        is True
    )


def test_rr040_credential_needs_refresh_respects_buffer() -> None:
    fresh = {"access_token": "t", "expires_at": _future_expires()}
    expired = {"access_token": "t", "expires_at": _past_expires()}
    assert oauth_mod._credential_needs_refresh(fresh) is False
    assert oauth_mod._credential_needs_refresh(expired) is True


def test_rr040_read_only_path_raises_when_expires_at_missing(tmp_path: Path) -> None:
    cred_path = tmp_path / "auth.json"
    scope = "https://auth.x.ai::test-client"
    payload = {
        scope: {
            "access_token": "live-token",
            "refresh_token": "refresh-token",
            # no expires_at
        }
    }
    cred_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar|refresh|expired|near expiry"):
        oauth_mod._get_xai_oauth_access_token_read_only(
            credential_path=cred_path,
            scope=scope,
        )


def test_rr040_read_only_path_returns_fresh_token(tmp_path: Path) -> None:
    cred_path = tmp_path / "auth.json"
    scope = "https://auth.x.ai::test-client"
    payload = {
        scope: {
            "access_token": "live-token",
            "refresh_token": "refresh-token",
            "expires_at": _future_expires(),
        }
    }
    cred_path.write_text(json.dumps(payload), encoding="utf-8")
    token = oauth_mod._get_xai_oauth_access_token_read_only(
        credential_path=cred_path,
        scope=scope,
    )
    assert token == "live-token"


def test_rr040_write_credential_payload_private_mode(tmp_path: Path) -> None:
    """Issue #2 closed: writes create private 0600 files (no umask window helper)."""
    target = tmp_path / "out.json"
    oauth_mod._write_credential_payload(
        target,
        {"access_token": "a", "refresh_token": "b"},
    )
    assert target.is_file()
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600
    assert json.loads(target.read_text(encoding="utf-8"))["access_token"] == "a"


def test_rr040_write_private_file_text_uses_os_open_mode() -> None:
    source = Path(oauth_mod.__file__).read_text(encoding="utf-8")
    assert "os.open" in source
    assert "O_CREAT" in source
    assert "0o600" in source


def test_rr040_module_documents_sidecar_only_refresh() -> None:
    doc = oauth_mod.__doc__ or ""
    assert "read-only" in doc.lower() or "sidecar" in doc.lower()
    assert "RR-040" in (doc + Path(oauth_mod.__file__).read_text(encoding="utf-8"))
