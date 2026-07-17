"""A5/RR-065: staged CLI home is 0700/0600 and cleanup removes it."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "antigravity_oauth_refresh.py"


def _load_antigravity_module():
    spec = importlib.util.spec_from_file_location(
        "antigravity_oauth_refresh_under_test", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def antigravity():
    return _load_antigravity_module()


def test_stage_cli_auth_home_uses_private_modes(antigravity):
    td = Path(tempfile.mkdtemp())
    try:
        seed = td / "seed.json"
        seed.write_text(
            '{"access_token":"a","refresh_token":"r"}', encoding="utf-8"
        )
        staged = td / "home"
        auth = antigravity._stage_cli_auth_home(staged, seed)
        assert auth.exists()
        assert auth.stat().st_mode & 0o777 == 0o600
        assert staged.stat().st_mode & 0o777 == 0o700
        antigravity._cleanup_staged_home(staged)
        assert not staged.exists()
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_cleanup_staged_home_none_is_noop(antigravity):
    antigravity._cleanup_staged_home(None)
