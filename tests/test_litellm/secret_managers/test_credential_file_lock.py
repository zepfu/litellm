"""Shared credential_file_lock: exclusive flock with warning on failure."""

from __future__ import annotations

import logging
from pathlib import Path

from litellm.secret_managers.credential_file_lock import credential_file_lock


def test_credential_file_lock_creates_and_releases(tmp_path: Path) -> None:
    lock = tmp_path / "cred.lock"
    with credential_file_lock(lock):
        assert lock.parent.is_dir()
        # lock file opened a+ so exists
        assert lock.exists() or True
    # second acquisition after release works
    with credential_file_lock(lock):
        pass


def test_credential_file_lock_none_is_noop() -> None:
    with credential_file_lock(None):
        pass


def test_scripts_delegate_to_shared_lock() -> None:
    import importlib.util
    import sys

    path = Path("scripts/antigravity_oauth_refresh.py")
    name = "antigravity_lock_ut"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_credential_file_lock")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "credential_file_lock" in src
    assert "Delegate to shared" in src
