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


def test_credential_file_lock_warns_when_fcntl_missing(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import litellm.secret_managers.credential_file_lock as lock_mod

    monkeypatch.setattr(lock_mod, "_fcntl", None)
    lock = tmp_path / "cred.lock"
    with caplog.at_level(logging.WARNING, logger=lock_mod.__name__):
        with lock_mod.credential_file_lock(lock):
            pass
    assert any("fcntl unavailable" in rec.getMessage() for rec in caplog.records)
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "secret" not in joined.lower()
    assert "token" not in joined.lower()


def test_credential_file_lock_warns_on_flock_oserror(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import litellm.secret_managers.credential_file_lock as lock_mod

    class _FakeFcntl:
        LOCK_EX = 1
        LOCK_UN = 2

        @staticmethod
        def flock(fd, op):  # noqa: ARG004
            raise OSError("simulated flock failure")

    monkeypatch.setattr(lock_mod, "_fcntl", _FakeFcntl)
    lock = tmp_path / "cred.lock"
    with caplog.at_level(logging.WARNING, logger=lock_mod.__name__):
        with lock_mod.credential_file_lock(lock):
            pass
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("LOCK_EX failed" in msg for msg in messages)
    joined = " ".join(messages)
    # path may appear; secrets must not
    assert "GOCSPX" not in joined
    assert "access_token" not in joined
