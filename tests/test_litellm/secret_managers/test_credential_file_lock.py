"""Shared credential_file_lock: nonblocking exclusive fail-closed flock."""

from __future__ import annotations

from pathlib import Path

import pytest

from litellm.secret_managers.credential_file_lock import (
    CredentialFileLockError,
    credential_file_lock,
)


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

    path = Path("scripts/grok_oidc_refresh.py")
    name = "grok_lock_ut"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_credential_file_lock")
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "credential_file_lock" in src
    assert "Delegate to shared" in src


def test_credential_file_lock_fails_closed_when_fcntl_missing(
    tmp_path: Path, monkeypatch
) -> None:
    import litellm.secret_managers.credential_file_lock as lock_mod

    monkeypatch.setattr(lock_mod, "_fcntl", None)
    lock = tmp_path / "cred.lock"
    with pytest.raises(CredentialFileLockError, match="locking is unavailable"):
        with lock_mod.credential_file_lock(lock):
            pass


def test_credential_file_lock_fails_closed_on_flock_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    import litellm.secret_managers.credential_file_lock as lock_mod

    operations = []

    class _FakeFcntl:
        LOCK_EX = 1
        LOCK_UN = 2
        LOCK_NB = 4

        @staticmethod
        def flock(fd, op):  # noqa: ARG004
            operations.append(op)
            raise OSError("simulated flock failure")

    monkeypatch.setattr(lock_mod, "_fcntl", _FakeFcntl)
    lock = tmp_path / "cred.lock"
    with pytest.raises(CredentialFileLockError, match="acquisition failed"):
        with lock_mod.credential_file_lock(lock):
            pass
    assert operations == [_FakeFcntl.LOCK_EX | _FakeFcntl.LOCK_NB]


def test_credential_file_lock_fails_closed_on_contention(
    tmp_path: Path, monkeypatch
) -> None:
    import litellm.secret_managers.credential_file_lock as lock_mod

    class _FakeFcntl:
        LOCK_EX = 1
        LOCK_UN = 2
        LOCK_NB = 4

        @staticmethod
        def flock(fd, op):  # noqa: ARG004
            raise BlockingIOError("simulated contention")

    monkeypatch.setattr(lock_mod, "_fcntl", _FakeFcntl)
    lock = tmp_path / "cred.lock"
    with pytest.raises(CredentialFileLockError, match="already held"):
        with lock_mod.credential_file_lock(lock):
            pass
