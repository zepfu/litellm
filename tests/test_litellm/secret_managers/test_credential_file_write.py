"""Shared private credential write/publish helpers (RR-065/074/075/092)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from litellm.secret_managers.credential_file_metadata import (
    CredentialFileMetadata,
    CredentialPathIsSymlinkError,
)
from litellm.secret_managers.credential_file_write import (
    is_symlink_path,
    make_private_temp_path,
    publish_private_credential_file,
    refuse_symlink_path,
    write_and_publish_private_text,
    write_private_file_text,
    write_private_temp_file_text,
)


def test_write_private_file_text_mode_0600_no_umask_window(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    write_private_file_text(target, '{"k":1}\n', mode=0o600)
    assert target.is_file()
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.read_text(encoding="utf-8") == '{"k":1}\n'


def test_write_private_file_text_clamps_unsafe_mode(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    write_private_file_text(target, "x", mode=0o644)
    assert target.stat().st_mode & 0o777 == 0o600


def test_write_private_file_text_refuses_preplanted_symlink(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.secret"
    real.write_text("victim", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "cred.tmp"
    link.symlink_to(real)
    with pytest.raises((CredentialPathIsSymlinkError, OSError)):
        write_private_file_text(link, "attacker\n", mode=0o600, exclusive=False)
    assert real.read_text(encoding="utf-8") == "victim"


def test_write_private_temp_is_exclusive_unpredictable_same_dir(tmp_path: Path) -> None:
    final = tmp_path / "auth.json"
    tmp = write_private_temp_file_text(final, "secret-payload\n", mode=0o600)
    assert tmp.parent == final.parent
    assert tmp.name.startswith(f".{final.name}.")
    assert tmp.name.endswith(".tmp")
    # Unpredictable token: not only pid-based.
    assert str(os.getpid()) in tmp.name
    parts = tmp.name.split(".")
    # .auth.json.<pid>.<token>.tmp
    assert any(len(p) >= 16 and all(c in "0123456789abcdef" for c in p) for p in parts)
    assert tmp.stat().st_mode & 0o777 == 0o600
    assert not is_symlink_path(tmp)


def test_write_private_temp_exclusive_rejects_preplant_collision(
    tmp_path: Path, monkeypatch
) -> None:
    final = tmp_path / "auth.json"
    fixed = final.with_name(f".{final.name}.fixed.tmp")

    def fixed_temp(path, **kwargs):  # noqa: ARG001
        return fixed

    monkeypatch.setattr(
        "litellm.secret_managers.credential_file_write.make_private_temp_path",
        fixed_temp,
    )
    # Preplant regular file at the would-be temp path.
    fixed.write_text("planted", encoding="utf-8")
    with pytest.raises(OSError):
        write_private_temp_file_text(final, "new\n", mode=0o600, max_attempts=2)
    assert fixed.read_text(encoding="utf-8") == "planted"


def test_publish_refuses_symlink_final_and_cleans_temp(tmp_path: Path) -> None:
    final = tmp_path / "auth.json"
    victim = tmp_path / "victim.json"
    victim.write_text("keep-me", encoding="utf-8")
    final.symlink_to(victim)
    tmp = write_private_temp_file_text(final, "new-secret\n", mode=0o600)
    assert tmp.exists()
    with pytest.raises(CredentialPathIsSymlinkError):
        publish_private_credential_file(
            tmp,
            final,
            metadata=CredentialFileMetadata(uid=None, gid=None, mode=0o600),
        )
    # Temp cleaned on error.
    assert not tmp.exists()
    assert victim.read_text(encoding="utf-8") == "keep-me"


def test_write_and_publish_private_text_success_and_mode(tmp_path: Path) -> None:
    final = tmp_path / "nested" / "auth.json"
    meta = CredentialFileMetadata(uid=None, gid=None, mode=0o600)
    write_and_publish_private_text(
        final,
        '{"access_token":"a"}\n',
        metadata=meta,
        default_mode=0o600,
    )
    assert final.is_file()
    assert not is_symlink_path(final)
    assert final.stat().st_mode & 0o777 == 0o600
    assert "access_token" in final.read_text(encoding="utf-8")
    # No leftover temps.
    leftovers = list(final.parent.glob(f".{final.name}.*.tmp"))
    assert leftovers == []


def test_write_and_publish_refuses_symlink_target(tmp_path: Path) -> None:
    final = tmp_path / "auth.json"
    real = tmp_path / "real.json"
    real.write_text("real", encoding="utf-8")
    final.symlink_to(real)
    with pytest.raises(CredentialPathIsSymlinkError):
        write_and_publish_private_text(final, "nope\n", mode=0o600)
    assert real.read_text(encoding="utf-8") == "real"
    leftovers = list(tmp_path.glob(".auth.json.*.tmp"))
    assert leftovers == []


def test_make_private_temp_path_same_dir(tmp_path: Path) -> None:
    final = tmp_path / "a.json"
    a = make_private_temp_path(final)
    b = make_private_temp_path(final)
    assert a.parent == tmp_path
    assert b.parent == tmp_path
    assert a != b


def test_refuse_symlink_path_helper(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_text("n", encoding="utf-8")
    refuse_symlink_path(p)  # no raise
    link = tmp_path / "l"
    link.symlink_to(p)
    with pytest.raises(CredentialPathIsSymlinkError):
        refuse_symlink_path(link, role="credential target")


def test_exclusive_create_uses_o_excl_and_nofollow_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[int] = []
    real_open = os.open

    def spy_open(path, flags, mode=0o777):  # noqa: ANN001
        seen.append(flags)
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", spy_open)
    final = tmp_path / "auth.json"
    write_private_temp_file_text(final, "z\n", mode=0o600)
    assert seen, "os.open should be used"
    flags = seen[0]
    assert flags & os.O_EXCL
    assert flags & os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        assert flags & os.O_NOFOLLOW
