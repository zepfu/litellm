"""Shared credential_file_metadata helpers (RR-065 residual #5 / RR-075)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from litellm.secret_managers.credential_file_metadata import (
    CredentialFileMetadata,
    CredentialPathIsSymlinkError,
    apply_credential_file_metadata,
    clamp_private_credential_file_mode,
    ensure_not_symlink_path,
    is_symlink_path,
    parse_optional_nonnegative_int,
    resolve_credential_file_metadata,
    resolve_credential_file_mode_override,
    snapshot_credential_file_metadata,
)


def test_parse_optional_nonnegative_int_supports_octal() -> None:
    assert parse_optional_nonnegative_int("0o600") == 0o600
    assert parse_optional_nonnegative_int("600") == 600
    assert parse_optional_nonnegative_int("") is None
    assert parse_optional_nonnegative_int("-1") is None
    assert parse_optional_nonnegative_int("nope") is None


def test_clamp_private_credential_file_mode() -> None:
    assert clamp_private_credential_file_mode(0o600) == 0o600
    assert clamp_private_credential_file_mode(0o400) == 0o400
    assert clamp_private_credential_file_mode(0o640) == 0o600
    assert clamp_private_credential_file_mode(0o644) == 0o600
    assert clamp_private_credential_file_mode(0o777, default_mode=0o400) == 0o400


def test_snapshot_missing_uses_default_mode(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    meta = snapshot_credential_file_metadata(missing, default_mode=0o600)
    assert meta == CredentialFileMetadata(uid=None, gid=None, mode=0o600)


def test_snapshot_existing_path(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o640)
    meta = snapshot_credential_file_metadata(target)
    assert meta.mode == 0o640
    assert meta.uid is not None
    assert meta.gid is not None


def test_snapshot_symlink_no_follow_and_optional_refuse(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("secret", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "cred.json"
    link.symlink_to(real)
    assert is_symlink_path(link)
    # Default: lstat metadata of the symlink itself, not the target mode-follow.
    meta = snapshot_credential_file_metadata(link)
    assert meta.uid is not None
    with pytest.raises(CredentialPathIsSymlinkError):
        snapshot_credential_file_metadata(link, refuse_symlink=True)
    with pytest.raises(CredentialPathIsSymlinkError):
        ensure_not_symlink_path(link)


def test_resolve_clamps_group_other_bits(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o644)
    meta = resolve_credential_file_metadata(target, default_mode=0o600)
    assert meta.mode == 0o600


def test_resolve_mode_override_env(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "cred.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AAWM_TEST_AUTH_FILE_MODE", "0o600")
    meta = resolve_credential_file_metadata(
        target,
        default_mode=0o600,
        mode_env="AAWM_TEST_AUTH_FILE_MODE",
    )
    assert meta.mode == 0o600


def test_resolve_mode_override_rejects_world_bits(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "cred.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AAWM_TEST_AUTH_FILE_MODE", "0o644")
    meta = resolve_credential_file_metadata(
        target,
        default_mode=0o600,
        mode_env="AAWM_TEST_AUTH_FILE_MODE",
    )
    assert meta.mode == 0o600
    assert (
        resolve_credential_file_mode_override(
            "AAWM_TEST_AUTH_FILE_MODE", default_mode=0o600
        )
        == 0o600
    )


def test_resolve_honors_uid_gid_env_and_base_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "cred.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AAWM_TEST_AUTH_FILE_UID", "1001")
    monkeypatch.setenv("AAWM_TEST_AUTH_FILE_GID", "1002")
    base = CredentialFileMetadata(uid=65534, gid=65534, mode=0o640)
    meta = resolve_credential_file_metadata(
        target,
        default_mode=0o600,
        mode_env="AAWM_TEST_AUTH_FILE_MODE",
        uid_env="AAWM_TEST_AUTH_FILE_UID",
        gid_env="AAWM_TEST_AUTH_FILE_GID",
        base_metadata=base,
    )
    assert meta.uid == 1001
    assert meta.gid == 1002
    # base mode had group/other bits; resolve clamps.
    assert meta.mode == 0o600


def test_apply_credential_file_metadata_mode(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    target.write_text("secret", encoding="utf-8")
    os.chmod(target, 0o644)
    apply_credential_file_metadata(
        target, CredentialFileMetadata(uid=None, gid=None, mode=0o600)
    )
    assert target.stat().st_mode & 0o777 == 0o600


def test_apply_credential_file_metadata_clamps_unsafe_mode(tmp_path: Path) -> None:
    target = tmp_path / "cred.json"
    target.write_text("secret", encoding="utf-8")
    os.chmod(target, 0o600)
    apply_credential_file_metadata(
        target, CredentialFileMetadata(uid=None, gid=None, mode=0o644)
    )
    assert target.stat().st_mode & 0o777 == 0o600


def test_apply_refuses_symlink_target(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("secret", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "cred.json"
    link.symlink_to(real)
    with pytest.raises(CredentialPathIsSymlinkError):
        apply_credential_file_metadata(
            link, CredentialFileMetadata(uid=None, gid=None, mode=0o600)
        )
    # Target content/mode untouched by refused apply.
    assert real.read_text(encoding="utf-8") == "secret"
    assert real.stat().st_mode & 0o777 == 0o600
