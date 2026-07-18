"""Shared credential-file ownership metadata helpers.

Used by AAWM OAuth refresh scripts so uid/gid/mode snapshot, resolve, and
apply behavior stays consistent (RR-065/074/075).

Symlink safety:
- Snapshot uses ``lstat`` (no-follow).
- Apply refuses symlink targets (no-follow) before ``chown``/``chmod``.
- Callers that need the shared exception type should import
  ``CredentialPathIsSymlinkError`` from ``credential_file_write`` or the local
  re-export below.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

DEFAULT_CREDENTIAL_FILE_MODE = 0o600

PathLike = Union[Path, str]


class CredentialPathIsSymlinkError(OSError):
    """Raised when a credential path is a symlink and must be refused."""


@dataclass(frozen=True)
class CredentialFileMetadata:
    """Ownership and permission metadata for a credential file."""

    uid: Optional[int]
    gid: Optional[int]
    mode: int


def parse_optional_nonnegative_int(value: Optional[str]) -> Optional[int]:
    """Parse a non-negative int from a string (supports 0o600-style literals)."""
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(str(value).strip(), 0)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def clamp_private_credential_file_mode(
    mode: int,
    *,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
) -> int:
    """Return a user-private mode; group/other bits fall back to ``default_mode``."""
    mode = mode & 0o777
    if mode & 0o077:
        return default_mode & 0o777
    return mode


def resolve_credential_file_mode_override(
    env_name: str,
    *,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
) -> Optional[int]:
    """Read a mode override env var; never allow group/other bits."""
    mode = parse_optional_nonnegative_int(os.getenv(env_name))
    if mode is None:
        return None
    return clamp_private_credential_file_mode(mode, default_mode=default_mode)


def is_symlink_path(path: PathLike) -> bool:
    """Return True if ``path`` is a symlink (lstat / no-follow)."""
    try:
        return stat.S_ISLNK(os.lstat(Path(path)).st_mode)
    except FileNotFoundError:
        return False
    except OSError:
        return False


def ensure_not_symlink_path(
    path: PathLike,
    *,
    role: str = "credential path",
) -> None:
    """Raise ``CredentialPathIsSymlinkError`` when ``path`` is a symlink."""
    p = Path(path)
    if is_symlink_path(p):
        raise CredentialPathIsSymlinkError(f"Refusing symlink {role}: {p}")


def snapshot_credential_file_metadata(
    credential_path: PathLike,
    *,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    refuse_symlink: bool = False,
) -> CredentialFileMetadata:
    """Snapshot uid/gid/mode for an existing path, or defaults if missing.

    Uses ``lstat`` so a symlink is not followed. When ``refuse_symlink`` is
    true, raises ``CredentialPathIsSymlinkError`` if the path is a symlink.
    """
    path = Path(credential_path)
    try:
        file_stat = os.lstat(path)
    except FileNotFoundError:
        return CredentialFileMetadata(
            uid=None,
            gid=None,
            mode=default_mode & 0o777,
        )
    if stat.S_ISLNK(file_stat.st_mode):
        if refuse_symlink:
            raise CredentialPathIsSymlinkError(
                f"Refusing symlink credential path for snapshot: {path}"
            )
        # Do not follow: report the symlink's own metadata only when allowed.
        return CredentialFileMetadata(
            uid=file_stat.st_uid,
            gid=file_stat.st_gid,
            mode=stat.S_IMODE(file_stat.st_mode),
        )
    return CredentialFileMetadata(
        uid=file_stat.st_uid,
        gid=file_stat.st_gid,
        mode=stat.S_IMODE(file_stat.st_mode),
    )


def resolve_credential_file_metadata(
    credential_path: PathLike,
    *,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    mode_env: Optional[str] = None,
    uid_env: Optional[str] = None,
    gid_env: Optional[str] = None,
    base_metadata: Optional[CredentialFileMetadata] = None,
    refuse_symlink: bool = False,
) -> CredentialFileMetadata:
    """Snapshot path metadata and apply optional env overrides.

    Group/other permission bits are always clamped to the private default.

    ``base_metadata`` lets callers inject a pre-snapshotted value (for example
    after monkeypatching a thin local snapshot wrapper) without re-statting the
    path inside this helper.

    When ``refuse_symlink`` is true and ``base_metadata`` is not provided, a
    symlink path is refused during snapshot.
    """
    metadata = (
        base_metadata
        if base_metadata is not None
        else snapshot_credential_file_metadata(
            credential_path,
            default_mode=default_mode,
            refuse_symlink=refuse_symlink,
        )
    )
    if refuse_symlink and base_metadata is None:
        # snapshot already refused; keep for API clarity
        pass
    elif refuse_symlink:
        ensure_not_symlink_path(credential_path, role="credential path")
    uid_override = (
        parse_optional_nonnegative_int(os.getenv(uid_env)) if uid_env else None
    )
    gid_override = (
        parse_optional_nonnegative_int(os.getenv(gid_env)) if gid_env else None
    )
    mode_override = (
        resolve_credential_file_mode_override(mode_env, default_mode=default_mode)
        if mode_env
        else None
    )
    mode = mode_override if mode_override is not None else metadata.mode
    mode = clamp_private_credential_file_mode(mode, default_mode=default_mode)
    return CredentialFileMetadata(
        uid=uid_override if uid_override is not None else metadata.uid,
        gid=gid_override if gid_override is not None else metadata.gid,
        mode=mode,
    )


def apply_credential_file_metadata(
    target_path: PathLike,
    metadata: CredentialFileMetadata,
    *,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    refuse_symlink: bool = True,
) -> None:
    """Apply uid/gid/mode to ``target_path`` when possible.

    Mode is always re-clamped so callers cannot accidentally reintroduce
    group/other permission bits on write or repair paths.

    By default refuses symlink targets (no-follow). Uses ``lchmod`` when
    available; otherwise re-checks non-symlink before ``chmod``.
    """
    path = Path(target_path)
    if refuse_symlink:
        ensure_not_symlink_path(path, role="credential path")
    mode = clamp_private_credential_file_mode(metadata.mode, default_mode=default_mode)
    if metadata.uid is not None or metadata.gid is not None:
        # os.chown follows symlinks by default unless follow_symlinks=False.
        try:
            os.chown(
                path,
                metadata.uid if metadata.uid is not None else -1,
                metadata.gid if metadata.gid is not None else -1,
                follow_symlinks=False,
            )
        except TypeError:
            # Platform without follow_symlinks kwarg: refuse symlink already done.
            if refuse_symlink:
                ensure_not_symlink_path(path, role="credential path")
            os.chown(
                path,
                metadata.uid if metadata.uid is not None else -1,
                metadata.gid if metadata.gid is not None else -1,
            )
        except NotImplementedError:
            # Some platforms reject follow_symlinks=False for chown.
            if refuse_symlink:
                ensure_not_symlink_path(path, role="credential path")
            os.chown(
                path,
                metadata.uid if metadata.uid is not None else -1,
                metadata.gid if metadata.gid is not None else -1,
            )
    try:
        if hasattr(os, "lchmod"):
            os.lchmod(path, mode)  # type: ignore[attr-defined]
        else:
            if refuse_symlink:
                ensure_not_symlink_path(path, role="credential path")
            os.chmod(path, mode)
    except NotImplementedError:
        if refuse_symlink:
            ensure_not_symlink_path(path, role="credential path")
        os.chmod(path, mode)
