"""Shared private credential-file temp write + atomic publish helpers.

Used by AAWM OAuth refresh scripts (RR-065/074/075/092) so temp creation,
symlink refusal, metadata apply, and failed-temp cleanup stay consistent.

Security goals:
- No umask window: create with private mode bits via ``os.open``.
- Exclusive create when possible (``O_EXCL``) with unpredictable same-dir names.
- Prefer ``O_NOFOLLOW`` so a symlink cannot redirect the write.
- Refuse symlink credential *targets* and refuse metadata apply on symlinks.
- Clean up failed temps best-effort.
"""

from __future__ import annotations

import errno
import os
import secrets
from pathlib import Path
from typing import Optional, Union

from litellm.secret_managers.credential_file_metadata import (
    DEFAULT_CREDENTIAL_FILE_MODE,
    CredentialFileMetadata,
    CredentialPathIsSymlinkError,
    apply_credential_file_metadata,
    clamp_private_credential_file_mode,
    ensure_not_symlink_path,
    is_symlink_path,
)

PathLike = Union[Path, str]

# Re-export for consumers that import write helpers + exception together.
__all__ = [
    "CredentialPathIsSymlinkError",
    "is_symlink_path",
    "refuse_symlink_path",
    "write_private_file_text",
    "make_private_temp_path",
    "write_private_temp_file_text",
    "publish_private_credential_file",
    "write_and_publish_private_text",
]


def refuse_symlink_path(path: PathLike, *, role: str = "credential path") -> None:
    """Raise ``CredentialPathIsSymlinkError`` when ``path`` is a symlink."""
    ensure_not_symlink_path(path, role=role)


def _as_path(path: PathLike) -> Path:
    return Path(path)


def _open_flags(*, exclusive: bool, nofollow: bool) -> int:
    flags = os.O_WRONLY | os.O_CREAT
    if exclusive:
        flags |= os.O_EXCL
    else:
        flags |= os.O_TRUNC
    if nofollow and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def write_private_file_text(
    path: PathLike,
    content: str,
    *,
    mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    exclusive: bool = False,
    refuse_symlink: bool = True,
) -> Path:
    """Create/write ``path`` with private mode at creation time (no umask window).

    - Clamps group/other bits to ``default_mode``.
    - Uses ``O_CREAT`` (+ ``O_EXCL`` when ``exclusive``) and ``O_NOFOLLOW`` when
      available so a preplanted symlink cannot redirect the write.
    - On failure, best-effort unlinks the path when we created it exclusively
      or when the write itself failed after open.
    """
    target = _as_path(path)
    if refuse_symlink and target.exists(follow_symlinks=False):
        refuse_symlink_path(target, role="write target")
    mode = clamp_private_credential_file_mode(mode, default_mode=default_mode)
    flags = _open_flags(exclusive=exclusive, nofollow=True)
    opened = False
    fd: Optional[int] = None
    try:
        fd = os.open(str(target), flags, mode)
        opened = True
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None  # fdopen owns it
            handle.write(content)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        # Only unlink if we successfully opened/created this path. Never remove
        # a pre-existing collision on exclusive create failure.
        if opened:
            try:
                if target.exists(follow_symlinks=False) or is_symlink_path(target):
                    target.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    # Defensive re-chmod via lchmod when available so we never follow a race.
    try:
        if hasattr(os, "lchmod"):
            os.lchmod(target, mode)  # type: ignore[attr-defined]
        else:
            if refuse_symlink:
                refuse_symlink_path(target, role="write target")
            os.chmod(target, mode)
    except OSError:
        pass
    return target


def make_private_temp_path(
    final_path: PathLike,
    *,
    prefix: str = ".",
    suffix: str = ".tmp",
) -> Path:
    """Return an unpredictable same-directory temp path next to ``final_path``."""
    final = _as_path(final_path)
    token = secrets.token_hex(16)
    name = f"{prefix}{final.name}.{os.getpid()}.{token}{suffix}"
    return final.with_name(name)


def write_private_temp_file_text(
    final_path: PathLike,
    content: str,
    *,
    mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    max_attempts: int = 8,
) -> Path:
    """Write content to a new exclusive same-dir private temp next to ``final_path``.

    Returns the temp path. Caller is responsible for publish/cleanup, or use
    :func:`publish_private_credential_file` / :func:`write_and_publish_private_text`.
    """
    final = _as_path(final_path)
    last_error: Optional[BaseException] = None
    for _ in range(max(1, max_attempts)):
        tmp = make_private_temp_path(final)
        try:
            write_private_file_text(
                tmp,
                content,
                mode=mode,
                default_mode=default_mode,
                exclusive=True,
                refuse_symlink=True,
            )
            return tmp
        except FileExistsError as exc:
            last_error = exc
            continue
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EEXIST:
                last_error = exc
                continue
            raise
    raise OSError(
        f"Failed to create exclusive private temp next to {final}"
    ) from last_error


def publish_private_credential_file(
    temp_path: PathLike,
    final_path: PathLike,
    *,
    metadata: Optional[CredentialFileMetadata] = None,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    refuse_final_symlink: bool = True,
    cleanup_temp_on_error: bool = True,
) -> Path:
    """Apply optional metadata to temp, then atomically replace into ``final_path``.

    Refuses when ``final_path`` is already a symlink (TOCTOU-limited; combined
    with exclusive nofollow temp create). On error, unlinks the temp when
    ``cleanup_temp_on_error`` is true.
    """
    tmp = _as_path(temp_path)
    final = _as_path(final_path)
    try:
        if refuse_final_symlink:
            refuse_symlink_path(final, role="credential target")
        refuse_symlink_path(tmp, role="credential temp")
        if metadata is not None:
            apply_credential_file_metadata(
                tmp, metadata, default_mode=default_mode, refuse_symlink=True
            )
        os.replace(tmp, final)
        if refuse_final_symlink and is_symlink_path(final):
            raise CredentialPathIsSymlinkError(
                f"Refusing symlink credential target after publish: {final}"
            )
        return final
    except Exception:
        if cleanup_temp_on_error:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_and_publish_private_text(
    final_path: PathLike,
    content: str,
    *,
    metadata: Optional[CredentialFileMetadata] = None,
    mode: Optional[int] = None,
    default_mode: int = DEFAULT_CREDENTIAL_FILE_MODE,
    mkdir_parents: bool = True,
) -> Path:
    """Write private exclusive temp next to ``final_path`` and publish it.

    Preferred one-shot API for OAuth credential writers:

    1. Refuse if final path is a symlink.
    2. Create exclusive same-dir temp with private mode (O_EXCL + O_NOFOLLOW).
    3. Apply ownership/mode metadata to the temp (symlink-refusing).
    4. ``os.replace`` temp → final.
    5. Clean temp on any failure.
    """
    final = _as_path(final_path)
    if mkdir_parents:
        final.parent.mkdir(parents=True, exist_ok=True)
    refuse_symlink_path(final, role="credential target")
    if metadata is not None:
        write_mode = clamp_private_credential_file_mode(
            metadata.mode, default_mode=default_mode
        )
    elif mode is not None:
        write_mode = clamp_private_credential_file_mode(
            mode, default_mode=default_mode
        )
    else:
        write_mode = default_mode & 0o777
    tmp = write_private_temp_file_text(
        final,
        content,
        mode=write_mode,
        default_mode=default_mode,
    )
    return publish_private_credential_file(
        tmp,
        final,
        metadata=metadata,
        default_mode=default_mode,
        refuse_final_symlink=True,
        cleanup_temp_on_error=True,
    )
