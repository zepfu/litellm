"""Shared advisory lock for OAuth credential file writers.

Used by the AAWM OAuth refresh scripts so fcntl flock behavior stays
identical (RR-040/065/074/075/092).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX
    _fcntl = None  # type: ignore[assignment]


class CredentialFileLockError(RuntimeError):
    """Raised when exclusive credential-writer locking is unavailable."""


@contextmanager
def credential_file_lock(lock_path: Optional[Path]) -> Iterator[None]:
    """Exclusive advisory flock around credential read/write sections.

    Acquisition is nonblocking and fail-closed. A competing writer or a
    platform without ``fcntl`` must not permit an unlocked credential write.
    """
    if lock_path is None:
        yield
        return

    lock_path = Path(lock_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise CredentialFileLockError(
            "Credential file lock could not be opened."
        ) from exc
    locked = False
    try:
        if _fcntl is None:
            raise CredentialFileLockError(
                "Credential file locking is unavailable on this platform."
            )
        lock_nonblocking = getattr(_fcntl, "LOCK_NB", None)
        if lock_nonblocking is None:
            raise CredentialFileLockError(
                "Nonblocking credential file locking is unavailable."
            )
        try:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | lock_nonblocking)
            locked = True
        except BlockingIOError as exc:
            raise CredentialFileLockError(
                "Credential file lock is already held."
            ) from exc
        except OSError as exc:
            raise CredentialFileLockError(
                "Credential file lock acquisition failed."
            ) from exc
        yield
    finally:
        if locked and _fcntl is not None:
            try:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
            except OSError as exc:
                logger.warning(
                    "credential_file_lock: flock LOCK_UN failed (%s)",
                    exc.__class__.__name__,
                )
        try:
            handle.close()
        except OSError:
            pass
