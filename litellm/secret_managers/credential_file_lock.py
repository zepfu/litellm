"""Shared advisory lock for OAuth credential file writers.

Used by litellm/llms/xai/oauth.py and the AAWM oauth refresh scripts so
fcntl flock behavior stays identical (RR-040/065/074/075/092).
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


@contextmanager
def credential_file_lock(lock_path: Optional[Path]) -> Iterator[None]:
    """Exclusive advisory flock around credential read/write sections.

    On platforms without fcntl, or if flock fails, proceeds without mutual
    exclusion but logs a warning (never silent).
    """
    if lock_path is None:
        yield
        return

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    locked = False
    try:
        if _fcntl is None:
            logger.warning(
                "credential_file_lock: fcntl unavailable; proceeding without "
                "mutual exclusion for %s",
                lock_path,
            )
        else:
            try:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
                locked = True
            except OSError as exc:
                logger.warning(
                    "credential_file_lock: flock LOCK_EX failed for %s (%s); "
                    "proceeding without mutual exclusion",
                    lock_path,
                    exc,
                )
        yield
    finally:
        if locked and _fcntl is not None:
            try:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
            except OSError as exc:
                logger.warning(
                    "credential_file_lock: flock LOCK_UN failed for %s (%s)",
                    lock_path,
                    exc,
                )
        try:
            handle.close()
        except OSError:
            pass
