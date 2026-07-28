"""Grok native client-version contract parser and resolver.

Pure-stdlib module (no LiteLLM imports) that validates a JSON cache
file written by the installed Grok CLI.  The file is re-read on every
call so atomic replacement is restart-free.

Cache file schema (schema_version 1)::

    {
      "schema_version": 1,
      "client": "grok-cli",
      "version": "0.1.211",
      "build": "a1b2c3d4",
      "channel": "stable",
      "source": "installed-grok-cli",
      "observed_at": "2026-07-28T12:00:00Z"
    }

Environment configuration
-------------------------
``AAWM_GROK_CLIENT_VERSION_CACHE_PATH``
    Override the default cache file path
    (default ``/run/aawm/grok/native-client-version.json``).

``AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS``
    Override the default maximum record age in seconds
    (default ``172800`` = 48 hours). Must be a canonical positive
    ASCII decimal.
"""

from __future__ import annotations

import dataclasses
import errno
import json
import os
import re
import stat
import time
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
GROK_VERSION_CACHE_PATH_ENV = "AAWM_GROK_CLIENT_VERSION_CACHE_PATH"
GROK_VERSION_CACHE_MAX_AGE_ENV = (
    "AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROK_VERSION_DEFAULT_CACHE_PATH = "/run/aawm/grok/native-client-version.json"
GROK_VERSION_DEFAULT_MAX_AGE_SECONDS = 172_800  # 48 hours
GROK_VERSION_SCHEMA_VERSION = 1
GROK_VERSION_MAX_BYTES = 65_536  # 64 KiB

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "client",
        "version",
        "build",
        "channel",
        "source",
        "observed_at",
    }
)
_EXPECTED_CLIENT = "grok-cli"
_EXPECTED_SOURCE = "installed-grok-cli"

# Strict dotted-numeric version with at least two ASCII-numeric segments.
_VERSION_RE = re.compile(r"\A[0-9]+(?:\.[0-9]+)+\Z")
_POSITIVE_ASCII_DECIMAL_RE = re.compile(r"\A[1-9][0-9]*\Z")
# Strict lowercase hex build hash.
_BUILD_RE = re.compile(r"\A[0-9a-f]+\Z")
# Safe channel token: alphanumeric, hyphens, underscores, dots.
_CHANNEL_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")
# RFC 3339 UTC timestamp ending in Z.
_RFC3339_Z_RE = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z"
)


class GrokNativeVersionError(Exception):
    """Raised when the version cache is missing, stale, or malformed.

    Messages never contain file contents, configured paths, or secrets.
    """


@dataclasses.dataclass(frozen=True)
class GrokNativeVersionRecord:
    """Validated, immutable snapshot of the Grok CLI version cache."""

    schema_version: int
    client: str
    version: str
    build: str
    channel: str
    source: str
    observed_at: str
    observed_at_epoch: float


@dataclasses.dataclass(frozen=True)
class GrokNativeVersionMetadata:
    """Sanitized source metadata for observability (no file contents)."""

    cache_path: str
    version: str
    build: str
    channel: str
    source: str
    age_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_cache_path(explicit_path: Optional[str] = None) -> str:
    path = (
        explicit_path
        if explicit_path is not None
        else os.environ.get(
            GROK_VERSION_CACHE_PATH_ENV,
            GROK_VERSION_DEFAULT_CACHE_PATH,
        )
    )
    if not isinstance(path, str) or not os.path.isabs(path):
        raise GrokNativeVersionError(
            "invalid version cache path: must be absolute"
        )
    return path


def _resolve_max_age() -> int:
    raw = os.environ.get(GROK_VERSION_CACHE_MAX_AGE_ENV)
    if raw is None:
        return GROK_VERSION_DEFAULT_MAX_AGE_SECONDS
    if _POSITIVE_ASCII_DECIMAL_RE.fullmatch(raw) is None:
        raise GrokNativeVersionError(
            f"invalid {GROK_VERSION_CACHE_MAX_AGE_ENV}: "
            "must be a canonical positive ASCII decimal"
        )
    try:
        return int(raw)
    except (ValueError, OverflowError):
        raise GrokNativeVersionError(
            f"invalid {GROK_VERSION_CACHE_MAX_AGE_ENV}: "
            "must be a canonical positive ASCII decimal"
        )


def _validate_version_string(version: str) -> None:
    """Validate a strict dotted-numeric version string."""
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise GrokNativeVersionError(
            "invalid version: must contain at least two ASCII-numeric "
            "segments separated by dots"
        )


def _parse_observed_at(value: str) -> float:
    """Parse an RFC 3339 UTC timestamp ending in Z to epoch seconds."""
    if not isinstance(value, str) or not _RFC3339_Z_RE.match(value):
        raise GrokNativeVersionError(
            "invalid observed_at: must be UTC RFC 3339 ending in Z"
        )
    try:
        # Python 3.7+ fromisoformat does not handle Z suffix.
        normalized = f"{value[:-1]}+00:00"
        dt = datetime.fromisoformat(normalized)
        return dt.timestamp()
    except (ValueError, OverflowError):
        raise GrokNativeVersionError(
            "invalid observed_at: unparseable timestamp"
        )


def _open_cache_file(path: str) -> int:
    """Open the configured path once without following a final symlink."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise GrokNativeVersionError(
            "secure version cache open is unsupported on this platform"
        )

    flags = os.O_RDONLY | nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec

    try:
        return os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            message = "version cache file is missing"
        elif exc.errno in (errno.ELOOP, errno.EMLINK):
            message = "version cache file is a symlink or unsafe path"
        else:
            message = "version cache file is unreadable"
        raise GrokNativeVersionError(message) from None
    except (TypeError, ValueError, UnicodeError):
        raise GrokNativeVersionError(
            "version cache file is unreadable"
        ) from None


def _read_cache_bytes(file_descriptor: int) -> bytes:
    """Read at most one byte beyond the allowed size from the open fd."""
    chunks: list[bytes] = []
    bytes_read = 0
    while bytes_read <= GROK_VERSION_MAX_BYTES:
        try:
            chunk = os.read(
                file_descriptor,
                min(8192, GROK_VERSION_MAX_BYTES + 1 - bytes_read),
            )
        except OSError:
            raise GrokNativeVersionError(
                "version cache file could not be read"
            ) from None
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)

    if bytes_read > GROK_VERSION_MAX_BYTES:
        raise GrokNativeVersionError(
            f"version cache file exceeds {GROK_VERSION_MAX_BYTES} bytes"
        )
    return b"".join(chunks)


def _read_cache_text(path: str) -> str:
    """Securely open, validate, and decode the configured cache file."""
    file_descriptor = _open_cache_file(path)
    try:
        try:
            file_status = os.fstat(file_descriptor)
        except OSError:
            raise GrokNativeVersionError(
                "version cache file metadata is unreadable"
            ) from None

        if not stat.S_ISREG(file_status.st_mode):
            raise GrokNativeVersionError(
                "version cache path is not a regular file"
            )
        if file_status.st_size > GROK_VERSION_MAX_BYTES:
            raise GrokNativeVersionError(
                f"version cache file exceeds {GROK_VERSION_MAX_BYTES} bytes"
            )
        raw_bytes = _read_cache_bytes(file_descriptor)
    finally:
        try:
            os.close(file_descriptor)
        except OSError:
            pass

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise GrokNativeVersionError(
            "version cache file is not valid UTF-8"
        ) from None



def _validate_fields(
    data: dict,
) -> tuple[int, str, str, str, str, str, str, float]:
    """Validate all cache fields and return the extracted values."""
    schema_version = data["schema_version"]
    if not isinstance(schema_version, int) or isinstance(
        schema_version, bool
    ):
        raise GrokNativeVersionError(
            "invalid schema_version: must be an integer"
        )
    if schema_version != GROK_VERSION_SCHEMA_VERSION:
        raise GrokNativeVersionError(
            f"unsupported schema_version: expected "
            f"{GROK_VERSION_SCHEMA_VERSION}, got {schema_version}"
        )

    client = data["client"]
    if client != _EXPECTED_CLIENT:
        raise GrokNativeVersionError(
            f"invalid client: expected {_EXPECTED_CLIENT!r}"
        )

    version = data["version"]
    _validate_version_string(version)

    build = data["build"]
    if not isinstance(build, str) or not _BUILD_RE.match(build):
        raise GrokNativeVersionError(
            "invalid build: must be a strict lowercase hex string"
        )

    channel = data["channel"]
    if not isinstance(channel, str) or not _CHANNEL_RE.match(channel):
        raise GrokNativeVersionError(
            "invalid channel: must be a non-empty safe token"
        )

    source = data["source"]
    if source != _EXPECTED_SOURCE:
        raise GrokNativeVersionError(
            f"invalid source: expected {_EXPECTED_SOURCE!r}"
        )

    observed_at = data["observed_at"]
    observed_epoch = _parse_observed_at(observed_at)
    return (
        schema_version, client, version, build,
        channel, source, observed_at, observed_epoch,
    )


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------


def resolve_grok_native_version(
    *,
    cache_path: Optional[str] = None,
    now: Optional[float] = None,
) -> tuple[GrokNativeVersionRecord, GrokNativeVersionMetadata]:
    """Read, validate, and return the Grok CLI version record.

    Re-reads the file on every call (no caching) so atomic file
    replacement is observed immediately without process restart.

    Raises :class:`GrokNativeVersionError` on any validation failure.
    Error messages never contain file contents or secrets.
    """
    path = _resolve_cache_path(cache_path)
    max_age = _resolve_max_age()
    current_time = now if now is not None else time.time()

    raw = _read_cache_text(path)

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        raise GrokNativeVersionError(
            "version cache file is not valid JSON"
        )

    if not isinstance(data, dict):
        raise GrokNativeVersionError(
            "version cache root must be a JSON object"
        )

    # --- Required fields ---
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise GrokNativeVersionError(
            f"version cache missing required fields: "
            f"{', '.join(sorted(missing))}"
        )

    unknown = set(data.keys()) - _REQUIRED_FIELDS
    if unknown:
        raise GrokNativeVersionError(
            f"version cache has unknown fields: "
            f"{', '.join(sorted(unknown))}"
        )

    # --- Field validation ---
    schema_version, client, version, build, channel, source, observed_at, observed_epoch = (
        _validate_fields(data)
    )

    # --- Time validation ---
    age = current_time - observed_epoch
    if observed_epoch > current_time:
        raise GrokNativeVersionError(
            "version cache observed_at is in the future"
        )

    if age > max_age:
        raise GrokNativeVersionError(
            f"version cache record is stale: age {age:.0f}s "
            f"exceeds max {max_age}s"
        )

    record = GrokNativeVersionRecord(
        schema_version=schema_version,
        client=client,
        version=version,
        build=build,
        channel=channel,
        source=source,
        observed_at=observed_at,
        observed_at_epoch=observed_epoch,
    )
    metadata = GrokNativeVersionMetadata(
        cache_path=path,
        version=version,
        build=build,
        channel=channel,
        source=source,
        age_seconds=max(0.0, age),
    )
    return record, metadata


def try_resolve_grok_native_version(
    *,
    cache_path: Optional[str] = None,
    now: Optional[float] = None,
) -> Optional[tuple[GrokNativeVersionRecord, GrokNativeVersionMetadata]]:
    """Non-raising variant: returns ``None`` on any validation failure."""
    try:
        return resolve_grok_native_version(
            cache_path=cache_path, now=now
        )
    except GrokNativeVersionError:
        return None
