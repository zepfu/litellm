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
    (default ``172800`` = 48 hours).  Must be a positive integer.
"""

from __future__ import annotations

import dataclasses
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
GROK_VERSION_FUTURE_SKEW_SECONDS = 300  # 5 min

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

# Strict dotted-numeric version: segments of digits separated by dots.
_VERSION_RE = re.compile(r"\A\d+(\.\d+)*\Z")
# Strict lowercase hex build hash.
_BUILD_RE = re.compile(r"\A[0-9a-f]+\Z")
# Safe channel token: alphanumeric, hyphens, underscores, dots.
_CHANNEL_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")
# RFC 3339 UTC timestamp ending in Z.
_RFC3339_Z_RE = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\Z"
)


class GrokNativeVersionError(Exception):
    """Raised when the version cache is missing, stale, or malformed.

    Messages never contain file contents or secrets.
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
    if explicit_path:
        return explicit_path
    return os.environ.get(
        GROK_VERSION_CACHE_PATH_ENV,
        GROK_VERSION_DEFAULT_CACHE_PATH,
    )


def _resolve_max_age() -> int:
    raw = os.environ.get(GROK_VERSION_CACHE_MAX_AGE_ENV)
    if raw is None:
        return GROK_VERSION_DEFAULT_MAX_AGE_SECONDS
    try:
        value = int(raw)
    except (ValueError, TypeError):
        raise GrokNativeVersionError(
            f"invalid {GROK_VERSION_CACHE_MAX_AGE_ENV}: "
            "must be a positive integer"
        )
    if value <= 0:
        raise GrokNativeVersionError(
            f"invalid {GROK_VERSION_CACHE_MAX_AGE_ENV}: "
            "must be a positive integer"
        )
    return value


def _validate_version_string(version: str) -> None:
    """Validate a strict dotted-numeric version string."""
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise GrokNativeVersionError(
            "invalid version: must be a strict dotted numeric string"
        )


def _parse_observed_at(value: str) -> float:
    """Parse an RFC 3339 UTC timestamp ending in Z to epoch seconds."""
    if not isinstance(value, str) or not _RFC3339_Z_RE.match(value):
        raise GrokNativeVersionError(
            "invalid observed_at: must be UTC RFC 3339 ending in Z"
        )
    try:
        # Python 3.7+ fromisoformat does not handle Z suffix.
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt.timestamp()
    except (ValueError, OverflowError):
        raise GrokNativeVersionError(
            "invalid observed_at: unparseable timestamp"
        )



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

    # --- File-level checks ---
    if not os.path.exists(path):
        raise GrokNativeVersionError(
            f"version cache file missing: {path}"
        )

    # Reject symlinks (security: prevent symlink attacks).
    if os.path.islink(path):
        raise GrokNativeVersionError(
            f"version cache file is a symlink: {path}"
        )

    try:
        st = os.stat(path)
    except OSError:
        raise GrokNativeVersionError(
            f"version cache file unreadable: {path}"
        )

    if not stat.S_ISREG(st.st_mode):
        raise GrokNativeVersionError(
            f"version cache path is not a regular file: {path}"
        )

    if st.st_size > GROK_VERSION_MAX_BYTES:
        raise GrokNativeVersionError(
            f"version cache file exceeds {GROK_VERSION_MAX_BYTES} bytes"
        )

    # --- Read and parse ---
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        raise GrokNativeVersionError(
            f"version cache file unreadable: {path}"
        )

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
    if age < -GROK_VERSION_FUTURE_SKEW_SECONDS:
        raise GrokNativeVersionError(
            "version cache observed_at is too far in the future"
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
