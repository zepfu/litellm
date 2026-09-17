"""Immutable-inventory directory loader for AAWM alias-routing config (CFG-002).

Scans the alias-config directory into a frozen ``_ConfigInventory`` of
captured raw bytes, content digests, and filesystem identity metadata using
descriptor-anchored, component-by-component ``O_NOFOLLOW`` traversal.
Compilation operates exclusively on the captured bytes -- never on live
filesystem paths -- so no subset, nondeterministic, absolute, empty, or
``..`` inventory input can be injected.

``activate_alias_config_directory`` scans once, compiles captured bytes,
scans the complete tree again immediately before activation, requires exact
inventory equality, then atomically swaps the process-local snapshot.  Drift
in add/remove/edit/replace-file/replace-root/replace-nested-directory fails
closed with readiness 503 and no static fallback.

This detects bounded-load changes between the two scans but is **not** a
filesystem transaction; changes that land after the final revalidation load
on the next restart.

Fail-closed contract: any parse, schema-validation, cross-file duplicate,
merge-conflict, compile error, symlink, unsupported file type, invalid
UTF-8, empty-aliases result, or inventory drift prevents activation and
marks startup as failed.  A failed startup clears any prior snapshot,
blocks ``/health/readiness`` (503), and prohibits fallback to a partial or
stale snapshot.

Hot-refresh consensus across workers is explicitly out of scope; each worker
performs its own deterministic directory load at startup.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import stat
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

import yaml

from litellm.secret_managers.codex_oauth_inventory import (
    CodexOAuthInventoryError,
    codex_oauth_inventory_generation_digest,
    is_codex_oauth_inventory_generation,
    load_codex_oauth_inventory,
)

from .config_compiler import compile_yaml
from .config_snapshot import RoutingSnapshot
from .snapshot_select import (
    get_active_routing_snapshot,
    set_active_routing_snapshot,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config directory
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_DIR: Path = (
    Path(__file__).resolve().parents[2] / "aawm_alias_config"
)

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})

_MAX_CONFIG_FILE_BYTES = 10 * 1024 * 1024

_CODEX_OAUTH_INVENTORY_GENERATION_QUERY = """
SELECT id, observed_at, created_at, status, error_class, metadata
FROM public.provider_auth_current
WHERE environment = $1::text
  AND provider = 'openai'
  AND auth_family = 'codex_oauth_inventory'
  AND credential_scope = 'inventory'
  AND COALESCE(auth_file_hash, '') = ''
ORDER BY observed_at DESC, id DESC
LIMIT 1
"""
_CODEX_OAUTH_GENERATION_QUERY_TIMEOUT_SECONDS = 1.5
_CODEX_OAUTH_INVENTORY_OBSERVATION_DEFAULT_CADENCE_SECONDS = 300.0
_CODEX_OAUTH_INVENTORY_OBSERVATION_FRESHNESS_MULTIPLIER = 2.0
_CODEX_OAUTH_OBSERVATION_ENVIRONMENT_ENV = (
    "AAWM_CODEX_OAUTH_QUOTA_OBSERVATION_ENVIRONMENT"
)
_CODEX_OAUTH_RUNTIME_ENVIRONMENT_VARS = (
    "AAWM_LITELLM_ENVIRONMENT",
    "LITELLM_INSTANCE_ENVIRONMENT",
    "LITELLM_ENVIRONMENT",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigDirectoryError(Exception):
    """Raised when the config directory cannot produce a valid merged document."""


class InventoryDriftError(ConfigDirectoryError):
    """Raised when the pre-activation rescan detects filesystem drift."""


# ---------------------------------------------------------------------------
# Immutable inventory records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _InventoryFile:
    """One captured regular file's immutable record."""

    relative_name: str
    raw_bytes: bytes
    content_digest: str
    size: int
    st_dev: int
    st_ino: int
    st_mtime_ns: int
    st_ctime_ns: int


@dataclass(frozen=True, slots=True)
class _InventoryDir:
    """One scanned directory's identity record."""

    relative_name: str
    st_dev: int
    st_ino: int
    st_mtime_ns: int
    st_ctime_ns: int
    child_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ConfigInventory:
    """Frozen, deterministic snapshot of the complete config directory tree.

    Contains captured raw bytes (never live paths), content digests, size,
    filesystem identity metadata (st_dev, st_ino, st_mtime_ns, st_ctime_ns),
    directory/root relative names, and sorted child names.
    """

    root_relative_name: str
    root_st_dev: int
    root_st_ino: int
    root_st_mtime_ns: int
    root_st_ctime_ns: int
    files: tuple[_InventoryFile, ...]
    directories: tuple[_InventoryDir, ...]

    @property
    def file_names(self) -> tuple[str, ...]:
        return tuple(f.relative_name for f in self.files)

    def identity_key(self) -> tuple:
        """Deterministic equality key for drift detection."""
        return (
            self.root_st_dev,
            self.root_st_ino,
            self.root_st_mtime_ns,
            self.root_st_ctime_ns,
            tuple(
                (f.relative_name, f.content_digest, f.size, f.st_dev, f.st_ino, f.st_mtime_ns, f.st_ctime_ns)
                for f in self.files
            ),
            tuple(
                (d.relative_name, d.st_dev, d.st_ino, d.st_mtime_ns, d.st_ctime_ns, d.child_names)
                for d in self.directories
            ),
        )


# ---------------------------------------------------------------------------
# Descriptor-anchored scanner
# ---------------------------------------------------------------------------


def _suffix_of(name: str) -> str:
    """Return the file suffix (including dot) or empty string."""
    idx = name.rfind(".")
    if idx <= 0:
        return ""
    return name[idx:]


def _bounded_read(fd: int, rel: str) -> bytes:
    """Read up to _MAX_CONFIG_FILE_BYTES from *fd*, handling short reads."""
    limit = _MAX_CONFIG_FILE_BYTES + 1
    chunks: list[bytes] = []
    total = 0
    while total < limit:
        try:
            chunk = os.read(fd, limit - total)
        except InterruptedError:
            continue
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    raw_bytes = b"".join(chunks)
    if len(raw_bytes) > _MAX_CONFIG_FILE_BYTES:
        raise ConfigDirectoryError(
            f"config file {rel} exceeds maximum size ({_MAX_CONFIG_FILE_BYTES} bytes)"
        )
    return raw_bytes


def _scan_inventory(config_dir: Path) -> _ConfigInventory:  # noqa: PLR0915
    """Scan *config_dir* into a frozen ``_ConfigInventory``.

    Uses descriptor-anchored, component-by-component ``O_NOFOLLOW`` traversal
    with pre/post ``fstat`` consistency checks.  All file content is captured
    as raw bytes; no live ``Path`` references escape the scanner.

    Fail-closed discovery contract:
    - The config directory root must not be a symlink.
    - Symlinks are rejected regardless of suffix.
    - Non-regular files (FIFOs, sockets, devices) are rejected.
    - Directories with any YAML-like suffix (any case) are rejected.
    - Unreadable nested directories are rejected.
    - Only lowercase ``.yaml`` / ``.yml`` regular files are accepted.
    - ``__init__.py`` is silently ignored (package infrastructure).
    - The exact ``__pycache__`` directory is silently ignored (package
      infrastructure); its contents are never inventoried.
    - Any other regular file is rejected (fail closed).
    """
    if config_dir.is_symlink():
        raise ConfigDirectoryError("config directory root must not be a symlink")
    if not config_dir.is_dir():
        raise ConfigDirectoryError(
            "config directory does not exist or is not a directory"
        )

    with ExitStack() as stack:
        # Open root anchored.
        try:
            root_fd = os.open(
                str(config_dir), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
        except OSError as exc:
            raise ConfigDirectoryError(
                f"cannot open config directory root: {type(exc).__name__}"
            ) from exc
        stack.callback(os.close, root_fd)

        # Pre-stat root.
        root_pre = os.fstat(root_fd)
        if not stat.S_ISDIR(root_pre.st_mode):
            raise ConfigDirectoryError("config directory root is not a directory")

        files: list[_InventoryFile] = []
        directories: list[_InventoryDir] = []

        def _scan_dir(dir_fd: int, rel_prefix: str) -> None:
            dir_stat = os.fstat(dir_fd)
            try:
                entries = sorted(os.listdir(dir_fd))
            except OSError as exc:
                raise ConfigDirectoryError(
                    f"cannot list directory {rel_prefix or '.'}: {type(exc).__name__}"
                ) from exc

            directories.append(_InventoryDir(
                relative_name=rel_prefix or ".",
                st_dev=dir_stat.st_dev,
                st_ino=dir_stat.st_ino,
                st_mtime_ns=dir_stat.st_mtime_ns,
                st_ctime_ns=dir_stat.st_ctime_ns,
                child_names=tuple(entries),
            ))

            for name in entries:
                child_rel = f"{rel_prefix}/{name}" if rel_prefix else name
                # lstat via dir fd to detect symlinks without following.
                try:
                    child_lstat = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except OSError as exc:
                    raise ConfigDirectoryError(
                        f"cannot stat {child_rel}: {type(exc).__name__}"
                    ) from exc

                if stat.S_ISLNK(child_lstat.st_mode):
                    raise ConfigDirectoryError(
                        f"symlink not allowed in config directory: {child_rel}"
                    )

                if stat.S_ISDIR(child_lstat.st_mode):
                    suffix = _suffix_of(name)
                    if suffix.lower() in _YAML_SUFFIXES:
                        raise ConfigDirectoryError(
                            f"directory with YAML suffix not allowed: {child_rel}"
                        )
                    if name == "__pycache__":
                        # Python bytecode cache: package infrastructure, not
                        # config content.  Skip without recursing so generated
                        # ``*.pyc`` files never reach the inventory.
                        continue
                    try:
                        child_fd = os.open(
                            name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=dir_fd,
                        )
                    except OSError as exc:
                        raise ConfigDirectoryError(
                            f"unreadable directory in config directory: {child_rel}"
                        ) from exc
                    stack.callback(os.close, child_fd)
                    _scan_dir(child_fd, child_rel)
                    continue

                if not stat.S_ISREG(child_lstat.st_mode):
                    raise ConfigDirectoryError(
                        f"unsupported non-regular file in config directory: {child_rel}"
                    )

                # Regular file: suffix policy.
                suffix = _suffix_of(name)
                if suffix in _YAML_SUFFIXES:
                    pass  # accepted
                elif suffix.lower() in _YAML_SUFFIXES:
                    raise ConfigDirectoryError(
                        f"unsupported case-variant YAML suffix in config directory: {child_rel}"
                    )
                elif name == "__init__.py":
                    continue
                else:
                    raise ConfigDirectoryError(
                        f"unsupported file type in config directory: {child_rel}"
                    )

                # Open file with O_NOFOLLOW | O_NONBLOCK.
                try:
                    file_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=dir_fd,
                    )
                except OSError as exc:
                    raise ConfigDirectoryError(
                        f"cannot read config file {child_rel}: {type(exc).__name__}"
                    ) from exc
                stack.callback(os.close, file_fd)

                # Pre-fstat.
                pre = os.fstat(file_fd)
                if not stat.S_ISREG(pre.st_mode):
                    raise ConfigDirectoryError(
                        f"cannot read config file {child_rel}: not a regular file after open"
                    )

                # Bounded read.
                raw_bytes = _bounded_read(file_fd, child_rel)

                # Post-fstat consistency.
                post = os.fstat(file_fd)
                if (
                    post.st_dev != pre.st_dev
                    or post.st_ino != pre.st_ino
                    or post.st_mtime_ns != pre.st_mtime_ns
                    or post.st_size != pre.st_size
                ):
                    raise ConfigDirectoryError(
                        f"config file {child_rel} changed during read (fstat inconsistency)"
                    )

                files.append(_InventoryFile(
                    relative_name=child_rel,
                    raw_bytes=raw_bytes,
                    content_digest=hashlib.sha256(raw_bytes).hexdigest(),
                    size=len(raw_bytes),
                    st_dev=pre.st_dev,
                    st_ino=pre.st_ino,
                    st_mtime_ns=pre.st_mtime_ns,
                    st_ctime_ns=pre.st_ctime_ns,
                ))

        _scan_dir(root_fd, "")

        # Post-stat root consistency.
        root_post = os.fstat(root_fd)
        if (
            root_post.st_dev != root_pre.st_dev
            or root_post.st_ino != root_pre.st_ino
        ):
            raise ConfigDirectoryError(
                "config directory root changed during scan (fstat inconsistency)"
            )

    # Sort files by relative_name for deterministic ordering.
    files.sort(key=lambda f: f.relative_name)
    directories.sort(key=lambda d: d.relative_name)

    return _ConfigInventory(
        root_relative_name=".",
        root_st_dev=root_pre.st_dev,
        root_st_ino=root_pre.st_ino,
        root_st_mtime_ns=root_pre.st_mtime_ns,
        root_st_ctime_ns=root_pre.st_ctime_ns,
        files=tuple(files),
        directories=tuple(directories),
    )


# ---------------------------------------------------------------------------
# Compile from captured inventory bytes (private)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ParsedFile:
    """Intermediate: one validated file's contribution to the merge."""

    relative_path: str
    raw_text: str
    raw_data: dict[str, Any]
    alias_names: tuple[str, ...]
    defaults: Optional[dict[str, Any]]


def _compile_inventory(inventory: _ConfigInventory) -> RoutingSnapshot:
    """Compile captured inventory bytes into a ``RoutingSnapshot``.

    Operates exclusively on the frozen ``inventory.files`` raw bytes -- never
    on live filesystem paths.  No subset, nondeterministic, absolute, empty,
    or ``..`` inventory input can be injected.
    """
    from . import config_schema as schema

    if not inventory.files:
        raise ConfigDirectoryError("no YAML config files found in inventory")

    parsed: list[_ParsedFile] = []
    for inv_file in inventory.files:
        rel = inv_file.relative_name
        # Validate relative name: no absolute, no '..'.
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ConfigDirectoryError(
                f"invalid relative name in inventory: {rel}"
            )
        try:
            raw_text = inv_file.raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigDirectoryError(
                f"cannot read config file {rel}: {type(exc).__name__}"
            ) from exc

        try:
            raw_data = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ConfigDirectoryError(
                f"invalid YAML in {rel}: {type(exc).__name__}"
            ) from exc

        if not isinstance(raw_data, dict):
            raise ConfigDirectoryError(
                f"config file {rel} must be a YAML mapping, got {type(raw_data).__name__}"
            )

        document = schema.RoutingConfigDocument.model_validate(raw_data)

        _lower_seen: dict[str, str] = {}
        for alias in document.aliases:
            key = alias.name.lower()
            if key in _lower_seen:
                raise ConfigDirectoryError(
                    f"case-insensitive duplicate alias {alias.name!r} in {rel} "
                    f"(conflicts with {_lower_seen[key]!r})"
                )
            _lower_seen[key] = alias.name

        alias_names = tuple(alias.name for alias in document.aliases)
        defaults_raw = raw_data.get("defaults")
        defaults = defaults_raw if isinstance(defaults_raw, dict) and defaults_raw else None

        parsed.append(_ParsedFile(
            relative_path=rel,
            raw_text=raw_text,
            raw_data=raw_data,
            alias_names=alias_names,
            defaults=defaults,
        ))

    merged_raw = _merge_parsed_files(parsed)

    if not merged_raw.get("aliases"):
        raise ConfigDirectoryError(
            "merged config document contains no aliases; "
            "an empty alias set is unhealthy and cannot enable fallback"
        )

    merged_yaml = yaml.dump(
        merged_raw,
        default_flow_style=False,
        sort_keys=True,
        allow_unicode=False,
    )
    return compile_yaml(merged_yaml)


def _merge_parsed_files(
    parsed: Sequence[_ParsedFile],
) -> dict[str, Any]:
    """Merge validated per-file documents into one combined raw dict."""
    seen_aliases: dict[str, str] = {}
    all_aliases: list[Any] = []
    merged_defaults: Optional[dict[str, Any]] = None
    defaults_source: Optional[str] = None

    for pf in parsed:
        for name in pf.alias_names:
            key = name.lower()
            if key in seen_aliases:
                raise ConfigDirectoryError(
                    f"duplicate alias {name!r} (case-insensitive match) "
                    f"defined in both "
                    f"{seen_aliases[key]!r} and {pf.relative_path!r}"
                )
            seen_aliases[key] = pf.relative_path

        if pf.defaults is not None:
            if merged_defaults is None:
                merged_defaults = pf.defaults
                defaults_source = pf.relative_path
            elif pf.defaults != merged_defaults:
                raise ConfigDirectoryError(
                    f"conflicting defaults blocks in {defaults_source!r} "
                    f"and {pf.relative_path!r}; defaults must be unambiguous"
                )

        file_aliases = pf.raw_data.get("aliases")
        if isinstance(file_aliases, list):
            all_aliases.extend(file_aliases)

    merged: dict[str, Any] = {}
    if merged_defaults is not None:
        merged["defaults"] = merged_defaults
    merged["aliases"] = all_aliases
    return merged


# ---------------------------------------------------------------------------
# Public compile API
# ---------------------------------------------------------------------------


def compile_directory(config_dir: Path) -> RoutingSnapshot:
    """Scan *config_dir*, validate, merge, and compile all YAML.

    Performs its own full scan -- no ``files=`` override exists.  Returns a
    fully built immutable ``RoutingSnapshot``.  Raises
    ``ConfigDirectoryError``, ``ConfigCompileError``, or
    ``pydantic.ValidationError`` on any failure.
    """
    inventory = _scan_inventory(config_dir)
    return _compile_inventory(inventory)


def compile_directory_with_file_names(
    config_dir: Path,
) -> tuple[RoutingSnapshot, tuple[str, ...]]:
    """Compile the full config directory and return its file names.

    This helper preserves startup scanner semantics (`_scan_inventory`) while
    exposing the deterministic file-name inventory for readout paths that need
    to report complete source metadata without recompiling the directory.
    """
    inventory = _scan_inventory(config_dir)
    return _compile_inventory(inventory), inventory.file_names


# ---------------------------------------------------------------------------
# Startup state (process-local, per-worker)
# ---------------------------------------------------------------------------


@dataclass
class _StartupState:
    """Mutable process-local startup bookkeeping."""

    activated: bool = False
    failed: bool = False
    error: Optional[str] = None
    error_class: Optional[str] = None
    snapshot: Optional[RoutingSnapshot] = None
    files_loaded: tuple[str, ...] = ()
    config_dir: Optional[str] = None


_state_lock = threading.Lock()
_startup_state = _StartupState()


def activate_alias_config_directory(
    config_dir: Optional[Path] = None,
) -> None:
    """Load, compile, and atomically activate the alias-config directory.

    Scans once, compiles captured bytes, scans the complete tree again
    immediately before activation, requires exact inventory equality, then
    swaps.  Drift in add/remove/edit/replace-file/replace-root/replace-
    nested-directory fails closed with readiness 503 and no static fallback.

    This detects bounded-load changes but is not a filesystem transaction;
    changes after final revalidation load on next restart.

    Called once per worker during proxy startup, before the lifespan yields.
    On any failure the startup state is marked failed, any prior snapshot is
    cleared, and readiness returns 503.  This function never raises.
    """
    resolved_dir = config_dir if config_dir is not None else DEFAULT_CONFIG_DIR
    try:
        # First scan: capture immutable inventory.
        inventory = _scan_inventory(resolved_dir)
        # Compile from captured bytes only.
        snapshot = _compile_inventory(inventory)
        # Second scan: revalidate immediately before activation.
        revalidation = _scan_inventory(resolved_dir)
        if inventory.identity_key() != revalidation.identity_key():
            raise InventoryDriftError(
                "config directory changed between scan and activation"
            )
        # Atomic install.
        set_active_routing_snapshot(snapshot)
        with _state_lock:
            _startup_state.activated = True
            _startup_state.failed = False
            _startup_state.error = None
            _startup_state.error_class = None
            _startup_state.snapshot = snapshot
            _startup_state.files_loaded = inventory.file_names
            _startup_state.config_dir = str(resolved_dir)
        logger.info(
            "AAWM alias-config directory activated: files=%s, "
            "config_version=%s, config_hash=%s, aliases=%s, result=success",
            list(inventory.file_names),
            snapshot.config_version,
            snapshot.config_hash,
            sorted(snapshot.aliases.keys()),
        )
    except Exception as exc:
        with _state_lock:
            _startup_state.activated = False
            _startup_state.failed = True
            _startup_state.error = type(exc).__name__
            _startup_state.error_class = type(exc).__name__
            _startup_state.snapshot = None
            _startup_state.config_dir = str(resolved_dir)
        set_active_routing_snapshot(None)
        logger.error(
            "AAWM alias-config directory startup FAILED (fail-closed): "
            "error_class=%s config_dir=%s",
            type(exc).__name__,
            resolved_dir.name,
        )


# ---------------------------------------------------------------------------
# Readiness gate + sanitized status
# ---------------------------------------------------------------------------


def is_startup_healthy() -> bool:
    """Return ``True`` only if the directory load succeeded and a snapshot is active."""
    with _state_lock:
        return _startup_state.activated and not _startup_state.failed


def is_startup_failed() -> bool:
    """Return ``True`` if the directory load was attempted and failed."""
    with _state_lock:
        return _startup_state.failed


def is_startup_not_loaded() -> bool:
    """Return ``True`` if startup was never attempted (not_loaded state)."""
    with _state_lock:
        return not _startup_state.activated and not _startup_state.failed


def _resolve_codex_oauth_observation_environment() -> Optional[str]:
    """Resolve the environment label written by the provider-status sidecar."""
    explicit = os.getenv(_CODEX_OAUTH_OBSERVATION_ENVIRONMENT_ENV)
    if explicit is not None:
        return explicit.strip() or None
    for environment_var in _CODEX_OAUTH_RUNTIME_ENVIRONMENT_VARS:
        environment = os.getenv(environment_var, "").strip()
        if environment:
            return environment
    return None


def _codex_oauth_row_values(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _codex_oauth_row_metadata(row: Any) -> Optional[Mapping[str, Any]]:
    metadata = _codex_oauth_row_values(row).get("metadata")
    if isinstance(metadata, Mapping):
        return metadata
    if isinstance(metadata, (str, bytes, bytearray)):
        try:
            decoded = json.loads(metadata)
        except (TypeError, ValueError):
            return None
        if isinstance(decoded, Mapping):
            return decoded
    return None


def _codex_oauth_generation_from_row(
    row: Any,
) -> tuple[str, Optional[str]]:
    metadata = _codex_oauth_row_metadata(row)
    if metadata is None:
        return "missing", None
    generation = metadata.get("codex_oauth_inventory_generation")
    if generation is None or generation == "":
        return "missing", None
    if is_codex_oauth_inventory_generation(generation):
        return "valid", generation
    return "invalid", None


def _codex_oauth_observation_timestamp(row: Any) -> Optional[datetime]:
    value = _codex_oauth_row_values(row).get("observed_at")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _codex_oauth_inventory_observation_freshness_budget_seconds(
    metadata: Optional[Mapping[str, Any]],
) -> float:
    cadence = _CODEX_OAUTH_INVENTORY_OBSERVATION_DEFAULT_CADENCE_SECONDS
    if metadata is not None:
        candidate = metadata.get("observation_cadence_seconds")
        try:
            parsed = float(candidate)
        except (TypeError, ValueError):
            parsed = cadence
        if math.isfinite(parsed) and parsed > 0:
            cadence = parsed
    return cadence * _CODEX_OAUTH_INVENTORY_OBSERVATION_FRESHNESS_MULTIPLIER


async def get_codex_oauth_inventory_generation_status(  # noqa: PLR0915 - bounded readiness evidence parser
    *,
    local_generation: Optional[str],
    get_pool: Optional[Callable[[], Awaitable[Any]]] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Compare the local inventory digest with current sidecar observations.

    Persisted observations remain informational.  A confirmed mismatch is
    surfaced to readiness, while missing or malformed observations remain
    visible as ``unknown`` so a sidecar/database startup race does not stop
    live routing.
    """
    safe_local_generation = (
        local_generation
        if is_codex_oauth_inventory_generation(local_generation)
        else None
    )
    environment = _resolve_codex_oauth_observation_environment()
    result: dict[str, Any] = {
        "status": "unknown",
        "health": "unknown",
        "readiness_degraded": False,
        "environment": environment,
        "local_generation": safe_local_generation,
        "sidecar_generation": None,
        "sidecar_generations": [],
        "observation_count": 0,
        "valid_observation_count": 0,
        "observation_id": None,
        "observed_at": None,
        "observation_age_seconds": None,
        "freshness_budget_seconds": None,
        "inventory_state": None,
        "sidecar_status": None,
        "reason": None,
    }
    if environment is None:
        result["reason"] = "observation_environment_unconfigured"
        return result
    if safe_local_generation is None:
        result["reason"] = "local_generation_unavailable"
        return result

    async def _fetch_rows() -> Any:
        from ..aawm_context_query import (
            _aawm_pool_fetch,
            _get_aawm_callback_pool,
        )

        pool = await (get_pool or _get_aawm_callback_pool)()
        return await _aawm_pool_fetch(
            pool,
            _CODEX_OAUTH_INVENTORY_GENERATION_QUERY,
            environment,
            get_timeout=lambda: _CODEX_OAUTH_GENERATION_QUERY_TIMEOUT_SECONDS,
        )

    try:
        rows = await asyncio.wait_for(
            _fetch_rows(),
            timeout=_CODEX_OAUTH_GENERATION_QUERY_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        result["reason"] = "observation_query_failed"
        result["error_class"] = exc.__class__.__name__
        return result

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        result["reason"] = "observation_rows_invalid"
        return result
    if not rows:
        result["reason"] = "no_current_codex_observation"
        return result

    row = rows[0]
    values = _codex_oauth_row_values(row)
    metadata = _codex_oauth_row_metadata(row)
    result["observation_count"] = 1
    result["observation_id"] = values.get("id")
    result["sidecar_status"] = values.get("status")
    inventory_state = metadata.get("inventory_state") if metadata else None
    result["inventory_state"] = inventory_state
    observed_at = _codex_oauth_observation_timestamp(row)
    if observed_at is None:
        result["reason"] = "observation_timestamp_invalid"
        return result
    result["observed_at"] = observed_at.isoformat().replace("+00:00", "Z")
    freshness_budget = _codex_oauth_inventory_observation_freshness_budget_seconds(
        metadata
    )
    result["freshness_budget_seconds"] = freshness_budget
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)
    age_seconds = (current_time - observed_at).total_seconds()
    result["observation_age_seconds"] = age_seconds
    if age_seconds < 0:
        result["reason"] = "observation_timestamp_in_future"
        return result
    if age_seconds > freshness_budget:
        result["reason"] = "observation_stale"
        return result

    generation_state, generation = _codex_oauth_generation_from_row(row)
    if generation_state == "valid" and generation is not None:
        result["valid_observation_count"] = 1
        result["sidecar_generation"] = generation
        result["sidecar_generations"] = [generation]
    if values.get("status") != "observed" or inventory_state != "valid":
        if inventory_state == "unconfigured":
            result["reason"] = "inventory_unconfigured"
        elif inventory_state == "invalid" or generation_state == "invalid":
            result["reason"] = "inventory_invalid"
        elif values.get("status") != "observed":
            result["reason"] = "inventory_observation_failed"
        else:
            result["reason"] = "sidecar_generation_missing"
        return result
    if generation_state != "valid" or generation is None:
        result["reason"] = (
            "sidecar_generation_invalid"
            if generation_state == "invalid"
            else "sidecar_generation_missing"
        )
        return result
    if generation == safe_local_generation:
        result.update(
            {
                "status": "matched",
                "health": "healthy",
                "reason": "generation_match",
            }
        )
    else:
        result.update(
            {
                "status": "mismatch",
                "health": "degraded",
                "readiness_degraded": True,
                "reason": "generation_mismatch",
            }
        )
    return result


def get_startup_status() -> dict[str, Any]:
    """Return sanitized activation metadata for health/readiness responses.

    Exposes only: state, config_hash, config_version, config_epoch, relative
    file names, alias names, alias_count, activation result.
    Failed state exposes only the error class name.
    Never exposes secrets, raw YAML, or absolute paths.

    Active config identity fields (config_hash, config_version, config_epoch,
    aliases) are read from the live active routing snapshot holder so that
    readiness reflects post-refresh state, not just the startup snapshot.
    File/source metadata and startup health gates remain owned by
    ``_startup_state``.
    """
    with _state_lock:
        st = _startup_state
        if st.failed:
            return {
                "state": "failed",
                "error_class": st.error_class or "Unknown",
            }
        if not st.activated:
            return {"state": "not_loaded"}
        startup_snapshot = st.snapshot
        assert startup_snapshot is not None
        files_loaded = st.files_loaded

    # Outside _state_lock: read the live active snapshot from the routing
    # holder (which has its own lock) to avoid nested-lock ordering issues.
    active_snapshot = get_active_routing_snapshot()
    identity_snapshot = (
        active_snapshot if active_snapshot is not None else startup_snapshot
    )
    try:
        inventory_generation = codex_oauth_inventory_generation_digest(
            load_codex_oauth_inventory()
        )
    except CodexOAuthInventoryError:
        inventory_generation = None

    return {
        "state": "active",
        "config_hash": identity_snapshot.config_hash,
        "config_version": identity_snapshot.config_version,
        "config_epoch": identity_snapshot.config_epoch,
        "files": list(files_loaded),
        "aliases": sorted(identity_snapshot.aliases.keys()),
        "alias_count": len(identity_snapshot.aliases),
        "codex_oauth_inventory_generation": inventory_generation,
        "activation_result": "success",
    }


def reset_startup_state() -> None:
    """Reset process-local startup state (test helper only)."""
    with _state_lock:
        _startup_state.activated = False
        _startup_state.failed = False
        _startup_state.error = None
        _startup_state.error_class = None
        _startup_state.snapshot = None
        _startup_state.files_loaded = ()
        _startup_state.config_dir = None


def set_startup_files_loaded(file_names: Sequence[str]) -> None:
    """Update readiness `files` metadata without changing activation state."""
    with _state_lock:
        _startup_state.files_loaded = tuple(file_names)
