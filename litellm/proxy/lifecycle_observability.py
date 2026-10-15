"""Secret-safe process and proxy lifecycle provenance."""

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import signal
import time
from typing import Any, Optional

_ALLOWED_EVENTS = {
    "proxy_lifespan_start",
    "proxy_shutdown",
    "config_source_selected",
    "config_applied",
    "initialize_started",
    "initialize_completed",
}
_ALLOWED_CONFIG_SOURCES = {
    "environment_config_file",
    "worker_config_file",
    "worker_config_remote",
    "worker_config_object",
    "worker_config_json",
    "initialize_config_file",
    "initialize_parameters",
    "config_bucket_gcs",
    "config_bucket_s3",
    "no_config",
    "unspecified",
}
_SERVER_MODULE_PREFIXES = ("uvicorn.", "gunicorn.", "hypercorn.")
_CONFIG_REVISION_HMAC_KEY = secrets.token_bytes(32)
_MAX_CONFIG_REVISION_BYTES = 1_048_576

_state_pid = os.getpid()
_lifecycle_id = 0
_last_signal_number = 0
_last_signal_time_ns = 0
_signal_observer_status = "not_installed"
_signal_wrappers: dict[int, Any] = {}
_log_window_started_ns = 0
_log_window_count = 0
_suppressed_log_count = 0
_MAX_LOGS_PER_SECOND = 20
_LOG_WINDOW_NS = 1_000_000_000


def _reset_after_fork() -> None:
    global _state_pid, _lifecycle_id, _last_signal_number
    global _last_signal_time_ns, _signal_observer_status, _signal_wrappers
    global _log_window_started_ns, _log_window_count, _suppressed_log_count

    current_pid = os.getpid()
    if current_pid != _state_pid:
        _state_pid = current_pid
        _lifecycle_id = 0
        _last_signal_number = 0
        _last_signal_time_ns = 0
        _signal_observer_status = "not_installed"
        _signal_wrappers = {}
        _log_window_started_ns = 0
        _log_window_count = 0
        _suppressed_log_count = 0


def _is_server_signal_handler(handler: Any) -> bool:
    if not callable(handler):
        return False

    owner = getattr(handler, "__self__", None)
    modules = (
        getattr(handler, "__module__", ""),
        getattr(type(owner), "__module__", ""),
    )
    return any(module.startswith(_SERVER_MODULE_PREFIXES) for module in modules if isinstance(module, str))


def _install_signal_observers() -> str:
    global _last_signal_number, _last_signal_time_ns, _signal_wrappers

    _last_signal_number = 0
    _last_signal_time_ns = 0
    observed = 0
    for signal_name in ("SIGINT", "SIGTERM"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is None:
            continue

        try:
            current_handler = signal.getsignal(signal_number)
        except Exception:
            continue

        if current_handler is _signal_wrappers.get(int(signal_number)):
            observed += 1
            continue

        inherited_handler = getattr(current_handler, "_litellm_lifecycle_original_handler", None)
        if inherited_handler is not None:
            current_handler = inherited_handler

        if not _is_server_signal_handler(current_handler):
            continue

        def observe_signal(
            received_signal: int,
            frame: Any,
            original_handler: Any = current_handler,
        ) -> None:
            global _last_signal_number, _last_signal_time_ns
            _last_signal_number = int(received_signal)
            _last_signal_time_ns = time.time_ns()
            original_handler(received_signal, frame)

        observe_signal._litellm_lifecycle_original_handler = current_handler
        try:
            signal.signal(signal_number, observe_signal)
        except Exception:
            continue

        _signal_wrappers[int(signal_number)] = observe_signal
        observed += 1

    return "installed" if observed else "unavailable"


def _process_start_ticks() -> int:
    try:
        with open("/proc/self/stat", encoding="ascii") as process_stat:
            stat_line = process_stat.read()
        fields = stat_line[stat_line.rfind(")") + 2 :].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return 0


def _container_instance_id() -> str:
    try:
        with open("/proc/self/cgroup", encoding="ascii") as cgroup_file:
            cgroup = cgroup_file.read()
    except OSError:
        return "unavailable"

    match = re.search(
        r"(?:docker|containerd|cri-containerd|crio|libpod)[-/]([0-9a-f]{12,64})",
        cgroup,
        re.IGNORECASE,
    )
    return match.group(1)[:12].lower() if match else "unavailable"


def _config_file_metadata(config_file_path: Optional[str]) -> tuple[str, int, int, int]:
    if not config_file_path:
        return "not_file_backed", 0, 0, 0
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", config_file_path):
        return "remote", 0, 0, 0

    try:
        config_stat = os.stat(config_file_path)
    except (OSError, TypeError, ValueError):
        return "unavailable", 0, 0, 0

    return (
        "file_stat",
        int(config_stat.st_mtime_ns),
        int(config_stat.st_size),
        int(config_stat.st_ino),
    )


def _reserve_lifecycle_log_slot() -> Optional[int]:
    global _log_window_started_ns, _log_window_count, _suppressed_log_count

    current_time_ns = time.monotonic_ns()
    if current_time_ns < _log_window_started_ns or current_time_ns - _log_window_started_ns >= _LOG_WINDOW_NS:
        _log_window_started_ns = current_time_ns
        _log_window_count = 0

    if _log_window_count >= _MAX_LOGS_PER_SECOND:
        _suppressed_log_count += 1
        return None

    _log_window_count += 1
    suppressed_count = _suppressed_log_count
    _suppressed_log_count = 0
    return suppressed_count


def _config_revision_id(config_snapshot: Any) -> tuple[str, str]:
    if config_snapshot is None:
        return "not_provided", "unavailable"

    digest = hmac.new(_CONFIG_REVISION_HMAC_KEY, digestmod=hashlib.sha256)
    size_bytes = 0
    try:
        encoder = json.JSONEncoder(
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        for chunk in encoder.iterencode(config_snapshot):
            if len(chunk) > _MAX_CONFIG_REVISION_BYTES:
                return "too_large", "unavailable"
            encoded_chunk = chunk.encode("utf-8")
            size_bytes += len(encoded_chunk)
            if size_bytes > _MAX_CONFIG_REVISION_BYTES:
                return "too_large", "unavailable"
            digest.update(encoded_chunk)
    except Exception:
        return "unavailable", "unavailable"

    return "hmac", digest.hexdigest()[:32]


def log_proxy_lifecycle(
    logger: logging.Logger,
    *,
    event: str,
    config_source: str = "unspecified",
    config_file_path: Optional[str] = None,
    config_snapshot: Any = None,
) -> None:
    _reset_after_fork()
    suppressed_count = _reserve_lifecycle_log_slot()
    if suppressed_count is None:
        return

    config_file_status, config_mtime_ns, config_size_bytes, config_inode = _config_file_metadata(config_file_path)
    config_revision_status, config_revision = _config_revision_id(config_snapshot)
    logger.info(
        "proxy_lifecycle event=%s lifecycle_id=%d pid=%d ppid=%d "
        "process_start_ticks=%d container_instance_id=%s timestamp_unix_ns=%d "
        "config_source=%s config_file_status=%s config_revision_status=%s "
        "config_revision_id=%s config_mtime_ns=%d config_size_bytes=%d "
        "config_inode=%d signal_observer_status=%s signal_number=%d "
        "signal_timestamp_unix_ns=%d suppressed_events=%d",
        event if event in _ALLOWED_EVENTS else "unknown",
        _lifecycle_id,
        os.getpid(),
        os.getppid(),
        _process_start_ticks(),
        _container_instance_id(),
        time.time_ns(),
        config_source if config_source in _ALLOWED_CONFIG_SOURCES else "unspecified",
        config_file_status,
        config_revision_status,
        config_revision,
        config_mtime_ns,
        config_size_bytes,
        config_inode,
        _signal_observer_status,
        _last_signal_number,
        _last_signal_time_ns,
        suppressed_count,
    )


def begin_proxy_lifecycle(logger: logging.Logger) -> int:
    global _lifecycle_id, _signal_observer_status

    _reset_after_fork()
    _lifecycle_id += 1
    _signal_observer_status = _install_signal_observers()
    log_proxy_lifecycle(logger, event="proxy_lifespan_start")
    return _lifecycle_id
