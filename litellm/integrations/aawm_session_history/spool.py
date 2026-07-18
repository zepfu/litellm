"""AAWM session_history spool module."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import re
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

from litellm._logging import verbose_logger

from litellm.integrations.aawm_session_history import runtime as _runtime


def _identity_host():
    return _runtime._identity_host()


def _state(name: str) -> Any:
    return _runtime._state(name)


def _mirror_state(name: str) -> None:
    _runtime._mirror_state(name)


def _set_state(name: str, value):
    _runtime._set_state(name, value)


def _call(name: str, *args, **kwargs):
    return _runtime._call(name, *args, **kwargs)


def _writer_time():
    return _runtime._writer_time()


def _writer_importlib():
    return _runtime._writer_importlib()


def _writer_get_secret_str(secret_name: str):
    return _runtime._writer_get_secret_str(secret_name)


def _get_persist_session_history_records():
    return _runtime._get_persist_session_history_records()


# Shared constants/state live in runtime; bind local names for readability and
# for historical call-sites that referenced module globals directly.
from litellm.integrations.aawm_session_history.runtime import (  # noqa: E402,F401
    _AAWM_SESSION_HISTORY_APPLICATION_NAME,
    _AAWM_SESSION_HISTORY_BATCH_SIZE,
    _AAWM_SESSION_HISTORY_COMMAND_TIMEOUT_SECONDS,
    _AAWM_SESSION_HISTORY_DEGRADED_SPOOL_SECONDS,
    _AAWM_SESSION_HISTORY_FAILED_FLUSH_MAX_RETRIES,
    _AAWM_SESSION_HISTORY_FAILED_FLUSH_RETRY_SECONDS,
    _AAWM_SESSION_HISTORY_FLUSH_INTERVAL_SECONDS,
    _AAWM_SESSION_HISTORY_OVERFLOW_FLUSHERS,
    _AAWM_SESSION_HISTORY_POOL_MAX_SIZE,
    _AAWM_SESSION_HISTORY_QUEUE_DRAIN_TO_SPOOL_MAX_RECORDS,
    _AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS,
    _AAWM_SESSION_HISTORY_RETRYABLE_EXCEPTION_NAMES,
    _AAWM_SESSION_HISTORY_RETRYABLE_MESSAGE_MARKERS,
    _AAWM_SESSION_HISTORY_SPOOL_DATETIME_MARKER,
    _AAWM_SESSION_HISTORY_SPOOL_DIR_DEFAULT,
    _AAWM_SESSION_HISTORY_SPOOL_DIR_ENV,
    _AAWM_SESSION_HISTORY_SPOOL_DRAIN_THREAD_NAME,
    _AAWM_SESSION_HISTORY_SPOOL_REPLAY_BACKOFF_SECONDS,
    _AAWM_SESSION_HISTORY_STATEMENT_CACHE_SIZE,
    _aawm_session_history_degraded_failure_fingerprint,
    _aawm_session_history_degraded_lock,
    _aawm_session_history_degraded_until_monotonic,
    _aawm_session_history_flush_failure_active,
    _aawm_session_history_flush_failure_lock,
    _aawm_session_history_overflow_flush_semaphore,
    _aawm_session_history_pool_lock,
    _aawm_session_history_pools,
    _aawm_session_history_queue,
    _aawm_session_history_schema_lock,
    _aawm_session_history_schema_ready,
    _aawm_session_history_spool_drain_lock,
    _aawm_session_history_spool_drainer,
    _aawm_session_history_spool_drainer_lock,
    _aawm_session_history_spool_startup_bootstrapped,
    _aawm_session_history_spool_startup_lock,
    _aawm_session_history_suppressed_flush_failures,
    _aawm_session_history_worker,
    _aawm_session_history_worker_lock,
)



def _get_session_history_degraded_spool_seconds() -> float:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_DEGRADED_SPOOL_SECONDS") or ""
    try:
        parsed_value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_DEGRADED_SPOOL_SECONDS
    return max(0.0, parsed_value)


def _get_session_history_spool_replay_backoff_seconds() -> Tuple[float, ...]:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_SPOOL_REPLAY_BACKOFF_SECONDS")
    if not raw_value:
        return _AAWM_SESSION_HISTORY_SPOOL_REPLAY_BACKOFF_SECONDS

    parsed_values: List[float] = []
    for raw_part in str(raw_value).split(","):
        stripped = raw_part.strip()
        if not stripped:
            continue
        try:
            parsed_values.append(max(0.0, float(stripped)))
        except (TypeError, ValueError):
            continue
    return tuple(parsed_values)


def _get_session_history_spool_dir() -> str:
    configured = _writer_get_secret_str(_AAWM_SESSION_HISTORY_SPOOL_DIR_ENV) or ""
    configured_path = str(configured).strip()
    if configured_path:
        return configured_path
    return _AAWM_SESSION_HISTORY_SPOOL_DIR_DEFAULT


def _ensure_session_history_spool_dir(spool_dir: str) -> None:
    try:
        os.makedirs(spool_dir, exist_ok=True)
    except FileExistsError:
        if os.path.isdir(spool_dir):
            return
        raise NotADirectoryError(
            f"session_history spool path exists but is not a directory: {spool_dir}"
        ) from None


def _mark_session_history_degraded_for_spooling(
    *,
    failure_fingerprint: Optional[str],
) -> None:

    degraded_seconds = _call("_get_session_history_degraded_spool_seconds", )
    if degraded_seconds <= 0:
        return
    with _state("_aawm_session_history_degraded_lock"):
        _set_state(
            "_aawm_session_history_degraded_until_monotonic",
            max(
                _state("_aawm_session_history_degraded_until_monotonic"),
                _writer_time().monotonic() + degraded_seconds,
            ),
        )
        _set_state(
            "_aawm_session_history_degraded_failure_fingerprint",
            failure_fingerprint,
        )


def _clear_session_history_degraded_spooling() -> None:

    with _state("_aawm_session_history_degraded_lock"):
        _set_state("_aawm_session_history_degraded_until_monotonic", 0.0)
        _set_state("_aawm_session_history_degraded_failure_fingerprint", None)


def _get_session_history_degraded_spooling_context() -> Optional[Dict[str, Any]]:
    with _state("_aawm_session_history_degraded_lock"):
        remaining_seconds = (
            _state("_aawm_session_history_degraded_until_monotonic") - _writer_time().monotonic()
        )
        if remaining_seconds <= 0:
            return None
        return {
            "remaining_seconds": round(remaining_seconds, 3),
            "failure_fingerprint": _state("_aawm_session_history_degraded_failure_fingerprint"),
        }


class _SessionHistorySpoolListing(NamedTuple):
    paths: Tuple[str, ...]
    availability: str


def _list_session_history_spool() -> _SessionHistorySpoolListing:
    spool_dir = _call("_get_session_history_spool_dir", )
    try:
        names = os.listdir(spool_dir)
    except FileNotFoundError:
        return _SessionHistorySpoolListing(paths=(), availability="missing")
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity: unable to list session_history spool directory "
            "%s: %s",
            spool_dir,
            _call("_format_exception_for_warning", exc),
        )
        return _SessionHistorySpoolListing(paths=(), availability="unavailable")
    return _SessionHistorySpoolListing(
        paths=tuple(
            sorted(
                os.path.join(spool_dir, name)
                for name in names
                if name.endswith(".jsonl") or name.endswith(".json")
            )
        ),
        availability="available",
    )


def _session_history_spool_paths() -> List[str]:
    return list(_call("_list_session_history_spool", ).paths)


def _session_history_spool_summary(
    paths: Optional[List[str]] = None,
    *,
    listing: Optional[_SessionHistorySpoolListing] = None,
) -> str:
    if listing is None:
        if paths is None:
            listing = _call("_list_session_history_spool", )
        else:
            listing = _SessionHistorySpoolListing(
                paths=tuple(paths),
                availability="available",
            )
    if listing.availability == "unavailable":
        return "spool_pending=unknown, spool_state=unavailable"
    if not listing.paths:
        return "spool_pending=0"
    paths_list = list(listing.paths)

    oldest_mtime: Optional[float] = None
    total_bytes = 0
    byte_count_known = True
    for path in paths_list:
        try:
            mtime = os.path.getmtime(path)
            total_bytes += os.path.getsize(path)
        except OSError:
            byte_count_known = False
            continue
        oldest_mtime = mtime if oldest_mtime is None else min(oldest_mtime, mtime)

    byte_summary = (
        f", spool_bytes={total_bytes}" if byte_count_known else ", spool_bytes=unknown"
    )
    if oldest_mtime is None:
        return (
            f"spool_pending={len(paths_list)}, oldest_pending_age_s=unknown"
            f"{byte_summary}"
        )
    oldest_age = max(0.0, time.time() - oldest_mtime)
    return (
        f"spool_pending={len(paths_list)}, oldest_pending_age_s={oldest_age:.1f}"
        f"{byte_summary}"
    )


def _encode_session_history_spool_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return {_AAWM_SESSION_HISTORY_SPOOL_DATETIME_MARKER: value.isoformat()}
    if isinstance(value, dict):
        return {
            str(key): _encode_session_history_spool_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_encode_session_history_spool_value(item) for item in value]
    return value


def _decode_session_history_spool_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_AAWM_SESSION_HISTORY_SPOOL_DATETIME_MARKER}:
            raw_datetime = value.get(_AAWM_SESSION_HISTORY_SPOOL_DATETIME_MARKER)
            if isinstance(raw_datetime, str):
                try:
                    return datetime.fromisoformat(raw_datetime.replace("Z", "+00:00"))
                except ValueError:
                    return raw_datetime
        return {
            key: _decode_session_history_spool_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_decode_session_history_spool_value(item) for item in value]
    return value


def _load_session_history_spool_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as spool_file:
        raw_payload = spool_file.read().strip()
    if not raw_payload:
        raise ValueError("session_history spool payload is empty")

    if path.endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        for line_number, raw_line in enumerate(raw_payload.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"session_history spool payload line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"session_history spool payload line {line_number} is not a JSON object"
                )
            line_type = payload.get("type")
            if line_type == "metadata":
                continue
            if line_type == "record":
                record = payload.get("record")
                if not isinstance(record, dict):
                    raise ValueError(
                        "session_history spool payload contains a non-object record"
                    )
                records.append(
                    cast(Dict[str, Any], _decode_session_history_spool_value(record))
                )
                continue
            raise ValueError(
                f"session_history spool payload line {line_number} has unsupported type"
            )
        if records:
            return records
        raise ValueError("session_history spool payload does not contain records")

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("session_history spool payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("session_history spool payload is not a JSON object")
    payload_records = payload.get("records")
    if isinstance(payload_records, list):
        decoded_records = _decode_session_history_spool_value(payload_records)
        if not all(isinstance(record, dict) for record in decoded_records):
            raise ValueError(
                "session_history spool payload contains a non-object record"
            )
        return cast(List[Dict[str, Any]], decoded_records)
    record = payload.get("record")
    if isinstance(record, dict):
        return [cast(Dict[str, Any], _decode_session_history_spool_value(record))]
    raise ValueError("session_history spool payload does not contain records")


def _load_session_history_spool_record(path: str) -> Dict[str, Any]:
    records = _call("_load_session_history_spool_records", path)
    if not records:
        raise ValueError("session_history spool payload does not contain records")
    return records[0]


def _session_history_spool_bad_record(path: str, exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        verbose_logger.warning(
            "AawmAgentIdentity: skipped missing session_history spool record "
            "during replay (path=%s): %s",
            path,
            _call("_format_exception_for_warning", exc),
        )
        return

    bad_path = f"{path}.bad"
    try:
        os.replace(path, bad_path)
    except OSError:
        pass
    verbose_logger.exception(
        "AawmAgentIdentity: moved unreadable session_history spool record to "
        "%s: %s",
        bad_path,
        _call("_format_exception_for_warning", exc),
    )


def _session_history_spool_drainer_main() -> None:
    if not _state("_aawm_session_history_spool_drain_lock").acquire(blocking=False):
        return
    try:
        batch_size = _call("_get_session_history_batch_size", )
        replay_backoff_seconds = _call("_get_session_history_spool_replay_backoff_seconds", )
        replay_retry_count = 0
        replay_started_at = _writer_time().perf_counter()
        while True:
            listing = _call("_list_session_history_spool", )
            if listing.availability == "unavailable":
                verbose_logger.warning(
                    "AawmAgentIdentity: session_history spool replay status is "
                    "unknown because the spool directory could not be listed (%s)",
                    _call("_session_history_spool_summary", listing=listing),
                )
                return
            paths = list(listing.paths)
            if not paths:
                return
            verbose_logger.warning(
                "AawmAgentIdentity: session_history spool replay started "
                "(spool_replay_started=true, pending_files=%d, batch_size=%d, %s)",
                len(paths),
                batch_size,
                _call("_session_history_spool_summary", paths),
            )

            batch_paths = paths[:batch_size]
            batch: List[Dict[str, Any]] = []
            kept_paths: List[str] = []
            for path in batch_paths:
                try:
                    batch.extend(_call("_load_session_history_spool_records", path))
                    kept_paths.append(path)
                except FileNotFoundError as exc:
                    verbose_logger.warning(
                        "AawmAgentIdentity: skipped missing session_history "
                        "spool record during replay (path=%s): %s",
                        path,
                        _call("_format_exception_for_warning", exc),
                    )
                except Exception as exc:
                    _session_history_spool_bad_record(path, exc)

            if not batch:
                continue

            if not _call("_flush_session_history_batch", batch):
                if replay_retry_count < len(replay_backoff_seconds):
                    wait_seconds = replay_backoff_seconds[replay_retry_count]
                    replay_retry_count += 1
                    verbose_logger.warning(
                        "AawmAgentIdentity: session_history spool replay failed; "
                        "retrying after backoff (spool_replay_retrying=true, "
                        "retry_count=%d, wait_seconds=%.1f, attempted_files=%d, "
                        "batch_size=%d, record_count=%d, %s)",
                        replay_retry_count,
                        wait_seconds,
                        len(batch_paths),
                        len(batch),
                        len(batch),
                        _call("_session_history_spool_summary", paths),
                    )
                    _writer_time().sleep(wait_seconds)
                    continue
                verbose_logger.warning(
                    "AawmAgentIdentity: session_history spool drain failed; "
                    "records remain spooled (spool_replay_failed=true, "
                    "retry_exhausted=true, retry_count=%d, attempted_files=%d, "
                    "batch_size=%d, record_count=%d, %s)",
                    replay_retry_count,
                    len(batch_paths),
                    len(batch),
                    len(batch),
                    _call("_session_history_spool_summary", paths),
                )
                return

            replay_retry_count = 0
            drained_record_count = len(batch)
            removed = 0
            for path in kept_paths:
                try:
                    os.remove(path)
                    removed += 1
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    verbose_logger.warning(
                        "AawmAgentIdentity: failed to remove drained "
                        "session_history spool record %s: %s",
                        path,
                        _call("_format_exception_for_warning", exc),
                    )
            verbose_logger.warning(
                "AawmAgentIdentity: recovered %d spooled session_history records "
                "from %d attempted files; removed %d files "
                "(spool_replay_recovered=true, record_count=%d, "
                "attempted_files=%d, removed_files=%d, duration_ms=%.2f, %s)",
                drained_record_count,
                len(kept_paths),
                removed,
                drained_record_count,
                len(kept_paths),
                removed,
                (_writer_time().perf_counter() - replay_started_at) * 1000.0,
                _call("_session_history_spool_summary", ),
            )
    finally:
        _state("_aawm_session_history_spool_drain_lock").release()


def _ensure_session_history_spool_drainer_started() -> None:

    if (
        _state("_aawm_session_history_spool_drainer") is not None
        and _state("_aawm_session_history_spool_drainer").is_alive()
    ):
        return

    listing = _call("_list_session_history_spool", )
    if listing.availability == "missing":
        return
    if listing.availability == "unavailable":
        verbose_logger.warning(
            "AawmAgentIdentity: session_history spool replay status is unknown "
            "because the spool directory could not be listed (%s); starting "
            "spool drainer to retry",
            _call("_session_history_spool_summary", listing=listing),
        )
    elif not listing.paths:
        return

    with _state("_aawm_session_history_spool_drainer_lock"):
        if (
            _state("_aawm_session_history_spool_drainer") is not None
            and _state("_aawm_session_history_spool_drainer").is_alive()
        ):
            return
        _set_state(
            "_aawm_session_history_spool_drainer",
            threading.Thread(
                target=_session_history_spool_drainer_main,
                name=_AAWM_SESSION_HISTORY_SPOOL_DRAIN_THREAD_NAME,
                daemon=True,
            ),
        )
        _state("_aawm_session_history_spool_drainer").start()


def _bootstrap_session_history_spool_drainer_once() -> None:

    if _state("_aawm_session_history_spool_startup_bootstrapped"):
        return

    with _state("_aawm_session_history_spool_startup_lock"):
        if _state("_aawm_session_history_spool_startup_bootstrapped"):
            return
        _set_state("_aawm_session_history_spool_startup_bootstrapped", True)
        try:
            _call("_ensure_session_history_spool_drainer_started", )
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity: failed to bootstrap session_history spool "
                "drainer: %s",
                _call("_format_exception_for_warning", exc),
            )


def _spool_session_history_record(record: Dict[str, Any]) -> None:
    _call("_spool_session_history_records", [record], reason="record")


def _session_history_spool_identity(records: List[Dict[str, Any]]) -> str:
    for field_name in ("trace_id", "session_id", "litellm_call_id"):
        for record in records:
            value = _identity_host()._clean_non_empty_string(record.get(field_name))
            if value:
                return value
    return "session-history"


def _sanitize_session_history_spool_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return sanitized[:96] or "session-history"


def _session_history_spool_filename(records: List[Dict[str, Any]]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    identity = _sanitize_session_history_spool_filename_component(
        _session_history_spool_identity(records)
    )
    digest_input = {
        "time_ns": time.time_ns(),
        "thread_id": threading.get_ident(),
        "identity": identity,
        "count": len(records),
    }
    digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{timestamp}-{identity}-{digest}.jsonl"


def _spool_session_history_records(
    records: List[Dict[str, Any]],
    *,
    reason: str,
    retry_count: Optional[int] = None,
    failure: Optional[Exception] = None,
    start_drainer: bool = True,
) -> str:
    if not records:
        raise ValueError("session_history spool requires at least one record")
    spool_dir = _call("_get_session_history_spool_dir", )
    _ensure_session_history_spool_dir(spool_dir)
    filename = _session_history_spool_filename(records)
    final_path = os.path.join(spool_dir, filename)
    tmp_path = f"{final_path}.{time.time_ns()}.{threading.get_ident()}.tmp"
    payload = {
        "spooled_at": datetime.now(timezone.utc),
        "reason": reason,
        "retry_count": retry_count,
        "record_count": len(records),
        "records": records,
    }
    if failure is not None:
        payload["failure"] = {
            "type": type(failure).__name__,
        }
    metadata_line = {
        "type": "metadata",
        "format_version": 1,
        "spooled_at": _encode_session_history_spool_value(payload["spooled_at"]),
        "reason": reason,
        "retry_count": retry_count,
        "record_count": len(records),
    }
    if failure is not None:
        metadata_line["failure"] = payload["failure"]
    with open(tmp_path, "w", encoding="utf-8") as spool_file:
        spool_file.write(
            json.dumps(metadata_line, separators=(",", ":"), sort_keys=True)
        )
        spool_file.write("\n")
        for index, record in enumerate(records):
            record_line = {
                "type": "record",
                "index": index,
                "record": _encode_session_history_spool_value(record),
            }
            spool_file.write(
                json.dumps(record_line, separators=(",", ":"), sort_keys=True)
            )
            spool_file.write("\n")
    os.replace(tmp_path, final_path)
    try:
        spool_bytes: Union[int, str] = os.path.getsize(final_path)
    except OSError:
        spool_bytes = "unknown"
    queue_size, queue_maxsize = _call("_session_history_queue_depth_values")
    verbose_logger.warning(
        "AawmAgentIdentity: protected %d session_history records by spooling "
        "for replay (spool_write_succeeded=true, path=%s, reason=%s, "
        "record_count=%d, spool_bytes=%s, queue_depth=%s/%s, %s)",
        len(records),
        final_path,
        reason,
        len(records),
        spool_bytes,
        queue_size,
        queue_maxsize,
        _call("_session_history_spool_summary", ),
    )
    if start_drainer:
        _call("_ensure_session_history_spool_drainer_started", )
    return final_path


def _start_session_history_spool_drainer_after_retry_exhaustion() -> None:
    try:
        _call("_ensure_session_history_spool_drainer_started", )
    except Exception as drainer_exc:
        verbose_logger.warning(
            "AawmAgentIdentity: failed to start session_history "
            "spool drainer after retry exhaustion: %s",
            _call("_format_exception_for_warning", drainer_exc),
        )


def _remove_recovered_session_history_retry_spool(
    retry_write_ahead_spool_path: Optional[str],
) -> bool:
    if retry_write_ahead_spool_path is None:
        return False
    try:
        os.remove(retry_write_ahead_spool_path)
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        verbose_logger.warning(
            "AawmAgentIdentity: failed to remove recovered "
            "session_history retry spool %s; replay may retry the "
            "idempotent batch: %s",
            retry_write_ahead_spool_path,
            _call("_format_exception_for_warning", exc),
        )
        return False


def _drain_session_history_queue_to_spool_on_shutdown(
    *,
    max_records=None,
    reason: str = "shutdown drain",
) -> int:
    """Best-effort drain of queued session_history records to the spool.

    Bounded by queue max size so shutdown cannot hang on unbounded work.
    """
    drain_limit = (
        max_records
        if max_records is not None
        else max(
            _AAWM_SESSION_HISTORY_QUEUE_DRAIN_TO_SPOOL_MAX_RECORDS,
            getattr(_state("_aawm_session_history_queue"), "maxsize", 0) or 1024,
        )
    )
    drained_records = _call("_drain_session_history_queue_for_spool", max(0, drain_limit))
    if not drained_records:
        return 0
    try:
        _call("_spool_session_history_records",
            drained_records,
            reason=reason,
            start_drainer=False,
        )
        verbose_logger.warning(
            "AawmAgentIdentity: spooled %d session_history records during "
            "shutdown drain (reason=%s, %s)",
            len(drained_records),
            reason,
            _call("_session_history_queue_depth_summary", ),
        )
    except Exception as exc:
        verbose_logger.exception(
            "AawmAgentIdentity: failed to spool session_history records during "
            "shutdown drain (record_count=%d, reason=%s): %s",
            len(drained_records),
            reason,
            _call("_format_exception_for_warning", exc),
        )
    return len(drained_records)


def _drain_session_history_queue_for_spool(max_records: int) -> List[Dict[str, Any]]:
    drained: List[Dict[str, Any]] = []
    get_nowait = getattr(_state("_aawm_session_history_queue"), "get_nowait", None)
    if not callable(get_nowait):
        return drained
    sentinel_seen = False
    while len(drained) < max_records:
        try:
            item = get_nowait()
        except queue.Empty:
            break
        if item is None:
            sentinel_seen = True
            break
        drained.append(item)

    if sentinel_seen:
        try:
            _state("_aawm_session_history_queue").put_nowait(None)
        except queue.Full:
            verbose_logger.warning(
                "AawmAgentIdentity: session_history shutdown sentinel could not "
                "be restored while draining queue for spool (%s)",
                _call("_session_history_queue_depth_summary", ),
            )
    return drained
