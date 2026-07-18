"""AAWM session_history retry module."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
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



def _session_history_retry_budget_remaining(
    retry_count: int, max_retries: Optional[int] = None
) -> Optional[int]:
    if max_retries is None:
        max_retries = _call("_get_session_history_failed_flush_max_retries", )
    return max(0, max_retries - retry_count)


def _session_history_persistence_telemetry_suffix(
    *,
    retry_count: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_write_ahead_spool_path: Optional[str] = None,
    spooled: Optional[bool] = None,
    degraded_telemetry: Optional[bool] = None,
    at_risk_of_loss: Optional[bool] = None,
) -> str:
    parts: List[str] = []
    if retry_count is not None:
        parts.append(f"retry_count={retry_count}")
        remaining = _session_history_retry_budget_remaining(
            retry_count, max_retries=max_retries
        )
        if remaining is not None:
            parts.append(f"retry_budget_remaining={remaining}")
    retry_wa_spooled = retry_write_ahead_spool_path is not None
    parts.append(f"retry_write_ahead_spooled={str(retry_wa_spooled).lower()}")
    parts.append(
        "retry_write_ahead_spool_path_present="
        f"{str(retry_wa_spooled).lower()}"
    )
    if spooled is not None:
        parts.append(f"spooled={str(spooled).lower()}")
    if degraded_telemetry is not None:
        parts.append(f"degraded_telemetry={str(degraded_telemetry).lower()}")
    if at_risk_of_loss is not None:
        parts.append(f"at_risk_of_loss={str(at_risk_of_loss).lower()}")
    return ", ".join(parts)

def _get_session_history_failed_flush_retry_seconds() -> float:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_FAILED_FLUSH_RETRY_SECONDS") or ""
    try:
        parsed_value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_FAILED_FLUSH_RETRY_SECONDS
    return max(0.1, parsed_value)


def _get_session_history_failed_flush_max_retries() -> int:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_FAILED_FLUSH_MAX_RETRIES") or ""
    try:
        parsed_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_FAILED_FLUSH_MAX_RETRIES
    return max(0, parsed_value)


def _mark_session_history_flush_failure_for_logging() -> bool:

    with _state("_aawm_session_history_flush_failure_lock"):
        if not _state("_aawm_session_history_flush_failure_active"):
            _set_state("_aawm_session_history_flush_failure_active", True)
            _set_state("_aawm_session_history_suppressed_flush_failures", 0)
            return True
        _set_state("_aawm_session_history_suppressed_flush_failures", _state("_aawm_session_history_suppressed_flush_failures") + (1))
        return False


def _reset_session_history_flush_failure_window() -> Optional[int]:

    with _state("_aawm_session_history_flush_failure_lock"):
        if not _state("_aawm_session_history_flush_failure_active"):
            return None
        suppressed_failures = _state("_aawm_session_history_suppressed_flush_failures")
        _set_state("_aawm_session_history_flush_failure_active", False)
        _set_state("_aawm_session_history_suppressed_flush_failures", 0)
        return suppressed_failures


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: Set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_retryable_session_history_persistence_failure(exc: Exception) -> bool:
    for chained_exc in _iter_exception_chain(exc):
        exc_name = type(chained_exc).__name__
        if exc_name in _AAWM_SESSION_HISTORY_RETRYABLE_EXCEPTION_NAMES:
            return True
        message = str(chained_exc).lower()
        if message and any(
            marker in message
            for marker in _AAWM_SESSION_HISTORY_RETRYABLE_MESSAGE_MARKERS
        ):
            return True
    return False


def _session_history_persistence_failure_fingerprint(exc: Exception) -> str:
    fingerprint_source = [
        {
            "type": type(chained_exc).__name__,
            "message": str(chained_exc),
        }
        for chained_exc in _iter_exception_chain(exc)
    ]
    return hashlib.sha256(
        json.dumps(
            fingerprint_source,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]


def _reset_session_history_pool_after_retryable_failure(
    loop: Optional[asyncio.AbstractEventLoop],
) -> int:
    if loop is None or loop.is_closed() or loop.is_running():
        return 0
    try:
        return int(
            loop.run_until_complete(_call("_drop_aawm_session_history_pools_for_current_loop", ))
        )
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity: failed to reset session_history DB pool before "
            "retry: %s",
            _call("_format_exception_for_warning", exc),
        )
        return 0


def _handle_session_history_retry_exhaustion(
    batch: List[Dict[str, Any]],
    *,
    retry_message: str,
    retry_count: int,
    last_failure: Optional[Exception],
    retry_write_ahead_spool_path: Optional[str],
) -> bool:
    if retry_write_ahead_spool_path is not None:
        verbose_logger.warning(
            "AawmAgentIdentity: %s failed after %d retries; retry "
            "write-ahead spool remains protected for replay "
            "(path=%s, batch_size=%d, %s, %s)",
            retry_message,
            retry_count,
            retry_write_ahead_spool_path,
            len(batch),
            _session_history_persistence_telemetry_suffix(
                retry_count=retry_count,
                retry_write_ahead_spool_path=retry_write_ahead_spool_path,
                spooled=True,
                degraded_telemetry=True,
                at_risk_of_loss=False,
            ),
            _call("_session_history_spool_summary", ),
        )
        _call("_start_session_history_spool_drainer_after_retry_exhaustion", )
        return True

    try:
        spool_path = _call("_spool_session_history_records",
            batch,
            reason=f"{retry_message} failed",
            retry_count=retry_count,
            failure=last_failure,
        )
        verbose_logger.warning(
            "AawmAgentIdentity: %s failed after %d retries; "
            "protected batch by spooling for replay (path=%s, batch_size=%d, %s, %s)",
            retry_message,
            retry_count,
            spool_path,
            len(batch),
            _session_history_persistence_telemetry_suffix(
                retry_count=retry_count,
                retry_write_ahead_spool_path=spool_path,
                spooled=True,
                degraded_telemetry=True,
                at_risk_of_loss=False,
            ),
            _call("_session_history_spool_summary", ),
        )
        return True
    except Exception as spool_exc:
        verbose_logger.exception(
            "AawmAgentIdentity: failed to spool %s after %d retries; "
            "potential session_history data loss until inline retry succeeds: %s "
            "(%s)",
            retry_message,
            retry_count,
            _call("_format_exception_for_warning", spool_exc),
            _session_history_persistence_telemetry_suffix(
                retry_count=retry_count,
                retry_write_ahead_spool_path=None,
                spooled=False,
                degraded_telemetry=True,
                at_risk_of_loss=True,
            ),
        )
        return False


def _prepare_session_history_retry_after_failure(
    batch: List[Dict[str, Any]],
    *,
    loop: Optional[asyncio.AbstractEventLoop],
    retry_message: str,
    retry_count: int,
    last_failure: Optional[Exception],
    retry_write_ahead_spool_path: Optional[str],
) -> Tuple[bool, Optional[str], int, Optional[str]]:
    if last_failure is None or not _call("_is_retryable_session_history_persistence_failure",
        last_failure
    ):
        return False, retry_write_ahead_spool_path, 0, None

    retryable_failure_fingerprint = _call("_session_history_persistence_failure_fingerprint",
        last_failure
    )
    _call("_mark_session_history_degraded_for_spooling",
        failure_fingerprint=retryable_failure_fingerprint,
    )
    if retry_write_ahead_spool_path is None:
        try:
            retry_write_ahead_spool_path = _call("_spool_session_history_records",
                batch,
                reason=f"{retry_message} retry write-ahead",
                retry_count=retry_count,
                failure=last_failure,
                start_drainer=False,
            )
        except Exception as spool_exc:
            verbose_logger.exception(
                "AawmAgentIdentity: failed to write retry-protection "
                "spool for %s; potential session_history data loss if "
                "the process exits before retry succeeds: %s (%s)",
                retry_message,
                _call("_format_exception_for_warning", spool_exc),
                _session_history_persistence_telemetry_suffix(
                    retry_count=retry_count,
                    retry_write_ahead_spool_path=None,
                    spooled=False,
                    degraded_telemetry=True,
                    at_risk_of_loss=True,
                ),
            )
    db_pool_reset_count = _call("_reset_session_history_pool_after_retryable_failure", loop)
    return (
        True,
        retry_write_ahead_spool_path,
        db_pool_reset_count,
        retryable_failure_fingerprint,
    )


def _log_session_history_retry(
    *,
    retry_message: str,
    retry_count: int,
    batch_size: int,
    retryable_last_failure: bool,
    db_pool_reset_this_retry: int,
    db_pool_reset_count: int,
    retry_write_ahead_spool_path: Optional[str],
    spooled: bool,
    degraded_telemetry: bool,
    failure_fingerprint: Optional[str],
) -> None:
    at_risk = retryable_last_failure and not spooled
    verbose_logger.warning(
        "AawmAgentIdentity: retrying %s within the configured retry budget "
        "(batch_size=%d, retryable=%s, db_pool_reset=%d, "
        "db_pool_reset_count=%d, failure_fingerprint=%s, %s, %s)",
        retry_message,
        batch_size,
        retryable_last_failure,
        db_pool_reset_this_retry,
        db_pool_reset_count,
        failure_fingerprint,
        _session_history_persistence_telemetry_suffix(
            retry_count=retry_count,
            retry_write_ahead_spool_path=retry_write_ahead_spool_path,
            spooled=spooled,
            degraded_telemetry=degraded_telemetry,
            at_risk_of_loss=at_risk,
        ),
        _call("_session_history_queue_depth_summary", ),
    )


def _log_recovered_retryable_session_history_flush(
    *,
    retry_count: int,
    batch_size: int,
    db_pool_reset_count: int,
    retry_write_ahead_spool_path: Optional[str],
    failure_fingerprint: Optional[str],
) -> None:
    _call("_clear_session_history_degraded_spooling", )
    removed_retry_spool = _call("_remove_recovered_session_history_retry_spool",
        retry_write_ahead_spool_path
    )
    verbose_logger.warning(
        "AawmAgentIdentity: session_history flush recovered after retry "
        "(flush_recovered=true, batch_size=%d, db_pool_reset_count=%d, "
        "retry_spool_removed=%s, failure_fingerprint=%s, %s, %s, %s)",
        batch_size,
        db_pool_reset_count,
        removed_retry_spool,
        failure_fingerprint,
        _session_history_persistence_telemetry_suffix(
            retry_count=retry_count,
            retry_write_ahead_spool_path=None,
            spooled=False,
            degraded_telemetry=False,
            at_risk_of_loss=False,
        ),
        _call("_session_history_queue_depth_summary", ),
        _call("_session_history_spool_summary", ),
    )
    if threading.current_thread().name != _AAWM_SESSION_HISTORY_SPOOL_DRAIN_THREAD_NAME:
        _call("_ensure_session_history_spool_drainer_started", )


def _flush_session_history_batch_with_retry(
    batch: List[Dict[str, Any]],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    retry_message: str = "session_history batch flush",
) -> None:
    retry_seconds = _call("_get_session_history_failed_flush_retry_seconds", )
    max_retries = _call("_get_session_history_failed_flush_max_retries", )
    retry_count = 0
    last_failure: Optional[Exception] = None
    retryable_failure_seen = False
    retryable_failure_fingerprint: Optional[str] = None
    retry_write_ahead_spool_path: Optional[str] = None
    db_pool_reset_count = 0

    def _capture_failure(exc: Exception) -> None:
        nonlocal last_failure
        last_failure = exc

    while not _call("_flush_session_history_batch",
        batch,
        loop=loop,
        log_exception=retry_count == 0,
        failure_callback=_capture_failure,
        ensure_spool_drainer=retry_write_ahead_spool_path is None,
        pending_retry_count=retry_count,
        retry_write_ahead_spool_path=retry_write_ahead_spool_path,
    ):
        if retry_count >= max_retries:
            if _call("_handle_session_history_retry_exhaustion",
                batch,
                retry_message=retry_message,
                retry_count=retry_count,
                last_failure=last_failure,
                retry_write_ahead_spool_path=retry_write_ahead_spool_path,
            ):
                return
            # Exhaustion handler could not spool; do not loop forever (B5).
            failure_text = (
                _call("_format_exception_for_warning", last_failure)
                if last_failure is not None
                else "unknown"
            )
            verbose_logger.error(
                "AawmAgentIdentity: dropping %d session_history records after "
                "retry exhaustion without durable spool "
                "(retry_message=%s, retry_count=%d, max_retries=%d, "
                "last_failure=%s, at_risk_of_loss=true)",
                len(batch),
                retry_message,
                retry_count,
                max_retries,
                failure_text,
            )
            return

        (
            retryable_last_failure,
            retry_write_ahead_spool_path,
            db_pool_reset_this_retry,
            current_failure_fingerprint,
        ) = _call("_prepare_session_history_retry_after_failure",
            batch,
            loop=loop,
            retry_message=retry_message,
            retry_count=retry_count,
            last_failure=last_failure,
            retry_write_ahead_spool_path=retry_write_ahead_spool_path,
        )
        if retryable_last_failure:
            retryable_failure_seen = True
            retryable_failure_fingerprint = current_failure_fingerprint
            db_pool_reset_count += db_pool_reset_this_retry

        retry_count += 1
        _call("_log_session_history_retry",
            retry_message=retry_message,
            retry_count=retry_count,
            batch_size=len(batch),
            retryable_last_failure=retryable_last_failure,
            db_pool_reset_this_retry=db_pool_reset_this_retry,
            db_pool_reset_count=db_pool_reset_count,
            retry_write_ahead_spool_path=retry_write_ahead_spool_path,
            spooled=retry_write_ahead_spool_path is not None,
            degraded_telemetry=retryable_failure_seen,
            failure_fingerprint=retryable_failure_fingerprint,
        )
        _writer_time().sleep(retry_seconds)

    if retryable_failure_seen:
        _call("_log_recovered_retryable_session_history_flush",
            retry_count=retry_count,
            batch_size=len(batch),
            db_pool_reset_count=db_pool_reset_count,
            retry_write_ahead_spool_path=retry_write_ahead_spool_path,
            failure_fingerprint=retryable_failure_fingerprint,
        )
