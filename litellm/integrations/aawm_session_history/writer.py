"""AAWM session_history durable queue/worker/pool service.

Owns enqueue, background worker, batch flush, and asyncpg pool lifecycle.
Filesystem spool lives in `spool.py`; retry/backoff in `retry.py`; shared state
and bridge helpers in `runtime.py`. This module re-exports those surfaces for
backward-compatible imports.
"""

from __future__ import annotations

import inspect

import asyncio
import atexit
import queue
import threading
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from litellm._logging import verbose_logger

from litellm.integrations.aawm_session_history.runtime import (  # noqa: F401
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
    _AAWM_SESSION_HISTORY_SHUTDOWN_BUDGET_SECONDS,
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
    _aawm_session_history_shutdown_records_abandoned,
    _aawm_session_history_worker_inflight_records,
    _aawm_session_history_spool_drain_lock,
    _aawm_session_history_spool_drainer,
    _aawm_session_history_spool_drainer_lock,
    _aawm_session_history_spool_startup_bootstrapped,
    _aawm_session_history_spool_startup_lock,
    _aawm_session_history_shutdown_deadline_monotonic,
    _aawm_session_history_shutdown_in_progress,
    _aawm_session_history_shutdown_lock,
    _aawm_session_history_shutdown_records_drained,
    _aawm_session_history_shutdown_records_spooled,
    _aawm_session_history_suppressed_flush_failures,
    _aawm_session_history_worker,
    _aawm_session_history_worker_lock,
    _call,
    _get_persist_session_history_records,
    _identity_host,
    _mirror_state,
    _set_state,
    _state,
    _writer_get_secret_str,
    _writer_importlib,
    _writer_time,
)

def _build_aawm_dsn_for_session_history() -> Optional[str]:
    host = _identity_host()
    return host._build_aawm_dsn()


def _append_aawm_dsn_query_params_for_session_history(
    dsn: str, params: Dict[str, str]
) -> str:
    host = _identity_host()
    return host._append_aawm_dsn_query_params(dsn, params)


def _get_session_history_batch_size() -> int:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_BATCH_SIZE") or ""
    try:
        parsed_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_BATCH_SIZE
    return max(1, parsed_value)



def _get_session_history_flush_interval_seconds() -> float:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_FLUSH_INTERVAL_MS") or ""
    try:
        parsed_value = float(str(raw_value).strip()) / 1000.0
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_FLUSH_INTERVAL_SECONDS
    return max(0.01, parsed_value)


def _get_session_history_pool_max_size() -> int:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_POOL_MAX_SIZE") or ""
    try:
        parsed_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_POOL_MAX_SIZE
    return max(1, parsed_value)


def _get_session_history_command_timeout_seconds() -> float:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_COMMAND_TIMEOUT_SECONDS") or ""
    try:
        parsed_value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_COMMAND_TIMEOUT_SECONDS
    return max(1.0, parsed_value)


def _get_session_history_statement_cache_size() -> int:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_STATEMENT_CACHE_SIZE") or ""
    try:
        parsed_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_STATEMENT_CACHE_SIZE
    return max(0, parsed_value)


def _get_session_history_shutdown_budget_seconds() -> float:
    raw_value = _writer_get_secret_str("AAWM_SESSION_HISTORY_SHUTDOWN_BUDGET_SECONDS") or ""
    try:
        parsed_value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_SHUTDOWN_BUDGET_SECONDS
    return max(0.0, parsed_value)


def _session_history_shutdown_deadline() -> float:
    with _state("_aawm_session_history_shutdown_lock"):
        if not _state("_aawm_session_history_shutdown_in_progress"):
            return 0.0
        return float(_state("_aawm_session_history_shutdown_deadline_monotonic"))


def _session_history_shutdown_disposition() -> Dict[str, int]:
    with _state("_aawm_session_history_shutdown_lock"):
        return {
            "drained": int(_state("_aawm_session_history_shutdown_records_drained")),
            "spooled": int(_state("_aawm_session_history_shutdown_records_spooled")),
            "abandoned": int(_state("_aawm_session_history_shutdown_records_abandoned")),
        }


def _session_history_mark_shutdown_disposition(
    *,
    drained: int = 0,
    spooled: int = 0,
    abandoned: int = 0,
) -> None:
    if drained == 0 and spooled == 0 and abandoned == 0:
        return
    with _state("_aawm_session_history_shutdown_lock"):
        if drained:
            _set_state(
                "_aawm_session_history_shutdown_records_drained",
                int(_state("_aawm_session_history_shutdown_records_drained")) + drained,
            )
        if spooled:
            _set_state(
                "_aawm_session_history_shutdown_records_spooled",
                int(_state("_aawm_session_history_shutdown_records_spooled")) + spooled,
            )
        if abandoned:
            _set_state(
                "_aawm_session_history_shutdown_records_abandoned",
                int(_state("_aawm_session_history_shutdown_records_abandoned")) + abandoned,
            )


def _session_history_track_worker_batch(records: List[Dict[str, Any]]) -> None:
    with _state("_aawm_session_history_shutdown_lock"):
        _set_state("_aawm_session_history_worker_inflight_records", records)


def _session_history_finish_worker_batch(
    records: List[Dict[str, Any]], *, persisted: bool = False
) -> bool:
    with _state("_aawm_session_history_shutdown_lock"):
        if _state("_aawm_session_history_worker_inflight_records") is not records:
            return False
        _set_state("_aawm_session_history_worker_inflight_records", [])
        if persisted and _state("_aawm_session_history_shutdown_in_progress"):
            _set_state(
                "_aawm_session_history_shutdown_records_drained",
                int(_state("_aawm_session_history_shutdown_records_drained"))
                + len(records),
            )
        return True


def _session_history_claim_worker_batch(
    records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    with _state("_aawm_session_history_shutdown_lock"):
        active_records = _state("_aawm_session_history_worker_inflight_records")
        if not active_records or (records is not None and active_records is not records):
            return []
        _set_state("_aawm_session_history_worker_inflight_records", [])
        return list(active_records)


def _session_history_worker_batch_was_claimed(records: List[Dict[str, Any]]) -> bool:
    with _state("_aawm_session_history_shutdown_lock"):
        return bool(
            _state("_aawm_session_history_shutdown_in_progress")
            and _state("_aawm_session_history_worker_inflight_records") is not records
        )


def _spool_session_history_records_after_shutdown(
    records: List[Dict[str, Any]], reason: str
) -> None:
    if not records:
        return
    record_count = len(records)
    try:
        _call(
            "_spool_session_history_records",
            records,
            reason=reason,
            start_drainer=False,
        )
        _session_history_mark_shutdown_disposition(spooled=record_count)
        verbose_logger.warning(
            "AawmAgentIdentity: spooled %d session_history records during "
            "shutdown (reason=%s, %s)",
            record_count,
            reason,
            _call("_session_history_queue_depth_summary"),
        )
    except Exception as exc:
        _session_history_mark_shutdown_disposition(abandoned=record_count)
        verbose_logger.exception(
            "AawmAgentIdentity: failed durable session_history disposition during "
            "shutdown (record_count=%d, reason=%s): %s",
            record_count,
            reason,
            _call("_format_exception_for_warning", exc),
        )


def _admit_session_history_record_before_shutdown(
    record: Dict[str, Any], *, timeout: float
) -> bool:
    with _state("_aawm_session_history_shutdown_lock"):
        if not _state("_aawm_session_history_shutdown_in_progress"):
            _state("_aawm_session_history_queue").put(record, timeout=timeout)
            _ensure_session_history_worker_started_locked()
            return True
    _spool_session_history_records_after_shutdown([record], "shutdown cutoff")
    return False


def _spool_session_history_record_if_shutdown_started(
    record: Dict[str, Any], reason: str
) -> bool:
    with _state("_aawm_session_history_shutdown_lock"):
        shutdown_started = bool(_state("_aawm_session_history_shutdown_in_progress"))
    if not shutdown_started:
        return False
    _spool_session_history_records_after_shutdown([record], reason)
    return True


def _session_history_queue_depth_summary() -> str:
    queue_size: Union[int, str]
    try:
        queue_size = _state("_aawm_session_history_queue").qsize()
    except Exception:
        queue_size = "unknown"
    max_size = getattr(_state("_aawm_session_history_queue"), "maxsize", "unknown")
    return f"queue_depth={queue_size}/{max_size}"


def _session_history_queue_depth_values() -> Tuple[Union[int, str], Union[int, str]]:
    queue_size: Union[int, str]
    try:
        queue_size = _state("_aawm_session_history_queue").qsize()
    except Exception:
        queue_size = "unknown"
    return queue_size, getattr(_state("_aawm_session_history_queue"), "maxsize", "unknown")


async def _close_aawm_session_history_pools_for_current_loop() -> None:
    loop = asyncio.get_running_loop()
    pools_to_close: List[Any] = []
    with _state("_aawm_session_history_pool_lock"):
        for key, pool in list(_state("_aawm_session_history_pools").items()):
            if key[0] is loop:
                pools_to_close.append(pool)
                del _state("_aawm_session_history_pools")[key]

    for pool in pools_to_close:
        await pool.close()


async def _drop_aawm_session_history_pools_for_current_loop() -> int:
    loop = asyncio.get_running_loop()
    pools_to_drop: List[Any] = []
    with _state("_aawm_session_history_pool_lock"):
        for key, pool in list(_state("_aawm_session_history_pools").items()):
            if key[0] is loop:
                pools_to_drop.append(pool)
                del _state("_aawm_session_history_pools")[key]

    for pool in pools_to_drop:
        try:
            terminate = getattr(pool, "terminate", None)
            if callable(terminate):
                result = terminate()
                if inspect.isawaitable(result):
                    await result
                continue
            close = getattr(pool, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity: failed to drop cached session_history "
                "pool during retry reset: %s",
                _call("_format_exception_for_warning", exc),
            )
    return len(pools_to_drop)


def _flush_session_history_batch(
    records: List[Dict[str, Any]],
    loop: Optional[asyncio.AbstractEventLoop] = None,
    *,
    timeout_seconds: Optional[float] = None,
    log_exception: bool = True,
    failure_callback: Optional[Callable[[Exception], None]] = None,
    ensure_spool_drainer: bool = True,
    pending_retry_count: Optional[int] = None,
    retry_write_ahead_spool_path: Optional[str] = None,
) -> bool:
    if not records:
        return True
    if (
        threading.current_thread().name == "aawm-session-history-writer"
        and _session_history_worker_batch_was_claimed(records)
    ):
        return True

    if loop is not None and loop.is_running():
        try:
            spool_paths = _call("_spool_session_history_records",
                records,
                reason="session_history batch flush deferred from running event loop",
            )
            verbose_logger.warning(
                "AawmAgentIdentity: deferred session_history flush from a "
                "running event loop by spooling for replay "
                "(spooled=true, degraded_telemetry=true, batch_size=%d, "
                "spool_path_count=%d, %s, %s)",
                len(records),
                len(spool_paths),
                _call("_session_history_queue_depth_summary", ),
                _call("_session_history_spool_summary", ),
            )
            return True
        except Exception as exc:
            if failure_callback is not None:
                failure_callback(exc)
            if log_exception:
                verbose_logger.exception(
                    "AawmAgentIdentity: failed to defer session_history flush "
                    "from a running event loop by spooling for replay "
                    "(spooled=false, degraded_telemetry=false, batch_size=%d, %s)",
                    len(records),
                    _call("_session_history_queue_depth_summary", ),
                )
            return False

    started_at = _writer_time().perf_counter()

    async def _persist_records() -> None:
        persist_coro = _get_persist_session_history_records()(records)
        if timeout_seconds is None:
            await persist_coro
            return
        await asyncio.wait_for(
            persist_coro,
            timeout=max(0.0, timeout_seconds),
        )

    try:
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_persist_records())
            finally:
                loop.run_until_complete(
                    _call("_close_aawm_session_history_pools_for_current_loop", )
                )
                loop.close()
                asyncio.set_event_loop(None)
        else:
            loop.run_until_complete(_persist_records())
    except Exception as exc:
        if failure_callback is not None:
            failure_callback(exc)
        retryable_failure = _call("_is_retryable_session_history_persistence_failure", exc)
        if retryable_failure:
            verbose_logger.warning(
                "AawmAgentIdentity: retryable session_history persistence "
                "degradation; retrying without active error intake: %s "
                "(batch_size=%d, failure_fingerprint=%s, %s, %s, %s)",
                _call("_format_exception_for_warning", exc),
                len(records),
                _call("_session_history_persistence_failure_fingerprint", exc),
                _session_history_persistence_telemetry_suffix(
                    retry_count=pending_retry_count,
                    retry_write_ahead_spool_path=retry_write_ahead_spool_path,
                    spooled=retry_write_ahead_spool_path is not None,
                    degraded_telemetry=True,
                    at_risk_of_loss=False,
                ),
                _call("_session_history_queue_depth_summary", ),
                _call("_session_history_spool_summary", ),
            )
        elif log_exception and _call("_mark_session_history_flush_failure_for_logging", ):
            verbose_logger.exception(
                "AawmAgentIdentity: failed to flush %d session_history records; "
                "retrying within the configured retry budget: %s (%s)",
                len(records),
                _call("_format_exception_for_warning", exc),
                _call("_session_history_queue_depth_summary", ),
            )
        else:
            verbose_logger.warning(
                "AawmAgentIdentity: session_history flush still failing within "
                "the configured retry budget: %s (batch_size=%d, %s, %s)",
                _call("_format_exception_for_warning", exc),
                len(records),
                _session_history_persistence_telemetry_suffix(
                    retry_count=pending_retry_count,
                    retry_write_ahead_spool_path=retry_write_ahead_spool_path,
                    spooled=retry_write_ahead_spool_path is not None,
                    degraded_telemetry=False,
                    at_risk_of_loss=False,
                ),
                _call("_session_history_queue_depth_summary", ),
            )
        return False

    _session_history_finish_worker_batch(records, persisted=True)
    suppressed_failures = _call("_reset_session_history_flush_failure_window", )
    if suppressed_failures is not None:
        verbose_logger.warning(
            "AawmAgentIdentity: session_history flush recovered "
            "(suppressed_full_tracebacks=%d, %s)",
            suppressed_failures,
            _call("_session_history_queue_depth_summary", ),
        )
    verbose_logger.debug(
        "AawmAgentIdentity: flushed %d session_history records in %.2fms (%s)",
        len(records),
        (_writer_time().perf_counter() - started_at) * 1000.0,
        _call("_session_history_queue_depth_summary", ),
    )
    if (
        ensure_spool_drainer
        and threading.current_thread().name != _AAWM_SESSION_HISTORY_SPOOL_DRAIN_THREAD_NAME
    ):
        _call("_ensure_session_history_spool_drainer_started", )
    return True


def _flush_session_history_overflow_record(record: Dict[str, Any]) -> None:
    try:
        _call("_flush_session_history_batch_with_retry",
            [record],
            retry_message="overflow session_history flush",
        )
    finally:
        _state("_aawm_session_history_overflow_flush_semaphore").release()


def _flush_session_history_worker_batch(
    records: List[Dict[str, Any]], loop: asyncio.AbstractEventLoop
) -> None:
    _session_history_track_worker_batch(records)
    try:
        shutdown_deadline = _session_history_shutdown_deadline()
        if shutdown_deadline > 0.0:
            remaining_shutdown_budget = (
                shutdown_deadline - _writer_time().monotonic()
            )
            if remaining_shutdown_budget <= 0.0:
                claimed_records = _session_history_claim_worker_batch(records)
                _spool_session_history_records_after_shutdown(
                    claimed_records,
                    "shutdown DB deadline expired",
                )
                return
            if not _call(
                "_flush_session_history_batch",
                records,
                loop=loop,
                timeout_seconds=remaining_shutdown_budget,
            ):
                claimed_records = _session_history_claim_worker_batch(records)
                _spool_session_history_records_after_shutdown(
                    claimed_records,
                    "shutdown DB flush failed",
                )
                return
            return
        _call("_flush_session_history_batch_with_retry", records, loop=loop)
    finally:
        _session_history_finish_worker_batch(records)


def _session_history_worker_main() -> None:
    flush_interval = _call("_get_session_history_flush_interval_seconds", )
    batch_size = _call("_get_session_history_batch_size", )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            shutdown_deadline = _session_history_shutdown_deadline()
            if shutdown_deadline > 0.0 and _writer_time().monotonic() >= shutdown_deadline:
                break

            effective_timeout = flush_interval
            if shutdown_deadline > 0.0:
                remaining = shutdown_deadline - _writer_time().monotonic()
                if remaining <= 0:
                    break
                effective_timeout = min(effective_timeout, remaining)
            try:
                first_item = _state("_aawm_session_history_queue").get(timeout=effective_timeout)
            except queue.Empty:
                continue

            if first_item is None:
                break

            batch: List[Dict[str, Any]] = [first_item]
            deadline = _writer_time().monotonic() + flush_interval
            if shutdown_deadline > 0.0:
                deadline = min(deadline, shutdown_deadline)
            stop_after_batch = False
            while len(batch) < batch_size:
                remaining = deadline - _writer_time().monotonic()
                if remaining <= 0:
                    break
                try:
                    next_item = _state("_aawm_session_history_queue").get(timeout=remaining)
                except queue.Empty:
                    break
                if next_item is None:
                    stop_after_batch = True
                    break
                batch.append(next_item)

            _flush_session_history_worker_batch(batch, loop)
            if stop_after_batch:
                break
    finally:
        loop.run_until_complete(_call("_close_aawm_session_history_pools_for_current_loop", ))
        loop.close()



def _ensure_session_history_worker_started() -> None:
    with _state("_aawm_session_history_shutdown_lock"):
        if _state("_aawm_session_history_shutdown_in_progress"):
            return

        _ensure_session_history_worker_started_locked()


def _ensure_session_history_worker_started_locked() -> None:
    if (
        _state("_aawm_session_history_worker") is not None
        and _state("_aawm_session_history_worker").is_alive()
    ):
        return

    with _state("_aawm_session_history_worker_lock"):
        if (
            _state("_aawm_session_history_worker") is not None
            and _state("_aawm_session_history_worker").is_alive()
        ):
            return

        _set_state(
            "_aawm_session_history_worker",
            threading.Thread(
                target=_session_history_worker_main,
                name="aawm-session-history-writer",
                daemon=True,
            ),
        )
        _state("_aawm_session_history_worker").start()



def _shutdown_session_history_worker() -> None:
    shutdown_budget_seconds = _call("_get_session_history_shutdown_budget_seconds", )
    shutdown_deadline = _writer_time().monotonic() + max(0.0, shutdown_budget_seconds)

    with _state("_aawm_session_history_shutdown_lock"):
        if _state("_aawm_session_history_shutdown_in_progress"):
            return
        _set_state("_aawm_session_history_shutdown_records_drained", 0)
        _set_state("_aawm_session_history_shutdown_records_spooled", 0)
        _set_state("_aawm_session_history_shutdown_records_abandoned", 0)
        _set_state("_aawm_session_history_shutdown_in_progress", True)
        _set_state("_aawm_session_history_shutdown_deadline_monotonic", shutdown_deadline)
        worker = _state("_aawm_session_history_worker")
    if worker is None:
        _call(
            "_drain_session_history_queue_to_spool_on_shutdown",
            reason="shutdown with no worker",
        )
        _close_session_history_worker_resources()
        return

    sentinel_budget = max(0.0, shutdown_deadline - _writer_time().monotonic())
    if sentinel_budget > 0.0:
        try:
            _state("_aawm_session_history_queue").put(
                None,
                timeout=sentinel_budget,
            )
        except queue.Full:
            verbose_logger.warning(
                "AawmAgentIdentity: session_history queue full during shutdown; "
                "remaining queued records will be spooled after worker join"
            )

    join_budget = max(0.0, shutdown_deadline - _writer_time().monotonic())
    if join_budget > 0.0:
        worker.join(timeout=join_budget)

    if not worker.is_alive():
        active_records = _session_history_claim_worker_batch()
        if active_records:
            _spool_session_history_records_after_shutdown(
                active_records,
                "shutdown unresolved stopped worker batch",
            )
    _call(
        "_drain_session_history_queue_to_spool_on_shutdown",
        reason="shutdown post-join drain",
    )
    _close_session_history_worker_resources()


def _close_session_history_worker_resources() -> None:
    with _state("_aawm_session_history_shutdown_lock"):
        worker = _state("_aawm_session_history_worker")
        if worker is not None and not worker.is_alive():
            with _state("_aawm_session_history_worker_lock"):
                if _state("_aawm_session_history_worker") is worker:
                    _set_state("_aawm_session_history_worker", None)
    disposition = _session_history_shutdown_disposition()
    verbose_logger.warning(
        "AawmAgentIdentity: session_history shutdown disposition complete "
        "(shutdown_records_drained=%d, shutdown_records_spooled=%d, "
        "shutdown_records_abandoned=%d)",
        disposition["drained"],
        disposition["spooled"],
        disposition["abandoned"],
    )


def _format_exception_for_warning(exc: Exception) -> str:
    message = str(exc)
    if message:
        return message
    return f"{type(exc).__name__}: {exc!r}"


def _enqueue_session_history_record(record: Dict[str, Any]) -> None:
    _call("_ensure_session_history_worker_started", )
    degraded_context = _call("_get_session_history_degraded_spooling_context", )
    if degraded_context is not None:
        drained_records = _call("_drain_session_history_queue_for_spool",
            max(
                0,
                min(
                    _call("_get_session_history_batch_size", ) - 1,
                    _AAWM_SESSION_HISTORY_QUEUE_DRAIN_TO_SPOOL_MAX_RECORDS,
                ),
            )
        )
        spool_batch = [*drained_records, record]
        verbose_logger.warning(
            "AawmAgentIdentity: session_history DB degraded; spooling enqueue "
            "batch for replay (queue_disposition=db_degraded_spooling, "
            "drained_queue_records=%d, record_count=%d, degraded_remaining_s=%s, "
            "failure_fingerprint=%s, %s)",
            len(drained_records),
            len(spool_batch),
            degraded_context.get("remaining_seconds"),
            degraded_context.get("failure_fingerprint"),
            _call("_session_history_queue_depth_summary", ),
        )
        try:
            _call("_spool_session_history_records",
                spool_batch,
                reason="db degraded enqueue",
                start_drainer=False,
            )
            return
        except Exception as exc:
            verbose_logger.exception(
                "AawmAgentIdentity: failed to spool session_history record "
                "during DB degraded mode; falling back to inline protection "
                "(queue_disposition=spool_write_failed, "
                "unprotected_record_risk=true, record_count=%d): %s",
                len(spool_batch),
                _call("_format_exception_for_warning", exc),
            )
            _call("_flush_session_history_batch_with_retry",
                spool_batch,
                retry_message="inline session_history degraded spool failure",
            )
            return

    try:
        if not _admit_session_history_record_before_shutdown(
            record,
            timeout=_AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS,
        ):
            return
    except queue.Full:
        if not _state("_aawm_session_history_overflow_flush_semaphore").acquire(blocking=False):
            try:
                if not _admit_session_history_record_before_shutdown(
                    record,
                    timeout=max(
                        _AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS,
                        _call("_get_session_history_flush_interval_seconds", ),
                    ),
                ):
                    return
                verbose_logger.warning(
                    "AawmAgentIdentity: session_history queue full; queued "
                    "record after waiting (queue_disposition=queued_after_wait, %s)",
                    _call("_session_history_queue_depth_summary", ),
                )
                return
            except queue.Full:
                pass

            drain_limit = max(
                0,
                min(
                    _call("_get_session_history_batch_size", ) - 1,
                    _AAWM_SESSION_HISTORY_QUEUE_DRAIN_TO_SPOOL_MAX_RECORDS,
                ),
            )
            drained_records = _call("_drain_session_history_queue_for_spool", drain_limit)
            spool_batch = [*drained_records, record]
            verbose_logger.warning(
                "AawmAgentIdentity: session_history queue full and overflow flusher "
                "busy; spooling overflow batch for retry "
                "(queue_disposition=spool_write_started, drained_queue_records=%d, "
                "record_count=%d, %s)",
                len(drained_records),
                len(spool_batch),
                _call("_session_history_queue_depth_summary", ),
            )
            try:
                _call("_spool_session_history_records",
                    spool_batch,
                    reason="queue full overflow",
                )
            except Exception as exc:
                verbose_logger.exception(
                    "AawmAgentIdentity: failed to spool session_history overflow "
                    "batch; potential data loss until inline retry succeeds "
                    "(queue_disposition=spool_write_failed, "
                    "unprotected_record_risk=true, record_count=%d): %s",
                    len(spool_batch),
                    _call("_format_exception_for_warning", exc),
                )
                _call("_flush_session_history_batch_with_retry",
                    spool_batch,
                    retry_message="inline session_history overflow after spool failure",
                )
            return

        if _spool_session_history_record_if_shutdown_started(
            record,
            "shutdown cutoff before overflow flush",
        ):
            _state("_aawm_session_history_overflow_flush_semaphore").release()
            return

        verbose_logger.warning(
            "AawmAgentIdentity: session_history queue full; flushing overflow "
            "record in background (queue_disposition=overflow_flush_started, %s)",
            _call("_session_history_queue_depth_summary", ),
        )
        try:
            threading.Thread(
                target=_flush_session_history_overflow_record,
                args=(record,),
                name="aawm-session-history-overflow",
                daemon=True,
            ).start()
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity: failed to start session_history overflow flusher; flushing inline: %s",
                _call("_format_exception_for_warning", exc),
            )
            try:
                _call("_flush_session_history_batch", [record])
            finally:
                _state("_aawm_session_history_overflow_flush_semaphore").release()


atexit.register(_shutdown_session_history_worker)


def _get_session_history_application_name() -> str:
    host = _identity_host()
    get_first = getattr(host, "_get_first_secret_value", None)
    env_vars = getattr(host, "_AAWM_DB_APPLICATION_NAME_ENV_VARS", ())
    if callable(get_first):
        value = get_first(env_vars)
        if value:
            return value
    return _AAWM_SESSION_HISTORY_APPLICATION_NAME



def _get_session_history_server_settings() -> Dict[str, str]:
    return {"application_name": _call("_get_session_history_application_name", )}



async def _initialize_session_history_connection(conn: Any) -> None:
    await conn.execute(
        "select set_config($1, $2, false)",
        "application_name",
        _call("_get_session_history_application_name", ),
    )



def _build_session_history_dsn() -> Optional[str]:
    dsn = _build_aawm_dsn_for_session_history()
    if not dsn:
        return None
    return _append_aawm_dsn_query_params_for_session_history(
        dsn,
        {"application_name": _call("_get_session_history_application_name", )},
    )



async def _open_aawm_session_history_connection() -> Any:
    dsn = _call("_build_session_history_dsn", )
    if not dsn:
        raise RuntimeError("AAWM session history database configuration is missing")

    try:
        asyncpg = _writer_importlib().import_module("asyncpg")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AAWM session history requires asyncpg to be installed"
        ) from exc

    # application_name is already set via server_settings startup packet; skip
    # the redundant set_config round trip used only by older pooled init paths.
    conn = await asyncpg.connect(
        dsn=dsn,
        command_timeout=_call("_get_session_history_command_timeout_seconds", ),
        statement_cache_size=_call("_get_session_history_statement_cache_size", ),
        server_settings=_call("_get_session_history_server_settings", ),
    )
    return conn



async def _get_aawm_session_history_pool() -> Any:
    dsn = _call("_build_session_history_dsn", )
    if not dsn:
        raise RuntimeError("AAWM session history database configuration is missing")

    loop = asyncio.get_running_loop()
    pool_key = (loop, dsn)
    with _state("_aawm_session_history_pool_lock"):
        pool = _state("_aawm_session_history_pools").get(pool_key)
    if pool is not None:
        return pool

    try:
        asyncpg = _writer_importlib().import_module("asyncpg")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AAWM session history requires asyncpg to be installed"
        ) from exc

    created_pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=0,
        max_size=_call("_get_session_history_pool_max_size", ),
        command_timeout=_call("_get_session_history_command_timeout_seconds", ),
        statement_cache_size=_call("_get_session_history_statement_cache_size", ),
        # application_name is set via server_settings startup packet only; avoid
        # redundant per-connection set_config init= and manual re-inits on write.
        server_settings=_call("_get_session_history_server_settings", ),
    )
    with _state("_aawm_session_history_pool_lock"):
        existing_pool = _state("_aawm_session_history_pools").get(pool_key)
        if existing_pool is None:
            _state("_aawm_session_history_pools")[pool_key] = created_pool
            return created_pool

    await created_pool.close()
    return existing_pool



async def _ensure_session_history_schema(conn: Any) -> None:
    """Mark schema readiness without issuing DDL (RR-006 #40).

    ALTER/CREATE statements live in package ``sql.py`` for operators/migrations.
    The hot-path writer only sets a process-local ready flag so we never pay
    100+ sequential bootstrap round-trips per worker process.
    """
    if _state("_aawm_session_history_schema_ready"):
        return

    with _state("_aawm_session_history_schema_lock"):
        if _state("_aawm_session_history_schema_ready"):
            return

        # Schema changes are migration-owned. The callback must not mutate,
        # recreate, or drop database structures at request/write time.
        _ = conn
        _set_state("_aawm_session_history_schema_ready", True)





# Compatibility re-exports: historical imports from writer still resolve.
from litellm.integrations.aawm_session_history.spool import (  # noqa: E402,F401
    _SessionHistorySpoolListing,
    _bootstrap_session_history_spool_drainer_once,
    _clear_session_history_degraded_spooling,
    _decode_session_history_spool_value,
    _drain_session_history_queue_for_spool,
    _drain_session_history_queue_to_spool_on_shutdown,
    _encode_session_history_spool_value,
    _ensure_session_history_spool_dir,
    _ensure_session_history_spool_drainer_started,
    _get_session_history_degraded_spool_seconds,
    _get_session_history_degraded_spooling_context,
    _get_session_history_spool_dir,
    _get_session_history_spool_replay_backoff_seconds,
    _list_session_history_spool,
    _load_session_history_spool_record,
    _load_session_history_spool_records,
    _mark_session_history_degraded_for_spooling,
    _remove_recovered_session_history_retry_spool,
    _sanitize_session_history_spool_filename_component,
    _session_history_spool_bad_record,
    _session_history_spool_drainer_main,
    _session_history_spool_filename,
    _session_history_spool_identity,
    _session_history_spool_paths,
    _session_history_spool_summary,
    _spool_session_history_record,
    _spool_session_history_records,
    _start_session_history_spool_drainer_after_retry_exhaustion,
)
from litellm.integrations.aawm_session_history.retry import (  # noqa: E402,F401
    _flush_session_history_batch_with_retry,
    _get_session_history_failed_flush_max_retries,
    _get_session_history_failed_flush_retry_seconds,
    _handle_session_history_retry_exhaustion,
    _is_retryable_session_history_persistence_failure,
    _iter_exception_chain,
    _log_recovered_retryable_session_history_flush,
    _log_session_history_retry,
    _mark_session_history_flush_failure_for_logging,
    _prepare_session_history_retry_after_failure,
    _reset_session_history_flush_failure_window,
    _reset_session_history_pool_after_retryable_failure,
    _session_history_persistence_failure_fingerprint,
    _session_history_persistence_telemetry_suffix,
    _session_history_retry_budget_remaining,
)
