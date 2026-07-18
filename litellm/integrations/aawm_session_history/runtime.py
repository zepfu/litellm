"""AAWM session_history runtime: shared bridge, config constants, and mutable service state.

Owns process-local queue/pool/spool/retry control state used by writer, spool, and
retry modules. Other session_history modules import state and `_call`/`_state`
from here so tests can monkeypatch via `aawm_agent_identity` re-exports.

Concurrency model (RR-006 #22): the durable session_history service is
threading-only — ``queue.Queue`` plus ``threading.Lock`` / semaphore for worker,
pool cache, spool drainer, and degraded-mode flags. Asyncio is used solely for
asyncpg I/O inside the worker thread's dedicated event loop, not for protecting
shared module state. Identity extraction on request threads does not share a
second lock scheme for this service.
"""

from __future__ import annotations

import importlib
import queue
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from litellm.secret_managers.main import get_secret_str

def _get_persist_session_history_records() -> Callable[[List[Dict[str, Any]]], Any]:
    """Resolve persist entrypoint via identity host so tests can monkeypatch."""
    host = _identity_host()
    fn = getattr(host, "_persist_session_history_records", None)
    if fn is None:
        raise RuntimeError("session_history persist entrypoint is not installed")
    return fn


def _identity_host():
    from litellm.integrations import aawm_agent_identity as identity

    return identity


def _state(name: str) -> Any:
    """Prefer identity-host binding so tests can monkeypatch aawm_agent_identity."""
    host = _identity_host()
    if hasattr(host, name):
        return getattr(host, name)
    if name in globals():
        return globals()[name]
    for mod_name in (
        "litellm.integrations.aawm_session_history.writer",
        "litellm.integrations.aawm_session_history.spool",
        "litellm.integrations.aawm_session_history.retry",
    ):
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except Exception:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    raise KeyError(name)


def _set_state(name: str, value: Any) -> None:
    """Assign process-local service state in runtime and mirror to identity host."""
    globals()[name] = value
    try:
        host = _identity_host()
    except Exception:
        return
    setattr(host, name, value)


def _mirror_state(name: str) -> None:
    """Copy runtime global onto identity host after local assignment."""
    _set_state(name, globals()[name])


def _call(name: str, *args, **kwargs):
    host = _identity_host()
    fn = getattr(host, name, None)
    if fn is None:
        try:
            fn = _state(name)
        except KeyError:
            fn = None
    if fn is None:
        raise AttributeError(name)
    return fn(*args, **kwargs)


def _writer_time():
    host = _identity_host()
    return getattr(host, "time", time)


def _writer_importlib():
    host = _identity_host()
    return getattr(host, "importlib", importlib)


def _writer_get_secret_str(secret_name: str):
    host = _identity_host()
    fn = getattr(host, "get_secret_str", None)
    if callable(fn):
        return fn(secret_name)
    return get_secret_str(secret_name)


_AAWM_SESSION_HISTORY_BATCH_SIZE = 32
_AAWM_SESSION_HISTORY_FLUSH_INTERVAL_SECONDS = 0.25
_AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS = 0.1
_AAWM_SESSION_HISTORY_POOL_MAX_SIZE = 2
_AAWM_SESSION_HISTORY_COMMAND_TIMEOUT_SECONDS = 60.0
_AAWM_SESSION_HISTORY_STATEMENT_CACHE_SIZE = 0
_AAWM_SESSION_HISTORY_APPLICATION_NAME = "aawm-litellm-session-history"
_AAWM_SESSION_HISTORY_OVERFLOW_FLUSHERS = 1
_AAWM_SESSION_HISTORY_FAILED_FLUSH_RETRY_SECONDS = 1.0
_AAWM_SESSION_HISTORY_FAILED_FLUSH_MAX_RETRIES = 3
_AAWM_SESSION_HISTORY_DEGRADED_SPOOL_SECONDS = 30.0
_AAWM_SESSION_HISTORY_SPOOL_REPLAY_BACKOFF_SECONDS: Tuple[float, ...] = (
    5.0,
    15.0,
    30.0,
    60.0,
    120.0,
)
_AAWM_SESSION_HISTORY_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "ConnectionDoesNotExistError",
        "ConnectionFailureError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "TimeoutError",
    }
)
_AAWM_SESSION_HISTORY_RETRYABLE_MESSAGE_MARKERS = (
    "connection was closed",
    "connection is closed",
    "connection has been closed",
    "closed in the middle of operation",
    "server closed the connection",
    "connection reset",
    "connection lost",
    "connection terminated",
)


_AAWM_SESSION_HISTORY_SPOOL_DIR_ENV = "AAWM_SESSION_HISTORY_SPOOL_DIR"
_AAWM_SESSION_HISTORY_SPOOL_DIR_DEFAULT = "/mnt/e/litellm/session_history"
_AAWM_SESSION_HISTORY_SPOOL_DATETIME_MARKER = "__aawm_datetime__"
_AAWM_SESSION_HISTORY_SPOOL_DRAIN_THREAD_NAME = "aawm-session-history-spool-drainer"
_AAWM_SESSION_HISTORY_QUEUE_DRAIN_TO_SPOOL_MAX_RECORDS = 64
_aawm_session_history_schema_ready = False
_aawm_session_history_schema_lock = threading.Lock()
_aawm_session_history_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=1024)
_aawm_session_history_worker: Optional[threading.Thread] = None
_aawm_session_history_worker_lock = threading.Lock()
_aawm_session_history_pool_lock = threading.Lock()
_aawm_session_history_pools: Dict[Tuple[Any, str], Any] = {}
_aawm_session_history_overflow_flush_semaphore = threading.BoundedSemaphore(
    value=_AAWM_SESSION_HISTORY_OVERFLOW_FLUSHERS
)
_aawm_session_history_spool_drainer: Optional[threading.Thread] = None
_aawm_session_history_spool_drainer_lock = threading.Lock()
_aawm_session_history_spool_drain_lock = threading.Lock()
_aawm_session_history_spool_startup_lock = threading.Lock()
_aawm_session_history_spool_startup_bootstrapped = False
_aawm_session_history_flush_failure_lock = threading.Lock()
_aawm_session_history_flush_failure_active = False
_aawm_session_history_suppressed_flush_failures = 0
_aawm_session_history_degraded_lock = threading.Lock()
_aawm_session_history_degraded_until_monotonic = 0.0
_aawm_session_history_degraded_failure_fingerprint: Optional[str] = None
