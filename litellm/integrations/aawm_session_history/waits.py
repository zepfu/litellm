"""Request-local wait checkpoints, persisted by the existing history writer."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
)
from uuid import uuid4

from litellm._logging import verbose_logger

_current: ContextVar[Optional["RequestWaits"]] = ContextVar(
    "session_history_waits", default=None
)
_T = TypeVar("_T")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RequestWaits:
    def __init__(self, call_id: str, session_id: Optional[str] = None) -> None:
        self.call_id = call_id
        self.session_id = session_id or f"request:{call_id}"
        self.model = "unknown"
        self.provider: Optional[str] = None
        self.started_at = _now()
        self.sequence = 0
        self.status = "running"
        self.claimed = False
        self.active: dict[str, tuple[str, datetime, float]] = {}
        self.durations: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def checkpoint(self) -> None:
        # Snapshots are independent queue/spool records, never mutable aliases.
        self.sequence += 1
        snapshot = {
            "version": 1,
            "sequence": self.sequence,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "updated_at": _now().isoformat(),
            "current_waits": [
                {"type": kind, "started_at": started.isoformat()}
                for kind, started, _ in self.active.values()
            ],
            "durations_ms": {
                kind: round(value, 3) for kind, value in self.durations.items()
            },
            "counts": dict(self.counts),
        }
        try:
            from litellm.integrations.aawm_session_history.writer import (
                _enqueue_session_history_record,
            )

            _enqueue_session_history_record(
                {
                    "_wait_checkpoint": True,
                    "litellm_call_id": self.call_id,
                    "session_id": self.session_id,
                    "model": self.model,
                    "provider": self.provider,
                    "start_time": self.started_at,
                    "metadata": {"wait_accounting": snapshot},
                }
            )
        except Exception:
            # Observability cannot change request/cancellation behavior.
            verbose_logger.warning(
                "Session history wait checkpoint enqueue failed", exc_info=True
            )

    def enrich(self, identity: Mapping[str, Any]) -> None:
        changed = False
        for attr, keys in (
            ("session_id", ("canonical_session_id", "session_id", "codex_session_id")),
            ("model", ("model",)),
            ("provider", ("provider",)),
        ):
            value = next((identity.get(key) for key in keys if identity.get(key)), None)
            if (
                isinstance(value, str)
                and value != "unknown"
                and value != getattr(self, attr)
            ):
                setattr(self, attr, value)
                changed = True
        if changed:
            self.checkpoint()

    def begin(self, kind: str) -> str:
        key = str(uuid4())
        self.active[key] = (kind, _now(), time.monotonic())
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.checkpoint()
        return key

    def end(self, key: str) -> None:
        active = self.active.pop(key, None)
        if active is not None:
            kind, _, started = active
            self.durations[kind] = (
                self.durations.get(kind, 0.0)
                + max(0.0, time.monotonic() - started) * 1000
            )
            self.checkpoint()

    def finish(self, status: str) -> None:
        for key in list(self.active):
            self.end(key)
        self.status = status
        self.checkpoint()


def claim_request_call_id(default: str) -> str:
    """Seed only the first logging call; later attempts keep distinct cost rows."""
    tracker = _current.get()
    if tracker is None or tracker.claimed:
        return default
    tracker.claimed = True
    return tracker.call_id


def enrich_request_waits(identity: Mapping[str, Any]) -> None:
    tracker = _current.get()
    if tracker is not None and tracker.status == "running":
        tracker.enrich(identity)


@contextmanager
def wait_state(kind: str) -> Iterator[None]:
    tracker = _current.get()
    key = (
        tracker.begin(kind)
        if tracker is not None and tracker.status == "running"
        else None
    )
    try:
        yield
    finally:
        if tracker is not None and key is not None:
            tracker.end(key)


async def wait_for(kind: str, operation: Awaitable[_T]) -> _T:
    with wait_state(kind):
        return await operation


def track_wait(kind: str) -> Callable:
    def decorate(function: Callable) -> Callable:
        @wraps(function)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            with wait_state(kind):
                return await function(*args, **kwargs)

        return wrapped

    return decorate


async def wait_iterator(kind: str, iterator: AsyncIterator[_T]) -> AsyncIterator[_T]:
    # Close each read before yielding: consumer work/send time is separate.
    while True:
        try:
            item = await wait_for(kind, iterator.__anext__())
        except StopAsyncIteration:
            return
        yield item


async def persist_wait_checkpoints(conn: Any, records: list[dict[str, Any]]) -> None:
    """Coalesce cumulative snapshots and bypass terminal/side-table writes."""
    from litellm.integrations.aawm_session_history.sql import (
        _AAWM_SESSION_HISTORY_WAIT_CHECKPOINT_SQL,
    )

    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        call_id = record["litellm_call_id"]
        sequence = record["metadata"]["wait_accounting"]["sequence"]
        previous = latest.get(call_id)
        if (
            previous is None
            or sequence > previous["metadata"]["wait_accounting"]["sequence"]
        ):
            latest[call_id] = record
    if latest:
        await conn.executemany(
            _AAWM_SESSION_HISTORY_WAIT_CHECKPOINT_SQL,
            [
                (
                    record["litellm_call_id"],
                    record["session_id"],
                    record["model"],
                    record.get("provider"),
                    record["start_time"],
                    json.dumps(record["metadata"]),
                )
                for record in latest.values()
            ],
        )


class SessionHistoryWaitMiddleware:
    """Pure ASGI admission/terminal accounting; no body consumption or timers."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        path = scope.get("path", "").rstrip("/")
        inference = path.endswith(
            (
                "/responses",
                "/chat/completions",
                "/completions",
                "/messages",
                "/embeddings",
            )
        ) or path in {"/grok", "/v1/grok"}
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or not inference
        ):
            await self.app(scope, receive, send)
            return
        from litellm.integrations.aawm_session_history.writer import (
            _build_session_history_dsn,
        )

        if not _build_session_history_dsn():
            await self.app(scope, receive, send)
            return
        # Only identity headers are read. No payload or credential capture.
        from starlette.datastructures import Headers

        headers = Headers(scope=scope)
        session_id = next(
            (
                headers.get(key)
                for key in (
                    "session_id",
                    "session-id",
                    "x-session-id",
                    "x-aawm-session-id",
                    "x-litellm-session-id",
                    "x-codex-session-id",
                    "x-claude-code-session-id",
                    "x-grok-session-id",
                )
                if headers.get(key)
            ),
            None,
        )
        tracker = RequestWaits(
            headers.get("x-litellm-call-id") or str(uuid4()), session_id
        )
        token = _current.set(tracker)
        tracker.checkpoint()
        body_complete = False
        disconnected = False
        response_complete = False
        response_status = 0
        status = "failed"

        async def tracked_receive() -> Any:
            nonlocal body_complete, disconnected
            # Do not label the long-lived disconnect watcher as body receipt.
            if body_complete:
                message = await receive()
            else:
                message = await wait_for("request_body", receive())
            if message["type"] == "http.request" and not message.get(
                "more_body", False
            ):
                body_complete = True
            elif message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def tracked_send(message: Any) -> None:
            nonlocal response_status, response_complete
            await wait_for("downstream_send", send(message))
            if message["type"] == "http.response.start":
                response_status = message["status"]
            elif message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_complete = True

        try:
            await self.app(scope, tracked_receive, tracked_send)
            status = (
                "completed" if response_complete and response_status < 400 else "failed"
            )
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            tracker.finish(
                "disconnected" if disconnected and not response_complete else status
            )
            _current.reset(token)
