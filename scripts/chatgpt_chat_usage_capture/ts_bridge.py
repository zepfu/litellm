"""Parent-side bridge for the bounded TypeScript ChatGPT usage worker.

The Python side owns PostgreSQL fences, browser preparation, process
supervision, and the bounded NDJSON transport. TypeScript owns scheduling,
history collection, reconstruction, and report shaping.
"""

from __future__ import annotations

import errno
import json
import math
import os
import selectors
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Mapping, Optional, Sequence
from uuid import uuid4

from .pg_collector_state import CollectorLease, PageAck, PgCollectorState
from .pg_ledger import LedgerError, LedgerScope, scope_key
from .privacy import PrivacyError, assert_no_secrets, sanitize_token


PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_REQUESTS = 256
MAX_REQUEST_ID_LENGTH = 128
MAX_CURSOR_BYTES = 512
MAX_JSON_DEPTH = 32
DEFAULT_CHILD_TTL_SECONDS = 3600
TERMINAL_CLEANUP_SECONDS = 5
TERMINAL_REAP_SECONDS = 1
DEFAULT_WORKER_ROOT = "/app/scripts/chatgpt_chat_usage_capture/ts"
DEFAULT_MODEL_MAPPING = {
    "version": "initial-unmapped",
    "canonicalFamilies": ["astra_pro", "sol_pro", "other_chat", "unknown"],
    "rules": [],
    "reviewStatus": "draft",
    "source": "collector-empty-seed",
    "createdAt": "2026-09-07T00:00:00.000Z",
    "changeKind": "prospective",
    "validFrom": None,
    "validUntil": None,
    "publishedAt": None,
    "supersedesVersion": None,
    "correctionOfVersion": None,
    "provenance": {},
    "warnings": [],
}
SUPPORTED_OPERATIONS = frozenset(
    {
        "loadState",
        "compareAndSetState",
        "readHistory",
        "loadConversationMetadata",
        "commitPage",
        "loadReportSnapshot",
        "finishRun",
        "cancel",
    }
)
SUPPORTED_CONTROLS = frozenset(
    {"prepareHistory", "inspectSessionIdentity", "cancelRun"}
)
BLOCKED_HISTORY_REASONS = frozenset({"authentication", "cooldown"})
_HISTORY_STATUS_RANK = {
    "ready": 0,
    "partial": 1,
    "unavailable": 2,
    "blocked": 3,
}
_IDENTITY_AUTH_STATES = frozenset({"ready"})
_UNAVAILABLE_CAPABILITY_STATES = frozenset(
    {
        "disabled",
        "not_observed",
        "unknown",
        "unavailable",
        "unsupported",
    }
)
_NATIVE_HISTORY_KINDS = frozenset(
    {"chat_history", "history", "ordinary_chat"}
)
_HISTORY_OPERATION_ALIASES = {
    "index": frozenset({"index", "list", "listConversations"}),
    "modern_detail": frozenset(
        {"detail", "modern_detail", "modernDetail", "conversationDetail"}
    ),
    "messages": frozenset({"messages", "message_pages", "messagePages"}),
}


@dataclass(frozen=True)
class BridgeConfig:
    node_executable: str = os.environ.get(
        "AAWM_CHATGPT_ORACLE_NODE_EXECUTABLE", "node"
    )
    worker_root: str = os.environ.get(
        "AAWM_CHATGPT_USAGE_BRIDGE_WORKER_ROOT",
        DEFAULT_WORKER_ROOT,
    )
    max_frame_bytes: int = MAX_FRAME_BYTES
    max_total_bytes: int = MAX_TOTAL_BYTES
    max_requests: int = MAX_REQUESTS
    child_ttl_seconds: int = DEFAULT_CHILD_TTL_SECONDS
    schedule_options: Mapping[str, Any] = field(
        default_factory=lambda: {"interval": "PT1H", "jitterSeconds": 60}
    )
    collection_request: Mapping[str, Any] = field(
        default_factory=lambda: {"mode": "incremental"}
    )
    mapping: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_MAPPING)
    )
    history_preparer: Optional[HistoryPreparer] = field(
        default=None,
        repr=False,
        compare=False,
    )
    binding_bootstrapper: Optional[BindingBootstrapper] = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_runtime(
        cls,
        *,
        node_executable: str,
        worker_root: str = DEFAULT_WORKER_ROOT,
        mapping: Optional[Mapping[str, Any]] = None,
        history_preparer: Optional[HistoryPreparer] = None,
        binding_bootstrapper: Optional[BindingBootstrapper] = None,
    ) -> "BridgeConfig":
        return cls(
            node_executable=node_executable,
            worker_root=worker_root,
            mapping=(
                dict(DEFAULT_MODEL_MAPPING)
                if mapping is None
                else _validate_model_mapping(mapping)
            ),
            history_preparer=history_preparer,
            binding_bootstrapper=binding_bootstrapper,
        )

    @property
    def worker_entrypoint(self) -> str:
        return str(Path(self.worker_root) / "dist" / "src" / "worker" / "main.js")

    def __post_init__(self) -> None:
        if not self.node_executable.strip():
            raise ValueError("node executable must not be empty")
        if not self.worker_root.strip():
            raise ValueError("worker root must not be empty")
        for value, name, maximum in (
            (self.max_frame_bytes, "max_frame_bytes", MAX_FRAME_BYTES),
            (self.max_total_bytes, "max_total_bytes", MAX_TOTAL_BYTES),
            (self.max_requests, "max_requests", MAX_REQUESTS),
            (self.child_ttl_seconds, "child_ttl_seconds", DEFAULT_CHILD_TTL_SECONDS),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value > maximum:
                raise ValueError(f"{name} cannot exceed the supported ceiling")
        if self.max_frame_bytes > self.max_total_bytes:
            raise ValueError("max_frame_bytes cannot exceed max_total_bytes")
        _validate_model_mapping(self.mapping)


class BridgeProtocolError(LedgerError):
    """A protocol or parent/child contract violation."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "protocol_invalid",
        retryable: bool = False,
        coverage_incomplete: bool = True,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.coverage_incomplete = coverage_incomplete


@dataclass(frozen=True)
class BridgeOperationContext:
    """Cancellation/deadline state passed to owned blocking operations."""

    deadline_monotonic: float
    deadline_at: datetime
    cancel_event: threading.Event
    run_id: Optional[str] = None
    collector_account_id: Optional[str] = None
    profile_id: Optional[str] = None
    terminal_deadline_monotonic: Optional[float] = None
    terminal_deadline_at: Optional[datetime] = None
    ignore_cancel: bool = False

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def remaining_ms(self) -> int:
        return _remaining_ms(self.deadline_monotonic)

    def check(self) -> None:
        if self.cancelled and not self.ignore_cancel:
            raise BridgeProtocolError(
                "worker operation was cancelled",
                code="cancelled",
                retryable=True,
            )
        if time.monotonic() >= self.deadline_monotonic:
            raise BridgeProtocolError(
                "worker deadline exceeded",
                code="bounds_exceeded",
                retryable=True,
            )

    def terminal(self) -> "BridgeOperationContext":
        deadline_monotonic = self.terminal_deadline_monotonic
        deadline_at = self.terminal_deadline_at
        if deadline_monotonic is None or deadline_at is None:
            deadline_monotonic = self.deadline_monotonic
            deadline_at = self.deadline_at
        cutoff = min(
            deadline_monotonic,
            time.monotonic() + TERMINAL_CLEANUP_SECONDS,
        )
        deadline_at -= timedelta(seconds=deadline_monotonic - cutoff)
        deadline_monotonic = cutoff
        return BridgeOperationContext(
            deadline_monotonic=deadline_monotonic - TERMINAL_REAP_SECONDS,
            deadline_at=deadline_at - timedelta(seconds=TERMINAL_REAP_SECONDS),
            cancel_event=self.cancel_event,
            run_id=self.run_id,
            collector_account_id=self.collector_account_id,
            profile_id=self.profile_id,
            terminal_deadline_monotonic=deadline_monotonic,
            terminal_deadline_at=deadline_at,
            ignore_cancel=True,
        )


HistoryPreparer = Callable[
    [LedgerScope, str, str, BridgeOperationContext],
    Mapping[str, Any],
]
BindingBootstrapper = Callable[
    [LedgerScope, datetime, BridgeOperationContext],
    LedgerScope,
]


@dataclass
class _Child:
    process: subprocess.Popen
    process_group_id: int
    total_bytes: int
    request_count: int
    deadline_monotonic: float
    terminal_deadline_monotonic: float
    read_buffer: bytearray = field(default_factory=bytearray)
    ready_frames: Deque[bytes] = field(default_factory=deque)
    write_failed: bool = False


@dataclass
class _RunContext:
    collector_account_id: str
    profile_id: str
    run_id: str
    scope: LedgerScope
    lease: CollectorLease
    child: _Child
    state_version: int
    operation: BridgeOperationContext
    trigger_id: Optional[str] = None
    prepared_history: Optional[Mapping[str, Any]] = None
    verified_history: Optional[Mapping[str, Any]] = None
    capabilities: Optional[Mapping[str, Any]] = None
    capability_manifest: Optional[Mapping[str, Any]] = None
    preparation_count: int = 0
    history_status: str = "unavailable"
    history_reason: Optional[str] = None
    retry_after_ms: Optional[int] = None
    finish_outcome: Optional[str] = None
    snapshot_ids: set[str] = field(default_factory=set)
    finished: bool = False
    cancelled: bool = False
    cleanup_errors: list[str] = field(default_factory=list)
    terminal_operation: Optional[BridgeOperationContext] = None

    def terminal(self) -> BridgeOperationContext:
        if self.terminal_operation is None:
            self.terminal_operation = self.operation.terminal()
            cutoff = self.terminal_operation.terminal_deadline_monotonic
            if cutoff is not None:
                self.child.terminal_deadline_monotonic = min(
                    self.child.terminal_deadline_monotonic,
                    cutoff,
                )
        return self.terminal_operation


class TsWorkerBridge:
    """Run one fenced TypeScript worker for one account/profile scope."""

    def __init__(self, state: PgCollectorState, config: BridgeConfig) -> None:
        self.state = state
        self.config = config
        self._children: dict[tuple[str, str], _Child] = {}
        self._lock = threading.Lock()
        self._admitting = True
        self._active_operation: Optional[BridgeOperationContext] = None

    def _call_with_deadline(
        self,
        deadline_monotonic: float,
        callback: Callable[..., Any],
        *args: Any,
        operation: Optional[BridgeOperationContext] = None,
        on_result: Optional[Callable[[Any], None]] = None,
        check_cancel: bool = True,
        forward_operation: bool = True,
        **kwargs: Any,
    ) -> Any:
        self._check_operation(
            operation,
            deadline_monotonic=deadline_monotonic,
            check_cancel=check_cancel,
        )
        try:
            if forward_operation:
                kwargs["operation"] = operation
            result = callback(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            self._check_operation(
                operation,
                deadline_monotonic=deadline_monotonic,
                check_cancel=check_cancel,
            )
            raise
        # Retain an acquired lease/resource before checking whether the
        # callback crossed the deadline. The caller then owns cleanup.
        if on_result is not None:
            on_result(result)
        self._check_operation(
            operation,
            deadline_monotonic=deadline_monotonic,
            check_cancel=check_cancel,
        )
        return result

    def _call_context(
        self,
        context: _RunContext,
        callback: Callable[..., Any],
        *args: Any,
        operation: Optional[BridgeOperationContext] = None,
        on_result: Optional[Callable[[Any], None]] = None,
        check_cancel: bool = True,
        forward_operation: bool = True,
        **kwargs: Any,
    ) -> Any:
        active_operation = operation or context.operation
        return self._call_with_deadline(
            active_operation.deadline_monotonic,
            callback,
            *args,
            operation=active_operation,
            on_result=on_result,
            check_cancel=check_cancel,
            forward_operation=forward_operation,
            **kwargs,
        )

    @staticmethod
    def _adopt_loaded_state_version(
        context: _RunContext,
        value: Any,
    ) -> None:
        if not isinstance(value, tuple) or not value:
            return
        header = value[0]
        context.state_version = (
            int(header.state_version) if header is not None else 0
        )

    @staticmethod
    def _check_operation(
        operation: Optional[BridgeOperationContext],
        *,
        deadline_monotonic: float,
        check_cancel: bool,
    ) -> None:
        if operation is not None and check_cancel:
            operation.check()
        else:
            TsWorkerBridge._check_deadline_monotonic(deadline_monotonic)

    @staticmethod
    def _check_deadline_monotonic(deadline_monotonic: float) -> None:
        if time.monotonic() >= deadline_monotonic:
            raise BridgeProtocolError(
                "worker deadline exceeded",
                code="bounds_exceeded",
                retryable=True,
            )

    def run_once(  # noqa: PLR0915 - owns one bounded run lifecycle
        self,
        *,
        collector_account_id: str,
        profile_id: str,
        scope: LedgerScope,
        seen_at: Optional[datetime] = None,
        schedule_options: Optional[Mapping[str, Any]] = None,
        collection_request: Optional[Mapping[str, Any]] = None,
        mapping: Optional[Mapping[str, Any]] = None,
        authentication_recovery_requested: bool = False,
    ) -> dict[str, Any]:
        if not self._admitting:
            raise BridgeProtocolError(
                "worker bridge is shutting down",
                code="cancelled",
                retryable=True,
            )
        account = _metadata_token(collector_account_id, "collector_account_id")
        profile = _metadata_token(profile_id, "profile_id")
        requested_scope = _scope_for_account(scope, account)
        if authentication_recovery_requested:
            # Recovery is an explicit action for an already configured scope;
            # it may not bootstrap or rebind an incomplete identity.
            _verified_scope_if_complete(requested_scope)
        run_id = str(uuid4())
        deadline_monotonic = time.monotonic() + self.config.child_ttl_seconds
        deadline_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.config.child_ttl_seconds
        )
        terminal_deadline_monotonic = (
            deadline_monotonic + TERMINAL_CLEANUP_SECONDS
        )
        terminal_deadline_at = deadline_at + timedelta(
            seconds=TERMINAL_CLEANUP_SECONDS
        )
        cancel_event = threading.Event()
        operation = BridgeOperationContext(
            deadline_monotonic=deadline_monotonic,
            deadline_at=deadline_at,
            cancel_event=cancel_event,
            run_id=run_id,
            collector_account_id=account,
            profile_id=profile,
            terminal_deadline_monotonic=terminal_deadline_monotonic,
            terminal_deadline_at=terminal_deadline_at,
        )
        self._active_operation = operation
        if not self._admitting:
            cancel_event.set()
        cleanup_operation: Optional[BridgeOperationContext] = None
        lease: Optional[CollectorLease] = None
        lease_holder: list[CollectorLease] = []
        child: Optional[_Child] = None
        context: Optional[_RunContext] = None
        observed_at = seen_at or datetime.now(timezone.utc)
        key = (account, profile)
        result: dict[str, Any] = {
            "ok": False,
            "status": "failed",
            "collectorAccountId": account,
            "profileId": profile,
            "runId": run_id,
            "errorCode": "history_reader_failed",
            "retryable": True,
            "coverageIncomplete": True,
        }
        control_error: Optional[BaseException] = None
        control_traceback = None

        try:
            operation.check()
            parent_scope, binding = self._capture_or_initialize_binding(
                requested_scope,
                seen_at=observed_at,
                operation=operation,
            )
            try:
                lease = self._call_with_deadline(
                    deadline_monotonic,
                    self.state.claim_lease,
                    operation=operation,
                    on_result=lease_holder.append,
                    collector_account_id=account,
                    profile_id=profile,
                    scope=parent_scope,
                    ttl_seconds=self.config.child_ttl_seconds,
                    expected_binding=binding,
                )
            except BaseException:
                if lease is None and lease_holder:
                    lease = lease_holder[-1]
                raise
            child = self._start_child(key, deadline_monotonic)
            context = _RunContext(
                collector_account_id=account,
                profile_id=profile,
                run_id=run_id,
                scope=parent_scope,
                lease=lease,
                child=child,
                state_version=0,
                operation=operation,
            )
            operation.check()
            header, _ = self._call_context(
                context,
                self.state.load_state,
                collector_account_id=account,
                profile_id=profile,
                operation=context.operation,
                on_result=lambda value: self._adopt_loaded_state_version(
                    context,
                    value,
                ),
            )
            context.state_version = header.state_version if header is not None else 0
            self._send_start_run(
                context,
                schedule_options=schedule_options or self.config.schedule_options,
                collection_request=collection_request
                or self.config.collection_request,
                mapping=(
                    self.config.mapping
                    if mapping is None
                    else mapping
                ),
                authentication_recovery_requested=authentication_recovery_requested,
            )
            result = self._serve_child(context)
            if not context.finished and not context.cancelled:
                raise BridgeProtocolError(
                    "worker ended without a committed finish",
                    code="history_reader_failed",
                    retryable=True,
                )
        except BaseException as exc:
            cleanup_operation = (
                context.terminal() if context is not None else operation.terminal()
            )
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                control_error = exc
                control_traceback = exc.__traceback__
            else:
                result = _failure_result(
                    account=account,
                    profile=profile,
                    run_id=run_id,
                    error=exc,
                    context=context,
                )
            cancel_event.set()
            if context is not None:
                context.cancelled = True
                cancel_signal_error = self._send_parent_cancel(
                    context,
                    reason=_failure_code(exc),
                    operation=cleanup_operation,
                )
                if cancel_signal_error is not None:
                    context.cleanup_errors.append(cancel_signal_error)
                cancel_error = self._cancel_state(
                    context,
                    exc,
                    operation=cleanup_operation,
                )
                if cancel_error is not None:
                    context.cleanup_errors.append(cancel_error)
            elif lease is not None:
                lease_error = self._release_lease_quietly(
                    lease,
                    operation=cleanup_operation,
                )
                if lease_error is not None:
                    result = _with_cleanup_failure(result, [lease_error])
        finally:
            if cleanup_operation is None:
                cleanup_operation = (
                    context.terminal()
                    if context is not None
                    else operation.terminal()
                )
            if context is not None:
                context.cleanup_errors.extend(
                    self._close_tracked_snapshots(
                        context,
                        operation=cleanup_operation,
                    )
                )
                cleanup_error = self._close_prepared_history(
                    context,
                    operation=cleanup_operation,
                )
                if cleanup_error is not None:
                    context.cleanup_errors.append(cleanup_error)
            if child is not None:
                reap_error = self._reap(
                    key=key,
                    child=child,
                    deadline_monotonic=(
                        cleanup_operation.terminal_deadline_monotonic
                    ),
                )
                if reap_error is not None:
                    if context is not None:
                        context.cleanup_errors.append(reap_error)
                    else:
                        result = _with_cleanup_failure(result, [reap_error])
            if context is not None and context.cleanup_errors:
                result = _with_cleanup_failure(result, context.cleanup_errors)
            self._active_operation = None
        if control_error is not None:
            raise control_error.with_traceback(control_traceback)
        return result

    def _capture_or_initialize_binding(
        self,
        scope: LedgerScope,
        *,
        seen_at: datetime,
        operation: BridgeOperationContext,
    ) -> tuple[LedgerScope, Any]:
        operation.check()
        parent_scope: Optional[LedgerScope] = None
        try:
            parent_scope = _verified_scope_if_complete(scope)
        except BridgeProtocolError:
            pass
        if parent_scope is not None:
            try:
                binding = self._call_with_deadline(
                    operation.deadline_monotonic,
                    self.state.ledger.capture_binding,
                    parent_scope,
                    seen_at=seen_at,
                    operation=operation,
                )
                return parent_scope, binding
            except LedgerError as exc:
                if "not initialized" not in str(exc).lower():
                    raise
        bootstrapper = self.config.binding_bootstrapper
        if bootstrapper is None:
            raise BridgeProtocolError(
                "collector scope binding is not initialized",
                code="history_contract_unavailable",
                retryable=True,
            )
        bootstrapped_scope = self._call_with_deadline(
            operation.deadline_monotonic,
            lambda candidate_scope, observed_at, *, operation: bootstrapper(
                candidate_scope,
                observed_at,
                operation,
            ),
            scope,
            seen_at,
            operation=operation,
        )
        if not isinstance(bootstrapped_scope, LedgerScope):
            raise BridgeProtocolError(
                "binding bootstrap did not return a verified scope",
                code="history_contract_unavailable",
                retryable=True,
            )
        parent_scope = _verified_scope(bootstrapped_scope, scope.collector_account_id)
        self._call_with_deadline(
            operation.deadline_monotonic,
            self.state.ledger.initialize_binding,
            parent_scope,
            seen_at=seen_at,
            operation=operation,
        )
        return parent_scope, self._call_with_deadline(
            operation.deadline_monotonic,
            self.state.ledger.capture_binding,
            parent_scope,
            seen_at=seen_at,
            operation=operation,
        )

    def _start_child(self, key: tuple[str, str], deadline_monotonic: float) -> _Child:
        self._check_deadline_monotonic(deadline_monotonic)
        with self._lock:
            previous = self._children.get(key)
        if previous is not None:
            reap_error = self._reap(key=key, child=previous)
            if reap_error is not None:
                raise BridgeProtocolError(
                    reap_error,
                    code="cleanup_failed",
                    retryable=True,
                )
        self._check_deadline_monotonic(deadline_monotonic)
        process = subprocess.Popen(
            [
                self.config.node_executable,
                self.config.worker_entrypoint,
                "--stdio-v1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
        )
        try:
            process_group_id = os.getpgid(process.pid)
        except OSError:
            process_group_id = process.pid
        child = _Child(
            process,
            process_group_id,
            0,
            0,
            deadline_monotonic,
            deadline_monotonic + TERMINAL_CLEANUP_SECONDS,
        )
        with self._lock:
            self._children[key] = child
        if process.stdin is None or process.stdout is None:
            return child
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        return child

    def _send_start_run(
        self,
        context: _RunContext,
        *,
        schedule_options: Mapping[str, Any],
        collection_request: Mapping[str, Any],
        mapping: Mapping[str, Any],
        authentication_recovery_requested: bool,
    ) -> None:
        mapping = _validate_model_mapping(mapping)
        if context.child.request_count >= self.config.max_requests:
            raise BridgeProtocolError(
                "worker request bound exceeded",
                code="bounds_exceeded",
                retryable=True,
            )
        envelope = {
            "runId": context.run_id,
            "collectorAccountId": context.collector_account_id,
            "profileId": context.profile_id,
            "bindingGeneration": context.lease.binding.binding_generation,
            "leaseFencingToken": context.lease.lease_fencing_token,
        }
        request = {
            "protocolVersion": PROTOCOL_VERSION,
            "control": "startRun",
            "envelope": envelope,
            "scheduleOptions": schedule_options,
            "collectionRequest": collection_request,
            "mapping": mapping,
            "bounds": {
                "maxFrameBytes": self.config.max_frame_bytes,
                "maxTotalBytes": self.config.max_total_bytes,
                "maxRequests": self.config.max_requests,
                "remainingMs": _remaining_ms(context.child.deadline_monotonic),
            },
        }
        if authentication_recovery_requested:
            request["authenticationRecoveryRequested"] = True
        _assert_bounded_safe(request, max_bytes=self.config.max_frame_bytes)
        request["scheduleOptions"] = dict(schedule_options)
        request["collectionRequest"] = dict(collection_request)
        context.child.request_count += 1
        self._write_frame(context.child, request)

    def _serve_child(self, context: _RunContext) -> dict[str, Any]:
        while not context.finished and not context.cancelled:
            message = self._read_message(context)
            if "control" in message:
                response = self._handle_control(context, message)
                if response is not None:
                    self._write_frame(context.child, response)
                continue
            response = self._handle_operation(context, message)
            terminal_operation = context.terminal_operation
            self._write_frame(
                context.child,
                response,
                enforce_deadline=terminal_operation is None,
                deadline_monotonic=(
                    terminal_operation.deadline_monotonic
                    if terminal_operation is not None
                    else None
                ),
            )
            if (
                message.get("operation") == "cancel"
                and response.get("ok") is not True
            ):
                error = response.get("error")
                code = (
                    error.get("code")
                    if isinstance(error, Mapping)
                    else "history_reader_failed"
                )
                raise BridgeProtocolError(
                    "durable worker cancellation failed",
                    code=(
                        str(code)
                        if isinstance(code, str)
                        else "history_reader_failed"
                    ),
                    retryable=True,
                )
            if (
                message.get("operation") == "finishRun"
                and response.get("ok") is True
            ):
                self._await_child_exit(context.child)
        succeeded = (
            context.finished
            and not context.cancelled
            and context.finish_outcome == "success"
            and context.history_status == "ready"
        )
        return {
            "ok": succeeded,
            "status": (
                "completed"
                if succeeded
                else "failed"
                if context.finished
                else "cancelled"
            ),
            "collectorAccountId": context.collector_account_id,
            "profileId": context.profile_id,
            "runId": context.run_id,
            "coverageIncomplete": (
                context.history_status != "ready"
                or context.finish_outcome != "success"
            ),
            "historyContract": context.history_status,
            "historyReason": context.history_reason,
            "retryAfterMs": context.retry_after_ms,
            "outcome": context.finish_outcome,
            "stateVersion": context.state_version,
            "requests": context.child.request_count,
        }

    def _handle_control(
        self,
        context: _RunContext,
        message: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        control = message.get("control")
        request_id = _safe_request_id(message.get("requestId"))
        if control not in SUPPORTED_CONTROLS:
            return _error_response(
                request_id,
                code="operation_unsupported",
                retryable=False,
                coverage_incomplete=True,
            )
        try:
            self._validate_message_fence(context, message)
        except Exception as exc:
            return _error_response(
                request_id,
                code=_failure_code(exc),
                retryable=_retryable(exc),
                coverage_incomplete=True,
            )
        if control == "prepareHistory":
            try:
                payload = _mapping_payload(message.get("payload"))
                result = self._prepare_history(context, payload)
                return _ok_response(request_id, result)
            except Exception as exc:
                return _error_response(
                    request_id,
                    code=_failure_code(exc),
                    retryable=_retryable(exc),
                    coverage_incomplete=True,
                )
        if control == "inspectSessionIdentity":
            if context.verified_history is None:
                return _error_response(
                    request_id,
                    code="history_contract_unavailable",
                    retryable=True,
                    coverage_incomplete=True,
                )
            return _ok_response(
                request_id,
                _wire_history_identity(context.verified_history),
            )
        return _error_response(
            request_id,
            code="operation_unsupported",
            retryable=False,
            coverage_incomplete=True,
        )

    def _handle_operation(
        self,
        context: _RunContext,
        message: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request_id = _safe_request_id(message.get("requestId"))
        if not request_id:
            return _error_response(
                "",
                code="protocol_invalid",
                retryable=False,
                coverage_incomplete=True,
            )
        try:
            self._validate_worker_request(context, message)
            operation = message.get("operation")
            result = self._dispatch_operation(
                context,
                str(operation),
                message.get("payload"),
            )
            return _ok_response(request_id, result)
        except Exception as exc:
            return _error_response(
                request_id,
                code=_failure_code(exc),
                retryable=_retryable(exc),
                coverage_incomplete=True,
            )

    def _consume_worker_request(self, context: _RunContext) -> None:
        context.child.request_count += 1
        if context.child.request_count > self.config.max_requests:
            raise BridgeProtocolError(
                "worker request bound exceeded",
                code="bounds_exceeded",
                retryable=True,
            )
        self._check_deadline(context.child)

    def _validate_message_fence(
        self,
        context: _RunContext,
        message: Mapping[str, Any],
    ) -> None:
        if message.get("protocolVersion") != PROTOCOL_VERSION:
            raise BridgeProtocolError("worker protocol version mismatch")
        envelope = message.get("envelope")
        if isinstance(envelope, Mapping):
            run_id = envelope.get("runId")
            account = envelope.get("collectorAccountId")
            profile = envelope.get("profileId")
            generation = envelope.get("bindingGeneration")
            fencing = envelope.get("leaseFencingToken")
        else:
            run_id = message.get("runId")
            account = message.get("collectorAccountId")
            profile = message.get("profileId")
            generation = message.get("bindingGeneration")
            fencing = message.get("leaseFencingToken")
        if (
            run_id != context.run_id
            or account != context.collector_account_id
            or profile != context.profile_id
            or generation != context.lease.binding.binding_generation
            or fencing != context.lease.lease_fencing_token
        ):
            raise BridgeProtocolError(
                "worker fence does not match the parent lease",
                code="fence_invalid",
            )

    def _validate_worker_request(
        self,
        context: _RunContext,
        message: Mapping[str, Any],
    ) -> None:
        if message.get("operation") not in SUPPORTED_OPERATIONS:
            raise BridgeProtocolError("unsupported worker operation")
        self._validate_message_fence(context, message)
        payload = message.get("payload")
        if payload is not None:
            _assert_bounded_safe(payload, max_bytes=self.config.max_frame_bytes)

    def _dispatch_operation(
        self,
        context: _RunContext,
        operation: str,
        payload: Any,
    ) -> Mapping[str, Any]:
        if operation == "loadState":
            return self._load_state(context, _mapping_payload(payload))
        if operation == "compareAndSetState":
            return self._compare_and_set_state(context, _mapping_payload(payload))
        if operation == "readHistory":
            return self._read_history(context, _mapping_payload(payload))
        if operation == "loadConversationMetadata":
            return self._load_conversation_metadata(
                context,
                _mapping_payload(payload),
            )
        if operation == "commitPage":
            return self._commit_page(context, _mapping_payload(payload))
        if operation == "loadReportSnapshot":
            return self._load_report_snapshot(context, _mapping_payload(payload))
        if operation == "finishRun":
            return self._finish_run(context, _mapping_payload(payload))
        if operation == "cancel":
            return self._cancel_run(context, _mapping_payload(payload))
        raise BridgeProtocolError("unsupported worker operation")

    def _load_state(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        kind = str(payload.get("kind") or "header")
        header, _ = self._call_context(
            context,
            self.state.load_state,
            collector_account_id=context.collector_account_id,
            profile_id=context.profile_id,
            operation=context.operation,
            on_result=lambda value: self._adopt_loaded_state_version(
                context,
                value,
            ),
        )
        state_version = header.state_version if header is not None else 0
        if kind == "header":
            return {
                "schemaVersion": "chatgpt-collector-state-v1",
                "stateVersion": state_version,
                "header": header.payload() if header is not None else None,
            }
        if kind == "schedule":
            state = header.schedule_transition if header is not None else None
            return {"stateVersion": state_version, "state": state}
        if kind == "candidates":
            limit = _bounded_limit(payload.get("limit"), default=64)
            expected = _optional_state_version(payload.get("expectedStateVersion"))
            if expected is None:
                raise BridgeProtocolError(
                    "candidate traversal requires an expected state version",
                    code="state_conflict",
                    retryable=True,
                )
            candidates = self._call_context(
                context,
                self.state.read_candidates,
                collector_account_id=context.collector_account_id,
                profile_id=context.profile_id,
                limit=limit,
                cursor=_optional_cursor(payload.get("cursor")),
                expected_state_version=expected,
                operation=context.operation,
                on_result=lambda value: setattr(
                    context,
                    "state_version",
                    int(value.state_version),
                ),
            )
            return {
                "stateVersion": candidates.state_version,
                "items": list(candidates.items),
                "nextCursor": candidates.cursor,
                "hasMore": candidates.has_more,
            }
        raise BridgeProtocolError("unsupported loadState variant")

    def _compare_and_set_state(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        kind = str(payload.get("kind") or "")
        if kind not in {"schedule", "history", "checkpoint"}:
            raise BridgeProtocolError("unsupported state mutation kind")
        expected = _required_state_version(payload.get("expectedStateVersion"))
        current, _ = self._call_context(
            context,
            self.state.load_state,
            collector_account_id=context.collector_account_id,
            profile_id=context.profile_id,
            operation=context.operation,
            on_result=lambda value: self._adopt_loaded_state_version(
                context,
                value,
            ),
        )
        expected_for_pg: Optional[int] = expected
        if current is None and expected == 0:
            expected_for_pg = None
        if current is not None and current.state_version != expected:
            raise BridgeProtocolError(
                "collector state version is stale",
                code="state_conflict",
                retryable=True,
            )
        state = payload.get("state", payload.get("next"))
        if state is not None and not isinstance(state, Mapping):
            raise BridgeProtocolError("state mutation must be an object or null")
        schedule = state if kind == "schedule" else None
        checkpoint = state if kind in {"history", "checkpoint"} else None
        active_trigger: Optional[Mapping[str, Any]] = None
        if kind == "schedule":
            if not isinstance(state, Mapping):
                raise BridgeProtocolError("schedule state must be an object")
            active = state.get("active")
            if isinstance(active, Mapping):
                active_trigger = dict(active)
                existing_run_id = active_trigger.get("runId")
                if existing_run_id is not None and existing_run_id != context.run_id:
                    raise BridgeProtocolError(
                        "schedule active trigger run does not match the parent run",
                        code="fence_invalid",
                    )
                active_trigger["runId"] = context.run_id
                fencing_token = active_trigger.get("fencingToken")
                if fencing_token is not None and fencing_token != (
                    context.lease.lease_fencing_token
                ):
                    raise BridgeProtocolError(
                        "schedule trigger fence does not match the parent lease",
                        code="fence_invalid",
                    )
                active_trigger["fencingToken"] = (
                    context.lease.lease_fencing_token
                )
            elif active is not None:
                raise BridgeProtocolError("schedule active trigger is invalid")
        def adopt_state_commit(value: Any) -> None:
            context.state_version = int(value.state_version)
            if kind == "schedule" and active_trigger is not None:
                context.trigger_id = _metadata_token(
                    active_trigger.get("triggerId"),
                    "triggerId",
                )

        header = self._call_context(
            context,
            self.state.compare_and_set_state,
            collector_account_id=context.collector_account_id,
            profile_id=context.profile_id,
            expected_state_version=expected_for_pg,
            schedule_transition=schedule,
            checkpoint=checkpoint,
            active_trigger=active_trigger,
            lease=context.lease,
            operation=context.operation,
            on_result=adopt_state_commit,
        )
        context.state_version = header.state_version
        if kind == "schedule" and active_trigger is not None:
            committed_trigger_id = _metadata_token(
                active_trigger.get("triggerId"),
                "triggerId",
            )
            context.trigger_id = committed_trigger_id
            self._verify_committed_trigger(context, committed_trigger_id)
        return {
            "stateVersion": header.state_version,
            "state": (
                header.schedule_transition
                if kind == "schedule"
                else header.checkpoint
            ),
        }

    def _prepare_history(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if context.preparation_count >= 1:
            raise BridgeProtocolError(
                "history preparation may only be requested once",
                code="protocol_invalid",
            )
        context.preparation_count += 1
        claimed_trigger_id = _metadata_token(
            payload.get("claimedTriggerId"),
            "claimedTriggerId",
        )
        if context.trigger_id != claimed_trigger_id:
            raise BridgeProtocolError(
                "history preparation is not bound to the claimed trigger",
                code="fence_invalid",
            )
        self._verify_committed_trigger(context, claimed_trigger_id)
        preparer = self.config.history_preparer
        if preparer is None:
            raise BridgeProtocolError(
                "native history contract is unavailable",
                code="history_contract_unavailable",
                retryable=True,
            )
        prepared = self._call_context(
            context,
            preparer,
            context.scope,
            context.collector_account_id,
            context.profile_id,
            context.operation,
            forward_operation=False,
            on_result=lambda value: self._retain_prepared_history(context, value),
        )
        if not isinstance(prepared, Mapping):
            raise BridgeProtocolError(
                "history preparation did not return a record",
                code="history_contract_unavailable",
                retryable=True,
            )
        # Keep the native owner before any privacy/type/deadline validation can
        # reject the returned record.
        context.prepared_history = prepared
        _assert_prepared_history_record(prepared)
        verified = _validate_prepared_history(context.scope, prepared)
        context.verified_history = verified
        context.capabilities = verified["capabilities"]
        context.capability_manifest = verified["capabilityManifest"]
        context.history_status = (
            "ready"
            if _native_history_capability_is_verified(
                context.capabilities,
                context.capability_manifest,
            )
            else "unavailable"
        )
        context.history_reason = None
        return verified

    def _verify_committed_trigger(
        self,
        context: _RunContext,
        trigger_id: str,
    ) -> None:
        header, _ = self._call_context(
            context,
            self.state.load_state,
            collector_account_id=context.collector_account_id,
            profile_id=context.profile_id,
            operation=context.operation,
            on_result=lambda value: self._adopt_loaded_state_version(
                context,
                value,
            ),
        )
        if header is None:
            raise BridgeProtocolError(
                "no committed active trigger exists",
                code="fence_invalid",
            )
        active = header.active_trigger
        if (
            not isinstance(active, Mapping)
            or active.get("triggerId") != trigger_id
            or active.get("runId") != context.run_id
            or active.get("fencingToken") != context.lease.lease_fencing_token
            or active.get("state", "active") not in {"active", "claimed", "running"}
            or header.state_version != context.state_version
        ):
            raise BridgeProtocolError(
                "claimed trigger is not active in durable state",
                code="fence_invalid",
            )
        binding = self._call_context(
            context,
            self.state.ledger.capture_binding,
            context.scope,
            seen_at=datetime.now(timezone.utc),
            operation=context.operation,
        )
        if (
            binding.scope_key != context.lease.binding.scope_key
            or binding.binding_generation
            != context.lease.binding.binding_generation
            or binding.binding_state != "active"
            or binding.identity_state != "verified"
        ):
            raise BridgeProtocolError(
                "claimed trigger binding is stale",
                code="fence_invalid",
            )

    def _read_history(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if context.prepared_history is None:
            raise BridgeProtocolError(
                "native history contract is unavailable",
                code="history_contract_unavailable",
                retryable=True,
            )
        kind = payload.get("kind")
        if kind not in {"index", "detail", "messages"}:
            raise BridgeProtocolError("unsupported history page kind")
        capability_for_kind = {
            "index": "index",
            "detail": "modern_detail",
            "messages": "messages",
        }[str(kind)]
        required_capability = payload.get("requiredCapability")
        if (
            required_capability not in {"index", "modern_detail", "messages"}
            or required_capability != capability_for_kind
        ):
            raise BridgeProtocolError(
                "history request must declare a required capability",
                code="history_contract_unavailable",
                retryable=True,
            )
        archive_scope = payload.get("archiveScope", payload.get("scope"))
        archived = payload.get("archived")
        if archived is not None and not isinstance(archived, bool):
            raise BridgeProtocolError(
                "history request archived flag is invalid",
                code="history_contract_unavailable",
                retryable=True,
            )
        implied_scope = "archived" if archived else "active"
        if archive_scope is None:
            archive_scope = implied_scope
        elif (
            not isinstance(archive_scope, str)
            or archive_scope not in {"active", "archived"}
            or (archived is not None and archive_scope != implied_scope)
        ):
            raise BridgeProtocolError(
                "history request archive scope is invalid",
                code="history_contract_unavailable",
                retryable=True,
            )
        if not _capability_is_available(
            context.capabilities,
            context.capability_manifest,
            str(required_capability),
            archive_scope=str(archive_scope),
        ):
            raise BridgeProtocolError(
                "requested history capability is unavailable",
                code="history_contract_unavailable",
                retryable=True,
            )
        reader = context.prepared_history.get("readHistory")
        if not callable(reader):
            raise BridgeProtocolError(
                "native history reader is unavailable",
                code="history_contract_unavailable",
                retryable=True,
            )
        if any(key in payload for key in ("url", "endpoint", "headers", "credentials")):
            raise BridgeProtocolError("history request contains forbidden transport fields")
        result = self._call_context(
            context,
            reader,
            dict(payload),
            context.operation,
            forward_operation=False,
        )
        if not isinstance(result, Mapping):
            raise BridgeProtocolError(
                "native history reader returned an invalid result",
                code="history_reader_failed",
                retryable=True,
            )
        _assert_bounded_safe(result, max_bytes=self.config.max_frame_bytes)
        status = result.get("status")
        if status == "blocked":
            reason = result.get("reason")
            if reason not in BLOCKED_HISTORY_REASONS:
                raise BridgeProtocolError("history blocked result has an invalid reason")
            retry_after = result.get("retryAfterMs")
            if retry_after is not None and (
                not isinstance(retry_after, int)
                or isinstance(retry_after, bool)
                or retry_after < 0
            ):
                raise BridgeProtocolError("history retryAfterMs is invalid")
            _record_history_status(
                context,
                "blocked",
                reason=str(reason),
                retry_after_ms=retry_after,
            )
        elif status in {"partial", "unavailable"}:
            normalized_status = str(status)
            _record_history_status(
                context,
                normalized_status,
                reason=str(result.get("reason") or normalized_status),
            )
        elif status == "ready":
            _record_history_status(context, "ready")
        elif status is not None:
            raise BridgeProtocolError("history result has an invalid status")
        return dict(result)

    def _load_conversation_metadata(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        conversation_id = _metadata_token(
            payload.get("conversationId"),
            "conversationId",
        )
        result = self._call_context(
            context,
            self.state.load_conversation_metadata,
            collector_account_id=context.collector_account_id,
            conversation_id=conversation_id,
            limit=_bounded_limit(payload.get("limit"), default=32),
            cursor=_optional_cursor(payload.get("cursor")),
            snapshot_id=_optional_cursor(payload.get("snapshotId")),
            deadline_at=context.operation.deadline_at,
            run_id=context.run_id,
            profile_id=context.profile_id,
            lease_fencing_token=context.lease.lease_fencing_token,
            operation=context.operation,
            on_result=lambda value: self._track_snapshot(context, payload, value),
        )
        _assert_bounded_safe(result, max_bytes=self.config.max_frame_bytes)
        return dict(result)

    def _commit_page(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        page_commit_id = _metadata_token(
            payload.get("pageCommitId")
            or _nested_value(payload, "mutations", "pageCommitId"),
            "pageCommitId",
        )
        expected = payload.get("expectedStateVersion")
        if expected is None and isinstance(payload.get("mutations"), Mapping):
            expected = payload["mutations"].get("expectedStateVersion")
        expected_version = _optional_state_version(expected)
        if expected_version is None:
            raise BridgeProtocolError(
                "page commit requires an expected state version",
                code="state_conflict",
                retryable=True,
            )
        canonical_payload = _validate_page_commit_payload(
            payload,
            expected_state_version=expected_version,
        )
        if context.trigger_id is None:
            raise BridgeProtocolError(
                "page commit is not bound to a committed trigger",
                code="fence_invalid",
            )
        def adopt_page_ack(value: Any) -> None:
            if isinstance(value, PageAck):
                context.state_version = int(value.state_version)

        ack: PageAck = self._call_context(
            context,
            self.state.commit_history_page,
            lease=context.lease,
            scope=context.scope,
            run_id=context.run_id,
            page_commit_id=page_commit_id,
            canonical_payload=canonical_payload,
            expected_state_version=expected_version,
            trigger_id=context.trigger_id,
            operation=context.operation,
            on_result=adopt_page_ack,
        )
        context.state_version = int(ack.state_version)
        return {
            "acknowledged": True,
            "pageCommitId": ack.page_commit_id,
            "stateVersion": ack.state_version,
            "payloadFingerprint": ack.payload_fingerprint,
        }

    def _load_report_snapshot(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = self._call_context(
            context,
            self.state.load_report_snapshot,
            collector_account_id=context.collector_account_id,
            limit=_bounded_limit(payload.get("limit"), default=256),
            cursor=_optional_cursor(payload.get("cursor")),
            snapshot_id=_optional_cursor(payload.get("snapshotId")),
            deadline_at=context.operation.deadline_at,
            run_id=context.run_id,
            profile_id=context.profile_id,
            lease_fencing_token=context.lease.lease_fencing_token,
            operation=context.operation,
            on_result=lambda value: self._track_snapshot(context, payload, value),
        )
        _assert_bounded_safe(result, max_bytes=self.config.max_frame_bytes)
        return dict(result)

    def _finish_run(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected = _required_state_version(payload.get("expectedVersion"))
        trigger_id = _metadata_token(payload.get("triggerId"), "triggerId")
        if context.trigger_id is None or trigger_id != context.trigger_id:
            raise BridgeProtocolError(
                "finish trigger does not match the claimed trigger",
                code="fence_invalid",
            )
        outcome = _metadata_token(payload.get("outcome"), "outcome")
        summary = _bounded_summary(payload.get("summary"))
        if context.history_status != "ready" and outcome == "success":
            raise BridgeProtocolError(
                "incomplete history cannot finish as success",
                code="state_conflict",
                retryable=True,
            )
        def adopt_finish(value: Any) -> None:
            context.state_version = int(value.state_version)
            context.finish_outcome = outcome
            context.finished = True

        header = self._call_context(
            context,
            self.state.finish_run,
            lease=context.lease,
            run_id=context.run_id,
            expected_state_version=expected,
            trigger_id=trigger_id,
            outcome=outcome,
            summary=summary,
            operation=context.operation,
            on_result=adopt_finish,
        )
        context.state_version = header.state_version
        context.finish_outcome = outcome
        context.finished = True
        return {"finished": True, "stateVersion": header.state_version}

    def _cancel_run(
        self,
        context: _RunContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected = _required_state_version(payload.get("expectedVersion"))
        trigger_id = _metadata_token(payload.get("triggerId"), "triggerId")
        outcome = _metadata_token(payload.get("outcome") or "cancelled", "outcome")
        summary = _bounded_summary(payload.get("summary"))
        if context.trigger_id is None or trigger_id != context.trigger_id:
            raise BridgeProtocolError(
                "cancel trigger does not match the claimed trigger",
                code="fence_invalid",
            )
        context.operation.cancel_event.set()
        context.cancelled = True
        terminal_operation = context.terminal()
        def adopt_cancel(value: Any) -> None:
            context.state_version = int(value.state_version)
            context.finish_outcome = outcome
            context.cancelled = True

        header = self._call_context(
            context,
            self.state.cancel,
            operation=terminal_operation,
            check_cancel=False,
            lease=context.lease,
            run_id=context.run_id,
            expected_state_version=expected,
            trigger_id=trigger_id,
            outcome=outcome,
            summary=summary,
            on_result=adopt_cancel,
        )
        context.state_version = header.state_version
        context.finish_outcome = outcome
        return {"cancelled": True, "stateVersion": header.state_version}

    def _await_child_exit(self, child: _Child) -> None:
        while True:
            if self._active_operation is not None:
                self._active_operation.check()
            remaining = max(0.0, child.deadline_monotonic - time.monotonic())
            try:
                return_code = child.process.wait(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired as exc:
                if remaining <= 0:
                    raise BridgeProtocolError(
                        "worker did not terminate after committed finish",
                        code="bounds_exceeded",
                        retryable=True,
                    ) from exc
        if return_code != 0:
            raise BridgeProtocolError(
                f"worker exited with status {return_code} after finish",
                code="history_reader_failed",
                retryable=True,
            )

    def _send_parent_cancel(
        self,
        context: _RunContext,
        *,
        reason: str,
        operation: Optional[BridgeOperationContext] = None,
    ) -> Optional[str]:
        if context.child.process.poll() is not None:
            return None
        if context.child.request_count >= self.config.max_requests:
            return None
        message = {
            "protocolVersion": PROTOCOL_VERSION,
            "control": "cancelRun",
            "requestId": str(uuid4()),
            "envelope": {
                "runId": context.run_id,
                "collectorAccountId": context.collector_account_id,
                "profileId": context.profile_id,
                "bindingGeneration": context.lease.binding.binding_generation,
                "leaseFencingToken": context.lease.lease_fencing_token,
            },
            "runId": context.run_id,
            "collectorAccountId": context.collector_account_id,
            "profileId": context.profile_id,
            "bindingGeneration": context.lease.binding.binding_generation,
            "leaseFencingToken": context.lease.lease_fencing_token,
            "reason": _metadata_token(reason, "reason"),
        }
        context.child.request_count += 1
        try:
            self._write_frame(
                context.child,
                message,
                enforce_deadline=False,
                deadline_monotonic=(
                    operation.deadline_monotonic
                    if operation is not None
                    else context.child.terminal_deadline_monotonic
                ),
            )
        except Exception as exc:
            return f"parent cancellation signal failed: {type(exc).__name__}"
        return None

    def _cancel_state(
        self,
        context: _RunContext,
        error: BaseException,
        *,
        operation: Optional[BridgeOperationContext] = None,
    ) -> Optional[str]:
        summary = {
            "reason": _failure_code(error),
            "coverageIncomplete": True,
        }
        active_operation = operation or context.terminal()
        cleanup_errors = self._close_tracked_snapshots(
            context,
            operation=active_operation,
        )
        if context.finished or context.finish_outcome is not None:
            return "; ".join(cleanup_errors) if cleanup_errors else None

        def adopt_cancel_state(value: Any) -> None:
            context.state_version = int(value.state_version)
            context.finish_outcome = "cancelled_after_start"
            context.cancelled = True

        try:
            self._call_context(
                context,
                self.state.cancel,
                operation=active_operation,
                check_cancel=False,
                lease=context.lease,
                run_id=context.run_id,
                expected_state_version=context.state_version,
                trigger_id=context.trigger_id,
                outcome="cancelled_after_start",
                summary=summary,
                on_result=adopt_cancel_state,
            )
        except Exception as exc:
            if context.finish_outcome is not None:
                return "; ".join(cleanup_errors) if cleanup_errors else None
            try:
                self._call_with_deadline(
                    active_operation.deadline_monotonic,
                    self.state.release_lease,
                    lease=context.lease,
                    operation=active_operation,
                    check_cancel=False,
                )
            except Exception as release_exc:
                cleanup_errors.append(
                    f"cancel state failed: {type(exc).__name__}; "
                    f"lease release failed: {type(release_exc).__name__}"
                )
            else:
                cleanup_errors.append(f"cancel state failed: {type(exc).__name__}")
        return "; ".join(cleanup_errors) if cleanup_errors else None

    def _retain_prepared_history(
        self,
        context: _RunContext,
        value: Any,
    ) -> None:
        if isinstance(value, Mapping):
            context.prepared_history = value

    def _close_prepared_history(
        self,
        context: _RunContext,
        *,
        operation: Optional[BridgeOperationContext] = None,
    ) -> Optional[str]:
        prepared = context.prepared_history
        if not isinstance(prepared, Mapping):
            return None
        capability = prepared.get("lifecycleCapability")
        registration = prepared.get("lifecycleRegistration")
        if capability is None and registration is None:
            return None
        if capability is None or registration is None:
            return "native history lifecycle owner is incomplete"
        retire = getattr(capability, "retire_native_history", None)
        retain = getattr(capability, "retain_native_history", None)
        cleanup = getattr(registration, "cleanup_callback", None)
        registration_id = getattr(registration, "registration_id", None)
        if (
            not callable(retire)
            or not callable(retain)
            or not callable(cleanup)
            or not isinstance(registration_id, str)
            or not registration_id
        ):
            return "native history lifecycle registration is invalid"
        active_operation = operation or context.terminal()
        cleanup_failure = getattr(registration, "cleanup_failure", None)
        if cleanup_failure:
            reason = "native history registration retained a cleanup failure"
            try:
                retain(registration, reason)
            except Exception as exc:
                return f"{reason}; retention failed: {type(exc).__name__}"
            return reason
        active_operation.cancel_event.set()
        try:
            completed = bool(cleanup(active_operation.deadline_monotonic))
        except Exception as exc:
            reason = f"native history retirement failed: {type(exc).__name__}"
            try:
                retain(registration, reason)
            except Exception as retain_exc:
                return f"{reason}; retention failed: {type(retain_exc).__name__}"
            return reason
        if getattr(registration, "cleanup_failure", None):
            reason = "native history registration retained a cleanup failure"
            try:
                retain(registration, reason)
            except Exception as exc:
                return f"{reason}; retention failed: {type(exc).__name__}"
            return reason
        if not completed or not bool(getattr(registration, "retired", False)):
            reason = "native history retirement was not proven"
            try:
                retain(registration, reason)
            except Exception as exc:
                return f"{reason}; retention failed: {type(exc).__name__}"
            return reason
        return None

    def _track_snapshot(
        self,
        context: _RunContext,
        request_payload: Mapping[str, Any],
        result: Any,
    ) -> None:
        if not isinstance(result, Mapping):
            return
        previous = request_payload.get("snapshotId")
        if (
            isinstance(previous, str)
            and previous
            and len(previous.encode("utf-8")) <= MAX_CURSOR_BYTES
            and not any(character in previous for character in ("\x00", "\r", "\n"))
        ):
            context.snapshot_ids.discard(previous)
        next_snapshot = result.get("snapshotId")
        if (
            isinstance(next_snapshot, str)
            and next_snapshot
            and len(next_snapshot.encode("utf-8")) <= MAX_CURSOR_BYTES
            and not any(
                character in next_snapshot for character in ("\x00", "\r", "\n")
            )
        ):
            context.snapshot_ids.add(next_snapshot)

    def _close_tracked_snapshots(
        self,
        context: _RunContext,
        *,
        operation: Optional[BridgeOperationContext] = None,
    ) -> list[str]:
        errors: list[str] = []
        active_operation = operation or context.terminal()
        for snapshot_id in tuple(context.snapshot_ids):
            closed = False
            try:
                self._call_with_deadline(
                    active_operation.deadline_monotonic,
                    self.state.close_snapshot,
                    snapshot_id,
                    operation=active_operation,
                    check_cancel=False,
                    forward_operation=False,
                )
                closed = True
            except Exception as exc:
                errors.append(
                    f"snapshot {snapshot_id} close failed: {type(exc).__name__}"
                )
            finally:
                if closed:
                    context.snapshot_ids.discard(snapshot_id)
        return errors

    def _release_lease_quietly(
        self,
        lease: CollectorLease,
        *,
        operation: Optional[BridgeOperationContext] = None,
    ) -> Optional[str]:
        try:
            if operation is None:
                self.state.release_lease(lease=lease)
            else:
                self._call_with_deadline(
                    operation.deadline_monotonic,
                    self.state.release_lease,
                    lease=lease,
                    operation=operation,
                    check_cancel=False,
                )
        except Exception as exc:
            return f"lease release failed: {type(exc).__name__}"
        return None

    def _write_frame(
        self,
        child: _Child,
        message: Mapping[str, Any],
        *,
        enforce_deadline: bool = True,
        deadline_monotonic: Optional[float] = None,
    ) -> None:
        if child.write_failed:
            raise BridgeProtocolError(
                "worker stdin is in a failed frame state",
                code="history_reader_failed",
                retryable=True,
            )
        remaining_total = self.config.max_total_bytes - child.total_bytes
        if remaining_total <= 0:
            raise BridgeProtocolError(
                "worker total byte bound exceeded",
                code="bounds_exceeded",
                retryable=True,
            )
        max_bytes = min(self.config.max_frame_bytes, remaining_total)
        _assert_bounded_safe(message, max_bytes=max_bytes)
        encoded = _encode_bounded_frame(
            message,
            max_bytes=max_bytes,
        )
        if child.process.stdin is None:
            raise BridgeProtocolError(
                "worker stdin is unavailable",
                code="history_reader_failed",
                retryable=True,
            )
        fd = child.process.stdin.fileno()
        child.total_bytes += len(encoded)
        write_reserved = True
        selector = selectors.DefaultSelector()
        view = memoryview(encoded)
        try:
            selector.register(fd, selectors.EVENT_WRITE)
            while view:
                if enforce_deadline:
                    self._check_deadline(child)
                    timeout = self._remaining_timeout(child)
                else:
                    terminal_deadline = (
                        deadline_monotonic
                        if deadline_monotonic is not None
                        else child.terminal_deadline_monotonic
                    )
                    timeout = max(0.0, terminal_deadline - time.monotonic())
                    if timeout <= 0:
                        raise BridgeProtocolError(
                            "worker terminal write deadline exceeded",
                            code="bounds_exceeded",
                            retryable=True,
                        )
                events = selector.select(min(0.1, timeout))
                if not events:
                    continue
                try:
                    written = os.write(fd, view)
                except BlockingIOError:
                    continue
                except BrokenPipeError as exc:
                    raise BridgeProtocolError(
                        "worker stdin closed",
                        code="history_reader_failed",
                        retryable=True,
                    ) from exc
                if written <= 0:
                    raise BridgeProtocolError(
                        "worker write made no progress",
                        code="history_reader_failed",
                        retryable=True,
                    )
                view = view[written:]
        except BaseException:
            if write_reserved:
                child.write_failed = True
            raise
        finally:
            selector.close()

    def _read_message(self, context: _RunContext) -> Mapping[str, Any]:
        frame = self._read_frame(context.child)
        # Count the complete inbound frame before JSON decoding, request-ID
        # validation, or control/operation dispatch.
        self._consume_worker_request(context)
        try:
            message = json.loads(frame.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeProtocolError("worker returned invalid JSON") from exc
        if not isinstance(message, Mapping):
            raise BridgeProtocolError("worker frame is not an object")
        _assert_bounded_safe(message, max_bytes=self.config.max_frame_bytes)
        return message

    def _read_frame(self, child: _Child) -> bytes:
        if child.process.stdout is None:
            raise BridgeProtocolError(
                "worker stdout is unavailable",
                code="history_reader_failed",
                retryable=True,
            )
        fd = child.process.stdout.fileno()
        selector = selectors.DefaultSelector()
        try:
            selector.register(fd, selectors.EVENT_READ)
            while True:
                self._check_deadline(child)
                if child.ready_frames:
                    return child.ready_frames.popleft()
                newline = child.read_buffer.find(b"\n")
                if newline >= 0:
                    frame = bytes(child.read_buffer[:newline])
                    del child.read_buffer[: newline + 1]
                    self._validate_frame_size(frame)
                    return frame
                if len(child.read_buffer) >= self.config.max_frame_bytes:
                    raise BridgeProtocolError(
                        "worker response frame exceeds the configured bound",
                        code="bounds_exceeded",
                        retryable=True,
                    )
                events = selector.select(min(0.1, self._remaining_timeout(child)))
                if not events:
                    continue
                remaining_frame = self.config.max_frame_bytes - len(child.read_buffer)
                remaining_total = self.config.max_total_bytes - child.total_bytes
                read_size = min(8192, remaining_frame, remaining_total)
                if read_size <= 0:
                    raise BridgeProtocolError(
                        "worker input bounds exceeded",
                        code="bounds_exceeded",
                        retryable=True,
                    )
                try:
                    chunk = os.read(fd, read_size)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    if exc.errno == errno.EINTR:
                        continue
                    raise BridgeProtocolError(
                        "worker stdout read failed",
                        code="history_reader_failed",
                        retryable=True,
                    ) from exc
                if not chunk:
                    if child.process.poll() is not None:
                        raise BridgeProtocolError(
                            "worker closed stdout before finish",
                            code="history_reader_failed",
                            retryable=True,
                        )
                    continue
                child.total_bytes += len(chunk)
                if child.total_bytes > self.config.max_total_bytes:
                    raise BridgeProtocolError(
                        "worker total byte bound exceeded",
                        code="bounds_exceeded",
                        retryable=True,
                    )
                child.read_buffer.extend(chunk)
                self._queue_complete_frames(child)
        finally:
            selector.close()

    def _queue_complete_frames(self, child: _Child) -> None:
        while True:
            newline = child.read_buffer.find(b"\n")
            if newline < 0:
                if len(child.read_buffer) >= self.config.max_frame_bytes:
                    raise BridgeProtocolError(
                        "worker response frame exceeds the configured bound",
                        code="bounds_exceeded",
                        retryable=True,
                    )
                return
            frame = bytes(child.read_buffer[:newline])
            del child.read_buffer[: newline + 1]
            self._validate_frame_size(frame)
            child.ready_frames.append(frame)

    def _validate_frame_size(self, frame: bytes) -> None:
        if len(frame) + 1 > self.config.max_frame_bytes:
            raise BridgeProtocolError(
                "worker response frame exceeds the configured bound",
                code="bounds_exceeded",
                retryable=True,
            )

    def _check_deadline(self, child: _Child) -> None:
        if self._active_operation is not None:
            self._active_operation.check()
        if child.process.poll() is not None and not child.ready_frames:
            raise BridgeProtocolError(
                "worker exited before completing the run",
                code="history_reader_failed",
                retryable=True,
            )
        if time.monotonic() >= child.deadline_monotonic:
            raise BridgeProtocolError(
                "worker deadline exceeded",
                code="bounds_exceeded",
                retryable=True,
            )

    def _remaining_timeout(self, child: _Child) -> float:
        return max(0.0, child.deadline_monotonic - time.monotonic())

    def _reap(
        self,
        *,
        key: tuple[str, str],
        child: _Child,
        deadline_monotonic: Optional[float] = None,
    ) -> Optional[str]:
        process = child.process
        resolved = False
        failure: Optional[str] = None
        terminal_deadline = min(
            time.monotonic() + TERMINAL_CLEANUP_SECONDS,
            child.terminal_deadline_monotonic,
            deadline_monotonic
            if deadline_monotonic is not None
            else child.terminal_deadline_monotonic,
        )

        def wait_for_retirement(deadline: float) -> bool:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    process.wait(timeout=min(0.05, remaining))
                except subprocess.TimeoutExpired:
                    pass
                except OSError as exc:
                    nonlocal failure
                    failure = (
                        failure
                        or f"worker reap wait failed: {type(exc).__name__}"
                    )
                    break
                if process.poll() is not None and not _process_group_exists(
                    child.process_group_id
                ):
                    return True
            return process.poll() is not None and not _process_group_exists(
                child.process_group_id
            )

        try:
            _terminate_process_group(process, child.process_group_id)
            term_deadline = min(
                terminal_deadline,
                time.monotonic()
                + min(1.0, max(0.0, terminal_deadline - time.monotonic()) / 2),
            )
            if not wait_for_retirement(term_deadline):
                _kill_process_group(process, child.process_group_id)
                resolved = wait_for_retirement(terminal_deadline)
            else:
                resolved = True
            if not resolved:
                failure = failure or "worker process group retirement was not proven"
        except Exception as exc:
            failure = failure or f"worker reap failed: {type(exc).__name__}"
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            with self._lock:
                if resolved and self._children.get(key) is child:
                    self._children.pop(key, None)
        return None if resolved else failure or "worker retirement was not proven"

    def request_shutdown(self) -> None:
        """Stop admission and cancel the current owned operation."""
        self._admitting = False
        operation = self._active_operation
        if operation is not None:
            operation.cancel_event.set()

    def close(self) -> list[str]:
        """Boundedly reap every child still owned by this bridge."""
        self.request_shutdown()
        return self.reap_pending()

    def reap_pending(self) -> list[str]:
        """Service retained children between sidecar runs without reopening admission."""
        with self._lock:
            children = list(self._children.items())
        errors: list[str] = []
        for key, child in children:
            error = self._reap(key=key, child=child)
            if error is not None:
                errors.append(error)
        return errors


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Permission denied is unresolved ownership, not proof of absence.
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen,
    process_group_id: Optional[int] = None,
) -> None:
    try:
        os.killpg(process_group_id or process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return


def _kill_process_group(
    process: subprocess.Popen,
    process_group_id: Optional[int] = None,
) -> None:
    try:
        os.killpg(process_group_id or process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _record_history_status(
    context: _RunContext,
    status: str,
    *,
    reason: Optional[str] = None,
    retry_after_ms: Optional[int] = None,
) -> None:
    if _HISTORY_STATUS_RANK[status] < _HISTORY_STATUS_RANK[context.history_status]:
        return
    context.history_status = status
    if status == "ready":
        context.history_reason = None
        return
    if reason is not None:
        context.history_reason = reason
    if status == "blocked":
        context.retry_after_ms = retry_after_ms


def _verified_scope(scope: LedgerScope, account: str) -> LedgerScope:
    if scope.collector_account_id != account:
        raise BridgeProtocolError("scope account does not match the run account")
    for value, name in (
        (scope.provider, "provider"),
        (scope.provider_user_id, "provider_user_id"),
        (scope.workspace_id, "workspace_id"),
        (scope.quota_owner_id, "quota_owner_id"),
        (scope.surface, "surface"),
    ):
        normalized = _metadata_token(value, name)
        if normalized.lower() in {"unknown", "none", "null"}:
            raise BridgeProtocolError(
                "verified provider identity is required before lease claim",
                code="history_contract_unavailable",
                retryable=True,
            )
    if scope.surface != "chat":
        raise BridgeProtocolError("usage bridge scope must remain surface=chat")
    return scope


def _scope_for_account(scope: LedgerScope, account: str) -> LedgerScope:
    if scope.collector_account_id != account:
        raise BridgeProtocolError("scope account does not match the run account")
    return scope


def _verified_scope_if_complete(scope: LedgerScope) -> LedgerScope:
    values = (
        scope.collector_account_id,
        scope.provider,
        scope.provider_user_id,
        scope.workspace_id,
        scope.quota_owner_id,
        scope.surface,
    )
    if any(value is None for value in values):
        raise BridgeProtocolError(
            "collector scope identity is incomplete",
            code="history_contract_unavailable",
            retryable=True,
        )
    return _verified_scope(scope, scope.collector_account_id)


def _validate_prepared_history(
    parent_scope: LedgerScope,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    identity = prepared.get("identity")
    capabilities = prepared.get("capabilities")
    manifest = prepared.get("capabilityManifest")
    scope_payload = prepared.get("scope")
    if not all(
        isinstance(value, Mapping)
        for value in (identity, capabilities, manifest, scope_payload)
    ):
        raise BridgeProtocolError(
            "history preparation lacks verified identity or capabilities",
            code="history_contract_unavailable",
            retryable=True,
        )
    verified_scope = _scope_from_payload(
        scope_payload,
        parent_scope.collector_account_id,
    )
    try:
        verified_scope = _verified_scope(
            verified_scope,
            parent_scope.collector_account_id,
        )
    except BridgeProtocolError as exc:
        raise BridgeProtocolError(
            "history identity is not a verified native scope",
            code="history_contract_unavailable",
            retryable=True,
        ) from exc
    if scope_key(verified_scope) != scope_key(parent_scope):
        raise BridgeProtocolError(
            "history identity does not match the parent scope",
            code="fence_invalid",
        )
    normalized_identity = _validate_identity_record(identity)
    provider_user_id = _first_value(identity, "providerUserId", "provider_user_id")
    workspace_id = _first_value(identity, "workspaceId", "workspace_id")
    quota_owner_id = _first_value(identity, "quotaOwnerId", "quota_owner_id")
    surface = _first_value(identity, "surface")
    auth_state = _first_value(identity, "authState", "auth_state")
    if (
        provider_user_id != parent_scope.provider_user_id
        or workspace_id != parent_scope.workspace_id
        or quota_owner_id != parent_scope.quota_owner_id
        or surface != parent_scope.surface
        or surface != "chat"
        or auth_state not in _IDENTITY_AUTH_STATES
        or normalized_identity["identityErrors"]
    ):
        raise BridgeProtocolError(
            "history identity is not freshly verified",
            code="history_contract_unavailable",
            retryable=True,
        )
    normalized_capabilities = _validate_capability_record(capabilities)
    normalized_manifest = _validate_capability_manifest(
        manifest,
        normalized_identity,
        normalized_capabilities,
    )
    return {
        "identity": normalized_identity,
        "capabilities": normalized_capabilities,
        "capabilityManifest": normalized_manifest,
        "scope": _scope_wire(verified_scope),
    }


def _assert_prepared_history_record(prepared: Any) -> None:
    if not isinstance(prepared, Mapping):
        raise BridgeProtocolError(
            "history preparation did not return an object",
            code="history_contract_unavailable",
            retryable=True,
        )
    json_fields = ("identity", "capabilities", "capabilityManifest", "scope")
    for field_name in json_fields:
        value = prepared.get(field_name)
        if not isinstance(value, Mapping):
            raise BridgeProtocolError(
                f"history preparation field {field_name} is invalid",
                code="history_contract_unavailable",
                retryable=True,
            )
        _assert_bounded_safe(value, max_bytes=MAX_FRAME_BYTES)
    reader = prepared.get("readHistory")
    if not callable(reader):
        raise BridgeProtocolError(
            "native history reader is invalid",
            code="history_contract_unavailable",
            retryable=True,
        )
    capability = prepared.get("lifecycleCapability")
    registration = prepared.get("lifecycleRegistration")
    if capability is None or registration is None:
        raise BridgeProtocolError(
            "native history lifecycle owner is incomplete",
            code="history_contract_unavailable",
            retryable=True,
        )
    if (
        not callable(getattr(capability, "retire_native_history", None))
        or not callable(getattr(capability, "retain_native_history", None))
        or not callable(getattr(registration, "cleanup_callback", None))
        or not isinstance(getattr(registration, "registration_id", None), str)
        or not getattr(registration, "registration_id", "").strip()
    ):
        raise BridgeProtocolError(
            "native history lifecycle registration is invalid",
            code="history_contract_unavailable",
            retryable=True,
        )


def _validate_identity_record(value: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "providerUserId",
        "workspaceId",
        "quotaOwnerId",
        "surface",
        "authState",
        "identityErrors",
    )
    if any(field_name not in value for field_name in required):
        raise BridgeProtocolError(
            "history identity record is incomplete",
            code="history_contract_unavailable",
            retryable=True,
        )
    for field_name in ("providerUserId", "workspaceId", "quotaOwnerId"):
        if not isinstance(value[field_name], str) or not value[field_name].strip():
            raise BridgeProtocolError(
                "history identity record contains an invalid owner field",
                code="history_contract_unavailable",
                retryable=True,
            )
    if value["surface"] != "chat" or not isinstance(value["authState"], str):
        raise BridgeProtocolError(
            "history identity surface or auth state is invalid",
            code="history_contract_unavailable",
            retryable=True,
        )
    errors = value["identityErrors"]
    if not isinstance(errors, list) or any(
        not isinstance(item, str) for item in errors
    ):
        raise BridgeProtocolError(
            "history identity errors are invalid",
            code="history_contract_unavailable",
            retryable=True,
        )
    return dict(value)


def _validate_capability_record(value: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "adapterVersion",
        "indexScopes",
        "archiveBehavior",
        "projectCoverage",
        "modernDetail",
        "pagination",
        "legacySupport",
        "branchVisibility",
        "modelMetadata",
        "quotaMetadata",
        "warnings",
    )
    if any(field_name not in value for field_name in required):
        raise BridgeProtocolError(
            "history capability record is incomplete",
            code="history_contract_unavailable",
            retryable=True,
        )
    if not isinstance(value["adapterVersion"], str) or not value["adapterVersion"].strip():
        raise BridgeProtocolError(
            "history capability adapter version is invalid",
            code="history_contract_unavailable",
            retryable=True,
        )
    index_scopes = value["indexScopes"]
    warnings = value["warnings"]
    if (
        not isinstance(index_scopes, list)
        or any(
            not isinstance(item, str)
            or not item.strip()
            or len(item.encode("utf-8")) > MAX_CURSOR_BYTES
            for item in index_scopes
        )
        or not isinstance(warnings, list)
        or any(not isinstance(item, str) for item in warnings)
    ):
        raise BridgeProtocolError(
            "history capability inventory contains invalid arrays",
            code="history_contract_unavailable",
            retryable=True,
        )
    if value["projectCoverage"] not in {"complete", "partial", "unknown"}:
        raise BridgeProtocolError(
            "history capability project coverage is invalid",
            code="history_contract_unavailable",
            retryable=True,
        )
    for field_name in (
        "archiveBehavior",
        "modernDetail",
        "pagination",
        "legacySupport",
        "branchVisibility",
        "modelMetadata",
        "quotaMetadata",
    ):
        if (
            not isinstance(value[field_name], str)
            or not value[field_name].strip()
        ):
            raise BridgeProtocolError(
                "history capability inventory contains an invalid field",
                code="history_contract_unavailable",
                retryable=True,
            )
    return dict(value)


def _validate_capability_manifest(
    value: Mapping[str, Any],
    identity: Mapping[str, Any],
    capabilities: Mapping[str, Any],
) -> dict[str, Any]:
    adapter_version = _first_value(value, "adapterVersion", "adapter_version")
    surface = _first_value(value, "surface", "conversationSurface")
    manifest_identity = value.get("identity")
    native_history = _first_value(value, "nativeHistory", "native_history", "history")
    if (
        adapter_version != capabilities["adapterVersion"]
        or surface != identity["surface"]
        or not isinstance(manifest_identity, Mapping)
        or not isinstance(native_history, Mapping)
    ):
        raise BridgeProtocolError(
            "history capability manifest does not match its inventory",
            code="history_contract_unavailable",
            retryable=True,
        )
    for field_name in ("providerUserId", "workspaceId", "quotaOwnerId", "surface"):
        if _first_value(
            manifest_identity,
            field_name,
            _snake_case(field_name),
        ) != identity[field_name]:
            raise BridgeProtocolError(
                "history capability manifest identity does not match the session",
                code="history_contract_unavailable",
                retryable=True,
            )
    kind = _first_value(native_history, "kind", "historyKind")
    archive_scopes = _first_value(
        native_history,
        "archiveScopes",
        "archive_scopes",
        "scopes",
    )
    operation_claims = _first_value(
        native_history,
        "operationClaims",
        "operation_claims",
        "operations",
    )
    status = _first_value(native_history, "status", "state")
    if (
        kind not in _NATIVE_HISTORY_KINDS
        or status not in (
            _UNAVAILABLE_CAPABILITY_STATES
            | {"available", "verified", "supported", True, False}
        )
        or not isinstance(archive_scopes, list)
        or any(
            not isinstance(scope, str)
            or scope not in {"active", "archived"}
            for scope in archive_scopes
        )
        or not isinstance(operation_claims, Mapping)
        or any(scope not in capabilities["indexScopes"] for scope in archive_scopes)
    ):
        raise BridgeProtocolError(
            "native history capability manifest is incomplete",
            code="history_contract_unavailable",
            retryable=True,
        )
    for capability_name, aliases in _HISTORY_OPERATION_ALIASES.items():
        claims = _first_value(operation_claims, capability_name, *aliases)
        if claims is None:
            raise BridgeProtocolError(
                "native history capability manifest lacks an operation claim",
                code="history_contract_unavailable",
                retryable=True,
            )
        if isinstance(claims, Mapping):
            if not claims:
                raise BridgeProtocolError(
                    "native history operation claim is empty",
                    code="history_contract_unavailable",
                    retryable=True,
                )
            claim_status = _first_value(claims, "status", "state", "available")
            claim_operations = _first_value(
                claims,
                "operations",
                "claims",
                "methods",
            )
            if (
                claim_status is not None
                and claim_status not in (
                    _UNAVAILABLE_CAPABILITY_STATES
                    | {"available", "verified", "supported", True, False}
                )
            ):
                raise BridgeProtocolError(
                    "native history operation claim is invalid",
                    code="history_contract_unavailable",
                    retryable=True,
                )
            if claim_operations is not None and (
                not isinstance(claim_operations, list)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in claim_operations
                )
            ):
                raise BridgeProtocolError(
                    "native history operation claim is invalid",
                    code="history_contract_unavailable",
                    retryable=True,
                )
            if claim_status is None and claim_operations is None:
                raise BridgeProtocolError(
                    "native history operation claim is incomplete",
                    code="history_contract_unavailable",
                    retryable=True,
                )
        elif isinstance(claims, list):
            if any(
                not isinstance(item, str) or not item.strip()
                for item in claims
            ):
                raise BridgeProtocolError(
                    "native history operation claim is invalid",
                    code="history_contract_unavailable",
                    retryable=True,
                )
        elif claims not in (
            _UNAVAILABLE_CAPABILITY_STATES
            | {True, False, "available", "verified", "supported"}
        ):
            raise BridgeProtocolError(
                "native history operation claim is invalid",
                code="history_contract_unavailable",
                retryable=True,
            )
    return dict(value)


def _native_history_capability_is_verified(
    capabilities: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    marker = _first_value(manifest, "nativeHistory", "native_history", "history")
    if isinstance(marker, Mapping):
        marker = marker.get("status") or marker.get("state")
    if marker not in {"available", "verified", "supported", True}:
        return False
    scopes = capabilities.get("indexScopes", ())
    if not scopes or any(scope not in {"active", "archived"} for scope in scopes):
        return False
    available = {
        "index": True,
        "modern_detail": capabilities.get("modernDetail")
        not in _UNAVAILABLE_CAPABILITY_STATES,
        "messages": capabilities.get("pagination")
        not in _UNAVAILABLE_CAPABILITY_STATES,
    }
    return any(
        inventory_available
        and _manifest_operation_is_available(manifest, operation)
        and any(
            _manifest_archive_scope_is_available(manifest, operation, scope)
            for scope in scopes
        )
        for operation, inventory_available in available.items()
    )


def _capability_is_available(
    capabilities: Optional[Mapping[str, Any]],
    manifest: Optional[Mapping[str, Any]],
    required_capability: str,
    *,
    archive_scope: str = "active",
) -> bool:
    if capabilities is None or manifest is None:
        return False
    if not _native_history_capability_is_verified(capabilities, manifest):
        return False
    if not _manifest_operation_is_available(manifest, required_capability):
        return False
    if not _manifest_archive_scope_is_available(
        manifest,
        required_capability,
        archive_scope,
    ):
        return False
    if required_capability == "index":
        return (
            bool(capabilities.get("indexScopes"))
            and archive_scope in capabilities["indexScopes"]
        )
    if required_capability == "modern_detail":
        return capabilities.get("modernDetail") not in _UNAVAILABLE_CAPABILITY_STATES
    if required_capability == "messages":
        return capabilities.get("pagination") not in _UNAVAILABLE_CAPABILITY_STATES
    return False


def _manifest_archive_scope_is_available(
    manifest: Mapping[str, Any],
    required_capability: str,
    archive_scope: str,
) -> bool:
    if archive_scope not in {"active", "archived"}:
        return False
    native_history = _first_value(
        manifest,
        "nativeHistory",
        "native_history",
        "history",
    )
    if not isinstance(native_history, Mapping):
        return False
    archive_scopes = _first_value(
        native_history,
        "archiveScopes",
        "archive_scopes",
        "scopes",
    )
    if not isinstance(archive_scopes, list) or archive_scope not in archive_scopes:
        return False
    operation_claims = _first_value(
        native_history,
        "operationClaims",
        "operation_claims",
        "operations",
    )
    if not isinstance(operation_claims, Mapping):
        return False
    claim = _first_value(
        operation_claims,
        required_capability,
        *_HISTORY_OPERATION_ALIASES.get(required_capability, ()),
    )
    if isinstance(claim, Mapping):
        claimed_scopes = _first_value(
            claim,
            "archiveScopes",
            "archive_scopes",
            "scopes",
            "availableScopes",
            "available_scopes",
        )
        if claimed_scopes is not None:
            return (
                isinstance(claimed_scopes, list)
                and archive_scope in claimed_scopes
            )
    return True


def _manifest_operation_is_available(
    manifest: Mapping[str, Any],
    required_capability: str,
) -> bool:
    native_history = _first_value(
        manifest,
        "nativeHistory",
        "native_history",
        "history",
    )
    if not isinstance(native_history, Mapping):
        return False
    operation_claims = _first_value(
        native_history,
        "operationClaims",
        "operation_claims",
        "operations",
    )
    if not isinstance(operation_claims, Mapping):
        return False
    aliases = _HISTORY_OPERATION_ALIASES.get(required_capability, ())
    claim = _first_value(operation_claims, required_capability, *aliases)
    if isinstance(claim, Mapping):
        status = _first_value(claim, "status", "state", "available")
        operations = _first_value(claim, "operations", "claims", "methods")
        if status is not None:
            return status in {"available", "verified", "supported", True}
        return isinstance(operations, list) and any(
            isinstance(item, str)
            and item.strip()
            and item not in _UNAVAILABLE_CAPABILITY_STATES
            for item in operations
        )
    if isinstance(claim, list):
        return any(
            isinstance(item, str)
            and item.strip()
            and item not in _UNAVAILABLE_CAPABILITY_STATES
            for item in claim
        )
    return claim in {True, "available", "verified", "supported"}


def _wire_history_identity(prepared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": dict(prepared["identity"]),
        "capabilities": dict(prepared["capabilities"]),
        "capabilityManifest": dict(prepared["capabilityManifest"]),
        "scope": dict(prepared["scope"]),
    }


def _scope_from_payload(payload: Mapping[str, Any], account: str) -> LedgerScope:
    return LedgerScope(
        collector_account_id=account,
        provider=_first_value(payload, "provider"),
        provider_user_id=_first_value(payload, "providerUserId", "provider_user_id"),
        workspace_id=_first_value(payload, "workspaceId", "workspace_id"),
        quota_owner_id=_first_value(payload, "quotaOwnerId", "quota_owner_id"),
        surface=_first_value(payload, "surface"),
    )


def _scope_wire(scope: LedgerScope) -> dict[str, Any]:
    return {
        "collectorAccountId": scope.collector_account_id,
        "provider": scope.provider,
        "providerUserId": scope.provider_user_id,
        "workspaceId": scope.workspace_id,
        "quotaOwnerId": scope.quota_owner_id,
        "surface": scope.surface,
    }


def _first_value(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _snake_case(value: str) -> str:
    return value[0].lower() + "".join(
        f"_{character.lower()}" if character.isupper() else character
        for character in value[1:]
    )


def _nested_value(payload: Mapping[str, Any], outer: str, inner: str) -> Any:
    nested = payload.get(outer)
    return nested.get(inner) if isinstance(nested, Mapping) else None


def _mapping_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeProtocolError("worker operation payload must be an object")
    _preflight_json_value(value, max_bytes=MAX_FRAME_BYTES)
    return value


def _metadata_token(value: Any, field_name: str) -> str:
    normalized = sanitize_token(value)
    if normalized is None:
        raise BridgeProtocolError(f"{field_name} is not a supported metadata token")
    return normalized


def _request_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_REQUEST_ID_LENGTH:
        raise BridgeProtocolError("worker request id is invalid")
    return value


def _safe_request_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value and len(value) <= MAX_REQUEST_ID_LENGTH:
        return value
    return ""


def _optional_cursor(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BridgeProtocolError("cursor must be a non-empty opaque string")
    if len(value.encode("utf-8")) > MAX_CURSOR_BYTES:
        raise BridgeProtocolError(
            "cursor exceeds the supported bound",
            code="bounds_exceeded",
            retryable=True,
        )
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise BridgeProtocolError("cursor contains a forbidden control character")
    return value


def _bounded_limit(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > 256:
        raise BridgeProtocolError("worker page limit is outside the supported range")
    return value


def _required_state_version(value: Any) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > 2**63 - 1
    ):
        raise BridgeProtocolError("worker state version is invalid")
    return value


def _optional_state_version(value: Any) -> Optional[int]:
    if value is None:
        return None
    return _required_state_version(value)


def _remaining_ms(deadline_monotonic: float) -> int:
    return max(0, int((deadline_monotonic - time.monotonic()) * 1000))


def _wall_deadline(child: _Child) -> datetime:
    remaining = max(0.0, child.deadline_monotonic - time.monotonic())
    return datetime.now(timezone.utc) + timedelta(seconds=remaining)


def _bounded_summary(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeProtocolError("finish summary must be an object")
    _preflight_json_value(value, max_bytes=64 * 1024)
    try:
        _encode_bounded_frame(value, max_bytes=(64 * 1024) + 1)
    except BridgeProtocolError as exc:
        if exc.code == "bounds_exceeded":
            raise BridgeProtocolError(
                "finish summary exceeds the supported bound",
                code="bounds_exceeded",
                retryable=True,
            ) from exc
        raise
    _assert_bounded_safe(value, max_bytes=64 * 1024)
    return dict(value)


def _encode_bounded_frame(
    value: Any,
    *,
    max_bytes: int,
) -> bytes:
    """Encode one JSON frame without allocating an unbounded serialized copy."""
    if max_bytes <= 1:
        raise BridgeProtocolError(
            "worker frame bound is too small",
            code="bounds_exceeded",
            retryable=True,
        )
    _preflight_json_value(value, max_bytes=max_bytes - 1)
    encoded = bytearray()
    encoder = json.JSONEncoder(
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        for chunk in encoder.iterencode(value):
            encoded_chunk = chunk.encode("utf-8")
            if len(encoded) + len(encoded_chunk) + 1 > max_bytes:
                raise BridgeProtocolError(
                    "worker frame exceeds the configured bound",
                    code="bounds_exceeded",
                    retryable=True,
                )
            encoded.extend(encoded_chunk)
    except BridgeProtocolError:
        raise
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise BridgeProtocolError(
            "worker frame is not valid JSON",
            code="protocol_invalid",
        ) from exc
    encoded.append(0x0A)
    return bytes(encoded)


def _assert_bounded_safe(value: Any, *, max_bytes: int) -> None:
    _preflight_json_value(value, max_bytes=max_bytes)
    try:
        assert_no_secrets(_privacy_walk_value(value))
    except PrivacyError as exc:
        raise BridgeProtocolError(
            "secret-like value survived sanitization",
            code="protocol_invalid",
        ) from exc


def _privacy_walk_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _privacy_walk_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_privacy_walk_value(child) for child in value]
    return value


def _preflight_json_value(  # noqa: PLR0915 - bounded recursive JSON walk
    value: Any,
    *,
    max_bytes: int,
    path: str = "root",
) -> None:
    """Bound JSON-compatible values before privacy walks or JSON encoding."""
    if max_bytes <= 0:
        raise BridgeProtocolError(
            "JSON value bound is too small",
            code="bounds_exceeded",
            retryable=True,
        )
    used = 0
    active_ids: set[int] = set()

    def add(amount: int, current_path: str) -> None:
        nonlocal used
        used += amount
        if used > max_bytes:
            raise BridgeProtocolError(
                f"JSON value exceeds the supported bound at {current_path}",
                code="bounds_exceeded",
                retryable=True,
            )

    def string_size(current: str, current_path: str) -> int:
        size = 2
        for character in current:
            codepoint = ord(character)
            if character in {'"', "\\"}:
                size += 2
            elif codepoint < 0x20:
                size += 2 if character in "\b\f\n\r\t" else 6
            else:
                try:
                    size += len(character.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise BridgeProtocolError(
                        f"JSON string is not valid UTF-8 at {current_path}",
                        code="protocol_invalid",
                    ) from exc
            if size > max_bytes:
                raise BridgeProtocolError(
                    f"JSON scalar exceeds the supported bound at {current_path}",
                    code="bounds_exceeded",
                    retryable=True,
                )
        return size

    def integer_size(current: int, current_path: str) -> int:
        bits = current.bit_length()
        # Reject oversized integers before allocating decimal text; then
        # count exactly so a supported boundary value is not rejected.
        minimum_digits = max(1, (max(0, bits - 1) * 3) // 10 + 1)
        sign_size = 1 if current < 0 else 0
        if used + minimum_digits + sign_size > max_bytes:
            add(minimum_digits + sign_size, current_path)
        try:
            return len(str(current))
        except ValueError as exc:
            raise BridgeProtocolError(
                f"JSON integer cannot be encoded at {current_path}",
                code="protocol_invalid",
            ) from exc

    def walk(current: Any, current_path: str, depth: int) -> None:
        if depth > MAX_JSON_DEPTH:
            raise BridgeProtocolError(
                f"JSON nesting exceeds the supported bound at {current_path}",
                code="bounds_exceeded",
                retryable=True,
            )
        if isinstance(current, str):
            add(string_size(current, current_path), current_path)
            return
        if current is None:
            add(4, current_path)
            return
        if isinstance(current, bool):
            add(4 if current else 5, current_path)
            return
        if isinstance(current, int):
            add(integer_size(current, current_path), current_path)
            return
        if isinstance(current, float):
            if not math.isfinite(current):
                raise BridgeProtocolError(
                    f"JSON number is not finite at {current_path}",
                    code="protocol_invalid",
                )
            add(len(repr(current)), current_path)
            return
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in active_ids:
                raise BridgeProtocolError(
                    f"JSON value contains a cycle at {current_path}",
                    code="protocol_invalid",
                )
            active_ids.add(marker)
            add(2, current_path)
            if len(current) > max_bytes:
                active_ids.discard(marker)
                raise BridgeProtocolError(
                    f"JSON object contains too many entries at {current_path}",
                    code="bounds_exceeded",
                    retryable=True,
                )
            for index, (raw_key, child) in enumerate(current.items()):
                if not isinstance(raw_key, str):
                    raise BridgeProtocolError(
                        f"JSON object key is not a string at {current_path}",
                        code="protocol_invalid",
                    )
                if index:
                    add(1, current_path)
                add(string_size(raw_key, f"{current_path}.<key>") + 1, current_path)
                walk(child, f"{current_path}.{raw_key}", depth + 1)
            active_ids.discard(marker)
            return
        if isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in active_ids:
                raise BridgeProtocolError(
                    f"JSON value contains a cycle at {current_path}",
                    code="protocol_invalid",
                )
            active_ids.add(marker)
            add(2, current_path)
            if len(current) > max_bytes:
                active_ids.discard(marker)
                raise BridgeProtocolError(
                    f"JSON array contains too many entries at {current_path}",
                    code="bounds_exceeded",
                    retryable=True,
                )
            for index, child in enumerate(current):
                if index:
                    add(1, current_path)
                walk(child, f"{current_path}[{index}]", depth + 1)
            active_ids.discard(marker)
            return
        raise BridgeProtocolError(
            f"JSON value has an unsupported type at {current_path}",
            code="protocol_invalid",
        )

    walk(value, path, 0)


def _validate_model_mapping(value: Any) -> dict[str, Any]:
    """Validate and normalize the wire-level ModelMappingVersion contract."""
    if not isinstance(value, Mapping):
        raise BridgeProtocolError(
            "model mapping must be an object",
            code="protocol_invalid",
        )
    _assert_bounded_safe(value, max_bytes=MAX_FRAME_BYTES)
    required_fields = {
        "version",
        "canonicalFamilies",
        "rules",
        "reviewStatus",
        "source",
        "createdAt",
    }
    optional_fields = {
        "reviewedAt",
        "reviewedBy",
        "changeKind",
        "validFrom",
        "validUntil",
        "publishedAt",
        "supersedesVersion",
        "correctionOfVersion",
        "provenance",
        "warnings",
    }
    unknown_fields = set(value) - required_fields - optional_fields
    if unknown_fields:
        raise BridgeProtocolError(
            "model mapping contains unsupported fields",
            code="protocol_invalid",
        )
    missing_fields = required_fields - set(value)
    if missing_fields:
        raise BridgeProtocolError(
            "model mapping is missing required fields",
            code="protocol_invalid",
        )
    normalized: dict[str, Any] = dict(value)
    for field_name in ("version", "source", "createdAt"):
        normalized[field_name] = _mapping_string(
            value[field_name],
            f"mapping.{field_name}",
        )
    review_status = value["reviewStatus"]
    if not isinstance(review_status, str) or review_status not in {
        "draft",
        "approved",
        "retired",
    }:
        raise BridgeProtocolError(
            "model mapping reviewStatus is invalid",
            code="protocol_invalid",
        )
    change_kind = value.get("changeKind")
    if change_kind is not None and (
        not isinstance(change_kind, str)
        or change_kind not in {
            "prospective",
            "historical_correction",
        }
    ):
        raise BridgeProtocolError(
            "model mapping changeKind is invalid",
            code="protocol_invalid",
        )
    normalized["changeKind"] = change_kind
    normalized_families = _validate_mapping_families(value["canonicalFamilies"])
    normalized["canonicalFamilies"] = normalized_families
    normalized["rules"] = _validate_mapping_rules(
        value["rules"],
        normalized_families,
    )
    _normalize_mapping_optional_strings(value, normalized)
    provenance = value.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise BridgeProtocolError(
            "model mapping provenance must be an object",
            code="protocol_invalid",
        )
    normalized["provenance"] = dict(provenance)
    warnings = value.get("warnings", [])
    if (
        not isinstance(warnings, Sequence)
        or isinstance(warnings, (str, bytes))
        or len(warnings) > 64
        or any(not isinstance(item, str) for item in warnings)
    ):
        raise BridgeProtocolError(
            "model mapping warnings are invalid",
            code="protocol_invalid",
        )
    normalized["warnings"] = list(warnings)
    try:
        _encode_bounded_frame(normalized, max_bytes=(64 * 1024) + 1)
    except BridgeProtocolError as exc:
        if exc.code == "bounds_exceeded":
            raise BridgeProtocolError(
                "model mapping exceeds the supported bound",
                code="bounds_exceeded",
                retryable=True,
            ) from exc
        raise
    return normalized


def _validate_mapping_families(value: Any) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or len(value) > 64
    ):
        raise BridgeProtocolError(
            "model mapping canonicalFamilies is invalid",
            code="protocol_invalid",
        )
    normalized: list[str] = []
    for family in value:
        normalized_family = _mapping_string(family, "mapping.canonicalFamilies[]")
        if normalized_family in normalized:
            raise BridgeProtocolError(
                "model mapping canonicalFamilies contains duplicates",
                code="protocol_invalid",
            )
        normalized.append(normalized_family)
    return normalized


def _validate_mapping_rules(
    value: Any,
    families: Sequence[str],
) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) > 256
    ):
        raise BridgeProtocolError(
            "model mapping rules is invalid",
            code="protocol_invalid",
        )
    allowed_fields = {
        "slug",
        "mode",
        "reasoningEffort",
        "family",
        "collectorAccountId",
        "reviewed",
        "source",
    }
    normalized_rules: list[dict[str, Any]] = []
    for index, raw_rule in enumerate(value):
        if not isinstance(raw_rule, Mapping):
            raise BridgeProtocolError(
                f"model mapping rule {index} is not an object",
                code="protocol_invalid",
            )
        if set(raw_rule) - allowed_fields:
            raise BridgeProtocolError(
                f"model mapping rule {index} contains unsupported fields",
                code="protocol_invalid",
            )
        if "slug" not in raw_rule or "family" not in raw_rule:
            raise BridgeProtocolError(
                f"model mapping rule {index} is missing identity",
                code="protocol_invalid",
            )
        rule = dict(raw_rule)
        rule["slug"] = _mapping_string(
            raw_rule["slug"],
            f"mapping.rules[{index}].slug",
        )
        rule["family"] = _mapping_string(
            raw_rule["family"],
            f"mapping.rules[{index}].family",
        )
        if rule["family"] not in families:
            raise BridgeProtocolError(
                f"model mapping rule {index} names an unknown family",
                code="protocol_invalid",
            )
        for field_name in ("mode", "reasoningEffort", "collectorAccountId", "source"):
            if field_name in raw_rule and raw_rule[field_name] is not None:
                rule[field_name] = _mapping_string(
                    raw_rule[field_name],
                    f"mapping.rules[{index}].{field_name}",
                )
        if "reviewed" in raw_rule and not isinstance(raw_rule["reviewed"], bool):
            raise BridgeProtocolError(
                f"model mapping rule {index} reviewed flag is invalid",
                code="protocol_invalid",
            )
        normalized_rules.append(rule)
    return normalized_rules


def _normalize_mapping_optional_strings(
    value: Mapping[str, Any],
    normalized: dict[str, Any],
) -> None:
    for field_name in (
        "reviewedAt",
        "reviewedBy",
        "validFrom",
        "validUntil",
        "publishedAt",
        "supersedesVersion",
        "correctionOfVersion",
    ):
        if field_name in value and value[field_name] is not None:
            normalized[field_name] = _mapping_string(
                value[field_name],
                f"mapping.{field_name}",
            )


def _mapping_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise BridgeProtocolError(
            f"{field_name} must be a string",
            code="protocol_invalid",
        )
    normalized = value.strip()
    if not normalized or len(normalized.encode("utf-8")) > 512:
        raise BridgeProtocolError(
            f"{field_name} is outside the supported bound",
            code="bounds_exceeded",
            retryable=True,
        )
    return normalized


def _bounded_sequence(
    value: Any,
    *,
    field_name: str,
    limit: int = MAX_REQUESTS,
) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BridgeProtocolError(f"{field_name} must be an array")
    if len(value) > limit:
        raise BridgeProtocolError(
            f"{field_name} exceeds the supported bound",
            code="bounds_exceeded",
            retryable=True,
        )
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise BridgeProtocolError(f"{field_name} contains a non-object")
        result.append(item)
    return tuple(result)


def _validate_page_commit_payload(
    payload: Mapping[str, Any],
    *,
    expected_state_version: int,
) -> dict[str, Any]:
    _assert_bounded_safe(payload, max_bytes=MAX_FRAME_BYTES)
    raw_mutations = payload.get("mutations")
    mutations = dict(raw_mutations) if isinstance(raw_mutations, Mapping) else dict(payload)
    for field_name in (
        "expectedStateVersion",
        "kind",
        "source",
        "context",
        "discovery",
        "page",
        "attempts",
        "checkpointMutations",
        "candidateMutations",
        "coverageMutations",
        "ingestOperations",
        "operations",
    ):
        if field_name in payload and field_name not in mutations:
            mutations[field_name] = payload[field_name]
    page_commit_id = payload.get("pageCommitId", mutations.get("pageCommitId"))
    normalized_page_commit_id = _metadata_token(page_commit_id, "pageCommitId")
    nested_page_commit_id = mutations.get("pageCommitId")
    if nested_page_commit_id is not None and _metadata_token(
        nested_page_commit_id,
        "pageCommitId",
    ) != normalized_page_commit_id:
        raise BridgeProtocolError("page commit identity conflicts with its payload")
    nested_expected = mutations.get("expectedStateVersion")
    if nested_expected is not None and _required_state_version(nested_expected) != expected_state_version:
        raise BridgeProtocolError(
            "page commit state version conflicts with its payload",
            code="state_conflict",
            retryable=True,
        )
    source = mutations.get("source", mutations.get("context"))
    if not isinstance(source, Mapping):
        raise BridgeProtocolError("page commit source must be an object")
    for field_name in (
        "attempts",
        "candidateMutations",
        "coverageMutations",
        "ingestOperations",
        "operations",
    ):
        _bounded_sequence(
            mutations.get(field_name),
            field_name=f"page commit {field_name}",
        )
    checkpoint_mutation = mutations.get("checkpointMutations")
    if checkpoint_mutation is not None and not isinstance(
        checkpoint_mutation,
        Mapping,
    ):
        raise BridgeProtocolError("page commit checkpoint mutation must be an object")
    for field_name in ("discovery", "page"):
        value = mutations.get(field_name)
        if value is not None and not isinstance(value, Mapping):
            raise BridgeProtocolError(f"page commit {field_name} must be an object")
    result = dict(payload)
    result.setdefault("pageCommitId", normalized_page_commit_id)
    result["mutations"] = dict(mutations)
    result["mutations"].setdefault("expectedStateVersion", expected_state_version)
    return result


def _ok_response(request_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "ok": True,
        "result": dict(result),
    }


def _error_response(
    request_id: str,
    *,
    code: str,
    retryable: bool,
    coverage_incomplete: bool,
) -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "ok": False,
        "error": {
            "code": code,
            "retryable": retryable,
            "coverageIncomplete": coverage_incomplete,
        },
    }


def _failure_code(error: BaseException) -> str:
    if isinstance(error, BridgeProtocolError):
        return error.code
    text = str(error).lower()
    if "stale" in text or "fence" in text or "binding" in text:
        return "fence_invalid"
    if "bound" in text or "deadline" in text or "timeout" in text:
        return "bounds_exceeded"
    if "history" in text or "browser" in text:
        return "history_reader_failed"
    return "history_reader_failed"


def _retryable(error: BaseException) -> bool:
    if isinstance(error, BridgeProtocolError):
        return error.retryable
    return _failure_code(error) in {
        "bounds_exceeded",
        "history_reader_failed",
        "history_contract_unavailable",
        "state_conflict",
    }


def _failure_result(
    *,
    account: str,
    profile: str,
    run_id: str,
    error: BaseException,
    context: Optional[_RunContext] = None,
) -> dict[str, Any]:
    history_status = (
        context.history_status if context is not None else "unavailable"
    )
    history_reason = context.history_reason if context is not None else None
    retry_after_ms = context.retry_after_ms if context is not None else None
    state_version = context.state_version if context is not None else None
    requests = context.child.request_count if context is not None else 0
    return {
        "ok": False,
        "status": "failed",
        "collectorAccountId": account,
        "profileId": profile,
        "runId": run_id,
        "errorCode": _failure_code(error),
        "retryable": _retryable(error),
        "coverageIncomplete": True,
        "historyContract": history_status,
        "historyReason": history_reason,
        "retryAfterMs": retry_after_ms,
        "stateVersion": state_version,
        "requests": requests,
    }


def _with_cleanup_failure(
    result: Mapping[str, Any],
    cleanup_errors: Sequence[str],
) -> dict[str, Any]:
    """Make cleanup failure terminal instead of returning computed success."""
    updated = dict(result)
    unique_errors = tuple(dict.fromkeys(str(error) for error in cleanup_errors))
    updated.update(
        {
            "ok": False,
            "status": "failed",
            "errorCode": "cleanup_failed",
            "retryable": True,
            "coverageIncomplete": True,
            "cleanupErrors": unique_errors,
        }
    )
    return updated
