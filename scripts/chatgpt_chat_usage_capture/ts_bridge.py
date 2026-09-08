"""Parent-side bridge for the bounded TypeScript ChatGPT usage worker.

The Python side owns PostgreSQL fences, browser preparation, process
supervision, and the bounded NDJSON transport. TypeScript owns scheduling,
history collection, reconstruction, and report shaping.
"""

from __future__ import annotations

import errno
import json
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
from .privacy import assert_no_secrets, sanitize_token


PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_REQUESTS = 256
MAX_REQUEST_ID_LENGTH = 128
DEFAULT_CHILD_TTL_SECONDS = 3600
DEFAULT_WORKER_ROOT = "/app/scripts/chatgpt_chat_usage_capture/ts"
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


HistoryPreparer = Callable[
    [LedgerScope, str, str],
    Mapping[str, Any],
]
BindingBootstrapper = Callable[
    [LedgerScope, datetime],
    LedgerScope,
]


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
    mapping: Mapping[str, Any] = field(default_factory=dict)
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
        history_preparer: Optional[HistoryPreparer] = None,
        binding_bootstrapper: Optional[BindingBootstrapper] = None,
    ) -> "BridgeConfig":
        return cls(
            node_executable=node_executable,
            worker_root=worker_root,
            history_preparer=history_preparer,
            binding_bootstrapper=binding_bootstrapper,
        )

    @property
    def worker_entrypoint(self) -> str:
        return str(Path(self.worker_root) / "dist" / "src" / "worker" / "main.js")


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


@dataclass
class _Child:
    process: subprocess.Popen
    total_bytes: int
    request_count: int
    deadline_monotonic: float
    read_buffer: bytearray = field(default_factory=bytearray)
    ready_frames: Deque[bytes] = field(default_factory=deque)


@dataclass
class _RunContext:
    collector_account_id: str
    profile_id: str
    run_id: str
    scope: LedgerScope
    lease: CollectorLease
    child: _Child
    state_version: int
    trigger_id: Optional[str] = None
    prepared_history: Optional[Mapping[str, Any]] = None
    history_status: str = "unavailable"
    history_reason: Optional[str] = None
    retry_after_ms: Optional[int] = None
    finish_outcome: Optional[str] = None
    snapshot_ids: set[str] = field(default_factory=set)
    finished: bool = False
    cancelled: bool = False


class TsWorkerBridge:
    """Run one fenced TypeScript worker for one account/profile scope."""

    def __init__(self, state: PgCollectorState, config: BridgeConfig) -> None:
        self.state = state
        self.config = config
        self._children: dict[tuple[str, str], _Child] = {}
        self._lock = threading.Lock()

    def run_once(
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
        account = _metadata_token(collector_account_id, "collector_account_id")
        profile = _metadata_token(profile_id, "profile_id")
        requested_scope = _scope_for_account(scope, account)
        run_id = str(uuid4())
        lease: Optional[CollectorLease] = None
        child: Optional[_Child] = None
        context: Optional[_RunContext] = None
        observed_at = seen_at or datetime.now(timezone.utc)
        key = (account, profile)

        try:
            parent_scope, binding = self._capture_or_initialize_binding(
                requested_scope,
                seen_at=observed_at,
            )
            lease = self.state.claim_lease(
                collector_account_id=account,
                profile_id=profile,
                scope=parent_scope,
                ttl_seconds=self.config.child_ttl_seconds,
                expected_binding=binding,
            )
            header, _ = self.state.load_state(
                collector_account_id=account,
                profile_id=profile,
            )
            child = self._start_child(key)
            context = _RunContext(
                collector_account_id=account,
                profile_id=profile,
                run_id=run_id,
                scope=parent_scope,
                lease=lease,
                child=child,
                state_version=header.state_version if header is not None else 0,
            )
            with self._lock:
                self._children[key] = child
            self._send_start_run(
                context,
                schedule_options=schedule_options or self.config.schedule_options,
                collection_request=collection_request
                or self.config.collection_request,
                mapping=mapping or self.config.mapping,
                authentication_recovery_requested=authentication_recovery_requested,
            )
            result = self._serve_child(context)
            if not context.finished and not context.cancelled:
                raise BridgeProtocolError(
                    "worker ended without a committed finish",
                    code="history_reader_failed",
                    retryable=True,
                )
            return result
        except Exception as exc:
            if context is not None:
                context.cancelled = True
                self._send_parent_cancel(context, reason=_failure_code(exc))
                self._cancel_state(context, exc)
            elif lease is not None:
                self._release_lease_quietly(lease)
            return _failure_result(
                account=account,
                profile=profile,
                run_id=run_id,
                error=exc,
                context=context,
            )
        finally:
            if child is not None:
                self._reap(key=key, child=child)

    def _capture_or_initialize_binding(
        self,
        scope: LedgerScope,
        *,
        seen_at: datetime,
    ) -> tuple[LedgerScope, Any]:
        parent_scope: Optional[LedgerScope] = None
        try:
            parent_scope = _verified_scope_if_complete(scope)
        except BridgeProtocolError:
            pass
        if parent_scope is not None:
            try:
                binding = self.state.ledger.capture_binding(
                    parent_scope,
                    seen_at=seen_at,
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
        bootstrapped_scope = bootstrapper(scope, seen_at)
        if not isinstance(bootstrapped_scope, LedgerScope):
            raise BridgeProtocolError(
                "binding bootstrap did not return a verified scope",
                code="history_contract_unavailable",
                retryable=True,
            )
        parent_scope = _verified_scope(bootstrapped_scope, scope.collector_account_id)
        self.state.ledger.initialize_binding(parent_scope, seen_at=seen_at)
        return parent_scope, self.state.ledger.capture_binding(
            parent_scope,
            seen_at=seen_at,
        )

    def _start_child(self, key: tuple[str, str]) -> _Child:
        with self._lock:
            previous = self._children.get(key)
        if previous is not None:
            self._reap(key=key, child=previous)
        deadline = time.monotonic() + self.config.child_ttl_seconds
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
        if process.stdin is None or process.stdout is None:
            child = _Child(process, 0, 0, deadline)
            self._reap(key=key, child=child)
            raise BridgeProtocolError(
                "worker child streams are unavailable",
                code="history_reader_failed",
                retryable=True,
            )
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        return _Child(process, 0, 0, deadline)

    def _send_start_run(
        self,
        context: _RunContext,
        *,
        schedule_options: Mapping[str, Any],
        collection_request: Mapping[str, Any],
        mapping: Mapping[str, Any],
        authentication_recovery_requested: bool,
    ) -> None:
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
            "scheduleOptions": dict(schedule_options),
            "collectionRequest": dict(collection_request),
            "mapping": dict(mapping),
            "bounds": {
                "maxFrameBytes": self.config.max_frame_bytes,
                "maxTotalBytes": self.config.max_total_bytes,
                "maxRequests": self.config.max_requests,
                "remainingMs": _remaining_ms(context.child.deadline_monotonic),
            },
        }
        if authentication_recovery_requested:
            request["authenticationRecoveryRequested"] = True
        assert_no_secrets(request)
        self._write_frame(context.child, request)

    def _serve_child(self, context: _RunContext) -> dict[str, Any]:
        while not context.finished and not context.cancelled:
            message = self._read_message(context.child)
            if "control" in message:
                response = self._handle_control(context, message)
                if response is not None:
                    self._write_frame(context.child, response)
                continue
            response = self._handle_operation(context, message)
            self._write_frame(context.child, response)
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
            if context.prepared_history is None:
                return _error_response(
                    request_id,
                    code="history_contract_unavailable",
                    retryable=True,
                    coverage_incomplete=True,
                )
            return _ok_response(
                request_id,
                _wire_history_identity(context.prepared_history),
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
            self._check_deadline(context.child)
            context.child.request_count += 1
            if context.child.request_count > self.config.max_requests:
                raise BridgeProtocolError(
                    "worker request bound exceeded",
                    code="bounds_exceeded",
                    retryable=True,
                )
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
            assert_no_secrets(payload)

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
        header, _ = self.state.load_state(
            collector_account_id=context.collector_account_id,
            profile_id=context.profile_id,
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
            if isinstance(state, Mapping):
                self._observe_trigger(context, state)
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
            candidates = self.state.read_candidates(
                collector_account_id=context.collector_account_id,
                profile_id=context.profile_id,
                limit=limit,
                cursor=_optional_cursor(payload.get("cursor")),
                expected_state_version=expected,
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
        current, _ = self.state.load_state(
            collector_account_id=context.collector_account_id,
            profile_id=context.profile_id,
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
                self._observe_trigger(context, state)
            elif active is not None:
                raise BridgeProtocolError("schedule active trigger is invalid")
        header = self.state.compare_and_set_state(
            collector_account_id=context.collector_account_id,
            profile_id=context.profile_id,
            expected_state_version=expected_for_pg,
            schedule_transition=schedule,
            checkpoint=checkpoint,
            active_trigger=active_trigger,
            lease=context.lease,
            )
        context.state_version = header.state_version
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
        prepared = preparer(
            context.scope,
            context.collector_account_id,
            context.profile_id,
        )
        if not isinstance(prepared, Mapping):
            raise BridgeProtocolError(
                "history preparation did not return a record",
                code="history_contract_unavailable",
                retryable=True,
            )
        assert_no_secrets(prepared)
        verified = _validate_prepared_history(context.scope, prepared)
        context.prepared_history = prepared
        context.history_status = "ready"
        context.history_reason = None
        return verified

    def _verify_committed_trigger(
        self,
        context: _RunContext,
        trigger_id: str,
    ) -> None:
        header, _ = self.state.load_state(
            collector_account_id=context.collector_account_id,
            profile_id=context.profile_id,
        )
        if header is None:
            raise BridgeProtocolError(
                "no committed active trigger exists",
                code="fence_invalid",
            )
        candidates = [header.active_trigger, header.schedule_transition]
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            active = candidate.get("active", candidate)
            if isinstance(active, Mapping) and active.get("triggerId") == trigger_id:
                return
        raise BridgeProtocolError(
            "claimed trigger is not active in durable state",
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
        reader = context.prepared_history.get("readHistory")
        if not callable(reader):
            raise BridgeProtocolError(
                "native history reader is unavailable",
                code="history_contract_unavailable",
                retryable=True,
            )
        if any(key in payload for key in ("url", "endpoint", "headers", "credentials")):
            raise BridgeProtocolError("history request contains forbidden transport fields")
        result = reader(dict(payload))
        if not isinstance(result, Mapping):
            raise BridgeProtocolError(
                "native history reader returned an invalid result",
                code="history_reader_failed",
                retryable=True,
            )
        assert_no_secrets(result)
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
        result = self.state.load_conversation_metadata(
            collector_account_id=context.collector_account_id,
            conversation_id=conversation_id,
            limit=_bounded_limit(payload.get("limit"), default=32),
            cursor=_optional_cursor(payload.get("cursor")),
            snapshot_id=_optional_cursor(payload.get("snapshotId")),
            deadline_at=_wall_deadline(context.child),
        )
        assert_no_secrets(result)
        self._track_snapshot(context, payload, result)
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
        ack: PageAck = self.state.commit_history_page(
            lease=context.lease,
            scope=context.scope,
            run_id=context.run_id,
            page_commit_id=page_commit_id,
            canonical_payload=canonical_payload,
            expected_state_version=expected_version,
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
        result = self.state.load_report_snapshot(
            collector_account_id=context.collector_account_id,
            limit=_bounded_limit(payload.get("limit"), default=256),
            cursor=_optional_cursor(payload.get("cursor")),
            snapshot_id=_optional_cursor(payload.get("snapshotId")),
            deadline_at=_wall_deadline(context.child),
        )
        assert_no_secrets(result)
        self._track_snapshot(context, payload, result)
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
        header = self.state.finish_run(
            lease=context.lease,
            run_id=context.run_id,
            expected_state_version=expected,
            trigger_id=trigger_id,
            outcome=outcome,
            summary=summary,
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
        header = self.state.cancel(
            lease=context.lease,
            run_id=context.run_id,
            expected_state_version=expected,
            trigger_id=trigger_id,
            outcome=outcome,
            summary=summary,
        )
        context.state_version = header.state_version
        context.finish_outcome = outcome
        context.cancelled = True
        return {"cancelled": True, "stateVersion": header.state_version}

    def _observe_trigger(
        self,
        context: _RunContext,
        state: Mapping[str, Any],
    ) -> None:
        active = state.get("active")
        if isinstance(active, Mapping):
            trigger_id = active.get("triggerId")
            if isinstance(trigger_id, str) and trigger_id.strip():
                context.trigger_id = trigger_id

    def _await_child_exit(self, child: _Child) -> None:
        remaining = max(0.0, child.deadline_monotonic - time.monotonic())
        try:
            child.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise BridgeProtocolError(
                "worker did not terminate after committed finish",
                code="bounds_exceeded",
                retryable=True,
            ) from exc

    def _send_parent_cancel(self, context: _RunContext, *, reason: str) -> None:
        if context.child.process.poll() is not None:
            return
        message = {
            "protocolVersion": PROTOCOL_VERSION,
            "control": "cancelRun",
            "requestId": str(uuid4()),
            "runId": context.run_id,
            "collectorAccountId": context.collector_account_id,
            "profileId": context.profile_id,
            "bindingGeneration": context.lease.binding.binding_generation,
            "leaseFencingToken": context.lease.lease_fencing_token,
            "reason": _metadata_token(reason, "reason"),
        }
        try:
            self._write_frame(context.child, message)
        except Exception:
            return

    def _cancel_state(self, context: _RunContext, error: BaseException) -> None:
        summary = {
            "errorCode": _failure_code(error),
            "coverageIncomplete": True,
        }
        self._close_tracked_snapshots(context)
        try:
            self.state.cancel(
                lease=context.lease,
                run_id=context.run_id,
                expected_state_version=context.state_version,
                trigger_id=context.trigger_id,
                outcome="cancelled_after_start",
                summary=summary,
            )
        except Exception:
            self._release_lease_quietly(context.lease)

    def _track_snapshot(
        self,
        context: _RunContext,
        request_payload: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> None:
        previous = _optional_cursor(request_payload.get("snapshotId"))
        if previous is not None:
            context.snapshot_ids.discard(previous)
        next_snapshot = _optional_cursor(result.get("snapshotId"))
        if next_snapshot is not None:
            context.snapshot_ids.add(next_snapshot)

    def _close_tracked_snapshots(self, context: _RunContext) -> None:
        for snapshot_id in tuple(context.snapshot_ids):
            try:
                self.state.close_snapshot(snapshot_id)
            except Exception:
                pass
            finally:
                context.snapshot_ids.discard(snapshot_id)

    def _release_lease_quietly(self, lease: CollectorLease) -> None:
        try:
            self.state.release_lease(lease=lease)
        except Exception:
            return

    def _write_frame(self, child: _Child, message: Mapping[str, Any]) -> None:
        assert_no_secrets(message)
        encoded = (
            json.dumps(message, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > self.config.max_frame_bytes:
            raise BridgeProtocolError(
                "worker frame exceeds the configured bound",
                code="bounds_exceeded",
                retryable=True,
            )
        if child.total_bytes + len(encoded) > self.config.max_total_bytes:
            raise BridgeProtocolError(
                "worker total byte bound exceeded",
                code="bounds_exceeded",
                retryable=True,
            )
        if child.process.stdin is None:
            raise BridgeProtocolError(
                "worker stdin is unavailable",
                code="history_reader_failed",
                retryable=True,
            )
        fd = child.process.stdin.fileno()
        selector = selectors.DefaultSelector()
        view = memoryview(encoded)
        try:
            selector.register(fd, selectors.EVENT_WRITE)
            while view:
                self._check_deadline(child)
                events = selector.select(self._remaining_timeout(child))
                if not events:
                    raise BridgeProtocolError(
                        "worker write deadline exceeded",
                        code="bounds_exceeded",
                        retryable=True,
                    )
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
            child.total_bytes += len(encoded)
        finally:
            selector.close()

    def _read_message(self, child: _Child) -> Mapping[str, Any]:
        frame = self._read_frame(child)
        try:
            message = json.loads(frame.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeProtocolError("worker returned invalid JSON") from exc
        if not isinstance(message, Mapping):
            raise BridgeProtocolError("worker frame is not an object")
        assert_no_secrets(message)
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
                events = selector.select(self._remaining_timeout(child))
                if not events:
                    raise BridgeProtocolError(
                        "worker read deadline exceeded",
                        code="bounds_exceeded",
                        retryable=True,
                    )
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

    def _reap(self, *, key: tuple[str, str], child: _Child) -> None:
        process = child.process
        _terminate_process_group(process)
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    return
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            with self._lock:
                if self._children.get(key) is child:
                    self._children.pop(key, None)

    def close(self) -> None:
        """Boundedly reap every child still owned by this bridge."""
        with self._lock:
            children = list(self._children.items())
        for key, child in children:
            self._reap(key=key, child=child)


def _terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
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
    provider_user_id = _first_value(identity, "providerUserId", "provider_user_id")
    workspace_id = _first_value(identity, "workspaceId", "workspace_id")
    quota_owner_id = _first_value(identity, "quotaOwnerId", "quota_owner_id")
    auth_state = _first_value(identity, "authState", "auth_state")
    if (
        provider_user_id != parent_scope.provider_user_id
        or workspace_id != parent_scope.workspace_id
        or quota_owner_id != parent_scope.quota_owner_id
        or auth_state not in {"ready", "verified"}
    ):
        raise BridgeProtocolError(
            "history identity is not freshly verified",
            code="history_contract_unavailable",
            retryable=True,
        )
    if not _native_history_capability_is_verified(capabilities, manifest):
        raise BridgeProtocolError(
            "native history capability is unavailable",
            code="history_contract_unavailable",
            retryable=True,
        )
    return {
        "identity": dict(identity),
        "capabilities": dict(capabilities),
        "capabilityManifest": dict(manifest),
        "scope": _scope_wire(verified_scope),
    }


def _native_history_capability_is_verified(
    capabilities: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    values = [
        capabilities.get("nativeHistory"),
        capabilities.get("native_history"),
        capabilities.get("history"),
        manifest.get("nativeHistory"),
        manifest.get("native_history"),
        manifest.get("history"),
    ]
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("status") or value.get("state")
        if value in {"available", "verified", "supported", True}:
            return True
    return False


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


def _nested_value(payload: Mapping[str, Any], outer: str, inner: str) -> Any:
    nested = payload.get(outer)
    return nested.get(inner) if isinstance(nested, Mapping) else None


def _mapping_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeProtocolError("worker operation payload must be an object")
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
    return _metadata_token(value, "cursor")


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
    serialized = json.dumps(value, separators=(",", ":"), default=str)
    if len(serialized.encode("utf-8")) > 64 * 1024:
        raise BridgeProtocolError(
            "finish summary exceeds the supported bound",
            code="bounds_exceeded",
            retryable=True,
        )
    assert_no_secrets(value)
    return dict(value)


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
    assert_no_secrets(payload)
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
