"""PostgreSQL persistence contract for ChatGPT ordinary-Chat usage metadata."""

from __future__ import annotations

import hashlib
import json
import re
import selectors
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Iterable, Mapping, Optional, Sequence

import psycopg
from psycopg import pq, waiting

from .models import AttemptRecord
from .privacy import (
    assert_no_secrets,
    classify_surface,
    sanitize_metadata,
    sanitize_token,
)
from .timeutil import ensure_utc, isoformat_utc, parse_datetime


MIGRATION_PATH = Path(__file__).resolve().parent.parent / "apply_chatgpt_usage_ledger_2026_09_08.sql"
PROVENANCE_ALLOWLIST = frozenset(
    {
        "adapter_version",
        "collector",
        "collector_account_id",
        "coverage",
        "detail_route",
        "endpoint",
        "fixture",
        "page_kind",
        "pagination_state",
        "revisit_reason",
        "route",
        "schema_version",
        "scopes",
        "source",
        "source_id",
        "source_kind",
        "surface",
        "transfer_version",
        "transfer_schema_version",
        "run_id",
        "evidence_id",
        "schema_fingerprint",
        "unknown_fields",
        "warnings",
    }
)

TRANSFER_SCHEMA_VERSION = "chatgpt-chat-history-v1"
_DROP = object()
_ALLOWED_SURFACES = frozenset(
    {
        "chat",
        "work",
        "codex",
        "deep_research",
        "agent_mode",
        "voice",
        "image_generation",
        "unknown",
    }
)
_ALLOWED_ORIGINS = frozenset({"shared", "imported", "copied", "local", "user", "unknown"})
_ALLOWED_IDENTITY_BASES = frozenset({"generation", "request", "provisional", "unresolved", "unknown"})
_ALLOWED_TIME_BASES = frozenset(
    {"provider", "dispatch", "user_message", "response_observed", "bounded_interval", "unknown"}
)
_ALLOWED_OUTCOMES = frozenset(
    {
        "completed",
        "failed_after_start",
        "cancelled_after_start",
        "rejected_after_start",
        "completion_unknown",
        "rejected_before_start",
        "unresolved",
        "unknown",
    }
)
_ALLOWED_ALIAS_KINDS = frozenset({"generation", "request", "prompt", "message", "branch"})
_STRONG_ALIAS_KINDS = frozenset({"generation", "message", "branch"})
_ALLOWED_GAP_STATES = frozenset({"open", "resolved", "unknown"})
_ALLOWED_QUARANTINE_STATES = frozenset({"clear", "quarantined", "unknown"})
_OBSERVATION_KEY_ALIASES = {
    "conversationId": "conversation_id",
    "continuation": "continuation",
    "schemaVersion": "schema_version",
    "paginationState": "pagination_state",
    "detailRoute": "detail_route",
    "coverage": "coverage",
    "exhausted": "exhausted",
    "messageId": "message_id",
    "nodeId": "node_id",
    "parentId": "parent_id",
    "currentNode": "current_node",
    "modelSlug": "model_slug",
    "requestedModel": "requested_model",
    "requestedModelRaw": "requested_model_raw",
    "requestedMode": "requested_mode",
    "requestedModeRaw": "requested_mode_raw",
    "reasoningEffort": "reasoning_effort",
    "requestedReasoningEffortRaw": "requested_reasoning_effort_raw",
    "defaultModelSlug": "default_model_slug",
    "recordedFinalModelRaw": "recorded_final_model_raw",
    "endTurn": "end_turn",
    "generationId": "generation_id",
    "requestId": "request_id",
    "messageRequestId": "message_request_id",
    "createdAt": "created_at",
    "updatedAt": "updated_at",
    "isArchived": "is_archived",
    "isStarred": "is_starred",
    "hasVersions": "has_versions",
    "hasPreviousPage": "has_previous_page",
    "hasNextPage": "has_next_page",
    "startCursor": "start_cursor",
    "endCursor": "end_cursor",
    "nextCursor": "next_cursor",
    "previousCursor": "previous_cursor",
    "projectId": "project_id",
    "workspaceId": "workspace_id",
    "schemaFingerprint": "schema_fingerprint",
    "errorType": "error_type",
    "errorCode": "error_code",
    "quarantineState": "quarantine_state",
    "transferSchemaVersion": "transfer_version",
    "transferVersion": "transfer_version",
    "transfer_schema_version": "transfer_version",
    "resolvedModel": "resolved_model",
    "resolvedModelRaw": "resolved_model_raw",
    "resolvedModelSlug": "resolved_model_slug",
    "futureTimestampQuarantined": "future_timestamp_quarantined",
    "quarantineTimestamp": "quarantine_timestamp",
    "quarantineAt": "quarantine_at",
    "quarantinedAt": "quarantined_at",
    "quarantineWarning": "quarantine_warning",
    "quarantineWarnings": "quarantine_warnings",
    "quarantineReason": "quarantine_reason",
    "projectionTruncated": "projection_truncated",
    "projectionTruncations": "projection_truncations",
    "truncationReasons": "projection_truncations",
    "truncation_reasons": "projection_truncations",
    "projectionIncomplete": "projection_incomplete",
    "incomplete": "projection_incomplete",
    "evidenceId": "evidence_id",
    "unknownFields": "unknown_fields",
}
_OBSERVATION_FIELDS = frozenset(
    {
        "id",
        "conversation_id",
        "continuation",
        "exhausted",
        "message_id",
        "node_id",
        "parent",
        "parent_id",
        "children",
        "current_node",
        "model_slug",
        "requested_model",
        "requested_model_raw",
        "requested_model_slug",
        "requested_mode",
        "requested_mode_raw",
        "reasoning_effort",
        "requested_reasoning_effort_raw",
        "default_model_slug",
        "recorded_final_model_raw",
        "resolved_model",
        "resolved_model_raw",
        "resolved_model_slug",
        "generation_id",
        "request_id",
        "message_request_id",
        "created_at",
        "updated_at",
        "create_time",
        "update_time",
        "timestamp_",
        "status",
        "role",
        "channel",
        "recipient",
        "end_turn",
        "weight",
        "is_archived",
        "is_starred",
        "has_versions",
        "has_previous_page",
        "has_next_page",
        "start_cursor",
        "end_cursor",
        "next_cursor",
        "previous_cursor",
        "offset",
        "limit",
        "total",
        "items",
        "messages",
        "mapping",
        "page_info",
        "metadata",
        "author",
        "workspace_id",
        "project_id",
        "gizmo_id",
        "surface",
        "origin",
        "shared",
        "imported",
        "copied",
        "coverage",
        "pagination_state",
        "detail_route",
        "schema_version",
        "error_type",
        "error_code",
        "quarantine",
        "quarantine_state",
        "future_timestamp_quarantined",
        "quarantine_timestamp",
        "quarantine_at",
        "quarantined_at",
        "quarantine_warning",
        "quarantine_warnings",
        "quarantine_reason",
        "warnings",
        "schema_fingerprint",
        "unknown_fields",
        "evidence_id",
        "provenance",
        "message",
        "transfer_version",
        "transfer_schema_version",
        "projection_status",
        "projection_error",
        "projection_truncated",
        "projection_truncations",
        "projection_incomplete",
    }
)
_OBSERVATION_TOKEN_FIELDS = frozenset(
    {
        "id",
        "conversation_id",
        "message_id",
        "continuation",
        "node_id",
        "parent",
        "parent_id",
        "children",
        "current_node",
        "model_slug",
        "requested_model",
        "requested_model_raw",
        "requested_model_slug",
        "requested_mode",
        "requested_mode_raw",
        "reasoning_effort",
        "requested_reasoning_effort_raw",
        "default_model_slug",
        "recorded_final_model_raw",
        "resolved_model",
        "resolved_model_raw",
        "resolved_model_slug",
        "generation_id",
        "request_id",
        "message_request_id",
        "created_at",
        "updated_at",
        "create_time",
        "update_time",
        "status",
        "role",
        "channel",
        "recipient",
        "start_cursor",
        "end_cursor",
        "next_cursor",
        "previous_cursor",
        "workspace_id",
        "project_id",
        "gizmo_id",
        "coverage",
        "pagination_state",
        "detail_route",
        "schema_version",
        "error_type",
        "error_code",
        "quarantine_state",
        "quarantine_reason",
        "quarantine_warning",
        "schema_fingerprint",
        "evidence_id",
        "transfer_version",
        "projection_status",
        "projection_error",
        "projection_truncated",
        "projection_incomplete",
    }
)
_OBSERVATION_TIMESTAMP_FIELDS = frozenset(
    {
        "created_at",
        "updated_at",
        "create_time",
        "update_time",
        "quarantine_timestamp",
        "quarantine_at",
        "quarantined_at",
    }
)
_OBSERVATION_BOOLEAN_FIELDS = frozenset(
    {
        "exhausted",
        "end_turn",
        "is_archived",
        "is_starred",
        "has_versions",
        "has_previous_page",
        "has_next_page",
        "shared",
        "imported",
        "copied",
        "future_timestamp_quarantined",
        "projection_incomplete",
    }
)
_OBSERVATION_NUMBER_FIELDS = frozenset({"offset", "limit", "total", "weight", "timestamp_"})
_OBSERVATION_LIST_FIELDS = frozenset({"children", "unknown_fields"})
_OBSERVATION_COLLECTION_FIELDS = frozenset({"items", "messages", "mapping"})
_OBSERVATION_WARNING_FIELDS = frozenset({"warnings", "quarantine_warnings"})
_OBSERVATION_MAX_DEPTH = 8
_OBSERVATION_MAX_FIELDS = 128
_OBSERVATION_MAX_ITEMS = 800
_OBSERVATION_NULLABLE_BOOLEAN_FIELDS = frozenset({"end_turn", "is_archived", "is_starred", "has_versions"})
_QUARANTINE_TIMESTAMP_FIELDS = frozenset(
    {"createdAt", "updatedAt", "attemptTime", "earliestPossibleAt", "latestPossibleAt"}
)

_SUMMARY_FIELDS = frozenset(
    {
        "id",
        "conversation_id",
        "created_at",
        "updated_at",
        "create_time",
        "update_time",
        "is_archived",
        "workspace_id",
        "project_id",
        "gizmo_id",
        "surface",
        "origin",
        "has_versions",
        "current_node",
        "coverage",
    }
)
_MESSAGE_FIELDS = frozenset(
    {
        "id",
        "conversation_id",
        "message_id",
        "node_id",
        "parent",
        "parent_id",
        "children",
        "author",
        "role",
        "channel",
        "created_at",
        "updated_at",
        "create_time",
        "update_time",
        "status",
        "end_turn",
        "requested_model",
        "requested_model_raw",
        "requested_model_slug",
        "requested_mode",
        "requested_mode_raw",
        "reasoning_effort",
        "requested_reasoning_effort_raw",
        "recorded_final_model_raw",
        "model_slug",
        "resolved_model",
        "resolved_model_raw",
        "resolved_model_slug",
        "generation_id",
        "request_id",
        "message_request_id",
        "surface",
        "origin",
        "metadata",
        "coverage",
        "quarantine",
        "quarantine_state",
        "warnings",
    }
)
_MAPPING_NODE_FIELDS = _MESSAGE_FIELDS | frozenset({"message"})
_PAGE_INFO_FIELDS = frozenset(
    {
        "has_previous_page",
        "has_next_page",
        "start_cursor",
        "end_cursor",
        "next_cursor",
        "previous_cursor",
        "offset",
        "limit",
        "total",
    }
)
_PAGE_FIELDS = frozenset(
    {
        "items",
        "messages",
        "mapping",
        "page_info",
        "continuation",
        "exhausted",
        "schema_version",
        "pagination_state",
        "coverage",
        "warnings",
        "surface",
        "origin",
        "detail_route",
        "conversation_id",
        "current_node",
        "workspace_id",
        "project_id",
        "gizmo_id",
        "created_at",
        "updated_at",
        "create_time",
        "update_time",
        "quarantine",
        "quarantine_state",
        "transfer_version",
        "transfer_schema_version",
        "schema_fingerprint",
        "unknown_fields",
        "evidence_id",
        "provenance",
        "projection_status",
        "projection_error",
        "projection_truncated",
        "projection_truncations",
        "projection_incomplete",
    }
)


@dataclass
class _ObservationProjectionState:
    """Bounded state shared by all shape-specific observation projectors."""

    truncations: list[dict[str, Any]] = field(default_factory=list)
    invalid: bool = False
    incomplete: bool = False

    def mark_invalid(self) -> None:
        self.invalid = True

    def mark_incomplete(self) -> None:
        self.incomplete = True

    def mark_truncated(
        self,
        *,
        path: str,
        reason: str,
        retained_count: Optional[int] = None,
        source_count: Optional[int] = None,
        source_count_lower_bound: bool = False,
    ) -> None:
        record: dict[str, Any] = {
            "path": path or "$",
            "reason": reason,
        }
        if retained_count is not None:
            record["retained_count"] = retained_count
        if source_count is not None:
            record["source_count"] = source_count
        if source_count_lower_bound:
            record["source_count_lower_bound"] = True
        self.truncations.append(record)


@dataclass(frozen=True)
class LedgerScope:
    collector_account_id: str
    provider: str
    provider_user_id: Optional[str]
    workspace_id: Optional[str]
    quota_owner_id: Optional[str]
    surface: str


@dataclass(frozen=True)
class LedgerBinding:
    collector_account_id: str
    scope_key: str
    binding_generation: int
    binding_state: str
    identity_state: str


@dataclass(frozen=True)
class IngestContext:
    run_id: str
    observed_at: datetime
    source_kind: str
    source_id: str
    schema_version: str
    provenance: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class AttemptUpsertResult:
    attempt_id: str
    status: str
    alias_conflicts: int


@dataclass(frozen=True)
class AliasMergeResult:
    conflicts: int = 0
    quarantine_conflicts: int = 0


@dataclass(frozen=True)
class AttemptRef:
    scope_key: str
    attempt_id: str


@dataclass(frozen=True)
class AttemptIdentityResolution:
    canonical_ref: AttemptRef
    conflicts: int = 0
    quarantine_reason: Optional[str] = None
    merge_refs: tuple[AttemptRef, ...] = ()


@dataclass(frozen=True)
class UsageCounts:
    total: int
    completed: int
    ambiguous: int = 0
    unknown_time: int = 0
    unknown_identity: int = 0
    unknown_surface: int = 0
    unknown_origin: int = 0
    unknown_model: int = 0
    excluded_surface: int = 0
    excluded_origin: int = 0
    excluded_non_generation: int = 0
    uncertain_outcome: int = 0
    observed_model_mismatches: int = 0
    by_requested_family: dict[str, int] = field(default_factory=dict)
    by_recorded_final_family: dict[str, int] = field(default_factory=dict)
    by_resolved_family: dict[str, int] = field(default_factory=dict)
    by_requested_model_raw: dict[str, int] = field(default_factory=dict)
    by_recorded_final_model_raw: dict[str, int] = field(default_factory=dict)
    by_resolved_model_raw: dict[str, int] = field(default_factory=dict)

    @property
    def unclassified_or_ambiguous(self) -> int:
        return self.ambiguous + self.unknown_time + self.unknown_identity + self.unknown_surface + self.unknown_model


class LedgerError(RuntimeError):
    pass


class _OperationCursor:
    """Cursor proxy that applies the caller-owned operation deadline to SQL."""

    def __init__(
        self,
        ledger: "PgLedger",
        cursor: psycopg.Cursor,
        operation: Any = None,
    ) -> None:
        self._ledger = ledger
        self._cursor = cursor
        self._operation = operation

    def execute(
        self,
        query: Any,
        params: Any = None,
        **kwargs: Any,
    ) -> "_OperationCursor":
        if self._operation is None:
            self._cursor.execute(query, params, **kwargs)
            return self
        deadline = _operation_deadline(self._operation, None)
        if deadline is None:
            raise LedgerError("collector database deadline is required")
        self._ledger.execute_with_deadline(
            self._cursor,
            query,
            params,
            deadline_at=deadline,
            operation=self._operation,
        )
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class PgLedger:
    """Source-only contract; database work happens only in explicit methods."""

    def __init__(
        self,
        dsn: str,
        *,
        application_name: str = "aawm-chatgpt-usage-ledger",
        lock_timeout_ms: int = 1000,
        statement_timeout_ms: int = 5000,
    ) -> None:
        self.dsn = dsn
        self.application_name = application_name
        self.lock_timeout_ms = lock_timeout_ms
        self.statement_timeout_ms = statement_timeout_ms

    def connect(
        self,
        *,
        deadline_at: Optional[datetime] = None,
        operation: Any = None,
    ) -> psycopg.Connection:
        """Open a connection, bounding establishment and session setup when requested."""
        deadline = _operation_deadline(operation, deadline_at)
        _check_operation(operation)
        conn = (
            self._connect_with_deadline(deadline, operation=operation)
            if deadline is not None
            else psycopg.connect(self.dsn)
        )
        try:
            if deadline is not None:
                with conn.cursor() as cur:
                    self.execute_with_deadline(
                        cur,
                        "SELECT set_config('application_name', %s, false)",
                        (self.application_name,),
                        deadline_at=deadline,
                        operation=operation,
                    )
                    remaining_ms = _remaining_deadline_ms(deadline, operation)
                    self.execute_with_deadline(
                        cur,
                        "SELECT set_config('lock_timeout', %s, true)",
                        (f"{min(self.lock_timeout_ms, remaining_ms)}ms",),
                        deadline_at=deadline,
                        operation=operation,
                    )
                    remaining_ms = _remaining_deadline_ms(deadline, operation)
                    self.execute_with_deadline(
                        cur,
                        "SELECT set_config('statement_timeout', %s, true)",
                        (f"{min(self.statement_timeout_ms, remaining_ms)}ms",),
                        deadline_at=deadline,
                        operation=operation,
                    )
                    _assert_before_deadline(deadline, operation)
            else:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT set_config('application_name', %s, false)",
                        (self.application_name,),
                    )
                    cur.execute(
                        "SELECT set_config('lock_timeout', %s, true)",
                        (f"{self.lock_timeout_ms}ms",),
                    )
                    cur.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        (f"{self.statement_timeout_ms}ms",),
                    )
            return conn
        except BaseException:
            conn.close()
            raise

    def execute_with_deadline(
        self,
        cursor: psycopg.Cursor,
        query: Any,
        params: Any = None,
        *,
        deadline_at: datetime,
        operation: Any = None,
    ) -> psycopg.Cursor:
        """Execute one cursor operation under the caller-owned absolute deadline."""
        deadline = _operation_deadline(operation, deadline_at)
        if deadline is None:
            raise LedgerError("collector database deadline is required")
        _check_operation(operation)
        connection = cursor.connection
        try:
            with connection.lock:
                _wait_operation(
                    cursor._execute_gen(query, params),
                    connection.pgconn.socket,
                    deadline,
                    operation=operation,
                )
        except LedgerError:
            connection.close()
            raise
        return cursor

    def rollback_with_deadline(
        self,
        connection: psycopg.Connection,
        *,
        deadline_at: datetime,
        operation: Any = None,
    ) -> None:
        """Rollback without allowing a stalled server response to outlive the deadline."""
        deadline = _operation_deadline(operation, deadline_at)
        if deadline is None:
            raise LedgerError("collector database deadline is required")
        _check_operation(operation)
        try:
            with connection.lock:
                _wait_operation(
                    connection._rollback_gen(),
                    connection.pgconn.socket,
                    deadline,
                    operation=operation,
                )
        except LedgerError:
            connection.close()
            raise

    def command_with_deadline(
        self,
        connection: psycopg.Connection,
        command: Any,
        *,
        deadline_at: datetime,
        operation: Any = None,
    ) -> None:
        """Run a protocol command without implicit transaction setup."""
        deadline = _operation_deadline(operation, deadline_at)
        if deadline is None:
            raise LedgerError("collector database deadline is required")
        _check_operation(operation)
        try:
            with connection.lock:
                _wait_operation(
                    connection._exec_command(command),
                    connection.pgconn.socket,
                    deadline,
                    operation=operation,
                )
        except LedgerError:
            connection.close()
            raise

    def commit_with_deadline(
        self,
        connection: psycopg.Connection,
        *,
        deadline_at: datetime,
        operation: Any = None,
    ) -> None:
        """Commit one owned transaction under the caller's absolute deadline."""
        deadline = _operation_deadline(operation, deadline_at)
        if deadline is None:
            raise LedgerError("collector database deadline is required")
        _check_operation(operation)
        try:
            with connection.lock:
                _wait_operation(
                    connection._commit_gen(),
                    connection.pgconn.socket,
                    deadline,
                    operation=operation,
                )
        except LedgerError:
            connection.close()
            raise

    @contextmanager
    def session(self, *, operation: Any = None) -> Iterator[psycopg.Connection]:
        """Own one connection and bound its commit or rollback cleanup."""
        deadline = _operation_deadline(operation, None)
        conn = self.connect(deadline_at=deadline, operation=operation)
        try:
            yield conn
        except BaseException:
            try:
                if (
                    deadline is not None
                    and _remaining_deadline_seconds(deadline, operation) > 0
                ):
                    try:
                        self.rollback_with_deadline(
                            conn,
                            deadline_at=deadline,
                            operation=operation,
                        )
                    except BaseException:
                        pass
            finally:
                conn.close()
            raise
        else:
            try:
                if deadline is not None:
                    self.commit_with_deadline(
                        conn,
                        deadline_at=deadline,
                        operation=operation,
                    )
                else:
                    conn.commit()
            finally:
                conn.close()

    @contextmanager
    def cursor(
        self,
        connection: psycopg.Connection,
        *,
        operation: Any = None,
    ) -> Iterator[_OperationCursor]:
        with connection.cursor() as cursor:
            yield _OperationCursor(self, cursor, operation)

    def _connect_with_deadline(
        self,
        deadline_at: datetime,
        *,
        operation: Any = None,
    ) -> psycopg.Connection:
        """Create one libpq connection while retaining ownership through timeout."""
        deadline = _operation_deadline(operation, deadline_at)
        if deadline is None:
            raise LedgerError("collector database deadline is required")
        wait_deadline = _operation_wait_deadline(deadline, operation)
        pgconn: Any = None
        try:
            pgconn = pq.PGconn.connect_start(self.dsn.encode("utf-8"))
            while True:
                _check_operation(operation)
                status = pq.PollingStatus(pgconn.connect_poll())
                if monotonic() >= wait_deadline:
                    raise LedgerError("collector database deadline has expired")
                if status == pq.PollingStatus.OK:
                    pgconn.nonblocking = 1
                    connection = psycopg.Connection(pgconn)
                    pgconn = None
                    if monotonic() >= wait_deadline:
                        connection.close()
                        raise LedgerError("collector database deadline has expired")
                    return connection
                if status == pq.PollingStatus.FAILED:
                    message = pgconn.get_error_message(pgconn._encoding)
                    finished = psycopg.errors.finish_pgconn(pgconn)
                    pgconn = None
                    raise psycopg.errors.OperationalError(
                        f"connection failed: {message}",
                        pgconn=finished,
                    )
                if status == pq.PollingStatus.READING:
                    events = selectors.EVENT_READ
                elif status == pq.PollingStatus.WRITING:
                    events = selectors.EVENT_WRITE
                else:
                    message = pgconn.get_error_message(pgconn._encoding)
                    finished = psycopg.errors.finish_pgconn(pgconn)
                    pgconn = None
                    raise psycopg.errors.OperationalError(
                        f"connection failed: {message}",
                        pgconn=finished,
                    )
                remaining = wait_deadline - monotonic()
                if remaining <= 0:
                    raise LedgerError("collector database deadline has expired")
                with selectors.DefaultSelector() as selector:
                    selector.register(pgconn.socket, events)
                    if not selector.select(timeout=remaining):
                        raise LedgerError("collector database deadline has expired")
        finally:
            if pgconn is not None:
                pgconn.finish()

    def ensure_schema(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        try:
            with self.connect() as conn, conn.cursor() as cur:
                for statement in _split_sql_statements(sql):
                    cur.execute(statement)
        except (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled) as exc:
            raise LedgerError(exc.__class__.__name__) from exc

    @contextmanager
    def transaction(
        self,
        *,
        scope: Optional[LedgerScope] = None,
        seen_at: Optional[datetime] = None,
        expected_binding: Optional[LedgerBinding] = None,
        operation: Any = None,
    ) -> Iterator["PgLedgerPage"]:
        """Open one page write transaction; provider/network work stays outside."""
        if scope is not None and expected_binding is None:
            raise LedgerError("expected_binding is required for scoped writes")
        safe_scope = _normalize_scope(scope) if scope is not None else None
        safe_seen_at = _utc_datetime(seen_at, "seen_at") if seen_at is not None else None
        with self.session(operation=operation) as conn:
            page = PgLedgerPage(
                self,
                conn,
                expected_binding=expected_binding,
                operation=operation,
            )
            if safe_scope is not None:
                page.bind_scope(
                    safe_scope,
                    seen_at=safe_seen_at or datetime.now().astimezone(),
                    expected_binding=expected_binding,
                )
            yield page

    page_transaction = transaction

    def capture_binding(
        self,
        scope: LedgerScope,
        *,
        seen_at: datetime,
        operation: Any = None,
    ) -> LedgerBinding:
        """Capture the current binding fence before an external/network read."""
        safe_scope = _normalize_scope(scope)
        _utc_datetime(seen_at, "seen_at")
        key = scope_key(safe_scope)
        with self.session(operation=operation) as conn, self.cursor(
            conn,
            operation=operation,
        ) as cur:
            cur.execute(
                """
                SELECT scope_key, binding_generation, binding_state
                FROM public.chatgpt_usage_scope_bindings
                WHERE collector_account_id = %s
                  AND binding_state = 'active'
                FOR SHARE
                """,
                (safe_scope.collector_account_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise LedgerError("scope binding is not initialized")
        if str(row[0]) != key:
            raise LedgerError("scope does not match the active binding; refine it explicitly")
        return LedgerBinding(
            collector_account_id=safe_scope.collector_account_id,
            scope_key=str(row[0]),
            binding_generation=int(row[1]),
            binding_state=str(row[2]),
            identity_state=_scope_identity_state(safe_scope),
        )

    def initialize_binding(
        self,
        scope: LedgerScope,
        *,
        seen_at: datetime,
        operation: Any = None,
    ) -> LedgerBinding:
        """Create the first binding for a collector, or refresh that same scope."""
        safe_scope = _normalize_scope(scope)
        safe_seen_at = _utc_datetime(seen_at, "seen_at")
        with self.transaction(operation=operation) as page:
            return page.bind_scope(
                safe_scope,
                seen_at=safe_seen_at,
                allow_initialize=True,
            )

    def refine_binding(
        self,
        scope: LedgerScope,
        *,
        expected_binding: LedgerBinding,
        seen_at: datetime,
        operation: Any = None,
    ) -> LedgerBinding:
        """Explicitly refine a fenced binding to a newly verified scope."""
        safe_scope = _normalize_scope(scope)
        safe_seen_at = _utc_datetime(seen_at, "seen_at")
        with self.transaction(
            expected_binding=expected_binding,
            operation=operation,
        ) as page:
            return page.bind_scope(
                safe_scope,
                seen_at=safe_seen_at,
                expected_binding=expected_binding,
                allow_rebind=True,
            )

    def upsert_scope(self, scope: LedgerScope, *, seen_at: datetime) -> str:
        return self.initialize_binding(scope, seen_at=seen_at).scope_key

    def record_observation(
        self,
        scope: LedgerScope,
        context: IngestContext,
        payload: Mapping[str, Any],
        *,
        expected_binding: LedgerBinding,
    ) -> tuple[str, bool]:
        safe_scope = _normalize_scope(scope)
        safe_context = _normalize_context(context)
        with self.transaction(
            scope=safe_scope,
            seen_at=safe_context.observed_at,
            expected_binding=expected_binding,
        ) as page:
            return page.record_observation(safe_scope, safe_context, payload)

    def upsert_attempt(
        self,
        scope: LedgerScope,
        attempt: AttemptRecord,
        context: IngestContext,
        *,
        expected_binding: LedgerBinding,
    ) -> AttemptUpsertResult:
        safe_scope = _normalize_scope(scope)
        safe_context = _normalize_context(context)
        with self.transaction(
            scope=safe_scope,
            seen_at=safe_context.observed_at,
            expected_binding=expected_binding,
        ) as page:
            return page.upsert_attempt(safe_scope, attempt, safe_context)

    def record_coverage_gap(
        self,
        scope: LedgerScope,
        *,
        source_kind: str,
        source_id: str,
        reason: str,
        state: str = "open",
        details: Optional[Mapping[str, Any]] = None,
        seen_at: datetime,
        expected_binding: LedgerBinding,
    ) -> str:
        safe_scope = _normalize_scope(scope)
        safe_seen_at = _utc_datetime(seen_at, "seen_at")
        with self.transaction(
            scope=safe_scope,
            seen_at=safe_seen_at,
            expected_binding=expected_binding,
        ) as page:
            return page.record_coverage_gap(
                safe_scope,
                source_kind=source_kind,
                source_id=source_id,
                reason=reason,
                state=state,
                details=details,
                seen_at=safe_seen_at,
            )

    def resolve_coverage_gaps(
        self,
        scope: LedgerScope,
        *,
        source_kind: str,
        source_id: str,
        seen_at: datetime,
        expected_binding: LedgerBinding,
    ) -> None:
        safe_scope = _normalize_scope(scope)
        safe_seen_at = _utc_datetime(seen_at, "seen_at")
        with self.transaction(
            scope=safe_scope,
            seen_at=safe_seen_at,
            expected_binding=expected_binding,
        ) as page:
            page.resolve_coverage_gaps(
                safe_scope,
                source_kind=source_kind,
                source_id=source_id,
                seen_at=safe_seen_at,
            )

    def scope_keys_for_account(self, collector_account_id: str) -> tuple[str, ...]:
        """Return active and retired scope keys bound to one collector."""
        account_token = _required_token(collector_account_id, "collector_account_id")
        self.assert_safe_record({"collector_account_id": account_token})
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE scope_tree(scope_key) AS (
                    SELECT scope_key
                    FROM public.chatgpt_usage_scope_bindings
                    WHERE collector_account_id = %s
                      AND binding_state = 'active'
                    UNION
                    SELECT r.retired_scope_key
                    FROM public.chatgpt_usage_scope_redirects AS r
                    JOIN scope_tree AS s ON s.scope_key = r.canonical_scope_key
                )
                SELECT scope_key FROM scope_tree ORDER BY scope_key
                """,
                (account_token,),
            )
            return tuple(row[0] for row in cur.fetchall())

    def count_attempts(
        self,
        account: str,
        *,
        model_family: Optional[str] = None,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
    ) -> UsageCounts:
        account_token = _required_token(account, "account")
        family_token = _required_token(model_family, "model_family") if model_family is not None else None
        start = _utc_datetime(window_start, "window_start") if window_start is not None else None
        end = _utc_datetime(window_end, "window_end") if window_end is not None else None
        if start is not None and end is not None and start >= end:
            raise LedgerError("window_start must be before window_end")
        time_state_sql, time_params = _time_state_sql(start, end)
        model_clause = ""
        params: list[Any] = [account_token]
        params.extend(time_params)
        if family_token is not None:
            model_clause = """
                WHERE (
                    requested_family = %s
                    OR recorded_final_family = %s
                    OR resolved_family = %s
                )
            """
            params.extend((family_token, family_token, family_token))
        query = f"""
            WITH RECURSIVE scope_chain AS (
                SELECT
                    scope_key AS stored_scope_key,
                    scope_key AS canonical_scope_key
                FROM public.chatgpt_usage_scopes
                UNION
                SELECT
                    chain.stored_scope_key,
                    redirects.canonical_scope_key
                FROM scope_chain AS chain
                JOIN public.chatgpt_usage_scope_redirects AS redirects
                  ON redirects.retired_scope_key = chain.canonical_scope_key
            ),
            canonical_scope_map AS (
                SELECT DISTINCT ON (chain.stored_scope_key)
                    chain.stored_scope_key,
                    chain.canonical_scope_key
                FROM scope_chain AS chain
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM public.chatgpt_usage_scope_redirects AS redirects
                    WHERE redirects.retired_scope_key = chain.canonical_scope_key
                )
                ORDER BY chain.stored_scope_key, chain.canonical_scope_key
            ),
            scoped AS (
                SELECT
                    attempts.*,
                    scope_map.canonical_scope_key,
                    canonical_scope.provider_user_id AS canonical_provider_user_id,
                    canonical_scope.workspace_id AS canonical_workspace_id,
                    canonical_scope.quota_owner_id AS canonical_quota_owner_id,
                    canonical_scope.identity_state AS canonical_identity_state,
                    canonical_scope.surface AS canonical_surface
                FROM public.chatgpt_usage_attempts AS attempts
                JOIN canonical_scope_map AS scope_map
                  ON scope_map.stored_scope_key = attempts.scope_key
                JOIN public.chatgpt_usage_scopes AS canonical_scope
                  ON canonical_scope.scope_key = scope_map.canonical_scope_key
                JOIN public.chatgpt_usage_scope_bindings AS bindings
                    ON bindings.scope_key = canonical_scope.scope_key
                 AND bindings.collector_account_id = %s
                 AND bindings.binding_state = 'active'
                WHERE NOT attempts.tombstone
            ),
            alias_rows AS (
                SELECT
                    scoped.canonical_scope_key,
                    scoped.scope_key,
                    scoped.attempt_id,
                    aliases.alias_kind,
                    aliases.alias_value
                FROM scoped
                JOIN public.chatgpt_usage_attempt_aliases AS aliases
                  ON aliases.scope_key = scoped.scope_key
                 AND aliases.attempt_id = scoped.attempt_id
            ),
            alias_values AS (
                SELECT
                    scoped.canonical_scope_key,
                    scoped.scope_key,
                    scoped.attempt_id,
                    string_agg(
                        alias_rows.alias_value,
                        ',' ORDER BY alias_rows.alias_value
                    ) FILTER (WHERE alias_rows.alias_kind = 'generation')
                        AS generation_aliases,
                    string_agg(
                        alias_rows.alias_value,
                        ',' ORDER BY alias_rows.alias_value
                    ) FILTER (WHERE alias_rows.alias_kind = 'message')
                        AS message_aliases,
                    string_agg(
                        alias_rows.alias_value,
                        ',' ORDER BY alias_rows.alias_value
                    ) FILTER (WHERE alias_rows.alias_kind = 'branch')
                        AS branch_aliases
                FROM scoped
                LEFT JOIN alias_rows
                  ON alias_rows.canonical_scope_key = scoped.canonical_scope_key
                 AND alias_rows.scope_key = scoped.scope_key
                 AND alias_rows.attempt_id = scoped.attempt_id
                GROUP BY
                    scoped.canonical_scope_key,
                    scoped.scope_key,
                    scoped.attempt_id
            ),
            generation_keys AS (
                SELECT
                    canonical_scope_key,
                    scope_key,
                    attempt_id,
                    'generation:' || generation_aliases AS generation_key
                FROM alias_values
                WHERE generation_aliases IS NOT NULL
            ),
            strong_edges AS (
                SELECT DISTINCT
                    left_alias.canonical_scope_key,
                    left_alias.scope_key AS left_scope_key,
                    left_alias.attempt_id AS left_attempt_id,
                    right_alias.scope_key AS right_scope_key,
                    right_alias.attempt_id AS right_attempt_id
                FROM alias_rows AS left_alias
                JOIN alias_rows AS right_alias
                  ON right_alias.canonical_scope_key = left_alias.canonical_scope_key
                 AND right_alias.alias_kind = left_alias.alias_kind
                 AND right_alias.alias_value = left_alias.alias_value
                WHERE left_alias.alias_kind IN ('message', 'branch')
                  AND (
                      left_alias.scope_key <> right_alias.scope_key
                      OR left_alias.attempt_id <> right_alias.attempt_id
                  )
            ),
            strong_reach (
                canonical_scope_key,
                node_scope_key,
                node_attempt_id,
                root_scope_key,
                root_attempt_id
            ) AS (
                SELECT
                    scoped.canonical_scope_key,
                    scoped.scope_key,
                    scoped.attempt_id,
                    scoped.scope_key,
                    scoped.attempt_id
                FROM scoped
                UNION
                SELECT
                    reach.canonical_scope_key,
                    CASE
                        WHEN edges.left_scope_key = reach.node_scope_key
                         AND edges.left_attempt_id = reach.node_attempt_id
                            THEN edges.right_scope_key
                        ELSE edges.left_scope_key
                    END,
                    CASE
                        WHEN edges.left_scope_key = reach.node_scope_key
                         AND edges.left_attempt_id = reach.node_attempt_id
                            THEN edges.right_attempt_id
                        ELSE edges.left_attempt_id
                    END,
                    reach.root_scope_key,
                    reach.root_attempt_id
                FROM strong_reach AS reach
                JOIN strong_edges AS edges
                  ON edges.canonical_scope_key = reach.canonical_scope_key
                 AND (
                     (
                         edges.left_scope_key = reach.node_scope_key
                         AND edges.left_attempt_id = reach.node_attempt_id
                     )
                     OR (
                         edges.right_scope_key = reach.node_scope_key
                         AND edges.right_attempt_id = reach.node_attempt_id
                     )
                 )
            ),
            strong_components AS (
                SELECT
                    canonical_scope_key,
                    node_scope_key AS scope_key,
                    node_attempt_id AS attempt_id,
                    MIN(
                        jsonb_build_array(root_scope_key, root_attempt_id)::text
                    ) AS component_key
                FROM strong_reach
                GROUP BY
                    canonical_scope_key,
                    node_scope_key,
                    node_attempt_id
            ),
            component_generation_keys AS (
                SELECT
                    components.canonical_scope_key,
                    components.component_key,
                    count(DISTINCT generations.generation_key)
                        AS generation_key_count,
                    MIN(generations.generation_key) AS generation_key
                FROM strong_components AS components
                LEFT JOIN strong_components AS members
                  ON members.canonical_scope_key = components.canonical_scope_key
                 AND members.component_key = components.component_key
                LEFT JOIN generation_keys AS generations
                  ON generations.canonical_scope_key = members.canonical_scope_key
                 AND generations.scope_key = members.scope_key
                 AND generations.attempt_id = members.attempt_id
                GROUP BY
                    components.canonical_scope_key,
                    components.component_key
            ),
            component_generation_count AS (
                SELECT
                    components.canonical_scope_key,
                    components.component_key,
                    count(DISTINCT generation_keys.generation_key) AS generation_key_count
                FROM strong_components AS components
                LEFT JOIN generation_keys
                  ON generation_keys.canonical_scope_key = components.canonical_scope_key
                 AND generation_keys.scope_key = components.scope_key
                 AND generation_keys.attempt_id = components.attempt_id
                GROUP BY
                    components.canonical_scope_key,
                    components.component_key
            ),
            identified AS (
                SELECT
                    scoped.*,
                    CASE
                        WHEN alias_values.generation_aliases IS NOT NULL
                            THEN 'generation:' || alias_values.generation_aliases
                        WHEN component_generation_keys.generation_key_count = 1
                            THEN component_generation_keys.generation_key
                        WHEN component_generation_keys.generation_key_count > 1
                            THEN 'attempt:' || scoped.scope_key || ':'
                                || scoped.attempt_id
                        WHEN component_generation_count.generation_key_count > 1
                            THEN 'attempt:' || scoped.scope_key || ':'
                                || scoped.attempt_id
                        WHEN alias_values.message_aliases IS NOT NULL
                          OR alias_values.branch_aliases IS NOT NULL
                            THEN 'strong:' || strong_components.component_key
                        ELSE 'attempt:' || scoped.scope_key || ':'
                            || scoped.attempt_id
                    END AS identity_key
                FROM scoped
                LEFT JOIN alias_values
                  ON alias_values.canonical_scope_key = scoped.canonical_scope_key
                 AND alias_values.scope_key = scoped.scope_key
                 AND alias_values.attempt_id = scoped.attempt_id
                JOIN strong_components
                  ON strong_components.canonical_scope_key = scoped.canonical_scope_key
                 AND strong_components.scope_key = scoped.scope_key
                 AND strong_components.attempt_id = scoped.attempt_id
                LEFT JOIN component_generation_keys
                  ON component_generation_keys.canonical_scope_key =
                         strong_components.canonical_scope_key
                 AND component_generation_keys.component_key =
                         strong_components.component_key
                LEFT JOIN component_generation_count
                  ON component_generation_count.canonical_scope_key =
                         strong_components.canonical_scope_key
                 AND component_generation_count.component_key =
                         strong_components.component_key
            ),
            canonicalized AS (
                SELECT DISTINCT ON (canonical_scope_key, identity_key)
                    identified.*
                FROM identified
                ORDER BY
                    canonical_scope_key,
                    identity_key,
                    observed_at DESC,
                    (COALESCE(quarantine_state, 'unknown') <> 'clear') DESC,
                    revision DESC,
                    last_seen_at DESC,
                    (scope_key = canonical_scope_key) DESC,
                    scope_key,
                    attempt_id
            ),
            classified AS (
                SELECT
                    canonicalized.*,
                    CASE
                        WHEN COALESCE(quarantine_state, 'unknown') <> 'clear'
                            THEN 'unknown_identity'
                        WHEN canonical_surface = 'unknown'
                          OR surface = 'unknown'
                            THEN 'unknown_surface'
                        WHEN canonical_surface <> 'chat'
                          OR surface <> 'chat'
                            THEN 'excluded_surface'
                        WHEN origin IN ('shared', 'imported', 'copied')
                            THEN 'excluded_origin'
                        WHEN outcome = 'rejected_before_start'
                            OR NOT (generation_started OR completed_answer)
                            THEN 'excluded_non_generation'
                        WHEN canonical_identity_state <> 'verified'
                          OR canonical_provider_user_id IS NULL
                          OR canonical_workspace_id IS NULL
                          OR canonical_quota_owner_id IS NULL
                          OR identity_basis IN ('unresolved', 'unknown')
                            THEN 'unknown_identity'
                        ELSE 'eligible'
                    END AS evidence_state,
                    {time_state_sql} AS time_state
                FROM canonicalized
            ),
            filtered AS (
                SELECT *
                FROM classified
                {model_clause}
            )
            SELECT
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state = 'definite'
                ) AS total,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state = 'definite'
                      AND completed_answer
                ) AS completed,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state = 'ambiguous'
                ) AS ambiguous,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state = 'unknown'
                ) AS unknown_time,
                count(*) FILTER (
                    WHERE evidence_state = 'unknown_identity'
                      AND time_state <> 'out'
                ) AS unknown_identity,
                count(*) FILTER (
                    WHERE evidence_state = 'unknown_surface'
                      AND time_state <> 'out'
                ) AS unknown_surface,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state <> 'out'
                      AND (origin IS NULL OR origin = 'unknown')
                ) AS unknown_origin,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state <> 'out'
                      AND COALESCE(
                          NULLIF(requested_family, 'unknown'),
                          NULLIF(recorded_final_family, 'unknown'),
                          NULLIF(resolved_family, 'unknown')
                      ) IS NULL
                ) AS unknown_model,
                count(*) FILTER (
                    WHERE evidence_state = 'excluded_surface'
                      AND time_state <> 'out'
                ) AS excluded_surface,
                count(*) FILTER (
                    WHERE evidence_state = 'excluded_origin'
                      AND time_state <> 'out'
                ) AS excluded_origin,
                count(*) FILTER (
                    WHERE evidence_state = 'excluded_non_generation'
                      AND time_state <> 'out'
                ) AS excluded_non_generation,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state = 'definite'
                      AND outcome IN (
                          'failed_after_start',
                          'cancelled_after_start',
                          'rejected_after_start',
                          'completion_unknown',
                          'unresolved',
                          'unknown'
                      )
                ) AS uncertain_outcome,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state = 'definite'
                      AND requested_family IS NOT NULL
                      AND recorded_final_family IS NOT NULL
                      AND requested_family <> recorded_final_family
                ) AS observed_model_mismatches,
                (
                    SELECT COALESCE(jsonb_object_agg(family, family_count),
                                    '{{}}'::jsonb)
                    FROM (
                        SELECT requested_family AS family, count(*) AS family_count
                        FROM filtered
                        WHERE evidence_state = 'eligible'
                          AND time_state = 'definite'
                          AND requested_family IS NOT NULL
                        GROUP BY requested_family
                    ) AS requested_families
                ) AS by_requested_family,
                (
                    SELECT COALESCE(jsonb_object_agg(family, family_count),
                                    '{{}}'::jsonb)
                    FROM (
                        SELECT recorded_final_family AS family, count(*) AS family_count
                        FROM filtered
                        WHERE evidence_state = 'eligible'
                          AND time_state = 'definite'
                          AND recorded_final_family IS NOT NULL
                        GROUP BY recorded_final_family
                    ) AS recorded_families
                ) AS by_recorded_final_family,
                (
                    SELECT COALESCE(jsonb_object_agg(family, family_count),
                                    '{{}}'::jsonb)
                    FROM (
                        SELECT resolved_family AS family, count(*) AS family_count
                        FROM filtered
                        WHERE evidence_state = 'eligible'
                          AND time_state = 'definite'
                          AND resolved_family IS NOT NULL
                        GROUP BY resolved_family
                    ) AS resolved_families
                ) AS by_resolved_family,
                (
                    SELECT COALESCE(jsonb_object_agg(model, model_count),
                                    '{{}}'::jsonb)
                    FROM (
                        SELECT requested_model_raw AS model, count(*) AS model_count
                        FROM filtered
                        WHERE evidence_state = 'eligible'
                          AND time_state = 'definite'
                          AND requested_model_raw IS NOT NULL
                        GROUP BY requested_model_raw
                    ) AS requested_models
                ) AS by_requested_model_raw,
                (
                    SELECT COALESCE(jsonb_object_agg(model, model_count),
                                    '{{}}'::jsonb)
                    FROM (
                        SELECT recorded_final_model_raw AS model, count(*) AS model_count
                        FROM filtered
                        WHERE evidence_state = 'eligible'
                          AND time_state = 'definite'
                          AND recorded_final_model_raw IS NOT NULL
                        GROUP BY recorded_final_model_raw
                    ) AS recorded_models
                ) AS by_recorded_final_model_raw,
                (
                    SELECT COALESCE(jsonb_object_agg(model, model_count),
                                    '{{}}'::jsonb)
                    FROM (
                        SELECT resolved_model_raw AS model, count(*) AS model_count
                        FROM filtered
                        WHERE evidence_state = 'eligible'
                          AND time_state = 'definite'
                          AND resolved_model_raw IS NOT NULL
                        GROUP BY resolved_model_raw
                    ) AS resolved_models
                ) AS by_resolved_model_raw
            FROM filtered
        """
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
        if row is None:
            raise LedgerError("count query returned no row")
        values = dict(
            zip(
                (
                    "total",
                    "completed",
                    "ambiguous",
                    "unknown_time",
                    "unknown_identity",
                    "unknown_surface",
                    "unknown_origin",
                    "unknown_model",
                    "excluded_surface",
                    "excluded_origin",
                    "excluded_non_generation",
                    "uncertain_outcome",
                    "observed_model_mismatches",
                    "by_requested_family",
                    "by_recorded_final_family",
                    "by_resolved_family",
                    "by_requested_model_raw",
                    "by_recorded_final_model_raw",
                    "by_resolved_model_raw",
                ),
                row,
            )
        )
        for key in (
            "by_requested_family",
            "by_recorded_final_family",
            "by_resolved_family",
            "by_requested_model_raw",
            "by_recorded_final_model_raw",
            "by_resolved_model_raw",
        ):
            values[key] = _json_count_map(values[key])
        return UsageCounts(
            total=int(values["total"] or 0),
            completed=int(values["completed"] or 0),
            ambiguous=int(values["ambiguous"] or 0),
            unknown_time=int(values["unknown_time"] or 0),
            unknown_identity=int(values["unknown_identity"] or 0),
            unknown_surface=int(values["unknown_surface"] or 0),
            unknown_origin=int(values["unknown_origin"] or 0),
            unknown_model=int(values["unknown_model"] or 0),
            excluded_surface=int(values["excluded_surface"] or 0),
            excluded_origin=int(values["excluded_origin"] or 0),
            excluded_non_generation=int(values["excluded_non_generation"] or 0),
            uncertain_outcome=int(values["uncertain_outcome"] or 0),
            observed_model_mismatches=int(values["observed_model_mismatches"] or 0),
            by_requested_family=values["by_requested_family"],
            by_recorded_final_family=values["by_recorded_final_family"],
            by_resolved_family=values["by_resolved_family"],
            by_requested_model_raw=values["by_requested_model_raw"],
            by_recorded_final_model_raw=values["by_recorded_final_model_raw"],
            by_resolved_model_raw=values["by_resolved_model_raw"],
        )

    def close(self) -> None:
        return None

    @staticmethod
    def assert_safe_record(value: Any) -> None:
        assert_no_secrets(value)


class PgLedgerPage:
    """One PostgreSQL transaction covering a complete collection page."""

    def __init__(
        self,
        ledger: PgLedger,
        conn: psycopg.Connection,
        *,
        expected_binding: Optional[LedgerBinding] = None,
        operation: Any = None,
    ) -> None:
        self.ledger = ledger
        self.conn = conn
        self._bindings: dict[str, LedgerBinding] = {}
        self._expected_binding = expected_binding
        self.operation = operation

    @staticmethod
    def assert_safe_record(value: Any) -> None:
        assert_no_secrets(value)

    @contextmanager
    def _cursor(self) -> Iterator[_OperationCursor]:
        with self.ledger.cursor(self.conn, operation=self.operation) as cursor:
            yield cursor

    def bind_scope(
        self,
        scope: LedgerScope,
        *,
        seen_at: datetime,
        expected_binding: Optional[LedgerBinding] = None,
        allow_initialize: bool = False,
        allow_rebind: bool = False,
    ) -> LedgerBinding:
        normalized_scope = _normalize_scope(scope)
        self.ledger.assert_safe_record(_scope_payload(normalized_scope))
        observed_at = ensure_utc(seen_at)
        expected = expected_binding if expected_binding is not None else self._expected_binding
        if expected is not None:
            return self._bind_fenced_scope(
                normalized_scope,
                expected=expected,
                seen_at=observed_at,
                allow_rebind=allow_rebind,
            )
        if not allow_initialize:
            raise LedgerError("expected binding is required for scoped writes")
        return self._initialize_scope_binding(normalized_scope, seen_at=observed_at)

    def _bind_fenced_scope(
        self,
        scope: LedgerScope,
        *,
        expected: LedgerBinding,
        seen_at: datetime,
        allow_rebind: bool,
    ) -> LedgerBinding:
        key = scope_key(scope)
        identity_state = _scope_identity_state(scope)
        if (
            expected.collector_account_id != scope.collector_account_id
            or expected.binding_state != "active"
        ):
            raise LedgerError("scope binding fence does not match the requested scope")
        with self._cursor() as cur:
            _lock_scope_registry(cur)
            _assert_active_binding(
                cur,
                collector_account_id=scope.collector_account_id,
                scope_key_value=expected.scope_key,
                binding_generation=expected.binding_generation,
            )
            if expected.scope_key == key:
                self._refresh_fenced_binding(cur, scope, expected, seen_at)
                return expected
            if not allow_rebind:
                raise LedgerError("scope binding changed; refine it explicitly")
            _assert_scope_not_retired(cur, key)
            _assert_scope_compatible(
                cur,
                previous_scope_key=expected.scope_key,
                scope=scope,
            )
            _ensure_scope_row(
                cur,
                scope=scope,
                key=key,
                identity_state=identity_state,
                seen_at=seen_at,
            )
            _assert_expected_binding_still_active(cur, scope, expected)
            prior_identity_state = _scope_row_identity_state(cur, expected.scope_key)
            self._retire_binding(
                cur,
                collector_account_id=scope.collector_account_id,
                seen_at=seen_at,
            )
            if prior_identity_state != "retired" and expected.scope_key != key:
                _redirect_scope(
                    cur,
                    retired_scope_key=expected.scope_key,
                    canonical_scope_key=key,
                    seen_at=seen_at,
                )
            generation = int(expected.binding_generation) + 1
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_scope_bindings (
                    scope_key, collector_account_id, binding_generation,
                    binding_state, first_seen_at, last_seen_at
                ) VALUES (%s, %s, %s, 'active', %s, %s)
                """,
                (key, scope.collector_account_id, generation, seen_at, seen_at),
            )
        return self._cache_binding(scope, key, generation, identity_state)

    def _initialize_scope_binding(
        self,
        scope: LedgerScope,
        *,
        seen_at: datetime,
    ) -> LedgerBinding:
        key = scope_key(scope)
        identity_state = _scope_identity_state(scope)
        cached = self._bindings.get(scope.collector_account_id)
        if cached is not None and cached.scope_key != key:
            raise LedgerError("scope binding changed; refine it explicitly")
        if cached is not None and cached.scope_key == key:
            return cached
        with self._cursor() as cur:
            _lock_scope_registry(cur)
            cur.execute(
                """
                SELECT scope_key, binding_generation
                FROM public.chatgpt_usage_scope_bindings
                WHERE collector_account_id = %s AND binding_state = 'active'
                FOR UPDATE
                """,
                (scope.collector_account_id,),
            )
            current = cur.fetchone()
            if current is not None and str(current[0]) != key:
                raise LedgerError("scope binding already exists; refine it explicitly")
            _assert_scope_not_retired(cur, key)
            _ensure_scope_row(
                cur,
                scope=scope,
                key=key,
                identity_state=identity_state,
                seen_at=seen_at,
            )
            if current is not None:
                generation = int(current[1])
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_scope_bindings
                    SET last_seen_at = GREATEST(last_seen_at, %s)
                    WHERE scope_key = %s
                      AND collector_account_id = %s
                      AND binding_generation = %s
                      AND binding_state = 'active'
                    """,
                    (seen_at, key, scope.collector_account_id, generation),
                )
                if cur.rowcount != 1:
                    raise LedgerError("scope binding fence is stale")
            else:
                generation = 1
                cur.execute(
                    """
                    INSERT INTO public.chatgpt_usage_scope_bindings (
                        scope_key, collector_account_id, binding_generation,
                        binding_state, first_seen_at, last_seen_at
                    ) VALUES (%s, %s, %s, 'active', %s, %s)
                    """,
                    (key, scope.collector_account_id, generation, seen_at, seen_at),
                )
        return self._cache_binding(scope, key, generation, identity_state)

    def _refresh_fenced_binding(
        self,
        cur: psycopg.Cursor,
        scope: LedgerScope,
        expected: LedgerBinding,
        seen_at: datetime,
    ) -> None:
        cur.execute(
            """
            UPDATE public.chatgpt_usage_scope_bindings
            SET last_seen_at = GREATEST(last_seen_at, %s)
            WHERE scope_key = %s
              AND collector_account_id = %s
              AND binding_generation = %s
              AND binding_state = 'active'
            """,
            (
                seen_at,
                expected.scope_key,
                scope.collector_account_id,
                expected.binding_generation,
            ),
        )
        if cur.rowcount != 1:
            raise LedgerError("scope binding fence is stale")
        self._bindings[scope.collector_account_id] = expected

    def _retire_binding(
        self,
        cur: psycopg.Cursor,
        *,
        collector_account_id: str,
        seen_at: datetime,
    ) -> None:
        cur.execute(
            """
            UPDATE public.chatgpt_usage_scope_bindings
            SET binding_state = 'retired', retired_at = %s,
                last_seen_at = GREATEST(last_seen_at, %s)
            WHERE collector_account_id = %s AND binding_state = 'active'
            """,
            (seen_at, seen_at, collector_account_id),
        )

    def _cache_binding(
        self,
        scope: LedgerScope,
        key: str,
        generation: int,
        identity_state: str,
    ) -> LedgerBinding:
        binding = LedgerBinding(
            collector_account_id=scope.collector_account_id,
            scope_key=key,
            binding_generation=generation,
            binding_state="active",
            identity_state=identity_state,
        )
        self._bindings[scope.collector_account_id] = binding
        return binding

    def record_observation(
        self,
        scope: LedgerScope,
        context: IngestContext,
        payload: Mapping[str, Any],
    ) -> tuple[str, bool]:
        safe_scope = _normalize_scope(scope)
        safe_context = _normalize_context(context)
        sanitized = _observation_envelope(payload)
        provenance = sanitize_provenance(safe_context.provenance)
        provenance.update(
            {
                "collector_account_id": safe_scope.collector_account_id,
                "run_id": safe_context.run_id,
                "schema_version": safe_context.schema_version,
                "source_id": safe_context.source_id,
                "source_kind": safe_context.source_kind,
            }
        )
        provenance.setdefault("transfer_schema_version", TRANSFER_SCHEMA_VERSION)
        self.ledger.assert_safe_record(
            _prepared_observation_record(
                safe_scope,
                safe_context,
                sanitized,
                provenance,
            )
        )
        binding = self.bind_scope(safe_scope, seen_at=safe_context.observed_at)
        stable_payload = _without_keys(sanitized, "provenance", "run_id")
        fingerprint_provenance = _without_keys(
            provenance,
            "collector_account_id",
            "run_id",
        )
        revision_fingerprint = fingerprint_value(
            {"payload": stable_payload, "provenance": fingerprint_provenance}
        )
        observed_at = safe_context.observed_at
        with self._cursor() as cur:
            _lock_scope(cur, binding.scope_key)
            prior = _latest_observation(cur, binding.scope_key, safe_context)
            current = _current_observation(cur, binding.scope_key, safe_context)
            matching = _matching_observation(
                cur,
                binding.scope_key,
                safe_context,
                revision_fingerprint,
            )
            if (
                matching is not None
                and current is not None
                and matching["observation_id"] == current["observation_id"]
            ):
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_observations
                    SET observed_at = CASE
                            WHEN %s >= observed_at THEN %s ELSE observed_at END,
                        last_seen_at = GREATEST(last_seen_at, %s),
                        last_seen_run_id = CASE
                            WHEN %s >= last_seen_at THEN %s ELSE last_seen_run_id END,
                        last_seen_provenance = CASE
                            WHEN %s >= last_seen_at THEN %s::jsonb
                            ELSE last_seen_provenance END
                    WHERE scope_key = %s AND observation_id = %s
                    """,
                    (
                        observed_at,
                        observed_at,
                        observed_at,
                        observed_at,
                        safe_context.run_id,
                        observed_at,
                        json.dumps(provenance, separators=(",", ":"), default=str),
                        binding.scope_key,
                        matching["observation_id"],
                    ),
                )
                _record_activity_provenance(
                    cur,
                    binding.scope_key,
                    "observation",
                    matching["observation_id"],
                    safe_scope.collector_account_id,
                    observed_at,
                )
                return matching["observation_id"], False
            if (
                matching is not None
                and current is not None
                and matching["observation_id"] != current["observation_id"]
                and observed_at <= current["observed_at"]
            ):
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_observations
                    SET last_seen_at = GREATEST(last_seen_at, %s),
                        last_seen_run_id = CASE
                            WHEN %s >= last_seen_at THEN %s ELSE last_seen_run_id END,
                        last_seen_provenance = CASE
                            WHEN %s >= last_seen_at THEN %s::jsonb
                            ELSE last_seen_provenance END
                    WHERE scope_key = %s AND observation_id = %s
                    """,
                    (
                        observed_at,
                        observed_at,
                        safe_context.run_id,
                        observed_at,
                        json.dumps(provenance, separators=(",", ":"), default=str),
                        binding.scope_key,
                        matching["observation_id"],
                    ),
                )
                _record_activity_provenance(
                    cur,
                    binding.scope_key,
                    "observation",
                    matching["observation_id"],
                    safe_scope.collector_account_id,
                    observed_at,
                )
                return matching["observation_id"], False
            if prior is not None and prior["revision_fingerprint"] == revision_fingerprint:
                should_promote = current is None or (
                    current["observation_id"] != prior["observation_id"]
                    and observed_at >= current["observed_at"]
                )
                if should_promote and current is not None:
                    cur.execute(
                        """
                        UPDATE public.chatgpt_usage_observations
                        SET is_current_projection = FALSE
                        WHERE scope_key = %s AND observation_id = %s
                        """,
                        (binding.scope_key, current["observation_id"]),
                    )
                if should_promote:
                    cur.execute(
                        """
                        UPDATE public.chatgpt_usage_observations
                        SET is_current_projection = TRUE,
                            observed_at = GREATEST(observed_at, %s),
                            supersedes_observation_id = %s
                        WHERE scope_key = %s AND observation_id = %s
                        """,
                        (
                            observed_at,
                            current["observation_id"] if current is not None else None,
                            binding.scope_key,
                            prior["observation_id"],
                        ),
                    )
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_observations
                    SET observed_at = CASE
                            WHEN %s >= observed_at THEN %s ELSE observed_at END,
                        last_seen_at = GREATEST(last_seen_at, %s),
                        last_seen_run_id = CASE
                            WHEN %s >= last_seen_at THEN %s ELSE last_seen_run_id END,
                        last_seen_provenance = CASE
                            WHEN %s >= last_seen_at THEN %s::jsonb
                            ELSE last_seen_provenance END
                    WHERE scope_key = %s AND observation_id = %s
                    """,
                    (
                        observed_at,
                        observed_at,
                        observed_at,
                        observed_at,
                        safe_context.run_id,
                        observed_at,
                        json.dumps(provenance, separators=(",", ":"), default=str),
                        binding.scope_key,
                        prior["observation_id"],
                    ),
                )
                _record_activity_provenance(
                    cur,
                    binding.scope_key,
                    "observation",
                    prior["observation_id"],
                    safe_scope.collector_account_id,
                    observed_at,
                )
                return prior["observation_id"], False

            occurrence_number = int(prior["occurrence_number"]) + 1 if prior else 1
            observation_id = stable_id(
                binding.scope_key,
                safe_context.source_kind,
                safe_context.source_id,
                "occurrence",
                str(occurrence_number),
            )
            is_current = current is None or observed_at >= current["observed_at"]
            if is_current and current is not None:
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_observations
                    SET is_current_projection = FALSE
                    WHERE scope_key = %s AND observation_id = %s
                    """,
                    (binding.scope_key, current["observation_id"]),
                )
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_observations (
                    observation_id, scope_key, collector_account_id, provider,
                    provider_user_id, workspace_id, quota_owner_id, source_kind,
                    source_id, revision_fingerprint, revision_number,
                    occurrence_number, surface, conversation_id, payload,
                    observed_at, run_id, schema_version, provenance,
                    first_seen_at, last_seen_at, last_seen_run_id,
                    last_seen_provenance, is_current_projection,
                    supersedes_observation_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb,
                    %s, %s
                )
                """,
                (
                    observation_id,
                    binding.scope_key,
                    safe_scope.collector_account_id,
                    safe_scope.provider,
                    safe_scope.provider_user_id,
                    safe_scope.workspace_id,
                    safe_scope.quota_owner_id,
                    safe_context.source_kind,
                    safe_context.source_id,
                    revision_fingerprint,
                    occurrence_number,
                    occurrence_number,
                    safe_scope.surface,
                    sanitized.get("conversation_id") or sanitized.get("conversationId"),
                    json.dumps(sanitized, separators=(",", ":"), default=str),
                    observed_at,
                    safe_context.run_id,
                    safe_context.schema_version,
                    json.dumps(provenance, separators=(",", ":"), default=str),
                    observed_at,
                    observed_at,
                    safe_context.run_id,
                    json.dumps(provenance, separators=(",", ":"), default=str),
                    is_current,
                    current["observation_id"] if is_current and current else None,
                ),
            )
            _record_activity_provenance(
                cur,
                binding.scope_key,
                "observation",
                observation_id,
                safe_scope.collector_account_id,
                observed_at,
            )
            return observation_id, True

    def upsert_attempt(
        self,
        scope: LedgerScope,
        attempt: AttemptRecord,
        context: IngestContext,
    ) -> AttemptUpsertResult:
        safe_scope = _normalize_scope(scope)
        safe_context = _normalize_context(context)
        aliases = _unique_aliases(attempt.aliases)
        safe_attempt = _normalize_attempt(attempt, aliases)
        initial_payload = _attempt_payload(safe_attempt, aliases)
        self.ledger.assert_safe_record(
            _prepared_attempt_record(
                safe_scope,
                safe_context,
                initial_payload,
                aliases,
            )
        )
        binding = self.bind_scope(safe_scope, seen_at=safe_context.observed_at)
        observed_at = safe_context.observed_at
        with self._cursor() as cur:
            lineage_keys = _scope_lineage(cur, binding.scope_key)
            _lock_scope_lineage(cur, lineage_keys)
            resolution = _resolve_attempt_identity(
                self,
                cur,
                safe_scope,
                binding.scope_key,
                safe_attempt,
                aliases,
                observed_at,
                lineage_keys,
            )
            canonical_ref = resolution.canonical_ref
            effective_id = canonical_ref.attempt_id
            attempt_payload = _attempt_payload(
                safe_attempt,
                aliases,
                attempt_id=effective_id,
                quarantine_reason=resolution.quarantine_reason,
            )
            self.ledger.assert_safe_record(
                _prepared_attempt_record(
                    safe_scope,
                    safe_context,
                    attempt_payload,
                    aliases,
                )
            )
            projection_fingerprint = fingerprint_value(attempt_payload)
            canonical_scope_key = canonical_ref.scope_key
            current = _attempt(cur, canonical_scope_key, effective_id)
            if (
                current is not None
                and current["projection_fingerprint"] == projection_fingerprint
                and observed_at < current["observed_at"]
            ):
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempts
                    SET last_seen_at = GREATEST(last_seen_at, %s)
                    WHERE scope_key = %s AND attempt_id = %s
                    """,
                    (observed_at, canonical_scope_key, effective_id),
                )
                return AttemptUpsertResult(
                    effective_id,
                    "quarantined"
                    if _attempt_quarantine_state(attempt_payload) != "clear"
                    else "stale",
                    resolution.conflicts,
                )
            if current is not None and current["tombstone"]:
                _record_identity_gap(
                    self,
                    cur,
                    safe_scope,
                    binding.scope_key,
                    source_id=effective_id,
                    reason="retired_attempt_reappeared",
                    details={
                        "participants": [
                            _attempt_participant(
                                canonical_ref,
                                role="reappeared",
                            )
                        ]
                    },
                    seen_at=observed_at,
                )
                return AttemptUpsertResult(
                    effective_id,
                    "quarantined",
                    resolution.conflicts + 1,
                )

            quarantine_state = _attempt_quarantine_state(attempt_payload)
            publish_aliases = (
                quarantine_state == "clear"
                and resolution.quarantine_reason is None
            )
            if current is not None and current["projection_fingerprint"] == projection_fingerprint:
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempts
                    SET observed_at = CASE
                            WHEN %s >= observed_at THEN %s ELSE observed_at END,
                        last_seen_at = GREATEST(last_seen_at, %s)
                    WHERE scope_key = %s AND attempt_id = %s
                    """,
                    (
                        observed_at,
                        observed_at,
                        observed_at,
                        canonical_scope_key,
                        effective_id,
                    ),
                )
                if publish_aliases:
                    _finalize_identity_merges(
                        cur,
                        canonical_ref=canonical_ref,
                        merge_refs=resolution.merge_refs,
                        seen_at=observed_at,
                    )
                merge_result = (
                    _merge_attempt_links(
                        self,
                        cur,
                        safe_scope,
                        scope_key_value=canonical_scope_key,
                        attempt_id=effective_id,
                        aliases=aliases,
                        seen_at=observed_at,
                        lineage_keys=lineage_keys,
                    )
                    if publish_aliases
                    else AliasMergeResult()
                )
                _record_activity_provenance(
                    cur,
                    canonical_scope_key,
                    "attempt",
                    effective_id,
                    safe_scope.collector_account_id,
                    observed_at,
                )
                return AttemptUpsertResult(
                    effective_id,
                    "quarantined" if quarantine_state != "clear" else "deduplicated",
                    resolution.conflicts + merge_result.conflicts + merge_result.quarantine_conflicts,
                )

            revision = _next_attempt_revision(cur, canonical_scope_key, effective_id)
            if current is None:
                _insert_attempt_head(
                    cur,
                    scope=safe_scope,
                    scope_key_value=canonical_scope_key,
                    attempt=safe_attempt,
                    attempt_id=effective_id,
                    revision=revision,
                    projection_fingerprint=projection_fingerprint,
                    observed_at=observed_at,
                    quarantine_state=quarantine_state,
                    warnings=attempt_payload["warnings"],
                )
                status = "quarantined" if quarantine_state != "clear" else "inserted"
            else:
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempts
                    SET collector_account_id = %s, conversation_id = %s,
                        identity_basis = %s, time_basis = %s,
                        attempt_time = %s, earliest_possible_at = %s,
                        latest_possible_at = %s, requested_model_raw = %s,
                        requested_mode_raw = %s,
                        requested_reasoning_effort_raw = %s,
                        recorded_final_model_raw = %s, resolved_model_raw = %s,
                        requested_family = %s, recorded_final_family = %s,
                        resolved_family = %s, mapping_version = %s,
                        outcome = %s, completed_answer = %s,
                        generation_started = %s, surface = %s, origin = %s,
                        revision = %s, projection_fingerprint = %s,
                        warnings = %s::jsonb, observed_at = %s,
                        updated_at = GREATEST(updated_at, %s),
                        last_seen_at = GREATEST(last_seen_at, %s),
                        quarantine_state = %s,
                        tombstone = FALSE, superseded_by_attempt_id = NULL
                    WHERE scope_key = %s AND attempt_id = %s
                    """,
                    _attempt_projection_params(
                        safe_scope,
                        safe_attempt,
                        revision,
                        projection_fingerprint,
                        observed_at,
                        canonical_scope_key,
                        effective_id,
                        quarantine_state,
                        attempt_payload["warnings"],
                    ),
                )
                status = "quarantined" if quarantine_state != "clear" else "updated"
            if current is not None:
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempt_revisions
                    SET is_current_projection = FALSE
                    WHERE scope_key = %s AND attempt_id = %s
                      AND is_current_projection = TRUE
                    """,
                    (canonical_scope_key, effective_id),
                )
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_attempt_revisions (
                    scope_key, attempt_id, revision, projection_fingerprint, payload,
                    source_kind, source_id, run_id, schema_version,
                    collector_account_id, recorded_at, is_current_projection
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    canonical_scope_key,
                    effective_id,
                    revision,
                    projection_fingerprint,
                    json.dumps(attempt_payload, separators=(",", ":"), default=str),
                    safe_context.source_kind,
                    safe_context.source_id,
                    safe_context.run_id,
                    safe_context.schema_version,
                    safe_scope.collector_account_id,
                    observed_at,
                    True,
                ),
            )
            if publish_aliases:
                _finalize_identity_merges(
                    cur,
                    canonical_ref=canonical_ref,
                    merge_refs=resolution.merge_refs,
                    seen_at=observed_at,
                )
            merge_result = (
                _merge_attempt_links(
                    self,
                    cur,
                    safe_scope,
                    scope_key_value=canonical_scope_key,
                    attempt_id=effective_id,
                    aliases=aliases,
                    seen_at=observed_at,
                    lineage_keys=lineage_keys,
                )
                if publish_aliases
                else AliasMergeResult()
            )
            _record_activity_provenance(
                cur,
                canonical_scope_key,
                "attempt",
                effective_id,
                safe_scope.collector_account_id,
                observed_at,
            )
            return AttemptUpsertResult(
                effective_id,
                status,
                resolution.conflicts + merge_result.conflicts + merge_result.quarantine_conflicts,
            )

    def record_coverage_gap(
        self,
        scope: LedgerScope,
        *,
        source_kind: str,
        source_id: str,
        reason: str,
        state: str = "open",
        details: Optional[Mapping[str, Any]] = None,
        seen_at: datetime,
    ) -> str:
        safe_scope = _normalize_scope(scope)
        safe_source_kind = _required_token(source_kind, "source_kind")
        safe_source_id = _required_token(source_id, "source_id")
        safe_reason = _required_token(reason, "reason")
        safe_state = _enum_token(state, _ALLOWED_GAP_STATES, "unknown")
        safe_details = _coverage_details_envelope(details)
        safe_seen_at = _utc_datetime(seen_at, "seen_at")
        self.ledger.assert_safe_record(
            _coverage_write_envelope(
                safe_scope,
                scope_key_value=scope_key(safe_scope),
                source_kind=safe_source_kind,
                source_id=safe_source_id,
                reason=safe_reason,
                state=safe_state,
                details=safe_details,
                seen_at=safe_seen_at,
            )
        )
        binding = self.bind_scope(safe_scope, seen_at=safe_seen_at)
        gap_id = stable_id(binding.scope_key, safe_source_kind, safe_source_id, safe_reason)
        with self._cursor() as cur:
            _lock_scope(cur, binding.scope_key)
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_coverage_gaps (
                    gap_id, scope_key, collector_account_id, source_kind, source_id,
                    reason, state, first_seen_at, last_seen_at, details
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (scope_key, source_kind, source_id, reason) DO UPDATE SET
                    state = CASE
                        WHEN EXCLUDED.last_seen_at
                            >= public.chatgpt_usage_coverage_gaps.last_seen_at
                        THEN EXCLUDED.state
                        ELSE public.chatgpt_usage_coverage_gaps.state
                    END,
                    last_seen_at = GREATEST(
                        public.chatgpt_usage_coverage_gaps.last_seen_at,
                        EXCLUDED.last_seen_at
                    ),
                    details = CASE
                        WHEN EXCLUDED.last_seen_at
                            >= public.chatgpt_usage_coverage_gaps.last_seen_at
                        THEN EXCLUDED.details
                        ELSE public.chatgpt_usage_coverage_gaps.details
                    END
                """,
                (
                    gap_id,
                    binding.scope_key,
                    safe_scope.collector_account_id,
                    safe_source_kind,
                    safe_source_id,
                    safe_reason,
                    safe_state,
                    safe_seen_at,
                    safe_seen_at,
                    json.dumps(safe_details, separators=(",", ":"), default=str),
                ),
            )
        return gap_id

    def resolve_coverage_gaps(
        self,
        scope: LedgerScope,
        *,
        source_kind: str,
        source_id: str,
        seen_at: datetime,
    ) -> None:
        safe_scope = _normalize_scope(scope)
        safe_source_kind = _required_token(source_kind, "source_kind")
        safe_source_id = _required_token(source_id, "source_id")
        safe_seen_at = _utc_datetime(seen_at, "seen_at")
        self.ledger.assert_safe_record(
            _coverage_write_envelope(
                safe_scope,
                scope_key_value=scope_key(safe_scope),
                source_kind=safe_source_kind,
                source_id=safe_source_id,
                reason="coverage_gap_resolution",
                state="resolved",
                details=None,
                seen_at=safe_seen_at,
            )
        )
        binding = self.bind_scope(safe_scope, seen_at=safe_seen_at)
        with self._cursor() as cur:
            _lock_scope(cur, binding.scope_key)
            cur.execute(
                """
                UPDATE public.chatgpt_usage_coverage_gaps
                SET state = 'resolved', last_seen_at = %s
                WHERE scope_key = %s AND source_kind = %s AND source_id = %s
                  AND state = 'open' AND last_seen_at <= %s
                """,
                (
                    safe_seen_at,
                    binding.scope_key,
                    safe_source_kind,
                    safe_source_id,
                    safe_seen_at,
                ),
            )


def scope_key(scope: LedgerScope) -> str:
    scope = _normalize_scope(scope)
    owner = [
        scope.provider,
        scope.provider_user_id,
        scope.workspace_id,
        scope.quota_owner_id,
        scope.surface,
    ]
    if _scope_identity_state(scope) != "verified":
        owner[:0] = ["provisional", scope.collector_account_id]
    return fingerprint_value(owner)


def _scope_identity_state(scope: LedgerScope) -> str:
    if scope.provider_user_id is not None and scope.workspace_id is not None and scope.quota_owner_id is not None:
        return "verified"
    return "provisional"


def _assert_scope_not_retired(cur: psycopg.Cursor, scope_key_value: str) -> None:
    cur.execute(
        """
        SELECT identity_state
        FROM public.chatgpt_usage_scopes
        WHERE scope_key = %s
        FOR UPDATE
        """,
        (scope_key_value,),
    )
    row = cur.fetchone()
    if row is not None and str(row[0]) == "retired":
        raise LedgerError("retired scope cannot be reactivated")
    cur.execute(
        """
        SELECT 1
        FROM public.chatgpt_usage_scope_redirects
        WHERE retired_scope_key = %s
        """,
        (scope_key_value,),
    )
    if cur.fetchone() is not None:
        raise LedgerError("retired scope cannot be reactivated")


def _assert_scope_compatible(
    cur: psycopg.Cursor,
    *,
    previous_scope_key: str,
    scope: LedgerScope,
) -> None:
    cur.execute(
        """
        SELECT provider, provider_user_id, workspace_id,
               quota_owner_id, surface, identity_state
        FROM public.chatgpt_usage_scopes
        WHERE scope_key = %s
        FOR UPDATE
        """,
        (previous_scope_key,),
    )
    previous = cur.fetchone()
    if previous is None:
        raise LedgerError("scope binding refers to a missing scope")
    if str(previous[5]) == "retired":
        raise LedgerError("retired scope cannot be refined")
    labels = (
        ("provider", previous[0], scope.provider),
        ("surface", previous[4], scope.surface),
    )
    for label, old_value, new_value in labels:
        if old_value is not None and new_value is not None and str(old_value) != str(new_value):
            raise LedgerError(f"scope identity component is incompatible: {label}")

    # Dropping a known identity component is information loss, not refinement.
    for index, label in ((1, "provider_user_id"), (2, "workspace_id"), (3, "quota_owner_id")):
        if previous[index] is not None and getattr(scope, label) is None:
            raise LedgerError(f"scope identity component cannot be removed: {label}")


def _ensure_scope_row(
    cur: psycopg.Cursor,
    *,
    scope: LedgerScope,
    key: str,
    identity_state: str,
    seen_at: datetime,
) -> None:
    cur.execute(
        """
        INSERT INTO public.chatgpt_usage_scopes (
            scope_key, collector_account_id, provider, provider_user_id,
            workspace_id, quota_owner_id, surface, identity_state,
            created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (scope_key) DO UPDATE SET
            provider = EXCLUDED.provider,
            provider_user_id = COALESCE(
                EXCLUDED.provider_user_id,
                public.chatgpt_usage_scopes.provider_user_id
            ),
            workspace_id = COALESCE(
                EXCLUDED.workspace_id,
                public.chatgpt_usage_scopes.workspace_id
            ),
            quota_owner_id = COALESCE(
                EXCLUDED.quota_owner_id,
                public.chatgpt_usage_scopes.quota_owner_id
            ),
            surface = EXCLUDED.surface,
            identity_state = CASE
                WHEN EXCLUDED.identity_state = 'verified' THEN 'verified'
                ELSE public.chatgpt_usage_scopes.identity_state
            END,
            updated_at = GREATEST(
                public.chatgpt_usage_scopes.updated_at,
                EXCLUDED.updated_at
            )
        """,
        (
            key,
            scope.collector_account_id,
            scope.provider,
            scope.provider_user_id,
            scope.workspace_id,
            scope.quota_owner_id,
            scope.surface,
            identity_state,
            seen_at,
            seen_at,
        ),
    )


def _redirect_scope(
    cur: psycopg.Cursor,
    *,
    retired_scope_key: str,
    canonical_scope_key: str,
    seen_at: datetime,
) -> None:
    if retired_scope_key == canonical_scope_key:
        raise LedgerError("scope redirect must change the canonical scope")
    cur.execute(
        """
        INSERT INTO public.chatgpt_usage_scope_redirects (
            retired_scope_key, canonical_scope_key, reason,
            first_seen_at, last_seen_at
        ) VALUES (%s, %s, 'identity_refinement', %s, %s)
        ON CONFLICT (retired_scope_key) DO NOTHING
        """,
        (retired_scope_key, canonical_scope_key, seen_at, seen_at),
    )
    cur.execute(
        """
        SELECT canonical_scope_key
        FROM public.chatgpt_usage_scope_redirects
        WHERE retired_scope_key = %s
        """,
        (retired_scope_key,),
    )
    row = cur.fetchone()
    if row is None or str(row[0]) != canonical_scope_key:
        raise LedgerError("scope redirect is immutable")
    cur.execute(
        """
        UPDATE public.chatgpt_usage_scopes
        SET identity_state = 'retired',
            superseded_by_scope_key = %s,
            updated_at = GREATEST(updated_at, %s)
        WHERE scope_key = %s
        """,
        (canonical_scope_key, seen_at, retired_scope_key),
    )


def stable_id(*parts: str) -> str:
    return fingerprint_value(list(parts))


def fingerprint_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(_sorted_value(value), separators=(",", ":"), ensure_ascii=False)


def sanitize_provenance(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if key not in PROVENANCE_ALLOWLIST:
            continue
        projected = _sanitize_provenance_value(raw_value)
        if projected is not None:
            out[key] = projected
    return out


def _latest_observation(
    cur: psycopg.Cursor,
    scope_key_value: str,
    context: IngestContext,
) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT observation_id, revision_number, occurrence_number,
               revision_fingerprint, observed_at
        FROM public.chatgpt_usage_observations
        WHERE scope_key = %s AND source_kind = %s AND source_id = %s
        ORDER BY occurrence_number DESC
        LIMIT 1
        """,
        (scope_key_value, context.source_kind, context.source_id),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(
        zip(
            (
                "observation_id",
                "revision_number",
                "occurrence_number",
                "revision_fingerprint",
                "observed_at",
            ),
            row,
        )
    )


def _current_observation(
    cur: psycopg.Cursor,
    scope_key_value: str,
    context: IngestContext,
) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT observation_id, observed_at, last_seen_at, occurrence_number,
               revision_fingerprint
        FROM public.chatgpt_usage_observations
        WHERE scope_key = %s AND source_kind = %s AND source_id = %s
          AND is_current_projection = TRUE
        ORDER BY observed_at DESC, occurrence_number DESC
        LIMIT 1
        """,
        (scope_key_value, context.source_kind, context.source_id),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(
        zip(
            (
                "observation_id",
                "observed_at",
                "last_seen_at",
                "occurrence_number",
                "revision_fingerprint",
            ),
            row,
        )
    )


def _matching_observation(
    cur: psycopg.Cursor,
    scope_key_value: str,
    context: IngestContext,
    revision_fingerprint: str,
) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT observation_id, observed_at, last_seen_at, occurrence_number,
               revision_fingerprint
        FROM public.chatgpt_usage_observations
        WHERE scope_key = %s AND source_kind = %s AND source_id = %s
          AND revision_fingerprint = %s
        ORDER BY observed_at DESC, occurrence_number DESC
        LIMIT 1
        """,
        (scope_key_value, context.source_kind, context.source_id, revision_fingerprint),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(
        zip(
            (
                "observation_id",
                "observed_at",
                "last_seen_at",
                "occurrence_number",
                "revision_fingerprint",
            ),
            row,
        )
    )


def _attempt(
    cur: psycopg.Cursor,
    scope_key_value: str,
    attempt_id: str,
) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT revision, projection_fingerprint, collector_account_id, tombstone,
               quarantine_state, identity_basis, observed_at, last_seen_at,
               superseded_by_attempt_id
        FROM public.chatgpt_usage_attempts
        WHERE scope_key = %s AND attempt_id = %s
        """,
        (scope_key_value, attempt_id),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(
        zip(
            (
                "revision",
                "projection_fingerprint",
                "collector_account_id",
                "tombstone",
                "quarantine_state",
                "identity_basis",
                "observed_at",
                "last_seen_at",
                "superseded_by_attempt_id",
            ),
            row,
        )
    )


def _matching_alias_refs(
    cur: psycopg.Cursor,
    scope_keys: Sequence[str],
    aliases: Sequence[tuple[str, str]],
) -> set[AttemptRef]:
    matches: set[AttemptRef] = set()
    for scope_key_value in scope_keys:
        for alias_kind, alias_value in aliases:
            safe_value = sanitize_token(alias_value)
            if safe_value is None:
                continue
            cur.execute(
                """
                SELECT scope_key, attempt_id
                FROM public.chatgpt_usage_attempt_aliases
                WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
                """,
                (scope_key_value, alias_kind, safe_value),
            )
            for row in cur.fetchall():
                matches.add(AttemptRef(str(row[0]), str(row[1])))
    return matches


def _attempt_alias_values(
    cur: psycopg.Cursor,
    ref: AttemptRef,
    alias_kind: str,
) -> set[str]:
    cur.execute(
        """
        SELECT alias_value
        FROM public.chatgpt_usage_attempt_aliases
        WHERE scope_key = %s AND attempt_id = %s AND alias_kind = %s
        """,
        (ref.scope_key, ref.attempt_id, alias_kind),
    )
    return {str(row[0]) for row in cur.fetchall()}


def _attempt_participant(
    ref: AttemptRef,
    *,
    role: str,
) -> dict[str, str]:
    return {
        "type": "attempt",
        "role": _required_token(role, "participant_role"),
        "scope_key": _required_token(ref.scope_key, "scope_key"),
        "attempt_id": _required_token(ref.attempt_id, "attempt_id"),
    }


def _identity_rank(identity_basis: str) -> int:
    return {
        "generation": 4,
        "request": 3,
        "provisional": 2,
        "unresolved": 1,
    }.get(identity_basis, 0)


def _follow_superseded_attempt(
    cur: psycopg.Cursor,
    ref: AttemptRef,
    *,
    max_depth: int = 64,
) -> tuple[AttemptRef, Optional[dict[str, Any]]]:
    current_ref = ref
    visited: set[AttemptRef] = set()
    for _ in range(max_depth):
        if current_ref in visited:
            break
        visited.add(current_ref)
        row = _attempt(cur, current_ref.scope_key, current_ref.attempt_id)
        if row is None:
            return current_ref, None
        row_scope_key = current_ref.scope_key
        successor = row.get("superseded_by_attempt_id")
        if not row["tombstone"] or not successor:
            return current_ref, row
        successor_ref = AttemptRef(row_scope_key, str(successor))
        if _attempt(cur, successor_ref.scope_key, successor_ref.attempt_id) is None:
            return current_ref, row
        current_ref = successor_ref
    return current_ref, _attempt(cur, current_ref.scope_key, current_ref.attempt_id)


def _weak_matches_without_generation_anchors(
    cur: psycopg.Cursor,
    aliases: Sequence[tuple[str, str]],
    matches: set[AttemptRef],
) -> set[AttemptRef]:
    if not _attempt_alias_values_for_input(aliases, "generation"):
        return matches
    return {
        matched_ref
        for matched_ref in matches
        if not _attempt_alias_values(
            cur,
            _follow_superseded_attempt(cur, matched_ref)[0],
            "generation",
        )
    }


def _resolve_attempt_identity(
    page: PgLedgerPage,
    cur: psycopg.Cursor,
    scope: LedgerScope,
    scope_key_value: str,
    attempt: AttemptRecord,
    aliases: Sequence[tuple[str, str]],
    seen_at: datetime,
    lineage_keys: Optional[Sequence[str]] = None,
) -> AttemptIdentityResolution:
    raw_incoming_ref = AttemptRef(scope_key_value, attempt.attempt_id)
    incoming_ref, incoming_row = _follow_superseded_attempt(cur, raw_incoming_ref)
    lineage_keys = tuple(lineage_keys or _scope_lineage(cur, scope_key_value))
    strong_aliases = [alias for alias in aliases if alias[0] in _STRONG_ALIAS_KINDS]
    weak_aliases = [alias for alias in aliases if alias[0] not in _STRONG_ALIAS_KINDS]
    strong_matches = _matching_alias_refs(cur, lineage_keys, strong_aliases)
    if strong_matches:
        matches = strong_matches
    else:
        matches = _matching_alias_refs(cur, lineage_keys, weak_aliases)
        if matches:
            matches = _weak_matches_without_generation_anchors(cur, aliases, matches)
    matches.discard(raw_incoming_ref)
    matches.discard(incoming_ref)

    candidate_refs: set[AttemptRef] = set()
    missing_refs: set[AttemptRef] = set()
    for matched_ref in matches:
        resolved_ref, candidate = _follow_superseded_attempt(cur, matched_ref)
        if candidate is None:
            missing_refs.add(matched_ref)
        else:
            candidate_refs.add(resolved_ref)
    if incoming_row is not None:
        candidate_refs.add(incoming_ref)

    if missing_refs:
        _record_identity_gap(
            page,
            cur,
            scope,
            scope_key_value,
            source_id=raw_incoming_ref.attempt_id,
            reason="alias_to_missing_attempt",
            details={
                "participants": [
                    _attempt_participant(raw_incoming_ref, role="incoming"),
                    *[
                        _attempt_participant(ref, role="missing")
                        for ref in sorted(
                            missing_refs,
                            key=lambda item: (item.scope_key, item.attempt_id),
                        )[:16]
                    ],
                ]
            },
            seen_at=seen_at,
        )
        return AttemptIdentityResolution(
            incoming_ref,
            conflicts=len(missing_refs),
            quarantine_reason="alias_to_missing_attempt",
        )

    if not candidate_refs:
        return AttemptIdentityResolution(incoming_ref)

    candidate_rows: dict[AttemptRef, dict[str, Any]] = {}
    for ref in candidate_refs:
        candidate = _attempt(cur, ref.scope_key, ref.attempt_id)
        if candidate is not None:
            candidate_rows[ref] = candidate
    if not candidate_rows:
        return AttemptIdentityResolution(incoming_ref)

    incoming_generations = _attempt_alias_values_for_input(aliases, "generation")
    candidate_generations = {
        ref: _attempt_alias_values(cur, ref, "generation")
        for ref in candidate_rows
    }
    incompatible = {
        ref
        for ref, generations in candidate_generations.items()
        if generations
        and incoming_generations
        and not generations.intersection(incoming_generations)
    }
    nonempty_generation_sets = [
        generations for generations in candidate_generations.values() if generations
    ]
    generation_ambiguity = any(
        not left.intersection(right)
        for index, left in enumerate(nonempty_generation_sets)
        for right in nonempty_generation_sets[index + 1 :]
    )
    weak_only = not strong_matches and bool(matches)
    weak_selection = (
        weak_only
        and (
            attempt.identity_basis == "generation"
            or len(candidate_rows) > 1
            or any(candidate_generations.values())
        )
    )
    if incompatible or generation_ambiguity or weak_selection:
        _record_identity_gap(
            page,
            cur,
            scope,
            scope_key_value,
            source_id=raw_incoming_ref.attempt_id,
            reason="ambiguous_attempt_alias",
            details={
                "participants": [
                    _attempt_participant(raw_incoming_ref, role="incoming"),
                    *[
                        _attempt_participant(ref, role="candidate")
                        for ref in sorted(
                            candidate_rows,
                            key=lambda item: (item.scope_key, item.attempt_id),
                        )[:16]
                    ],
                ]
            },
            seen_at=seen_at,
        )
        return AttemptIdentityResolution(
            incoming_ref,
            conflicts=len(candidate_rows),
            quarantine_reason="ambiguous_attempt_alias",
        )

    tombstoned = {
        ref for ref, candidate in candidate_rows.items() if candidate["tombstone"]
    }
    if tombstoned:
        _record_identity_gap(
            page,
            cur,
            scope,
            scope_key_value,
            source_id=raw_incoming_ref.attempt_id,
            reason="alias_to_retired_attempt",
            details={
                "participants": [
                    _attempt_participant(raw_incoming_ref, role="incoming"),
                    *[
                        _attempt_participant(ref, role="retired")
                        for ref in sorted(
                            tombstoned,
                            key=lambda item: (item.scope_key, item.attempt_id),
                        )[:16]
                    ],
                ]
            },
            seen_at=seen_at,
        )
        return AttemptIdentityResolution(
            incoming_ref,
            conflicts=len(tombstoned),
            quarantine_reason="alias_to_retired_attempt",
        )

    def candidate_key(ref: AttemptRef) -> tuple[Any, ...]:
        candidate = candidate_rows[ref]
        return (
            candidate["observed_at"] is not None,
            candidate["observed_at"],
            _identity_rank(str(candidate["identity_basis"])),
            ref.scope_key == scope_key_value,
            int(candidate["revision"] or 0),
            ref.scope_key,
            ref.attempt_id,
        )

    canonical_candidate = max(candidate_rows, key=candidate_key)
    candidate = candidate_rows[canonical_candidate]
    prefer_incoming = (
        _identity_rank(attempt.identity_basis)
        > _identity_rank(str(candidate["identity_basis"]))
        and str(candidate["identity_basis"]) in {"provisional", "unresolved"}
        and candidate["observed_at"] is not None
        and seen_at >= candidate["observed_at"]
    )
    canonical_ref = incoming_ref if prefer_incoming else canonical_candidate
    merge_refs = tuple(
        sorted(
            candidate_rows.keys() - {canonical_ref},
            key=lambda item: (item.scope_key, item.attempt_id),
        )
    )
    return AttemptIdentityResolution(
        canonical_ref=canonical_ref,
        conflicts=0,
        merge_refs=merge_refs,
    )


def _attempt_alias_values_for_input(
    aliases: Sequence[tuple[str, str]],
    alias_kind: str,
) -> set[str]:
    return {value for kind, value in aliases if kind == alias_kind}


def _tombstone_attempt(
    cur: psycopg.Cursor,
    *,
    previous_ref: AttemptRef,
    canonical_ref: AttemptRef,
    seen_at: datetime,
) -> None:
    cur.execute(
        """
        UPDATE public.chatgpt_usage_attempts
        SET tombstone = TRUE,
            superseded_by_attempt_id = %s,
            updated_at = GREATEST(updated_at, %s),
            last_seen_at = GREATEST(last_seen_at, %s)
        WHERE scope_key = %s AND attempt_id = %s
        """,
        (
            canonical_ref.attempt_id,
            seen_at,
            seen_at,
            previous_ref.scope_key,
            previous_ref.attempt_id,
        ),
    )
    cur.execute(
        """
        UPDATE public.chatgpt_usage_attempt_revisions
        SET is_current_projection = FALSE
        WHERE scope_key = %s AND attempt_id = %s
        """,
        (previous_ref.scope_key, previous_ref.attempt_id),
    )


def _reassign_attempt_aliases(
    cur: psycopg.Cursor,
    *,
    previous_ref: AttemptRef,
    canonical_ref: AttemptRef,
    seen_at: datetime,
) -> None:
    cur.execute(
        """
        SELECT alias_kind, alias_value, first_seen_at, last_seen_at
        FROM public.chatgpt_usage_attempt_aliases
        WHERE scope_key = %s AND attempt_id = %s
        """,
        (previous_ref.scope_key, previous_ref.attempt_id),
    )
    aliases = cur.fetchall()
    for alias_kind, alias_value, first_seen_at, last_seen_at in aliases:
        transfer_seen_at = max(last_seen_at, seen_at)
        if previous_ref.scope_key == canonical_ref.scope_key:
            cur.execute(
                """
                UPDATE public.chatgpt_usage_attempt_aliases
                SET attempt_id = %s,
                    first_seen_at = LEAST(first_seen_at, %s),
                    last_seen_at = GREATEST(last_seen_at, %s)
                WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
                  AND attempt_id = %s
                """,
                (
                    canonical_ref.attempt_id,
                    first_seen_at,
                    transfer_seen_at,
                    previous_ref.scope_key,
                    alias_kind,
                    alias_value,
                    previous_ref.attempt_id,
                ),
            )
            continue
        cur.execute(
            """
            SELECT attempt_id
            FROM public.chatgpt_usage_attempt_aliases
            WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
            """,
            (
                canonical_ref.scope_key,
                alias_kind,
                alias_value,
            ),
        )
        target = cur.fetchone()
        if target is None:
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_attempt_aliases (
                    scope_key, alias_kind, alias_value, attempt_id,
                    first_seen_at, last_seen_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    canonical_ref.scope_key,
                    alias_kind,
                    alias_value,
                    canonical_ref.attempt_id,
                    first_seen_at,
                    transfer_seen_at,
                ),
            )
            target_attempt_id = canonical_ref.attempt_id
        else:
            target_attempt_id = str(target[0])
            if target_attempt_id == canonical_ref.attempt_id:
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempt_aliases
                    SET first_seen_at = LEAST(first_seen_at, %s),
                        last_seen_at = GREATEST(last_seen_at, %s)
                    WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
                    """,
                    (
                        first_seen_at,
                        transfer_seen_at,
                        canonical_ref.scope_key,
                        alias_kind,
                        alias_value,
                    ),
                )
        if target_attempt_id == canonical_ref.attempt_id:
            cur.execute(
                """
                DELETE FROM public.chatgpt_usage_attempt_aliases
                WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
                  AND attempt_id = %s
                """,
                (
                    previous_ref.scope_key,
                    alias_kind,
                    alias_value,
                    previous_ref.attempt_id,
                ),
            )


def _finalize_identity_merges(
    cur: psycopg.Cursor,
    *,
    canonical_ref: AttemptRef,
    merge_refs: Sequence[AttemptRef],
    seen_at: datetime,
) -> None:
    for previous_ref in merge_refs:
        if previous_ref == canonical_ref:
            continue
        _reassign_attempt_aliases(
            cur,
            previous_ref=previous_ref,
            canonical_ref=canonical_ref,
            seen_at=seen_at,
        )
        _tombstone_attempt(
            cur,
            previous_ref=previous_ref,
            canonical_ref=canonical_ref,
            seen_at=seen_at,
        )


def _record_identity_gap(
    page: PgLedgerPage,
    cur: psycopg.Cursor,
    scope: LedgerScope,
    scope_key_value: str,
    *,
    source_id: str,
    reason: str,
    details: Mapping[str, Any],
    seen_at: datetime,
) -> None:
    safe_scope = _normalize_scope(scope)
    safe_source_id = _required_token(source_id, "source_id")
    safe_reason = _required_token(reason, "reason")
    safe_details = _coverage_details_envelope(details)
    safe_seen_at = _utc_datetime(seen_at, "seen_at")
    page.ledger.assert_safe_record(
        _coverage_write_envelope(
            safe_scope,
            scope_key_value=scope_key_value,
            source_kind="attempt_identity",
            source_id=safe_source_id,
            reason=safe_reason,
            state="open",
            details=safe_details,
            seen_at=safe_seen_at,
        )
    )
    gap_id = stable_id(scope_key_value, "attempt_identity", safe_source_id, safe_reason)
    cur.execute(
        """
        INSERT INTO public.chatgpt_usage_coverage_gaps (
            gap_id, scope_key, collector_account_id, source_kind, source_id,
            reason, state, first_seen_at, last_seen_at, details
        ) VALUES (
            %s, %s, %s, 'attempt_identity', %s, %s, 'open', %s, %s, %s::jsonb
        )
        ON CONFLICT (scope_key, source_kind, source_id, reason) DO UPDATE SET
            state = CASE
                WHEN EXCLUDED.last_seen_at
                    >= public.chatgpt_usage_coverage_gaps.last_seen_at
                THEN EXCLUDED.state
                ELSE public.chatgpt_usage_coverage_gaps.state
            END,
            last_seen_at = GREATEST(
                public.chatgpt_usage_coverage_gaps.last_seen_at,
                EXCLUDED.last_seen_at
            ),
            details = CASE
                WHEN EXCLUDED.last_seen_at
                    >= public.chatgpt_usage_coverage_gaps.last_seen_at
                THEN EXCLUDED.details
                ELSE public.chatgpt_usage_coverage_gaps.details
            END
        """,
        (
            gap_id,
            scope_key_value,
            safe_scope.collector_account_id,
            safe_source_id,
            safe_reason,
            safe_seen_at,
            safe_seen_at,
            json.dumps(safe_details, separators=(",", ":"), default=str),
        ),
    )


def _next_attempt_revision(
    cur: psycopg.Cursor,
    scope_key_value: str,
    attempt_id: str,
) -> int:
    cur.execute(
        """
        SELECT COALESCE(MAX(revision), 0) + 1
        FROM public.chatgpt_usage_attempt_revisions
        WHERE scope_key = %s AND attempt_id = %s
        """,
        (scope_key_value, attempt_id),
    )
    row = cur.fetchone()
    if row is None:
        raise LedgerError("revision query returned no row")
    return int(row[0])


def _insert_attempt_head(
    cur: psycopg.Cursor,
    *,
    scope: LedgerScope,
    scope_key_value: str,
    attempt: AttemptRecord,
    attempt_id: str,
    revision: int,
    projection_fingerprint: str,
    observed_at: datetime,
    quarantine_state: str,
    warnings: Sequence[str],
) -> None:
    cur.execute(
        """
        INSERT INTO public.chatgpt_usage_attempts (
            attempt_id, scope_key, collector_account_id, conversation_id,
            identity_basis, time_basis, attempt_time, earliest_possible_at,
            latest_possible_at, requested_model_raw, requested_mode_raw,
            requested_reasoning_effort_raw, recorded_final_model_raw,
            resolved_model_raw, requested_family, recorded_final_family,
            resolved_family, mapping_version, outcome, completed_answer,
            generation_started, surface, origin, revision,
            projection_fingerprint, warnings, quarantine_state, tombstone, observed_at,
            updated_at, last_seen_at, superseded_by_attempt_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, FALSE, %s, %s,
            %s, NULL
        )
        """,
        (
            attempt_id,
            scope_key_value,
            scope.collector_account_id,
            attempt.conversation_id,
            attempt.identity_basis,
            attempt.time_basis,
            _utc_or_none(attempt.attempt_time),
            _utc_or_none(attempt.earliest_possible_at),
            _utc_or_none(attempt.latest_possible_at),
            attempt.requested_model_raw,
            attempt.requested_mode_raw,
            attempt.requested_reasoning_effort_raw,
            attempt.recorded_final_model_raw,
            attempt.resolved_model_raw,
            attempt.requested_family,
            attempt.recorded_final_family,
            attempt.resolved_family,
            attempt.mapping_version,
            attempt.outcome,
            attempt.completed_answer,
            attempt.generation_started,
            attempt.surface,
            attempt.origin,
            revision,
            projection_fingerprint,
            json.dumps(warnings, separators=(",", ":"), default=str),
            quarantine_state,
            observed_at,
            observed_at,
            observed_at,
        ),
    )


def _attempt_projection_params(
    scope: LedgerScope,
    attempt: AttemptRecord,
    revision: int,
    projection_fingerprint: str,
    observed_at: datetime,
    scope_key_value: str,
    attempt_id: str,
    quarantine_state: str,
    warnings: Sequence[str],
) -> tuple[Any, ...]:
    return (
        scope.collector_account_id,
        attempt.conversation_id,
        attempt.identity_basis,
        attempt.time_basis,
        _utc_or_none(attempt.attempt_time),
        _utc_or_none(attempt.earliest_possible_at),
        _utc_or_none(attempt.latest_possible_at),
        attempt.requested_model_raw,
        attempt.requested_mode_raw,
        attempt.requested_reasoning_effort_raw,
        attempt.recorded_final_model_raw,
        attempt.resolved_model_raw,
        attempt.requested_family,
        attempt.recorded_final_family,
        attempt.resolved_family,
        attempt.mapping_version,
        attempt.outcome,
        attempt.completed_answer,
        attempt.generation_started,
        attempt.surface,
        attempt.origin,
        revision,
        projection_fingerprint,
        json.dumps(list(warnings), separators=(",", ":"), default=str),
        observed_at,
        observed_at,
        observed_at,
        quarantine_state,
        scope_key_value,
        attempt_id,
    )


def _merge_attempt_links(
    ledger: PgLedger | PgLedgerPage,
    cur: psycopg.Cursor,
    scope: LedgerScope,
    *,
    scope_key_value: str,
    attempt_id: str,
    aliases: Sequence[tuple[str, str]],
    seen_at: datetime,
    lineage_keys: Sequence[str],
) -> AliasMergeResult:
    conflicts = 0
    quarantine_conflicts = 0
    safe_scope = _normalize_scope(scope)
    safe_aliases = _unique_aliases(aliases)
    safe_seen_at = _utc_datetime(seen_at, "seen_at")
    ledger.assert_safe_record(safe_aliases)
    conflicts_by_alias: dict[tuple[str, str], set[AttemptRef]] = {}
    for alias_kind, safe_value in safe_aliases:
        existing_refs = _matching_alias_refs(
            cur,
            lineage_keys,
            ((alias_kind, safe_value),),
        )
        existing_refs.discard(AttemptRef(scope_key_value, attempt_id))
        if existing_refs:
            conflicts_by_alias[(alias_kind, safe_value)] = existing_refs
            conflicts += 1
            quarantine_conflicts += 1 if alias_kind in _STRONG_ALIAS_KINDS else 0
    if conflicts_by_alias:
        for (alias_kind, safe_value), existing_refs in conflicts_by_alias.items():
            existing_ref = sorted(
                existing_refs,
                key=lambda item: (item.scope_key, item.attempt_id),
            )[0]
            _record_alias_collision(
                ledger,
                cur,
                safe_scope,
                alias_kind,
                safe_value,
                existing_ref=existing_ref,
                incoming_ref=AttemptRef(scope_key_value, attempt_id),
                seen_at=safe_seen_at,
            )
        return AliasMergeResult(
            conflicts=conflicts,
            quarantine_conflicts=quarantine_conflicts,
        )
    for alias_kind, safe_value in safe_aliases:
        cur.execute(
            """
            SELECT first_seen_at, last_seen_at
            FROM public.chatgpt_usage_attempt_aliases
            WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
            """,
            (scope_key_value, alias_kind, safe_value),
        )
        row = cur.fetchone()
        if row is not None:
            cur.execute(
                """
                UPDATE public.chatgpt_usage_attempt_aliases
                SET last_seen_at = GREATEST(last_seen_at, %s)
                WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
                """,
                (safe_seen_at, scope_key_value, alias_kind, safe_value),
            )
            continue
        cur.execute(
            """
            INSERT INTO public.chatgpt_usage_attempt_aliases (
                scope_key, alias_kind, alias_value, attempt_id,
                first_seen_at, last_seen_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                scope_key_value,
                alias_kind,
                safe_value,
                attempt_id,
                safe_seen_at,
                safe_seen_at,
            ),
        )
    return AliasMergeResult(
        conflicts=conflicts,
        quarantine_conflicts=quarantine_conflicts,
    )


def _record_alias_collision(
    ledger: PgLedger | PgLedgerPage,
    cur: psycopg.Cursor,
    scope: LedgerScope,
    alias_kind: str,
    alias_value: str,
    *,
    existing_ref: AttemptRef,
    incoming_ref: AttemptRef,
    seen_at: datetime,
) -> None:
    safe_scope = _normalize_scope(scope)
    safe_alias_kind = _required_token(alias_kind, "alias_kind")
    safe_alias_value = _required_token(alias_value, "alias_value")
    safe_seen_at = _utc_datetime(seen_at, "seen_at")
    key = scope_key(safe_scope)
    source_id = f"{safe_alias_kind}:{safe_alias_value}"
    gap_id = stable_id(key, "attempt_alias", source_id, "alias_collision")
    details = _coverage_details_envelope(
        {
            "alias": {
                "kind": safe_alias_kind,
                "value": safe_alias_value,
            },
            "participants": [
                _attempt_participant(existing_ref, role="existing"),
                _attempt_participant(incoming_ref, role="incoming"),
            ],
        }
    )
    ledger.assert_safe_record(
        _coverage_write_envelope(
            safe_scope,
            scope_key_value=key,
            source_kind="attempt_alias",
            source_id=source_id,
            reason="alias_collision",
            state="open",
            details=details,
            seen_at=safe_seen_at,
        )
    )
    cur.execute(
        """
        INSERT INTO public.chatgpt_usage_coverage_gaps (
            gap_id, scope_key, collector_account_id, source_kind, source_id,
            reason, state, first_seen_at, last_seen_at, details
        ) VALUES (
            %s, %s, %s, 'attempt_alias', %s, 'alias_collision', 'open', %s, %s,
            %s::jsonb
        )
        ON CONFLICT (scope_key, source_kind, source_id, reason) DO UPDATE SET
            state = CASE
                WHEN EXCLUDED.last_seen_at
                    >= public.chatgpt_usage_coverage_gaps.last_seen_at
                THEN EXCLUDED.state
                ELSE public.chatgpt_usage_coverage_gaps.state
            END,
            last_seen_at = GREATEST(
                public.chatgpt_usage_coverage_gaps.last_seen_at,
                EXCLUDED.last_seen_at
            ),
            details = CASE
                WHEN EXCLUDED.last_seen_at
                    >= public.chatgpt_usage_coverage_gaps.last_seen_at
                THEN EXCLUDED.details
                ELSE public.chatgpt_usage_coverage_gaps.details
            END
        """,
        (
            gap_id,
            key,
            safe_scope.collector_account_id,
            source_id,
            safe_seen_at,
            safe_seen_at,
            json.dumps(details, separators=(",", ":"), default=str),
        ),
    )


def _record_activity_provenance(
    cur: psycopg.Cursor,
    scope_key_value: str,
    activity_kind: str,
    activity_id: str,
    collector_account_id: str,
    seen_at: datetime,
) -> None:
    cur.execute(
        """
        INSERT INTO public.chatgpt_usage_activity_provenance (
            scope_key, activity_kind, activity_id, collector_account_id,
            first_seen_at, last_seen_at
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (
            scope_key, activity_kind, activity_id, collector_account_id
        ) DO UPDATE SET
            first_seen_at = LEAST(
                public.chatgpt_usage_activity_provenance.first_seen_at,
                EXCLUDED.first_seen_at
            ),
            last_seen_at = GREATEST(
                public.chatgpt_usage_activity_provenance.last_seen_at,
                EXCLUDED.last_seen_at
            )
        """,
        (
            scope_key_value,
            activity_kind,
            activity_id,
            collector_account_id,
            ensure_utc(seen_at),
            ensure_utc(seen_at),
        ),
    )


def _split_sql_statements(sql: str) -> list[str]:
    statements = [statement.strip() for statement in sql.split(";")]
    return [statement for statement in statements if statement and statement.upper() not in {"BEGIN", "COMMIT"}]


def _lock_scope(cur: psycopg.Cursor, key: str) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)", (key,))


def _scope_lineage(cur: psycopg.Cursor, scope_key_value: str) -> tuple[str, ...]:
    cur.execute(
        """
        WITH RECURSIVE lineage(scope_key) AS (
            SELECT %s
            UNION
            SELECT redirects.retired_scope_key
            FROM public.chatgpt_usage_scope_redirects AS redirects
            JOIN lineage
              ON lineage.scope_key = redirects.canonical_scope_key
        )
        SELECT scope_key
        FROM lineage
        ORDER BY scope_key
        """,
        (scope_key_value,),
    )
    return tuple(str(row[0]) for row in cur.fetchall())


def _lock_scope_lineage(cur: psycopg.Cursor, scope_keys: Sequence[str]) -> None:
    for scope_key_value in sorted(set(scope_keys)):
        _lock_scope(cur, scope_key_value)


def _lock_scope_registry(cur: psycopg.Cursor) -> None:
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
        ("chatgpt-usage-scope-registry",),
    )


def _assert_active_binding(
    cur: psycopg.Cursor,
    *,
    collector_account_id: str,
    scope_key_value: str,
    binding_generation: int,
) -> None:
    cur.execute(
        """
        SELECT scope_key, binding_generation, binding_state
        FROM public.chatgpt_usage_scope_bindings
        WHERE collector_account_id = %s AND binding_state = 'active'
        FOR UPDATE
        """,
        (collector_account_id,),
    )
    row = cur.fetchone()
    if row is None or (
        str(row[0]) != scope_key_value or int(row[1]) != int(binding_generation) or str(row[2]) != "active"
    ):
        raise LedgerError("scope binding fence is stale")


def _assert_expected_binding_still_active(
    cur: psycopg.Cursor,
    scope: LedgerScope,
    expected: LedgerBinding,
) -> None:
    cur.execute(
        """
        SELECT scope_key, binding_generation
        FROM public.chatgpt_usage_scope_bindings
        WHERE collector_account_id = %s AND binding_state = 'active'
        FOR UPDATE
        """,
        (scope.collector_account_id,),
    )
    row = cur.fetchone()
    if row is None or (
        str(row[0]) != expected.scope_key
        or int(row[1]) != int(expected.binding_generation)
    ):
        raise LedgerError("scope binding fence is stale")


def _scope_row_identity_state(cur: psycopg.Cursor, scope_key_value: str) -> str:
    cur.execute(
        """
        SELECT identity_state
        FROM public.chatgpt_usage_scopes
        WHERE scope_key = %s
        FOR UPDATE
        """,
        (scope_key_value,),
    )
    row = cur.fetchone()
    if row is None:
        raise LedgerError("scope binding refers to a missing scope")
    return str(row[0])


def _unique_aliases(aliases: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return _sanitize_aliases(aliases)


def _attempt_payload(
    attempt: AttemptRecord,
    aliases: Sequence[tuple[str, str]],
    *,
    attempt_id: Optional[str] = None,
    quarantine_reason: Optional[str] = None,
) -> dict[str, Any]:
    warnings = list(_safe_warning_tokens(attempt.warnings))
    normalized_quarantine = _normalize_quarantine(attempt.quarantine)
    for warning in normalized_quarantine["warnings"]:
        if warning not in warnings:
            warnings.append(warning)
    if quarantine_reason is not None:
        quarantine_warning = f"quarantine:{quarantine_reason}"
        if quarantine_warning not in warnings:
            warnings.append(quarantine_warning)
    warnings = _safe_warning_tokens(warnings)
    return {
        "transferSchemaVersion": TRANSFER_SCHEMA_VERSION,
        "attemptId": attempt_id or attempt.attempt_id,
        "conversationId": attempt.conversation_id,
        "identityBasis": attempt.identity_basis,
        "timeBasis": attempt.time_basis,
        "attemptTime": isoformat_utc(attempt.attempt_time),
        "earliestPossibleAt": isoformat_utc(attempt.earliest_possible_at),
        "latestPossibleAt": isoformat_utc(attempt.latest_possible_at),
        "requestedModelRaw": attempt.requested_model_raw,
        "requestedModeRaw": attempt.requested_mode_raw,
        "requestedReasoningEffortRaw": attempt.requested_reasoning_effort_raw,
        "recordedFinalModelRaw": attempt.recorded_final_model_raw,
        "resolvedModelRaw": attempt.resolved_model_raw,
        "requestedFamily": attempt.requested_family,
        "recordedFinalFamily": attempt.recorded_final_family,
        "resolvedFamily": attempt.resolved_family,
        "mappingVersion": attempt.mapping_version,
        "outcome": attempt.outcome,
        "completedAnswer": attempt.completed_answer,
        "generationStarted": attempt.generation_started,
        "surface": attempt.surface,
        "origin": attempt.origin,
        "aliases": [list(alias) for alias in sorted(aliases)],
        "evidenceMessageIds": sorted(attempt.evidence_message_ids),
        "warnings": warnings,
        "quarantine": _quarantine_envelope(warnings, normalized_quarantine),
    }


def _attempt_quarantine_state(payload: Mapping[str, Any]) -> str:
    quarantine = payload.get("quarantine")
    if not isinstance(quarantine, Mapping):
        return "unknown"
    state = _quarantine_state_token(quarantine.get("state"))
    return state or "unknown"


def _scope_payload(scope: LedgerScope) -> dict[str, Any]:
    return {
        "collector_account_id": scope.collector_account_id,
        "provider": scope.provider,
        "provider_user_id": scope.provider_user_id,
        "workspace_id": scope.workspace_id,
        "quota_owner_id": scope.quota_owner_id,
        "surface": scope.surface,
        "identity_state": _scope_identity_state(scope),
    }


def _context_payload(context: IngestContext) -> dict[str, Any]:
    return {
        "run_id": context.run_id,
        "observed_at": isoformat_utc(context.observed_at),
        "source_kind": context.source_kind,
        "source_id": context.source_id,
        "schema_version": context.schema_version,
        "provenance": sanitize_provenance(context.provenance),
    }


def _prepared_observation_record(
    scope: LedgerScope,
    context: IngestContext,
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "scope": _scope_payload(scope),
        "context": _context_payload(context),
        "payload": dict(payload),
        "provenance": dict(provenance),
    }


def _prepared_attempt_record(
    scope: LedgerScope,
    context: IngestContext,
    payload: Mapping[str, Any],
    aliases: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "scope": _scope_payload(scope),
        "context": _context_payload(context),
        "payload": dict(payload),
        "aliases": [list(alias) for alias in aliases],
        "warnings": list(payload.get("warnings", [])),
        "provenance": sanitize_provenance(context.provenance),
    }


def _normalize_scope(scope: LedgerScope) -> LedgerScope:
    if not isinstance(scope, LedgerScope):
        raise LedgerError("scope must be a LedgerScope")
    return LedgerScope(
        collector_account_id=_required_token(scope.collector_account_id, "collector_account_id"),
        provider=_required_token(scope.provider, "provider"),
        provider_user_id=_optional_token(scope.provider_user_id),
        workspace_id=_optional_token(scope.workspace_id),
        quota_owner_id=_optional_token(scope.quota_owner_id),
        surface=_surface_token(scope.surface),
    )


def _normalize_context(context: IngestContext) -> IngestContext:
    if not isinstance(context, IngestContext):
        raise LedgerError("context must be an IngestContext")
    if context.provenance is not None and not isinstance(context.provenance, Mapping):
        raise LedgerError("provenance must be a mapping")
    return IngestContext(
        run_id=_required_token(context.run_id, "run_id"),
        observed_at=_utc_datetime(context.observed_at, "observed_at"),
        source_kind=_required_token(context.source_kind, "source_kind"),
        source_id=_required_token(context.source_id, "source_id"),
        schema_version=_required_token(context.schema_version, "schema_version"),
        provenance=sanitize_provenance(context.provenance),
    )


def _normalize_attempt(
    attempt: AttemptRecord,
    aliases: Sequence[tuple[str, str]],
) -> AttemptRecord:
    if not isinstance(attempt, AttemptRecord):
        raise LedgerError("attempt must be an AttemptRecord")
    try:
        revision = int(attempt.revision)
    except (TypeError, ValueError) as exc:
        raise LedgerError("attempt revision must be an integer") from exc
    if revision < 1:
        raise LedgerError("attempt revision must be positive")
    return AttemptRecord(
        attempt_id=_required_token(attempt.attempt_id, "attempt_id"),
        conversation_id=_required_token(attempt.conversation_id, "conversation_id"),
        identity_basis=_enum_token(attempt.identity_basis, _ALLOWED_IDENTITY_BASES, "unknown"),
        time_basis=_enum_token(attempt.time_basis, _ALLOWED_TIME_BASES, "unknown"),
        attempt_time=_utc_or_none(attempt.attempt_time),
        earliest_possible_at=_utc_or_none(attempt.earliest_possible_at),
        latest_possible_at=_utc_or_none(attempt.latest_possible_at),
        requested_model_raw=_optional_token(attempt.requested_model_raw),
        requested_mode_raw=_optional_token(attempt.requested_mode_raw),
        requested_reasoning_effort_raw=_optional_token(attempt.requested_reasoning_effort_raw),
        recorded_final_model_raw=_optional_token(attempt.recorded_final_model_raw),
        resolved_model_raw=_optional_token(attempt.resolved_model_raw),
        requested_family=_optional_token(attempt.requested_family),
        recorded_final_family=_optional_token(attempt.recorded_final_family),
        resolved_family=_optional_token(attempt.resolved_family),
        mapping_version=_required_token(attempt.mapping_version, "mapping_version"),
        outcome=_enum_token(attempt.outcome, _ALLOWED_OUTCOMES, "unknown"),
        completed_answer=_required_bool(attempt.completed_answer, "completed_answer"),
        generation_started=_required_bool(attempt.generation_started, "generation_started"),
        surface=_surface_token(attempt.surface),
        origin=_origin_token(attempt.origin),
        aliases=tuple(aliases),
        evidence_message_ids=tuple(
            token for token in (_optional_token(value) for value in attempt.evidence_message_ids) if token is not None
        ),
        revision=revision,
        warnings=tuple(_safe_warning_tokens(attempt.warnings)),
        quarantine=_normalize_quarantine(attempt.quarantine),
    )


def _required_token(value: Any, field_name: str) -> str:
    token = _optional_token(value)
    if token is None:
        raise LedgerError(f"{field_name} must be a safe metadata token")
    return token


def _required_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise LedgerError(f"{field_name} must be a boolean")
    return value


def _utc_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise LedgerError(f"{field_name} must be a datetime")
    return ensure_utc(value)


def _operation_deadline(
    operation: Any,
    deadline_at: Optional[datetime],
) -> Optional[datetime]:
    explicit = (
        _utc_datetime(deadline_at, "deadline_at")
        if deadline_at is not None
        else None
    )
    if operation is None:
        return explicit
    operation_deadline = getattr(operation, "deadline_at", None)
    if operation_deadline is None:
        raise LedgerError("collector operation deadline is required")
    normalized = _utc_datetime(operation_deadline, "operation.deadline_at")
    if explicit is None:
        return normalized
    return min(explicit, normalized)


def _check_operation(operation: Any) -> None:
    if operation is None:
        return
    if not bool(getattr(operation, "ignore_cancel", False)):
        checker = getattr(operation, "check", None)
        if callable(checker):
            checker()
        else:
            cancel_event = getattr(operation, "cancel_event", None)
            if cancel_event is not None and cancel_event.is_set():
                raise LedgerError("collector operation was cancelled")
    deadline_monotonic = getattr(operation, "deadline_monotonic", None)
    if isinstance(deadline_monotonic, (int, float)) and not isinstance(
        deadline_monotonic,
        bool,
    ):
        if monotonic() >= deadline_monotonic:
            raise LedgerError("collector database deadline has expired")


def _assert_before_deadline(deadline_at: datetime, operation: Any = None) -> None:
    operation_wall_deadline = getattr(operation, "deadline_at", None)
    if (
        isinstance(operation_wall_deadline, datetime)
        and ensure_utc(deadline_at) == ensure_utc(operation_wall_deadline)
    ):
        _check_operation(operation)
        return
    if datetime.now().astimezone() >= deadline_at:
        raise LedgerError("collector database deadline has expired")


def _remaining_deadline_seconds(
    deadline_at: datetime,
    operation: Any = None,
) -> float:
    if operation is not None:
        remaining = _operation_wait_deadline(deadline_at, operation) - monotonic()
    else:
        remaining = (deadline_at - datetime.now().astimezone()).total_seconds()
    if remaining <= 0:
        raise LedgerError("collector database deadline has expired")
    return remaining


def _operation_wait_deadline(
    deadline_at: datetime,
    operation: Any = None,
) -> float:
    operation_deadline = getattr(operation, "deadline_monotonic", None)
    operation_wall_deadline = getattr(operation, "deadline_at", None)
    if (
        isinstance(operation_deadline, (int, float))
        and not isinstance(operation_deadline, bool)
        and isinstance(operation_wall_deadline, datetime)
        and ensure_utc(deadline_at) == ensure_utc(operation_wall_deadline)
    ):
        if monotonic() >= operation_deadline:
            raise LedgerError("collector database deadline has expired")
        return float(operation_deadline)
    return monotonic() + _remaining_deadline_seconds(deadline_at)


def _wait_operation(
    generator: Any,
    socket: int,
    deadline_at: datetime,
    *,
    operation: Any = None,
) -> Any:
    """Consume one Psycopg operation generator with an owned absolute deadline."""
    wait_deadline = _operation_wait_deadline(deadline_at, operation)
    try:
        _check_operation(operation)
        wait = next(generator)
        with selectors.DefaultSelector() as selector:
            while True:
                _check_operation(operation)
                remaining = wait_deadline - monotonic()
                if remaining <= 0:
                    raise LedgerError("collector database deadline has expired")
                selector.register(socket, int(wait))
                try:
                    ready = selector.select(timeout=remaining)
                finally:
                    selector.unregister(socket)
                if not ready:
                    raise LedgerError("collector database deadline has expired")
                _check_operation(operation)
                wait = generator.send(waiting.Ready(ready[0][1]))
    except StopIteration as exc:
        if monotonic() >= wait_deadline:
            raise LedgerError("collector database deadline has expired") from None
        return exc.value
    except BaseException:
        try:
            generator.close()
        except BaseException:
            pass
        raise


def _remaining_deadline_ms(deadline_at: datetime, operation: Any = None) -> int:
    remaining_ms = int(_remaining_deadline_seconds(deadline_at, operation) * 1000)
    if remaining_ms <= 0:
        raise LedgerError("collector database deadline has expired")
    return remaining_ms


def _optional_token(value: Any) -> Optional[str]:
    return sanitize_token(value)


def _enum_token(value: Any, allowed: Sequence[str], default: str) -> str:
    token = _optional_token(value)
    return token if token in allowed else default


def _surface_token(value: Any) -> str:
    token = _optional_token(value)
    return token if token in _ALLOWED_SURFACES else "unknown"


def _origin_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = _optional_token(value)
    return token if token in _ALLOWED_ORIGINS else "unknown"


def _safe_tokens(values: Iterable[Any], *, limit: int = 64) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)):
        return []
    out: list[str] = []
    for value in values:
        token = _optional_token(value)
        if token is not None and token not in out:
            out.append(token)
        if len(out) >= limit:
            break
    return out


def _is_quarantine_warning(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return (
        normalized == "future_timestamp_quarantined"
        or normalized.startswith("future_timestamp")
        or normalized.startswith("quarantine:")
    )


def _safe_warning_tokens(values: Iterable[Any], *, limit: int = 64) -> list[str]:
    """Retain safety warnings before applying the ordinary warning cap."""
    if isinstance(values, (str, bytes, bytearray)):
        return []
    ordinary: list[str] = []
    safety: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = _optional_token(value)
        if token is None or token in seen:
            continue
        seen.add(token)
        if _is_quarantine_warning(token):
            safety.append(token)
        elif len(ordinary) < limit:
            ordinary.append(token)
    if len(safety) >= limit:
        return sorted(safety[:limit])
    return sorted(safety + ordinary[: limit - len(safety)])


def _sanitize_aliases(
    aliases: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in aliases:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        alias_kind = _optional_token(item[0])
        alias_value = _optional_token(item[1])
        if alias_kind not in _ALLOWED_ALIAS_KINDS or alias_value is None:
            continue
        pair = (alias_kind, alias_value)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _looks_like_page(payload: Mapping[str, Any]) -> bool:
    page_markers = {"items", "messages", "mapping", "page_info", "continuation", "exhausted"}
    for index, raw_key in enumerate(payload):
        if index >= _OBSERVATION_MAX_FIELDS:
            break
        key = _OBSERVATION_KEY_ALIASES.get(str(raw_key), str(raw_key))
        if key in page_markers:
            return True
    return False


def _observation_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise LedgerError("observation payload must be a mapping")
    state = _ObservationProjectionState()
    projected, unknown_field_count, source_count = _project_observation_root(payload, state)
    _finish_observation_projection(
        payload,
        projected,
        state,
        unknown_field_count=unknown_field_count,
        source_count=source_count,
    )
    return projected


def _project_observation_root(
    payload: Mapping[str, Any],
    state: _ObservationProjectionState,
) -> tuple[dict[str, Any], int, Optional[int]]:
    projected: dict[str, Any] = {"transfer_schema_version": TRANSFER_SCHEMA_VERSION}
    unknown_field_count = 0
    source_count = _safe_len(payload)
    root_fields = _PAGE_FIELDS if _looks_like_page(payload) else _OBSERVATION_FIELDS
    for index, (raw_key, raw_value) in enumerate(payload.items()):
        if index >= _OBSERVATION_MAX_FIELDS:
            state.mark_truncated(
                path="$",
                reason="max_fields",
                retained_count=len(projected),
                source_count=source_count,
                source_count_lower_bound=source_count is None,
            )
            break
        key = _OBSERVATION_KEY_ALIASES.get(str(raw_key), str(raw_key))
        if key not in root_fields:
            unknown_field_count += 1
            continue
        value = _project_observation_field(
            key,
            raw_value,
            state=state,
            path=f"$.{key}",
            shape="root",
            depth=0,
        )
        if value is not _DROP:
            projected[key] = value
    return projected, unknown_field_count, source_count


def _finish_observation_projection(
    payload: Mapping[str, Any],
    projected: dict[str, Any],
    state: _ObservationProjectionState,
    *,
    unknown_field_count: int,
    source_count: Optional[int],
) -> None:
    surface = classify_surface(payload, default=None)
    projected["surface"] = _surface_token(projected.get("surface", surface))
    projected["unknown_field_count"] = min(unknown_field_count, _OBSERVATION_MAX_FIELDS)
    if source_count is not None and source_count > _OBSERVATION_MAX_FIELDS:
        projected["unknown_field_count_lower_bound"] = True
    _record_observation_collection_counts(projected)
    _apply_observation_quarantine(projected)
    _apply_transfer_version_status(projected)
    _apply_projection_status(projected, state)


def _record_observation_collection_counts(projected: dict[str, Any]) -> None:
    for collection_key, count_key in (
        ("items", "item_count"),
        ("messages", "message_count"),
        ("mapping", "mapping_node_count"),
    ):
        value = projected.get(collection_key)
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
            projected[count_key] = len(value)


def _apply_observation_quarantine(projected: dict[str, Any]) -> None:
    quarantine = projected.get("quarantine")
    safety_warnings: list[str] = []
    for warning_key in ("warnings", "quarantine_warnings"):
        warning_values = projected.get(warning_key)
        if isinstance(warning_values, list):
            safety_warnings.extend(
                warning
                for warning in warning_values
                if isinstance(warning, str) and _is_quarantine_warning(warning)
            )
    future_timestamp_quarantined = projected.get("future_timestamp_quarantined") is True
    if future_timestamp_quarantined or safety_warnings or isinstance(quarantine, Mapping):
        envelope = _quarantine_envelope(
            safety_warnings,
            quarantine if isinstance(quarantine, Mapping) else None,
            future_timestamp_quarantined=future_timestamp_quarantined,
        )
        projected["quarantine"] = envelope
        projected["quarantine_state"] = envelope["state"]


def _apply_transfer_version_status(projected: dict[str, Any]) -> None:
    source_transfer_version = projected.get("transfer_version")
    if source_transfer_version is None or source_transfer_version == TRANSFER_SCHEMA_VERSION:
        return
    projected["projection_status"] = "unsupported_transfer_version"
    projected["projection_error"] = f"unsupported_transfer_version:{source_transfer_version}"
    projected["coverage"] = "unrecognized"


def _apply_projection_status(
    projected: dict[str, Any],
    state: _ObservationProjectionState,
) -> None:
    upstream_truncated = projected.get("projection_truncated") is True
    upstream_incomplete = projected.get("projection_incomplete") is True
    upstream_reasons = projected.get("projection_truncations")
    if not isinstance(upstream_reasons, list):
        upstream_reasons = []
    truncations = list(upstream_reasons)
    for truncation in state.truncations:
        if truncation not in truncations:
            truncations.append(truncation)
    has_truncation = upstream_truncated or bool(truncations)
    if has_truncation:
        projected["projection_truncated"] = True
        projected["projection_truncations"] = truncations[:_OBSERVATION_MAX_FIELDS]
        if projected.get("coverage") != "unrecognized":
            projected["coverage"] = "partial"
        if projected.get("exhausted") is True:
            projected["exhausted"] = False
        projected.setdefault("projection_status", "partial")
    elif "projection_truncated" not in projected:
        projected["projection_truncated"] = False
    if "projection_truncations" not in projected:
        projected["projection_truncations"] = []
    if state.incomplete or upstream_incomplete:
        projected["projection_incomplete"] = True
    if state.invalid:
        projected["coverage"] = "unrecognized"
        projected["projection_status"] = "unrecognized"
    elif (state.incomplete or upstream_incomplete) and projected.get("coverage") != "unrecognized":
        projected["coverage"] = "partial"
        projected.setdefault("projection_status", "partial")
    elif has_truncation and projected.get("projection_status") != "unsupported_transfer_version":
        projected["projection_status"] = "partial"


def _project_observation_field(
    key: str,
    value: Any,
    *,
    state: Optional[_ObservationProjectionState] = None,
    path: str = "$",
    shape: str = "root",
    depth: int = 0,
) -> Any:
    projection_state = state or _ObservationProjectionState()
    if depth > _OBSERVATION_MAX_DEPTH:
        projection_state.mark_truncated(path=path, reason="max_depth", source_count_lower_bound=True)
        return _DROP
    if key in {"surface", "origin"}:
        return _surface_token(value) if key == "surface" else _origin_token(value)
    if key == "continuation":
        projected = _continuation_token(value)
        if projected is _DROP:
            projection_state.mark_incomplete()
        return projected
    if key in _OBSERVATION_TIMESTAMP_FIELDS:
        projected = _safe_timestamp(value)
        if projected is _DROP:
            projection_state.mark_incomplete()
        return projected
    if key in _OBSERVATION_BOOLEAN_FIELDS:
        if value is None and key in _OBSERVATION_NULLABLE_BOOLEAN_FIELDS:
            return None
        if not isinstance(value, bool):
            projection_state.mark_incomplete()
            return _DROP
        return value
    if key in _OBSERVATION_NUMBER_FIELDS:
        projected = _safe_number(value)
        if projected is _DROP:
            projection_state.mark_incomplete()
        return projected
    if key in _OBSERVATION_COLLECTION_FIELDS:
        return _observation_collection_envelope(
            key,
            value,
            state=projection_state,
            path=path,
            depth=depth + 1,
        )
    if key in _OBSERVATION_WARNING_FIELDS:
        return _bounded_token_sequence(
            value,
            state=projection_state,
            path=path,
            limit=64,
            prioritize_safety=True,
        )
    if key == "provenance":
        if not isinstance(value, Mapping):
            projection_state.mark_incomplete()
            return _DROP
        return sanitize_provenance(value)
    if key == "quarantine_state":
        return _quarantine_state_token(value)
    if key == "transfer_version":
        projected = _transfer_version_token(value)
        if projected is _DROP:
            projection_state.mark_incomplete()
        return projected
    if key in _OBSERVATION_LIST_FIELDS:
        return _bounded_token_sequence(
            value,
            state=projection_state,
            path=path,
            limit=_OBSERVATION_MAX_ITEMS,
        )
    if key == "metadata":
        return _bounded_metadata_projection(value, state=projection_state, path=path)
    if key == "author":
        return _project_author(value, state=projection_state, path=path, depth=depth + 1)
    if key == "page_info":
        return _project_shape_object(
            value,
            _PAGE_INFO_FIELDS,
            state=projection_state,
            path=path,
            shape="page_info",
            depth=depth + 1,
        )
    if key == "message":
        return _project_message(
            value,
            state=projection_state,
            path=path,
            depth=depth + 1,
        )
    if key == "quarantine":
        return _quarantine_value(value)
    if key == "transfer_schema_version":
        return TRANSFER_SCHEMA_VERSION
    if key == "projection_truncated":
        if isinstance(value, bool):
            return value
        projection_state.mark_incomplete()
        return _DROP
    if key == "projection_incomplete":
        if isinstance(value, bool):
            return value
        projection_state.mark_incomplete()
        return _DROP
    if key == "projection_truncations":
        return _bounded_projection_truncations(value, state=projection_state, path=path)
    if key in _OBSERVATION_TOKEN_FIELDS:
        projected = _optional_token(value)
        if value is not None and projected is None:
            projection_state.mark_incomplete()
        return projected
    return _DROP


def _observation_collection_envelope(
    key: str,
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
    depth: int,
) -> Any:
    if key == "items":
        return _bounded_object_sequence(
            value,
            projector=lambda item, item_path: _project_collection_item(
                item,
                state=state,
                path=item_path,
                depth=depth,
            ),
            state=state,
            path=path,
            collection_name=key,
        )
    if key == "messages":
        return _bounded_object_sequence(
            value,
            projector=lambda item, item_path: _project_message(
                item,
                state=state,
                path=item_path,
                depth=depth,
            ),
            state=state,
            path=path,
            collection_name=key,
        )
    if key == "mapping":
        return _bounded_mapping(
            value,
            projector=lambda item, item_path: _project_mapping_node(
                item,
                state=state,
                path=item_path,
                depth=depth,
            ),
            state=state,
            path=path,
            collection_name=key,
        )
    return _DROP


def _project_collection_item(
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
    depth: int,
) -> Any:
    if not isinstance(value, Mapping):
        state.mark_incomplete()
        return _DROP
    normalized_keys: set[str] = set()
    source_count = _safe_len(value)
    for index, raw_key in enumerate(value):
        if index >= _OBSERVATION_MAX_FIELDS:
            state.mark_truncated(
                path=path,
                reason="max_fields",
                retained_count=len(normalized_keys),
                source_count=source_count,
                source_count_lower_bound=source_count is None,
            )
            break
        normalized_keys.add(_OBSERVATION_KEY_ALIASES.get(str(raw_key), str(raw_key)))
    if "message" in normalized_keys or "children" in normalized_keys:
        return _project_mapping_node(value, state=state, path=path, depth=depth)
    if normalized_keys.intersection(
        {
            "message_id",
            "node_id",
            "role",
            "author",
            "end_turn",
            "requested_mode_raw",
            "requested_reasoning_effort_raw",
        }
    ):
        return _project_message(value, state=state, path=path, depth=depth)
    return _project_summary(value, state=state, path=path, depth=depth)


def _project_summary(
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
    depth: int,
) -> Any:
    return _project_shape_object(
        value,
        _SUMMARY_FIELDS,
        state=state,
        path=path,
        shape="summary",
        depth=depth,
    )


def _project_message(
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
    depth: int,
) -> Any:
    return _project_shape_object(
        value,
        _MESSAGE_FIELDS,
        state=state,
        path=path,
        shape="message",
        depth=depth,
    )


def _project_mapping_node(
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
    depth: int,
) -> Any:
    return _project_shape_object(
        value,
        _MAPPING_NODE_FIELDS,
        state=state,
        path=path,
        shape="mapping_node",
        depth=depth,
    )


def _project_shape_object(
    value: Any,
    fields: frozenset[str],
    *,
    state: _ObservationProjectionState,
    path: str,
    shape: str,
    depth: int,
) -> Any:
    if depth > _OBSERVATION_MAX_DEPTH:
        state.mark_truncated(path=path, reason="max_depth", source_count_lower_bound=True)
        return _DROP
    if not isinstance(value, Mapping):
        state.mark_incomplete()
        return _DROP
    source_count = _safe_len(value)
    projected: dict[str, Any] = {}
    for index, (raw_key, raw_value) in enumerate(value.items()):
        if index >= _OBSERVATION_MAX_FIELDS:
            state.mark_truncated(
                path=path,
                reason="max_fields",
                retained_count=len(projected),
                source_count=source_count,
                source_count_lower_bound=source_count is None,
            )
            break
        key = _OBSERVATION_KEY_ALIASES.get(str(raw_key), str(raw_key))
        if key not in fields:
            continue
        child = _project_observation_field(
            key,
            raw_value,
            state=state,
            path=f"{path}.{key}",
            shape=shape,
            depth=depth + 1,
        )
        if child is not _DROP:
            projected[key] = child
    return projected


def _project_author(
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
    depth: int,
) -> Any:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        state.mark_incomplete()
        return _DROP
    role = _optional_token(value.get("role"))
    if value.get("role") is not None and role is None:
        state.mark_incomplete()
    return {"role": role}


def _page_info_envelope(value: Any) -> Any:
    state = _ObservationProjectionState()
    return _project_shape_object(
        value,
        _PAGE_INFO_FIELDS,
        state=state,
        path="$.page_info",
        shape="page_info",
        depth=0,
    )


def _bounded_object_sequence(
    value: Any,
    *,
    projector: Any,
    state: _ObservationProjectionState,
    path: str,
    collection_name: str,
) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        state.mark_incomplete()
        return _DROP
    source_count = _safe_len(value)
    out: list[Any] = []
    for index, item in enumerate(value):
        if index >= _OBSERVATION_MAX_ITEMS:
            state.mark_truncated(
                path=path,
                reason="max_items",
                retained_count=len(out),
                source_count=source_count,
                source_count_lower_bound=source_count is None,
            )
            break
        projected = projector(item, f"{path}[{index}]")
        if projected is not _DROP:
            out.append(projected)
    return out


def _bounded_mapping(
    value: Any,
    *,
    projector: Any,
    state: _ObservationProjectionState,
    path: str,
    collection_name: str,
) -> Any:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        state.mark_incomplete()
        return _DROP
    source_count = _safe_len(value)
    out: dict[str, Any] = {}
    for index, (raw_key, item) in enumerate(value.items()):
        if index >= _OBSERVATION_MAX_ITEMS:
            state.mark_truncated(
                path=path,
                reason="max_items",
                retained_count=len(out),
                source_count=source_count,
                source_count_lower_bound=source_count is None,
            )
            break
        safe_key = _optional_token(str(raw_key))
        if safe_key is None:
            state.mark_incomplete()
            continue
        projected = projector(item, f"{path}.{safe_key}")
        if projected is not _DROP:
            out[safe_key] = projected
    return out


def _bounded_token_sequence(
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
    limit: int,
    prioritize_safety: bool = False,
) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        state.mark_incomplete()
        return _DROP
    source_count = _safe_len(value)
    if prioritize_safety:
        for item in value:
            if item is not None and _optional_token(item) is None:
                state.mark_incomplete()
        projected = _safe_warning_tokens(value, limit=limit)
        if source_count is not None and source_count > limit:
            state.mark_truncated(
                path=path,
                reason="max_items",
                retained_count=len(projected),
                source_count=source_count,
                source_count_lower_bound=False,
            )
        return projected
    out: list[str] = []
    for index, item in enumerate(value):
        if index >= limit:
            state.mark_truncated(
                path=path,
                reason="max_items",
                retained_count=len(out),
                source_count=source_count,
                source_count_lower_bound=source_count is None,
            )
            break
        token = _optional_token(item)
        if token is not None:
            out.append(token)
        elif item is not None:
            state.mark_incomplete()
    return out


def _bounded_metadata_projection(
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
) -> Any:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        state.mark_incomplete()
        return _DROP
    source_count = _safe_len(value)
    out: dict[str, Any] = {}
    for index, (raw_key, raw_value) in enumerate(value.items()):
        if index >= _OBSERVATION_MAX_FIELDS:
            state.mark_truncated(
                path=path,
                reason="max_fields",
                retained_count=len(out),
                source_count=source_count,
                source_count_lower_bound=source_count is None,
            )
            break
        key = _OBSERVATION_KEY_ALIASES.get(str(raw_key), str(raw_key))
        projected = sanitize_metadata({key: raw_value})
        if key in projected:
            out[key] = projected[key]
    return out


def _bounded_projection_truncations(
    value: Any,
    *,
    state: _ObservationProjectionState,
    path: str,
) -> Any:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        state.mark_incomplete()
        return _DROP
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if index >= _OBSERVATION_MAX_FIELDS:
            state.mark_truncated(
                path=path,
                reason="max_items",
                retained_count=len(out),
                source_count=_safe_len(value),
                source_count_lower_bound=False,
            )
            break
        if not isinstance(item, Mapping):
            state.mark_incomplete()
            continue
        record: dict[str, Any] = {}
        for key in ("path", "reason"):
            token = _optional_token(item.get(key))
            if key == "path" and token is None and isinstance(item.get(key), str):
                token = _projection_path_token(item[key])
            if token is not None:
                record[key] = token
        for key in ("retained_count", "source_count"):
            number = _safe_number(item.get(key))
            if number is not _DROP:
                record[key] = number
        if item.get("source_count_lower_bound") is True:
            record["source_count_lower_bound"] = True
        if "path" in record and "reason" in record:
            out.append(record)
    return out


def _projection_path_token(value: str) -> Optional[str]:
    """Validate projector paths without treating path punctuation as identity syntax."""
    if len(value) > 256 or not value.startswith("$."):
        return None

    position = 2
    while position < len(value):
        field_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", value[position:])
        if field_match is None:
            return None
        position += field_match.end()
        while position < len(value) and value[position] == "[":
            index_match = re.match(r"\[[0-9]+\]", value[position:])
            if index_match is None:
                return None
            position += index_match.end()
        if position < len(value) and value[position] != ".":
            return None
        position += 1
    return value


def _safe_len(value: Any) -> Optional[int]:
    try:
        result = len(value)
    except (TypeError, AttributeError):
        return None
    return result if isinstance(result, int) and result >= 0 else None


def _continuation_token(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return _DROP
    if isinstance(value, str):
        return _optional_token(value)
    return _safe_number(value)


def _quarantine_envelope(
    warnings: Iterable[str],
    quarantine: Optional[Mapping[str, Any]] = None,
    *,
    future_timestamp_quarantined: bool = False,
) -> dict[str, Any]:
    normalized = _normalize_quarantine(quarantine)
    safe_warnings = [
        warning
        for warning in _safe_warning_tokens(warnings, limit=64)
        if _is_quarantine_warning(warning)
    ]
    if future_timestamp_quarantined and "future_timestamp_quarantined" not in safe_warnings:
        safe_warnings.append("future_timestamp_quarantined")
    merged_warnings = _safe_warning_tokens(
        [*normalized["warnings"], *safe_warnings],
        limit=64,
    )
    if normalized["state"] == "unknown" and not (
        normalized["timestamps"] or merged_warnings
    ):
        state = "unknown"
    else:
        state = "quarantined" if (
            normalized["state"] == "quarantined"
            or normalized["timestamps"]
            or merged_warnings
        ) else "clear"
    return {
        "state": state,
        "warnings": merged_warnings,
        "timestamps": list(normalized["timestamps"]),
    }


def _quarantine_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return _quarantine_envelope(["future_timestamp_quarantined"] if value else [])
    if isinstance(value, str):
        token = _quarantine_state_token(value)
        if token is None:
            return _DROP
        return _quarantine_envelope([], {"state": token})
    if isinstance(value, Mapping):
        return _quarantine_envelope([], value)
    return _DROP


def _normalize_quarantine(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"state": "clear", "warnings": [], "timestamps": []}
    state = _quarantine_state_token(value.get("state")) or "unknown"
    raw_warnings = value.get("warnings", value.get("reasons", []))
    warnings = _safe_warning_tokens(
        raw_warnings
        if isinstance(raw_warnings, Sequence) and not isinstance(raw_warnings, (str, bytes, bytearray))
        else []
    )
    if value.get("future_timestamp_quarantined") is True and "future_timestamp_quarantined" not in warnings:
        warnings.append("future_timestamp_quarantined")
    timestamps: list[dict[str, Any]] = []
    raw_timestamps = value.get("timestamps", [])
    if isinstance(raw_timestamps, Sequence) and not isinstance(raw_timestamps, (str, bytes, bytearray)):
        for item in raw_timestamps:
            if not isinstance(item, Mapping):
                continue
            field = _optional_token(item.get("field"))
            if field not in _QUARANTINE_TIMESTAMP_FIELDS:
                continue
            timestamp_value = _safe_timestamp(item.get("value"))
            observed_at = _safe_timestamp(item.get("observedAt", item.get("observed_at")))
            if (
                timestamp_value is _DROP
                or observed_at is _DROP
                or timestamp_value is None
                or observed_at is None
            ):
                continue
            evidence: dict[str, Any] = {
                "field": field,
                "value": timestamp_value,
                "observedAt": observed_at,
            }
            message_id = _optional_token(item.get("messageId", item.get("message_id")))
            if message_id is not None:
                evidence["messageId"] = message_id
            if evidence not in timestamps:
                timestamps.append(evidence)
            if len(timestamps) >= 64:
                break
    if state == "quarantined" or warnings or timestamps:
        normalized_state = "quarantined"
    elif state == "unknown":
        normalized_state = "unknown"
    else:
        normalized_state = "clear"
    return {
        "state": normalized_state,
        "warnings": warnings,
        "timestamps": timestamps,
    }


def _coverage_details_envelope(details: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if details is None:
        projected = {"transfer_schema_version": TRANSFER_SCHEMA_VERSION}
        assert_no_secrets(projected)
        return projected
    if not isinstance(details, Mapping):
        raise LedgerError("coverage details must be a mapping")
    projected = _observation_envelope(details)
    participants = _coverage_participants(details.get("participants"))
    if participants:
        projected["participants"] = participants
    alias = _coverage_alias(details.get("alias"))
    if alias:
        projected["alias"] = alias
    projected["transfer_schema_version"] = TRANSFER_SCHEMA_VERSION
    assert_no_secrets(projected)
    return projected


def _coverage_write_envelope(
    scope: LedgerScope,
    *,
    scope_key_value: str,
    source_kind: str,
    source_id: str,
    reason: str,
    state: str,
    details: Optional[Mapping[str, Any]],
    seen_at: datetime,
) -> dict[str, Any]:
    safe_scope = _normalize_scope(scope)
    safe_scope_key = _required_token(scope_key_value, "scope_key")
    safe_source_kind = _required_token(source_kind, "source_kind")
    safe_source_id = _required_token(source_id, "source_id")
    safe_reason = _required_token(reason, "reason")
    safe_state = _enum_token(state, _ALLOWED_GAP_STATES, "unknown")
    safe_seen_at = _utc_datetime(seen_at, "seen_at")
    safe_details = _coverage_details_envelope(details)
    envelope = {
        "scope": _scope_payload(safe_scope),
        "scope_key": safe_scope_key,
        "context": {
            "source_kind": safe_source_kind,
            "source_id": safe_source_id,
            "reason": safe_reason,
            "state": safe_state,
            "seen_at": isoformat_utc(safe_seen_at),
        },
        "details": safe_details,
    }
    assert_no_secrets(envelope)
    return envelope


def _coverage_participants(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    participants: list[dict[str, str]] = []
    for item in list(value)[:32]:
        if not isinstance(item, Mapping):
            continue
        participant: dict[str, str] = {}
        for key in (
            "type",
            "role",
            "scope_key",
            "attempt_id",
            "identity_basis",
        ):
            token = _optional_token(item.get(key))
            if token is not None:
                participant[key] = token
        if participant:
            participants.append(participant)
    return participants


def _coverage_alias(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    alias: dict[str, str] = {}
    for key in ("kind", "value"):
        token = _optional_token(value.get(key))
        if token is not None:
            alias[key] = token
    return alias


def _safe_number(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _DROP
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return _DROP
    return value


def _safe_timestamp(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return _DROP
    try:
        parsed = parse_datetime(value)
    except (OverflowError, TypeError, ValueError, OSError):
        return _DROP
    return isoformat_utc(parsed) if parsed is not None else _DROP


def _quarantine_state_token(value: Any) -> Optional[str]:
    token = _optional_token(value)
    return token if token in _ALLOWED_QUARANTINE_STATES else "unknown"


def _transfer_version_token(value: Any) -> Any:
    token = _optional_token(value)
    if token is None:
        return _DROP
    # Keep an unsupported source version as evidence; the envelope's own
    # transfer_schema_version remains the current persistence contract.
    return token


def _json_count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = _optional_token(raw_key)
        if key is None:
            continue
        try:
            count = int(raw_value)
        except (TypeError, ValueError):
            continue
        if count >= 0:
            out[key] = count
    return out


def _time_state_sql(
    start: Optional[datetime],
    end: Optional[datetime],
) -> tuple[str, list[Any]]:
    if start is None and end is None:
        return (
            """
            CASE
                WHEN attempt_time IS NOT NULL THEN 'definite'
                WHEN earliest_possible_at IS NOT NULL
                 AND latest_possible_at IS NOT NULL
                 AND latest_possible_at >= earliest_possible_at
                    THEN 'definite'
                ELSE 'unknown'
            END
            """,
            [],
        )
    if start is not None and end is not None:
        return (
            """
            CASE
                WHEN attempt_time IS NOT NULL THEN
                    CASE
                        WHEN attempt_time >= %s AND attempt_time < %s
                            THEN 'definite'
                        ELSE 'out'
                    END
                WHEN earliest_possible_at IS NOT NULL
                 AND latest_possible_at IS NOT NULL
                 AND latest_possible_at < earliest_possible_at
                    THEN 'unknown'
                WHEN latest_possible_at IS NOT NULL
                 AND latest_possible_at < %s
                    THEN 'out'
                WHEN earliest_possible_at IS NOT NULL
                 AND earliest_possible_at >= %s
                    THEN 'out'
                WHEN earliest_possible_at IS NULL
                  OR latest_possible_at IS NULL
                    THEN 'unknown'
                WHEN earliest_possible_at >= %s
                 AND latest_possible_at < %s
                    THEN 'definite'
                ELSE 'ambiguous'
            END
            """,
            [start, end, start, end, start, end],
        )
    if start is not None:
        return (
            """
            CASE
            WHEN attempt_time IS NOT NULL THEN
                CASE WHEN attempt_time < %s THEN 'out' ELSE 'unknown' END
                WHEN earliest_possible_at IS NOT NULL
                 AND latest_possible_at IS NOT NULL
                 AND latest_possible_at < earliest_possible_at
                    THEN 'unknown'
                WHEN latest_possible_at IS NOT NULL
                 AND latest_possible_at < %s THEN 'out'
                ELSE 'unknown'
            END
            """,
            [start, start],
        )
    return (
        """
        CASE
            WHEN attempt_time IS NOT NULL THEN
                CASE WHEN attempt_time >= %s THEN 'out' ELSE 'unknown' END
            WHEN earliest_possible_at IS NOT NULL
             AND latest_possible_at IS NOT NULL
             AND latest_possible_at < earliest_possible_at
                THEN 'unknown'
            WHEN earliest_possible_at IS NOT NULL
             AND earliest_possible_at >= %s THEN 'out'
            ELSE 'unknown'
        END
        """,
        [end, end],
    )


def _without_keys(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _utc_or_none(value: Optional[datetime]) -> Optional[datetime]:
    return ensure_utc(value) if value is not None else None


def _sanitize_provenance_value(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = _safe_number(value)
        return number if number is not _DROP else None
    if isinstance(value, str):
        return sanitize_token(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[str] = []
        for item in value:
            token = sanitize_token(item) if isinstance(item, str) else None
            if token is not None:
                values.append(token)
        return values[:32]
    return None


def _sorted_value(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sorted_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _sorted_value(value[key]) for key in sorted(value, key=str)}
    return value
