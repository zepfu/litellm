"""Durable coordination state for the ChatGPT Chat history collector bridge."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

import psycopg

from .models import AttemptRecord
from .privacy import assert_no_secrets, sanitize_token
from .pg_ledger import (
    LedgerBinding,
    LedgerError,
    LedgerScope,
    PgLedger,
    PgLedgerPage,
    IngestContext,
    _lock_scope_registry,
    _observation_envelope,
    _assert_active_binding,
    fingerprint_value,
    scope_key,
)
from .timeutil import ensure_utc, parse_datetime


PROTOCOL_VERSION = 1
SCHEMA_VERSION = "chatgpt-collector-state-v1"
MAX_QUEUE_PAGE = 256
MAX_QUEUE_ITEM_BYTES = 16 * 1024
MAX_STATE_FIELD_BYTES = 64 * 1024
MAX_PAGE_PAYLOAD_BYTES = 1024 * 1024
MAX_SNAPSHOT_AGE_SECONDS = 300
MAX_CURSOR_BYTES = 512
STATE_SCHEDULE_KEYS = {
    "interval",
    "intervalMs",
    "anchorAt",
    "jitterSeconds",
    "jitterMs",
    "nextTickIndex",
    "nextDueAt",
    "pending",
    "active",
    "retryNotBefore",
    "serverRetryNotBefore",
    "failureStreak",
    "authPausedUntil",
    "lastTriggerAt",
    "lastCompletedAt",
    "scope",
    "stateVersion",
}
STATE_TRIGGER_KEYS = {
    "triggerId",
    "kind",
    "missedCount",
    "requestedAt",
    "dueAt",
    "jitterMs",
    "claimedAt",
    "fencingToken",
    "state",
    "runId",
    "outcome",
    "summary",
}
STATE_SCOPE_KEYS = {"collectorAccountId", "profileId", "accountId", "scope"}
STATE_PENDING_KEYS = {
    "kind",
    "missedCount",
    "requestedAt",
    "triggerId",
    "dueAt",
    "jitterMs",
}
STATE_CHECKPOINT_KEYS = {
    "stateVersion",
    "accountId",
    "scope",
    "status",
    "mode",
    "range",
    "candidateCutoff",
    "scanStartedAt",
    "continuation",
    "pagesFetched",
    "pageBudget",
    "lastCompleteDiscoveryStartedAt",
    "lastPageAt",
    "paginationState",
    "warnings",
    "headFingerprint",
    "candidateQueue",
    "olderHistoryAudit",
    "updatedAt",
}
STATE_ACCOUNT_KEYS = {
    "status",
    "reason",
    "pausedAt",
    "cooldownUntil",
    "lastError",
}
STATE_RANGE_KEYS = {"start", "end"}
STATE_PAGINATION_KEYS = {
    "continuation",
    "exhausted",
    "paginationState",
    "schemaVersion",
    "coverage",
    "warnings",
}
STATE_SUMMARY_KEYS = {
    "conversationId",
    "createdAt",
    "updatedAt",
    "isArchived",
    "workspaceId",
    "projectId",
    "surface",
    "origin",
    "hasVersions",
    "currentNode",
    "coverage",
}
STATE_COVERAGE_KEYS = {
    "active",
    "archived",
    "history",
    "projects",
    "branches",
    "overall",
    "gaps",
    "enabled",
    "status",
    "scope",
    "coverage",
    "pagesFetched",
    "candidates",
    "continuation",
    "paginationState",
    "candidateCutoff",
    "warnings",
    "olderHistoryAudit",
}
STATE_SCOPE_COVERAGE_KEYS = {
    "scope",
    "status",
    "coverage",
    "pagesFetched",
    "candidates",
    "continuation",
    "paginationState",
    "candidateCutoff",
    "warnings",
    "olderHistoryAudit",
}
STATE_HISTORY_COVERAGE_KEYS = {
    "active",
    "archived",
    "projects",
    "branches",
    "olderHistoryAudit",
    "overall",
    "gaps",
}
STATE_AUDIT_RESULT_KEYS = {
    "enabled",
    "status",
    "continuation",
    "pagesFetched",
    "conversationsAudited",
    "lastStartedAt",
    "lastPageAt",
    "lastCompletedAt",
}
STATE_AUDIT_AGGREGATE_KEYS = {"enabled", "status", "active", "archived"}
STATE_IDENTITY_KEYS = {
    "providerUserId",
    "workspaceId",
    "quotaOwnerId",
    "surface",
    "authState",
    "identityErrors",
}
STATE_TERMINAL_SUMMARY_KEYS = {
    "accountId",
    "mode",
    "range",
    "scanStartedAt",
    "status",
    "accountState",
    "identity",
    "scopes",
    "coverage",
    "historyCoverage",
    "overall",
    "gaps",
    "warnings",
    "pagesFetched",
    "observations",
    "attempts",
    "candidates",
    "coverageIncomplete",
    "startedAt",
    "completedAt",
    "updatedAt",
    "reason",
    "outcome",
}
STATE_CANDIDATE_KEYS = {"summary", "missingUpdateTime", "revisit"}
STATE_REVISIT_KEYS = {
    "stateVersion",
    "accountId",
    "conversationId",
    "scopes",
    "status",
    "reason",
    "firstSeenAt",
    "lastSeenAt",
    "attempts",
    "nextEligibleAt",
    "lastError",
    "detailPagesFetched",
    "continuation",
    "continuationRevision",
    "malformedPage",
    "outstandingGeneration",
}
STATE_MALFORMED_PAGE_KEYS = {"reason", "warnings"}
STATE_OUTSTANDING_GENERATION_KEYS = {"state", "since", "timedOut"}
STATE_AUDIT_KEYS = {
    "enabled",
    "status",
    "continuation",
    "pagesFetched",
    "conversationsAudited",
    "lastStartedAt",
    "lastPageAt",
    "lastCompletedAt",
}
STATE_QUARANTINE_KEYS = {"state", "warnings", "timestamps"}
STATE_QUARANTINE_TIMESTAMP_KEYS = {
    "field",
    "value",
    "observedAt",
    "messageId",
}
STATE_FORBIDDEN_KEYS = {
    "title",
    "body",
    "content",
    "text",
    "prompt",
    "answer",
    "message",
    "messages",
    "parts",
    "html",
    "markdown",
}
STATE_BOOLEAN_KEYS = {
    "active",
    "enabled",
    "exhausted",
    "isArchived",
    "hasVersions",
    "missingUpdateTime",
    "timedOut",
    "coverageIncomplete",
}
STATE_NUMBER_KEYS = {
    "stateVersion",
    "missedCount",
    "requestedAt",
    "dueAt",
    "jitterMs",
    "claimedAt",
    "fencingToken",
    "intervalMs",
    "anchorAt",
    "jitterSeconds",
    "nextTickIndex",
    "nextDueAt",
    "retryNotBefore",
    "serverRetryNotBefore",
    "failureStreak",
    "authPausedUntil",
    "lastTriggerAt",
    "lastCompletedAt",
    "pagesFetched",
    "pageBudget",
    "attempts",
    "detailPagesFetched",
    "conversationsAudited",
    "pageNumber",
    "observations",
    "candidates",
}
STATE_STRING_KEYS = {
    "accountId",
    "collectorAccountId",
    "profileId",
    "scope",
    "status",
    "mode",
    "kind",
    "triggerId",
    "runId",
    "outcome",
    "interval",
    "range",
    "candidateCutoff",
    "scanStartedAt",
    "lastCompleteDiscoveryStartedAt",
    "lastStartedAt",
    "lastPageAt",
    "updatedAt",
    "nextEligibleAt",
    "lastError",
    "reason",
    "paginationState",
    "continuationRevision",
    "surface",
    "origin",
    "authState",
    "providerUserId",
    "workspaceId",
    "quotaOwnerId",
    "startedAt",
    "completedAt",
    "pausedAt",
    "cooldownUntil",
    "since",
    "observedAt",
    "value",
    "messageId",
    "field",
}
STATE_STRING_ARRAY_KEYS = {
    "warnings",
    "gaps",
    "identityErrors",
    "evidenceMessageIds",
    "unknownFields",
}
STATE_OBJECT_ARRAY_KEYS = {"candidateQueue", "scopes", "timestamps"}
STATE_IDENTIFIER_KEYS = {
    "accountId",
    "collectorAccountId",
    "profileId",
    "providerUserId",
    "workspaceId",
    "quotaOwnerId",
    "scope",
    "triggerId",
    "runId",
    "conversationId",
    "currentNode",
    "candidateCutoff",
    "continuationRevision",
    "sourceId",
    "sourceKind",
}
STATE_ENUM_VALUES = {
    "status": {
        "not_started",
        "in_progress",
        "complete",
        "partial",
        "ready",
        "paused",
        "pending",
        "cancelled",
        "active",
        "claimed",
        "running",
        "idle",
        "disabled",
        "blocked",
        "unknown",
    },
    "mode": {"backfill", "incremental", "reconciliation"},
    "kind": {"scheduled", "manual", "coalesced"},
    "coverage": {"complete", "partial", "unknown", "validated_page", "unrecognized"},
    "paginationState": {
        "complete",
        "continuation",
        "contradictory",
        "unknown",
        "repeated_cursor",
        "budget_exhausted",
    },
    "scope": {"active", "archived"},
    "outcome": {
        "success",
        "failure",
        "authentication",
        "completed",
        "failed_after_start",
        "cancelled_after_start",
        "rejected_after_start",
        "completion_unknown",
        "rejected_before_start",
        "unresolved",
        "unknown",
    },
    "reason": {
        "authentication",
        "cooldown",
        "incomplete_detail",
        "partial_detail",
        "unrecognized_detail",
        "repeated_cursor",
        "bad_continuation",
        "page_budget",
        "unknown_pagination",
        "contradictory_pagination",
        "detail_unavailable",
        "missing_update_time",
        "nonterminal_generation",
        "coverage_gap_resolution",
    },
}
_UNSET = object()


@dataclass(frozen=True)
class CollectorStateHeader:
    profile_id: str
    collector_account_id: str
    state_version: int
    schedule_transition: Optional[Mapping[str, Any]]
    checkpoint: Optional[Mapping[str, Any]]
    active_trigger: Optional[Mapping[str, Any]]
    updated_at: datetime

    def payload(self) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "schemaVersion": SCHEMA_VERSION,
            "profileId": self.profile_id,
            "collectorAccountId": self.collector_account_id,
            "stateVersion": self.state_version,
            "scheduleTransition": self.schedule_transition,
            "checkpoint": self.checkpoint,
            "activeTrigger": self.active_trigger,
            "updatedAt": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class CandidatePage:
    items: tuple[Mapping[str, Any], ...]
    has_more: bool
    cursor: Optional[str] = None
    state_version: int = 0


@dataclass(frozen=True)
class CollectorLease:
    profile_id: str
    collector_account_id: str
    lease_fencing_token: int
    lease_expires_at: datetime
    binding: LedgerBinding

    def payload(self) -> dict[str, Any]:
        return {
            "profileId": self.profile_id,
            "collectorAccountId": self.collector_account_id,
            "leaseFencingToken": self.lease_fencing_token,
            "leaseExpiresAt": self.lease_expires_at.isoformat(),
            "bindingGeneration": self.binding.binding_generation,
        }


@dataclass(frozen=True)
class PageAck:
    run_id: str
    page_commit_id: str
    state_version: int
    payload_fingerprint: str


@dataclass(frozen=True)
class _PreparedPageCommit:
    account: str
    profile: str
    run_id: str
    page_commit_id: str
    wire_payload: Mapping[str, Any]
    ingest_operations: tuple[Mapping[str, Any], ...]
    candidate_mutations: tuple[Mapping[str, Any], ...]
    coverage_mutations: tuple[Mapping[str, Any], ...]
    expected_state_version: Optional[int]
    trigger_id: str
    fingerprint: str


@dataclass
class _ReadSnapshot:
    snapshot_id: str
    account: str
    profile: Optional[str]
    kind: str
    connection: psycopg.Connection
    created_at: datetime
    deadline_at: datetime
    owner_run_id: str
    owner_profile: str
    owner_fencing_token: int
    operation: Any = None


def _operation_deadline(
    operation: Any = None,
    *,
    explicit: Optional[datetime] = None,
) -> Optional[datetime]:
    if operation is None:
        return ensure_utc(explicit) if explicit is not None else None
    if not bool(getattr(operation, "ignore_cancel", False)):
        checker = getattr(operation, "check", None)
        if callable(checker):
            checker()
    operation_deadline = getattr(operation, "deadline_at", None)
    if not isinstance(operation_deadline, datetime):
        raise LedgerError("collector operation deadline is required")
    normalized = ensure_utc(operation_deadline)
    if explicit is None:
        return normalized
    return min(normalized, ensure_utc(explicit))


class PgCollectorState:
    """PostgreSQL-owned collector state; scheduling remains in TypeScript."""

    def __init__(self, ledger: PgLedger) -> None:
        self.ledger = ledger
        self._snapshots: dict[str, _ReadSnapshot] = {}

    @staticmethod
    def assert_safe_record(value: Any) -> None:
        assert_no_secrets(value)

    def ensure_schema(self) -> None:
        self.ledger.ensure_schema()

    def load_state(
        self,
        *,
        collector_account_id: str,
        profile_id: str,
        operation: Any = None,
    ) -> tuple[Optional[CollectorStateHeader], CandidatePage]:
        """Load only the durable header; queue reads are explicit and versioned."""
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        _operation_deadline(operation)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._lock_state_authority(cur, account, profile)
            header = self._load_header(cur, account, profile)
        return header, CandidatePage(
            (),
            has_more=False,
            state_version=header.state_version if header is not None else 0,
        )

    def compare_and_set_state(
        self,
        *,
        collector_account_id: str,
        profile_id: str,
        expected_state_version: Optional[int],
        schedule_transition: Optional[Mapping[str, Any]],
        checkpoint: Optional[Mapping[str, Any]],
        active_trigger: Optional[Mapping[str, Any]],
        lease: CollectorLease,
        operation: Any = None,
    ) -> CollectorStateHeader:
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        payload = {
            "scheduleTransition": schedule_transition,
            "checkpoint": checkpoint,
            "activeTrigger": active_trigger,
        }
        self.assert_safe_record(payload)
        _operation_deadline(operation)
        _validate_state_field("scheduleTransition", schedule_transition, STATE_SCHEDULE_KEYS)
        _validate_state_field("checkpoint", checkpoint, STATE_CHECKPOINT_KEYS)
        _validate_state_field("activeTrigger", active_trigger, STATE_TRIGGER_KEYS)
        now = datetime.now(timezone.utc)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._require_active_lease(
                cur,
                account=account,
                profile_id=profile,
                expected=lease,
            )
            self._lock_state_authority(cur, account, profile)
            current = self._load_header(cur, account, profile, lock=True)
            if current is None:
                if expected_state_version not in (None, 0):
                    raise LedgerError("collector state version is stale")
                state_version = 1
            else:
                if current.state_version != expected_state_version:
                    raise LedgerError("collector state version is stale")
                state_version = current.state_version + 1
            current_schedule = current.schedule_transition if current is not None else None
            current_checkpoint = current.checkpoint if current is not None else None
            current_trigger = current.active_trigger if current is not None else None
            next_schedule = (
                schedule_transition if schedule_transition is not None else current_schedule
            )
            next_checkpoint = checkpoint if checkpoint is not None else current_checkpoint
            next_trigger = active_trigger if active_trigger is not None else current_trigger
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_collector_state (
                    profile_id, collector_account_id, state_version,
                    schedule_transition, checkpoint, active_trigger, updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (profile_id, collector_account_id) DO UPDATE SET
                    state_version = EXCLUDED.state_version,
                    schedule_transition = EXCLUDED.schedule_transition,
                    checkpoint = EXCLUDED.checkpoint,
                    active_trigger = EXCLUDED.active_trigger,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    profile,
                    account,
                    state_version,
                    _json(next_schedule),
                    _json(next_checkpoint),
                    _json(next_trigger),
                    now,
                ),
            )
            return CollectorStateHeader(
                profile,
                account,
                state_version,
                next_schedule,
                next_checkpoint,
                next_trigger,
                now,
            )

    def claim_lease(
        self,
        *,
        collector_account_id: str,
        profile_id: str,
        scope: LedgerScope,
        ttl_seconds: int,
        expected_binding: Optional[LedgerBinding] = None,
        operation: Any = None,
    ) -> CollectorLease:
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        if ttl_seconds <= 0 or ttl_seconds > 3600:
            raise LedgerError("collector lease ttl is outside the supported range")
        _operation_deadline(operation)
        now = datetime.now(timezone.utc)
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, timezone.utc)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._lock_lease_authority(cur, profile)
            binding = self._binding_for_scope(cur, scope, account, expected_binding)
            cur.execute(
                """
                SELECT collector_account_id, lease_fencing_token, lease_expires_at
                FROM public.chatgpt_usage_collector_leases
                WHERE profile_id = %s
                FOR UPDATE
                """,
                (profile,),
            )
            rows = cur.fetchall()
            own_row = next(
                (row for row in rows if str(row[0]) == account),
                None,
            )
            for row in rows:
                if str(row[0]) != account and ensure_utc(row[2]) > now:
                    raise LedgerError("collector profile is already active for another account")
            if own_row is not None and ensure_utc(own_row[2]) > now:
                raise LedgerError(
                    "collector lease is already active; use heartbeat for renewal"
                )
            next_token = 1 if own_row is None else int(own_row[1]) + 1
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_collector_leases (
                    profile_id, collector_account_id, scope_key,
                    binding_generation, lease_fencing_token, lease_expires_at,
                    claimed_at, renewed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_id, collector_account_id) DO UPDATE SET
                    scope_key = EXCLUDED.scope_key,
                    binding_generation = EXCLUDED.binding_generation,
                    lease_fencing_token = EXCLUDED.lease_fencing_token,
                    lease_expires_at = EXCLUDED.lease_expires_at,
                    claimed_at = EXCLUDED.claimed_at,
                    renewed_at = EXCLUDED.renewed_at
                """,
                (
                    profile,
                    account,
                    binding.scope_key,
                    binding.binding_generation,
                    next_token,
                    expires,
                    now,
                    now,
                ),
            )
            return CollectorLease(profile, account, next_token, expires, binding)

    def heartbeat_lease(
        self,
        *,
        lease: CollectorLease,
        ttl_seconds: int,
        operation: Any = None,
    ) -> CollectorLease:
        if ttl_seconds <= 0 or ttl_seconds > 3600:
            raise LedgerError("collector lease ttl is outside the supported range")
        _operation_deadline(operation)
        now = datetime.now(timezone.utc)
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, timezone.utc)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._require_active_lease(
                cur,
                account=lease.collector_account_id,
                profile_id=lease.profile_id,
                expected=lease,
            )
            cur.execute(
                """
                UPDATE public.chatgpt_usage_collector_leases
                SET lease_expires_at = %s, renewed_at = %s
                WHERE profile_id = %s AND collector_account_id = %s
                  AND lease_fencing_token = %s
                """,
                (
                    expires,
                    now,
                    lease.profile_id,
                    lease.collector_account_id,
                    lease.lease_fencing_token,
                ),
            )
        return CollectorLease(
            lease.profile_id,
            lease.collector_account_id,
            lease.lease_fencing_token,
            expires,
            lease.binding,
        )

    def release_lease(self, *, lease: CollectorLease, operation: Any = None) -> None:
        _operation_deadline(operation)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._require_active_lease(
                cur,
                account=lease.collector_account_id,
                profile_id=lease.profile_id,
                expected=lease,
            )
            self._release_lease_locked(cur, lease)

    def close_snapshot(self, snapshot_id: str) -> None:
        safe_snapshot = _token(snapshot_id, "snapshot_id")
        snapshot = self._snapshots.pop(safe_snapshot, None)
        if snapshot is not None:
            # The snapshot is read-only; closing the owned libpq connection
            # releases the server transaction without a second network wait.
            snapshot.connection.close()

    def _release_lease_locked(
        self,
        cur: psycopg.Cursor,
        lease: CollectorLease,
    ) -> None:
        cur.execute(
            """
            UPDATE public.chatgpt_usage_collector_leases
            SET lease_expires_at = NOW()
            WHERE profile_id = %s AND collector_account_id = %s
              AND lease_fencing_token = %s
            """,
            (
                lease.profile_id,
                lease.collector_account_id,
                lease.lease_fencing_token,
            ),
        )
        if cur.rowcount != 1:
            raise LedgerError("collector lease fence is stale")

    def _prepare_page_commit(
        self,
        *,
        lease: CollectorLease,
        run_id: str,
        page_commit_id: str,
        canonical_payload: Mapping[str, Any],
        ingest_operations: Sequence[Mapping[str, Any]],
        candidate_mutations: Sequence[Mapping[str, Any]],
        coverage_mutations: Sequence[Mapping[str, Any]],
        expected_state_version: Optional[int],
        trigger_id: Optional[str],
    ) -> _PreparedPageCommit:
        account = _account(lease.collector_account_id)
        profile = _token(lease.profile_id, "profile_id")
        safe_run = _token(run_id, "run_id")
        safe_commit = _token(page_commit_id, "page_commit_id")
        if not isinstance(canonical_payload, Mapping):
            raise LedgerError("collector page payload must be an object")
        wire_payload = canonical_payload.get("mutations", canonical_payload)
        if not isinstance(wire_payload, Mapping):
            raise LedgerError("collector page mutations must be an object")
        wire_page_commit_id = wire_payload.get("pageCommitId")
        if wire_page_commit_id is not None and _token(
            wire_page_commit_id, "pageCommitId"
        ) != safe_commit:
            raise LedgerError("collector page commit identity conflicts with its payload")
        derived_candidates = self._derive_candidate_mutations(wire_payload)
        actual_candidates = (
            tuple(candidate_mutations) + tuple(derived_candidates)
            if candidate_mutations
            else tuple(derived_candidates)
        )
        derived_operations = self._derive_ingest_operations(
            wire_payload,
            default_run_id=safe_run,
            candidate_mutations=actual_candidates,
        )
        actual_operations = tuple(ingest_operations) + tuple(derived_operations)
        derived_coverage = self._derive_coverage_mutations(wire_payload)
        actual_coverage = (
            tuple(coverage_mutations) + tuple(derived_coverage)
            if coverage_mutations
            else tuple(derived_coverage)
        )
        expected_version = (
            expected_state_version
            if expected_state_version is not None
            else _optional_int(wire_payload.get("expectedStateVersion"))
        )
        payload_trigger_id = wire_payload.get("triggerId")
        if payload_trigger_id is None:
            source = wire_payload.get("source", wire_payload.get("context"))
            if isinstance(source, Mapping):
                payload_trigger_id = source.get("triggerId")
        safe_trigger = _token(
            trigger_id if trigger_id is not None else payload_trigger_id,
            "trigger_id",
        )
        if (
            trigger_id is not None
            and payload_trigger_id is not None
            and safe_trigger != _token(payload_trigger_id, "triggerId")
        ):
            raise LedgerError("collector page trigger identity conflicts with its payload")
        self.assert_safe_record(canonical_payload)
        serialized_payload = json.dumps(
            canonical_payload,
            separators=(",", ":"),
            default=str,
        )
        if len(serialized_payload.encode("utf-8")) > MAX_PAGE_PAYLOAD_BYTES:
            raise LedgerError("collector page commit payload exceeds the supported bound")
        if len(actual_operations) > MAX_QUEUE_PAGE:
            raise LedgerError("collector page operation batch exceeds the supported bound")
        if len(actual_candidates) > MAX_QUEUE_PAGE:
            raise LedgerError("collector candidate batch exceeds the supported bound")
        if len(actual_coverage) > MAX_QUEUE_PAGE:
            raise LedgerError("collector coverage batch exceeds the supported bound")
        for operation in actual_operations:
            self.assert_safe_record(operation)
            self._validate_ingest_operation(operation)
        for mutation in actual_candidates:
            self.assert_safe_record(mutation)
        for mutation in actual_coverage:
            self.assert_safe_record(mutation)
        fingerprint = fingerprint_value(
            {
                "payload": canonical_payload,
                "ingestOperations": list(actual_operations),
                "candidateMutations": list(actual_candidates),
                "coverageMutations": list(actual_coverage),
                "triggerId": safe_trigger,
            }
        )
        return _PreparedPageCommit(
            account,
            profile,
            safe_run,
            safe_commit,
            wire_payload,
            actual_operations,
            actual_candidates,
            actual_coverage,
            expected_version,
            safe_trigger,
            fingerprint,
        )

    def commit_history_page(
        self,
        *,
        lease: CollectorLease,
        scope: LedgerScope,
        run_id: str,
        page_commit_id: str,
        canonical_payload: Mapping[str, Any],
        ingest_operations: Sequence[Mapping[str, Any]] = (),
        candidate_mutations: Sequence[Mapping[str, Any]] = (),
        coverage_mutations: Sequence[Mapping[str, Any]] = (),
        expected_state_version: Optional[int] = None,
        trigger_id: Optional[str] = None,
        operation: Any = None,
    ) -> PageAck:
        _operation_deadline(operation)
        prepared = self._prepare_page_commit(
            lease=lease,
            run_id=run_id,
            page_commit_id=page_commit_id,
            canonical_payload=canonical_payload,
            ingest_operations=ingest_operations,
            candidate_mutations=candidate_mutations,
            coverage_mutations=coverage_mutations,
            expected_state_version=expected_state_version,
            trigger_id=trigger_id,
        )
        acknowledgment: Optional[PageAck] = None
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._require_active_lease(
                cur,
                account=prepared.account,
                profile_id=prepared.profile,
                expected=lease,
            )
            self._lock_state_authority(cur, prepared.account, prepared.profile)
            binding = self._binding_for_scope(
                cur, scope, prepared.account, lease.binding
            )
            cur.execute(
                """
                SELECT state_version, payload_fingerprint
                FROM public.chatgpt_usage_collector_page_acks
                WHERE profile_id = %s AND collector_account_id = %s
                  AND run_id = %s AND page_commit_id = %s
                """,
                (
                    prepared.profile,
                    prepared.account,
                    prepared.run_id,
                    prepared.page_commit_id,
                ),
            )
            replay = cur.fetchone()
            if replay is not None:
                if str(replay[1]) != prepared.fingerprint:
                    raise LedgerError(
                        "collector page commit identity conflicts with its retained payload"
                    )
                acknowledgment = PageAck(
                    prepared.run_id,
                    prepared.page_commit_id,
                    int(replay[0]),
                    prepared.fingerprint,
                )
            else:
                self._assert_expected_state_version(
                    cur,
                    account=prepared.account,
                    profile=prepared.profile,
                    expected_state_version=prepared.expected_state_version,
                )
                self._apply_ingest_operations(
                    cur,
                    scope,
                    binding,
                    prepared.ingest_operations,
                    operation=operation,
                )
                self._apply_coverage_mutations(
                    cur,
                    scope,
                    binding,
                    prepared.coverage_mutations,
                    operation=operation,
                )
                next_state = self._publish_checkpoint(
                    cur,
                    lease,
                    run_id=prepared.run_id,
                    checkpoint=self._checkpoint_from_canonical(prepared.wire_payload),
                    expected_state_version=prepared.expected_state_version,
                    trigger_id=prepared.trigger_id,
                )
                for rank, mutation in enumerate(
                    prepared.candidate_mutations, start=1
                ):
                    self._apply_candidate_mutation(
                        cur,
                        prepared.profile,
                        prepared.account,
                        mutation,
                        rank,
                    )
                cur.execute(
                    """
                    SELECT COALESCE(MAX(serial), 0)
                    FROM public.chatgpt_usage_collector_page_acks
                    WHERE profile_id = %s AND collector_account_id = %s
                    """,
                    (prepared.profile, prepared.account),
                )
                latest_serial = int(cur.fetchone()[0])
                if latest_serial:
                    cur.execute(
                        """
                        DELETE FROM public.chatgpt_usage_collector_page_acks
                        WHERE profile_id = %s AND collector_account_id = %s
                          AND serial < %s
                        """,
                        (prepared.profile, prepared.account, latest_serial),
                    )
                cur.execute(
                    """
                    INSERT INTO public.chatgpt_usage_collector_page_acks (
                        profile_id, collector_account_id, run_id, page_commit_id,
                        serial, state_version, payload_fingerprint, acked_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        prepared.profile,
                        prepared.account,
                        prepared.run_id,
                        prepared.page_commit_id,
                        latest_serial + 1,
                        next_state.state_version,
                        prepared.fingerprint,
                    ),
                )
                acknowledgment = PageAck(
                    prepared.run_id,
                    prepared.page_commit_id,
                    next_state.state_version,
                    prepared.fingerprint,
                )
        if acknowledgment is None:
            raise LedgerError("collector page commit did not produce an acknowledgment")
        return acknowledgment

    def load_conversation_metadata(
        self,
        *,
        collector_account_id: str,
        conversation_id: str,
        limit: int = 32,
        cursor: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        deadline_at: Optional[datetime] = None,
        run_id: str,
        profile_id: str,
        lease_fencing_token: int,
        operation: Any = None,
    ) -> Mapping[str, Any]:
        account = _account(collector_account_id)
        conversation = _token(conversation_id, "conversation_id")
        if cursor is not None and snapshot_id is None:
            raise LedgerError("collector metadata cursor requires a snapshot")
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("conversation metadata limit is outside the supported range")
        snapshot_deadline = _operation_deadline(
            operation,
            explicit=deadline_at,
        )
        snapshot, _ = self._get_snapshot(
            account=account,
            profile=profile_id,
            kind="metadata",
            snapshot_id=snapshot_id,
            deadline_at=snapshot_deadline,
            operation=operation,
            owner_run_id=run_id,
            owner_profile=profile_id,
            owner_fencing_token=lease_fencing_token,
        )
        try:
            conn = snapshot.connection
            self._refresh_snapshot_timeout(snapshot)
            with conn.cursor() as cur:
                cursor_values = _decode_cursor(cursor, "observation")
                cursor_time = (
                    parse_datetime(cursor_values["observedAt"])
                    if cursor_values is not None
                    else None
                )
                cursor_id = cursor_values["id"] if cursor_values is not None else None
                scope_cte = """
                    WITH RECURSIVE scope_chain(
                        stored_scope_key, canonical_scope_key
                    ) AS (
                        SELECT scope.scope_key, scope.scope_key
                        FROM public.chatgpt_usage_scopes AS scope
                        UNION
                        SELECT chain.stored_scope_key,
                               redirect.canonical_scope_key
                        FROM scope_chain AS chain
                        JOIN public.chatgpt_usage_scope_redirects AS redirect
                          ON redirect.retired_scope_key =
                             chain.canonical_scope_key
                    ),
                    canonical_scope_map AS (
                        SELECT DISTINCT ON (chain.stored_scope_key)
                               chain.stored_scope_key,
                               chain.canonical_scope_key
                        FROM scope_chain AS chain
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM public.chatgpt_usage_scope_redirects AS redirect
                            WHERE redirect.retired_scope_key =
                                  chain.canonical_scope_key
                        )
                        ORDER BY chain.stored_scope_key, chain.canonical_scope_key
                    ),
                    account_scope_map AS (
                        SELECT map.stored_scope_key,
                               map.canonical_scope_key,
                               binding.collector_account_id
                        FROM canonical_scope_map AS map
                        JOIN public.chatgpt_usage_scope_bindings AS binding
                          ON binding.scope_key = map.canonical_scope_key
                         AND binding.collector_account_id = %s
                         AND binding.binding_state = 'active'
                    )
                """
                self._refresh_snapshot_timeout(snapshot)
                self.ledger.execute_with_deadline(
                    cur,
                    scope_cte
                    + """
                    SELECT COUNT(*)
                    FROM public.chatgpt_usage_observations AS observation
                    JOIN account_scope_map AS scope_map
                      ON scope_map.stored_scope_key = observation.scope_key
                    WHERE observation.conversation_id = %s
                      AND observation.is_current_projection
                    """,
                    (account, conversation),
                    deadline_at=snapshot.deadline_at,
                    operation=snapshot.operation,
                )
                total = int(cur.fetchone()[0])
                self._refresh_snapshot_timeout(snapshot)
                self.ledger.execute_with_deadline(
                    cur,
                    scope_cte
                    + """
                    SELECT observation.observation_id, observation.scope_key,
                           observation.source_kind, observation.source_id,
                           observation.run_id, observation.observed_at,
                           observation.schema_version, observation.provenance,
                           observation.payload
                    FROM public.chatgpt_usage_observations AS observation
                    JOIN account_scope_map AS scope_map
                      ON scope_map.stored_scope_key = observation.scope_key
                    WHERE observation.conversation_id = %s
                      AND observation.is_current_projection
                      AND (
                          %s::timestamptz IS NULL
                          OR observation.observed_at < %s::timestamptz
                          OR (
                              observation.observed_at = %s::timestamptz
                              AND observation.observation_id < %s
                          )
                      )
                    ORDER BY observation.observed_at DESC, observation.observation_id DESC
                    LIMIT %s
                    """,
                    (
                        account,
                        conversation,
                        cursor_time,
                        cursor_time,
                        cursor_time,
                        cursor_id,
                        limit + 1,
                    ),
                    deadline_at=snapshot.deadline_at,
                    operation=snapshot.operation,
                )
                rows = cur.fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            items = tuple(self._observation_snapshot(row) for row in rows)
            next_cursor = (
                _encode_cursor(
                    "observation",
                    {
                        "observedAt": ensure_utc(rows[-1][5]).isoformat(),
                        "id": str(rows[-1][0]),
                    },
                )
                if has_more and rows
                else None
            )
            result = {
                "schemaVersion": "chatgpt-chat-history-v1",
                "snapshotId": snapshot.snapshot_id if has_more else None,
                "items": items,
                "totalObservations": total,
                "returnedObservations": len(items),
                "hasMore": has_more,
                "coverage": {
                    "truncated": has_more,
                    "partial": has_more,
                    "totalObservations": total,
                    "returnedObservations": len(items),
                },
                "nextCursor": next_cursor,
                "cursor": next_cursor,
            }
            if not has_more:
                self.close_snapshot(snapshot.snapshot_id)
            return result
        except BaseException:
            self.close_snapshot(snapshot.snapshot_id)
            raise

    def load_report_snapshot(
        self,
        *,
        collector_account_id: str,
        limit: int = 256,
        cursor: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        deadline_at: Optional[datetime] = None,
        run_id: str,
        profile_id: str,
        lease_fencing_token: int,
        operation: Any = None,
    ) -> Mapping[str, Any]:
        account = _account(collector_account_id)
        if cursor is not None and snapshot_id is None:
            raise LedgerError("collector report cursor requires a snapshot")
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("report snapshot limit is outside the supported range")
        snapshot_deadline = _operation_deadline(
            operation,
            explicit=deadline_at,
        )
        snapshot, _ = self._get_snapshot(
            account=account,
            profile=profile_id,
            kind="report",
            snapshot_id=snapshot_id,
            deadline_at=snapshot_deadline,
            operation=operation,
            owner_run_id=run_id,
            owner_profile=profile_id,
            owner_fencing_token=lease_fencing_token,
        )
        try:
            cursor_values = _decode_cursor(cursor, "attempt")
            cursor_scope = (
                cursor_values["canonicalScopeKey"] if cursor_values else None
            )
            cursor_identity = (
                cursor_values["identityKey"] if cursor_values else None
            )
            conn = snapshot.connection
            self._refresh_snapshot_timeout(snapshot)
            with conn.cursor() as cur:
                scope_cte = """
                    WITH RECURSIVE scope_chain(
                        stored_scope_key, canonical_scope_key
                    ) AS (
                        SELECT scope.scope_key, scope.scope_key
                        FROM public.chatgpt_usage_scopes AS scope
                        UNION
                        SELECT chain.stored_scope_key,
                               redirect.canonical_scope_key
                        FROM scope_chain AS chain
                        JOIN public.chatgpt_usage_scope_redirects AS redirect
                          ON redirect.retired_scope_key =
                             chain.canonical_scope_key
                    ),
                    canonical_scope_map AS (
                        SELECT DISTINCT ON (chain.stored_scope_key)
                               chain.stored_scope_key,
                               chain.canonical_scope_key
                        FROM scope_chain AS chain
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM public.chatgpt_usage_scope_redirects AS redirect
                            WHERE redirect.retired_scope_key =
                                  chain.canonical_scope_key
                        )
                        ORDER BY chain.stored_scope_key, chain.canonical_scope_key
                    ),
                    account_scope_map AS (
                        SELECT map.stored_scope_key,
                               map.canonical_scope_key,
                               binding.collector_account_id
                        FROM canonical_scope_map AS map
                        JOIN public.chatgpt_usage_scope_bindings AS binding
                          ON binding.scope_key = map.canonical_scope_key
                         AND binding.collector_account_id = %s
                         AND binding.binding_state = 'active'
                    )
                """
                self._refresh_snapshot_timeout(snapshot)
                self.ledger.execute_with_deadline(
                    cur,
                    scope_cte
                    + """
                    SELECT COUNT(*)
                    FROM public.chatgpt_usage_attempts AS attempt
                    JOIN account_scope_map AS scope_map
                      ON scope_map.stored_scope_key = attempt.scope_key
                    WHERE NOT attempt.tombstone
                    """,
                    (account,),
                    deadline_at=snapshot.deadline_at,
                    operation=snapshot.operation,
                )
                total_attempts = int(cur.fetchone()[0])
                self._refresh_snapshot_timeout(snapshot)
                self.ledger.execute_with_deadline(
                    cur,
                    scope_cte
                    + """
                    SELECT COUNT(*)
                    FROM public.chatgpt_usage_coverage_gaps AS gap
                    JOIN account_scope_map AS scope_map
                      ON scope_map.stored_scope_key = gap.scope_key
                    WHERE gap.state <> 'resolved'
                    """,
                    (account,),
                    deadline_at=snapshot.deadline_at,
                    operation=snapshot.operation,
                )
                open_gaps = int(cur.fetchone()[0])
                self._refresh_snapshot_timeout(snapshot)
                self.ledger.execute_with_deadline(
                    cur,
                    scope_cte
                    + """
                    SELECT stored_scope_key, canonical_scope_key
                    FROM account_scope_map
                    ORDER BY stored_scope_key
                    """,
                    (account,),
                    deadline_at=snapshot.deadline_at,
                    operation=snapshot.operation,
                )
                scope_rows = cur.fetchall()
                self._refresh_snapshot_timeout(snapshot)
                self.ledger.execute_with_deadline(
                    cur,
                    scope_cte
                    + """
                    SELECT gap.source_kind, gap.source_id, gap.reason,
                           gap.state, gap.first_seen_at, gap.last_seen_at,
                           gap.details
                    FROM public.chatgpt_usage_coverage_gaps AS gap
                    JOIN account_scope_map AS scope_map
                      ON scope_map.stored_scope_key = gap.scope_key
                    WHERE true
                    ORDER BY gap.last_seen_at DESC, gap.source_kind, gap.source_id
                    LIMIT %s
                    """,
                    (account, MAX_QUEUE_PAGE),
                    deadline_at=snapshot.deadline_at,
                    operation=snapshot.operation,
                )
                gap_rows = cur.fetchall()
                self._refresh_snapshot_timeout(snapshot)
                self.ledger.execute_with_deadline(
                    cur,
                    scope_cte
                    + """
                    , scoped_attempts AS (
                        SELECT attempt.*,
                               scope_map.canonical_scope_key,
                               scope_map.collector_account_id
                                   AS canonical_collector_account_id,
                               canonical_scope.provider,
                               canonical_scope.provider_user_id,
                               canonical_scope.workspace_id,
                               canonical_scope.quota_owner_id,
                               canonical_scope.surface AS canonical_surface,
                               canonical_scope.identity_state,
                               revisions.payload AS projection_payload,
                               (
                                   revisions.payload - ARRAY[
                                       'attemptId', 'identityBasis',
                                       'sourceIdentityBasis', 'aliases',
                                       'evidenceMessageIds'
                                   ] || jsonb_build_object(
                                       'sourceTimeBasis', COALESCE(
                                           revisions.payload->'sourceTimeBasis',
                                           revisions.payload->'timeBasis'
                                       ),
                                       'sourceOutcome', COALESCE(
                                           revisions.payload->'sourceOutcome',
                                           revisions.payload->'outcome'
                                       )
                                   )
                               ) AS business_projection
                        FROM public.chatgpt_usage_attempts AS attempt
                        LEFT JOIN public.chatgpt_usage_attempt_revisions AS revisions
                          ON revisions.scope_key = attempt.scope_key
                         AND revisions.attempt_id = attempt.attempt_id
                         AND revisions.revision = attempt.revision
                         AND revisions.projection_fingerprint =
                             attempt.projection_fingerprint
                        JOIN account_scope_map AS scope_map
                          ON scope_map.stored_scope_key = attempt.scope_key
                        JOIN public.chatgpt_usage_scopes AS canonical_scope
                          ON canonical_scope.scope_key = scope_map.canonical_scope_key
                        WHERE NOT attempt.tombstone
                    ),
                    alias_rows AS (
                        SELECT scoped_attempts.canonical_scope_key,
                               scoped_attempts.scope_key,
                               scoped_attempts.attempt_id,
                               aliases.alias_kind,
                               aliases.alias_value
                        FROM scoped_attempts
                        JOIN public.chatgpt_usage_attempt_aliases AS aliases
                          ON aliases.scope_key = scoped_attempts.scope_key
                         AND aliases.attempt_id = scoped_attempts.attempt_id
                    ),
                    alias_values AS (
                        SELECT scoped_attempts.canonical_scope_key,
                               scoped_attempts.scope_key,
                               scoped_attempts.attempt_id,
                               count(alias_rows.alias_value)
                                   FILTER (WHERE alias_rows.alias_kind = 'generation')
                                   AS generation_alias_count,
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
                        FROM scoped_attempts
                        LEFT JOIN alias_rows
                          ON alias_rows.canonical_scope_key =
                                 scoped_attempts.canonical_scope_key
                         AND alias_rows.scope_key = scoped_attempts.scope_key
                         AND alias_rows.attempt_id = scoped_attempts.attempt_id
                        GROUP BY scoped_attempts.canonical_scope_key,
                                 scoped_attempts.scope_key,
                                 scoped_attempts.attempt_id
                    ),
                    generation_keys AS (
                        SELECT canonical_scope_key,
                               scope_key,
                               attempt_id,
                               'generation:' || generation_aliases AS generation_key
                        FROM alias_values
                        WHERE generation_alias_count = 1
                    ),
                    contested_aliases AS (
                        SELECT DISTINCT
                               scope_map.canonical_scope_key,
                               gap.details->'alias'->>'kind' AS alias_kind,
                               gap.details->'alias'->>'value' AS alias_value
                        FROM public.chatgpt_usage_coverage_gaps AS gap
                        JOIN canonical_scope_map AS scope_map
                          ON scope_map.stored_scope_key = gap.scope_key
                        WHERE gap.source_kind = 'attempt_alias'
                          AND gap.reason = 'alias_collision'
                          AND gap.state = 'open'
                          AND gap.details->'alias'->>'kind'
                              IN ('message', 'branch', 'request', 'prompt')
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
                          ON right_alias.canonical_scope_key =
                                 left_alias.canonical_scope_key
                         AND right_alias.alias_kind = left_alias.alias_kind
                         AND right_alias.alias_value = left_alias.alias_value
                        LEFT JOIN generation_keys AS left_generation
                          ON left_generation.scope_key = left_alias.scope_key
                         AND left_generation.attempt_id = left_alias.attempt_id
                        LEFT JOIN generation_keys AS right_generation
                          ON right_generation.scope_key = right_alias.scope_key
                         AND right_generation.attempt_id = right_alias.attempt_id
                        WHERE left_alias.alias_kind IN ('message', 'branch')
                          AND left_generation.generation_key IS NULL
                          AND right_generation.generation_key IS NULL
                          AND (
                              left_alias.scope_key <> right_alias.scope_key
                              OR left_alias.attempt_id <> right_alias.attempt_id
                          )
                    ),
                    strong_reach(
                        canonical_scope_key,
                        node_scope_key,
                        node_attempt_id,
                        root_scope_key,
                        root_attempt_id
                    ) AS (
                        SELECT scoped_attempts.canonical_scope_key,
                               scoped_attempts.scope_key,
                               scoped_attempts.attempt_id,
                               scoped_attempts.scope_key,
                               scoped_attempts.attempt_id
                        FROM scoped_attempts
                        LEFT JOIN generation_keys
                          ON generation_keys.scope_key = scoped_attempts.scope_key
                         AND generation_keys.attempt_id = scoped_attempts.attempt_id
                        WHERE generation_keys.generation_key IS NULL
                        UNION
                        SELECT reach.canonical_scope_key,
                               CASE
                                   WHEN edges.left_scope_key =
                                        reach.node_scope_key
                                    AND edges.left_attempt_id =
                                        reach.node_attempt_id
                                       THEN edges.right_scope_key
                                   ELSE edges.left_scope_key
                               END,
                               CASE
                                   WHEN edges.left_scope_key =
                                        reach.node_scope_key
                                    AND edges.left_attempt_id =
                                        reach.node_attempt_id
                                       THEN edges.right_attempt_id
                                   ELSE edges.left_attempt_id
                               END,
                               reach.root_scope_key,
                               reach.root_attempt_id
                        FROM strong_reach AS reach
                        JOIN strong_edges AS edges
                          ON edges.canonical_scope_key =
                                 reach.canonical_scope_key
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
                        SELECT canonical_scope_key,
                               node_scope_key AS scope_key,
                               node_attempt_id AS attempt_id,
                               MIN(
                                   jsonb_build_array(root_scope_key, root_attempt_id)::text
                               ) AS component_key
                        FROM strong_reach
                        GROUP BY canonical_scope_key, node_scope_key, node_attempt_id
                    ),
                    contested_components AS (
                        SELECT DISTINCT
                               members.canonical_scope_key,
                               members.component_key
                        FROM strong_components AS members
                        JOIN alias_rows AS member_alias
                          ON member_alias.canonical_scope_key =
                                 members.canonical_scope_key
                         AND member_alias.scope_key = members.scope_key
                         AND member_alias.attempt_id = members.attempt_id
                        JOIN contested_aliases
                          ON contested_aliases.canonical_scope_key =
                                 members.canonical_scope_key
                         AND contested_aliases.alias_kind =
                                 member_alias.alias_kind
                         AND contested_aliases.alias_value =
                                 member_alias.alias_value
                    ),
                    component_generation_keys AS (
                        SELECT components.canonical_scope_key,
                               components.component_key,
                               count(DISTINCT generations.generation_key)
                                   AS generation_key_count,
                               MIN(generations.generation_key) AS generation_key,
                               bool_or(
                                   contested_components.component_key IS NOT NULL
                               ) AS contested_association,
                               bool_or(
                                   member_values.generation_alias_count > 1
                               ) AS contradictory_generation_claims
                        FROM (
                            SELECT DISTINCT canonical_scope_key, component_key
                            FROM strong_components
                        ) AS components
                        LEFT JOIN strong_components AS members
                          ON members.canonical_scope_key =
                                 components.canonical_scope_key
                         AND members.component_key = components.component_key
                        LEFT JOIN alias_values AS member_values
                          ON member_values.scope_key = members.scope_key
                         AND member_values.attempt_id = members.attempt_id
                        LEFT JOIN alias_rows AS member_alias
                          ON member_alias.scope_key = members.scope_key
                         AND member_alias.attempt_id = members.attempt_id
                         AND member_alias.alias_kind IN ('message', 'branch')
                        LEFT JOIN alias_rows AS anchor_alias
                          ON anchor_alias.canonical_scope_key =
                                 members.canonical_scope_key
                         AND anchor_alias.alias_kind = member_alias.alias_kind
                         AND anchor_alias.alias_value = member_alias.alias_value
                        LEFT JOIN generation_keys AS generations
                          ON generations.canonical_scope_key =
                                 anchor_alias.canonical_scope_key
                         AND generations.scope_key = anchor_alias.scope_key
                         AND generations.attempt_id = anchor_alias.attempt_id
                        LEFT JOIN contested_components
                          ON contested_components.canonical_scope_key =
                                 components.canonical_scope_key
                         AND contested_components.component_key =
                                 components.component_key
                        GROUP BY components.canonical_scope_key,
                                 components.component_key
                    ),
                    identified AS (
                        SELECT scoped_attempts.*,
                               COALESCE(
                                   alias_values.generation_alias_count > 1
                                   OR (
                                       alias_values.generation_alias_count <> 1
                                       AND (
                                           component_generation_keys.generation_key_count > 1
                                           OR component_generation_keys.contested_association
                                           OR component_generation_keys.contradictory_generation_claims
                                       )
                                   ),
                                   FALSE
                               ) AS ambiguous_generation_component,
                               COALESCE(
                                   alias_values.generation_alias_count = 1
                                   OR component_generation_keys.generation_key_count = 1,
                                   FALSE
                               ) AS canonical_generation_known,
                               CASE
                                   WHEN alias_values.generation_alias_count = 1
                                       THEN 'generation:' ||
                                            alias_values.generation_aliases
                                   WHEN alias_values.generation_alias_count > 1
                                       THEN 'attempt:' || scoped_attempts.scope_key ||
                                            ':' || scoped_attempts.attempt_id
                                   WHEN component_generation_keys.contested_association
                                     OR component_generation_keys.contradictory_generation_claims
                                       THEN 'uncertainty:' ||
                                            strong_components.component_key
                                   WHEN component_generation_keys.generation_key_count = 1
                                       THEN component_generation_keys.generation_key
                                   WHEN component_generation_keys.generation_key_count > 1
                                       THEN 'uncertainty:' ||
                                            strong_components.component_key
                                   WHEN alias_values.message_aliases IS NOT NULL
                                     OR alias_values.branch_aliases IS NOT NULL
                                       THEN 'strong:' || strong_components.component_key
                                   ELSE 'attempt:' || scoped_attempts.scope_key ||
                                        ':' || scoped_attempts.attempt_id
                               END AS identity_key
                        FROM scoped_attempts
                        LEFT JOIN alias_values
                          ON alias_values.canonical_scope_key =
                                 scoped_attempts.canonical_scope_key
                         AND alias_values.scope_key = scoped_attempts.scope_key
                         AND alias_values.attempt_id = scoped_attempts.attempt_id
                        LEFT JOIN strong_components
                          ON strong_components.canonical_scope_key =
                                 scoped_attempts.canonical_scope_key
                         AND strong_components.scope_key = scoped_attempts.scope_key
                         AND strong_components.attempt_id = scoped_attempts.attempt_id
                        LEFT JOIN component_generation_keys
                          ON component_generation_keys.canonical_scope_key =
                                 strong_components.canonical_scope_key
                         AND component_generation_keys.component_key =
                                 strong_components.component_key
                    ),
                    identity_freshness AS (
                        SELECT canonical_scope_key, identity_key,
                               max(observed_at) AS observed_at
                        FROM identified
                        GROUP BY canonical_scope_key, identity_key
                    ),
                    identity_conflicts AS (
                        SELECT identified.canonical_scope_key,
                               identified.identity_key,
                               (
                                   count(DISTINCT identified.business_projection) > 1
                                   OR bool_or(identified.projection_payload IS NULL)
                               ) AS projection_conflict
                        FROM identified
                        JOIN identity_freshness AS freshness
                          ON freshness.canonical_scope_key =
                                 identified.canonical_scope_key
                         AND freshness.identity_key = identified.identity_key
                         AND freshness.observed_at = identified.observed_at
                        GROUP BY identified.canonical_scope_key,
                                 identified.identity_key
                    ),
                    canonicalized_attempts AS (
                        SELECT DISTINCT ON (
                                   identified.canonical_scope_key,
                                   identified.identity_key
                               )
                               identified.*,
                               identity_conflicts.projection_conflict
                        FROM identified
                        JOIN identity_conflicts
                          ON identity_conflicts.canonical_scope_key =
                                 identified.canonical_scope_key
                         AND identity_conflicts.identity_key =
                                 identified.identity_key
                        ORDER BY identified.canonical_scope_key,
                                 identified.identity_key,
                                 observed_at DESC,
                                 (COALESCE(quarantine_state, 'unknown') <> 'clear') DESC,
                                 revision DESC,
                                 last_seen_at DESC,
                                 (scope_key = identified.canonical_scope_key) DESC,
                                 scope_key, attempt_id
                    )
                    SELECT attempt.scope_key, attempt.canonical_scope_key,
                           attempt.identity_key,
                           attempt.attempt_id, attempt.conversation_id,
                           attempt.identity_basis, attempt.time_basis,
                           attempt.attempt_time, attempt.earliest_possible_at,
                           attempt.latest_possible_at, attempt.requested_model_raw,
                           attempt.requested_mode_raw,
                           attempt.requested_reasoning_effort_raw,
                           attempt.recorded_final_model_raw,
                           attempt.resolved_model_raw, attempt.requested_family,
                           attempt.recorded_final_family, attempt.resolved_family,
                           attempt.mapping_version, attempt.outcome,
                           attempt.completed_answer, attempt.generation_started,
                           attempt.surface, attempt.origin, attempt.revision,
                           attempt.warnings, attempt.quarantine_state,
                           attempt.observed_at, attempt.last_seen_at,
                           COALESCE((
                               SELECT jsonb_agg(
                                   jsonb_build_array(alias.alias_kind, alias.alias_value)
                                   ORDER BY alias.alias_kind, alias.alias_value
                               )
                               FROM public.chatgpt_usage_attempt_aliases AS alias
                               WHERE alias.scope_key = attempt.scope_key
                                 AND alias.attempt_id = attempt.attempt_id
                           ), '[]'::jsonb) AS aliases,
                           COALESCE((
                               SELECT revision.payload
                               FROM public.chatgpt_usage_attempt_revisions AS revision
                               WHERE revision.scope_key = attempt.scope_key
                                 AND revision.attempt_id = attempt.attempt_id
                                 AND revision.is_current_projection
                               LIMIT 1
                           ), '{}'::jsonb) AS revision_payload,
                           attempt.canonical_collector_account_id,
                           attempt.provider,
                           attempt.provider_user_id,
                           attempt.workspace_id,
                           attempt.quota_owner_id,
                           attempt.canonical_surface,
                           attempt.identity_state,
                           attempt.ambiguous_generation_component,
                           attempt.projection_conflict
                    FROM canonicalized_attempts AS attempt
                    WHERE (
                          %s::text IS NULL
                          OR attempt.canonical_scope_key > %s::text
                          OR (
                              attempt.canonical_scope_key = %s::text
                              AND attempt.identity_key > %s::text
                          )
                      )
                    ORDER BY attempt.canonical_scope_key, attempt.identity_key
                    LIMIT %s
                    """,
                    (
                        account,
                        cursor_scope,
                        cursor_scope,
                        cursor_scope,
                        cursor_identity,
                        limit + 1,
                    ),
                    deadline_at=snapshot.deadline_at,
                    operation=snapshot.operation,
                )
                rows = cur.fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            attempts = tuple(self._attempt_snapshot(row) for row in rows)
            next_cursor = (
                _encode_cursor(
                    "attempt",
                    {
                        "canonicalScopeKey": str(rows[-1][1]),
                        "identityKey": str(rows[-1][2]),
                    },
                )
                if has_more and rows
                else None
            )
            scope_keys = [
                {
                    "scopeKey": str(row[0]),
                    "canonicalScopeKey": str(row[1]),
                }
                for row in scope_rows
            ]
            coverage_gaps = tuple(
                _coverage_gap_snapshot(row) for row in gap_rows
            )
            history_coverage = _unknown_history_coverage(
                coverage_gaps=coverage_gaps,
                truncated=has_more,
            )
            report_coverage = {
                "history": history_coverage["overall"],
                "overall": history_coverage["overall"],
                "projects": _usage_coverage_level(history_coverage["projects"]),
                "branches": _usage_coverage_level(history_coverage["branches"]),
                "gaps": history_coverage["gaps"],
            }
            result = {
                "snapshotVersion": 1,
                "schemaVersion": "chatgpt-chat-history-v1",
                "collectorAccountId": account,
                "snapshotId": snapshot.snapshot_id if has_more else None,
                "asOf": snapshot.created_at.isoformat(),
                "attempts": attempts,
                "coverage": report_coverage,
                "historyCoverage": history_coverage,
                "coverageDetails": {
                    "truncated": has_more,
                    "totalAttempts": total_attempts,
                    "returnedAttempts": len(attempts),
                    "openCoverageGaps": open_gaps,
                    "gaps": coverage_gaps,
                },
                "scope": {
                    "collectorAccountId": account,
                    "bindings": scope_keys,
                },
                "nextCursor": next_cursor,
                "cursor": next_cursor,
            }
            if not has_more:
                self.close_snapshot(snapshot.snapshot_id)
            return result
        except BaseException:
            self.close_snapshot(snapshot.snapshot_id)
            raise

    def read_candidates(
        self,
        *,
        collector_account_id: str,
        profile_id: str,
        limit: int = 64,
        cursor: Optional[str] = None,
        expected_state_version: Optional[int] = None,
        operation: Any = None,
    ) -> CandidatePage:
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("candidate read limit is outside the supported range")
        _operation_deadline(operation)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._lock_state_authority(cur, account, profile)
            header = self._load_header(cur, account, profile, lock=True)
            current_version = header.state_version if header is not None else 0
            if expected_state_version is None:
                raise LedgerError("collector state version is required")
            if current_version != expected_state_version:
                raise LedgerError("collector state version is stale")
            cursor_values = _decode_cursor(cursor, "candidate")
            cursor_rank = int(cursor_values["rank"]) if cursor_values else None
            cursor_key = cursor_values["key"] if cursor_values else None
            cur.execute(
                """
                SELECT candidate_key, operation, payload, has_more, rank
                FROM public.chatgpt_usage_collector_candidates
                WHERE profile_id = %s AND collector_account_id = %s
                  AND (
                      %s::integer IS NULL
                      OR rank > %s::integer
                      OR (rank = %s::integer AND candidate_key > %s::text)
                  )
                ORDER BY rank, candidate_key
                LIMIT %s
                """,
                (
                    profile,
                    account,
                    cursor_rank,
                    cursor_rank,
                    cursor_rank,
                    cursor_key,
                    limit + 1,
                ),
            )
            rows = cur.fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        next_cursor = (
            _encode_cursor(
                "candidate",
                {"rank": int(rows[-1][4]), "key": str(rows[-1][0])},
            )
            if has_more and rows
            else None
        )
        return CandidatePage(
            tuple(self._safe_candidate(row) for row in rows),
            has_more=has_more,
            cursor=next_cursor,
            state_version=current_version,
        )

    def replace_candidates(
        self,
        *,
        lease: CollectorLease,
        mutations: Sequence[Mapping[str, Any]],
        expected_state_version: Optional[int] = None,
        operation: Any = None,
    ) -> CollectorStateHeader:
        if len(mutations) > MAX_QUEUE_PAGE:
            raise LedgerError("candidate mutation batch exceeds the supported bound")
        if expected_state_version is None:
            raise LedgerError("collector state version is required")
        _operation_deadline(operation)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._require_active_lease(
                cur,
                account=lease.collector_account_id,
                profile_id=lease.profile_id,
                expected=lease,
            )
            self._lock_state_authority(
                cur,
                lease.collector_account_id,
                lease.profile_id,
            )
            current = self._load_header(
                cur,
                lease.collector_account_id,
                lease.profile_id,
                lock=True,
            )
            current_version = current.state_version if current is not None else 0
            if current_version != expected_state_version:
                raise LedgerError("collector state version is stale")
            cur.execute(
                """
                DELETE FROM public.chatgpt_usage_collector_candidates
                WHERE profile_id = %s AND collector_account_id = %s
                """,
                (lease.profile_id, lease.collector_account_id),
            )
            for rank, mutation in enumerate(mutations, start=1):
                self.assert_safe_record(mutation)
                self._apply_candidate_mutation(
                    cur,
                    lease.profile_id,
                    lease.collector_account_id,
                    mutation,
                    rank,
                )
            now = datetime.now(timezone.utc)
            next_version = current_version + 1
            schedule = current.schedule_transition if current else None
            checkpoint = current.checkpoint if current else None
            active_trigger = current.active_trigger if current else None
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_collector_state (
                    profile_id, collector_account_id, state_version,
                    schedule_transition, checkpoint, active_trigger, updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (profile_id, collector_account_id) DO UPDATE SET
                    state_version = EXCLUDED.state_version,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    lease.profile_id,
                    lease.collector_account_id,
                    next_version,
                    _json(schedule),
                    _json(checkpoint),
                    _json(active_trigger),
                    now,
                ),
            )
            return CollectorStateHeader(
                lease.profile_id,
                lease.collector_account_id,
                next_version,
                schedule,
                checkpoint,
                active_trigger,
                now,
            )

    def finish_run(
        self,
        *,
        lease: CollectorLease,
        run_id: str,
        expected_state_version: Optional[int] = None,
        trigger_id: Optional[str] = None,
        outcome: Optional[str] = None,
        summary: Optional[Mapping[str, Any]] = None,
        operation: Any = None,
    ) -> CollectorStateHeader:
        if expected_state_version is None:
            raise LedgerError("collector state version is required")
        if trigger_id is None:
            raise LedgerError("collector trigger is required")
        safe_run = _token(run_id, "run_id")
        safe_trigger = _token(trigger_id, "trigger_id")
        safe_fence = _positive_int(
            lease.lease_fencing_token,
            "lease_fencing_token",
        )
        safe_outcome = _token(outcome, "outcome") if outcome is not None else None
        safe_summary = _safe_terminal_summary(summary)
        _operation_deadline(operation)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._require_active_lease(
                cur,
                account=lease.collector_account_id,
                profile_id=lease.profile_id,
                expected=lease,
            )
            current = self._transition_active_run(
                cur,
                lease,
                safe_run,
                state="idle",
                expected_state_version=expected_state_version,
                trigger_id=safe_trigger,
            )
            trigger = dict(current.active_trigger or {})
            trigger.update({"state": "idle", "runId": safe_run})
            if safe_trigger is not None:
                trigger["triggerId"] = safe_trigger
            if safe_outcome is not None:
                trigger["outcome"] = safe_outcome
            if safe_summary is not None:
                trigger["summary"] = safe_summary
            next_version = current.state_version + 1
            cur.execute(
                """
                UPDATE public.chatgpt_usage_collector_state
                SET active_trigger = %s::jsonb,
                    updated_at = NOW(),
                    state_version = %s
                WHERE profile_id = %s AND collector_account_id = %s
                  AND state_version = %s
                """,
                (
                    _json(trigger),
                    next_version,
                    lease.profile_id,
                    lease.collector_account_id,
                    current.state_version,
                ),
            )
            if cur.rowcount != 1:
                raise LedgerError("collector state version is stale")
            self._release_lease_locked(cur, lease)
            result = CollectorStateHeader(
                lease.profile_id,
                lease.collector_account_id,
                next_version,
                current.schedule_transition,
                current.checkpoint,
                trigger,
                datetime.now(timezone.utc),
            )
        self._close_owned_snapshots(
            account=lease.collector_account_id,
            profile=lease.profile_id,
            run_id=safe_run,
            fencing_token=safe_fence,
        )
        return result

    def cancel(
        self,
        *,
        lease: CollectorLease,
        run_id: str,
        expected_state_version: Optional[int] = None,
        trigger_id: Optional[str] = None,
        outcome: Optional[str] = None,
        summary: Optional[Mapping[str, Any]] = None,
        operation: Any = None,
    ) -> CollectorStateHeader:
        if expected_state_version is None:
            raise LedgerError("collector state version is required")
        if trigger_id is None:
            raise LedgerError("collector trigger is required")
        safe_run = _token(run_id, "run_id")
        safe_trigger = _token(trigger_id, "trigger_id")
        safe_fence = _positive_int(
            lease.lease_fencing_token,
            "lease_fencing_token",
        )
        safe_outcome = _token(outcome, "outcome") if outcome is not None else None
        safe_summary = _safe_terminal_summary(summary)
        _operation_deadline(operation)
        with self.ledger.session(operation=operation) as conn, self.ledger.cursor(
            conn,
            operation=operation,
        ) as cur:
            self._require_active_lease(
                cur,
                account=lease.collector_account_id,
                profile_id=lease.profile_id,
                expected=lease,
            )
            current = self._transition_active_run(
                cur,
                lease,
                safe_run,
                state="cancelled",
                expected_state_version=expected_state_version,
                trigger_id=safe_trigger,
            )
            trigger = dict(current.active_trigger or {})
            trigger.update({"state": "cancelled", "runId": safe_run})
            if safe_trigger is not None:
                trigger["triggerId"] = safe_trigger
            if safe_outcome is not None:
                trigger["outcome"] = safe_outcome
            if safe_summary is not None:
                trigger["summary"] = safe_summary
            next_version = current.state_version + 1
            cur.execute(
                """
                UPDATE public.chatgpt_usage_collector_state
                SET active_trigger = %s::jsonb,
                    updated_at = NOW(),
                    state_version = %s
                WHERE profile_id = %s AND collector_account_id = %s
                  AND state_version = %s
                """,
                (
                    _json(trigger),
                    next_version,
                    lease.profile_id,
                    lease.collector_account_id,
                    current.state_version,
                ),
            )
            if cur.rowcount != 1:
                raise LedgerError("collector state version is stale")
            self._release_lease_locked(cur, lease)
            result = CollectorStateHeader(
                lease.profile_id,
                lease.collector_account_id,
                next_version,
                current.schedule_transition,
                current.checkpoint,
                trigger,
                datetime.now(timezone.utc),
            )
        self._close_owned_snapshots(
            account=lease.collector_account_id,
            profile=lease.profile_id,
            run_id=safe_run,
            fencing_token=safe_fence,
        )
        return result

    def _load_header(
        self,
        cur: psycopg.Cursor,
        account: str,
        profile: str,
        *,
        lock: bool = False,
    ) -> Optional[CollectorStateHeader]:
        lock_sql = "FOR UPDATE" if lock else ""
        cur.execute(
            f"""
            SELECT state_version, schedule_transition, checkpoint,
                   active_trigger, updated_at
            FROM public.chatgpt_usage_collector_state
            WHERE profile_id = %s AND collector_account_id = %s
            {lock_sql}
            """,
            (profile, account),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return CollectorStateHeader(
            profile,
            account,
            int(row[0]),
            dict(row[1]) if row[1] else None,
            dict(row[2]) if row[2] else None,
            dict(row[3]) if row[3] else None,
            ensure_utc(row[4]),
        )

    def _binding_for_scope(
        self,
        cur: psycopg.Cursor,
        scope: LedgerScope,
        account: str,
        expected: Optional[LedgerBinding],
    ) -> LedgerBinding:
        if expected is not None:
            _assert_active_binding(
                cur,
                collector_account_id=account,
                scope_key_value=expected.scope_key,
                binding_generation=expected.binding_generation,
            )
            if scope_key(scope) != expected.scope_key:
                raise LedgerError("collector scope does not match its binding")
            return expected
        cur.execute(
            """
            SELECT scope_key, binding_generation, binding_state
            FROM public.chatgpt_usage_scope_bindings
            WHERE collector_account_id = %s AND binding_state = 'active'
            FOR SHARE
            """,
            (account,),
        )
        row = cur.fetchone()
        if row is None or str(row[0]) != scope_key(scope):
            raise LedgerError("collector scope binding is not initialized")
        return LedgerBinding(
            account,
            str(row[0]),
            int(row[1]),
            str(row[2]),
            "verified",
        )

    def _transition_active_run(
        self,
        cur: psycopg.Cursor,
        lease: CollectorLease,
        run_id: str,
        *,
        state: str,
        expected_state_version: Optional[int],
        trigger_id: Optional[str],
    ) -> CollectorStateHeader:
        current = self._load_header(
            cur,
            lease.collector_account_id,
            lease.profile_id,
            lock=True,
        )
        trigger = current.active_trigger if current else None
        if current is None or not isinstance(trigger, Mapping):
            raise LedgerError("collector run is not active")
        if str(trigger.get("runId")) != run_id:
            raise LedgerError("collector run is not active")
        if trigger_id is not None and str(trigger.get("triggerId")) != trigger_id:
            raise LedgerError("collector trigger is not active")
        trigger_state = trigger.get("state", "active")
        if trigger_state not in {"active", "claimed", "running"}:
            raise LedgerError("collector trigger is not active")
        fencing_token = trigger.get("fencingToken")
        if (
            isinstance(fencing_token, bool)
            or not isinstance(fencing_token, int)
            or fencing_token != lease.lease_fencing_token
        ):
            raise LedgerError("collector trigger fence is stale")
        if (
            expected_state_version is not None
            and current.state_version != expected_state_version
        ):
            raise LedgerError("collector state version is stale")
        if state not in {"idle", "cancelled"}:
            raise LedgerError("collector run transition is unsupported")
        return current

    def _require_active_lease(
        self,
        cur: psycopg.Cursor,
        *,
        account: str,
        profile_id: str,
        expected: CollectorLease,
    ) -> None:
        if (
            expected.profile_id != profile_id
            or expected.collector_account_id != account
            or expected.binding.collector_account_id != account
        ):
            raise LedgerError("collector lease identity is stale")
        self._lock_lease_authority(cur, profile_id)
        _assert_active_binding(
            cur,
            collector_account_id=account,
            scope_key_value=expected.binding.scope_key,
            binding_generation=expected.binding.binding_generation,
        )
        cur.execute(
            """
            SELECT lease_fencing_token, lease_expires_at, scope_key,
                   binding_generation
            FROM public.chatgpt_usage_collector_leases
            WHERE profile_id = %s AND collector_account_id = %s
            FOR UPDATE
            """,
            (profile_id, account),
        )
        row = cur.fetchone()
        if (
            row is None
            or profile_id != expected.profile_id
            or account != expected.collector_account_id
            or int(row[0]) != expected.lease_fencing_token
            or ensure_utc(row[1]) <= datetime.now(timezone.utc)
            or str(row[2]) != expected.binding.scope_key
            or int(row[3]) != expected.binding.binding_generation
        ):
            raise LedgerError("collector lease fence is stale")

    @staticmethod
    def _lock_lease_authority(cur: psycopg.Cursor, profile_id: str) -> None:
        """Acquire the shared lock order: registry, then physical profile."""
        _lock_scope_registry(cur)
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
            (f"chatgpt-collector-profile|{profile_id}",),
        )

    @staticmethod
    def _lock_state_authority(
        cur: psycopg.Cursor,
        account: str,
        profile: str,
    ) -> None:
        """Serialize state reads and absent-row creation for one collector."""
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
            (f"chatgpt-collector-state|{profile}|{account}",),
        )

    def _assert_expected_state_version(
        self,
        cur: psycopg.Cursor,
        *,
        account: str,
        profile: str,
        expected_state_version: Optional[int],
    ) -> None:
        current = self._load_header(cur, account, profile, lock=True)
        current_version = current.state_version if current is not None else 0
        if expected_state_version is None:
            if current is not None:
                raise LedgerError("collector state version is required")
            return
        if current_version != expected_state_version:
            raise LedgerError("collector state version is stale")

    @staticmethod
    def _checkpoint_from_canonical(payload: Mapping[str, Any]) -> Any:
        if "checkpointMutations" in payload:
            mutation = payload.get("checkpointMutations")
            if mutation is None:
                return _UNSET
            if isinstance(mutation, Mapping) and "value" in mutation:
                mutation = mutation.get("value")
            if isinstance(mutation, Mapping):
                return dict(mutation)
            raise LedgerError("collector checkpoint mutation is not an object")
        checkpoint = payload.get("checkpoint")
        if checkpoint is not None:
            if not isinstance(checkpoint, Mapping):
                raise LedgerError("collector checkpoint is not an object")
            return dict(checkpoint)
        discovery = payload.get("discovery")
        if isinstance(discovery, Mapping):
            checkpoint = discovery.get("checkpoint")
            if checkpoint is None:
                return _UNSET
            if not isinstance(checkpoint, Mapping):
                raise LedgerError("discovery checkpoint is not an object")
            return dict(checkpoint)
        page = payload.get("page")
        if isinstance(page, Mapping) and "checkpoint" in page:
            checkpoint = page.get("checkpoint")
            if checkpoint is None:
                return _UNSET
            if not isinstance(checkpoint, Mapping):
                raise LedgerError("page checkpoint is not an object")
            return dict(checkpoint)
        return _UNSET

    def _publish_checkpoint(
        self,
        cur: psycopg.Cursor,
        lease: CollectorLease,
        *,
        run_id: str,
        checkpoint: Any,
        expected_state_version: Optional[int],
        trigger_id: str,
    ) -> CollectorStateHeader:
        current = self._load_header(
            cur,
            lease.collector_account_id,
            lease.profile_id,
            lock=True,
        )
        current_version = current.state_version if current is not None else 0
        if expected_state_version is None:
            if current is not None:
                raise LedgerError("collector state version is required")
        elif expected_state_version != current_version:
            raise LedgerError("collector state version is stale")
        active_trigger = current.active_trigger if current else None
        if (
            current is None
            or not isinstance(active_trigger, Mapping)
            or str(active_trigger.get("runId")) != run_id
        ):
            raise LedgerError("collector run is not active")
        if str(active_trigger.get("triggerId")) != trigger_id:
            raise LedgerError("collector trigger is not active")
        trigger_state = active_trigger.get("state", "active")
        if trigger_state not in {"active", "claimed", "running"}:
            raise LedgerError("collector trigger is not active")
        fencing_token = active_trigger.get("fencingToken")
        if (
            isinstance(fencing_token, bool)
            or not isinstance(fencing_token, int)
            or fencing_token != lease.lease_fencing_token
        ):
            raise LedgerError("collector trigger fence is stale")
        next_version = current_version + 1
        schedule_transition = current.schedule_transition if current else None
        prior_checkpoint = current.checkpoint if current else None
        next_checkpoint = (
            prior_checkpoint if checkpoint is _UNSET else checkpoint
        )
        _validate_state_field("checkpoint", next_checkpoint, STATE_CHECKPOINT_KEYS)
        now = datetime.now(timezone.utc)
        cur.execute(
            """
            INSERT INTO public.chatgpt_usage_collector_state (
                profile_id, collector_account_id, state_version,
                schedule_transition, checkpoint, active_trigger, updated_at
            ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s)
            ON CONFLICT (profile_id, collector_account_id) DO UPDATE SET
                state_version = EXCLUDED.state_version,
                checkpoint = EXCLUDED.checkpoint,
                updated_at = EXCLUDED.updated_at
            """,
            (
                lease.profile_id,
                lease.collector_account_id,
                next_version,
                _json(schedule_transition),
                _json(next_checkpoint),
                _json(active_trigger),
                now,
            ),
        )
        return CollectorStateHeader(
            lease.profile_id,
            lease.collector_account_id,
            next_version,
            schedule_transition,
            next_checkpoint,
            active_trigger,
            now,
        )

    @staticmethod
    def _derive_ingest_operations(
        payload: Mapping[str, Any],
        *,
        default_run_id: Optional[str] = None,
        candidate_mutations: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[Mapping[str, Any], ...]:
        explicit = payload.get("ingestOperations", payload.get("operations"))
        operations: list[Mapping[str, Any]] = []
        if explicit is not None:
            if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
                raise LedgerError("collector ingest operations are not an array")
            for item in explicit:
                if not isinstance(item, Mapping):
                    raise LedgerError("collector ingest operation is not an object")
                operations.append(item)
        source = payload.get("source", payload.get("context"))
        source_map = dict(source) if isinstance(source, Mapping) else {}
        source_map.setdefault("runId", payload.get("runId", default_run_id))
        source_map.setdefault(
            "observedAt",
            payload.get("observedAt", payload.get("scanStartedAt")),
        )
        for key in (
            "sourceKind",
            "sourceId",
            "schemaVersion",
            "provenance",
        ):
            if key not in source_map:
                source_map[key] = payload.get(key)
        for required_key in ("runId", "observedAt", "sourceKind", "sourceId", "schemaVersion"):
            if source_map.get(required_key) is None:
                raise LedgerError(
                    f"collector ingest context is missing {required_key}"
                )
        canonical_sections = [
            section
            for section in (payload.get("discovery"), payload.get("page"))
            if isinstance(section, Mapping)
        ]
        # Discovery/page are the canonical wire records.  Always derive the
        # ledger observation from those records; an undeclared top-level
        # ``observations`` field must not suppress page evidence.
        for section in canonical_sections:
            operations.append(
                PgCollectorState._canonical_observation_operation(
                    section,
                    source_map=source_map,
                    default_run_id=default_run_id,
                    candidate_mutations=candidate_mutations,
                )
            )
        attempts = payload.get("attempts", ())
        if attempts is not None:
            if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
                raise LedgerError("collector attempts are not an array")
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    raise LedgerError("collector attempt is not an object")
                operations.append(
                    {
                        "kind": "attempt",
                        **dict(source_map),
                        **dict(attempt),
                    }
                )
        return tuple(operations)

    @staticmethod
    def _canonical_observation_operation(
        section: Mapping[str, Any],
        *,
        source_map: Mapping[str, Any],
        default_run_id: Optional[str],
        candidate_mutations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        section_source = dict(source_map)
        for key in (
            "runId",
            "observedAt",
            "sourceKind",
            "sourceId",
            "schemaVersion",
            "provenance",
        ):
            section_value = section.get(key)
            source_value = section_source.get(key)
            if section_value is not None and source_value is not None:
                if str(section_value) != str(source_value):
                    raise LedgerError(
                        f"canonical page {key} conflicts with its ingest context"
                    )
            elif section_value is not None and source_value is None:
                raise LedgerError(
                    f"canonical page {key} is not supplied by its ingest context"
                )
        if section_source.get("runId") is None:
            section_source["runId"] = default_run_id
        if section_source.get("observedAt") is None:
            section_source["observedAt"] = section.get("scanStartedAt")
        if section_source.get("observedAt") is None:
            section_checkpoint = section.get("checkpoint")
            if isinstance(section_checkpoint, Mapping):
                section_source["observedAt"] = section_checkpoint.get("updatedAt")
        if (
            section_source.get("runId") is None
            or section_source.get("observedAt") is None
            or section_source.get("sourceKind") is None
            or section_source.get("sourceId") is None
            or section_source.get("schemaVersion") is None
        ):
            raise LedgerError("canonical page is missing its ingest context")
        if "checkpoint" in section:
            checkpoint = section.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise LedgerError("discovery checkpoint is not an object")
            return {
                "kind": "observation",
                **section_source,
                "payload": _discovery_observation_payload(
                    checkpoint,
                    coverage=section.get("coverage"),
                    warnings=section.get("warnings"),
                    candidate_mutations=candidate_mutations,
                ),
            }
        return {
            "kind": "observation",
            **section_source,
            "sourceKind": section_source["sourceKind"],
            "sourceId": section_source["sourceId"],
            "schemaVersion": section_source["schemaVersion"],
            "payload": _page_observation_payload(section),
        }

    @staticmethod
    def _derive_candidate_mutations(
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        has_explicit = "candidateMutations" in payload
        explicit = payload.get("candidateMutations", ())
        if explicit is not None:
            if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
                raise LedgerError("collector candidate mutations are not an array")
            mutations_list: list[Mapping[str, Any]] = []
            for item in explicit:
                if not isinstance(item, Mapping):
                    raise LedgerError("collector candidate mutation is not an object")
                mutations_list.append(item)
            mutations = tuple(mutations_list)
        else:
            mutations = ()
        if has_explicit:
            return mutations
        page = payload.get("page")
        if isinstance(page, Mapping):
            summary = page.get("summary")
            conversation_id = page.get("conversationId")
            if conversation_id is None:
                conversation_id = (
                    summary.get("conversationId")
                    if isinstance(summary, Mapping)
                    else None
                )
            if conversation_id is not None:
                revisit = page.get("revisit")
                mutations += (
                    {
                        "candidateKey": conversation_id,
                        "operation": "replace" if revisit is not None else "remove",
                        "payload": {
                            "summary": summary if isinstance(summary, Mapping) else {},
                            **({"revisit": revisit} if revisit is not None else {}),
                        },
                    },
                )
        return mutations

    @staticmethod
    def _derive_coverage_mutations(
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        explicit = payload.get("coverageMutations", ())
        if explicit is not None:
            if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
                raise LedgerError("collector coverage mutations are not an array")
            mutations: list[Mapping[str, Any]] = []
            for item in explicit:
                if not isinstance(item, Mapping):
                    raise LedgerError("collector coverage mutation is not an object")
                mutations.append(item)
            return tuple(mutations)
        return ()

    def _apply_ingest_operations(
        self,
        cur: psycopg.Cursor,
        scope: LedgerScope,
        binding: LedgerBinding,
        operations: Sequence[Mapping[str, Any]],
        *,
        operation: Any = None,
    ) -> dict[str, int]:
        """Apply metadata-only operations through the current ledger page."""
        page = PgLedgerPage(
            self.ledger,
            cur.connection,
            expected_binding=binding,
            operation=operation,
        )
        accepted = 0
        for operation in operations:
            kind = str(operation["kind"])
            safe_operation = self._safe_operation(operation)
            self.assert_safe_record(safe_operation)
            if kind == "observation":
                context = IngestContext(
                    run_id=str(safe_operation["runId"]),
                    observed_at=parse_datetime(safe_operation["observedAt"]),
                    source_kind=str(safe_operation["sourceKind"]),
                    source_id=str(safe_operation["sourceId"]),
                    schema_version=str(safe_operation["schemaVersion"]),
                    provenance=safe_operation.get("provenance"),
                )
                page.record_observation(scope, context, safe_operation["payload"])
                accepted += 1
            elif kind == "attempt":
                context = IngestContext(
                    run_id=str(safe_operation["runId"]),
                    observed_at=parse_datetime(safe_operation["observedAt"]),
                    source_kind=str(safe_operation["sourceKind"]),
                    source_id=str(safe_operation["sourceId"]),
                    schema_version=str(safe_operation["schemaVersion"]),
                    provenance=safe_operation.get("provenance"),
                )
                page.upsert_attempt(
                    scope,
                    AttemptRecord(
                        attempt_id=str(safe_operation["attemptId"]),
                        conversation_id=str(safe_operation["conversationId"]),
                        identity_basis=str(safe_operation["identityBasis"]),
                        time_basis=str(safe_operation["timeBasis"]),
                        attempt_time=parse_datetime(safe_operation.get("attemptTime")),
                        earliest_possible_at=parse_datetime(
                            safe_operation.get("earliestPossibleAt")
                        ),
                        latest_possible_at=parse_datetime(
                            safe_operation.get("latestPossibleAt")
                        ),
                        requested_model_raw=safe_operation.get("requestedModelRaw"),
                        requested_mode_raw=safe_operation.get("requestedModeRaw"),
                        requested_reasoning_effort_raw=safe_operation.get(
                            "requestedReasoningEffortRaw"
                        ),
                        recorded_final_model_raw=safe_operation.get(
                            "recordedFinalModelRaw"
                        ),
                        resolved_model_raw=safe_operation.get("resolvedModelRaw"),
                        requested_family=safe_operation.get("requestedFamily"),
                        recorded_final_family=safe_operation.get("recordedFinalFamily"),
                        resolved_family=safe_operation.get("resolvedFamily"),
                        mapping_version=str(safe_operation["mappingVersion"]),
                        outcome=str(safe_operation["outcome"]),
                        completed_answer=bool(safe_operation["completedAnswer"]),
                        generation_started=bool(safe_operation["generationStarted"]),
                        surface=str(safe_operation["surface"]),
                        origin=safe_operation.get("origin"),
                        aliases=tuple(
                            (str(alias["kind"]), str(alias["value"]))
                            for alias in safe_operation.get("aliases", ())
                        ),
                        evidence_message_ids=tuple(
                            str(value)
                            for value in safe_operation.get("evidenceMessageIds", ())
                        ),
                        revision=int(safe_operation.get("revision", 1)),
                        warnings=tuple(
                            str(value) for value in safe_operation.get("warnings", ())
                        ),
                        quarantine=safe_operation.get("quarantine"),
                    ),
                    context,
                )
                accepted += 1
            elif kind == "coverageGap":
                page.record_coverage_gap(
                    scope,
                    source_kind=str(safe_operation["sourceKind"]),
                    source_id=str(safe_operation["sourceId"]),
                    reason=str(safe_operation["reason"]),
                    state=str(safe_operation.get("state", "open")),
                    details=safe_operation.get("details"),
                    seen_at=parse_datetime(safe_operation["seenAt"]),
                )
                accepted += 1
            elif kind == "coverageResolution":
                page.resolve_coverage_gaps(
                    scope,
                    source_kind=str(safe_operation["sourceKind"]),
                    source_id=str(safe_operation["sourceId"]),
                    seen_at=parse_datetime(safe_operation["seenAt"]),
                )
                accepted += 1
        return {"accepted": accepted}

    @staticmethod
    def _safe_operation(operation: Mapping[str, Any]) -> dict[str, Any]:
        safe = {
            key: value
            for key, value in operation.items()
            if key not in {"payload", "aliases", "quarantine", "details"}
        }
        if isinstance(operation.get("payload"), Mapping):
            safe["payload"] = _observation_envelope(operation["payload"])
        if isinstance(operation.get("aliases"), Sequence) and not isinstance(
            operation.get("aliases"), (str, bytes)
        ):
            aliases: list[dict[str, str]] = []
            for alias in operation.get("aliases", ()):
                if isinstance(alias, Mapping):
                    alias_kind = _token(alias.get("kind"), "alias.kind")
                    alias_value = _token(alias.get("value"), "alias.value")
                elif isinstance(alias, Sequence) and not isinstance(alias, (str, bytes)) and len(alias) == 2:
                    alias_kind = _token(alias[0], "alias.kind")
                    alias_value = _token(alias[1], "alias.value")
                else:
                    raise LedgerError("collector alias is not a pair")
                aliases.append({"kind": alias_kind, "value": alias_value})
            safe["aliases"] = aliases
        if isinstance(operation.get("quarantine"), Mapping):
            safe["quarantine"] = _safe_metadata_mapping(
                "operation.quarantine",
                operation["quarantine"],
                allowed_keys={"state", "warnings", "timestamps"},
            )
        if isinstance(operation.get("details"), Mapping):
            safe["details"] = _observation_envelope(operation["details"])
        if isinstance(operation.get("provenance"), Mapping):
            safe["provenance"] = _copy_metadata_value(
                operation["provenance"],
                "operation.provenance",
            )
        assert_no_secrets(safe)
        return safe

    @staticmethod
    def _validate_ingest_operation(operation: Mapping[str, Any]) -> None:
        kind = _token(operation.get("kind"), "operation.kind")
        if kind not in {
            "observation",
            "attempt",
            "coverageGap",
            "coverageResolution",
        }:
            raise LedgerError("collector page contains an unsupported ingest operation")

    def _apply_coverage_mutations(
        self,
        cur: psycopg.Cursor,
        scope: LedgerScope,
        binding: LedgerBinding,
        mutations: Sequence[Mapping[str, Any]],
        *,
        operation: Any = None,
    ) -> None:
        if not mutations:
            return
        operations: list[Mapping[str, Any]] = []
        for mutation in mutations:
            if not isinstance(mutation, Mapping):
                raise LedgerError("collector coverage mutation is not an object")
            operation = str(mutation.get("operation", "record"))
            if operation in {"resolve", "resolved"}:
                operations.append(
                    {
                        "kind": "coverageResolution",
                        "sourceKind": mutation.get("sourceKind"),
                        "sourceId": mutation.get("sourceId"),
                        "seenAt": mutation.get("seenAt"),
                    }
                )
            else:
                operations.append(
                    {
                        "kind": "coverageGap",
                        "sourceKind": mutation.get("sourceKind"),
                        "sourceId": mutation.get("sourceId"),
                        "reason": mutation.get("reason"),
                        "state": mutation.get("state", "open"),
                        "details": mutation.get("details"),
                        "seenAt": mutation.get("seenAt"),
                    }
                )
        self._apply_ingest_operations(
            cur,
            scope,
            binding,
            operations,
            operation=operation,
        )

    def _apply_candidate_mutation(
        self,
        cur: psycopg.Cursor,
        profile_id: str,
        account: str,
        mutation: Mapping[str, Any],
        rank: int,
    ) -> None:
        candidate_key = _token(mutation.get("candidateKey"), "candidate_key")
        operation = _enum(mutation.get("operation"), {"replace", "remove"}, "candidate_operation")
        payload = mutation.get("payload")
        if payload is None and isinstance(mutation.get("candidate"), Mapping):
            payload = mutation.get("candidate")
        candidate_payload = (
            _candidate_envelope(payload) if isinstance(payload, Mapping) else {}
        )
        safe_payload = {
            "candidateKey": candidate_key,
            "operation": operation,
            "payload": candidate_payload,
            "hasMore": bool(mutation.get("hasMore", False)),
        }
        assert_no_secrets(safe_payload)
        serialized = _json(candidate_payload)
        if len(serialized.encode("utf-8")) > MAX_QUEUE_ITEM_BYTES:
            raise LedgerError("collector candidate payload exceeds the supported bound")
        if operation == "remove":
            cur.execute(
                """
                DELETE FROM public.chatgpt_usage_collector_candidates
                WHERE profile_id = %s AND collector_account_id = %s
                  AND candidate_key = %s
                """,
                (profile_id, account, candidate_key),
            )
            return
        cur.execute(
            """
            INSERT INTO public.chatgpt_usage_collector_candidates (
                profile_id, collector_account_id, candidate_key, rank,
                operation, payload, has_more
            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (profile_id, collector_account_id, candidate_key)
            DO UPDATE SET
                rank = EXCLUDED.rank,
                operation = EXCLUDED.operation,
                payload = EXCLUDED.payload,
                has_more = EXCLUDED.has_more
            """,
            (
                profile_id,
                account,
                candidate_key,
                int(mutation.get("rank", rank)),
                operation,
                serialized,
                bool(mutation.get("hasMore", False)),
            ),
        )

    def _get_snapshot(
        self,
        *,
        account: str,
        profile: Optional[str],
        kind: str,
        snapshot_id: Optional[str],
        deadline_at: Optional[datetime] = None,
        owner_run_id: str,
        owner_profile: str,
        owner_fencing_token: int,
        operation: Any = None,
    ) -> tuple[_ReadSnapshot, bool]:
        safe_owner_run = _token(owner_run_id, "run_id")
        safe_owner_profile = _token(owner_profile, "profile_id")
        if profile is not None and profile != safe_owner_profile:
            raise LedgerError("collector snapshot profile does not match its owner")
        safe_owner_fence = _positive_int(owner_fencing_token, "lease_fencing_token")
        if snapshot_id is not None:
            return self._reuse_snapshot(
                account=account,
                profile=safe_owner_profile,
                kind=kind,
                snapshot_id=snapshot_id,
                owner_run_id=safe_owner_run,
                owner_fencing_token=safe_owner_fence,
                operation=operation,
            )
        requested_deadline = (
            ensure_utc(deadline_at) if deadline_at is not None else None
        )
        created_at = datetime.now(timezone.utc)
        snapshot_deadline = created_at + timedelta(seconds=MAX_SNAPSHOT_AGE_SECONDS)
        if requested_deadline is not None:
            if requested_deadline <= created_at:
                raise LedgerError("collector read snapshot deadline has expired")
            snapshot_deadline = min(snapshot_deadline, requested_deadline)
        conn: Optional[psycopg.Connection] = None
        try:
            self._remaining_snapshot_ms(snapshot_deadline, created_at=created_at)
            conn = self.ledger.connect(
                deadline_at=snapshot_deadline,
                operation=operation,
            )
            # PgLedger.connect() opens a setup transaction. Roll it back before
            # starting the bounded repeatable-read snapshot so the transaction-
            # local timeout settings apply to the actual read.
            self._remaining_snapshot_ms(snapshot_deadline, created_at=created_at)
            self.ledger.rollback_with_deadline(
                conn,
                deadline_at=snapshot_deadline,
                operation=operation,
            )
            self._remaining_snapshot_ms(snapshot_deadline, created_at=created_at)
            self.ledger.command_with_deadline(
                conn,
                "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY",
                deadline_at=snapshot_deadline,
                operation=operation,
            )
            self._remaining_snapshot_ms(snapshot_deadline, created_at=created_at)
            safe_snapshot = uuid4().hex
            snapshot = _ReadSnapshot(
                safe_snapshot,
                account,
                safe_owner_profile,
                kind,
                conn,
                created_at,
                snapshot_deadline,
                safe_owner_run,
                safe_owner_profile,
                safe_owner_fence,
                operation,
            )
            self._refresh_snapshot_timeout(snapshot)
            with conn.cursor() as cur:
                self.ledger.execute_with_deadline(
                    cur,
                    """
                    SELECT lease.lease_fencing_token, lease.lease_expires_at,
                           state.active_trigger
                    FROM public.chatgpt_usage_collector_leases AS lease
                    LEFT JOIN public.chatgpt_usage_collector_state AS state
                      ON state.profile_id = lease.profile_id
                     AND state.collector_account_id = lease.collector_account_id
                    WHERE lease.profile_id = %s
                      AND lease.collector_account_id = %s
                    """,
                    (safe_owner_profile, account),
                    deadline_at=snapshot_deadline,
                    operation=operation,
                )
                owner_row = cur.fetchone()
            active_trigger = (
                owner_row[2] if owner_row is not None else None
            )
            if (
                owner_row is None
                or int(owner_row[0]) != safe_owner_fence
                or ensure_utc(owner_row[1]) <= created_at
                or not isinstance(active_trigger, Mapping)
                or str(active_trigger.get("runId")) != safe_owner_run
                or active_trigger.get("state", "active")
                not in {"active", "claimed", "running"}
            ):
                raise LedgerError("collector read snapshot owner lease is stale")
            self._snapshots[safe_snapshot] = snapshot
            return snapshot, True
        except BaseException:
            if conn is not None:
                conn.close()
            raise

    def _reuse_snapshot(
        self,
        *,
        account: str,
        profile: str,
        kind: str,
        snapshot_id: str,
        owner_run_id: str,
        owner_fencing_token: int,
        operation: Any = None,
    ) -> tuple[_ReadSnapshot, bool]:
        safe_snapshot = _token(snapshot_id, "snapshot_id")
        snapshot = self._snapshots.get(safe_snapshot)
        if snapshot is None:
            raise LedgerError("collector read snapshot is unavailable")
        if (
            snapshot.account != account
            or snapshot.profile != profile
            or snapshot.kind != kind
            or snapshot.owner_run_id != owner_run_id
            or snapshot.owner_profile != profile
            or snapshot.owner_fencing_token != owner_fencing_token
        ):
            raise LedgerError("collector read snapshot identity is stale")
        now = datetime.now(timezone.utc)
        age = (now - snapshot.created_at).total_seconds()
        if age > MAX_SNAPSHOT_AGE_SECONDS or now >= snapshot.deadline_at:
            self.close_snapshot(safe_snapshot)
            raise LedgerError("collector read snapshot expired")
        try:
            if operation is not None:
                snapshot.operation = operation
            self._refresh_snapshot_timeout(snapshot)
        except BaseException:
            self.close_snapshot(safe_snapshot)
            raise
        return snapshot, False

    def _close_owned_snapshots(
        self,
        *,
        account: str,
        profile: str,
        run_id: str,
        fencing_token: int,
    ) -> None:
        safe_run = _token(run_id, "run_id")
        safe_profile = _token(profile, "profile_id")
        safe_fence = _positive_int(fencing_token, "lease_fencing_token")
        for snapshot_id, snapshot in tuple(self._snapshots.items()):
            if (
                snapshot.account == account
                and snapshot.profile == safe_profile
                and snapshot.owner_run_id == safe_run
                and snapshot.owner_profile == safe_profile
                and snapshot.owner_fencing_token == safe_fence
            ):
                self.close_snapshot(snapshot_id)

    def _refresh_snapshot_timeout(self, snapshot: _ReadSnapshot) -> None:
        remaining_ms = self._remaining_snapshot_ms(snapshot.deadline_at)
        with snapshot.connection.cursor() as cur:
            self.ledger.execute_with_deadline(
                cur,
                "SELECT set_config('statement_timeout', %s, true)",
                (f"{min(self.ledger.statement_timeout_ms, remaining_ms)}ms",),
                deadline_at=snapshot.deadline_at,
                operation=snapshot.operation,
            )
            remaining_ms = self._remaining_snapshot_ms(snapshot.deadline_at)
            self.ledger.execute_with_deadline(
                cur,
                "SELECT set_config('lock_timeout', %s, true)",
                (f"{min(self.ledger.lock_timeout_ms, remaining_ms)}ms",),
                deadline_at=snapshot.deadline_at,
                operation=snapshot.operation,
            )

    @staticmethod
    def _remaining_snapshot_ms(
        deadline_at: datetime,
        *,
        created_at: Optional[datetime] = None,
    ) -> int:
        now = datetime.now(timezone.utc)
        if now >= deadline_at:
            raise LedgerError("collector read snapshot expired")
        if created_at is not None and (
            now - created_at
        ).total_seconds() > MAX_SNAPSHOT_AGE_SECONDS:
            raise LedgerError("collector read snapshot expired")
        return max(1, int((deadline_at - now).total_seconds() * 1000))

    @staticmethod
    def _observation_snapshot(row: Sequence[Any]) -> dict[str, Any]:
        payload = dict(row[8]) if isinstance(row[8], Mapping) else {}
        item = {
            "observationId": str(row[0]),
            "scopeKey": str(row[1]),
            "payload": payload,
            "context": {
                "runId": str(row[4]),
                "observedAt": ensure_utc(row[5]).isoformat(),
                "sourceKind": str(row[2]),
                "sourceId": str(row[3]),
                "schemaVersion": str(row[6]),
                "provenance": dict(row[7]) if isinstance(row[7], Mapping) else {},
            },
        }
        assert_no_secrets(item)
        return item

    @staticmethod
    def _safe_candidate(row: Sequence[Any]) -> dict[str, Any]:
        payload = row[2] if isinstance(row[2], Mapping) else {}
        # Rows written by the pre-FCB body wrapped the original candidate.
        # Read them compatibly, but persist only the original payload.
        if (
            isinstance(payload, Mapping)
            and isinstance(payload.get("payload"), Mapping)
            and payload.get("candidateKey") is not None
        ):
            payload = payload["payload"]
        value = {
            "candidateKey": str(row[0]),
            "operation": str(row[1]),
            "payload": dict(payload),
            "hasMore": bool(row[3]),
        }
        assert_no_secrets(value)
        return value

    @staticmethod
    def _attempt_snapshot(row: Sequence[Any]) -> dict[str, Any]:
        revision_payload = (
            row[30] if len(row) > 30 and isinstance(row[30], Mapping) else {}
        )
        aliases = (
            row[29] if len(row) > 29 and isinstance(row[29], Sequence) else ()
        )
        revision_quarantine = (
            revision_payload.get("quarantine")
            if isinstance(revision_payload.get("quarantine"), Mapping)
            else None
        )
        ambiguous_generation = bool(row[38]) if len(row) > 38 else False
        projection_conflict = bool(row[39]) if len(row) > 39 else False
        quarantine = _validated_quarantine_snapshot(
            revision_quarantine,
            fallback_state=(
                "quarantined"
                if ambiguous_generation or projection_conflict
                else row[26] if len(row) > 26 else "unknown"
            ),
        )
        if ambiguous_generation or projection_conflict:
            quarantine["state"] = "quarantined"
        item = {
            "scopeKey": str(row[0]),
            "canonicalScopeKey": str(row[1]),
            "identityKey": str(row[2]),
            "attemptId": str(row[3]),
            "conversationId": str(row[4]),
            "identityBasis": str(row[5]),
            "timeBasis": str(row[6]),
            "attemptTime": row[7].isoformat() if row[7] else None,
            "earliestPossibleAt": row[8].isoformat() if row[8] else None,
            "latestPossibleAt": row[9].isoformat() if row[9] else None,
            "requestedModelRaw": row[10],
            "requestedModeRaw": row[11],
            "requestedReasoningEffortRaw": row[12],
            "recordedFinalModelRaw": row[13],
            "resolvedModelRaw": row[14],
            "requestedFamily": row[15],
            "recordedFinalFamily": row[16],
            "resolvedFamily": row[17],
            "mappingVersion": str(row[18]),
            "outcome": str(row[19]),
            "completedAnswer": bool(row[20]),
            "generationStarted": bool(row[21]),
            "surface": str(row[22]),
            "origin": row[23],
            "revision": int(row[24]),
            "warnings": [
                *list(row[25] or []),
                *(
                    ["ambiguous_generation_identity"]
                    if ambiguous_generation
                    else []
                ),
                *(
                    ["projection_conflict"]
                    if projection_conflict
                    else []
                ),
            ],
            "quarantine": quarantine,
            "quarantineState": str(row[26]),
            "observedAt": row[27].isoformat() if row[27] else None,
            "lastSeenAt": row[28].isoformat() if row[28] else None,
            "aliases": [
                [str(alias[0]), str(alias[1])]
                for alias in aliases
                if isinstance(alias, Sequence) and len(alias) == 2
            ],
            "evidenceMessageIds": [
                str(value)
                for value in revision_payload.get("evidenceMessageIds", ())
            ],
            "revisionPayload": _copy_metadata_value(
                revision_payload,
                "report.attempt.revisionPayload",
            ),
            "scope": {
                "collectorAccountId": str(row[31]),
                "provider": str(row[32]),
                "providerUserId": row[33],
                "workspaceId": row[34],
                "quotaOwnerId": row[35],
                "surface": str(row[36]),
            },
        }
        assert_no_secrets(item)
        return item


def _coverage_gap_snapshot(row: Sequence[Any]) -> dict[str, Any]:
    details = dict(row[6]) if len(row) > 6 and isinstance(row[6], Mapping) else {}
    item = {
        "sourceKind": str(row[0]),
        "sourceId": str(row[1]),
        "reason": str(row[2]),
        "state": str(row[3]),
        "firstSeenAt": ensure_utc(row[4]).isoformat() if row[4] else None,
        "lastSeenAt": ensure_utc(row[5]).isoformat() if row[5] else None,
        "details": details,
    }
    assert_no_secrets(item)
    return item


def _validated_quarantine_snapshot(
    value: Any,
    *,
    fallback_state: Any,
) -> dict[str, Any]:
    fallback = (
        str(fallback_state)
        if isinstance(fallback_state, str)
        and fallback_state in {"clear", "quarantined", "unknown"}
        else "unknown"
    )
    if value is None:
        return {"state": fallback, "warnings": [], "timestamps": []}
    if not isinstance(value, Mapping):
        raise LedgerError("stored attempt quarantine is not an object")
    state = value.get("state", fallback)
    if state not in {"clear", "quarantined", "unknown"}:
        raise LedgerError("stored attempt quarantine state is invalid")
    raw_warnings = value.get("warnings", ())
    if not isinstance(raw_warnings, Sequence) or isinstance(
        raw_warnings, (str, bytes)
    ):
        raise LedgerError("stored attempt quarantine warnings are invalid")
    warnings: list[str] = []
    for warning in raw_warnings:
        if not isinstance(warning, str) or sanitize_token(warning) is None:
            raise LedgerError("stored attempt quarantine warning is invalid")
        warnings.append(warning)
    raw_timestamps = value.get("timestamps", ())
    if not isinstance(raw_timestamps, Sequence) or isinstance(
        raw_timestamps, (str, bytes)
    ):
        raise LedgerError("stored attempt quarantine timestamps are invalid")
    timestamps: list[dict[str, str]] = []
    for evidence in raw_timestamps:
        if not isinstance(evidence, Mapping):
            raise LedgerError("stored attempt quarantine timestamp is invalid")
        unknown_keys = set(evidence) - STATE_QUARANTINE_TIMESTAMP_KEYS
        if unknown_keys:
            raise LedgerError("stored attempt quarantine timestamp has unknown fields")
        field = evidence.get("field")
        value_text = evidence.get("value")
        observed_at = evidence.get("observedAt")
        if field not in {
            "createdAt",
            "updatedAt",
            "attemptTime",
            "earliestPossibleAt",
            "latestPossibleAt",
        }:
            raise LedgerError("stored attempt quarantine timestamp field is invalid")
        if not isinstance(value_text, str) or not isinstance(observed_at, str):
            raise LedgerError("stored attempt quarantine timestamp values are invalid")
        try:
            parse_datetime(observed_at)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise LedgerError(
                "stored attempt quarantine timestamp observedAt is invalid"
            ) from exc
        normalized = {
            "field": field,
            "value": value_text,
            "observedAt": observed_at,
        }
        if "messageId" in evidence:
            message_id = evidence["messageId"]
            if not isinstance(message_id, str) or sanitize_token(message_id) is None:
                raise LedgerError("stored attempt quarantine messageId is invalid")
            normalized["messageId"] = message_id
        timestamps.append(normalized)
    result = {
        "state": str(state),
        "warnings": warnings,
        "timestamps": timestamps,
    }
    assert_no_secrets(result)
    return result


def _unknown_scope_coverage(scope: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "status": "not_started",
        "coverage": "unknown",
        "pagesFetched": 0,
        "candidates": 0,
        "continuation": None,
        "paginationState": "unknown",
        "candidateCutoff": "",
        "warnings": ["coverage_not_persisted_in_report_snapshot"],
        "olderHistoryAudit": {
            "enabled": False,
            "status": "disabled",
            "continuation": None,
            "pagesFetched": 0,
            "conversationsAudited": 0,
            "lastStartedAt": None,
            "lastPageAt": None,
            "lastCompletedAt": None,
        },
    }


def _usage_coverage_level(value: Any) -> str:
    if value in {"complete", "partial", "unknown"}:
        return str(value)
    if value in {
        "validated_for_discovered_projects",
        "version_metadata_observed",
        "active_branch_only",
    }:
        return "partial"
    return "unknown"


def _unknown_history_coverage(
    *,
    coverage_gaps: Sequence[Mapping[str, Any]],
    truncated: bool,
) -> dict[str, Any]:
    gaps = [
        str(gap["reason"])
        for gap in coverage_gaps
        if isinstance(gap, Mapping) and gap.get("reason") is not None
    ]
    if truncated:
        gaps.append("report_snapshot_truncated")
    return {
        "active": _unknown_scope_coverage("active"),
        "archived": _unknown_scope_coverage("archived"),
        "projects": "unknown",
        "branches": "unknown",
        "olderHistoryAudit": {
            "enabled": False,
            "status": "disabled",
            "active": _unknown_scope_coverage("active")["olderHistoryAudit"],
            "archived": _unknown_scope_coverage("archived")["olderHistoryAudit"],
        },
        "overall": "unknown",
        "gaps": sorted(set(gaps)),
    }


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    summary = candidate.get("summary")
    if isinstance(summary, Mapping):
        conversation_id = summary.get("conversationId")
        if conversation_id is not None:
            return str(conversation_id)
    conversation_id = candidate.get("conversationId")
    if conversation_id is not None:
        return str(conversation_id)
    raise LedgerError("discovery candidate is missing conversationId")


def _discovery_observation_payload(
    checkpoint: Mapping[str, Any],
    *,
    coverage: Any = None,
    warnings: Any = None,
    candidate_mutations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    items: list[Mapping[str, Any]] = []
    for mutation in candidate_mutations:
        if not isinstance(mutation, Mapping):
            continue
        if mutation.get("queueKind") != "candidate":
            continue
        if mutation.get("operation") != "replace":
            continue
        payload = mutation.get("payload")
        if not isinstance(payload, Mapping):
            continue
        candidate = payload.get("candidate", payload)
        if not isinstance(candidate, Mapping):
            continue
        summary = candidate.get("summary", candidate)
        if isinstance(summary, Mapping):
            items.append(dict(summary))
    pagination_state = checkpoint.get("paginationState", "unknown")
    coverage = (
        "validated_page"
        if pagination_state == "complete"
        else "partial"
        if pagination_state in {"continuation", "budget_exhausted"}
        else "unrecognized"
    )
    payload = {
        "items": items,
        "continuation": checkpoint.get("continuation"),
        "exhausted": checkpoint.get("continuation") is None
        and checkpoint.get("status") == "complete",
        "paginationState": pagination_state,
        "coverage": coverage,
        "warnings": checkpoint.get("warnings", ()),
        "updatedAt": checkpoint.get("updatedAt"),
        "schemaVersion": "chatgpt-chat-history-v1",
        "surface": "chat",
    }
    for key in (
        "accountId",
        "mode",
        "range",
        "candidateCutoff",
        "pagesFetched",
        "pageBudget",
        "lastCompleteDiscoveryStartedAt",
        "lastPageAt",
    ):
        if checkpoint.get(key) is not None:
            payload[key] = checkpoint[key]
    if coverage is not None:
        payload["coverage"] = coverage
    if warnings is not None:
        payload["warnings"] = warnings
    return payload


def _page_observation_payload(page: Mapping[str, Any]) -> dict[str, Any]:
    detail = page.get("detail")
    payload: dict[str, Any] = dict(detail) if isinstance(detail, Mapping) else {}
    summary = page.get("summary")
    if isinstance(summary, Mapping):
        for key in (
            "conversationId",
            "createdAt",
            "updatedAt",
            "currentNode",
            "surface",
            "origin",
            "hasVersions",
            "workspaceId",
            "projectId",
            "coverage",
        ):
            if key in summary and key not in payload:
                payload[key] = summary[key]
    messages = page.get("messages")
    if messages is None and isinstance(detail, Mapping):
        messages = detail.get("messages")
    if messages is not None:
        payload["messages"] = messages
    if page.get("coverage") is not None:
        payload["coverage"] = page.get("coverage")
    if page.get("warnings") is not None:
        payload["warnings"] = page.get("warnings")
    if page.get("nextContinuation") is not None:
        payload["continuation"] = page.get("nextContinuation")
    elif "continuation" not in payload:
        payload["continuation"] = None
    if "exhausted" not in payload:
        payload["exhausted"] = page.get("nextContinuation") is None
    return payload


def _account(value: str) -> str:
    return _token(value, "collector_account_id")


def _token(value: Any, field_name: str) -> str:
    normalized = sanitize_token(value)
    if normalized is None:
        raise LedgerError(f"{field_name} is not a supported metadata token")
    return normalized


def _enum(value: Any, allowed: set[str], field_name: str) -> str:
    normalized = _token(value, field_name)
    if normalized not in allowed:
        raise LedgerError(f"{field_name} is unsupported")
    return normalized


def _json(value: Optional[Mapping[str, Any]]) -> str:
    return json.dumps(value or {}, separators=(",", ":"), default=str)


def _validate_state_field(
    field_name: str,
    value: Any,
    allowed_keys: Optional[set[str]],
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise LedgerError(f"collector state field {field_name} is not an object")
    serialized = json.dumps(value, separators=(",", ":"), default=str)
    if len(serialized.encode("utf-8")) > MAX_STATE_FIELD_BYTES:
        raise LedgerError(f"collector state field {field_name} exceeds the supported bound")
    for raw_key, child in value.items():
        if not isinstance(raw_key, str):
            raise LedgerError(f"collector state field {field_name} has a non-string key")
        key = raw_key
        if key in STATE_FORBIDDEN_KEYS:
            raise LedgerError(f"collector state field {field_name} contains unsupported content")
        if allowed_keys is not None and key not in allowed_keys:
            raise LedgerError(f"collector state field {field_name} contains an unknown key")
        shape = _state_child_shape(key, field_name)
        if shape == "mapping":
            if child is None:
                continue
            if not isinstance(child, Mapping):
                raise LedgerError(
                    f"collector state field {field_name}.{key} must be an object"
                )
            nested_keys = _nested_state_keys(key, field_name)
            if nested_keys is None:
                raise LedgerError(
                    f"collector state field {field_name}.{key} has an unsupported object"
                )
            _validate_state_field(f"{field_name}.{key}", child, nested_keys)
        elif shape == "array":
            if child is None:
                continue
            if not isinstance(child, Sequence) or isinstance(child, (str, bytes)):
                raise LedgerError(
                    f"collector state field {field_name}.{key} must be an array"
                )
            if len(child) > MAX_QUEUE_PAGE:
                raise LedgerError(f"collector state field {field_name}.{key} is too large")
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    nested_keys = _state_array_item_keys(key, field_name)
                    if nested_keys is None:
                        raise LedgerError(
                            f"collector state field {field_name}.{key} has unsupported objects"
                        )
                    _validate_state_field(
                        f"{field_name}.{key}[{index}]",
                        item,
                        nested_keys,
                    )
                elif isinstance(item, (str, int, float, bool)) or item is None:
                    if key in STATE_STRING_ARRAY_KEYS and not isinstance(item, str):
                        raise LedgerError(
                            f"collector state field {field_name}.{key}[{index}] must be a string"
                        )
                    if _state_array_item_keys(key, field_name) is not None:
                        raise LedgerError(
                            f"collector state field {field_name}.{key}[{index}] must be an object"
                        )
                    _validate_state_scalar(
                        f"{field_name}.{key}[{index}]",
                        key,
                        item,
                    )
                    continue
                else:
                    raise LedgerError(
                        f"collector state field {field_name}.{key} contains unsupported data"
                    )
        elif isinstance(child, (Mapping, Sequence)) and not isinstance(
            child, (str, bytes)
        ):
            raise LedgerError(
                f"collector state field {field_name}.{key} must be a scalar"
            )
        elif child is not None:
            _validate_state_scalar(field_name, key, child)


def _state_child_shape(key: str, field_name: str) -> str:
    """Return the contract shape for one state field in its parent context."""
    if key in STATE_STRING_ARRAY_KEYS or key in STATE_OBJECT_ARRAY_KEYS:
        return "array"
    if key == "scope":
        return "mapping" if field_name == "scheduleTransition" else "scalar"
    if key == "coverage":
        return "mapping" if field_name == "terminalSummary" else "scalar"
    if key in {"active", "archived"}:
        if (
            "coverage" in field_name.lower()
            or "olderHistoryAudit" in field_name
            or field_name == "scheduleTransition"
        ):
            return "mapping"
        return "scalar"
    if key in {
        "pending",
        "range",
        "summary",
        "olderHistoryAudit",
        "historyCoverage",
        "identity",
        "accountState",
        "terminalSummary",
        "candidate",
        "revisit",
        "malformedPage",
        "outstandingGeneration",
        "quarantine",
    }:
        return "mapping"
    return "scalar"


def _state_array_item_keys(
    key: str,
    field_name: str,
) -> Optional[set[str]]:
    if key == "candidateQueue":
        return STATE_CANDIDATE_KEYS
    if key == "timestamps":
        return STATE_QUARANTINE_TIMESTAMP_KEYS
    if key == "scopes" and "revisit" not in field_name:
        return STATE_SCOPE_COVERAGE_KEYS
    return None


def _validate_state_scalar(field_name: str, key: str, value: Any) -> None:
    """Validate scalar fields with the schema that owns the field."""
    audit_path = "olderHistoryAudit" in field_name
    if key == "continuation":
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise LedgerError(
                f"collector state field {field_name}.{key} must be a cursor or offset"
            )
    elif key == "scopes":
        if value not in {"active", "archived"}:
            raise LedgerError(
                f"collector state field {field_name}.{key} has an invalid scope"
            )
    elif key == "lastCompletedAt" and audit_path:
        if not isinstance(value, str):
            raise LedgerError(
                f"collector state field {field_name}.{key} must be an ISO timestamp"
            )
        try:
            parse_datetime(value)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise LedgerError(
                f"collector state field {field_name}.{key} must be an ISO timestamp"
            ) from exc
    elif key in STATE_BOOLEAN_KEYS and not isinstance(value, bool):
        raise LedgerError(f"collector state field {field_name}.{key} must be boolean")
    elif key in STATE_NUMBER_KEYS and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise LedgerError(f"collector state field {field_name}.{key} must be numeric")
    elif key in STATE_IDENTIFIER_KEYS:
        if (
            key == "candidateCutoff"
            and value == ""
            and _is_coverage_object_path(field_name)
        ):
            pass
        elif not isinstance(value, str) or sanitize_token(value) is None:
            raise LedgerError(
                f"collector state field {field_name}.{key} must be an identifier"
            )
    elif key in STATE_STRING_KEYS and not isinstance(value, str):
        raise LedgerError(f"collector state field {field_name}.{key} must be a string")
    elif not isinstance(value, (str, int, float, bool)):
        raise LedgerError(
            f"collector state field {field_name}.{key} contains unsupported data"
        )
    if isinstance(value, str) and len(value.encode("utf-8")) > 4096:
        raise LedgerError(f"collector state field {field_name}.{key} is too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise LedgerError(f"collector state field {field_name}.{key} is not finite")
    allowed = _state_allowed_values(field_name, key)
    if allowed is not None and value not in allowed:
        raise LedgerError(f"collector state field {field_name}.{key} has an invalid value")


def _state_allowed_values(field_name: str, key: str) -> Optional[set[str]]:
    if key == "state" and "quarantine" in field_name:
        return {"clear", "quarantined", "unknown"}
    if key == "field" and "timestamps" in field_name:
        return {
            "createdAt",
            "updatedAt",
            "attemptTime",
            "earliestPossibleAt",
            "latestPossibleAt",
        }
    if key == "scope":
        if (
            "checkpoint" in field_name
            or "historyCoverage" in field_name
            or "coverage" in field_name
            or "scopes[" in field_name
        ):
            return {"active", "archived"}
        return None
    if key == "projects":
        return {"validated_for_discovered_projects", "unknown", "complete", "partial"}
    if key == "branches":
        return {"version_metadata_observed", "active_branch_only", "unknown", "complete", "partial"}
    if key in {"history", "overall"} and "coverage" in field_name:
        return {"complete", "partial", "unknown"}
    if key == "reason":
        if "revisit" in field_name:
            return STATE_ENUM_VALUES["reason"]
        if "accountState" in field_name:
            return {"authentication", "cooldown"}
        return None
    if key == "coverage":
        if field_name == "summary" or field_name.endswith(".summary"):
            return {"validated_page", "partial", "unrecognized"}
        if _is_coverage_object_path(field_name):
            return {"complete", "partial", "unknown"}
        return STATE_ENUM_VALUES["coverage"]
    if key == "paginationState":
        return STATE_ENUM_VALUES["paginationState"]
    if key != "status":
        return STATE_ENUM_VALUES.get(key)
    if "olderHistoryAudit" in field_name:
        return {"disabled", "in_progress", "partial", "complete"}
    if "accountState" in field_name:
        return {"ready", "paused"}
    if "revisit" in field_name:
        return {"pending", "complete"}
    if "coverage" in field_name.lower():
        return {"not_started", "in_progress", "complete", "partial"}
    if "activeTrigger" in field_name or field_name.endswith(".active"):
        return {"active", "claimed", "running", "idle", "cancelled"}
    if field_name == "terminalSummary":
        return {"complete", "partial", "blocked"}
    return {"not_started", "in_progress", "complete", "partial"}


def _is_coverage_object_path(field_name: str) -> bool:
    return "coverage" in field_name.lower() or "scopes[" in field_name


def _nested_state_keys(key: str, field_name: str) -> Optional[set[str]]:
    if key in {"active", "archived"}:
        if "olderHistoryAudit" in field_name:
            return STATE_AUDIT_RESULT_KEYS
        if "coverage" in field_name.lower():
            return STATE_SCOPE_COVERAGE_KEYS
        return STATE_TRIGGER_KEYS
    if key == "olderHistoryAudit":
        if (
            "historyCoverage" in field_name
            and not field_name.endswith((".active", ".archived"))
        ) or field_name.endswith(".coverage"):
            return STATE_AUDIT_AGGREGATE_KEYS
        return STATE_AUDIT_RESULT_KEYS
    if key == "coverage" and "historyCoverage" in field_name:
        return STATE_HISTORY_COVERAGE_KEYS
    if key == "historyCoverage":
        return STATE_HISTORY_COVERAGE_KEYS
    if key == "scope" and field_name == "scheduleTransition":
        return STATE_SCOPE_KEYS
    return {
        "checkpoint": STATE_CHECKPOINT_KEYS,
        "pending": STATE_PENDING_KEYS,
        "range": STATE_RANGE_KEYS,
        "candidateQueue": STATE_CANDIDATE_KEYS,
        "summary": STATE_SUMMARY_KEYS | STATE_TERMINAL_SUMMARY_KEYS,
        "coverage": STATE_COVERAGE_KEYS,
        "identity": STATE_IDENTITY_KEYS,
        "accountState": STATE_ACCOUNT_KEYS,
        "terminalSummary": STATE_TERMINAL_SUMMARY_KEYS,
        "scopes": STATE_SCOPE_COVERAGE_KEYS,
        "candidate": STATE_CANDIDATE_KEYS | STATE_REVISIT_KEYS,
        "revisit": STATE_REVISIT_KEYS,
        "malformedPage": STATE_MALFORMED_PAGE_KEYS,
        "outstandingGeneration": STATE_OUTSTANDING_GENERATION_KEYS,
        "quarantine": STATE_QUARANTINE_KEYS,
        "timestamps": STATE_QUARANTINE_TIMESTAMP_KEYS,
    }.get(key)


def _safe_metadata_mapping(
    field_name: str,
    value: Mapping[str, Any],
    *,
    allowed_keys: Optional[set[str]],
) -> dict[str, Any]:
    _validate_state_field(field_name, value, allowed_keys)
    projected = _copy_metadata_value(value, field_name)
    if not isinstance(projected, dict):
        raise LedgerError(f"{field_name} is not an object")
    assert_no_secrets(projected)
    return projected


def _safe_terminal_summary(
    summary: Optional[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    if summary is None:
        return None
    if not isinstance(summary, Mapping):
        raise LedgerError("collector terminal summary is not an object")
    _validate_state_field("terminalSummary", summary, STATE_TERMINAL_SUMMARY_KEYS)
    projected = _copy_metadata_value(summary, "terminalSummary")
    if not isinstance(projected, dict):
        raise LedgerError("collector terminal summary is not an object")
    assert_no_secrets(projected)
    return projected


def _copy_metadata_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            out[str(key)] = _copy_metadata_value(child, f"{field_name}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _copy_metadata_value(child, f"{field_name}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float) and value == value and abs(value) < float("inf"):
        return value
    raise LedgerError(f"{field_name} contains unsupported metadata")


def _candidate_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise LedgerError("candidate payload must be an object")
    _validate_state_field("candidate", payload, STATE_CANDIDATE_KEYS | STATE_REVISIT_KEYS)
    return _copy_metadata_value(payload, "candidate")


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerError("state version must be an integer")
    if value < 0:
        raise LedgerError("state version must not be negative")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LedgerError(f"{field_name} must be a positive integer")
    return value


def _encode_cursor(kind: str, values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"kind": kind, **dict(values)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    if len(encoded.encode("ascii")) > MAX_CURSOR_BYTES:
        raise LedgerError("collector cursor exceeds the supported bound")
    return encoded


def _decode_cursor(cursor: Optional[str], expected_kind: str) -> Optional[dict[str, Any]]:
    if cursor is None:
        return None
    safe_cursor = _cursor_token(cursor)
    try:
        padded = safe_cursor + ("=" * (-len(safe_cursor) % 4))
        value = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise LedgerError("collector cursor is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("kind") != expected_kind
    ):
        raise LedgerError("collector cursor kind is invalid")
    return dict(value)


def _cursor_token(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value.encode("ascii", "ignore")) > MAX_CURSOR_BYTES:
        raise LedgerError("cursor is outside the supported bound")
    if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=" for character in value):
        raise LedgerError("cursor contains unsupported characters")
    return value
