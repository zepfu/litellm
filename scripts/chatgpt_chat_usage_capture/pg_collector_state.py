"""Durable coordination state for the ChatGPT Chat history collector bridge."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
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
STATE_PENDING_KEYS = {"kind", "missedCount", "requestedAt"}
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
    fingerprint: str


@dataclass
class _ReadSnapshot:
    snapshot_id: str
    account: str
    profile: Optional[str]
    kind: str
    connection: psycopg.Connection
    created_at: datetime


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
    ) -> tuple[Optional[CollectorStateHeader], CandidatePage]:
        """Load only the durable header; queue reads are explicit and versioned."""
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        with self.ledger.connect() as conn, conn.cursor() as cur:
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
    ) -> CollectorStateHeader:
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        payload = {
            "scheduleTransition": schedule_transition,
            "checkpoint": checkpoint,
            "activeTrigger": active_trigger,
        }
        self.assert_safe_record(payload)
        _validate_state_field("scheduleTransition", schedule_transition, STATE_SCHEDULE_KEYS)
        _validate_state_field("checkpoint", checkpoint, STATE_CHECKPOINT_KEYS)
        _validate_state_field("activeTrigger", active_trigger, STATE_TRIGGER_KEYS)
        now = datetime.now(timezone.utc)
        with self.ledger.connect() as conn, conn.cursor() as cur:
            self._require_active_lease(
                cur,
                account=account,
                profile_id=profile,
                expected=lease,
            )
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
    ) -> CollectorLease:
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        if ttl_seconds <= 0 or ttl_seconds > 3600:
            raise LedgerError("collector lease ttl is outside the supported range")
        now = datetime.now(timezone.utc)
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, timezone.utc)
        with self.ledger.connect() as conn, conn.cursor() as cur:
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

    def heartbeat_lease(self, *, lease: CollectorLease, ttl_seconds: int) -> CollectorLease:
        if ttl_seconds <= 0 or ttl_seconds > 3600:
            raise LedgerError("collector lease ttl is outside the supported range")
        now = datetime.now(timezone.utc)
        expires = datetime.fromtimestamp(now.timestamp() + ttl_seconds, timezone.utc)
        with self.ledger.connect() as conn, conn.cursor() as cur:
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

    def release_lease(self, *, lease: CollectorLease) -> None:
        with self.ledger.connect() as conn, conn.cursor() as cur:
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
        derived_operations = self._derive_ingest_operations(wire_payload)
        actual_operations = tuple(ingest_operations) + tuple(derived_operations)
        derived_candidates = self._derive_candidate_mutations(wire_payload)
        actual_candidates = (
            tuple(candidate_mutations) + tuple(derived_candidates)
            if candidate_mutations
            else tuple(derived_candidates)
        )
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
    ) -> PageAck:
        prepared = self._prepare_page_commit(
            lease=lease,
            run_id=run_id,
            page_commit_id=page_commit_id,
            canonical_payload=canonical_payload,
            ingest_operations=ingest_operations,
            candidate_mutations=candidate_mutations,
            coverage_mutations=coverage_mutations,
            expected_state_version=expected_state_version,
        )
        acknowledgment: Optional[PageAck] = None
        with self.ledger.connect() as conn, conn.cursor() as cur:
            self._require_active_lease(
                cur,
                account=prepared.account,
                profile_id=prepared.profile,
                expected=lease,
            )
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
                )
                self._apply_coverage_mutations(
                    cur,
                    scope,
                    binding,
                    prepared.coverage_mutations,
                )
                next_state = self._publish_checkpoint(
                    cur,
                    lease,
                    run_id=prepared.run_id,
                    checkpoint=self._checkpoint_from_canonical(prepared.wire_payload),
                    expected_state_version=prepared.expected_state_version,
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
    ) -> Mapping[str, Any]:
        account = _account(collector_account_id)
        conversation = _token(conversation_id, "conversation_id")
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("conversation metadata limit is outside the supported range")
        snapshot, new_snapshot = self._get_snapshot(
            account=account,
            profile=None,
            kind="metadata",
            snapshot_id=snapshot_id,
        )
        try:
            conn = snapshot.connection
            with conn.cursor() as cur:
                cursor_values = _decode_cursor(cursor, "observation")
                cursor_time = (
                    parse_datetime(cursor_values["observedAt"])
                    if cursor_values is not None
                    else None
                )
                cursor_id = cursor_values["id"] if cursor_values is not None else None
                cur.execute(
                    """
                    WITH RECURSIVE scope_chain(scope_key) AS (
                        SELECT binding.scope_key
                        FROM public.chatgpt_usage_scope_bindings AS binding
                        WHERE binding.collector_account_id = %s
                          AND binding.binding_state = 'active'
                        UNION
                        SELECT redirect.retired_scope_key
                        FROM public.chatgpt_usage_scope_redirects AS redirect
                        JOIN scope_chain
                          ON scope_chain.scope_key = redirect.canonical_scope_key
                    )
                    SELECT COUNT(*)
                    FROM public.chatgpt_usage_observations AS observation
                    WHERE observation.collector_account_id = %s
                      AND observation.conversation_id = %s
                      AND observation.is_current_projection
                      AND observation.scope_key IN (SELECT scope_key FROM scope_chain)
                    """,
                    (account, account, conversation),
                )
                total = int(cur.fetchone()[0])
                cur.execute(
                    """
                    WITH RECURSIVE scope_chain(scope_key) AS (
                        SELECT binding.scope_key
                        FROM public.chatgpt_usage_scope_bindings AS binding
                        WHERE binding.collector_account_id = %s
                          AND binding.binding_state = 'active'
                        UNION
                        SELECT redirect.retired_scope_key
                        FROM public.chatgpt_usage_scope_redirects AS redirect
                        JOIN scope_chain
                          ON scope_chain.scope_key = redirect.canonical_scope_key
                    )
                    SELECT observation.observation_id, observation.scope_key,
                           observation.source_kind, observation.source_id,
                           observation.run_id, observation.observed_at,
                           observation.schema_version, observation.provenance,
                           observation.payload
                    FROM public.chatgpt_usage_observations AS observation
                    WHERE observation.collector_account_id = %s
                      AND observation.conversation_id = %s
                      AND observation.is_current_projection
                      AND observation.scope_key IN (SELECT scope_key FROM scope_chain)
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
                        account,
                        conversation,
                        cursor_time,
                        cursor_time,
                        cursor_time,
                        cursor_id,
                        limit + 1,
                    ),
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
            if new_snapshot:
                self.close_snapshot(snapshot.snapshot_id)
            raise

    def load_report_snapshot(
        self,
        *,
        collector_account_id: str,
        limit: int = 256,
        cursor: Optional[str] = None,
        snapshot_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        account = _account(collector_account_id)
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("report snapshot limit is outside the supported range")
        snapshot, new_snapshot = self._get_snapshot(
            account=account,
            profile=None,
            kind="report",
            snapshot_id=snapshot_id,
        )
        try:
            cursor_values = _decode_cursor(cursor, "attempt")
            cursor_scope = cursor_values["scopeKey"] if cursor_values else None
            cursor_attempt = cursor_values["attemptId"] if cursor_values else None
            conn = snapshot.connection
            with conn.cursor() as cur:
                scope_cte = """
                    WITH RECURSIVE scope_chain(
                        stored_scope_key, canonical_scope_key
                    ) AS (
                        SELECT binding.scope_key, binding.scope_key
                        FROM public.chatgpt_usage_scope_bindings AS binding
                        WHERE binding.collector_account_id = %s
                          AND binding.binding_state = 'active'
                        UNION
                        SELECT redirect.retired_scope_key,
                               scope_chain.canonical_scope_key
                        FROM public.chatgpt_usage_scope_redirects AS redirect
                        JOIN scope_chain
                          ON scope_chain.stored_scope_key =
                             redirect.canonical_scope_key
                    )
                """
                cur.execute(
                    scope_cte
                    + """
                    SELECT COUNT(*)
                    FROM public.chatgpt_usage_attempts AS attempt
                    WHERE attempt.collector_account_id = %s
                      AND NOT attempt.tombstone
                      AND attempt.scope_key IN (
                          SELECT stored_scope_key FROM scope_chain
                      )
                    """,
                    (account, account),
                )
                total_attempts = int(cur.fetchone()[0])
                cur.execute(
                    scope_cte
                    + """
                    SELECT COUNT(*)
                    FROM public.chatgpt_usage_coverage_gaps AS gap
                    WHERE gap.collector_account_id = %s
                      AND gap.state <> 'resolved'
                      AND gap.scope_key IN (
                          SELECT stored_scope_key FROM scope_chain
                      )
                    """,
                    (account, account),
                )
                open_gaps = int(cur.fetchone()[0])
                cur.execute(
                    scope_cte
                    + """
                    SELECT DISTINCT stored_scope_key, canonical_scope_key
                    FROM scope_chain
                    ORDER BY stored_scope_key
                    """,
                    (account,),
                )
                scope_rows = cur.fetchall()
                cur.execute(
                    scope_cte
                    + """
                    SELECT attempt.scope_key, scope_chain.canonical_scope_key,
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
                           ), '{}'::jsonb) AS revision_payload
                    FROM public.chatgpt_usage_attempts AS attempt
                    JOIN scope_chain
                      ON scope_chain.stored_scope_key = attempt.scope_key
                    WHERE attempt.collector_account_id = %s
                      AND NOT attempt.tombstone
                      AND (
                          %s::text IS NULL
                          OR attempt.scope_key > %s::text
                          OR (
                              attempt.scope_key = %s::text
                              AND attempt.attempt_id > %s::text
                          )
                      )
                    ORDER BY attempt.scope_key, attempt.attempt_id
                    LIMIT %s
                    """,
                    (
                        account,
                        account,
                        cursor_scope,
                        cursor_scope,
                        cursor_scope,
                        cursor_attempt,
                        limit + 1,
                    ),
                )
                rows = cur.fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            attempts = tuple(self._attempt_snapshot(row) for row in rows)
            next_cursor = (
                _encode_cursor(
                    "attempt",
                    {
                        "scopeKey": str(rows[-1][0]),
                        "attemptId": str(rows[-1][2]),
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
            result = {
                "snapshotVersion": 1,
                "schemaVersion": "chatgpt-chat-history-v1",
                "collectorAccountId": account,
                "snapshotId": snapshot.snapshot_id if has_more else None,
                "asOf": snapshot.created_at.isoformat(),
                "attempts": attempts,
                "coverage": {
                    "truncated": has_more,
                    "partial": has_more or open_gaps > 0,
                    "totalAttempts": total_attempts,
                    "returnedAttempts": len(attempts),
                    "openCoverageGaps": open_gaps,
                },
                "historyCoverage": {
                    "openCoverageGaps": open_gaps,
                    "complete": open_gaps == 0 and not has_more,
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
            if new_snapshot:
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
    ) -> CandidatePage:
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("candidate read limit is outside the supported range")
        with self.ledger.connect() as conn, conn.cursor() as cur:
            header = self._load_header(cur, account, profile)
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
    ) -> None:
        if len(mutations) > MAX_QUEUE_PAGE:
            raise LedgerError("candidate mutation batch exceeds the supported bound")
        with self.ledger.connect() as conn, conn.cursor() as cur:
            self._require_active_lease(
                cur,
                account=lease.collector_account_id,
                profile_id=lease.profile_id,
                expected=lease,
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

    def finish_run(
        self,
        *,
        lease: CollectorLease,
        run_id: str,
        expected_state_version: Optional[int] = None,
        trigger_id: Optional[str] = None,
        outcome: Optional[str] = None,
        summary: Optional[Mapping[str, Any]] = None,
    ) -> CollectorStateHeader:
        if expected_state_version is None:
            raise LedgerError("collector state version is required")
        if trigger_id is None:
            raise LedgerError("collector trigger is required")
        safe_run = _token(run_id, "run_id")
        safe_trigger = _token(trigger_id, "trigger_id")
        safe_outcome = _token(outcome, "outcome") if outcome is not None else None
        _validate_state_field("finishSummary", summary, None)
        with self.ledger.connect() as conn, conn.cursor() as cur:
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
            if summary is not None:
                trigger["summary"] = dict(summary)
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
    ) -> CollectorStateHeader:
        if expected_state_version is None:
            raise LedgerError("collector state version is required")
        if trigger_id is None:
            raise LedgerError("collector trigger is required")
        safe_run = _token(run_id, "run_id")
        safe_trigger = _token(trigger_id, "trigger_id")
        safe_outcome = _token(outcome, "outcome") if outcome is not None else None
        _validate_state_field("cancelSummary", summary, None)
        with self.ledger.connect() as conn, conn.cursor() as cur:
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
            if summary is not None:
                trigger["summary"] = dict(summary)
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
        if "checkpoint" not in payload:
            return _UNSET
        checkpoint = payload.get("checkpoint")
        if checkpoint is None:
            return _UNSET
        if not isinstance(checkpoint, Mapping):
            raise LedgerError("collector checkpoint is not an object")
        return dict(checkpoint)

    def _publish_checkpoint(
        self,
        cur: psycopg.Cursor,
        lease: CollectorLease,
        *,
        run_id: str,
        checkpoint: Any,
        expected_state_version: Optional[int],
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
    ) -> tuple[Mapping[str, Any], ...]:
        explicit = payload.get("ingestOperations", payload.get("operations"))
        operations: list[Mapping[str, Any]] = []
        if explicit is not None:
            if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
                raise LedgerError("collector ingest operations are not an array")
            operations.extend(
                item for item in explicit if isinstance(item, Mapping)
            )
        source = payload.get("source", payload.get("context"))
        source_map = source if isinstance(source, Mapping) else {}
        if "runId" not in source_map or "observedAt" not in source_map:
            source_map = {
                "runId": payload.get("runId"),
                "observedAt": payload.get("observedAt"),
                "sourceKind": payload.get("sourceKind"),
                "sourceId": payload.get("sourceId"),
                "schemaVersion": payload.get("schemaVersion"),
                "provenance": payload.get("provenance"),
            }
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
        observations = payload.get("observations", ())
        if observations is not None:
            if not isinstance(observations, Sequence) or isinstance(
                observations, (str, bytes)
            ):
                raise LedgerError("collector observations are not an array")
            for observation in observations:
                if not isinstance(observation, Mapping):
                    raise LedgerError("collector observation is not an object")
                observation_payload = observation.get("payload", observation)
                operations.append(
                    {
                        "kind": "observation",
                        **dict(source_map),
                        "payload": observation_payload,
                    }
                )
        return tuple(operations)

    @staticmethod
    def _derive_candidate_mutations(
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        explicit = payload.get("candidateMutations", ())
        if explicit is not None:
            if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
                raise LedgerError("collector candidate mutations are not an array")
            return tuple(
                item for item in explicit if isinstance(item, Mapping)
            )
        return ()

    @staticmethod
    def _derive_coverage_mutations(
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        explicit = payload.get("coverageMutations", ())
        if explicit is not None:
            if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
                raise LedgerError("collector coverage mutations are not an array")
            return tuple(
                item for item in explicit if isinstance(item, Mapping)
            )
        return ()

    def _apply_ingest_operations(
        self,
        cur: psycopg.Cursor,
        scope: LedgerScope,
        binding: LedgerBinding,
        operations: Sequence[Mapping[str, Any]],
    ) -> dict[str, int]:
        """Apply metadata-only operations through the current ledger page."""
        page = PgLedgerPage(self.ledger, cur.connection, expected_binding=binding)
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
            safe["details"] = _safe_metadata_mapping(
                "operation.details",
                operation["details"],
                allowed_keys=None,
            )
        if isinstance(operation.get("provenance"), Mapping):
            safe["provenance"] = _safe_metadata_mapping(
                "operation.provenance",
                operation["provenance"],
                allowed_keys=None,
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
        self._apply_ingest_operations(cur, scope, binding, operations)

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
        safe_payload = {
            "candidateKey": candidate_key,
            "operation": operation,
            "payload": _candidate_envelope(payload) if isinstance(payload, Mapping) else {},
            "hasMore": bool(mutation.get("hasMore", False)),
        }
        assert_no_secrets(safe_payload)
        serialized = _json(safe_payload)
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
    ) -> tuple[_ReadSnapshot, bool]:
        if snapshot_id is not None:
            safe_snapshot = _token(snapshot_id, "snapshot_id")
            snapshot = self._snapshots.get(safe_snapshot)
            if snapshot is None:
                raise LedgerError("collector read snapshot is unavailable")
            if (
                snapshot.account != account
                or snapshot.profile != profile
                or snapshot.kind != kind
            ):
                raise LedgerError("collector read snapshot identity is stale")
            age = (datetime.now(timezone.utc) - snapshot.created_at).total_seconds()
            if age > MAX_SNAPSHOT_AGE_SECONDS:
                self.close_snapshot(safe_snapshot)
                raise LedgerError("collector read snapshot expired")
            return snapshot, False
        conn = self.ledger.connect()
        # PgLedger.connect() configures the session in its first transaction;
        # commit that setup before opening the bounded repeatable-read view.
        conn.commit()
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
        safe_snapshot = uuid4().hex
        snapshot = _ReadSnapshot(
            safe_snapshot,
            account,
            profile,
            kind,
            conn,
            datetime.now(timezone.utc),
        )
        self._snapshots[safe_snapshot] = snapshot
        return snapshot, True

    def _close_owned_snapshots(
        self,
        *,
        account: str,
        profile: Optional[str],
    ) -> None:
        for snapshot_id, snapshot in tuple(self._snapshots.items()):
            if snapshot.account == account and (
                profile is None or snapshot.profile in {None, profile}
            ):
                self.close_snapshot(snapshot_id)

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
        value = {
            "candidateKey": str(row[0]),
            "operation": str(row[1]),
            "payload": dict(row[2]),
            "hasMore": bool(row[3]),
        }
        assert_no_secrets(value)
        return value

    @staticmethod
    def _attempt_snapshot(row: Sequence[Any]) -> dict[str, Any]:
        revision_payload = row[29] if len(row) > 29 and isinstance(row[29], Mapping) else {}
        aliases = row[28] if len(row) > 28 and isinstance(row[28], Sequence) else ()
        item = {
            "scopeKey": str(row[0]),
            "canonicalScopeKey": str(row[1]),
            "attemptId": str(row[2]),
            "conversationId": str(row[3]),
            "identityBasis": str(row[4]),
            "timeBasis": str(row[5]),
            "attemptTime": row[6].isoformat() if row[6] else None,
            "earliestPossibleAt": row[7].isoformat() if row[7] else None,
            "latestPossibleAt": row[8].isoformat() if row[8] else None,
            "requestedModelRaw": row[9],
            "requestedModeRaw": row[10],
            "requestedReasoningEffortRaw": row[11],
            "recordedFinalModelRaw": row[12],
            "resolvedModelRaw": row[13],
            "requestedFamily": row[14],
            "recordedFinalFamily": row[15],
            "resolvedFamily": row[16],
            "mappingVersion": str(row[17]),
            "outcome": str(row[18]),
            "completedAnswer": bool(row[19]),
            "generationStarted": bool(row[20]),
            "surface": str(row[21]),
            "origin": row[22],
            "revision": int(row[23]),
            "warnings": list(row[24] or []),
            "quarantine": {
                "state": str(row[25]),
                "warnings": list(row[24] or []),
            },
            "quarantineState": str(row[25]),
            "observedAt": row[26].isoformat() if row[26] else None,
            "lastSeenAt": row[27].isoformat() if row[27] else None,
            "aliases": [
                [str(alias[0]), str(alias[1])]
                for alias in aliases
                if isinstance(alias, Sequence) and len(alias) == 2
            ],
            "evidenceMessageIds": list(revision_payload.get("evidenceMessageIds", ()))
        }
        assert_no_secrets(item)
        return item


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
        if isinstance(child, Mapping):
            nested_keys = _nested_state_keys(key)
            _validate_state_field(f"{field_name}.{key}", child, nested_keys)
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            if len(child) > MAX_QUEUE_PAGE:
                raise LedgerError(f"collector state field {field_name}.{key} is too large")
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    _validate_state_field(
                        f"{field_name}.{key}[{index}]",
                        item,
                        _nested_state_keys(key),
                    )
                elif isinstance(item, (str, int, float, bool)) or item is None:
                    continue
                else:
                    raise LedgerError(
                        f"collector state field {field_name}.{key} contains unsupported data"
                    )
        elif not isinstance(child, (str, int, float, bool)) and child is not None:
            raise LedgerError(
                f"collector state field {field_name}.{key} contains unsupported data"
            )
        if isinstance(child, float) and not child.is_integer() and not abs(child) < float("inf"):
            raise LedgerError(f"collector state field {field_name}.{key} is not finite")


def _nested_state_keys(key: str) -> Optional[set[str]]:
    return {
        "checkpoint": STATE_CHECKPOINT_KEYS,
        "scope": STATE_SCOPE_KEYS,
        "pending": STATE_PENDING_KEYS,
        "active": STATE_TRIGGER_KEYS,
        "range": STATE_RANGE_KEYS,
        "candidateQueue": STATE_CANDIDATE_KEYS,
        "summary": STATE_SUMMARY_KEYS,
        "revisit": STATE_REVISIT_KEYS,
        "malformedPage": STATE_MALFORMED_PAGE_KEYS,
        "outstandingGeneration": STATE_OUTSTANDING_GENERATION_KEYS,
        "olderHistoryAudit": STATE_AUDIT_KEYS,
        "accountState": STATE_ACCOUNT_KEYS,
        "paginationState": STATE_PAGINATION_KEYS,
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
    safe_cursor = _token(cursor, "cursor")
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
