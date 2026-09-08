"""Durable coordination state for the ChatGPT Chat history collector bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import psycopg

from .models import AttemptRecord, IngestContext
from .privacy import assert_no_secrets, sanitize_metadata, sanitize_token
from .pg_ledger import (
    LedgerBinding,
    LedgerError,
    LedgerScope,
    PgLedger,
    PgLedgerPage,
    _assert_active_binding,
    _lock_scope,
    fingerprint_value,
    scope_key,
)
from .timeutil import ensure_utc, parse_datetime


PROTOCOL_VERSION = 1
SCHEMA_VERSION = "chatgpt-collector-state-v1"
MAX_QUEUE_PAGE = 256
MAX_QUEUE_ITEM_BYTES = 16 * 1024
MAX_STATE_FIELD_BYTES = 64 * 1024
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
}
STATE_FORBIDDEN_KEYS = {"title", "body", "content", "text"}


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


class PgCollectorState:
    """PostgreSQL-owned collector state; scheduling remains in TypeScript."""

    def __init__(self, ledger: PgLedger) -> None:
        self.ledger = ledger

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
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        with self.ledger.connect() as conn, conn.cursor() as cur:
            header = self._load_header(cur, account, profile)
            cur.execute(
                """
                SELECT candidate_key, operation, payload, has_more
                FROM public.chatgpt_usage_collector_candidates
                WHERE profile_id = %s AND collector_account_id = %s
                ORDER BY rank
                LIMIT %s
                """,
                (profile, account, MAX_QUEUE_PAGE + 1),
            )
            rows = cur.fetchall()
        items = tuple(self._safe_candidate(row) for row in rows[:MAX_QUEUE_PAGE])
        return header, CandidatePage(items, has_more=len(rows) > MAX_QUEUE_PAGE)

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
        _validate_state_field("checkpoint", checkpoint, None)
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
                if expected_state_version is not None:
                    raise LedgerError("collector state version is stale")
                state_version = 1
            else:
                if current.state_version != expected_state_version:
                    raise LedgerError("collector state version is stale")
                state_version = current.state_version + 1
            next_schedule = (
                schedule_transition if schedule_transition is not None else current.schedule_transition
            )
            next_checkpoint = checkpoint if checkpoint is not None else current.checkpoint
            next_trigger = active_trigger if active_trigger is not None else current.active_trigger
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
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
                (f"{account}|{profile}",),
            )
            binding = self._binding_for_scope(cur, scope, account, expected_binding)
            cur.execute(
                """
                SELECT lease_fencing_token, lease_expires_at
                FROM public.chatgpt_usage_collector_leases
                WHERE profile_id = %s AND collector_account_id = %s
                FOR UPDATE
                """,
                (profile, account),
            )
            row = cur.fetchone()
            if row is not None and ensure_utc(row[1]) > now:
                raise LedgerError("collector lease is already active")
            next_token = 1 if row is None else int(row[0]) + 1
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

    def commit_history_page(
        self,
        *,
        lease: CollectorLease,
        scope: LedgerScope,
        run_id: str,
        page_commit_id: str,
        canonical_payload: Mapping[str, Any],
        ingest_operations: Sequence[Mapping[str, Any]],
        candidate_mutations: Sequence[Mapping[str, Any]] = (),
    ) -> PageAck:
        account = _account(lease.collector_account_id)
        profile = lease.profile_id
        safe_run = _token(run_id, "run_id")
        safe_commit = _token(page_commit_id, "page_commit_id")
        self.assert_safe_record(canonical_payload)
        serialized_payload = json.dumps(
            canonical_payload,
            separators=(",", ":"),
            default=str,
        )
        if len(serialized_payload.encode("utf-8")) > MAX_QUEUE_ITEM_BYTES:
            raise LedgerError("collector page commit payload exceeds the supported bound")
        if len(ingest_operations) > MAX_QUEUE_PAGE:
            raise LedgerError("collector page operation batch exceeds the supported bound")
        if len(candidate_mutations) > MAX_QUEUE_PAGE:
            raise LedgerError("collector candidate batch exceeds the supported bound")
        for operation in ingest_operations:
            self.assert_safe_record(operation)
            self._validate_ingest_operation(operation)
        for mutation in candidate_mutations:
            self.assert_safe_record(mutation)
        fingerprint = fingerprint_value(canonical_payload)
        with self.ledger.connect() as conn, conn.cursor() as cur:
            self._require_active_lease(
                cur,
                account=account,
                profile_id=profile,
                expected=lease,
            )
            binding = self._binding_for_scope(cur, scope, account, lease.binding)
            _lock_scope(cur, binding.scope_key)
            cur.execute(
                """
                SELECT state_version, payload_fingerprint
                FROM public.chatgpt_usage_collector_page_acks
                WHERE profile_id = %s AND collector_account_id = %s
                  AND run_id = %s AND page_commit_id = %s
                """,
                (profile, account, safe_run, safe_commit),
            )
            replay = cur.fetchone()
            if replay is not None:
                if str(replay[1]) != fingerprint:
                    raise LedgerError(
                        "collector page commit identity conflicts with its retained payload"
                    )
                return PageAck(safe_run, safe_commit, int(replay[0]), fingerprint)

            self._apply_ingest_operations(cur, scope, binding, ingest_operations)
            next_state = self._publish_checkpoint(
                cur,
                lease,
                checkpoint=self._checkpoint_from_canonical(canonical_payload),
            )
            for rank, mutation in enumerate(candidate_mutations, start=1):
                self._apply_candidate_mutation(cur, profile, account, mutation, rank)
            cur.execute(
                """
                DELETE FROM public.chatgpt_usage_collector_page_acks
                WHERE profile_id = %s AND collector_account_id = %s
                  AND serial = (
                      SELECT MAX(serial) - 1
                      FROM public.chatgpt_usage_collector_page_acks
                      WHERE profile_id = %s AND collector_account_id = %s
                  )
                """,
                (profile, account, profile, account),
            )
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_collector_page_acks (
                    profile_id, collector_account_id, run_id, page_commit_id,
                    serial, state_version, payload_fingerprint, acked_at
                ) VALUES (
                    %s, %s, %s, %s,
                    (SELECT COALESCE(MAX(serial), 0) + 1
                     FROM public.chatgpt_usage_collector_page_acks
                     WHERE profile_id = %s AND collector_account_id = %s),
                    %s, %s, NOW()
                )
                """,
                (
                    profile,
                    account,
                    safe_run,
                    safe_commit,
                    profile,
                    account,
                    next_state.state_version,
                    fingerprint,
                ),
            )
            return PageAck(safe_run, safe_commit, next_state.state_version, fingerprint)

    def load_conversation_metadata(
        self,
        *,
        collector_account_id: str,
        conversation_id: str,
        limit: int = 32,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        account = _account(collector_account_id)
        conversation = _token(conversation_id, "conversation_id")
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("conversation metadata limit is outside the supported range")
        with self.ledger.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT scope_key, binding_generation
                FROM public.chatgpt_usage_scope_bindings
                WHERE collector_account_id = %s AND binding_state = 'active'
                FOR SHARE
                """,
                (account,),
            )
            active_bindings = cur.fetchall()
            if not active_bindings:
                return {
                    "schemaVersion": SCHEMA_VERSION,
                    "items": (),
                    "totalAttempts": 0,
                    "returnedAttempts": 0,
                    "hasMore": False,
                    "truncated": False,
                }
            placeholders = ",".join("%s" for _ in active_bindings)
            params: list[Any] = [account]
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM public.chatgpt_usage_observations AS observation
                WHERE observation.collector_account_id = %s
                  AND observation.conversation_id = %s
                  AND observation.is_current_projection
                  AND observation.scope_key IN ({placeholders})
                """,
                (account, conversation, *(str(row[0]) for row in active_bindings)),
            )
            total = int(cur.fetchone()[0])
            pagination_params = list(params)
            pagination_params.append(conversation)
            pagination_params.extend(str(row[0]) for row in active_bindings)
            cur.execute(
                f"""
                SELECT observation.payload
                FROM public.chatgpt_usage_observations AS observation
                WHERE observation.collector_account_id = %s
                  AND observation.conversation_id = %s
                  AND observation.is_current_projection
                  AND observation.scope_key IN ({placeholders})
                  AND (%s::text IS NULL OR observation.observation_id > %s::text)
                ORDER BY observation.observed_at DESC
                LIMIT %s
                """,
                (*pagination_params, cursor, cursor, limit + 1),
            )
            rows = cur.fetchall()
            has_more = len(rows) > limit
            next_cursor = None
            if has_more:
                next_cursor = str(rows[limit - 1][0].get("observationId"))
                rows = rows[:limit]
            items = tuple(dict(row[0]) for row in rows)
            for item in items:
                item.setdefault("observationId", str(item.get("id")))
            return {
                "schemaVersion": SCHEMA_VERSION,
                "items": items,
                "totalAttempts": total,
                "returnedAttempts": len(items),
                "hasMore": has_more,
                "truncated": has_more,
                "cursor": next_cursor,
            }

    def load_report_snapshot(
        self,
        *,
        collector_account_id: str,
        limit: int = 256,
        cursor: Optional[str] = None,
    ) -> Mapping[str, Any]:
        account = _account(collector_account_id)
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("report snapshot limit is outside the supported range")
        counts = self.ledger.count_attempts(account)
        with self.ledger.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(array_agg(DISTINCT scope_key), ARRAY[]::text[])
                FROM public.chatgpt_usage_attempts
                WHERE collector_account_id = %s
                """,
                (account,),
            )
            scope_keys = tuple(str(value) for value in cur.fetchone()[0])
            cur.execute(
                """
                SELECT COUNT(*)
                FROM public.chatgpt_usage_attempts
                WHERE collector_account_id = %s AND NOT tombstone
                """,
                (account,),
            )
            total_attempts = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT attempt.scope_key, attempt.attempt_id,
                       attempt.conversation_id, attempt.identity_basis,
                       attempt.time_basis, attempt.attempt_time,
                       attempt.earliest_possible_at, attempt.latest_possible_at,
                       attempt.requested_model_raw, attempt.requested_mode_raw,
                       attempt.requested_reasoning_effort_raw,
                       attempt.recorded_final_model_raw,
                       attempt.resolved_model_raw, attempt.requested_family,
                       attempt.recorded_final_family, attempt.resolved_family,
                       attempt.mapping_version, attempt.outcome,
                       attempt.completed_answer, attempt.generation_started,
                       attempt.surface, attempt.origin, attempt.warnings,
                       attempt.quarantine_state, attempt.observed_at,
                       attempt.last_seen_at
                FROM public.chatgpt_usage_attempts AS attempt
                WHERE attempt.collector_account_id = %s
                  AND NOT attempt.tombstone
                ORDER BY attempt.scope_key, attempt.attempt_id
                OFFSET COALESCE(%s::bigint, 0)
                LIMIT %s
                """,
                (account, cursor, limit + 1),
            )
            rows = cur.fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            next_cursor = str(limit + int(cursor or 0)) if has_more else None
        attempts = tuple(self._attempt_snapshot(row) for row in rows)
        return {
            "snapshotVersion": 1,
            "schemaVersion": SCHEMA_VERSION,
            "collectorAccountId": account,
            "asOf": datetime.now(timezone.utc).isoformat(),
            "counts": _usage_counts_payload(counts),
            "coverage": {
                "timeWindow": None,
                "truncated": has_more,
                "capped": has_more,
                "totalAttempts": total_attempts,
                "returnedAttempts": len(attempts),
            },
            "scope": {
                "collectorAccountId": account,
                "scopeKeys": list(scope_keys),
            },
            "quarantine": {
                "quarantined": counts.excluded_non_generation,
                "unknown": counts.unknown_model,
                "clear": max(0, total_attempts - counts.excluded_non_generation - counts.unknown_model),
            },
            "attempts": attempts,
            "cursor": next_cursor,
        }

    def read_candidates(
        self,
        *,
        collector_account_id: str,
        profile_id: str,
        limit: int = 64,
    ) -> CandidatePage:
        account = _account(collector_account_id)
        profile = _token(profile_id, "profile_id")
        if limit <= 0 or limit > MAX_QUEUE_PAGE:
            raise LedgerError("candidate read limit is outside the supported range")
        with self.ledger.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT candidate_key, operation, payload, has_more
                FROM public.chatgpt_usage_collector_candidates
                WHERE profile_id = %s AND collector_account_id = %s
                ORDER BY rank
                LIMIT %s
                """,
                (profile, account, limit + 1),
            )
            rows = cur.fetchall()
        return CandidatePage(
            tuple(self._safe_candidate(row) for row in rows[:limit]),
            has_more=len(rows) > limit,
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

    def finish_run(self, *, lease: CollectorLease, run_id: str) -> None:
        safe_run = _token(run_id, "run_id")
        with self.ledger.connect() as conn, conn.cursor() as cur:
            self._require_active_lease(
                cur,
                account=lease.collector_account_id,
                profile_id=lease.profile_id,
                expected=lease,
            )
            self._transition_active_run(
                cur,
                lease,
                safe_run,
                state="cancelled",
            )
            cur.execute(
                """
                UPDATE public.chatgpt_usage_collector_state
                SET active_trigger =
                        jsonb_build_object('state', 'idle', 'runId', %s),
                    updated_at = NOW()
                WHERE profile_id = %s AND collector_account_id = %s
                """,
                (lease.profile_id, lease.collector_account_id),
            )
        self.release_lease(lease=lease)

    def cancel(self, *, lease: CollectorLease, run_id: str) -> None:
        safe_run = _token(run_id, "run_id")
        with self.ledger.connect() as conn, conn.cursor() as cur:
            self._require_active_lease(
                cur,
                account=lease.collector_account_id,
                profile_id=lease.profile_id,
                expected=lease,
            )
            cur.execute(
                """
                UPDATE public.chatgpt_usage_collector_state
                SET active_trigger =
                        jsonb_build_object('state', 'cancelled', 'runId', %s),
                    updated_at = NOW()
                WHERE profile_id = %s AND collector_account_id = %s
                """,
                (safe_run, lease.profile_id, lease.collector_account_id),
            )
        self.release_lease(lease=lease)

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

    def _require_active_lease(
        self,
        cur: psycopg.Cursor,
        *,
        account: str,
        profile_id: str,
        expected: CollectorLease,
    ) -> None:
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
    def _checkpoint_from_canonical(payload: Mapping[str, Any]) -> dict[str, Any]:
        checkpoint = payload.get("checkpoint")
        return dict(checkpoint) if isinstance(checkpoint, Mapping) else {}

    def _publish_checkpoint(
        self,
        cur: psycopg.Cursor,
        lease: CollectorLease,
        *,
        checkpoint: Mapping[str, Any],
    ) -> CollectorStateHeader:
        current = self._load_header(
            cur,
            lease.collector_account_id,
            lease.profile_id,
            lock=True,
        )
        next_version = current.state_version + 1 if current else 1
        schedule_transition = current.schedule_transition if current else None
        active_trigger = current.active_trigger if current else None
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
                _json(checkpoint),
                _json(active_trigger),
                now,
            ),
        )
        return CollectorStateHeader(
            lease.profile_id,
            lease.collector_account_id,
            next_version,
            schedule_transition,
            checkpoint,
            active_trigger,
            now,
        )

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
            safe_operation = _safe_operation(operation)
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
        return {"accepted": accepted}

    @staticmethod
    def _validate_ingest_operation(operation: Mapping[str, Any]) -> None:
        kind = _token(operation.get("kind"), "operation.kind")
        if kind not in {"observation", "attempt", "coverageGap"}:
            raise LedgerError("collector page contains an unsupported ingest operation")

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
        safe_payload = sanitize_metadata(payload if isinstance(payload, Mapping) else {})
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
            ) VALUES (%s, %s, %s, %s, 'candidate', %s::jsonb, FALSE)
            ON CONFLICT (profile_id, collector_account_id, candidate_key)
            DO UPDATE SET
                rank = EXCLUDED.rank,
                payload = EXCLUDED.payload,
                has_more = EXCLUDED.has_more
            """,
            (profile_id, account, candidate_key, rank, serialized),
        )

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
        return {
            "scopeKey": str(row[0]),
            "attemptId": str(row[1]),
            "conversationId": str(row[2]),
            "identityBasis": str(row[3]),
            "timeBasis": str(row[4]),
            "attemptTime": row[5].isoformat() if row[5] else None,
            "earliestPossibleAt": row[6].isoformat() if row[6] else None,
            "latestPossibleAt": row[7].isoformat() if row[7] else None,
            "requestedModelRaw": row[8],
            "requestedModeRaw": row[9],
            "requestedReasoningEffortRaw": row[10],
            "recordedFinalModelRaw": row[11],
            "resolvedModelRaw": row[12],
            "requestedFamily": row[13],
            "recordedFinalFamily": row[14],
            "resolvedFamily": row[15],
            "mappingVersion": str(row[16]),
            "outcome": str(row[17]),
            "completedAnswer": bool(row[18]),
            "generationStarted": bool(row[19]),
            "surface": str(row[20]),
            "origin": row[21],
            "warnings": list(row[22] or []),
            "quarantineState": str(row[23]),
            "observedAt": row[24].isoformat() if row[24] else None,
            "lastSeenAt": row[25].isoformat() if row[25] else None,
        }


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


def _safe_operation(operation: Mapping[str, Any]) -> dict[str, Any]:
    safe = {
        key: value
        for key, value in operation.items()
        if key not in {"payload", "aliases", "quarantine", "details"}
    }
    if isinstance(operation.get("payload"), Mapping):
        safe["payload"] = sanitize_metadata(operation["payload"])
    if isinstance(operation.get("aliases"), Sequence) and not isinstance(
        operation.get("aliases"), str
    ):
        safe["aliases"] = [
            {
                "kind": _token(alias.get("kind"), "alias.kind"),
                "value": _token(alias.get("value"), "alias.value"),
            }
            for alias in operation.get("aliases", ())
        ]
    if isinstance(operation.get("quarantine"), Mapping):
        safe["quarantine"] = sanitize_metadata(operation["quarantine"])
    if isinstance(operation.get("details"), Mapping):
        safe["details"] = sanitize_metadata(operation["details"])
    return safe


def _usage_counts_payload(counts: Any) -> dict[str, Any]:
    return {
        "total": counts.total,
        "completed": counts.completed,
        "ambiguous": counts.ambiguous,
        "unknown": {
            "time": counts.unknown_time,
            "identity": counts.unknown_identity,
            "surface": counts.unknown_surface,
            "origin": counts.unknown_origin,
            "model": counts.unknown_model,
        },
        "excluded": {
            "surface": counts.excluded_surface,
            "origin": counts.excluded_origin,
            "nonGeneration": counts.excluded_non_generation,
        },
        "uncertainOutcome": counts.uncertain_outcome,
        "observedModelMismatches": counts.observed_model_mismatches,
        "byRequestedFamily": counts.by_requested_family,
        "byRecordedFinalFamily": counts.by_recorded_final_family,
        "byResolvedFamily": counts.by_resolved_family,
        "byRequestedModelRaw": counts.by_requested_model_raw,
        "byRecordedFinalModelRaw": counts.by_recorded_final_model_raw,
        "byResolvedModelRaw": counts.by_resolved_model_raw,
    }


def _validate_state_field(
    field_name: str,
    value: Any,
    allowed_keys: Optional[set[str]],
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise LedgerError(f"collector state field {field_name} is not an object")
    serialized = _json(value)
    if len(serialized.encode("utf-8")) > MAX_STATE_FIELD_BYTES:
        raise LedgerError(f"collector state field {field_name} exceeds the supported bound")
    for raw_key, child in value.items():
        key = str(raw_key)
        if key in STATE_FORBIDDEN_KEYS:
            raise LedgerError(f"collector state field {field_name} contains unsupported content")
        if allowed_keys is not None and key not in allowed_keys:
            raise LedgerError(f"collector state field {field_name} contains an unknown key")
        if isinstance(child, Mapping):
            _validate_state_field(f"{field_name}.{key}", child, None)
        elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    _validate_state_field(f"{field_name}.{key}[{index}]", item, None)
