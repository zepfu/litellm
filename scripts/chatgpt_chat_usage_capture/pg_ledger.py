"""PostgreSQL persistence contract for ChatGPT ordinary-Chat usage metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import psycopg

from .models import AttemptRecord
from .privacy import assert_no_secrets, sanitize_mapping, sanitize_token
from .timeutil import ensure_utc, isoformat_utc


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "apply_chatgpt_usage_ledger_2026_09_08.sql"
)
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
    }
)


@dataclass(frozen=True)
class LedgerScope:
    collector_account_id: str
    provider: str
    provider_user_id: Optional[str]
    workspace_id: Optional[str]
    quota_owner_id: Optional[str]
    surface: str


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
class UsageCounts:
    total: int
    completed: int
    by_requested_family: dict[str, int] = field(default_factory=dict)
    by_recorded_final_family: dict[str, int] = field(default_factory=dict)
    by_resolved_family: dict[str, int] = field(default_factory=dict)
    by_requested_model_raw: dict[str, int] = field(default_factory=dict)
    by_recorded_final_model_raw: dict[str, int] = field(default_factory=dict)
    by_resolved_model_raw: dict[str, int] = field(default_factory=dict)


class LedgerError(RuntimeError):
    pass


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

    def connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn)
        conn.execute(
            "SELECT set_config('application_name', %s, false)",
            (self.application_name,),
        )
        conn.execute(
            "SELECT set_config('lock_timeout', %s, true)",
            (f"{self.lock_timeout_ms}ms",),
        )
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (f"{self.statement_timeout_ms}ms",),
        )
        return conn

    def ensure_schema(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        try:
            with self.connect() as conn, conn.cursor() as cur:
                for statement in _split_sql_statements(sql):
                    cur.execute(statement)
        except (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled) as exc:
            raise LedgerError(exc.__class__.__name__) from exc

    def upsert_scope(self, scope: LedgerScope, *, seen_at: datetime) -> str:
        key = scope_key(scope)
        self.assert_safe_record(_scope_payload(scope))
        with self.connect() as conn, conn.cursor() as cur:
            _lock_scope(cur, key)
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_scopes (
                    scope_key, collector_account_id, provider, provider_user_id,
                    workspace_id, quota_owner_id, surface, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scope_key) DO UPDATE SET
                    provider = EXCLUDED.provider,
                    provider_user_id = EXCLUDED.provider_user_id,
                    workspace_id = EXCLUDED.workspace_id,
                    quota_owner_id = EXCLUDED.quota_owner_id,
                    surface = EXCLUDED.surface,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    key,
                    scope.collector_account_id,
                    scope.provider,
                    scope.provider_user_id,
                    scope.workspace_id,
                    scope.quota_owner_id,
                    scope.surface,
                    ensure_utc(seen_at),
                    ensure_utc(seen_at),
                ),
            )
        return key

    def record_observation(
        self,
        scope: LedgerScope,
        context: IngestContext,
        payload: Mapping[str, Any],
    ) -> tuple[str, bool]:
        sanitized = sanitize_mapping(payload)
        self.assert_safe_record(sanitized)
        provenance = sanitize_provenance(
            {
                **(context.provenance or {}),
                "collector_account_id": scope.collector_account_id,
                "schema_version": context.schema_version,
                "source_id": context.source_id,
                "source_kind": context.source_kind,
            }
        )
        self.assert_safe_record(provenance)
        key = scope_key(scope)
        stable_payload = _without_keys(sanitized, "provenance", "run_id")
        revision_fingerprint = fingerprint_value(
            {"payload": stable_payload, "provenance": provenance}
        )
        observation_id = stable_id(
            key,
            context.source_kind,
            context.source_id,
            revision_fingerprint,
        )
        with self.connect() as conn, conn.cursor() as cur:
            _lock_scope(cur, key)
            prior = _latest_observation(cur, key, context)
            if prior is not None and prior["revision_fingerprint"] == revision_fingerprint:
                return prior["observation_id"], False
            revision_number = int(prior["revision_number"]) + 1 if prior else 1
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_observations (
                    observation_id, scope_key, collector_account_id, provider,
                    provider_user_id, workspace_id, quota_owner_id, source_kind,
                    source_id, revision_fingerprint, revision_number, surface,
                    conversation_id, payload, observed_at, run_id, schema_version,
                    provenance, supersedes_observation_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                    %s, %s, %s, %s::jsonb, %s
                )
                """,
                (
                    observation_id,
                    key,
                    scope.collector_account_id,
                    scope.provider,
                    scope.provider_user_id,
                    scope.workspace_id,
                    scope.quota_owner_id,
                    context.source_kind,
                    context.source_id,
                    revision_fingerprint,
                    revision_number,
                    scope.surface,
                    sanitized.get("conversation_id") or sanitized.get("conversationId"),
                    json.dumps(sanitized, separators=(",", ":"), default=str),
                    ensure_utc(context.observed_at),
                    context.run_id,
                    context.schema_version,
                    json.dumps(provenance, separators=(",", ":"), default=str),
                    prior["observation_id"] if prior else None,
                ),
            )
            _record_activity_provenance(
                cur,
                key,
                "observation",
                observation_id,
                scope.collector_account_id,
                context.observed_at,
            )
        return observation_id, True

    def upsert_attempt(
        self,
        scope: LedgerScope,
        attempt: AttemptRecord,
        context: IngestContext,
    ) -> AttemptUpsertResult:
        key = scope_key(scope)
        aliases = _unique_aliases(attempt.aliases)
        attempt_payload = _attempt_payload(attempt, aliases)
        self.assert_safe_record(attempt_payload)
        projection_fingerprint = fingerprint_value(attempt_payload)
        with self.connect() as conn, conn.cursor() as cur:
            _lock_scope(cur, key)
            current = _attempt(cur, key, attempt.attempt_id)
            status = "inserted"
            if current is not None:
                if (
                    current["projection_fingerprint"] == projection_fingerprint
                    and current["collector_account_id"] == scope.collector_account_id
                    and not current["tombstone"]
                ):
                    conflicts = _merge_attempt_links(
                        self,
                        cur,
                        scope,
                        attempt.attempt_id,
                        aliases,
                        context.observed_at,
                    )
                    return AttemptUpsertResult(
                        attempt.attempt_id,
                        "deduplicated",
                        conflicts,
                    )
                revision = int(current["revision"]) + 1
                status = "updated"
            else:
                revision = 1
            if attempt.revision and current is None:
                revision = int(attempt.revision)
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
                    projection_fingerprint, warnings, tombstone, observed_at,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, FALSE, %s, %s
                )
                """,
                (
                    attempt.attempt_id,
                    key,
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
                    json.dumps(attempt.warnings, separators=(",", ":"), default=str),
                    _utc_or_none(context.observed_at),
                    _utc_or_none(context.observed_at),
                ),
            )
            cur.execute(
                """
                INSERT INTO public.chatgpt_usage_attempt_revisions (
                    scope_key, attempt_id, revision, projection_fingerprint, payload,
                    source_kind, source_id, run_id, schema_version,
                    collector_account_id, recorded_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
                """,
                (
                    key,
                    attempt.attempt_id,
                    revision,
                    projection_fingerprint,
                    json.dumps(attempt_payload, separators=(",", ":"), default=str),
                    context.source_kind,
                    context.source_id,
                    context.run_id,
                    context.schema_version,
                    scope.collector_account_id,
                    _utc_or_none(context.observed_at),
                ),
            )
            conflicts = _merge_attempt_links(
                self,
                cur,
                scope,
                attempt.attempt_id,
                aliases,
                context.observed_at,
            )
            _record_activity_provenance(
                cur,
                key,
                "attempt",
                attempt.attempt_id,
                scope.collector_account_id,
                context.observed_at,
            )
        return AttemptUpsertResult(attempt.attempt_id, status, conflicts)

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
        key = scope_key(scope)
        safe_details = sanitize_mapping(details or {})
        self.assert_safe_record(safe_details)
        gap_id = stable_id(key, source_kind, source_id, reason)
        with self.connect() as conn, conn.cursor() as cur:
            _lock_scope(cur, key)
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
                    key,
                    scope.collector_account_id,
                    source_kind,
                    source_id,
                    reason,
                    state,
                    ensure_utc(seen_at),
                    ensure_utc(seen_at),
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
        key = scope_key(scope)
        with self.connect() as conn, conn.cursor() as cur:
            _lock_scope(cur, key)
            cur.execute(
                """
                UPDATE public.chatgpt_usage_coverage_gaps
                SET state = 'resolved', last_seen_at = %s
                WHERE scope_key = %s AND source_kind = %s AND source_id = %s
                  AND state = 'open' AND last_seen_at <= %s
                """,
                (ensure_utc(seen_at), key, source_kind, source_id, ensure_utc(seen_at)),
            )

    def count_attempts(
        self,
        account: str,
        *,
        model_family: Optional[str] = None,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
    ) -> UsageCounts:
        clauses = [
            "scope_key IN ("
            "SELECT scope_key FROM public.chatgpt_usage_scopes "
            "WHERE collector_account_id = %s)",
            "NOT tombstone",
        ]
        params: list[Any] = [account]
        if model_family is not None:
            clauses.append(
                "(requested_family = %s OR recorded_final_family = %s "
                "OR resolved_family = %s)"
            )
            params.extend((model_family, model_family, model_family))
        if window_start is not None:
            clauses.append(
                "COALESCE(attempt_time, earliest_possible_at, latest_possible_at) >= %s"
            )
            params.append(ensure_utc(window_start))
        if window_end is not None:
            clauses.append(
                "COALESCE(attempt_time, earliest_possible_at, latest_possible_at) < %s"
            )
            params.append(ensure_utc(window_end))
        where = " AND ".join(clauses)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM public.chatgpt_usage_attempts "
                f"WHERE {where}",
                params,
            )
            total = int(cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*) FROM public.chatgpt_usage_attempts "
                f"WHERE {where} AND completed_answer",
                params,
            )
            completed = int(cur.fetchone()[0])
            result = UsageCounts(total=total, completed=completed)
            for field_name, column in _COUNT_COLUMNS:
                cur.execute(
                    "SELECT "
                    f"{column}, COUNT(*) FROM public.chatgpt_usage_attempts "
                    f"WHERE {where} AND {column} IS NOT NULL GROUP BY {column}",
                    params,
                )
                setattr(result, field_name, {row[0]: int(row[1]) for row in cur.fetchall()})
        return result

    def close(self) -> None:
        return None

    @staticmethod
    def assert_safe_record(value: Any) -> None:
        assert_no_secrets(value)


def scope_key(scope: LedgerScope) -> str:
    owner = [
        scope.provider,
        scope.provider_user_id,
        scope.workspace_id,
        scope.quota_owner_id,
        scope.surface,
    ]
    if (
        scope.provider_user_id is None
        or scope.workspace_id is None
        or scope.quota_owner_id is None
    ):
        owner[:0] = ["unverified", scope.collector_account_id]
    return fingerprint_value(owner)


def stable_id(*parts: str) -> str:
    return fingerprint_value(list(parts))


def fingerprint_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(_sorted_value(value), separators=(",", ":"), ensure_ascii=False)


def sanitize_provenance(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if value is None:
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
        SELECT observation_id, revision_number, revision_fingerprint
        FROM public.chatgpt_usage_observations
        WHERE scope_key = %s AND source_kind = %s AND source_id = %s
        ORDER BY revision_number DESC
        LIMIT 1
        """,
        (scope_key_value, context.source_kind, context.source_id),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return dict(
        zip(("observation_id", "revision_number", "revision_fingerprint"), row)
    )


def _attempt(
    cur: psycopg.Cursor,
    scope_key_value: str,
    attempt_id: str,
) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT revision, projection_fingerprint, collector_account_id, tombstone
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
            ("revision", "projection_fingerprint", "collector_account_id", "tombstone"),
            row,
        )
    )


def _merge_attempt_links(
    ledger: PgLedger,
    cur: psycopg.Cursor,
    scope: LedgerScope,
    attempt_id: str,
    aliases: Sequence[tuple[str, str]],
    seen_at: datetime,
) -> int:
    conflicts = 0
    key = scope_key(scope)
    for alias_kind, alias_value in aliases:
        safe_value = sanitize_token(alias_value)
        if safe_value is None:
            continue
        cur.execute(
            """
            SELECT attempt_id FROM public.chatgpt_usage_attempt_aliases
            WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
            """,
            (key, alias_kind, safe_value),
        )
        row = cur.fetchone()
        if row is not None and row[0] != attempt_id:
            conflicts += 1
            _record_alias_collision(
                ledger,
                cur,
                scope,
                alias_kind,
                safe_value,
                str(row[0]),
                attempt_id,
                seen_at,
            )
            continue
        if row is not None:
            cur.execute(
                """
                UPDATE public.chatgpt_usage_attempt_aliases
                SET last_seen_at = GREATEST(last_seen_at, %s)
                WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
                """,
                (ensure_utc(seen_at), key, alias_kind, safe_value),
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
                key,
                alias_kind,
                safe_value,
                attempt_id,
                ensure_utc(seen_at),
                ensure_utc(seen_at),
            ),
        )
    return conflicts


def _record_alias_collision(
    ledger: PgLedger,
    cur: psycopg.Cursor,
    scope: LedgerScope,
    alias_kind: str,
    alias_value: str,
    existing_attempt_id: str,
    incoming_attempt_id: str,
    seen_at: datetime,
) -> None:
    key = scope_key(scope)
    source_id = f"{alias_kind}:{alias_value}"
    gap_id = stable_id(key, "attempt_alias", source_id, "alias_collision")
    details = {
        "alias_kind": alias_kind,
        "existing_attempt_id": existing_attempt_id,
        "incoming_attempt_id": incoming_attempt_id,
    }
    ledger.assert_safe_record(details)
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
            last_seen_at = GREATEST(
                public.chatgpt_usage_coverage_gaps.last_seen_at,
                EXCLUDED.last_seen_at
            ),
            details = EXCLUDED.details
        """,
        (
            gap_id,
            key,
            scope.collector_account_id,
            source_id,
            ensure_utc(seen_at),
            ensure_utc(seen_at),
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
    return [
        statement
        for statement in statements
        if statement and statement.upper() not in {"BEGIN", "COMMIT"}
    ]


def _lock_scope(cur: psycopg.Cursor, key: str) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)", (key,))


def _unique_aliases(aliases: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for item in aliases:
        normalized = (str(item[0]), str(item[1]))
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _attempt_payload(
    attempt: AttemptRecord,
    aliases: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "attemptId": attempt.attempt_id,
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
        "aliases": sorted(aliases),
        "evidenceMessageIds": sorted(attempt.evidence_message_ids),
        "warnings": sorted(attempt.warnings),
    }


def _scope_payload(scope: LedgerScope) -> dict[str, Any]:
    return {
        "collector_account_id": scope.collector_account_id,
        "provider": scope.provider,
        "provider_user_id": scope.provider_user_id,
        "workspace_id": scope.workspace_id,
        "quota_owner_id": scope.quota_owner_id,
        "surface": scope.surface,
    }


def _without_keys(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _utc_or_none(value: Optional[datetime]) -> Optional[datetime]:
    return ensure_utc(value) if value is not None else None


def _sanitize_provenance_value(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return sanitize_token(value)
    if isinstance(value, Sequence):
        values: list[str] = []
        for item in value:
            token = sanitize_token(item) if isinstance(item, str) else None
            if token is not None:
                values.append(token)
        return values[:32]
    return None


def _sorted_value(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_sorted_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _sorted_value(value[key]) for key in sorted(value, key=str)}
    return value


_COUNT_COLUMNS = (
    ("by_requested_family", "requested_family"),
    ("by_recorded_final_family", "recorded_final_family"),
    ("by_resolved_family", "resolved_family"),
    ("by_requested_model_raw", "requested_model_raw"),
    ("by_recorded_final_model_raw", "recorded_final_model_raw"),
    ("by_resolved_model_raw", "resolved_model_raw"),
)
