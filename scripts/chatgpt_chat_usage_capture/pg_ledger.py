"""PostgreSQL persistence contract for ChatGPT ordinary-Chat usage metadata."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Iterable, Mapping, Optional, Sequence

import psycopg

from .models import AttemptRecord
from .privacy import (
    assert_no_secrets,
    classify_surface,
    sanitize_metadata,
    sanitize_token,
)
from .timeutil import ensure_utc, isoformat_utc


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
        "transfer_schema_version",
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
_ALLOWED_TIME_BASES = frozenset({"provider", "dispatch", "user_message", "bounded_interval", "unknown"})
_ALLOWED_OUTCOMES = frozenset(
    {
        "completed",
        "failed_after_start",
        "cancelled_after_start",
        "completion_unknown",
        "rejected_before_start",
        "unresolved",
        "unknown",
    }
)
_ALLOWED_ALIAS_KINDS = frozenset({"generation", "request", "prompt"})
_ALLOWED_GAP_STATES = frozenset({"open", "resolved", "unknown"})
_OBSERVATION_KEY_ALIASES = {
    "conversationId": "conversation_id",
    "messageId": "message_id",
    "nodeId": "node_id",
    "parentId": "parent_id",
    "currentNode": "current_node",
    "modelSlug": "model_slug",
    "requestedModel": "requested_model",
    "requestedMode": "requested_mode",
    "reasoningEffort": "reasoning_effort",
    "defaultModelSlug": "default_model_slug",
    "generationId": "generation_id",
    "requestId": "request_id",
    "messageRequestId": "message_request_id",
    "createdAt": "created_at",
    "updatedAt": "updated_at",
    "isArchived": "is_archived",
    "isStarred": "is_starred",
    "hasVersions": "has_versions",
    "hasPreviousPage": "has_previous_page",
    "startCursor": "start_cursor",
    "schemaVersion": "schema_version",
    "errorType": "error_type",
    "errorCode": "error_code",
    "quarantineState": "quarantine_state",
    "transferSchemaVersion": "transfer_schema_version",
}
_OBSERVATION_FIELDS = frozenset(
    {
        "id",
        "conversation_id",
        "message_id",
        "node_id",
        "parent",
        "parent_id",
        "current_node",
        "model_slug",
        "requested_model",
        "requested_model_slug",
        "requested_mode",
        "reasoning_effort",
        "default_model_slug",
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
        "start_cursor",
        "offset",
        "limit",
        "total",
        "page_info",
        "metadata",
        "author",
        "workspace_id",
        "gizmo_id",
        "surface",
        "origin",
        "shared",
        "imported",
        "copied",
        "coverage",
        "schema_version",
        "error_type",
        "error_code",
        "quarantine",
        "quarantine_state",
        "warnings",
        "schema_fingerprint",
        "transfer_schema_version",
    }
)
_OBSERVATION_TOKEN_FIELDS = frozenset(
    {
        "id",
        "conversation_id",
        "message_id",
        "node_id",
        "parent",
        "parent_id",
        "current_node",
        "model_slug",
        "requested_model",
        "requested_model_slug",
        "requested_mode",
        "reasoning_effort",
        "default_model_slug",
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
        "workspace_id",
        "gizmo_id",
        "coverage",
        "schema_version",
        "error_type",
        "error_code",
        "quarantine_state",
        "schema_fingerprint",
    }
)
_OBSERVATION_BOOLEAN_FIELDS = frozenset(
    {
        "end_turn",
        "is_archived",
        "is_starred",
        "has_versions",
        "has_previous_page",
        "shared",
        "imported",
        "copied",
    }
)
_OBSERVATION_NUMBER_FIELDS = frozenset({"offset", "limit", "total", "weight", "timestamp_"})
_OBSERVATION_LIST_FIELDS = frozenset({"children"})


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

    @contextmanager
    def transaction(
        self,
        *,
        scope: Optional[LedgerScope] = None,
        seen_at: Optional[datetime] = None,
    ) -> Iterator["PgLedgerPage"]:
        """Open one page write transaction; provider/network work stays outside."""
        with self.connect() as conn:
            page = PgLedgerPage(self, conn)
            if scope is not None:
                page.bind_scope(scope, seen_at=seen_at or datetime.now().astimezone())
            yield page

    page_transaction = transaction

    def upsert_scope(self, scope: LedgerScope, *, seen_at: datetime) -> str:
        with self.transaction() as page:
            return page.bind_scope(scope, seen_at=seen_at).scope_key

    def record_observation(
        self,
        scope: LedgerScope,
        context: IngestContext,
        payload: Mapping[str, Any],
    ) -> tuple[str, bool]:
        with self.transaction(scope=scope, seen_at=context.observed_at) as page:
            return page.record_observation(scope, context, payload)

    def upsert_attempt(
        self,
        scope: LedgerScope,
        attempt: AttemptRecord,
        context: IngestContext,
    ) -> AttemptUpsertResult:
        with self.transaction(scope=scope, seen_at=context.observed_at) as page:
            return page.upsert_attempt(scope, attempt, context)

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
        with self.transaction(scope=scope, seen_at=seen_at) as page:
            return page.record_coverage_gap(
                scope,
                source_kind=source_kind,
                source_id=source_id,
                reason=reason,
                state=state,
                details=details,
                seen_at=seen_at,
            )

    def resolve_coverage_gaps(
        self,
        scope: LedgerScope,
        *,
        source_kind: str,
        source_id: str,
        seen_at: datetime,
    ) -> None:
        with self.transaction(scope=scope, seen_at=seen_at) as page:
            page.resolve_coverage_gaps(
                scope,
                source_kind=source_kind,
                source_id=source_id,
                seen_at=seen_at,
            )

    def scope_keys_for_account(self, collector_account_id: str) -> tuple[str, ...]:
        """Return active and retired scope keys bound to one collector."""
        self.assert_safe_record({"collector_account_id": collector_account_id})
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE scope_tree(scope_key) AS (
                    SELECT scope_key
                    FROM public.chatgpt_usage_scope_bindings
                    WHERE collector_account_id = %s
                    UNION
                    SELECT r.canonical_scope_key
                    FROM public.chatgpt_usage_scope_redirects AS r
                    JOIN scope_tree AS s ON s.scope_key = r.retired_scope_key
                )
                SELECT scope_key FROM scope_tree ORDER BY scope_key
                """,
                (collector_account_id,),
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
            total_row = cur.fetchone()
            if total_row is None:
                raise LedgerError("count query returned no row")
            total = int(total_row[0])
            cur.execute(
                "SELECT COUNT(*) FROM public.chatgpt_usage_attempts "
                f"WHERE {where} AND completed_answer",
                params,
            )
            completed_row = cur.fetchone()
            if completed_row is None:
                raise LedgerError("completed count query returned no row")
            completed = int(completed_row[0])
            result = UsageCounts(total=total, completed=completed)
            for field_name, column in _COUNT_COLUMNS:
                cur.execute(
                    "SELECT "
                    f"{column}, COUNT(*) FROM public.chatgpt_usage_attempts "
                    f"WHERE {where} AND {column} IS NOT NULL GROUP BY {column}",
                    params,
                )
            """
            params.extend((family_token, family_token, family_token))
        params.extend(time_params)
        query = f"""
            WITH scoped AS (
                SELECT attempts.*
                FROM public.chatgpt_usage_attempts AS attempts
                WHERE EXISTS (
                    SELECT 1
                    FROM public.chatgpt_usage_scopes AS scopes
                    WHERE scopes.scope_key = attempts.scope_key
                      AND scopes.collector_account_id = %s
                )
                  AND NOT attempts.tombstone
                  {model_clause}
            ),
            classified AS (
                SELECT
                    scoped.*,
                    CASE
                        WHEN surface = 'unknown' THEN 'unknown_surface'
                        WHEN surface <> 'chat' THEN 'excluded_surface'
                        WHEN origin IN ('shared', 'imported', 'copied')
                            THEN 'excluded_origin'
                        WHEN outcome = 'rejected_before_start'
                            OR NOT (generation_started OR completed_answer)
                            THEN 'excluded_non_generation'
                        WHEN identity_basis IN ('unresolved', 'unknown')
                            THEN 'unknown_identity'
                        ELSE 'eligible'
                    END AS evidence_state,
                    {time_state_sql} AS time_state
                FROM scoped
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
                ) AS unknown_identity,
                count(*) FILTER (
                    WHERE evidence_state = 'unknown_surface'
                ) AS unknown_surface,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND (origin IS NULL OR origin = 'unknown')
                ) AS unknown_origin,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state = 'definite'
                      AND COALESCE(
                          NULLIF(requested_family, 'unknown'),
                          NULLIF(recorded_final_family, 'unknown'),
                          NULLIF(resolved_family, 'unknown')
                      ) IS NULL
                ) AS unknown_model,
                count(*) FILTER (
                    WHERE evidence_state = 'excluded_surface'
                ) AS excluded_surface,
                count(*) FILTER (
                    WHERE evidence_state = 'excluded_origin'
                ) AS excluded_origin,
                count(*) FILTER (
                    WHERE evidence_state = 'excluded_non_generation'
                ) AS excluded_non_generation,
                count(*) FILTER (
                    WHERE evidence_state = 'eligible'
                      AND time_state = 'definite'
                      AND outcome IN (
                          'failed_after_start',
                          'cancelled_after_start',
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
                        FROM classified
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
                        FROM classified
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
                        FROM classified
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
                        FROM classified
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
                        FROM classified
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
                        FROM classified
                        WHERE evidence_state = 'eligible'
                          AND time_state = 'definite'
                          AND resolved_model_raw IS NOT NULL
                        GROUP BY resolved_model_raw
                    ) AS resolved_models
                ) AS by_resolved_model_raw
            FROM classified
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

    def __init__(self, ledger: PgLedger, conn: psycopg.Connection) -> None:
        self.ledger = ledger
        self.conn = conn
        self._bindings: dict[str, LedgerBinding] = {}

    @staticmethod
    def assert_safe_record(value: Any) -> None:
        assert_no_secrets(value)

    def bind_scope(self, scope: LedgerScope, *, seen_at: datetime) -> LedgerBinding:
        self.ledger.assert_safe_record(_scope_payload(scope))
        key = scope_key(scope)
        identity_state = _scope_identity_state(scope)
        observed_at = ensure_utc(seen_at)
        cached = self._bindings.get(scope.collector_account_id)
        if cached is not None and cached.scope_key == key:
            with self.conn.cursor() as cur:
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
                        observed_at,
                        key,
                        scope.collector_account_id,
                        cached.binding_generation,
                    ),
                )
            return cached

        with self.conn.cursor() as cur:
            _lock_scope_registry(cur)
            _ensure_scope_row(
                cur,
                scope=scope,
                key=key,
                identity_state=identity_state,
                seen_at=observed_at,
            )
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
            if current is not None and current[0] == key:
                generation = int(current[1])
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_scope_bindings
                    SET last_seen_at = GREATEST(last_seen_at, %s)
                    WHERE scope_key = %s
                      AND collector_account_id = %s
                      AND binding_generation = %s
                    """,
                    (observed_at, key, scope.collector_account_id, generation),
                )
            else:
                prior_scope_key = str(current[0]) if current is not None else None
                prior_generation = int(current[1]) if current is not None else 0
                prior_identity_state = None
                if prior_scope_key is not None:
                    cur.execute(
                        """
                        SELECT identity_state
                        FROM public.chatgpt_usage_scopes
                        WHERE scope_key = %s
                        FOR UPDATE
                        """,
                        (prior_scope_key,),
                    )
                    prior_scope = cur.fetchone()
                    prior_identity_state = (
                        str(prior_scope[0]) if prior_scope is not None else None
                    )
                    cur.execute(
                        """
                        UPDATE public.chatgpt_usage_scope_bindings
                        SET binding_state = 'retired', retired_at = %s,
                            last_seen_at = GREATEST(last_seen_at, %s)
                        WHERE collector_account_id = %s
                          AND binding_state = 'active'
                        """,
                        (
                            observed_at,
                            observed_at,
                            scope.collector_account_id,
                        ),
                    )
                    if (
                        prior_identity_state == "provisional"
                        and identity_state == "verified"
                    ):
                        _redirect_scope(
                            cur,
                            retired_scope_key=prior_scope_key,
                            canonical_scope_key=key,
                            seen_at=observed_at,
                        )
                generation = prior_generation + 1
                cur.execute(
                    """
                    INSERT INTO public.chatgpt_usage_scope_bindings (
                        scope_key, collector_account_id, binding_generation,
                        binding_state, first_seen_at, last_seen_at
                    ) VALUES (%s, %s, %s, 'active', %s, %s)
                    """,
                    (
                        key,
                        scope.collector_account_id,
                        generation,
                        observed_at,
                        observed_at,
                    ),
                )
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
        binding = self.bind_scope(scope, seen_at=context.observed_at)
        sanitized = sanitize_mapping(payload)
        self.ledger.assert_safe_record(sanitized)
        provenance = sanitize_provenance(
            {
                **(context.provenance or {}),
                "collector_account_id": scope.collector_account_id,
                "schema_version": context.schema_version,
                "source_id": context.source_id,
                "source_kind": context.source_kind,
            }
        )
        self.ledger.assert_safe_record(provenance)
        stable_payload = _without_keys(sanitized, "provenance", "run_id")
        revision_fingerprint = fingerprint_value(
            {"payload": stable_payload, "provenance": provenance}
        )
        observed_at = ensure_utc(context.observed_at)
        with self.conn.cursor() as cur:
            _lock_scope(cur, binding.scope_key)
            prior = _latest_observation(cur, binding.scope_key, context)
            if prior is not None and prior["revision_fingerprint"] == revision_fingerprint:
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
                        context.run_id,
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
                    scope.collector_account_id,
                    observed_at,
                )
                return prior["observation_id"], False

            current = _current_observation(cur, binding.scope_key, context)
            occurrence_number = int(prior["occurrence_number"]) + 1 if prior else 1
            observation_id = stable_id(
                binding.scope_key,
                context.source_kind,
                context.source_id,
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
                    scope.collector_account_id,
                    scope.provider,
                    scope.provider_user_id,
                    scope.workspace_id,
                    scope.quota_owner_id,
                    context.source_kind,
                    context.source_id,
                    revision_fingerprint,
                    occurrence_number,
                    occurrence_number,
                    scope.surface,
                    sanitized.get("conversation_id") or sanitized.get("conversationId"),
                    json.dumps(sanitized, separators=(",", ":"), default=str),
                    observed_at,
                    context.run_id,
                    context.schema_version,
                    json.dumps(provenance, separators=(",", ":"), default=str),
                    observed_at,
                    observed_at,
                    context.run_id,
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
                scope.collector_account_id,
                observed_at,
            )
            return observation_id, True

    def upsert_attempt(
        self,
        scope: LedgerScope,
        attempt: AttemptRecord,
        context: IngestContext,
    ) -> AttemptUpsertResult:
        binding = self.bind_scope(scope, seen_at=context.observed_at)
        aliases = _unique_aliases(attempt.aliases)
        observed_at = ensure_utc(context.observed_at)
        with self.conn.cursor() as cur:
            _lock_scope(cur, binding.scope_key)
            (
                effective_id,
                identity_conflicts,
                retired_attempt_id,
            ) = _resolve_attempt_identity(
                self,
                cur,
                scope,
                binding.scope_key,
                attempt,
                aliases,
                observed_at,
            )
            attempt_payload = _attempt_payload(
                attempt,
                aliases,
                attempt_id=effective_id,
            )
            self.ledger.assert_safe_record(attempt_payload)
            projection_fingerprint = fingerprint_value(attempt_payload)
            current = _attempt(cur, binding.scope_key, effective_id)
            if current is not None and current["tombstone"]:
                _record_identity_gap(
                    self,
                    cur,
                    scope,
                    binding.scope_key,
                    source_id=effective_id,
                    reason="retired_attempt_reappeared",
                    details={"attempt_id": effective_id},
                    seen_at=observed_at,
                )
                return AttemptUpsertResult(
                    effective_id,
                    "quarantined",
                    identity_conflicts + 1,
                )
            if (
                current is not None
                and current["projection_fingerprint"] == projection_fingerprint
            ):
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempts
                    SET last_seen_at = GREATEST(last_seen_at, %s)
                    WHERE scope_key = %s AND attempt_id = %s
                    """,
                    (observed_at, binding.scope_key, effective_id),
                )
                conflicts = _merge_attempt_links(
                    self,
                    cur,
                    scope,
                    effective_id,
                    aliases,
                    observed_at,
                )
                _record_activity_provenance(
                    cur,
                    binding.scope_key,
                    "attempt",
                    effective_id,
                    scope.collector_account_id,
                    observed_at,
                )
                return AttemptUpsertResult(
                    effective_id,
                    "deduplicated",
                    identity_conflicts + conflicts,
                )

            revision = _next_attempt_revision(cur, binding.scope_key, effective_id)
            stale = current is not None and observed_at < current["observed_at"]
            if current is None:
                _insert_attempt_head(
                    cur,
                    scope=scope,
                    scope_key_value=binding.scope_key,
                    attempt=attempt,
                    attempt_id=effective_id,
                    revision=revision,
                    projection_fingerprint=projection_fingerprint,
                    observed_at=observed_at,
                )
                status = "inserted"
            elif not stale:
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
                        tombstone = FALSE, superseded_by_attempt_id = NULL
                    WHERE scope_key = %s AND attempt_id = %s
                    """,
                    _attempt_projection_params(
                        scope,
                        attempt,
                        revision,
                        projection_fingerprint,
                        observed_at,
                        binding.scope_key,
                        effective_id,
                    ),
                )
                status = "updated"
            else:
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempts
                    SET last_seen_at = GREATEST(last_seen_at, %s)
                    WHERE scope_key = %s AND attempt_id = %s
                    """,
                    (observed_at, binding.scope_key, effective_id),
                )
                status = "stale"

            if retired_attempt_id is not None:
                _reassign_attempt_aliases(
                    cur,
                    scope_key_value=binding.scope_key,
                    previous_attempt_id=retired_attempt_id,
                    canonical_attempt_id=effective_id,
                    seen_at=observed_at,
                )
            if not stale:
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempt_revisions
                    SET is_current_projection = FALSE
                    WHERE scope_key = %s AND attempt_id = %s
                      AND is_current_projection = TRUE
                    """,
                    (binding.scope_key, effective_id),
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
                    binding.scope_key,
                    effective_id,
                    revision,
                    projection_fingerprint,
                    json.dumps(attempt_payload, separators=(",", ":"), default=str),
                    context.source_kind,
                    context.source_id,
                    context.run_id,
                    context.schema_version,
                    scope.collector_account_id,
                    observed_at,
                    not stale,
                ),
            )
            conflicts = _merge_attempt_links(
                self,
                cur,
                scope,
                effective_id,
                aliases,
                observed_at,
            )
            _record_activity_provenance(
                cur,
                binding.scope_key,
                "attempt",
                effective_id,
                scope.collector_account_id,
                observed_at,
            )
            return AttemptUpsertResult(
                effective_id,
                status,
                identity_conflicts + conflicts,
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
        binding = self.bind_scope(scope, seen_at=seen_at)
        safe_details = sanitize_mapping(details or {})
        self.ledger.assert_safe_record(safe_details)
        gap_id = stable_id(binding.scope_key, source_kind, source_id, reason)
        observed_at = ensure_utc(seen_at)
        with self.conn.cursor() as cur:
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
                    scope.collector_account_id,
                    source_kind,
                    source_id,
                    reason,
                    state,
                    observed_at,
                    observed_at,
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
        binding = self.bind_scope(scope, seen_at=seen_at)
        observed_at = ensure_utc(seen_at)
        with self.conn.cursor() as cur:
            _lock_scope(cur, binding.scope_key)
            cur.execute(
                """
                UPDATE public.chatgpt_usage_coverage_gaps
                SET state = 'resolved', last_seen_at = %s
                WHERE scope_key = %s AND source_kind = %s AND source_id = %s
                  AND state = 'open' AND last_seen_at <= %s
                """,
                (
                    observed_at,
                    binding.scope_key,
                    source_kind,
                    source_id,
                    observed_at,
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
    if (
        scope.provider_user_id is not None
        and scope.workspace_id is not None
        and scope.quota_owner_id is not None
    ):
        return "verified"
    return "provisional"


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
    cur.execute(
        """
        INSERT INTO public.chatgpt_usage_scope_redirects (
            retired_scope_key, canonical_scope_key, reason,
            first_seen_at, last_seen_at
        ) VALUES (%s, %s, 'verified_identity', %s, %s)
        ON CONFLICT (retired_scope_key) DO UPDATE SET
            canonical_scope_key = EXCLUDED.canonical_scope_key,
            last_seen_at = GREATEST(
                public.chatgpt_usage_scope_redirects.last_seen_at,
                EXCLUDED.last_seen_at
            )
        """,
        (retired_scope_key, canonical_scope_key, seen_at, seen_at),
    )
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
        SELECT observation_id, observed_at, occurrence_number, revision_fingerprint
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
            ("observation_id", "observed_at", "occurrence_number", "revision_fingerprint"),
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
               identity_basis, observed_at, last_seen_at
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
                "identity_basis",
                "observed_at",
                "last_seen_at",
            ),
            row,
        )
    )


def _matching_alias_ids(
    cur: psycopg.Cursor,
    scope_key_value: str,
    aliases: Sequence[tuple[str, str]],
) -> set[str]:
    matches: set[str] = set()
    for alias_kind, alias_value in aliases:
        safe_value = sanitize_token(alias_value)
        if safe_value is None:
            continue
        cur.execute(
            """
            SELECT attempt_id
            FROM public.chatgpt_usage_attempt_aliases
            WHERE scope_key = %s AND alias_kind = %s AND alias_value = %s
            """,
            (scope_key_value, alias_kind, safe_value),
        )
        row = cur.fetchone()
        if row is not None:
            matches.add(str(row[0]))
    return matches


def _identity_rank(identity_basis: str) -> int:
    return {
        "generation": 4,
        "request": 3,
        "provisional": 2,
        "unresolved": 1,
    }.get(identity_basis, 0)


def _resolve_attempt_identity(
    page: PgLedgerPage,
    cur: psycopg.Cursor,
    scope: LedgerScope,
    scope_key_value: str,
    attempt: AttemptRecord,
    aliases: Sequence[tuple[str, str]],
    seen_at: datetime,
) -> tuple[str, int, Optional[str]]:
    incoming_id = attempt.attempt_id
    matches = _matching_alias_ids(cur, scope_key_value, aliases)
    matches.discard(incoming_id)
    if len(matches) > 1:
        _record_identity_gap(
            page,
            cur,
            scope,
            scope_key_value,
            source_id=incoming_id,
            reason="ambiguous_attempt_alias",
            details={
                "attempt_id": incoming_id,
                "matching_attempt_ids": sorted(matches)[:16],
            },
            seen_at=seen_at,
        )
        return incoming_id, len(matches), None
    if not matches:
        return incoming_id, 0, None

    candidate_id = next(iter(matches))
    incoming = _attempt(cur, scope_key_value, incoming_id)
    candidate = _attempt(cur, scope_key_value, candidate_id)
    if incoming is not None:
        _record_identity_gap(
            page,
            cur,
            scope,
            scope_key_value,
            source_id=incoming_id,
            reason="attempt_alias_collision",
            details={
                "attempt_id": incoming_id,
                "matching_attempt_id": candidate_id,
            },
            seen_at=seen_at,
        )
        return incoming_id, 1, None
    if candidate is None or candidate["tombstone"]:
        _record_identity_gap(
            page,
            cur,
            scope,
            scope_key_value,
            source_id=incoming_id,
            reason="alias_to_retired_attempt",
            details={
                "attempt_id": incoming_id,
                "matching_attempt_id": candidate_id,
            },
            seen_at=seen_at,
        )
        return incoming_id, 1, None

    if (
        _identity_rank(attempt.identity_basis)
        > _identity_rank(str(candidate["identity_basis"]))
        and str(candidate["identity_basis"]) in {"provisional", "unresolved"}
    ):
        _retire_provisional_attempt(
            cur,
            scope_key_value=scope_key_value,
            previous_attempt_id=candidate_id,
            canonical_attempt_id=incoming_id,
            seen_at=seen_at,
        )
        return incoming_id, 0, candidate_id
    return candidate_id, 0, None


def _retire_provisional_attempt(
    cur: psycopg.Cursor,
    *,
    scope_key_value: str,
    previous_attempt_id: str,
    canonical_attempt_id: str,
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
            canonical_attempt_id,
            seen_at,
            seen_at,
            scope_key_value,
            previous_attempt_id,
        ),
    )
    cur.execute(
        """
        UPDATE public.chatgpt_usage_attempt_revisions
        SET is_current_projection = FALSE
        WHERE scope_key = %s AND attempt_id = %s
        """,
        (scope_key_value, previous_attempt_id),
    )


def _reassign_attempt_aliases(
    cur: psycopg.Cursor,
    *,
    scope_key_value: str,
    previous_attempt_id: str,
    canonical_attempt_id: str,
    seen_at: datetime,
) -> None:
    cur.execute(
        """
        UPDATE public.chatgpt_usage_attempt_aliases
        SET attempt_id = %s,
            last_seen_at = GREATEST(last_seen_at, %s)
        WHERE scope_key = %s AND attempt_id = %s
        """,
        (canonical_attempt_id, seen_at, scope_key_value, previous_attempt_id),
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
    safe_details = sanitize_mapping(details)
    page.ledger.assert_safe_record(safe_details)
    gap_id = stable_id(scope_key_value, "attempt_identity", source_id, reason)
    cur.execute(
        """
        INSERT INTO public.chatgpt_usage_coverage_gaps (
            gap_id, scope_key, collector_account_id, source_kind, source_id,
            reason, state, first_seen_at, last_seen_at, details
        ) VALUES (
            %s, %s, %s, 'attempt_identity', %s, %s, 'open', %s, %s, %s::jsonb
        )
        ON CONFLICT (scope_key, source_kind, source_id, reason) DO UPDATE SET
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
            scope.collector_account_id,
            source_id,
            reason,
            seen_at,
            seen_at,
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
            projection_fingerprint, warnings, tombstone, observed_at,
            updated_at, last_seen_at, superseded_by_attempt_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, FALSE, %s, %s, %s,
            NULL
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
            json.dumps(attempt.warnings, separators=(",", ":"), default=str),
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
        json.dumps(attempt.warnings, separators=(",", ":"), default=str),
        observed_at,
        observed_at,
        observed_at,
        scope_key_value,
        attempt_id,
    )


def _merge_attempt_links(
    ledger: PgLedger | PgLedgerPage,
    cur: psycopg.Cursor,
    scope: LedgerScope,
    attempt_id: str,
    aliases: Sequence[tuple[str, str]],
    seen_at: datetime,
) -> int:
    conflicts = 0
    key = scope_key(scope)
    for alias_kind, alias_value in aliases:
        if alias_kind not in _ALLOWED_ALIAS_KINDS:
            continue
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
    ledger: PgLedger | PgLedgerPage,
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
    return [statement for statement in statements if statement and statement.upper() not in {"BEGIN", "COMMIT"}]


def _lock_scope(cur: psycopg.Cursor, key: str) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)", (key,))


def _lock_scope_registry(cur: psycopg.Cursor) -> None:
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
        ("chatgpt-usage-scope-registry",),
    )


def _unique_aliases(aliases: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return _sanitize_aliases(aliases)


def _attempt_payload(
    attempt: AttemptRecord,
    aliases: Sequence[tuple[str, str]],
    *,
    attempt_id: Optional[str] = None,
) -> dict[str, Any]:
    return {
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
        "aliases": sorted(aliases),
        "evidenceMessageIds": sorted(attempt.evidence_message_ids),
        "warnings": sorted(attempt.warnings),
        "quarantine": _quarantine_envelope(attempt.warnings),
    }


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
    return IngestContext(
        run_id=_required_token(context.run_id, "run_id"),
        observed_at=ensure_utc(context.observed_at),
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
        warnings=tuple(_safe_tokens(attempt.warnings)),
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


def _observation_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise LedgerError("observation payload must be a mapping")
    projected: dict[str, Any] = {
        "transfer_schema_version": TRANSFER_SCHEMA_VERSION,
    }
    unknown_field_count = 0
    for raw_key, raw_value in payload.items():
        key = _OBSERVATION_KEY_ALIASES.get(str(raw_key), str(raw_key))
        if key not in _OBSERVATION_FIELDS:
            unknown_field_count += 1
            continue
        value = _project_observation_field(key, raw_value)
        if value is not _DROP:
            projected[key] = value
    surface = classify_surface(payload, default=None)
    projected["surface"] = _surface_token(projected.get("surface", surface))
    projected["unknown_field_count"] = min(unknown_field_count, 128)
    for collection_key, count_key in (
        ("items", "item_count"),
        ("messages", "message_count"),
        ("mapping", "mapping_node_count"),
    ):
        if collection_key in payload:
            value = payload[collection_key]
            if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
                projected[count_key] = min(len(value), 800)
    return projected


def _project_observation_field(key: str, value: Any) -> Any:
    if key in {"surface", "origin"}:
        return _surface_token(value) if key == "surface" else _origin_token(value)
    if key in _OBSERVATION_BOOLEAN_FIELDS:
        return value if isinstance(value, bool) else _DROP
    if key in _OBSERVATION_NUMBER_FIELDS:
        return _safe_number(value)
    if key in _OBSERVATION_TOKEN_FIELDS:
        return _optional_token(value)
    if key in _OBSERVATION_LIST_FIELDS:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return _DROP
        return _safe_tokens(value, limit=800)
    if key == "metadata":
        return sanitize_metadata(value) if isinstance(value, Mapping) else _DROP
    if key == "author":
        if not isinstance(value, Mapping):
            return _DROP
        role = _optional_token(value.get("role"))
        return {"role": role} if role is not None else {}
    if key == "page_info":
        return _page_info_envelope(value)
    if key == "quarantine":
        return _quarantine_value(value)
    if key == "warnings":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return _DROP
        return _safe_tokens(value, limit=64)
    if key == "transfer_schema_version":
        return TRANSFER_SCHEMA_VERSION
    return _DROP


def _page_info_envelope(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return _DROP
    out: dict[str, Any] = {}
    for key in ("has_previous_page", "has_next_page"):
        if isinstance(value.get(key), bool):
            out[key] = value[key]
    for key in ("start_cursor", "end_cursor"):
        token = _optional_token(value.get(key))
        if token is not None:
            out[key] = token
    for key in ("offset", "limit", "total"):
        number = _safe_number(value.get(key))
        if number is not _DROP:
            out[key] = number
    return out


def _quarantine_envelope(warnings: Iterable[str]) -> dict[str, Any]:
    reasons = [
        warning.split(":", 1)[1]
        for warning in _safe_tokens(warnings)
        if warning.startswith("quarantine:") and ":" in warning
    ]
    return {
        "state": "quarantined" if reasons else "clear",
        "reasons": reasons[:32],
    }


def _quarantine_value(value: Any) -> Any:
    if isinstance(value, bool):
        return {"state": "quarantined" if value else "clear", "reasons": []}
    if isinstance(value, str):
        token = _optional_token(value)
        return {"state": token or "unknown", "reasons": []}
    if isinstance(value, Mapping):
        state = _optional_token(value.get("state")) or "unknown"
        raw_reasons = value.get("reasons", [])
        reasons = _safe_tokens(
            raw_reasons
            if isinstance(raw_reasons, Sequence) and not isinstance(raw_reasons, (str, bytes, bytearray))
            else []
        )
        return {"state": state, "reasons": reasons[:32]}
    return _DROP


def _coverage_details_envelope(details: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(details, Mapping):
        return {"transfer_schema_version": TRANSFER_SCHEMA_VERSION}
    projected = _observation_envelope(details)
    projected["transfer_schema_version"] = TRANSFER_SCHEMA_VERSION
    return projected


def _safe_number(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _DROP
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return _DROP
    return value


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
                WHEN earliest_possible_at IS NULL
                  OR latest_possible_at IS NULL
                  OR latest_possible_at < earliest_possible_at
                    THEN 'unknown'
                WHEN latest_possible_at < %s
                  OR earliest_possible_at >= %s
                    THEN 'out'
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
                WHEN earliest_possible_at IS NULL
                  OR latest_possible_at IS NULL
                  OR latest_possible_at < earliest_possible_at
                    THEN 'unknown'
                WHEN latest_possible_at < %s THEN 'out'
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
            WHEN earliest_possible_at IS NULL
              OR latest_possible_at IS NULL
              OR latest_possible_at < earliest_possible_at
                THEN 'unknown'
            WHEN earliest_possible_at >= %s THEN 'out'
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
