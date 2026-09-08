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
    sanitize_mapping,
    sanitize_token,
    sanitize_value,
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
_ALLOWED_ALIAS_KINDS = frozenset({"generation", "request", "prompt", "message", "branch"})
_STRONG_ALIAS_KINDS = frozenset({"generation", "message", "branch"})
_ALLOWED_GAP_STATES = frozenset({"open", "resolved", "unknown"})
_ALLOWED_QUARANTINE_STATES = frozenset({"clear", "quarantined", "unknown"})
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
    "hasNextPage": "has_next_page",
    "startCursor": "start_cursor",
    "endCursor": "end_cursor",
    "schemaVersion": "schema_version",
    "schemaFingerprint": "schema_fingerprint",
    "errorType": "error_type",
    "errorCode": "error_code",
    "quarantineState": "quarantine_state",
    "transferSchemaVersion": "transfer_version",
    "transferVersion": "transfer_version",
    "transfer_schema_version": "transfer_version",
    "resolvedModel": "resolved_model",
    "resolvedModelRaw": "resolved_model_raw",
    "requestedModelRaw": "requested_model_raw",
    "recordedFinalModelRaw": "recorded_final_model_raw",
    "futureTimestampQuarantined": "future_timestamp_quarantined",
    "quarantineTimestamp": "quarantine_timestamp",
    "quarantineAt": "quarantine_at",
    "quarantinedAt": "quarantined_at",
    "quarantineWarning": "quarantine_warning",
    "quarantineWarnings": "quarantine_warnings",
    "quarantineReason": "quarantine_reason",
    "evidenceId": "evidence_id",
    "unknownFields": "unknown_fields",
}
_OBSERVATION_FIELDS = frozenset(
    {
        "id",
        "conversation_id",
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
        "reasoning_effort",
        "default_model_slug",
        "recorded_final_model_raw",
        "resolved_model",
        "resolved_model_raw",
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
        "transfer_version",
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
        "children",
        "current_node",
        "model_slug",
        "requested_model",
        "requested_model_raw",
        "requested_model_slug",
        "requested_mode",
        "reasoning_effort",
        "default_model_slug",
        "recorded_final_model_raw",
        "resolved_model",
        "resolved_model_raw",
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
        "quarantine_reason",
        "quarantine_warning",
        "schema_fingerprint",
        "evidence_id",
        "transfer_version",
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
    }
)
_OBSERVATION_NUMBER_FIELDS = frozenset({"offset", "limit", "total", "weight", "timestamp_"})
_OBSERVATION_LIST_FIELDS = frozenset({"children", "unknown_fields"})
_OBSERVATION_COLLECTION_FIELDS = frozenset({"items", "messages", "mapping"})
_OBSERVATION_WARNING_FIELDS = frozenset({"warnings", "quarantine_warnings"})
_OBSERVATION_MAX_DEPTH = 8
_OBSERVATION_MAX_FIELDS = 128
_OBSERVATION_MAX_ITEMS = 800


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
        expected_binding: Optional[LedgerBinding] = None,
    ) -> Iterator["PgLedgerPage"]:
        """Open one page write transaction; provider/network work stays outside."""
        if scope is not None and expected_binding is None:
            raise LedgerError("expected_binding is required for scoped writes")
        safe_scope = _normalize_scope(scope) if scope is not None else None
        safe_seen_at = _utc_datetime(seen_at, "seen_at") if seen_at is not None else None
        with self.connect() as conn:
            page = PgLedgerPage(self, conn, expected_binding=expected_binding)
            if safe_scope is not None:
                page.bind_scope(
                    safe_scope,
                    seen_at=safe_seen_at or datetime.now().astimezone(),
                    expected_binding=expected_binding,
                )
            yield page

    page_transaction = transaction

    def capture_binding(self, scope: LedgerScope, *, seen_at: datetime) -> LedgerBinding:
        """Capture the current binding fence before an external/network read."""
        safe_scope = _normalize_scope(scope)
        _utc_datetime(seen_at, "seen_at")
        key = scope_key(safe_scope)
        with self.connect() as conn, conn.cursor() as cur:
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

    def initialize_binding(self, scope: LedgerScope, *, seen_at: datetime) -> LedgerBinding:
        """Create the first binding for a collector, or refresh that same scope."""
        safe_scope = _normalize_scope(scope)
        safe_seen_at = _utc_datetime(seen_at, "seen_at")
        with self.transaction() as page:
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
    ) -> LedgerBinding:
        """Explicitly refine a fenced binding to a newly verified scope."""
        safe_scope = _normalize_scope(scope)
        safe_seen_at = _utc_datetime(seen_at, "seen_at")
        with self.transaction(expected_binding=expected_binding) as page:
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
            alias_values AS (
                SELECT
                    aliases.scope_key,
                    aliases.attempt_id,
                    string_agg(
                        aliases.alias_value,
                        ',' ORDER BY aliases.alias_value
                    ) FILTER (WHERE aliases.alias_kind = 'generation')
                        AS generation_aliases,
                    string_agg(
                        aliases.alias_value,
                        ',' ORDER BY aliases.alias_value
                    ) FILTER (WHERE aliases.alias_kind = 'message')
                        AS message_aliases,
                    string_agg(
                        aliases.alias_value,
                        ',' ORDER BY aliases.alias_value
                    ) FILTER (WHERE aliases.alias_kind = 'branch')
                        AS branch_aliases
                FROM public.chatgpt_usage_attempt_aliases AS aliases
                GROUP BY aliases.scope_key, aliases.attempt_id
            ),
            identified AS (
                SELECT
                    scoped.*,
                    CASE
                        WHEN alias_values.generation_aliases IS NOT NULL
                            THEN 'generation:' || alias_values.generation_aliases
                        WHEN alias_values.message_aliases IS NOT NULL
                            OR alias_values.branch_aliases IS NOT NULL
                            THEN 'strong:'
                                || COALESCE(alias_values.message_aliases, '')
                                || '|'
                                || COALESCE(alias_values.branch_aliases, '')
                        ELSE 'attempt:' || scoped.attempt_id
                    END AS identity_key
                FROM scoped
                LEFT JOIN alias_values
                  ON alias_values.scope_key = scoped.scope_key
                 AND alias_values.attempt_id = scoped.attempt_id
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
    ) -> None:
        self.ledger = ledger
        self.conn = conn
        self._bindings: dict[str, LedgerBinding] = {}
        self._expected_binding = expected_binding

    @staticmethod
    def assert_safe_record(value: Any) -> None:
        assert_no_secrets(value)

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
        with self.conn.cursor() as cur:
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
            if prior_identity_state == "provisional" and identity_state == "verified":
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
        with self.conn.cursor() as cur:
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
        binding = self.bind_scope(safe_scope, seen_at=safe_context.observed_at)
        sanitized = _observation_envelope(payload)
        self.ledger.assert_safe_record(sanitized)
        provenance = sanitize_provenance(safe_context.provenance)
        provenance.update(
            {
                "collector_account_id": safe_scope.collector_account_id,
                "schema_version": safe_context.schema_version,
                "source_id": safe_context.source_id,
                "source_kind": safe_context.source_kind,
            }
        )
        provenance.setdefault("transfer_schema_version", TRANSFER_SCHEMA_VERSION)
        self.ledger.assert_safe_record(provenance)
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
        with self.conn.cursor() as cur:
            _lock_scope(cur, binding.scope_key)
            prior = _latest_observation(cur, binding.scope_key, safe_context)
            current = _current_observation(cur, binding.scope_key, safe_context)
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
        binding = self.bind_scope(safe_scope, seen_at=safe_context.observed_at)
        observed_at = safe_context.observed_at
        with self.conn.cursor() as cur:
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
            self.ledger.assert_safe_record(attempt_payload)
            projection_fingerprint = fingerprint_value(attempt_payload)
            canonical_scope_key = canonical_ref.scope_key
            current = _attempt(cur, canonical_scope_key, effective_id)
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
                _finalize_identity_merges(
                    cur,
                    canonical_ref=canonical_ref,
                    merge_refs=resolution.merge_refs,
                    seen_at=observed_at,
                )
                merge_result = _merge_attempt_links(
                    self,
                    cur,
                    safe_scope,
                    scope_key_value=canonical_scope_key,
                    attempt_id=effective_id,
                    aliases=aliases,
                    seen_at=observed_at,
                    lineage_keys=lineage_keys,
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
            stale = current is not None and observed_at < current["observed_at"]
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
            else:
                cur.execute(
                    """
                    UPDATE public.chatgpt_usage_attempts
                    SET last_seen_at = GREATEST(last_seen_at, %s)
                    WHERE scope_key = %s AND attempt_id = %s
                    """,
                    (observed_at, canonical_scope_key, effective_id),
                )
                status = "quarantined" if quarantine_state != "clear" else "stale"

            if not stale:
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
                    not stale,
                ),
            )
            _finalize_identity_merges(
                cur,
                canonical_ref=canonical_ref,
                merge_refs=resolution.merge_refs,
                seen_at=observed_at,
            )
            merge_result = _merge_attempt_links(
                self,
                cur,
                safe_scope,
                scope_key_value=canonical_scope_key,
                attempt_id=effective_id,
                aliases=aliases,
                seen_at=observed_at,
                lineage_keys=lineage_keys,
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
        binding = self.bind_scope(safe_scope, seen_at=safe_seen_at)
        self.ledger.assert_safe_record(safe_details)
        gap_id = stable_id(binding.scope_key, safe_source_kind, safe_source_id, safe_reason)
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
        binding = self.bind_scope(safe_scope, seen_at=safe_seen_at)
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
        ("provider_user_id", previous[1], scope.provider_user_id),
        ("workspace_id", previous[2], scope.workspace_id),
        ("quota_owner_id", previous[3], scope.quota_owner_id),
        ("surface", previous[4], scope.surface),
    )
    for label, old_value, new_value in labels:
        if old_value is not None and new_value is not None and str(old_value) != str(new_value):
            raise LedgerError(f"scope identity component is incompatible: {label}")


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
        ) VALUES (%s, %s, 'verified_identity', %s, %s)
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


def _attempt(
    cur: psycopg.Cursor,
    scope_key_value: str,
    attempt_id: str,
) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT revision, projection_fingerprint, collector_account_id, tombstone,
               quarantine_state, identity_basis, observed_at, last_seen_at
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
    incoming_ref = AttemptRef(scope_key_value, attempt.attempt_id)
    lineage_keys = tuple(lineage_keys or _scope_lineage(cur, scope_key_value))
    strong_aliases = [alias for alias in aliases if alias[0] in _STRONG_ALIAS_KINDS]
    weak_aliases = [alias for alias in aliases if alias[0] not in _STRONG_ALIAS_KINDS]
    strong_matches = _matching_alias_refs(cur, lineage_keys, strong_aliases)
    if strong_matches:
        matches = strong_matches
    elif attempt.identity_basis == "generation":
        # Request/prompt aliases are grouping evidence once a generation is
        # known; reusing one must never collapse distinct generations.
        return AttemptIdentityResolution(incoming_ref)
    else:
        matches = _matching_alias_refs(cur, lineage_keys, weak_aliases)
    matches.discard(incoming_ref)
    if not matches:
        return AttemptIdentityResolution(incoming_ref)

    candidate_rows: dict[AttemptRef, dict[str, Any]] = {}
    for ref in matches:
        candidate = _attempt(cur, ref.scope_key, ref.attempt_id)
        if candidate is not None:
            candidate_rows[ref] = candidate
    if len(candidate_rows) != len(matches):
        _record_identity_gap(
            page,
            cur,
            scope,
            scope_key_value,
            source_id=incoming_ref.attempt_id,
            reason="alias_to_missing_attempt",
            details={
                "participants": [
                    _attempt_participant(incoming_ref, role="incoming"),
                    *[
                        _attempt_participant(ref, role="missing")
                        for ref in sorted(
                            matches - candidate_rows.keys(),
                            key=lambda item: (item.scope_key, item.attempt_id),
                        )[:16]
                    ],
                ]
            },
            seen_at=seen_at,
        )
        return AttemptIdentityResolution(
            incoming_ref,
            conflicts=len(matches),
            quarantine_reason="alias_to_missing_attempt",
        )

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
        and generations != incoming_generations
    }
    nonempty_generation_groups = {
        tuple(sorted(generations))
        for generations in candidate_generations.values()
        if generations
    }
    if incompatible or len(nonempty_generation_groups) > 1:
        _record_identity_gap(
            page,
            cur,
            scope,
            scope_key_value,
            source_id=incoming_ref.attempt_id,
            reason="ambiguous_attempt_alias",
            details={
                "participants": [
                    _attempt_participant(incoming_ref, role="incoming"),
                    *[
                        _attempt_participant(ref, role="candidate")
                        for ref in sorted(
                            matches,
                            key=lambda item: (item.scope_key, item.attempt_id),
                        )[:16]
                    ],
                ]
            },
            seen_at=seen_at,
        )
        return AttemptIdentityResolution(
            incoming_ref,
            conflicts=len(matches),
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
            source_id=incoming_ref.attempt_id,
            reason="alias_to_retired_attempt",
            details={
                "participants": [
                    _attempt_participant(incoming_ref, role="incoming"),
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
            matches - {canonical_ref},
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
    for alias_kind, alias_value, first_seen_at, last_seen_at in cur.fetchall():
        cur.execute(
            """
            INSERT INTO public.chatgpt_usage_attempt_aliases (
                scope_key, alias_kind, alias_value, attempt_id,
                first_seen_at, last_seen_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (scope_key, alias_kind, alias_value) DO UPDATE SET
                last_seen_at = GREATEST(
                    public.chatgpt_usage_attempt_aliases.last_seen_at,
                    EXCLUDED.last_seen_at
                )
            """,
            (
                canonical_ref.scope_key,
                alias_kind,
                alias_value,
                canonical_ref.attempt_id,
                first_seen_at,
                max(last_seen_at, seen_at),
            ),
        )
    cur.execute(
        """
        DELETE FROM public.chatgpt_usage_attempt_aliases
        WHERE scope_key = %s AND attempt_id = %s
        """,
        (previous_ref.scope_key, previous_ref.attempt_id),
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
        _tombstone_attempt(
            cur,
            previous_ref=previous_ref,
            canonical_ref=canonical_ref,
            seen_at=seen_at,
        )
        _reassign_attempt_aliases(
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
    page.ledger.assert_safe_record(safe_details)
    safe_seen_at = _utc_datetime(seen_at, "seen_at")
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
        json.dumps(warnings, separators=(",", ":"), default=str),
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
    for alias_kind, safe_value in safe_aliases:
        existing_refs = _matching_alias_refs(
            cur,
            lineage_keys,
            ((alias_kind, safe_value),),
        )
        existing_refs.discard(AttemptRef(scope_key_value, attempt_id))
        if existing_refs:
            existing_ref = sorted(
                existing_refs,
                key=lambda item: (item.scope_key, item.attempt_id),
            )[0]
            conflicts += 1
            quarantine_conflicts += 1 if alias_kind in _STRONG_ALIAS_KINDS else 0
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
            continue
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
    warnings = list(_safe_tokens(attempt.warnings))
    if quarantine_reason is not None:
        quarantine_warning = f"quarantine:{quarantine_reason}"
        if quarantine_warning not in warnings:
            warnings.append(quarantine_warning)
    warnings = sorted(warnings)
    quarantine = _quarantine_envelope(warnings)
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
        "aliases": sorted(aliases),
        "evidenceMessageIds": sorted(attempt.evidence_message_ids),
        "warnings": warnings,
        "quarantine": quarantine,
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


def _utc_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise LedgerError(f"{field_name} must be a datetime")
    return ensure_utc(value)


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
        value = projected.get(collection_key)
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
            projected[count_key] = min(len(value), _OBSERVATION_MAX_ITEMS)
    future_timestamp_quarantined = projected.get("future_timestamp_quarantined") is True
    if future_timestamp_quarantined:
        projected["quarantine_state"] = "quarantined"
        quarantine = projected.get("quarantine")
        if isinstance(quarantine, Mapping):
            reasons = _safe_tokens(quarantine.get("reasons", []), limit=32)
            if "future_timestamp_quarantined" not in reasons:
                reasons.append("future_timestamp_quarantined")
            projected["quarantine"] = {
                **quarantine,
                "transferSchemaVersion": TRANSFER_SCHEMA_VERSION,
                "state": "quarantined",
                "reasons": reasons[:32],
            }
        else:
            projected["quarantine"] = {
                "transferSchemaVersion": TRANSFER_SCHEMA_VERSION,
                "state": "quarantined",
                "reasons": ["future_timestamp_quarantined"],
            }
    return projected


def _project_observation_field(key: str, value: Any) -> Any:
    if key in {"surface", "origin"}:
        return _surface_token(value) if key == "surface" else _origin_token(value)
    if key in _OBSERVATION_TIMESTAMP_FIELDS:
        return _safe_timestamp(value)
    if key in _OBSERVATION_BOOLEAN_FIELDS:
        return value if isinstance(value, bool) else _DROP
    if key in _OBSERVATION_NUMBER_FIELDS:
        return _safe_number(value)
    if key in _OBSERVATION_COLLECTION_FIELDS:
        return _observation_collection_envelope(value)
    if key in _OBSERVATION_WARNING_FIELDS:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return _DROP
        return _safe_tokens(value, limit=64)
    if key == "provenance":
        return sanitize_provenance(value) if isinstance(value, Mapping) else _DROP
    if key == "quarantine_state":
        return _quarantine_state_token(value)
    if key == "transfer_version":
        return _transfer_version_token(value)
    if key in _OBSERVATION_LIST_FIELDS:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return _DROP
        return _safe_tokens(value, limit=800)
    if key in _OBSERVATION_TOKEN_FIELDS:
        return _optional_token(value)
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
    if key == "transfer_schema_version":
        return TRANSFER_SCHEMA_VERSION
    return _DROP


def _observation_collection_envelope(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: Any = sanitize_mapping(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sanitized = [sanitize_value(item) for item in value]
    else:
        return _DROP
    return _bound_observation_value(sanitized)


def _bound_observation_value(value: Any, *, depth: int = 0) -> Any:
    if depth > _OBSERVATION_MAX_DEPTH:
        return _DROP
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:_OBSERVATION_MAX_FIELDS]:
            child = _bound_observation_value(raw_value, depth=depth + 1)
            if child is not _DROP:
                out[str(raw_key)] = child
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out_list: list[Any] = []
        for raw_value in list(value)[:_OBSERVATION_MAX_ITEMS]:
            child = _bound_observation_value(raw_value, depth=depth + 1)
            if child is not _DROP:
                out_list.append(child)
        return out_list
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return _safe_number(value)
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


def _quarantine_envelope(
    warnings: Iterable[str],
    *,
    future_timestamp_quarantined: bool = False,
) -> dict[str, Any]:
    safe_warnings = _safe_tokens(warnings)
    reasons = [
        warning.split(":", 1)[1] for warning in safe_warnings if warning.startswith("quarantine:") and ":" in warning
    ]
    future_timestamp_quarantined = future_timestamp_quarantined or any(
        warning in {"future_timestamp_quarantined", "future_timestamp_quarantined=true"} for warning in safe_warnings
    )
    if future_timestamp_quarantined and "future_timestamp_quarantined" not in reasons:
        reasons.append("future_timestamp_quarantined")
    return {
        "transferSchemaVersion": TRANSFER_SCHEMA_VERSION,
        "state": "quarantined" if reasons or future_timestamp_quarantined else "clear",
        "reasons": sorted(set(reasons))[:32],
    }


def _quarantine_value(value: Any) -> Any:
    if isinstance(value, bool):
        return {
            "transferSchemaVersion": TRANSFER_SCHEMA_VERSION,
            "state": "quarantined" if value else "clear",
            "reasons": [],
        }
    if isinstance(value, str):
        token = _quarantine_state_token(value)
        return {
            "transferSchemaVersion": TRANSFER_SCHEMA_VERSION,
            "state": token or "unknown",
            "reasons": [],
        }
    if isinstance(value, Mapping):
        state = _quarantine_state_token(value.get("state")) or "unknown"
        raw_reasons = value.get("reasons", [])
        reasons = _safe_tokens(
            raw_reasons
            if isinstance(raw_reasons, Sequence) and not isinstance(raw_reasons, (str, bytes, bytearray))
            else []
        )
        if value.get("future_timestamp_quarantined") is True:
            state = "quarantined"
            if "future_timestamp_quarantined" not in reasons:
                reasons.append("future_timestamp_quarantined")
        return {
            "transferSchemaVersion": TRANSFER_SCHEMA_VERSION,
            "state": state,
            "reasons": sorted(set(reasons))[:32],
        }
    return _DROP


def _coverage_details_envelope(details: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if details is None:
        return {"transfer_schema_version": TRANSFER_SCHEMA_VERSION}
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
    return projected


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
