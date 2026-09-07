"""SQLite usage ledger for ChatGPT ordinary Chat observations and attempts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from .models import AttemptRecord, ConversationSummary, MessageRecord
from .privacy import SURFACE_CHAT, assert_no_secrets, evidence_identity
from .timeutil import isoformat_utc, parse_datetime

SCHEMA_VERSION = 3
LEDGER_APPLICATION = "chatgpt-chat-usage-capture"

DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    collector_account_id TEXT PRIMARY KEY,
    provider_user_id TEXT,
    workspace_id TEXT,
    quota_owner_id TEXT NOT NULL,
    surface TEXT NOT NULL,
    auth_state TEXT NOT NULL,
    plan_policy_id TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    profile_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collector_runs (
    run_id TEXT PRIMARY KEY,
    collector_account_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    scheduled_for TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    result TEXT,
    missed_intervals INTEGER NOT NULL DEFAULT 0,
    new_attempts INTEGER NOT NULL DEFAULT 0,
    updated_attempts INTEGER NOT NULL DEFAULT 0,
    deduplicated_attempts INTEGER NOT NULL DEFAULT 0,
    coverage TEXT,
    error_class TEXT,
    details_json TEXT,
    FOREIGN KEY (collector_account_id) REFERENCES accounts(collector_account_id)
);

CREATE TABLE IF NOT EXISTS discovery_state (
    collector_account_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    last_complete_discovery_started_at TEXT,
    last_complete_discovery_finished_at TEXT,
    continuation TEXT,
    watermark_at TEXT,
    coverage TEXT,
    PRIMARY KEY (collector_account_id, scope)
);

CREATE TABLE IF NOT EXISTS conversation_state (
    collector_account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    workspace_id TEXT,
    created_at TEXT,
    updated_at TEXT,
    is_archived INTEGER NOT NULL DEFAULT 0,
    surface TEXT NOT NULL,
    origin TEXT,
    page_coverage TEXT,
    pending INTEGER NOT NULL DEFAULT 0,
    last_seen_run_id TEXT,
    PRIMARY KEY (collector_account_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    collector_account_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    revision_fingerprint TEXT NOT NULL,
    surface TEXT NOT NULL,
    conversation_id TEXT,
    payload_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    run_id TEXT NOT NULL,
    UNIQUE (collector_account_id, source_kind, source_id, revision_fingerprint)
);

CREATE TABLE IF NOT EXISTS message_records (
    collector_account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    node_id TEXT,
    parent_id TEXT,
    children_json TEXT,
    role TEXT,
    channel TEXT,
    created_at TEXT,
    status TEXT,
    end_turn INTEGER,
    requested_model_raw TEXT,
    requested_mode_raw TEXT,
    requested_reasoning_effort_raw TEXT,
    recorded_final_model_raw TEXT,
    generation_id TEXT,
    request_id TEXT,
    surface TEXT NOT NULL,
    origin TEXT,
    metadata_json TEXT,
    PRIMARY KEY (collector_account_id, conversation_id, message_id)
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    collector_account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    identity_basis TEXT NOT NULL,
    time_basis TEXT NOT NULL,
    attempt_time TEXT,
    earliest_possible_at TEXT,
    latest_possible_at TEXT,
    requested_model_raw TEXT,
    requested_mode_raw TEXT,
    requested_reasoning_effort_raw TEXT,
    recorded_final_model_raw TEXT,
    resolved_model_raw TEXT,
    requested_family TEXT,
    recorded_final_family TEXT,
    resolved_family TEXT,
    mapping_version TEXT NOT NULL,
    outcome TEXT NOT NULL,
    completed_answer INTEGER NOT NULL,
    generation_started INTEGER NOT NULL,
    surface TEXT NOT NULL,
    origin TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    warnings_json TEXT,
    tombstone INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS attempt_mapping_history (
    mapping_history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    collector_account_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    mapping_version TEXT NOT NULL,
    requested_family TEXT,
    recorded_final_family TEXT,
    resolved_family TEXT,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL,
    UNIQUE (attempt_id, mapping_version),
    FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)
);

CREATE TABLE IF NOT EXISTS attempt_aliases (
    collector_account_id TEXT NOT NULL,
    alias_kind TEXT NOT NULL,
    alias_value TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    PRIMARY KEY (collector_account_id, alias_kind, alias_value),
    FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id)
);

CREATE TABLE IF NOT EXISTS attempt_evidence (
    attempt_id TEXT NOT NULL,
    evidence_kind TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    PRIMARY KEY (attempt_id, evidence_kind, evidence_id)
);

CREATE TABLE IF NOT EXISTS model_mapping_versions (
    version TEXT PRIMARY KEY,
    rules_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quota_policy_versions (
    policy_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    surface TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quota_window_definitions (
    collector_account_id TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    window_type TEXT NOT NULL,
    start_at TEXT,
    end_at TEXT,
    timezone TEXT,
    duration TEXT,
    evidence TEXT,
    reason TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (collector_account_id, bucket_id)
);

CREATE TABLE IF NOT EXISTS quota_observations (
    observation_id TEXT PRIMARY KEY,
    collector_account_id TEXT NOT NULL,
    bucket_id TEXT NOT NULL,
    source TEXT NOT NULL,
    remaining INTEGER,
    capacity INTEGER,
    observed_at TEXT NOT NULL,
    reset_at TEXT,
    window_start TEXT,
    raw_type TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS coverage_gaps (
    gap_id TEXT PRIMARY KEY,
    collector_account_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    reason TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS aggregate_revisions (
    revision_id TEXT PRIMARY KEY,
    collector_account_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    policy_id TEXT,
    mapping_version TEXT,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scheduler_state (
    collector_account_id TEXT PRIMARY KEY,
    refresh_interval TEXT NOT NULL,
    next_due_at TEXT,
    last_started_at TEXT,
    last_finished_at TEXT,
    lease_owner TEXT,
    lease_token TEXT,
    lease_until TEXT,
    missed_intervals INTEGER NOT NULL DEFAULT 0,
    backoff_until TEXT,
    pending_work_json TEXT,
    schedule_anchor_at TEXT,
    last_jitter_seconds INTEGER,
    fencing_generation INTEGER NOT NULL DEFAULT 0,
    last_heartbeat_at TEXT
);

CREATE TABLE IF NOT EXISTS daily_aggregates (
    collector_account_id TEXT NOT NULL,
    local_date TEXT NOT NULL,
    timezone TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revision_id TEXT,
    PRIMARY KEY (collector_account_id, local_date, timezone)
);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id TEXT PRIMARY KEY,
    collector_account_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (collector_account_id, idempotency_key)
);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        with self.transaction():
            self.conn.executescript(DDL)
            self._ensure_column("scheduler_state", "schedule_anchor_at", "TEXT")
            self._ensure_column("scheduler_state", "last_jitter_seconds", "INTEGER")
            self._ensure_column("scheduler_state", "fencing_generation", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("scheduler_state", "last_heartbeat_at", "TEXT")
            row = self.conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
            ).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, isoformat_utc(datetime.now().astimezone()) or ""),
                )
            elif int(row["version"]) < SCHEMA_VERSION:
                self.conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, isoformat_utc(datetime.now().astimezone()) or ""),
                )

    def _ensure_column(self, table: str, column: str, decl: str) -> None:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        names = {row[1] for row in rows}
        if column not in names:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        in_transaction = self.conn.in_transaction
        if not in_transaction:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        else:
            savepoint = f"sp_{id(self):x}"
            self.conn.execute(f"SAVEPOINT {savepoint}")
            try:
                yield self.conn
                self.conn.execute(f"RELEASE {savepoint}")
            except Exception:
                self.conn.execute(f"ROLLBACK TO {savepoint}")
                raise

    def upsert_account(self, account: Mapping[str, Any]) -> None:
        now = isoformat_utc(datetime.now().astimezone())
        surface = account.get("surface") or SURFACE_CHAT
        if surface != SURFACE_CHAT:
            raise ValueError("ledger accounts for this collector must remain surface=chat")
        self.conn.execute(
            """
            INSERT INTO accounts (
                collector_account_id, provider_user_id, workspace_id, quota_owner_id,
                surface, auth_state, plan_policy_id, enabled, profile_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collector_account_id) DO UPDATE SET
                provider_user_id=excluded.provider_user_id,
                workspace_id=excluded.workspace_id,
                quota_owner_id=excluded.quota_owner_id,
                surface=excluded.surface,
                auth_state=excluded.auth_state,
                plan_policy_id=excluded.plan_policy_id,
                enabled=excluded.enabled,
                profile_path=excluded.profile_path,
                updated_at=excluded.updated_at
            """,
            (
                account["collector_account_id"],
                account.get("provider_user_id"),
                account.get("workspace_id"),
                account["quota_owner_id"],
                SURFACE_CHAT,
                account.get("auth_state") or "unconfigured",
                account.get("plan_policy_id"),
                1 if account.get("enabled", True) else 0,
                account.get("profile_path"),
                now,
                now,
            ),
        )

    def record_run_start(
        self,
        *,
        run_id: str,
        account_id: str,
        mode: str,
        started_at: datetime,
        scheduled_for: Optional[datetime] = None,
        missed_intervals: int = 0,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO collector_runs (
                run_id, collector_account_id, mode, scheduled_for, started_at, missed_intervals
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                account_id,
                mode,
                isoformat_utc(scheduled_for),
                isoformat_utc(started_at),
                missed_intervals,
            ),
        )

    def finish_run(
        self,
        run_id: str,
        *,
        ended_at: datetime,
        result: str,
        new_attempts: int = 0,
        updated_attempts: int = 0,
        deduplicated_attempts: int = 0,
        coverage: str | None = None,
        error_class: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE collector_runs
            SET ended_at=?, result=?, new_attempts=?, updated_attempts=?,
                deduplicated_attempts=?, coverage=?, error_class=?, details_json=?
            WHERE run_id=?
            """,
            (
                isoformat_utc(ended_at),
                result,
                new_attempts,
                updated_attempts,
                deduplicated_attempts,
                coverage,
                error_class,
                json.dumps(details or {}, separators=(",", ":")),
                run_id,
            ),
        )

    def get_discovery_state(self, account_id: str, scope: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM discovery_state WHERE collector_account_id=? AND scope=?",
            (account_id, scope),
        ).fetchone()
        return dict(row) if row else None

    def upsert_discovery_state(
        self,
        *,
        account_id: str,
        scope: str,
        last_complete_discovery_started_at: Optional[datetime] = None,
        last_complete_discovery_finished_at: Optional[datetime] = None,
        continuation: Optional[str] = None,
        watermark_at: Optional[datetime] = None,
        coverage: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO discovery_state (
                collector_account_id, scope, last_complete_discovery_started_at,
                last_complete_discovery_finished_at, continuation, watermark_at, coverage
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collector_account_id, scope) DO UPDATE SET
                last_complete_discovery_started_at=COALESCE(excluded.last_complete_discovery_started_at, discovery_state.last_complete_discovery_started_at),
                last_complete_discovery_finished_at=COALESCE(excluded.last_complete_discovery_finished_at, discovery_state.last_complete_discovery_finished_at),
                continuation=excluded.continuation,
                watermark_at=COALESCE(excluded.watermark_at, discovery_state.watermark_at),
                coverage=COALESCE(excluded.coverage, discovery_state.coverage)
            """,
            (
                account_id,
                scope,
                isoformat_utc(last_complete_discovery_started_at),
                isoformat_utc(last_complete_discovery_finished_at),
                continuation,
                isoformat_utc(watermark_at),
                coverage,
            ),
        )

    def upsert_conversation(self, account_id: str, summary: ConversationSummary, run_id: str) -> None:
        self.conn.execute(
            """
            INSERT INTO conversation_state (
                collector_account_id, conversation_id, workspace_id, created_at, updated_at,
                is_archived, surface, origin, page_coverage, pending, last_seen_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collector_account_id, conversation_id) DO UPDATE SET
                workspace_id=COALESCE(excluded.workspace_id, conversation_state.workspace_id),
                created_at=COALESCE(excluded.created_at, conversation_state.created_at),
                updated_at=excluded.updated_at,
                is_archived=excluded.is_archived,
                surface=excluded.surface,
                origin=COALESCE(excluded.origin, conversation_state.origin),
                last_seen_run_id=excluded.last_seen_run_id
            """,
            (
                account_id,
                summary.conversation_id,
                summary.workspace_id,
                isoformat_utc(summary.created_at),
                isoformat_utc(summary.updated_at),
                1 if summary.is_archived else 0,
                summary.surface,
                summary.origin,
                summary.coverage,
                1,
                run_id,
            ),
        )

    def mark_conversation_fetched(self, account_id: str, conversation_id: str, coverage: str) -> None:
        self.conn.execute(
            """
            UPDATE conversation_state
            SET pending=0, page_coverage=?
            WHERE collector_account_id=? AND conversation_id=?
            """,
            (coverage, account_id, conversation_id),
        )

    def pending_conversations(self, account_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM conversation_state
            WHERE collector_account_id=? AND pending=1
            ORDER BY updated_at DESC
            """,
            (account_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def insert_observation(
        self,
        *,
        account_id: str,
        source_kind: str,
        source_id: str,
        revision: str,
        surface: str,
        conversation_id: Optional[str],
        payload: Mapping[str, Any],
        observed_at: datetime,
        run_id: str,
    ) -> bool:
        assert_no_secrets(payload)
        observation_id = evidence_identity(source_kind, source_id, revision)
        try:
            self.conn.execute(
                """
                INSERT INTO observations (
                    observation_id, collector_account_id, source_kind, source_id,
                    revision_fingerprint, surface, conversation_id, payload_json,
                    observed_at, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    account_id,
                    source_kind,
                    source_id,
                    revision,
                    surface,
                    conversation_id,
                    json.dumps(payload, separators=(",", ":"), default=str),
                    isoformat_utc(observed_at),
                    run_id,
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def upsert_message(self, account_id: str, record: MessageRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO message_records (
                collector_account_id, conversation_id, message_id, node_id, parent_id,
                children_json, role, channel, created_at, status, end_turn,
                requested_model_raw, requested_mode_raw, requested_reasoning_effort_raw,
                recorded_final_model_raw, generation_id, request_id, surface, origin,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collector_account_id, conversation_id, message_id) DO UPDATE SET
                node_id=excluded.node_id,
                parent_id=excluded.parent_id,
                children_json=excluded.children_json,
                role=excluded.role,
                channel=excluded.channel,
                created_at=excluded.created_at,
                status=excluded.status,
                end_turn=excluded.end_turn,
                requested_model_raw=excluded.requested_model_raw,
                requested_mode_raw=excluded.requested_mode_raw,
                requested_reasoning_effort_raw=excluded.requested_reasoning_effort_raw,
                recorded_final_model_raw=excluded.recorded_final_model_raw,
                generation_id=excluded.generation_id,
                request_id=excluded.request_id,
                surface=excluded.surface,
                origin=excluded.origin,
                metadata_json=excluded.metadata_json
            """,
            (
                account_id,
                record.conversation_id,
                record.message_id,
                record.node_id,
                record.parent_id,
                json.dumps(list(record.children)),
                record.role,
                record.channel,
                isoformat_utc(record.created_at),
                record.status,
                None if record.end_turn is None else int(record.end_turn),
                record.requested_model_raw,
                record.requested_mode_raw,
                record.requested_reasoning_effort_raw,
                record.recorded_final_model_raw,
                record.generation_id,
                record.request_id,
                record.surface,
                record.origin,
                json.dumps(record.metadata, separators=(",", ":"), default=str),
            ),
        )

    def messages_for(self, account_id: str, conversation_id: str) -> list[MessageRecord]:
        rows = self.conn.execute(
            """
            SELECT * FROM message_records
            WHERE collector_account_id=? AND conversation_id=?
            """,
            (account_id, conversation_id),
        ).fetchall()
        records: list[MessageRecord] = []
        for row in rows:
            records.append(
                MessageRecord(
                    conversation_id=row["conversation_id"],
                    message_id=row["message_id"],
                    node_id=row["node_id"],
                    parent_id=row["parent_id"],
                    children=tuple(json.loads(row["children_json"] or "[]")),
                    role=row["role"],
                    channel=row["channel"],
                    created_at=parse_datetime(row["created_at"]),
                    status=row["status"],
                    end_turn=None if row["end_turn"] is None else bool(row["end_turn"]),
                    requested_model_raw=row["requested_model_raw"],
                    requested_mode_raw=row["requested_mode_raw"],
                    requested_reasoning_effort_raw=row["requested_reasoning_effort_raw"],
                    recorded_final_model_raw=row["recorded_final_model_raw"],
                    generation_id=row["generation_id"],
                    request_id=row["request_id"],
                    surface=row["surface"],
                    origin=row["origin"],
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
            )
        return records

    def upsert_attempt(self, account_id: str, attempt: AttemptRecord) -> str:
        existing = self._existing_attempt(account_id, attempt)
        if existing is not None:
            self.conn.execute(
                """
                UPDATE attempts SET
                    identity_basis=?, time_basis=?, attempt_time=?, earliest_possible_at=?,
                    latest_possible_at=?, requested_model_raw=?, requested_mode_raw=?,
                    requested_reasoning_effort_raw=?, recorded_final_model_raw=?,
                    resolved_model_raw=?, requested_family=?, recorded_final_family=?,
                    resolved_family=?, mapping_version=?, outcome=?, completed_answer=?,
                    generation_started=?, surface=?, origin=?, revision=revision+1,
                    warnings_json=?
                WHERE attempt_id=?
                """,
                (
                    attempt.identity_basis,
                    attempt.time_basis,
                    isoformat_utc(attempt.attempt_time),
                    isoformat_utc(attempt.earliest_possible_at),
                    isoformat_utc(attempt.latest_possible_at),
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
                    int(attempt.completed_answer),
                    int(attempt.generation_started),
                    attempt.surface,
                    attempt.origin,
                    json.dumps(list(attempt.warnings)),
                    existing,
                ),
            )
            attempt_id = existing
            status = "deduplicated"
        else:
            self.conn.execute(
                """
                INSERT INTO attempts (
                    attempt_id, collector_account_id, conversation_id, identity_basis,
                    time_basis, attempt_time, earliest_possible_at, latest_possible_at,
                    requested_model_raw, requested_mode_raw, requested_reasoning_effort_raw,
                    recorded_final_model_raw, resolved_model_raw, requested_family,
                    recorded_final_family, resolved_family, mapping_version, outcome,
                    completed_answer, generation_started, surface, origin, revision,
                    warnings_json, tombstone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    attempt.attempt_id,
                    account_id,
                    attempt.conversation_id,
                    attempt.identity_basis,
                    attempt.time_basis,
                    isoformat_utc(attempt.attempt_time),
                    isoformat_utc(attempt.earliest_possible_at),
                    isoformat_utc(attempt.latest_possible_at),
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
                    int(attempt.completed_answer),
                    int(attempt.generation_started),
                    attempt.surface,
                    attempt.origin,
                    attempt.revision,
                    json.dumps(list(attempt.warnings)),
                ),
            )
            attempt_id = attempt.attempt_id
            status = "inserted"
        for kind, value in attempt.aliases:
            self.conn.execute(
                """
                INSERT INTO attempt_aliases (collector_account_id, alias_kind, alias_value, attempt_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(collector_account_id, alias_kind, alias_value) DO NOTHING
                """,
                (account_id, kind, value, attempt_id),
            )
        for message_id in attempt.evidence_message_ids:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO attempt_evidence (attempt_id, evidence_kind, evidence_id)
                VALUES (?, 'message', ?)
                """,
                (attempt_id, message_id),
            )
        return status

    def _existing_attempt(self, account_id: str, attempt: AttemptRecord) -> Optional[str]:
        row = self.conn.execute(
            "SELECT attempt_id FROM attempts WHERE attempt_id=?",
            (attempt.attempt_id,),
        ).fetchone()
        if row:
            return row["attempt_id"]

        generation_aliases = [
            (kind, value) for kind, value in attempt.aliases if kind == "generation"
        ]
        if generation_aliases:
            # A request ID is grouping evidence only. Once a generation ID is
            # present, an exact generation match wins. A request alias may
            # still upgrade a provisional attempt that has no generation ID.
            for kind, value in generation_aliases:
                row = self.conn.execute(
                    """
                    SELECT attempt_id FROM attempt_aliases
                    WHERE collector_account_id=? AND alias_kind=? AND alias_value=?
                    """,
                    (account_id, kind, value),
                ).fetchone()
                if row:
                    return row["attempt_id"]
            for kind, value in attempt.aliases:
                if kind == "generation":
                    continue
                row = self.conn.execute(
                    """
                    SELECT attempt_id FROM attempt_aliases
                    WHERE collector_account_id=? AND alias_kind=? AND alias_value=?
                    """,
                    (account_id, kind, value),
                ).fetchone()
                if row:
                    generation = self.conn.execute(
                        """
                        SELECT 1 FROM attempt_aliases
                        WHERE collector_account_id=? AND alias_kind='generation' AND attempt_id=?
                        LIMIT 1
                        """,
                        (account_id, row["attempt_id"]),
                    ).fetchone()
                    if generation is None:
                        return row["attempt_id"]
            return None

        for kind, value in attempt.aliases:
            row = self.conn.execute(
                """
                SELECT attempt_id FROM attempt_aliases
                WHERE collector_account_id=? AND alias_kind=? AND alias_value=?
                """,
                (account_id, kind, value),
            ).fetchone()
            if row:
                generation = self.conn.execute(
                    """
                    SELECT 1 FROM attempt_aliases
                    WHERE collector_account_id=? AND alias_kind='generation' AND attempt_id=?
                    LIMIT 1
                    """,
                    (account_id, row["attempt_id"]),
                ).fetchone()
                if generation is None:
                    return row["attempt_id"]
        return None

    def reclassify_attempt_mappings(
        self,
        account_id: str,
        projections: Sequence[Mapping[str, Any]],
        *,
        recorded_at: datetime,
        source: str = "aggregate_rebuild",
    ) -> int:
        """Persist current mapping projections while retaining prior versions."""
        changed = 0
        recorded_stamp = isoformat_utc(recorded_at) or ""
        for projection in projections:
            attempt_id = str(projection.get("attempt_id") or "")
            if not attempt_id:
                continue
            row = self.conn.execute(
                """
                SELECT mapping_version, requested_family, recorded_final_family, resolved_family
                FROM attempts
                WHERE collector_account_id=? AND attempt_id=?
                """,
                (account_id, attempt_id),
            ).fetchone()
            if row is None:
                continue
            next_values = (
                projection.get("mapping_version"),
                projection.get("requested_family"),
                projection.get("recorded_final_family"),
                projection.get("resolved_family"),
            )
            current_values = (
                row["mapping_version"],
                row["requested_family"],
                row["recorded_final_family"],
                row["resolved_family"],
            )
            if next_values == current_values:
                continue
            self._record_mapping_history(
                account_id=account_id,
                attempt_id=attempt_id,
                mapping_version=str(row["mapping_version"]),
                requested_family=row["requested_family"],
                recorded_final_family=row["recorded_final_family"],
                resolved_family=row["resolved_family"],
                recorded_at=recorded_stamp,
                source="prior_projection",
            )
            self._record_mapping_history(
                account_id=account_id,
                attempt_id=attempt_id,
                mapping_version=str(projection.get("mapping_version") or ""),
                requested_family=projection.get("requested_family"),
                recorded_final_family=projection.get("recorded_final_family"),
                resolved_family=projection.get("resolved_family"),
                recorded_at=recorded_stamp,
                source=source,
            )
            self.conn.execute(
                """
                UPDATE attempts
                SET requested_family=?, recorded_final_family=?, resolved_family=?,
                    mapping_version=?, revision=revision+1
                WHERE collector_account_id=? AND attempt_id=?
                """,
                (
                    projection.get("requested_family"),
                    projection.get("recorded_final_family"),
                    projection.get("resolved_family"),
                    projection.get("mapping_version"),
                    account_id,
                    attempt_id,
                ),
            )
            changed += 1
        return changed

    def _record_mapping_history(
        self,
        *,
        account_id: str,
        attempt_id: str,
        mapping_version: str,
        requested_family: Any,
        recorded_final_family: Any,
        resolved_family: Any,
        recorded_at: str,
        source: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO attempt_mapping_history (
                collector_account_id, attempt_id, mapping_version,
                requested_family, recorded_final_family, resolved_family,
                recorded_at, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(attempt_id, mapping_version) DO NOTHING
            """,
            (
                account_id,
                attempt_id,
                mapping_version,
                requested_family,
                recorded_final_family,
                resolved_family,
                recorded_at,
                source,
            ),
        )

    def list_attempt_mapping_history(
        self, account_id: str, attempt_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT * FROM attempt_mapping_history
            WHERE collector_account_id=?
        """
        params: list[Any] = [account_id]
        if attempt_id is not None:
            sql += " AND attempt_id=?"
            params.append(attempt_id)
        sql += " ORDER BY mapping_history_id"
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def list_attempts(
        self,
        account_id: str,
        *,
        include_tombstones: bool = False,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT * FROM attempts
            WHERE collector_account_id=?
        """
        params: list[Any] = [account_id]
        if not include_tombstones:
            sql += " AND tombstone=0"
        if cursor:
            sql += " AND attempt_id > ?"
            params.append(cursor)
        sql += " ORDER BY COALESCE(attempt_time, earliest_possible_at), attempt_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def set_window(
        self,
        account_id: str,
        bucket_id: str,
        *,
        window_type: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timezone_name: Optional[str] = None,
        duration: Optional[str] = None,
        evidence: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        now = isoformat_utc(datetime.now().astimezone())
        self.conn.execute(
            """
            INSERT INTO quota_window_definitions (
                collector_account_id, bucket_id, window_type, start_at, end_at,
                timezone, duration, evidence, reason, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collector_account_id, bucket_id) DO UPDATE SET
                window_type=excluded.window_type,
                start_at=excluded.start_at,
                end_at=excluded.end_at,
                timezone=excluded.timezone,
                duration=excluded.duration,
                evidence=excluded.evidence,
                reason=excluded.reason,
                updated_at=excluded.updated_at
            """,
            (
                account_id,
                bucket_id,
                window_type,
                isoformat_utc(start),
                isoformat_utc(end),
                timezone_name,
                duration,
                evidence,
                reason,
                now,
            ),
        )

    def get_window(self, account_id: str, bucket_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT * FROM quota_window_definitions
            WHERE collector_account_id=? AND bucket_id=?
            """,
            (account_id, bucket_id),
        ).fetchone()
        return dict(row) if row else None

    def record_quota_observation(
        self,
        *,
        observation_id: str,
        account_id: str,
        bucket_id: str,
        source: str,
        remaining: Optional[int],
        capacity: Optional[int],
        observed_at: datetime,
        reset_at: Optional[datetime] = None,
        window_start: Optional[datetime] = None,
        raw_type: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO quota_observations (
                observation_id, collector_account_id, bucket_id, source, remaining,
                capacity, observed_at, reset_at, window_start, raw_type, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                account_id,
                bucket_id,
                source,
                remaining,
                capacity,
                isoformat_utc(observed_at),
                isoformat_utc(reset_at),
                isoformat_utc(window_start),
                raw_type,
                notes,
            ),
        )

    def latest_quota_observation(self, account_id: str, bucket_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT * FROM quota_observations
            WHERE collector_account_id=? AND bucket_id=?
            ORDER BY observed_at DESC LIMIT 1
            """,
            (account_id, bucket_id),
        ).fetchone()
        return dict(row) if row else None

    def get_scheduler_state(self, account_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM scheduler_state WHERE collector_account_id=?",
            (account_id,),
        ).fetchone()
        return dict(row) if row else None

    def upsert_scheduler_state(self, account_id: str, **fields: Any) -> None:
        current = self.get_scheduler_state(account_id) or {
            "collector_account_id": account_id,
            "refresh_interval": "PT1H",
            "next_due_at": None,
            "last_started_at": None,
            "last_finished_at": None,
            "lease_owner": None,
            "lease_token": None,
            "lease_until": None,
            "missed_intervals": 0,
            "backoff_until": None,
            "pending_work_json": None,
            "schedule_anchor_at": None,
            "last_jitter_seconds": None,
            "fencing_generation": 0,
            "last_heartbeat_at": None,
        }
        current.update(fields)
        self.conn.execute(
            """
            INSERT INTO scheduler_state (
                collector_account_id, refresh_interval, next_due_at, last_started_at,
                last_finished_at, lease_owner, lease_token, lease_until, missed_intervals,
                backoff_until, pending_work_json, schedule_anchor_at, last_jitter_seconds,
                fencing_generation, last_heartbeat_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(collector_account_id) DO UPDATE SET
                refresh_interval=excluded.refresh_interval,
                next_due_at=excluded.next_due_at,
                last_started_at=excluded.last_started_at,
                last_finished_at=excluded.last_finished_at,
                lease_owner=excluded.lease_owner,
                lease_token=excluded.lease_token,
                lease_until=excluded.lease_until,
                missed_intervals=excluded.missed_intervals,
                backoff_until=excluded.backoff_until,
                pending_work_json=excluded.pending_work_json,
                schedule_anchor_at=excluded.schedule_anchor_at,
                last_jitter_seconds=excluded.last_jitter_seconds,
                fencing_generation=excluded.fencing_generation,
                last_heartbeat_at=excluded.last_heartbeat_at
            """,
            (
                account_id,
                current.get("refresh_interval") or "PT1H",
                current.get("next_due_at"),
                current.get("last_started_at"),
                current.get("last_finished_at"),
                current.get("lease_owner"),
                current.get("lease_token"),
                current.get("lease_until"),
                int(current.get("missed_intervals") or 0),
                current.get("backoff_until"),
                current.get("pending_work_json"),
                current.get("schedule_anchor_at"),
                current.get("last_jitter_seconds"),
                int(current.get("fencing_generation") or 0),
                current.get("last_heartbeat_at"),
            ),
        )

    def record_gap(self, gap_id: str, account_id: str, scope: str, reason: str, now: datetime) -> None:
        stamp = isoformat_utc(now)
        self.conn.execute(
            """
            INSERT INTO coverage_gaps (
                gap_id, collector_account_id, scope, reason, first_seen_at, last_seen_at, state
            ) VALUES (?, ?, ?, ?, ?, ?, 'open')
            ON CONFLICT(gap_id) DO UPDATE SET
                last_seen_at=excluded.last_seen_at
            """,
            (gap_id, account_id, scope, reason, stamp, stamp),
        )

    def compare_and_set_lease(
        self,
        account_id: str,
        *,
        expected_token: Optional[str],
        owner: str,
        token: str,
        lease_until: datetime,
        heartbeat_at: Optional[datetime] = None,
        generation: Optional[int] = None,
    ) -> bool:
        """Atomically claim or re-fence a lease. Stale tokens must not overwrite newer state."""
        now = heartbeat_at or lease_until
        current = self.get_scheduler_state(account_id)
        if current is None:
            self.upsert_scheduler_state(
                account_id,
                lease_owner=owner,
                lease_token=token,
                lease_until=isoformat_utc(lease_until),
                last_heartbeat_at=isoformat_utc(now),
                fencing_generation=int(generation or 1),
            )
            return True
        stored_token = current.get("lease_token")
        if expected_token is None:
            if stored_token not in (None, ""):
                return False
        elif stored_token != expected_token:
            return False
        next_generation = int(generation if generation is not None else (current.get("fencing_generation") or 0) + 1)
        cursor = self.conn.execute(
            """
            UPDATE scheduler_state
            SET lease_owner=?, lease_token=?, lease_until=?, last_heartbeat_at=?, fencing_generation=?
            WHERE collector_account_id=? AND ((? IS NULL AND (lease_token IS NULL OR lease_token='')) OR lease_token=?)
            """,
            (
                owner,
                token,
                isoformat_utc(lease_until),
                isoformat_utc(now),
                next_generation,
                account_id,
                expected_token,
                expected_token,
            ),
        )
        return cursor.rowcount == 1

    def heartbeat_lease(self, account_id: str, token: str, *, lease_until: datetime, heartbeat_at: datetime) -> bool:
        cursor = self.conn.execute(
            """
            UPDATE scheduler_state
            SET lease_until=?, last_heartbeat_at=?
            WHERE collector_account_id=? AND lease_token=?
            """,
            (isoformat_utc(lease_until), isoformat_utc(heartbeat_at), account_id, token),
        )
        return cursor.rowcount == 1

    def release_lease(self, account_id: str, token: str) -> bool:
        cursor = self.conn.execute(
            """
            UPDATE scheduler_state
            SET lease_owner=NULL, lease_token=NULL, lease_until=NULL
            WHERE collector_account_id=? AND lease_token=?
            """,
            (account_id, token),
        )
        return cursor.rowcount == 1

    def pending_work(self, account_id: str) -> list[dict[str, Any]]:
        state = self.get_scheduler_state(account_id) or {}
        raw = state.get("pending_work_json")
        if not raw:
            return []
        payload = json.loads(raw)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

    def enqueue_pending_work(self, account_id: str, item: Mapping[str, Any]) -> list[dict[str, Any]]:
        queued = self.pending_work(account_id)
        key = (item.get("kind"), item.get("mode"), item.get("since"))
        if not any((existing.get("kind"), existing.get("mode"), existing.get("since")) == key for existing in queued):
            queued.append(dict(item))
        self.upsert_scheduler_state(account_id, pending_work_json=json.dumps(queued, separators=(",", ":")))
        return queued

    def clear_pending_work(self, account_id: str) -> None:
        self.upsert_scheduler_state(account_id, pending_work_json=None)

    def list_accounts(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM accounts ORDER BY collector_account_id").fetchall()
        return [dict(row) for row in rows]

    def list_windows(self, account_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM quota_window_definitions
            WHERE collector_account_id=?
            ORDER BY bucket_id
            """,
            (account_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_quota_observations(
        self,
        account_id: str,
        *,
        bucket_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM quota_observations WHERE collector_account_id=?"
        params: list[Any] = [account_id]
        if bucket_id:
            sql += " AND bucket_id=?"
            params.append(bucket_id)
        sql += " ORDER BY observed_at DESC LIMIT ?"
        params.append(int(limit))
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def list_runs(self, account_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM collector_runs
            WHERE collector_account_id=?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (account_id, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_coverage_gaps(self, account_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM coverage_gaps
            WHERE collector_account_id=?
            ORDER BY first_seen_at
            """,
            (account_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def insert_aggregate_revision(
        self,
        *,
        revision_id: str,
        account_id: str,
        created_at: datetime,
        policy_id: Optional[str],
        mapping_version: Optional[str],
        payload: Mapping[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO aggregate_revisions (
                revision_id, collector_account_id, created_at, policy_id, mapping_version, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                account_id,
                isoformat_utc(created_at),
                policy_id,
                mapping_version,
                json.dumps(payload, separators=(",", ":"), default=str),
            ),
        )

    def get_latest_revision(self, account_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT * FROM aggregate_revisions
            WHERE collector_account_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (account_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item.get("payload_json") or "{}")
        return item

    def upsert_daily_aggregate(
        self,
        *,
        account_id: str,
        local_date: str,
        timezone_name: str,
        payload: Mapping[str, Any],
        created_at: datetime,
        revision_id: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO daily_aggregates (
                collector_account_id, local_date, timezone, payload_json, created_at, revision_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(collector_account_id, local_date, timezone) DO UPDATE SET
                payload_json=excluded.payload_json,
                created_at=excluded.created_at,
                revision_id=excluded.revision_id
            """,
            (
                account_id,
                local_date,
                timezone_name,
                json.dumps(payload, separators=(",", ":"), default=str),
                isoformat_utc(created_at),
                revision_id,
            ),
        )

    def list_daily_aggregates(self, account_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM daily_aggregates
            WHERE collector_account_id=?
            ORDER BY local_date
            """,
            (account_id,),
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.get("payload_json") or "{}")
            items.append(item)
        return items

    def prune_retention(
        self,
        account_id: str,
        *,
        now: datetime,
        observation_days: int,
        attempt_days: int,
        daily_aggregate_days: int,
        preserve_active_window_evidence: bool = True,
    ) -> dict[str, Any]:
        """Prune expired observations/attempts while keeping tombstones and aliases."""
        from datetime import timedelta, timezone
        now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        observation_cutoff = isoformat_utc(now_utc - timedelta(days=observation_days))
        attempt_cutoff = isoformat_utc(now_utc - timedelta(days=attempt_days))
        aggregate_cutoff_date = (now_utc.date() - timedelta(days=daily_aggregate_days)).isoformat()
        warnings: list[str] = []
        protected_starts: list[str] = []
        if preserve_active_window_evidence:
            for window in self.list_windows(account_id):
                if window.get("start_at"):
                    protected_starts.append(window["start_at"])
        deleted_observations = self.conn.execute(
            """
            DELETE FROM observations
            WHERE collector_account_id=? AND observed_at < ?
            """,
            (account_id, observation_cutoff),
        ).rowcount
        deleted_quota = self.conn.execute(
            """
            DELETE FROM quota_observations
            WHERE collector_account_id=? AND observed_at < ?
            """,
            (account_id, observation_cutoff),
        ).rowcount
        expired_attempts = self.conn.execute(
            """
            SELECT attempt_id FROM attempts
            WHERE collector_account_id=? AND tombstone=0
              AND COALESCE(attempt_time, earliest_possible_at, '') < ?
            """,
            (account_id, attempt_cutoff),
        ).fetchall()
        tombstoned = 0
        for row in expired_attempts:
            self.conn.execute(
                "UPDATE attempts SET tombstone=1 WHERE attempt_id=?",
                (row["attempt_id"],),
            )
            tombstoned += 1
        deleted_aggregates = self.conn.execute(
            """
            DELETE FROM daily_aggregates
            WHERE collector_account_id=? AND local_date < ?
            """,
            (account_id, aggregate_cutoff_date),
        ).rowcount
        remaining_observations = self.conn.execute(
            "SELECT COUNT(*) AS n FROM observations WHERE collector_account_id=?",
            (account_id,),
        ).fetchone()["n"]
        if remaining_observations == 0 and (deleted_observations or tombstoned):
            warnings.append("raw observations pruned; rebuild from original website payloads is no longer possible")
        if protected_starts:
            earliest_protected = min(protected_starts)
            if observation_cutoff is not None and observation_cutoff > earliest_protected:
                warnings.append("retention cutoff is later than an active window start; rebuildability of that window is reduced")
        return {
            "deleted_observations": deleted_observations,
            "deleted_quota_observations": deleted_quota,
            "tombstoned_attempts": tombstoned,
            "deleted_daily_aggregates": deleted_aggregates,
            "aliases_preserved": True,
            "warnings": warnings,
        }
