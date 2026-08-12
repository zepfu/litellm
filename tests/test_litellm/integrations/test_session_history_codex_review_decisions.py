"""Focused D1-616 schema and record-behavior tests for Codex review decisions.

Covers the standalone `session_history_codex_review_decisions` persistence
body in `litellm/integrations/aawm_session_history`:

- schema shape (stable identities, outcome CHECK, indexes, idempotent insert
  with immutable identity enrichment on conflict)
- record builder behavior: explicit stable decision/attempt identity
  requirement, outcome-change idempotency, concurrent-parent distinction,
  canonical-session + parent-actor/thread attribution (D1-615 parity),
  sanitization (512-char rationale, ANSI/control cleanup, whitespace
  collapse, secret redaction) and bounded labels
- persistence failure propagation into the durable session-history
  retry/spool path (never silently swallowed)

No live database, harness expansion, or migration execution is involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

import litellm.integrations.aawm_agent_identity as identity
from litellm.integrations.aawm_session_history import sql as sh_sql
from litellm.proxy.aawm_route_logging import record_aawm_route_rollup_turn


REPO_ROOT = Path(__file__).resolve().parents[3]
D1_616_MIGRATION = (
    REPO_ROOT
    / "scripts/apply_session_history_codex_review_decisions_2026_08_12.sql"
)
D1_616_DOCS = REPO_ROOT / "docs/aawm-session-history.md"


def _d1_616_docs_section() -> str:
    document = D1_616_DOCS.read_text(encoding="utf-8")
    return document.split("## Codex Review Decision Persistence", 1)[1].split(
        "\n## ", 1
    )[0]


def _base_record() -> Dict[str, Any]:
    return {
        "litellm_call_id": "call-review-1",
        "session_id": "session-review-1",
        "trace_id": "trace-review-1",
        "provider": "openai",
        "model": "codex-auto-review:low",
        "agent_name": "codex-auto-review",
        "agent_id": "agent-review-1",
    }


def _event(**overrides: Any) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "outcome": "allow",
        "rationale": "safe read-only change",
        "risk_level": "low",
        "reviewer_litellm_call_id": "call-review-1",
        "reviewer_session_id": "session-review-1",
        "review_attempt_number": 1,
        "review_attempt_key": "attempt-1",
    }
    event.update(overrides)
    return event


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


def test_d1_616_migration_uses_required_parameterized_database_guard() -> None:
    migration = D1_616_MIGRATION.read_text(encoding="utf-8")
    assert "aawm_tristore" not in migration
    assert r"\set ON_ERROR_STOP on" in migration
    assert r"\if :{?expected_database}" in migration
    assert (
        "SELECT CASE\n"
        "    WHEN NULLIF(btrim(:'expected_database'), '') = current_database()\n"
        "        THEN 'true'\n"
        "    ELSE 'false'\n"
        "END AS d1_616_database_matches \\gset"
        in migration
    )
    assert "THEN 'true'" in migration
    assert "ELSE 'false'" in migration
    assert r"\if :d1_616_database_matches" in migration
    assert (
        "D1-616 abort: expected_database is empty or does not match "
        "current_database()" in migration
    )
    assert (
        "D1-616 abort: required psql variable expected_database is missing"
        in migration
    )
    assert r"\set guard_statement 'SELECT 1'" in migration
    assert (
        r"\set guard_statement 'SELECT 1 / 0 AS d1_616_database_guard_failure'"
        in migration
    )
    assert migration.splitlines().count(
        r"\set guard_statement 'SELECT 1 / 0 AS d1_616_database_guard_failure'"
    ) == 2
    assert migration.splitlines().count(
        r"\set guard_statement 'SELECT 1'"
    ) == 1
    assert ":guard_statement;" in migration
    assert r"\quit" not in migration
    assert "DO $$" not in migration
    assert "expected_database_name" not in migration
    assert "RAISE EXCEPTION" not in migration
    begin_idx = migration.index("BEGIN;")
    assert migration.index(r"\set guard_statement 'SELECT 1'") < begin_idx
    assert migration.index(
        r"\set guard_statement 'SELECT 1 / 0 AS d1_616_database_guard_failure'"
    ) < begin_idx
    assert migration.index(r"\if :{?expected_database}") < migration.index(
        "SELECT CASE"
    )
    assert migration.index("SELECT CASE") < begin_idx
    assert migration.index(":guard_statement;") < begin_idx


def test_d1_616_migration_keeps_idempotent_ddl_after_guard() -> None:
    migration = D1_616_MIGRATION.read_text(encoding="utf-8")
    begin_idx = migration.index("BEGIN;")
    assert migration.index("CREATE TABLE IF NOT EXISTS", begin_idx) > begin_idx
    assert migration.count("CREATE INDEX IF NOT EXISTS") == 7
    assert migration.endswith("COMMIT;\n")


def test_d1_616_docs_keep_dev_prod_migration_commands_in_parity() -> None:
    section = _d1_616_docs_section()
    normalized_section = " ".join(section.split())
    assert "beside `public.session_history`" in normalized_section
    assert "--set=expected_database=aawm_tristore" not in section
    assert "--dbname=aawm_tristore" not in section
    assert (
        "Production apply requires separate operator authorization and is not"
        in section
    )
    for database in ("litellm_dev", "litellm_prod"):
        command = (
            "psql --set=ON_ERROR_STOP=1 \\\n"
            f"  --set=expected_database={database} \\\n"
            f"  --dbname={database} \\\n"
            "  --file=scripts/apply_session_history_codex_review_decisions_2026_08_12.sql"
        )
        assert command in section


def test_decisions_table_defines_stable_identities_and_bounds() -> None:
    ddl = sh_sql._AAWM_SESSION_HISTORY_CODEX_REVIEW_DECISIONS_TABLE_SQL
    for column in (
        "decision_key TEXT NOT NULL",
        "reviewer_litellm_call_id TEXT NOT NULL",
        "reviewer_session_id TEXT NOT NULL",
        "session_id TEXT",
        "parent_litellm_call_id TEXT",
        "parent_thread_id TEXT",
        "parent_agent_id TEXT",
        "rationale TEXT",
        "rationale_truncated BOOLEAN NOT NULL DEFAULT FALSE",
        "parser_version TEXT",
        "contract_version TEXT",
        "review_attempt_number INTEGER",
        "review_attempt_key TEXT",
        "governed_tool_call_id TEXT",
        "governed_tool_activity_key TEXT",
        "correlation_status TEXT NOT NULL DEFAULT 'unattributed'",
    ):
        assert column in ddl, f"missing column definition: {column}"
    assert "CHECK (outcome IN ('allow', 'deny'))" in ddl
    assert (
        "CHECK (correlation_status IN ('attributed', 'unattributed'))" in ddl
    )
    assert "UNIQUE (decision_key)" in ddl


def test_decisions_indexes_cover_session_reviewer_parent_and_governed_ids() -> None:
    statements = " ".join(
        sh_sql._AAWM_SESSION_HISTORY_CODEX_REVIEW_DECISIONS_INDEX_STATEMENTS
    )
    assert "session_history_codex_review_decisions_session_created_idx" in statements
    assert "session_history_codex_review_decisions_reviewer_call_idx" in statements
    assert "session_history_codex_review_decisions_parent_call_idx" in statements
    assert "WHERE parent_litellm_call_id IS NOT NULL" in statements
    assert "session_history_codex_review_decisions_governed_tool_call_idx" in statements
    assert "WHERE governed_tool_call_id IS NOT NULL" in statements


def test_decisions_insert_is_idempotent_on_decision_key() -> None:
    insert_sql = sh_sql._AAWM_SESSION_HISTORY_CODEX_REVIEW_DECISION_INSERT_SQL
    assert insert_sql.count("$") >= 26
    assert "ON CONFLICT (decision_key) DO UPDATE SET" in insert_sql


def test_decisions_conflict_enrichment_never_touches_immutable_identity() -> None:
    insert_sql = sh_sql._AAWM_SESSION_HISTORY_CODEX_REVIEW_DECISION_INSERT_SQL
    conflict_clause = insert_sql.split("ON CONFLICT", 1)[1]
    # Immutable identity and linkage columns are never updated on conflict.
    for immutable_column in (
        "outcome",
        "reviewer_litellm_call_id",
        "reviewer_session_id",
        "reviewer_model",
        "reviewer_agent_name",
        "reviewer_agent_id",
        "session_id",
        "parent_litellm_call_id",
        "parent_session_id",
        "parent_thread_id",
        "parent_agent_name",
        "parent_agent_id",
        "correlation_status",
        "parser_version",
        "contract_version",
        "review_attempt_number",
        "review_attempt_key",
        "governed_tool_call_id",
        "governed_tool_activity_key",
    ):
        assert (
            f"{immutable_column} =" not in conflict_clause
        ), f"immutable column updated on conflict: {immutable_column}"
    # Safe non-identity enrichment fields may fill empty values only.
    for safe_column in (
        "reviewer_trace_id",
        "rationale",
        "rationale_truncated",
        "risk_level",
        "user_authorization",
        "metadata",
    ):
        assert (
            f"{safe_column} =" in conflict_clause
        ), f"safe enrichment column missing from conflict clause: {safe_column}"


def test_decisions_conflict_replay_is_stored_first_coalesce() -> None:
    # Conflict replay must keep the stored row winning: every enrichment
    # column uses stored-first COALESCE ordering so replay only fills
    # NULL/empty stored values and never overwrites a set one. Assert the
    # exact operands and order, not mere assignment existence.
    conflict_clause = (
        sh_sql._AAWM_SESSION_HISTORY_CODEX_REVIEW_DECISION_INSERT_SQL.split(
            "ON CONFLICT", 1
        )[1]
    )
    stored_table = "session_history_codex_review_decisions"
    for column in (
        "reviewer_trace_id",
        "rationale",
        "risk_level",
        "user_authorization",
    ):
        expected = (
            f"{column} = COALESCE(NULLIF({stored_table}.{column}, ''), "
            f"NULLIF(EXCLUDED.{column}, ''))"
        )
        assert expected in conflict_clause, (
            f"stored-first COALESCE ordering wrong for {column}"
        )
    # rationale_truncated may OR stored and incoming values.
    assert (
        f"rationale_truncated = {stored_table}.rationale_truncated "
        "OR EXCLUDED.rationale_truncated" in conflict_clause
    )
    # Existing metadata keys win: merge incoming first, stored second.
    expected_metadata = (
        "metadata = COALESCE(EXCLUDED.metadata, '{}'::jsonb) || "
        f"COALESCE({stored_table}.metadata, '{{}}'::jsonb)"
    )
    assert expected_metadata in conflict_clause


# ---------------------------------------------------------------------------
# Decision identity: explicit, stable, retry-safe
# ---------------------------------------------------------------------------


def test_missing_attempt_identity_is_rejected() -> None:
    # No decision_id and no review_attempt_key: the decision cannot be
    # uniquely identified across retries, so it must not be persisted.
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(review_attempt_key=None, review_attempt_number=None)
            ],
        }
    )
    assert payloads == []

    # Either explicit identity alone is sufficient.
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(review_attempt_key=None, decision_id="decision-1")
            ],
        }
    )
    assert len(payloads) == 1
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(decision_id=None, review_attempt_key="attempt-1")
            ],
        }
    )
    assert len(payloads) == 1


def test_outcome_change_same_decision_key_is_idempotent() -> None:
    # The decision key must not depend on mutable outcome/rationale: an
    # outcome change for the same explicit attempt identity maps to the same
    # key, so the first persisted immutable identity wins on replay.
    allow_key = identity._build_codex_review_decision_db_payloads(
        {**_base_record(), "codex_review_decisions": [_event()]}
    )[0][0]
    deny_key = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(outcome="deny", rationale="reconsidered")
            ],
        }
    )[0][0]
    assert allow_key == deny_key

    # Rationale enrichment also never changes the key.
    enriched_key = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(rationale="late rationale enrichment")
            ],
        }
    )[0][0]
    assert enriched_key == allow_key


def test_retries_produce_distinct_decision_keys_and_replays_merge() -> None:
    first = identity._build_codex_review_decision_db_payloads(
        {**_base_record(), "codex_review_decisions": [_event()]}
    )[0][0]
    retry = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(review_attempt_number=2, review_attempt_key="attempt-2")
            ],
        }
    )[0][0]
    replay = identity._build_codex_review_decision_db_payloads(
        {**_base_record(), "codex_review_decisions": [_event()]}
    )[0][0]
    assert first != retry
    assert first == replay


def test_multiple_decisions_for_concurrent_parents() -> None:
    record = {
        **_base_record(),
        "codex_review_decisions": [
            _event(parent_litellm_call_id="call-a"),
            _event(parent_litellm_call_id="call-b"),
        ],
    }
    payloads = identity._build_codex_review_decision_db_payloads(record)
    assert len(payloads) == 2
    assert {payload[13] for payload in payloads} == {"call-a", "call-b"}
    # Same reviewer call, same attempt, same outcome: only the explicit
    # parent identity differs, so the decision keys must remain distinct.
    assert {payload[7] for payload in payloads} == {"allow"}
    assert payloads[0][0] != payloads[1][0]


def test_malformed_events_are_not_persisted() -> None:
    record = {
        **_base_record(),
        "codex_review_decisions": [
            _event(outcome="maybe"),  # unsupported outcome
            _event(outcome=""),  # empty outcome
            "not-a-dict",
            None,
        ],
    }
    assert identity._build_codex_review_decision_db_payloads(record) == []

    # An event lacking explicit reviewer identity falls back to the reviewer
    # record's own stable call/session identity (the reviewer call is the
    # record itself). When NO stable identity exists anywhere, the event is
    # rejected rather than persisting a fabricated identity.
    anonymous = {
        "codex_review_decisions": [_event(reviewer_litellm_call_id=None)]
    }
    assert identity._build_codex_review_decision_db_payloads(anonymous) == []

    # A bare outcome event has no explicit stable decision identity either:
    # reject rather than keying on mutable content.
    fallback = identity._build_codex_review_decision_db_payloads(
        {**_base_record(), "codex_review_decisions": [{"outcome": "allow"}]}
    )
    assert fallback == []

    fallback = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                {"outcome": "allow", "review_attempt_key": "attempt-x"}
            ],
        }
    )
    assert len(fallback) == 1
    assert fallback[0][1] == "call-review-1"
    assert fallback[0][2] == "session-review-1"


def test_duplicate_events_are_deduplicated_and_metadata_events_read() -> None:
    event = _event()
    record = {
        **_base_record(),
        "codex_review_decisions": [event, dict(event)],
        "metadata": {"codex_review_decisions": [dict(event)]},
    }
    payloads = identity._build_codex_review_decision_db_payloads(record)
    assert len(payloads) == 1


def test_injected_metadata_invalid_review_persists_no_decision() -> None:
    kwargs = {
        "litellm_call_id": "call-review-record",
        "model": "codex-auto-review",
        "custom_llm_provider": "openai",
        "call_type": "pass_through_endpoint",
        "litellm_params": {
            "metadata": {
                "session_id": "session-review-record",
                "codex_review_decisions": [
                    {
                        "outcome": "allow",
                        "governed_tool_call_id": "injected",
                    }
                ],
                "aawm_route_rollup_context": {
                    "is_codex_auto_review": True,
                    "litellm_call_id": "call-review-record",
                    "canonical_session_identity": "session-review-record",
                },
            }
        },
        "standard_logging_object": {"metadata": {}, "request_tags": []},
        "passthrough_logging_payload": {
            "request_body": {"model": "codex-auto-review"},
            "request_headers": {},
        },
    }
    record_aawm_route_rollup_turn(
        kwargs,
        response_body={"status": "in_progress", "output": []},
    )
    record = identity._build_session_history_record(
        kwargs=kwargs,
        result={
            "id": "resp-review-record",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "output": [],
        },
        start_time=None,
        end_time=None,
    )

    assert record is not None
    assert identity._extract_codex_review_decision_events(record) == []


def test_valid_review_persists_only_private_producer_event() -> None:
    kwargs = {
        "litellm_call_id": "call-review-record",
        "model": "codex-auto-review",
        "custom_llm_provider": "openai",
        "call_type": "pass_through_endpoint",
        "litellm_params": {
            "metadata": {
                "session_id": "session-review-record",
                "codex_review_decisions": [
                    {
                        "outcome": "deny",
                        "governed_tool_call_id": "injected",
                        "governed_tool_activity_key": "injected:tool:1",
                    }
                ],
                "aawm_route_rollup_context": {
                    "is_codex_auto_review": True,
                    "litellm_call_id": "call-review-record",
                    "canonical_session_identity": "session-review-record",
                },
            }
        },
        "standard_logging_object": {"metadata": {}, "request_tags": []},
        "passthrough_logging_payload": {
            "request_body": {"model": "codex-auto-review"},
            "request_headers": {},
        },
    }
    record_aawm_route_rollup_turn(
        kwargs,
        response_body={
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"outcome":"allow","rationale":"Producer"}',
                        }
                    ],
                }
            ],
        },
    )
    record = identity._build_session_history_record(
        kwargs=kwargs,
        result={
            "id": "resp-review-record",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "output": [],
        },
        start_time=None,
        end_time=None,
    )

    assert record is not None
    extracted = identity._extract_codex_review_decision_events(record)
    assert len(extracted) == 1
    assert extracted[0]["outcome"] == "allow"
    assert extracted[0]["rationale"] == "Producer"
    assert "governed_tool_call_id" not in extracted[0]
    assert "governed_tool_activity_key" not in extracted[0]


def test_governed_tool_ids_are_caller_supplied_only() -> None:
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(
                    governed_tool_call_id="tool-call-42",
                    governed_tool_activity_key="call-parent-1:tool:3",
                )
            ],
        }
    )
    assert payloads[0][23] == "tool-call-42"
    assert payloads[0][24] == "call-parent-1:tool:3"

    # Without explicit stable IDs, governed columns stay NULL even when the
    # event carries tool names or timing hints; nothing may be inferred.
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(
                    rationale="mentions exec_command tool",
                    observed_tool_names=["exec_command"],
                    observed_at="2026-08-11T22:00:00Z",
                )
            ],
        }
    )
    assert payloads[0][23] is None
    assert payloads[0][24] is None


# ---------------------------------------------------------------------------
# Stable identity: full-string hashing, collision-proof bounded storage
# ---------------------------------------------------------------------------


def test_over_long_ids_sharing_prefix_produce_distinct_identities() -> None:
    # Two stable IDs longer than the 128-char bound share a 150-char prefix:
    # naive prefix truncation would collide both the decision key and the
    # stored identity. Full-value hashing and prefix+SHA-256 bounded
    # storage must keep them distinct.
    shared_prefix = "d" * 150
    decision_a = f"{shared_prefix}A"
    decision_b = f"{shared_prefix}B"
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(review_attempt_key=None, decision_id=decision_a),
                _event(review_attempt_key=None, decision_id=decision_b),
            ],
        }
    )
    assert len(payloads) == 2
    assert payloads[0][0] != payloads[1][0]

    # Stored bounded representations stay <= 128 chars and distinct.
    stored_a = identity._bounded_stable_id(decision_a)
    stored_b = identity._bounded_stable_id(decision_b)
    assert stored_a is not None and stored_b is not None
    assert len(stored_a) <= 128 and len(stored_b) <= 128
    assert stored_a != stored_b

    # Short values pass through unchanged; equal full values are stable.
    assert identity._bounded_stable_id("decision-1") == "decision-1"
    assert (
        identity._bounded_stable_id(decision_a)
        == identity._bounded_stable_id(decision_a)
    )

    # Review attempt keys follow the same contract: the FULL sanitized value
    # feeds the decision-key hash, so two over-long attempt keys sharing the
    # same prefix produce distinct decision keys, while the stored column
    # (payload index 22) holds the bounded prefix+SHA-256 representation.
    attempt_a = f"{shared_prefix}A"
    attempt_b = f"{shared_prefix}B"
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(review_attempt_key=attempt_a),
                _event(review_attempt_key=attempt_b),
            ],
        }
    )
    assert len(payloads) == 2
    assert payloads[0][0] != payloads[1][0]
    stored_attempt_a = payloads[0][22]
    stored_attempt_b = payloads[1][22]
    assert stored_attempt_a is not None and stored_attempt_b is not None
    assert len(stored_attempt_a) <= 128 and len(stored_attempt_b) <= 128
    assert stored_attempt_a != stored_attempt_b
    assert stored_attempt_a == identity._bounded_stable_id(attempt_a)
    assert stored_attempt_b == identity._bounded_stable_id(attempt_b)


def test_parent_actor_only_correlation_distinct_keys_for_prefix_shared_actors() -> None:
    # Actor-only correlation: same reviewer call, decision, and attempt, but
    # two different parent actor IDs/names sharing a prefix longer than the
    # 128-char bound must produce DISTINCT decision keys (the key hashes the
    # FULL sanitized actor strings, never the bounded stored form), and the
    # stored actor columns stay bounded and distinct.
    shared_prefix = "a" * 150
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(
                    parent_agent_id=f"{shared_prefix}-agent-A",
                    parent_agent_name=f"{shared_prefix}-name-A",
                ),
                _event(
                    parent_agent_id=f"{shared_prefix}-agent-B",
                    parent_agent_name=f"{shared_prefix}-name-B",
                ),
            ],
        }
    )
    assert len(payloads) == 2
    assert payloads[0][0] != payloads[1][0]
    for payload in payloads:
        assert payload[18] == "attributed"
    stored_name_a, stored_id_a = payloads[0][16], payloads[0][17]
    stored_name_b, stored_id_b = payloads[1][16], payloads[1][17]
    assert stored_id_a is not None and stored_id_b is not None
    assert stored_name_a is not None and stored_name_b is not None
    assert len(stored_id_a) <= 128 and len(stored_id_b) <= 128
    assert len(stored_name_a) <= 128 and len(stored_name_b) <= 128
    assert stored_id_a != stored_id_b
    assert stored_name_a != stored_name_b


def test_parent_identity_fields_reach_decision_hash_as_full_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Distinct decision keys and distinct stored values alone cannot prove the
    # D1-616 fix: `_bounded_stable_id` appends a full-value SHA-256 digest to
    # the stored representation, so even the predecessor behavior (bounding
    # before the decision-key hash) produced distinct keys and stored values.
    # Capture the arguments fed to `_codex_review_decision_key` and assert the
    # parent identity fields arrive as the COMPLETE sanitized over-long
    # strings, while payload indexes 13/14/15 hold only the bounded
    # prefix+SHA-256 stable representations.
    shared_prefix = "p" * 150
    key_kwargs: list[Dict[str, Any]] = []
    real_key = identity._codex_review_decision_key

    def capturing_key(**kwargs: Any) -> str:
        key_kwargs.append(kwargs)
        return real_key(**kwargs)

    monkeypatch.setattr(
        identity, "_codex_review_decision_key", capturing_key
    )

    field_names = (
        "parent_litellm_call_id",
        "parent_session_id",
        "parent_thread_id",
    )
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(**{name: f"{shared_prefix}-A" for name in field_names}),
                _event(**{name: f"{shared_prefix}-B" for name in field_names}),
            ],
        }
    )

    assert len(payloads) == 2
    assert len(key_kwargs) == 2
    assert payloads[0][0] != payloads[1][0]

    for event_index, variant in enumerate(("A", "B")):
        for name in field_names:
            hash_input = key_kwargs[event_index][name]
            assert hash_input == f"{shared_prefix}-{variant}"
            assert len(hash_input) > 128
            stored = payloads[event_index][13 + field_names.index(name)]
            assert stored == identity._bounded_stable_id(hash_input)
            assert stored is not None and len(stored) <= 128
            assert stored != hash_input


# ---------------------------------------------------------------------------
# Attribution: D1-615 parity (canonical session + parent actor/thread)
# ---------------------------------------------------------------------------


def test_parent_correlation_requires_session_plus_actor_or_thread() -> None:
    record = {
        **_base_record(),
        "codex_review_decisions": [
            _event(
                parent_litellm_call_id="call-parent-1",
                parent_thread_id="thread-parent-1",
                parent_agent_name="alibaba",
                parent_agent_id="agent-parent-1",
                parent_session_id="session-root-1",
            )
        ],
    }
    payload = identity._build_codex_review_decision_db_payloads(record)[0]
    assert payload[12] == "session-review-1"
    assert payload[13] == "call-parent-1"
    assert payload[14] == "session-root-1"
    assert payload[15] == "thread-parent-1"
    assert payload[16] == "alibaba"
    assert payload[17] == "agent-parent-1"
    assert payload[18] == "attributed"

    # Timestamps/order/model-only hints must not attribute: absent explicit
    # parent identity stays unattributed with no parent fields.
    payload = identity._build_codex_review_decision_db_payloads(
        {**_base_record(), "codex_review_decisions": [_event()]}
    )[0]
    assert payload[13] is None
    assert payload[14] is None
    assert payload[15] is None
    assert payload[16] is None
    assert payload[17] is None
    assert payload[18] == "unattributed"


def test_parent_actor_alone_with_canonical_session_is_attributed() -> None:
    # D1-615 parity: canonical session plus a parent actor (either actor
    # field) attributes even without a parent thread or call.
    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(parent_agent_id="agent-parent-1")
            ],
        }
    )[0]
    assert payload[12] == "session-review-1"
    assert payload[17] == "agent-parent-1"
    assert payload[18] == "attributed"

    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(parent_agent_name="alibaba")
            ],
        }
    )[0]
    assert payload[16] == "alibaba"
    assert payload[18] == "attributed"


def test_parent_thread_alone_with_canonical_session_is_attributed() -> None:
    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(parent_thread_id="thread-parent-1")
            ],
        }
    )[0]
    assert payload[12] == "session-review-1"
    assert payload[15] == "thread-parent-1"
    assert payload[18] == "attributed"


def test_parent_call_only_is_narrowing_and_never_attributes() -> None:
    # parent_litellm_call_id is a narrowing field only: even with the
    # canonical session present, call identity without a parent actor or
    # thread stays unattributed.
    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(parent_litellm_call_id="call-parent-1")
            ],
        }
    )[0]
    assert payload[12] == "session-review-1"
    assert payload[13] == "call-parent-1"
    assert payload[18] == "unattributed"


def test_parent_identity_without_canonical_session_is_unattributed() -> None:
    # Parent actor/thread identity without a canonical session identity must
    # NOT attribute and must not create a linked relationship.
    record = {
        "litellm_call_id": "call-review-1",
        "codex_review_decisions": [
            _event(
                parent_litellm_call_id="call-parent-1",
                parent_thread_id="thread-parent-1",
                parent_agent_name="alibaba",
            )
        ],
    }
    payload = identity._build_codex_review_decision_db_payloads(record)[0]
    assert payload[12] is None  # canonical session_id absent
    assert payload[13] == "call-parent-1"
    assert payload[15] == "thread-parent-1"
    assert payload[16] == "alibaba"
    assert payload[18] == "unattributed"

    # Explicit canonical session identity on the event restores attribution.
    payload = identity._build_codex_review_decision_db_payloads(
        {
            "litellm_call_id": "call-review-1",
            "codex_review_decisions": [
                _event(
                    session_id="session-canonical-1",
                    parent_thread_id="thread-parent-1",
                )
            ],
        }
    )[0]
    assert payload[12] == "session-canonical-1"
    assert payload[18] == "attributed"


# ---------------------------------------------------------------------------
# D1-615-parity sanitization, redaction, and bounds
# ---------------------------------------------------------------------------


def test_allow_with_rationale_builds_payload() -> None:
    payloads = identity._build_codex_review_decision_db_payloads(
        {**_base_record(), "codex_review_decisions": [_event()]}
    )
    assert len(payloads) == 1
    payload = payloads[0]
    decision_key = payload[0]
    assert decision_key.startswith("call-review-1:codex-review:")
    assert payload[1] == "call-review-1"  # reviewer call
    assert payload[2] == "session-review-1"  # reviewer session
    assert payload[7] == "allow"  # outcome
    assert payload[8] == "safe read-only change"  # rationale
    assert payload[9] is False  # rationale_truncated
    assert payload[10] == "low"  # risk_level
    assert payload[18] == "unattributed"  # correlation_status
    assert payload[19] == "codex-review-parser/1"
    assert payload[20] == "d1-616-contract/1"
    assert payload[21] == 1  # review_attempt_number
    assert payload[22] == "attempt-1"  # review_attempt_key


def test_allow_without_rationale_preserves_rationaleless_approval() -> None:
    payloads = identity._build_codex_review_decision_db_payloads(
        {**_base_record(), "codex_review_decisions": [_event(rationale=None)]}
    )
    assert len(payloads) == 1
    assert payloads[0][8] is None
    assert payloads[0][9] is False


def test_deny_with_rationale_builds_payload() -> None:
    payloads = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(outcome="deny", rationale="destructive command blocked")
            ],
        }
    )
    assert len(payloads) == 1
    assert payloads[0][7] == "deny"
    assert payloads[0][8] == "destructive command blocked"


def test_rationale_is_bounded_to_512_and_sanitized() -> None:
    long_rationale = "x" * 5000
    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [_event(rationale=long_rationale)],
        }
    )[0]
    assert payload[8] == "x" * 512
    assert payload[9] is True

    nul_rationale = "ok\x00reason\x00"
    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [_event(rationale=nul_rationale)],
        }
    )[0]
    assert "\x00" not in payload[8]
    assert payload[8] == "okreason"
    assert payload[9] is False


def test_rationale_strips_ansi_and_control_chars_and_collapses_whitespace() -> None:
    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(rationale="\x1b[31mred\x1b[0m   text\x01here\nnow\tend")
            ],
        }
    )[0]
    assert "\x1b" not in payload[8]
    assert "\x01" not in payload[8]
    assert payload[8] == "red text here now end"


def test_rationale_redacts_bearer_token_key_secret_password() -> None:
    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(
                    rationale=(
                        "using Bearer abc123XYZ9 and sk-abcdef123456 "
                        "with api_key=hunter2secret and password: supersecret9 "
                        "plus secret=topsecret9"
                    )
                )
            ],
        }
    )[0]
    rationale = payload[8]
    assert "abc123XYZ9" not in rationale
    assert "sk-abcdef123456" not in rationale
    assert "hunter2secret" not in rationale
    assert "supersecret9" not in rationale
    assert "topsecret9" not in rationale
    assert "Bearer [REDACTED]" in rationale
    assert "[REDACTED]" in rationale


def test_free_form_identity_labels_are_bounded_and_sanitized() -> None:
    payload = identity._build_codex_review_decision_db_payloads(
        {
            **_base_record(),
            "codex_review_decisions": [
                _event(
                    risk_level="z" * 500,
                    user_authorization="token=abcd1234 " + "u" * 500,
                    reviewer_model="m" * 500,
                    reviewer_agent_name="agent\x00name\x1b[1m",
                    parser_version="p" * 500,
                    contract_version="c" * 500,
                    governed_tool_call_id="g" * 500,
                    review_attempt_key="k" * 500,
                )
            ],
        }
    )[0]
    # Over-long labels are bounded via prefix + SHA-256 suffix of the full
    # value, never a bare prefix.
    assert payload[10] == identity._bounded_stable_id("z" * 500)
    assert payload[10].startswith("z" * 111)
    assert len(payload[10]) <= 128  # risk_level bounded
    assert payload[11] is not None
    assert len(payload[11]) <= 128  # user_authorization bounded
    assert "abcd1234" not in payload[11]  # token redacted
    assert len(payload[4]) <= 128  # reviewer_model bounded
    assert "\x00" not in payload[5]  # agent name sanitized
    assert "\x1b" not in payload[5]
    assert len(payload[19]) <= 128  # parser_version bounded
    assert len(payload[20]) <= 128  # contract_version bounded
    assert len(payload[23]) <= 128  # governed_tool_call_id bounded
    # review_attempt_key is stored bounded as well (index 22).
    assert payload[22] == identity._bounded_stable_id("k" * 500)
    assert payload[22] is not None and len(payload[22]) <= 128


# ---------------------------------------------------------------------------
# Persist behavior: propagation into durable retry/spool semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_codex_review_decisions_executes_insert() -> None:
    conn = AsyncMock()
    record = {**_base_record(), "codex_review_decisions": [_event()]}
    await identity._persist_codex_review_decisions(conn, [record])
    conn.executemany.assert_awaited_once()
    sql_arg = conn.executemany.await_args.args[0]
    payloads = conn.executemany.await_args.args[1]
    assert sql_arg is sh_sql._AAWM_SESSION_HISTORY_CODEX_REVIEW_DECISION_INSERT_SQL
    assert len(payloads) == 1


@pytest.mark.asyncio
async def test_persist_codex_review_decisions_no_events_no_write() -> None:
    conn = AsyncMock()
    await identity._persist_codex_review_decisions(conn, [_base_record()])
    conn.executemany.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_codex_review_decisions_failure_propagates_for_retry() -> None:
    conn = AsyncMock()
    conn.executemany.side_effect = RuntimeError("db unavailable")
    record = {**_base_record(), "codex_review_decisions": [_event()]}
    # Persistence failures must propagate into the durable session-history
    # retry/spool path instead of being silently swallowed after the primary
    # write; both the primary insert and decision insert are idempotent on
    # their stable keys, so retry is safe.
    with pytest.raises(RuntimeError):
        await identity._persist_codex_review_decisions(conn, [record])
    conn.executemany.assert_awaited_once()


class _FakePoolAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _FakePoolAcquire(self.conn)


@pytest.mark.asyncio
async def test_record_level_decision_failure_propagates_past_primary_write(
    monkeypatch,
) -> None:
    record = {
        "litellm_call_id": "call-dec-1",
        "session_id": "session-dec-1",
        "trace_id": "trace-dec-1",
        "provider": "openai",
        "model": "codex-auto-review:low",
        "agent_name": "codex-auto-review",
        "codex_review_decisions": [_event()],
    }
    mock_conn = AsyncMock()
    # Decision insert fails; the exception must escape the record-level
    # persist path so the durable writer retry/spool machinery can act.
    mock_conn.executemany.side_effect = RuntimeError("decision insert failed")
    monkeypatch.setattr(
        "litellm.integrations.aawm_agent_identity._get_aawm_session_history_pool",
        AsyncMock(return_value=_FakePool(mock_conn)),
    )
    monkeypatch.setattr(
        "litellm.integrations.aawm_agent_identity._ensure_session_history_schema",
        AsyncMock(),
    )

    with pytest.raises(RuntimeError):
        await identity._persist_session_history_record(record)

    # Primary session_history write ran before the decision write failed.
    assert mock_conn.execute.await_count >= 1
    first_sql = mock_conn.execute.await_args_list[0].args[0]
    assert "INSERT INTO public.session_history" in first_sql


def test_record_apis_published_on_identity_host() -> None:
    for name in (
        "_extract_codex_review_decision_events",
        "_sanitize_codex_review_decision_text",
        "_bounded_stable_id",
        "_sanitize_codex_review_decision_label",
        "_sanitize_codex_review_decision_rationale",
        "_codex_review_decision_key",
        "_build_codex_review_decision_db_payload",
        "_build_codex_review_decision_db_payloads",
        "_persist_codex_review_decisions",
    ):
        assert callable(getattr(identity, name)), name
