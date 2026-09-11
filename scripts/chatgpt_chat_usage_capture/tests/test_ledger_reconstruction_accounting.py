"""Focused regressions for D1-752 ledger, reconstruction, and accounting."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.chatgpt_chat_usage_capture.accounting import (
    rebuild_aggregates,
    summarize_attempts,
    working_contribution,
)
from scripts.chatgpt_chat_usage_capture.ledger import Ledger
from scripts.chatgpt_chat_usage_capture.models import AttemptRecord, MessageRecord
from scripts.chatgpt_chat_usage_capture.privacy import SURFACE_CHAT
from scripts.chatgpt_chat_usage_capture.reconstruct import reconstruct_attempts
from scripts.chatgpt_chat_usage_capture.tests.test_collector import _make_config


NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def _message(
    message_id: str,
    role: str,
    *,
    children: tuple[str, ...] = (),
    status: str | None = None,
    end_turn: bool | None = None,
    channel: str | None = None,
    generation_id: str | None = None,
    request_id: str | None = None,
    created_at: datetime = NOW,
    requested_model: str | None = "gpt-test",
    recorded_model: str | None = "gpt-test",
) -> MessageRecord:
    return MessageRecord(
        conversation_id="conv-1",
        message_id=message_id,
        node_id=message_id,
        parent_id=None if role == "user" else "user-1",
        children=children,
        role=role,
        channel=channel,
        created_at=created_at,
        status=status,
        end_turn=end_turn,
        requested_model_raw=requested_model if role == "user" else None,
        requested_mode_raw=None,
        requested_reasoning_effort_raw=None,
        recorded_final_model_raw=recorded_model if role == "assistant" else None,
        generation_id=generation_id,
        request_id=request_id,
        surface=SURFACE_CHAT,
        origin=None,
        metadata={},
    )


def _account(ledger: Ledger, account_id: str) -> None:
    ledger.upsert_account(
        {
            "collector_account_id": account_id,
            "quota_owner_id": "quota-owner",
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
        }
    )


def _attempt(
    *,
    mapping_version: str,
    requested_family: str,
    attempt_id: str = "attempt-1",
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        conversation_id="conv-1",
        identity_basis="generation",
        time_basis="user_message",
        attempt_time=NOW,
        earliest_possible_at=NOW,
        latest_possible_at=NOW,
        requested_model_raw="gpt-test",
        requested_mode_raw=None,
        requested_reasoning_effort_raw=None,
        recorded_final_model_raw="gpt-test",
        resolved_model_raw=None,
        requested_family=requested_family,
        recorded_final_family=requested_family,
        resolved_family=None,
        mapping_version=mapping_version,
        outcome="completed",
        completed_answer=True,
        generation_started=True,
        surface=SURFACE_CHAT,
        origin=None,
        aliases=(("generation", f"generation-{attempt_id}"),),
        evidence_message_ids=(f"message-{attempt_id}",),
        revision=1,
        warnings=(),
    )


def test_shared_request_id_keeps_distinct_generations_and_exact_repeats_idempotent(
    tmp_path: Path,
) -> None:
    messages = [
        _message(
            "user-1",
            "user",
            children=("assistant-1", "assistant-2"),
            status="finished_successfully",
            end_turn=True,
        ),
        _message(
            "assistant-1",
            "assistant",
            status="finished_successfully",
            end_turn=True,
            generation_id="generation-1",
            request_id="request-shared",
            created_at=NOW + timedelta(seconds=5),
        ),
        _message(
            "assistant-2",
            "assistant",
            status="finished_successfully",
            end_turn=True,
            generation_id="generation-2",
            request_id="request-shared",
            created_at=NOW + timedelta(seconds=6),
        ),
    ]
    attempts = reconstruct_attempts(
        messages,
        mapping_version="mapping-v1",
        mapping_rules=({"slug": "gpt-test", "family": "astra_pro"},),
        conversation_id="conv-1",
    )
    assert len(attempts) == 2
    assert {attempt.identity_basis for attempt in attempts} == {"generation"}
    assert {attempt.attempt_id for attempt in attempts}.__len__() == 2

    ledger = Ledger(tmp_path / "usage.sqlite")
    _account(ledger, "account-1")
    assert [ledger.upsert_attempt("account-1", attempt) for attempt in attempts] == [
        "inserted",
        "inserted",
    ]
    assert [ledger.upsert_attempt("account-1", attempt) for attempt in attempts] == [
        "deduplicated",
        "deduplicated",
    ]

    stored = ledger.list_attempts("account-1")
    assert len(stored) == 2
    generation_aliases = ledger.conn.execute(
        """
        SELECT alias_value, attempt_id
        FROM attempt_aliases
        WHERE collector_account_id=? AND alias_kind='generation'
        ORDER BY alias_value
        """,
        ("account-1",),
    ).fetchall()
    assert [(row["alias_value"], row["attempt_id"]) for row in generation_aliases] == [
        ("generation-1", attempts[0].attempt_id),
        ("generation-2", attempts[1].attempt_id),
    ]


@pytest.mark.parametrize(
    ("status", "expected_outcome"),
    [
        ("failed", "failed_after_start"),
        ("cancelled", "cancelled_after_start"),
        ("in_progress", "completion_unknown"),
    ],
)
def test_nonterminal_assistant_nodes_remain_uncertain(
    status: str,
    expected_outcome: str,
) -> None:
    attempts = reconstruct_attempts(
        [
            _message(
                "user-1",
                "user",
                children=("assistant-1",),
                status="finished_successfully",
                end_turn=True,
            ),
            _message(
                "assistant-1",
                "assistant",
                status=status,
                end_turn=True,
                generation_id=f"generation-{status}",
                request_id=f"request-{status}",
                created_at=NOW + timedelta(seconds=5),
            ),
        ],
        mapping_version="mapping-v1",
        mapping_rules=({"slug": "gpt-test", "family": "astra_pro"},),
        conversation_id="conv-1",
    )
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.outcome == expected_outcome
    assert attempt.completed_answer is False
    assert attempt.recorded_final_model_raw is None

    config = _make_config()
    summary = summarize_attempts(
        [asdict(attempt)],
        config,
        start=NOW - timedelta(minutes=1),
        end=NOW + timedelta(minutes=1),
    )
    assert summary["completed_final"] == {}
    assert summary["working"] == {}
    assert summary["unknown_debit"] == 1
    assert working_contribution(asdict(attempt), config) is None


def test_finished_analysis_does_not_complete_in_progress_final() -> None:
    attempts = reconstruct_attempts(
        [
            _message(
                "user-1",
                "user",
                children=("analysis-1", "final-1"),
                status="finished_successfully",
                end_turn=True,
            ),
            _message(
                "analysis-1",
                "assistant",
                status="finished_successfully",
                end_turn=True,
                channel="analysis",
                generation_id="generation-mixed",
                request_id="request-mixed",
                created_at=NOW + timedelta(seconds=5),
            ),
            _message(
                "final-1",
                "assistant",
                status="in_progress",
                end_turn=False,
                generation_id="generation-mixed",
                request_id="request-mixed",
                created_at=NOW + timedelta(seconds=6),
            ),
        ],
        mapping_version="mapping-v1",
        mapping_rules=({"slug": "gpt-test", "family": "astra_pro"},),
        conversation_id="conv-1",
    )

    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.completed_answer is False
    assert attempt.outcome == "completion_unknown"
    assert attempt.recorded_final_model_raw is None


def test_rebuild_reclassifies_retained_raw_evidence_and_preserves_history(
    tmp_path: Path,
) -> None:
    old_config = _make_config(
        {
            "model_mapping": {
                "version": "mapping-v1",
                "canonical_families": ["astra_pro", "sol_pro", "other_chat", "unknown"],
                "exact_rules": [{"slug": "gpt-test", "family": "astra_pro"}],
                "unknown_behavior": "preserve_and_report",
            }
        }
    )
    new_config = _make_config(
        {
            "model_mapping": {
                "version": "mapping-v2",
                "canonical_families": ["astra_pro", "sol_pro", "other_chat", "unknown"],
                "exact_rules": [{"slug": "gpt-test", "family": "sol_pro"}],
                "unknown_behavior": "preserve_and_report",
            }
        }
    )
    ledger = Ledger(tmp_path / "usage.sqlite")
    _account(ledger, old_config.account().id)
    ledger.upsert_attempt(
        old_config.account().id,
        _attempt(mapping_version="mapping-v1", requested_family="astra_pro"),
    )
    initial = rebuild_aggregates(old_config, ledger, apply=True, now=NOW)
    assert initial["payload"]["observed_attempts_by_requested_family"] == {"astra_pro": 1}

    preview = rebuild_aggregates(new_config, ledger, apply=False, now=NOW + timedelta(minutes=1))
    assert preview["reclassified_attempts"] == 1
    assert preview["payload"]["mapping_version"] == "mapping-v2"
    assert preview["payload"]["observed_attempts_by_requested_family"] == {"sol_pro": 1}
    assert ledger.list_attempts(new_config.account().id)[0]["mapping_version"] == "mapping-v1"

    applied = rebuild_aggregates(new_config, ledger, apply=True, now=NOW + timedelta(minutes=1))
    assert applied["reclassified_attempts"] == 1
    stored = ledger.list_attempts(new_config.account().id)
    assert stored[0]["mapping_version"] == "mapping-v2"
    assert stored[0]["requested_family"] == "sol_pro"
    history = ledger.list_attempt_mapping_history(
        new_config.account().id,
        stored[0]["attempt_id"],
    )
    assert [(item["mapping_version"], item["requested_family"]) for item in history] == [
        ("mapping-v1", "astra_pro"),
        ("mapping-v2", "sol_pro"),
    ]
    revisions = ledger.conn.execute(
        """
        SELECT mapping_version, payload_json
        FROM aggregate_revisions
        WHERE collector_account_id=?
        ORDER BY created_at, revision_id
        """,
        (new_config.account().id,),
    ).fetchall()
    assert [row["mapping_version"] for row in revisions] == ["mapping-v1", "mapping-v2"]
