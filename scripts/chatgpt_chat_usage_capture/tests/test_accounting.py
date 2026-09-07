"""Accounting windows, rebuild, and retention tests (D1-752 semantic lane)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.chatgpt_chat_usage_capture.accounting import (
    evaluate_window,
    record_manual_observation,
    rebuild_aggregates,
    set_explicit_window,
)
from scripts.chatgpt_chat_usage_capture.config import QuotaBucket, QuotaWindow
from scripts.chatgpt_chat_usage_capture.ledger import Ledger
from scripts.chatgpt_chat_usage_capture.models import AttemptRecord
from scripts.chatgpt_chat_usage_capture.privacy import SURFACE_CHAT
from scripts.chatgpt_chat_usage_capture.reporting import build_report
from scripts.chatgpt_chat_usage_capture.tests.test_collector import _make_config


NOW = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "usage.sqlite")


def _bucket(*, bucket_id: str, families: tuple[str, ...], capacity: int, window: QuotaWindow) -> QuotaBucket:
    return QuotaBucket(
        id=bucket_id,
        families=families,
        capacity=capacity,
        unit="message",
        documented_period_hint="day",
        window=window,
        membership="union_once_per_attempt",
    )


def _attempt(
    attempt_id: str,
    *,
    family: str,
    when: datetime,
    surface: str = SURFACE_CHAT,
    origin: str | None = None,
    requested: str | None = None,
    recorded: str | None = None,
) -> AttemptRecord:
    requested = requested if requested is not None else family
    recorded = recorded if recorded is not None else family
    return AttemptRecord(
        attempt_id=attempt_id,
        conversation_id="conv-1",
        identity_basis="generation",
        time_basis="user_message",
        attempt_time=when,
        earliest_possible_at=when,
        latest_possible_at=when,
        requested_model_raw=family,
        requested_mode_raw=None,
        requested_reasoning_effort_raw=None,
        recorded_final_model_raw=recorded,
        resolved_model_raw=None,
        requested_family=requested,
        recorded_final_family=recorded,
        resolved_family=None,
        mapping_version="test",
        outcome="completed",
        completed_answer=True,
        generation_started=True,
        surface=surface,
        origin=origin,
        aliases=(("generation", attempt_id),),
        evidence_message_ids=(f"msg-{attempt_id}",),
        revision=1,
        warnings=(),
    )


def test_unknown_window_usage_is_null_not_zero(tmp_path: Path) -> None:
    config = _make_config()
    ledger = _ledger(tmp_path)
    account = config.account()
    ledger.upsert_account(
        {
            "collector_account_id": account.id,
            "quota_owner_id": account.quota_owner_id,
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
        }
    )
    ledger.upsert_attempt(account.id, _attempt("att-1", family="astra_pro", when=NOW))
    report = build_report(config, ledger, now=NOW)
    bucket = next(item for item in report["quota_buckets"] if item["bucket_id"] == "pro200-astra-chat")
    assert bucket["window"]["type"] == "unknown"
    assert bucket["working_usage_estimate"] is None
    assert bucket["working_remaining_estimate"] is None
    assert report["working_quota_usage_estimate_by_bucket"]["pro200-astra-chat"] is None
    assert report["observed_attempts_by_requested_family"]["astra_pro"] == 1


def test_union_once_per_attempt_shared_bucket(tmp_path: Path) -> None:
    config = _make_config(
        {
            "quota_policies": [
                {
                    "id": "pro200-chat-2026-09-05",
                    "status": "documented_seed_requires_account_verification",
                    "surface": "chat",
                    "buckets": [
                        {
                            "id": "astra_weekly",
                            "families": ["astra_pro"],
                            "capacity": 200,
                            "unit": "message",
                            "window": {
                                "type": "operator_explicit",
                                "start": "2026-09-06T00:00:00Z",
                                "end": "2026-09-13T00:00:00Z",
                            },
                        },
                        {
                            "id": "sol_daily",
                            "families": ["sol_pro"],
                            "capacity": 170,
                            "unit": "message",
                            "window": {
                                "type": "operator_explicit",
                                "start": "2026-09-06T00:00:00Z",
                                "end": "2026-09-07T00:00:00Z",
                            },
                        },
                        {
                            "id": "pro_combined_daily",
                            "families": ["astra_pro", "sol_pro"],
                            "capacity": 200,
                            "unit": "message",
                            "membership": "union_once_per_attempt",
                            "window": {
                                "type": "operator_explicit",
                                "start": "2026-09-06T00:00:00Z",
                                "end": "2026-09-07T00:00:00Z",
                            },
                        },
                    ],
                }
            ]
        }
    )
    ledger = _ledger(tmp_path)
    account = config.account()
    ledger.upsert_account(
        {
            "collector_account_id": account.id,
            "quota_owner_id": account.quota_owner_id,
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
        }
    )
    when = datetime(2026, 9, 6, 1, 0, tzinfo=timezone.utc)
    ledger.upsert_attempt(account.id, _attempt("att-astra", family="astra_pro", when=when))
    ledger.upsert_attempt(account.id, _attempt("att-sol", family="sol_pro", when=when))
    ledger.upsert_attempt(
        account.id,
        _attempt(
            "att-mismatch",
            family="astra_pro",
            when=when,
            requested="astra_pro",
            recorded="sol_pro",
        ),
    )
    report = build_report(config, ledger, now=NOW)
    by_id = {item["bucket_id"]: item for item in report["quota_buckets"]}
    assert by_id["astra_weekly"]["working_usage_estimate"] == 2
    assert by_id["sol_daily"]["working_usage_estimate"] == 1
    assert by_id["pro_combined_daily"]["working_usage_estimate"] == 3


def test_window_types_do_not_invent_unknown_bounds() -> None:
    now = datetime(2026, 9, 6, 18, 0, tzinfo=timezone.utc)
    unknown = evaluate_window(
        _bucket(
            bucket_id="u",
            families=("astra_pro",),
            capacity=200,
            window=QuotaWindow(type="unknown"),
        ),
        None,
        now=now,
    )
    assert unknown["known"] is False
    assert unknown["start"] is None
    assert unknown["end"] is None

    rolling = evaluate_window(
        _bucket(
            bucket_id="r",
            families=("astra_pro",),
            capacity=200,
            window=QuotaWindow(type="rolling_elapsed", duration=timedelta(hours=24)),
        ),
        None,
        now=now,
    )
    assert rolling["known"] is True
    assert rolling["end"] == now
    assert rolling["start"] == now - timedelta(hours=24)

    anchored = evaluate_window(
        _bucket(
            bucket_id="a",
            families=("astra_pro",),
            capacity=200,
            window=QuotaWindow(
                type="anchored_elapsed",
                start=datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc),
                duration=timedelta(days=7),
            ),
        ),
        None,
        now=now,
    )
    assert anchored["start"] == datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc)
    assert anchored["end"] == datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc)

    calendar = evaluate_window(
        _bucket(
            bucket_id="c",
            families=("sol_pro",),
            capacity=170,
            window=QuotaWindow(type="calendar", timezone="America/New_York"),
        ),
        None,
        now=now,
    )
    assert calendar["known"] is True
    assert calendar["timezone"] == "America/New_York"
    assert calendar["end"] > calendar["start"]


def test_manual_observation_does_not_set_window_start(tmp_path: Path) -> None:
    config = _make_config()
    ledger = _ledger(tmp_path)
    account = config.account()
    recorded = record_manual_observation(
        ledger,
        account_id=account.id,
        bucket_id="pro200-astra-chat",
        remaining=120,
        capacity=200,
        observed_at=NOW,
        source="operator-ui",
    )
    assert recorded["remaining"] == 120
    assert recorded["window_start"] is None
    assert ledger.get_window(account.id, "pro200-astra-chat") is None


def test_rebuild_publishes_only_after_commit(tmp_path: Path) -> None:
    config = _make_config()
    ledger = _ledger(tmp_path)
    account = config.account()
    ledger.upsert_account(
        {
            "collector_account_id": account.id,
            "quota_owner_id": account.quota_owner_id,
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
        }
    )
    ledger.upsert_attempt(account.id, _attempt("att-1", family="astra_pro", when=NOW))
    preview = rebuild_aggregates(config, ledger, account_id=account.id, apply=False, now=NOW)
    assert preview["dry_run"] is True
    assert preview["revision_id"] is None
    assert ledger.get_latest_revision(account.id) is None

    applied = rebuild_aggregates(config, ledger, account_id=account.id, apply=True, now=NOW)
    assert applied["published"] is True
    assert applied["revision_id"]
    latest = ledger.get_latest_revision(account.id)
    assert latest is not None
    assert latest["revision_id"] == applied["revision_id"]

    original = ledger.insert_aggregate_revision

    def boom(**kwargs):
        raise RuntimeError("crash during aggregate rebuild")

    ledger.insert_aggregate_revision = boom  # type: ignore[method-assign]
    try:
        raised = False
        try:
            rebuild_aggregates(config, ledger, account_id=account.id, apply=True, now=NOW + timedelta(minutes=1))
        except RuntimeError:
            raised = True
        assert raised
        after = ledger.get_latest_revision(account.id)
        assert after is not None
        assert after["revision_id"] == applied["revision_id"]
    finally:
        ledger.insert_aggregate_revision = original  # type: ignore[method-assign]


def test_retention_tombstones_and_warns_without_raw_observations(tmp_path: Path) -> None:
    config = _make_config()
    ledger = _ledger(tmp_path)
    account = config.account()
    ledger.upsert_account(
        {
            "collector_account_id": account.id,
            "quota_owner_id": account.quota_owner_id,
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
        }
    )
    old = NOW - timedelta(days=200)
    attempt = _attempt("att-old", family="astra_pro", when=old)
    ledger.upsert_attempt(account.id, attempt)
    ledger.insert_observation(
        account_id=account.id,
        source_kind="conversation",
        source_id="conv-old",
        revision="rev-1",
        surface=SURFACE_CHAT,
        conversation_id="conv-1",
        payload={"id": "conv-old", "surface": SURFACE_CHAT},
        observed_at=old,
        run_id="run-old",
    )
    result = ledger.prune_retention(
        account.id,
        now=NOW,
        observation_days=45,
        attempt_days=180,
        daily_aggregate_days=400,
    )
    assert result["tombstoned_attempts"] == 1
    assert result["aliases_preserved"] is True
    assert any("raw observations pruned" in warning for warning in result["warnings"])
    live = ledger.list_attempts(account.id)
    assert live == []
    tombstoned = ledger.list_attempts(account.id, include_tombstones=True)
    assert len(tombstoned) == 1
    assert tombstoned[0]["tombstone"] == 1
    rebuild = rebuild_aggregates(config, ledger, account_id=account.id, apply=False, now=NOW)
    assert rebuild["payload"]["rebuild_source"] == "retained_attempts"
    assert any("raw observations absent" in warning for warning in rebuild["warnings"])


def test_set_explicit_window_and_boundary_membership(tmp_path: Path) -> None:
    config = _make_config()
    ledger = _ledger(tmp_path)
    account = config.account()
    start = datetime(2026, 9, 6, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)
    stored = set_explicit_window(
        ledger,
        account_id=account.id,
        bucket_id="pro200-astra-chat",
        start=start,
        end=end,
        evidence="operator-assumption",
        reason="test window",
    )
    assert stored["window_type"] == "operator_explicit"
    ledger.upsert_account(
        {
            "collector_account_id": account.id,
            "quota_owner_id": account.quota_owner_id,
            "surface": SURFACE_CHAT,
            "auth_state": "ready",
        }
    )
    ledger.upsert_attempt(account.id, _attempt("in-window", family="astra_pro", when=start))
    ledger.upsert_attempt(account.id, _attempt("at-end", family="astra_pro", when=end))
    report = build_report(config, ledger, window_bucket="pro200-astra-chat", now=NOW)
    bucket = next(item for item in report["quota_buckets"] if item["bucket_id"] == "pro200-astra-chat")
    assert bucket["working_usage_estimate"] == 1
    assert bucket["working_remaining_estimate"] == 199
