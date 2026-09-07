"""Scheduler lease, catch-up, backoff, and anchor tests (D1-752 semantic lane)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.chatgpt_chat_usage_capture.collector import RunResult
from scripts.chatgpt_chat_usage_capture.config import parse_config
from scripts.chatgpt_chat_usage_capture.ledger import Ledger
from scripts.chatgpt_chat_usage_capture.scheduler import LeaseHeldError, Scheduler
from scripts.chatgpt_chat_usage_capture.tests.test_collector import MINIMAL_CONFIG, _make_config
from scripts.chatgpt_chat_usage_capture.timeutil import isoformat_utc, parse_datetime


class FakeCollector:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def collect(self, **kwargs: Any) -> RunResult:
        self.calls.append(kwargs)
        return RunResult(
            run_id="run-1",
            mode=str(kwargs.get("mode") or "refresh"),
            result="complete",
            coverage="complete",
            new_attempts=0,
            updated_attempts=0,
            deduplicated_attempts=0,
            conversations_seen=0,
            pages_fetched=0,
            missed_intervals=int(kwargs.get("missed_intervals") or 0),
            warnings=[],
            requests=[],
        )


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "usage.sqlite")


def test_catch_up_coalesces_six_missed_hours(tmp_path: Path) -> None:
    config = _make_config()
    account = config.account()
    ledger = _ledger(tmp_path)
    collector = FakeCollector()
    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    scheduler = Scheduler(
        config,
        ledger,
        collector,  # type: ignore[arg-type]
        clock=lambda: now,
        jitter_fn=lambda _maximum: 0,
        owner="worker-a",
    )
    due = now - timedelta(hours=6)
    ledger.upsert_scheduler_state(
        account.id,
        refresh_interval="PT1H",
        schedule_anchor_at=isoformat_utc(due),
        next_due_at=isoformat_utc(due),
    )
    result = scheduler.run_if_due(account.id)
    assert result is not None
    assert result.missed_intervals == 6
    assert collector.calls[0]["mode"] == "catch_up"
    assert len(collector.calls) == 1
    assert collector.calls[0]["missed_intervals"] == 6
    assert collector.calls[0]["scheduled_for"] == due
    state = ledger.get_scheduler_state(account.id)
    assert state is not None
    next_due = parse_datetime(state["next_due_at"])
    assert next_due is not None
    assert next_due > now


def test_stale_worker_cannot_overwrite_newer_lease(tmp_path: Path) -> None:
    config = _make_config()
    account = config.account()
    ledger = _ledger(tmp_path)
    first = Scheduler(config, ledger, FakeCollector(), owner="worker-a", jitter_fn=lambda _m: 0)  # type: ignore[arg-type]
    token = first.acquire_lease(account)
    second = Scheduler(config, ledger, FakeCollector(), owner="worker-a", jitter_fn=lambda _m: 0)  # type: ignore[arg-type]
    try:
        second.acquire_lease(account)
        raised = False
    except LeaseHeldError:
        raised = True
    assert raised
    pending = ledger.pending_work(account.id)
    assert pending
    # Stale token must not release a newer fence.
    ledger.upsert_scheduler_state(account.id, lease_token="newer-token", lease_owner="worker-b")
    first.release_lease(account, token)
    state = ledger.get_scheduler_state(account.id)
    assert state is not None
    assert state["lease_token"] == "newer-token"


def test_retry_after_http_date_persists_across_restart(tmp_path: Path) -> None:
    config = _make_config()
    account = config.account()
    ledger = _ledger(tmp_path)
    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    scheduler = Scheduler(config, ledger, FakeCollector(), clock=lambda: now, jitter_fn=lambda _m: 0)  # type: ignore[arg-type]
    until = scheduler.persist_retry_after(account, "Wed, 09 Sep 2026 12:00:00 GMT")
    assert until.year == 2026
    restarted = Scheduler(config, ledger, FakeCollector(), clock=lambda: now, jitter_fn=lambda _m: 0)  # type: ignore[arg-type]
    decision = restarted.decide(account)
    assert decision.reason == "backoff"
    assert decision.due is False
    report_state = ledger.get_scheduler_state(account.id)
    assert report_state is not None
    assert report_state["backoff_until"] is not None


def test_interval_change_realigns_from_anchor(tmp_path: Path) -> None:
    config = _make_config()
    account = config.account()
    ledger = _ledger(tmp_path)
    now = datetime(2026, 9, 6, 12, 10, tzinfo=timezone.utc)
    scheduler = Scheduler(config, ledger, FakeCollector(), clock=lambda: now, jitter_fn=lambda _m: 7)  # type: ignore[arg-type]
    anchor = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
    ledger.upsert_scheduler_state(
        account.id,
        refresh_interval="PT1H",
        schedule_anchor_at=isoformat_utc(anchor),
        next_due_at=isoformat_utc(datetime(2026, 9, 6, 13, 0, tzinfo=timezone.utc)),
        last_jitter_seconds=7,
    )
    result = scheduler.set_interval(account, timedelta(hours=3))
    assert result["refresh_interval"] == "PT3H"
    next_due = parse_datetime(str(result["next_due_at"]))
    assert next_due is not None
    # Anchor 10:00 + 3h slots; after 12:10 the next slot is 13:00 plus recorded jitter 7s.
    assert next_due == datetime(2026, 9, 6, 13, 0, 7, tzinfo=timezone.utc)
    attempts_before = ledger.list_attempts(account.id)
    assert attempts_before == []


def test_interval_change_refused_during_active_lease(tmp_path: Path) -> None:
    config = _make_config()
    account = config.account()
    ledger = _ledger(tmp_path)
    scheduler = Scheduler(config, ledger, FakeCollector(), jitter_fn=lambda _m: 0)  # type: ignore[arg-type]
    scheduler.acquire_lease(account)
    try:
        scheduler.set_interval(account, timedelta(hours=3))
        raised = False
    except LeaseHeldError:
        raised = True
    assert raised


def test_expired_lease_can_be_reclaimed(tmp_path: Path) -> None:
    config = _make_config()
    account = config.account()
    ledger = _ledger(tmp_path)
    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    scheduler = Scheduler(config, ledger, FakeCollector(), clock=lambda: now, jitter_fn=lambda _m: 0)  # type: ignore[arg-type]
    ledger.upsert_scheduler_state(
        account.id,
        lease_owner="worker-a",
        lease_token="expired-token",
        lease_until=isoformat_utc(now - timedelta(minutes=1)),
    )
    token = scheduler.acquire_lease(account)
    assert token != "expired-token"
    state = ledger.get_scheduler_state(account.id)
    assert state is not None
    assert state["lease_token"] == token


def test_example_config_parses_spec_buckets() -> None:
    path = Path("scripts/chatgpt_chat_usage_capture/config.example.yaml")
    payload = path.read_text(encoding="utf-8")
    import yaml

    config = parse_config(yaml.safe_load(payload), source_path=path)
    assert config.retention.daily_aggregate_days == 400
    assert config.application.local_api_auth == "required"
    buckets = {bucket.id for bucket in config.quota_policies[0].buckets}
    assert buckets == {"astra_weekly", "sol_daily", "pro_combined_daily"}
    assert MINIMAL_CONFIG["quota_policies"][0]["buckets"][0]["id"] == "pro200-astra-chat"
