"""Lease-bound collector scheduler with coalesced catch-up."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from uuid import uuid4

from .collector import Collector, RunResult
from .config import AccountConfig, CollectorConfig
from .ledger import Ledger
from .timeutil import ensure_utc, format_iso_duration, isoformat_utc, parse_datetime, parse_retry_after

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ScheduleDecision:
    due: bool
    missed_intervals: int
    next_due_at: datetime
    jitter_seconds: int
    coalesced: bool
    reason: str


class LeaseHeldError(RuntimeError):
    """Another collector already holds the account lease."""


class Scheduler:
    def __init__(
        self,
        config: CollectorConfig,
        ledger: Ledger,
        collector: Collector,
        *,
        clock: Clock | None = None,
        owner: str = "local-cli",
        jitter_fn: Callable[[int], int] | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.collector = collector
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.owner = owner
        self.jitter_fn = jitter_fn or (lambda maximum: __import__("random").randint(0, max(0, maximum)))

    def decide(self, account: AccountConfig, *, force: bool = False) -> ScheduleDecision:
        now = ensure_utc(self.clock())
        state = self.ledger.get_scheduler_state(account.id)
        interval = account.scheduler.refresh_interval
        jitter = int(account.scheduler.jitter_seconds)
        backoff_until = parse_datetime(state.get("backoff_until")) if state else None
        # Clear stale backoff that has already passed
        if backoff_until is not None and ensure_utc(backoff_until) <= now:
            self.ledger.upsert_scheduler_state(account.id, backoff_until=None)
            backoff_until = None
        if backoff_until is not None and not force:
            return ScheduleDecision(False, 0, now, jitter, False, "backoff")
        if state is None or not state.get("next_due_at"):
            next_due = now
            missed = 0
            if force:
                return ScheduleDecision(True, 0, next_due, jitter, False, "forced")
            return ScheduleDecision(True, 0, next_due, jitter, False, "uninitialized")
        next_due = ensure_utc(parse_datetime(state["next_due_at"]) or now)
        if now < next_due and not force:
            return ScheduleDecision(False, 0, next_due, jitter, False, "not_due")
        missed = 0
        cursor = next_due
        while cursor + interval <= now:
            missed += 1
            cursor = cursor + interval
        coalesced = missed > 0
        if force and now < next_due:
            return ScheduleDecision(True, 0, next_due, jitter, False, "forced")
        return ScheduleDecision(True, missed, next_due, jitter, coalesced, "due")

    def acquire_lease(self, account: AccountConfig, *, ttl: timedelta = timedelta(minutes=25)) -> str:
        now = ensure_utc(self.clock())
        state = self.ledger.get_scheduler_state(account.id) or {}
        lease_until = parse_datetime(state.get("lease_until"))
        lease_owner = state.get("lease_owner")
        lease_token = state.get("lease_token")
        backoff_until = parse_datetime(state.get("backoff_until"))
        if backoff_until is not None and ensure_utc(backoff_until) > now:
            raise LeaseHeldError(f"backoff active until {backoff_until}")
        if lease_until is not None and ensure_utc(lease_until) > now:
            if lease_owner != self.owner:
                raise LeaseHeldError(f"lease held by {lease_owner}")
            if lease_token:
                # Re-fence: extend the lease so it doesn't expire mid-run
                with self.ledger.transaction():
                    self.ledger.upsert_scheduler_state(
                        account.id,
                        lease_until=isoformat_utc(now + ttl),
                    )
                return str(lease_token)
        token = str(uuid4())
        with self.ledger.transaction():
            self.ledger.upsert_scheduler_state(
                account.id,
                refresh_interval=format_iso_duration(account.scheduler.refresh_interval),
                lease_owner=self.owner,
                lease_token=token,
                lease_until=isoformat_utc(now + ttl),
            )
        return token

    def release_lease(self, account: AccountConfig, token: str) -> None:
        state = self.ledger.get_scheduler_state(account.id) or {}
        if state.get("lease_token") != token:
            return
        with self.ledger.transaction():
            self.ledger.upsert_scheduler_state(
                account.id,
                lease_owner=None,
                lease_token=None,
                lease_until=None,
            )

    def set_interval(self, account: AccountConfig, interval: timedelta) -> dict[str, object]:
        if interval.total_seconds() < 300:
            raise ValueError("refresh interval must be at least PT5M")
        now = ensure_utc(self.clock())
        state = self.ledger.get_scheduler_state(account.id)
        next_due = now + interval
        if state and state.get("next_due_at"):
            previous = parse_datetime(state["next_due_at"])
            if previous is not None and ensure_utc(previous) > now:
                next_due = ensure_utc(previous)
        with self.ledger.transaction():
            self.ledger.upsert_scheduler_state(
                account.id,
                refresh_interval=format_iso_duration(interval),
                next_due_at=isoformat_utc(next_due),
            )
        return {
            "account_id": account.id,
            "refresh_interval": format_iso_duration(interval),
            "next_due_at": isoformat_utc(next_due),
        }

    def persist_retry_after(self, account: AccountConfig, retry_after: str) -> datetime:
        now = ensure_utc(self.clock())
        until = parse_retry_after(retry_after, now=now)
        with self.ledger.transaction():
            self.ledger.upsert_scheduler_state(
                account.id,
                backoff_until=isoformat_utc(until),
            )
        return until

    def run_if_due(
        self,
        account_id: Optional[str] = None,
        *,
        force: bool = False,
        mode: str = "refresh",
        since: Optional[datetime] = None,
    ) -> Optional[RunResult]:
        account = self.config.account(account_id)
        decision = self.decide(account, force=force)
        if not decision.due:
            return None
        token = self.acquire_lease(account)
        now = ensure_utc(self.clock())
        jitter = self.jitter_fn(decision.jitter_seconds)
        try:
            result = self.collector.collect(
                account_id=account.id,
                mode="catch_up" if decision.coalesced else mode,
                since=since,
                missed_intervals=decision.missed_intervals,
                scheduled_for=decision.next_due_at,
            )
            next_due = decision.next_due_at + account.scheduler.refresh_interval + timedelta(seconds=jitter)
            with self.ledger.transaction():
                self.ledger.upsert_scheduler_state(
                    account.id,
                    refresh_interval=format_iso_duration(account.scheduler.refresh_interval),
                    next_due_at=isoformat_utc(next_due),
                    last_started_at=isoformat_utc(now),
                    last_finished_at=isoformat_utc(ensure_utc(self.clock())),
                    missed_intervals=decision.missed_intervals,
                    backoff_until=None,
                    pending_work_json=None,
                )
            result.missed_intervals = decision.missed_intervals
            return result
        finally:
            self.release_lease(account, token)
