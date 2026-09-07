"""Lease-bound collector scheduler with anchored due times and coalesced catch-up."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import floor
from typing import Callable, Optional
from uuid import uuid4

from .collector import Collector, RunResult
from .config import AccountConfig, CollectorConfig
from .ledger import Ledger
from .timeutil import ensure_utc, format_iso_duration, isoformat_utc, parse_datetime, parse_iso_duration, parse_retry_after

Clock = Callable[[], datetime]
DEFAULT_LEASE_TTL = timedelta(minutes=25)


@dataclass(frozen=True)
class ScheduleDecision:
    due: bool
    missed_intervals: int
    next_due_at: datetime
    jitter_seconds: int
    coalesced: bool
    reason: str
    pending_work: tuple[dict, ...] = ()


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
        lease_ttl: timedelta = DEFAULT_LEASE_TTL,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.collector = collector
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.owner = owner
        self.jitter_fn = jitter_fn or (lambda maximum: __import__("random").randint(0, max(0, maximum)))
        self.lease_ttl = lease_ttl

    def decide(self, account: AccountConfig, *, force: bool = False) -> ScheduleDecision:
        now = ensure_utc(self.clock())
        state = self.ledger.get_scheduler_state(account.id)
        interval = account.scheduler.refresh_interval
        jitter_max = int(account.scheduler.jitter_seconds)
        pending = tuple(self.ledger.pending_work(account.id))
        backoff_until = parse_datetime(state.get("backoff_until")) if state else None
        if backoff_until is not None and ensure_utc(backoff_until) <= now:
            self.ledger.upsert_scheduler_state(account.id, backoff_until=None)
            backoff_until = None
        if backoff_until is not None and not force:
            due_at = ensure_utc(backoff_until)
            return ScheduleDecision(False, 0, due_at, jitter_max, False, "backoff", pending)
        if state is None or not state.get("next_due_at"):
            if force:
                return ScheduleDecision(True, 0, now, jitter_max, bool(pending), "forced", pending)
            return ScheduleDecision(True, 0, now, jitter_max, bool(pending), "uninitialized", pending)
        next_due = ensure_utc(parse_datetime(state["next_due_at"]) or now)
        if now < next_due and not force:
            return ScheduleDecision(False, 0, next_due, jitter_max, False, "not_due", pending)
        # Count whole intervals after the stored due time. The current slot is due,
        # not an extra miss: due at T-6h with PT1H and now=T records six missed hours.
        missed = 0
        interval_seconds = interval.total_seconds()
        if now >= next_due and interval_seconds > 0:
            missed = int((now - next_due).total_seconds() // interval_seconds)
        coalesced = missed > 0 or bool(pending)
        if force and now < next_due:
            return ScheduleDecision(True, 0, next_due, jitter_max, bool(pending), "forced", pending)
        return ScheduleDecision(True, missed, next_due, jitter_max, coalesced, "due", pending)

    def acquire_lease(
        self,
        account: AccountConfig,
        *,
        ttl: timedelta | None = None,
        force: bool = False,
    ) -> str:
        now = ensure_utc(self.clock())
        ttl = ttl or self.lease_ttl
        state = self.ledger.get_scheduler_state(account.id) or {}
        lease_until = parse_datetime(state.get("lease_until"))
        lease_owner = state.get("lease_owner")
        lease_token = state.get("lease_token")
        backoff_until = parse_datetime(state.get("backoff_until"))
        if backoff_until is not None and ensure_utc(backoff_until) > now and not force:
            raise LeaseHeldError(f"backoff active until {backoff_until}")
        active = lease_until is not None and ensure_utc(lease_until) > now and bool(lease_token)
        if active:
            # Owner string is not a fence. Any other worker, including the same owner
            # name, must coalesce rather than steal or dual-run.
            self.ledger.enqueue_pending_work(
                account.id,
                {"kind": "coalesce", "mode": "refresh", "requested_at": isoformat_utc(now)},
            )
            raise LeaseHeldError(f"lease held by {lease_owner}")
        token = str(uuid4())
        with self.ledger.transaction():
            self.ledger.upsert_scheduler_state(
                account.id,
                refresh_interval=format_iso_duration(account.scheduler.refresh_interval),
            )
            current = self.ledger.get_scheduler_state(account.id) or {}
            current_until = parse_datetime(current.get("lease_until"))
            current_token = current.get("lease_token")
            if current_until is not None and ensure_utc(current_until) > now and current_token:
                self.ledger.enqueue_pending_work(
                    account.id,
                    {"kind": "coalesce", "mode": "refresh", "requested_at": isoformat_utc(now)},
                )
                raise LeaseHeldError(f"lease held by {current.get('lease_owner')}")
            claimed = self.ledger.compare_and_set_lease(
                account.id,
                expected_token=current_token,
                owner=self.owner,
                token=token,
                lease_until=now + ttl,
                heartbeat_at=now,
            )
            if not claimed:
                claimed = self.ledger.compare_and_set_lease(
                    account.id,
                    expected_token=None,
                    owner=self.owner,
                    token=token,
                    lease_until=now + ttl,
                    heartbeat_at=now,
                )
            if not claimed:
                self.ledger.enqueue_pending_work(
                    account.id,
                    {"kind": "coalesce", "mode": "refresh", "requested_at": isoformat_utc(now)},
                )
                raise LeaseHeldError("lease claim lost")
        return token

    def heartbeat(self, account: AccountConfig, token: str, *, ttl: timedelta | None = None) -> bool:
        now = ensure_utc(self.clock())
        ttl = ttl or self.lease_ttl
        with self.ledger.transaction():
            return self.ledger.heartbeat_lease(
                account.id,
                token,
                lease_until=now + ttl,
                heartbeat_at=now,
            )

    def release_lease(self, account: AccountConfig, token: str) -> None:
        with self.ledger.transaction():
            self.ledger.release_lease(account.id, token)

    def set_interval(self, account: AccountConfig, interval: timedelta) -> dict[str, object]:
        if interval.total_seconds() < 300:
            raise ValueError("refresh interval must be at least PT5M")
        now = ensure_utc(self.clock())
        state = self.ledger.get_scheduler_state(account.id) or {}
        if state.get("lease_token"):
            lease_until = parse_datetime(state.get("lease_until"))
            if lease_until is not None and ensure_utc(lease_until) > now:
                raise LeaseHeldError("refusing interval change during an overlapping run")
        anchor = parse_datetime(state.get("schedule_anchor_at")) or parse_datetime(state.get("next_due_at")) or now
        anchor = ensure_utc(anchor)
        jitter = self._recorded_jitter(account, state)
        next_due = _next_aligned_due(anchor, interval, now) + timedelta(seconds=jitter)
        with self.ledger.transaction():
            self.ledger.upsert_scheduler_state(
                account.id,
                refresh_interval=format_iso_duration(interval),
                schedule_anchor_at=isoformat_utc(anchor),
                next_due_at=isoformat_utc(next_due),
                last_jitter_seconds=jitter,
            )
        return {
            "account_id": account.id,
            "refresh_interval": format_iso_duration(interval),
            "schedule_anchor_at": isoformat_utc(anchor),
            "next_due_at": isoformat_utc(next_due),
            "jitter_seconds": jitter,
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
        try:
            token = self.acquire_lease(account, force=force)
        except LeaseHeldError:
            self.ledger.enqueue_pending_work(
                account.id,
                {
                    "kind": "coalesce",
                    "mode": mode,
                    "since": isoformat_utc(since),
                    "requested_at": isoformat_utc(ensure_utc(self.clock())),
                },
            )
            return None
        now = ensure_utc(self.clock())
        state = self.ledger.get_scheduler_state(account.id) or {}
        jitter = self._recorded_jitter(account, state, refresh=True)
        try:
            if not self.heartbeat(account, token):
                raise LeaseHeldError("lost lease before collect")
            run_mode = "catch_up" if decision.coalesced else mode
            result = self.collector.collect(
                account_id=account.id,
                mode=run_mode,
                since=since,
                missed_intervals=decision.missed_intervals,
                scheduled_for=decision.next_due_at,
            )
            interval = _interval_from_state(account, state)
            anchor = parse_datetime(state.get("schedule_anchor_at")) or decision.next_due_at
            anchor = ensure_utc(anchor)
            # Next due is the next aligned slot after the scheduled instant, plus this run's jitter.
            next_due = _next_aligned_due(anchor, interval, max(now, decision.next_due_at)) + timedelta(seconds=jitter)
            with self.ledger.transaction():
                if not self.ledger.heartbeat_lease(
                    account.id,
                    token,
                    lease_until=now + self.lease_ttl,
                    heartbeat_at=ensure_utc(self.clock()),
                ):
                    raise LeaseHeldError("stale worker cannot overwrite newer scheduler state")
                self.ledger.upsert_scheduler_state(
                    account.id,
                    refresh_interval=format_iso_duration(interval),
                    schedule_anchor_at=isoformat_utc(anchor),
                    next_due_at=isoformat_utc(next_due),
                    last_started_at=isoformat_utc(now),
                    last_finished_at=isoformat_utc(ensure_utc(self.clock())),
                    missed_intervals=decision.missed_intervals,
                    last_jitter_seconds=jitter,
                    backoff_until=None,
                    pending_work_json=None,
                )
            result.missed_intervals = decision.missed_intervals
            return result
        finally:
            self.release_lease(account, token)

    def _recorded_jitter(
        self,
        account: AccountConfig,
        state: dict,
        *,
        refresh: bool = False,
    ) -> int:
        maximum = max(0, int(account.scheduler.jitter_seconds))
        stored = state.get("last_jitter_seconds")
        if not refresh and stored is not None:
            return max(0, min(int(stored), maximum))
        return int(self.jitter_fn(maximum))

    def _token_matches(self, token: object, state: dict) -> bool:
        stored = state.get("lease_token")
        return stored is not None and str(stored) == str(token)


def _interval_from_state(account: AccountConfig, state: dict) -> timedelta:
    raw = state.get("refresh_interval")
    if raw:
        try:
            return parse_iso_duration(str(raw))
        except Exception:
            pass
    return account.scheduler.refresh_interval


def _next_aligned_due(anchor: datetime, interval: timedelta, after: datetime) -> datetime:
    """Return the next slot at or after `after`, aligned to an explicit UTC anchor."""
    anchor = ensure_utc(anchor)
    after = ensure_utc(after)
    seconds = interval.total_seconds()
    if seconds <= 0:
        raise ValueError("refresh interval must be positive")
    if after <= anchor:
        return anchor
    elapsed = (after - anchor).total_seconds()
    steps = floor(elapsed / seconds) + 1
    return anchor + timedelta(seconds=steps * seconds)
