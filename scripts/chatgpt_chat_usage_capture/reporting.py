"""Independent observed, local-estimate, and server-reported Chat usage views."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from .config import CollectorConfig, QuotaBucket, QuotaPolicy
from .ledger import Ledger
from .privacy import SURFACE_CHAT
from .timeutil import (
    calendar_day_bounds,
    ensure_utc,
    interval_membership,
    isoformat_utc,
    iter_calendar_days,
    parse_datetime,
)

WORKING_ESTIMATOR = "requested_if_known_else_recorded_final"


def build_report(
    config: CollectorConfig,
    ledger: Ledger,
    *,
    account_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    lookback: Optional[timedelta] = None,
    window_bucket: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    account = config.account(account_id)
    policy = config.policy_for(account)
    evaluated_at = ensure_utc(now or datetime.now(timezone.utc))
    report_end = ensure_utc(end) if end is not None else evaluated_at
    if start is not None:
        report_start: Optional[datetime] = ensure_utc(start)
        range_kind = "absolute"
    elif lookback is not None:
        report_start = report_end - lookback
        range_kind = "elapsed"
    elif window_bucket:
        report_start, report_end, range_kind = _window_bounds(
            ledger, account.id, policy, window_bucket, evaluated_at
        )
    else:
        report_start = report_end - config.default_lookback
        range_kind = "elapsed_default"

    attempts = ledger.list_attempts(account.id)
    in_range: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for attempt in attempts:
        membership = _attempt_membership(attempt, report_start, report_end)
        if membership == "in":
            in_range.append(attempt)
        elif membership == "ambiguous":
            ambiguous.append(attempt)

    observed_requested: Counter[str] = Counter()
    completed_final: Counter[str] = Counter()
    mismatches = 0
    unclassified = 0
    working: Counter[str] = Counter()
    unknown_debit = 0
    excluded_surface = 0
    excluded_origin = 0
    for attempt in in_range:
        if attempt["surface"] != SURFACE_CHAT:
            excluded_surface += 1
            unclassified += 1
            continue
        if attempt.get("origin") in {"shared", "imported", "copied"}:
            excluded_origin += 1
            unclassified += 1
            continue
        requested_family = attempt.get("requested_family") or "unknown"
        recorded_family = attempt.get("recorded_final_family") or "unknown"
        observed_requested[requested_family] += 1
        if attempt.get("completed_answer"):
            completed_final[recorded_family] += 1
        if (
            attempt.get("requested_family")
            and attempt.get("recorded_final_family")
            and attempt["requested_family"] != attempt["recorded_final_family"]
        ):
            mismatches += 1
        if attempt.get("identity_basis") == "unresolved" or attempt.get("outcome") in {
            "completion_unknown",
            "unresolved",
        }:
            unclassified += 1
        contribution = _working_contribution(attempt, config)
        if contribution is None:
            if attempt.get("generation_started") and attempt.get("outcome") not in {
                "rejected_before_start"
            }:
                unknown_debit += 1
            continue
        working[contribution] += 1

    calendar = _calendar_table(
        in_range,
        report_start,
        report_end,
        config.application.report_timezone,
        config,
    )
    buckets = [
        _bucket_view(
            ledger,
            account.id,
            policy,
            bucket,
            working,
            evaluated_at,
            unknown_debit=unknown_debit,
            unclassified=unclassified,
        )
        for bucket in policy.buckets
    ]
    freshness = _freshness(ledger, account.id, evaluated_at)
    return {
        "account_id": account.id,
        "surface": SURFACE_CHAT,
        "range": {
            "kind": range_kind,
            "start": isoformat_utc(report_start),
            "end": isoformat_utc(report_end),
            "timezone": config.application.report_timezone,
            "label": _range_label(range_kind, lookback, window_bucket),
        },
        "observed_attempts_by_requested_family": dict(observed_requested),
        "completed_answers_by_recorded_final_family": dict(completed_final),
        "observed_model_mismatches": mismatches,
        "unclassified_or_ambiguous_attempts": unclassified + len(ambiguous),
        "working_quota_usage_estimate_by_bucket": {
            bucket.id: _bucket_usage(bucket, working) for bucket in policy.buckets
        },
        "unknown_debit_attempts": unknown_debit,
        "excluded_non_chat_attempts": excluded_surface,
        "excluded_imported_or_shared_attempts": excluded_origin,
        "ambiguous_window_attempts": len(ambiguous),
        "calendar_days": calendar,
        "quota_buckets": buckets,
        "freshness": freshness,
        "label": "Working estimate; not an official remaining quota",
        "estimator": config.accounting.working_estimator,
    }


def _range_label(kind: str, lookback: Optional[timedelta], window_bucket: Optional[str]) -> str:
    if kind.startswith("elapsed") and lookback is not None:
        hours = int(lookback.total_seconds() // 3600)
        days = int(lookback.total_seconds() // 86400)
        if lookback == timedelta(days=7) or days == 7:
            return "last_seven_days_elapsed_or_default"
        if hours == 24:
            return "last_24_hours_elapsed"
        return "elapsed_lookback"
    if window_bucket:
        return f"quota_window:{window_bucket}"
    if kind == "absolute":
        return "absolute_range"
    return "default_lookback"


def _attempt_membership(
    attempt: Mapping[str, Any], start: Optional[datetime], end: Optional[datetime]
) -> str:
    return interval_membership(
        start,
        end,
        parse_datetime(attempt.get("attempt_time")),
        parse_datetime(attempt.get("earliest_possible_at")),
        parse_datetime(attempt.get("latest_possible_at")),
    )


def _working_contribution(attempt: Mapping[str, Any], config: CollectorConfig) -> Optional[str]:
    if attempt.get("surface") != SURFACE_CHAT:
        return None
    if attempt.get("origin") in {"shared", "imported", "copied"}:
        return None
    if attempt.get("identity_basis") == "unresolved":
        return None
    if attempt.get("outcome") == "rejected_before_start":
        return None
    if config.accounting.count_only_generation_started_or_completed and not (
        attempt.get("generation_started") or attempt.get("completed_answer")
    ):
        return None
    if attempt.get("outcome") in {"failed_after_start", "cancelled_after_start", "completion_unknown"}:
        if not config.accounting.uncertain_attempts_in_working_estimate:
            return None
    requested = attempt.get("requested_family")
    recorded = attempt.get("recorded_final_family")
    if requested and requested != "unknown":
        return requested
    if recorded and recorded != "unknown":
        return recorded
    return None


def _bucket_usage(bucket: QuotaBucket, working: Counter) -> int:
    families = set(bucket.families)
    return sum(count for family, count in working.items() if family in families)


def _bucket_view(
    ledger: Ledger,
    account_id: str,
    policy: QuotaPolicy,
    bucket: QuotaBucket,
    working: Counter,
    evaluated_at: datetime,
    *,
    unknown_debit: int,
    unclassified: int,
) -> dict[str, Any]:
    stored = ledger.get_window(account_id, bucket.id)
    window_type = stored["window_type"] if stored else bucket.window.type
    start = parse_datetime(stored["start_at"]) if stored else bucket.window.start
    end = parse_datetime(stored["end_at"]) if stored else bucket.window.end
    evidence = stored["evidence"] if stored else bucket.window.evidence
    usage = _bucket_usage(bucket, working)
    window_known = start is not None and end is not None and window_type != "unknown"
    remaining_estimate = None
    remaining_unclamped = None
    if window_known and bucket.capacity is not None:
        remaining_unclamped = bucket.capacity - usage
        remaining_estimate = max(0, remaining_unclamped)
    elif not window_known:
        remaining_estimate = None
        remaining_unclamped = None
    server = ledger.latest_quota_observation(account_id, bucket.id)
    server_remaining = None
    if server is not None:
        server_remaining = server.get("remaining")
    return {
        "account_id": account_id,
        "surface": SURFACE_CHAT,
        "bucket_id": bucket.id,
        "policy_version": policy.id,
        "window": {
            "type": window_type,
            "start": isoformat_utc(start),
            "end": isoformat_utc(end),
            "evidence": evidence or "unknown",
        },
        "capacity": bucket.capacity,
        "working_usage_estimate": usage if window_known else None,
        "working_remaining_estimate": remaining_estimate,
        "working_remaining_unclamped": remaining_unclamped,
        "server_reported_remaining": server_remaining,
        "unknown_debit_attempts": unknown_debit,
        "unclassified_attempts": unclassified,
        "label": "Working estimate; not an official remaining quota",
        "evaluated_at": isoformat_utc(evaluated_at),
    }


def _window_bounds(
    ledger: Ledger,
    account_id: str,
    policy: QuotaPolicy,
    bucket_id: str,
    now: datetime,
) -> tuple[Optional[datetime], datetime, str]:
    stored = ledger.get_window(account_id, bucket_id)
    bucket = next((item for item in policy.buckets if item.id == bucket_id), None)
    start = parse_datetime(stored["start_at"]) if stored else (bucket.window.start if bucket else None)
    end = parse_datetime(stored["end_at"]) if stored else (bucket.window.end if bucket else None)
    if start is None and end is None:
        return None, now, "unknown_window"
    return start, end or now, "quota_window"


def _calendar_table(
    attempts: list[dict[str, Any]],
    start: Optional[datetime],
    end: datetime,
    timezone_name: str,
    config: CollectorConfig,
) -> list[dict[str, Any]]:
    if start is None:
        return []
    rows = []
    for day_start in iter_calendar_days(start, end, timezone_name):
        day_end = calendar_day_bounds(day_start, timezone_name)[1]
        requested: Counter[str] = Counter()
        recorded: Counter[str] = Counter()
        working: Counter[str] = Counter()
        unclassified = 0
        for attempt in attempts:
            membership = _attempt_membership(attempt, day_start, day_end)
            if membership != "in":
                if membership == "ambiguous":
                    unclassified += 1
                continue
            requested[attempt.get("requested_family") or "unknown"] += 1
            if attempt.get("completed_answer"):
                recorded[attempt.get("recorded_final_family") or "unknown"] += 1
            family = _working_contribution(attempt, config)
            if family:
                working[family] += 1
            elif attempt.get("surface") == SURFACE_CHAT:
                unclassified += 1
        local_date = ensure_utc(day_start).astimezone(
            __import__("zoneinfo").ZoneInfo(timezone_name)
        ).date().isoformat()
        rows.append(
            {
                "date": local_date,
                "start": isoformat_utc(day_start),
                "end": isoformat_utc(day_end),
                "observed_attempts_by_requested_family": dict(requested),
                "completed_answers_by_recorded_final_family": dict(recorded),
                "working_families": dict(working),
                "unclassified_or_ambiguous_attempts": unclassified,
            }
        )
    return rows


def _freshness(ledger: Ledger, account_id: str, evaluated_at: datetime) -> dict[str, Any]:
    state = ledger.get_scheduler_state(account_id) or {}
    last = parse_datetime(state.get("last_finished_at"))
    return {
        "history_last_completed_at": isoformat_utc(last),
        "evaluated_at": isoformat_utc(evaluated_at),
        "next_due_at": state.get("next_due_at"),
        "status": "unknown" if last is None else "fresh",
        "missed_intervals": int(state.get("missed_intervals") or 0),
    }
