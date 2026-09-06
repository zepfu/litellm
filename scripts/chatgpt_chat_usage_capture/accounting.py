"""Reset-aware accounting, rebuild, and window evaluation for Chat usage capture."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from .config import CollectorConfig, QuotaBucket, QuotaPolicy, WINDOW_TYPES
from .ledger import Ledger
from .privacy import SURFACE_CHAT
from .timeutil import (
    calendar_day_bounds,
    ensure_utc,
    format_iso_duration,
    interval_membership,
    isoformat_utc,
    iter_calendar_days,
    parse_datetime,
    parse_iso_duration,
)

WORKING_ESTIMATOR = "requested_if_known_else_recorded_final"


class AccountingError(ValueError):
    """Invalid accounting or window operation."""


def evaluate_window(
    bucket: QuotaBucket,
    stored: Optional[Mapping[str, Any]],
    *,
    now: datetime,
    report_end: Optional[datetime] = None,
) -> dict[str, Any]:
    """Resolve a half-open [start, end) window without inventing unknown bounds."""
    now = ensure_utc(now)
    as_of = ensure_utc(report_end) if report_end is not None else now
    window_type = stored["window_type"] if stored and stored.get("window_type") else bucket.window.type
    if window_type not in WINDOW_TYPES:
        raise AccountingError(f"unsupported window type: {window_type}")
    timezone_name = (
        (stored.get("timezone") if stored else None)
        or bucket.window.timezone
        or "UTC"
    )
    duration = None
    raw_duration = (stored.get("duration") if stored else None) or (
        format_iso_duration(bucket.window.duration) if bucket.window.duration else None
    )
    if raw_duration:
        duration = parse_iso_duration(str(raw_duration))
    start = parse_datetime(stored.get("start_at") if stored else None) or bucket.window.start
    end = parse_datetime(stored.get("end_at") if stored else None) or bucket.window.end
    evidence = (stored.get("evidence") if stored else None) or bucket.window.evidence or "unknown"
    reason = (stored.get("reason") if stored else None) or bucket.window.reason

    if window_type == "unknown":
        start, end = None, None
    elif window_type in {"provider_explicit", "operator_explicit"}:
        pass
    elif window_type == "anchored_elapsed":
        if start is None or duration is None:
            start, end = None, None
        else:
            start = ensure_utc(start)
            elapsed = (as_of - start).total_seconds()
            if elapsed < 0:
                end = start + duration
            else:
                steps = int(elapsed // duration.total_seconds())
                start = start + timedelta(seconds=steps * duration.total_seconds())
                end = start + duration
    elif window_type == "calendar":
        zone = ZoneInfo(timezone_name)
        local = as_of.astimezone(zone)
        hint = (bucket.documented_period_hint or "day").lower()
        if hint in {"week", "weekly"}:
            weekday = local.weekday()  # Monday=0
            start_local = local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=weekday)
            end_local = start_local + timedelta(days=7)
        else:
            start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
            end_local = start_local + timedelta(days=1)
        start = start_local.astimezone(timezone.utc)
        end = end_local.astimezone(timezone.utc)
    elif window_type == "rolling_elapsed":
        if duration is None:
            start, end = None, None
        else:
            end = as_of
            start = as_of - duration

    known = start is not None and end is not None and window_type != "unknown"
    return {
        "type": window_type,
        "start": start,
        "end": end,
        "timezone": timezone_name,
        "duration": format_iso_duration(duration) if duration else None,
        "evidence": evidence,
        "reason": reason,
        "known": known,
    }


def working_contribution(attempt: Mapping[str, Any], config: CollectorConfig) -> Optional[str]:
    if attempt.get("surface") != SURFACE_CHAT:
        return None
    if attempt.get("origin") in {"shared", "imported", "copied"}:
        return None
    if attempt.get("identity_basis") == "unresolved":
        return None
    if attempt.get("outcome") == "rejected_before_start":
        return None
    if int(attempt.get("tombstone") or 0):
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


def attempt_membership(
    attempt: Mapping[str, Any], start: Optional[datetime], end: Optional[datetime]
) -> str:
    return interval_membership(
        start,
        end,
        parse_datetime(attempt.get("attempt_time")),
        parse_datetime(attempt.get("earliest_possible_at")),
        parse_datetime(attempt.get("latest_possible_at")),
    )


def bucket_usage(bucket: QuotaBucket, working: Counter[str], in_range: list[Mapping[str, Any]], config: CollectorConfig) -> int:
    """Count unique eligible attempts once per shared bucket (union_once_per_attempt)."""
    families = set(bucket.families)
    if bucket.membership != "union_once_per_attempt":
        return sum(count for family, count in working.items() if family in families)
    counted = 0
    for attempt in in_range:
        contribution = working_contribution(attempt, config)
        if contribution in families:
            counted += 1
    return counted


def summarize_attempts(
    attempts: list[Mapping[str, Any]],
    config: CollectorConfig,
    *,
    start: Optional[datetime],
    end: Optional[datetime],
) -> dict[str, Any]:
    in_range: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unknown_window = start is None and end is None
    for attempt in attempts:
        if int(attempt.get("tombstone") or 0):
            continue
        membership = attempt_membership(attempt, start, end)
        if unknown_window or membership == "in":
            in_range.append(dict(attempt))
        elif membership == "ambiguous":
            ambiguous.append(dict(attempt))
        elif membership == "unknown" and (start is None or end is None):
            ambiguous.append(dict(attempt))

    observed_requested: Counter[str] = Counter()
    completed_final: Counter[str] = Counter()
    mismatches = 0
    unclassified = 0
    working: Counter[str] = Counter()
    unknown_debit = 0
    excluded_surface = 0
    excluded_origin = 0
    for attempt in in_range:
        if attempt.get("surface") != SURFACE_CHAT:
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
        contribution = working_contribution(attempt, config)
        if contribution is None:
            if attempt.get("generation_started") and attempt.get("outcome") not in {
                "rejected_before_start"
            }:
                unknown_debit += 1
            continue
        working[contribution] += 1
    return {
        "in_range": in_range,
        "ambiguous": ambiguous,
        "observed_requested": observed_requested,
        "completed_final": completed_final,
        "mismatches": mismatches,
        "unclassified": unclassified,
        "working": working,
        "unknown_debit": unknown_debit,
        "excluded_surface": excluded_surface,
        "excluded_origin": excluded_origin,
    }


def bucket_view(
    ledger: Ledger,
    config: CollectorConfig,
    account_id: str,
    policy: QuotaPolicy,
    bucket: QuotaBucket,
    *,
    evaluated_at: datetime,
    summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    stored = ledger.get_window(account_id, bucket.id)
    window = evaluate_window(bucket, stored, now=evaluated_at)
    if summary is None:
        attempts = ledger.list_attempts(account_id)
        summary = summarize_attempts(
            attempts,
            config,
            start=window["start"] if window["known"] else None,
            end=window["end"] if window["known"] else None,
        )
    usage = bucket_usage(bucket, summary["working"], summary["in_range"], config)
    remaining_estimate = None
    remaining_unclamped = None
    if window["known"] and bucket.capacity is not None:
        remaining_unclamped = bucket.capacity - usage
        remaining_estimate = max(0, remaining_unclamped)
    server = ledger.latest_quota_observation(account_id, bucket.id)
    server_remaining = server.get("remaining") if server is not None else None
    return {
        "account_id": account_id,
        "surface": SURFACE_CHAT,
        "bucket_id": bucket.id,
        "policy_version": policy.id,
        "window": {
            "type": window["type"],
            "start": isoformat_utc(window["start"]),
            "end": isoformat_utc(window["end"]),
            "evidence": window["evidence"],
            "reason": window.get("reason"),
        },
        "capacity": bucket.capacity,
        "working_usage_estimate": usage if window["known"] else None,
        "working_remaining_estimate": remaining_estimate,
        "working_remaining_unclamped": remaining_unclamped,
        "server_reported_remaining": server_remaining,
        "unknown_debit_attempts": summary["unknown_debit"],
        "unclassified_attempts": summary["unclassified"] + len(summary["ambiguous"]),
        "label": "Working estimate; not an official remaining quota",
        "evaluated_at": isoformat_utc(evaluated_at),
        "activity_only": not window["known"],
    }


def calendar_table(
    attempts: list[Mapping[str, Any]],
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
            membership = attempt_membership(attempt, day_start, day_end)
            if membership != "in":
                if membership == "ambiguous":
                    unclassified += 1
                continue
            requested[attempt.get("requested_family") or "unknown"] += 1
            if attempt.get("completed_answer"):
                recorded[attempt.get("recorded_final_family") or "unknown"] += 1
            family = working_contribution(attempt, config)
            if family:
                working[family] += 1
            elif attempt.get("surface") == SURFACE_CHAT:
                unclassified += 1
        local_date = ensure_utc(day_start).astimezone(ZoneInfo(timezone_name)).date().isoformat()
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


def freshness(
    ledger: Ledger,
    account_id: str,
    evaluated_at: datetime,
    interval: timedelta,
    *,
    alerts: Any = None,
) -> dict[str, Any]:
    state = ledger.get_scheduler_state(account_id) or {}
    last = parse_datetime(state.get("last_finished_at"))
    soft_mult = getattr(alerts, "stale_soft_multiplier", 2) or 2
    hard_mult = getattr(alerts, "stale_hard_multiplier", 6) or 6
    grace = timedelta(minutes=int(getattr(alerts, "stale_grace_minutes", 5) or 5))
    status = "unknown" if last is None else "fresh"
    if last is not None:
        age = evaluated_at - ensure_utc(last)
        if age > (interval * hard_mult) + grace:
            status = "hard_stale"
        elif age > (interval * soft_mult) + grace:
            status = "soft_stale"
    return {
        "history_last_completed_at": isoformat_utc(last),
        "evaluated_at": isoformat_utc(evaluated_at),
        "next_due_at": state.get("next_due_at"),
        "status": status,
        "missed_intervals": int(state.get("missed_intervals") or 0),
        "schedule_anchor_at": state.get("schedule_anchor_at"),
        "backoff_until": state.get("backoff_until"),
        "lease_owner": state.get("lease_owner"),
        "pending_work": json_pending(state.get("pending_work_json")),
    }


def json_pending(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    import json

    payload = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def rebuild_aggregates(
    config: CollectorConfig,
    ledger: Ledger,
    *,
    account_id: Optional[str] = None,
    apply: bool = False,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Deterministically rebuild daily aggregates and publish a revision only after commit."""
    from .reporting import build_report

    account = config.account(account_id)
    evaluated_at = ensure_utc(now or datetime.now(timezone.utc))
    preview = build_report(config, ledger, account_id=account.id, now=evaluated_at)
    payload = {
        "account_id": account.id,
        "policy_id": account.plan_policy_id,
        "mapping_version": config.model_mapping.version,
        "evaluated_at": isoformat_utc(evaluated_at),
        "calendar_days": preview.get("calendar_days") or [],
        "quota_buckets": preview.get("quota_buckets") or [],
        "working_quota_usage_estimate_by_bucket": preview.get("working_quota_usage_estimate_by_bucket") or {},
        "observed_attempts_by_requested_family": preview.get("observed_attempts_by_requested_family") or {},
        "label": "Working estimate; not an official remaining quota",
    }
    warnings: list[str] = []
    observation_count = ledger.conn.execute(
        "SELECT COUNT(*) AS n FROM observations WHERE collector_account_id=?",
        (account.id,),
    ).fetchone()["n"]
    if observation_count == 0:
        warnings.append("raw observations absent; rebuild uses retained attempt projections only")
        payload["rebuild_source"] = "retained_attempts"
    else:
        payload["rebuild_source"] = "ledger"
    result = {
        "account_id": account.id,
        "dry_run": not apply,
        "revision_id": None,
        "payload": payload,
        "warnings": warnings,
    }
    if not apply:
        return result
    revision_id = str(uuid4())
    with ledger.transaction():
        for row in payload["calendar_days"]:
            ledger.upsert_daily_aggregate(
                account_id=account.id,
                local_date=row["date"],
                timezone_name=config.application.report_timezone,
                payload=row,
                created_at=evaluated_at,
                revision_id=revision_id,
            )
        ledger.insert_aggregate_revision(
            revision_id=revision_id,
            account_id=account.id,
            created_at=evaluated_at,
            policy_id=account.plan_policy_id,
            mapping_version=config.model_mapping.version,
            payload=payload,
        )
    result["revision_id"] = revision_id
    result["published"] = True
    return result


def set_explicit_window(
    ledger: Ledger,
    *,
    account_id: str,
    bucket_id: str,
    start: datetime,
    end: Optional[datetime],
    evidence: str,
    reason: str,
    window_type: str = "operator_explicit",
    timezone_name: Optional[str] = None,
    duration: Optional[str] = None,
) -> dict[str, Any]:
    if window_type not in WINDOW_TYPES:
        raise AccountingError(f"unsupported window type: {window_type}")
    if not evidence or not reason:
        raise AccountingError("manual window set requires evidence and reason")
    if window_type in {"provider_explicit", "operator_explicit"} and start is None:
        raise AccountingError("explicit windows require a start bound")
    with ledger.transaction():
        ledger.set_window(
            account_id,
            bucket_id,
            window_type=window_type,
            start=start,
            end=end,
            timezone_name=timezone_name,
            duration=duration,
            evidence=evidence,
            reason=reason,
        )
    stored = ledger.get_window(account_id, bucket_id)
    return stored or {}


def record_manual_observation(
    ledger: Ledger,
    *,
    account_id: str,
    bucket_id: str,
    remaining: Optional[int],
    capacity: Optional[int],
    observed_at: datetime,
    source: str,
    reset_at: Optional[datetime] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Record an operator snapshot without implying a window start."""
    observation_id = str(uuid4())
    ledger.record_quota_observation(
        observation_id=observation_id,
        account_id=account_id,
        bucket_id=bucket_id,
        source=source,
        remaining=remaining,
        capacity=capacity,
        observed_at=observed_at,
        reset_at=reset_at,
        window_start=None,
        raw_type="operator_snapshot",
        notes=notes,
    )
    return ledger.latest_quota_observation(account_id, bucket_id) or {}
