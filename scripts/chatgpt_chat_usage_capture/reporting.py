"""Independent observed, local-estimate, and server-reported Chat usage views."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .accounting import (
    attempt_membership,
    bucket_usage,
    bucket_view,
    calendar_table,
    evaluate_window,
    freshness,
    summarize_attempts,
    working_contribution,
)
from .config import CollectorConfig
from .ledger import Ledger
from .privacy import SURFACE_CHAT
from .timeutil import ensure_utc, isoformat_utc

WORKING_ESTIMATOR = "requested_if_known_else_recorded_final"

# Re-export accounting helpers used by tests and CLI.
_attempt_membership = attempt_membership
_working_contribution = working_contribution


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
    range_kind = "elapsed_default"
    report_start: Optional[datetime]
    if start is not None:
        report_start = ensure_utc(start)
        range_kind = "absolute"
    elif lookback is not None:
        report_start = report_end - lookback
        range_kind = "elapsed"
    elif window_bucket:
        bucket = next((item for item in policy.buckets if item.id == window_bucket), None)
        stored = ledger.get_window(account.id, window_bucket)
        if bucket is None:
            report_start, range_kind = None, "unknown_window"
        else:
            window = evaluate_window(bucket, stored, now=evaluated_at, report_end=report_end)
            report_start = window["start"]
            report_end = window["end"] or report_end
            range_kind = "quota_window" if window["known"] else "unknown_window"
    else:
        report_start = report_end - config.default_lookback
        range_kind = "elapsed_default"

    attempts = ledger.list_attempts(account.id)
    summary_end = report_end
    if end is None and not window_bucket:
        # A live report covers observations through its evaluation instant.
        summary_end += timedelta(microseconds=1)
    summary = summarize_attempts(attempts, config, start=report_start, end=summary_end)
    calendar = calendar_table(
        summary["in_range"],
        report_start,
        report_end,
        config.application.report_timezone,
        config,
    )
    bucket_summaries = []
    working_by_bucket = {}
    for bucket in policy.buckets:
        stored = ledger.get_window(account.id, bucket.id)
        window = evaluate_window(bucket, stored, now=evaluated_at, report_end=evaluated_at)
        window_summary = summarize_attempts(
            attempts,
            config,
            start=window["start"] if window["known"] else None,
            end=window["end"] if window["known"] else None,
        )
        view = bucket_view(
            ledger,
            config,
            account.id,
            policy,
            bucket,
            evaluated_at=evaluated_at,
            summary=window_summary,
        )
        bucket_summaries.append(view)
        usage = bucket_usage(bucket, window_summary["working"], window_summary["in_range"], config)
        working_by_bucket[bucket.id] = usage if window["known"] else None

    interval = account.scheduler.refresh_interval
    fresh = freshness(ledger, account.id, evaluated_at, interval, alerts=config.alerts)
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
        "observed_attempts_by_requested_family": dict(summary["observed_requested"]),
        "completed_answers_by_recorded_final_family": dict(summary["completed_final"]),
        "observed_model_mismatches": summary["mismatches"],
        "unclassified_or_ambiguous_attempts": summary["unclassified"] + len(summary["ambiguous"]),
        "working_quota_usage_estimate_by_bucket": working_by_bucket,
        "unknown_debit_attempts": summary["unknown_debit"],
        "excluded_non_chat_attempts": summary["excluded_surface"],
        "excluded_imported_or_shared_attempts": summary["excluded_origin"],
        "ambiguous_window_attempts": len(summary["ambiguous"]),
        "calendar_days": calendar,
        "quota_buckets": bucket_summaries,
        "freshness": fresh,
        "label": "Working estimate; not an official remaining quota",
        "estimator": config.accounting.working_estimator,
        "activity_only": range_kind == "unknown_window",
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
