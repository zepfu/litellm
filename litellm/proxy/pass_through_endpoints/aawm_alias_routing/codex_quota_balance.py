"""Bounded, detached evidence for managed Codex account selection."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence

POLL_SOURCE = "codex_quota_poll"
_PERIOD_MINUTES = {"five_hour": 300, "seven_day": 10080}


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _period(row: Mapping[str, Any]) -> Optional[str]:
    period = str(row.get("quota_period") or "").strip().lower()
    minutes = _number(row.get("window_minutes"))
    if period in _PERIOD_MINUTES:
        return period
    return next(
        (name for name, duration in _PERIOD_MINUTES.items() if minutes == duration),
        None,
    )


def _window(row: Mapping[str, Any], *, now: float, horizon: float) -> dict[str, Any]:
    period = _period(row)
    minutes = _number(row.get("window_minutes"))
    raw_period = str(row.get("quota_period") or "").strip().lower()
    remaining = _number(row.get("remaining_pct"))
    observed_at = _number(row.get("observed_at"))
    reset_at = _number(row.get("expected_reset_at"))
    age = now - observed_at if observed_at is not None else None
    reason: Optional[str] = None
    if (
        period is None
        or (raw_period and raw_period != period)
        or (
            row.get("window_minutes") is not None
            and minutes != _PERIOD_MINUTES.get(period)
        )
    ):
        reason = "invalid_window"
    elif row.get("status") != "fresh":
        reason = "observation_not_fresh"
    elif remaining is None or not 0 <= remaining <= 100:
        reason = "invalid_remaining_pct"
    elif age is None or age < 0 or age > horizon:
        reason = "invalid_observation_age"
    elif reset_at is None:
        reason = "missing_reset"
    elif reset_at <= now:
        reason = "expired_reset"
    elif observed_at is not None and (
        reset_at - observed_at > _PERIOD_MINUTES[period] * 60
    ):
        reason = "invalid_reset_window"
    return {
        "quota_period": period,
        "window_minutes": minutes,
        "remaining_pct": remaining,
        "observed_at": observed_at,
        "observation_age_seconds": age,
        "expected_reset_at": reset_at,
        "source": row.get("source"),
        "environment": row.get("environment"),
        "quota_key": row.get("quota_key"),
        "model": row.get("model"),
        "quota_family": row.get("quota_family"),
        "status": row.get("status"),
        # Missing reset prevents ranking, not an otherwise fresh confirmed zero.
        "exhausted": remaining == 0 and reason in {None, "missing_reset"},
        "unusable_reason": reason,
    }


def account_evidence(
    observations: Sequence[Mapping[str, Any]],
    *,
    family: Optional[str],
    environment: Optional[str],
    runtime_environment: Optional[str],
    now: float,
    horizon: float,
) -> dict[str, Any]:
    """Resolve current logical windows before judging their usability.

    Model/quota-key aliases of one account window are not independent capacity.
    A newer stale or invalid row must shadow older usable comparison evidence.
    Local response observations can retain hard exclusions, but only the shared
    poll scope supplies cross-runtime weekly ranking.
    """
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in observations:
        if family is not None and row.get("quota_family") != family:
            continue
        scope = str(row.get("environment") or "")
        source = str(row.get("source") or "")
        if not scope or scope not in {environment, runtime_environment}:
            continue
        if source == POLL_SOURCE and scope != (environment or runtime_environment):
            continue
        if not source:
            continue
        period = _period(row)
        if period is None:
            continue
        key = (scope, source, str(row.get("quota_family") or ""), period)
        window = _window(row, now=now, horizon=horizon)
        previous = latest.get(key)
        observed_at = window["observed_at"]
        previous_at = previous.get("observed_at") if previous else None
        if (
            previous is None
            or observed_at is None
            or (previous_at is not None and observed_at > previous_at)
        ):
            latest[key] = window
        elif observed_at == previous_at and any(
            window.get(field) != previous.get(field)
            for field in (
                "remaining_pct",
                "expected_reset_at",
                "status",
                "window_minutes",
            )
        ):
            previous["unusable_reason"] = "conflicting_current_windows"
    windows = list(latest.values())
    weekly = latest.get((environment or "", POLL_SOURCE, family or "", "seven_day"))
    missing_reason = (
        "unknown_quota_family"
        if family is None
        else "missing_observation_scope"
        if environment is None
        else "missing_weekly_observation"
    )
    return {
        "evaluated_at": now,
        "environment": environment,
        "quota_family": family,
        "validity_horizon_seconds": horizon,
        "weekly": weekly,
        "unusable_reason": (
            weekly["unusable_reason"] if weekly is not None else missing_reason
        ),
        "valid_windows": [
            window
            for window in windows
            if window["unusable_reason"] is None or window["exhausted"]
        ],
    }


def decision_account(state: Mapping[str, Any], *, eligible: bool) -> dict[str, Any]:
    """Project only named, nonsecret fields; identities remain redactable."""
    candidate = state["candidate"]
    evidence = state.get("codex_oauth_quota_evidence") or {}
    weekly = evidence.get("weekly")
    return {
        "account_label": candidate.get("codex_oauth_account_label"),
        "account_hash": candidate.get("codex_oauth_account_hash"),
        "eligible": eligible,
        "exclusion_reason": state.get("skip_reason")
        or ("cooldown" if not eligible else None),
        "quota_family": evidence.get("quota_family"),
        "environment": evidence.get("environment"),
        "evaluated_at": evidence.get("evaluated_at"),
        "weekly": dict(weekly) if isinstance(weekly, dict) else None,
        "unusable_reason": evidence.get("unusable_reason", "missing_evidence"),
    }


def snapshot_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze account comparison and outer choice at attempt creation."""
    result = {
        field: deepcopy(selection[field])
        for field in (
            "quota_snapshot_age_seconds",
            "quota_windows",
            "failover_ordinal",
            "prior_account_outcome",
            "terminal_reset",
        )
        if selection.get(field) is not None
    }
    balancing = selection.get("quota_balancing")
    if isinstance(balancing, dict):
        result["quota_balancing"] = deepcopy(balancing)
    diagnostics = selection.get("selection_diagnostics")
    if isinstance(diagnostics, dict):
        result["selection_diagnostics"] = {
            field: deepcopy(diagnostics[field])
            for field in ("strategy", "group", "selected_choice", "reselection_count")
            if field in diagnostics
        }
    return result
