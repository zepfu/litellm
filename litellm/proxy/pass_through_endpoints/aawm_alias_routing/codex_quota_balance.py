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


def paid_credit_fields(usage: Mapping[str, Any]) -> dict[str, Any]:
    credits = usage.get("credits")
    if not isinstance(credits, Mapping):
        return {}
    fields = {
        key: credits[key]
        for key in ("has_credits", "unlimited", "overage_limit_reached")
        if isinstance(credits.get(key), bool)
    }
    if _number(credits.get("balance")) is not None:
        fields["balance"] = str(credits["balance"])
    spend_control = usage.get("spend_control")
    if isinstance(spend_control, Mapping):
        if isinstance(spend_control.get("reached"), bool):
            fields["spend_control_reached"] = spend_control["reached"]
        if "individual_limit" in spend_control:
            limit = spend_control["individual_limit"]
            if limit is None or _number(limit) is not None:
                fields["individual_limit"] = limit
    return fields


def alpha_weekly_paid_credit_observation(
    usage: Mapping[str, Any], *, account_hash: str, observed_at: float
) -> Optional[dict[str, Any]]:
    """Normalize an account-bound live usage response for alpha's selector."""
    credits = paid_credit_fields(usage)
    rate_limit = usage.get("rate_limit")
    if not credits or not isinstance(rate_limit, Mapping):
        return None
    weekly_seconds = _PERIOD_MINUTES["seven_day"] * 60
    weeks = [
        window
        for key in ("primary_window", "secondary_window")
        if isinstance(window := rate_limit.get(key), Mapping)
        and _number(window.get("limit_window_seconds")) == weekly_seconds
    ]
    if len(weeks) != 1:
        return None
    used = _number(weeks[0].get("used_percent"))
    reset = _number(weeks[0].get("reset_at"))
    if (
        used is None or not 0 <= used <= 100
        or reset is None or not observed_at < reset <= observed_at + weekly_seconds
    ):
        return None
    return {
        "account_hash": account_hash,
        "environment": "litellm-alpha",
        "quota_family": "overall",
        "quota_period": "seven_day",
        "window_minutes": _PERIOD_MINUTES["seven_day"],
        "remaining_pct": 100 - used,
        "observed_at": observed_at,
        "expected_reset_at": reset,
        "source": POLL_SOURCE,
        "status": "fresh",
        "evidence": {"codex_paid_credits": credits},
    }


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
    elif row.get("status") not in {"fresh", "stale"}:
        reason = "invalid_observation_status"
    elif remaining is None or not 0 <= remaining <= 100:
        reason = "invalid_remaining_pct"
    elif age is None or age < 0:
        reason = "invalid_observation_age"
    elif reset_at is None:
        reason = "missing_reset"
    elif observed_at is not None and (
        reset_at <= observed_at or reset_at - observed_at > _PERIOD_MINUTES[period] * 60
    ):
        reason = "invalid_reset_window"
    evidence = row.get("evidence")
    paid_credits = evidence.get("codex_paid_credits") if isinstance(evidence, Mapping) else None
    window = {
        "quota_period": period,
        "window_minutes": minutes,
        "remaining_pct": remaining,
        "observed_at": observed_at,
        "observation_age_seconds": age,
        "expected_reset_at": reset_at,
        "source": row.get("source"),
        "environment": row.get("environment"),
        "quota_key": row.get("quota_key"),
        "limit_scope": row.get("limit_scope"),
        "quota_type": row.get("quota_type"),
        "model": row.get("model"),
        "quota_family": row.get("quota_family"),
        "status": row.get("status"),
        # Historical quota can rank accounts, but cannot impose a hard block.
        "exhausted": (
            remaining == 0
            and reason in {None, "missing_reset"}
            and row.get("status") == "fresh"
            and age is not None
            and age <= horizon
            and (reset_at is None or reset_at > now)
        ),
        "unusable_reason": reason,
    }
    if isinstance(paid_credits, Mapping):
        window["paid_credits"] = dict(paid_credits)
    return window


def account_evidence(
    observations: Sequence[Mapping[str, Any]],
    *,
    family: Optional[str],
    environment: Optional[str],
    runtime_environment: Optional[str],
    now: float,
    horizon: float,
) -> dict[str, Any]:
    """Resolve latest logical windows before judging their usability.

    Model/quota-key aliases of one account window are not independent capacity.
    A newer invalid row must shadow older usable comparison evidence.
    Historical weekly quota can rank accounts without extending hard blocks.
    Local response observations can retain hard exclusions, but only the shared
    poll source supplies cross-runtime weekly ranking. ``environment`` and
    ``runtime_environment`` are retained as compatibility inputs and are not
    part of the observation identity.
    """
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in observations:
        if family is not None and row.get("quota_family") != family:
            continue
        source = str(row.get("source") or "")
        if not source:
            continue
        period = _period(row)
        if period is None:
            continue
        key = (source, str(row.get("quota_family") or ""), period)
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
                "paid_credits",
            )
        ):
            if window["exhausted"] and not previous["exhausted"]:
                # Keep the zero's value and provenance for the hard quota gate.
                latest[key] = window
                previous = window
            previous["unusable_reason"] = "conflicting_current_windows"
    windows = list(latest.values())
    weekly = latest.get((POLL_SOURCE, family or "", "seven_day"))
    paid_credits = None
    if (
        family == "overall"
        and weekly is not None
        and weekly["unusable_reason"] is None
        and weekly["status"] == "fresh"
        and weekly["observation_age_seconds"] <= horizon
        and weekly["expected_reset_at"] > now
        and isinstance(weekly.get("paid_credits"), dict)
    ):
        paid_credits = dict(weekly["paid_credits"])
        paid_credits["usable"] = (
            paid_credits.get("overage_limit_reached") is False
            and paid_credits.get("spend_control_reached") is False
            and (
                paid_credits.get("unlimited") is True
                or (
                    paid_credits.get("has_credits") is True
                    and (_number(paid_credits.get("balance")) or 0) > 0
                )
            )
        )
        if paid_credits["usable"]:
            weekly["exhausted"] = False
    missing_reason = (
        "unknown_quota_family"
        if family is None
        else "missing_weekly_observation"
    )
    return {
        "evaluated_at": now,
        # Account quota is shared across runtime deployments. Keep the
        # configured scope out of comparison identity; each window still
        # retains its source environment for provenance.
        "environment": "shared",
        "quota_family": family,
        "validity_horizon_seconds": horizon,
        "weekly": weekly,
        "paid_credits": paid_credits,
        "unusable_reason": (
            weekly["unusable_reason"] if weekly is not None else missing_reason
        ),
        # Preserve current-window consumers outside weekly account ranking.
        "valid_windows": [
            window
            for window in windows
            if (
                window["unusable_reason"] is None
                and window["status"] == "fresh"
                and window["observation_age_seconds"] <= horizon
                and window["expected_reset_at"] > now
            )
            or window["exhausted"]
        ],
    }


def decision_account(state: Mapping[str, Any], *, eligible: bool) -> dict[str, Any]:
    """Project only named, nonsecret fields; identities remain redactable."""
    candidate = state["candidate"]
    evidence = state.get("codex_oauth_quota_evidence") or {}
    weekly = evidence.get("weekly")
    paid_credits = evidence.get("paid_credits") or {}
    return {
        "account_label": candidate.get("codex_oauth_account_label"),
        "account_hash": candidate.get("codex_oauth_account_hash"),
        "eligible": eligible,
        "exclusion_reason": state.get("skip_reason")
        or ("cooldown" if not eligible else None),
        "quota_family": evidence.get("quota_family"),
        "environment": evidence.get("environment"),
        "evaluated_at": evidence.get("evaluated_at"),
        "weekly": (
            {key: value for key, value in weekly.items() if key != "paid_credits"}
            if isinstance(weekly, dict) else None
        ),
        "paid_credits_usable": paid_credits.get("usable"),
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
