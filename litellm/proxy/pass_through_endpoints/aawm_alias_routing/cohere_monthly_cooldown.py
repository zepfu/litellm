"""Cohere monthly quota cooldown horizon.

The shared cooldown publisher still chooses the cooldown key. This module only
replaces that plan's TTL when a structured monthly exhaustion marker is
present, so a confirmed monthly Cohere failure is not retried on the generic
three-hour usage-limit schedule.

An authoritative reset timestamp is used exactly. Without one, the cooldown
runs until the documented UTC calendar-month boundary plus a bounded safety
margin. A reset that has already passed publishes a zero TTL with shrink so
the previous horizon is not retained. Ordinary per-model RPM failures never
carry this marker and keep the publisher's original duration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Optional

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
)

COHERE_MONTHLY_QUOTA_MARKER_ATTR = "_aawm_cohere_monthly_quota"
COHERE_MONTHLY_FAILURE_KIND = "cohere_monthly_trial_exhausted"
# Skew allowance after the UTC calendar-month boundary. Capped so the fallback
# cannot grow into another multi-hour usage-limit hold.
COHERE_MONTHLY_RESET_SAFETY_MARGIN_SECONDS = 60.0
COHERE_MONTHLY_RESET_SAFETY_MARGIN_MAX_SECONDS = 300.0
# A monthly reset further out than two calendar months is not authoritative.
_MAX_AUTHORITATIVE_RESET_LEAD_SECONDS = 63 * 24 * 60 * 60.0
_ABSOLUTE_RESET_EPOCH_FLOOR = 1_000_000_000.0
_MILLISECOND_RESET_EPOCH_FLOOR = 1_000_000_000_000.0
_EXHAUSTED_STATUSES = frozenset({"exhausted", "quota_exhausted"})
_MONTHLY_PERIODS = frozenset({"month", "monthly", "calendar_month"})
_EXPLICIT_RESET_KEYS = (
    "expected_reset_at",
    "monthly_reset_at",
    "quota_reset_at",
)
_SCOPED_RESET_KEYS = (
    "reset_at",
    "resets_at",
    "provider_resets_at",
)
_NESTED_ERROR_KEYS = ("error", "detail", "body")


@dataclass(frozen=True)
class CohereMonthlyCooldownHorizon:
    """Exact seconds and epoch at which a monthly Cohere cooldown must end."""

    ttl_seconds: float
    expires_at_epoch: float
    authoritative_reset: bool


def bounded_calendar_month_safety_margin_seconds() -> float:
    """Return the calendar-month fallback margin, clamped to its upper bound."""

    margin = float(COHERE_MONTHLY_RESET_SAFETY_MARGIN_SECONDS)
    if not math.isfinite(margin) or margin < 0:
        return 0.0
    return min(margin, float(COHERE_MONTHLY_RESET_SAFETY_MARGIN_MAX_SECONDS))


def documented_calendar_month_reset_epoch(now: datetime) -> float:
    """Return the exclusive UTC calendar-month end used by accepted-call accounting."""

    current = _as_utc(now)
    if current.month == 12:
        month_end = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        month_end = datetime(
            current.year,
            current.month + 1,
            1,
            tzinfo=timezone.utc,
        )
    return month_end.timestamp()


def cohere_monthly_exhaustion_marker(value: Any) -> Optional[Mapping[str, Any]]:
    """Return a structured Cohere monthly exhaustion marker, or None.

    HTTP 429 by itself is not monthly scope. The marker must name Cohere, an
    exhausted status, and a monthly period.
    """

    if not isinstance(value, Mapping):
        return None
    provider = str(value.get("provider") or "").strip().lower()
    status = str(value.get("status") or "").strip().lower()
    period = str(value.get("quota_period") or value.get("quota_type") or "").strip().lower()
    if provider != "cohere":
        return None
    if status not in _EXHAUSTED_STATUSES:
        return None
    if period not in _MONTHLY_PERIODS:
        return None
    return value


def stamp_cohere_monthly_quota_marker(
    exc: Exception,
) -> Mapping[str, Any]:
    """Attach the structured monthly marker for one confirmed Cohere failure."""

    marker_fields: dict[str, Any] = {
        "provider": "cohere",
        "status": "exhausted",
        "quota_period": "calendar_month",
        "quota_type": "monthly",
        "failure_kind": COHERE_MONTHLY_FAILURE_KIND,
    }
    reset_epoch = extract_authoritative_reset_epoch(exc)
    if reset_epoch is not None:
        marker_fields["expected_reset_at"] = reset_epoch
    marker = MappingProxyType(marker_fields)
    setattr(exc, COHERE_MONTHLY_QUOTA_MARKER_ATTR, marker)
    return marker


def extract_authoritative_reset_epoch(exc: Exception) -> Optional[float]:
    """Read a structured monthly reset timestamp from an exception payload."""

    for payload in _exception_payloads(exc):
        for error_object in _walk_error_dicts(payload):
            for key in _EXPLICIT_RESET_KEYS:
                if key not in error_object:
                    continue
                parsed = _parse_reset_epoch(error_object.get(key))
                if parsed is not None:
                    return parsed
            if not _dict_has_monthly_scope(error_object):
                continue
            for key in _SCOPED_RESET_KEYS:
                if key not in error_object:
                    continue
                parsed = _parse_reset_epoch(error_object.get(key))
                if parsed is not None:
                    return parsed
    return None


def resolve_cohere_monthly_cooldown_horizon(
    marker: Any,
    *,
    now: Optional[datetime] = None,
) -> Optional[CohereMonthlyCooldownHorizon]:
    """Resolve the monthly TTL, or None when the marker is not monthly scope.

    Future authoritative resets expire exactly at that timestamp. A timestamp
    at or before ``now`` yields TTL zero so the cooldown is not retained. The
    calendar-month fallback expires at the documented boundary plus the bounded
    safety margin.
    """

    confirmed = cohere_monthly_exhaustion_marker(marker)
    if confirmed is None:
        return None
    current = datetime.now(timezone.utc) if now is None else _as_utc(now)
    now_epoch = current.timestamp()
    reset_epoch = _authoritative_reset_epoch(confirmed.get("expected_reset_at"), now_epoch)
    if reset_epoch is not None:
        ttl_seconds = reset_epoch - now_epoch
        if ttl_seconds <= 0:
            return CohereMonthlyCooldownHorizon(
                ttl_seconds=0.0,
                expires_at_epoch=reset_epoch,
                authoritative_reset=True,
            )
        return CohereMonthlyCooldownHorizon(
            ttl_seconds=ttl_seconds,
            expires_at_epoch=reset_epoch,
            authoritative_reset=True,
        )
    expires_at_epoch = documented_calendar_month_reset_epoch(current) + bounded_calendar_month_safety_margin_seconds()
    return CohereMonthlyCooldownHorizon(
        ttl_seconds=expires_at_epoch - now_epoch,
        expires_at_epoch=expires_at_epoch,
        authoritative_reset=False,
    )


def apply_cohere_monthly_cooldown_horizon(
    plan: Any,
    exc: Exception,
    *,
    now: Optional[datetime] = None,
    marker: Any = None,
) -> Any:
    """Replace only duration and shrink on a plan the publisher already built.

    Key identity stays on ``plan``. A missing or non-monthly marker leaves the
    publisher duration unchanged, including ordinary per-model RPM cooldowns.
    Plans that are not publication plans pass through untouched.
    """

    if not isinstance(plan, CooldownPublicationPlan):
        return plan
    selected_marker = marker if marker is not None else getattr(exc, COHERE_MONTHLY_QUOTA_MARKER_ATTR, None)
    horizon = resolve_cohere_monthly_cooldown_horizon(selected_marker, now=now)
    if horizon is None or not _plan_publishes_cooldown(plan):
        return plan
    return replace(
        plan,
        duration_seconds=horizon.ttl_seconds,
        allow_ttl_shrink=True,
    )


def _plan_publishes_cooldown(plan: CooldownPublicationPlan) -> bool:
    if plan.applied_scope == "none" and not plan.memory_keys and not plan.durable_keys:
        return False
    return bool(plan.memory_keys or plan.durable_keys or plan.request_local_action)


def _authoritative_reset_epoch(value: Any, now_epoch: float) -> Optional[float]:
    reset_epoch = _parse_reset_epoch(value)
    if reset_epoch is None:
        return None
    if reset_epoch - now_epoch > _MAX_AUTHORITATIVE_RESET_LEAD_SECONDS:
        return None
    return reset_epoch


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _dict_has_monthly_scope(error_object: Mapping[str, Any]) -> bool:
    period = str(error_object.get("quota_period") or error_object.get("quota_type") or "").strip().lower()
    if period in _MONTHLY_PERIODS:
        return True
    failure_kind = str(error_object.get("failure_kind") or "").strip().lower()
    return failure_kind == COHERE_MONTHLY_FAILURE_KIND


def _parse_reset_epoch(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, datetime):
        number = _as_utc(value).timestamp()
    elif isinstance(value, str):
        number = _parse_reset_text(value)
        if number is None:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    if number >= _MILLISECOND_RESET_EPOCH_FLOOR:
        number = number / 1000.0
    if number < _ABSOLUTE_RESET_EPOCH_FLOOR:
        return None
    return number


def _parse_reset_text(value: str) -> Optional[float]:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    iso_text = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        return None
    return _as_utc(parsed).timestamp()


def _exception_payloads(exc: Exception) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        payloads.append(detail)
    coerced = _coerce_upstream_error_payload(_extract_passthrough_exception_detail(exc))
    if isinstance(coerced, dict) and all(coerced is not item for item in payloads):
        payloads.append(coerced)
    return payloads


def _walk_error_dicts(
    value: Any,
    *,
    depth: int = 0,
) -> list[dict[str, Any]]:
    if depth > 4 or not isinstance(value, dict):
        return []
    found = [value]
    for key in _NESTED_ERROR_KEYS:
        nested = value.get(key)
        if isinstance(nested, dict):
            found.extend(_walk_error_dicts(nested, depth=depth + 1))
    return found
