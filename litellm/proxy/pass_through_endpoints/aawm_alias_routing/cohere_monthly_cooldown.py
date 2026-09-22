"""Cohere monthly quota cooldown horizon.

The shared cooldown publisher still chooses the cooldown key. This module only
replaces that plan's TTL when monthly exhaustion evidence is present, so a
confirmed monthly Cohere failure is not retried on the generic three-hour
usage-limit schedule.

An admissible monthly reset timestamp is used exactly. An RPM-scoped or
unusable timestamp is skipped. Without a usable monthly reset, the cooldown
runs until the documented UTC calendar-month boundary plus a bounded safety
margin. Publication expires at that absolute deadline. A deadline that has
already passed clears the existing keys instead of starting another relative
hold. Ordinary per-model RPM failures never carry this marker.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Optional

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
    delete_aawm_alias_routing_durable_key,
    get_aawm_alias_routing_dual_cache,
)
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
_INADMISSIBLE_RESET_YEAR = 9999
_EXHAUSTED_STATUSES = frozenset({"exhausted", "quota_exhausted"})
_MONTHLY_PERIODS = frozenset({"month", "monthly", "calendar_month"})
_RPM_PERIODS = frozenset(
    {
        "second",
        "seconds",
        "minute",
        "minutes",
        "hour",
        "hours",
        "rpm",
        "rate_limit",
        "requests_per_minute",
        "per_minute",
        "request",
        "requests",
    }
)
_MONTHLY_EXHAUSTION_TEXT_MARKERS = (
    "monthly trial",
    "trial monthly",
    "monthly quota",
    "monthly limit",
    "monthly usage",
    "calendar month",
    "calendar-month",
)
_RESET_FIELD_ORDER = (
    "expected_reset_at",
    "quota_reset_at",
    "monthly_reset_at",
    "reset_at",
    "resets_at",
    "provider_resets_at",
)
_EXPLICIT_RESET_KEYS = frozenset({"expected_reset_at", "quota_reset_at"})
_SCOPED_RESET_KEYS = frozenset({"reset_at", "resets_at", "provider_resets_at"})
_PERIOD_KEYS = ("quota_period", "quota_type", "reset_scope", "limit_period", "scope")
_NESTED_ERROR_KEYS = ("error", "detail", "body")
_TEXT_KEYS = ("message", "detail", "error", "type", "code")
_RESET_CONVERSION_ERRORS = (OverflowError, ValueError, OSError, TypeError, ArithmeticError)


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
        month_end = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
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


def cohere_failure_has_monthly_exhaustion_evidence(exc: Exception) -> bool:
    """Return whether ``exc`` shows Cohere monthly exhaustion, not trial RPM wording."""

    try:
        text = _exception_text(exc)
    except _RESET_CONVERSION_ERRORS:
        text = ""
    if any(marker in text for marker in _MONTHLY_EXHAUSTION_TEXT_MARKERS):
        return True
    try:
        payloads = _exception_payloads(exc)
    except _RESET_CONVERSION_ERRORS:
        return False
    for payload in payloads:
        for error_object in _walk_error_dicts(payload):
            period = _period_token(error_object)
            if _is_monthly_period(period) and not _is_rpm_period(period):
                return True
    return False


def stamp_cohere_monthly_quota_marker(exc: Exception) -> Optional[Mapping[str, Any]]:
    """Attach the structured monthly marker for one confirmed Cohere failure.

    Trial RPM wording does not qualify. Timestamp conversion failures are
    skipped; when no usable monthly reset remains, the marker is still attached
    so the calendar-month fallback can run. This never raises into classification.
    """

    try:
        confirmed = cohere_failure_has_monthly_exhaustion_evidence(exc)
    except _RESET_CONVERSION_ERRORS:
        return None
    if not confirmed:
        return None
    reset_epoch: Optional[float] = None
    try:
        reset_epoch = extract_authoritative_reset_epoch(exc)
    except _RESET_CONVERSION_ERRORS:
        reset_epoch = None
    marker_fields: dict[str, Any] = {
        "provider": "cohere",
        "status": "exhausted",
        "quota_period": "calendar_month",
        "quota_type": "monthly",
        "failure_kind": COHERE_MONTHLY_FAILURE_KIND,
    }
    if reset_epoch is not None:
        marker_fields["expected_reset_at"] = reset_epoch
    marker = MappingProxyType(marker_fields)
    try:
        setattr(exc, COHERE_MONTHLY_QUOTA_MARKER_ATTR, marker)
    except _RESET_CONVERSION_ERRORS:
        return marker
    return marker


def extract_authoritative_reset_epoch(
    exc: Exception,
    *,
    now: Optional[datetime] = None,
) -> Optional[float]:
    """Return the first usable monthly reset, skipping RPM and bad timestamps.

    An RPM-scoped ``quota_reset_at`` does not hide a nested monthly reset.
    Unusable values, including year 9999 and timestamps that fail conversion,
    are skipped. ``None`` means no usable monthly reset remains.
    """

    now_epoch = time.time() if now is None else _as_utc(now).timestamp()
    try:
        payloads = _exception_payloads(exc)
    except _RESET_CONVERSION_ERRORS:
        return None
    for payload in payloads:
        for error_object in _walk_error_dicts(payload):
            for key in _RESET_FIELD_ORDER:
                if not _field_is_monthly_reset(error_object, key):
                    continue
                parsed = _parse_reset_epoch(error_object.get(key))
                if parsed is None or not _is_usable_monthly_reset(parsed, now_epoch):
                    continue
                return parsed
    return None


def resolve_cohere_monthly_cooldown_horizon(
    marker: Any,
    *,
    now: Optional[datetime] = None,
) -> Optional[CohereMonthlyCooldownHorizon]:
    """Resolve the monthly TTL, or None when the marker is not monthly scope.

    Future authoritative resets expire exactly at that timestamp. A timestamp
    at or before ``now`` yields TTL zero so publication can clear the hold.
    The calendar-month fallback expires at the documented boundary plus the
    bounded safety margin, and only after no usable monthly reset remains.
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
    """Replace duration, shrink, and the absolute deadline on a publisher plan.

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
        expires_at_epoch=horizon.expires_at_epoch,
    )


def align_cohere_monthly_publication_plan(
    plan: Any,
    *,
    now_epoch: Optional[float] = None,
) -> tuple[Any, bool]:
    """Return the plan timed to its absolute deadline, and whether that deadline passed.

    Plans without a monthly deadline are unchanged. A passed deadline keeps the
    same keys and reports duration zero so the caller can clear the hold
    instead of publishing a positive TTL.
    """

    if not isinstance(plan, CooldownPublicationPlan):
        return plan, False
    deadline = plan.expires_at_epoch
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
        return plan, False
    current = time.time() if now_epoch is None else float(now_epoch)
    try:
        remaining = float(deadline) - current
    except _RESET_CONVERSION_ERRORS:
        remaining = 0.0
    if not math.isfinite(remaining) or remaining <= 0:
        return replace(plan, duration_seconds=0.0, allow_ttl_shrink=True), True
    if remaining == plan.duration_seconds:
        return plan, False
    return replace(plan, duration_seconds=remaining, allow_ttl_shrink=True), False


def cohere_monthly_publication_duration(
    plan: Any,
    *,
    now: Optional[datetime] = None,
) -> float:
    """Return the TTL still left until a monthly deadline, or the plan duration."""

    now_epoch = None if now is None else _as_utc(now).timestamp()
    aligned, _expired = align_cohere_monthly_publication_plan(plan, now_epoch=now_epoch)
    try:
        return float(getattr(aligned, "duration_seconds", 0.0) or 0.0)
    except _RESET_CONVERSION_ERRORS:
        return 0.0


async def expire_cohere_monthly_cooldown_hold(
    *,
    alias_family: str,
    plan: CooldownPublicationPlan,
    family_state: Any,
) -> None:
    """Drop a monthly hold whose absolute deadline has already passed.

    Memory and durable values are removed from the plan's existing keys. No
    replacement cooldown is written, so a zero TTL cannot become a one-second
    Redis key.
    """

    keys = tuple(dict.fromkeys((*plan.memory_keys, *plan.durable_keys)))
    clear = getattr(family_state, "clear_cooldown_state", None)
    if callable(clear) and keys:
        clear(cooldown_keys=keys)
    if not plan.durable_keys or get_aawm_alias_routing_dual_cache() is None:
        return
    for key in plan.durable_keys:
        await delete_aawm_alias_routing_durable_key(
            alias_family=alias_family,
            state_kind="cooldown",
            state_key=key,
        )


def _plan_publishes_cooldown(plan: CooldownPublicationPlan) -> bool:
    if plan.applied_scope == "none" and not plan.memory_keys and not plan.durable_keys:
        return False
    return bool(plan.memory_keys or plan.durable_keys or plan.request_local_action)


def _authoritative_reset_epoch(value: Any, now_epoch: float) -> Optional[float]:
    reset_epoch = _parse_reset_epoch(value)
    if reset_epoch is None or not _is_usable_monthly_reset(reset_epoch, now_epoch):
        return None
    return reset_epoch


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _dict_has_monthly_scope(error_object: Mapping[str, Any]) -> bool:
    period = _period_token(error_object)
    if _is_monthly_period(period):
        return True
    failure_kind = str(error_object.get("failure_kind") or "").strip().lower()
    return failure_kind == COHERE_MONTHLY_FAILURE_KIND


def _period_token(error_object: Mapping[str, Any]) -> str:
    for key in _PERIOD_KEYS:
        raw = error_object.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower().replace("-", "_").replace(" ", "_")
    return ""


def _is_rpm_period(token: str) -> bool:
    if not token:
        return False
    if token in _RPM_PERIODS or token.endswith("_rpm"):
        return True
    return "per_minute" in token


def _is_monthly_period(token: str) -> bool:
    return token in _MONTHLY_PERIODS


def _field_is_monthly_reset(error_object: Mapping[str, Any], key: str) -> bool:
    if key not in error_object:
        return False
    if key == "monthly_reset_at":
        return True
    period = _period_token(error_object)
    if _is_rpm_period(period):
        return False
    if key in _EXPLICIT_RESET_KEYS:
        if period and not _is_monthly_period(period) and not _dict_has_monthly_scope(error_object):
            return False
        return True
    if key in _SCOPED_RESET_KEYS:
        return _dict_has_monthly_scope(error_object)
    return False


def _is_usable_monthly_reset(epoch: float, now_epoch: float) -> bool:
    if not math.isfinite(epoch) or not math.isfinite(now_epoch):
        return False
    try:
        if epoch - now_epoch > _MAX_AUTHORITATIVE_RESET_LEAD_SECONDS:
            return False
        year = datetime.fromtimestamp(epoch, timezone.utc).year
    except _RESET_CONVERSION_ERRORS:
        return False
    return year < _INADMISSIBLE_RESET_YEAR


def _parse_reset_epoch(value: Any) -> Optional[float]:
    try:
        return _coerce_reset_epoch(value)
    except _RESET_CONVERSION_ERRORS:
        return None


def _coerce_reset_epoch(value: Any) -> Optional[float]:
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
    except _RESET_CONVERSION_ERRORS:
        pass
    iso_text = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    parsed = datetime.fromisoformat(iso_text)
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


def _exception_text(exc: Exception) -> str:
    parts: list[str] = []
    detail = getattr(exc, "detail", None)
    if isinstance(detail, str):
        parts.append(detail)
    try:
        payloads = _exception_payloads(exc)
    except _RESET_CONVERSION_ERRORS:
        payloads = []
    for payload in payloads:
        parts.extend(_iter_text(payload))
    if not parts:
        message = getattr(exc, "message", None)
        if isinstance(message, str) and message.strip():
            parts.append(message)
        else:
            parts.append(str(exc))
    return " ".join(parts).lower()


def _iter_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts: list[str] = []
        for key in _TEXT_KEYS:
            if key in value:
                parts.extend(_iter_text(value[key]))
        return parts
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.extend(_iter_text(item))
        return parts
    return []


def _walk_error_dicts(value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 4 or not isinstance(value, dict):
        return []
    found = [value]
    for key in _NESTED_ERROR_KEYS:
        nested = value.get(key)
        if isinstance(nested, dict):
            found.extend(_walk_error_dicts(nested, depth=depth + 1))
    return found
