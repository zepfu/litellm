"""Duration, timestamp, and half-open interval helpers for Chat usage capture."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional, Union
from zoneinfo import ZoneInfo

NumberOrStr = Union[int, float, str, datetime]


class DurationError(ValueError):
    """Raised when a duration cannot be used as elapsed time."""


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_datetime(value: Optional[NumberOrStr]) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return ensure_utc(datetime.fromisoformat(raw))
    except ValueError:
        pass
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"unrecognized timestamp: {value!r}") from exc


def isoformat_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return ensure_utc(value).isoformat().replace("+00:00", "Z")


def parse_iso_duration(value: str) -> timedelta:
    """Parse elapsed ISO-8601 durations. Month/year forms are rejected as ambiguous."""
    raw = (value or "").strip().upper()
    if not raw.startswith("P"):
        raise DurationError(f"duration must be ISO-8601: {value!r}")
    date_part, _, time_part = raw[1:].partition("T")
    if "Y" in date_part or "W" in date_part or "M" in date_part:
        raise DurationError(f"ambiguous month/year/week duration: {value!r}")
    days = _duration_component(date_part, "D")
    hours = _duration_component(time_part, "H")
    minutes = _duration_component(time_part, "M")
    seconds = _duration_component(time_part, "S")
    if days is None and hours is None and minutes is None and seconds is None:
        raise DurationError(f"empty duration: {value!r}")
    return timedelta(
        days=days or 0,
        hours=hours or 0,
        minutes=minutes or 0,
        seconds=seconds or 0,
    )


def _duration_component(part: str, unit: str) -> Optional[int]:
    if not part or unit not in part:
        return None
    prefix = part.split(unit, 1)[0]
    digits = ""
    for char in reversed(prefix):
        if char.isdigit():
            digits = char + digits
        else:
            break
    if not digits:
        raise DurationError(f"invalid duration component {unit} in {part!r}")
    return int(digits)


def format_iso_duration(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds % 86400 == 0 and total_seconds >= 86400:
        return f"P{total_seconds // 86400}D"
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    body = "T"
    if hours:
        body += f"{hours}H"
    if minutes:
        body += f"{minutes}M"
    if seconds or body == "T":
        body += f"{seconds}S"
    return f"P{body}"


def half_open_contains(start: Optional[datetime], end: Optional[datetime], instant: datetime) -> Optional[bool]:
    """Return True/False for [start, end), or None when the bound is unknown."""
    instant = ensure_utc(instant)
    if start is None and end is None:
        return None
    if start is not None and instant < ensure_utc(start):
        return False
    if end is not None and instant >= ensure_utc(end):
        return False
    if start is None or end is None:
        return None
    return True


def interval_membership(
    start: Optional[datetime],
    end: Optional[datetime],
    attempt_time: Optional[datetime],
    earliest: Optional[datetime] = None,
    latest: Optional[datetime] = None,
) -> str:
    """Classify an attempt against a half-open window without inventing bounds."""
    if start is None and end is None:
        return "unknown"
    if attempt_time is not None:
        contained = half_open_contains(start, end, attempt_time)
        if contained is True:
            return "in"
        if contained is False:
            return "out"
        return "unknown"
    if earliest is None or latest is None:
        return "unknown"
    earliest_utc = ensure_utc(earliest)
    latest_utc = ensure_utc(latest)
    if latest_utc < earliest_utc:
        return "unknown"
    start_utc = ensure_utc(start) if start is not None else None
    end_utc = ensure_utc(end) if end is not None else None
    if start_utc is not None and latest_utc < start_utc:
        return "out"
    if end_utc is not None and earliest_utc >= end_utc:
        return "out"
    if start_utc is not None and end_utc is not None:
        overlaps = earliest_utc < end_utc and latest_utc >= start_utc
        fully_inside = start_utc <= earliest_utc and latest_utc < end_utc
        if fully_inside:
            return "in"
        if overlaps:
            return "ambiguous"
        return "out"
    return "unknown"


def calendar_day_bounds(day_value: datetime, timezone_name: str) -> tuple[datetime, datetime]:
    """Return [local midnight, next local midnight) as UTC instants."""
    zone = ZoneInfo(timezone_name)
    local = ensure_utc(day_value).astimezone(zone)
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    next_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), next_local.astimezone(timezone.utc)


def iter_calendar_days(start: datetime, end: datetime, timezone_name: str) -> list[datetime]:
    """Yield local calendar-day starts covering [start, end) in the display timezone."""
    zone = ZoneInfo(timezone_name)
    start_utc = ensure_utc(start)
    end_utc = ensure_utc(end)
    cursor_local = start_utc.astimezone(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    days: list[datetime] = []
    while cursor_local.astimezone(timezone.utc) < end_utc:
        days.append(cursor_local.astimezone(timezone.utc))
        cursor_local = cursor_local + timedelta(days=1)
    return days


def parse_retry_after(value: str, *, now: datetime) -> datetime:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("empty Retry-After")
    if raw.isdigit():
        return ensure_utc(now) + timedelta(seconds=int(raw))
    parsed = parsedate_to_datetime(raw)
    if parsed is None:
        raise ValueError(f"unrecognized Retry-After: {value!r}")
    return ensure_utc(parsed)
