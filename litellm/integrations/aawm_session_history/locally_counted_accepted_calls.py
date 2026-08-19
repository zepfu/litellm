"""Generic persistence for locally counted accepted provider calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Sequence

from litellm.integrations.aawm_session_history.writer import (
    _get_aawm_session_history_pool,
)
from litellm.secret_managers.main import get_secret_str

CountPeriod = Literal["monthly", "daily", "rolling_seconds"]

_LOCALLY_COUNTED_DATABASE_NAME = "aawm_tristore"
_LOCALLY_COUNTED_OBSERVATION_SOURCE = "locally_counted"
_VALID_PERIODS = frozenset({"monthly", "daily", "rolling_seconds"})

COHERE_PROVIDER = "cohere"
COHERE_CREDENTIAL_SCOPE = "cohere_trial_default"
COHERE_LANE = "cohere_native"
COHERE_MONTHLY_LIMIT = 1000
COHERE_RPM_WINDOW_SECONDS = 60

OPENROUTER_PROVIDER = "openrouter"
OPENROUTER_FREE_DAILY_CREDENTIAL_SCOPE = "openrouter_free_daily_shared"
OPENROUTER_FREE_DAILY_LIMIT_ENV = "AAWM_OPENROUTER_FREE_DAILY_REQUEST_LIMIT"
OPENROUTER_FREE_DAILY_REQUEST_LIMIT_DEFAULT = 1000

OPENCODE_ZEN_PROVIDER = "opencode_zen"
NVIDIA_NIM_PROVIDER = "nvidia_nim"

_CURRENT_DATABASE_SQL = "SELECT current_database()"
_ADVISORY_LOCK_SQL = """
SELECT pg_advisory_xact_lock(
    hashtext($1::text),
    hashtext($2::text)
)
"""
_INSERT_SQL = """
INSERT INTO public.locally_counted_accepted_calls (
    accepted_at,
    provider,
    credential_scope,
    lane,
    model,
    litellm_call_id,
    session_id,
    trace_id,
    source,
    evidence
) VALUES (
    $1::timestamptz,
    $2::text,
    $3::text,
    $4::text,
    $5::text,
    $6::text,
    $7::text,
    $8::text,
    $9::text,
    $10::jsonb
)
ON CONFLICT (provider, credential_scope, litellm_call_id) DO NOTHING
RETURNING litellm_call_id
"""
_RANGE_COUNT_SQL = """
SELECT COUNT(*)::integer
FROM public.locally_counted_accepted_calls
WHERE provider = $1::text
  AND credential_scope = $2::text
  AND accepted_at >= $3::timestamptz
  AND accepted_at < $4::timestamptz
"""
_RANGE_MODEL_COUNT_SQL = """
SELECT COUNT(*)::integer
FROM public.locally_counted_accepted_calls
WHERE provider = $1::text
  AND credential_scope = $2::text
  AND accepted_at >= $3::timestamptz
  AND accepted_at < $4::timestamptz
  AND model IS NOT DISTINCT FROM $5::text
"""
_ROLLING_COUNT_SQL = """
SELECT COUNT(*)::integer
FROM public.locally_counted_accepted_calls
WHERE provider = $1::text
  AND credential_scope = $2::text
  AND accepted_at > $3::timestamptz - ($4::integer * INTERVAL '1 second')
  AND accepted_at <= $3::timestamptz
"""
_ROLLING_MODEL_COUNT_SQL = """
SELECT COUNT(*)::integer
FROM public.locally_counted_accepted_calls
WHERE provider = $1::text
  AND credential_scope = $2::text
  AND accepted_at > $3::timestamptz - ($4::integer * INTERVAL '1 second')
  AND accepted_at <= $3::timestamptz
  AND model IS NOT DISTINCT FROM $5::text
"""


@dataclass(frozen=True)
class CountWindow:
    name: str
    period: CountPeriod
    limit: Optional[int] = None
    model_scoped: bool = False
    window_seconds: Optional[int] = None


@dataclass(frozen=True)
class WindowCount:
    name: str
    period: str
    used: int
    remaining: Optional[int]
    limit: Optional[int]
    model_scoped: bool = False
    window_seconds: Optional[int] = None


@dataclass(frozen=True)
class LocallyCountedAcceptedCallState:
    counted: bool
    windows: Mapping[str, WindowCount]


def _normalize_utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("accepted_at must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def calendar_month_bounds(value: datetime) -> tuple[datetime, datetime]:
    accepted_at = _normalize_utc_datetime(value)
    month_start = datetime(accepted_at.year, accepted_at.month, 1, tzinfo=timezone.utc)
    if accepted_at.month == 12:
        month_end = datetime(accepted_at.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        month_end = datetime(accepted_at.year, accepted_at.month + 1, 1, tzinfo=timezone.utc)
    return month_start, month_end


def calendar_day_bounds(value: datetime) -> tuple[datetime, datetime]:
    accepted_at = _normalize_utc_datetime(value)
    day_start = datetime(
        accepted_at.year,
        accepted_at.month,
        accepted_at.day,
        tzinfo=timezone.utc,
    )
    return day_start, day_start + timedelta(days=1)


def _clean_required_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("text fields must be strings or None")
    cleaned = value.strip()
    return cleaned or None


def _count_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _remaining(*, used: int, limit: Optional[int]) -> Optional[int]:
    if limit is None:
        return None
    return max(0, int(limit) - used)


def _optional_limit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CountWindow.limit must be an int or None") from exc
    if parsed < 0:
        raise ValueError("CountWindow.limit must be >= 0")
    return parsed


def _validate_window(window: CountWindow) -> CountWindow:
    if not isinstance(window, CountWindow):
        raise TypeError("windows must contain CountWindow instances")
    name = _clean_required_text(window.name, field_name="CountWindow.name")
    if window.period not in _VALID_PERIODS:
        raise ValueError(f"unsupported CountWindow.period: {window.period!r}")
    limit = _optional_limit(window.limit)
    window_seconds = window.window_seconds
    if window.period == "rolling_seconds":
        try:
            parsed_seconds = int(window_seconds) if window_seconds is not None else 0
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rolling_seconds windows require window_seconds > 0"
            ) from exc
        if parsed_seconds <= 0:
            raise ValueError("rolling_seconds windows require window_seconds > 0")
        window_seconds = parsed_seconds
    elif window_seconds is not None:
        try:
            window_seconds = int(window_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("CountWindow.window_seconds must be an int or None") from exc
    if (
        name == window.name
        and limit == window.limit
        and window_seconds == window.window_seconds
    ):
        return window
    return CountWindow(
        name=name,
        period=window.period,
        limit=limit,
        model_scoped=bool(window.model_scoped),
        window_seconds=window_seconds,
    )


def _normalize_windows(windows: Sequence[CountWindow]) -> tuple[CountWindow, ...]:
    normalized: list[CountWindow] = []
    seen: set[str] = set()
    for window in windows:
        item = _validate_window(window)
        if item.name in seen:
            raise ValueError(f"duplicate CountWindow.name: {item.name!r}")
        seen.add(item.name)
        normalized.append(item)
    return tuple(normalized)


async def _count_window(
    conn: Any,
    *,
    window: CountWindow,
    provider: str,
    credential_scope: str,
    accepted_at: datetime,
    model: Optional[str],
) -> int:
    if window.period in {"monthly", "daily"}:
        start, end = (
            calendar_month_bounds(accepted_at)
            if window.period == "monthly"
            else calendar_day_bounds(accepted_at)
        )
        sql = _RANGE_MODEL_COUNT_SQL if window.model_scoped else _RANGE_COUNT_SQL
        args: tuple[Any, ...] = (provider, credential_scope, start, end)
        if window.model_scoped:
            args = (*args, model)
        return _count_value(await conn.fetchval(sql, *args))

    sql = _ROLLING_MODEL_COUNT_SQL if window.model_scoped else _ROLLING_COUNT_SQL
    args = (provider, credential_scope, accepted_at, int(window.window_seconds or 0))
    if window.model_scoped:
        args = (*args, model)
    return _count_value(await conn.fetchval(sql, *args))


def openrouter_free_daily_request_limit() -> int:
    raw = get_secret_str(OPENROUTER_FREE_DAILY_LIMIT_ENV)
    if raw is None:
        return OPENROUTER_FREE_DAILY_REQUEST_LIMIT_DEFAULT
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return OPENROUTER_FREE_DAILY_REQUEST_LIMIT_DEFAULT
    if parsed <= 0:
        return OPENROUTER_FREE_DAILY_REQUEST_LIMIT_DEFAULT
    return parsed


def cohere_trial_windows(*, rpm_limit: Optional[int] = None) -> tuple[CountWindow, ...]:
    return (
        CountWindow(
            name="monthly",
            period="monthly",
            limit=COHERE_MONTHLY_LIMIT,
            model_scoped=False,
        ),
        CountWindow(
            name="rpm",
            period="rolling_seconds",
            limit=rpm_limit,
            model_scoped=True,
            window_seconds=COHERE_RPM_WINDOW_SECONDS,
        ),
    )


def openrouter_free_daily_windows(
    *,
    limit: Optional[int] = None,
) -> tuple[CountWindow, ...]:
    resolved = openrouter_free_daily_request_limit() if limit is None else limit
    return (
        CountWindow(
            name="daily",
            period="daily",
            limit=resolved,
            model_scoped=False,
        ),
    )


def unmetered_window(
    *,
    name: str,
    period: CountPeriod,
    model_scoped: bool = False,
    window_seconds: Optional[int] = None,
) -> CountWindow:
    return CountWindow(
        name=name,
        period=period,
        limit=None,
        model_scoped=model_scoped,
        window_seconds=window_seconds,
    )


def opencode_zen_windows() -> tuple[CountWindow, ...]:
    return (unmetered_window(name="usage", period="daily"),)


def nvidia_nim_windows() -> tuple[CountWindow, ...]:
    return (unmetered_window(name="usage", period="daily"),)


async def record_locally_counted_accepted_call(
    *,
    litellm_call_id: str,
    accepted_at: datetime,
    provider: str,
    credential_scope: str,
    source: str,
    model: Optional[str] = None,
    lane: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    windows: Sequence[CountWindow],
) -> LocallyCountedAcceptedCallState:
    call_id = _clean_required_text(litellm_call_id, field_name="litellm_call_id")
    provider_value = _clean_required_text(provider, field_name="provider")
    scope_value = _clean_required_text(credential_scope, field_name="credential_scope")
    source_value = _clean_required_text(source, field_name="source")
    accepted_at_utc = _normalize_utc_datetime(accepted_at)
    model_value = _clean_optional_text(model)
    lane_value = _clean_optional_text(lane)
    session_value = _clean_optional_text(session_id)
    trace_value = _clean_optional_text(trace_id)
    requested_windows = _normalize_windows(windows)
    evidence = json.dumps(
        {"observation_source": _LOCALLY_COUNTED_OBSERVATION_SOURCE},
        separators=(",", ":"),
        sort_keys=True,
    )

    pool = await _get_aawm_session_history_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            database_name = await conn.fetchval(_CURRENT_DATABASE_SQL)
            if database_name != _LOCALLY_COUNTED_DATABASE_NAME:
                raise RuntimeError(
                    "Locally counted accepted-call persistence requires database "
                    f"{_LOCALLY_COUNTED_DATABASE_NAME}; connected database is "
                    f"{database_name!r}"
                )

            await conn.fetchval(_ADVISORY_LOCK_SQL, provider_value, scope_value)
            inserted_row = await conn.fetchrow(
                _INSERT_SQL,
                accepted_at_utc,
                provider_value,
                scope_value,
                lane_value,
                model_value,
                call_id,
                session_value,
                trace_value,
                source_value,
                evidence,
            )
            counted_windows: dict[str, WindowCount] = {}
            for window in requested_windows:
                used = await _count_window(
                    conn,
                    window=window,
                    provider=provider_value,
                    credential_scope=scope_value,
                    accepted_at=accepted_at_utc,
                    model=model_value,
                )
                counted_windows[window.name] = WindowCount(
                    name=window.name,
                    period=window.period,
                    used=used,
                    remaining=_remaining(used=used, limit=window.limit),
                    limit=window.limit,
                    model_scoped=window.model_scoped,
                    window_seconds=window.window_seconds,
                )

    return LocallyCountedAcceptedCallState(
        counted=inserted_row is not None,
        windows=MappingProxyType(counted_windows),
    )


__all__ = [
    "COHERE_CREDENTIAL_SCOPE",
    "COHERE_LANE",
    "COHERE_MONTHLY_LIMIT",
    "COHERE_PROVIDER",
    "COHERE_RPM_WINDOW_SECONDS",
    "CountPeriod",
    "CountWindow",
    "LocallyCountedAcceptedCallState",
    "NVIDIA_NIM_PROVIDER",
    "OPENCODE_ZEN_PROVIDER",
    "OPENROUTER_FREE_DAILY_CREDENTIAL_SCOPE",
    "OPENROUTER_FREE_DAILY_LIMIT_ENV",
    "OPENROUTER_FREE_DAILY_REQUEST_LIMIT_DEFAULT",
    "OPENROUTER_PROVIDER",
    "WindowCount",
    "calendar_day_bounds",
    "calendar_month_bounds",
    "cohere_trial_windows",
    "nvidia_nim_windows",
    "opencode_zen_windows",
    "openrouter_free_daily_request_limit",
    "openrouter_free_daily_windows",
    "record_locally_counted_accepted_call",
    "unmetered_window",
]
