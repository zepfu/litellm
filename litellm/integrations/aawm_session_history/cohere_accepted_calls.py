"""Persistence for accepted direct Cohere trial calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from litellm.integrations.aawm_session_history.writer import (
    _get_aawm_session_history_pool,
)
from litellm.utils import get_model_info

_COHERE_DATABASE_NAME = "aawm_tristore"
_COHERE_PROVIDER = "cohere"
_COHERE_CREDENTIAL_SCOPE = "cohere_trial_default"
_COHERE_MONTHLY_LIMIT = 1000
_COHERE_RPM_WINDOW_SECONDS = 60
_COHERE_DEFAULT_SOURCE = "codex_cohere_chat_completions_adapter"
_COHERE_OBSERVATION_SOURCE = "locally_counted"

_COHERE_CURRENT_DATABASE_SQL = "SELECT current_database()"
_COHERE_ACCEPTED_CALL_ADVISORY_LOCK_SQL = """
SELECT pg_advisory_xact_lock(
    hashtext($1::text)
)
"""
_COHERE_ACCEPTED_CALL_INSERT_SQL = """
INSERT INTO public.cohere_accepted_calls (
    accepted_at,
    month_start,
    provider,
    credential_scope,
    model,
    litellm_call_id,
    session_id,
    trace_id,
    source,
    evidence
) VALUES (
    $1::timestamptz,
    $2::date,
    'cohere',
    'cohere_trial_default',
    $3::text,
    $4::text,
    $5::text,
    $6::text,
    $7::text,
    $8::jsonb
)
ON CONFLICT (litellm_call_id) DO NOTHING
RETURNING litellm_call_id
"""
_COHERE_ACCEPTED_CALL_MONTHLY_COUNT_SQL = """
SELECT COUNT(*)::integer
FROM public.cohere_accepted_calls
WHERE provider = 'cohere'
  AND credential_scope = 'cohere_trial_default'
  AND month_start = $1::date
"""
_COHERE_ACCEPTED_CALL_RPM_COUNT_SQL = """
SELECT COUNT(*)::integer
FROM public.cohere_accepted_calls
WHERE provider = 'cohere'
  AND credential_scope = 'cohere_trial_default'
  AND accepted_at > $1::timestamptz - INTERVAL '60 seconds'
  AND accepted_at <= $1::timestamptz
  AND model IS NOT DISTINCT FROM $2::text
"""


@dataclass(frozen=True)
class CohereAcceptedCallState:
    counted: bool
    monthly_used: int
    monthly_remaining: int
    monthly_limit: int
    rpm_used: int
    rpm_remaining: Optional[int]
    rpm_limit: Optional[int]
    month_start: datetime
    month_end: datetime
    observation_source: str = _COHERE_OBSERVATION_SOURCE


def _normalize_utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("accepted_at must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _month_bounds(value: datetime) -> tuple[datetime, datetime]:
    month_start = datetime(value.year, value.month, 1, tzinfo=timezone.utc)
    if value.month == 12:
        month_end = datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        month_end = datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)
    return month_start, month_end


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("text fields must be strings or None")
    cleaned = value.strip()
    return cleaned or None


def _resolve_rpm_limit(model: Optional[str]) -> Optional[int]:
    if model is None:
        return None
    try:
        model_info = get_model_info(model=model, custom_llm_provider=_COHERE_PROVIDER)
        if isinstance(model_info, Mapping):
            rpm = model_info.get("rpm")
        else:
            rpm = getattr(model_info, "rpm", None)
    except Exception:
        return None
    try:
        return max(0, int(rpm)) if rpm is not None else None
    except (TypeError, ValueError):
        return None


def _count_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


async def record_cohere_accepted_call(
    *,
    litellm_call_id: str,
    accepted_at: datetime,
    model: Optional[str],
    session_id: Optional[str],
    trace_id: Optional[str],
    source: str = _COHERE_DEFAULT_SOURCE,
) -> CohereAcceptedCallState:
    if not isinstance(litellm_call_id, str) or not litellm_call_id.strip():
        raise ValueError("litellm_call_id must be a non-blank string")

    call_id = litellm_call_id.strip()
    accepted_at_utc = _normalize_utc_datetime(accepted_at)
    model_value = _clean_optional_text(model)
    session_value = _clean_optional_text(session_id)
    trace_value = _clean_optional_text(trace_id)
    source_value = _clean_optional_text(source) or _COHERE_DEFAULT_SOURCE
    month_start, month_end = _month_bounds(accepted_at_utc)
    rpm_limit = _resolve_rpm_limit(model_value)
    evidence = json.dumps(
        {"observation_source": _COHERE_OBSERVATION_SOURCE},
        separators=(",", ":"),
        sort_keys=True,
    )

    pool = await _get_aawm_session_history_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            database_name = await conn.fetchval(_COHERE_CURRENT_DATABASE_SQL)
            if database_name != _COHERE_DATABASE_NAME:
                raise RuntimeError(
                    "Cohere accepted-call persistence requires database "
                    f"{_COHERE_DATABASE_NAME}; connected database is "
                    f"{database_name!r}"
                )

            await conn.fetchval(
                _COHERE_ACCEPTED_CALL_ADVISORY_LOCK_SQL,
                _COHERE_CREDENTIAL_SCOPE,
            )
            inserted_row = await conn.fetchrow(
                _COHERE_ACCEPTED_CALL_INSERT_SQL,
                accepted_at_utc,
                month_start.date(),
                model_value,
                call_id,
                session_value,
                trace_value,
                source_value,
                evidence,
            )
            monthly_used = _count_value(
                await conn.fetchval(
                    _COHERE_ACCEPTED_CALL_MONTHLY_COUNT_SQL,
                    month_start.date(),
                )
            )
            rpm_used = _count_value(
                await conn.fetchval(
                    _COHERE_ACCEPTED_CALL_RPM_COUNT_SQL,
                    accepted_at_utc,
                    model_value,
                )
            )

    return CohereAcceptedCallState(
        counted=inserted_row is not None,
        monthly_used=monthly_used,
        monthly_remaining=max(0, _COHERE_MONTHLY_LIMIT - monthly_used),
        monthly_limit=_COHERE_MONTHLY_LIMIT,
        rpm_used=rpm_used,
        rpm_remaining=(
            None if rpm_limit is None else max(0, rpm_limit - rpm_used)
        ),
        rpm_limit=rpm_limit,
        month_start=month_start,
        month_end=month_end,
    )


__all__ = [
    "CohereAcceptedCallState",
    "record_cohere_accepted_call",
]
