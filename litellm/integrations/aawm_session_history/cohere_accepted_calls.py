"""Cohere wrapper around the generic locally counted accepted-call ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

from litellm.integrations.aawm_session_history.locally_counted_accepted_calls import (
    COHERE_CREDENTIAL_SCOPE,
    COHERE_LANE,
    COHERE_MONTHLY_LIMIT,
    COHERE_PROVIDER,
    calendar_month_bounds,
    cohere_trial_windows,
    record_locally_counted_accepted_call,
)
from litellm.utils import get_model_info

_COHERE_DEFAULT_SOURCE = "codex_cohere_chat_completions_adapter"
_COHERE_OBSERVATION_SOURCE = "locally_counted"


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


def _resolve_rpm_limit(model: Optional[str]) -> Optional[int]:
    if model is None:
        return None
    try:
        model_info = get_model_info(model=model, custom_llm_provider=COHERE_PROVIDER)
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


async def record_cohere_accepted_call(
    *,
    litellm_call_id: str,
    accepted_at: datetime,
    model: Optional[str],
    session_id: Optional[str],
    trace_id: Optional[str],
    source: str = _COHERE_DEFAULT_SOURCE,
) -> CohereAcceptedCallState:
    rpm_limit = _resolve_rpm_limit(model)
    state = await record_locally_counted_accepted_call(
        litellm_call_id=litellm_call_id,
        accepted_at=accepted_at,
        provider=COHERE_PROVIDER,
        credential_scope=COHERE_CREDENTIAL_SCOPE,
        source=source,
        model=model,
        lane=COHERE_LANE,
        session_id=session_id,
        trace_id=trace_id,
        windows=cohere_trial_windows(rpm_limit=rpm_limit),
    )
    monthly = state.windows["monthly"]
    rpm = state.windows["rpm"]
    month_start, month_end = calendar_month_bounds(accepted_at)
    return CohereAcceptedCallState(
        counted=state.counted,
        monthly_used=monthly.used,
        monthly_remaining=(
            monthly.remaining
            if monthly.remaining is not None
            else max(0, COHERE_MONTHLY_LIMIT - monthly.used)
        ),
        monthly_limit=monthly.limit if monthly.limit is not None else COHERE_MONTHLY_LIMIT,
        rpm_used=rpm.used,
        rpm_remaining=rpm.remaining,
        rpm_limit=rpm.limit,
        month_start=month_start,
        month_end=month_end,
    )


__all__ = [
    "CohereAcceptedCallState",
    "record_cohere_accepted_call",
]
