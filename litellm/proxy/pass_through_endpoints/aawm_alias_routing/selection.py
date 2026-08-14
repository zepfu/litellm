"""Candidate selection, state construction, availability shaping, and selection errors.

Wave 5B extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Dependencies on the god module are injected via :func:`configure_selection_runtime`.
Direct imports from sibling Wave 4/5A modules (``lane_keys``, ``snapshot_select``,
``openrouter_quota``, ``policy``) are used where those modules own the symbols.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from fastapi import HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.utils import get_model_info

from . import cooldown_state as _cooldown_state
from .cooldown_state import _attach_aawm_alias_routing_state_sources
from .lane_keys import (
    _codex_auto_agent_candidate_key,
    _resolve_anthropic_auto_agent_native_cooldown_lane_key,
    _resolve_codex_auto_agent_openai_cooldown_lane_key,
    _resolve_codex_auto_agent_xai_lane_key,
)
from .openrouter_quota import _apply_openrouter_durable_quota_candidate_cooldown
from .policy import (
    ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER as _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY as _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER as _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    CODEX_AUTO_AGENT_COHERE_LANE_KEY as _CODEX_AUTO_AGENT_COHERE_LANE_KEY,
    CODEX_AUTO_AGENT_COHERE_PROVIDER as _CODEX_AUTO_AGENT_COHERE_PROVIDER,
    CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY as _CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
    CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER as _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
    CODEX_AUTO_AGENT_NATIVE_PROVIDER as _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
    CODEX_AUTO_AGENT_OPENCODE_LANE_KEY as _CODEX_AUTO_AGENT_OPENCODE_LANE_KEY,
    CODEX_AUTO_AGENT_OPENCODE_PROVIDER as _CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
    CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY as _CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY,
    CODEX_AUTO_AGENT_OPENROUTER_PROVIDER as _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
    CODEX_AUTO_AGENT_XAI_PROVIDER as _CODEX_AUTO_AGENT_XAI_PROVIDER,
)
from .snapshot_select import (
    _commit_round_robin_selection,
    _lookup_active_snapshot_canonical_alias,
    _resolve_aawm_alias_selection_enumeration,
    _select_snapshot_candidates,
)
from .state import alias_routing_state

_aawm_selection = sys.modules[__name__]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / cooldown_state.py dependencies)
# ---------------------------------------------------------------------------
_get_codex_active_cooldown_state: Optional[
    Callable[[str], Awaitable[tuple[float, str]]]
] = None
_get_anthropic_active_cooldown_state: Optional[
    Callable[[str], Awaitable[tuple[float, str]]]
] = None
_get_anthropic_merged_codex_openai_cooldown_state: Optional[
    Callable[[str], Awaitable[tuple[float, str]]]
] = None
_set_codex_cooldown: Optional[Callable[[str, float], Awaitable[object]]] = None
_set_anthropic_cooldown: Optional[Callable[[str, float], Awaitable[object]]] = None
_get_codex_session_affinity: Optional[
    Callable[[Optional[str]], Awaitable[Optional[dict[str, Any]]]]
] = None
_get_anthropic_session_affinity: Optional[
    Callable[[Optional[str]], Awaitable[Optional[dict[str, Any]]]]
] = None
_get_openrouter_adapter_active_cooldown_seconds: Optional[
    Callable[[Optional[str]], Awaitable[float]]
] = None
_extract_client_product_label: Optional[
    Callable[[Request, dict[str, Any]], Optional[str]]
] = None
_resolve_codex_session_key: Optional[Callable[..., Optional[str]]] = None
_resolve_anthropic_session_key: Optional[Callable[..., Optional[str]]] = None
_has_continuation_state: Optional[Callable[[Any], bool]] = None
_is_grok_account_quota_candidate: Optional[
    Callable[[Optional[dict[str, Any]]], bool]
] = None
_get_grok_account_quota_lane_cooldown_key: Optional[
    Callable[[Any, Optional[str]], Optional[str]]
] = None
_is_kimi_code_candidate: Optional[Callable[[Optional[dict[str, Any]]], bool]] = None
_get_kimi_managed_account_cooldown_key: Optional[Callable[[], str]] = None
_get_codex_quota_observation_pool: Optional[
    Callable[[], Awaitable[Any]]
] = None
_get_codex_quota_observation_environment: Optional[
    Callable[[], Optional[str]]
] = None

_CODEX_OAUTH_QUOTA_CACHE_TTL_SECONDS = 30.0
_CODEX_OAUTH_QUOTA_FAILURE_RETRY_SECONDS = 5.0
_CODEX_OAUTH_QUOTA_LOOKUP_TIMEOUT_SECONDS = 0.5
_CODEX_OAUTH_QUOTA_CLIENT = "codex"
_CODEX_OAUTH_QUOTA_SOURCE = "codex_quota_poll"
_CODEX_OAUTH_WEEKLY_BALANCE_BAND_PP = 10.0
_CODEX_OAUTH_QUOTA_FAMILY_OVERALL = "overall"
_CODEX_OAUTH_QUOTA_FAMILY_SPARK = "spark"
_CODEX_OAUTH_QUOTA_CURRENT_ROWS_SQL = """
SELECT DISTINCT ON (
    NULLIF(BTRIM(evidence->>'environment'), ''),
    account_hash,
    COALESCE(model, ''),
    quota_key
)
    observed_at,
    provider,
    model,
    account_hash,
    quota_key,
    quota_period,
    quota_type,
    expected_reset_at,
    remaining_pct,
    raw_provider_fields,
    evidence,
    NULLIF(BTRIM(evidence->>'environment'), '') AS environment,
    source
FROM public.rate_limit_observations
WHERE provider = $1
  AND client = $2
  AND source = $3
  AND NULLIF(BTRIM(evidence->>'environment'), '') = $4
  AND account_hash = ANY($5::text[])
ORDER BY
    NULLIF(BTRIM(evidence->>'environment'), ''),
    account_hash,
    COALESCE(model, ''),
    quota_key,
    observed_at DESC,
    id DESC
"""


def configure_selection_runtime(
    *,
    get_codex_active_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
    get_anthropic_active_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
    get_anthropic_merged_codex_openai_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
    set_codex_cooldown: Callable[[str, float], Awaitable[object]],
    set_anthropic_cooldown: Callable[[str, float], Awaitable[object]],
    get_codex_session_affinity: Callable[[Optional[str]], Awaitable[Optional[dict[str, Any]]]],
    get_anthropic_session_affinity: Callable[[Optional[str]], Awaitable[Optional[dict[str, Any]]]],
    get_openrouter_adapter_active_cooldown_seconds: Callable[[Optional[str]], Awaitable[float]],
    extract_client_product_label: Callable[[Request, dict[str, Any]], Optional[str]],
    resolve_codex_session_key: Callable[..., Optional[str]],
    resolve_anthropic_session_key: Callable[..., Optional[str]],
    has_continuation_state: Callable[[Any], bool],
    is_grok_account_quota_candidate: Callable[[Optional[dict[str, Any]]], bool],
    get_grok_account_quota_lane_cooldown_key: Callable[[Any, Optional[str]], Optional[str]],
    is_kimi_code_candidate: Callable[[Optional[dict[str, Any]]], bool],
    get_kimi_managed_account_cooldown_key: Callable[[], str],
    get_codex_quota_observation_pool: Optional[
        Callable[[], Awaitable[Any]]
    ] = None,
    get_codex_quota_observation_environment: Optional[
        Callable[[], Optional[str]]
    ] = None,
) -> None:
    """Bind god-module / cooldown_state.py owned dependencies."""
    global _get_codex_active_cooldown_state
    _get_codex_active_cooldown_state = get_codex_active_cooldown_state
    global _get_anthropic_active_cooldown_state
    _get_anthropic_active_cooldown_state = get_anthropic_active_cooldown_state
    global _get_anthropic_merged_codex_openai_cooldown_state
    _get_anthropic_merged_codex_openai_cooldown_state = get_anthropic_merged_codex_openai_cooldown_state
    global _set_codex_cooldown
    _set_codex_cooldown = set_codex_cooldown
    global _set_anthropic_cooldown
    _set_anthropic_cooldown = set_anthropic_cooldown
    global _get_codex_session_affinity
    _get_codex_session_affinity = get_codex_session_affinity
    global _get_anthropic_session_affinity
    _get_anthropic_session_affinity = get_anthropic_session_affinity
    global _get_openrouter_adapter_active_cooldown_seconds
    _get_openrouter_adapter_active_cooldown_seconds = get_openrouter_adapter_active_cooldown_seconds
    global _extract_client_product_label
    _extract_client_product_label = extract_client_product_label
    global _resolve_codex_session_key
    _resolve_codex_session_key = resolve_codex_session_key
    global _resolve_anthropic_session_key
    _resolve_anthropic_session_key = resolve_anthropic_session_key
    global _has_continuation_state
    _has_continuation_state = has_continuation_state
    global _is_grok_account_quota_candidate
    _is_grok_account_quota_candidate = is_grok_account_quota_candidate
    global _get_grok_account_quota_lane_cooldown_key
    _get_grok_account_quota_lane_cooldown_key = get_grok_account_quota_lane_cooldown_key
    global _is_kimi_code_candidate
    _is_kimi_code_candidate = is_kimi_code_candidate
    global _get_kimi_managed_account_cooldown_key
    _get_kimi_managed_account_cooldown_key = get_kimi_managed_account_cooldown_key
    global _get_codex_quota_observation_pool
    _get_codex_quota_observation_pool = get_codex_quota_observation_pool
    global _get_codex_quota_observation_environment
    _get_codex_quota_observation_environment = (
        get_codex_quota_observation_environment
    )

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _auto_agent_alias_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_codex_request_redispatch_ordinal(
    request_body: dict[str, Any],
) -> Optional[int]:
    metadata = request_body.get("litellm_metadata")
    for candidate in (
        metadata if isinstance(metadata, dict) else None,
        request_body,
    ):
        if not isinstance(candidate, dict):
            continue
        for field in (
            "redispatch_ordinal",
            "agent_redispatch_ordinal",
            "dispatch_ordinal",
            "aawm_redispatch_ordinal",
        ):
            value = candidate.get(field)
            if isinstance(value, bool):
                continue
            float_value = _auto_agent_alias_float(value)
            if float_value is None or float_value <= 0:
                continue
            if not math.isfinite(float_value):
                continue
            int_value = int(float_value)
            if int_value <= 0:
                continue
            if float(int_value) != float_value:
                continue
            return int_value
    return None


def _resolve_codex_request_mode_and_ordinal(
    *,
    has_continuation_state: bool,
    request_body: dict[str, Any],
) -> tuple[str, Optional[int]]:
    redispatch_ordinal = _extract_codex_request_redispatch_ordinal(request_body)
    if redispatch_ordinal is not None:
        return "fresh_redispatch", redispatch_ordinal
    if has_continuation_state:
        return "ordinary_continuation", None
    return "fresh", None


# ---------------------------------------------------------------------------
# Candidate public shaping
# ---------------------------------------------------------------------------


def _codex_auto_agent_candidate_public_shape(
    candidate: dict[str, Any],
    *,
    lane_key: Optional[str] = None,
    cooldown_seconds: Optional[float] = None,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    shaped: dict[str, Any] = {
        "provider": candidate["provider"],
        "model": candidate["model"],
        "route_family": candidate["route_family"],
        "last_resort": bool(candidate.get("last_resort")),
    }
    for source_field, public_field in (
        ("codex_oauth_account_label", "account_label"),
        ("codex_oauth_account_hash", "account_hash"),
        ("codex_oauth_lane_key", "account_lane"),
        ("codex_oauth_account_priority", "account_priority"),
        ("codex_oauth_account_weight", "account_weight"),
        ("codex_oauth_credential_affinity", "credential_affinity"),
        ("codex_oauth_selection_strategy", "selection_strategy"),
        ("codex_oauth_balance_band_pp", "balance_band_percentage_points"),
        ("codex_oauth_within_band_strategy", "within_band_strategy"),
    ):
        value = candidate.get(source_field)
        if value is not None:
            shaped[public_field] = value
    if lane_key is not None:
        shaped["lane_key"] = lane_key
    if cooldown_seconds is not None:
        shaped["cooldown_seconds"] = round(float(cooldown_seconds), 3)
    if reason is not None:
        shaped["reason"] = reason
    return shaped


def _codex_oauth_routing_candidate_fields(inventory: Any) -> dict[str, Any]:
    routing = inventory.routing
    return {
        "codex_oauth_credential_affinity": routing.credential_affinity,
        "codex_oauth_selection_strategy": routing.strategy,
        "codex_oauth_balance_band_pp": routing.balance_band_percentage_points,
        "codex_oauth_within_band_strategy": routing.within_band_strategy,
    }


# ---------------------------------------------------------------------------
# Candidate availability
# ---------------------------------------------------------------------------


def _is_auto_agent_candidate_state_available(state: dict[str, Any]) -> bool:
    return state["cooldown_seconds"] <= 0 and state.get("skip_reason") is None


def _cohere_observation_exhausted(
    observation: Mapping[str, Any],
    *,
    period: str,
    rpm_limit: Optional[float] = None,
) -> bool:
    if period == "rpm" and rpm_limit is None:
        return False
    used_names = (
        ("quota_used", "monthly_used", "used_monthly", "used")
        if period == "monthly"
        else ("quota_used", "rpm_used", "rolling_minute_used", "used")
    )
    used = next(
        (
            float(value)
            for name in used_names
            if (value := observation.get(name)) is not None
            and not isinstance(value, bool)
            and _is_finite_number(value)
        ),
        None,
    )
    if used is None:
        return False
    if period == "monthly":
        return used >= 1000
    return used >= float(rpm_limit)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _cohere_local_quota_exhausted(
    candidate: Mapping[str, Any],
    *,
    state_manager: Any = alias_routing_state,
    now_epoch: Optional[float] = None,
) -> tuple[bool, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if (
        str(candidate.get("provider") or "").strip().lower() != "cohere"
        or str(candidate.get("lane_key") or "").strip() != "cohere_native"
    ):
        return False, None, None
    model = str(candidate.get("model") or "").strip()
    monthly = state_manager.resolve_cohere_monthly_observation(
        now_epoch=now_epoch
    )
    if _cohere_observation_exhausted(monthly or {}, period="monthly"):
        return True, monthly, None
    try:
        model_info = get_model_info(
            model=model,
            custom_llm_provider="cohere",
        )
    except Exception:
        model_info = None
    rpm = model_info.get("rpm") if isinstance(model_info, Mapping) else None
    if not _is_finite_number(rpm) or float(rpm) <= 0:
        return False, monthly, None
    rpm_observation = state_manager.resolve_cohere_rpm_observation(
        model=model,
        now_epoch=now_epoch,
    )
    return (
        _cohere_observation_exhausted(
            rpm_observation or {},
            period="rpm",
            rpm_limit=float(rpm),
        ),
        monthly,
        rpm_observation,
    )


def _apply_cohere_local_quota_state(
    state: dict[str, Any],
    *,
    now_epoch: Optional[float] = None,
) -> dict[str, Any]:
    candidate = state.get("candidate")
    if not isinstance(candidate, Mapping):
        return state
    candidate_with_lane = {**candidate, "lane_key": state.get("lane_key")}
    exhausted, monthly, rpm = _cohere_local_quota_exhausted(
        candidate_with_lane,
        now_epoch=now_epoch,
    )
    for period, observation in (("monthly", monthly), ("rpm", rpm)):
        if observation is not None:
            state.setdefault("cohere_quota_observations", {})[period] = observation
    if exhausted and state.get("skip_reason") is None:
        state["skip_reason"] = "quota_exhausted"
        state["cooldown_state_source"] = "cohere_local_quota"
    return state


def _build_auto_agent_skipped_candidates_from_states(
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for state in states:
        if _is_auto_agent_candidate_state_available(state):
            continue
        shaped = _codex_auto_agent_candidate_public_shape(
            state["candidate"],
            lane_key=state["lane_key"],
            cooldown_seconds=(state["cooldown_seconds"] if state["cooldown_seconds"] > 0 else None),
            reason=state.get("skip_reason") or "cooldown",
        )
        for field in (
            "cooldown_state_source",
            "cooldown_scope",
            "failure_phase",
            "attempted_provider_call",
            "auth_status",
            "quota_remaining_pct",
            "quota_snapshot_age_seconds",
            "quota_windows",
        ):
            if field in state:
                shaped[field] = state[field]
        skipped.append(shaped)
    return skipped


# ---------------------------------------------------------------------------
# Request-local cooldown / exclusion helpers
# ---------------------------------------------------------------------------


def _get_codex_auto_agent_request_local_cooldown_key(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
) -> str:
    return _codex_auto_agent_candidate_key(
        candidate,
        lane_key or "__default__",
        epoch_tag=candidate.get("config_epoch_tag"),
    )


def _get_codex_auto_agent_request_local_cooldown_state(
    request: Request,
) -> dict[str, float]:
    state = getattr(request.state, "aawm_alias_request_local_cooldown_until", None)
    if isinstance(state, dict):
        return state
    state = {}
    setattr(request.state, "aawm_alias_request_local_cooldown_until", state)
    return state


def _get_codex_auto_agent_request_local_cooldown_seconds(
    request: Request,
    *,
    cooldown_key: str,
) -> float:
    until = _get_codex_auto_agent_request_local_cooldown_state(request).get(
        cooldown_key,
        0.0,
    )
    remaining = max(0.0, until - time.monotonic())
    if remaining <= 0:
        _get_codex_auto_agent_request_local_cooldown_state(request).pop(
            cooldown_key,
            None,
        )
        return 0.0
    return remaining


def _set_codex_auto_agent_request_local_cooldown(
    request: Request,
    *,
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    ttl_seconds = max(0.0, float(cooldown_seconds))
    if ttl_seconds <= 0:
        return
    until = time.monotonic() + ttl_seconds
    state = _get_codex_auto_agent_request_local_cooldown_state(request)
    current_until = state.get(cooldown_key, 0.0)
    if until > current_until:
        state[cooldown_key] = until


def _get_codex_auto_agent_request_local_excluded_keys(
    request: Request,
) -> set[str]:
    excluded = getattr(request.state, "aawm_alias_request_local_excluded_keys", None)
    if isinstance(excluded, set):
        return excluded
    excluded = set()
    setattr(request.state, "aawm_alias_request_local_excluded_keys", excluded)
    return excluded


def _codex_oauth_candidate_slot(
    candidate: dict[str, Any],
) -> Optional[str]:
    if not candidate.get("codex_oauth_account_hash"):
        return None
    return "{}:{}:{}:{}".format(
        candidate.get("provider") or "",
        candidate.get("model") or "",
        candidate.get("route_family") or "",
        candidate.get("config_epoch_tag") or "",
    )


def _get_codex_oauth_request_local_blocked_slots(
    request: Request,
) -> set[str]:
    blocked = getattr(
        request.state,
        "aawm_codex_oauth_request_local_blocked_slots",
        None,
    )
    if isinstance(blocked, set):
        return blocked
    blocked = set()
    setattr(
        request.state,
        "aawm_codex_oauth_request_local_blocked_slots",
        blocked,
    )
    return blocked


def _block_codex_oauth_request_local_candidate_slot(
    request: Request,
    *,
    candidate: dict[str, Any],
) -> None:
    slot = _codex_oauth_candidate_slot(candidate)
    if slot is not None:
        _get_codex_oauth_request_local_blocked_slots(request).add(slot)


def _get_codex_oauth_request_local_failover_context(
    request: Request,
) -> Optional[dict[str, Any]]:
    context = getattr(
        request.state,
        "aawm_codex_oauth_request_local_failover_context",
        None,
    )
    return dict(context) if isinstance(context, dict) else None


def _apply_codex_oauth_failover_context_to_state(
    request: Request,
    state: dict[str, Any],
) -> dict[str, Any]:
    candidate = state["candidate"]
    if not _is_codex_oauth_account_candidate(candidate):
        return state
    state["failover_ordinal"] = 0
    context = _get_codex_oauth_request_local_failover_context(request)
    if (
        context is None
        or context.get("slot") != _codex_oauth_candidate_slot(candidate)
        or context.get("prior_account_hash")
        == candidate.get("codex_oauth_account_hash")
    ):
        return state
    state["failover_ordinal"] = 1
    state["prior_account_outcome"] = dict(
        context.get("prior_account_outcome") or {}
    )
    return state


def _plan_codex_oauth_account_failover(
    request: Request,
    *,
    candidate: dict[str, Any],
    selection: dict[str, Any],
    attempt_record: dict[str, Any],
    error_class: str,
    has_continuation_state: bool,
    has_previous_response_id: bool = False,
) -> bool:
    """Plan the sole request-local account move after a pre-response failure."""
    if not _is_codex_oauth_account_candidate(candidate):
        return False
    interchangeable = (
        candidate.get("codex_oauth_credential_affinity") == "interchangeable"
    )
    if has_continuation_state and (
        not interchangeable or has_previous_response_id
    ):
        return False

    failover_ordinal = int(selection.get("failover_ordinal") or 0)
    existing = _get_codex_oauth_request_local_failover_context(request)
    if failover_ordinal > 0 or existing is not None:
        _block_codex_oauth_request_local_candidate_slot(
            request,
            candidate=candidate,
        )
        attempt_record["account_failover_limit_reached"] = True
        return False

    if error_class not in {
        "capacity_exhausted",
        "rate_limited",
        "usage_limit_reached",
        "candidate_unavailable",
    }:
        return False

    prior_account_outcome: dict[str, Any] = {
        "account_label": candidate.get("codex_oauth_account_label"),
        "account_hash": candidate.get("codex_oauth_account_hash"),
        "account_lane": candidate.get("codex_oauth_lane_key"),
        "outcome": error_class,
        "failure_phase": attempt_record.get("failure_phase"),
        "attempted_provider_call": attempt_record.get(
            "attempted_provider_call"
        ),
    }
    for field in (
        "quota_snapshot_age_seconds",
        "quota_windows",
        "terminal_reset",
    ):
        value = selection.get(field)
        if value is not None:
            prior_account_outcome[field] = value
    prior_account_outcome = {
        key: value
        for key, value in prior_account_outcome.items()
        if value is not None
    }
    setattr(
        request.state,
        "aawm_codex_oauth_request_local_failover_context",
        {
            "slot": _codex_oauth_candidate_slot(candidate),
            "prior_account_hash": candidate.get(
                "codex_oauth_account_hash"
            ),
            "prior_account_outcome": prior_account_outcome,
        },
    )
    _exclude_codex_auto_agent_request_local_candidate_without_cooldown(
        request,
        candidate=candidate,
        lane_key=selection.get("lane_key"),
    )
    attempt_record["account_failover_planned"] = True
    return True


def _exclude_codex_auto_agent_request_local_candidate(
    request: Request,
    *,
    cooldown_key: str,
) -> None:
    _get_codex_auto_agent_request_local_excluded_keys(request).add(cooldown_key)


def _exclude_codex_auto_agent_request_local_candidate_without_cooldown(
    request: Request,
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
) -> None:
    _exclude_codex_auto_agent_request_local_candidate(
        request,
        cooldown_key=_get_codex_auto_agent_request_local_cooldown_key(
            candidate=candidate,
            lane_key=lane_key,
        ),
    )


def _apply_request_local_cooldown_from_plan(
    request: Request,
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
) -> None:
    """Apply a request-local cooldown + exclusion for a request-local plan (R3-1)."""
    request_local_key = _get_codex_auto_agent_request_local_cooldown_key(
        candidate=candidate,
        lane_key=lane_key,
    )
    _set_codex_auto_agent_request_local_cooldown(
        request,
        cooldown_key=request_local_key,
        cooldown_seconds=cooldown_seconds,
    )
    _exclude_codex_auto_agent_request_local_candidate(
        request,
        cooldown_key=request_local_key,
    )


def _apply_codex_auto_agent_request_local_candidate_state(
    request: Request,
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
) -> tuple[float, Optional[str], Optional[str]]:
    request_local_cooldown_key = _get_codex_auto_agent_request_local_cooldown_key(
        candidate=candidate,
        lane_key=lane_key,
    )
    request_local_cooldown_seconds = _get_codex_auto_agent_request_local_cooldown_seconds(
        request,
        cooldown_key=request_local_cooldown_key,
    )
    if request_local_cooldown_seconds > cooldown_seconds:
        cooldown_seconds = request_local_cooldown_seconds
        cooldown_state_source = "request_local"
    if request_local_cooldown_key in _get_codex_auto_agent_request_local_excluded_keys(request):
        if request_local_cooldown_seconds <= 0:
            cooldown_seconds = max(cooldown_seconds, 0.001)
        cooldown_state_source = "request_local"
        skip_reason = "request_local_transient_failure"
    account_slot = _codex_oauth_candidate_slot(candidate)
    if (
        account_slot is not None
        and account_slot
        in _get_codex_oauth_request_local_blocked_slots(request)
    ):
        cooldown_state_source = "request_local"
        skip_reason = "account_failover_limit"
    return cooldown_seconds, cooldown_state_source, skip_reason


# ---------------------------------------------------------------------------
# Forced / adapter / Kimi / Grok lane cooldown application
# ---------------------------------------------------------------------------


async def _apply_codex_auto_agent_forced_candidate_cooldown(
    *,
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    assert _set_codex_cooldown is not None
    await _set_codex_cooldown(cooldown_key, cooldown_seconds)


async def _apply_anthropic_auto_agent_forced_candidate_cooldown(
    *,
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    assert _set_anthropic_cooldown is not None
    await _set_anthropic_cooldown(cooldown_key, cooldown_seconds)


async def _apply_codex_auto_agent_adapter_local_candidate_cooldown(
    *,
    candidate: dict[str, Any],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
) -> tuple[float, Optional[str], Optional[str]]:
    """Merge process-local adapter cooldown evidence into alias selection state."""
    provider = candidate.get("provider")
    model = candidate.get("model")
    adapter_seconds = 0.0
    if provider == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        assert _get_openrouter_adapter_active_cooldown_seconds is not None
        adapter_seconds = await _get_openrouter_adapter_active_cooldown_seconds(model)
    if adapter_seconds <= 0:
        return cooldown_seconds, cooldown_state_source, skip_reason
    if adapter_seconds > cooldown_seconds:
        cooldown_seconds = adapter_seconds
        cooldown_state_source = "adapter_local"
    if cooldown_seconds > 0 and skip_reason is None:
        skip_reason = "adapter_cooldown"
    return cooldown_seconds, cooldown_state_source, skip_reason


async def _apply_kimi_code_managed_account_lane_cooldown(
    *,
    candidate: dict[str, Any],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
    get_active_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
) -> tuple[float, Optional[str], Optional[str], Optional[str]]:
    """Apply one Kimi managed-account lane without conflating candidate gates."""
    assert _is_kimi_code_candidate is not None
    assert _get_kimi_managed_account_cooldown_key is not None
    if not _is_kimi_code_candidate(candidate):
        return cooldown_seconds, cooldown_state_source, skip_reason, None
    managed_seconds, managed_source = await get_active_cooldown_state(_get_kimi_managed_account_cooldown_key())
    if managed_seconds <= 0:
        return cooldown_seconds, cooldown_state_source, skip_reason, None
    if managed_seconds >= cooldown_seconds:
        cooldown_seconds = managed_seconds
        cooldown_state_source = f"kimi_managed_account:{managed_source}"
    return cooldown_seconds, cooldown_state_source, skip_reason, "managed_account"


async def _apply_codex_auto_agent_grok_account_lane_cooldown(
    *,
    candidate: Any,
    lane_key: Optional[str],
    cooldown_seconds: float,
    cooldown_state_source: Optional[str],
    skip_reason: Optional[str],
    get_active_cooldown_state: Callable[[str], Awaitable[tuple[float, str]]],
) -> tuple[float, Optional[str], Optional[str]]:
    assert _get_grok_account_quota_lane_cooldown_key is not None
    lane_cooldown_key = _get_grok_account_quota_lane_cooldown_key(
        candidate,
        lane_key,
    )
    if lane_cooldown_key is None:
        return cooldown_seconds, cooldown_state_source, skip_reason
    lane_seconds, lane_source = await get_active_cooldown_state(lane_cooldown_key)
    if lane_seconds > cooldown_seconds:
        cooldown_seconds = lane_seconds
        cooldown_state_source = lane_source
    if lane_seconds > 0 and skip_reason is None:
        skip_reason = "account_quota_cooldown"
    return cooldown_seconds, cooldown_state_source, skip_reason


# ---------------------------------------------------------------------------
# Candidate lookup
# ---------------------------------------------------------------------------


def _find_codex_auto_agent_candidate(
    provider: Any,
    model: Any,
    *,
    alias_model: str,
    client_product_label: Optional[str] = None,
    request: Request,
) -> Optional[dict[str, Any]]:
    candidates: Sequence[dict[str, Any]] = _resolve_aawm_alias_selection_enumeration(
        request,
        alias_model,
        ingress="codex",
        client_product_label=client_product_label,
    ).candidates
    for candidate in candidates:
        if candidate["provider"] == provider and candidate["model"] == model:
            return dict(candidate)
    return None


def _find_codex_auto_agent_affinity_candidate(
    affinity: dict[str, Any],
    *,
    alias_model: str,
    client_product_label: Optional[str],
    request: Request,
) -> Optional[dict[str, Any]]:
    """Resolve a pinned candidate without applying new-request eligibility gates.

    Snapshot-established affinity is checked against the captured snapshot's
    full alias membership so schedule-only changes do not evict an in-flight
    continuation.
    """
    for candidate in _select_snapshot_candidates(
        alias_model,
        ingress="codex",
        client_product_label=client_product_label,
        request=request,
        include_out_of_schedule=True,
    ):
        if (
            candidate["provider"] == affinity.get("provider")
            and candidate["model"] == affinity.get("model")
            and candidate.get("route_family")
            == affinity.get("route_family")
        ):
            return dict(candidate)
    return None


def _find_anthropic_auto_agent_candidate(
    provider: Any,
    model: Any,
    *,
    alias_model: str,
    client_product_label: Optional[str] = None,
    request: Request,
) -> Optional[dict[str, Any]]:
    candidates = _resolve_aawm_alias_selection_enumeration(
        request,
        alias_model,
        ingress="anthropic",
        client_product_label=client_product_label,
    ).candidates
    for candidate in candidates:
        if candidate["provider"] == provider and candidate["model"] == model:
            return dict(candidate)
    return None


def _find_anthropic_auto_agent_affinity_candidate(
    affinity: dict[str, Any],
    *,
    alias_model: str,
    client_product_label: Optional[str],
    request: Request,
) -> Optional[dict[str, Any]]:
    """Resolve a pinned Anthropic candidate without applying new-request eligibility gates.

    Mirrors _find_codex_auto_agent_affinity_candidate: snapshot-established
    affinity is checked against the active snapshot's full alias membership so
    schedule/TUI-only changes do not evict an in-flight continuation.
    """
    for candidate in _select_snapshot_candidates(
        alias_model,
        ingress="anthropic",
        client_product_label=client_product_label,
        request=request,
        include_out_of_schedule=True,
    ):
        if (
            candidate["provider"] == affinity.get("provider")
            and candidate["model"] == affinity.get("model")
            and candidate.get("route_family")
            == affinity.get("route_family")
        ):
            return dict(candidate)
    return None


def _candidate_uses_codex_oauth(
    candidate: Optional[dict[str, Any]],
) -> bool:
    return bool(
        isinstance(candidate, dict)
        and candidate.get("provider") == _CODEX_AUTO_AGENT_NATIVE_PROVIDER
        and candidate.get("route_family")
        in {
            "codex_responses",
            "anthropic_openai_responses_adapter",
        }
    )


def _is_codex_oauth_account_candidate(
    candidate: Optional[dict[str, Any]],
) -> bool:
    return bool(
        _candidate_uses_codex_oauth(candidate)
        and isinstance(candidate, dict)
        and candidate.get("codex_oauth_account_label")
        and candidate.get("codex_oauth_account_hash")
        and candidate.get("codex_oauth_lane_key")
    )


def _candidate_matches_affinity(
    candidate: dict[str, Any],
    affinity: dict[str, Any],
) -> bool:
    if not (
        candidate.get("provider") == affinity.get("provider")
        and candidate.get("model") == affinity.get("model")
        and candidate.get("route_family") == affinity.get("route_family")
    ):
        return False
    if not _candidate_uses_codex_oauth(candidate):
        return True
    if (
        affinity.get("codex_oauth_credential_affinity")
        == "interchangeable"
    ):
        return True
    return all(
        candidate.get(field) == affinity.get(field)
        for field in (
            "codex_oauth_account_label",
            "codex_oauth_account_hash",
            "codex_oauth_lane_key",
        )
    )


def _apply_codex_oauth_inventory_affinity_policy(
    affinity: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not isinstance(affinity, dict):
        return affinity
    if affinity.get("provider") != _CODEX_AUTO_AGENT_NATIVE_PROVIDER:
        return affinity
    if affinity.get("route_family") not in {
        "codex_responses",
        "anthropic_openai_responses_adapter",
    }:
        return affinity
    try:
        from litellm.secret_managers.codex_oauth_inventory import (
            load_codex_oauth_inventory,
        )

        routing = load_codex_oauth_inventory().routing
    except Exception:  # noqa: BLE001
        return affinity
    if not routing.accounts_are_interchangeable:
        return affinity
    adjusted = dict(affinity)
    for field in (
        "codex_oauth_account_label",
        "codex_oauth_account_hash",
        "codex_oauth_lane_key",
    ):
        adjusted.pop(field, None)
    adjusted["codex_oauth_credential_affinity"] = "interchangeable"
    return adjusted


def _codex_oauth_quota_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _codex_oauth_quota_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if str(parsed) == str(value).strip() else None


def _codex_oauth_quota_observation_from_row(
    row: Any,
    *,
    expected_environment: str,
) -> dict[str, Any]:
    try:
        values = dict(row)
    except Exception:
        return {}
    environment = str(values.get("environment") or "").strip()
    if environment != expected_environment:
        return {}
    raw_provider_fields = _codex_oauth_quota_json_mapping(
        values.get("raw_provider_fields")
    )
    evidence = _codex_oauth_quota_json_mapping(values.get("evidence"))
    freshness = str(
        raw_provider_fields.get("freshness_state")
        or evidence.get("freshness_state")
        or ""
    ).strip().lower()
    remaining_pct = values.get("remaining_pct")
    try:
        exhausted = freshness == "fresh" and float(remaining_pct) <= 0
    except (TypeError, ValueError):
        exhausted = False
    return {
        "provider": values.get("provider"),
        "model": values.get("model"),
        "account_hash": values.get("account_hash"),
        "environment": environment,
        "quota_key": values.get("quota_key"),
        "quota_period": values.get("quota_period"),
        "quota_type": values.get("quota_type"),
        "limit_scope": (
            evidence.get("upstream_limit_scope")
            or raw_provider_fields.get("limit_scope")
        ),
        "window_minutes": _codex_oauth_quota_int(
            raw_provider_fields.get("window_minutes")
        ),
        "remaining_pct": remaining_pct,
        "observed_at": values.get("observed_at"),
        "expected_reset_at": values.get("expected_reset_at"),
        "status": freshness or None,
        "exhausted": exhausted,
        "source": values.get("source"),
    }


async def _hydrate_codex_oauth_quota_observations(
    contexts: Sequence[dict[str, Any]],
) -> None:
    if (
        _get_codex_quota_observation_pool is None
        or _get_codex_quota_observation_environment is None
    ):
        return
    try:
        environment = str(
            _get_codex_quota_observation_environment() or ""
        ).strip()
    except Exception as exc:
        verbose_proxy_logger.debug(
            "Codex OAuth durable quota environment resolution failed open "
            "(error_class=%s)",
            exc.__class__.__name__,
        )
        return
    if not environment:
        return
    account_hashes = tuple(
        dict.fromkeys(
            str(context.get("candidate", {}).get("codex_oauth_account_hash") or "")
            .strip()
            for context in contexts
            if str(
                context.get("candidate", {}).get("codex_oauth_account_hash")
                or ""
            ).strip()
        )
    )
    due_account_hashes = (
        alias_routing_state.codex_quota_hydration_due_account_hashes(
            account_hashes,
            environment=environment,
        )
    )
    if not due_account_hashes:
        return

    async with alias_routing_state.codex_quota_hydration_lock:
        due_account_hashes = (
            alias_routing_state.codex_quota_hydration_due_account_hashes(
                account_hashes,
                environment=environment,
            )
        )
        if not due_account_hashes:
            return
        try:

            async def _fetch_rows() -> Any:
                pool = await _get_codex_quota_observation_pool()
                return await pool.fetch(
                    _CODEX_OAUTH_QUOTA_CURRENT_ROWS_SQL,
                    _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
                    _CODEX_OAUTH_QUOTA_CLIENT,
                    _CODEX_OAUTH_QUOTA_SOURCE,
                    environment,
                    list(due_account_hashes),
                )

            rows = await asyncio.wait_for(
                _fetch_rows(),
                timeout=_CODEX_OAUTH_QUOTA_LOOKUP_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            alias_routing_state.defer_codex_quota_hydration(
                due_account_hashes,
                environment=environment,
                ttl_seconds=_CODEX_OAUTH_QUOTA_FAILURE_RETRY_SECONDS,
            )
            verbose_proxy_logger.debug(
                "Codex OAuth durable quota hydration failed open "
                "(error_class=%s)",
                exc.__class__.__name__,
            )
            return

        observations = [
            observation
            for row in rows
            if (
                observation := _codex_oauth_quota_observation_from_row(
                    row,
                    expected_environment=environment,
                )
            )
        ]
        alias_routing_state.replace_normalized_quota_observations(
            observations,
            provider=_CODEX_AUTO_AGENT_NATIVE_PROVIDER,
            source=_CODEX_OAUTH_QUOTA_SOURCE,
            account_hashes=due_account_hashes,
        )
        alias_routing_state.defer_codex_quota_hydration(
            due_account_hashes,
            environment=environment,
            ttl_seconds=_CODEX_OAUTH_QUOTA_CACHE_TTL_SECONDS,
        )


async def _resolve_codex_oauth_account_candidate_contexts(
    request: Request,
    *,
    candidate_template: dict[str, Any],
    affinity: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Resolve ordered, auth-checked account contexts without carrying secrets."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.codex_oauth import (
        _codex_oauth_account_lane_key,
        _load_codex_oauth_headers_for_record,
    )
    from litellm.secret_managers.codex_oauth_inventory import (
        CodexOAuthInventoryError,
        load_codex_oauth_inventory,
    )

    model = str(candidate_template.get("model") or "")
    pinned_label: Optional[str] = None
    pinned_hash: Optional[str] = None
    pinned_lane: Optional[str] = None
    interchangeable_affinity = bool(
        affinity is not None
        and affinity.get("codex_oauth_credential_affinity")
        == "interchangeable"
    )
    if affinity is not None:
        pinned_label = str(
            affinity.get("codex_oauth_account_label") or ""
        ).strip() or None
        pinned_hash = str(
            affinity.get("codex_oauth_account_hash") or ""
        ).strip() or None
        pinned_lane = str(
            affinity.get("codex_oauth_lane_key") or ""
        ).strip() or None
        if (
            not interchangeable_affinity
            and not all((pinned_label, pinned_hash, pinned_lane))
        ):
            return [
                {
                    "candidate": {
                        **candidate_template,
                        **(
                            {"codex_oauth_account_label": pinned_label}
                            if pinned_label
                            else {}
                        ),
                        **(
                            {"codex_oauth_account_hash": pinned_hash}
                            if pinned_hash
                            else {}
                        ),
                        **(
                            {"codex_oauth_lane_key": pinned_lane}
                            if pinned_lane
                            else {}
                        ),
                    },
                    "lane_key": pinned_lane or "codex-oauth:unavailable",
                    "auth_status": "degraded",
                    "skip_reason": "auth_degraded",
                    "failure_phase": "affinity_account_context_missing",
                    "attempted_provider_call": False,
                }
            ]

    try:
        inventory = load_codex_oauth_inventory()
        routing_fields = {
            "codex_oauth_credential_affinity": (
                inventory.routing.credential_affinity
            ),
            "codex_oauth_selection_strategy": inventory.routing.strategy,
            "codex_oauth_balance_band_pp": (
                inventory.routing.balance_band_percentage_points
            ),
            "codex_oauth_within_band_strategy": (
                inventory.routing.within_band_strategy
            ),
        }
        if pinned_label is not None and not interchangeable_affinity:
            records = (
                inventory.select_record(label=pinned_label, model=model),
            )
        else:
            records = inventory.ordered_records(
                enabled_only=True,
                model=model,
            )
    except CodexOAuthInventoryError:
        records = ()
        routing_fields = {}

    if not records:
        unavailable_candidate = dict(candidate_template)
        for field, value in (
            ("codex_oauth_account_label", pinned_label),
            ("codex_oauth_account_hash", pinned_hash),
            ("codex_oauth_lane_key", pinned_lane),
        ):
            if value is not None:
                unavailable_candidate[field] = value
        return [
            {
                "candidate": unavailable_candidate,
                "lane_key": pinned_lane or "codex-oauth:unavailable",
                "auth_status": "degraded",
                "skip_reason": "auth_degraded",
                "failure_phase": (
                    "affinity_account_unavailable"
                    if affinity is not None
                    else "account_inventory_unavailable"
                ),
                "attempted_provider_call": False,
            }
        ]

    contexts: list[dict[str, Any]] = []
    for record in records:
        lane_key = _codex_oauth_account_lane_key(
            account_label=record.label,
            account_hash=record.expected_account_hash,
        )
        account_candidate = {
            **candidate_template,
            "codex_oauth_account_label": record.label,
            "codex_oauth_account_hash": record.expected_account_hash,
            "codex_oauth_lane_key": lane_key,
            "codex_oauth_account_priority": record.priority,
            "codex_oauth_account_weight": record.weight,
            **routing_fields,
        }
        context: dict[str, Any] = {
            "candidate": account_candidate,
            "lane_key": lane_key,
            "auth_status": "healthy",
        }
        if (
            pinned_hash is not None
            and (
                record.expected_account_hash != pinned_hash
                or pinned_lane != lane_key
            )
        ):
            context.update(
                {
                    "auth_status": "degraded",
                    "skip_reason": "auth_degraded",
                    "failure_phase": "affinity_account_identity_mismatch",
                    "attempted_provider_call": False,
                }
            )
            contexts.append(context)
            continue
        try:
            loaded = await _load_codex_oauth_headers_for_record(
                request,
                record,
            )
        except HTTPException:
            context.update(
                {
                    "auth_status": "degraded",
                    "skip_reason": "auth_degraded",
                    "failure_phase": "pre_dispatch_auth",
                    "attempted_provider_call": False,
                }
            )
        else:
            if (
                loaded.account_hash != record.expected_account_hash
                or loaded.lane_key != lane_key
            ):
                context.update(
                    {
                        "auth_status": "degraded",
                        "skip_reason": "auth_degraded",
                        "failure_phase": "account_identity_mismatch",
                        "attempted_provider_call": False,
                    }
                )
        contexts.append(context)
    return contexts


# ---------------------------------------------------------------------------
# Candidate state construction
# ---------------------------------------------------------------------------


async def _get_anthropic_auto_agent_candidate_cooldown_state(
    *,
    provider: str,
    cooldown_key: str,
) -> tuple[float, str]:
    """Dual-ingress candidates merge Anthropic + Codex cooldown; others use Anthropic-only."""
    assert _get_anthropic_merged_codex_openai_cooldown_state is not None
    assert _get_anthropic_active_cooldown_state is not None
    if provider in {
        _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    }:
        return await _get_anthropic_merged_codex_openai_cooldown_state(cooldown_key)
    return await _get_anthropic_active_cooldown_state(cooldown_key)


async def _build_codex_auto_agent_candidate_state(  # noqa: PLR0915
    request: Request,
    *,
    candidate_template: dict[str, Any],
    openai_lane_key: Optional[str] = None,
) -> dict[str, Any]:
    assert _get_codex_active_cooldown_state is not None
    candidate = dict(candidate_template)
    account_lane_key = candidate.get("codex_oauth_lane_key")
    if isinstance(account_lane_key, str) and account_lane_key:
        openai_lane_key = account_lane_key
    elif openai_lane_key is None:
        openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    forced_cooldown_seconds: Optional[float] = None
    skip_reason: Optional[str] = None
    cooldown_state_source_override: Optional[str] = None
    failure_phase: Optional[str] = None
    attempted_provider_call: Optional[bool] = None
    if candidate["provider"] == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_XAI_PROVIDER:
        lane_key = _resolve_codex_auto_agent_xai_lane_key(candidate)
    elif candidate["provider"] == _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_COHERE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_COHERE_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENCODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
    else:
        lane_key = openai_lane_key
    # Snapshot-resolved candidates carry the active config epoch into state keys.
    _epoch_tag = candidate.get("config_epoch_tag")
    cooldown_key = _codex_auto_agent_candidate_key(candidate, lane_key, epoch_tag=_epoch_tag)
    (
        cooldown_seconds,
        initial_cooldown_state_source,
    ) = await _get_codex_active_cooldown_state(cooldown_key)
    cooldown_state_source: Optional[str] = initial_cooldown_state_source
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_codex_auto_agent_grok_account_lane_cooldown(
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
        get_active_cooldown_state=_get_codex_active_cooldown_state,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
        managed_account_cooldown_scope,
    ) = await _apply_kimi_code_managed_account_lane_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
        get_active_cooldown_state=_get_codex_active_cooldown_state,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = _apply_codex_auto_agent_request_local_candidate_state(
        request,
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_codex_auto_agent_adapter_local_candidate_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_openrouter_durable_quota_candidate_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    quota_state = _apply_cohere_local_quota_state(
        {
            "candidate": candidate,
            "lane_key": lane_key,
            "skip_reason": skip_reason,
            "cooldown_state_source": cooldown_state_source,
        }
    )
    skip_reason = quota_state.get("skip_reason")
    cooldown_state_source = quota_state.get("cooldown_state_source")
    if forced_cooldown_seconds is not None and forced_cooldown_seconds > cooldown_seconds:
        await _apply_codex_auto_agent_forced_candidate_cooldown(
            cooldown_key=cooldown_key,
            cooldown_seconds=forced_cooldown_seconds,
        )
        cooldown_seconds = forced_cooldown_seconds
        cooldown_state_source = cooldown_state_source_override or "forced_candidate_cooldown"
    state: dict[str, Any] = {
        "candidate": candidate,
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "cooldown_seconds": cooldown_seconds,
        "cooldown_state_source": cooldown_state_source,
    }
    if _epoch_tag is not None:
        state["config_epoch_tag"] = _epoch_tag
    if skip_reason is not None:
        state["skip_reason"] = skip_reason
    if failure_phase is not None:
        state["failure_phase"] = failure_phase
    if attempted_provider_call is not None:
        state["attempted_provider_call"] = attempted_provider_call
    if managed_account_cooldown_scope is not None:
        state["cooldown_scope"] = managed_account_cooldown_scope
    if quota_state.get("cohere_quota_observations"):
        state["cohere_quota_observations"] = quota_state[
            "cohere_quota_observations"
        ]
    return state



def _is_codex_oauth_spark_model(model: Any) -> bool:
    """Return True when the request/upstream model is Spark-scoped."""
    text = str(model or "").strip().lower()
    return "spark" in text


def _codex_oauth_quota_family_for_model(model: Any) -> str:
    """Map a request model onto the independent overall/Spark quota family."""
    if _is_codex_oauth_spark_model(model):
        return "spark"
    return "overall"


def _codex_oauth_quota_observation_family(observation: Mapping[str, Any]) -> str:
    """Classify one quota row/window using the durable model/quota_key schema.

    The observation schema discriminates families primarily via ``model`` (for
    example ``gpt-5.3-codex-spark`` vs overall Codex models or blank poll model
    rows) and secondarily via ``quota_key`` text. Families are never collapsed.
    """
    model = str(observation.get("model") or "").strip().lower()
    quota_key = str(observation.get("quota_key") or "").strip().lower()
    if "spark" in model or "spark" in quota_key:
        return "spark"
    return "overall"


def _filter_codex_oauth_quota_windows_for_family(
    windows: Sequence[Any],
    *,
    family: str,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        if _codex_oauth_quota_observation_family(window) == family:
            filtered.append(window)
    return filtered


def _codex_oauth_window_remaining_pct(window: Mapping[str, Any]) -> Optional[float]:
    try:
        remaining = float(window.get("remaining_pct"))
    except (TypeError, ValueError):
        return None
    if remaining != remaining or remaining in {float("inf"), float("-inf")}:
        return None
    return remaining


def _codex_oauth_window_is_weekly(window: Mapping[str, Any]) -> bool:
    period = str(
        window.get("quota_period") or window.get("window") or ""
    ).strip().lower()
    try:
        window_minutes = int(window.get("window_minutes"))
    except (TypeError, ValueError):
        window_minutes = 0
    return period == "seven_day" or window_minutes == 10080


def _codex_oauth_weekly_remaining_pct_from_windows(
    windows: Sequence[Any],
) -> Optional[float]:
    """Return the conservative fresh weekly remaining percentage, if known."""
    weekly_values: list[float] = []
    for window in windows:
        if not isinstance(window, dict) or not _codex_oauth_window_is_weekly(window):
            continue
        status = str(window.get("status") or "").strip().lower()
        if status and status != "fresh":
            continue
        remaining = _codex_oauth_window_remaining_pct(window)
        if remaining is not None:
            weekly_values.append(remaining)
    if not weekly_values:
        return None
    return min(weekly_values)


def _codex_oauth_state_weekly_remaining_pct(state: Mapping[str, Any]) -> Optional[float]:
    windows = state.get("quota_windows")
    if isinstance(windows, list):
        remaining = _codex_oauth_weekly_remaining_pct_from_windows(windows)
        if remaining is not None:
            return remaining
    observation = state.get("quota_observation")
    if isinstance(observation, Mapping):
        obs_windows = observation.get("windows")
        if isinstance(obs_windows, list):
            family = _codex_oauth_quota_family_for_model(
                (state.get("candidate") or {}).get("model")
                if isinstance(state.get("candidate"), Mapping)
                else None
            )
            filtered = _filter_codex_oauth_quota_windows_for_family(
                obs_windows,
                family=family,
            )
            return _codex_oauth_weekly_remaining_pct_from_windows(filtered)
    return None


def _prefer_codex_oauth_weekly_balanced_state(
    states: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Soft weekly account balancing across otherwise eligible OAuth accounts.

    Prefer the less-depleted account when the applicable weekly remaining spread
    is at least 10 percentage points. Within the band, either eligible account
    is permitted and inventory/candidate order is preserved. Confirmed five-hour
    exhaustion is handled before this function by excluding those accounts from
    ``states``.
    """
    oauth_states = [
        state
        for state in states
        if isinstance(state.get("candidate"), dict)
        and _is_codex_oauth_account_candidate(state["candidate"])
    ]
    if len(oauth_states) < 2:
        return None

    scored: list[tuple[float, dict[str, Any]]] = []
    unscored: list[dict[str, Any]] = []
    for state in oauth_states:
        remaining = _codex_oauth_state_weekly_remaining_pct(state)
        if remaining is None:
            unscored.append(state)
        else:
            scored.append((remaining, state))

    if len(scored) < 2:
        # Balancing requires comparable weekly evidence for at least two accounts.
        return None

    max_remaining = max(remaining for remaining, _state in scored)
    min_remaining = min(remaining for remaining, _state in scored)
    spread = max_remaining - min_remaining
    if spread + 1e-9 < 10.0:
        # Within the soft band: either eligible account is permitted.
        return None

    preferred = [
        state
        for remaining, state in scored
        if abs(remaining - max_remaining) <= 0.001
    ]
    if not preferred:
        return None
    selected = preferred[0]
    selected = dict(selected)
    selected["selection_diagnostics"] = {
        **dict(selected.get("selection_diagnostics") or {}),
        "strategy": "weekly_quota_balance",
        "quota_family": _codex_oauth_quota_family_for_model(
            selected.get("candidate", {}).get("model")
        ),
        "weekly_balance_band_pp": 10.0,
        "weekly_remaining_spread_pp": round(spread, 3),
        "preferred_weekly_remaining_pct": max_remaining,
    }
    return selected


def _codex_oauth_dual_family_remaining(
    state: Mapping[str, Any],
) -> dict[str, Optional[float]]:
    candidate = state.get("candidate")
    if not isinstance(candidate, Mapping):
        return {"overall": None, "spark": None}
    account_hash = str(
        candidate.get("codex_oauth_account_hash") or ""
    ).strip()
    windows = alias_routing_state.resolve_normalized_quota_windows_for_account(
        provider=str(candidate.get("provider") or ""),
        account_hash=account_hash,
        max_age_seconds=30.0,
    )
    return {
        family: _codex_oauth_weekly_remaining_pct_from_windows(
            _filter_codex_oauth_quota_windows_for_family(
                windows,
                family=family,
            )
        )
        for family in ("overall", "spark")
    }


def _select_codex_oauth_weighted_round_robin_state(
    states: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    weighted_states: list[tuple[dict[str, Any], float]] = []
    for state in states:
        candidate = state["candidate"]
        try:
            weight = float(candidate.get("codex_oauth_account_weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weighted_states.append((state, max(0.01, weight)))
    minimum_weight = min(weight for _state, weight in weighted_states)
    slot_counts = [
        max(1, min(100, int(round(weight / minimum_weight))))
        for _state, weight in weighted_states
    ]
    slots = [
        state
        for ordinal in range(max(slot_counts))
        for (state, _weight), slot_count in zip(weighted_states, slot_counts)
        if ordinal < slot_count
    ]
    identities = ",".join(
        str(state["candidate"].get("codex_oauth_account_hash") or "")
        for state in states
    )
    cursor_key = ("codex_oauth_accounts", identities)
    cursor = alias_routing_state.round_robin_cursor.get(cursor_key, 0)
    selected = slots[cursor % len(slots)]
    alias_routing_state.round_robin_cursor[cursor_key] = cursor + 1
    return selected


def _prefer_codex_oauth_dual_quota_balanced_state(
    states: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if len(states) < 2:
        return states[0] if states else None
    candidate = states[0].get("candidate")
    if not isinstance(candidate, Mapping):
        return None
    if candidate.get("codex_oauth_selection_strategy") != "dual_quota_balance":
        return None
    try:
        band = float(candidate.get("codex_oauth_balance_band_pp", 10.0))
    except (TypeError, ValueError):
        band = 10.0

    observations = {
        str(state["candidate"].get("codex_oauth_account_hash") or ""): (
            _codex_oauth_dual_family_remaining(state)
        )
        for state in states
    }
    spreads: dict[str, float] = {}
    for family in ("overall", "spark"):
        values = [
            family_values[family]
            for family_values in observations.values()
            if family_values[family] is not None
        ]
        if len(values) >= 2:
            spreads[family] = max(values) - min(values)

    constrained = [
        family for family, spread in spreads.items()
        if spread + 1e-9 >= band
    ]
    if constrained:
        controlling_family = max(
            constrained,
            key=lambda family: (spreads[family], family == "overall"),
        )
        scored = [
            (
                observations[
                    str(state["candidate"].get("codex_oauth_account_hash") or "")
                ][controlling_family],
                state,
            )
            for state in states
        ]
        known = [(value, state) for value, state in scored if value is not None]
        if known:
            selected = max(known, key=lambda item: item[0])[1]
            selected = dict(selected)
            selected["selection_diagnostics"] = {
                **dict(selected.get("selection_diagnostics") or {}),
                "strategy": "dual_quota_balance",
                "balance_band_percentage_points": band,
                "controlling_quota_family": controlling_family,
                "weekly_remaining_spreads_pp": {
                    key: round(value, 3) for key, value in spreads.items()
                },
                "account_weekly_remaining_pct": observations,
            }
            return selected

    if (
        candidate.get("codex_oauth_within_band_strategy")
        == "weighted_round_robin"
    ):
        selected = dict(
            _select_codex_oauth_weighted_round_robin_state(states)
        )
        selected["selection_diagnostics"] = {
            **dict(selected.get("selection_diagnostics") or {}),
            "strategy": "weighted_round_robin",
            "balance_band_percentage_points": band,
            "weekly_remaining_spreads_pp": {
                key: round(value, 3) for key, value in spreads.items()
            },
            "account_weekly_remaining_pct": observations,
        }
        return selected
    return None


def _select_first_available_codex_oauth_account_state(
    states: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Pick the first eligible account, applying soft weekly balancing when useful."""
    available = [
        state
        for state in states
        if _is_auto_agent_candidate_state_available(state)
    ]
    if not available:
        return None
    dual_balanced = _prefer_codex_oauth_dual_quota_balanced_state(available)
    if dual_balanced is not None:
        return dual_balanced
    balanced = _prefer_codex_oauth_weekly_balanced_state(available)
    return balanced if balanced is not None else available[0]


def _format_codex_oauth_quota_reset_at(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)):
        return None
    try:
        return (
            datetime.fromtimestamp(float(value), tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OSError, OverflowError, ValueError):
        return None


def _codex_oauth_quota_window_public_shape(
    observation: dict[str, Any],
) -> dict[str, Any]:
    shaped: dict[str, Any] = {
        "remaining_pct": observation.get("remaining_pct"),
        "scope": observation.get("limit_scope"),
        "window": observation.get("quota_period"),
        "window_minutes": observation.get("window_minutes"),
        "quota_type": observation.get("quota_type"),
        "status": observation.get("status"),
        "exhausted": observation.get("exhausted"),
        "reset_at": _format_codex_oauth_quota_reset_at(
            observation.get("expected_reset_at")
        ),
        "snapshot_age_seconds": round(
            float(observation.get("observation_age_seconds") or 0.0),
            3,
        ),
    }
    return {
        key: value
        for key, value in shaped.items()
        if value is not None
    }


def _codex_oauth_quota_window_is_confirmed_exhausted(
    observation: dict[str, Any],
) -> bool:
    quota_period = str(observation.get("quota_period") or "").strip().lower()
    try:
        window_minutes = int(observation.get("window_minutes"))
    except (TypeError, ValueError):
        window_minutes = 0
    try:
        remaining_pct = float(observation.get("remaining_pct"))
    except (TypeError, ValueError):
        return False
    return (
        (quota_period in {"five_hour", "seven_day"}
         or window_minutes in {300, 10080})
        and str(observation.get("status") or "").strip().lower() == "fresh"
        and observation.get("exhausted") is True
        and remaining_pct <= 0
    )


def _build_codex_oauth_terminal_reset_information(
    states: Sequence[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    accounts: list[dict[str, Any]] = []
    reset_values: list[str] = []
    for state in states:
        if state.get("skip_reason") != "quota_exhausted":
            continue
        candidate = state.get("candidate")
        if not isinstance(candidate, dict):
            continue
        windows = state.get("quota_exhausted_windows")
        if not isinstance(windows, list) or not windows:
            continue
        account = {
            "account_label": candidate.get(
                "codex_oauth_account_label"
            ),
            "account_hash": candidate.get("codex_oauth_account_hash"),
            "account_lane": candidate.get("codex_oauth_lane_key"),
            "exhausted_windows": windows,
        }
        accounts.append(
            {
                key: value
                for key, value in account.items()
                if value is not None
            }
        )
        for window in windows:
            if isinstance(window, dict) and isinstance(
                window.get("reset_at"), str
            ):
                reset_values.append(window["reset_at"])
    if not accounts:
        return None
    terminal: dict[str, Any] = {
        "reason": "codex_oauth_quota_exhausted",
        "accounts": accounts,
    }
    if reset_values:
        terminal["next_reset_at"] = min(reset_values)
    return terminal


def _codex_oauth_public_quota_windows(
    windows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Shape quota windows while retaining inspectable family identity fields."""
    shaped_windows: list[dict[str, Any]] = []
    for window in windows:
        shaped = _codex_oauth_quota_window_public_shape(window)
        model_value = window.get("model")
        if model_value is not None and "model" not in shaped:
            shaped["model"] = model_value
        quota_key = window.get("quota_key")
        if quota_key is not None and "quota_key" not in shaped:
            shaped["quota_key"] = quota_key
        shaped_windows.append(shaped)
    return shaped_windows


def _build_family_quota_observation(
    observation: Mapping[str, Any],
    *,
    family_windows: Sequence[dict[str, Any]],
    quota_family: str,
) -> Optional[dict[str, Any]]:
    remaining_values = [
        remaining
        for window in family_windows
        if (remaining := _codex_oauth_window_remaining_pct(window)) is not None
    ]
    if not remaining_values:
        return None

    family_observation = dict(observation)
    family_observation["windows"] = list(family_windows)
    family_observation["window_count"] = len(family_windows)
    family_observation["remaining_pct"] = min(remaining_values)
    family_observation["quota_family"] = quota_family
    selected_window = min(
        family_windows,
        key=lambda window: (
            _codex_oauth_window_remaining_pct(window)
            if _codex_oauth_window_remaining_pct(window) is not None
            else float("inf")
        ),
    )
    for field in (
        "quota_key",
        "quota_type",
        "limit_scope",
        "quota_period",
        "window_minutes",
        "status",
        "exhausted",
        "expected_reset_at",
        "observed_at",
        "model",
        "source",
    ):
        if field in selected_window:
            family_observation[field] = selected_window.get(field)
    try:
        age_source = (
            selected_window.get("observation_age_seconds")
            if selected_window.get("observation_age_seconds") is not None
            else observation.get("observation_age_seconds") or 0.0
        )
        family_observation["observation_age_seconds"] = max(0.0, float(age_source))
    except (TypeError, ValueError):
        family_observation["observation_age_seconds"] = observation.get(
            "observation_age_seconds"
        )
    return family_observation


def _attach_normalized_quota_state(
    state: dict[str, Any],
    *,
    account_hash: Optional[str] = None,
) -> dict[str, Any]:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        alias_routing_state as _alias_routing_state,
    )

    candidate = state["candidate"]
    selected_account_hash = (
        account_hash
        or candidate.get("codex_oauth_account_hash")
        or None
    )
    request_model = str(candidate.get("model") or "")
    quota_family = _codex_oauth_quota_family_for_model(request_model)
    observation = _alias_routing_state.resolve_normalized_quota_observation(
        provider=str(candidate.get("provider") or ""),
        model=request_model,
        account_hash=(
            str(selected_account_hash)
            if selected_account_hash is not None
            else None
        ),
    )
    if observation is None:
        return state

    raw_windows = observation.get("windows")
    family_windows = (
        _filter_codex_oauth_quota_windows_for_family(
            raw_windows,
            family=quota_family,
        )
        if isinstance(raw_windows, list)
        else []
    )
    if not family_windows:
        # Do not let the other quota family hide or invent exhaustion/remaining.
        return state

    family_observation = _build_family_quota_observation(
        observation,
        family_windows=family_windows,
        quota_family=quota_family,
    )
    if family_observation is None:
        return state

    state["quota_observation"] = family_observation
    state["quota_family"] = quota_family
    state["quota_remaining_pct"] = family_observation["remaining_pct"]
    state["quota_snapshot_age_seconds"] = round(
        float(family_observation.get("observation_age_seconds") or 0.0),
        3,
    )
    state["quota_windows"] = _codex_oauth_public_quota_windows(family_windows)
    exhausted_source = [
        window
        for window in family_windows
        if _codex_oauth_quota_window_is_confirmed_exhausted(window)
    ]
    if exhausted_source:
        state["quota_exhausted_windows"] = _codex_oauth_public_quota_windows(
            exhausted_source
        )
    if (
        state.get("quota_exhausted_windows")
        and state.get("skip_reason") is None
    ):
        state["skip_reason"] = "quota_exhausted"
        state["cooldown_state_source"] = "normalized_quota_observation"
        state["terminal_reset"] = (
            _build_codex_oauth_terminal_reset_information([state])
        )
    return state


async def _build_anthropic_auto_agent_candidate_state(  # noqa: PLR0915
    request: Request,
    *,
    candidate_template: dict[str, Any],
    openai_lane_key: Optional[str] = None,
    anthropic_lane_key: Optional[str] = None,
) -> dict[str, Any]:
    assert _get_anthropic_active_cooldown_state is not None
    candidate = dict(candidate_template)
    account_lane_key = candidate.get("codex_oauth_lane_key")
    if isinstance(account_lane_key, str) and account_lane_key:
        openai_lane_key = account_lane_key
    elif openai_lane_key is None:
        openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    if anthropic_lane_key is None:
        anthropic_lane_key = _resolve_anthropic_auto_agent_native_cooldown_lane_key(request)
    forced_cooldown_seconds: Optional[float] = None
    skip_reason: Optional[str] = None
    cooldown_state_source_override: Optional[str] = None
    failure_phase: Optional[str] = None
    attempted_provider_call: Optional[bool] = None
    if candidate["provider"] == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_XAI_PROVIDER:
        lane_key = _resolve_codex_auto_agent_xai_lane_key(candidate)
    elif candidate["provider"] == _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENCODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
    elif candidate["provider"] == _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER:
        lane_key = anthropic_lane_key
    else:
        lane_key = openai_lane_key
    _epoch_tag = candidate.get("config_epoch_tag")
    cooldown_key = _codex_auto_agent_candidate_key(candidate, lane_key, epoch_tag=_epoch_tag)
    (
        cooldown_seconds,
        initial_cooldown_state_source,
    ) = await _get_anthropic_auto_agent_candidate_cooldown_state(
        provider=candidate["provider"],
        cooldown_key=cooldown_key,
    )
    cooldown_state_source: Optional[str] = initial_cooldown_state_source
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_codex_auto_agent_grok_account_lane_cooldown(
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
        get_active_cooldown_state=_get_anthropic_active_cooldown_state,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
        managed_account_cooldown_scope,
    ) = await _apply_kimi_code_managed_account_lane_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
        get_active_cooldown_state=lambda cooldown_key: (
            _get_anthropic_auto_agent_candidate_cooldown_state(
                provider=_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
                cooldown_key=cooldown_key,
            )
        ),
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = _apply_codex_auto_agent_request_local_candidate_state(
        request,
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_codex_auto_agent_adapter_local_candidate_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    (
        cooldown_seconds,
        cooldown_state_source,
        skip_reason,
    ) = await _apply_openrouter_durable_quota_candidate_cooldown(
        candidate=candidate,
        cooldown_seconds=cooldown_seconds,
        cooldown_state_source=cooldown_state_source,
        skip_reason=skip_reason,
    )
    if forced_cooldown_seconds is not None and forced_cooldown_seconds > cooldown_seconds:
        await _apply_anthropic_auto_agent_forced_candidate_cooldown(
            cooldown_key=cooldown_key,
            cooldown_seconds=forced_cooldown_seconds,
        )
        cooldown_seconds = forced_cooldown_seconds
        cooldown_state_source = cooldown_state_source_override or "forced_candidate_cooldown"
    state: dict[str, Any] = {
        "candidate": candidate,
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "cooldown_seconds": cooldown_seconds,
        "cooldown_state_source": cooldown_state_source,
    }
    if _epoch_tag is not None:
        state["config_epoch_tag"] = _epoch_tag
    if skip_reason is not None:
        state["skip_reason"] = skip_reason
    if failure_phase is not None:
        state["failure_phase"] = failure_phase
    if attempted_provider_call is not None:
        state["attempted_provider_call"] = attempted_provider_call
    if managed_account_cooldown_scope is not None:
        state["cooldown_scope"] = managed_account_cooldown_scope
    return state


def _apply_codex_oauth_account_context_to_state(
    request: Request,
    state: dict[str, Any],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    state["auth_status"] = context.get("auth_status") or "degraded"
    for field in (
        "skip_reason",
        "failure_phase",
        "attempted_provider_call",
    ):
        if field in context:
            state[field] = context[field]
    account_hash = state["candidate"].get("codex_oauth_account_hash")
    if isinstance(account_hash, str) and account_hash:
        state = _attach_normalized_quota_state(
            state,
            account_hash=account_hash,
        )
    return _apply_codex_oauth_failover_context_to_state(request, state)


async def _build_codex_auto_agent_affinity_candidate_state(
    request: Request,
    *,
    candidate_template: dict[str, Any],
    affinity: dict[str, Any],
) -> dict[str, Any]:
    if not _candidate_uses_codex_oauth(candidate_template):
        return _attach_normalized_quota_state(
            await _build_codex_auto_agent_candidate_state(
                request,
                candidate_template=candidate_template,
            )
        )
    contexts = await _resolve_codex_oauth_account_candidate_contexts(
        request,
        candidate_template=candidate_template,
        affinity=affinity,
    )
    await _aawm_selection._hydrate_codex_oauth_quota_observations(contexts)
    states: list[dict[str, Any]] = []
    for context in contexts:
        state = await _build_codex_auto_agent_candidate_state(
            request,
            candidate_template=context["candidate"],
            openai_lane_key=context["lane_key"],
        )
        states.append(
            _apply_codex_oauth_account_context_to_state(
                request,
                state,
                context=context,
            )
        )
    if (
        affinity.get("codex_oauth_credential_affinity")
        == "interchangeable"
    ):
        return (
            _select_first_available_codex_oauth_account_state(states)
            or states[0]
        )
    return states[0]


async def _build_anthropic_auto_agent_affinity_candidate_state(
    request: Request,
    *,
    candidate_template: dict[str, Any],
    affinity: dict[str, Any],
) -> dict[str, Any]:
    if not _candidate_uses_codex_oauth(candidate_template):
        return _attach_normalized_quota_state(
            await _build_anthropic_auto_agent_candidate_state(
                request,
                candidate_template=candidate_template,
            )
        )
    contexts = await _resolve_codex_oauth_account_candidate_contexts(
        request,
        candidate_template=candidate_template,
        affinity=affinity,
    )
    await _aawm_selection._hydrate_codex_oauth_quota_observations(contexts)
    context = contexts[0]
    state = await _build_anthropic_auto_agent_candidate_state(
        request,
        candidate_template=context["candidate"],
        openai_lane_key=context["lane_key"],
    )
    return _apply_codex_oauth_account_context_to_state(
        request,
        state,
        context=context,
    )


async def _build_codex_auto_agent_candidate_states(
    request: Request,
    *,
    alias_model: str,
    client_product_label: Optional[str] = None,
) -> list[dict[str, Any]]:
    openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    states: list[dict[str, Any]] = []
    for candidate_template in _resolve_aawm_alias_selection_enumeration(
        request,
        alias_model,
        ingress="codex",
        client_product_label=client_product_label,
    ).candidates:
        if _candidate_uses_codex_oauth(candidate_template):
            contexts = await _resolve_codex_oauth_account_candidate_contexts(
                request,
                candidate_template=candidate_template,
            )
            await _aawm_selection._hydrate_codex_oauth_quota_observations(
                contexts
            )
            for context in contexts:
                state = await _build_codex_auto_agent_candidate_state(
                    request,
                    candidate_template=context["candidate"],
                    openai_lane_key=context["lane_key"],
                )
                states.append(
                    _apply_codex_oauth_account_context_to_state(
                        request,
                        state,
                        context=context,
                    )
                )
            continue
        states.append(
            _attach_normalized_quota_state(
                await _build_codex_auto_agent_candidate_state(
                    request,
                    candidate_template=candidate_template,
                    openai_lane_key=openai_lane_key,
                )
            )
        )
    return states


async def _build_anthropic_auto_agent_candidate_states(
    request: Request,
    *,
    alias_model: str,
    client_product_label: Optional[str] = None,
) -> list[dict[str, Any]]:
    openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    anthropic_lane_key = _resolve_anthropic_auto_agent_native_cooldown_lane_key(request)
    states: list[dict[str, Any]] = []
    for candidate_template in _resolve_aawm_alias_selection_enumeration(
        request,
        alias_model,
        ingress="anthropic",
        client_product_label=client_product_label,
    ).candidates:
        if _candidate_uses_codex_oauth(candidate_template):
            contexts = await _resolve_codex_oauth_account_candidate_contexts(
                request,
                candidate_template=candidate_template,
            )
            await _aawm_selection._hydrate_codex_oauth_quota_observations(
                contexts
            )
            for context in contexts:
                state = await _build_anthropic_auto_agent_candidate_state(
                    request,
                    candidate_template=context["candidate"],
                    openai_lane_key=context["lane_key"],
                    anthropic_lane_key=anthropic_lane_key,
                )
                states.append(
                    _apply_codex_oauth_account_context_to_state(
                        request,
                        state,
                        context=context,
                    )
                )
            continue
        states.append(
            _attach_normalized_quota_state(
                await _build_anthropic_auto_agent_candidate_state(
                    request,
                    candidate_template=candidate_template,
                    openai_lane_key=openai_lane_key,
                    anthropic_lane_key=anthropic_lane_key,
                )
            )
        )
    return states


def _get_request_selection_choices(request: Request) -> dict[str, str]:
    choices = getattr(request.state, "aawm_alias_selected_choices", None)
    if isinstance(choices, dict):
        return choices
    choices = {}
    setattr(request.state, "aawm_alias_selected_choices", choices)
    return choices


def _get_request_reselection_counts(request: Request) -> dict[str, int]:
    counts = getattr(request.state, "aawm_alias_reselection_counts", None)
    if isinstance(counts, dict):
        return counts
    counts = {}
    setattr(request.state, "aawm_alias_reselection_counts", counts)
    return counts


def _weighted_choice(
    choices: Sequence[str],
    weights: dict[str, float],
) -> str:
    total = sum(max(0.0, weights.get(choice, 0.0)) for choice in choices)
    if total <= 0:
        return choices[0]
    pick = random.random() * total
    cumulative = 0.0
    for choice in choices:
        cumulative += max(0.0, weights.get(choice, 0.0))
        if pick <= cumulative:
            return choice
    return choices[-1]


def _select_round_robin_available_state(
    request: Request,
    tier: Sequence[dict[str, Any]],
    *,
    group: str,
    ingress: str,
) -> dict[str, Any]:
    token = _resolve_aawm_alias_selection_enumeration(
        request,
        group,
        ingress=ingress,
    ).commit_token
    if token is None:
        return tier[0]

    cursor_key = (token.epoch_tag, token.alias_name)
    cursor = alias_routing_state.round_robin_cursor.get(
        cursor_key,
        token.start_index,
    )
    available_by_identity = {
        (
            str(state["candidate"].get("provider") or ""),
            str(state["candidate"].get("model") or ""),
        ): state
        for state in tier
    }
    selected = tier[0]
    for offset in range(len(token.tied_candidate_ids)):
        identity = token.tied_candidate_ids[
            (cursor + offset) % len(token.tied_candidate_ids)
        ]
        if identity in available_by_identity:
            selected = available_by_identity[identity]
            break
    _commit_round_robin_selection(
        token,
        selected_candidate=selected["candidate"],
    )
    return selected


def _select_available_state(
    request: Request,
    states: Sequence[dict[str, Any]],
    *,
    ingress: str,
    last_resort: bool,
) -> Optional[dict[str, Any]]:
    available = [
        state
        for state in states
        if bool(state["candidate"].get("last_resort")) is last_resort
        and (
            (
                state["candidate"].get("last_resort")
                and state.get("skip_reason") is None
            )
            or _is_auto_agent_candidate_state_available(state)
        )
    ]
    if not available:
        return None
    highest_priority = max(
        int(state["candidate"].get("selection_priority", 0))
        for state in available
    )
    tier = [
        state
        for state in available
        if int(state["candidate"].get("selection_priority", 0))
        == highest_priority
    ]
    group = tier[0]["candidate"].get("selection_group")
    strategy = tier[0]["candidate"].get("selection_strategy")
    if not group or not strategy:
        balanced = _prefer_codex_oauth_weekly_balanced_state(tier)
        return balanced if balanced is not None else tier[0]

    states_by_choice: dict[str, list[dict[str, Any]]] = {}
    weights: dict[str, float] = {}
    for state in tier:
        candidate = state["candidate"]
        choice = str(candidate.get("selection_choice") or "")
        if not choice:
            return tier[0]
        states_by_choice.setdefault(choice, []).append(state)
        weights.setdefault(
            choice,
            float(candidate.get("selection_weight", 1.0)),
        )
    choices = list(states_by_choice)
    selected_by_group = _get_request_selection_choices(request)
    selected_choice = selected_by_group.get(str(group))
    selected_state: Optional[dict[str, Any]] = None
    if selected_choice not in states_by_choice:
        if selected_choice is not None:
            counts = _get_request_reselection_counts(request)
            counts[str(group)] = counts.get(str(group), 0) + 1
        if strategy == "proportional":
            selected_choice = _weighted_choice(choices, weights)
        elif strategy == "round_robin":
            selected_state = _select_round_robin_available_state(
                request,
                tier,
                group=str(group),
                ingress=ingress,
            )
            selected_choice = str(
                selected_state["candidate"].get("selection_choice") or ""
            )
        elif strategy in {
            "highest_quota_available",
            "lowest_quota_available",
        }:
            quota_by_choice: dict[str, float] = {}
            for choice, choice_states in states_by_choice.items():
                values = [
                    float(state["quota_remaining_pct"])
                    for state in choice_states
                    if state.get("quota_remaining_pct") is not None
                ]
                if values:
                    quota_by_choice[choice] = min(values)
            if not quota_by_choice:
                return None
            target = (
                max(quota_by_choice.values())
                if strategy == "highest_quota_available"
                else min(quota_by_choice.values())
            )
            tied = [
                choice
                for choice in choices
                if choice in quota_by_choice
                and abs(quota_by_choice[choice] - target) <= 0.001
            ]
            selected_choice = random.choice(tied)
        else:
            selected_choice = choices[0]
        selected_by_group[str(group)] = selected_choice

    selected = selected_state or states_by_choice[selected_choice][0]
    total_weight = sum(max(0.0, weights[choice]) for choice in choices)
    selected["selection_diagnostics"] = {
        "strategy": strategy,
        "group": group,
        "available_choices": choices,
        "normalized_available_weights": {
            choice: (
                max(0.0, weights[choice]) / total_weight
                if total_weight > 0
                else 1.0 / len(choices)
            )
            for choice in choices
        },
        "selected_choice": selected_choice,
        "reselection_count": _get_request_reselection_counts(request).get(
            str(group), 0
        ),
    }
    return selected


# ---------------------------------------------------------------------------
# Selection errors: in-flight cooldown + redispatch
# ---------------------------------------------------------------------------


def _raise_codex_auto_agent_in_flight_cooldown(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
) -> None:
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_session_affinity_cooldown",
    )
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "Codex auto-agent alias target is cooling down for an in-flight "
                    "session; provider switching is disabled for stateful agent "
                    "continuations. Redispatch a fresh agent attempt to re-run the "
                    "auto selector."
                ),
                "type": "invalid_request_error",
                "code": "aawm_codex_auto_agent_in_flight_provider_cooling_down",
            },
            "candidate": shaped_candidate,
        },
        headers={"Retry-After": str(int(max(1.0, cooldown_seconds)))},
    )


def _raise_anthropic_auto_agent_in_flight_cooldown(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
) -> None:
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_session_affinity_cooldown",
    )
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "Anthropic auto-agent alias target is cooling down for an "
                    "in-flight session; provider switching is disabled for "
                    "stateful Claude continuations. Redispatch a fresh agent "
                    "attempt to re-run the auto selector."
                ),
                "type": "rate_limit_error",
                "code": "aawm_anthropic_auto_agent_in_flight_provider_cooling_down",
            },
            "candidate": shaped_candidate,
        },
        headers={"Retry-After": str(int(max(1.0, cooldown_seconds)))},
    )


def _build_auto_agent_redispatch_http_exception_detail(
    *,
    alias_family: str,
    alias_model: str,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    error_class: Optional[str] = None,
    cooldown_scope: Optional[str] = None,
    error_status_code: Optional[Any] = None,
    error_type: Optional[str] = None,
    error_code: Optional[Any] = None,
    retry_after_seconds: Optional[Any] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    audit_events: Optional[list[dict[str, Any]]] = None,
    attempts: Optional[list[dict[str, Any]]] = None,
    skipped_candidates: Optional[list[dict[str, Any]]] = None,
    terminal_reset: Optional[dict[str, Any]] = None,
    code: str,
    message: str,
) -> dict[str, Any]:
    retry_after = int(
        max(
            1.0,
            float(retry_after_seconds) if retry_after_seconds is not None else cooldown_seconds,
        )
    )
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_retryable_provider_exhaustion",
    )
    detail: dict[str, Any] = {
        "error": {
            "message": message,
            "type": "rate_limit_error",
            "code": code,
        },
        "alias_family": alias_family,
        "alias_model": alias_model,
        "redispatch_model": alias_model,
        "redispatch_reason": "in_flight_retryable_provider_exhaustion",
        "redispatch_required": True,
        "selected_provider": candidate.get("provider"),
        "selected_model": candidate.get("model"),
        "selected_route_family": candidate.get("route_family"),
        "cooldown_seconds": round(float(cooldown_seconds), 3),
        "cooldown_scope": cooldown_scope,
        "retry_after_seconds": retry_after,
        "error_tokens": sorted(error_tokens),
        "candidate": shaped_candidate,
    }
    if error_class is not None:
        detail["failure_class"] = error_class
    if error_status_code is not None:
        detail["error_status_code"] = error_status_code
    if error_type is not None:
        detail["error_type"] = error_type
    if error_code is not None:
        detail["error_code"] = str(error_code)
    if failure_phase is not None:
        detail["failure_phase"] = failure_phase
    if attempted_provider_call is not None:
        detail["attempted_provider_call"] = attempted_provider_call
    if isinstance(audit_events, list):
        detail["aawm_alias_routing_audit_events"] = audit_events
    if isinstance(attempts, list):
        detail["attempts"] = attempts
    if isinstance(skipped_candidates, list):
        detail["skipped_candidates"] = skipped_candidates
    if isinstance(terminal_reset, dict):
        detail["terminal_reset"] = terminal_reset
    return detail


def _raise_codex_auto_agent_redispatch_required(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    alias_model: str,
    error_class: Optional[str] = None,
    cooldown_scope: Optional[str] = None,
    error_status_code: Optional[Any] = None,
    error_type: Optional[str] = None,
    error_code: Optional[Any] = None,
    retry_after_seconds: Optional[Any] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    audit_events: Optional[list[dict[str, Any]]] = None,
    attempts: Optional[list[dict[str, Any]]] = None,
    skipped_candidates: Optional[list[dict[str, Any]]] = None,
) -> None:
    terminal_reset = candidate.get("_codex_oauth_terminal_reset")
    detail = _build_auto_agent_redispatch_http_exception_detail(
        alias_family="codex_auto_agent",
        alias_model=alias_model,
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        error_tokens=error_tokens,
        error_class=error_class,
        cooldown_scope=cooldown_scope,
        error_status_code=error_status_code,
        error_type=error_type,
        error_code=error_code,
        retry_after_seconds=retry_after_seconds,
        failure_phase=failure_phase,
        attempted_provider_call=attempted_provider_call,
        audit_events=audit_events,
        attempts=attempts,
        skipped_candidates=skipped_candidates,
        terminal_reset=terminal_reset,
        code="aawm_codex_auto_agent_redispatch_required",
        message=(
            "Codex auto-agent alias target hit retryable provider exhaustion "
            "for an in-flight session. Do not continue this child agent. "
            f"Redispatch a fresh subagent using model {alias_model} "
            "so the auto selector can choose the next available candidate."
        ),
    )
    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(detail["retry_after_seconds"])},
    )


def _raise_anthropic_auto_agent_redispatch_required(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    alias_model: str,
    error_class: Optional[str] = None,
    cooldown_scope: Optional[str] = None,
    error_status_code: Optional[Any] = None,
    error_type: Optional[str] = None,
    error_code: Optional[Any] = None,
    retry_after_seconds: Optional[Any] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    audit_events: Optional[list[dict[str, Any]]] = None,
    attempts: Optional[list[dict[str, Any]]] = None,
    skipped_candidates: Optional[list[dict[str, Any]]] = None,
) -> None:
    terminal_reset = candidate.get("_codex_oauth_terminal_reset")
    detail = _build_auto_agent_redispatch_http_exception_detail(
        alias_family="anthropic_auto_agent",
        alias_model=alias_model,
        candidate=candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        error_tokens=error_tokens,
        error_class=error_class,
        cooldown_scope=cooldown_scope,
        error_status_code=error_status_code,
        error_type=error_type,
        error_code=error_code,
        retry_after_seconds=retry_after_seconds,
        failure_phase=failure_phase,
        attempted_provider_call=attempted_provider_call,
        audit_events=audit_events,
        attempts=attempts,
        skipped_candidates=skipped_candidates,
        terminal_reset=terminal_reset,
        code="aawm_anthropic_auto_agent_redispatch_required",
        message=(
            "Anthropic auto-agent alias target hit retryable provider "
            "exhaustion for an in-flight session. Do not continue this "
            f"child agent. Redispatch a fresh subagent using model {alias_model} "
            "so the auto selector can choose the next available candidate."
        ),
    )
    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(detail["retry_after_seconds"])},
    )



def _session_affinity_mod():
    """Lazy load session_affinity (safe under module rebinding)."""
    import sys

    mod = sys.modules.get(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity"
    )
    if mod is not None:
        return mod
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity as mod,
    )
    return mod


def _attach_session_owner_selection_fields(
    selection: dict[str, Any],
    *,
    canonical_session_identity: Optional[str],
    session_owner_guard: Any,
) -> dict[str, Any]:
    sa = _session_affinity_mod()
    provenance = getattr(session_owner_guard, "provenance", None) or sa.build_session_owner_provenance(
        session_identity=canonical_session_identity,
        decision=getattr(
            getattr(session_owner_guard, "decision", None),
            "value",
            str(getattr(session_owner_guard, "decision", "")),
        ),
        owner_record=getattr(session_owner_guard, "owner_record", None),
        owner_id=getattr(session_owner_guard, "owner_id", None),
        mismatch_reason=getattr(session_owner_guard, "mismatch_reason", None),
        cache_key=getattr(session_owner_guard, "cache_key", None),
        reservation_token=getattr(session_owner_guard, "reservation_token", None),
    )
    selection["canonical_session_identity"] = canonical_session_identity
    selection["session_owner_decision"] = provenance.get("session_owner_decision")
    selection["session_owner_id"] = provenance.get("session_owner_id")
    selection["session_owner_mismatch_reason"] = provenance.get(
        "session_owner_mismatch_reason"
    )
    selection["session_owner_provenance"] = provenance
    selection["session_owner_reservation_token"] = getattr(
        session_owner_guard, "reservation_token", None
    )
    selection["session_owner_held_reservation"] = bool(
        getattr(session_owner_guard, "held_reservation", False)
    )
    return selection


# ---------------------------------------------------------------------------
# Codex selector
# ---------------------------------------------------------------------------


async def _select_codex_auto_agent_candidate(  # noqa: PLR0915
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    assert _extract_client_product_label is not None
    assert _resolve_codex_session_key is not None
    assert _has_continuation_state is not None
    assert _get_codex_session_affinity is not None
    alias_model = _lookup_active_snapshot_canonical_alias(
        request_body.get("model"),
        request=request,
    )
    if alias_model is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "The requested model is not an active configured AAWM alias.",
                    "type": "invalid_request_error",
                    "code": "aawm_alias_unknown_or_config_unavailable",
                }
            },
        )
    client_product_label = _extract_client_product_label(request, request_body)
    session_key = _resolve_codex_session_key(
        request,
        request_body,
        alias_model=alias_model,
    )
    has_continuation_state = _has_continuation_state(request_body)
    request_mode, redispatch_ordinal = _resolve_codex_request_mode_and_ordinal(
        has_continuation_state=has_continuation_state,
        request_body=request_body,
    )

    sa = _session_affinity_mod()
    canonical_session_identity = sa.resolve_canonical_session_identity(
        request,
        request_body,
    )
    # Read-path ownership check before free selection. Reservation happens at
    # pre-egress once a concrete candidate is chosen (candidate_loop).
    session_owner_record, _cache_key, session_owner_error = await sa.get_session_owner_record(
        session_identity=canonical_session_identity,
        request=request,
        wait_for_foreign_reservation=True,
    )
    if session_owner_error is not None:
        sa.raise_session_owner_redispatch_required(
            session_identity=canonical_session_identity,
            alias_model=alias_model,
            failure_phase="session_owner_redis_unavailable",
            message=(
                "Session ownership could not be verified against durable storage. "
                "Fail closed before provider selection; redispatch with a new "
                "session identity after Redis recovers."
            ),
            guard=sa.SessionOwnerGuardResult(
                decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=canonical_session_identity,
                cache_key=_cache_key,
                mismatch_reason=session_owner_error,
                provenance=sa.build_session_owner_provenance(
                    session_identity=canonical_session_identity,
                    decision="redispatch_required",
                    mismatch_reason=session_owner_error,
                    cache_key=_cache_key,
                ),
            ),
            request=request,
        )

    affinity = None
    affinity_bypassed = False
    session_owner_guard_meta = {
        "decision": "unowned",
        "owner_record": session_owner_record,
        "owner_id": (
            session_owner_record.get("owner")
            if isinstance(session_owner_record, dict)
            else None
        ),
        "mismatch_reason": None,
        "provenance": sa.build_session_owner_provenance(
            session_identity=canonical_session_identity,
            decision="unowned",
            owner_record=session_owner_record,
            cache_key=_cache_key,
        ),
        "reservation_token": None,
        "held_reservation": False,
        "cache_key": _cache_key,
    }

    if isinstance(session_owner_record, dict) and sa._record_state(session_owner_record) == "owned":
        if request_mode == "fresh_redispatch" and has_continuation_state:
            sa.raise_session_owner_redispatch_required(
                session_identity=canonical_session_identity,
                alias_model=alias_model,
                failure_phase="session_owner_redispatch_metadata_cannot_bypass",
                message=(
                    "Redispatch metadata cannot erase or bypass ownership for an "
                    "existing session with continuation state. Start a fresh "
                    "dispatch with a new session identity."
                ),
                guard=sa.SessionOwnerGuardResult(
                    decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                    session_identity=canonical_session_identity,
                    cache_key=_cache_key,
                    owner_record=session_owner_record,
                    owner_id=session_owner_record.get("owner"),
                    mismatch_reason="session_owner: redispatch metadata cannot bypass owner",
                    provenance=sa.build_session_owner_provenance(
                        session_identity=canonical_session_identity,
                        decision="redispatch_required",
                        owner_record=session_owner_record,
                        owner_id=session_owner_record.get("owner"),
                        mismatch_reason="session_owner: redispatch metadata cannot bypass owner",
                        cache_key=_cache_key,
                    ),
                ),
                request=request,
            )
        affinity = sa.owner_record_as_affinity_hint(session_owner_record)
        session_owner_guard_meta.update(
            {
                "decision": "compatible_owner",
                "provenance": sa.build_session_owner_provenance(
                    session_identity=canonical_session_identity,
                    decision="compatible_owner",
                    owner_record=session_owner_record,
                    owner_id=session_owner_record.get("owner"),
                    cache_key=_cache_key,
                ),
            }
        )
        if affinity is None:
            # Owned but unusable attributes => fail before free selection.
            sa.raise_session_owner_redispatch_required(
                session_identity=canonical_session_identity,
                alias_model=alias_model,
                failure_phase="session_owner_owned_record_unusable",
                guard=sa.SessionOwnerGuardResult(
                    decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                    session_identity=canonical_session_identity,
                    cache_key=_cache_key,
                    owner_record=session_owner_record,
                    owner_id=session_owner_record.get("owner"),
                    mismatch_reason="session_owner: owned record missing usable attributes",
                    provenance=sa.build_session_owner_provenance(
                        session_identity=canonical_session_identity,
                        decision="redispatch_required",
                        owner_record=session_owner_record,
                        owner_id=session_owner_record.get("owner"),
                        mismatch_reason="session_owner: owned record missing usable attributes",
                        cache_key=_cache_key,
                    ),
                ),
                request=request,
            )
    elif isinstance(session_owner_record, dict) and sa._record_state(session_owner_record) == "reserved":
        sa.raise_session_owner_redispatch_required(
            session_identity=canonical_session_identity,
            alias_model=alias_model,
            failure_phase="session_owner_competing_reservation",
            guard=sa.SessionOwnerGuardResult(
                decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=canonical_session_identity,
                cache_key=_cache_key,
                owner_record=session_owner_record,
                owner_id=session_owner_record.get("owner"),
                mismatch_reason="session_owner: concurrent reservation held by another request",
                provenance=sa.build_session_owner_provenance(
                    session_identity=canonical_session_identity,
                    decision="redispatch_required",
                    owner_record=session_owner_record,
                    owner_id=session_owner_record.get("owner"),
                    mismatch_reason="session_owner: concurrent reservation held by another request",
                    cache_key=_cache_key,
                ),
            ),
            request=request,
        )
    elif request_mode == "ordinary_continuation":
        affinity = await _get_codex_session_affinity(session_key)
    elif request_mode == "fresh_redispatch":
        existing_affinity = await _get_codex_session_affinity(session_key)
        if existing_affinity is not None:
            affinity_bypassed = True

    affinity = _apply_codex_oauth_inventory_affinity_policy(affinity)
    if affinity is not None:
        affinity_candidate = _find_codex_auto_agent_affinity_candidate(
            affinity,
            alias_model=alias_model,
            client_product_label=client_product_label,
            request=request,
        )
        if affinity_candidate is None:
            pinned_candidate_shape = {
                "provider": affinity.get("provider"),
                "model": affinity.get("model"),
                "route_family": affinity.get("route_family"),
                "last_resort": bool(affinity.get("last_resort")),
            }
            for field in (
                "codex_oauth_account_label",
                "codex_oauth_account_hash",
                "codex_oauth_lane_key",
            ):
                if affinity.get(field) is not None:
                    pinned_candidate_shape[field] = affinity.get(field)
            _raise_codex_auto_agent_redispatch_required(
                candidate=pinned_candidate_shape,
                lane_key=affinity.get("codex_oauth_lane_key"),
                cooldown_seconds=0.0,
                error_tokens=set(),
                alias_model=alias_model,
                failure_phase="affinity_continuation_removed",
                attempted_provider_call=False,
            )
        assert affinity_candidate is not None
        affinity_state = await _build_codex_auto_agent_affinity_candidate_state(
            request,
            candidate_template=affinity_candidate,
            affinity=affinity,
        )
        if (
            _candidate_matches_affinity(
                affinity_state["candidate"],
                affinity,
            )
            and _is_auto_agent_candidate_state_available(affinity_state)
        ):
            return _attach_session_owner_selection_fields(
                _attach_aawm_alias_routing_state_sources(
                    {
                        **affinity_state,
                        "alias_model": alias_model,
                        "session_key": session_key,
                        "selection_reason": "session_affinity",
                        "skipped": [],
                        "in_flight_session": True,
                        "request_mode": request_mode,
                        "redispatch_ordinal": redispatch_ordinal,
                        "affinity_bypassed": affinity_bypassed,
                    },
                    affinity=affinity,
                    selected_state=affinity_state,
                ),
                canonical_session_identity=canonical_session_identity,
                session_owner_guard=type("G", (), session_owner_guard_meta)(),
            )
        if affinity_state["cooldown_seconds"] > 0:
            # Owned/cooldown owner is unavailable: fail before free selection.
            if session_owner_record is not None and sa._record_state(session_owner_record) == "owned":
                sa.raise_session_owner_redispatch_required(
                    session_identity=canonical_session_identity,
                    alias_model=alias_model,
                    candidate=affinity_state.get("candidate"),
                    failure_phase="session_owner_owner_cooldown",
                    guard=sa.SessionOwnerGuardResult(
                        decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                        session_identity=canonical_session_identity,
                        cache_key=_cache_key,
                        owner_record=session_owner_record,
                        owner_id=session_owner_record.get("owner"),
                        mismatch_reason="session_owner: owner unavailable (cooldown)",
                        provenance=sa.build_session_owner_provenance(
                            session_identity=canonical_session_identity,
                            decision="redispatch_required",
                            owner_record=session_owner_record,
                            owner_id=session_owner_record.get("owner"),
                            mismatch_reason="session_owner: owner unavailable (cooldown)",
                            cache_key=_cache_key,
                        ),
                    ),
                    request=request,
                )
            _raise_codex_auto_agent_in_flight_cooldown(
                candidate=affinity_state["candidate"],
                lane_key=affinity_state.get("lane_key"),
                cooldown_seconds=affinity_state["cooldown_seconds"],
            )
        affinity_skipped = _build_auto_agent_skipped_candidates_from_states(
            [affinity_state]
        )
        terminal_reset = _build_codex_oauth_terminal_reset_information(
            [affinity_state]
        )
        redispatch_candidate = dict(affinity_state["candidate"])
        if terminal_reset is not None:
            redispatch_candidate["_codex_oauth_terminal_reset"] = (
                terminal_reset
            )
        _raise_codex_auto_agent_redispatch_required(
            candidate=redispatch_candidate,
            lane_key=affinity_state.get("lane_key"),
            cooldown_seconds=0.0,
            error_tokens=set(),
            alias_model=alias_model,
            error_class=(
                "usage_limit_reached"
                if affinity_state.get("skip_reason") == "quota_exhausted"
                else "candidate_unavailable"
            ),
            cooldown_scope="account",
            failure_phase=affinity_state.get("failure_phase")
            or "affinity_account_unavailable",
            attempted_provider_call=False,
            skipped_candidates=affinity_skipped,
        )

    states = await _build_codex_auto_agent_candidate_states(
        request,
        alias_model=alias_model,
        client_product_label=client_product_label,
    )
    skipped = _build_auto_agent_skipped_candidates_from_states(states)

    state = _select_available_state(
        request,
        states,
        ingress="codex",
        last_resort=False,
    )
    if state is not None:
        selection_reason = (
            "codex_oauth_account_failover"
            if int(state.get("failover_ordinal") or 0) > 0
            else "first_available"
        )
        return _attach_session_owner_selection_fields(
            _attach_aawm_alias_routing_state_sources(
                {
                    **state,
                    "alias_model": alias_model,
                    "session_key": session_key,
                    "selection_reason": selection_reason,
                    "skipped": skipped,
                    "request_mode": request_mode,
                    "redispatch_ordinal": redispatch_ordinal,
                    "affinity_bypassed": affinity_bypassed,
                },
                selected_state=state,
            ),
            canonical_session_identity=canonical_session_identity,
            session_owner_guard=type("G", (), session_owner_guard_meta)(),
        )

    state = _select_available_state(
        request,
        states,
        ingress="codex",
        last_resort=True,
    )
    if state is not None:
        selection_reason = (
            "codex_oauth_account_failover"
            if int(state.get("failover_ordinal") or 0) > 0
            else "last_resort"
        )
        return _attach_session_owner_selection_fields(
            _attach_aawm_alias_routing_state_sources(
                {
                    **state,
                    "alias_model": alias_model,
                    "session_key": session_key,
                    "selection_reason": selection_reason,
                    "skipped": skipped,
                    "request_mode": request_mode,
                    "redispatch_ordinal": redispatch_ordinal,
                    "affinity_bypassed": affinity_bypassed,
                },
                selected_state=state,
            ),
            canonical_session_identity=canonical_session_identity,
            session_owner_guard=type("G", (), session_owner_guard_meta)(),
        )

    detail: dict[str, Any] = {
        "error": {
            "message": (
                "All Codex auto-agent alias candidates are currently "
                "cooled down or unavailable."
            ),
            "type": "rate_limit_error",
            "code": "aawm_codex_auto_agent_all_candidates_cooling_down",
        },
        "candidates": skipped,
    }
    terminal_reset = _build_codex_oauth_terminal_reset_information(states)
    if terminal_reset is not None:
        detail["terminal_reset"] = terminal_reset
    raise HTTPException(
        status_code=429,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Anthropic selector
# ---------------------------------------------------------------------------


async def _select_anthropic_auto_agent_candidate(  # noqa: PLR0915
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    assert _resolve_anthropic_session_key is not None
    assert _has_continuation_state is not None
    assert _get_anthropic_session_affinity is not None
    assert _extract_client_product_label is not None
    alias_model = _lookup_active_snapshot_canonical_alias(
        request_body.get("model"),
        request=request,
    )
    if alias_model is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "The requested model is not an active configured AAWM alias.",
                    "type": "invalid_request_error",
                    "code": "aawm_alias_unknown_or_config_unavailable",
                }
            },
        )
    client_product_label = _extract_client_product_label(request, request_body)
    session_key = _resolve_anthropic_session_key(
        request,
        request_body,
        alias_model=alias_model,
    )
    has_continuation_state = _has_continuation_state(request_body)
    request_mode, redispatch_ordinal = _resolve_codex_request_mode_and_ordinal(
        has_continuation_state=has_continuation_state,
        request_body=request_body,
    )

    sa = _session_affinity_mod()
    canonical_session_identity = sa.resolve_canonical_session_identity(
        request,
        request_body,
    )
    session_owner_record, _cache_key, session_owner_error = await sa.get_session_owner_record(
        session_identity=canonical_session_identity,
        request=request,
        wait_for_foreign_reservation=True,
    )
    if session_owner_error is not None:
        sa.raise_session_owner_redispatch_required(
            session_identity=canonical_session_identity,
            alias_model=alias_model,
            failure_phase="session_owner_redis_unavailable",
            message=(
                "Session ownership could not be verified against durable storage. "
                "Fail closed before provider selection; redispatch with a new "
                "session identity after Redis recovers."
            ),
            guard=sa.SessionOwnerGuardResult(
                decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=canonical_session_identity,
                cache_key=_cache_key,
                mismatch_reason=session_owner_error,
                provenance=sa.build_session_owner_provenance(
                    session_identity=canonical_session_identity,
                    decision="redispatch_required",
                    mismatch_reason=session_owner_error,
                    cache_key=_cache_key,
                ),
            ),
            request=request,
        )

    affinity = None
    affinity_bypassed = False
    session_owner_guard_meta = {
        "decision": "unowned",
        "owner_record": session_owner_record,
        "owner_id": (
            session_owner_record.get("owner")
            if isinstance(session_owner_record, dict)
            else None
        ),
        "mismatch_reason": None,
        "provenance": sa.build_session_owner_provenance(
            session_identity=canonical_session_identity,
            decision="unowned",
            owner_record=session_owner_record,
            cache_key=_cache_key,
        ),
        "reservation_token": None,
        "held_reservation": False,
        "cache_key": _cache_key,
    }

    if isinstance(session_owner_record, dict) and sa._record_state(session_owner_record) == "owned":
        if request_mode == "fresh_redispatch" and has_continuation_state:
            sa.raise_session_owner_redispatch_required(
                session_identity=canonical_session_identity,
                alias_model=alias_model,
                failure_phase="session_owner_redispatch_metadata_cannot_bypass",
                message=(
                    "Redispatch metadata cannot erase or bypass ownership for an "
                    "existing session with continuation state. Start a fresh "
                    "dispatch with a new session identity."
                ),
                guard=sa.SessionOwnerGuardResult(
                    decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                    session_identity=canonical_session_identity,
                    cache_key=_cache_key,
                    owner_record=session_owner_record,
                    owner_id=session_owner_record.get("owner"),
                    mismatch_reason="session_owner: redispatch metadata cannot bypass owner",
                    provenance=sa.build_session_owner_provenance(
                        session_identity=canonical_session_identity,
                        decision="redispatch_required",
                        owner_record=session_owner_record,
                        owner_id=session_owner_record.get("owner"),
                        mismatch_reason="session_owner: redispatch metadata cannot bypass owner",
                        cache_key=_cache_key,
                    ),
                ),
                request=request,
            )
        affinity = sa.owner_record_as_affinity_hint(session_owner_record)
        session_owner_guard_meta.update(
            {
                "decision": "compatible_owner",
                "provenance": sa.build_session_owner_provenance(
                    session_identity=canonical_session_identity,
                    decision="compatible_owner",
                    owner_record=session_owner_record,
                    owner_id=session_owner_record.get("owner"),
                    cache_key=_cache_key,
                ),
            }
        )
        if affinity is None:
            sa.raise_session_owner_redispatch_required(
                session_identity=canonical_session_identity,
                alias_model=alias_model,
                failure_phase="session_owner_owned_record_unusable",
                guard=sa.SessionOwnerGuardResult(
                    decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                    session_identity=canonical_session_identity,
                    cache_key=_cache_key,
                    owner_record=session_owner_record,
                    owner_id=session_owner_record.get("owner"),
                    mismatch_reason="session_owner: owned record missing usable attributes",
                    provenance=sa.build_session_owner_provenance(
                        session_identity=canonical_session_identity,
                        decision="redispatch_required",
                        owner_record=session_owner_record,
                        owner_id=session_owner_record.get("owner"),
                        mismatch_reason="session_owner: owned record missing usable attributes",
                        cache_key=_cache_key,
                    ),
                ),
                request=request,
            )
    elif isinstance(session_owner_record, dict) and sa._record_state(session_owner_record) == "reserved":
        sa.raise_session_owner_redispatch_required(
            session_identity=canonical_session_identity,
            alias_model=alias_model,
            failure_phase="session_owner_competing_reservation",
            guard=sa.SessionOwnerGuardResult(
                decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=canonical_session_identity,
                cache_key=_cache_key,
                owner_record=session_owner_record,
                owner_id=session_owner_record.get("owner"),
                mismatch_reason="session_owner: concurrent reservation held by another request",
                provenance=sa.build_session_owner_provenance(
                    session_identity=canonical_session_identity,
                    decision="redispatch_required",
                    owner_record=session_owner_record,
                    owner_id=session_owner_record.get("owner"),
                    mismatch_reason="session_owner: concurrent reservation held by another request",
                    cache_key=_cache_key,
                ),
            ),
            request=request,
        )
    elif request_mode == "ordinary_continuation":
        affinity = await _get_anthropic_session_affinity(session_key)
    elif request_mode == "fresh_redispatch":
        existing_affinity = await _get_anthropic_session_affinity(session_key)
        if existing_affinity is not None:
            affinity_bypassed = True

    if affinity is not None:
        affinity_candidate = _find_anthropic_auto_agent_affinity_candidate(
            affinity,
            alias_model=alias_model,
            client_product_label=client_product_label,
            request=request,
        )
        if affinity_candidate is None:
            pinned_candidate_shape = {
                "provider": affinity.get("provider"),
                "model": affinity.get("model"),
                "route_family": affinity.get("route_family"),
                "last_resort": bool(affinity.get("last_resort")),
            }
            for field in (
                "codex_oauth_account_label",
                "codex_oauth_account_hash",
                "codex_oauth_lane_key",
            ):
                if affinity.get(field) is not None:
                    pinned_candidate_shape[field] = affinity.get(field)
            _raise_anthropic_auto_agent_redispatch_required(
                candidate=pinned_candidate_shape,
                lane_key=affinity.get("codex_oauth_lane_key"),
                cooldown_seconds=0.0,
                error_tokens=set(),
                alias_model=alias_model,
                failure_phase="affinity_continuation_removed",
                attempted_provider_call=False,
            )
        assert affinity_candidate is not None
        affinity_state = (
            await _build_anthropic_auto_agent_affinity_candidate_state(
                request,
                candidate_template=affinity_candidate,
                affinity=affinity,
            )
        )
        if (
            _candidate_matches_affinity(
                affinity_state["candidate"],
                affinity,
            )
            and _is_auto_agent_candidate_state_available(affinity_state)
        ):
            return _attach_session_owner_selection_fields(
                _attach_aawm_alias_routing_state_sources(
                    {
                        **affinity_state,
                        "alias_model": alias_model,
                        "session_key": session_key,
                        "selection_reason": "session_affinity",
                        "skipped": [],
                        "in_flight_session": True,
                        "request_mode": request_mode,
                        "redispatch_ordinal": redispatch_ordinal,
                        "affinity_bypassed": affinity_bypassed,
                    },
                    affinity=affinity,
                    selected_state=affinity_state,
                ),
                canonical_session_identity=canonical_session_identity,
                session_owner_guard=type("G", (), session_owner_guard_meta)(),
            )
        if affinity_state["cooldown_seconds"] > 0:
            if session_owner_record is not None and sa._record_state(session_owner_record) == "owned":
                sa.raise_session_owner_redispatch_required(
                    session_identity=canonical_session_identity,
                    alias_model=alias_model,
                    candidate=affinity_state.get("candidate"),
                    failure_phase="session_owner_owner_cooldown",
                    guard=sa.SessionOwnerGuardResult(
                        decision=sa.SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                        session_identity=canonical_session_identity,
                        cache_key=_cache_key,
                        owner_record=session_owner_record,
                        owner_id=session_owner_record.get("owner"),
                        mismatch_reason="session_owner: owner unavailable (cooldown)",
                        provenance=sa.build_session_owner_provenance(
                            session_identity=canonical_session_identity,
                            decision="redispatch_required",
                            owner_record=session_owner_record,
                            owner_id=session_owner_record.get("owner"),
                            mismatch_reason="session_owner: owner unavailable (cooldown)",
                            cache_key=_cache_key,
                        ),
                    ),
                    request=request,
                )
            _raise_anthropic_auto_agent_in_flight_cooldown(
                candidate=affinity_state["candidate"],
                lane_key=affinity_state.get("lane_key"),
                cooldown_seconds=affinity_state["cooldown_seconds"],
            )
        affinity_skipped = _build_auto_agent_skipped_candidates_from_states(
            [affinity_state]
        )
        terminal_reset = _build_codex_oauth_terminal_reset_information(
            [affinity_state]
        )
        redispatch_candidate = dict(affinity_state["candidate"])
        if terminal_reset is not None:
            redispatch_candidate["_codex_oauth_terminal_reset"] = (
                terminal_reset
            )
        _raise_anthropic_auto_agent_redispatch_required(
            candidate=redispatch_candidate,
            lane_key=affinity_state.get("lane_key"),
            cooldown_seconds=0.0,
            error_tokens=set(),
            alias_model=alias_model,
            error_class=(
                "usage_limit_reached"
                if affinity_state.get("skip_reason") == "quota_exhausted"
                else "candidate_unavailable"
            ),
            cooldown_scope="account",
            failure_phase=affinity_state.get("failure_phase")
            or "affinity_account_unavailable",
            attempted_provider_call=False,
            skipped_candidates=affinity_skipped,
        )

    states = await _build_anthropic_auto_agent_candidate_states(
        request,
        alias_model=alias_model,
        client_product_label=client_product_label,
    )
    skipped = _build_auto_agent_skipped_candidates_from_states(states)

    state = _select_available_state(
        request,
        states,
        ingress="anthropic",
        last_resort=False,
    )
    if state is not None:
        selection_reason = (
            "codex_oauth_account_failover"
            if int(state.get("failover_ordinal") or 0) > 0
            else "first_available"
        )
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": selection_reason,
                "skipped": skipped,
                "in_flight_session": has_continuation_state,
                "request_mode": request_mode,
                "redispatch_ordinal": redispatch_ordinal,
                "affinity_bypassed": affinity_bypassed,
            },
            selected_state=state,
        )

    state = _select_available_state(
        request,
        states,
        ingress="anthropic",
        last_resort=True,
    )
    if state is not None:
        selection_reason = (
            "codex_oauth_account_failover"
            if int(state.get("failover_ordinal") or 0) > 0
            else "last_resort"
        )
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": selection_reason,
                "skipped": skipped,
                "in_flight_session": has_continuation_state,
                "request_mode": request_mode,
                "redispatch_ordinal": redispatch_ordinal,
                "affinity_bypassed": affinity_bypassed,
            },
            selected_state=state,
        )

    detail: dict[str, Any] = {
        "error": {
            "message": (
                "All Anthropic auto-agent alias candidates are currently "
                "cooled down or unavailable."
            ),
            "type": "rate_limit_error",
            "code": "aawm_anthropic_auto_agent_all_candidates_cooling_down",
        },
        "candidates": skipped,
    }
    terminal_reset = _build_codex_oauth_terminal_reset_information(states)
    if terminal_reset is not None:
        detail["terminal_reset"] = terminal_reset
    raise HTTPException(
        status_code=429,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Host-globals rebinding (Wave 5B)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType

_HOST_FUNCTION_NAMES = (
    "_auto_agent_alias_float",
    "_extract_codex_request_redispatch_ordinal",
    "_resolve_codex_request_mode_and_ordinal",
    "_codex_auto_agent_candidate_public_shape",
    "_codex_oauth_routing_candidate_fields",
    "_is_auto_agent_candidate_state_available",
    "_build_auto_agent_skipped_candidates_from_states",
    "_get_codex_auto_agent_request_local_cooldown_key",
    "_get_codex_auto_agent_request_local_cooldown_state",
    "_get_codex_auto_agent_request_local_cooldown_seconds",
    "_set_codex_auto_agent_request_local_cooldown",
    "_get_codex_auto_agent_request_local_excluded_keys",
    "_codex_oauth_candidate_slot",
    "_get_codex_oauth_request_local_blocked_slots",
    "_block_codex_oauth_request_local_candidate_slot",
    "_get_codex_oauth_request_local_failover_context",
    "_apply_codex_oauth_failover_context_to_state",
    "_plan_codex_oauth_account_failover",
    "_exclude_codex_auto_agent_request_local_candidate",
    "_exclude_codex_auto_agent_request_local_candidate_without_cooldown",
    "_apply_request_local_cooldown_from_plan",
    "_apply_codex_auto_agent_request_local_candidate_state",
    "_apply_codex_auto_agent_forced_candidate_cooldown",
    "_apply_anthropic_auto_agent_forced_candidate_cooldown",
    "_apply_codex_auto_agent_adapter_local_candidate_cooldown",
    "_apply_kimi_code_managed_account_lane_cooldown",
    "_apply_codex_auto_agent_grok_account_lane_cooldown",
    "_find_codex_auto_agent_candidate",
    "_find_codex_auto_agent_affinity_candidate",
    "_find_anthropic_auto_agent_candidate",
    "_find_anthropic_auto_agent_affinity_candidate",
    "_candidate_uses_codex_oauth",
    "_is_codex_oauth_account_candidate",
    "_candidate_matches_affinity",
    "_apply_codex_oauth_inventory_affinity_policy",
    "_resolve_codex_oauth_account_candidate_contexts",
    "_get_anthropic_auto_agent_candidate_cooldown_state",
    "_build_codex_auto_agent_candidate_state",
    "_build_anthropic_auto_agent_candidate_state",
    "_format_codex_oauth_quota_reset_at",
    "_codex_oauth_quota_window_public_shape",
    "_codex_oauth_quota_window_is_confirmed_exhausted",
    "_build_codex_oauth_terminal_reset_information",
    "_attach_normalized_quota_state",
    "_codex_oauth_public_quota_windows",
    "_build_family_quota_observation",
    "_is_codex_oauth_spark_model",
    "_codex_oauth_quota_family_for_model",
    "_codex_oauth_quota_observation_family",
    "_filter_codex_oauth_quota_windows_for_family",
    "_codex_oauth_window_remaining_pct",
    "_codex_oauth_window_is_weekly",
    "_codex_oauth_weekly_remaining_pct_from_windows",
    "_codex_oauth_state_weekly_remaining_pct",
    "_prefer_codex_oauth_weekly_balanced_state",
    "_codex_oauth_dual_family_remaining",
    "_select_codex_oauth_weighted_round_robin_state",
    "_prefer_codex_oauth_dual_quota_balanced_state",
    "_select_first_available_codex_oauth_account_state",
    "_apply_codex_oauth_account_context_to_state",
    "_build_codex_auto_agent_affinity_candidate_state",
    "_build_anthropic_auto_agent_affinity_candidate_state",
    "_build_codex_auto_agent_candidate_states",
    "_build_anthropic_auto_agent_candidate_states",
    "_get_request_selection_choices",
    "_get_request_reselection_counts",
    "_weighted_choice",
    "_select_round_robin_available_state",
    "_select_available_state",
    "_raise_codex_auto_agent_in_flight_cooldown",
    "_raise_anthropic_auto_agent_in_flight_cooldown",
    "_build_auto_agent_redispatch_http_exception_detail",
    "_raise_codex_auto_agent_redispatch_required",
    "_raise_anthropic_auto_agent_redispatch_required",
    "_select_codex_auto_agent_candidate",
    "_select_anthropic_auto_agent_candidate",
    "_session_affinity_mod",
    "_attach_session_owner_selection_fields",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = _FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = _rebound
        host_globals[_name] = _rebound
    _mod["_attach_aawm_alias_routing_state_sources"] = (
        _cooldown_state._attach_aawm_alias_routing_state_sources
    )
    host_globals["_attach_aawm_alias_routing_state_sources"] = (
        _cooldown_state._attach_aawm_alias_routing_state_sources
    )
    for _name in (
        "_is_finite_number",
        "_cohere_observation_exhausted",
        "_cohere_local_quota_exhausted",
        "_apply_cohere_local_quota_state",
    ):
        host_globals[_name] = _mod[_name]
    # Copy seam variables into host_globals so rebound functions resolve them.
    host_globals.update({
        "alias_routing_state": alias_routing_state,
        "math": math,
        "_CODEX_OAUTH_WEEKLY_BALANCE_BAND_PP": _CODEX_OAUTH_WEEKLY_BALANCE_BAND_PP,
        "_CODEX_OAUTH_QUOTA_FAMILY_OVERALL": _CODEX_OAUTH_QUOTA_FAMILY_OVERALL,
        "_CODEX_OAUTH_QUOTA_FAMILY_SPARK": _CODEX_OAUTH_QUOTA_FAMILY_SPARK,
        "_get_codex_active_cooldown_state": _get_codex_active_cooldown_state,
        "_get_anthropic_active_cooldown_state": _get_anthropic_active_cooldown_state,
        "_get_anthropic_merged_codex_openai_cooldown_state": _get_anthropic_merged_codex_openai_cooldown_state,
        "_set_codex_cooldown": _set_codex_cooldown,
        "_set_anthropic_cooldown": _set_anthropic_cooldown,
        "_get_codex_session_affinity": _get_codex_session_affinity,
        "_get_anthropic_session_affinity": _get_anthropic_session_affinity,
        "_extract_client_product_label": _extract_client_product_label,
        "_resolve_codex_session_key": _resolve_codex_session_key,
        "_resolve_anthropic_session_key": _resolve_anthropic_session_key,
        "_has_continuation_state": _has_continuation_state,
        "_find_anthropic_auto_agent_affinity_candidate": _find_anthropic_auto_agent_affinity_candidate,
        "_lookup_active_snapshot_canonical_alias": _lookup_active_snapshot_canonical_alias,
        "_resolve_aawm_alias_selection_enumeration": _resolve_aawm_alias_selection_enumeration,
        "_select_snapshot_candidates": _select_snapshot_candidates,
        "_is_grok_account_quota_candidate": _is_grok_account_quota_candidate,
        "_get_grok_account_quota_lane_cooldown_key": _get_grok_account_quota_lane_cooldown_key,
        "_is_kimi_code_candidate": _is_kimi_code_candidate,
        "_get_kimi_managed_account_cooldown_key": _get_kimi_managed_account_cooldown_key,
        "datetime": datetime,
        "timezone": timezone,
    })
