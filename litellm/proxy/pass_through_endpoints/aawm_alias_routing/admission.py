"""D1-564 provider/account lane admission.

Fail-fast admission for resolved upstream provider/account lanes:

* runs after candidate selection and before attempt-start / probe / provider I/O
* keys capacity by a non-reversible provider+account fingerprint (not alias/session)
* reuses alias-routing Redis + normalized rate-limit observations
* keeps admission state separate from alias cooldown and D1-612 session ownership
* reserves a bounded in-flight lease atomically (Lua) and releases it
* confirmed current account/usage exhaustion returns an immediate structured 429
* never queues, sleeps, background-retries, or waits through long resets

Admission remains fail-fast: it never waits for capacity or a provider reset.
Token estimates are bounded reservations, not usage accounting or billing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping, Optional

from fastapi import HTTPException

from . import durable
from .state import alias_routing_state

logger = logging.getLogger("LiteLLM Proxy")

# Durable keyspace is intentionally distinct from cooldown / session-owner keys.
_ADMISSION_STATE_FAMILY = "provider-lane"
_ADMISSION_STATE_KIND = "admission"
_ADMISSION_LEASE_STATE_KIND = "admission-lease"

_DEFAULT_MAX_IN_FLIGHT = 64
_DEFAULT_LEASE_TTL_SECONDS = 120.0
_MIN_LEASE_TTL_SECONDS = 5.0
_MAX_LEASE_TTL_SECONDS = 900.0
_DEFAULT_LARGE_CONTEXT_TOKENS = 1_000_000
_DEFAULT_LARGE_CONTEXT_MAX_IN_FLIGHT = 2
_DEFAULT_RESERVED_INTERACTIVE_CAPACITY = 0
_DEFAULT_TOKEN_WEIGHT_QUANTUM = 64_000
_DEFAULT_WARNING_INTERVAL_SECONDS = 30.0
_DEFAULT_WARNING_SUMMARY_INTERVAL_SECONDS = 300.0
_MAX_WARNING_STATES = 1024

_SUPPORTED_LIMIT_SCOPES = frozenset(
    {
        "requests",
        "input",
        "output",
        "unified",
        "concurrency",
        "unknown",
    }
)

# Confirmed long-window exhaustion (operator fail-fast constraint).
_CONFIRMED_USAGE_QUOTA_PERIODS = frozenset({"five_hour", "seven_day"})
_CONFIRMED_USAGE_WINDOW_MINUTES = frozenset({300, 10080})


class AdmissionDenyReason(str, Enum):
    CONFIRMED_EXHAUSTED = "confirmed_exhausted"
    CAPACITY_UNAVAILABLE = "capacity_unavailable"


class AdmissionAdmitReason(str, Enum):
    RESERVED = "reserved"
    REDIS_UNAVAILABLE_DEGRADED = "redis_unavailable_degraded"
    LOCAL_RESERVED = "local_reserved"


@dataclass(frozen=True)
class AdmissionLease:
    """Bounded hold on one provider/account lane (released after attempt)."""

    lane_fingerprint: str
    reservation_token: str
    counter_cache_key: str
    lease_cache_key: str
    units: int
    lease_ttl_seconds: float
    provider: str
    account_hash: Optional[str]
    held: bool = True
    durable: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    unified_tokens: int = 0
    weighted_units: int = 1
    capacity_class: str = "standard"
    large_context: bool = False
    reserved_interactive: bool = False


@dataclass(frozen=True)
class AdmissionDecision:
    """Result of one pre-I/O admission check + optional reservation."""

    allowed: bool
    reason: str
    lane_fingerprint: str
    provider: str
    account_hash: Optional[str] = None
    lease: Optional[AdmissionLease] = None
    limit_scope: Optional[str] = None
    reset_at: Optional[str] = None
    reset_hint_seconds: Optional[int] = None
    exhaustion_kind: Optional[str] = None
    quota_period: Optional[str] = None
    inflight_after: Optional[int] = None
    max_in_flight: Optional[int] = None
    detail_code: str = "aawm_provider_lane_admission_denied"
    input_tokens: int = 0
    output_tokens: int = 0
    unified_tokens: int = 0
    weighted_units: int = 1
    capacity_class: str = "standard"
    large_context: bool = False
    reserved_interactive: bool = False
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_unified_tokens: Optional[int] = None
    max_large_context_in_flight: Optional[int] = None
    reserved_interactive_capacity: Optional[int] = None


# Process-local counter used only when Redis is unconfigured (unit tests /
# single-process). Never replaces durable multi-worker reservation.
_local_inflight_by_lane: dict[str, int] = {}
_local_leases: dict[str, dict[str, Any]] = {}
_local_accounting_by_lane: dict[str, dict[str, int]] = {}
_admission_warning_states: dict[str, dict[str, Any]] = {}


def reset_admission_state_for_tests() -> None:
    """Clear process-local admission counters/leases (tests only)."""
    _local_inflight_by_lane.clear()
    _local_leases.clear()
    _local_accounting_by_lane.clear()
    _admission_warning_states.clear()


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


def _parse_positive_int(value: Optional[str], *, default: int) -> int:
    if value is None or not str(value).strip():
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _parse_nonnegative_int(value: Any, *, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    cleaned = (_clean_str(value) or "").lower()
    if cleaned in {"1", "true", "yes", "on", "enabled"}:
        return True
    if cleaned in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _env_nonnegative_int(name: str, *, default: int) -> int:
    return _parse_nonnegative_int(os.getenv(name), default=default)


def _parse_ttl_seconds(value: Optional[str], *, default: float) -> float:
    if value is None or not str(value).strip():
        return default
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(_MIN_LEASE_TTL_SECONDS, min(_MAX_LEASE_TTL_SECONDS, parsed))


def get_provider_lane_admission_max_in_flight() -> int:
    return _parse_positive_int(
        os.getenv("AAWM_PROVIDER_LANE_ADMISSION_MAX_IN_FLIGHT"),
        default=_DEFAULT_MAX_IN_FLIGHT,
    )


def get_provider_lane_admission_lease_ttl_seconds() -> float:
    return _parse_ttl_seconds(
        os.getenv("AAWM_PROVIDER_LANE_ADMISSION_LEASE_TTL_SECONDS"),
        default=_DEFAULT_LEASE_TTL_SECONDS,
    )


def get_provider_lane_admission_token_weight_quantum() -> int:
    return _parse_positive_int(
        os.getenv("AAWM_PROVIDER_LANE_ADMISSION_TOKEN_WEIGHT_QUANTUM"),
        default=_DEFAULT_TOKEN_WEIGHT_QUANTUM,
    )


def get_provider_lane_admission_large_context_tokens() -> int:
    return _parse_positive_int(
        os.getenv("AAWM_PROVIDER_LANE_ADMISSION_LARGE_CONTEXT_TOKENS"),
        default=_DEFAULT_LARGE_CONTEXT_TOKENS,
    )


def get_provider_lane_admission_large_context_max_in_flight() -> int:
    return _parse_positive_int(
        os.getenv("AAWM_PROVIDER_LANE_ADMISSION_LARGE_CONTEXT_MAX_IN_FLIGHT"),
        default=_DEFAULT_LARGE_CONTEXT_MAX_IN_FLIGHT,
    )


def get_provider_lane_admission_reserved_interactive_capacity() -> int:
    return _env_nonnegative_int(
        "AAWM_PROVIDER_LANE_ADMISSION_RESERVED_INTERACTIVE_CAPACITY",
        default=_DEFAULT_RESERVED_INTERACTIVE_CAPACITY,
    )


def get_provider_lane_admission_max_input_tokens(
    *, max_in_flight: Optional[int] = None
) -> int:
    cap = (
        max_in_flight
        if max_in_flight is not None
        else get_provider_lane_admission_max_in_flight()
    )
    return _env_nonnegative_int(
        "AAWM_PROVIDER_LANE_ADMISSION_MAX_INPUT_TOKENS",
        default=cap * get_provider_lane_admission_token_weight_quantum(),
    )


def get_provider_lane_admission_max_output_tokens(
    *, max_in_flight: Optional[int] = None
) -> int:
    cap = (
        max_in_flight
        if max_in_flight is not None
        else get_provider_lane_admission_max_in_flight()
    )
    return _env_nonnegative_int(
        "AAWM_PROVIDER_LANE_ADMISSION_MAX_OUTPUT_TOKENS",
        default=cap * get_provider_lane_admission_token_weight_quantum(),
    )


def get_provider_lane_admission_max_unified_tokens(
    *, max_in_flight: Optional[int] = None
) -> int:
    cap = (
        max_in_flight
        if max_in_flight is not None
        else get_provider_lane_admission_max_in_flight()
    )
    return _env_nonnegative_int(
        "AAWM_PROVIDER_LANE_ADMISSION_MAX_UNIFIED_TOKENS",
        default=cap * get_provider_lane_admission_token_weight_quantum(),
    )


def get_provider_lane_admission_warning_interval_seconds() -> float:
    return _parse_ttl_seconds(
        os.getenv("AAWM_PROVIDER_LANE_ADMISSION_WARNING_INTERVAL_SECONDS"),
        default=_DEFAULT_WARNING_INTERVAL_SECONDS,
    )


def get_provider_lane_admission_warning_summary_interval_seconds() -> float:
    return _parse_ttl_seconds(
        os.getenv("AAWM_PROVIDER_LANE_ADMISSION_WARNING_SUMMARY_INTERVAL_SECONDS"),
        default=_DEFAULT_WARNING_SUMMARY_INTERVAL_SECONDS,
    )


def _mapping_value(roots: list[Mapping[str, Any]], keys: tuple[str, ...]) -> Any:
    for root in roots:
        for key in keys:
            value = root.get(key)
            if value is not None:
                return value
    return None


def _request_roots(
    candidate: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    roots: list[Mapping[str, Any]] = []
    if selection is not None:
        roots.append(selection)
    roots.append(candidate)
    for root in tuple(roots):
        for key in (
            "request_body",
            "prepared_request_body",
            "request",
            "body",
            "token_estimates",
            "usage",
            "metadata",
        ):
            nested = root.get(key)
            if isinstance(nested, Mapping):
                roots.append(nested)
    return roots


def estimate_provider_lane_tokens(
    *,
    candidate: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    unified_tokens: Optional[int] = None,
    estimated_input_tokens: Optional[int] = None,
    estimated_output_tokens: Optional[int] = None,
    estimated_unified_tokens: Optional[int] = None,
) -> tuple[int, int, int]:
    """Resolve bounded request token estimates without inspecting secrets."""
    roots = _request_roots(candidate, selection)
    input_value = input_tokens
    if input_value is None:
        input_value = estimated_input_tokens
    if input_value is None:
        input_value = _mapping_value(
            roots,
            (
                "input_tokens",
                "estimated_input_tokens",
                "prompt_tokens",
                "input_token_count",
            ),
        )
    output_value = output_tokens
    if output_value is None:
        output_value = estimated_output_tokens
    if output_value is None:
        output_value = _mapping_value(
            roots,
            (
                "output_tokens",
                "estimated_output_tokens",
                "completion_tokens",
                "max_tokens",
                "max_output_tokens",
            ),
        )
    unified_value = unified_tokens
    if unified_value is None:
        unified_value = estimated_unified_tokens
    if unified_value is None:
        unified_value = _mapping_value(
            roots,
            (
                "unified_tokens",
                "estimated_unified_tokens",
                "total_tokens",
                "total_token_count",
            ),
        )
    input_count = _parse_nonnegative_int(input_value)
    output_count = _parse_nonnegative_int(output_value)
    unified_count = _parse_nonnegative_int(unified_value)
    return input_count, output_count, max(unified_count, input_count + output_count)


def resolve_reserved_interactive(
    *,
    candidate: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]] = None,
    reserved_interactive: Optional[bool] = None,
) -> bool:
    if reserved_interactive is not None:
        return bool(reserved_interactive)
    for root in (selection, candidate):
        if isinstance(root, Mapping) and _parse_bool(root.get("in_flight_session")):
            return True
    return False


def resolve_large_context(
    *,
    candidate: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]] = None,
    unified_tokens: int,
    large_context: Optional[bool] = None,
) -> bool:
    if large_context is not None:
        return bool(large_context)
    explicit = _mapping_value(
        _request_roots(candidate, selection),
        ("large_context", "is_large_context", "large_context_request"),
    )
    if explicit is not None:
        return _parse_bool(explicit)
    return unified_tokens >= get_provider_lane_admission_large_context_tokens()


def _capacity_class(*, large_context: bool, reserved_interactive: bool) -> str:
    if large_context and reserved_interactive:
        return "large_context_interactive"
    if large_context:
        return "large_context"
    if reserved_interactive:
        return "interactive"
    return "standard"


def weighted_token_units(
    unified_tokens: int,
    *,
    quantum: Optional[int] = None,
    minimum_units: int = 1,
) -> int:
    """Convert an estimated unified-token request into bounded fair units."""
    quantum_i = quantum or get_provider_lane_admission_token_weight_quantum()
    return max(
        max(1, int(minimum_units)),
        int(math.ceil(max(0, int(unified_tokens)) / max(1, quantum_i))),
    )


def _warning_context(
    *,
    candidate: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]],
    alias_model: Optional[str] = None,
    alias_family: Optional[str] = None,
    client_id: Optional[str] = None,
) -> tuple[str, str, str]:
    roots = _request_roots(candidate, selection)
    model = alias_model or _clean_str(
        _mapping_value(roots, ("alias_model", "requested_model", "model"))
    ) or "unknown"
    alias = alias_family or _clean_str(
        _mapping_value(roots, ("alias_family", "route_family", "alias"))
    ) or "unknown"
    client = client_id or _clean_str(
        _mapping_value(
            roots,
            ("client_id", "client", "user_id", "session_id", "request_id"),
        )
    ) or "unknown"
    return model, alias, client


def record_provider_lane_admission_warning(
    *,
    decision: AdmissionDecision,
    candidate: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]] = None,
    alias_model: Optional[str] = None,
    alias_family: Optional[str] = None,
    client_id: Optional[str] = None,
    now_epoch: Optional[float] = None,
) -> None:
    """Log the first lane warning, then bounded aggregate summaries."""
    if decision.allowed:
        return
    now = time.time() if now_epoch is None else float(now_epoch)
    model, alias, client = _warning_context(
        candidate=candidate,
        selection=selection,
        alias_model=alias_model,
        alias_family=alias_family,
        client_id=client_id,
    )
    scope = decision.limit_scope or "unknown"
    key = f"{decision.lane_fingerprint}:{decision.reason}:{scope}"
    state = _admission_warning_states.get(key)
    if state is None:
        if len(_admission_warning_states) >= _MAX_WARNING_STATES:
            oldest_key = min(
                _admission_warning_states,
                key=lambda item: float(
                    _admission_warning_states[item].get("last_event_at") or 0.0
                ),
            )
            _admission_warning_states.pop(oldest_key, None)
        state = {
            "first_logged": False,
            "first_at": now,
            "last_event_at": now,
            "last_summary_at": now,
            "summary_count": 0,
            "total_count": 0,
        }
        _admission_warning_states[key] = state

    state["last_event_at"] = now
    state["total_count"] = int(state.get("total_count") or 0) + 1
    fields = {
        "provider": decision.provider,
        "model": model,
        "alias": alias,
        "client": client,
        "lane_fingerprint": decision.lane_fingerprint,
        "limit_scope": scope,
        "reset_at": decision.reset_at or "unknown",
        "reason": decision.reason,
    }
    if not state["first_logged"]:
        state["first_logged"] = True
        logger.warning("provider-lane admission warning: %s", fields)
        return

    state["summary_count"] = int(state.get("summary_count") or 0) + 1
    summary_interval = get_provider_lane_admission_warning_summary_interval_seconds()
    last_summary_at = float(state.get("last_summary_at") or 0.0)
    if now - last_summary_at < summary_interval:
        return
    summary_count = int(state.get("summary_count") or 0)
    if summary_count <= 0:
        state["last_summary_at"] = now
        return
    logger.warning(
        "provider-lane admission warning summary: %s count=%d total=%d",
        fields,
        summary_count,
        int(state.get("total_count") or 0),
    )
    state["summary_count"] = 0
    state["last_summary_at"] = now


def normalize_limit_scope(value: Any) -> str:
    cleaned = (_clean_str(value) or "unknown").lower()
    if cleaned in _SUPPORTED_LIMIT_SCOPES:
        return cleaned
    # Map common observation aliases onto the supported vocabulary.
    if cleaned in {"request", "rpm", "requests_per_minute"}:
        return "requests"
    if cleaned in {"input_tokens", "input_token"}:
        return "input"
    if cleaned in {"output_tokens", "output_token"}:
        return "output"
    if cleaned in {"tokens", "token", "unified_tokens"}:
        return "unified"
    if cleaned in {"concurrent", "in_flight", "inflight"}:
        return "concurrency"
    return "unknown"


def resolve_candidate_account_hash(
    candidate: Mapping[str, Any],
    *,
    selection: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return a non-secret account identity already present on the candidate."""
    sources: list[Any] = [
        candidate.get("codex_oauth_account_hash"),
        candidate.get("provider_account_hash"),
        candidate.get("account_hash"),
        candidate.get("codex_auto_agent_selected_account_hash"),
    ]
    if selection is not None:
        sources.extend(
            [
                selection.get("codex_oauth_account_hash"),
                selection.get("provider_account_hash"),
                selection.get("account_hash"),
            ]
        )
        quota_observation = selection.get("quota_observation")
        if isinstance(quota_observation, Mapping):
            sources.append(quota_observation.get("account_hash"))
    for value in sources:
        cleaned = _clean_str(value)
        if cleaned is not None:
            return cleaned
    return None


def build_provider_account_lane_fingerprint(
    *,
    provider: str,
    account_hash: Optional[str] = None,
    lane_key: Optional[str] = None,
) -> str:
    """Non-reversible provider+account lane id (never alias/session/raw secret)."""
    provider_n = (_clean_str(provider) or "unknown").lower()
    account_n = (_clean_str(account_hash) or "").lower()
    if account_n:
        material = f"provider-account-v1:{provider_n}:{account_n}"
    else:
        # Fall back to a hashed lane_key so multi-credential lanes without an
        # explicit account hash still isolate capacity without leaking secrets.
        lane_n = _clean_str(lane_key) or ""
        if lane_n:
            lane_digest = hashlib.sha256(lane_n.encode("utf-8")).hexdigest()[:24]
            material = f"provider-lane-v1:{provider_n}:{lane_digest}"
        else:
            material = f"provider-v1:{provider_n}:unknown-account"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def build_admission_counter_cache_key(*, lane_fingerprint: str) -> str:
    return durable.build_aawm_alias_routing_durable_cache_key(
        alias_family=_ADMISSION_STATE_FAMILY,
        state_kind=_ADMISSION_STATE_KIND,
        state_key=lane_fingerprint,
    )


def build_admission_lease_cache_key(*, reservation_token: str) -> str:
    return durable.build_aawm_alias_routing_durable_cache_key(
        alias_family=_ADMISSION_STATE_FAMILY,
        state_kind=_ADMISSION_LEASE_STATE_KIND,
        state_key=reservation_token,
    )


def _observation_window_minutes(observation: Mapping[str, Any]) -> int:
    try:
        return int(observation.get("window_minutes") or 0)
    except (TypeError, ValueError):
        return 0


def _observation_remaining_pct(observation: Mapping[str, Any]) -> Optional[float]:
    try:
        if observation.get("remaining_pct") is None:
            return None
        return float(observation.get("remaining_pct"))
    except (TypeError, ValueError):
        return None


def _format_reset_at(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001
            return None
    cleaned = _clean_str(value)
    return cleaned


def is_confirmed_account_usage_exhaustion(
    observation: Optional[Mapping[str, Any]],
    *,
    now_epoch: Optional[float] = None,
) -> bool:
    """True only for confirmed current account/usage exhaustion with active reset.

    Unknown-scope pressure and ordinary short model rate limits do not qualify.
    """
    if not isinstance(observation, Mapping):
        return False
    if observation.get("exhausted") is not True:
        return False
    remaining_pct = _observation_remaining_pct(observation)
    if remaining_pct is None or remaining_pct > 0:
        return False

    status = (_clean_str(observation.get("status")) or "").lower()
    if status and status not in {"fresh", "exhausted", "quota_exhausted", "observed"}:
        # Reject explicitly stale/invalid snapshots.
        if status in {"stale", "expired", "unknown"}:
            return False

    quota_period = (_clean_str(observation.get("quota_period")) or "").lower()
    window_minutes = _observation_window_minutes(observation)
    exhaustion_kind = (_clean_str(observation.get("exhaustion_kind")) or "").lower()
    limit_scope = normalize_limit_scope(observation.get("limit_scope"))

    confirmed_long_window = (
        quota_period in _CONFIRMED_USAGE_QUOTA_PERIODS
        or window_minutes in _CONFIRMED_USAGE_WINDOW_MINUTES
    )
    confirmed_usage_kind = exhaustion_kind in {
        "usage_limit_reached",
        "account_usage",
        "quota_exhausted",
        "usage_limit",
    }
    # Account-wide scopes may confirm without five_hour/seven_day labels when
    # the observation already marks exhausted usage for this account.
    confirmed_account_scope = limit_scope in {"unified", "requests", "unknown"} and (
        confirmed_usage_kind or bool(observation.get("exhausted"))
    )

    if not (confirmed_long_window or confirmed_usage_kind):
        # Keep ordinary projected/unknown 429 pressure out of the hard gate.
        if not (
            confirmed_account_scope
            and exhaustion_kind in {"usage_limit_reached", "quota_exhausted"}
        ):
            return False

    now = time.time() if now_epoch is None else float(now_epoch)
    reset_at = observation.get("provider_resets_at") or observation.get(
        "expected_reset_at"
    )
    if reset_at is not None:
        try:
            if hasattr(reset_at, "timestamp"):
                reset_epoch = float(reset_at.timestamp())
            else:
                reset_epoch = float(reset_at)
            if reset_epoch <= now:
                return False
        except (TypeError, ValueError):
            pass
    return True


def _select_confirmed_exhaustion_observation(
    observation: Optional[Mapping[str, Any]],
    *,
    now_epoch: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    if not isinstance(observation, Mapping):
        return None
    windows = observation.get("windows")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(windows, list):
        candidates.extend(window for window in windows if isinstance(window, Mapping))
    candidates.append(observation)
    for window in candidates:
        if is_confirmed_account_usage_exhaustion(window, now_epoch=now_epoch):
            return dict(window)
    return None


def resolve_lane_quota_observation(
    *,
    candidate: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]] = None,
    account_hash: Optional[str] = None,
    state_manager: Any = None,
) -> Optional[dict[str, Any]]:
    """Prefer selection-attached observation; else query normalized state."""
    if selection is not None:
        attached = selection.get("quota_observation")
        if isinstance(attached, Mapping):
            return dict(attached)
        exhausted_windows = selection.get("quota_exhausted_windows")
        if isinstance(exhausted_windows, list) and exhausted_windows:
            first = exhausted_windows[0]
            if isinstance(first, Mapping):
                return dict(first)

    mgr = state_manager if state_manager is not None else alias_routing_state
    resolve = getattr(mgr, "resolve_normalized_quota_observation", None)
    if not callable(resolve):
        return None
    provider = _clean_str(candidate.get("provider")) or ""
    model = _clean_str(candidate.get("model")) or ""
    selected_account = account_hash or resolve_candidate_account_hash(
        candidate, selection=selection
    )
    try:
        observation = resolve(
            provider=provider,
            model=model,
            account_hash=selected_account,
        )
    except TypeError:
        observation = resolve(
            provider=provider,
            model=model,
            account_hash=selected_account,
            max_age_seconds=900.0,
        )
    if isinstance(observation, Mapping):
        return dict(observation)
    return None


def _decision_from_exhaustion(
    *,
    lane_fingerprint: str,
    provider: str,
    account_hash: Optional[str],
    observation: Mapping[str, Any],
    input_tokens: int = 0,
    output_tokens: int = 0,
    unified_tokens: int = 0,
    weighted_units: int = 1,
    capacity_class: str = "standard",
    large_context: bool = False,
    reserved_interactive: bool = False,
    max_in_flight: Optional[int] = None,
    max_input_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    max_unified_tokens: Optional[int] = None,
    max_large_context_in_flight: Optional[int] = None,
    reserved_interactive_capacity: Optional[int] = None,
) -> AdmissionDecision:
    reset_hint = observation.get("reset_hint_seconds")
    try:
        reset_hint_i = int(reset_hint) if reset_hint is not None else None
    except (TypeError, ValueError):
        reset_hint_i = None
    return AdmissionDecision(
        allowed=False,
        reason=AdmissionDenyReason.CONFIRMED_EXHAUSTED.value,
        lane_fingerprint=lane_fingerprint,
        provider=provider,
        account_hash=account_hash,
        limit_scope=normalize_limit_scope(observation.get("limit_scope")),
        reset_at=_format_reset_at(
            observation.get("provider_resets_at")
            or observation.get("expected_reset_at")
            or observation.get("reset_at")
        ),
        reset_hint_seconds=reset_hint_i,
        exhaustion_kind=_clean_str(observation.get("exhaustion_kind")),
        quota_period=_clean_str(observation.get("quota_period")),
        detail_code="aawm_provider_lane_account_exhausted",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        unified_tokens=unified_tokens,
        weighted_units=weighted_units,
        capacity_class=capacity_class,
        large_context=large_context,
        reserved_interactive=reserved_interactive,
        max_in_flight=max_in_flight,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_unified_tokens=max_unified_tokens,
        max_large_context_in_flight=max_large_context_in_flight,
        reserved_interactive_capacity=reserved_interactive_capacity,
    )


async def _get_redis_cache() -> tuple[Optional[Any], Optional[str]]:
    dual_cache = durable.get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return None, "admission: durable cache unavailable"
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        return None, "admission: durable cache missing redis_cache"
    return redis_cache, None


async def _raw_redis_client(redis_cache: Any) -> Any:
    init_fn = getattr(redis_cache, "init_async_client", None)
    if callable(init_fn):
        client = init_fn()
        if hasattr(client, "__await__"):
            client = await client
        return client
    client = getattr(redis_cache, "async_client", None) or getattr(
        redis_cache, "redis_client", None
    )
    return client


def _namespaced_key(redis_cache: Any, cache_key: str) -> str:
    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)
    if callable(fix_ns):
        return fix_ns(key=cache_key)
    return cache_key


# Inflight accounting uses a HASH of active lease_key -> canonical lease JSON
# plus per-lease TTL keys. Reservation always reclaims expired/missing or
# malformed leases first so durable multi-worker accounting stays exact.
_LUA_RESERVE = """
local inflight_hash = KEYS[1]
local lease_key = KEYS[2]
local max_in_flight = tonumber(ARGV[1])
local weighted_units = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local token = ARGV[4]
local payload = ARGV[5]
local max_input_tokens = tonumber(ARGV[6])
local max_output_tokens = tonumber(ARGV[7])
local max_unified_tokens = tonumber(ARGV[8])
local max_large_context_in_flight = tonumber(ARGV[9])

if max_in_flight == nil or weighted_units == nil or ttl == nil or
   max_input_tokens == nil or max_output_tokens == nil or
   max_unified_tokens == nil or max_large_context_in_flight == nil then
  return {-2, 0}
end
if max_in_flight < 0 or weighted_units < 1 or ttl < 1 or
   max_input_tokens < 0 or max_output_tokens < 0 or
   max_unified_tokens < 0 or max_large_context_in_flight < 0 then
  return {-2, 0}
end

if redis.call('EXISTS', lease_key) == 1 then
  return {-3, 0}
end

local payload_ok, requested = pcall(cjson.decode, payload)
if not payload_ok or type(requested) ~= 'table' then
  return {-2, 0}
end
local requested_input = tonumber(requested['input_tokens'])
local requested_output = tonumber(requested['output_tokens'])
local requested_unified = tonumber(requested['unified_tokens'])
local requested_weighted = tonumber(requested['weighted_units'])
local requested_large = requested['large_context'] == true and 1 or 0
if requested_input == nil or requested_output == nil or requested_unified == nil or
   requested_weighted == nil or requested_input < 0 or requested_output < 0 or
   requested_unified < 0 or requested_weighted ~= weighted_units or
   requested['reservation_token'] ~= token or
   type(requested['large_context']) ~= 'boolean' then
  return {-2, 0}
end

local all = redis.call('HGETALL', inflight_hash)
local current_weighted = 0
local current_input = 0
local current_output = 0
local current_unified = 0
local current_large = 0
for i = 1, #all, 2 do
  local existing_lease = all[i]
  local existing_raw = all[i + 1]
  local existing_ok, existing = pcall(cjson.decode, existing_raw)
  if redis.call('EXISTS', existing_lease) == 0 then
    redis.call('HDEL', inflight_hash, existing_lease)
  elseif not existing_ok or type(existing) ~= 'table' then
    redis.call('HDEL', inflight_hash, existing_lease)
    redis.call('DEL', existing_lease)
  else
    local existing_weighted = tonumber(existing['weighted_units'])
    local existing_input = tonumber(existing['input_tokens'])
    local existing_output = tonumber(existing['output_tokens'])
    local existing_unified = tonumber(existing['unified_tokens'])
    if existing_weighted == nil or existing_input == nil or
       existing_output == nil or existing_unified == nil or
       existing_weighted < 1 or existing_input < 0 or
       existing_output < 0 or existing_unified < 0 or
       type(existing['reservation_token']) ~= 'string' or
       type(existing['large_context']) ~= 'boolean' then
      redis.call('HDEL', inflight_hash, existing_lease)
      redis.call('DEL', existing_lease)
    else
      current_weighted = current_weighted + existing_weighted
      current_input = current_input + existing_input
      current_output = current_output + existing_output
      current_unified = current_unified + existing_unified
      if existing['large_context'] == true then
        current_large = current_large + 1
      end
    end
  end
end

local function deny(code)
  if current_weighted <= 0 then
    redis.call('DEL', inflight_hash)
  else
    redis.call('EXPIRE', inflight_hash, ttl)
  end
  return {code, current_weighted}
end

if (current_weighted + weighted_units) > max_in_flight then
  return deny(0)
end
if (current_input + requested_input) > max_input_tokens then
  return deny(-4)
end
if (current_output + requested_output) > max_output_tokens then
  return deny(-5)
end
if (current_unified + requested_unified) > max_unified_tokens then
  return deny(-6)
end
if requested_large == 1 and
   (current_large + requested_large) > max_large_context_in_flight then
  return deny(-7)
end

redis.call('HSET', inflight_hash, lease_key, payload)
redis.call('SET', lease_key, payload, 'EX', ttl)
redis.call('EXPIRE', inflight_hash, ttl)
return {1, current_weighted + weighted_units}
"""


_LUA_RELEASE = """
local inflight_hash = KEYS[1]
local lease_key = KEYS[2]
local token = ARGV[1]

local raw = redis.call('GET', lease_key)
local owned = false
if raw then
  local ok, current = pcall(cjson.decode, raw)
  if ok and type(current) == 'table' then
    owned = current['reservation_token'] == token
  else
    redis.call('DEL', lease_key)
  end
end

if owned then
  redis.call('HDEL', inflight_hash, lease_key)
  redis.call('DEL', lease_key)
end

-- Always reclaim expired/missing hash entries so release of a vanished lease
-- still drops its units when the hash retained them.
local all = redis.call('HGETALL', inflight_hash)
local remaining = 0
for i = 1, #all, 2 do
  local existing_lease = all[i]
  local existing_raw = all[i + 1]
  local existing_ok, existing = pcall(cjson.decode, existing_raw)
  if redis.call('EXISTS', existing_lease) == 0 then
    redis.call('HDEL', inflight_hash, existing_lease)
  elseif not existing_ok or type(existing) ~= 'table' then
    redis.call('HDEL', inflight_hash, existing_lease)
    redis.call('DEL', existing_lease)
  else
    local existing_weighted = tonumber(existing['weighted_units'])
    local existing_input = tonumber(existing['input_tokens'])
    local existing_output = tonumber(existing['output_tokens'])
    local existing_unified = tonumber(existing['unified_tokens'])
    if existing_weighted == nil or existing_input == nil or
       existing_output == nil or existing_unified == nil or
       existing_weighted < 1 or existing_input < 0 or
       existing_output < 0 or existing_unified < 0 or
       type(existing['reservation_token']) ~= 'string' or
       type(existing['large_context']) ~= 'boolean' then
      redis.call('HDEL', inflight_hash, existing_lease)
      redis.call('DEL', existing_lease)
    else
      remaining = remaining + existing_weighted
    end
  end
end

if remaining <= 0 then
  redis.call('DEL', inflight_hash)
  remaining = 0
end

-- Lease already gone: treat as reclaimed if hash entry was dropped.
return {1, remaining}
"""


def _reclaim_expired_local_leases(*, now_epoch: Optional[float] = None) -> None:
    """Drop expired local leases and subtract their exact accounting."""
    now = time.time() if now_epoch is None else float(now_epoch)
    expired_tokens = [
        token
        for token, record in list(_local_leases.items())
        if float(record.get("expires_at") or 0.0) <= now
    ]
    for token in expired_tokens:
        record = _local_leases.pop(token, None)
        if record is None:
            continue
        lane = str(record.get("lane_fingerprint") or "")
        if not lane:
            continue
        units = int(record.get("units") or 1)
        current = int(_local_inflight_by_lane.get(lane, 0))
        remaining = max(0, current - units)
        if remaining == 0:
            _local_inflight_by_lane.pop(lane, None)
        else:
            _local_inflight_by_lane[lane] = remaining
        accounting = _local_accounting_by_lane.get(lane)
        if accounting is None:
            continue
        for field in ("input_tokens", "output_tokens", "unified_tokens"):
            accounting[field] = max(
                0,
                int(accounting.get(field) or 0) - int(record.get(field) or 0),
            )
        accounting["large_context_count"] = max(
            0,
            int(accounting.get("large_context_count") or 0)
            - int(record.get("large_context_count") or 0),
        )
        if not any(accounting.values()):
            _local_accounting_by_lane.pop(lane, None)


def _local_reserve(
    *,
    lane_fingerprint: str,
    units: int,
    max_in_flight: int,
    reservation_token: str,
    lease_ttl_seconds: float,
    provider: str,
    account_hash: Optional[str],
    input_tokens: int,
    output_tokens: int,
    unified_tokens: int,
    weighted_units: int,
    capacity_class: str,
    large_context: bool,
    reserved_interactive: bool,
    max_input_tokens: int,
    max_output_tokens: int,
    max_unified_tokens: int,
    max_large_context_in_flight: int,
    reserved_interactive_capacity: int,
    now_epoch: Optional[float] = None,
) -> AdmissionDecision:
    _reclaim_expired_local_leases(now_epoch=now_epoch)
    accounting = _local_accounting_by_lane.get(lane_fingerprint) or {}
    accounting_checks = (
        ("input", "input_tokens", input_tokens, max_input_tokens),
        ("output", "output_tokens", output_tokens, max_output_tokens),
        ("unified", "unified_tokens", unified_tokens, max_unified_tokens),
    )
    for limit_scope, field, requested, maximum in accounting_checks:
        if int(accounting.get(field) or 0) + requested > maximum:
            return AdmissionDecision(
                allowed=False,
                reason=AdmissionDenyReason.CAPACITY_UNAVAILABLE.value,
                lane_fingerprint=lane_fingerprint,
                provider=provider,
                account_hash=account_hash,
                max_in_flight=max_in_flight,
                limit_scope=limit_scope,
                detail_code="aawm_provider_lane_capacity_unavailable",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                unified_tokens=unified_tokens,
                weighted_units=weighted_units,
                capacity_class=capacity_class,
                large_context=large_context,
                reserved_interactive=reserved_interactive,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_unified_tokens=max_unified_tokens,
                max_large_context_in_flight=max_large_context_in_flight,
                reserved_interactive_capacity=reserved_interactive_capacity,
            )

    current = int(_local_inflight_by_lane.get(lane_fingerprint, 0))
    if current + units > max_in_flight:
        return AdmissionDecision(
            allowed=False,
            reason=AdmissionDenyReason.CAPACITY_UNAVAILABLE.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            inflight_after=current,
            max_in_flight=max_in_flight,
            limit_scope="concurrency",
            detail_code=(
                "aawm_provider_lane_large_context_capacity_unavailable"
                if large_context
                else "aawm_provider_lane_capacity_unavailable"
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_unified_tokens=max_unified_tokens,
            max_large_context_in_flight=max_large_context_in_flight,
            reserved_interactive_capacity=reserved_interactive_capacity,
        )
    current_large_context = int(accounting.get("large_context_count") or 0)
    if large_context and current_large_context + 1 > max_large_context_in_flight:
        return AdmissionDecision(
            allowed=False,
            reason=AdmissionDenyReason.CAPACITY_UNAVAILABLE.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            inflight_after=current,
            max_in_flight=max_in_flight,
            limit_scope="concurrency",
            detail_code="aawm_provider_lane_large_context_capacity_unavailable",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_unified_tokens=max_unified_tokens,
            max_large_context_in_flight=max_large_context_in_flight,
            reserved_interactive_capacity=reserved_interactive_capacity,
        )

    new_count = current + units
    _local_inflight_by_lane[lane_fingerprint] = new_count
    _local_accounting_by_lane[lane_fingerprint] = {
        "input_tokens": int(accounting.get("input_tokens") or 0) + input_tokens,
        "output_tokens": int(accounting.get("output_tokens") or 0) + output_tokens,
        "unified_tokens": int(accounting.get("unified_tokens") or 0)
        + unified_tokens,
        "large_context_count": current_large_context + int(large_context),
    }
    now = time.time() if now_epoch is None else float(now_epoch)
    _local_leases[reservation_token] = {
        "lane_fingerprint": lane_fingerprint,
        "units": units,
        "reservation_token": reservation_token,
        "expires_at": now + lease_ttl_seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "unified_tokens": unified_tokens,
        "large_context_count": int(large_context),
    }
    lease = AdmissionLease(
        lane_fingerprint=lane_fingerprint,
        reservation_token=reservation_token,
        counter_cache_key=build_admission_counter_cache_key(
            lane_fingerprint=lane_fingerprint
        ),
        lease_cache_key=build_admission_lease_cache_key(
            reservation_token=reservation_token
        ),
        units=units,
        lease_ttl_seconds=lease_ttl_seconds,
        provider=provider,
        account_hash=account_hash,
        held=True,
        durable=False,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        unified_tokens=unified_tokens,
        weighted_units=weighted_units,
        capacity_class=capacity_class,
        large_context=large_context,
        reserved_interactive=reserved_interactive,
    )
    return AdmissionDecision(
        allowed=True,
        reason=AdmissionAdmitReason.LOCAL_RESERVED.value,
        lane_fingerprint=lane_fingerprint,
        provider=provider,
        account_hash=account_hash,
        lease=lease,
        inflight_after=new_count,
        max_in_flight=max_in_flight,
        limit_scope="concurrency",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        unified_tokens=unified_tokens,
        weighted_units=weighted_units,
        capacity_class=capacity_class,
        large_context=large_context,
        reserved_interactive=reserved_interactive,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_unified_tokens=max_unified_tokens,
        max_large_context_in_flight=max_large_context_in_flight,
        reserved_interactive_capacity=reserved_interactive_capacity,
    )


def _local_release(lease: AdmissionLease) -> bool:
    record = _local_leases.pop(lease.reservation_token, None)
    if record is None:
        return False
    lane = str(record.get("lane_fingerprint") or lease.lane_fingerprint)
    units = int(record.get("units") or lease.units or 1)
    current = int(_local_inflight_by_lane.get(lane, 0))
    remaining = max(0, current - units)
    if remaining == 0:
        _local_inflight_by_lane.pop(lane, None)
    else:
        _local_inflight_by_lane[lane] = remaining
    accounting = _local_accounting_by_lane.get(lane)
    if accounting is not None:
        for field in ("input_tokens", "output_tokens", "unified_tokens"):
            accounting[field] = max(
                0,
                int(accounting.get(field) or 0) - int(record.get(field) or 0),
            )
        accounting["large_context_count"] = max(
            0,
            int(accounting.get("large_context_count") or 0)
            - int(record.get("large_context_count") or 0),
        )
        if not any(accounting.values()):
            _local_accounting_by_lane.pop(lane, None)
    return True

def _parse_lua_pair(result: Any) -> tuple[int, int]:
    code = 0
    inflight_after = 0
    if isinstance(result, (list, tuple)) and result:
        try:
            code = int(result[0])
        except (TypeError, ValueError):
            code = 0
        if len(result) > 1:
            try:
                inflight_after = int(result[1])
            except (TypeError, ValueError):
                inflight_after = 0
        return code, inflight_after
    try:
        return int(result or 0), 0
    except (TypeError, ValueError):
        return 0, 0


def _decision_from_reserve_code(
    *,
    code: int,
    inflight_after: int,
    lane_fingerprint: str,
    provider: str,
    account_hash: Optional[str],
    reservation_token: str,
    counter_key: str,
    lease_key: str,
    reserve_units: int,
    ttl: float,
    cap: int,
    input_tokens: int,
    output_tokens: int,
    unified_tokens: int,
    weighted_units: int,
    capacity_class: str,
    large_context: bool,
    reserved_interactive: bool,
    max_input_tokens: int,
    max_output_tokens: int,
    max_unified_tokens: int,
    max_large_context_in_flight: int,
    reserved_interactive_capacity: int,
) -> AdmissionDecision:
    if code == 1:
        lease = AdmissionLease(
            lane_fingerprint=lane_fingerprint,
            reservation_token=reservation_token,
            counter_cache_key=counter_key,
            lease_cache_key=lease_key,
            units=reserve_units,
            lease_ttl_seconds=ttl,
            provider=provider,
            account_hash=account_hash,
            held=True,
            durable=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
        )
        return AdmissionDecision(
            allowed=True,
            reason=AdmissionAdmitReason.RESERVED.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            lease=lease,
            inflight_after=inflight_after,
            max_in_flight=cap,
            limit_scope="concurrency",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_unified_tokens=max_unified_tokens,
            max_large_context_in_flight=max_large_context_in_flight,
            reserved_interactive_capacity=reserved_interactive_capacity,
        )
    denial_scope_by_code = {
        0: "concurrency",
        -4: "input",
        -5: "output",
        -6: "unified",
        -7: "concurrency",
    }
    if code in denial_scope_by_code:
        return AdmissionDecision(
            allowed=False,
            reason=AdmissionDenyReason.CAPACITY_UNAVAILABLE.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            inflight_after=inflight_after,
            max_in_flight=cap,
            limit_scope=denial_scope_by_code[code],
            detail_code=(
                "aawm_provider_lane_large_context_capacity_unavailable"
                if code == -7
                else "aawm_provider_lane_capacity_unavailable"
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_unified_tokens=max_unified_tokens,
            max_large_context_in_flight=max_large_context_in_flight,
            reserved_interactive_capacity=reserved_interactive_capacity,
        )
    return AdmissionDecision(
        allowed=True,
        reason=AdmissionAdmitReason.REDIS_UNAVAILABLE_DEGRADED.value,
        lane_fingerprint=lane_fingerprint,
        provider=provider,
        account_hash=account_hash,
        max_in_flight=cap,
        limit_scope="concurrency",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        unified_tokens=unified_tokens,
        weighted_units=weighted_units,
        capacity_class=capacity_class,
        large_context=large_context,
        reserved_interactive=reserved_interactive,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_unified_tokens=max_unified_tokens,
        max_large_context_in_flight=max_large_context_in_flight,
        reserved_interactive_capacity=reserved_interactive_capacity,
    )

async def reserve_provider_lane_admission(  # noqa: PLR0915
    *,
    candidate: Mapping[str, Any],
    selection: Optional[Mapping[str, Any]] = None,
    units: int = 1,
    max_in_flight: Optional[int] = None,
    lease_ttl_seconds: Optional[float] = None,
    state_manager: Any = None,
    now_epoch: Optional[float] = None,
    allow_local_fallback: bool = True,
) -> AdmissionDecision:
    """Fail-fast reserve one lane after selection and before provider I/O.

    Never sleeps, queues, or waits for reset. Confirmed exhaustion denies
    immediately. Redis unavailability degrades open (no durable lease) so
    admission does not become a hard dependency outage; exhaustion still
    gates from normalized observations.
    """
    provider = _clean_str(candidate.get("provider")) or "unknown"
    account_hash = resolve_candidate_account_hash(candidate, selection=selection)
    lane_key = None
    if selection is not None:
        lane_key = _clean_str(selection.get("lane_key"))
    if lane_key is None:
        lane_key = _clean_str(candidate.get("codex_oauth_lane_key")) or _clean_str(
            candidate.get("lane_key")
        )
    lane_fingerprint = build_provider_account_lane_fingerprint(
        provider=provider,
        account_hash=account_hash,
        lane_key=lane_key,
    )

    input_tokens, output_tokens, unified_tokens = estimate_provider_lane_tokens(
        candidate=candidate,
        selection=selection,
    )
    weighted_units = weighted_token_units(unified_tokens, minimum_units=units)
    large_context = resolve_large_context(
        candidate=candidate,
        selection=selection,
        unified_tokens=unified_tokens,
    )
    reserved_interactive = resolve_reserved_interactive(
        candidate=candidate,
        selection=selection,
    )
    capacity_class = _capacity_class(
        large_context=large_context,
        reserved_interactive=reserved_interactive,
    )
    original_cap = max(
        0,
        int(max_in_flight)
        if max_in_flight is not None
        else get_provider_lane_admission_max_in_flight(),
    )
    max_input_tokens = get_provider_lane_admission_max_input_tokens(
        max_in_flight=original_cap
    )
    max_output_tokens = get_provider_lane_admission_max_output_tokens(
        max_in_flight=original_cap
    )
    max_unified_tokens = get_provider_lane_admission_max_unified_tokens(
        max_in_flight=original_cap
    )
    max_large_context_in_flight = (
        get_provider_lane_admission_large_context_max_in_flight()
    )
    reserved_interactive_capacity = (
        get_provider_lane_admission_reserved_interactive_capacity()
    )
    cap = (
        original_cap
        if reserved_interactive
        else max(0, original_cap - reserved_interactive_capacity)
    )

    observation = resolve_lane_quota_observation(
        candidate=candidate,
        selection=selection,
        account_hash=account_hash,
        state_manager=state_manager,
    )
    exhausted = _select_confirmed_exhaustion_observation(
        observation, now_epoch=now_epoch
    )
    if exhausted is not None:
        decision = _decision_from_exhaustion(
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            observation=exhausted,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
            max_in_flight=cap,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_unified_tokens=max_unified_tokens,
            max_large_context_in_flight=max_large_context_in_flight,
            reserved_interactive_capacity=reserved_interactive_capacity,
        )
        record_provider_lane_admission_warning(
            decision=decision,
            candidate=candidate,
            selection=selection,
            now_epoch=now_epoch,
        )
        return decision

    reserve_units = weighted_units
    ttl = (
        float(lease_ttl_seconds)
        if lease_ttl_seconds is not None
        else get_provider_lane_admission_lease_ttl_seconds()
    )
    ttl = max(_MIN_LEASE_TTL_SECONDS, min(_MAX_LEASE_TTL_SECONDS, ttl))
    reservation_token = uuid.uuid4().hex

    redis_cache, redis_error = await _get_redis_cache()
    if redis_cache is None:
        if allow_local_fallback:
            # Unconfigured Redis (tests / memory mode): local accounting only.
            decision = _local_reserve(
                lane_fingerprint=lane_fingerprint,
                units=reserve_units,
                max_in_flight=cap,
                reservation_token=reservation_token,
                lease_ttl_seconds=ttl,
                provider=provider,
                account_hash=account_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                unified_tokens=unified_tokens,
                weighted_units=weighted_units,
                capacity_class=capacity_class,
                large_context=large_context,
                reserved_interactive=reserved_interactive,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                max_unified_tokens=max_unified_tokens,
                max_large_context_in_flight=max_large_context_in_flight,
                reserved_interactive_capacity=reserved_interactive_capacity,
                now_epoch=now_epoch,
            )
            record_provider_lane_admission_warning(
                decision=decision,
                candidate=candidate,
                selection=selection,
                now_epoch=now_epoch,
            )
            return decision
        return AdmissionDecision(
            allowed=True,
            reason=AdmissionAdmitReason.REDIS_UNAVAILABLE_DEGRADED.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            max_in_flight=cap,
            limit_scope="concurrency",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_unified_tokens=max_unified_tokens,
            max_large_context_in_flight=max_large_context_in_flight,
            reserved_interactive_capacity=reserved_interactive_capacity,
        )

    client = await _raw_redis_client(redis_cache)
    if client is None or not callable(getattr(client, "eval", None)):
        # Configured-but-unusable Redis: degrade open (no queue/wait).
        logger.warning(
            "provider-lane admission redis unavailable; degrading open (%s)",
            redis_error or "client/eval missing",
        )
        return AdmissionDecision(
            allowed=True,
            reason=AdmissionAdmitReason.REDIS_UNAVAILABLE_DEGRADED.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            max_in_flight=cap,
            limit_scope="concurrency",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_unified_tokens=max_unified_tokens,
            max_large_context_in_flight=max_large_context_in_flight,
            reserved_interactive_capacity=reserved_interactive_capacity,
        )

    counter_key = build_admission_counter_cache_key(lane_fingerprint=lane_fingerprint)
    lease_key = build_admission_lease_cache_key(reservation_token=reservation_token)
    ns_counter = _namespaced_key(redis_cache, counter_key)
    ns_lease = _namespaced_key(redis_cache, lease_key)
    payload = json.dumps(
        {
            "reservation_token": reservation_token,
            "lane_fingerprint": lane_fingerprint,
            "units": reserve_units,
            "provider": provider,
            "account_hash": account_hash,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "unified_tokens": unified_tokens,
            "weighted_units": weighted_units,
            "capacity_class": capacity_class,
            "large_context": large_context,
            "reserved_interactive": reserved_interactive,
            "original_max_in_flight": original_cap,
            "max_in_flight": cap,
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
            "max_unified_tokens": max_unified_tokens,
            "max_large_context_in_flight": max_large_context_in_flight,
            "reserved_interactive_capacity": reserved_interactive_capacity,
            "reserved_at_epoch": time.time() if now_epoch is None else float(now_epoch),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        result = await client.eval(
            _LUA_RESERVE,
            2,
            ns_counter,
            ns_lease,
            str(cap),
            str(reserve_units),
            str(int(math.ceil(ttl))),
            reservation_token,
            payload,
            str(max_input_tokens),
            str(max_output_tokens),
            str(max_unified_tokens),
            str(max_large_context_in_flight),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "provider-lane admission reserve failed; degrading open: %s",
            type(exc).__name__,
        )
        return AdmissionDecision(
            allowed=True,
            reason=AdmissionAdmitReason.REDIS_UNAVAILABLE_DEGRADED.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            max_in_flight=cap,
            limit_scope="concurrency",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            unified_tokens=unified_tokens,
            weighted_units=weighted_units,
            capacity_class=capacity_class,
            large_context=large_context,
            reserved_interactive=reserved_interactive,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_unified_tokens=max_unified_tokens,
            max_large_context_in_flight=max_large_context_in_flight,
            reserved_interactive_capacity=reserved_interactive_capacity,
        )

    code, inflight_after = _parse_lua_pair(result)
    decision = _decision_from_reserve_code(
        code=code,
        inflight_after=inflight_after,
        lane_fingerprint=lane_fingerprint,
        provider=provider,
        account_hash=account_hash,
        reservation_token=reservation_token,
        counter_key=counter_key,
        lease_key=lease_key,
        reserve_units=reserve_units,
        ttl=ttl,
        cap=cap,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        unified_tokens=unified_tokens,
        weighted_units=weighted_units,
        capacity_class=capacity_class,
        large_context=large_context,
        reserved_interactive=reserved_interactive,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_unified_tokens=max_unified_tokens,
        max_large_context_in_flight=max_large_context_in_flight,
        reserved_interactive_capacity=reserved_interactive_capacity,
    )
    record_provider_lane_admission_warning(
        decision=decision,
        candidate=candidate,
        selection=selection,
        now_epoch=now_epoch,
    )
    return decision


async def release_provider_lane_admission(
    lease: Optional[AdmissionLease],
) -> bool:
    """Release a held admission lease. Idempotent; never raises to callers."""
    if lease is None or not lease.held:
        return False
    if not lease.durable:
        try:
            return _local_release(lease)
        except Exception:  # noqa: BLE001
            return False

    try:
        redis_cache, _error = await _get_redis_cache()
        if redis_cache is None:
            return False
        client = await _raw_redis_client(redis_cache)
        if client is None or not callable(getattr(client, "eval", None)):
            return False
        ns_counter = _namespaced_key(redis_cache, lease.counter_cache_key)
        ns_lease = _namespaced_key(redis_cache, lease.lease_cache_key)
        result = await client.eval(
            _LUA_RELEASE,
            2,
            ns_counter,
            ns_lease,
            lease.reservation_token,
        )
        code = 0
        if isinstance(result, (list, tuple)) and result:
            code = int(result[0] or 0)
        else:
            code = int(result or 0)
        return code == 1
    except Exception:  # noqa: BLE001
        return False


def build_admission_rejection_detail(
    decision: AdmissionDecision,
    *,
    candidate: Mapping[str, Any],
    alias_model: Optional[str] = None,
    alias_family: Optional[str] = None,
    lane_key: Optional[str] = None,
) -> dict[str, Any]:
    """Structured 429 body with provider, safe lane id, scope, and exact reset."""
    if decision.reason == AdmissionDenyReason.CONFIRMED_EXHAUSTED.value:
        message = (
            "Provider account/usage capacity is currently exhausted for the "
            "selected upstream lane; retry after the provider reset."
        )
    else:
        message = (
            "Provider lane admission denied because the selected upstream "
            "account lane has no immediately available capacity."
        )
    error: dict[str, Any] = {
        "message": message,
        "type": "rate_limit_error",
        "code": decision.detail_code,
        "provider": decision.provider or candidate.get("provider"),
        "lane_fingerprint": decision.lane_fingerprint,
        "limit_scope": decision.limit_scope or "unknown",
    }
    if decision.account_hash:
        error["account_hash"] = decision.account_hash
    if decision.reset_at:
        error["reset_at"] = decision.reset_at
    if decision.reset_hint_seconds is not None:
        error["reset_hint_seconds"] = int(decision.reset_hint_seconds)
    if decision.exhaustion_kind:
        error["exhaustion_kind"] = decision.exhaustion_kind
    if decision.quota_period:
        error["quota_period"] = decision.quota_period
    if decision.max_in_flight is not None:
        error["max_in_flight"] = int(decision.max_in_flight)
    if decision.inflight_after is not None:
        error["inflight"] = int(decision.inflight_after)

    detail: dict[str, Any] = {
        "error": error,
        "admission": {
            "allowed": False,
            "reason": decision.reason,
            "lane_fingerprint": decision.lane_fingerprint,
            "provider": decision.provider or candidate.get("provider"),
            "account_hash": decision.account_hash,
            "limit_scope": decision.limit_scope or "unknown",
            "reset_at": decision.reset_at,
            "reset_hint_seconds": decision.reset_hint_seconds,
            "attempted_provider_call": False,
        },
        "candidate": {
            "provider": candidate.get("provider"),
            "model": candidate.get("model"),
            "route_family": candidate.get("route_family"),
            "lane_key": lane_key,
            "last_resort": bool(candidate.get("last_resort")),
        },
    }
    if alias_model is not None:
        detail["alias_model"] = alias_model
    if alias_family is not None:
        detail["alias_family"] = alias_family
    # Drop Nones from nested admission block for stable client payloads.
    detail["admission"] = {
        key: value
        for key, value in detail["admission"].items()
        if value is not None
    }
    detail["candidate"] = {
        key: value for key, value in detail["candidate"].items() if value is not None
    }
    return detail


def raise_provider_lane_admission_rejected(
    decision: AdmissionDecision,
    *,
    candidate: Mapping[str, Any],
    alias_model: Optional[str] = None,
    alias_family: Optional[str] = None,
    lane_key: Optional[str] = None,
) -> None:
    """Raise immediate structured HTTP 429. Never sleeps or schedules retry."""
    detail = build_admission_rejection_detail(
        decision,
        candidate=candidate,
        alias_model=alias_model,
        alias_family=alias_family,
        lane_key=lane_key,
    )
    headers: dict[str, str] = {}
    retry_after: Optional[int] = None
    if decision.reset_hint_seconds is not None and decision.reset_hint_seconds > 0:
        retry_after = max(1, int(decision.reset_hint_seconds))
    if retry_after is not None:
        # Surface exact known wait when provided by the observation. This is
        # advisory only: LiteLLM does not sleep or background-retry on it.
        headers["Retry-After"] = str(retry_after)
        detail["error"]["retry_after_seconds"] = retry_after
    raise HTTPException(status_code=429, detail=detail, headers=headers or None)


def attach_admission_metadata(
    target: MutableMapping[str, Any],
    decision: AdmissionDecision,
) -> None:
    """Stamp safe admission fields onto selection/attempt records."""
    target["admission_decision"] = decision.reason
    target["admission_lane_fingerprint"] = decision.lane_fingerprint
    target["admission_allowed"] = bool(decision.allowed)
    if decision.account_hash:
        target["admission_account_hash"] = decision.account_hash
    if decision.limit_scope:
        target["admission_limit_scope"] = decision.limit_scope
    if decision.reset_at:
        target["admission_reset_at"] = decision.reset_at
    if decision.lease is not None:
        target["admission_reservation_token"] = decision.lease.reservation_token
        target["admission_lease_held"] = True
    else:
        target["admission_lease_held"] = False


def admission_deny_error_class(decision: AdmissionDecision) -> str:
    """Map admission denial onto existing retry/failover vocabulary."""
    if decision.reason == AdmissionDenyReason.CONFIRMED_EXHAUSTED.value:
        return "usage_limit_reached"
    return "capacity_exhausted"


async def admit_selected_candidate(
    *,
    candidate: Mapping[str, Any],
    selection: MutableMapping[str, Any],
    attempt_record: Optional[MutableMapping[str, Any]] = None,
    units: int = 1,
    max_in_flight: Optional[int] = None,
    lease_ttl_seconds: Optional[float] = None,
    state_manager: Any = None,
) -> AdmissionDecision:
    """Component seam used by the candidate loop (pre-I/O)."""
    decision = await reserve_provider_lane_admission(
        candidate=candidate,
        selection=selection,
        units=units,
        max_in_flight=max_in_flight,
        lease_ttl_seconds=lease_ttl_seconds,
        state_manager=state_manager,
    )
    attach_admission_metadata(selection, decision)
    if attempt_record is not None:
        attach_admission_metadata(attempt_record, decision)
        if not decision.allowed:
            attempt_record["status"] = "admission_denied"
            attempt_record["failure_phase"] = "provider_lane_admission"
            attempt_record["attempted_provider_call"] = False
            attempt_record["error_class"] = admission_deny_error_class(decision)
    return decision


__all__ = [
    "AdmissionAdmitReason",
    "AdmissionDecision",
    "AdmissionDenyReason",
    "AdmissionLease",
    "admit_selected_candidate",
    "attach_admission_metadata",
    "admission_deny_error_class",
    "build_admission_counter_cache_key",
    "build_admission_lease_cache_key",
    "build_admission_rejection_detail",
    "build_provider_account_lane_fingerprint",
    "get_provider_lane_admission_lease_ttl_seconds",
    "get_provider_lane_admission_max_in_flight",
    "is_confirmed_account_usage_exhaustion",
    "normalize_limit_scope",
    "raise_provider_lane_admission_rejected",
    "release_provider_lane_admission",
    "reserve_provider_lane_admission",
    "reset_admission_state_for_tests",
    "resolve_candidate_account_hash",
    "resolve_lane_quota_observation",
]
