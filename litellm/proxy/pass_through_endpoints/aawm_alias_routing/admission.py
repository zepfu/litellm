"""D1-564 provider/account lane admission (first durable body).

Fail-fast admission for resolved upstream provider/account lanes:

* runs after candidate selection and before attempt-start / probe / provider I/O
* keys capacity by a non-reversible provider+account fingerprint (not alias/session)
* reuses alias-routing Redis + normalized rate-limit observations
* keeps admission state separate from alias cooldown and D1-612 session ownership
* reserves a bounded in-flight lease atomically (Lua) and releases it
* confirmed current account/usage exhaustion returns an immediate structured 429
* never queues, sleeps, background-retries, or waits through long resets

Later D1-564 bodies own adaptive fairness, warning aggregation, token-weight
budgets, and live canary proof. This module only provides the reservation seam.
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

logger = logging.getLogger("LiteLLMProxy")

# Durable keyspace is intentionally distinct from cooldown / session-owner keys.
_ADMISSION_STATE_FAMILY = "provider-lane"
_ADMISSION_STATE_KIND = "admission"
_ADMISSION_LEASE_STATE_KIND = "admission-lease"

_DEFAULT_MAX_IN_FLIGHT = 64
_DEFAULT_LEASE_TTL_SECONDS = 120.0
_MIN_LEASE_TTL_SECONDS = 5.0
_MAX_LEASE_TTL_SECONDS = 900.0

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


# Process-local counter used only when Redis is unconfigured (unit tests /
# single-process). Never replaces durable multi-worker reservation.
_local_inflight_by_lane: dict[str, int] = {}
_local_leases: dict[str, dict[str, Any]] = {}


def reset_admission_state_for_tests() -> None:
    """Clear process-local admission counters/leases (tests only)."""
    _local_inflight_by_lane.clear()
    _local_leases.clear()


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


# Inflight accounting uses a HASH of active lease_key -> units plus per-lease
# keys. Reservation always reclaims expired/missing leases first so a shared
# counter cannot be kept artificially high by newer traffic refreshing TTL.
_LUA_RESERVE = """
local inflight_hash = KEYS[1]
local lease_key = KEYS[2]
local max_in_flight = tonumber(ARGV[1])
local units = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local token = ARGV[4]
local payload = ARGV[5]

if max_in_flight == nil or units == nil or ttl == nil then
  return {-2, 0}
end
if units < 1 then
  return {-2, 0}
end

if redis.call('EXISTS', lease_key) == 1 then
  return {-3, 0}
end

-- Reclaim expired/missing leases before capacity check (bounded by max_in_flight).
local all = redis.call('HGETALL', inflight_hash)
local current = 0
for i = 1, #all, 2 do
  local existing_lease = all[i]
  local existing_units = tonumber(all[i + 1]) or 0
  if redis.call('EXISTS', existing_lease) == 1 then
    current = current + existing_units
  else
    redis.call('HDEL', inflight_hash, existing_lease)
  end
end

if (current + units) > max_in_flight then
  if current <= 0 then
    redis.call('DEL', inflight_hash)
  else
    redis.call('EXPIRE', inflight_hash, ttl)
  end
  return {0, current}
end

redis.call('HSET', inflight_hash, lease_key, units)
redis.call('SET', lease_key, payload, 'EX', ttl)
redis.call('EXPIRE', inflight_hash, ttl)
return {1, current + units}
"""


_LUA_RELEASE = """
local inflight_hash = KEYS[1]
local lease_key = KEYS[2]
local token = ARGV[1]

local raw = redis.call('GET', lease_key)
if raw then
  local ok, current = pcall(cjson.decode, raw)
  if ok and type(current) == 'table' then
    if current['reservation_token'] ~= token then
      -- Not our lease; still reclaim any expired siblings below.
      raw = nil
    end
  else
    redis.call('DEL', lease_key)
    raw = nil
  end
end

if raw then
  redis.call('HDEL', inflight_hash, lease_key)
  redis.call('DEL', lease_key)
end

-- Always reclaim expired/missing hash entries so release of a vanished lease
-- still drops its units when the hash retained them.
local all = redis.call('HGETALL', inflight_hash)
local remaining = 0
for i = 1, #all, 2 do
  local existing_lease = all[i]
  local existing_units = tonumber(all[i + 1]) or 0
  if existing_lease == lease_key then
    -- Drop our entry even if the lease key already expired.
    redis.call('HDEL', inflight_hash, existing_lease)
  elseif redis.call('EXISTS', existing_lease) == 1 then
    remaining = remaining + existing_units
  else
    redis.call('HDEL', inflight_hash, existing_lease)
  end
end

if remaining <= 0 then
  redis.call('DEL', inflight_hash)
  remaining = 0
end

if raw then
  return {1, remaining}
end
-- Lease already gone: treat as reclaimed if hash entry was dropped.
return {1, remaining}
"""


def _reclaim_expired_local_leases(*, now_epoch: Optional[float] = None) -> None:
    """Drop expired local leases and subtract their units (test/memory mode)."""
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
        units = int(record.get("units") or 1)
        if not lane:
            continue
        current = int(_local_inflight_by_lane.get(lane, 0))
        remaining = max(0, current - units)
        if remaining == 0:
            _local_inflight_by_lane.pop(lane, None)
        else:
            _local_inflight_by_lane[lane] = remaining


def _local_reserve(
    *,
    lane_fingerprint: str,
    units: int,
    max_in_flight: int,
    reservation_token: str,
    lease_ttl_seconds: float,
    provider: str,
    account_hash: Optional[str],
    now_epoch: Optional[float] = None,
) -> AdmissionDecision:
    _reclaim_expired_local_leases(now_epoch=now_epoch)
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
            detail_code="aawm_provider_lane_capacity_unavailable",
        )
    new_count = current + units
    _local_inflight_by_lane[lane_fingerprint] = new_count
    now = time.time() if now_epoch is None else float(now_epoch)
    _local_leases[reservation_token] = {
        "lane_fingerprint": lane_fingerprint,
        "units": units,
        "reservation_token": reservation_token,
        "expires_at": now + lease_ttl_seconds,
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
        )
    if code == 0:
        return AdmissionDecision(
            allowed=False,
            reason=AdmissionDenyReason.CAPACITY_UNAVAILABLE.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            inflight_after=inflight_after,
            max_in_flight=cap,
            limit_scope="concurrency",
            detail_code="aawm_provider_lane_capacity_unavailable",
        )
    return AdmissionDecision(
        allowed=True,
        reason=AdmissionAdmitReason.REDIS_UNAVAILABLE_DEGRADED.value,
        lane_fingerprint=lane_fingerprint,
        provider=provider,
        account_hash=account_hash,
        max_in_flight=cap,
        limit_scope="concurrency",
    )

async def reserve_provider_lane_admission(
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
        return _decision_from_exhaustion(
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            observation=exhausted,
        )

    reserve_units = max(1, int(units))
    cap = (
        int(max_in_flight)
        if max_in_flight is not None
        else get_provider_lane_admission_max_in_flight()
    )
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
            # Unconfigured Redis (tests / memory mode): local counter only.
            return _local_reserve(
                lane_fingerprint=lane_fingerprint,
                units=reserve_units,
                max_in_flight=cap,
                reservation_token=reservation_token,
                lease_ttl_seconds=ttl,
                provider=provider,
                account_hash=account_hash,
                now_epoch=now_epoch,
            )
        return AdmissionDecision(
            allowed=True,
            reason=AdmissionAdmitReason.REDIS_UNAVAILABLE_DEGRADED.value,
            lane_fingerprint=lane_fingerprint,
            provider=provider,
            account_hash=account_hash,
            max_in_flight=cap,
            limit_scope="concurrency",
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
        )

    code, inflight_after = _parse_lua_pair(result)
    return _decision_from_reserve_code(
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
    )


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
