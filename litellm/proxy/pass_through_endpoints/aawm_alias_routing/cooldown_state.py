"""Cooldown and session-affinity state for Codex/Anthropic alias families.

The ``AliasRoutingStateManager`` singleton is injected via
:func:`configure_cooldown_state_runtime` (the integrator wires it during
god-module facade setup).  Durable Redis helpers come from ``.durable``;
bounded-memory helpers come from ``.memory``.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from .durable import (
    get_aawm_alias_routing_dual_cache,
    parse_aawm_alias_routing_durable_expiry,
    read_aawm_alias_routing_durable_payload,
    write_aawm_alias_routing_durable_payload,
)
from .lane_keys import _CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS
from .memory import (
    DEFAULT_MEMORY_STATE_MAX_SIZE,
    bound_memory_map,
    hydrate_affinity_memory,
    hydrate_cooldown_memory,
)
from .state import AliasRoutingStateManager, validate_alias_family

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_AAWM_COOLDOWN_NEGATIVE_CACHE_TTL_SECONDS = 5.0

# ---------------------------------------------------------------------------
# Injected runtime state
# ---------------------------------------------------------------------------
_manager: Optional[AliasRoutingStateManager] = None


def configure_cooldown_state_runtime(
    *,
    manager: AliasRoutingStateManager,
) -> None:
    """Bind the process-local alias-routing state manager."""
    global _manager
    _manager = manager


def _require_manager() -> AliasRoutingStateManager:
    if _manager is None:
        raise RuntimeError(
            "cooldown_state runtime not configured; "
            "call configure_cooldown_state_runtime() first"
        )
    return _manager


# ---------------------------------------------------------------------------
# Codex active cooldown
# ---------------------------------------------------------------------------


async def _get_codex_auto_agent_active_cooldown_state(
    cooldown_key: str,
) -> tuple[float, str]:
    mgr = _require_manager()
    family = mgr.codex
    async with family.lock:
        now = time.monotonic()
        until = family.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
        if until > now:
            return max(0.0, until - now), "memory"
        family.cooldown_until_monotonic_by_key.pop(cooldown_key, None)
    # RR-054 #30: negative-cache durable misses so healthy keys do not Redis-hit every call.
    async with family.lock:
        neg_until = family.cooldown_negative_until_monotonic_by_key.get(cooldown_key, 0.0)
        if neg_until > time.monotonic():
            return 0.0, "negative_cache"
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return 0.0, "local_fallback"
    # CFG-004 Defect 3: per-key read/clear barrier.  The barrier lock is
    # acquired BEFORE capturing the generation and held through the durable
    # read and hydration.  A clear (via clear_alias_family_cooldown_state)
    # acquires the same barrier lock before bumping the generation and
    # deleting the durable key.  This guarantees that a read which started
    # before a clear cannot capture the new generation and hydrate the old
    # durable value, while unrelated keys remain fully concurrent.
    _barrier = await mgr.key_barrier_lock(cooldown_key)
    async with _barrier:
        gen_before = family.get_generation(cooldown_key)
        async with mgr.lane_state_cache_lock:
            durable_payload = await read_aawm_alias_routing_durable_payload(
                alias_family="codex",
                state_kind="cooldown",
                state_key=cooldown_key,
            )
            if durable_payload is None:
                async with family.lock:
                    if family.get_generation(cooldown_key) != gen_before:
                        return 0.0, "local_fallback"
                    family.cooldown_negative_until_monotonic_by_key[cooldown_key] = (
                        time.monotonic() + _AAWM_COOLDOWN_NEGATIVE_CACHE_TTL_SECONDS
                    )
                    bound_memory_map(family.cooldown_negative_until_monotonic_by_key, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
                return 0.0, "local_fallback"
            expires_at_epoch = parse_aawm_alias_routing_durable_expiry(durable_payload)
            if expires_at_epoch is None:
                async with family.lock:
                    if family.get_generation(cooldown_key) != gen_before:
                        return 0.0, "local_fallback"
                    family.cooldown_negative_until_monotonic_by_key[cooldown_key] = (
                        time.monotonic() + _AAWM_COOLDOWN_NEGATIVE_CACHE_TTL_SECONDS
                    )
                    bound_memory_map(family.cooldown_negative_until_monotonic_by_key, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
                return 0.0, "local_fallback"
            # Generation guard + hydrate atomically under the family lock so a
            # concurrent clear (which advances generation) cannot interleave
            # between the check and the hydration write.
            async with family.lock:
                if family.get_generation(cooldown_key) != gen_before:
                    return 0.0, "local_fallback"
                family.cooldown_negative_until_monotonic_by_key.pop(cooldown_key, None)
                hydrate_cooldown_memory(
                    memory_map=family.cooldown_until_monotonic_by_key,
                    cooldown_key=cooldown_key,
                    expires_at_epoch=expires_at_epoch,
                    max_size=DEFAULT_MEMORY_STATE_MAX_SIZE,
                )
                until = family.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
                return max(0.0, until - time.monotonic()), "durable_cache"


async def _get_codex_auto_agent_active_cooldown_seconds(
    cooldown_key: str,
) -> float:
    seconds, _ = await _get_codex_auto_agent_active_cooldown_state(cooldown_key)
    return seconds


async def _set_codex_auto_agent_cooldown(
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    mgr = _require_manager()
    family = mgr.codex
    ttl_seconds = max(0.0, float(cooldown_seconds))
    async with family.lock:
        until = time.monotonic() + ttl_seconds
        current_until = family.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
        if until > current_until:
            family.cooldown_until_monotonic_by_key[cooldown_key] = until
            family.cooldown_negative_until_monotonic_by_key.pop(cooldown_key, None)
            bound_memory_map(family.cooldown_until_monotonic_by_key, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
    if ttl_seconds <= 0:
        return
    await write_aawm_alias_routing_durable_payload(
        alias_family="codex",
        state_kind="cooldown",
        state_key=cooldown_key,
        payload={"cooldown_key": cooldown_key},
        ttl_seconds=ttl_seconds,
    )


# ---------------------------------------------------------------------------
# Barrier-protected clear (CFG-004 Defect 3)
# ---------------------------------------------------------------------------


async def clear_alias_family_cooldown_state(
    *,
    alias_family: str,
    canonical_aliases: Sequence[str],
    cooldown_keys: Sequence[str],
    delete_durable: bool = True,
) -> "Any":
    """Clear cooldown state under canonical lock order (Defect 1/3).

    Canonical lock order: barrier (sorted) -> family mutation lock.
    This matches the read path (barrier -> family) and the endpoint clear
    path (barrier -> family -> probe), forming a consistent partial order
    with no cycles.

    Returns the ``CooldownClearResult`` from the manager's sync clear.
    """
    from .durable import delete_aawm_alias_routing_durable_key

    mgr = _require_manager()
    canonical = validate_alias_family(alias_family)
    sorted_keys = sorted(set(cooldown_keys))
    family_state = mgr.family(canonical)

    # Acquire per-key barrier locks in sorted order (deadlock-free).
    barriers = [await mgr.key_barrier_lock(k) for k in sorted_keys]
    for b in barriers:
        await b.acquire()
    try:
        # Acquire family mutation lock (canonical order: barrier -> family).
        async with family_state.lock:
            result = mgr.clear_cooldown_state(
                alias_family=canonical,
                canonical_aliases=canonical_aliases,
                cooldown_keys=sorted_keys,
            )
        # Durable deletion outside the family lock (barrier locks still held,
        # preventing stale reads; family lock not needed for Redis I/O).
        if delete_durable:
            for key in sorted_keys:
                try:
                    await delete_aawm_alias_routing_durable_key(
                        alias_family=canonical,
                        state_kind="cooldown",
                        state_key=key,
                    )
                except Exception:
                    # Durable deletion failure is non-fatal: the generation
                    # bump already invalidates in-flight reads.  The durable
                    # key will expire via TTL or be cleaned up on next clear.
                    pass
        return result
    finally:
        for b in reversed(barriers):
            b.release()


# ---------------------------------------------------------------------------
# Codex session affinity
# ---------------------------------------------------------------------------


async def _get_codex_auto_agent_session_affinity(
    session_key: Optional[str],
) -> Optional[dict[str, Any]]:
    if session_key is None:
        return None
    mgr = _require_manager()
    family = mgr.codex
    async with family.lock:
        affinity = family.session_affinity_by_key.get(session_key)
        if isinstance(affinity, dict):
            expires_at = affinity.get("expires_at_monotonic", 0.0)
            if isinstance(expires_at, (int, float)) and expires_at > time.monotonic():
                hydrated = dict(affinity)
                hydrated["affinity_state_source"] = affinity.get("affinity_state_source", "memory")
                return hydrated
            family.session_affinity_by_key.pop(session_key, None)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return None
    durable_payload = await read_aawm_alias_routing_durable_payload(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_key,
    )
    if durable_payload is None:
        return None
    expires_at_epoch = parse_aawm_alias_routing_durable_expiry(durable_payload)
    if expires_at_epoch is None:
        return None
    async with family.lock:
        affinity = hydrate_affinity_memory(
            memory_map=family.session_affinity_by_key,
            session_key=session_key,
            payload=durable_payload,
            expires_at_epoch=expires_at_epoch,
            max_size=DEFAULT_MEMORY_STATE_MAX_SIZE,
        )
        if not affinity:
            return None
        affinity["affinity_state_source"] = "durable_cache"
        return dict(affinity)


async def _set_codex_auto_agent_session_affinity(
    session_key: Optional[str],
    candidate: dict[str, Any],
) -> None:
    if session_key is None:
        return
    mgr = _require_manager()
    family = mgr.codex
    # Wave 3 R3-4: carry the semantic config digest observed when affinity
    # was established.  Continuations validate provider/model/route_family
    # compatibility against the active enumeration -- NOT the config hash --
    # so a priority/weight/schedule change does not break a valid pin.
    config_hash = candidate.get("config_epoch_tag")
    affinity_payload: dict[str, Any] = {
        "provider": candidate["provider"],
        "model": candidate["model"],
        "route_family": candidate["route_family"],
        "last_resort": bool(candidate.get("last_resort")),
        "config_hash": config_hash,
    }
    for field in (
        "codex_oauth_account_label",
        "codex_oauth_account_hash",
        "codex_oauth_lane_key",
    ):
        value = candidate.get(field)
        if isinstance(value, str) and value:
            affinity_payload[field] = value
    async with family.lock:
        family.session_affinity_by_key[session_key] = {
            **affinity_payload,
            "expires_at_monotonic": (
                time.monotonic()
                + _CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS
            ),
            "affinity_state_source": "memory",
        }
        bound_memory_map(family.session_affinity_by_key, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
    durable_payload: dict[str, Any] = {
        key: value
        for key, value in affinity_payload.items()
        if key != "config_hash"
    }
    if config_hash is not None:
        durable_payload["config_hash"] = config_hash
    await write_aawm_alias_routing_durable_payload(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_key,
        payload=durable_payload,
        ttl_seconds=_CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS,
    )


# ---------------------------------------------------------------------------
# Anthropic active cooldown
# ---------------------------------------------------------------------------


async def _get_anthropic_auto_agent_active_cooldown_state(
    cooldown_key: str,
) -> tuple[float, str]:
    mgr = _require_manager()
    family = mgr.anthropic
    async with family.lock:
        now = time.monotonic()
        until = family.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
        if until > now:
            return max(0.0, until - now), "memory"
        family.cooldown_until_monotonic_by_key.pop(cooldown_key, None)
        neg_until = family.cooldown_negative_until_monotonic_by_key.get(cooldown_key, 0.0)
        if neg_until > now:
            return 0.0, "negative_cache"
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return 0.0, "local_fallback"
    # CFG-004 Defect 3: per-key read/clear barrier (mirrors codex path).
    _barrier = await mgr.key_barrier_lock(cooldown_key)
    async with _barrier:
        gen_before = family.get_generation(cooldown_key)
        async with mgr.lane_state_cache_lock:
            durable_payload = await read_aawm_alias_routing_durable_payload(
                alias_family="anthropic",
                state_kind="cooldown",
                state_key=cooldown_key,
            )
            if durable_payload is None:
                async with family.lock:
                    if family.get_generation(cooldown_key) != gen_before:
                        return 0.0, "local_fallback"
                    family.cooldown_negative_until_monotonic_by_key[cooldown_key] = (
                        time.monotonic() + _AAWM_COOLDOWN_NEGATIVE_CACHE_TTL_SECONDS
                    )
                    bound_memory_map(family.cooldown_negative_until_monotonic_by_key, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
                return 0.0, "local_fallback"
            expires_at_epoch = parse_aawm_alias_routing_durable_expiry(durable_payload)
            if expires_at_epoch is None:
                async with family.lock:
                    if family.get_generation(cooldown_key) != gen_before:
                        return 0.0, "local_fallback"
                    family.cooldown_negative_until_monotonic_by_key[cooldown_key] = (
                        time.monotonic() + _AAWM_COOLDOWN_NEGATIVE_CACHE_TTL_SECONDS
                    )
                    bound_memory_map(family.cooldown_negative_until_monotonic_by_key, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
                return 0.0, "local_fallback"
            async with family.lock:
                if family.get_generation(cooldown_key) != gen_before:
                    return 0.0, "local_fallback"
                family.cooldown_negative_until_monotonic_by_key.pop(cooldown_key, None)
                hydrate_cooldown_memory(
                    memory_map=family.cooldown_until_monotonic_by_key,
                    cooldown_key=cooldown_key,
                    expires_at_epoch=expires_at_epoch,
                    max_size=DEFAULT_MEMORY_STATE_MAX_SIZE,
                )
                until = family.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
                return max(0.0, until - time.monotonic()), "durable_cache"


async def _get_anthropic_auto_agent_active_cooldown_seconds(
    cooldown_key: str,
) -> float:
    seconds, _ = await _get_anthropic_auto_agent_active_cooldown_state(cooldown_key)
    return seconds


async def _set_anthropic_auto_agent_cooldown(
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    mgr = _require_manager()
    family = mgr.anthropic
    ttl_seconds = max(0.0, float(cooldown_seconds))
    async with family.lock:
        until = time.monotonic() + ttl_seconds
        current_until = family.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
        if until > current_until:
            family.cooldown_until_monotonic_by_key[cooldown_key] = until
            family.cooldown_negative_until_monotonic_by_key.pop(cooldown_key, None)
            bound_memory_map(family.cooldown_until_monotonic_by_key, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
    if ttl_seconds <= 0:
        return
    await write_aawm_alias_routing_durable_payload(
        alias_family="anthropic",
        state_kind="cooldown",
        state_key=cooldown_key,
        payload={"cooldown_key": cooldown_key},
        ttl_seconds=ttl_seconds,
    )


# ---------------------------------------------------------------------------
# Anthropic session affinity
# ---------------------------------------------------------------------------


async def _get_anthropic_auto_agent_session_affinity(
    session_key: Optional[str],
) -> Optional[dict[str, Any]]:
    if session_key is None:
        return None
    mgr = _require_manager()
    family = mgr.anthropic
    async with family.lock:
        affinity = family.session_affinity_by_key.get(session_key)
        if isinstance(affinity, dict):
            expires_at = affinity.get("expires_at_monotonic", 0.0)
            if isinstance(expires_at, (int, float)) and expires_at > time.monotonic():
                hydrated = dict(affinity)
                hydrated["affinity_state_source"] = affinity.get("affinity_state_source", "memory")
                return hydrated
            family.session_affinity_by_key.pop(session_key, None)
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return None
    durable_payload = await read_aawm_alias_routing_durable_payload(
        alias_family="anthropic",
        state_kind="affinity",
        state_key=session_key,
    )
    if durable_payload is None:
        return None
    expires_at_epoch = parse_aawm_alias_routing_durable_expiry(durable_payload)
    if expires_at_epoch is None:
        return None
    async with family.lock:
        affinity = hydrate_affinity_memory(
            memory_map=family.session_affinity_by_key,
            session_key=session_key,
            payload=durable_payload,
            expires_at_epoch=expires_at_epoch,
            max_size=DEFAULT_MEMORY_STATE_MAX_SIZE,
        )
        if not affinity:
            return None
        affinity["affinity_state_source"] = "durable_cache"
        return dict(affinity)


async def _set_anthropic_auto_agent_session_affinity(
    session_key: Optional[str],
    candidate: dict[str, Any],
) -> None:
    if session_key is None:
        return
    mgr = _require_manager()
    family = mgr.anthropic
    # CFG-001: carry the semantic config digest observed when affinity
    # was established, mirroring the Codex setter.  Without this the
    # snapshot-membership check in _find_anthropic_auto_agent_affinity_candidate
    # is bypassed (config_hash is None) and affinity silently falls through
    # to the eligibility-filtered candidate list.
    config_hash = candidate.get("config_epoch_tag")
    affinity_payload: dict[str, Any] = {
        "provider": candidate["provider"],
        "model": candidate["model"],
        "route_family": candidate["route_family"],
        "last_resort": bool(candidate.get("last_resort")),
        "config_hash": config_hash,
    }
    for field in (
        "codex_oauth_account_label",
        "codex_oauth_account_hash",
        "codex_oauth_lane_key",
    ):
        value = candidate.get(field)
        if isinstance(value, str) and value:
            affinity_payload[field] = value
    async with family.lock:
        family.session_affinity_by_key[session_key] = {
            **affinity_payload,
            "expires_at_monotonic": (
                time.monotonic()
                + _CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS
            ),
            "affinity_state_source": "memory",
        }
        bound_memory_map(family.session_affinity_by_key, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE)
    durable_payload: dict[str, Any] = {
        key: value
        for key, value in affinity_payload.items()
        if key != "config_hash"
    }
    if config_hash is not None:
        durable_payload["config_hash"] = config_hash
    await write_aawm_alias_routing_durable_payload(
        alias_family="anthropic",
        state_kind="affinity",
        state_key=session_key,
        payload=durable_payload,
        ttl_seconds=_CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS,
    )


# ---------------------------------------------------------------------------
# Merged Codex/OpenAI cooldown state
# ---------------------------------------------------------------------------


def _format_merged_alias_family_cooldown_state_source(
    *,
    anthropic_seconds: float,
    anthropic_source: str,
    codex_seconds: float,
    codex_source: str,
) -> tuple[float, str]:
    family_states: list[tuple[float, str]] = []
    if anthropic_seconds > 0:
        family_states.append(
            (
                anthropic_seconds,
                f"anthropic_family:{anthropic_source}",
            )
        )
    if codex_seconds > 0:
        family_states.append((codex_seconds, f"codex_family:{codex_source}"))
    if not family_states:
        return 0.0, "local_fallback"
    family_states.sort(key=lambda item: item[0], reverse=True)
    merged_seconds = family_states[0][0]
    merged_source = "+".join(source for _, source in family_states)
    return merged_seconds, merged_source


async def _get_anthropic_auto_agent_merged_codex_openai_cooldown_state(
    cooldown_key: str,
) -> tuple[float, str]:
    (
        anthropic_seconds,
        anthropic_source,
    ) = await _get_anthropic_auto_agent_active_cooldown_state(cooldown_key)
    codex_seconds, codex_source = await _get_codex_auto_agent_active_cooldown_state(cooldown_key)
    return _format_merged_alias_family_cooldown_state_source(
        anthropic_seconds=anthropic_seconds,
        anthropic_source=anthropic_source,
        codex_seconds=codex_seconds,
        codex_source=codex_source,
    )


# ---------------------------------------------------------------------------
# Synchronous cooldown-memory publication (R3-1)
# ---------------------------------------------------------------------------


def _publish_codex_cooldown_memory(*, keys: Sequence[str], seconds: float) -> None:
    """Synchronously publish cooldowns into codex family memory (R3-1).

    Direct ``state.py`` writes -- no awaitable lock -- so the retry loop can
    call this inside the probe lock without violating the
    ``probe_lock -> (nothing awaitable)`` ordering.
    """
    mgr = _require_manager()
    for key in keys:
        mgr.codex.set_cooldown_memory(key, seconds)


def _publish_anthropic_cooldown_memory(*, keys: Sequence[str], seconds: float) -> None:
    """Synchronously publish cooldowns into anthropic family memory (R3-1)."""
    mgr = _require_manager()
    for key in keys:
        mgr.anthropic.set_cooldown_memory(key, seconds)


# ---------------------------------------------------------------------------
# State-source attachment
# ---------------------------------------------------------------------------


def _attach_aawm_alias_routing_state_sources(
    selection: dict[str, Any],
    *,
    affinity: Optional[dict[str, Any]] = None,
    selected_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    enriched = dict(selection)
    if affinity is not None:
        enriched["affinity_state_source"] = affinity.get("affinity_state_source", "local_fallback")
    if selected_state is not None:
        enriched["cooldown_state_source"] = selected_state.get("cooldown_state_source", "local_fallback")
    return enriched


# ---------------------------------------------------------------------------
# Host-globals rebinding (Wave 5B)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType

_HOST_FUNCTION_NAMES = (
    "_get_codex_auto_agent_active_cooldown_state",
    "_get_codex_auto_agent_active_cooldown_seconds",
    "_set_codex_auto_agent_cooldown",
    "_get_codex_auto_agent_session_affinity",
    "_set_codex_auto_agent_session_affinity",
    "_get_anthropic_auto_agent_active_cooldown_state",
    "_get_anthropic_auto_agent_active_cooldown_seconds",
    "_set_anthropic_auto_agent_cooldown",
    "_get_anthropic_auto_agent_session_affinity",
    "_set_anthropic_auto_agent_session_affinity",
    "_format_merged_alias_family_cooldown_state_source",
    "_get_anthropic_auto_agent_merged_codex_openai_cooldown_state",
    "_publish_codex_cooldown_memory",
    "_publish_anthropic_cooldown_memory",
    "_attach_aawm_alias_routing_state_sources",
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
    host_globals.update(
        {
            "_require_manager": _require_manager,
            "DEFAULT_MEMORY_STATE_MAX_SIZE": DEFAULT_MEMORY_STATE_MAX_SIZE,
            "bound_memory_map": (
                lambda cache, *, max_size=DEFAULT_MEMORY_STATE_MAX_SIZE: (
                    host_globals["_bound_aawm_alias_routing_memory_map"](
                        cache,
                        max_size=max_size,
                    )
                )
            ),
            "hydrate_affinity_memory": (
                lambda **kwargs: host_globals[
                    "_hydrate_aawm_alias_routing_affinity_memory"
                ](
                    memory_map=kwargs["memory_map"],
                    session_key=kwargs["session_key"],
                    payload=kwargs["payload"],
                    expires_at_epoch=kwargs["expires_at_epoch"],
                )
            ),
            "hydrate_cooldown_memory": (
                lambda **kwargs: host_globals[
                    "_hydrate_aawm_alias_routing_cooldown_memory"
                ](
                    memory_map=kwargs["memory_map"],
                    cooldown_key=kwargs["cooldown_key"],
                    expires_at_epoch=kwargs["expires_at_epoch"],
                )
            ),
            "get_aawm_alias_routing_dual_cache": (
                lambda: host_globals["_get_aawm_alias_routing_dual_cache"]()
            ),
            "parse_aawm_alias_routing_durable_expiry": (
                lambda payload: host_globals[
                    "_parse_aawm_alias_routing_durable_expiry"
                ](payload)
            ),
            "read_aawm_alias_routing_durable_payload": (
                lambda **kwargs: host_globals[
                    "_read_aawm_alias_routing_durable_payload"
                ](**kwargs)
            ),
            "write_aawm_alias_routing_durable_payload": (
                lambda **kwargs: host_globals[
                    "_write_aawm_alias_routing_durable_payload"
                ](**kwargs)
            ),
        }
    )
