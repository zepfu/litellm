"""Bounded in-memory map helpers for AAWM alias routing (RR-054 #1/#12)."""

from __future__ import annotations

import time
from typing import MutableMapping, Optional, TypeVar

from .types import Payload

DEFAULT_MEMORY_STATE_MAX_SIZE = 4096
MapKeyT = TypeVar("MapKeyT")
MapValueT = TypeVar("MapValueT")


def bound_memory_map(
    cache: MutableMapping[MapKeyT, MapValueT],
    *,
    max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
) -> None:
    """FIFO-trim a process-local map to ``max_size`` entries."""
    while len(cache) > max_size:
        try:
            oldest = next(iter(cache))
        except StopIteration:
            break
        cache.pop(oldest, None)


def hydrate_cooldown_memory(
    *,
    memory_map: MutableMapping[str, float],
    cooldown_key: str,
    expires_at_epoch: float,
    max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
) -> None:
    remaining = max(0.0, float(expires_at_epoch) - time.time())
    if remaining <= 0:
        return
    until = time.monotonic() + remaining
    current_until = float(memory_map.get(cooldown_key, 0.0) or 0.0)
    if until > current_until:
        memory_map[cooldown_key] = until
        bound_memory_map(memory_map, max_size=max_size)


def hydrate_affinity_memory(
    *,
    memory_map: MutableMapping[str, Payload],
    session_key: str,
    payload: Payload,
    expires_at_epoch: float,
    max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
) -> Payload:
    remaining = max(0.0, float(expires_at_epoch) - time.time())
    if remaining <= 0:
        return {}
    affinity: Payload = {
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "route_family": payload.get("route_family"),
        "last_resort": bool(payload.get("last_resort")),
        "expires_at_monotonic": time.monotonic() + remaining,
    }
    # Do not clobber a fresher in-memory affinity written while Redis read ran.
    existing = memory_map.get(session_key)
    if isinstance(existing, dict):
        existing_until = existing.get("expires_at_monotonic", 0.0)
        affinity_until = affinity["expires_at_monotonic"]
        if (
            isinstance(existing_until, (int, float))
            and isinstance(affinity_until, (int, float))
            and float(existing_until) > float(affinity_until)
        ):
            return dict(existing)
    memory_map[session_key] = affinity
    bound_memory_map(memory_map, max_size=max_size)
    return dict(affinity)


def extend_monotonic_cooldown(
    memory_map: MutableMapping[str, float],
    key: str,
    wait_seconds: float,
    *,
    max_size: Optional[int] = DEFAULT_MEMORY_STATE_MAX_SIZE,
) -> float:
    """Set ``key`` cooldown to max(existing, now+wait). Returns the until-monotonic."""
    until = time.monotonic() + max(0.0, float(wait_seconds))
    current_until = float(memory_map.get(key, 0.0) or 0.0)
    if until > current_until:
        memory_map[key] = until
        if max_size is not None:
            bound_memory_map(memory_map, max_size=max_size)
        return until
    return current_until


def remaining_cooldown_seconds(
    memory_map: MutableMapping[str, float],
    key: str,
    *,
    now: Optional[float] = None,
) -> float:
    clock = time.monotonic() if now is None else now
    until = float(memory_map.get(key, 0.0) or 0.0)
    return max(0.0, until - clock)


def max_remaining_cooldown_seconds(
    memory_map: MutableMapping[str, float],
    keys: list[str],
    *,
    now: Optional[float] = None,
) -> float:
    clock = time.monotonic() if now is None else now
    if not keys:
        return 0.0
    return max(
        (remaining_cooldown_seconds(memory_map, key, now=clock) for key in keys),
        default=0.0,
    )
