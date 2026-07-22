"""Process-local alias-routing state manager (RR-054 #1).

Owns cooldown / affinity / lane-cache dicts and their asyncio.Locks so the
pass-through god-module does not declare the state maps itself.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from .oauth_token_cache import (
    antigravity_oauth_access_token_cache,
    google_oauth_access_token_cache,
)
from .memory import (
    DEFAULT_MEMORY_STATE_MAX_SIZE,
    bound_memory_map,
    extend_monotonic_cooldown,
    hydrate_affinity_memory,
    hydrate_cooldown_memory,
    remaining_cooldown_seconds,
)
from .types import Payload


@dataclass
class AliasFamilyState:
    """Cooldown + affinity state for one auto-agent alias family (codex/anthropic)."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cooldown_until_monotonic_by_key: dict[str, float] = field(default_factory=dict)
    cooldown_negative_until_monotonic_by_key: dict[str, float] = field(default_factory=dict)
    session_affinity_by_key: dict[str, Payload] = field(default_factory=dict)
    evidence_events_by_key: dict[str, list[float]] = field(default_factory=dict)

    def get_memory_cooldown_remaining(self, cooldown_key: str) -> float:
        now = time.monotonic()
        until = self.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
        if until > now:
            return max(0.0, until - now)
        self.cooldown_until_monotonic_by_key.pop(cooldown_key, None)
        return 0.0

    def is_negative_cached(self, cooldown_key: str) -> bool:
        now = time.monotonic()
        neg_until = self.cooldown_negative_until_monotonic_by_key.get(cooldown_key, 0.0)
        return neg_until > now

    def mark_negative_cache(
        self,
        cooldown_key: str,
        *,
        ttl_seconds: float,
        max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
    ) -> None:
        self.cooldown_negative_until_monotonic_by_key[cooldown_key] = time.monotonic() + max(0.0, float(ttl_seconds))
        bound_memory_map(self.cooldown_negative_until_monotonic_by_key, max_size=max_size)

    def clear_negative_cache(self, cooldown_key: str) -> None:
        self.cooldown_negative_until_monotonic_by_key.pop(cooldown_key, None)

    def set_cooldown_memory(
        self,
        cooldown_key: str,
        cooldown_seconds: float,
        *,
        max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
    ) -> None:
        until = time.monotonic() + max(0.0, float(cooldown_seconds))
        current_until = self.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
        if until > current_until:
            self.cooldown_until_monotonic_by_key[cooldown_key] = until
            self.clear_negative_cache(cooldown_key)
            bound_memory_map(self.cooldown_until_monotonic_by_key, max_size=max_size)

    def hydrate_cooldown(
        self,
        cooldown_key: str,
        expires_at_epoch: float,
        *,
        max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
    ) -> float:
        self.clear_negative_cache(cooldown_key)
        hydrate_cooldown_memory(
            memory_map=self.cooldown_until_monotonic_by_key,
            cooldown_key=cooldown_key,
            expires_at_epoch=expires_at_epoch,
            max_size=max_size,
        )
        return remaining_cooldown_seconds(self.cooldown_until_monotonic_by_key, cooldown_key)

    def record_failure_evidence(
        self,
        *,
        cooldown_key: str,
        confidence: str,
        window_seconds: float,
        max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
        now_monotonic: Optional[float] = None,
    ) -> int:
        """Record one failure-evidence timestamp for ``cooldown_key``.

        Trims events outside ``window_seconds`` and bounds the overall
        key-space to ``max_size`` (FIFO), mirroring ``bound_memory_map``.
        Returns the number of events currently within the window.
        """
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        events = self.evidence_events_by_key.setdefault(cooldown_key, [])
        events.append(now)
        cutoff = now - max(0.0, float(window_seconds))
        events[:] = [timestamp for timestamp in events if timestamp >= cutoff]
        bound_memory_map(self.evidence_events_by_key, max_size=max_size)
        # confidence is accepted for call-site symmetry with the cooldown
        # evidence gate; only marker-tier evidence needs sliding-window
        # counting, but recording is confidence-agnostic here.
        _ = confidence
        return len(events)

    def evidence_count_within_window(
        self,
        *,
        cooldown_key: str,
        window_seconds: float,
        now_monotonic: Optional[float] = None,
    ) -> int:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        events = self.evidence_events_by_key.get(cooldown_key, [])
        cutoff = now - max(0.0, float(window_seconds))
        return len([timestamp for timestamp in events if timestamp >= cutoff])

    def evidence_map_size(self) -> int:
        return len(self.evidence_events_by_key)

    def get_affinity_memory(self, session_key: str) -> Optional[Payload]:
        affinity = self.session_affinity_by_key.get(session_key)
        if not isinstance(affinity, dict):
            return None
        expires_at = affinity.get("expires_at_monotonic", 0.0)
        if isinstance(expires_at, (int, float)) and expires_at > time.monotonic():
            hydrated = dict(affinity)
            hydrated["affinity_state_source"] = affinity.get("affinity_state_source", "memory")
            return hydrated
        self.session_affinity_by_key.pop(session_key, None)
        return None

    def set_affinity_memory(
        self,
        session_key: str,
        candidate: Payload,
        *,
        ttl_seconds: float,
        max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
    ) -> None:
        self.session_affinity_by_key[session_key] = {
            "provider": candidate["provider"],
            "model": candidate["model"],
            "route_family": candidate["route_family"],
            "last_resort": bool(candidate.get("last_resort")),
            "expires_at_monotonic": time.monotonic() + max(0.0, float(ttl_seconds)),
            "affinity_state_source": "memory",
        }
        bound_memory_map(self.session_affinity_by_key, max_size=max_size)

    def hydrate_affinity(
        self,
        session_key: str,
        payload: Payload,
        expires_at_epoch: float,
        *,
        max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE,
    ) -> Payload:
        return hydrate_affinity_memory(
            memory_map=self.session_affinity_by_key,
            session_key=session_key,
            payload=payload,
            expires_at_epoch=expires_at_epoch,
            max_size=max_size,
        )


@dataclass
class MonotonicCooldownMap:
    """Generic process-local cooldown map + lock (OpenRouter/Google rate limits)."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    until_monotonic_by_key: dict[str, float] = field(default_factory=dict)

    def extend(
        self,
        key: str,
        wait_seconds: float,
        *,
        max_size: Optional[int] = DEFAULT_MEMORY_STATE_MAX_SIZE,
    ) -> float:
        return extend_monotonic_cooldown(
            self.until_monotonic_by_key,
            key,
            wait_seconds,
            max_size=max_size,
        )

    def remaining(self, key: str) -> float:
        return remaining_cooldown_seconds(self.until_monotonic_by_key, key)

    def max_remaining(self, keys: list[str]) -> float:
        now = time.monotonic()
        if not keys:
            return 0.0
        return max(
            (remaining_cooldown_seconds(self.until_monotonic_by_key, k, now=now) for k in keys),
            default=0.0,
        )


class AliasRoutingStateManager:
    """Single owner of alias-routing process-local maps + locks (RR-054 #1)."""

    def __init__(self, *, max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE) -> None:
        self.max_size = max_size
        self.codex = AliasFamilyState()
        self.anthropic = AliasFamilyState()
        self.lane_state_cache_lock = asyncio.Lock()
        self.google_lane_key_until_monotonic_by_key: dict[str, float] = {}
        self.google_lane_key_by_key: dict[str, str] = {}
        self.google_lane_negative_until_monotonic = 0.0
        self.antigravity_lane_key_until_monotonic_by_key: dict[str, float] = {}
        self.antigravity_lane_key_by_key: dict[str, str] = {}
        self.antigravity_auth_degraded_log_until_monotonic = 0.0
        self.log_until_monotonic_by_key: dict[str, float] = {}
        self.candidate_probe_locks: dict[str, asyncio.Lock] = {}
        self.candidate_probe_locks_guard = asyncio.Lock()
        self.openrouter_rate_limit = MonotonicCooldownMap()
        self.openrouter_failure_circuit = MonotonicCooldownMap()
        self.google_rate_limit = MonotonicCooldownMap()
        self.google_oauth = google_oauth_access_token_cache
        self.antigravity_oauth = antigravity_oauth_access_token_cache

    def family(self, alias_family: str) -> AliasFamilyState:
        if alias_family == "anthropic":
            return self.anthropic
        return self.codex

    async def candidate_probe_lock(
        self,
        *,
        alias_family: str,
        cooldown_key: str,
    ) -> asyncio.Lock:
        """Return one bounded process-local single-flight lock per candidate lane."""
        key = f"{alias_family}:{cooldown_key}"
        async with self.candidate_probe_locks_guard:
            lock = self.candidate_probe_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self.candidate_probe_locks[key] = lock
                bound_memory_map(self.candidate_probe_locks, max_size=self.max_size)
            return lock


# Process-wide singleton used by llm_passthrough_endpoints re-exports.
alias_routing_state = AliasRoutingStateManager()
