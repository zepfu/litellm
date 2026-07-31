"""Process-local alias-routing state manager (RR-054 #1).

Owns cooldown and affinity maps, their asyncio.Locks, probe-lock state, the
read-pilot evidence gate, round-robin cursor, and OpenRouter caches so the
pass-through god-module does not declare the state maps itself.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from .memory import (
    DEFAULT_MEMORY_STATE_MAX_SIZE,
    bound_memory_map,
    extend_monotonic_cooldown,
    hydrate_affinity_memory,
    hydrate_cooldown_memory,
    remaining_cooldown_seconds,
)
from .types import Payload

_VALID_ALIAS_FAMILIES = frozenset({"codex", "anthropic"})


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

    def clear_cooldown_state(
        self,
        *,
        cooldown_keys: Sequence[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Remove positive/negative cooldown and evidence for specific keys.

        Preserves session affinity and all unrelated keys.  Returns
        (positive_cleared, negative_cleared, evidence_cleared) lists
        naming the keys actually removed from each map.
        """
        positive_cleared: list[str] = []
        negative_cleared: list[str] = []
        evidence_cleared: list[str] = []
        for key in cooldown_keys:
            if self.cooldown_until_monotonic_by_key.pop(key, None) is not None:
                positive_cleared.append(key)
            if self.cooldown_negative_until_monotonic_by_key.pop(key, None) is not None:
                negative_cleared.append(key)
            if self.evidence_events_by_key.pop(key, None) is not None:
                evidence_cleared.append(key)
        return positive_cleared, negative_cleared, evidence_cleared

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

    def clear_for_tests(self) -> None:
        """Clear every process-local map IN PLACE (test-support only).

        Uses ``.clear()`` so any module-global alias bound to these same dict
        objects (see ``llm_passthrough_endpoints``) observes the reset without
        needing to be rebound.
        """
        self.cooldown_until_monotonic_by_key.clear()
        self.cooldown_negative_until_monotonic_by_key.clear()
        self.session_affinity_by_key.clear()
        self.evidence_events_by_key.clear()


@dataclass
class CooldownClearResult:
    """Typed result of a targeted cooldown-state clear operation (CFG-004)."""

    alias_family: str
    positive_keys_cleared: list[str] = field(default_factory=list)
    negative_keys_cleared: list[str] = field(default_factory=list)
    evidence_keys_cleared: list[str] = field(default_factory=list)
    read_pilot_keys_cleared: list[str] = field(default_factory=list)
    affinity_keys_preserved: int = 0


class LaneIdentityIndex:
    """Bounded reverse index from opaque identity hash to lane keys (CFG-004).

    Maps credential-derived identity hashes to the set of cooldown/lane keys
    that reference them, enabling targeted cleanup without exposing raw
    credentials or hashes to external callers.  All access is through internal
    methods; raw identity hashes and lane keys are never leaked outside the
    index boundary.

    Bounded by max_identities (FIFO eviction of oldest identity) and
    max_lanes_per_identity (arbitrary lane evicted within an identity).
    """

    def __init__(
        self,
        *,
        max_identities: int = 4096,
        max_lanes_per_identity: int = 64,
    ) -> None:
        self._max_identities = max(1, int(max_identities))
        self._max_lanes_per_identity = max(1, int(max_lanes_per_identity))
        self._lock = threading.Lock()
        self._index: dict[str, set[str]] = {}

    def register(self, *, identity_hash: str, lane_key: str) -> bool:
        """Associate lane_key with identity_hash.

        Returns True if a new mapping was added, False if the lane was
        already present (no-op).  Thread-safe: the whole read-modify-write
        (including bounds eviction) runs under ``self._lock``.
        """
        with self._lock:
            lanes = self._index.get(identity_hash)
            if lanes is None:
                if len(self._index) >= self._max_identities:
                    try:
                        oldest = next(iter(self._index))
                        self._index.pop(oldest, None)
                    except StopIteration:
                        pass
                lanes = set()
                self._index[identity_hash] = lanes
            if lane_key in lanes:
                return False
            if len(lanes) >= self._max_lanes_per_identity:
                try:
                    evict = next(iter(lanes))
                    lanes.discard(evict)
                except StopIteration:
                    pass
            lanes.add(lane_key)
            return True

    def lanes_for(self, identity_hash: str) -> frozenset[str]:
        """Return the lane keys registered for identity_hash.  Thread-safe."""
        with self._lock:
            lanes = self._index.get(identity_hash)
            if lanes is None:
                return frozenset()
            return frozenset(lanes)

    def unregister_lane(self, *, identity_hash: str, lane_key: str) -> bool:
        """Remove one lane from an identity.  Returns True if removed.  Thread-safe."""
        with self._lock:
            lanes = self._index.get(identity_hash)
            if lanes is None:
                return False
            removed = lane_key in lanes
            lanes.discard(lane_key)
            if not lanes:
                self._index.pop(identity_hash, None)
            return removed

    def remove_identity(self, identity_hash: str) -> frozenset[str]:
        """Remove an identity entirely, returning all its lane keys.  Thread-safe."""
        with self._lock:
            lanes = self._index.pop(identity_hash, None)
            if lanes is None:
                return frozenset()
            return frozenset(lanes)

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def clear(self) -> None:
        """Remove all identities (test-support / reset).  Thread-safe."""
        with self._lock:
            self._index.clear()


@dataclass
class MonotonicCooldownMap:
    """Generic process-local cooldown map and lock."""

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

    def clear_for_tests(self) -> None:
        """Clear the cooldown map IN PLACE (test-support only)."""
        self.until_monotonic_by_key.clear()


class AliasRoutingStateManager:
    """Single owner of alias-routing process-local maps + locks (RR-054 #1)."""

    def __init__(self, *, max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE) -> None:
        from .classification import CooldownEvidenceGate  # lazy: avoid state->classification->state cycle
        self.max_size = max_size
        self.codex = AliasFamilyState()
        self.anthropic = AliasFamilyState()
        self.lane_identity_index = LaneIdentityIndex()
        self.lane_state_cache_lock = asyncio.Lock()
        self.log_until_monotonic_by_key: dict[str, float] = {}
        self.candidate_probe_locks: dict[str, asyncio.Lock] = {}
        self.candidate_probe_locks_guard = asyncio.Lock()
        self.openrouter_rate_limit = MonotonicCooldownMap()
        self.openrouter_failure_circuit = MonotonicCooldownMap()
        # Wave 5B: read-pilot evidence gate with its own separate AliasFamilyState
        self.read_pilot_gate = CooldownEvidenceGate(family_state=AliasFamilyState())
        # Wave 5B: per-alias round-robin rotation cursor
        self.round_robin_cursor: dict[tuple[str, str], int] = {}
        # Wave 5B: OpenRouter free-daily-quota cache (immutable tuple) + lock
        self._openrouter_free_quota_cache: Tuple[Optional[float], float] = (None, 0.0)
        self.openrouter_free_quota_lock = asyncio.Lock()

    def family(self, alias_family: str) -> AliasFamilyState:
        if alias_family == "anthropic":
            return self.anthropic
        return self.codex

    def clear_cooldown_state(
        self,
        *,
        alias_family: str,
        cooldown_keys: Sequence[str],
    ) -> CooldownClearResult:
        """Targeted removal of cooldown-derived state for named keys (CFG-004).

        Removes positive cooldown, negative cache, evidence events, and
        read-pilot gate state (codex only) for the given keys.  Preserves
        session affinity and all unrelated keys.  Durable/Redis state is NOT
        touched here; callers use ``durable.delete_aawm_alias_routing_durable_key``
        for that.

        Raises ``ValueError`` for unknown ``alias_family``.
        """
        normalized = alias_family.strip().lower()
        if normalized not in _VALID_ALIAS_FAMILIES:
            raise ValueError(
                f"Unknown alias_family {alias_family!r}; "
                f"expected one of {sorted(_VALID_ALIAS_FAMILIES)}"
            )
        family = self.family(normalized)
        positive, negative, evidence = family.clear_cooldown_state(
            cooldown_keys=cooldown_keys,
        )
        read_pilot_cleared: list[str] = []
        # Read-pilot gate is codex-owned; never clear it for anthropic.
        if normalized == "codex":
            for key in cooldown_keys:
                if self.read_pilot_gate._key_state.pop(key, None) is not None:
                    read_pilot_cleared.append(key)
                self.read_pilot_gate._family_state.evidence_events_by_key.pop(key, None)
        return CooldownClearResult(
            alias_family=normalized,
            positive_keys_cleared=positive,
            negative_keys_cleared=negative,
            evidence_keys_cleared=evidence,
            read_pilot_keys_cleared=read_pilot_cleared,
            affinity_keys_preserved=len(family.session_affinity_by_key),
        )

    def get_openrouter_free_quota_cache(self) -> Tuple[Optional[float], float]:
        """Return the current OpenRouter free-daily-quota cache tuple."""
        return self._openrouter_free_quota_cache

    def set_openrouter_free_quota_cache(self, value: Tuple[Optional[float], float]) -> None:
        """Replace the OpenRouter free-daily-quota cache tuple (immutable update)."""
        self._openrouter_free_quota_cache = value

    def reset_for_tests(self) -> None:
        """Clear all manager-owned process-local state IN PLACE (test-support only).

        Every map is cleared with ``.clear()`` -- never reassigned -- so the
        module-global aliases in ``llm_passthrough_endpoints`` (which are bound
        to these exact dict objects) stay valid and observe the reset. The
        OpenRouter quota cache is reset by replacing its immutable tuple.
        """
        self.codex.clear_for_tests()
        self.anthropic.clear_for_tests()
        self.log_until_monotonic_by_key.clear()
        self.candidate_probe_locks.clear()
        self.openrouter_rate_limit.clear_for_tests()
        self.openrouter_failure_circuit.clear_for_tests()
        # Wave 5B: gate, cursor, quota
        self.read_pilot_gate._key_state.clear()
        self.read_pilot_gate._family_state.evidence_events_by_key.clear()
        self.round_robin_cursor.clear()
        self.lane_identity_index.clear()
        self._openrouter_free_quota_cache = (None, 0.0)

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
