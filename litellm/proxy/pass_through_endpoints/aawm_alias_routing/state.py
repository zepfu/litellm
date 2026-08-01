"""Process-local alias-routing state manager (RR-054 #1).

Owns cooldown and affinity maps, their asyncio.Locks, probe-lock state, the
read-pilot evidence gate, round-robin cursor, OpenRouter caches, and
CFG-004 publication-intent tracking so the pass-through god-module does not
declare the state maps itself.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum, auto
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

# Canonical family names accepted by validation paths (e.g. clear_cooldown_state).
_VALID_ALIAS_FAMILIES = frozenset({"codex", "anthropic"})

# ---------------------------------------------------------------------------
# Shared exact alias-family canonicalization (single source of truth)
# ---------------------------------------------------------------------------
#
# Every known alias_family label maps to exactly one canonical family name.
# Substring matching is intentionally NOT used: labels such as
# ``not_anthropic``, ``xanthropicx``, or ``codex_anthropic`` must NOT
# resolve to the Anthropic family.  Unknown labels default to ``"codex"``
# per the established default-family contract.

_ALIAS_FAMILY_CANONICAL_MAP: dict[str, str] = {
    "codex": "codex",
    "codex_auto_agent": "codex",
    "anthropic": "anthropic",
    "anthropic_auto_agent": "anthropic",
}

_DEFAULT_CANONICAL_FAMILY = "codex"


def validate_alias_family(alias_family: str) -> str:
    """Strictly validate and canonicalize an alias_family label.

    Accepts only canonical ``codex``/``anthropic`` and production aliases
    ``codex_auto_agent``/``anthropic_auto_agent`` (case/whitespace-insensitive).
    Raises ``ValueError`` for unknown labels BEFORE any state lookup, lock
    acquisition, generation bump, or clear can occur.
    """
    stripped = alias_family.strip().lower()
    canonical = _ALIAS_FAMILY_CANONICAL_MAP.get(stripped)
    if canonical is None:
        raise ValueError(
            f"Unknown alias_family {alias_family!r}; "
            f"expected one of {sorted(_ALIAS_FAMILY_CANONICAL_MAP)}"
        )
    return canonical


def canonicalize_alias_family(alias_family: str) -> str:
    """Resolve an alias_family label to its canonical family name.

    Uses exact matching against ``_ALIAS_FAMILY_CANONICAL_MAP``.  Unknown
    labels default to ``"codex"`` (established default-family contract).
    """
    return _ALIAS_FAMILY_CANONICAL_MAP.get(
        alias_family.strip().lower(),
        _DEFAULT_CANONICAL_FAMILY,
    )


class RegisterBatchOutcome(Enum):
    """Result of LaneIdentityIndex.register_batch (CFG-004).

    ADDED: at least one new lane mapping was created.
    IDEMPOTENT: all lane_keys were already registered (repeated publication).
    CAPACITY_REJECTED: capacity limit exceeded; no mutation performed.
    """

    ADDED = "added"
    IDEMPOTENT = "idempotent"
    CAPACITY_REJECTED = "capacity_rejected"


class ClearReservationStatus(Enum):
    """Lifecycle status of a clear reservation (CFG-004 Wave A)."""

    ACTIVE = auto()
    COMPLETED = auto()


@dataclass
class ClearReservation:
    """Atomic multi-identity clear reservation blocking first-ever publication.

    Created when an operator or automated process clears cooldown state for
    one or more identities.  While ACTIVE, any candidate loop that encounters
    a cooldown_key covered by this reservation must wait/reselect WITHOUT
    performing provider I/O.  This prevents a race where a first-ever lane
    publication succeeds between the clear request and the clear execution.
    """

    reservation_id: str
    alias_family: str
    identity_hashes: frozenset[str]
    cooldown_keys: frozenset[str]
    status: ClearReservationStatus = ClearReservationStatus.ACTIVE
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def complete(self) -> None:
        self.status = ClearReservationStatus.COMPLETED
        self.done.set()


@dataclass
class AliasFamilyState:
    """Cooldown + affinity state for one auto-agent alias family (codex/anthropic)."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cooldown_until_monotonic_by_key: dict[str, float] = field(default_factory=dict)
    cooldown_negative_until_monotonic_by_key: dict[str, float] = field(default_factory=dict)
    session_affinity_by_key: dict[str, Payload] = field(default_factory=dict)
    evidence_events_by_key: dict[str, list[float]] = field(default_factory=dict)
    # CFG-004 Wave A: monotonic generation counter incremented on every clear.
    # Durable reads capture the generation before I/O and discard/retry if it
    # changes before hydration, preventing stale rehydration after a clear.
    # Per-key generation: each key tracks its own clear count so an unrelated
    # clear of key A cannot discard a valid in-flight read for key B.
    cooldown_generation_by_key: dict[str, int] = field(default_factory=dict)

    def get_generation(self, cooldown_key: str) -> int:
        """Return the per-key generation counter (0 for never-cleared keys)."""
        return self.cooldown_generation_by_key.get(cooldown_key, 0)

    def bump_generation(self, keys: Sequence[str]) -> None:
        """Increment per-key generation counters for the given keys.

        Called by ``clear_cooldown_state`` so stale durable reads that
        captured a prior generation detect the intervening clear.
        """
        for key in keys:
            self.cooldown_generation_by_key[key] = (
                self.cooldown_generation_by_key.get(key, 0) + 1
            )

    def get_memory_cooldown_remaining(self, cooldown_key: str) -> float:
        now = time.monotonic()
        until = self.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
        if until > now:
            return max(0.0, until - now)
        self.cooldown_until_monotonic_by_key.pop(cooldown_key, None)
        return 0.0

    def peek_cooldown_remaining(self, cooldown_key: str) -> float:
        """Non-mutating, lock-free cooldown remaining check (TOCTOU guard).

        Safe to call while holding the probe lock: the publication
        transaction also holds this probe lock while mutating, so the
        read sees a consistent snapshot.  Does NOT acquire the family
        lock (preserving canonical family->probe ordering) and does NOT
        pop expired entries (no mutation).
        """
        until = self.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0)
        now = time.monotonic()
        return max(0.0, until - now) if until > now else 0.0

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
        # Advance per-key generation so stale durable reads are detected.
        self.bump_generation(cooldown_keys)
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
        # Bump generation for every tracked key so any in-flight durable read
        # (which captured a prior generation) is invalidated.
        self.bump_generation(list(self.cooldown_generation_by_key))


@dataclass
class CooldownClearResult:
    """Typed result of a targeted cooldown-state clear operation (CFG-004)."""

    alias_family: str
    positive_keys_cleared: list[str] = field(default_factory=list)
    negative_keys_cleared: list[str] = field(default_factory=list)
    evidence_keys_cleared: list[str] = field(default_factory=list)
    read_pilot_keys_cleared: list[str] = field(default_factory=list)
    affinity_keys_preserved: int = 0



@dataclass
class PublicationIntent:
    """Manager-owned publication intent for CFG-004 single-flight probing.

    Created under the selected probe lock BEFORE provider I/O.  Followers
    that acquire the same probe lock while the intent is active await
    done and retry selection instead of re-probing.  The leader
    attaches the immutable plan on failure, releases the probe lock,
    performs the cooldown mutation with NO pre-held lock, then signals
    done.
    """

    transaction_id: str
    alias_family: str
    cooldown_keys: frozenset[str]
    identity_hash: str = ""
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: Optional[BaseException] = None
    plan: Optional[object] = None  # CooldownPublicationPlan, set on failure

    def complete(self, *, error: Optional[BaseException] = None) -> None:
        self.error = error
        self.done.set()


@dataclass
class ReserveOrClaimResult:
    """Result of PublicationIntentRegistry.reserve_or_claim (CFG-004 Wave A).

    ``is_leader`` is True when this call created the intent; False when an
    existing active intent was found (follower path).
    """

    intent: PublicationIntent
    is_leader: bool


class ClaimOutcome(Enum):
    """Outcome of PublicationIntentRegistry.claim_publication_or_wait (CFG-004)."""

    LEADER = auto()
    FOLLOWER = auto()
    BLOCKED_BY_CLEAR = auto()


@dataclass
class ClaimPublicationResult:
    """Result of atomic publication-claim vs clear-reservation check.

    Closes the race where a clear reservation can be created between the
    intent claim and the separate clear-reservation check (Defect 1).
    Under the registry threading lock, this atomically checks for active
    clear reservations AND active intents, then claims a new leader intent
    only if neither blocks the publication.
    """

    outcome: ClaimOutcome
    intent: Optional[PublicationIntent] = None
    clear_reservation: Optional[ClearReservation] = None


def _new_transaction_id() -> str:
    return uuid.uuid4().hex


class PublicationIntentRegistry:
    """Tracks active publication intents keyed by cooldown_key.

    Thread-safe for the check/register/remove operations.  The asyncio
    Event inside each intent is used for coroutine-level coordination.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._intents: dict[tuple[str, str], PublicationIntent] = {}
        self._identity_intents: dict[tuple[str, str], PublicationIntent] = {}
        self._clear_reservations: dict[tuple[str, str], ClearReservation] = {}
        self._identity_reservations: dict[tuple[str, str], ClearReservation] = {}

    def create(
        self,
        *,
        alias_family: str,
        cooldown_keys: frozenset[str],
        identity_hash: str = "",
    ) -> PublicationIntent:
        canonical = validate_alias_family(alias_family)
        intent = PublicationIntent(
            transaction_id=_new_transaction_id(),
            alias_family=canonical,
            cooldown_keys=cooldown_keys,
            identity_hash=identity_hash,
        )
        with self._lock:
            for key in cooldown_keys:
                self._intents[(canonical, key)] = intent
            if identity_hash:
                self._identity_intents[(canonical, identity_hash)] = intent
        return intent

    def reserve_or_claim(
        self,
        *,
        alias_family: str,
        cooldown_keys: frozenset[str],
        identity_hash: str = "",
    ) -> ReserveOrClaimResult:
        """Atomically check for existing intents and claim if none exist.

        Closes the lookup/create race: under the threading lock, checks all
        ``cooldown_keys`` for active (not done) intents.  If any key has an
        active intent, returns that intent as a follower (``is_leader=False``).
        Otherwise creates a new intent, registers it for all keys, and returns
        it as the leader (``is_leader=True``).

        Overlapping multi-identity reservations: keys that already have a
        *completed* intent are simply overwritten.  Keys with *active* intents
        cause a follower return, preventing overwrite of a concurrent leader.
        Thread-safe.
        """
        canonical = validate_alias_family(alias_family)
        with self._lock:
            for key in cooldown_keys:
                existing = self._intents.get((canonical, key))
                if existing is not None and not existing.done.is_set():
                    return ReserveOrClaimResult(intent=existing, is_leader=False)
            intent = PublicationIntent(
                transaction_id=_new_transaction_id(),
                alias_family=canonical,
                cooldown_keys=cooldown_keys,
                identity_hash=identity_hash,
            )
            for key in cooldown_keys:
                self._intents[(canonical, key)] = intent
            if identity_hash:
                self._identity_intents[(canonical, identity_hash)] = intent
            return ReserveOrClaimResult(intent=intent, is_leader=True)

    def claim_publication_or_wait(
        self,
        *,
        alias_family: str,
        cooldown_keys: frozenset[str],
        identity_hash: str = "",
    ) -> ClaimPublicationResult:
        """Atomically check clear reservations AND active intents, then claim.

        Under the registry threading lock this performs THREE checks in one
        critical section, closing the Defect-1 race where a clear reservation
        could be created between the intent claim and the separate
        ``get_clear_reservation`` check:

        1. If any ``cooldown_key`` has an ACTIVE clear reservation, return
           ``BLOCKED_BY_CLEAR`` with that reservation (caller awaits its
           ``done`` event, then reselects -- no provider I/O).
        2. If any ``cooldown_key`` has an active (not-done) publication
           intent, return ``FOLLOWER`` with that intent (caller awaits
           ``done``, then reselects).
        3. Otherwise, create a new leader intent registered for all keys
           and return ``LEADER``.

        Thread-safe.  The caller must hold the probe lock for the primary
        cooldown_key before calling this (single-flight serialization).
        """
        canonical = validate_alias_family(alias_family)
        with self._lock:
            # 1. Check clear reservations first (higher priority block).
            for key in cooldown_keys:
                res = self._clear_reservations.get((canonical, key))
                if res is not None and res.status is ClearReservationStatus.ACTIVE:
                    return ClaimPublicationResult(
                        outcome=ClaimOutcome.BLOCKED_BY_CLEAR,
                        clear_reservation=res,
                    )
            # 1b. Identity-scoped clear reservation (Finding 2): blocks
            #     first-ever publication before any cooldown_key is known.
            if identity_hash:
                res = self._identity_reservations.get((canonical, identity_hash))
                if res is not None and res.status is ClearReservationStatus.ACTIVE:
                    return ClaimPublicationResult(
                        outcome=ClaimOutcome.BLOCKED_BY_CLEAR,
                        clear_reservation=res,
                    )
            # 2. Check active publication intents.
            for key in cooldown_keys:
                existing = self._intents.get((canonical, key))
                if existing is not None and not existing.done.is_set():
                    return ClaimPublicationResult(
                        outcome=ClaimOutcome.FOLLOWER,
                        intent=existing,
                    )
            # 3. Claim leader.
            intent = PublicationIntent(
                transaction_id=_new_transaction_id(),
                alias_family=canonical,
                cooldown_keys=cooldown_keys,
                identity_hash=identity_hash,
            )
            for key in cooldown_keys:
                self._intents[(canonical, key)] = intent
            if identity_hash:
                self._identity_intents[(canonical, identity_hash)] = intent
            return ClaimPublicationResult(
                outcome=ClaimOutcome.LEADER,
                intent=intent,
            )

    def release_claim(self, intent: PublicationIntent) -> None:
        """Complete and remove a leader intent that will not probe.

        Convenience wrapper for the blocked/skip paths: signals followers
        and removes the intent from the registry in one call.
        """
        if not intent.done.is_set():
            intent.complete()
        self.remove(intent)

    def get(self, alias_family: str, cooldown_key: str) -> Optional[PublicationIntent]:
        canonical = validate_alias_family(alias_family)
        with self._lock:
            return self._intents.get((canonical, cooldown_key))

    def scan_active_intents_by_identity(
        self,
        alias_family: str,
        identity_hashes: frozenset[str],
    ) -> list[PublicationIntent]:
        """Scan active intents matching any of the given identity hashes.

        Returns deduplicated snapshots of active (not done) intents whose
        ``identity_hash`` matches any hash in ``identity_hashes`` for the
        canonical family.  Thread-safe.  Does NOT mutate registry state.

        Finding 1: enables the clear endpoint to drain already-leading
        unindexed publications that may not yet have cooldown_keys
        registered in the key-based index.  Uses the identity-keyed index
        so intents with empty cooldown_keys are still discoverable.
        """
        canonical = validate_alias_family(alias_family)
        if not identity_hashes:
            return []
        with self._lock:
            seen_ids: set[int] = set()
            results: list[PublicationIntent] = []
            # Scan identity-keyed index (catches empty-cooldown-key intents).
            for id_hash in identity_hashes:
                intent = self._identity_intents.get((canonical, id_hash))
                if intent is None or intent.done.is_set():
                    continue
                if id(intent) in seen_ids:
                    continue
                seen_ids.add(id(intent))
                results.append(intent)
            # Also scan key-based index for intents with identity hashes.
            for (fam, _key), intent in self._intents.items():
                if fam != canonical:
                    continue
                if intent.done.is_set():
                    continue
                if intent.identity_hash not in identity_hashes:
                    continue
                if id(intent) in seen_ids:
                    continue
                seen_ids.add(id(intent))
                results.append(intent)
            return results


    def remove(self, intent: PublicationIntent) -> None:
        with self._lock:
            for key in intent.cooldown_keys:
                composite = (intent.alias_family, key)
                if self._intents.get(composite) is intent:
                    self._intents.pop(composite, None)
            if intent.identity_hash:
                id_composite = (intent.alias_family, intent.identity_hash)
                if self._identity_intents.get(id_composite) is intent:
                    self._identity_intents.pop(id_composite, None)

    # ------------------------------------------------------------------
    # Clear reservations (CFG-004 Wave A)
    # ------------------------------------------------------------------

    def create_clear_reservation(
        self,
        *,
        alias_family: str,
        identity_hashes: frozenset[str],
        cooldown_keys: frozenset[str],
    ) -> ClearReservation:
        """Atomically create a clear reservation with transitive coalescing.

        Blocks first-ever lane publication for all cooldown_keys until
        the reservation is completed.

        Transitive all-or-none coalescing: if the new keys overlap TWO or
        more existing ACTIVE reservations (bridge topology), ALL overlapping
        reservations are merged into a single survivor.  Non-survivor
        reservations are completed (their ``done`` events fire) so their
        waiters wake and reselect, converging on the survivor.  This prevents
        split objects, orphaned waiters, and deadlocks.

        Thread-safe.
        """
        canonical = validate_alias_family(alias_family)
        with self._lock:
            # Find ALL distinct ACTIVE reservations overlapping any of our keys.
            overlapping: list[ClearReservation] = []
            seen_ids: set[int] = set()
            for key in cooldown_keys:
                res = self._clear_reservations.get((canonical, key))
                if res is not None and res.status is ClearReservationStatus.ACTIVE:
                    if id(res) not in seen_ids:
                        seen_ids.add(id(res))
                        overlapping.append(res)

            if overlapping:
                # Deterministic survivor: lowest reservation_id.
                survivor = min(overlapping, key=lambda r: r.reservation_id)
                merged_identities: frozenset[str] = identity_hashes
                merged_keys: set[str] = set(cooldown_keys)
                for res in overlapping:
                    merged_identities = merged_identities | res.identity_hashes
                    merged_keys = merged_keys | set(res.cooldown_keys)

                object.__setattr__(survivor, "identity_hashes", merged_identities)
                object.__setattr__(survivor, "cooldown_keys", frozenset(merged_keys))

                # Point ALL merged keys to the survivor.
                for key in merged_keys:
                    self._clear_reservations[(canonical, key)] = survivor
                for id_hash in merged_identities:
                    self._identity_reservations[(canonical, id_hash)] = survivor

                # Complete non-survivors so their waiters wake and reselect.
                for res in overlapping:
                    if res is not survivor:
                        res.complete()

                return survivor

            # No overlap: create fresh reservation.
            reservation = ClearReservation(
                reservation_id=_new_transaction_id(),
                alias_family=canonical,
                identity_hashes=identity_hashes,
                cooldown_keys=cooldown_keys,
            )
            for key in cooldown_keys:
                self._clear_reservations[(canonical, key)] = reservation
            for id_hash in identity_hashes:
                self._identity_reservations[(canonical, id_hash)] = reservation
            return reservation

    def extend_clear_reservation(
        self,
        reservation: ClearReservation,
        *,
        cooldown_keys: frozenset[str],
        identity_hashes: frozenset[str] = frozenset(),
    ) -> ClearReservation:
        """Extend an active reservation with newly discovered cooldown keys.

        Called after durable hydration discovers lane keys that were not
        known when the identity-scoped reservation was first created
        (Finding 2).  Thread-safe.
        """
        canonical = validate_alias_family(reservation.alias_family)
        with self._lock:
            if reservation.status is not ClearReservationStatus.ACTIVE:
                return reservation
            merged_keys = set(reservation.cooldown_keys) | set(cooldown_keys)
            merged_identities = set(reservation.identity_hashes) | set(
                identity_hashes
            )
            object.__setattr__(
                reservation, "cooldown_keys", frozenset(merged_keys)
            )
            object.__setattr__(
                reservation, "identity_hashes", frozenset(merged_identities)
            )
            for key in cooldown_keys:
                self._clear_reservations[(canonical, key)] = reservation
            for id_hash in identity_hashes:
                self._identity_reservations[(canonical, id_hash)] = reservation
            return reservation

    def get_clear_reservation_by_identity(
        self, alias_family: str, identity_hash: str
    ) -> Optional[ClearReservation]:
        """Return the ACTIVE clear reservation for an identity hash, or None."""
        canonical = validate_alias_family(alias_family)
        with self._lock:
            res = self._identity_reservations.get((canonical, identity_hash))
            if res is not None and res.status is ClearReservationStatus.ACTIVE:
                return res
            return None

    def get_clear_reservation(
        self, alias_family: str, cooldown_key: str
    ) -> Optional[ClearReservation]:
        """Return the ACTIVE clear reservation for a key, or None."""
        canonical = validate_alias_family(alias_family)
        with self._lock:
            res = self._clear_reservations.get((canonical, cooldown_key))
            if res is not None and res.status is ClearReservationStatus.ACTIVE:
                return res
            return None

    def complete_clear_reservation(self, reservation: ClearReservation) -> None:
        """Mark reservation completed and remove its key mappings."""
        reservation.complete()
        with self._lock:
            for key in reservation.cooldown_keys:
                composite = (reservation.alias_family, key)
                if self._clear_reservations.get(composite) is reservation:
                    self._clear_reservations.pop(composite, None)
            for id_hash in reservation.identity_hashes:
                composite = (reservation.alias_family, id_hash)
                if self._identity_reservations.get(composite) is reservation:
                    self._identity_reservations.pop(composite, None)

    def clear(self) -> None:
        with self._lock:
            self._intents.clear()
            self._identity_intents.clear()
            self._clear_reservations.clear()
            self._identity_reservations.clear()


class LaneIdentityIndex:
    """Bounded reverse index from opaque identity hash to lane keys (CFG-004).

    Maps credential-derived identity hashes to the set of cooldown/lane keys
    that reference them, enabling targeted cleanup without exposing raw
    credentials or hashes to external callers.  All access is through internal
    methods; raw identity hashes and lane keys are never leaked outside the
    index boundary.

    Bounded by max_identities and max_lanes_per_identity.  Capacity
    violations REJECT the registration (no silent eviction).
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
        """Associate lane_key with identity_hash (no eviction).

        Returns True if a new mapping was added, False if the lane was
        already present (no-op) or capacity is exhausted (reject).
        Thread-safe.
        """
        with self._lock:
            lanes = self._index.get(identity_hash)
            if lanes is None:
                if len(self._index) >= self._max_identities:
                    return False  # reject: identity capacity full
                lanes = set()
                self._index[identity_hash] = lanes
            if lane_key in lanes:
                return False
            if len(lanes) >= self._max_lanes_per_identity:
                return False  # reject: lane capacity full
            lanes.add(lane_key)
            return True

    def preflight_capacity(
        self,
        *,
        identity_hash: str,
        lane_keys: Sequence[str],
    ) -> bool:
        """Return True if registering lane_keys under identity_hash
        would fit within capacity.  Does NOT mutate.  Thread-safe."""
        with self._lock:
            lanes = self._index.get(identity_hash)
            if lanes is None:
                if len(self._index) >= self._max_identities:
                    return False
                new_count = len(set(lane_keys))
                return new_count <= self._max_lanes_per_identity
            new_keys = set(lane_keys) - lanes
            return (len(lanes) + len(new_keys)) <= self._max_lanes_per_identity

    def register_batch(
        self,
        *,
        identity_hash: str,
        lane_keys: Sequence[str],
    ) -> RegisterBatchOutcome:
        """Atomically register multiple lane_keys under identity_hash.

        Preflights capacity; rejects the ENTIRE batch if any key would
        exceed capacity (no partial mutation, no eviction).  Returns
        ADDED if at least one new mapping was created, IDEMPOTENT if all
        keys were already present (repeated publication is safe), or
        CAPACITY_REJECTED if capacity is exhausted.  Thread-safe.
        """
        with self._lock:
            lanes = self._index.get(identity_hash)
            if lanes is None:
                if len(self._index) >= self._max_identities:
                    return RegisterBatchOutcome.CAPACITY_REJECTED
                unique = set(lane_keys)
                if len(unique) > self._max_lanes_per_identity:
                    return RegisterBatchOutcome.CAPACITY_REJECTED
                self._index[identity_hash] = unique
                return RegisterBatchOutcome.ADDED
            new_keys = set(lane_keys) - lanes
            if not new_keys:
                return RegisterBatchOutcome.IDEMPOTENT
            if (len(lanes) + len(new_keys)) > self._max_lanes_per_identity:
                return RegisterBatchOutcome.CAPACITY_REJECTED
            lanes.update(new_keys)
            return RegisterBatchOutcome.ADDED

    def unregister_batch(
        self,
        *,
        identity_hash: str,
        lane_keys: Sequence[str],
    ) -> int:
        """Remove multiple lane_keys from an identity.  Returns count removed.
        Thread-safe."""
        with self._lock:
            lanes = self._index.get(identity_hash)
            if lanes is None:
                return 0
            removed = 0
            for key in lane_keys:
                if key in lanes:
                    lanes.discard(key)
                    removed += 1
            if not lanes:
                self._index.pop(identity_hash, None)
            return removed

    def restore_membership(
        self,
        *,
        identity_hash: str,
        lane_keys: frozenset[str],
    ) -> bool:
        """Atomically replace the lane set for identity_hash (rollback helper).

        Used by the CFG-004 clear rollback path to restore an exact captured
        preimage after a local/postcondition failure.  An empty ``lane_keys``
        removes the identity entry entirely (matching the state produced when
        ``unregister_batch`` drains the last lane).  Returns True when the
        stored membership differs from the requested preimage (i.e. a change
        was applied).  Thread-safe.
        """
        with self._lock:
            if not lane_keys:
                return self._index.pop(identity_hash, None) is not None
            existing = self._index.get(identity_hash)
            changed = existing != lane_keys
            self._index[identity_hash] = set(lane_keys)
            return changed

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


@dataclass
class CooldownInspectionResult:
    """Result of local cooldown inspection/absence verification (CFG-004 Wave A)."""

    alias_family: str
    cooldown_key: str
    exists: bool
    remaining_seconds: float
    generation: int
    negative_cached: bool = False
    evidence_present: bool = False
    read_pilot_present: bool = False


def inspect_cooldown_absence(
    mgr: "AliasRoutingStateManager",
    *,
    alias_family: str,
    cooldown_key: str,
) -> CooldownInspectionResult:
    """Verify local cooldown absence for a key (lock-free peek, no mutation).

    Checks ALL cooldown-derived process state: positive cooldown, negative
    cache, evidence events, and read-pilot gate (codex only).  Returns the
    current generation so callers can detect concurrent clears.  Does NOT
    acquire the family lock (safe under probe lock).
    """
    family = mgr.family(alias_family)
    remaining = family.peek_cooldown_remaining(cooldown_key)
    negative_cached = family.is_negative_cached(cooldown_key)
    evidence_present = cooldown_key in family.evidence_events_by_key
    # Read-pilot gate is codex-owned only.
    canonical = canonicalize_alias_family(alias_family)
    read_pilot_present = False
    if canonical == "codex":
        # Finding 4: read-pilot evidence can exist in the gate's family
        # evidence map BEFORE any _key_state entry is created (marker-tier
        # evidence accumulates before a cooldown decision).  Both maps must
        # be inspected so classification-marker evidence cannot survive a
        # clear or be misclassified as absent.
        read_pilot_present = (
            cooldown_key in mgr.read_pilot_gate._key_state
            or cooldown_key
            in mgr.read_pilot_gate._family_state.evidence_events_by_key
        )
    return CooldownInspectionResult(
        alias_family=canonical,
        cooldown_key=cooldown_key,
        exists=remaining > 0 or negative_cached or evidence_present or read_pilot_present,
        remaining_seconds=remaining,
        generation=family.get_generation(cooldown_key),
        negative_cached=negative_cached,
        evidence_present=evidence_present,
        read_pilot_present=read_pilot_present,
    )


class AliasRoutingStateManager:
    """Single owner of alias-routing process-local maps + locks (RR-054 #1)."""

    def __init__(self, *, max_size: int = DEFAULT_MEMORY_STATE_MAX_SIZE) -> None:
        from .classification import CooldownEvidenceGate  # lazy: avoid state->classification->state cycle
        self.max_size = max_size
        self.codex = AliasFamilyState()
        self.anthropic = AliasFamilyState()
        self.lane_identity_index = LaneIdentityIndex()
        self.publication_intents = PublicationIntentRegistry()
        self.lane_state_cache_lock = asyncio.Lock()
        self.log_until_monotonic_by_key: dict[str, float] = {}
        self.candidate_probe_locks: dict[str, asyncio.Lock] = {}
        self.candidate_probe_locks_guard = asyncio.Lock()
        # CFG-004 Defect 3: per-key read/clear barrier locks.  A durable read
        # acquires the barrier lock for its key BEFORE capturing the generation
        # and holds it through hydration.  A clear acquires the same barrier
        # lock for each key BEFORE bumping the generation.  This guarantees
        # that a read which started before a clear cannot capture the new
        # generation and hydrate the old durable value, while unrelated keys
        # remain fully concurrent (each key has its own barrier lock).
        self._key_barrier_locks: dict[str, asyncio.Lock] = {}
        self._key_barrier_locks_guard = asyncio.Lock()
        self.openrouter_rate_limit = MonotonicCooldownMap()
        self.openrouter_failure_circuit = MonotonicCooldownMap()
        # Wave 5B: read-pilot evidence gate with its own separate AliasFamilyState
        self.read_pilot_gate = CooldownEvidenceGate(family_state=AliasFamilyState())
        # Wave 5B: per-alias round-robin rotation cursor
        self.round_robin_cursor: dict[tuple[str, str], int] = {}
        # Wave 5B: OpenRouter free-daily-quota cache (immutable tuple) + lock
        self._openrouter_free_quota_cache: Tuple[Optional[float], float] = (None, 0.0)
        self.openrouter_free_quota_lock = asyncio.Lock()

    async def key_barrier_lock(self, cooldown_key: str) -> asyncio.Lock:
        """Return the per-key barrier lock for read/clear serialization (Defect 3)."""
        async with self._key_barrier_locks_guard:
            lock = self._key_barrier_locks.get(cooldown_key)
            if lock is None:
                lock = asyncio.Lock()
                self._key_barrier_locks[cooldown_key] = lock
                bound_memory_map(self._key_barrier_locks, max_size=self.max_size)
            return lock

    @staticmethod
    def _resolve_family_name(alias_family: str) -> str:
        """Canonical family resolution via the shared exact mapping."""
        return canonicalize_alias_family(alias_family)

    def family(self, alias_family: str) -> AliasFamilyState:
        resolved = self._resolve_family_name(alias_family)
        return self.anthropic if resolved == "anthropic" else self.codex

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

        Accepts production family labels (``codex_auto_agent``,
        ``anthropic_auto_agent``) via canonicalization (Defect 4).
        Raises ``ValueError`` for labels not in the canonical map.
        """
        normalized = validate_alias_family(alias_family)
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
        self.publication_intents.clear()
        self._key_barrier_locks.clear()
        self._openrouter_free_quota_cache = (None, 0.0)

    async def candidate_probe_lock(
        self,
        *,
        alias_family: str,
        cooldown_key: str,
    ) -> asyncio.Lock:
        """Return one bounded process-local single-flight lock per candidate lane.

        The lock key uses the CANONICAL family name so production labels
        (``codex_auto_agent``, ``anthropic_auto_agent``) and bare labels
        (``codex``, ``anthropic``) share the same lock identity (Defect 4).
        """
        canonical = validate_alias_family(alias_family)
        key = f"{canonical}:{cooldown_key}"
        async with self.candidate_probe_locks_guard:
            lock = self.candidate_probe_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self.candidate_probe_locks[key] = lock
                bound_memory_map(self.candidate_probe_locks, max_size=self.max_size)
            return lock


# Process-wide singleton used by llm_passthrough_endpoints re-exports.
alias_routing_state = AliasRoutingStateManager()
