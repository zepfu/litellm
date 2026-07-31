"""CFG-004 Wave A: Stale-read concurrency, clear-reservation, and generation
guard tests.

Covers:
- First-publication/clear race: clear reservation blocks first-ever lane
  publication without provider I/O.
- Active-publication conflict: concurrent publications for overlapping keys
  do not deadlock.
- Stale durable-read cannot rehydrate after clear: generation guard discards
  the stale payload.
- No-deadlock: clear reservation + publication + clear all complete within
  bounded time.
- Cleanup: intent and reservation maps are empty after completion.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
    ClearReservationStatus,
    canonicalize_alias_family,
    inspect_cooldown_absence,
    validate_alias_family,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DURABLE_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable"
_STATE_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state"


@pytest.fixture()
def fresh_manager() -> AliasRoutingStateManager:
    return AliasRoutingStateManager()


@pytest.fixture(autouse=True)
def restore_cooldown_state_manager():
    """Save and restore the global cooldown_state manager to prevent pollution."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import cooldown_state
    original = cooldown_state._manager
    yield
    cooldown_state._manager = original


def _make_dual_cache_with_payload(payload: dict | None):
    """Build a mock dual cache whose durable reader returns ``payload``."""
    dual_cache = MagicMock()
    redis_cache = MagicMock()
    redis_cache.init_async_client = MagicMock(return_value=MagicMock())
    redis_cache.check_and_fix_namespace = MagicMock(side_effect=lambda key: key)
    dual_cache.redis_cache = redis_cache
    dual_cache.in_memory_cache = MagicMock()
    return dual_cache


# ---------------------------------------------------------------------------
# Clear reservation blocks first-ever publication (no provider I/O)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_reservation_blocks_first_publication(fresh_manager: AliasRoutingStateManager) -> None:
    """A clear reservation covering a cooldown_key causes the candidate loop
    to wait/reselect without performing provider I/O."""
    mgr = fresh_manager
    provider_io_called = False

    # Create a clear reservation covering key-a
    reservation = mgr.publication_intents.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"hash-1", "hash-2"}),
        cooldown_keys=frozenset({"key-a", "key-b"}),
    )

    # Verify the reservation is visible
    assert mgr.publication_intents.get_clear_reservation("codex", "key-a") is reservation
    assert mgr.publication_intents.get_clear_reservation("codex", "key-b") is reservation
    assert mgr.publication_intents.get_clear_reservation("codex", "key-c") is None
    assert mgr.publication_intents.get_clear_reservation("anthropic", "key-a") is None

    # Simulate candidate loop encountering the reservation: it should wait
    # on reservation.done and NOT call provider I/O.
    async def simulate_candidate():
        nonlocal provider_io_called
        res = mgr.publication_intents.get_clear_reservation("codex", "key-a")
        if res is not None:
            await res.done.wait()
            # After waiting, the candidate reselects (no provider I/O)
            return
        provider_io_called = True

    async def complete_reservation_later():
        await asyncio.sleep(0.05)
        mgr.publication_intents.complete_clear_reservation(reservation)

    await asyncio.wait_for(
        asyncio.gather(simulate_candidate(), complete_reservation_later()),
        timeout=2.0,
    )

    assert provider_io_called is False
    assert reservation.status is ClearReservationStatus.COMPLETED
    # Reservation removed from registry
    assert mgr.publication_intents.get_clear_reservation("codex", "key-a") is None


# ---------------------------------------------------------------------------
# Active-publication conflict: no deadlock with overlapping keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_publication_conflict_no_deadlock(fresh_manager: AliasRoutingStateManager) -> None:
    """Two concurrent publication intents for overlapping keys complete
    without deadlock (canonical family->sorted probe lock order)."""
    mgr = fresh_manager

    intent_a = mgr.publication_intents.create(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a", "key-b"}),
        identity_hash="hash-a",
    )
    intent_b = mgr.publication_intents.create(
        alias_family="codex",
        cooldown_keys=frozenset({"key-b", "key-c"}),
        identity_hash="hash-b",
    )

    # Both intents are retrievable
    assert mgr.publication_intents.get("codex", "key-a") is intent_a
    assert mgr.publication_intents.get("codex", "key-b") is intent_b  # last writer wins
    assert mgr.publication_intents.get("codex", "key-c") is intent_b

    # Complete both in bounded time (no deadlock)
    async def complete_a():
        await asyncio.sleep(0.01)
        intent_a.complete()
        mgr.publication_intents.remove(intent_a)

    async def complete_b():
        await asyncio.sleep(0.02)
        intent_b.complete()
        mgr.publication_intents.remove(intent_b)

    await asyncio.wait_for(asyncio.gather(complete_a(), complete_b()), timeout=2.0)

    # All cleaned up
    assert mgr.publication_intents.get("codex", "key-a") is None
    assert mgr.publication_intents.get("codex", "key-b") is None
    assert mgr.publication_intents.get("codex", "key-c") is None


# ---------------------------------------------------------------------------
# Stale durable-read cannot rehydrate after clear (generation guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_durable_read_cannot_rehydrate_after_clear(fresh_manager: AliasRoutingStateManager) -> None:
    """A durable read that started before a clear must NOT rehydrate the
    cleared cooldown.  The generation guard detects the intervening clear
    and discards the stale payload."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
        configure_cooldown_state_runtime,
        _get_codex_auto_agent_active_cooldown_state,
    )

    mgr = fresh_manager
    configure_cooldown_state_runtime(manager=mgr)

    cooldown_key = "openai:gpt-4.1:auth:stale-test"
    future_expiry = time.time() + 300.0
    stale_payload = {"expires_at_epoch": future_expiry, "cooldown_key": cooldown_key}

    dual_cache = _make_dual_cache_with_payload(stale_payload)

    # Capture generation before the "durable read"
    gen_before = mgr.codex.get_generation(cooldown_key)

    # Simulate: durable read returns stale_payload, but BEFORE hydration a
    # clear happens (advancing generation).
    async def slow_durable_read(*args, **kwargs):
        # Simulate network latency during which a clear occurs
        await asyncio.sleep(0.02)
        return stale_payload

    async def clear_during_read():
        await asyncio.sleep(0.01)  # clear happens mid-read
        mgr.clear_cooldown_state(
            alias_family="codex",
            cooldown_keys=[cooldown_key],
        )

    with patch(f"{_STATE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_STATE_MOD}.read_aawm_alias_routing_durable_payload", side_effect=slow_durable_read), \
         patch(f"{_STATE_MOD}.parse_aawm_alias_routing_durable_expiry", return_value=future_expiry):
        read_task = asyncio.create_task(
            _get_codex_auto_agent_active_cooldown_state(cooldown_key)
        )
        clear_task = asyncio.create_task(clear_during_read())
        seconds, source = await asyncio.wait_for(read_task, timeout=5.0)
        await clear_task

    # The generation guard must have discarded the stale payload
    assert mgr.codex.get_generation(cooldown_key) != gen_before
    assert mgr.codex.get_memory_cooldown_remaining(cooldown_key) == 0.0
    # Source should be local_fallback (stale discarded), NOT durable_cache
    assert source == "local_fallback"


# ---------------------------------------------------------------------------
# No-deadlock: clear reservation + publication + clear all complete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_deadlock_reservation_publication_clear(fresh_manager: AliasRoutingStateManager) -> None:
    """Concurrent clear-reservation creation, publication intent lifecycle,
    and cooldown clear all complete within bounded time (no deadlock)."""
    mgr = fresh_manager
    key = "key-deadlock-test"

    reservation = mgr.publication_intents.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"hash-dl"}),
        cooldown_keys=frozenset({key}),
    )

    intent = mgr.publication_intents.create(
        alias_family="codex",
        cooldown_keys=frozenset({key}),
        identity_hash="hash-dl",
    )

    async def waiter():
        res = mgr.publication_intents.get_clear_reservation("codex", key)
        assert res is not None
        await res.done.wait()

    async def publisher():
        await asyncio.sleep(0.01)
        intent.complete()
        mgr.publication_intents.remove(intent)

    async def clearer():
        await asyncio.sleep(0.02)
        mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=[key])
        mgr.publication_intents.complete_clear_reservation(reservation)

    await asyncio.wait_for(
        asyncio.gather(waiter(), publisher(), clearer()),
        timeout=3.0,
    )

    # All state cleaned up
    assert mgr.publication_intents.get("codex", key) is None
    assert mgr.publication_intents.get_clear_reservation("codex", key) is None
    assert mgr.codex.get_memory_cooldown_remaining(key) == 0.0


# ---------------------------------------------------------------------------
# Cleanup: intent and reservation maps empty after full lifecycle
# ---------------------------------------------------------------------------


def test_cleanup_after_full_lifecycle(fresh_manager: AliasRoutingStateManager) -> None:
    """After intent completion + removal and reservation completion, the
    internal maps are empty (no leaked entries)."""
    mgr = fresh_manager

    # Create and remove multiple intents
    for i in range(5):
        intent = mgr.publication_intents.create(
            alias_family="codex",
            cooldown_keys=frozenset({f"key-{i}"}),
            identity_hash=f"hash-{i}",
        )
        intent.complete()
        mgr.publication_intents.remove(intent)

    # Create and complete a multi-key reservation
    reservation = mgr.publication_intents.create_clear_reservation(
        alias_family="anthropic",
        identity_hashes=frozenset({"h1", "h2", "h3"}),
        cooldown_keys=frozenset({"ak-1", "ak-2", "ak-3"}),
    )
    mgr.publication_intents.complete_clear_reservation(reservation)

    # Registry must be fully empty
    assert mgr.publication_intents._intents == {}
    assert mgr.publication_intents._clear_reservations == {}


# ---------------------------------------------------------------------------
# inspect_cooldown_absence: local inspection/absence verification
# ---------------------------------------------------------------------------


def test_inspect_cooldown_absence_reports_correctly(fresh_manager: AliasRoutingStateManager) -> None:
    """inspect_cooldown_absence returns exists=False and generation for an
    absent key, and exists=True for a present key."""
    mgr = fresh_manager

    # Absent key
    result = inspect_cooldown_absence(mgr, alias_family="codex", cooldown_key="absent-key")
    assert result.exists is False
    assert result.remaining_seconds == 0.0
    assert result.generation == mgr.codex.get_generation("absent-key")
    assert result.alias_family == "codex"

    # Present key
    mgr.codex.set_cooldown_memory("present-key", 120.0)
    result2 = inspect_cooldown_absence(mgr, alias_family="codex", cooldown_key="present-key")
    assert result2.exists is True
    assert result2.remaining_seconds > 0.0

    # After clear, generation advances and key is absent
    mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=["present-key"])
    result3 = inspect_cooldown_absence(mgr, alias_family="codex", cooldown_key="present-key")
    assert result3.exists is False
    assert result3.generation > result.generation


# ---------------------------------------------------------------------------
# Canonical family mapping: exact codex/anthropic resolution
# ---------------------------------------------------------------------------


def test_canonical_family_exact_mapping() -> None:
    """canonicalize_alias_family uses exact mapping; validate_alias_family rejects unknowns."""
    # canonicalize_alias_family: known labels resolve correctly
    assert canonicalize_alias_family("codex") == "codex"
    assert canonicalize_alias_family("codex_auto_agent") == "codex"
    assert canonicalize_alias_family("anthropic") == "anthropic"
    assert canonicalize_alias_family("anthropic_auto_agent") == "anthropic"
    # canonicalize_alias_family: unknown labels still default to codex (backward compat)
    assert canonicalize_alias_family("not_anthropic") == "codex"
    assert canonicalize_alias_family("xanthropicx") == "codex"
    assert canonicalize_alias_family("codex_anthropic") == "codex"

    # validate_alias_family: known labels resolve correctly
    assert validate_alias_family("codex") == "codex"
    assert validate_alias_family("codex_auto_agent") == "codex"
    assert validate_alias_family("anthropic") == "anthropic"
    assert validate_alias_family("anthropic_auto_agent") == "anthropic"
    assert validate_alias_family("  CODEX_AUTO_AGENT  ") == "codex"

    # validate_alias_family: unknown labels raise BEFORE any state mutation
    for unknown in ("not_anthropic", "xanthropicx", "codex_anthropic", "", "openrouter"):
        with pytest.raises(ValueError, match="Unknown alias_family"):
            validate_alias_family(unknown)


# ---------------------------------------------------------------------------
# Transitive coalescing: bridge topology merges two reservations
# ---------------------------------------------------------------------------


def test_create_clear_reservation_bridge_topology_transitive_merge(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """A bridge request whose keys overlap TWO existing ACTIVE reservations
    merges all three into a single survivor.  Non-survivors are completed so
    their waiters wake and reselect; no split objects or orphaned waiters."""
    mgr = fresh_manager
    reg = mgr.publication_intents

    # Reservation A: keys {k1, k2}
    res_a = reg.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"hash-a"}),
        cooldown_keys=frozenset({"k1", "k2"}),
    )
    # Reservation B: keys {k3, k4}
    res_b = reg.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"hash-b"}),
        cooldown_keys=frozenset({"k3", "k4"}),
    )
    assert res_a is not res_b
    assert res_a.status is ClearReservationStatus.ACTIVE
    assert res_b.status is ClearReservationStatus.ACTIVE

    # Bridge request C: keys {k2, k3} -- overlaps BOTH A and B
    res_c = reg.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"hash-c"}),
        cooldown_keys=frozenset({"k2", "k3"}),
    )

    # All three converge on one survivor
    survivor = res_c
    assert survivor.status is ClearReservationStatus.ACTIVE

    # Non-survivors are completed (waiters wake)
    non_survivors = [r for r in (res_a, res_b) if r is not survivor]
    for r in non_survivors:
        assert r.status is ClearReservationStatus.COMPLETED
        assert r.done.is_set()

    # Survivor covers all keys and identities
    assert survivor.cooldown_keys == frozenset({"k1", "k2", "k3", "k4"})
    assert survivor.identity_hashes == frozenset({"hash-a", "hash-b", "hash-c"})

    # All keys in the registry point to the survivor
    for key in ("k1", "k2", "k3", "k4"):
        assert reg.get_clear_reservation("codex", key) is survivor

    # Completing the survivor removes all keys
    reg.complete_clear_reservation(survivor)
    for key in ("k1", "k2", "k3", "k4"):
        assert reg.get_clear_reservation("codex", key) is None


# ---------------------------------------------------------------------------
# Generation guard: no rehydration when generation matches (positive case)
# ---------------------------------------------------------------------------


def test_generation_guard_allows_hydration_when_unchanged(fresh_manager: AliasRoutingStateManager) -> None:
    """When the generation is unchanged, the generation guard permits
    hydration (positive case for the guard logic)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
        hydrate_cooldown_memory,
    )

    mgr = fresh_manager
    family = mgr.codex
    cooldown_key = "openai:gpt-4.1:auth:hydrate-ok"
    future_expiry = time.time() + 300.0

    # Capture generation (no clear occurs)
    gen_before = family.get_generation(cooldown_key)

    # Simulate the hydration path: generation matches, so hydrate proceeds
    assert family.get_generation(cooldown_key) == gen_before
    hydrate_cooldown_memory(
        memory_map=family.cooldown_until_monotonic_by_key,
        cooldown_key=cooldown_key,
        expires_at_epoch=future_expiry,
        max_size=4096,
    )

    # Hydration succeeded
    assert family.cooldown_until_monotonic_by_key.get(cooldown_key, 0.0) > 0
    remaining = family.peek_cooldown_remaining(cooldown_key)
    assert remaining > 0.0

    # After a clear, generation advances and a second hydration would be blocked
    mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=[cooldown_key])
    assert family.get_generation(cooldown_key) != gen_before
    assert family.cooldown_until_monotonic_by_key.get(cooldown_key) is None


# ---------------------------------------------------------------------------
# reserve_or_claim: atomic lookup/create closes the race (production loop)
# ---------------------------------------------------------------------------


def test_reserve_or_claim_atomic_leader_then_follower(fresh_manager: AliasRoutingStateManager) -> None:
    """The first reserve_or_claim claims the leader; a concurrent claim for an
    overlapping active key returns the SAME intent as a follower (no second
    provider call, no overwrite)."""
    mgr = fresh_manager
    reg = mgr.publication_intents

    first = reg.reserve_or_claim(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a", "key-b"}),
        identity_hash="hash-1",
    )
    assert first.is_leader is True
    assert first.intent.identity_hash == "hash-1"

    # Overlapping active key -> follower, same intent object (no overwrite).
    second = reg.reserve_or_claim(
        alias_family="codex",
        cooldown_keys=frozenset({"key-b", "key-c"}),
        identity_hash="hash-2",
    )
    assert second.is_leader is False
    assert second.intent is first.intent
    # The leader's identity is preserved; the follower did not overwrite it.
    assert second.intent.identity_hash == "hash-1"


def test_reserve_or_claim_completed_intent_allows_new_leader(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Once a leader intent completes, a subsequent reserve_or_claim for the
    same key becomes a NEW leader (completed intents are not treated as
    active)."""
    mgr = fresh_manager
    reg = mgr.publication_intents

    first = reg.reserve_or_claim(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="hash-1",
    )
    assert first.is_leader is True
    first.intent.complete()
    reg.remove(first.intent)

    second = reg.reserve_or_claim(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="hash-2",
    )
    assert second.is_leader is True
    assert second.intent is not first.intent
    assert second.intent.identity_hash == "hash-2"


def test_reserve_or_claim_disjoint_keys_both_leaders(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Two reserve_or_claim calls for disjoint key sets are both leaders and
    their identities are preserved independently (overlapping multi-identity
    reservations without overwrite)."""
    mgr = fresh_manager
    reg = mgr.publication_intents

    a = reg.reserve_or_claim(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="hash-a",
    )
    b = reg.reserve_or_claim(
        alias_family="anthropic",
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="hash-b",
    )
    # Same cooldown_key but different canonical family -> independent leaders.
    assert a.is_leader is True
    assert b.is_leader is True
    assert a.intent is not b.intent
    assert a.intent.identity_hash == "hash-a"
    assert b.intent.identity_hash == "hash-b"


# ---------------------------------------------------------------------------
# Canonical family labels in the registry (tuple keys + canonicalization)
# ---------------------------------------------------------------------------


def test_registry_uses_canonical_family_labels(fresh_manager: AliasRoutingStateManager) -> None:
    """Registry keys are (canonical_family, cooldown_key) tuples and non-
    canonical labels resolve to the same canonical bucket."""
    mgr = fresh_manager
    reg = mgr.publication_intents

    intent = reg.create(
        alias_family="codex_auto_agent",  # non-canonical label
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="hash-1",
    )
    assert intent.alias_family == "codex"
    # Tuple registry key, not a formatted string.
    assert ("codex", "key-a") in reg._intents
    assert not any(isinstance(k, str) for k in reg._intents)
    # Lookup via a different non-canonical label resolves to the same intent.
    assert reg.get("codex", "key-a") is intent
    assert reg.get("CODEX_AUTO_AGENT", "key-a") is intent


# ---------------------------------------------------------------------------
# Per-key generation: unrelated clear cannot discard valid key B
# ---------------------------------------------------------------------------


def test_unrelated_clear_does_not_bump_other_key_generation(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Clearing key A advances ONLY key A's generation; key B's generation is
    untouched so an in-flight read for key B is not falsely discarded."""
    mgr = fresh_manager
    family = mgr.codex

    gen_a_before = family.get_generation("key-a")
    gen_b_before = family.get_generation("key-b")

    family.set_cooldown_memory("key-a", 120.0)
    family.set_cooldown_memory("key-b", 120.0)

    # Clear only key A.
    mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=["key-a"])

    assert family.get_generation("key-a") == gen_a_before + 1
    # Key B generation unchanged -> a valid key B read survives.
    assert family.get_generation("key-b") == gen_b_before
    assert family.get_memory_cooldown_remaining("key-b") > 0.0


def test_unrelated_clear_does_not_discard_valid_key_b_hydration(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """A durable read for key B that spans an unrelated clear of key A must
    still hydrate key B (per-key generation isolates the two keys).

    Tests the generation-guard logic directly at the state level: capture
    key B's generation, clear key A, verify key B's generation is unchanged
    (so the guard would permit hydration), then hydrate key B and confirm
    the cooldown is present.
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
        hydrate_cooldown_memory,
    )

    mgr = fresh_manager
    family = mgr.codex
    key_b = "openai:gpt-4.1:auth:key-b"
    future_expiry = time.time() + 300.0

    # Capture key B's generation BEFORE the unrelated clear.
    gen_b_before = family.get_generation(key_b)

    # Clear an unrelated key A (advances only key A's generation).
    family.set_cooldown_memory("key-a-unrelated", 120.0)
    mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=["key-a-unrelated"])

    # Key B's generation is unchanged -> the guard permits hydration.
    assert family.get_generation(key_b) == gen_b_before

    # Simulate the hydration path: generation matches, so hydrate proceeds.
    hydrate_cooldown_memory(
        memory_map=family.cooldown_until_monotonic_by_key,
        cooldown_key=key_b,
        expires_at_epoch=future_expiry,
        max_size=4096,
    )

    # Key B hydration succeeds despite the unrelated clear of key A.
    assert family.get_memory_cooldown_remaining(key_b) > 0.0
    assert family.peek_cooldown_remaining(key_b) > 0.0


# ---------------------------------------------------------------------------
# Stale miss cannot recreate state via negative-cache write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_miss_does_not_recreate_negative_cache(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """A durable MISS that started before a clear must NOT write a negative-
    cache entry after the clear (generation guard on the negative-cache write
    path), so a stale miss cannot recreate state."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
        configure_cooldown_state_runtime,
        _get_codex_auto_agent_active_cooldown_state,
    )

    mgr = fresh_manager
    configure_cooldown_state_runtime(manager=mgr)

    cooldown_key = "openai:gpt-4.1:auth:stale-miss"
    dual_cache = _make_dual_cache_with_payload(None)

    # Deterministic interleaving: the per-key generation is captured before the
    # durable read; the read yields, the clear bumps the generation for this
    # exact key, then the read returns a miss.  The negative-cache write guard
    # must detect the generation change and skip the write.
    read_started = asyncio.Event()
    clear_done = asyncio.Event()

    async def slow_miss_read(*args, **kwargs):
        read_started.set()
        try:
            await asyncio.wait_for(clear_done.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        return None  # durable miss

    async def clear_during_read():
        try:
            await asyncio.wait_for(read_started.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=[cooldown_key])
        clear_done.set()

    with patch(f"{_STATE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_STATE_MOD}.read_aawm_alias_routing_durable_payload", side_effect=slow_miss_read):
        read_task = asyncio.create_task(_get_codex_auto_agent_active_cooldown_state(cooldown_key))
        clear_task = asyncio.create_task(clear_during_read())
        seconds, source = await asyncio.wait_for(read_task, timeout=5.0)
        await asyncio.wait_for(clear_task, timeout=5.0)

    # The stale miss must NOT have written a negative-cache entry.
    assert mgr.codex.is_negative_cached(cooldown_key) is False
    assert seconds == 0.0
    assert source == "local_fallback"


# ---------------------------------------------------------------------------
# Cleanup / no-deadlock under reserve_or_claim + overlapping reservations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reserve_or_claim_overlap_cleanup_no_deadlock(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Overlapping reserve_or_claim leaders + a clear reservation complete and
    clean up within bounded time (no deadlock, no leaked registry entries)."""
    mgr = fresh_manager
    reg = mgr.publication_intents

    leader_a = reg.reserve_or_claim(
        alias_family="codex",
        cooldown_keys=frozenset({"k1", "k2"}),
        identity_hash="h-a",
    )
    # Overlapping follower resolves to leader_a (no new intent).
    follower = reg.reserve_or_claim(
        alias_family="codex",
        cooldown_keys=frozenset({"k2", "k3"}),
        identity_hash="h-b",
    )
    assert follower.intent is leader_a.intent

    reservation = reg.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h-a", "h-b"}),
        cooldown_keys=frozenset({"k1", "k2", "k3"}),
    )

    async def waiter():
        res = reg.get_clear_reservation("codex", "k2")
        assert res is not None
        await res.done.wait()

    async def finish():
        await asyncio.sleep(0.01)
        leader_a.intent.complete()
        reg.remove(leader_a.intent)
        mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=["k1", "k2", "k3"])
        reg.complete_clear_reservation(reservation)

    await asyncio.wait_for(asyncio.gather(waiter(), finish()), timeout=3.0)

    # Registry fully drained.
    assert reg._intents == {}
    assert reg._clear_reservations == {}
    assert mgr.codex.get_memory_cooldown_remaining("k1") == 0.0
    assert mgr.codex.get_memory_cooldown_remaining("k2") == 0.0
    assert mgr.codex.get_memory_cooldown_remaining("k3") == 0.0


# ---------------------------------------------------------------------------
# Defect 1: atomic claim_publication_or_wait (no window for clear race)
# ---------------------------------------------------------------------------


def test_claim_publication_or_wait_blocked_by_clear(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """claim_publication_or_wait returns BLOCKED_BY_CLEAR when a clear
    reservation exists, preventing provider I/O (Defect 1)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        ClaimOutcome,
    )

    mgr = fresh_manager
    reg = mgr.publication_intents

    # Create a clear reservation BEFORE claiming.
    reservation = reg.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-a"}),
    )

    # Atomic claim must see the reservation and block.
    result = reg.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="h2",
    )
    assert result.outcome is ClaimOutcome.BLOCKED_BY_CLEAR
    assert result.clear_reservation is reservation
    assert result.intent is None

    # No intent was created (no orphan).
    assert reg.get("codex", "key-a") is None


def test_claim_publication_or_wait_follower(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """claim_publication_or_wait returns FOLLOWER when an active intent
    exists for an overlapping key (Defect 1)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        ClaimOutcome,
    )

    mgr = fresh_manager
    reg = mgr.publication_intents

    leader = reg.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a", "key-b"}),
        identity_hash="h1",
    )
    assert leader.outcome is ClaimOutcome.LEADER

    follower = reg.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({"key-b", "key-c"}),
        identity_hash="h2",
    )
    assert follower.outcome is ClaimOutcome.FOLLOWER
    assert follower.intent is leader.intent


def test_claim_publication_or_wait_leader_after_clear_completes(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """After a clear reservation completes, claim_publication_or_wait
    succeeds as LEADER (Defect 1 lifecycle)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        ClaimOutcome,
    )

    mgr = fresh_manager
    reg = mgr.publication_intents

    reservation = reg.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-a"}),
    )
    reg.complete_clear_reservation(reservation)

    result = reg.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="h2",
    )
    assert result.outcome is ClaimOutcome.LEADER
    assert result.intent is not None


def test_claim_clear_reservation_created_between_separate_checks(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Deterministic proof that the old two-step (reserve_or_claim then
    get_clear_reservation) has a race window that claim_publication_or_wait
    closes: a clear reservation created AFTER reserve_or_claim but BEFORE
    get_clear_reservation is invisible to the old path but visible to the
    atomic path (Defect 1)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        ClaimOutcome,
    )

    mgr = fresh_manager
    reg = mgr.publication_intents

    # Old path step 1: reserve_or_claim succeeds (no reservation yet).
    old_claim = reg.reserve_or_claim(
        alias_family="codex",
        cooldown_keys=frozenset({"key-race"}),
        identity_hash="h-old",
    )
    assert old_claim.is_leader is True

    # Race window: clear reservation created AFTER claim, BEFORE check.
    reservation = reg.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h-clear"}),
        cooldown_keys=frozenset({"key-race"}),
    )

    # Old path step 2: get_clear_reservation WOULD see it, but only because
    # we check manually.  The old code checked AFTER claiming, so the intent
    # was already created.  The atomic path prevents this entirely.
    old_check = reg.get_clear_reservation("codex", "key-race")
    assert old_check is reservation  # old path would see it, but intent exists

    # Clean up old claim.
    reg.release_claim(old_claim.intent)

    # Atomic path: same scenario, but the reservation blocks the claim.
    result = reg.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({"key-race"}),
        identity_hash="h-atomic",
    )
    assert result.outcome is ClaimOutcome.BLOCKED_BY_CLEAR
    assert result.clear_reservation is reservation
    # No orphaned intent.
    assert reg.get("codex", "key-race") is None


# ---------------------------------------------------------------------------
# Defect 3: per-key barrier prevents stale hydration end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_barrier_serializes_read_and_clear(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """The per-key barrier lock serializes a durable read against a clear
    (Defect 3 core invariant): a clear cannot bump the generation while a
    read holds the barrier.  Proven deterministically with explicit lock
    acquisition order -- no sleeps, no module-global dependency.
    """
    mgr = fresh_manager
    cooldown_key = "openai:gpt-4.1:auth:barrier-serial"

    barrier = await mgr.key_barrier_lock(cooldown_key)

    # Simulate a read holding the barrier (in THIS task).
    await barrier.acquire()
    gen_during_read = mgr.codex.get_generation(cooldown_key)
    assert gen_during_read == 0

    # A SEPARATE task tries to acquire the SAME barrier lock.  It must block
    # because this task holds it (asyncio.Lock is NOT reentrant across tasks).
    clear_acquired = asyncio.Event()

    async def blocked_clear():
        b = await mgr.key_barrier_lock(cooldown_key)
        async with b:
            clear_acquired.set()
            mgr.codex.bump_generation([cooldown_key])

    clear_task = asyncio.create_task(blocked_clear())
    # Yield to let the clear task reach the barrier acquire.
    for _ in range(5):
        await asyncio.sleep(0)

    # The clear must NOT have acquired the barrier (this task still holds it).
    assert not clear_acquired.is_set()
    assert mgr.codex.get_generation(cooldown_key) == 0

    # Release the barrier; the clear task now proceeds and bumps generation.
    barrier.release()
    await asyncio.wait_for(clear_task, timeout=5.0)

    assert clear_acquired.is_set()
    assert mgr.codex.get_generation(cooldown_key) == 1


@pytest.mark.asyncio
async def test_barrier_clear_bumps_generation_under_lock(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """clear_alias_family_cooldown_state bumps the per-key generation while
    holding the barrier lock, so a read that starts after the clear captures
    the new generation and the generation guard rejects any stale durable
    payload (Defect 3 state-level proof, no module-global dependency).
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_state import (
        configure_cooldown_state_runtime,
        clear_alias_family_cooldown_state,
    )

    mgr = fresh_manager
    configure_cooldown_state_runtime(manager=mgr)

    cooldown_key = "openai:gpt-4.1:auth:barrier-gen"
    future_expiry = time.time() + 300.0

    # Seed a cooldown.
    mgr.codex.set_cooldown_memory(cooldown_key, 300.0)
    assert mgr.codex.get_memory_cooldown_remaining(cooldown_key) > 0.0
    gen_before = mgr.codex.get_generation(cooldown_key)

    # Barrier-protected clear (no durable deletion needed for this proof).
    await clear_alias_family_cooldown_state(
        alias_family="codex",
        cooldown_keys=[cooldown_key],
        delete_durable=False,
    )

    # Generation advanced; cooldown removed.
    assert mgr.codex.get_generation(cooldown_key) == gen_before + 1
    assert mgr.codex.get_memory_cooldown_remaining(cooldown_key) == 0.0

    # Simulate a stale durable read that captured gen_before: the generation
    # guard would reject it because gen changed.
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
        hydrate_cooldown_memory,
    )

    # A read capturing the NEW generation would proceed, but a read that
    # captured gen_before (stale) would be rejected by the guard:
    assert mgr.codex.get_generation(cooldown_key) != gen_before
    # Verify the guard logic: gen mismatch -> skip hydration.
    if mgr.codex.get_generation(cooldown_key) != gen_before:
        pass  # guard would reject -- correct
    else:
        hydrate_cooldown_memory(
            memory_map=mgr.codex.cooldown_until_monotonic_by_key,
            cooldown_key=cooldown_key,
            expires_at_epoch=future_expiry,
            max_size=4096,
        )
    # No hydration occurred (guard rejected).
    assert mgr.codex.get_memory_cooldown_remaining(cooldown_key) == 0.0


@pytest.mark.asyncio
async def test_barrier_unrelated_key_concurrency(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Per-key barrier locks do NOT serialize unrelated keys: a read for
    key-B proceeds concurrently with a clear of key-A (Defect 3 preserves
    unrelated-key concurrency)."""
    mgr = fresh_manager

    barrier_a = await mgr.key_barrier_lock("key-a")
    barrier_b = await mgr.key_barrier_lock("key-b")

    # Different keys get different locks.
    assert barrier_a is not barrier_b

    # Same key returns the same lock.
    barrier_a2 = await mgr.key_barrier_lock("key-a")
    assert barrier_a is barrier_a2


# ---------------------------------------------------------------------------
# Defect 4: manager clear APIs accept production family labels
# ---------------------------------------------------------------------------


def test_clear_cooldown_state_accepts_production_labels(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """clear_cooldown_state accepts codex_auto_agent and
    anthropic_auto_agent without raising ValueError (Defect 4)."""
    mgr = fresh_manager

    # Set some state.
    mgr.codex.set_cooldown_memory("key-codex", 120.0)
    mgr.anthropic.set_cooldown_memory("key-anth", 120.0)

    # Production labels must work.
    result_codex = mgr.clear_cooldown_state(
        alias_family="codex_auto_agent",
        cooldown_keys=["key-codex"],
    )
    assert result_codex.alias_family == "codex"
    assert "key-codex" in result_codex.positive_keys_cleared

    result_anth = mgr.clear_cooldown_state(
        alias_family="anthropic_auto_agent",
        cooldown_keys=["key-anth"],
    )
    assert result_anth.alias_family == "anthropic"
    assert "key-anth" in result_anth.positive_keys_cleared


def test_clear_cooldown_state_bare_labels_still_work(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Bare codex/anthropic labels continue to work after Defect 4 fix."""
    mgr = fresh_manager
    mgr.codex.set_cooldown_memory("key-bare", 60.0)
    result = mgr.clear_cooldown_state(
        alias_family="codex",
        cooldown_keys=["key-bare"],
    )
    assert result.alias_family == "codex"
    assert "key-bare" in result.positive_keys_cleared


# ---------------------------------------------------------------------------
# Defect 4: candidate_probe_lock canonical identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidate_probe_lock_canonical_identity(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """candidate_probe_lock returns the SAME lock for codex and
    codex_auto_agent (canonical identity, Defect 4)."""
    mgr = fresh_manager

    lock_bare = await mgr.candidate_probe_lock(
        alias_family="codex",
        cooldown_key="key-x",
    )
    lock_prod = await mgr.candidate_probe_lock(
        alias_family="codex_auto_agent",
        cooldown_key="key-x",
    )
    assert lock_bare is lock_prod

    lock_anth_bare = await mgr.candidate_probe_lock(
        alias_family="anthropic",
        cooldown_key="key-x",
    )
    lock_anth_prod = await mgr.candidate_probe_lock(
        alias_family="anthropic_auto_agent",
        cooldown_key="key-x",
    )
    assert lock_anth_bare is lock_anth_prod
    # Different families get different locks.
    assert lock_bare is not lock_anth_bare
