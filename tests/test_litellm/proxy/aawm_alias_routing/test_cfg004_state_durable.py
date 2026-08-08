"""CFG-004 Wave 1: State and durable primitive tests.

Tests targeted cooldown-state removal, durable payload inspection/deletion,
the bounded reverse identity index, and adversarial fail-closed behavior.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
    DurableKeyInspection,
    build_aawm_alias_routing_durable_cache_key,
    delete_aawm_alias_routing_durable_key,
    inspect_aawm_alias_routing_durable_key,
    verify_aawm_alias_routing_durable_absence,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasFamilyState,
    AliasRoutingStateManager,
    CooldownClearResult,
    LaneIdentityIndex,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DURABLE_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable"


@pytest.fixture()
def fresh_manager() -> AliasRoutingStateManager:
    return AliasRoutingStateManager()


def _decode(cached_response):
    """Mirror RedisCache._get_cache_logic for test mocks."""
    if cached_response is None:
        return None
    if isinstance(cached_response, bytes):
        cached_response = cached_response.decode("utf-8")
    return json.loads(cached_response)


def _make_strict_dual_cache(
    *,
    get_return=None,
    get_side_effect=None,
    ttl_return=None,
    ttl_side_effect=None,
    delete_return=1,
    delete_side_effect=None,
    mem_get_return=None,
    mem_get_side_effect=None,
    mem_delete_side_effect=None,
    include_in_memory=True,
    include_mem_get=True,
):
    """Build a mock dual cache whose redis_cache exposes a strict client."""
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=get_return, side_effect=get_side_effect)
    redis_client.ttl = AsyncMock(return_value=ttl_return, side_effect=ttl_side_effect)
    redis_client.delete = AsyncMock(
        return_value=delete_return, side_effect=delete_side_effect
    )
    redis_cache = MagicMock()
    redis_cache.init_async_client = MagicMock(return_value=redis_client)
    redis_cache.check_and_fix_namespace = MagicMock(side_effect=lambda key: key)
    redis_cache._get_cache_logic = MagicMock(side_effect=_decode)
    dual_cache = MagicMock()
    dual_cache.redis_cache = redis_cache
    if include_in_memory:
        in_memory = MagicMock()
        in_memory.delete_cache = MagicMock(side_effect=mem_delete_side_effect)
        if include_mem_get:
            in_memory.get_cache = MagicMock(
                return_value=mem_get_return, side_effect=mem_get_side_effect
            )
        else:
            # Simulate missing get_cache method
            del in_memory.get_cache
        dual_cache.in_memory_cache = in_memory
    else:
        dual_cache.in_memory_cache = None
    return dual_cache, redis_client


# ---------------------------------------------------------------------------
# AliasFamilyState.clear_cooldown_state tests
# ---------------------------------------------------------------------------


def test_clear_cooldown_state_removes_positive_negative_evidence() -> None:
    family = AliasFamilyState()
    family.cooldown_until_monotonic_by_key["key1"] = time.monotonic() + 100
    family.cooldown_negative_until_monotonic_by_key["key1"] = time.monotonic() + 50
    family.evidence_events_by_key["key1"] = [time.monotonic()]

    positive, negative, evidence = family.clear_cooldown_state(cooldown_keys=["key1"])

    assert positive == ["key1"]
    assert negative == ["key1"]
    assert evidence == ["key1"]
    assert "key1" not in family.cooldown_until_monotonic_by_key
    assert "key1" not in family.cooldown_negative_until_monotonic_by_key
    assert "key1" not in family.evidence_events_by_key


def test_clear_cooldown_state_preserves_unrelated_keys() -> None:
    family = AliasFamilyState()
    family.cooldown_until_monotonic_by_key["key1"] = time.monotonic() + 100
    family.cooldown_until_monotonic_by_key["key2"] = time.monotonic() + 200
    family.cooldown_negative_until_monotonic_by_key["key1"] = time.monotonic() + 50
    family.cooldown_negative_until_monotonic_by_key["key2"] = time.monotonic() + 75

    positive, negative, _evidence = family.clear_cooldown_state(cooldown_keys=["key1"])

    assert positive == ["key1"]
    assert negative == ["key1"]
    assert "key1" not in family.cooldown_until_monotonic_by_key
    assert "key2" in family.cooldown_until_monotonic_by_key
    assert "key1" not in family.cooldown_negative_until_monotonic_by_key
    assert "key2" in family.cooldown_negative_until_monotonic_by_key


def test_clear_cooldown_state_preserves_session_affinity() -> None:
    family = AliasFamilyState()
    family.cooldown_until_monotonic_by_key["key1"] = time.monotonic() + 100
    family.session_affinity_by_key["session1"] = {
        "provider": "openai",
        "model": "gpt-4",
        "expires_at_monotonic": time.monotonic() + 1000,
    }

    family.clear_cooldown_state(cooldown_keys=["key1"])

    assert "session1" in family.session_affinity_by_key


def test_clear_cooldown_state_idempotent_absence() -> None:
    family = AliasFamilyState()
    positive, negative, evidence = family.clear_cooldown_state(
        cooldown_keys=["nonexistent"]
    )
    assert positive == []
    assert negative == []
    assert evidence == []


# ---------------------------------------------------------------------------
# AliasRoutingStateManager.clear_cooldown_state tests
# ---------------------------------------------------------------------------


def test_manager_clear_cooldown_state_codex_family(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    mgr = fresh_manager
    mgr.codex.cooldown_until_monotonic_by_key["key1"] = time.monotonic() + 100
    mgr.codex.cooldown_negative_until_monotonic_by_key["key1"] = time.monotonic() + 50
    mgr.codex.evidence_events_by_key["key1"] = [time.monotonic()]

    result = mgr.clear_cooldown_state(
        alias_family="codex",
        canonical_aliases=[],
        cooldown_keys=["key1"],
    )

    assert isinstance(result, CooldownClearResult)
    assert result.alias_family == "codex"
    assert result.positive_keys_cleared == ["key1"]
    assert result.negative_keys_cleared == ["key1"]
    assert result.evidence_keys_cleared == ["key1"]


def test_manager_clear_cooldown_state_anthropic_family(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    mgr = fresh_manager
    mgr.anthropic.cooldown_until_monotonic_by_key["key1"] = time.monotonic() + 100

    result = mgr.clear_cooldown_state(
        alias_family="anthropic",
        canonical_aliases=[],
        cooldown_keys=["key1"],
    )

    assert result.alias_family == "anthropic"
    assert result.positive_keys_cleared == ["key1"]


def test_manager_clear_cooldown_state_clears_explicit_codex_failure_evidence(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Codex evidence clears only for the explicit alias and Codex family."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.classification import (
        _KeyCooldownState,
    )

    mgr = fresh_manager
    canonical_alias = "test-alias"
    gate = mgr.codex_failure_evidence_gate.gate_for_alias(
        canonical_alias=canonical_alias,
        create=True,
    )
    assert gate is not None
    gate._key_state["key1"] = _KeyCooldownState()
    gate._family_state.evidence_events_by_key["key1"] = [time.monotonic()]

    result_codex = mgr.clear_cooldown_state(
        alias_family="codex",
        canonical_aliases=[canonical_alias],
        cooldown_keys=["key1"],
    )
    assert result_codex.codex_failure_evidence_entries_cleared == [
        (canonical_alias, "key1")
    ]
    assert (
        mgr.codex_failure_evidence_gate.gate_for_alias(
            canonical_alias=canonical_alias
        )
        is None
    )

    gate = mgr.codex_failure_evidence_gate.gate_for_alias(
        canonical_alias=canonical_alias,
        create=True,
    )
    assert gate is not None
    gate._key_state["key2"] = _KeyCooldownState()
    gate._family_state.evidence_events_by_key["key2"] = [time.monotonic()]
    result_anthropic = mgr.clear_cooldown_state(
        alias_family="anthropic",
        canonical_aliases=[canonical_alias],
        cooldown_keys=["key2"],
    )
    assert result_anthropic.codex_failure_evidence_entries_cleared == []
    assert "key2" in gate._key_state


def test_manager_clear_cooldown_state_unknown_family_raises(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    with pytest.raises(ValueError, match="Unknown alias_family"):
        fresh_manager.clear_cooldown_state(
            alias_family="openai",
            canonical_aliases=[],
            cooldown_keys=["k"],
        )


def test_manager_clear_cooldown_state_reports_affinity_count(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    mgr = fresh_manager
    mgr.codex.session_affinity_by_key["session1"] = {"provider": "openai"}
    mgr.codex.session_affinity_by_key["session2"] = {"provider": "anthropic"}

    result = mgr.clear_cooldown_state(
        alias_family="codex",
        canonical_aliases=[],
        cooldown_keys=["key1"],
    )

    assert result.affinity_keys_preserved == 2


# ---------------------------------------------------------------------------
# LaneIdentityIndex tests
# ---------------------------------------------------------------------------


def test_lane_identity_index_register_and_lookup() -> None:
    index = LaneIdentityIndex()
    added = index.register(identity_hash="hash1", lane_key="lane1")
    assert added is True
    assert index.lanes_for("hash1") == frozenset({"lane1"})


def test_lane_identity_index_register_duplicate_returns_false() -> None:
    index = LaneIdentityIndex()
    index.register(identity_hash="hash1", lane_key="lane1")
    assert index.register(identity_hash="hash1", lane_key="lane1") is False


def test_lane_identity_index_multiple_lanes_per_identity() -> None:
    index = LaneIdentityIndex()
    index.register(identity_hash="hash1", lane_key="lane1")
    index.register(identity_hash="hash1", lane_key="lane2")
    index.register(identity_hash="hash1", lane_key="lane3")
    assert index.lanes_for("hash1") == frozenset({"lane1", "lane2", "lane3"})


def test_lane_identity_index_unregister_lane() -> None:
    index = LaneIdentityIndex()
    index.register(identity_hash="hash1", lane_key="lane1")
    index.register(identity_hash="hash1", lane_key="lane2")
    assert index.unregister_lane(identity_hash="hash1", lane_key="lane1") is True
    assert index.lanes_for("hash1") == frozenset({"lane2"})


def test_lane_identity_index_unregister_last_lane_removes_identity() -> None:
    index = LaneIdentityIndex()
    index.register(identity_hash="hash1", lane_key="lane1")
    index.unregister_lane(identity_hash="hash1", lane_key="lane1")
    assert len(index) == 0
    assert index.lanes_for("hash1") == frozenset()


def test_lane_identity_index_remove_identity() -> None:
    index = LaneIdentityIndex()
    index.register(identity_hash="hash1", lane_key="lane1")
    index.register(identity_hash="hash1", lane_key="lane2")
    removed = index.remove_identity("hash1")
    assert removed == frozenset({"lane1", "lane2"})
    assert len(index) == 0


def test_lane_identity_index_max_identities_capacity_rejection() -> None:
    """Fail-closed: identity capacity rejects new registrations without eviction."""
    index = LaneIdentityIndex(max_identities=2)
    assert index.register(identity_hash="hash1", lane_key="lane1") is True
    assert index.register(identity_hash="hash2", lane_key="lane2") is True
    # At capacity: reject, no eviction
    assert index.register(identity_hash="hash3", lane_key="lane3") is False
    # All original entries preserved
    assert index.lanes_for("hash1") == frozenset({"lane1"})
    assert index.lanes_for("hash2") == frozenset({"lane2"})
    assert index.lanes_for("hash3") == frozenset()


def test_lane_identity_index_max_lanes_per_identity_capacity_rejection() -> None:
    """Fail-closed: lane capacity rejects new lanes without eviction."""
    index = LaneIdentityIndex(max_lanes_per_identity=2)
    assert index.register(identity_hash="hash1", lane_key="lane1") is True
    assert index.register(identity_hash="hash1", lane_key="lane2") is True
    # At capacity: reject, no eviction
    assert index.register(identity_hash="hash1", lane_key="lane3") is False
    lanes = index.lanes_for("hash1")
    assert lanes == frozenset({"lane1", "lane2"})


def test_lane_identity_index_clear() -> None:
    index = LaneIdentityIndex()
    index.register(identity_hash="hash1", lane_key="lane1")
    index.register(identity_hash="hash2", lane_key="lane2")
    index.clear()
    assert len(index) == 0


def test_lane_identity_index_barrier_forced_same_identity_concurrent() -> None:
    """Barrier-forced same-identity registration must not lose any lane."""
    index = LaneIdentityIndex(max_identities=16, max_lanes_per_identity=1024)
    num_threads = 8
    lanes_per_thread = 50
    barrier = threading.Barrier(num_threads)
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            barrier.wait(timeout=5)
            for i in range(lanes_per_thread):
                index.register(
                    identity_hash="shared-identity",
                    lane_key=f"lane-{thread_id}-{i}",
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    lanes = index.lanes_for("shared-identity")
    expected = {f"lane-{t}-{i}" for t in range(num_threads) for i in range(lanes_per_thread)}
    assert lanes == expected, f"Lost {len(expected) - len(lanes)} lanes: {expected - lanes}"


# ---------------------------------------------------------------------------
# Durable inspection tests (strict / fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_durable_key_exists_with_actual_ttl() -> None:
    """Actual cache TTL=7 must be reported, not derived from payload."""
    payload = {"cooldown_key": "key1", "expires_at_epoch": time.time() + 999}
    raw = json.dumps(payload).encode("utf-8")
    dual_cache, _client = _make_strict_dual_cache(get_return=raw, ttl_return=7)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await inspect_aawm_alias_routing_durable_key(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )

    assert isinstance(result, DurableKeyInspection)
    assert result.exists is True
    assert result.payload == payload
    assert result.ttl_remaining_seconds == 7.0


@pytest.mark.asyncio
async def test_inspect_durable_key_absent() -> None:
    dual_cache, _client = _make_strict_dual_cache(get_return=None)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await inspect_aawm_alias_routing_durable_key(
            alias_family="codex", state_kind="cooldown", state_key="nonexistent",
        )

    assert result.exists is False
    assert result.payload is None


@pytest.mark.asyncio
async def test_inspect_durable_key_no_cache_fails_closed() -> None:
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=None):
        with pytest.raises(RuntimeError, match="no Redis cache available"):
            await inspect_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_inspect_durable_key_swallowed_redis_exception_fails_closed() -> None:
    dual_cache, _client = _make_strict_dual_cache(
        get_side_effect=ConnectionError("Redis down"),
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="redis get failed"):
            await inspect_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_inspect_durable_key_missing_method_fails_closed() -> None:
    redis_client = MagicMock(spec=[])
    redis_cache = MagicMock()
    redis_cache.init_async_client = MagicMock(return_value=redis_client)
    redis_cache.check_and_fix_namespace = MagicMock(side_effect=lambda key: key)
    dual_cache = MagicMock()
    dual_cache.redis_cache = redis_cache

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="missing get"):
            await inspect_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_inspect_durable_key_malformed_value_fails_closed() -> None:
    dual_cache, _client = _make_strict_dual_cache(get_return=b'"just a string"')

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="malformed cached value"):
            await inspect_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_inspect_durable_key_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown alias_family"):
        await inspect_aawm_alias_routing_durable_key(
            alias_family="openai", state_kind="cooldown", state_key="key1",
        )


# ---------------------------------------------------------------------------
# Durable deletion tests (strict / fail-closed / dual-tier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_durable_key_existing_clears_both_tiers() -> None:
    payload = {"cooldown_key": "key1"}
    raw = json.dumps(payload).encode("utf-8")
    dual_cache, client = _make_strict_dual_cache(
        get_side_effect=[raw, None],
        mem_get_return=None,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        deleted = await delete_aawm_alias_routing_durable_key(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )

    assert deleted is True
    client.delete.assert_called_once()
    dual_cache.in_memory_cache.delete_cache.assert_called_once()


@pytest.mark.asyncio
async def test_delete_durable_key_write_populate_memory_then_reread_none() -> None:
    """Simulate write -> populate memory -> delete -> both tiers absent."""
    payload = {"cooldown_key": "key1", "expires_at_epoch": time.time() + 300}
    raw = json.dumps(payload).encode("utf-8")
    dual_cache, client = _make_strict_dual_cache(
        get_side_effect=[raw, None],
        mem_get_return=None,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        deleted = await delete_aawm_alias_routing_durable_key(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )

    assert deleted is True
    dual_cache.in_memory_cache.delete_cache.assert_called_once()
    dual_cache.in_memory_cache.get_cache.assert_called_once()


@pytest.mark.asyncio
async def test_delete_durable_key_absent() -> None:
    dual_cache, _client = _make_strict_dual_cache(
        get_return=None,
        mem_get_return=None,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        deleted = await delete_aawm_alias_routing_durable_key(
            alias_family="codex", state_kind="cooldown", state_key="nonexistent",
        )

    assert deleted is False


@pytest.mark.asyncio
async def test_delete_durable_key_no_cache_fails_closed() -> None:
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=None):
        with pytest.raises(RuntimeError, match="no Redis cache available"):
            await delete_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_delete_durable_key_redis_error_fails_closed() -> None:
    dual_cache, _client = _make_strict_dual_cache(
        get_side_effect=TimeoutError("timeout"),
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="redis get failed"):
            await delete_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_delete_durable_key_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown alias_family"):
        await delete_aawm_alias_routing_durable_key(
            alias_family="gemini", state_kind="cooldown", state_key="key1",
        )


@pytest.mark.asyncio
async def test_delete_durable_key_memory_tier_failure_raises() -> None:
    payload = {"cooldown_key": "key1"}
    raw = json.dumps(payload).encode("utf-8")
    dual_cache, _client = _make_strict_dual_cache(
        get_side_effect=[raw, None],
        mem_delete_side_effect=OSError("memory tier locked"),
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="in-memory delete failed"):
            await delete_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_delete_durable_key_redis_still_present_after_delete_raises() -> None:
    payload = {"cooldown_key": "key1"}
    raw = json.dumps(payload).encode("utf-8")
    dual_cache, _client = _make_strict_dual_cache(
        get_side_effect=[raw, raw],
        mem_get_return=None,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="still present in Redis"):
            await delete_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_delete_durable_key_memory_still_present_after_delete_raises() -> None:
    """If in-memory tier still has data after delete, must raise."""
    payload = {"cooldown_key": "key1"}
    raw = json.dumps(payload).encode("utf-8")
    dual_cache, _client = _make_strict_dual_cache(
        get_side_effect=[raw, None],
        mem_get_return=payload,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="still present in in-memory tier"):
            await delete_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_delete_durable_key_mem_get_oserror_raises() -> None:
    """In-memory get_cache raising OSError during post-delete verify -> RuntimeError."""
    payload = {"cooldown_key": "key1"}
    raw = json.dumps(payload).encode("utf-8")
    dual_cache, _client = _make_strict_dual_cache(
        get_side_effect=[raw, None],
        mem_get_side_effect=OSError("disk error"),
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="in-memory get failed"):
            await delete_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_delete_durable_key_missing_mem_get_method_raises() -> None:
    """Missing get_cache on in_memory_cache during post-delete verify -> RuntimeError."""
    payload = {"cooldown_key": "key1"}
    raw = json.dumps(payload).encode("utf-8")
    dual_cache, _client = _make_strict_dual_cache(
        get_side_effect=[raw, None],
        include_mem_get=False,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="missing get_cache"):
            await delete_aawm_alias_routing_durable_key(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


# ---------------------------------------------------------------------------
# Durable absence verification tests (strict / fail-closed / dual-tier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_durable_absence_confirmed_both_tiers() -> None:
    dual_cache, _client = _make_strict_dual_cache(
        get_return=None,
        mem_get_return=None,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        absent = await verify_aawm_alias_routing_durable_absence(
            alias_family="codex", state_kind="cooldown", state_key="nonexistent",
        )

    assert absent is True


@pytest.mark.asyncio
async def test_verify_durable_absence_redis_present() -> None:
    raw = json.dumps({"cooldown_key": "key1"}).encode("utf-8")
    dual_cache, _client = _make_strict_dual_cache(
        get_return=raw,
        mem_get_return=None,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        absent = await verify_aawm_alias_routing_durable_absence(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )

    assert absent is False


@pytest.mark.asyncio
async def test_verify_durable_absence_redis_empty_memory_populated() -> None:
    """Redis empty but memory tier populated -> verify must return False."""
    dual_cache, _client = _make_strict_dual_cache(
        get_return=None,
        mem_get_return={"cooldown_key": "key1", "stale": True},
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        absent = await verify_aawm_alias_routing_durable_absence(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )

    assert absent is False


@pytest.mark.asyncio
async def test_verify_durable_absence_no_cache_fails_closed() -> None:
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=None):
        with pytest.raises(RuntimeError, match="no Redis cache available"):
            await verify_aawm_alias_routing_durable_absence(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_verify_durable_absence_redis_error_fails_closed() -> None:
    dual_cache, _client = _make_strict_dual_cache(
        get_side_effect=ConnectionError("connection refused"),
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="redis get failed"):
            await verify_aawm_alias_routing_durable_absence(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_verify_durable_absence_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown alias_family"):
        await verify_aawm_alias_routing_durable_absence(
            alias_family="mistral", state_kind="cooldown", state_key="key1",
        )


@pytest.mark.asyncio
async def test_verify_durable_absence_mem_get_oserror_raises() -> None:
    """In-memory get_cache raising OSError -> RuntimeError (fail closed)."""
    dual_cache, _client = _make_strict_dual_cache(
        get_return=None,
        mem_get_side_effect=OSError("disk error"),
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="in-memory get failed"):
            await verify_aawm_alias_routing_durable_absence(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_verify_durable_absence_missing_mem_get_method_raises() -> None:
    """Missing get_cache on in_memory_cache -> RuntimeError (fail closed)."""
    dual_cache, _client = _make_strict_dual_cache(
        get_return=None,
        include_mem_get=False,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="missing get_cache"):
            await verify_aawm_alias_routing_durable_absence(
                alias_family="codex", state_kind="cooldown", state_key="key1",
            )


@pytest.mark.asyncio
async def test_verify_with_real_dual_cache_instance() -> None:
    """Use a real DualCache + InMemoryCache to prove strict path works.

    DualCache.async_get_cache swallows exceptions; our strict path uses
    the underlying in_memory_cache.get_cache directly.
    """
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.in_memory_cache import InMemoryCache

    in_mem = InMemoryCache()
    # Build a mock redis_cache that returns None (absent)
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.ttl = AsyncMock(return_value=None)
    redis_client.delete = AsyncMock(return_value=0)
    redis_cache_mock = MagicMock()
    redis_cache_mock.init_async_client = MagicMock(return_value=redis_client)
    redis_cache_mock.check_and_fix_namespace = MagicMock(side_effect=lambda key: key)
    redis_cache_mock._get_cache_logic = MagicMock(side_effect=_decode)

    dual = DualCache(in_memory_cache=in_mem, redis_cache=redis_cache_mock)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual):
        # Both tiers empty -> absent
        absent = await verify_aawm_alias_routing_durable_absence(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )
        assert absent is True

        # Populate in-memory tier directly
        cache_key = build_aawm_alias_routing_durable_cache_key(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )
        in_mem.set_cache(key=cache_key, value={"cooldown_key": "key1"})

        # Redis still empty but memory populated -> NOT absent
        absent2 = await verify_aawm_alias_routing_durable_absence(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )
        assert absent2 is False


# ---------------------------------------------------------------------------
# Namespace isolation tests
# ---------------------------------------------------------------------------


def test_durable_cache_key_namespace_isolation() -> None:
    key1 = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="key1",
    )
    with patch(
        f"{_DURABLE_MOD}.get_aawm_alias_routing_state_namespace",
        return_value="different-namespace",
    ):
        key2 = build_aawm_alias_routing_durable_cache_key(
            alias_family="codex", state_kind="cooldown", state_key="key1",
        )
    assert key1 != key2
    assert "aawm:alias-routing:" in key1
    assert "different-namespace" in key2


def test_durable_cache_key_family_isolation() -> None:
    key1 = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="key1",
    )
    key2 = build_aawm_alias_routing_durable_cache_key(
        alias_family="anthropic", state_kind="cooldown", state_key="key1",
    )
    assert key1 != key2
    assert ":codex:" in key1
    assert ":anthropic:" in key2


def test_durable_cache_key_state_kind_isolation() -> None:
    key1 = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="key1",
    )
    key2 = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="affinity", state_key="key1",
    )
    assert key1 != key2
    assert ":cooldown:" in key1
    assert ":affinity:" in key2


# ---------------------------------------------------------------------------
# Anthropic isolation
# ---------------------------------------------------------------------------


def test_anthropic_clear_does_not_touch_codex_state(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    mgr = fresh_manager
    mgr.codex.cooldown_until_monotonic_by_key["shared-key"] = time.monotonic() + 100
    mgr.anthropic.cooldown_until_monotonic_by_key["shared-key"] = time.monotonic() + 200

    result = mgr.clear_cooldown_state(
        alias_family="anthropic",
        canonical_aliases=[],
        cooldown_keys=["shared-key"],
    )

    assert result.positive_keys_cleared == ["shared-key"]
    assert "shared-key" in mgr.codex.cooldown_until_monotonic_by_key
    assert "shared-key" not in mgr.anthropic.cooldown_until_monotonic_by_key


# ---------------------------------------------------------------------------
# Persistent hydration end-to-end (no float TypeError)
# ---------------------------------------------------------------------------


def test_hydrate_cooldown_memory_persistent_no_type_error() -> None:
    """hydrate_cooldown_memory accepts UNBOUNDED_EXPIRY without TypeError."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        UNBOUNDED_EXPIRY,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
        hydrate_cooldown_memory,
    )

    memory_map: dict[str, float] = {}
    hydrate_cooldown_memory(
        memory_map=memory_map,
        cooldown_key="persistent-key",
        expires_at_epoch=UNBOUNDED_EXPIRY,
    )
    assert "persistent-key" in memory_map
    assert memory_map["persistent-key"] == float("inf")


def test_hydrate_affinity_memory_persistent_no_type_error() -> None:
    """hydrate_affinity_memory accepts UNBOUNDED_EXPIRY without TypeError."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        UNBOUNDED_EXPIRY,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
        hydrate_affinity_memory,
    )

    memory_map: dict[str, dict] = {}
    result = hydrate_affinity_memory(
        memory_map=memory_map,
        session_key="persistent-session",
        payload={
            "provider": "openai",
            "model": "gpt-4.1",
            "route_family": "codex_openai_responses_adapter",
            "last_resort": False,
        },
        expires_at_epoch=UNBOUNDED_EXPIRY,
    )
    assert result["provider"] == "openai"
    assert result["expires_at_monotonic"] == float("inf")
    assert "persistent-session" in memory_map


def test_hydrate_cooldown_memory_finite_still_works() -> None:
    """Finite epoch hydration still works after persistent support."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
        hydrate_cooldown_memory,
    )

    memory_map: dict[str, float] = {}
    future_epoch = time.time() + 300
    hydrate_cooldown_memory(
        memory_map=memory_map,
        cooldown_key="finite-key",
        expires_at_epoch=future_epoch,
    )
    assert "finite-key" in memory_map
    assert memory_map["finite-key"] > time.monotonic()
    assert memory_map["finite-key"] < float("inf")


def test_hydrate_affinity_memory_finite_still_works() -> None:
    """Finite epoch affinity hydration still works after persistent support."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.memory import (
        hydrate_affinity_memory,
    )

    memory_map: dict[str, dict] = {}
    future_epoch = time.time() + 300
    result = hydrate_affinity_memory(
        memory_map=memory_map,
        session_key="finite-session",
        payload={
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "route_family": "anthropic_native_adapter",
            "last_resort": True,
        },
        expires_at_epoch=future_epoch,
    )
    assert result["provider"] == "anthropic"
    assert result["last_resort"] is True
    assert result["expires_at_monotonic"] < float("inf")
    assert result["expires_at_monotonic"] > time.monotonic()


# ---------------------------------------------------------------------------
# Legacy 10-year sentinel removal regression
# ---------------------------------------------------------------------------


def test_far_future_finite_expiry_not_treated_as_persistent() -> None:
    """A far-future finite epoch (>10 years) must remain finite, not persistent."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        UNBOUNDED_EXPIRY,
        parse_aawm_alias_routing_durable_expiry,
    )

    far_future = time.time() + 400_000_000  # ~12.7 years
    payload = {"expires_at_epoch": far_future}
    result = parse_aawm_alias_routing_durable_expiry(payload)
    assert result is not UNBOUNDED_EXPIRY
    assert isinstance(result, float)
    assert result == far_future


def test_explicit_persistent_marker_still_works() -> None:
    """Explicit persistent:true marker must return UNBOUNDED_EXPIRY."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        UNBOUNDED_EXPIRY,
        parse_aawm_alias_routing_durable_expiry,
    )

    payload = {"persistent": True, "cooldown_key": "k1"}
    result = parse_aawm_alias_routing_durable_expiry(payload)
    assert result is UNBOUNDED_EXPIRY


def test_persistent_false_not_treated_as_persistent() -> None:
    """persistent:false must NOT return UNBOUNDED_EXPIRY."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        UNBOUNDED_EXPIRY,
        parse_aawm_alias_routing_durable_expiry,
    )

    payload = {"persistent": False, "expires_at_epoch": time.time() + 300}
    result = parse_aawm_alias_routing_durable_expiry(payload)
    assert result is not UNBOUNDED_EXPIRY
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Persistent raw Redis write must not forward ttl=-1 to local write-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistent_write_does_not_forward_negative_ttl_to_local() -> None:
    """Persistent write-through must skip local cache (ttl=-1 would expire)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        write_aawm_alias_routing_durable_payload,
    )

    existing_persistent = {"persistent": True, "cooldown_key": "k1"}

    redis_client = MagicMock()
    redis_client.set = AsyncMock(return_value=True)
    redis_client.persist = AsyncMock(return_value=True)

    redis_cache = MagicMock()
    redis_cache.init_async_client = MagicMock(return_value=redis_client)
    redis_cache.check_and_fix_namespace = MagicMock(side_effect=lambda key: key)
    redis_cache.async_set_cache = AsyncMock(return_value=True)
    redis_cache.async_get_cache = AsyncMock(return_value=existing_persistent)

    local_set_cache = AsyncMock(return_value=True)

    dual_cache = MagicMock()
    dual_cache.redis_cache = redis_cache
    dual_cache.async_get_cache = AsyncMock(return_value=existing_persistent)
    dual_cache.async_set_cache = local_set_cache

    with patch(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable.get_aawm_alias_routing_dual_cache",
        return_value=dual_cache,
    ):
        result = await write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="k1",
            payload={"cooldown_key": "k1"},
            ttl_seconds=300.0,
        )

    assert result is True
    # Raw Redis SET + PERSIST must have been called (persistent path).
    redis_client.set.assert_called_once()
    redis_client.persist.assert_called_once()
    # Local write-through must NOT have been called with ttl=-1.
    for call in local_set_cache.call_args_list:
        ttl_arg = call.kwargs.get("ttl", call[1].get("ttl") if len(call) > 1 else None)
        assert ttl_arg != -1.0, "ttl=-1 must not be forwarded to local write-through"


# ===========================================================================
# CFG-004 Wave B: Identity-set inspection + compare-and-clear tests
# ===========================================================================

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
    ClearIndeterminateError,
    ClearTransactionJournal,
    IdentitySetInspection,
    IdentitySetOverBoundError,
    MembershipDriftError,
    PHASE_CLEAR_COMMITTED,
    RollbackDriftError,
    RollbackReceiptMissingError,
    UNBOUNDED_EXPIRY,
    clear_cooldown_transaction,
    inspect_identity_set,
    reconcile_clear_transaction,
    rollback_clear_transaction,
)

# ---------------------------------------------------------------------------
# fakeredis + lupa availability
# ---------------------------------------------------------------------------

try:
    import fakeredis.aioredis as _fakeredis_aioredis
    import lupa as _lupa  # noqa: F401

    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False

_fakeredis_skip = pytest.mark.skipif(
    not _FAKEREDIS_AVAILABLE,
    reason="fakeredis[lua] not installed",
)


def _make_fakeredis_dual_cache(fake_redis_client, *, in_memory_cache=None):
    """Wrap a fakeredis client in the dual-cache shape expected by durable.py."""
    redis_cache = MagicMock()
    redis_cache.init_async_client = MagicMock(return_value=fake_redis_client)
    redis_cache.check_and_fix_namespace = MagicMock(side_effect=lambda key: key)
    dual_cache = MagicMock()
    dual_cache.redis_cache = redis_cache
    dual_cache.in_memory_cache = in_memory_cache
    return dual_cache


# ---------------------------------------------------------------------------
# inspect_identity_set tests
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_inspect_identity_set_existing_members() -> None:
    """Bounded inspection returns members, cardinality, and TTL."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash1"
    )
    await r.sadd(id_key, "lane-a", "lane-b")
    await r.expire(id_key, 600)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await inspect_identity_set(
            alias_family="codex", identity_hash="hash1"
        )

    assert isinstance(result, IdentitySetInspection)
    assert result.exists is True
    assert result.members == frozenset({"lane-a", "lane-b"})
    assert result.cardinality == 2
    assert result.ttl_remaining_seconds is not None
    assert 0 < result.ttl_remaining_seconds <= 600


@_fakeredis_skip
@pytest.mark.asyncio
async def test_inspect_identity_set_persistent_ttl() -> None:
    """Persistent identity set (no expiry) reports UNBOUNDED_EXPIRY."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-p"
    )
    await r.sadd(id_key, "lane-x")
    # No EXPIRE -> TTL -1

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await inspect_identity_set(
            alias_family="codex", identity_hash="hash-p"
        )

    assert result.exists is True
    assert result.ttl_remaining_seconds is UNBOUNDED_EXPIRY


@_fakeredis_skip
@pytest.mark.asyncio
async def test_inspect_identity_set_absent() -> None:
    """Absent identity set returns exists=False, empty members."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await inspect_identity_set(
            alias_family="codex", identity_hash="nonexistent"
        )

    assert result.exists is False
    assert result.members == frozenset()
    assert result.cardinality == 0


@_fakeredis_skip
@pytest.mark.asyncio
async def test_inspect_identity_set_exceeds_bound_raises() -> None:
    """Cardinality exceeding max_members raises RuntimeError."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-big"
    )
    for i in range(10):
        await r.sadd(id_key, f"lane-{i}")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="exceeds bound"):
            await inspect_identity_set(
                alias_family="codex", identity_hash="hash-big", max_members=5
            )


@pytest.mark.asyncio
async def test_inspect_identity_set_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown alias_family"):
        await inspect_identity_set(alias_family="gemini", identity_hash="h")


@pytest.mark.asyncio
async def test_inspect_identity_set_no_cache_fails_closed() -> None:
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=None):
        with pytest.raises(RuntimeError, match="no Redis cache available"):
            await inspect_identity_set(alias_family="codex", identity_hash="h")


# ---------------------------------------------------------------------------
# clear_cooldown_transaction: finite TTL
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_finite_ttl_deletes_keys_and_members() -> None:
    """Clear with finite-TTL cooldown keys: keys deleted, members removed."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key1 = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="ck1"
    )
    cd_key2 = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="ck2"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-f"
    )
    await r.set(cd_key1, b'{"cooldown_key":"ck1","expires_at_epoch":9999999999}')
    await r.expire(cd_key1, 300)
    await r.set(cd_key2, b'{"cooldown_key":"ck2","expires_at_epoch":9999999999}')
    await r.expire(cd_key2, 600)
    await r.sadd(id_key, "lane-a", "lane-b")
    await r.expire(id_key, 300)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-f",
            cooldown_keys=["ck1", "ck2"],
            expected_members=["lane-a", "lane-b"],
        )

    assert result.phase == PHASE_CLEAR_COMMITTED
    assert result.keys_deleted == 2
    assert result.members_removed == 2
    # Cooldown keys must be gone.
    assert await r.get(cd_key1) is None
    assert await r.get(cd_key2) is None
    # Identity set must be gone (all members removed -> empty -> DEL).
    assert await r.exists(id_key) == 0


# ---------------------------------------------------------------------------
# clear_cooldown_transaction: persistent TTL
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_persistent_ttl_keys_deleted() -> None:
    """Persistent (TTL -1) cooldown keys are deleted; receipt journals -1."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="persist-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-p2"
    )
    await r.set(cd_key, b'{"persistent":true}')
    # No EXPIRE -> TTL -1
    await r.sadd(id_key, "lane-p")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-p2",
            cooldown_keys=["persist-ck"],
            expected_members=["lane-p"],
        )

    assert result.phase == PHASE_CLEAR_COMMITTED
    assert await r.get(cd_key) is None
    assert await r.exists(id_key) == 0


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_namespace_isolation() -> None:
    """Clear in one namespace does not affect another namespace's keys."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Seed keys in default namespace.
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="ns-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-ns"
    )
    await r.set(cd_key, b'{"cooldown_key":"ns-ck"}')
    await r.sadd(id_key, "lane-ns")

    # Seed unrelated key in a different namespace prefix.
    foreign_key = "aawm:alias-routing:other-ns:codex:cooldown:foreign"
    await r.set(foreign_key, b'{"foreign":true}')

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-ns",
            cooldown_keys=["ns-ck"],
            expected_members=["lane-ns"],
        )

    # Target keys gone.
    assert await r.get(cd_key) is None
    # Foreign key untouched.
    assert await r.get(foreign_key) == b'{"foreign":true}'


# ---------------------------------------------------------------------------
# Unrelated key/member preservation
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_preserves_unrelated_keys_and_members() -> None:
    """Clear only removes selected cooldown keys and members."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_selected = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="sel-ck"
    )
    cd_unrelated = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="unrel-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-up"
    )
    await r.set(cd_selected, b'{"cooldown_key":"sel-ck"}')
    await r.set(cd_unrelated, b'{"cooldown_key":"unrel-ck"}')
    await r.sadd(id_key, "lane-sel", "lane-keep")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-up",
            cooldown_keys=["sel-ck"],
            expected_members=["lane-sel", "lane-keep"],
            lane_members=["lane-sel"],
        )

    assert result.keys_deleted == 1
    assert result.members_removed == 1
    # Selected key gone.
    assert await r.get(cd_selected) is None
    # Unrelated key preserved.
    assert await r.get(cd_unrelated) == b'{"cooldown_key":"unrel-ck"}'
    # Unrelated member preserved.
    assert await r.sismember(id_key, "lane-keep") == 1
    assert await r.sismember(id_key, "lane-sel") == 0
    # Identity set NOT deleted (still has lane-keep).
    assert await r.exists(id_key) == 1


# ---------------------------------------------------------------------------
# Membership drift
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_membership_drift_raises() -> None:
    """Extra member in identity set -> MembershipDriftError, no mutations."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="drift-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-drift"
    )
    await r.set(cd_key, b'{"cooldown_key":"drift-ck"}')
    await r.sadd(id_key, "lane-a", "lane-unexpected")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(MembershipDriftError):
            await clear_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash-drift",
                cooldown_keys=["drift-ck"],
                expected_members=["lane-a"],
            )

    # No mutations: key and members intact.
    assert await r.get(cd_key) is not None
    assert await r.sismember(id_key, "lane-a") == 1
    assert await r.sismember(id_key, "lane-unexpected") == 1


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_membership_drift_missing_member() -> None:
    """Expected member absent from identity set -> MembershipDriftError."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="drift2-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-drift2"
    )
    await r.set(cd_key, b'{"cooldown_key":"drift2-ck"}')
    await r.sadd(id_key, "lane-a")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(MembershipDriftError):
            await clear_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash-drift2",
                cooldown_keys=["drift2-ck"],
                expected_members=["lane-a", "lane-missing"],
            )

    assert await r.get(cd_key) is not None


# ---------------------------------------------------------------------------
# Idempotent absence
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_idempotent_absence() -> None:
    """Clear on absent keys with empty identity set succeeds idempotently."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-empty",
            cooldown_keys=["absent-ck"],
            expected_members=[],
            lane_members=[],
        )

    assert result.phase == PHASE_CLEAR_COMMITTED
    assert result.keys_deleted == 1
    assert result.members_removed == 0


# ---------------------------------------------------------------------------
# DualCache invalidation
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_invalidates_dual_cache() -> None:
    """Successful clear invalidates in-memory cache for each cooldown key."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    in_memory = MagicMock()
    in_memory.delete_cache = MagicMock()
    dual_cache = _make_fakeredis_dual_cache(r, in_memory_cache=in_memory)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="inv-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-inv"
    )
    await r.set(cd_key, b'{"cooldown_key":"inv-ck"}')
    await r.sadd(id_key, "lane-inv")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-inv",
            cooldown_keys=["inv-ck"],
            expected_members=["lane-inv"],
        )

    in_memory.delete_cache.assert_called_once_with(cd_key)


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_mem_delete_failure_raises() -> None:
    """In-memory delete_cache failure raises RuntimeError."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    in_memory = MagicMock()
    in_memory.delete_cache = MagicMock(side_effect=OSError("locked"))
    dual_cache = _make_fakeredis_dual_cache(r, in_memory_cache=in_memory)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="memfail-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-mf"
    )
    await r.set(cd_key, b'{"cooldown_key":"memfail-ck"}')
    await r.sadd(id_key, "lane-mf")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="in-memory invalidation failed"):
            await clear_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash-mf",
                cooldown_keys=["memfail-ck"],
                expected_members=["lane-mf"],
            )


# ---------------------------------------------------------------------------
# Lost-response reconciliation
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_lost_response_reconciles_committed() -> None:
    """EVAL raises but receipt exists -> reconcile returns committed result."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="lost-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-lost"
    )
    await r.set(cd_key, b'{"cooldown_key":"lost-ck"}')
    await r.sadd(id_key, "lane-lost")

    call_count = 0
    original_eval = r.eval

    async def _eval_then_lose(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Execute the script (commits server-side), then simulate lost response.
        _ = await original_eval(*args, **kwargs)
        raise ConnectionError("response lost")

    r.eval = _eval_then_lose

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-lost",
            cooldown_keys=["lost-ck"],
            expected_members=["lane-lost"],
        )

    assert result.phase == PHASE_CLEAR_COMMITTED
    # Keys were actually deleted by the Lua script.
    assert await r.get(cd_key) is None


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_lost_response_indeterminate() -> None:
    """EVAL raises and no receipt -> ClearIndeterminateError."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="indet-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-indet"
    )
    await r.set(cd_key, b'{"cooldown_key":"indet-ck"}')
    await r.sadd(id_key, "lane-indet")

    async def _eval_fail(*args, **kwargs):
        raise ConnectionError("connection lost before commit")

    r.eval = _eval_fail

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(ClearIndeterminateError):
            await clear_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash-indet",
                cooldown_keys=["indet-ck"],
                expected_members=["lane-indet"],
            )

    # Key untouched (no commit occurred).
    assert await r.get(cd_key) is not None


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_rollback_clear_restores_finite_preimages() -> None:
    """Rollback restores cooldown values, TTLs, and identity membership."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="rb-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-rb"
    )
    original_value = b'{"cooldown_key":"rb-ck","expires_at_epoch":9999999999}'
    await r.set(cd_key, original_value)
    await r.expire(cd_key, 300)
    await r.sadd(id_key, "lane-rb")
    await r.expire(id_key, 300)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-rb",
            cooldown_keys=["rb-ck"],
            expected_members=["lane-rb"],
        )

    assert result.phase == PHASE_CLEAR_COMMITTED
    assert await r.get(cd_key) is None

    # Rollback.
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await rollback_clear_transaction(
            alias_family="codex",
            journal=result.journal,
        )

    # Cooldown key restored.
    restored = await r.get(cd_key)
    assert restored is not None
    assert b"rb-ck" in restored
    # TTL restored (approximately 300).
    ttl = await r.ttl(cd_key)
    assert 250 < ttl <= 300
    # Identity set restored.
    assert await r.sismember(id_key, "lane-rb") == 1


@_fakeredis_skip
@pytest.mark.asyncio
async def test_rollback_clear_restores_persistent_preimages() -> None:
    """Rollback restores persistent (TTL -1) cooldown key."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="rbp-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-rbp"
    )
    await r.set(cd_key, b'{"persistent":true}')
    # No EXPIRE -> TTL -1
    await r.sadd(id_key, "lane-rbp")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-rbp",
            cooldown_keys=["rbp-ck"],
            expected_members=["lane-rbp"],
        )

    assert await r.get(cd_key) is None

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await rollback_clear_transaction(
            alias_family="codex",
            journal=result.journal,
        )

    restored = await r.get(cd_key)
    assert restored is not None
    assert b"persistent" in restored
    # Must be persistent (TTL -1).
    ttl = await r.ttl(cd_key)
    assert ttl == -1


@_fakeredis_skip
@pytest.mark.asyncio
async def test_rollback_clear_missing_receipt_raises_indeterminate() -> None:
    """Rollback with missing/expired receipt raises RollbackReceiptMissingError."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    journal = ClearTransactionJournal(
        transaction_id="nonexistent-txn",
        phase=PHASE_CLEAR_COMMITTED,
        alias_family="codex",
        identity_hash="hash-noop",
        cooldown_keys=["noop-ck"],
        lane_members=["lane-noop"],
        expected_members=["lane-noop"],
        identity_key=build_aawm_alias_routing_durable_cache_key(
            alias_family="codex", state_kind="lane_identity", state_key="hash-noop"
        ),
        receipt_key=build_aawm_alias_routing_durable_cache_key(
            alias_family="codex", state_kind="txn_receipt", state_key="clear-receipt:nonexistent-txn"
        ),
        receipt_ttl=300,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RollbackReceiptMissingError, match="receipt missing or expired"):
            await rollback_clear_transaction(
                alias_family="codex",
                journal=journal,
            )


# ---------------------------------------------------------------------------
# reconcile_clear_transaction
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_reconcile_clear_committed() -> None:
    """Receipt present + postconditions satisfied -> reconcile returns True."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    receipt_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="txn_receipt", state_key="clear-receipt:txn123"
    )
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="rcpt-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-rcpt"
    )
    await r.set(receipt_key, b'{"txn_id":"txn123"}')
    # Cooldown key absent, member absent: clear postconditions satisfied.

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        committed = await reconcile_clear_transaction(
            alias_family="codex",
            transaction_id="txn123",
            cooldown_cache_keys=[cd_key],
            identity_cache_key=id_key,
            lane_members=["lane-rcpt"],
        )

    assert committed is True


@_fakeredis_skip
@pytest.mark.asyncio
async def test_reconcile_clear_not_committed() -> None:
    """Receipt absent -> reconcile returns False (postconditions not reached)."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="absent-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-absent"
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        committed = await reconcile_clear_transaction(
            alias_family="codex",
            transaction_id="absent-txn",
            cooldown_cache_keys=[cd_key],
            identity_cache_key=id_key,
            lane_members=["lane-absent"],
        )

    assert committed is False


# ---------------------------------------------------------------------------
# Postcondition verification failure
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_postcondition_failure_raises() -> None:
    """If a cooldown key reappears after Lua commit, postcondition raises."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="post-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-post"
    )
    await r.set(cd_key, b'{"cooldown_key":"post-ck"}')
    await r.sadd(id_key, "lane-post")

    original_eval = r.eval

    async def _eval_then_reseed(*args, **kwargs):
        result = await original_eval(*args, **kwargs)
        # Simulate a concurrent writer re-creating the key after commit.
        await r.set(cd_key, b'{"cooldown_key":"post-ck","recreated":true}')
        return result

    r.eval = _eval_then_reseed

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="still present after clear"):
            await clear_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash-post",
                cooldown_keys=["post-ck"],
                expected_members=["lane-post"],
            )


# ---------------------------------------------------------------------------
# Anthropic family isolation
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_anthropic_does_not_touch_codex() -> None:
    """Clear on anthropic family does not affect codex keys."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    codex_cd = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="shared-ck"
    )
    anthro_cd = build_aawm_alias_routing_durable_cache_key(
        alias_family="anthropic", state_kind="cooldown", state_key="shared-ck"
    )
    anthro_id = build_aawm_alias_routing_durable_cache_key(
        alias_family="anthropic", state_kind="lane_identity", state_key="hash-iso"
    )
    await r.set(codex_cd, b'{"family":"codex"}')
    await r.set(anthro_cd, b'{"family":"anthropic"}')
    await r.sadd(anthro_id, "lane-iso")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await clear_cooldown_transaction(
            alias_family="anthropic",
            identity_hash="hash-iso",
            cooldown_keys=["shared-ck"],
            expected_members=["lane-iso"],
        )

    assert await r.get(anthro_cd) is None
    assert await r.get(codex_cd) == b'{"family":"codex"}'


# ---------------------------------------------------------------------------
# Unknown family
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_transaction_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown alias_family"):
        await clear_cooldown_transaction(
            alias_family="openai",
            identity_hash="h",
            cooldown_keys=["k"],
            expected_members=[],
        )


@pytest.mark.asyncio
async def test_clear_transaction_no_cache_fails_closed() -> None:
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=None):
        with pytest.raises(RuntimeError, match="no Redis cache available"):
            await clear_cooldown_transaction(
                alias_family="codex",
                identity_hash="h",
                cooldown_keys=["k"],
                expected_members=[],
            )


# ---------------------------------------------------------------------------
# TOCTOU bound: atomic Lua inspection rejects oversized sets
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_inspect_identity_set_toctou_bound_rejects_oversized() -> None:
    """Atomic Lua SCARD-before-SMEMBERS rejects sets exceeding max_members."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-toctou"
    )
    # Add 10 members
    for i in range(10):
        await r.sadd(id_key, f"lane-{i}")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(IdentitySetOverBoundError, match="cardinality 10 exceeds bound 5"):
            await inspect_identity_set(
                alias_family="codex",
                identity_hash="hash-toctou",
                max_members=5,
            )


@_fakeredis_skip
@pytest.mark.asyncio
async def test_inspect_identity_set_toctou_bound_accepts_within_limit() -> None:
    """Atomic Lua inspection succeeds when cardinality is within bound."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-ok"
    )
    await r.sadd(id_key, "lane-a", "lane-b")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await inspect_identity_set(
            alias_family="codex",
            identity_hash="hash-ok",
            max_members=5,
        )

    assert result.cardinality == 2
    assert result.members == frozenset({"lane-a", "lane-b"})


# ---------------------------------------------------------------------------
# Lost-response reconciliation: mandatory DualCache invalidation
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_lost_response_mandatory_invalidation() -> None:
    """Lost-response reconciliation runs mandatory DualCache invalidation."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    in_memory = MagicMock()
    in_memory.delete_cache = MagicMock()
    dual_cache = _make_fakeredis_dual_cache(r, in_memory_cache=in_memory)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="lost-inv-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-lost-inv"
    )
    await r.set(cd_key, b'{"cooldown_key":"lost-inv-ck"}')
    await r.sadd(id_key, "lane-lost-inv")

    call_count = 0
    original_eval = r.eval

    async def _eval_then_lose(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        _ = await original_eval(*args, **kwargs)
        raise ConnectionError("response lost")

    r.eval = _eval_then_lose

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-lost-inv",
            cooldown_keys=["lost-inv-ck"],
            expected_members=["lane-lost-inv"],
        )

    assert result.phase == PHASE_CLEAR_COMMITTED
    # Mandatory invalidation was called.
    in_memory.delete_cache.assert_called_once_with(cd_key)


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_lost_response_invalidation_failure_raises() -> None:
    """Lost-response reconciliation raises on invalidation failure (never swallows)."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    in_memory = MagicMock()
    in_memory.delete_cache = MagicMock(side_effect=OSError("locked"))
    dual_cache = _make_fakeredis_dual_cache(r, in_memory_cache=in_memory)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="lost-fail-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-lost-fail"
    )
    await r.set(cd_key, b'{"cooldown_key":"lost-fail-ck"}')
    await r.sadd(id_key, "lane-lost-fail")

    original_eval = r.eval

    async def _eval_then_lose(*args, **kwargs):
        _ = await original_eval(*args, **kwargs)
        raise ConnectionError("response lost")

    r.eval = _eval_then_lose

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="in-memory invalidation failed"):
            await clear_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash-lost-fail",
                cooldown_keys=["lost-fail-ck"],
                expected_members=["lane-lost-fail"],
            )


# ---------------------------------------------------------------------------
# Cross-worker rollback drift detection
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_rollback_clear_drift_detection_rejects_modified_state() -> None:
    """Rollback rejects when current state drifted from receipt post-image."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="drift-rb-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-drift-rb"
    )
    await r.set(cd_key, b'{"cooldown_key":"drift-rb-ck"}')
    await r.sadd(id_key, "lane-drift-rb")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-drift-rb",
            cooldown_keys=["drift-rb-ck"],
            expected_members=["lane-drift-rb"],
        )

    assert result.phase == PHASE_CLEAR_COMMITTED
    assert await r.get(cd_key) is None

    # Simulate cross-worker drift: another writer re-creates the key.
    await r.set(cd_key, b'{"cooldown_key":"drift-rb-ck","modified_by":"worker2"}')

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RollbackDriftError, match="rollback drift detected"):
            await rollback_clear_transaction(
                alias_family="codex",
                journal=result.journal,
            )

    # Receipt retained for forensics (not deleted on drift).
    receipt_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex",
        state_kind="txn_receipt",
        state_key=f"clear-receipt:{result.transaction_id}",
    )
    assert await r.get(receipt_key) is not None


@_fakeredis_skip
@pytest.mark.asyncio
async def test_rollback_clear_drift_detection_identity_membership() -> None:
    """Rollback rejects when identity set membership drifted."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="drift-id-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-drift-id"
    )
    await r.set(cd_key, b'{"cooldown_key":"drift-id-ck"}')
    await r.sadd(id_key, "lane-a", "lane-b")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-drift-id",
            cooldown_keys=["drift-id-ck"],
            expected_members=["lane-a", "lane-b"],
            lane_members=["lane-a"],
        )

    assert await r.sismember(id_key, "lane-a") == 0
    assert await r.sismember(id_key, "lane-b") == 1

    # Simulate drift: another worker adds a new member.
    await r.sadd(id_key, "lane-c")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RollbackDriftError, match="rollback drift detected"):
            await rollback_clear_transaction(
                alias_family="codex",
                journal=result.journal,
            )


# ---------------------------------------------------------------------------
# Redaction: no identity-hash prefixes in exception strings
# ---------------------------------------------------------------------------


def test_exception_strings_redact_identity_hash() -> None:
    """Exception messages must not contain identity-hash prefixes."""
    exc = MembershipDriftError(
        phase="PREPARED",
        family="codex",
        transaction_id_prefix="abc123",
        identity_prefix="def456",
        key_count=2,
        exception_classes=(),
    )
    msg = str(exc)
    # identity_prefix parameter is accepted but NOT included in message
    assert "identity=" not in msg
    assert "def456" not in msg
    assert "phase=PREPARED" in msg
    assert "family=codex" in msg


def test_exception_strings_redact_identity_hash_rollback_drift() -> None:
    """RollbackDriftError message must not contain identity-hash prefixes."""
    exc = RollbackDriftError(
        phase="CLEAR_COMMITTED",
        family="anthropic",
        transaction_id_prefix="txn789",
        identity_prefix="hash999",
        key_count=1,
        exception_classes=(),
    )
    msg = str(exc)
    assert "identity=" not in msg
    assert "hash999" not in msg
    assert "rollback drift detected" in msg


def test_exception_strings_redact_identity_hash_receipt_missing() -> None:
    """RollbackReceiptMissingError message must not contain identity-hash prefixes."""
    exc = RollbackReceiptMissingError(
        phase="CLEAR_COMMITTED",
        family="codex",
        transaction_id_prefix="txn000",
        identity_prefix="hashABC",
        key_count=3,
        exception_classes=(),
    )
    msg = str(exc)
    assert "identity=" not in msg
    assert "hashABC" not in msg
    assert "receipt missing or expired" in msg


# ---------------------------------------------------------------------------
# Receipt-only false success prevention (reconcile postconditions)
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_reconcile_clear_receipt_present_but_keys_still_exist_raises() -> None:
    """Receipt exists but cooldown key still present -> RuntimeError, not True."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        reconcile_clear_transaction,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    receipt_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="txn_receipt", state_key="clear-receipt:txn-false"
    )
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="false-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-false"
    )
    # Receipt present (simulates committed Lua) but cooldown key NOT deleted.
    await r.set(receipt_key, b'{"txn_id":"txn-false"}')
    await r.set(cd_key, b'{"cooldown_key":"false-ck"}')
    await r.sadd(id_key, "lane-false")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="cooldown key still present"):
            await reconcile_clear_transaction(
                alias_family="codex",
                transaction_id="txn-false",
                cooldown_cache_keys=[cd_key],
                identity_cache_key=id_key,
                lane_members=["lane-false"],
            )


@_fakeredis_skip
@pytest.mark.asyncio
async def test_reconcile_clear_receipt_present_member_still_in_set_raises() -> None:
    """Receipt exists, keys deleted, but member still in identity set -> RuntimeError."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        reconcile_clear_transaction,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    receipt_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="txn_receipt", state_key="clear-receipt:txn-mem"
    )
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="mem-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-mem"
    )
    await r.set(receipt_key, b'{"txn_id":"txn-mem"}')
    # Cooldown key deleted (correct) but member NOT removed from set.
    await r.sadd(id_key, "lane-mem")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="member still in identity set"):
            await reconcile_clear_transaction(
                alias_family="codex",
                transaction_id="txn-mem",
                cooldown_cache_keys=[cd_key],
                identity_cache_key=id_key,
                lane_members=["lane-mem"],
            )


@_fakeredis_skip
@pytest.mark.asyncio
async def test_reconcile_publication_receipt_present_but_keys_absent_raises() -> None:
    """Receipt exists but cooldown key absent -> RuntimeError, not True."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        reconcile_cooldown_transaction,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    receipt_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="txn_receipt", state_key="txn-receipt:txn-pub"
    )
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="pub-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-pub"
    )
    # Receipt present but cooldown key NOT written (publication didn't commit data).
    await r.set(receipt_key, b'{"txn_id":"txn-pub"}')
    await r.sadd(id_key, "lane-pub")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="cooldown key absent after reconciled publication"):
            await reconcile_cooldown_transaction(
                alias_family="codex",
                transaction_id="txn-pub",
                cooldown_cache_keys=[cd_key],
                identity_cache_key=id_key,
                lane_members=["lane-pub"],
            )


@_fakeredis_skip
@pytest.mark.asyncio
async def test_reconcile_publication_receipt_present_member_not_in_set_raises() -> None:
    """Receipt exists, keys written, but member not in identity set -> RuntimeError."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        reconcile_cooldown_transaction,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    receipt_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="txn_receipt", state_key="txn-receipt:txn-pub2"
    )
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="pub2-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-pub2"
    )
    await r.set(receipt_key, b'{"txn_id":"txn-pub2"}')
    await r.set(cd_key, b'{"cooldown_key":"pub2-ck"}')
    # Identity set exists but member NOT added.
    await r.sadd(id_key, "other-lane")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="lane member absent from identity set"):
            await reconcile_cooldown_transaction(
                alias_family="codex",
                transaction_id="txn-pub2",
                cooldown_cache_keys=[cd_key],
                identity_cache_key=id_key,
                lane_members=["lane-pub2"],
            )


@_fakeredis_skip
@pytest.mark.asyncio
async def test_reconcile_clear_postconditions_pass_when_state_correct() -> None:
    """Receipt present, keys deleted, members removed -> True."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        reconcile_clear_transaction,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    receipt_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="txn_receipt", state_key="clear-receipt:txn-ok"
    )
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="ok-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-ok"
    )
    await r.set(receipt_key, b'{"txn_id":"txn-ok"}')
    # Cooldown key deleted, member removed (empty set or absent).

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        committed = await reconcile_clear_transaction(
            alias_family="codex",
            transaction_id="txn-ok",
            cooldown_cache_keys=[cd_key],
            identity_cache_key=id_key,
            lane_members=["lane-ok"],
        )

    assert committed is True


@_fakeredis_skip
@pytest.mark.asyncio
async def test_reconcile_publication_postconditions_pass_when_state_correct() -> None:
    """Receipt present, keys written, members registered -> True."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        reconcile_cooldown_transaction,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    receipt_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="txn_receipt", state_key="txn-receipt:txn-ok2"
    )
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="ok2-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-ok2"
    )
    await r.set(receipt_key, b'{"txn_id":"txn-ok2"}')
    await r.set(cd_key, b'{"cooldown_key":"ok2-ck"}')
    await r.sadd(id_key, "lane-ok2")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        committed = await reconcile_cooldown_transaction(
            alias_family="codex",
            transaction_id="txn-ok2",
            cooldown_cache_keys=[cd_key],
            identity_cache_key=id_key,
            lane_members=["lane-ok2"],
        )

    assert committed is True


# ---------------------------------------------------------------------------
# Production family label canonicalization
# ---------------------------------------------------------------------------


def test_validate_alias_family_accepts_codex_auto_agent() -> None:
    """Production label codex_auto_agent canonicalizes to codex."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        _validate_alias_family,
    )

    assert _validate_alias_family("codex_auto_agent", "test") == "codex"
    assert _validate_alias_family("CODEX_AUTO_AGENT", "test") == "codex"
    assert _validate_alias_family(" codex_auto_agent ", "test") == "codex"


def test_validate_alias_family_accepts_anthropic_auto_agent() -> None:
    """Production label anthropic_auto_agent canonicalizes to anthropic."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        _validate_alias_family,
    )

    assert _validate_alias_family("anthropic_auto_agent", "test") == "anthropic"
    assert _validate_alias_family("ANTHROPIC_AUTO_AGENT", "test") == "anthropic"


def test_validate_alias_family_rejects_unknown() -> None:
    """Unknown families still raise ValueError."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        _validate_alias_family,
    )

    with pytest.raises(ValueError, match="unknown alias_family"):
        _validate_alias_family("gemini_auto_agent", "test")


@_fakeredis_skip
@pytest.mark.asyncio
async def test_clear_transaction_accepts_production_family_label() -> None:
    """clear_cooldown_transaction accepts codex_auto_agent and uses codex keys."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Pre-populate using canonical "codex" family keys.
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="prod-ck"
    )
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-prod"
    )
    await r.set(cd_key, b'{"cooldown_key":"prod-ck"}')
    await r.sadd(id_key, "lane-prod")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await clear_cooldown_transaction(
            alias_family="codex_auto_agent",
            identity_hash="hash-prod",
            cooldown_keys=["prod-ck"],
            expected_members=["lane-prod"],
        )

    assert result.phase == PHASE_CLEAR_COMMITTED
    assert await r.get(cd_key) is None
