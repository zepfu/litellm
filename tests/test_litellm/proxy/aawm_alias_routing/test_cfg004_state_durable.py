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

    result = mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=["key1"])

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

    result = mgr.clear_cooldown_state(alias_family="anthropic", cooldown_keys=["key1"])

    assert result.alias_family == "anthropic"
    assert result.positive_keys_cleared == ["key1"]


def test_manager_clear_cooldown_state_clears_read_pilot_gate_codex_only(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    """Read-pilot gate is codex-owned; clearing anthropic must not touch it."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.classification import (
        _KeyCooldownState,
    )

    mgr = fresh_manager
    mgr.read_pilot_gate._key_state["key1"] = _KeyCooldownState()
    mgr.read_pilot_gate._family_state.evidence_events_by_key["key1"] = [time.monotonic()]

    result_codex = mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=["key1"])
    assert result_codex.read_pilot_keys_cleared == ["key1"]
    assert "key1" not in mgr.read_pilot_gate._key_state

    mgr.read_pilot_gate._key_state["key2"] = _KeyCooldownState()
    mgr.read_pilot_gate._family_state.evidence_events_by_key["key2"] = [time.monotonic()]
    result_anthropic = mgr.clear_cooldown_state(
        alias_family="anthropic", cooldown_keys=["key2"]
    )
    assert result_anthropic.read_pilot_keys_cleared == []
    assert "key2" in mgr.read_pilot_gate._key_state


def test_manager_clear_cooldown_state_unknown_family_raises(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    with pytest.raises(ValueError, match="Unknown alias_family"):
        fresh_manager.clear_cooldown_state(alias_family="openai", cooldown_keys=["k"])


def test_manager_clear_cooldown_state_reports_affinity_count(
    fresh_manager: AliasRoutingStateManager,
) -> None:
    mgr = fresh_manager
    mgr.codex.session_affinity_by_key["session1"] = {"provider": "openai"}
    mgr.codex.session_affinity_by_key["session2"] = {"provider": "anthropic"}

    result = mgr.clear_cooldown_state(alias_family="codex", cooldown_keys=["key1"])

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


def test_lane_identity_index_max_identities_eviction() -> None:
    index = LaneIdentityIndex(max_identities=2)
    index.register(identity_hash="hash1", lane_key="lane1")
    index.register(identity_hash="hash2", lane_key="lane2")
    index.register(identity_hash="hash3", lane_key="lane3")
    assert index.lanes_for("hash1") == frozenset()
    assert index.lanes_for("hash2") == frozenset({"lane2"})
    assert index.lanes_for("hash3") == frozenset({"lane3"})


def test_lane_identity_index_max_lanes_per_identity_eviction() -> None:
    index = LaneIdentityIndex(max_lanes_per_identity=2)
    index.register(identity_hash="hash1", lane_key="lane1")
    index.register(identity_hash="hash1", lane_key="lane2")
    index.register(identity_hash="hash1", lane_key="lane3")
    lanes = index.lanes_for("hash1")
    assert len(lanes) == 2
    assert "lane3" in lanes


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

    result = mgr.clear_cooldown_state(alias_family="anthropic", cooldown_keys=["shared-key"])

    assert result.positive_keys_cleared == ["shared-key"]
    assert "shared-key" in mgr.codex.cooldown_until_monotonic_by_key
    assert "shared-key" not in mgr.anthropic.cooldown_until_monotonic_by_key
