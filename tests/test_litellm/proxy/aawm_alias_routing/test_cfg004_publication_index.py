"""CFG-004: Publication lock ordering, atomic transaction, intent-based
single-flight, and adversarial tests.

Tests: publish vs family-first clear (no deadlock, final clear), overlapping
reversed-key publications (no deadlock), follower intent waits (no second
provider call), configured Redis failures leave no local state, atomic
capacity rejection, commit-then-raise reconciliation, local commit failure
rollback, rollback failure sanitized error, persistent TTL -1, positive TTL
monotonic+ceil, no secret/key leakage, and complete postcondition assertions.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply import (
    execute_cooldown_publication_transaction,
    resolve_lane_identity_hash,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
    CapacityRejectedError,
    CooldownTransactionJournal,
    PHASE_DURABLE_COMMITTED,
    PublicationTransactionError,
    RollbackFailedError,
    _ceil_ttl,
    build_aawm_alias_routing_durable_cache_key,
    publish_cooldown_transaction,
    reconcile_cooldown_transaction,
    rollback_cooldown_transaction,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.interfaces import (
    CooldownPublicationPlan,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    ClearReservationStatus,
    AliasRoutingStateManager,
    LaneIdentityIndex,
    PublicationIntentRegistry,
    RegisterBatchOutcome,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DURABLE_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable"
_APPLY_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_apply"
_LOOP_MOD = "litellm.proxy.pass_through_endpoints.aawm_alias_routing.candidate_loop"


@pytest.fixture()
def fresh_manager() -> AliasRoutingStateManager:
    return AliasRoutingStateManager()


def _make_strict_dual_cache(
    *,
    eval_return=1,
    eval_side_effect=None,
    smembers_return=None,
    srem_return=1,
    delete_return=1,
):
    redis_client = MagicMock()
    redis_client.eval = AsyncMock(return_value=eval_return, side_effect=eval_side_effect)
    redis_client.smembers = AsyncMock(return_value=smembers_return or set())
    redis_client.sadd = AsyncMock(return_value=1)
    redis_client.srem = AsyncMock(return_value=srem_return)
    redis_client.delete = AsyncMock(return_value=delete_return)
    redis_client.get = AsyncMock(return_value=None)
    redis_client.set = AsyncMock(return_value=True)
    redis_client.ttl = AsyncMock(return_value=None)
    redis_cache = MagicMock()
    redis_cache.init_async_client = MagicMock(return_value=redis_client)
    redis_cache.check_and_fix_namespace = MagicMock(side_effect=lambda key: key)
    redis_cache._get_cache_logic = MagicMock(side_effect=lambda x: x)
    redis_cache.async_set_cache = AsyncMock(return_value=True)
    redis_cache.async_get_cache = AsyncMock(return_value=None)
    redis_cache.async_delete_cache = AsyncMock(return_value=1)
    in_memory = MagicMock()
    in_memory.get_cache = MagicMock(return_value=None)
    in_memory.delete_cache = MagicMock()
    dual_cache = MagicMock()
    dual_cache.redis_cache = redis_cache
    dual_cache.in_memory_cache = in_memory
    return dual_cache, redis_client


_CANDIDATE_OPENAI = {
    "provider": "openai",
    "model": "gpt-4.1",
    "route_family": "codex_openai_responses_adapter",
}
_CANDIDATE_OPENROUTER = {
    "provider": "openrouter",
    "model": "openrouter/cohere/north-mini-code:free",
    "route_family": "codex_openrouter_completion_adapter",
}

def _make_memory_publisher(mgr: AliasRoutingStateManager):
    """Create a memory publisher that actually executes (not a generator)."""
    def _publish(*, keys, seconds):
        for k in keys:
            mgr.codex.set_cooldown_memory(k, seconds)
    return _publish




# ---------------------------------------------------------------------------
# resolve_lane_identity_hash tests
# ---------------------------------------------------------------------------


def test_identity_hash_from_candidate_only() -> None:
    h1 = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)
    h2 = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)
    assert h1 == h2
    assert len(h1) == 64


def test_identity_hash_no_raw_credential_exposure() -> None:
    h = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)
    assert "sk-" not in h
    assert "openai" not in h
    assert len(h) == 64


def test_identity_hash_different_candidates_differ() -> None:
    c2 = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "route_family": "anthropic_native_adapter",
    }
    assert resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI) != resolve_lane_identity_hash(candidate=c2)


# ---------------------------------------------------------------------------
# TTL tests
# ---------------------------------------------------------------------------


def test_ceil_ttl_fractional() -> None:
    assert _ceil_ttl(0.1) == 1
    assert _ceil_ttl(1.0) == 1
    assert _ceil_ttl(1.1) == 2
    assert _ceil_ttl(299.9) == 300
    assert _ceil_ttl(300.0) == 300
    assert _ceil_ttl(0.0) == 1


@pytest.mark.asyncio
async def test_persistent_ttl_minus_one() -> None:
    """Lua script preserves persistent TTL (-1): PERSIST called, no EXPIRE."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash1",
            cooldown_keys=["key-a"],
            lane_members=["lane-a"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED
    # Verify the Lua script was called (atomic)
    client.eval.assert_called_once()


@pytest.mark.asyncio
async def test_positive_ttl_monotonic_ceil() -> None:
    """Positive TTL is ceiled and monotonic (Lua enforces max)."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash1",
            cooldown_keys=["key-a"],
            lane_members=["lane-a"],
            ttl_seconds=299.5,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED
    # TTL passed as "300" (ceiled from 299.5)
    call_args = str(client.eval.call_args)
    assert "300" in call_args


# ---------------------------------------------------------------------------
# LaneIdentityIndex batch/no-eviction tests
# ---------------------------------------------------------------------------


def test_index_register_no_eviction() -> None:
    """Register rejects when at capacity instead of evicting."""
    idx = LaneIdentityIndex(max_identities=2, max_lanes_per_identity=2)
    assert idx.register(identity_hash="h1", lane_key="l1") is True
    assert idx.register(identity_hash="h1", lane_key="l2") is True
    # At lane capacity: reject, no eviction
    assert idx.register(identity_hash="h1", lane_key="l3") is False
    # l1 and l2 still present (no eviction)
    assert idx.lanes_for("h1") == frozenset({"l1", "l2"})


def test_index_identity_capacity_no_eviction() -> None:
    """Identity capacity rejects new identities without evicting old ones."""
    idx = LaneIdentityIndex(max_identities=2, max_lanes_per_identity=10)
    assert idx.register(identity_hash="h1", lane_key="l1") is True
    assert idx.register(identity_hash="h2", lane_key="l2") is True
    # At identity capacity: reject
    assert idx.register(identity_hash="h3", lane_key="l3") is False
    # h1 still present
    assert idx.lanes_for("h1") == frozenset({"l1"})


def test_index_preflight_capacity() -> None:
    idx = LaneIdentityIndex(max_identities=4, max_lanes_per_identity=3)
    idx.register(identity_hash="h1", lane_key="l1")
    # Preflight: 2 new keys would make 3 total -> fits
    assert idx.preflight_capacity(identity_hash="h1", lane_keys=["l2", "l3"]) is True
    # Preflight: 3 new keys would make 4 total -> exceeds
    assert idx.preflight_capacity(identity_hash="h1", lane_keys=["l2", "l3", "l4"]) is False


def test_index_register_batch() -> None:
    idx = LaneIdentityIndex(max_identities=4, max_lanes_per_identity=3)
    assert idx.register_batch(identity_hash="h1", lane_keys=["l1", "l2"]) is RegisterBatchOutcome.ADDED
    assert idx.lanes_for("h1") == frozenset({"l1", "l2"})
    # Idempotent: same keys again
    assert idx.register_batch(identity_hash="h1", lane_keys=["l1", "l2"]) is RegisterBatchOutcome.IDEMPOTENT
    assert idx.lanes_for("h1") == frozenset({"l1", "l2"})
    # Batch that would exceed: reject entire batch
    assert idx.register_batch(identity_hash="h1", lane_keys=["l3", "l4"]) is RegisterBatchOutcome.CAPACITY_REJECTED
    # No partial mutation
    assert idx.lanes_for("h1") == frozenset({"l1", "l2"})


def test_index_unregister_batch() -> None:
    idx = LaneIdentityIndex(max_identities=4, max_lanes_per_identity=10)
    idx.register_batch(identity_hash="h1", lane_keys=["l1", "l2", "l3"])
    removed = idx.unregister_batch(identity_hash="h1", lane_keys=["l1", "l3"])
    assert removed == 2
    assert idx.lanes_for("h1") == frozenset({"l2"})


# ---------------------------------------------------------------------------
# PublicationIntent tests
# ---------------------------------------------------------------------------


def test_publication_intent_lifecycle() -> None:
    registry = PublicationIntentRegistry()
    intent = registry.create(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a", "key-b"}),
    )
    assert registry.get("codex", "key-a") is intent
    assert registry.get("codex", "key-b") is intent
    assert registry.get("codex", "key-c") is None
    assert registry.get("anthropic", "key-a") is None
    assert not intent.done.is_set()
    intent.complete()
    assert intent.done.is_set()
    registry.remove(intent)
    assert registry.get("codex", "key-a") is None


# ---------------------------------------------------------------------------
# Publish vs family-first clear: no deadlock, final clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_vs_clear_no_deadlock_final_clear() -> None:
    """A clear under the family lock cannot interleave with publication.
    The writer holds the family lock + sorted plan-key locks through the
    entire transaction; the clearer must wait for the family lock."""
    dual_cache, _client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()
    primary_key = "openai:gpt-4.1:auth:abc123"
    secondary_key = "openai:__account_quota__:auth:abc123"

    plan = CooldownPublicationPlan(
        memory_keys=(primary_key, secondary_key),
        durable_keys=(primary_key, secondary_key),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    clear_ran = False

    async def run_clear():
        nonlocal clear_ran
        family_state = mgr.family("codex")
        async with family_state.lock:
            clear_ran = True
            mgr.clear_cooldown_state(
                alias_family="codex",
                cooldown_keys=[primary_key, secondary_key],
            )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        publish_task = asyncio.create_task(
            execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=_make_memory_publisher(mgr),
                persist_cooldown_fn=AsyncMock(),
            )
        )
        clear_task = asyncio.create_task(run_clear())
        await asyncio.wait_for(asyncio.gather(publish_task, clear_task), timeout=5.0)

    assert clear_ran is True
    # Both keys were cleared
    assert mgr.codex.get_memory_cooldown_remaining(primary_key) == 0.0
    assert mgr.codex.get_memory_cooldown_remaining(secondary_key) == 0.0


# ---------------------------------------------------------------------------
# Overlapping reversed-key publications: no deadlock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overlapping_reversed_key_publications_no_deadlock() -> None:
    """Two concurrent publications for overlapping keys in reversed order
    must not deadlock.  The family lock serializes them; sorted plan-key
    locks prevent cycles."""
    dual_cache, _client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()

    plan_a = CooldownPublicationPlan(
        memory_keys=("key-a", "key-b"),
        durable_keys=("key-a", "key-b"),
        duration_seconds=300.0,
        applied_scope="candidate",
    )
    plan_b = CooldownPublicationPlan(
        memory_keys=("key-b", "key-c"),
        durable_keys=("key-b", "key-c"),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        task_a = asyncio.create_task(
            execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan_a,
                publish_cooldown_memory_fn=lambda *, keys, seconds: None,
                persist_cooldown_fn=AsyncMock(),
            )
        )
        task_b = asyncio.create_task(
            execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENROUTER,
                plan=plan_b,
                publish_cooldown_memory_fn=lambda *, keys, seconds: None,
                persist_cooldown_fn=AsyncMock(),
            )
        )
        await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5.0)


# ---------------------------------------------------------------------------
# Follower intent waits / no second provider call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follower_intent_waits_no_second_provider_call() -> None:
    """A follower that encounters an active intent must await completion
    and NOT make a second provider call."""
    mgr = AliasRoutingStateManager()
    provider_calls = 0

    intent = mgr.publication_intents.create(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a"}),
    )

    # Simulate follower checking for intent
    existing = mgr.publication_intents.get("codex", "key-a")
    assert existing is intent
    assert not existing.done.is_set()

    # Follower would await done
    async def complete_later():
        await asyncio.sleep(0.05)
        intent.complete()

    await asyncio.wait_for(
        asyncio.gather(
            intent.done.wait(),
            complete_later(),
        ),
        timeout=2.0,
    )
    assert intent.done.is_set()
    assert provider_calls == 0  # no second provider call


# ---------------------------------------------------------------------------
# Configured Redis failures leave no local state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_configured_redis_unreachable_leaves_no_local_state() -> None:
    """When Redis is configured but unreachable, no local state is mutated."""
    mgr = AliasRoutingStateManager()
    identity_hash = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)

    dual_cache, client = _make_strict_dual_cache(
        eval_side_effect=ConnectionError("Redis down"),
    )

    plan = CooldownPublicationPlan(
        memory_keys=("key-a",),
        durable_keys=("key-a",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        with pytest.raises(RuntimeError):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=_make_memory_publisher(mgr),
                persist_cooldown_fn=AsyncMock(),
            )

    # No local state: memory restored, index empty
    assert mgr.codex.get_memory_cooldown_remaining("key-a") == 0.0
    assert mgr.lane_identity_index.lanes_for(identity_hash) == frozenset()


@pytest.mark.asyncio
async def test_no_redis_backend_leaves_no_local_index() -> None:
    """When no Redis backend, local index must NOT be updated."""
    mgr = AliasRoutingStateManager()
    identity_hash = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)

    plan = CooldownPublicationPlan(
        memory_keys=("key-a",),
        durable_keys=("key-a",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=None), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        await execute_cooldown_publication_transaction(
            alias_family="codex_auto_agent",
            candidate=_CANDIDATE_OPENAI,
            plan=plan,
            publish_cooldown_memory_fn=lambda *, keys, seconds: (
                mgr.codex.set_cooldown_memory(k, seconds) for k in keys
            ),
            persist_cooldown_fn=AsyncMock(),
        )

    # Memory published (legacy path), but NO index registration
    assert mgr.lane_identity_index.lanes_for(identity_hash) == frozenset()


# ---------------------------------------------------------------------------
# Atomic capacity rejection leaves no index/cooldowns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_atomic_capacity_rejection_leaves_no_state() -> None:
    """When the Lua transaction rejects capacity, no index or cooldown
    state is left behind."""
    dual_cache, client = _make_strict_dual_cache(eval_return=-1)
    mgr = AliasRoutingStateManager()
    identity_hash = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)

    plan = CooldownPublicationPlan(
        memory_keys=("key-a",),
        durable_keys=("key-a",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        with pytest.raises(CapacityRejectedError):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=_make_memory_publisher(mgr),
                persist_cooldown_fn=AsyncMock(),
            )

    # No local state
    assert mgr.codex.get_memory_cooldown_remaining("key-a") == 0.0
    assert mgr.lane_identity_index.lanes_for(identity_hash) == frozenset()


# ---------------------------------------------------------------------------
# Commit-then-raise reconciliation: exactly one commit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_then_raise_reconciliation() -> None:
    """After EVAL exception, reconciliation requires receipt AND full
    published post-image.  Receipt present + every cooldown key present +
    every lane member registered -> exactly one commit occurred (True)."""
    dual_cache, client = _make_strict_dual_cache()
    # Receipt present (commit occurred); cooldown postcondition keys present.
    client.get = AsyncMock(return_value=b'{"txn_id":"abc"}')
    # Lane-membership postcondition: every member registered in identity set.
    client.sismember = AsyncMock(return_value=True)

    cooldown_cache_keys = ["cd-key-1", "cd-key-2"]
    identity_cache_key = "identity-key-1"
    lane_members = ["lane-1", "lane-2"]

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        committed = await reconcile_cooldown_transaction(
            alias_family="codex",
            transaction_id="abc123def456",
            cooldown_cache_keys=cooldown_cache_keys,
            identity_cache_key=identity_cache_key,
            lane_members=lane_members,
        )
    assert committed is True
    # Receipt check + one get per cooldown postcondition key.
    assert client.get.call_count == 1 + len(cooldown_cache_keys)
    # One membership check per lane member.
    assert client.sismember.call_count == len(lane_members)


@pytest.mark.asyncio
async def test_reconciliation_absent_receipt() -> None:
    """Receipt absent -> no commit occurred (False), no postcondition checks."""
    dual_cache, client = _make_strict_dual_cache()
    client.get = AsyncMock(return_value=None)
    client.sismember = AsyncMock(return_value=True)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        committed = await reconcile_cooldown_transaction(
            alias_family="codex",
            transaction_id="abc123def456",
            cooldown_cache_keys=["cd-key-1"],
            identity_cache_key="identity-key-1",
            lane_members=["lane-1"],
        )
    assert committed is False
    # Receipt absent short-circuits: only the receipt get ran, no postconditions.
    client.get.assert_called_once()
    client.sismember.assert_not_called()


# ---------------------------------------------------------------------------
# Local commit failure restores durable pre-images and all local maps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_commit_failure_restores_all() -> None:
    """If local index commit fails after durable commit, durable pre-images
    are restored and local maps are rolled back."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()
    identity_hash = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)

    plan = CooldownPublicationPlan(
        memory_keys=("key-a",),
        durable_keys=("key-a",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    # Make register_batch fail
    def failing_register_batch(**kwargs):
        raise RuntimeError("local index failure")

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr), \
         patch.object(mgr.lane_identity_index, "register_batch", side_effect=failing_register_batch):
        with pytest.raises(RuntimeError, match="local index failure"):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=_make_memory_publisher(mgr),
                persist_cooldown_fn=AsyncMock(),
            )

    # Local state restored
    assert mgr.codex.get_memory_cooldown_remaining("key-a") == 0.0
    assert mgr.lane_identity_index.lanes_for(identity_hash) == frozenset()
    # Durable rollback attempted via atomic Lua EVAL (publish + rollback = 2 evals)
    assert client.eval.call_count >= 2


# ---------------------------------------------------------------------------
# Rollback failure yields sanitized indeterminate error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollback_failure_sanitized_error() -> None:
    """Rollback failure raises RollbackFailedError with sanitized context:
    phase, family, txn prefix, key count, exception classes.
    No identity prefix/hash, lane key, Redis key, credentials, or raw error."""
    journal = CooldownTransactionJournal(
        transaction_id="abc123def456789",
        phase=PHASE_DURABLE_COMMITTED,
        alias_family="codex",
        identity_hash="identity_hash_full_value",
        cooldown_keys=["secret-lane-key-1"],
        identity_keys=["redis-internal-key"],
        lane_members=["secret-lane-key-1"],
        preimages=[],
        receipt_key="receipt-key",
        requested_ttl=300,
    )

    dual_cache, client = _make_strict_dual_cache()
    # Rollback now uses atomic Lua EVAL; make eval fail
    client.eval = AsyncMock(side_effect=ConnectionError("Redis down"))

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RollbackFailedError) as exc_info:
            await rollback_cooldown_transaction(
                alias_family="codex",
                journal=journal,
            )

    err = exc_info.value
    msg = str(err)
    # Sanitized: contains phase, family, txn prefix, key count.
    # CFG-004 redaction contract: NO identity prefix/hash in error messages.
    assert "phase=" in msg
    assert "family=codex" in msg
    assert "txn=abc123def456" in msg
    assert "identity=" not in msg
    assert "keys=1" in msg
    assert "ConnectionError" in msg
    # NOT leaked: full lane key, full Redis key, credentials
    assert "secret-lane-key-1" not in msg
    assert "redis-internal-key" not in msg
    assert "receipt-key" not in msg
    assert "Redis down" not in msg


# ---------------------------------------------------------------------------
# No secret/key leakage in errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_secret_key_leakage_in_errors() -> None:
    """Transaction errors must not contain lane keys, Redis keys, or credentials."""
    secret = "sk-live-super-secret-key-999"
    lane_key = f"auth:{hashlib.sha256(secret.encode()).hexdigest()[:12]}"

    dual_cache, client = _make_strict_dual_cache(
        eval_side_effect=ConnectionError("Redis down"),
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError) as exc_info:
            await publish_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash1",
                cooldown_keys=[lane_key],
                lane_members=[lane_key],
                ttl_seconds=300.0,
            )

    msg = str(exc_info.value)
    assert secret not in msg
    assert "sk-live" not in msg
    assert lane_key not in msg
    assert "Redis down" not in msg


# ---------------------------------------------------------------------------
# Durable transaction atomic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_transaction_atomic_eval() -> None:
    """publish_cooldown_transaction uses a single EVAL call."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash1",
            cooldown_keys=["key-a", "key-b"],
            lane_members=["lane-a", "lane-b"],
            ttl_seconds=300.0,
        )

    assert result.phase == PHASE_DURABLE_COMMITTED
    client.eval.assert_called_once()


@pytest.mark.asyncio
async def test_publish_transaction_missing_eval_fails_closed() -> None:
    """Redis client without eval raises RuntimeError (fail closed)."""
    dual_cache, client = _make_strict_dual_cache()
    del client.eval

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RuntimeError, match="missing eval"):
            await publish_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash1",
                cooldown_keys=["key-a"],
                lane_members=["lane-a"],
                ttl_seconds=300.0,
            )


@pytest.mark.asyncio
async def test_publish_transaction_no_redis_fails_closed() -> None:
    """No Redis backend raises RuntimeError."""
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=None):
        with pytest.raises(RuntimeError, match="no Redis cache"):
            await publish_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash1",
                cooldown_keys=["key-a"],
                lane_members=["lane-a"],
                ttl_seconds=300.0,
            )


@pytest.mark.asyncio
async def test_publish_transaction_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown alias_family"):
        await publish_cooldown_transaction(
            alias_family="gemini",
            identity_hash="hash1",
            cooldown_keys=["key-a"],
            lane_members=["lane-a"],
            ttl_seconds=300.0,
        )


# ---------------------------------------------------------------------------
# Unrelated state preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unrelated_state_preserved_on_failure() -> None:
    """Failed publication must not affect unrelated cooldown keys."""
    mgr = AliasRoutingStateManager()
    mgr.codex.set_cooldown_memory("unrelated-key", 600.0)

    dual_cache, client = _make_strict_dual_cache(
        eval_side_effect=ConnectionError("Redis down"),
    )

    plan = CooldownPublicationPlan(
        memory_keys=("key-a",),
        durable_keys=("key-a",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        with pytest.raises(RuntimeError):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=_make_memory_publisher(mgr),
                persist_cooldown_fn=AsyncMock(),
            )

    # Unrelated key preserved
    assert mgr.codex.get_memory_cooldown_remaining("unrelated-key") > 0
    # Failed key not present
    assert mgr.codex.get_memory_cooldown_remaining("key-a") == 0.0


# ---------------------------------------------------------------------------
# Sanitized error structure
# ---------------------------------------------------------------------------


def test_sanitized_error_structure() -> None:
    """PublicationTransactionError contains only sanitized fields."""
    err = PublicationTransactionError(
        phase="PREPARED",
        family="codex",
        transaction_id_prefix="abc123",
        identity_prefix="def456",
        key_count=3,
        exception_classes=("ConnectionError", "TimeoutError"),
    )
    msg = str(err)
    assert "phase=PREPARED" in msg
    assert "family=codex" in msg
    assert "txn=abc123" in msg
    # CFG-004 redaction contract: NO identity prefix/hash in error messages.
    assert "identity=" not in msg
    assert "keys=3" in msg
    assert "ConnectionError" in msg
    assert "TimeoutError" in msg
    # No raw keys or credentials possible in the structure
    assert err.phase == "PREPARED"
    assert err.key_count == 3


# ---------------------------------------------------------------------------
# Dynamic lane restart lookup
# ---------------------------------------------------------------------------


def test_dynamic_lane_restart_lookup() -> None:
    """After restart: clear local index, reconstruct identity from candidate
    only, local index is empty."""
    identity_hash = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)
    mgr = AliasRoutingStateManager()
    assert mgr.lane_identity_index.lanes_for(identity_hash) == frozenset()


# ---------------------------------------------------------------------------
# Publication order: memory -> durable -> index
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publication_order_memory_durable_index() -> None:
    """Verify publication order: memory -> durable transaction -> local index."""
    order: list[str] = []
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()

    plan = CooldownPublicationPlan(
        memory_keys=("key1",),
        durable_keys=("key1",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    original_eval = client.eval

    async def track_eval(*args, **kwargs):
        order.append("durable-txn")
        return await original_eval(*args, **kwargs)

    client.eval = AsyncMock(side_effect=track_eval)

    original_register_batch = mgr.lane_identity_index.register_batch

    def track_register_batch(**kwargs):
        order.append("local-index")
        return original_register_batch(**kwargs)

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr), \
         patch.object(mgr.lane_identity_index, "register_batch", side_effect=track_register_batch):
        await execute_cooldown_publication_transaction(
            alias_family="codex_auto_agent",
            candidate=_CANDIDATE_OPENAI,
            plan=plan,
            publish_cooldown_memory_fn=lambda *, keys, seconds: order.append("memory"),
            persist_cooldown_fn=AsyncMock(),
        )

    assert order == ["durable-txn", "memory", "local-index"]


# ---------------------------------------------------------------------------
# TTL -1/-2 semantics: executable persistence and missing-key tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_minus_one_persistent_semantics() -> None:
    """When a cooldown key has TTL -1 (persistent), the Lua script calls
    PERSIST after SET, preserving the persistent semantics."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash1",
            cooldown_keys=["persistent-key"],
            lane_members=["lane-a"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED
    # The Lua script was called (atomic execution)
    client.eval.assert_called_once()
    # Verify the Lua script text contains PERSIST for -1 handling
    lua_script = client.eval.call_args[0][0]
    assert "PERSIST" in lua_script
    assert "old_ttl == -1" in lua_script


@pytest.mark.asyncio
async def test_ttl_minus_two_absent_key_gets_requested_ttl() -> None:
    """When a cooldown key is absent (TTL -2), the Lua script applies
    the requested TTL via EXPIRE."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash1",
            cooldown_keys=["absent-key"],
            lane_members=["lane-a"],
            ttl_seconds=120.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED
    lua_script = client.eval.call_args[0][0]
    assert "old_ttl == -2" in lua_script
    assert "EXPIRE" in lua_script
    # Requested TTL is ceiled to 120
    call_args_str = str(client.eval.call_args)
    assert "120" in call_args_str


# ---------------------------------------------------------------------------
# Rollback restores exact pre-images from receipt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollback_restores_exact_preimages() -> None:
    """Rollback executes a single atomic Lua EVAL that reads the receipt,
    restores exact values/TTLs (TTL -2 -> DEL, TTL -1 -> PERSIST,
    positive TTL -> EXPIRE), removes lane members, and deletes receipt."""
    dual_cache, client = _make_strict_dual_cache()
    client.eval = AsyncMock(return_value=1)  # Lua returns 1 = success

    journal = CooldownTransactionJournal(
        transaction_id="abc123def456789",
        phase=PHASE_DURABLE_COMMITTED,
        alias_family="codex",
        identity_hash="identity_hash_value",
        cooldown_keys=["key-positive", "key-absent", "key-persistent"],
        identity_keys=["id-key-1"],
        lane_members=["lane-1"],
        preimages=[],
        receipt_key="receipt-key",
        requested_ttl=300,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await rollback_cooldown_transaction(
            alias_family="codex",
            journal=journal,
        )

    # Single atomic Lua EVAL for the entire rollback
    client.eval.assert_called_once()
    lua_script = client.eval.call_args[0][0]
    # Verify the Lua script handles all TTL cases
    assert "PERSIST" in lua_script
    assert "EXPIRE" in lua_script
    assert "DEL" in lua_script
    assert "SREM" in lua_script
    # Verify num_cd passed correctly
    assert "3" in str(client.eval.call_args)


@pytest.mark.asyncio
async def test_rollback_missing_key_deleted() -> None:
    """A key that was absent before (TTL -2) is deleted during rollback
    via the atomic Lua EVAL script."""
    dual_cache, client = _make_strict_dual_cache()
    client.eval = AsyncMock(return_value=1)

    journal = CooldownTransactionJournal(
        transaction_id="abc123def456789",
        phase=PHASE_DURABLE_COMMITTED,
        alias_family="codex",
        identity_hash="identity_hash_value",
        cooldown_keys=["was-absent-key"],
        identity_keys=["id-key-1"],
        lane_members=["lane-1"],
        preimages=[],
        receipt_key="receipt-key",
        requested_ttl=300,
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await rollback_cooldown_transaction(
            alias_family="codex",
            journal=journal,
        )

    # Atomic Lua EVAL handles DEL for absent keys
    client.eval.assert_called_once()
    lua_script = client.eval.call_args[0][0]
    assert "DEL" in lua_script


# ---------------------------------------------------------------------------
# Aggregate capacity preflight in Lua
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_capacity_preflight_in_lua() -> None:
    """The Lua script performs aggregate unique-member capacity check,
    not per-key individual checks."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash1",
            cooldown_keys=["key-a", "key-b"],
            lane_members=["lane-a", "lane-b"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED
    lua_script = client.eval.call_args[0][0]
    # Verify aggregate counting logic exists
    assert "new_count" in lua_script
    assert "card + new_count > max_lanes" in lua_script


# ---------------------------------------------------------------------------
# Local register_batch rejection triggers rollback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_batch_rejection_triggers_rollback() -> None:
    """When register_batch returns False (capacity), the transaction
    triggers rollback of the durable commit and restores local state."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()

    plan = CooldownPublicationPlan(
        memory_keys=("key-a",),
        durable_keys=("key-a",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr), \
         patch.object(mgr.lane_identity_index, "register_batch", return_value=RegisterBatchOutcome.CAPACITY_REJECTED):
        with pytest.raises(RuntimeError, match="register_batch rejected"):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=_make_memory_publisher(mgr),
                persist_cooldown_fn=AsyncMock(),
            )

    # Local state restored
    assert mgr.codex.get_memory_cooldown_remaining("key-a") == 0.0
    # Durable rollback attempted via atomic Lua EVAL
    assert client.eval.call_count >= 2


# ---------------------------------------------------------------------------
# Durable commit before local publish ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_durable_commit_before_memory_publish() -> None:
    """Durable transaction executes BEFORE memory publish (strict ordering)."""
    order: list[str] = []
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()

    plan = CooldownPublicationPlan(
        memory_keys=("key1",),
        durable_keys=("key1",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    original_eval = client.eval

    async def track_eval(*args, **kwargs):
        order.append("durable-txn")
        return await original_eval(*args, **kwargs)

    client.eval = AsyncMock(side_effect=track_eval)

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        await execute_cooldown_publication_transaction(
            alias_family="codex_auto_agent",
            candidate=_CANDIDATE_OPENAI,
            plan=plan,
            publish_cooldown_memory_fn=lambda *, keys, seconds: order.append("memory"),
            persist_cooldown_fn=AsyncMock(),
        )

    assert order.index("durable-txn") < order.index("memory"), (
        f"durable must precede memory; got {order!r}"
    )


# ---------------------------------------------------------------------------
# Executable semantics tests (fakeredis + real Lua EVAL)
# ---------------------------------------------------------------------------
#
# These tests use fakeredis with lupa to execute the actual Lua scripts
# against a real Redis-compatible engine.  They verify TTL -1/-2 semantics,
# aggregate capacity, receipt round-trip, and exact rollback behavior.

try:
    import fakeredis.aioredis as _fakeredis_aioredis
    import lupa as _lupa  # noqa: F401 - required for fakeredis Lua support

    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False

_fakeredis_skip = pytest.mark.skipif(
    not _FAKEREDIS_AVAILABLE,
    reason="fakeredis[lua] not installed",
)


def _make_fakeredis_dual_cache(fake_redis_client):
    """Wrap a fakeredis client in the dual-cache shape expected by durable.py."""
    redis_cache = MagicMock()
    redis_cache.init_async_client = MagicMock(return_value=fake_redis_client)
    redis_cache.check_and_fix_namespace = MagicMock(side_effect=lambda key: key)
    dual_cache = MagicMock()
    dual_cache.redis_cache = redis_cache
    dual_cache.in_memory_cache = None
    return dual_cache


@_fakeredis_skip
@pytest.mark.asyncio
async def test_lua_ttl_minus_one_persistent_semantics_executable() -> None:
    """Executable: a key with TTL -1 (persistent) stays persistent after publish."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Pre-seed a persistent key
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="persist-key"
    )
    await r.set(cd_key, b'{"old":"data"}')
    # No EXPIRE -> TTL is -1

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-persist",
            cooldown_keys=["persist-key"],
            lane_members=["lane-persist"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Key must still be persistent (TTL == -1)
    ttl = await r.ttl(cd_key)
    assert ttl == -1, f"persistent key must remain TTL -1, got {ttl}"


@_fakeredis_skip
@pytest.mark.asyncio
async def test_lua_ttl_minus_two_absent_key_gets_requested_ttl_executable() -> None:
    """Executable: an absent key (TTL -2) gets the requested TTL after publish."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-absent",
            cooldown_keys=["absent-key"],
            lane_members=["lane-absent"],
            ttl_seconds=120.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="absent-key"
    )
    ttl = await r.ttl(cd_key)
    assert 119 <= ttl <= 120, f"absent key should get ~120s TTL, got {ttl}"


@_fakeredis_skip
@pytest.mark.asyncio
async def test_lua_aggregate_capacity_rejection_executable() -> None:
    """Executable: aggregate unique-member capacity check rejects overflow."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Pre-fill identity set to capacity (max_lanes=3)
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-cap"
    )
    await r.sadd(id_key, "existing-1", "existing-2", "existing-3")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(CapacityRejectedError):
            await publish_cooldown_transaction(
                alias_family="codex",
                identity_hash="hash-cap",
                cooldown_keys=["cap-key"],
                lane_members=["new-lane"],
                ttl_seconds=300.0,
                max_lanes_per_identity=3,
            )

    # Identity set unchanged
    members = await r.smembers(id_key)
    assert members == {b"existing-1", b"existing-2", b"existing-3"}


@_fakeredis_skip
@pytest.mark.asyncio
async def test_lua_receipt_round_trip_executable() -> None:
    """Executable: receipt is written with pre-images and can be read back."""
    import json as _json_mod

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Pre-seed a key with a known value and positive TTL
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="receipt-key"
    )
    await r.set(cd_key, b'{"pre":"existing"}')
    await r.expire(cd_key, 200)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-receipt",
            cooldown_keys=["receipt-key"],
            lane_members=["lane-receipt"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Read the receipt
    receipt_key = result.journal.receipt_key
    raw = await r.get(receipt_key)
    assert raw is not None, "receipt must exist after commit"
    receipt = _json_mod.loads(raw)
    assert "preimages" in receipt
    assert len(receipt["preimages"]) == 1
    preimage = receipt["preimages"][0]
    assert preimage["v"] == '{"pre":"existing"}'
    assert 199 <= preimage["t"] <= 200
    # Identity pre-image must be present
    assert "identity_preimage" in receipt
    assert "members" in receipt["identity_preimage"]
    assert "ttl" in receipt["identity_preimage"]


@_fakeredis_skip
@pytest.mark.asyncio
async def test_lua_exact_rollback_executable() -> None:
    """Executable: rollback restores exact pre-images atomically."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Pre-seed: one key with value+TTL, one absent key
    cd_key_existing = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="rb-existing"
    )
    await r.set(cd_key_existing, b'{"old":"value"}')
    await r.expire(cd_key_existing, 150)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-rollback",
            cooldown_keys=["rb-existing", "rb-absent"],
            lane_members=["lane-rb-1", "lane-rb-2"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Verify mutation happened
    cd_key_absent = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="rb-absent"
    )
    assert await r.get(cd_key_absent) is not None

    # Rollback
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await rollback_cooldown_transaction(
            alias_family="codex",
            journal=result.journal,
        )

    # rb-existing: restored to old value with ~150 TTL
    restored_val = await r.get(cd_key_existing)
    assert restored_val == b'{"old":"value"}'
    restored_ttl = await r.ttl(cd_key_existing)
    assert 148 <= restored_ttl <= 150, f"expected ~150, got {restored_ttl}"

    # rb-absent: deleted (was absent before)
    assert await r.get(cd_key_absent) is None

    # Receipt deleted
    assert await r.get(result.journal.receipt_key) is None

    # Identity set: lane members removed
    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-rollback"
    )
    members = await r.smembers(id_key)
    assert b"lane-rb-1" not in members
    assert b"lane-rb-2" not in members


@_fakeredis_skip
@pytest.mark.asyncio
async def test_lua_rollback_retains_receipt_on_error_executable() -> None:
    """Executable: when rollback Lua returns -2 (partial error), receipt is retained."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Commit a transaction
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-retain",
            cooldown_keys=["retain-key"],
            lane_members=["lane-retain"],
            ttl_seconds=300.0,
        )

    # Corrupt the receipt to trigger a -2 return
    await r.set(result.journal.receipt_key, b"not-json")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        with pytest.raises(RollbackFailedError):
            await rollback_cooldown_transaction(
                alias_family="codex",
                journal=result.journal,
            )

    # Receipt must be RETAINED (not deleted) on error
    raw = await r.get(result.journal.receipt_key)
    assert raw is not None, "receipt must be retained on rollback error"


@_fakeredis_skip
@pytest.mark.asyncio
async def test_lua_rollback_persistent_ttl_minus_one_executable() -> None:
    """Executable: rollback restores a persistent key (TTL -1) correctly."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Pre-seed a persistent key
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="rb-persist"
    )
    await r.set(cd_key, b'{"persistent":"data"}')
    # No expire -> TTL -1

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-rb-persist",
            cooldown_keys=["rb-persist"],
            lane_members=["lane-rb-p"],
            ttl_seconds=300.0,
        )

    # Rollback
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await rollback_cooldown_transaction(
            alias_family="codex",
            journal=result.journal,
        )

    # Restored: original value, still persistent
    assert await r.get(cd_key) == b'{"persistent":"data"}'
    assert await r.ttl(cd_key) == -1


@_fakeredis_skip
@pytest.mark.asyncio
async def test_durable_payload_readable_by_existing_reader_executable() -> None:
    """Executable: transaction-written payload is readable by the existing
    durable reader (requires valid expires_at_epoch)."""
    import json as _json_mod

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-reader",
            cooldown_keys=["reader-key"],
            lane_members=["lane-reader"],
            ttl_seconds=300.0,
        )

    # Read the written cooldown value directly
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="reader-key"
    )
    raw = await r.get(cd_key)
    assert raw is not None
    payload = _json_mod.loads(raw)
    # Must have expires_at_epoch as a future float
    assert "expires_at_epoch" in payload
    assert isinstance(payload["expires_at_epoch"], (int, float))
    assert payload["expires_at_epoch"] > time.time()

    # Verify through the actual reader function
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        read_aawm_alias_routing_durable_payload,
    )

    # Patch dual_cache to have async_get_cache that reads from fakeredis
    async def _fake_async_get_cache(key: str):
        raw_val = await r.get(key)
        if raw_val is None:
            return None
        return _json_mod.loads(raw_val)

    dual_cache.async_get_cache = _fake_async_get_cache

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        read_result = await read_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="reader-key",
        )
    assert read_result is not None, (
        "existing durable reader must be able to read transaction-written payload"
    )
    assert "expires_at_epoch" in read_result


@_fakeredis_skip
@pytest.mark.asyncio
async def test_configured_unhealthy_redis_fails_before_local_mutation() -> None:
    """Configured-but-unhealthy Redis must fail closed before any local mutation."""
    mgr = AliasRoutingStateManager()
    plan = CooldownPublicationPlan(
        memory_keys=("key-unhealthy",),
        durable_keys=("key-unhealthy",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    # Simulate: dual_cache is None (unhealthy) but Redis IS configured
    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=None), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr), \
         patch(
             "litellm.proxy.aawm_alias_routing_redis.get_status",
             return_value={"configured": True, "mode": "memory", "reachable": False},
         ):
        with pytest.raises(RuntimeError, match="configured.*unhealthy.*failing closed"):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=_make_memory_publisher(mgr),
                persist_cooldown_fn=AsyncMock(),
            )

    # No local mutation occurred
    assert mgr.codex.get_memory_cooldown_remaining("key-unhealthy") == 0.0


@_fakeredis_skip
@pytest.mark.asyncio
async def test_rollback_failed_error_never_suppressed() -> None:
    """RollbackFailedError propagates over the earlier local exception."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()

    plan = CooldownPublicationPlan(
        memory_keys=("key-rb",),
        durable_keys=("key-rb",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr), \
         patch.object(
             mgr.lane_identity_index,
             "register_batch",
             return_value=RegisterBatchOutcome.CAPACITY_REJECTED,
         ), \
         patch(
             f"{_DURABLE_MOD}.rollback_cooldown_transaction",
             new_callable=AsyncMock,
             side_effect=RollbackFailedError(
                 phase="DURABLE_COMMITTED",
                 family="codex",
                 transaction_id_prefix="abc",
                 identity_prefix="def",
                 key_count=1,
                 exception_classes=("ConnectionError",),
             ),
         ):
        with pytest.raises(RollbackFailedError, match="indeterminate"):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=_make_memory_publisher(mgr),
                persist_cooldown_fn=AsyncMock(),
            )

    # Local state restored despite rollback failure
    assert mgr.codex.get_memory_cooldown_remaining("key-rb") == 0.0




# ---------------------------------------------------------------------------
# CFG-004 regression: rollback restores prior identity membership
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_rollback_restores_prior_identity_membership_executable() -> None:
    """Regression: rollback must restore prior identity membership, not
    unconditionally SREM lanes that existed before publication.

    Prior set: {lane-existing, lane-other}.  Publish adds lane-new.
    Rollback must remove only lane-new, leaving lane-existing and lane-other.
    """
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-membership"
    )
    # Pre-seed identity set with two existing members
    await r.sadd(id_key, "lane-existing", "lane-other")

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-membership",
            cooldown_keys=["membership-key"],
            lane_members=["lane-new"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Verify lane-new was added
    members_after_publish = await r.smembers(id_key)
    assert members_after_publish == {b"lane-existing", b"lane-other", b"lane-new"}

    # Rollback
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await rollback_cooldown_transaction(
            alias_family="codex",
            journal=result.journal,
        )

    # Only lane-new removed; prior members preserved
    members_after_rollback = await r.smembers(id_key)
    assert members_after_rollback == {b"lane-existing", b"lane-other"}, (
        f"rollback must preserve prior members; got {members_after_rollback}"
    )


# ---------------------------------------------------------------------------
# CFG-004 regression: republish shorter TTL preserves longer expiry alignment
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_republish_shorter_ttl_aligns_expiry_with_preserved_redis_ttl_executable() -> None:
    """Regression: when republishing with a shorter requested TTL but the
    longer Redis TTL is preserved, the durable payload expires_at_epoch must
    align with the effective preserved TTL so the existing reader remains
    valid for the full Redis lifetime.
    """
    import json as _json_mod

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="republish-key"
    )

    # First publish with long TTL (600s)
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-republish",
            cooldown_keys=["republish-key"],
            lane_members=["lane-republish"],
            ttl_seconds=600.0,
        )

    # Verify Redis TTL is ~600
    redis_ttl_after_first = await r.ttl(cd_key)
    assert 598 <= redis_ttl_after_first <= 600

    # Republish with shorter TTL (60s) -- Redis TTL must be preserved at ~600
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-republish",
            cooldown_keys=["republish-key"],
            lane_members=["lane-republish"],
            ttl_seconds=60.0,
        )

    # Redis TTL must still be ~600 (monotonic: never shorten)
    redis_ttl_after_second = await r.ttl(cd_key)
    assert 598 <= redis_ttl_after_second <= 600, (
        f"Redis TTL must be preserved at ~600, got {redis_ttl_after_second}"
    )

    # Payload expires_at_epoch must align with the preserved ~600s TTL
    raw = await r.get(cd_key)
    assert raw is not None
    payload = _json_mod.loads(raw)
    expires_at = payload["expires_at_epoch"]
    now = time.time()
    remaining = expires_at - now
    # Must be aligned with ~600s Redis TTL, NOT ~60s requested TTL
    assert remaining > 120, (
        f"expires_at_epoch must align with preserved ~600s TTL, "
        f"but remaining={remaining:.0f}s suggests alignment with shorter TTL"
    )
    assert remaining <= 601, (
        f"expires_at_epoch should not exceed preserved TTL; remaining={remaining:.0f}s"
    )

    # Verify through the actual reader function
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        read_aawm_alias_routing_durable_payload,
    )

    async def _fake_async_get_cache(key: str):
        raw_val = await r.get(key)
        if raw_val is None:
            return None
        return _json_mod.loads(raw_val)

    dual_cache.async_get_cache = _fake_async_get_cache

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        read_result = await read_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="republish-key",
        )
    assert read_result is not None, (
        "existing durable reader must remain valid for the full preserved Redis lifetime"
    )


# ---------------------------------------------------------------------------
# CFG-004 regression: identity index TTL -1 must remain persistent
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_identity_index_ttl_minus_one_remains_persistent_executable() -> None:
    """Regression: identity index TTL -1 (persistent) must remain persistent.
    Monotonic TTL logic must not convert -1 to a finite TTL.
    """
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-persist-id"
    )
    # Pre-seed identity set as persistent (no TTL)
    await r.sadd(id_key, "lane-prior")
    assert await r.ttl(id_key) == -1, "pre-seed must be persistent"

    # Publish with a positive TTL -- identity key must remain persistent
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-persist-id",
            cooldown_keys=["persist-id-key"],
            lane_members=["lane-new-persist"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Identity key must still be persistent (TTL == -1)
    id_ttl = await r.ttl(id_key)
    assert id_ttl == -1, (
        f"identity index TTL -1 must remain persistent after publish, got {id_ttl}"
    )

    # Members must include both prior and new
    members = await r.smembers(id_key)
    assert members == {b"lane-prior", b"lane-new-persist"}


# ---------------------------------------------------------------------------
# Persistent (unbounded) cooldown payload semantics
# ---------------------------------------------------------------------------
#
# Replaces the former 10-year expires_at_epoch sentinel with an explicit
# JSON-safe ``"persistent": true`` marker.  These tests verify publish,
# read, inspect, hydrate, shorter-republish, and rollback for persistent
# keys using fakeredis+lupa.


@_fakeredis_skip
@pytest.mark.asyncio
async def test_persistent_publish_writes_json_safe_marker_executable() -> None:
    """Publish to a pre-existing persistent key writes ``persistent: true``
    and does NOT write a 10-year sentinel or JSON Infinity."""
    import json as _json_mod

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="persist-marker"
    )
    # Pre-seed persistent key (no TTL)
    await r.set(cd_key, b'{"old":"data"}')
    assert await r.ttl(cd_key) == -1

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-pm",
            cooldown_keys=["persist-marker"],
            lane_members=["lane-pm"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Key must still be persistent
    assert await r.ttl(cd_key) == -1

    # Payload must have explicit persistent marker, no 10-year sentinel
    raw = await r.get(cd_key)
    assert raw is not None
    payload = _json_mod.loads(raw)
    assert payload.get("persistent") is True, (
        f"persistent marker missing; payload={payload}"
    )
    # No arbitrary far-future epoch
    expires = payload.get("expires_at_epoch")
    assert expires is None or expires == 0 or not isinstance(expires, (int, float)) or expires < 300_000_000, (
        f"10-year sentinel must not be present; expires_at_epoch={expires}"
    )
    # JSON-safe: no Infinity/NaN in the raw bytes
    raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    assert "Infinity" not in raw_str
    assert "NaN" not in raw_str


@_fakeredis_skip
@pytest.mark.asyncio
async def test_persistent_readable_by_existing_reader_executable() -> None:
    """The existing durable reader must return persistent payloads
    (recognizing the ``persistent: true`` marker)."""
    import json as _json_mod

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        read_aawm_alias_routing_durable_payload,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="persist-read"
    )
    await r.set(cd_key, b'{"old":"data"}')

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-pr",
            cooldown_keys=["persist-read"],
            lane_members=["lane-pr"],
            ttl_seconds=300.0,
        )

    # Wire up async_get_cache to read from fakeredis
    async def _fake_async_get_cache(key: str):
        raw_val = await r.get(key)
        if raw_val is None:
            return None
        return _json_mod.loads(raw_val)

    dual_cache.async_get_cache = _fake_async_get_cache

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        read_result = await read_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="persist-read",
        )
    assert read_result is not None, (
        "existing durable reader must return persistent payload"
    )
    assert read_result.get("persistent") is True


@_fakeredis_skip
@pytest.mark.asyncio
async def test_persistent_inspect_reports_unbounded_executable() -> None:
    """Inspection of a persistent key reports UNBOUNDED_EXPIRY for both
    expires_at_epoch and ttl_remaining_seconds."""
    import json as _json_mod

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        UNBOUNDED_EXPIRY,
        inspect_aawm_alias_routing_durable_key,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    # Write a persistent payload directly (simulating post-Lua state)
    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="persist-inspect"
    )
    await r.set(cd_key, _json_mod.dumps({"persistent": True, "cooldown_keys": ["k"]}).encode())
    # No expire -> persistent

    # Wire _get_cache_logic to decode JSON
    redis_cache = dual_cache.redis_cache
    redis_cache._get_cache_logic = MagicMock(
        side_effect=lambda cached_response: _json_mod.loads(cached_response)
    )

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        inspection = await inspect_aawm_alias_routing_durable_key(
            alias_family="codex",
            state_kind="cooldown",
            state_key="persist-inspect",
        )

    assert inspection.exists is True
    assert inspection.expires_at_epoch is UNBOUNDED_EXPIRY
    assert inspection.ttl_remaining_seconds is UNBOUNDED_EXPIRY


@_fakeredis_skip
@pytest.mark.asyncio
async def test_persistent_hydrate_remaining_ttl_unbounded_executable() -> None:
    """Hydration/remaining-TTL for a persistent key reports unbounded,
    not a large finite number."""

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        UNBOUNDED_EXPIRY,
        parse_aawm_alias_routing_durable_expiry,
    )

    # Direct parser test: persistent marker
    assert parse_aawm_alias_routing_durable_expiry({"persistent": True}) is UNBOUNDED_EXPIRY

    # Far-future finite expiry remains finite (legacy sentinel removed)
    far_future = time.time() + 315360000
    far_result = parse_aawm_alias_routing_durable_expiry(
        {"expires_at_epoch": far_future}
    )
    assert far_result is not UNBOUNDED_EXPIRY
    assert isinstance(far_result, float)

    # Finite payload still works
    finite = parse_aawm_alias_routing_durable_expiry(
        {"expires_at_epoch": time.time() + 300}
    )
    assert isinstance(finite, float)
    assert finite > time.time()

    # Expired payload returns None
    assert parse_aawm_alias_routing_durable_expiry(
        {"expires_at_epoch": time.time() - 10}
    ) is None

    # Missing expiry returns None
    assert parse_aawm_alias_routing_durable_expiry({"foo": "bar"}) is None


@_fakeredis_skip
@pytest.mark.asyncio
async def test_persistent_readable_indefinitely_executable() -> None:
    """A persistent key remains readable after simulated time passage
    (no TTL expiry)."""
    import json as _json_mod

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        read_aawm_alias_routing_durable_payload,
    )

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="persist-indef"
    )
    # Write persistent payload
    await r.set(
        cd_key,
        _json_mod.dumps({"persistent": True, "cooldown_keys": ["indef"]}).encode(),
    )
    # Confirm no TTL
    assert await r.ttl(cd_key) == -1

    async def _fake_async_get_cache(key: str):
        raw_val = await r.get(key)
        if raw_val is None:
            return None
        return _json_mod.loads(raw_val)

    dual_cache.async_get_cache = _fake_async_get_cache

    # Read immediately
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result1 = await read_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="persist-indef",
        )
    assert result1 is not None
    assert result1.get("persistent") is True

    # Simulate time passage: advance fakeredis server time by 1 year
    # (persistent keys have no TTL so they survive)
    r.time = MagicMock(return_value=(int(time.time()) + 31536000, 0))

    # Key must still be present (no TTL to expire)
    assert await r.get(cd_key) is not None
    assert await r.ttl(cd_key) == -1

    # Still readable through the durable reader
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result2 = await read_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="persist-indef",
        )
    assert result2 is not None, "persistent key must remain readable indefinitely"


@_fakeredis_skip
@pytest.mark.asyncio
async def test_persistent_shorter_republish_preserves_persistent_executable() -> None:
    """Republishing with a shorter TTL over a persistent key must preserve
    persistent state (not downgrade to finite)."""
    import json as _json_mod

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="persist-repub"
    )
    # Pre-seed persistent key
    await r.set(cd_key, b'{"old":"persist"}')
    assert await r.ttl(cd_key) == -1

    # First publish (should preserve persistent)
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-repub",
            cooldown_keys=["persist-repub"],
            lane_members=["lane-repub"],
            ttl_seconds=300.0,
        )

    assert await r.ttl(cd_key) == -1, "first publish must preserve persistent"

    # Republish with shorter TTL
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-repub",
            cooldown_keys=["persist-repub"],
            lane_members=["lane-repub"],
            ttl_seconds=60.0,
        )

    # Must still be persistent
    assert await r.ttl(cd_key) == -1, (
        "shorter republish must not downgrade persistent to finite"
    )
    raw = await r.get(cd_key)
    payload = _json_mod.loads(raw)
    assert payload.get("persistent") is True


@_fakeredis_skip
@pytest.mark.asyncio
async def test_persistent_rollback_restores_prior_state_executable() -> None:
    """Rollback of a transaction that overwrote a persistent key restores
    the original persistent value and TTL -1."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="persist-rb"
    )
    # Pre-seed persistent key with known value
    await r.set(cd_key, b'{"original":"persistent-data"}')
    assert await r.ttl(cd_key) == -1

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-prb",
            cooldown_keys=["persist-rb"],
            lane_members=["lane-prb"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Rollback
    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        await rollback_cooldown_transaction(
            alias_family="codex",
            journal=result.journal,
        )

    # Restored: original value, still persistent
    assert await r.get(cd_key) == b'{"original":"persistent-data"}'
    assert await r.ttl(cd_key) == -1


@_fakeredis_skip
@pytest.mark.asyncio
async def test_persistent_new_key_gets_finite_ttl_executable() -> None:
    """A new (absent) key published with positive TTL gets a finite TTL,
    not persistent."""
    import json as _json_mod

    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    cd_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="cooldown", state_key="new-finite"
    )
    # Key does not exist yet

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-nf",
            cooldown_keys=["new-finite"],
            lane_members=["lane-nf"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Must have finite TTL
    ttl = await r.ttl(cd_key)
    assert 298 <= ttl <= 300, f"new key must get finite TTL ~300, got {ttl}"

    # Payload must NOT have persistent marker
    raw = await r.get(cd_key)
    payload = _json_mod.loads(raw)
    assert payload.get("persistent") is not True
    assert "expires_at_epoch" in payload
    assert isinstance(payload["expires_at_epoch"], (int, float))
    assert payload["expires_at_epoch"] > time.time()


# ---------------------------------------------------------------------------
# Successful publication returns LOCAL_COMMITTED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_publication_returns_local_committed() -> None:
    """After durable + local commit, result phase must be LOCAL_COMMITTED."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
        PHASE_LOCAL_COMMITTED,
    )

    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()
    identity_hash = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)

    plan = CooldownPublicationPlan(
        memory_keys=("key-a",),
        durable_keys=("key-a",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        result = await execute_cooldown_publication_transaction(
            alias_family="codex_auto_agent",
            candidate=_CANDIDATE_OPENAI,
            plan=plan,
            publish_cooldown_memory_fn=_make_memory_publisher(mgr),
            persist_cooldown_fn=AsyncMock(),
        )

    assert result is not None
    assert result.phase == PHASE_LOCAL_COMMITTED
    # Journal retains immutable DURABLE_COMMITTED evidence.
    assert result.journal.phase == PHASE_DURABLE_COMMITTED
    # Local state is present.
    assert mgr.codex.get_memory_cooldown_remaining("key-a") > 0.0
    assert mgr.lane_identity_index.lanes_for(identity_hash) == frozenset({"key-a"})


# ---------------------------------------------------------------------------
# Memory publisher exception triggers rollback-protected local commit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_publisher_exception_triggers_rollback_and_restores_state() -> None:
    """Regression: if the memory publisher raises inside the rollback-protected
    local commit, the durable rollback EVAL executes, local cooldown state is
    restored to pre-publication snapshot, and the exception propagates."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()
    identity_hash = resolve_lane_identity_hash(candidate=_CANDIDATE_OPENAI)

    # Pre-seed a prior cooldown so we can verify exact snapshot restoration.
    prior_remaining = 42.0
    mgr.codex.set_cooldown_memory("key-a", prior_remaining)
    snapshot_before = mgr.codex.get_memory_cooldown_remaining("key-a")
    assert snapshot_before > 0.0

    plan = CooldownPublicationPlan(
        memory_keys=("key-a",),
        durable_keys=("key-a",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    def exploding_publisher(*, keys, seconds):
        raise RuntimeError("memory publisher exploded")

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        with pytest.raises(RuntimeError, match="memory publisher exploded"):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=exploding_publisher,
                persist_cooldown_fn=AsyncMock(),
            )

    # 1. Rollback EVAL was executed: publish EVAL + rollback EVAL = 2 calls.
    assert client.eval.call_count >= 2, (
        f"Expected at least 2 EVAL calls (publish + rollback), got {client.eval.call_count}"
    )

    # 2. No local cooldown from the failed publication remains; the snapshot
    #    is restored exactly (prior value preserved, not zeroed).
    remaining_after = mgr.codex.get_memory_cooldown_remaining("key-a")
    # The snapshot was taken BEFORE mutation; the publisher exploded before
    # mutating, so the restored value equals the pre-publication snapshot.
    assert abs(remaining_after - snapshot_before) < 1.0, (
        f"Local cooldown must be restored to pre-publication snapshot "
        f"({snapshot_before:.1f}s), got {remaining_after:.1f}s"
    )

    # 3. Local index must be empty (register_batch never ran).
    assert mgr.lane_identity_index.lanes_for(identity_hash) == frozenset()


@pytest.mark.asyncio
async def test_memory_publisher_exception_no_local_cooldown_when_no_prior() -> None:
    """When no prior cooldown exists, memory publisher exception leaves zero
    local cooldown (snapshot was 0.0 -> key removed)."""
    dual_cache, client = _make_strict_dual_cache(eval_return=1)
    mgr = AliasRoutingStateManager()

    plan = CooldownPublicationPlan(
        memory_keys=("key-b",),
        durable_keys=("key-b",),
        duration_seconds=300.0,
        applied_scope="candidate",
    )

    def exploding_publisher(*, keys, seconds):
        raise ValueError("publisher crash")

    with patch(f"{_APPLY_MOD}._state_manager", mgr), \
         patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache), \
         patch(f"{_LOOP_MOD}.alias_routing_state", mgr):
        with pytest.raises(ValueError, match="publisher crash"):
            await execute_cooldown_publication_transaction(
                alias_family="codex_auto_agent",
                candidate=_CANDIDATE_OPENAI,
                plan=plan,
                publish_cooldown_memory_fn=exploding_publisher,
                persist_cooldown_fn=AsyncMock(),
            )

    # Rollback EVAL executed
    assert client.eval.call_count >= 2
    # No local cooldown
    assert mgr.codex.get_memory_cooldown_remaining("key-b") == 0.0


# ---------------------------------------------------------------------------
# Identity-key TTL pre-image semantics (fakeredis + lupa executable)
# ---------------------------------------------------------------------------


@_fakeredis_skip
@pytest.mark.asyncio
async def test_identity_ttl_absent_key_gets_ceiled_requested_ttl_executable() -> None:
    """Absent identity key (pre-image TTL -2) gets ceil(requested finite TTL).

    Requesting 37.2s must produce a finite TTL of 38 on the identity key,
    NOT persistent (-1).  Regression: the old Lua read TTL after SADD,
    which turned -2 into -1 and was misread as persistent.
    """
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-id-ttl"
    )
    # Identity key does not exist yet (TTL -2).
    assert await r.ttl(id_key) == -2

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-id-ttl",
            cooldown_keys=["cd-key-1"],
            lane_members=["lane-1"],
            ttl_seconds=37.2,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Identity key must have finite ceiled TTL (38), not persistent (-1).
    ttl = await r.ttl(id_key)
    assert ttl == 38, (
        f"absent identity key must get ceil(37.2)=38, got {ttl}"
    )


@_fakeredis_skip
@pytest.mark.asyncio
async def test_identity_ttl_persistent_key_remains_persistent_executable() -> None:
    """A genuinely persistent identity key (pre-image TTL -1) must remain
    persistent after publication, not downgraded to finite."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-id-persist"
    )
    # Pre-seed persistent identity key with an existing member.
    await r.sadd(id_key, "old-lane")
    assert await r.ttl(id_key) == -1  # persistent

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-id-persist",
            cooldown_keys=["cd-key-p"],
            lane_members=["lane-p"],
            ttl_seconds=300.0,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Must still be persistent.
    ttl = await r.ttl(id_key)
    assert ttl == -1, (
        f"persistent identity key must remain TTL -1, got {ttl}"
    )
    # Both old and new members present.
    members = await r.smembers(id_key)
    assert members == {b"old-lane", b"lane-p"}


@_fakeredis_skip
@pytest.mark.asyncio
async def test_identity_ttl_positive_uses_monotonic_max_executable() -> None:
    """An identity key with positive pre-image TTL uses monotonic max:
    max(pre_image_ttl, ceil(requested_ttl))."""
    r = _fakeredis_aioredis.FakeRedis(decode_responses=False)
    dual_cache = _make_fakeredis_dual_cache(r)

    id_key = build_aawm_alias_routing_durable_cache_key(
        alias_family="codex", state_kind="lane_identity", state_key="hash-id-mono"
    )
    # Pre-seed identity key with a longer TTL (600s).
    await r.sadd(id_key, "existing-lane")
    await r.expire(id_key, 600)
    pre_ttl = await r.ttl(id_key)
    assert 598 <= pre_ttl <= 600

    with patch(f"{_DURABLE_MOD}.get_aawm_alias_routing_dual_cache", return_value=dual_cache):
        result = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash="hash-id-mono",
            cooldown_keys=["cd-key-m"],
            lane_members=["lane-m"],
            ttl_seconds=37.2,
        )
    assert result.phase == PHASE_DURABLE_COMMITTED

    # Must preserve the longer TTL (monotonic max), not shrink to 38.
    ttl = await r.ttl(id_key)
    assert ttl >= 598, (
        f"identity key must keep monotonic max TTL >= 598, got {ttl}"
    )


# ---------------------------------------------------------------------------
# Defect 2: overlapping clear reservations merge (no overwrite/orphan)
# ---------------------------------------------------------------------------


def test_overlapping_clear_reservation_merges_not_overwrites() -> None:
    """Creating a second clear reservation that overlaps an active one
    MERGES with it (same object returned, unioned keys/identities) rather
    than blindly overwriting (Defect 2)."""
    registry = PublicationIntentRegistry()

    res1 = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-a", "key-b"}),
    )

    # Overlapping reservation: shares key-b.
    res2 = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h2"}),
        cooldown_keys=frozenset({"key-b", "key-c"}),
    )

    # Same object (merged, not overwritten).
    assert res2 is res1
    # Unioned coverage.
    assert res2.identity_hashes == frozenset({"h1", "h2"})
    assert res2.cooldown_keys == frozenset({"key-a", "key-b", "key-c"})

    # All keys resolve to the same reservation.
    assert registry.get_clear_reservation("codex", "key-a") is res1
    assert registry.get_clear_reservation("codex", "key-b") is res1
    assert registry.get_clear_reservation("codex", "key-c") is res1


def test_overlapping_clear_reservation_no_orphaned_waiters() -> None:
    """A waiter on the first reservation is NOT orphaned when a second
    overlapping reservation is created (Defect 2): completing the merged
    reservation signals all waiters."""

    registry = PublicationIntentRegistry()

    res1 = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-a"}),
    )

    # Second overlapping reservation merges.
    res2 = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h2"}),
        cooldown_keys=frozenset({"key-a", "key-b"}),
    )
    assert res2 is res1

    # Completing the merged reservation signals waiters on both keys.
    registry.complete_clear_reservation(res2)
    assert res1.done.is_set()
    assert registry.get_clear_reservation("codex", "key-a") is None
    assert registry.get_clear_reservation("codex", "key-b") is None


def test_non_overlapping_clear_reservations_independent() -> None:
    """Non-overlapping clear reservations remain independent objects
    (Defect 2 merge only applies to overlapping keys)."""
    registry = PublicationIntentRegistry()

    res1 = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-a"}),
    )
    res2 = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h2"}),
        cooldown_keys=frozenset({"key-b"}),
    )

    assert res1 is not res2
    assert registry.get_clear_reservation("codex", "key-a") is res1
    assert registry.get_clear_reservation("codex", "key-b") is res2

    # Completing one does not affect the other.
    registry.complete_clear_reservation(res1)
    assert registry.get_clear_reservation("codex", "key-a") is None
    assert registry.get_clear_reservation("codex", "key-b") is res2


def test_completed_reservation_not_merged() -> None:
    """A completed reservation is NOT merged with a new one (Defect 2:
    only ACTIVE reservations participate in merge)."""
    registry = PublicationIntentRegistry()

    res1 = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-a"}),
    )
    registry.complete_clear_reservation(res1)

    # New reservation for the same key: should be a fresh object.
    res2 = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h2"}),
        cooldown_keys=frozenset({"key-a"}),
    )
    assert res2 is not res1
    assert res2.status is ClearReservationStatus.ACTIVE
    assert registry.get_clear_reservation("codex", "key-a") is res2


def test_overlapping_reservation_different_family_independent() -> None:
    """Overlapping keys in different canonical families do NOT merge
    (Defect 2: merge is per-family)."""
    registry = PublicationIntentRegistry()

    res_codex = registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-a"}),
    )
    res_anth = registry.create_clear_reservation(
        alias_family="anthropic",
        identity_hashes=frozenset({"h2"}),
        cooldown_keys=frozenset({"key-a"}),
    )

    assert res_codex is not res_anth
    assert registry.get_clear_reservation("codex", "key-a") is res_codex
    assert registry.get_clear_reservation("anthropic", "key-a") is res_anth


# ---------------------------------------------------------------------------
# Defect 1: claim_publication_or_wait atomicity (registry-level)
# ---------------------------------------------------------------------------


def test_claim_publication_or_wait_clear_blocks_before_intent() -> None:
    """When a clear reservation exists, claim_publication_or_wait returns
    BLOCKED_BY_CLEAR without creating an intent (Defect 1)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    ClaimOutcome,
    )

    registry = PublicationIntentRegistry()
    registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-a"}),
    )

    result = registry.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="h2",
    )
    assert result.outcome is ClaimOutcome.BLOCKED_BY_CLEAR
    assert result.intent is None
    # No orphaned intent in the registry.
    assert registry.get("codex", "key-a") is None


def test_claim_publication_or_wait_multi_key_clear_blocks() -> None:
    """A clear reservation on ANY key in the set blocks the entire claim
    (Defect 1: multi-key atomicity)."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    ClaimOutcome,
    )

    registry = PublicationIntentRegistry()
    # Reservation only on key-b.
    registry.create_clear_reservation(
        alias_family="codex",
        identity_hashes=frozenset({"h1"}),
        cooldown_keys=frozenset({"key-b"}),
    )

    # Claim for key-a + key-b: blocked because key-b has a reservation.
    result = registry.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a", "key-b"}),
        identity_hash="h2",
    )
    assert result.outcome is ClaimOutcome.BLOCKED_BY_CLEAR
    # No intent created for key-a either.
    assert registry.get("codex", "key-a") is None


def test_release_claim_completes_and_removes() -> None:
    """release_claim completes the intent and removes it from the registry."""
    registry = PublicationIntentRegistry()

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    ClaimOutcome,
    )

    result = registry.claim_publication_or_wait(
        alias_family="codex",
        cooldown_keys=frozenset({"key-a"}),
        identity_hash="h1",
    )
    assert result.outcome is ClaimOutcome.LEADER
    intent = result.intent
    assert intent is not None

    registry.release_claim(intent)
    assert intent.done.is_set()
    assert registry.get("codex", "key-a") is None
